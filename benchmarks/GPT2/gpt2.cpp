// GPT-2 124M inference benchmark using dalotia + Boost.Multi
// Single source for both CPU and GPU:
//   CPU (g++/clang++): multi::blas::gemm dispatches to cblas, host loops
//   GPU (nvc++ -cuda): multi::blas::gemm dispatches to cuBLAS, CUDA kernels
//                      All ops on a single CUDA stream (fully async)
//
// GPU memory via dalotia CUDA PMR allocators:
//   Model weights: cuda_managed_resource (persistent, host-accessible for loading)
//   Scratch buffers: cuda_async_memory_resource (stream-ordered)
//
// GPT-2 uses Conv1D-style weights (transposed): matmul is x @ W, not W @ x.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "dalotia.hpp"
#include "dalotia_safetensors_file.hpp"

#ifdef DALOTIA_E_WITH_CUBLAS
#include <boost/multi/adaptors/cuda/cublas.hpp>
#include <boost/multi/adaptors/blas.hpp>
#include <boost/multi/array.hpp>
#include <cuda_runtime.h>
#include <thrust/system/cuda/pointer.h>
#else
#include <boost/multi/array.hpp>
#include <boost/multi/adaptors/blas.hpp>
#endif

namespace multi = boost::multi;
using multi::blas::operators::operator*;  // C = A * B  (gemm)

// ── Pointer-type abstraction ────────────────────────────────────────────
#ifdef DALOTIA_E_WITH_CUBLAS
using fptr       = thrust::cuda::pointer<float>;
using const_fptr = thrust::cuda::pointer<float const>;
inline fptr       make_fptr(float* p)       { return fptr(p); }
inline const_fptr make_cfptr(const float* p) { return const_fptr(p); }

#define CHECK_CUDA(call) do {                                              \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                  << " : " << cudaGetErrorString(err) << std::endl;        \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)
#else
using fptr       = float*;
using const_fptr = float const*;
inline fptr       make_fptr(float* p)       { return p; }
inline const_fptr make_cfptr(const float* p) { return p; }
#endif

using mat_ref       = multi::array_ref<float, 2, fptr>;
using const_mat_ref = multi::array_ref<float const, 2, const_fptr>;

// ── Memory resources ────────────────────────────────────────────────────
// GPU: no managed memory. Weights loaded directly to device.
//      Scratch buffers use stream-ordered async allocation (cudaMallocAsync).
// CPU: standard new/delete for everything.
#ifdef DALOTIA_E_WITH_CUBLAS
static cudaStream_t inference_stream = 0;

// Weights: dalotia loads directly into device pointers (GDS or host-staging).
std::pmr::memory_resource* device_resource() {
    return dalotia::cuda_device_resource();
}

// Scratch: stream-ordered async allocation (cudaMallocAsync on stream 0).
static dalotia::cuda_async_memory_resource* scratch_res = nullptr;

std::pmr::memory_resource* scratch_resource() {
    if (!scratch_res) scratch_res = new dalotia::cuda_async_memory_resource(/*stream=*/0);
    return scratch_res;
}
#else
std::pmr::memory_resource* scratch_resource() { return std::pmr::new_delete_resource(); }
#endif

// Scratch buffer: raw PMR allocation, no zero-initialization.
// On GPU this uses cudaMallocAsync; on CPU it uses new[].
struct ScratchBuf {
    float* ptr = nullptr;
    size_t count = 0;
    std::pmr::memory_resource* mr = nullptr;

    ScratchBuf() = default;
    ScratchBuf(size_t n, std::pmr::memory_resource* r) : count(n), mr(r) {
        ptr = static_cast<float*>(mr->allocate(n * sizeof(float), alignof(float)));
    }
    ~ScratchBuf() { if (ptr) mr->deallocate(ptr, count * sizeof(float), alignof(float)); }
    ScratchBuf(ScratchBuf&& o) noexcept : ptr(o.ptr), count(o.count), mr(o.mr) { o.ptr = nullptr; }
    ScratchBuf& operator=(ScratchBuf&& o) noexcept {
        if (this != &o) { this->~ScratchBuf(); ptr = o.ptr; count = o.count; mr = o.mr; o.ptr = nullptr; }
        return *this;
    }
    ScratchBuf(const ScratchBuf&) = delete;
    ScratchBuf& operator=(const ScratchBuf&) = delete;
    float* data() const { return ptr; }
};

ScratchBuf make_buffer(size_t n, std::pmr::memory_resource* mr) {
    return ScratchBuf(n, mr);
}

// ── GPT-2 124M hyperparameters ──────────────────────────────────────────
constexpr int N_LAYER = 12;
constexpr int N_HEAD  = 12;
constexpr int N_EMBD  = 768;
constexpr int VOCAB_SIZE = 50257;
constexpr int HEAD_DIM = N_EMBD / N_HEAD;
constexpr int FFN_DIM  = 4 * N_EMBD;

// ── Element-wise operations ─────────────────────────────────────────────
// GPU: __global__ CUDA kernels launched on inference_stream
// CPU: plain host loops

#ifdef DALOTIA_E_WITH_CUBLAS
constexpr int BLOCK = 256;
inline int grid(int n) { return (n + BLOCK - 1) / BLOCK; }

// Block-wide sum reduction using shared memory.
__device__ float block_reduce_sum(float val) {
    __shared__ float smem[32];
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;
    int nwarps = blockDim.x / warpSize;

    // Intra-warp reduction
    for (int off = warpSize/2; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);

    // Write each warp's result to shared memory
    if (lane == 0) smem[wid] = val;
    __syncthreads();

    // First warp reads all partial sums and reduces
    val = (threadIdx.x < nwarps) ? smem[threadIdx.x] : 0.0f;
    if (wid == 0) {
        for (int off = warpSize/2; off > 0; off >>= 1)
            val += __shfl_down_sync(0xffffffff, val, off);
    }

    // Broadcast result from thread 0 via shared memory
    if (threadIdx.x == 0) smem[0] = val;
    __syncthreads();
    return smem[0];
}

// Out-of-place LayerNorm: reads from src, writes to dst.
// Eliminates the need for a separate copy before in-place LayerNorm.
__global__ void layer_norm_kernel(const float* src, float* dst, int seq_len, int dim,
                                  const float* weight, const float* bias, float eps) {
    int s = blockIdx.x; if (s >= seq_len) return;
    const float* in  = src + s * dim;
    float*       out = dst + s * dim;

    float val = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) val += in[i];
    float mean = block_reduce_sum(val) / dim;

    val = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) { float d = in[i]-mean; val += d*d; }
    float inv_std = rsqrtf(block_reduce_sum(val) / dim + eps);

    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        out[i] = (in[i] - mean) * inv_std * weight[i] + bias[i];
}

__global__ void gelu_kernel(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = x[idx];
    x[idx] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v*v*v)));
}

__device__ float block_reduce_max(float val) {
    __shared__ float smem[32];
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;
    int nwarps = blockDim.x / warpSize;

    for (int off = warpSize/2; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, off));
    if (lane == 0) smem[wid] = val;
    __syncthreads();

    val = (threadIdx.x < nwarps) ? smem[threadIdx.x] : -1e30f;
    if (wid == 0) {
        for (int off = warpSize/2; off > 0; off >>= 1)
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, off));
    }
    if (threadIdx.x == 0) smem[0] = val;
    __syncthreads();
    return smem[0];
}

__global__ void softmax_rows_kernel(float* data, int rows, int cols) {
    int r = blockIdx.x; if (r >= rows) return;
    float* row = data + r * cols;

    float lmax = -1e30f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) lmax = fmaxf(lmax, row[c]);
    float mx = block_reduce_max(lmax);

    float lsum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) { float v = expf(row[c]-mx); row[c] = v; lsum += v; }
    float inv = 1.0f / block_reduce_sum(lsum);

    for (int c = threadIdx.x; c < cols; c += blockDim.x) row[c] *= inv;
}

__global__ void causal_mask_kernel(float* attn, int n_head, int seq_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_head * seq_len * seq_len) return;
    int rem = idx % (seq_len * seq_len);
    if (rem % seq_len > rem / seq_len) attn[idx] = -1e9f;
}

__global__ void embed_kernel(float* dst, const float* wte, const float* wpe,
                             const int* tokens, int seq_len, int embd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * embd) return;
    int s = idx / embd, i = idx % embd;
    dst[idx] = wte[tokens[s] * embd + i] + wpe[s * embd + i];
}

__global__ void split_qkv_kernel(const float* qkv, float* Q, float* K, float* V,
                                 int seq_len, int n_head, int head_dim, int n_embd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_head * seq_len * head_dim) return;
    int h = idx / (seq_len * head_dim), rem = idx % (seq_len * head_dim);
    int s = rem / head_dim, d = rem % head_dim;
    int src = s * 3 * n_embd + h * head_dim + d;
    Q[idx] = qkv[src]; K[idx] = qkv[src + n_embd]; V[idx] = qkv[src + 2 * n_embd];
}

__global__ void concat_heads_kernel(float* dst, const float* src,
                                    int seq_len, int n_head, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_embd = n_head * head_dim;
    if (idx >= seq_len * n_embd) return;
    int s = idx / n_embd, e = idx % n_embd;
    dst[idx] = src[(e / head_dim) * seq_len * head_dim + s * head_dim + e % head_dim];
}

// Dispatch wrappers — launch on inference_stream
// Out-of-place LayerNorm: dst = LN(src)
void layer_norm(const float* src, float* dst, int seq_len, int dim, const float* w, const float* b) {
    layer_norm_kernel<<<seq_len, BLOCK, 0, inference_stream>>>(src, dst, seq_len, dim, w, b, 1e-5f);
}
void gelu_inplace(float* x, int n) {
    gelu_kernel<<<grid(n), BLOCK, 0, inference_stream>>>(x, n);
}
void softmax_rows_inplace(float* data, int rows, int cols) {
    for (int r = 0; r < rows; ++r)
        softmax_rows_kernel<<<1, BLOCK, 0, inference_stream>>>(data + r*cols, 1, cols);
}
void apply_causal_mask(float* attn, int n_head, int seq_len) {
    causal_mask_kernel<<<grid(n_head*seq_len*seq_len), BLOCK, 0, inference_stream>>>(attn, n_head, seq_len);
}

#else  // CPU path

// Out-of-place LayerNorm: dst = LN(src)
void layer_norm(const float* src, float* dst, int seq_len, int dim, const float* weight, const float* bias) {
    for (int s = 0; s < seq_len; ++s) {
        const float* in  = src + s * dim;
        float*       out = dst + s * dim;
        float mean = 0.0f;
        for (int i = 0; i < dim; ++i) mean += in[i];
        mean /= dim;
        float var = 0.0f;
        for (int i = 0; i < dim; ++i) { float d = in[i] - mean; var += d * d; }
        var /= dim;
        float inv_std = 1.0f / std::sqrt(var + 1e-5f);
        for (int i = 0; i < dim; ++i)
            out[i] = (in[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

void gelu_inplace(float* x, int n) {
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
    for (int i = 0; i < n; ++i) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + std::tanh(sqrt_2_over_pi * (v + 0.044715f * v*v*v)));
    }
}

void softmax_rows_inplace(float* data, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float* row = data + r * cols;
        float max_val = *std::max_element(row, row + cols);
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) { row[c] = std::exp(row[c] - max_val); sum += row[c]; }
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; ++c) row[c] *= inv_sum;
    }
}

void apply_causal_mask(float* attn, int n_head, int seq_len) {
    for (int h = 0; h < n_head; ++h)
        for (int i = 0; i < seq_len; ++i)
            for (int j = i + 1; j < seq_len; ++j)
                attn[h * seq_len * seq_len + i * seq_len + j] = -1e9f;
}

#endif  // DALOTIA_E_WITH_CUBLAS

// ── Shared BLAS ops (dispatch to cblas or cuBLAS via multi::blas) ───────

using vec_ref       = multi::array_ref<float, 1, fptr>;
using const_vec_ref = multi::array_ref<float const, 1, const_fptr>;

// Residual add: dst += src  (via multi::blas::axpy → cblas/cuBLAS)
void add_vecs(float* dst, const float* src, int n) {
    auto s = const_vec_ref(make_cfptr(src), {n});
    auto d = vec_ref(make_fptr(dst), {n});
    multi::blas::axpy(1.0f, s, d);
}

// Bias add: for each row, y[row,:] += bias[:]  (via multi::blas::axpy)
void add_bias(float* x, const float* bias, int rows, int cols) {
    auto b = const_vec_ref(make_cfptr(bias), {cols});
    for (int r = 0; r < rows; ++r) {
        auto row = vec_ref(make_fptr(x + r * cols), {cols});
        multi::blas::axpy(1.0f, b, row);
    }
}

// ── Model weights ───────────────────────────────────────────────────────
// On GPU: device memory. On CPU: standard heap memory.

struct TransformerBlock {
    ScratchBuf ln_1_weight, ln_1_bias;
    ScratchBuf c_attn_weight, c_attn_bias;
    ScratchBuf c_proj_weight, c_proj_bias;
    ScratchBuf ln_2_weight, ln_2_bias;
    ScratchBuf c_fc_weight, c_fc_bias;
    ScratchBuf c_proj_mlp_weight, c_proj_mlp_bias;
};

struct GPT2Model {
    ScratchBuf wte, wpe;
    std::vector<TransformerBlock> blocks;
    ScratchBuf ln_f_weight, ln_f_bias;
};

GPT2Model load_model(const std::string& filename) {
    auto file = std::unique_ptr<dalotia::TensorFile>(
        dalotia::make_tensor_file(filename));
    GPT2Model model;

#ifdef DALOTIA_E_WITH_CUBLAS
    // Allocate device buffer, load directly into it via dalotia.
    // dalotia detects the device pointer and uses GDS or host-staging internally.
    auto* buf_mr = device_resource();
#else
    auto* buf_mr = std::pmr::new_delete_resource();
#endif

    auto load = [&](const std::string& name) -> ScratchBuf {
        auto extents = file->get_tensor_extents(name);
        size_t n = std::accumulate(extents.begin(), extents.end(), size_t{1}, std::multiplies<>());
        ScratchBuf buf(n, buf_mr);
        file->load_tensor_dense(name, dalotia_float_32, dalotia_C_ordering,
                                reinterpret_cast<dalotia_byte*>(buf.data()));
        return buf;
    };

    auto load_transposed = [&](const std::string& name) -> ScratchBuf {
        auto extents = file->get_tensor_extents(name);
        size_t n = std::accumulate(extents.begin(), extents.end(), size_t{1}, std::multiplies<>());
        ScratchBuf buf(n, buf_mr);
        file->load_tensor_dense(name, dalotia_float_32, dalotia_C_ordering,
                                reinterpret_cast<dalotia_byte*>(buf.data()), {1, 0});
        return buf;
    };

    model.wte = load("wte.weight");
    model.wpe = load("wpe.weight");

    model.blocks.resize(N_LAYER);
    for (int i = 0; i < N_LAYER; ++i) {
        std::string p = "h." + std::to_string(i) + ".";
        auto& b = model.blocks[i];
        b.ln_1_weight = load(p+"ln_1.weight"); b.ln_1_bias = load(p+"ln_1.bias");
        b.c_attn_weight = load_transposed(p+"attn.c_attn.weight"); b.c_attn_bias = load(p+"attn.c_attn.bias");
        b.c_proj_weight = load(p+"attn.c_proj.weight"); b.c_proj_bias = load(p+"attn.c_proj.bias");
        b.ln_2_weight = load(p+"ln_2.weight"); b.ln_2_bias = load(p+"ln_2.bias");
        b.c_fc_weight = load(p+"mlp.c_fc.weight"); b.c_fc_bias = load(p+"mlp.c_fc.bias");
        b.c_proj_mlp_weight = load(p+"mlp.c_proj.weight"); b.c_proj_mlp_bias = load(p+"mlp.c_proj.bias");
    }
    model.ln_f_weight = load("ln_f.weight");
    model.ln_f_bias   = load("ln_f.bias");
    return model;
}

// ── Forward pass ────────────────────────────────────────────────────────
// GPU: all ops on default stream 0 — implicitly ordered, no inter-op sync.
//      All buffers are device memory (no managed). Only sync is the final
//      cudaStreamSynchronize before reading logits back to host.
// CPU: sequential host ops.

std::vector<float> forward(const GPT2Model& model,
                           const std::vector<int>& token_ids) {
    const int S = static_cast<int>(token_ids.size());
    assert(S > 0 && S <= 1024);

    auto* smr = scratch_resource();

    auto x           = make_buffer(S * N_EMBD, smr);
    auto ln_out      = make_buffer(S * N_EMBD, smr);
    auto qkv         = make_buffer(S * 3 * N_EMBD, smr);
    auto Q           = make_buffer(N_HEAD * S * HEAD_DIM, smr);
    auto K           = make_buffer(N_HEAD * S * HEAD_DIM, smr);
    auto V           = make_buffer(N_HEAD * S * HEAD_DIM, smr);
    auto attn_scores = make_buffer(N_HEAD * S * S, smr);
    auto attn_out    = make_buffer(S * N_EMBD, smr);
    auto proj_out    = make_buffer(S * N_EMBD, smr);
    auto ffn_hidden  = make_buffer(S * FFN_DIM, smr);
    auto ffn_out     = make_buffer(S * N_EMBD, smr);
    auto attn_concat = make_buffer(S * N_EMBD, smr);

    // ── Embeddings ──────────────────────────────────────────────────
#ifdef DALOTIA_E_WITH_CUBLAS
    // Upload token IDs to device and run embed kernel
    int* d_tok;
    CHECK_CUDA(cudaMallocAsync(&d_tok, S * sizeof(int), inference_stream));
    CHECK_CUDA(cudaMemcpyAsync(d_tok, token_ids.data(), S * sizeof(int),
                               cudaMemcpyHostToDevice, inference_stream));
    embed_kernel<<<grid(S*N_EMBD), BLOCK, 0, inference_stream>>>(
        x.data(), model.wte.data(), model.wpe.data(), d_tok, S, N_EMBD);
    CHECK_CUDA(cudaFreeAsync(d_tok, inference_stream));
#else
    for (int s = 0; s < S; ++s) {
        const float* tok_emb = model.wte.data() + token_ids[s] * N_EMBD;
        const float* pos_emb = model.wpe.data() + s * N_EMBD;
        float* dst = x.data() + s * N_EMBD;
        for (int i = 0; i < N_EMBD; ++i) dst[i] = tok_emb[i] + pos_emb[i];
    }
#endif

    // 2D views over flat buffers
    auto&& X       = mat_ref(make_fptr(x.data()),           {S, N_EMBD});
    auto&& LN      = mat_ref(make_fptr(ln_out.data()),      {S, N_EMBD});
    auto&& QKV     = mat_ref(make_fptr(qkv.data()),         {S, 3*N_EMBD});
    auto&& Proj    = mat_ref(make_fptr(proj_out.data()),     {S, N_EMBD});
    auto&& FFN_H   = mat_ref(make_fptr(ffn_hidden.data()),  {S, FFN_DIM});
    auto&& FFN_O   = mat_ref(make_fptr(ffn_out.data()),     {S, N_EMBD});
    auto&& Concat  = mat_ref(make_fptr(attn_concat.data()), {S, N_EMBD});

    for (int layer = 0; layer < N_LAYER; ++layer) {
        const auto& blk = model.blocks[layer];

        // Weight views (const, reused each layer)
        const auto& W_attn = const_mat_ref(make_cfptr(blk.c_attn_weight.data()), {3*N_EMBD, N_EMBD});
        const auto& W_proj = const_mat_ref(make_cfptr(blk.c_proj_weight.data()), {N_EMBD, N_EMBD});
        const auto& W_fc   = const_mat_ref(make_cfptr(blk.c_fc_weight.data()),   {N_EMBD, FFN_DIM});
        const auto& W_fc_p = const_mat_ref(make_cfptr(blk.c_proj_mlp_weight.data()), {FFN_DIM, N_EMBD});

        // Pre-attention LayerNorm: LN = LayerNorm(X)
        layer_norm(x.data(), ln_out.data(), S, N_EMBD,
                   blk.ln_1_weight.data(), blk.ln_1_bias.data());

        // QKV = LN @ W_attn + bias
        QKV = LN * W_attn.transposed();
        add_bias(qkv.data(), blk.c_attn_bias.data(), S, 3*N_EMBD);

        // Split Q,K,V and transpose to [N_HEAD, S, HEAD_DIM]
#ifdef DALOTIA_E_WITH_CUBLAS
        split_qkv_kernel<<<grid(N_HEAD*S*HEAD_DIM), BLOCK, 0, inference_stream>>>(
            qkv.data(), Q.data(), K.data(), V.data(), S, N_HEAD, HEAD_DIM, N_EMBD);
#else
        for (int s = 0; s < S; ++s)
            for (int h = 0; h < N_HEAD; ++h)
                for (int d = 0; d < HEAD_DIM; ++d) {
                    int src = s * 3 * N_EMBD + h * HEAD_DIM + d;
                    int dst = h * S * HEAD_DIM + s * HEAD_DIM + d;
                    Q.data()[dst] = qkv.data()[src];
                    K.data()[dst] = qkv.data()[src + N_EMBD];
                    V.data()[dst] = qkv.data()[src + 2*N_EMBD];
                }
#endif

        // Per-head attention: scores = scale * Qh @ Kh^T, output = scores @ Vh
        float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
        for (int h = 0; h < N_HEAD; ++h) {
            const auto& Qh = const_mat_ref(make_cfptr(Q.data()+h*S*HEAD_DIM), {S, HEAD_DIM});
            const auto& Kh = const_mat_ref(make_cfptr(K.data()+h*S*HEAD_DIM), {S, HEAD_DIM});
            const auto& Vh = const_mat_ref(make_cfptr(V.data()+h*S*HEAD_DIM), {S, HEAD_DIM});
            auto&& Sh = mat_ref(make_fptr(attn_scores.data()+h*S*S), {S, S});
            auto&& Oh = mat_ref(make_fptr(attn_out.data()+h*S*HEAD_DIM), {S, HEAD_DIM});

            multi::blas::gemm(scale, Qh, Kh.transposed(), 0.0f, Sh);
            apply_causal_mask(attn_scores.data()+h*S*S, 1, S);
            softmax_rows_inplace(attn_scores.data()+h*S*S, S, S);
            Oh = Sh * Vh;
        }

        // Concat heads: [N_HEAD, S, HEAD_DIM] → [S, N_EMBD]
#ifdef DALOTIA_E_WITH_CUBLAS
        concat_heads_kernel<<<grid(S*N_EMBD), BLOCK, 0, inference_stream>>>(
            attn_concat.data(), attn_out.data(), S, N_HEAD, HEAD_DIM);
#else
        for (int s = 0; s < S; ++s)
            for (int h = 0; h < N_HEAD; ++h)
                for (int d = 0; d < HEAD_DIM; ++d)
                    attn_concat.data()[s*N_EMBD + h*HEAD_DIM + d] =
                        attn_out.data()[h*S*HEAD_DIM + s*HEAD_DIM + d];
#endif

        // Output projection into Proj, add bias, then X += Proj
        Proj = Concat * W_proj;
        add_bias(proj_out.data(), blk.c_proj_bias.data(), S, N_EMBD);
        add_vecs(x.data(), proj_out.data(), S * N_EMBD);

        // Pre-FFN LayerNorm: LN = LayerNorm(X)
        layer_norm(x.data(), ln_out.data(), S, N_EMBD,
                   blk.ln_2_weight.data(), blk.ln_2_bias.data());

        // FFN: X += GELU(LN @ W_fc + bias) @ W_fc_proj + bias
        FFN_H = LN * W_fc;
        add_bias(ffn_hidden.data(), blk.c_fc_bias.data(), S, FFN_DIM);
        gelu_inplace(ffn_hidden.data(), S * FFN_DIM);
        FFN_O = FFN_H * W_fc_p;
        add_bias(ffn_out.data(), blk.c_proj_mlp_bias.data(), S, N_EMBD);
        add_vecs(x.data(), ffn_out.data(), S * N_EMBD);
    }

    // Final LayerNorm: LN = LayerNorm(X)
    layer_norm(x.data(), ln_out.data(), S, N_EMBD,
               model.ln_f_weight.data(), model.ln_f_bias.data());

    // Logits = LN @ Wte^T
    auto logits_buf = make_buffer(S * VOCAB_SIZE, smr);
    {
        auto   Wte    = const_mat_ref(make_cfptr(model.wte.data()), {VOCAB_SIZE, N_EMBD});
        auto&& Logits = mat_ref(make_fptr(logits_buf.data()), {S, VOCAB_SIZE});
        Logits = LN * Wte.transposed();
    }

    // Single sync point: copy logits to host
    std::vector<float> logits(S * VOCAB_SIZE);
#ifdef DALOTIA_E_WITH_CUBLAS
    CHECK_CUDA(cudaMemcpyAsync(logits.data(), logits_buf.data(),
                               logits.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, inference_stream));
    CHECK_CUDA(cudaStreamSynchronize(inference_stream));
#else
    std::memcpy(logits.data(), logits_buf.data(), logits.size() * sizeof(float));
#endif
    return logits;
}

// ── Main ────────────────────────────────────────────────────────────────

int argmax(const float* data, int n) {
    return static_cast<int>(std::distance(data, std::max_element(data, data + n)));
}

int main(int argc, char* argv[]) {
    std::string model_path = "./model.safetensors";
    int num_generate = 20;

    if (argc > 1) model_path = argv[1];
    if (argc > 2) num_generate = std::stoi(argv[2]);

    std::vector<int> prompt_tokens = {464, 3616, 286, 1204, 318};

#ifdef DALOTIA_E_WITH_CUBLAS
    CHECK_CUDA(cudaFree(nullptr));  // force CUDA init

    inference_stream = 0;

    // All GPU ops on default stream 0 — implicitly ordered.
    auto& ctx = multi::cuda::cublas::context::get_instance();
    ctx.set_sync(false);

    std::cout << "Loading GPT-2 124M (GPU/cuBLAS) from "
              << model_path << " ..." << std::endl;
#else
    std::cout << "Loading GPT-2 124M (CPU/cblas) from " << model_path << " ..." << std::endl;
#endif

    auto t0 = std::chrono::high_resolution_clock::now();
    const GPT2Model model = load_model(model_path);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Model loaded in " << std::chrono::duration<double>(t1-t0).count() << "s" << std::endl;

    std::cout << "Generating " << num_generate << " tokens..." << std::endl;
    std::vector<int> tokens = prompt_tokens;
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < num_generate; ++step) {
        auto logits = forward(model, tokens);
        tokens.push_back(argmax(logits.data() + (tokens.size()-1) * VOCAB_SIZE, VOCAB_SIZE));
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = t3 - t2;
    std::cout << "Inference time: " << dt.count() << "s ("
              << num_generate << " tokens, " << dt.count()/num_generate << "s/token)" << std::endl;

    std::cout << "Generated token IDs: [";
    for (size_t i = 0; i < tokens.size(); ++i) { if (i) std::cout << ", "; std::cout << tokens[i]; }
    std::cout << "]" << std::endl;

    {
        auto logits = forward(model, prompt_tokens);
        for (int i = 0; i < int(prompt_tokens.size()) * VOCAB_SIZE; ++i)
            if (!std::isfinite(logits[i])) { std::cerr << "FAIL: non-finite logits!" << std::endl; return 1; }
        int first_pred = argmax(logits.data() + (prompt_tokens.size()-1) * VOCAB_SIZE, VOCAB_SIZE);
        std::cout << "First predicted token after prompt: " << first_pred << std::endl;
        if (first_pred != 407) {
            std::cerr << "FAIL: expected token 407, got " << first_pred << std::endl;
            return 1;
        }
    }

#ifdef DALOTIA_E_WITH_CUBLAS
    delete scratch_res; scratch_res = nullptr;
#endif

    std::cout << "success!" << std::endl;
    return 0;
}
