// GPT-2 124M inference benchmark using dalotia + Boost.Multi
// Single source for both CPU and GPU:
//   CPU (g++/clang++): multi::blas::gemm dispatches to cblas
//   GPU (nvc++ -cuda): multi::blas::gemm dispatches to cuBLAS via thrust
//   pointers
//                      Element-wise ops run on host over managed memory.
//
// GPT-2 uses Conv1D-style weights (transposed): matmul is x @ W, not W @ x.
// Weight shapes: c_attn.weight [768, 2304], c_fc.weight [768, 3072], etc.

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

// For GPU builds, include cuBLAS adaptor BEFORE blas.hpp so that
// default_context_of for thrust pointers is visible at gemm instantiation.
#ifdef DALOTIA_E_WITH_CUBLAS
#include <boost/multi/adaptors/blas.hpp>
#include <boost/multi/adaptors/cuda/cublas.hpp>
#include <boost/multi/array.hpp>
#include <cuda_runtime.h>
#include <thrust/system/cuda/pointer.h>
#else
#include <boost/multi/adaptors/blas.hpp>
#include <boost/multi/array.hpp>
#endif

namespace multi = boost::multi;

// ── Pointer-type abstraction ────────────────────────────────────────────
// On GPU: thrust::cuda::pointer<float> triggers cuBLAS dispatch.
// On CPU: plain float* triggers cblas dispatch.
#ifdef DALOTIA_E_WITH_CUBLAS
using fptr = thrust::cuda::pointer<float>;
using const_fptr = thrust::cuda::pointer<float const>;
inline fptr make_fptr(float *p) { return fptr(p); }
inline const_fptr make_cfptr(const float *p) { return const_fptr(p); }

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " : "    \
                << cudaGetErrorString(err) << std::endl;                       \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)
#else
using fptr = float *;
using const_fptr = float const *;
inline fptr make_fptr(float *p) { return p; }
inline const_fptr make_cfptr(const float *p) { return p; }
#endif

using mat_ref = multi::array_ref<float, 2, fptr>;
using const_mat_ref = multi::array_ref<float const, 2, const_fptr>;

// ── Memory management ───────────────────────────────────────────────────
// GPU builds support two allocation modes:
//   managed: cudaMallocManaged — host can access data directly (simpler, some
//   overhead) device:  cudaMalloc — device-only, element-wise ops use host
//   staging buffers
// CPU builds use standard new[].
// The mode is selected at runtime via the global `use_device_memory` flag.

#ifdef DALOTIA_E_WITH_CUBLAS
static bool use_device_memory = false; // default: managed
#endif

struct Buffer {
  float *ptr = nullptr;
  size_t count = 0;
#ifdef DALOTIA_E_WITH_CUBLAS
  bool is_device = false; // true = cudaMalloc, false = managed or host
#endif

  Buffer() = default;
  explicit Buffer(size_t n) : count(n) {
#ifdef DALOTIA_E_WITH_CUBLAS
    is_device = use_device_memory;
    if (is_device)
      CHECK_CUDA(cudaMalloc(&ptr, n * sizeof(float)));
    else
      CHECK_CUDA(cudaMallocManaged(&ptr, n * sizeof(float)));
#else
    ptr = new float[n]();
#endif
  }
  ~Buffer() {
#ifdef DALOTIA_E_WITH_CUBLAS
    if (ptr)
      cudaFree(ptr);
#else
    delete[] ptr;
#endif
  }
  Buffer(Buffer &&o) noexcept : ptr(o.ptr), count(o.count) {
#ifdef DALOTIA_E_WITH_CUBLAS
    is_device = o.is_device;
#endif
    o.ptr = nullptr;
  }
  Buffer &operator=(Buffer &&o) noexcept {
    if (this != &o) {
      this->~Buffer();
      ptr = o.ptr;
      count = o.count;
#ifdef DALOTIA_E_WITH_CUBLAS
      is_device = o.is_device;
#endif
      o.ptr = nullptr;
    }
    return *this;
  }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;

  float *data() const { return ptr; }
  size_t size() const { return count; }

  // For device-only buffers: download to host, run a function, upload back.
  // For managed buffers: sync, run directly, sync.
  template <class Fn> void on_host(Fn &&fn) {
#ifdef DALOTIA_E_WITH_CUBLAS
    if (is_device) {
      std::vector<float> host(count);
      CHECK_CUDA(cudaMemcpy(host.data(), ptr, count * sizeof(float),
                            cudaMemcpyDeviceToHost));
      fn(host.data());
      CHECK_CUDA(cudaMemcpy(ptr, host.data(), count * sizeof(float),
                            cudaMemcpyHostToDevice));
    } else {
      CHECK_CUDA(cudaDeviceSynchronize());
      fn(ptr);
    }
#else
    fn(ptr);
#endif
  }

  // Read-only host access (no upload).
  template <class Fn> void on_host_ro(Fn &&fn) const {
#ifdef DALOTIA_E_WITH_CUBLAS
    if (is_device) {
      std::vector<float> host(count);
      CHECK_CUDA(cudaMemcpy(host.data(), ptr, count * sizeof(float),
                            cudaMemcpyDeviceToHost));
      fn(host.data());
    } else {
      CHECK_CUDA(cudaDeviceSynchronize());
      fn(ptr);
    }
#else
    fn(ptr);
#endif
  }
};

// Upload dalotia-loaded data into a Buffer.
Buffer to_buffer(const dalotia::vector<float> &src) {
  Buffer buf(src.size());
#ifdef DALOTIA_E_WITH_CUBLAS
  CHECK_CUDA(cudaMemcpy(buf.data(), src.data(), src.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaDeviceSynchronize());
#else
  std::memcpy(buf.data(), src.data(), src.size() * sizeof(float));
#endif
  return buf;
}

// ── GPT-2 124M hyperparameters ──────────────────────────────────────────
constexpr int N_LAYER = 12;
constexpr int N_HEAD = 12;
constexpr int N_EMBD = 768;
constexpr int N_CTX = 1024;
constexpr int VOCAB_SIZE = 50257;
constexpr int HEAD_DIM = N_EMBD / N_HEAD; // 64
constexpr int FFN_DIM = 4 * N_EMBD;       // 3072

// ── Element-wise operations ─────────────────────────────────────────────
// These run on the host. On GPU builds with managed memory, the host can
// access the data directly (with appropriate synchronization).

void layer_norm(float *x, int seq_len, int dim, const float *weight,
                const float *bias, float eps = 1e-5f) {
  for (int s = 0; s < seq_len; ++s) {
    float *row = x + s * dim;
    float mean = 0.0f;
    for (int i = 0; i < dim; ++i)
      mean += row[i];
    mean /= dim;
    float var = 0.0f;
    for (int i = 0; i < dim; ++i) {
      float d = row[i] - mean;
      var += d * d;
    }
    var /= dim;
    float inv_std = 1.0f / std::sqrt(var + eps);
    for (int i = 0; i < dim; ++i)
      row[i] = (row[i] - mean) * inv_std * weight[i] + bias[i];
  }
}

void gelu_inplace(float *x, int n) {
  constexpr float sqrt_2_over_pi = 0.7978845608028654f;
  for (int i = 0; i < n; ++i) {
    float v = x[i];
    x[i] = 0.5f * v *
           (1.0f + std::tanh(sqrt_2_over_pi * (v + 0.044715f * v * v * v)));
  }
}

void softmax_rows_inplace(float *data, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    float *row = data + r * cols;
    float max_val = *std::max_element(row, row + cols);
    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) {
      row[c] = std::exp(row[c] - max_val);
      sum += row[c];
    }
    float inv_sum = 1.0f / sum;
    for (int c = 0; c < cols; ++c)
      row[c] *= inv_sum;
  }
}

void apply_causal_mask(float *attn, int n_head, int seq_len) {
  for (int h = 0; h < n_head; ++h)
    for (int i = 0; i < seq_len; ++i)
      for (int j = i + 1; j < seq_len; ++j)
        attn[h * seq_len * seq_len + i * seq_len + j] = -1e9f;
}

void add_bias(float *x, const float *bias, int rows, int cols) {
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      x[i * cols + j] += bias[j];
}

// ── Model weights ───────────────────────────────────────────────────────

struct TransformerBlock {
  Buffer ln_1_weight, ln_1_bias;
  Buffer c_attn_weight, c_attn_bias;
  Buffer c_proj_weight, c_proj_bias;
  Buffer ln_2_weight, ln_2_bias;
  Buffer c_fc_weight, c_fc_bias;
  Buffer c_proj_mlp_weight, c_proj_mlp_bias;
};

struct GPT2Model {
  Buffer wte, wpe;
  std::vector<TransformerBlock> blocks;
  Buffer ln_f_weight, ln_f_bias;
};

GPT2Model load_model(const std::string &filename) {
  auto file =
      std::unique_ptr<dalotia::TensorFile>(dalotia::make_tensor_file(filename));
  GPT2Model model;

  auto load = [&](const std::string &name) {
    return to_buffer(
        file->load_tensor_dense<float>(name, dalotia_float_32).second);
  };

  model.wte = load("wte.weight");
  model.wpe = load("wpe.weight");

  model.blocks.resize(N_LAYER);
  for (int i = 0; i < N_LAYER; ++i) {
    std::string p = "h." + std::to_string(i) + ".";
    auto &b = model.blocks[i];
    b.ln_1_weight = load(p + "ln_1.weight");
    b.ln_1_bias = load(p + "ln_1.bias");
    b.c_attn_weight = load(p + "attn.c_attn.weight");
    b.c_attn_bias = load(p + "attn.c_attn.bias");
    b.c_proj_weight = load(p + "attn.c_proj.weight");
    b.c_proj_bias = load(p + "attn.c_proj.bias");
    b.ln_2_weight = load(p + "ln_2.weight");
    b.ln_2_bias = load(p + "ln_2.bias");
    b.c_fc_weight = load(p + "mlp.c_fc.weight");
    b.c_fc_bias = load(p + "mlp.c_fc.bias");
    b.c_proj_mlp_weight = load(p + "mlp.c_proj.weight");
    b.c_proj_mlp_bias = load(p + "mlp.c_proj.bias");
  }
  model.ln_f_weight = load("ln_f.weight");
  model.ln_f_bias = load("ln_f.bias");
  return model;
}

// ── Forward pass ────────────────────────────────────────────────────────
// The ONLY difference between CPU and GPU is the pointer type used for
// array_ref: float* dispatches to cblas, thrust::cuda::pointer dispatches
// to cuBLAS. The element-wise ops run on the host in both cases.

inline void sync_device() {
#ifdef DALOTIA_E_WITH_CUBLAS
  cudaDeviceSynchronize();
#endif
}

// Helper: apply element-wise ops that need host access to multiple buffers.
// With device memory, this round-trips through host staging.
// With managed memory, this just syncs and operates in-place.
#ifdef DALOTIA_E_WITH_CUBLAS
void copy_buf(Buffer &dst, const Buffer &src) {
  CHECK_CUDA(cudaMemcpy(dst.data(), src.data(), src.size() * sizeof(float),
                        cudaMemcpyDeviceToDevice));
}
#else
void copy_buf(Buffer &dst, const Buffer &src) {
  std::memcpy(dst.data(), src.data(), src.size() * sizeof(float));
}
#endif

std::vector<float> forward(const GPT2Model &model,
                           const std::vector<int> &token_ids) {
  const int seq_len = static_cast<int>(token_ids.size());
  assert(seq_len > 0 && seq_len <= N_CTX);

  Buffer x(seq_len * N_EMBD);
  Buffer ln_out(seq_len * N_EMBD);
  Buffer qkv(seq_len * 3 * N_EMBD);
  Buffer Q(N_HEAD * seq_len * HEAD_DIM), K(N_HEAD * seq_len * HEAD_DIM);
  Buffer V(N_HEAD * seq_len * HEAD_DIM);
  Buffer attn_scores(N_HEAD * seq_len * seq_len);
  Buffer attn_out(seq_len * N_EMBD);
  Buffer proj_out(seq_len * N_EMBD);
  Buffer ffn_hidden(seq_len * FFN_DIM);
  Buffer ffn_out(seq_len * N_EMBD);
  Buffer attn_concat(seq_len * N_EMBD);

  sync_device();

  // x = wte[token_ids] + wpe[0..seq_len-1]
  {
    // Need host access to model.wte and model.wpe (read) and x (write)
    std::vector<float> h_x(seq_len * N_EMBD);
    std::vector<float> h_wte, h_wpe;

    model.wte.on_host_ro([&](const float *wte) {
      model.wpe.on_host_ro([&](const float *wpe) {
        for (int s = 0; s < seq_len; ++s) {
          const float *tok_emb = wte + token_ids[s] * N_EMBD;
          const float *pos_emb = wpe + s * N_EMBD;
          for (int i = 0; i < N_EMBD; ++i)
            h_x[s * N_EMBD + i] = tok_emb[i] + pos_emb[i];
        }
      });
    });
#ifdef DALOTIA_E_WITH_CUBLAS
    CHECK_CUDA(cudaMemcpy(x.data(), h_x.data(), h_x.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
#else
    std::memcpy(x.data(), h_x.data(), h_x.size() * sizeof(float));
#endif
  }

  for (int layer = 0; layer < N_LAYER; ++layer) {
    const auto &blk = model.blocks[layer];

    // Pre-attention LayerNorm (host-side)
    copy_buf(ln_out, x);
    ln_out.on_host([&](float *p) {
      blk.ln_1_weight.on_host_ro([&](const float *w) {
        blk.ln_1_bias.on_host_ro(
            [&](const float *b) { layer_norm(p, seq_len, N_EMBD, w, b); });
      });
    });

    // QKV projection — gemm dispatches to cblas or cuBLAS
    sync_device();
    {
      auto A = const_mat_ref(make_cfptr(ln_out.data()), {seq_len, N_EMBD});
      auto B = const_mat_ref(make_cfptr(blk.c_attn_weight.data()),
                             {N_EMBD, 3 * N_EMBD});
      auto C = mat_ref(make_fptr(qkv.data()), {seq_len, 3 * N_EMBD});
      multi::blas::gemm(1.0f, A, B, 0.0f, C);
    }
    sync_device();

    // Bias + split QKV (host-side)
    qkv.on_host([&](float *qkv_p) {
      blk.c_attn_bias.on_host_ro([&](const float *bias_p) {
        add_bias(qkv_p, bias_p, seq_len, 3 * N_EMBD);
      });
      // Split Q, K, V: [seq_len, 3*N_EMBD] -> Q,K,V each [N_HEAD, seq_len,
      // HEAD_DIM] We need Q, K, V host data to upload
      std::vector<float> hQ(N_HEAD * seq_len * HEAD_DIM), hK(hQ.size()),
          hV(hQ.size());
      for (int s = 0; s < seq_len; ++s) {
        for (int h = 0; h < N_HEAD; ++h) {
          for (int d = 0; d < HEAD_DIM; ++d) {
            int src_base = s * 3 * N_EMBD + h * HEAD_DIM + d;
            int dst_idx = h * seq_len * HEAD_DIM + s * HEAD_DIM + d;
            hQ[dst_idx] = qkv_p[src_base];
            hK[dst_idx] = qkv_p[src_base + N_EMBD];
            hV[dst_idx] = qkv_p[src_base + 2 * N_EMBD];
          }
        }
      }
#ifdef DALOTIA_E_WITH_CUBLAS
      CHECK_CUDA(cudaMemcpy(Q.data(), hQ.data(), hQ.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(K.data(), hK.data(), hK.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(V.data(), hV.data(), hV.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
#else
      std::memcpy(Q.data(), hQ.data(), hQ.size() * sizeof(float));
      std::memcpy(K.data(), hK.data(), hK.size() * sizeof(float));
      std::memcpy(V.data(), hV.data(), hV.size() * sizeof(float));
#endif
    });

    // Attention scores: Q @ K^T / sqrt(HEAD_DIM)
    float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    sync_device();
    for (int h = 0; h < N_HEAD; ++h) {
      auto Qh = const_mat_ref(make_cfptr(Q.data() + h * seq_len * HEAD_DIM),
                              {seq_len, HEAD_DIM});
      auto Kh = const_mat_ref(make_cfptr(K.data() + h * seq_len * HEAD_DIM),
                              {seq_len, HEAD_DIM});
      auto Sh = mat_ref(make_fptr(attn_scores.data() + h * seq_len * seq_len),
                        {seq_len, seq_len});
      multi::blas::gemm(scale, Qh, Kh.transposed(), 0.0f, Sh);
    }
    sync_device();

    // Causal mask + softmax (host-side)
    attn_scores.on_host([&](float *p) {
      apply_causal_mask(p, N_HEAD, seq_len);
      for (int h = 0; h < N_HEAD; ++h)
        softmax_rows_inplace(p + h * seq_len * seq_len, seq_len, seq_len);
    });

    // Attention output: scores @ V
    sync_device();
    for (int h = 0; h < N_HEAD; ++h) {
      auto Sh =
          const_mat_ref(make_cfptr(attn_scores.data() + h * seq_len * seq_len),
                        {seq_len, seq_len});
      auto Vh = const_mat_ref(make_cfptr(V.data() + h * seq_len * HEAD_DIM),
                              {seq_len, HEAD_DIM});
      auto Oh = mat_ref(make_fptr(attn_out.data() + h * seq_len * HEAD_DIM),
                        {seq_len, HEAD_DIM});
      multi::blas::gemm(1.0f, Sh, Vh, 0.0f, Oh);
    }
    sync_device();

    // Reshape [N_HEAD, seq_len, HEAD_DIM] -> [seq_len, N_EMBD] (host-side)
    attn_out.on_host_ro([&](const float *ao_p) {
      std::vector<float> h_cat(seq_len * N_EMBD);
      for (int s = 0; s < seq_len; ++s)
        for (int h = 0; h < N_HEAD; ++h)
          for (int d = 0; d < HEAD_DIM; ++d)
            h_cat[s * N_EMBD + h * HEAD_DIM + d] =
                ao_p[h * seq_len * HEAD_DIM + s * HEAD_DIM + d];
#ifdef DALOTIA_E_WITH_CUBLAS
      CHECK_CUDA(cudaMemcpy(attn_concat.data(), h_cat.data(),
                            h_cat.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
#else
      std::memcpy(attn_concat.data(), h_cat.data(),
                  h_cat.size() * sizeof(float));
#endif
    });

    // Output projection + residual
    sync_device();
    {
      auto A = const_mat_ref(make_cfptr(attn_concat.data()), {seq_len, N_EMBD});
      auto B =
          const_mat_ref(make_cfptr(blk.c_proj_weight.data()), {N_EMBD, N_EMBD});
      auto C = mat_ref(make_fptr(proj_out.data()), {seq_len, N_EMBD});
      multi::blas::gemm(1.0f, A, B, 0.0f, C);
    }
    sync_device();
    proj_out.on_host([&](float *proj_p) {
      blk.c_proj_bias.on_host_ro([&](const float *bias_p) {
        add_bias(proj_p, bias_p, seq_len, N_EMBD);
      });
    });
    // x += proj_out  (both are device/managed buffers — use on_host for device)
    x.on_host([&](float *x_p) {
      proj_out.on_host_ro([&](const float *p_p) {
        for (int i = 0; i < seq_len * N_EMBD; ++i)
          x_p[i] += p_p[i];
      });
    });

    // Pre-FFN LayerNorm
    copy_buf(ln_out, x);
    ln_out.on_host([&](float *p) {
      blk.ln_2_weight.on_host_ro([&](const float *w) {
        blk.ln_2_bias.on_host_ro(
            [&](const float *b) { layer_norm(p, seq_len, N_EMBD, w, b); });
      });
    });

    // FFN up + GELU
    sync_device();
    {
      auto A = const_mat_ref(make_cfptr(ln_out.data()), {seq_len, N_EMBD});
      auto B =
          const_mat_ref(make_cfptr(blk.c_fc_weight.data()), {N_EMBD, FFN_DIM});
      auto C = mat_ref(make_fptr(ffn_hidden.data()), {seq_len, FFN_DIM});
      multi::blas::gemm(1.0f, A, B, 0.0f, C);
    }
    sync_device();
    ffn_hidden.on_host([&](float *p) {
      blk.c_fc_bias.on_host_ro(
          [&](const float *bias_p) { add_bias(p, bias_p, seq_len, FFN_DIM); });
      gelu_inplace(p, seq_len * FFN_DIM);
    });

    // FFN down + residual
    sync_device();
    {
      auto A = const_mat_ref(make_cfptr(ffn_hidden.data()), {seq_len, FFN_DIM});
      auto B = const_mat_ref(make_cfptr(blk.c_proj_mlp_weight.data()),
                             {FFN_DIM, N_EMBD});
      auto C = mat_ref(make_fptr(ffn_out.data()), {seq_len, N_EMBD});
      multi::blas::gemm(1.0f, A, B, 0.0f, C);
    }
    sync_device();
    ffn_out.on_host([&](float *fo_p) {
      blk.c_proj_mlp_bias.on_host_ro([&](const float *bias_p) {
        add_bias(fo_p, bias_p, seq_len, N_EMBD);
      });
    });
    x.on_host([&](float *x_p) {
      ffn_out.on_host_ro([&](const float *fo_p) {
        for (int i = 0; i < seq_len * N_EMBD; ++i)
          x_p[i] += fo_p[i];
      });
    });
  }

  // Final LayerNorm
  x.on_host([&](float *p) {
    model.ln_f_weight.on_host_ro([&](const float *w) {
      model.ln_f_bias.on_host_ro(
          [&](const float *b) { layer_norm(p, seq_len, N_EMBD, w, b); });
    });
  });

  // Logits: x @ wte^T
  Buffer logits_buf(seq_len * VOCAB_SIZE);
  sync_device();
  {
    auto A = const_mat_ref(make_cfptr(x.data()), {seq_len, N_EMBD});
    auto B = const_mat_ref(make_cfptr(model.wte.data()), {VOCAB_SIZE, N_EMBD});
    auto C = mat_ref(make_fptr(logits_buf.data()), {seq_len, VOCAB_SIZE});
    multi::blas::gemm(1.0f, A, B.transposed(), 0.0f, C);
  }
  sync_device();

  std::vector<float> logits(seq_len * VOCAB_SIZE);
  logits_buf.on_host_ro([&](const float *p) {
    std::memcpy(logits.data(), p, logits.size() * sizeof(float));
  });
  return logits;
}


int argmax(const float *data, int n) {
  return static_cast<int>(
      std::distance(data, std::max_element(data, data + n)));
}

// ── Main ────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
  std::string model_path = "./model.safetensors";
  int num_generate = 20;

  // Usage: gpt2 [model.safetensors] [num_tokens] [--device|--managed]
  if (argc > 1)
    model_path = argv[1];
  if (argc > 2) {
    std::string a2 = argv[2];
    if (a2 != "--device" && a2 != "--managed")
      num_generate = std::stoi(a2);
  }
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
#ifdef DALOTIA_E_WITH_CUBLAS
    if (arg == "--device")
      use_device_memory = true;
    if (arg == "--managed")
      use_device_memory = false;
#endif
  }

// Prompt tokens for "The meaning of life is"
// (GPT-2 BPE token IDs — hardcoded to avoid tokenizer dependency)
  std::vector<int> prompt_tokens = {464, 3616, 286, 1204, 318};

#ifdef DALOTIA_E_WITH_CUBLAS
  std::cout << "Loading GPT-2 124M (GPU/cuBLAS, "
            << (use_device_memory ? "device" : "managed") << " memory) from "
            << model_path << " ..." << std::endl;
#else
  std::cout << "Loading GPT-2 124M (CPU/cblas) from " << model_path << " ..."
            << std::endl;
#endif
  auto t0 = std::chrono::high_resolution_clock::now();
  GPT2Model model = load_model(model_path);
  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << "Model loaded in "
            << std::chrono::duration<double>(t1 - t0).count() << "s"
            << std::endl;

  std::cout << "Generating " << num_generate << " tokens..." << std::endl;
  std::vector<int> tokens = prompt_tokens;
  auto t2 = std::chrono::high_resolution_clock::now();
  for (int step = 0; step < num_generate; ++step) {
    auto logits = forward(model, tokens);
    tokens.push_back(
        argmax(logits.data() + (tokens.size() - 1) * VOCAB_SIZE, VOCAB_SIZE));
  }
  auto t3 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> dt = t3 - t2;
  std::cout << "Inference time: " << dt.count() << "s (" << num_generate
            << " tokens, " << dt.count() / num_generate << "s/token)"
            << std::endl;

  std::cout << "Generated token IDs: [";
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (i)
      std::cout << ", ";
    std::cout << tokens[i];
  }
  std::cout << "]" << std::endl;

  {
    auto logits = forward(model, prompt_tokens);
    for (int i = 0; i < int(prompt_tokens.size()) * VOCAB_SIZE; ++i)
      if (!std::isfinite(logits[i])) {
        std::cerr << "FAIL: non-finite logits!" << std::endl;
        return 1;
      }
    std::cout << "First predicted token after prompt: "
              << argmax(logits.data() + (prompt_tokens.size() - 1) * VOCAB_SIZE,
                        VOCAB_SIZE)
              << std::endl;
  }
  std::cout << "success!" << std::endl;
  return 0;
}
