// GPT-2 124M inference benchmark using dalotia + Boost.Multi
// Loads weights from safetensors, runs a single forward pass, reports timings.
//
// Boost.Multi dispatches to cblas or cublas based on the allocator used.
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

#include <boost/multi/array.hpp>
#include <boost/multi/adaptors/blas.hpp>

namespace multi = boost::multi;

using mat_ref = multi::array_ref<float, 2>;
using const_mat_ref = multi::array_ref<float const, 2>;

// ── GPT-2 124M hyperparameters ──────────────────────────────────────────
constexpr int N_LAYER = 12;
constexpr int N_HEAD = 12;
constexpr int N_EMBD = 768;
constexpr int N_CTX = 1024;
constexpr int VOCAB_SIZE = 50257;
constexpr int HEAD_DIM = N_EMBD / N_HEAD;  // 64
constexpr int FFN_DIM = 4 * N_EMBD;        // 3072

// ── Element-wise operations ─────────────────────────────────────────────

// LayerNorm: x = (x - mean) / sqrt(var + eps) * weight + bias
// Operates in-place on each row of x[seq_len, dim].
void layer_norm(float* x, int seq_len, int dim,
                const float* weight, const float* bias, float eps = 1e-5f) {
    for (int s = 0; s < seq_len; ++s) {
        float* row = x + s * dim;
        float mean = 0.0f;
        for (int i = 0; i < dim; ++i) mean += row[i];
        mean /= dim;
        float var = 0.0f;
        for (int i = 0; i < dim; ++i) {
            float d = row[i] - mean;
            var += d * d;
        }
        var /= dim;
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (int i = 0; i < dim; ++i) {
            row[i] = (row[i] - mean) * inv_std * weight[i] + bias[i];
        }
    }
}

// GELU (approximate, "gelu_new" used by GPT-2):
//   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
void gelu_inplace(float* x, int n) {
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
    for (int i = 0; i < n; ++i) {
        float v = x[i];
        float cube = v * v * v;
        float inner = sqrt_2_over_pi * (v + 0.044715f * cube);
        x[i] = 0.5f * v * (1.0f + std::tanh(inner));
    }
}

// Softmax in-place along last axis.
void softmax_rows_inplace(float* data, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float* row = data + r * cols;
        float max_val = *std::max_element(row, row + cols);
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            row[c] = std::exp(row[c] - max_val);
            sum += row[c];
        }
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; ++c) {
            row[c] *= inv_sum;
        }
    }
}

// Apply causal mask: set upper triangle (j > i) to -infinity.
void apply_causal_mask(float* attn, int n_head, int seq_len) {
    for (int h = 0; h < n_head; ++h) {
        for (int i = 0; i < seq_len; ++i) {
            for (int j = i + 1; j < seq_len; ++j) {
                attn[h * seq_len * seq_len + i * seq_len + j] = -1e9f;
            }
        }
    }
}

// Add bias to each row: x[i, :] += bias[:] for i in [0, rows).
void add_bias(float* x, const float* bias, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            x[i * cols + j] += bias[j];
        }
    }
}

// ── Model weights ───────────────────────────────────────────────────────

struct TransformerBlock {
    dalotia::vector<float> ln_1_weight, ln_1_bias;
    dalotia::vector<float> c_attn_weight, c_attn_bias;       // [768, 2304]
    dalotia::vector<float> c_proj_weight, c_proj_bias;        // [768, 768]
    dalotia::vector<float> ln_2_weight, ln_2_bias;
    dalotia::vector<float> c_fc_weight, c_fc_bias;            // [768, 3072]
    dalotia::vector<float> c_proj_mlp_weight, c_proj_mlp_bias; // [3072, 768]
};

struct GPT2Model {
    dalotia::vector<float> wte;  // [50257, 768]
    dalotia::vector<float> wpe;  // [1024, 768]
    std::vector<TransformerBlock> blocks;
    dalotia::vector<float> ln_f_weight, ln_f_bias;
};

GPT2Model load_model(const std::string& filename) {
    auto file = std::unique_ptr<dalotia::TensorFile>(
        dalotia::make_tensor_file(filename));

    GPT2Model model;

    {
        auto [ext, data] = file->load_tensor_dense<float>("wte.weight", dalotia_float_32);
        assert(ext == std::vector<int>({VOCAB_SIZE, N_EMBD}));
        model.wte = std::move(data);
    }
    {
        auto [ext, data] = file->load_tensor_dense<float>("wpe.weight", dalotia_float_32);
        assert(ext == std::vector<int>({N_CTX, N_EMBD}));
        model.wpe = std::move(data);
    }

    model.blocks.resize(N_LAYER);
    for (int i = 0; i < N_LAYER; ++i) {
        std::string prefix = "h." + std::to_string(i) + ".";
        auto& blk = model.blocks[i];

        auto load = [&](const std::string& name) {
            auto [ext, data] = file->load_tensor_dense<float>(
                prefix + name, dalotia_float_32);
            return std::move(data);
        };

        blk.ln_1_weight = load("ln_1.weight");
        blk.ln_1_bias = load("ln_1.bias");
        blk.c_attn_weight = load("attn.c_attn.weight");
        blk.c_attn_bias = load("attn.c_attn.bias");
        blk.c_proj_weight = load("attn.c_proj.weight");
        blk.c_proj_bias = load("attn.c_proj.bias");
        blk.ln_2_weight = load("ln_2.weight");
        blk.ln_2_bias = load("ln_2.bias");
        blk.c_fc_weight = load("mlp.c_fc.weight");
        blk.c_fc_bias = load("mlp.c_fc.bias");
        blk.c_proj_mlp_weight = load("mlp.c_proj.weight");
        blk.c_proj_mlp_bias = load("mlp.c_proj.bias");
    }

    {
        auto [ext, data] = file->load_tensor_dense<float>("ln_f.weight", dalotia_float_32);
        model.ln_f_weight = std::move(data);
    }
    {
        auto [ext, data] = file->load_tensor_dense<float>("ln_f.bias", dalotia_float_32);
        model.ln_f_bias = std::move(data);
    }

    return model;
}

// ── Forward pass ────────────────────────────────────────────────────────

std::vector<float> forward(const GPT2Model& model,
                           const std::vector<int>& token_ids) {
    const int seq_len = static_cast<int>(token_ids.size());
    assert(seq_len > 0 && seq_len <= N_CTX);

    // x = wte[token_ids] + wpe[0..seq_len-1]  -> [seq_len, N_EMBD]
    std::vector<float> x(seq_len * N_EMBD);
    for (int s = 0; s < seq_len; ++s) {
        const float* tok_emb = model.wte.data() + token_ids[s] * N_EMBD;
        const float* pos_emb = model.wpe.data() + s * N_EMBD;
        float* dst = x.data() + s * N_EMBD;
        for (int i = 0; i < N_EMBD; ++i) {
            dst[i] = tok_emb[i] + pos_emb[i];
        }
    }

    // Scratch buffers (reused across layers)
    std::vector<float> ln_out(seq_len * N_EMBD);
    std::vector<float> qkv(seq_len * 3 * N_EMBD);
    std::vector<float> attn_scores(N_HEAD * seq_len * seq_len);
    std::vector<float> attn_out(seq_len * N_EMBD);
    std::vector<float> proj_out(seq_len * N_EMBD);
    std::vector<float> ffn_hidden(seq_len * FFN_DIM);
    std::vector<float> ffn_out(seq_len * N_EMBD);

    for (int layer = 0; layer < N_LAYER; ++layer) {
        const auto& blk = model.blocks[layer];

        // ── Pre-attention LayerNorm ─────────────────────────────────
        std::copy(x.begin(), x.end(), ln_out.begin());
        layer_norm(ln_out.data(), seq_len, N_EMBD,
                   blk.ln_1_weight.data(), blk.ln_1_bias.data());

        // ── QKV projection: qkv = ln_out @ c_attn.weight + c_attn.bias
        {
            auto A = const_mat_ref({seq_len, N_EMBD}, ln_out.data());
            auto B = const_mat_ref({N_EMBD, 3 * N_EMBD}, blk.c_attn_weight.data());
            auto C = mat_ref({seq_len, 3 * N_EMBD}, qkv.data());
            multi::blas::gemm(1.0f, A, B, 0.0f, C);
        }
        add_bias(qkv.data(), blk.c_attn_bias.data(), seq_len, 3 * N_EMBD);

        // ── Split Q, K, V and reshape for multi-head attention ──────
        // Q = qkv[:, 0:768], K = qkv[:, 768:1536], V = qkv[:, 1536:2304]
        // Reshape [seq_len, N_HEAD, HEAD_DIM] -> [N_HEAD, seq_len, HEAD_DIM]
        std::vector<float> Q(N_HEAD * seq_len * HEAD_DIM);
        std::vector<float> K(N_HEAD * seq_len * HEAD_DIM);
        std::vector<float> V(N_HEAD * seq_len * HEAD_DIM);

        for (int s = 0; s < seq_len; ++s) {
            for (int h = 0; h < N_HEAD; ++h) {
                for (int d = 0; d < HEAD_DIM; ++d) {
                    int src_q = s * 3 * N_EMBD + h * HEAD_DIM + d;
                    int src_k = s * 3 * N_EMBD + N_EMBD + h * HEAD_DIM + d;
                    int src_v = s * 3 * N_EMBD + 2 * N_EMBD + h * HEAD_DIM + d;
                    int dst_idx = h * seq_len * HEAD_DIM + s * HEAD_DIM + d;
                    Q[dst_idx] = qkv[src_q];
                    K[dst_idx] = qkv[src_k];
                    V[dst_idx] = qkv[src_v];
                }
            }
        }

        // ── Attention scores: Q @ K^T / sqrt(HEAD_DIM) ─────────────
        float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
        for (int h = 0; h < N_HEAD; ++h) {
            auto Qh = const_mat_ref({seq_len, HEAD_DIM},
                                    Q.data() + h * seq_len * HEAD_DIM);
            auto Kh = const_mat_ref({seq_len, HEAD_DIM},
                                    K.data() + h * seq_len * HEAD_DIM);
            auto Sh = mat_ref({seq_len, seq_len},
                              attn_scores.data() + h * seq_len * seq_len);
            // Sh = scale * Qh @ Kh^T
            multi::blas::gemm(scale, Qh, Kh.transposed(), 0.0f, Sh);
        }

        // ── Causal mask + softmax ───────────────────────────────────
        apply_causal_mask(attn_scores.data(), N_HEAD, seq_len);
        for (int h = 0; h < N_HEAD; ++h) {
            softmax_rows_inplace(
                attn_scores.data() + h * seq_len * seq_len,
                seq_len, seq_len);
        }

        // ── Attention output: attn_weights @ V ─────────────────────
        for (int h = 0; h < N_HEAD; ++h) {
            auto Sh = const_mat_ref({seq_len, seq_len},
                                    attn_scores.data() + h * seq_len * seq_len);
            auto Vh = const_mat_ref({seq_len, HEAD_DIM},
                                    V.data() + h * seq_len * HEAD_DIM);
            auto Oh = mat_ref({seq_len, HEAD_DIM},
                              attn_out.data() + h * seq_len * HEAD_DIM);
            multi::blas::gemm(1.0f, Sh, Vh, 0.0f, Oh);
        }

        // ── Reshape [N_HEAD, seq_len, HEAD_DIM] -> [seq_len, N_EMBD] ─
        std::vector<float> attn_concat(seq_len * N_EMBD);
        for (int s = 0; s < seq_len; ++s) {
            for (int h = 0; h < N_HEAD; ++h) {
                for (int d = 0; d < HEAD_DIM; ++d) {
                    attn_concat[s * N_EMBD + h * HEAD_DIM + d] =
                        attn_out[h * seq_len * HEAD_DIM + s * HEAD_DIM + d];
                }
            }
        }

        // ── Attention output projection ─────────────────────────────
        {
            auto A = const_mat_ref({seq_len, N_EMBD}, attn_concat.data());
            auto B = const_mat_ref({N_EMBD, N_EMBD}, blk.c_proj_weight.data());
            auto C = mat_ref({seq_len, N_EMBD}, proj_out.data());
            multi::blas::gemm(1.0f, A, B, 0.0f, C);
        }
        add_bias(proj_out.data(), blk.c_proj_bias.data(), seq_len, N_EMBD);

        // ── Residual connection ─────────────────────────────────────
        for (int i = 0; i < seq_len * N_EMBD; ++i) {
            x[i] += proj_out[i];
        }

        // ── Pre-FFN LayerNorm ───────────────────────────────────────
        std::copy(x.begin(), x.end(), ln_out.begin());
        layer_norm(ln_out.data(), seq_len, N_EMBD,
                   blk.ln_2_weight.data(), blk.ln_2_bias.data());

        // ── FFN: up projection ──────────────────────────────────────
        {
            auto A = const_mat_ref({seq_len, N_EMBD}, ln_out.data());
            auto B = const_mat_ref({N_EMBD, FFN_DIM}, blk.c_fc_weight.data());
            auto C = mat_ref({seq_len, FFN_DIM}, ffn_hidden.data());
            multi::blas::gemm(1.0f, A, B, 0.0f, C);
        }
        add_bias(ffn_hidden.data(), blk.c_fc_bias.data(), seq_len, FFN_DIM);

        // ── GELU activation ─────────────────────────────────────────
        gelu_inplace(ffn_hidden.data(), seq_len * FFN_DIM);

        // ── FFN: down projection ────────────────────────────────────
        {
            auto A = const_mat_ref({seq_len, FFN_DIM}, ffn_hidden.data());
            auto B = const_mat_ref({FFN_DIM, N_EMBD}, blk.c_proj_mlp_weight.data());
            auto C = mat_ref({seq_len, N_EMBD}, ffn_out.data());
            multi::blas::gemm(1.0f, A, B, 0.0f, C);
        }
        add_bias(ffn_out.data(), blk.c_proj_mlp_bias.data(), seq_len, N_EMBD);

        // ── Residual connection ─────────────────────────────────────
        for (int i = 0; i < seq_len * N_EMBD; ++i) {
            x[i] += ffn_out[i];
        }
    }

    // ── Final LayerNorm ─────────────────────────────────────────────────
    layer_norm(x.data(), seq_len, N_EMBD,
               model.ln_f_weight.data(), model.ln_f_bias.data());

    // ── Logits: x @ wte^T (tied embeddings) ────────────────────────────
    std::vector<float> logits(seq_len * VOCAB_SIZE);
    {
        auto A = const_mat_ref({seq_len, N_EMBD}, x.data());
        auto B = const_mat_ref({VOCAB_SIZE, N_EMBD}, model.wte.data());
        auto C = mat_ref({seq_len, VOCAB_SIZE}, logits.data());
        multi::blas::gemm(1.0f, A, B.transposed(), 0.0f, C);
    }

    return logits;
}

// ── Greedy decode helper ────────────────────────────────────────────────

int argmax(const float* data, int n) {
    return static_cast<int>(
        std::distance(data, std::max_element(data, data + n)));
}

// ── Main ────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    std::string model_path = "./model.safetensors";
    if (argc > 1) model_path = argv[1];

    // Prompt tokens for "The meaning of life is"
    // (GPT-2 BPE token IDs — hardcoded to avoid tokenizer dependency)
    std::vector<int> prompt_tokens = {464, 3616, 286, 1204, 318};

    int num_generate = 20;
    if (argc > 2) num_generate = std::stoi(argv[2]);

    // ── Load model ──────────────────────────────────────────────────────
    std::cout << "Loading GPT-2 124M from " << model_path << " ..." << std::endl;
    auto t_load_start = std::chrono::high_resolution_clock::now();
    GPT2Model model = load_model(model_path);
    auto t_load_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> load_duration = t_load_end - t_load_start;
    std::cout << "Model loaded in " << load_duration.count() << "s" << std::endl;

    // ── Inference: greedy autoregressive generation ─────────────────────
    std::cout << "Generating " << num_generate << " tokens..." << std::endl;
    std::vector<int> tokens = prompt_tokens;

    auto t_infer_start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < num_generate; ++step) {
        std::vector<float> logits = forward(model, tokens);
        const float* last_logits = logits.data() + (tokens.size() - 1) * VOCAB_SIZE;
        int next_token = argmax(last_logits, VOCAB_SIZE);
        tokens.push_back(next_token);
    }
    auto t_infer_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> infer_duration = t_infer_end - t_infer_start;

    std::cout << "Inference time: " << infer_duration.count() << "s ("
              << num_generate << " tokens, "
              << infer_duration.count() / num_generate << "s/token)" << std::endl;

    // ── Print generated token IDs ───────────────────────────────────────
    std::cout << "Generated token IDs: [";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << tokens[i];
    }
    std::cout << "]" << std::endl;

    // ── Basic sanity check ──────────────────────────────────────────────
    {
        std::vector<float> logits = forward(model, prompt_tokens);
        bool all_finite = true;
        for (int i = 0; i < static_cast<int>(prompt_tokens.size()) * VOCAB_SIZE; ++i) {
            if (!std::isfinite(logits[i])) { all_finite = false; break; }
        }
        if (!all_finite) {
            std::cerr << "FAIL: logits contain non-finite values!" << std::endl;
            return 1;
        }
        int first_pred = argmax(
            logits.data() + (prompt_tokens.size() - 1) * VOCAB_SIZE, VOCAB_SIZE);
        std::cout << "First predicted token after prompt: " << first_pred << std::endl;
    }

    std::cout << "success!" << std::endl;
    return 0;
}
