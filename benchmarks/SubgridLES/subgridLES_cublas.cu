// cuBLAS benchmark for the SubgridLES 2-layer MLP (10 -> 300 -> 6)
// Mirrors run_inference_cblas from subgridLES.cpp but runs on GPU.

#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "dalotia.hpp"
#include "dalotia_safetensors_file.hpp"
#ifdef DALOTIA_WITH_CUFILE
#include "dalotia_cufile.hpp"
#endif

struct CudaDeleter {
    void operator()(void *p) const noexcept { if (p) cudaFree(p); }
};
using CudaPtr = std::unique_ptr<void, CudaDeleter>;

// Typed convenience: returns a CudaPtr, provides a raw T* via .get()
template <typename T>
struct CudaBuffer {
    CudaPtr ptr;
    size_t count;

    CudaBuffer(size_t n) : count(n) {
        void *raw = nullptr;
        cudaError_t err = cudaMalloc(&raw, n * sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
        }
        ptr.reset(raw);
    }

    T *get() const noexcept { return static_cast<T *>(ptr.get()); }
    size_t bytes() const noexcept { return count * sizeof(T); }
};

struct CublasHandleDeleter {
    void operator()(cublasHandle_t *h) const noexcept {
        if (h) { cublasDestroy(*h); delete h; }
    }
};

auto make_cublas_handle() {
    auto h = std::make_unique<cublasHandle_t>();
    cublasStatus_t status = cublasCreate(h.get());
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasCreate failed: " +
                                 std::to_string(status));
    }
    return std::unique_ptr<cublasHandle_t, CublasHandleDeleter>(
        h.release());
}

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__   \
                      << " : " << cudaGetErrorString(err) << std::endl;    \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

#define CHECK_CUBLAS(call)                                                 \
    do {                                                                   \
        cublasStatus_t status = (call);                                    \
        if (status != CUBLAS_STATUS_SUCCESS) {                             \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ \
                      << " : " << status << std::endl;                     \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

// ── Kernels ──────────────────────────────────────────────────────────────

__global__ void relu_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = fmaxf(data[idx], 0.0f);
    }
}

__global__ void broadcast_bias_kernel(float *out, const float *bias,
                                      int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        out[idx] = bias[idx % rows];
    }
}

// ── Main ─────────────────────────────────────────────────────────────────

void assert_close(float a, float b, float tol = 1e-4) {
    if (std::abs(a - b) >= tol) {
        std::cerr << "assertion failed: " << a << " != " << b << std::endl;
        assert(false);
    }
}

int main(int argc, char *argv[]) {
    int num_inputs = 16 * 16 * 16;
    if (argc > 1) {
        num_inputs = std::stoi(argv[1]);
    }

    // ── Load input / expected output on CPU ──────────────────────────────
    auto [input_extents, input_tensor] = dalotia::load_tensor_dense<float>(
        "./input_SubgridLESNet.safetensors", "random_input",
        dalotia_float_32, dalotia_C_ordering);
    assert(input_extents == std::vector<int>({16 * 16 * 16, 10}));
    assert_close(input_tensor[0], 0.4963);

    auto [output_extents, expected_output_tensor] =
        dalotia::load_tensor_dense<float>(
            "./output_SubgridLESNet.safetensors", "output",
            dalotia_float_32, dalotia_C_ordering);
    assert(output_extents == std::vector<int>({16 * 16 * 16, 6}));
    assert_close(expected_output_tensor[0], 2.84722);

    // Resize input/output if needed
    if (input_extents[0] != num_inputs) {
        std::cout << "Resizing input/output tensor from " << input_extents[0]
                  << " to " << num_inputs << std::endl;
        size_t initial_input_size = input_tensor.size();
        input_tensor.resize(num_inputs * 10);
        input_extents[0] = num_inputs;
        size_t initial_output_size = expected_output_tensor.size();
        expected_output_tensor.resize(num_inputs * 6);
        output_extents[0] = num_inputs;
        for (size_t i = initial_input_size; i < input_tensor.size(); ++i)
            input_tensor[i] = input_tensor[i % initial_input_size];
        for (size_t i = initial_output_size; i < expected_output_tensor.size(); ++i)
            expected_output_tensor[i] = expected_output_tensor[i % initial_output_size];
    }

    const int num_input_features = 10;
    const int num_hidden_neurons = 300;
    const int num_output_features = 6;

    // ── Load model weights to GPU ────────────────────────────────────────
    std::string weights_file = "./weights_SubgridLESNet.safetensors";

    CudaBuffer<float> d_weights_1(num_hidden_neurons * num_input_features);
    CudaBuffer<float> d_biases_1(num_hidden_neurons);
    CudaBuffer<float> d_weights_2(num_output_features * num_hidden_neurons);
    CudaBuffer<float> d_biases_2(num_output_features);

    std::cout << "Loading weights with GPU Direct Storage" << std::endl;
    dalotia::CuFileDriver gds_driver;
    auto dalotia_file = std::unique_ptr<dalotia::TensorFile>(
        dalotia::make_tensor_file(weights_file));
    dalotia_file->load_tensor_dense("fc1.weight", dalotia_float_32,
        dalotia_C_ordering, reinterpret_cast<dalotia_byte *>(d_weights_1.get()));
    dalotia_file->load_tensor_dense("fc1.bias", dalotia_float_32,
        dalotia_C_ordering, reinterpret_cast<dalotia_byte *>(d_biases_1.get()));
    dalotia_file->load_tensor_dense("fc2.weight", dalotia_float_32,
        dalotia_C_ordering, reinterpret_cast<dalotia_byte *>(d_weights_2.get()));
    dalotia_file->load_tensor_dense("fc2.bias", dalotia_float_32,
        dalotia_C_ordering, reinterpret_cast<dalotia_byte *>(d_biases_2.get()));

    // ── Allocate inference buffers ───────────────────────────────────────
    CudaBuffer<float> d_input(num_inputs * num_input_features);
    CudaBuffer<float> d_hidden(num_inputs * num_hidden_neurons);
    CudaBuffer<float> d_output(num_inputs * num_output_features);

    CHECK_CUDA(cudaMemcpy(d_input.get(), input_tensor.data(),
                          d_input.bytes(), cudaMemcpyHostToDevice));

    // ── cuBLAS setup ─────────────────────────────────────────────────────
    auto cublas = make_cublas_handle();

    const float alpha = 1.0f;
    const float beta_add = 1.0f;
    constexpr int threads_per_block = 256;

    const size_t num_repetitions = 1000;
    std::cout << "Running inference with cuBLAS (" << num_repetitions
              << " repetitions, " << num_inputs << " inputs)" << std::endl;

    // ── Timed inference loop ─────────────────────────────────────────────
    CHECK_CUDA(cudaDeviceSynchronize());
    const auto start = std::chrono::high_resolution_clock::now();

    for (size_t r = 0; r < num_repetitions; ++r) {
        // Layer 1: hidden = weights_1^T * input + bias_1
        {
            int n = num_hidden_neurons * num_inputs;
            broadcast_bias_kernel<<<(n + threads_per_block - 1) / threads_per_block,
                                    threads_per_block>>>(
                d_hidden.get(), d_biases_1.get(), num_hidden_neurons, num_inputs);
        }

        CHECK_CUBLAS(cublasSgemm(*cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            num_hidden_neurons, num_inputs, num_input_features,
            &alpha,
            d_weights_1.get(), num_input_features,
            d_input.get(), num_input_features,
            &beta_add,
            d_hidden.get(), num_hidden_neurons));

        // ReLU
        {
            int n = num_hidden_neurons * num_inputs;
            relu_kernel<<<(n + threads_per_block - 1) / threads_per_block,
                          threads_per_block>>>(d_hidden.get(), n);
        }

        // Layer 2: output = weights_2^T * hidden + bias_2
        {
            int n = num_output_features * num_inputs;
            broadcast_bias_kernel<<<(n + threads_per_block - 1) / threads_per_block,
                                    threads_per_block>>>(
                d_output.get(), d_biases_2.get(), num_output_features, num_inputs);
        }

        CHECK_CUBLAS(cublasSgemm(*cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            num_output_features, num_inputs, num_hidden_neurons,
            &alpha,
            d_weights_2.get(), num_hidden_neurons,
            d_hidden.get(), num_hidden_neurons,
            &beta_add,
            d_output.get(), num_output_features));
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    const auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Duration: " << duration.count() << "s" << std::endl;
    std::cout << "On average: " << duration.count() / static_cast<double>(num_repetitions)
              << "s" << std::endl;

    // ── Copy results back and verify ─────────────────────────────────────
    std::vector<float> results(num_inputs * num_output_features);
    CHECK_CUDA(cudaMemcpy(results.data(), d_output.get(),
                          results.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_inputs; ++i) {
        for (int j = 0; j < num_output_features; ++j) {
            float got = results[j + i * num_output_features];
            float expected = expected_output_tensor[i * num_output_features + j];
            if (std::abs(got - expected) > 1e-3) {
                std::cerr << "results[" << i << "," << j << "] = " << got
                          << " != expected " << expected << std::endl;
                std::cerr << "FAIL: results do not match expected output" << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
    }
    std::cout << "success!" << std::endl;
    return 0;
}
