#include <cassert>
#include <chrono>
#include <cstring>  // std::memcpy
#include <iostream>

#include "dalotia.hpp"
#include "dalotia_safetensors_file.hpp"

#include <torch/script.h> // this exec is only built with libtorch


void assert_close(float a, float b, float tol = 1e-4) {
#ifndef NDEBUG
    if (std::abs(a - b) >= tol) {
        std::cerr << "assertion failed: " << a << " != " << b << std::endl;
    }
    assert(std::abs(a - b) < tol);
#endif  // NDEBUG
}

std::chrono::duration<double> run_inference_libtorch(
    const dalotia::vector<float> &inputs,
    const std::vector<int> &input_sizes,
    size_t num_repetitions,
    const std::vector<int> &output_sizes,
    dalotia::vector<float> &results) {
    constexpr size_t batch_size = 64;

    torch::jit::script::Module module = torch::jit::load("traced_SubgridLESNet.pt");
    module = torch::jit::optimize_for_inference(module);

    const auto start = std::chrono::high_resolution_clock::now();
    // todo keep const semantics on input -> seems that's not a thing in libtorch
    const auto input_tensor = torch::from_blob(
            reinterpret_cast<void*>(const_cast<float*>(inputs.data())), 
            at::IntArrayRef({input_sizes[0], input_sizes[1]})
        );
    // If I can't figure out how to pass the output address to forward(), then most other people also won't 
    // auto output_tensor = torch::from_blob(
    //         reinterpret_cast<void*>(results.data()), 
    //         at::IntArrayRef({output_sizes[0], output_sizes[1]})
    //     );
    for (size_t r = 0; r < num_repetitions; ++r) {
        auto output_tensor = module.forward({input_tensor}).toTensor();
        // assign to output
        std::memcpy(results.data(), output_tensor.data_ptr(), output_tensor.numel() * sizeof(float));
    }
    return std::chrono::high_resolution_clock::now() - start;
}

int main(int, char **) {
    // load all weights, biases, input, and output
    // the data used here is generated with generate_models.py
    std::string filename = "./weights_SubgridLESNet.safetensors";

    // unpermuted for now
    auto [input_extents, input_tensor] = dalotia::load_tensor_dense<float>("./input_SubgridLESNet.safetensors", "random_input",
                                          dalotia_WeightFormat::dalotia_float_32, dalotia_Ordering::dalotia_C_ordering);
    assert(input_extents == std::vector<int>({4096, 10}));
    assert_close(input_tensor[0], 0.4963);
    assert_close(input_tensor[1], 0.7682);
    assert_close(input_tensor[10], 0.3489);
    auto [output_extents, expected_output_tensor] =
        dalotia::load_tensor_dense<float>("./output_SubgridLESNet.safetensors", "output",
                                          dalotia_WeightFormat::dalotia_float_32, dalotia_Ordering::dalotia_C_ordering);
    assert(output_extents == std::vector<int>({4096, 6}));
    assert_close(expected_output_tensor[0], 1.0331);
    assert_close(expected_output_tensor[1], 0.0446);
    assert_close(expected_output_tensor[6], 0.8264);
    
    const size_t num_repetitions = 1000;
    dalotia::vector<float> results(expected_output_tensor.size());
    std::cout << "Running inference with libtorch" << std::endl;
    const auto duration = run_inference_libtorch(input_tensor, input_extents, num_repetitions, output_extents, results);
    std::cout << "Duration: " << duration.count() << "s" << std::endl;
    std::cout << "On average: " << duration.count() / static_cast<float>(num_repetitions) << "s" << std::endl;

    // check correctness of the output
    for (size_t i = 0; i < results.size(); ++i) {
        if (results[i] != expected_output_tensor[i]) {
            std::cerr << "results[" << i << "] = " << results[i] << 
                            " != expected_output_tensor[" << i << "] = " << expected_output_tensor[i] << std::endl;
            throw std::runtime_error("results != expected_output_tensor");
        }
    }

    return 0;
}
