#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>  // std::memset
#include <iostream>

#include "dalotia.hpp"
#include "dalotia_safetensors_file.hpp"

#ifdef DALOTIA_E_WITH_NDIRECT
#include <NDIRECT_direct.h>
#endif  // DALOTIA_E_WITH_NDIRECT

#ifdef DALOTIA_E_WITH_LIBTORCH
#include <torch/script.h>
#endif  // DALOTIA_E_WITH_LIBTORCH

std::tuple<dalotia::vector<int>, dalotia::vector<float>, dalotia::vector<int>, dalotia::vector<float>> test_load(
    std::string filename, std::string layer_name) {
    std::string tensor_name_weight = layer_name + ".weight";
    std::string tensor_name_bias = layer_name + ".bias";
    const dalotia_Ordering ordering = dalotia_Ordering::dalotia_C_ordering;
    constexpr dalotia_WeightFormat weightFormat =
        dalotia_WeightFormat::dalotia_float_32;
    // unpermuted for now
    auto [extents_weight, tensor_weight_cpp] =
        dalotia::load_tensor_dense<float>(filename, tensor_name_weight,
                                          weightFormat, ordering);
    auto [extents_bias, tensor_bias_cpp] = dalotia::load_tensor_dense<float>(
        filename, tensor_name_bias, weightFormat, ordering);
    return {extents_weight, tensor_weight_cpp, extents_bias, tensor_bias_cpp};
}

void assert_close(float a, float b, float tol = 1e-4) {
#ifndef NDEBUG
    if (std::abs(a - b) >= tol) {
        std::cerr << "assertion failed: " << a << " != " << b << std::endl;
    }
    assert(std::abs(a - b) < tol);
#endif  // NDEBUG
}

template <int dim>
std::function<int(std::array<int, dim>)> get_tensor_indexer(
    const std::array<int, dim> &extents) {
    std::array<int, dim> strides;
    // std::exclusive_scan(extents.rbegin(), extents.rend(), strides.rbegin(),
    // 1,
    //                     std::multiplies<int>());
    strides.back() = 1;  // TODO detect if exclusive_scan is available?
    for (size_t i = dim - 1; i > 0; --i) {
        strides[i -1] = strides[i] * extents[i];
    }

    const std::array<int, dim> const_strides = strides;
    return [const_strides, &extents](std::array<int, dim> indices) {
#ifndef NDEBUG
        for (int i = 0; i < dim; ++i) {
            if (indices[i] < 0 || indices[i] >= extents[i]) {
                for (const auto &index : indices) {
                    std::cerr << index << ", ";
                }
                std::cerr << " / ";
                for (const auto &extent : extents) {
                    std::cerr << extent << ", ";
                }
                std::cerr << ": index " << i << " out of bounds: " << indices[i]
                          << " not in [0, " << extents[i] << ")" << std::endl;
            }
            assert(indices[i] >= 0 && indices[i] < extents[i]);
        }
#endif  // NDEBUG
        return std::inner_product(indices.begin(), indices.end(),
                                  const_strides.begin(), 0);
    };
}

constexpr std::array<int, 5> conv1_weight_extents = {8, 3, 3, 3, 3};
constexpr std::array<int, 5> conv2_weight_extents = {8, 8, 3, 3, 3};
constexpr std::array<int, 5> conv3_weight_extents = {4, 8, 3, 3, 3};
constexpr std::array<int, 5> conv4_weight_extents = {1, 4, 2, 2, 2};



std::chrono::duration<double> run_inference_slow_loops(
    const dalotia::vector<float> &conv1_weight,
    const dalotia::vector<float> &conv1_bias,
    const dalotia::vector<float> &conv2_weight,
    const dalotia::vector<float> &conv2_bias,
    const dalotia::vector<float> &conv3_weight,
    const dalotia::vector<float> &conv3_bias,
    const dalotia::vector<float> &conv4_weight,
    const dalotia::vector<float> &conv4_bias,
    const dalotia::vector<float> &input_tensor,
    size_t num_repetitions,
    dalotia::vector<float> &results){

    const auto conv1_weight_indexer =
        get_tensor_indexer<5>(conv1_weight_extents);
    const auto conv2_weight_indexer =
        get_tensor_indexer<5>(conv2_weight_extents);

    const int num_inputs = input_tensor.size() / (6*6*6) / 3;
    const std::array<int, 5> input_extents = {num_inputs, 3, 6, 6, 6};
    const auto input_indexer =
        get_tensor_indexer<5>(input_extents);
    const std::array<int, 5> input_padded_extents = {num_inputs, 3, 8, 8, 8};
    const auto input_padded_indexer =
        get_tensor_indexer<5>(input_padded_extents);
    auto input_vector_padded = dalotia::vector<float>(std::accumulate(input_padded_extents.begin(), input_padded_extents.end(), 1, std::multiplies<int>()));


    const std::array<int, 5> conv1_output_extents = {num_inputs, 8, 6, 6, 6};
    const auto conv1_output_indexer = get_tensor_indexer<5>(conv1_output_extents);
    auto conv1_output = dalotia::vector<float>(std::accumulate(conv1_output_extents.begin(), conv1_output_extents.end(), 1, std::multiplies<int>()));

    const auto start = std::chrono::high_resolution_clock::now();
    for (size_t batch_index = 0; batch_index < num_repetitions; ++batch_index) {
        // copy data to larger array for zero-padding at the edges
        for (int o = 0; o < num_inputs; ++o) {
            for (int c = 0; c < 3; ++c) {
                for (int k = 0; k < 6; ++k) {
                    for (int i = 0; i < 6; ++i) {
                        for (int j = 0; j < 6; ++j) {
                            input_vector_padded[input_padded_indexer(
                                {o, c, k + 1, i + 1, j + 1})] =
                                input_tensor[input_indexer(
                                    {o, c, k, i, j})];
                        }
                    }
                }
            }
        }

        // apply first convolution + ReLU
        for (int o = 0; o < num_inputs; ++o) {
            for (int c = 0; c < 3; ++c) {
                for (int k = 1; k < 7; ++k) {
                    for (int i = 1; i < 7; ++i) {
                        for (int j = 1; j < 7; ++j) {
                            for (int f = 0; f < 3; ++f) {
                                float value = conv1_bias[c];
                                for (int l = 0; l < 3; ++l) {
                                    for (int m = 0; m < 3; ++m) {
                                        for (int n = 0; n < 3; ++n) {
                                            value +=
                                                conv1_weight[conv1_weight_indexer(
                                                    {f, c, l, m, n})] *
                                                input_vector_padded[input_padded_indexer(
                                                    {o, c, k + l - 1, i + m - 1, j + n - 1})];
                                        }
                                    }
                                }
                                if (value < 0.) {
                                    value = 0.;
                                }
                                conv1_output[conv1_output_indexer(
                                    {o, c, k - 1, i - 1, j - 1})] = value;
                            }
                        }
                    }
                }
            }
        }
    }
    return std::chrono::high_resolution_clock::now() - start;
}



int main(int, char **) {
    // the data used here is generated with generate_models.py
    std::string filename = "./weights_DeepRLEddyNet.safetensors";

    const auto [read_conv1_weight_extents, conv1_weight, conv1_bias_extents, conv1_bias] = test_load(filename, "conv1");
    assert(read_conv1_weight_extents == std::vector<int>({8, 3, 3, 3, 3}));
    const auto [read_conv2_weight_extents, conv2_weight, conv2_bias_extents,conv2_bias] = test_load(filename, "conv2");
    assert(read_conv2_weight_extents == std::vector<int>({8, 8, 3, 3, 3}));
    const auto [read_conv3_weight_extents, conv3_weight, conv3_bias_extents,conv3_bias] = test_load(filename, "conv3");
    assert(read_conv3_weight_extents == std::vector<int>({4, 8, 3, 3, 3}));
    const auto [read_conv4_weight_extents, conv4_weight, conv4_bias_extents,conv4_bias] = test_load(filename, "conv4");
    assert(read_conv4_weight_extents == std::vector<int>({1, 4, 2, 2, 2}));

    // unpermuted for now
    auto [input_extents, input_tensor] =dalotia::load_tensor_dense<float>("./input_DeepRLEddyNet.safetensors", "random_input",
                                          dalotia_WeightFormat::dalotia_float_32, dalotia_Ordering::dalotia_C_ordering);
    assert(input_extents == std::vector<int>({16*16*16, 3, 6, 6, 6}));
    assert_close(input_tensor[0], 4.9626e-01);
    assert_close(input_tensor[1], 7.6822e-01);
    assert_close(input_tensor[input_tensor.size() - 1], 7.8506e-01);
    auto [output_extents, expected_output_tensor] =
        dalotia::load_tensor_dense<float>("./output_DeepRLEddyNet.safetensors", "output",
                                          dalotia_WeightFormat::dalotia_float_32, dalotia_Ordering::dalotia_C_ordering);
    assert(output_extents == std::vector<int>(1, 16*16*16));
    assert_close(expected_output_tensor[0], 0.4075);
    assert_close(expected_output_tensor[1], 0.4049);
    assert_close(expected_output_tensor[4095], 0.4164);

    typedef std::function<std::chrono::duration<double>(
        const dalotia::vector<float> &conv1_weight,
        const dalotia::vector<float> &conv1_bias,
        const dalotia::vector<float> &conv2_weight,
        const dalotia::vector<float> &conv2_bias,
        const dalotia::vector<float> &conv3_weight,
        const dalotia::vector<float> &conv3_bias,
        const dalotia::vector<float> &conv4_weight,
        const dalotia::vector<float> &conv4_bias,
        const dalotia::vector<float> &input_tensor,
        size_t num_repetitions,
        dalotia::vector<float> &results)>
        inference_function;
    std::unordered_map<std::string, inference_function> inference_functions;

    inference_functions["slow_loops"] = run_inference_slow_loops;

    const size_t num_repetitions = 1;
    dalotia::vector<float> results(expected_output_tensor.size());
    for (const auto &inference_function_pair : inference_functions) {
        const auto& inference_function = inference_function_pair.second;
        std::cout << "Running inference with " << inference_function_pair.first << std::endl;
        std::memset(results.data(), 0, results.size() * sizeof(int));

        const auto duration = inference_function(
            conv1_weight, conv1_bias, conv2_weight,
            conv2_bias, conv3_weight, conv3_bias, conv4_weight, conv4_bias,
            input_tensor, num_repetitions, results);
        // check correctness of the output
        if (results != expected_output_tensor) {
            throw std::runtime_error("results != expected_output_tensor");
        }
        std::cout << "Duration: " << duration.count() << "s" << std::endl;
        std::cout << "On average: " << duration.count() / static_cast<float>(num_repetitions) << "s" << std::endl;
    }

    std::cout << "All benched!" << std::endl;

    return 0;
}

