#include <algorithm>
#include <cassert>
#include <chrono>
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


int main(int, char **) {
    // the data used here is generated with generate_models.py
    std::string filename = "./weights_DeepRLEddyNet.safetensors";

    const auto [conv1_weight_extents, conv1_weight, conv1_bias_extents, conv1_bias] = test_load(filename, "conv1");
    assert(conv1_weight_extents == std::vector<int>({8, 3, 3, 3, 3}));
    const auto [conv2_weight_extents, conv2_weight, conv2_bias_extents,conv2_bias] = test_load(filename, "conv2");
    assert(conv2_weight_extents == std::vector<int>({8, 8, 3, 3, 3}));
    const auto [conv3_weight_extents, conv3_weight, conv3_bias_extents,conv3_bias] = test_load(filename, "conv3");
    assert(conv3_weight_extents == std::vector<int>({4, 8, 3, 3, 3}));
    const auto [conv4_weight_extents, conv4_weight, conv4_bias_extents,conv4_bias] = test_load(filename, "conv4");
    assert(conv4_weight_extents == std::vector<int>({1, 4, 2, 2, 2}));

    return 0;
}

