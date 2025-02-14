#include <algorithm>
#include <cassert>
#include <cmath>
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

#ifdef DALOTIA_E_WITH_ONEDNN
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_debug.h>
#endif  // DALOTIA_E_WITH_ONEDNN

std::tuple<std::vector<int>, dalotia::vector<float>, std::vector<int>, dalotia::vector<float>> test_load(
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

bool is_close(float a, float b, float tol = 1e-4) {
    return std::abs(a - b) < tol;
}

void assert_close(float a, float b, float tol = 1e-4) {
#ifndef NDEBUG
    auto is_close_result = is_close(a, b, tol);
    if (!is_close_result) {
        std::cerr << "assertion failed: " << a << " != " << b << std::endl;
    }
    assert(is_close_result);
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
#warning "asserts are enabled -- takes almost forever!"
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

void conv3d_with_relu_naive(
            const dalotia::vector<float> &conv_weight,
            const std::array<int, 5> &conv_weight_extents,
            const dalotia::vector<float> &conv_bias,
            const dalotia::vector<float> &input_tensor,
            const std::array<int, 5> &input_extents,
            dalotia::vector<float> &output) {
    const auto& [F, C, L, M, N] = conv_weight_extents;
    const auto& [num_inputs, C2, K, I, J] = input_extents;
    assert(C2 == C);
    const std::array<int, 5> output_extents = {num_inputs, F, K - 2, I - 2, J - 2};
    assert(output.size() == std::accumulate(output_extents.begin(), output_extents.end(), 1, std::multiplies<int>()));
    const auto conv_weight_indexer = get_tensor_indexer<5>(conv_weight_extents);
    const auto input_indexer = get_tensor_indexer<5>(input_extents);
    const auto output_indexer = get_tensor_indexer<5>(output_extents);

    // stride 1, padding 1
#pragma omp parallel for schedule(static) collapse(2)
    for (int o = 0; o < num_inputs; ++o) {
        for (int f = 0; f < F; ++f) {
            for (int k = 0; k < (K - L + 1); ++k) {
                for (int i = 0; i < (I - M + 1); ++i) {
                    for (int j = 0; j < (J - N + 1); ++j) {
                        float value = conv_bias[f];
                        for (int c = 0; c < C; ++c) {
                            for (int l = 0; l < L; ++l) {
                                for (int m = 0; m < M; ++m) {
                                    for (int n = 0; n < N; ++n) {
                                        value +=
                                            conv_weight[conv_weight_indexer(
                                                {f, c, l, m, n})] *
                                            input_tensor[input_indexer(
                                                {o, c, k + l, i + m, j + n})];
                                    }
                                }
                            }
                        }
                        output[output_indexer({o, f, k, i, j})] = std::max(0.f, value);
                    }
                }
            }
        }
    }
}

std::chrono::duration<double> run_inference_slow_loops(
    const dalotia::vector<float> &input_tensor,
    const std::vector<int> &input_extents,
    size_t num_repetitions,
    dalotia::vector<float> &results){

    std::string filename = "./weights_DeepRLEddyNet.safetensors";

    const auto [read_conv1_weight_extents, conv1_weight, conv1_bias_extents, conv1_bias] = test_load(filename, "conv1");
    assert(read_conv1_weight_extents == std::vector<int>({8, 3, 3, 3, 3}));
    const auto [read_conv2_weight_extents, conv2_weight, conv2_bias_extents, conv2_bias] = test_load(filename, "conv2");
    assert(read_conv2_weight_extents == std::vector<int>({8, 8, 3, 3, 3}));
    const auto [read_conv3_weight_extents, conv3_weight, conv3_bias_extents, conv3_bias] = test_load(filename, "conv3");
    assert(read_conv3_weight_extents == std::vector<int>({4, 8, 3, 3, 3}));
    const auto [read_conv4_weight_extents, conv4_weight, conv4_bias_extents, conv4_bias] = test_load(filename, "conv4");
    assert(read_conv4_weight_extents == std::vector<int>({1, 4, 2, 2, 2}));

    const auto conv4_weight_indexer =
        get_tensor_indexer<5>(conv4_weight_extents);

    const int num_inputs = input_tensor.size() / (6*6*6) / 3;
    std::array<int, 5> input_extents_array;
    std::copy(input_extents.begin(), input_extents.end(), input_extents_array.begin());
    const auto input_indexer =
        get_tensor_indexer<5>(input_extents_array);
    const std::array<int, 5> input_padded_extents = {num_inputs, 3, 8, 8, 8};
    const auto input_padded_indexer =
        get_tensor_indexer<5>(input_padded_extents);
    auto input_vector_padded = dalotia::vector<float>(std::accumulate(input_padded_extents.begin(), input_padded_extents.end(), 1, std::multiplies<int>()));


    const std::array<int, 5> conv1_output_extents = {num_inputs, 8, 6, 6, 6};
    auto conv1_output = dalotia::vector<float>(std::accumulate(conv1_output_extents.begin(), conv1_output_extents.end(), 1, std::multiplies<int>()));
    const std::array<int, 5> conv2_output_extents = {num_inputs, 8, 4, 4, 4};
    auto conv2_output = dalotia::vector<float>(std::accumulate(conv2_output_extents.begin(), conv2_output_extents.end(), 1, std::multiplies<int>()));
    const std::array<int, 5> conv3_output_extents = {num_inputs, 4, 2, 2, 2};
    auto conv3_output = dalotia::vector<float>(std::accumulate(conv3_output_extents.begin(), conv3_output_extents.end(), 1, std::multiplies<int>()));
    const auto conv3_output_indexer =
        get_tensor_indexer<5>(conv3_output_extents);
    auto conv4_output = dalotia::vector<float>(num_inputs);

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
        conv3d_with_relu_naive(conv1_weight, conv1_weight_extents, conv1_bias, input_vector_padded, input_padded_extents, conv1_output);
        assert_close(conv1_output[0], 0.2333);
        conv3d_with_relu_naive(conv2_weight, conv2_weight_extents, conv2_bias, conv1_output, conv1_output_extents, conv2_output);
        assert_close(conv2_output[0], 1.3398);
        conv3d_with_relu_naive(conv3_weight, conv3_weight_extents, conv3_bias, conv2_output, conv2_output_extents, conv3_output);
        assert_close(conv3_output[0], 3.4653);
        for (int o = 0; o < num_inputs; ++o) {
            float value = conv4_bias[0];
            for (int c = 0; c < 4; ++c) {
                for (int l = 0; l < 2; ++l) {
                    for (int m = 0; m < 2; ++m) {
                        for (int n = 0; n < 2; ++n) {
                            value += conv4_weight[conv4_weight_indexer({0, c, l, m, n})] * conv3_output[conv3_output_indexer({o, c, l, m, n})];
                        }
                    }
                }
            }
            // apply half-sigmoid activation
            results[o] = 0.5 * 1. / (1. + std::exp(-value));
        }
    }
    return std::chrono::high_resolution_clock::now() - start;
}

#ifdef DALOTIA_E_WITH_ONEDNN
std::chrono::duration<double> run_inference_onednn(
    const dalotia::vector<float> &input_tensor,
    const std::vector<int> &input_extents,
    size_t num_repetitions,
    dalotia::vector<float> &results) {

    std::string filename = "./weights_DeepRLEddyNet.safetensors";
    dalotia::vector<float> conv1_weight;
    dalotia::vector<float> conv2_weight;
    dalotia::vector<float> conv3_weight;
    dalotia::vector<float> conv4_weight;
    dalotia::vector<float> conv1_bias;
    dalotia::vector<float> conv2_bias;
    dalotia::vector<float> conv3_bias;
    dalotia::vector<float> conv4_bias;
    void* conv1_weight_ptr = nullptr;
    void* conv2_weight_ptr = nullptr;
    void* conv3_weight_ptr = nullptr;
    void* conv4_weight_ptr = nullptr;
    void* conv1_bias_ptr = nullptr;
    void* conv2_bias_ptr = nullptr;
    void* conv3_bias_ptr = nullptr;
    void* conv4_bias_ptr = nullptr;

    bool use_raw_mmap_pointer = true; // to avoid another copy of tensors
    auto dalotia_file = std::unique_ptr<dalotia::TensorFile>(nullptr);
    if (use_raw_mmap_pointer) {
        // open the file but leave open until end of function so that
        // mmap pointers remain valid
        dalotia_file = std::move(std::unique_ptr<dalotia::TensorFile>(dalotia::make_tensor_file(filename)));
        // sorry for casting away const again
        conv1_weight_ptr = reinterpret_cast<void*>(const_cast<dalotia_byte*>(dalotia_file->get_mmap_tensor_pointers("conv1.weight")[0]));
        conv2_weight_ptr = reinterpret_cast<void*>(const_cast<dalotia_byte*>(dalotia_file->get_mmap_tensor_pointers("conv2.weight")[0]));
        conv3_weight_ptr = reinterpret_cast<void*>(const_cast<dalotia_byte*>(dalotia_file->get_mmap_tensor_pointers("conv3.weight")[0]));
        conv4_weight_ptr = reinterpret_cast<void*>(const_cast<dalotia_byte*>(dalotia_file->get_mmap_tensor_pointers("conv4.weight")[0]));
        conv1_bias_ptr = reinterpret_cast<void*>(const_cast<dalotia_byte*>(dalotia_file->get_mmap_tensor_pointers("conv1.bias")[0]));
        conv2_bias_ptr = reinterpret_cast<void*>(const_cast<dalotia_byte*>(dalotia_file->get_mmap_tensor_pointers("conv2.bias")[0]));
        conv3_bias_ptr = reinterpret_cast<void*>(const_cast<dalotia_byte*>(dalotia_file->get_mmap_tensor_pointers("conv3.bias")[0]));
        conv4_bias_ptr = reinterpret_cast<void*>(const_cast<dalotia_byte*>(dalotia_file->get_mmap_tensor_pointers("conv4.bias")[0]));
    } else {
        // open the file only once and close at end of scope
        auto dalotia_file_ = std::unique_ptr<dalotia::TensorFile>(dalotia::make_tensor_file(filename));
        std::vector<int> tensor_extents_dummy;
        std::tie(tensor_extents_dummy, conv1_weight) = dalotia_file_->load_tensor_dense<float>("conv1.weight");
        std::tie(tensor_extents_dummy, conv2_weight) = dalotia_file_->load_tensor_dense<float>("conv2.weight");
        std::tie(tensor_extents_dummy, conv3_weight) = dalotia_file_->load_tensor_dense<float>("conv3.weight");
        std::tie(tensor_extents_dummy, conv4_weight) = dalotia_file_->load_tensor_dense<float>("conv4.weight");
        std::tie(tensor_extents_dummy, conv1_bias) = dalotia_file_->load_tensor_dense<float>("conv1.bias");
        std::tie(tensor_extents_dummy, conv2_bias) = dalotia_file_->load_tensor_dense<float>("conv2.bias");
        std::tie(tensor_extents_dummy, conv3_bias) = dalotia_file_->load_tensor_dense<float>("conv3.bias");
        std::tie(tensor_extents_dummy, conv4_bias) = dalotia_file_->load_tensor_dense<float>("conv4.bias");
        conv1_weight_ptr = reinterpret_cast<void*>(conv1_weight.data());
        conv2_weight_ptr = reinterpret_cast<void*>(conv2_weight.data());
        conv3_weight_ptr = reinterpret_cast<void*>(conv3_weight.data());
        conv4_weight_ptr = reinterpret_cast<void*>(conv4_weight.data());
        conv1_bias_ptr = reinterpret_cast<void*>(conv1_bias.data());
        conv2_bias_ptr = reinterpret_cast<void*>(conv2_bias.data());
        conv3_bias_ptr = reinterpret_cast<void*>(conv3_bias.data());
        conv4_bias_ptr = reinterpret_cast<void*>(conv4_bias.data());
    }
    // cf. https://github.com/oneapi-src/oneDNN/blob/main/examples/cnn_inference_f32.cpp
    const dnnl::engine::kind engine_kind = dnnl::engine::kind::cpu;
    dnnl::engine eng(engine_kind, 0);
    dnnl::stream s(eng);    
    std::vector<dnnl::primitive> net;
    std::vector<std::unordered_map<int, dnnl::memory>> net_args;

    int num_inputs = input_extents[0];
    dnnl::memory::dims input_tz(input_extents.begin(), input_extents.end());
    dnnl::memory::dims conv1_weights_tz(conv1_weight_extents.begin(), conv1_weight_extents.end());
    dnnl::memory::dims conv2_weights_tz(conv2_weight_extents.begin(), conv2_weight_extents.end());
    dnnl::memory::dims conv3_weights_tz(conv3_weight_extents.begin(), conv3_weight_extents.end());
    dnnl::memory::dims conv4_weights_tz(conv4_weight_extents.begin(), conv4_weight_extents.end());
    dnnl::memory::dims conv1_bias_tz({conv1_weight_extents[0]});
    dnnl::memory::dims conv2_bias_tz({conv2_weight_extents[0]});
    dnnl::memory::dims conv3_bias_tz({conv3_weight_extents[0]});
    dnnl::memory::dims conv4_bias_tz({conv4_weight_extents[0]});
    dnnl::memory::dims conv1_output_extents = {input_extents[0], conv1_weight_extents[0], 6, 6, 6};
    dnnl::memory::dims conv2_output_extents = {input_extents[0], conv2_weight_extents[0], 4, 4, 4};
    dnnl::memory::dims conv3_output_extents = {input_extents[0], conv3_weight_extents[0], 2, 2, 2};
    dnnl::memory::dims conv_strides = {1, 1, 1};
    dnnl::memory::dims conv_yes_to_padding = {1, 1, 1};
    dnnl::memory::dims conv_no_to_padding = {0, 0, 0};

    auto conv1_original_weights_md = dnnl::memory::desc({conv1_weights_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::oidhw);
    auto conv2_original_weights_md = dnnl::memory::desc({conv2_weights_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::oidhw);
    auto conv3_original_weights_md = dnnl::memory::desc({conv3_weights_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::oidhw);
    auto conv4_original_weights_md = dnnl::memory::desc({conv4_weights_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::oidhw);
    auto conv1_original_bias_md = dnnl::memory::desc({conv1_bias_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
    auto conv2_original_bias_md = dnnl::memory::desc({conv2_bias_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
    auto conv3_original_bias_md = dnnl::memory::desc({conv3_bias_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
    auto conv4_original_bias_md = dnnl::memory::desc({conv4_bias_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);

    auto original_input_memory = dnnl::memory({{input_tz}, dnnl::memory::data_type::f32, 
        dnnl::memory::format_tag::ncdhw}, eng, reinterpret_cast<void*>(const_cast<float*>(input_tensor.data())));
    auto conv1_original_weights_memory = dnnl::memory(conv1_original_weights_md, eng, conv1_weight_ptr);
    auto conv2_original_weights_memory = dnnl::memory(conv2_original_weights_md, eng, conv2_weight_ptr);
    auto conv3_original_weights_memory = dnnl::memory(conv3_original_weights_md, eng, conv3_weight_ptr);
    auto conv4_original_weights_memory = dnnl::memory(conv4_original_weights_md, eng, conv4_weight_ptr);
    auto conv1_original_bias_memory = dnnl::memory(conv1_original_bias_md, eng, conv1_bias_ptr);
    auto conv2_original_bias_memory = dnnl::memory(conv2_original_bias_md, eng, conv2_bias_ptr);
    auto conv3_original_bias_memory = dnnl::memory(conv3_original_bias_md, eng, conv3_bias_ptr);
    auto conv4_original_bias_memory = dnnl::memory(conv4_original_bias_md, eng, conv4_bias_ptr);

    auto adapted_input_md = dnnl::memory::desc({input_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
    auto conv1_adapted_weights_md = dnnl::memory::desc({conv1_weights_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
    auto conv2_adapted_weights_md = dnnl::memory::desc({conv2_weights_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
    auto conv3_adapted_weights_md = dnnl::memory::desc({conv3_weights_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
    auto conv4_adapted_weights_md = dnnl::memory::desc({conv4_weights_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
    auto conv1_adapted_bias_md = dnnl::memory::desc({conv1_bias_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
    auto conv2_adapted_bias_md = dnnl::memory::desc({conv2_bias_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
    auto conv3_adapted_bias_md = dnnl::memory::desc({conv3_bias_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
    auto conv4_adapted_bias_md = dnnl::memory::desc({conv4_bias_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
    auto conv1_adapted_output_md = dnnl::memory::desc({conv1_output_extents}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
    auto conv2_adapted_output_md = dnnl::memory::desc({conv2_output_extents}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
    auto conv3_adapted_output_md = dnnl::memory::desc({conv3_output_extents}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
    auto conv4_adapted_output_md = dnnl::memory::desc({num_inputs, 1, 1, 1, 1}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);

    auto conv1_prim_desc = dnnl::convolution_forward::primitive_desc(eng,
            dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
            adapted_input_md, conv1_adapted_weights_md,
            conv1_adapted_bias_md, conv1_adapted_output_md, conv_strides, conv_yes_to_padding,
            conv_yes_to_padding);
    auto conv1_src_memory = original_input_memory;
    if (conv1_prim_desc.src_desc() != original_input_memory.get_desc()) {
        conv1_src_memory = dnnl::memory(conv1_prim_desc.src_desc(), eng);
        net.push_back(dnnl::reorder(original_input_memory, conv1_src_memory));
        net_args.push_back({{DNNL_ARG_FROM, original_input_memory},
                {DNNL_ARG_TO, conv1_src_memory}});
    }
    auto conv1_weights_memory = conv1_original_weights_memory;
    if (conv1_prim_desc.weights_desc() != conv1_original_weights_md) {
        //TODO if the layout is unexpected, load through dalotia permuted instead
        conv1_weights_memory = dnnl::memory(conv1_prim_desc.weights_desc(), eng);
        dnnl::reorder(conv1_original_weights_memory, conv1_weights_memory)
                .execute(s, conv1_original_weights_memory, conv1_weights_memory);
    }  
    auto conv1_dst_memory = dnnl::memory(conv1_prim_desc.dst_desc(), eng);

    net.push_back(dnnl::convolution_forward(conv1_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv1_src_memory},
            {DNNL_ARG_WEIGHTS, conv1_weights_memory},
            {DNNL_ARG_BIAS, conv1_original_bias_memory},
            {DNNL_ARG_DST, conv1_dst_memory}});

    const float negative_relu_slope = 0.0f;
    auto relu1_prim_desc = dnnl::eltwise_forward::primitive_desc(eng, dnnl::prop_kind::forward_inference,
            dnnl::algorithm::eltwise_relu, conv1_dst_memory.get_desc(), conv1_dst_memory.get_desc(), negative_relu_slope);
    net.push_back(dnnl::eltwise_forward(relu1_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv1_dst_memory},
            {DNNL_ARG_DST, conv1_dst_memory}});

    auto conv2_prim_desc = dnnl::convolution_forward::primitive_desc(eng,
            dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
            conv1_dst_memory.get_desc(), conv2_adapted_weights_md,
            conv2_adapted_bias_md, conv2_adapted_output_md, conv_strides, conv_no_to_padding,
            conv_no_to_padding);
    auto conv2_src_memory = conv1_dst_memory;
    if (conv2_prim_desc.src_desc() != conv1_dst_memory.get_desc()) {
        conv2_src_memory = dnnl::memory(conv2_prim_desc.src_desc(), eng);
        net.push_back(dnnl::reorder(conv1_dst_memory, conv2_src_memory));
        net_args.push_back({{DNNL_ARG_FROM, conv1_dst_memory},
                {DNNL_ARG_TO, conv2_src_memory}});
    }
    auto conv2_weights_memory = conv2_original_weights_memory;
    if (conv2_prim_desc.weights_desc() != conv2_original_weights_md) {
        conv2_weights_memory = dnnl::memory(conv2_prim_desc.weights_desc(), eng);
        dnnl::reorder(conv2_original_weights_memory, conv2_weights_memory)
                .execute(s, conv2_original_weights_memory, conv2_weights_memory);
    }
    auto conv2_dst_memory = dnnl::memory(conv2_prim_desc.dst_desc(), eng);
    net.push_back(dnnl::convolution_forward(conv2_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv2_src_memory},
            {DNNL_ARG_WEIGHTS, conv2_weights_memory},
            {DNNL_ARG_BIAS, conv2_original_bias_memory},
            {DNNL_ARG_DST, conv2_dst_memory}});

    auto relu2_prim_desc = dnnl::eltwise_forward::primitive_desc(eng, dnnl::prop_kind::forward_inference,
            dnnl::algorithm::eltwise_relu, conv2_dst_memory.get_desc(), conv2_dst_memory.get_desc(), negative_relu_slope);
    net.push_back(dnnl::eltwise_forward(relu2_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv2_dst_memory},
            {DNNL_ARG_DST, conv2_dst_memory}});

    auto conv3_prim_desc = dnnl::convolution_forward::primitive_desc(eng,
            dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
            conv2_dst_memory.get_desc(), conv3_adapted_weights_md,
            conv3_adapted_bias_md, conv3_adapted_output_md, conv_strides, conv_no_to_padding,
            conv_no_to_padding);
    auto conv3_src_memory = conv2_dst_memory;
    if (conv3_prim_desc.src_desc() != conv2_dst_memory.get_desc()) {
        conv3_src_memory = dnnl::memory(conv3_prim_desc.src_desc(), eng);
        net.push_back(dnnl::reorder(conv2_dst_memory, conv3_src_memory));
        net_args.push_back({{DNNL_ARG_FROM, conv2_dst_memory},
                {DNNL_ARG_TO, conv3_src_memory}});
    }
    auto conv3_weights_memory = conv3_original_weights_memory;
    if (conv3_prim_desc.weights_desc() != conv3_original_weights_md) {
        conv3_weights_memory = dnnl::memory(conv3_prim_desc.weights_desc(), eng);
        dnnl::reorder(conv3_original_weights_memory, conv3_weights_memory)
                .execute(s, conv3_original_weights_memory, conv3_weights_memory);
    }
    auto conv3_dst_memory = dnnl::memory(conv3_prim_desc.dst_desc(), eng);
    net.push_back(dnnl::convolution_forward(conv3_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv3_src_memory},
            {DNNL_ARG_WEIGHTS, conv3_weights_memory},
            {DNNL_ARG_BIAS, conv3_original_bias_memory},
            {DNNL_ARG_DST, conv3_dst_memory}});
    auto relu3_prim_desc = dnnl::eltwise_forward::primitive_desc(eng, dnnl::prop_kind::forward_inference,
            dnnl::algorithm::eltwise_relu, conv3_dst_memory.get_desc(), conv3_dst_memory.get_desc(), negative_relu_slope);
    net.push_back(dnnl::eltwise_forward(relu3_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv3_dst_memory},
            {DNNL_ARG_DST, conv3_dst_memory}});

    auto conv4_prim_desc = dnnl::convolution_forward::primitive_desc(eng,
            dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
            conv3_dst_memory.get_desc(), conv4_adapted_weights_md,
            conv4_adapted_bias_md, conv4_adapted_output_md, conv_strides, conv_no_to_padding,
            conv_no_to_padding);
    auto conv4_src_memory = conv3_dst_memory;
    if (conv4_prim_desc.src_desc() != conv3_dst_memory.get_desc()) {
        conv4_src_memory = dnnl::memory(conv4_prim_desc.src_desc(), eng);
        net.push_back(dnnl::reorder(conv3_dst_memory, conv4_src_memory));
        net_args.push_back({{DNNL_ARG_FROM, conv3_dst_memory},
                {DNNL_ARG_TO, conv4_src_memory}});
    }
    auto conv4_weights_memory = conv4_original_weights_memory;
    if (conv4_prim_desc.weights_desc() != conv4_original_weights_md) {
        conv4_weights_memory = dnnl::memory(conv4_prim_desc.weights_desc(), eng);
        dnnl::reorder(conv4_original_weights_memory, conv4_weights_memory)
                .execute(s, conv4_original_weights_memory, conv4_weights_memory);
    }
    auto conv4_dst_memory = dnnl::memory(conv4_prim_desc.dst_desc(), eng);
    net.push_back(dnnl::convolution_forward(conv4_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv4_src_memory},
            {DNNL_ARG_WEIGHTS, conv4_weights_memory},
            {DNNL_ARG_BIAS, conv4_original_bias_memory},
            {DNNL_ARG_DST, conv4_dst_memory}});

    auto sigmoid_desc = dnnl::eltwise_forward::primitive_desc(eng, dnnl::prop_kind::forward_inference,
            dnnl::algorithm::eltwise_logistic, conv4_dst_memory.get_desc(), conv4_dst_memory.get_desc(), 0.0f);
    net.push_back(dnnl::eltwise_forward(sigmoid_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv4_dst_memory},
            {DNNL_ARG_DST, conv4_dst_memory}});
    
    auto half_desc = dnnl::eltwise_forward::primitive_desc(eng, dnnl::prop_kind::forward_inference,
            dnnl::algorithm::eltwise_linear, conv4_dst_memory.get_desc(), conv4_dst_memory.get_desc(), 0.5f);
    net.push_back(dnnl::eltwise_forward(half_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv4_dst_memory},
            {DNNL_ARG_DST, conv4_dst_memory}});

    const auto start = std::chrono::high_resolution_clock::now();
    for (size_t r = 0; r < num_repetitions; ++r) {
        // execute net
        for (size_t i = 0; i < net.size(); ++i) {
            net.at(i).execute(s, net_args.at(i));
        }
        // copy back //TODO set output memory same as results
        auto* output_ptr = static_cast<float*>(conv4_dst_memory.get_data_handle());
        results.assign(output_ptr, output_ptr + results.size());
    }
    return std::chrono::high_resolution_clock::now() - start;
}

#endif // DALOTIA_E_WITH_ONEDNN

#ifdef DALOTIA_E_WITH_LIBTORCH
std::chrono::duration<double> run_inference_libtorch(
    const dalotia::vector<float> &inputs,
    const std::vector<int> &input_extents,
    size_t num_repetitions,
    dalotia::vector<float> &results) {

    torch::jit::script::Module module = torch::jit::load("traced_DeepRLEddyNet.pt");
    module = torch::jit::optimize_for_inference(module);

    // todo keep const semantics on input -> seems that's not a thing in libtorch
    const auto input_tensor = torch::from_blob(
            reinterpret_cast<void*>(const_cast<float*>(inputs.data())), 
            at::IntArrayRef({input_extents[0], input_extents[1], input_extents[2], input_extents[3], input_extents[4]})
        );
    const auto start = std::chrono::high_resolution_clock::now();
    for (size_t r = 0; r < num_repetitions; ++r) {
        auto output_tensor = module.forward({input_tensor}).toTensor();
        // assign to output
        std::memcpy(results.data(), output_tensor.data_ptr(), output_tensor.numel() * sizeof(float));
    }
    return std::chrono::high_resolution_clock::now() - start;
}
#endif  // DALOTIA_E_WITH_LIBTORCH

int main(int, char **) {
    // the data used here is generated with generate_models.py

    // unpermuted for now
    auto [input_extents, input_tensor] =dalotia::load_tensor_dense<float>("./input_DeepRLEddyNet.safetensors", "random_input",
                                          dalotia_WeightFormat::dalotia_float_32, dalotia_Ordering::dalotia_C_ordering);
    assert(input_extents == std::vector<int>({16*16*16, 3, 6, 6, 6}));
    assert_close(input_tensor[0], 4.9626e-01);
    assert_close(input_tensor[1], 7.6822e-01);
    assert_close(input_tensor[input_tensor.size() - 1], 7.8506e-01);
#ifdef DALOTIA_E_FOR_MEMORY_TRACE 
    auto output_extents = std::vector<int>(1, 16*16*16);
#else
    auto [output_extents, expected_output_tensor] =
        dalotia::load_tensor_dense<float>("./output_DeepRLEddyNet.safetensors", "output",
                                          dalotia_WeightFormat::dalotia_float_32, dalotia_Ordering::dalotia_C_ordering);
    assert(output_extents == std::vector<int>(1, 16*16*16));
    assert_close(expected_output_tensor[0], 0.4487);
    assert_close(expected_output_tensor[1], 0.4471);
    assert_close(expected_output_tensor[4095], 0.4541);
#endif // DALOTIA_E_FOR_MEMORY_TRACE

    typedef std::function<std::chrono::duration<double>(
        const dalotia::vector<float> &input_tensor,
        const std::vector<int> &input_extents,
        size_t num_repetitions,
        dalotia::vector<float> &results)>
        inference_function;
    std::unordered_map<std::string, inference_function> inference_functions;

    inference_functions["slow_loops"] = run_inference_slow_loops;

#ifdef DALOTIA_E_WITH_LIBTORCH
    inference_functions["libtorch"] = run_inference_libtorch;
#else
    std::cout << "libtorch not enabled" << std::endl;
#endif  // DALOTIA_E_WITH_LIBTORCH

#ifdef DALOTIA_E_WITH_ONEDNN
    inference_functions["onednn"] = run_inference_onednn;
#else
    std::cout << "onednn not enabled" << std::endl;
#endif  // DALOTIA_E_WITH_ONEDNN
#ifdef DALOTIA_E_FOR_MEMORY_TRACE // plus comment out above the parts I don't want to measure w/ pinatrace
    const size_t num_repetitions = 1;
#else
    const size_t num_repetitions = 1000;
#endif // DALOTIA_E_FOR_MEMORY_TRACE
    dalotia::vector<float> results(expected_output_tensor.size());
    for (const auto &inference_function_pair : inference_functions) {
        const auto& inference_function = inference_function_pair.second;
#ifndef DALOTIA_E_FOR_MEMORY_TRACE
        std::cout << "Running inference with " << inference_function_pair.first << std::endl;
        std::memset(results.data(), 0, results.size() * sizeof(int));
#endif // not DALOTIA_E_FOR_MEMORY_TRACE

        const auto duration = inference_function(
            input_tensor, input_extents, num_repetitions, results);
#ifndef DALOTIA_E_FOR_MEMORY_TRACE
        std::cout << "Duration: " << duration.count() << "s" << std::endl;
        std::cout << "On average: " << duration.count() / static_cast<float>(num_repetitions) << "s" << std::endl;
        // check correctness of the output
        for (size_t i = 0; i < results.size(); ++i) {
            if (! is_close(results[i], expected_output_tensor[i])) {
                std::cerr << "results[" << i << "] = " << results[i] << 
                             " != expected_output_tensor[" << i << "] = " << expected_output_tensor[i] << std::endl;
                throw std::runtime_error("results != expected_output_tensor");
            }
        }
#endif // not DALOTIA_E_FOR_MEMORY_TRACE
    }
#ifndef DALOTIA_E_FOR_MEMORY_TRACE
    std::cout << "All benched!" << std::endl;
#endif // not DALOTIA_E_FOR_MEMORY_TRACE

    return 0;
}

