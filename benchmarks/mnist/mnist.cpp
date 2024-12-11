#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>  // std::memset
#include <fstream>
#include <iostream>

#include "dalotia.hpp"
#include "dalotia_safetensors_file.hpp"
#ifdef DALOTIA_E_WITH_BOOST_MULTI
// #include "mdspan/mdspan.hpp"
#include <boost/multi/array.hpp>
#include <multi/adaptors/blas.hpp>
// #include <multi/adaptors/tblis.hpp>
#endif  // DALOTIA_E_WITH_BOOST_MULTI

#ifdef DALOTIA_E_WITH_LIBTORCH
#include <torch/script.h>
#endif  // DALOTIA_E_WITH_LIBTORCH

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
    return std::make_pair(tensor_weight_cpp, tensor_bias_cpp);
}

// cf. https://stackoverflow.com/a/10409376/7272382
int reverseEndianness(int i) {
    int c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
dalotia::vector<float> read_mnist_scaled(std::string full_path) {
    dalotia::vector<uint8_t> vector_of_images;
    dalotia::vector<float> vector_of_images_scaled;
    std::ifstream file(full_path, std::ios::binary);
    std::cout << "Reading " << full_path << std::endl;
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseEndianness(magic_number);
        if (full_path.find("images") != std::string::npos) {
            std::cout << "Reading images" << std::endl;
            int n_rows = 0;
            int n_cols = 0;
            assert(magic_number == 2051);
            file.read((char *)&number_of_images, sizeof(number_of_images));
            number_of_images = reverseEndianness(number_of_images);
            assert(number_of_images == 10000);
            file.read((char *)&n_rows, sizeof(n_rows));
            n_rows = reverseEndianness(n_rows);
            assert(n_rows == 28);
            file.read((char *)&n_cols, sizeof(n_cols));
            n_cols = reverseEndianness(n_cols);
            assert(n_cols == 28);
            vector_of_images.resize(number_of_images * n_rows * n_cols);
            file.read((char *)vector_of_images.data(), vector_of_images.size());
            std::transform(vector_of_images.begin(), vector_of_images.end(),
                           std::back_inserter(vector_of_images_scaled),
                           [](uint8_t x) { return x / 255.0; });
        } else {
            std::cout << "Reading labels" << std::endl;
            assert(magic_number == 2049);
            file.read((char *)&number_of_images, sizeof(number_of_images));
            number_of_images = reverseEndianness(number_of_images);
            assert(number_of_images == 10000);
            vector_of_images.resize(number_of_images);
            file.read((char *)vector_of_images.data(), vector_of_images.size());
            std::transform(vector_of_images.begin(), vector_of_images.end(),
                           std::back_inserter(vector_of_images_scaled),
                           [](uint8_t x) { return x; });
        }
        file.close();
    } else {
        throw std::runtime_error("Cannot open file `" + full_path + "`");
    }
    return vector_of_images_scaled;
}

template <int dim>
std::function<int(std::array<int, dim>)> get_tensor_indexer(
    const std::array<int, dim> &extents) {
    std::array<int, dim> strides;
    std::exclusive_scan(extents.rbegin(), extents.rend(), strides.rbegin(), 1,
                        std::multiplies<int>());
    const std::array<int, dim> const_strides = strides;
    return [const_strides, &extents](std::array<int, dim> indices) {
#ifndef NDEBUG
        for (int i = 0; i < dim; ++i) {
            if (indices[i] < 0 || indices[i] >= extents[i]) {
                std::cerr << "index " << i << " out of bounds: " << indices[i]
                          << " not in [0, " << extents[i] << ")" << std::endl;
            }
            assert(indices[i] >= 0 && indices[i] < extents[i]);
        }
#endif  // NDEBUG
        return std::inner_product(indices.begin(), indices.end(),
                                  const_strides.begin(), 0);
    };
}

template <typename ExtentsType>
void print_first_formatted(const dalotia::vector<float> &output,
                           const ExtentsType &extents, std::ostream &os) {
    for (int i = 0; i < 1; ++i) {  // extents[0]
        os << "output for image " << i << ": " << std::endl;
        for (int j = 0; j < extents[1]; ++j) {
            for (int k = 0; k < extents[2]; ++k) {
                os << output[0 + j * extents[2] + k] << ", ";
            }
            os << std::endl;
        }
        os << std::endl;
    }
}

void assert_close(float a, float b, float tol = 1e-4) {
#ifndef NDEBUG
    if (std::abs(a - b) >= tol) {
        std::cerr << "assertion failed: " << a << " != " << b << std::endl;
    }
    assert(std::abs(a - b) < tol);
#endif  // NDEBUG
}

std::chrono::duration<double> run_inference_slow_loops(
    const dalotia::vector<float> &conv1_weight,
    const dalotia::vector<float> &conv1_bias,
    const std::array<int, 4> &conv1_weight_extents,
    const dalotia::vector<float> &conv2_weight,
    const dalotia::vector<float> &conv2_bias,
    const std::array<int, 4> &conv2_weight_extents,
    const dalotia::vector<float> &fc1_weight,
    const dalotia::vector<float> &fc1_bias,
    const std::array<int, 2> &fc1_weight_extents,
    const dalotia::vector<float> &images, const dalotia::vector<float> &labels,
    dalotia::vector<int> &results) {
    const auto conv1_weight_indexer =
        get_tensor_indexer<4>(conv1_weight_extents);
    const auto conv2_weight_indexer =
        get_tensor_indexer<4>(conv2_weight_extents);
    const auto fc1_weight_indexer = get_tensor_indexer<2>(fc1_weight_extents);

    constexpr size_t batch_size = 64;

    auto image_vector_padded = dalotia::vector<float>(batch_size * 30 * 30);
    const std::array<int, 3> image_padded_extents = {batch_size, 30, 30};
    const auto image_padded_indexer =
        get_tensor_indexer<3>(image_padded_extents);
    auto conv1_output = dalotia::vector<float>(batch_size * 8 * 28 * 28);
    const std::array<int, 4> conv1_output_extents = {batch_size, 8, 28, 28};
    const auto conv1_output_indexer =
        get_tensor_indexer<4>(conv1_output_extents);
    auto conv1_padded_output_pooled =
        dalotia::vector<float>(batch_size * 8 * 16 * 16);
    const std::array<int, 4> conv1_padded_output_pooled_extents = {batch_size,
                                                                   8, 16, 16};
    const auto conv1_padded_output_pooled_indexer =
        get_tensor_indexer<4>(conv1_padded_output_pooled_extents);
    auto conv2_output = dalotia::vector<float>(batch_size * 16 * 14 * 14);
    const std::array<int, 4> conv2_output_extents = {batch_size, 16, 14, 14};
    const auto conv2_output_indexer =
        get_tensor_indexer<4>(conv2_output_extents);
    auto conv2_output_pooled = dalotia::vector<float>(batch_size * 16 * 7 * 7);
    const std::array<int, 4> conv2_output_pooled_extents = {batch_size, 16, 7,
                                                            7};
    const std::array<int, 2> conv2_output_flat_extents = {batch_size,
                                                          16 * 7 * 7};
    const auto conv2_output_flat_indexer =
        get_tensor_indexer<2>(conv2_output_flat_extents);

    auto fc1_output = dalotia::vector<float>(batch_size * 10);
    const std::array<int, 2> fc1_output_extents = {batch_size, 10};
    const auto fc1_output_indexer = get_tensor_indexer<2>(fc1_output_extents);

    // minibatching
    auto total_num_images = labels.size();
    auto num_batches = static_cast<int>(
        std::ceil(total_num_images / static_cast<float>(batch_size)));

    const auto start = std::chrono::high_resolution_clock::now();
    for (size_t batch_index = 0; batch_index < num_batches; ++batch_index) {
        auto num_images_in_batch =
            std::min(batch_size, total_num_images - batch_index * batch_size);
        auto inum_images_in_batch = static_cast<int>(num_images_in_batch);
        assert(inum_images_in_batch > 0);
        std::cout << "batch index: " << batch_index << " / " << num_batches
                  << " num images in batch: " << num_images_in_batch
                  << std::endl;
        // zero all vector data structures
        // std::memset(
        //     image_vector_padded.data(), 0,
        //     image_vector_padded.size() * sizeof(float));
        std::memset(conv1_output.data(), 0,
                    conv1_output.size() * sizeof(float));
        std::memset(conv1_padded_output_pooled.data(), 0,
                    conv1_padded_output_pooled.size() * sizeof(float));
        std::memset(conv2_output.data(), 0,
                    conv2_output.size() * sizeof(float));
        std::memset(conv2_output_pooled.data(), 0,
                    conv2_output_pooled.size() * sizeof(float));
        std::memset(fc1_output.data(), 0, fc1_output.size() * sizeof(float));

        // apply first convolution
        // copy data to larger array for zero-padding at the edges
        for (int o = 0; o < inum_images_in_batch; ++o) {
            for (int i = 0; i < 28; ++i) {
                for (int j = 0; j < 28; ++j) {
                    image_vector_padded[image_padded_indexer(
                        {o, i + 1, j + 1})] =
                        images[batch_index * (batch_size * 28 * 28) +
                               o * (28 * 28) + i * 28 + j];
                }
            }
        }
        for (int o = 0; o < inum_images_in_batch; ++o) {
            for (int k = 0; k < 8; ++k) {
                for (int i = 1; i < 29; ++i) {
                    for (int j = 1; j < 29; ++j) {
                        float value = 0;
                        for (int m = 0; m < 3; ++m) {
                            for (int n = 0; n < 3; ++n) {
                                value +=
                                    conv1_weight[conv1_weight_indexer(
                                        {k, 0, m, n})] *
                                    image_vector_padded[image_padded_indexer(
                                        {o, i + m - 1, j + n - 1})];
                            }
                        }
                        value += conv1_bias[k];
                        if (value < 0.) {
                            value = 0.;
                        }
                        conv1_output[conv1_output_indexer(
                            {o, k, i - 1, j - 1})] = value;
                    }
                }
            }
        }

        // apply max pooling
        // use larger array for zero-padding at the edges
        for (int o = 0; o < inum_images_in_batch; ++o) {
            for (int k = 0; k < 8; ++k) {
                for (int i = 0; i < 14; ++i) {
                    for (int j = 0; j < 14; ++j) {
                        float max_val = 0;
                        for (int m = 0; m < 2; ++m) {
                            for (int n = 0; n < 2; ++n) {
                                max_val = std::max(
                                    max_val,
                                    conv1_output[conv1_output_indexer(
                                        {o, k, 2 * i + m, 2 * j + n})]);
                            }
                        }
                        conv1_padded_output_pooled
                            [conv1_padded_output_pooled_indexer(
                                {o, k, i + 1, j + 1})] = max_val;
                    }
                }
            }
        }

        // print_first_formatted(conv1_padded_output_pooled,
        // conv1_padded_output_pooled_extents,
        //                       std::cout);

        // apply second convolution
        for (int o = 0; o < inum_images_in_batch; ++o) {
            for (int k = 0; k < 16; ++k) {
                for (int i = 1; i < 15; ++i) {
                    for (int j = 1; j < 15; ++j) {
                        float value = 0;
                        for (int l = 0; l < 8; ++l) {
                            for (int m = 0; m < 3; ++m) {
                                for (int n = 0; n < 3; ++n) {
                                    value +=
                                        conv2_weight[conv2_weight_indexer(
                                            {k, l, m, n})] *
                                        conv1_padded_output_pooled
                                            [conv1_padded_output_pooled_indexer(
                                                {o, l, i + m - 1, j + n - 1})];
                                }
                            }
                        }
                        value += conv2_bias[k];
                        if (value < 0.) {
                            value = 0.;
                        }
                        conv2_output[conv2_output_indexer(
                            {o, k, i - 1, j - 1})] = value;
                    }
                }
            }
        }

        // apply max pooling
        auto conv2_output_pooled_indexer =
            get_tensor_indexer<4>(conv2_output_pooled_extents);
        for (int o = 0; o < inum_images_in_batch; ++o) {
            for (int i = 0; i < 7; ++i) {
                for (int j = 0; j < 7; ++j) {
                    for (int k = 0; k < 16; ++k) {
                        float max_val = 0;
                        for (int m = 0; m < 2; ++m) {
                            for (int n = 0; n < 2; ++n) {
                                max_val = std::max(
                                    max_val,
                                    conv2_output[conv2_output_indexer(
                                        {o, k, 2 * i + m, 2 * j + n})]);
                            }
                        }
                        conv2_output_pooled[conv2_output_pooled_indexer(
                            {o, k, i, j})] = max_val;
                    }
                }
            }
        }

        // apply dense layer
        for (int o = 0; o < inum_images_in_batch; ++o) {
            for (int k = 0; k < 10; ++k) {
                float value = 0.;
                for (int l = 0; l < 16 * 7 * 7; ++l) {
                    value +=
                        fc1_weight[fc1_weight_indexer({k, l})] *
                        conv2_output_pooled[conv2_output_flat_indexer({o, l})];
                }
                value += fc1_bias[k];
                fc1_output[fc1_output_indexer({o, k})] = value;
            }
        }

        // apply softmax and compare to labels
        for (int o = 0; o < inum_images_in_batch; ++o) {
            auto result = std::max_element(fc1_output.begin() + o * 10,
                                           fc1_output.begin() + (o + 1) * 10) -
                          (fc1_output.begin() + o * 10);
            results[batch_index * batch_size + o] = result;
        }

#ifndef NDEBUG
        if (batch_index == 0) {  // compare to Python results
            assert_close(conv1_output[0], 0.1796);
            assert_close(conv1_output[conv1_output_indexer({0, 7, 27, 27})],
                         0.6550);
            assert_close(
                conv1_padded_output_pooled[conv1_padded_output_pooled_indexer(
                    {0, 0, 1, 1})],
                0.1796);
            assert_close(
                conv1_padded_output_pooled[conv1_padded_output_pooled_indexer(
                    {0, 7, 14, 14})],
                0.6550);
            assert_close(conv2_output[0], 0.4063);
            assert_close(conv2_output[conv2_output_indexer({0, 1, 13, 13})],
                         0.1789);
            assert_close(conv2_output[conv2_output_indexer({0, 13, 0, 0})],
                         0.1906);
            assert_close(conv2_output[conv2_output_indexer({0, 15, 13, 13})],
                         0.0);
            assert_close(conv2_output_pooled[0], 0.4063);
            assert_close(
                conv2_output_pooled[conv2_output_pooled_indexer({0, 1, 0, 0})],
                0.04935);
            assert_close(
                conv2_output_pooled[conv2_output_pooled_indexer({0, 1, 6, 6})],
                0.49008);
            assert_close(
                conv2_output_pooled[conv2_output_pooled_indexer({0, 13, 0, 0})],
                0.1906);
            assert_close(
                conv2_output_pooled[conv2_output_pooled_indexer({0, 15, 6, 6})],
                0.0);

            assert_close(conv2_output_pooled[0], 0.40625, 1e-5);
            assert_close(conv2_output_pooled[1], 0.0);
            assert_close(conv2_output_pooled[8], 6.2347);
            assert_close(conv2_output_pooled[783], 0.0);
            assert_close(fc1_output[0], -80.9247);
            assert_close(fc1_output[1], -34.6855);
            assert_close(fc1_output[2], -31.1533);
            assert_close(fc1_output[3], -12.5210);
            assert_close(fc1_output[4], -44.1289);
            assert_close(fc1_output[5], -40.3522);
            assert_close(fc1_output[6], -139.7097);
            assert_close(fc1_output[7], 38.1572);
            assert_close(fc1_output[8], -47.0220);
            assert_close(fc1_output[9], -7.9544);
        }
#endif  // NDEBUG
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> diff = end - start;

    return diff;
}

#ifdef DALOTIA_E_WITH_BOOST_MULTI
namespace multi = boost::multi;
// namespace tblis = multi::tblis;
// using namespace multi::tblis;
// using namespace multi::operators; // not yet found
// output helper function for debugging
template <long int N>
std::ostream &operator<<(
    std::ostream &os,
    multi::subarray<float, N, float *,
                    boost::multi::layout_t<N, long int>> const &arrayref) {
    os << "subarray of " << N << "D array" << std::endl;
    if constexpr (N == 2) {
        auto [ni, nj] = arrayref.sizes();
        std::cout << "sizes: " << ni << " " << nj << std::endl;
        auto [is, js] = arrayref.extensions();
        for (const auto &i : is) {
            for (const auto &j : js) {
                os << arrayref[i][j] << ' ';
            }
            os << std::endl;
        }
    } else if constexpr (N == 3) {
        auto [ni, nj, nk] = arrayref.sizes();
        std::cout << "sizes: " << ni << " " << nj << " " << nk << std::endl;
        auto [is, js, ks] = arrayref.extensions();
        // arrayref.extensions().template get<0>()
        // arrayref.extensions()[0]
        // arrayref.sizes()[1]
        for (const auto &i : is) {
            for (const auto &j : js) {
                for (const auto &k : ks) {
                    os << arrayref[i][j][k] << ' ';
                }
                os << std::endl;
            }
            os << std::endl;
        }
    } else {
        throw std::runtime_error("only < ?D arrays supported");
    }
    os << std::endl;
    return os;
}

void run_inference_boost_multi(std::string filename) {
    using span_4d_float = multi::array_ref<float, 4>;
    using span_3d_float = multi::array_ref<float, 3>;
    using span_2d_float = multi::array_ref<float, 2>;

    auto [conv1_weight, conv1_bias] =
        test_load(filename, "conv1");  // TODO why can't I make them const?
    const auto conv1_weight_span =
        span_4d_float({8, 1, 3, 3}, conv1_weight.data());
    const auto conv1_bias_span = span_2d_float({8, 1}, conv1_bias.data());
    assert(conv1_weight_span.sizes().get<1>() == 1);  // 1 input channel

    auto [conv2_weight, conv2_bias] = test_load(filename, "conv2");
    const auto conv2_weight_span =
        span_4d_float({16, 8, 3, 3}, conv1_weight.data());
    const auto conv2_bias_span = span_2d_float({16, 1}, conv2_bias.data());

    auto [fc1_weight, fc1_bias] = test_load(filename, "fc1");
    const auto fc1_weight_span = span_2d_float({10, 784}, fc1_weight.data());
    const auto fc1_bias_span = span_2d_float({10, 1}, fc1_bias.data());

    // load the mnist test data // as in
    // https://medium.com/@myringoleMLGOD/simple-convolutional-neural-network-cnn-for-dummies-in-pytorch-a-step-by-step-guide-6f4109f6df80
    // too
    std::string mnist_test_images_filename = "t10k-images-idx3-ubyte";
    std::string mnist_test_labels_filename = "t10k-labels-idx3-ubyte";

    auto images = read_mnist_scaled(mnist_test_images_filename);
    // auto labels = read_mnist(mnist_test_labels_filename);
    auto total_num_images = images.size() / (28 * 28);

    // minibatching
    constexpr size_t batch_size = 64;
    auto num_batches = static_cast<int>(
        std::ceil(total_num_images / static_cast<float>(batch_size)));
    for (size_t batch_index = 0; batch_index < 1; ++batch_index) {
        auto num_images_in_batch =
            std::min(batch_size, total_num_images - batch_index * batch_size);
        auto inum_images_in_batch = static_cast<int>(num_images_in_batch);
        std::cout << "batch index: " << batch_index << " / " << num_batches
                  << " num images in batch: " << num_images_in_batch
                  << std::endl;

        // apply first convolution
        // copy data to larger array for zero-padding at the edges
        auto image_vector_padded =
            dalotia::vector<float>(num_images_in_batch * 30 * 30);
        auto image_padded_span = span_3d_float({inum_images_in_batch, 30, 30},
                                               image_vector_padded.data());

        std::cout << "image_padded "
                  << image_padded_span(
                         0, multi::_,
                         multi::_)  // <- TODO why does this segfault on fugaku?
                  << std::endl;

        image_padded_span(multi::_, {1, 29}, {1, 29}) =
            span_3d_float({inum_images_in_batch, 28, 28},
                          images.data() + batch_index * (batch_size * 28 * 28));

        std::cout << "image_padded " << image_padded_span(0, multi::_, multi::_)
                  << std::endl;

        auto conv1_output =
            dalotia::vector<float>(num_images_in_batch * 8 * 28 * 28);
        auto conv1_output_span = span_4d_float(
            {inum_images_in_batch, 8, 28, 28}, conv1_output.data());
#pragma omp parallel for
        for (int o = 0; o < image_padded_span.sizes().get<0>(); ++o) {
            for (int k = 0; k < conv1_weight_span.sizes().get<0>(); ++k) {
                for (int i = 1; i < image_padded_span.sizes().get<1>() - 1;
                     ++i) {
                    for (int j = 1; j < image_padded_span.sizes().get<2>() - 1;
                         ++j) {
                        // sum_m_n(conv1_weight_span[k, 0, m, n] *
                        // image_padded_span[i + m -1, j + n -1] ) + bias[k]
                        // (=> 10 terms per o and k)
                        conv1_output_span(o, k, i - 1, j - 1) =
                            conv1_weight_span(k, 0, 0, 0) *
                                image_padded_span(o, i - 1, j - 1) +
                            conv1_weight_span(k, 0, 0, 1) *
                                image_padded_span(o, i - 1, j + 0) +
                            conv1_weight_span(k, 0, 0, 2) *
                                image_padded_span(o, i - 1, j + 1) +
                            conv1_weight_span(k, 0, 1, 0) *
                                image_padded_span(o, i + 0, j - 1) +
                            conv1_weight_span(k, 0, 1, 1) *
                                image_padded_span(o, i + 0, j + 0) +
                            conv1_weight_span(k, 0, 1, 2) *
                                image_padded_span(o, i + 0, j + 1) +
                            conv1_weight_span(k, 0, 2, 0) *
                                image_padded_span(o, i + 1, j - 1) +
                            conv1_weight_span(k, 0, 2, 1) *
                                image_padded_span(o, i + 1, j + 0) +
                            conv1_weight_span(k, 0, 2, 2) *
                                image_padded_span(o, i + 1, j + 1) +
                            conv1_bias_span(k, 0);
                        // apply first activation function (relu)
                        if (conv1_output_span(o, k, i - 1, j - 1) < 0.) {
                            conv1_output_span(o, k, i - 1, j - 1) = 0.;
                        }
                    }
                }
            }
        }

        // {
        //     using namespace tblis::indices;
        //     tblis::mult(conv1_weight_span(k, 0, m, n), image_padded_span(o, i
        //     + m, j + n),
        //                 conv1_output_span(o, k, i + m, j + n));
        // }

        // apply max pooling
        dalotia::vector<float> conv1_output_pooled(num_images_in_batch * 8 *
                                                   14 * 14);
        auto conv1_output_pooled_span = span_4d_float(
            {inum_images_in_batch, 8, 14, 14}, conv1_output_pooled.data());
#pragma omp parallel for
        for (int o = 0; o < num_images_in_batch; ++o) {
            for (int k = 0; k < 8; ++k) {
                for (int i = 0; i < 14; ++i) {
                    for (int j = 0; j < 14; ++j) {
                        auto window = conv1_output_span(
                            o, k, {2 * i, 2 * i + 1}, {2 * j, 2 * j + 1});
                        auto max_val = (*std::max_element(window.begin(),
                                                          window.end()))[0];
                        conv1_output_pooled_span(o, k, i, j) = max_val;
                    }
                }
            }
        }

        // apply second convolution
        // copy data to larger array for zero-padding at the edges
        auto feature_vector_padded =
            dalotia::vector<float>(num_images_in_batch * 8 * 16 * 16);
        auto feature_padded_span = span_4d_float(
            {inum_images_in_batch, 8, 16, 16}, feature_vector_padded.data());
        feature_padded_span(multi::_, multi::_, {1, 15}, {1, 15}) =
            conv1_output_pooled_span;
        const auto &c_feature_padded_span = feature_padded_span;

        auto conv2_output =
            dalotia::vector<float>(num_images_in_batch * 16 * 14 * 14);
        auto conv2_output_span = span_4d_float(
            {inum_images_in_batch, 16, 14, 14}, conv2_output.data());

        // assert that the sizes match
        assert(conv2_weight_span.sizes().get<0>() ==
               conv2_output_span.sizes().get<1>());  // 16 output channels
        assert(conv2_bias_span.sizes().get<0>() ==
               conv2_weight_span.sizes().get<0>());
        assert(conv2_weight_span.sizes().get<1>() ==
               c_feature_padded_span.sizes().get<1>());  // 8 input channels
        assert(conv2_output_span.sizes().get<2>() ==
               c_feature_padded_span.sizes().get<2>() - 2);
        assert(conv2_output_span.sizes().get<3>() ==
               c_feature_padded_span.sizes().get<3>() - 2);
        assert(c_feature_padded_span.sizes().get<0>() == inum_images_in_batch);
        assert(conv2_output_span.sizes().get<0>() == inum_images_in_batch);
#pragma omp parallel for
        for (int o = 0; o < c_feature_padded_span.sizes().get<0>(); ++o) {
            for (int k = 0; k < conv2_weight_span.sizes().get<0>(); ++k) {
                for (int i = 1; i < c_feature_padded_span.sizes().get<2>() - 1;
                     ++i) {
                    for (int j = 1;
                         j < c_feature_padded_span.sizes().get<3>() - 1; ++j) {
                        float value = 0;
                        for (int l = 0; l < conv2_weight_span.sizes().get<1>();
                             ++l) {
                            value +=
                                conv2_weight_span(l, k, 0, 0) *
                                    c_feature_padded_span(o, l, i - 1, j - 1) +
                                conv2_weight_span(l, k, 0, 1) *
                                    c_feature_padded_span(o, l, i - 1, j + 0) +
                                conv2_weight_span(l, k, 0, 2) *
                                    c_feature_padded_span(o, l, i - 1, j + 1) +
                                conv2_weight_span(l, k, 1, 0) *
                                    c_feature_padded_span(o, l, i + 0, j - 1) +
                                conv2_weight_span(l, k, 1, 1) *
                                    c_feature_padded_span(o, l, i + 0, j + 0) +
                                conv2_weight_span(l, k, 1, 2) *
                                    c_feature_padded_span(o, l, i + 0, j + 1) +
                                conv2_weight_span(l, k, 2, 0) *
                                    c_feature_padded_span(o, l, i + 1, j - 1) +
                                conv2_weight_span(l, k, 2, 1) *
                                    c_feature_padded_span(o, l, i + 1, j + 0) +
                                conv2_weight_span(l, k, 2, 2) *
                                    c_feature_padded_span(o, l, i + 1, j + 1) +
                                conv2_bias_span(l, 0);
                        }
                        // apply activation function (relu)
                        if (value < 0.) {
                            value = 0.;
                        }
                        conv2_output_span(o, k, i - 1, j - 1) = value;
                    }
                }
            }
        }

        std::cout << conv2_output_span(0, multi::_, multi::_, multi::_)
                  << std::endl;

        if (batch_index == 0) {  // compare to python result
            assert(conv2_output_span[0][0][0][0] < 0.4063);
            assert(conv2_output_span[0][0][0][0] > 0.4062);
        }

        // apply max pooling
        dalotia::vector<float> conv2_output_pooled(num_images_in_batch * 16 *
                                                   7 * 7);
        auto conv2_output_pooled_span = span_4d_float(
            {inum_images_in_batch, 16, 7, 7}, conv2_output_pooled.data());
#pragma omp parallel for
        for (int o = 0; o < num_images_in_batch; ++o) {
            for (int i = 0; i < 7; ++i) {
                for (int j = 0; j < 7; ++j) {
                    for (int k = 0; k < 16; ++k) {
                        auto window = conv2_output_span(
                            o, k, {2 * i, 2 * i + 1}, {2 * j, 2 * j + 1});
                        auto max_val = (*std::max_element(window.begin(),
                                                          window.end()))[0];
                        conv2_output_pooled_span(o, k, i, j) = max_val;
                    }
                }
            }
        }

        // apply dense layer
        dalotia::vector<float> fc1_output(num_images_in_batch * 10);
        auto fc1_output_span =
            span_2d_float({inum_images_in_batch, 10}, fc1_output.data());
        auto conv2_output_flattened = span_2d_float(
            {inum_images_in_batch, 16 * 7 * 7}, conv2_output_pooled.data());
        // fc1_output_span = multi::blas::gemm(1., conv2_output_flattened,
        // //TODO use one of them!
        //                                     fc1_weight_span.transposed());
        // using multi::operator+=; // doesn't work yet? ->
        // https://github.com/correaa/boost-multi/blob/master/include/boost/multi/adaptors/blas/README.md
        // footnote 3
        // std::transform(fc1_bias_span.begin(), fc1_bias_span.end(),
        //                // appears to not work
        //                fc1_output_span.begin(), fc1_output_span.begin(),
        //                [](auto ex, auto ey) {
        //                    return ex[0] + ey[0];
        //                });  // this would also be nicer without the [0]
        //                indexing

        // {
        //     using namespace tblis::indices;
        //     tblis::mult(fc1_weight_span(a, b), conv2_output_flattened(o,
        //     b),
        //                 fc1_output(o, a));
        // }

        std::transform(fc1_bias.begin(), fc1_bias.end(), fc1_output.begin(),
                       fc1_output.begin(),
                       [](auto ex, auto ey) { return ex + ey; });

        // output first image's result
        std::cout << "output for first image: ";
        for (int i = 0; i < 10; ++i) {
            std::cout << fc1_output_span[i][0] << " ";
        }
    }
}
#endif  // DALOTIA_E_WITH_BOOST_MULTI

#ifdef DALOTIA_E_WITH_LIBTORCH
std::chrono::duration<double> run_inference_libtorch(
    const dalotia::vector<float> &conv1_weight,
    const dalotia::vector<float> &conv1_bias,
    const std::array<int, 4> &conv1_weight_extents,
    const dalotia::vector<float> &conv2_weight,
    const dalotia::vector<float> &conv2_bias,
    const std::array<int, 4> &conv2_weight_extents,
    const dalotia::vector<float> &fc1_weight,
    const dalotia::vector<float> &fc1_bias,
    const std::array<int, 2> &fc1_weight_extents,
    const dalotia::vector<float> &images, const dalotia::vector<float> &labels,
    dalotia::vector<int> &results) {
    constexpr size_t batch_size = 64;

    torch::jit::script::Module module = torch::jit::load("mnist_fugaku.pt");
    // https://pytorch.org/cppdocs/api/function_namespacetorch_1_1jit_1a68b22ff064eaee414e83674ce412130f.html
    // Module torch::jit::optimize_for_inference(Module &module, const std::vector<std::string> &other_methods = {})

    // minibatching
    const auto total_num_images = labels.size();
    const auto num_batches = static_cast<int>(
        std::ceil(total_num_images / static_cast<float>(batch_size)));

    const auto start = std::chrono::high_resolution_clock::now();
    for (size_t batch_index = 0; batch_index < num_batches; ++batch_index) {
        auto num_images_in_batch =
            std::min(batch_size, total_num_images - batch_index * batch_size);
        auto inum_images_in_batch = static_cast<int>(num_images_in_batch);
        assert(inum_images_in_batch > 0);
        std::cout << "batch index: " << batch_index << " / " << num_batches
                  << " num images in batch: " << num_images_in_batch
                  << std::endl;
        auto input_tensor = torch::from_blob(
            reinterpret_cast<void*>(const_cast<float*>(images.data()) + (batch_index * batch_size * 28 * 28)), {64, 1, 28, 28});
        torch::Tensor out_tensor = module.forward({input_tensor}).toTensor();
    }
}
#endif  // DALOTIA_E_WITH_LIBTORCH

int main(int, char **) {
    // the model used here is generated as in
    // https://medium.com/@myringoleMLGOD/simple-convolutional-neural-network-cnn-for-dummies-in-pytorch-a-step-by-step-guide-6f4109f6df80
    // (but trained for 100 epochs)
    // and then saved with safetensors.torch.save_model(model,
    // "model.safetensors")
    std::string filename = "./model-mnist.safetensors";

    const auto [conv1_weight, conv1_bias] = test_load(filename, "conv1");
    const std::array<int, 4> conv1_weight_extents = {8, 1, 3, 3};
    const auto [conv2_weight, conv2_bias] = test_load(filename, "conv2");
    const std::array<int, 4> conv2_weight_extents = {16, 8, 3, 3};
    const auto [fc1_weight, fc1_bias] = test_load(filename, "fc1");
    const std::array<int, 2> fc1_weight_extents = {10, 784};

    // load the mnist test data // as in
    // https://medium.com/@myringoleMLGOD/simple-convolutional-neural-network-cnn-for-dummies-in-pytorch-a-step-by-step-guide-6f4109f6df80
    // too
    const std::string mnist_test_images_filename = "./t10k-images-idx3-ubyte";
    const std::string mnist_test_labels_filename = "./t10k-labels-idx1-ubyte";

    const auto images = read_mnist_scaled(mnist_test_images_filename);
    // assert on the first non-0 value
    assert_close(images[7 * 28 + 6], 0.3294);
    const auto labels = read_mnist_scaled(mnist_test_labels_filename);
    const auto total_num_images = images.size() / (28 * 28);
    assert(total_num_images == labels.size());

    dalotia::vector<int> results(total_num_images);
    typedef std::function<std::chrono::duration<double>(
        const dalotia::vector<float> &conv1_weight,
        const dalotia::vector<float> &conv1_bias,
        const std::array<int, 4> &conv1_weight_extents,
        const dalotia::vector<float> &conv2_weight,
        const dalotia::vector<float> &conv2_bias,
        const std::array<int, 4> &conv2_weight_extents,
        const dalotia::vector<float> &fc1_weight,
        const dalotia::vector<float> &fc1_bias,
        const std::array<int, 2> &fc1_weight_extents,
        const dalotia::vector<float> &images,
        const dalotia::vector<float> &labels, dalotia::vector<int> &results)>
        inference_function;
    std::vector<inference_function> inference_functions = {
        run_inference_slow_loops};

#ifdef DALOTIA_E_WITH_BOOST_MULTI
    inference_functions.push_back(run_inference_boost_multi);
#else
    std::cout << "BOOST_MULTI not enabled" << std::endl;
#endif  // DALOTIA_E_WITH_BOOST_MULTI
#ifdef DALOTIA_E_WITH_NDIRECT
    inference_functions.push_back(run_inference_ndirect);
#else
    std::cout << "NDIRECT not enabled" << std::endl;
#endif  // DALOTIA_E_WITH_NDIRECT
#ifdef DALOTIA_E_WITH_LIBTORCH
    inference_functions.push_back(run_inference_libtorch);
#else
    std::cout << "LIBTORCH not enabled" << std::endl;
#endif  // DALOTIA_E_WITH_LIBTORCH

    std::reverse(inference_functions.begin(), inference_functions.end());

    for (const auto &inference_function : inference_functions) {
        std::memset(results.data(), 0, results.size() * sizeof(int));

        const auto duration = inference_function(
            conv1_weight, conv1_bias, conv1_weight_extents, conv2_weight,
            conv2_bias, conv2_weight_extents, fc1_weight, fc1_bias,
            fc1_weight_extents, images, labels, results);
        int num_correct = 0;
        int num_incorrect = 0;
        for (int o = 0; o < total_num_images; ++o) {
            if (results[o] == labels[o]) {
                num_correct++;
            } else {
                num_incorrect++;
            }
        }
        assert(num_correct + num_incorrect == total_num_images);
        assert(num_correct == 9862);
        std::cout << "Got " << num_correct << "/" << total_num_images
                  << std::endl;
        std::cout << "Duration: " << duration.count() << "s" << std::endl;
    }

    return 0;
}