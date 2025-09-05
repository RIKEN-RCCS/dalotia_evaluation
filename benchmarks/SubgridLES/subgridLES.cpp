#include <cassert>
#include <chrono>
#include <cstring>  // std::memcpy
#include <iostream>

#include "dalotia.hpp"
#include "dalotia_safetensors_file.hpp"
#include "cacheflush.h"

#ifdef DALOTIA_E_WITH_LIBTORCH
#include <torch/script.h> // this exec is only built with libtorch
#else
#include "cblas.h"
#endif

#ifdef LIKWID_PERFMON
#include <likwid-marker.h>
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_CLOSE
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#endif  // LIKWID_PERFMON

#include <omp.h>
class OpenMPPrivateMemoryResource : public std::pmr::memory_resource
{
public:
    explicit OpenMPPrivateMemoryResource(int NUMA_node = -1) noexcept : NUMA_node_(NUMA_node) {}
    // (NUMA_node == -1) -> allocate on the NUMA node of the calling thread

private:
    omp_allocator_handle_t make_allocator(int NUMA_node, std::size_t alignment) const noexcept{
        omp_memspace_handle_t ms = omp_low_lat_mem_space; // or omp_default_mem_space
        omp_alloctrait_t traits[] {
            {omp_atk_sync_hint, omp_atv_private},
            {omp_atk_access, omp_atv_thread},
            {omp_atk_partition, omp_atv_nearest},
            {omp_atk_pinned, omp_atv_true},
            {omp_atk_alignment, alignment},
        }; // cf. https://www.openmp.org/spec-html/5.0/openmpsu53.html
        return omp_init_allocator(ms, 2, traits);
    }

    void* do_allocate(std::size_t bytes, std::size_t alignment = 16) override {
        if (bytes == 0) return nullptr;
        allocator_ = make_allocator(NUMA_node_, alignment);
        void* p = omp_alloc(bytes, allocator_);
        if (!p) throw std::bad_alloc{};
        return p;
    }

    void do_deallocate(void* p, std::size_t /*bytes*/, std::size_t /*alignment*/) noexcept override {
        omp_free(p, allocator_);
        omp_destroy_allocator(allocator_);
    }

    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override{
        return dynamic_cast<const OpenMPPrivateMemoryResource*>(&other) != nullptr;
    }

    int NUMA_node_;
    omp_allocator_handle_t allocator_ = omp_null_allocator;
};

void assert_close(float a, float b, float tol = 1e-4) {
#ifndef NDEBUG
    if (std::abs(a - b) >= tol) {
        std::cerr << "assertion failed: " << a << " != " << b << std::endl;
    }
    assert(std::abs(a - b) < tol);
#endif  // NDEBUG
}
#ifdef DALOTIA_E_WITH_LIBTORCH
std::chrono::duration<double> run_inference_libtorch(
    const std::vector<dalotia::vector<float>> &input_tensors,
    const std::vector<int> &input_sizes,
    size_t num_repetitions,
    const std::vector<int> &output_sizes,
    std::vector<dalotia::vector<float>> &result_tensors) {
    torch::jit::script::Module module = torch::jit::load("traced_SubgridLESNet.pt");
    module = torch::jit::optimize_for_inference(module);

    LIKWID_MARKER_START("libtorch");
    const auto start = std::chrono::high_resolution_clock::now();
    // If I can't figure out how to pass the output address to forward(), then most other people also won't 
    // auto output_tensor = torch::from_blob(
    //         reinterpret_cast<void*>(results.data()), 
    //         at::IntArrayRef({output_sizes[0], output_sizes[1]})
    //     );
    for (size_t r = 0; r < num_repetitions; ++r) {
        auto& inputs = input_tensors[r];
        auto& results = result_tensors[r];
            
        // todo keep const semantics on input -> seems that's not a thing in libtorch
        const auto input_tensor = torch::from_blob(
                reinterpret_cast<void*>(const_cast<float*>(inputs.data())), 
                at::IntArrayRef({input_sizes[0], input_sizes[1]})
            );
        auto output_tensor = module.forward({input_tensor}).toTensor();
        // assign to output
        std::memcpy(results.data(), output_tensor.data_ptr(), output_tensor.numel() * sizeof(float));
    }
    LIKWID_MARKER_STOP("libtorch");
    return std::chrono::high_resolution_clock::now() - start;
}

#else // DALOTIA_E_WITH_LIBTORCH

std::chrono::duration<double> run_inference_cblas(
    const std::vector<dalotia::vector<float>> &input_tensors,
    const std::vector<int> &input_sizes,
    size_t num_repetitions,
    const std::vector<int> &output_sizes,
    std::vector<dalotia::vector<float>> &result_tensors) {
    // load the model
    std::string filename = "./weights_SubgridLESNet.safetensors";
    auto dalotia_file = std::unique_ptr<dalotia::TensorFile>(dalotia::make_tensor_file(filename));
    
    std::chrono::duration<double> total_duration(0);
    int num_threads = 0;
    #pragma omp parallel reduction(+:num_threads)
    num_threads += 1;
    #pragma omp parallel default(none) \
        shared(input_tensors, input_sizes, num_repetitions, output_sizes, result_tensors, \
               dalotia_file, total_duration, num_threads, std::cout)
    {
        int target_node = omp_get_place_num();
        OpenMPPrivateMemoryResource res(target_node);
        // test dalotia w/ annd w/o custom allocator: no difference on CPU, leave in as an example
        const auto [weights_extents_1, weights_1] = dalotia_file->load_tensor_dense<float>("fc1.weight", dalotia_float_32, dalotia_C_ordering,{}, &res);
        const auto [biases_extents_1, biases_1] = dalotia_file->load_tensor_dense<float>("fc1.bias", dalotia_float_32, dalotia_C_ordering,{}, &res);
        const auto [weights_extents_2, weights_2] = dalotia_file->load_tensor_dense<float>("fc2.weight", dalotia_float_32, dalotia_C_ordering,{}, &res);
        const auto [biases_extents_2, biases_2] = dalotia_file->load_tensor_dense<float>("fc2.bias", dalotia_float_32, dalotia_C_ordering,{}, &res);
        const int num_output_features = output_sizes[1];
        const int num_input_features = input_sizes[1];
        const int num_inputs = input_sizes[0];
        const int num_hidden_neurons = weights_extents_1[0];
        assert(num_input_features == 10);
        assert(num_hidden_neurons == 300);
        assert(num_output_features == 6);
        assert(weights_extents_1 == std::vector<int>({num_hidden_neurons, num_input_features}));
        assert(biases_extents_1 == std::vector<int>({num_hidden_neurons})); 
        assert(weights_extents_2 == std::vector<int>({num_output_features, num_hidden_neurons}));
        assert(biases_extents_2 == std::vector<int>({num_output_features})); 
        assert(output_sizes == std::vector<int>({num_inputs, num_output_features}));

        int batch_size = (num_inputs + num_threads - 1) / num_threads;
        #pragma omp single
        {
            std::cout << "Using " << num_threads << " threads with batch size " << batch_size << std::endl;
        }
        dalotia::vector<float> hidden_values(batch_size * num_hidden_neurons, 0., &res);

        auto thread_num = omp_get_thread_num();
        #pragma omp barrier
        LIKWID_MARKER_START("cblas");
        const auto start = std::chrono::high_resolution_clock::now();
        for (size_t r = 0; r < num_repetitions; ++r) {
            auto& inputs = input_tensors[r];
            auto& results = result_tensors[r];

            const size_t my_inputs_start_index = thread_num * batch_size * num_input_features;
            int my_num_inputs = std::min(batch_size, num_inputs - thread_num * batch_size);
            if (my_num_inputs <= 0) throw std::runtime_error("Too many threads for the given input size");
            const size_t my_results_start_index = thread_num * batch_size * num_output_features;

            // fill hidden vector with bias
            for (size_t i = 0; i < my_num_inputs; ++i) {
                for (size_t j = 0; j < num_hidden_neurons; ++j) {
                    hidden_values[i * num_hidden_neurons + j] = biases_1[j];
                }
            }
            // todo compare to cblasRowMajor
            cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, num_hidden_neurons,
                        my_num_inputs, num_input_features, 1.0, weights_1.data(),
                        num_input_features, &inputs[my_inputs_start_index], num_input_features, 1.0,
                        hidden_values.data(), num_hidden_neurons);
            // ReLU
            for (auto& value : hidden_values) {
                value = value < 0. ? 0. : value;
            }

            // fill results vector with bias
            for (size_t i = 0; i < my_num_inputs; ++i) {
                for (size_t j = 0; j < output_sizes[1]; ++j) {
                    results[my_results_start_index + i * output_sizes[1] + j] = biases_2[j];
                }
            }
            cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, num_output_features,
                        my_num_inputs, num_hidden_neurons, 1.0, weights_2.data(),
                        num_hidden_neurons, hidden_values.data(), num_hidden_neurons, 1.0,
                        &results[my_results_start_index], num_output_features);
        }
        LIKWID_MARKER_STOP("cblas");
        auto my_duration = std::chrono::high_resolution_clock::now() - start;
        #pragma omp reduction(+:total_duration)
        total_duration += my_duration;
    }
    #pragma omp barrier
    return total_duration;
}
#endif // DALOTIA_E_WITH_LIBTORCH

int main(int argc, char *argv[]) {
    int num_inputs = 16*16*16;
    if (argc > 1) {
        num_inputs = std::stoi(argv[1]);
    }
#ifdef DALOTIA_E_FOR_MEMORY_TRACE
    std::vector<int> input_extents = {num_inputs, 10};
    std::vector<int> output_extents = {num_inputs, 6};
    dalotia::vector<float> input_tensor(num_inputs * 10);
#else
    // load input and output
    // the data used here is generated with generate_models.py
    auto [input_extents, input_tensor] = dalotia::load_tensor_dense<float>("./input_SubgridLESNet.safetensors", "random_input",
                                          dalotia_WeightFormat::dalotia_float_32, dalotia_Ordering::dalotia_C_ordering);
    assert(input_extents == std::vector<int>({16*16*16, 10}));
    assert_close(input_tensor[0], 0.4963);
    assert_close(input_tensor[1], 0.7682);
    assert_close(input_tensor[10], 0.3489);
    auto [output_extents, expected_output_tensor] =
        dalotia::load_tensor_dense<float>("./output_SubgridLESNet.safetensors", "output",
                                          dalotia_WeightFormat::dalotia_float_32, dalotia_Ordering::dalotia_C_ordering);
    assert(output_extents == std::vector<int>({16*16*16, 6}));
    assert_close(expected_output_tensor[0], 2.84722);
    assert_close(expected_output_tensor[1], 0.524039);
    assert_close(expected_output_tensor[6], 2.55544);

    // if the desired input length is different, we need to truncate or repeat the input tensor
    if (input_extents[0] != num_inputs) {
        std::cout << "Resizing input/output tensor from " << input_extents[0] << " to " << num_inputs << std::endl;
        size_t initial_input_size = input_tensor.size();
        input_tensor.resize(num_inputs * 10);
        input_extents[0] = num_inputs;
        size_t initial_output_size = expected_output_tensor.size();
        expected_output_tensor.resize(num_inputs *6);
        output_extents[0] = num_inputs;
        if (input_tensor.size() > initial_input_size) {
            for (size_t i = initial_input_size; i < input_tensor.size(); ++i) {
                input_tensor[i] = input_tensor[i % initial_input_size];
            }
            for (size_t i = initial_output_size; i < expected_output_tensor.size(); ++i) {
                expected_output_tensor[i] = expected_output_tensor[i % initial_output_size];
            }
        }
    }
    // initialize cache flushing
    if (cf_init() != 0){
        throw std::runtime_error("Cache flushing not enabled");
    }
#endif // DALOTIA_E_FOR_MEMORY_TRACE

#ifdef DALOTIA_E_FOR_MEMORY_TRACE
    const size_t num_repetitions = 1;
#else // DALOTIA_E_FOR_MEMORY_TRACE
    const size_t num_repetitions = 1000;
#ifdef DALOTIA_E_WITH_LIBTORCH
    std::cout << "Running inference with libtorch" << std::endl;
#else
    std::cout << "Running inference with cblas" << std::endl;
#endif
#endif // DALOTIA_E_FOR_MEMORY_TRACE
    std::vector<dalotia::vector<float>> input_tensors(num_repetitions, input_tensor);
    if (num_repetitions > 0) {
        assert(input_tensors[0].data() != input_tensors.back().data());
    }
    dalotia::vector<float> results(input_extents[0] * 6);
    std::vector<dalotia::vector<float>> result_tensors(num_repetitions, results);

    LIKWID_MARKER_INIT;
#ifndef DALOTIA_E_FOR_MEMORY_TRACE
    // flush caches to avoid the input and output being cached after initialization
    if (cf_flush(_CF_L3_) != 0) throw std::runtime_error("Cache flush failed!");
#endif // DALOTIA_E_FOR_MEMORY_TRACE
#ifdef DALOTIA_E_WITH_LIBTORCH
    LIKWID_MARKER_REGISTER("libtorch");
    const auto duration = run_inference_libtorch(input_tensors, input_extents, num_repetitions, output_extents, result_tensors);
#else
    LIKWID_MARKER_REGISTER("cblas");
    const auto duration = run_inference_cblas(input_tensors, input_extents, num_repetitions, output_extents, result_tensors);
#endif // DALOTIA_E_WITH_LIBTORCH
    LIKWID_MARKER_CLOSE;
#ifndef DALOTIA_E_FOR_MEMORY_TRACE
    std::cout << "Duration: " << duration.count() << "s" << std::endl;
    std::cout << "On average: " << duration.count() / static_cast<float>(num_repetitions) << "s" << std::endl;

    // check correctness of the output
    for (const auto& results : result_tensors) {
        for (size_t i = 0; i < results.size(); ++i) {
            if (std::abs(results[i] - expected_output_tensor[i]) > 1e-5) {
                std::cerr << "results[" << i << "] = " << results[i] << 
                                " != expected_output_tensor[" << i << "] = " << expected_output_tensor[i] << std::endl;
                throw std::runtime_error("results != expected_output_tensor");
            }
        }
    }
    if (cf_finalize() != 0) {
        throw std::runtime_error("Could not finalize cache flush");
    }
    std::cout << "success!" << std::endl;
#endif // not DALOTIA_E_FOR_MEMORY_TRACE

    return 0;
}
