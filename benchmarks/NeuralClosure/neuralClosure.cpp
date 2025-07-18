#include <cassert>
#include <chrono>
#include <cstring> // std::memcpy
#include <iostream>
#include <vector>

#include "cacheflush.h"
#include "dalotia.hpp"
#include "dalotia_safetensors_file.hpp"

#ifdef DALOTIA_E_WITH_CPPFLOW
#include "cppflow/cppflow.h"
#endif

#include "cblas.h"

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
#endif // LIKWID_PERFMON

bool is_close(float a, float b, float tol = 1e-4) {
  return std::abs(a - b) < tol;
}
void assert_close(float a, float b, float tol = 1e-5) {
#ifndef NDEBUG
  auto is_close_result = is_close(a, b, tol);
  if (!is_close_result) {
    std::cerr << "assertion failed: " << a << " != " << b << std::endl;
  }
  assert(is_close_result);
#endif // NDEBUG
}

#ifdef DALOTIA_E_WITH_CPPFLOW

std::chrono::duration<double>
run_inference_cppflow(const std::vector<std::vector<float>> &input_tensors,
                      const std::vector<int> &input_sizes,
                      size_t num_repetitions,
                      const std::vector<int> &output_sizes,
                      std::vector<std::vector<float>> &result_tensors) {
  std::string tfModelName = "Monomial_Mk11_M3_2D_gamma3";
  std::string tfModelPath = "./" + tfModelName + "/";

  std::unique_ptr<cppflow::model> tfModel =
      std::make_unique<cppflow::model>(tfModelPath); // load model

  // TODO does TF have something like optimize_for_inference?
  long int nSystem = 10;
  long int servingSize = input_tensors[0].size() / (nSystem - 1);

  LIKWID_MARKER_START("cppflow");
  const auto start = std::chrono::high_resolution_clock::now();
  for (size_t r = 0; r < num_repetitions; ++r) {
    auto &inputs = input_tensors[r];
    assert(servingSize * (nSystem - 1) == inputs.size());
    auto &results = result_tensors[r];

    // the next few lines are heavily inspired by KiT-RT
    // this constructor only supports std::vector (no PMR)
    auto modelInput = cppflow::tensor(inputs, {servingSize, nSystem - 1});

    // cf.
    // https://github.com/KiT-RT/kitrt_code/blob/bcf4d1b6ad2af3a7d88943cef5cdf90fea752401/src/optimizers/neuralnetworkoptimizer.cpp#L381
    std::vector<cppflow::tensor> TFresults = tfModel->operator()(
        {{"serving_default_input_1:0", modelInput}},
        {"StatefulPartitionedCall:0", "StatefulPartitionedCall:1",
         "StatefulPartitionedCall:2"});

    // move to output
    assert(TFresults[1].get_data<float>().size() == results.size());
    results = std::move(TFresults[1].get_data<float>());
  }
  LIKWID_MARKER_STOP("cppflow");
  return std::chrono::high_resolution_clock::now() - start;
}

#endif // DALOTIA_E_WITH_CPPFLOW

// helper function to get the number of layers
std::set<int> get_layer_numbers(const dalotia::TensorFile *dalotia_file) {
  std::set<int> layer_numbers;
  for (const auto &name : dalotia_file->get_tensor_names()) {
    if (dalotia_file->get_num_dimensions(name) == 0) {
      continue; // skip scalars
    }
    auto pos = name.find("layer_");
    if (pos != std::string::npos) {
      // parse the integer that comes after "layer_" ; next character after is
      // either d or n
      auto pos_after = name.find_first_of("dn", pos + 6);
      assert(pos_after != std::string::npos);
      int layer_num = std::stoi(name.substr(pos + 6, pos_after - (pos + 6)));
      layer_numbers.insert(layer_num);
    }
  }
  assert(layer_numbers.size() > 0);
  return layer_numbers;
}

std::chrono::duration<double>
run_inference_cblas(const std::vector<std::vector<float>> &input_tensors,
                    const std::vector<int> &input_sizes, size_t num_repetitions,
                    const std::vector<int> &output_sizes,
                    std::vector<std::vector<float>> &result_tensors) {
  // load the model tensors
  std::string tfModelName = "Monomial_Mk11_M3_2D_gamma3";
  std::string tfModelPath = "./" + tfModelName + "/";
  auto dalotia_file = std::unique_ptr<dalotia::TensorFile>(
      dalotia::make_tensor_file(tfModelPath));

  const int num_inputs = input_sizes[0];
  const int num_input_features = input_sizes[1];
  const int num_output_features = output_sizes[1];
  const std::set<int> layer_numbers = get_layer_numbers(dalotia_file.get());
  const int num_layers = layer_numbers.size();
  std::vector<std::vector<float>> weights(num_layers);
  std::vector<std::vector<int>> weights_extents(num_layers);
  std::vector<std::vector<float>> biases(num_layers);
  std::vector<std::vector<int>> biases_extents(num_layers);
  for (const auto &layer : layer_numbers) {
    // load weights and biases for all layers
    // TODO what do the variants with /m/ or /v/ mean?
    std::string weight_name;
    std::string bias_name;
    if (layer < 0) {
      continue;
      weight_name = "layer_" + std::to_string(layer) +
                    "_input/kernel/Read/ReadVariableOp";
      bias_name =
          "layer_" + std::to_string(layer) + "_input/bias/Read/ReadVariableOp";
    } else {
      weight_name = "layer_" + std::to_string(layer) +
                    "dense_component/kernel/Read/ReadVariableOp";
      bias_name = "layer_" + std::to_string(layer) +
                  "nn_component/bias/Read/ReadVariableOp";
    }
    std::cout << "Loading weights and biases for layer " << layer << ": "
              << weight_name << ", " << bias_name << std::endl;
    biases_extents[layer] = dalotia_file->get_tensor_extents(bias_name);
    std::cout << "Layer " << layer << ": "
              << "biases extents: " << biases_extents[layer][0] << std::endl;
  }
  assert(weights_extents[0][0] == num_input_features);
  assert(weights_extents[num_layers - 1][1] == num_output_features);
  assert(weights_extents[0][1] == weights_extents[1][0]);

  LIKWID_MARKER_START("cppflow");
  const auto start = std::chrono::high_resolution_clock::now();
  for (size_t r = 0; r < num_repetitions; ++r) {
    auto &inputs = input_tensors[r];
    auto &results = result_tensors[r];
    // ...
  }
  LIKWID_MARKER_STOP("cblas");
  return std::chrono::high_resolution_clock::now() - start;
}

int main(int argc, char *argv[]) {
  constexpr int kitrt_servingSize = 12920;
  int num_inputs = kitrt_servingSize;
  if (argc > 1) {
    num_inputs = std::stoi(argv[1]);
  }
  int num_input_channels = 9;  // number of input channels
  int num_output_channels = 9; // number of output channels
#ifdef DALOTIA_E_FOR_MEMORY_TRACE
  std::vector<int> input_extents = {num_inputs, num_input_channels};
  std::vector<int> output_extents = {num_inputs, num_output_channels};
  std::vector<float> input_tensor(num_inputs * num_input_channels);
#else
  // read input and output in binary
  std::vector<float> input_tensor(kitrt_servingSize * num_input_channels);
  std::vector<int> input_extents = {kitrt_servingSize, num_input_channels};
  std::vector<float> expected_output_tensor(kitrt_servingSize *
                                            num_output_channels);
  std::vector<int> output_extents = {kitrt_servingSize, num_output_channels};
  int iter = 1367;
  {
    std::string inputs_file_name = "inputs" + std::to_string(iter) + ".bin";
    std::ifstream inputsFile(inputs_file_name, std::ios::binary);
    if (!inputsFile.is_open()) {
      std::cerr << "Error opening inputs file " + inputs_file_name +
                       " for reading."
                << std::endl;
      return -1;
    }
    inputsFile.read(reinterpret_cast<char *>(input_tensor.data()),
                    input_tensor.size() * sizeof(float));
    assert(inputsFile.good());
    assert(inputsFile.gcount() == input_tensor.size() * sizeof(float));
    inputsFile.close();
    assert_close(input_tensor[0], -0.178185);
    assert_close(input_tensor[1], 0.00465835);
    assert_close(input_tensor[2], 0.334049);
  }
  {
    std::string outputs_file_name = "outputs" + std::to_string(iter) + ".bin";
    std::ifstream outputsFile(outputs_file_name, std::ios::binary);
    if (!outputsFile.is_open()) {
      std::cerr << "Error opening outputs file for reading." << std::endl;
      return -1;
    }
    outputsFile.read(reinterpret_cast<char *>(expected_output_tensor.data()),
                     expected_output_tensor.size() * sizeof(float));
    assert(outputsFile.good());
    assert(outputsFile.gcount() ==
           expected_output_tensor.size() * sizeof(float));
    outputsFile.close();
    assert_close(expected_output_tensor[0], -0.621178);
    assert_close(expected_output_tensor[1], -0.106261);
    assert_close(expected_output_tensor[2], -0.0985599);
    assert_close(expected_output_tensor[9], 0.693398);
    assert_close(expected_output_tensor[10], -0.114202);
    assert_close(expected_output_tensor[11], -0.0827721);
    assert_close(expected_output_tensor[18], -0.642729);
    assert_close(expected_output_tensor[19], -0.012028);
    assert_close(expected_output_tensor[116270], -6.22051);
    assert_close(expected_output_tensor[116271], 0.69463);
    assert_close(expected_output_tensor[116278], -1.24646);
    assert_close(expected_output_tensor[116279], -6.46241);
  }

  // if the desired input length is different, we need to truncate or
  // repeat the input tensor
  if (input_extents[0] != num_inputs) {
    std::cout << "Resizing input/output tensor from " << input_extents[0]
              << " to " << num_inputs << std::endl;
    size_t initial_input_size = input_tensor.size();
    input_tensor.resize(num_inputs * num_input_channels);
    input_extents[0] = num_inputs;
    size_t initial_output_size = expected_output_tensor.size();
    expected_output_tensor.resize(num_inputs * num_output_channels);
    output_extents[0] = num_inputs;
    if (input_tensor.size() > initial_input_size) {
      for (size_t i = initial_input_size; i < input_tensor.size(); ++i) {
        input_tensor[i] = input_tensor[i % initial_input_size];
      }
      for (size_t i = initial_output_size; i < expected_output_tensor.size();
           ++i) {
        expected_output_tensor[i] =
            expected_output_tensor[i % initial_output_size];
      }
    }
  }
  // initialize cache flushing
  if (cf_init() != 0) {
    throw std::runtime_error("Cache flushing not enabled");
  }
#endif // DALOTIA_E_FOR_MEMORY_TRACE

#ifdef DALOTIA_E_FOR_MEMORY_TRACE
  const size_t num_repetitions = 1;
#else // DALOTIA_E_FOR_MEMORY_TRACE
  const size_t num_repetitions = 1000;
#ifdef DALOTIA_E_WITH_CPPFLOW
  std::cout << "Running inference with cppflow" << std::endl;
#else
  std::cout << "Running inference with cblas" << std::endl;
#endif
#endif // DALOTIA_E_FOR_MEMORY_TRACE
  std::vector<std::vector<float>> input_tensors(num_repetitions, input_tensor);
  if (num_repetitions > 1) {
    assert(input_tensors[0].data() != input_tensors.back().data());
  }
  std::vector<float> results(num_inputs * num_output_channels, 0.);
  std::vector<std::vector<float>> result_tensors(num_repetitions, results);

  LIKWID_MARKER_INIT;
#ifndef DALOTIA_E_FOR_MEMORY_TRACE
  // flush caches to avoid the input and output being cached after
  // initialization
  if (cf_flush(_CF_L3_) != 0)
    throw std::runtime_error("Cache flush failed!");
#endif // DALOTIA_E_FOR_MEMORY_TRACE
#ifdef DALOTIA_E_WITH_CPPFLOW
  LIKWID_MARKER_REGISTER("cppflow");
  const auto duration =
      run_inference_cppflow(input_tensors, input_extents, num_repetitions,
                            output_extents, result_tensors);
#else
  LIKWID_MARKER_REGISTER("cblas");
  const auto duration =
      run_inference_cblas(input_tensors, input_extents, num_repetitions,
                          output_extents, result_tensors);
#endif // DALOTIA_E_WITH_CPPFLOW
  LIKWID_MARKER_CLOSE;
#ifndef DALOTIA_E_FOR_MEMORY_TRACE
  std::cout << "Duration: " << duration.count() << "s" << std::endl;
  std::cout << "On average: "
            << duration.count() / static_cast<float>(num_repetitions) << "s"
            << std::endl;

  // check correctness of the output
  for (const auto &results : result_tensors) {
    for (size_t i = 0; i < results.size(); ++i) {
      if (std::abs(results[i] - expected_output_tensor[i]) > 1e-4) {
        // 1e-5 was too strict
        std::cerr << "results[" << i << "] = " << results[i]
                  << " != expected_output_tensor[" << i
                  << "] = " << expected_output_tensor[i] << std::endl;
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
