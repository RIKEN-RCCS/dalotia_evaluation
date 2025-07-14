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

#ifdef DALOTIA_E_WITH_CPPFLOW
std::chrono::duration<double>
run_inference_cppflow(const std::vector<dalotia::vector<float>> &input_tensors,
                      const std::vector<int> &input_sizes,
                      size_t num_repetitions,
                      const std::vector<int> &output_sizes,
                      std::vector<dalotia::vector<float>> &result_tensors) {
  std::string tfModelPath = "./Harmonic_Mk11_M1_2D_gamma2/";
  std::unique_ptr<cppflow::model> tfModel =
      std::make_unique<cppflow::model>(tfModelPath); // load model
  // TODO does TF have something like optimize_for_inference?

  LIKWID_MARKER_START("cppflow");
  const auto start = std::chrono::high_resolution_clock::now();
  for (size_t r = 0; r < num_repetitions; ++r) {
    auto &inputs = input_tensors[r];
    auto &results = result_tensors[r];
    // ...
  }
  LIKWID_MARKER_STOP("cppflow");
  return std::chrono::high_resolution_clock::now() - start;
}

#endif // DALOTIA_E_WITH_CPPFLOW

std::chrono::duration<double>
run_inference_cblas(const std::vector<dalotia::vector<float>> &input_tensors,
                    const std::vector<int> &input_sizes, size_t num_repetitions,
                    const std::vector<int> &output_sizes,
                    std::vector<dalotia::vector<float>> &result_tensors) {
  // load the model tensors
  std::string tfModelPath = "./Harmonic_Mk11_M1_2D_gamma2/";
  auto dalotia_file = std::unique_ptr<dalotia::TensorFile>(
      dalotia::make_tensor_file(tfModelPath));
  for (const auto &name : dalotia_file->get_tensor_names()) {
    if (dalotia_file->get_num_dimensions(name) == 0) {
      continue; // skip scalars
    }
    std::vector<int> extents = dalotia_file->get_tensor_extents(name);
    std::cout << "Tensor name: " << name << ", extents: ";
    for (const auto &extent : extents) {
      std::cout << extent << " ";
    }
    std::cout << std::endl;
  }

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
  int num_inputs = 16 * 16 * 16;
  if (argc > 1) {
    num_inputs = std::stoi(argv[1]);
  }
  std::vector<int> input_extents = {num_inputs, 10};
  std::vector<int> output_extents = {num_inputs, 6};
  dalotia::vector<float> input_tensor(num_inputs * 10);
  const size_t num_repetitions = 1000;
#ifdef DALOTIA_E_WITH_CPPFLOW
  std::cout << "Running inference with cppflow" << std::endl;
#else
  std::cout << "Running inference with cblas" << std::endl;
#endif
  std::vector<dalotia::vector<float>> input_tensors(num_repetitions,
                                                    input_tensor);
  if (num_repetitions > 0) {
    assert(input_tensors[0].data() != input_tensors.back().data());
  }
  dalotia::vector<float> results(input_extents[0] * 6);
  std::vector<dalotia::vector<float>> result_tensors(num_repetitions, results);

  LIKWID_MARKER_INIT;
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
  return 0;
}
