# GPT-2 124M inference benchmark — single source for CPU and GPU
#
# Three supported compiler configurations (all compile gpt2.cpp):
#
#   1. CPU (g++/clang++):  gemm → cblas,  element-wise → host loops
#      cmake -DDALOTIA_E_WITH_BOOST_MULTI=ON ...
#
#   2. GPU via nvc++:      gemm → cuBLAS, element-wise → CUDA kernels
#      cmake -DCMAKE_CXX_COMPILER=nvc++
#            -DCMAKE_CXX_FLAGS="-cuda -Mnorpath --diag_suppress=code_is_unreachable --diag_suppress=cuda_compile"
#            -DDALOTIA_E_WITH_CUBLAS=ON -DDALOTIA_E_WITH_BOOST_MULTI=ON ...
#
#   3. GPU via clang++:    gemm → cuBLAS, element-wise → CUDA kernels
#      cmake -DCMAKE_CXX_COMPILER=clang++
#            -DCUDA_GPU_ARCH=sm_90
#            -DDALOTIA_E_WITH_CUBLAS=ON -DDALOTIA_E_WITH_BOOST_MULTI=ON ...
#      (Prerequisite: spack install llvm +clang ~flang ~gold ~lldb ~lld ~polly)
#      Clang CUDA flags (-x cuda, --cuda-gpu-arch, --cuda-path) are set
#      automatically per-target; no global CMAKE_CXX_FLAGS needed.
#
# Requires DALOTIA_E_WITH_BOOST_MULTI=ON

set(GPT2_MODEL_URL "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors")
set(GPT2_MODEL_FILE "${CMAKE_CURRENT_BINARY_DIR}/model.safetensors")

if(NOT EXISTS ${GPT2_MODEL_FILE})
    message(STATUS "Downloading GPT-2 model.safetensors from HuggingFace...")
    file(DOWNLOAD ${GPT2_MODEL_URL} ${GPT2_MODEL_FILE}
         SHOW_PROGRESS
         STATUS GPT2_DOWNLOAD_STATUS)
    list(GET GPT2_DOWNLOAD_STATUS 0 GPT2_DOWNLOAD_ERROR)
    if(NOT GPT2_DOWNLOAD_ERROR EQUAL 0)
        list(GET GPT2_DOWNLOAD_STATUS 1 GPT2_DOWNLOAD_MSG)
        file(REMOVE ${GPT2_MODEL_FILE})
        message(WARNING "GPT-2 model download failed: ${GPT2_DOWNLOAD_MSG}")
    endif()
endif()

if (NOT DALOTIA_E_WITH_BOOST_MULTI)
    message(STATUS "GPT2: skipping (requires DALOTIA_E_WITH_BOOST_MULTI=ON)")
    return()
endif()

if (TARGET BLAS::BLAS)
    add_executable(gpt2_cpp gpt2.cpp)
    target_link_libraries(gpt2_cpp PUBLIC dalotia::dalotia_cpp multi)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(gpt2_cpp PUBLIC OpenMP::OpenMP_CXX)
    endif()
    if (BLAS_INCLUDE_DIRS)
        target_include_directories(gpt2_cpp PUBLIC ${BLAS_INCLUDE_DIRS})
    endif()
endif()

# GPU target — same source, compiled with DALOTIA_E_WITH_CUBLAS
if (DALOTIA_E_WITH_CUBLAS)
    add_executable(gpt2_cublas gpt2.cpp)
    target_compile_definitions(gpt2_cublas PRIVATE DALOTIA_E_WITH_CUBLAS)
    target_link_libraries(gpt2_cublas PUBLIC dalotia::dalotia_cpp multi)
    target_link_libraries(gpt2_cublas PRIVATE CUDA::cudart CUDA::cublas)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(gpt2_cublas PUBLIC OpenMP::OpenMP_CXX)
    endif()
    if (BLAS_INCLUDE_DIRS)
        target_include_directories(gpt2_cublas PUBLIC ${BLAS_INCLUDE_DIRS})
    endif()

    # Clang CUDA: compile gpt2.cpp as CUDA via -x cuda + --cuda-gpu-arch
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        if (NOT DEFINED CUDA_GPU_ARCH)
            set(CUDA_GPU_ARCH "sm_90" CACHE STRING "CUDA GPU architecture for clang (e.g. sm_90, sm_120)")
        endif()
        # Detect CUDA toolkit path from CMake's CUDAToolkit
        if (CUDAToolkit_BIN_DIR)
            get_filename_component(_CUDA_PATH "${CUDAToolkit_BIN_DIR}" DIRECTORY)
        else()
            set(_CUDA_PATH "/usr/local/cuda")
        endif()
        target_compile_options(gpt2_cublas PRIVATE
            -x cuda
            --cuda-gpu-arch=${CUDA_GPU_ARCH}
            --cuda-path=${_CUDA_PATH}
        )
    endif()
endif()
