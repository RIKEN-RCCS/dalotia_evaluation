# GPT-2 124M inference benchmark — single source for CPU and GPU
# CPU: compiled with the project's C++ compiler, gemm dispatches to cblas
# GPU: same source compiled with -DDALOTIA_E_WITH_CUBLAS, needs nvc++ -cuda
#      or a compiler that supports thrust + cuBLAS
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

# GPU target
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
endif()
