# GPT-2 124M inference benchmark
# Downloads model.safetensors from HuggingFace at configure time
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
    message(STATUS "GPT2: skipping gpt2_cpp (requires DALOTIA_E_WITH_BOOST_MULTI=ON)")
    return()
endif()

add_executable(gpt2_cpp gpt2.cpp)
target_link_libraries(gpt2_cpp PUBLIC dalotia::dalotia_cpp multi)

if (OpenMP_CXX_FOUND)
    target_link_libraries(gpt2_cpp PUBLIC OpenMP::OpenMP_CXX)
endif()

# Boost.Multi links BLAS internally; add include dirs for dalotia headers
if (BLAS_INCLUDE_DIRS)
    target_include_directories(gpt2_cpp PUBLIC ${BLAS_INCLUDE_DIRS})
endif()

if (DALOTIA_E_WITH_CUBLAS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/gpt2_cublas.cu")
    add_executable(gpt2_cublas gpt2_cublas.cu)
    set_target_properties(gpt2_cublas PROPERTIES CUDA_STANDARD 17)
    target_link_libraries(gpt2_cublas PUBLIC dalotia::dalotia_cpp)
    target_link_libraries(gpt2_cublas PRIVATE CUDA::cudart CUDA::cublas)
endif()
