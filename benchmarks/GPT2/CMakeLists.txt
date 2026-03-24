# GPT-2 124M inference benchmark
# Downloads model.safetensors from HuggingFace at configure time

set(GPT2_MODEL_URL "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors")
set(GPT2_MODEL_FILE "${CMAKE_CURRENT_BINARY_DIR}/model.safetensors")

if(NOT EXISTS ${GPT2_MODEL_FILE})
    message(STATUS "Downloading GPT-2 model.safetensors from HuggingFace...")
    FetchContent_Declare(
        gpt2_model
        URL ${GPT2_MODEL_URL}
        DOWNLOAD_NO_EXTRACT true
    )
    if(NOT gpt2_model_POPULATED)
        FetchContent_MakeAvailable(gpt2_model)
    endif()
    file(COPY "${gpt2_model_SOURCE_DIR}/model.safetensors"
         DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

add_executable(gpt2_cpp gpt2.cpp)
target_link_libraries(gpt2_cpp PUBLIC dalotia::dalotia_cpp)
target_include_directories(gpt2_cpp PUBLIC ${BLAS_INCLUDE_DIRS})
target_link_libraries(gpt2_cpp PRIVATE BLAS::BLAS)

if (OpenMP_CXX_FOUND)
    target_link_libraries(gpt2_cpp PUBLIC OpenMP::OpenMP_CXX)
endif()

if (DALOTIA_E_WITH_BOOST_MULTI)
    target_compile_options(gpt2_cpp PUBLIC "-DDALOTIA_E_WITH_BOOST_MULTI")
    target_include_directories(gpt2_cpp PUBLIC ${MULTI_CPP_INCLUDE_DIR} ${MULTI_DIR})
    add_dependencies(gpt2_cpp multi)
endif(DALOTIA_E_WITH_BOOST_MULTI)

if (DALOTIA_E_WITH_CUBLAS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/gpt2_cublas.cu")
    add_executable(gpt2_cublas gpt2_cublas.cu)
    set_target_properties(gpt2_cublas PROPERTIES CUDA_STANDARD 17)
    target_link_libraries(gpt2_cublas PUBLIC dalotia::dalotia_cpp)
    target_link_libraries(gpt2_cublas PRIVATE CUDA::cudart CUDA::cublas)
endif()
