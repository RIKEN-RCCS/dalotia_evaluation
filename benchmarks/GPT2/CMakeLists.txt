# GPT-2 124M inference benchmark — single source for CPU and GPU
# CPU: compiled with the default C++ compiler, gemm dispatches to cblas
# GPU: compiled with nvc++ -cuda, gemm dispatches to cuBLAS via thrust pointers
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
if (BLAS_INCLUDE_DIRS)
    target_include_directories(gpt2_cpp PUBLIC ${BLAS_INCLUDE_DIRS})
endif()

# ── GPU target (nvc++ -cuda with cuBLAS) — same source file ──────────────
if (DALOTIA_E_WITH_CUBLAS)
    find_program(NVCPP nvc++
        HINTS ENV NVHPC
        PATH_SUFFIXES Linux_aarch64/25.7/compilers/bin Linux_x86_64/25.7/compilers/bin
              compilers/bin bin
        PATHS /spack/opt/spack/linux-neoverse_v2/nvhpc-25.7-7cf6nw5xs6ofs2i7ecosceemvp4lkgz2
    )
    if (NVCPP)
        message(STATUS "GPT2: GPU target using nvc++ at ${NVCPP}")

        # Resolve include dirs at configure time
        FetchContent_GetProperties(multi SOURCE_DIR _MULTI_SRC)

        # Resolve dalotia include & lib
        get_target_property(_DAL_INCS dalotia::dalotia_cpp INTERFACE_INCLUDE_DIRECTORIES)
        # Filter out generator expressions
        set(_DAL_INC_FLAGS "")
        foreach(d ${_DAL_INCS})
            string(REGEX MATCH "^[^$]" _ok "${d}")
            if (_ok)
                list(APPEND _DAL_INC_FLAGS "-I${d}")
            endif()
        endforeach()

        # Find the dalotia static libs
        find_library(_DAL_LIB dalotia_cpp PATHS "${dalotia_DIR}/../../../lib" "${dalotia_DIR}/../../lib" NO_DEFAULT_PATH)
        find_library(_ST_LIB safetensors_cpp PATHS "${dalotia_DIR}/../../../lib" "${dalotia_DIR}/../../lib" NO_DEFAULT_PATH)

        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gpt2_cublas
            COMMAND ${NVCPP} -cuda -gpu=cc120 -std=c++17 -O2
                -DDALOTIA_E_WITH_CUBLAS
                -DDALOTIA_WITH_CPP_PMR -DDALOTIA_WITH_CUDA -DDALOTIA_WITH_CUFILE
                -DDALOTIA_WITH_SAFETENSORS_CPP
                ${_DAL_INC_FLAGS}
                -I${_MULTI_SRC}/include
                -I${BLAS_INCLUDE_DIRS}
                ${CMAKE_CURRENT_SOURCE_DIR}/gpt2.cpp
                -o ${CMAKE_CURRENT_BINARY_DIR}/gpt2_cublas
                ${_DAL_LIB} ${_ST_LIB}
                -lcublas -lcudart -lcuda
                -L/usr/local/cuda-13.2/targets/sbsa-linux/lib -lcufile
                -L/spack/opt/spack/linux-neoverse_v2/openblas-0.3.30-jvbxem4bguhzw4xxkuqn3oq3muao35qe/lib -lopenblas
                --diag_suppress=code_is_unreachable
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gpt2.cpp
            COMMENT "Building gpt2_cublas with nvc++ -cuda (same source as CPU)"
            VERBATIM
        )
        add_custom_target(gpt2_cublas ALL
            DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/gpt2_cublas)
    else()
        message(STATUS "GPT2: nvc++ not found, skipping GPU target")
    endif()
endif()
