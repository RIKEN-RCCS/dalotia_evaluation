cmake_minimum_required(VERSION 3.24)

project(dalotia_evaluation)

option(DALOTIA_E_FOR_MEMORY_TRACE "avoid output and checks to run memory usage profile (pinatrace)" OFF)
option(DALOTIA_E_WITH_LIKWID_MARKERS "instrument inference stages with likwid perfmon" OFF)
option(DALOTIA_E_WITH_BOOST_MULTI "include boost::multi for tensor operations" OFF)
option(DALOTIA_E_WITH_NDIRECT "include ndirect for convolution operations on ARM" OFF)
option(DALOTIA_E_WITH_LIBTORCH "include libtorch for jitted tensor operations" OFF)
option(DALOTIA_E_WITH_ONEDNN "include oneDNN for tensor operations" OFF)
option(DALOTIA_E_WITH_FORTRAN "Build Fortran executables" ON)
if (DALOTIA_E_WITH_FORTRAN)
    enable_language(Fortran)
endif (DALOTIA_E_WITH_FORTRAN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MAKE_PROGRAM "make")

message(STATUS "CMAKE BUILD TYPE: ${CMAKE_BUILD_TYPE}")

include(ExternalProject)
include(FetchContent)
if (NOT DEFINED dalotia_DIR OR dalotia_DIR MATCHES "fetch")
  FetchContent_Declare(
    dalotia
    GIT_REPOSITORY https://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@github.com/RIKEN-RCCS/dalotia.git #TODO replace w/ https
    GIT_TAG v1.0.0
    # CMAKE_ARGS  # I believe these are ignored, set like below
    #       -D DALOTIA_BUILD_TESTS=ON
    OVERRIDE_FIND_PACKAGE
  )
  set(DALOTIA_BUILD_TESTS OFF)
  FetchContent_MakeAvailable(dalotia)
  FetchContent_GetProperties(dalotia SOURCE_DIR DALOTIA_CPP_INCLUDE_DIR BINARY_DIR DALOTIA_CPP_LIBRARY)
  message(STATUS "DALOTIA_CPP_INCLUDE_DIR: ${DALOTIA_CPP_INCLUDE_DIR}")
  message(STATUS "DALOTIA_CPP_LIBRARY: ${DALOTIA_CPP_LIBRARY}")
else()
  find_package(safetensors-cpp REQUIRED)
endif()  
find_package(dalotia REQUIRED)

if(DALOTIA_E_WITH_LIBTORCH)
  # the default pip-downloaded libtorch is built with CXX11_ABI=0 -> to use it everything else needs that too
  set(DALOTIA_E_USE_OLD_ABI ON)
endif(DALOTIA_E_WITH_LIBTORCH)

if (DALOTIA_E_USE_OLD_ABI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
  target_compile_definitions(dalotia_cpp PUBLIC _GLIBCXX_USE_CXX11_ABI=0)
  target_compile_definitions(safetensors_cpp PUBLIC _GLIBCXX_USE_CXX11_ABI=0)
endif (DALOTIA_E_USE_OLD_ABI)

set(BLA_STATIC ON)
if (NOT DEFINED BLA_VENDOR OR BLA_VENDOR MATCHES "default")
  set(BLA_VENDOR "OpenBLAS") # or: Fujitsu_SSL2, Intel10_64lp_seq, OpenBLAS...
endif()
find_package(BLAS) 
# careful: torch may be shipped with its own blas
# careful: loading the mkl module always changes the BLAS in dnnl, regardless of BLA_VENDOR

if (NOT BLAS_INCLUDE_DIRS OR BLAS_INCLUDE_DIRS STREQUAL "")
  # fix for finding the include dirs in OpenBLAS default install
  # list with blas libraries but get their parent directory
  SET(BLAS_LIBRARIES_PARENTS "")
   FOREACH(l BLAS_LIBRARIES)
      cmake_path(GET ${l} PARENT_PATH p)
      LIST(APPEND BLAS_LIBRARIES_PARENTS "${p}/..") #include?
   ENDFOREACH(l)
  find_path(BLAS_INCLUDE_DIRS cblas.h
  BLAS_LIBRARIES_PARENTS)
endif()

# optional / conditional dependencies
if (DALOTIA_E_WITH_LIKWID_MARKERS)
  find_package(likwid REQUIRED)
endif (DALOTIA_E_WITH_LIKWID_MARKERS)

if (DALOTIA_E_WITH_NDIRECT)
  include(ExternalProject)
  ExternalProject_Add(
    ndirect
    GIT_REPOSITORY https://github.com/freifrauvonbleifrei/nDIRECT.git
    GIT_TAG main #TODO replace with final url/tag
    BUILD_IN_SOURCE 1
    SOURCE_DIR ${ndirect_SOURCE_DIR} #/direct_cnn
    CONFIGURE_COMMAND ""
    BUILD_COMMAND cd direct_cnn && ${CMAKE_MAKE_PROGRAM} -j 1
    INSTALL_COMMAND ""
  )
  ExternalProject_Get_Property(ndirect SOURCE_DIR)
  message(STATUS "ndirect_SOURCE_DIR: ${SOURCE_DIR}")
  set(NDIRECT_CPP_INCLUDE_DIR ${SOURCE_DIR}/direct_cnn)
  set(NDIRECT_CPP_LIBRARY_DIR ${SOURCE_DIR}/direct_cnn)
endif (DALOTIA_E_WITH_NDIRECT)

if (DALOTIA_E_WITH_ONEDNN)
  if (NOT DEFINED dnnl_DIR OR dnnl_DIR MATCHES "fetch")
    FetchContent_Declare(
      dnnl
      GIT_REPOSITORY https://github.com/oneapi-src/oneDNN.git
      GIT_TAG v3.6.2
      OVERRIDE_FIND_PACKAGE
    )
    set(DNNL_LIBRARY_TYPE STATIC)
    FetchContent_MakeAvailable(dnnl)
    find_package(dnnl 3.6.2 CONFIG REQUIRED)
  else()
    # to pass DNNL_DIR on the command line:
    find_package(dnnl REQUIRED CMAKE_FIND_ROOT_PATH_BOTH)
  endif()
endif (DALOTIA_E_WITH_ONEDNN)

if (DALOTIA_E_WITH_LIBTORCH)
  # # beware: this can take half a day to download and compile
  # set(FETCHCONTENT_QUIET FALSE)
  # set(pytorch_patch git apply ${CMAKE_CURRENT_SOURCE_DIR}/cmake/pytorch_cmake_no_force_cache.patch)
  # #cf. https://github.com/pytorch/pytorch/tree/v2.5.1/third_party
  # FetchContent_Declare(
  #   libtorch
  #   PATCH_COMMAND ${pytorch_patch}
  #   UPDATE_DISCONNECTED 1
  #   GIT_REPOSITORY "https://github.com/pytorch/pytorch"
  #   GIT_TAG "v2.5.1"
  #   GIT_SUBMODULES third_party/FP16 third_party/FXdiv third_party/NNPACK third_party/NVTX third_party/XNNPACK third_party/cpuinfo third_party/cutlass third_party/eigen third_party/flatbuffers third_party/fmt third_party/ideep third_party/nlohmann third_party/onnx third_party/pocketfft third_party/protobuf third_party/psimd third_party/pthreadpool third_party/tensorpipe third_party/sleef
  #   FIND_PACKAGE_ARGS
  # )
  # # cf. https://github.com/pytorch/pytorch/blob/v2.5.1/cmake/Dependencies.cmake
  # set(BUILD_BINARY ON)
  # set(BUILD_CAFFE2 OFF)
  # set(BUILD_SHARED_LIBS OFF)
  # set(BUILD_PYTHON OFF)
  # set(BUILD_TEST OFF)
  # set(BUILD_MOBILE_BENCHMARK OFF)
  # set(BUILD_MOBILE_TEST OFF)
  # set(USE_NATIVE_ARCH ON)
  # # cf. https://github.com/fujitsu/pytorch/blob/cf86c94a480b0e9bb038b7eccc646983fa6f5980/scripts/fujitsu/5_pytorch.sh#L131C25-L133C26
  # set(USE_MKLDNN ON)
  # set(USE_MKLDNN_CBLAS ON)
  # set(USE_DISTRIBUTED OFF)
  # set(USE_KINETO OFF)
  # set(USE_ROCM OFF)
  # set(USE_NCCL OFF)
  # set(USE_MPI OFF)
  # set(USE_CUDA OFF) #TODO maybe set on for GPU comparison
  # set(USE_CUDNN OFF) #TODO maybe set on for GPU comparison
  # set(USE_GLOO OFF) #TODO maybe set on for GPU comparison
  # set(USE_MAGMA OFF) #TODO maybe set on for GPU comparison
  # set(USE_PROF OFF)
  # set(USE_XPU OFF)
  # set(USE_VULKAN OFF)
  # set(USE_SOURCE_DEBUG_ON_MOBILE OFF)
  # set(USE_ASAN OFF)
  # set(USE_TSAN OFF)
  # set(USE_FBGEMM OFF)
  # set(USE_FAKELOWP OFF)
  # set(USE_NNPACK OFF) #??
  # # set(USE_PYTORCH_QNNPACK OFF)
  # # set(USE_XNNPACK OFF)
  # # set(BLAS_INFO "open")
  # # set(WITH_BLAS "open")
  # set(USE_NUMA OFF) #???
  # # needs loaded: python@3.8.0: py-pyyaml, py-typing-extensions
  # FetchContent_MakeAvailable(libtorch)
  find_package(Torch PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../venv/lib64/python3.12/site-packages/ NO_DEFAULT_PATH)
  find_package(Torch REQUIRED)
  set(BUILD_SHARED_LIBS OFF)

  # # addresses: libgfortran-daac5196.so.5.0.0, needed by 
  # # .../venv/lib64/python3.12/site-packages/torch/lib/../../torch.libs/libopenblasp-r0-2e4c6a40.3.25.so, not found
  # message(STATUS "I Found Torch lib: ${TORCH_LIBRARY}")
  # message(STATUS "I Found Torch dir: ${Torch_DIR}")
  # cmake_path(GET ${Torch_DIR} PARENT_PATH p) #?? undefined?!?‽
  # LIST(APPEND TORCH_LIBRARIES "${p}/../../../../torch.libs")
  # message(STATUS "Torch libraries are: ${TORCH_LIBRARIES}")
  # well then I'll just add it manually for now
  # link_directories("../venv/lib64/python3.12/site-packages/torch.libs/")
endif(DALOTIA_E_WITH_LIBTORCH)

if (DALOTIA_E_WITH_BOOST_MULTI)
  # boost::multi instead of mdspan
  include(FetchContent)
  FetchContent_Declare(
    multi
    GIT_REPOSITORY https://github.com/correaa/boost-multi.git
    GIT_TAG f549688718345435aabaa1ac49c392965ca082a5
    FIND_PACKAGE_ARGS #MULTI_BUILD_TESTS?
  )
  FetchContent_MakeAvailable(multi)
  FetchContent_GetProperties(multi SOURCE_DIR MULTI_CPP_INCLUDE_DIR BINARY_DIR MULTI_CPP_LIBRARY)
  # target_include_directories(dalotia_cpp PUBLIC ${MULTI_CPP_INCLUDE_DIR} ${MULTI_DIR})
  # target_link_libraries(dalotia_cpp multi)
  target_link_libraries(multi INTERFACE ${BLAS_LIBRARIES})
endif (DALOTIA_E_WITH_BOOST_MULTI)

find_package(OpenMP) #TODO this should not be always on?

add_subdirectory(cacheflush)
add_subdirectory(benchmarks)
