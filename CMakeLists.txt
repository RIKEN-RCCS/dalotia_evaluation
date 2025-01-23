cmake_minimum_required(VERSION 3.24)

project(dalotia_evaluation)

option(DALOTIA_E_WITH_BOOST_MULTI "include boost::multi for tensor operations" ON)
option(DALOTIA_E_WITH_NDIRECT "include ndirect for convolution operations on ARM" ON)
option(DALOTIA_E_WITH_LIBTORCH "include libtorch for jitted tensor operations" ON)
option(DALOTIA_E_WITH_FORTRAN "Build Fortran interface" ON)
if (DALOTIA_E_WITH_FORTRAN)
    enable_language(Fortran)
    set( CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include)
    install(DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
endif (DALOTIA_E_WITH_FORTRAN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MAKE_PROGRAM "make")

message(STATUS "CMAKE BUILD TYPE: ${CMAKE_BUILD_TYPE}")

include(ExternalProject)
include(FetchContent)
FetchContent_Declare(
  dalotia
  GIT_REPOSITORY https://github.com/RIKEN-RCCS/dalotia.git #TODO replace w/ https
  GIT_TAG main # TODO replace with release tag
  # CMAKE_ARGS  # I believe these are ignored, set like below
  #       -D DALOTIA_BUILD_TESTS=ON
  # FIND_PACKAGE_ARGS
  OVERRIDE_FIND_PACKAGE
)
set(DALOTIA_BUILD_TESTS OFF)
FetchContent_MakeAvailable(dalotia)
FetchContent_GetProperties(dalotia SOURCE_DIR DALOTIA_CPP_INCLUDE_DIR BINARY_DIR DALOTIA_CPP_LIBRARY)
message(STATUS "DALOTIA_CPP_INCLUDE_DIR: ${DALOTIA_CPP_INCLUDE_DIR}")
message(STATUS "DALOTIA_CPP_LIBRARY: ${DALOTIA_CPP_LIBRARY}")
find_package(dalotia)
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
set(BLA_VENDOR "OpenBLAS") # or: Fujitsu_SSL2...
find_package(BLAS)
if (NOT BLAS_FOUND)
  # try to find any BLAS
  unset(BLA_VENDOR)
  find_package(BLAS REQUIRED)
endif (NOT BLAS_FOUND)

if (NOT BLAS_INCLUDE_DIRS OR BLAS_INCLUDE_DIRS STREQUAL "")
  # list with blas libraries but get their parent directory
  SET(BLAS_LIBRARIES_PARENTS "")
   FOREACH(l BLAS_LIBRARIES)
      cmake_path(GET ${l} PARENT_PATH p)
      LIST(APPEND BLAS_LIBRARIES_PARENTS "${p}/..") #include?
   ENDFOREACH(l)
  message(WARNING "PARENTS ${BLAS_LIBRARIES_PARENTS}")
  find_path(BLAS_INCLUDE_DIRS cblas.h
  BLAS_LIBRARIES_PARENTS)
  message(WARNING "BLAS_INCLUDE_DIRS ${BLAS_INCLUDE_DIRS}")
endif()

# optional / conditional dependencies
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
  find_package (Torch PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../pytorch NO_DEFAULT_PATH)
  find_package(Torch REQUIRED)
  set(BUILD_SHARED_LIBS OFF)
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


add_subdirectory(benchmarks)


