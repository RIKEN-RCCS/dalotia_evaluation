cmake_minimum_required(VERSION 3.24)

project(dalotia_evaluation)

option(DALOTIA_E_WITH_BOOST_MULTI "include boost::multi for tensor operations" ON)
option(DALOTIA_E_WITH_NDIRECT "include ndirect for convolution operations on ARM" ON)
option(DALOTIA_E_WITH_FORTRAN "Build Fortran interface" ON)
if (DALOTIA_E_WITH_FORTRAN)
    enable_language(Fortran)
    set( CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include)
    install(DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
endif (DALOTIA_E_WITH_FORTRAN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MAKE_PROGRAM "make")

message(STATUS "CMAKE BUILD TYPE: ${CMAKE_BUILD_TYPE}")

include(FetchContent)
  FetchContent_Declare(
    dalotia
    GIT_REPOSITORY git@github.com:RIKEN-RCCS/dalotia.git #TODO replace w/ https
    GIT_TAG main # TODO replace with release tag
    # CMAKE_ARGS  # I believe these are ignored, set like below
    #       -D DALOTIA_BUILD_TESTS=ON
    FIND_PACKAGE_ARGS
  )
  set(DALOTIA_BUILD_TESTS OFF)
  FetchContent_MakeAvailable(dalotia)
  FetchContent_GetProperties(dalotia SOURCE_DIR DALOTIA_CPP_INCLUDE_DIR BINARY_DIR DALOTIA_CPP_LIBRARY)
  message(status "DALOTIA_CPP_INCLUDE_DIR: ${DALOTIA_CPP_INCLUDE_DIR}")
  message(status "DALOTIA_CPP_LIBRARY: ${DALOTIA_CPP_LIBRARY}")

if (DALOTIA_E_WITH_NDIRECT)
  include(ExternalProject)
  ExternalProject_Add(
    ndirect
    GIT_REPOSITORY https://github.com/nDIRECT/nDIRECT.git
    GIT_TAG 01ad5b38c5b6187018e713c644eb5018babc30b7
    BUILD_IN_SOURCE 1
    SOURCE_DIR ${ndirect_SOURCE_DIR} #/direct_cnn
    CONFIGURE_COMMAND ""
    BUILD_COMMAND cd direct_cnn && make -j 1
    INSTALL_COMMAND ""
  )
  ExternalProject_Get_Property(ndirect SOURCE_DIR)
  message(STATUS "ndirect_SOURCE_DIR: ${SOURCE_DIR}") #TODO test on fugaku
endif (DALOTIA_E_WITH_NDIRECT)


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
  target_include_directories(dalotia_cpp PUBLIC ${MULTI_CPP_INCLUDE_DIR} ${MULTI_DIR})
  target_link_libraries(dalotia_cpp multi)

  # download the mnist test data
  # function(download_and_extract url dest) #TODO?
  FetchContent_Declare(data_mnist_test
    URL https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
    URL_HASH SHA256=f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6
    DOWNLOAD_NO_EXTRACT true
  )
  if(NOT data_mnist_test_POPULATED)
    FetchContent_Populate(data_mnist_test)
  endif()
  file(ARCHIVE_EXTRACT INPUT ${data_mnist_test_SOURCE_DIR}/t10k-labels-idx1-ubyte.gz DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/mnist/data)

endif (DALOTIA_E_WITH_BOOST_MULTI)

find_package(OpenMP) #TODO if this should not be always on, parameterize in dalotia as well
if (OpenMP_CXX_FOUND)
  target_link_libraries(dalotia_cpp OpenMP::OpenMP_CXX)
endif ()

find_package(BLAS REQUIRED BLA_STATIC)

add_subdirectory(benchmarks)


