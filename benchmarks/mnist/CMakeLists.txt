add_executable( mnist_cpp mnist.cpp )
target_link_libraries( mnist_cpp dalotia_cpp )
target_include_directories( mnist_cpp PUBLIC ${SAFETENSORS_CPP_INCLUDE_DIR} ${safetensors-cpp_DIR})
if (DALOTIA_E_WITH_BOOST_MULTI)
    target_compile_options( mnist_cpp PUBLIC "-DDALOTIA_E_WITH_BOOST_MULTI")
    add_dependencies( mnist_cpp multi ) #tblis #ndirect
endif(DALOTIA_E_WITH_BOOST_MULTI)
if (DALOTIA_E_WITH_NDIRECT)
    target_compile_options( mnist_cpp PUBLIC "-DDALOTIA_E_WITH_NDIRECT")
    add_dependencies( mnist_cpp ndirect )
endif(DALOTIA_E_WITH_NDIRECT)
add_test( mnist_cpp mnist_cpp )

# download the mnist test data
# function(download_and_extract url hash dest) #TODO?
FetchContent_Declare(data_mnist_test
    URL https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
    URL_HASH SHA256=f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6
    DOWNLOAD_NO_EXTRACT true
)
if(NOT data_mnist_test_POPULATED)
    FetchContent_Populate(data_mnist_test)
endif()

# file(ARCHIVE_EXTRACT INPUT ${data_mnist_test_SOURCE_DIR}/t10k-labels-idx1-ubyte.gz DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data) # <- doesn't work with .gz
message(STATUS "data_mnist_test_SOURCE_DIR: ${data_mnist_test_SOURCE_DIR}/t10k-labels-idx1-ubyte.gz" "${CMAKE_CURRENT_SOURCE_DIR}")
file(COPY "${data_mnist_test_SOURCE_DIR}/t10k-labels-idx1-ubyte.gz" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
execute_process(
    COMMAND "gzip" "-d" "t10k-labels-idx1-ubyte.gz"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks/mnist
) #TODO test

file(COPY ${DALOTIA_CPP_INCLUDE_DIR}/data/model-mnist.safetensors DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
