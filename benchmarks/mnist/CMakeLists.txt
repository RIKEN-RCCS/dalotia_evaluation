add_executable( mnist_cpp mnist.cpp )
target_link_libraries( mnist_cpp PUBLIC dalotia_cpp )
# use openblas header
target_include_directories( mnist_cpp PUBLIC ${OpenBLAS_INCLUDE_DIR})
target_link_libraries(mnist_cpp PRIVATE BLAS::BLAS)
if (DALOTIA_E_WITH_BOOST_MULTI)
    target_compile_options( mnist_cpp PUBLIC "-DDALOTIA_E_WITH_BOOST_MULTI")
    target_include_directories( mnist_cpp PUBLIC ${MULTI_CPP_INCLUDE_DIR} ${MULTI_DIR})
    message(STATUS "multi: ${MULTI_DIR}")
    add_dependencies( mnist_cpp multi ) #TODO tblis?
endif(DALOTIA_E_WITH_BOOST_MULTI)
if (DALOTIA_E_WITH_NDIRECT)
    target_compile_options( mnist_cpp PUBLIC "-DDALOTIA_E_WITH_NDIRECT")
    target_include_directories( mnist_cpp PRIVATE ${ndirect_SOURCE_DIR} ${NDIRECT_CPP_INCLUDE_DIR})
    target_link_libraries( mnist_cpp PRIVATE ${NDIRECT_CPP_LIBRARY_DIR}/libnd.a )
    add_dependencies( mnist_cpp ndirect )
endif(DALOTIA_E_WITH_NDIRECT)

if (OpenMP_CXX_FOUND)
  target_link_libraries(mnist_cpp PUBLIC OpenMP::OpenMP_CXX)
endif ()
# add_test( mnist_cpp mnist_cpp ) #TODO

if (DALOTIA_E_WITH_FORTRAN)
    add_executable( mnist_fortran mnist.f03 ) #TODO this does not put the executable in the compile / link line??? -> try newer cmake maybe?
    # add sources
    # target_include_directories( mnist_fortran PUBLIC ${CMAKE_Fortran_MODULE_DIRECTORY})
    target_include_directories( mnist_fortran PUBLIC $<TARGET_PROPERTY:dalotia_fortran,Fortran_MODULE_DIRECTORY>)
    # set_target_properties(mnist_fortran PROPERTIES LANGUAGE Fortran)
    set_target_properties(mnist_fortran PROPERTIES LINKER_LANGUAGE Fortran)
    target_link_libraries( mnist_fortran dalotia_fortran )
endif (DALOTIA_E_WITH_FORTRAN)

# download the mnist test data
function(download_and_extract_gz target_name url hash)
    # extract the unzipped name
    string(REGEX MATCH "([^\/]+$)" downloaded_archive_name ${url})
    string(REGEX REPLACE ".gz$" "" downloaded_file_name ${downloaded_archive_name})

    # only download if not already downloaded
    if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/${downloaded_file_name})
        message(STATUS "Downloading ${downloaded_archive_name} from ${url}")
        FetchContent_Declare( ${target_name}
            URL ${url}
            URL_HASH SHA256=${hash}
            DOWNLOAD_NO_EXTRACT true
        )
        if(NOT ${target_name}_POPULATED)
            FetchContent_MakeAvailable(${target_name})
        endif()

        # file(ARCHIVE_EXTRACT INPUT ${${target_name}_SOURCE_DIR}/${downloaded_archive_name} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data) # <- doesn't work with .gz
        message(STATUS "unpack ${target_name}_SOURCE_DIR: ${${target_name}_SOURCE_DIR}/${downloaded_archive_name} at ${CMAKE_CURRENT_BINARY_DIR}")
        file(COPY "${${target_name}_SOURCE_DIR}/${downloaded_archive_name}" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
        execute_process(
            COMMAND "gzip" "-d" "${downloaded_archive_name}"
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            # COMMAND_ERROR_IS_FATAL ANY
        )
    endif(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/${downloaded_file_name})
endfunction()

download_and_extract_gz(data_mnist_test_labels https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6)
download_and_extract_gz(data_mnist_test_images https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz 8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6)

file(COPY ${DALOTIA_CPP_INCLUDE_DIR}/data/model-mnist.safetensors DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
