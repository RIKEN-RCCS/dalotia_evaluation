add_executable( deeprleddy_cpp deeprleddy.cpp )
target_link_libraries( deeprleddy_cpp PUBLIC dalotia::dalotia_cpp )
target_link_libraries( deeprleddy_cpp PRIVATE cacheflush )

# use openblas header
target_include_directories( deeprleddy_cpp PUBLIC ${BLAS_INCLUDE_DIRS})
target_link_libraries(deeprleddy_cpp PRIVATE BLAS::BLAS)

if (DALOTIA_E_WITH_LIBTORCH)
    target_compile_options( deeprleddy_cpp PUBLIC "-DDALOTIA_E_WITH_LIBTORCH")
    target_link_libraries(deeprleddy_cpp PRIVATE "${TORCH_LIBRARIES}")
    add_dependencies( deeprleddy_cpp torch )
endif(DALOTIA_E_WITH_LIBTORCH)
if (DALOTIA_E_WITH_ONEDNN)
    target_compile_options( deeprleddy_cpp PUBLIC "-DDALOTIA_E_WITH_ONEDNN")
    target_link_libraries(deeprleddy_cpp PRIVATE dnnl)
endif(DALOTIA_E_WITH_ONEDNN)
if (DALOTIA_E_FOR_MEMORY_TRACE)
  target_compile_definitions(deeprleddy_cpp PRIVATE DALOTIA_E_FOR_MEMORY_TRACE)
endif(DALOTIA_E_FOR_MEMORY_TRACE)

if (DALOTIA_E_WITH_LIKWID_MARKERS)
  target_compile_definitions(deeprleddy_cpp PRIVATE LIKWID_PERFMON)
  target_include_directories(deeprleddy_cpp PUBLIC ${LIKWID_INCLUDE_DIR})
  target_link_libraries(deeprleddy_cpp PRIVATE likwid::likwid)
  add_dependencies(deeprleddy_cpp likwid::likwid)
endif(DALOTIA_E_WITH_LIKWID_MARKERS)

if (DALOTIA_E_WITH_FORTRAN)
    add_executable( deeprleddy_fortran deeprleddy.f90 )
    set_source_files_properties(
      deeprleddy.f90
      PROPERTIES Fortran_PREPROCESS ON
    )
    target_include_directories( deeprleddy_fortran PUBLIC $<TARGET_PROPERTY:dalotia::dalotia_fortran,Fortran_MODULE_DIRECTORY>)
    set_target_properties(deeprleddy_fortran PROPERTIES LINKER_LANGUAGE Fortran)
    target_link_libraries(deeprleddy_fortran PRIVATE BLAS::BLAS)
    target_link_libraries( deeprleddy_fortran PUBLIC dalotia::dalotia_fortran )
    target_link_libraries( deeprleddy_fortran PRIVATE cacheflush )
    if (DALOTIA_E_FOR_MEMORY_TRACE)
        target_compile_definitions(deeprleddy_fortran PRIVATE DALOTIA_E_FOR_MEMORY_TRACE)
    endif()
    if (DALOTIA_E_WITH_LIKWID_MARKERS)
      target_compile_definitions(deeprleddy_fortran PRIVATE LIKWID_PERFMON)
      target_include_directories(deeprleddy_fortran PRIVATE ${LIKWID_INCLUDE_DIR})
      target_link_libraries(deeprleddy_fortran PRIVATE likwid::likwid)
      add_dependencies(deeprleddy_fortran likwid::likwid)
    endif(DALOTIA_E_WITH_LIKWID_MARKERS)
endif (DALOTIA_E_WITH_FORTRAN)

# if (OpenMP_CXX_FOUND)
#   target_link_libraries(deeprleddy_cpp PUBLIC OpenMP::OpenMP_CXX)
# endif ()
# add_test( deeprleddy_cpp deeprleddy_cpp ) #TODO

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/input_DeepRLEddyNet.safetensors DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/output_DeepRLEddyNet.safetensors DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/weights_DeepRLEddyNet.safetensors DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/traced_DeepRLEddyNet.pt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
