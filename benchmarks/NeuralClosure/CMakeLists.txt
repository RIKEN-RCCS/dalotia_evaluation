add_executable( neuralClosure_cpp neuralClosure.cpp )
target_link_libraries( neuralClosure_cpp PUBLIC dalotia::dalotia_cpp )
target_link_libraries( neuralClosure_cpp PRIVATE cacheflush )

if (DALOTIA_E_FOR_MEMORY_TRACE)
    target_compile_definitions(neuralClosure_cpp PRIVATE DALOTIA_E_FOR_MEMORY_TRACE)
endif()
if (DALOTIA_E_WITH_LIKWID_MARKERS)
  target_compile_definitions(neuralClosure_cpp PRIVATE LIKWID_PERFMON)
  target_include_directories(neuralClosure_cpp PRIVATE ${LIKWID_INCLUDE_DIR})
  target_link_libraries(neuralClosure_cpp PRIVATE likwid::likwid)
  add_dependencies(neuralClosure_cpp likwid::likwid)
endif(DALOTIA_E_WITH_LIKWID_MARKERS)

if (DALOTIA_E_WITH_CPPFLOW)
  add_dependencies( neuralClosure_cpp cppflow )
  target_include_directories( neuralClosure_cpp PUBLIC ${CPPFLOW_INCLUDE_DIR} )
  target_compile_options( neuralClosure_cpp PUBLIC "-DDALOTIA_E_WITH_CPPFLOW")
endif (DALOTIA_E_WITH_CPPFLOW)
target_include_directories( neuralClosure_cpp PUBLIC ${BLAS_INCLUDE_DIRS})
target_link_libraries(neuralClosure_cpp PRIVATE BLAS::BLAS)

# if (OpenMP_CXX_FOUND)
#   target_link_libraries(neuralClosure_fortran PUBLIC OpenMP::OpenMP_CXX)
# endif ()

# choose models here: https://github.com/KiT-RT/kitrt_code/tree/master/tfmodels
set(KITRT_TF_MODEL_NAME "Harmonic_Mk11_M3_2D_gamma3")
set(KITRT_TF_MODEL_PATH "${CMAKE_CURRENT_BINARY_DIR}/${KITRT_TF_MODEL_NAME}/")
file(MAKE_DIRECTORY ${KITRT_TF_MODEL_PATH})
file(DOWNLOAD
  https://github.com/KiT-RT/kitrt_code/raw/refs/heads/master/tfmodels/${KITRT_TF_MODEL_NAME}/best_model/saved_model.pb
  ${KITRT_TF_MODEL_PATH}/saved_model.pb
)
file(DOWNLOAD
  https://github.com/KiT-RT/kitrt_code/raw/refs/heads/master/tfmodels/${KITRT_TF_MODEL_NAME}/best_model/keras_metadata.pb
  ${KITRT_TF_MODEL_PATH}/keras_metadata.pb
)
file(DOWNLOAD
  https://github.com/KiT-RT/kitrt_code/raw/refs/heads/master/tfmodels/${KITRT_TF_MODEL_NAME}/best_model/variables/variables.index
  ${KITRT_TF_MODEL_PATH}/variables/variables.index
)
file(DOWNLOAD
  https://github.com/KiT-RT/kitrt_code/raw/refs/heads/master/tfmodels/${KITRT_TF_MODEL_NAME}/best_model/variables/variables.data-00000-of-00001
  ${KITRT_TF_MODEL_PATH}/variables/variables.data-00000-of-00001
)

file(COPY 
     ${CMAKE_CURRENT_SOURCE_DIR}/inputs1367.bin
     ${CMAKE_CURRENT_SOURCE_DIR}/outputs1367.bin
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})