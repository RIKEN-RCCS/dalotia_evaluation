if (DALOTIA_E_WITH_LIBTORCH)
    add_executable( subgridLES_cpp subgridLES.cpp )
    target_link_libraries( subgridLES_cpp PUBLIC dalotia_cpp )
    target_link_libraries(subgridLES_cpp PRIVATE "${TORCH_LIBRARIES}")
    add_dependencies( subgridLES_cpp torch )
endif(DALOTIA_E_WITH_LIBTORCH)

if (DALOTIA_E_WITH_FORTRAN)
    add_executable( subgridLES_fortran subgridLES.f03 )
    # add sources
    target_include_directories( subgridLES_fortran PUBLIC $<TARGET_PROPERTY:dalotia_fortran,Fortran_MODULE_DIRECTORY>)
    set_target_properties(subgridLES_fortran PROPERTIES LINKER_LANGUAGE Fortran)
    target_link_libraries(subgridLES_fortran PRIVATE BLAS::BLAS)
    target_link_libraries( subgridLES_fortran PUBLIC dalotia_fortran )
endif (DALOTIA_E_WITH_FORTRAN)

if (OpenMP_CXX_FOUND)
  target_link_libraries(subgridLES_fortran PUBLIC OpenMP::OpenMP_CXX)
endif ()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/input_SubgridLESNet.safetensors DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/output_SubgridLESNet.safetensors DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/weights_SubgridLESNet.safetensors DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/traced_SubgridLESNet.pt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
