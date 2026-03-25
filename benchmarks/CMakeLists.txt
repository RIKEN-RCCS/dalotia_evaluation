if (TARGET BLAS::BLAS)
    add_subdirectory(mnist)
    add_subdirectory(DeepRLEddy)
    add_subdirectory(NeuralClosure)
    add_subdirectory(SubgridLES)
else()
    message(STATUS "Skipping CPU-only benchmarks (no host BLAS found)")
endif()

# GPT2 has both CPU and GPU targets; handles BLAS guard internally
add_subdirectory(GPT2)
