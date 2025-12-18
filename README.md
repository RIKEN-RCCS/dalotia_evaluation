# Benchmark code for inference with [dalotia](https://github.com/RIKEN-RCCS/dalotia)

## Installation

Assuming you have a working C++/Fortran compiler and a BLAS installation in your path,
run the following:

```bash
git clone https://github.com/RIKEN-RCCS/dalotia_evaluation.git
cd dalotia_evaluation
mkdir -p build
cd build
cmake ..
make -j
```

This will put the executables in the build directory
(it will automatically pull dalotia and build it as dependency.)


## Running the benchmarks and postprocessing

[See this Zenodo repository](https://zenodo.org/records/15129650) for additional scripts and commands 
to perform the experiments [published at SCA'24](https://doi.org/10.1145/3773656.3773664).
