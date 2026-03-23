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

If using a pre-installed dalotia (rather than FetchContent), point CMake to its install:

```bash
cmake .. \
  -Ddalotia_DIR=/path/to/dalotia/install/lib/cmake/dalotia
make -j
```

### GPU benchmarks (cuBLAS)

To build the cuBLAS GPU benchmark, you need the CUDA Toolkit (nvcc, cuBLAS) installed.
If dalotia was built with `-DDALOTIA_WITH_CUFILE=ON`, the benchmark will also attempt
to load model weights via GPU Direct Storage.

```bash
cmake .. -DDALOTIA_E_WITH_CUBLAS=ON
make -j
```

This produces the `subgridLES_cublas` executable alongside the CPU benchmarks.
Run it from its build directory:

```bash
cd build/benchmarks/SubgridLES
./subgridLES_cublas        # default: 4096 inputs, 1000 repetitions
./subgridLES_cublas 8192   # custom input count
```

### Additional CMake options

- `DALOTIA_E_WITH_FORTRAN`, default ON — build Fortran benchmarks
- `DALOTIA_E_WITH_CUBLAS`, default OFF — build cuBLAS GPU benchmarks (requires CUDA Toolkit)
- `DALOTIA_E_WITH_CACHEFLUSH`, default ON — flush CPU caches before measurement (x86 only)
- `DALOTIA_E_WITH_LIBTORCH`, default OFF — build libtorch comparison benchmarks
- `DALOTIA_E_WITH_ONEDNN`, default OFF — build oneDNN comparison benchmarks
- `BLA_VENDOR` — select BLAS implementation (e.g., `OpenBLAS`, `Intel10_64lp_seq`)

## Running the benchmarks and postprocessing

[See this Zenodo repository](https://zenodo.org/records/15129650) for additional scripts and commands 
to perform the experiments [published at SCA'26](https://doi.org/10.1145/3773656.3773664).
