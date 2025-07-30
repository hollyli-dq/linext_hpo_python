# Approximate Counting of Linear Extensions in Practice

This package contains the code used for the experiments in the paper "Approximate Counting of Linear Extensions in Practice" (unpublished manuscript, under review) by Topi Talvitie and Mikko Koivisto (University of Helsinki).

- The `src` directory contains the C++ code for the linear extension counting algorithms.
- The `instances` directory contains Python scripts and source data for experiment instance generation.

## Building

To compile the source code into an executable named `linext`, run

```
make -j5 linext
```

(The `-j` argument specifies the number of parallel compilation tasks.) The source is compiled using GCC by default, but Clang is also supported; to use it, append `CXX=clang++` to the command. For the compilation to succeed, the Boost C++ libraries need to be installed (more exactly, the `gamma_q` function from the header-only Boost.Math library is required). A debug executable `linext.debug` can be built using command `make -j5 linext.debug`. The code is compiled for the native CPU architecture, and thus the generated binary may not work on machines with CPUs different from that of the compiling machine.

By default, the CUDA GPU algorithm implementation of the relaxation TPA algorithm is not compiled. To enable it, append `USE_CUDA=yes` to the `make` command. The CUDA code is optimized for the NVIDIA Turing architecture, but the code generation target can be set using the `CUFLAGS_GENCODE` variable: for example, to generate code for the Volta architecture, append `CUFLAGS_GENCODE=-gencode=arch=compute_70,code=sm_70` (the default value is `-gencode=arch=compute_75,code=sm_75`). For the `USE_CUDA` and `CUFLAGS_GENCODE` changes to take effect, you need to run `make clean` to clean the build output before recompiling.

To generate the poset instances as `.txt` files to the `instances` directory, ensure that Python 3 is installed and run

```
make instances
```

## Usage

The `linext` program takes four command line arguments:
- The poset filename.
- The algorithm to use (run `linext` without arguments to get a list of compiled algorithms).
- The epsilon parameter: the maximum relative error.
- The delta parameter: the maximum probability of exceeding relative error of epsilon.

The `linext` program reads the poset from a file in an adjacency matrix format. Consider, for example, the following poset:

```
1 ---> 2 ---> 3
       |      |
       v      v
4 ---> 5 ---> 6
```

The adjacency matrix representation of this 6-element poset is a 6x6 binary matrix where we have 1 on row i and column j if there is a constraint that i must be before j:

```
0 1 0 0 0 0
0 0 1 0 1 0
0 0 0 0 0 1
0 0 0 0 1 0
0 0 0 0 0 1
0 0 0 0 0 0
```

If we save this matrix to `poset.txt`, then we can count its linear extensions for example by running

```
./linext poset.txt exact 0.01 0.01
```

This command uses the `exact` algorithm (exact dynamic programming) and thus ignores the epsilon and delta parameters. The output of the command contains a log in a simple machine-readable text format with timestamps and information about the various phases of the algorithm. The result is given by the `LINEXT_LOG_COUNT` line at the end of the log, for example

```
0.002787 0.0027861 -- LINEXT_LOG_COUNT 1.94591014906
```

The number `1.94591014906` is the natural logarithm of the number of linear extensions (we use logarithms to avoid problems with large numbers). `exp(1.94591014906)` is 7 (with some rounding error), which means that this poset has 7 linear extensions. To use the telescope product Brightwell-Winkler estimator with the Gibbs linear extension sampler and epsilon and delta set to 0.01, run

```
./linext poset.txt telescope-basic-gibbs 0.01 0.01
```

The `LINEXT_LOG_COUNT` entry in the output is the logarithm of the approximate result that should be within 1% relative error of the exact value with 99% probability.

The following algorithms are supported:
- `exact`: Exact dynamic programming (Kangas et al., 2016).
- `armc`: Adaptive relaxation Monte Carlo.
- `relaxtpa-loose2`: Relaxation TPA using analysis based on Chernoff bounds.
- `relaxtpa-loose1`: Variant of `relaxtpa-loose2` using a slightly looser bounds from the arXiv version https://arxiv.org/pdf/1010.4981.pdf of Banks et al. (2018).
- `relaxtpa`: Relaxation TPA using the enhanced TPA analysis.
- `relaxtpa-avx2`: Optimized implementation of `relaxtpa` using AVX2 instructions.
- `relaxtpa-avx512`: Optimized implementation of `relaxtpa` using AVX512 instructions. Only available if `linext` was compiled on a CPU with AVX512 support.
- `relaxtpa-avx512-short`: Optimized implementation of `relaxtpa` using AVX512 instructions with 256-bit vectors instead of the 512-bit vectors used in `relaxtpa-avx512`. Only available if `linext` was compiled on a CPU with AVX512 support.
- `relaxtpa-gpu`: Optimized implementation of `relaxtpa` using GPU. Only available if `linext` was compiled with `USE_CUDA=yes`.
- `telescope-basic-swap`: Basic telescope product Brightwell-Winkler estimator using the bounding chain due to Huber (2006) for the Karzanov-Khachiyan chain.
- `telescope-basic-gibbs`: Basic telescope product Brightwell-Winkler estimator using the Gibbs sampler due to Huber (2014).
- `telescope-decomposition-gibbs`: The enhanced Telescope Tree scheme using the Gibbs sampler due to Huber (2014).

The algorithm names used in the Experimental Results section of the paper correspond to the names in the `linext` program as follows:
- `ARMC`: `armc`
- `BasicTelescope`: `telescope-basic-gibbs`
- `Telescope`: `telescope-decomposition-gibbs`
- `ChernoffTPA`: `relaxtpa-loose2`
- `TPA`: `relaxtpa`
- `AVX2TPA`: `relaxtpa-avx2`
- `AVX512TPA`: `relaxtpa-avx512`
- `GPUTPA`: `relaxtpa-gpu`

## Licenses

The linear extension counting algorithms in the `src` directory were written by Topi Talvitie while employed at the University of Helsinki and are licensed under the MIT license. See `src/LICENSE` for more information.

The Bayesian networks in the `instances/networks` directory were obtained from the Bayesian Network Repository in https://www.bnlearn.com/bnrepository/ by Marco Scutari licensed under the Creative Commons Attribution-Share Alike License https://creativecommons.org/licenses/by-sa/3.0/.
