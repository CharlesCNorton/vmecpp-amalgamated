# VMEC++ single-file amalgamation

The [VMEC++](https://github.com/proximafusion/vmecpp) 3D MHD equilibrium solver
merged into one C++ translation unit, `vmecpp_amalgamated.cc`. It is VMEC++'s own
source merged into a single file, not a reimplementation, so it builds the same
solver and writes the same output.

This is an unofficial redistribution and is not affiliated with or endorsed by
Proxima Fusion.

## What is in the file

`vmecpp_amalgamated.cc` contains the full solver: fixed and free boundary,
stellarator-symmetric and asymmetric (`lasym`) equilibria, all profile
parameterizations, and the complete output suite, plus the standalone
command-line `main()`. abscab (the Biot-Savart routines the free-boundary path
uses) is inlined as well.

Two optional paths that upstream keeps behind build defines are left out. The
FFTX/SPIRAL toroidal transform (`VMECPP_USE_FFTX`) is dropped, leaving the
partial-DFT routines VMEC++ falls back to when FFTX is off; upstream's
`fft_toroidal_test` is what checks the two against each other. The Enzyme
autodiff translation units (`VMECPP_ENABLE_ENZYME`) are dropped too, since they
compile only under a Clang/Enzyme plugin, and their call sites in
`ideal_mhd_model.cc` sit behind that same define.

Tests, benchmarks, mockups, the `makegrid` CLI, and the Python (pybind) module
are not included.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/vmecpp input.json [n_threads]    # writes input.out.h5
```

The provided `CMakeLists.txt` fetches the dependencies VMEC++ pins (Eigen 5.0.1,
abseil-cpp 20260107.1, nlohmann/json 3.11.3) and links the system HDF5 (C++
API), NetCDF and OpenMP. The first configure builds abseil from source, which
takes a few minutes. abseil must be recent enough to provide `absl/log`, which
the version Ubuntu 22.04/24.04 ships in `libabsl-dev` is not, hence the pinned
fetch.

A direct compile without CMake works too, given the include and link flags for
those libraries, at `-std=c++20 -O3 -DNDEBUG -fno-math-errno -fopenmp
-DEIGEN_DONT_PARALLELIZE -DEIGEN_MAX_ALIGN_BYTES=32
-DEIGEN_MAX_STATIC_ALIGN_BYTES=32`.

## Run

`vmecpp` takes a VMEC++ JSON input and an optional thread count, and writes
`<input>.out.h5`:

```sh
./build/vmecpp w7x.json 8
```

Classic Fortran VMEC `input.*` namelists can be converted to VMEC++ JSON with
upstream's `indata2json` tool.

## How it is generated

`amalgamate.py` produces `vmecpp_amalgamated.cc` from a VMEC++ checkout and an
abscab checkout. Its translation-unit list mirrors the `vmecpp_sources` list
upstream's CMakeLists assemble. It inlines each project header once in
dependency order, leaves external includes in place, inlines
`abscab.hh`/`abscab.cc`, and drops the `VMECPP_USE_FFTX` branches.

```sh
git clone https://github.com/proximafusion/vmecpp
git clone https://github.com/jonathanschilling/abscab-cpp
git -C abscab-cpp checkout 5cfa473b90aab06d7f70d986da0c46c46c1ebe9c  # v1.0.3

python amalgamate.py \
    --cpp-root vmecpp/src/vmecpp/cpp \
    --abscab-root abscab-cpp \
    --out vmecpp_amalgamated.cc
```

## Verification

Checked against a conventional multi-file build of the same sources: upstream's
own CMake build at the same commit, configured with `-DVMECPP_USE_FFTX=OFF` so
both binaries take the partial-DFT path, both compiled by GCC 13.3 against the
pinned dependency versions. Every case in upstream's `test_data` was run
single-threaded by both binaries and the resulting HDF5 files compared dataset
by dataset with `tools/compare_outputs.py`.

All 16 cases came out bit-for-bit identical, 436 datasets each and 440 for the
two `lasym` cases, with no differing values. They cover
fixed boundary (`solovev`, `solovev_analytical`, `solovev_no_axis`,
`circular_tokamak`, `cma`, `near_axis_iota_nfp4`, `li383_low_res`,
`cth_like_fixed_bdy`, `cth_like_fixed_bdy_nzeta_37`,
`cth_like_fixed_bdy_spline_pressure`, `up_down_asym`) and free boundary
(`solovev_free_bdy`, `solovev_free_bdy_lforbal`, `cth_like_free_bdy`,
`cth_like_free_bdy_multigrid`, `cth_like_free_bdy_asym`), and between them the
asymmetric solver, spline-parameterized pressure, multigrid `ns` sequences, the
guessed magnetic axis, and the free-boundary Nestor and abscab paths.

The file also compiles and runs under Clang 18.

Single-threaded runs are deterministic. With multiple threads, OpenMP reductions
sum in nondeterministic order, so the last bits can vary between runs, as in
upstream.

## License

VMEC++ and the tooling here are under the MIT License; the inlined abscab
routines are under the Apache License 2.0. See [LICENSE](LICENSE),
[NOTICE](NOTICE), and [THIRD_PARTY_LICENSES/](THIRD_PARTY_LICENSES).
