# VMEC++ single-file amalgamation

The [VMEC++](https://github.com/proximafusion/vmecpp) 3D MHD equilibrium solver
merged into one C++ translation unit, `vmecpp_amalgamated.cc`. It is VMEC++'s own
source merged into a single file, not a reimplementation, so it builds the same
solver and writes the same output.

This is an unofficial redistribution and is not affiliated with or endorsed by
Proxima Fusion.

## What is in the file

`vmecpp_amalgamated.cc` contains the full solver: fixed and free boundary, all
profile parameterizations, and the complete output suite, plus the standalone
command-line `main()`. abscab (the Biot-Savart routines the free-boundary path
uses) is inlined as well.

The optional FFTX/SPIRAL kernel path (`VMECPP_USE_FFTX`) is omitted. VMEC++
falls back to a partial-DFT toroidal transform when FFTX is off, and that is the
path kept here, so the numerics are the ones upstream produces by default.

Tests, benchmarks, mockups, the `makegrid` CLI, and the Python (pybind) module
are not included.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/vmecpp input.json [n_threads]    # writes input.out.h5
```

The provided `CMakeLists.txt` fetches the dependencies VMEC++ pins (Eigen 5.0.1,
abseil-cpp @ `4447c756`, nlohmann/json 3.11.3) and links the system HDF5 (C++
API), NetCDF, LAPACK and OpenMP. The first configure builds abseil from source,
which takes a few minutes. abseil must be recent enough to provide `absl/log`;
the version Ubuntu 22.04/24.04 ships in `libabsl-dev` is too old, which is why
the build fetches a pinned commit.

A direct compile without CMake works too, given the include and link flags for
those libraries, at `-std=c++20 -O3 -DNDEBUG -fno-math-errno -fopenmp
-DEIGEN_DONT_PARALLELIZE`.

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
abscab checkout. It inlines each project header once in dependency order, leaves
external includes in place, inlines `abscab.hh`/`abscab.cc`, and drops the
`VMECPP_USE_FFTX` branches.

```sh
git clone https://github.com/proximafusion/vmecpp
git clone https://github.com/jonathanschilling/abscab-cpp
git -C abscab-cpp checkout 5cfa473b90aab06d7f70d986da0c46c46c1ebe9c

python amalgamate.py \
    --cpp-root vmecpp/src/vmecpp/cpp \
    --abscab-root abscab-cpp \
    --out vmecpp_amalgamated.cc
```

## Verification

Checked against a conventional multi-file build of the same sources, compiled
with identical flags and pinned dependency versions, by running both
single-threaded and comparing the output HDF5 dataset by dataset
(`tools/compare_outputs.py`):

- `solovev` (fixed boundary): all 434 datasets identical.
- `cth_like_free_bdy` (free boundary): 433 of 434 identical; the remaining one
  is a diagnostic array element that is run-to-run nondeterministic in upstream
  VMEC++ as well.

Single-threaded runs are deterministic. With multiple threads, OpenMP reductions
sum in nondeterministic order, so the last bits can vary between runs, as in
upstream.

## License

VMEC++ and the tooling here are under the MIT License; the inlined abscab
routines are under the Apache License 2.0. See [LICENSE](LICENSE),
[NOTICE](NOTICE), and [THIRD_PARTY_LICENSES/](THIRD_PARTY_LICENSES).
