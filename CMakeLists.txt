# Build the VMEC++ single-file amalgamation.
#
# Fetches the dependencies VMEC++ pins (Eigen 5.0.1, abseil-cpp @ 4447c756,
# nlohmann/json 3.11.3) so the result matches upstream bit-for-bit, and uses
# the system HDF5 (C++ API), NetCDF, LAPACK and OpenMP. abscab is already
# inlined into vmecpp_amalgamated.cc, so it is not fetched here.
#
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#   ./build/vmecpp input.json [n_threads]      # writes input.out.h5
#
# The first configure builds abseil from source (a few minutes).

cmake_minimum_required(VERSION 3.16)
project(vmecpp_amalgamated CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Match VMEC++'s Release flags so floating-point results are reproducible.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -fno-math-errno")

find_package(OpenMP REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C CXX)
find_package(LAPACK REQUIRED)

# netCDF: prefer its CMake package, fall back to pkg-config (as VMEC++ does).
find_package(netCDF QUIET)
if(NOT netCDF_FOUND)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(netCDF REQUIRED IMPORTED_TARGET netcdf>=4.3.0)
  set(NETCDF_LINK PkgConfig::netCDF)
else()
  set(NETCDF_LINK ${netCDF_LIBRARIES})
  include_directories(${netCDF_INCLUDE_DIRS})
endif()

include(FetchContent)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

# Eigen (header-only): pinned to the version VMEC++ fetches.
FetchContent_Declare(eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 5.0.1
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(eigen)
include_directories(${eigen_SOURCE_DIR})

# nlohmann/json.
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(nlohmann_json)

# abseil-cpp: pinned commit; must provide absl/log (apt's 2022 release does not).
# A system abseil new enough for absl/log also works via find_package(absl).
set(ABSL_PROPAGATE_CXX_STD ON)
set(ABSL_ENABLE_INSTALL OFF)
set(BUILD_TESTING OFF)
FetchContent_Declare(absl
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
  GIT_TAG 4447c7562e3bc702ade25105912dce503f0c4010
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(absl)

add_executable(vmecpp vmecpp_amalgamated.cc)

target_include_directories(vmecpp PRIVATE
  ${HDF5_INCLUDE_DIRS} ${HDF5_CXX_INCLUDE_DIRS})

# https://libeigen.gitlab.io/eigen/docs-3.3/TopicMultiThreading.html : VMEC++
# multithreads at the outer loop level, not inside individual Eigen operations.
target_compile_definitions(vmecpp PRIVATE EIGEN_DONT_PARALLELIZE)

target_link_libraries(vmecpp PRIVATE
  ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES}
  ${NETCDF_LINK}
  nlohmann_json::nlohmann_json
  LAPACK::LAPACK
  absl::algorithm absl::base absl::synchronization absl::strings
  absl::str_format absl::log absl::string_view absl::check
  absl::status absl::statusor
  OpenMP::OpenMP_CXX)
