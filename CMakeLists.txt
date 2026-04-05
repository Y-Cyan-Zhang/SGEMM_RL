cmake_minimum_required(VERSION 3.18)
project(SGEMM_RL LANGUAGES CXX CUDA)

# Require C++17 for cooperative_groups
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Find required packages
find_package(CUDAToolkit REQUIRED)

# -----------------------------------------------------------------------
# Executable
# -----------------------------------------------------------------------
add_executable(sgemm
  main.cpp
  runner.cu
)

# Include the project headers from the source directory
target_include_directories(sgemm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link CUDA and cuBLAS
target_link_libraries(sgemm PRIVATE
  CUDA::cudart
  CUDA::cublas
)

# CUDA architecture — T4 = sm_75
# Override at configure time: cmake -DCMAKE_CUDA_ARCHITECTURES=75 ..
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# Compiler options
target_compile_options(sgemm PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    -O3
    --use_fast_math
  >
)

# -----------------------------------------------------------------------
# Install (optional)
# -----------------------------------------------------------------------
install(TARGETS sgemm DESTINATION bin)
