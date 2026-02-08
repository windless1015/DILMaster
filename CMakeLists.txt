cmake_minimum_required(VERSION 3.18)

project(DILMaster LANGUAGES CXX CUDA)

# ------------------------------------------------------------------------------
# Policies
# ------------------------------------------------------------------------------
cmake_policy(SET CMP0135 NEW)

# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------
option(BUILD_APPS "Build application executables" ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING
      "CUDA architectures to build for (e.g. 75;86, native). Default = native" FORCE)
endif()

# ------------------------------------------------------------------------------
# Language standards
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------------------
# CUDA Toolkit
# ------------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

# ------------------------------------------------------------------------------
# toml++ (local thirdparty preferred, fallback to FetchContent)
# ------------------------------------------------------------------------------
set(TOMLPP_LOCAL_DIR "${CMAKE_SOURCE_DIR}/thirdparty/tomlplusplus")
set(TOMLPP_LOCAL_HEADER "${TOMLPP_LOCAL_DIR}/include/toml++/toml.hpp")

if(EXISTS "${TOMLPP_LOCAL_HEADER}")
  add_subdirectory("${TOMLPP_LOCAL_DIR}" "${CMAKE_BINARY_DIR}/_deps/tomlplusplus-build")
else()
  include(FetchContent)
  FetchContent_Declare(
    tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG v3.4.0
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(tomlplusplus)
endif()

if(NOT TARGET tomlplusplus::tomlplusplus)
  if(TARGET tomlplusplus_tomlplusplus)
    add_library(tomlplusplus::tomlplusplus ALIAS tomlplusplus_tomlplusplus)
  elseif(TARGET tomlplusplus)
    add_library(tomlplusplus::tomlplusplus ALIAS tomlplusplus)
  else()
    message(FATAL_ERROR "toml++ target not found after add_subdirectory/FetchContent.")
  endif()
endif()

# ------------------------------------------------------------------------------
# Physics Library (LBM Module)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Physics Library (LBM + IBM + DEM Modules)
# ------------------------------------------------------------------------------
set(PHYSICS_SOURCES
  src/physics/lbm/LBMSolver.cu
  src/physics/lbm/LBMCore.cu
  src/physics/lbm/FreeSurfaceModule.cu
  src/physics/lbm/cuda/LBMBackend.cu
  src/physics/lbm/cuda/LBMKernels.cu
  src/physics/lbm/cuda/LBMDeviceConstants.cu
  # DEM Sources
  src/physics/dem/DEMCore.cu
  src/physics/dem/DEMSolver.cu
  # IBM Sources
  src/physics/ibm/IBMSolver.cu
  src/physics/ibm/IBMCore.cu
  src/physics/ibm/IBMMotion.cpp
  
  # Geometry
  src/geometry/STLGeometryLoader.cpp
)

add_library(physics ${PHYSICS_SOURCES})

target_include_directories(physics PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/src/core
  ${CMAKE_CURRENT_SOURCE_DIR}/src/physics
  ${CMAKE_CURRENT_SOURCE_DIR}/src/physics/lbm
  ${CMAKE_CURRENT_SOURCE_DIR}/src/physics/lbm/cuda
  ${CMAKE_CURRENT_SOURCE_DIR}/src/physics/dem
)

target_link_libraries(physics PUBLIC
  CUDA::cudart
  tomlplusplus::tomlplusplus
)

set_target_properties(physics PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# ------------------------------------------------------------------------------
# Services Library (VTK output)
# ------------------------------------------------------------------------------
add_library(services
  src/services/VTKService.cpp
)

target_include_directories(services PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/src/services
)

target_link_libraries(services PUBLIC
  physics
  tomlplusplus::tomlplusplus
)

# ------------------------------------------------------------------------------
# Test Suites
# ------------------------------------------------------------------------------
if(BUILD_APPS)
  add_subdirectory(lbm_test)
  add_subdirectory(ib_test)
  add_subdirectory(dem_test)
endif()
