# ib_test/CMakeLists.txt

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

# ==============================================================================
# Existing Test: IBM Fundamentals
# ==============================================================================
add_executable(test_ibm_fundamentals
    test_ibm_fundamentals.cu
)

set_target_properties(test_ibm_fundamentals PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(test_ibm_fundamentals PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_features(test_ibm_fundamentals PRIVATE cxx_std_17)

target_link_libraries(test_ibm_fundamentals PRIVATE 
    GTest::gtest 
    GTest::gtest_main
)

gtest_discover_tests(test_ibm_fundamentals)

# ==============================================================================
# New Test: IBM MDF Regression
# ==============================================================================
add_executable(test_ibm_mdf
    test_ibm_mdf.cu
    ../src/physics/ibm/IBMCore.cu
)

set_target_properties(test_ibm_mdf PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(test_ibm_mdf PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_definitions(test_ibm_mdf PRIVATE IBM_TESTING)

# ==========================================
# IBM Couette 3D Regression Test
# ==========================================
# ==========================================
# IBM Couette 3D Regression Test
# ==========================================

# Couette Flow
add_executable(ibm_couette3d ibm_couette3d.cu ../src/physics/ibm/IBMCore.cu)
set_target_properties(ibm_couette3d PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(ibm_couette3d PRIVATE physics ${CUDA_LIBRARIES})
target_include_directories(ibm_couette3d PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/physics
    ${CMAKE_SOURCE_DIR}/src/physics/lbm
    ${CMAKE_SOURCE_DIR}/src/physics/lbm/cuda
)

# Poiseuille Flow
add_executable(ibm_poiseuille3d ibm_poiseuille3d.cu ../src/physics/ibm/IBMCore.cu)
set_target_properties(ibm_poiseuille3d PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(ibm_poiseuille3d PRIVATE physics ${CUDA_LIBRARIES})
target_include_directories(ibm_poiseuille3d PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/physics
    ${CMAKE_SOURCE_DIR}/src/physics/lbm
    ${CMAKE_SOURCE_DIR}/src/physics/lbm/cuda
)
target_compile_definitions(ibm_poiseuille3d PRIVATE IBM_TESTING)

# Sphere Drag 3D
add_executable(ibm_sphere_drag3d ibm_sphere_drag3d.cu ../src/physics/ibm/IBMCore.cu)
set_target_properties(ibm_sphere_drag3d PROPERTIES CUDA_SEPARABLE_COMPILATION ON CXX_STANDARD 17)
target_link_libraries(ibm_sphere_drag3d PRIVATE physics ${CUDA_LIBRARIES})
target_include_directories(ibm_sphere_drag3d PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/physics
    ${CMAKE_SOURCE_DIR}/src/physics/lbm
    ${CMAKE_SOURCE_DIR}/src/physics/lbm/cuda
)
target_compile_definitions(ibm_sphere_drag3d PRIVATE IBM_TESTING)

# Sphere Translate 3D
add_executable(ibm_sphere_translate3d 
    ibm_sphere_translate3d.cu 
    ../src/physics/ibm/IBMCore.cu
    ../src/services/VTKService.cpp
    ../src/services/MarkerVTKService.cpp
)
set_target_properties(ibm_sphere_translate3d PROPERTIES CUDA_SEPARABLE_COMPILATION ON CXX_STANDARD 17)
target_link_libraries(ibm_sphere_translate3d PRIVATE physics ${CUDA_LIBRARIES})
target_include_directories(ibm_sphere_translate3d PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/physics
    ${CMAKE_SOURCE_DIR}/src/physics/lbm
    ${CMAKE_SOURCE_DIR}/src/physics/lbm/cuda
)
target_compile_definitions(ibm_sphere_translate3d PRIVATE IBM_TESTING)
target_compile_definitions(ibm_couette3d PRIVATE IBM_TESTING)
set_target_properties(ibm_couette3d PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_features(test_ibm_mdf PRIVATE cxx_std_17)

target_link_libraries(test_ibm_mdf PRIVATE 
    GTest::gtest 
    GTest::gtest_main
)

gtest_discover_tests(test_ibm_mdf)
