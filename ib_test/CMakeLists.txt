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

target_compile_features(test_ibm_mdf PRIVATE cxx_std_17)

target_link_libraries(test_ibm_mdf PRIVATE 
    GTest::gtest 
    GTest::gtest_main
)

gtest_discover_tests(test_ibm_mdf)
