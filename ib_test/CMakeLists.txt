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

# Define the test executable
add_executable(test_ibm_fundamentals
    test_ibm_fundamentals.cu
)

# Enable CUDA language for the target (though project enables it globally)
set_target_properties(test_ibm_fundamentals PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


# Include directories
# We need to access src/physics/ibm/IBMKernel.h
target_include_directories(test_ibm_fundamentals PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

# Enable C++17
target_compile_features(test_ibm_fundamentals PRIVATE cxx_std_17)

# Link GTest
target_link_libraries(test_ibm_fundamentals PRIVATE 
    GTest::gtest 
    GTest::gtest_main
)

# Add to CTest
include(GoogleTest)
gtest_discover_tests(test_ibm_fundamentals)
