cmake_minimum_required(VERSION 3.18)

# DEM Test Executable (Standalone, no GTest)
add_executable(dem_test
    test_dem_fundamentals.cu
)

# Link against DILMaster core physics libraries
target_link_libraries(dem_test PRIVATE
    physics
)

# Set CUDA properties
set_target_properties(dem_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;70;75;80"
)

# Add test to CTest
add_test(NAME dem_test COMMAND dem_test)
