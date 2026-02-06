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
# Add test to CTest
add_test(NAME dem_test COMMAND dem_test)

# Sedimentation Validation Test
add_executable(lbm_dem_sedimentation_test
    lbm_dem_sedimentation/lbm_dem_sedimentation_test.cu
)
target_link_libraries(lbm_dem_sedimentation_test PRIVATE physics)
set_target_properties(lbm_dem_sedimentation_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;70;75;80"
)
