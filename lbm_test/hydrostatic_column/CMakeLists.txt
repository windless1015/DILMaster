# lbm_test/hydrostatic_column/CMakeLists.txt

add_executable(hydrostatic_column
  main.cu
)

target_include_directories(hydrostatic_column PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/lbm_test/common
)

target_link_libraries(hydrostatic_column PRIVATE
  lbm_test_common
)

set_target_properties(hydrostatic_column PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
