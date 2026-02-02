# lbm_test/sloshing/CMakeLists.txt

add_executable(sloshing
  main.cu
)

target_include_directories(sloshing PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/lbm_test/common
)

target_link_libraries(sloshing PRIVATE
  lbm_test_common
)

set_target_properties(sloshing PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
