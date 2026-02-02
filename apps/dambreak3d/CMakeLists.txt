# apps/dambreak3d/CMakeLists.txt
# Dam Break 3D Standalone Example

add_executable(dambreak3d
  main.cu
)

target_include_directories(dambreak3d PRIVATE
  ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(dambreak3d PRIVATE
  physics
  services
)

set_target_properties(dambreak3d PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
