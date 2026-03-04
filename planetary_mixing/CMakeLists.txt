# planetary_mixing/CMakeLists.txt

add_executable(planetary_paddle_mesh
  main.cpp
  PlanetaryPaddleModule.cpp
)

set_target_properties(planetary_paddle_mesh PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

target_include_directories(planetary_paddle_mesh PRIVATE
  ${CMAKE_SOURCE_DIR}/src/geometry
)

target_link_libraries(planetary_paddle_mesh PRIVATE
  CUDA::cudart
)

# Inject project root so code can find tools/ without absolute paths
target_compile_definitions(planetary_paddle_mesh PRIVATE
  PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}"
)

# Copy STL files to build directory for convenience
add_custom_command(TARGET planetary_paddle_mesh POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:planetary_paddle_mesh>/tools"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/tools/pp_close.stl"
    "$<TARGET_FILE_DIR:planetary_paddle_mesh>/tools/pp_close.stl"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/tools/pp_far.stl"
    "$<TARGET_FILE_DIR:planetary_paddle_mesh>/tools/pp_far.stl"
)
