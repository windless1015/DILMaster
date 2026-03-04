# planetary_mixing/planetary_spinning_test/CMakeLists.txt

add_executable(planetary_spinning_test
  planetary_spinning_test.cpp
  PlanetaryPaddleModule.cpp
)

set_target_properties(planetary_spinning_test PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

target_include_directories(planetary_spinning_test PRIVATE
  ${CMAKE_SOURCE_DIR}/src/geometry
)

target_link_libraries(planetary_spinning_test PRIVATE
  CUDA::cudart
)

# Inject project root so code can find tools/ without absolute paths
target_compile_definitions(planetary_spinning_test PRIVATE
  PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}"
)
