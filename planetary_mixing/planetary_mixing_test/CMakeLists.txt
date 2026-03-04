# planetary_mixing/planetary_mixing_test/CMakeLists.txt

add_executable(planetary_mixing_test
  planetary_mixing_test.cu
  CylinderDomain.hpp
  ../planetary_spinning_test/PlanetaryPaddleModule.cpp
  ../planetary_spinning_test/PlanetaryPaddleModule.hpp
)

set_target_properties(planetary_mixing_test PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CXX_STANDARD 17
)

target_include_directories(planetary_mixing_test PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/geometry
  ${CMAKE_SOURCE_DIR}/planetary_mixing/planetary_spinning_test
)

target_link_libraries(planetary_mixing_test PRIVATE
  physics
  services
  ${CUDA_LIBRARIES}
)

target_compile_definitions(planetary_mixing_test PRIVATE
  IBM_TESTING
  PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}"
)
