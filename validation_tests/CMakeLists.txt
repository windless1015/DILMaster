# validation_tests/CMakeLists.txt

# Validation Framework Library
add_library(validation_framework INTERFACE)

target_include_directories(validation_framework INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/core
  ${CMAKE_SOURCE_DIR}/src/physics
  ${CMAKE_SOURCE_DIR}/src/physics/lbm
  ${CMAKE_SOURCE_DIR}/src/physics/ibm
  ${CMAKE_SOURCE_DIR}/src/physics/dem
)

target_link_libraries(validation_framework INTERFACE
  physics
  services
)

# Validation Executor Executable
add_executable(validation_executor
  validation_executor.cpp
)

target_link_libraries(validation_executor PRIVATE
  validation_framework
)

set_target_properties(validation_executor PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  LINKER_LANGUAGE CUDA
)

# Demo Executable
add_executable(validation_demo
  demo.cpp
)

target_link_libraries(validation_demo PRIVATE
  validation_framework
)

set_target_properties(validation_demo PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  LINKER_LANGUAGE CUDA
)

# Quick Test Executable
add_executable(quick_dkt_test
  quick_dkt_test.cpp
)

target_link_libraries(quick_dkt_test PRIVATE
  validation_framework
)

set_target_properties(quick_dkt_test PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  LINKER_LANGUAGE CUDA
)

# Add tests to CTest
add_test(NAME validation_executor COMMAND validation_executor)
add_test(NAME validation_demo COMMAND validation_demo)
add_test(NAME quick_dkt_test COMMAND quick_dkt_test)
