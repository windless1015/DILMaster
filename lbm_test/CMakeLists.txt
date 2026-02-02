# lbm_test tests
add_library(lbm_test_common
  common/ScenarioRunner.cpp
)

target_include_directories(lbm_test_common PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/common
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_link_libraries(lbm_test_common PUBLIC
  physics
  services
)

add_subdirectory(dambreak3d)
add_subdirectory(hydrostatic_column)
add_subdirectory(sloshing)
