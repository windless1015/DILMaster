#pragma once

#include "core/FieldStore.hpp"
#include "core/StepContext.hpp"
#include "physics/lbm/FreeSurfaceModule.hpp"
#include "physics/lbm/LBMCore.hpp"
#include "physics/lbm/LBMSolver.hpp"
#include "physics/lbm/LBMConfig.hpp"

#include <functional>
#include <string>

namespace lbm_test {

struct ScenarioConfig {
  std::string name;
  lbm::LBMConfig config;
  int steps = 2000;
  int output_interval = 100;
  float dt = 1.0f;
  std::string output_dir; // VTK output directory (empty = no VTK output)
  std::function<void(lbm::FreeSurfaceModule &)> prepareGeometry;
};

void runScenario(const ScenarioConfig &config);

} // namespace lbm_test
