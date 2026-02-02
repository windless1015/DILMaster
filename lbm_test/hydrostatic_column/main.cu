#include <cuda_runtime.h>
#include <cstdlib>

#include <filesystem>
#include <iostream>
#include "ScenarioRunner.hpp"

namespace fs = std::filesystem;

int main(int argc, char **argv) {
  lbm::LBMConfig config{};
  config.nx = 64;
  config.ny = 64;
  config.nz = 64;
  config.tau = 3.0f * 0.005f + 0.5f;
  config.rho0 = 1.0f;
  config.u0 = make_float3(0.0f, 0.0f, 0.0f);
  config.gravity = make_float3(0.0f, 0.0f, -0.0002f);
  config.sigma = 1.0e-4f;
  config.wallFlags = lbm::WALL_ALL;
  config.enableFreeSurface = true;

  int steps = 2500;
  if (argc > 1) {
    steps = std::atoi(argv[1]);
  }

  lbm_test::ScenarioConfig scenario{};
  scenario.name = "Hydrostatic Column";
  scenario.config = config;
  scenario.steps = steps;
  scenario.output_interval = 200;
  scenario.output_dir = "hydrostatic_output";
  
  if (!fs::exists(scenario.output_dir)) {
      fs::create_directories(scenario.output_dir);
      std::cout << "Created output directory: " << scenario.output_dir << std::endl;
  }
  scenario.prepareGeometry = [config](lbm::FreeSurfaceModule &fs) {
    fs.setRegion(0, config.nx - 1, 0, config.ny - 1, 0, config.nz - 1,
                 lbm::CellType::GAS, 0.0f, config.rho0);
    fs.setRegion(0, config.nx - 1, 0, config.ny - 1, 0, config.nz / 2 - 1,
                 lbm::CellType::LIQUID, 1.0f, config.rho0);
    fs.fixInterfaceLayer();
  };

  lbm_test::runScenario(scenario);
  return 0;
}
