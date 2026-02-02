#include <cuda_runtime.h>
#include <cstdlib>

#include <filesystem>
#include <iostream>
#include "ScenarioRunner.hpp"

namespace fs = std::filesystem;

int main(int argc, char **argv) {
  lbm::LBMConfig config{};
  config.nx = 80;
  config.ny = 32;
  config.nz = 32;
  config.tau = 3.0f * 0.005f + 0.5f;
  config.rho0 = 1.0f;
  config.u0 = make_float3(0.2f, 0.0f, 0.0f);
  config.gravity = make_float3(0.0f, 0.0f, -0.00015f);
  config.sigma = 1.0e-4f;
  config.wallFlags = lbm::WALL_ALL;
  config.enableFreeSurface = true;

  int steps = 10000;
  if (argc > 1) {
    steps = std::atoi(argv[1]);
  }

  lbm_test::ScenarioConfig scenario{};
  scenario.name = "Sloshing Tank";
  scenario.config = config;
  scenario.steps = steps;
  scenario.output_interval = 100;
  scenario.output_dir = "sloshing_output";
  
  if (!fs::exists(scenario.output_dir)) {
      fs::create_directories(scenario.output_dir);
      std::cout << "Created output directory: " << scenario.output_dir << std::endl;
  }
  scenario.prepareGeometry = [config](lbm::FreeSurfaceModule &fs) {
    fs.setRegion(0, config.nx - 1, 0, config.ny - 1, 0, config.nz - 1,
                 lbm::CellType::GAS, 0.0f, config.rho0);
    const int water_x = config.nx / 2;
    const int water_z = config.nz / 2;
    fs.setRegion(0, water_x - 1, 0, config.ny - 1, 0, water_z - 1,
                 lbm::CellType::LIQUID, 1.0f, config.rho0);
    fs.fixInterfaceLayer();
  };

  lbm_test::runScenario(scenario);
  return 0;
}
