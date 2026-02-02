#include "ScenarioRunner.hpp"
#include "services/VTKService.hpp"

#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

namespace lbm_test {
namespace {
constexpr uint8_t TYPE_SOLID = 0x01;
constexpr uint8_t TYPE_LIQUID = 0x08;
constexpr uint8_t TYPE_INTERFACE = 0x10;
constexpr uint8_t TYPE_GAS = 0x20;

double calculateTotalMass(const std::vector<float> &phi,
                          const std::vector<uint8_t> &flags, float rho0) {
  double mass = 0.0;
  for (size_t i = 0; i < flags.size(); ++i) {
    if (flags[i] == TYPE_LIQUID || flags[i] == TYPE_INTERFACE) {
      mass += static_cast<double>(phi[i]) * rho0;
    }
  }
  return mass;
}

void printStatus(int step, const std::vector<uint8_t> &flags, double mass,
                 double initialMass) {
  int liquid = 0;
  int interface = 0;
  int gas = 0;
  int solid = 0;
  for (auto flag : flags) {
    switch (flag) {
    case TYPE_LIQUID:
      ++liquid;
      break;
    case TYPE_INTERFACE:
      ++interface;
      break;
    case TYPE_GAS:
      ++gas;
      break;
    case TYPE_SOLID:
      ++solid;
      break;
    default:
      break;
    }
  }

  double diffPercent = (initialMass > 0.0)
                           ? (mass - initialMass) / initialMass * 100.0
                           : 0.0;

  std::cout << "Step " << std::setw(6) << step << ": L=" << liquid
            << " I=" << interface << " G=" << gas << "  Mass="
            << std::fixed << std::setprecision(4) << mass << " ("
            << std::showpos << std::setprecision(3) << diffPercent
            << std::noshowpos << "%)" << std::endl;
}

} // namespace

void runScenario(const ScenarioConfig &spec) {
  std::cout << "=== Scenario: " << spec.name << " ===" << std::endl;
  auto cfg = spec.config;
  const size_t nCells = static_cast<size_t>(cfg.nx) * cfg.ny * cfg.nz;

  LBMSolver solver;
  solver.setConfig(cfg);

  auto fsModule = std::make_unique<lbm::FreeSurfaceModule>();
  auto *fsPtr = fsModule.get();
  solver.addModule(std::move(fsModule));

  StepContext ctx;
  ctx.step = 0;
  ctx.time = 0.0;
  ctx.dt = spec.dt;

  FieldStore fieldStore;
  fieldStore.create(FieldDesc{"flags", nCells, sizeof(uint8_t)});
  fieldStore.create(FieldDesc{"velocity", nCells * 3, sizeof(float)});
  ctx.fields = &fieldStore;

  solver.allocate(ctx);
  solver.initialize(ctx);

  // Wire up FreeSurfaceModule's backend (required for setRegion to work:
  // without it, PHI/MASS handles remain null and setRegion silently no-ops)
  if (fsPtr) {
    fsPtr->setBackend(&solver.getCore()->backend());
    fsPtr->initialize(); // re-initialize to register PHI/MASS buffers
  }

  if (spec.prepareGeometry && fsPtr) {
    spec.prepareGeometry(*fsPtr, *solver.getMemoryManager());
    solver.getCore()->refreshDistributions();
  }

  // VTK output (optional, enabled when output_dir is non-empty)
  std::unique_ptr<VTKService> vtkService;
  if (!spec.output_dir.empty()) {
    VTKService::Config vtkCfg;
    vtkCfg.output_dir = spec.output_dir;
    vtkCfg.interval = spec.output_interval;
    vtkCfg.fields = {"flags"};
    vtkCfg.nx = cfg.nx;
    vtkCfg.ny = cfg.ny;
    vtkCfg.nz = cfg.nz;
    vtkService = std::make_unique<VTKService>(vtkCfg);
    vtkService->initialize(ctx);
  }

  std::vector<float> h_rho(nCells);
  std::vector<float> h_u(nCells * 3);
  std::vector<uint8_t> h_flags(nCells);
  std::vector<float> h_phi(nCells);

  auto download_fields = [&]() {
    solver.getCore()->backend().download_fields(h_rho.data(), h_u.data(),
                                                h_flags.data(), h_phi.data());
  };

  download_fields();
  double initialMass = calculateTotalMass(h_phi, h_flags, cfg.rho0);
  printStatus(0, h_flags, initialMass, initialMass);

  for (int step = 1; step <= spec.steps; ++step) {
    ctx.step = step;
    ctx.time = step * spec.dt;
    solver.step(ctx);

    if (step % spec.output_interval == 0 || step == spec.steps) {
      download_fields();
      double currentMass = calculateTotalMass(h_phi, h_flags, cfg.rho0);
      printStatus(step, h_flags, currentMass, initialMass);

      // Update FieldStore and write VTK
      if (vtkService) {
        {
          auto handle = fieldStore.get("flags");
          std::memcpy(handle.data(), h_flags.data(), nCells * sizeof(uint8_t));
        }
        {
          auto handle = fieldStore.get("velocity");
          std::memcpy(handle.data(), h_u.data(), nCells * 3 * sizeof(float));
        }
        vtkService->onStepEnd(ctx);
      }
    }
  }

  if (vtkService) {
    vtkService->finalize(ctx);
  }
  solver.finalize(ctx);
  std::cout << "=== Scenario complete ===" << std::endl << std::endl;
}

} // namespace lbm_test
