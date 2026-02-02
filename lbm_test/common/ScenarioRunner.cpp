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

double calculateTotalMass(const float *phi, const uint8_t *flags,
                          size_t nCells, float rho0) {
  double mass = 0.0;
  for (size_t i = 0; i < nCells; ++i) {
    if (flags[i] == TYPE_LIQUID || flags[i] == TYPE_INTERFACE) {
      mass += static_cast<double>(phi[i]) * rho0;
    }
  }
  return mass;
}

void printStatus(int step, const uint8_t *flags, size_t nCells, double mass,
                 double initialMass) {
  int liquid = 0;
  int interface_ = 0;
  int gas = 0;
  int solid = 0;
  for (size_t i = 0; i < nCells; ++i) {
    switch (flags[i]) {
    case TYPE_LIQUID:
      ++liquid;
      break;
    case TYPE_INTERFACE:
      ++interface_;
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
            << " I=" << interface_ << " G=" << gas << "  Mass="
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
  // VTK 用的别名字段（VTKService 期望 "flags" 和 "velocity"）
  fieldStore.create(FieldDesc{"flags", nCells, sizeof(uint8_t)});
  fieldStore.create(FieldDesc{"velocity", nCells * 3, sizeof(float)});
  ctx.fields = &fieldStore;

  // allocate() 会在 FieldStore 中创建 fluid.* 字段并初始化 LBMCore
  solver.allocate(ctx);
  solver.initialize(ctx);

  // prepareGeometry 回调设置初始几何
  // 注意：不调用 refreshDistributions()。初始分布函数在 LBMCore::initialize() 中
  // 已设为 f_eq(rho0, u0)，对所有单元一致。setRegion 只修改 flags/phi/mass，
  // 自由表面模块会在前几步自然调整界面处的分布函数。
  // 若调用 refreshDistributions()，kernel_initialize() 会按 flags 重新初始化分布，
  // 可能将 GAS 单元清零或按 phi 缩放 INTERFACE 单元，破坏初始平衡。
  if (spec.prepareGeometry && fsPtr) {
    spec.prepareGeometry(*fsPtr);
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

  // 读取初始状态（通过 backend 下载，因为还没执行 step）
  std::vector<float> h_rho(nCells);
  std::vector<float> h_u(nCells * 3);
  std::vector<uint8_t> h_flags(nCells);
  std::vector<float> h_phi(nCells);

  solver.getCore()->backend().download_fields(h_rho.data(), h_u.data(),
                                              h_flags.data(), h_phi.data());
  double initialMass = calculateTotalMass(h_phi.data(), h_flags.data(), nCells, cfg.rho0);
  printStatus(0, h_flags.data(), nCells, initialMass, initialMass);
  // Write initial state VTK
  if (vtkService) {
    {
      auto handle = fieldStore.get("flags");
      std::memcpy(handle.data(), h_flags.data(), nCells * sizeof(uint8_t));
    }
    {
      auto velHandle = fieldStore.get("fluid.velocity");
      auto vtkHandle = fieldStore.get("velocity");
      // Note: h_u is already in host memory, we can copy from there or from fieldStore
      // fieldStore "fluid.velocity" (if allocated on host) might not be populated yet if we only did download_fields into local h_u
      // But allocate() created fieldStore entries with device pointers.
      // vtkHandle is host-only alias.
      // We should copy from h_u to vtkHandle.
      std::memcpy(vtkHandle.data(), h_u.data(), nCells * 3 * sizeof(float));
    }
    vtkService->onStepEnd(ctx);
  }

  for (int step = 1; step <= spec.steps; ++step) {
    ctx.step = step - 1; // LBMCore internal step starts at 0
    ctx.time = step * spec.dt;
    solver.step(ctx); // step() 内部会同步 GPU → FieldStore 主机缓冲区

    if (step % spec.output_interval == 0 || step == spec.steps) {
      // solver.step() 已将数据同步到 FieldStore，直接读取
      auto flagsHandle = fieldStore.get("fluid.flags");
      auto phiHandle = fieldStore.get("fluid.phi");
      const uint8_t *flagsPtr = flagsHandle.as<uint8_t>();
      const float *phiPtr = phiHandle.as<float>();

      double currentMass = calculateTotalMass(phiPtr, flagsPtr, nCells, cfg.rho0);
      
      // Temporarily sync ctx.step with user-facing 'step' for output
      // This ensures:
      // 1. VTKService interval check passes (e.g. 100 % 100 == 0)
      // 2. Output filename uses user-facing step count (step_000100.vti)
      ctx.step = step;
      
      printStatus(step, flagsPtr, nCells, currentMass, initialMass);

      // Update VTK alias fields and write
      if (vtkService) {
        {
          auto handle = fieldStore.get("flags");
          std::memcpy(handle.data(), flagsPtr, nCells * sizeof(uint8_t));
        }
        {
          auto velHandle = fieldStore.get("fluid.velocity");
          auto vtkHandle = fieldStore.get("velocity");
          std::memcpy(vtkHandle.data(), velHandle.data(), nCells * 3 * sizeof(float));
        }
        vtkService->onStepEnd(ctx);
      }
      
      // Restore physics step logic if we were continuing inside the loop
      // (not strictly necessary here as ctx.step is overwritten at top of loop, 
      // but good practice if code is added after)
      ctx.step = step - 1; 
    }
  }

  if (vtkService) {
    vtkService->finalize(ctx);
  }
  solver.finalize(ctx);
  std::cout << "=== Scenario complete ===" << std::endl << std::endl;
}

} // namespace lbm_test
