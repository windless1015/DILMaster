/**
 * dambreak3d - Dam Break 3D Simulation (DILMaster Standalone)
 *
 * 支持两种运行模式:
 *   --mode legacy   : 原始手动组装模式 (默认)
 *   --mode scenario : 使用 ScenarioRunner 标准化流程
 *
 * 使用 LBMCore + FreeSurfaceModule 直接驱动三维溃坝场景模拟。
 * 输出 VTI 文件 (flags + velocity)，可在 ParaView 中可视化。
 */
#include <cuda_runtime.h>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "core/FieldStore.hpp"
#include "core/StepContext.hpp"
#include "physics/lbm/FreeSurfaceModule.hpp"
#include "physics/lbm/LBMConfig.hpp"
#include "physics/lbm/LBMCore.hpp"
#include "services/VTKService.hpp"

// ScenarioRunner 标准化路径
#include "ScenarioRunner.hpp"

namespace fs = std::filesystem;

// ============================================================================
// 共享配置
// ============================================================================
struct DamBreak3DConfig {
    // 网格尺寸
    int nx = 64;
    int ny = 96;
    int nz = 96;

    // 物理参数
    float nu = 0.005f;
    float gravity[3] = { 0.0f, 0.0f, -0.0002f };
    float sigma = 0.0001f;
    float rho0 = 1.0f;

    // 初始水块范围 (比例)
    float water_z_ratio = 6.0f / 8.0f; // z 方向水位高度
    float water_y_ratio = 1.0f / 8.0f; // y 方向水块宽度

    // 模拟控制
    int steps = 20000;
    std::string output_dir = "dambreak3d_output/";
    int output_interval = 100;
};

// ============================================================================
// 辅助函数
// ============================================================================
static void ensureOutputDir(const std::string& path) {
    if (!fs::exists(path)) {
        fs::create_directories(path);
        std::cout << "Created output directory: " << path << std::endl;
    }
}

static double calculateTotalMass(const std::vector<float>& phi,
    const std::vector<unsigned char>& flags,
    int nCells, float rho0) {
    double totalMass = 0.0;
    for (int i = 0; i < nCells; ++i) {
        // TYPE_F=0x08 (Liquid), TYPE_I=0x10 (Interface)
        if (flags[i] == 0x08 || flags[i] == 0x10) {
            totalMass += phi[i] * rho0;
        }
    }
    return totalMass;
}

// ============================================================================
// Mode 1: Legacy — 原始手动组装模式
// ============================================================================
static int runLegacy(const DamBreak3DConfig& cfg) {
    std::cout << "=== DILMaster Dam Break 3D [Legacy Mode] ===" << std::endl;

    // ------------------------------------------------------------------
    // 配置 LBM 参数
    // ------------------------------------------------------------------
    lbm::LBMConfig lbmConfig{};
    lbmConfig.nx = cfg.nx;
    lbmConfig.ny = cfg.ny;
    lbmConfig.nz = cfg.nz;
    lbmConfig.tau = 3.0f * cfg.nu + 0.5f;
    lbmConfig.rho0 = cfg.rho0;
    lbmConfig.u0 = make_float3(0.0f, 0.0f, 0.0f);
    lbmConfig.gravity =
        make_float3(cfg.gravity[0], cfg.gravity[1], cfg.gravity[2]);
    lbmConfig.sigma = cfg.sigma;
    lbmConfig.wallFlags = lbm::WALL_ALL;
    lbmConfig.enableFreeSurface = true;

    std::cout << "Grid: " << cfg.nx << " x " << cfg.ny << " x " << cfg.nz
        << std::endl;
    std::cout << "nu = " << cfg.nu << ", tau = " << lbmConfig.tau << std::endl;
    std::cout << "gravity = (" << cfg.gravity[0] << ", " << cfg.gravity[1]
        << ", " << cfg.gravity[2] << ")" << std::endl;
    std::cout << "sigma = " << cfg.sigma << std::endl;

    // ------------------------------------------------------------------
    // 创建 LBMCore（不再需要 LBMMemoryManager）
    // ------------------------------------------------------------------
    lbm::LBMCore lbmCore(lbmConfig);
    lbmCore.initialize();

    const int nCells = cfg.nx * cfg.ny * cfg.nz;

    // ------------------------------------------------------------------
    // 创建 FieldStore 并注册 fluid 字段（带 device pointer）
    // ------------------------------------------------------------------
    FieldStore fieldStore;
    const size_t n = static_cast<size_t>(nCells);
    fieldStore.create(FieldDesc{"fluid.density",  n,     sizeof(float),
                                 lbmCore.densityDevicePtr()});
    fieldStore.create(FieldDesc{"fluid.velocity", n * 3, sizeof(float),
                                 lbmCore.velocityDevicePtr()});
    fieldStore.create(FieldDesc{"fluid.flags",    n,     sizeof(uint8_t),
                                 lbmCore.flagsDevicePtr()});
    fieldStore.create(FieldDesc{"fluid.phi",      n,     sizeof(float),
                                 lbmCore.phiDevicePtr()});
    fieldStore.create(FieldDesc{"fluid.mass",     n,     sizeof(float),
                                 lbmCore.massDevicePtr()});
    // VTK 用的别名字段
    fieldStore.create(
        FieldDesc{ "flags", static_cast<size_t>(nCells), sizeof(unsigned char) });
    fieldStore.create(
        FieldDesc{ "velocity", static_cast<size_t>(nCells * 3), sizeof(float) });

    // ------------------------------------------------------------------
    // 自由表面模块：配置 + 分配 + 初始化
    // ------------------------------------------------------------------
    lbm::FreeSurfaceModule fsModule;
    fsModule.configure(lbmConfig);
    fsModule.allocate(fieldStore);
    fsModule.initialize(fieldStore);

    // 1) 全域设为 GAS
    fsModule.setRegion(0, cfg.nx - 1, 0, cfg.ny - 1, 0, cfg.nz - 1,
        lbm::CellType::GAS, 0.0f, cfg.rho0);

    // 2) 水块区域设为 LIQUID
    const int water_z = static_cast<int>(cfg.nz * cfg.water_z_ratio);
    const int water_y = static_cast<int>(cfg.ny * cfg.water_y_ratio);
    std::cout << "Water block: x=[0," << cfg.nx - 1 << "], y=[0," << water_y - 1
        << "], z=[0," << water_z - 1 << "]" << std::endl;

    fsModule.setRegion(0, cfg.nx - 1, 0, water_y - 1, 0, water_z - 1,
        lbm::CellType::LIQUID, 1.0f, cfg.rho0);

    // 3) 修复界面层
    fsModule.fixInterfaceLayer();

    // ------------------------------------------------------------------
    // VTKService 配置
    // ------------------------------------------------------------------
    VTKService::Config vtkConfig;
    vtkConfig.output_dir = cfg.output_dir;
    vtkConfig.interval = cfg.output_interval;
    vtkConfig.fields = { "flags" };
    vtkConfig.nx = cfg.nx;
    vtkConfig.ny = cfg.ny;
    vtkConfig.nz = cfg.nz;
    VTKService vtkService(vtkConfig);

    StepContext ctx;
    ctx.step = 0;
    ctx.time = 0.0;
    ctx.dt = 1.0;
    ctx.fields = &fieldStore;
    ctx.backend = &lbmCore.backend(); // 模块通过 ctx.backend 访问后端
    vtkService.initialize(ctx);

    // ------------------------------------------------------------------
    // 主机端缓冲区
    // ------------------------------------------------------------------
    std::vector<unsigned char> h_flags(nCells);
    std::vector<float> h_phi(nCells);
    std::vector<float> h_rho(nCells);
    std::vector<float> h_u(nCells * 3);

    // 下载初始数据统计
    lbmCore.backend().download_fields(h_rho.data(), h_u.data(), h_flags.data(),
        h_phi.data());
    const double initialMass =
        calculateTotalMass(h_phi, h_flags, nCells, cfg.rho0);

    int initLiquid = 0, initInterface = 0, initGas = 0, initSolid = 0;
    for (int i = 0; i < nCells; ++i) {
        if (h_flags[i] == 0x08)
            initLiquid++;
        else if (h_flags[i] == 0x10)
            initInterface++;
        else if (h_flags[i] == 0x20)
            initGas++;
        else if (h_flags[i] == 0x01)
            initSolid++;
    }

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Initial State:" << std::endl;
    std::cout << "  Liquid: " << initLiquid << ", Interface: " << initInterface
        << ", Gas: " << initGas << ", Solid: " << initSolid << std::endl;
    std::cout << "  Total Mass: " << std::fixed << std::setprecision(6)
        << initialMass << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // ------------------------------------------------------------------
    // 模拟主循环
    // ------------------------------------------------------------------
    std::cout << "Starting simulation for " << cfg.steps << " steps..."
        << std::endl;

    for (int step = 0; step <= cfg.steps; ++step) {
        ctx.step = step;
        ctx.time = step * ctx.dt;

        if (step % cfg.output_interval == 0) {
            // GPU → Host
            lbmCore.backend().download_fields(h_rho.data(), h_u.data(),
                h_flags.data(), h_phi.data());

            // 更新 FieldStore 中的 flags (VTK 别名)
            {
                auto handle = fieldStore.get("flags");
                memcpy(handle.data(), h_flags.data(),
                    nCells * sizeof(unsigned char));
            }
            // 更新 FieldStore 中的 velocity (VTK 别名)
            {
                auto handle = fieldStore.get("velocity");
                memcpy(handle.data(), h_u.data(), nCells * 3 * sizeof(float));
            }

            // 写 VTI 文件
            vtkService.onStepEnd(ctx);

            // 控制台统计
            double currentMass =
                calculateTotalMass(h_phi, h_flags, nCells, cfg.rho0);
            double massDiffPercent =
                (initialMass > 0)
                ? ((currentMass - initialMass) / initialMass * 100.0)
                : 0.0;

            int liquidCells = 0, interfaceCells = 0, gasCells = 0;
            for (int i = 0; i < nCells; ++i) {
                if (h_flags[i] == 0x08)
                    liquidCells++;
                else if (h_flags[i] == 0x10)
                    interfaceCells++;
                else if (h_flags[i] == 0x20)
                    gasCells++;
            }

            std::cout << "Step " << std::setw(6) << step
                << ": L=" << liquidCells << " I=" << interfaceCells
                << " G=" << gasCells << "  Mass=" << std::fixed
                << std::setprecision(4) << currentMass << " ("
                << std::showpos << std::setprecision(3) << massDiffPercent
                << std::noshowpos << "%)" << std::endl;
        }

        // LBM 核心计算各阶段（自由表面嵌入在 streaming 和后宏更新之间）
        fsModule.preStream(ctx);         // 1. 备份旧分布
        lbmCore.streamCollide();         // 2. Streaming + Collision
        fsModule.postStream(ctx);        // 3-5. 质量更新 + 标志转换 + Phi再分配
        lbmCore.updateMacroscopic();     // 6. 更新宏观量 + 边界 + 同步

        // 健壮性检查
        if (!lbmCore.checkHealth()) {
            std::cerr << "[FAIL] Simulation exploded at step " << step << std::endl;
            break;
        }
    }

    // ------------------------------------------------------------------
    // 最终报告
    // ------------------------------------------------------------------
    vtkService.finalize(ctx);

    lbmCore.backend().download_fields(h_rho.data(), h_u.data(), h_flags.data(),
        h_phi.data());
    double finalMass = calculateTotalMass(h_phi, h_flags, nCells, cfg.rho0);

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "FINAL MASS REPORT:" << std::endl;
    std::cout << "  Initial: " << std::fixed << std::setprecision(6)
        << initialMass << std::endl;
    std::cout << "  Final:   " << finalMass << std::endl;
    std::cout << "  Diff:    " << std::showpos << (finalMass - initialMass)
        << std::noshowpos << " (" << std::showpos << std::setprecision(4)
        << ((initialMass > 0)
            ? ((finalMass - initialMass) / initialMass * 100.0)
            : 0.0)
        << std::noshowpos << "%)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "=== Simulation Complete ===" << std::endl;
    std::cout << "VTI files written to: " << cfg.output_dir << std::endl;

    return 0;
}

// ============================================================================
// Mode 2: Scenario — 使用 ScenarioRunner 标准化流程
// ============================================================================
static int runScenarioMode(const DamBreak3DConfig& cfg) {
    std::cout << "=== DILMaster Dam Break 3D [Scenario Mode] ===" << std::endl;

    // 构建 LBMConfig
    lbm::LBMConfig lbmConfig{};
    lbmConfig.nx = cfg.nx;
    lbmConfig.ny = cfg.ny;
    lbmConfig.nz = cfg.nz;
    lbmConfig.tau = 3.0f * cfg.nu + 0.5f;
    lbmConfig.rho0 = cfg.rho0;
    lbmConfig.u0 = make_float3(0.0f, 0.0f, 0.0f);
    lbmConfig.gravity =
        make_float3(cfg.gravity[0], cfg.gravity[1], cfg.gravity[2]);

    lbmConfig.sigma = cfg.sigma;
    lbmConfig.wallFlags = lbm::WALL_ALL;
    lbmConfig.enableFreeSurface = true;

    // 捕获水块参数用于 lambda
    const int water_z = static_cast<int>(cfg.nz * cfg.water_z_ratio);
    const int water_y = static_cast<int>(cfg.ny * cfg.water_y_ratio);

    std::cout << "Grid: " << cfg.nx << " x " << cfg.ny << " x " << cfg.nz
        << std::endl;
    std::cout << "Water block: x=[0," << cfg.nx - 1 << "], y=[0," << water_y - 1
        << "], z=[0," << water_z - 1 << "]" << std::endl;

    // 组装 ScenarioConfig
    lbm_test::ScenarioConfig scenario;
    scenario.name = "DamBreak3D";
    scenario.config = lbmConfig;
    scenario.steps = cfg.steps;
    scenario.output_interval = cfg.output_interval;
    scenario.dt = 1.0f;
    scenario.output_dir = cfg.output_dir;

    // 通过 prepareGeometry 回调设置初始几何
    scenario.prepareGeometry =
        [&cfg, water_y, water_z](lbm::FreeSurfaceModule& fs) {
            // 1) 全域设为 GAS
            fs.setRegion(0, cfg.nx - 1, 0, cfg.ny - 1, 0, cfg.nz - 1,
                lbm::CellType::GAS, 0.0f, cfg.rho0);

            // 2) 水块区域设为 LIQUID
            fs.setRegion(0, cfg.nx - 1, 0, water_y - 1, 0, water_z - 1,
                lbm::CellType::LIQUID, 1.0f, cfg.rho0);

            // 3) 修复界面层
            fs.fixInterfaceLayer();
        };

    // 一行调用完成整个模拟
    lbm_test::runScenario(scenario);

    return 0;
}

// ============================================================================
// 命令行帮助
// ============================================================================
static void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " [options]\n"
              << "\n"
              << "Options:\n"
              << "  --mode <legacy|scenario>  Run mode (default: legacy)\n"
              << "  --steps <N>               Number of simulation steps (default: 20000)\n"
              << "  -h, --help                Show this help message\n"
              << std::endl;
}

// ============================================================================
// Main — 模式分发
// ============================================================================
int main(int argc, char** argv) {
    DamBreak3DConfig cfg;
    std::string mode = "scenario"; // 默认使用 scenario 模式

    cfg.steps = 10000;
    
    // Default output dir (relative to CWD)
    // If user wants it in build/.../dambreak3d_output, they should run from there OR we can try to detect.
    // However, explicitly creating it ensures it exists regardless of CWD.
    
    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--mode") && i + 1 < argc) {
            mode = argv[++i];
        } else if ((arg == "--steps") && i + 1 < argc) {
            cfg.steps = std::atoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else {
            // 兼容旧行为: 无名参数视为 steps
            cfg.steps = std::atoi(argv[i]);
        }
    }

    ensureOutputDir(cfg.output_dir);
    
    if (mode == "scenario") {
        return runScenarioMode(cfg);
    } else if (mode == "legacy") {
        return runLegacy(cfg);
    } else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        std::cerr << "Valid modes: legacy, scenario" << std::endl;
        printUsage(argv[0]);
        return 1;
    }
}
