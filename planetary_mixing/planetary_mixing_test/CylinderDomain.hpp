#pragma once
/**
 * CylinderDomain.hpp - 圆柱体仿真域初始化
 *
 * 在直角网格上构造圆柱体流域:
 *   - 圆柱外的格点 → SOLID (固体壁面)
 *   - 圆柱内的底部 → FLUID (流体)
 *   - 圆柱内的顶部 → GAS   (空气/自由面)
 *   - 流体-气体交界 → INTERFACE (自由面界面)
 */

#include <vector>
#include <cstdint>
#include <cmath>
#include <iostream>

#include "physics/lbm/cuda/LBMTypes.cuh"

struct CylinderDomainConfig {
    int nx, ny, nz;          // 网格尺寸
    float cx, cz;            // 圆柱中心 (XZ 平面)
    float radius;            // 圆柱半径 (格点单位)
    float fluid_fraction;    // 流体填充比 (0.7 = 底部70%)
    int axis;                // 圆柱轴向: 0=X, 1=Y, 2=Z (default Y=1)
};

struct CylinderDomainResult {
    std::vector<uint8_t> flags;
    std::vector<float>   phi;   // VOF: 1.0=fluid, 0.0=gas
    std::vector<float>   mass;  // mass = rho * phi
    int fluid_cells;
    int gas_cells;
    int solid_cells;
    int interface_cells;
};

/**
 * 生成圆柱域的 flags / phi / mass 数组
 *
 * 圆柱轴沿 Y 方向 (axis=1):
 *   - XZ 平面: 距 (cx, cz) > radius 的格点为 SOLID
 *   - Y 方向:  [0, fluid_height) = FLUID, [fluid_height] = INTERFACE, 以上 = GAS
 *   - 圆柱底面 (y=0) 和顶面 (y=ny-1) 由 LBM 边界条件控制
 */
inline CylinderDomainResult buildCylinderDomain(const CylinderDomainConfig& cfg) {
    using namespace lbm::cuda::CellFlag;

    const int N = cfg.nx * cfg.ny * cfg.nz;
    CylinderDomainResult res;
    res.flags.resize(N, GAS);
    res.phi.resize(N, 0.0f);
    res.mass.resize(N, 0.0f);
    res.fluid_cells = 0;
    res.gas_cells = 0;
    res.solid_cells = 0;
    res.interface_cells = 0;

    // 流体高度 (Y 方向)
    int fluid_height = static_cast<int>(cfg.ny * cfg.fluid_fraction);
    if (fluid_height < 1) fluid_height = 1;
    if (fluid_height >= cfg.ny) fluid_height = cfg.ny - 1;

    float R2 = cfg.radius * cfg.radius;

    for (int z = 0; z < cfg.nz; ++z) {
        for (int y = 0; y < cfg.ny; ++y) {
            for (int x = 0; x < cfg.nx; ++x) {
                int idx = z * cfg.nx * cfg.ny + y * cfg.nx + x;

                // 检查是否在圆柱内 (XZ 平面)
                float dx = (float)x - cfg.cx;
                float dz = (float)z - cfg.cz;
                float r2 = dx * dx + dz * dz;

                if (r2 > R2) {
                    // 圆柱外 → 固体壁面
                    res.flags[idx] = SOLID;
                    res.phi[idx] = 0.0f;
                    res.mass[idx] = 0.0f;
                    res.solid_cells++;
                } else if (y < fluid_height) {
                    // 流体区域
                    res.flags[idx] = FLUID;
                    res.phi[idx] = 1.0f;
                    res.mass[idx] = 1.0f;  // rho0 * phi
                    res.fluid_cells++;
                } else if (y == fluid_height) {
                    // 自由面界面
                    res.flags[idx] = INTERFACE;
                    res.phi[idx] = 0.5f;
                    res.mass[idx] = 0.5f;
                    res.interface_cells++;
                } else {
                    // 气体区域
                    res.flags[idx] = GAS;
                    res.phi[idx] = 0.0f;
                    res.mass[idx] = 0.0f;
                    res.gas_cells++;
                }
            }
        }
    }

    std::cout << "[CylinderDomain] " << cfg.nx << "x" << cfg.ny << "x" << cfg.nz
              << " R=" << cfg.radius << " fluid_h=" << fluid_height
              << " | FLUID=" << res.fluid_cells
              << " INTERFACE=" << res.interface_cells
              << " GAS=" << res.gas_cells
              << " SOLID=" << res.solid_cells << std::endl;

    return res;
}
