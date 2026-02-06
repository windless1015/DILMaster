#pragma once
/**
 * LBMConfig.hpp - LBM 配置结构
 *
 * 支持:
 * - 碰撞模型选择 (SRT/TRT/MRT)
 * - 自由表面开关
 * - 边界条件配置
 */

#include <cstdint>
#include <cuda_runtime.h>


namespace lbm {

// Precision configuration
using real = float;
using real3 = float3;

// ============================================================================
// 碰撞模型
// ============================================================================
enum class CollisionModel : int {
  SRT = 0, // 单松弛时间 (BGK)
  TRT = 1, // 双松弛时间
  MRT = 2  // 多松弛时间
};

// ============================================================================
// 边界标志 (位掩码)
// ============================================================================
enum WallFlags {
  WALL_NONE = 0,
  WALL_X_MIN = 1 << 0,
  WALL_X_MAX = 1 << 1,
  WALL_Y_MIN = 1 << 2,
  WALL_Y_MAX = 1 << 3,
  WALL_Z_MIN = 1 << 4,
  WALL_Z_MAX = 1 << 5,
  WALL_ALL = 0xFF
};

// ============================================================================
// 边界条件类型
// ============================================================================
enum BoundaryType : uint8_t {
  BC_BOUNCE_BACK = 0, // 半程反弹 (无滑移固体壁面)
  BC_EQUILIBRIUM = 1, // 平衡态外推
  BC_OPEN = 2,        // 开放边界
  BC_PERIODIC = 3,    // 周期性
  BC_PRESSURE_OUTLET = 4 // 压力出口 (Grad U = 0, P = ref)
};

// ============================================================================
// LBM 配置结构
// ============================================================================
struct LBMConfig {
  // 网格尺寸
  int nx = 64;
  int ny = 64;
  int nz = 64;

  // 物理参数
  real tau = 1.0f;                    // 松弛时间
  real3 gravity = {0.0f, 0.0f, 0.0f}; // 外力
  real sigma = 0.0f;                  // 表面张力系数

  // 碰撞模型
  CollisionModel collisionModel = CollisionModel::SRT;
  real cSmago = 0.14f; // Smagorinsky 常数 (如果使用)

  // ======================== 自由表面配置 ========================
  bool enableFreeSurface = true; // 是否启用自由表面

  // 边界条件 - 哪些壁面存在 (位掩码)
  int wallFlags = WALL_ALL;

  // 各壁面的边界条件类型
  BoundaryType bcXMin = BC_BOUNCE_BACK;
  BoundaryType bcXMax = BC_BOUNCE_BACK;
  BoundaryType bcYMin = BC_BOUNCE_BACK;
  BoundaryType bcYMax = BC_BOUNCE_BACK;
  BoundaryType bcZMin = BC_BOUNCE_BACK;
  BoundaryType bcZMax = BC_OPEN;

  // 初始化参数
  real rho0 = 1.0f;              // 初始密度
  real3 u0 = {0.0f, 0.0f, 0.0f}; // 初始速度
  real pressure_outlet_rho = 1.0f; // 压力出口参考密度 (P ~ rho)
};

} // namespace lbm
