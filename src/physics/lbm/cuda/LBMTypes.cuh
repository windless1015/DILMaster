#pragma once
/**
 * LBMTypes.cuh - LBM 基础类型定义
 *
 * 包含:
 * - 碰撞模型枚举
 * - 运行时参数结构
 * - 单元格类型标志
 * - 速度集常量
 */

#include <cstdint>

namespace lbm {
namespace cuda {

// ============================================================================
// 碰撞模型
// ============================================================================
enum class CollisionModel : int {
  SRT = 0, // 单松弛时间 (BGK)
  TRT = 1, // 双松弛时间
  MRT = 2  // 多松弛时间
};

// ============================================================================
// D3Q19 速度集常量
// ============================================================================
constexpr unsigned int kVelocitySet = 19u;
constexpr unsigned int kDimension = 3u;

// ============================================================================
// 单元格类型标志 (用于 flags 数组)
// ============================================================================
namespace CellFlag {
constexpr uint8_t SOLID = 0x01;     // 固体边界
constexpr uint8_t PRESSURE_OUTLET = 0x02; // 压力出口
constexpr uint8_t FLUID = 0x08;     // 流体
constexpr uint8_t INTERFACE = 0x10; // 界面
constexpr uint8_t GAS = 0x20;       // 气体

// 过渡状态 (用于自由表面标志转换)
constexpr uint8_t TO_FLUID = 0x18;     // 0x08 | 0x10 - 即将变为流体
constexpr uint8_t TO_GAS = 0x30;       // 0x10 | 0x20 - 即将变为气体
constexpr uint8_t TO_INTERFACE = 0x38; // 0x08 | 0x10 | 0x20 - 即将变为界面

// 掩码
constexpr uint8_t BOUNDARY_MASK = 0x03; // 边界位 (SOLID | PRESSURE_OUTLET)
constexpr uint8_t SURFACE_MASK = 0x38;  // 表面状态位
} // namespace CellFlag

// ============================================================================
// 运行时参数结构
// ============================================================================
struct LBMParams {
  // 网格尺寸
  unsigned int Nx;
  unsigned int Ny;
  unsigned int Nz;
  unsigned long long N; // 总单元数

  // 物理参数
  float nu; // 运动粘度
  float w;  // 松弛频率 omega = 1/(3*nu + 0.5)

  // 外力
  float fx, fy, fz;

  // 表面张力
  float sigma;       // 表面张力系数
  float def_6_sigma; // 6 * sigma (预计算)

  // 碰撞模型
  CollisionModel collisionModel;

  // FreeSurface 开关
  bool enableFreeSurface;

  // Boundary Config
  float pressure_outlet_rho;
  uint8_t boundaryTypeXMin;
  uint8_t boundaryTypeXMax;
  uint8_t boundaryTypeYMin;
  uint8_t boundaryTypeYMax;
  uint8_t boundaryTypeZMin;
  uint8_t boundaryTypeZMax;
};

// ============================================================================
// FP16 相关常量
// ============================================================================
constexpr float kFP16Scale = 32768.0f;
constexpr float kFP16InvScale = 1.0f / 32768.0f; // 3.0517578e-5f

// ============================================================================
// 物理常量
// ============================================================================
constexpr float kMaxVelocity = 0.57735027f; // 1/sqrt(3) - 最大速度限制
constexpr float kCs2 = 1.0f / 3.0f;         // 声速平方

// ============================================================================
// D3Q19 权重
// ============================================================================
constexpr float kW0 = 1.0f / 3.0f;  // 中心权重
constexpr float kWs = 1.0f / 18.0f; // 面邻居权重
constexpr float kWe = 1.0f / 36.0f; // 边邻居权重

} // namespace cuda
} // namespace lbm
