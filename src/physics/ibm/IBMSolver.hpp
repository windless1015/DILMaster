#pragma once
/**
 * @file IBMSolver.hpp
 * @brief IBM (Immersed Boundary Method) 求解器
 *
 * 职责：
 * - 管理浸没边界标记点（markers）的位置、速度、受力
 * - 驱动标记点运动学（如螺旋桨旋转、活塞运动）
 * - 作为 LBM 和固体边界之间的接口
 *
 * 设计：
 * - 继承 ISolver 接口，与 LBM/DEM 求解器完全解耦
 * - 内部持有 IBMCore 进行 CUDA 加速计算
 * - 通过 FieldStore 管理数据，便于 Coupler 耦合
 * - 几何由 GeometryUtils::loadSTL + sampleSurface 预处理
 *
 * =============================================================================
 * 字段命名规范（供 Coupler 使用）
 * =============================================================================
 *
 * 本求解器在 FieldStore 中创建以下字段：
 *
 *   字段名                  | 类型    | 说明
 *   ------------------------|---------|---------------------------
 *   ibm.markers             | float3  | 标记点位置 (x, y, z)
 *   ibm.velocity            | float3  | 标记点速度 (vx, vy, vz)
 *   ibm.force               | float3  | 流体作用力 (fx, fy, fz)
 *   ibm.normals             | float3  | 标记点法向量（可选）
 *
 * 命名规则：<domain>.<property>
 *   - domain:   物理域标识，IBM 使用 "ibm"
 *   - property: 属性名，如 markers, velocity 等
 *
 * Coupler 可通过 ctx.fields->get("ibm.markers") 获取字段句柄，
 * 实现 LBM-IBM 双向耦合（如流体力计算、速度插值）。
 * =============================================================================
 */

#include "../ISolver.hpp"
#include "IBMMotion.hpp"
#include "../../geometry/IBMMarker.h"
#include <cstddef>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

// 前向声明 IBMCore（避免头文件依赖）
namespace ibm {
class IBMCore;
}

// =============================================================================
// IBM 标准字段名称（用于多物理场耦合）
// =============================================================================
//
// 命名规范：<domain>.<property>
//   - domain:   "ibm" (IBM 物理域)
//   - property: 属性名
//
// 与 LBM/DEM 字段命名风格一致
// Coupler 通过这些常量访问字段，确保求解器间的松耦合。
//
// =============================================================================

/**
 * @brief IBM 字段名称常量
 *
 * 用于 FieldStore 字段注册和访问，确保多求解器间命名一致性。
 *
 * 使用示例：
 * @code
 *   // 在 Coupler 中获取标记点位置
 *   auto markers = ctx.fields->get(IBMFields::MARKERS);
 *   float3* positions = static_cast<float3*>(markers.data());
 *
 *   // 读取标记点速度用于速度边界条件
 *   auto vel = ctx.fields->get(IBMFields::VELOCITY);
 *
 *   // 将计算的流体力写入
 *   auto force = ctx.fields->get(IBMFields::FORCE);
 * @endcode
 */
namespace IBMFields {

/**
 * @brief 标记点位置字段
 *
 * 类型: float3 (x, y, z)
 * 单位: 米 [m] 或格子单位 [lu]
 * 用途: 浸没边界标记点在全局坐标系中的位置
 */
inline constexpr const char *MARKERS = "ibm.markers";

/**
 * @brief 标记点速度字段
 *
 * 类型: float3 (vx, vy, vz)
 * 单位: 米/秒 [m/s] 或格子单位 [lu/ts]
 * 用途: 标记点的运动速度（由运动学设定或刚体动力学计算）
 *       LBM 使用此速度设置边界条件
 */
inline constexpr const char *VELOCITY = "ibm.velocity";

/**
 * @brief 标记点受力字段
 *
 * 类型: float3 (fx, fy, fz)
 * 单位: 牛顿 [N] 或格子单位
 * 用途: 流体作用在标记点上的力
 *       由 LBM Coupler 计算后写入
 */
inline constexpr const char *FORCE = "ibm.force";

/**
 * @brief 标记点法向量字段
 *
 * 类型: float3 (nx, ny, nz)
 * 单位: 无量纲（单位向量）
 * 用途: 标记点处的表面法向量，用于力计算和流动分析
 */
inline constexpr const char *NORMALS = "ibm.normals";

/**
 * @brief 标记点面积字段
 *
 * 类型: float
 * 单位: 平方米 [m²] 或格子单位
 * 用途: 每个标记点代表的表面面积，用于力积分
 */
inline constexpr const char *AREA = "ibm.area";

/**
 * @brief 固体标识场字段
 *
 * 类型: uint8_t 或 int
 * 单位: 无量纲（0=流体, 1=固体）
 * 用途: 标识流体网格中哪些节点被固体占据
 *       LBM 使用此字段应用边界条件
 */
inline constexpr const char *SOLID_MASK = "ibm.solid_mask";

} // namespace IBMFields

// =============================================================================
// 运动类型枚举
// =============================================================================

/**
 * @brief 标记点运动类型
 */
enum class IBMMotionType : int {
  STATIC = 0,      // 静止不动
  ROTATION = 1,    // 绕轴旋转（如螺旋桨）
  TRANSLATION = 2, // 平移运动（如活塞）
  OSCILLATION = 3, // 振荡运动
  PRESCRIBED = 4,  // 预设运动（从文件读取）
  RIGID_BODY = 5   // 刚体动力学（由流体力驱动）
};

// =============================================================================
// IBM 配置
// =============================================================================

/**
 * @brief IBM 求解器配置参数
 */
struct IBMConfig {
  // 几何配置
  std::string stl_file;         // STL 文件路径
  float marker_spacing = 0.01f; // 标记点间距 [m]
  std::size_t num_markers = 0;  // 标记点数量（自动计算或手动设置）

  // 运动配置
  IBMMotionType motion_type = IBMMotionType::STATIC;

  // 旋转参数（motion_type == ROTATION）
  float rotation_axis_x = 0.0f;
  float rotation_axis_y = 0.0f;
  float rotation_axis_z = 1.0f; // 默认绕 Z 轴旋转
  float rotation_center_x = 0.5f;
  float rotation_center_y = 0.5f;
  float rotation_center_z = 0.5f;
  float angular_velocity = 0.0f; // 角速度 [rad/s]

  // 平移参数（motion_type == TRANSLATION）
  float translation_velocity_x = 0.0f;
  float translation_velocity_y = 0.0f;
  float translation_velocity_z = 0.0f;

  // 振荡参数（motion_type == OSCILLATION）
  float oscillation_amplitude = 0.0f;
  float oscillation_frequency = 0.0f;

  // IBM 计算参数
  float stencil_width = 2.0f;  // 插值模板宽度（格子单位）
  int interpolation_order = 2; // 插值阶数

  // 刚体参数（motion_type == RIGID_BODY）
  float mass = 1.0f;              // 刚体质量 [kg]
  float moment_of_inertia = 1.0f; // 转动惯量 [kg·m²]
};

// =============================================================================
// IBMSolver
// =============================================================================

/**
 * @brief IBM 求解器
 *
 * 实现浸没边界法，可与 LBM 流体求解器通过 Coupler 耦合
 */
class IBMSolver : public ISolver {
public:
  IBMSolver();
  ~IBMSolver() override;

  // 禁止拷贝/移动（CUDA 资源管理）
  IBMSolver(const IBMSolver &) = delete;
  IBMSolver &operator=(const IBMSolver &) = delete;

  // === ISolver 接口实现 ===

  /**
   * @brief 获取求解器名称
   */
  std::string name() const override { return "IBM"; }

  /**
   * @brief 从 YAML 配置节点读取参数
   *
   * 支持的配置项：
   *   ibm:
   *     stl_file: "propeller.stl"
   *     marker_spacing: 0.01
   *     motion_type: "rotation"
   *     rotation:
   *       axis: [0, 0, 1]
   *       center: [0.5, 0.5, 0.5]
   *       angular_velocity: 10.0  # rad/s
   *     stencil_width: 2.0
   */
  void configure(const ConfigNode &node) override;

  /**
   * @brief 分配计算资源和字段
   *
   * 在 FieldStore 中创建以下字段：
   *   - ibm.markers (float3)
   *   - ibm.velocity (float3)
   *   - ibm.force (float3)
   *   - ibm.normals (float3) [可选]
   *   - ibm.area (float) [可选]
   */
  void allocate(StepContext &ctx) override;

  /**
   * @brief 初始化标记点
   *
   * 调用 IBMCore 初始化 CUDA 资源
   * 如果配置了 STL 文件，加载几何并采样标记点
   */
  void initialize(StepContext &ctx) override;

  /**
   * @brief 执行一个 IBM 时间步
   *
   * 流程：
   *   1. 更新标记点运动学（stepKinematics）
   *   2. 调用 IBMCore 进行力计算（如果有外部力输入）
   */
  void step(StepContext &ctx) override;

  /**
   * @brief 清理资源
   */
  void finalize(StepContext &ctx) override;

  // === 运动学接口 ===

  /**
   * @brief 更新标记点运动学
   * @param dt 时间步长
   *
   * 根据 motion_type 更新标记点位置和速度：
   *   - ROTATION: 绕轴旋转
   *   - TRANSLATION: 平移
   *   - OSCILLATION: 振荡
   *   - PRESCRIBED: 从预设数据读取
   */
  void stepKinematics(double dt);

  /**
   * @brief 计算总流体力和力矩
   * @param[out] total_force 总力 (fx, fy, fz)
   * @param[out] total_torque 总力矩 (tx, ty, tz)
   */
  void computeTotalForceAndTorque(float *total_force,
                                  float *total_torque) const;

  // === 直接配置方法（绕过 YAML 依赖）===

  /**
   * @brief 设置字段前缀 (用于多体模拟)
   * @param prefix 前缀字符串 (例如 "sphere_0")
   * 
   * 字段名将变为 "{prefix}.markers", "{prefix}.velocity", 等。
   * 默认为 "ibm"。
   */
  void setPrefix(const std::string& prefix);

  /**
   * @brief 设置重力 (Lattice Units)
   */
  void setGravity(float gx, float gy, float gz);

  /**
   * @brief 直接设置配置
   */
  void setConfig(const IBMConfig &config);

  /**
   * @brief 设置标记点数量
   */
  void setNumMarkers(std::size_t n) { config_.num_markers = n; }

  /**
   * @brief 设置 STL 文件路径
   */
  void setSTLFile(const std::string &path) { config_.stl_file = path; }

  /**
   * @brief 设置标记点间距
   */
  void setMarkerSpacing(float spacing) { config_.marker_spacing = spacing; }

  /**
   * @brief 设置运动类型
   */
  void setMotionType(IBMMotionType type) { config_.motion_type = type; }

  /**
   * @brief 设置旋转参数
   */
  void setRotation(float axis_x, float axis_y, float axis_z, float center_x,
                   float center_y, float center_z, float angular_velocity) {
    config_.rotation_axis_x = axis_x;
    config_.rotation_axis_y = axis_y;
    config_.rotation_axis_z = axis_z;
    config_.rotation_center_x = center_x;
    config_.rotation_center_y = center_y;
    config_.rotation_center_z = center_z;
    config_.angular_velocity = angular_velocity;
  }

  /**
   * @brief 设置平移速度
   */
  void setTranslationVelocity(float vx, float vy, float vz) {
    config_.translation_velocity_x = vx;
    config_.translation_velocity_y = vy;
    config_.translation_velocity_z = vz;
  }

  /**
   * @brief 手动设置标记点位置（用于测试或程序化生成）
   * @param positions 标记点位置数组 (flat: x0,y0,z0, x1,y1,z1, ...)
   * @param count 标记点数量
   */
  void setMarkerPositions(const float *positions, std::size_t count);

  // === 访问器 ===

  /**
   * @brief 获取 IBM 核心（用于高级操作）
   */
  ibm::IBMCore *getCore() { return core_.get(); }
  const ibm::IBMCore *getCore() const { return core_.get(); }

  /**
   * @brief 获取当前配置
   */
  const IBMConfig &config() const { return config_; }

  /**
   * @brief 获取标记点数量
   */
  std::size_t numMarkers() const { return config_.num_markers; }
  
  /**
   * @brief 获取标记点列表 (用于测试/调试)
   */
  const std::vector<IBMMarker>& getMarkers() const { return markers_; }

  /**
   * @brief 获取当前模拟时间
   */
  double currentTime() const { return current_time_; }

  /**
   * @brief 从 FieldStore 读取力到运动学模块
   */
  void applyForcesFromFieldStore(StepContext &ctx);

  /**
   * @brief 获取运动学模块
   */
  IBMMotion *getMotion() { return motion_.get(); }
  const IBMMotion *getMotion() const { return motion_.get(); }

  /**
   * @brief Get rigid body state from motion module safely
   */
  const RigidBodyState& getBodyState() const {
      if (motion_) return motion_->getState();
      static RigidBodyState empty;
      return empty;
  }

private:
  std::unique_ptr<ibm::IBMCore> core_; // IBM CUDA 核心
  std::unique_ptr<IBMMotion> motion_;  // 运动学模块 (CPU only)
  IBMConfig config_;                   // 配置参数
  std::string prefix_ = "ibm";         // 字段名称前缀
  float3 desired_gravity_ = {0, 0, 0}; // 预设重力 (初始化时应用)

  // 状态
  bool fields_allocated_ = false;
  double current_time_ = 0.0;

  // 临时缓冲
  std::vector<float3> host_markers_;
  std::vector<float3> host_velocities_;
  std::vector<IBMMarker> markers_; // 用于 IBMMotion 计算的完整标记点信息
  std::vector<float3> initial_rel_pos_; // 初始相对位置（Body Frame），用于无漂移旋转

  // 内部方法
  void updateRotation(double dt);
  void updateTranslation(double dt);
  void updateOscillation(double dt);
};
