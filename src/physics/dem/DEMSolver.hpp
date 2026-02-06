#pragma once
/**
 * @file DEMSolver.hpp
 * @brief DEM (Discrete Element Method) 求解器
 *
 * 职责：
 * - 管理离散粒子的位置、速度、受力
 * - 执行粒子碰撞检测与接触力计算
 * - 时间积分更新粒子状态
 *
 * 设计：
 * - 继承 ISolver 接口，与 LBM 等求解器完全解耦
 * - 内部持有 DEMCore 进行 CUDA 加速计算
 * - 通过 FieldStore 管理数据，便于 Coupler 耦合
 *
 * =============================================================================
 * 字段命名规范（供 Coupler 使用）
 * =============================================================================
 *
 * 本求解器在 FieldStore 中创建以下字段：
 *
 *   字段名                  | 类型    | 说明
 *   ------------------------|---------|---------------------------
 *   particles.position      | float3  | 粒子位置 (x, y, z)
 *   particles.velocity      | float3  | 粒子速度 (vx, vy, vz)
 *   particles.force         | float3  | 粒子受力 (fx, fy, fz)
 *   particles.radius        | float   | 粒子半径
 *
 * 命名规则：<domain>.<property>
 *   - domain:   物理域标识，DEM 使用 "particles"
 *   - property: 属性名，如 position, velocity 等
 *
 * Coupler 可通过 ctx.fields->get("particles.position") 获取字段句柄，
 * 实现 CFD-DEM 双向耦合（如流体力传递、粒子体积分数计算）。
 * =============================================================================
 */

#include "../ISolver.hpp"
#include <cstddef>
#include <memory>

// 前向声明 DEMCore（避免头文件依赖）
namespace dem {
class DEMCore;
}

// =============================================================================
// DEM 标准字段名称（用于多物理场耦合）
// =============================================================================
//
// 命名规范：<domain>.<property>
//   - domain:   "particles" (DEM 物理域)
//   - property: 属性名
//
// 与 LBM 字段命名风格一致（如 LBM 使用 "fluid.velocity", "fluid.density"）
// Coupler 通过这些常量访问字段，确保求解器间的松耦合。
//
// =============================================================================

/**
 * @brief DEM 字段名称常量
 *
 * 用于 FieldStore 字段注册和访问，确保多求解器间命名一致性。
 *
 * 使用示例：
 * @code
 *   // 在 Coupler 中获取粒子位置
 *   auto pos = ctx.fields->get(DEMFields::POSITION);
 *   float3* positions = static_cast<float3*>(pos.data());
 *
 *   // 写入流体力到粒子
 *   auto force = ctx.fields->get(DEMFields::FORCE);
 *   float3* forces = static_cast<float3*>(force.data());
 * @endcode
 */
namespace DEMFields {

/**
 * @brief 粒子位置字段
 *
 * 类型: float3 (x, y, z)
 * 单位: 米 [m]
 * 用途: 粒子质心在全局坐标系中的位置
 */
inline constexpr const char *POSITION = "particles.position";

/**
 * @brief 粒子速度字段
 *
 * 类型: float3 (vx, vy, vz)
 * 单位: 米/秒 [m/s]
 * 用途: 粒子质心的平动速度
 */
inline constexpr const char *VELOCITY = "particles.velocity";

/**
 * @brief 粒子受力字段
 *
 * 类型: float3 (fx, fy, fz)
 * 单位: 牛顿 [N]
 * 用途: 作用在粒子上的合力（接触力 + 外力 + 流体力）
 *       Coupler 将流体阻力写入此字段
 */
inline constexpr const char *FORCE = "particles.force";

/**
 * @brief 粒子半径字段
 *
 * 类型: float
 * 单位: 米 [m]
 * 用途: 球形粒子的半径，用于接触检测和体积计算
 */
inline constexpr const char *RADIUS = "particles.radius";

/**
 * @brief 粒子密度字段
 *
 * 类型: float
 * 单位: 千克/立方米 [kg/m³]
 * 用途: 粒子材料密度，用于质量和惯性计算
 */
inline constexpr const char *DENSITY = "particles.density";

/**
 * @brief 粒子质量字段（可选，可由密度和半径计算）
 *
 * 类型: float
 * 单位: 千克 [kg]
 * 用途: 粒子质量 (m = 4/3 * π * r³ * ρ)
 */
inline constexpr const char *MASS = "particles.mass";

/**
 * @brief 粒子角速度字段（可选，用于旋转粒子）
 *
 * 类型: float3 (ωx, ωy, ωz)
 * 单位: 弧度/秒 [rad/s]
 * 用途: 粒子绕质心的转动角速度
 */
inline constexpr const char *ANGULAR_VELOCITY = "particles.angular_velocity";

/**
 * @brief 粒子力矩字段（可选，用于旋转粒子）
 *
 * 类型: float3 (τx, τy, τz)
 * 单位: 牛顿·米 [N·m]
 * 用途: 作用在粒子上的合力矩
 */
inline constexpr const char *TORQUE = "particles.torque";

} // namespace DEMFields

/**
 * @brief DEM 求解器配置参数
 */
struct DEMConfig {
  // 粒子属性
  std::size_t num_particles = 0;    // 粒子数量
  float particle_radius = 0.01f;    // 默认粒子半径 [m]
  float particle_density = 2500.0f; // 粒子密度 [kg/m^3]

  // 接触模型参数
  float restitution = 0.9f; // 恢复系数 (0~1)
  float friction = 0.3f;    // 摩擦系数
  float kn = 1e5f;          // 法向刚度 [N/m]
  float kt = 1e4f;          // 切向刚度 [N/m]
  float damping = 0.1f;     // 阻尼系数

  // 重力
  float gravity_x = 0.0f;
  float gravity_y = 0.0f;
  float gravity_z = -9.81f;

  // 计算域
  float domain_min_x = 0.0f;
  float domain_min_y = 0.0f;
  float domain_min_z = 0.0f;
  float domain_max_x = 1.0f;
  float domain_max_y = 1.0f;
  float domain_max_z = 1.0f;
};

/**
 * @brief DEM 求解器
 *
 * 实现离散元法颗粒模拟，可与 LBM 流体求解器通过 Coupler 耦合
 */
class DEMSolver : public ISolver {
public:
  DEMSolver();
  ~DEMSolver() override;

  // 禁止拷贝/移动（CUDA 资源管理）
  DEMSolver(const DEMSolver &) = delete;
  DEMSolver &operator=(const DEMSolver &) = delete;

  // === ISolver 接口实现 ===

  /**
   * @brief 获取求解器名称
   */
  std::string name() const override { return "DEM"; }

  /**
   * @brief 从 YAML 配置节点读取参数
   *
   * 支持的配置项：
   *   dem:
   *     num_particles: 1000
   *     particle_radius: 0.01
   *     particle_density: 2500
   *     restitution: 0.9
   *     friction: 0.3
   *     kn: 100000
   *     kt: 10000
   *     damping: 0.1
   *     gravity: [0, 0, -9.81]
   *     domain:
   *       min: [0, 0, 0]
   *       max: [1, 1, 1]
   */
  void configure(const ConfigNode &node) override;

  /**
   * @brief 分配计算资源和字段
   *
   * 在 FieldStore 中创建以下字段：
   *   - particles.position (float3)
   *   - particles.velocity (float3)
   *   - particles.force (float3)
   *   - particles.radius (float)
   */
  void allocate(StepContext &ctx) override;

  /**
   * @brief 初始化粒子状态
   *
   * 调用 DEMCore 初始化 CUDA 资源和初始粒子分布
   */
  void initialize(StepContext &ctx) override;

  /**
   * @brief 执行一个 DEM 时间步
   *
   * 流程：
   *   1. 碰撞检测（邻域搜索）
   *   2. 接触力计算（Hertz-Mindlin 或线性弹簧）
   *   3. 合外力计算（接触力 + 外部力 + 重力）
   *   4. 时间积分（Velocity Verlet）
   *   5. 边界处理
   */
  void step(StepContext &ctx) override;

  /**
   * @brief 清理资源
   */
  void finalize(StepContext &ctx) override;

  // === 直接配置方法（绕过 YAML 依赖）===

  /**
   * @brief 直接设置基本配置
   *
   * 用于测试或不使用 YAML 配置的场景
   */
  void setConfig(const DEMConfig &config);

  /**
   * @brief 设置粒子数量
   */
  void setNumParticles(std::size_t n) { config_.num_particles = n; }

  /**
   * @brief 设置默认粒子半径
   */
  void setParticleRadius(float r) { config_.particle_radius = r; }

  /**
   * @brief 设置恢复系数
   */
  void setRestitution(float e) { config_.restitution = e; }

  /**
   * @brief 设置摩擦系数
   */
  void setFriction(float mu) { config_.friction = mu; }

  /**
   * @brief 设置法向刚度
   */
  void setNormalStiffness(float kn) { config_.kn = kn; }

  /**
   * @brief 设置切向刚度
   */
  void setTangentialStiffness(float kt) { config_.kt = kt; }

  /**
   * @brief 设置重力
   */
  void setGravity(float gx, float gy, float gz) {
    config_.gravity_x = gx;
    config_.gravity_y = gy;
    config_.gravity_z = gz;
  }

  /**
   * @brief 设置计算域边界
   */
  void setDomain(float min_x, float min_y, float min_z, float max_x,
                 float max_y, float max_z) {
    config_.domain_min_x = min_x;
    config_.domain_min_y = min_y;
    config_.domain_min_z = min_z;
    config_.domain_max_x = max_x;
    config_.domain_max_y = max_y;
    config_.domain_max_z = max_z;
  }

  // === 访问器 ===

  /**
   * @brief 获取 DEM 核心（用于高级操作）
   */
  dem::DEMCore *getCore() { return core_.get(); }
  const dem::DEMCore *getCore() const { return core_.get(); }

  /**
   * @brief 获取当前配置
   */
  const DEMConfig &config() const { return config_; }

  /**
   * @brief 获取粒子数量
   */
  std::size_t numParticles() const { return config_.num_particles; }

private:
  std::unique_ptr<dem::DEMCore> core_; // DEM CUDA 核心
  DEMConfig config_;                   // 配置参数

  // 字段句柄（可选缓存，用于快速访问）
  bool fields_allocated_ = false;
};
