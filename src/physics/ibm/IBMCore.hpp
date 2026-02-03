#pragma once
/**
 * @file IBMCore.hpp
 * @brief IBM (Immersed Boundary Method) CUDA 核心
 *
 * 职责：
 * - 管理标记点数据的 GPU 内存（位置、速度、力）
 * - 执行 IBM 力计算（Multi-Direct Forcing 方法）
 * - 提供设备指针访问接口
 *
 * =============================================================================
 * 内存布局说明
 * =============================================================================
 *
 * 本实现采用 **AoS (Array of Structures)** 布局：
 *   float3* position_
 *   float3* velocity_
 *   float3* force_
 *
 * =============================================================================
 * Multi-Direct Forcing (MDF)
 * =============================================================================
 *
 * 实现了迭代力修正算法：
 * 1. 插值速度 U_interp
 * 2. 计算修正力 dF = (U_target - U_interp) / dt
 * 3. 投射 dF 到流体网格
 * 4. 更新流体速度 U_fluid += dF * dt
 * 5. 重复 1-4
 *
 * =============================================================================
 */

#include <cstddef>
#include <cuda_runtime.h>

namespace ibm {

// ============================================================================
// 枚举和配置
// ============================================================================

/**
 * @brief IBM 力计算方法
 */
enum class IBMForceMethod : int {
  DIRECT_FORCING = 0, // 直接力方法 (MDF)
  PENALTY = 1         // 惩罚力方法
};

/**
 * @brief IBM CUDA 参数
 */
struct IBMParams {
  std::size_t nMarkers = 0; // 标记点数量

  // 网格参数 (必须设置，用于空间索引)
  float dx = 1.0f; // 网格间距
  int nx = 1;      // X 方向网格数
  int ny = 1;      // Y 方向网格数
  int nz = 1;      // Z 方向网格数

  // 计算域原点
  float domain_origin_x = 0.0f;
  float domain_origin_y = 0.0f;
  float domain_origin_z = 0.0f;

  // IBM 参数
  int stencil_width = 2; // 插值模板宽度 (2: 4-point, 1: 2-point)
  int mdf_iterations = 3;     // MDF 迭代次数
  IBMForceMethod force_method = IBMForceMethod::DIRECT_FORCING;

  // 惩罚力参数
  float penalty_stiffness = 1e4f; // 惩罚刚度
};

// ============================================================================
// IBMBackend - CUDA 后端（RAII 内存管理）
// ============================================================================

class IBMBackend {
public:
  IBMBackend() = default;
  ~IBMBackend();

  // 禁止拷贝
  IBMBackend(const IBMBackend &) = delete;
  IBMBackend &operator=(const IBMBackend &) = delete;

  // ========================================================================
  // 初始化
  // ========================================================================
  void initialize(const IBMParams &params);
  bool is_initialized() const { return allocated_; }

  // ========================================================================
  // 标记点数据更新
  // ========================================================================
  void updateMarkers(const float3 *new_positions, const float3 *new_velocities);
  void updatePositions(const float3 *new_positions);
  void updateVelocities(const float3 *new_velocities);

  // ========================================================================
  // IBM 力计算
  // ========================================================================

  /**
   * @brief 计算 IBM 力 (MDF)
   * @param fluid_velocity 流体速度场（设备指针，只读输入）
   * @param fluid_density 流体密度场（设备指针）
   * @param fluid_force_out 输出流体力场（设备指针，累加输出）
   * @param dt 时间步长
   */
  void computeForces(const float3 *fluid_velocity, const float *fluid_density,
                     float3 *fluid_force_out, float dt);

  /**
   * @brief 清零力数组 (Marker forces)
   */
  void clearForces();

  // ========================================================================
  // 设备指针访问 (AoS 格式 - float3*)
  // ========================================================================
  float3 *getMarkerPositionDevice() { return position_; }
  const float3 *getMarkerPositionDevice() const { return position_; }

  float3 *getMarkerVelocityDevice() { return velocity_; }
  const float3 *getMarkerVelocityDevice() const { return velocity_; }

  float3 *getForceDevice() { return force_; }
  const float3 *getForceDevice() const { return force_; }

  // ========================================================================
  // 主机-设备数据传输
  // ========================================================================
  void uploadPositions(const float3 *host_positions);
  void uploadVelocities(const float3 *host_velocities);
  void downloadPositions(float3 *host_positions) const;
  void downloadVelocities(float3 *host_velocities) const;
  void downloadForces(float3 *host_forces) const;

  // ========================================================================
  // 同步
  // ========================================================================
  void synchronize() const;

  // ========================================================================
  // Diagnostics / Conversion
  // ========================================================================
  void convertForceAoSToSoA(const float3 *force_aos, float *force_soa,
                            int nCells) const;

  std::size_t nMarkers() const { return params_.nMarkers; }
  const IBMParams &params() const { return params_; }

private:
  IBMParams params_{};
  bool allocated_ = false;

  // GPU Buffers (Markers)
  float3 *position_ = nullptr; // 标记点位置
  float3 *velocity_ = nullptr; // 标记点目标速度
  float3 *force_ = nullptr;    // 标记点总拉格朗日力
  
  // MDF 临时缓冲 (Markers)
  float3 *interpolated_velocity_ = nullptr; // 插值得到的流体速度
  float3 *delta_force_ = nullptr;           // MDF 迭代中的力修正量

  // MDF 临时缓冲 (Grid - Size: nx*ny*nz)
  float3 *temp_fluid_velocity_ = nullptr;   // 临时流体速度场 (用于迭代更新)

  // 内部辅助方法
  void allocate_memory();
  void free_memory();
  
  // 核心内核包装器
  void interpolateVelocity(const float3 *grid_u);
  void spreadForce(const float3 *marker_force, float3 *grid_force);
  void updateTempVelocity(const float3 *grid_force, const float *rho, float dt);
};

// ============================================================================
// IBMCore - 高层 IBM 求解器核心
// ============================================================================

class IBMCore {
public:
  IBMCore(std::size_t nMarkers, float dx, int nx, int ny, int nz);
  explicit IBMCore(const IBMParams &params);
  ~IBMCore();

  void initialize();

  // 更新标记点状态
  void updateMarkers(const float3 *new_positions, const float3 *new_velocities);

  // 计算耦合力
  void computeForces(const float3 *fluid_velocity, const float *fluid_density,
                     float3 *fluid_force_out, float dt);

  void clearForces();

  // 访问接口
  float3 *getMarkerPositionDevice() { return backend_.getMarkerPositionDevice(); }
  const float3 *getMarkerPositionDevice() const { return backend_.getMarkerPositionDevice(); }
  float3 *getMarkerVelocityDevice() { return backend_.getMarkerVelocityDevice(); }
  const float3 *getMarkerVelocityDevice() const { return backend_.getMarkerVelocityDevice(); }
  float3 *getForceDevice() { return backend_.getForceDevice(); }
  const float3 *getForceDevice() const { return backend_.getForceDevice(); }

  // 数据传输
  void uploadPositions(const float3 *host_positions);
  void uploadVelocities(const float3 *host_velocities);
  void downloadPositions(float3 *host_positions) const;
  void downloadForces(float3 *host_forces) const;

  // 刚体运动辅助
  void applyRotation(float3 axis, float3 center, float angle);
  void applyTranslation(float3 displacement);

  bool checkHealth() const;
  void synchronize() const;

private:
  IBMParams params_;
  IBMBackend backend_;
  bool initialized_ = false;
};

} // namespace ibm
