#pragma once
/**
 * @file IBMCore.hpp
 * @brief IBM (Immersed Boundary Method) CUDA 核心
 *
 * 职责：
 * - 管理标记点数据的 GPU 内存（位置、速度、力、面积）
 * - 执行 IBM 力计算（Multi-Direct Forcing 方法）
 * - 提供设备指针访问接口
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
  float mdf_beta = 0.5f;      // MDF 欠松弛因子 (Stability)
  
  // Free Surface (Masked) Extensions
  bool enable_masked_fs = false; // 是否启用基于 mask 的 FS 支持
  float mask_eps = 1e-12f;       // 最小有效权重阈值
  bool use_fill_weight = true;   // Spread 时即使 mask=1 也乘以 fill
  
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
   * @param fluid_density 流体密度场（设备指针，可选）
   * @param fluid_force_out 输出流体力场（设备指针，累加输出）
   * @param dt 时间步长
   * @param valid_mask 有效流体单元掩码 (Host/Device? Device pointer. 1=Valid, 0=Invalid/Gas) [Optional]
   * @param fill_fraction 单元填充率 (Device pointer. 0.0~1.0) [Optional]
   */
  void computeForces(const float3 *fluid_velocity, const float *fluid_density,
                     float3 *fluid_force_out, float dt,
                     const uint8_t *valid_mask = nullptr,
                     const float *fill_fraction = nullptr);

  /**
   * @brief 清零力数组 (Marker forces)
   */
  void clearForces();
  
  // Debug Statistics
  unsigned int getFallbackCount() const;

  // ========================================================================
  // 设备指针访问 (AoS 格式 - float3*)
  // ========================================================================

  // ========================================================================
  // 设备指针访问 (AoS 格式 - float3*)
  // ========================================================================
  float3 *getMarkerPositionDevice() { return position_; }
  const float3 *getMarkerPositionDevice() const { return position_; }

  float3 *getMarkerVelocityDevice() { return velocity_; }
  const float3 *getMarkerVelocityDevice() const { return velocity_; }
  
  float *getMarkerAreaDevice() { return area_; }
  const float *getMarkerAreaDevice() const { return area_; }

  float3 *getForceDevice() { return force_; }
  const float3 *getForceDevice() const { return force_; }

  // ========================================================================
  // 主机-设备数据传输
  // ========================================================================
  void uploadPositions(const float3 *host_positions);
  void uploadVelocities(const float3 *host_velocities);
  void uploadAreas(const float *host_areas);
  void downloadPositions(float3 *host_positions) const;
  void downloadVelocities(float3 *host_velocities) const;
  void downloadForces(float3 *host_forces) const;

#ifdef IBM_TESTING
  void downloadInterpolatedVelocity(float3 *host_u_interp) const;
#endif

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
  float *area_ = nullptr;      // 标记点面积/权重
  float3 *force_ = nullptr;    // 标记点总拉格朗日力
  
  // MDF 临时缓冲 (Markers)
  float3 *interpolated_velocity_ = nullptr; // 插值得到的流体速度
  float *interpolated_density_ = nullptr;   // 插值得到的流体密度 [New]
  float3 *delta_force_ = nullptr;           // MDF 迭代中的力修正量

  // MDF 临时缓冲 (Grid - Size: nx*ny*nz)
  float3 *temp_fluid_velocity_ = nullptr;   // 临时流体速度场 (用于迭代更新)
  
  // Debug / Stats
  unsigned int *d_fallback_count_ = nullptr; // Fallback 计数器

  // 内部辅助方法
  void allocate_memory();
  void free_memory();
  
  // 核心内核包装器
  void interpolateVelocity(const float3 *grid_u, const uint8_t* mask = nullptr);
  void spreadForce(const float3 *marker_force, float3 *grid_force, 
                   const uint8_t* mask = nullptr, const float* fill = nullptr);
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

  // 更新标记点状态 (兼容旧接口)
  void updateMarkers(const float3 *new_positions, const float3 *new_velocities);
  
  // 更新标记点状态 (包含面积)
  void updateMarkers(const float3 *new_positions, const float3 *new_velocities, const float *new_areas);

  // 计算耦合力 (支持 Masked FS)
  void computeForces(const float3 *fluid_velocity, const float *fluid_density,
                     float3 *fluid_force_out, float dt,
                     const uint8_t *valid_mask = nullptr,
                     const float *fill_fraction = nullptr);

  void clearForces();
  
  unsigned int getFallbackCount() const { return backend_.getFallbackCount(); }

  // 访问接口
  float3 *getMarkerPositionDevice() { return backend_.getMarkerPositionDevice(); }
  const float3 *getMarkerPositionDevice() const { return backend_.getMarkerPositionDevice(); }
  float3 *getMarkerVelocityDevice() { return backend_.getMarkerVelocityDevice(); }
  const float3 *getMarkerVelocityDevice() const { return backend_.getMarkerVelocityDevice(); }
  float *getMarkerAreaDevice() { return backend_.getMarkerAreaDevice(); }
  const float *getMarkerAreaDevice() const { return backend_.getMarkerAreaDevice(); }
  float3 *getForceDevice() { return backend_.getForceDevice(); }
  const float3 *getForceDevice() const { return backend_.getForceDevice(); }

  // 数据传输
  void uploadPositions(const float3 *host_positions);
  void uploadVelocities(const float3 *host_velocities);
  void uploadAreas(const float *host_areas);
  void downloadPositions(float3 *host_positions) const;
  void downloadForces(float3 *host_forces) const;

#ifdef IBM_TESTING
  // Debug/Test Interfaces
  void downloadInterpolatedVelocity(float3 *host_u_interp) const {
      backend_.downloadInterpolatedVelocity(host_u_interp);
  }
#endif

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
