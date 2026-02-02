#pragma once
/**
 * LBMCore.hpp - LBM 求解器的底层 CUDA 核心
 *
 * 职责：
 * - 管理 DDF (Distribution Function) 字段
 * - 执行 streaming + collision + free surface
 * - 提供宏观变量访问接口
 *
 * 设计：封装 LBMBackend，提供高层接口
 */

#include "LBMConfig.hpp"
#include "LBMMemoryManager.hpp"
#include "cuda/LBMBackend.hpp"
#include <cstdint>
#include <memory>

namespace lbm {

class LBMCore {
public:
  // 传统构造函数（向后兼容）
  explicit LBMCore(const LBMConfig &config);

  // 新构造函数：使用 MemoryManager（可选）
  LBMCore(const LBMConfig &config, LBMMemoryManager *memMgr);

  ~LBMCore();

  // 禁止拷贝/移动
  LBMCore(const LBMCore &) = delete;
  LBMCore &operator=(const LBMCore &) = delete;

  // === 生命周期 ===
  void initialize();
  void step();

  // === 运行时配置 ===
  void setExternalForce(float3 force);
  void setCollisionModel(CollisionModel model);
  void setFreeSurfaceEnabled(bool enabled);
  void uploadExternalForce(const float *force_host); // Upload AoS force from host
  void setExternalForceFromDeviceAoS(const float3 *force_device_aos); // Upload AoS force from device pointer

  // === 数据访问（设备指针）===
  const float *getDensityField() const;
  const float3 *getVelocityField() const;
  uint8_t *getFlagsDevice() const;
  float *getPhiDevice() const;
  float *getMassDevice() const;

  // === 诊断 ===
  bool checkHealth() const;
  void synchronize() const;

  // === 尺寸信息 ===
  int nx() const { return config_.nx; }
  int ny() const { return config_.ny; }
  int nz() const { return config_.nz; }
  int nCells() const { return nCells_; }

  // === 高级访问 ===
  CudaLBMBackend &backend() { return backend_; }
  const CudaLBMBackend &backend() const { return backend_; }
  void refreshDistributions();
  unsigned long long stepCount() const { return stepCount_; }

  // === 配置访问 ===
  const LBMConfig &config() const { return config_; }
  bool isFreeSurfaceEnabled() const { return config_.enableFreeSurface; }
  CollisionModel collisionModel() const { return config_.collisionModel; }

  // === MemoryManager 访问 ===
  bool hasMemoryManager() const { return memMgr_ != nullptr; }
  LBMMemoryManager* getMemoryManager() const { return memMgr_; }

private:
  LBMConfig config_;
  int nCells_;
  unsigned long long stepCount_;
  bool initialized_;

  // CUDA 后端
  CudaLBMBackend backend_;

  // 额外的 AoS 速度缓冲区（用于外部访问）
  float3 *u_aos_;

  // 可选的内存管理器（不拥有）
  LBMMemoryManager *memMgr_;
  bool buffersRegistered_;

  // 内部方法
  void allocateMemory();
  void freeMemory();
  void packVelocityAoS();
};

} // namespace lbm
