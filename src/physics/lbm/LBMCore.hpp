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
 * 注意：LBMCore 不管理 GPU 内存分配，所有字段内存由 CudaLBMBackend 内部管理。
 *       外部通过 *DevicePtr() 访问器获取设备指针。
 */

#include "LBMConfig.hpp"
#include "cuda/LBMBackend.hpp"
#include <cstdint>
#include <memory>

namespace lbm {

class LBMCore {
public:
  explicit LBMCore(const LBMConfig &config);
  ~LBMCore();

  // 禁止拷贝/移动
  LBMCore(const LBMCore &) = delete;
  LBMCore &operator=(const LBMCore &) = delete;

  // === 生命周期 ===
  void initialize();
  void step();

  // === 分阶段步进（供模块钩子使用）===
  void streamCollide();      // 仅执行 streaming + collision
  void updateMacroscopic();  // 更新宏观场 + 打包速度 + 同步 + stepCount++

  // === 运行时配置 ===
  void setExternalForce(float3 force);
  void setCollisionModel(CollisionModel model);
  void setFreeSurfaceEnabled(bool enabled);
  void uploadExternalForce(const float *force_host);
  void setExternalForceFromDeviceAoS(const float3 *force_device_aos);

  // === 设备指针访问器（绑定接口）===
  float *densityDevicePtr() const { return backend_.rho_device(); }
  float *velocityDevicePtr() const { return backend_.u_device(); }    // SoA
  float3 *velocityAoSPtr() const { return u_aos_; }
  uint8_t *flagsDevicePtr() const { return backend_.flags_device(); }
  float *phiDevicePtr() const { return backend_.phi_device(); }
  float *massDevicePtr() const { return backend_.mass_device(); }
  float *massExDevicePtr() const { return backend_.massex_device(); }
  float *forceDevicePtr() const { return backend_.force_device(); }

  // === 旧版访问器（兼容性）===
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

private:
  LBMConfig config_;
  int nCells_;
  unsigned long long stepCount_;
  bool initialized_;

  // CUDA 后端
  CudaLBMBackend backend_;

  // 额外的 AoS 速度缓冲区（用于外部访问）
  float3 *u_aos_;

  // 内部方法
  void allocateMemory();
  void freeMemory();
  void packVelocityAoS();
};

} // namespace lbm
