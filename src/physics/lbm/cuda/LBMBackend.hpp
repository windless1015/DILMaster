#pragma once
/**
 * LBMBackend.hpp - LBM CUDA 后端接口
 *
 * 替代原 lbm_cuda.hpp，提供相同的公共接口
 * 内部使用模块化内核
 */

#include "LBMTypes.cuh"
#include <cuda_runtime.h>

namespace lbm {
namespace cuda {

/**
 * @brief CUDA LBM 后端
 *
 * 管理 GPU 内存和内核调用
 * 外部接口与原 CudaLBMBackend 保持一致
 */
class LBMBackend {
public:
  LBMBackend() = default;
  ~LBMBackend();

  // 禁止拷贝
  LBMBackend(const LBMBackend &) = delete;
  LBMBackend &operator=(const LBMBackend &) = delete;

  // ========================================================================
  // 初始化
  // ========================================================================
  void initialize(const LBMParams &params);
  bool is_initialized() const { return allocated_; }

  // ========================================================================
  // 主机-设备数据传输
  // ========================================================================
  void upload_host_fields(const float *rho_h, const float *u_h,
                          const unsigned char *flags_h, const float *phi_h);
  void upload_force(const float *force_soa_h); // New method
  void upload_force_from_device_aos(const float3 *force_aos_d); // New method efficient coupling
  void download_fields(float *rho_h, float *u_h, unsigned char *flags_h,
                       float *phi_h) const;

  // ========================================================================
  // 内核调用
  // ========================================================================
  void kernel_initialize();
  void kernel_stream_collide(unsigned long long t);
  void kernel_update_fields(unsigned long long t);

  // 自由表面内核 (仅当 enableFreeSurface 为 true 时有效)
  void kernel_surface_capture_outgoing(unsigned long long t);
  void kernel_surface_mass_exchange();
  void kernel_surface_flag_transition(unsigned long long t);
  void kernel_surface_phi_recompute();

  // Boundary Ops
  void kernel_setup_boundaries();
  void kernel_apply_boundaries(unsigned long long t);

  // ========================================================================
  // 同步和参数设置
  // ========================================================================
  void synchronize() const;
  void set_force(float fx, float fy, float fz);
  void set_collision_model(CollisionModel model);
  void set_free_surface_enabled(bool enabled);

  // ========================================================================
  // 设备指针访问
  // ========================================================================
  float *rho_device() const { return rho_; }
  float *u_device() const { return u_; }
  unsigned char *flags_device() const { return flags_; }
  float *phi_device() const { return phi_; }
  float *mass_device() const { return mass_; }
  float *massex_device() const { return massex_; }
  float *force_device() const { return force_; } // New accessor
  void *fi_device() const { return fi_; }

  // ========================================================================
  // 参数访问
  // ========================================================================
  const LBMParams &params() const { return params_; }
  unsigned long long N() const { return params_.N; }

private:
  LBMParams params_{};
  bool allocated_ = false;

  // 设备指针
  void *fi_ = nullptr;             // 分布函数 (FP16)
  float *rho_ = nullptr;           // 密度
  float *u_ = nullptr;             // 速度 (SoA: ux, uy, uz)
  unsigned char *flags_ = nullptr; // 单元格标志
  float *mass_ = nullptr;          // 质量
  float *massex_ = nullptr;        // 多余质量
  float *phi_ = nullptr;           // 填充分数
  float *force_ = nullptr;         // 外部力场 (SoA: fx, fy, fz) - 默认为 nullptr，按需分配
};

} // namespace cuda

// 兼容性别名 (保持与原 CudaLBMBackend 接口一致)
using CudaLBMBackend = cuda::LBMBackend;
using CudaLBMParams = cuda::LBMParams;

} // namespace lbm
