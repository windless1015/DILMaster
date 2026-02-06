/**
 * LBMBackend.cu - LBM CUDA 后端实现
 *
 * 简化后的实现，使用合并的头文件
 */

#include "LBMBackend.hpp"
#include "LBMTypes.cuh"
#include "cuda_utils.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>


namespace lbm {
namespace cuda {

// 前向声明内核 (定义在 LBMKernels.cu)
__global__ void kernel_initialize(__half *fi, const float *rho, float *u,
                                  unsigned char *flags, float *mass,
                                  float *massex, float *phi, LBMParams params);
__global__ void kernel_stream_collide(__half *fi, float *rho, float *u,
                                      unsigned char *flags,
                                      const unsigned long long t,
                                      const float *mass, const float *force, LBMParams params);
__global__ void kernel_update_fields(const __half *fi, float *rho, float *u,
                                     const unsigned char *flags,
                                     const unsigned long long t,
                                     const float *force, LBMParams params);
__global__ void kernel_surface_capture(__half *fi, const float *rho,
                                       const float *u,
                                       const unsigned char *flags, float *mass,
                                       const float *massex, const float *phi,
                                       const unsigned long long t,
                                       LBMParams params);
__global__ void kernel_surface_propagate(unsigned char *flags,
                                         LBMParams params);
__global__ void kernel_surface_transition(__half *fi, const float *rho,
                                          const float *u, unsigned char *flags,
                                          const unsigned long long t,
                                          LBMParams params);
__global__ void kernel_surface_finalize(const float *rho, unsigned char *flags,
                                        float *mass, float *massex, float *phi,
                                        LBMParams params);

// Boundary Kernels
__global__ void kernel_setup_boundaries(unsigned char* flags, LBMParams params);
__global__ void kernel_apply_boundaries(__half* fi, float* rho, float* u, 
                                        const unsigned char* flags, 
                                        unsigned long long t, 
                                        LBMParams params);

LBMBackend::~LBMBackend() {
  if (fi_)
    cudaFree(fi_);
  if (rho_)
    cudaFree(rho_);
  if (u_)
    cudaFree(u_);
  if (flags_)
    cudaFree(flags_);
  if (mass_)
    cudaFree(mass_);
  if (massex_)
    cudaFree(massex_);
  if (phi_)
    cudaFree(phi_);
  if (force_)
    cudaFree(force_);
}

void LBMBackend::initialize(const LBMParams &params) {
  params_ = params;
  if (allocated_)
    return;

  const size_t N = params_.N;
  CUDA_CHECK(cudaMalloc(&fi_, sizeof(__half) * N * kVelocitySet));
  CUDA_CHECK(cudaMalloc(&rho_, sizeof(float) * N));
  CUDA_CHECK(cudaMalloc(&u_, sizeof(float) * N * 3ull));
  CUDA_CHECK(cudaMalloc(&flags_, sizeof(unsigned char) * N));
  CUDA_CHECK(cudaMalloc(&mass_, sizeof(float) * N));
  CUDA_CHECK(cudaMalloc(&massex_, sizeof(float) * N));
  CUDA_CHECK(cudaMalloc(&phi_, sizeof(float) * N));
  allocated_ = true;
}

void LBMBackend::upload_host_fields(const float *rho_h, const float *u_h,
                                    const unsigned char *flags_h,
                                    const float *phi_h) {
  if (!allocated_)
    return;
  CUDA_CHECK(cudaMemcpy(rho_, rho_h, sizeof(float) * params_.N,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(u_, u_h, sizeof(float) * params_.N * 3ull,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(flags_, flags_h, sizeof(unsigned char) * params_.N,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(phi_, phi_h, sizeof(float) * params_.N,
                        cudaMemcpyHostToDevice));
}

void LBMBackend::download_fields(float *rho_h, float *u_h,
                                 unsigned char *flags_h, float *phi_h) const {
  if (!allocated_)
    return;
  CUDA_CHECK(cudaMemcpy(rho_h, rho_, sizeof(float) * params_.N,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(u_h, u_, sizeof(float) * params_.N * 3ull,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(flags_h, flags_, sizeof(unsigned char) * params_.N,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(phi_h, phi_, sizeof(float) * params_.N,
                        cudaMemcpyDeviceToHost));
}

void LBMBackend::kernel_initialize() {
  const dim3 block(256), grid((unsigned int)((params_.N + 255ull) / 256ull));
  lbm::cuda::kernel_initialize<<<grid, block>>>((__half *)fi_, rho_, u_, flags_,
                                                mass_, massex_, phi_, params_);
  CUDA_CHECK(cudaGetLastError());
}

void LBMBackend::kernel_stream_collide(unsigned long long t) {
  const dim3 block(256), grid((unsigned int)((params_.N + 255ull) / 256ull));
  lbm::cuda::kernel_stream_collide<<<grid, block>>>((__half *)fi_, rho_, u_,
                                                    flags_, t, mass_, force_, params_);
  CUDA_CHECK(cudaGetLastError());
}

void LBMBackend::kernel_update_fields(unsigned long long t) {
  const dim3 block(256), grid((unsigned int)((params_.N + 255ull) / 256ull));
  lbm::cuda::kernel_update_fields<<<grid, block>>>((__half *)fi_, rho_, u_,
                                                   flags_, t, force_, params_);
  CUDA_CHECK(cudaGetLastError());
}

void LBMBackend::kernel_surface_capture_outgoing(unsigned long long t) {
  if (!params_.enableFreeSurface)
    return;
  const dim3 block(256), grid((unsigned int)((params_.N + 255ull) / 256ull));
  lbm::cuda::kernel_surface_capture<<<grid, block>>>(
      (__half *)fi_, rho_, u_, flags_, mass_, massex_, phi_, t, params_);
  CUDA_CHECK(cudaGetLastError());
}

void LBMBackend::kernel_surface_mass_exchange() {
  if (!params_.enableFreeSurface)
    return;
  const dim3 block(256), grid((unsigned int)((params_.N + 255ull) / 256ull));
  lbm::cuda::kernel_surface_propagate<<<grid, block>>>(flags_, params_);
  CUDA_CHECK(cudaGetLastError());
}

void LBMBackend::kernel_surface_flag_transition(unsigned long long t) {
  if (!params_.enableFreeSurface)
    return;
  const dim3 block(256), grid((unsigned int)((params_.N + 255ull) / 256ull));
  lbm::cuda::kernel_surface_transition<<<grid, block>>>((__half *)fi_, rho_, u_,
                                                        flags_, t, params_);
  CUDA_CHECK(cudaGetLastError());
}

void LBMBackend::kernel_surface_phi_recompute() {
  if (!params_.enableFreeSurface)
    return;
  const dim3 block(256), grid((unsigned int)((params_.N + 255ull) / 256ull));
  lbm::cuda::kernel_surface_finalize<<<grid, block>>>(rho_, flags_, mass_,
                                                      massex_, phi_, params_);
  CUDA_CHECK(cudaGetLastError());
}

void LBMBackend::kernel_setup_boundaries() {
  const dim3 block(256), grid((unsigned int)((params_.N + 255ull) / 256ull));
  lbm::cuda::kernel_setup_boundaries<<<grid, block>>>(flags_, params_);
  CUDA_CHECK(cudaGetLastError());
}

void LBMBackend::kernel_apply_boundaries(unsigned long long t) {
  const dim3 block(256), grid((unsigned int)((params_.N + 255ull) / 256ull));
  lbm::cuda::kernel_apply_boundaries<<<grid, block>>>((__half*)fi_, rho_, u_, flags_, t, params_);
  CUDA_CHECK(cudaGetLastError());
}

void LBMBackend::synchronize() const { CUDA_CHECK(cudaDeviceSynchronize()); }
void LBMBackend::set_force(float fx, float fy, float fz) {
  params_.fx = fx;
  params_.fy = fy;
  params_.fz = fz;
}

void LBMBackend::set_collision_model(CollisionModel model) {
  params_.collisionModel = model;
}

void LBMBackend::set_free_surface_enabled(bool enabled) {
  params_.enableFreeSurface = enabled;
}


void LBMBackend::upload_force(const float *force_soa_h) {
  if (!allocated_) return;
  if (!force_) {
      CUDA_CHECK(cudaMalloc(&force_, sizeof(float) * params_.N * 3ull));
      CUDA_CHECK(cudaMemset(force_, 0, sizeof(float) * params_.N * 3ull));
  }
  if (force_soa_h) {
      CUDA_CHECK(cudaMemcpy(force_, force_soa_h, sizeof(float) * params_.N * 3ull, cudaMemcpyHostToDevice));
  } else {
      CUDA_CHECK(cudaMemset(force_, 0, sizeof(float) * params_.N * 3ull));
  }
}

namespace {
__global__ void ker_convert_aos_to_soa(const float3 *f_aos, float *f_soa, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  float3 f = f_aos[idx];
  f_soa[idx] = f.x;
  f_soa[N + idx] = f.y;
  f_soa[2 * N + idx] = f.z;
}
}

void LBMBackend::upload_force_from_device_aos(const float3 *force_aos_d) {
    if (!allocated_) return;
    if (!force_) {
        CUDA_CHECK(cudaMalloc(&force_, sizeof(float) * params_.N * 3ull));
    }
    
    if (force_aos_d) {
        int blockSize = 256;
        int gridSize = (params_.N + blockSize - 1) / blockSize;
        ker_convert_aos_to_soa<<<gridSize, blockSize>>>(force_aos_d, force_, (int)params_.N);
        CUDA_CHECK(cudaGetLastError());
    } else {
        CUDA_CHECK(cudaMemset(force_, 0, sizeof(float) * params_.N * 3ull));
    }
}

} // namespace cuda
} // namespace lbm
