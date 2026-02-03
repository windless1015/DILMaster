/**
 * @file IBMCore.cu
 * @brief IBM CUDA 核心实现 (Multi-Direct Forcing)
 *
 * 实现了基于迭代的 Multi-Direct Forcing (MDF) 算法，
 * 包含高效的 GPU Delta 函数插值和力投射。
 */

#include "IBMCore.hpp"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string("CUDA error: ") +                   \
                               cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

namespace ibm {

// ============================================================================
// Device Helper Functions (Delta Functions)
// ============================================================================
namespace {

// 4-point Cosine Delta Function (Roma et al., 1999)
// Support: [-2, 2]
__device__ inline float phi_4(float r) {
  r = fabsf(r);
  if (r > 2.0f)
    return 0.0f;
  return 0.25f * (1.0f + cosf(M_PI * r * 0.5f));
}

// 辅助：获取网格索引
__device__ inline int get_idx(int x, int y, int z, int nx, int ny, int nz) {
  if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz)
    return -1;
  return z * (nx * ny) + y * nx + x;
}

} // namespace

// ============================================================================
// CUDA Kernels
// ============================================================================
namespace kernels {

// 速度插值内核 (Interpolation)
__global__ void interpolate_velocity(const float3 *marker_pos,
                                     const float3 *grid_vel,
                                     float3 *interp_vel, int nMarkers,
                                     float dx, int nx, int ny, int nz,
                                     float domain_x, float domain_y,
                                     float domain_z, int stencil_width) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nMarkers)
    return;

  float3 pos = marker_pos[i];
  
  float gx = (pos.x - domain_x) / dx;
  float gy = (pos.y - domain_y) / dx;
  float gz = (pos.z - domain_z) / dx;

  int x_start = (int)floorf(gx - 1.5f);
  int y_start = (int)floorf(gy - 1.5f);
  int z_start = (int)floorf(gz - 1.5f);
  int support = 4;

  float3 u_sum = {0.0f, 0.0f, 0.0f};
  float w_sum = 0.0f; 

  for (int k = 0; k < support; ++k) {
    for (int j = 0; j < support; ++j) {
      for (int l = 0; l < support; ++l) { 
        int cx = x_start + l;
        int cy = y_start + j;
        int cz = z_start + k;

        int grid_idx = get_idx(cx, cy, cz, nx, ny, nz);
        
        float dist_x = gx - cx;
        float dist_y = gy - cy;
        float dist_z = gz - cz;
        float w = phi_4(dist_x) * phi_4(dist_y) * phi_4(dist_z);

        if (grid_idx >= 0) {
          float3 u = grid_vel[grid_idx];
          u_sum.x += u.x * w;
          u_sum.y += u.y * w;
          u_sum.z += u.z * w;
          w_sum += w;
        }
      }
    }
  }
  
  if (w_sum > 1e-9f) {
      float inv_w = 1.0f / w_sum;
      interp_vel[i] = make_float3(u_sum.x * inv_w, u_sum.y * inv_w, u_sum.z * inv_w);
  } else {
      interp_vel[i] = make_float3(0.0f, 0.0f, 0.0f);
  }
}

// 标量插值内核 (密度)
__global__ void interpolate_scalar(const float3 *marker_pos,
                                   const float *grid_scalar,
                                   float *interp_scalar, int nMarkers,
                                   float dx, int nx, int ny, int nz,
                                   float domain_x, float domain_y,
                                   float domain_z) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nMarkers) return;

  float3 pos = marker_pos[i];
  float gx = (pos.x - domain_x) / dx;
  float gy = (pos.y - domain_y) / dx;
  float gz = (pos.z - domain_z) / dx;

  int x_start = (int)floorf(gx - 1.5f);
  int y_start = (int)floorf(gy - 1.5f);
  int z_start = (int)floorf(gz - 1.5f);
  int support = 4;

  float s_sum = 0.0f;
  float w_sum = 0.0f;

  for (int k = 0; k < support; ++k) {
    for (int j = 0; j < support; ++j) {
      for (int l = 0; l < support; ++l) { 
        int cx = x_start + l;
        int cy = y_start + j;
        int cz = z_start + k;
        int grid_idx = get_idx(cx, cy, cz, nx, ny, nz);
        float dist_x = gx - cx;
        float dist_y = gy - cy;
        float dist_z = gz - cz;
        float w = phi_4(dist_x) * phi_4(dist_y) * phi_4(dist_z);
        if (grid_idx >= 0) {
          s_sum += grid_scalar[grid_idx] * w;
          w_sum += w;
        }
      }
    }
  }
  
  if (w_sum > 1e-9f) {
      interp_scalar[i] = s_sum / w_sum;
  } else {
      interp_scalar[i] = 1.0f; // Default rho=1
  }
}

// ----------------------------------------------------------------------------
// Masked Kernels (Free Surface Support)
// ----------------------------------------------------------------------------

__global__ void interpolate_velocity_masked(const float3 *marker_pos,
                                            const float3 *grid_vel,
                                            const uint8_t *mask_valid, // 1=Valid, 0=Invalid
                                            float3 *interp_vel, 
                                            unsigned int *fallback_count, // Atomic counter
                                            int nMarkers,
                                            float dx, int nx, int ny, int nz,
                                            float domain_x, float domain_y,
                                            float domain_z, 
                                            float eps) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nMarkers) return;

  float3 pos = marker_pos[i];
  
  float gx = (pos.x - domain_x) / dx;
  float gy = (pos.y - domain_y) / dx;
  float gz = (pos.z - domain_z) / dx;

  int x_start = (int)floorf(gx - 1.5f);
  int y_start = (int)floorf(gy - 1.5f);
  int z_start = (int)floorf(gz - 1.5f);
  int support = 4;

  float3 u_sum = {0.0f, 0.0f, 0.0f};
  float w_sum = 0.0f; 

  for (int k = 0; k < support; ++k) {
    for (int j = 0; j < support; ++j) {
      for (int l = 0; l < support; ++l) { 
        int cx = x_start + l;
        int cy = y_start + j;
        int cz = z_start + k;

        int grid_idx = get_idx(cx, cy, cz, nx, ny, nz);
        
        float dist_x = gx - cx;
        float dist_y = gy - cy;
        float dist_z = gz - cz;
        float w = phi_4(dist_x) * phi_4(dist_y) * phi_4(dist_z);

        if (grid_idx >= 0) {
            // Check Mask
            bool is_valid = (mask_valid == nullptr) || (mask_valid[grid_idx] != 0);
            if (is_valid) {
              float3 u = grid_vel[grid_idx];
              u_sum.x += u.x * w;
              u_sum.y += u.y * w;
              u_sum.z += u.z * w;
              w_sum += w;
            }
        }
      }
    }
  }
  
  if (w_sum > eps) {
      float inv_w = 1.0f / w_sum;
      interp_vel[i] = make_float3(u_sum.x * inv_w, u_sum.y * inv_w, u_sum.z * inv_w);
  } else {
      interp_vel[i] = make_float3(0.0f, 0.0f, 0.0f); // Default safe value
      if (fallback_count) atomicAdd(fallback_count, 1);
  }
}

__global__ void spread_force_masked(const float3 *marker_pos, const float3 *marker_force,
                                    float3 *grid_force, 
                                    const uint8_t *mask_valid,
                                    const float *fill_fraction,
                                    int nMarkers, float dx,
                                    int nx, int ny, int nz, 
                                    float domain_x, float domain_y, float domain_z, 
                                    float vol_scale, bool use_fill) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nMarkers) return;

  float3 pos = marker_pos[i];
  float3 f = marker_force[i];
  
  float gx = (pos.x - domain_x) / dx;
  float gy = (pos.y - domain_y) / dx;
  float gz = (pos.z - domain_z) / dx;

  int x_start = (int)floorf(gx - 1.5f);
  int y_start = (int)floorf(gy - 1.5f);
  int z_start = (int)floorf(gz - 1.5f);
  
  for (int k = 0; k < 4; ++k) {
    for (int j = 0; j < 4; ++j) {
      for (int l = 0; l < 4; ++l) {
        int cx = x_start + l;
        int cy = y_start + j;
        int cz = z_start + k;
        
        int grid_idx = get_idx(cx, cy, cz, nx, ny, nz);
        if (grid_idx >= 0) {
           // Check Mask
           bool is_valid = (mask_valid == nullptr) || (mask_valid[grid_idx] != 0);
           
           if (is_valid) {
               float dist_x = gx - cx;
               float dist_y = gy - cy;
               float dist_z = gz - cz;
               float w = phi_4(dist_x) * phi_4(dist_y) * phi_4(dist_z);
               
               float val = w * vol_scale;
               
               // Apply Fill Weighting
               if (use_fill && fill_fraction != nullptr) {
                   val *= fill_fraction[grid_idx];
               }
               
               atomicAdd(&grid_force[grid_idx].x, f.x * val);
               atomicAdd(&grid_force[grid_idx].y, f.y * val);
               atomicAdd(&grid_force[grid_idx].z, f.z * val);
           }
        }
      }
    }
  }
}

// 计算修正力内核
// dF = rho * (U_target - U_interp) / dt * Area * Beta
__global__ void compute_correction_force(const float3 *marker_vel,
                                         const float3 *interp_vel,
                                         const float *marker_area, 
                                         const float *interp_rho, // [Rho]
                                         float3 *delta_force,
                                         float3 *accum_force, 
                                         float dt,
                                         float beta,
                                         int nMarkers) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nMarkers) return;
    
  float rho = (interp_rho != nullptr) ? interp_rho[i] : 1.0f;
  float area = (marker_area != nullptr) ? marker_area[i] : 1.0f;

  float3 u_t = marker_vel[i]; 
  float3 u_i = interp_vel[i]; 
  
  float3 df;
  float factor = beta * rho * area / dt;
  
  df.x = factor * (u_t.x - u_i.x);
  df.y = factor * (u_t.y - u_i.y);
  df.z = factor * (u_t.z - u_i.z);
  
  delta_force[i] = df;
  
  accum_force[i].x += df.x;
  accum_force[i].y += df.y;
  accum_force[i].z += df.z;
}

// 力投射内核 (Spreading)
__global__ void spread_force(const float3 *marker_pos, const float3 *marker_force,
                             float3 *grid_force, int nMarkers, float dx,
                             int nx, int ny, int nz, float domain_x,
                             float domain_y, float domain_z, float vol_scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nMarkers) return;

  float3 pos = marker_pos[i];
  float3 f = marker_force[i];
  
  float gx = (pos.x - domain_x) / dx;
  float gy = (pos.y - domain_y) / dx;
  float gz = (pos.z - domain_z) / dx;

  int x_start = (int)floorf(gx - 1.5f);
  int y_start = (int)floorf(gy - 1.5f);
  int z_start = (int)floorf(gz - 1.5f);
  
  for (int k = 0; k < 4; ++k) {
    for (int j = 0; j < 4; ++j) {
      for (int l = 0; l < 4; ++l) {
        int cx = x_start + l;
        int cy = y_start + j;
        int cz = z_start + k;
        
        int grid_idx = get_idx(cx, cy, cz, nx, ny, nz);
        if (grid_idx >= 0) {
           float dist_x = gx - cx;
           float dist_y = gy - cy;
           float dist_z = gz - cz;
           float w = phi_4(dist_x) * phi_4(dist_y) * phi_4(dist_z);
           
           float val = w * vol_scale;
           
           atomicAdd(&grid_force[grid_idx].x, f.x * val);
           atomicAdd(&grid_force[grid_idx].y, f.y * val);
           atomicAdd(&grid_force[grid_idx].z, f.z * val);
        }
      }
    }
  }
}

// 更新流体速度内核 (MDF step)
__global__ void update_fluid_vel(float3 *u, const float3 *f_grid, const float* rho_field, 
                                 float dt, float rho_default, int nCells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nCells) return;
  
  float3 force = f_grid[i];
  if (force.x != 0.0f || force.y != 0.0f || force.z != 0.0f) {
      float rho = (rho_field != nullptr) ? rho_field[i] : rho_default;
      float inv_rho = 1.0f / (rho + 1e-9f);
      
      u[i].x += force.x * dt * inv_rho;
      u[i].y += force.y * dt * inv_rho;
      u[i].z += force.z * dt * inv_rho;
  }
}

// 拷贝内核
__global__ void copy_float3_array(const float3* src, float3* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dst[i] = src[i];
}

// AoS -> SoA conversion for grid force
__global__ void convert_force_aos_to_soa(const float3 *f_aos, float *f_soa,
                                         int nCells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nCells) return;
  f_soa[i] = f_aos[i].x;
  f_soa[nCells + i] = f_aos[i].y;
  f_soa[2 * nCells + i] = f_aos[i].z;
}

} // namespace kernels


// ============================================================================
// IBMBackend Implementation
// ============================================================================

IBMBackend::~IBMBackend() { free_memory(); }

void IBMBackend::allocate_memory() {
  size_t n = params_.nMarkers;
  if (n > 0) {
      CUDA_CHECK(cudaMalloc(&position_, n * sizeof(float3)));
      CUDA_CHECK(cudaMalloc(&velocity_, n * sizeof(float3)));
      CUDA_CHECK(cudaMalloc(&force_, n * sizeof(float3)));
      CUDA_CHECK(cudaMalloc(&area_, n * sizeof(float))); 
      
      // MDF Buffers
      CUDA_CHECK(cudaMalloc(&interpolated_velocity_, n * sizeof(float3)));
      CUDA_CHECK(cudaMalloc(&interpolated_density_, n * sizeof(float))); // [Rho]
      CUDA_CHECK(cudaMalloc(&interpolated_density_, n * sizeof(float))); // [Rho]
      CUDA_CHECK(cudaMalloc(&delta_force_, n * sizeof(float3)));
  }
  
  // Debug Counter
  CUDA_CHECK(cudaMalloc(&d_fallback_count_, sizeof(unsigned int)));

  // Grid Buffer
  size_t nCells = params_.nx * params_.ny * params_.nz;
  if (nCells > 0) {
      CUDA_CHECK(cudaMalloc(&temp_fluid_velocity_, nCells * sizeof(float3)));
  }
}

void IBMBackend::free_memory() {
  if (position_) cudaFree(position_);
  if (velocity_) cudaFree(velocity_);
  if (force_) cudaFree(force_);
  if (area_) cudaFree(area_);
  if (interpolated_velocity_) cudaFree(interpolated_velocity_);
  if (interpolated_density_) cudaFree(interpolated_density_);
  if (delta_force_) cudaFree(delta_force_);
  if (temp_fluid_velocity_) cudaFree(temp_fluid_velocity_);
  if (d_fallback_count_) cudaFree(d_fallback_count_);
  
  position_ = nullptr;
  velocity_ = nullptr;
  force_ = nullptr;
  area_ = nullptr;
  interpolated_velocity_ = nullptr;
  interpolated_density_ = nullptr;
  delta_force_ = nullptr;
  temp_fluid_velocity_ = nullptr;
  d_fallback_count_ = nullptr;
}

void IBMBackend::initialize(const IBMParams &params) {
    if (allocated_) free_memory();
    params_ = params;
    allocate_memory();
    
    if (position_) CUDA_CHECK(cudaMemset(position_, 0, params_.nMarkers * sizeof(float3)));
    if (velocity_) CUDA_CHECK(cudaMemset(velocity_, 0, params_.nMarkers * sizeof(float3)));
    if (force_) CUDA_CHECK(cudaMemset(force_, 0, params_.nMarkers * sizeof(float3)));
    if (area_) CUDA_CHECK(cudaMemset(area_, 0, params_.nMarkers * sizeof(float)));
    if (interpolated_density_) CUDA_CHECK(cudaMemset(interpolated_density_, 0, params_.nMarkers * sizeof(float)));
    if (d_fallback_count_) CUDA_CHECK(cudaMemset(d_fallback_count_, 0, sizeof(unsigned int)));
    allocated_ = true;
}

void IBMBackend::interpolateVelocity(const float3* grid_u, const uint8_t* mask) {
    int n = (int)params_.nMarkers;
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    if (params_.enable_masked_fs) {
        // Reset counter before interp? Or accumulate? Usually reset per step or per call.
        // Let's not reset inside helper, user might want cumulative.
        // Actually, we should probably reset at start of computeForces.
        kernels::interpolate_velocity_masked<<<gridSize, blockSize>>>(
            position_, grid_u, mask, interpolated_velocity_, d_fallback_count_, n,
            params_.dx, params_.nx, params_.ny, params_.nz,
            params_.domain_origin_x, params_.domain_origin_y, params_.domain_origin_z,
            params_.mask_eps
        );
    } else {
        kernels::interpolate_velocity<<<gridSize, blockSize>>>(
            position_, grid_u, interpolated_velocity_, n,
            params_.dx, params_.nx, params_.ny, params_.nz,
            params_.domain_origin_x, params_.domain_origin_y, params_.domain_origin_z,
            params_.stencil_width
        );
    }
    CUDA_CHECK(cudaGetLastError());
}

// [Rho]
void IBMBackend::computeForces(const float3 *fluid_velocity, const float *fluid_density,
                               float3 *fluid_force_out, float dt,
                               const uint8_t *valid_mask, const float *fill_fraction) {
   if (!allocated_) return;
   
   int nCells = params_.nx * params_.ny * params_.nz;
   int blockSize = 256;
   int gridSize = (nCells + blockSize - 1) / blockSize;
   
   // 1. 初始化
   kernels::copy_float3_array<<<gridSize, blockSize>>>(fluid_velocity, temp_fluid_velocity_, nCells);
   CUDA_CHECK(cudaMemset(force_, 0, params_.nMarkers * sizeof(float3)));
   CUDA_CHECK(cudaMemset(fluid_force_out, 0, nCells * sizeof(float3)));
   // Reset fallback count for this step
   if (d_fallback_count_) CUDA_CHECK(cudaMemset(d_fallback_count_, 0, sizeof(unsigned int)));
   
   // 2. 插值流体密度 (只做一次)
   int mGrid = ((int)params_.nMarkers + blockSize - 1) / blockSize;
   if (fluid_density) {
       kernels::interpolate_scalar<<<mGrid, blockSize>>>(
           position_, fluid_density, interpolated_density_, (int)params_.nMarkers,
           params_.dx, params_.nx, params_.ny, params_.nz,
           params_.domain_origin_x, params_.domain_origin_y, params_.domain_origin_z
       );
   }
   
   // 3. MDF 迭代
   for (int k = 0; k < params_.mdf_iterations; ++k) {
       // A. 插值 (From current u_pred)
       interpolateVelocity(temp_fluid_velocity_, valid_mask);
       
       // B. 计算修正力 delta_F
       kernels::compute_correction_force<<<mGrid, blockSize>>>(
           velocity_, interpolated_velocity_, area_, 
           (fluid_density ? interpolated_density_ : nullptr), 
           delta_force_, force_,
           dt, params_.mdf_beta, (int)params_.nMarkers
       );
       
       // C. Spread
       spreadForce(delta_force_, fluid_force_out, valid_mask, fill_fraction);
       
       // D. Update u_pred
       // Note: updateTempVelocity modifies temp_fluid_velocity_ in place based on grid_force
       // We need to re-copy base velocity first? 
       // Logic: U_pred = U_base + Force * dt / rho
       // So we copy base, then add force.
       kernels::copy_float3_array<<<gridSize, blockSize>>>(fluid_velocity, temp_fluid_velocity_, nCells);
       updateTempVelocity(fluid_force_out, fluid_density, dt);
   }
   
   CUDA_CHECK(cudaGetLastError());
}

void IBMBackend::spreadForce(const float3* marker_force, float3* grid_force,
                             const uint8_t* mask, const float* fill) {
    int n = (int)params_.nMarkers;
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // Scale factor: 1/dx^3
    float dVol = params_.dx * params_.dx * params_.dx;
    float vol_scale = 1.0f / dVol;
    
    if (params_.enable_masked_fs) {
        kernels::spread_force_masked<<<gridSize, blockSize>>>(
            position_, marker_force, grid_force, 
            mask, fill,
            n, params_.dx, params_.nx, params_.ny, params_.nz,
            params_.domain_origin_x, params_.domain_origin_y, params_.domain_origin_z,
            vol_scale, params_.use_fill_weight
        );
    } else {
        kernels::spread_force<<<gridSize, blockSize>>>(
            position_, marker_force, grid_force, n,
            params_.dx, params_.nx, params_.ny, params_.nz,
            params_.domain_origin_x, params_.domain_origin_y, params_.domain_origin_z,
            vol_scale
        );
    }
    CUDA_CHECK(cudaGetLastError());
}

unsigned int IBMBackend::getFallbackCount() const {
    if (!allocated_ || !d_fallback_count_) return 0;
    unsigned int h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_fallback_count_, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    return h_count;
}

void IBMBackend::updateTempVelocity(const float3* grid_force, const float* rho, float dt) {
    int nCells = params_.nx * params_.ny * params_.nz;
    int blockSize = 256;
    int gridSize = (nCells + blockSize - 1) / blockSize;
    
    float rho_default = 1.0f; 
    
    kernels::update_fluid_vel<<<gridSize, blockSize>>>(
        temp_fluid_velocity_, grid_force, rho, dt, rho_default, nCells
    );
    CUDA_CHECK(cudaGetLastError());
}

// ... (Other backend methods wrappers) ...
void IBMBackend::uploadPositions(const float3 *h_p) {
    if (allocated_) CUDA_CHECK(cudaMemcpy(position_, h_p, params_.nMarkers * sizeof(float3), cudaMemcpyHostToDevice));
}
void IBMBackend::uploadVelocities(const float3 *h_v) {
    if (allocated_) CUDA_CHECK(cudaMemcpy(velocity_, h_v, params_.nMarkers * sizeof(float3), cudaMemcpyHostToDevice));
}
void IBMBackend::uploadAreas(const float *h_a) {
    if (allocated_) CUDA_CHECK(cudaMemcpy(area_, h_a, params_.nMarkers * sizeof(float), cudaMemcpyHostToDevice));
}
void IBMBackend::downloadPositions(float3 *h_p) const {
    if (allocated_) CUDA_CHECK(cudaMemcpy(h_p, position_, params_.nMarkers * sizeof(float3), cudaMemcpyDeviceToHost));
}
void IBMBackend::downloadForces(float3 *h_f) const {
    if (allocated_) CUDA_CHECK(cudaMemcpy(h_f, force_, params_.nMarkers * sizeof(float3), cudaMemcpyDeviceToHost));
}
void IBMBackend::downloadVelocities(float3 *h_v) const {
    if (allocated_) CUDA_CHECK(cudaMemcpy(h_v, velocity_, params_.nMarkers * sizeof(float3), cudaMemcpyDeviceToHost));
}

#ifdef IBM_TESTING
void IBMBackend::downloadInterpolatedVelocity(float3 *h_u) const {
    if (allocated_ && interpolated_velocity_) {
        CUDA_CHECK(cudaMemcpy(h_u, interpolated_velocity_, params_.nMarkers * sizeof(float3), cudaMemcpyDeviceToHost));
    }
}
#endif

void IBMBackend::clearForces() {
     if (allocated_) CUDA_CHECK(cudaMemset(force_, 0, params_.nMarkers * sizeof(float3)));
}
void IBMBackend::synchronize() const {
    CUDA_CHECK(cudaDeviceSynchronize());
}

void IBMBackend::updateMarkers(const float3 *new_pos, const float3 *new_vel) {
    if (new_pos) uploadPositions(new_pos);
    if (new_vel) uploadVelocities(new_vel);
}
void IBMBackend::convertForceAoSToSoA(const float3 *force_aos, float *force_soa,
                                      int nCells) const {
    if (!force_aos || !force_soa || nCells <= 0) return;
    int blockSize = 256;
    int gridSize = (nCells + blockSize - 1) / blockSize;
    kernels::convert_force_aos_to_soa<<<gridSize, blockSize>>>(force_aos, force_soa, nCells);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// IBMCore Implementation
// ============================================================================

IBMCore::IBMCore(std::size_t nMarkers, float dx, int nx, int ny, int nz) {
  params_.nMarkers = nMarkers;
  params_.dx = dx;
  params_.nx = nx;
  params_.ny = ny;
  params_.nz = nz;
}

IBMCore::IBMCore(const IBMParams &params) : params_(params) {}
IBMCore::~IBMCore() = default;

void IBMCore::initialize() {
  if (!initialized_) {
      backend_.initialize(params_);
      initialized_ = true;
  }
}

void IBMCore::updateMarkers(const float3 *new_pos, const float3 *new_vel) {
    if (!initialized_) initialize();
    backend_.updateMarkers(new_pos, new_vel);
}

void IBMCore::updateMarkers(const float3 *new_pos, const float3 *new_vel, const float *new_areas) {
    if (!initialized_) initialize();
    backend_.uploadPositions(new_pos);
    backend_.uploadVelocities(new_vel);
    backend_.uploadAreas(new_areas);
}

void IBMCore::computeForces(const float3 *fluid_velocity, const float *fluid_density,
                            float3 *fluid_force_out, float dt,
                            const uint8_t *valid_mask, const float *fill_fraction) {
    if (!initialized_) initialize();
    backend_.computeForces(fluid_velocity, fluid_density, fluid_force_out, dt, valid_mask, fill_fraction);
}

void IBMCore::clearForces() { backend_.clearForces(); }

void IBMCore::uploadPositions(const float3 *p) { backend_.uploadPositions(p); }
void IBMCore::uploadVelocities(const float3 *v) { backend_.uploadVelocities(v); }
void IBMCore::uploadAreas(const float *a) { backend_.uploadAreas(a); } // [Area]
void IBMCore::downloadPositions(float3 *p) const { backend_.downloadPositions(p); }
void IBMCore::downloadForces(float3 *f) const { backend_.downloadForces(f); }

void IBMCore::applyRotation(float3 axis, float3 center, float angle) { }
void IBMCore::applyTranslation(float3 displacement) { }
bool IBMCore::checkHealth() const { return backend_.is_initialized(); }
void IBMCore::synchronize() const { backend_.synchronize(); }

} // namespace ibm
