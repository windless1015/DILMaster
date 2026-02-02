/**
 * LBMKernels.cu - LBM CUDA 内核实现
 *
 * 使用合并后的 LBMKernelHelpers.cuh
 */

#include "LBMKernelHelpers.cuh"
#include "LBMTypes.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>


namespace lbm {
namespace cuda {

// ============================================================================
// 初始化内核
// ============================================================================
__global__ void kernel_initialize(__half *fi, const float *rho, float *u,
                                  unsigned char *flags, float *mass,
                                  float *massex, float *phi, LBMParams params) {
  const unsigned long long n =
      (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= params.N)
    return;

  unsigned char flagsn = flags[n];
  const unsigned char flagsn_bo = flagsn & CellFlag::BOUNDARY_MASK;

  unsigned long long j[kVelocitySet];
  neighbors(n, params.Nx, params.Ny, params.Nz, j);

  unsigned char flagsj[kVelocitySet];
  for (unsigned int i = 1u; i < kVelocitySet; i++)
    flagsj[i] = flags[j[i]];

  if (flagsn_bo == CellFlag::SOLID) {
    u[n] = 0.0f;
    u[params.N + n] = 0.0f;
    u[2ull * params.N + n] = 0.0f;
  }

  float feq[kVelocitySet];
  calculate_f_eq(rho[n], u[n], u[params.N + n], u[2ull * params.N + n], feq);

  float phin = phi[n];
  if (!(flagsn & (CellFlag::SOLID | CellFlag::FLUID | CellFlag::INTERFACE))) {
    flagsn = (flagsn & ~CellFlag::SURFACE_MASK) | CellFlag::GAS;
  }

  if (params.enableFreeSurface &&
      (flagsn & CellFlag::SURFACE_MASK) == CellFlag::GAS) {
    bool change = false;
    for (unsigned int i = 1u; i < kVelocitySet; i++)
      change =
          change || ((flagsj[i] & CellFlag::SURFACE_MASK) == CellFlag::FLUID);
    if (change) {
      flagsn = (flagsn & ~CellFlag::SURFACE_MASK) | CellFlag::INTERFACE;
      phin = 0.5f;
      float rhon, uxn, uyn, uzn;
      average_neighbors_fluid(n, rho, u, flags, params.Nx, params.Ny, params.Nz,
                              &rhon, &uxn, &uyn, &uzn);
      calculate_f_eq(rhon, uxn, uyn, uzn, feq);
    }
  }

  if ((flagsn & CellFlag::SURFACE_MASK) == CellFlag::GAS) {
    u[n] = 0.0f;
    u[params.N + n] = 0.0f;
    u[2ull * params.N + n] = 0.0f;
    phin = 0.0f;
  } else if ((flagsn & CellFlag::SURFACE_MASK) == CellFlag::INTERFACE &&
             (phin < 0.0f || phin > 1.0f)) {
    phin = 0.5f;
  } else if ((flagsn & CellFlag::SURFACE_MASK) == CellFlag::FLUID) {
    phin = 1.0f;
  }

  phi[n] = phin;
  mass[n] = phin * rho[n];
  massex[n] = 0.0f;
  flags[n] = flagsn;
  store_f(n, feq, fi, j, 1ull, params.N);
}

// ============================================================================
// 流动碰撞内核
// ============================================================================
__global__ void kernel_stream_collide(__half *fi, float *rho, float *u,
                                      unsigned char *flags,
                                      const unsigned long long t,
                                      const float *mass, const float *force,
                                      LBMParams params) {
  const unsigned long long n =
      (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= params.N)
    return;

  const unsigned char flagsn = flags[n];
  if ((flagsn & CellFlag::BOUNDARY_MASK) == CellFlag::SOLID ||
      (flagsn & CellFlag::SURFACE_MASK) == CellFlag::GAS)
    return;

  unsigned long long j[kVelocitySet];
  neighbors(n, params.Nx, params.Ny, params.Nz, j);
  float fhn[kVelocitySet];
  load_f(n, fhn, fi, j, t, params.N);
  float rhon, uxn, uyn, uzn;
  calculate_rho_u(fhn, &rhon, &uxn, &uyn, &uzn);

  if (params.enableFreeSurface &&
      (flagsn & CellFlag::SURFACE_MASK) == CellFlag::INTERFACE) {
    bool no_f = true, no_g = true;
    for (unsigned int i = 1u; i < kVelocitySet; i++) {
      const unsigned char flagsji_su = flags[j[i]] & CellFlag::SURFACE_MASK;
      no_f = no_f && flagsji_su != CellFlag::FLUID;
      no_g = no_g && flagsji_su != CellFlag::GAS;
    }
    const float massn = mass[n];
    if (massn > rhon || no_g)
      flags[n] = (flagsn & ~CellFlag::SURFACE_MASK) | CellFlag::TO_FLUID;
    else if (massn < 0.0f || no_f)
      flags[n] = (flagsn & ~CellFlag::SURFACE_MASK) | CellFlag::TO_GAS;
  }

  const float dt = 1.0f;
  const float rho2 = 0.5f * dt / rhon;
  
  // Apply Global + Local Forces
  float fx = params.fx;
  float fy = params.fy;
  float fz = params.fz;
  
  if (force) {
      fx += force[n];
      fy += force[params.N + n];
      fz += force[2ull * params.N + n];
  }
  
  uxn = clampf(fmaf(fx, rho2, uxn), -kMaxVelocity, kMaxVelocity);
  uyn = clampf(fmaf(fy, rho2, uyn), -kMaxVelocity, kMaxVelocity);
  uzn = clampf(fmaf(fz, rho2, uzn), -kMaxVelocity, kMaxVelocity);

  float Fin[kVelocitySet];
  calculate_forcing_terms(uxn, uyn, uzn, fx, fy, fz, Fin);
  rho[n] = rhon;
  u[n] = uxn;
  u[params.N + n] = uyn;
  u[2ull * params.N + n] = uzn;

  float feq[kVelocitySet];
  calculate_f_eq(rhon, uxn, uyn, uzn, feq);
  collide_dispatch(fhn, feq, Fin, params.w, params);
  store_f(n, fhn, fi, j, t, params.N);
}

// ============================================================================
// 更新宏观场内核
// ============================================================================
__global__ void kernel_update_fields(const __half *fi, float *rho, float *u,
                                     const unsigned char *flags,
                                     const unsigned long long t,
                                     const float *force,
                                     LBMParams params) {
  const unsigned long long n =
      (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= params.N)
    return;
  const unsigned char flagsn = flags[n];
  if ((flagsn & CellFlag::BOUNDARY_MASK) == CellFlag::SOLID ||
      (flagsn & CellFlag::SURFACE_MASK) == CellFlag::GAS)
    return;

  unsigned long long j[kVelocitySet];
  neighbors(n, params.Nx, params.Ny, params.Nz, j);
  float fhn[kVelocitySet];
  load_f(n, fhn, fi, j, t, params.N);
  float rhon, uxn, uyn, uzn;
  calculate_rho_u(fhn, &rhon, &uxn, &uyn, &uzn);
  const float dt = 1.0f;
  const float rho2 = 0.5f * dt / rhon;
  float fx = params.fx;
  float fy = params.fy;
  float fz = params.fz;
  if (force) {
    fx += force[n];
    fy += force[params.N + n];
    fz += force[2ull * params.N + n];
  }
  rho[n] = rhon;
  u[n] = clampf(fmaf(fx, rho2, uxn), -kMaxVelocity, kMaxVelocity);
  u[params.N + n] =
      clampf(fmaf(fy, rho2, uyn), -kMaxVelocity, kMaxVelocity);
  u[2ull * params.N + n] =
      clampf(fmaf(fz, rho2, uzn), -kMaxVelocity, kMaxVelocity);
}

// ============================================================================
// 自由表面内核 - 质量捕获
// ============================================================================
__global__ void kernel_surface_capture(__half *fi, const float *rho,
                                       const float *u,
                                       const unsigned char *flags, float *mass,
                                       const float *massex, const float *phi,
                                       const unsigned long long t,
                                       LBMParams params) {
  const unsigned long long n =
      (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= params.N)
    return;
  const unsigned char flagsn = flags[n],
                      flagsn_su = flagsn & CellFlag::SURFACE_MASK;
  if ((flagsn & CellFlag::BOUNDARY_MASK) == CellFlag::SOLID ||
      flagsn_su == CellFlag::GAS)
    return;

  unsigned long long j[kVelocitySet];
  neighbors(n, params.Nx, params.Ny, params.Nz, j);
  float fhn[kVelocitySet];
  load_f(n, fhn, fi, j, t, params.N);
  float fon[kVelocitySet];
  fon[0] = fhn[0];
  load_f_outgoing(n, fon, fi, j, t, params.N);
  float massn = mass[n];
  for (unsigned int i = 1u; i < kVelocitySet; i++)
    massn += massex[j[i]];

  if (flagsn_su == CellFlag::FLUID) {
    for (unsigned int i = 1u; i < kVelocitySet; i++)
      massn += fhn[i] - fon[i];
  } else if (flagsn_su == CellFlag::INTERFACE) {
    float phij[kVelocitySet];
    for (unsigned int i = 1u; i < kVelocitySet; i++)
      phij[i] = phi[j[i]];
    float rhon, uxn, uyn, uzn;
    calculate_rho_u(fon, &rhon, &uxn, &uyn, &uzn);
    uxn = clampf(uxn, -kMaxVelocity, kMaxVelocity);
    uyn = clampf(uyn, -kMaxVelocity, kMaxVelocity);
    uzn = clampf(uzn, -kMaxVelocity, kMaxVelocity);
    phij[0] = calculate_phi(rhon, massn, flagsn);
    float rho_laplace =
        params.def_6_sigma != 0.0f
            ? params.def_6_sigma * calculate_curvature(n, phij, phi, params.Nx,
                                                       params.Ny, params.Nz)
            : 0.0f;

    float feg[kVelocitySet];
    const float rho2tmp = 0.5f / rhon;
    calculate_f_eq(
        1.0f - rho_laplace,
        clampf(fmaf(params.fx, rho2tmp, uxn), -kMaxVelocity, kMaxVelocity),
        clampf(fmaf(params.fy, rho2tmp, uyn), -kMaxVelocity, kMaxVelocity),
        clampf(fmaf(params.fz, rho2tmp, uzn), -kMaxVelocity, kMaxVelocity),
        feg);

    unsigned char flagsj_su[kVelocitySet];
    for (unsigned int i = 1u; i < kVelocitySet; i++)
      flagsj_su[i] = flags[j[i]] & CellFlag::SURFACE_MASK;
    for (unsigned int i = 1u; i < kVelocitySet; i += 2u) {
      if (flagsj_su[i] & (CellFlag::FLUID | CellFlag::INTERFACE))
        massn += (flagsj_su[i] == CellFlag::FLUID)
                     ? (fhn[i + 1] - fon[i])
                     : 0.5f * (phij[i] + phij[0]) * (fhn[i + 1] - fon[i]);
      if (flagsj_su[i + 1u] & (CellFlag::FLUID | CellFlag::INTERFACE))
        massn += (flagsj_su[i + 1u] == CellFlag::FLUID)
                     ? (fhn[i] - fon[i + 1u])
                     : 0.5f * (phij[i + 1u] + phij[0]) * (fhn[i] - fon[i + 1u]);
    }
    for (unsigned int i = 1u; i < kVelocitySet; i += 2u) {
      fhn[i] = feg[i + 1u] - fon[i + 1u] + feg[i];
      fhn[i + 1u] = feg[i] - fon[i] + feg[i + 1u];
    }
    store_f_reconstructed(n, fhn, fi, j, t, params.N, flagsj_su);
  }
  mass[n] = massn;
}

// ============================================================================
// 自由表面内核 - 标志传播
// ============================================================================
__global__ void kernel_surface_propagate(unsigned char *flags,
                                         LBMParams params) {
  const unsigned long long n =
      (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= params.N)
    return;
  const unsigned char flagsn_sus =
      flags[n] & (CellFlag::SURFACE_MASK | CellFlag::BOUNDARY_MASK);
  if (flagsn_sus == CellFlag::TO_FLUID) {
    unsigned long long j[kVelocitySet];
    neighbors(n, params.Nx, params.Ny, params.Nz, j);
    for (unsigned int i = 1u; i < kVelocitySet; i++) {
      const unsigned char flagsji = flags[j[i]],
                          flagsji_su = flagsji & (CellFlag::SURFACE_MASK |
                                                  CellFlag::BOUNDARY_MASK);
      if (flagsji_su == CellFlag::TO_GAS)
        flags[j[i]] = (flagsji & ~CellFlag::SURFACE_MASK) | CellFlag::INTERFACE;
      else if (flagsji_su == CellFlag::GAS)
        flags[j[i]] =
            (flagsji & ~CellFlag::SURFACE_MASK) | CellFlag::TO_INTERFACE;
    }
  }
}

// ============================================================================
// 自由表面内核 - 标志转换
// ============================================================================
__global__ void kernel_surface_transition(__half *fi, const float *rho,
                                          const float *u, unsigned char *flags,
                                          const unsigned long long t,
                                          LBMParams params) {
  const unsigned long long n =
      (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= params.N)
    return;
  const unsigned char flagsn_sus =
      flags[n] & (CellFlag::SURFACE_MASK | CellFlag::BOUNDARY_MASK);
  if (flagsn_sus == CellFlag::TO_INTERFACE) {
    float rhon, uxn, uyn, uzn;
    average_neighbors_non_gas(n, rho, u, flags, params.Nx, params.Ny, params.Nz,
                              &rhon, &uxn, &uyn, &uzn);
    float feq[kVelocitySet];
    calculate_f_eq(rhon, uxn, uyn, uzn, feq);
    unsigned long long j[kVelocitySet];
    neighbors(n, params.Nx, params.Ny, params.Nz, j);
    store_f(n, feq, fi, j, t, params.N);
  } else if (flagsn_sus == CellFlag::TO_GAS) {
    unsigned long long j[kVelocitySet];
    neighbors(n, params.Nx, params.Ny, params.Nz, j);
    for (unsigned int i = 1u; i < kVelocitySet; i++) {
      const unsigned char flagsji = flags[j[i]],
                          flagsji_su = flagsji & (CellFlag::SURFACE_MASK |
                                                  CellFlag::BOUNDARY_MASK);
      if (flagsji_su == CellFlag::FLUID || flagsji_su == CellFlag::TO_FLUID)
        flags[j[i]] = (flagsji & ~CellFlag::SURFACE_MASK) | CellFlag::INTERFACE;
    }
  }
}

// ============================================================================
// 自由表面内核 - 最终化
// ============================================================================
__global__ void kernel_surface_finalize(const float *rho, unsigned char *flags,
                                        float *mass, float *massex, float *phi,
                                        LBMParams params) {
  const unsigned long long n =
      (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= params.N)
    return;
  const unsigned char flagsn_sus =
      flags[n] & (CellFlag::SURFACE_MASK | CellFlag::BOUNDARY_MASK);
  if (flagsn_sus & CellFlag::BOUNDARY_MASK)
    return;

  const float rhon = rho[n];
  float massn = mass[n], massexn = 0.0f, phin = 0.0f;
  if (flagsn_sus == CellFlag::FLUID) {
    massexn = massn - rhon;
    massn = rhon;
    phin = 1.0f;
  } else if (flagsn_sus == CellFlag::INTERFACE) {
    massexn = massn > rhon ? massn - rhon : (massn < 0.0f ? massn : 0.0f);
    massn = clampf(massn, 0.0f, rhon);
    phin = calculate_phi(rhon, massn, CellFlag::INTERFACE);
  } else if (flagsn_sus == CellFlag::GAS) {
    massexn = massn;
    massn = 0.0f;
    phin = 0.0f;
  } else if (flagsn_sus == CellFlag::TO_FLUID) {
    flags[n] = (flags[n] & ~CellFlag::SURFACE_MASK) | CellFlag::FLUID;
    massexn = massn - rhon;
    massn = rhon;
    phin = 1.0f;
  } else if (flagsn_sus == CellFlag::TO_GAS) {
    flags[n] = (flags[n] & ~CellFlag::SURFACE_MASK) | CellFlag::GAS;
    massexn = massn;
    massn = 0.0f;
    phin = 0.0f;
  } else if (flagsn_sus == CellFlag::TO_INTERFACE) {
    flags[n] = (flags[n] & ~CellFlag::SURFACE_MASK) | CellFlag::INTERFACE;
    massexn = massn > rhon ? massn - rhon : (massn < 0.0f ? massn : 0.0f);
    massn = clampf(massn, 0.0f, rhon);
    phin = calculate_phi(rhon, massn, CellFlag::INTERFACE);
  }

  unsigned long long j[kVelocitySet];
  neighbors(n, params.Nx, params.Ny, params.Nz, j);
  unsigned int counter = 0u;
  for (unsigned int i = 1u; i < kVelocitySet; i++) {
    const unsigned char flagsji_su =
        flags[j[i]] & (CellFlag::SURFACE_MASK | CellFlag::BOUNDARY_MASK);
    counter += (unsigned int)(flagsji_su == CellFlag::FLUID ||
                              flagsji_su == CellFlag::INTERFACE ||
                              flagsji_su == CellFlag::TO_FLUID ||
                              flagsji_su == CellFlag::TO_INTERFACE);
  }
  massn += counter > 0u ? 0.0f : massexn;
  massexn = counter > 0u ? massexn / (float)counter : 0.0f;
  mass[n] = massn;
  massex[n] = massexn;
  phi[n] = phin;
}

} // namespace cuda
} // namespace lbm
