#pragma once
/**
 * LBMKernelHelpers.cuh - LBM CUDA 内核辅助函数
 *
 * 合并了所有 __device__ 辅助函数：
 * - 索引计算
 * - 平衡态分布
 * - 外力项
 * - 碰撞模型
 * - 流动
 * - 自由表面
 * - 曲率计算
 */

#include "LBMTypes.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>


namespace lbm {
namespace cuda {

// ============================================================================
// 辅助数学函数
// ============================================================================
__device__ __forceinline__ float sq(const float v) { return v * v; }
__device__ __forceinline__ float cb(const float v) { return v * v * v; }
__device__ __forceinline__ float clampf(const float v, const float lo,
                                        const float hi) {
  return fminf(fmaxf(v, lo), hi);
}
__device__ __forceinline__ float fdimf_custom(const float a, const float b) {
  return fmaxf(a - b, 0.0f);
}
__device__ __forceinline__ float3 make_float3v(const float x, const float y,
                                               const float z) {
  float3 v;
  v.x = x;
  v.y = y;
  v.z = z;
  return v;
}
__device__ __forceinline__ float dot3(const float3 a, const float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __forceinline__ float3 cross3(const float3 a, const float3 b) {
  return make_float3v(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                      a.x * b.y - a.y * b.x);
}
__device__ __forceinline__ float3 normalize3(const float3 v) {
  const float len = rsqrtf(fmaxf(dot3(v, v), 1.0e-20f));
  return make_float3v(v.x * len, v.y * len, v.z * len);
}

// ============================================================================
// 索引计算
// ============================================================================
__device__ __forceinline__ unsigned long long
index_f(const unsigned long long n, const unsigned int i,
        const unsigned long long N) {
  return (unsigned long long)i * N + n;
}

__device__ __forceinline__ void coordinates(const unsigned long long n,
                                            const unsigned int Nx,
                                            const unsigned int Ny,
                                            unsigned int *x, unsigned int *y,
                                            unsigned int *z) {
  const unsigned long long t = n % (unsigned long long)(Nx * Ny);
  *x = (unsigned int)(t % (unsigned long long)Nx);
  *y = (unsigned int)(t / (unsigned long long)Nx);
  *z = (unsigned int)(n / (unsigned long long)(Nx * Ny));
}

__device__ __forceinline__ void calculate_indices(
    const unsigned long long n, const unsigned int Nx, const unsigned int Ny,
    const unsigned int Nz, unsigned long long *x0, unsigned long long *xp,
    unsigned long long *xm, unsigned long long *y0, unsigned long long *yp,
    unsigned long long *ym, unsigned long long *z0, unsigned long long *zp,
    unsigned long long *zm) {
  unsigned int x, y, z;
  coordinates(n, Nx, Ny, &x, &y, &z);
  *x0 = (unsigned long long)x;
  *xp = (unsigned long long)((x + 1u) % Nx);
  *xm = (unsigned long long)((x + Nx - 1u) % Nx);
  *y0 = (unsigned long long)(y * Nx);
  *yp = (unsigned long long)(((y + 1u) % Ny) * Nx);
  *ym = (unsigned long long)(((y + Ny - 1u) % Ny) * Nx);
  *z0 = (unsigned long long)(z * (unsigned long long)(Ny * Nx));
  *zp = (unsigned long long)(((z + 1u) % Nz) * (unsigned long long)(Ny * Nx));
  *zm = (unsigned long long)(((z + Nz - 1u) % Nz) *
                             (unsigned long long)(Ny * Nx));
}

__device__ __forceinline__ void
neighbors(const unsigned long long n, const unsigned int Nx,
          const unsigned int Ny, const unsigned int Nz, unsigned long long *j) {
  unsigned long long x0, xp, xm, y0, yp, ym, z0, zp, zm;
  calculate_indices(n, Nx, Ny, Nz, &x0, &xp, &xm, &y0, &yp, &ym, &z0, &zp, &zm);
  j[0] = n;
  j[1] = xp + y0 + z0;
  j[2] = xm + y0 + z0;
  j[3] = x0 + yp + z0;
  j[4] = x0 + ym + z0;
  j[5] = x0 + y0 + zp;
  j[6] = x0 + y0 + zm;
  j[7] = xp + yp + z0;
  j[8] = xm + ym + z0;
  j[9] = xp + y0 + zp;
  j[10] = xm + y0 + zm;
  j[11] = x0 + yp + zp;
  j[12] = x0 + ym + zm;
  j[13] = xp + ym + z0;
  j[14] = xm + yp + z0;
  j[15] = xp + y0 + zm;
  j[16] = xm + y0 + zp;
  j[17] = x0 + yp + zm;
  j[18] = x0 + ym + zp;
}

// ============================================================================
// FP16 流动
// ============================================================================
__device__ __forceinline__ float load_fpxx(const __half *fi,
                                           const unsigned long long idx) {
  return __half2float(fi[idx]) * kFP16InvScale;
}
__device__ __forceinline__ void
store_fpxx(__half *fi, const unsigned long long idx, const float v) {
  fi[idx] = __float2half_rn(v * kFP16Scale);
}

__device__ __forceinline__ void load_f(const unsigned long long n, float *fhn,
                                       const __half *fi,
                                       const unsigned long long *j,
                                       const unsigned long long t,
                                       const unsigned long long N) {
  fhn[0] = load_fpxx(fi, index_f(n, 0u, N));
  for (unsigned int i = 1u; i < kVelocitySet; i += 2u) {
    fhn[i] = load_fpxx(fi, index_f(n, t % 2ull ? i : i + 1u, N));
    fhn[i + 1] = load_fpxx(fi, index_f(j[i], t % 2ull ? i + 1u : i, N));
  }
}

__device__ __forceinline__ void store_f(const unsigned long long n,
                                        const float *fhn, __half *fi,
                                        const unsigned long long *j,
                                        const unsigned long long t,
                                        const unsigned long long N) {
  store_fpxx(fi, index_f(n, 0u, N), fhn[0]);
  for (unsigned int i = 1u; i < kVelocitySet; i += 2u) {
    store_fpxx(fi, index_f(j[i], t % 2ull ? i + 1u : i, N), fhn[i]);
    store_fpxx(fi, index_f(n, t % 2ull ? i : i + 1u, N), fhn[i + 1]);
  }
}

__device__ __forceinline__ void load_f_outgoing(const unsigned long long n,
                                                float *fon, const __half *fi,
                                                const unsigned long long *j,
                                                const unsigned long long t,
                                                const unsigned long long N) {
  for (unsigned int i = 1u; i < kVelocitySet; i += 2u) {
    fon[i] = load_fpxx(fi, index_f(j[i], t % 2ull ? i : i + 1u, N));
    fon[i + 1] = load_fpxx(fi, index_f(n, t % 2ull ? i + 1u : i, N));
  }
}

__device__ __forceinline__ void
store_f_reconstructed(const unsigned long long n, const float *fhn, __half *fi,
                      const unsigned long long *j, const unsigned long long t,
                      const unsigned long long N,
                      const unsigned char *flagsj_su) {
  for (unsigned int i = 1u; i < kVelocitySet; i += 2u) {
    if (flagsj_su[i + 1] == CellFlag::GAS)
      store_fpxx(fi, index_f(n, t % 2ull ? i : i + 1u, N), fhn[i]);
    if (flagsj_su[i] == CellFlag::GAS)
      store_fpxx(fi, index_f(j[i], t % 2ull ? i + 1u : i, N), fhn[i + 1]);
  }
}

// ============================================================================
// 平衡态分布
// ============================================================================
__device__ __forceinline__ void calculate_f_eq(const float rho, float ux,
                                               float uy, float uz, float *feq) {
  const float rhom1 = rho - 1.0f, c3 = -3.0f * (sq(ux) + sq(uy) + sq(uz));
  uz *= 3.0f;
  ux *= 3.0f;
  uy *= 3.0f;
  feq[0] = kW0 * fmaf(rho, 0.5f * c3, rhom1);
  const float u0 = ux + uy, u1 = ux + uz, u2 = uy + uz, u3 = ux - uy,
              u4 = ux - uz, u5 = uy - uz;
  const float rhos = kWs * rho, rhoe = kWe * rho, rhom1s = kWs * rhom1,
              rhom1e = kWe * rhom1;
  feq[1] = fmaf(rhos, fmaf(0.5f, fmaf(ux, ux, c3), ux), rhom1s);
  feq[2] = fmaf(rhos, fmaf(0.5f, fmaf(ux, ux, c3), -ux), rhom1s);
  feq[3] = fmaf(rhos, fmaf(0.5f, fmaf(uy, uy, c3), uy), rhom1s);
  feq[4] = fmaf(rhos, fmaf(0.5f, fmaf(uy, uy, c3), -uy), rhom1s);
  feq[5] = fmaf(rhos, fmaf(0.5f, fmaf(uz, uz, c3), uz), rhom1s);
  feq[6] = fmaf(rhos, fmaf(0.5f, fmaf(uz, uz, c3), -uz), rhom1s);
  feq[7] = fmaf(rhoe, fmaf(0.5f, fmaf(u0, u0, c3), u0), rhom1e);
  feq[8] = fmaf(rhoe, fmaf(0.5f, fmaf(u0, u0, c3), -u0), rhom1e);
  feq[9] = fmaf(rhoe, fmaf(0.5f, fmaf(u1, u1, c3), u1), rhom1e);
  feq[10] = fmaf(rhoe, fmaf(0.5f, fmaf(u1, u1, c3), -u1), rhom1e);
  feq[11] = fmaf(rhoe, fmaf(0.5f, fmaf(u2, u2, c3), u2), rhom1e);
  feq[12] = fmaf(rhoe, fmaf(0.5f, fmaf(u2, u2, c3), -u2), rhom1e);
  feq[13] = fmaf(rhoe, fmaf(0.5f, fmaf(u3, u3, c3), u3), rhom1e);
  feq[14] = fmaf(rhoe, fmaf(0.5f, fmaf(u3, u3, c3), -u3), rhom1e);
  feq[15] = fmaf(rhoe, fmaf(0.5f, fmaf(u4, u4, c3), u4), rhom1e);
  feq[16] = fmaf(rhoe, fmaf(0.5f, fmaf(u4, u4, c3), -u4), rhom1e);
  feq[17] = fmaf(rhoe, fmaf(0.5f, fmaf(u5, u5, c3), u5), rhom1e);
  feq[18] = fmaf(rhoe, fmaf(0.5f, fmaf(u5, u5, c3), -u5), rhom1e);
}

__device__ __forceinline__ void calculate_rho_u(const float *f, float *rhon,
                                                float *uxn, float *uyn,
                                                float *uzn) {
  float rho = f[0];
  for (unsigned int i = 1u; i < kVelocitySet; i++)
    rho += f[i];
  rho += 1.0f;
  const float ux =
      f[1] - f[2] + f[7] - f[8] + f[9] - f[10] + f[13] - f[14] + f[15] - f[16];
  const float uy =
      f[3] - f[4] + f[7] - f[8] + f[11] - f[12] + f[14] - f[13] + f[17] - f[18];
  const float uz = f[5] - f[6] + f[9] - f[10] + f[11] - f[12] + f[16] - f[15] +
                   f[18] - f[17];
  *rhon = rho;
  *uxn = ux / rho;
  *uyn = uy / rho;
  *uzn = uz / rho;
}

// ============================================================================
// 外力项 (Guo格式)
// ============================================================================
__device__ __forceinline__ void
calculate_forcing_terms(const float ux, const float uy, const float uz,
                        const float fx, const float fy, const float fz,
                        float *Fin) {
  const float uF = -0.33333334f * fmaf(ux, fx, fmaf(uy, fy, uz * fz));
  Fin[0] = 9.0f * kW0 * uF;
  for (unsigned int i = 1u; i < kVelocitySet; i++) {
    const float cx = (i == 1 || i == 7 || i == 9 || i == 13 || i == 15) ? 1.0f
                     : (i == 2 || i == 8 || i == 10 || i == 14 || i == 16)
                         ? -1.0f
                         : 0.0f;
    const float cy = (i == 3 || i == 7 || i == 11 || i == 14 || i == 17) ? 1.0f
                     : (i == 4 || i == 8 || i == 12 || i == 13 || i == 18)
                         ? -1.0f
                         : 0.0f;
    const float cz = (i == 5 || i == 9 || i == 11 || i == 16 || i == 18) ? 1.0f
                     : (i == 6 || i == 10 || i == 12 || i == 15 || i == 17)
                         ? -1.0f
                         : 0.0f;
    const float wi = (i <= 6u) ? kWs : kWe;
    Fin[i] = 9.0f * wi *
             fmaf(cx * fx + cy * fy + cz * fz,
                  cx * ux + cy * uy + cz * uz + 0.33333334f, uF);
  }
}

// ============================================================================
// 碰撞模型
// ============================================================================
__device__ __forceinline__ void collide_srt(float *f, const float *feq,
                                            const float *Fin, float omega) {
  const float c_tau = fmaf(omega, -0.5f, 1.0f);
  for (unsigned int i = 0u; i < kVelocitySet; i++) {
    f[i] = fmaf(1.0f - omega, f[i], fmaf(omega, feq[i], Fin[i] * c_tau));
  }
}

__device__ __forceinline__ void collide_dispatch(float *f, const float *feq,
                                                 const float *Fin, float omega,
                                                 const LBMParams &params) {
  // 目前仅 SRT 实现，TRT/MRT 可在此扩展
  collide_srt(f, feq, Fin, omega);
}

// ============================================================================
// 自由表面辅助
// ============================================================================
__device__ __forceinline__ float
calculate_phi(const float rhon, const float massn, const unsigned char flagsn) {
  return (flagsn & CellFlag::FLUID) ? 1.0f
         : (flagsn & CellFlag::INTERFACE)
             ? (rhon > 0.0f ? clampf(massn / rhon, 0.0f, 1.0f) : 0.5f)
             : 0.0f;
}

__device__ __forceinline__ void average_neighbors_non_gas(
    const unsigned long long n, const float *rho, const float *u,
    const unsigned char *flags, const unsigned int Nx, const unsigned int Ny,
    const unsigned int Nz, float *rhon, float *uxn, float *uyn, float *uzn) {
  unsigned long long j[kVelocitySet];
  neighbors(n, Nx, Ny, Nz, j);
  float rhot = 0.0f, uxt = 0.0f, uyt = 0.0f, uzt = 0.0f, counter = 0.0f;
  const unsigned long long N = (unsigned long long)Nx * Ny * Nz;
  for (unsigned int i = 1u; i < kVelocitySet; i++) {
    const unsigned char flagsji_sus =
        flags[j[i]] & (CellFlag::SURFACE_MASK | CellFlag::BOUNDARY_MASK);
    if (flagsji_sus == CellFlag::FLUID || flagsji_sus == CellFlag::INTERFACE ||
        flagsji_sus == CellFlag::TO_FLUID) {
      counter += 1.0f;
      rhot += rho[j[i]];
      uxt += u[j[i]];
      uyt += u[N + j[i]];
      uzt += u[2ull * N + j[i]];
    }
  }
  *rhon = counter > 0.0f ? rhot / counter : 1.0f;
  *uxn = counter > 0.0f ? uxt / counter : 0.0f;
  *uyn = counter > 0.0f ? uyt / counter : 0.0f;
  *uzn = counter > 0.0f ? uzt / counter : 0.0f;
}

__device__ __forceinline__ void average_neighbors_fluid(
    const unsigned long long n, const float *rho, const float *u,
    const unsigned char *flags, const unsigned int Nx, const unsigned int Ny,
    const unsigned int Nz, float *rhon, float *uxn, float *uyn, float *uzn) {
  unsigned long long j[kVelocitySet];
  neighbors(n, Nx, Ny, Nz, j);
  float rhot = 0.0f, uxt = 0.0f, uyt = 0.0f, uzt = 0.0f, counter = 0.0f;
  const unsigned long long N = (unsigned long long)Nx * Ny * Nz;
  for (unsigned int i = 1u; i < kVelocitySet; i++) {
    const unsigned char flagsji_su = flags[j[i]] & CellFlag::SURFACE_MASK;
    if (flagsji_su == CellFlag::FLUID) {
      counter += 1.0f;
      rhot += rho[j[i]];
      uxt += u[j[i]];
      uyt += u[N + j[i]];
      uzt += u[2ull * N + j[i]];
    }
  }
  *rhon = counter > 0.0f ? rhot / counter : 1.0f;
  *uxn = counter > 0.0f ? uxt / counter : 0.0f;
  *uyn = counter > 0.0f ? uyt / counter : 0.0f;
  *uzn = counter > 0.0f ? uzt / counter : 0.0f;
}

// ============================================================================
// 曲率计算
// ============================================================================
__device__ __forceinline__ void lu_solve(float *M, float *x, float *b,
                                         const int N, const int Nsol) {
  for (int i = 0; i < Nsol; i++) {
    for (int j = i + 1; j < Nsol; j++) {
      M[N * j + i] /= M[N * i + i];
      for (int k = i + 1; k < Nsol; k++)
        M[N * j + k] -= M[N * j + i] * M[N * i + k];
    }
  }
  for (int i = 0; i < Nsol; i++) {
    x[i] = b[i];
    for (int k = 0; k < i; k++)
      x[i] -= M[N * i + k] * x[k];
  }
  for (int i = Nsol - 1; i >= 0; i--) {
    for (int k = i + 1; k < Nsol; k++)
      x[i] -= M[N * i + k] * x[k];
    x[i] /= M[N * i + i];
  }
}

__device__ __forceinline__ float3 calculate_normal_py(const float *phij) {
  float3 n;
  n.x = 4.0f * (phij[2] - phij[1]) +
        2.0f * (phij[8] - phij[7] + phij[10] - phij[9] + phij[14] - phij[13] +
                phij[16] - phij[15]) +
        phij[20] - phij[19] + phij[22] - phij[21] + phij[24] - phij[23] +
        phij[25] - phij[26];
  n.y = 4.0f * (phij[4] - phij[3]) +
        2.0f * (phij[8] - phij[7] + phij[12] - phij[11] + phij[13] - phij[14] +
                phij[18] - phij[17]) +
        phij[20] - phij[19] + phij[22] - phij[21] + phij[23] - phij[24] +
        phij[26] - phij[25];
  n.z = 4.0f * (phij[6] - phij[5]) +
        2.0f * (phij[10] - phij[9] + phij[12] - phij[11] + phij[15] - phij[16] +
                phij[17] - phij[18]) +
        phij[20] - phij[19] + phij[21] - phij[22] + phij[24] - phij[23] +
        phij[26] - phij[25];
  return normalize3(n);
}

__device__ __forceinline__ float plic_cube_reduced(const float V,
                                                   const float n1,
                                                   const float n2,
                                                   const float n3) {
  const float n12 = n1 + n2, n3V = n3 * V;
  if (n12 <= 2.0f * n3V)
    return n3V + 0.5f * n12;
  const float sqn1 = sq(n1), n26 = 6.0f * n2, v1 = sqn1 / n26;
  if (v1 <= n3V && n3V < v1 + 0.5f * (n2 - n1))
    return 0.5f * (n1 + sqrtf(sqn1 + 8.0f * n2 * (n3V - v1)));
  const float V6 = n1 * n26 * n3V;
  if (n3V < v1)
    return cbrtf(V6);
  const float v3 = n3 < n12
                       ? (sq(n3) * (3.0f * n12 - n3) + sqn1 * (n1 - 3.0f * n3) +
                          sq(n2) * (n2 - 3.0f * n3)) /
                             (n1 * n26)
                       : 0.5f * n12;
  const float sqn12 = sqn1 + sq(n2), V6cbn12 = V6 - cb(n1) - cb(n2);
  const bool case34 = n3V < v3;
  const float a = case34 ? V6cbn12 : 0.5f * (V6cbn12 - cb(n3));
  const float b = case34 ? sqn12 : 0.5f * (sqn12 + sq(n3));
  const float c = case34 ? n12 : 0.5f;
  const float t = sqrtf(sq(c) - b);
  return c - 2.0f * t *
                 sinf(0.33333334f *
                      asinf((cb(c) - 0.5f * a - 1.5f * b * c) / cb(t)));
}

__device__ __forceinline__ float plic_cube(const float V0, const float3 n) {
  const float ax = fabsf(n.x), ay = fabsf(n.y), az = fabsf(n.z),
              V = 0.5f - fabsf(V0 - 0.5f), l = ax + ay + az;
  const float n1 = fminf(fminf(ax, ay), az) / l,
              n3 = fmaxf(fmaxf(ax, ay), az) / l,
              n2 = fdimf_custom(1.0f, n1 + n3);
  return l * copysignf(0.5f - plic_cube_reduced(V, n1, n2, n3), V0 - 0.5f);
}

__device__ __forceinline__ float c_D3Q27(const unsigned int i) {
  const float c[81] = {0,  1,  -1, 0,  0,  0,  0,  1,  -1, 1, -1, 0,  0,  1,
                       -1, 1,  -1, 0,  0,  1,  -1, 1,  -1, 1, -1, -1, 1,  0,
                       0,  0,  1,  -1, 0,  0,  1,  -1, 0,  0, 1,  -1, -1, 1,
                       0,  0,  1,  -1, 1,  -1, 1,  -1, -1, 1, 1,  -1, 0,  0,
                       0,  0,  0,  1,  -1, 0,  0,  1,  -1, 1, -1, 0,  0,  -1,
                       1,  -1, 1,  1,  -1, -1, 1,  1,  -1, 1, -1};
  return c[i];
}

__device__ __forceinline__ void
get_remaining_neighbor_phij(const unsigned long long n, const float *phit,
                            const float *phi, const unsigned int Nx,
                            const unsigned int Ny, const unsigned int Nz,
                            float *phij) {
  unsigned long long x0, xp, xm, y0, yp, ym, z0, zp, zm;
  calculate_indices(n, Nx, Ny, Nz, &x0, &xp, &xm, &y0, &yp, &ym, &z0, &zp, &zm);
  unsigned long long j[8];
  j[0] = xp + yp + zp;
  j[1] = xm + ym + zm;
  j[2] = xp + yp + zm;
  j[3] = xm + ym + zp;
  j[4] = xp + ym + zp;
  j[5] = xm + yp + zm;
  j[6] = xm + yp + zp;
  j[7] = xp + ym + zm;
  for (unsigned int i = 0u; i < 19u; i++)
    phij[i] = phit[i];
  for (unsigned int i = 19u; i < 27u; i++)
    phij[i] = phi[j[i - 19u]];
}

__device__ __forceinline__ float
calculate_curvature(const unsigned long long n, const float *phit,
                    const float *phi, const unsigned int Nx,
                    const unsigned int Ny, const unsigned int Nz) {
  float phij[27];
  get_remaining_neighbor_phij(n, phit, phi, Nx, Ny, Nz, phij);
  const float3 bz = calculate_normal_py(phij);
  const float3 rn = make_float3v(0.56270900f, 0.32704452f, 0.75921047f);
  const float3 by = normalize3(cross3(bz, rn));
  const float3 bx = cross3(by, bz);
  unsigned int number = 0u;
  float3 p[24];
  const float center_offset = plic_cube(phij[0], bz);
  for (unsigned int i = 1u; i < 27u; i++) {
    if (phij[i] > 0.0f && phij[i] < 1.0f) {
      const float3 ei =
          make_float3v(c_D3Q27(i), c_D3Q27(27u + i), c_D3Q27(54u + i));
      const float offset = plic_cube(phij[i], bz) - center_offset;
      p[number++] =
          make_float3v(dot3(ei, bx), dot3(ei, by), dot3(ei, bz) + offset);
    }
  }
  float M[25], x[5] = {0, 0, 0, 0, 0}, b[5] = {0, 0, 0, 0, 0};
  for (unsigned int i = 0u; i < 25u; i++)
    M[i] = 0.0f;
  for (unsigned int i = 0u; i < number; i++) {
    const float x0 = p[i].x, y0 = p[i].y, z0 = p[i].z, x2 = x0 * x0,
                y2 = y0 * y0, x3 = x2 * x0, y3 = y2 * y0;
    M[0] += x2 * x2;
    M[1] += x2 * y2;
    M[2] += x3 * y0;
    M[3] += x3;
    M[4] += x2 * y0;
    b[0] += x2 * z0;
    M[6] += y2 * y2;
    M[7] += x0 * y3;
    M[8] += x0 * y2;
    M[9] += y3;
    b[1] += y2 * z0;
    M[12] += x2 * y2;
    M[13] += x2 * y0;
    M[14] += x0 * y2;
    b[2] += x0 * y0 * z0;
    M[18] += x2;
    M[19] += x0 * y0;
    b[3] += x0 * z0;
    M[24] += y2;
    b[4] += y0 * z0;
  }
  for (unsigned int i = 1u; i < 5u; i++) {
    for (unsigned int j = 0u; j < i; j++)
      M[i * 5u + j] = M[j * 5u + i];
  }
  if (number >= 5u)
    lu_solve(M, x, b, 5, 5);
  else
    lu_solve(M, x, b, 5, number > 5u ? 5u : (int)number);
  const float A = x[0], B = x[1], C = x[2], H = x[3], I = x[4];
  const float K = (A * (I * I + 1.0f) + B * (H * H + 1.0f) - C * H * I) *
                  cb(rsqrtf(H * H + I * I + 1.0f));
  return clampf(K, -1.0f, 1.0f);
}

} // namespace cuda
} // namespace lbm
