#include "FreeSurfaceModule.hpp"
#include "../../core/FieldStore.hpp"
#include "cuda/cuda_utils.hpp"
#include <cuda_runtime.h>

namespace lbm {

namespace {
constexpr uint8_t TYPE_S = 0x01; // Solid
constexpr uint8_t TYPE_F = 0x08; // Fluid
constexpr uint8_t TYPE_I = 0x10; // Interface
constexpr uint8_t TYPE_G = 0x20; // Gas

__device__ __forceinline__ bool in_region(int x, int y, int z, int x0, int x1,
                                          int y0, int y1, int z0, int z1) {
  return x >= x0 && x <= x1 && y >= y0 && y <= y1 && z >= z0 && z <= z1;
}

__global__ void set_region_kernel(int nx, int ny, int nz, int x0, int x1,
                                  int y0, int y1, int z0, int z1, uint8_t type,
                                  float fill, float rho0, uint8_t *cellType,
                                  uint8_t *flags, float *phi, float *mass) {
  const int n = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nCells = nx * ny * nz;
  if (n >= nCells)
    return;
  const int z = n / (nx * ny);
  const int rem = n - z * nx * ny;
  const int y = rem / nx;
  const int x = rem - y * nx;
  if (!in_region(x, y, z, x0, x1, y0, y1, z0, z1))
    return;

  cellType[n] = type;
  if (type == (uint8_t)CellType::SOLID) {
    flags[n] = TYPE_S;
    phi[n] = 0.0f;
    mass[n] = 0.0f;
  } else if (type == (uint8_t)CellType::LIQUID) {
    flags[n] = TYPE_F;
    phi[n] = 1.0f;
    mass[n] = rho0;
  } else if (type == (uint8_t)CellType::INTERFACE) {
    const float clamped = fminf(fmaxf(fill, 0.0f), 1.0f);
    flags[n] = TYPE_I;
    phi[n] = clamped;
    mass[n] = clamped * rho0;
  } else { // GAS
    flags[n] = TYPE_G;
    phi[n] = 0.0f;
    mass[n] = 0.0f;
  }
}

__global__ void fix_interface_kernel(int nx, int ny, int nz, float rho0,
                                     uint8_t *cellType, uint8_t *flags,
                                     float *phi, float *mass) {
  const int n = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nCells = nx * ny * nz;
  if (n >= nCells)
    return;
  if (cellType[n] != (uint8_t)CellType::GAS)
    return;

  const int z = n / (nx * ny);
  const int rem = n - z * nx * ny;
  const int y = rem / nx;
  const int x = rem - y * nx;

  const int xm = (x + nx - 1) % nx;
  const int xp = (x + 1) % nx;
  const int ym = (y + ny - 1) % ny;
  const int yp = (y + 1) % ny;
  const int zm = (z + nz - 1) % nz;
  const int zp = (z + 1) % nz;

  const int n_xp = xp + y * nx + z * nx * ny;
  const int n_xm = xm + y * nx + z * nx * ny;
  const int n_yp = x + yp * nx + z * nx * ny;
  const int n_ym = x + ym * nx + z * nx * ny;
  const int n_zp = x + y * nx + zp * nx * ny;
  const int n_zm = x + y * nx + zm * nx * ny;

  if (cellType[n_xp] == (uint8_t)CellType::LIQUID ||
      cellType[n_xm] == (uint8_t)CellType::LIQUID ||
      cellType[n_yp] == (uint8_t)CellType::LIQUID ||
      cellType[n_ym] == (uint8_t)CellType::LIQUID ||
      cellType[n_zp] == (uint8_t)CellType::LIQUID ||
      cellType[n_zm] == (uint8_t)CellType::LIQUID) {
    cellType[n] = (uint8_t)CellType::INTERFACE;
    flags[n] = TYPE_I;
    phi[n] = 0.5f;
    mass[n] = 0.5f * rho0;
  }
}

__global__ void apply_walls_kernel(int nx, int ny, int nz, int wallFlags,
                                   float rho0, uint8_t *cellType,
                                   uint8_t *flags, float *phi, float *mass) {
  const int n = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nCells = nx * ny * nz;
  if (n >= nCells)
    return;

  const int z = n / (nx * ny);
  const int rem = n - z * nx * ny;
  const int y = rem / nx;
  const int x = rem - y * nx;

  const bool wall_xmin = (wallFlags & WALL_X_MIN) && x == 0;
  const bool wall_xmax = (wallFlags & WALL_X_MAX) && x == nx - 1;
  const bool wall_ymin = (wallFlags & WALL_Y_MIN) && y == 0;
  const bool wall_ymax = (wallFlags & WALL_Y_MAX) && y == ny - 1;
  const bool wall_zmin = (wallFlags & WALL_Z_MIN) && z == 0;
  const bool wall_zmax = (wallFlags & WALL_Z_MAX) && z == nz - 1;

  if (!(wall_xmin || wall_xmax || wall_ymin || wall_ymax || wall_zmin ||
        wall_zmax))
    return;

  cellType[n] = (uint8_t)CellType::SOLID;
  flags[n] = TYPE_S;
  phi[n] = 0.0f;
  mass[n] = 0.0f;
  (void)rho0;
}

} // anonymous namespace

FreeSurfaceModule::FreeSurfaceModule()
    : enabled_(false), nx_(0), ny_(0), nz_(0), nCells_(0), rho0_(1.0f),
      wallFlags_(0) {}

FreeSurfaceModule::~FreeSurfaceModule() { finalize(); }

void FreeSurfaceModule::configure(const LBMConfig &config) {
  nx_ = config.nx;
  ny_ = config.ny;
  nz_ = config.nz;
  nCells_ = nx_ * ny_ * nz_;
  rho0_ = config.rho0;
  wallFlags_ = config.wallFlags;
  enabled_ = config.enableFreeSurface;
}

void FreeSurfaceModule::allocate(FieldStore &fields) {
  if (nCells_ <= 0)
    return;

  // 分配模块自有的 GPU 缓冲区
  if (!d_cellType_) {
    CUDA_CHECK(cudaMalloc(&d_cellType_, sizeof(uint8_t) * (size_t)nCells_));
  }
  if (!d_massDiff_) {
    CUDA_CHECK(cudaMalloc(&d_massDiff_, sizeof(double) * 4));
  }

  // 注册到 FieldStore（带 device pointer）
  fields.create(FieldDesc{"fs.cellType", (size_t)nCells_, sizeof(uint8_t), d_cellType_});
  fields.create(FieldDesc{"fs.massDifference", 4, sizeof(double), d_massDiff_});
}

void FreeSurfaceModule::initialize(FieldStore &fields) {
  // 从 FieldStore 获取 fluid 字段的设备指针
  if (fields.exists("fluid.flags")) {
    d_flags_ = fields.get("fluid.flags").device_as<uint8_t>();
  }
  if (fields.exists("fluid.phi")) {
    d_phi_ = fields.get("fluid.phi").device_as<float>();
  }
  if (fields.exists("fluid.mass")) {
    d_mass_ = fields.get("fluid.mass").device_as<float>();
  }

  // 清零自有缓冲区
  if (d_cellType_) {
    CUDA_CHECK(cudaMemset(d_cellType_, 0, sizeof(uint8_t) * (size_t)nCells_));
  }
  if (d_massDiff_) {
    CUDA_CHECK(cudaMemset(d_massDiff_, 0, sizeof(double) * 4));
  }
}

void FreeSurfaceModule::preStream(StepContext &ctx) {
  if (!enabled_)
    return;
  auto *be = static_cast<CudaLBMBackend *>(ctx.backend);
  if (!be)
    return;
  // 捕获出站分布（用于质量交换），必须在 streaming 之前执行
  be->kernel_surface_capture_outgoing((unsigned long long)ctx.step);
}

void FreeSurfaceModule::postStream(StepContext &ctx) {
  if (!enabled_)
    return;
  auto *be = static_cast<CudaLBMBackend *>(ctx.backend);
  if (!be)
    return;
  // streaming 之后、update_fields 之前，按物理顺序执行自由表面内核
  be->kernel_surface_mass_exchange();
  be->kernel_surface_flag_transition((unsigned long long)ctx.step);
  be->kernel_surface_phi_recompute();
}

void FreeSurfaceModule::finalize() {
  if (d_cellType_) {
    cudaFree(d_cellType_);
    d_cellType_ = nullptr;
  }
  if (d_massDiff_) {
    cudaFree(d_massDiff_);
    d_massDiff_ = nullptr;
  }
  d_flags_ = nullptr;
  d_phi_ = nullptr;
  d_mass_ = nullptr;
}

uint8_t *FreeSurfaceModule::getCellType() const {
  return d_cellType_;
}

float *FreeSurfaceModule::getFill() const {
  return d_phi_;
}

float *FreeSurfaceModule::getMass() const {
  return d_mass_;
}

double *FreeSurfaceModule::getMassDifference() const {
  return d_massDiff_;
}

void FreeSurfaceModule::initFlatSurface(float level, float rho0) {
  const int zmax = (int)level;
  setRegion(0, nx_ - 1, 0, ny_ - 1, 0, zmax, CellType::LIQUID, 1.0f, rho0);
}

void FreeSurfaceModule::setRegion(int x0, int x1, int y0, int y1, int z0,
                                  int z1, CellType type, float fill,
                                  float rho0) {
  if (d_cellType_ == nullptr || d_phi_ == nullptr || d_mass_ == nullptr ||
      d_flags_ == nullptr)
    return;

  const dim3 block(256);
  const dim3 grid((unsigned int)((nCells_ + block.x - 1) / block.x));

  set_region_kernel<<<grid, block>>>(nx_, ny_, nz_, x0, x1, y0, y1, z0, z1,
                                     (uint8_t)type, fill, rho0, d_cellType_,
                                     d_flags_, d_phi_, d_mass_);
  CUDA_CHECK(cudaGetLastError());
}

void FreeSurfaceModule::fixInterfaceLayer() {
  if (d_cellType_ == nullptr || d_phi_ == nullptr || d_mass_ == nullptr ||
      d_flags_ == nullptr)
    return;

  const dim3 block(256);
  const dim3 grid((unsigned int)((nCells_ + block.x - 1) / block.x));

  fix_interface_kernel<<<grid, block>>>(nx_, ny_, nz_, rho0_, d_cellType_,
                                        d_flags_, d_phi_, d_mass_);
  CUDA_CHECK(cudaGetLastError());

  apply_walls_kernel<<<grid, block>>>(nx_, ny_, nz_, wallFlags_, rho0_,
                                      d_cellType_, d_flags_, d_phi_, d_mass_);
  CUDA_CHECK(cudaGetLastError());
}

// Stub implementations for future physics steps
void FreeSurfaceModule::updateMass(const float *f_soa) { (void)f_soa; }
void FreeSurfaceModule::completeInterface(float *f_soa, const float3 *u_field) {
  (void)f_soa;
  (void)u_field;
}
void FreeSurfaceModule::updateMacroscopic() {}
void FreeSurfaceModule::distributeMass() {}
void FreeSurfaceModule::interfaceTransitions() {}
void FreeSurfaceModule::removeFalseInterfaceCells() {}
void FreeSurfaceModule::redistributeLostMass() {}
void FreeSurfaceModule::cleanupInterface() {}
void FreeSurfaceModule::reclassifyCells() {}
void FreeSurfaceModule::reconstructInterface(float *f, const float3 *u_field) {
  (void)f;
  (void)u_field;
}

bool FreeSurfaceModule::checkHealth() {
  if (!enabled_)
    return true;
  return d_cellType_ != nullptr && d_massDiff_ != nullptr &&
         d_flags_ != nullptr && d_phi_ != nullptr &&
         d_mass_ != nullptr;
}

} // namespace lbm
