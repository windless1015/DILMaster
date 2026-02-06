#include "LBMCore.hpp"
#include "cuda/LBMDeviceConstants.cuh"
#include "cuda/cuda_utils.hpp"
#include <cuda_runtime.h>
#include <vector>

namespace lbm {

namespace {
__global__ void pack_velocity_kernel(const float *u_in, float3 *u_out,
                                     int nCells) {
  const int n = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (n >= nCells)
    return;
  const int N = nCells;
  u_out[n] = make_float3(u_in[n], u_in[N + n], u_in[2 * N + n]);
}
} // namespace

LBMCore::LBMCore(const LBMConfig &config)
    : config_(config), nCells_(config.nx * config.ny * config.nz),
      stepCount_(0), initialized_(false), u_aos_(nullptr) {}

LBMCore::~LBMCore() { freeMemory(); }

void LBMCore::allocateMemory() {
  if (initialized_)
    return;

  // 计算 tau 和 nu
  const float tau = config_.tau > 0.0f ? config_.tau : (1.0f / 3.0f + 0.5f);
  const float nu = (tau - 0.5f) / 3.0f;

  // 设置 CUDA 后端参数 (使用新的 LBMParams 结构)
  cuda::LBMParams p{};
  p.Nx = (unsigned int)config_.nx;
  p.Ny = (unsigned int)config_.ny;
  p.Nz = (unsigned int)config_.nz;
  p.N = (unsigned long long)nCells_;
  p.nu = nu;
  p.fx = config_.gravity.x;
  p.fy = config_.gravity.y;
  p.fz = config_.gravity.z;
  printf("[LBMCore::allocateMemory] Gravity=(%f, %f, %f), tau=%f, nu=%f\n", p.fx, p.fy, p.fz, tau, nu);
  p.sigma = config_.sigma;
  p.w = 1.0f / (3.0f * p.nu + 0.5f);
  p.def_6_sigma = 6.0f * config_.sigma;

  // 从 LBMConfig 同步碰撞模型和自由表面设置
  p.collisionModel = static_cast<cuda::CollisionModel>(
      static_cast<int>(config_.collisionModel));
  p.enableFreeSurface = config_.enableFreeSurface;
  p.pressure_outlet_rho = config_.pressure_outlet_rho;
  p.boundaryTypeXMin = (uint8_t)config_.bcXMin;
  p.boundaryTypeXMax = (uint8_t)config_.bcXMax;
  p.boundaryTypeYMin = (uint8_t)config_.bcYMin;
  p.boundaryTypeYMax = (uint8_t)config_.bcYMax;
  p.boundaryTypeZMin = (uint8_t)config_.bcZMin;
  p.boundaryTypeZMax = (uint8_t)config_.bcZMax;

  // 初始化 CUDA 后端
  backend_.initialize(p);

  // 分配 AoS 速度缓冲区
  if (u_aos_ == nullptr) {
    CUDA_CHECK(cudaMalloc(&u_aos_, sizeof(float3) * (size_t)nCells_));
  }

  // 加载设备常量
  dev_const::loadConstants(config_.gravity);
}

void LBMCore::freeMemory() {
  if (u_aos_) {
    cudaFree(u_aos_);
    u_aos_ = nullptr;
  }
  initialized_ = false;
}

void LBMCore::initialize() {
  if (initialized_)
    return;
  allocateMemory();

  const size_t n = static_cast<size_t>(nCells_);

  // 准备主机端初始数据
  std::vector<float> rho_host(n, config_.rho0);
  std::vector<float> u_host(n * 3u, 0.0f);
  for (size_t i = 0u; i < n; i++) {
    u_host[i] = config_.u0.x;
    u_host[n + i] = config_.u0.y;
    u_host[2u * n + i] = config_.u0.z;
  }

  // Cell flags: FLUID for single-phase, GAS for free-surface initial state
  uint8_t default_flag = config_.enableFreeSurface ? cuda::CellFlag::GAS : cuda::CellFlag::FLUID;
  std::vector<uint8_t> flags_host(n, default_flag);
  std::vector<float> phi_host(n, config_.enableFreeSurface ? 0.0f : 1.0f);

  // 上传到设备
  backend_.upload_host_fields(rho_host.data(), u_host.data(), flags_host.data(),
                              phi_host.data());

  // 初始化分布函数
  backend_.kernel_initialize();
  backend_.kernel_setup_boundaries(); // 初始化边界标志
  backend_.synchronize();

  // 打包速度到 AoS 格式
  packVelocityAoS();

  initialized_ = true;
}

void LBMCore::refreshDistributions() {
  if (!backend_.is_initialized())
    return;
  backend_.kernel_initialize();
  packVelocityAoS();
  backend_.synchronize();
}

void LBMCore::packVelocityAoS() {
  if (u_aos_ == nullptr)
    return;
  const dim3 block(256);
  const dim3 grid((unsigned int)((nCells_ + block.x - 1) / block.x));
  const float *u_soa = backend_.u_device();
  pack_velocity_kernel<<<grid, block>>>(u_soa, u_aos_, nCells_);
  CUDA_CHECK(cudaGetLastError());
}

void LBMCore::step() {
  if (!initialized_)
    initialize();

  // 基础流动计算（不含自由表面逻辑）
  streamCollide();
  updateMacroscopic();
}

void LBMCore::streamCollide() {
  backend_.kernel_stream_collide((unsigned long long)stepCount_);
  backend_.kernel_apply_boundaries((unsigned long long)stepCount_ + 1); // Set for next step
}

void LBMCore::updateMacroscopic() {
  backend_.kernel_update_fields((unsigned long long)stepCount_);
  packVelocityAoS();
  backend_.synchronize();
  stepCount_++;
}

void LBMCore::setExternalForce(float3 force) {
  config_.gravity = force;
  dev_const::updateGravity(force);
  backend_.set_force(force.x, force.y, force.z);
}

void LBMCore::uploadExternalForce(const float *force_host) {
  if (!force_host) {
      backend_.upload_force(nullptr);
      return;
  }

  // AoS (Host) -> SoA (Host Temp)
  const size_t n = static_cast<size_t>(nCells_);
  std::vector<float> force_soa(n * 3);

  #pragma omp parallel for
  for(long long i=0; i<(long long)n; ++i) {
      force_soa[i] = force_host[i*3 + 0];
      force_soa[n + i] = force_host[i*3 + 1];
      force_soa[2*n + i] = force_host[i*3 + 2];
  }

  backend_.upload_force(force_soa.data());
}

void LBMCore::setExternalForceFromDeviceAoS(const float3 *force_device_aos) {
    if (!force_device_aos) {
         backend_.upload_force(nullptr);
         return;
    }
    backend_.upload_force_from_device_aos(force_device_aos);
}

void LBMCore::setCollisionModel(CollisionModel model) {
  config_.collisionModel = model;
  backend_.set_collision_model(
      static_cast<cuda::CollisionModel>(static_cast<int>(model)));
}

void LBMCore::setFreeSurfaceEnabled(bool enabled) {
  config_.enableFreeSurface = enabled;
  backend_.set_free_surface_enabled(enabled);
}

const float *LBMCore::getDensityField() const { return backend_.rho_device(); }

const float3 *LBMCore::getVelocityField() const { return u_aos_; }

uint8_t *LBMCore::getFlagsDevice() const { return backend_.flags_device(); }

float *LBMCore::getPhiDevice() const { return backend_.phi_device(); }

float *LBMCore::getMassDevice() const { return backend_.mass_device(); }

bool LBMCore::checkHealth() const {
  return initialized_ && backend_.is_initialized();
}

void LBMCore::synchronize() const { backend_.synchronize(); }

} // namespace lbm
