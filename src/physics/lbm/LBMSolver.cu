/**
 * @file LBMSolver.cu
 * @brief LBM Solver implementation using LBMConfig
 *
 * This implementation uses the config struct directly without any assumptions.
 * All behavior is determined by the config settings.
 * All data exchange goes through FieldStore.
 */

#include "LBMConfig.hpp"
#include "LBMCore.hpp"
#include "LBMSolver.hpp"
#include "../../core/FieldStore.hpp"
#include <cuda_runtime.h>

LBMSolver::LBMSolver() = default;
LBMSolver::~LBMSolver() = default;

void LBMSolver::setConfig(const lbm::LBMConfig& config) {
  config_ = config;
}

void LBMSolver::configure(const ConfigNode & /*node*/) {
  // Note: physics library doesn't link yaml-cpp
  // Configuration should be done via setConfig() or config()
}

void LBMSolver::allocate(StepContext &ctx) {
  // 创建 LBMCore（不再需要 LBMMemoryManager）
  core_ = std::make_unique<lbm::LBMCore>(config_);
  core_->initialize();

  const size_t n = static_cast<size_t>(config_.nx) * config_.ny * config_.nz;

  // 在 FieldStore 中注册 fluid 字段（带 device pointer）
  // 注意：velocity 使用 AoS (float3) 格式，方便外部耦合模块使用
  if (ctx.fields) {
    ctx.fields->create(FieldDesc{"fluid.density",  n,     sizeof(float),
                                  core_->densityDevicePtr()});
    // velocity: 使用 AoS 格式 (float3*)，元素数量为 n，每个元素 sizeof(float3)
    ctx.fields->create(FieldDesc{"fluid.velocity", n, sizeof(float3),
                                  core_->velocityAoSPtr()});
    ctx.fields->create(FieldDesc{"fluid.flags",    n,     sizeof(uint8_t),
                                  core_->flagsDevicePtr()});
    ctx.fields->create(FieldDesc{"fluid.phi",      n,     sizeof(float),
                                  core_->phiDevicePtr()});
    ctx.fields->create(FieldDesc{"fluid.mass",     n,     sizeof(float),
                                  core_->massDevicePtr()});
    ctx.fields->create(FieldDesc{"fluid.massex",   n,     sizeof(float),
                                  core_->massExDevicePtr()});

    printf("[LBMSolver::allocate] core=%p u_aos=%p\n", core_.get(), core_->velocityAoSPtr());
  }

  // Module lifecycle: configure + allocate
  for (auto &module : modules_) {
    module->configure(config_);
    if (ctx.fields) {
      module->allocate(*ctx.fields);
    }
  }
}

void LBMSolver::initialize(StepContext &ctx) {
  // core_->initialize() 已在 allocate() 中调用
  for (auto &module : modules_) {
    if (ctx.fields) {
      module->initialize(*ctx.fields);
    }
  }
}

void LBMSolver::step(StepContext &ctx) {
  if (!core_)
    return;

  // 将后端指针设置到 StepContext，供模块使用
  ctx.backend = &core_->backend();

  // [COUPLING FIX] Apply external forces from FieldStore if available
  if (ctx.fields && ctx.fields->exists("fluid.force")) {
      auto forceHandle = ctx.fields->get("fluid.force");
      const float* forceHost = forceHandle.as<float>();
      if (core_) core_->uploadExternalForce(forceHost);
  }

  // 分阶段步进：preStream → streamCollide → postStream → updateMacroscopic
  for (auto &module : modules_) {
    module->preStream(ctx);
  }

  core_->streamCollide();

  if (ctx.step % 100 == 0) {
      printf("[LBMSolver::step] core=%p u_aos=%p\n", core_.get(), core_->velocityAoSPtr());
  }

  for (auto &module : modules_) {
    module->postStream(ctx);
  }

  core_->updateMacroscopic();

  // 同步 GPU 数据到 FieldStore 的主机缓冲区
  if (ctx.fields) {
    syncFieldsToHost(ctx);
  }

  ctx.backend = nullptr;
}

void LBMSolver::syncFieldsToHost(StepContext &ctx) {
  if (!core_ || !ctx.fields)
    return;

  const size_t n = static_cast<size_t>(core_->nCells());

  if (ctx.fields->exists("fluid.density")) {
    auto h = ctx.fields->get("fluid.density");
    cudaMemcpy(h.data(), core_->densityDevicePtr(),
               n * sizeof(float), cudaMemcpyDeviceToHost);
  }
  if (ctx.fields->exists("fluid.velocity")) {
    auto h = ctx.fields->get("fluid.velocity");
    // velocity 使用 AoS (float3) 格式
    cudaMemcpy(h.data(), core_->velocityAoSPtr(),
               n * sizeof(float3), cudaMemcpyDeviceToHost);
  }
  if (ctx.fields->exists("fluid.flags")) {
    auto h = ctx.fields->get("fluid.flags");
    cudaMemcpy(h.data(), core_->flagsDevicePtr(),
               n * sizeof(uint8_t), cudaMemcpyDeviceToHost);
  }
  if (ctx.fields->exists("fluid.phi")) {
    auto h = ctx.fields->get("fluid.phi");
    cudaMemcpy(h.data(), core_->phiDevicePtr(),
               n * sizeof(float), cudaMemcpyDeviceToHost);
  }
}

void LBMSolver::finalize(StepContext &ctx) {
  for (auto &module : modules_) {
    module->finalize();
  }
  if (core_) {
    core_->synchronize();
  }
  core_.reset();
  (void)ctx;
}

void LBMSolver::addModule(std::unique_ptr<ILBMModule> module) {
  modules_.push_back(std::move(module));
}
