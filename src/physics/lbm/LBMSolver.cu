/**
 * @file LBMSolver.cu
 * @brief LBM Solver implementation using LBMConfig
 * 
 * This implementation uses the config struct directly without any assumptions.
 * All behavior is determined by the config settings.
 */

#include "LBMConfig.hpp"
#include "LBMCore.hpp"
#include "LBMSolver.hpp"
#include "../../core/FieldStore.hpp"

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
  // Create memory manager and LBM core with the current config
  memMgr_ = std::make_unique<lbm::LBMMemoryManager>();
  core_ = std::make_unique<lbm::LBMCore>(config_, memMgr_.get());

  // Module lifecycle: configure + allocate
  for (auto &module : modules_) {
    module->configure(config_);
    module->allocate(*memMgr_);
  }
  (void)ctx;
}

void LBMSolver::initialize(StepContext &ctx) {
  if (core_) {
    core_->initialize();
  }
  for (auto &module : modules_) {
    module->initialize();
  }
  (void)ctx;
}

void LBMSolver::step(StepContext &ctx) {
  if (!core_)
    return;

  // Call all modules' preStream
  for (auto &module : modules_) {
    module->preStream(ctx);
  }

  // Execute LBM step
  // Execute LBM step
  
  // [COUPLING FIX] Apply external forces from FieldStore if available
  if (ctx.fields && ctx.fields->exists("fluid.force")) {
      auto forceHandle = ctx.fields->get("fluid.force");
      // Check for Device pointer support (future) or Host pointer (legacy)
      // Currently FieldStore only supports Host std::vector
      const float* forceHost = forceHandle.as<float>();
      
      // Upload to GPU
      // TODO: optimize this to avoid H2D transfer if possible (requires FieldStore GPU support)
      if (core_) core_->uploadExternalForce(forceHost);
  }

  core_->step();

  // Call all modules' postStream
  for (auto &module : modules_) {
    module->postStream(ctx);
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
  memMgr_.reset();
  (void)ctx;
}

void LBMSolver::addModule(std::unique_ptr<ILBMModule> module) {
  modules_.push_back(std::move(module));
}
