#pragma once
/**
 * @file LBMSolver.hpp
 * @brief LBM Solver with complete configuration interface
 * 
 * IMPORTANT: This solver accepts LBMConfig directly. All parameters should be
 * set in the config struct. DO NOT add new setter methods here - instead,
 * add new fields to LBMConfig.hpp if needed.
 */

#include "../ISolver.hpp"
#include "LBMConfig.hpp"
#include "ILBMModule.hpp"
#include "LBMMemoryManager.hpp"
#include <memory>
#include <vector>

// Forward declaration
namespace lbm {
class LBMCore;
}

class LBMSolver : public ISolver {
public:
  LBMSolver();
  ~LBMSolver() override;

  std::string name() const override { return "LBM"; }
  
  // =========================================================================
  // Configuration Interface - Use ONE of these methods
  // =========================================================================
  
  /**
   * @brief Set complete LBM configuration (RECOMMENDED)
   * 
   * Example usage:
   * @code
   * lbm::LBMConfig cfg;
   * cfg.nx = 100; cfg.ny = 100; cfg.nz = 50;
   * cfg.tau = 0.6f;  // or compute from viscosity
   * cfg.u0 = make_float3(0.1f, 0.0f, 0.0f);
   * cfg.enableFreeSurface = false;
   * cfg.bcXMin = lbm::BC_EQUILIBRIUM;
   * cfg.bcXMax = lbm::BC_OPEN;
   * solver->setConfig(cfg);
   * @endcode
   */
  void setConfig(const lbm::LBMConfig& config);
  
  /**
   * @brief Get mutable reference to config for modification
   * 
   * Note: Call this BEFORE allocate() to ensure changes take effect.
   */
  lbm::LBMConfig& config() { return config_; }
  const lbm::LBMConfig& config() const { return config_; }

  // =========================================================================
  // ISolver Interface
  // =========================================================================
  void configure(const ConfigNode &node) override;
  void allocate(StepContext &ctx) override;
  void initialize(StepContext &ctx) override;
  void step(StepContext &ctx) override;
  void finalize(StepContext &ctx) override;

  // =========================================================================
  // Module Support
  // =========================================================================
  void addModule(std::unique_ptr<ILBMModule> module);

  // =========================================================================
  // Accessors
  // =========================================================================
  lbm::LBMCore *getCore() { return core_.get(); }
  const lbm::LBMCore *getCore() const { return core_.get(); }

  lbm::LBMMemoryManager *getMemoryManager() { return memMgr_.get(); }
  const lbm::LBMMemoryManager *getMemoryManager() const { return memMgr_.get(); }
  
  std::size_t getNx() const { return static_cast<std::size_t>(config_.nx); }
  std::size_t getNy() const { return static_cast<std::size_t>(config_.ny); }
  std::size_t getNz() const { return static_cast<std::size_t>(config_.nz); }

  // =========================================================================
  // Utility: Compute tau from viscosity
  // =========================================================================
  static float computeTauFromViscosity(float nu) {
    return 3.0f * nu + 0.5f;
  }

private:
  lbm::LBMConfig config_;
  std::unique_ptr<lbm::LBMMemoryManager> memMgr_;
  std::unique_ptr<lbm::LBMCore> core_;
  std::vector<std::unique_ptr<ILBMModule>> modules_;
};
