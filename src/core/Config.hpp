#pragma once
#include <string>
#include <stdexcept>

/**
 * @brief Minimal ConfigNode stub for DILMaster standalone build.
 *
 * ISolver::configure() 在 LBMSolver 中为空操作 (no-op)，
 * 所以这里只需提供类型定义即可编译。
 */
struct ConfigNode {
  // Stub: LBMSolver::configure() does nothing with this
};

class Config {
public:
  Config() = default;
  ~Config() = default;

  void load(const std::string & /*path*/) {
    throw std::runtime_error(
        "Config::load() not available in DILMaster standalone build. "
        "Use LBMConfig directly.");
  }
};
