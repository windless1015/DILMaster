#pragma once

#include "../core/StepContext.hpp"

/**
 * @brief 服务接口：用于非核心功能（VTK 输出等）
 */
class IService {
public:
  virtual ~IService() = default;

  virtual void initialize(const StepContext &ctx) = 0;
  virtual void onStepBegin(StepContext &ctx) = 0;
  virtual void onStepEnd(StepContext &ctx) = 0;
  virtual void finalize(const StepContext &ctx) = 0;
};
