#pragma once

#include "../../core/StepContext.hpp"
#include "LBMConfig.hpp"

// 前向声明
class FieldStore;

// LBM 求解器的可插拔模块接口
class ILBMModule {
public:
  virtual ~ILBMModule() = default;

  // 生命周期：配置 + 分配 + 初始化
  virtual void configure(const lbm::LBMConfig &config) = 0;
  virtual void allocate(FieldStore &fields) = 0;
  virtual void initialize(FieldStore &fields) = 0;

  // LBM streaming/collision 前后钩子
  virtual void preStream(StepContext &ctx) = 0;
  virtual void postStream(StepContext &ctx) = 0;

  // 资源释放
  virtual void finalize() = 0;
};
