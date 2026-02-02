#pragma once
#include <cstddef>

// 前向声明
class FieldStore;

/**
 * @brief 求解器步进上下文
 *
 * 在每个时间步中传递给各个求解器和服务的共享状态
 */
struct StepContext {
  std::size_t step = 0;         // 当前步数
  double time = 0.0;            // 当前模拟时间
  double dt = 0.001;            // 当前时间步长
  FieldStore *fields = nullptr; // 字段存储指针

  // 便捷构造函数
  StepContext() = default;
  StepContext(std::size_t s, double t, double d, FieldStore *f = nullptr)
      : step(s), time(t), dt(d), fields(f) {}
};
