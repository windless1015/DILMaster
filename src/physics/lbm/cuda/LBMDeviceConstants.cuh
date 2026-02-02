#pragma once
#include "../LBMConfig.hpp"
#include <cuda_runtime.h>

namespace lbm {
namespace dev_const {

// 注意：CUDA __constant__ 变量需要在头文件中用 extern 声明，
// 在唯一的 .cu 文件中定义（不用 extern）
// 这里只声明函数接口

void loadConstants(const real3 &gravity);
void updateGravity(const real3 &gravity);

} // namespace dev_const
} // namespace lbm
