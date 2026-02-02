#include "../LBMConstants.hpp" // 获取 constants::c_dir 等静态常量
#include "LBMDeviceConstants.cuh"
#include "cuda_utils.hpp"


namespace lbm {
namespace dev_const {

__constant__ int c_dir[19][3];
__constant__ int c_opposite[19];
__constant__ real c_w[19];
__constant__ real3 c_gravity;

void loadConstants(const real3 &gravity) {
  CUDA_CHECK(
      cudaMemcpyToSymbol(c_dir, constants::c_dir, sizeof(constants::c_dir)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_opposite, constants::c_opposite,
                                sizeof(constants::c_opposite)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_w, constants::c_w, sizeof(constants::c_w)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_gravity, &gravity, sizeof(real3)));
}

void updateGravity(const real3 &gravity) {
  CUDA_CHECK(cudaMemcpyToSymbol(c_gravity, &gravity, sizeof(real3)));
}

} // namespace dev_const
} // namespace lbm
