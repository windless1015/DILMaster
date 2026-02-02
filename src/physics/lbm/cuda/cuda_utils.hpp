#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

inline void cuda_check(const cudaError_t err, const char* file, const int line) {
	if(err != cudaSuccess) {
		const std::string msg = std::string("CUDA error: ")+cudaGetErrorString(err)+" at "+file+":"+std::to_string(line);
		throw std::runtime_error(msg);
	}
}

#define CUDA_CHECK(x) cuda_check((x), __FILE__, __LINE__)
