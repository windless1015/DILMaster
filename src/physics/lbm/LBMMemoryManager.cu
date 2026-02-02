#include "LBMMemoryManager.hpp"
#include "cuda/cuda_utils.hpp"
#include <iostream>
#include <iomanip>

namespace lbm {

LBMMemoryManager::~LBMMemoryManager() {
    deallocateAll();
}

BufferHandle LBMMemoryManager::allocate(BufferHandle::Type type,
                                        size_t elementCount,
                                        size_t elementSize,
                                        BufferHandle::Layout layout,
                                        const std::string& name) {
    // Allocate GPU memory
    void* devicePtr = nullptr;
    size_t totalBytes = elementCount * elementSize;

    CUDA_CHECK(cudaMalloc(&devicePtr, totalBytes));

    // Create buffer handle
    BufferHandle handle(type, layout, devicePtr, elementCount, elementSize, name);

    // Register in manager
    BufferInfo info;
    info.handle = handle;
    info.ownedByManager = true;
    buffers_.push_back(info);

    totalAllocatedBytes_ += totalBytes;

    return handle;
}

void LBMMemoryManager::deallocate(const BufferHandle& handle) {
    if (!handle.isValid()) {
        return;
    }

    // Find and remove from buffers list
    for (auto it = buffers_.begin(); it != buffers_.end(); ++it) {
        if (it->handle.getDevicePtr() == handle.getDevicePtr()) {
            if (it->ownedByManager) {
                CUDA_CHECK(cudaFree(handle.getDevicePtr()));
                totalAllocatedBytes_ -= handle.getTotalBytes();
            }
            buffers_.erase(it);
            return;
        }
    }
}

void LBMMemoryManager::deallocateAll() {
    for (auto& info : buffers_) {
        if (info.ownedByManager && info.handle.isValid()) {
            cudaFree(info.handle.getDevicePtr());
            totalAllocatedBytes_ -= info.handle.getTotalBytes();
        }
    }
    buffers_.clear();
    totalAllocatedBytes_ = 0;
}

void LBMMemoryManager::upload(const BufferHandle& handle, const void* hostPtr,
                               size_t size) {
    if (!handle.isValid() || !hostPtr) {
        return;
    }

    size_t copySize = std::min(size, handle.getTotalBytes());
    CUDA_CHECK(cudaMemcpy(handle.getDevicePtr(), hostPtr, copySize,
                          cudaMemcpyHostToDevice));
}

void LBMMemoryManager::download(const BufferHandle& handle, void* hostPtr,
                                 size_t size) {
    if (!handle.isValid() || !hostPtr) {
        return;
    }

    size_t copySize = std::min(size, handle.getTotalBytes());
    CUDA_CHECK(cudaMemcpy(hostPtr, handle.getDevicePtr(), copySize,
                          cudaMemcpyDeviceToHost));
}

namespace {
// CUDA kernel for SoA to AoS conversion (float3)
__global__ void kernel_soa_to_aos(const float* soa_x, const float* soa_y,
                                  const float* soa_z, float3* aos, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    aos[idx] = make_float3(soa_x[idx], soa_y[idx], soa_z[idx]);
}

// CUDA kernel for AoS to SoA conversion (float3)
__global__ void kernel_aos_to_soa(const float3* aos, float* soa_x, float* soa_y,
                                  float* soa_z, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float3 val = aos[idx];
    soa_x[idx] = val.x;
    soa_y[idx] = val.y;
    soa_z[idx] = val.z;
}
} // anonymous namespace

void LBMMemoryManager::convert(const BufferHandle& src, const BufferHandle& dst) {
    if (!src.isValid() || !dst.isValid()) {
        return;
    }

    // Check that buffers have compatible sizes
    if (src.getElementCount() != dst.getElementCount()) {
        std::cerr << "Error: Buffer size mismatch in convert()" << std::endl;
        return;
    }

    // Check that layouts are different
    if (src.getLayout() == dst.getLayout()) {
        std::cerr << "Warning: Source and destination have same layout" << std::endl;
        return;
    }

    int N = static_cast<int>(src.getElementCount());
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    if (src.isSoA() && dst.isAoS()) {
        // SoA → AoS
        const float* soa_ptr = static_cast<const float*>(src.getDevicePtr());
        float3* aos_ptr = static_cast<float3*>(dst.getDevicePtr());

        kernel_soa_to_aos<<<grid, block>>>(
            soa_ptr,           // x component
            soa_ptr + N,       // y component
            soa_ptr + 2 * N,   // z component
            aos_ptr,
            N
        );
        CUDA_CHECK(cudaGetLastError());
    } else if (src.isAoS() && dst.isSoA()) {
        // AoS → SoA
        const float3* aos_ptr = static_cast<const float3*>(src.getDevicePtr());
        float* soa_ptr = static_cast<float*>(dst.getDevicePtr());

        kernel_aos_to_soa<<<grid, block>>>(
            aos_ptr,
            soa_ptr,           // x component
            soa_ptr + N,       // y component
            soa_ptr + 2 * N,   // z component
            N
        );
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

BufferHandle LBMMemoryManager::registerExternal(BufferHandle::Type type,
                                                void* devicePtr,
                                                size_t elementCount,
                                                size_t elementSize,
                                                BufferHandle::Layout layout,
                                                const std::string& name) {
    BufferHandle handle(type, layout, devicePtr, elementCount, elementSize, name);

    BufferInfo info;
    info.handle = handle;
    info.ownedByManager = false;  // External memory, don't free
    buffers_.push_back(info);

    return handle;
}

void LBMMemoryManager::printStatistics() const {
    std::cout << "\n=== LBM Memory Manager Statistics ===" << std::endl;
    std::cout << "Total buffers: " << buffers_.size() << std::endl;
    std::cout << "Total allocated: " << (totalAllocatedBytes_ / (1024.0 * 1024.0))
              << " MB" << std::endl;

    std::cout << "\nBuffer details:" << std::endl;
    for (const auto& info : buffers_) {
        const auto& h = info.handle;
        std::cout << "  - " << std::setw(20) << h.getName()
                  << " | Type: " << static_cast<int>(h.getType())
                  << " | Layout: " << (h.isSoA() ? "SoA" : "AoS")
                  << " | Size: " << (h.getTotalBytes() / 1024.0) << " KB"
                  << " | " << (info.ownedByManager ? "Owned" : "External")
                  << std::endl;
    }
    std::cout << "====================================\n" << std::endl;
}

BufferHandle LBMMemoryManager::getBuffer(BufferHandle::Type type) const {
    for (const auto& info : buffers_) {
        if (info.handle.getType() == type) {
            return info.handle;
        }
    }
    return BufferHandle{};
}

BufferHandle LBMMemoryManager::getBuffer(BufferHandle::Type type,
                                         BufferHandle::Layout layout) const {
    for (const auto& info : buffers_) {
        if (info.handle.getType() == type &&
            info.handle.getLayout() == layout) {
            return info.handle;
        }
    }
    return BufferHandle{};
}

LBMMemoryManager::BufferInfo* LBMMemoryManager::findBuffer(void* devicePtr) {
    for (auto& info : buffers_) {
        if (info.handle.getDevicePtr() == devicePtr) {
            return &info;
        }
    }
    return nullptr;
}

} // namespace lbm
