#pragma once
/**
 * LBMMemoryManager.hpp - Unified GPU memory management for LBM
 *
 * Centralized management of all GPU buffers used in LBM simulation.
 * Provides allocation, deallocation, and format conversion services.
 */

#include "BufferHandle.hpp"
#include <cuda_runtime.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace lbm {

/**
 * LBMMemoryManager - Manages all GPU memory for LBM simulation
 *
 * Features:
 * - Centralized allocation/deallocation
 * - Automatic SoA ↔ AoS conversion
 * - Type-safe buffer access
 * - Memory tracking and debugging
 */
class LBMMemoryManager {
public:
    LBMMemoryManager() = default;
    ~LBMMemoryManager();

    // Disable copy, enable move
    LBMMemoryManager(const LBMMemoryManager&) = delete;
    LBMMemoryManager& operator=(const LBMMemoryManager&) = delete;
    LBMMemoryManager(LBMMemoryManager&&) noexcept = default;
    LBMMemoryManager& operator=(LBMMemoryManager&&) noexcept = default;

    /**
     * Allocate a new GPU buffer
     *
     * @param type Buffer type (DENSITY, VELOCITY, etc.)
     * @param elementCount Number of elements
     * @param elementSize Size of each element in bytes
     * @param layout Data layout (SoA or AoS for vector fields)
     * @param name Optional name for debugging
     * @return BufferHandle to the allocated buffer
     */
    BufferHandle allocate(BufferHandle::Type type, size_t elementCount,
                          size_t elementSize,
                          BufferHandle::Layout layout = BufferHandle::Layout::SoA,
                          const std::string& name = "");

    /**
     * Deallocate a specific buffer
     */
    void deallocate(const BufferHandle& handle);

    /**
     * Deallocate all managed buffers
     */
    void deallocateAll();

    /**
     * Upload data from host to device
     *
     * @param handle Target buffer handle
     * @param hostPtr Source host pointer
     * @param size Size in bytes to copy
     */
    void upload(const BufferHandle& handle, const void* hostPtr, size_t size);

    /**
     * Download data from device to host
     *
     * @param handle Source buffer handle
     * @param hostPtr Target host pointer
     * @param size Size in bytes to copy
     */
    void download(const BufferHandle& handle, void* hostPtr, size_t size);

    /**
     * Convert between SoA and AoS layouts (for vector fields)
     *
     * @param src Source buffer
     * @param dst Destination buffer (must have opposite layout)
     */
    void convert(const BufferHandle& src, const BufferHandle& dst);

    /**
     * Register an externally allocated buffer
     * (For backward compatibility with existing code)
     *
     * @param type Buffer type
     * @param devicePtr External device pointer
     * @param elementCount Number of elements
     * @param elementSize Size per element
     * @param layout Data layout
     * @param name Optional name
     * @return BufferHandle wrapping the external pointer
     */
    BufferHandle registerExternal(BufferHandle::Type type, void* devicePtr,
                                   size_t elementCount, size_t elementSize,
                                   BufferHandle::Layout layout = BufferHandle::Layout::SoA,
                                   const std::string& name = "");

    /**
     * Get total allocated memory in bytes
     */
    size_t getTotalAllocatedBytes() const { return totalAllocatedBytes_; }

    /**
     * Get number of managed buffers
     */
    size_t getBufferCount() const { return buffers_.size(); }

    /**
     * Print memory usage statistics
     */
    void printStatistics() const;

    /**
     * Find a buffer by type (returns invalid handle if not found)
     */
    BufferHandle getBuffer(BufferHandle::Type type) const;

    /**
     * Find a buffer by type and layout (returns invalid handle if not found)
     */
    BufferHandle getBuffer(BufferHandle::Type type,
                           BufferHandle::Layout layout) const;

private:
    struct BufferInfo {
        BufferHandle handle;
        bool ownedByManager;  // true if allocated by manager, false if external
    };

    std::vector<BufferInfo> buffers_;
    size_t totalAllocatedBytes_ = 0;

    // Helper: find buffer info by device pointer
    BufferInfo* findBuffer(void* devicePtr);
};

} // namespace lbm
