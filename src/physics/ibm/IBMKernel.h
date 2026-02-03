#pragma once

#include "../../geometry/VectorTypes.h"
#include <array>
#include <cmath>

namespace ibm {

enum class KernelType : int {
    Trilinear = 0
};

inline const char* kernelTypeName(KernelType type) {
    switch (type) {
        case KernelType::Trilinear: return "Trilinear";
        default: return "Unknown";
    }
}

struct KernelWeight {
    int3 idx;
    float w;
};

// No changes needed for structs as they are POD-like and widely supported.
// But verifying if 'items{}' initialization works on device. 
// It should be fine in C++17 CUDA.

struct KernelWeights {
    KernelWeight items[64];
    int count = 0;
};

inline __host__ __device__ KernelWeights computeDeltaWeights(
    const float3& position,
    const float3& minBound,
    float dx,
    KernelType type
) {
    KernelWeights out;
    out.count = 0;

    if (dx <= 0.0f) {
        return out;
    }

    const float gx = (position.x - minBound.x) / dx;
    const float gy = (position.y - minBound.y) / dx;
    const float gz = (position.z - minBound.z) / dx;

    switch (type) {
        case KernelType::Trilinear: {
            const int i0 = static_cast<int>(std::floor(gx - 0.5f));
            const int j0 = static_cast<int>(std::floor(gy - 0.5f));
            const int k0 = static_cast<int>(std::floor(gz - 0.5f));

            for (int k = k0; k <= k0 + 1; ++k) {
                for (int j = j0; j <= j0 + 1; ++j) {
                    for (int i = i0; i <= i0 + 1; ++i) {
                        const float wx = 1.0f - std::abs(gx - (static_cast<float>(i) + 0.5f));
                        const float wy = 1.0f - std::abs(gy - (static_cast<float>(j) + 0.5f));
                        const float wz = 1.0f - std::abs(gz - (static_cast<float>(k) + 0.5f));
                        if (wx > 0.0f && wy > 0.0f && wz > 0.0f) {
                            KernelWeight w;
                            w.idx = make_int3(i, j, k);
                            w.w = wx * wy * wz;
                            out.items[out.count++] = w;
                        }
                    }
                }
            }
            break;
        }
        default:
            break;
    }

    return out;
}

} // namespace ibm
