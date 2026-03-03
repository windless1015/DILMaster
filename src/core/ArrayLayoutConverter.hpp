#pragma once

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <vector_types.h> // For float3, double3 etc

namespace core {

/**
 * @brief Utility class for converting between CPU-friendly AoS and GPU-friendly SoA layouts.
 * 
 * AoS (Array of Structures): [XYZ XYZ XYZ XYZ] -> typically used by VTK and Host logic.
 * SoA (Structure of Arrays): [XXXX YYYY ZZZZ]   -> typically used by optimized GPU kernels.
 */
class ArrayLayoutConverter {
public:

    /**
     * @brief Convert a raw SoA buffer from GPU into an AoS float3 vector.
     * 
     * @param soa_buffer Pointer to SoA formatted array: [X_0..X_N-1, Y_0..Y_N-1, Z_0..Z_N-1]
     * @param count Number of particles/elements (N)
     * @return std::vector<float3> AoS formatted array: [XYZ_0, XYZ_1, ..., XYZ_N-1]
     */
    static std::vector<float3> SoAToAoS_float3(const float* soa_buffer, size_t count) {
        if (!soa_buffer && count > 0) {
            throw std::runtime_error("ArrayLayoutConverter: soa_buffer is null");
        }
        
        std::vector<float3> aos_data(count);
        for (size_t i = 0; i < count; ++i) {
            aos_data[i].x = soa_buffer[i];
            aos_data[i].y = soa_buffer[i + count];
            aos_data[i].z = soa_buffer[i + 2 * count];
        }
        return aos_data;
    }

    /**
     * @brief Convert an AoS float3 vector into a raw SoA buffer for the GPU.
     * 
     * @param aos_data AoS formatted array: [XYZ_0, XYZ_1, ..., XYZ_N-1]
     * @param soa_buffer_out Pointer to pre-allocated SoA formatted array, must be at least 3*count floats.
     */
    static void AoSToSoA_float3(const std::vector<float3>& aos_data, float* soa_buffer_out) {
        if (!soa_buffer_out && !aos_data.empty()) {
            throw std::runtime_error("ArrayLayoutConverter: soa_buffer_out is null");
        }

        size_t count = aos_data.size();
        for (size_t i = 0; i < count; ++i) {
            soa_buffer_out[i]             = aos_data[i].x;
            soa_buffer_out[i + count]     = aos_data[i].y;
            soa_buffer_out[i + 2 * count] = aos_data[i].z;
        }
    }
};

} // namespace core
