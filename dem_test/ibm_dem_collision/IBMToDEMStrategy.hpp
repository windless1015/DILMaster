#pragma once
#include "../../src/core/StepContext.hpp"
#include "../../src/core/FieldStore.hpp"
#include "../../src/physics/ibm/IBMSolver.hpp"
#include "../../src/physics/dem/DEMSolver.hpp"
#include <cmath>
#include <vector>
#include <iostream>

#include <cuda_runtime.h>
#include <vector>

class IBMToDEMStrategy {
public:
    struct Params {
        float stiffness = 1.0e5f;
        float influence_radius = 0.002f; // Extra margin for detection
    };

    IBMToDEMStrategy(const Params& params) : params_(params) {}

    void execute(StepContext& ctx) {
        auto& fields = *ctx.fields;

        if (!fields.exists(IBMFields::MARKERS) || 
            !fields.exists(DEMFields::POSITION) || 
            !fields.exists(DEMFields::FORCE) ||
            !fields.exists(DEMFields::RADIUS)) {
            return;
        }

        auto ibm_markers = fields.get(IBMFields::MARKERS);
        auto dem_pos = fields.get(DEMFields::POSITION);
        auto dem_force = fields.get(DEMFields::FORCE);
        auto dem_radius = fields.get(DEMFields::RADIUS);

        size_t n_markers = ibm_markers.count();
        size_t n_particles = dem_pos.count();

        // Host buffers
        std::vector<float3> h_markers(n_markers);
        std::vector<float3> h_p_pos(n_particles);
        std::vector<float3> h_p_force(n_particles);
        std::vector<float> h_p_rad(n_particles);

        // Download from Device
        cudaMemcpy(h_markers.data(), ibm_markers.data(), n_markers * sizeof(float3), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_p_pos.data(), dem_pos.data(), n_particles * sizeof(float3), cudaMemcpyDeviceToHost);
        // Force is read-modify-write, so download initial (should be 0 cleared before this)
        cudaMemcpy(h_p_force.data(), dem_force.data(), n_particles * sizeof(float3), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_p_rad.data(), dem_radius.data(), n_particles * sizeof(float), cudaMemcpyDeviceToHost);

        // CPU Collision Logic
        for (size_t i = 0; i < n_particles; ++i) {
            float3 pos = h_p_pos[i];
            float r_p = h_p_rad[i];
            float contact_threshold = r_p + params_.influence_radius;
            
            float min_dist_sq = 1e30f;
            float3 closest_marker = {0,0,0};
            bool contact_found = false;

            for (size_t m = 0; m < n_markers; ++m) {
                float3 m_pos = h_markers[m];
                float dx = pos.x - m_pos.x;
                float dy = pos.y - m_pos.y;
                float dz = pos.z - m_pos.z;
                float dist_sq = dx*dx + dy*dy + dz*dz;

                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    closest_marker = m_pos;
                }
            }

            float min_dist = std::sqrt(min_dist_sq);
            if (min_dist < contact_threshold) {
                contact_found = true;
                float overlap = contact_threshold - min_dist;
                
                // Normal: Marker -> Particle
                float3 n_dir = {
                    pos.x - closest_marker.x,
                    pos.y - closest_marker.y,
                    pos.z - closest_marker.z
                };
                float len = std::sqrt(n_dir.x*n_dir.x + n_dir.y*n_dir.y + n_dir.z*n_dir.z);
                if (len > 1e-12f) {
                    n_dir.x /= len; n_dir.y /= len; n_dir.z /= len;
                } else {
                    n_dir = {0,0,1}; 
                }

                // Force
                float f_mag = params_.stiffness * overlap;
                h_p_force[i].x += f_mag * n_dir.x;
                h_p_force[i].y += f_mag * n_dir.y;
                h_p_force[i].z += f_mag * n_dir.z;
            }
        }

        // Upload Force Back
        cudaMemcpy(dem_force.data(), h_p_force.data(), n_particles * sizeof(float3), cudaMemcpyHostToDevice);
    }

private:
    Params params_;
};
