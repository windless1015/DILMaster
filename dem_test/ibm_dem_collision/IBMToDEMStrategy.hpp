#pragma once
#include "../../src/core/StepContext.hpp"
#include "../../src/core/FieldStore.hpp"
#include "../../src/physics/ibm/IBMSolver.hpp"
#include "../../src/physics/dem/DEMSolver.hpp"
#include <cmath>
#include <vector>
#include <iostream>
#include <cstring>
#include <limits>

#include <cuda_runtime.h>
#include <vector>

class IBMToDEMStrategy {
public:
    struct Params {
        float stiffness = 1.0e5f;
        float damping = 5.0f;
        float influence_radius = 0.002f; // Extra margin for detection
    };

    IBMToDEMStrategy(const Params& params) : params_(params) {}

    void execute(StepContext& ctx) {
        auto& fields = *ctx.fields;

        if (!fields.exists(IBMFields::MARKERS) || 
            !fields.exists(IBMFields::VELOCITY) ||
            !fields.exists(DEMFields::POSITION) || 
            !fields.exists(DEMFields::VELOCITY) ||
            !fields.exists(DEMFields::FORCE) ||
            !fields.exists(DEMFields::RADIUS)) {
            return;
        }

        auto ibm_markers = fields.get(IBMFields::MARKERS);
        auto ibm_vel = fields.get(IBMFields::VELOCITY);
        auto dem_pos = fields.get(DEMFields::POSITION);
        auto dem_vel = fields.get(DEMFields::VELOCITY);
        auto dem_force = fields.get(DEMFields::FORCE);
        auto dem_radius = fields.get(DEMFields::RADIUS);

        size_t n_markers = ibm_markers.count();
        size_t n_particles = dem_pos.count();

        // Host buffers
        std::vector<float3> h_markers(n_markers);
        std::vector<float3> h_marker_vel(n_markers);
        std::vector<float3> h_p_pos(n_particles);
        std::vector<float3> h_p_vel(n_particles);
        std::vector<float3> h_p_force(n_particles);
        std::vector<float> h_p_rad(n_particles);

        auto copyToHost = [](const FieldHandle& src, void* dst, std::size_t bytes) {
            if (bytes == 0) return;
            if (src.has_device() && src.device_data()) {
                cudaMemcpy(dst, src.device_data(), bytes, cudaMemcpyDeviceToHost);
            } else {
                std::memcpy(dst, src.data(), bytes);
            }
        };
        auto copyFromHost = [](FieldHandle& dst, const void* src, std::size_t bytes) {
            if (bytes == 0) return;
            if (dst.has_device() && dst.device_data()) {
                cudaMemcpy(dst.device_data(), src, bytes, cudaMemcpyHostToDevice);
            } else {
                std::memcpy(dst.data(), src, bytes);
            }
        };

        // FieldStore is host-resident by default; handle both host/device-backed fields safely.
        copyToHost(ibm_markers, h_markers.data(), n_markers * sizeof(float3));
        copyToHost(ibm_vel, h_marker_vel.data(), n_markers * sizeof(float3));
        copyToHost(dem_pos, h_p_pos.data(), n_particles * sizeof(float3));
        copyToHost(dem_vel, h_p_vel.data(), n_particles * sizeof(float3));
        copyToHost(dem_force, h_p_force.data(), n_particles * sizeof(float3));
        copyToHost(dem_radius, h_p_rad.data(), n_particles * sizeof(float));

        // CPU Collision Logic
        for (size_t i = 0; i < n_particles; ++i) {
            float3 pos = h_p_pos[i];
            float r_p = h_p_rad[i];
            float contact_threshold = r_p + params_.influence_radius;
            
            float min_dist_sq = 1e30f;
            float3 closest_marker = {0,0,0};
            std::size_t closest_idx = 0;
            for (size_t m = 0; m < n_markers; ++m) {
                float3 m_pos = h_markers[m];
                float dx = pos.x - m_pos.x;
                float dy = pos.y - m_pos.y;
                float dz = pos.z - m_pos.z;
                float dist_sq = dx*dx + dy*dy + dz*dz;

                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    closest_marker = m_pos;
                    closest_idx = m;
                }
            }

            float min_dist = std::sqrt(min_dist_sq);
            if (min_dist < contact_threshold) {
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

                // Kelvin-Voigt normal contact:
                // F_n = k*delta - c*v_n, clamped to repulsive-only.
                float3 vp = h_p_vel[i];
                float3 vm = h_marker_vel.empty() ? make_float3(0, 0, 0)
                                                 : h_marker_vel[closest_idx];
                float vnx = vp.x - vm.x;
                float vny = vp.y - vm.y;
                float vnz = vp.z - vm.z;
                float v_n = vnx * n_dir.x + vny * n_dir.y + vnz * n_dir.z;
                float f_mag = params_.stiffness * overlap - params_.damping * v_n;
                if (f_mag < 0.0f) f_mag = 0.0f;
                h_p_force[i].x += f_mag * n_dir.x;
                h_p_force[i].y += f_mag * n_dir.y;
                h_p_force[i].z += f_mag * n_dir.z;
            }
        }

        // Upload Force Back
        copyFromHost(dem_force, h_p_force.data(), n_particles * sizeof(float3));
    }

private:
    Params params_;
};
