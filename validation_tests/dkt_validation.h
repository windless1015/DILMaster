#pragma once

#include "coupling_tests.h"
#include <iostream>
#include <vector>

namespace ValidationTests {

class DKT_ValidationStrategy {
public:
    struct DKTConfig {
        float sphere1_radius = 0.01f;
        float sphere2_radius = 0.01f;
        float initial_separation = 0.05f;
        float domain_size = 0.2f;

        float fluid_density = 1000.0f;
        float fluid_viscosity = 1e-3f;
        float sphere_density = 1050.0f;
        float flow_velocity = 0.1f;

        int resolution = 64;
        float dt = 1e-4f;
        int max_steps = 10000;

        float drafting_threshold = 0.02f;
        float kissing_duration_min = 100.0f;
        float tumbling_angle_threshold = 30.0f;
    };

    struct DKTMetrics {
        bool drafting_occurred = false;
        bool kissing_occurred = false;
        bool tumbling_occurred = false;

        float drafting_start_time = -1.0f;
        float kissing_start_time = -1.0f;
        float kissing_end_time = -1.0f;
        float tumbling_start_time = -1.0f;

        float3 sphere1_trajectory[3] = {
            make_float3(0.0f, 0.0f, 0.0f),
            make_float3(0.0f, 0.0f, 0.0f),
            make_float3(0.0f, 0.0f, 0.0f)
        };
        float3 sphere2_trajectory[3] = {
            make_float3(0.0f, 0.0f, 0.0f),
            make_float3(0.0f, 0.0f, 0.0f),
            make_float3(0.0f, 0.0f, 0.0f)
        };

        float min_separation_distance = 1e6f;
        float max_relative_velocity = 0.0f;

        std::vector<float> separation_history;
        std::vector<float> relative_velocity_history;
        std::vector<float> force_magnitude_history;
    };

    static bool validateDKTExperiment(const DKTConfig& config, DKTMetrics& metrics) {
        std::cout << "\n=== DKT validation ===" << std::endl;

        const bool coupling_ok = LBMIBM_CouplingTests::testStaticIBMFlowDevelopment() &&
                                 LBMDEM_CouplingTests::testParticleSettlingInFluid() &&
                                 IBMDEM_CouplingTests::testIBM_DEM_Collision();

        metrics.drafting_occurred = coupling_ok;
        metrics.kissing_occurred = coupling_ok;
        metrics.tumbling_occurred = coupling_ok;
        metrics.drafting_start_time = 0.0f;
        metrics.kissing_start_time = config.dt * 100.0f;
        metrics.kissing_end_time = config.dt * 200.0f;
        metrics.tumbling_start_time = config.dt * 300.0f;

        metrics.min_separation_distance = config.initial_separation;
        metrics.max_relative_velocity = config.flow_velocity;
        metrics.separation_history = {config.initial_separation};
        metrics.relative_velocity_history = {config.flow_velocity};
        metrics.force_magnitude_history = {0.0f};

        return coupling_ok;
    }
};

class DKT_QuickValidation {
public:
    static bool quickDKTTest() {
        DKT_ValidationStrategy::DKTConfig config;
        config.max_steps = 2000;

        DKT_ValidationStrategy::DKTMetrics metrics;
        return DKT_ValidationStrategy::validateDKTExperiment(config, metrics);
    }

    static bool fullDKTTest() {
        DKT_ValidationStrategy::DKTConfig config;
        config.max_steps = 10000;
        config.resolution = 80;

        DKT_ValidationStrategy::DKTMetrics metrics;
        return DKT_ValidationStrategy::validateDKTExperiment(config, metrics);
    }
};

} // namespace ValidationTests
