#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <memory>
#include <string>
#include <cassert>

// Include paths relative to source tree
#include "../src/physics/dem/DEMCore.hpp"
#include "../src/physics/dem/DEMConfig.hpp"

namespace fs = std::filesystem;

// ============================================================================
// Mini Test Framework
// ============================================================================
int g_tests_passed = 0;
int g_tests_failed = 0;

#define ASSERT_TRUE(cond) \
    if (!(cond)) { \
        std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__ << " Assertion failed: " #cond << std::endl; \
        return false; \
    }

#define ASSERT_NEAR(val1, val2, tol) \
    if (std::abs((val1) - (val2)) > (tol)) { \
        std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__ << " " << (val1) << " not near " << (val2) << " (tol=" << (tol) << ")" << std::endl; \
        return false; \
    }

#define ASSERT_LT(val1, val2) \
    if (!((val1) < (val2))) { \
        std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__ << " " << (val1) << " >= " << (val2) << std::endl; \
        return false; \
    }

#define ASSERT_GT(val1, val2) \
    if (!((val1) > (val2))) { \
        std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__ << " " << (val1) << " <= " << (val2) << std::endl; \
        return false; \
    }

#define RUN_TEST(func) \
    std::cout << "[RUN ] " << #func << std::endl; \
    if (func()) { \
        std::cout << "[PASS] " << #func << std::endl; \
        g_tests_passed++; \
    } else { \
        std::cout << "[FAIL] " << #func << std::endl; \
        g_tests_failed++; \
    }

// ============================================================================
// Helper
// ============================================================================
std::unique_ptr<dem::DEMCore> createCore(const dem::DEMConfig& cfg) {
    auto core = std::make_unique<dem::DEMCore>();
    core->configure(cfg);
    core->allocate();
    return core;
}

// ============================================================================
// Tests
// ============================================================================

bool Test_T0_ElasticBounce() {
    dem::DEMConfig cfg;
    cfg.num_particles = 1;
    cfg.particle_radius = 0.1f;
    cfg.particle_density = 2500.0f;
    cfg.gravity_z = -9.81f;
    cfg.material.restitution = 1.0f; // Elastic
    cfg.material.kn = 1e7f; // Stiffer
    cfg.material.mu = 0.0f;
    cfg.dem_substeps = 10; // Use substeps for stability
    
    // Explicit domain to avoid wall issues
    cfg.domain_min_x = 0.0f; cfg.domain_max_x = 1.0f;
    cfg.domain_min_y = 0.0f; cfg.domain_max_y = 1.0f;
    cfg.domain_min_z = 0.0f; cfg.domain_max_z = 10.0f;
    
    auto core = createCore(cfg);
    core->checkHealth();

    std::vector<float> pos(3, 0.5f); pos[2] = 5.0f; // Center X/Y
    std::vector<float> vel(3, 0.0f);
    std::vector<float> rad(1, 0.1f);
    
    core->uploadPositions(pos.data());
    core->uploadVelocities(vel.data());
    core->uploadRadii(rad.data());
    core->initMassProperties();

    float dt = 1e-3f; // Outer step
    // Inner step will be 1e-4
    
    float max_h1 = 0.0f;
    bool has_bounced = false;
    float h0 = 5.0f;
    
    for (int i = 0; i < 2000; ++i) { // 2 seconds
        core->stepMultiple(dt, cfg.dem_substeps);
        core->downloadPositions(pos.data());
        core->downloadVelocities(vel.data());
        
        float z = pos[2];
        float vz = vel[2];
        
        // Bounce detection: moving up and near floor (radius 0.1)
        if (vz > 1.0f && z > 0.1f) has_bounced = true;
        if (has_bounced) {
            if (z > max_h1) max_h1 = z;
        }
    }
    
    printf("    [T0] Elastic Bounce (e=1.0) H0=%.3f, H1=%.3f\n", h0, max_h1);
    // Should be close to 5.0
    ASSERT_NEAR(max_h1, h0, 0.1f);
    return true;
}

bool Test_T1_BounceRestitution() {
    dem::DEMConfig cfg;
    cfg.num_particles = 1;
    cfg.particle_radius = 0.1f;
    cfg.particle_density = 2500.0f;
    cfg.gravity_z = -9.81f;
    cfg.material.restitution = 0.8f;
    cfg.material.kn = 1e7f; // Stiffer to reduce overlap
    cfg.material.mu = 0.0f;
    cfg.dem_substeps = 10;
    
    cfg.domain_min_x = 0.0f; cfg.domain_max_x = 1.0f;
    cfg.domain_min_y = 0.0f; cfg.domain_max_y = 1.0f;
    cfg.domain_min_z = 0.0f; cfg.domain_max_z = 10.0f;
    
    auto core = createCore(cfg);
    std::vector<float> pos(3, 0.5f); pos[2] = 5.0f; // Center X/Y
    std::vector<float> vel(3, 0.0f);
    std::vector<float> rad(1, 0.1f);
    
    core->uploadPositions(pos.data());
    core->uploadVelocities(vel.data());
    core->uploadRadii(rad.data());
    core->initMassProperties();

    float dt = 1e-3f;
    float max_h1 = 0.0f;
    bool has_bounced = false;
    float h0 = 5.0f;
    
    for (int i = 0; i < 2500; ++i) { 
        core->stepMultiple(dt, cfg.dem_substeps);
        core->downloadPositions(pos.data());
        core->downloadVelocities(vel.data());
        float z = pos[2];
        float vz = vel[2];
        
        if (vz > 1.0f && z > 0.1f) has_bounced = true;
        if (has_bounced) {
            if (z > max_h1) max_h1 = z;
        }
    }

    float drop_h = h0 - cfg.particle_radius;
    float expected_h1 = cfg.particle_radius + drop_h * pow(cfg.material.restitution, 2.0f);
    
    // Physics parameters for logging
    float m_particle = (4.0f/3.0f) * 3.14159f * pow(cfg.particle_radius, 3) * cfg.particle_density;
    // t_c = pi * sqrt(m/kn)
    float t_c = 3.14159f * std::sqrt(m_particle / cfg.material.kn);
    // gamma (analytical)
    float ln_e = std::log(cfg.material.restitution);
    float gamma_theory = -2.0f * ln_e * std::sqrt(cfg.material.kn * m_particle) 
                       / std::sqrt(3.14159f*3.14159f + ln_e*ln_e);

    printf("    [T1] e=%.2f, kn=%.0f, m=%.4e, t_c=%.4e, gamma=%.4f\n", 
           cfg.material.restitution, cfg.material.kn, m_particle, t_c, gamma_theory);
    printf("    [T1] H0=%.3f, H1_sim=%.3f, H1_theory=%.3f\n", h0, max_h1, expected_h1);
    
    float error = std::abs(max_h1 - expected_h1) / expected_h1;
    ASSERT_LT(error, 0.02f); // Strict goal < 2%
    return true;
}

bool Test_T2_Momentum() {
    dem::DEMConfig cfg;
    cfg.num_particles = 2;
    cfg.particle_radius = 0.1f;
    cfg.particle_density = 1000.0f;
    cfg.gravity_z = 0.0f;
    cfg.material.restitution = 1.0f;
    cfg.material.kn = 1e6f;
    cfg.material.mu = 0.0f;
    cfg.dem_substeps = 10;
    
    // Ensure sufficient space
    cfg.domain_min_x = -2.0f; cfg.domain_max_x = 2.0f;
    cfg.domain_min_y = -1.0f; cfg.domain_max_y = 1.0f;
    cfg.domain_min_z = -1.0f; cfg.domain_max_z = 1.0f;
    
    auto core = createCore(cfg);
    
    // SoA Layout: X... Y... Z...
    std::vector<float> pos(3 * 2); 
    pos[0] = -0.6f; pos[1] = 0.6f; // P0.x, P1.x
    pos[2] = 0.0f;  pos[3] = 0.0f; // P0.y, P1.y
    pos[4] = 0.0f;  pos[5] = 0.0f; // P0.z, P1.z
    
    std::vector<float> vel(3 * 2);
    vel[0] = 1.0f;  vel[1] = -1.0f; // P0.vx, P1.vx
    vel[2] = 0.0f;  vel[3] = 0.0f;
    vel[4] = 0.0f;  vel[5] = 0.0f;
    
    std::vector<float> rad = { 0.1f, 0.1f };

    core->uploadPositions(pos.data());
    core->uploadVelocities(vel.data());
    core->uploadRadii(rad.data());
    core->initMassProperties();

    float dt = 1e-3f;
    for (int i = 0; i < 1000; ++i) core->stepMultiple(dt, cfg.dem_substeps);
    
    core->downloadVelocities(vel.data());
     printf("    [T2] V_left=%.4f, V_right=%.4f\n", vel[0], vel[1]);
    
    // Should rebound
    ASSERT_LT(vel[0], -0.9f); // Left particle moving left
    ASSERT_GT(vel[1], 0.9f);  // Right particle moving right

    float p_final = vel[0] + vel[1];
    printf("    [T2] Final V_sum: %.4f\n", p_final);
    ASSERT_NEAR(p_final, 0.0f, 1e-3f);
    return true;
}

bool Test_T3_InclineFriction() {
    dem::DEMConfig cfg;
    cfg.num_particles = 1;
    cfg.particle_radius = 0.1f;
    cfg.particle_density = 2500.0f;
    cfg.material.restitution = 0.0f;
    cfg.material.kn = 1e6f;
    cfg.material.kt = 1e6f; // Increase tangential stiffness
    cfg.dem_substeps = 5;
    float mu_static = 0.5f;
    cfg.material.mu = mu_static;
    
    // Use standard 0..1 domain
    cfg.domain_min_x = 0; cfg.domain_max_x = 10.0;
    cfg.domain_min_y = 0; cfg.domain_max_y = 1.0;
    cfg.domain_min_z = 0; cfg.domain_max_z = 1.0;
    
    // Case A: Stick
    {
        float theta = 20.0f * 3.14159f / 180.0f;
        cfg.gravity_z = -9.81f * cos(theta);
        cfg.gravity_x =  9.81f * sin(theta);
        auto core = createCore(cfg);
        std::vector<float> pos = {0.5f, 0.5f, cfg.particle_radius - 1e-4f};
        std::vector<float> vel = {0,0,0};
        std::vector<float> rad = {cfg.particle_radius};
        core->uploadPositions(pos.data());
        core->uploadVelocities(vel.data());
        core->uploadRadii(rad.data());
        core->initMassProperties();
        for(int i=0; i<4000; ++i) core->stepMultiple(1e-3f, cfg.dem_substeps); // more steps to verify stick
        core->downloadVelocities(vel.data());
        printf("    [T3-A] Stick V=%.4f\n", vel[0]);
        ASSERT_NEAR(vel[0], 0.0f, 0.05f);
    }
    // Case B: Slide
    {
        float theta = 35.0f * 3.14159f / 180.0f;
        cfg.gravity_z = -9.81f * cos(theta);
        cfg.gravity_x =  9.81f * sin(theta);
        auto core = createCore(cfg);
        std::vector<float> pos = {0.5f, 0.5f, cfg.particle_radius - 1e-4f};
        std::vector<float> vel = {0,0,0};
        std::vector<float> rad = {cfg.particle_radius};
        core->uploadPositions(pos.data());
        core->uploadVelocities(vel.data());
        core->uploadRadii(rad.data());
        core->initMassProperties();
        for(int i=0; i<2000; ++i) core->stepMultiple(1e-3f, cfg.dem_substeps);
        core->downloadVelocities(vel.data());
        printf("    [T3-B] Slide V=%.4f\n", vel[0]);
        ASSERT_GT(vel[0], 0.5f);
    }
    return true;
}

bool Test_T4_Packing() {
    dem::DEMConfig cfg;
    int N_side = 4; int N = N_side*N_side*N_side;
    cfg.num_particles = N;
    cfg.particle_radius = 0.05f;
    cfg.particle_density = 2500.0f;
    cfg.gravity_z = -9.81f;
    cfg.material.kn = 1e6f; // Stiffer
    cfg.material.mu = 0.3f;
    cfg.dem_substeps = 5;
    cfg.domain_min_x = 0; cfg.domain_max_x = 1.0;
    cfg.domain_min_y = 0; cfg.domain_max_y = 1.0;
    cfg.domain_min_z = 0; cfg.domain_max_z = 2.0;

    auto core = createCore(cfg);
    std::vector<float> pos(3*N), vel(3*N, 0.0f), rad(N, 0.05f);
    int idx = 0;
    for(int z=0; z<N_side; ++z) {
        for(int y=0; y<N_side; ++y) {
            for(int x=0; x<N_side; ++x) {
                pos[idx]     = 0.2f + x * 0.12f;
                pos[idx+N]   = 0.2f + y * 0.12f;
                pos[idx+2*N] = 0.5f + z * 0.12f;
                idx++;
            }
        }
    }
    core->uploadPositions(pos.data());
    core->uploadVelocities(vel.data());
    core->uploadRadii(rad.data());
    core->initMassProperties();

    float dt = 1e-3f;
    for(int i=0; i<2000; ++i) core->stepMultiple(dt, cfg.dem_substeps);
    
    core->downloadVelocities(vel.data());
    float total_ke = 0.0f;
    for(float v : vel) total_ke += v*v;
    auto stats = core->lastStepStats();
    
    printf("    [T4] KE_sum=%.4f, MaxOverlap=%.4e\n", total_ke, stats.max_overlap);
    ASSERT_LT(total_ke, 2.0f);
    ASSERT_LT(stats.max_overlap, 0.05f * 0.05f);
    return true;
}

bool Test_T5_Convergence() {
    auto run_sim = [&](float dt) -> float {
        dem::DEMConfig cfg;
        cfg.num_particles = 1;
        cfg.particle_radius = 0.1f;
        cfg.particle_density = 2500.0f;
        cfg.gravity_z = -9.81f;
        cfg.material.restitution = 0.8f;
        cfg.material.kn = 5e7f; // Stiff to keep overlap < 5%
        
        cfg.domain_min_x = 0.0f; cfg.domain_max_x = 1.0f;
        cfg.domain_min_y = 0.0f; cfg.domain_max_y = 1.0f;
        cfg.domain_min_z = 0.0f; cfg.domain_max_z = 10.0f;
        
        cfg.domain_min_x = 0.0f; cfg.domain_max_x = 1.0f;
        cfg.domain_min_y = 0.0f; cfg.domain_max_y = 1.0f;
        cfg.domain_min_z = 0.0f; cfg.domain_max_z = 10.0f;
        
        cfg.dem_substeps = 1; // Ignored effectively by adaptive logic below
        auto core = createCore(cfg);
        std::vector<float> pos(3, 0.5f); pos[2] = 5.0f; // Center
        std::vector<float> vel(3, 0.0f);
        std::vector<float> rad(1, 0.1f);
        core->uploadPositions(pos.data());
        core->uploadVelocities(vel.data());
        core->uploadRadii(rad.data());
        core->initMassProperties();
        
        // Adaptive substepping logic
        float t_c = core->getCriticalTimeStep();
        int n_sub = 1;
        if (t_c > 1e-12f) {
             // Textbook stability safety factor 0.2
             n_sub = static_cast<int>(std::ceil(dt / (t_c * 0.2f)));
        }
        if (n_sub < 1) n_sub = 1;
        
        printf("    [T5] dt=%.1e, t_c=%.1e, n_sub=%d\n", dt, t_c, n_sub);

        float max_z = 0.0f;
        bool bounced = false;
        int steps = static_cast<int>(2.0f / dt);
        
        for(int i=0; i<steps; ++i) {
            core->stepMultiple(dt, n_sub);
            
            // Check overlaps
            if (i % 100 == 0) {
                 auto stats = core->lastStepStats();
                 // Expect minimal overlap
                 if (stats.max_overlap > cfg.particle_radius * 0.05f) {
                     printf("[T5 WARNING] High overlap %.4e at step %d\n", stats.max_overlap, i);
                 }
            }

            core->downloadPositions(pos.data());
            core->downloadVelocities(vel.data());
            if(vel[2] > 0.5f && pos[2] > 0.11f) bounced = true;
            if(bounced && pos[2] > max_z) max_z = pos[2];
        }
        return max_z;
    };
    
    float h1 = run_sim(1e-4f);
    float h2 = run_sim(5e-5f); // Smaller dt should yield similar result
    printf("    [T5] dt=1e-4 H=%.4f, dt=5e-5 H=%.4f\n", h1, h2);
    ASSERT_NEAR(h1, h2, 0.1f);  
    return true;
}

int main() {
    std::cout << "Running DEM Fundamentals Test Suite..." << std::endl;
    
    RUN_TEST(Test_T0_ElasticBounce);
    RUN_TEST(Test_T1_BounceRestitution);
    RUN_TEST(Test_T2_Momentum);
    RUN_TEST(Test_T3_InclineFriction);
    RUN_TEST(Test_T4_Packing);
    RUN_TEST(Test_T5_Convergence);
    
    std::cout << "\nSummary: " << g_tests_passed << " PASSED, " << g_tests_failed << " FAILED." << std::endl;
    return (g_tests_failed == 0) ? 0 : 1;
}
