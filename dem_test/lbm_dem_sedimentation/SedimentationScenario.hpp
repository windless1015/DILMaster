#pragma once

#include "../../src/core/StepContext.hpp"
#include "../../src/core/FieldStore.hpp"
#include "../../src/physics/lbm/LBMSolver.hpp"
#include "../../src/physics/dem/DEMSolver.hpp"
#include "../../src/physics/dem/DEMCore.hpp"   // Added
#include "../../src/physics/dem/DEMConfig.hpp" // Added
#include <iostream>

class SedimentationScenario {
public:
    struct Config {
        float domain_x = 0.05f;
        float domain_y = 0.05f;
        float domain_z = 0.10f;
        
        float particle_d = 0.01f;
        float particle_rho = 2500.0f;
        
        float fluid_rho = 1000.0f;
        float fluid_nu = 1e-6f;
        
        float g_z = -9.81f;
        
        // Discretization
        float dx = 0.001f; // 10 cells per diameter
    };

    SedimentationScenario(const Config& cfg) : cfg_(cfg) {}

    void setup(LBMSolver& lbm, DEMSolver& dem, StepContext& ctx) {
        // 1. Setup LBM
        setupLBM(lbm);
        
        // 2. Setup DEM
        setupDEM(dem);
        
        // 3. Initialize Solvers
        lbm.allocate(ctx);
        lbm.initialize(ctx);
        
        dem.allocate(ctx);
        dem.initialize(ctx);
        
        // 4. Initial Position (Manual Override if needed, though DEMSolver::init does basic)
        // We want particle at top (z=0.08?) center
        auto posF = ctx.fields->get(DEMFields::POSITION);
        float3* pos = posF.as<float3>();
        pos[0] = make_float3(cfg_.domain_x * 0.5f, cfg_.domain_y * 0.5f, 0.08f);
        
        // Upload initial state specifically
        dem.getCore()->uploadPositions(static_cast<float*>(posF.data()));
        dem.getCore()->checkHealth();
        
        std::cout << "[Scenario] Setup Complete. Particle at (0.025, 0.025, 0.08)" << std::endl;
    }

private:
    Config cfg_;

    void setupLBM(LBMSolver& lbm) {
        lbm::LBMConfig lCfg;
        lCfg.nx = static_cast<int>(cfg_.domain_x / cfg_.dx);
        lCfg.ny = static_cast<int>(cfg_.domain_y / cfg_.dx);
        lCfg.nz = static_cast<int>(cfg_.domain_z / cfg_.dx);
        
        // Fluid Properties
        lCfg.rho0 = 1.0f; // Lattice density
        
        // Viscosity/Time Step logic
        // We fix dt to be small enough for stability.
        // Try dt = 5e-5 as derived in planning.
        // nu_lb = nu_phys * dt / dx^2
        // tau = 3 * nu_lb + 0.5
        // If we want tau > 0.51, need nu_lb > 0.0033
        // 1e-6 * dt / 1e-6 > 0.0033 => dt > 0.0033
        // But dt > 0.0033 gives High Gravity.
        // Stability Trade-off: Use TRT (if avail) or just set Tau=0.51
        // and Back-calculate dt.
        // Let's set Tau = 0.52 (safe-ish).
        // nu_lb = (0.52 - 0.5)/3 = 0.02 / 3 = 0.00666.
        // dt = nu_lb * dx^2 / nu_phys = 0.00666 * 1e-6 / 1e-6 = 0.00666 s.
        // Gravity_lb = 9.81 * (0.0066)^2 / 0.001 = 9.81 * 4.4e-5 / 1e-3 = 9.81 * 0.044 = 0.43.
        // g_lb = 0.43 is STILL VERY HIGH.
        // This setup (water in LBM with 1mm grid) is notoriously hard due to Mach constraints.
        // However, since we are doing "Unresolved" validation where LBM is just a background,
        // we can cheat: Set u_f = 0. LBM stability under high gravity?
        // If u=0, rho gradient balances gravity. rho = rho0 * exp(g*h/cs^2).
        // If g is large, density stratifies hugely.
        // 
        // TRICK: Do we NEED to run LBM if u_f is assumed 0?
        // Requirement: "Reuse LBMSolver". "Run LBM".
        // If we enable "Collisions" but disable "Gravity" in LBM?
        // Buoyancy in DEM is (rho_p - rho_f) * g * V.
        // LBM treats fluid. If we want LBM to represent the fluid, it should have gravity?
        // Usually we run LBM with Boussinesq or just pressure gradient.
        // For Validation of DRAG, the fluid can be considered 0-gravity (hydrostatic subtracted) 
        // and we apply buoyancy to DEM manually or via pressure?
        // In the DEM Core, we adding `m * g`. This is F_gravity.
        // Buoyancy is F_b = - rho_f * V * g.
        // Net force F_net = F_g + F_b + F_d.
        // F_net = (rho_p - rho_f) V g + F_d.
        // My `LBMToDEM` strategy adds F_d.
        // Does `DEMSolver` add F_b? No.
        // Should `LBMToDEM` add F_b? Yes, usually. "Unresolved coupling" includes buoyancy + drag + lift + added mass.
        // Tenneti is just Drag.
        // I should ADD Buoyancy in the Strategy.
        // F_buoyancy = - rho_f * Volume * g.
        // Direction opposite to gravity.
        
        lCfg.tau = 0.6f; // Safe value
        lCfg.gravity = {0,0,0}; // Disable LBM gravity to prevent stratification instability
        lCfg.enableFreeSurface = false;
        
        // Boundaries: Periodic sides, Bounce bottom/top?
        lCfg.bcXMin = lbm::BC_BOUNCE_BACK; lCfg.bcXMax = lbm::BC_BOUNCE_BACK;
        lCfg.bcYMin = lbm::BC_BOUNCE_BACK; lCfg.bcYMax = lbm::BC_BOUNCE_BACK; // Box
        lCfg.bcZMin = lbm::BC_BOUNCE_BACK; lCfg.bcZMax = lbm::BC_BOUNCE_BACK; 

        lbm.setConfig(lCfg);
        std::cout << "[Scenario] LBM Configured. Size: " << lCfg.nx << "x" << lCfg.ny << "x" << lCfg.nz << std::endl;
    }

    void setupDEM(DEMSolver& dem) {
        DEMConfig dCfg;
        dCfg.num_particles = 1;
        dCfg.particle_radius = cfg_.particle_d * 0.5f;
        dCfg.particle_density = cfg_.particle_rho;
        
        dCfg.gravity_z = cfg_.g_z;
        dCfg.gravity_x = 0;
        dCfg.gravity_y = 0;
        
        dCfg.domain_min_x = 0; dCfg.domain_max_x = cfg_.domain_x;
        dCfg.domain_min_y = 0; dCfg.domain_max_y = cfg_.domain_y;
        dCfg.domain_min_z = -100.0f; dCfg.domain_max_z = 100.0f; // Infinite fall
        
        // Global DEMConfig uses flattened params
        dCfg.kn = 1e6f;
        dCfg.restitution = 0.5f;
        
        dem.setConfig(dCfg);
    }
};
