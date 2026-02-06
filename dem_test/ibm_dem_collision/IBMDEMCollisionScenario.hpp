#pragma once
#include "../../src/core/StepContext.hpp"
#include "../../src/core/FieldStore.hpp"
#include "../../src/physics/ibm/IBMSolver.hpp"
#include "../../src/physics/ibm/IBMCore.hpp" // Added for getCore()
#include "../../src/physics/dem/DEMSolver.hpp"
#include "../../src/physics/dem/DEMCore.hpp" // Added for getCore()
#include "../../src/physics/dem/DEMConfig.hpp"
#include <iostream>

class IBMDEMCollisionScenario {
public:
    struct Config {
        std::string stl_file = "tools/propeller_no_stick.stl";
        float marker_spacing = 0.002f;
        float particle_radius = 0.01f;
        float particle_rho = 2500.0f;
        float stiffness = 1.0e5f;
    };

    IBMDEMCollisionScenario(const Config& cfg) : cfg_(cfg) {}

    void setup(IBMSolver& ibm, DEMSolver& dem, StepContext& ctx) {
        setupIBM(ibm);
        setupDEM(dem);
        
        // Allocate & Init
        // Allocate & Init (Fix: Init first to load STL/count, then Allocate fields, then Init again to upload)
        ibm.initialize(ctx); 
        ibm.allocate(ctx);
        ibm.initialize(ctx); 
        
        dem.allocate(ctx);
        dem.initialize(ctx);
        
        // Manual override for collision setup
        // Propeller rotates at (0.5, 0.5, 0.5).
        // Particle starts at (0.5 + 0.1, 0.5, 0.5) to be hit by blade
        // depending on blade geometry. Propeller radius ~0.2? (Assuming standard asset)
        
        // Let's position particle in the path.
        auto posF = ctx.fields->get(DEMFields::POSITION);
        float3* pos = posF.as<float3>();
        
        // Position: slightly offset from center, in blade path
        // Assume propeller is in XY plane or spinning around Z
        // center (0.5,0.5,0.5).
        // Let's check mesh bounds to be sure? 
        // For now, assume generic prop size ~0.1-0.2 radius.
        pos[0] = make_float3(0.6f, 0.5f, 0.5f); 
        
        // Initial velocity zero
        auto velF = ctx.fields->get(DEMFields::VELOCITY);
        float3* vel = velF.as<float3>();
        vel[0] = make_float3(0,0,0);

        dem.getCore()->uploadPositions(reinterpret_cast<float*>(pos));
        dem.getCore()->uploadVelocities(reinterpret_cast<float*>(vel));
    }

private:
    Config cfg_;

    void setupIBM(IBMSolver& ibm) {
        std::string full_path = "D:/st_lbm/LIDMaster/DILMaster/" + cfg_.stl_file;
        ibm.setSTLFile(full_path);
        ibm.setMarkerSpacing(cfg_.marker_spacing);
        ibm.setMotionType(IBMMotionType::ROTATION);
        ibm.setRotation(0,0,1, 0.5f,0.5f,0.5f, 10.0f); // 10 rad/s
        
        // Fix warning: mdf_beta should be negative
        IBMConfig iCfg;
        iCfg.mdf_beta = -0.5f; 
        iCfg.mdf_iterations = 3;
        // Restore motion settings as setConfig might reset them? 
        // IBMSolver::setConfig overwrites motion type!
        // So we must set config FIRST, or set config values correctly.
        iCfg.motion_type = IBMMotionType::ROTATION;
        iCfg.angular_velocity = 10.0f;
        iCfg.rotation_axis_z = 1.0f;
        iCfg.rotation_center_x = 0.5f; iCfg.rotation_center_y = 0.5f; iCfg.rotation_center_z = 0.5f;
        iCfg.num_markers = 0; // Will be set by STL load
        ibm.setConfig(iCfg);
        // Re-set STL just in case setConfig clears it (it doesn't usually)
        ibm.setSTLFile(full_path);
    }

    void setupDEM(DEMSolver& dem) {
        DEMConfig dCfg;
        dCfg.num_particles = 1;
        dCfg.particle_radius = cfg_.particle_radius;
        dCfg.particle_density = cfg_.particle_rho;
        
        dCfg.gravity_x = 0; dCfg.gravity_y = 0; dCfg.gravity_z = 0; // No gravity
        dCfg.kn = cfg_.stiffness; // Use same stiffness if possible, or internal material
        dCfg.restitution = 1.0f; // Elastic
        
        dCfg.domain_min_x = 0; dCfg.domain_max_x = 1.0f;
        dCfg.domain_min_y = 0; dCfg.domain_max_y = 1.0f;
        dCfg.domain_min_z = 0; dCfg.domain_max_z = 1.0f;

        dem.setConfig(dCfg);
    }
};
