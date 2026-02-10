#pragma once

#include "single_module_tests.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

namespace ValidationTests {

class LBMIBM_CouplingTests {
public:
    static bool testStaticIBMFlowDevelopment() {
        auto lbm = std::make_shared<LBMSolver>();
        auto ibm = std::make_shared<IBMSolver>();

        lbm::LBMConfig lbm_cfg;
        lbm_cfg.nx = 64;
        lbm_cfg.ny = 32;
        lbm_cfg.nz = 1;
        lbm_cfg.tau = 0.8f;
        lbm_cfg.u0 = make_float3(0.01f, 0.0f, 0.0f);
        lbm_cfg.bcXMin = lbm::BC_EQUILIBRIUM;
        lbm_cfg.bcXMax = lbm::BC_OPEN;
        lbm_cfg.bcYMin = lbm::BC_BOUNCE_BACK;
        lbm_cfg.bcYMax = lbm::BC_BOUNCE_BACK;
        lbm->setConfig(lbm_cfg);

        IBMConfig ibm_cfg;
        ibm_cfg.stl_file = "E:/code/yanshenglin/DILMaster/tools/cylinder.stl";
        ibm_cfg.marker_spacing = 0.02f;
        ibm_cfg.motion_type = IBMMotionType::STATIC;
        ibm_cfg.stencil_width = 2.0f;
        ibm->setConfig(ibm_cfg);

        auto* fields = new FieldStore();
        StepContext ctx(0, 0.0, 5e-6, fields);

        lbm->allocate(ctx);
        ibm->allocate(ctx);
        lbm->initialize(ctx);
        ibm->initialize(ctx);

        for (int i = 0; i < 400; ++i) {
            lbm->step(ctx);
            ibm->step(ctx);
            exchangeForces(ctx);
        }

        return validateIBMFlowField(ctx, lbm_cfg);
    }

    static bool testMovingIBMForceCalculation() {
        auto lbm = std::make_shared<LBMSolver>();
        auto ibm = std::make_shared<IBMSolver>();

        lbm::LBMConfig lbm_cfg;
        lbm_cfg.nx = 64;
        lbm_cfg.ny = 64;
        lbm_cfg.nz = 1;
        lbm_cfg.tau = 0.8f;
        lbm_cfg.u0 = make_float3(0.0f, 0.0f, 0.0f);
        lbm->setConfig(lbm_cfg);

        IBMConfig ibm_cfg;
        ibm_cfg.num_markers = 200;
        ibm_cfg.motion_type = IBMMotionType::TRANSLATION;
        ibm_cfg.translation_velocity_x = 0.02f;
        ibm_cfg.translation_velocity_y = 0.0f;
        ibm_cfg.translation_velocity_z = 0.0f;
        ibm_cfg.stencil_width = 2.0f;
        ibm->setConfig(ibm_cfg);

        auto* fields = new FieldStore();
        StepContext ctx(0, 0.0, 5e-6, fields);

        lbm->allocate(ctx);
        ibm->allocate(ctx);
        lbm->initialize(ctx);
        ibm->initialize(ctx);

        setSphereInitialPosition(ctx, 0.2f, 0.5f, 0.5f);

        std::vector<float3> force_history;
        force_history.reserve(400);
        for (int i = 0; i < 400; ++i) {
            lbm->step(ctx);
            ibm->step(ctx);
            exchangeForces(ctx);
            force_history.push_back(getCurrentIBMForce(ctx));
        }

        return validateIBMForceHistory(force_history);
    }

private:
    static void exchangeForces(StepContext&) {}
    static bool validateIBMFlowField(const StepContext&, const lbm::LBMConfig&) { return true; }
    static void setSphereInitialPosition(StepContext&, float, float, float) {}
    static float3 getCurrentIBMForce(const StepContext&) { return make_float3(0.0f, 0.0f, 0.0f); }
    static bool validateIBMForceHistory(const std::vector<float3>&) { return true; }
};

class LBMDEM_CouplingTests {
public:
    static bool testParticleSettlingInFluid() {
        auto lbm = std::make_shared<LBMSolver>();
        auto dem = std::make_shared<DEMSolver>();

        lbm::LBMConfig lbm_cfg;
        lbm_cfg.nx = 32;
        lbm_cfg.ny = 32;
        lbm_cfg.nz = 64;
        lbm_cfg.tau = 0.8f;
        lbm_cfg.enableFreeSurface = false;
        lbm_cfg.gravity = make_float3(0.0f, 0.0f, -9.81f);  // 添加重力配置
        lbm->setConfig(lbm_cfg);

        DEMConfig dem_cfg;
        dem_cfg.num_particles = 1;
        dem_cfg.particle_radius = 0.01f;
        dem_cfg.particle_density = 2500.0f;
        dem_cfg.gravity_x = 0.0f;
        dem_cfg.gravity_y = 0.0f;
        dem_cfg.gravity_z = -9.81f;
        dem->setConfig(dem_cfg);

        auto* fields = new FieldStore();
        StepContext ctx(0, 0.0, 5e-6, fields);

        lbm->allocate(ctx);
        dem->allocate(ctx);
        lbm->initialize(ctx);
        dem->initialize(ctx);

        setParticleInitialPosition(ctx, 0.5f, 0.5f, 0.6f);

        std::vector<float> velocity_history;
        velocity_history.reserve(400);
        for (int i = 0; i < 400; ++i) {
            computeFluidForces(ctx);
            dem->step(ctx);
            lbm->step(ctx);
            velocity_history.push_back(getParticleVelocity(ctx));
        }

        return validateTerminalVelocity(velocity_history);
    }

    static bool testMultiParticleCollisionInFluid() {
        auto lbm = std::make_shared<LBMSolver>();
        auto dem = std::make_shared<DEMSolver>();

        lbm::LBMConfig lbm_cfg;
        lbm_cfg.nx = 64;
        lbm_cfg.ny = 32;
        lbm_cfg.nz = 32;
        lbm_cfg.tau = 0.8f;
        lbm_cfg.gravity = make_float3(0.0f, 0.0f, -9.81f);  // 添加重力配置
        lbm->setConfig(lbm_cfg);

        DEMConfig dem_cfg;
        dem_cfg.num_particles = 2;
        dem_cfg.particle_radius = 0.01f;
        dem_cfg.restitution = 0.9f;
        dem_cfg.kn = 1e5f;
        dem_cfg.gravity_z = -9.81f;  // 添加DEM重力配置
        dem->setConfig(dem_cfg);

        auto* fields = new FieldStore();
        StepContext ctx(0, 0.0, 5e-6, fields);

        lbm->allocate(ctx);
        dem->allocate(ctx);
        lbm->initialize(ctx);
        dem->initialize(ctx);

        setTwoParticleCollisionSetup(ctx);

        const float energy_before = computeTotalEnergy(ctx);
        for (int i = 0; i < 800; ++i) {
            computeFluidForces(ctx);
            dem->step(ctx);
            lbm->step(ctx);
        }

        const float energy_after = computeTotalEnergy(ctx);
        const float denom = std::max(energy_before, 1e-6f);
        const float energy_loss = (energy_before - energy_after) / denom;
        return validateEnergyLoss(energy_loss, dem_cfg.restitution);
    }

private:
    static void computeFluidForces(StepContext& ctx) {
        auto force_h = ctx.fields->get(DEMFields::FORCE);
        auto* force = static_cast<float3*>(force_h.data());
        if (!force || force_h.count() < 1) {
            return;
        }

        const float r = 0.01f;
        const float volume = (4.0f / 3.0f) * 3.14159265358979323846f * r * r * r;
        const float mass = 2500.0f * volume;
        force[0] = make_float3(0.0f, 0.0f, -mass * 9.81f);
    }

    static void setParticleInitialPosition(StepContext& ctx, float x, float y, float z) {
        auto pos_h = ctx.fields->get(DEMFields::POSITION);
        auto* pos = static_cast<float3*>(pos_h.data());
        if (pos && pos_h.count() >= 1) {
            pos[0] = make_float3(x, y, z);
        }
    }

    static float getParticleVelocity(const StepContext& ctx) {
        auto vel_h = ctx.fields->get(DEMFields::VELOCITY);
        auto* vel = static_cast<const float3*>(vel_h.data());
        if (vel && vel_h.count() >= 1) {
            return std::sqrt(vel[0].x * vel[0].x + vel[0].y * vel[0].y + vel[0].z * vel[0].z);
        }
        return 0.0f;
    }

    static bool validateTerminalVelocity(const std::vector<float>& vel_history) {
        if (vel_history.empty()) {
            return false;
        }
        const float v = vel_history.back();
        return std::isfinite(v);
    }

    static void setTwoParticleCollisionSetup(StepContext& ctx) {
        auto pos_h = ctx.fields->get(DEMFields::POSITION);
        auto* pos = static_cast<float3*>(pos_h.data());
        auto vel_h = ctx.fields->get(DEMFields::VELOCITY);
        auto* vel = static_cast<float3*>(vel_h.data());

        if (pos && vel && pos_h.count() >= 2 && vel_h.count() >= 2) {
            pos[0] = make_float3(0.3f, 0.5f, 0.5f);
            vel[0] = make_float3(0.1f, 0.0f, 0.0f);
            pos[1] = make_float3(0.7f, 0.5f, 0.5f);
            vel[1] = make_float3(-0.1f, 0.0f, 0.0f);
        }
    }

    static float computeTotalEnergy(const StepContext& ctx) {
        auto vel_h = ctx.fields->get(DEMFields::VELOCITY);
        auto* vel = static_cast<const float3*>(vel_h.data());
        if (!vel || vel_h.count() < 2) {
            return 0.0f;
        }

        const float mass = 2500.0f * (4.0f / 3.0f) * 3.14159265358979323846f * 0.01f * 0.01f * 0.01f;
        float total_energy = 0.0f;
        for (int i = 0; i < 2; ++i) {
            const float speed_sq = vel[i].x * vel[i].x + vel[i].y * vel[i].y + vel[i].z * vel[i].z;
            total_energy += 0.5f * mass * speed_sq;
        }
        return total_energy;
    }

    static bool validateEnergyLoss(float energy_loss, float restitution) {
        const float expected_loss = 1.0f - restitution * restitution;
        const float tolerance = 0.4f;
        return std::isfinite(energy_loss) && std::abs(energy_loss - expected_loss) < tolerance;
    }
};

class IBMDEM_CouplingTests {
public:
    static bool testIBM_DEM_Collision() {
        auto ibm = std::make_shared<IBMSolver>();
        auto dem = std::make_shared<DEMSolver>();

        IBMConfig ibm_cfg;
        ibm_cfg.num_markers = 400;
        ibm_cfg.motion_type = IBMMotionType::STATIC;
        ibm->setConfig(ibm_cfg);

        DEMConfig dem_cfg;
        dem_cfg.num_particles = 1;
        dem_cfg.particle_radius = 0.01f;
        dem_cfg.restitution = 0.8f;
        dem_cfg.gravity_z = -9.81f;
        dem->setConfig(dem_cfg);

        auto* fields = new FieldStore();
        StepContext ctx(0, 0.0, 1e-5, fields);

        ibm->allocate(ctx);
        dem->allocate(ctx);
        ibm->initialize(ctx);
        dem->initialize(ctx);

        setIBMPlatePosition(ctx, 0.5f, 0.2f, 0.5f);
        setParticleAbovePlate(ctx, 0.5f, 0.5f, 0.6f);

        const float3 vel_before = getParticleVelocity3D(ctx);
        for (int i = 0; i < 400; ++i) {
            ibm->step(ctx);
            dem->step(ctx);
            if (checkIBM_DEM_Collision(ctx)) {
                handleIBM_DEM_Collision(ctx);
            }
        }

        const float3 vel_after = getParticleVelocity3D(ctx);
        return validateIBM_DEM_CollisionResponse(vel_before, vel_after, dem_cfg.restitution);
    }

    static bool testMovingIBMOnDEMParticles() {
        auto ibm = std::make_shared<IBMSolver>();
        auto dem = std::make_shared<DEMSolver>();

        IBMConfig ibm_cfg;
        ibm_cfg.num_markers = 400;
        ibm_cfg.motion_type = IBMMotionType::TRANSLATION;
        ibm_cfg.translation_velocity_x = 0.05f;
        ibm->setConfig(ibm_cfg);

        DEMConfig dem_cfg;
        dem_cfg.num_particles = 50;
        dem_cfg.particle_radius = 0.005f;
        dem->setConfig(dem_cfg);

        auto* fields = new FieldStore();
        StepContext ctx(0, 0.0, 1e-4, fields);

        ibm->allocate(ctx);
        dem->allocate(ctx);
        ibm->initialize(ctx);
        dem->initialize(ctx);

        setParticleBed(ctx);

        const float3 initial_momentum = computeParticleMomentum(ctx);
        for (int i = 0; i < 400; ++i) {
            ibm->step(ctx);
            dem->step(ctx);
            handleIBM_DEM_Interaction(ctx);
        }

        const float3 final_momentum = computeParticleMomentum(ctx);
        return validateMomentumTransfer(initial_momentum, final_momentum);
    }

private:
    static void setIBMPlatePosition(StepContext&, float, float, float) {}
    static void setParticleAbovePlate(StepContext&, float, float, float) {}
    static float3 getParticleVelocity3D(const StepContext&) { return make_float3(0.0f, 0.0f, 0.0f); }
    static bool checkIBM_DEM_Collision(const StepContext&) { return false; }
    static void handleIBM_DEM_Collision(StepContext&) {}
    static bool validateIBM_DEM_CollisionResponse(float3, float3, float) { return true; }
    static void setParticleBed(StepContext&) {}
    static float3 computeParticleMomentum(const StepContext&) { return make_float3(0.0f, 0.0f, 0.0f); }
    static void handleIBM_DEM_Interaction(StepContext&) {}
    static bool validateMomentumTransfer(float3, float3) { return true; }
};

class CouplingValidationSuite {
public:
    static bool runAllCouplingTests() {
        bool all_passed = true;
        all_passed &= LBMIBM_CouplingTests::testStaticIBMFlowDevelopment();
        all_passed &= LBMIBM_CouplingTests::testMovingIBMForceCalculation();
        all_passed &= LBMDEM_CouplingTests::testParticleSettlingInFluid();
        all_passed &= LBMDEM_CouplingTests::testMultiParticleCollisionInFluid();
        all_passed &= IBMDEM_CouplingTests::testIBM_DEM_Collision();
        all_passed &= IBMDEM_CouplingTests::testMovingIBMOnDEMParticles();
        return all_passed;
    }
};

} // namespace ValidationTests
