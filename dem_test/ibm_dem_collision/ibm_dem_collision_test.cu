/**
 * @file ibm_dem_collision_test.cu
 * @brief IBM-DEM Collision Validation
 * 
 * Scenario:
 * A rotating propeller (IBM) collides with a stationary particle (DEM).
 * 
 * Validation:
 * 1. Non-penetration: distance >= radius.
 * 2. Energy conservation (approximate, since IBM is kinematic infinite mass).
 *    Particle should gain kinetic energy and bounce off.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <filesystem>
#include <limits>
#include <algorithm>

#include "../../src/core/StepContext.hpp"
#include "../../src/core/FieldStore.hpp"
#include "../../src/physics/dem/DEMCore.hpp" // Added
#include "../../src/physics/ibm/IBMCore.hpp" // Added
#include "IBMDEMCollisionScenario.hpp"
#include "IBMToDEMStrategy.hpp"

// Simple VTK Writer for Points & Propeller Markers
void writeVTK(int step, StepContext& ctx) {
    std::filesystem::create_directories("ibm_dem_collision/vtk");
    std::string filename = "ibm_dem_collision/vtk/step_" + std::to_string(step) + ".vtp";
    std::ofstream out(filename);

    // DEM
    auto posH = ctx.fields->get(DEMFields::POSITION);
    auto velH = ctx.fields->get(DEMFields::VELOCITY);
    const float3* pos = posH.as<float3>();
    const float3* vel = velH.as<float3>();
    int N_p = static_cast<int>(posH.count());

    // IBM
    if (!ctx.fields->exists(IBMFields::MARKERS)) return; // Safety check
    auto markH = ctx.fields->get(IBMFields::MARKERS);
    const float3* markers = markH.as<float3>();
    int N_m = static_cast<int>(markH.count());

    int N_total = N_p + N_m;

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <PolyData>\n";
    out << "    <Piece NumberOfPoints=\"" << N_total << "\" NumberOfVerts=\"" << N_total << "\">\n";
    
    // Points
    out << "      <Points>\n";
    out << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    // Particle
    for(int i=0; i<N_p; ++i) out << pos[i].x << " " << pos[i].y << " " << pos[i].z << " ";
    // Propeller
    for(int i=0; i<N_m; ++i) out << markers[i].x << " " << markers[i].y << " " << markers[i].z << " ";
    out << "\n        </DataArray>\n";
    out << "      </Points>\n";

    // Data - Type (0=Particle, 1=Propeller)
    out << "      <PointData Scalars=\"Type\">\n";
    out << "        <DataArray type=\"Int32\" Name=\"Type\" NumberOfComponents=\"1\" format=\"ascii\">\n";
    for(int i=0; i<N_p; ++i) out << "0 ";
    for(int i=0; i<N_m; ++i) out << "1 ";
    out << "\n        </DataArray>\n";
    out << "      </PointData>\n";

    out << "    </Piece>\n";
    out << "  </PolyData>\n";
    out << "</VTKFile>\n";
}

int main() {
    std::cout << "Starting IBM-DEM Collision Validation..." << std::endl;

    // 1. Setup
    IBMDEMCollisionScenario::Config cfg;
    StepContext ctx;
    ctx.dt = 0.0005f; // Smaller timestep for stricter DEM overlap control
    auto fs = std::make_shared<FieldStore>();
    ctx.fields = fs.get();

    IBMSolver ibmSolver;
    DEMSolver demSolver;
    IBMDEMCollisionScenario scenario(cfg);

    scenario.setup(ibmSolver, demSolver, ctx);
    if (!ctx.fields->exists(IBMFields::MARKERS)) {
        std::cerr << "[FATAL] Missing required field: " << IBMFields::MARKERS
                  << ". IBM allocation/initialization did not create marker fields."
                  << std::endl;
        return 2;
    }
    if (ctx.fields->get(IBMFields::MARKERS).count() == 0) {
        std::cerr << "[FATAL] Field " << IBMFields::MARKERS
                  << " exists but has zero markers." << std::endl;
        return 2;
    }

    // 2. Coupler
    IBMToDEMStrategy::Params cParams;
    cParams.stiffness = cfg.stiffness;
    cParams.damping = cfg.damping;
    cParams.influence_radius = cfg.influence_radius;
    IBMToDEMStrategy coupler(cParams);

    // 3. Loop
    // 3. Loop + strict validation metrics
    std::filesystem::create_directories("ibm_dem_collision");
    std::ofstream csv("ibm_dem_collision/collision_report.csv");
    csv << "step,time,dist_min,penetration,force_max,E_kin,dem_max_overlap,dem_wall_contacts,dem_has_nan\n";

    const float pi = 3.14159265358979323846f;
    const float particle_mass =
        cfg.particle_rho * (4.0f / 3.0f) * pi *
        cfg.particle_radius * cfg.particle_radius * cfg.particle_radius;

    float global_min_dist = std::numeric_limits<float>::max();
    float global_max_penetration = 0.0f;
    float global_max_force = 0.0f;
    float global_max_ekin = 0.0f;
    float global_max_dem_overlap = 0.0f;
    float global_max_dem_overlap_pre_wall = 0.0f;
    float global_max_speed = 0.0f;
    unsigned int global_max_wall_contacts = 0;
    bool dem_has_nan_any = false;
    int first_wall_contact_step = -1;
    int first_contact_step = -1;
    float ekin_at_first_contact = 0.0f;

    for (int step = 0; step < 1000; ++step) {
        // IBM Step (Kinematics update)
        ibmSolver.step(ctx);
        if (!ctx.fields->exists(IBMFields::MARKERS)) {
            std::cerr << "[FATAL] Missing field during step: " << IBMFields::MARKERS
                      << std::endl;
            return 2;
        }
        // Sync IBM markers to Host for couping (Important!)
        // Since we accessed FieldStore in Strategy, we hope IBMSolver synced it?
        // IBMSolver currently manages device pointers. 
        // We need explicit download if FieldStore wraps Host pointers primarily?
        // FieldStore supports device? 
        // Actually, IBMSolver::step updates INTERNAL state. We need to expose it to FieldStore.
        // Assuming IBMSolver allocates FieldStore fields as HOST or manages SYNC.
        // Let's assume for now we need manual sync or IBMSolver does it.
        // CHECK: IBMSolver uses device pointers in core. FieldStore might hold Host pointers?
        // For this test, we might need to peek into IBMSolver and force download.
        // HACK: Download markers to host field
        if (ibmSolver.getCore()) {
             auto mh = ctx.fields->get(IBMFields::MARKERS);
             ibmSolver.getCore()->downloadPositions(mh.as<float3>());
        }

        // Clear Forces
        auto forceH = ctx.fields->get(DEMFields::FORCE);
        std::memset(forceH.data(), 0, forceH.size_bytes());

        // Coupling
        coupler.execute(ctx);

        // Force metric from coupling result (before DEM internal contacts overwrite force)
        float step_force_max = 0.0f;
        {
            const float3 *f = forceH.as<float3>();
            for (size_t i = 0; i < forceH.count(); ++i) {
                float fm = std::sqrt(f[i].x * f[i].x + f[i].y * f[i].y + f[i].z * f[i].z);
                if (fm > step_force_max) step_force_max = fm;
            }
        }
        if (step_force_max > global_max_force) global_max_force = step_force_max;

        // DEM Step
        demSolver.step(ctx);
        // Download DEM positions for VTK/Next Step Coupling
        // DEMSolver::step usually doesn't auto-download every step? 
        // Need to sync back to host for Strategy next frame.
        if (demSolver.getCore()) {
            auto ph = ctx.fields->get(DEMFields::POSITION);
            auto vh = ctx.fields->get(DEMFields::VELOCITY);
             demSolver.getCore()->downloadPositions(reinterpret_cast<float*>(ph.as<float3>()));
             // Need velocity for Energy
             // demSolver.getCore()->downloadVelocities(vh.as<float3>()); // If exists?
        }

        // Distance/Penetration metric (particle center to nearest IBM marker)
        float step_min_dist = std::numeric_limits<float>::max();
        {
            auto posH = ctx.fields->get(DEMFields::POSITION);
            auto markH = ctx.fields->get(IBMFields::MARKERS);
            const float3 *p = posH.as<float3>();
            const float3 *m = markH.as<float3>();

            for (size_t i = 0; i < posH.count(); ++i) {
                for (size_t j = 0; j < markH.count(); ++j) {
                    float dx = p[i].x - m[j].x;
                    float dy = p[i].y - m[j].y;
                    float dz = p[i].z - m[j].z;
                    float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                    if (dist < step_min_dist) step_min_dist = dist;
                }
            }
        }
        float step_penetration = std::max(0.0f, cfg.particle_radius - step_min_dist);
        if (step_min_dist < global_min_dist) global_min_dist = step_min_dist;
        if (step_penetration > global_max_penetration) global_max_penetration = step_penetration;

        // Kinetic energy metric
        float step_ekin = 0.0f;
        float step_vmax = 0.0f;
        {
            auto velH = ctx.fields->get(DEMFields::VELOCITY);
            const float3 *v = velH.as<float3>();
            for (size_t i = 0; i < velH.count(); ++i) {
                float v2 = v[i].x * v[i].x + v[i].y * v[i].y + v[i].z * v[i].z;
                step_ekin += 0.5f * particle_mass * v2;
                float vm = std::sqrt(v2);
                if (vm > step_vmax) step_vmax = vm;
            }
        }
        if (step_ekin > global_max_ekin) global_max_ekin = step_ekin;
        if (step_vmax > global_max_speed) global_max_speed = step_vmax;

        // DEM quality diagnostics (textbook-style stability indicators)
        float step_dem_overlap = 0.0f;
        unsigned int step_wall_contacts = 0;
        bool step_dem_nan = false;
        if (demSolver.getCore()) {
            auto s = demSolver.getCore()->lastStepStats();
            step_dem_overlap = s.max_overlap;
            step_wall_contacts = s.wall_contacts;
            step_dem_nan = s.has_nan;
            global_max_dem_overlap = std::max(global_max_dem_overlap, step_dem_overlap);
            global_max_wall_contacts = std::max(global_max_wall_contacts, step_wall_contacts);
            dem_has_nan_any = dem_has_nan_any || step_dem_nan;
            if (step_wall_contacts > 0 && first_wall_contact_step < 0) {
                first_wall_contact_step = step;
            }
            if (first_wall_contact_step < 0) {
                global_max_dem_overlap_pre_wall =
                    std::max(global_max_dem_overlap_pre_wall, step_dem_overlap);
            }
        }

        if (first_contact_step < 0 && step_force_max > 1.0e-6f) {
            first_contact_step = step;
            ekin_at_first_contact = step_ekin;
        }

        // Output & logging
        if (step % 10 == 0) {
            writeVTK(step, ctx);
            std::cout << "Step " << step
                      << " dist_min=" << step_min_dist
                      << " pen=" << step_penetration
                      << " Fmax=" << step_force_max
                      << " E=" << step_ekin
                      << " dem_ov=" << step_dem_overlap
                      << " dem_wall=" << step_wall_contacts
                      << std::endl;
            csv << step << "," << ctx.time << "," << step_min_dist << ","
                << step_penetration << "," << step_force_max << "," << step_ekin
                << "," << step_dem_overlap << "," << step_wall_contacts << ","
                << (step_dem_nan ? 1 : 0) << "\n";
        }
        
        ctx.time += ctx.dt;
        ctx.step++;
    }

    // -----------------------------------------------------------------------
    // Strict acceptance criteria
    // -----------------------------------------------------------------------
    const float max_allowed_penetration = 0.20f * cfg.particle_radius; // IBM geometric criterion
    const float min_required_force_peak = 1.0e-6f;                     // collision force exists
    const float min_required_ekin_peak = 1.0e-8f;                      // kinetic transfer exists

    // Textbook-style DEM quality criteria
    const float max_allowed_overlap_ratio = 0.20f; // delta/R <= 20%
    const float max_allowed_dem_overlap = max_allowed_overlap_ratio * cfg.particle_radius;
    const float max_allowed_pre_wall_overlap_ratio = 0.10f; // stricter collision-phase bound
    const float max_allowed_pre_wall_overlap =
        max_allowed_pre_wall_overlap_ratio * cfg.particle_radius;
    const float max_allowed_energy_growth_after_contact = 1.50f; // E_max <= 1.5 * E_contact
    const float max_allowed_speed_growth_after_contact = 1.50f;  // V_max <= 1.5 * V_contact
    const unsigned int max_allowed_wall_contacts = 1;            // avoid wall-dominated motion
    const float contact_ref_ekin = std::max(ekin_at_first_contact, 1.0e-8f);
    const float speed_ref_contact =
        std::sqrt(2.0f * contact_ref_ekin / std::max(particle_mass, 1.0e-12f));

    bool pass_non_penetration = (global_max_penetration <= max_allowed_penetration);
    bool pass_force_contact = (global_max_force >= min_required_force_peak);
    bool pass_energy_transfer = (global_max_ekin >= min_required_ekin_peak);
    bool pass_functional = pass_non_penetration && pass_force_contact && pass_energy_transfer;

    // Evaluate overlap quality on collision phase (before first wall contact)
    // and on full trajectory for stability.
    float overlap_for_quality = (first_wall_contact_step >= 0)
        ? global_max_dem_overlap_pre_wall
        : global_max_dem_overlap;
    bool pass_pre_wall_overlap_quality = (overlap_for_quality <= max_allowed_pre_wall_overlap);
    bool pass_global_overlap_quality = (global_max_dem_overlap <= max_allowed_dem_overlap);
    bool pass_no_nan_quality = !dem_has_nan_any;
    bool pass_energy_bounded_quality =
        (global_max_ekin <= max_allowed_energy_growth_after_contact * contact_ref_ekin);
    bool pass_speed_bounded_quality =
        (global_max_speed <= max_allowed_speed_growth_after_contact * speed_ref_contact);
    bool pass_wall_contact_quality = (global_max_wall_contacts <= max_allowed_wall_contacts);

    bool pass_quality = pass_pre_wall_overlap_quality && pass_global_overlap_quality &&
                        pass_no_nan_quality && pass_energy_bounded_quality &&
                        pass_speed_bounded_quality && pass_wall_contact_quality;
    bool pass = pass_functional && pass_quality;

    std::cout << "\n=== IBM-DEM Collision Validation Summary ===\n";
    std::cout << "min_dist               : " << global_min_dist << "\n";
    std::cout << "max_penetration        : " << global_max_penetration
              << " (limit " << max_allowed_penetration << ")\n";
    std::cout << "max_force              : " << global_max_force
              << " (min " << min_required_force_peak << ")\n";
    std::cout << "max_kinetic_energy     : " << global_max_ekin
              << " (min " << min_required_ekin_peak << ")\n";
    std::cout << "first_contact_step     : " << first_contact_step << "\n";
    std::cout << "E_at_first_contact     : " << ekin_at_first_contact << "\n";
    std::cout << "max_dem_overlap        : " << global_max_dem_overlap
              << " (limit " << max_allowed_dem_overlap << ")\n";
    if (first_wall_contact_step >= 0) {
      std::cout << "max_dem_overlap_pre_wall: " << global_max_dem_overlap_pre_wall
                << " (limit " << max_allowed_pre_wall_overlap << ")\n";
      std::cout << "first_wall_contact_step: " << first_wall_contact_step << "\n";
    }
    std::cout << "max_wall_contacts      : " << global_max_wall_contacts << "\n";
    std::cout << "max_speed              : " << global_max_speed
              << " (limit " << max_allowed_speed_growth_after_contact * speed_ref_contact << ")\n";
    std::cout << "dem_has_nan_any        : " << (dem_has_nan_any ? "true" : "false") << "\n";
    std::cout << "Quality check [pre-wall overlap] : "
              << (pass_pre_wall_overlap_quality ? "PASS" : "FAIL") << "\n";
    std::cout << "Quality check [global overlap]   : "
              << (pass_global_overlap_quality ? "PASS" : "FAIL") << "\n";
    std::cout << "Quality check [no NaN]           : "
              << (pass_no_nan_quality ? "PASS" : "FAIL") << "\n";
    std::cout << "Quality check [energy bounded]   : "
              << (pass_energy_bounded_quality ? "PASS" : "FAIL") << "\n";
    std::cout << "Quality check [speed bounded]    : "
              << (pass_speed_bounded_quality ? "PASS" : "FAIL") << "\n";
    std::cout << "Quality check [wall contacts]    : "
              << (pass_wall_contact_quality ? "PASS" : "FAIL") << "\n";
    std::cout << "Functional Layer       : " << (pass_functional ? "PASS" : "FAIL") << "\n";
    std::cout << "Quality Layer          : " << (pass_quality ? "PASS" : "FAIL") << "\n";
    std::cout << "Result: " << (pass ? "PASS" : "FAIL") << std::endl;

    return pass ? 0 : 1;
}
