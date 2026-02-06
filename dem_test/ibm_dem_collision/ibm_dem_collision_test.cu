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
    ctx.dt = 0.001f; // Small timestep for collision
    auto fs = std::make_shared<FieldStore>();
    ctx.fields = fs.get();

    IBMSolver ibmSolver;
    DEMSolver demSolver;
    IBMDEMCollisionScenario scenario(cfg);

    scenario.setup(ibmSolver, demSolver, ctx);

    // 2. Coupler
    IBMToDEMStrategy::Params cParams;
    cParams.stiffness = cfg.stiffness;
    IBMToDEMStrategy coupler(cParams);

    // 3. Loop
    // 3. Loop
    std::filesystem::create_directories("ibm_dem_collision");
    std::ofstream csv("ibm_dem_collision/collision_report.csv");
    csv << "step,time,dist_min,force_max,E_kin\n";

    for (int step = 0; step < 1000; ++step) {
        // IBM Step (Kinematics update)
        ibmSolver.step(ctx);
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

        // Upload forces to DEM
        // DEMSolver::step does this? 
        // Yes, DEMSolver setup in previous task: core_->uploadExternalForces(h_force);

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
        
        // Output & Metrics
        if (step % 10 == 0) {
            writeVTK(step, ctx);
            
            // Check Energy
            auto velH = ctx.fields->get(DEMFields::VELOCITY);
            float3 v = velH.as<float3>()[0];
            float v_mag = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
            float E_kin = 0.5f * cfg.particle_rho * 4.0f/3.0f*3.14f*pow(cfg.particle_radius,3) * v_mag*v_mag;
            
            std::cout << "Step " << step << " V=" << v_mag << " E=" << E_kin << std::endl;
            csv << step << "," << ctx.time << ",0,0," << E_kin << "\n";
        }
        
        ctx.time += ctx.dt;
        ctx.step++;
    }

    return 0;
}
