
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <map>
#include <numeric>

#include <cuda_runtime.h>
#include "../src/physics/lbm/LBMCore.hpp"
#include "../src/physics/lbm/FreeSurfaceModule.hpp"
#include "../src/physics/ibm/IBMCore.hpp"
#include "../src/physics/lbm/LBMConfig.hpp"
#include "../src/geometry/VectorTypes.h"
#include "../src/geometry/STLReader.h"
#include "../src/geometry/STLGeometryLoader.hpp"

// Services for Time-Series VTK Output
#include "../src/services/VTKService.hpp"
#include "../src/services/MarkerVTKService.hpp"
#include "../src/core/FieldStore.hpp"
#include "../src/core/StepContext.hpp"

namespace fs = std::filesystem;

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// Kernel Helpers
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

__global__ void ker_sum_force(const float3* force_field, double* sum_x, double* sum_y, double* sum_z, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(sum_x, (double)force_field[idx].x);
        atomicAdd(sum_y, (double)force_field[idx].y);
        atomicAdd(sum_z, (double)force_field[idx].z);
    }
}

// Helper to rotate mesh points around Z axis
void rotateMeshZ(STLMesh& mesh, float angle_deg) {
    float rad = angle_deg * 3.14159265359f / 180.0f;
    float c = cosf(rad);
    float s = sinf(rad);
    
    for(auto& tri : mesh.triangles) {
        // Rotate vertices
        for(int k=0; k<3; ++k) {
            float x = tri.vertices[k].x;
            float y = tri.vertices[k].y;
            float nx = x * c - y * s;
            float ny = x * s + y * c;
            tri.vertices[k].x = nx;
            tri.vertices[k].y = ny;
        }
        // Rotate normal
        float nx = tri.normal.x;
        float ny = tri.normal.y;
        tri.normal.x = nx * c - ny * s;
        tri.normal.y = nx * s + ny * c;
    }
    mesh.calculateBoundingBox();
}

// -----------------------------------------------------------------------------
// Main - Free Surface Version
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    // =========================================================================
    // PARAMETERS
    // =========================================================================
    std::string stl_path = "../../tools/capsule.stl";
    
    // Domain - LARGER for free surface
    int nx = 512, ny = 128, nz = 128;
    
    // LBM Parameters
    float tau = 0.8f;
    float U0 = 0.08f;  // FASTER speed for visible wake
    float spacing_req = 1.0f;
    
    // IBM Parameters
    int mdf_iter = 5;
    float beta = -0.5f;
    float angle = 0.0f;
    float scale = 1.0f;
    
    // Simulation
    int steps = 15000;
    int output_every = 100;
    std::string out_dir = "out/capsule_fs";
    
    // =========================================================================
    // FREE SURFACE PARAMETERS (NEW)
    // =========================================================================
    float fluid_fraction = 0.7f;  // Bottom 70% is fluid, top 30% is air
    float capsule_x_ratio = 0.15f;  // Capsule starts at 15% from left edge
    float capsule_depth_ratio = 0.5f; // Capsule center at 50% of fluid depth (below surface)
    
    // Args
    for(int i=1; i<argc; ++i) {
        std::string arg = argv[i];
        if(arg == "--stl") stl_path = argv[++i];
        else if(arg == "--nx") nx = std::stoi(argv[++i]);
        else if(arg == "--ny") ny = std::stoi(argv[++i]);
        else if(arg == "--nz") nz = std::stoi(argv[++i]);
        else if(arg == "--tau") tau = std::stof(argv[++i]);
        else if(arg == "--U0") U0 = std::stof(argv[++i]);
        else if(arg == "--spacing") spacing_req = std::stof(argv[++i]);
        else if(arg == "--mdfIter") mdf_iter = std::stoi(argv[++i]);
        else if(arg == "--beta") beta = std::stof(argv[++i]);
        else if(arg == "--angle_deg") angle = std::stof(argv[++i]);
        else if(arg == "--scale") scale = std::stof(argv[++i]);
        else if(arg == "--steps") steps = std::stoi(argv[++i]);
        else if(arg == "--outputEvery") output_every = std::stoi(argv[++i]);
        else if(arg == "--outDir") out_dir = argv[++i];
        // Free surface specific
        else if(arg == "--fluidFraction") fluid_fraction = std::stof(argv[++i]);
        else if(arg == "--capsuleX") capsule_x_ratio = std::stof(argv[++i]);
        else if(arg == "--capsuleDepth") capsule_depth_ratio = std::stof(argv[++i]);
    }

    fs::create_directories(out_dir);

    // =========================================================================
    // 1. Load and prepare Geometry
    // =========================================================================
    std::cout << "\n>>> LOADING STL CAPSULE [" << stl_path << "] <<<" << std::endl;
    STLMesh mesh;
    if (!STLReader::readSTL(stl_path, mesh)) {
        std::cerr << "Failed to load STL: " << stl_path << std::endl;
        exit(1);
    }
    
    mesh.centerAtOrigin(); 
    if (scale != 1.0f) mesh.transform(make_float3(0,0,0), scale);
    if (angle != 0.0f) rotateMeshZ(mesh, angle);
    
    std::cout << "Geometry Prepared (Centered at Origin):" << std::endl;
    std::cout << "  Bounds: " << mesh.minBound.x << "," << mesh.minBound.y << "," << mesh.minBound.z 
              << " to " << mesh.maxBound.x << "," << mesh.maxBound.y << "," << mesh.maxBound.z << std::endl;

    // 2. Sample Markers
    float3 center_zero = make_float3(0,0,0);
    std::vector<IBMMarker> markers = STLGeometryLoader::sampleSurfaceMarkers(mesh, center_zero, (float)spacing_req);
    std::cout << "Generated " << markers.size() << " markers." << std::endl;
    size_t nMarkers = markers.size();

    std::vector<float3> rel_pos(nMarkers);
    std::vector<float> area_vec(nMarkers);
    for(size_t i=0; i<nMarkers; ++i) {
        rel_pos[i] = markers[i].pos;
        area_vec[i] = markers[i].area;
    }
    
    float D = mesh.getSize().y;

    // =========================================================================
    // 3. LBM Setup - FREE SURFACE ENABLED
    // =========================================================================
    float dx = 1.0f;
    float nu = (tau - 0.5f) / 3.0f;
    float Re = U0 * D / nu;

    // Calculate free surface level (Z coordinate of water surface)
    float water_level_z = nz * fluid_fraction;  // 70% of nz
    
    std::cout << "\n>>> Moving Capsule 3D (Free Surface) <<<" << std::endl;
    std::cout << "Domain: " << nx << "x" << ny << "x" << nz << std::endl;
    std::cout << "Capsule: D=" << D << ", U0=" << U0 << ", Re=" << Re << std::endl;
    std::cout << "Free Surface: " << (fluid_fraction * 100) << "% fluid, " 
              << ((1.0f - fluid_fraction) * 100) << "% air" << std::endl;
    std::cout << "Water Level Z: " << water_level_z << std::endl;

    lbm::LBMConfig lbm_cfg;
    lbm_cfg.nx = nx; lbm_cfg.ny = ny; lbm_cfg.nz = nz;
    lbm_cfg.tau = tau;
    lbm_cfg.enableFreeSurface = true;  // *** FREE SURFACE ENABLED ***
    lbm_cfg.collisionModel = lbm::CollisionModel::SRT;
    lbm_cfg.bcXMin = lbm::BC_BOUNCE_BACK; lbm_cfg.bcXMax = lbm::BC_BOUNCE_BACK;
    lbm_cfg.bcYMin = lbm::BC_PERIODIC; lbm_cfg.bcYMax = lbm::BC_PERIODIC;
    lbm_cfg.bcZMin = lbm::BC_BOUNCE_BACK; lbm_cfg.bcZMax = lbm::BC_OPEN; // Bottom wall, top open
    lbm_cfg.gravity = make_float3(0, 0, -0.0001f); // Slight gravity for free surface stability
    lbm_cfg.rho0 = 1.0f;
    lbm_cfg.u0 = make_float3(0.0f, 0.0f, 0.0f);

    lbm::LBMCore lbm(lbm_cfg);
    lbm.initialize();
    
    // =========================================================================
    // 3b. Initialize Free Surface: Bottom 70% fluid, Top 30% air
    // =========================================================================
    std::cout << "Initializing free surface..." << std::endl;
    
    // Create FreeSurfaceModule and configure it
    lbm::FreeSurfaceModule fsModule;
    fsModule.configure(lbm_cfg);
    
    // Create a FieldStore and register LBM device pointers for the module
    const int nCells = nx * ny * nz;
    FieldStore fsFields;  // Separate FieldStore for free surface initialization
    fsFields.create(FieldDesc{"fluid.density",  (size_t)nCells,     sizeof(float), lbm.densityDevicePtr()});
    fsFields.create(FieldDesc{"fluid.velocity", (size_t)nCells * 3, sizeof(float), lbm.velocityDevicePtr()});
    fsFields.create(FieldDesc{"fluid.flags",    (size_t)nCells,     sizeof(uint8_t), lbm.flagsDevicePtr()});
    fsFields.create(FieldDesc{"fluid.phi",      (size_t)nCells,     sizeof(float), lbm.phiDevicePtr()});
    fsFields.create(FieldDesc{"fluid.mass",     (size_t)nCells,     sizeof(float), lbm.massDevicePtr()});
    
    fsModule.allocate(fsFields);
    fsModule.initialize(fsFields);
    
    // Set regions: First set everything to GAS, then set water region to LIQUID
    int water_z = (int)water_level_z;
    
    // 1) Full domain = GAS
    fsModule.setRegion(0, nx - 1, 0, ny - 1, 0, nz - 1, 
                       lbm::CellType::GAS, 0.0f, lbm_cfg.rho0);
    
    // 2) Water block (bottom 70%) = LIQUID
    fsModule.setRegion(0, nx - 1, 0, ny - 1, 0, water_z - 1, 
                       lbm::CellType::LIQUID, 1.0f, lbm_cfg.rho0);
    
    // 3) Fix interface layer between liquid and gas
    fsModule.fixInterfaceLayer();
    
    std::cout << "Free surface initialized: Water region z=[0, " << water_z - 1 << "]" << std::endl;

    // =========================================================================
    // 4. IBM Setup
    // =========================================================================
    ibm::IBMParams ibm_p;
    ibm_p.nx = nx; ibm_p.ny = ny; ibm_p.nz = nz;
    ibm_p.dx = dx;
    ibm_p.nMarkers = (int)nMarkers;
    ibm_p.mdf_iterations = mdf_iter;
    ibm_p.mdf_beta = beta; 
    ibm_p.force_method = ibm::IBMForceMethod::DIRECT_FORCING;

    ibm::IBMCore ibm(ibm_p);

    // =========================================================================
    // 5. Capsule Initial Position - ADJUSTABLE
    // =========================================================================
    // Capsule center: x at capsule_x_ratio, y at center, z at depth below surface
    float capsule_z = water_level_z - (water_level_z * capsule_depth_ratio);
    float3 center = make_float3(nx * capsule_x_ratio, ny * 0.5f, capsule_z);
    float3 U_obj = make_float3(U0, 0.0f, 0.0f);
    
    std::cout << "Capsule Initial Position: (" << center.x << ", " << center.y << ", " << center.z << ")" << std::endl;
    std::cout << "Capsule Depth Below Surface: " << (water_level_z - center.z) << " cells" << std::endl;

    std::vector<float3> h_pos(nMarkers);
    std::vector<float3> h_vel(nMarkers);
    for (size_t k = 0; k < nMarkers; ++k) h_vel[k] = U_obj;

    float3* d_force; CHECK_CUDA(cudaMalloc(&d_force, nx*ny*nz * sizeof(float3)));
    double* d_grid_force_sum_x; double* d_grid_force_sum_y; double* d_grid_force_sum_z;
    CHECK_CUDA(cudaMalloc(&d_grid_force_sum_x, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_grid_force_sum_y, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_grid_force_sum_z, sizeof(double)));

    std::ofstream log_csv(out_dir + "/drag_time.csv");
    log_csv << "step,Cx,Cz,DragM,LiftM,Re_est,moving\n";

    // =========================================================================
    // 6. VTK Services Setup
    // =========================================================================
    FieldStore fields;
    StepContext ctx;
    ctx.fields = &fields;
    ctx.backend = &lbm.backend();  // Required for FreeSurfaceModule

    VTKService::Config vtk_cfg;
    vtk_cfg.output_dir = out_dir;
    vtk_cfg.nx = nx; vtk_cfg.ny = ny; vtk_cfg.nz = nz; vtk_cfg.dx = dx;
    vtk_cfg.interval = output_every;
    vtk_cfg.binary = true;
    vtk_cfg.fields = { "Density", "Speed", "Velocity", "Fill" }; // Fill = phi (free surface fill fraction)

    MarkerVTKService::Config m_cfg;
    m_cfg.output_dir = out_dir;
    m_cfg.interval = output_every;
    m_cfg.marker_field = "ibm.markers";
    m_cfg.binary = true;

    VTKService vtk_svc(vtk_cfg);
    MarkerVTKService marker_svc(m_cfg);

    vtk_svc.initialize(ctx);
    marker_svc.initialize(ctx);

    // =========================================================================
    // 7. Pre-relaxation (shorter for free surface)
    // =========================================================================
    std::cout << "Pre-relaxation: 200 steps..." << std::endl;
    float3 center_init = center;
    for (int pre = 0; pre < 200; ++pre) {
        float ramp_factor = (pre < 100) ? 0.0f : (pre - 100) / 100.0f;
        float3 current_vel = make_float3(U0 * ramp_factor, 0.0f, 0.0f);
        
        for (size_t k = 0; k < nMarkers; ++k) {
            h_pos[k].x = center_init.x + rel_pos[k].x;
            h_pos[k].y = center_init.y + rel_pos[k].y;
            h_pos[k].z = center_init.z + rel_pos[k].z;
            h_vel[k] = current_vel;
        }
        ibm.updateMarkers(h_pos.data(), h_vel.data(), area_vec.data());
        
        CHECK_CUDA(cudaMemset(d_force, 0, nx * ny * nz * sizeof(float3)));
        float3* u_aos = lbm.velocityAoSPtr();
        ibm.computeForces(u_aos, nullptr, d_force, 1.0f);
        lbm.setExternalForceFromDeviceAoS(d_force);
        lbm.streamCollide();
        lbm.updateMacroscopic();
    }
    std::cout << "Pre-relaxation complete. Starting simulation..." << std::endl;

    // =========================================================================
    // 8. Main Simulation Loop
    // =========================================================================
    for(int t=0; t<=steps; ++t) {
        ctx.step = t;
        ctx.time = t * 1.0;

        // A. Move Capsule
        bool moving = true;
        if (center.x > 0.85f * nx) { // Stop when capsule reaches 85% of domain
            moving = false;
            U_obj = make_float3(0, 0, 0);
            for (auto& v : h_vel) v = U_obj;
        } else {
            center.x += U0;
        }

        // Update Marker Positions
        for (size_t k = 0; k < nMarkers; ++k) {
            h_pos[k].x = center.x + rel_pos[k].x;
            h_pos[k].y = center.y + rel_pos[k].y;
            h_pos[k].z = center.z + rel_pos[k].z;
        }
        ibm.updateMarkers(h_pos.data(), h_vel.data(), area_vec.data());

        // B. Clear Force
        CHECK_CUDA(cudaMemset(d_force, 0, nx*ny*nz * sizeof(float3)));

        // C. IBM Force Calculation
        ibm.computeForces(lbm.velocityAoSPtr(), nullptr, d_force, 1.0f);

        // D. LBM Step with Free Surface
        lbm.setExternalForceFromDeviceAoS(d_force);
        fsModule.preStream(ctx);          // <- Backup old distributions
        lbm.streamCollide();              // <- LBM streaming + collision
        fsModule.postStream(ctx);         // <- Mass update + interface transitions
        lbm.updateMacroscopic();

        // E. Diagnostics & Output
        if (t % 100 == 0) {
            std::vector<float3> m_forces(nMarkers);
            ibm.downloadForces(m_forces.data());
            double fx_m = 0, fy_m = 0;
            for(const auto& f : m_forces) { fx_m += f.x; fy_m += f.y; }
            
            double drag_m = -fx_m;
            double lift_m = -fy_m;

            std::cout << "Step " << t << " Cx=" << std::fixed << std::setprecision(1) << center.x 
                      << " Cz=" << center.z
                      << " DragM=" << std::scientific << drag_m 
                      << " Re=" << std::fixed << std::setprecision(1) << Re
                      << (moving ? " [MOV]" : " [STOP]") << std::endl;
            
            log_csv << t << "," << center.x << "," << center.z << "," << drag_m << "," << lift_m << "," << Re << "," << (moving ? 1 : 0) << "\n";

            // OUTPUT via SERVICES
            if (t > 0 && t % output_every == 0) {
                 auto h_rho = fields.create({ "Density", (size_t)nx * ny * nz, sizeof(float) });
                 auto h_u = fields.create({ "Velocity", (size_t)nx * ny * nz, sizeof(float) * 3 });
                 
                 const float* d_rho = lbm.getDensityField();
                 float3* u_aos = lbm.velocityAoSPtr();
                 
                 cudaMemcpy(h_rho.data(), d_rho, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(h_u.data(), u_aos, nx * ny * nz * sizeof(float3), cudaMemcpyDeviceToHost);

                 // Speed
                 std::vector<float> speed(nx * ny * nz);
                 float3* cpu_u_ptr = (float3*)h_u.data();
                 for (int i = 0; i < nx * ny * nz; ++i) {
                      float ux = cpu_u_ptr[i].x;
                      float uy = cpu_u_ptr[i].y;
                      float uz = cpu_u_ptr[i].z;
                      speed[i] = sqrtf(ux*ux + uy*uy + uz*uz);
                 }
                 auto h_speed = fields.create({ "Speed", (size_t)nx * ny * nz, sizeof(float) });
                 memcpy(h_speed.data(), speed.data(), speed.size() * sizeof(float));

                 // Fill (Free Surface fill fraction) - using phiDevicePtr()
                 auto h_fill = fields.create({ "Fill", (size_t)nx * ny * nz, sizeof(float) });
                 float* d_phi = lbm.phiDevicePtr();
                 if (d_phi) {
                     cudaMemcpy(h_fill.data(), d_phi, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost);
                 }

                // Marker Fields
                auto h_mk = fields.create({ "ibm.markers", nMarkers, sizeof(float) * 3 });
                memcpy(h_mk.data(), h_pos.data(), nMarkers * sizeof(float) * 3);
                auto h_mv = fields.create({ "ibm.velocity", nMarkers, sizeof(float) * 3 });
                memcpy(h_mv.data(), h_vel.data(), nMarkers * sizeof(float) * 3);
                auto h_mf = fields.create({ "ibm.force", nMarkers, sizeof(float) * 3 });
                memcpy(h_mf.data(), m_forces.data(), nMarkers * sizeof(float) * 3);
                auto h_ma = fields.create({ "ibm.area", nMarkers, sizeof(float) });
                memcpy(h_ma.data(), area_vec.data(), nMarkers * sizeof(float));

                // CALL SERVICES
                vtk_svc.onStepEnd(ctx);
                marker_svc.onStepEnd(ctx);
            }
        }
    }
    
    // Cleanup
    vtk_svc.finalize(ctx);
    marker_svc.finalize(ctx);
    log_csv.close();
    cudaFree(d_force);
    cudaFree(d_grid_force_sum_x);
    cudaFree(d_grid_force_sum_y);
    cudaFree(d_grid_force_sum_z);
    
    std::cout << "Simulation finished." << std::endl;
    return 0;
}
