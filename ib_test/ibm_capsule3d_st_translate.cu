
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
// Main
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    // Defaults
    std::string stl_path = "../../tools/capsule.stl";
    int nx=384, ny=128, nz=128;
    float tau = 0.8f; // Match sphere example
    float U0 = 0.05f;  // Positive, move RIGHT (like sphere)
    float spacing_req = 1.0f; // User requested 1.0 (dense markers)
    
    int mdf_iter = 5;   // Match sphere example (was 2)
    float beta = -0.5f;  // Match sphere example (was -0.05)
    float angle = 0.0f;
    float scale = 1.0f;
    int steps = 10000;
    int output_every = 100;
    std::string out_dir = "out/capsule_translate_stl";

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
    }

    fs::create_directories(out_dir);

    // 1. Load and prepare Geometry
    std::cout << "\n>>> LOADING STL CAPSULE [" << stl_path << "] <<<" << std::endl;
    STLMesh mesh;
    if (!STLReader::readSTL(stl_path, mesh)) {
        std::cerr << "Failed to load STL: " << stl_path << std::endl;
        exit(1);
    }
    
    // Recenter STL to (0,0,0) based on bbox center
    mesh.centerAtOrigin(); 
    
    // Scale
    if (scale != 1.0f) {
        mesh.transform(make_float3(0,0,0), scale);
    }
    
    // Rotate
    if (angle != 0.0f) {
        rotateMeshZ(mesh, angle);
    }
    
    std::cout << "Geometry Prepared (Centered at Origin):" << std::endl;
    std::cout << "  Bounds: " << mesh.minBound.x << "," << mesh.minBound.y << "," << mesh.minBound.z 
              << " to " << mesh.maxBound.x << "," << mesh.maxBound.y << "," << mesh.maxBound.z << std::endl;

    // 2. Sample Markers (Relative to mesh center at origin)
    float3 center_zero = make_float3(0,0,0);
    std::cout << "Sampling markers with spacing " << spacing_req << "..." << std::endl;
    std::vector<IBMMarker> markers = STLGeometryLoader::sampleSurfaceMarkers(mesh, center_zero, (float)spacing_req);
    
    std::cout << "Generated " << markers.size() << " markers." << std::endl;
    size_t nMarkers = markers.size();

    // Store RELATIVE positions (to be added to actual center)
    std::vector<float3> rel_pos(nMarkers);
    std::vector<float> area_vec(nMarkers);
    for(size_t i=0; i<nMarkers; ++i) {
        rel_pos[i] = markers[i].pos; // Already relative since mesh was centered
        area_vec[i] = markers[i].area;
    }
    
    // Compute characteristic length D (for Re)
    float D = mesh.getSize().y; // Height/Width

    // 3. LBM Setup
    float dx = 1.0f;
    float nu = (tau - 0.5f) / 3.0f;
    float Re = U0 * D / nu;

    std::cout << ">>> Moving Capsule 3D (STL) <<<" << std::endl;
    std::cout << "Domain: " << nx << "x" << ny << "x" << nz << std::endl;
    std::cout << "Capsule: D=" << D << ", U0=" << U0 << ", Re=" << Re << std::endl;

    lbm::LBMConfig lbm_cfg;
    lbm_cfg.nx = nx; lbm_cfg.ny = ny; lbm_cfg.nz = nz;
    lbm_cfg.tau = tau;
    lbm_cfg.enableFreeSurface = false;
    lbm_cfg.collisionModel = lbm::CollisionModel::SRT;
    lbm_cfg.bcXMin = lbm::BC_BOUNCE_BACK; lbm_cfg.bcXMax = lbm::BC_BOUNCE_BACK; // Walls at X
    lbm_cfg.bcYMin = lbm::BC_PERIODIC; lbm_cfg.bcYMax = lbm::BC_PERIODIC;
    lbm_cfg.bcZMin = lbm::BC_PERIODIC; lbm_cfg.bcZMax = lbm::BC_PERIODIC;
    lbm_cfg.gravity = make_float3(0,0,0); // No body force, object moves
    lbm_cfg.rho0 = 1.0f;
    lbm_cfg.u0 = make_float3(0.0f, 0.0f, 0.0f); // Fluid starts at rest

    lbm::LBMCore lbm(lbm_cfg);
    lbm.initialize();

    // 4. IBM Setup
    ibm::IBMParams ibm_p;
    ibm_p.nx = nx; ibm_p.ny = ny; ibm_p.nz = nz;
    ibm_p.dx = dx;
    ibm_p.nMarkers = (int)nMarkers;
    ibm_p.mdf_iterations = mdf_iter;
    ibm_p.mdf_beta = beta; 
    ibm_p.force_method = ibm::IBMForceMethod::DIRECT_FORCING;

    ibm::IBMCore ibm(ibm_p);

    // 5. Buffers - Start capsule at LEFT side (25%), moving RIGHT
    float3 center = make_float3(nx * 0.25f, ny * 0.5f, nz * 0.5f);
    float3 U_obj = make_float3(U0, 0.0f, 0.0f); // U0 > 0 for right movement

    std::vector<float3> h_pos(nMarkers);
    std::vector<float3> h_vel(nMarkers);
    for (size_t k = 0; k < nMarkers; ++k) h_vel[k] = U_obj;

    float3* d_force; CHECK_CUDA(cudaMalloc(&d_force, nx*ny*nz * sizeof(float3)));
    double* d_grid_force_sum_x; double* d_grid_force_sum_y; double* d_grid_force_sum_z;
    CHECK_CUDA(cudaMalloc(&d_grid_force_sum_x, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_grid_force_sum_y, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_grid_force_sum_z, sizeof(double)));

    std::ofstream log_csv(out_dir + "/drag_time.csv");
    log_csv << "step,Cx,DragM,LiftM,Re_est,moving\n";

    // 6. VTK Services Setup
    FieldStore fields;
    StepContext ctx;
    ctx.fields = &fields;

    VTKService::Config vtk_cfg;
    vtk_cfg.output_dir = out_dir;
    vtk_cfg.nx = nx; vtk_cfg.ny = ny; vtk_cfg.nz = nz; vtk_cfg.dx = dx;
    vtk_cfg.interval = output_every;
    vtk_cfg.binary = true;
    vtk_cfg.fields = { "Density", "Speed", "Velocity" };

    MarkerVTKService::Config m_cfg;
    m_cfg.output_dir = out_dir;
    m_cfg.interval = output_every;
    m_cfg.marker_field = "ibm.markers";
    m_cfg.binary = true;

    VTKService vtk_svc(vtk_cfg);
    MarkerVTKService marker_svc(m_cfg);

    vtk_svc.initialize(ctx);
    marker_svc.initialize(ctx);

    // 7. Pre-relaxation
    std::cout << "Pre-relaxation: 500 steps (200 stationary + 300 ramp-up)..." << std::endl;
    float3 center_init = make_float3(nx * 0.25f, ny * 0.5f, nz * 0.5f); // Match main loop
    for (int pre = 0; pre < 500; ++pre) {
        float ramp_factor = (pre < 200) ? 0.0f : (pre - 200) / 300.0f;
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

        // [New Diagnostic] - REMOVED

    }
    std::cout << "Pre-relaxation complete. Starting simulation..." << std::endl;

    // 8. Main Simulation Loop
    for(int t=0; t<=steps; ++t) {
        ctx.step = t;
        ctx.time = t * 1.0;

        // A. Move Capsule
        bool moving = true;
        if (center.x > 0.8f * nx) { // Stop when capsule reaches 80% of domain (right side)
            moving = false;
            U_obj = make_float3(0, 0, 0);
            for (auto& v : h_vel) v = U_obj;
        } else {
            center.x += U0; // dx=1, dt=1
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

        // D. LBM Step
        lbm.setExternalForceFromDeviceAoS(d_force);
        lbm.streamCollide();
        lbm.updateMacroscopic();

        // E. Diagnostics & Output
        if (t % 100 == 0) {
            // Marker Force Sum
            std::vector<float3> m_forces(nMarkers);
            ibm.downloadForces(m_forces.data());
            double fx_m = 0, fy_m = 0;
            for(const auto& f : m_forces) { fx_m += f.x; fy_m += f.y; }
            
            double drag_m = -fx_m;
            double lift_m = -fy_m;

            std::cout << "Step " << t << " Cx=" << std::fixed << std::setprecision(1) << center.x 
                      << " DragM=" << std::scientific << drag_m 
                      << " Re=" << std::fixed << std::setprecision(1) << Re
                      << (moving ? " [MOV]" : " [STOP]") << std::endl;
            
            log_csv << t << "," << center.x << "," << drag_m << "," << lift_m << "," << Re << "," << (moving ? 1 : 0) << "\n";

            // OUTPUT via SERVICES
            if (t > 0 && t % output_every == 0) {
                 // 1. Fluid Fields
                 auto h_rho = fields.create({ "Density", (size_t)nx * ny * nz, sizeof(float) });
                 auto h_u = fields.create({ "Velocity", (size_t)nx * ny * nz, sizeof(float) * 3 });
                 
                 const float* d_rho = lbm.getDensityField();
                 float3* u_aos = lbm.velocityAoSPtr();
                 
                 cudaMemcpy(h_rho.data(), d_rho, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(h_u.data(), u_aos, nx * ny * nz * sizeof(float3), cudaMemcpyDeviceToHost);

                 // 2. Speed
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


                // B. Marker Fields
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
