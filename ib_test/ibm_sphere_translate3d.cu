#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <fstream>

#include "physics/lbm/LBMCore.hpp"
#include "physics/ibm/IBMCore.hpp"
// Include Services
#include "services/VTKService.hpp"
#include "services/MarkerVTKService.hpp"
#include "core/FieldStore.hpp"
#include "core/StepContext.hpp"

namespace fs = std::filesystem;

#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

// ---- Kernels ----
__global__ void kComputeFluidProps(float3* u_aos, float* rho, float3* f_ib, float* p, float3* omega,
    int nx, int ny, int nz, float cs2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    int idx = z * nx * ny + y * nx + x;

    // Pressure
    float r = rho[idx];
    p[idx] = cs2 * (r - 1.0f);

    // Vorticity: Curl(U)
    // Central difference where possible
    int xp = (x + 1) % nx; int xm = (x - 1 + nx) % nx;
    int yp = (y + 1) % ny; int ym = (y - 1 + ny) % ny;
    int zp = (z + 1) % nz; int zm = (z - 1 + nz) % nz;

    float3 u = u_aos[idx];

    // dy_uz - dz_uy
    // dz_ux - dx_uz
    // dx_uy - dy_ux

    // Need neighbors
    float3 u_xp = u_aos[z * nx * ny + y * nx + xp];
    float3 u_xm = u_aos[z * nx * ny + y * nx + xm];
    float3 u_yp = u_aos[z * nx * ny + yp * nx + x];
    float3 u_ym = u_aos[z * nx * ny + ym * nx + x];
    float3 u_zp = u_aos[zp * nx * ny + y * nx + x];
    float3 u_zm = u_aos[zm * nx * ny + y * nx + x];

    float dy_ux = 0.5f * (u_yp.x - u_ym.x);
    float dz_ux = 0.5f * (u_zp.x - u_zm.x);

    float dx_uy = 0.5f * (u_xp.y - u_xm.y);
    float dz_uy = 0.5f * (u_zp.y - u_zm.y);

    float dx_uz = 0.5f * (u_xp.z - u_xm.z);
    float dy_uz = 0.5f * (u_yp.z - u_ym.z);

    omega[idx] = make_float3(dy_uz - dz_uy, dz_ux - dx_uz, dx_uy - dy_ux);
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 600
__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull =
        (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val +
                __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#else
__device__ double atomicAddDouble(double* address, double val) {
    return atomicAdd(address, val);
}
#endif

__global__ void ker_sum_force(float3* f_grid, double* sum_x, int N) {
    // Simple atomic sum for validation
    // Not efficient for large N but OK for diagnostics
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float fx = f_grid[i].x;
        if (abs(fx) > 1e-10) atomicAddDouble(sum_x, (double)fx);
    }
}

// ---- Main ----
int main(int argc, char** argv) {
    // Defaults
    int nx = 256, ny = 96, nz = 96;
    float R = 10.0f;
    float U0 = 0.05f;
    float tau = 0.8f;
    int steps = 30000;
    int output_every = 500;
    std::string out_dir = "out/sphere_move3d";

    // Parse
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--nx") nx = std::stoi(argv[++i]);
        if (arg == "--ny") ny = std::stoi(argv[++i]);
        if (arg == "--nz") nz = std::stoi(argv[++i]);
        if (arg == "--R") R = std::stof(argv[++i]);
        if (arg == "--U0") U0 = std::stof(argv[++i]);
        if (arg == "--tau") tau = std::stof(argv[++i]);
        if (arg == "--steps") steps = std::stoi(argv[++i]);
        if (arg == "--outputEvery") output_every = std::stoi(argv[++i]);
        if (arg == "--out") out_dir = argv[++i];
    }

    fs::create_directories(out_dir);

    // Physics
    float nu = (tau - 0.5f) / 3.0f;
    float Re = U0 * (2.0f * R) / nu;

    std::cout << ">>> Moving Sphere 3D (Service Based) <<<" << std::endl;
    std::cout << "Domain: " << nx << "x" << ny << "x" << nz << std::endl;
    std::cout << "Sphere: R=" << R << ", U0=" << U0 << ", Re=" << Re << std::endl;

    // LBM Setup
    lbm::LBMConfig lbm_cfg;
    lbm_cfg.nx = nx; lbm_cfg.ny = ny; lbm_cfg.nz = nz;
    lbm_cfg.tau = tau;
    lbm_cfg.enableFreeSurface = false;
    lbm_cfg.collisionModel = lbm::CollisionModel::SRT;
    lbm_cfg.bcXMin = lbm::BC_PERIODIC; lbm_cfg.bcXMax = lbm::BC_PERIODIC; // Scheme A: Large Periodic
    lbm_cfg.bcYMin = lbm::BC_PERIODIC; lbm_cfg.bcYMax = lbm::BC_PERIODIC;
    lbm_cfg.bcZMin = lbm::BC_PERIODIC; lbm_cfg.bcZMax = lbm::BC_PERIODIC;

    lbm::LBMCore lbm(lbm_cfg);
    lbm.initialize();

    // IBM Setup (Moving)
    // 1. Generate Local Sphere Points (Relative to center)
    std::vector<float3> rel_pos;
    std::vector<float> area;
    float dx = 1.0f;
    float spacing = 0.75f * dx;
    float marker_area = spacing * spacing;

    // Fibonacci Sphere Sampling
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

    float surface_area = 4.0f * (float)M_PI * R * R;
    // marker_area already defined
    int n_markers_target = (int)std::ceil(surface_area / marker_area);

    float golden_ratio = (1.0f + sqrtf(5.0f)) / 2.0f;
    float golden_angle = 2.0f * (float)M_PI * (1.0f - 1.0f / golden_ratio);

    for (int i = 0; i < n_markers_target; ++i) {
        float z = 1.0f - (2.0f * i + 1.0f) / n_markers_target;
        float r_slice = sqrtf(1.0f - z * z);

        float theta = golden_angle * i;

        float x = r_slice * cosf(theta);
        float y = r_slice * sinf(theta);

        // Relative position
        rel_pos.push_back(make_float3(x * R, y * R, z * R));
        area.push_back(marker_area);
    }
    std::cout << "Generated " << rel_pos.size() << " markers (Fibonacci) for R=" << R << std::endl;
    size_t nMarkers = rel_pos.size();

    ibm::IBMParams ibm_p;
    ibm_p.nx = nx; ibm_p.ny = ny; ibm_p.nz = nz;
    ibm_p.nMarkers = nMarkers;
    ibm_p.mdf_iterations = 5;
    ibm_p.mdf_beta = -0.5f; // Stability
    ibm_p.force_method = ibm::IBMForceMethod::DIRECT_FORCING;
    ibm::IBMCore ibm(ibm_p);

    // Initial State
    float3 center = make_float3(nx * 0.25f, ny * 0.5f, nz * 0.5f);
    float3 U_obj = make_float3(U0, 0.0f, 0.0f);

    // Buffers
    std::vector<float3> h_pos(nMarkers);
    std::vector<float3> h_vel(nMarkers); // Always U_obj
    for (size_t k = 0; k < nMarkers; ++k) h_vel[k] = U_obj;

    float3* d_force; CHECK_CUDA(cudaMalloc(&d_force, nx * ny * nz * sizeof(float3)));
    float* d_p; CHECK_CUDA(cudaMalloc(&d_p, nx * ny * nz * sizeof(float)));
    float3* d_omega; CHECK_CUDA(cudaMalloc(&d_omega, nx * ny * nz * sizeof(float3)));
    double* d_drag_sum; CHECK_CUDA(cudaMalloc(&d_drag_sum, sizeof(double)));

    std::ofstream log(out_dir + "/drag_time.csv");
    log << "step,drag_marker,drag_grid,mean_ux,max_slip,rho_min,rho_max,Re,Cx\n";

    // ---- SETUP SERVICES ----
    FieldStore fields;
    StepContext ctx;
    ctx.fields = &fields;

    // Config Services
    VTKService::Config vtk_cfg;
    vtk_cfg.output_dir = out_dir;
    vtk_cfg.nx = nx; vtk_cfg.ny = ny; vtk_cfg.nz = nz; vtk_cfg.dx = dx;
    vtk_cfg.interval = output_every;
    vtk_cfg.binary = true; // Use Binary
    vtk_cfg.fields = { "Density", "Pressure", "Speed", "Velocity", "Force", "Vorticity" };

    MarkerVTKService::Config m_cfg;
    m_cfg.output_dir = out_dir;
    m_cfg.interval = output_every;
    m_cfg.marker_field = "ibm.markers";
    m_cfg.binary = true;

    VTKService vtk_svc(vtk_cfg);
    MarkerVTKService marker_svc(m_cfg);

    vtk_svc.initialize(ctx);
    marker_svc.initialize(ctx);

    // Loop
    for (int t = 0; t <= steps; ++t) {
        ctx.step = t;
        ctx.time = t * 1.0;

        // 1. Move Sphere
        // If center.x > 0.8 * nx, stop moving but keep simulating flow (or wrap? User said stop or wrap)
        // Let's stop motion to avoid boundary crash, effectively becoming fixed sphere in moving fluid... wait, that changes physics.
        // User: "allows sphere to move only to x < 0.8*nx, then stop... continuing statistics"
        bool moving = true;
        if (center.x > 0.8f * nx) {
            moving = false;
            // Zero out velocity target
            U_obj = make_float3(0, 0, 0);
            for (auto& v : h_vel) v = U_obj;
        }
        else {
            center.x += U0; // dx=1, dt=1
        }

        // Update Marker Pos
        for (size_t k = 0; k < nMarkers; ++k) {
            h_pos[k].x = center.x + rel_pos[k].x;
            h_pos[k].y = center.y + rel_pos[k].y;
            h_pos[k].z = center.z + rel_pos[k].z;
        }

        // Upload (Force update)
        // Note: updateMarkers updates d_markers internally
        ibm.updateMarkers(h_pos.data(), h_vel.data(), area.data());

        // 2. Clear Force
        CHECK_CUDA(cudaMemset(d_force, 0, nx * ny * nz * sizeof(float3)));

        // 3. IBM Calc
        float3* u_aos = lbm.velocityAoSPtr();
        ibm.computeForces(u_aos, nullptr, d_force, 1.0f);

        // 4. LBM Step
        lbm.setExternalForceFromDeviceAoS(d_force);
        lbm.streamCollide();
        lbm.updateMacroscopic();

        // 5. Diagnostics & Output
        if (t % 100 == 0) {
            // Drag Grid
            CHECK_CUDA(cudaMemset(d_drag_sum, 0, sizeof(double)));
            int blocks = (nx * ny * nz + 255) / 256;
            ker_sum_force << <blocks, 256 >> > (d_force, d_drag_sum, nx * ny * nz);
            double drag_grid = 0;
            CHECK_CUDA(cudaMemcpy(&drag_grid, d_drag_sum, sizeof(double), cudaMemcpyDeviceToHost));
            drag_grid = -drag_grid; // Force on fluid -> Force on body

            // Drag Marker
            std::vector<float3> m_forces(nMarkers);
            ibm.downloadForces(m_forces.data());
            double drag_marker = 0;
            for (auto f : m_forces) drag_marker += f.x;
            drag_marker = -drag_marker;

            // Stats
            float* h_rho_gpu = const_cast<float*>(lbm.getDensityField()); // Should verify access
            // Need download for rho_min/max
            std::vector<float> cpu_rho_vec(nx * ny * nz);
            CHECK_CUDA(cudaMemcpy(cpu_rho_vec.data(), h_rho_gpu, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost));
            float rho_min = 1e9, rho_max = -1e9;
            for (float r : cpu_rho_vec) {
                if (r < rho_min) rho_min = r;
                if (r > rho_max) rho_max = r;
            }

            std::cout << "Step " << t << " Cx=" << center.x << " DragM=" << drag_marker
                << " DragG=" << drag_grid << " Re=" << Re << (moving ? " [MOV]" : " [STOP]") << std::endl;

            log << t << "," << drag_marker << "," << drag_grid << "," << 0.0 << ","
                << 0.0 << "," << rho_min << "," << rho_max << "," << Re << "," << center.x << "\n";

            // OUTPUT via SERVICES
            if (t > 0 && t % output_every == 0) {
                // 1. Compute Derived Fields on GPU
                dim3 b(8, 8, 8);
                dim3 g((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);
                kComputeFluidProps << <g, b >> > (u_aos, const_cast<float*>(lbm.getDensityField()), d_force, d_p, d_omega, nx, ny, nz, 1.0f / 3.0f);

                // 2. Download and populate FieldStore
                // Note: FieldStore creates shared_ptr arrays. We need to copy data in.

                // A. Fluid Fields
                auto h_rho = fields.create({ "Density", (size_t)nx * ny * nz, sizeof(float) });
                CHECK_CUDA(cudaMemcpy(h_rho.data(), lbm.getDensityField(), nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost));

                auto h_p = fields.create({ "Pressure", (size_t)nx * ny * nz, sizeof(float) });
                CHECK_CUDA(cudaMemcpy(h_p.data(), d_p, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost));

                auto h_f = fields.create({ "Force", (size_t)nx * ny * nz, sizeof(float) * 3 });
                CHECK_CUDA(cudaMemcpy(h_f.data(), d_force, nx * ny * nz * sizeof(float3), cudaMemcpyDeviceToHost));

                auto h_om = fields.create({ "Vorticity", (size_t)nx * ny * nz, sizeof(float) * 3 });
                CHECK_CUDA(cudaMemcpy(h_om.data(), d_omega, nx * ny * nz * sizeof(float3), cudaMemcpyDeviceToHost));

                // Velocity (Special name "velocity" for VTKService)
                // NOTE: VTKService expects SoA for "velocity" output block ([Ux...][Uy...][Uz...])?
                // Let's check VTKService.cpp:131 "vdata[i]=ux, vdata[N+i]=uy"
                // Yes, existing VTKService expects SoA.
                // BUT our `u_aos` is AoS.
                // We must transpose or modify VTKService to accept AoS.
                // OR passing "Velocity" as generic field works too, because I updated generic fields to handle float3 (AoS).
                // IF I use the name "velocity" (lowercase), it triggers the Special SoA path in VTKService.
                // IF I use "Velocity" (capital), it hits generic float3 path (AoS).
                // To avoid SoA conversion cost, I'll register it as "Velocity" (generic).
                // But wait, standard is usually lowercase "velocity".
                // Let's use "velocity" and convert to SoA just to be safe and compatible with existing service logic? 
                // Actually, in my modified VTKService.cpp, I added a velocity special case that does SoA->AoS conversion for binary write assuming input IS SoA.
                // Wait, LBM usually uses SoA on GPU? `u_aos` implies AoS.
                // If I pass AoS data to a service expecting SoA, it will be garbled.
                // Current `VTKService` logic for "velocity" field assumes SoA input layout (N+i offsets).
                // So I should provide SoA data OR change the service.
                // Changing service to adapt to data shape is hard (FieldDesc doesn't say layout).
                // Easier: Pass "Velocity" (Capital V) to avoid the hardcoded "velocity" check, and rely on Generic float3 path which writes xyz xyz.
                // Let's do that.
                auto h_u = fields.create({ "Velocity", (size_t)nx * ny * nz, sizeof(float) * 3 });
                CHECK_CUDA(cudaMemcpy(h_u.data(), u_aos, nx * ny * nz * sizeof(float3), cudaMemcpyDeviceToHost));

                // Speed
                std::vector<float> speed(nx * ny * nz);
                float3* cpu_u_ptr = (float3*)h_u.data();
                for (int i = 0; i < nx * ny * nz; ++i) speed[i] = sqrtf(cpu_u_ptr[i].x * cpu_u_ptr[i].x + cpu_u_ptr[i].y * cpu_u_ptr[i].y + cpu_u_ptr[i].z * cpu_u_ptr[i].z);
                auto h_speed = fields.create({ "Speed", (size_t)nx * ny * nz, sizeof(float) });
                memcpy(h_speed.data(), speed.data(), speed.size() * sizeof(float));

                // B. Marker Fields
                auto h_mk = fields.create({ "ibm.markers", nMarkers, sizeof(float) * 3 });
                memcpy(h_mk.data(), h_pos.data(), nMarkers * sizeof(float) * 3);

                auto h_mv = fields.create({ "ibm.velocity", nMarkers, sizeof(float) * 3 });
                memcpy(h_mv.data(), h_vel.data(), nMarkers * sizeof(float) * 3);

                auto h_mf = fields.create({ "ibm.force", nMarkers, sizeof(float) * 3 });
                // Need force from IBM forces (vector<float3>)
                memcpy(h_mf.data(), m_forces.data(), nMarkers * sizeof(float) * 3);

                auto h_ma = fields.create({ "ibm.area", nMarkers, sizeof(float) });
                memcpy(h_ma.data(), area.data(), nMarkers * sizeof(float));

                // CALL SERVICES
                vtk_svc.onStepEnd(ctx);
                marker_svc.onStepEnd(ctx);
            }
        }
    }

    // Cleanup
    vtk_svc.finalize(ctx);
    marker_svc.finalize(ctx);

    return 0;
}
