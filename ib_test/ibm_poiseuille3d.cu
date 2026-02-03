#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iomanip>

#include <cuda_runtime.h>
#include "../src/physics/lbm/LBMCore.hpp"
#include "../src/physics/ibm/IBMCore.hpp"
#include "../src/physics/lbm/LBMConfig.hpp"
#include "../src/geometry/VectorTypes.h" 

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// Write profile to CSV
void write_profile(const std::string& filename, const std::vector<double>& y_coords, 
                  const std::vector<double>& u_mean, const std::vector<double>& u_exact) {
    std::ofstream out(filename);
    out << "y,u_mean,u_exact,abs_error,rel_error\n";
    for(size_t i=0; i<y_coords.size(); ++i) {
        double abs_err = std::abs(u_mean[i] - u_exact[i]);
        double rel_err = (std::abs(u_exact[i]) > 1e-6) ? (abs_err / std::abs(u_exact[i])) : 0.0;
        out << y_coords[i] << "," << u_mean[i] << "," << u_exact[i] << "," 
            << abs_err << "," << rel_err << "\n";
    }
    out.close();
    std::cout << "Profile written to " << filename << std::endl;
}

// ==========================================
// CONFIGURATION SCHEMES
// ==========================================
// Select ONE scheme:
// 1: Full-Gap Poiseuille (Walls at boundaries, Global Gravity) - CLEANEST
// 2: Gap-Only Driving (Internal Walls, Gravity only in gap) - COMPATIBLE
#define SCHEME 2

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

// Helper kernel to sum grid forces
__global__ void ker_sum_force(const float3* force_field, double* sum_x, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(sum_x, (double)force_field[idx].x);
    }
}

__global__ void ker_apply_gap_gravity(float3* force_field, const float* rho_field, 
                                      float gx, int nx, int ny, int nz, 
                                      float y_bot, float y_top) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx*ny*nz) return;
    
    int i = idx % nx;
    int j = (idx / nx) % ny;
    int k = (idx / (nx*ny));
    
    // Check if in gap
    float y = (float)j; // Node coordinate (0.5 offset handled in geometry conceptually)
    // Gap definition: Strictly between walls
    // Wall positions are floating point. 
    bool in_gap = (y > y_bot && y < y_top);

    if (in_gap) {
        // Add gravity force density: f = rho * g
        // LBM Core expects 'force' input. 
        // Note: If LBM interprets input as acceleration, we pass g. 
        // If it interprets as Force Density, we pass rho*g.
        // Based on internal code analysis, standard LBM usually takes Acceleration if divided by rho, 
        // or Force Density if rho~1. 
        // Here we apply to the same array as IBM force. 
        // IBM Force is typically Force Density (F/dV). 
        // So we should add Body Force Density: rho * gx.
        float rho = rho_field ? rho_field[idx] : 1.0f;
        force_field[idx].x += rho * gx;
    }
}


int main(int argc, char** argv) {
    // ==========================================
    // 1. Configuration
    // ==========================================
    int nx = 64;
    int ny = 32;
    int nz = 32;
    float tau = 0.8f; 
    float nu = (tau - 0.5f) / 3.0f;
    float dx = 1.0f;
    
    // Scheme Setup
#if SCHEME == 1
    std::cout << "SCHEME 1: Full-Gap Poiseuille (Global Gravity)" << std::endl;
    float y_bot = 0.5f * dx;
    float y_top = (float)ny * dx - 0.5f * dx;
#else
    std::cout << "SCHEME 2: Gap-Only Driving (Internal Walls)" << std::endl;
    float y_bot = 4.0f * dx;
    float y_top = (float)ny * dx - 5.0f * dx; // 27.0
#endif
    
    float H = y_top - y_bot;
    
    // Physics
    float target_Umax = 0.05f;
    float gx = (target_Umax * 8.0f * nu) / (H * H);
    
    // Empirical Correction (Keep from previous step)
    gx = -gx; // Negative gravity for Positive Velocity in this solver
    
    int steps = 20000; 
    int diag_interval = 2000;

    std::cout << "Params: Tau=" << tau << ", gx=" << gx << std::endl;

    // LBM Config
    lbm::LBMConfig lbm_cfg;
    lbm_cfg.nx = nx; lbm_cfg.ny = ny; lbm_cfg.nz = nz;
    lbm_cfg.tau = tau;
    lbm_cfg.enableFreeSurface = false;
    lbm_cfg.collisionModel = lbm::CollisionModel::SRT;
    lbm_cfg.bcXMin = lbm::BC_PERIODIC; lbm_cfg.bcXMax = lbm::BC_PERIODIC;
    lbm_cfg.bcZMin = lbm::BC_PERIODIC; lbm_cfg.bcZMax = lbm::BC_PERIODIC;
    
#if SCHEME == 1
    // Scheme 1: Periodic Y to emulate infinite channel or Neutral walls
    // If Periodic, top connects to bottom. 
    // We place walls at 0.5 (bot) and ny-0.5 (top).
    // Effectively they conform the boundary.
    lbm_cfg.wallFlags = 0; // No LBM walls
    lbm_cfg.bcYMin = lbm::BC_PERIODIC; 
    lbm_cfg.bcYMax = lbm::BC_PERIODIC;
    
    // Global Gravity
    lbm_cfg.gravity = make_float3(gx, 0.0f, 0.0f);
#else
    // Scheme 2: Internal Walls. BounceBack at Domain Boundary.
    lbm_cfg.wallFlags = lbm::WALL_Y_MIN | lbm::WALL_Y_MAX;
    lbm_cfg.bcYMin = lbm::BC_BOUNCE_BACK; 
    lbm_cfg.bcYMax = lbm::BC_BOUNCE_BACK;
    
    // NO Global Gravity (We apply manually in gap)
    lbm_cfg.gravity = make_float3(0.0f, 0.0f, 0.0f);
#endif

    lbm_cfg.rho0 = 1.0f;
    lbm_cfg.u0 = make_float3(0.0f, 0.0f, 0.0f);

    lbm::LBMCore lbm(lbm_cfg);
    lbm.initialize();

    // IBM Setup
    std::vector<float3> pos;
    std::vector<float3> vel;
    std::vector<float> area;
    
    float spacing = 0.75f * dx; 
    float marker_area = (spacing * spacing);
    int nMx = (int)(nx * dx / spacing);
    int nMz = (int)(nz * dx / spacing);
    
    // Generate Markers
    auto add_plane = [&](float y_pos) {
        for(int i=0; i<nMx; ++i) {
            for(int k=0; k<nMz; ++k) {
                float x = (i + 0.5f) * spacing;
                float z = (k + 0.5f) * spacing;
                pos.push_back(make_float3(x, y_pos, z));
                vel.push_back(make_float3(0.0f, 0.0f, 0.0f));
                area.push_back(marker_area);
            }
        }
    };
    
    add_plane(y_bot);
    add_plane(y_top);
    
    size_t nMarkers = pos.size();
    std::cout << "Generated " << nMarkers << " markers." << std::endl;
    
    ibm::IBMParams ibm_p;
    ibm_p.nx = nx; ibm_p.ny = ny; ibm_p.nz = nz;
    ibm_p.dx = 1.0f;
    ibm_p.nMarkers = nMarkers;
    ibm_p.mdf_iterations = 3; 
    ibm_p.mdf_beta = -0.1f; 
    ibm_p.force_method = ibm::IBMForceMethod::DIRECT_FORCING;

    ibm::IBMCore ibm(ibm_p);
    ibm.updateMarkers(pos.data(), vel.data(), area.data());

    float3* d_force; // Total External Force (IBM + Gravity if manual)
    CHECK_CUDA(cudaMalloc(&d_force, nx*ny*nz * sizeof(float3)));
    
    double* d_grid_force_sum;
    CHECK_CUDA(cudaMalloc(&d_grid_force_sum, sizeof(double)));

    // Simulation Loop
    for(int t=0; t<=steps; ++t) {
        float3* u_aos = lbm.velocityAoSPtr();
        const float* rho_field = lbm.getDensityField();
        
        // Zero force buffer
        CHECK_CUDA(cudaMemset(d_force, 0, nx*ny*nz * sizeof(float3)));
        
        // 1. Compute IBM Forces (fills d_force)
        ibm.computeForces(u_aos, nullptr, d_force, 1.0f);
        
#if SCHEME == 2
        // 2. Add Gap Gravity Manualy
        {
            int threads = 256;
            int blocks = (nx*ny*nz + threads - 1) / threads;
            ker_apply_gap_gravity<<<blocks, threads>>>(d_force, rho_field, gx, nx, ny, nz, y_bot, y_top);
        }
#endif
        
        // 3. Apply Total Force to LBM
        lbm.setExternalForceFromDeviceAoS(d_force);
        
        lbm.streamCollide();
        lbm.updateMacroscopic();
        
        if (t % diag_interval == 0 || t == steps) {
            // Diagnostics
            
            // A. Grid IB Force Sum (Only IB part? No, d_force contains both now in Scheme 2)
            // Need to separate? 
            // Better: Re-calculate IBM part only for diagnostics, OR assume we want Total Grid Force Balance = 0?
            // User requested: "F_grid_ib_x = Sum(f_ib * dV)". 
            // In Scheme 2, d_force has gravity added. 
            // To get pure IB force, we should sum d_force BEFORE adding gravity.
            // But we already added it. 
            // Workaround: We know Total = IBM + Gravity. 
            // Sum(Total) should be 0 (Steady State). 
            // Let's check Sum(Total).
            
            CHECK_CUDA(cudaMemset(d_grid_force_sum, 0, sizeof(double)));
            int blocks = (nx*ny*nz + 255)/256;
            ker_sum_force<<<blocks, 256>>>(d_force, d_grid_force_sum, nx*ny*nz);
            double f_total_grid_x = 0;
            CHECK_CUDA(cudaMemcpy(&f_total_grid_x, d_grid_force_sum, sizeof(double), cudaMemcpyDeviceToHost));
            
            // B. Marker Force Sum
            std::vector<float3> m_forces(nMarkers);
            ibm.downloadForces(m_forces.data());
            double f_marker_sum_x = 0.0;
            for(const auto& f : m_forces) f_marker_sum_x += f.x;
            
            // C. Body Force Calculation (Analytical/Field based)
            std::vector<float> h_rho(nx*ny*nz);
            CHECK_CUDA(cudaMemcpy(h_rho.data(), lbm.getDensityField(), nx*ny*nz*sizeof(float), cudaMemcpyDeviceToHost));
            
            double f_body_all = 0.0;
            double f_body_gap = 0.0;
            
            for(int k=0; k<nz; ++k) {
                for(int j=0; j<ny; ++j) {
                    for(int i=0; i<nx; ++i) {
                         double f_cell = h_rho[k*nx*ny + j*nx + i] * gx;
                         f_body_all += f_cell;
                         if (j > y_bot && j < y_top) {
                             f_body_gap += f_cell;
                         }
                    }
                }
            }
            
            // D. Derived IB Grid Force
            // In Scheme 2: F_total_grid = F_IBM_grid + F_body_gap (applied manuall)
            // -> F_IBM_grid = F_total_grid - F_body_gap
            // In Scheme 1: F_total_grid = F_IBM_grid (Gravity applied internally by LBM)
            // -> F_IBM_grid = F_total_grid
            
            double f_ibm_grid_x = 0.0;
#if SCHEME == 2
            f_ibm_grid_x = f_total_grid_x - f_body_gap;
#else
            f_ibm_grid_x = f_total_grid_x;
#endif
            
            // E. Balance Error
            // Scheme 1: Balance against ALL
            double err_all = std::abs(f_ibm_grid_x + f_body_all) / std::abs(f_body_all);
            // Scheme 2: Balance against GAP
            double err_gap = std::abs(f_ibm_grid_x + f_body_gap) / std::abs(f_body_gap);

            std::cout << "Step " << t << " Force Balance:" << std::endl;
            std::cout << "  F_body_all     = " << f_body_all << std::endl;
            std::cout << "  F_body_gap     = " << f_body_gap << std::endl;
            std::cout << "  F_marker_sum_x = " << f_marker_sum_x << std::endl;
            std::cout << "  F_grid_ib_x    = " << f_ibm_grid_x << " (Should match MarkerSum)" << std::endl;
            
#if SCHEME == 2
            std::cout << "  Target: GAP Driving" << std::endl;
            std::cout << "  Balance(Gap)   = " << err_gap * 100.0 << " %" << std::endl;
#else
            std::cout << "  Target: ALL Driving" << std::endl;
            std::cout << "  Balance(All)   = " << err_all * 100.0 << " %" << std::endl;
#endif
            
            // Velocity check
             std::vector<float3> h_u(nx*ny*nz);
            CHECK_CUDA(cudaMemcpy(h_u.data(), u_aos, nx*ny*nz*sizeof(float3), cudaMemcpyDeviceToHost));
            double u_mid_sum = 0; int mid_cnt=0;
            int slice_y = ny/2;
            for(int k=0; k<nz; ++k) for(int i=0; i<nx; ++i) {
                u_mid_sum += h_u[k*nx*ny + slice_y*nx + i].x;
                mid_cnt++;
            }
            std::cout << "  U_mid = " << u_mid_sum/mid_cnt << std::endl;
        }
    }

    // ... (Verification part remains similar) ...
     // Download final LBM velocity
    std::vector<float3> h_u(nx*ny*nz);
    CHECK_CUDA(cudaMemcpy(h_u.data(), lbm.velocityAoSPtr(), nx*ny*nz*sizeof(float3), cudaMemcpyDeviceToHost));
    
    std::vector<double> u_mean_profile(ny, 0.0);
    std::vector<double> y_coords(ny, 0.0);
    std::vector<double> u_exact_profile(ny, 0.0);
    
    double max_abs_error = 0.0;
    double computed_Umax = 0.0;
    
    for(int j=0; j<ny; ++j) {
        double sum_u = 0.0;
        int count = 0;
        for(int k=0; k<nz; ++k) {
            for(int i=0; i<nx; ++i) {
                sum_u += h_u[k*(nx*ny) + j*nx + i].x;
                count++;
            }
        }
        u_mean_profile[j] = sum_u / count;
        y_coords[j] = (double)j;
        
        bool in_gap = (j >= y_bot + 1.5 && j <= y_top - 1.5);
        
        if (in_gap) {
            double y_eff = j - y_bot;
            double u_exact = (std::abs(gx) / (2.0 * nu)) * y_eff * (H - y_eff);
            u_exact_profile[j] = u_exact;
            
            double abs_err = std::abs(u_mean_profile[j] - u_exact);
            if(abs_err > max_abs_error) max_abs_error = abs_err;
            if(u_mean_profile[j] > computed_Umax) computed_Umax = u_mean_profile[j];
        }
    }
    
    write_profile("poiseuille_profile.csv", y_coords, u_mean_profile, u_exact_profile);
    
    std::cout << "MaxAbsError: " << max_abs_error << " Umax: " << computed_Umax << std::endl;
    
    if (max_abs_error < 0.01) {
        std::cout << "TEST PASSED" << std::endl;
        return 0;
    } else {
        std::cout << "TEST FAILED" << std::endl;
        return 1;
    }
}
