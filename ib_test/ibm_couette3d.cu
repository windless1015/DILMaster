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

void write_profile(const std::string& filename, const std::vector<double>& y_coords, 
                  const std::vector<double>& u_mean, const std::vector<double>& u_exact) {
    std::ofstream out(filename);
    out << "y,u_mean,u_exact,error\n";
    for(size_t i=0; i<y_coords.size(); ++i) {
        out << y_coords[i] << "," << u_mean[i] << "," << u_exact[i] << "," << std::abs(u_mean[i] - u_exact[i]) << "\n";
    }
    out.close();
    std::cout << "Profile written to " << filename << std::endl;
}

int main(int argc, char** argv) {
    // 1. Configuration
    int nx = 64;
    int ny = 32;
    int nz = 32;
    float tau = 1.0f; 
    float U0 = 0.05f; // Restore user target
    int steps = 20000; 
    int diag_interval = 2000;

    lbm::LBMConfig lbm_cfg;
    lbm_cfg.nx = nx; lbm_cfg.ny = ny; lbm_cfg.nz = nz;
    lbm_cfg.tau = tau;
    lbm_cfg.enableFreeSurface = false;
    lbm_cfg.collisionModel = lbm::CollisionModel::SRT;
    lbm_cfg.wallFlags = lbm::WALL_Y_MIN | lbm::WALL_Y_MAX;
    lbm_cfg.bcXMin = lbm::BC_PERIODIC; lbm_cfg.bcXMax = lbm::BC_PERIODIC;
    lbm_cfg.bcYMin = lbm::BC_BOUNCE_BACK; lbm_cfg.bcYMax = lbm::BC_BOUNCE_BACK;
    lbm_cfg.bcZMin = lbm::BC_PERIODIC; lbm_cfg.bcZMax = lbm::BC_PERIODIC;
    lbm_cfg.rho0 = 1.0f;
    lbm_cfg.u0 = make_float3(0.0f, 0.0f, 0.0f);

    std::cout << "Initializing LBM Couette 3D..." << std::endl;
    std::cout << "Domain: " << nx << "x" << ny << "x" << nz << std::endl;
    std::cout << "Params: Tau=" << tau << ", U0=" << U0 << std::endl;

    // 2. Setup Solvers
    lbm::LBMCore lbm(lbm_cfg);
    lbm.initialize();

    float dx = 1.0f;
    float spacing = 0.75f * dx; 
    float y_bot = 4.0f * dx;
    float y_top = (float)ny * dx - 5.0f * dx; // 27.0
    float H = y_top - y_bot;
    
    std::vector<float3> pos;
    std::vector<float3> vel;
    std::vector<float> area;
    
    int nMx = (int)(nx * dx / spacing);
    int nMz = (int)(nz * dx / spacing);
    float marker_area = (spacing * spacing);
    
    // Bottom Markers (Static)
    for(int i=0; i<nMx; ++i) {
        for(int k=0; k<nMz; ++k) {
            float x = (i + 0.5f) * spacing;
            float z = (k + 0.5f) * spacing;
            pos.push_back(make_float3(x, y_bot, z));
            vel.push_back(make_float3(0.0f, 0.0f, 0.0f));
            area.push_back(marker_area);
        }
    }
    
    // Top Markers (Moving U0)
    for(int i=0; i<nMx; ++i) {
        for(int k=0; k<nMz; ++k) {
            float x = (i + 0.5f) * spacing;
            float z = (k + 0.5f) * spacing;
            pos.push_back(make_float3(x, y_top, z));
            vel.push_back(make_float3(U0, 0.0f, 0.0f));
            area.push_back(marker_area);
        }
    }
    
    size_t nMarkers = pos.size();
    std::cout << "Generated " << nMarkers << " markers." << std::endl;
    
    ibm::IBMParams ibm_p;
    ibm_p.nx = nx; ibm_p.ny = ny; ibm_p.nz = nz;
    ibm_p.dx = 1.0f;
    ibm_p.nMarkers = nMarkers;
    ibm_p.mdf_iterations = 3; 
    ibm_p.mdf_beta = -0.1f; // Negative beta confirmed working
    ibm_p.force_method = ibm::IBMForceMethod::DIRECT_FORCING;

    ibm::IBMCore ibm(ibm_p);
    ibm.updateMarkers(pos.data(), vel.data(), area.data());

    float3* d_force;
    CHECK_CUDA(cudaMalloc(&d_force, nx*ny*nz * sizeof(float3)));
    CHECK_CUDA(cudaMemset(d_force, 0, nx*ny*nz * sizeof(float3)));

    // 3. Simulation Loop
    for(int t=0; t<=steps; ++t) {
        float3* u_aos = lbm.velocityAoSPtr();
        ibm.computeForces(u_aos, nullptr, d_force, 1.0f);
        // Note: d_force computed with negative beta
        lbm.setExternalForceFromDeviceAoS(d_force);
        lbm.streamCollide();
        lbm.updateMacroscopic();
        
        
        if (t % diag_interval == 0 || t == steps) {
            // Check Slip
#ifdef IBM_TESTING
            std::vector<float3> u_interp(nMarkers);
            ibm.downloadInterpolatedVelocity(u_interp.data());
            
            double max_slip = 0.0;
            for(size_t i=0; i<nMarkers; ++i) {
                float dx = u_interp[i].x - vel[i].x;
                float dy = u_interp[i].y - vel[i].y;
                float dz = u_interp[i].z - vel[i].z;
                float slip = std::sqrt(dx*dx + dy*dy + dz*dz);
                if(slip > max_slip) max_slip = slip;
            }
            
            std::vector<float3> m_forces(nMarkers);
            ibm.downloadForces(m_forces.data());
            double fx_bot = 0, fx_top = 0;
            int half = nMarkers/2;
            for(int i=0; i<half; ++i) fx_bot += m_forces[i].x;
            for(int i=half; i<nMarkers; ++i) fx_top += m_forces[i].x;

            // Check Mid Velocity
            std::vector<float3> h_u_chk(nx*ny*nz);
            CHECK_CUDA(cudaMemcpy(h_u_chk.data(), lbm.velocityAoSPtr(), nx*ny*nz*sizeof(float3), cudaMemcpyDeviceToHost));
            double u_mid_sum = 0;
            int slice_y = ny/2;
            int count_mid = 0;
            for(int k=0; k<nz; ++k) {
                for(int i=0; i<nx; ++i) {
                     u_mid_sum += h_u_chk[k*(nx*ny) + slice_y*nx + i].x;
                     count_mid++;
                }
            }
            double u_mid_avg = u_mid_sum / count_mid;

            // Check Rho
            std::vector<float> h_rho(nx*ny*nz);
            CHECK_CUDA(cudaMemcpy(h_rho.data(), lbm.getDensityField(), nx*ny*nz*sizeof(float), cudaMemcpyDeviceToHost));
            double rho_min = 1e9, rho_max = -1e9;
            for(float r : h_rho) {
                if(r < rho_min) rho_min = r;
                if(r > rho_max) rho_max = r;
            }

            std::cout << "Step " << t << ": MaxSlip=" << max_slip 
                      << " Fx_Bot=" << fx_bot << " Fx_Top=" << fx_top 
                      << " U_mid(" << slice_y << ")=" << u_mid_avg 
                      << " Rho[" << rho_min << ", " << rho_max << "]" << std::endl;
#else
            std::cout << "Step " << t << " (IBM_TESTING disabled)" << std::endl;
#endif
        }
    }
    
    // 4. Verification
    std::vector<float3> h_u(nx*ny*nz);
    CHECK_CUDA(cudaMemcpy(h_u.data(), lbm.velocityAoSPtr(), nx*ny*nz*sizeof(float3), cudaMemcpyDeviceToHost));
    
    std::vector<double> u_mean_profile(ny, 0.0);
    std::vector<double> y_coords(ny, 0.0);
    std::vector<double> u_exact_profile(ny, 0.0);
    
    double max_error = 0.0;
    double l2_error = 0.0;
    
    for(int j=0; j<ny; ++j) {
        double sum_u = 0.0;
        int count = 0;
        for(int k=0; k<nz; ++k) {
            for(int i=0; i<nx; ++i) {
                int idx = k*(nx*ny) + j*nx + i;
                sum_u += h_u[idx].x;
                count++;
            }
        }
        u_mean_profile[j] = sum_u / count;
        
        float grid_y = (float)j;
        y_coords[j] = grid_y;
        
        bool in_gap = (grid_y >= y_bot + 1.5f && grid_y <= y_top - 1.5f);
        
        if (in_gap) {
            double u_exact = U0 * (grid_y - y_bot) / H;
            u_exact_profile[j] = u_exact;
            double err = std::abs(u_mean_profile[j] - u_exact);
            if(err > max_error) max_error = err;
            l2_error += err*err;
        } else {
             u_exact_profile[j] = 0.0; 
        }
    }
    
    int valid_j = 0;
    for(int j=0; j<ny; ++j) {
        if (j >= y_bot + 1.5f && j <= y_top - 1.5f) valid_j++;
    }
    
    l2_error = std::sqrt(l2_error / valid_j); 
    
    std::cout << "Verification Result (Nodes " << y_bot+1.5 << " to " << y_top-1.5 << "):" << std::endl;
    std::cout << "Max Error (Main Gap): " << max_error << std::endl;
    std::cout << "L2 Error (Main Gap): " << l2_error << std::endl;
    
    write_profile("couette_profile.csv", y_coords, u_mean_profile, u_exact_profile);
    cudaFree(d_force);
    
    if (l2_error < 0.005 && max_error < 0.01) {
        std::cout << "TEST PASSED" << std::endl;
        return 0;
    } else {
        std::cout << "TEST FAILED (Error too high)" << std::endl;
        return 1;
    }
}
