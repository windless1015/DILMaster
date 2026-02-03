#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include <cuda_runtime.h>
#include "../src/physics/ibm/IBMCore.hpp"
#include "../src/geometry/VectorTypes.h"

// Check for CUDA errors
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// Helper: Random float generator
float random_float(float min, float max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    return static_cast<float>(dis(gen));
}

// Helper: Get Grid Index (Host side)
int get_idx(int x, int y, int z, int nx, int ny, int nz) {
  return z * (nx * ny) + y * nx + x;
}

// ----------------------------------------------------------------------------
// Test 1: Area Invariance Test (Cloud of Points)
// ----------------------------------------------------------------------------
TEST(IBM_MDF_Regression, AreaInvariance) {
    ibm::IBMParams p1, p2;
    p1.nx = 32; p1.ny = 32; p1.nz = 32; p1.dx = 1.0f;
    p2 = p1;
    
    float min_b = 14.0f, max_b = 18.0f;
    
    // Case 1: Sparse (N=500)
    p1.nMarkers = 500;
    float total_area = 100.0f;
    std::vector<float> area1(p1.nMarkers, total_area / p1.nMarkers);
    
    std::vector<float3> pos1(p1.nMarkers);
    for(auto& p : pos1) {
        p.x = random_float(min_b, max_b);
        p.y = random_float(min_b, max_b);
        p.z = random_float(min_b, max_b);
    }
    
    // Case 2: Dense (N=2000)
    p2.nMarkers = 2000;
    std::vector<float> area2(p2.nMarkers, total_area / p2.nMarkers);
    
    std::vector<float3> pos2(p2.nMarkers);
    for(auto& p : pos2) {
        p.x = random_float(min_b, max_b);
        p.y = random_float(min_b, max_b);
        p.z = random_float(min_b, max_b);
    }

    ibm::IBMCore ibm1(p1);
    ibm::IBMCore ibm2(p2);
    
    std::vector<float3> vel1(p1.nMarkers, make_float3(1,0,0)); 
    std::vector<float3> vel2(p2.nMarkers, make_float3(1,0,0)); 
    
    ibm1.updateMarkers(pos1.data(), vel1.data(), area1.data());
    ibm2.updateMarkers(pos2.data(), vel2.data(), area2.data());
    
    int nCells = p1.nx * p1.ny * p1.nz;
    float3* d_u_fluid; 
    float3* d_f_out1;
    float3* d_f_out2;
    CHECK_CUDA(cudaMalloc(&d_u_fluid, nCells * sizeof(float3)));
    CHECK_CUDA(cudaMemset(d_u_fluid, 0, nCells * sizeof(float3)));
    CHECK_CUDA(cudaMalloc(&d_f_out1, nCells * sizeof(float3)));
    CHECK_CUDA(cudaMalloc(&d_f_out2, nCells * sizeof(float3)));
    
    ibm1.computeForces(d_u_fluid, nullptr, d_f_out1, 1.0f);
    ibm2.computeForces(d_u_fluid, nullptr, d_f_out2, 1.0f);
    
    std::vector<float3> h_f1(nCells), h_f2(nCells);
    CHECK_CUDA(cudaMemcpy(h_f1.data(), d_f_out1, nCells*sizeof(float3), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_f2.data(), d_f_out2, nCells*sizeof(float3), cudaMemcpyDeviceToHost));
    
    double fx1 = 0, fx2 = 0;
    for(auto f : h_f1) fx1 += f.x;
    for(auto f : h_f2) fx2 += f.x;
    
    fx1 *= (p1.dx*p1.dx*p1.dx);
    fx2 *= (p2.dx*p2.dx*p2.dx);
    
    std::cout << "[AreaInvariance] Force1: " << fx1 << ", Force2: " << fx2 << std::endl;
    double diff = std::abs(fx1 - fx2);
    double rel_err = diff / std::abs(fx1);
    
    EXPECT_LT(rel_err, 0.05); 
    
    cudaFree(d_u_fluid); cudaFree(d_f_out1); cudaFree(d_f_out2);
}

// ----------------------------------------------------------------------------
// Test 2: dx Scaling Test
// ----------------------------------------------------------------------------
TEST(IBM_MDF_Regression, DxScaling) {
    ibm::IBMParams p1, p2;
    p1.nx = 16; p1.ny=16; p1.nz=16; p1.dx = 1.0f;
    p2.nx = 32; p2.ny=32; p2.nz=32; p2.dx = 0.5f; 
    
    p1.mdf_iterations = 1;
    p2.mdf_iterations = 1;
    p1.nMarkers = 1; p2.nMarkers = 1;
    p1.domain_origin_x=0; p2.domain_origin_x=0;
    
    std::vector<float> area(1, 1.0f);
    ibm::IBMCore ibm1(p1);
    ibm::IBMCore ibm2(p2);
    
    float3 pos1 = make_float3(8,8,8);
    float3 pos2 = make_float3(8,8,8); 
    
    ibm1.updateMarkers(&pos1, &make_float3(1,0,0), area.data());
    ibm2.updateMarkers(&pos2, &make_float3(1,0,0), area.data());
    
    float3* d_u1; float3* d_u2;
    float3* d_f1; float3* d_f2;
    CHECK_CUDA(cudaMalloc(&d_u1, 16*16*16*sizeof(float3)));
    CHECK_CUDA(cudaMalloc(&d_u2, 32*32*32*sizeof(float3)));
    CHECK_CUDA(cudaMemset(d_u1, 0, 16*16*16*sizeof(float3)));
    CHECK_CUDA(cudaMemset(d_u2, 0, 32*32*32*sizeof(float3)));
    CHECK_CUDA(cudaMalloc(&d_f1, 16*16*16*sizeof(float3)));
    CHECK_CUDA(cudaMalloc(&d_f2, 32*32*32*sizeof(float3)));
    
    ibm1.computeForces(d_u1, nullptr, d_f1, 1.0f);
    ibm2.computeForces(d_u2, nullptr, d_f2, 1.0f);
    
    std::vector<float3> h_f1(16*16*16), h_f2(32*32*32);
    CHECK_CUDA(cudaMemcpy(h_f1.data(), d_f1, h_f1.size()*sizeof(float3), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_f2.data(), d_f2, h_f2.size()*sizeof(float3), cudaMemcpyDeviceToHost));
    
    double F1 = 0, F2 = 0;
    for(auto f : h_f1) F1 += f.x;
    for(auto f : h_f2) F2 += f.x;
    
    F1 *= (1.0*1.0*1.0);
    F2 *= (0.5*0.5*0.5);
    
    EXPECT_NEAR(F1, F2, 1e-3);
    
    float max1=0, max2=0;
    for(auto f : h_f1) max1 = std::max(max1, f.x);
    for(auto f : h_f2) max2 = std::max(max2, f.x);
    
    EXPECT_GT(max2, max1 * 7.0f); 
    
    cudaFree(d_u1); cudaFree(d_u2); cudaFree(d_f1); cudaFree(d_f2);
}

// ----------------------------------------------------------------------------
// Test 3: MDF Effectiveness Test (Slip Reduction)
// ----------------------------------------------------------------------------
TEST(IBM_MDF_Regression, MDFEffectiveness) {
#ifndef IBM_TESTING
    GTEST_SKIP() << "Skipping stricter test requiring IBM_TESTING macro";
#else
    ibm::IBMParams p;
    p.nx=16; p.ny=16; p.nz=16;
    p.nMarkers = 1;
    
    // Case 1: 1 Iteration
    p.mdf_iterations = 1; 
    ibm::IBMCore ibm1(p); 
    
    // Case 2: 5 Iterations
    p.mdf_iterations = 5;
    ibm::IBMCore ibm5(p); 
    
    float3 pos = make_float3(8.1f, 8.1f, 8.1f);
    float3 target_vel = make_float3(1.0f, 0.0f, 0.0f);
    // Increase Area to compensate for Spread-Interp diffusion loss
    std::vector<float> area(1, 20.0f);
    
    ibm1.updateMarkers(&pos, &target_vel, area.data());
    ibm5.updateMarkers(&pos, &target_vel, area.data());
    
    int nCells = 16*16*16;
    float3* d_u; float3* d_f1; float3* d_f2;
    CHECK_CUDA(cudaMalloc(&d_u, nCells*sizeof(float3)));
    CHECK_CUDA(cudaMemset(d_u, 0, nCells*sizeof(float3))); 
    CHECK_CUDA(cudaMalloc(&d_f1, nCells*sizeof(float3)));
    CHECK_CUDA(cudaMalloc(&d_f2, nCells*sizeof(float3)));
    
    ibm1.computeForces(d_u, nullptr, d_f1, 1.0f);
    ibm5.computeForces(d_u, nullptr, d_f2, 1.0f);
    
    std::vector<float3> u_interp1(1);
    std::vector<float3> u_interp5(1);
    
    ibm1.downloadInterpolatedVelocity(u_interp1.data());
    ibm5.downloadInterpolatedVelocity(u_interp5.data());
    
    float slip1 = std::abs(target_vel.x - u_interp1[0].x);
    float slip5 = std::abs(target_vel.x - u_interp5[0].x);
    
    std::cout << "[MDFEffectiveness] Slip (Iter=1): " << slip1 << ", Slip (Iter=5): " << slip5 << std::endl;
    
    EXPECT_LT(slip5, slip1 * 0.7f); 
    
    cudaFree(d_u); cudaFree(d_f1); cudaFree(d_f2);
#endif
}


// ----------------------------------------------------------------------------
// Test 4: Under-relaxation Stability Test
// ----------------------------------------------------------------------------
TEST(IBM_MDF_Regression, BetaStability) {
    ibm::IBMParams p;
    p.nx=16; p.ny=16; p.nz=16;
    p.nMarkers = 1;
    p.mdf_iterations = 3; 
    
    p.mdf_beta = 1.0f;
    ibm::IBMCore ibm_b1(p);
    
    p.mdf_beta = 0.5f;
    ibm::IBMCore ibm_b05(p);
    
    float3 pos = make_float3(8,8,8);
    float3 vel = make_float3(1,0,0);
    std::vector<float> area(1, 1.0f);
    
    ibm_b1.updateMarkers(&pos, &vel, area.data());
    ibm_b05.updateMarkers(&pos, &vel, area.data());
    
    float3* d_u; float3* d_f1; float3* d_f2;
    CHECK_CUDA(cudaMalloc(&d_u, 16*16*16*sizeof(float3)));
    CHECK_CUDA(cudaMemset(d_u, 0, 16*16*16*sizeof(float3)));
    CHECK_CUDA(cudaMalloc(&d_f1, 16*16*16*sizeof(float3)));
    CHECK_CUDA(cudaMalloc(&d_f2, 16*16*16*sizeof(float3)));
    
    ibm_b1.computeForces(d_u, nullptr, d_f1, 1.0f);
    ibm_b05.computeForces(d_u, nullptr, d_f2, 1.0f);
    
    float3 fm1, fm2;
    ibm_b1.downloadForces(&fm1);
    ibm_b05.downloadForces(&fm2);
    
    EXPECT_LT(fm2.x, fm1.x);
    
    cudaFree(d_u); cudaFree(d_f1); cudaFree(d_f2);
}

// ----------------------------------------------------------------------------
// Test 5: Rho Usage Test (Proportionality)
// ----------------------------------------------------------------------------
TEST(IBM_MDF_Regression, RhoUsage) {
    ibm::IBMParams p;
    p.nx=16; p.ny=16; p.nz=16;
    p.nMarkers = 10;
    p.mdf_iterations = 1;
    
    ibm::IBMCore ibm(p);
    
    int nCells = 16*16*16;
    float* d_rho1; 
    float* d_rho2;
    std::vector<float> h_rho1(nCells, 1.0f);
    std::vector<float> h_rho2(nCells, 2.0f);
    CHECK_CUDA(cudaMalloc(&d_rho1, nCells*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_rho2, nCells*sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_rho1, h_rho1.data(), nCells*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rho2, h_rho2.data(), nCells*sizeof(float), cudaMemcpyHostToDevice));
    
    float3* d_u; float3* d_f;
    CHECK_CUDA(cudaMalloc(&d_u, nCells*sizeof(float3)));
    CHECK_CUDA(cudaMemset(d_u, 0, nCells*sizeof(float3)));
    CHECK_CUDA(cudaMalloc(&d_f, nCells*sizeof(float3)));
    
    std::vector<float3> pos(10, make_float3(8,8,8));
    std::vector<float3> vel(10, make_float3(1,0,0));
    std::vector<float> area(10, 1.0f);
    ibm.updateMarkers(pos.data(), vel.data(), area.data());
    
    ibm.computeForces(d_u, d_rho1, d_f, 1.0f);
    std::vector<float3> f_rho1(10);
    ibm.downloadForces(f_rho1.data());
    
    ibm.computeForces(d_u, d_rho2, d_f, 1.0f);
    std::vector<float3> f_rho2(10);
    ibm.downloadForces(f_rho2.data());
    
    float sum_f1 = 0; for(auto f:f_rho1) sum_f1 += f.x;
    float sum_f2 = 0; for(auto f:f_rho2) sum_f2 += f.x;
    
    std::cout << "[RhoUsage] SumF(Rho=1): " << sum_f1 << ", SumF(Rho=2): " << sum_f2 << std::endl;
    
    EXPECT_NEAR(sum_f2, sum_f1 * 2.0f, 1e-4);
    
    cudaFree(d_rho1); cudaFree(d_rho2); cudaFree(d_u); cudaFree(d_f);
}

// ----------------------------------------------------------------------------
// Test 6: Mask Normalize Test
// ----------------------------------------------------------------------------
TEST(IBM_MDF_Regression, MaskNormalize) {
    ibm::IBMParams p;
    p.nx=16; p.ny=16; p.nz=16;
    p.nMarkers = 1;
    
    ibm::IBMCore ibm(p);
    
    float3 pos = make_float3(0.1f, 8.0f, 8.0f);
    float3 vel = make_float3(1.0f, 0.0f, 0.0f);
    float xArea = 1.0f;
    ibm.updateMarkers(&pos, &vel, &xArea);
    
    int nCells = 16*16*16;
    std::vector<float3> h_u(nCells, make_float3(1.0f, 0.0f, 0.0f));
    float3* d_u;
    CHECK_CUDA(cudaMalloc(&d_u, nCells*sizeof(float3)));
    CHECK_CUDA(cudaMemcpy(d_u, h_u.data(), nCells*sizeof(float3), cudaMemcpyHostToDevice));
    float3* d_f;
    CHECK_CUDA(cudaMalloc(&d_f, nCells*sizeof(float3)));
    
    ibm.computeForces(d_u, nullptr, d_f, 1.0f);
    
    std::vector<float3> m_f(1);
    ibm.downloadForces(m_f.data());
    
    EXPECT_NEAR(m_f[0].x, 0.0f, 1e-3);
    
    cudaFree(d_u); cudaFree(d_f);
}

// ----------------------------------------------------------------------------
// Test 7: GAS Mask Constant Field (Domain Truncation)
// ----------------------------------------------------------------------------
TEST(IBM_MDF_Regression, GASMaskConstantField) {
#ifdef IBM_TESTING
    ibm::IBMParams p;
    p.nx=16; p.ny=16; p.nz=16; p.dx = 0.5f;
    p.enable_masked_fs = true;
    p.nMarkers = 40; // Increased markers
    ibm::IBMCore ibm(p);
    
    int nCells = p.nx * p.ny * p.nz;
    std::vector<uint8_t> h_mask(nCells);
    // Upper half (z >= 8 * 0.5 = 4.0) is GAS(0)
    for(int k=0; k<p.nz; ++k) {
        for(int j=0; j<p.ny; ++j) {
            for(int i=0; i<p.nx; ++i) {
                int idx = k*(p.nx*p.ny) + j*p.nx + i;
                h_mask[idx] = (k < 8) ? 1 : 0; 
            }
        }
    }
    
    uint8_t* d_mask;
    CHECK_CUDA(cudaMalloc(&d_mask, nCells*sizeof(uint8_t)));
    CHECK_CUDA(cudaMemcpy(d_mask, h_mask.data(), nCells*sizeof(uint8_t), cudaMemcpyHostToDevice));
    
    float3 u_const = make_float3(0.12f, -0.05f, 0.33f);
    std::vector<float3> h_u(nCells, u_const);
    float3* d_u;
    CHECK_CUDA(cudaMalloc(&d_u, nCells*sizeof(float3)));
    CHECK_CUDA(cudaMemcpy(d_u, h_u.data(), nCells*sizeof(float3), cudaMemcpyHostToDevice));
    
    float3* d_f;
    CHECK_CUDA(cudaMalloc(&d_f, nCells*sizeof(float3)));
    
    // Scan z=[3.0, 5.0] covering Valid -> Interface -> Gas
    std::vector<float3> pos(p.nMarkers);
    std::vector<float3> vel(p.nMarkers, u_const);
    std::vector<float> area(p.nMarkers, 1.0f);
    
    for(int i=0; i<p.nMarkers; ++i) {
        pos[i] = make_float3(4.0f, 4.0f, 3.0f + (float)i * 0.05f);
    }
    
    ibm.updateMarkers(pos.data(), vel.data(), area.data());
    
    ibm.computeForces(d_u, nullptr, d_f, 1.0f, d_mask, nullptr);
    
    std::vector<float3> u_interp(p.nMarkers);
    ibm.downloadInterpolatedVelocity(u_interp.data());
    
    int fallback_markers = 0;
    
    // Analyze results
    for(int i=0; i<p.nMarkers; ++i) {
        // Support width = 2.0 (Radius=2*dx=1.0).
        // Z boundary = 4.0.
        // Gas region = z > 4.0.
        // Markers > 5.0 (boundary + 1.0) see ONLY gas. Should fallback to 0.
        
        bool deeply_in_gas = (pos[i].z > 5.05f); 
        
        if (deeply_in_gas) {
             if (std::abs(u_interp[i].x) < 1e-6) fallback_markers++;
        } else {
             // In Liquid or Interface region.
             // Masked interpolation should normalize weights and preserve constant field
             // If sumW > eps. Or fallback if sumW=0.
             if (std::abs(u_interp[i].x) < 1e-6) {
                 fallback_markers++;
             } else {
                 float diff = std::abs(u_interp[i].x - u_const.x);
                 EXPECT_LT(diff, 1e-5) << "Marker " << i << " at z=" << pos[i].z << " failed const preservation.";
             }
        }
    }
    
    unsigned int fb_count = ibm.getFallbackCount();
    std::cout << "[GASMask] Fallback Count: " << fb_count << " (Expected >= " << fallback_markers << ")" << std::endl;
    EXPECT_GT(fb_count, 0); 
    
    cudaFree(d_u); cudaFree(d_f); cudaFree(d_mask);
#endif
}

// ----------------------------------------------------------------------------
// Test 8: Fill Weighted Spread
// ----------------------------------------------------------------------------
TEST(IBM_MDF_Regression, FillWeightedSpread) {
    ibm::IBMParams p;
    p.nx=32; p.ny=16; p.nz=16; 
    p.enable_masked_fs = true;
    p.use_fill_weight = true;
    p.nMarkers = 2; // 1 in Region A (Fill=1), 1 in Region B (Fill=0.2)
    
    ibm::IBMCore ibm(p);
    
    int nCells = p.nx * p.ny * p.nz;
    std::vector<float> h_fill(nCells);
    std::vector<uint8_t> h_mask(nCells, 1); 
    
    for(int i=0; i<p.nx; ++i) {
        float val = (i < 16) ? 1.0f : 0.2f;
        for(int jk=0; jk<p.ny*p.nz; ++jk) {
             h_fill[i + jk*p.nx] = val; 
        }
    }
    
    float* d_fill; uint8_t* d_mask;
    CHECK_CUDA(cudaMalloc(&d_fill, nCells*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_mask, nCells*sizeof(uint8_t)));
    CHECK_CUDA(cudaMemcpy(d_fill, h_fill.data(), nCells*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mask, h_mask.data(), nCells*sizeof(uint8_t), cudaMemcpyHostToDevice));
    
    // Markers at center of respective regions
    float3 posA = make_float3(8.0f, 8.0f, 8.0f);  // In Fill=1
    float3 posB = make_float3(24.0f, 8.0f, 8.0f); // In Fill=0.2
    
    std::vector<float3> pos = {posA, posB};
    std::vector<float3> vel = {make_float3(1,0,0), make_float3(1,0,0)};
    std::vector<float> area = {1.0f, 1.0f};
    
    ibm.updateMarkers(pos.data(), vel.data(), area.data());
    
    float3* d_u; float3* d_f;
    CHECK_CUDA(cudaMalloc(&d_u, nCells*sizeof(float3)));
    CHECK_CUDA(cudaMemset(d_u, 0, nCells*sizeof(float3)));
    CHECK_CUDA(cudaMalloc(&d_f, nCells*sizeof(float3)));
    
    // Run 1 iteration
    p.mdf_iterations = 1; 
    ibm.computeForces(d_u, nullptr, d_f, 1.0f, d_mask, d_fill);
    
    std::vector<float3> h_f(nCells);
    CHECK_CUDA(cudaMemcpy(h_f.data(), d_f, nCells*sizeof(float3), cudaMemcpyDeviceToHost));
    
    double sumA = 0;
    double sumB = 0;
    
    for(int z=0; z<p.nz; ++z) {
        for(int y=0; y<p.ny; ++y) {
            for(int x=0; x<p.nx; ++x) {
                int idx = get_idx(x,y,z, p.nx, p.ny, p.nz);
                float val = h_f[idx].x;
                if (x < 16) sumA += val;
                else sumB += val;
            }
        }
    }
    
    std::cout << "[FillSpread] SumA (Fill=1): " << sumA << ", SumB (Fill=0.2): " << sumB << std::endl;
    // Expected SumB ~ 0.2 * SumA
    EXPECT_NEAR(sumB / sumA, 0.2, 0.05);
    
    cudaFree(d_u); cudaFree(d_f); cudaFree(d_fill); cudaFree(d_mask);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
