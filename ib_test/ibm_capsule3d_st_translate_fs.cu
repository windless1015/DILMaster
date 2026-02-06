
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "../src/geometry/STLGeometryLoader.hpp"
#include "../src/geometry/STLReader.h"
#include "../src/geometry/VectorTypes.h"
#include "../src/physics/ibm/IBMCore.hpp"
#include "../src/physics/lbm/FreeSurfaceModule.hpp"
#include "../src/physics/lbm/LBMConfig.hpp"
#include "../src/physics/lbm/LBMCore.hpp"
#include <cuda_runtime.h>

// Services for Time-Series VTK Output
#include "../src/core/FieldStore.hpp"
#include "../src/core/StepContext.hpp"
#include "../src/services/MarkerVTKService.hpp"
#include "../src/services/VTKService.hpp"

// DIAGNOSTICS MODULE
#include "diagnostics/free_surface_diagnostics.cuh"

namespace fs = std::filesystem;

#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line "    \
                << __LINE__ << std::endl;                                      \
      exit(1);                                                                 \
    }                                                                          \
  }

// Kernel Helpers
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

// Helper to rotate mesh points around Z axis
void rotateMeshZ(STLMesh &mesh, float angle_deg) {
  float rad = angle_deg * 3.14159265359f / 180.0f;
  float c = cosf(rad);
  float s = sinf(rad);

  for (auto &tri : mesh.triangles) {
    // Rotate vertices
    for (int k = 0; k < 3; ++k) {
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

struct SimStats {
  float max_amp;
  float avg_corr;
  float last_corr;
  bool is_stable;
  float wake_angle;
};

// Extracted Simulation Logic
SimStats
run_simulation(int nx, int ny, int nz, float tau, float U0,
               float fluid_fraction, float capsule_x_ratio,
               float capsule_depth_ratio, // Used if explicit_depth_R < 0
               float explicit_depth_R,    // If >= 0, sets depth = R * this
               std::string out_dir, int steps, int output_every,
               std::string stl_path, float spacing_req, int mdf_iter,
               float beta, float angle, float scale, bool verbose) {
  // Re-create dir
  fs::create_directories(out_dir);

  // 1. Load Geometry
  STLMesh mesh;
  if (!STLReader::readSTL(stl_path, mesh)) {
    std::cerr << "Failed to load STL: " << stl_path << std::endl;
    exit(1);
  }
  mesh.centerAtOrigin();
  if (scale != 1.0f)
    mesh.transform(make_float3(0, 0, 0), scale);
  if (angle != 0.0f)
    rotateMeshZ(mesh, angle);
  float D = mesh.getSize().y;
  float R = D * 0.5f;

  // 2. Sample Markers
  float3 center_zero = make_float3(0, 0, 0);
  std::vector<IBMMarker> markers = STLGeometryLoader::sampleSurfaceMarkers(
      mesh, center_zero, (float)spacing_req);
  size_t nMarkers = markers.size();
  std::vector<float3> rel_pos(nMarkers);
  std::vector<float> area_vec(nMarkers);
  for (size_t i = 0; i < nMarkers; ++i) {
    rel_pos[i] = markers[i].pos;
    area_vec[i] = markers[i].area;
  }

  // 3. LBM Setup
  float dx = 1.0f;
  float nu = (tau - 0.5f) / 3.0f;
  float Re = U0 * D / nu;
  float water_level_z = nz * fluid_fraction;

  if (verbose) {
    std::cout << "\n>>> Simulation Start: U=" << U0 << ", Re=" << Re << " <<<"
              << std::endl;
    std::cout << "Domain: " << nx << "x" << ny << "x" << nz << std::endl;
    std::cout << "Water Level Z: " << water_level_z << std::endl;
  }

  lbm::LBMConfig lbm_cfg;
  lbm_cfg.nx = nx;
  lbm_cfg.ny = ny;
  lbm_cfg.nz = nz;
  lbm_cfg.tau = tau;
  lbm_cfg.enableFreeSurface = true;
  lbm_cfg.collisionModel = lbm::CollisionModel::SRT;
  lbm_cfg.bcXMin = lbm::BC_PRESSURE_OUTLET;
  lbm_cfg.bcXMax = lbm::BC_PRESSURE_OUTLET;
  lbm_cfg.bcYMin = lbm::BC_PERIODIC;
  lbm_cfg.bcYMax = lbm::BC_PERIODIC;
  lbm_cfg.bcZMin = lbm::BC_BOUNCE_BACK;
  lbm_cfg.bcZMax = lbm::BC_OPEN;
  lbm_cfg.gravity = make_float3(0, 0, -0.0001f);
  lbm_cfg.rho0 = 1.0f;
  lbm_cfg.pressure_outlet_rho = 1.0f; // Reference pressure
  lbm_cfg.u0 = make_float3(0.0f, 0.0f, 0.0f);

  lbm::LBMCore lbm(lbm_cfg);
  lbm.initialize();

  // Free Surface
  lbm::FreeSurfaceModule fsModule;
  fsModule.configure(lbm_cfg);
  FieldStore fsFields;
  fsFields.create(FieldDesc{"fluid.density", (size_t)(nx * ny * nz),
                            sizeof(float), lbm.densityDevicePtr()});
  fsFields.create(FieldDesc{"fluid.velocity", (size_t)(nx * ny * nz) * 3,
                            sizeof(float), lbm.velocityDevicePtr()});
  fsFields.create(FieldDesc{"fluid.flags", (size_t)(nx * ny * nz),
                            sizeof(uint8_t), lbm.flagsDevicePtr()});
  fsFields.create(FieldDesc{"fluid.phi", (size_t)(nx * ny * nz), sizeof(float),
                            lbm.phiDevicePtr()});
  fsFields.create(FieldDesc{"fluid.mass", (size_t)(nx * ny * nz), sizeof(float),
                            lbm.massDevicePtr()});
  fsModule.allocate(fsFields);
  fsModule.initialize(fsFields);

  int water_z = (int)water_level_z;
  fsModule.setRegion(0, nx - 1, 0, ny - 1, 0, nz - 1, lbm::CellType::GAS, 0.0f,
                     lbm_cfg.rho0);
  fsModule.setRegion(0, nx - 1, 0, ny - 1, 0, water_z - 1,
                     lbm::CellType::LIQUID, 1.0f, lbm_cfg.rho0);
  fsModule.fixInterfaceLayer();

  // 4. IBM Setup
  ibm::IBMParams ibm_p;
  ibm_p.nx = nx;
  ibm_p.ny = ny;
  ibm_p.nz = nz;
  ibm_p.dx = dx;
  ibm_p.nMarkers = (int)nMarkers;
  ibm_p.mdf_iterations = mdf_iter;
  ibm_p.mdf_beta = beta;
  ibm_p.force_method = ibm::IBMForceMethod::DIRECT_FORCING;
  ibm::IBMCore ibm(ibm_p);

  // 5. Capsule Position
  float capsule_z;
  if (explicit_depth_R >= 0.0f) {
    // depth is R * val below surface
    capsule_z = water_level_z - explicit_depth_R * R;
  } else {
    capsule_z = water_level_z - (water_level_z * capsule_depth_ratio);
  }

  float3 center = make_float3(nx * capsule_x_ratio, ny * 0.5f, capsule_z);
  float3 U_obj = make_float3(U0, 0.0f, 0.0f);

  if (verbose) {
    std::cout << "Capsule Position: (" << center.x << ", " << center.y << ", "
              << center.z << ")" << std::endl;
    std::cout << "Depth/R: " << ((water_level_z - center.z) / R) << std::endl;
  }

  std::vector<float3> h_pos(nMarkers);
  std::vector<float3> h_vel(nMarkers);
  for (size_t k = 0; k < nMarkers; ++k)
    h_vel[k] = U_obj;

  float3 *d_force;
  CHECK_CUDA(cudaMalloc(&d_force, nx * ny * nz * sizeof(float3)));

  // Services
  FieldStore fields;
  StepContext ctx;
  ctx.fields = &fields;
  ctx.backend = &lbm.backend();

  VTKService::Config vtk_cfg;
  vtk_cfg.output_dir = out_dir;
  vtk_cfg.nx = nx;
  vtk_cfg.ny = ny;
  vtk_cfg.nz = nz;
  vtk_cfg.dx = dx;
  vtk_cfg.interval = output_every;
  vtk_cfg.binary = true;
  vtk_cfg.fields = {"Density", "Speed", "Velocity", "Fill"};

  MarkerVTKService::Config m_cfg;
  m_cfg.output_dir = out_dir;
  m_cfg.interval = output_every;
  m_cfg.marker_field = "ibm.markers";
  m_cfg.binary = true;

  VTKService vtk_svc(vtk_cfg);
  MarkerVTKService marker_svc(m_cfg);
  vtk_svc.initialize(ctx);
  marker_svc.initialize(ctx);

  // Diagnostics
  diagnostics::FreeSurfaceDiagnostics fsDiag(
      nx, ny, nz, lbm_cfg.rho0, lbm_cfg.gravity.z, water_level_z, out_dir);
  // Set probe above center
  fsDiag.setProbeLocation((int)center.x, (int)center.y);

  // Pre-relaxation
  float3 center_init = center;
  for (int pre = 0; pre < 200; ++pre) {
    float ramp = (pre < 100) ? 0.0f : (pre - 100) / 100.0f;
    float3 cv = make_float3(U0 * ramp, 0.0f, 0.0f);
    for (size_t k = 0; k < nMarkers; ++k) {
      h_pos[k].x = center_init.x + rel_pos[k].x;
      h_pos[k].y = center_init.y + rel_pos[k].y;
      h_pos[k].z = center_init.z + rel_pos[k].z;
      h_vel[k] = cv;
    }
    ibm.updateMarkers(h_pos.data(), h_vel.data(), area_vec.data());
    CHECK_CUDA(cudaMemset(d_force, 0, nx * ny * nz * sizeof(float3)));
    ibm.computeForces(lbm.velocityAoSPtr(), nullptr, d_force, 1.0f);
    lbm.setExternalForceFromDeviceAoS(d_force);
    lbm.streamCollide();
    lbm.updateMacroscopic();
  }

  std::ofstream log_csv(out_dir + "/drag_time.csv");
  log_csv << "step,Cx,Cz,DragM,LiftM,Re_est\n";

  // Main Loop
  float sum_corr = 0.0f;
  int corr_count = 0;

  for (int t = 0; t <= steps; ++t) {
    ctx.step = t;
    ctx.time = t * 1.0;

    // Move Body
    bool moving = true;
    if (center.x > 0.85f * nx) {
      moving = false;
      U_obj = make_float3(0, 0, 0);
      for (auto &v : h_vel)
        v = U_obj;
    } else {
      center.x += U0;
    }

    // Check if probe needs to move to stay relative to body?
    // User said: "In body coordinate system: probe directly above capsule
    // center" But capsule is moving. "对胶囊中心正上方一个探针点记录" -> Record
    // probe at (center.x, center.y).
    fsDiag.setProbeLocation((int)center.x, (int)center.y);

    for (size_t k = 0; k < nMarkers; ++k) {
      h_pos[k].x = center.x + rel_pos[k].x;
      h_pos[k].y = center.y + rel_pos[k].y;
      h_pos[k].z = center.z + rel_pos[k].z;
    }
    ibm.updateMarkers(h_pos.data(), h_vel.data(), area_vec.data());

    CHECK_CUDA(cudaMemset(d_force, 0, nx * ny * nz * sizeof(float3)));
    ibm.computeForces(lbm.velocityAoSPtr(), nullptr, d_force, 1.0f);
    lbm.setExternalForceFromDeviceAoS(d_force);
    fsModule.preStream(ctx);
    lbm.streamCollide();
    fsModule.postStream(ctx);
    lbm.updateMacroscopic();

    // Diagnostics
    if (t % output_every == 0) {
      bool save = (t > 0);

      // Call Diagnostics Module
      // This computes Eta, P, Correlation, and saves VTK/CSV if save=true
      float corr = fsDiag.process(t, lbm.phiDevicePtr(), lbm.densityDevicePtr(),
                                  save, save);

      if (save) {
        sum_corr += corr;
        corr_count++;
      }

      // Standard VTK output (Code from original)
      if (save) {
        // ... Copy logic for VTKService ...
        // Minimal copy for brevity here as specialized diagnostics handle key
        // outputs
        auto h_rho =
            fields.create({"Density", (size_t)nx * ny * nz, sizeof(float)});
        auto h_u = fields.create(
            {"Velocity", (size_t)nx * ny * nz, sizeof(float) * 3});
        CHECK_CUDA(cudaMemcpy(h_rho.data(), lbm.getDensityField(),
                              nx * ny * nz * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_u.data(), lbm.velocityAoSPtr(),
                              nx * ny * nz * sizeof(float3),
                              cudaMemcpyDeviceToHost));

        // Fill
        auto h_fill =
            fields.create({"Fill", (size_t)nx * ny * nz, sizeof(float)});
        CHECK_CUDA(cudaMemcpy(h_fill.data(), lbm.phiDevicePtr(),
                              nx * ny * nz * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // Markers
        std::vector<float3> m_forces(nMarkers);
        ibm.downloadForces(m_forces.data());
        auto h_mk = fields.create({"ibm.markers", nMarkers, sizeof(float) * 3});
        memcpy(h_mk.data(), h_pos.data(), nMarkers * sizeof(float) * 3);

        vtk_svc.onStepEnd(ctx);
        marker_svc.onStepEnd(ctx);

        // Log
        double fx = 0, fy = 0;
        for (auto &f : m_forces) {
          fx += f.x;
          fy += f.y;
        }
        log_csv << t << "," << center.x << "," << center.z << "," << -fx << ","
                << -fy << "," << Re << "\n";
      }
    }
  }

  // Final check
  fsDiag.checkWakeAngle((int)center.x, (int)center.y);
  bool stable = fsDiag.isStable();
  float corr_final = (corr_count > 0) ? (sum_corr / corr_count) : 0.0f;

  if (verbose) {
    fsDiag.printFinalVerdict(corr_final > 0.7f, stable);
  }

  vtk_svc.finalize(ctx);
  marker_svc.finalize(ctx);
  cudaFree(d_force);
  log_csv.close();

  return {fsDiag.max_amplitude, corr_final, fsDiag.last_corr, stable,
          fsDiag.wake_angle_deg};
}

// -----------------------------------------------------------------------------
// Helper to parse lists "1.5,2.0,3.0"
std::vector<float> parseList(std::string str) {
  std::vector<float> res;
  std::stringstream ss(str);
  std::string item;
  while (std::getline(ss, item, ',')) {
    try {
      res.push_back(std::stof(item));
    } catch (...) {
    }
  }
  return res;
}

int main(int argc, char **argv) {
  std::cout << "--- FreeSurface Validation Enabled ---" << std::endl;
  std::cout << "Expect drawdown over body if negative pressure region exists."
            << std::endl;

  // Defaults
  std::string stl_path = "../../tools/capsule.stl";
  int nx = 512, ny = 128, nz = 128; // Increased Z for depth
  float tau = 0.8f;
  float U0 = 0.08f;
  float spacing_req = 1.0f;
  int mdf_iter = 5;
  float beta = -0.5f;
  float angle = 0.0f;
  float scale = 1.0f;
  int steps = 5000;
  int output_every = 100;
  std::string out_dir = "out/capsule_fs";
  float fluid_fraction = 0.7f;
  float capsule_x_ratio = 0.15f;
  float capsule_depth_ratio = 0.5f;

  // Loop lists
  std::string depth_list_str = "";
  std::string U_list_str = "";

  // Args Parsing
  if (argc <= 1) {
    // VS Default Mode: Activate Scans
    std::cout
        << "\n[VS Mode] No arguments detected. Running automated diagnostics "
           "scans...\n";

    // 1. Depth Scan
    std::vector<float> depth_list = {1.5f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::ofstream sum_csv(out_dir + "/summary_depth.csv");
    sum_csv << "h/R,A_max,Corr\n";

    for (float h : depth_list) {
      std::cout << "\n>>> AUTO: Running Depth h/R = " << h << " <<<\n";
      std::string sub_out = out_dir + "/depth_" + std::to_string(h);
      SimStats stats = run_simulation(
          nx, ny, nz, tau, U0, fluid_fraction, capsule_x_ratio,
          capsule_depth_ratio, h, sub_out, 8000,
          output_every, // 8000 steps sufficient
          stl_path, spacing_req, mdf_iter, beta, angle, scale, false);

      sum_csv << h << "," << stats.max_amp << "," << stats.avg_corr << "\n";
      sum_csv.flush();
    }
    std::cout << "Depth scan complete used default list.\n";

    // 2. Velocity Scan
    std::vector<float> U_list = {0.04f, 0.06f, 0.08f};
    std::ofstream sum_v_csv(out_dir + "/summary_velocity.csv");
    sum_v_csv << "U,Fr,A_max,Corr\n";

    // Need Length for Fr. Load STL to get D? Or reuse D.
    STLMesh mesh;
    STLReader::readSTL(stl_path, mesh);
    float D = mesh.getSize().y;
    float g = 0.0001f;

    for (float U : U_list) {
      std::cout << "\n>>> AUTO: Running Velocity U = " << U << " <<<\n";
      std::string sub_out = out_dir + "/vel_" + std::to_string(U);
      SimStats stats = run_simulation(
          nx, ny, nz, tau, U, fluid_fraction, capsule_x_ratio,
          capsule_depth_ratio, -1.0f, sub_out, 8000, output_every, stl_path,
          spacing_req, mdf_iter, beta, angle, scale, false);

      float Fr = U / sqrt(g * D);
      sum_v_csv << U << "," << Fr << "," << stats.max_amp << ","
                << stats.avg_corr << "\n";
      sum_v_csv.flush();
    }
    std::cout << "Velocity scan complete used default list.\n";

    return 0;
  }

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--stl")
      stl_path = argv[++i];
    else if (arg == "--nx")
      nx = std::stoi(argv[++i]);
    else if (arg == "--ny")
      ny = std::stoi(argv[++i]);
    else if (arg == "--nz")
      nz = std::stoi(argv[++i]);
    else if (arg == "--tau")
      tau = std::stof(argv[++i]);
    else if (arg == "--U0")
      U0 = std::stof(argv[++i]);
    else if (arg == "--depth_list")
      depth_list_str = argv[++i];
    else if (arg == "--U_list")
      U_list_str = argv[++i];
    else if (arg == "--steps")
      steps = std::stoi(argv[++i]);
    else if (arg == "--outDir")
      out_dir = argv[++i];
    // ... (other args support)
  }

  if (!depth_list_str.empty()) {
    auto list = parseList(depth_list_str);
    std::ofstream sum_csv("summary_depth.csv");
    sum_csv << "h/R,A_max,Corr\n";

    for (float h : list) {
      std::cout << "\n>>> AUTO: Running Depth h/R = " << h << " <<<\n";
      std::string sub_out = out_dir + "/depth_" + std::to_string(h);
      SimStats stats = run_simulation(
          nx, ny, nz, tau, U0, fluid_fraction, capsule_x_ratio,
          capsule_depth_ratio, h, sub_out, steps, output_every, stl_path,
          spacing_req, mdf_iter, beta, angle, scale, false);

      sum_csv << h << "," << stats.max_amp << "," << stats.avg_corr << "\n";
      sum_csv.flush();
    }
    std::cout << "Depth scan complete. Saved to summary_depth.csv\n";
  } else if (!U_list_str.empty()) {
    auto list = parseList(U_list_str);
    std::ofstream sum_csv("summary_velocity.csv");
    sum_csv << "U,Fr,A_max,Corr\n";

    // Need Length for Fr. Load STL to get D? Or reuse D.
    STLMesh mesh;
    STLReader::readSTL(stl_path, mesh);
    float D = mesh.getSize().y; // Assuming Y is D as per prior code
    float g = 0.0001f;

    for (float U : list) {
      std::cout << "\n>>> AUTO: Running Velocity U = " << U << " <<<\n";
      std::string sub_out = out_dir + "/vel_" + std::to_string(U);
      SimStats stats = run_simulation(
          nx, ny, nz, tau, U, fluid_fraction, capsule_x_ratio,
          capsule_depth_ratio, -1.0f, sub_out, steps, output_every, stl_path,
          spacing_req, mdf_iter, beta, angle, scale, false);

      float Fr = U / sqrt(g * D);
      sum_csv << U << "," << Fr << "," << stats.max_amp << "," << stats.avg_corr
              << "\n";
      sum_csv.flush();
    }
    std::cout << "Velocity scan complete. Saved to summary_velocity.csv\n";
  } else {
    // Single Run
    run_simulation(nx, ny, nz, tau, U0, fluid_fraction, capsule_x_ratio,
                   capsule_depth_ratio, -1.0f, out_dir, steps, output_every,
                   stl_path, spacing_req, mdf_iter, beta, angle, scale, true);
  }

  return 0;
}
