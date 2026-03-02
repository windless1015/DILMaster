
#include <algorithm>
#include <cmath>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "../../src/geometry/STLGeometryLoader.hpp"
#include "../../src/geometry/STLReader.h"
#include "../../src/geometry/VectorTypes.h"
#include "../../src/physics/ibm/IBMCore.hpp"
#include "../../src/physics/lbm/FreeSurfaceModule.hpp"
#include "../../src/physics/lbm/LBMConfig.hpp"
#include "../../src/physics/lbm/LBMCore.hpp"
#include <cuda_runtime.h>
#include <toml++/toml.hpp>

// Services for Time-Series VTK Output
#include "../../src/core/FieldStore.hpp"
#include "../../src/core/StepContext.hpp"
#include "../../src/services/MarkerVTKService.hpp"
#include "../../src/services/VTKService.hpp"

// DIAGNOSTICS MODULE
#include "../diagnostics/free_surface_diagnostics.cuh"

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

struct AppConfig {
  struct PhysicalConfig {
    bool enabled;
    float domain_lx_m;
    float domain_ly_m;
    float domain_lz_m;
    float u0_mps;
    float dt_s;
    float capsule_diameter_m;
    float single_depth_m;
    std::vector<float> depth_list_m;
    std::vector<float> velocity_list_mps;
  };

  std::string stl_path;
  int nx;
  int ny;
  int nz;
  float tau;
  float U0;
  float spacing_req;
  int mdf_iter;
  float beta;
  float angle;
  float scale;
  int steps;
  int output_every;
  std::string out_dir;
  float fluid_fraction;
  float capsule_x_ratio;
  float capsule_depth_ratio;
  float single_depth_R;
  std::vector<float> depth_list;
  std::vector<float> velocity_list;
  PhysicalConfig physical;
};

#ifndef IBM_CAPSULE3D_ST_TRANSLATE_FS_CONFIG
#define IBM_CAPSULE3D_ST_TRANSLATE_FS_CONFIG "config.toml"
#endif

AppConfig loadConfig(const std::string &config_path) {
  AppConfig cfg{
      "../../tools/capsule.stl",
      512,
      128,
      128,
      0.8f,
      0.08f,
      1.0f,
      5,
      -0.5f,
      0.0f,
      1.0f,
      5000,
      100,
      "out/capsule_fs",
      0.7f,
      0.15f,
      0.5f,
      -1.0f,
      {1.5f, 2.0f, 3.0f, 4.0f, 5.0f},
      {0.04f, 0.06f, 0.08f},
      {false, 100.0f, 25.0f, 25.0f, 10.0f, 0.0f, 6.0f, -1.0f, {}, {}}};

  toml::table tbl = toml::parse_file(config_path);
  auto sim = tbl["simulation"].as_table();
  auto geom = tbl["geometry"].as_table();
  auto output = tbl["output"].as_table();
  auto scan = tbl["scan"].as_table();
  auto physical = tbl["physical"].as_table();

  auto readInt = [](const toml::table *t, const char *k, int &v) {
    if (!t) return;
    if (auto val = (*t)[k].value<std::int64_t>()) {
      v = static_cast<int>(*val);
    }
  };
  auto readFloat = [](const toml::table *t, const char *k, float &v) {
    if (!t) return;
    if (auto d = (*t)[k].value<double>()) {
      v = static_cast<float>(*d);
    } else if (auto i = (*t)[k].value<std::int64_t>()) {
      v = static_cast<float>(*i);
    }
  };
  auto readString = [](const toml::table *t, const char *k, std::string &v) {
    if (!t) return;
    if (auto s = (*t)[k].value<std::string>()) {
      v = *s;
    }
  };
  auto readFloatArray = [](const toml::table *t, const char *k, std::vector<float> &v) {
    if (!t) return;
    const auto *arr = (*t)[k].as_array();
    if (!arr) return;
    v.clear();
    for (const auto &node : *arr) {
      if (auto d = node.value<double>()) {
        v.push_back(static_cast<float>(*d));
      } else if (auto i = node.value<std::int64_t>()) {
        v.push_back(static_cast<float>(*i));
      }
    }
  };
  auto readBool = [](const toml::table *t, const char *k, bool &v) {
    if (!t) return;
    if (auto b = (*t)[k].value<bool>()) {
      v = *b;
    }
  };

  readString(sim, "stl_path", cfg.stl_path);
  readInt(sim, "nx", cfg.nx);
  readInt(sim, "ny", cfg.ny);
  readInt(sim, "nz", cfg.nz);
  readFloat(sim, "tau", cfg.tau);
  readFloat(sim, "U0", cfg.U0);
  readFloat(sim, "spacing_req", cfg.spacing_req);
  readInt(sim, "mdf_iter", cfg.mdf_iter);
  readFloat(sim, "beta", cfg.beta);
  readFloat(sim, "angle", cfg.angle);
  readFloat(sim, "scale", cfg.scale);
  readInt(sim, "steps", cfg.steps);
  readInt(sim, "output_every", cfg.output_every);
  readFloat(sim, "single_depth_R", cfg.single_depth_R);

  readString(output, "out_dir", cfg.out_dir);

  readFloat(geom, "fluid_fraction", cfg.fluid_fraction);
  readFloat(geom, "capsule_x_ratio", cfg.capsule_x_ratio);
  readFloat(geom, "capsule_depth_ratio", cfg.capsule_depth_ratio);

  readFloatArray(scan, "depth_list", cfg.depth_list);
  readFloatArray(scan, "velocity_list", cfg.velocity_list);
  readFloatArray(scan, "depth_list_m", cfg.physical.depth_list_m);
  readFloatArray(scan, "velocity_list_mps", cfg.physical.velocity_list_mps);

  readBool(physical, "enabled", cfg.physical.enabled);
  readFloat(physical, "domain_lx_m", cfg.physical.domain_lx_m);
  readFloat(physical, "domain_ly_m", cfg.physical.domain_ly_m);
  readFloat(physical, "domain_lz_m", cfg.physical.domain_lz_m);
  readFloat(physical, "u0_mps", cfg.physical.u0_mps);
  readFloat(physical, "dt_s", cfg.physical.dt_s);
  readFloat(physical, "capsule_diameter_m", cfg.physical.capsule_diameter_m);
  readFloat(physical, "single_depth_m", cfg.physical.single_depth_m);

  return cfg;
}

void runLegacyBatchScans(const AppConfig& cfg, float U0, float single_depth_R, std::vector<float> depth_list, std::vector<float> U_list, bool has_runtime_args, bool depth_list_from_cli, bool velocity_list_from_cli) {
  int nx = cfg.nx, ny = cfg.ny, nz = cfg.nz;
  float tau = cfg.tau;
  float spacing_req = cfg.spacing_req;
  int mdf_iter = cfg.mdf_iter;
  float beta = cfg.beta;
  float angle = cfg.angle;
  float scale = cfg.scale;
  int steps = cfg.steps;
  int output_every = cfg.output_every;
  std::string stl_path = cfg.stl_path;
  std::string out_dir = cfg.out_dir;
  float fluid_fraction = cfg.fluid_fraction;
  float capsule_x_ratio = cfg.capsule_x_ratio;
  float capsule_depth_ratio = cfg.capsule_depth_ratio;

  if (!has_runtime_args) {
    std::cout << "\n[Config Mode] No runtime args detected. Running configured scans...\n";
    std::ofstream sum_csv("summary_depth.csv");
    sum_csv << "h/R,A_max,Corr\n";

    for (float h : depth_list) {
      std::cout << "\n>>> AUTO: Running Depth h/R = " << h << " <<<\n";
      std::string sub_out = out_dir + "/depth_" + std::to_string(h);
      SimStats stats = run_simulation(
          nx, ny, nz, tau, U0, fluid_fraction, capsule_x_ratio,
          capsule_depth_ratio, h, sub_out, steps, output_every, stl_path,
          spacing_req, mdf_iter, beta, angle, scale, false);
      sum_csv << h << "," << stats.max_amp << "," << stats.avg_corr << "\n";
      sum_csv.flush();
    }
    std::cout << "Depth scan complete used configured list.\n";

    std::ofstream sum_v_csv("summary_velocity.csv");
    sum_v_csv << "U,Fr,A_max,Corr\n";

    STLMesh mesh;
    STLReader::readSTL(stl_path, mesh);
    float D = mesh.getSize().y;
    float g = 0.0001f;

    for (float U : U_list) {
      std::cout << "\n>>> AUTO: Running Velocity U = " << U << " <<<\n";
      std::string sub_out = out_dir + "/vel_" + std::to_string(U);
      SimStats stats = run_simulation(
          nx, ny, nz, tau, U, fluid_fraction, capsule_x_ratio, capsule_depth_ratio,
          -1.0f, sub_out, steps, output_every, stl_path,
          spacing_req, mdf_iter, beta, angle, scale, false);

      float Fr = U / sqrt(g * D);
      sum_v_csv << U << "," << Fr << "," << stats.max_amp << "," << stats.avg_corr << "\n";
      sum_v_csv.flush();
    }
    std::cout << "Velocity scan complete used configured list.\n";
  } else {
    if (depth_list_from_cli) {
      std::ofstream sum_csv("summary_depth.csv");
      sum_csv << "h/R,A_max,Corr\n";
      for (float h : depth_list) {
        std::string sub_out = out_dir + "/depth_" + std::to_string(h);
        SimStats stats = run_simulation(
            nx, ny, nz, tau, U0, fluid_fraction, capsule_x_ratio, capsule_depth_ratio, h,
            sub_out, steps, output_every, stl_path, spacing_req, mdf_iter, beta, angle,
            scale, false);
        sum_csv << h << "," << stats.max_amp << "," << stats.avg_corr << "\n";
      }
      std::cout << "Depth scan complete. Saved to summary_depth.csv\n";
    } else if (velocity_list_from_cli) {
      std::ofstream sum_csv("summary_velocity.csv");
      sum_csv << "U,Fr,A_max,Corr\n";
      STLMesh mesh;
      STLReader::readSTL(stl_path, mesh);
      float D = mesh.getSize().y;
      float g = 0.0001f;
      for (float U : U_list) {
        std::string sub_out = out_dir + "/vel_" + std::to_string(U);
        SimStats stats = run_simulation(
            nx, ny, nz, tau, U, fluid_fraction, capsule_x_ratio, capsule_depth_ratio,
            -1.0f, sub_out, steps, output_every, stl_path, spacing_req, mdf_iter, beta,
            angle, scale, false);
        float Fr = U / sqrt(g * D);
        sum_csv << U << "," << Fr << "," << stats.max_amp << "," << stats.avg_corr
                << "\n";
      }
      std::cout << "Velocity scan complete. Saved to summary_velocity.csv\n";
    }
  }
}

int main(int argc, char **argv) {
  std::cout << "--- FreeSurface Validation Enabled ---" << std::endl;
  std::cout << "Expect drawdown over body if negative pressure region exists."
            << std::endl;

  std::string config_path = IBM_CAPSULE3D_ST_TRANSLATE_FS_CONFIG;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--config" && i + 1 < argc) {
      config_path = argv[++i];
    }
  }

  AppConfig cfg;
  try {
    cfg = loadConfig(config_path);
    std::cout << "Loaded config: " << config_path << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Failed to load config '" << config_path << "': " << e.what()
              << std::endl;
    return 1;
  }

  std::string stl_path = cfg.stl_path;
  int nx = cfg.nx, ny = cfg.ny, nz = cfg.nz;
  float tau = cfg.tau;
  float U0 = cfg.U0;
  float spacing_req = cfg.spacing_req;
  int mdf_iter = cfg.mdf_iter;
  float beta = cfg.beta;
  float angle = cfg.angle;
  float scale = cfg.scale;
  int steps = cfg.steps;
  int output_every = cfg.output_every;
  std::string out_dir = cfg.out_dir;
  float fluid_fraction = cfg.fluid_fraction;
  float capsule_x_ratio = cfg.capsule_x_ratio;
  float capsule_depth_ratio = cfg.capsule_depth_ratio;
  float single_depth_R = cfg.single_depth_R;
  std::vector<float> depth_list = cfg.depth_list;
  std::vector<float> U_list = cfg.velocity_list;
  bool has_runtime_args = false;
  bool depth_list_from_cli = false;
  bool velocity_list_from_cli = false;
  bool u0_from_cli = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--config") {
      ++i;
    } else if (arg == "--stl") {
      has_runtime_args = true;
      stl_path = argv[++i];
    } else if (arg == "--nx") {
      has_runtime_args = true;
      nx = std::stoi(argv[++i]);
    } else if (arg == "--ny") {
      has_runtime_args = true;
      ny = std::stoi(argv[++i]);
    } else if (arg == "--nz") {
      has_runtime_args = true;
      nz = std::stoi(argv[++i]);
    } else if (arg == "--tau") {
      has_runtime_args = true;
      tau = std::stof(argv[++i]);
    } else if (arg == "--U0") {
      has_runtime_args = true;
      u0_from_cli = true;
      U0 = std::stof(argv[++i]);
    } else if (arg == "--depth_list") {
      has_runtime_args = true;
      depth_list_from_cli = true;
      depth_list = parseList(argv[++i]);
    } else if (arg == "--U_list") {
      has_runtime_args = true;
      velocity_list_from_cli = true;
      U_list = parseList(argv[++i]);
    } else if (arg == "--steps") {
      has_runtime_args = true;
      steps = std::stoi(argv[++i]);
    } else if (arg == "--outDir") {
      has_runtime_args = true;
      out_dir = argv[++i];
    }
    // ... (other args support)
  }

  if (cfg.physical.enabled) {
    float dx_m = cfg.physical.domain_lx_m / static_cast<float>(std::max(nx, 1));
    float dy_m = cfg.physical.domain_ly_m / static_cast<float>(std::max(ny, 1));
    float dz_m = cfg.physical.domain_lz_m / static_cast<float>(std::max(nz, 1));
    float dt_s = cfg.physical.dt_s;

    if (dt_s <= 0.0f && cfg.physical.u0_mps > 0.0f && U0 > 0.0f && dx_m > 0.0f) {
      dt_s = U0 * dx_m / cfg.physical.u0_mps;
    }

    if (!u0_from_cli && cfg.physical.u0_mps > 0.0f && dt_s > 0.0f && dx_m > 0.0f) {
      U0 = cfg.physical.u0_mps * dt_s / dx_m;
    }

    if (!depth_list_from_cli && !cfg.physical.depth_list_m.empty() &&
        cfg.physical.capsule_diameter_m > 0.0f) {
      depth_list.clear();
      const float R_m = 0.5f * cfg.physical.capsule_diameter_m;
      for (float h_m : cfg.physical.depth_list_m) {
        depth_list.push_back(h_m / R_m);
      }
    }

    if (cfg.physical.single_depth_m > 0.0f && cfg.physical.capsule_diameter_m > 0.0f) {
      const float R_m = 0.5f * cfg.physical.capsule_diameter_m;
      single_depth_R = cfg.physical.single_depth_m / R_m;
    }

    if (!velocity_list_from_cli && !cfg.physical.velocity_list_mps.empty() &&
        dt_s > 0.0f && dx_m > 0.0f) {
      U_list.clear();
      for (float u_mps : cfg.physical.velocity_list_mps) {
        U_list.push_back(u_mps * dt_s / dx_m);
      }
    }

    std::cout << "\n[Physical Mapping]\n";
    std::cout << "Domain(m): Lx=" << cfg.physical.domain_lx_m
              << ", Ly=" << cfg.physical.domain_ly_m
              << ", Lz=" << cfg.physical.domain_lz_m << "\n";
    std::cout << "Grid: nx=" << nx << ", ny=" << ny << ", nz=" << nz << "\n";
    std::cout << "dx(m): " << dx_m << ", dy(m): " << dy_m << ", dz(m): " << dz_m
              << "\n";
    if (dt_s > 0.0f) {
      std::cout << "dt(s): " << dt_s << "\n";
      std::cout << "U0 mapping: lattice=" << U0 << " <-> physical="
                << (U0 * dx_m / dt_s) << " m/s\n";
    } else {
      std::cout << "dt(s): unresolved (set [physical].dt_s or [physical].u0_mps)\n";
    }
  }

  // =========================================================================
  // Interactive User Interface
  // =========================================================================
  float user_depth_R = single_depth_R >= 0.0f ? single_depth_R : 1.5f;
  float user_U0 = U0;
  std::string user_out_dir = out_dir;

  std::cout << "\n======================================================\n";
  std::cout << "      Capsule Free-Surface Interactive Setup            \n";
  std::cout << "======================================================\n";
  
  std::cout << "Enter the depth ratio of the capsule to the liquid surface (h/R), suggested range [1.0 ~ 5.0] (default " << user_depth_R << "): ";
  std::string input;
  std::getline(std::cin, input);
  if (!input.empty()) {
      try { user_depth_R = std::stof(input); } catch (...) {}
  }
  
  std::cout << "Enter the lattice velocity of the capsule (U0), suggested range [0.04 ~ 0.10] (default " << user_U0 << "): ";
  std::getline(std::cin, input);
  if (!input.empty()) {
      try { user_U0 = std::stof(input); } catch (...) {}
  }
  
  std::cout << "Enter the output directory base path (e.g. out/test_case) (default " << user_out_dir << "): ";
  std::getline(std::cin, input);
  if (!input.empty()) {
      user_out_dir = input;
  }
  
  std::cout << "Enter the total number of simulation steps (default 8000): ";
  int user_steps = 8000;
  std::getline(std::cin, input);
  if (!input.empty()) {
      try { user_steps = std::stoi(input); } catch (...) {}
  }
  
  // Create Timestamp string
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
  
  // Format the sub-folder name
  std::stringstream fdr;
  fdr << user_out_dir << "/depth" << user_depth_R 
      << "_vol" << user_U0 << "_" << ss.str();
  user_out_dir = fdr.str();
  
  while (true) {
      try {
          fs::create_directories(user_out_dir);
          if (fs::exists(user_out_dir) && fs::is_directory(user_out_dir)) {
              std::cout << "Output directory created: " << user_out_dir << "\n";
              break;
          } else {
              std::cout << "Failed to create directory. Path: " << user_out_dir << "\n";
              break; // break to avoid infinite loop on permission issue
          }
      } catch (const std::exception& e) {
          std::cout << "Error creating directory: " << e.what() << "\n";
          break;
      }
  }
  
  std::cout << "\n[Starting Simulation]\n"
            << "  · Depth (h/R) = " << user_depth_R << "\n"
            << "  · Velocity (U0) = " << user_U0 << "\n"
            << "  · Steps = " << user_steps << "\n"
            << "  · Output Dir = " << user_out_dir << "\n\n";

  run_simulation(nx, ny, nz, tau, user_U0, fluid_fraction, capsule_x_ratio,
                 capsule_depth_ratio, user_depth_R, user_out_dir, user_steps, output_every,
                 stl_path, spacing_req, mdf_iter, beta, angle, scale, true);

  return 0;
}
