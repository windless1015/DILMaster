#pragma once
#include "../../src/core/FieldStore.hpp"
#include "../../src/core/StepContext.hpp"
#include "../../src/physics/dem/DEMConfig.hpp"
#include "../../src/physics/dem/DEMCore.hpp" // Added for getCore()
#include "../../src/physics/dem/DEMSolver.hpp"
#include "../../src/physics/ibm/IBMCore.hpp" // Added for getCore()
#include "../../src/physics/ibm/IBMSolver.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <vector>


class IBMDEMCollisionScenario {
public:
  struct Config {
    std::string stl_file = "../tools/propeller_no_stick.stl";
    float marker_spacing = 1.0f;
    float particle_radius = 0.01f;
    float particle_rho = 2500.0f;
    float stiffness = 5.0e4f;
    float damping = 5.0f;
    float dem_kn = 5.0e6f;
    float dem_restitution = 1.0f;
    float influence_radius = 0.002f;
    float initial_gap_factor = 0.5f; // particle offset = (R + influence) * factor
  };

  IBMDEMCollisionScenario(const Config &cfg) : cfg_(cfg) {}

  void setup(IBMSolver &ibm, DEMSolver &dem, StepContext &ctx) {
    setupIBM(ibm);

    // Two-pass IBM init:
    // 1) initialize() loads STL and determines marker count
    ibm.initialize(ctx);
    normalizeMarkersToUnitDomain(ibm);
    // 2) allocate() creates FieldStore fields with correct count
    // 3) initialize() uploads marker data into allocated buffers/fields
    ibm.allocate(ctx);
    ibm.initialize(ctx);

    setupDEM(dem);
    dem.allocate(ctx);
    dem.initialize(ctx);

    // Place particle close to propeller surface to guarantee collision force.
    auto posF = ctx.fields->get(DEMFields::POSITION);
    float3 *pos = posF.as<float3>();
    float3 center = make_float3(kCenterX, kCenterY, kCenterZ);
    float3 seed = make_float3(0.6f, 0.5f, 0.5f);

    const auto &markers = ibm.getMarkers();
    if (!markers.empty()) {
      // Use tip-side marker (max radial distance in XY) to avoid dead zone near hub.
      float max_r2 = -1.0f;
      float3 tip = markers[0].pos;
      for (const auto &mk : markers) {
        float dx = mk.pos.x - center.x;
        float dy = mk.pos.y - center.y;
        float r2 = dx * dx + dy * dy;
        if (r2 > max_r2) {
          max_r2 = r2;
          tip = mk.pos;
        }
      }

      float3 n = make_float3(tip.x - center.x, tip.y - center.y, tip.z - center.z);
      float nlen = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
      if (nlen < 1e-8f) {
        n = make_float3(1.0f, 0.0f, 0.0f);
      } else {
        n.x /= nlen;
        n.y /= nlen;
        n.z /= nlen;
      }

      float offset =
          (cfg_.particle_radius + cfg_.influence_radius) * cfg_.initial_gap_factor;
      seed = make_float3(tip.x + n.x * offset, tip.y + n.y * offset, tip.z + n.z * offset);
    }
    pos[0] = seed;
    std::cout << "[IBMDEMCollisionScenario] Initial particle position = ("
              << pos[0].x << ", " << pos[0].y << ", " << pos[0].z << ")"
              << std::endl;

    // Initial velocity zero
    auto velF = ctx.fields->get(DEMFields::VELOCITY);
    float3 *vel = velF.as<float3>();
    vel[0] = make_float3(0, 0, 0);

    dem.getCore()->uploadPositions(reinterpret_cast<float *>(pos));
    dem.getCore()->uploadVelocities(reinterpret_cast<float *>(vel));
  }

private:
  Config cfg_;
  static constexpr float kCenterX = 0.5f;
  static constexpr float kCenterY = 0.5f;
  static constexpr float kCenterZ = 0.5f;

  void normalizeMarkersToUnitDomain(IBMSolver &ibm) {
    const auto &markers = ibm.getMarkers();
    if (markers.empty()) return;

    float min_x = markers[0].pos.x, max_x = markers[0].pos.x;
    float min_y = markers[0].pos.y, max_y = markers[0].pos.y;
    float min_z = markers[0].pos.z, max_z = markers[0].pos.z;
    for (const auto &mk : markers) {
      min_x = std::min(min_x, mk.pos.x);
      min_y = std::min(min_y, mk.pos.y);
      min_z = std::min(min_z, mk.pos.z);
      max_x = std::max(max_x, mk.pos.x);
      max_y = std::max(max_y, mk.pos.y);
      max_z = std::max(max_z, mk.pos.z);
    }

    const float cx = 0.5f * (min_x + max_x);
    const float cy = 0.5f * (min_y + max_y);
    const float cz = 0.5f * (min_z + max_z);
    const float lx = std::max(1e-6f, max_x - min_x);
    const float ly = std::max(1e-6f, max_y - min_y);
    const float lz = std::max(1e-6f, max_z - min_z);
    const float max_len = std::max(lx, std::max(ly, lz));

    // Fit propeller inside ~40% of domain size to avoid wall-contact artifacts.
    const float scale = 0.4f / max_len;

    std::vector<float> flat(3 * markers.size(), 0.0f);
    for (size_t i = 0; i < markers.size(); ++i) {
      flat[3 * i + 0] = kCenterX + (markers[i].pos.x - cx) * scale;
      flat[3 * i + 1] = kCenterY + (markers[i].pos.y - cy) * scale;
      flat[3 * i + 2] = kCenterZ + (markers[i].pos.z - cz) * scale;
    }

    ibm.setRotation(0, 0, 1, kCenterX, kCenterY, kCenterZ, 10.0f);
    ibm.setMarkerPositions(flat.data(), markers.size());
    std::cout << "[IBMDEMCollisionScenario] Normalized markers to unit domain, scale="
              << scale << std::endl;
  }

  static std::string resolveSTLPath(const std::string &path) {
    namespace fs = std::filesystem;
    const fs::path p(path);
    if (fs::exists(p)) return p.string();

    const fs::path cwd = fs::current_path();
    const fs::path alt1 = cwd / ".." / path;          // build/<cfg> -> repo relative
    const fs::path alt2 = cwd / ".." / ".." / path;   // build/<cfg>/<subdir> -> repo relative
    const fs::path alt3 = cwd / ".." / ".." / ".." / path;
    if (fs::exists(alt1)) return alt1.lexically_normal().string();
    if (fs::exists(alt2)) return alt2.lexically_normal().string();
    if (fs::exists(alt3)) return alt3.lexically_normal().string();
    return path;
  }

  void setupIBM(IBMSolver &ibm) {
    std::string full_path = resolveSTLPath(cfg_.stl_file);
    ibm.setSTLFile(full_path);
    ibm.setMarkerSpacing(cfg_.marker_spacing);
    ibm.setMotionType(IBMMotionType::ROTATION);
    ibm.setRotation(0, 0, 1, kCenterX, kCenterY, kCenterZ, 10.0f); // 10 rad/s

    // Fix warning: mdf_beta should be negative
    IBMConfig iCfg;
    iCfg.mdf_beta = -0.5f;
    iCfg.mdf_iterations = 3;
    // Restore motion settings as setConfig might reset them?
    // IBMSolver::setConfig overwrites motion type!
    // So we must set config FIRST, or set config values correctly.
    iCfg.motion_type = IBMMotionType::ROTATION;
    iCfg.angular_velocity = 10.0f;
    iCfg.rotation_axis_z = 1.0f;
    iCfg.rotation_center_x = kCenterX;
    iCfg.rotation_center_y = kCenterY;
    iCfg.rotation_center_z = kCenterZ;
    iCfg.num_markers = 0; // Will be set by STL load
    ibm.setConfig(iCfg);
    // setConfig overrides stl_file from iCfg defaults, so set it again.
    ibm.setSTLFile(full_path);
    ibm.setMarkerSpacing(cfg_.marker_spacing);

    if (!std::filesystem::exists(full_path)) {
      std::cerr << "[IBMDEMCollisionScenario] STL not found: " << full_path
                << std::endl;
    }
  }

  void setupDEM(DEMSolver &dem) {
    DEMConfig dCfg;
    dCfg.num_particles = 1;
    dCfg.particle_radius = cfg_.particle_radius;
    dCfg.particle_density = cfg_.particle_rho;

    dCfg.gravity_x = 0;
    dCfg.gravity_y = 0;
    dCfg.gravity_z = 0; // No gravity
    dCfg.kn = cfg_.dem_kn;
    dCfg.restitution = cfg_.dem_restitution;

    // Use a larger domain to avoid wall-dominated collisions in this validation.
    dCfg.domain_min_x = -1.0f;
    dCfg.domain_max_x = 2.0f;
    dCfg.domain_min_y = -1.0f;
    dCfg.domain_max_y = 2.0f;
    dCfg.domain_min_z = -1.0f;
    dCfg.domain_max_z = 2.0f;

    dem.setConfig(dCfg);
  }
};
