#include "../../core/FieldStore.hpp"
#include "../../core/ArrayLayoutConverter.hpp"
#include "DEMCore.hpp"
#include "DEMSolver.hpp"
#include <iostream>
#include <vector>

DEMSolver::DEMSolver() = default;
DEMSolver::~DEMSolver() = default;

void DEMSolver::configure(const ConfigNode & /*node*/) {
  // Stub: in a real app this would parse YAML
  // For now, relies on setConfig() or manual config injection
}

void DEMSolver::setConfig(const DEMConfig &config) { config_ = config; }

void DEMSolver::allocate(StepContext &ctx) {
  if (!ctx.fields) return;

  const std::size_t n = config_.num_particles;
  if (n == 0) return;

  // 1. Create fields in FieldStore if they don't exist
  // We use create_or_get semantics implicitly by checking before creating if needed,
  // but FieldStore::create usually throws if exists. Assuming standard behavior:
  
  auto create_field = [&](const char* name, int components) {
      if (!ctx.fields->exists(name)) {
          ctx.fields->create(FieldDesc{name, n, sizeof(float) * components});
      }
  };

  create_field(DEMFields::POSITION, 3);
  create_field(DEMFields::VELOCITY, 3);
  create_field(DEMFields::FORCE, 3);
  create_field(DEMFields::RADIUS, 1);
  // Optional
  create_field(DEMFields::DENSITY, 1);

  fields_allocated_ = true;

  // 2. Prepare DEMCore config
  dem::DEMConfig core_cfg;
  core_cfg.num_particles = config_.num_particles;
  core_cfg.particle_radius = config_.particle_radius;
  core_cfg.particle_density = config_.particle_density;
  
  core_cfg.material.kn = config_.kn;
  core_cfg.material.kt = config_.kt;
  core_cfg.material.mu = config_.friction;
  core_cfg.material.restitution = config_.restitution;
  
  core_cfg.gravity_x = config_.gravity_x;
  core_cfg.gravity_y = config_.gravity_y;
  core_cfg.gravity_z = config_.gravity_z;

  core_cfg.domain_min_x = config_.domain_min_x;
  core_cfg.domain_min_y = config_.domain_min_y;
  core_cfg.domain_min_z = config_.domain_min_z;
  core_cfg.domain_max_x = config_.domain_max_x;
  core_cfg.domain_max_y = config_.domain_max_y;
  core_cfg.domain_max_z = config_.domain_max_z;

  // 3. Instantiate and Allocate DEMCore
  core_ = std::make_unique<dem::DEMCore>();
  core_->configure(core_cfg);
  core_->allocate();
}

void DEMSolver::initialize(StepContext &ctx) {
  if (!fields_allocated_ || !core_) return;
  const std::size_t n = config_.num_particles;

  // 1. Initialize FieldStore data (if not already set by Scenario)
  auto posF = ctx.fields->get(DEMFields::POSITION);
  auto velF = ctx.fields->get(DEMFields::VELOCITY);
  auto radF = ctx.fields->get(DEMFields::RADIUS);
  
  float* h_radius = static_cast<float*>(radF.data());
  // Set default radius if zero (simple heuristic) or just trust config
  for (size_t i = 0; i < config_.num_particles; ++i) {
      if (h_radius[i] <= 0.0f) h_radius[i] = config_.particle_radius;
  }

  // 2. Upload initial state from FieldStore to GPU
  // FieldStore stores float3 AoS, while DEMCore expects SoA [x... y... z...]
  
  std::vector<float3> aos_pos_tmp(n);
  std::vector<float3> aos_vel_tmp(n);
  float3* pRaw = static_cast<float3*>(posF.data());
  float3* vRaw = static_cast<float3*>(velF.data());
  for(size_t i=0; i<n; ++i) { aos_pos_tmp[i] = pRaw[i]; aos_vel_tmp[i] = vRaw[i]; }

  std::vector<float> pos_soa(3 * n);
  std::vector<float> vel_soa(3 * n);
  core::ArrayLayoutConverter::AoSToSoA_float3(aos_pos_tmp, pos_soa.data());
  core::ArrayLayoutConverter::AoSToSoA_float3(aos_vel_tmp, vel_soa.data());

  core_->uploadPositions(pos_soa.data());
  core_->uploadVelocities(vel_soa.data());
  core_->uploadRadii(h_radius);

  // 3. Compute derived mass properties on GPU
  core_->initMassProperties();
  
  // 4. Ensure no initial nans
  core_->checkHealth();
}

void DEMSolver::step(StepContext &ctx) {
  if (!core_) return;
  const std::size_t n = config_.num_particles;

  // 1. Sync Coupling Forces: FieldStore (Host) -> DEMCore (Device)
  // We need to upload EXTERNAL forces that were reduced onto particles (e.g. drag)
  // The DEMCore clears forces at start of step, BUT if we want to add external forces,
  // we should do it cautiously. 
  // Code design: DEMCore::step clears forces. So we can't upload forces *before* step 
  // unless we modify DEMCore to accept external forces or not clear them.
  // 
  // Correct approach per DEMCore.hpp:
  // "Use clearForcesTorquePublic() if coupling code wants to write external forces before step()"
  
  // 1. Sync Coupling Forces: FieldStore (Host) -> DEMCore (Device)
  auto forceF = ctx.fields->get(DEMFields::FORCE);
  std::vector<float3> aos_force_tmp(n);
  float3* fRaw = static_cast<float3*>(forceF.data());
  for(size_t i=0; i<n; ++i) { aos_force_tmp[i] = fRaw[i]; }

  std::vector<float> force_soa(3 * n);
  core::ArrayLayoutConverter::AoSToSoA_float3(aos_force_tmp, force_soa.data());
  
  // Inject into DEMCore (added in refactor)
  core_->uploadExternalForces(force_soa.data());
  
  // 2. Execute Physics (Adaptive Substepping)
  // Textbook accuracy requires resolving the contact duration t_c.
  // Stability typically requires dt < t_c / 10.
  // We use n_sub = ceil(dt / (t_c * safety_factor))
  float t_c = core_->getCriticalTimeStep();
  int substeps = 1;
  const float safety_factor = 0.2f; // Resolve contact with at least 5 steps (conservative)

  if (t_c > 1e-12f) {
      substeps = static_cast<int>(std::ceil(ctx.dt / (t_c * safety_factor)));
  }
  
  if (substeps < 1) substeps = 1;
  
  // Cap at reasonable limit to prevent freeze if stiffness is insane
  if (substeps > 1000) {
      std::cerr << "[DEM WARNING] Excessive substeps requested: " << substeps 
                << " (dt=" << ctx.dt << ", t_c=" << t_c << "). Clamping to 1000.\n";
      substeps = 1000;
  }

  core_->stepMultiple(ctx.dt, substeps);

  // 3. Sync Results: DEMCore (Device) -> FieldStore (Host)
  auto posF = ctx.fields->get(DEMFields::POSITION);
  auto velF = ctx.fields->get(DEMFields::VELOCITY);

  std::vector<float> pos_soa(3 * n, 0.0f);
  std::vector<float> vel_soa(3 * n, 0.0f);
  std::vector<float> frc_soa(3 * n, 0.0f);
  core_->downloadPositions(pos_soa.data());
  core_->downloadVelocities(vel_soa.data());
  core_->downloadForces(frc_soa.data());

  auto aos_pos = core::ArrayLayoutConverter::SoAToAoS_float3(pos_soa.data(), n);
  auto aos_vel = core::ArrayLayoutConverter::SoAToAoS_float3(vel_soa.data(), n);
  auto aos_frc = core::ArrayLayoutConverter::SoAToAoS_float3(frc_soa.data(), n);

  float3* pRaw = static_cast<float3*>(posF.data());
  float3* vRaw = static_cast<float3*>(velF.data());
  fRaw = static_cast<float3*>(forceF.data());
  for(size_t i=0; i<n; ++i) {
      pRaw[i] = aos_pos[i];
      vRaw[i] = aos_vel[i];
      fRaw[i] = aos_frc[i];
  }
}

void DEMSolver::finalize(StepContext &ctx) {
  if (core_) {
    core_->synchronize();
    core_.reset(); // Free GPU memory
  }
  fields_allocated_ = false;
  (void)ctx;
}

