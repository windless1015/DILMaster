#include "../../core/FieldStore.hpp"
#include "DEMCore.hpp"
#include "DEMSolver.hpp"
#include <iostream>

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

  // 1. Initialize FieldStore data (if not already set by Scenario)
  auto posF = ctx.fields->get(DEMFields::POSITION);
  auto radF = ctx.fields->get(DEMFields::RADIUS);
  
  float* h_radius = static_cast<float*>(radF.data());
  // Set default radius if zero (simple heuristic) or just trust config
  for (size_t i = 0; i < config_.num_particles; ++i) {
      if (h_radius[i] <= 0.0f) h_radius[i] = config_.particle_radius;
  }

  // 2. Upload initial state from FieldStore to GPU
  // Assumption: Positions have been set by the Scenario/Generator
  core_->uploadPositions(static_cast<float*>(posF.data()));
  core_->uploadVelocities(static_cast<float*>(ctx.fields->get(DEMFields::VELOCITY).data()));
  core_->uploadRadii(h_radius);

  // 3. Compute derived mass properties on GPU
  core_->initMassProperties();
  
  // 4. Ensure no initial nans
  core_->checkHealth();
}

void DEMSolver::step(StepContext &ctx) {
  if (!core_) return;

  // 1. Sync Coupling Forces: FieldStore (Host) -> DEMCore (Device)
  // We need to upload EXTERNAL forces that were reduced onto particles (e.g. drag)
  // The DEMCore clears forces at start of step, BUT if we want to add external forces,
  // we should do it cautiously. 
  // Code design: DEMCore::step clears forces. So we can't upload forces *before* step 
  // unless we modify DEMCore to accept external forces or not clear them.
  // 
  // Correct approach per DEMCore.hpp:
  // "Use clearForcesTorquePublic() if coupling code wants to write external forces before step()"
  
  auto forceF = ctx.fields->get(DEMFields::FORCE);
  float* h_force = static_cast<float*>(forceF.data());
  
  // Optional: If coupling is active, we might need to copy these to device
  // However, standard DEMCore::step() does [Clear -> AddGravity -> BroadPhase -> Contacts...]
  // If we upload forces now, they will be cleared by step().
  // 
  // We need to clarify the interface. For now, assuming standard DEM only.
  // If coupling is needed later, we'd inject it between Clear and Integrate.
  // 
  // Since request says "MVP... no LBM dependencies", we focus on internal DEM physics.
  // We will pull the latest positions/velocities from GPU to Host for verification/output.
  
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
  
  core_->downloadPositions(static_cast<float*>(posF.data()));
  core_->downloadVelocities(static_cast<float*>(velF.data()));
  
  // Also download forces for visualization?
  core_->downloadForces(static_cast<float*>(forceF.data()));
}

void DEMSolver::finalize(StepContext &ctx) {
  if (core_) {
    core_->synchronize();
    core_.reset(); // Free GPU memory
  }
  fields_allocated_ = false;
  (void)ctx;
}

