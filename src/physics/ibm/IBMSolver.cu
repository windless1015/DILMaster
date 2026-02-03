/**
 * @file IBMSolver.cu
 * @brief IBM 求解器实现
 */

#include "../../core/FieldStore.hpp"
#include "IBMCore.hpp"
#include "IBMSolver.hpp"
#include <cmath>
#include <cstring>

IBMSolver::IBMSolver() = default;
IBMSolver::~IBMSolver() = default;

void IBMSolver::configure(const ConfigNode & /*node*/) {
  // 注意：physics 库不链接 yaml-cpp
  // 配置应该由上层通过 setConfig() 等方法传入
}

void IBMSolver::setPrefix(const std::string& prefix) {
    prefix_ = prefix;
}

void IBMSolver::setGravity(float gx, float gy, float gz) {
    desired_gravity_ = make_float3(gx, gy, gz);
    if (motion_) {
        motion_->setGravity(desired_gravity_);
    }
}

void IBMSolver::setConfig(const IBMConfig &config) { 
    config_ = config;

    // Apply to Motion
    if (motion_) {
        if (config_.motion_type == IBMMotionType::ROTATION) {
            motion_->setMotionType(MotionType::FIXED_ROTATION);
            motion_->setRotationAxis(
                make_float3(config_.rotation_axis_x, config_.rotation_axis_y, config_.rotation_axis_z), 
                config_.angular_velocity
            );
        } else if (config_.motion_type == IBMMotionType::RIGID_BODY) {
             motion_->setMotionType(MotionType::FREE_FALL);
        } else {
             motion_->setMotionType(MotionType::FIXED);
        }
    }

    // Recompute initial_rel_pos if ROTATION
    // Note: We use host_markers_ which holds the CURRENT (initially set) positions
    if (config_.motion_type == IBMMotionType::ROTATION && !host_markers_.empty()) {
       initial_rel_pos_.resize(host_markers_.size());
       float3 center = make_float3(config_.rotation_center_x, 
                                   config_.rotation_center_y, 
                                   config_.rotation_center_z);
       for(size_t i=0; i<host_markers_.size(); ++i) {
           initial_rel_pos_[i] = host_markers_[i] - center;
       }
    }
}

void IBMSolver::allocate(StepContext &ctx) {
  // ===========================================================================
  // 字段分配 - 供耦合策略（如 IBMToLBMStrategy）通过 FieldStore 访问
  // ===========================================================================
  //
  // 【设计原则】
  // 1. 不暴露 IBMCore 指针，保持封装性
  // 2. 所有耦合数据通过 FieldStore 字段交换
  // 3. 耦合策略通过 ctx.fields->get(IBMFields::FORCE) 等方式访问
  //
  // 【耦合依赖关系】
  //
  //   ┌─────────────────────────────────────────────────────────────────────┐
  //   │                        耦合策略字段依赖                              │
  //   ├───────────────────────┬───────────────────┬─────────────────────────┤
  //   │ 策略                   │ 读取字段           │ 写入字段 │
  //   ├───────────────────────┼───────────────────┼─────────────────────────┤
  //   │ IBMToLBMStrategy      │ ibm.markers       │ ibm.force               │
  //   │                       │ ibm.velocity      │                         │
  //   │                       │ ibm.normals       │                         │
  //   ├───────────────────────┼───────────────────┼─────────────────────────┤
  //   │ LBMToIBMStrategy      │ ibm.force         │ (应用到 LBM 边界)        │
  //   │                       │ ibm.markers       │                         │
  //   ├───────────────────────┼───────────────────┼─────────────────────────┤
  //   │ IBMToDEMStrategy      │ ibm.markers       │ particles.force         │
  //   │ (颗粒用 IBM 表示)      │ ibm.force         │                         │
  //   └───────────────────────┴───────────────────┴─────────────────────────┘
  //
  // 【字段详情】
  // ┌─────────────────────┬─────────┬──────────────────────────────────────┐
  // │ 字段名               │ 类型    │ 用途                                  │
  // ├─────────────────────┼─────────┼──────────────────────────────────────┤
  // │ ibm.markers         │ float3  │ 标记点位置，LBM 插值速度使用           │
  // │ ibm.velocity        │ float3  │ 标记点速度，LBM 设置边界条件           │
  // │ ibm.force           │ float3  │ 流体力，LBM 计算后写入                 │
  // │ ibm.normals         │ float3  │ 表面法向量，力计算参考                 │
  // │ ibm.area            │ float   │ 标记点面积，力积分权重                 │
  // │ ibm.solid_mask      │ uint8_t │ 固体标识，LBM 边界条件                 │
  // └─────────────────────┴─────────┴──────────────────────────────────────┘
  //
  // 【使用示例 - IBMToLBMStrategy】
  //
  //   void IBMToLBMStrategy::couple(StepContext& ctx) {
  //     // 读取标记点数据
  //     auto markers = ctx.fields->get(IBMFields::MARKERS);
  //     auto velocity = ctx.fields->get(IBMFields::VELOCITY);
  //     float3* pos = static_cast<float3*>(markers.data());
  //     float3* vel = static_cast<float3*>(velocity.data());
  //
  //     // 计算并写入流体力
  //     auto force = ctx.fields->get(IBMFields::FORCE);
  //     float3* forces = static_cast<float3*>(force.data());
  //     // ... 计算力 ...
  //   }
  //
  // ===========================================================================

  if (!ctx.fields) {
    return;
  }

  const std::size_t n = config_.num_markers;
  if (n == 0) {
    return;
  }

  // ---------------------------------------------------------------------------
  // 核心耦合字段（必需）
  // ---------------------------------------------------------------------------

  // 标记点位置 - IBMToLBMStrategy 读取用于速度插值
  ctx.fields->create(FieldDesc{
      prefix_ + ".markers", n,
      sizeof(float) * 3 // float3
  });

  // 标记点速度 - LBM 边界条件设置使用
  ctx.fields->create(FieldDesc{prefix_ + ".velocity", n, sizeof(float) * 3});

  // 流体作用力 - IBMToLBMStrategy 计算后写入
  ctx.fields->create(FieldDesc{prefix_ + ".force", n, sizeof(float) * 3});

  // ---------------------------------------------------------------------------
  // 辅助字段
  // ---------------------------------------------------------------------------

  // 表面法向量 - 用于力方向计算
  ctx.fields->create(FieldDesc{prefix_ + ".normals", n, sizeof(float) * 3});

  // 标记点面积 - 用于力积分
  ctx.fields->create(FieldDesc{prefix_ + ".area", n, sizeof(float)});

  fields_allocated_ = true;

  // ---------------------------------------------------------------------------
  // 创建 IBMCore（内部使用，不暴露给 Coupler）
  // ---------------------------------------------------------------------------
  ibm::IBMParams params;
  params.nMarkers = n;
  params.stencil_width = config_.stencil_width;
  core_ = std::make_unique<ibm::IBMCore>(params);
}

void IBMSolver::initialize(StepContext &ctx) {
  // 1. 加载/生成几何
  // 1. 加载/生成几何
  if (host_markers_.empty()) {
      // 尝试从 FieldStore 加载
      if (ctx.fields && ctx.fields->exists(prefix_ + ".markers")) {
          auto marker_field = ctx.fields->get(prefix_ + ".markers");
          // Validate
    // marker_field.checkType<float3>();
          
          size_t num_mc = marker_field.count();
          
          host_markers_.resize(num_mc);
          std::memcpy(host_markers_.data(), marker_field.data(), marker_field.size_bytes());
          
          // Also init velocities
          host_velocities_.resize(num_mc, make_float3(0,0,0));
          
          std::cout << "IBMSolver: Loaded " << num_mc << " markers from FieldStore." << std::endl;
          setNumMarkers(num_mc);
      } else {
          // 默认放一个点
          host_markers_.push_back(make_float3(config_.rotation_center_x, config_.rotation_center_y, config_.rotation_center_z));
          host_velocities_.push_back(make_float3(0,0,0));
          setNumMarkers(1);
      }
  } else {
      setNumMarkers(host_markers_.size());
  }

  // 2. 初始化 IBM Core (CUDA)
  if (!core_) {
      ibm::IBMParams core_params;
      core_params.nMarkers = host_markers_.size();
      core_params.stencil_width = config_.stencil_width;
      // Set other params if needed
      
      core_ = std::make_unique<ibm::IBMCore>(core_params);
  }
  core_->initialize();
  
  // 3. 将 host_markers_ 同步到 markers_ (IBMMarker)
  markers_.resize(host_markers_.size());
  for(size_t i=0; i<host_markers_.size(); ++i) {
      markers_[i].pos = host_markers_[i];
      markers_[i].u_current = host_velocities_[i];
      markers_[i].force = make_float3(0,0,0);
      markers_[i].rel_pos = make_float3(0,0,0); // Temporary
  }

  // 4. 初始化 Motion 模块
  if (!motion_) {
      motion_ = std::make_unique<IBMMotion>();
  }
  MassProperties props;
  props.mass = config_.mass;
  props.Ixx = props.Iyy = props.Izz = config_.moment_of_inertia;
  // 初始位置设为旋转中心或第一个点
  float3 startPos = make_float3(config_.rotation_center_x, config_.rotation_center_y, config_.rotation_center_z);
  
  motion_->initialize(props, startPos);
  
  if (config_.motion_type == IBMMotionType::ROTATION) {
      motion_->setMotionType(MotionType::FIXED_ROTATION);
      motion_->setRotationAxis(make_float3(config_.rotation_axis_x, config_.rotation_axis_y, config_.rotation_axis_z), config_.angular_velocity);
  } else if (config_.motion_type == IBMMotionType::RIGID_BODY) {
      motion_->setMotionType(MotionType::FREE_FALL); 
  } else {
      motion_->setMotionType(MotionType::FIXED);
  }
  
  // Apply gravity
  motion_->setGravity(desired_gravity_);

  // 5. 初始化初始相对位置 (Body Frame)
  // [FIX] Always initialize for ROTATION, using the recently loaded host_markers_
  if (config_.motion_type == IBMMotionType::ROTATION && !host_markers_.empty()) {
       initial_rel_pos_.resize(host_markers_.size());
       float3 center = make_float3(config_.rotation_center_x, 
                                   config_.rotation_center_y, 
                                   config_.rotation_center_z);
       if (initial_rel_pos_.size() != host_markers_.size()) {
           initial_rel_pos_.resize(host_markers_.size());
       }
       for(size_t i=0; i<host_markers_.size(); ++i) {
           initial_rel_pos_[i] = host_markers_[i] - center;
       }
       
       std::cout << "[IBMSolver] Initialized " << initial_rel_pos_.size() 
                 << " relative positions for rotation." << std::endl;
  }

  // 6. 上传到 GPU
  if (core_) {
      core_->uploadPositions(host_markers_.data());
      core_->uploadVelocities(host_velocities_.data());
  }

  current_time_ = 0.0;
}

void IBMSolver::step(StepContext &ctx) {
    if (!motion_) return;

    // 0. Apply forces from coupling (if any)
    applyForcesFromFieldStore(ctx);

    // 1. Update Kinematics (computes new positions/velocities on CPU)
    stepKinematics(ctx.dt);
    
    // 2. Write updated markers back to FieldStore (for VTK & Coupling)
    if (ctx.fields && ctx.fields->exists(prefix_ + ".markers")) {
        auto marker_field = ctx.fields->get(prefix_ + ".markers");
        if (marker_field.count() == host_markers_.size()) {
             std::memcpy(marker_field.data(), host_markers_.data(), 
                         host_markers_.size() * sizeof(float3));
        }
    }
    
    // 3. Write updated velocities back to FieldStore (for VTK & Coupling)
    if (ctx.fields) {
        // Ensure velocity field exists (usually created by Strategy or Init)
        if (!ctx.fields->exists(prefix_ + ".velocity") && !host_velocities_.empty()) {
            ctx.fields->create(FieldDesc{prefix_ + ".velocity", host_velocities_.size(), sizeof(float) * 3});
        }
        
        if (ctx.fields->exists(prefix_ + ".velocity")) {
            auto vel_field = ctx.fields->get(prefix_ + ".velocity");
            if (vel_field.count() == host_velocities_.size()) {
                 std::memcpy(vel_field.data(), host_velocities_.data(), 
                             host_velocities_.size() * sizeof(float3));
                 
                 // Debug first marker velocity
                 /*
                 if (!host_velocities_.empty() && ctx.step % 1000 == 0) {
                     float3 v = host_velocities_[0];
                     std::cout << "[IBMSolver] " << prefix_ << " write vel[0]=(" << v.x << "," << v.y << "," << v.z << ")" << std::endl;
                 }
                 */
            } else {
                 std::cout << "[IBMSolver] ERROR: Velocity field size mismatch! Field=" << vel_field.count() << " Host=" << host_velocities_.size() << std::endl;
            }
        } else {
             std::cout << "[IBMSolver] ERROR: Velocity field " << prefix_ << ".velocity not found!" << std::endl;
        }
    }
}

void IBMSolver::applyForcesFromFieldStore(StepContext &ctx) {
    if (!ctx.fields->exists(prefix_ + ".force")) return;
    
    auto force_handle = ctx.fields->get(prefix_ + ".force");
// force_handle.checkType<float3>();
    // force_handle.checkCount(markers_.size());

    const float3* forces = force_handle.as<float3>();
    size_t n = markers_.size();
    
    // 更新 markers_ 中的力
    for(size_t i=0; i<n; ++i) {
        markers_[i].force = forces[i];
        // 更新 rel_pos 用于力矩计算 (r = x - x_cm)
        markers_[i].rel_pos = markers_[i].pos - motion_->getState().position;
    }
    
    motion_->updateForces(markers_);
}

void IBMSolver::finalize(StepContext &ctx) {
  core_.reset();
  fields_allocated_ = false;
  (void)ctx;
}

void IBMSolver::stepKinematics(double dt) {
    if (!motion_) return;

    if (config_.motion_type == IBMMotionType::ROTATION) {
        // 1. 积分旋转状态 (Orientation)
        motion_->integrate((float)dt);
        const auto& state = motion_->getState();
        
        // 2. 根据初始相对位置和当前 Orientation 更新绝对位置
        // Pos = Center + Rotate(Rel_Body, Q)
        // 这种方法避免了欧拉积分的漂移
        float3 center = state.position; // 对于 ROTATION，position 是旋转中心
        float3 omega = state.angular_velocity;
        
        static int debug_k = 0;
        if (debug_k++ % 100 == 0) {
            std::cout << "[IBMSolver::stepKinematics] Step=" << debug_k 
                      << " Omega=(" << omega.x << "," << omega.y << "," << omega.z << ")"
                      << " Center=(" << center.x << "," << center.y << "," << center.z << ")"
                      << std::endl;
        }

        for(size_t i=0; i<markers_.size(); ++i) {
            // (1) Update Position (Exact)
            float3 r_world = motion_->rotateVector(initial_rel_pos_[i], state.orientation);
            float3 new_pos = center + r_world;
            
            markers_[i].pos = new_pos;
            markers_[i].rel_pos = r_world; // World frame relative pos for Torque calc if needed
            
            // (2) Update Velocity: v = omega x r_world
            float3 vel = motion_->crossProduct(omega, r_world);
            markers_[i].u_desired = vel;
            
            // Sync to host buffers
            host_markers_[i] = new_pos;
            host_velocities_[i] = vel;
        }
    } 
    else if (config_.motion_type == IBMMotionType::RIGID_BODY) {
        // ... (Existing rigid body logic)
        motion_->integrate((float)dt);
        motion_->updateMarkerVelocities(markers_);
        const auto& state = motion_->getState();
        for(size_t i=0; i<markers_.size(); ++i) {
             float3 vel = markers_[i].u_desired;
             markers_[i].pos = markers_[i].pos + vel * (float)dt;
             host_markers_[i] = markers_[i].pos;
             host_velocities_[i] = markers_[i].u_desired;
        }
    } 
    else {
        // FIXED / TRANSLATION
        if (config_.translation_velocity_x != 0 || 
            config_.translation_velocity_y != 0 || 
            config_.translation_velocity_z != 0) {
             updateTranslation(dt);
        }
    }
    
    // 上传
    if (core_) {
        core_->uploadPositions(host_markers_.data());
        core_->uploadVelocities(host_velocities_.data());
    }
}

void IBMSolver::updateRotation(double dt) {
  // 绕轴旋转更新 (使用 AoS float3 格式)
  if (host_markers_.empty() || core_ == nullptr) {
    return;
  }

  const float omega = config_.angular_velocity;
  const float dtheta = static_cast<float>(omega * dt);
  const float cos_t = std::cos(dtheta);
  const float sin_t = std::sin(dtheta);

  // 旋转轴（归一化）
  float ax = config_.rotation_axis_x;
  float ay = config_.rotation_axis_y;
  float az = config_.rotation_axis_z;
  const float len = std::sqrt(ax * ax + ay * ay + az * az);
  if (len > 0) {
    ax /= len;
    ay /= len;
    az /= len;
  }

  // 旋转中心
  const float cx = config_.rotation_center_x;
  const float cy = config_.rotation_center_y;
  const float cz = config_.rotation_center_z;

  const std::size_t n = config_.num_markers;

  // 使用 Rodrigues 旋转公式更新每个标记点
  for (std::size_t i = 0; i < n; ++i) {
    // 相对于旋转中心的位置
    float px = host_markers_[i].x - cx;
    float py = host_markers_[i].y - cy;
    float pz = host_markers_[i].z - cz;

    // Rodrigues 旋转
    float dot = ax * px + ay * py + az * pz;
    float cross_x = ay * pz - az * py;
    float cross_y = az * px - ax * pz;
    float cross_z = ax * py - ay * px;

    float new_px = px * cos_t + cross_x * sin_t + ax * dot * (1 - cos_t);
    float new_py = py * cos_t + cross_y * sin_t + ay * dot * (1 - cos_t);
    float new_pz = pz * cos_t + cross_z * sin_t + az * dot * (1 - cos_t);

    // 更新位置 (AoS)
    host_markers_[i].x = new_px + cx;
    host_markers_[i].y = new_py + cy;
    host_markers_[i].z = new_pz + cz;

    // 计算速度 (v = omega × r)
    host_velocities_[i].x = omega * (ay * pz - az * py);
    host_velocities_[i].y = omega * (az * px - ax * pz);
    host_velocities_[i].z = omega * (ax * py - ay * px);
  }

  // 上传到 GPU
  if (core_) {
    core_->uploadPositions(host_markers_.data());
    core_->uploadVelocities(host_velocities_.data());
  }
}

void IBMSolver::updateTranslation(double dt) {
  if (host_markers_.empty()) {
    return;
  }

  const std::size_t n = config_.num_markers;
  const float dx = config_.translation_velocity_x * static_cast<float>(dt);
  const float dy = config_.translation_velocity_y * static_cast<float>(dt);
  const float dz = config_.translation_velocity_z * static_cast<float>(dt);

  for (std::size_t i = 0; i < n; ++i) {
    host_markers_[i].x += dx;
    host_markers_[i].y += dy;
    host_markers_[i].z += dz;

    host_velocities_[i].x = config_.translation_velocity_x;
    host_velocities_[i].y = config_.translation_velocity_y;
    host_velocities_[i].z = config_.translation_velocity_z;
  }

  if (core_) {
    core_->uploadPositions(host_markers_.data());
    core_->uploadVelocities(host_velocities_.data());
  }
}

void IBMSolver::updateOscillation(double dt) {
  // TODO: 实现振荡运动
  (void)dt;
}

void IBMSolver::setMarkerPositions(const float *positions, std::size_t count) {
  if (count == 0) return;

  config_.num_markers = count;
  host_markers_.resize(count);
  host_velocities_.resize(count);
  markers_.resize(count); // Sync internal markers vector

  // 1. Update Host Buffers & Internal Markers
  for (std::size_t i = 0; i < count; ++i) {
    float3 pos = make_float3(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
    host_markers_[i] = pos;
    host_velocities_[i] = make_float3(0.0f, 0.0f, 0.0f); // Reset velocity or keep? usually clean start
    
    // Update internal IBMMarker
    markers_[i].pos = pos;
    markers_[i].u_current = make_float3(0,0,0);
    markers_[i].u_desired = make_float3(0,0,0);
    markers_[i].force = make_float3(0,0,0);
  }

  // 2. Recompute Initial Relative Positions for Rotation
  // This is CRITICAL: otherwise rotation is applied to old/wrong relative positions
  if (config_.motion_type == IBMMotionType::ROTATION) {
       initial_rel_pos_.resize(count);
       float3 center = make_float3(config_.rotation_center_x, 
                                   config_.rotation_center_y, 
                                   config_.rotation_center_z);
       
       for(size_t i=0; i<count; ++i) {
           initial_rel_pos_[i] = host_markers_[i] - center;
           markers_[i].rel_pos = initial_rel_pos_[i]; // Also sync current rel_pos
       }
  }

  // 3. Upload to GPU
  if (core_) {
      // Re-initialize core to ensure buffers are correct size
      ibm::IBMParams params;
      params.nMarkers = count;
      params.stencil_width = config_.stencil_width;
      core_ = std::make_unique<ibm::IBMCore>(params);
      core_->initialize();
      
      core_->uploadPositions(host_markers_.data());
      core_->uploadVelocities(host_velocities_.data());
  }
}

void IBMSolver::computeTotalForceAndTorque(float *total_force,
                                           float *total_torque) const {
  // TODO: 实现总力和力矩计算
  if (total_force) {
    total_force[0] = total_force[1] = total_force[2] = 0.0f;
  }
  if (total_torque) {
    total_torque[0] = total_torque[1] = total_torque[2] = 0.0f;
  }
}
