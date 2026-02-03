#include "IBMMotion.hpp"
#include <cmath>

void IBMMotion::initialize(const MassProperties& props, const float3& initialPos) {
    state_.mass = props.mass;
    state_.inertia = make_float3(props.Ixx, props.Iyy, props.Izz);
    state_.position = initialPos;
    
    // Reset dynamics
    state_.velocity = make_float3(0, 0, 0);
    state_.angular_velocity = make_float3(0, 0, 0);
    state_.orientation = Quaternion(0, 0, 0, 1);
    
    totalForce_ = make_float3(0, 0, 0);
    totalTorque_ = make_float3(0, 0, 0);
    gravity_ = make_float3(0, 0, -9.81f);
}

void IBMMotion::setRotationAxis(const float3& axis, float omega) {
    float len = std::sqrt(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
    if (len > 1e-6f) {
        rotationAxis_ = make_float3(axis.x / len, axis.y / len, axis.z / len);
    } else {
        rotationAxis_ = make_float3(0, 0, 1);
    }
    fixedOmega_ = omega;
}

float3 IBMMotion::crossProduct(const float3& a, const float3& b) const {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

void IBMMotion::updateForces(const std::vector<IBMMarker>& markers) {
    // Reset accumulators
    totalForce_ = make_float3(0, 0, 0);
    totalTorque_ = make_float3(0, 0, 0);

    // Sum forces from markers (IBM force is force EXERTED by fluid on marker? 
    // Usually IBM calculates force density on fluid. The force on body is opposite.)
    // Assuming 'markers[i].force' is the force ON THE FLUID.
    // So force ON BODY is -F.
    // However, if the solver stores force ON MARKER, then it's +F.
    // Standard IBM: f_ibm = (u - u_solid) / dt. This drives fluid to solid vel.
    // Force on fluid = f_ibm. Force on solid = -f_ibm.
    // We will assume markers[i].force is the hydrodynamic force ON THE MARKER.
    
    for (const auto& m : markers) {
        // Accumulate force
        totalForce_ = totalForce_ + m.force;
        
        // Accumulate torque: r x F
        // r should be relative to CoM in world frame?
        // m.rel_pos is usually defined in BODY frame or WORLD frame?
        // Typically IBMMarker.rel_pos is vector from CoM to marker in WORLD frame, 
        // or BODY frame rotated to world?
        // Let's assume rel_pos is r_i (vector from CoM to marker).
        // Check updateMarkerVelocities logic to confirm usage.
        
        // Torque += r x F
        float3 torque = crossProduct(m.rel_pos, m.force);
        totalTorque_ = totalTorque_ + torque;
    }
}

void IBMMotion::integrate(float dt) {
    if (type_ == MotionType::FIXED) {
        state_.velocity = make_float3(0,0,0);
        state_.angular_velocity = make_float3(0,0,0);
        return;
    }
    
    if (type_ == MotionType::FIXED_ROTATION) {
        state_.velocity = make_float3(0,0,0);
        // Angular velocity is constant aligned with axis
        state_.angular_velocity = rotationAxis_ * fixedOmega_;
        
        // Integrate orientation using quaternion kinematics: dq/dt = 0.5 * omega * q
        Quaternion q = state_.orientation;
        Quaternion w(state_.angular_velocity.x, state_.angular_velocity.y, state_.angular_velocity.z, 0);
        
        // dq = 0.5 * w * q
        Quaternion dq = multiplyQuat(w, q);
        
        q.x += dq.x * 0.5f * dt;
        q.y += dq.y * 0.5f * dt;
        q.z += dq.z * 0.5f * dt;
        q.w += dq.w * 0.5f * dt;
        
        // Normalize
        float norm = std::sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
        state_.orientation = Quaternion(q.x/norm, q.y/norm, q.z/norm, q.w/norm);
        
        return;
    }
    
    if (type_ == MotionType::FREE_FALL) {
        // 1. Linear Motion
        // F_total = F_hydro + m * g
        float3 gravityForce = gravity_ * state_.mass;
        float3 netForce = totalForce_ + gravityForce;
        
        float3 acceleration = netForce / state_.mass;
        
        static int debug_motion = 0;
        if (debug_motion++ % 1000 == 0) {
            std::cout << "[IBMMotion] dt=" << dt 
                      << " Mass=" << state_.mass
                      << " G=(" << gravity_.z << ")"
                      << " F_hydro=(" << totalForce_.z << ")"
                      << " F_net=(" << netForce.z << ")"
                      << " Acc=(" << acceleration.z << ")"
                      << " Vel=(" << state_.velocity.z << ")"
                      << " AngVel=(" << state_.angular_velocity.x << "," << state_.angular_velocity.y << "," << state_.angular_velocity.z << ")"
                      << " Pos=(" << state_.position.z << ")"
                      << std::endl;
        }

        // v += a * dt
        state_.velocity = state_.velocity + acceleration * dt;
        // x += v * dt
        state_.position = state_.position + state_.velocity * dt;
        
        // 2. Angular Motion
        // Euler's equations for rigid body rotation
        // I * domega/dt + omega x (I * omega) = Torque
        // For simplicity, we assume diagonal inertia tensor in body frame and ignore gyroscopic terms 
        // OR implement full Newton-Euler.
        // Let's do a simplified approach where we ignore off-diagonal/gyroscopic for stability first, 
        // or just apply torque / I.
        
        // To do it properly:
        // Angular momentum conservation in world frame or body frame.
        // Here we keep everything simpler.
        // alpha = Torque / I (component-wise approximation if aligned)
        
        // Better: Just update angular velocity directly if inertia is isotropic-ish
        // domega/dt = inv(I) * (Torque - omega x (I * omega))
        
        // We will assume I is constant in body frame.
        // Converting torque to body frame is complex without full matrix support.
        // FALLBACK: Simplest Euler integration.
        // alpha = Torque / inertia (assuming roughly spherical or axis-aligned for now for stability)
        
        float3 alpha;
        // Prevent div by zero
        alpha.x = (state_.inertia.x > 0) ? totalTorque_.x / state_.inertia.x : 0;
        alpha.y = (state_.inertia.y > 0) ? totalTorque_.y / state_.inertia.y : 0;
        alpha.z = (state_.inertia.z > 0) ? totalTorque_.z / state_.inertia.z : 0;
        
        state_.angular_velocity = state_.angular_velocity + alpha * dt;
        
        // Integrate quaternion
        // q_new = q + 0.5 * w * q * dt
        Quaternion q = state_.orientation;
        Quaternion w(state_.angular_velocity.x, state_.angular_velocity.y, state_.angular_velocity.z, 0);
        Quaternion dq = multiplyQuat(w, q);
        
        q.x += dq.x * 0.5f * dt;
        q.y += dq.y * 0.5f * dt;
        q.z += dq.z * 0.5f * dt;
        q.w += dq.w * 0.5f * dt;
        
        float norm = std::sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
        state_.orientation = Quaternion(q.x/norm, q.y/norm, q.z/norm, q.w/norm);
    }
}

void IBMMotion::updateMarkerVelocities(std::vector<IBMMarker>& markers) {
    // For each marker, compute u_desired = v_cm + omega x r
    // r is the vector from CM to marker position. 
    // markers[i].pos is absolute position.
    // So r = markers[i].pos - state_.position;
    
    // Also update rel_pos while we are at it, to ensure consistency
    for (auto& m : markers) {
        m.rel_pos = m.pos - state_.position;
        
        float3 v_rot = crossProduct(state_.angular_velocity, m.rel_pos);
        m.u_desired = state_.velocity + v_rot;
    }
}

float3 IBMMotion::rotateVector(const float3& v, const Quaternion& q) const {
    // Rotate vector v by quaternion q
    // v' = q * v * q_conj
    // Standard formula: v + 2*w*(q.xyz x v) + 2*(q.xyz x (q.xyz x v))
    float3 u = make_float3(q.x, q.y, q.z);
    float s = q.w;
    
    float3 uv = crossProduct(u, v);
    float3 uuv = crossProduct(u, uv);
    
    // v + 2*s*uv + 2*uuv
    return v + (uv * 2.0f * s) + (uuv * 2.0f);
}

Quaternion IBMMotion::multiplyQuat(const Quaternion& q1, const Quaternion& q2) const {
    return Quaternion(
        q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y,
        q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x,
        q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w,
        q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z
    );
}
