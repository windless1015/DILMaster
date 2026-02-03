/**
 * @file IBMMotion.hpp
 * @brief Rigid body kinematics for IBM objects
 * 
 * Handles rigid body dynamics including 6DOF integration using quaternions.
 * CPU-only implementation.
 */
#pragma once

#include "../../geometry/STLReader.h"
#include "../../geometry/IBMMarker.h"
#include "../../geometry/MassProperties.h"
#include <vector>

/**
 * @brief Types of prescribed or dynamic motion
 */
enum class MotionType {
    FIXED,              ///< Static body
    FIXED_ROTATION,     ///< Constant rotation around an axis
    FREE_FALL,          ///< Fully coupled 6DOF dynamics
    TRANSLATION,        ///< Constant translation (not fully implemented in this simplified version)
    OSCILLATION         ///< Periodic motion (not fully implemented)
};

/**
 * @brief Simple Quaternion struct (x, y, z, w)
 */
struct Quaternion {
    float x, y, z, w;
    Quaternion(float x_=0, float y_=0, float z_=0, float w_=1) 
        : x(x_), y(y_), z(z_), w(w_) {}
};

/**
 * @brief State of a rigid body
 */
struct RigidBodyState {
    float3 position;            ///< Center of mass position
    float3 velocity;            ///< Linear velocity
    float3 angular_velocity;    ///< Angular velocity vector (rad/s)
    Quaternion orientation;     ///< Orientation (rotation from initial)
    float mass;                 ///< Total mass
    float3 inertia;             ///< Principal moments of inertia (Ixx, Iyy, Izz)
    
    RigidBodyState() 
        : orientation(0,0,0,1), mass(1.0f) {
        // Initialize float3s using make_float3 if default ctor is not enough
        // STLReader.h float3 default ctor inits to 0 already
        position = make_float3(0,0,0);
        velocity = make_float3(0,0,0);
        angular_velocity = make_float3(0,0,0);
        inertia = make_float3(1,1,1);
    }
};

class IBMMotion {
public:
    /**
     * @brief Initialize rigid body state
     */
    void initialize(const MassProperties& props, const float3& initialPos);

    /**
     * @brief Accumulate forces and torques from Lagrangian markers
     * @param markers List of markers containing 'force' and 'rel_pos'
     */
    void updateForces(const std::vector<IBMMarker>& markers);

    /**
     * @brief Integrate equations of motion (Newton-Euler)
     * @param dt Time step
     */
    void integrate(float dt);

    /**
     * @brief Update desired velocity for all markers based on current RB state
     * u_desired = v_cm + omega x r
     * @param markers Markers to update
     */
    void updateMarkerVelocities(std::vector<IBMMarker>& markers);

    // Getters and Setters
    const RigidBodyState& getState() const { return state_; }
    void setMotionType(MotionType type) { type_ = type; }
    void setRotationAxis(const float3& axis, float omega);
    void setGravity(const float3& g) { gravity_ = g; }

    // Helpers (Public for Solver usage)
    float3 rotateVector(const float3& v, const Quaternion& q) const;
    Quaternion multiplyQuat(const Quaternion& q1, const Quaternion& q2) const;
    float3 crossProduct(const float3& a, const float3& b) const;

private:
    RigidBodyState state_;
    MotionType type_ = MotionType::FIXED;

    float3 totalForce_;
    float3 totalTorque_;
    float3 gravity_;

    // Fixed rotation parameters
    float3 rotationAxis_;
    float fixedOmega_ = 0.0f;
};
