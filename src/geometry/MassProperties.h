#pragma once
#include "VectorTypes.h"

struct MassProperties {
    float mass = 1.0f;
    float3 centerOfMass = make_float3(0,0,0);
    float volume = 1.0f;
    
    // Inertia Tensor components
    float Ixx = 1.0f;
    float Iyy = 1.0f;
    float Izz = 1.0f;
    float Ixy = 0.0f;
    float Ixz = 0.0f;
    float Iyz = 0.0f;
    float equivalentRadius = 0.0f;
};
