#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

// =================================================================================================
// CUDA Vector Type Operators (C++ Helper)
// =================================================================================================
// Since cuda_runtime.h (vector_types.h) defines float3/int3 as simple structs without operators,
// we define them here for C++ convenience in both Host and Device code.

inline __host__ __device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

inline __host__ __device__ float3 operator*(float s, const float3& a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

inline __host__ __device__ float3 operator/(const float3& a, float s) {
    return make_float3(a.x / s, a.y / s, a.z / s);
}

inline __host__ __device__ float3 operator-(const float3& a) {
    return make_float3(-a.x, -a.y, -a.z);
}

inline __host__ __device__ float3& operator+=(float3& a, const float3& b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
}

inline __host__ __device__ float3& operator-=(float3& a, const float3& b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
    return a;
}

// Helper lengths
inline __host__ __device__ float length(const float3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline __host__ __device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float3 cross(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}

// IO
inline std::ostream& operator<<(std::ostream& os, const float3& v) {
    os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return os;
}
