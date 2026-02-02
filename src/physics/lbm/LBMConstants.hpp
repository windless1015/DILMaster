#pragma once

#include "LBMConfig.hpp"

namespace lbm {

// Basic Lattice constants
constexpr int Q = 19;
constexpr int DIM = 3;

// D3Q19 constants check
// Order: Rest(0), Face(1-6), Edge(7-18)
// 0: (0,0,0)
// 1-2: (+-1, 0, 0)
// 3-4: (0, +-1, 0)
// 5-6: (0, 0, +-1)
// 7-10: (+-1, +-1, 0)
// 11-14: (+-1, 0, +-1)
// 15-18: (0, +-1, +-1)

namespace constants {

static constexpr int c_dir[19][3] = {
    {0, 0, 0},               // 0: rest
    {1, 0, 0},  {-1, 0, 0},  // 1-2: x-axis
    {0, 1, 0},  {0, -1, 0},  // 3-4: y-axis
    {0, 0, 1},  {0, 0, -1},  // 5-6: z-axis
    {1, 1, 0},  {-1, -1, 0}, // 7-8: xy-plane diagonals
    {1, -1, 0}, {-1, 1, 0},  // 9-10: xy-plane diagonals
    {1, 0, 1},  {-1, 0, -1}, // 11-12: xz-plane diagonals
    {1, 0, -1}, {-1, 0, 1},  // 13-14: xz-plane diagonals
    {0, 1, 1},  {0, -1, -1}, // 15-16: yz-plane diagonals
    {0, 1, -1}, {0, -1, 1}   // 17-18: yz-plane diagonals
};

static constexpr int c_opposite[19] = {
    0,      // 0 -> 0
    2,  1,  // 1 <-> 2
    4,  3,  // 3 <-> 4
    6,  5,  // 5 <-> 6
    8,  7,  // 7 <-> 8
    10, 9,  // 9 <-> 10
    12, 11, // 11 <-> 12
    14, 13, // 13 <-> 14
    16, 15, // 15 <-> 16
    18, 17  // 17 <-> 18
};

static constexpr real c_w[19] = {
    1.0f / 3.0f,                              // Rest (0)
    1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, // Face (1-6)
    1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 36.0f, 1.0f / 36.0f,
    1.0f / 36.0f, // Edge (7-18)
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f,
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f};

// Speed of sound squared in lattice units
static constexpr real cs2 = 1.0f / 3.0f;
static constexpr real inv_cs2 = 3.0f;

} // namespace constants
} // namespace lbm
