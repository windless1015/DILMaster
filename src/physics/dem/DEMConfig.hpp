#pragma once
/**
 * @file DEMConfig.hpp
 * @brief DEM configuration structs and utility functions
 *
 * Provides:
 *   - MaterialParams: contact model parameters (kn, kt, gamma, mu, e)
 *   - DEMConfig: user-facing configuration for DEMSolver/DEMCore
 *   - computeDampingFromRestitution(): derive gamma_n from e
 */

#include <cmath>
#include <cstddef>

namespace dem {

// ============================================================================
// Material Parameters
// ============================================================================

/**
 * @brief Contact model material parameters
 *
 * Supports per-material index in the future (currently global).
 */
struct MaterialParams {
    float kn = 1e5f;            ///< Normal spring stiffness [N/m]
    float kt = 2.857e4f;       ///< Tangential spring stiffness [N/m] (default 2/7*kn)
    // float gamma_n; // Removed: Auto-computed from restitution
    // float gamma_t; // Removed: Auto-computed from restitution
    float mu = 0.3f;            ///< Coulomb friction coefficient
    float mu_roll = 0.0f;       ///< Rolling friction coefficient (optional)
    float restitution = 0.9f;   ///< Coefficient of restitution [0..1]
};

// ============================================================================
// Damping from Restitution Utility
// ============================================================================

/**
 * @brief Compute normal damping coefficient from coefficient of restitution
 *
 * Formula based on Thornton (1997) / simplified spring-dashpot:
 *   gamma_n = -2 * ln(e) * sqrt(kn * m_eff) / sqrt(pi^2 + ln(e)^2)
 *
 * @param e         Coefficient of restitution (0 < e <= 1)
 * @param kn        Normal spring stiffness [N/m]
 * @param m_eff     Effective mass [kg]
 * @return          Normal damping coefficient gamma_n [N*s/m]
 */
inline float computeDampingFromRestitution(float e, float kn, float m_eff) {
    if (e >= 1.0f) return 0.0f;        // Elastic
    if (e <= 1e-4f) e = 1e-4f;         // Avoid singularity
    
    // User requested formula: 
    // gamma_n = -2 * m_eff * log(e) / sqrt(pi^2 + log(e)^2) * sqrt(kn / m_eff)
    //         = -2 * log(e) * sqrt(m_eff * kn) / sqrt(pi^2 + log(e)^2)
    // Matches standard analytical solution.
    
    const float pi = 3.14159265358979323846f;
    float ln_e = std::log(e);
    float term1 = -2.0f * ln_e;
    float term2 = std::sqrt(pi * pi + ln_e * ln_e);
    float term3 = std::sqrt(kn * m_eff); // sqrt(kn/m_eff) * m_eff = sqrt(kn*m_eff)
    
    return (term1 / term2) * term3;
}

/**
 * @brief Compute mass of a sphere from radius and density
 */
inline float sphereMass(float radius, float density) {
    const float pi = 3.14159265358979323846f;
    return (4.0f / 3.0f) * pi * radius * radius * radius * density;
}

/**
 * @brief Compute moment of inertia of a solid sphere
 *
 * I = (2/5) * m * r^2
 */
inline float sphereInertia(float mass, float radius) {
    return 0.4f * mass * radius * radius;
}

// ============================================================================
// DEM Configuration (user-facing)
// ============================================================================

/**
 * @brief DEM solver configuration
 *
 * All parameters for a DEM simulation, set via DEMSolver::setConfig()
 * or directly on DEMCore::configure().
 */
struct DEMConfig {
    // -- Particle properties --
    std::size_t num_particles = 0;
    float particle_radius  = 0.01f;     ///< Default radius [m]
    float particle_density = 2500.0f;   ///< Density [kg/m^3]

    // -- Material / Contact model --
    MaterialParams material;

    // -- Gravity [m/s^2] --
    float gravity_x = 0.0f;
    float gravity_y = 0.0f;
    float gravity_z = -9.81f;

    // -- Domain boundaries [m] --
    float domain_min_x = 0.0f;
    float domain_min_y = 0.0f;
    float domain_min_z = 0.0f;
    float domain_max_x = 1.0f;
    float domain_max_y = 1.0f;
    float domain_max_z = 1.0f;

    // -- Time integration --
    int dem_substeps = 1;               ///< Substeps per outer step

    // -- Spatial grid --
    float cell_size = 0.0f;            ///< Grid cell size [m]; 0 = auto (2*max_radius)

    // -- Debug / safety --
    bool  use_naive_n2     = false;     ///< Force N^2 collision (debug only)
    float max_overlap_warn = 0.2f;      ///< Warn if overlap > this fraction of radius
};

} // namespace dem
