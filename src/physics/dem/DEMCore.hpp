#pragma once
/**
 * @file DEMCore.hpp
 * @brief DEM CUDA computation core
 *
 * Owns all GPU memory for particle state, spatial grid, and contact history.
 * Implements the complete DEM pipeline per substep:
 *   1. Clear force/torque
 *   2. Add gravity
 *   3. Build spatial hash grid (uniform grid, O(N) broad phase)
 *   4. Swap contact history tables
 *   5. Compute particle-particle contacts (with tangential friction + history)
 *   6. Compute particle-wall contacts   (same contact model)
 *   7. Integrate (Symplectic Euler: v, omega, x)
 *
 * No LBM/IB dependencies. Fully self-contained.
 */

#include "DEMConfig.hpp"
#include <cuda_runtime.h>
#include <cstddef>

namespace dem {

// ============================================================================
// Contact History Hash Table Entry
// ============================================================================

/**
 * @brief Single entry in the contact history hash table (GPU-resident).
 *
 * Key encodes a contact pair (i, j) with i < j.
 * For wall contacts, j = num_particles + wall_face_id (0..5).
 */
struct ContactEntry {
    unsigned long long key;          ///< packed(i, j); UINT64_MAX = empty slot
    float xi_tx, xi_ty, xi_tz;      ///< tangential displacement history
};

/// Sentinel for empty hash table slots
static constexpr unsigned long long CONTACT_EMPTY_KEY = 0xFFFFFFFFFFFFFFFFULL;

/**
 * @brief Pack two indices into a contact key (host or device).
 *
 * Ensures lo < hi ordering so (i,j) and (j,i) map to the same key.
 */
inline __host__ __device__ unsigned long long makeContactKey(unsigned int i,
                                                             unsigned int j) {
    unsigned int lo = i < j ? i : j;
    unsigned int hi = i < j ? j : i;
    return (static_cast<unsigned long long>(lo) << 32) |
           static_cast<unsigned long long>(hi);
}

// ============================================================================
// Step Diagnostics
// ============================================================================

/**
 * @brief Per-step diagnostic counters (downloaded from GPU after each step).
 */
struct DEMStepStats {
    unsigned int particle_contacts = 0;
    unsigned int wall_contacts     = 0;
    float        max_overlap       = 0.0f;
    bool         has_nan           = false;
};

// ============================================================================
// DEMCore
// ============================================================================

/**
 * @brief DEM CUDA computation core
 *
 * Lifecycle:
 *   1. configure(cfg)  -- set parameters, compute grid dimensions
 *   2. allocate()       -- allocate all GPU memory
 *   3. upload*(...)     -- upload initial particle state
 *   4. initMassProperties() -- compute mass/inertia from radii+density
 *   5. step(dt) or stepMultiple(dt, subs) -- run simulation
 *   6. download*(...)   -- retrieve results
 *   7. destructor frees GPU memory
 */
class DEMCore {
public:
    DEMCore();
    ~DEMCore();

    // Non-copyable
    DEMCore(const DEMCore &) = delete;
    DEMCore &operator=(const DEMCore &) = delete;

    // ========================================================================
    // Setup
    // ========================================================================

    /** @brief Set configuration and compute derived parameters. */
    void configure(const DEMConfig &cfg);

    /** @brief Allocate all GPU memory. Must call configure() first. */
    void allocate();

    // ========================================================================
    // Host <-> Device Data Transfer (SoA layout)
    // ========================================================================
    //
    // SoA format for vec3 fields:
    //   [x0, x1, ..., xN-1, y0, y1, ..., yN-1, z0, z1, ..., zN-1]
    //
    // All host pointers must have at least 3*N floats for vec3 or N for scalar.

    void uploadPositions(const float *pos_h);          ///< 3N floats
    void uploadVelocities(const float *vel_h);         ///< 3N floats
    void uploadAngularVelocities(const float *omega_h);///< 3N floats
    void uploadRadii(const float *radii_h);            ///< N floats

    void downloadPositions(float *pos_h)          const; ///< 3N floats
    void downloadVelocities(float *vel_h)         const; ///< 3N floats
    void downloadAngularVelocities(float *omega_h)const; ///< 3N floats
    void downloadForces(float *force_h)           const; ///< 3N floats
    void downloadTorques(float *torque_h)         const; ///< 3N floats
    void downloadRadii(float *radii_h)            const; ///< N floats

    /**
     * @brief Compute mass, inv_mass, inertia, inv_inertia from current
     *        device-side radii + config density.
     *
     * Must be called after uploadRadii() and before the first step.
     */
    void initMassProperties();

    // ========================================================================
    // Simulation
    // ========================================================================

    /** @brief Execute a single DEM substep. */
    void step(float dt);

    /** @brief Execute multiple substeps: dt_sub = dt / substeps. */
    void stepMultiple(float dt, int substeps);

    // ========================================================================
    // Device Pointer Access (for coupling or external kernels)
    // ========================================================================

    float *positionDevice()        { return d_pos_;    }
    float *velocityDevice()        { return d_vel_;    }
    float *omegaDevice()           { return d_omega_;  }
    float *forceDevice()           { return d_force_;  }
    float *torqueDevice()          { return d_torque_; }
    float *radiusDevice()          { return d_radius_; }
    float *massDevice()            { return d_mass_;   }

    const float *positionDevice()  const { return d_pos_;    }
    const float *velocityDevice()  const { return d_vel_;    }
    const float *omegaDevice()     const { return d_omega_;  }
    const float *forceDevice()     const { return d_force_;  }
    const float *torqueDevice()    const { return d_torque_; }
    const float *radiusDevice()    const { return d_radius_; }

    // ========================================================================
    // Diagnostics
    // ========================================================================

    /** @brief Check for NaN / excessive overlap. Downloads counters. */
    bool checkHealth() const;

    /** @brief Last step diagnostics (valid after step()). */
    DEMStepStats lastStepStats() const { return last_stats_; }

    // ========================================================================
    // Info
    // ========================================================================

    std::size_t         numParticles() const { return cfg_.num_particles; }
    unsigned long long  stepCount()    const { return step_count_;        }
    const DEMConfig    &config()       const { return cfg_;               }
    void                synchronize()  const;
    
    // -- Physics Query --
    float getCriticalTimeStep() const { return t_critical_; }

    // ========================================================================
    // External force injection (for coupling, called before step)
    // ========================================================================

    /**
     * @brief Clear force and torque arrays on device.
     *
     * Useful when coupling code wants to write external forces before step().
     */
    void clearForcesTorquePublic();

private:
    // -- Configuration --
    DEMConfig cfg_;
    bool      allocated_  = false;
    unsigned long long step_count_ = 0;
    DEMStepStats last_stats_{};

    // -- Derived parameters (computed in configure()) --
    // -- Derived parameters (computed in configure()) --
    float gamma_n_pp_  = 0.0f; // Particle-Particle damping
    float gamma_n_pw_  = 0.0f; // Particle-Wall damping
    float gamma_t_pp_  = 0.0f;
    float gamma_t_pw_  = 0.0f;
    
    // Parameters for time-step check
    float mass_min_    = 0.0f; 
    float t_critical_  = 0.0f;
    float cell_size_   = 0.0f;
    int   grid_nx_     = 0;
    int   grid_ny_     = 0;
    int   grid_nz_     = 0;
    int   num_cells_   = 0;
    int   contact_capacity_ = 0;   ///< hash table capacity (power of 2)
    int   contact_mask_     = 0;   ///< capacity - 1 (for fast modulo)

    // -- Device particle buffers (SoA, 3N or N floats) --
    float *d_pos_          = nullptr;   // 3N  position
    float *d_vel_          = nullptr;   // 3N  velocity
    float *d_omega_        = nullptr;   // 3N  angular velocity
    float *d_force_        = nullptr;   // 3N  force accumulator
    float *d_torque_       = nullptr;   // 3N  torque accumulator
    float *d_radius_       = nullptr;   // N   radius
    float *d_mass_         = nullptr;   // N   mass
    float *d_inv_mass_     = nullptr;   // N   1/mass
    float *d_inertia_      = nullptr;   // N   moment of inertia
    float *d_inv_inertia_  = nullptr;   // N   1/inertia

    // -- Spatial grid buffers --
    unsigned int *d_cell_id_    = nullptr;   // N
    unsigned int *d_sorted_idx_ = nullptr;   // N
    unsigned int *d_cell_start_ = nullptr;   // num_cells
    unsigned int *d_cell_end_   = nullptr;   // num_cells

    // -- Contact history (ping-pong hash tables) --
    ContactEntry *d_contacts_old_ = nullptr;  // contact_capacity
    ContactEntry *d_contacts_new_ = nullptr;  // contact_capacity

    // -- Diagnostic counters (device-side) --
    // Layout: [particle_contacts, wall_contacts, nan_flag]
    unsigned int *d_diag_ = nullptr;          // 3 uints
    float        *d_diag_max_overlap_ = nullptr; // 1 float

    // -- Internal substep pipeline --
    void clearForcesTorque();
    void addGravity();
    void buildSpatialGrid();
    void swapContactTables();
    void computeParticleContacts(float dt);
    void computeWallContacts(float dt);
    void integrateSymplecticEuler(float dt);
    void gatherDiagnostics();

    void freeMemory();
};

} // namespace dem
