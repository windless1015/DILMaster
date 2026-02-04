/**
 * @file STLGeometryLoader.hpp
 * @brief STL geometry loader for IBM body creation
 * 
 * Provides factory functions to:
 * - Compute mass properties from STL mesh
 * - Sample surface markers for IBM
 * - Extract mesh vertices for visualization
 * 
 * All operations are CPU-only.
 */
#pragma once

#include "STLReader.h"
#include "MassProperties.h"
#include "IBMMarker.h"
#include <vector>
#include <string>

/**
 * @brief Static utility class for loading and processing STL geometry
 * 
 * Data flow:
 *   STL file → STLReader::readSTL → STLMesh
 *   STLMesh → computeMassProperties → MassProperties
 *   STLMesh → sampleSurfaceMarkers → vector<IBMMarker>
 *   STLMesh → extractMeshVertices → vector<float3>
 */
class STLGeometryLoader {
public:
    /**
     * @brief Parameters for adaptive IBM marker sampling
     */
    struct MarkerSamplingParams {
        float dx = 1.0f;
        float ds_requested = 1.0f;
        float ds_min = 0.5f;
        float ds_max = 1.0f;
        int max_markers = 200000;
        int min_markers = 1000;
        float aabb_tolerance = 1e-4f;
        float nn_p10_min_ratio = 0.25f;
        float nn_p90_max_ratio = 2.0f;
        float nn_min_abs_ratio = 0.05f;
        float area_per_marker_min = 0.25f;
        float area_per_marker_max = 4.0f;
        int adaptive_iters = 6;
        bool strict_fail = true;
    };

    /**
     * @brief Report for adaptive IBM marker sampling
     */
    struct MarkerSamplingReport {
        float ds_final = 0.0f;
        int marker_count = 0;
        float surface_area = 0.0f;
        float aabb_outside_fraction = 0.0f;
        float min_nn = 0.0f, avg_nn = 0.0f, max_nn = 0.0f;
        float p10_nn = 0.0f, p50_nn = 0.0f, p90_nn = 0.0f;
        float area_per_marker = 0.0f;
        float tri_point_corr = 0.0f;
        int too_dense_pairs = 0;
        int too_sparse_holes = 0;
        int effective_min_markers = 0;
        bool pass = false;
        std::string fail_reason;
    };

    /**
     * @brief Compute mass properties using divergence theorem (fast O(n) method)
     * 
     * Uses signed volume of tetrahedra formed by triangles and origin to compute
     * total volume and center of mass. Inertia tensor is approximated using
     * surface integral method.
     * 
     * @param mesh Input STL mesh (should be centered at origin for best accuracy)
     * @param density Material density [kg/m³]
     * @return Computed mass properties
     */
    static MassProperties computeMassProperties(const STLMesh& mesh, float density);

    /**
     * @brief Sample surface markers on STL mesh for IBM
     * 
     * Uses grid-based barycentric sampling on each triangle.
     * Number of samples per triangle scales with area / (spacing²).
     * 
     * @param mesh Input STL mesh
     * @param centerOfMass Body center of mass (for computing relative positions)
     * @param markerSpacing Target spacing between markers
     * @return Vector of IBM markers with positions and relative offsets
     */
    static std::vector<IBMMarker> sampleSurfaceMarkers(
        const STLMesh& mesh,
        const float3& centerOfMass,
        float markerSpacing
    );

    /**
     * @brief Adaptive IBM marker sampling with validation and report
     */
    static MarkerSamplingReport sampleIBMMarkersAdaptive(
        const STLMesh& mesh,
        const float3& centerOfMass,
        std::vector<IBMMarker>& outMarkers,
        const MarkerSamplingParams& params
    );

    /**
     * @brief Extract unique mesh vertices for visualization
     * 
     * Returns the relative vertex positions (centered on mesh origin).
     * Use with body position for world-space visualization.
     * 
     * @param mesh Input STL mesh
     * @return Vector of unique vertex positions
     */
    static std::vector<float3> extractMeshVertices(const STLMesh& mesh);

    /**
     * @brief Write STL mesh to VTP (VTK PolyData) for visualization
     */
    static bool writeMeshVTP(const STLMesh& mesh, const std::string& filename);

    /**
     * @brief Write IBM markers to VTP (VTK PolyData) for visualization
     */
    static bool writeMarkersVTP(const std::vector<IBMMarker>& markers, const std::string& filename);

    /**
     * @brief Check if a point is inside the STL mesh using ray casting
     * 
     * @param mesh Input STL mesh (must be closed and manifold)
     * @param point Point to check
     * @return true if point is inside
     */
    static bool isInside(const STLMesh& mesh, const float3& point);

    /**
     * @brief Voxelize the STL mesh into a grid
     * 
     * @param mesh Input STL mesh
     * @param voxelSize Size of each voxel
     * @param dims Output: dimensions of the grid
     * @param minBound Output: minimum bound of the grid
     * @return Flat vector of voxel flags (1 for inside, 0 for outside)
     */
    static std::vector<int> voxelize(
        const STLMesh& mesh,
        float voxelSize,
        int3& dims,
        float3& minBound
    );

private:
    /**
     * @brief Ray-triangle intersection for voxelization
     */
    static bool rayTriangleIntersect(
        const float3& origin, const float3& direction,
        const float3& v0, const float3& v1, const float3& v2,
        float& t
    );
    /**
     * @brief Sample points on a single triangle
     */
    static void sampleTriangle(
        const STLTriangle& tri,
        const float3& com,
        float markerSpacing,
        std::vector<IBMMarker>& markers
    );
};
