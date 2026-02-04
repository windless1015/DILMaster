/**
 * @file STLGeometryLoader.cpp
 * @brief Implementation of STL geometry loading functions
 * 
 * Algorithms adapted from reference code for CPU-only execution.
 */

#include "STLGeometryLoader.hpp"
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <unordered_map>
#include <vector>
#include <random>
#include <sstream>
#include <stdexcept>
#include <limits>
#include <fstream>
#include "../physics/ibm/IBMKernelValidation.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
// Mass Properties Computation
// =============================================================================

MassProperties STLGeometryLoader::computeMassProperties(const STLMesh& mesh, float density) {
    MassProperties props;

    // Use divergence theorem for fast volume/CoM calculation from surface mesh
    // This is O(triangles) instead of O(voxels × triangles)
    printf("Computing mass properties (surface integral method)...\n");

    double signedVolume = 0.0;
    double comSumX = 0.0;
    double comSumY = 0.0;
    double comSumZ = 0.0;

    // First pass: volume and center of mass using divergence theorem
    for (const auto& tri : mesh.triangles) {
        // Signed volume of tetrahedron formed by triangle and origin
        // V = (a · (b × c)) / 6
        float3 a = tri.vertices[0];
        float3 b = tri.vertices[1];
        float3 c = tri.vertices[2];

        // Cross product b × c
        float3 bc;
        bc.x = b.y * c.z - b.z * c.y;
        bc.y = b.z * c.x - b.x * c.z;
        bc.z = b.x * c.y - b.y * c.x;

        // Dot product a · (b × c)
        double signedVol = (static_cast<double>(a.x) * bc.x +
                            static_cast<double>(a.y) * bc.y +
                            static_cast<double>(a.z) * bc.z) / 6.0;
        signedVolume += signedVol;

        // Centroid of tetrahedron is (a + b + c + origin) / 4 = (a + b + c) / 4
        // Weighted by signed volume
        float3 centroid;
        centroid.x = (a.x + b.x + c.x) / 4.0f;
        centroid.y = (a.y + b.y + c.y) / 4.0f;
        centroid.z = (a.z + b.z + c.z) / 4.0f;

        comSumX += signedVol * static_cast<double>(centroid.x);
        comSumY += signedVol * static_cast<double>(centroid.y);
        comSumZ += signedVol * static_cast<double>(centroid.z);
    }

    double absVolume = std::fabs(signedVolume);
    props.volume = static_cast<float>(absVolume);
    props.mass = density * props.volume;

    if (absVolume > 1e-10) {
        props.centerOfMass.x = static_cast<float>(comSumX / signedVolume);
        props.centerOfMass.y = static_cast<float>(comSumY / signedVolume);
        props.centerOfMass.z = static_cast<float>(comSumZ / signedVolume);
    } else {
        printf("  ERROR: Signed volume too small (%.6e), CoM fallback to AABB center.\n", signedVolume);
        props.centerOfMass = mesh.getCenter();
    }

    printf("  Signed volume: %.6e, Volume: %.2f, CoM: (%.2f, %.2f, %.2f)\n",
           signedVolume, props.volume, props.centerOfMass.x, props.centerOfMass.y, props.centerOfMass.z);

    // Second pass: inertia tensor using surface integral approximation
    // Using formula from "Implicit Fairing of Irregular Meshes" (Taubin 95)
    float Ixx = 0, Iyy = 0, Izz = 0, Ixy = 0, Ixz = 0, Iyz = 0;

    for (const auto& tri : mesh.triangles) {
        // Triangle centroid relative to CoM
        float3 tc;
        tc.x = (tri.vertices[0].x + tri.vertices[1].x + tri.vertices[2].x) / 3.0f - props.centerOfMass.x;
        tc.y = (tri.vertices[0].y + tri.vertices[1].y + tri.vertices[2].y) / 3.0f - props.centerOfMass.y;
        tc.z = (tri.vertices[0].z + tri.vertices[1].z + tri.vertices[2].z) / 3.0f - props.centerOfMass.z;

        float area = tri.area();

        // Approximate contribution using shell inertia
        Ixx += area * (tc.y * tc.y + tc.z * tc.z);
        Iyy += area * (tc.x * tc.x + tc.z * tc.z);
        Izz += area * (tc.x * tc.x + tc.y * tc.y);
        Ixy -= area * tc.x * tc.y;
        Ixz -= area * tc.x * tc.z;
        Iyz -= area * tc.y * tc.z;
    }

    // Scale by density and approximate thickness
    float thickness = std::pow(props.volume, 1.0f / 3.0f) * 0.1f;  // Approximate shell thickness
    float scale = density * thickness;

    props.Ixx = Ixx * scale;
    props.Iyy = Iyy * scale;
    props.Izz = Izz * scale;
    props.Ixy = Ixy * scale;
    props.Ixz = Ixz * scale;
    props.Iyz = Iyz * scale;

    // Equivalent radius of sphere with same volume
    props.equivalentRadius = std::pow(3.0f * props.volume / (4.0f * static_cast<float>(M_PI)), 1.0f / 3.0f);

    printf("  Inertia: Ixx=%.0f, Iyy=%.0f, Izz=%.0f\n", props.Ixx, props.Iyy, props.Izz);
    printf("  Equivalent radius: %.3f\n", props.equivalentRadius);

    return props;
}

// =============================================================================
// Surface Marker Sampling
// =============================================================================

void STLGeometryLoader::sampleTriangle(
    const STLTriangle& tri,
    const float3& com,
    float markerSpacing,
    std::vector<IBMMarker>& markers
) {
    float area = tri.area();
    float targetArea = markerSpacing * markerSpacing;
    int nSamples = std::max(1, static_cast<int>(area / targetArea + 0.5f));

    // Use barycentric coordinates for uniform sampling
    for (int s = 0; s < nSamples; ++s) {
        // Simple grid-based sampling for determinism
        int gridSize = static_cast<int>(std::sqrt(static_cast<float>(nSamples))) + 1;
        int si = s % gridSize;
        int sj = s / gridSize;

        float u = (static_cast<float>(si) + 0.5f) / static_cast<float>(gridSize);
        float v = (static_cast<float>(sj) + 0.5f) / static_cast<float>(gridSize);

        // Ensure inside triangle
        if (u + v > 1.0f) {
            u = 1.0f - u;
            v = 1.0f - v;
        }
        float w = 1.0f - u - v;

        // Compute position using barycentric interpolation
        float3 pos;
        pos.x = w * tri.vertices[0].x + u * tri.vertices[1].x + v * tri.vertices[2].x;
        pos.y = w * tri.vertices[0].y + u * tri.vertices[1].y + v * tri.vertices[2].y;
        pos.z = w * tri.vertices[0].z + u * tri.vertices[1].z + v * tri.vertices[2].z;

        // Relative position from center of mass
        float3 rel;
        rel.x = pos.x - com.x;
        rel.y = pos.y - com.y;
        rel.z = pos.z - com.z;

        markers.push_back(IBMMarker(pos, rel));
    }
}

namespace {

struct CellKey {
    int x;
    int y;
    int z;
    bool operator==(const CellKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct CellKeyHash {
    std::size_t operator()(const CellKey& k) const {
        std::size_t h1 = std::hash<int>{}(k.x);
        std::size_t h2 = std::hash<int>{}(k.y);
        std::size_t h3 = std::hash<int>{}(k.z);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

float3 normalizeSafe(const float3& v) {
    float len = length(v);
    if (len <= 0.0f) {
        return make_float3(0, 0, 0);
    }
    return v / len;
}

float triangleArea(const STLTriangle& tri) {
    float3 e0 = tri.vertices[1] - tri.vertices[0];
    float3 e1 = tri.vertices[2] - tri.vertices[0];
    return 0.5f * length(cross(e0, e1));
}

bool pointInAABB(const float3& p, const float3& minB, const float3& maxB, float tol) {
    return (p.x >= minB.x - tol && p.x <= maxB.x + tol &&
            p.y >= minB.y - tol && p.y <= maxB.y + tol &&
            p.z >= minB.z - tol && p.z <= maxB.z + tol);
}

CellKey toCell(const float3& p, float cellSize) {
    return CellKey{
        static_cast<int>(std::floor(p.x / cellSize)),
        static_cast<int>(std::floor(p.y / cellSize)),
        static_cast<int>(std::floor(p.z / cellSize))
    };
}

float computePearsonCorr(const std::vector<float>& a, const std::vector<int>& b) {
    if (a.size() != b.size() || a.size() < 2) {
        return 0.0f;
    }
    const std::size_t n = a.size();
    double meanA = 0.0;
    double meanB = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        meanA += a[i];
        meanB += b[i];
    }
    meanA /= static_cast<double>(n);
    meanB /= static_cast<double>(n);

    double cov = 0.0;
    double varA = 0.0;
    double varB = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double da = a[i] - meanA;
        double db = static_cast<double>(b[i]) - meanB;
        cov += da * db;
        varA += da * da;
        varB += db * db;
    }
    if (varA <= 0.0 || varB <= 0.0) {
        return 0.0f;
    }
    return static_cast<float>(cov / std::sqrt(varA * varB));
}

void computeNearestNeighborStats(
    const std::vector<float3>& positions,
    float cellSize,
    float& minNN,
    float& avgNN,
    float& maxNN,
    float& p10,
    float& p50,
    float& p90,
    int& tooDenseCount,
    int& tooSparseCount,
    float denseThresh,
    float sparseThresh
) {
    minNN = 0.0f;
    avgNN = 0.0f;
    maxNN = 0.0f;
    p10 = p50 = p90 = 0.0f;
    tooDenseCount = 0;
    tooSparseCount = 0;

    if (positions.size() < 2) {
        return;
    }

    std::unordered_map<CellKey, std::vector<int>, CellKeyHash> grid;
    grid.reserve(positions.size());

    for (std::size_t i = 0; i < positions.size(); ++i) {
        CellKey key = toCell(positions[i], cellSize);
        grid[key].push_back(static_cast<int>(i));
    }

    std::vector<float> nn;
    nn.resize(positions.size(), std::numeric_limits<float>::max());

    auto searchRadius = [&](const float3& p, const CellKey& key, std::size_t selfIdx, int radius) {
        float best = std::numeric_limits<float>::max();
        for (int dz = -radius; dz <= radius; ++dz) {
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    CellKey nk{key.x + dx, key.y + dy, key.z + dz};
                    auto it = grid.find(nk);
                    if (it == grid.end()) continue;
                    for (int j : it->second) {
                        if (static_cast<std::size_t>(j) == selfIdx) continue;
                        float3 d = positions[j] - p;
                        float dist = length(d);
                        if (dist < best) {
                            best = dist;
                        }
                    }
                }
            }
        }
        return best;
    };

    for (std::size_t i = 0; i < positions.size(); ++i) {
        const float3& p = positions[i];
        CellKey key = toCell(p, cellSize);
        float best = searchRadius(p, key, i, 1);
        if (!std::isfinite(best) || best == std::numeric_limits<float>::max()) {
            best = searchRadius(p, key, i, 2);
        }
        nn[i] = best;
    }

    minNN = *std::min_element(nn.begin(), nn.end());
    maxNN = *std::max_element(nn.begin(), nn.end());
    double sum = 0.0;
    for (float v : nn) sum += v;
    avgNN = static_cast<float>(sum / static_cast<double>(nn.size()));

    std::vector<float> sorted = nn;
    auto nth = [&](float q) {
        std::size_t idx = static_cast<std::size_t>(q * (sorted.size() - 1));
        std::nth_element(sorted.begin(), sorted.begin() + idx, sorted.end());
        return sorted[idx];
    };
    p10 = nth(0.10f);
    p50 = nth(0.50f);
    p90 = nth(0.90f);

    for (float v : nn) {
        if (v < denseThresh) tooDenseCount++;
        if (v > sparseThresh) tooSparseCount++;
    }
}

} // namespace

std::vector<IBMMarker> STLGeometryLoader::sampleSurfaceMarkers(
    const STLMesh& mesh,
    const float3& centerOfMass,
    float markerSpacing
) {
    MarkerSamplingParams params;
    params.dx = markerSpacing;
    params.ds_requested = markerSpacing;
    params.ds_min = 0.5f * markerSpacing;
    params.ds_max = 1.0f * markerSpacing;

    std::vector<IBMMarker> markers;
    MarkerSamplingReport report = sampleIBMMarkersAdaptive(mesh, centerOfMass, markers, params);

    printf("Surface sampling: Generated %zu IBM markers (ds=%.3f)\n",
           markers.size(), report.ds_final);

    return markers;
}

STLGeometryLoader::MarkerSamplingReport STLGeometryLoader::sampleIBMMarkersAdaptive(
    const STLMesh& mesh,
    const float3& centerOfMass,
    std::vector<IBMMarker>& outMarkers,
    const MarkerSamplingParams& params
) {
    MarkerSamplingReport report;
    outMarkers.clear();

    float ds = std::max(params.ds_min, std::min(params.ds_requested, params.ds_max));
    if (ds <= 0.0f) {
        throw std::runtime_error("IBM marker sampling: ds must be positive.");
    }

    const float kAreaFactor = 1.0f;
    const float cellSize = ds;
    const float denseThresh = params.nn_p10_min_ratio * ds;
    const float sparseThresh = params.nn_p90_max_ratio * ds;

    std::mt19937 rng(42u);
    std::uniform_real_distribution<float> jitter(0.0f, 1.0f);

    const float tol = params.aabb_tolerance * params.dx;

    float rejectFactor = 0.6f;
    int maxAttemptFactor = 10;
    int noImproveSparseIters = 0;
    int lastMarkerCount = -1;

    for (int iter = 0; iter < params.adaptive_iters; ++iter) {
        outMarkers.clear();
        report = MarkerSamplingReport();
        report.ds_final = ds;
        const float rejectDist = rejectFactor * ds;

        float totalArea = 0.0f;
        float minEdge = std::numeric_limits<float>::max();
        std::vector<float> triAreas(mesh.triangles.size(), 0.0f);
        std::vector<int> triCounts(mesh.triangles.size(), 0);
        std::vector<int> markerTriIdx;

        for (std::size_t i = 0; i < mesh.triangles.size(); ++i) {
            const STLTriangle& tri = mesh.triangles[i];
            float a = triangleArea(tri);
            triAreas[i] = a;
            totalArea += a;
            float e0 = length(tri.vertices[1] - tri.vertices[0]);
            float e1 = length(tri.vertices[2] - tri.vertices[0]);
            float e2 = length(tri.vertices[2] - tri.vertices[1]);
            minEdge = std::min(minEdge, std::min(e0, std::min(e1, e2)));
        }
        if (!std::isfinite(minEdge)) {
            minEdge = 0.0f;
        }

        if (totalArea <= 0.0f) {
            report.fail_reason = "Surface area is zero.";
            report.pass = false;
            if (params.strict_fail) {
                throw std::runtime_error("IBM marker sampling failed: " + report.fail_reason);
            }
            return report;
        }

        std::unordered_map<CellKey, std::vector<int>, CellKeyHash> grid;
        grid.reserve(static_cast<std::size_t>(totalArea / (ds * ds)) + 16);

        for (std::size_t i = 0; i < mesh.triangles.size(); ++i) {
            const STLTriangle& tri = mesh.triangles[i];
            float area = triAreas[i];
            if (area <= 0.0f) continue;

            int target = static_cast<int>(std::round(area / (kAreaFactor * ds * ds)));
            target = std::max(1, target);

            int maxByArea = static_cast<int>(area / (0.2f * ds * ds));
            maxByArea = std::max(1, maxByArea);
            int n_i_max = std::min(2000, maxByArea);
            target = std::min(target, n_i_max);

            int gridSize = static_cast<int>(std::ceil(std::sqrt(static_cast<float>(target))));
            int attempts = 0;
            int accepted = 0;
            int maxAttempts = target * maxAttemptFactor + 10;

            float3 e0 = tri.vertices[1] - tri.vertices[0];
            float3 e1 = tri.vertices[2] - tri.vertices[0];
            float3 triNormal = normalizeSafe(cross(e0, e1));

            for (int s = 0; s < gridSize * gridSize && accepted < target; ++s) {
                if (attempts > maxAttempts) break;
                int si = s % gridSize;
                int sj = s / gridSize;

                float u = (static_cast<float>(si) + jitter(rng)) / static_cast<float>(gridSize);
                float v = (static_cast<float>(sj) + jitter(rng)) / static_cast<float>(gridSize);
                if (u + v > 1.0f) {
                    u = 1.0f - u;
                    v = 1.0f - v;
                }
                float w = 1.0f - u - v;

                float3 pos;
                pos.x = w * tri.vertices[0].x + u * tri.vertices[1].x + v * tri.vertices[2].x;
                pos.y = w * tri.vertices[0].y + u * tri.vertices[1].y + v * tri.vertices[2].y;
                pos.z = w * tri.vertices[0].z + u * tri.vertices[1].z + v * tri.vertices[2].z;

                bool accept = true;
                CellKey key = toCell(pos, cellSize);
                for (int dz = -1; dz <= 1 && accept; ++dz) {
                    for (int dy = -1; dy <= 1 && accept; ++dy) {
                        for (int dx = -1; dx <= 1 && accept; ++dx) {
                            CellKey nk{key.x + dx, key.y + dy, key.z + dz};
                            auto it = grid.find(nk);
                            if (it == grid.end()) continue;
                            for (int idx : it->second) {
                                float3 d = outMarkers[idx].pos - pos;
                                if (length(d) < rejectDist) {
                                    accept = false;
                                    break;
                                }
                            }
                        }
                    }
                }

                attempts++;
                if (!accept) continue;

                IBMMarker marker(pos, pos - centerOfMass);
                marker.normal = triNormal;
                outMarkers.push_back(marker);
                markerTriIdx.push_back(static_cast<int>(i));
                triCounts[i]++;
                grid[key].push_back(static_cast<int>(outMarkers.size() - 1));
                accepted++;

                if (static_cast<int>(outMarkers.size()) >= params.max_markers) {
                    break;
                }
            }

            if (triCounts[i] == 0) {
                float3 centroid = tri.center();
                IBMMarker marker(centroid, centroid - centerOfMass);
                marker.normal = triNormal;
                outMarkers.push_back(marker);
                markerTriIdx.push_back(static_cast<int>(i));
                triCounts[i]++;
                CellKey key = toCell(centroid, cellSize);
                grid[key].push_back(static_cast<int>(outMarkers.size() - 1));
            }

            if (static_cast<int>(outMarkers.size()) >= params.max_markers) {
                break;
            }
        }

        for (std::size_t i = 0; i < outMarkers.size(); ++i) {
            int triIdx = markerTriIdx[i];
            float area = triAreas[triIdx];
            int count = std::max(1, triCounts[triIdx]);
            outMarkers[i].area = area / static_cast<float>(count);
        }

        report.marker_count = static_cast<int>(outMarkers.size());
        report.surface_area = totalArea;
        report.area_per_marker = (report.marker_count > 0)
            ? totalArea / static_cast<float>(report.marker_count)
            : 0.0f;

        const int nonZeroTriCount = static_cast<int>(
            std::count_if(triAreas.begin(), triAreas.end(),
                          [](float a) { return a > 0.0f; })
        );
        const bool atMinPerTri = (report.marker_count <= nonZeroTriCount);
        const float meanTriArea = (nonZeroTriCount > 0)
            ? totalArea / static_cast<float>(nonZeroTriCount)
            : 0.0f;
        const float triBasedDs = (meanTriArea > 0.0f) ? std::sqrt(meanTriArea) : ds;
        const float nnRef = atMinPerTri
            ? std::min(ds, std::max(params.ds_min, triBasedDs))
            : ds;

        int effectiveMinMarkers = params.min_markers;
        if (params.area_per_marker_max > 0.0f && params.ds_min > 0.0f) {
            int feasibleMin = static_cast<int>(
                std::floor(totalArea / (params.area_per_marker_max * params.ds_min * params.ds_min))
            );
            if (feasibleMin < 1) feasibleMin = 1;
            if (effectiveMinMarkers > feasibleMin) {
                effectiveMinMarkers = feasibleMin;
                printf("[IBM Sampling] WARN: min_markers=%d infeasible at ds_min=%.4f, using %d\n",
                       params.min_markers, params.ds_min, effectiveMinMarkers);
            }
        }
        report.effective_min_markers = effectiveMinMarkers;

        int outsideCount = 0;
        std::vector<float3> positions;
        positions.reserve(outMarkers.size());
        for (const auto& m : outMarkers) {
            positions.push_back(m.pos);
            if (!pointInAABB(m.pos, mesh.minBound, mesh.maxBound, tol)) {
                outsideCount++;
            }
        }
        report.aabb_outside_fraction = (report.marker_count > 0)
            ? static_cast<float>(outsideCount) / static_cast<float>(report.marker_count)
            : 0.0f;

        computeNearestNeighborStats(
            positions,
            cellSize,
            report.min_nn,
            report.avg_nn,
            report.max_nn,
            report.p10_nn,
            report.p50_nn,
            report.p90_nn,
            report.too_dense_pairs,
            report.too_sparse_holes,
            denseThresh,
            sparseThresh
        );

        report.tri_point_corr = computePearsonCorr(triAreas, triCounts);

        std::ostringstream fail;
        if (report.marker_count < effectiveMinMarkers) {
            fail << "marker_count < min_markers; ";
        }
        if (report.marker_count > params.max_markers) {
            fail << "marker_count > max_markers; ";
        }
        if (!atMinPerTri && report.area_per_marker < params.area_per_marker_min * ds * ds) {
            fail << "area_per_marker too small; ";
        }
        if (report.area_per_marker > params.area_per_marker_max * ds * ds) {
            fail << "area_per_marker too large; ";
        }
        bool denseMeshByEdge = (minEdge > 0.0f && minEdge < 0.1f * nnRef);
        bool skipMinNnCheck = denseMeshByEdge;
        bool skipP10Check = denseMeshByEdge;
        if (!skipMinNnCheck && report.min_nn < params.nn_min_abs_ratio * nnRef) {
            fail << "min_nn too small; ";
        }
        if (!skipP10Check && report.p10_nn < params.nn_p10_min_ratio * nnRef) {
            fail << "p10_nn too small; ";
        }
        if (report.p90_nn > params.nn_p90_max_ratio * nnRef) {
            fail << "p90_nn too large; ";
        }
        if (report.aabb_outside_fraction > 0.001f) {
            fail << "markers outside AABB; ";
        }

        report.fail_reason = fail.str();
        report.pass = report.fail_reason.empty();

        printf("[IBM Sampling] A=%.6f, ds=%.4f, N=%d, A/N=%.6f\n",
               report.surface_area, ds, report.marker_count, report.area_per_marker);
        printf("[IBM Sampling] nn: min=%.4f avg=%.4f p10=%.4f p50=%.4f p90=%.4f max=%.4f\n",
               report.min_nn, report.avg_nn, report.p10_nn, report.p50_nn,
               report.p90_nn, report.max_nn);
        printf("[IBM Sampling] outside_frac=%.6f, tri_corr=%.3f\n",
               report.aabb_outside_fraction, report.tri_point_corr);

        if (report.pass) {
            ibm::KernelValidationParams kv;
            kv.strict_fail = params.strict_fail;
            ibm::KernelValidationReport kreport = ibm::validatePartitionOfUnity(
                positions, params.dx, ibm::KernelType::Trilinear, kv);
            std::cout << "[IBM Kernel PU] type=" << ibm::kernelTypeName(ibm::KernelType::Trilinear)
                      << " tested=" << kreport.tested
                      << " fail_sum=" << kreport.fail_sum
                      << " fail_negative=" << kreport.fail_negative
                      << " worst_sum_err=" << kreport.worst_sum_err
                      << " min_weight_seen=" << kreport.min_weight_seen
                      << std::endl;
            return report;
        }

        bool tooDense = ((!skipP10Check && report.p10_nn < params.nn_p10_min_ratio * nnRef) ||
                        ((!atMinPerTri && report.area_per_marker < params.area_per_marker_min * ds * ds) ||
                         (atMinPerTri && report.area_per_marker < 0.5f * meanTriArea)) ||
                        (report.marker_count > params.max_markers));
        bool tooSparse = (report.p90_nn > params.nn_p90_max_ratio * nnRef) ||
                         (report.area_per_marker > params.area_per_marker_max * ds * ds) ||
                         (report.marker_count < effectiveMinMarkers);

        if (report.aabb_outside_fraction > 0.001f) {
            if (params.strict_fail) {
                throw std::runtime_error("IBM marker sampling failed: " + report.fail_reason);
            }
            return report;
        }

        if (tooSparse) {
            if (lastMarkerCount >= 0 && report.marker_count <= lastMarkerCount + 1) {
                noImproveSparseIters++;
            } else {
                noImproveSparseIters = 0;
            }
            lastMarkerCount = report.marker_count;
        } else {
            noImproveSparseIters = 0;
            lastMarkerCount = report.marker_count;
        }

        if (tooDense) {
            ds = std::min(ds * 1.2f, params.ds_max);
            rejectFactor = std::min(0.9f, rejectFactor * 1.2f);
            maxAttemptFactor = std::max(10, maxAttemptFactor - 3);
        } else if (tooSparse) {
            ds = std::max(ds * 0.85f, params.ds_min);
            rejectFactor = std::max(0.2f, rejectFactor * 0.7f);
            maxAttemptFactor = std::min(50, maxAttemptFactor + 8);
        } else {
            ds = std::min(ds * 1.1f, params.ds_max);
        }

        if (tooSparse && noImproveSparseIters >= 2 && rejectFactor > 0.0f) {
            printf("[IBM Sampling] WARN: tooSparse persists, disabling min-distance filter.\n");
            rejectFactor = 0.0f;
            maxAttemptFactor = std::min(80, maxAttemptFactor + 20);
        }
    }

    if (params.strict_fail) {
        throw std::runtime_error("IBM marker sampling validation failed: " + report.fail_reason);
    }
    return report;
}

// =============================================================================
// Mesh Vertex Extraction
// =============================================================================

std::vector<float3> STLGeometryLoader::extractMeshVertices(const STLMesh& mesh) {
    std::vector<float3> vertices;
    vertices.reserve(mesh.triangles.size() * 3);

    for (const auto& tri : mesh.triangles) {
        vertices.push_back(tri.vertices[0]);
        vertices.push_back(tri.vertices[1]);
        vertices.push_back(tri.vertices[2]);
    }

    printf("Mesh extraction: %zu vertices\n", vertices.size());

    return vertices;
}

// =============================================================================
// VTK Output Helpers
// =============================================================================

bool STLGeometryLoader::writeMeshVTP(const STLMesh& mesh, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    const std::size_t nTriangles = mesh.triangles.size();
    if (nTriangles == 0) {
        return false;
    }
    const std::size_t nVertices = nTriangles * 3;

    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    file << "  <PolyData>\n";
    file << "    <Piece NumberOfPoints=\"" << nVertices
         << "\" NumberOfVerts=\"0\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\""
         << nTriangles << "\">\n";

    file << "      <Points>\n";
    file << "        <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (const auto& tri : mesh.triangles) {
        for (int k = 0; k < 3; ++k) {
            const float3& v = tri.vertices[k];
            file << v.x << " " << v.y << " " << v.z << " ";
        }
        file << "\n";
    }
    file << "        </DataArray>\n";
    file << "      </Points>\n";

    file << "      <Polys>\n";
    file << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (std::size_t i = 0; i < nTriangles; ++i) {
        file << (i * 3) << " " << (i * 3 + 1) << " " << (i * 3 + 2) << "\n";
    }
    file << "        </DataArray>\n";
    file << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (std::size_t i = 0; i < nTriangles; ++i) {
        file << (i * 3 + 3) << "\n";
    }
    file << "        </DataArray>\n";
    file << "      </Polys>\n";

    file << "    </Piece>\n";
    file << "  </PolyData>\n";
    file << "</VTKFile>\n";
    return true;
}

bool STLGeometryLoader::writeMarkersVTP(const std::vector<IBMMarker>& markers, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    const std::size_t n = markers.size();
    if (n == 0) {
        return false;
    }

    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    file << "  <PolyData>\n";
    file << "    <Piece NumberOfPoints=\"" << n
         << "\" NumberOfVerts=\"" << n
         << "\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n";

    file << "      <Points>\n";
    file << "        <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (const auto& m : markers) {
        file << m.pos.x << " " << m.pos.y << " " << m.pos.z << " ";
    }
    file << "\n";
    file << "        </DataArray>\n";
    file << "      </Points>\n";

    file << "      <PointData>\n";
    file << "        <DataArray type=\"Float32\" Name=\"normal\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (const auto& m : markers) {
        file << m.normal.x << " " << m.normal.y << " " << m.normal.z << " ";
    }
    file << "\n";
    file << "        </DataArray>\n";
    file << "        <DataArray type=\"Float32\" Name=\"area\" NumberOfComponents=\"1\" format=\"ascii\">\n";
    for (const auto& m : markers) {
        file << m.area << " ";
    }
    file << "\n";
    file << "        </DataArray>\n";
    file << "      </PointData>\n";

    file << "      <Verts>\n";
    file << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (std::size_t i = 0; i < n; ++i) {
        file << i << " ";
    }
    file << "\n";
    file << "        </DataArray>\n";
    file << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (std::size_t i = 0; i < n; ++i) {
        file << (i + 1) << " ";
    }
    file << "\n";
    file << "        </DataArray>\n";
    file << "      </Verts>\n";

    file << "    </Piece>\n";
    file << "  </PolyData>\n";
    file << "</VTKFile>\n";
    return true;
}

// =============================================================================
// Voxelization and Point-in-Mesh
// =============================================================================

bool STLGeometryLoader::rayTriangleIntersect(
    const float3& origin, const float3& direction,
    const float3& v0, const float3& v1, const float3& v2,
    float& t
) {
    const float EPSILON = 1e-7f;
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;

    // h = direction × e2
    float3 h;
    h.x = direction.y * e2.z - direction.z * e2.y;
    h.y = direction.z * e2.x - direction.x * e2.z;
    h.z = direction.x * e2.y - direction.y * e2.x;

    float a = e1.x * h.x + e1.y * h.y + e1.z * h.z;
    if (std::fabs(a) < EPSILON) return false;

    float f = 1.0f / a;
    float3 s = origin - v0;
    float u = f * (s.x * h.x + s.y * h.y + s.z * h.z);
    if (u < 0.0f || u > 1.0f) return false;

    // q = s × e1
    float3 q;
    q.x = s.y * e1.z - s.z * e1.y;
    q.y = s.z * e1.x - s.x * e1.z;
    q.z = s.x * e1.y - s.y * e1.x;

    float v = f * (direction.x * q.x + direction.y * q.y + direction.z * q.z);
    if (v < 0.0f || u + v > 1.0f) return false;

    t = f * (e2.x * q.x + e2.y * q.y + e2.z * q.z);
    return t > EPSILON;
}

bool STLGeometryLoader::isInside(const STLMesh& mesh, const float3& point) {
    // Robust point-in-mesh check using majority vote from 3 ray directions (X, Y, Z)
    // This helps handle numerical instabilities and near-edge cases in non-perfect meshes.
    
    int votes = 0;
    float3 directions[3] = {
        make_float3(1.0f, 0.00013f, 0.00017f), // ~X
        make_float3(0.00011f, 1.0f, 0.00019f), // ~Y
        make_float3(0.00015f, 0.00023f, 1.0f)  // ~Z
    };

    for (int i = 0; i < 3; ++i) {
        int intersections = 0;
        for (const auto& tri : mesh.triangles) {
            float t;
            if (rayTriangleIntersect(point, directions[i], tri.vertices[0], tri.vertices[1], tri.vertices[2], t)) {
                intersections++;
            }
        }
        if (intersections % 2 == 1) votes++;
    }

    return votes >= 2; // Majority vote
}

std::vector<int> STLGeometryLoader::voxelize(
    const STLMesh& mesh,
    float voxelSize,
    int3& dims,
    float3& minBound
) {
    minBound = mesh.minBound;
    float3 maxBound = mesh.maxBound;

    // Pad by one voxel to ensure coverage
    minBound.x -= voxelSize * 0.5f;
    minBound.y -= voxelSize * 0.5f;
    minBound.z -= voxelSize * 0.5f;
    maxBound.x += voxelSize * 0.1f; 
    maxBound.y += voxelSize * 0.1f;
    maxBound.z += voxelSize * 0.1f;

    dims.x = (int)((maxBound.x - minBound.x) / voxelSize) + 1;
    dims.y = (int)((maxBound.y - minBound.y) / voxelSize) + 1;
    dims.z = (int)((maxBound.z - minBound.z) / voxelSize) + 1;

    printf("Voxelizing mesh into %d x %d x %d grid...\n", dims.x, dims.y, dims.z);
    std::vector<int> grid(dims.x * dims.y * dims.z, 0);

    for (int k = 0; k < dims.z; ++k) {
        for (int j = 0; j < dims.y; ++j) {
            for (int i = 0; i < dims.x; ++i) {
                float3 p;
                p.x = minBound.x + (i + 0.5f) * voxelSize;
                p.y = minBound.y + (j + 0.5f) * voxelSize;
                p.z = minBound.z + (k + 0.5f) * voxelSize;

                if (isInside(mesh, p)) {
                    grid[k * dims.x * dims.y + j * dims.x + i] = 1;
                }
            }
        }
    }

    return grid;
}
