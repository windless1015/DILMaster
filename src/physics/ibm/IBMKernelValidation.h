#pragma once

#include "IBMKernel.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace ibm {

struct KernelValidationParams {
    float sum_tol = 1e-6f;
    float neg_tol = -1e-8f;
    int max_report = 10;
    int sample_count = 1024;
    bool strict_fail = true;
};

struct KernelValidationReport {
    int tested = 0;
    int fail_sum = 0;
    int fail_negative = 0;
    float worst_sum_err = 0.0f;
    float min_weight_seen = 0.0f;
};

inline KernelValidationReport validatePartitionOfUnity(
    const std::vector<float3>& marker_positions,
    float dx,
    KernelType type,
    const KernelValidationParams& p,
    const float3& minBound = make_float3(0, 0, 0)
) {
    KernelValidationReport report;
    if (marker_positions.empty() || dx <= 0.0f) {
        report.min_weight_seen = 0.0f;
        return report;
    }

    const int n = static_cast<int>(marker_positions.size());
    const int target = std::min(p.sample_count, n);

    std::vector<int> indices(n);
    for (int i = 0; i < n; ++i) indices[i] = i;

    std::mt19937 rng(42u);
    std::shuffle(indices.begin(), indices.end(), rng);

    report.min_weight_seen = std::numeric_limits<float>::infinity();
    int reported = 0;

    for (int s = 0; s < target; ++s) {
        const int idx = indices[s];
        const float3& pos = marker_positions[idx];

        KernelWeights weights = computeDeltaWeights(pos, minBound, dx, type);
        float sum_w = 0.0f;
        float min_w = std::numeric_limits<float>::infinity();

        for (int i = 0; i < weights.count; ++i) {
            const float w = weights.items[i].w;
            sum_w += w;
            if (w < min_w) min_w = w;
        }

        if (!std::isfinite(min_w)) min_w = 0.0f;
        const float sum_err = std::abs(sum_w - 1.0f);
        report.worst_sum_err = std::max(report.worst_sum_err, sum_err);
        report.min_weight_seen = std::min(report.min_weight_seen, min_w);

        if (sum_err > p.sum_tol) {
            report.fail_sum++;
            if (reported < p.max_report) {
                std::cerr << "[IBM Kernel PU] sum(w) mismatch at idx=" << idx
                          << " pos=" << pos
                          << " sum=" << std::setprecision(9) << sum_w
                          << " err=" << sum_err
                          << " kernel=" << kernelTypeName(type) << "\n";
                reported++;
            }
        }

        if (min_w < p.neg_tol) {
            report.fail_negative++;
            if (reported < p.max_report) {
                std::cerr << "[IBM Kernel PU] negative weight at idx=" << idx
                          << " pos=" << pos
                          << " min_w=" << std::setprecision(9) << min_w
                          << " kernel=" << kernelTypeName(type) << "\n";
                reported++;
            }
        }

        report.tested++;
    }

    if (p.strict_fail && (report.fail_sum > 0 || report.fail_negative > 0)) {
        std::ostringstream ss;
        ss << "IBM kernel partition-of-unity validation failed: "
           << "tested=" << report.tested
           << " fail_sum=" << report.fail_sum
           << " fail_negative=" << report.fail_negative
           << " worst_sum_err=" << report.worst_sum_err
           << " min_weight_seen=" << report.min_weight_seen
           << " kernel=" << kernelTypeName(type);
        throw std::runtime_error(ss.str());
    }

    if (!std::isfinite(report.min_weight_seen)) {
        report.min_weight_seen = 0.0f;
    }

    return report;
}

} // namespace ibm
