/**
 * @file IBMMarker.h
 * @brief IBM 标记点数据结构
 *
 * 用于 LBM-IBM 耦合的标记点数据结构。
 * 
 * 设计说明：
 * - 纯数据结构，无 GPU 依赖
 * - 与 FieldStore 架构完全兼容
 * - 标记点位置存储为绝对坐标和相对于质心的坐标
 *
 * @copyright LIDMaster Project
 */
#pragma once

#include "VectorTypes.h"
#include <vector>

/**
 * @brief IBM 标记点
 *
 * 表示浸没边界上的一个 Lagrangian 标记点。
 * 用于 LBM-IBM 耦合中的力计算和速度插值。
 */
struct IBMMarker {
    // =========================================================================
    // 位置
    // =========================================================================
    
    float3 pos;         ///< 绝对位置 [m]
    float3 rel_pos;     ///< 相对于刚体质心的位置 [m]

    // =========================================================================
    // 速度
    // =========================================================================
    
    float3 u_desired;   ///< 期望速度（从刚体运动计算）[m/s]
    float3 u_current;   ///< 当前流体速度（从 LBM 插值）[m/s]

    // =========================================================================
    // 力
    // =========================================================================
    
    float3 force;       ///< 作用在标记点上的流体力 [N]

    // =========================================================================
    // 关联
    // =========================================================================
    
    int particleIdx;    ///< 关联的 DEM 颗粒索引（用于耦合）

    // =========================================================================
    // 扩展属性
    // =========================================================================
    
    float area;         ///< 标记点代表的表面积 [m²]
    float3 normal;      ///< 表面法向量（指向流体侧）

    // =========================================================================
    // 构造函数
    // =========================================================================
    
    IBMMarker()
        : pos(make_float3(0, 0, 0))
        , rel_pos(make_float3(0, 0, 0))
        , u_desired(make_float3(0, 0, 0))
        , u_current(make_float3(0, 0, 0))
        , force(make_float3(0, 0, 0))
        , particleIdx(-1)
        , area(0)
        , normal(make_float3(0, 0, 0)) {}

    IBMMarker(const float3& position, const float3& relative_pos, int idx = -1)
        : pos(position)
        , rel_pos(relative_pos)
        , u_desired(make_float3(0, 0, 0))
        , u_current(make_float3(0, 0, 0))
        , force(make_float3(0, 0, 0))
        , particleIdx(idx)
        , area(0)
        , normal(make_float3(0, 0, 0)) {}
};

/**
 * @brief IBM 标记点集合的辅助函数
 */
namespace IBMMarkerUtils {

/**
 * @brief 计算标记点总面积
 */
inline float totalArea(const std::vector<IBMMarker>& markers) {
    float area = 0;
    for (const auto& m : markers) {
        area += m.area;
    }
    return area;
}

/**
 * @brief 计算标记点总力
 */
inline float3 totalForce(const std::vector<IBMMarker>& markers) {
    float3 f = make_float3(0, 0, 0);
    for (const auto& m : markers) {
        f.x += m.force.x;
        f.y += m.force.y;
        f.z += m.force.z;
    }
    return f;
}

/**
 * @brief 计算标记点总力矩（相对于原点）
 */
inline float3 totalTorque(const std::vector<IBMMarker>& markers, const float3& pivot) {
    float3 t = make_float3(0, 0, 0);
    for (const auto& m : markers) {
        float3 r = m.pos - pivot;
        float3 tau = cross(r, m.force);
        t.x += tau.x;
        t.y += tau.y;
        t.z += tau.z;
    }
    return t;
}

/**
 * @brief 将标记点转换为 flat 数组（用于 FieldStore）
 * @param markers 标记点向量
 * @param positions 输出位置数组 [x0,y0,z0,x1,y1,z1,...]
 */
inline void toFlatPositions(const std::vector<IBMMarker>& markers, std::vector<float>& positions) {
    positions.resize(markers.size() * 3);
    for (std::size_t i = 0; i < markers.size(); ++i) {
        positions[i * 3 + 0] = markers[i].pos.x;
        positions[i * 3 + 1] = markers[i].pos.y;
        positions[i * 3 + 2] = markers[i].pos.z;
    }
}

/**
 * @brief 将标记点转换为 flat 力数组（用于 FieldStore）
 */
inline void toFlatForces(const std::vector<IBMMarker>& markers, std::vector<float>& forces) {
    forces.resize(markers.size() * 3);
    for (std::size_t i = 0; i < markers.size(); ++i) {
        forces[i * 3 + 0] = markers[i].force.x;
        forces[i * 3 + 1] = markers[i].force.y;
        forces[i * 3 + 2] = markers[i].force.z;
    }
}

} // namespace IBMMarkerUtils
