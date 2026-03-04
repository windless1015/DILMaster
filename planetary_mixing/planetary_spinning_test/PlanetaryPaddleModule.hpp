#pragma once

#include "STLReader.h"
#include "VectorTypes.h"
#include <string>

#ifndef PROJECT_ROOT_DIR
#define PROJECT_ROOT_DIR "."
#endif

class PlanetaryPaddleModule {
public:
    enum class RotationMode {
        Forward,
        Reverse
    };

    struct Config {
        std::string near_stl = std::string(PROJECT_ROOT_DIR) + "/tools/pp_close.stl";
        std::string far_stl = std::string(PROJECT_ROOT_DIR) + "/tools/pp_far.stl";
        float3 center = {0.0f, 0.0f, 0.0f};
        float3 axis = {0.0f, 1.0f, 0.0f};
        float near_offset = 0.01475f; // 14.75 mm

        float far_offset = 0.0295f;  // 29.5 mm
        float revolution_rpm = 30.0f;
        float revolution_phase_deg = 0.0f;
        float near_orbit_phase_deg = 0.0f;
        float near_spin_rpm = 30.0f;
        float near_spin_phase_deg = 0.0f;
        float far_spin_rpm = 60.0f;
        float far_spin_phase_deg = 0.0f;
        RotationMode mode = RotationMode::Forward;
    };

    explicit PlanetaryPaddleModule(const Config& config);

    bool load();
    void writeVTPFrame(const std::string& output_dir, int frame, double time_sec) const;
    void writeVTPSeries(const std::string& output_dir, int frames, double dt_sec) const;
    float minDistance(double time_sec, int stride) const;

    const STLMesh& nearMesh() const { return near_mesh_; }
    const STLMesh& farMesh() const { return far_mesh_; }

private:
    struct Centers {
        float3 near_center;
        float3 far_center;
    };

    Config config_;
    STLMesh near_mesh_;
    STLMesh far_mesh_;
    float3 axis_;
    float revolution_omega_ = 0.0f;
    float near_spin_omega_ = 0.0f;
    float far_spin_omega_ = 0.0f;
    bool loaded_ = false;

    static float rpmToRad(float rpm);
    static float degToRad(float deg);
    static float3 normalizeAxis(const float3& axis);
    static float3 rotateAroundAxis(const float3& v, const float3& axis, float angle);
    void computeAngularVelocities();
    Centers computeCenters(double time_sec) const;
    void writeVTP(const std::string& path, double time_sec) const;

};
