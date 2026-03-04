#include "PlanetaryPaddleModule.hpp"
#include <cmath>
#include <limits>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;

namespace {
struct Tri {
    float3 v0;
    float3 v1;
    float3 v2;
};

float pointTriangleDistanceSquared(const float3& p, const Tri& t) {
    const float3 ab = t.v1 - t.v0;
    const float3 ac = t.v2 - t.v0;
    const float3 ap = p - t.v0;

    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        return dot(ap, ap);
    }

    const float3 bp = p - t.v1;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) {
        return dot(bp, bp);
    }

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        const float v = d1 / (d1 - d3);
        const float3 proj = t.v0 + ab * v;
        return dot(p - proj, p - proj);
    }

    const float3 cp = p - t.v2;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) {
        return dot(cp, cp);
    }

    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        const float w = d2 / (d2 - d6);
        const float3 proj = t.v0 + ac * w;
        return dot(p - proj, p - proj);
    }

    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        const float3 proj = t.v1 + (t.v2 - t.v1) * w;
        return dot(p - proj, p - proj);
    }

    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    const float3 proj = t.v0 + ab * v + ac * w;
    return dot(p - proj, p - proj);
}
} // namespace

PlanetaryPaddleModule::PlanetaryPaddleModule(const Config& config)
    : config_(config) {
    axis_ = normalizeAxis(config_.axis);
    computeAngularVelocities();
}

bool PlanetaryPaddleModule::load() {
    near_mesh_.triangles.clear();
    far_mesh_.triangles.clear();

    if (!STLReader::readSTL(config_.near_stl, near_mesh_)) {
        return false;
    }
    if (!STLReader::readSTL(config_.far_stl, far_mesh_)) {
        return false;
    }

    near_mesh_.centerAtOrigin();
    far_mesh_.centerAtOrigin();

    loaded_ = true;
    return true;
}

void PlanetaryPaddleModule::writeVTPSeries(const std::string& output_dir, int frames, double dt_sec) const {
    if (!loaded_) return;

    fs::create_directories(output_dir);
    std::ostringstream pvd_name;
    pvd_name << output_dir << "/planetary_paddle.pvd";
    std::ofstream pvd(pvd_name.str());
    if (pvd.is_open()) {
        pvd << "<?xml version=\"1.0\"?>\n";
        pvd << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        pvd << "  <Collection>\n";
    }

    for (int i = 0; i < frames; ++i) {
        double t = static_cast<double>(i) * dt_sec;
        writeVTPFrame(output_dir, i, t);
        if (pvd.is_open()) {
            std::ostringstream vtp_name;
            vtp_name << "planetary_paddle_" << std::setfill('0') << std::setw(6) << i << ".vtp";
            pvd << "    <DataSet timestep=\"" << (t * 1000.0)
                << "\" group=\"\" part=\"0\" file=\"" << vtp_name.str() << "\"/>\n";
        }
    }
    if (pvd.is_open()) {
        pvd << "  </Collection>\n";
        pvd << "</VTKFile>\n";
    }
}

float PlanetaryPaddleModule::minDistance(double time_sec, int stride) const {
    if (!loaded_) return 0.0f;
    if (stride < 1) stride = 1;

    const float kPi = 3.14159265358979323846f;
    float rev_angle = static_cast<float>(revolution_omega_ * time_sec)
                      + degToRad(config_.revolution_phase_deg);
    float near_orbit_angle = rev_angle + degToRad(config_.near_orbit_phase_deg);
    float far_rev_angle = near_orbit_angle + kPi;
    float near_spin_angle = static_cast<float>(near_spin_omega_ * time_sec)
                            + degToRad(config_.near_spin_phase_deg);
    float far_spin_angle = static_cast<float>(far_spin_omega_ * time_sec)
                           + degToRad(config_.far_spin_phase_deg);
    // Always carry orbit orientation into spin (locks effective far spin = self + revolution).
    near_spin_angle += near_orbit_angle;
    far_spin_angle += far_rev_angle;

    Centers centers = computeCenters(time_sec);

    std::vector<Tri> near_tris;
    std::vector<Tri> far_tris;
    near_tris.reserve(near_mesh_.triangles.size() / stride + 1);
    far_tris.reserve(far_mesh_.triangles.size() / stride + 1);

    for (size_t i = 0; i < near_mesh_.triangles.size(); i += static_cast<size_t>(stride)) {
        const auto& tri = near_mesh_.triangles[i];
        Tri t;
        t.v0 = centers.near_center + rotateAroundAxis(tri.vertices[0], axis_, near_spin_angle);
        t.v1 = centers.near_center + rotateAroundAxis(tri.vertices[1], axis_, near_spin_angle);
        t.v2 = centers.near_center + rotateAroundAxis(tri.vertices[2], axis_, near_spin_angle);
        near_tris.push_back(t);
    }
    for (size_t i = 0; i < far_mesh_.triangles.size(); i += static_cast<size_t>(stride)) {
        const auto& tri = far_mesh_.triangles[i];
        Tri t;
        t.v0 = centers.far_center + rotateAroundAxis(tri.vertices[0], axis_, far_spin_angle);
        t.v1 = centers.far_center + rotateAroundAxis(tri.vertices[1], axis_, far_spin_angle);
        t.v2 = centers.far_center + rotateAroundAxis(tri.vertices[2], axis_, far_spin_angle);
        far_tris.push_back(t);
    }

    float min_dist2 = std::numeric_limits<float>::max();
    for (const auto& near_t : near_tris) {
        const float3 near_pts[3] = {near_t.v0, near_t.v1, near_t.v2};
        for (const auto& far_t : far_tris) {
            for (const auto& p : near_pts) {
                float d2 = pointTriangleDistanceSquared(p, far_t);
                if (d2 < min_dist2) {
                    min_dist2 = d2;
                }
            }
        }
    }
    for (const auto& far_t : far_tris) {
        const float3 far_pts[3] = {far_t.v0, far_t.v1, far_t.v2};
        for (const auto& near_t : near_tris) {
            for (const auto& p : far_pts) {
                float d2 = pointTriangleDistanceSquared(p, near_t);
                if (d2 < min_dist2) {
                    min_dist2 = d2;
                }
            }
        }
    }

    return std::sqrt(min_dist2);
}

void PlanetaryPaddleModule::writeVTPFrame(const std::string& output_dir, int frame, double time_sec) const {
    if (!loaded_) return;

    std::ostringstream filename;
    filename << output_dir << "/planetary_paddle_"
             << std::setfill('0') << std::setw(6) << frame << ".vtp";
    writeVTP(filename.str(), time_sec);
    std::cout << "Frame " << frame << " t=" << time_sec << "s" << std::endl;
}

float PlanetaryPaddleModule::rpmToRad(float rpm) {
    const float kPi = 3.14159265358979323846f;
    return rpm * (2.0f * kPi / 60.0f);
}

float PlanetaryPaddleModule::degToRad(float deg) {
    const float kPi = 3.14159265358979323846f;
    return deg * (kPi / 180.0f);
}

float3 PlanetaryPaddleModule::normalizeAxis(const float3& axis) {
    float len = length(axis);
    if (len <= 1e-6f) {
        return make_float3(0.0f, 0.0f, 1.0f);
    }
    return axis / len;
}

float3 PlanetaryPaddleModule::rotateAroundAxis(const float3& v, const float3& axis, float angle) {
    float c = std::cos(angle);
    float s = std::sin(angle);
    return v * c + cross(axis, v) * s + axis * (dot(axis, v) * (1.0f - c));
}

void PlanetaryPaddleModule::computeAngularVelocities() {
    // Convention: positive RPM means clockwise when looking along +axis.
    float rev = -rpmToRad(config_.revolution_rpm);
    float near_spin = -rpmToRad(config_.near_spin_rpm);
    float far_spin = -rpmToRad(config_.far_spin_rpm);

    if (config_.mode == RotationMode::Forward) {
        revolution_omega_ = rev;
        near_spin_omega_ = near_spin;
        far_spin_omega_ = far_spin;
    } else {
        // Reverse mode: Flip all signs
        revolution_omega_ = -rev;
        near_spin_omega_ = -near_spin;
        far_spin_omega_ = -far_spin;
    }
}

PlanetaryPaddleModule::Centers PlanetaryPaddleModule::computeCenters(double time_sec) const {
    const float kPi = 3.14159265358979323846f;
    float rev_angle = static_cast<float>(revolution_omega_ * time_sec)
                      + degToRad(config_.revolution_phase_deg);
    float near_orbit_angle = rev_angle + degToRad(config_.near_orbit_phase_deg);
    float far_rev_angle = near_orbit_angle + kPi;

    float3 near_offset = rotateAroundAxis(make_float3(config_.near_offset, 0.0f, 0.0f), axis_, near_orbit_angle);
    float3 far_offset = rotateAroundAxis(make_float3(config_.far_offset, 0.0f, 0.0f), axis_, far_rev_angle);

    Centers centers;
    centers.near_center = config_.center + near_offset;
    centers.far_center = config_.center + far_offset;
    return centers;
}

void PlanetaryPaddleModule::writeVTP(const std::string& path, double time_sec) const {
    const size_t near_tri = near_mesh_.triangles.size();
    const size_t far_tri = far_mesh_.triangles.size();
    const size_t total_tri = near_tri + far_tri;
    const size_t total_pts = total_tri * 3;

    std::ofstream file(path);
    if (!file.is_open()) return;

    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    file << "  <PolyData>\n";
    file << "    <Piece NumberOfPoints=\"" << total_pts
         << "\" NumberOfVerts=\"0\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\""
         << total_tri << "\">\n";

    file << "      <FieldData>\n";
    file << "        <DataArray type=\"Float32\" Name=\"time_ms\" NumberOfTuples=\"1\" format=\"ascii\">\n";
    file << static_cast<float>(time_sec * 1000.0) << "\n";
    file << "        </DataArray>\n";
    file << "      </FieldData>\n";

    file << "      <Points>\n";
    file << "        <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n";

    const float kPi = 3.14159265358979323846f;
    float rev_angle = static_cast<float>(revolution_omega_ * time_sec)
                      + degToRad(config_.revolution_phase_deg);
    float near_orbit_angle = rev_angle + degToRad(config_.near_orbit_phase_deg);
    float far_rev_angle = near_orbit_angle + kPi;
    float near_spin_angle = static_cast<float>(near_spin_omega_ * time_sec)
                            + degToRad(config_.near_spin_phase_deg);
    float far_spin_angle = static_cast<float>(far_spin_omega_ * time_sec)
                           + degToRad(config_.far_spin_phase_deg);
    // Always carry orbit orientation into spin.
    near_spin_angle += near_orbit_angle;
    far_spin_angle += far_rev_angle;

    Centers centers = computeCenters(time_sec);

    auto write_mesh = [&](const STLMesh& mesh, float spin_angle, const float3& center) {
        for (const auto& tri : mesh.triangles) {
            for (int k = 0; k < 3; ++k) {
                float3 v = tri.vertices[k];
                float3 v_spin = rotateAroundAxis(v, axis_, spin_angle);
                float3 v_world = center + v_spin;
                file << v_world.x << " " << v_world.y << " " << v_world.z << " ";
            }
            file << "\n";
        }
    };

    write_mesh(near_mesh_, near_spin_angle, centers.near_center);
    write_mesh(far_mesh_, far_spin_angle, centers.far_center);

    file << "        </DataArray>\n";
    file << "      </Points>\n";

    file << "      <Polys>\n";
    file << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (size_t i = 0; i < total_tri; ++i) {
        file << (i * 3) << " " << (i * 3 + 1) << " " << (i * 3 + 2) << "\n";
    }
    file << "        </DataArray>\n";
    file << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (size_t i = 0; i < total_tri; ++i) {
        file << (i * 3 + 3) << "\n";
    }
    file << "        </DataArray>\n";
    file << "      </Polys>\n";

    file << "    </Piece>\n";
    file << "  </PolyData>\n";
    file << "</VTKFile>\n";
}

