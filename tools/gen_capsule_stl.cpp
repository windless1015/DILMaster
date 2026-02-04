#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <algorithm>
#include <iomanip>

// Use M_PI if available, else define
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Simple Vector3 Struct
struct Vec3 {
    double x, y, z;
};

// Triangle Struct for STL
struct Triangle {
    Vec3 v1, v2, v3;
    Vec3 normal;
};

// Utils
Vec3 rotateZ(Vec3 p, double angle_rad) {
    double c = std::cos(angle_rad);
    double s = std::sin(angle_rad);
    return { p.x * c - p.y * s, p.x * s + p.y * c, p.z };
}

Vec3 add(Vec3 a, Vec3 b) {
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}

Vec3 sub(Vec3 a, Vec3 b) {
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

Vec3 cross(Vec3 a, Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

Vec3 normalize(Vec3 v) {
    double len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len < 1e-9) return {0,0,0};
    return { v.x / len, v.y / len, v.z / len };
}

void writeSTL(const std::string& filename, const std::vector<Triangle>& tris) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error opening " << filename << std::endl;
        return;
    }

    out << "solid capsule" << std::endl;
    for (const auto& t : tris) {
        out << "  facet normal " << t.normal.x << " " << t.normal.y << " " << t.normal.z << std::endl;
        out << "    outer loop" << std::endl;
        out << "      vertex " << t.v1.x << " " << t.v1.y << " " << t.v1.z << std::endl;
        out << "      vertex " << t.v2.x << " " << t.v2.y << " " << t.v2.z << std::endl;
        out << "      vertex " << t.v3.x << " " << t.v3.y << " " << t.v3.z << std::endl;
        out << "    endloop" << std::endl;
        out << "  endfacet" << std::endl;
    }
    out << "endsolid capsule" << std::endl;
    std::cout << "Wrote " << tris.size() << " triangles to " << filename << std::endl;
}

int main(int argc, char** argv) {
    // Defaults
    double R = 10.0;
    double Lc = 60.0; // Cylinder Length
    double AR = 0.0;  // If set > 0, Lc is derived: AR = (Lc+2R)/(2R)
    double angle_deg = 0.0;
    double cx = 0.0, cy = 0.0, cz = 0.0;
    std::string output = "capsule.stl";
    
    // Mesh Resolution
    int n_theta = 32; // Circumference
    int n_phi = 16;   // Cap Latitude
    int n_z = 10;     // Cylinder Length segments

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--R") R = std::stod(argv[++i]);
        if (arg == "--Lc") Lc = std::stod(argv[++i]);
        if (arg == "--AR") AR = std::stod(argv[++i]);
        if (arg == "--angle") angle_deg = std::stod(argv[++i]);
        if (arg == "--center") {
            cx = std::stod(argv[++i]);
            cy = std::stod(argv[++i]);
            cz = std::stod(argv[++i]);
        }
        if (arg == "--out") output = argv[++i];
    }

    if (AR > 0.0) {
        // AR = (Lc + 2R) / 2R => 2R*AR = Lc + 2R => Lc = 2R(AR - 1)
        Lc = 2.0 * R * (AR - 1.0);
        if (Lc < 0) Lc = 0; // Sphere
        std::cout << "Derived Lc=" << Lc << " from AR=" << AR << " R=" << R << std::endl;
    }

    std::cout << "Generating Capsule: R=" << R << " Lc=" << Lc 
              << " Angle=" << angle_deg << " Center=(" << cx << "," << cy << "," << cz << ")" 
              << " Output=" << output << std::endl;

    std::vector<Triangle> triangles;
    double halfL = Lc / 2.0;

    // Helper to add quad as 2 triangles with Automatic Outward Normal
    auto addQuad = [&](Vec3 p1, Vec3 p2, Vec3 p3, Vec3 p4) {
        Vec3 geometric_center = {cx, cy, cz};
        
        auto addTriRobust = [&](Vec3 a, Vec3 b, Vec3 c) {
             Vec3 n = normalize(cross(sub(b, a), sub(c, a)));
             Vec3 tri_center = { (a.x+b.x+c.x)/3.0, (a.y+b.y+c.y)/3.0, (a.z+b.z+c.z)/3.0 };
             Vec3 dir = sub(tri_center, geometric_center);
             
             // Dot product
             double dot = n.x*dir.x + n.y*dir.y + n.z*dir.z;
             if (dot < 0) {
                 // Flip
                 triangles.push_back({ a, c, b, {-n.x, -n.y, -n.z} });
             } else {
                 triangles.push_back({ a, b, c, n });
             }
        };

        // 1-2-3
        addTriRobust(p1, p2, p3);
        // 1-3-4
        addTriRobust(p1, p3, p4);
    };

    double angle_rad = angle_deg * M_PI / 180.0;

    // 1. Cylinder Body (x from -halfL to +halfL)
    if (Lc > 1e-6) {
        for (int i = 0; i < n_z; ++i) {
            double z1 = -halfL + (Lc * i) / n_z;
            double z2 = -halfL + (Lc * (i + 1)) / n_z;

            for (int j = 0; j < n_theta; ++j) {
                double t1 = (2.0 * M_PI * j) / n_theta;
                double t2 = (2.0 * M_PI * (j + 1)) / n_theta;

                // Local coords (x along axis)
                // y = R cos t, z = R sin t
                Vec3 p1_local = { z1, R * std::cos(t1), R * std::sin(t1) };
                Vec3 p2_local = { z2, R * std::cos(t1), R * std::sin(t1) };
                Vec3 p3_local = { z2, R * std::cos(t2), R * std::sin(t2) };
                Vec3 p4_local = { z1, R * std::cos(t2), R * std::sin(t2) };

                auto transform = [&](Vec3 p) {
                    p = rotateZ(p, angle_rad);
                    return add(p, {cx, cy, cz});
                };

                addQuad(transform(p1_local), transform(p2_local), transform(p3_local), transform(p4_local));
            }
        }
    }

    // 2. Front Cap (x > halfL) -> Hemisphere centered at (halfL, 0, 0)
    // 3. Back Cap (x < -halfL) -> Hemisphere centered at (-halfL, 0, 0)
    
    // Generic Hemisphere Generator
    // polarity: +1 for front (x+), -1 for back (x-)
    auto addCap = [&](double centerX, double polarity) {
        for (int i = 0; i < n_phi; ++i) {
            // phi goes from 0 to pi/2
            double phi1 = (M_PI / 2.0 * i) / n_phi;
            double phi2 = (M_PI / 2.0 * (i + 1)) / n_phi;

            for (int j = 0; j < n_theta; ++j) {
                double t1 = (2.0 * M_PI * j) / n_theta;
                double t2 = (2.0 * M_PI * (j + 1)) / n_theta;

                // Spherical Coords relative to cap center
                // x = R * sin(phi) * polarity
                // y = R * cos(phi) * cos(theta)
                // z = R * cos(phi) * sin(theta)
                // WAIT: To stitch with cylinder (y=R cos t, z= R sin t) at phi=0:
                // We want radius to decrease as x increases away from cylinder.
                // Circle at base: x=0 (relative), radius R.
                // x_rel = R * sin(phi)
                // r_rel = R * cos(phi)
                // Then y = r_rel * cos(theta), z = r_rel * sin(theta)
                
                auto capVertex = [&](double phi, double theta) {
                    double r_loc = R * std::cos(phi);
                    double x_loc = R * std::sin(phi) * polarity;
                    double y = r_loc * std::cos(theta);
                    double z = r_loc * std::sin(theta);
                    return Vec3{ centerX + x_loc, y, z };
                };

                Vec3 p1 = capVertex(phi1, t1);
                Vec3 p2 = capVertex(phi2, t1);
                Vec3 p3 = capVertex(phi2, t2);
                Vec3 p4 = capVertex(phi1, t2);

                auto transform = [&](Vec3 p) {
                    p = rotateZ(p, angle_rad);
                    return add(p, {cx, cy, cz});
                };
                
                // Winding order?
                // Outside surface normal.
                // If polarity +1 (Front): x increases. Surface points out (positive x).
                // p1 (phi small) -> p2 (phi large). x increases.
                // 1-2-3-4?
                // Cylinder winding was CCW around X axis?
                // Visual check in ParaView will confirm. Assuming CCW.
                if (polarity > 0)
                    addQuad(transform(p1), transform(p2), transform(p3), transform(p4));
                else
                    // Back cap: x decreases (moves left). Normal should point left (-x).
                    // Revert winding
                    addQuad(transform(p1), transform(p4), transform(p3), transform(p2));
            }
        }
    };

    addCap(halfL, 1.0);  // Front
    addCap(-halfL, -1.0); // Back

    writeSTL(output, triangles);
    return 0;
}
