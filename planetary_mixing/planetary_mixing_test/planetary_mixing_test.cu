/**
 * planetary_mixing_test.cu
 *
 * Phase 1: Planetary paddle mixing in cylindrical container
 *   - LBM free-surface fluid solver
 *   - Cylinder wall via SOLID flags
 *   - Top 30% = GAS, Bottom 70% = FLUID
 *   - Paddle VTP + Cylinder VTP visualization
 */

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "physics/lbm/LBMCore.hpp"
#include "physics/lbm/FreeSurfaceModule.hpp"
#include "physics/ibm/IBMCore.hpp"
#include "geometry/STLGeometryLoader.hpp"
#include "geometry/IBMMarker.h"
#include "services/VTKService.hpp"
#include "core/FieldStore.hpp"
#include "core/StepContext.hpp"
#include "CylinderDomain.hpp"

// Reuse paddle module
#include "../planetary_spinning_test/PlanetaryPaddleModule.hpp"

namespace fs = std::filesystem;

#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Cylinder surface VTP output (static geometry, written once)
// ============================================================================
void writeCylinderVTP(const std::string& path,
                      float cx, float cz, float radius,
                      int ny_height, int n_segments = 64) {
    std::ofstream file(path);
    if (!file.is_open()) return;

    // Cylinder side + bottom triangular mesh
    // Side: 2 * n_segments triangles
    // Bottom: n_segments triangles (fan)
    int n_side_pts = n_segments * 2;  // top + bottom rings
    int n_bottom_pts = n_segments + 1; // center + edge
    int total_pts = n_side_pts + n_bottom_pts;
    int n_side_tri = n_segments * 2;
    int n_bottom_tri = n_segments;
    int total_tri = n_side_tri + n_bottom_tri;

    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    file << "  <PolyData>\n";
    file << "    <Piece NumberOfPoints=\"" << total_pts
         << "\" NumberOfVerts=\"0\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\""
         << total_tri << "\">\n";

    // Points
    file << "      <Points>\n";
    file << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";

    // Bottom ring (y=0)
    for (int i = 0; i < n_segments; ++i) {
        float angle = 2.0f * (float)M_PI * i / n_segments;
        float x = cx + radius * cosf(angle);
        float z = cz + radius * sinf(angle);
        file << x << " 0 " << z << " ";
    }
    file << "\n";
    // Top ring (y=ny-1)
    for (int i = 0; i < n_segments; ++i) {
        float angle = 2.0f * (float)M_PI * i / n_segments;
        float x = cx + radius * cosf(angle);
        float z = cz + radius * sinf(angle);
        file << x << " " << (ny_height - 1) << " " << z << " ";
    }
    file << "\n";
    // Bottom center
    file << cx << " 0 " << cz << "\n";
    // Bottom edge = same as bottom ring
    for (int i = 0; i < n_segments; ++i) {
        float angle = 2.0f * (float)M_PI * i / n_segments;
        float x = cx + radius * cosf(angle);
        float z = cz + radius * sinf(angle);
        file << x << " 0 " << z << " ";
    }
    file << "\n";

    file << "        </DataArray>\n";
    file << "      </Points>\n";

    // Polys
    file << "      <Polys>\n";
    file << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";

    // Side triangles
    for (int i = 0; i < n_segments; ++i) {
        int i_next = (i + 1) % n_segments;
        // lower triangle
        file << i << " " << i_next << " " << (n_segments + i) << "\n";
        // upper triangle
        file << i_next << " " << (n_segments + i_next) << " " << (n_segments + i) << "\n";
    }
    // Bottom triangles
    int center_idx = n_side_pts;
    int base_idx = n_side_pts + 1;
    for (int i = 0; i < n_segments; ++i) {
        int i_next = (i + 1) % n_segments;
        file << center_idx << " " << (base_idx + i) << " " << (base_idx + i_next) << "\n";
    }

    file << "        </DataArray>\n";
    file << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (int i = 0; i < total_tri; ++i) {
        file << (i + 1) * 3 << "\n";
    }
    file << "        </DataArray>\n";
    file << "      </Polys>\n";

    file << "    </Piece>\n";
    file << "  </PolyData>\n";
    file << "</VTKFile>\n";

    std::cout << "[Cylinder VTP] Written: " << path << " (" << total_tri << " triangles)" << std::endl;
}

// ============================================================================
// Paddle frame VTP output (in lattice coordinates)
// ============================================================================
void writePaddleFrameVTP(const std::string& path,
                         const PlanetaryPaddleModule& module,
                         const PlanetaryPaddleModule::Config& cfg,
                         double time_sec,
                         float3 center_lattice,
                         float scale)
{
    const auto& near_mesh = module.nearMesh();
    const auto& far_mesh  = module.farMesh();
    size_t near_tri_n = near_mesh.triangles.size();
    size_t far_tri_n  = far_mesh.triangles.size();
    size_t total_tri  = near_tri_n + far_tri_n;
    size_t total_pts  = total_tri * 3;

    // Mirror PlanetaryPaddleModule::computeAngularVelocities() + writeVTP()
    float kPi = (float)M_PI;
    auto rpmToRad = [kPi](float rpm) { return rpm * (2.0f * kPi / 60.0f); };
    auto degToRad = [kPi](float deg) { return deg * (kPi / 180.0f); };

    // Angular velocities (same sign convention as PlanetaryPaddleModule)
    float rev_omega       = -rpmToRad(cfg.revolution_rpm);
    float near_spin_omega = -rpmToRad(cfg.near_spin_rpm);
    float far_spin_omega  = -rpmToRad(cfg.far_spin_rpm);

    // Angles (matching writeVTP exactly)
    float rev_angle = rev_omega * (float)time_sec + degToRad(cfg.revolution_phase_deg);
    float near_orbit_angle = rev_angle + degToRad(cfg.near_orbit_phase_deg);
    float far_rev_angle = near_orbit_angle + kPi;

    float near_spin_angle = near_spin_omega * (float)time_sec + degToRad(cfg.near_spin_phase_deg);
    float far_spin_angle  = far_spin_omega * (float)time_sec + degToRad(cfg.far_spin_phase_deg);
    // Carry orbit orientation into spin
    near_spin_angle += near_orbit_angle;
    far_spin_angle  += far_rev_angle;

    // Paddle center positions (lattice coordinates)
    float near_offset_lu = cfg.near_offset * scale;
    float far_offset_lu  = cfg.far_offset * scale;

    float3 axis = make_float3(0, 1, 0);

    auto rotateAxis = [](const float3& v, const float3& ax, float angle) -> float3 {
        float c = cosf(angle), s = sinf(angle);
        float d = ax.x * v.x + ax.y * v.y + ax.z * v.z;
        return make_float3(
            v.x * c + (ax.y * v.z - ax.z * v.y) * s + ax.x * d * (1 - c),
            v.y * c + (ax.z * v.x - ax.x * v.z) * s + ax.y * d * (1 - c),
            v.z * c + (ax.x * v.y - ax.y * v.x) * s + ax.z * d * (1 - c)
        );
    };

    float3 near_offset_vec = rotateAxis(make_float3(near_offset_lu, 0, 0), axis, near_orbit_angle);
    float3 far_offset_vec  = rotateAxis(make_float3(far_offset_lu, 0, 0), axis, far_rev_angle);

    float3 near_center = make_float3(center_lattice.x + near_offset_vec.x,
                                      center_lattice.y + near_offset_vec.y,
                                      center_lattice.z + near_offset_vec.z);
    float3 far_center  = make_float3(center_lattice.x + far_offset_vec.x,
                                      center_lattice.y + far_offset_vec.y,
                                      center_lattice.z + far_offset_vec.z);

    std::ofstream file(path);
    if (!file.is_open()) return;

    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    file << "  <PolyData>\n";
    file << "    <Piece NumberOfPoints=\"" << total_pts
         << "\" NumberOfVerts=\"0\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\""
         << total_tri << "\">\n";

    file << "      <Points>\n";
    file << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";

    auto write_mesh = [&](const STLMesh& mesh, float spin_angle, const float3& center) {
        for (const auto& tri : mesh.triangles) {
            for (int k = 0; k < 3; ++k) {
                // scale -> rotate -> translate

                float3 v = make_float3(tri.vertices[k].x * scale,
                                        tri.vertices[k].y * scale,
                                        tri.vertices[k].z * scale);
                float3 v_spin = rotateAxis(v, axis, spin_angle);
                file << (center.x + v_spin.x) << " "
                     << (center.y + v_spin.y) << " "
                     << (center.z + v_spin.z) << " ";
            }
            file << "\n";
        }
    };

    write_mesh(near_mesh, near_spin_angle, near_center);
    write_mesh(far_mesh,  far_spin_angle,  far_center);

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

// Main
// ============================================================================
int main(int argc, char** argv) {
    // ---- Simulation parameters ----
    float fluid_fraction = 0.7f;
    float tau = 0.8f;
    float gravity_y = -1e-5f;
    int steps = -1;  // will be set interactively if not given via CLI
    int output_every = 500;
    std::string out_dir = "output/planetary_mixing";

    // Physical -> lattice scale: 1 LU = 1mm
    float phys_to_lattice = 1000.0f;

    // Margin around paddle (percentage)
    float cylinder_margin = 1.5f;  // cylinder_radius = max_paddle_extent * 1.5
    float domain_margin = 8.0f;    // extra LU around cylinder for SOLID cells

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--steps" && i + 1 < argc) steps = std::stoi(argv[++i]);
        if (arg == "--output" && i + 1 < argc) out_dir = argv[++i];
        if (arg == "--tau" && i + 1 < argc) tau = std::stof(argv[++i]);
        if (arg == "--scale" && i + 1 < argc) phys_to_lattice = std::stof(argv[++i]);
    }

    // Interactive step count input (if not set via --steps)
    if (steps < 0) {
        const int MIN_STEPS = 10000;
        const int MAX_STEPS = 100000;
        while (true) {
            std::cout << "========================================" << std::endl;
            std::cout << "Enter number of simulation steps" << std::endl;
            std::cout << "  Range: " << MIN_STEPS << " - " << MAX_STEPS << std::endl;
            std::cout << "  (1 revolution at 30 RPM = ~20000 steps)" << std::endl;
            std::cout << ">> ";

            std::string input;
            std::getline(std::cin, input);

            try {
                int val = std::stoi(input);
                if (val < MIN_STEPS) {
                    std::cout << "ERROR: " << val << " is below minimum (" << MIN_STEPS << "). Try again." << std::endl;
                } else if (val > MAX_STEPS) {
                    std::cout << "ERROR: " << val << " exceeds maximum (" << MAX_STEPS << "). Try again." << std::endl;
                } else {
                    steps = val;
                    std::cout << "OK: " << steps << " steps selected ("
                              << std::fixed << std::setprecision(2)
                              << (steps * 1e-4) << " seconds, "
                              << (steps * 1e-4 / 2.0) << " revolutions at 30 RPM)"
                              << std::endl;
                    break;
                }
            } catch (...) {
                std::cout << "ERROR: Invalid input '" << input << "'. Please enter a number." << std::endl;
            }
        }
        std::cout << "========================================" << std::endl;
    }

    fs::create_directories(out_dir);

    // ---- 1. Load paddle FIRST to determine dimensions ----
    PlanetaryPaddleModule::Config paddle_cfg;
    paddle_cfg.near_offset = 0.01475f;
    paddle_cfg.far_offset = 0.0295f;
    paddle_cfg.revolution_rpm = 30.0f;
    paddle_cfg.revolution_phase_deg = 0.0f;
    paddle_cfg.near_orbit_phase_deg = 180.0f;
    paddle_cfg.near_spin_rpm = -30.0f;
    paddle_cfg.near_spin_phase_deg = 0.0f;
    paddle_cfg.far_spin_rpm = 60.0f;
    paddle_cfg.far_spin_phase_deg = -30.0f;  // from planetary_spinning_test

    PlanetaryPaddleModule paddle(paddle_cfg);
    bool paddle_loaded = paddle.load();

    // ---- 2. Compute cylinder size from paddle mesh ----
    float far_offset_lu  = paddle_cfg.far_offset * phys_to_lattice;
    float near_offset_lu = paddle_cfg.near_offset * phys_to_lattice;

    // Measure max vertex extent from mesh origin (XZ plane)
    float max_blade_extent = 0.0f;
    if (paddle_loaded) {
        auto measureExtent = [&](const STLMesh& mesh) {
            for (const auto& tri : mesh.triangles) {
                for (int k = 0; k < 3; ++k) {
                    float ex = std::abs(tri.vertices[k].x) * phys_to_lattice;
                    float ez = std::abs(tri.vertices[k].z) * phys_to_lattice;
                    float ey = std::abs(tri.vertices[k].y) * phys_to_lattice;
                    float r = sqrtf(ex * ex + ez * ez);
                    if (r > max_blade_extent) max_blade_extent = r;
                    // Also track Y extent for NY sizing
                }
            }
        };
        measureExtent(paddle.nearMesh());
        measureExtent(paddle.farMesh());
        std::cout << "[Paddle] Loaded OK. near_tri=" << paddle.nearMesh().triangles.size()
                  << " far_tri=" << paddle.farMesh().triangles.size() << std::endl;
        std::cout << "[Paddle] Max blade extent (from center): " << max_blade_extent << " LU" << std::endl;
        std::cout << "[Paddle] Far orbit offset: " << far_offset_lu << " LU" << std::endl;
    } else {
        std::cerr << "Paddle STL load FAILED! Using defaults." << std::endl;
        max_blade_extent = 20.0f;
    }

    // Max paddle reach = orbit offset + blade extent
    float max_paddle_reach = far_offset_lu + max_blade_extent;
    float cylinder_radius = max_paddle_reach * cylinder_margin;

    // Domain size: cylinder must fit inside with margin for SOLID cells
    int nx = (int)(2.0f * cylinder_radius + 2.0f * domain_margin);
    int nz = nx;  // symmetric in XZ
    int ny = (int)(cylinder_radius * 2.0f);  // taller cylinder
    // Round up to multiples of 4 for GPU alignment
    nx = ((nx + 3) / 4) * 4;
    ny = ((ny + 3) / 4) * 4;
    nz = ((nz + 3) / 4) * 4;

    float nu = (tau - 0.5f) / 3.0f;
    int fluid_height = (int)(ny * fluid_fraction);

    // Paddle rotation center = EXACT cylinder center, Y = mid fluid height
    float3 paddle_center = make_float3(nx / 2.0f,
                                        fluid_height / 2.0f,
                                        nz / 2.0f);

    std::cout << "=== Planetary Paddle Mixing ===" << std::endl;
    std::cout << "Max paddle reach: " << max_paddle_reach << " LU" << std::endl;
    std::cout << "Cylinder R: " << cylinder_radius << " LU (margin " << (int)(cylinder_margin*100) << "%)" << std::endl;
    std::cout << "Domain: " << nx << "x" << ny << "x" << nz << std::endl;
    std::cout << "Fluid height: " << fluid_height << " / " << ny << std::endl;
    std::cout << "Paddle center: (" << paddle_center.x << ", "
              << paddle_center.y << ", " << paddle_center.z << ")" << std::endl;
    std::cout << "Scale: " << phys_to_lattice << " (1 LU = 1mm)" << std::endl;
    std::cout << "tau=" << tau << " nu=" << nu << std::endl;

    // ---- 3. Build cylinder domain ----
    CylinderDomainConfig cyl_cfg;
    cyl_cfg.nx = nx; cyl_cfg.ny = ny; cyl_cfg.nz = nz;
    cyl_cfg.cx = nx / 2.0f;
    cyl_cfg.cz = nz / 2.0f;
    cyl_cfg.radius = cylinder_radius;
    cyl_cfg.fluid_fraction = fluid_fraction;
    CylinderDomainResult cyl = buildCylinderDomain(cyl_cfg);

    // Write cylinder wall VTP (static, once)
    writeCylinderVTP(out_dir + "/cylinder_wall.vtp",
                     cyl_cfg.cx, cyl_cfg.cz, cylinder_radius, ny);

    // ---- 2. Init LBM ----
    lbm::LBMConfig lbm_cfg;
    lbm_cfg.nx = nx; lbm_cfg.ny = ny; lbm_cfg.nz = nz;
    lbm_cfg.tau = tau;
    lbm_cfg.enableFreeSurface = true;
    lbm_cfg.collisionModel = lbm::CollisionModel::SRT;
    lbm_cfg.gravity = make_float3(0.0f, gravity_y, 0.0f);
    lbm_cfg.bcXMin = lbm::BC_BOUNCE_BACK;
    lbm_cfg.bcXMax = lbm::BC_BOUNCE_BACK;
    lbm_cfg.bcYMin = lbm::BC_BOUNCE_BACK;
    lbm_cfg.bcYMax = lbm::BC_OPEN;
    lbm_cfg.bcZMin = lbm::BC_BOUNCE_BACK;
    lbm_cfg.bcZMax = lbm::BC_BOUNCE_BACK;

    lbm::LBMCore lbm(lbm_cfg);
    lbm.initialize();

    int N = nx * ny * nz;

    // ---- 5. Free Surface Module ----
    lbm::FreeSurfaceModule fsModule;
    fsModule.configure(lbm_cfg);

    FieldStore fsFields;
    fsFields.create(FieldDesc{"fluid.density", (size_t)N, sizeof(float), lbm.densityDevicePtr()});
    fsFields.create(FieldDesc{"fluid.velocity", (size_t)N * 3, sizeof(float), lbm.velocityDevicePtr()});
    fsFields.create(FieldDesc{"fluid.flags", (size_t)N, sizeof(uint8_t), lbm.flagsDevicePtr()});
    fsFields.create(FieldDesc{"fluid.phi", (size_t)N, sizeof(float), lbm.phiDevicePtr()});
    fsFields.create(FieldDesc{"fluid.mass", (size_t)N, sizeof(float), lbm.massDevicePtr()});
    fsModule.allocate(fsFields);
    fsModule.initialize(fsFields);

    // Set cylinder domain: outside = SOLID, inside bottom = LIQUID, top = GAS
    // First: everything GAS
    fsModule.setRegion(0, nx-1, 0, ny-1, 0, nz-1, lbm::CellType::GAS, 0.0f, 1.0f);

    // Inside cylinder: LIQUID up to fluid_height, GAS above
    // We do this by setting LIQUID for the full box then letting SOLID override
    fsModule.setRegion(0, nx-1, 0, fluid_height-1, 0, nz-1, lbm::CellType::LIQUID, 1.0f, 1.0f);

    // Now apply cylinder mask: set cells outside cylinder radius to SOLID
    {
        std::vector<uint8_t> h_flags(N);
        CHECK_CUDA(cudaMemcpy(h_flags.data(), lbm.flagsDevicePtr(), N * sizeof(uint8_t), cudaMemcpyDeviceToHost));

        float R2 = cylinder_radius * cylinder_radius;
        float ccx = nx / 2.0f;
        float ccz = nz / 2.0f;

        for (int z = 0; z < nz; ++z)
            for (int y = 0; y < ny; ++y)
                for (int x = 0; x < nx; ++x) {
                    float dx = (float)x - ccx;
                    float dz = (float)z - ccz;
                    if (dx*dx + dz*dz > R2) {
                        int idx = z * nx * ny + y * nx + x;
                        h_flags[idx] = lbm::cuda::CellFlag::SOLID;
                    }
                }

        CHECK_CUDA(cudaMemcpy(lbm.flagsDevicePtr(), h_flags.data(), N * sizeof(uint8_t), cudaMemcpyHostToDevice));
    }

    fsModule.fixInterfaceLayer();
    lbm.refreshDistributions();
    std::cout << "[FreeSurface] Initialized: LIQUID y<" << fluid_height
              << ", GAS above, SOLID outside R=" << cylinder_radius << std::endl;

    // ---- 6. IBM Setup ----
    // Sample markers from BOTH paddle STL meshes (in physical coords, centered at origin)
    float ibm_spacing = 1.0f;  // marker spacing in lattice units
    float phys_spacing = ibm_spacing / phys_to_lattice;  // spacing in physical units
    float3 zero_com = make_float3(0, 0, 0);

    std::vector<IBMMarker> near_markers_ref, far_markers_ref;
    size_t nNearMarkers = 0, nFarMarkers = 0, nTotalMarkers = 0;

    if (paddle_loaded) {
        near_markers_ref = STLGeometryLoader::sampleSurfaceMarkers(
            paddle.nearMesh(), zero_com, phys_spacing);
        far_markers_ref = STLGeometryLoader::sampleSurfaceMarkers(
            paddle.farMesh(), zero_com, phys_spacing);
        nNearMarkers = near_markers_ref.size();
        nFarMarkers  = far_markers_ref.size();
        nTotalMarkers = nNearMarkers + nFarMarkers;
        std::cout << "[IBM] Markers: near=" << nNearMarkers
                  << " far=" << nFarMarkers
                  << " total=" << nTotalMarkers << std::endl;
    }

    ibm::IBMParams ibm_p;
    ibm_p.nx = nx; ibm_p.ny = ny; ibm_p.nz = nz;
    ibm_p.dx = 1.0f;
    ibm_p.nMarkers = (int)nTotalMarkers;
    ibm_p.mdf_iterations = 3;
    ibm_p.mdf_beta = -0.5f;
    ibm_p.force_method = ibm::IBMForceMethod::DIRECT_FORCING;
    ibm::IBMCore ibm(ibm_p);

    // Marker arrays (world-space positions + velocities + areas)
    std::vector<float3> h_pos(nTotalMarkers);
    std::vector<float3> h_vel(nTotalMarkers);
    std::vector<float>  h_area(nTotalMarkers);

    // Store relative (physical) positions for re-use each step
    std::vector<float3> near_rel(nNearMarkers), far_rel(nFarMarkers);
    for (size_t i = 0; i < nNearMarkers; ++i) {
        near_rel[i] = near_markers_ref[i].pos;
        h_area[i] = near_markers_ref[i].area * phys_to_lattice * phys_to_lattice;
    }
    for (size_t i = 0; i < nFarMarkers; ++i) {
        far_rel[i] = far_markers_ref[i].pos;
        h_area[nNearMarkers + i] = far_markers_ref[i].area * phys_to_lattice * phys_to_lattice;
    }

    // Write initial paddle frame
    if (paddle_loaded) {
        writePaddleFrameVTP(out_dir + "/paddle_000000.vtp",
                            paddle, paddle_cfg, 0.0, paddle_center, phys_to_lattice);
    }

    // ---- 7. GPU force buffer ----
    float3* d_force;
    CHECK_CUDA(cudaMalloc(&d_force, N * sizeof(float3)));

    // ---- 8. VTK output ----
    FieldStore fields;
    StepContext ctx;
    ctx.fields = &fields;
    ctx.backend = &lbm.backend();

    VTKService::Config vtk_cfg;
    vtk_cfg.output_dir = out_dir;
    vtk_cfg.nx = nx; vtk_cfg.ny = ny; vtk_cfg.nz = nz;
    vtk_cfg.interval = output_every;
    vtk_cfg.binary = true;
    vtk_cfg.fields = {"Density", "Speed", "Velocity", "Fill"};

    VTKService vtk_svc(vtk_cfg);
    vtk_svc.initialize(ctx);

    // ---- 9. Simulation loop ----
    std::cout << "Starting simulation: " << steps << " steps" << std::endl;
    double dt_phys = 1e-4;

    // Rotation helpers (mirrors PlanetaryPaddleModule logic)
    float kPi = (float)M_PI;
    auto rpmToRad = [kPi](float rpm) { return rpm * (2.0f * kPi / 60.0f); };
    auto degToRad = [kPi](float deg) { return deg * (kPi / 180.0f); };
    float rev_omega       = -rpmToRad(paddle_cfg.revolution_rpm);
    float near_spin_omega = -rpmToRad(paddle_cfg.near_spin_rpm);
    float far_spin_omega  = -rpmToRad(paddle_cfg.far_spin_rpm);

    auto rotateAxis = [](const float3& v, const float3& ax, float angle) -> float3 {
        float c = cosf(angle), s = sinf(angle);
        float d = ax.x * v.x + ax.y * v.y + ax.z * v.z;
        return make_float3(
            v.x * c + (ax.y * v.z - ax.z * v.y) * s + ax.x * d * (1 - c),
            v.y * c + (ax.z * v.x - ax.x * v.z) * s + ax.y * d * (1 - c),
            v.z * c + (ax.x * v.y - ax.y * v.x) * s + ax.z * d * (1 - c)
        );
    };
    float3 axis = make_float3(0, 1, 0);

    for (int t = 0; t <= steps; ++t) {
        ctx.step = t;
        ctx.time = t;
        double time_sec = t * dt_phys;

        // (a) Update IBM markers: rotate paddle to current angle
        if (paddle_loaded && nTotalMarkers > 0) {
            float rev_angle = rev_omega * (float)time_sec + degToRad(paddle_cfg.revolution_phase_deg);
            float near_orbit_angle = rev_angle + degToRad(paddle_cfg.near_orbit_phase_deg);
            float far_rev_angle = near_orbit_angle + kPi;

            float near_spin_angle = near_spin_omega * (float)time_sec + degToRad(paddle_cfg.near_spin_phase_deg);
            float far_spin_angle  = far_spin_omega * (float)time_sec + degToRad(paddle_cfg.far_spin_phase_deg);
            near_spin_angle += near_orbit_angle;
            far_spin_angle  += far_rev_angle;

            // Near paddle center
            float near_offset_lu = paddle_cfg.near_offset * phys_to_lattice;
            float3 near_ofs = rotateAxis(make_float3(near_offset_lu, 0, 0), axis, near_orbit_angle);
            float3 near_center = make_float3(paddle_center.x + near_ofs.x,
                                              paddle_center.y + near_ofs.y,
                                              paddle_center.z + near_ofs.z);

            // Far paddle center
            float far_offset_lu = paddle_cfg.far_offset * phys_to_lattice;
            float3 far_ofs = rotateAxis(make_float3(far_offset_lu, 0, 0), axis, far_rev_angle);
            float3 far_center = make_float3(paddle_center.x + far_ofs.x,
                                             paddle_center.y + far_ofs.y,
                                             paddle_center.z + far_ofs.z);

            // Marker velocity: v = omega_revolution x r_from_center + omega_self_spin x r_from_paddle_center
            // CRITICAL: multiply by dt_phys to convert from LU/physical_second to LU/LBM_step
            float dt_f = (float)dt_phys;

            // Update near markers
            for (size_t i = 0; i < nNearMarkers; ++i) {
                // Position: scale + spin rotate + translate
                float3 v_phys = near_rel[i];
                float3 v_lu = make_float3(v_phys.x * phys_to_lattice,
                                           v_phys.y * phys_to_lattice,
                                           v_phys.z * phys_to_lattice);
                float3 v_rot = rotateAxis(v_lu, axis, near_spin_angle);
                h_pos[i] = make_float3(near_center.x + v_rot.x,
                                        near_center.y + v_rot.y,
                                        near_center.z + v_rot.z);

                // Velocity = revolution contribution + self-spin contribution
                // Revolution: v_rev = omega_rev x (pos - revolution_center)
                float3 r_rev = make_float3(h_pos[i].x - paddle_center.x, 0.0f,
                                            h_pos[i].z - paddle_center.z);
                float3 v_revolution = make_float3(rev_omega * (-r_rev.z), 0.0f,
                                                   rev_omega * r_rev.x);
                // Self-spin: v_spin = omega_self x (pos - paddle_center_near)
                float3 r_spin = make_float3(h_pos[i].x - near_center.x, 0.0f,
                                             h_pos[i].z - near_center.z);
                float3 v_spin = make_float3(near_spin_omega * (-r_spin.z), 0.0f,
                                             near_spin_omega * r_spin.x);
                h_vel[i] = make_float3((v_revolution.x + v_spin.x) * dt_f,
                                        0.0f,
                                        (v_revolution.z + v_spin.z) * dt_f);
            }

            // Update far markers
            for (size_t i = 0; i < nFarMarkers; ++i) {
                float3 v_phys = far_rel[i];
                float3 v_lu = make_float3(v_phys.x * phys_to_lattice,
                                           v_phys.y * phys_to_lattice,
                                           v_phys.z * phys_to_lattice);
                float3 v_rot = rotateAxis(v_lu, axis, far_spin_angle);
                size_t idx = nNearMarkers + i;
                h_pos[idx] = make_float3(far_center.x + v_rot.x,
                                          far_center.y + v_rot.y,
                                          far_center.z + v_rot.z);

                float3 r_rev = make_float3(h_pos[idx].x - paddle_center.x, 0.0f,
                                            h_pos[idx].z - paddle_center.z);
                float3 v_revolution = make_float3(rev_omega * (-r_rev.z), 0.0f,
                                                   rev_omega * r_rev.x);
                float3 r_spin = make_float3(h_pos[idx].x - far_center.x, 0.0f,
                                             h_pos[idx].z - far_center.z);
                float3 v_spin = make_float3(far_spin_omega * (-r_spin.z), 0.0f,
                                             far_spin_omega * r_spin.x);
                h_vel[idx] = make_float3((v_revolution.x + v_spin.x) * dt_f,
                                          0.0f,
                                          (v_revolution.z + v_spin.z) * dt_f);
            }

            ibm.updateMarkers(h_pos.data(), h_vel.data(), h_area.data());
        }

        // (b) IBM force
        CHECK_CUDA(cudaMemset(d_force, 0, N * sizeof(float3)));
        if (paddle_loaded && nTotalMarkers > 0) {
            ibm.computeForces(lbm.velocityAoSPtr(), nullptr, d_force, 1.0f);
        }
        lbm.setExternalForceFromDeviceAoS(d_force);

        // (c) Free surface + LBM step
        fsModule.preStream(ctx);
        lbm.streamCollide();
        fsModule.postStream(ctx);
        lbm.updateMacroscopic();

        // Diagnostics
        if (t % 200 == 0) {
            std::vector<float> h_rho(N);
            CHECK_CUDA(cudaMemcpy(h_rho.data(), lbm.getDensityField(),
                                  N * sizeof(float), cudaMemcpyDeviceToHost));
            float rho_min = 1e9f, rho_max = -1e9f;
            for (float r : h_rho) {
                if (r > 0.01f) {
                    if (r < rho_min) rho_min = r;
                    if (r > rho_max) rho_max = r;
                }
            }

            // Max velocity magnitude
            std::vector<float3> h_u_diag(N);
            CHECK_CUDA(cudaMemcpy(h_u_diag.data(), lbm.velocityAoSPtr(),
                                  N * sizeof(float3), cudaMemcpyDeviceToHost));
            float u_max = 0.0f;
            for (int i = 0; i < N; ++i) {
                float u2 = h_u_diag[i].x*h_u_diag[i].x + h_u_diag[i].y*h_u_diag[i].y + h_u_diag[i].z*h_u_diag[i].z;
                if (u2 > u_max) u_max = u2;
            }
            u_max = sqrtf(u_max);

            // Max marker velocity
            float v_marker_max = 0.0f;
            for (size_t i = 0; i < nTotalMarkers; ++i) {
                float v2 = h_vel[i].x*h_vel[i].x + h_vel[i].y*h_vel[i].y + h_vel[i].z*h_vel[i].z;
                if (v2 > v_marker_max) v_marker_max = v2;
            }
            v_marker_max = sqrtf(v_marker_max);

            std::cout << "Step " << t << " t=" << std::fixed << std::setprecision(4)
                      << time_sec << "s rho=[" << rho_min << "," << rho_max
                      << "] u_max=" << std::setprecision(6) << u_max
                      << " v_marker=" << v_marker_max
                      << std::endl;
        }

        // Output
        if (t > 0 && t % output_every == 0) {
            auto h_rho = fields.create({"Density", (size_t)N, sizeof(float)});
            CHECK_CUDA(cudaMemcpy(h_rho.data(), lbm.getDensityField(),
                                  N * sizeof(float), cudaMemcpyDeviceToHost));

            auto h_u = fields.create({"Velocity", (size_t)N, sizeof(float) * 3});
            CHECK_CUDA(cudaMemcpy(h_u.data(), lbm.velocityAoSPtr(), N * sizeof(float3), cudaMemcpyDeviceToHost));

            std::vector<float> speed(N);
            float3* cpu_u = (float3*)h_u.data();
            for (int i = 0; i < N; ++i)
                speed[i] = sqrtf(cpu_u[i].x*cpu_u[i].x + cpu_u[i].y*cpu_u[i].y + cpu_u[i].z*cpu_u[i].z);
            auto h_speed = fields.create({"Speed", (size_t)N, sizeof(float)});
            memcpy(h_speed.data(), speed.data(), N * sizeof(float));

            // Fill (phi) for free surface visualization
            auto h_fill = fields.create({"Fill", (size_t)N, sizeof(float)});
            CHECK_CUDA(cudaMemcpy(h_fill.data(), lbm.phiDevicePtr(), N * sizeof(float), cudaMemcpyDeviceToHost));

            vtk_svc.onStepEnd(ctx);

            // Paddle VTP
            if (paddle_loaded) {
                std::ostringstream paddle_path;
                paddle_path << out_dir << "/paddle_"
                            << std::setfill('0') << std::setw(6) << t << ".vtp";
                writePaddleFrameVTP(paddle_path.str(), paddle, paddle_cfg, time_sec,
                                    paddle_center, phys_to_lattice);
            }

            std::cout << "  Output: step " << t << std::endl;
        }
    }

    vtk_svc.finalize(ctx);
    fsModule.finalize();
    CHECK_CUDA(cudaFree(d_force));
    std::cout << "=== Simulation complete ===" << std::endl;
    return 0;
}

