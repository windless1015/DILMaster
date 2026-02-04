#include "MeshVTKService.hpp"
#include "../physics/ibm/IBMMotion.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace fs = std::filesystem;

MeshVTKService::MeshVTKService(const Config& config, 
                               std::shared_ptr<IBMSolver> ibm,
                               std::shared_ptr<PropellerPureLiquidScenario> scenario)
    : config_(config), ibm_(ibm), scenario_(scenario) {}

void MeshVTKService::initialize(const StepContext& ctx) {
    fs::create_directories(config_.output_dir);
    initialized_ = true;
}

void MeshVTKService::onStepBegin(StepContext& ctx) {}

void MeshVTKService::onStepEnd(StepContext& ctx) {
    if (!initialized_) return;
    if (ctx.step % config_.interval != 0) return;

    if (!ibm_) {
        std::cerr << "MeshVTKService: ibm_ is null" << std::endl;
        return;
    }
    if (!scenario_) {
        std::cerr << "MeshVTKService: scenario_ is null" << std::endl;
        return;
    }
    
    auto motion = ibm_->getMotion();
    if (!motion) {
        std::cerr << "MeshVTKService: motion is null at step " << ctx.step << std::endl;
        return;
    }
    
    const auto& state = motion->getState();
    writeVTPFile(ctx.step, state);
}

void MeshVTKService::finalize(const StepContext& ctx) {
    initialized_ = false;
}

void MeshVTKService::writeVTPFile(int step, const RigidBodyState& state) {
    std::ostringstream filename;
    filename << config_.output_dir << "/" << config_.filename_prefix 
             << std::setfill('0') << std::setw(6) << step << ".vtp";

    std::ofstream file(filename.str());
    if (!file.is_open()) return;

    const auto& mesh = scenario_->getMesh(); 
    size_t nTriangles = mesh.triangles.size();
    
    // Safety check: Don't write if no triangles
    if (nTriangles == 0) {
        std::cerr << "MeshVTKService: No triangles in mesh, skipping VTP output." << std::endl;
        return;
    }
    
    size_t nVertices = nTriangles * 3;

    // XML PolyData Header
    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    file << "  <PolyData>\n";
    file << "    <Piece NumberOfPoints=\"" << nVertices << "\" NumberOfVerts=\"0\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"" << nTriangles << "\">\n";

    // 1. Points (Vertices) - Transformed
    file << "      <Points>\n";
    file << "        <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    
    // Transform parameters
    float3 center = state.position; // Rotation center/COM
    Quaternion q = state.orientation;
    
    // We need to rotate mesh vertices. 
    // MESH stored in STLMesh is usually in local coordinates relative to something, 
    // OR absolute coordinates if loaded directly.
    // However, usually STL is loaded "as is". 
    // Ideally, the mesh vertices should be defined relative to the Center of Rotation for simple q rotation.
    // OR: Mesh = Original. Pos = Center + Rotate(Original - Center0).
    // Let's assume STL is Centered at (0,0,0) OR we computed COM relative.
    
    // In PropellerScenario: 
    // mesh_.centerAtOrigin(); was called?
    // Let's check logic:
    // STLReader reads file.
    // We want mesh to rotate.
    // If we assume the stored 'mesh' in scenario is the "Reference/Initial Layout" (Body Frame).
    // Then WorldPos = BodyPos + Rotate(Vertex_Body) (if BodyPos is translation)
    // The IBMSolver uses: Pos = Center + Rotate(Rel_Pos)
    // So we need Vertex_Rel = Vertex_Initial - Initial_Center.
    
    // Let's assume Scenario provides the INITIAL mesh (World Frame at t=0).
    // And Setup computed Center.
    // Rel = Vertex - Center(t=0).
    
    // BUT! PropellerMixingScenario loads mesh, centers it (?) or not.
    // Let's look at Scenario again. Wait, I will assume consistent logic with IBMSolver.
    // Scenario.cpp:66 mesh_.centerAtOrigin(); -> This puts it at 0,0,0.
    // Then it moves it? No.
    // markers_ = sample(mesh_ at 0,0,0).
    // Then applyInitial moves markers to config_.center.
    // So mesh_ is at (0,0,0).
    // So Vertex_Body = Vertex_Mesh.
    // WorldPos = State.position + Rotate(Vertex_Body).
    
    // Wait, state.position is Rotation Center.
    // If mesh is centered at origin, then Vertex IS relative vector.
    // Correct.
    
    // Helper to rotate vector
    auto rotate = [&](float3 v) {
        // v' = q * v * q_conj
        // Implementation from IBMMotion logic duplicate here or expose helper?
        // Let's just implement standard Quat rotate here to avoid linking issues or dependency hell.
        float3 u = make_float3(q.x, q.y, q.z);
        float s = q.w;
        
        // cross(u, v)
        float3 uv;
        uv.x = u.y * v.z - u.z * v.y;
        uv.y = u.z * v.x - u.x * v.z;
        uv.z = u.x * v.y - u.y * v.x;
        
        // cross(u, uv)
        float3 uuv;
        uuv.x = u.y * uv.z - u.z * uv.y;
        uuv.y = u.z * uv.x - u.x * uv.z;
        uuv.z = u.x * uv.y - u.y * uv.x;
        
        // v + 2s*uv + 2*uuv
        float3 res;
        res.x = v.x + 2.0f * s * uv.x + 2.0f * uuv.x;
        res.y = v.y + 2.0f * s * uv.y + 2.0f * uuv.y;
        res.z = v.z + 2.0f * s * uv.z + 2.0f * uuv.z;
        return res;
    };
    
    for (const auto& tri : mesh.triangles) {
        for (int k=0; k<3; ++k) {
            float3 v_body = tri.vertices[k];
            float3 v_world_rel = rotate(v_body);
            float3 v_final;
            v_final.x = center.x + v_world_rel.x;
            v_final.y = center.y + v_world_rel.y;
            v_final.z = center.z + v_world_rel.z;
            
            file << v_final.x << " " << v_final.y << " " << v_final.z << " ";
        }
        file << "\n";
    }
    file << "        </DataArray>\n";
    file << "      </Points>\n";

    // 2. PointData - Velocity at each vertex
    // For rigid rotation: v = omega × r_rotated
    file << "      <PointData Vectors=\"Velocity\">\n";
    file << "        <DataArray type=\"Float32\" Name=\"Velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    
    float3 omega = state.angular_velocity;

    
    for (const auto& tri : mesh.triangles) {
        for (int k=0; k<3; ++k) {
            float3 v_body = tri.vertices[k];
            float3 r_rotated = rotate(v_body);  // Same rotation as position
            
            // v = omega × r_rotated (cross product)
            float3 vel;
            vel.x = omega.y * r_rotated.z - omega.z * r_rotated.y;
            vel.y = omega.z * r_rotated.x - omega.x * r_rotated.z;
            vel.z = omega.x * r_rotated.y - omega.y * r_rotated.x;
            
            file << vel.x << " " << vel.y << " " << vel.z << " ";
        }
        file << "\n";
    }
    file << "        </DataArray>\n";
    file << "      </PointData>\n";

    // 3. Polys (Topology)
    file << "      <Polys>\n";
    file << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (size_t i = 0; i < nTriangles; ++i) {
        file << (i*3) << " " << (i*3+1) << " " << (i*3+2) << "\n";
    }
    file << "        </DataArray>\n";
    
    file << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (size_t i = 0; i < nTriangles; ++i) {
        file << (i*3+3) << "\n";
    }
    file << "        </DataArray>\n";
    file << "      </Polys>\n";

    file << "    </Piece>\n";
    file << "  </PolyData>\n";
    file << "</VTKFile>\n";
}
