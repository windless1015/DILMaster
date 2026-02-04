#include "MarkerVTKService.hpp"
#include "../core/FieldStore.hpp"
#include "../core/StepContext.hpp"
#include "../geometry/VectorTypes.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <cstdint>

namespace fs = std::filesystem;

MarkerVTKService::MarkerVTKService(const Config &config) : config_(config) {}

MarkerVTKService::~MarkerVTKService() {}

void MarkerVTKService::initialize(const StepContext &ctx) {
  fs::create_directories(config_.output_dir);
  initialized_ = true;
  (void)ctx;
}

void MarkerVTKService::onStepEnd(StepContext &ctx) {
  if (!initialized_) return;
  if (static_cast<int>(ctx.step) % config_.interval != 0) return;
  if (!ctx.fields || !ctx.fields->exists(config_.marker_field)) return;

  writeVTPFile(ctx);
}

void MarkerVTKService::finalize(const StepContext &ctx) {
  initialized_ = false;
  (void)ctx;
}

void MarkerVTKService::writeVTPFile(const StepContext &ctx) {
  // Generate filename: output_dir/markers_XXXXXX.vtp
  std::ostringstream filename;
  filename << config_.output_dir << "/markers_" << std::setfill('0')
           << std::setw(6) << ctx.step << ".vtp";

  // Use binary mode for file stream to safely write raw bytes if needed
  std::ofstream file(filename.str(), std::ios::out | std::ios::binary);
  if (!file.is_open()) return;

  auto handle = ctx.fields->get(config_.marker_field);
  size_t nPoints = handle.count();
  const float3* points = handle.as<float3>();
  
  // Calculate offsets for AppendedData
  uint64_t offset = 0;
  auto get_offset_and_advance = [&](size_t size) {
      uint64_t current = offset;
      offset += sizeof(uint32_t) + size;
      return current;
  };

  // Sizes for binary blocks
  size_t points_size = nPoints * 3 * sizeof(float);
  size_t vel_size = nPoints * 3 * sizeof(float);
  size_t force_size = nPoints * 3 * sizeof(float);
  size_t conn_size = nPoints * sizeof(int32_t);
  size_t offs_size = nPoints * sizeof(int32_t);

  uint64_t off_points = get_offset_and_advance(points_size);
  uint64_t off_vel = 0;
  if(ctx.fields->exists("ibm.velocity")) off_vel = get_offset_and_advance(vel_size);
  
  uint64_t off_force = 0;
  if(ctx.fields->exists("ibm.force")) off_force = get_offset_and_advance(force_size);

  uint64_t off_conn = get_offset_and_advance(conn_size);
  uint64_t off_offs = get_offset_and_advance(offs_size);

  // XML Header
  file << "<?xml version=\"1.0\"?>\n";
  file << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt32\">\n";
  file << "  <PolyData>\n";
  file << "    <Piece NumberOfPoints=\"" << nPoints << "\" NumberOfVerts=\"" << nPoints 
       << "\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n";
  
  // -- Points --
  file << "      <Points>\n";
  file << "        <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\""
       << (config_.binary ? "appended" : "ascii") << "\"";
  if(config_.binary) file << " offset=\"" << off_points << "\"";
  file << ">\n";
  
  if(!config_.binary) {
      for (size_t i = 0; i < nPoints; ++i) {
          file << points[i].x << " " << points[i].y << " " << points[i].z;
          if ((i+1)%3 == 0) file << "\n"; else file << " ";
      }
  }
  file << "        </DataArray>\n";
  file << "      </Points>\n";
  
  // -- Point Data --
  file << "      <PointData Vectors=\"Velocity\">\n";
  
  // Velocity
  if (ctx.fields->exists("ibm.velocity")) {
       file << "        <DataArray type=\"Float32\" Name=\"Velocity\" NumberOfComponents=\"3\" format=\""
            << (config_.binary ? "appended" : "ascii") << "\"";
       if(config_.binary) file << " offset=\"" << off_vel << "\"";
       file << ">\n";
       
       if(!config_.binary) {
           auto h = ctx.fields->get("ibm.velocity");
           const float3* vel = h.as<float3>();
           for (size_t i = 0; i < nPoints; ++i) {
               file << vel[i].x << " " << vel[i].y << " " << vel[i].z;
               if ((i+1)%10 == 0) file << "\n"; else file << " ";
           }
       }
       file << "        </DataArray>\n";
  }

  // Force
  if (ctx.fields->exists("ibm.force")) {
       file << "        <DataArray type=\"Float32\" Name=\"Force\" NumberOfComponents=\"3\" format=\""
            << (config_.binary ? "appended" : "ascii") << "\"";
       if(config_.binary) file << " offset=\"" << off_force << "\"";
       file << ">\n";
       
       if(!config_.binary) {
           auto h = ctx.fields->get("ibm.force");
           const float3* force = h.as<float3>();
           for (size_t i = 0; i < nPoints; ++i) {
               file << force[i].x << " " << force[i].y << " " << force[i].z;
               if ((i+1)%10 == 0) file << "\n"; else file << " ";
           }
       }
       file << "        </DataArray>\n";
  }

  file << "      </PointData>\n";
  
  // -- Verts (Topology) --
  file << "      <Verts>\n";
  
  // Connectivity
  file << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\""
       << (config_.binary ? "appended" : "ascii") << "\"";
  if(config_.binary) file << " offset=\"" << off_conn << "\"";
  file << ">\n";
  if(!config_.binary) {
      for (size_t i = 0; i < nPoints; ++i) {
          file << i;
          if ((i+1)%20 == 0) file << "\n"; else file << " ";
      }
  }
  file << "        </DataArray>\n";
  
  // Offsets
  file << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\""
       << (config_.binary ? "appended" : "ascii") << "\"";
  if(config_.binary) file << " offset=\"" << off_offs << "\"";
  file << ">\n";
  if(!config_.binary) {
      for (size_t i = 0; i < nPoints; ++i) {
          file << (i+1);
          if ((i+1)%20 == 0) file << "\n"; else file << " ";
      }
  }
  file << "        </DataArray>\n";
  
  file << "      </Verts>\n";
  file << "    </Piece>\n";
  file << "  </PolyData>\n";
  
  // -- Appended Data Section --
  if(config_.binary) {
      file << "  <AppendedData encoding=\"raw\">\n";
      file << "    _"; // Start of data
      
      // Helper to write data block: [size: uint32][data: raw bytes]
      auto write_block = [&](const void* data, size_t size) {
          uint32_t s = (uint32_t)size;
          file.write((const char*)&s, sizeof(uint32_t));
          file.write((const char*)data, size);
      };
      
      // Helper to repack float3 to vector<float>
      auto repack_float3 = [&](const float3* src, size_t count) {
          std::vector<float> data(count * 3);
          for(size_t i=0; i<count; ++i) {
              data[3*i+0] = src[i].x;
              data[3*i+1] = src[i].y;
              data[3*i+2] = src[i].z;
          }
          return data;
      };
      
      // 1. Points
      {
          auto data = repack_float3(points, nPoints);
          write_block(data.data(), points_size);
      }
      
      // 2. Velocity
      if (ctx.fields->exists("ibm.velocity")) {
          auto h = ctx.fields->get("ibm.velocity");
          auto data = repack_float3(h.as<float3>(), nPoints);
          write_block(data.data(), vel_size);
      }
      
      // 3. Force
      if (ctx.fields->exists("ibm.force")) {
          auto h = ctx.fields->get("ibm.force");
          auto data = repack_float3(h.as<float3>(), nPoints);
          write_block(data.data(), force_size);
      }
      
      // 4. Connectivity (0...N-1)
      {
          std::vector<int32_t> conn(nPoints);
          for(size_t i=0; i<nPoints; ++i) conn[i] = (int32_t)i;
          write_block(conn.data(), conn_size);
      }
      
      // 5. Offsets (1...N)
      {
          std::vector<int32_t> offs(nPoints);
          for(size_t i=0; i<nPoints; ++i) offs[i] = (int32_t)(i+1);
          write_block(offs.data(), offs_size);
      }
      
      file << "\n  </AppendedData>\n";
  }
  
  file << "</VTKFile>\n";
}
