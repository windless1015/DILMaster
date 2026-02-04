// src/services/VTKService.cpp
// VTI 格式输出 — flags (Int32 标量) + velocity (Float32 向量, SoA→交错)
#include "VTKService.hpp"
#include "../core/FieldStore.hpp"
#include "../core/StepContext.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;

VTKService::VTKService(const Config &config) : config_(config) {}
VTKService::~VTKService() {}

void VTKService::initialize(const StepContext &ctx) {
  fs::create_directories(config_.output_dir);
  initialized_ = true;
  (void)ctx;
}

void VTKService::onStepEnd(StepContext &ctx) {
  if (!initialized_)
    return;
  if (static_cast<int>(ctx.step) % config_.interval != 0)
    return;
  if (!ctx.fields)
    return;

  writeVTIFile(ctx);
}

void VTKService::finalize(const StepContext &ctx) {
  initialized_ = false;
  (void)ctx;
}

void VTKService::writeVTIFile(const StepContext &ctx) {
  std::ostringstream filename;
  filename << config_.output_dir << "/step_" << std::setfill('0')
           << std::setw(6) << ctx.step << ".vti";

  std::ofstream file(filename.str(), std::ios::binary);
  if (!file.is_open())
    return;

  int nx = config_.nx;
  int ny = config_.ny;
  int nz = config_.nz;
  int nPoints = nx * ny * nz;

  // Header
  file << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt32\">\n";
  file << "  <ImageData WholeExtent=\"0 " << (nx - 1) << " 0 " << (ny - 1) << " 0 " << (nz - 1) << "\" ";
  file << "Origin=\"0 0 0\" Spacing=\"" << config_.dx << " " << config_.dx << " " << config_.dx << "\">\n";
  file << "    <Piece Extent=\"0 " << (nx - 1) << " 0 " << (ny - 1) << " 0 " << (nz - 1) << "\">\n";
  file << "      <PointData>\n";

  // Data Arrays Definitions
  uint64_t offset = 0;

  // List of active fields to write
  struct FieldInfo {
      std::string name;
      std::string vtk_name; // e.g. "Velocity"
      int components;
      size_t data_size;
  };
  std::vector<FieldInfo> active_fields;

  // Generic Fields from config (including Velocity)
  for(const auto& name : config_.fields) {
      if(ctx.fields->exists(name)) {
          auto h = ctx.fields->get(name);
          size_t comps = 1;
          if(h.element_size() == 12) comps = 3; // float3 = 12 bytes
          active_fields.push_back({name, name, (int)comps, h.size_bytes()});
      }
  }

  // XML Declaration
  for(const auto& f : active_fields) {
      file << "        <DataArray type=\"Float32\" Name=\"" << f.vtk_name 
           << "\" NumberOfComponents=\"" << f.components << "\" format=\"" 
           << (config_.binary ? "appended" : "ascii") << "\"";
      if(config_.binary) {
          file << " offset=\"" << offset << "\"";
          offset += sizeof(uint32_t) + f.data_size;
      }
      file << "/>\n";
  }

  file << "      </PointData>\n";
  file << "    </Piece>\n";
  file << "  </ImageData>\n";
  
  if(config_.binary) {
      file << "  <AppendedData encoding=\"raw\">\n";
      file << "    _";
      
      for(const auto& f : active_fields) {
          auto h = ctx.fields->get(f.name);
          const char* ptr = (const char*)h.data();
          uint32_t size = (uint32_t)f.data_size;
          
          // All fields are assumed to be in correct format (AoS for vectors)
          file.write((const char*)&size, sizeof(uint32_t));
          file.write(ptr, size);
      }
      file << "\n  </AppendedData>\n";
  }
  
  file << "</VTKFile>\n";
}

