// src/services/VTKService.cpp
// VTI 格式输出（与 ref/dam_break_d3.cu 一致）
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
  if (config_.fields.empty())
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
  // 生成文件名: output_dir/step_XXXXXX.vti
  std::ostringstream filename;
  filename << config_.output_dir << "/step_" << std::setfill('0')
           << std::setw(6) << ctx.step << ".vti";

  std::ofstream file(filename.str());
  if (!file.is_open())
    return;

  int nx = config_.nx;
  int ny = config_.ny;
  int nz = config_.nz;

  // VTI 格式（XML ImageData）
  file << "<?xml version=\"1.0\"?>\n";
  file << "<VTKFile type=\"ImageData\" version=\"0.1\" "
          "byte_order=\"LittleEndian\">\n";
  file << "  <ImageData WholeExtent=\"0 " << (nx - 1) << " 0 " << (ny - 1)
       << " 0 " << (nz - 1) << "\" ";
  file << "Origin=\"0 0 0\" Spacing=\"" << config_.dx << " " << config_.dx << " " << config_.dx << "\">\n";
  file << "    <Piece Extent=\"0 " << (nx - 1) << " 0 " << (ny - 1) << " 0 "
       << (nz - 1) << "\">\n";
  file << "      <PointData>\n";

  // 写每个字段
  for (const auto &field_name : config_.fields) {
    if (!ctx.fields->exists(field_name))
      continue;

    auto handle = ctx.fields->get(field_name);
    const void *data = handle.data();
    if (!data)
      continue;

    std::size_t elem_size = handle.element_size();
    std::size_t count = handle.count();

    if (elem_size == sizeof(float)) {
      // 标量 float
      file << "        <DataArray type=\"Float32\" Name=\"" << field_name
           << "\" format=\"ascii\">\n";
      const float *fdata = static_cast<const float *>(data);
      for (std::size_t i = 0; i < count; ++i) {
        file << fdata[i];
        if ((i + 1) % 10 == 0)
          file << "\n";
        else
          file << " ";
      }
      file << "\n        </DataArray>\n";
    } else if (elem_size == sizeof(unsigned char)) {
      // uint8 标量
      file << "        <DataArray type=\"Int32\" Name=\"" << field_name
           << "\" format=\"ascii\">\n";
      const unsigned char *udata = static_cast<const unsigned char *>(data);
      for (std::size_t i = 0; i < count; ++i) {
        file << static_cast<int>(udata[i]);
        if ((i + 1) % 20 == 0)
          file << "\n";
        else
          file << " ";
      }
      file << "\n        </DataArray>\n";
    } else if (elem_size == sizeof(float) * 3) {
      // float3 (AOS)
      file << "        <DataArray type=\"Float32\" Name=\"" << field_name
           << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
      const float *fdata = static_cast<const float *>(data);
      for (std::size_t i = 0; i < count; ++i) {
        // AOS: x, y, z are contiguous
        file << fdata[i * 3 + 0] << " " << fdata[i * 3 + 1] << " " << fdata[i * 3 + 2];
        if ((i + 1) % 10 == 0)
          file << "\n";
        else
          file << " ";
      }
      file << "\n        </DataArray>\n";
    }
  }

  // 处理向量字段（velocity 特殊处理）
  if (ctx.fields->exists("velocity")) {
    auto handle = ctx.fields->get("velocity");
    const float *vdata = static_cast<const float *>(handle.data());
    int nPoints = nx * ny * nz;

    file << "        <DataArray type=\"Float32\" Name=\"Velocity\" "
            "NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i = 0; i < nPoints; ++i) {
      file << vdata[i] << " " << vdata[nPoints + i] << " "
           << vdata[2 * nPoints + i];
      if ((i + 1) % 4 == 0)
        file << "\n";
      else
        file << " ";
    }
    file << "\n        </DataArray>\n";
  }

  file << "      </PointData>\n";
  file << "    </Piece>\n";
  file << "  </ImageData>\n";
  file << "</VTKFile>\n";
}