#pragma once

#include "ILBMModule.hpp"
#include "LBMConfig.hpp"
#include "cuda/LBMBackend.hpp"
#include <cstdint>

// 前向声明
class FieldStore;

namespace lbm {

// Cell Types for Free Surface
enum class CellType : uint8_t { GAS = 0, LIQUID = 1, INTERFACE = 2, SOLID = 3 };

class FreeSurfaceModule : public ILBMModule {
public:
  FreeSurfaceModule();
  ~FreeSurfaceModule() override;

  // Disable copy/move
  FreeSurfaceModule(const FreeSurfaceModule &) = delete;
  FreeSurfaceModule &operator=(const FreeSurfaceModule &) = delete;

  // ILBMModule lifecycle
  void configure(const LBMConfig &config) override;
  void allocate(FieldStore &fields) override;
  void initialize(FieldStore &fields) override;
  void preStream(StepContext &ctx) override;
  void postStream(StepContext &ctx) override;
  void finalize() override;

  // Configuration
  bool isEnabled() const { return enabled_; }
  void setEnabled(bool e) { enabled_ = e; }

  // Data Access (Device Pointers)
  uint8_t *getCellType() const;
  float *getFill() const;
  float *getMass() const;

  // Mass bookkeeping array (double precision, 4 buckets)
  double *getMassDifference() const;

  // Dimensions
  int nCells() const { return nCells_; }
  int nx() const { return nx_; }
  int ny() const { return ny_; }
  int nz() const { return nz_; }

  // Geometry Setup
  void initFlatSurface(float level, float rho0);
  void setRegion(int x0, int x1, int y0, int y1, int z0, int z1, CellType type,
                 float fill, float rho0);
  void fixInterfaceLayer();

  // Physics Steps (stubs for future implementation)
  void updateMass(const float *f_soa);
  void completeInterface(float *f_soa, const float3 *u_field);
  void updateMacroscopic();
  void distributeMass();
  void interfaceTransitions();
  void removeFalseInterfaceCells();
  void redistributeLostMass();
  void cleanupInterface();
  void reclassifyCells();
  void reconstructInterface(float *f, const float3 *u_field);

  // Diagnostics
  bool checkHealth();

private:
  bool enabled_;
  int nx_, ny_, nz_, nCells_;
  float rho0_;
  int wallFlags_;

  // 设备指针：从 FieldStore 获取（非拥有）
  uint8_t *d_flags_ = nullptr;   // "fluid.flags"
  float *d_phi_ = nullptr;       // "fluid.phi"
  float *d_mass_ = nullptr;      // "fluid.mass"

  // 模块自有的 GPU 缓冲区（通过 cudaMalloc 分配）
  uint8_t *d_cellType_ = nullptr;    // fs.cellType
  double *d_massDiff_ = nullptr;     // fs.massDifference
};

} // namespace lbm
