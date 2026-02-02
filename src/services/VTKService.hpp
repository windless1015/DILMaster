#pragma once
#include "IService.hpp"
#include <string>
#include <vector>

class StepContext;

class VTKService : public IService {
public:
  struct Config {
    std::string output_dir = "output";
    int interval = 100;
    std::vector<std::string> fields; // 通用标量/向量字段名
    int nx = 1;
    int ny = 1;
    int nz = 1;
    float dx = 1.0f;
  };

  explicit VTKService(const Config &config);
  ~VTKService() override;

  void initialize(const StepContext &ctx) override;
  void onStepBegin(StepContext &ctx) override {}
  void onStepEnd(StepContext &ctx) override;
  void finalize(const StepContext &ctx) override;

private:
  void writeVTIFile(const StepContext &ctx);

  Config config_;
  bool initialized_ = false;
};
