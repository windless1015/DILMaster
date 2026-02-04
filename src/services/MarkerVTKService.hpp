#pragma once
#include "IService.hpp"
#include <string>
#include <vector>

class StepContext;

class MarkerVTKService : public IService {
public:
  struct Config {
    std::string output_dir = "output";
    int interval = 10;
    std::string marker_field = "ibm.markers";
    bool binary = true;
  };

  explicit MarkerVTKService(const Config &config);
  ~MarkerVTKService() override;

  void initialize(const StepContext &ctx) override;
  void onStepBegin(StepContext &ctx) override {}
  void onStepEnd(StepContext &ctx) override;
  void finalize(const StepContext &ctx) override;

private:
  void writeVTPFile(const StepContext &ctx);

  Config config_;
  bool initialized_ = false;
};
