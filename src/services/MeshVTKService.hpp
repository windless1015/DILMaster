#pragma once

#include "IService.hpp"
#include "physics/ibm/IBMSolver.hpp"
#include "PropellerPureLiquidScenario.hpp"
#include "geometry/STLReader.h"
#include <memory>
#include <string>

class MeshVTKService : public IService {
public:
    struct Config {
        std::string output_dir = "output";
        std::string filename_prefix = "mesh_";
        int interval = 10;
    };

    MeshVTKService(const Config& config, 
                  std::shared_ptr<IBMSolver> ibm,
                  std::shared_ptr<PropellerPureLiquidScenario> scenario);

    void initialize(const StepContext& ctx) override;
    void onStepBegin(StepContext& ctx) override;
    void onStepEnd(StepContext& ctx) override;
    void finalize(const StepContext& ctx) override;

private:
    void writeVTPFile(int step, const RigidBodyState& state);
    void writePolygonVTP(std::ofstream& file, const std::vector<float3>& vertices); // Helper

    Config config_;
    std::shared_ptr<IBMSolver> ibm_;
    std::shared_ptr<PropellerPureLiquidScenario> scenario_;
    bool initialized_ = false;
};
