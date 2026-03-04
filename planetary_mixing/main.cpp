#include "PlanetaryPaddleModule.hpp"
#include <iostream>

int main(int argc, char** argv) {
    PlanetaryPaddleModule::Config config;
    std::string output_dir = "output/planetary_paddle_mesh";
    int frames = 200;
    double dt = 0.01;
    // Set all parameters here for easy tweaking.
    config.near_offset = 0.01475f;
    config.far_offset = 0.0295f;
    config.revolution_rpm = 30.0f;
    config.revolution_phase_deg = 0.0f;
    config.near_orbit_phase_deg = 180.0f;
    config.near_spin_rpm = -30.0f;
    config.near_spin_phase_deg = 0.0f;
    config.far_spin_rpm = 60.0f;
    config.far_spin_phase_deg = -30.0f;
        
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--reverse") {
            config.mode = PlanetaryPaddleModule::RotationMode::Reverse;
        } else if (arg == "--spin-absolute") {
                    } else if (arg == "--far-phase-deg" && i + 1 < argc) {
            config.far_spin_phase_deg = std::stof(argv[++i]);
        } else if (arg == "--rev-phase-deg" && i + 1 < argc) {
            config.revolution_phase_deg = std::stof(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--frames" && i + 1 < argc) {
            frames = std::stoi(argv[++i]);
        } else if (arg == "--dt" && i + 1 < argc) {
            dt = std::stod(argv[++i]);
        }
    }

    PlanetaryPaddleModule module(config);
    if (!module.load()) {
        std::cerr << "Failed to load STL files: "
                  << config.near_stl << ", " << config.far_stl << std::endl;
        return 1;
    }

    module.writeVTPSeries(output_dir, frames, dt);
    std::cout << "Wrote VTP frames to " << output_dir << std::endl;
    return 0;
}
