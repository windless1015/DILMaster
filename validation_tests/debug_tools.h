#pragma once

#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <map>
#include <string>
#include <cuda_runtime.h>

namespace ValidationTests {

// ============= 调试工具类 =============
class DebugTools {
public:
    // 性能监测器
    struct PerformanceMetrics {
        double total_time = 0.0;
        double lbm_time = 0.0;
        double ibm_time = 0.0;
        double dem_time = 0.0;
        double coupling_time = 0.0;
        
        size_t memory_usage = 0;
        size_t peak_memory = 0;
        
        std::map<std::string, double> kernel_times;
        std::map<std::string, size_t> kernel_calls;
    };
    
    // 场数据验证器
    struct FieldValidator {
        template<typename T>
        static bool checkNaN(const T* data, size_t size, const std::string& field_name) {
            for (size_t i = 0; i < size; ++i) {
                if (std::isnan(data[i])) {
                    std::cerr << "NaN detected in field: " << field_name << " at index " << i << std::endl;
                    return false;
                }
            }
            return true;
        }
        
        template<typename T>
        static bool checkInfinity(const T* data, size_t size, const std::string& field_name) {
            for (size_t i = 0; i < size; ++i) {
                if (std::isinf(data[i])) {
                    std::cerr << "Infinity detected in field: " << field_name << " at index " << i << std::endl;
                    return false;
                }
            }
            return true;
        }
        
        template<typename T>
        static T getMinValue(const T* data, size_t size) {
            T min_val = data[0];
            for (size_t i = 1; i < size; ++i) {
                if (data[i] < min_val) min_val = data[i];
            }
            return min_val;
        }
        
        template<typename T>
        static T getMaxValue(const T* data, size_t size) {
            T max_val = data[0];
            for (size_t i = 1; i < size; ++i) {
                if (data[i] > max_val) max_val = data[i];
            }
            return max_val;
        }
    };
    
    // CUDA错误检查器
    static void checkCudaError(cudaError_t error, const std::string& message) {
        if (error != cudaSuccess) {
            std::cerr << "CUDA Error: " << message << " - " << cudaGetErrorString(error) << std::endl;
            throw std::runtime_error("CUDA error in: " + message);
        }
    }
    
    // 内存使用监测
    static size_t getCurrentMemoryUsage() {
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        return total_mem - free_mem;
    }
    
    // 计时器类
    class Timer {
    private:
        std::chrono::high_resolution_clock::time_point start_time;
        std::string operation_name;
        PerformanceMetrics* metrics;
        std::string kernel_name;
        
    public:
        Timer(const std::string& name, PerformanceMetrics* perf = nullptr, const std::string& kernel = "") 
            : operation_name(name), metrics(perf), kernel_name(kernel) {
            start_time = std::chrono::high_resolution_clock::now();
        }
        
        ~Timer() {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            double seconds = duration.count() / 1e6;
            
            std::cout << "[TIMER] " << operation_name << ": " << std::fixed << std::setprecision(6) 
                      << seconds << " seconds" << std::endl;
            
            if (metrics) {
                if (!kernel_name.empty()) {
                    metrics->kernel_times[kernel_name] += seconds;
                    metrics->kernel_calls[kernel_name]++;
                }
                
                if (operation_name.find("LBM") != std::string::npos) {
                    metrics->lbm_time += seconds;
                } else if (operation_name.find("IBM") != std::string::npos) {
                    metrics->ibm_time += seconds;
                } else if (operation_name.find("DEM") != std::string::npos) {
                    metrics->dem_time += seconds;
                } else if (operation_name.find("Coupling") != std::string::npos) {
                    metrics->coupling_time += seconds;
                }
                
                metrics->total_time += seconds;
            }
        }
    };
    
    // 数据导出工具
    static void exportFieldToCSV(const std::string& filename, const float* data, 
                                size_t nx, size_t ny, size_t nz, 
                                const std::string& field_name) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }
        
        file << "x,y,z," << field_name << std::endl;
        
        for (size_t k = 0; k < nz; ++k) {
            for (size_t j = 0; j < ny; ++j) {
                for (size_t i = 0; i < nx; ++i) {
                    size_t idx = i + j * nx + k * nx * ny;
                    file << i << "," << j << "," << k << "," << data[idx] << std::endl;
                }
            }
        }
        
        file.close();
        std::cout << "Exported field " << field_name << " to " << filename << std::endl;
    }
    
    // 收敛性分析
    static bool analyzeConvergence(const std::vector<double>& residuals, 
                                  double tolerance = 1e-6, 
                                  int min_iterations = 10) {
        if (residuals.size() < min_iterations) {
            std::cout << "Insufficient iterations for convergence analysis" << std::endl;
            return false;
        }
        
        // 检查最后几个残差是否低于容差
        int check_iterations = std::min(5, static_cast<int>(residuals.size()));
        bool converged = true;
        
        for (int i = residuals.size() - check_iterations; i < residuals.size(); ++i) {
            if (residuals[i] > tolerance) {
                converged = false;
                break;
            }
        }
        
        // 检查残差是否单调递减
        bool monotonic = true;
        for (size_t i = 1; i < residuals.size(); ++i) {
            if (residuals[i] > residuals[i-1] * 1.1) { // 允许10%的波动
                monotonic = false;
                break;
            }
        }
        
        std::cout << "Convergence analysis:" << std::endl;
        std::cout << "  Final residual: " << residuals.back() << std::endl;
        std::cout << "  Converged: " << (converged ? "Yes" : "No") << std::endl;
        std::cout << "  Monotonic: " << (monotonic ? "Yes" : "No") << std::endl;
        
        return converged && monotonic;
    }
    
    // 能量守恒检查
    template<typename T>
    static bool checkEnergyConservation(const std::vector<T>& energy_history, 
                                       T tolerance = 1e-3) {
        if (energy_history.size() < 2) return true;
        
        T initial_energy = energy_history.front();
        T final_energy = energy_history.back();
        T relative_error = std::abs(final_energy - initial_energy) / initial_energy;
        
        std::cout << "Energy conservation check:" << std::endl;
        std::cout << "  Initial energy: " << initial_energy << std::endl;
        std::cout << "  Final energy: " << final_energy << std::endl;
        std::cout << "  Relative error: " << relative_error << std::endl;
        std::cout << "  Within tolerance: " << (relative_error < tolerance ? "Yes" : "No") << std::endl;
        
        return relative_error < tolerance;
    }
    
    // 创建调试日志
    static void createDebugLog(const std::string& filename, 
                              const PerformanceMetrics& metrics,
                              const std::string& additional_info = "") {
        std::ofstream log(filename);
        if (!log.is_open()) {
            std::cerr << "Failed to create debug log: " << filename << std::endl;
            return;
        }
        
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        log << "=== DKT Simulation Debug Log ===" << std::endl;
        log << "Timestamp: " << std::ctime(&time_t);
        log << "Additional Info: " << additional_info << std::endl;
        log << std::endl;
        
        log << "=== Performance Metrics ===" << std::endl;
        log << "Total time: " << metrics.total_time << " s" << std::endl;
        log << "LBM time: " << metrics.lbm_time << " s (" 
            << (metrics.lbm_time / metrics.total_time * 100) << "%)" << std::endl;
        log << "IBM time: " << metrics.ibm_time << " s (" 
            << (metrics.ibm_time / metrics.total_time * 100) << "%)" << std::endl;
        log << "DEM time: " << metrics.dem_time << " s (" 
            << (metrics.dem_time / metrics.total_time * 100) << "%)" << std::endl;
        log << "Coupling time: " << metrics.coupling_time << " s (" 
            << (metrics.coupling_time / metrics.total_time * 100) << "%)" << std::endl;
        log << std::endl;
        
        log << "=== Memory Usage ===" << std::endl;
        log << "Peak memory: " << (metrics.peak_memory / 1024.0 / 1024.0) << " MB" << std::endl;
        log << "Final memory: " << (metrics.memory_usage / 1024.0 / 1024.0) << " MB" << std::endl;
        log << std::endl;
        
        log << "=== Kernel Performance ===" << std::endl;
        for (const auto& kernel : metrics.kernel_times) {
            log << kernel.first << ": " << kernel.second << " s ("
                << metrics.kernel_calls.at(kernel.first) << " calls)" << std::endl;
        }
        
        log.close();
        std::cout << "Debug log saved to: " << filename << std::endl;
    }
    
    // 实时监测器
    class RealtimeMonitor {
    private:
        PerformanceMetrics* metrics;
        std::chrono::steady_clock::time_point last_update;
        int update_frequency;
        
    public:
        RealtimeMonitor(PerformanceMetrics* perf, int freq = 100) 
            : metrics(perf), update_frequency(freq) {
            last_update = std::chrono::steady_clock::now();
        }
        
        void update(int step, int total_steps) {
            if (step % update_frequency == 0) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_update);
                
                if (elapsed.count() > 0) {
                    double progress = static_cast<double>(step) / total_steps * 100.0;
                    double remaining_time = (total_steps - step) * 
                        (metrics->total_time / step);
                    
                    std::cout << "\r[Progress] " << std::fixed << std::setprecision(1) 
                              << progress << "% | Step: " << step << "/" << total_steps
                              << " | Time: " << metrics->total_time << "s"
                              << " | ETA: " << remaining_time << "s"
                              << " | Memory: " << (getCurrentMemoryUsage() / 1024.0 / 1024.0) << "MB"
                              << std::flush;
                    
                    last_update = now;
                }
            }
        }
        
        void finalize() {
            std::cout << std::endl; // 换行
        }
    };
};

// ============= 专门的DKT调试工具 =============
class DKT_DebugTools {
public:
    // DKT状态监测器
    struct DKT_State {
        float sphere1_pos[3];
        float sphere2_pos[3];
        float sphere1_vel[3];
        float sphere2_vel[3];
        float separation_distance;
        float relative_velocity;
        float hydrodynamic_force[2][3]; // [sphere][component]
        int current_phase; // 0: drafting, 1: kissing, 2: tumbling
        
        void print() const {
            std::cout << "DKT State:" << std::endl;
            std::cout << "  Sphere1: (" << sphere1_pos[0] << ", " << sphere1_pos[1] 
                      << ", " << sphere1_pos[2] << ")" << std::endl;
            std::cout << "  Sphere2: (" << sphere2_pos[0] << ", " << sphere2_pos[1] 
                      << ", " << sphere2_pos[2] << ")" << std::endl;
            std::cout << "  Separation: " << separation_distance << std::endl;
            std::cout << "  Relative velocity: " << relative_velocity << std::endl;
            std::cout << "  Phase: " << current_phase << std::endl;
        }
    };
    
    // DKT历史记录器
    class DKT_History {
    private:
        std::vector<DKT_State> states;
        std::vector<float> separation_history;
        std::vector<float> velocity_history;
        std::vector<float> force_history;
        
    public:
        void record(const DKT_State& state) {
            states.push_back(state);
            separation_history.push_back(state.separation_distance);
            velocity_history.push_back(state.relative_velocity);
            
            float total_force = 0.0f;
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 3; j++) {
                    total_force += std::abs(state.hydrodynamic_force[i][j]);
                }
            }
            force_history.push_back(total_force);
        }
        
        void saveToFile(const std::string& prefix) {
            // 保存分离距离历史
            std::ofstream sep_file(prefix + "_separation.dat");
            for (size_t i = 0; i < separation_history.size(); i++) {
                sep_file << i << " " << separation_history[i] << std::endl;
            }
            sep_file.close();
            
            // 保存相对速度历史
            std::ofstream vel_file(prefix + "_velocity.dat");
            for (size_t i = 0; i < velocity_history.size(); i++) {
                vel_file << i << " " << velocity_history[i] << std::endl;
            }
            vel_file.close();
            
            // 保存力历史
            std::ofstream force_file(prefix + "_force.dat");
            for (size_t i = 0; i < force_history.size(); i++) {
                force_file << i << " " << force_history[i] << std::endl;
            }
            force_file.close();
            
            std::cout << "DKT history saved with prefix: " << prefix << std::endl;
        }
        
        DKT_State getStateAtStep(int step) const {
            if (step < states.size()) {
                return states[step];
            }
            return DKT_State();
        }
        
        // 分析DKT阶段
        std::vector<std::pair<int, int>> analyzePhases() const {
            std::vector<std::pair<int, int>> phases;
            int current_phase = -1;
            int phase_start = 0;
            
            for (size_t i = 0; i < states.size(); i++) {
                if (states[i].current_phase != current_phase) {
                    if (current_phase != -1) {
                        phases.push_back({current_phase, phase_start});
                    }
                    current_phase = states[i].current_phase;
                    phase_start = i;
                }
            }
            
            if (current_phase != -1) {
                phases.push_back({current_phase, phase_start});
            }
            
            return phases;
        }
    };
    
    // DKT异常检测器
    static bool detectAnomalies(const DKT_History& history) {
        const auto& states = history.states;
        
        // 检查突然的位置跳跃
        for (size_t i = 1; i < states.size(); i++) {
            float pos_jump = std::abs(states[i].sphere1_pos[0] - states[i-1].sphere1_pos[0]);
            if (pos_jump > 0.1f) { // 10倍网格间距
                std::cout << "Warning: Large position jump detected at step " << i << std::endl;
                return true;
            }
        }
        
        // 检查不合理的分离距离
        for (size_t i = 0; i < states.size(); i++) {
            if (states[i].separation_distance < -0.02f) { // 过度重叠
                std::cout << "Warning: Excessive overlap detected at step " << i << std::endl;
                return true;
            }
        }
        
        // 检查速度异常
        for (size_t i = 0; i < states.size(); i++) {
            float speed1 = std::sqrt(states[i].sphere1_vel[0] * states[i].sphere1_vel[0] +
                                     states[i].sphere1_vel[1] * states[i].sphere1_vel[1] +
                                     states[i].sphere1_vel[2] * states[i].sphere1_vel[2]);
            if (speed1 > 1.0f) { // 超过流体速度10倍
                std::cout << "Warning: Excessive velocity detected at step " << i << std::endl;
                return true;
            }
        }
        
        return false;
    }
    
    // DKT可视化数据生成器
    static void generateVisualizationData(const DKT_History& history, 
                                        const std::string& filename) {
        std::ofstream viz_file(filename);
        if (!viz_file.is_open()) {
            std::cerr << "Failed to create visualization file: " << filename << std::endl;
            return;
        }
        
        viz_file << "# DKT Visualization Data" << std::endl;
        viz_file << "# Columns: step, sphere1_x, sphere1_y, sphere1_z, sphere2_x, sphere2_y, sphere2_z, separation, velocity, phase" << std::endl;
        
        for (size_t i = 0; i < history.states.size(); i++) {
            const auto& state = history.states[i];
            viz_file << i << " ";
            viz_file << state.sphere1_pos[0] << " " << state.sphere1_pos[1] << " " << state.sphere1_pos[2] << " ";
            viz_file << state.sphere2_pos[0] << " " << state.sphere2_pos[1] << " " << state.sphere2_pos[2] << " ";
            viz_file << state.separation_distance << " ";
            viz_file << state.relative_velocity << " ";
            viz_file << state.current_phase << std::endl;
        }
        
        viz_file.close();
        std::cout << "Visualization data saved to: " << filename << std::endl;
    }
};

} // namespace ValidationTests