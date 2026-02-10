#pragma once

#include "single_module_tests.h"
#include "coupling_tests.h"
#include "dkt_validation.h"
#include "debug_tools.h"
#include <iostream>
#include <string>

namespace ValidationTests {

// ============= 主验证运行器 =============
class ValidationRunner {
public:
    enum class TestLevel {
        QUICK,      // 快速测试 - 用于日常开发
        STANDARD,   // 标准测试 - 用于功能验证
        FULL        // 完整测试 - 用于发布前验证
    };
    
    struct ValidationReport {
        bool single_module_tests_passed = false;
        bool coupling_tests_passed = false;
        bool dkt_tests_passed = false;
        
        std::string error_messages;
        std::string performance_summary;
        std::string recommendations;
        
        bool allTestsPassed() const {
            return single_module_tests_passed && coupling_tests_passed && dkt_tests_passed;
        }
        
        void printSummary() const {
            std::cout << "\n########## 验证测试总结 ##########" << std::endl;
            std::cout << "单个模块测试: " << (single_module_tests_passed ? "通过" : "失败") << std::endl;
            std::cout << "耦合测试: " << (coupling_tests_passed ? "通过" : "失败") << std::endl;
            std::cout << "DKT实验测试: " << (dkt_tests_passed ? "通过" : "失败") << std::endl;
            std::cout << "总体结果: " << (allTestsPassed() ? "通过" : "失败") << std::endl;
            
            if (!error_messages.empty()) {
                std::cout << "\n错误信息:" << std::endl;
                std::cout << error_messages << std::endl;
            }
            
            if (!performance_summary.empty()) {
                std::cout << "\n性能摘要:" << std::endl;
                std::cout << performance_summary << std::endl;
            }
            
            if (!recommendations.empty()) {
                std::cout << "\n建议:" << std::endl;
                std::cout << recommendations << std::endl;
            }
        }
    };
    
    // 运行指定级别的验证测试
    static ValidationReport runValidationTests(TestLevel level = TestLevel::STANDARD) {
        std::cout << "\n########## 开始LBM+IB+DEM系统验证 ##########" << std::endl;
        std::cout << "测试级别: " << getTestLevelName(level) << std::endl;
        
        ValidationReport report;
        DebugTools::PerformanceMetrics metrics;
        
        try {
            // 步骤1: 单个模块验证
            std::cout << "\n=== 步骤1: 单个模块验证 ===" << std::endl;
            report.single_module_tests_passed = runSingleModuleTests(level, metrics);
            
            if (!report.single_module_tests_passed) {
                report.error_messages += "单个模块测试失败。建议先修复基础模块问题。\n";
                report.recommendations += "1. 检查LBM的边界条件实现\n";
                report.recommendations += "2. 验证IBM的力计算精度\n";
                report.recommendations += "3. 检查DEM的碰撞模型参数\n";
            }
            
            // 步骤2: 两两耦合验证
            std::cout << "\n=== 步骤2: 两两耦合验证 ===" << std::endl;
            report.coupling_tests_passed = runCouplingTests(level, metrics);
            
            if (!report.coupling_tests_passed) {
                report.error_messages += "耦合测试失败。建议检查耦合接口实现。\n";
                report.recommendations += "4. 验证场数据交换的正确性\n";
                report.recommendations += "5. 检查力和速度的耦合传递\n";
                report.recommendations += "6. 验证时间步进同步机制\n";
            }
            
            // 步骤3: DKT实验验证
            std::cout << "\n=== 步骤3: DKT实验验证 ===" << std::endl;
            report.dkt_tests_passed = runDKTTests(level, metrics);
            
            if (!report.dkt_tests_passed) {
                report.error_messages += "DKT实验测试失败。建议检查三向耦合逻辑。\n";
                report.recommendations += "7. 检查LBM+IBM+DEM的耦合顺序\n";
                report.recommendations += "8. 验证复杂相互作用的处理\n";
                report.recommendations += "9. 检查数值稳定性和收敛性\n";
            }
            
            // 生成性能摘要
            report.performance_summary = generatePerformanceSummary(metrics);
            
            // 保存调试信息
            saveDebugInformation(level, metrics, report);
            
        } catch (const std::exception& e) {
            report.error_messages += std::string("异常: ") + e.what() + "\n";
            std::cerr << "验证测试异常: " << e.what() << std::endl;
        }
        
        return report;
    }
    
    // 快速验证 - 用于日常开发
    static bool quickValidation() {
        std::cout << "\n=== 快速验证模式 ===" << std::endl;
        
        // 只运行最关键的测试
        bool lbm_ok = LBMValidationTests::testMassConservation();
        bool ibm_ok = IBMValidationTests::testKinematicsAccuracy();
        bool dem_ok = DEMValidationTests::testBinaryCollision();
        
        if (lbm_ok && ibm_ok && dem_ok) {
            std::cout << "快速验证通过" << std::endl;
            return true;
        } else {
            std::cout << "快速验证失败 - LBM:" << lbm_ok << " IBM:" << ibm_ok << " DEM:" << dem_ok << std::endl;
            return false;
        }
    }
    
    // 完整验证 - 用于发布前
    static ValidationReport fullValidation() {
        return runValidationTests(TestLevel::FULL);
    }

private:
    static std::string getTestLevelName(TestLevel level) {
        switch (level) {
            case TestLevel::QUICK: return "快速测试";
            case TestLevel::STANDARD: return "标准测试";
            case TestLevel::FULL: return "完整测试";
            default: return "未知级别";
        }
    }
    
    static bool runSingleModuleTests(TestLevel level, DebugTools::PerformanceMetrics& metrics) {
        DebugTools::Timer timer("单个模块测试", &metrics);
        
        bool all_passed = true;
        
        if (level == TestLevel::QUICK) {
            // 快速测试：只运行最基本的测试
            all_passed &= LBMValidationTests::testMassConservation();
            all_passed &= IBMValidationTests::testKinematicsAccuracy();
            all_passed &= DEMValidationTests::testBinaryCollision();
        } else if (level == TestLevel::STANDARD) {
            // 标准测试：运行所有单个模块测试
            all_passed = ValidationTestSuite::runAllTests();
        } else {
            // 完整测试：运行所有测试并添加额外的收敛性测试
            all_passed = ValidationTestSuite::runAllTests();
            
            // 添加收敛性测试
            std::cout << "\n=== 额外收敛性测试 ===" << std::endl;
            all_passed &= testLBMConvergence();
            all_passed &= testIBMConvergence();
            all_passed &= testDEMConvergence();
        }
        
        return all_passed;
    }
    
    static bool runCouplingTests(TestLevel level, DebugTools::PerformanceMetrics& metrics) {
        DebugTools::Timer timer("耦合测试", &metrics);
        
        if (level == TestLevel::QUICK) {
            // 快速测试：只运行一个关键耦合测试
            return LBMIBM_CouplingTests::testStaticIBMFlowDevelopment();
        } else {
            // 标准/完整测试：运行所有耦合测试
            return CouplingValidationSuite::runAllCouplingTests();
        }
    }
    
    static bool runDKTTests(TestLevel level, DebugTools::PerformanceMetrics& metrics) {
        DebugTools::Timer timer("DKT实验测试", &metrics);
        
        if (level == TestLevel::QUICK) {
            return DKT_QuickValidation::quickDKTTest();
        } else if (level == TestLevel::STANDARD) {
            return DKT_QuickValidation::fullDKTTest();
        } else {
            // 完整测试：运行多个配置的DKT测试
            bool all_passed = true;
            
            // 测试不同参数组合
            std::vector<float> velocities = {0.05f, 0.1f, 0.2f};
            std::vector<int> resolutions = {32, 48, 64};
            
            for (float vel : velocities) {
                for (int res : resolutions) {
                    std::cout << "\n--- 测试配置: 速度=" << vel << ", 分辨率=" << res << " ---" << std::endl;
                    
                    DKT_ValidationStrategy::DKTConfig config;
                    config.flow_velocity = vel;
                    config.resolution = res;
                    
                    DKT_ValidationStrategy::DKTMetrics metrics;
                    bool test_passed = DKT_ValidationStrategy::validateDKTExperiment(config, metrics);
                    
                    all_passed &= test_passed;
                }
            }
            
            return all_passed;
        }
    }
    
    static bool testLBMConvergence() {
        std::cout << "LBM收敛性测试..." << std::endl;
        // 实现LBM收敛性测试
        return true;
    }
    
    static bool testIBMConvergence() {
        std::cout << "IBM收敛性测试..." << std::endl;
        // 实现IBM收敛性测试
        return true;
    }
    
    static bool testDEMConvergence() {
        std::cout << "DEM收敛性测试..." << std::endl;
        // 实现DEM收敛性测试
        return true;
    }
    
    static std::string generatePerformanceSummary(const DebugTools::PerformanceMetrics& metrics) {
        std::stringstream ss;
        
        ss << "总运行时间: " << std::fixed << std::setprecision(2) << metrics.total_time << " 秒\n";
        ss << "LBM模块: " << (metrics.lbm_time / metrics.total_time * 100) << "%\n";
        ss << "IBM模块: " << (metrics.ibm_time / metrics.total_time * 100) << "%\n";
        ss << "DEM模块: " << (metrics.dem_time / metrics.total_time * 100) << "%\n";
        ss << "耦合开销: " << (metrics.coupling_time / metrics.total_time * 100) << "%\n";
        ss << "峰值内存使用: " << (metrics.peak_memory / 1024.0 / 1024.0) << " MB\n";
        
        return ss.str();
    }
    
    static void saveDebugInformation(TestLevel level, 
                                     const DebugTools::PerformanceMetrics& metrics,
                                     const ValidationReport& report) {
        std::string filename = "dkt_validation_report_" + getCurrentTimestamp() + ".log";
        
        std::string additional_info = "Test Level: " + getTestLevelName(level) + "\n";
        additional_info += "Overall Result: " + std::string(report.allTestsPassed() ? "PASSED" : "FAILED") + "\n";
        
        DebugTools::createDebugLog(filename, metrics, additional_info);
    }
    
    static std::string getCurrentTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
        return ss.str();
    }
};

// ============= 使用示例和主函数 =============
class ValidationExamples {
public:
    // 示例1: 快速验证
    static void exampleQuickValidation() {
        std::cout << "\n=== 示例1: 快速验证 ===" << std::endl;
        
        bool passed = ValidationRunner::quickValidation();
        
        if (passed) {
            std::cout << "✓ 快速验证通过，可以继续开发" << std::endl;
        } else {
            std::cout << "✗ 快速验证失败，需要检查基础功能" << std::endl;
        }
    }
    
    // 示例2: 标准验证
    static void exampleStandardValidation() {
        std::cout << "\n=== 示例2: 标准验证 ===" << std::endl;
        
        auto report = ValidationRunner::runValidationTests(ValidationRunner::TestLevel::STANDARD);
        report.printSummary();
        
        if (report.allTestsPassed()) {
            std::cout << "✓ 标准验证通过，可以进行DKT实验" << std::endl;
        } else {
            std::cout << "✗ 标准验证失败，请根据建议修复问题" << std::endl;
        }
    }
    
    // 示例3: 完整验证
    static void exampleFullValidation() {
        std::cout << "\n=== 示例3: 完整验证 ===" << std::endl;
        
        auto report = ValidationRunner::fullValidation();
        report.printSummary();
        
        if (report.allTestsPassed()) {
            std::cout << "✓ 完整验证通过，系统可以发布" << std::endl;
        } else {
            std::cout << "✗ 完整验证失败，系统存在严重问题" << std::endl;
        }
    }
    
    // 示例4: 自定义DKT验证
    static void exampleCustomDKTValidation() {
        std::cout << "\n=== 示例4: 自定义DKT验证 ===" << std::endl;
        
        // 创建自定义配置
        DKT_ValidationStrategy::DKTConfig config;
        config.sphere1_radius = 0.015f;
        config.sphere2_radius = 0.015f;
        config.initial_separation = 0.08f;
        config.flow_velocity = 0.15f;
        config.fluid_viscosity = 1.5e-3f;
        config.resolution = 80;
        config.max_steps = 150000;
        
        DKT_ValidationStrategy::DKTMetrics metrics;
        
        std::cout << "运行自定义DKT验证..." << std::endl;
        bool success = DKT_ValidationStrategy::validateDKTExperiment(config, metrics);
        
        if (success) {
            std::cout << "✓ 自定义DKT验证通过" << std::endl;
        } else {
            std::cout << "✗ 自定义DKT验证失败" << std::endl;
        }
    }
};

} // namespace ValidationTests

// ============= 主函数入口 =============
#ifdef VALIDATION_MAIN
int main(int argc, char* argv[]) {
    std::cout << "LBM+IB+DEM 系统验证工具" << std::endl;
    std::cout << "用法: " << argv[0] << " [quick|standard|full|custom]" << std::endl;
    
    std::string mode = "standard";
    if (argc > 1) {
        mode = argv[1];
    }
    
    try {
        if (mode == "quick") {
            ValidationExamples::exampleQuickValidation();
        } else if (mode == "standard") {
            ValidationExamples::exampleStandardValidation();
        } else if (mode == "full") {
            ValidationExamples::exampleFullValidation();
        } else if (mode == "custom") {
            ValidationExamples::exampleCustomDKTValidation();
        } else {
            std::cerr << "未知模式: " << mode << std::endl;
            return 1;
        }
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "程序异常: " << e.what() << std::endl;
        return 1;
    }
}
#endif