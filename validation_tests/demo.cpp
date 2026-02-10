#include "validation_tests/validation_framework.h"
#include <iostream>

int main() {
    std::cout << "=== LBM+IB+DEM 验证框架演示 ===" << std::endl;
    
    // 演示1: 快速验证
    std::cout << "\n1. 快速验证演示:" << std::endl;
    bool quick_result = ValidationFramework::quickCheck();
    std::cout << "快速验证结果: " << (quick_result ? "通过" : "失败") << std::endl;
    
    // 演示2: 单个模块测试
    std::cout << "\n2. 单个模块测试演示:" << std::endl;
    bool lbm_ok = ValidationFramework::SingleModule::testLBM();
    bool ibm_ok = ValidationFramework::SingleModule::testIBM(); 
    bool dem_ok = ValidationFramework::SingleModule::testDEM();
    
    std::cout << "LBM模块: " << (lbm_ok ? "通过" : "失败") << std::endl;
    std::cout << "IBM模块: " << (ibm_ok ? "通过" : "失败") << std::endl;
    std::cout << "DEM模块: " << (dem_ok ? "通过" : "失败") << std::endl;
    
    // 演示3: 耦合测试
    std::cout << "\n3. 耦合测试演示:" << std::endl;
    bool lbm_ibm_ok = ValidationFramework::Coupling::testLBM_IBM();
    bool lbm_dem_ok = ValidationFramework::Coupling::testLBM_DEM();
    bool ibm_dem_ok = ValidationFramework::Coupling::testIBM_DEM();
    
    std::cout << "LBM+IBM耦合: " << (lbm_ibm_ok ? "通过" : "失败") << std::endl;
    std::cout << "LBM+DEM耦合: " << (lbm_dem_ok ? "通过" : "失败") << std::endl;
    std::cout << "IBM+DEM耦合: " << (ibm_dem_ok ? "通过" : "失败") << std::endl;
    
    // 演示4: DKT测试
    std::cout << "\n4. DKT实验测试演示:" << std::endl;
    bool dkt_quick = ValidationFramework::DKT::quickTest();
    std::cout << "DKT快速测试: " << (dkt_quick ? "通过" : "失败") << std::endl;
    
    // 演示5: 自定义DKT配置
    std::cout << "\n5. 自定义DKT配置演示:" << std::endl;
    ValidationTests::DKT_ValidationStrategy::DKTConfig custom_config;
    custom_config.sphere1_radius = 0.012f;
    custom_config.sphere2_radius = 0.012f;
    custom_config.flow_velocity = 0.08f;
    custom_config.resolution = 48;
    custom_config.max_steps = 80000;
    
    bool dkt_custom = ValidationFramework::DKT::customTest(custom_config);
    std::cout << "自定义DKT测试: " << (dkt_custom ? "通过" : "失败") << std::endl;
    
    // 演示6: 完整验证报告
    std::cout << "\n6. 完整验证报告演示:" << std::endl;
    auto report = ValidationFramework::standardValidation();
    report.printSummary();
    
    std::cout << "\n=== 演示结束 ===" << std::endl;
    std::cout << "建议: ";
    if (report.allTestsPassed()) {
        std::cout << "所有测试通过，可以开始DKT实验！" << std::endl;
    } else {
        std::cout << "部分测试失败，请根据报告修复问题后再进行DKT实验。" << std::endl;
    }
    
    return 0;
}