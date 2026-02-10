#include "validation_tests/validation_framework.h"
#include <iostream>
#include <chrono>

// 验证执行器 - 按照推荐顺序执行验证
class ValidationExecutor {
public:
    
    // ===== 推荐的验证顺序 =====
    static void executeRecommendedSequence() {
        std::cout << "\n========== LBM+IB+DEM 系统验证序列 ==========" << std::endl;
        std::cout << "按照问题发现概率从高到低排序\n" << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 步骤1: 快速完整性检查 (30秒)
        if (!executeStep1_QuickCheck()) {
            std::cout << "\n⚠️  快速检查失败，建议先修复基础问题再继续" << std::endl;
            return;
        }
        
        // 步骤2: 单个模块验证 (5-10分钟)
        if (!executeStep2_SingleModules()) {
            std::cout << "\n⚠️  单个模块测试失败，定位到具体模块问题" << std::endl;
            return;
        }
        
        // 步骤3: 两两耦合验证 (10-20分钟)  
        if (!executeStep3_Coupling()) {
            std::cout << "\n⚠️  耦合测试失败，这是您之前遇到的问题类型" << std::endl;
            return;
        }
        
        // 步骤4: DKT特定验证 (15-30分钟)
        if (!executeStep4_DKTSpecific()) {
            std::cout << "\n⚠️  DKT测试失败，检查三向耦合逻辑" << std::endl;
            return;
        }
        
        // 步骤5: 完整系统验证 (30-60分钟)
        executeStep5_FullValidation();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time);
        
        std::cout << "\n✅ 验证序列完成！总耗时: " << duration.count() << " 分钟" << std::endl;
        std::cout << "\n🎯 您的系统已准备好进行DKT实验" << std::endl;
    }
    
    // ===== 针对您的问题的专项验证 =====
    static void executeProblemFocusedValidation() {
        std::cout << "\n========== 针对耦合问题的专项验证 ==========" << std::endl;
        std::cout << "重点验证LBM+IB+DEM两两交互逻辑\n" << std::endl;
        
        // 基于您提到的"两两交互有重大缺陷"，重点检查耦合逻辑
        std::cout << "🔍 重点检查区域:" << std::endl;
        std::cout << "  1. LBM⇄IBM 数据交换时机和格式" << std::endl;
        std::cout << "  2. IBM⇄DEM 力传递机制" << std::endl;
        std::cout << "  3. LBM⇄DEM 动量守恒" << std::endl;
        std::cout << "  4. 三模块时序同步" << std::endl;
        
        // 先运行单个模块确保基础正确
        std::cout << "\n--- 阶段1: 验证单个模块基础功能 ---" << std::endl;
        bool single_ok = ValidationFramework::SingleModule::testAll();
        if (!single_ok) {
            std::cout << "❌ 单个模块存在问题，请先修复" << std::endl;
            return;
        }
        std::cout << "✅ 单个模块验证通过" << std::endl;
        
        // 重点运行耦合测试
        std::cout << "\n--- 阶段2: 重点验证两两耦合 ---" << std::endl;
        
        std::cout << "\n🔍 LBM+IBM耦合详细检查:" << std::endl;
        bool lbm_ibm_1 = ValidationTests::LBMIBM_CouplingTests::testStaticIBMFlowDevelopment();
        bool lbm_ibm_2 = ValidationTests::LBMIBM_CouplingTests::testMovingIBMForceCalculation();
        bool lbm_ibm_3 = ValidationTests::LBMIBM_CouplingTests::testLBMIBM_MassConservation();
        std::cout << "  静态IBM流场: " << (lbm_ibm_1 ? "✅" : "❌") << std::endl;
        std::cout << "  移动IBM力计算: " << (lbm_ibm_2 ? "✅" : "❌") << std::endl;
        std::cout << "  质量守恒: " << (lbm_ibm_3 ? "✅" : "❌") << std::endl;
        
        std::cout << "\n🔍 LBM+DEM耦合详细检查:" << std::endl;
        bool lbm_dem_1 = ValidationTests::LBMDEM_CouplingTests::testParticleSettlingInFluid();
        bool lbm_dem_2 = ValidationTests::LBMDEM_CouplingTests::testMultiParticleCollisionInFluid();
        bool lbm_dem_3 = ValidationTests::LBMDEM_CouplingTests::testLBMDEM_MomentumConservation();
        std::cout << "  颗粒沉降: " << (lbm_dem_1 ? "✅" : "❌") << std::endl;
        std::cout << "  多颗粒碰撞: " << (lbm_dem_2 ? "✅" : "❌") << std::endl;
        std::cout << "  动量守恒: " << (lbm_dem_3 ? "✅" : "❌") << std::endl;
        
        std::cout << "\n🔍 IBM+DEM耦合详细检查:" << std::endl;
        bool ibm_dem_1 = ValidationTests::IBMDEM_CouplingTests::testIBM_DEM_Collision();
        bool ibm_dem_2 = ValidationTests::IBMDEM_CouplingTests::testMovingIBMOnDEMParticles();
        bool ibm_dem_3 = ValidationTests::IBMDEM_CouplingTests::testIBM_DEM_ForceTransfer();
        std::cout << "  IBM-DEM碰撞: " << (ibm_dem_1 ? "✅" : "❌") << std::endl;
        std::cout << "  移动IBM在DEM上: " << (ibm_dem_2 ? "✅" : "❌") << std::endl;
        std::cout << "  力传递机制: " << (ibm_dem_3 ? "✅" : "❌") << std::endl;
        
        bool all_coupling_ok = (lbm_ibm_1 && lbm_ibm_2 && lbm_ibm_3 && 
                               lbm_dem_1 && lbm_dem_2 && lbm_dem_3 &&
                               ibm_dem_1 && ibm_dem_2 && ibm_dem_3);
        
        if (!all_coupling_ok) {
            std::cout << "\n⚠️  发现耦合问题！建议:" << std::endl;
            if (!lbm_ibm_1 || !lbm_ibm_2 || !lbm_ibm_3) {
                std::cout << "  - 检查LBM⇄IBM数据交换格式和时机" << std::endl;
            }
            if (!lbm_dem_1 || !lbm_dem_2 || !lbm_dem_3) {
                std::cout << "  - 验证LBM⇄DEM动量交换算法" << std::endl;
            }
            if (!ibm_dem_1 || !ibm_dem_2 || !ibm_dem_3) {
                std::cout << "  - 检查IBM⇄DEM力传递机制" << std::endl;
            }
            return;
        }
        
        std::cout << "\n✅ 两两耦合验证通过" << std::endl;
        
        // 最后运行DKT测试
        std::cout << "\n--- 阶段3: DKT三向耦合测试 ---" << std::endl;
        bool dkt_ok = ValidationFramework::DKT::quickTest();
        if (dkt_ok) {
            std::cout << "✅ DKT三向耦合正常" << std::endl;
        } else {
            std::cout << "❌ DKT三向耦合存在问题" << std::endl;
            std::cout << "建议: 检查三模块的时序同步和复杂相互作用处理" << std::endl;
        }
    }
    
private:
    
    static bool executeStep1_QuickCheck() {
        std::cout << "\n【步骤1】快速完整性检查 (预计30秒)" << std::endl;
        std::cout << "检查最基本的数值稳定性和数据完整性..." << std::endl;
        
        bool passed = ValidationFramework::quickCheck();
        
        if (passed) {
            std::cout << "✅ 快速检查通过 - 基础数值计算正常" << std::endl;
        } else {
            std::cout << "❌ 快速检查失败 - 发现基础数值问题" << std::endl;
            std::cout << "🔧 建议: 检查NaN/Inf、内存越界、基本算法实现" << std::endl;
        }
        
        return passed;
    }
    
    static bool executeStep2_SingleModules() {
        std::cout << "\n【步骤2】单个模块验证 (预计5-10分钟)" << std::endl;
        std::cout << "确保每个模块独立工作正常..." << std::endl;
        
        std::cout << "\n  测试LBM模块:" << std::endl;
        bool lbm_ok = ValidationFramework::SingleModule::testLBM();
        std::cout << "  " << (lbm_ok ? "✅" : "❌") << " LBM基础功能" << std::endl;
        
        std::cout << "\n  测试IBM模块:" << std::endl;
        bool ibm_ok = ValidationFramework::SingleModule::testIBM();
        std::cout << "  " << (ibm_ok ? "✅" : "❌") << " IBM基础功能" << std::endl;
        
        std::cout << "\n  测试DEM模块:" << std::endl;
        bool dem_ok = ValidationFramework::SingleModule::testDEM();
        std::cout << "  " << (dem_ok ? "✅" : "❌") << " DEM基础功能" << std::endl;
        
        bool all_ok = lbm_ok && ibm_ok && dem_ok;
        
        if (all_ok) {
            std::cout << "\n✅ 所有单个模块验证通过" << std::endl;
        } else {
            std::cout << "\n❌ 单个模块验证失败" << std::endl;
            if (!lbm_ok) std::cout << "🔧 LBM问题: 检查质量守恒、边界条件、松弛时间" << std::endl;
            if (!ibm_ok) std::cout << "🔧 IBM问题: 检查标记点分布、力计算、运动学" << std::endl;
            if (!dem_ok) std::cout << "🔧 DEM问题: 检查碰撞模型、时间步长、接触力" << std::endl;
        }
        
        return all_ok;
    }
    
    static bool executeStep3_Coupling() {
        std::cout << "\n【步骤3】两两耦合验证 (预计10-20分钟)" << std::endl;
        std::cout << "这是您之前遇到问题的关键步骤..." << std::endl;
        
        std::cout << "\n  测试LBM+IBM耦合:" << std::endl;
        bool lbm_ibm_ok = ValidationFramework::Coupling::testLBM_IBM();
        std::cout << "  " << (lbm_ibm_ok ? "✅" : "❌") << " LBM-IBM耦合" << std::endl;
        
        std::cout << "\n  测试LBM+DEM耦合:" << std::endl;
        bool lbm_dem_ok = ValidationFramework::Coupling::testLBM_DEM();
        std::cout << "  " << (lbm_dem_ok ? "✅" : "❌") << " LBM-DEM耦合" << std::endl;
        
        std::cout << "\n  测试IBM+DEM耦合:" << std::endl;
        bool ibm_dem_ok = ValidationFramework::Coupling::testIBM_DEM();
        std::cout << "  " << (ibm_dem_ok ? "✅" : "❌") << " IBM-DEM耦合" << std::endl;
        
        bool all_coupling_ok = lbm_ibm_ok && lbm_dem_ok && ibm_dem_ok;
        
        if (all_coupling_ok) {
            std::cout << "\n✅ 所有两两耦合验证通过" << std::endl;
        } else {
            std::cout << "\n❌ 耦合验证失败 - 这正是您遇到的问题类型" << std::endl;
            if (!lbm_ibm_ok) {
                std::cout << "🔧 LBM+IBM问题: 检查数据交换时机、插值函数、力-速度耦合" << std::endl;
            }
            if (!lbm_dem_ok) {
                std::cout << "🔧 LBM+DEM问题: 检查动量交换、颗粒-流体相互作用、双向耦合" << std::endl;
            }
            if (!ibm_dem_ok) {
                std::cout << "🔧 IBM+DEM问题: 检查复杂边界处理、接触检测、力传递机制" << std::endl;
            }
        }
        
        return all_coupling_ok;
    }
    
    static bool executeStep4_DKTSpecific() {
        std::cout << "\n【步骤4】DKT实验特定验证 (预计15-30分钟)" << std::endl;
        std::cout << "验证三向耦合在DKT场景下的正确性..." << std::endl;
        
        std::cout << "\n  运行DKT快速测试:" << std::endl;
        bool dkt_ok = ValidationFramework::DKT::quickTest();
        std::cout << "  " << (dkt_ok ? "✅" : "❌") << " DKT三阶段验证" << std::endl;
        
        if (dkt_ok) {
            std::cout << "\n✅ DKT特定验证通过" << std::endl;
            std::cout << "  drafting阶段: 牵引效应正常" << std::endl;
            std::cout << "  kissing阶段: 近距离相互作用正常" << std::endl;
            std::cout << "  tumbling阶段: 翻滚动力学正常" << std::endl;
        } else {
            std::cout << "\n❌ DKT验证失败" << std::endl;
            std::cout << "🔧 建议: 检查三模块时序同步、复杂相互作用、数值稳定性" << std::endl;
        }
        
        return dkt_ok;
    }
    
    static void executeStep5_FullValidation() {
        std::cout << "\n【步骤5】完整系统验证 (预计30-60分钟)" << std::endl;
        std::cout << "运行所有测试用例和收敛性分析..." << std::endl;
        
        auto report = ValidationFramework::fullValidation();
        
        std::cout << "\n📊 完整验证报告:" << std::endl;
        report.printSummary();
        
        if (report.allTestsPassed()) {
            std::cout << "\n🎉 恭喜！您的LBM+IB+DEM系统完全通过验证！" << std::endl;
            std::cout << "📋 可以安全地进行DKT实验，成功率很高" << std::endl;
        } else {
            std::cout << "\n⚠️  系统仍存在一些问题，建议修复后再进行DKT实验" << std::endl;
            if (!report.single_module_tests_passed) {
                std::cout << "🔧 基础模块需要修复" << std::endl;
            }
            if (!report.coupling_tests_passed) {
                std::cout << "🔧 耦合逻辑需要修复" << std::endl;
            }
            if (!report.dkt_tests_passed) {
                std::cout << "🔧 DKT特定逻辑需要修复" << std::endl;
            }
        }
    }
};

// ===== 主函数：选择验证模式 =====
int main(int argc, char* argv[]) {
    std::cout << "LBM+IB+DEM 验证执行器" << std::endl;
    std::cout << "针对您提到的DKT实验问题进行专项验证\n" << std::endl;
    
    std::cout << "请选择验证模式:" << std::endl;
    std::cout << "1. 推荐序列验证 (全面但耗时)" << std::endl;
    std::cout << "2. 耦合问题专项验证 (针对您的问题)" << std::endl;
    std::cout << "3. 快速检查 (30秒)" << std::endl;
    std::cout << "\n输入选择 (1/2/3): ";
    
    int choice;
    std::cin >> choice;
    
    try {
        switch (choice) {
            case 1:
                ValidationExecutor::executeRecommendedSequence();
                break;
            case 2:
                ValidationExecutor::executeProblemFocusedValidation();
                break;
            case 3:
                std::cout << "\n=== 快速检查模式 ===" << std::endl;
                if (ValidationFramework::quickCheck()) {
                    std::cout << "✅ 快速检查通过" << std::endl;
                } else {
                    std::cout << "❌ 快速检查失败 - 建议运行完整验证" << std::endl;
                }
                break;
            default:
                std::cout << "无效选择，运行推荐序列验证" << std::endl;
                ValidationExecutor::executeRecommendedSequence();
        }
    } catch (const std::exception& e) {
        std::cerr << "验证过程异常: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}