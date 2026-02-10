# LBM+IB+DEM 验证框架使用指南

## 概述

这个验证框架旨在帮助您提前发现和解决LBM+IB+DEM模块中的问题，确保DKT实验能够顺利进行。框架采用分层验证策略，从单个模块到两两耦合，再到完整的DKT实验验证。

## 快速开始

### 1. 快速检查（日常开发）

```cpp
#include "validation_tests/validation_framework.h"

// 快速验证 - 只需要几秒钟
bool ok = ValidationFramework::quickCheck();
if (ok) {
    std::cout << "基础功能正常，可以继续开发" << std::endl;
} else {
    std::cout << "发现基础问题，需要检查" << std::endl;
}
```

### 2. 标准验证（功能验证）

```cpp
#include "validation_tests/validation_framework.h"

// 标准验证 - 完整的系统检查
auto report = ValidationFramework::standardValidation();
report.printSummary();

if (report.allTestsPassed()) {
    std::cout << "系统验证通过，可以进行DKT实验" << std::endl;
} else {
    std::cout << "系统存在问题，请查看建议" << std::endl;
}
```

### 3. 完整验证（发布前）

```cpp
#include "validation_tests/validation_framework.h"

// 完整验证 - 包含所有测试用例和收敛性分析
auto report = ValidationFramework::fullValidation();
report.printSummary();
```

## 分层验证策略

### 第一层：单个模块验证

```cpp
// 测试LBM模块
bool lbm_ok = ValidationFramework::SingleModule::testLBM();

// 测试IBM模块  
bool ibm_ok = ValidationFramework::SingleModule::testIBM();

// 测试DEM模块
bool dem_ok = ValidationFramework::SingleModule::testDEM();

// 测试所有单个模块
bool all_ok = ValidationFramework::SingleModule::testAll();
```

#### LBM验证内容：
- **质量守恒**：检查流体密度是否保持恒定
- **泊肃叶流**：验证管道流的解析解
- **收敛性**：不同分辨率下的收敛性分析

#### IBM验证内容：
- **静态圆柱绕流**：验证阻力系数
- **运动精度**：检查标记点运动学
- **力计算**：验证浸没边界力的计算

#### DEM验证内容：
- **二体碰撞**：验证碰撞物理
- **颗粒堆积**：检查静态堆积稳定性
- **能量守恒**：验证系统总能量

### 第二层：两两耦合验证

```cpp
// 测试LBM+IBM耦合
bool lbm_ibm_ok = ValidationFramework::Coupling::testLBM_IBM();

// 测试LBM+DEM耦合
bool lbm_dem_ok = ValidationFramework::Coupling::testLBM_DEM();

// 测试IBM+DEM耦合
bool ibm_dem_ok = ValidationFramework::Coupling::testIBM_DEM();

// 测试所有耦合
bool all_coupling_ok = ValidationFramework::Coupling::testAll();
```

#### LBM+IBM耦合验证：
- **静态IBM流场发展**：验证流场与浸没边界的相互作用
- **运动IBM力计算**：检查移动边界的力传递
- **质量守恒**：确保耦合后的质量守恒

#### LBM+DEM耦合验证：
- **颗粒沉降**：验证颗粒在流体中的沉降
- **多颗粒碰撞**：检查流体中颗粒碰撞
- **动量传递**：验证流体与颗粒的动量交换

#### IBM+DEM耦合验证：
- **IBM-DEM碰撞**：处理浸没边界与颗粒的相互作用
- **移动IBM在DEM颗粒上**：验证复杂几何的相互作用

### 第三层：DKT实验验证

```cpp
// 快速DKT测试（推荐首次使用）
bool dkt_quick = ValidationFramework::DKT::quickTest();

// 完整DKT测试
bool dkt_full = ValidationFramework::DKT::fullTest();

// 自定义参数DKT测试
ValidationTests::DKT_ValidationStrategy::DKTConfig config;
config.sphere1_radius = 0.015f;
config.sphere2_radius = 0.015f;
config.flow_velocity = 0.1f;
config.resolution = 64;

bool dkt_custom = ValidationFramework::DKT::customTest(config);
```

#### DKT验证内容：
- **三阶段验证**：drafting、kissing、tumbling
- **文献对比**：与已发表的DKT结果对比
- **收敛性分析**：不同分辨率下的结果稳定性
- **物理合理性**：检查能量守恒、动量守恒

## 调试工具使用

### 性能监测

```cpp
#include "validation_tests/debug_tools.h"

DebugTools::PerformanceMetrics metrics;

// 开始计时
DebugTools::Timer timer("我的测试", &metrics);

// ... 你的代码 ...

// 自动结束计时并记录

// 查看性能数据
std::cout << "LBM时间: " << metrics.lbm_time << "秒" << std::endl;
std::cout << "内存使用: " << metrics.memory_usage << "字节" << std::endl;
```

### 场数据验证

```cpp
// 检查NaN
float* velocity_field = /* 你的速度场 */;
size_t field_size = /* 场大小 */;

bool has_nan = DebugTools::FieldValidator::checkNaN(
    velocity_field, field_size, "velocity_field"
);

// 检查无穷大
bool has_inf = DebugTools::FieldValidator::checkInf(
    velocity_field, field_size, "velocity_field"
);

// 检查数值范围
bool in_range = DebugTools::FieldValidator::checkRange(
    velocity_field, field_size, -10.0f, 10.0f, "velocity_field"
);
```

### DKT专用调试

```cpp
#include "validation_tests/debug_tools.h"

// 记录DKT状态
ValidationTests::DKT_DebugTools::DKT_State state;
state.sphere1_pos[0] = x1; state.sphere1_pos[1] = y1; state.sphere1_pos[2] = z1;
state.sphere2_pos[0] = x2; state.sphere2_pos[1] = y2; state.sphere2_pos[2] = z2;
state.separation_distance = sqrt(pow(x2-x1,2) + pow(y2-y1,2) + pow(z2-z1,2));

// 打印状态
state.print();

// 检测异常
ValidationTests::DKT_DebugTools::DKT_History history;
// ... 填充历史数据 ...
bool anomalies = ValidationTests::DKT_DebugTools::detectAnomalies(history);
```

## 常见问题解决

### 1. 单个模块测试失败

**LBM测试失败：**
- 检查边界条件实现
- 验证松弛时间参数
- 确认格子声速设置

**IBM测试失败：**
- 检查标记点分布密度
- 验证力分布函数
- 确认边界条件处理

**DEM测试失败：**
- 检查碰撞模型参数
- 验证时间步长设置
- 确认接触力模型

### 2. 耦合测试失败

**LBM+IBM耦合失败：**
- 检查场数据交换时机
- 验证力-速度耦合逻辑
- 确认插值函数实现

**LBM+DEM耦合失败：**
- 检查动量交换算法
- 验证颗粒-流体相互作用
- 确认双向耦合顺序

**IBM+DEM耦合失败：**
- 检查复杂边界处理
- 验证接触检测算法
- 确认力传递机制

### 3. DKT实验失败

**drafting阶段失败：**
- 检查初始条件设置
- 验证流体-颗粒相互作用
- 确认阻力计算精度

**kissing阶段失败：**
- 检查近距离相互作用
- 验证润滑力模型
- 确认碰撞处理逻辑

**tumbling阶段失败：**
- 检查旋转动力学
- 验证力矩计算
- 确认稳定性处理

## 最佳实践

### 1. 开发流程建议

```cpp
// 1. 每次修改后运行快速验证
if (!ValidationFramework::quickCheck()) {
    std::cout << "修改引入了基础问题" << std::endl;
    return;
}

// 2. 功能完成后运行标准验证
auto report = ValidationFramework::standardValidation();
if (!report.allTestsPassed()) {
    std::cout << "功能实现存在问题" << std::endl;
    // 查看具体失败项
    report.printSummary();
    return;
}

// 3. 准备DKT实验前运行完整验证
auto full_report = ValidationFramework::fullValidation();
if (full_report.allTestsPassed()) {
    std::cout << "系统准备好进行DKT实验" << std::endl;
} else {
    std::cout << "系统尚未准备好" << std::endl;
}
```

### 2. 调试建议

1. **分层调试**：从单个模块开始，逐步到耦合，最后到完整系统
2. **对比验证**：与文献结果或解析解对比
3. **参数扫描**：测试不同参数组合下的稳定性
4. **实时监控**：使用调试工具实时监测系统状态

### 3. 性能优化

```cpp
// 使用性能监测找到瓶颈
DebugTools::PerformanceMetrics metrics;
{
    DebugTools::Timer timer("我的优化目标", &metrics);
    // 你的代码
}

// 分析时间分布
std::cout << "LBM: " << (metrics.lbm_time / metrics.total_time * 100) << "%" << std::endl;
std::cout << "IBM: " << (metrics.ibm_time / metrics.total_time * 100) << "%" << std::endl;
std::cout << "DEM: " << (metrics.dem_time / metrics.total_time * 100) << "%" << std::endl;
```

## 总结

这个验证框架提供了系统性的方法来验证LBM+IB+DEM系统。通过分层验证策略，您可以：

1. **提前发现问题**：在DKT实验之前发现并修复问题
2. **定位问题根源**：通过分层测试快速定位问题所在
3. **确保系统稳定**：通过全面的验证确保系统可靠性
4. **优化性能**：通过性能监测找到并优化性能瓶颈

建议按照快速验证→标准验证→完整验证的顺序使用，确保每次DKT实验前系统都处于最佳状态。