#pragma once

#include "single_module_tests.h"
#include "coupling_tests.h"
#include "dkt_validation.h"

namespace ValidationFramework {

inline bool quickCheck() {
    return ValidationTests::LBMValidationTests::testMassConservation() &&
           ValidationTests::IBMValidationTests::testKinematicsAccuracy() &&
           ValidationTests::DEMValidationTests::testBinaryCollision();
}

namespace SingleModule {
inline bool testLBM() {
    return ValidationTests::LBMValidationTests::testMassConservation() &&
           ValidationTests::LBMValidationTests::testPoiseuilleFlow();
}

inline bool testIBM() {
    return ValidationTests::IBMValidationTests::testStaticCylinderFlow() &&
           ValidationTests::IBMValidationTests::testKinematicsAccuracy();
}

inline bool testDEM() {
    return ValidationTests::DEMValidationTests::testBinaryCollision() &&
           ValidationTests::DEMValidationTests::testParticlePacking();
}

inline bool testAll() {
    return ValidationTests::ValidationTestSuite::runAllTests();
}
} // namespace SingleModule

namespace Coupling {
inline bool testLBM_IBM() {
    return ValidationTests::LBMIBM_CouplingTests::testStaticIBMFlowDevelopment() &&
           ValidationTests::LBMIBM_CouplingTests::testMovingIBMForceCalculation();
}

inline bool testLBM_DEM() {
    return ValidationTests::LBMDEM_CouplingTests::testParticleSettlingInFluid() &&
           ValidationTests::LBMDEM_CouplingTests::testMultiParticleCollisionInFluid();
}

inline bool testIBM_DEM() {
    return ValidationTests::IBMDEM_CouplingTests::testIBM_DEM_Collision() &&
           ValidationTests::IBMDEM_CouplingTests::testMovingIBMOnDEMParticles();
}

inline bool testAll() {
    return ValidationTests::CouplingValidationSuite::runAllCouplingTests();
}
} // namespace Coupling

namespace DKT {
inline bool quickTest() {
    return ValidationTests::DKT_QuickValidation::quickDKTTest();
}

inline bool fullTest() {
    return ValidationTests::DKT_QuickValidation::fullDKTTest();
}

inline bool customTest(const ValidationTests::DKT_ValidationStrategy::DKTConfig& config) {
    ValidationTests::DKT_ValidationStrategy::DKTMetrics metrics;
    return ValidationTests::DKT_ValidationStrategy::validateDKTExperiment(config, metrics);
}
} // namespace DKT

} // namespace ValidationFramework
