#include "validation_framework.h"

#include <iostream>

int main() {
    std::cout << "=== Quick DKT diagnostic test ===" << std::endl;

    std::cout << "[1/3] Basic module checks..." << std::endl;
    const bool basic_ok = ValidationFramework::quickCheck();
    if (!basic_ok) {
        std::cout << "FAIL: basic modules are not healthy." << std::endl;
        return 1;
    }

    std::cout << "[2/3] Coupling checks..." << std::endl;
    const bool coupling_ok = ValidationFramework::Coupling::testAll();
    if (!coupling_ok) {
        std::cout << "FAIL: coupling checks failed." << std::endl;
        return 1;
    }

    std::cout << "[3/3] DKT quick check..." << std::endl;
    const bool dkt_ok = ValidationFramework::DKT::quickTest();
    if (!dkt_ok) {
        std::cout << "FAIL: DKT quick check failed." << std::endl;
        std::cout << "Hint: inspect LBM->IBM->DEM data exchange order and synchronization." << std::endl;
        return 1;
    }

    std::cout << "PASS: quick DKT diagnostic test passed." << std::endl;
    return 0;
}
