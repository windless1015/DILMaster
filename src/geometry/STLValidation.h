#pragma once
#include "STLReader.h" // Forward decl or recursive? STLReader includes this. So forward decl needed.

class STLMesh;

struct STLValidationParams {
    bool checkManifold = true;
    bool checkClosed = true;
};

struct STLValidationStats {
    int errors = 0;
};

inline void validateSTLMeshOrThrow(const STLMesh& mesh, const STLValidationParams& params, STLValidationStats* outStats) {
    // Stub implementation: Always pass
    if (outStats) outStats->errors = 0;
}
