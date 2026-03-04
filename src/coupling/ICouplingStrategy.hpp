#pragma once
#include "../core/StepContext.hpp"

namespace coupling {

class ICouplingStrategy {
public:
    virtual ~ICouplingStrategy() = default;
    virtual void execute(StepContext& ctx) = 0;
};

} // namespace coupling
