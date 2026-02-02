#pragma once
#include <string>
#include "../core/Config.hpp"
#include "../core/StepContext.hpp"

class ISolver {
public:
    virtual ~ISolver() = default;
    virtual std::string name() const = 0;
    virtual void configure(const ConfigNode& node) = 0;
    virtual void allocate(StepContext& ctx) = 0;
    virtual void initialize(StepContext& ctx) = 0;
    virtual void step(StepContext& ctx) = 0;
    virtual void finalize(StepContext& ctx) = 0;
};