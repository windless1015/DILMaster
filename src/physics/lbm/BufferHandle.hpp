#pragma once
/**
 * BufferHandle.hpp - Type-safe GPU buffer management
 *
 * Provides abstraction over raw device pointers with type information
 * and layout metadata.
 */

#include <cstddef>
#include <string>

namespace lbm {

/**
 * BufferHandle - Type-safe wrapper for GPU buffers
 *
 * Encapsulates:
 * - Buffer type (density, velocity, flags, etc.)
 * - Data layout (SoA or AoS)
 * - Device pointer
 * - Size information
 */
class BufferHandle {
public:
    // Buffer types used in LBM simulation
    enum class Type {
        DISTRIBUTION,   // f_i distribution functions (FP16)
        DENSITY,        // rho density field
        VELOCITY,       // u velocity field
        FLAGS,          // cell type flags
        PHI,            // fill fraction (free surface)
        MASS,           // mass (free surface)
        MASS_EXCESS,    // excess mass (free surface)
        FORCE,          // external force field
        CUSTOM          // user-defined buffers
    };

    // Data layout for vector fields
    enum class Layout {
        SoA,  // Structure of Arrays: [x0,x1,...,xN, y0,y1,...,yN, z0,z1,...,zN]
        AoS   // Array of Structures: [{x0,y0,z0}, {x1,y1,z1}, ..., {xN,yN,zN}]
    };

    // Default constructor (invalid handle)
    BufferHandle()
        : type_(Type::CUSTOM), layout_(Layout::SoA), devicePtr_(nullptr),
          elementCount_(0), elementSize_(0), name_("") {}

    // Constructor for scalar fields
    BufferHandle(Type type, void* devicePtr, size_t elementCount, size_t elementSize,
                 const std::string& name = "")
        : type_(type), layout_(Layout::SoA), devicePtr_(devicePtr),
          elementCount_(elementCount), elementSize_(elementSize), name_(name) {}

    // Constructor for vector fields with layout
    BufferHandle(Type type, Layout layout, void* devicePtr, size_t elementCount,
                 size_t elementSize, const std::string& name = "")
        : type_(type), layout_(layout), devicePtr_(devicePtr),
          elementCount_(elementCount), elementSize_(elementSize), name_(name) {}

    // Getters
    Type getType() const { return type_; }
    Layout getLayout() const { return layout_; }
    void* getDevicePtr() const { return devicePtr_; }
    size_t getElementCount() const { return elementCount_; }
    size_t getElementSize() const { return elementSize_; }
    size_t getTotalBytes() const { return elementCount_ * elementSize_; }
    const std::string& getName() const { return name_; }

    // Check if handle is valid
    bool isValid() const { return devicePtr_ != nullptr && elementCount_ > 0; }

    // Type checking helpers
    bool isDensity() const { return type_ == Type::DENSITY; }
    bool isVelocity() const { return type_ == Type::VELOCITY; }
    bool isFlags() const { return type_ == Type::FLAGS; }
    bool isPhi() const { return type_ == Type::PHI; }
    bool isMass() const { return type_ == Type::MASS; }

    // Layout checking
    bool isSoA() const { return layout_ == Layout::SoA; }
    bool isAoS() const { return layout_ == Layout::AoS; }

private:
    Type type_;
    Layout layout_;
    void* devicePtr_;
    size_t elementCount_;
    size_t elementSize_;
    std::string name_;
};

} // namespace lbm
