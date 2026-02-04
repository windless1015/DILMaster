/**
 * @file STLReader.h
 * @brief STL file reader supporting both ASCII and binary formats
 * 
 * Adapted from reference code for CPU-only execution.
 * Provides parsing of STL files into triangle mesh representation.
 */
#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <cstdint>
#include "VectorTypes.h"
#include "STLValidation.h"

/**
 * @brief Single triangle from STL mesh
 */
struct STLTriangle {
    float3 normal;
    float3 vertices[3];

    STLTriangle()
        : normal(make_float3(0, 0, 0)) {
        vertices[0] = make_float3(0, 0, 0);
        vertices[1] = make_float3(0, 0, 0);
        vertices[2] = make_float3(0, 0, 0);
    }

    STLTriangle(const float3& n, const float3& v0, const float3& v1, const float3& v2)
        : normal(n) {
        vertices[0] = v0;
        vertices[1] = v1;
        vertices[2] = v2;
    }

    /**
     * @brief Calculate area of triangle using cross product
     */
    float area() const {
        float3 e1, e2;
        e1.x = vertices[1].x - vertices[0].x;
        e1.y = vertices[1].y - vertices[0].y;
        e1.z = vertices[1].z - vertices[0].z;
        e2.x = vertices[2].x - vertices[0].x;
        e2.y = vertices[2].y - vertices[0].y;
        e2.z = vertices[2].z - vertices[0].z;
        
        // Cross product
        float3 cross;
        cross.x = e1.y * e2.z - e1.z * e2.y;
        cross.y = e1.z * e2.x - e1.x * e2.z;
        cross.z = e1.x * e2.y - e1.y * e2.x;
        
        return 0.5f * std::sqrt(cross.x * cross.x + cross.y * cross.y + cross.z * cross.z);
    }

    /**
     * @brief Get centroid of triangle
     */
    float3 center() const {
        float3 c;
        c.x = (vertices[0].x + vertices[1].x + vertices[2].x) / 3.0f;
        c.y = (vertices[0].y + vertices[1].y + vertices[2].y) / 3.0f;
        c.z = (vertices[0].z + vertices[1].z + vertices[2].z) / 3.0f;
        return c;
    }
};

/**
 * @brief STL Mesh container with bounds and transforms
 */
class STLMesh {
public:
    std::vector<STLTriangle> triangles;
    float3 minBound;
    float3 maxBound;
    std::string header;

    STLMesh() : minBound(make_float3(0, 0, 0)), maxBound(make_float3(0, 0, 0)) {}

    /**
     * @brief Calculate axis-aligned bounding box
     */
    void calculateBoundingBox() {
        if (triangles.empty()) return;

        minBound = make_float3(1e10f, 1e10f, 1e10f);
        maxBound = make_float3(-1e10f, -1e10f, -1e10f);

        for (const auto& tri : triangles) {
            for (int i = 0; i < 3; ++i) {
                minBound.x = std::min(minBound.x, tri.vertices[i].x);
                minBound.y = std::min(minBound.y, tri.vertices[i].y);
                minBound.z = std::min(minBound.z, tri.vertices[i].z);
                maxBound.x = std::max(maxBound.x, tri.vertices[i].x);
                maxBound.y = std::max(maxBound.y, tri.vertices[i].y);
                maxBound.z = std::max(maxBound.z, tri.vertices[i].z);
            }
        }
    }

    /**
     * @brief Apply translation and uniform scale to all vertices
     */
    void transform(const float3& translation, float scale) {
        for (auto& tri : triangles) {
            for (int i = 0; i < 3; ++i) {
                tri.vertices[i].x = tri.vertices[i].x * scale + translation.x;
                tri.vertices[i].y = tri.vertices[i].y * scale + translation.y;
                tri.vertices[i].z = tri.vertices[i].z * scale + translation.z;
            }
        }
        calculateBoundingBox();
    }

    /**
     * @brief Translate mesh so its center is at origin
     */
    void centerAtOrigin() {
        float3 c;
        c.x = (minBound.x + maxBound.x) * 0.5f;
        c.y = (minBound.y + maxBound.y) * 0.5f;
        c.z = (minBound.z + maxBound.z) * 0.5f;
        transform(make_float3(-c.x, -c.y, -c.z), 1.0f);
    }

    /**
     * @brief Get dimensions of bounding box
     */
    float3 getSize() const {
        return make_float3(maxBound.x - minBound.x, 
                     maxBound.y - minBound.y, 
                     maxBound.z - minBound.z);
    }

    /**
     * @brief Get center of bounding box
     */
    float3 getCenter() const {
        return make_float3((minBound.x + maxBound.x) * 0.5f,
                     (minBound.y + maxBound.y) * 0.5f,
                     (minBound.z + maxBound.z) * 0.5f);
    }

    /**
     * @brief Print mesh info to console
     */
    void show() const {
        std::cout << "STL Mesh: " << triangles.size() << " triangles" << std::endl;
        std::cout << "  Bounds: (" << minBound.x << ", " << minBound.y << ", " << minBound.z << ") - ("
                  << maxBound.x << ", " << maxBound.y << ", " << maxBound.z << ")" << std::endl;
    }

    /**
     * @brief Validate mesh geometry and orientation (throws on failure)
     */
    void validateOrThrow(
        const STLValidationParams& params = STLValidationParams(),
        STLValidationStats* outStats = nullptr
    ) {
        validateSTLMeshOrThrow(*this, params, outStats);
    }
};

/**
 * @brief Static class for reading STL files
 */
class STLReader {
private:
    /**
     * @brief Detect if file is binary STL by checking header and expected size
     */
    static bool isBinarySTL(std::ifstream& file) {
        file.seekg(0, std::ios::beg);
        char header[80];
        file.read(header, 80);

        std::string headerStr(header, 80);
        if (headerStr.find("solid") == 0) {
            file.seekg(80);
            uint32_t numTriangles;
            file.read(reinterpret_cast<char*>(&numTriangles), sizeof(uint32_t));
            file.seekg(0, std::ios::end);
            size_t fileSize = static_cast<size_t>(file.tellg());
            size_t expectedSize = 80 + 4 + static_cast<size_t>(numTriangles) * 50;
            file.seekg(0, std::ios::beg);
            return (fileSize == expectedSize);
        }
        file.seekg(0, std::ios::beg);
        return true;
    }

    /**
     * @brief Read binary format STL file
     */
    static bool readBinarySTL(std::ifstream& file, STLMesh& mesh) {
        char header[80];
        file.read(header, 80);
        mesh.header = std::string(header, 80);

        uint32_t numTriangles;
        file.read(reinterpret_cast<char*>(&numTriangles), sizeof(uint32_t));
        mesh.triangles.reserve(numTriangles);

        for (uint32_t i = 0; i < numTriangles; ++i) {
            float data[12];
            file.read(reinterpret_cast<char*>(data), 12 * sizeof(float));

            STLTriangle tri;
            tri.normal = make_float3(data[0], data[1], data[2]);
            tri.vertices[0] = make_float3(data[3], data[4], data[5]);
            tri.vertices[1] = make_float3(data[6], data[7], data[8]);
            tri.vertices[2] = make_float3(data[9], data[10], data[11]);
            mesh.triangles.push_back(tri);

            uint16_t attr;
            file.read(reinterpret_cast<char*>(&attr), sizeof(uint16_t));
        }
        return true;
    }

    /**
     * @brief Read ASCII format STL file
     */
    static bool readAsciiSTL(std::ifstream& file, STLMesh& mesh) {
        std::string line, word;
        float3 normal, v0, v1, v2;
        int vertexCount = 0;

        while (std::getline(file, line)) {
            std::istringstream iss(line);
            iss >> word;

            if (word == "solid") {
                std::getline(iss, mesh.header);
            } else if (word == "facet") {
                iss >> word; // "normal"
                iss >> normal.x >> normal.y >> normal.z;
                vertexCount = 0;
            } else if (word == "vertex") {
                float x, y, z;
                iss >> x >> y >> z;
                if (vertexCount == 0) v0 = make_float3(x, y, z);
                else if (vertexCount == 1) v1 = make_float3(x, y, z);
                else if (vertexCount == 2) {
                    v2 = make_float3(x, y, z);
                    mesh.triangles.push_back(STLTriangle(normal, v0, v1, v2));
                }
                vertexCount++;
            }
        }
        return !mesh.triangles.empty();
    }

public:
    /**
     * @brief Read STL file (auto-detects ASCII or binary format)
     * @param filename Path to STL file
     * @param mesh Output mesh container
     * @return true if successful
     */
    static bool readSTL(const std::string& filename, STLMesh& mesh) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "ERROR: Cannot open STL file: " << filename << std::endl;
            return false;
        }

        bool success = false;
        if (isBinarySTL(file)) {
            std::cout << "Reading binary STL: " << filename << std::endl;
            success = readBinarySTL(file, mesh);
        } else {
            std::cout << "Reading ASCII STL: " << filename << std::endl;
            file.close();
            file.open(filename);
            success = readAsciiSTL(file, mesh);
        }
        file.close();

        if (success) {
            mesh.calculateBoundingBox();
            mesh.validateOrThrow();
            mesh.show();
        }
        return success;
    }
};
