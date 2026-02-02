#pragma once
#include <cstddef>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * @brief 字段描述符
 */
struct FieldDesc {
  std::string name;         // 字段名
  std::size_t count;        // 元素数量
  std::size_t element_size; // 每个元素的字节大小
  void *device_ptr = nullptr; // 可选的 GPU 设备指针（非拥有）

  std::size_t size_bytes() const { return count * element_size; }
};

/**
 * @brief 字段句柄 - 提供对字段数据的访问
 */
class FieldHandle {
public:
  FieldHandle() = default;

  FieldHandle(const FieldDesc &desc, std::shared_ptr<std::vector<char>> data)
      : desc_(desc), data_(std::move(data)), device_ptr_(desc.device_ptr) {}

  const FieldDesc &desc() const { return desc_; }
  const std::string &name() const { return desc_.name; }
  std::size_t count() const { return desc_.count; }
  std::size_t element_size() const { return desc_.element_size; }
  std::size_t size_bytes() const { return desc_.size_bytes(); }

  // 主机端数据访问
  const void *data() const { return data_ ? data_->data() : nullptr; }
  void *data() { return data_ ? data_->data() : nullptr; }

  template <typename T> const T *as() const {
    return data_ ? reinterpret_cast<const T *>(data_->data()) : nullptr;
  }

  template <typename T> T *as() {
    return data_ ? reinterpret_cast<T *>(data_->data()) : nullptr;
  }

  // 设备端指针访问
  void *device_data() { return device_ptr_; }
  const void *device_data() const { return device_ptr_; }

  template <typename T> T *device_as() {
    return device_ptr_ ? reinterpret_cast<T *>(device_ptr_) : nullptr;
  }

  template <typename T> const T *device_as() const {
    return device_ptr_ ? reinterpret_cast<const T *>(device_ptr_) : nullptr;
  }

  bool has_device() const { return device_ptr_ != nullptr; }
  void set_device_ptr(void *ptr) { device_ptr_ = ptr; }

  bool valid() const { return data_ != nullptr && !desc_.name.empty(); }

private:
  FieldDesc desc_{};
  std::shared_ptr<std::vector<char>> data_;
  void *device_ptr_ = nullptr;
};

/**
 * @brief 字段存储管理器
 */
class FieldStore {
public:
  virtual ~FieldStore() = default;

  virtual FieldHandle create(const FieldDesc &desc) {
    auto data = std::make_shared<std::vector<char>>(desc.size_bytes(), 0);
    auto handle = FieldHandle(desc, data);
    fields_[desc.name] = handle;
    return handle;
  }

  virtual FieldHandle get(const std::string &name) const {
    auto it = fields_.find(name);
    if (it == fields_.end()) {
      throw std::runtime_error("Field not found: " + name);
    }
    return it->second;
  }

  bool exists(const std::string &name) const {
    return fields_.find(name) != fields_.end();
  }

  void setDevicePtr(const std::string &name, void *ptr) {
    auto it = fields_.find(name);
    if (it != fields_.end()) {
      it->second.set_device_ptr(ptr);
    }
  }

  std::vector<std::string> names() const {
    std::vector<std::string> result;
    result.reserve(fields_.size());
    for (auto it = fields_.begin(); it != fields_.end(); ++it) {
      result.push_back(it->first);
    }
    return result;
  }

protected:
  std::unordered_map<std::string, FieldHandle> fields_;
};
