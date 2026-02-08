#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include <toml++/toml.hpp>

class ConfigManager {
public:
  explicit ConfigManager(const std::string& config_path)
      : table_(toml::parse_file(config_path)) {}

  int getInt(const std::string& key, int default_value) const {
    if (auto v = table_[key].value<std::int64_t>()) {
      return static_cast<int>(*v);
    }
    return default_value;
  }

  double getFloat(const std::string& key, double default_value) const {
    if (auto v = table_[key].value<double>()) {
      return *v;
    }
    if (auto i = table_[key].value<std::int64_t>()) {
      return static_cast<double>(*i);
    }
    return default_value;
  }

  bool getBool(const std::string& key, bool default_value) const {
    if (auto v = table_[key].value<bool>()) {
      return *v;
    }
    return default_value;
  }

private:
  toml::table table_;
};
