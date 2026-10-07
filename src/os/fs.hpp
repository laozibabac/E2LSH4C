#pragma once
// Portable directory helpers (create + unique temp dir).

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace e2lsh {
namespace os {

inline void createDir(const std::string& path) {
  std::error_code ec;
  std::filesystem::create_directories(path, ec);
  if (ec) {
    throw std::runtime_error("create_directories failed: " + path + " : " + ec.message());
  }
}

inline std::string makeTempDir(const std::string& prefix = "e2lsh_") {
  const auto base = std::filesystem::temp_directory_path();
  const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (int i = 0; i < 10000; ++i) {
    const auto name = prefix + std::to_string(static_cast<std::uint64_t>(now) + static_cast<std::uint64_t>(i));
    const auto p = base / name;
    std::error_code ec;
    if (std::filesystem::create_directory(p, ec) && !ec) return p.string();
  }
  throw std::runtime_error("makeTempDir failed under " + base.string());
}

}  // namespace os
}  // namespace e2lsh
