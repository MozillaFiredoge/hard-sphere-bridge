#pragma once

#include <cstddef>
#include <string>

namespace particles::cuda {

struct DeviceInfo {
    std::string name;
    int compute_capability_major{};
    int compute_capability_minor{};
    std::size_t total_global_memory_bytes{};
};

struct DeviceMemoryInfo {
    std::size_t free_bytes{};
    std::size_t total_bytes{};
};

// Returns zero when no CUDA-capable device or driver is available.
[[nodiscard]] int device_count() noexcept;
[[nodiscard]] DeviceInfo current_device_info();
[[nodiscard]] DeviceMemoryInfo device_memory_info() noexcept;

}  // namespace particles::cuda
