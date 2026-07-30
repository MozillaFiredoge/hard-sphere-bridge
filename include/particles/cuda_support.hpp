#pragma once

namespace particles::cuda {

// Returns zero when no CUDA-capable device or driver is available.
[[nodiscard]] int device_count() noexcept;

}  // namespace particles::cuda

