#include "particles/cuda_support.hpp"

#include <cuda_runtime_api.h>

namespace particles::cuda {

int device_count() noexcept {
    int count = 0;
    const cudaError_t result = cudaGetDeviceCount(&count);
    return result == cudaSuccess ? count : 0;
}

DeviceInfo current_device_info() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return {};
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        return {};
    }
    return DeviceInfo{
        .name = properties.name,
        .compute_capability_major = properties.major,
        .compute_capability_minor = properties.minor,
        .total_global_memory_bytes = properties.totalGlobalMem,
    };
}

DeviceMemoryInfo device_memory_info() noexcept {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        return {};
    }
    return DeviceMemoryInfo{
        .free_bytes = free_bytes,
        .total_bytes = total_bytes,
    };
}

}  // namespace particles::cuda
