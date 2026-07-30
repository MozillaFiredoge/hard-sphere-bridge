#include "particles/cuda_support.hpp"

#include <cuda_runtime_api.h>

namespace particles::cuda {

int device_count() noexcept {
    int count = 0;
    const cudaError_t result = cudaGetDeviceCount(&count);
    return result == cudaSuccess ? count : 0;
}

}  // namespace particles::cuda

