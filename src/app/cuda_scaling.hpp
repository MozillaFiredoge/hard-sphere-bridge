#pragma once

#include <filesystem>

namespace particles::app {

int run_cuda_scaling_benchmark(
    const std::filesystem::path& config_path,
    const std::filesystem::path& output_path
);

}  // namespace particles::app
