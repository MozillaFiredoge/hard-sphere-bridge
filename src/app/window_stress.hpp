#pragma once

#include <filesystem>

namespace particles::app {

int run_parallel_window_stress(
    const std::filesystem::path& config_path,
    const std::filesystem::path& output_directory
);

int run_adaptive_parallel_window_stress(
    const std::filesystem::path& config_path,
    const std::filesystem::path& output_directory
);

}  // namespace particles::app
