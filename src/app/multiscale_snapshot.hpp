#pragma once

#include <filesystem>

namespace particles::app {

int run_multiscale_snapshot(
    const std::filesystem::path& config_path,
    const std::filesystem::path& output_directory
);

int run_shock_tube(
    const std::filesystem::path& config_path,
    const std::filesystem::path& output_directory,
    double end_time_override = -1.0
);

}  // namespace particles::app
