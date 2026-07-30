#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

#include "particles/config.hpp"
#include "particles/event_driven_simulator.hpp"

namespace particles {

struct VelocityHistogram {
    double minimum{};
    double maximum{};
    double bin_width{};
    std::vector<std::uint64_t> counts;
    std::uint64_t underflow{};
    std::uint64_t overflow{};

    [[nodiscard]] std::uint64_t samples() const noexcept;
    [[nodiscard]] double bin_lower(std::size_t index) const;
    [[nodiscard]] double bin_upper(std::size_t index) const;
    [[nodiscard]] double bin_center(std::size_t index) const;
    [[nodiscard]] double probability_density(std::size_t index) const;
};

struct EquilibriumDiagnosticFrame {
    double time{};
    std::size_t particle_count{};

    Vec3 momentum{};
    Vec3 bulk_velocity{};
    double kinetic_energy{};
    double thermal_energy{};
    double temperature{};

    std::size_t cumulative_particle_collisions{};
    std::size_t interval_particle_collisions{};
    double interval_duration{};
    double total_collision_rate{};
    double per_particle_collision_rate{};

    VelocityHistogram velocity_x;
};

[[nodiscard]] EquilibriumDiagnosticFrame measure_equilibrium(
    const std::vector<Particle>& particles,
    double time,
    double particle_mass,
    double boltzmann_constant,
    const EquilibriumDiagnosticsConfig& histogram_config,
    std::size_t cumulative_particle_collisions,
    std::size_t interval_particle_collisions,
    double interval_duration
);

class EquilibriumCsvWriter {
public:
    explicit EquilibriumCsvWriter(const std::filesystem::path& output_directory);

    void write(const EquilibriumDiagnosticFrame& frame);

    [[nodiscard]] const std::filesystem::path& summary_path() const noexcept {
        return summary_path_;
    }

    [[nodiscard]] const std::filesystem::path& histogram_path() const noexcept {
        return histogram_path_;
    }

private:
    std::filesystem::path summary_path_;
    std::filesystem::path histogram_path_;
    std::ofstream summary_;
    std::ofstream histogram_;
};

}  // namespace particles

