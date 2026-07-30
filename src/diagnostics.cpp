#include "particles/diagnostics.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <stdexcept>
#include <system_error>

namespace particles {

std::uint64_t VelocityHistogram::samples() const noexcept {
    return underflow +
           overflow +
           std::accumulate(
               counts.begin(),
               counts.end(),
               std::uint64_t{0}
           );
}

double VelocityHistogram::bin_lower(std::size_t index) const {
    if (index >= counts.size()) {
        throw std::out_of_range("Velocity histogram bin index is out of range.");
    }
    return minimum + static_cast<double>(index) * bin_width;
}

double VelocityHistogram::bin_upper(std::size_t index) const {
    return bin_lower(index) + bin_width;
}

double VelocityHistogram::bin_center(std::size_t index) const {
    return bin_lower(index) + 0.5 * bin_width;
}

double VelocityHistogram::probability_density(std::size_t index) const {
    if (index >= counts.size()) {
        throw std::out_of_range("Velocity histogram bin index is out of range.");
    }

    const std::uint64_t total_samples = samples();
    if (total_samples == 0 || bin_width <= 0.0) {
        return 0.0;
    }

    return static_cast<double>(counts[index]) /
           (static_cast<double>(total_samples) * bin_width);
}

EquilibriumDiagnosticFrame measure_equilibrium(
    const std::vector<Particle>& particles,
    double time,
    double particle_mass,
    double boltzmann_constant,
    const EquilibriumDiagnosticsConfig& histogram_config,
    std::size_t cumulative_particle_collisions,
    std::size_t interval_particle_collisions,
    double interval_duration
) {
    if (particles.empty()) {
        throw std::invalid_argument(
            "Equilibrium diagnostics require at least one particle."
        );
    }
    if (!std::isfinite(time) || time < 0.0) {
        throw std::invalid_argument(
            "Diagnostic time must be finite and non-negative."
        );
    }
    if (!std::isfinite(particle_mass) || particle_mass <= 0.0) {
        throw std::invalid_argument("Particle mass must be finite and positive.");
    }
    if (!std::isfinite(boltzmann_constant) || boltzmann_constant <= 0.0) {
        throw std::invalid_argument(
            "Boltzmann constant must be finite and positive."
        );
    }
    if (!std::isfinite(interval_duration) || interval_duration < 0.0) {
        throw std::invalid_argument(
            "Diagnostic interval must be finite and non-negative."
        );
    }
    if (
        histogram_config.velocity_x_bins == 0 ||
        !std::isfinite(histogram_config.velocity_x_min) ||
        !std::isfinite(histogram_config.velocity_x_max) ||
        histogram_config.velocity_x_max <= histogram_config.velocity_x_min
    ) {
        throw std::invalid_argument("Velocity histogram configuration is invalid.");
    }

    EquilibriumDiagnosticFrame frame{};
    frame.time = time;
    frame.particle_count = particles.size();
    frame.cumulative_particle_collisions = cumulative_particle_collisions;
    frame.interval_particle_collisions = interval_particle_collisions;
    frame.interval_duration = interval_duration;

    for (const Particle& particle : particles) {
        frame.momentum += particle_mass * particle.velocity;
        frame.kinetic_energy +=
            0.5 * particle_mass * squared_norm(particle.velocity);
    }

    const double total_mass =
        particle_mass * static_cast<double>(particles.size());
    frame.bulk_velocity = frame.momentum / total_mass;

    for (const Particle& particle : particles) {
        frame.thermal_energy +=
            0.5 *
            particle_mass *
            squared_norm(particle.velocity - frame.bulk_velocity);
    }

    frame.temperature =
        2.0 * frame.thermal_energy /
        (
            3.0 *
            boltzmann_constant *
            static_cast<double>(particles.size())
        );

    if (interval_duration > 0.0) {
        frame.total_collision_rate =
            static_cast<double>(interval_particle_collisions) /
            interval_duration;
        frame.per_particle_collision_rate =
            2.0 *
            static_cast<double>(interval_particle_collisions) /
            (
                static_cast<double>(particles.size()) *
                interval_duration
            );
    }

    frame.velocity_x.minimum = histogram_config.velocity_x_min;
    frame.velocity_x.maximum = histogram_config.velocity_x_max;
    frame.velocity_x.bin_width =
        (
            histogram_config.velocity_x_max -
            histogram_config.velocity_x_min
        ) /
        static_cast<double>(histogram_config.velocity_x_bins);
    frame.velocity_x.counts.assign(
        histogram_config.velocity_x_bins,
        std::uint64_t{0}
    );

    for (const Particle& particle : particles) {
        const double peculiar_velocity_x =
            particle.velocity.x - frame.bulk_velocity.x;

        if (peculiar_velocity_x < frame.velocity_x.minimum) {
            ++frame.velocity_x.underflow;
            continue;
        }
        if (peculiar_velocity_x >= frame.velocity_x.maximum) {
            ++frame.velocity_x.overflow;
            continue;
        }

        const double normalized =
            (peculiar_velocity_x - frame.velocity_x.minimum) /
            frame.velocity_x.bin_width;
        const std::size_t index =
            std::min(
                static_cast<std::size_t>(normalized),
                frame.velocity_x.counts.size() - 1
            );
        ++frame.velocity_x.counts[index];
    }

    return frame;
}

EquilibriumCsvWriter::EquilibriumCsvWriter(
    const std::filesystem::path& output_directory
) {
    std::error_code error;
    std::filesystem::create_directories(output_directory, error);
    if (error) {
        throw std::runtime_error(
            "Cannot create diagnostic output directory " +
            output_directory.string() + ": " + error.message()
        );
    }

    summary_path_ = output_directory / "summary.csv";
    histogram_path_ = output_directory / "velocity_histogram.csv";

    summary_.open(summary_path_);
    histogram_.open(histogram_path_);

    if (!summary_) {
        throw std::runtime_error(
            "Cannot open diagnostic summary: " + summary_path_.string()
        );
    }
    if (!histogram_) {
        throw std::runtime_error(
            "Cannot open velocity histogram: " + histogram_path_.string()
        );
    }

    summary_ << std::setprecision(17)
             << "time,particle_count,momentum_x,momentum_y,momentum_z,"
                "bulk_velocity_x,bulk_velocity_y,bulk_velocity_z,"
                "kinetic_energy,thermal_energy,temperature,"
                "cumulative_particle_collisions,interval_particle_collisions,"
                "interval_duration,total_collision_rate,"
                "per_particle_collision_rate\n";

    histogram_ << std::setprecision(17)
               << "time,bin,lower,upper,center,count,probability_density,"
                  "underflow,overflow\n";
}

void EquilibriumCsvWriter::write(const EquilibriumDiagnosticFrame& frame) {
    summary_ << frame.time << ','
             << frame.particle_count << ','
             << frame.momentum.x << ','
             << frame.momentum.y << ','
             << frame.momentum.z << ','
             << frame.bulk_velocity.x << ','
             << frame.bulk_velocity.y << ','
             << frame.bulk_velocity.z << ','
             << frame.kinetic_energy << ','
             << frame.thermal_energy << ','
             << frame.temperature << ','
             << frame.cumulative_particle_collisions << ','
             << frame.interval_particle_collisions << ','
             << frame.interval_duration << ','
             << frame.total_collision_rate << ','
             << frame.per_particle_collision_rate << '\n';

    for (std::size_t index = 0;
         index < frame.velocity_x.counts.size();
         ++index) {
        histogram_ << frame.time << ','
                   << index << ','
                   << frame.velocity_x.bin_lower(index) << ','
                   << frame.velocity_x.bin_upper(index) << ','
                   << frame.velocity_x.bin_center(index) << ','
                   << frame.velocity_x.counts[index] << ','
                   << frame.velocity_x.probability_density(index) << ','
                   << frame.velocity_x.underflow << ','
                   << frame.velocity_x.overflow << '\n';
    }

    if (!summary_ || !histogram_) {
        throw std::runtime_error("Failed while writing equilibrium diagnostics.");
    }
}

}  // namespace particles
