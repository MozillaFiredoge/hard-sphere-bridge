#include "particles/diagnostics.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <system_error>

namespace particles {

std::size_t PhaseSpaceHistogram::flat_index(
    std::size_t x_bin,
    std::size_t velocity_x_bin
) const {
    if (
        x_bin >= x_bin_count ||
        velocity_x_bin >= velocity_x_bin_count
    ) {
        throw std::out_of_range(
            "Phase-space histogram bin index is out of range."
        );
    }
    return x_bin * velocity_x_bin_count + velocity_x_bin;
}

std::uint64_t PhaseSpaceHistogram::count(
    std::size_t x_bin,
    std::size_t velocity_x_bin
) const {
    return counts[flat_index(x_bin, velocity_x_bin)];
}

std::uint64_t PhaseSpaceHistogram::samples() const noexcept {
    return velocity_underflow +
           velocity_overflow +
           std::accumulate(
               counts.begin(),
               counts.end(),
               std::uint64_t{0}
           );
}

double PhaseSpaceHistogram::probability_density(
    std::size_t x_bin,
    std::size_t velocity_x_bin
) const {
    const std::uint64_t total_samples = samples();
    if (
        total_samples == 0 ||
        x_bin_width <= 0.0 ||
        velocity_x_bin_width <= 0.0
    ) {
        return 0.0;
    }
    return static_cast<double>(count(x_bin, velocity_x_bin)) /
           (
               static_cast<double>(total_samples) *
               x_bin_width *
               velocity_x_bin_width
           );
}

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

MultiscaleDiagnosticFrame measure_multiscale(
    const std::vector<Particle>& particles,
    double time,
    double particle_mass,
    double boltzmann_constant,
    const DomainConfig& domain,
    const DiagnosticsConfig& config
) {
    if (particles.empty()) {
        throw std::invalid_argument(
            "Multiscale diagnostics require at least one particle."
        );
    }
    if (!std::isfinite(time) || time < 0.0) {
        throw std::invalid_argument(
            "Multiscale diagnostic time must be finite and non-negative."
        );
    }
    if (!std::isfinite(particle_mass) || particle_mass <= 0.0) {
        throw std::invalid_argument(
            "Multiscale particle mass must be finite and positive."
        );
    }
    if (
        !std::isfinite(boltzmann_constant) ||
        boltzmann_constant <= 0.0
    ) {
        throw std::invalid_argument(
            "Multiscale Boltzmann constant must be finite and positive."
        );
    }
    if (
        !std::isfinite(domain.extent.x) ||
        !std::isfinite(domain.extent.y) ||
        !std::isfinite(domain.extent.z) ||
        domain.extent.x <= 0.0 ||
        domain.extent.y <= 0.0 ||
        domain.extent.z <= 0.0
    ) {
        throw std::invalid_argument(
            "Multiscale domain extents must be finite and positive."
        );
    }
    if (
        config.microscopic_particle_samples == 0 ||
        config.phase_space_x_bins == 0 ||
        config.phase_space_vx_bins == 0 ||
        config.macroscopic_x_slabs == 0
    ) {
        throw std::invalid_argument(
            "Multiscale diagnostic bin and sample counts must be positive."
        );
    }
    if (
        !std::isfinite(config.phase_space_velocity_x_min) ||
        !std::isfinite(config.phase_space_velocity_x_max) ||
        config.phase_space_velocity_x_max <=
            config.phase_space_velocity_x_min
    ) {
        throw std::invalid_argument(
            "Multiscale phase-space velocity range is invalid."
        );
    }
    if (
        config.phase_space_x_bins >
        std::numeric_limits<std::size_t>::max() /
            config.phase_space_vx_bins
    ) {
        throw std::overflow_error(
            "Multiscale phase-space histogram is too large."
        );
    }

    MultiscaleDiagnosticFrame frame{};
    frame.time = time;
    frame.particle_count = particles.size();
    frame.phase_space = PhaseSpaceHistogram{
        .x_minimum = 0.0,
        .x_maximum = domain.extent.x,
        .velocity_x_minimum =
            config.phase_space_velocity_x_min,
        .velocity_x_maximum =
            config.phase_space_velocity_x_max,
        .x_bin_width =
            domain.extent.x /
            static_cast<double>(config.phase_space_x_bins),
        .velocity_x_bin_width =
            (
                config.phase_space_velocity_x_max -
                config.phase_space_velocity_x_min
            ) /
            static_cast<double>(config.phase_space_vx_bins),
        .x_bin_count = config.phase_space_x_bins,
        .velocity_x_bin_count = config.phase_space_vx_bins,
        .counts = std::vector<std::uint64_t>(
            config.phase_space_x_bins *
                config.phase_space_vx_bins,
            std::uint64_t{0}
        ),
    };

    const double slab_width =
        domain.extent.x /
        static_cast<double>(config.macroscopic_x_slabs);
    const double slab_volume =
        slab_width * domain.extent.y * domain.extent.z;
    frame.macroscopic_slabs.resize(
        config.macroscopic_x_slabs
    );
    for (std::size_t slab_index = 0;
         slab_index < frame.macroscopic_slabs.size();
         ++slab_index) {
        MacroscopicSlabState& slab =
            frame.macroscopic_slabs[slab_index];
        slab.slab_index = slab_index;
        slab.x_lower =
            static_cast<double>(slab_index) * slab_width;
        slab.x_upper = slab.x_lower + slab_width;
        slab.x_center = slab.x_lower + 0.5 * slab_width;
        slab.volume = slab_volume;
    }

    for (const Particle& particle : particles) {
        if (
            !std::isfinite(particle.position.x) ||
            !std::isfinite(particle.position.y) ||
            !std::isfinite(particle.position.z) ||
            particle.position.x < 0.0 ||
            particle.position.x > domain.extent.x ||
            particle.position.y < 0.0 ||
            particle.position.y > domain.extent.y ||
            particle.position.z < 0.0 ||
            particle.position.z > domain.extent.z ||
            !std::isfinite(particle.velocity.x) ||
            !std::isfinite(particle.velocity.y) ||
            !std::isfinite(particle.velocity.z)
        ) {
            throw std::invalid_argument(
                "Multiscale diagnostics received an invalid particle."
            );
        }

        frame.momentum += particle_mass * particle.velocity;
        const double particle_kinetic_energy =
            0.5 * particle_mass * squared_norm(particle.velocity);
        frame.kinetic_energy += particle_kinetic_energy;

        const std::size_t slab_index = std::min(
            static_cast<std::size_t>(
                particle.position.x / slab_width
            ),
            frame.macroscopic_slabs.size() - 1U
        );
        MacroscopicSlabState& slab =
            frame.macroscopic_slabs[slab_index];
        ++slab.particle_count;
        slab.momentum += particle_mass * particle.velocity;
        slab.kinetic_energy += particle_kinetic_energy;

        const std::size_t x_bin = std::min(
            static_cast<std::size_t>(
                particle.position.x /
                frame.phase_space.x_bin_width
            ),
            frame.phase_space.x_bin_count - 1U
        );
        if (
            particle.velocity.x <
            frame.phase_space.velocity_x_minimum
        ) {
            ++frame.phase_space.velocity_underflow;
        } else if (
            particle.velocity.x >=
            frame.phase_space.velocity_x_maximum
        ) {
            ++frame.phase_space.velocity_overflow;
        } else {
            const std::size_t velocity_bin = std::min(
                static_cast<std::size_t>(
                    (
                        particle.velocity.x -
                        frame.phase_space.velocity_x_minimum
                    ) /
                    frame.phase_space.velocity_x_bin_width
                ),
                frame.phase_space.velocity_x_bin_count - 1U
            );
            ++frame.phase_space.counts[
                frame.phase_space.flat_index(
                    x_bin,
                    velocity_bin
                )
            ];
        }
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

    for (MacroscopicSlabState& slab : frame.macroscopic_slabs) {
        if (slab.particle_count != 0) {
            const double slab_mass =
                particle_mass *
                static_cast<double>(slab.particle_count);
            slab.bulk_velocity = slab.momentum / slab_mass;
            slab.thermal_energy = std::max(
                0.0,
                slab.kinetic_energy -
                    0.5 *
                    slab_mass *
                    squared_norm(slab.bulk_velocity)
            );
            slab.temperature =
                2.0 * slab.thermal_energy /
                (
                    3.0 *
                    boltzmann_constant *
                    static_cast<double>(slab.particle_count)
                );
        }
        slab.number_density =
            static_cast<double>(slab.particle_count) /
            slab.volume;
        slab.mass_density =
            particle_mass * slab.number_density;
        slab.pressure =
            slab.number_density *
            boltzmann_constant *
            slab.temperature;

        frame.reconstructed_particle_count +=
            slab.particle_count;
        frame.reconstructed_momentum += slab.momentum;
        frame.reconstructed_kinetic_energy +=
            slab.kinetic_energy;
    }
    frame.momentum_reconstruction_error = norm(
        frame.reconstructed_momentum - frame.momentum
    );
    frame.kinetic_energy_reconstruction_error = std::abs(
        frame.reconstructed_kinetic_energy -
        frame.kinetic_energy
    );

    const std::size_t microscopic_count = std::min(
        config.microscopic_particle_samples,
        particles.size()
    );
    frame.microscopic_particles.reserve(microscopic_count);
    const std::size_t quotient =
        particles.size() / microscopic_count;
    const std::size_t remainder =
        particles.size() % microscopic_count;
    for (std::size_t sample_index = 0;
         sample_index < microscopic_count;
         ++sample_index) {
        const std::size_t particle_index =
            sample_index * quotient +
            std::min(sample_index, remainder);
        const Particle& particle = particles[particle_index];
        frame.microscopic_particles.push_back(
            MicroscopicParticleSample{
                .particle_index = particle_index,
                .position = particle.position,
                .velocity = particle.velocity,
                .speed = norm(particle.velocity),
            }
        );
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

MultiscaleCsvWriter::MultiscaleCsvWriter(
    const std::filesystem::path& output_directory
) {
    std::error_code error;
    std::filesystem::create_directories(output_directory, error);
    if (error) {
        throw std::runtime_error(
            "Cannot create multiscale output directory " +
            output_directory.string() + ": " + error.message()
        );
    }

    summary_path_ = output_directory / "summary.csv";
    microscopic_path_ = output_directory / "microscopic.csv";
    phase_space_path_ = output_directory / "phase_space.csv";
    macroscopic_path_ = output_directory / "macroscopic.csv";
    summary_.open(summary_path_);
    microscopic_.open(microscopic_path_);
    phase_space_.open(phase_space_path_);
    macroscopic_.open(macroscopic_path_);
    if (
        !summary_ ||
        !microscopic_ ||
        !phase_space_ ||
        !macroscopic_
    ) {
        throw std::runtime_error(
            "Cannot open one or more multiscale diagnostic outputs."
        );
    }

    summary_
        << std::setprecision(17)
        << "frame,time,particle_count,microscopic_samples,"
           "phase_space_samples,phase_space_underflow,"
           "phase_space_overflow,macroscopic_particles,"
           "momentum_x,momentum_y,momentum_z,"
           "bulk_velocity_x,bulk_velocity_y,bulk_velocity_z,"
           "kinetic_energy,thermal_energy,temperature,"
           "momentum_reconstruction_error,"
           "momentum_reconstruction_error_per_particle,"
           "kinetic_energy_reconstruction_error,"
           "relative_kinetic_energy_reconstruction_error\n";
    microscopic_
        << std::setprecision(17)
        << "frame,time,particle_index,position_x,position_y,position_z,"
           "velocity_x,velocity_y,velocity_z,speed\n";
    phase_space_
        << std::setprecision(17)
        << "frame,time,x_bin,velocity_x_bin,x_lower,x_upper,"
           "velocity_x_lower,velocity_x_upper,count,"
           "probability_density,velocity_underflow,velocity_overflow\n";
    macroscopic_
        << std::setprecision(17)
        << "frame,time,slab,x_lower,x_upper,x_center,volume,"
           "particle_count,number_density,mass_density,"
           "momentum_x,momentum_y,momentum_z,"
           "bulk_velocity_x,bulk_velocity_y,bulk_velocity_z,"
           "kinetic_energy,thermal_energy,temperature,pressure\n";
}

void MultiscaleCsvWriter::write(
    const MultiscaleDiagnosticFrame& frame
) {
    const std::size_t current_frame = frame_index_++;
    summary_
        << current_frame << ','
        << frame.time << ','
        << frame.particle_count << ','
        << frame.microscopic_particles.size() << ','
        << frame.phase_space.samples() << ','
        << frame.phase_space.velocity_underflow << ','
        << frame.phase_space.velocity_overflow << ','
        << frame.reconstructed_particle_count << ','
        << frame.momentum.x << ','
        << frame.momentum.y << ','
        << frame.momentum.z << ','
        << frame.bulk_velocity.x << ','
        << frame.bulk_velocity.y << ','
        << frame.bulk_velocity.z << ','
        << frame.kinetic_energy << ','
        << frame.thermal_energy << ','
        << frame.temperature << ','
        << frame.momentum_reconstruction_error << ','
        << frame.momentum_reconstruction_error /
               static_cast<double>(frame.particle_count)
        << ','
        << frame.kinetic_energy_reconstruction_error << ','
        << (
               frame.kinetic_energy == 0.0
                   ? 0.0
                   : frame.kinetic_energy_reconstruction_error /
                         std::abs(frame.kinetic_energy)
           ) << '\n';

    for (const MicroscopicParticleSample& sample :
         frame.microscopic_particles) {
        microscopic_
            << current_frame << ','
            << frame.time << ','
            << sample.particle_index << ','
            << sample.position.x << ','
            << sample.position.y << ','
            << sample.position.z << ','
            << sample.velocity.x << ','
            << sample.velocity.y << ','
            << sample.velocity.z << ','
            << sample.speed << '\n';
    }

    const std::uint64_t phase_space_samples =
        frame.phase_space.samples();
    const double phase_space_normalization =
        static_cast<double>(phase_space_samples) *
        frame.phase_space.x_bin_width *
        frame.phase_space.velocity_x_bin_width;
    for (std::size_t x_bin = 0;
         x_bin < frame.phase_space.x_bin_count;
         ++x_bin) {
        const double x_lower =
            frame.phase_space.x_minimum +
            static_cast<double>(x_bin) *
                frame.phase_space.x_bin_width;
        for (std::size_t velocity_bin = 0;
             velocity_bin <
                 frame.phase_space.velocity_x_bin_count;
             ++velocity_bin) {
            const double velocity_lower =
                frame.phase_space.velocity_x_minimum +
                static_cast<double>(velocity_bin) *
                    frame.phase_space.velocity_x_bin_width;
            phase_space_
                << current_frame << ','
                << frame.time << ','
                << x_bin << ','
                << velocity_bin << ','
                << x_lower << ','
                << x_lower + frame.phase_space.x_bin_width << ','
                << velocity_lower << ','
                << velocity_lower +
                       frame.phase_space.velocity_x_bin_width
                << ','
                << frame.phase_space.count(x_bin, velocity_bin) << ','
                << (
                       phase_space_normalization > 0.0
                           ? static_cast<double>(
                                 frame.phase_space.count(
                                     x_bin,
                                     velocity_bin
                                 )
                             ) /
                                 phase_space_normalization
                           : 0.0
                   ) << ','
                << frame.phase_space.velocity_underflow << ','
                << frame.phase_space.velocity_overflow << '\n';
        }
    }

    for (const MacroscopicSlabState& slab :
         frame.macroscopic_slabs) {
        macroscopic_
            << current_frame << ','
            << frame.time << ','
            << slab.slab_index << ','
            << slab.x_lower << ','
            << slab.x_upper << ','
            << slab.x_center << ','
            << slab.volume << ','
            << slab.particle_count << ','
            << slab.number_density << ','
            << slab.mass_density << ','
            << slab.momentum.x << ','
            << slab.momentum.y << ','
            << slab.momentum.z << ','
            << slab.bulk_velocity.x << ','
            << slab.bulk_velocity.y << ','
            << slab.bulk_velocity.z << ','
            << slab.kinetic_energy << ','
            << slab.thermal_energy << ','
            << slab.temperature << ','
            << slab.pressure << '\n';
    }

    summary_.flush();
    microscopic_.flush();
    phase_space_.flush();
    macroscopic_.flush();
    if (!summary_ || !microscopic_ || !phase_space_ || !macroscopic_) {
        throw std::runtime_error(
            "Failed while writing multiscale diagnostics."
        );
    }
}

}  // namespace particles
