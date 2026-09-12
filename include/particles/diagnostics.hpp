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

struct MicroscopicParticleSample {
    std::size_t particle_index{};
    Vec3 position{};
    Vec3 velocity{};
    double speed{};
};

struct PhaseSpaceHistogram {
    double x_minimum{};
    double x_maximum{};
    double velocity_x_minimum{};
    double velocity_x_maximum{};
    double x_bin_width{};
    double velocity_x_bin_width{};
    std::size_t x_bin_count{};
    std::size_t velocity_x_bin_count{};
    std::vector<std::uint64_t> counts;
    std::uint64_t velocity_underflow{};
    std::uint64_t velocity_overflow{};

    [[nodiscard]] std::size_t flat_index(
        std::size_t x_bin,
        std::size_t velocity_x_bin
    ) const;
    [[nodiscard]] std::uint64_t count(
        std::size_t x_bin,
        std::size_t velocity_x_bin
    ) const;
    [[nodiscard]] std::uint64_t samples() const noexcept;
    [[nodiscard]] double probability_density(
        std::size_t x_bin,
        std::size_t velocity_x_bin
    ) const;
};

struct MacroscopicSlabState {
    std::size_t slab_index{};
    double x_lower{};
    double x_upper{};
    double x_center{};
    double volume{};
    std::size_t particle_count{};
    double number_density{};
    double mass_density{};
    Vec3 momentum{};
    Vec3 bulk_velocity{};
    double kinetic_energy{};
    double thermal_energy{};
    double temperature{};
    double pressure{};
    // Second velocity moments accumulated over slab particles (volume-moment
    // method).  These give the kinetic stress tensor directly:
    //   tau_kin,xx = m * velocity_x_squared_sum / volume - rho*u_x^2 - p
    //   tau_kin,xy = m * velocity_x_y_product_sum / volume  (Couette shear)
    // Unlike the interface-crossing estimator, they are exact instantaneous
    // volume averages and carry no recrossing bias.
    double velocity_x_squared_sum{};
    double velocity_x_y_product_sum{};
};

struct MultiscaleDiagnosticFrame {
    double time{};
    std::size_t particle_count{};
    std::vector<MicroscopicParticleSample> microscopic_particles;
    PhaseSpaceHistogram phase_space;
    std::vector<MacroscopicSlabState> macroscopic_slabs;

    Vec3 momentum{};
    Vec3 bulk_velocity{};
    double kinetic_energy{};
    double thermal_energy{};
    double temperature{};

    std::size_t reconstructed_particle_count{};
    Vec3 reconstructed_momentum{};
    double reconstructed_kinetic_energy{};
    double momentum_reconstruction_error{};
    double kinetic_energy_reconstruction_error{};
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

[[nodiscard]] MultiscaleDiagnosticFrame measure_multiscale(
    const std::vector<Particle>& particles,
    double time,
    double particle_mass,
    double boltzmann_constant,
    const DomainConfig& domain,
    const DiagnosticsConfig& config
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

class MultiscaleCsvWriter {
public:
    explicit MultiscaleCsvWriter(
        const std::filesystem::path& output_directory
    );

    void write(const MultiscaleDiagnosticFrame& frame);

    [[nodiscard]] const std::filesystem::path& summary_path() const noexcept {
        return summary_path_;
    }
    [[nodiscard]] const std::filesystem::path& microscopic_path()
        const noexcept {
        return microscopic_path_;
    }
    [[nodiscard]] const std::filesystem::path& phase_space_path()
        const noexcept {
        return phase_space_path_;
    }
    [[nodiscard]] const std::filesystem::path& macroscopic_path()
        const noexcept {
        return macroscopic_path_;
    }

private:
    std::size_t frame_index_{};
    std::filesystem::path summary_path_;
    std::filesystem::path microscopic_path_;
    std::filesystem::path phase_space_path_;
    std::filesystem::path macroscopic_path_;
    std::ofstream summary_;
    std::ofstream microscopic_;
    std::ofstream phase_space_;
    std::ofstream macroscopic_;
};

// ---- Paper A: 界面通量追踪 ----

struct InterfaceFluxRecord {
    std::size_t frame{};
    double time{};
    std::size_t interface{};
    double x{};
    double mass_flux{};           // 净质量通量 [m/At]
    double momentum_x_flux{};     // 净 x 动量通量 [m·v_x/At]
    double energy_flux{};         // 净能量通量 [½m|v|²/At]
};

[[nodiscard]] std::vector<InterfaceFluxRecord> compute_interface_fluxes(
    const std::vector<Particle>& particles_old,
    const std::vector<Particle>& particles_new,
    double particle_mass,
    double time_interval,
    const DomainConfig& domain,
    std::size_t n_interfaces
);

class InterfaceFluxCsvWriter {
public:
    explicit InterfaceFluxCsvWriter(
        const std::filesystem::path& output_directory
    );
    void write(const std::vector<InterfaceFluxRecord>& records);
    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
    std::ofstream stream_;
};

}  // namespace particles
