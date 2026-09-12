#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "particles/vec3.hpp"

namespace particles {

enum class BoundaryKind : std::uint8_t {
    periodic,
    reflecting,
};

struct DomainConfig {
    Vec3 extent{4.0, 1.0, 1.0};
    std::array<BoundaryKind, 3> boundary{
        BoundaryKind::reflecting,
        BoundaryKind::periodic,
        BoundaryKind::periodic,
    };
};

struct MacroscopicState {
    double relative_number_density{1.0};
    double temperature{1.0};
    Vec3 bulk_velocity{};
};

struct DiagnosticsConfig {
    std::size_t microscopic_particle_samples{2'048};
    std::size_t phase_space_x_bins{128};
    std::size_t phase_space_vx_bins{128};
    double phase_space_velocity_x_min{-4.0};
    double phase_space_velocity_x_max{4.0};
    std::size_t macroscopic_x_slabs{256};
};

struct EquilibriumDiagnosticsConfig {
    double sample_interval{0.25};
    std::size_t velocity_x_bins{64};
    double velocity_x_min{-4.0};
    double velocity_x_max{4.0};
};

struct WindowConvergenceConfig {
    double duration{1.0};
    double maximum_window{0.05};
    std::size_t refinement_levels{3};
    std::size_t maximum_steps{100'000};
};

struct AdaptiveWindowConfig {
    double minimum_window{1.0e-7};
    std::size_t maximum_attempts{1'000'000};
    std::size_t successful_steps_before_growth{4};
    double growth_factor{2.0};
};

struct ShockTubeEvolutionConfig {
    double end_time{0.2};
    double frame_interval{0.02};
    double cell_width_factor{2.0};
    double maximum_window{5.0e-4};
    std::size_t exact_recovery_batch_limit{1'000};
    AdaptiveWindowConfig adaptive_window{
        .minimum_window = 1.0e-7,
        .maximum_attempts = 2'000'000,
        .successful_steps_before_growth = 4,
        .growth_factor = 2.0,
    };
};

struct ParticleStateConfig {
    Vec3 position{};
    Vec3 velocity{};
};

struct ParallelWindowScenarioConfig {
    std::uint32_t schema_version{1};
    std::string scenario{"causal_chain"};
    double particle_mass{1.0};
    DomainConfig domain{
        .extent = {12.0, 4.0, 4.0},
        .boundary = {
            BoundaryKind::periodic,
            BoundaryKind::periodic,
            BoundaryKind::periodic,
        },
    };
    double particle_diameter{1.0};
    std::vector<ParticleStateConfig> particles;
    WindowConvergenceConfig window_convergence{
        .duration = 0.8,
        .maximum_window = 0.5,
        .refinement_levels = 3,
        .maximum_steps = 100,
    };
};

struct ParallelWindowStressConfig {
    std::uint32_t schema_version{1};
    std::uint64_t base_random_seed{20260730};
    std::size_t seed_count{4};
    double particle_mass{1.0};
    double boltzmann_constant{1.0};
    DomainConfig domain{
        .extent = {6.0, 6.0, 6.0},
        .boundary = {
            BoundaryKind::periodic,
            BoundaryKind::periodic,
            BoundaryKind::periodic,
        },
    };
    std::size_t particle_count{256};
    double particle_diameter{0.5};
    double temperature{1.0};
    Vec3 bulk_velocity{};
    WindowConvergenceConfig window_convergence{
        .duration = 0.5,
        .maximum_window = 0.06,
        .refinement_levels = 8,
        .maximum_steps = 100'000,
    };
    AdaptiveWindowConfig adaptive_window{};
};

struct CudaScalingConfig {
    std::uint32_t schema_version{1};
    std::uint64_t random_seed{20260730};
    double particle_mass{1.0};
    double boltzmann_constant{1.0};
    std::size_t base_particle_count{1U << 10U};
    std::size_t level_count{11};
    double number_density{32.0 / 27.0};
    double particle_diameter{0.5};
    double temperature{1.0};
    Vec3 bulk_velocity{};
    double cell_width_factor{1.25};
    double warmup_duration{0.002};
    double measured_duration{0.01};
    double maximum_window{0.001};
    AdaptiveWindowConfig adaptive_window{};
    std::size_t maximum_placement_attempts_per_particle{10'000};
};

struct ShockTubeConfig {
    std::uint32_t schema_version{1};
    std::uint64_t random_seed{20260730};

    double particle_mass{1.0};
    double boltzmann_constant{1.0};
    double reference_temperature{1.0};

    DomainConfig domain{};

    std::size_t particle_count{1U << 20U};
    double particle_diameter{0.00415};

    double interface_x{2.0};
    double transition_width{0.08};
    MacroscopicState left{};
    MacroscopicState right{
        .relative_number_density = 0.25,
        .temperature = 0.8,
        .bulk_velocity = {},
    };

    // Optional transverse shear wave, u_y(x) = A sin(2 pi n x / Lx), used to
    // measure the shear viscosity from the stress/strain-rate ratio.  The
    // profile is an exact solution of the (linearised) Navier-Stokes equation
    // with no walls, so no new boundary condition is required: the x axis is
    // simply periodic.  A = 0 reproduces the previous behaviour exactly.
    double transverse_shear_amplitude{0.0};
    std::size_t transverse_shear_mode{1};

    DiagnosticsConfig diagnostics{};
    ShockTubeEvolutionConfig evolution{};
};

struct EquilibriumConfig {
    std::uint32_t schema_version{1};
    std::uint64_t random_seed{20260730};

    double particle_mass{1.0};
    double boltzmann_constant{1.0};

    DomainConfig domain{
        .extent = {10.0, 10.0, 10.0},
        .boundary = {
            BoundaryKind::periodic,
            BoundaryKind::periodic,
            BoundaryKind::periodic,
        },
    };

    std::size_t particle_count{64};
    double particle_diameter{0.5};
    double temperature{1.0};
    Vec3 bulk_velocity{};
    double end_time{5.0};
    EquilibriumDiagnosticsConfig diagnostics{};
    WindowConvergenceConfig window_convergence{};
};

class ConfigError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

[[nodiscard]] constexpr ShockTubeConfig default_shock_tube_config() noexcept {
    return {};
}

[[nodiscard]] constexpr EquilibriumConfig default_equilibrium_config() noexcept {
    return {};
}

void validate(const ShockTubeConfig& config);
void validate(const EquilibriumConfig& config);
void validate(const ParallelWindowScenarioConfig& config);
void validate(const ParallelWindowStressConfig& config);
void validate(const CudaScalingConfig& config);

[[nodiscard]] ShockTubeConfig load_shock_tube_config(
    const std::filesystem::path& path
);

[[nodiscard]] EquilibriumConfig load_equilibrium_config(
    const std::filesystem::path& path
);

[[nodiscard]] ParallelWindowScenarioConfig
load_parallel_window_scenario_config(
    const std::filesystem::path& path
);

[[nodiscard]] ParallelWindowStressConfig
load_parallel_window_stress_config(
    const std::filesystem::path& path
);

[[nodiscard]] CudaScalingConfig load_cuda_scaling_config(
    const std::filesystem::path& path
);

}  // namespace particles
