#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "particles/config.hpp"
#include "particles/hard_sphere.hpp"

namespace particles {

struct UniformGasInitialization {
    std::uint64_t random_seed{20260730};
    std::size_t particle_count{64};
    double particle_diameter{0.5};
    double particle_mass{1.0};
    double boltzmann_constant{1.0};
    double temperature{1.0};
    Vec3 bulk_velocity{};
    DomainConfig domain{
        .extent = {10.0, 10.0, 10.0},
        .boundary = {
            BoundaryKind::periodic,
            BoundaryKind::periodic,
            BoundaryKind::periodic,
        },
    };
    std::size_t maximum_attempts_per_particle{10'000};
};

struct InitializationDiagnostics {
    std::size_t placement_attempts{};
    std::size_t left_particle_count{};
    std::size_t right_particle_count{};
    double minimum_separation{};
    double realized_thermal_temperature{};
    Vec3 momentum_before_correction{};
    Vec3 momentum_after_correction{};
};

struct InitializationResult {
    std::vector<Particle> particles;
    InitializationDiagnostics diagnostics;
};

[[nodiscard]] InitializationResult initialize_uniform_gas(
    const UniformGasInitialization& config
);

[[nodiscard]] InitializationResult initialize_equilibrium_gas(
    const EquilibriumConfig& config
);

[[nodiscard]] InitializationResult initialize_shock_tube(
    const ShockTubeConfig& config,
    std::size_t maximum_attempts_per_particle = 10'000
);

}  // namespace particles

