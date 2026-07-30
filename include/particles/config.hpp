#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

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
    std::size_t phase_space_x_bins{128};
    std::size_t phase_space_vx_bins{128};
    std::size_t macroscopic_x_slabs{256};
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

    DiagnosticsConfig diagnostics{};
};

[[nodiscard]] constexpr ShockTubeConfig default_shock_tube_config() noexcept {
    return {};
}

}  // namespace particles

