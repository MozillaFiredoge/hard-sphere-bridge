#include "particles/initialization.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace particles {
namespace {

struct LocalState {
    double relative_number_density{1.0};
    double temperature{1.0};
    Vec3 bulk_velocity{};
};

[[nodiscard]] double component(const Vec3& value, std::size_t axis) {
    switch (axis) {
        case 0:
            return value.x;
        case 1:
            return value.y;
        case 2:
            return value.z;
        default:
            throw std::out_of_range("A three-dimensional axis must be 0, 1, or 2.");
    }
}

[[nodiscard]] double& component(Vec3& value, std::size_t axis) {
    switch (axis) {
        case 0:
            return value.x;
        case 1:
            return value.y;
        case 2:
            return value.z;
        default:
            throw std::out_of_range("A three-dimensional axis must be 0, 1, or 2.");
    }
}

[[nodiscard]] bool finite(const Vec3& value) {
    return std::isfinite(value.x) &&
           std::isfinite(value.y) &&
           std::isfinite(value.z);
}

class RandomStream {
public:
    explicit RandomStream(std::uint64_t seed)
        : engine_(seed) {}

    [[nodiscard]] double uniform_open01() {
        constexpr double inverse_2_to_53 = 0x1.0p-53;
        const std::uint64_t mantissa = engine_() >> 11U;
        return (static_cast<double>(mantissa) + 0.5) * inverse_2_to_53;
    }

    [[nodiscard]] double uniform(double lower, double upper) {
        return lower + (upper - lower) * uniform_open01();
    }

    [[nodiscard]] double standard_normal() {
        if (has_spare_normal_) {
            has_spare_normal_ = false;
            return spare_normal_;
        }

        const double radius = std::sqrt(-2.0 * std::log(uniform_open01()));
        const double angle =
            2.0 * std::numbers::pi_v<double> * uniform_open01();

        spare_normal_ = radius * std::sin(angle);
        has_spare_normal_ = true;
        return radius * std::cos(angle);
    }

private:
    std::mt19937_64 engine_;
    bool has_spare_normal_{};
    double spare_normal_{};
};

struct CellKey {
    std::int64_t x{};
    std::int64_t y{};
    std::int64_t z{};

    [[nodiscard]] bool operator==(const CellKey&) const noexcept = default;
};

struct CellKeyHash {
    [[nodiscard]] std::size_t operator()(const CellKey& key) const noexcept {
        std::size_t result = std::hash<std::int64_t>{}(key.x);
        result ^= std::hash<std::int64_t>{}(key.y) +
                  0x9e3779b97f4a7c15ULL +
                  (result << 6U) +
                  (result >> 2U);
        result ^= std::hash<std::int64_t>{}(key.z) +
                  0x9e3779b97f4a7c15ULL +
                  (result << 6U) +
                  (result >> 2U);
        return result;
    }
};

class PlacementGrid {
public:
    PlacementGrid(
        DomainConfig domain,
        double diameter,
        std::size_t expected_particles
    )
        : domain_(domain),
          diameter_(diameter),
          diameter_squared_(diameter * diameter) {
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const double cell_count =
                std::floor(component(domain_.extent, axis) / diameter_);
            if (
                cell_count < 1.0 ||
                cell_count >
                    static_cast<double>(std::numeric_limits<std::int64_t>::max())
            ) {
                throw std::invalid_argument(
                    "Initialization grid dimension is outside the supported range."
                );
            }
            dimensions_[axis] = static_cast<std::int64_t>(cell_count);
            cell_widths_[axis] =
                component(domain_.extent, axis) /
                static_cast<double>(dimensions_[axis]);
        }

        cells_.reserve(expected_particles);
    }

    [[nodiscard]] bool can_insert(
        const Vec3& position,
        const std::vector<Particle>& particles
    ) {
        const CellKey center = cell_key(position);
        double local_minimum_squared = std::numeric_limits<double>::infinity();

        for (std::int64_t offset_x = -1; offset_x <= 1; ++offset_x) {
            for (std::int64_t offset_y = -1; offset_y <= 1; ++offset_y) {
                for (std::int64_t offset_z = -1; offset_z <= 1; ++offset_z) {
                    const std::array<std::int64_t, 3> unwrapped{
                        center.x + offset_x,
                        center.y + offset_y,
                        center.z + offset_z,
                    };

                    CellKey neighbor{};
                    bool valid = true;
                    std::int64_t* coordinates[3]{
                        &neighbor.x,
                        &neighbor.y,
                        &neighbor.z,
                    };

                    for (std::size_t axis = 0; axis < 3; ++axis) {
                        std::int64_t coordinate = unwrapped[axis];

                        if (domain_.boundary[axis] == BoundaryKind::periodic) {
                            coordinate %= dimensions_[axis];
                            if (coordinate < 0) {
                                coordinate += dimensions_[axis];
                            }
                        } else if (
                            coordinate < 0 ||
                            coordinate >= dimensions_[axis]
                        ) {
                            valid = false;
                            break;
                        }

                        *coordinates[axis] = coordinate;
                    }

                    if (!valid) {
                        continue;
                    }

                    const auto found = cells_.find(neighbor);
                    if (found == cells_.end()) {
                        continue;
                    }

                    for (const std::size_t index : found->second) {
                        Vec3 displacement = position - particles[index].position;

                        for (std::size_t axis = 0; axis < 3; ++axis) {
                            if (
                                domain_.boundary[axis] !=
                                BoundaryKind::periodic
                            ) {
                                continue;
                            }

                            const double length =
                                component(domain_.extent, axis);
                            double& value = component(displacement, axis);
                            value -= std::round(value / length) * length;
                        }

                        const double distance_squared =
                            squared_norm(displacement);
                        if (distance_squared < diameter_squared_) {
                            return false;
                        }
                        local_minimum_squared =
                            std::min(local_minimum_squared, distance_squared);
                    }
                }
            }
        }

        if (std::isfinite(local_minimum_squared)) {
            minimum_neighbor_separation_squared_ = std::min(
                minimum_neighbor_separation_squared_,
                local_minimum_squared
            );
        }

        return true;
    }

    void insert(const Vec3& position, std::size_t particle_index) {
        cells_[cell_key(position)].push_back(particle_index);
    }

    [[nodiscard]] double minimum_neighbor_separation() const {
        if (!std::isfinite(minimum_neighbor_separation_squared_)) {
            return diameter_;
        }
        return std::sqrt(minimum_neighbor_separation_squared_);
    }

private:
    [[nodiscard]] CellKey cell_key(const Vec3& position) const {
        CellKey result{};
        std::int64_t* coordinates[3]{&result.x, &result.y, &result.z};

        for (std::size_t axis = 0; axis < 3; ++axis) {
            const double raw = std::floor(
                component(position, axis) / cell_widths_[axis]
            );
            std::int64_t coordinate = static_cast<std::int64_t>(raw);

            if (coordinate == dimensions_[axis]) {
                coordinate = dimensions_[axis] - 1;
            }

            *coordinates[axis] = std::clamp<std::int64_t>(
                coordinate,
                0,
                dimensions_[axis] - 1
            );
        }

        return result;
    }

    DomainConfig domain_;
    double diameter_{};
    double diameter_squared_{};
    std::array<std::int64_t, 3> dimensions_{};
    std::array<double, 3> cell_widths_{};
    std::unordered_map<
        CellKey,
        std::vector<std::size_t>,
        CellKeyHash
    > cells_;
    double minimum_neighbor_separation_squared_{
        std::numeric_limits<double>::infinity()
    };
};

void validate_uniform_config(const UniformGasInitialization& config) {
    if (config.particle_count == 0) {
        throw std::invalid_argument("Particle count must be positive.");
    }
    if (
        !std::isfinite(config.particle_diameter) ||
        config.particle_diameter <= 0.0
    ) {
        throw std::invalid_argument("Particle diameter must be finite and positive.");
    }
    if (!std::isfinite(config.particle_mass) || config.particle_mass <= 0.0) {
        throw std::invalid_argument("Particle mass must be finite and positive.");
    }
    if (
        !std::isfinite(config.boltzmann_constant) ||
        config.boltzmann_constant <= 0.0
    ) {
        throw std::invalid_argument(
            "Boltzmann constant must be finite and positive."
        );
    }
    if (!std::isfinite(config.temperature) || config.temperature <= 0.0) {
        throw std::invalid_argument("Temperature must be finite and positive.");
    }
    if (!finite(config.bulk_velocity)) {
        throw std::invalid_argument("Bulk velocity must be finite.");
    }
    if (config.maximum_attempts_per_particle == 0) {
        throw std::invalid_argument(
            "Maximum placement attempts per particle must be positive."
        );
    }

    const double extents[3]{
        config.domain.extent.x,
        config.domain.extent.y,
        config.domain.extent.z,
    };
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (
            !std::isfinite(extents[axis]) ||
            extents[axis] <= config.particle_diameter
        ) {
            throw std::invalid_argument(
                "Domain extents must exceed the particle diameter."
            );
        }
        if (
            config.domain.boundary[axis] == BoundaryKind::periodic &&
            extents[axis] <= 2.0 * config.particle_diameter
        ) {
            throw std::invalid_argument(
                "Periodic extents must exceed two particle diameters."
            );
        }
    }
}

[[nodiscard]] Vec3 sample_position(
    RandomStream& random,
    const DomainConfig& domain,
    double diameter
) {
    Vec3 position{};
    const double radius = 0.5 * diameter;

    for (std::size_t axis = 0; axis < 3; ++axis) {
        const double length = component(domain.extent, axis);
        const double lower =
            domain.boundary[axis] == BoundaryKind::reflecting ? radius : 0.0;
        const double upper =
            domain.boundary[axis] == BoundaryKind::reflecting
                ? length - radius
                : length;
        component(position, axis) = random.uniform(lower, upper);
    }

    return position;
}

template <typename StateAtPosition>
[[nodiscard]] InitializationResult initialize_with_profile(
    std::uint64_t random_seed,
    std::size_t particle_count,
    double particle_diameter,
    double particle_mass,
    double boltzmann_constant,
    const DomainConfig& domain,
    std::size_t maximum_attempts_per_particle,
    double maximum_relative_density,
    double interface_x,
    StateAtPosition&& state_at_position
) {
    RandomStream random{random_seed};
    PlacementGrid grid{domain, particle_diameter, particle_count};

    InitializationResult result{};
    result.particles.reserve(particle_count);

    while (result.particles.size() < particle_count) {
        std::size_t attempts_for_particle = 0;
        Vec3 position{};
        LocalState state{};

        while (true) {
            ++attempts_for_particle;
            ++result.diagnostics.placement_attempts;

            if (attempts_for_particle > maximum_attempts_per_particle) {
                throw std::runtime_error(
                    "Unable to place a non-overlapping particle; reduce packing "
                    "fraction or increase the placement-attempt limit."
                );
            }

            position = sample_position(random, domain, particle_diameter);
            state = state_at_position(position);

            if (
                random.uniform_open01() * maximum_relative_density >
                state.relative_number_density
            ) {
                continue;
            }
            if (!grid.can_insert(position, result.particles)) {
                continue;
            }
            break;
        }

        const std::size_t particle_index = result.particles.size();
        result.particles.push_back(Particle{.position = position});
        grid.insert(position, particle_index);

        if (position.x < interface_x) {
            ++result.diagnostics.left_particle_count;
        } else {
            ++result.diagnostics.right_particle_count;
        }
    }

    Vec3 sampled_velocity_sum{};
    Vec3 target_velocity_sum{};

    for (Particle& particle : result.particles) {
        const LocalState state = state_at_position(particle.position);
        const double standard_deviation = std::sqrt(
            boltzmann_constant * state.temperature / particle_mass
        );

        particle.velocity = state.bulk_velocity + Vec3{
            standard_deviation * random.standard_normal(),
            standard_deviation * random.standard_normal(),
            standard_deviation * random.standard_normal(),
        };

        sampled_velocity_sum += particle.velocity;
        target_velocity_sum += state.bulk_velocity;
    }

    result.diagnostics.momentum_before_correction =
        particle_mass * sampled_velocity_sum;

    const Vec3 velocity_correction =
        (sampled_velocity_sum - target_velocity_sum) /
        static_cast<double>(particle_count);

    Vec3 corrected_velocity_sum{};
    double thermal_energy_twice = 0.0;

    for (Particle& particle : result.particles) {
        particle.velocity -= velocity_correction;
        corrected_velocity_sum += particle.velocity;

        const LocalState state = state_at_position(particle.position);
        thermal_energy_twice +=
            particle_mass * squared_norm(particle.velocity - state.bulk_velocity);
    }

    result.diagnostics.momentum_after_correction =
        particle_mass * corrected_velocity_sum;
    result.diagnostics.minimum_separation =
        grid.minimum_neighbor_separation();
    result.diagnostics.realized_thermal_temperature =
        thermal_energy_twice /
        (
            3.0 *
            boltzmann_constant *
            static_cast<double>(particle_count)
        );

    return result;
}

}  // namespace

InitializationResult initialize_uniform_gas(
    const UniformGasInitialization& config
) {
    validate_uniform_config(config);

    const auto state_at_position = [&config](const Vec3&) {
        return LocalState{
            .relative_number_density = 1.0,
            .temperature = config.temperature,
            .bulk_velocity = config.bulk_velocity,
        };
    };

    return initialize_with_profile(
        config.random_seed,
        config.particle_count,
        config.particle_diameter,
        config.particle_mass,
        config.boltzmann_constant,
        config.domain,
        config.maximum_attempts_per_particle,
        1.0,
        0.5 * config.domain.extent.x,
        state_at_position
    );
}

InitializationResult initialize_equilibrium_gas(
    const EquilibriumConfig& config
) {
    validate(config);

    return initialize_uniform_gas(UniformGasInitialization{
        .random_seed = config.random_seed,
        .particle_count = config.particle_count,
        .particle_diameter = config.particle_diameter,
        .particle_mass = config.particle_mass,
        .boltzmann_constant = config.boltzmann_constant,
        .temperature = config.temperature,
        .bulk_velocity = config.bulk_velocity,
        .domain = config.domain,
    });
}

InitializationResult initialize_shock_tube(
    const ShockTubeConfig& config,
    std::size_t maximum_attempts_per_particle
) {
    validate(config);

    if (maximum_attempts_per_particle == 0) {
        throw std::invalid_argument(
            "Maximum placement attempts per particle must be positive."
        );
    }

    const auto state_at_position = [&config](const Vec3& position) {
        const double blend = 0.5 * (
            1.0 +
            std::tanh(
                (position.x - config.interface_x) /
                config.transition_width
            )
        );

        LocalState state{
            .relative_number_density =
                (1.0 - blend) * config.left.relative_number_density +
                blend * config.right.relative_number_density,
            .temperature =
                (1.0 - blend) * config.left.temperature +
                blend * config.right.temperature,
            .bulk_velocity =
                (1.0 - blend) * config.left.bulk_velocity +
                blend * config.right.bulk_velocity,
        };

        // Optional transverse shear wave u_y(x) = A sin(2 pi n x / Lx).
        // With the x axis periodic this profile joins itself exactly at the
        // boundary, so it needs no wall and no moving boundary condition.
        if (config.transverse_shear_amplitude != 0.0) {
            constexpr double two_pi = 6.283185307179586476925286766559;
            state.bulk_velocity.y +=
                config.transverse_shear_amplitude *
                std::sin(
                    two_pi * static_cast<double>(config.transverse_shear_mode) *
                    position.x / config.domain.extent.x
                );
        }

        return state;
    };

    return initialize_with_profile(
        config.random_seed,
        config.particle_count,
        config.particle_diameter,
        config.particle_mass,
        config.boltzmann_constant,
        config.domain,
        maximum_attempts_per_particle,
        std::max(
            config.left.relative_number_density,
            config.right.relative_number_density
        ),
        config.interface_x,
        state_at_position
    );
}

}  // namespace particles
