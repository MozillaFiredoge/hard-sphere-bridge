#pragma once

#include <optional>

#include "particles/vec3.hpp"

namespace particles {

struct Particle {
    Vec3 position{};
    Vec3 velocity{};
};

// Advances one freely moving particle without applying boundaries.
void advance(Particle& particle, double delta_time);

// Finds the first dynamically relevant contact in [0, horizon].
//
// A grazing trajectory produces no impulse and is therefore not returned as a
// collision event. Invalid input or a pre-existing overlap throws.
[[nodiscard]] std::optional<double> time_to_collision(
    const Particle& first,
    const Particle& second,
    double diameter,
    double horizon,
    double tolerance = 1.0e-12
);

// Applies an equal-mass elastic impulse to two particles at contact.
//
// Returns false when the particles are touching but not approaching. Invalid
// input, overlap, or particles that are not at contact throws.
[[nodiscard]] bool resolve_elastic_collision(
    Particle& first,
    Particle& second,
    double diameter,
    double tolerance = 1.0e-12
);

[[nodiscard]] double kinetic_energy(const Particle& particle, double mass = 1.0);
[[nodiscard]] Vec3 momentum(const Particle& particle, double mass = 1.0);

}  // namespace particles

