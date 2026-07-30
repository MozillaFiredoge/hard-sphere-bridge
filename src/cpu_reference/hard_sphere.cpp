#include "particles/hard_sphere.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace particles {
namespace {

void validate_common_input(double diameter, double tolerance) {
    if (!std::isfinite(diameter) || diameter <= 0.0) {
        throw std::invalid_argument("Hard-sphere diameter must be finite and positive.");
    }
    if (!std::isfinite(tolerance) || tolerance <= 0.0) {
        throw std::invalid_argument("Collision tolerance must be finite and positive.");
    }
}

[[nodiscard]] double geometric_scale(const Vec3& relative_position, double diameter) {
    return std::max({1.0, squared_norm(relative_position), diameter * diameter});
}

}  // namespace

void advance(Particle& particle, double delta_time) {
    if (!std::isfinite(delta_time) || delta_time < 0.0) {
        throw std::invalid_argument("Free-flight time must be finite and non-negative.");
    }
    particle.position += particle.velocity * delta_time;
}

std::optional<double> time_to_collision(
    const Particle& first,
    const Particle& second,
    double diameter,
    double horizon,
    double tolerance
) {
    validate_common_input(diameter, tolerance);

    if (std::isnan(horizon) || horizon < 0.0) {
        throw std::invalid_argument("Collision horizon must be non-negative.");
    }

    const Vec3 relative_position = first.position - second.position;
    const Vec3 relative_velocity = first.velocity - second.velocity;

    const double scale = geometric_scale(relative_position, diameter);
    double separation_term =
        squared_norm(relative_position) - diameter * diameter;

    if (separation_term < -tolerance * scale) {
        throw std::domain_error("Cannot predict a hard-sphere collision from an overlap.");
    }
    if (separation_term < 0.0) {
        separation_term = 0.0;
    }

    const double speed_squared = squared_norm(relative_velocity);
    if (speed_squared <= tolerance * tolerance) {
        return std::nullopt;
    }

    const double approach = dot(relative_position, relative_velocity);
    if (approach >= 0.0) {
        return std::nullopt;
    }

    double discriminant =
        approach * approach - speed_squared * separation_term;
    const double discriminant_scale =
        approach * approach + speed_squared * std::abs(separation_term) + 1.0;

    if (discriminant <= tolerance * discriminant_scale) {
        // A negative discriminant misses. A zero discriminant only grazes and
        // has no normal impulse, so neither is a collision event.
        return std::nullopt;
    }

    discriminant = std::max(0.0, discriminant);
    const double square_root = std::sqrt(discriminant);

    // This form avoids subtracting nearly equal values for distant contacts.
    double collision_time =
        separation_term == 0.0
            ? 0.0
            : separation_term / (-approach + square_root);

    if (collision_time < -tolerance) {
        return std::nullopt;
    }
    collision_time = std::max(0.0, collision_time);

    if (collision_time > horizon + tolerance) {
        return std::nullopt;
    }

    return std::min(collision_time, horizon);
}

bool resolve_elastic_collision(
    Particle& first,
    Particle& second,
    double diameter,
    double tolerance
) {
    validate_common_input(diameter, tolerance);

    const Vec3 relative_position = first.position - second.position;
    const double distance_squared = squared_norm(relative_position);
    const double scale = geometric_scale(relative_position, diameter);
    const double contact_error = distance_squared - diameter * diameter;

    if (contact_error < -tolerance * scale) {
        throw std::domain_error("Cannot resolve a hard-sphere collision from an overlap.");
    }
    if (std::abs(contact_error) > tolerance * scale) {
        throw std::invalid_argument("Hard spheres must be at contact before resolution.");
    }

    const double distance = std::sqrt(distance_squared);
    if (distance <= tolerance) {
        throw std::domain_error("Collision normal is undefined for coincident centers.");
    }

    const Vec3 normal = relative_position / distance;
    const Vec3 relative_velocity = first.velocity - second.velocity;
    const double normal_relative_speed = dot(relative_velocity, normal);

    if (normal_relative_speed >= -tolerance) {
        return false;
    }

    const Vec3 normal_velocity = normal_relative_speed * normal;
    first.velocity -= normal_velocity;
    second.velocity += normal_velocity;
    return true;
}

double kinetic_energy(const Particle& particle, double mass) {
    if (!std::isfinite(mass) || mass <= 0.0) {
        throw std::invalid_argument("Particle mass must be finite and positive.");
    }
    return 0.5 * mass * squared_norm(particle.velocity);
}

Vec3 momentum(const Particle& particle, double mass) {
    if (!std::isfinite(mass) || mass <= 0.0) {
        throw std::invalid_argument("Particle mass must be finite and positive.");
    }
    return mass * particle.velocity;
}

}  // namespace particles

