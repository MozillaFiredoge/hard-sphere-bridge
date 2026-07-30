#include "particles/event_driven_simulator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace particles {
namespace {

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

[[nodiscard]] int event_priority(EventKind kind) {
    switch (kind) {
        case EventKind::particle_collision:
            return 0;
        case EventKind::reflecting_wall:
            return 1;
        case EventKind::periodic_wrap:
            return 2;
    }
    throw std::logic_error("Unknown event kind.");
}

[[nodiscard]] Vec3 minimum_image_displacement(
    Vec3 displacement,
    const DomainConfig& domain
) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (domain.boundary[axis] != BoundaryKind::periodic) {
            continue;
        }

        const double length = component(domain.extent, axis);
        double& coordinate = component(displacement, axis);
        coordinate -= std::round(coordinate / length) * length;
    }
    return displacement;
}

}  // namespace

struct EventDrivenSimulator::EventCandidate {
    EventKind kind{EventKind::particle_collision};
    double delta_time{};
    std::size_t first{no_particle};
    std::size_t second{no_particle};
    std::size_t axis{no_axis};
    Vec3 periodic_image_shift{};
};

EventDrivenSimulator::EventDrivenSimulator(
    std::vector<Particle> particles,
    double diameter,
    DomainConfig domain,
    double tolerance
)
    : particles_(std::move(particles)),
      diameter_(diameter),
      domain_(domain),
      tolerance_(tolerance) {
    validate_initial_state();
}

std::optional<SimulationEvent> EventDrivenSimulator::advance_to_next_event(
    double horizon
) {
    if (!std::isfinite(horizon) || horizon < 0.0) {
        throw std::invalid_argument("Event horizon must be finite and non-negative.");
    }

    const std::optional<EventCandidate> candidate = find_next_event(horizon);

    if (!candidate) {
        advance_all(horizon);
        time_ += horizon;
        return std::nullopt;
    }

    advance_all(candidate->delta_time);
    time_ += candidate->delta_time;
    resolve(*candidate);

    return SimulationEvent{
        .kind = candidate->kind,
        .time = time_,
        .delta_time = candidate->delta_time,
        .first = candidate->first,
        .second = candidate->second,
        .axis = candidate->axis,
        .periodic_image_shift = candidate->periodic_image_shift,
    };
}

void EventDrivenSimulator::run_until(
    double end_time,
    std::size_t maximum_events
) {
    if (!std::isfinite(end_time) || end_time < time_) {
        throw std::invalid_argument(
            "Simulation end time must be finite and not precede the current time."
        );
    }

    std::size_t events = 0;

    while (time_ < end_time) {
        const double remaining = end_time - time_;
        const std::optional<SimulationEvent> event =
            advance_to_next_event(remaining);

        if (!event) {
            break;
        }

        ++events;
        if (events > maximum_events) {
            throw std::runtime_error(
                "Event limit exceeded; possible zero-time event loop."
            );
        }
    }
}

std::optional<EventDrivenSimulator::EventCandidate>
EventDrivenSimulator::find_next_event(double horizon) const {
    std::optional<EventCandidate> best;

    const auto consider = [&](const EventCandidate& candidate) {
        if (candidate.delta_time < -tolerance_ ||
            candidate.delta_time > horizon + tolerance_) {
            return;
        }

        EventCandidate normalized = candidate;
        normalized.delta_time =
            std::clamp(normalized.delta_time, 0.0, horizon);

        if (!best) {
            best = normalized;
            return;
        }

        if (normalized.delta_time < best->delta_time - tolerance_) {
            best = normalized;
            return;
        }

        if (std::abs(normalized.delta_time - best->delta_time) > tolerance_) {
            return;
        }

        const auto normalized_key = std::tuple{
            event_priority(normalized.kind),
            normalized.first,
            normalized.second,
            normalized.axis,
        };
        const auto best_key = std::tuple{
            event_priority(best->kind),
            best->first,
            best->second,
            best->axis,
        };

        if (normalized_key < best_key) {
            best = normalized;
        }
    };

    for (std::size_t first = 0; first < particles_.size(); ++first) {
        for (std::size_t second = first + 1; second < particles_.size(); ++second) {
            const std::array<int, 3> lower_shift{
                domain_.boundary[0] == BoundaryKind::periodic ? -1 : 0,
                domain_.boundary[1] == BoundaryKind::periodic ? -1 : 0,
                domain_.boundary[2] == BoundaryKind::periodic ? -1 : 0,
            };
            const std::array<int, 3> upper_shift{
                domain_.boundary[0] == BoundaryKind::periodic ? 1 : 0,
                domain_.boundary[1] == BoundaryKind::periodic ? 1 : 0,
                domain_.boundary[2] == BoundaryKind::periodic ? 1 : 0,
            };

            for (int shift_x = lower_shift[0];
                 shift_x <= upper_shift[0];
                 ++shift_x) {
                for (int shift_y = lower_shift[1];
                     shift_y <= upper_shift[1];
                     ++shift_y) {
                    for (int shift_z = lower_shift[2];
                         shift_z <= upper_shift[2];
                         ++shift_z) {
                        const Vec3 image_shift{
                            static_cast<double>(shift_x) * domain_.extent.x,
                            static_cast<double>(shift_y) * domain_.extent.y,
                            static_cast<double>(shift_z) * domain_.extent.z,
                        };

                        Particle second_image = particles_[second];
                        second_image.position += image_shift;

                        const std::optional<double> collision_time =
                            time_to_collision(
                                particles_[first],
                                second_image,
                                diameter_,
                                horizon,
                                tolerance_
                            );

                        if (!collision_time) {
                            continue;
                        }

                        consider(EventCandidate{
                            .kind = EventKind::particle_collision,
                            .delta_time = *collision_time,
                            .first = first,
                            .second = second,
                            .axis = no_axis,
                            .periodic_image_shift = image_shift,
                        });
                    }
                }
            }
        }
    }

    const double radius = 0.5 * diameter_;

    for (std::size_t particle_index = 0;
         particle_index < particles_.size();
         ++particle_index) {
        const Particle& particle = particles_[particle_index];

        for (std::size_t axis = 0; axis < 3; ++axis) {
            const double position = component(particle.position, axis);
            const double velocity = component(particle.velocity, axis);

            if (std::abs(velocity) <= tolerance_) {
                continue;
            }

            const double length = component(domain_.extent, axis);
            const BoundaryKind boundary = domain_.boundary[axis];
            double delta_time = 0.0;

            if (boundary == BoundaryKind::periodic) {
                delta_time = velocity > 0.0
                    ? (length - position) / velocity
                    : (0.0 - position) / velocity;
            } else {
                const double wall = velocity > 0.0 ? length - radius : radius;
                delta_time = (wall - position) / velocity;
            }

            consider(EventCandidate{
                .kind = boundary == BoundaryKind::periodic
                    ? EventKind::periodic_wrap
                    : EventKind::reflecting_wall,
                .delta_time = delta_time,
                .first = particle_index,
                .second = no_particle,
                .axis = axis,
                .periodic_image_shift = {},
            });
        }
    }

    return best;
}

void EventDrivenSimulator::advance_all(double delta_time) {
    for (Particle& particle : particles_) {
        advance(particle, delta_time);
    }
}

void EventDrivenSimulator::resolve(const EventCandidate& candidate) {
    Particle& first = particles_.at(candidate.first);

    switch (candidate.kind) {
        case EventKind::particle_collision: {
            Particle& second = particles_.at(candidate.second);
            Particle second_image = second;
            second_image.position += candidate.periodic_image_shift;

            const bool resolved = resolve_elastic_collision(
                first,
                second_image,
                diameter_,
                tolerance_
            );

            if (!resolved) {
                throw std::logic_error(
                    "Scheduled particle collision had no approaching normal velocity."
                );
            }

            second.velocity = second_image.velocity;
            ++statistics_.particle_collisions;
            break;
        }

        case EventKind::reflecting_wall: {
            const double length = component(domain_.extent, candidate.axis);
            const double radius = 0.5 * diameter_;
            double& position = component(first.position, candidate.axis);
            double& velocity = component(first.velocity, candidate.axis);

            position = velocity > 0.0 ? length - radius : radius;
            velocity = -velocity;
            ++statistics_.reflecting_wall_collisions;
            break;
        }

        case EventKind::periodic_wrap: {
            const double length = component(domain_.extent, candidate.axis);
            double& position = component(first.position, candidate.axis);
            const double velocity = component(first.velocity, candidate.axis);

            position = velocity > 0.0 ? 0.0 : length;
            ++statistics_.periodic_wraps;
            break;
        }
    }
}

void EventDrivenSimulator::validate_initial_state() const {
    if (!std::isfinite(diameter_) || diameter_ <= 0.0) {
        throw std::invalid_argument("Hard-sphere diameter must be finite and positive.");
    }
    if (!std::isfinite(tolerance_) || tolerance_ <= 0.0) {
        throw std::invalid_argument("Simulation tolerance must be finite and positive.");
    }
    if (!finite(domain_.extent)) {
        throw std::invalid_argument("Domain extents must be finite.");
    }

    const double radius = 0.5 * diameter_;

    for (std::size_t axis = 0; axis < 3; ++axis) {
        const double length = component(domain_.extent, axis);
        if (length <= diameter_) {
            throw std::invalid_argument(
                "Every domain extent must exceed the hard-sphere diameter."
            );
        }
        if (
            domain_.boundary[axis] == BoundaryKind::periodic &&
            length <= 2.0 * diameter_
        ) {
            throw std::invalid_argument(
                "A periodic extent must exceed two hard-sphere diameters."
            );
        }
    }

    for (const Particle& particle : particles_) {
        if (!finite(particle.position) || !finite(particle.velocity)) {
            throw std::invalid_argument(
                "Particle positions and velocities must be finite."
            );
        }

        for (std::size_t axis = 0; axis < 3; ++axis) {
            const double position = component(particle.position, axis);
            const double length = component(domain_.extent, axis);

            if (domain_.boundary[axis] == BoundaryKind::periodic) {
                if (position < 0.0 || position > length) {
                    throw std::invalid_argument(
                        "Periodic particle position lies outside the domain."
                    );
                }
            } else if (
                position < radius - tolerance_ ||
                position > length - radius + tolerance_
            ) {
                throw std::invalid_argument(
                    "Particle center crosses a reflecting hard wall."
                );
            }
        }
    }

    const double diameter_squared = diameter_ * diameter_;

    for (std::size_t first = 0; first < particles_.size(); ++first) {
        for (std::size_t second = first + 1; second < particles_.size(); ++second) {
            const Vec3 displacement = minimum_image_displacement(
                particles_[first].position - particles_[second].position,
                domain_
            );
            const double scale =
                std::max({1.0, squared_norm(displacement), diameter_squared});

            if (squared_norm(displacement) <
                diameter_squared - tolerance_ * scale) {
                throw std::invalid_argument(
                    "Initial hard-sphere configuration contains an overlap."
                );
            }
        }
    }
}

}  // namespace particles
