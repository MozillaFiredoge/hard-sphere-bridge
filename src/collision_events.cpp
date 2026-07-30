#include "particles/collision_events.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
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

void wrap_periodic_position(Particle& particle, const UniformGrid& grid) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (grid.boundary[axis] != BoundaryKind::periodic) {
            continue;
        }

        const double extent = component(grid.extent, axis);
        double& position = component(particle.position, axis);
        position -= std::floor(position / extent) * extent;
        if (position >= extent) {
            position = 0.0;
        }
    }
}

}  // namespace

double conservative_candidate_horizon(
    const std::vector<Particle>& particles,
    const UniformGrid& grid,
    double diameter
) {
    if (!std::isfinite(diameter) || diameter <= 0.0) {
        throw std::invalid_argument("Particle diameter must be finite and positive.");
    }
    if (!std::isfinite(grid.cell_width) || grid.cell_width < diameter) {
        throw std::invalid_argument(
            "Candidate cells must be at least one particle diameter wide."
        );
    }

    double maximum_speed = 0.0;
    for (const Particle& particle : particles) {
        maximum_speed = std::max(maximum_speed, norm(particle.velocity));
    }

    double horizon = std::numeric_limits<double>::infinity();
    if (maximum_speed > 0.0) {
        horizon =
            (grid.cell_width - diameter) / (2.0 * maximum_speed);
    }

    const double radius = 0.5 * diameter;
    for (const Particle& particle : particles) {
        for (std::size_t axis = 0; axis < 3; ++axis) {
            if (grid.boundary[axis] != BoundaryKind::reflecting) {
                continue;
            }

            const double velocity = component(particle.velocity, axis);
            if (velocity == 0.0) {
                continue;
            }

            const double position = component(particle.position, axis);
            const double extent = component(grid.extent, axis);
            const double wall_time = velocity > 0.0
                ? (extent - radius - position) / velocity
                : (radius - position) / velocity;

            if (wall_time < 0.0) {
                throw std::invalid_argument(
                    "Particle lies beyond a reflecting-wall event surface."
                );
            }
            horizon = std::min(horizon, wall_time);
        }
    }

    return horizon;
}

std::vector<CollisionEvent> evaluate_host_collision_events(
    const std::vector<Particle>& particles,
    const std::vector<CandidatePair>& candidate_pairs,
    const UniformGrid& grid,
    double diameter,
    double horizon,
    double tolerance
) {
    if (!std::isfinite(diameter) || diameter <= 0.0) {
        throw std::invalid_argument("Particle diameter must be finite and positive.");
    }
    if (!std::isfinite(horizon) || horizon < 0.0) {
        throw std::invalid_argument(
            "Collision-event horizon must be finite and non-negative."
        );
    }
    if (!std::isfinite(tolerance) || tolerance <= 0.0) {
        throw std::invalid_argument(
            "Collision-event tolerance must be finite and positive."
        );
    }

    std::vector<CollisionEvent> events;

    for (const CandidatePair pair : candidate_pairs) {
        if (
            pair.first >= particles.size() ||
            pair.second >= particles.size() ||
            pair.first >= pair.second
        ) {
            throw std::invalid_argument(
                "Candidate pairs must contain valid ordered particle indices."
            );
        }

        std::optional<double> earliest_time;
        PeriodicImageShift earliest_shift{};

        const int lower_x =
            grid.boundary[0] == BoundaryKind::periodic ? -1 : 0;
        const int upper_x =
            grid.boundary[0] == BoundaryKind::periodic ? 1 : 0;
        const int lower_y =
            grid.boundary[1] == BoundaryKind::periodic ? -1 : 0;
        const int upper_y =
            grid.boundary[1] == BoundaryKind::periodic ? 1 : 0;
        const int lower_z =
            grid.boundary[2] == BoundaryKind::periodic ? -1 : 0;
        const int upper_z =
            grid.boundary[2] == BoundaryKind::periodic ? 1 : 0;

        for (int shift_z = lower_z; shift_z <= upper_z; ++shift_z) {
            for (int shift_y = lower_y; shift_y <= upper_y; ++shift_y) {
                for (int shift_x = lower_x; shift_x <= upper_x; ++shift_x) {
                    Particle second_image = particles[pair.second];
                    second_image.position += Vec3{
                        static_cast<double>(shift_x) * grid.extent.x,
                        static_cast<double>(shift_y) * grid.extent.y,
                        static_cast<double>(shift_z) * grid.extent.z,
                    };

                    const std::optional<double> time = time_to_collision(
                        particles[pair.first],
                        second_image,
                        diameter,
                        horizon,
                        tolerance
                    );
                    if (!time) {
                        continue;
                    }
                    if (
                        !earliest_time ||
                        *time < *earliest_time - tolerance
                    ) {
                        earliest_time = *time;
                        earliest_shift = PeriodicImageShift{
                            .x = static_cast<std::int8_t>(shift_x),
                            .y = static_cast<std::int8_t>(shift_y),
                            .z = static_cast<std::int8_t>(shift_z),
                        };
                    }
                }
            }
        }

        if (earliest_time) {
            events.push_back(CollisionEvent{
                .first = pair.first,
                .second = pair.second,
                .time = *earliest_time,
                .image_shift = earliest_shift,
            });
        }
    }

    return events;
}

CollisionBatchAnalysis analyze_earliest_collision_batch(
    const std::vector<CollisionEvent>& events,
    std::size_t particle_count,
    double tolerance
) {
    if (!std::isfinite(tolerance) || tolerance <= 0.0) {
        throw std::invalid_argument(
            "Collision-batch tolerance must be finite and positive."
        );
    }
    if (
        events.size() >
        static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max()
        )
    ) {
        throw std::overflow_error(
            "Collision-batch event indices exceed the uint32_t range."
        );
    }

    CollisionBatchAnalysis analysis{};
    if (events.empty()) {
        return analysis;
    }

    for (const CollisionEvent& event : events) {
        if (
            event.first >= particle_count ||
            event.second >= particle_count ||
            event.first >= event.second
        ) {
            throw std::invalid_argument(
                "Collision events must contain valid ordered particle indices."
            );
        }
        if (!std::isfinite(event.time) || event.time < 0.0) {
            throw std::invalid_argument(
                "Collision-event times must be finite and non-negative."
            );
        }
        analysis.earliest_time =
            std::min(analysis.earliest_time, event.time);
    }

    std::vector<std::uint32_t> particle_participation(particle_count, 0U);
    std::vector<std::uint32_t> batch_indices;
    batch_indices.reserve(events.size());

    for (std::size_t index = 0; index < events.size(); ++index) {
        const CollisionEvent& event = events[index];
        if (event.time - analysis.earliest_time > tolerance) {
            continue;
        }

        if (
            particle_participation[event.first] ==
                std::numeric_limits<std::uint32_t>::max() ||
            particle_participation[event.second] ==
                std::numeric_limits<std::uint32_t>::max()
        ) {
            throw std::overflow_error(
                "Collision-batch particle participation count overflow."
            );
        }

        ++particle_participation[event.first];
        ++particle_participation[event.second];
        batch_indices.push_back(static_cast<std::uint32_t>(index));
    }

    analysis.independent_event_indices.reserve(batch_indices.size());
    analysis.conflicting_event_indices.reserve(batch_indices.size());

    for (const std::uint32_t index : batch_indices) {
        const CollisionEvent& event = events[index];
        const bool independent =
            particle_participation[event.first] == 1U &&
            particle_participation[event.second] == 1U;
        std::vector<std::uint32_t>& destination = independent
            ? analysis.independent_event_indices
            : analysis.conflicting_event_indices;
        destination.push_back(index);
    }

    return analysis;
}

CollisionBatchAdvanceResult advance_host_earliest_collision_batch(
    std::vector<Particle>& particles,
    const std::vector<CollisionEvent>& events,
    const UniformGrid& grid,
    double diameter,
    double tolerance
) {
    if (!std::isfinite(diameter) || diameter <= 0.0) {
        throw std::invalid_argument(
            "Collision-batch diameter must be finite and positive."
        );
    }

    const CollisionBatchAnalysis analysis =
        analyze_earliest_collision_batch(
            events,
            particles.size(),
            tolerance
        );
    if (events.empty()) {
        throw std::invalid_argument(
            "Cannot advance an empty collision-event batch."
        );
    }

    std::vector<Particle> advanced_particles = particles;
    for (Particle& particle : advanced_particles) {
        advance(particle, analysis.earliest_time);
    }

    std::vector<std::uint32_t> resolution_indices =
        analysis.independent_event_indices;
    std::size_t serialized_conflict_event_count = 0;
    if (!analysis.conflicting_event_indices.empty()) {
        const auto selected = std::min_element(
            analysis.conflicting_event_indices.begin(),
            analysis.conflicting_event_indices.end(),
            [&](std::uint32_t left_index, std::uint32_t right_index) {
                const CollisionEvent& left = events[left_index];
                const CollisionEvent& right = events[right_index];
                return std::tuple{
                    left.first,
                    left.second,
                    left.image_shift.x,
                    left.image_shift.y,
                    left.image_shift.z,
                } < std::tuple{
                    right.first,
                    right.second,
                    right.image_shift.x,
                    right.image_shift.y,
                    right.image_shift.z,
                };
            }
        );
        resolution_indices.push_back(*selected);
        serialized_conflict_event_count = 1;
    }

    for (const std::uint32_t event_index : resolution_indices) {
        const CollisionEvent& event = events[event_index];
        const double contact_offset =
            event.time - analysis.earliest_time;

        Particle first_at_contact = advanced_particles[event.first];
        Particle second_at_contact = advanced_particles[event.second];
        advance(first_at_contact, contact_offset);
        advance(second_at_contact, contact_offset);
        second_at_contact.position += Vec3{
            static_cast<double>(event.image_shift.x) * grid.extent.x,
            static_cast<double>(event.image_shift.y) * grid.extent.y,
            static_cast<double>(event.image_shift.z) * grid.extent.z,
        };

        if (
            !resolve_elastic_collision(
                first_at_contact,
                second_at_contact,
                diameter,
                tolerance
            )
        ) {
            throw std::runtime_error(
                "An independent collision event was no longer approaching."
            );
        }

        advanced_particles[event.first].velocity =
            first_at_contact.velocity;
        advanced_particles[event.second].velocity =
            second_at_contact.velocity;
    }

    for (Particle& particle : advanced_particles) {
        wrap_periodic_position(particle, grid);
    }
    particles = std::move(advanced_particles);

    return CollisionBatchAdvanceResult{
        .delta_time = analysis.earliest_time,
        .resolved_event_count = resolution_indices.size(),
        .serialized_conflict_event_count =
            serialized_conflict_event_count,
        .deferred_conflicting_event_count = 0,
    };
}

ReflectingWallBatch find_host_reflecting_wall_batch(
    const std::vector<Particle>& particles,
    const UniformGrid& grid,
    double diameter,
    double horizon,
    double tolerance
) {
    if (!std::isfinite(diameter) || diameter <= 0.0) {
        throw std::invalid_argument(
            "Reflecting-wall diameter must be finite and positive."
        );
    }
    if (!std::isfinite(horizon) || horizon < 0.0) {
        throw std::invalid_argument(
            "Reflecting-wall horizon must be finite and non-negative."
        );
    }
    if (!std::isfinite(tolerance) || tolerance <= 0.0) {
        throw std::invalid_argument(
            "Reflecting-wall tolerance must be finite and positive."
        );
    }
    if (
        particles.size() >
        static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max()
        )
    ) {
        throw std::overflow_error(
            "Reflecting-wall particle indices exceed the uint32_t range."
        );
    }

    const double radius = 0.5 * diameter;
    std::vector<ReflectingWallEvent> candidates;
    candidates.reserve(particles.size());

    for (std::size_t particle_index = 0;
         particle_index < particles.size();
         ++particle_index) {
        const Particle& particle = particles[particle_index];
        double particle_time = std::numeric_limits<double>::infinity();
        std::uint8_t axis_mask = 0U;

        for (std::size_t axis = 0; axis < 3; ++axis) {
            if (grid.boundary[axis] != BoundaryKind::reflecting) {
                continue;
            }

            const double extent = component(grid.extent, axis);
            const double position = component(particle.position, axis);
            const double velocity = component(particle.velocity, axis);
            const double lower = radius;
            const double upper = extent - radius;
            const double geometric_tolerance =
                tolerance * std::max(1.0, extent * extent);

            if (
                !std::isfinite(position) ||
                !std::isfinite(velocity) ||
                position < lower - geometric_tolerance ||
                position > upper + geometric_tolerance
            ) {
                throw std::domain_error(
                    "Particle lies outside a reflecting-wall domain."
                );
            }
            if (std::abs(velocity) <= tolerance) {
                continue;
            }

            double wall_time = velocity > 0.0
                ? (upper - position) / velocity
                : (lower - position) / velocity;
            if (wall_time < -tolerance) {
                throw std::domain_error(
                    "Reflecting-wall event lies in the past."
                );
            }
            wall_time = std::max(0.0, wall_time);
            if (wall_time > horizon + tolerance) {
                continue;
            }
            wall_time = std::min(wall_time, horizon);

            const std::uint8_t axis_bit =
                static_cast<std::uint8_t>(1U << axis);
            if (wall_time < particle_time - tolerance) {
                particle_time = wall_time;
                axis_mask = axis_bit;
            } else if (
                std::abs(wall_time - particle_time) <= tolerance
            ) {
                axis_mask = static_cast<std::uint8_t>(
                    axis_mask | axis_bit
                );
            }
        }

        if (axis_mask != 0U) {
            candidates.push_back(ReflectingWallEvent{
                .particle =
                    static_cast<std::uint32_t>(particle_index),
                .time = particle_time,
                .axis_mask = axis_mask,
            });
        }
    }

    ReflectingWallBatch batch{};
    for (const ReflectingWallEvent& event : candidates) {
        batch.earliest_time =
            std::min(batch.earliest_time, event.time);
    }
    if (candidates.empty()) {
        return batch;
    }

    for (const ReflectingWallEvent& event : candidates) {
        if (event.time - batch.earliest_time <= tolerance) {
            batch.events.push_back(event);
        }
    }
    return batch;
}

ReflectingWallAdvanceResult advance_host_reflecting_wall_batch(
    std::vector<Particle>& particles,
    const ReflectingWallBatch& batch,
    const UniformGrid& grid,
    double diameter,
    double tolerance
) {
    if (!std::isfinite(diameter) || diameter <= 0.0) {
        throw std::invalid_argument(
            "Reflecting-wall diameter must be finite and positive."
        );
    }
    if (!std::isfinite(tolerance) || tolerance <= 0.0) {
        throw std::invalid_argument(
            "Reflecting-wall tolerance must be finite and positive."
        );
    }
    if (
        !std::isfinite(batch.earliest_time) ||
        batch.earliest_time < 0.0 ||
        batch.events.empty()
    ) {
        throw std::invalid_argument(
            "Cannot advance an empty reflecting-wall batch."
        );
    }

    std::vector<Particle> advanced_particles = particles;
    for (Particle& particle : advanced_particles) {
        advance(particle, batch.earliest_time);
    }

    std::vector<bool> seen_particle(particles.size(), false);
    std::size_t reflected_axis_count = 0;
    const double radius = 0.5 * diameter;

    for (const ReflectingWallEvent& event : batch.events) {
        if (
            event.particle >= advanced_particles.size() ||
            event.axis_mask == 0U ||
            (event.axis_mask & static_cast<std::uint8_t>(~0x07U)) != 0U ||
            seen_particle[event.particle]
        ) {
            throw std::invalid_argument(
                "Reflecting-wall batch contains an invalid event."
            );
        }
        seen_particle[event.particle] = true;
        Particle& particle = advanced_particles[event.particle];

        for (std::size_t axis = 0; axis < 3; ++axis) {
            const std::uint8_t axis_bit =
                static_cast<std::uint8_t>(1U << axis);
            if ((event.axis_mask & axis_bit) == 0U) {
                continue;
            }
            if (grid.boundary[axis] != BoundaryKind::reflecting) {
                throw std::invalid_argument(
                    "Wall event references a non-reflecting axis."
                );
            }

            double& position = component(particle.position, axis);
            double& velocity = component(particle.velocity, axis);
            if (std::abs(velocity) <= tolerance) {
                throw std::runtime_error(
                    "Reflecting-wall event has no outgoing velocity."
                );
            }

            const double extent = component(grid.extent, axis);
            position = velocity > 0.0 ? extent - radius : radius;
            velocity = -velocity;
            ++reflected_axis_count;
        }
    }

    for (Particle& particle : advanced_particles) {
        wrap_periodic_position(particle, grid);
    }
    particles = std::move(advanced_particles);

    return ReflectingWallAdvanceResult{
        .delta_time = batch.earliest_time,
        .reflected_particle_count = batch.events.size(),
        .reflected_axis_count = reflected_axis_count,
    };
}

void advance_host_free_flight_in_grid(
    std::vector<Particle>& particles,
    const UniformGrid& grid,
    double delta_time,
    double diameter,
    double tolerance
) {
    if (!std::isfinite(delta_time) || delta_time < 0.0) {
        throw std::invalid_argument(
            "Grid free-flight time must be finite and non-negative."
        );
    }
    if (!std::isfinite(diameter) || diameter <= 0.0) {
        throw std::invalid_argument(
            "Grid free-flight diameter must be finite and positive."
        );
    }
    if (!std::isfinite(tolerance) || tolerance <= 0.0) {
        throw std::invalid_argument(
            "Grid free-flight tolerance must be finite and positive."
        );
    }

    std::vector<Particle> advanced_particles = particles;
    const double radius = 0.5 * diameter;
    for (Particle& particle : advanced_particles) {
        advance(particle, delta_time);

        for (std::size_t axis = 0; axis < 3; ++axis) {
            if (grid.boundary[axis] != BoundaryKind::reflecting) {
                continue;
            }
            const double extent = component(grid.extent, axis);
            const double lower = radius;
            const double upper = extent - radius;
            const double geometric_tolerance =
                tolerance * std::max(1.0, extent * extent);
            double& position = component(particle.position, axis);
            if (
                position < lower - geometric_tolerance ||
                position > upper + geometric_tolerance
            ) {
                throw std::domain_error(
                    "Grid free flight crossed an unresolved reflecting wall."
                );
            }
            position = std::clamp(position, lower, upper);
        }
        wrap_periodic_position(particle, grid);
    }
    particles = std::move(advanced_particles);
}

TimeWindowAdvanceResult advance_host_time_window(
    std::vector<Particle>& particles,
    const UniformGrid& grid,
    double duration,
    double diameter,
    std::size_t maximum_batches,
    double tolerance
) {
    if (!std::isfinite(duration) || duration < 0.0) {
        throw std::invalid_argument(
            "Time-window duration must be finite and non-negative."
        );
    }
    if (!std::isfinite(diameter) || diameter <= 0.0) {
        throw std::invalid_argument(
            "Time-window diameter must be finite and positive."
        );
    }
    if (!std::isfinite(tolerance) || tolerance <= 0.0) {
        throw std::invalid_argument(
            "Time-window tolerance must be finite and positive."
        );
    }
    if (grid.cell_width <= diameter) {
        throw std::invalid_argument(
            "Time-window cell width must exceed the particle diameter."
        );
    }
    if (maximum_batches == 0) {
        throw std::invalid_argument(
            "Time-window batch limit must be positive."
        );
    }

    TimeWindowAdvanceResult result{
        .requested_time = duration,
    };

    while (result.advanced_time < duration) {
        if (result.batch_count >= maximum_batches) {
            result.stop_reason = TimeWindowStopReason::batch_limit;
            return result;
        }

        const double remaining = duration - result.advanced_time;
        const double safe_horizon = std::min(
            remaining,
            conservative_candidate_horizon(
                particles,
                grid,
                diameter
            )
        );
        if (!std::isfinite(safe_horizon) || safe_horizon < 0.0) {
            throw std::runtime_error(
                "Host time window produced an invalid safe horizon."
            );
        }

        const std::vector<CandidatePair> candidates =
            build_host_candidate_pairs(particles, grid);
        const std::vector<CollisionEvent> collision_events =
            evaluate_host_collision_events(
                particles,
                candidates,
                grid,
                diameter,
                safe_horizon,
                tolerance
            );
        const CollisionBatchAnalysis collision_batch =
            analyze_earliest_collision_batch(
                collision_events,
                particles.size(),
                tolerance
            );
        const ReflectingWallBatch wall_batch =
            find_host_reflecting_wall_batch(
                particles,
                grid,
                diameter,
                safe_horizon,
                tolerance
            );
        ++result.candidate_rebuild_count;

        const bool has_collision = !collision_events.empty();
        const bool has_wall = !wall_batch.events.empty();
        if (
            has_collision &&
            has_wall &&
            std::abs(
                collision_batch.earliest_time -
                wall_batch.earliest_time
            ) <= tolerance
        ) {
            advance_host_free_flight_in_grid(
                particles,
                grid,
                std::min(
                    collision_batch.earliest_time,
                    wall_batch.earliest_time
                ),
                diameter,
                tolerance
            );
            result.advanced_time += std::min(
                collision_batch.earliest_time,
                wall_batch.earliest_time
            );
            ++result.batch_count;
            result.deferred_conflicting_event_count =
                collision_batch.batch_size() +
                wall_batch.events.size();
            result.stop_reason = TimeWindowStopReason::mixed_event_tie;
            return result;
        }

        if (
            has_collision &&
            (
                !has_wall ||
                collision_batch.earliest_time <
                    wall_batch.earliest_time - tolerance
            )
        ) {
            const CollisionBatchAdvanceResult batch_result =
                advance_host_earliest_collision_batch(
                    particles,
                    collision_events,
                    grid,
                    diameter,
                    tolerance
                );
            result.advanced_time += batch_result.delta_time;
            result.resolved_collision_count +=
                batch_result.resolved_event_count;
            result.serialized_conflict_resolution_count +=
                batch_result.serialized_conflict_event_count;
            ++result.batch_count;

            if (batch_result.blocked_by_conflicts()) {
                result.deferred_conflicting_event_count =
                    batch_result.deferred_conflicting_event_count;
                result.stop_reason =
                    TimeWindowStopReason::collision_conflict;
                return result;
            }
            continue;
        }

        if (has_wall) {
            const ReflectingWallAdvanceResult wall_result =
                advance_host_reflecting_wall_batch(
                    particles,
                    wall_batch,
                    grid,
                    diameter,
                    tolerance
                );
            result.advanced_time += wall_result.delta_time;
            result.reflected_particle_count +=
                wall_result.reflected_particle_count;
            result.reflected_axis_count +=
                wall_result.reflected_axis_count;
            ++result.batch_count;
            continue;
        }

        if (safe_horizon == 0.0) {
            throw std::runtime_error(
                "Host time window made no progress without an event."
            );
        }
        advance_host_free_flight_in_grid(
            particles,
            grid,
            safe_horizon,
            diameter,
            tolerance
        );
        result.advanced_time += safe_horizon;
        ++result.batch_count;
    }

    result.advanced_time = duration;
    result.stop_reason = TimeWindowStopReason::completed;
    return result;
}

}  // namespace particles
