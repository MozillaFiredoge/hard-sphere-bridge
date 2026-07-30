#include "particles/parallel_window.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

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
            throw std::out_of_range(
                "A three-dimensional axis must be 0, 1, or 2."
            );
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
            throw std::out_of_range(
                "A three-dimensional axis must be 0, 1, or 2."
            );
    }
}

void validate_parallel_window_grid(
    const UniformGrid& grid,
    double diameter
) {
    if (!std::isfinite(grid.cell_width) || grid.cell_width <= 0.0) {
        throw std::invalid_argument(
            "Parallel-window cell width must be finite and positive."
        );
    }
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (
            !std::isfinite(component(grid.extent, axis)) ||
            component(grid.extent, axis) <= 0.0
        ) {
            throw std::invalid_argument(
                "Parallel-window extents must be finite and positive."
            );
        }
        if (
            grid.boundary[axis] == BoundaryKind::reflecting &&
            component(grid.extent, axis) <= diameter
        ) {
            throw std::invalid_argument(
                "A reflecting parallel-window extent must exceed the "
                "particle diameter."
            );
        }
    }
    if (!std::isfinite(diameter) || diameter <= 0.0) {
        throw std::invalid_argument(
            "Parallel-window diameter must be finite and positive."
        );
    }
    if (grid.cell_width <= diameter) {
        throw std::invalid_argument(
            "Parallel-window cell width must exceed the diameter."
        );
    }
}

void validate_fully_periodic_grid(const UniformGrid& grid) {
    if (!std::isfinite(grid.cell_width) || grid.cell_width <= 0.0) {
        throw std::invalid_argument(
            "Periodic state-error cell width must be finite and positive."
        );
    }
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (
            !std::isfinite(component(grid.extent, axis)) ||
            component(grid.extent, axis) <= 0.0 ||
            grid.boundary[axis] != BoundaryKind::periodic
        ) {
            throw std::invalid_argument(
                "Periodic state error requires a fully periodic domain."
            );
        }
    }
}

void advance_with_boundaries(
    Particle& particle,
    const UniformGrid& grid,
    double diameter,
    double duration
) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const double extent = component(grid.extent, axis);
        double& position = component(particle.position, axis);
        double& velocity = component(particle.velocity, axis);

        if (grid.boundary[axis] == BoundaryKind::periodic) {
            position += velocity * duration;
            position -= std::floor(position / extent) * extent;
            if (position >= extent) {
                position = 0.0;
            }
            continue;
        }

        const double lower = 0.5 * diameter;
        const double width = extent - diameter;
        const double unfolded =
            position - lower + velocity * duration;
        const double period = 2.0 * width;
        double phase = std::fmod(unfolded, period);
        if (phase < 0.0) {
            phase += period;
        }

        if (phase == 0.0) {
            position = lower;
            velocity = std::abs(velocity);
        } else if (phase == width) {
            position = extent - lower;
            velocity = -std::abs(velocity);
        } else if (phase < width) {
            position = lower + phase;
        } else {
            position = lower + period - phase;
            velocity = -velocity;
        }
    }
}

[[nodiscard]] std::optional<double> earliest_reflecting_wall_time(
    const Particle& particle,
    const UniformGrid& grid,
    double diameter,
    double horizon,
    double tolerance
) {
    std::optional<double> earliest;
    const double radius = 0.5 * diameter;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (grid.boundary[axis] == BoundaryKind::periodic) {
            continue;
        }

        const double position = component(particle.position, axis);
        const double velocity = component(particle.velocity, axis);
        if (std::abs(velocity) <= tolerance) {
            continue;
        }

        double time = velocity > 0.0
            ? (
                  component(grid.extent, axis) -
                  radius -
                  position
              ) /
                  velocity
            : (radius - position) / velocity;
        if (time < -tolerance) {
            throw std::invalid_argument(
                "A particle lies outside a reflecting boundary."
            );
        }
        time = std::max(0.0, time);
        if (time <= horizon + tolerance) {
            time = std::min(time, horizon);
            if (!earliest || time < *earliest) {
                earliest = time;
            }
        }
    }
    return earliest;
}

[[nodiscard]] Vec3 minimum_image(
    Vec3 displacement,
    const UniformGrid& grid
) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (grid.boundary[axis] != BoundaryKind::periodic) {
            continue;
        }
        const double extent = component(grid.extent, axis);
        double& value = component(displacement, axis);
        value -= std::round(value / extent) * extent;
    }
    return displacement;
}

[[nodiscard]] bool event_key_precedes(
    const CollisionEvent& left,
    const CollisionEvent& right
) {
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

[[nodiscard]] double parallel_candidate_horizon(
    const std::vector<Particle>& particles,
    const UniformGrid& grid,
    double diameter
) {
    double maximum_speed = 0.0;
    for (const Particle& particle : particles) {
        maximum_speed = std::max(
            maximum_speed,
            norm(particle.velocity)
        );
    }
    return maximum_speed > 0.0
        ? (
              grid.cell_width - diameter
          ) /
              (2.0 * maximum_speed)
        : std::numeric_limits<double>::infinity();
}

struct GapDiagnostics {
    std::size_t overlap_pair_count{};
    double minimum_surface_gap{std::numeric_limits<double>::infinity()};
};

[[nodiscard]] GapDiagnostics measure_candidate_gaps(
    const std::vector<Particle>& particles,
    const UniformGrid& grid,
    double diameter,
    double tolerance
) {
    const std::vector<CandidatePair> candidates =
        build_host_candidate_pairs(particles, grid);
    GapDiagnostics result{};
    const double penetration_tolerance =
        tolerance * std::max(1.0, diameter);

    for (const CandidatePair pair : candidates) {
        const Vec3 displacement = minimum_image(
            particles[pair.first].position -
                particles[pair.second].position,
            grid
        );
        const double gap = norm(displacement) - diameter;
        result.minimum_surface_gap =
            std::min(result.minimum_surface_gap, gap);
        if (gap < -penetration_tolerance) {
            ++result.overlap_pair_count;
        }
    }

    return result;
}

}  // namespace

std::string_view parallel_window_stop_reason_name(
    ParallelWindowStopReason reason
) noexcept {
    switch (reason) {
        case ParallelWindowStopReason::completed:
            return "completed";
        case ParallelWindowStopReason::overlap_detected:
            return "overlap_detected";
        case ParallelWindowStopReason::step_limit:
            return "step_limit";
    }
    return "unknown";
}

std::string_view adaptive_parallel_window_stop_reason_name(
    AdaptiveParallelWindowStopReason reason
) noexcept {
    switch (reason) {
        case AdaptiveParallelWindowStopReason::completed:
            return "completed";
        case AdaptiveParallelWindowStopReason::minimum_window_exhausted:
            return "minimum_window_exhausted";
        case AdaptiveParallelWindowStopReason::attempt_limit:
            return "attempt_limit";
    }
    return "unknown";
}

ParallelWindowStepResult advance_host_parallel_window_step(
    std::vector<Particle>& particles,
    const UniformGrid& grid,
    double requested_window,
    double diameter,
    double tolerance
) {
    validate_parallel_window_grid(grid, diameter);
    if (!std::isfinite(requested_window) || requested_window <= 0.0) {
        throw std::invalid_argument(
            "Parallel-window duration must be finite and positive."
        );
    }
    if (!std::isfinite(tolerance) || tolerance <= 0.0) {
        throw std::invalid_argument(
            "Parallel-window tolerance must be finite and positive."
        );
    }

    // Reflecting walls are local particle events in this algorithm, so they
    // do not shorten the global candidate-validity horizon.
    const double candidate_horizon =
        parallel_candidate_horizon(particles, grid, diameter);
    if (
        (!std::isfinite(candidate_horizon) &&
         candidate_horizon !=
             std::numeric_limits<double>::infinity()) ||
        candidate_horizon <= 0.0
    ) {
        throw std::runtime_error(
            "Parallel-window candidate horizon made no progress."
        );
    }
    const double actual_window =
        std::min(requested_window, candidate_horizon);

    const std::vector<CandidatePair> candidates =
        build_host_candidate_pairs(particles, grid);
    const std::vector<CollisionEvent> events =
        evaluate_host_collision_events(
            particles,
            candidates,
            grid,
            diameter,
            actual_window,
            tolerance
        );

    const std::uint32_t no_event =
        std::numeric_limits<std::uint32_t>::max();
    std::vector<double> earliest_wall_time(
        particles.size(),
        std::numeric_limits<double>::infinity()
    );
    std::vector<double> earliest_time(
        particles.size(),
        std::numeric_limits<double>::infinity()
    );
    std::vector<std::uint32_t> earliest_event(
        particles.size(),
        no_event
    );

    for (std::size_t particle_index = 0;
         particle_index < particles.size();
         ++particle_index) {
        const std::optional<double> wall_time =
            earliest_reflecting_wall_time(
                particles[particle_index],
                grid,
                diameter,
                actual_window,
                tolerance
            );
        if (wall_time) {
            earliest_wall_time[particle_index] = *wall_time;
            earliest_time[particle_index] = *wall_time;
        }
    }

    for (std::size_t event_index = 0;
         event_index < events.size();
         ++event_index) {
        if (
            event_index >=
            static_cast<std::size_t>(
                std::numeric_limits<std::uint32_t>::max()
            )
        ) {
            throw std::overflow_error(
                "Parallel-window event indices exceed uint32_t."
            );
        }
        const CollisionEvent& event = events[event_index];
        for (const std::uint32_t particle_index :
             {event.first, event.second}) {
            earliest_time[particle_index] = std::min(
                earliest_time[particle_index],
                event.time
            );
        }
    }

    for (std::size_t event_index = 0;
         event_index < events.size();
         ++event_index) {
        const CollisionEvent& event = events[event_index];
        const std::uint32_t index =
            static_cast<std::uint32_t>(event_index);

        for (const std::uint32_t particle_index :
             {event.first, event.second}) {
            if (
                earliest_wall_time[particle_index] <=
                    event.time + tolerance ||
                event.time - earliest_time[particle_index] >
                tolerance
            ) {
                continue;
            }
            const std::uint32_t previous =
                earliest_event[particle_index];
            if (
                previous == no_event ||
                event_key_precedes(event, events[previous])
            ) {
                earliest_event[particle_index] = index;
            }
        }
    }

    std::vector<std::uint32_t> accepted_event_indices;
    accepted_event_indices.reserve(events.size());
    for (std::size_t event_index = 0;
         event_index < events.size();
         ++event_index) {
        const CollisionEvent& event = events[event_index];
        if (
            earliest_event[event.first] == event_index &&
            earliest_event[event.second] == event_index
        ) {
            accepted_event_indices.push_back(
                static_cast<std::uint32_t>(event_index)
            );
        }
    }

    std::vector<Particle> advanced_particles = particles;
    for (Particle& particle : advanced_particles) {
        advance_with_boundaries(
            particle,
            grid,
            diameter,
            actual_window
        );
    }

    for (const std::uint32_t event_index : accepted_event_indices) {
        const CollisionEvent& event = events[event_index];
        Particle first_at_contact = particles[event.first];
        Particle second_at_contact = particles[event.second];
        advance(first_at_contact, event.time);
        advance(second_at_contact, event.time);
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
                "A mutually selected parallel-window event was no longer "
                "approaching."
            );
        }

        const double remaining = actual_window - event.time;
        advance_with_boundaries(
            first_at_contact,
            grid,
            diameter,
            remaining
        );
        advance_with_boundaries(
            second_at_contact,
            grid,
            diameter,
            remaining
        );
        advanced_particles[event.first] = first_at_contact;
        advanced_particles[event.second] = second_at_contact;
    }

    const GapDiagnostics gaps = measure_candidate_gaps(
        advanced_particles,
        grid,
        diameter,
        tolerance
    );
    particles = std::move(advanced_particles);

    return ParallelWindowStepResult{
        .requested_window = requested_window,
        .advanced_time = actual_window,
        .candidate_horizon = candidate_horizon,
        .candidate_pair_count = candidates.size(),
        .detected_collision_count = events.size(),
        .accepted_collision_count = accepted_event_indices.size(),
        .contended_collision_count =
            events.size() - accepted_event_indices.size(),
        .overlap_pair_count = gaps.overlap_pair_count,
        .minimum_surface_gap = gaps.minimum_surface_gap,
        .limited_by_candidate_horizon =
            actual_window < requested_window - tolerance,
    };
}

ParallelWindowRunResult advance_host_parallel_windows(
    std::vector<Particle>& particles,
    const UniformGrid& grid,
    double duration,
    double maximum_window,
    double diameter,
    std::size_t maximum_steps,
    double tolerance
) {
    validate_parallel_window_grid(grid, diameter);
    if (!std::isfinite(duration) || duration < 0.0) {
        throw std::invalid_argument(
            "Parallel-window run duration must be finite and non-negative."
        );
    }
    if (!std::isfinite(maximum_window) || maximum_window <= 0.0) {
        throw std::invalid_argument(
            "Parallel-window maximum must be finite and positive."
        );
    }
    if (maximum_steps == 0) {
        throw std::invalid_argument(
            "Parallel-window step limit must be positive."
        );
    }

    ParallelWindowRunResult result{
        .requested_time = duration,
        .maximum_window = maximum_window,
    };
    const double time_tolerance =
        16.0 *
        std::numeric_limits<double>::epsilon() *
        std::max(1.0, duration);

    while (result.advanced_time < duration - time_tolerance) {
        if (result.step_count >= maximum_steps) {
            result.stop_reason = ParallelWindowStopReason::step_limit;
            return result;
        }

        const double remaining = duration - result.advanced_time;
        const ParallelWindowStepResult step =
            advance_host_parallel_window_step(
                particles,
                grid,
                std::min(maximum_window, remaining),
                diameter,
                tolerance
            );

        result.advanced_time += step.advanced_time;
        ++result.step_count;
        ++result.candidate_rebuild_count;
        result.horizon_limited_step_count +=
            step.limited_by_candidate_horizon ? 1U : 0U;
        result.candidate_pair_count += step.candidate_pair_count;
        result.detected_collision_count +=
            step.detected_collision_count;
        result.accepted_collision_count +=
            step.accepted_collision_count;
        result.contended_collision_count +=
            step.contended_collision_count;
        result.overlap_pair_count += step.overlap_pair_count;
        result.minimum_surface_gap = std::min(
            result.minimum_surface_gap,
            step.minimum_surface_gap
        );

        if (step.overlap_pair_count != 0) {
            result.stop_reason =
                ParallelWindowStopReason::overlap_detected;
            return result;
        }
    }

    result.advanced_time = duration;
    result.stop_reason = ParallelWindowStopReason::completed;
    return result;
}

AdaptiveParallelWindowRunResult
advance_host_adaptive_parallel_windows(
    std::vector<Particle>& particles,
    const UniformGrid& grid,
    double duration,
    double maximum_window,
    double minimum_window,
    double diameter,
    std::size_t maximum_attempts,
    double tolerance,
    std::size_t successful_steps_before_growth,
    double growth_factor,
    std::size_t exact_recovery_batch_limit
) {
    validate_parallel_window_grid(grid, diameter);
    if (!std::isfinite(duration) || duration < 0.0) {
        throw std::invalid_argument(
            "Adaptive parallel-window run duration must be finite and "
            "non-negative."
        );
    }
    if (!std::isfinite(maximum_window) || maximum_window <= 0.0) {
        throw std::invalid_argument(
            "Adaptive parallel-window maximum must be finite and positive."
        );
    }
    if (
        !std::isfinite(minimum_window) ||
        minimum_window <= 0.0 ||
        minimum_window > maximum_window
    ) {
        throw std::invalid_argument(
            "Adaptive parallel-window minimum must be positive and no "
            "larger than its maximum."
        );
    }
    if (maximum_attempts == 0) {
        throw std::invalid_argument(
            "Adaptive parallel-window attempt limit must be positive."
        );
    }
    if (!std::isfinite(tolerance) || tolerance <= 0.0) {
        throw std::invalid_argument(
            "Adaptive parallel-window tolerance must be finite and "
            "positive."
        );
    }
    if (successful_steps_before_growth == 0) {
        throw std::invalid_argument(
            "Adaptive parallel-window growth requires at least one "
            "successful step."
        );
    }
    if (!std::isfinite(growth_factor) || growth_factor <= 1.0) {
        throw std::invalid_argument(
            "Adaptive parallel-window growth factor must be finite and "
            "exceed one."
        );
    }

    AdaptiveParallelWindowRunResult result{
        .requested_time = duration,
        .maximum_window = maximum_window,
        .minimum_window = minimum_window,
        .growth_factor = growth_factor,
        .successful_steps_before_growth =
            successful_steps_before_growth,
    };
    const double time_tolerance =
        16.0 *
        std::numeric_limits<double>::epsilon() *
        std::max(1.0, duration);
    double target_window = maximum_window;
    std::size_t successful_steps_at_window = 0;

    while (result.advanced_time < duration - time_tolerance) {
        const std::vector<Particle> checkpoint = particles;
        double attempt_window = std::min(
            target_window,
            duration - result.advanced_time
        );
        std::size_t refinement_depth = 0;

        while (true) {
            if (result.attempt_count >= maximum_attempts) {
                particles = checkpoint;
                result.stop_reason =
                    AdaptiveParallelWindowStopReason::attempt_limit;
                return result;
            }

            const ParallelWindowStepResult step =
                advance_host_parallel_window_step(
                    particles,
                    grid,
                    attempt_window,
                    diameter,
                    tolerance
                );
            ++result.attempt_count;
            ++result.candidate_rebuild_count;
            result.horizon_limited_attempt_count +=
                step.limited_by_candidate_horizon ? 1U : 0U;
            result.minimum_attempted_window = std::min(
                result.minimum_attempted_window,
                step.advanced_time
            );

            if (step.overlap_pair_count == 0) {
                result.advanced_time += step.advanced_time;
                ++result.accepted_step_count;
                result.maximum_refinement_depth = std::max(
                    result.maximum_refinement_depth,
                    refinement_depth
                );
                result.minimum_accepted_window = std::min(
                    result.minimum_accepted_window,
                    step.advanced_time
                );
                result.candidate_pair_count +=
                    step.candidate_pair_count;
                result.detected_collision_count +=
                    step.detected_collision_count;
                result.accepted_collision_count +=
                    step.accepted_collision_count;
                result.contended_collision_count +=
                    step.contended_collision_count;
                result.minimum_surface_gap = std::min(
                    result.minimum_surface_gap,
                    step.minimum_surface_gap
                );

                if (
                    refinement_depth != 0 ||
                    step.limited_by_candidate_horizon
                ) {
                    target_window = step.advanced_time;
                    successful_steps_at_window = 0;
                } else if (
                    attempt_window >=
                    target_window - time_tolerance
                ) {
                    ++successful_steps_at_window;
                    if (
                        successful_steps_at_window >=
                        successful_steps_before_growth
                    ) {
                        const double grown_window = std::min(
                            maximum_window,
                            growth_factor * target_window
                        );
                        if (
                            grown_window >
                            target_window + time_tolerance
                        ) {
                            target_window = grown_window;
                            ++result.window_growth_count;
                        }
                        successful_steps_at_window = 0;
                    }
                }
                break;
            }

            particles = checkpoint;
            ++result.rollback_count;
            result.rejected_candidate_pair_count +=
                step.candidate_pair_count;
            result.rejected_detected_collision_count +=
                step.detected_collision_count;
            result.rejected_accepted_collision_count +=
                step.accepted_collision_count;
            result.rejected_contended_collision_count +=
                step.contended_collision_count;
            result.rejected_overlap_pair_count +=
                step.overlap_pair_count;
            result.minimum_rejected_surface_gap = std::min(
                result.minimum_rejected_surface_gap,
                step.minimum_surface_gap
            );

            const auto try_exact_recovery = [&]() {
                if (exact_recovery_batch_limit == 0) {
                    return false;
                }
                const TimeWindowAdvanceResult recovery =
                    advance_host_time_window(
                        particles,
                        grid,
                        step.advanced_time,
                        diameter,
                        exact_recovery_batch_limit,
                        tolerance
                    );
                if (!recovery.completed()) {
                    particles = checkpoint;
                    return false;
                }

                result.advanced_time += recovery.advanced_time;
                ++result.accepted_step_count;
                ++result.exact_recovery_count;
                result.candidate_rebuild_count +=
                    recovery.candidate_rebuild_count;
                result.detected_collision_count +=
                    recovery.resolved_collision_count;
                result.accepted_collision_count +=
                    recovery.resolved_collision_count;
                result.exact_recovery_batch_count +=
                    recovery.batch_count;
                result.exact_recovery_candidate_rebuild_count +=
                    recovery.candidate_rebuild_count;
                result.exact_recovery_collision_count +=
                    recovery.resolved_collision_count;
                result.
                    exact_recovery_serialized_conflict_resolution_count +=
                    recovery.serialized_conflict_resolution_count;
                result.exact_recovery_reflected_particle_count +=
                    recovery.reflected_particle_count;
                result.exact_recovery_reflected_axis_count +=
                    recovery.reflected_axis_count;
                result.maximum_refinement_depth = std::max(
                    result.maximum_refinement_depth,
                    refinement_depth
                );
                result.minimum_accepted_window = std::min(
                    result.minimum_accepted_window,
                    recovery.advanced_time
                );
                target_window = recovery.advanced_time;
                successful_steps_at_window = 0;
                return true;
            };

            if (
                step.advanced_time <=
                minimum_window + time_tolerance
            ) {
                if (try_exact_recovery()) {
                    break;
                }
                result.maximum_refinement_depth = std::max(
                    result.maximum_refinement_depth,
                    refinement_depth
                );
                result.stop_reason =
                    AdaptiveParallelWindowStopReason::
                        minimum_window_exhausted;
                return result;
            }

            const double refined_window = std::max(
                minimum_window,
                0.5 * step.advanced_time
            );
            if (
                refined_window >=
                step.advanced_time - time_tolerance
            ) {
                if (try_exact_recovery()) {
                    break;
                }
                result.stop_reason =
                    AdaptiveParallelWindowStopReason::
                        minimum_window_exhausted;
                return result;
            }
            attempt_window = refined_window;
            ++refinement_depth;
        }
    }

    result.advanced_time = duration;
    result.stop_reason =
        AdaptiveParallelWindowStopReason::completed;
    return result;
}

PeriodicStateError measure_periodic_state_error(
    const std::vector<Particle>& approximate,
    const std::vector<Particle>& reference,
    const UniformGrid& grid
) {
    validate_fully_periodic_grid(grid);
    if (approximate.size() != reference.size()) {
        throw std::invalid_argument(
            "State-error particle arrays must have equal sizes."
        );
    }
    if (approximate.empty()) {
        return {};
    }

    double squared_position_error = 0.0;
    double squared_velocity_error = 0.0;
    PeriodicStateError result{};

    for (std::size_t index = 0;
         index < approximate.size();
         ++index) {
        const double position_error = norm(
            minimum_image(
                approximate[index].position -
                    reference[index].position,
                grid
            )
        );
        const double velocity_error = norm(
            approximate[index].velocity -
                reference[index].velocity
        );
        squared_position_error += position_error * position_error;
        squared_velocity_error += velocity_error * velocity_error;
        result.maximum_position =
            std::max(result.maximum_position, position_error);
        result.maximum_velocity =
            std::max(result.maximum_velocity, velocity_error);
    }

    const double inverse_count =
        1.0 / static_cast<double>(approximate.size());
    result.rms_position =
        std::sqrt(squared_position_error * inverse_count);
    result.rms_velocity =
        std::sqrt(squared_velocity_error * inverse_count);
    return result;
}

}  // namespace particles
