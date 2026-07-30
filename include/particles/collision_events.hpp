#pragma once

#include <compare>
#include <cstdint>
#include <limits>
#include <vector>

#include "particles/uniform_grid.hpp"

namespace particles {

struct PeriodicImageShift {
    std::int8_t x{};
    std::int8_t y{};
    std::int8_t z{};

    [[nodiscard]] auto operator<=>(
        const PeriodicImageShift&
    ) const noexcept = default;
};

struct CollisionEvent {
    std::uint32_t first{};
    std::uint32_t second{};
    double time{};
    PeriodicImageShift image_shift{};

    [[nodiscard]] auto operator<=>(const CollisionEvent&) const noexcept = default;
};

// Analysis of the events in [earliest_time, earliest_time + tolerance].
//
// An event is independent only when both of its particles occur exactly once
// in the batch. Every other event is reported as conflicting and remains
// unresolved. Indices refer to the input event array.
struct CollisionBatchAnalysis {
    double earliest_time{std::numeric_limits<double>::infinity()};
    std::vector<std::uint32_t> independent_event_indices;
    std::vector<std::uint32_t> conflicting_event_indices;

    [[nodiscard]] std::size_t batch_size() const noexcept {
        return independent_event_indices.size() +
               conflicting_event_indices.size();
    }
};

struct CollisionBatchAdvanceResult {
    double delta_time{};
    std::size_t resolved_event_count{};
    std::size_t serialized_conflict_event_count{};
    std::size_t deferred_conflicting_event_count{};

    [[nodiscard]] bool blocked_by_conflicts() const noexcept {
        return deferred_conflicting_event_count != 0;
    }
};

struct ReflectingWallEvent {
    std::uint32_t particle{};
    double time{};
    std::uint8_t axis_mask{};

    [[nodiscard]] auto operator<=>(
        const ReflectingWallEvent&
    ) const noexcept = default;
};

struct ReflectingWallBatch {
    double earliest_time{std::numeric_limits<double>::infinity()};
    std::vector<ReflectingWallEvent> events;
};

struct ReflectingWallAdvanceResult {
    double delta_time{};
    std::size_t reflected_particle_count{};
    std::size_t reflected_axis_count{};
};

enum class TimeWindowStopReason {
    completed,
    collision_conflict,
    mixed_event_tie,
    batch_limit,
};

struct TimeWindowAdvanceResult {
    double requested_time{};
    double advanced_time{};
    std::size_t batch_count{};
    std::size_t candidate_rebuild_count{};
    std::size_t resolved_collision_count{};
    std::size_t serialized_conflict_resolution_count{};
    std::size_t reflected_particle_count{};
    std::size_t reflected_axis_count{};
    std::size_t deferred_conflicting_event_count{};
    TimeWindowStopReason stop_reason{TimeWindowStopReason::completed};

    [[nodiscard]] bool completed() const noexcept {
        return stop_reason == TimeWindowStopReason::completed;
    }
};

// Conservative window in which a 27-cell candidate list remains complete.
//
// The cell-width bound limits the total relative displacement. Reflecting-wall
// event times are also included because a wall reflection invalidates a
// free-flight particle-particle prediction. Periodic crossings remain valid
// through image shifts.
[[nodiscard]] double conservative_candidate_horizon(
    const std::vector<Particle>& particles,
    const UniformGrid& grid,
    double diameter
);

[[nodiscard]] std::vector<CollisionEvent> evaluate_host_collision_events(
    const std::vector<Particle>& particles,
    const std::vector<CandidatePair>& candidate_pairs,
    const UniformGrid& grid,
    double diameter,
    double horizon,
    double tolerance = 1.0e-12
);

[[nodiscard]] CollisionBatchAnalysis analyze_earliest_collision_batch(
    const std::vector<CollisionEvent>& events,
    std::size_t particle_count,
    double tolerance = 1.0e-12
);

// Advances all particles to the earliest batch anchor and resolves only events
// whose particles do not occur in any other event in that batch. Periodic
// positions are wrapped after applying the independent collision impulses.
[[nodiscard]] CollisionBatchAdvanceResult
advance_host_earliest_collision_batch(
    std::vector<Particle>& particles,
    const std::vector<CollisionEvent>& events,
    const UniformGrid& grid,
    double diameter,
    double tolerance = 1.0e-12
);

[[nodiscard]] ReflectingWallBatch find_host_reflecting_wall_batch(
    const std::vector<Particle>& particles,
    const UniformGrid& grid,
    double diameter,
    double horizon,
    double tolerance = 1.0e-12
);

[[nodiscard]] ReflectingWallAdvanceResult
advance_host_reflecting_wall_batch(
    std::vector<Particle>& particles,
    const ReflectingWallBatch& batch,
    const UniformGrid& grid,
    double diameter,
    double tolerance = 1.0e-12
);

void advance_host_free_flight_in_grid(
    std::vector<Particle>& particles,
    const UniformGrid& grid,
    double delta_time,
    double diameter,
    double tolerance = 1.0e-12
);

[[nodiscard]] TimeWindowAdvanceResult advance_host_time_window(
    std::vector<Particle>& particles,
    const UniformGrid& grid,
    double duration,
    double diameter,
    std::size_t maximum_batches = 1'000'000,
    double tolerance = 1.0e-12
);

}  // namespace particles
