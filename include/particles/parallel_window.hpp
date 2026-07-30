#pragma once

#include <cstddef>
#include <limits>
#include <string_view>
#include <vector>

#include "particles/collision_events.hpp"

namespace particles {

enum class ParallelWindowStopReason {
    completed,
    overlap_detected,
    step_limit,
};

[[nodiscard]] std::string_view parallel_window_stop_reason_name(
    ParallelWindowStopReason reason
) noexcept;

struct ParallelWindowStepResult {
    double requested_window{};
    double advanced_time{};
    double candidate_horizon{};
    std::size_t candidate_pair_count{};
    std::size_t detected_collision_count{};
    std::size_t accepted_collision_count{};
    std::size_t contended_collision_count{};
    std::size_t overlap_pair_count{};
    double minimum_surface_gap{std::numeric_limits<double>::infinity()};
    bool limited_by_candidate_horizon{};
};

struct ParallelWindowRunResult {
    double requested_time{};
    double advanced_time{};
    double maximum_window{};
    std::size_t step_count{};
    std::size_t candidate_rebuild_count{};
    std::size_t horizon_limited_step_count{};
    std::size_t candidate_pair_count{};
    std::size_t detected_collision_count{};
    std::size_t accepted_collision_count{};
    std::size_t contended_collision_count{};
    std::size_t overlap_pair_count{};
    double minimum_surface_gap{std::numeric_limits<double>::infinity()};
    ParallelWindowStopReason stop_reason{
        ParallelWindowStopReason::completed
    };

    [[nodiscard]] bool completed() const noexcept {
        return stop_reason == ParallelWindowStopReason::completed;
    }
};

struct PeriodicStateError {
    double rms_position{};
    double maximum_position{};
    double rms_velocity{};
    double maximum_velocity{};
};

// Resolves every collision that is the deterministic earliest event for both
// participating particles. The accepted events are therefore disjoint and can
// be executed in parallel. Other events are counted as contended and are
// invalidated at the end of the window.
//
// This is a controlled approximation, not a globally event-driven integrator.
// The first version intentionally supports only fully periodic domains.
[[nodiscard]] ParallelWindowStepResult advance_host_parallel_window_step(
    std::vector<Particle>& particles,
    const UniformGrid& grid,
    double requested_window,
    double diameter,
    double tolerance = 1.0e-12
);

[[nodiscard]] ParallelWindowRunResult advance_host_parallel_windows(
    std::vector<Particle>& particles,
    const UniformGrid& grid,
    double duration,
    double maximum_window,
    double diameter,
    std::size_t maximum_steps = 100'000,
    double tolerance = 1.0e-12
);

[[nodiscard]] PeriodicStateError measure_periodic_state_error(
    const std::vector<Particle>& approximate,
    const std::vector<Particle>& reference,
    const UniformGrid& grid
);

}  // namespace particles
