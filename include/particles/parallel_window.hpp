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

enum class AdaptiveParallelWindowStopReason {
    completed,
    minimum_window_exhausted,
    attempt_limit,
};

[[nodiscard]] std::string_view adaptive_parallel_window_stop_reason_name(
    AdaptiveParallelWindowStopReason reason
) noexcept;

// A rejected step is transactional: its particle state is restored before a
// retry. Collision counters without the "rejected" prefix therefore describe
// only committed steps and can be compared directly with an exact trajectory.
struct AdaptiveParallelWindowRunResult {
    double requested_time{};
    double advanced_time{};
    double maximum_window{};
    double minimum_window{};
    double growth_factor{};
    double minimum_attempted_window{
        std::numeric_limits<double>::infinity()
    };
    double minimum_accepted_window{
        std::numeric_limits<double>::infinity()
    };
    std::size_t accepted_step_count{};
    std::size_t attempt_count{};
    std::size_t rollback_count{};
    std::size_t maximum_refinement_depth{};
    std::size_t successful_steps_before_growth{};
    std::size_t window_growth_count{};
    std::size_t candidate_rebuild_count{};
    std::size_t horizon_limited_attempt_count{};
    std::size_t candidate_pair_count{};
    std::size_t detected_collision_count{};
    std::size_t accepted_collision_count{};
    std::size_t contended_collision_count{};
    std::size_t rejected_candidate_pair_count{};
    std::size_t rejected_detected_collision_count{};
    std::size_t rejected_accepted_collision_count{};
    std::size_t rejected_contended_collision_count{};
    std::size_t rejected_overlap_pair_count{};
    std::size_t exact_recovery_count{};
    std::size_t exact_recovery_batch_count{};
    std::size_t exact_recovery_candidate_rebuild_count{};
    std::size_t exact_recovery_collision_count{};
    std::size_t exact_recovery_serialized_conflict_resolution_count{};
    std::size_t exact_recovery_reflected_particle_count{};
    std::size_t exact_recovery_reflected_axis_count{};
    double minimum_surface_gap{std::numeric_limits<double>::infinity()};
    double minimum_rejected_surface_gap{
        std::numeric_limits<double>::infinity()
    };
    AdaptiveParallelWindowStopReason stop_reason{
        AdaptiveParallelWindowStopReason::completed
    };

    [[nodiscard]] bool completed() const noexcept {
        return stop_reason ==
               AdaptiveParallelWindowStopReason::completed;
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
// Periodic and reflecting axes may be mixed. A reflecting-wall event competes
// with particle collisions for each particle's earliest event; boundary-aware
// free flight then handles the remainder of the accepted window.
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

// Retries any penetrating step from its original state with half the previous
// attempted duration. A successful step commits atomically; a failed terminal
// attempt leaves particles at the last valid time.
[[nodiscard]] AdaptiveParallelWindowRunResult
advance_host_adaptive_parallel_windows(
    std::vector<Particle>& particles,
    const UniformGrid& grid,
    double duration,
    double maximum_window,
    double minimum_window,
    double diameter,
    std::size_t maximum_attempts = 1'000'000,
    double tolerance = 1.0e-12,
    std::size_t successful_steps_before_growth = 4,
    double growth_factor = 2.0,
    std::size_t exact_recovery_batch_limit = 0
);

[[nodiscard]] PeriodicStateError measure_periodic_state_error(
    const std::vector<Particle>& approximate,
    const std::vector<Particle>& reference,
    const UniformGrid& grid
);

}  // namespace particles
