#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "particles/collision_events.hpp"
#include "particles/hard_sphere.hpp"
#include "particles/parallel_window.hpp"
#include "particles/uniform_grid.hpp"

namespace particles::cuda {

class CudaError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct CellListSnapshot {
    std::vector<std::uint32_t> sorted_cell_ids;
    std::vector<std::uint32_t> particle_indices;
    std::vector<std::uint32_t> cell_start;
    std::vector<std::uint32_t> cell_end;
};

struct CandidatePairSnapshot {
    std::vector<CandidatePair> pairs;
};

struct CollisionEventSnapshot {
    std::vector<CollisionEvent> events;
};

struct CollisionBatchSnapshot {
    double earliest_time{std::numeric_limits<double>::infinity()};
    std::vector<std::uint32_t> independent_event_indices;
    std::vector<std::uint32_t> conflicting_event_indices;

    [[nodiscard]] std::size_t batch_size() const noexcept {
        return independent_event_indices.size() +
               conflicting_event_indices.size();
    }
};

struct ReflectingWallBatchSnapshot {
    double earliest_time{std::numeric_limits<double>::infinity()};
    std::vector<ReflectingWallEvent> events;
};

struct ParticleGridDeviceView {
    double* position_x{};
    double* position_y{};
    double* position_z{};
    double* velocity_x{};
    double* velocity_y{};
    double* velocity_z{};

    const std::uint32_t* sorted_cell_ids{};
    const std::uint32_t* particle_indices{};
    const std::uint32_t* cell_start{};
    const std::uint32_t* cell_end{};
    const std::uint32_t* candidate_first{};
    const std::uint32_t* candidate_second{};
    const std::uint32_t* event_first{};
    const std::uint32_t* event_second{};
    const double* event_time{};
    const std::int8_t* event_shift_x{};
    const std::int8_t* event_shift_y{};
    const std::int8_t* event_shift_z{};
    const std::uint32_t* independent_event_indices{};
    const std::uint32_t* conflicting_event_indices{};
    const std::uint32_t* reflecting_wall_particles{};
    const std::uint8_t* reflecting_wall_axis_masks{};

    std::size_t particle_count{};
    std::size_t candidate_count{};
    std::size_t event_count{};
    std::size_t independent_event_count{};
    std::size_t conflicting_event_count{};
    std::size_t reflecting_wall_event_count{};
    UniformGrid grid{};
};

class ParticleGrid {
public:
    explicit ParticleGrid(std::size_t particle_count);
    ~ParticleGrid();

    ParticleGrid(ParticleGrid&&) noexcept;
    ParticleGrid& operator=(ParticleGrid&&) noexcept;

    ParticleGrid(const ParticleGrid&) = delete;
    ParticleGrid& operator=(const ParticleGrid&) = delete;

    void upload(const std::vector<Particle>& particles);
    void advance_free_flight(double delta_time);
    void build_cell_list(const UniformGrid& grid);
    void build_candidate_pairs();
    void evaluate_collision_events(
        double diameter,
        double horizon,
        double tolerance = 1.0e-12
    );
    void analyze_earliest_collision_batch(double tolerance = 1.0e-12);
    [[nodiscard]] CollisionBatchAdvanceResult
    advance_earliest_collision_batch(
        double diameter,
        double tolerance = 1.0e-12
    );
    void evaluate_reflecting_wall_batch(
        double diameter,
        double horizon,
        double tolerance = 1.0e-12
    );
    [[nodiscard]] ReflectingWallAdvanceResult
    advance_reflecting_wall_batch(
        double diameter,
        double tolerance = 1.0e-12
    );
    void advance_free_flight_in_grid(
        double delta_time,
        double diameter,
        double tolerance = 1.0e-12
    );
    [[nodiscard]] TimeWindowAdvanceResult advance_time_window(
        const UniformGrid& grid,
        double duration,
        double diameter,
        std::size_t maximum_batches = 1'000'000,
        double tolerance = 1.0e-12
    );
    [[nodiscard]] ParallelWindowStepResult advance_parallel_window_step(
        const UniformGrid& grid,
        double requested_window,
        double diameter,
        double tolerance = 1.0e-12
    );
    [[nodiscard]] ParallelWindowRunResult advance_parallel_windows(
        const UniformGrid& grid,
        double duration,
        double maximum_window,
        double diameter,
        std::size_t maximum_steps = 100'000,
        double tolerance = 1.0e-12
    );
    [[nodiscard]] AdaptiveParallelWindowRunResult
    advance_adaptive_parallel_windows(
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

    [[nodiscard]] std::vector<Particle> download_particles() const;
    [[nodiscard]] CellListSnapshot download_cell_list() const;
    [[nodiscard]] CandidatePairSnapshot download_candidate_pairs() const;
    [[nodiscard]] CollisionEventSnapshot download_collision_events() const;
    [[nodiscard]] CollisionBatchSnapshot download_collision_batch() const;
    [[nodiscard]] ReflectingWallBatchSnapshot
    download_reflecting_wall_batch() const;
    [[nodiscard]] ParticleGridDeviceView device_view() noexcept;

    [[nodiscard]] std::size_t particle_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace particles::cuda
