#include "particles/cuda_particle_grid.hpp"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace particles::cuda {
namespace {

struct DeviceGrid {
    double extent_x{};
    double extent_y{};
    double extent_z{};
    double cell_width{};
    std::uint32_t dimension_x{};
    std::uint32_t dimension_y{};
    std::uint32_t dimension_z{};
    std::uint8_t periodic_x{};
    std::uint8_t periodic_y{};
    std::uint8_t periodic_z{};
};

__device__ double wrap_periodic_coordinate(
    double position,
    double extent
);

__device__ void advance_boundary_coordinate(
    double& position,
    double& velocity,
    double duration,
    double extent,
    std::uint8_t periodic,
    double diameter
);

void check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw CudaError(
            std::string(operation) + ": " + cudaGetErrorString(result)
        );
    }
}

void free_device(void* pointer) noexcept {
    if (pointer != nullptr) {
        static_cast<void>(cudaFree(pointer));
    }
}

template <typename Value>
void allocate(Value*& pointer, std::size_t count, const char* operation) {
    check(
        cudaMalloc(
            reinterpret_cast<void**>(&pointer),
            count * sizeof(Value)
        ),
        operation
    );
}

__device__ std::uint32_t coordinate(
    double position,
    double cell_width,
    std::uint32_t dimension
) {
    const double raw = floor(position / cell_width);
    const std::uint64_t unclamped =
        raw <= 0.0 ? 0ULL : static_cast<std::uint64_t>(raw);
    const std::uint64_t maximum =
        static_cast<std::uint64_t>(dimension - 1U);
    return static_cast<std::uint32_t>(
        unclamped < maximum ? unclamped : maximum
    );
}

__global__ void compute_cell_ids(
    const double* position_x,
    const double* position_y,
    const double* position_z,
    std::uint32_t* cell_ids,
    std::uint32_t* particle_indices,
    std::size_t particle_count,
    DeviceGrid grid
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= particle_count) {
        return;
    }

    const std::uint32_t x = coordinate(
        min(max(position_x[index], 0.0), grid.extent_x),
        grid.cell_width,
        grid.dimension_x
    );
    const std::uint32_t y = coordinate(
        min(max(position_y[index], 0.0), grid.extent_y),
        grid.cell_width,
        grid.dimension_y
    );
    const std::uint32_t z = coordinate(
        min(max(position_z[index], 0.0), grid.extent_z),
        grid.cell_width,
        grid.dimension_z
    );

    const std::uint64_t cell_id =
        static_cast<std::uint64_t>(x) +
        static_cast<std::uint64_t>(grid.dimension_x) *
            (
                static_cast<std::uint64_t>(y) +
                static_cast<std::uint64_t>(grid.dimension_y) *
                    static_cast<std::uint64_t>(z)
            );

    cell_ids[index] = static_cast<std::uint32_t>(cell_id);
    particle_indices[index] = static_cast<std::uint32_t>(index);
}

__global__ void advance_free_flight_kernel(
    double* position_x,
    double* position_y,
    double* position_z,
    const double* velocity_x,
    const double* velocity_y,
    const double* velocity_z,
    std::size_t particle_count,
    double delta_time
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= particle_count) {
        return;
    }

    position_x[index] += velocity_x[index] * delta_time;
    position_y[index] += velocity_y[index] * delta_time;
    position_z[index] += velocity_z[index] * delta_time;
}

__global__ void build_cell_ranges(
    const std::uint32_t* sorted_cell_ids,
    std::uint32_t* cell_start,
    std::uint32_t* cell_end,
    std::size_t particle_count
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= particle_count) {
        return;
    }

    const std::uint32_t cell_id = sorted_cell_ids[index];

    if (index == 0 || sorted_cell_ids[index - 1] != cell_id) {
        cell_start[cell_id] = static_cast<std::uint32_t>(index);
    }
    if (
        index + 1 == particle_count ||
        sorted_cell_ids[index + 1] != cell_id
    ) {
        cell_end[cell_id] = static_cast<std::uint32_t>(index + 1);
    }
}

__device__ std::uint64_t enumerate_candidate_pairs(
    std::uint32_t particle_index,
    const std::uint32_t* particle_cell_ids,
    const std::uint32_t* sorted_particle_indices,
    const std::uint32_t* cell_start,
    const std::uint32_t* cell_end,
    DeviceGrid grid,
    std::uint32_t* candidate_first,
    std::uint32_t* candidate_second,
    std::uint64_t output_offset
) {
    const std::uint32_t center_cell = particle_cell_ids[particle_index];
    const std::uint32_t center_x = center_cell % grid.dimension_x;
    const std::uint32_t yz = center_cell / grid.dimension_x;
    const std::uint32_t center_y = yz % grid.dimension_y;
    const std::uint32_t center_z = yz / grid.dimension_y;

    const std::uint32_t centers[3]{center_x, center_y, center_z};
    const std::uint32_t dimensions[3]{
        grid.dimension_x,
        grid.dimension_y,
        grid.dimension_z,
    };
    const std::uint8_t periodic[3]{
        grid.periodic_x,
        grid.periodic_y,
        grid.periodic_z,
    };

    std::uint32_t neighbor_cell_ids[27]{};
    int neighbor_cell_count = 0;
    std::uint64_t pair_count = 0;

    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
        for (int offset_y = -1; offset_y <= 1; ++offset_y) {
            for (int offset_x = -1; offset_x <= 1; ++offset_x) {
                const int offsets[3]{offset_x, offset_y, offset_z};
                std::uint32_t neighbor[3]{};
                bool valid = true;

                for (int axis = 0; axis < 3; ++axis) {
                    std::int64_t value =
                        static_cast<std::int64_t>(centers[axis]) +
                        offsets[axis];
                    const std::int64_t dimension =
                        static_cast<std::int64_t>(dimensions[axis]);

                    if (periodic[axis] != 0U) {
                        value %= dimension;
                        if (value < 0) {
                            value += dimension;
                        }
                    } else if (value < 0 || value >= dimension) {
                        valid = false;
                        break;
                    }

                    neighbor[axis] = static_cast<std::uint32_t>(value);
                }

                if (!valid) {
                    continue;
                }

                const std::uint32_t neighbor_cell =
                    neighbor[0] +
                    grid.dimension_x *
                        (neighbor[1] + grid.dimension_y * neighbor[2]);

                bool duplicate_cell = false;
                for (int seen = 0; seen < neighbor_cell_count; ++seen) {
                    if (neighbor_cell_ids[seen] == neighbor_cell) {
                        duplicate_cell = true;
                        break;
                    }
                }
                if (duplicate_cell) {
                    continue;
                }
                neighbor_cell_ids[neighbor_cell_count++] = neighbor_cell;

                const std::uint32_t start = cell_start[neighbor_cell];
                if (start == 0xFFFFFFFFU) {
                    continue;
                }
                const std::uint32_t end = cell_end[neighbor_cell];

                for (std::uint32_t sorted_index = start;
                     sorted_index < end;
                     ++sorted_index) {
                    const std::uint32_t other =
                        sorted_particle_indices[sorted_index];
                    if (particle_index >= other) {
                        continue;
                    }

                    if (
                        candidate_first != nullptr &&
                        candidate_second != nullptr
                    ) {
                        candidate_first[output_offset + pair_count] =
                            particle_index;
                        candidate_second[output_offset + pair_count] = other;
                    }
                    ++pair_count;
                }
            }
        }
    }

    return pair_count;
}

__global__ void count_candidate_pairs(
    const std::uint32_t* particle_cell_ids,
    const std::uint32_t* sorted_particle_indices,
    const std::uint32_t* cell_start,
    const std::uint32_t* cell_end,
    std::uint64_t* candidate_counts,
    std::size_t particle_count,
    DeviceGrid grid
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= particle_count) {
        return;
    }

    candidate_counts[index] = enumerate_candidate_pairs(
        static_cast<std::uint32_t>(index),
        particle_cell_ids,
        sorted_particle_indices,
        cell_start,
        cell_end,
        grid,
        nullptr,
        nullptr,
        0
    );
}

__global__ void write_candidate_pairs(
    const std::uint32_t* particle_cell_ids,
    const std::uint32_t* sorted_particle_indices,
    const std::uint32_t* cell_start,
    const std::uint32_t* cell_end,
    const std::uint64_t* candidate_offsets,
    std::uint32_t* candidate_first,
    std::uint32_t* candidate_second,
    std::size_t particle_count,
    DeviceGrid grid
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= particle_count) {
        return;
    }

    static_cast<void>(
        enumerate_candidate_pairs(
            static_cast<std::uint32_t>(index),
            particle_cell_ids,
            sorted_particle_indices,
            cell_start,
            cell_end,
            grid,
            candidate_first,
            candidate_second,
            candidate_offsets[index]
        )
    );
}

__device__ bool collision_time_for_image(
    double relative_x,
    double relative_y,
    double relative_z,
    double relative_velocity_x,
    double relative_velocity_y,
    double relative_velocity_z,
    double diameter,
    double horizon,
    double tolerance,
    double& collision_time
) {
    const double distance_squared =
        relative_x * relative_x +
        relative_y * relative_y +
        relative_z * relative_z;
    const double diameter_squared = diameter * diameter;
    const double scale = fmax(1.0, fmax(distance_squared, diameter_squared));
    double separation_term = distance_squared - diameter_squared;

    if (separation_term < -tolerance * scale) {
        return false;
    }
    separation_term = fmax(0.0, separation_term);

    const double speed_squared =
        relative_velocity_x * relative_velocity_x +
        relative_velocity_y * relative_velocity_y +
        relative_velocity_z * relative_velocity_z;
    if (speed_squared <= tolerance * tolerance) {
        return false;
    }

    const double approach =
        relative_x * relative_velocity_x +
        relative_y * relative_velocity_y +
        relative_z * relative_velocity_z;
    if (approach >= 0.0) {
        return false;
    }

    const double discriminant =
        approach * approach - speed_squared * separation_term;
    const double discriminant_scale =
        approach * approach +
        speed_squared * fabs(separation_term) +
        1.0;
    if (discriminant <= tolerance * discriminant_scale) {
        return false;
    }

    const double square_root = sqrt(fmax(0.0, discriminant));
    double time = separation_term == 0.0
        ? 0.0
        : separation_term / (-approach + square_root);

    if (time < -tolerance || time > horizon + tolerance) {
        return false;
    }

    collision_time = fmin(fmax(time, 0.0), horizon);
    return true;
}

__global__ void evaluate_candidate_collision_times(
    const double* position_x,
    const double* position_y,
    const double* position_z,
    const double* velocity_x,
    const double* velocity_y,
    const double* velocity_z,
    const std::uint32_t* candidate_first,
    const std::uint32_t* candidate_second,
    double* candidate_time,
    std::int8_t* candidate_shift_x,
    std::int8_t* candidate_shift_y,
    std::int8_t* candidate_shift_z,
    std::uint64_t* event_flags,
    std::size_t candidate_count,
    DeviceGrid grid,
    double diameter,
    double horizon,
    double tolerance
) {
    const std::size_t candidate_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (candidate_index >= candidate_count) {
        return;
    }

    const std::uint32_t first = candidate_first[candidate_index];
    const std::uint32_t second = candidate_second[candidate_index];
    const double relative_velocity_x =
        velocity_x[first] - velocity_x[second];
    const double relative_velocity_y =
        velocity_y[first] - velocity_y[second];
    const double relative_velocity_z =
        velocity_z[first] - velocity_z[second];

    const int lower_x = grid.periodic_x != 0U ? -1 : 0;
    const int upper_x = grid.periodic_x != 0U ? 1 : 0;
    const int lower_y = grid.periodic_y != 0U ? -1 : 0;
    const int upper_y = grid.periodic_y != 0U ? 1 : 0;
    const int lower_z = grid.periodic_z != 0U ? -1 : 0;
    const int upper_z = grid.periodic_z != 0U ? 1 : 0;

    bool found = false;
    double earliest_time = horizon;
    std::int8_t earliest_shift_x = 0;
    std::int8_t earliest_shift_y = 0;
    std::int8_t earliest_shift_z = 0;

    for (int shift_z = lower_z; shift_z <= upper_z; ++shift_z) {
        for (int shift_y = lower_y; shift_y <= upper_y; ++shift_y) {
            for (int shift_x = lower_x; shift_x <= upper_x; ++shift_x) {
                const double relative_x =
                    position_x[first] -
                    (
                        position_x[second] +
                        static_cast<double>(shift_x) * grid.extent_x
                    );
                const double relative_y =
                    position_y[first] -
                    (
                        position_y[second] +
                        static_cast<double>(shift_y) * grid.extent_y
                    );
                const double relative_z =
                    position_z[first] -
                    (
                        position_z[second] +
                        static_cast<double>(shift_z) * grid.extent_z
                    );

                double time = 0.0;
                if (
                    !collision_time_for_image(
                        relative_x,
                        relative_y,
                        relative_z,
                        relative_velocity_x,
                        relative_velocity_y,
                        relative_velocity_z,
                        diameter,
                        horizon,
                        tolerance,
                        time
                    )
                ) {
                    continue;
                }

                if (!found || time < earliest_time - tolerance) {
                    found = true;
                    earliest_time = time;
                    earliest_shift_x =
                        static_cast<std::int8_t>(shift_x);
                    earliest_shift_y =
                        static_cast<std::int8_t>(shift_y);
                    earliest_shift_z =
                        static_cast<std::int8_t>(shift_z);
                }
            }
        }
    }

    event_flags[candidate_index] = found ? 1ULL : 0ULL;
    if (found) {
        candidate_time[candidate_index] = earliest_time;
        candidate_shift_x[candidate_index] = earliest_shift_x;
        candidate_shift_y[candidate_index] = earliest_shift_y;
        candidate_shift_z[candidate_index] = earliest_shift_z;
    }
}

__global__ void compact_collision_events(
    const std::uint32_t* candidate_first,
    const std::uint32_t* candidate_second,
    const double* candidate_time,
    const std::int8_t* candidate_shift_x,
    const std::int8_t* candidate_shift_y,
    const std::int8_t* candidate_shift_z,
    const std::uint64_t* event_flags,
    const std::uint64_t* event_offsets,
    std::uint32_t* event_first,
    std::uint32_t* event_second,
    double* event_time,
    std::int8_t* event_shift_x,
    std::int8_t* event_shift_y,
    std::int8_t* event_shift_z,
    std::size_t candidate_count
) {
    const std::size_t candidate_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (
        candidate_index >= candidate_count ||
        event_flags[candidate_index] == 0ULL
    ) {
        return;
    }

    const std::uint64_t output_index = event_offsets[candidate_index];
    event_first[output_index] = candidate_first[candidate_index];
    event_second[output_index] = candidate_second[candidate_index];
    event_time[output_index] = candidate_time[candidate_index];
    event_shift_x[output_index] = candidate_shift_x[candidate_index];
    event_shift_y[output_index] = candidate_shift_y[candidate_index];
    event_shift_z[output_index] = candidate_shift_z[candidate_index];
}

__global__ void count_earliest_batch_participation(
    const std::uint32_t* event_first,
    const std::uint32_t* event_second,
    const double* event_time,
    std::uint32_t* particle_participation,
    std::size_t event_count,
    double earliest_time,
    double tolerance
) {
    const std::size_t event_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (
        event_index >= event_count ||
        event_time[event_index] - earliest_time > tolerance
    ) {
        return;
    }

    atomicAdd(&particle_participation[event_first[event_index]], 1U);
    atomicAdd(&particle_participation[event_second[event_index]], 1U);
}

__global__ void classify_earliest_batch_events(
    const std::uint32_t* event_first,
    const std::uint32_t* event_second,
    const double* event_time,
    const std::uint32_t* particle_participation,
    std::uint64_t* selection_flags,
    std::size_t event_count,
    double earliest_time,
    double tolerance,
    bool select_conflicting
) {
    const std::size_t event_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (event_index >= event_count) {
        return;
    }

    const bool in_batch =
        event_time[event_index] - earliest_time <= tolerance;
    bool conflicting = false;
    if (in_batch) {
        conflicting =
            particle_participation[event_first[event_index]] > 1U ||
            particle_participation[event_second[event_index]] > 1U;
    }

    selection_flags[event_index] =
        in_batch && conflicting == select_conflicting ? 1ULL : 0ULL;
}

__global__ void compact_event_indices(
    const std::uint64_t* selection_flags,
    const std::uint64_t* selection_offsets,
    std::uint32_t* output_indices,
    std::size_t event_count
) {
    const std::size_t event_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (
        event_index >= event_count ||
        selection_flags[event_index] == 0ULL
    ) {
        return;
    }

    output_indices[selection_offsets[event_index]] =
        static_cast<std::uint32_t>(event_index);
}

__global__ void initialize_parallel_window_selection(
    const double* particle_wall_time,
    std::uint64_t* particle_minimum_time_bits,
    std::uint64_t* particle_event_key,
    std::size_t particle_count,
    double window,
    double tolerance
) {
    const std::size_t particle_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (particle_index >= particle_count) {
        return;
    }

    const double wall_time = particle_wall_time[particle_index];
    const bool wall_in_window =
        isfinite(wall_time) &&
        wall_time <= window + tolerance;
    particle_minimum_time_bits[particle_index] =
        static_cast<std::uint64_t>(
            __double_as_longlong(
                wall_in_window ? fmin(wall_time, window) : INFINITY
            )
        );
    particle_event_key[particle_index] = 0xFFFFFFFFFFFFFFFFULL;
}

__global__ void select_parallel_window_minimum_times(
    const std::uint32_t* event_first,
    const std::uint32_t* event_second,
    const double* event_time,
    std::uint64_t* particle_minimum_time_bits,
    std::size_t event_count
) {
    const std::size_t event_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (event_index >= event_count) {
        return;
    }

    const std::uint64_t time_bits = static_cast<std::uint64_t>(
        __double_as_longlong(event_time[event_index])
    );
    atomicMin(
        reinterpret_cast<unsigned long long*>(
            &particle_minimum_time_bits[event_first[event_index]]
        ),
        static_cast<unsigned long long>(time_bits)
    );
    atomicMin(
        reinterpret_cast<unsigned long long*>(
            &particle_minimum_time_bits[event_second[event_index]]
        ),
        static_cast<unsigned long long>(time_bits)
    );
}

__global__ void select_parallel_window_wall_event_keys(
    const double* particle_wall_time,
    const std::uint64_t* particle_minimum_time_bits,
    std::uint64_t* particle_event_key,
    std::size_t particle_count,
    double window,
    double tolerance
) {
    const std::size_t particle_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (particle_index >= particle_count) {
        return;
    }

    const double wall_time = particle_wall_time[particle_index];
    const double minimum_time = __longlong_as_double(
        static_cast<long long>(
            particle_minimum_time_bits[particle_index]
        )
    );
    if (
        isfinite(wall_time) &&
        wall_time <= window + tolerance &&
        wall_time - minimum_time <= tolerance
    ) {
        // Zero is reserved for a reflecting-wall event. Every valid ordered
        // particle-pair key is positive, so a wall wins deterministic ties.
        particle_event_key[particle_index] = 0ULL;
    }
}

__global__ void select_parallel_window_event_keys(
    const std::uint32_t* event_first,
    const std::uint32_t* event_second,
    const double* event_time,
    const std::uint64_t* particle_minimum_time_bits,
    std::uint64_t* particle_event_key,
    std::size_t event_count,
    double tolerance
) {
    const std::size_t event_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (event_index >= event_count) {
        return;
    }

    const std::uint32_t first = event_first[event_index];
    const std::uint32_t second = event_second[event_index];
    const std::uint64_t event_key =
        (static_cast<std::uint64_t>(first) << 32U) |
        static_cast<std::uint64_t>(second);
    const double first_minimum = __longlong_as_double(
        static_cast<long long>(particle_minimum_time_bits[first])
    );
    const double second_minimum = __longlong_as_double(
        static_cast<long long>(particle_minimum_time_bits[second])
    );

    if (event_time[event_index] - first_minimum <= tolerance) {
        atomicMin(
            reinterpret_cast<unsigned long long*>(
                &particle_event_key[first]
            ),
            static_cast<unsigned long long>(event_key)
        );
    }
    if (event_time[event_index] - second_minimum <= tolerance) {
        atomicMin(
            reinterpret_cast<unsigned long long*>(
                &particle_event_key[second]
            ),
            static_cast<unsigned long long>(event_key)
        );
    }
}

__global__ void mark_mutual_parallel_window_events(
    const std::uint32_t* event_first,
    const std::uint32_t* event_second,
    const std::uint64_t* particle_event_key,
    std::uint64_t* event_flags,
    std::size_t event_count
) {
    const std::size_t event_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (event_index >= event_count) {
        return;
    }

    const std::uint32_t first = event_first[event_index];
    const std::uint32_t second = event_second[event_index];
    const std::uint64_t event_key =
        (static_cast<std::uint64_t>(first) << 32U) |
        static_cast<std::uint64_t>(second);
    event_flags[event_index] =
        particle_event_key[first] == event_key &&
        particle_event_key[second] == event_key
            ? 1ULL
            : 0ULL;
}

__global__ void mark_parallel_window_event_particles(
    const std::uint32_t* event_first,
    const std::uint32_t* event_second,
    const std::uint32_t* accepted_event_indices,
    std::uint32_t* particle_participation,
    std::size_t accepted_event_count
) {
    const std::size_t selection_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (selection_index >= accepted_event_count) {
        return;
    }

    const std::uint32_t event_index =
        accepted_event_indices[selection_index];
    particle_participation[event_first[event_index]] = 1U;
    particle_participation[event_second[event_index]] = 1U;
}

__global__ void advance_parallel_window_unselected_particles(
    double* position_x,
    double* position_y,
    double* position_z,
    double* velocity_x,
    double* velocity_y,
    double* velocity_z,
    const std::uint32_t* particle_participation,
    std::size_t particle_count,
    DeviceGrid grid,
    double diameter,
    double window
) {
    const std::size_t particle_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (
        particle_index >= particle_count ||
        particle_participation[particle_index] != 0U
    ) {
        return;
    }

    advance_boundary_coordinate(
        position_x[particle_index],
        velocity_x[particle_index],
        window,
        grid.extent_x,
        grid.periodic_x,
        diameter
    );
    advance_boundary_coordinate(
        position_y[particle_index],
        velocity_y[particle_index],
        window,
        grid.extent_y,
        grid.periodic_y,
        diameter
    );
    advance_boundary_coordinate(
        position_z[particle_index],
        velocity_z[particle_index],
        window,
        grid.extent_z,
        grid.periodic_z,
        diameter
    );
}

__global__ void resolve_parallel_window_events(
    double* position_x,
    double* position_y,
    double* position_z,
    double* velocity_x,
    double* velocity_y,
    double* velocity_z,
    const std::uint32_t* event_first,
    const std::uint32_t* event_second,
    const double* event_time,
    const std::int8_t* event_shift_x,
    const std::int8_t* event_shift_y,
    const std::int8_t* event_shift_z,
    const std::uint32_t* accepted_event_indices,
    std::size_t accepted_event_count,
    DeviceGrid grid,
    double diameter,
    double window
) {
    const std::size_t selection_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (selection_index >= accepted_event_count) {
        return;
    }

    const std::uint32_t event_index =
        accepted_event_indices[selection_index];
    const std::uint32_t first = event_first[event_index];
    const std::uint32_t second = event_second[event_index];
    const double remaining =
        fmax(0.0, window - event_time[event_index]);

    const double first_velocity_x = velocity_x[first];
    const double first_velocity_y = velocity_y[first];
    const double first_velocity_z = velocity_z[first];
    const double second_velocity_x = velocity_x[second];
    const double second_velocity_y = velocity_y[second];
    const double second_velocity_z = velocity_z[second];

    const double first_contact_x =
        position_x[first] +
        first_velocity_x * event_time[event_index];
    const double first_contact_y =
        position_y[first] +
        first_velocity_y * event_time[event_index];
    const double first_contact_z =
        position_z[first] +
        first_velocity_z * event_time[event_index];
    const double second_contact_x =
        position_x[second] +
        second_velocity_x * event_time[event_index] +
        static_cast<double>(event_shift_x[event_index]) * grid.extent_x;
    const double second_contact_y =
        position_y[second] +
        second_velocity_y * event_time[event_index] +
        static_cast<double>(event_shift_y[event_index]) * grid.extent_y;
    const double second_contact_z =
        position_z[second] +
        second_velocity_z * event_time[event_index] +
        static_cast<double>(event_shift_z[event_index]) * grid.extent_z;

    const double relative_x = first_contact_x - second_contact_x;
    const double relative_y = first_contact_y - second_contact_y;
    const double relative_z = first_contact_z - second_contact_z;
    const double inverse_distance = rsqrt(
        relative_x * relative_x +
        relative_y * relative_y +
        relative_z * relative_z
    );
    const double normal_x = relative_x * inverse_distance;
    const double normal_y = relative_y * inverse_distance;
    const double normal_z = relative_z * inverse_distance;
    const double normal_relative_speed =
        (first_velocity_x - second_velocity_x) * normal_x +
        (first_velocity_y - second_velocity_y) * normal_y +
        (first_velocity_z - second_velocity_z) * normal_z;
    const double impulse_x = normal_relative_speed * normal_x;
    const double impulse_y = normal_relative_speed * normal_y;
    const double impulse_z = normal_relative_speed * normal_z;

    const double first_post_x = first_velocity_x - impulse_x;
    const double first_post_y = first_velocity_y - impulse_y;
    const double first_post_z = first_velocity_z - impulse_z;
    const double second_post_x = second_velocity_x + impulse_x;
    const double second_post_y = second_velocity_y + impulse_y;
    const double second_post_z = second_velocity_z + impulse_z;

    position_x[first] = first_contact_x;
    position_y[first] = first_contact_y;
    position_z[first] = first_contact_z;
    position_x[second] = second_contact_x;
    position_y[second] = second_contact_y;
    position_z[second] = second_contact_z;
    velocity_x[first] = first_post_x;
    velocity_y[first] = first_post_y;
    velocity_z[first] = first_post_z;
    velocity_x[second] = second_post_x;
    velocity_y[second] = second_post_y;
    velocity_z[second] = second_post_z;

    advance_boundary_coordinate(
        position_x[first],
        velocity_x[first],
        remaining,
        grid.extent_x,
        grid.periodic_x,
        diameter
    );
    advance_boundary_coordinate(
        position_y[first],
        velocity_y[first],
        remaining,
        grid.extent_y,
        grid.periodic_y,
        diameter
    );
    advance_boundary_coordinate(
        position_z[first],
        velocity_z[first],
        remaining,
        grid.extent_z,
        grid.periodic_z,
        diameter
    );
    advance_boundary_coordinate(
        position_x[second],
        velocity_x[second],
        remaining,
        grid.extent_x,
        grid.periodic_x,
        diameter
    );
    advance_boundary_coordinate(
        position_y[second],
        velocity_y[second],
        remaining,
        grid.extent_y,
        grid.periodic_y,
        diameter
    );
    advance_boundary_coordinate(
        position_z[second],
        velocity_z[second],
        remaining,
        grid.extent_z,
        grid.periodic_z,
        diameter
    );
}

__global__ void measure_candidate_surface_gaps(
    const double* position_x,
    const double* position_y,
    const double* position_z,
    const std::uint32_t* candidate_first,
    const std::uint32_t* candidate_second,
    double* candidate_gap,
    std::uint64_t* overlap_flags,
    std::size_t candidate_count,
    DeviceGrid grid,
    double diameter,
    double tolerance
) {
    const std::size_t candidate_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (candidate_index >= candidate_count) {
        return;
    }

    const std::uint32_t first = candidate_first[candidate_index];
    const std::uint32_t second = candidate_second[candidate_index];
    double relative_x = position_x[first] - position_x[second];
    double relative_y = position_y[first] - position_y[second];
    double relative_z = position_z[first] - position_z[second];
    if (grid.periodic_x != 0U) {
        relative_x -= round(relative_x / grid.extent_x) * grid.extent_x;
    }
    if (grid.periodic_y != 0U) {
        relative_y -= round(relative_y / grid.extent_y) * grid.extent_y;
    }
    if (grid.periodic_z != 0U) {
        relative_z -= round(relative_z / grid.extent_z) * grid.extent_z;
    }

    const double gap = sqrt(
        relative_x * relative_x +
        relative_y * relative_y +
        relative_z * relative_z
    ) - diameter;
    candidate_gap[candidate_index] = gap;
    overlap_flags[candidate_index] =
        gap < -tolerance * fmax(1.0, diameter) ? 1ULL : 0ULL;
}

__global__ void write_conflicting_event_keys(
    const std::uint32_t* conflicting_event_indices,
    const std::uint32_t* event_first,
    const std::uint32_t* event_second,
    std::uint64_t* conflict_keys,
    std::size_t conflicting_event_count
) {
    const std::size_t conflict_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (conflict_index >= conflicting_event_count) {
        return;
    }

    const std::uint32_t event_index =
        conflicting_event_indices[conflict_index];
    conflict_keys[conflict_index] =
        (static_cast<std::uint64_t>(event_first[event_index]) << 32U) |
        static_cast<std::uint64_t>(event_second[event_index]);
}

__global__ void find_minimum_conflict_event(
    const std::uint32_t* conflicting_event_indices,
    const std::uint32_t* event_first,
    const std::uint32_t* event_second,
    const std::uint64_t* minimum_conflict_key,
    std::uint32_t* selected_event_index,
    std::size_t conflicting_event_count
) {
    const std::size_t conflict_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (conflict_index >= conflicting_event_count) {
        return;
    }

    const std::uint32_t event_index =
        conflicting_event_indices[conflict_index];
    const std::uint64_t key =
        (static_cast<std::uint64_t>(event_first[event_index]) << 32U) |
        static_cast<std::uint64_t>(event_second[event_index]);
    if (key == *minimum_conflict_key) {
        *selected_event_index = event_index;
    }
}

__global__ void validate_independent_collision_events(
    const double* position_x,
    const double* position_y,
    const double* position_z,
    const double* velocity_x,
    const double* velocity_y,
    const double* velocity_z,
    const std::uint32_t* event_first,
    const std::uint32_t* event_second,
    const double* event_time,
    const std::int8_t* event_shift_x,
    const std::int8_t* event_shift_y,
    const std::int8_t* event_shift_z,
    const std::uint32_t* independent_event_indices,
    std::size_t independent_event_count,
    DeviceGrid grid,
    double diameter,
    double tolerance,
    std::uint32_t* failure_count
) {
    const std::size_t selection_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (selection_index >= independent_event_count) {
        return;
    }

    const std::uint32_t event_index =
        independent_event_indices[selection_index];
    const std::uint32_t first = event_first[event_index];
    const std::uint32_t second = event_second[event_index];
    const double time = event_time[event_index];

    const double relative_x =
        position_x[first] + velocity_x[first] * time -
        (
            position_x[second] + velocity_x[second] * time +
            static_cast<double>(event_shift_x[event_index]) * grid.extent_x
        );
    const double relative_y =
        position_y[first] + velocity_y[first] * time -
        (
            position_y[second] + velocity_y[second] * time +
            static_cast<double>(event_shift_y[event_index]) * grid.extent_y
        );
    const double relative_z =
        position_z[first] + velocity_z[first] * time -
        (
            position_z[second] + velocity_z[second] * time +
            static_cast<double>(event_shift_z[event_index]) * grid.extent_z
        );
    const double distance_squared =
        relative_x * relative_x +
        relative_y * relative_y +
        relative_z * relative_z;
    const double diameter_squared = diameter * diameter;
    const double scale =
        fmax(1.0, fmax(distance_squared, diameter_squared));
    const double contact_error = distance_squared - diameter_squared;
    const double distance = sqrt(fmax(0.0, distance_squared));
    const double relative_velocity_x =
        velocity_x[first] - velocity_x[second];
    const double relative_velocity_y =
        velocity_y[first] - velocity_y[second];
    const double relative_velocity_z =
        velocity_z[first] - velocity_z[second];
    const double normal_relative_speed = distance > tolerance
        ? (
              relative_velocity_x * relative_x +
              relative_velocity_y * relative_y +
              relative_velocity_z * relative_z
          ) / distance
        : 0.0;

    if (
        fabs(contact_error) > tolerance * scale ||
        distance <= tolerance ||
        normal_relative_speed >= -tolerance
    ) {
        atomicAdd(failure_count, 1U);
    }
}

__global__ void resolve_independent_collision_events(
    const double* position_x,
    const double* position_y,
    const double* position_z,
    double* velocity_x,
    double* velocity_y,
    double* velocity_z,
    const std::uint32_t* event_first,
    const std::uint32_t* event_second,
    const double* event_time,
    const std::int8_t* event_shift_x,
    const std::int8_t* event_shift_y,
    const std::int8_t* event_shift_z,
    const std::uint32_t* independent_event_indices,
    std::size_t independent_event_count,
    DeviceGrid grid,
    double earliest_time
) {
    const std::size_t selection_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (selection_index >= independent_event_count) {
        return;
    }

    const std::uint32_t event_index =
        independent_event_indices[selection_index];
    const std::uint32_t first = event_first[event_index];
    const std::uint32_t second = event_second[event_index];
    const double contact_offset =
        fmax(0.0, event_time[event_index] - earliest_time);

    const double first_velocity_x = velocity_x[first];
    const double first_velocity_y = velocity_y[first];
    const double first_velocity_z = velocity_z[first];
    const double second_velocity_x = velocity_x[second];
    const double second_velocity_y = velocity_y[second];
    const double second_velocity_z = velocity_z[second];

    const double relative_x =
        position_x[first] + first_velocity_x * contact_offset -
        (
            position_x[second] +
            second_velocity_x * contact_offset +
            static_cast<double>(event_shift_x[event_index]) * grid.extent_x
        );
    const double relative_y =
        position_y[first] + first_velocity_y * contact_offset -
        (
            position_y[second] +
            second_velocity_y * contact_offset +
            static_cast<double>(event_shift_y[event_index]) * grid.extent_y
        );
    const double relative_z =
        position_z[first] + first_velocity_z * contact_offset -
        (
            position_z[second] +
            second_velocity_z * contact_offset +
            static_cast<double>(event_shift_z[event_index]) * grid.extent_z
        );
    const double inverse_distance = rsqrt(
        relative_x * relative_x +
        relative_y * relative_y +
        relative_z * relative_z
    );
    const double normal_x = relative_x * inverse_distance;
    const double normal_y = relative_y * inverse_distance;
    const double normal_z = relative_z * inverse_distance;
    const double normal_relative_speed =
        (first_velocity_x - second_velocity_x) * normal_x +
        (first_velocity_y - second_velocity_y) * normal_y +
        (first_velocity_z - second_velocity_z) * normal_z;
    const double impulse_x = normal_relative_speed * normal_x;
    const double impulse_y = normal_relative_speed * normal_y;
    const double impulse_z = normal_relative_speed * normal_z;

    velocity_x[first] = first_velocity_x - impulse_x;
    velocity_y[first] = first_velocity_y - impulse_y;
    velocity_z[first] = first_velocity_z - impulse_z;
    velocity_x[second] = second_velocity_x + impulse_x;
    velocity_y[second] = second_velocity_y + impulse_y;
    velocity_z[second] = second_velocity_z + impulse_z;
}

__device__ double wrap_periodic_coordinate(double position, double extent) {
    double wrapped = position - floor(position / extent) * extent;
    if (wrapped >= extent) {
        wrapped = 0.0;
    }
    return wrapped;
}

__device__ void advance_boundary_coordinate(
    double& position,
    double& velocity,
    double duration,
    double extent,
    std::uint8_t periodic,
    double diameter
) {
    if (periodic != 0U) {
        position = wrap_periodic_coordinate(
            position + velocity * duration,
            extent
        );
        return;
    }

    const double lower = 0.5 * diameter;
    const double width = extent - diameter;
    const double period = 2.0 * width;
    const double unfolded =
        position - lower + velocity * duration;
    double phase = fmod(unfolded, period);
    if (phase < 0.0) {
        phase += period;
    }

    if (phase == 0.0) {
        position = lower;
        velocity = fabs(velocity);
    } else if (phase == width) {
        position = extent - lower;
        velocity = -fabs(velocity);
    } else if (phase < width) {
        position = lower + phase;
    } else {
        position = lower + period - phase;
        velocity = -velocity;
    }
}

__global__ void normalize_grid_positions(
    double* position_x,
    double* position_y,
    double* position_z,
    std::size_t particle_count,
    DeviceGrid grid,
    double diameter
) {
    const std::size_t particle_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (particle_index >= particle_count) {
        return;
    }

    if (grid.periodic_x != 0U) {
        position_x[particle_index] = wrap_periodic_coordinate(
            position_x[particle_index],
            grid.extent_x
        );
    } else {
        position_x[particle_index] = min(
            max(position_x[particle_index], 0.5 * diameter),
            grid.extent_x - 0.5 * diameter
        );
    }
    if (grid.periodic_y != 0U) {
        position_y[particle_index] = wrap_periodic_coordinate(
            position_y[particle_index],
            grid.extent_y
        );
    } else {
        position_y[particle_index] = min(
            max(position_y[particle_index], 0.5 * diameter),
            grid.extent_y - 0.5 * diameter
        );
    }
    if (grid.periodic_z != 0U) {
        position_z[particle_index] = wrap_periodic_coordinate(
            position_z[particle_index],
            grid.extent_z
        );
    } else {
        position_z[particle_index] = min(
            max(position_z[particle_index], 0.5 * diameter),
            grid.extent_z - 0.5 * diameter
        );
    }
}

__global__ void compute_reflecting_wall_times(
    const double* position_x,
    const double* position_y,
    const double* position_z,
    const double* velocity_x,
    const double* velocity_y,
    const double* velocity_z,
    double* particle_wall_time,
    std::uint8_t* particle_wall_axis_mask,
    std::uint32_t* failure_count,
    std::size_t particle_count,
    DeviceGrid grid,
    double diameter,
    double horizon,
    double tolerance
) {
    const std::size_t particle_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (particle_index >= particle_count) {
        return;
    }

    const double positions[3]{
        position_x[particle_index],
        position_y[particle_index],
        position_z[particle_index],
    };
    const double velocities[3]{
        velocity_x[particle_index],
        velocity_y[particle_index],
        velocity_z[particle_index],
    };
    const double extents[3]{
        grid.extent_x,
        grid.extent_y,
        grid.extent_z,
    };
    const std::uint8_t periodic[3]{
        grid.periodic_x,
        grid.periodic_y,
        grid.periodic_z,
    };

    const double radius = 0.5 * diameter;
    double earliest_time = INFINITY;
    std::uint8_t axis_mask = 0U;

    for (int axis = 0; axis < 3; ++axis) {
        if (periodic[axis] != 0U) {
            continue;
        }

        const double position = positions[axis];
        const double velocity = velocities[axis];
        const double extent = extents[axis];
        const double lower = radius;
        const double upper = extent - radius;
        const double geometric_tolerance =
            tolerance * fmax(1.0, extent * extent);

        if (
            !isfinite(position) ||
            !isfinite(velocity) ||
            position < lower - geometric_tolerance ||
            position > upper + geometric_tolerance
        ) {
            atomicAdd(failure_count, 1U);
            continue;
        }
        if (fabs(velocity) <= tolerance) {
            continue;
        }

        double wall_time = velocity > 0.0
            ? (upper - position) / velocity
            : (lower - position) / velocity;
        if (wall_time < -tolerance) {
            atomicAdd(failure_count, 1U);
            continue;
        }
        wall_time = fmax(0.0, wall_time);
        if (wall_time > horizon + tolerance) {
            continue;
        }
        wall_time = fmin(wall_time, horizon);

        const std::uint8_t axis_bit =
            static_cast<std::uint8_t>(1U << axis);
        if (wall_time < earliest_time - tolerance) {
            earliest_time = wall_time;
            axis_mask = axis_bit;
        } else if (fabs(wall_time - earliest_time) <= tolerance) {
            axis_mask = static_cast<std::uint8_t>(
                axis_mask | axis_bit
            );
        }
    }

    particle_wall_time[particle_index] = earliest_time;
    particle_wall_axis_mask[particle_index] = axis_mask;
}

__global__ void select_reflecting_wall_batch(
    const double* particle_wall_time,
    std::uint64_t* selection_flags,
    std::size_t particle_count,
    double earliest_time,
    double horizon,
    double tolerance
) {
    const std::size_t particle_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (particle_index >= particle_count) {
        return;
    }

    const double wall_time = particle_wall_time[particle_index];
    selection_flags[particle_index] =
        isfinite(wall_time) &&
        wall_time <= horizon + tolerance &&
        wall_time - earliest_time <= tolerance
            ? 1ULL
            : 0ULL;
}

__global__ void compact_reflecting_wall_batch(
    const double* particle_wall_time,
    const std::uint8_t* particle_wall_axis_mask,
    const std::uint64_t* selection_flags,
    const std::uint64_t* selection_offsets,
    std::uint32_t* wall_event_particle,
    double* wall_event_time,
    std::uint8_t* wall_event_axis_mask,
    std::uint32_t* reflected_axis_count,
    std::size_t particle_count,
    double horizon
) {
    const std::size_t particle_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (
        particle_index >= particle_count ||
        selection_flags[particle_index] == 0ULL
    ) {
        return;
    }

    const std::uint64_t output_index =
        selection_offsets[particle_index];
    const std::uint8_t axis_mask =
        particle_wall_axis_mask[particle_index];
    wall_event_particle[output_index] =
        static_cast<std::uint32_t>(particle_index);
    wall_event_time[output_index] =
        fmin(particle_wall_time[particle_index], horizon);
    wall_event_axis_mask[output_index] = axis_mask;
    atomicAdd(
        reflected_axis_count,
        static_cast<std::uint32_t>(
            (axis_mask & 0x01U) != 0U
        ) +
        static_cast<std::uint32_t>(
            (axis_mask & 0x02U) != 0U
        ) +
        static_cast<std::uint32_t>(
            (axis_mask & 0x04U) != 0U
        )
    );
}

__global__ void resolve_reflecting_wall_batch(
    double* position_x,
    double* position_y,
    double* position_z,
    double* velocity_x,
    double* velocity_y,
    double* velocity_z,
    const std::uint32_t* wall_event_particle,
    const std::uint8_t* wall_event_axis_mask,
    std::size_t wall_event_count,
    DeviceGrid grid,
    double diameter
) {
    const std::size_t event_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (event_index >= wall_event_count) {
        return;
    }

    const std::uint32_t particle =
        wall_event_particle[event_index];
    const std::uint8_t axis_mask =
        wall_event_axis_mask[event_index];
    const double radius = 0.5 * diameter;

    if ((axis_mask & 0x01U) != 0U) {
        position_x[particle] = velocity_x[particle] > 0.0
            ? grid.extent_x - radius
            : radius;
        velocity_x[particle] = -velocity_x[particle];
    }
    if ((axis_mask & 0x02U) != 0U) {
        position_y[particle] = velocity_y[particle] > 0.0
            ? grid.extent_y - radius
            : radius;
        velocity_y[particle] = -velocity_y[particle];
    }
    if ((axis_mask & 0x04U) != 0U) {
        position_z[particle] = velocity_z[particle] > 0.0
            ? grid.extent_z - radius
            : radius;
        velocity_z[particle] = -velocity_z[particle];
    }
}

__global__ void validate_grid_free_flight(
    const double* position_x,
    const double* position_y,
    const double* position_z,
    const double* velocity_x,
    const double* velocity_y,
    const double* velocity_z,
    std::uint32_t* failure_count,
    std::size_t particle_count,
    DeviceGrid grid,
    double delta_time,
    double diameter,
    double tolerance
) {
    const std::size_t particle_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (particle_index >= particle_count) {
        return;
    }

    const double positions[3]{
        position_x[particle_index] +
            velocity_x[particle_index] * delta_time,
        position_y[particle_index] +
            velocity_y[particle_index] * delta_time,
        position_z[particle_index] +
            velocity_z[particle_index] * delta_time,
    };
    const double extents[3]{
        grid.extent_x,
        grid.extent_y,
        grid.extent_z,
    };
    const std::uint8_t periodic[3]{
        grid.periodic_x,
        grid.periodic_y,
        grid.periodic_z,
    };
    const double radius = 0.5 * diameter;

    for (int axis = 0; axis < 3; ++axis) {
        if (periodic[axis] != 0U) {
            continue;
        }
        const double geometric_tolerance =
            tolerance * fmax(1.0, extents[axis] * extents[axis]);
        if (
            positions[axis] < radius - geometric_tolerance ||
            positions[axis] >
                extents[axis] - radius + geometric_tolerance
        ) {
            atomicAdd(failure_count, 1U);
            return;
        }
    }
}

__global__ void compute_particle_speeds(
    const double* velocity_x,
    const double* velocity_y,
    const double* velocity_z,
    double* particle_speed,
    std::size_t particle_count
) {
    const std::size_t particle_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (particle_index >= particle_count) {
        return;
    }

    particle_speed[particle_index] = sqrt(
        velocity_x[particle_index] * velocity_x[particle_index] +
        velocity_y[particle_index] * velocity_y[particle_index] +
        velocity_z[particle_index] * velocity_z[particle_index]
    );
}

}  // namespace

struct ParticleGrid::Impl {
    explicit Impl(std::size_t count)
        : particle_count(count) {
        if (particle_count == 0) {
            throw std::invalid_argument(
                "CUDA particle storage requires at least one particle."
            );
        }
        if (
            particle_count >=
            static_cast<std::size_t>(std::numeric_limits<int>::max())
        ) {
            throw std::overflow_error(
                "CUDA scan requires fewer than INT_MAX particles."
            );
        }

        try {
            allocate(position_x, particle_count, "cudaMalloc position_x");
            allocate(position_y, particle_count, "cudaMalloc position_y");
            allocate(position_z, particle_count, "cudaMalloc position_z");
            allocate(velocity_x, particle_count, "cudaMalloc velocity_x");
            allocate(velocity_y, particle_count, "cudaMalloc velocity_y");
            allocate(velocity_z, particle_count, "cudaMalloc velocity_z");
            allocate(
                checkpoint_position_x,
                particle_count,
                "cudaMalloc checkpoint_position_x"
            );
            allocate(
                checkpoint_position_y,
                particle_count,
                "cudaMalloc checkpoint_position_y"
            );
            allocate(
                checkpoint_position_z,
                particle_count,
                "cudaMalloc checkpoint_position_z"
            );
            allocate(
                checkpoint_velocity_x,
                particle_count,
                "cudaMalloc checkpoint_velocity_x"
            );
            allocate(
                checkpoint_velocity_y,
                particle_count,
                "cudaMalloc checkpoint_velocity_y"
            );
            allocate(
                checkpoint_velocity_z,
                particle_count,
                "cudaMalloc checkpoint_velocity_z"
            );

            allocate(cell_ids, particle_count, "cudaMalloc cell_ids");
            allocate(
                particle_indices,
                particle_count,
                "cudaMalloc particle_indices"
            );
            allocate(
                sorted_cell_ids,
                particle_count,
                "cudaMalloc sorted_cell_ids"
            );
            allocate(
                sorted_particle_indices,
                particle_count,
                "cudaMalloc sorted_particle_indices"
            );
            allocate(
                candidate_counts,
                particle_count + 1U,
                "cudaMalloc candidate_counts"
            );
            allocate(
                candidate_offsets,
                particle_count + 1U,
                "cudaMalloc candidate_offsets"
            );
            allocate(
                particle_batch_participation,
                particle_count,
                "cudaMalloc particle_batch_participation"
            );
            allocate(
                parallel_window_minimum_time_bits,
                particle_count,
                "cudaMalloc parallel_window_minimum_time_bits"
            );
            allocate(
                parallel_window_event_key,
                particle_count,
                "cudaMalloc parallel_window_event_key"
            );
            allocate(
                minimum_event_time,
                1U,
                "cudaMalloc minimum_event_time"
            );
            allocate(
                collision_validation_failures,
                1U,
                "cudaMalloc collision_validation_failures"
            );
            allocate(
                particle_wall_time,
                particle_count,
                "cudaMalloc particle_wall_time"
            );
            allocate(
                particle_wall_axis_mask,
                particle_count,
                "cudaMalloc particle_wall_axis_mask"
            );
            allocate(
                minimum_wall_time,
                1U,
                "cudaMalloc minimum_wall_time"
            );
            allocate(
                wall_event_particle,
                particle_count,
                "cudaMalloc wall_event_particle"
            );
            allocate(
                wall_event_time,
                particle_count,
                "cudaMalloc wall_event_time"
            );
            allocate(
                wall_event_axis_mask,
                particle_count,
                "cudaMalloc wall_event_axis_mask"
            );
            allocate(
                reflected_axis_count_device,
                1U,
                "cudaMalloc reflected_axis_count"
            );
            allocate(
                minimum_conflict_key,
                1U,
                "cudaMalloc minimum_conflict_key"
            );
            allocate(
                selected_conflict_event_index,
                1U,
                "cudaMalloc selected_conflict_event_index"
            );
        } catch (...) {
            free_device(position_x);
            free_device(position_y);
            free_device(position_z);
            free_device(velocity_x);
            free_device(velocity_y);
            free_device(velocity_z);
            free_device(checkpoint_position_x);
            free_device(checkpoint_position_y);
            free_device(checkpoint_position_z);
            free_device(checkpoint_velocity_x);
            free_device(checkpoint_velocity_y);
            free_device(checkpoint_velocity_z);
            free_device(cell_ids);
            free_device(particle_indices);
            free_device(sorted_cell_ids);
            free_device(sorted_particle_indices);
            free_device(candidate_counts);
            free_device(candidate_offsets);
            free_device(particle_batch_participation);
            free_device(parallel_window_minimum_time_bits);
            free_device(parallel_window_event_key);
            free_device(minimum_event_time);
            free_device(collision_validation_failures);
            free_device(particle_wall_time);
            free_device(particle_wall_axis_mask);
            free_device(minimum_wall_time);
            free_device(wall_event_particle);
            free_device(wall_event_time);
            free_device(wall_event_axis_mask);
            free_device(reflected_axis_count_device);
            free_device(minimum_conflict_key);
            free_device(selected_conflict_event_index);
            throw;
        }
    }

    ~Impl() {
        free_device(position_x);
        free_device(position_y);
        free_device(position_z);
        free_device(velocity_x);
        free_device(velocity_y);
        free_device(velocity_z);
        free_device(checkpoint_position_x);
        free_device(checkpoint_position_y);
        free_device(checkpoint_position_z);
        free_device(checkpoint_velocity_x);
        free_device(checkpoint_velocity_y);
        free_device(checkpoint_velocity_z);
        free_device(cell_ids);
        free_device(particle_indices);
        free_device(sorted_cell_ids);
        free_device(sorted_particle_indices);
        free_device(cell_start);
        free_device(cell_end);
        free_device(sort_temporary_storage);
        free_device(candidate_counts);
        free_device(candidate_offsets);
        free_device(scan_temporary_storage);
        free_device(candidate_first);
        free_device(candidate_second);
        free_device(candidate_collision_time);
        free_device(candidate_shift_x);
        free_device(candidate_shift_y);
        free_device(candidate_shift_z);
        free_device(event_flags);
        free_device(event_offsets);
        free_device(event_first);
        free_device(event_second);
        free_device(event_time);
        free_device(event_shift_x);
        free_device(event_shift_y);
        free_device(event_shift_z);
        free_device(particle_batch_participation);
        free_device(parallel_window_minimum_time_bits);
        free_device(parallel_window_event_key);
        free_device(minimum_event_time);
        free_device(collision_validation_failures);
        free_device(independent_event_indices);
        free_device(conflicting_event_indices);
        free_device(particle_wall_time);
        free_device(particle_wall_axis_mask);
        free_device(minimum_wall_time);
        free_device(wall_event_particle);
        free_device(wall_event_time);
        free_device(wall_event_axis_mask);
        free_device(reflected_axis_count_device);
        free_device(minimum_conflict_key);
        free_device(selected_conflict_event_index);
    }

    std::size_t particle_count{};

    double* position_x{};
    double* position_y{};
    double* position_z{};
    double* velocity_x{};
    double* velocity_y{};
    double* velocity_z{};
    double* checkpoint_position_x{};
    double* checkpoint_position_y{};
    double* checkpoint_position_z{};
    double* checkpoint_velocity_x{};
    double* checkpoint_velocity_y{};
    double* checkpoint_velocity_z{};

    std::uint32_t* cell_ids{};
    std::uint32_t* particle_indices{};
    std::uint32_t* sorted_cell_ids{};
    std::uint32_t* sorted_particle_indices{};
    std::uint32_t* cell_start{};
    std::uint32_t* cell_end{};

    void* sort_temporary_storage{};
    std::size_t sort_temporary_bytes{};
    std::uint64_t* candidate_counts{};
    std::uint64_t* candidate_offsets{};
    void* scan_temporary_storage{};
    std::size_t scan_temporary_bytes{};
    std::uint32_t* candidate_first{};
    std::uint32_t* candidate_second{};
    double* candidate_collision_time{};
    std::int8_t* candidate_shift_x{};
    std::int8_t* candidate_shift_y{};
    std::int8_t* candidate_shift_z{};
    std::uint64_t* event_flags{};
    std::uint64_t* event_offsets{};
    std::size_t candidate_capacity{};
    std::size_t candidate_count{};
    std::uint32_t* event_first{};
    std::uint32_t* event_second{};
    double* event_time{};
    std::int8_t* event_shift_x{};
    std::int8_t* event_shift_y{};
    std::int8_t* event_shift_z{};
    std::size_t event_capacity{};
    std::size_t event_count{};
    std::uint32_t* particle_batch_participation{};
    std::uint64_t* parallel_window_minimum_time_bits{};
    std::uint64_t* parallel_window_event_key{};
    double* minimum_event_time{};
    std::uint32_t* collision_validation_failures{};
    std::uint32_t* independent_event_indices{};
    std::uint32_t* conflicting_event_indices{};
    std::size_t batch_index_capacity{};
    std::size_t independent_event_count{};
    std::size_t conflicting_event_count{};
    double earliest_event_time{std::numeric_limits<double>::infinity()};
    double* particle_wall_time{};
    std::uint8_t* particle_wall_axis_mask{};
    double* minimum_wall_time{};
    std::uint32_t* wall_event_particle{};
    double* wall_event_time{};
    std::uint8_t* wall_event_axis_mask{};
    std::uint32_t* reflected_axis_count_device{};
    std::uint64_t* minimum_conflict_key{};
    std::uint32_t* selected_conflict_event_index{};
    std::size_t wall_event_count{};
    std::size_t reflected_axis_count{};
    double earliest_wall_time{std::numeric_limits<double>::infinity()};
    std::uint32_t allocated_cell_count{};

    UniformGrid grid{};
    bool particles_uploaded{};
    bool cell_list_built{};
    bool candidates_built{};
    bool events_built{};
    bool collision_batch_built{};
    bool wall_batch_built{};
};

ParticleGrid::ParticleGrid(std::size_t particle_count)
    : impl_(std::make_unique<Impl>(particle_count)) {}

ParticleGrid::~ParticleGrid() = default;
ParticleGrid::ParticleGrid(ParticleGrid&&) noexcept = default;
ParticleGrid& ParticleGrid::operator=(ParticleGrid&&) noexcept = default;

void ParticleGrid::upload(const std::vector<Particle>& particles) {
    if (particles.size() != impl_->particle_count) {
        throw std::invalid_argument(
            "Uploaded particle count does not match CUDA storage."
        );
    }

    std::vector<double> position_x(particles.size());
    std::vector<double> position_y(particles.size());
    std::vector<double> position_z(particles.size());
    std::vector<double> velocity_x(particles.size());
    std::vector<double> velocity_y(particles.size());
    std::vector<double> velocity_z(particles.size());

    for (std::size_t index = 0; index < particles.size(); ++index) {
        position_x[index] = particles[index].position.x;
        position_y[index] = particles[index].position.y;
        position_z[index] = particles[index].position.z;
        velocity_x[index] = particles[index].velocity.x;
        velocity_y[index] = particles[index].velocity.y;
        velocity_z[index] = particles[index].velocity.z;
    }

    const std::size_t bytes = particles.size() * sizeof(double);
    check(
        cudaMemcpy(
            impl_->position_x,
            position_x.data(),
            bytes,
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy position_x"
    );
    check(
        cudaMemcpy(
            impl_->position_y,
            position_y.data(),
            bytes,
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy position_y"
    );
    check(
        cudaMemcpy(
            impl_->position_z,
            position_z.data(),
            bytes,
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy position_z"
    );
    check(
        cudaMemcpy(
            impl_->velocity_x,
            velocity_x.data(),
            bytes,
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy velocity_x"
    );
    check(
        cudaMemcpy(
            impl_->velocity_y,
            velocity_y.data(),
            bytes,
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy velocity_y"
    );
    check(
        cudaMemcpy(
            impl_->velocity_z,
            velocity_z.data(),
            bytes,
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy velocity_z"
    );

    impl_->particles_uploaded = true;
    impl_->cell_list_built = false;
    impl_->candidates_built = false;
    impl_->events_built = false;
    impl_->collision_batch_built = false;
    impl_->wall_batch_built = false;
    impl_->candidate_count = 0;
    impl_->event_count = 0;
    impl_->independent_event_count = 0;
    impl_->conflicting_event_count = 0;
    impl_->earliest_event_time = std::numeric_limits<double>::infinity();
    impl_->wall_event_count = 0;
    impl_->reflected_axis_count = 0;
    impl_->earliest_wall_time = std::numeric_limits<double>::infinity();
}

void ParticleGrid::advance_free_flight(double delta_time) {
    if (!impl_->particles_uploaded) {
        throw std::logic_error(
            "Particle data must be uploaded before CUDA free flight."
        );
    }
    if (!std::isfinite(delta_time) || delta_time < 0.0) {
        throw std::invalid_argument(
            "CUDA free-flight time must be finite and non-negative."
        );
    }
    if (delta_time == 0.0) {
        return;
    }

    constexpr unsigned int block_size = 256;
    const unsigned int block_count = static_cast<unsigned int>(
        (impl_->particle_count + block_size - 1U) / block_size
    );

    advance_free_flight_kernel<<<block_count, block_size>>>(
        impl_->position_x,
        impl_->position_y,
        impl_->position_z,
        impl_->velocity_x,
        impl_->velocity_y,
        impl_->velocity_z,
        impl_->particle_count,
        delta_time
    );
    check(cudaGetLastError(), "advance_free_flight kernel launch");
    check(cudaDeviceSynchronize(), "CUDA free-flight synchronization");

    impl_->cell_list_built = false;
    impl_->candidates_built = false;
    impl_->events_built = false;
    impl_->collision_batch_built = false;
    impl_->wall_batch_built = false;
    impl_->candidate_count = 0;
    impl_->event_count = 0;
    impl_->independent_event_count = 0;
    impl_->conflicting_event_count = 0;
    impl_->earliest_event_time = std::numeric_limits<double>::infinity();
    impl_->wall_event_count = 0;
    impl_->reflected_axis_count = 0;
    impl_->earliest_wall_time = std::numeric_limits<double>::infinity();
}

void ParticleGrid::build_cell_list(const UniformGrid& grid) {
    if (!impl_->particles_uploaded) {
        throw std::logic_error(
            "Particle data must be uploaded before building a CUDA cell list."
        );
    }
    if (grid.cell_count == 0) {
        throw std::invalid_argument("CUDA grid must contain at least one cell.");
    }

    if (impl_->allocated_cell_count != grid.cell_count) {
        free_device(impl_->cell_start);
        free_device(impl_->cell_end);
        impl_->cell_start = nullptr;
        impl_->cell_end = nullptr;

        allocate(
            impl_->cell_start,
            grid.cell_count,
            "cudaMalloc cell_start"
        );
        allocate(
            impl_->cell_end,
            grid.cell_count,
            "cudaMalloc cell_end"
        );
        impl_->allocated_cell_count = grid.cell_count;
    }

    constexpr unsigned int block_size = 256;
    const unsigned int block_count = static_cast<unsigned int>(
        (impl_->particle_count + block_size - 1U) / block_size
    );
    const DeviceGrid device_grid{
        .extent_x = grid.extent.x,
        .extent_y = grid.extent.y,
        .extent_z = grid.extent.z,
        .cell_width = grid.cell_width,
        .dimension_x = grid.dimensions[0],
        .dimension_y = grid.dimensions[1],
        .dimension_z = grid.dimensions[2],
        .periodic_x = static_cast<std::uint8_t>(
            grid.boundary[0] == BoundaryKind::periodic
        ),
        .periodic_y = static_cast<std::uint8_t>(
            grid.boundary[1] == BoundaryKind::periodic
        ),
        .periodic_z = static_cast<std::uint8_t>(
            grid.boundary[2] == BoundaryKind::periodic
        ),
    };

    compute_cell_ids<<<block_count, block_size>>>(
        impl_->position_x,
        impl_->position_y,
        impl_->position_z,
        impl_->cell_ids,
        impl_->particle_indices,
        impl_->particle_count,
        device_grid
    );
    check(cudaGetLastError(), "compute_cell_ids kernel launch");

    std::size_t required_temporary_bytes = 0;
    check(
        cub::DeviceRadixSort::SortPairs(
            nullptr,
            required_temporary_bytes,
            impl_->cell_ids,
            impl_->sorted_cell_ids,
            impl_->particle_indices,
            impl_->sorted_particle_indices,
            static_cast<int>(impl_->particle_count)
        ),
        "CUB radix-sort storage query"
    );

    if (required_temporary_bytes > impl_->sort_temporary_bytes) {
        free_device(impl_->sort_temporary_storage);
        impl_->sort_temporary_storage = nullptr;
        check(
            cudaMalloc(
                &impl_->sort_temporary_storage,
                required_temporary_bytes
            ),
            "cudaMalloc radix-sort temporary storage"
        );
        impl_->sort_temporary_bytes = required_temporary_bytes;
    }

    check(
        cub::DeviceRadixSort::SortPairs(
            impl_->sort_temporary_storage,
            impl_->sort_temporary_bytes,
            impl_->cell_ids,
            impl_->sorted_cell_ids,
            impl_->particle_indices,
            impl_->sorted_particle_indices,
            static_cast<int>(impl_->particle_count)
        ),
        "CUB radix sort"
    );

    const std::size_t range_bytes =
        static_cast<std::size_t>(grid.cell_count) * sizeof(std::uint32_t);
    check(
        cudaMemset(impl_->cell_start, 0xFF, range_bytes),
        "cudaMemset cell_start"
    );
    check(
        cudaMemset(impl_->cell_end, 0xFF, range_bytes),
        "cudaMemset cell_end"
    );

    build_cell_ranges<<<block_count, block_size>>>(
        impl_->sorted_cell_ids,
        impl_->cell_start,
        impl_->cell_end,
        impl_->particle_count
    );
    check(cudaGetLastError(), "build_cell_ranges kernel launch");
    check(cudaDeviceSynchronize(), "CUDA cell-list synchronization");

    impl_->grid = grid;
    impl_->cell_list_built = true;
    impl_->candidates_built = false;
    impl_->events_built = false;
    impl_->collision_batch_built = false;
    impl_->wall_batch_built = false;
    impl_->candidate_count = 0;
    impl_->event_count = 0;
    impl_->independent_event_count = 0;
    impl_->conflicting_event_count = 0;
    impl_->earliest_event_time = std::numeric_limits<double>::infinity();
    impl_->wall_event_count = 0;
    impl_->reflected_axis_count = 0;
    impl_->earliest_wall_time = std::numeric_limits<double>::infinity();
}

void ParticleGrid::build_candidate_pairs() {
    if (!impl_->cell_list_built) {
        throw std::logic_error(
            "CUDA cell list must be built before candidate pairs."
        );
    }

    constexpr unsigned int block_size = 256;
    const unsigned int block_count = static_cast<unsigned int>(
        (impl_->particle_count + block_size - 1U) / block_size
    );
    const UniformGrid& grid = impl_->grid;
    const DeviceGrid device_grid{
        .extent_x = grid.extent.x,
        .extent_y = grid.extent.y,
        .extent_z = grid.extent.z,
        .cell_width = grid.cell_width,
        .dimension_x = grid.dimensions[0],
        .dimension_y = grid.dimensions[1],
        .dimension_z = grid.dimensions[2],
        .periodic_x = static_cast<std::uint8_t>(
            grid.boundary[0] == BoundaryKind::periodic
        ),
        .periodic_y = static_cast<std::uint8_t>(
            grid.boundary[1] == BoundaryKind::periodic
        ),
        .periodic_z = static_cast<std::uint8_t>(
            grid.boundary[2] == BoundaryKind::periodic
        ),
    };

    check(
        cudaMemset(
            impl_->candidate_counts,
            0,
            (impl_->particle_count + 1U) * sizeof(std::uint64_t)
        ),
        "cudaMemset candidate_counts"
    );

    count_candidate_pairs<<<block_count, block_size>>>(
        impl_->cell_ids,
        impl_->sorted_particle_indices,
        impl_->cell_start,
        impl_->cell_end,
        impl_->candidate_counts,
        impl_->particle_count,
        device_grid
    );
    check(cudaGetLastError(), "count_candidate_pairs kernel launch");

    std::size_t required_temporary_bytes = 0;
    check(
        cub::DeviceScan::ExclusiveSum(
            nullptr,
            required_temporary_bytes,
            impl_->candidate_counts,
            impl_->candidate_offsets,
            static_cast<int>(impl_->particle_count + 1U)
        ),
        "CUB candidate-scan storage query"
    );

    if (required_temporary_bytes > impl_->scan_temporary_bytes) {
        free_device(impl_->scan_temporary_storage);
        impl_->scan_temporary_storage = nullptr;
        check(
            cudaMalloc(
                &impl_->scan_temporary_storage,
                required_temporary_bytes
            ),
            "cudaMalloc candidate-scan temporary storage"
        );
        impl_->scan_temporary_bytes = required_temporary_bytes;
    }

    check(
        cub::DeviceScan::ExclusiveSum(
            impl_->scan_temporary_storage,
            impl_->scan_temporary_bytes,
            impl_->candidate_counts,
            impl_->candidate_offsets,
            static_cast<int>(impl_->particle_count + 1U)
        ),
        "CUB candidate exclusive scan"
    );

    std::uint64_t candidate_count = 0;
    check(
        cudaMemcpy(
            &candidate_count,
            impl_->candidate_offsets + impl_->particle_count,
            sizeof(candidate_count),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy candidate count to host"
    );

    if (
        candidate_count >
        static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max()
        )
    ) {
        throw std::overflow_error(
            "CUDA candidate-pair count exceeds the host size_t range."
        );
    }
    const std::size_t required_capacity =
        static_cast<std::size_t>(candidate_count);
    if (
        required_capacity >=
        static_cast<std::size_t>(std::numeric_limits<int>::max())
    ) {
        throw std::overflow_error(
            "CUDA event scan requires fewer than INT_MAX candidate pairs."
        );
    }
    if (
        required_capacity >
        std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t)
    ) {
        throw std::overflow_error(
            "CUDA candidate-pair arrays exceed the addressable byte range."
        );
    }

    if (required_capacity > impl_->candidate_capacity) {
        free_device(impl_->candidate_first);
        free_device(impl_->candidate_second);
        free_device(impl_->candidate_collision_time);
        free_device(impl_->candidate_shift_x);
        free_device(impl_->candidate_shift_y);
        free_device(impl_->candidate_shift_z);
        free_device(impl_->event_flags);
        free_device(impl_->event_offsets);
        impl_->candidate_first = nullptr;
        impl_->candidate_second = nullptr;
        impl_->candidate_collision_time = nullptr;
        impl_->candidate_shift_x = nullptr;
        impl_->candidate_shift_y = nullptr;
        impl_->candidate_shift_z = nullptr;
        impl_->event_flags = nullptr;
        impl_->event_offsets = nullptr;
        impl_->candidate_capacity = 0;

        allocate(
            impl_->candidate_first,
            required_capacity,
            "cudaMalloc candidate_first"
        );
        allocate(
            impl_->candidate_second,
            required_capacity,
            "cudaMalloc candidate_second"
        );
        allocate(
            impl_->candidate_collision_time,
            required_capacity,
            "cudaMalloc candidate_collision_time"
        );
        allocate(
            impl_->candidate_shift_x,
            required_capacity,
            "cudaMalloc candidate_shift_x"
        );
        allocate(
            impl_->candidate_shift_y,
            required_capacity,
            "cudaMalloc candidate_shift_y"
        );
        allocate(
            impl_->candidate_shift_z,
            required_capacity,
            "cudaMalloc candidate_shift_z"
        );
        allocate(
            impl_->event_flags,
            required_capacity + 1U,
            "cudaMalloc event_flags"
        );
        allocate(
            impl_->event_offsets,
            required_capacity + 1U,
            "cudaMalloc event_offsets"
        );
        impl_->candidate_capacity = required_capacity;
    }

    impl_->candidate_count = required_capacity;

    if (impl_->candidate_count != 0) {
        write_candidate_pairs<<<block_count, block_size>>>(
            impl_->cell_ids,
            impl_->sorted_particle_indices,
            impl_->cell_start,
            impl_->cell_end,
            impl_->candidate_offsets,
            impl_->candidate_first,
            impl_->candidate_second,
            impl_->particle_count,
            device_grid
        );
        check(cudaGetLastError(), "write_candidate_pairs kernel launch");
    }

    check(cudaDeviceSynchronize(), "CUDA candidate-pair synchronization");
    impl_->candidates_built = true;
    impl_->events_built = false;
    impl_->collision_batch_built = false;
    impl_->wall_batch_built = false;
    impl_->event_count = 0;
    impl_->independent_event_count = 0;
    impl_->conflicting_event_count = 0;
    impl_->earliest_event_time = std::numeric_limits<double>::infinity();
    impl_->wall_event_count = 0;
    impl_->reflected_axis_count = 0;
    impl_->earliest_wall_time = std::numeric_limits<double>::infinity();
}

void ParticleGrid::evaluate_collision_events(
    double diameter,
    double horizon,
    double tolerance
) {
    if (!impl_->candidates_built) {
        throw std::logic_error(
            "CUDA candidate pairs must be built before collision events."
        );
    }
    if (!std::isfinite(diameter) || diameter <= 0.0) {
        throw std::invalid_argument(
            "CUDA collision diameter must be finite and positive."
        );
    }
    if (!std::isfinite(horizon) || horizon < 0.0) {
        throw std::invalid_argument(
            "CUDA collision horizon must be finite and non-negative."
        );
    }
    if (!std::isfinite(tolerance) || tolerance <= 0.0) {
        throw std::invalid_argument(
            "CUDA collision tolerance must be finite and positive."
        );
    }
    if (
        impl_->candidate_count >=
        static_cast<std::size_t>(std::numeric_limits<int>::max())
    ) {
        throw std::overflow_error(
            "CUDA event compaction requires fewer than INT_MAX candidates."
        );
    }

    impl_->event_count = 0;
    impl_->events_built = false;
    impl_->collision_batch_built = false;
    impl_->independent_event_count = 0;
    impl_->conflicting_event_count = 0;
    impl_->earliest_event_time = std::numeric_limits<double>::infinity();

    if (impl_->candidate_count == 0) {
        impl_->events_built = true;
        return;
    }

    constexpr unsigned int block_size = 256;
    const unsigned int block_count = static_cast<unsigned int>(
        (impl_->candidate_count + block_size - 1U) / block_size
    );
    const UniformGrid& grid = impl_->grid;
    const DeviceGrid device_grid{
        .extent_x = grid.extent.x,
        .extent_y = grid.extent.y,
        .extent_z = grid.extent.z,
        .cell_width = grid.cell_width,
        .dimension_x = grid.dimensions[0],
        .dimension_y = grid.dimensions[1],
        .dimension_z = grid.dimensions[2],
        .periodic_x = static_cast<std::uint8_t>(
            grid.boundary[0] == BoundaryKind::periodic
        ),
        .periodic_y = static_cast<std::uint8_t>(
            grid.boundary[1] == BoundaryKind::periodic
        ),
        .periodic_z = static_cast<std::uint8_t>(
            grid.boundary[2] == BoundaryKind::periodic
        ),
    };

    check(
        cudaMemset(
            impl_->event_flags,
            0,
            (impl_->candidate_count + 1U) * sizeof(std::uint64_t)
        ),
        "cudaMemset collision event flags"
    );

    evaluate_candidate_collision_times<<<block_count, block_size>>>(
        impl_->position_x,
        impl_->position_y,
        impl_->position_z,
        impl_->velocity_x,
        impl_->velocity_y,
        impl_->velocity_z,
        impl_->candidate_first,
        impl_->candidate_second,
        impl_->candidate_collision_time,
        impl_->candidate_shift_x,
        impl_->candidate_shift_y,
        impl_->candidate_shift_z,
        impl_->event_flags,
        impl_->candidate_count,
        device_grid,
        diameter,
        horizon,
        tolerance
    );
    check(
        cudaGetLastError(),
        "evaluate_candidate_collision_times kernel launch"
    );

    std::size_t required_temporary_bytes = 0;
    check(
        cub::DeviceScan::ExclusiveSum(
            nullptr,
            required_temporary_bytes,
            impl_->event_flags,
            impl_->event_offsets,
            static_cast<int>(impl_->candidate_count + 1U)
        ),
        "CUB event-scan storage query"
    );

    if (required_temporary_bytes > impl_->scan_temporary_bytes) {
        free_device(impl_->scan_temporary_storage);
        impl_->scan_temporary_storage = nullptr;
        check(
            cudaMalloc(
                &impl_->scan_temporary_storage,
                required_temporary_bytes
            ),
            "cudaMalloc event-scan temporary storage"
        );
        impl_->scan_temporary_bytes = required_temporary_bytes;
    }

    check(
        cub::DeviceScan::ExclusiveSum(
            impl_->scan_temporary_storage,
            impl_->scan_temporary_bytes,
            impl_->event_flags,
            impl_->event_offsets,
            static_cast<int>(impl_->candidate_count + 1U)
        ),
        "CUB collision-event exclusive scan"
    );

    std::uint64_t event_count = 0;
    check(
        cudaMemcpy(
            &event_count,
            impl_->event_offsets + impl_->candidate_count,
            sizeof(event_count),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy collision-event count to host"
    );
    if (
        event_count >
        static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max()
        )
    ) {
        throw std::overflow_error(
            "CUDA collision-event count exceeds the host size_t range."
        );
    }

    const std::size_t required_capacity =
        static_cast<std::size_t>(event_count);
    if (required_capacity > impl_->event_capacity) {
        free_device(impl_->event_first);
        free_device(impl_->event_second);
        free_device(impl_->event_time);
        free_device(impl_->event_shift_x);
        free_device(impl_->event_shift_y);
        free_device(impl_->event_shift_z);
        impl_->event_first = nullptr;
        impl_->event_second = nullptr;
        impl_->event_time = nullptr;
        impl_->event_shift_x = nullptr;
        impl_->event_shift_y = nullptr;
        impl_->event_shift_z = nullptr;
        impl_->event_capacity = 0;

        allocate(
            impl_->event_first,
            required_capacity,
            "cudaMalloc event_first"
        );
        allocate(
            impl_->event_second,
            required_capacity,
            "cudaMalloc event_second"
        );
        allocate(
            impl_->event_time,
            required_capacity,
            "cudaMalloc event_time"
        );
        allocate(
            impl_->event_shift_x,
            required_capacity,
            "cudaMalloc event_shift_x"
        );
        allocate(
            impl_->event_shift_y,
            required_capacity,
            "cudaMalloc event_shift_y"
        );
        allocate(
            impl_->event_shift_z,
            required_capacity,
            "cudaMalloc event_shift_z"
        );
        impl_->event_capacity = required_capacity;
    }

    impl_->event_count = required_capacity;

    if (impl_->event_count != 0) {
        compact_collision_events<<<block_count, block_size>>>(
            impl_->candidate_first,
            impl_->candidate_second,
            impl_->candidate_collision_time,
            impl_->candidate_shift_x,
            impl_->candidate_shift_y,
            impl_->candidate_shift_z,
            impl_->event_flags,
            impl_->event_offsets,
            impl_->event_first,
            impl_->event_second,
            impl_->event_time,
            impl_->event_shift_x,
            impl_->event_shift_y,
            impl_->event_shift_z,
            impl_->candidate_count
        );
        check(
            cudaGetLastError(),
            "compact_collision_events kernel launch"
        );
    }

    check(cudaDeviceSynchronize(), "CUDA collision-event synchronization");
    impl_->events_built = true;
}

void ParticleGrid::analyze_earliest_collision_batch(double tolerance) {
    if (!impl_->events_built) {
        throw std::logic_error(
            "CUDA collision events must be built before batch analysis."
        );
    }
    if (!std::isfinite(tolerance) || tolerance <= 0.0) {
        throw std::invalid_argument(
            "CUDA collision-batch tolerance must be finite and positive."
        );
    }

    impl_->collision_batch_built = false;
    impl_->independent_event_count = 0;
    impl_->conflicting_event_count = 0;
    impl_->earliest_event_time = std::numeric_limits<double>::infinity();

    if (impl_->event_count == 0) {
        impl_->collision_batch_built = true;
        return;
    }

    std::size_t required_temporary_bytes = 0;
    check(
        cub::DeviceReduce::Min(
            nullptr,
            required_temporary_bytes,
            impl_->event_time,
            impl_->minimum_event_time,
            static_cast<int>(impl_->event_count)
        ),
        "CUB event-minimum storage query"
    );

    if (required_temporary_bytes > impl_->scan_temporary_bytes) {
        free_device(impl_->scan_temporary_storage);
        impl_->scan_temporary_storage = nullptr;
        check(
            cudaMalloc(
                &impl_->scan_temporary_storage,
                required_temporary_bytes
            ),
            "cudaMalloc event-minimum temporary storage"
        );
        impl_->scan_temporary_bytes = required_temporary_bytes;
    }

    check(
        cub::DeviceReduce::Min(
            impl_->scan_temporary_storage,
            impl_->scan_temporary_bytes,
            impl_->event_time,
            impl_->minimum_event_time,
            static_cast<int>(impl_->event_count)
        ),
        "CUB earliest collision-event reduction"
    );
    check(
        cudaMemcpy(
            &impl_->earliest_event_time,
            impl_->minimum_event_time,
            sizeof(double),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy earliest collision-event time to host"
    );

    check(
        cudaMemset(
            impl_->particle_batch_participation,
            0,
            impl_->particle_count * sizeof(std::uint32_t)
        ),
        "cudaMemset particle batch participation"
    );

    constexpr unsigned int block_size = 256;
    const unsigned int block_count = static_cast<unsigned int>(
        (impl_->event_count + block_size - 1U) / block_size
    );

    count_earliest_batch_participation<<<block_count, block_size>>>(
        impl_->event_first,
        impl_->event_second,
        impl_->event_time,
        impl_->particle_batch_participation,
        impl_->event_count,
        impl_->earliest_event_time,
        tolerance
    );
    check(
        cudaGetLastError(),
        "count_earliest_batch_participation kernel launch"
    );

    if (impl_->event_count > impl_->batch_index_capacity) {
        free_device(impl_->independent_event_indices);
        free_device(impl_->conflicting_event_indices);
        impl_->independent_event_indices = nullptr;
        impl_->conflicting_event_indices = nullptr;
        impl_->batch_index_capacity = 0;

        allocate(
            impl_->independent_event_indices,
            impl_->event_count,
            "cudaMalloc independent_event_indices"
        );
        allocate(
            impl_->conflicting_event_indices,
            impl_->event_count,
            "cudaMalloc conflicting_event_indices"
        );
        impl_->batch_index_capacity = impl_->event_count;
    }

    const auto classify_and_compact = [&](
        bool select_conflicting,
        std::uint32_t* output_indices,
        std::size_t& output_count
    ) {
        check(
            cudaMemset(
                impl_->event_flags,
                0,
                (impl_->event_count + 1U) * sizeof(std::uint64_t)
            ),
            select_conflicting
                ? "cudaMemset conflicting-event flags"
                : "cudaMemset independent-event flags"
        );

        classify_earliest_batch_events<<<block_count, block_size>>>(
            impl_->event_first,
            impl_->event_second,
            impl_->event_time,
            impl_->particle_batch_participation,
            impl_->event_flags,
            impl_->event_count,
            impl_->earliest_event_time,
            tolerance,
            select_conflicting
        );
        check(
            cudaGetLastError(),
            select_conflicting
                ? "classify conflicting events kernel launch"
                : "classify independent events kernel launch"
        );

        std::size_t scan_bytes = 0;
        check(
            cub::DeviceScan::ExclusiveSum(
                nullptr,
                scan_bytes,
                impl_->event_flags,
                impl_->event_offsets,
                static_cast<int>(impl_->event_count + 1U)
            ),
            "CUB collision-batch scan storage query"
        );

        if (scan_bytes > impl_->scan_temporary_bytes) {
            free_device(impl_->scan_temporary_storage);
            impl_->scan_temporary_storage = nullptr;
            check(
                cudaMalloc(
                    &impl_->scan_temporary_storage,
                    scan_bytes
                ),
                "cudaMalloc collision-batch scan temporary storage"
            );
            impl_->scan_temporary_bytes = scan_bytes;
        }

        check(
            cub::DeviceScan::ExclusiveSum(
                impl_->scan_temporary_storage,
                impl_->scan_temporary_bytes,
                impl_->event_flags,
                impl_->event_offsets,
                static_cast<int>(impl_->event_count + 1U)
            ),
            "CUB collision-batch exclusive scan"
        );

        std::uint64_t selected_count = 0;
        check(
            cudaMemcpy(
                &selected_count,
                impl_->event_offsets + impl_->event_count,
                sizeof(selected_count),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy collision-batch count to host"
        );
        output_count = static_cast<std::size_t>(selected_count);

        if (output_count != 0) {
            compact_event_indices<<<block_count, block_size>>>(
                impl_->event_flags,
                impl_->event_offsets,
                output_indices,
                impl_->event_count
            );
            check(
                cudaGetLastError(),
                select_conflicting
                    ? "compact conflicting events kernel launch"
                    : "compact independent events kernel launch"
            );
        }
    };

    classify_and_compact(
        false,
        impl_->independent_event_indices,
        impl_->independent_event_count
    );
    classify_and_compact(
        true,
        impl_->conflicting_event_indices,
        impl_->conflicting_event_count
    );

    check(
        cudaDeviceSynchronize(),
        "CUDA collision-batch analysis synchronization"
    );
    impl_->collision_batch_built = true;
}

CollisionBatchAdvanceResult
ParticleGrid::advance_earliest_collision_batch(
    double diameter,
    double tolerance
) {
    if (!impl_->collision_batch_built) {
        throw std::logic_error(
            "CUDA collision batch must be analyzed before it is advanced."
        );
    }
    if (!std::isfinite(diameter) || diameter <= 0.0) {
        throw std::invalid_argument(
            "CUDA collision-batch diameter must be finite and positive."
        );
    }
    if (!std::isfinite(tolerance) || tolerance <= 0.0) {
        throw std::invalid_argument(
            "CUDA collision-batch tolerance must be finite and positive."
        );
    }
    if (
        !std::isfinite(impl_->earliest_event_time) ||
        impl_->independent_event_count +
            impl_->conflicting_event_count == 0
    ) {
        throw std::logic_error(
            "Cannot advance an empty CUDA collision-event batch."
        );
    }

    const UniformGrid& grid = impl_->grid;
    const DeviceGrid device_grid{
        .extent_x = grid.extent.x,
        .extent_y = grid.extent.y,
        .extent_z = grid.extent.z,
        .cell_width = grid.cell_width,
        .dimension_x = grid.dimensions[0],
        .dimension_y = grid.dimensions[1],
        .dimension_z = grid.dimensions[2],
        .periodic_x = static_cast<std::uint8_t>(
            grid.boundary[0] == BoundaryKind::periodic
        ),
        .periodic_y = static_cast<std::uint8_t>(
            grid.boundary[1] == BoundaryKind::periodic
        ),
        .periodic_z = static_cast<std::uint8_t>(
            grid.boundary[2] == BoundaryKind::periodic
        ),
    };
    constexpr unsigned int block_size = 256;
    std::size_t serialized_conflict_event_count = 0;

    if (impl_->conflicting_event_count != 0) {
        const unsigned int conflict_block_count =
            static_cast<unsigned int>(
                (
                    impl_->conflicting_event_count +
                    block_size -
                    1U
                ) /
                block_size
            );
        write_conflicting_event_keys<<<conflict_block_count, block_size>>>(
            impl_->conflicting_event_indices,
            impl_->event_first,
            impl_->event_second,
            impl_->event_flags,
            impl_->conflicting_event_count
        );
        check(
            cudaGetLastError(),
            "write_conflicting_event_keys kernel launch"
        );

        std::size_t reduction_bytes = 0;
        check(
            cub::DeviceReduce::Min(
                nullptr,
                reduction_bytes,
                impl_->event_flags,
                impl_->minimum_conflict_key,
                static_cast<int>(impl_->conflicting_event_count)
            ),
            "CUB conflict-key minimum storage query"
        );
        if (reduction_bytes > impl_->scan_temporary_bytes) {
            free_device(impl_->scan_temporary_storage);
            impl_->scan_temporary_storage = nullptr;
            check(
                cudaMalloc(
                    &impl_->scan_temporary_storage,
                    reduction_bytes
                ),
                "cudaMalloc conflict-key temporary storage"
            );
            impl_->scan_temporary_bytes = reduction_bytes;
        }
        check(
            cub::DeviceReduce::Min(
                impl_->scan_temporary_storage,
                impl_->scan_temporary_bytes,
                impl_->event_flags,
                impl_->minimum_conflict_key,
                static_cast<int>(impl_->conflicting_event_count)
            ),
            "CUB minimum conflicting-event key"
        );
        check(
            cudaMemset(
                impl_->selected_conflict_event_index,
                0xFF,
                sizeof(std::uint32_t)
            ),
            "cudaMemset selected conflict event"
        );
        find_minimum_conflict_event<<<conflict_block_count, block_size>>>(
            impl_->conflicting_event_indices,
            impl_->event_first,
            impl_->event_second,
            impl_->minimum_conflict_key,
            impl_->selected_conflict_event_index,
            impl_->conflicting_event_count
        );
        check(
            cudaGetLastError(),
            "find_minimum_conflict_event kernel launch"
        );

        std::uint32_t selected_event_index =
            std::numeric_limits<std::uint32_t>::max();
        check(
            cudaMemcpy(
                &selected_event_index,
                impl_->selected_conflict_event_index,
                sizeof(selected_event_index),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy selected conflict event to host"
        );
        if (selected_event_index >= impl_->event_count) {
            throw CudaError(
                "CUDA conflict serialization did not select a valid event."
            );
        }
        check(
            cudaMemcpy(
                impl_->independent_event_indices +
                    impl_->independent_event_count,
                &selected_event_index,
                sizeof(selected_event_index),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy serialized conflict event to device"
        );
        ++impl_->independent_event_count;
        serialized_conflict_event_count = 1;
    }

    if (impl_->independent_event_count != 0) {
        check(
            cudaMemset(
                impl_->collision_validation_failures,
                0,
                sizeof(std::uint32_t)
            ),
            "cudaMemset collision validation failures"
        );
        const unsigned int independent_block_count =
            static_cast<unsigned int>(
                (
                    impl_->independent_event_count +
                    block_size -
                    1U
                ) /
                block_size
            );
        validate_independent_collision_events
            <<<independent_block_count, block_size>>>(
                impl_->position_x,
                impl_->position_y,
                impl_->position_z,
                impl_->velocity_x,
                impl_->velocity_y,
                impl_->velocity_z,
                impl_->event_first,
                impl_->event_second,
                impl_->event_time,
                impl_->event_shift_x,
                impl_->event_shift_y,
                impl_->event_shift_z,
                impl_->independent_event_indices,
                impl_->independent_event_count,
                device_grid,
                diameter,
                tolerance,
                impl_->collision_validation_failures
            );
        check(
            cudaGetLastError(),
            "validate independent collision events kernel launch"
        );

        std::uint32_t validation_failures = 0;
        check(
            cudaMemcpy(
                &validation_failures,
                impl_->collision_validation_failures,
                sizeof(validation_failures),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy collision validation failures to host"
        );
        if (validation_failures != 0) {
            throw CudaError(
                std::to_string(validation_failures) +
                " independent CUDA collision event(s) failed contact "
                "validation."
            );
        }
    }

    const unsigned int particle_block_count = static_cast<unsigned int>(
        (impl_->particle_count + block_size - 1U) / block_size
    );
    advance_free_flight_kernel<<<particle_block_count, block_size>>>(
        impl_->position_x,
        impl_->position_y,
        impl_->position_z,
        impl_->velocity_x,
        impl_->velocity_y,
        impl_->velocity_z,
        impl_->particle_count,
        impl_->earliest_event_time
    );
    check(
        cudaGetLastError(),
        "advance earliest collision batch kernel launch"
    );

    if (impl_->independent_event_count != 0) {
        const unsigned int independent_block_count =
            static_cast<unsigned int>(
                (
                    impl_->independent_event_count +
                    block_size -
                    1U
                ) /
                block_size
            );
        resolve_independent_collision_events
            <<<independent_block_count, block_size>>>(
                impl_->position_x,
                impl_->position_y,
                impl_->position_z,
                impl_->velocity_x,
                impl_->velocity_y,
                impl_->velocity_z,
                impl_->event_first,
                impl_->event_second,
                impl_->event_time,
                impl_->event_shift_x,
                impl_->event_shift_y,
                impl_->event_shift_z,
                impl_->independent_event_indices,
                impl_->independent_event_count,
                device_grid,
                impl_->earliest_event_time
            );
        check(
            cudaGetLastError(),
            "resolve independent collision events kernel launch"
        );
    }

    normalize_grid_positions<<<particle_block_count, block_size>>>(
        impl_->position_x,
        impl_->position_y,
        impl_->position_z,
        impl_->particle_count,
        device_grid,
        diameter
    );
    check(
        cudaGetLastError(),
        "wrap periodic positions kernel launch"
    );
    check(
        cudaDeviceSynchronize(),
        "CUDA collision-batch advance synchronization"
    );

    const CollisionBatchAdvanceResult result{
        .delta_time = impl_->earliest_event_time,
        .resolved_event_count = impl_->independent_event_count,
        .serialized_conflict_event_count =
            serialized_conflict_event_count,
        .deferred_conflicting_event_count = 0,
    };

    impl_->cell_list_built = false;
    impl_->candidates_built = false;
    impl_->events_built = false;
    impl_->collision_batch_built = false;
    impl_->wall_batch_built = false;
    impl_->candidate_count = 0;
    impl_->event_count = 0;
    impl_->independent_event_count = 0;
    impl_->conflicting_event_count = 0;
    impl_->earliest_event_time = std::numeric_limits<double>::infinity();
    impl_->wall_event_count = 0;
    impl_->reflected_axis_count = 0;
    impl_->earliest_wall_time = std::numeric_limits<double>::infinity();

    return result;
}

void ParticleGrid::evaluate_reflecting_wall_batch(
    double diameter,
    double horizon,
    double tolerance
) {
    if (!impl_->cell_list_built) {
        throw std::logic_error(
            "CUDA cell list must be built before reflecting-wall events."
        );
    }
    if (!std::isfinite(diameter) || diameter <= 0.0) {
        throw std::invalid_argument(
            "CUDA reflecting-wall diameter must be finite and positive."
        );
    }
    if (!std::isfinite(horizon) || horizon < 0.0) {
        throw std::invalid_argument(
            "CUDA reflecting-wall horizon must be finite and non-negative."
        );
    }
    if (!std::isfinite(tolerance) || tolerance <= 0.0) {
        throw std::invalid_argument(
            "CUDA reflecting-wall tolerance must be finite and positive."
        );
    }

    impl_->wall_batch_built = false;
    impl_->wall_event_count = 0;
    impl_->reflected_axis_count = 0;
    impl_->earliest_wall_time = std::numeric_limits<double>::infinity();

    const UniformGrid& grid = impl_->grid;
    const DeviceGrid device_grid{
        .extent_x = grid.extent.x,
        .extent_y = grid.extent.y,
        .extent_z = grid.extent.z,
        .cell_width = grid.cell_width,
        .dimension_x = grid.dimensions[0],
        .dimension_y = grid.dimensions[1],
        .dimension_z = grid.dimensions[2],
        .periodic_x = static_cast<std::uint8_t>(
            grid.boundary[0] == BoundaryKind::periodic
        ),
        .periodic_y = static_cast<std::uint8_t>(
            grid.boundary[1] == BoundaryKind::periodic
        ),
        .periodic_z = static_cast<std::uint8_t>(
            grid.boundary[2] == BoundaryKind::periodic
        ),
    };
    constexpr unsigned int block_size = 256;
    const unsigned int block_count = static_cast<unsigned int>(
        (impl_->particle_count + block_size - 1U) / block_size
    );

    check(
        cudaMemset(
            impl_->collision_validation_failures,
            0,
            sizeof(std::uint32_t)
        ),
        "cudaMemset reflecting-wall validation failures"
    );
    compute_reflecting_wall_times<<<block_count, block_size>>>(
        impl_->position_x,
        impl_->position_y,
        impl_->position_z,
        impl_->velocity_x,
        impl_->velocity_y,
        impl_->velocity_z,
        impl_->particle_wall_time,
        impl_->particle_wall_axis_mask,
        impl_->collision_validation_failures,
        impl_->particle_count,
        device_grid,
        diameter,
        horizon,
        tolerance
    );
    check(
        cudaGetLastError(),
        "compute_reflecting_wall_times kernel launch"
    );

    std::uint32_t validation_failures = 0;
    check(
        cudaMemcpy(
            &validation_failures,
            impl_->collision_validation_failures,
            sizeof(validation_failures),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy reflecting-wall validation failures to host"
    );
    if (validation_failures != 0) {
        throw CudaError(
            std::to_string(validation_failures) +
            " CUDA particle(s) failed reflecting-wall validation."
        );
    }

    std::size_t reduction_bytes = 0;
    check(
        cub::DeviceReduce::Min(
            nullptr,
            reduction_bytes,
            impl_->particle_wall_time,
            impl_->minimum_wall_time,
            static_cast<int>(impl_->particle_count)
        ),
        "CUB wall-minimum storage query"
    );
    if (reduction_bytes > impl_->scan_temporary_bytes) {
        free_device(impl_->scan_temporary_storage);
        impl_->scan_temporary_storage = nullptr;
        check(
            cudaMalloc(
                &impl_->scan_temporary_storage,
                reduction_bytes
            ),
            "cudaMalloc wall-minimum temporary storage"
        );
        impl_->scan_temporary_bytes = reduction_bytes;
    }
    check(
        cub::DeviceReduce::Min(
            impl_->scan_temporary_storage,
            impl_->scan_temporary_bytes,
            impl_->particle_wall_time,
            impl_->minimum_wall_time,
            static_cast<int>(impl_->particle_count)
        ),
        "CUB earliest reflecting-wall reduction"
    );

    double raw_earliest_time = std::numeric_limits<double>::infinity();
    check(
        cudaMemcpy(
            &raw_earliest_time,
            impl_->minimum_wall_time,
            sizeof(raw_earliest_time),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy earliest reflecting-wall time to host"
    );
    if (
        !std::isfinite(raw_earliest_time) ||
        raw_earliest_time > horizon + tolerance
    ) {
        impl_->wall_batch_built = true;
        return;
    }
    impl_->earliest_wall_time =
        std::min(raw_earliest_time, horizon);

    check(
        cudaMemset(
            impl_->candidate_counts,
            0,
            (impl_->particle_count + 1U) * sizeof(std::uint64_t)
        ),
        "cudaMemset reflecting-wall selection flags"
    );
    select_reflecting_wall_batch<<<block_count, block_size>>>(
        impl_->particle_wall_time,
        impl_->candidate_counts,
        impl_->particle_count,
        raw_earliest_time,
        horizon,
        tolerance
    );
    check(
        cudaGetLastError(),
        "select_reflecting_wall_batch kernel launch"
    );

    std::size_t scan_bytes = 0;
    check(
        cub::DeviceScan::ExclusiveSum(
            nullptr,
            scan_bytes,
            impl_->candidate_counts,
            impl_->candidate_offsets,
            static_cast<int>(impl_->particle_count + 1U)
        ),
        "CUB wall-batch scan storage query"
    );
    if (scan_bytes > impl_->scan_temporary_bytes) {
        free_device(impl_->scan_temporary_storage);
        impl_->scan_temporary_storage = nullptr;
        check(
            cudaMalloc(
                &impl_->scan_temporary_storage,
                scan_bytes
            ),
            "cudaMalloc wall-batch scan temporary storage"
        );
        impl_->scan_temporary_bytes = scan_bytes;
    }
    check(
        cub::DeviceScan::ExclusiveSum(
            impl_->scan_temporary_storage,
            impl_->scan_temporary_bytes,
            impl_->candidate_counts,
            impl_->candidate_offsets,
            static_cast<int>(impl_->particle_count + 1U)
        ),
        "CUB reflecting-wall batch exclusive scan"
    );

    std::uint64_t wall_event_count = 0;
    check(
        cudaMemcpy(
            &wall_event_count,
            impl_->candidate_offsets + impl_->particle_count,
            sizeof(wall_event_count),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy reflecting-wall event count to host"
    );
    impl_->wall_event_count =
        static_cast<std::size_t>(wall_event_count);

    check(
        cudaMemset(
            impl_->reflected_axis_count_device,
            0,
            sizeof(std::uint32_t)
        ),
        "cudaMemset reflected axis count"
    );
    if (impl_->wall_event_count != 0) {
        compact_reflecting_wall_batch<<<block_count, block_size>>>(
            impl_->particle_wall_time,
            impl_->particle_wall_axis_mask,
            impl_->candidate_counts,
            impl_->candidate_offsets,
            impl_->wall_event_particle,
            impl_->wall_event_time,
            impl_->wall_event_axis_mask,
            impl_->reflected_axis_count_device,
            impl_->particle_count,
            horizon
        );
        check(
            cudaGetLastError(),
            "compact_reflecting_wall_batch kernel launch"
        );
        check(
            cudaMemcpy(
                &impl_->reflected_axis_count,
                impl_->reflected_axis_count_device,
                sizeof(std::uint32_t),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy reflected axis count to host"
        );
    }

    check(
        cudaDeviceSynchronize(),
        "CUDA reflecting-wall batch synchronization"
    );
    impl_->wall_batch_built = true;
}

ReflectingWallAdvanceResult
ParticleGrid::advance_reflecting_wall_batch(
    double diameter,
    double tolerance
) {
    if (!impl_->wall_batch_built || impl_->wall_event_count == 0) {
        throw std::logic_error(
            "A non-empty CUDA reflecting-wall batch must be built first."
        );
    }
    if (!std::isfinite(diameter) || diameter <= 0.0) {
        throw std::invalid_argument(
            "CUDA reflecting-wall diameter must be finite and positive."
        );
    }
    if (!std::isfinite(tolerance) || tolerance <= 0.0) {
        throw std::invalid_argument(
            "CUDA reflecting-wall tolerance must be finite and positive."
        );
    }

    const UniformGrid& grid = impl_->grid;
    const DeviceGrid device_grid{
        .extent_x = grid.extent.x,
        .extent_y = grid.extent.y,
        .extent_z = grid.extent.z,
        .cell_width = grid.cell_width,
        .dimension_x = grid.dimensions[0],
        .dimension_y = grid.dimensions[1],
        .dimension_z = grid.dimensions[2],
        .periodic_x = static_cast<std::uint8_t>(
            grid.boundary[0] == BoundaryKind::periodic
        ),
        .periodic_y = static_cast<std::uint8_t>(
            grid.boundary[1] == BoundaryKind::periodic
        ),
        .periodic_z = static_cast<std::uint8_t>(
            grid.boundary[2] == BoundaryKind::periodic
        ),
    };
    constexpr unsigned int block_size = 256;
    const unsigned int particle_block_count = static_cast<unsigned int>(
        (impl_->particle_count + block_size - 1U) / block_size
    );
    const unsigned int wall_block_count = static_cast<unsigned int>(
        (impl_->wall_event_count + block_size - 1U) / block_size
    );

    advance_free_flight_kernel<<<particle_block_count, block_size>>>(
        impl_->position_x,
        impl_->position_y,
        impl_->position_z,
        impl_->velocity_x,
        impl_->velocity_y,
        impl_->velocity_z,
        impl_->particle_count,
        impl_->earliest_wall_time
    );
    check(
        cudaGetLastError(),
        "advance reflecting-wall batch kernel launch"
    );
    resolve_reflecting_wall_batch<<<wall_block_count, block_size>>>(
        impl_->position_x,
        impl_->position_y,
        impl_->position_z,
        impl_->velocity_x,
        impl_->velocity_y,
        impl_->velocity_z,
        impl_->wall_event_particle,
        impl_->wall_event_axis_mask,
        impl_->wall_event_count,
        device_grid,
        diameter
    );
    check(
        cudaGetLastError(),
        "resolve_reflecting_wall_batch kernel launch"
    );
    normalize_grid_positions<<<particle_block_count, block_size>>>(
        impl_->position_x,
        impl_->position_y,
        impl_->position_z,
        impl_->particle_count,
        device_grid,
        diameter
    );
    check(
        cudaGetLastError(),
        "normalize reflecting-wall positions kernel launch"
    );
    check(
        cudaDeviceSynchronize(),
        "CUDA reflecting-wall advance synchronization"
    );

    const ReflectingWallAdvanceResult result{
        .delta_time = impl_->earliest_wall_time,
        .reflected_particle_count = impl_->wall_event_count,
        .reflected_axis_count = impl_->reflected_axis_count,
    };

    impl_->cell_list_built = false;
    impl_->candidates_built = false;
    impl_->events_built = false;
    impl_->collision_batch_built = false;
    impl_->wall_batch_built = false;
    impl_->candidate_count = 0;
    impl_->event_count = 0;
    impl_->independent_event_count = 0;
    impl_->conflicting_event_count = 0;
    impl_->wall_event_count = 0;
    impl_->reflected_axis_count = 0;
    impl_->earliest_event_time = std::numeric_limits<double>::infinity();
    impl_->earliest_wall_time = std::numeric_limits<double>::infinity();

    return result;
}

void ParticleGrid::advance_free_flight_in_grid(
    double delta_time,
    double diameter,
    double tolerance
) {
    if (!impl_->cell_list_built) {
        throw std::logic_error(
            "CUDA cell list must be built before grid-aware free flight."
        );
    }
    if (!std::isfinite(delta_time) || delta_time < 0.0) {
        throw std::invalid_argument(
            "CUDA grid free-flight time must be finite and non-negative."
        );
    }
    if (!std::isfinite(diameter) || diameter <= 0.0) {
        throw std::invalid_argument(
            "CUDA grid free-flight diameter must be finite and positive."
        );
    }
    if (!std::isfinite(tolerance) || tolerance <= 0.0) {
        throw std::invalid_argument(
            "CUDA grid free-flight tolerance must be finite and positive."
        );
    }

    const UniformGrid& grid = impl_->grid;
    const DeviceGrid device_grid{
        .extent_x = grid.extent.x,
        .extent_y = grid.extent.y,
        .extent_z = grid.extent.z,
        .cell_width = grid.cell_width,
        .dimension_x = grid.dimensions[0],
        .dimension_y = grid.dimensions[1],
        .dimension_z = grid.dimensions[2],
        .periodic_x = static_cast<std::uint8_t>(
            grid.boundary[0] == BoundaryKind::periodic
        ),
        .periodic_y = static_cast<std::uint8_t>(
            grid.boundary[1] == BoundaryKind::periodic
        ),
        .periodic_z = static_cast<std::uint8_t>(
            grid.boundary[2] == BoundaryKind::periodic
        ),
    };
    constexpr unsigned int block_size = 256;
    const unsigned int block_count = static_cast<unsigned int>(
        (impl_->particle_count + block_size - 1U) / block_size
    );

    check(
        cudaMemset(
            impl_->collision_validation_failures,
            0,
            sizeof(std::uint32_t)
        ),
        "cudaMemset grid free-flight validation failures"
    );
    validate_grid_free_flight<<<block_count, block_size>>>(
        impl_->position_x,
        impl_->position_y,
        impl_->position_z,
        impl_->velocity_x,
        impl_->velocity_y,
        impl_->velocity_z,
        impl_->collision_validation_failures,
        impl_->particle_count,
        device_grid,
        delta_time,
        diameter,
        tolerance
    );
    check(
        cudaGetLastError(),
        "validate_grid_free_flight kernel launch"
    );

    std::uint32_t validation_failures = 0;
    check(
        cudaMemcpy(
            &validation_failures,
            impl_->collision_validation_failures,
            sizeof(validation_failures),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy grid free-flight validation failures to host"
    );
    if (validation_failures != 0) {
        throw CudaError(
            std::to_string(validation_failures) +
            " CUDA particle(s) crossed an unresolved reflecting wall."
        );
    }

    advance_free_flight_kernel<<<block_count, block_size>>>(
        impl_->position_x,
        impl_->position_y,
        impl_->position_z,
        impl_->velocity_x,
        impl_->velocity_y,
        impl_->velocity_z,
        impl_->particle_count,
        delta_time
    );
    check(
        cudaGetLastError(),
        "grid-aware free-flight kernel launch"
    );
    normalize_grid_positions<<<block_count, block_size>>>(
        impl_->position_x,
        impl_->position_y,
        impl_->position_z,
        impl_->particle_count,
        device_grid,
        diameter
    );
    check(
        cudaGetLastError(),
        "normalize grid free-flight positions kernel launch"
    );
    check(
        cudaDeviceSynchronize(),
        "CUDA grid free-flight synchronization"
    );

    impl_->cell_list_built = false;
    impl_->candidates_built = false;
    impl_->events_built = false;
    impl_->collision_batch_built = false;
    impl_->wall_batch_built = false;
    impl_->candidate_count = 0;
    impl_->event_count = 0;
    impl_->independent_event_count = 0;
    impl_->conflicting_event_count = 0;
    impl_->wall_event_count = 0;
    impl_->reflected_axis_count = 0;
    impl_->earliest_event_time = std::numeric_limits<double>::infinity();
    impl_->earliest_wall_time = std::numeric_limits<double>::infinity();
}

TimeWindowAdvanceResult ParticleGrid::advance_time_window(
    const UniformGrid& grid,
    double duration,
    double diameter,
    std::size_t maximum_batches,
    double tolerance
) {
    if (!impl_->particles_uploaded) {
        throw std::logic_error(
            "CUDA particles must be uploaded before a time window."
        );
    }
    if (!std::isfinite(duration) || duration < 0.0) {
        throw std::invalid_argument(
            "CUDA time-window duration must be finite and non-negative."
        );
    }
    if (!std::isfinite(diameter) || diameter <= 0.0) {
        throw std::invalid_argument(
            "CUDA time-window diameter must be finite and positive."
        );
    }
    if (!std::isfinite(tolerance) || tolerance <= 0.0) {
        throw std::invalid_argument(
            "CUDA time-window tolerance must be finite and positive."
        );
    }
    if (grid.cell_width <= diameter) {
        throw std::invalid_argument(
            "CUDA time-window cell width must exceed the diameter."
        );
    }
    if (maximum_batches == 0) {
        throw std::invalid_argument(
            "CUDA time-window batch limit must be positive."
        );
    }

    TimeWindowAdvanceResult result{
        .requested_time = duration,
    };
    constexpr unsigned int block_size = 256;
    const unsigned int particle_block_count =
        static_cast<unsigned int>(
            (impl_->particle_count + block_size - 1U) / block_size
        );

    while (result.advanced_time < duration) {
        if (result.batch_count >= maximum_batches) {
            result.stop_reason = TimeWindowStopReason::batch_limit;
            return result;
        }

        const double remaining = duration - result.advanced_time;
        build_cell_list(grid);
        build_candidate_pairs();
        evaluate_reflecting_wall_batch(
            diameter,
            remaining,
            tolerance
        );

        compute_particle_speeds<<<particle_block_count, block_size>>>(
            impl_->velocity_x,
            impl_->velocity_y,
            impl_->velocity_z,
            impl_->particle_wall_time,
            impl_->particle_count
        );
        check(
            cudaGetLastError(),
            "compute_particle_speeds kernel launch"
        );

        std::size_t reduction_bytes = 0;
        check(
            cub::DeviceReduce::Max(
                nullptr,
                reduction_bytes,
                impl_->particle_wall_time,
                impl_->minimum_event_time,
                static_cast<int>(impl_->particle_count)
            ),
            "CUB maximum-speed storage query"
        );
        if (reduction_bytes > impl_->scan_temporary_bytes) {
            free_device(impl_->scan_temporary_storage);
            impl_->scan_temporary_storage = nullptr;
            check(
                cudaMalloc(
                    &impl_->scan_temporary_storage,
                    reduction_bytes
                ),
                "cudaMalloc maximum-speed temporary storage"
            );
            impl_->scan_temporary_bytes = reduction_bytes;
        }
        check(
            cub::DeviceReduce::Max(
                impl_->scan_temporary_storage,
                impl_->scan_temporary_bytes,
                impl_->particle_wall_time,
                impl_->minimum_event_time,
                static_cast<int>(impl_->particle_count)
            ),
            "CUB maximum particle-speed reduction"
        );

        double maximum_speed = 0.0;
        check(
            cudaMemcpy(
                &maximum_speed,
                impl_->minimum_event_time,
                sizeof(maximum_speed),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy maximum particle speed to host"
        );
        const double motion_horizon = maximum_speed > 0.0
            ? (
                  grid.cell_width - diameter
              ) /
              (2.0 * maximum_speed)
            : std::numeric_limits<double>::infinity();
        const double safe_horizon = std::min({
            remaining,
            motion_horizon,
            impl_->earliest_wall_time,
        });
        if (!std::isfinite(safe_horizon) || safe_horizon < 0.0) {
            throw std::runtime_error(
                "CUDA time window produced an invalid safe horizon."
            );
        }

        evaluate_collision_events(
            diameter,
            safe_horizon,
            tolerance
        );
        analyze_earliest_collision_batch(tolerance);
        ++result.candidate_rebuild_count;

        const bool has_collision = impl_->event_count != 0;
        const bool wall_is_current =
            impl_->wall_event_count != 0 &&
            impl_->earliest_wall_time <= safe_horizon + tolerance;
        if (
            has_collision &&
            wall_is_current &&
            std::abs(
                impl_->earliest_event_time -
                impl_->earliest_wall_time
            ) <= tolerance
        ) {
            const double tie_time = std::min(
                impl_->earliest_event_time,
                impl_->earliest_wall_time
            );
            const std::size_t deferred_event_count =
                impl_->independent_event_count +
                impl_->conflicting_event_count +
                impl_->wall_event_count;
            advance_free_flight_in_grid(
                tie_time,
                diameter,
                tolerance
            );
            result.advanced_time += tie_time;
            ++result.batch_count;
            result.deferred_conflicting_event_count =
                deferred_event_count;
            result.stop_reason = TimeWindowStopReason::mixed_event_tie;
            return result;
        }

        if (
            has_collision &&
            (
                !wall_is_current ||
                impl_->earliest_event_time <
                    impl_->earliest_wall_time - tolerance
            )
        ) {
            const CollisionBatchAdvanceResult batch_result =
                advance_earliest_collision_batch(
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

        if (wall_is_current) {
            const ReflectingWallAdvanceResult wall_result =
                advance_reflecting_wall_batch(
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
                "CUDA time window made no progress without an event."
            );
        }
        advance_free_flight_in_grid(
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

ParallelWindowStepResult ParticleGrid::advance_parallel_window_step(
    const UniformGrid& grid,
    double requested_window,
    double diameter,
    double tolerance
) {
    if (!impl_->particles_uploaded) {
        throw std::logic_error(
            "CUDA particles must be uploaded before a parallel window."
        );
    }
    if (!std::isfinite(requested_window) || requested_window <= 0.0) {
        throw std::invalid_argument(
            "CUDA parallel-window duration must be finite and positive."
        );
    }
    if (!std::isfinite(diameter) || diameter <= 0.0) {
        throw std::invalid_argument(
            "CUDA parallel-window diameter must be finite and positive."
        );
    }
    if (!std::isfinite(tolerance) || tolerance <= 0.0) {
        throw std::invalid_argument(
            "CUDA parallel-window tolerance must be finite and positive."
        );
    }
    if (!std::isfinite(grid.cell_width) || grid.cell_width <= diameter) {
        throw std::invalid_argument(
            "CUDA parallel-window cell width must exceed the diameter."
        );
    }
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const double extent = axis == 0
            ? grid.extent.x
            : (axis == 1 ? grid.extent.y : grid.extent.z);
        if (
            grid.boundary[axis] == BoundaryKind::reflecting &&
            extent <= diameter
        ) {
            throw std::invalid_argument(
                "A reflecting CUDA parallel-window extent must exceed "
                "the particle diameter."
            );
        }
    }

    constexpr unsigned int block_size = 256;
    const unsigned int particle_block_count =
        static_cast<unsigned int>(
            (impl_->particle_count + block_size - 1U) / block_size
        );
    const DeviceGrid device_grid{
        .extent_x = grid.extent.x,
        .extent_y = grid.extent.y,
        .extent_z = grid.extent.z,
        .cell_width = grid.cell_width,
        .dimension_x = grid.dimensions[0],
        .dimension_y = grid.dimensions[1],
        .dimension_z = grid.dimensions[2],
        .periodic_x = static_cast<std::uint8_t>(
            grid.boundary[0] == BoundaryKind::periodic
        ),
        .periodic_y = static_cast<std::uint8_t>(
            grid.boundary[1] == BoundaryKind::periodic
        ),
        .periodic_z = static_cast<std::uint8_t>(
            grid.boundary[2] == BoundaryKind::periodic
        ),
    };

    build_cell_list(grid);
    build_candidate_pairs();
    const std::size_t initial_candidate_count =
        impl_->candidate_count;

    compute_particle_speeds<<<particle_block_count, block_size>>>(
        impl_->velocity_x,
        impl_->velocity_y,
        impl_->velocity_z,
        impl_->particle_wall_time,
        impl_->particle_count
    );
    check(
        cudaGetLastError(),
        "compute parallel-window particle speeds kernel launch"
    );

    std::size_t speed_reduction_bytes = 0;
    check(
        cub::DeviceReduce::Max(
            nullptr,
            speed_reduction_bytes,
            impl_->particle_wall_time,
            impl_->minimum_event_time,
            static_cast<int>(impl_->particle_count)
        ),
        "CUB parallel-window maximum-speed storage query"
    );
    if (speed_reduction_bytes > impl_->scan_temporary_bytes) {
        free_device(impl_->scan_temporary_storage);
        impl_->scan_temporary_storage = nullptr;
        check(
            cudaMalloc(
                &impl_->scan_temporary_storage,
                speed_reduction_bytes
            ),
            "cudaMalloc parallel-window speed temporary storage"
        );
        impl_->scan_temporary_bytes = speed_reduction_bytes;
    }
    check(
        cub::DeviceReduce::Max(
            impl_->scan_temporary_storage,
            impl_->scan_temporary_bytes,
            impl_->particle_wall_time,
            impl_->minimum_event_time,
            static_cast<int>(impl_->particle_count)
        ),
        "CUB parallel-window maximum-speed reduction"
    );

    double maximum_speed = 0.0;
    check(
        cudaMemcpy(
            &maximum_speed,
            impl_->minimum_event_time,
            sizeof(maximum_speed),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy parallel-window maximum speed to host"
    );
    const double candidate_horizon = maximum_speed > 0.0
        ? (
              grid.cell_width - diameter
          ) /
          (2.0 * maximum_speed)
        : std::numeric_limits<double>::infinity();
    if (
        (!std::isfinite(candidate_horizon) &&
         candidate_horizon !=
             std::numeric_limits<double>::infinity()) ||
        candidate_horizon <= 0.0
    ) {
        throw std::runtime_error(
            "CUDA parallel-window candidate horizon made no progress."
        );
    }
    const double actual_window =
        std::min(requested_window, candidate_horizon);

    check(
        cudaMemset(
            impl_->collision_validation_failures,
            0,
            sizeof(std::uint32_t)
        ),
        "cudaMemset parallel-window wall validation failures"
    );
    compute_reflecting_wall_times
        <<<particle_block_count, block_size>>>(
            impl_->position_x,
            impl_->position_y,
            impl_->position_z,
            impl_->velocity_x,
            impl_->velocity_y,
            impl_->velocity_z,
            impl_->particle_wall_time,
            impl_->particle_wall_axis_mask,
            impl_->collision_validation_failures,
            impl_->particle_count,
            device_grid,
            diameter,
            actual_window,
            tolerance
        );
    check(
        cudaGetLastError(),
        "compute parallel-window reflecting-wall times kernel launch"
    );
    std::uint32_t wall_validation_failures = 0;
    check(
        cudaMemcpy(
            &wall_validation_failures,
            impl_->collision_validation_failures,
            sizeof(wall_validation_failures),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy parallel-window wall validation failures"
    );
    if (wall_validation_failures != 0) {
        throw CudaError(
            std::to_string(wall_validation_failures) +
            " CUDA parallel-window particle(s) failed reflecting-wall "
            "validation."
        );
    }

    evaluate_collision_events(
        diameter,
        actual_window,
        tolerance
    );
    const std::size_t detected_event_count = impl_->event_count;
    std::size_t accepted_event_count = 0;

    if (impl_->event_count != 0) {
        if (impl_->event_count > impl_->batch_index_capacity) {
            free_device(impl_->independent_event_indices);
            free_device(impl_->conflicting_event_indices);
            impl_->independent_event_indices = nullptr;
            impl_->conflicting_event_indices = nullptr;
            impl_->batch_index_capacity = 0;
            allocate(
                impl_->independent_event_indices,
                impl_->event_count,
                "cudaMalloc parallel-window accepted event indices"
            );
            allocate(
                impl_->conflicting_event_indices,
                impl_->event_count,
                "cudaMalloc parallel-window spare event indices"
            );
            impl_->batch_index_capacity = impl_->event_count;
        }

        const unsigned int event_block_count =
            static_cast<unsigned int>(
                (impl_->event_count + block_size - 1U) / block_size
            );
        initialize_parallel_window_selection
            <<<particle_block_count, block_size>>>(
                impl_->particle_wall_time,
                impl_->parallel_window_minimum_time_bits,
                impl_->parallel_window_event_key,
                impl_->particle_count,
                actual_window,
                tolerance
            );
        check(
            cudaGetLastError(),
            "initialize parallel-window selection kernel launch"
        );
        select_parallel_window_minimum_times
            <<<event_block_count, block_size>>>(
                impl_->event_first,
                impl_->event_second,
                impl_->event_time,
                impl_->parallel_window_minimum_time_bits,
                impl_->event_count
            );
        check(
            cudaGetLastError(),
            "select parallel-window minimum times kernel launch"
        );
        select_parallel_window_wall_event_keys
            <<<particle_block_count, block_size>>>(
                impl_->particle_wall_time,
                impl_->parallel_window_minimum_time_bits,
                impl_->parallel_window_event_key,
                impl_->particle_count,
                actual_window,
                tolerance
            );
        check(
            cudaGetLastError(),
            "select parallel-window wall event keys kernel launch"
        );
        select_parallel_window_event_keys
            <<<event_block_count, block_size>>>(
                impl_->event_first,
                impl_->event_second,
                impl_->event_time,
                impl_->parallel_window_minimum_time_bits,
                impl_->parallel_window_event_key,
                impl_->event_count,
                tolerance
            );
        check(
            cudaGetLastError(),
            "select parallel-window event keys kernel launch"
        );

        check(
            cudaMemset(
                impl_->event_flags,
                0,
                (impl_->event_count + 1U) * sizeof(std::uint64_t)
            ),
            "cudaMemset parallel-window event flags"
        );
        mark_mutual_parallel_window_events
            <<<event_block_count, block_size>>>(
                impl_->event_first,
                impl_->event_second,
                impl_->parallel_window_event_key,
                impl_->event_flags,
                impl_->event_count
            );
        check(
            cudaGetLastError(),
            "mark mutual parallel-window events kernel launch"
        );

        std::size_t selection_scan_bytes = 0;
        check(
            cub::DeviceScan::ExclusiveSum(
                nullptr,
                selection_scan_bytes,
                impl_->event_flags,
                impl_->event_offsets,
                static_cast<int>(impl_->event_count + 1U)
            ),
            "CUB parallel-window selection storage query"
        );
        if (selection_scan_bytes > impl_->scan_temporary_bytes) {
            free_device(impl_->scan_temporary_storage);
            impl_->scan_temporary_storage = nullptr;
            check(
                cudaMalloc(
                    &impl_->scan_temporary_storage,
                    selection_scan_bytes
                ),
                "cudaMalloc parallel-window selection temporary storage"
            );
            impl_->scan_temporary_bytes = selection_scan_bytes;
        }
        check(
            cub::DeviceScan::ExclusiveSum(
                impl_->scan_temporary_storage,
                impl_->scan_temporary_bytes,
                impl_->event_flags,
                impl_->event_offsets,
                static_cast<int>(impl_->event_count + 1U)
            ),
            "CUB parallel-window selection scan"
        );

        std::uint64_t accepted_count_u64 = 0;
        check(
            cudaMemcpy(
                &accepted_count_u64,
                impl_->event_offsets + impl_->event_count,
                sizeof(accepted_count_u64),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy parallel-window accepted count to host"
        );
        accepted_event_count =
            static_cast<std::size_t>(accepted_count_u64);
        if (accepted_event_count != 0) {
            compact_event_indices<<<event_block_count, block_size>>>(
                impl_->event_flags,
                impl_->event_offsets,
                impl_->independent_event_indices,
                impl_->event_count
            );
            check(
                cudaGetLastError(),
                "compact parallel-window events kernel launch"
            );

            check(
                cudaMemset(
                    impl_->collision_validation_failures,
                    0,
                    sizeof(std::uint32_t)
                ),
                "cudaMemset parallel-window validation failures"
            );
            const unsigned int accepted_block_count =
                static_cast<unsigned int>(
                    (
                        accepted_event_count +
                        block_size -
                        1U
                    ) /
                    block_size
                );
            validate_independent_collision_events
                <<<accepted_block_count, block_size>>>(
                    impl_->position_x,
                    impl_->position_y,
                    impl_->position_z,
                    impl_->velocity_x,
                    impl_->velocity_y,
                    impl_->velocity_z,
                    impl_->event_first,
                    impl_->event_second,
                    impl_->event_time,
                    impl_->event_shift_x,
                    impl_->event_shift_y,
                    impl_->event_shift_z,
                    impl_->independent_event_indices,
                    accepted_event_count,
                    device_grid,
                    diameter,
                    tolerance,
                    impl_->collision_validation_failures
                );
            check(
                cudaGetLastError(),
                "validate parallel-window events kernel launch"
            );

            std::uint32_t validation_failures = 0;
            check(
                cudaMemcpy(
                    &validation_failures,
                    impl_->collision_validation_failures,
                    sizeof(validation_failures),
                    cudaMemcpyDeviceToHost
                ),
                "cudaMemcpy parallel-window validation failures to host"
            );
            if (validation_failures != 0) {
                throw CudaError(
                    std::to_string(validation_failures) +
                    " CUDA parallel-window event(s) failed contact "
                    "validation."
                );
            }
        }
    }

    check(
        cudaMemset(
            impl_->particle_batch_participation,
            0,
            impl_->particle_count * sizeof(std::uint32_t)
        ),
        "cudaMemset parallel-window accepted-particle markers"
    );

    if (accepted_event_count != 0) {
        const unsigned int accepted_block_count =
            static_cast<unsigned int>(
                (
                    accepted_event_count +
                    block_size -
                    1U
                ) /
                block_size
            );
        mark_parallel_window_event_particles
            <<<accepted_block_count, block_size>>>(
                impl_->event_first,
                impl_->event_second,
                impl_->independent_event_indices,
                impl_->particle_batch_participation,
                accepted_event_count
            );
        check(
            cudaGetLastError(),
            "mark CUDA parallel-window event particles kernel launch"
        );
        resolve_parallel_window_events
            <<<accepted_block_count, block_size>>>(
                impl_->position_x,
                impl_->position_y,
                impl_->position_z,
                impl_->velocity_x,
                impl_->velocity_y,
                impl_->velocity_z,
                impl_->event_first,
                impl_->event_second,
                impl_->event_time,
                impl_->event_shift_x,
                impl_->event_shift_y,
                impl_->event_shift_z,
                impl_->independent_event_indices,
                accepted_event_count,
                device_grid,
                diameter,
                actual_window
            );
        check(
            cudaGetLastError(),
            "resolve CUDA parallel-window events kernel launch"
        );
    }

    advance_parallel_window_unselected_particles
        <<<particle_block_count, block_size>>>(
            impl_->position_x,
            impl_->position_y,
            impl_->position_z,
            impl_->velocity_x,
            impl_->velocity_y,
            impl_->velocity_z,
            impl_->particle_batch_participation,
            impl_->particle_count,
            device_grid,
            diameter,
            actual_window
        );
    check(
        cudaGetLastError(),
        "advance CUDA parallel-window unselected particles kernel launch"
    );

    normalize_grid_positions<<<particle_block_count, block_size>>>(
        impl_->position_x,
        impl_->position_y,
        impl_->position_z,
        impl_->particle_count,
        device_grid,
        diameter
    );
    check(
        cudaGetLastError(),
        "normalize CUDA parallel-window positions kernel launch"
    );
    check(
        cudaDeviceSynchronize(),
        "CUDA parallel-window response synchronization"
    );

    impl_->cell_list_built = false;
    impl_->candidates_built = false;
    impl_->events_built = false;
    impl_->collision_batch_built = false;
    impl_->wall_batch_built = false;
    impl_->candidate_count = 0;
    impl_->event_count = 0;
    impl_->independent_event_count = 0;
    impl_->conflicting_event_count = 0;
    impl_->earliest_event_time = std::numeric_limits<double>::infinity();
    impl_->wall_event_count = 0;
    impl_->reflected_axis_count = 0;
    impl_->earliest_wall_time = std::numeric_limits<double>::infinity();

    build_cell_list(grid);
    build_candidate_pairs();

    std::size_t overlap_pair_count = 0;
    double minimum_surface_gap =
        std::numeric_limits<double>::infinity();
    if (impl_->candidate_count != 0) {
        const unsigned int gap_block_count =
            static_cast<unsigned int>(
                (impl_->candidate_count + block_size - 1U) / block_size
            );
        measure_candidate_surface_gaps<<<gap_block_count, block_size>>>(
            impl_->position_x,
            impl_->position_y,
            impl_->position_z,
            impl_->candidate_first,
            impl_->candidate_second,
            impl_->candidate_collision_time,
            impl_->event_flags,
            impl_->candidate_count,
            device_grid,
            diameter,
            tolerance
        );
        check(
            cudaGetLastError(),
            "measure CUDA parallel-window gaps kernel launch"
        );

        std::size_t gap_minimum_bytes = 0;
        std::size_t overlap_sum_bytes = 0;
        check(
            cub::DeviceReduce::Min(
                nullptr,
                gap_minimum_bytes,
                impl_->candidate_collision_time,
                impl_->minimum_event_time,
                static_cast<int>(impl_->candidate_count)
            ),
            "CUB parallel-window minimum-gap storage query"
        );
        check(
            cub::DeviceReduce::Sum(
                nullptr,
                overlap_sum_bytes,
                impl_->event_flags,
                impl_->event_offsets,
                static_cast<int>(impl_->candidate_count)
            ),
            "CUB parallel-window overlap-count storage query"
        );
        const std::size_t diagnostic_temporary_bytes =
            std::max(gap_minimum_bytes, overlap_sum_bytes);
        if (
            diagnostic_temporary_bytes >
            impl_->scan_temporary_bytes
        ) {
            free_device(impl_->scan_temporary_storage);
            impl_->scan_temporary_storage = nullptr;
            check(
                cudaMalloc(
                    &impl_->scan_temporary_storage,
                    diagnostic_temporary_bytes
                ),
                "cudaMalloc parallel-window diagnostic temporary storage"
            );
            impl_->scan_temporary_bytes =
                diagnostic_temporary_bytes;
        }
        check(
            cub::DeviceReduce::Min(
                impl_->scan_temporary_storage,
                impl_->scan_temporary_bytes,
                impl_->candidate_collision_time,
                impl_->minimum_event_time,
                static_cast<int>(impl_->candidate_count)
            ),
            "CUB parallel-window minimum-gap reduction"
        );
        check(
            cub::DeviceReduce::Sum(
                impl_->scan_temporary_storage,
                impl_->scan_temporary_bytes,
                impl_->event_flags,
                impl_->event_offsets,
                static_cast<int>(impl_->candidate_count)
            ),
            "CUB parallel-window overlap-count reduction"
        );

        std::uint64_t overlap_count_u64 = 0;
        check(
            cudaMemcpy(
                &minimum_surface_gap,
                impl_->minimum_event_time,
                sizeof(minimum_surface_gap),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy parallel-window minimum gap to host"
        );
        check(
            cudaMemcpy(
                &overlap_count_u64,
                impl_->event_offsets,
                sizeof(overlap_count_u64),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy parallel-window overlap count to host"
        );
        overlap_pair_count =
            static_cast<std::size_t>(overlap_count_u64);
    }

    return ParallelWindowStepResult{
        .requested_window = requested_window,
        .advanced_time = actual_window,
        .candidate_horizon = candidate_horizon,
        .candidate_pair_count = initial_candidate_count,
        .detected_collision_count = detected_event_count,
        .accepted_collision_count = accepted_event_count,
        .contended_collision_count =
            detected_event_count - accepted_event_count,
        .overlap_pair_count = overlap_pair_count,
        .minimum_surface_gap = minimum_surface_gap,
        .limited_by_candidate_horizon =
            actual_window < requested_window - tolerance,
    };
}

ParallelWindowRunResult ParticleGrid::advance_parallel_windows(
    const UniformGrid& grid,
    double duration,
    double maximum_window,
    double diameter,
    std::size_t maximum_steps,
    double tolerance
) {
    if (!impl_->particles_uploaded) {
        throw std::logic_error(
            "CUDA particles must be uploaded before parallel windows."
        );
    }
    if (!std::isfinite(duration) || duration < 0.0) {
        throw std::invalid_argument(
            "CUDA parallel-window run duration must be finite and "
            "non-negative."
        );
    }
    if (!std::isfinite(maximum_window) || maximum_window <= 0.0) {
        throw std::invalid_argument(
            "CUDA parallel-window maximum must be finite and positive."
        );
    }
    if (maximum_steps == 0) {
        throw std::invalid_argument(
            "CUDA parallel-window step limit must be positive."
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

        const ParallelWindowStepResult step =
            advance_parallel_window_step(
                grid,
                std::min(
                    maximum_window,
                    duration - result.advanced_time
                ),
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
ParticleGrid::advance_adaptive_parallel_windows(
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
    if (!impl_->particles_uploaded) {
        throw std::logic_error(
            "CUDA particles must be uploaded before adaptive parallel "
            "windows."
        );
    }
    if (!std::isfinite(duration) || duration < 0.0) {
        throw std::invalid_argument(
            "CUDA adaptive parallel-window duration must be finite and "
            "non-negative."
        );
    }
    if (!std::isfinite(maximum_window) || maximum_window <= 0.0) {
        throw std::invalid_argument(
            "CUDA adaptive parallel-window maximum must be finite and "
            "positive."
        );
    }
    if (
        !std::isfinite(minimum_window) ||
        minimum_window <= 0.0 ||
        minimum_window > maximum_window
    ) {
        throw std::invalid_argument(
            "CUDA adaptive parallel-window minimum must be positive and "
            "no larger than its maximum."
        );
    }
    if (maximum_attempts == 0) {
        throw std::invalid_argument(
            "CUDA adaptive parallel-window attempt limit must be positive."
        );
    }
    if (!std::isfinite(tolerance) || tolerance <= 0.0) {
        throw std::invalid_argument(
            "CUDA adaptive parallel-window tolerance must be finite and "
            "positive."
        );
    }
    if (successful_steps_before_growth == 0) {
        throw std::invalid_argument(
            "CUDA adaptive parallel-window growth requires at least one "
            "successful step."
        );
    }
    if (!std::isfinite(growth_factor) || growth_factor <= 1.0) {
        throw std::invalid_argument(
            "CUDA adaptive parallel-window growth factor must be finite "
            "and exceed one."
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
    const std::size_t state_bytes =
        impl_->particle_count * sizeof(double);
    double target_window = maximum_window;
    std::size_t successful_steps_at_window = 0;

    const auto copy_component = [&](
        double* destination,
        const double* source,
        const char* operation
    ) {
        check(
            cudaMemcpy(
                destination,
                source,
                state_bytes,
                cudaMemcpyDeviceToDevice
            ),
            operation
        );
    };
    const auto save_checkpoint = [&] {
        copy_component(
            impl_->checkpoint_position_x,
            impl_->position_x,
            "cudaMemcpy adaptive checkpoint position_x"
        );
        copy_component(
            impl_->checkpoint_position_y,
            impl_->position_y,
            "cudaMemcpy adaptive checkpoint position_y"
        );
        copy_component(
            impl_->checkpoint_position_z,
            impl_->position_z,
            "cudaMemcpy adaptive checkpoint position_z"
        );
        copy_component(
            impl_->checkpoint_velocity_x,
            impl_->velocity_x,
            "cudaMemcpy adaptive checkpoint velocity_x"
        );
        copy_component(
            impl_->checkpoint_velocity_y,
            impl_->velocity_y,
            "cudaMemcpy adaptive checkpoint velocity_y"
        );
        copy_component(
            impl_->checkpoint_velocity_z,
            impl_->velocity_z,
            "cudaMemcpy adaptive checkpoint velocity_z"
        );
    };
    const auto restore_checkpoint = [&] {
        copy_component(
            impl_->position_x,
            impl_->checkpoint_position_x,
            "cudaMemcpy adaptive restore position_x"
        );
        copy_component(
            impl_->position_y,
            impl_->checkpoint_position_y,
            "cudaMemcpy adaptive restore position_y"
        );
        copy_component(
            impl_->position_z,
            impl_->checkpoint_position_z,
            "cudaMemcpy adaptive restore position_z"
        );
        copy_component(
            impl_->velocity_x,
            impl_->checkpoint_velocity_x,
            "cudaMemcpy adaptive restore velocity_x"
        );
        copy_component(
            impl_->velocity_y,
            impl_->checkpoint_velocity_y,
            "cudaMemcpy adaptive restore velocity_y"
        );
        copy_component(
            impl_->velocity_z,
            impl_->checkpoint_velocity_z,
            "cudaMemcpy adaptive restore velocity_z"
        );
        impl_->cell_list_built = false;
        impl_->candidates_built = false;
        impl_->events_built = false;
        impl_->collision_batch_built = false;
        impl_->wall_batch_built = false;
        impl_->candidate_count = 0;
        impl_->event_count = 0;
    };

    while (result.advanced_time < duration - time_tolerance) {
        save_checkpoint();
        double attempt_window = std::min(
            target_window,
            duration - result.advanced_time
        );
        std::size_t refinement_depth = 0;

        while (true) {
            if (result.attempt_count >= maximum_attempts) {
                restore_checkpoint();
                result.stop_reason =
                    AdaptiveParallelWindowStopReason::attempt_limit;
                return result;
            }

            const ParallelWindowStepResult step =
                advance_parallel_window_step(
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

            restore_checkpoint();
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
                    advance_time_window(
                        grid,
                        step.advanced_time,
                        diameter,
                        exact_recovery_batch_limit,
                        tolerance
                    );
                if (!recovery.completed()) {
                    restore_checkpoint();
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

std::vector<Particle> ParticleGrid::download_particles() const {
    if (!impl_->particles_uploaded) {
        throw std::logic_error("No CUDA particle data has been uploaded.");
    }

    std::vector<double> position_x(impl_->particle_count);
    std::vector<double> position_y(impl_->particle_count);
    std::vector<double> position_z(impl_->particle_count);
    std::vector<double> velocity_x(impl_->particle_count);
    std::vector<double> velocity_y(impl_->particle_count);
    std::vector<double> velocity_z(impl_->particle_count);

    const std::size_t bytes = impl_->particle_count * sizeof(double);
    check(
        cudaMemcpy(
            position_x.data(),
            impl_->position_x,
            bytes,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy position_x to host"
    );
    check(
        cudaMemcpy(
            position_y.data(),
            impl_->position_y,
            bytes,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy position_y to host"
    );
    check(
        cudaMemcpy(
            position_z.data(),
            impl_->position_z,
            bytes,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy position_z to host"
    );
    check(
        cudaMemcpy(
            velocity_x.data(),
            impl_->velocity_x,
            bytes,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy velocity_x to host"
    );
    check(
        cudaMemcpy(
            velocity_y.data(),
            impl_->velocity_y,
            bytes,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy velocity_y to host"
    );
    check(
        cudaMemcpy(
            velocity_z.data(),
            impl_->velocity_z,
            bytes,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy velocity_z to host"
    );

    std::vector<Particle> result(impl_->particle_count);
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = Particle{
            .position = {
                position_x[index],
                position_y[index],
                position_z[index],
            },
            .velocity = {
                velocity_x[index],
                velocity_y[index],
                velocity_z[index],
            },
        };
    }
    return result;
}

CellListSnapshot ParticleGrid::download_cell_list() const {
    if (!impl_->cell_list_built) {
        throw std::logic_error("No CUDA cell list has been built.");
    }

    CellListSnapshot result{};
    result.sorted_cell_ids.resize(impl_->particle_count);
    result.particle_indices.resize(impl_->particle_count);
    result.cell_start.resize(impl_->grid.cell_count);
    result.cell_end.resize(impl_->grid.cell_count);

    const std::size_t particle_bytes =
        impl_->particle_count * sizeof(std::uint32_t);
    const std::size_t cell_bytes =
        static_cast<std::size_t>(impl_->grid.cell_count) *
        sizeof(std::uint32_t);

    check(
        cudaMemcpy(
            result.sorted_cell_ids.data(),
            impl_->sorted_cell_ids,
            particle_bytes,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy sorted_cell_ids to host"
    );
    check(
        cudaMemcpy(
            result.particle_indices.data(),
            impl_->sorted_particle_indices,
            particle_bytes,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy particle_indices to host"
    );
    check(
        cudaMemcpy(
            result.cell_start.data(),
            impl_->cell_start,
            cell_bytes,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy cell_start to host"
    );
    check(
        cudaMemcpy(
            result.cell_end.data(),
            impl_->cell_end,
            cell_bytes,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy cell_end to host"
    );

    return result;
}

CandidatePairSnapshot ParticleGrid::download_candidate_pairs() const {
    if (!impl_->candidates_built) {
        throw std::logic_error("No CUDA candidate-pair list has been built.");
    }

    std::vector<std::uint32_t> first(impl_->candidate_count);
    std::vector<std::uint32_t> second(impl_->candidate_count);
    const std::size_t bytes =
        impl_->candidate_count * sizeof(std::uint32_t);

    if (impl_->candidate_count != 0) {
        check(
            cudaMemcpy(
                first.data(),
                impl_->candidate_first,
                bytes,
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy candidate_first to host"
        );
        check(
            cudaMemcpy(
                second.data(),
                impl_->candidate_second,
                bytes,
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy candidate_second to host"
        );
    }

    CandidatePairSnapshot result{};
    result.pairs.resize(impl_->candidate_count);
    for (std::size_t index = 0; index < result.pairs.size(); ++index) {
        result.pairs[index] = CandidatePair{
            .first = first[index],
            .second = second[index],
        };
    }
    return result;
}

CollisionEventSnapshot ParticleGrid::download_collision_events() const {
    if (!impl_->events_built) {
        throw std::logic_error("No CUDA collision-event list has been built.");
    }

    std::vector<std::uint32_t> first(impl_->event_count);
    std::vector<std::uint32_t> second(impl_->event_count);
    std::vector<double> time(impl_->event_count);
    std::vector<std::int8_t> shift_x(impl_->event_count);
    std::vector<std::int8_t> shift_y(impl_->event_count);
    std::vector<std::int8_t> shift_z(impl_->event_count);

    if (impl_->event_count != 0) {
        const std::size_t index_bytes =
            impl_->event_count * sizeof(std::uint32_t);
        const std::size_t time_bytes =
            impl_->event_count * sizeof(double);
        const std::size_t shift_bytes =
            impl_->event_count * sizeof(std::int8_t);

        check(
            cudaMemcpy(
                first.data(),
                impl_->event_first,
                index_bytes,
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy event_first to host"
        );
        check(
            cudaMemcpy(
                second.data(),
                impl_->event_second,
                index_bytes,
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy event_second to host"
        );
        check(
            cudaMemcpy(
                time.data(),
                impl_->event_time,
                time_bytes,
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy event_time to host"
        );
        check(
            cudaMemcpy(
                shift_x.data(),
                impl_->event_shift_x,
                shift_bytes,
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy event_shift_x to host"
        );
        check(
            cudaMemcpy(
                shift_y.data(),
                impl_->event_shift_y,
                shift_bytes,
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy event_shift_y to host"
        );
        check(
            cudaMemcpy(
                shift_z.data(),
                impl_->event_shift_z,
                shift_bytes,
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy event_shift_z to host"
        );
    }

    CollisionEventSnapshot result{};
    result.events.resize(impl_->event_count);
    for (std::size_t index = 0; index < result.events.size(); ++index) {
        result.events[index] = CollisionEvent{
            .first = first[index],
            .second = second[index],
            .time = time[index],
            .image_shift = PeriodicImageShift{
                .x = shift_x[index],
                .y = shift_y[index],
                .z = shift_z[index],
            },
        };
    }
    return result;
}

CollisionBatchSnapshot ParticleGrid::download_collision_batch() const {
    if (!impl_->collision_batch_built) {
        throw std::logic_error(
            "No CUDA collision-batch analysis has been built."
        );
    }

    CollisionBatchSnapshot result{
        .earliest_time = impl_->earliest_event_time,
    };
    result.independent_event_indices.resize(
        impl_->independent_event_count
    );
    result.conflicting_event_indices.resize(
        impl_->conflicting_event_count
    );

    if (impl_->independent_event_count != 0) {
        check(
            cudaMemcpy(
                result.independent_event_indices.data(),
                impl_->independent_event_indices,
                impl_->independent_event_count * sizeof(std::uint32_t),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy independent event indices to host"
        );
    }
    if (impl_->conflicting_event_count != 0) {
        check(
            cudaMemcpy(
                result.conflicting_event_indices.data(),
                impl_->conflicting_event_indices,
                impl_->conflicting_event_count * sizeof(std::uint32_t),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy conflicting event indices to host"
        );
    }

    return result;
}

ReflectingWallBatchSnapshot
ParticleGrid::download_reflecting_wall_batch() const {
    if (!impl_->wall_batch_built) {
        throw std::logic_error(
            "No CUDA reflecting-wall batch has been built."
        );
    }

    std::vector<std::uint32_t> particles(impl_->wall_event_count);
    std::vector<double> times(impl_->wall_event_count);
    std::vector<std::uint8_t> axis_masks(impl_->wall_event_count);
    if (impl_->wall_event_count != 0) {
        check(
            cudaMemcpy(
                particles.data(),
                impl_->wall_event_particle,
                impl_->wall_event_count * sizeof(std::uint32_t),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy reflecting-wall particles to host"
        );
        check(
            cudaMemcpy(
                times.data(),
                impl_->wall_event_time,
                impl_->wall_event_count * sizeof(double),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy reflecting-wall times to host"
        );
        check(
            cudaMemcpy(
                axis_masks.data(),
                impl_->wall_event_axis_mask,
                impl_->wall_event_count * sizeof(std::uint8_t),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy reflecting-wall axis masks to host"
        );
    }

    ReflectingWallBatchSnapshot result{
        .earliest_time = impl_->earliest_wall_time,
    };
    result.events.resize(impl_->wall_event_count);
    for (std::size_t index = 0; index < result.events.size(); ++index) {
        result.events[index] = ReflectingWallEvent{
            .particle = particles[index],
            .time = times[index],
            .axis_mask = axis_masks[index],
        };
    }
    return result;
}

ParticleGridDeviceView ParticleGrid::device_view() noexcept {
    return ParticleGridDeviceView{
        .position_x = impl_->position_x,
        .position_y = impl_->position_y,
        .position_z = impl_->position_z,
        .velocity_x = impl_->velocity_x,
        .velocity_y = impl_->velocity_y,
        .velocity_z = impl_->velocity_z,
        .sorted_cell_ids = impl_->sorted_cell_ids,
        .particle_indices = impl_->sorted_particle_indices,
        .cell_start = impl_->cell_start,
        .cell_end = impl_->cell_end,
        .candidate_first = impl_->candidate_first,
        .candidate_second = impl_->candidate_second,
        .event_first = impl_->event_first,
        .event_second = impl_->event_second,
        .event_time = impl_->event_time,
        .event_shift_x = impl_->event_shift_x,
        .event_shift_y = impl_->event_shift_y,
        .event_shift_z = impl_->event_shift_z,
        .independent_event_indices = impl_->independent_event_indices,
        .conflicting_event_indices = impl_->conflicting_event_indices,
        .reflecting_wall_particles = impl_->wall_event_particle,
        .reflecting_wall_axis_masks = impl_->wall_event_axis_mask,
        .particle_count = impl_->particle_count,
        .candidate_count = impl_->candidate_count,
        .event_count = impl_->event_count,
        .independent_event_count = impl_->independent_event_count,
        .conflicting_event_count = impl_->conflicting_event_count,
        .reflecting_wall_event_count = impl_->wall_event_count,
        .grid = impl_->grid,
    };
}

std::size_t ParticleGrid::particle_count() const noexcept {
    return impl_->particle_count;
}

}  // namespace particles::cuda
