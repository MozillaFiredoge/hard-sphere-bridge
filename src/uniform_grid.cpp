#include "particles/uniform_grid.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
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

void validate_grid(const UniformGrid& grid) {
    if (!std::isfinite(grid.cell_width) || grid.cell_width <= 0.0) {
        throw std::invalid_argument("Grid cell width must be finite and positive.");
    }
    if (
        !std::isfinite(grid.extent.x) ||
        !std::isfinite(grid.extent.y) ||
        !std::isfinite(grid.extent.z) ||
        grid.extent.x <= 0.0 ||
        grid.extent.y <= 0.0 ||
        grid.extent.z <= 0.0
    ) {
        throw std::invalid_argument("Grid extents must be finite and positive.");
    }
    if (
        grid.dimensions[0] == 0 ||
        grid.dimensions[1] == 0 ||
        grid.dimensions[2] == 0 ||
        grid.cell_count == 0
    ) {
        throw std::invalid_argument("Grid dimensions must be positive.");
    }
}

}  // namespace

UniformGrid make_uniform_grid(
    const DomainConfig& domain,
    double cell_width
) {
    if (!std::isfinite(cell_width) || cell_width <= 0.0) {
        throw std::invalid_argument("Grid cell width must be finite and positive.");
    }

    UniformGrid grid{
        .extent = domain.extent,
        .cell_width = cell_width,
        .boundary = domain.boundary,
    };
    std::uint64_t total_cells = 1;

    for (std::size_t axis = 0; axis < 3; ++axis) {
        const double extent = component(domain.extent, axis);
        if (!std::isfinite(extent) || extent <= 0.0) {
            throw std::invalid_argument(
                "Grid extents must be finite and positive."
            );
        }

        // Keep every cell at least as wide as the requested safety width.
        // ceil() creates a narrow final cell; across a periodic boundary that
        // can place the second-last and first cells within the interaction
        // range even though a 27-cell stencil does not connect them.
        const double dimension = std::max(
            1.0,
            std::floor(extent / cell_width)
        );
        if (
            dimension <= 0.0 ||
            dimension >
                static_cast<double>(std::numeric_limits<std::uint32_t>::max())
        ) {
            throw std::overflow_error(
                "A grid dimension exceeds the uint32 range."
            );
        }

        grid.dimensions[axis] = static_cast<std::uint32_t>(dimension);
        total_cells *= grid.dimensions[axis];

        if (total_cells > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error(
                "Uniform grid contains more than uint32-addressable cells."
            );
        }
    }

    grid.cell_count = static_cast<std::uint32_t>(total_cells);
    return grid;
}

std::array<std::uint32_t, 3> cell_coordinates(
    const Vec3& position,
    const UniformGrid& grid
) {
    validate_grid(grid);

    std::array<std::uint32_t, 3> coordinates{};

    for (std::size_t axis = 0; axis < 3; ++axis) {
        const double value = component(position, axis);
        const double extent = component(grid.extent, axis);

        if (!std::isfinite(value) || value < 0.0 || value > extent) {
            throw std::out_of_range("Particle position lies outside the grid.");
        }

        const double raw_coordinate = std::floor(value / grid.cell_width);
        const std::uint64_t unclamped =
            static_cast<std::uint64_t>(raw_coordinate);
        coordinates[axis] = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(
                unclamped,
                static_cast<std::uint64_t>(grid.dimensions[axis] - 1U)
            )
        );
    }

    return coordinates;
}

std::uint32_t linear_cell_id(
    const Vec3& position,
    const UniformGrid& grid
) {
    const std::array<std::uint32_t, 3> coordinate =
        cell_coordinates(position, grid);

    const std::uint64_t id =
        static_cast<std::uint64_t>(coordinate[0]) +
        static_cast<std::uint64_t>(grid.dimensions[0]) *
            (
                static_cast<std::uint64_t>(coordinate[1]) +
                static_cast<std::uint64_t>(grid.dimensions[1]) *
                    static_cast<std::uint64_t>(coordinate[2])
            );

    if (id >= grid.cell_count) {
        throw std::logic_error("Computed cell id exceeds the grid.");
    }
    return static_cast<std::uint32_t>(id);
}

HostCellList build_host_cell_list(
    const std::vector<Particle>& particles,
    const UniformGrid& grid
) {
    validate_grid(grid);

    if (particles.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(
            "Host cell list supports at most uint32-addressable particles."
        );
    }

    std::vector<std::pair<std::uint32_t, std::uint32_t>> pairs;
    pairs.reserve(particles.size());

    for (std::size_t index = 0; index < particles.size(); ++index) {
        pairs.emplace_back(
            linear_cell_id(particles[index].position, grid),
            static_cast<std::uint32_t>(index)
        );
    }

    std::sort(pairs.begin(), pairs.end());

    HostCellList result{};
    result.sorted_cell_ids.resize(pairs.size());
    result.particle_indices.resize(pairs.size());
    result.cell_start.assign(grid.cell_count, empty_cell);
    result.cell_end.assign(grid.cell_count, empty_cell);

    for (std::size_t sorted_index = 0;
         sorted_index < pairs.size();
         ++sorted_index) {
        const auto [cell_id, particle_index] = pairs[sorted_index];
        result.sorted_cell_ids[sorted_index] = cell_id;
        result.particle_indices[sorted_index] = particle_index;

        if (
            sorted_index == 0 ||
            pairs[sorted_index - 1].first != cell_id
        ) {
            result.cell_start[cell_id] =
                static_cast<std::uint32_t>(sorted_index);
        }
        if (
            sorted_index + 1 == pairs.size() ||
            pairs[sorted_index + 1].first != cell_id
        ) {
            result.cell_end[cell_id] =
                static_cast<std::uint32_t>(sorted_index + 1);
        }
    }

    return result;
}

std::vector<CandidatePair> build_host_candidate_pairs(
    const std::vector<Particle>& particles,
    const UniformGrid& grid
) {
    const HostCellList cell_list = build_host_cell_list(particles, grid);
    std::vector<CandidatePair> candidates;

    for (std::size_t particle_index = 0;
         particle_index < particles.size();
         ++particle_index) {
        const std::array<std::uint32_t, 3> center =
            cell_coordinates(particles[particle_index].position, grid);
        std::array<std::uint32_t, 27> neighbor_cell_ids{};
        std::size_t neighbor_cell_count = 0;

        for (int offset_z = -1; offset_z <= 1; ++offset_z) {
            for (int offset_y = -1; offset_y <= 1; ++offset_y) {
                for (int offset_x = -1; offset_x <= 1; ++offset_x) {
                    const int offsets[3]{offset_x, offset_y, offset_z};
                    std::array<std::uint32_t, 3> neighbor{};
                    bool valid = true;

                    for (std::size_t axis = 0; axis < 3; ++axis) {
                        std::int64_t coordinate =
                            static_cast<std::int64_t>(center[axis]) +
                            offsets[axis];
                        const std::int64_t dimension =
                            static_cast<std::int64_t>(grid.dimensions[axis]);

                        if (grid.boundary[axis] == BoundaryKind::periodic) {
                            coordinate %= dimension;
                            if (coordinate < 0) {
                                coordinate += dimension;
                            }
                        } else if (
                            coordinate < 0 ||
                            coordinate >= dimension
                        ) {
                            valid = false;
                            break;
                        }

                        neighbor[axis] =
                            static_cast<std::uint32_t>(coordinate);
                    }

                    if (!valid) {
                        continue;
                    }

                    const std::uint64_t neighbor_cell_id =
                        static_cast<std::uint64_t>(neighbor[0]) +
                        static_cast<std::uint64_t>(grid.dimensions[0]) *
                            (
                                static_cast<std::uint64_t>(neighbor[1]) +
                                static_cast<std::uint64_t>(
                                    grid.dimensions[1]
                                ) *
                                    static_cast<std::uint64_t>(neighbor[2])
                            );
                    const std::uint32_t cell_id =
                        static_cast<std::uint32_t>(neighbor_cell_id);

                    bool duplicate_cell = false;
                    for (std::size_t seen = 0;
                         seen < neighbor_cell_count;
                         ++seen) {
                        if (neighbor_cell_ids[seen] == cell_id) {
                            duplicate_cell = true;
                            break;
                        }
                    }
                    if (duplicate_cell) {
                        continue;
                    }
                    neighbor_cell_ids[neighbor_cell_count++] = cell_id;

                    const std::uint32_t start = cell_list.cell_start[cell_id];
                    if (start == empty_cell) {
                        continue;
                    }
                    const std::uint32_t end = cell_list.cell_end[cell_id];

                    for (std::uint32_t sorted_index = start;
                         sorted_index < end;
                         ++sorted_index) {
                        const std::uint32_t other =
                            cell_list.particle_indices[sorted_index];
                        if (particle_index < other) {
                            candidates.push_back(CandidatePair{
                                .first =
                                    static_cast<std::uint32_t>(particle_index),
                                .second = other,
                            });
                        }
                    }
                }
            }
        }
    }

    std::sort(candidates.begin(), candidates.end());
    const auto unique_end = std::unique(candidates.begin(), candidates.end());
    candidates.erase(unique_end, candidates.end());
    return candidates;
}

}  // namespace particles
