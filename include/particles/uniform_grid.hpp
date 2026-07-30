#pragma once

#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "particles/config.hpp"
#include "particles/hard_sphere.hpp"

namespace particles {

inline constexpr std::uint32_t empty_cell =
    std::numeric_limits<std::uint32_t>::max();

struct UniformGrid {
    Vec3 extent{};
    double cell_width{};
    std::array<std::uint32_t, 3> dimensions{};
    std::uint32_t cell_count{};
    std::array<BoundaryKind, 3> boundary{
        BoundaryKind::reflecting,
        BoundaryKind::reflecting,
        BoundaryKind::reflecting,
    };
};

struct CandidatePair {
    std::uint32_t first{};
    std::uint32_t second{};

    [[nodiscard]] auto operator<=>(const CandidatePair&) const noexcept = default;
};

struct HostCellList {
    std::vector<std::uint32_t> sorted_cell_ids;
    std::vector<std::uint32_t> particle_indices;
    std::vector<std::uint32_t> cell_start;
    std::vector<std::uint32_t> cell_end;
};

[[nodiscard]] UniformGrid make_uniform_grid(
    const DomainConfig& domain,
    double cell_width
);

[[nodiscard]] std::array<std::uint32_t, 3> cell_coordinates(
    const Vec3& position,
    const UniformGrid& grid
);

[[nodiscard]] std::uint32_t linear_cell_id(
    const Vec3& position,
    const UniformGrid& grid
);

[[nodiscard]] HostCellList build_host_cell_list(
    const std::vector<Particle>& particles,
    const UniformGrid& grid
);

[[nodiscard]] std::vector<CandidatePair> build_host_candidate_pairs(
    const std::vector<Particle>& particles,
    const UniformGrid& grid
);

}  // namespace particles
