#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "particles/uniform_grid.hpp"

namespace {

class UniformGridTestFailure final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void expect_true(bool condition, std::string_view message) {
    if (!condition) {
        throw UniformGridTestFailure(std::string(message));
    }
}

[[nodiscard]] particles::DomainConfig test_domain() {
    return particles::DomainConfig{
        .extent = {4.0, 2.0, 2.0},
        .boundary = {
            particles::BoundaryKind::periodic,
            particles::BoundaryKind::periodic,
            particles::BoundaryKind::periodic,
        },
    };
}

void test_grid_shape_and_linear_ids() {
    const particles::UniformGrid grid =
        particles::make_uniform_grid(test_domain(), 1.0);

    expect_true(grid.dimensions[0] == 4, "grid x dimension");
    expect_true(grid.dimensions[1] == 2, "grid y dimension");
    expect_true(grid.dimensions[2] == 2, "grid z dimension");
    expect_true(grid.cell_count == 16, "grid cell count");

    expect_true(
        particles::linear_cell_id({0.1, 0.1, 0.1}, grid) == 0,
        "origin cell"
    );
    expect_true(
        particles::linear_cell_id({1.1, 0.1, 0.1}, grid) == 1,
        "x-major cell"
    );
    expect_true(
        particles::linear_cell_id({0.1, 1.1, 0.1}, grid) == 4,
        "y cell"
    );
    expect_true(
        particles::linear_cell_id({0.1, 0.1, 1.1}, grid) == 8,
        "z cell"
    );
    expect_true(
        particles::linear_cell_id({4.0, 2.0, 2.0}, grid) == 15,
        "upper boundary clamps to final cell"
    );
}

void test_host_cell_list_sort_and_ranges() {
    const particles::UniformGrid grid =
        particles::make_uniform_grid(test_domain(), 1.0);
    const std::vector<particles::Particle> particles{
        particles::Particle{.position = {0.1, 0.1, 0.1}},
        particles::Particle{.position = {3.9, 1.9, 1.9}},
        particles::Particle{.position = {1.1, 0.1, 0.1}},
        particles::Particle{.position = {0.2, 0.2, 0.2}},
    };

    const particles::HostCellList list =
        particles::build_host_cell_list(particles, grid);

    expect_true(
        list.sorted_cell_ids ==
            std::vector<std::uint32_t>{0, 0, 1, 15},
        "sorted cell ids"
    );
    expect_true(
        list.particle_indices ==
            std::vector<std::uint32_t>{0, 3, 2, 1},
        "sorted particle indices"
    );
    expect_true(list.cell_start[0] == 0, "cell zero start");
    expect_true(list.cell_end[0] == 2, "cell zero end");
    expect_true(list.cell_start[1] == 2, "cell one start");
    expect_true(list.cell_end[1] == 3, "cell one end");
    expect_true(list.cell_start[15] == 3, "cell fifteen start");
    expect_true(list.cell_end[15] == 4, "cell fifteen end");
    expect_true(
        list.cell_start[2] == particles::empty_cell &&
        list.cell_end[2] == particles::empty_cell,
        "empty cell sentinel"
    );
}

void test_position_outside_grid_is_rejected() {
    const particles::UniformGrid grid =
        particles::make_uniform_grid(test_domain(), 1.0);

    bool rejected = false;
    try {
        static_cast<void>(
            particles::linear_cell_id({-0.01, 0.0, 0.0}, grid)
        );
    } catch (const std::out_of_range&) {
        rejected = true;
    }

    expect_true(rejected, "negative position should be rejected");
}

void test_periodic_candidate_pairs_are_unique() {
    const particles::UniformGrid grid =
        particles::make_uniform_grid(test_domain(), 1.0);
    const std::vector<particles::Particle> particles{
        particles::Particle{.position = {0.1, 0.1, 0.1}},
        particles::Particle{.position = {3.9, 0.1, 0.1}},
        particles::Particle{.position = {1.1, 0.1, 0.1}},
        particles::Particle{.position = {0.2, 0.2, 0.2}},
    };

    const std::vector<particles::CandidatePair> pairs =
        particles::build_host_candidate_pairs(particles, grid);
    const std::vector<particles::CandidatePair> expected{
        {0, 1},
        {0, 2},
        {0, 3},
        {1, 3},
        {2, 3},
    };

    expect_true(pairs == expected, "periodic candidate-pair set");
}

void test_reflecting_boundary_does_not_wrap_candidates() {
    particles::DomainConfig domain = test_domain();
    domain.boundary[0] = particles::BoundaryKind::reflecting;
    const particles::UniformGrid grid =
        particles::make_uniform_grid(domain, 1.0);
    const std::vector<particles::Particle> particles{
        particles::Particle{.position = {0.1, 0.1, 0.1}},
        particles::Particle{.position = {3.9, 0.1, 0.1}},
    };

    const std::vector<particles::CandidatePair> pairs =
        particles::build_host_candidate_pairs(particles, grid);

    expect_true(pairs.empty(), "reflecting x boundary must not wrap cells");
}

}  // namespace

void run_uniform_grid_tests() {
    using Test = std::pair<std::string_view, std::function<void()>>;

    const std::vector<Test> tests{
        {"uniform grid shape and ids", test_grid_shape_and_linear_ids},
        {"host cell-list ranges", test_host_cell_list_sort_and_ranges},
        {"uniform grid bounds", test_position_outside_grid_is_rejected},
        {
            "periodic candidate-pair uniqueness",
            test_periodic_candidate_pairs_are_unique,
        },
        {
            "reflecting candidate boundary",
            test_reflecting_boundary_does_not_wrap_candidates,
        },
    };

    std::size_t failures = 0;

    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "[PASS] " << name << '\n';
        } catch (const std::exception& exception) {
            ++failures;
            std::cerr << "[FAIL] " << name << ": " << exception.what() << '\n';
        }
    }

    if (failures != 0) {
        throw UniformGridTestFailure(
            std::to_string(failures) + " uniform-grid test(s) failed"
        );
    }
}
