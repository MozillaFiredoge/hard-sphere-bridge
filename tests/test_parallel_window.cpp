#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "particles/event_driven_simulator.hpp"
#include "particles/parallel_window.hpp"

namespace {

class ParallelWindowTestFailure final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void expect_true(bool condition, std::string_view message) {
    if (!condition) {
        throw ParallelWindowTestFailure(std::string(message));
    }
}

void expect_near(
    double actual,
    double expected,
    double tolerance,
    std::string_view message
) {
    if (std::abs(actual - expected) > tolerance) {
        throw ParallelWindowTestFailure(
            std::string(message) + ": expected " +
            std::to_string(expected) + ", got " +
            std::to_string(actual)
        );
    }
}

[[nodiscard]] particles::DomainConfig periodic_domain() {
    return particles::DomainConfig{
        .extent = {12.0, 4.0, 4.0},
        .boundary = {
            particles::BoundaryKind::periodic,
            particles::BoundaryKind::periodic,
            particles::BoundaryKind::periodic,
        },
    };
}

void test_step_resolves_disjoint_events() {
    const particles::DomainConfig domain = periodic_domain();
    const particles::UniformGrid grid =
        particles::make_uniform_grid(domain, 2.0);
    std::vector<particles::Particle> state{
        particles::Particle{
            .position = {1.0, 1.0, 1.0},
            .velocity = {0.5, 0.0, 0.0},
        },
        particles::Particle{
            .position = {2.2, 1.0, 1.0},
            .velocity = {-0.5, 0.0, 0.0},
        },
        particles::Particle{
            .position = {7.0, 3.0, 3.0},
            .velocity = {0.5, 0.0, 0.0},
        },
        particles::Particle{
            .position = {8.2, 3.0, 3.0},
            .velocity = {-0.5, 0.0, 0.0},
        },
    };

    const particles::Vec3 momentum_before =
        particles::momentum(state[0]) +
        particles::momentum(state[1]) +
        particles::momentum(state[2]) +
        particles::momentum(state[3]);
    const double energy_before =
        particles::kinetic_energy(state[0]) +
        particles::kinetic_energy(state[1]) +
        particles::kinetic_energy(state[2]) +
        particles::kinetic_energy(state[3]);

    const particles::ParallelWindowStepResult result =
        particles::advance_host_parallel_window_step(
            state,
            grid,
            0.5,
            1.0
        );

    expect_near(result.advanced_time, 0.5, 1.0e-12, "step duration");
    expect_true(
        result.detected_collision_count == 2,
        "two collisions detected"
    );
    expect_true(
        result.accepted_collision_count == 2,
        "two disjoint collisions accepted"
    );
    expect_true(
        result.contended_collision_count == 0,
        "no disjoint collision is contended"
    );
    expect_true(
        result.overlap_pair_count == 0,
        "disjoint collision step has no overlap"
    );
    expect_near(state[0].position.x, 0.95, 1.0e-12, "first position");
    expect_near(state[1].position.x, 2.25, 1.0e-12, "second position");
    expect_near(state[0].velocity.x, -0.5, 1.0e-12, "first velocity");
    expect_near(state[1].velocity.x, 0.5, 1.0e-12, "second velocity");

    const particles::Vec3 momentum_after =
        particles::momentum(state[0]) +
        particles::momentum(state[1]) +
        particles::momentum(state[2]) +
        particles::momentum(state[3]);
    const double energy_after =
        particles::kinetic_energy(state[0]) +
        particles::kinetic_energy(state[1]) +
        particles::kinetic_energy(state[2]) +
        particles::kinetic_energy(state[3]);
    expect_near(
        particles::norm(momentum_after - momentum_before),
        0.0,
        1.0e-12,
        "parallel step momentum"
    );
    expect_near(
        energy_after,
        energy_before,
        1.0e-12,
        "parallel step energy"
    );
}

void test_step_reports_contended_chain_overlap() {
    const particles::DomainConfig domain = periodic_domain();
    const particles::UniformGrid grid =
        particles::make_uniform_grid(domain, 2.0);
    std::vector<particles::Particle> state{
        particles::Particle{
            .position = {1.0, 2.0, 2.0},
            .velocity = {1.0, 0.0, 0.0},
        },
        particles::Particle{
            .position = {2.2, 2.0, 2.0},
            .velocity = {0.0, 0.0, 0.0},
        },
        particles::Particle{
            .position = {3.4, 2.0, 2.0},
            .velocity = {-1.0, 0.0, 0.0},
        },
    };

    const particles::ParallelWindowStepResult result =
        particles::advance_host_parallel_window_step(
            state,
            grid,
            0.5,
            1.0
        );

    expect_true(
        result.detected_collision_count == 2,
        "chain detects both initial contacts"
    );
    expect_true(
        result.accepted_collision_count == 1,
        "mutual-earliest rule accepts one chain edge"
    );
    expect_true(
        result.contended_collision_count == 1,
        "other chain edge is contended"
    );
    expect_true(
        result.overlap_pair_count == 1,
        "unresolved chain creates a measured overlap"
    );
    expect_true(
        result.minimum_surface_gap < 0.0,
        "chain penetration depth is reported"
    );
}

void test_run_matches_exact_two_particle_reference() {
    const particles::DomainConfig domain = periodic_domain();
    const particles::UniformGrid grid =
        particles::make_uniform_grid(domain, 2.0);
    const std::vector<particles::Particle> initial{
        particles::Particle{
            .position = {1.0, 2.0, 2.0},
            .velocity = {1.0, 0.0, 0.0},
        },
        particles::Particle{
            .position = {2.4, 2.0, 2.0},
            .velocity = {-1.0, 0.0, 0.0},
        },
    };

    particles::EventDrivenSimulator reference{
        initial,
        1.0,
        domain,
    };
    reference.run_until(0.6);

    std::vector<particles::Particle> approximate = initial;
    const particles::ParallelWindowRunResult result =
        particles::advance_host_parallel_windows(
            approximate,
            grid,
            0.6,
            0.3,
            1.0
        );
    const particles::PeriodicStateError error =
        particles::measure_periodic_state_error(
            approximate,
            reference.particles(),
            grid
        );

    expect_true(result.completed(), "two-particle run completes");
    expect_true(
        result.accepted_collision_count == 1,
        "two-particle run resolves one collision"
    );
    expect_near(error.rms_position, 0.0, 1.0e-12, "position error");
    expect_near(error.rms_velocity, 0.0, 1.0e-12, "velocity error");
}

void test_state_error_uses_minimum_periodic_image() {
    const particles::DomainConfig domain = periodic_domain();
    const particles::UniformGrid grid =
        particles::make_uniform_grid(domain, 2.0);
    const std::vector<particles::Particle> approximate{
        particles::Particle{
            .position = {0.1, 1.0, 1.0},
            .velocity = {1.0, 0.0, 0.0},
        },
    };
    const std::vector<particles::Particle> reference{
        particles::Particle{
            .position = {11.9, 1.0, 1.0},
            .velocity = {0.5, 0.0, 0.0},
        },
    };

    const particles::PeriodicStateError error =
        particles::measure_periodic_state_error(
            approximate,
            reference,
            grid
        );
    expect_near(error.rms_position, 0.2, 1.0e-12, "periodic position");
    expect_near(error.rms_velocity, 0.5, 1.0e-12, "velocity error");
}

void test_step_respects_candidate_horizon() {
    const particles::DomainConfig domain = periodic_domain();
    const particles::UniformGrid grid =
        particles::make_uniform_grid(domain, 2.0);
    std::vector<particles::Particle> state{
        particles::Particle{
            .position = {1.0, 1.0, 1.0},
            .velocity = {10.0, 0.0, 0.0},
        },
    };

    const particles::ParallelWindowStepResult result =
        particles::advance_host_parallel_window_step(
            state,
            grid,
            0.5,
            1.0
        );

    expect_true(
        result.limited_by_candidate_horizon,
        "fast particle clips requested window"
    );
    expect_near(
        result.candidate_horizon,
        0.05,
        1.0e-12,
        "candidate horizon"
    );
    expect_near(
        result.advanced_time,
        0.05,
        1.0e-12,
        "clipped step duration"
    );
}

}  // namespace

void run_parallel_window_tests() {
    using Test = std::pair<std::string_view, std::function<void()>>;

    const std::vector<Test> tests{
        {
            "parallel window resolves disjoint events",
            test_step_resolves_disjoint_events,
        },
        {
            "parallel window reports chain overlap",
            test_step_reports_contended_chain_overlap,
        },
        {
            "parallel window matches two-particle reference",
            test_run_matches_exact_two_particle_reference,
        },
        {
            "parallel window periodic state error",
            test_state_error_uses_minimum_periodic_image,
        },
        {
            "parallel window candidate horizon",
            test_step_respects_candidate_horizon,
        },
    };

    std::size_t failures = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "[PASS] " << name << '\n';
        } catch (const std::exception& exception) {
            ++failures;
            std::cerr << "[FAIL] " << name << ": "
                      << exception.what() << '\n';
        }
    }

    if (failures != 0) {
        throw ParallelWindowTestFailure(
            std::to_string(failures) +
            " parallel-window test(s) failed"
        );
    }
}
