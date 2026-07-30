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

namespace {

using particles::BoundaryKind;
using particles::DomainConfig;
using particles::EventDrivenSimulator;
using particles::EventKind;
using particles::Particle;
using particles::Vec3;

class EventTestFailure final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void expect_true(bool condition, std::string_view message) {
    if (!condition) {
        throw EventTestFailure(std::string(message));
    }
}

void expect_near(double actual, double expected, double tolerance, std::string_view message) {
    if (std::abs(actual - expected) > tolerance) {
        throw EventTestFailure(
            std::string(message) + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
        );
    }
}

void expect_vec_near(
    const Vec3& actual,
    const Vec3& expected,
    double tolerance,
    std::string_view message
) {
    expect_near(actual.x, expected.x, tolerance, std::string(message) + " (x)");
    expect_near(actual.y, expected.y, tolerance, std::string(message) + " (y)");
    expect_near(actual.z, expected.z, tolerance, std::string(message) + " (z)");
}

[[nodiscard]] DomainConfig reflecting_domain(double length = 10.0) {
    return DomainConfig{
        .extent = {length, length, length},
        .boundary = {
            BoundaryKind::reflecting,
            BoundaryKind::reflecting,
            BoundaryKind::reflecting,
        },
    };
}

[[nodiscard]] DomainConfig periodic_domain(double length = 10.0) {
    return DomainConfig{
        .extent = {length, length, length},
        .boundary = {
            BoundaryKind::periodic,
            BoundaryKind::periodic,
            BoundaryKind::periodic,
        },
    };
}

void test_reflecting_wall_event() {
    EventDrivenSimulator simulator{
        {
            Particle{
                .position = {0.75, 1.0, 1.0},
                .velocity = {1.0, 0.0, 0.0},
            },
        },
        0.5,
        reflecting_domain(2.0),
    };

    const auto event = simulator.advance_to_next_event(2.0);

    expect_true(event.has_value(), "reflecting wall event should occur");
    expect_true(event->kind == EventKind::reflecting_wall, "event kind");
    expect_true(event->axis == 0, "wall axis");
    expect_near(event->time, 1.0, 1.0e-12, "wall event time");
    expect_near(simulator.particles()[0].position.x, 1.75, 1.0e-12, "wall contact");
    expect_near(simulator.particles()[0].velocity.x, -1.0, 1.0e-12, "reflection");
    expect_true(
        simulator.statistics().reflecting_wall_collisions == 1,
        "wall collision count"
    );
}

void test_periodic_wrap_event() {
    DomainConfig domain{
        .extent = {1.0, 2.0, 2.0},
        .boundary = {
            BoundaryKind::periodic,
            BoundaryKind::reflecting,
            BoundaryKind::reflecting,
        },
    };

    EventDrivenSimulator simulator{
        {
            Particle{
                .position = {0.9, 1.0, 1.0},
                .velocity = {0.2, 0.0, 0.0},
            },
        },
        0.1,
        domain,
    };

    const auto event = simulator.advance_to_next_event(1.0);

    expect_true(event.has_value(), "periodic wrap should occur");
    expect_true(event->kind == EventKind::periodic_wrap, "event kind");
    expect_near(event->time, 0.5, 1.0e-12, "wrap time");
    expect_near(simulator.particles()[0].position.x, 0.0, 1.0e-12, "wrapped x");
    expect_near(simulator.particles()[0].velocity.x, 0.2, 1.0e-12, "unchanged vx");
    expect_true(simulator.statistics().periodic_wraps == 1, "wrap count");
}

void test_collision_across_periodic_boundary() {
    DomainConfig domain{
        .extent = {4.0, 2.0, 2.0},
        .boundary = {
            BoundaryKind::periodic,
            BoundaryKind::reflecting,
            BoundaryKind::reflecting,
        },
    };

    EventDrivenSimulator simulator{
        {
            Particle{
                .position = {0.3, 1.0, 1.0},
                .velocity = {-0.1, 0.0, 0.0},
            },
            Particle{
                .position = {3.7, 1.0, 1.0},
                .velocity = {0.1, 0.0, 0.0},
            },
        },
        0.2,
        domain,
    };

    const auto event = simulator.advance_to_next_event(3.0);

    expect_true(event.has_value(), "periodic image collision should occur");
    expect_true(event->kind == EventKind::particle_collision, "event kind");
    expect_near(event->time, 2.0, 1.0e-11, "periodic collision time");
    expect_near(simulator.particles()[0].velocity.x, 0.1, 1.0e-12, "first vx");
    expect_near(simulator.particles()[1].velocity.x, -0.1, 1.0e-12, "second vx");
    expect_true(
        simulator.statistics().particle_collisions == 1,
        "particle collision count"
    );
}

void test_three_particle_collision_chain() {
    std::vector<Particle> particles{
        Particle{
            .position = {1.0, 5.0, 5.0},
            .velocity = {1.0, 0.0, 0.0},
        },
        Particle{
            .position = {3.0, 5.0, 5.0},
            .velocity = {0.0, 0.0, 0.0},
        },
        Particle{
            .position = {5.0, 5.0, 5.0},
            .velocity = {0.0, 0.0, 0.0},
        },
    };

    const Vec3 momentum_before =
        particles::momentum(particles[0]) +
        particles::momentum(particles[1]) +
        particles::momentum(particles[2]);
    const double energy_before =
        particles::kinetic_energy(particles[0]) +
        particles::kinetic_energy(particles[1]) +
        particles::kinetic_energy(particles[2]);

    EventDrivenSimulator simulator{
        std::move(particles),
        1.0,
        reflecting_domain(),
    };

    simulator.run_until(2.0);

    expect_near(simulator.time(), 2.0, 1.0e-12, "simulation end time");
    expect_true(
        simulator.statistics().particle_collisions == 2,
        "two collisions should propagate through the chain"
    );
    expect_vec_near(
        simulator.particles()[0].velocity,
        {0.0, 0.0, 0.0},
        1.0e-12,
        "first final velocity"
    );
    expect_vec_near(
        simulator.particles()[1].velocity,
        {0.0, 0.0, 0.0},
        1.0e-12,
        "second final velocity"
    );
    expect_vec_near(
        simulator.particles()[2].velocity,
        {1.0, 0.0, 0.0},
        1.0e-12,
        "third final velocity"
    );

    const Vec3 momentum_after =
        particles::momentum(simulator.particles()[0]) +
        particles::momentum(simulator.particles()[1]) +
        particles::momentum(simulator.particles()[2]);
    const double energy_after =
        particles::kinetic_energy(simulator.particles()[0]) +
        particles::kinetic_energy(simulator.particles()[1]) +
        particles::kinetic_energy(simulator.particles()[2]);

    expect_vec_near(momentum_after, momentum_before, 1.0e-12, "chain momentum");
    expect_near(energy_after, energy_before, 1.0e-12, "chain energy");
}

void test_no_event_advances_full_horizon() {
    EventDrivenSimulator simulator{
        {
            Particle{
                .position = {5.0, 5.0, 5.0},
                .velocity = {0.1, 0.0, 0.0},
            },
        },
        1.0,
        reflecting_domain(),
    };

    const auto event = simulator.advance_to_next_event(1.0);

    expect_true(!event, "no event should occur in the short horizon");
    expect_near(simulator.time(), 1.0, 1.0e-12, "advanced time");
    expect_near(simulator.particles()[0].position.x, 5.1, 1.0e-12, "advanced x");
    expect_true(simulator.statistics().total() == 0, "event count should remain zero");
}

void test_periodic_long_time_conservation() {
    std::vector<Particle> particles{
        Particle{
            .position = {2.0, 2.0, 2.0},
            .velocity = {0.7, 0.1, 0.0},
        },
        Particle{
            .position = {5.0, 2.0, 2.0},
            .velocity = {-0.4, 0.1, 0.0},
        },
        Particle{
            .position = {2.0, 7.0, 7.0},
            .velocity = {0.3, -0.2, 0.1},
        },
        Particle{
            .position = {6.0, 7.0, 7.0},
            .velocity = {-0.6, -0.2, 0.1},
        },
    };

    Vec3 momentum_before{};
    double energy_before = 0.0;
    for (const Particle& particle : particles) {
        momentum_before += particles::momentum(particle);
        energy_before += particles::kinetic_energy(particle);
    }

    EventDrivenSimulator simulator{
        std::move(particles),
        0.5,
        periodic_domain(),
    };

    simulator.run_until(50.0);

    Vec3 momentum_after{};
    double energy_after = 0.0;
    for (const Particle& particle : simulator.particles()) {
        momentum_after += particles::momentum(particle);
        energy_after += particles::kinetic_energy(particle);
    }

    expect_near(simulator.time(), 50.0, 1.0e-12, "long run end time");
    expect_true(
        simulator.statistics().particle_collisions >= 2,
        "long run should contain particle collisions"
    );
    expect_true(
        simulator.statistics().periodic_wraps > 0,
        "long run should contain periodic wraps"
    );
    expect_vec_near(
        momentum_after,
        momentum_before,
        1.0e-11,
        "long-run momentum conservation"
    );
    expect_near(
        energy_after,
        energy_before,
        1.0e-11,
        "long-run energy conservation"
    );
}

}  // namespace

void run_event_driven_simulator_tests() {
    using Test = std::pair<std::string_view, std::function<void()>>;

    const std::vector<Test> tests{
        {"reflecting wall event", test_reflecting_wall_event},
        {"periodic wrap event", test_periodic_wrap_event},
        {"periodic image collision", test_collision_across_periodic_boundary},
        {"three-particle collision chain", test_three_particle_collision_chain},
        {"no-event horizon", test_no_event_advances_full_horizon},
        {"periodic long-time conservation", test_periodic_long_time_conservation},
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
        throw EventTestFailure(
            std::to_string(failures) + " event-driven test(s) failed"
        );
    }
}
