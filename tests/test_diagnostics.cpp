#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "particles/diagnostics.hpp"

namespace {

using particles::Particle;

class DiagnosticTestFailure final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void expect_true(bool condition, std::string_view message) {
    if (!condition) {
        throw DiagnosticTestFailure(std::string(message));
    }
}

void expect_near(double actual, double expected, double tolerance, std::string_view message) {
    if (std::abs(actual - expected) > tolerance) {
        throw DiagnosticTestFailure(
            std::string(message) + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
        );
    }
}

void test_known_equilibrium_moments_and_collision_rates() {
    const std::vector<Particle> particles{
        Particle{
            .position = {},
            .velocity = {-1.0, 0.0, 0.0},
        },
        Particle{
            .position = {},
            .velocity = {1.0, 0.0, 0.0},
        },
    };
    const particles::EquilibriumDiagnosticsConfig config{
        .sample_interval = 0.5,
        .velocity_x_bins = 4,
        .velocity_x_min = -2.0,
        .velocity_x_max = 2.0,
    };

    const particles::EquilibriumDiagnosticFrame frame =
        particles::measure_equilibrium(
            particles,
            2.0,
            2.0,
            1.0,
            config,
            10,
            4,
            2.0
        );

    expect_near(frame.momentum.x, 0.0, 0.0, "momentum x");
    expect_near(frame.bulk_velocity.x, 0.0, 0.0, "bulk velocity x");
    expect_near(frame.kinetic_energy, 2.0, 0.0, "kinetic energy");
    expect_near(frame.thermal_energy, 2.0, 0.0, "thermal energy");
    expect_near(frame.temperature, 2.0 / 3.0, 1.0e-15, "temperature");
    expect_near(frame.total_collision_rate, 2.0, 0.0, "total collision rate");
    expect_near(
        frame.per_particle_collision_rate,
        2.0,
        0.0,
        "per-particle collision rate"
    );
    expect_true(frame.velocity_x.samples() == 2, "histogram samples");
    expect_true(frame.velocity_x.counts[1] == 1, "negative velocity bin");
    expect_true(frame.velocity_x.counts[3] == 1, "positive velocity bin");
    expect_near(frame.velocity_x.bin_width, 1.0, 0.0, "histogram width");
    expect_near(
        frame.velocity_x.probability_density(1),
        0.5,
        0.0,
        "histogram density"
    );
}

void test_histogram_underflow_and_overflow() {
    const std::vector<Particle> particles{
        Particle{.velocity = {-10.0, 0.0, 0.0}},
        Particle{.velocity = {0.0, 0.0, 0.0}},
        Particle{.velocity = {10.0, 0.0, 0.0}},
    };
    const particles::EquilibriumDiagnosticsConfig config{
        .sample_interval = 1.0,
        .velocity_x_bins = 2,
        .velocity_x_min = -1.0,
        .velocity_x_max = 1.0,
    };

    const particles::EquilibriumDiagnosticFrame frame =
        particles::measure_equilibrium(
            particles,
            0.0,
            1.0,
            1.0,
            config,
            0,
            0,
            0.0
        );

    expect_true(frame.velocity_x.underflow == 1, "histogram underflow");
    expect_true(frame.velocity_x.overflow == 1, "histogram overflow");
    expect_true(frame.velocity_x.counts[1] == 1, "in-range sample");
    expect_true(frame.velocity_x.samples() == 3, "all histogram samples");
    expect_near(frame.total_collision_rate, 0.0, 0.0, "initial total rate");
    expect_near(
        frame.per_particle_collision_rate,
        0.0,
        0.0,
        "initial per-particle rate"
    );
}

}  // namespace

void run_diagnostic_tests() {
    using Test = std::pair<std::string_view, std::function<void()>>;

    const std::vector<Test> tests{
        {
            "known equilibrium diagnostics",
            test_known_equilibrium_moments_and_collision_rates,
        },
        {
            "velocity histogram overflow",
            test_histogram_underflow_and_overflow,
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
        throw DiagnosticTestFailure(
            std::to_string(failures) + " diagnostic test(s) failed"
        );
    }
}

