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

void test_multiscale_layers_share_particle_moments() {
    const std::vector<Particle> state{
        Particle{
            .position = {0.5, 0.5, 0.5},
            .velocity = {-1.0, 0.0, 0.0},
        },
        Particle{
            .position = {1.5, 0.5, 0.5},
            .velocity = {1.0, 0.0, 0.0},
        },
        Particle{
            .position = {2.5, 0.5, 0.5},
            .velocity = {2.0, 0.0, 0.0},
        },
        Particle{
            .position = {3.5, 0.5, 0.5},
            .velocity = {2.0, 0.0, 0.0},
        },
    };
    const particles::DomainConfig domain{
        .extent = {4.0, 1.0, 1.0},
        .boundary = {
            particles::BoundaryKind::reflecting,
            particles::BoundaryKind::periodic,
            particles::BoundaryKind::periodic,
        },
    };
    const particles::DiagnosticsConfig config{
        .microscopic_particle_samples = 2,
        .phase_space_x_bins = 2,
        .phase_space_vx_bins = 4,
        .phase_space_velocity_x_min = -2.0,
        .phase_space_velocity_x_max = 2.0,
        .macroscopic_x_slabs = 2,
    };

    const particles::MultiscaleDiagnosticFrame frame =
        particles::measure_multiscale(
            state,
            0.25,
            1.0,
            1.0,
            domain,
            config
        );

    expect_true(
        frame.microscopic_particles.size() == 2 &&
        frame.microscopic_particles[0].particle_index == 0 &&
        frame.microscopic_particles[1].particle_index == 2,
        "microscopic samples use stable stratified indices"
    );
    expect_true(
        frame.phase_space.samples() == state.size(),
        "phase-space histogram accounts for every particle"
    );
    expect_true(
        frame.phase_space.velocity_overflow == 2,
        "phase-space overflow counts upper-edge velocities"
    );
    expect_true(
        frame.phase_space.count(0, 1) == 1 &&
        frame.phase_space.count(0, 3) == 1,
        "phase-space x-vx bins contain the left particles"
    );

    expect_true(
        frame.macroscopic_slabs.size() == 2 &&
        frame.macroscopic_slabs[0].particle_count == 2 &&
        frame.macroscopic_slabs[1].particle_count == 2,
        "macroscopic slabs partition the particles"
    );
    expect_near(
        frame.macroscopic_slabs[0].number_density,
        1.0,
        0.0,
        "left slab number density"
    );
    expect_near(
        frame.macroscopic_slabs[0].bulk_velocity.x,
        0.0,
        0.0,
        "left slab bulk velocity"
    );
    expect_near(
        frame.macroscopic_slabs[0].temperature,
        1.0 / 3.0,
        1.0e-15,
        "left slab temperature"
    );
    expect_near(
        frame.macroscopic_slabs[0].pressure,
        1.0 / 3.0,
        1.0e-15,
        "left slab pressure"
    );
    expect_near(
        frame.macroscopic_slabs[1].bulk_velocity.x,
        2.0,
        0.0,
        "right slab bulk velocity"
    );
    expect_near(
        frame.macroscopic_slabs[1].temperature,
        0.0,
        0.0,
        "right slab temperature"
    );

    expect_true(
        frame.reconstructed_particle_count == state.size(),
        "macroscopic particle count reconstructs the microscopic count"
    );
    expect_near(frame.momentum.x, 4.0, 0.0, "global momentum");
    expect_near(frame.kinetic_energy, 5.0, 0.0, "global kinetic energy");
    expect_near(
        frame.momentum_reconstruction_error,
        0.0,
        0.0,
        "macroscopic momentum reconstruction"
    );
    expect_near(
        frame.kinetic_energy_reconstruction_error,
        0.0,
        0.0,
        "macroscopic energy reconstruction"
    );
}

void test_transport_second_moments() {
    const std::vector<Particle> state{
        Particle{.position = {0.25, 0.5, 0.5}, .velocity = {1.0, 2.0, 0.0}},
        Particle{.position = {0.75, 0.5, 0.5}, .velocity = {3.0, -1.0, 0.0}},
    };
    const particles::DomainConfig domain{.extent = {2.0, 1.0, 1.0}};
    particles::DiagnosticsConfig config{};
    config.macroscopic_x_slabs = 2;
    const auto frame = particles::measure_multiscale(
        state, 0.0, 2.0, 1.0, domain, config
    );
    const auto& slab = frame.macroscopic_slabs[0];
    expect_near(slab.velocity_x_squared_sum, 10.0, 0.0, "raw xx moment");
    expect_near(slab.velocity_x_y_product_sum, -1.0, 0.0, "raw xy moment");
    expect_near(
        2.0 * slab.velocity_x_squared_sum -
            slab.mass_density * slab.bulk_velocity.x * slab.bulk_velocity.x,
        4.0, 1.0e-14, "centered xx pressure for unit slab volume"
    );
    expect_near(
        2.0 * slab.velocity_x_y_product_sum -
            slab.mass_density * slab.bulk_velocity.x * slab.bulk_velocity.y,
        -6.0, 1.0e-14, "xy stress subtracts convective momentum"
    );
    expect_near(frame.macroscopic_slabs[1].velocity_x_squared_sum,
                0.0, 0.0, "empty slab xx moment");
    expect_near(frame.macroscopic_slabs[1].velocity_x_y_product_sum,
                0.0, 0.0, "empty slab xy moment");
}

void test_ballistic_endpoint_flux_signs() {
    const std::vector<Particle> before{
        Particle{.position = {0.25, 0.5, 0.5}, .velocity = {1.0, 0.0, 0.0}},
        Particle{.position = {0.75, 0.5, 0.5}, .velocity = {-1.0, 0.0, 0.0}},
    };
    auto after = before;
    after[0].position.x = 0.75;
    after[1].position.x = 0.25;
    const particles::DomainConfig domain{.extent = {1.0, 2.0, 1.0}};
    const auto flux = particles::compute_interface_fluxes(
        before, after, 3.0, 0.5, domain, 3
    );
    expect_true(flux.size() == 3, "flux interface count");
    expect_near(flux[1].x, 0.5, 0.0, "middle interface position");
    expect_near(flux[1].mass_flux, 0.0, 0.0, "opposite mass fluxes cancel");
    expect_near(flux[1].momentum_x_flux, 6.0, 0.0,
                "opposite crossings add positive normal momentum");
    expect_near(flux[1].energy_flux, 0.0, 0.0, "opposite energy fluxes cancel");
    expect_near(flux[0].momentum_x_flux, 0.0, 0.0, "no left boundary crossing");
    expect_near(flux[2].momentum_x_flux, 0.0, 0.0, "no right boundary crossing");
    const auto forward = particles::compute_interface_fluxes(
        {before[0]}, {after[0]}, 3.0, 0.5, domain, 3
    );
    expect_near(forward[1].mass_flux, 3.0, 0.0, "forward mass normalization");
    expect_near(forward[1].energy_flux, 1.5, 0.0, "forward energy normalization");
}

}  // namespace

void run_diagnostic_tests() {
    using Test = std::pair<std::string_view, std::function<void()>>;

    const std::vector<Test> tests{
        {"transport second velocity moments", test_transport_second_moments},
        {"ballistic endpoint flux signs", test_ballistic_endpoint_flux_signs},
        {
            "known equilibrium diagnostics",
            test_known_equilibrium_moments_and_collision_rates,
        },
        {
            "velocity histogram overflow",
            test_histogram_underflow_and_overflow,
        },
        {
            "multiscale shared particle moments",
            test_multiscale_layers_share_particle_moments,
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
