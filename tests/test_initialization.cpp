#include <cmath>
#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "particles/config.hpp"
#include "particles/event_driven_simulator.hpp"
#include "particles/initialization.hpp"

namespace {

using particles::BoundaryKind;
using particles::DomainConfig;
using particles::Particle;
using particles::Vec3;

class InitializationTestFailure final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void expect_true(bool condition, std::string_view message) {
    if (!condition) {
        throw InitializationTestFailure(std::string(message));
    }
}

void expect_near(double actual, double expected, double tolerance, std::string_view message) {
    if (std::abs(actual - expected) > tolerance) {
        throw InitializationTestFailure(
            std::string(message) + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
        );
    }
}

[[nodiscard]] Vec3 minimum_image(
    Vec3 displacement,
    const DomainConfig& domain
) {
    double* values[3]{
        &displacement.x,
        &displacement.y,
        &displacement.z,
    };
    const double lengths[3]{
        domain.extent.x,
        domain.extent.y,
        domain.extent.z,
    };

    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (domain.boundary[axis] == BoundaryKind::periodic) {
            *values[axis] -=
                std::round(*values[axis] / lengths[axis]) * lengths[axis];
        }
    }
    return displacement;
}

void expect_no_overlap(
    const std::vector<Particle>& particles,
    double diameter,
    const DomainConfig& domain
) {
    const double diameter_squared = diameter * diameter;

    for (std::size_t first = 0; first < particles.size(); ++first) {
        for (std::size_t second = first + 1; second < particles.size(); ++second) {
            const Vec3 displacement = minimum_image(
                particles[first].position - particles[second].position,
                domain
            );
            expect_true(
                particles::squared_norm(displacement) >= diameter_squared,
                "initializer produced overlapping hard spheres"
            );
        }
    }
}

void test_loads_repository_configs() {
    const std::filesystem::path source_dir{PARTICLES_SOURCE_DIR};

    const particles::ShockTubeConfig shock =
        particles::load_shock_tube_config(
            source_dir / "configs" / "shock_tube.json"
        );
    const particles::EquilibriumConfig equilibrium =
        particles::load_equilibrium_config(
            source_dir / "configs" / "equilibrium.json"
        );

    expect_true(shock.particle_count == 1'048'576, "shock particle count");
    expect_near(shock.domain.extent.x, 4.0, 0.0, "shock x extent");
    expect_true(
        shock.domain.boundary[0] == BoundaryKind::reflecting,
        "shock x boundary"
    );
    expect_near(shock.right.temperature, 0.8, 0.0, "shock right temperature");

    expect_true(equilibrium.particle_count == 64, "equilibrium particle count");
    expect_true(
        equilibrium.domain.boundary[2] == BoundaryKind::periodic,
        "equilibrium z boundary"
    );
    expect_near(equilibrium.end_time, 5.0, 0.0, "equilibrium end time");
    expect_near(
        equilibrium.diagnostics.sample_interval,
        0.25,
        0.0,
        "equilibrium sample interval"
    );
    expect_true(
        equilibrium.diagnostics.velocity_x_bins == 64,
        "equilibrium velocity bins"
    );
    expect_near(
        equilibrium.window_convergence.duration,
        1.0,
        0.0,
        "window-convergence duration"
    );
    expect_near(
        equilibrium.window_convergence.maximum_window,
        0.05,
        0.0,
        "window-convergence maximum window"
    );
    expect_true(
        equilibrium.window_convergence.refinement_levels == 3,
        "window-convergence refinement levels"
    );

    bool rejected_mismatched_experiment = false;
    try {
        static_cast<void>(
            particles::load_shock_tube_config(
                source_dir / "configs" / "equilibrium.json"
            )
        );
    } catch (const particles::ConfigError&) {
        rejected_mismatched_experiment = true;
    }
    expect_true(
        rejected_mismatched_experiment,
        "shock loader should reject an equilibrium configuration"
    );

    particles::EquilibriumConfig invalid = equilibrium;
    invalid.temperature = 0.0;

    bool rejected_invalid_temperature = false;
    try {
        particles::validate(invalid);
    } catch (const particles::ConfigError&) {
        rejected_invalid_temperature = true;
    }
    expect_true(
        rejected_invalid_temperature,
        "validation should reject a non-positive temperature"
    );
}

void test_uniform_initialization_is_reproducible_and_non_overlapping() {
    const particles::UniformGasInitialization config{
        .random_seed = 123456789,
        .particle_count = 256,
        .particle_diameter = 0.3,
        .particle_mass = 1.0,
        .boltzmann_constant = 1.0,
        .temperature = 1.25,
        .bulk_velocity = {0.2, -0.1, 0.05},
        .domain = DomainConfig{
            .extent = {8.0, 8.0, 8.0},
            .boundary = {
                BoundaryKind::periodic,
                BoundaryKind::periodic,
                BoundaryKind::periodic,
            },
        },
    };

    const particles::InitializationResult first =
        particles::initialize_uniform_gas(config);
    const particles::InitializationResult second =
        particles::initialize_uniform_gas(config);

    expect_true(first.particles.size() == config.particle_count, "particle count");
    expect_true(
        first.particles.size() == second.particles.size(),
        "repeated particle count"
    );

    for (std::size_t index = 0; index < first.particles.size(); ++index) {
        const Particle& left = first.particles[index];
        const Particle& right = second.particles[index];

        expect_true(
            left.position.x == right.position.x &&
            left.position.y == right.position.y &&
            left.position.z == right.position.z,
            "positions should be exactly reproducible"
        );
        expect_true(
            left.velocity.x == right.velocity.x &&
            left.velocity.y == right.velocity.y &&
            left.velocity.z == right.velocity.z,
            "velocities should be exactly reproducible"
        );
    }

    expect_no_overlap(first.particles, config.particle_diameter, config.domain);
    expect_true(
        first.diagnostics.minimum_separation >= config.particle_diameter,
        "reported minimum separation"
    );

    Vec3 momentum{};
    for (const Particle& particle : first.particles) {
        momentum += particles::momentum(particle, config.particle_mass);
    }
    const Vec3 target =
        static_cast<double>(config.particle_count) *
        config.particle_mass *
        config.bulk_velocity;

    expect_near(momentum.x, target.x, 1.0e-11, "corrected momentum x");
    expect_near(momentum.y, target.y, 1.0e-11, "corrected momentum y");
    expect_near(momentum.z, target.z, 1.0e-11, "corrected momentum z");
}

void test_maxwell_temperature_statistics() {
    const particles::UniformGasInitialization config{
        .random_seed = 987654321,
        .particle_count = 4096,
        .particle_diameter = 0.1,
        .particle_mass = 2.0,
        .boltzmann_constant = 1.0,
        .temperature = 1.5,
        .bulk_velocity = {},
        .domain = DomainConfig{
            .extent = {20.0, 20.0, 20.0},
            .boundary = {
                BoundaryKind::periodic,
                BoundaryKind::periodic,
                BoundaryKind::periodic,
            },
        },
    };

    const particles::InitializationResult result =
        particles::initialize_uniform_gas(config);

    expect_near(
        result.diagnostics.realized_thermal_temperature,
        config.temperature,
        0.06,
        "sampled Maxwell temperature"
    );
    expect_near(
        particles::norm(result.diagnostics.momentum_after_correction),
        0.0,
        1.0e-11,
        "zero corrected momentum"
    );
}

void test_shock_tube_density_sampling() {
    particles::ShockTubeConfig config =
        particles::default_shock_tube_config();
    config.random_seed = 424242;
    config.particle_count = 4096;
    config.particle_diameter = 0.02;

    const particles::InitializationResult result =
        particles::initialize_shock_tube(config);

    expect_true(result.particles.size() == config.particle_count, "shock count");
    expect_no_overlap(result.particles, config.particle_diameter, config.domain);

    const double left_fraction =
        static_cast<double>(result.diagnostics.left_particle_count) /
        static_cast<double>(config.particle_count);

    expect_true(
        left_fraction > 0.76 && left_fraction < 0.84,
        "shock density ratio should produce approximately 80% left particles"
    );
    expect_near(
        particles::norm(result.diagnostics.momentum_after_correction),
        0.0,
        1.0e-10,
        "shock corrected momentum"
    );
}

void test_initialized_equilibrium_run_conserves_energy() {
    particles::EquilibriumConfig config =
        particles::default_equilibrium_config();
    config.particle_count = 32;
    config.end_time = 2.0;

    particles::InitializationResult initialized =
        particles::initialize_equilibrium_gas(config);

    double energy_before = 0.0;
    for (const Particle& particle : initialized.particles) {
        energy_before += particles::kinetic_energy(
            particle,
            config.particle_mass
        );
    }

    particles::EventDrivenSimulator simulator{
        std::move(initialized.particles),
        config.particle_diameter,
        config.domain,
    };
    simulator.run_until(config.end_time);

    double energy_after = 0.0;
    for (const Particle& particle : simulator.particles()) {
        energy_after += particles::kinetic_energy(
            particle,
            config.particle_mass
        );
    }

    expect_near(
        energy_after,
        energy_before,
        1.0e-10,
        "initialized equilibrium energy conservation"
    );
}

}  // namespace

void run_initialization_tests() {
    using Test = std::pair<std::string_view, std::function<void()>>;

    const std::vector<Test> tests{
        {"repository config loading", test_loads_repository_configs},
        {
            "reproducible non-overlapping initialization",
            test_uniform_initialization_is_reproducible_and_non_overlapping,
        },
        {"Maxwell temperature statistics", test_maxwell_temperature_statistics},
        {"shock density sampling", test_shock_tube_density_sampling},
        {
            "initialized equilibrium conservation",
            test_initialized_equilibrium_run_conserves_energy,
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
        throw InitializationTestFailure(
            std::to_string(failures) + " initialization test(s) failed"
        );
    }
}
