#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "particles/hard_sphere.hpp"

namespace {

using particles::Particle;
using particles::Vec3;

class TestFailure final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void expect_true(bool condition, std::string_view message) {
    if (!condition) {
        throw TestFailure(std::string(message));
    }
}

void expect_near(double actual, double expected, double tolerance, std::string_view message) {
    if (std::abs(actual - expected) > tolerance) {
        throw TestFailure(
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

template <typename Exception, typename Callable>
void expect_throws(Callable&& callable, std::string_view message) {
    try {
        std::invoke(std::forward<Callable>(callable));
    } catch (const Exception&) {
        return;
    } catch (...) {
        throw TestFailure(std::string(message) + ": wrong exception type");
    }
    throw TestFailure(std::string(message) + ": no exception");
}

void test_head_on_collision_time_and_response() {
    Particle first{
        .position = {-1.0, 0.0, 0.0},
        .velocity = {0.5, 0.0, 0.0},
    };
    Particle second{
        .position = {1.0, 0.0, 0.0},
        .velocity = {-0.5, 0.0, 0.0},
    };

    const std::optional<double> time =
        particles::time_to_collision(first, second, 1.0, 2.0);

    expect_true(time.has_value(), "head-on collision should be detected");
    expect_near(*time, 1.0, 1.0e-12, "head-on collision time");

    particles::advance(first, *time);
    particles::advance(second, *time);

    expect_near(
        particles::norm(first.position - second.position),
        1.0,
        1.0e-12,
        "particles should be at contact"
    );

    const bool resolved =
        particles::resolve_elastic_collision(first, second, 1.0);

    expect_true(resolved, "approaching particles should receive an impulse");
    expect_vec_near(first.velocity, {-0.5, 0.0, 0.0}, 1.0e-12, "first velocity");
    expect_vec_near(second.velocity, {0.5, 0.0, 0.0}, 1.0e-12, "second velocity");
}

void test_oblique_collision_conserves_momentum_and_energy() {
    Particle first{
        .position = {-1.0, 0.25, 0.0},
        .velocity = {1.0, 0.2, -0.1},
    };
    Particle second{
        .position = {1.0, -0.25, 0.0},
        .velocity = {-1.0, -0.1, 0.3},
    };

    const Vec3 momentum_before =
        particles::momentum(first) + particles::momentum(second);
    const double energy_before =
        particles::kinetic_energy(first) + particles::kinetic_energy(second);

    const auto time = particles::time_to_collision(first, second, 1.0, 2.0);
    expect_true(time.has_value(), "oblique collision should be detected");

    particles::advance(first, *time);
    particles::advance(second, *time);

    const Vec3 normal =
        (first.position - second.position) /
        particles::norm(first.position - second.position);
    const double approach_before =
        particles::dot(first.velocity - second.velocity, normal);
    expect_true(approach_before < 0.0, "particles should be approaching at contact");

    expect_true(
        particles::resolve_elastic_collision(first, second, 1.0),
        "oblique collision should resolve"
    );

    const Vec3 momentum_after =
        particles::momentum(first) + particles::momentum(second);
    const double energy_after =
        particles::kinetic_energy(first) + particles::kinetic_energy(second);
    const double approach_after =
        particles::dot(first.velocity - second.velocity, normal);

    expect_vec_near(momentum_after, momentum_before, 1.0e-12, "momentum conservation");
    expect_near(energy_after, energy_before, 1.0e-12, "energy conservation");
    expect_near(
        approach_after,
        -approach_before,
        1.0e-12,
        "normal relative speed reversal"
    );
}

void test_diverging_particles_do_not_collide() {
    const Particle first{
        .position = {-1.0, 0.0, 0.0},
        .velocity = {-1.0, 0.0, 0.0},
    };
    const Particle second{
        .position = {1.0, 0.0, 0.0},
        .velocity = {1.0, 0.0, 0.0},
    };

    expect_true(
        !particles::time_to_collision(first, second, 1.0, 10.0),
        "diverging particles must not collide"
    );
}

void test_parallel_particles_do_not_collide() {
    const Particle first{
        .position = {-1.0, 0.0, 0.0},
        .velocity = {1.0, 0.0, 0.0},
    };
    const Particle second{
        .position = {1.0, 2.0, 0.0},
        .velocity = {1.0, 0.0, 0.0},
    };

    expect_true(
        !particles::time_to_collision(first, second, 1.0, 10.0),
        "equal parallel velocities must not collide"
    );
}

void test_grazing_contact_is_not_an_impulse_event() {
    const Particle first{
        .position = {-2.0, 1.0, 0.0},
        .velocity = {1.0, 0.0, 0.0},
    };
    const Particle second{
        .position = {0.0, 0.0, 0.0},
        .velocity = {0.0, 0.0, 0.0},
    };

    expect_true(
        !particles::time_to_collision(first, second, 1.0, 3.0),
        "a perfect tangent has no normal impulse"
    );
}

void test_horizon_excludes_later_collision() {
    const Particle first{
        .position = {-1.0, 0.0, 0.0},
        .velocity = {0.5, 0.0, 0.0},
    };
    const Particle second{
        .position = {1.0, 0.0, 0.0},
        .velocity = {-0.5, 0.0, 0.0},
    };

    expect_true(
        !particles::time_to_collision(first, second, 1.0, 0.5),
        "horizon should exclude a later collision"
    );
}

void test_touching_and_separating_is_not_resolved_twice() {
    Particle first{
        .position = {-0.5, 0.0, 0.0},
        .velocity = {-0.5, 0.0, 0.0},
    };
    Particle second{
        .position = {0.5, 0.0, 0.0},
        .velocity = {0.5, 0.0, 0.0},
    };

    expect_true(
        !particles::resolve_elastic_collision(first, second, 1.0),
        "separating contact should not receive a second impulse"
    );
}

void test_overlap_is_rejected() {
    const Particle first{
        .position = {0.0, 0.0, 0.0},
        .velocity = {0.0, 0.0, 0.0},
    };
    const Particle second{
        .position = {0.5, 0.0, 0.0},
        .velocity = {0.0, 0.0, 0.0},
    };

    expect_throws<std::domain_error>(
        [&] {
            static_cast<void>(
                particles::time_to_collision(first, second, 1.0, 1.0)
            );
        },
        "overlap should be rejected"
    );
}

}  // namespace

int main() {
    using Test = std::pair<std::string_view, std::function<void()>>;

    const std::vector<Test> tests{
        {"head-on collision", test_head_on_collision_time_and_response},
        {"oblique conservation", test_oblique_collision_conserves_momentum_and_energy},
        {"diverging particles", test_diverging_particles_do_not_collide},
        {"parallel particles", test_parallel_particles_do_not_collide},
        {"grazing contact", test_grazing_contact_is_not_an_impulse_event},
        {"collision horizon", test_horizon_excludes_later_collision},
        {"separating contact", test_touching_and_separating_is_not_resolved_twice},
        {"overlap rejection", test_overlap_is_rejected},
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
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }

    std::cout << tests.size() << " test(s) passed\n";
    return 0;
}
