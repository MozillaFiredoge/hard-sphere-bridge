#include <iomanip>
#include <iostream>

#include "particles/config.hpp"
#include "particles/hard_sphere.hpp"

#if PARTICLES_HAS_CUDA
#include "particles/cuda_support.hpp"
#endif

int main() {
    using particles::Particle;
    using particles::Vec3;

    const auto config = particles::default_shock_tube_config();

    std::cout << "particles " << '\n'
              << "  configured particles: " << config.particle_count << '\n'
              << "  domain: " << config.domain.extent.x << " x "
              << config.domain.extent.y << " x " << config.domain.extent.z << '\n'
              << "  hard-sphere diameter: " << config.particle_diameter << '\n';

#if PARTICLES_HAS_CUDA
    std::cout << "  CUDA devices: " << particles::cuda::device_count() << '\n';
#else
    std::cout << "  CUDA support: disabled at configure time\n";
#endif

    Particle first{
        .position = Vec3{-1.0, 0.0, 0.0},
        .velocity = Vec3{0.5, 0.0, 0.0},
    };
    Particle second{
        .position = Vec3{1.0, 0.0, 0.0},
        .velocity = Vec3{-0.5, 0.0, 0.0},
    };

    constexpr double demonstration_diameter = 1.0;
    constexpr double horizon = 10.0;
    const auto collision_time = particles::time_to_collision(
        first,
        second,
        demonstration_diameter,
        horizon
    );

    if (!collision_time) {
        std::cerr << "Analytic two-sphere smoke test did not find a collision.\n";
        return 1;
    }

    particles::advance(first, *collision_time);
    particles::advance(second, *collision_time);
    const bool resolved = particles::resolve_elastic_collision(
        first,
        second,
        demonstration_diameter
    );

    if (!resolved) {
        std::cerr << "Analytic two-sphere smoke test did not resolve a collision.\n";
        return 1;
    }

    std::cout << std::fixed << std::setprecision(6)
              << "  two-sphere collision time: " << *collision_time << '\n'
              << "  post-collision velocities: "
              << first.velocity.x << ", " << second.velocity.x << '\n';

    return 0;
}

