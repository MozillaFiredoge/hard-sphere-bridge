#include "cuda_scaling.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "particles/config.hpp"
#include "particles/hard_sphere.hpp"
#include "particles/initialization.hpp"
#include "particles/parallel_window.hpp"
#include "particles/uniform_grid.hpp"

#if PARTICLES_HAS_CUDA
#include "particles/cuda_particle_grid.hpp"
#include "particles/cuda_support.hpp"
#endif

namespace particles::app {
namespace {

[[nodiscard]] std::pair<Vec3, double> conserved_quantities(
    const std::vector<Particle>& state,
    double mass
) {
    Vec3 momentum{};
    double energy = 0.0;
    for (const Particle& particle : state) {
        momentum += particles::momentum(particle, mass);
        energy += particles::kinetic_energy(particle, mass);
    }
    return {momentum, energy};
}

[[nodiscard]] double seconds_since(
    std::chrono::steady_clock::time_point start
) {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start
    ).count();
}

}  // namespace

int run_cuda_scaling_benchmark(
    const std::filesystem::path& config_path,
    const std::filesystem::path& output_path
) {
    const CudaScalingConfig config =
        load_cuda_scaling_config(config_path);

#if !PARTICLES_HAS_CUDA
    static_cast<void>(output_path);
    throw std::runtime_error(
        "CUDA scaling support was disabled at configure time."
    );
#else
    if (cuda::device_count() == 0) {
        throw std::runtime_error(
            "CUDA scaling requires a visible CUDA device."
        );
    }
    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(
            output_path.parent_path()
        );
    }
    std::ofstream output{output_path};
    if (!output) {
        throw std::runtime_error(
            "Cannot open CUDA-scaling CSV output."
        );
    }

    const cuda::DeviceInfo device_info =
        cuda::current_device_info();
    const cuda::DeviceMemoryInfo initial_memory =
        cuda::device_memory_info();
    if (initial_memory.total_bytes == 0) {
        throw std::runtime_error(
            "Cannot query CUDA device memory."
        );
    }

    output
        << std::setprecision(17)
        << "level,device_name,compute_capability_major,"
           "compute_capability_minor,particles,random_seed,"
           "extent_x,extent_y,extent_z,"
           "grid_x,grid_y,grid_z,total_cells,cell_width,"
           "number_density,packing_fraction,placement_attempts,"
           "initialization_seconds,allocation_seconds,upload_seconds,"
           "warmup_completed,warmup_stop_reason,warmup_advanced_time,"
           "warmup_attempts,warmup_rollbacks,measurement_started,"
           "measured_completed,stop_reason,"
           "measured_advanced_time,accepted_steps,attempts,rollbacks,"
           "window_growths,maximum_refinement_depth,"
           "minimum_accepted_window,candidate_pairs,"
           "rejected_candidate_pairs,candidates_per_particle_attempt,"
           "detected_collisions,accepted_collisions,"
           "contended_collisions,rejected_overlap_pairs,"
           "minimum_surface_gap,minimum_rejected_surface_gap,"
           "measured_seconds,mean_attempt_milliseconds,"
           "committed_particle_windows_per_second,"
           "attempted_particle_windows_per_second,"
           "candidate_pairs_per_second,collisions_per_second,"
           "simulated_time_per_second,download_seconds,"
           "observed_device_bytes,observed_bytes_per_particle,"
           "device_free_bytes_after_run,device_total_bytes,"
           "momentum_drift,momentum_drift_per_particle,"
           "relative_energy_drift,conservation_pass,no_penetration\n";

    std::cout << std::setprecision(9)
              << "CUDA weak-scaling benchmark\n"
              << "  config: " << config_path.string() << '\n'
              << "  device: " << device_info.name << " (sm_"
              << device_info.compute_capability_major
              << device_info.compute_capability_minor << ")\n"
              << "  total device memory: "
              << static_cast<double>(
                     device_info.total_global_memory_bytes
                 ) /
                     (1024.0 * 1024.0 * 1024.0)
              << " GiB\n"
              << "  particle levels: " << config.level_count << '\n'
              << "  measured duration: "
              << config.measured_duration << '\n';

    std::size_t particle_count = config.base_particle_count;
    for (std::size_t level = 0;
         level < config.level_count;
         ++level) {
        const double extent = std::cbrt(
            static_cast<double>(particle_count) /
            config.number_density
        );
        const DomainConfig domain{
            .extent = {extent, extent, extent},
            .boundary = {
                BoundaryKind::periodic,
                BoundaryKind::periodic,
                BoundaryKind::periodic,
            },
        };
        const double cell_width =
            config.cell_width_factor * config.particle_diameter;
        const UniformGrid grid =
            make_uniform_grid(domain, cell_width);
        const double particle_volume =
            std::numbers::pi_v<double> *
            config.particle_diameter *
            config.particle_diameter *
            config.particle_diameter / 6.0;
        const double packing_fraction =
            config.number_density * particle_volume;

        try {
            const auto initialization_start =
                std::chrono::steady_clock::now();
            const InitializationResult initialized =
                initialize_uniform_gas(UniformGasInitialization{
                    .random_seed =
                        config.random_seed +
                        static_cast<std::uint64_t>(level),
                    .particle_count = particle_count,
                    .particle_diameter = config.particle_diameter,
                    .particle_mass = config.particle_mass,
                    .boltzmann_constant = config.boltzmann_constant,
                    .temperature = config.temperature,
                    .bulk_velocity = config.bulk_velocity,
                    .domain = domain,
                    .maximum_attempts_per_particle =
                        config.
                            maximum_placement_attempts_per_particle,
                });
            const double initialization_seconds =
                seconds_since(initialization_start);
            const auto [initial_momentum, initial_energy] =
                conserved_quantities(
                    initialized.particles,
                    config.particle_mass
                );

            const cuda::DeviceMemoryInfo memory_before =
                cuda::device_memory_info();
            const auto allocation_start =
                std::chrono::steady_clock::now();
            cuda::ParticleGrid device{particle_count};
            const double allocation_seconds =
                seconds_since(allocation_start);

            const auto upload_start =
                std::chrono::steady_clock::now();
            device.upload(initialized.particles);
            const double upload_seconds =
                seconds_since(upload_start);

            AdaptiveParallelWindowRunResult warmup_result{
                .requested_time = config.warmup_duration,
                .maximum_window = config.maximum_window,
                .minimum_window =
                    config.adaptive_window.minimum_window,
                .growth_factor =
                    config.adaptive_window.growth_factor,
                .successful_steps_before_growth =
                    config.adaptive_window.
                        successful_steps_before_growth,
            };
            if (config.warmup_duration > 0.0) {
                warmup_result =
                    device.advance_adaptive_parallel_windows(
                        grid,
                        config.warmup_duration,
                        config.maximum_window,
                        config.adaptive_window.minimum_window,
                        config.particle_diameter,
                        config.adaptive_window.maximum_attempts,
                        1.0e-12,
                        config.adaptive_window.
                            successful_steps_before_growth,
                        config.adaptive_window.growth_factor
                    );
            }

            AdaptiveParallelWindowRunResult measured_result{
                .requested_time = config.measured_duration,
                .maximum_window = config.maximum_window,
                .minimum_window =
                    config.adaptive_window.minimum_window,
                .growth_factor =
                    config.adaptive_window.growth_factor,
                .successful_steps_before_growth =
                    config.adaptive_window.
                        successful_steps_before_growth,
            };
            double measured_seconds = 0.0;
            const bool measurement_started = warmup_result.completed();
            if (measurement_started) {
                const auto measured_start =
                    std::chrono::steady_clock::now();
                measured_result =
                    device.advance_adaptive_parallel_windows(
                        grid,
                        config.measured_duration,
                        config.maximum_window,
                        config.adaptive_window.minimum_window,
                        config.particle_diameter,
                        config.adaptive_window.maximum_attempts,
                        1.0e-12,
                        config.adaptive_window.
                            successful_steps_before_growth,
                        config.adaptive_window.growth_factor
                    );
                measured_seconds = seconds_since(measured_start);
            }

            const cuda::DeviceMemoryInfo memory_after =
                cuda::device_memory_info();
            const auto download_start =
                std::chrono::steady_clock::now();
            const std::vector<Particle> final_state =
                device.download_particles();
            const double download_seconds =
                seconds_since(download_start);
            const auto [final_momentum, final_energy] =
                conserved_quantities(
                    final_state,
                    config.particle_mass
                );

            const double momentum_drift =
                norm(final_momentum - initial_momentum);
            const double momentum_drift_per_particle =
                momentum_drift /
                static_cast<double>(particle_count);
            const double relative_energy_drift =
                initial_energy == 0.0
                    ? 0.0
                    : (final_energy - initial_energy) /
                          initial_energy;
            const bool conservation_pass =
                momentum_drift_per_particle <= 1.0e-10 &&
                std::abs(relative_energy_drift) <= 1.0e-10;
            const bool no_penetration =
                warmup_result.completed() &&
                measured_result.completed();

            const std::size_t observed_device_bytes =
                memory_before.free_bytes > memory_after.free_bytes
                    ? memory_before.free_bytes -
                          memory_after.free_bytes
                    : 0U;
            const double observed_bytes_per_particle =
                static_cast<double>(observed_device_bytes) /
                static_cast<double>(particle_count);
            const std::size_t total_candidate_pairs =
                measured_result.candidate_pair_count +
                measured_result.rejected_candidate_pair_count;
            const double particle_attempts =
                static_cast<double>(particle_count) *
                static_cast<double>(measured_result.attempt_count);
            const double particle_commits =
                static_cast<double>(particle_count) *
                static_cast<double>(
                    measured_result.accepted_step_count
                );
            const double candidates_per_particle_attempt =
                measured_result.attempt_count == 0
                    ? 0.0
                    : static_cast<double>(total_candidate_pairs) /
                          particle_attempts;
            const double mean_attempt_milliseconds =
                measured_result.attempt_count == 0
                    ? 0.0
                    : 1.0e3 * measured_seconds /
                          static_cast<double>(
                              measured_result.attempt_count
                          );
            const double inverse_measured_seconds =
                measured_seconds > 0.0
                    ? 1.0 / measured_seconds
                    : 0.0;

            output
                << level << ','
                << '"' << device_info.name << '"' << ','
                << device_info.compute_capability_major << ','
                << device_info.compute_capability_minor << ','
                << particle_count << ','
                << config.random_seed +
                       static_cast<std::uint64_t>(level)
                << ','
                << extent << ',' << extent << ',' << extent << ','
                << grid.dimensions[0] << ','
                << grid.dimensions[1] << ','
                << grid.dimensions[2] << ','
                << grid.cell_count << ','
                << grid.cell_width << ','
                << config.number_density << ','
                << packing_fraction << ','
                << initialized.diagnostics.placement_attempts << ','
                << initialization_seconds << ','
                << allocation_seconds << ','
                << upload_seconds << ','
                << (warmup_result.completed() ? 1 : 0) << ','
                << adaptive_parallel_window_stop_reason_name(
                       warmup_result.stop_reason
                   ) << ','
                << warmup_result.advanced_time << ','
                << warmup_result.attempt_count << ','
                << warmup_result.rollback_count << ','
                << (measurement_started ? 1 : 0) << ','
                << (measured_result.completed() ? 1 : 0) << ','
                << (
                       measurement_started
                           ? adaptive_parallel_window_stop_reason_name(
                                 measured_result.stop_reason
                             )
                           : std::string_view{"not_run"}
                   ) << ','
                << measured_result.advanced_time << ','
                << measured_result.accepted_step_count << ','
                << measured_result.attempt_count << ','
                << measured_result.rollback_count << ','
                << measured_result.window_growth_count << ','
                << measured_result.maximum_refinement_depth << ','
                << measured_result.minimum_accepted_window << ','
                << measured_result.candidate_pair_count << ','
                << measured_result.rejected_candidate_pair_count
                << ','
                << candidates_per_particle_attempt << ','
                << measured_result.detected_collision_count << ','
                << measured_result.accepted_collision_count << ','
                << measured_result.contended_collision_count << ','
                << measured_result.rejected_overlap_pair_count << ','
                << measured_result.minimum_surface_gap << ','
                << measured_result.minimum_rejected_surface_gap << ','
                << measured_seconds << ','
                << mean_attempt_milliseconds << ','
                << particle_commits * inverse_measured_seconds << ','
                << particle_attempts * inverse_measured_seconds << ','
                << static_cast<double>(total_candidate_pairs) *
                       inverse_measured_seconds
                << ','
                << static_cast<double>(
                       measured_result.accepted_collision_count
                   ) * inverse_measured_seconds
                << ','
                << measured_result.advanced_time *
                       inverse_measured_seconds
                << ','
                << download_seconds << ','
                << observed_device_bytes << ','
                << observed_bytes_per_particle << ','
                << memory_after.free_bytes << ','
                << memory_after.total_bytes << ','
                << momentum_drift << ','
                << momentum_drift_per_particle << ','
                << relative_energy_drift << ','
                << (conservation_pass ? 1 : 0) << ','
                << (no_penetration ? 1 : 0) << '\n';
            output.flush();

            std::cout
                << "  level " << level
                << ": N=" << particle_count
                << ", cells=" << grid.cell_count
                << ", attempts="
                << (
                       measurement_started
                           ? measured_result.attempt_count
                           : warmup_result.attempt_count
                   )
                << ", rollbacks="
                << (
                       measurement_started
                           ? measured_result.rollback_count
                           : warmup_result.rollback_count
                   )
                << ", candidates/(N attempt)="
                << candidates_per_particle_attempt
                << ", attempted particle-windows/s="
                << particle_attempts * inverse_measured_seconds
                << ", memory="
                << static_cast<double>(observed_device_bytes) /
                       (1024.0 * 1024.0)
                << " MiB, status=";
            if (measurement_started) {
                std::cout
                    << adaptive_parallel_window_stop_reason_name(
                           measured_result.stop_reason
                       );
            } else {
                std::cout
                    << "warmup_"
                    << adaptive_parallel_window_stop_reason_name(
                           warmup_result.stop_reason
                       );
            }
            std::cout << '\n';
        } catch (const std::exception& exception) {
            output.flush();
            std::cerr
                << "  level " << level << " (N=" << particle_count
                << ") failed: " << exception.what() << '\n'
                << "  partial CSV: " << output_path.string() << '\n';
            return 1;
        }

        if (level + 1U < config.level_count) {
            particle_count *= 2U;
        }
    }

    if (!output) {
        throw std::runtime_error(
            "Failed while writing CUDA-scaling CSV."
        );
    }
    std::cout << "  report CSV: " << output_path.string() << '\n';
    return 0;
#endif
}

}  // namespace particles::app
