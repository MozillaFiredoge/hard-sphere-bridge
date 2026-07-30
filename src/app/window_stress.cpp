#include "window_stress.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "particles/config.hpp"
#include "particles/event_driven_simulator.hpp"
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

struct ApproximateRun {
    std::string backend;
    std::size_t level{};
    double maximum_window{};
    ParallelWindowRunResult result{};
    std::vector<Particle> state;
    double wall_seconds{};
};

struct ReferenceSnapshot {
    double time{};
    std::vector<Particle> state;
    std::size_t collision_count{};
};

struct SummaryAccumulator {
    std::size_t run_count{};
    std::size_t completed_count{};
    std::size_t overlap_stop_count{};
    std::size_t total_detected_collisions{};
    std::size_t total_accepted_collisions{};
    std::size_t total_contended_collisions{};
    long double collision_error_sum{};
    long double absolute_collision_error_sum{};
    long double rms_position_error_sum{};
    long double rms_velocity_error_sum{};
    long double momentum_drift_sum{};
    long double wall_seconds_sum{};
    double maximum_rms_position_error{};
    double maximum_rms_velocity_error{};
    double maximum_absolute_relative_energy_drift{};
    double minimum_surface_gap{std::numeric_limits<double>::infinity()};
    double earliest_failure_time{std::numeric_limits<double>::infinity()};
};

struct AdaptiveApproximateRun {
    std::string backend;
    AdaptiveParallelWindowRunResult result{};
    std::vector<Particle> state;
    double wall_seconds{};
};

struct AdaptiveSummaryAccumulator {
    std::size_t run_count{};
    std::size_t completed_count{};
    std::size_t accepted_step_count{};
    std::size_t attempt_count{};
    std::size_t rollback_count{};
    std::size_t window_growth_count{};
    std::size_t maximum_refinement_depth{};
    std::size_t accepted_collision_count{};
    std::size_t contended_collision_count{};
    std::size_t rejected_overlap_pair_count{};
    std::int64_t collision_count_error_sum{};
    std::uint64_t absolute_collision_count_error_sum{};
    long double rms_position_error_sum{};
    long double rms_velocity_error_sum{};
    long double momentum_drift_sum{};
    long double wall_seconds_sum{};
    double maximum_rms_position_error{};
    double maximum_rms_velocity_error{};
    double maximum_absolute_relative_energy_drift{};
    double minimum_accepted_window{
        std::numeric_limits<double>::infinity()
    };
    double minimum_surface_gap{std::numeric_limits<double>::infinity()};
    double minimum_rejected_surface_gap{
        std::numeric_limits<double>::infinity()
    };
};

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

[[nodiscard]] const ReferenceSnapshot& find_reference_snapshot(
    const std::vector<ReferenceSnapshot>& snapshots,
    double time,
    double tolerance
) {
    const auto found = std::find_if(
        snapshots.begin(),
        snapshots.end(),
        [&](const ReferenceSnapshot& snapshot) {
            return std::abs(snapshot.time - time) <= tolerance;
        }
    );
    if (found == snapshots.end()) {
        throw std::logic_error(
            "No exact reference snapshot matches an approximate stop time."
        );
    }
    return *found;
}

}  // namespace

int run_parallel_window_stress(
    const std::filesystem::path& config_path,
    const std::filesystem::path& output_directory
) {
    const ParallelWindowStressConfig config =
        load_parallel_window_stress_config(config_path);
    const UniformGrid grid = make_uniform_grid(
        config.domain,
        2.0 * config.particle_diameter
    );
    std::filesystem::create_directories(output_directory);
    const std::filesystem::path run_path =
        output_directory / "runs.csv";
    const std::filesystem::path summary_path =
        output_directory / "summary.csv";
    std::ofstream run_output{run_path};
    std::ofstream summary_output{summary_path};
    if (!run_output || !summary_output) {
        throw std::runtime_error(
            "Cannot open parallel-window stress output files."
        );
    }

    run_output
        << std::setprecision(17)
        << "seed,backend,level,maximum_window,requested_time,"
           "advanced_time,completed,stop_reason,steps,"
           "horizon_limited_steps,candidate_pairs,"
           "detected_collisions,accepted_collisions,"
           "contended_collisions,reference_collisions_at_time,"
           "reference_final_collisions,collision_count_error,"
           "overlap_pairs,minimum_surface_gap,rms_position_error,"
           "maximum_position_error,rms_velocity_error,"
           "maximum_velocity_error,momentum_drift,"
           "relative_energy_drift,wall_seconds,"
           "reference_wall_seconds,cpu_rms_position_difference,"
           "cpu_rms_velocity_difference,placement_attempts,"
           "initial_minimum_separation,realized_temperature\n";
    summary_output
        << std::setprecision(17)
        << "backend,level,maximum_window,seeds,completed,"
           "completion_rate,overlap_stops,overlap_stop_rate,"
           "detected_collisions,accepted_collisions,"
           "contended_collisions,contended_rate,"
           "mean_collision_count_error,"
           "mean_absolute_collision_count_error,"
           "mean_rms_position_error,maximum_rms_position_error,"
           "mean_rms_velocity_error,maximum_rms_velocity_error,"
           "minimum_surface_gap,earliest_failure_time,"
           "mean_momentum_drift,"
           "maximum_absolute_relative_energy_drift,"
           "mean_wall_seconds\n";

#if PARTICLES_HAS_CUDA
    const bool run_cuda = cuda::device_count() != 0;
#else
    constexpr bool run_cuda = false;
#endif
    const std::size_t level_count =
        config.window_convergence.refinement_levels;
    std::vector<SummaryAccumulator> cpu_summary(level_count);
    std::vector<SummaryAccumulator> cuda_summary(level_count);
    const double time_tolerance =
        32.0 *
        std::numeric_limits<double>::epsilon() *
        std::max(1.0, config.window_convergence.duration);

    std::cout << std::setprecision(9)
              << "random parallel-window stress experiment\n"
              << "  config: " << config_path.string() << '\n'
              << "  particles: " << config.particle_count << '\n'
              << "  seeds: " << config.seed_count << '\n'
              << "  duration: "
              << config.window_convergence.duration << '\n'
              << "  CUDA backend: "
              << (run_cuda ? "enabled" : "no visible device") << '\n';

    for (std::size_t seed_index = 0;
         seed_index < config.seed_count;
         ++seed_index) {
        const std::uint64_t seed =
            config.base_random_seed +
            static_cast<std::uint64_t>(seed_index);
        const InitializationResult initialized =
            initialize_uniform_gas(UniformGasInitialization{
                .random_seed = seed,
                .particle_count = config.particle_count,
                .particle_diameter = config.particle_diameter,
                .particle_mass = config.particle_mass,
                .boltzmann_constant = config.boltzmann_constant,
                .temperature = config.temperature,
                .bulk_velocity = config.bulk_velocity,
                .domain = config.domain,
            });
        const auto [initial_momentum, initial_energy] =
            conserved_quantities(
                initialized.particles,
                config.particle_mass
            );

        std::vector<ApproximateRun> runs;
        runs.reserve(level_count * (run_cuda ? 2U : 1U));
        for (std::size_t level = 0; level < level_count; ++level) {
            const double maximum_window = std::ldexp(
                config.window_convergence.maximum_window,
                -static_cast<int>(level)
            );
            std::vector<Particle> state = initialized.particles;
            const auto start = std::chrono::steady_clock::now();
            const ParallelWindowRunResult result =
                advance_host_parallel_windows(
                    state,
                    grid,
                    config.window_convergence.duration,
                    maximum_window,
                    config.particle_diameter,
                    config.window_convergence.maximum_steps
                );
            const double wall_seconds =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start
                ).count();
            runs.push_back(ApproximateRun{
                .backend = "cpu",
                .level = level,
                .maximum_window = maximum_window,
                .result = result,
                .state = std::move(state),
                .wall_seconds = wall_seconds,
            });
        }

#if PARTICLES_HAS_CUDA
        if (run_cuda) {
            for (std::size_t level = 0; level < level_count; ++level) {
                const double maximum_window = std::ldexp(
                    config.window_convergence.maximum_window,
                    -static_cast<int>(level)
                );
                cuda::ParticleGrid device{initialized.particles.size()};
                device.upload(initialized.particles);
                const auto start = std::chrono::steady_clock::now();
                const ParallelWindowRunResult result =
                    device.advance_parallel_windows(
                        grid,
                        config.window_convergence.duration,
                        maximum_window,
                        config.particle_diameter,
                        config.window_convergence.maximum_steps
                    );
                const double wall_seconds =
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - start
                    ).count();
                runs.push_back(ApproximateRun{
                    .backend = "cuda",
                    .level = level,
                    .maximum_window = maximum_window,
                    .result = result,
                    .state = device.download_particles(),
                    .wall_seconds = wall_seconds,
                });
            }
        }
#endif

        std::vector<double> reference_times{
            config.window_convergence.duration,
        };
        for (const ApproximateRun& run : runs) {
            reference_times.push_back(run.result.advanced_time);
        }
        std::sort(reference_times.begin(), reference_times.end());
        reference_times.erase(
            std::unique(
                reference_times.begin(),
                reference_times.end(),
                [&](double left, double right) {
                    return std::abs(left - right) <= time_tolerance;
                }
            ),
            reference_times.end()
        );

        EventDrivenSimulator reference{
            initialized.particles,
            config.particle_diameter,
            config.domain,
        };
        std::vector<ReferenceSnapshot> snapshots;
        snapshots.reserve(reference_times.size());
        const auto reference_start = std::chrono::steady_clock::now();
        for (const double time : reference_times) {
            reference.run_until(time);
            snapshots.push_back(ReferenceSnapshot{
                .time = time,
                .state = reference.particles(),
                .collision_count =
                    reference.statistics().particle_collisions,
            });
        }
        const double reference_wall_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - reference_start
            ).count();
        const ReferenceSnapshot& final_snapshot =
            find_reference_snapshot(
                snapshots,
                config.window_convergence.duration,
                time_tolerance
            );

        std::cout << "  seed " << seed
                  << ": exact collisions="
                  << final_snapshot.collision_count
                  << ", placement attempts="
                  << initialized.diagnostics.placement_attempts << '\n';

        for (const ApproximateRun& run : runs) {
            const ReferenceSnapshot& matched_reference =
                find_reference_snapshot(
                    snapshots,
                    run.result.advanced_time,
                    time_tolerance
                );
            const PeriodicStateError reference_error =
                measure_periodic_state_error(
                    run.state,
                    matched_reference.state,
                    grid
                );
            const auto [momentum, energy] = conserved_quantities(
                run.state,
                config.particle_mass
            );
            const double momentum_drift =
                norm(momentum - initial_momentum);
            const double relative_energy_drift =
                initial_energy == 0.0
                    ? 0.0
                    : (energy - initial_energy) / initial_energy;
            const std::int64_t collision_count_error =
                static_cast<std::int64_t>(
                    run.result.accepted_collision_count
                ) -
                static_cast<std::int64_t>(
                    matched_reference.collision_count
                );

            const double not_available =
                std::numeric_limits<double>::quiet_NaN();
            PeriodicStateError cpu_difference{
                .rms_position = run.backend == "cpu"
                    ? 0.0
                    : not_available,
                .maximum_position = run.backend == "cpu"
                    ? 0.0
                    : not_available,
                .rms_velocity = run.backend == "cpu"
                    ? 0.0
                    : not_available,
                .maximum_velocity = run.backend == "cpu"
                    ? 0.0
                    : not_available,
            };
            if (run.backend == "cuda") {
                const auto cpu_run = std::find_if(
                    runs.begin(),
                    runs.end(),
                    [&](const ApproximateRun& candidate) {
                        return candidate.backend == "cpu" &&
                               candidate.level == run.level &&
                               std::abs(
                                   candidate.result.advanced_time -
                                   run.result.advanced_time
                               ) <= time_tolerance;
                    }
                );
                if (cpu_run != runs.end()) {
                    cpu_difference = measure_periodic_state_error(
                        run.state,
                        cpu_run->state,
                        grid
                    );
                }
            }

            run_output
                << seed << ','
                << run.backend << ','
                << run.level << ','
                << run.maximum_window << ','
                << run.result.requested_time << ','
                << run.result.advanced_time << ','
                << (run.result.completed() ? 1 : 0) << ','
                << parallel_window_stop_reason_name(
                       run.result.stop_reason
                   ) << ','
                << run.result.step_count << ','
                << run.result.horizon_limited_step_count << ','
                << run.result.candidate_pair_count << ','
                << run.result.detected_collision_count << ','
                << run.result.accepted_collision_count << ','
                << run.result.contended_collision_count << ','
                << matched_reference.collision_count << ','
                << final_snapshot.collision_count << ','
                << collision_count_error << ','
                << run.result.overlap_pair_count << ','
                << run.result.minimum_surface_gap << ','
                << reference_error.rms_position << ','
                << reference_error.maximum_position << ','
                << reference_error.rms_velocity << ','
                << reference_error.maximum_velocity << ','
                << momentum_drift << ','
                << relative_energy_drift << ','
                << run.wall_seconds << ','
                << reference_wall_seconds << ','
                << cpu_difference.rms_position << ','
                << cpu_difference.rms_velocity << ','
                << initialized.diagnostics.placement_attempts << ','
                << initialized.diagnostics.minimum_separation << ','
                << initialized.diagnostics.realized_thermal_temperature
                << '\n';

            SummaryAccumulator& summary =
                run.backend == "cpu"
                    ? cpu_summary[run.level]
                    : cuda_summary[run.level];
            ++summary.run_count;
            summary.completed_count +=
                run.result.completed() ? 1U : 0U;
            summary.overlap_stop_count +=
                run.result.stop_reason ==
                        ParallelWindowStopReason::overlap_detected
                    ? 1U
                    : 0U;
            summary.total_detected_collisions +=
                run.result.detected_collision_count;
            summary.total_accepted_collisions +=
                run.result.accepted_collision_count;
            summary.total_contended_collisions +=
                run.result.contended_collision_count;
            summary.collision_error_sum +=
                static_cast<long double>(collision_count_error);
            summary.absolute_collision_error_sum +=
                static_cast<long double>(
                    std::abs(collision_count_error)
                );
            summary.rms_position_error_sum +=
                reference_error.rms_position;
            summary.rms_velocity_error_sum +=
                reference_error.rms_velocity;
            summary.momentum_drift_sum += momentum_drift;
            summary.wall_seconds_sum += run.wall_seconds;
            summary.maximum_rms_position_error = std::max(
                summary.maximum_rms_position_error,
                reference_error.rms_position
            );
            summary.maximum_rms_velocity_error = std::max(
                summary.maximum_rms_velocity_error,
                reference_error.rms_velocity
            );
            summary.maximum_absolute_relative_energy_drift = std::max(
                summary.maximum_absolute_relative_energy_drift,
                std::abs(relative_energy_drift)
            );
            summary.minimum_surface_gap = std::min(
                summary.minimum_surface_gap,
                run.result.minimum_surface_gap
            );
            if (!run.result.completed()) {
                summary.earliest_failure_time = std::min(
                    summary.earliest_failure_time,
                    run.result.advanced_time
                );
            }

            std::cout << "    " << run.backend
                      << " level " << run.level
                      << ": "
                      << parallel_window_stop_reason_name(
                             run.result.stop_reason
                         )
                      << ", accepted="
                      << run.result.accepted_collision_count
                      << ", contended="
                      << run.result.contended_collision_count
                      << ", collision error="
                      << collision_count_error
                      << ", rms x="
                      << reference_error.rms_position
                      << ", min gap="
                      << run.result.minimum_surface_gap << '\n';
        }
    }

    const auto write_summary = [&](
        std::string_view backend,
        const std::vector<SummaryAccumulator>& accumulators
    ) {
        for (std::size_t level = 0; level < level_count; ++level) {
            const SummaryAccumulator& value = accumulators[level];
            if (value.run_count == 0) {
                continue;
            }
            const double count =
                static_cast<double>(value.run_count);
            const double contended_rate =
                value.total_detected_collisions == 0
                    ? 0.0
                    : static_cast<double>(
                          value.total_contended_collisions
                      ) /
                      static_cast<double>(
                          value.total_detected_collisions
                      );
            const double earliest_failure =
                std::isfinite(value.earliest_failure_time)
                    ? value.earliest_failure_time
                    : std::numeric_limits<double>::quiet_NaN();
            const double maximum_window = std::ldexp(
                config.window_convergence.maximum_window,
                -static_cast<int>(level)
            );

            summary_output
                << backend << ','
                << level << ','
                << maximum_window << ','
                << value.run_count << ','
                << value.completed_count << ','
                << static_cast<double>(value.completed_count) / count
                << ','
                << value.overlap_stop_count << ','
                << static_cast<double>(value.overlap_stop_count) / count
                << ','
                << value.total_detected_collisions << ','
                << value.total_accepted_collisions << ','
                << value.total_contended_collisions << ','
                << contended_rate << ','
                << static_cast<double>(
                       value.collision_error_sum /
                       static_cast<long double>(value.run_count)
                   ) << ','
                << static_cast<double>(
                       value.absolute_collision_error_sum /
                       static_cast<long double>(value.run_count)
                   ) << ','
                << static_cast<double>(
                       value.rms_position_error_sum /
                       static_cast<long double>(value.run_count)
                   ) << ','
                << value.maximum_rms_position_error << ','
                << static_cast<double>(
                       value.rms_velocity_error_sum /
                       static_cast<long double>(value.run_count)
                   ) << ','
                << value.maximum_rms_velocity_error << ','
                << value.minimum_surface_gap << ','
                << earliest_failure << ','
                << static_cast<double>(
                       value.momentum_drift_sum /
                       static_cast<long double>(value.run_count)
                   ) << ','
                << value.maximum_absolute_relative_energy_drift << ','
                << static_cast<double>(
                       value.wall_seconds_sum /
                       static_cast<long double>(value.run_count)
                   ) << '\n';

            std::cout << "  summary " << backend
                      << " level " << level
                      << ": completed="
                      << value.completed_count << " / "
                      << value.run_count
                      << ", contended rate=" << contended_rate
                      << ", mean |collision error|="
                      << static_cast<double>(
                             value.absolute_collision_error_sum /
                             static_cast<long double>(value.run_count)
                         )
                      << ", mean rms x="
                      << static_cast<double>(
                             value.rms_position_error_sum /
                             static_cast<long double>(value.run_count)
                         )
                      << '\n';
        }
    };
    write_summary("cpu", cpu_summary);
    write_summary("cuda", cuda_summary);

    if (!run_output || !summary_output) {
        throw std::runtime_error(
            "Failed while writing parallel-window stress reports."
        );
    }
    std::cout << "  run CSV: " << run_path.string() << '\n'
              << "  summary CSV: " << summary_path.string() << '\n';
    return 0;
}

int run_adaptive_parallel_window_stress(
    const std::filesystem::path& config_path,
    const std::filesystem::path& output_directory
) {
    const ParallelWindowStressConfig config =
        load_parallel_window_stress_config(config_path);
    const UniformGrid grid = make_uniform_grid(
        config.domain,
        2.0 * config.particle_diameter
    );
    std::filesystem::create_directories(output_directory);
    const std::filesystem::path run_path =
        output_directory / "runs.csv";
    const std::filesystem::path summary_path =
        output_directory / "summary.csv";
    std::ofstream run_output{run_path};
    std::ofstream summary_output{summary_path};
    if (!run_output || !summary_output) {
        throw std::runtime_error(
            "Cannot open adaptive parallel-window stress output files."
        );
    }

    run_output
        << std::setprecision(17)
        << "seed,backend,requested_time,advanced_time,completed,"
           "stop_reason,maximum_window,minimum_window,"
           "growth_factor,successful_steps_before_growth,"
           "accepted_steps,attempts,rollbacks,maximum_refinement_depth,"
           "window_growths,"
           "minimum_attempted_window,minimum_accepted_window,"
           "candidate_rebuilds,horizon_limited_attempts,"
           "candidate_pairs,detected_collisions,accepted_collisions,"
           "contended_collisions,rejected_candidate_pairs,"
           "rejected_detected_collisions,"
           "rejected_accepted_collisions,"
           "rejected_contended_collisions,rejected_overlap_pairs,"
           "minimum_surface_gap,minimum_rejected_surface_gap,"
           "reference_collisions_at_time,collision_count_error,"
           "rms_position_error,maximum_position_error,"
           "rms_velocity_error,maximum_velocity_error,"
           "momentum_drift,relative_energy_drift,wall_seconds,"
           "reference_wall_seconds,cpu_rms_position_difference,"
           "cpu_rms_velocity_difference,placement_attempts\n";
    summary_output
        << std::setprecision(17)
        << "backend,seeds,completed,completion_rate,accepted_steps,"
           "attempts,rollbacks,rollback_rate,"
           "window_growths,maximum_refinement_depth,"
           "minimum_accepted_window,"
           "accepted_collisions,contended_collisions,"
           "rejected_overlap_pairs,mean_collision_count_error,"
           "mean_absolute_collision_count_error,"
           "mean_rms_position_error,maximum_rms_position_error,"
           "mean_rms_velocity_error,maximum_rms_velocity_error,"
           "minimum_surface_gap,minimum_rejected_surface_gap,"
           "mean_momentum_drift,"
           "maximum_absolute_relative_energy_drift,"
           "mean_wall_seconds\n";

#if PARTICLES_HAS_CUDA
    const bool run_cuda = cuda::device_count() != 0;
#else
    constexpr bool run_cuda = false;
#endif
    AdaptiveSummaryAccumulator cpu_summary;
    AdaptiveSummaryAccumulator cuda_summary;
    const double duration = config.window_convergence.duration;
    const double maximum_window =
        config.window_convergence.maximum_window;
    const double minimum_window =
        config.adaptive_window.minimum_window;
    const double time_tolerance =
        32.0 *
        std::numeric_limits<double>::epsilon() *
        std::max(1.0, duration);

    std::cout << std::setprecision(9)
              << "adaptive parallel-window rollback stress experiment\n"
              << "  config: " << config_path.string() << '\n'
              << "  particles: " << config.particle_count << '\n'
              << "  seeds: " << config.seed_count << '\n'
              << "  duration: " << duration << '\n'
              << "  maximum window: " << maximum_window << '\n'
              << "  minimum window: " << minimum_window << '\n'
              << "  growth after successes: "
              << config.adaptive_window.successful_steps_before_growth
              << '\n'
              << "  growth factor: "
              << config.adaptive_window.growth_factor << '\n'
              << "  CUDA backend: "
              << (run_cuda ? "enabled" : "no visible device") << '\n';

    for (std::size_t seed_index = 0;
         seed_index < config.seed_count;
         ++seed_index) {
        const std::uint64_t seed =
            config.base_random_seed +
            static_cast<std::uint64_t>(seed_index);
        const InitializationResult initialized =
            initialize_uniform_gas(UniformGasInitialization{
                .random_seed = seed,
                .particle_count = config.particle_count,
                .particle_diameter = config.particle_diameter,
                .particle_mass = config.particle_mass,
                .boltzmann_constant = config.boltzmann_constant,
                .temperature = config.temperature,
                .bulk_velocity = config.bulk_velocity,
                .domain = config.domain,
            });
        const auto [initial_momentum, initial_energy] =
            conserved_quantities(
                initialized.particles,
                config.particle_mass
            );

        std::vector<AdaptiveApproximateRun> runs;
        runs.reserve(run_cuda ? 2U : 1U);
        std::vector<Particle> cpu_state = initialized.particles;
        const auto cpu_start = std::chrono::steady_clock::now();
        const AdaptiveParallelWindowRunResult cpu_result =
            advance_host_adaptive_parallel_windows(
                cpu_state,
                grid,
                duration,
                maximum_window,
                minimum_window,
                config.particle_diameter,
                config.adaptive_window.maximum_attempts,
                1.0e-12,
                config.adaptive_window.successful_steps_before_growth,
                config.adaptive_window.growth_factor
            );
        const double cpu_wall_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - cpu_start
            ).count();
        runs.push_back(AdaptiveApproximateRun{
            .backend = "cpu",
            .result = cpu_result,
            .state = std::move(cpu_state),
            .wall_seconds = cpu_wall_seconds,
        });

#if PARTICLES_HAS_CUDA
        if (run_cuda) {
            cuda::ParticleGrid device{initialized.particles.size()};
            device.upload(initialized.particles);
            const auto cuda_start = std::chrono::steady_clock::now();
            const AdaptiveParallelWindowRunResult cuda_result =
                device.advance_adaptive_parallel_windows(
                    grid,
                    duration,
                    maximum_window,
                    minimum_window,
                    config.particle_diameter,
                    config.adaptive_window.maximum_attempts,
                    1.0e-12,
                    config.adaptive_window.successful_steps_before_growth,
                    config.adaptive_window.growth_factor
                );
            const double cuda_wall_seconds =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - cuda_start
                ).count();
            runs.push_back(AdaptiveApproximateRun{
                .backend = "cuda",
                .result = cuda_result,
                .state = device.download_particles(),
                .wall_seconds = cuda_wall_seconds,
            });
        }
#endif

        std::vector<double> reference_times{duration};
        for (const AdaptiveApproximateRun& run : runs) {
            reference_times.push_back(run.result.advanced_time);
        }
        std::sort(reference_times.begin(), reference_times.end());
        reference_times.erase(
            std::unique(
                reference_times.begin(),
                reference_times.end(),
                [&](double left, double right) {
                    return std::abs(left - right) <= time_tolerance;
                }
            ),
            reference_times.end()
        );

        EventDrivenSimulator reference{
            initialized.particles,
            config.particle_diameter,
            config.domain,
        };
        std::vector<ReferenceSnapshot> snapshots;
        snapshots.reserve(reference_times.size());
        const auto reference_start = std::chrono::steady_clock::now();
        for (const double time : reference_times) {
            reference.run_until(time);
            snapshots.push_back(ReferenceSnapshot{
                .time = time,
                .state = reference.particles(),
                .collision_count =
                    reference.statistics().particle_collisions,
            });
        }
        const double reference_wall_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - reference_start
            ).count();

        std::cout << "  seed " << seed << '\n';
        for (const AdaptiveApproximateRun& run : runs) {
            const ReferenceSnapshot& matched_reference =
                find_reference_snapshot(
                    snapshots,
                    run.result.advanced_time,
                    time_tolerance
                );
            const PeriodicStateError reference_error =
                measure_periodic_state_error(
                    run.state,
                    matched_reference.state,
                    grid
                );
            const auto [momentum, energy] = conserved_quantities(
                run.state,
                config.particle_mass
            );
            const double momentum_drift =
                norm(momentum - initial_momentum);
            const double relative_energy_drift =
                initial_energy == 0.0
                    ? 0.0
                    : (energy - initial_energy) / initial_energy;
            const std::int64_t collision_count_error =
                static_cast<std::int64_t>(
                    run.result.accepted_collision_count
                ) -
                static_cast<std::int64_t>(
                    matched_reference.collision_count
                );

            const double not_available =
                std::numeric_limits<double>::quiet_NaN();
            PeriodicStateError cpu_difference{
                .rms_position = run.backend == "cpu"
                    ? 0.0
                    : not_available,
                .maximum_position = run.backend == "cpu"
                    ? 0.0
                    : not_available,
                .rms_velocity = run.backend == "cpu"
                    ? 0.0
                    : not_available,
                .maximum_velocity = run.backend == "cpu"
                    ? 0.0
                    : not_available,
            };
            if (run.backend == "cuda") {
                cpu_difference = measure_periodic_state_error(
                    run.state,
                    runs.front().state,
                    grid
                );
            }

            run_output
                << seed << ','
                << run.backend << ','
                << run.result.requested_time << ','
                << run.result.advanced_time << ','
                << (run.result.completed() ? 1 : 0) << ','
                << adaptive_parallel_window_stop_reason_name(
                       run.result.stop_reason
                   ) << ','
                << run.result.maximum_window << ','
                << run.result.minimum_window << ','
                << run.result.growth_factor << ','
                << run.result.successful_steps_before_growth << ','
                << run.result.accepted_step_count << ','
                << run.result.attempt_count << ','
                << run.result.rollback_count << ','
                << run.result.maximum_refinement_depth << ','
                << run.result.window_growth_count << ','
                << run.result.minimum_attempted_window << ','
                << run.result.minimum_accepted_window << ','
                << run.result.candidate_rebuild_count << ','
                << run.result.horizon_limited_attempt_count << ','
                << run.result.candidate_pair_count << ','
                << run.result.detected_collision_count << ','
                << run.result.accepted_collision_count << ','
                << run.result.contended_collision_count << ','
                << run.result.rejected_candidate_pair_count << ','
                << run.result.rejected_detected_collision_count << ','
                << run.result.rejected_accepted_collision_count << ','
                << run.result.rejected_contended_collision_count << ','
                << run.result.rejected_overlap_pair_count << ','
                << run.result.minimum_surface_gap << ','
                << run.result.minimum_rejected_surface_gap << ','
                << matched_reference.collision_count << ','
                << collision_count_error << ','
                << reference_error.rms_position << ','
                << reference_error.maximum_position << ','
                << reference_error.rms_velocity << ','
                << reference_error.maximum_velocity << ','
                << momentum_drift << ','
                << relative_energy_drift << ','
                << run.wall_seconds << ','
                << reference_wall_seconds << ','
                << cpu_difference.rms_position << ','
                << cpu_difference.rms_velocity << ','
                << initialized.diagnostics.placement_attempts << '\n';

            AdaptiveSummaryAccumulator& summary =
                run.backend == "cpu" ? cpu_summary : cuda_summary;
            ++summary.run_count;
            summary.completed_count +=
                run.result.completed() ? 1U : 0U;
            summary.accepted_step_count +=
                run.result.accepted_step_count;
            summary.attempt_count += run.result.attempt_count;
            summary.rollback_count += run.result.rollback_count;
            summary.window_growth_count +=
                run.result.window_growth_count;
            summary.maximum_refinement_depth = std::max(
                summary.maximum_refinement_depth,
                run.result.maximum_refinement_depth
            );
            summary.accepted_collision_count +=
                run.result.accepted_collision_count;
            summary.contended_collision_count +=
                run.result.contended_collision_count;
            summary.rejected_overlap_pair_count +=
                run.result.rejected_overlap_pair_count;
            summary.collision_count_error_sum +=
                collision_count_error;
            summary.absolute_collision_count_error_sum +=
                static_cast<std::uint64_t>(
                    std::abs(collision_count_error)
                );
            summary.rms_position_error_sum +=
                reference_error.rms_position;
            summary.rms_velocity_error_sum +=
                reference_error.rms_velocity;
            summary.momentum_drift_sum += momentum_drift;
            summary.wall_seconds_sum += run.wall_seconds;
            summary.maximum_rms_position_error = std::max(
                summary.maximum_rms_position_error,
                reference_error.rms_position
            );
            summary.maximum_rms_velocity_error = std::max(
                summary.maximum_rms_velocity_error,
                reference_error.rms_velocity
            );
            summary.maximum_absolute_relative_energy_drift = std::max(
                summary.maximum_absolute_relative_energy_drift,
                std::abs(relative_energy_drift)
            );
            summary.minimum_accepted_window = std::min(
                summary.minimum_accepted_window,
                run.result.minimum_accepted_window
            );
            summary.minimum_surface_gap = std::min(
                summary.minimum_surface_gap,
                run.result.minimum_surface_gap
            );
            summary.minimum_rejected_surface_gap = std::min(
                summary.minimum_rejected_surface_gap,
                run.result.minimum_rejected_surface_gap
            );

            std::cout
                << "    " << run.backend << ": "
                << adaptive_parallel_window_stop_reason_name(
                       run.result.stop_reason
                   )
                << ", attempts=" << run.result.attempt_count
                << ", rollbacks=" << run.result.rollback_count
                << ", growths=" << run.result.window_growth_count
                << ", max depth="
                << run.result.maximum_refinement_depth
                << ", min accepted="
                << run.result.minimum_accepted_window
                << ", collisions="
                << run.result.accepted_collision_count << " / "
                << matched_reference.collision_count
                << ", rms x=" << reference_error.rms_position
                << ", CPU/GPU rms x="
                << cpu_difference.rms_position << '\n';
        }
    }

    const auto write_summary = [&](
        std::string_view backend,
        const AdaptiveSummaryAccumulator& value
    ) {
        if (value.run_count == 0) {
            return;
        }
        const long double count =
            static_cast<long double>(value.run_count);
        const double rollback_rate =
            value.attempt_count == 0
                ? 0.0
                : static_cast<double>(value.rollback_count) /
                  static_cast<double>(value.attempt_count);
        summary_output
            << backend << ','
            << value.run_count << ','
            << value.completed_count << ','
            << static_cast<double>(
                   static_cast<long double>(value.completed_count) /
                   count
               ) << ','
            << value.accepted_step_count << ','
            << value.attempt_count << ','
            << value.rollback_count << ','
            << rollback_rate << ','
            << value.window_growth_count << ','
            << value.maximum_refinement_depth << ','
            << value.minimum_accepted_window << ','
            << value.accepted_collision_count << ','
            << value.contended_collision_count << ','
            << value.rejected_overlap_pair_count << ','
            << static_cast<double>(
                   static_cast<long double>(
                       value.collision_count_error_sum
                   ) / count
               ) << ','
            << static_cast<double>(
                   static_cast<long double>(
                       value.absolute_collision_count_error_sum
                   ) / count
               ) << ','
            << static_cast<double>(
                   value.rms_position_error_sum / count
               ) << ','
            << value.maximum_rms_position_error << ','
            << static_cast<double>(
                   value.rms_velocity_error_sum / count
               ) << ','
            << value.maximum_rms_velocity_error << ','
            << value.minimum_surface_gap << ','
            << value.minimum_rejected_surface_gap << ','
            << static_cast<double>(
                   value.momentum_drift_sum / count
               ) << ','
            << value.maximum_absolute_relative_energy_drift << ','
            << static_cast<double>(
                   value.wall_seconds_sum / count
               ) << '\n';

        std::cout
            << "  summary " << backend << ": completed="
            << value.completed_count << " / " << value.run_count
            << ", attempts=" << value.attempt_count
            << ", rollbacks=" << value.rollback_count
            << " (" << rollback_rate << ')'
            << ", growths=" << value.window_growth_count
            << ", max depth=" << value.maximum_refinement_depth
            << ", mean |collision error|="
            << static_cast<double>(
                   static_cast<long double>(
                       value.absolute_collision_count_error_sum
                   ) / count
               )
            << ", mean rms x="
            << static_cast<double>(
                   value.rms_position_error_sum / count
               ) << '\n';
    };
    write_summary("cpu", cpu_summary);
    write_summary("cuda", cuda_summary);

    if (!run_output || !summary_output) {
        throw std::runtime_error(
            "Failed while writing adaptive parallel-window reports."
        );
    }
    std::cout << "  run CSV: " << run_path.string() << '\n'
              << "  summary CSV: " << summary_path.string() << '\n';
    return 0;
}

}  // namespace particles::app
