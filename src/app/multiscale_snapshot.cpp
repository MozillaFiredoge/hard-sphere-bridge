#include "multiscale_snapshot.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "particles/config.hpp"
#include "particles/diagnostics.hpp"
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

[[nodiscard]] double seconds_since(
    std::chrono::steady_clock::time_point start
) {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start
    ).count();
}

}  // namespace

int run_multiscale_snapshot(
    const std::filesystem::path& config_path,
    const std::filesystem::path& output_directory
) {
    const ShockTubeConfig config =
        load_shock_tube_config(config_path);

    const auto initialization_start =
        std::chrono::steady_clock::now();
    const InitializationResult initialized =
        initialize_shock_tube(config);
    const double initialization_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() -
            initialization_start
        ).count();

    const auto diagnostic_start =
        std::chrono::steady_clock::now();
    const MultiscaleDiagnosticFrame frame =
        measure_multiscale(
            initialized.particles,
            0.0,
            config.particle_mass,
            config.boltzmann_constant,
            config.domain,
            config.diagnostics
        );
    const double diagnostic_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() -
            diagnostic_start
        ).count();

    MultiscaleCsvWriter writer{output_directory};
    const auto output_start = std::chrono::steady_clock::now();
    writer.write(frame);
    const double output_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() -
            output_start
        ).count();

    std::size_t left_particle_count = 0;
    std::size_t right_particle_count = 0;
    double left_volume = 0.0;
    double right_volume = 0.0;
    double left_temperature_sum = 0.0;
    double right_temperature_sum = 0.0;
    for (const MacroscopicSlabState& slab :
         frame.macroscopic_slabs) {
        if (slab.x_center < config.interface_x) {
            left_particle_count += slab.particle_count;
            left_volume += slab.volume;
            left_temperature_sum +=
                static_cast<double>(slab.particle_count) *
                slab.temperature;
        } else {
            right_particle_count += slab.particle_count;
            right_volume += slab.volume;
            right_temperature_sum +=
                static_cast<double>(slab.particle_count) *
                slab.temperature;
        }
    }
    const double left_number_density =
        static_cast<double>(left_particle_count) / left_volume;
    const double right_number_density =
        static_cast<double>(right_particle_count) / right_volume;
    const double left_temperature =
        left_temperature_sum /
        static_cast<double>(left_particle_count);
    const double right_temperature =
        right_temperature_sum /
        static_cast<double>(right_particle_count);

    std::cout << std::setprecision(9)
              << "multiscale shock-tube snapshot\n"
              << "  config: " << config_path.string() << '\n'
              << "  particles: " << frame.particle_count << '\n'
              << "  initialization seconds: "
              << initialization_seconds << '\n'
              << "  diagnostics seconds: "
              << diagnostic_seconds << '\n'
              << "  CSV output seconds: "
              << output_seconds << '\n'
              << "  microscopic samples: "
              << frame.microscopic_particles.size() << '\n'
              << "  phase-space samples: "
              << frame.phase_space.samples() << '\n'
              << "  phase-space velocity underflow/overflow: "
              << frame.phase_space.velocity_underflow << " / "
              << frame.phase_space.velocity_overflow << '\n'
              << "  macroscopic slabs: "
              << frame.macroscopic_slabs.size() << '\n'
              << "  left/right mean number density ratio: "
              << left_number_density / right_number_density << '\n'
              << "  left/right particle-weighted temperature: "
              << left_temperature << " / "
              << right_temperature << '\n'
              << "  reconstructed particles: "
              << frame.reconstructed_particle_count << '\n'
              << "  momentum reconstruction error: "
              << frame.momentum_reconstruction_error
              << " (per particle "
              << frame.momentum_reconstruction_error /
                     static_cast<double>(frame.particle_count)
              << ")\n"
              << "  kinetic-energy reconstruction error: "
              << frame.kinetic_energy_reconstruction_error
              << " (relative "
              << frame.kinetic_energy_reconstruction_error /
                     frame.kinetic_energy
              << ")\n"
              << "  summary CSV: "
              << writer.summary_path().string() << '\n'
              << "  microscopic CSV: "
              << writer.microscopic_path().string() << '\n'
              << "  phase-space CSV: "
              << writer.phase_space_path().string() << '\n'
              << "  macroscopic CSV: "
              << writer.macroscopic_path().string() << '\n';

    return
        frame.phase_space.samples() == frame.particle_count &&
        frame.reconstructed_particle_count == frame.particle_count
            ? 0
            : 1;
}

int run_shock_tube(
    const std::filesystem::path& config_path,
    const std::filesystem::path& output_directory,
    double end_time_override
) {
    ShockTubeConfig config =
        load_shock_tube_config(config_path);
    if (end_time_override >= 0.0) {
        if (
            !std::isfinite(end_time_override) ||
            end_time_override <= 0.0
        ) {
            throw std::invalid_argument(
                "Shock-tube end-time override must be finite and positive."
            );
        }
        config.evolution.end_time = end_time_override;
    }
    const auto initialization_start =
        std::chrono::steady_clock::now();
    InitializationResult initialized =
        initialize_shock_tube(config);
    const double initialization_seconds =
        seconds_since(initialization_start);

    const UniformGrid grid = make_uniform_grid(
        config.domain,
        config.evolution.cell_width_factor *
            config.particle_diameter
    );
    MultiscaleCsvWriter writer{output_directory};
    std::ofstream evolution{
        output_directory / "evolution.csv"
    };
    if (!evolution) {
        throw std::runtime_error(
            "Cannot open shock-tube evolution CSV output."
        );
    }
    evolution
        << std::setprecision(17)
        << "frame,backend,target_time,time,status,"
           "simulation_seconds,diagnostic_seconds,output_seconds,"
           "accepted_steps,attempts,rollbacks,window_growths,"
           "maximum_refinement_depth,minimum_accepted_window,"
           "candidate_pairs,accepted_collisions,contended_collisions,"
           "rejected_overlap_pairs,minimum_rejected_surface_gap,"
           "exact_recoveries,exact_recovery_batches,"
           "exact_recovery_candidate_rebuilds,"
           "exact_recovery_collisions,"
           "exact_recovery_serialized_conflicts,"
           "exact_recovery_reflected_particles,"
           "exact_recovery_reflected_axes,minimum_surface_gap,"
           "momentum_x,momentum_y,momentum_z,kinetic_energy,"
           "relative_energy_drift,"
           "transverse_momentum_drift_per_particle\n";

    const MultiscaleDiagnosticFrame initial_frame =
        measure_multiscale(
            initialized.particles,
            0.0,
            config.particle_mass,
            config.boltzmann_constant,
            config.domain,
            config.diagnostics
        );
    writer.write(initial_frame);
    const Vec3 initial_momentum = initial_frame.momentum;
    const double initial_energy = initial_frame.kinetic_energy;

    std::size_t frame_index = 0;
    bool diagnostics_pass = true;
    const auto record_frame = [&](
        std::string_view backend,
        double target_time,
        double time,
        std::string_view status,
        double simulation_seconds,
        const AdaptiveParallelWindowRunResult* run,
        const std::vector<Particle>& state
    ) {
        const auto diagnostic_start =
            std::chrono::steady_clock::now();
        const MultiscaleDiagnosticFrame frame =
            measure_multiscale(
                state,
                time,
                config.particle_mass,
                config.boltzmann_constant,
                config.domain,
                config.diagnostics
            );
        const double diagnostic_seconds =
            seconds_since(diagnostic_start);
        const auto output_start =
            std::chrono::steady_clock::now();
        writer.write(frame);
        const double output_seconds =
            seconds_since(output_start);

        const double relative_energy_drift =
            initial_energy == 0.0
                ? 0.0
                : (
                      frame.kinetic_energy -
                      initial_energy
                  ) /
                      initial_energy;
        const double transverse_momentum_drift =
            std::hypot(
                frame.momentum.y - initial_momentum.y,
                frame.momentum.z - initial_momentum.z
            ) /
            static_cast<double>(frame.particle_count);
        const bool frame_pass =
            frame.phase_space.samples() == frame.particle_count &&
            frame.reconstructed_particle_count ==
                frame.particle_count &&
            frame.momentum_reconstruction_error /
                    static_cast<double>(frame.particle_count) <=
                1.0e-10 &&
            std::abs(
                frame.kinetic_energy_reconstruction_error /
                frame.kinetic_energy
            ) <= 1.0e-10 &&
            std::abs(relative_energy_drift) <= 1.0e-10 &&
            transverse_momentum_drift <= 1.0e-10;
        diagnostics_pass = diagnostics_pass && frame_pass;

        const AdaptiveParallelWindowRunResult empty{};
        const AdaptiveParallelWindowRunResult& result =
            run == nullptr ? empty : *run;
        evolution
            << frame_index << ','
            << backend << ','
            << target_time << ','
            << time << ','
            << status << ','
            << simulation_seconds << ','
            << diagnostic_seconds << ','
            << output_seconds << ','
            << result.accepted_step_count << ','
            << result.attempt_count << ','
            << result.rollback_count << ','
            << result.window_growth_count << ','
            << result.maximum_refinement_depth << ','
            << result.minimum_accepted_window << ','
            << result.candidate_pair_count << ','
            << result.accepted_collision_count << ','
            << result.contended_collision_count << ','
            << result.rejected_overlap_pair_count << ','
            << result.minimum_rejected_surface_gap << ','
            << result.exact_recovery_count << ','
            << result.exact_recovery_batch_count << ','
            << result.exact_recovery_candidate_rebuild_count << ','
            << result.exact_recovery_collision_count << ','
            << result.
                   exact_recovery_serialized_conflict_resolution_count
            << ','
            << result.exact_recovery_reflected_particle_count << ','
            << result.exact_recovery_reflected_axis_count << ','
            << result.minimum_surface_gap << ','
            << frame.momentum.x << ','
            << frame.momentum.y << ','
            << frame.momentum.z << ','
            << frame.kinetic_energy << ','
            << relative_energy_drift << ','
            << transverse_momentum_drift << '\n';
        evolution.flush();
        if (!evolution) {
            throw std::runtime_error(
                "Failed while writing shock-tube evolution CSV."
            );
        }

        std::cout
            << "  frame " << frame_index
            << ": target=" << target_time
            << ", time=" << time
            << ", status=" << status
            << ", attempts=" << result.attempt_count
            << ", rollbacks=" << result.rollback_count
            << ", collisions=" << result.accepted_collision_count
            << ", exact recoveries="
            << result.exact_recovery_count
            << ", min gap=" << result.minimum_surface_gap
            << ", rejected min gap="
            << result.minimum_rejected_surface_gap
            << ", relative energy drift="
            << relative_energy_drift
            << ", wall=" << simulation_seconds << " s\n";
        ++frame_index;
    };

#if PARTICLES_HAS_CUDA
    const bool use_cuda = cuda::device_count() != 0;
#else
    constexpr bool use_cuda = false;
#endif
    const std::string_view backend =
        use_cuda ? "cuda" : "cpu";

    std::cout
        << std::setprecision(9)
        << "shock-tube evolution\n"
        << "  config: " << config_path.string() << '\n'
        << "  backend: " << backend << '\n'
        << "  particles: " << initialized.particles.size() << '\n'
        << "  initialization seconds: "
        << initialization_seconds << '\n'
        << "  grid: "
        << grid.dimensions[0] << " x "
        << grid.dimensions[1] << " x "
        << grid.dimensions[2]
        << " (" << grid.cell_count << " cells)\n"
        << "  end time: " << config.evolution.end_time << '\n'
        << "  frame interval: "
        << config.evolution.frame_interval << '\n'
        << "  maximum/minimum window: "
        << config.evolution.maximum_window << " / "
        << config.evolution.adaptive_window.minimum_window
        << "\n  exact-recovery batch limit: "
        << config.evolution.exact_recovery_batch_limit
        << '\n';

    evolution
        << frame_index << ','
        << backend << ",0,0,initial,0,0,0,0,0,0,0,0,"
        << std::numeric_limits<double>::infinity()
        << ",0,0,0,0,"
        << std::numeric_limits<double>::infinity()
        << ",0,0,0,0,0,0,0,"
        << std::numeric_limits<double>::infinity() << ','
        << initial_frame.momentum.x << ','
        << initial_frame.momentum.y << ','
        << initial_frame.momentum.z << ','
        << initial_frame.kinetic_energy
        << ",0,0\n";
    evolution.flush();
    std::cout << "  frame 0: target=0, time=0, status=initial\n";
    ++frame_index;

    double current_time = 0.0;
    const double time_tolerance =
        16.0 *
        std::numeric_limits<double>::epsilon() *
        std::max(1.0, config.evolution.end_time);

#if PARTICLES_HAS_CUDA
    if (use_cuda) {
        cuda::ParticleGrid device{
            initialized.particles.size()
        };
        device.upload(initialized.particles);
        while (
            current_time <
            config.evolution.end_time - time_tolerance
        ) {
            const double target_time = std::min(
                config.evolution.end_time,
                static_cast<double>(frame_index) *
                    config.evolution.frame_interval
            );
            const auto simulation_start =
                std::chrono::steady_clock::now();
            const AdaptiveParallelWindowRunResult result =
                device.advance_adaptive_parallel_windows(
                    grid,
                    target_time - current_time,
                    config.evolution.maximum_window,
                    config.evolution.adaptive_window.minimum_window,
                    config.particle_diameter,
                    config.evolution.adaptive_window.maximum_attempts,
                    1.0e-12,
                    config.evolution.adaptive_window.
                        successful_steps_before_growth,
                    config.evolution.adaptive_window.growth_factor,
                    config.evolution.exact_recovery_batch_limit
                );
            const double simulation_seconds =
                seconds_since(simulation_start);
            current_time += result.advanced_time;
            const std::vector<Particle> state =
                device.download_particles();
            record_frame(
                backend,
                target_time,
                current_time,
                adaptive_parallel_window_stop_reason_name(
                    result.stop_reason
                ),
                simulation_seconds,
                &result,
                state
            );
            if (!result.completed()) {
                return 1;
            }
            current_time = target_time;
        }
    } else
#endif
    {
        std::vector<Particle> state =
            std::move(initialized.particles);
        while (
            current_time <
            config.evolution.end_time - time_tolerance
        ) {
            const double target_time = std::min(
                config.evolution.end_time,
                static_cast<double>(frame_index) *
                    config.evolution.frame_interval
            );
            const auto simulation_start =
                std::chrono::steady_clock::now();
            const AdaptiveParallelWindowRunResult result =
                advance_host_adaptive_parallel_windows(
                    state,
                    grid,
                    target_time - current_time,
                    config.evolution.maximum_window,
                    config.evolution.adaptive_window.minimum_window,
                    config.particle_diameter,
                    config.evolution.adaptive_window.maximum_attempts,
                    1.0e-12,
                    config.evolution.adaptive_window.
                        successful_steps_before_growth,
                    config.evolution.adaptive_window.growth_factor,
                    config.evolution.exact_recovery_batch_limit
                );
            const double simulation_seconds =
                seconds_since(simulation_start);
            current_time += result.advanced_time;
            record_frame(
                backend,
                target_time,
                current_time,
                adaptive_parallel_window_stop_reason_name(
                    result.stop_reason
                ),
                simulation_seconds,
                &result,
                state
            );
            if (!result.completed()) {
                return 1;
            }
            current_time = target_time;
        }
    }

    std::cout
        << "  evolution CSV: "
        << (output_directory / "evolution.csv").string() << '\n'
        << "  multiscale summary CSV: "
        << writer.summary_path().string() << '\n';
    return diagnostics_pass ? 0 : 1;
}

}  // namespace particles::app
