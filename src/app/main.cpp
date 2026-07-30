#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "particles/config.hpp"
#include "particles/diagnostics.hpp"
#include "particles/event_driven_simulator.hpp"
#include "particles/hard_sphere.hpp"
#include "particles/initialization.hpp"
#include "particles/parallel_window.hpp"
#include "particles/uniform_grid.hpp"

#if PARTICLES_HAS_CUDA
#include "particles/cuda_particle_grid.hpp"
#include "particles/cuda_support.hpp"
#endif

namespace {

int run_window_convergence(
    const std::filesystem::path& config_path,
    const std::filesystem::path& output_path
) {
    const particles::EquilibriumConfig config =
        particles::load_equilibrium_config(config_path);
    for (const particles::BoundaryKind boundary : config.domain.boundary) {
        if (boundary != particles::BoundaryKind::periodic) {
            throw std::invalid_argument(
                "The first window-convergence experiment requires periodic "
                "boundaries on all three axes."
            );
        }
    }

    const particles::InitializationResult initialized =
        particles::initialize_equilibrium_gas(config);
    const particles::UniformGrid grid = particles::make_uniform_grid(
        config.domain,
        2.0 * config.particle_diameter
    );

    const auto conserved_quantities = [&config](
        const std::vector<particles::Particle>& state
    ) {
        particles::Vec3 momentum{};
        double energy = 0.0;
        for (const particles::Particle& particle : state) {
            momentum += particles::momentum(
                particle,
                config.particle_mass
            );
            energy += particles::kinetic_energy(
                particle,
                config.particle_mass
            );
        }
        return std::pair{momentum, energy};
    };
    const auto [initial_momentum, initial_energy] =
        conserved_quantities(initialized.particles);

    particles::EventDrivenSimulator reference{
        initialized.particles,
        config.particle_diameter,
        config.domain,
    };
    const auto reference_start = std::chrono::steady_clock::now();
    reference.run_until(config.window_convergence.duration);
    const double reference_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - reference_start
        ).count();
    const std::size_t reference_collisions =
        reference.statistics().particle_collisions;

    const std::filesystem::path parent = output_path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream output{output_path};
    if (!output) {
        throw std::runtime_error(
            "Cannot open convergence report: " + output_path.string()
        );
    }
    output << std::setprecision(17)
           << "backend,level,maximum_window,requested_time,advanced_time,"
              "completed,stop_reason,steps,horizon_limited_steps,"
              "candidate_pairs,detected_collisions,accepted_collisions,"
              "contended_collisions,reference_collisions,"
              "collision_count_error,overlap_pairs,minimum_surface_gap,"
              "rms_position_error,maximum_position_error,"
              "rms_velocity_error,maximum_velocity_error,"
              "momentum_drift,relative_energy_drift,wall_seconds,"
              "reference_wall_seconds,cpu_rms_position_difference,"
              "cpu_rms_velocity_difference\n";

    std::size_t completed_levels = 0;
#if PARTICLES_HAS_CUDA
    const bool run_cuda = particles::cuda::device_count() != 0;
    std::size_t cuda_completed_levels = 0;
#endif
    std::cout << std::setprecision(9)
              << "parallel-window convergence experiment\n"
              << "  config: " << config_path.string() << '\n'
              << "  particles: " << initialized.particles.size() << '\n'
              << "  duration: "
              << config.window_convergence.duration << '\n'
              << "  exact reference collisions: "
              << reference_collisions << '\n'
              << "  exact reference wall time: "
              << reference_seconds << " s\n";

    for (std::size_t level = 0;
         level < config.window_convergence.refinement_levels;
         ++level) {
        const double maximum_window = std::ldexp(
            config.window_convergence.maximum_window,
            -static_cast<int>(level)
        );
        std::vector<particles::Particle> approximate =
            initialized.particles;
        const auto start = std::chrono::steady_clock::now();
        const particles::ParallelWindowRunResult run =
            particles::advance_host_parallel_windows(
                approximate,
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

        const double not_available =
            std::numeric_limits<double>::quiet_NaN();
        particles::PeriodicStateError error{
            .rms_position = not_available,
            .maximum_position = not_available,
            .rms_velocity = not_available,
            .maximum_velocity = not_available,
        };
        if (run.completed()) {
            error = particles::measure_periodic_state_error(
                approximate,
                reference.particles(),
                grid
            );
            ++completed_levels;
        }

        const auto [momentum, energy] =
            conserved_quantities(approximate);
        const double momentum_drift =
            particles::norm(momentum - initial_momentum);
        const double relative_energy_drift =
            initial_energy == 0.0
                ? 0.0
                : (energy - initial_energy) / initial_energy;
        const std::int64_t collision_count_error =
            static_cast<std::int64_t>(run.accepted_collision_count) -
            static_cast<std::int64_t>(reference_collisions);

        output << "cpu,"
               << level << ','
               << maximum_window << ','
               << run.requested_time << ','
               << run.advanced_time << ','
               << (run.completed() ? 1 : 0) << ','
               << particles::parallel_window_stop_reason_name(
                      run.stop_reason
                  ) << ','
               << run.step_count << ','
               << run.horizon_limited_step_count << ','
               << run.candidate_pair_count << ','
               << run.detected_collision_count << ','
               << run.accepted_collision_count << ','
               << run.contended_collision_count << ','
               << reference_collisions << ','
               << collision_count_error << ','
               << run.overlap_pair_count << ','
               << run.minimum_surface_gap << ','
               << error.rms_position << ','
               << error.maximum_position << ','
               << error.rms_velocity << ','
               << error.maximum_velocity << ','
               << momentum_drift << ','
               << relative_energy_drift << ','
               << wall_seconds << ','
               << reference_seconds << ','
               << 0.0 << ','
               << 0.0 << '\n';

        std::cout << "  CPU level " << level
                  << ": max dt=" << maximum_window
                  << ", status="
                  << particles::parallel_window_stop_reason_name(
                         run.stop_reason
                     )
                  << ", steps=" << run.step_count
                  << ", accepted=" << run.accepted_collision_count
                  << ", contended=" << run.contended_collision_count
                  << ", collision error=" << collision_count_error
                  << ", rms x=" << error.rms_position
                  << ", rms v=" << error.rms_velocity
                  << ", overlaps=" << run.overlap_pair_count
                  << ", wall=" << wall_seconds << " s\n";

#if PARTICLES_HAS_CUDA
        if (run_cuda) {
            particles::cuda::ParticleGrid device{
                initialized.particles.size()
            };
            device.upload(initialized.particles);
            const auto cuda_start = std::chrono::steady_clock::now();
            const particles::ParallelWindowRunResult cuda_run =
                device.advance_parallel_windows(
                    grid,
                    config.window_convergence.duration,
                    maximum_window,
                    config.particle_diameter,
                    config.window_convergence.maximum_steps
                );
            const double cuda_wall_seconds =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - cuda_start
                ).count();
            const std::vector<particles::Particle> cuda_state =
                device.download_particles();

            particles::PeriodicStateError cuda_reference_error{
                .rms_position = not_available,
                .maximum_position = not_available,
                .rms_velocity = not_available,
                .maximum_velocity = not_available,
            };
            particles::PeriodicStateError cpu_difference{
                .rms_position = not_available,
                .maximum_position = not_available,
                .rms_velocity = not_available,
                .maximum_velocity = not_available,
            };
            if (cuda_run.completed()) {
                cuda_reference_error =
                    particles::measure_periodic_state_error(
                        cuda_state,
                        reference.particles(),
                        grid
                    );
                ++cuda_completed_levels;
            }
            if (
                std::abs(
                    cuda_run.advanced_time - run.advanced_time
                ) <= 1.0e-11
            ) {
                cpu_difference =
                    particles::measure_periodic_state_error(
                        cuda_state,
                        approximate,
                        grid
                    );
            }

            const auto [cuda_momentum, cuda_energy] =
                conserved_quantities(cuda_state);
            const double cuda_momentum_drift =
                particles::norm(
                    cuda_momentum - initial_momentum
                );
            const double cuda_relative_energy_drift =
                initial_energy == 0.0
                    ? 0.0
                    : (
                          cuda_energy - initial_energy
                      ) /
                      initial_energy;
            const std::int64_t cuda_collision_count_error =
                static_cast<std::int64_t>(
                    cuda_run.accepted_collision_count
                ) -
                static_cast<std::int64_t>(reference_collisions);

            output << "cuda,"
                   << level << ','
                   << maximum_window << ','
                   << cuda_run.requested_time << ','
                   << cuda_run.advanced_time << ','
                   << (cuda_run.completed() ? 1 : 0) << ','
                   << particles::parallel_window_stop_reason_name(
                          cuda_run.stop_reason
                      ) << ','
                   << cuda_run.step_count << ','
                   << cuda_run.horizon_limited_step_count << ','
                   << cuda_run.candidate_pair_count << ','
                   << cuda_run.detected_collision_count << ','
                   << cuda_run.accepted_collision_count << ','
                   << cuda_run.contended_collision_count << ','
                   << reference_collisions << ','
                   << cuda_collision_count_error << ','
                   << cuda_run.overlap_pair_count << ','
                   << cuda_run.minimum_surface_gap << ','
                   << cuda_reference_error.rms_position << ','
                   << cuda_reference_error.maximum_position << ','
                   << cuda_reference_error.rms_velocity << ','
                   << cuda_reference_error.maximum_velocity << ','
                   << cuda_momentum_drift << ','
                   << cuda_relative_energy_drift << ','
                   << cuda_wall_seconds << ','
                   << reference_seconds << ','
                   << cpu_difference.rms_position << ','
                   << cpu_difference.rms_velocity << '\n';

            std::cout << "  CUDA level " << level
                      << ": max dt=" << maximum_window
                      << ", status="
                      << particles::parallel_window_stop_reason_name(
                             cuda_run.stop_reason
                         )
                      << ", steps=" << cuda_run.step_count
                      << ", accepted="
                      << cuda_run.accepted_collision_count
                      << ", contended="
                      << cuda_run.contended_collision_count
                      << ", collision error="
                      << cuda_collision_count_error
                      << ", rms x="
                      << cuda_reference_error.rms_position
                      << ", CPU/GPU rms x="
                      << cpu_difference.rms_position
                      << ", overlaps="
                      << cuda_run.overlap_pair_count
                      << ", wall=" << cuda_wall_seconds << " s\n";
        }
#endif
    }

    if (!output) {
        throw std::runtime_error(
            "Failed while writing convergence report: " +
            output_path.string()
        );
    }

    std::cout << "  completed CPU levels: " << completed_levels << " / "
              << config.window_convergence.refinement_levels << '\n'
#if PARTICLES_HAS_CUDA
              << "  CUDA backend: "
              << (run_cuda ? "enabled" : "no visible device")
              << '\n';
    if (run_cuda) {
        std::cout << "  completed CUDA levels: "
                  << cuda_completed_levels << " / "
                  << config.window_convergence.refinement_levels
                  << '\n';
    }
    std::cout
#endif
              << "  report CSV: " << output_path.string() << '\n'
              << "  algorithm: controlled approximation "
                 "(mutual-earliest events per window)\n";
    return 0;
}

int run_equilibrium(
    const std::filesystem::path& config_path,
    const std::filesystem::path& output_directory
) {
    const particles::EquilibriumConfig config =
        particles::load_equilibrium_config(config_path);
    particles::InitializationResult initialized =
        particles::initialize_equilibrium_gas(config);

    particles::EventDrivenSimulator simulator{
        std::move(initialized.particles),
        config.particle_diameter,
        config.domain,
    };
    particles::EquilibriumCsvWriter writer{output_directory};

    std::size_t previous_collision_count = 0;
    double previous_sample_time = 0.0;
    std::size_t frame_count = 0;

    const auto measure_and_write = [&]() {
        const std::size_t cumulative_collisions =
            simulator.statistics().particle_collisions;
        const std::size_t interval_collisions =
            cumulative_collisions - previous_collision_count;
        const double interval_duration =
            simulator.time() - previous_sample_time;

        particles::EquilibriumDiagnosticFrame frame =
            particles::measure_equilibrium(
                simulator.particles(),
                simulator.time(),
                config.particle_mass,
                config.boltzmann_constant,
                config.diagnostics,
                cumulative_collisions,
                interval_collisions,
                interval_duration
            );
        writer.write(frame);

        previous_collision_count = cumulative_collisions;
        previous_sample_time = simulator.time();
        ++frame_count;
        return frame;
    };

    const particles::EquilibriumDiagnosticFrame initial_frame =
        measure_and_write();

    particles::EquilibriumDiagnosticFrame final_frame = initial_frame;
    const double time_tolerance =
        16.0 *
        std::numeric_limits<double>::epsilon() *
        std::max(1.0, config.end_time);
    std::size_t sample_index = 1;

    while (simulator.time() < config.end_time - time_tolerance) {
        const long double ideal_sample_time =
            static_cast<long double>(sample_index) *
            static_cast<long double>(config.diagnostics.sample_interval);
        const double target_time = std::min(
            config.end_time,
            static_cast<double>(ideal_sample_time)
        );

        if (target_time <= simulator.time() + time_tolerance) {
            ++sample_index;
            continue;
        }

        simulator.run_until(target_time);
        final_frame = measure_and_write();
        ++sample_index;
    }

    const double relative_energy_drift =
        initial_frame.kinetic_energy == 0.0
            ? 0.0
            : (
                final_frame.kinetic_energy -
                initial_frame.kinetic_energy
            ) /
            initial_frame.kinetic_energy;

    std::cout << std::fixed << std::setprecision(9)
              << "equilibrium reference run\n"
              << "  config: " << config_path.string() << '\n'
              << "  particles: " << simulator.particles().size() << '\n'
              << "  placement attempts: "
              << initialized.diagnostics.placement_attempts << '\n'
              << "  minimum checked separation: "
              << initialized.diagnostics.minimum_separation << '\n'
              << "  sampled temperature: "
              << initialized.diagnostics.realized_thermal_temperature << '\n'
              << "  simulated time: " << simulator.time() << '\n'
              << "  diagnostic frames: " << frame_count << '\n'
              << "  particle collisions: "
              << simulator.statistics().particle_collisions << '\n'
              << "  periodic wraps: "
              << simulator.statistics().periodic_wraps << '\n'
              << "  relative energy drift: "
              << relative_energy_drift << '\n'
              << "  momentum drift norm: "
              << particles::norm(
                     final_frame.momentum - initial_frame.momentum
                 ) << '\n'
              << "  summary CSV: " << writer.summary_path().string() << '\n'
              << "  histogram CSV: "
              << writer.histogram_path().string() << '\n';

    return 0;
}

int run_smoke_test() {
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

    std::vector<Particle> chain{
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
    particles::DomainConfig reference_domain{
        .extent = {10.0, 10.0, 10.0},
        .boundary = {
            particles::BoundaryKind::reflecting,
            particles::BoundaryKind::reflecting,
            particles::BoundaryKind::reflecting,
        },
    };
    particles::EventDrivenSimulator reference{
        std::move(chain),
        1.0,
        reference_domain,
    };
    reference.run_until(2.0);

    std::cout << "  reference chain collisions: "
              << reference.statistics().particle_collisions << '\n'
              << "  final chain velocities: "
              << reference.particles()[0].velocity.x << ", "
              << reference.particles()[1].velocity.x << ", "
              << reference.particles()[2].velocity.x << '\n';

    return 0;
}

#if PARTICLES_HAS_CUDA
int run_cuda_grid_check(const std::filesystem::path& config_path) {
    if (particles::cuda::device_count() == 0) {
        std::cerr << "No CUDA device is available for the grid check.\n";
        return 2;
    }

    const particles::EquilibriumConfig config =
        particles::load_equilibrium_config(config_path);
    particles::InitializationResult initialized =
        particles::initialize_equilibrium_gas(config);
    const particles::UniformGrid grid = particles::make_uniform_grid(
        config.domain,
        2.0 * config.particle_diameter
    );
    const particles::HostCellList host =
        particles::build_host_cell_list(initialized.particles, grid);
    const std::vector<particles::CandidatePair> host_candidates =
        particles::build_host_candidate_pairs(initialized.particles, grid);
    const double conservative_horizon =
        particles::conservative_candidate_horizon(
            initialized.particles,
            grid,
            config.particle_diameter
        );
    const double event_horizon = std::min(
        config.diagnostics.sample_interval,
        conservative_horizon
    );
    const std::vector<particles::CollisionEvent> host_events =
        particles::evaluate_host_collision_events(
            initialized.particles,
            host_candidates,
            grid,
            config.particle_diameter,
            event_horizon
        );
    const particles::CollisionBatchAnalysis host_batch =
        particles::analyze_earliest_collision_batch(
            host_events,
            initialized.particles.size()
        );

    particles::cuda::ParticleGrid device{initialized.particles.size()};
    device.upload(initialized.particles);
    device.build_cell_list(grid);
    const particles::cuda::CellListSnapshot cuda =
        device.download_cell_list();
    device.build_candidate_pairs();
    particles::cuda::CandidatePairSnapshot cuda_candidates =
        device.download_candidate_pairs();
    std::sort(
        cuda_candidates.pairs.begin(),
        cuda_candidates.pairs.end()
    );
    device.evaluate_collision_events(
        config.particle_diameter,
        event_horizon
    );
    device.analyze_earliest_collision_batch();
    particles::cuda::CollisionEventSnapshot cuda_events =
        device.download_collision_events();
    const particles::cuda::CollisionBatchSnapshot cuda_batch =
        device.download_collision_batch();
    const auto event_order = [](
        const particles::CollisionEvent& left,
        const particles::CollisionEvent& right
    ) {
        if (left.first != right.first) {
            return left.first < right.first;
        }
        if (left.second != right.second) {
            return left.second < right.second;
        }
        return left.time < right.time;
    };
    const auto gather_events = [](
        const std::vector<particles::CollisionEvent>& events,
        const std::vector<std::uint32_t>& indices
    ) {
        std::vector<particles::CollisionEvent> selected;
        selected.reserve(indices.size());
        for (const std::uint32_t index : indices) {
            if (index >= events.size()) {
                throw std::runtime_error(
                    "Collision-batch index exceeds its event array."
                );
            }
            selected.push_back(events[index]);
        }
        return selected;
    };
    std::vector<particles::CollisionEvent> host_independent_events =
        gather_events(
            host_events,
            host_batch.independent_event_indices
        );
    std::vector<particles::CollisionEvent> host_conflicting_events =
        gather_events(
            host_events,
            host_batch.conflicting_event_indices
        );
    std::vector<particles::CollisionEvent> cuda_independent_events =
        gather_events(
            cuda_events.events,
            cuda_batch.independent_event_indices
        );
    std::vector<particles::CollisionEvent> cuda_conflicting_events =
        gather_events(
            cuda_events.events,
            cuda_batch.conflicting_event_indices
        );

    std::sort(
        cuda_events.events.begin(),
        cuda_events.events.end(),
        event_order
    );
    std::sort(
        host_independent_events.begin(),
        host_independent_events.end(),
        event_order
    );
    std::sort(
        host_conflicting_events.begin(),
        host_conflicting_events.end(),
        event_order
    );
    std::sort(
        cuda_independent_events.begin(),
        cuda_independent_events.end(),
        event_order
    );
    std::sort(
        cuda_conflicting_events.begin(),
        cuda_conflicting_events.end(),
        event_order
    );

    const bool cell_list_matches =
        cuda.sorted_cell_ids == host.sorted_cell_ids &&
        cuda.particle_indices == host.particle_indices &&
        cuda.cell_start == host.cell_start &&
        cuda.cell_end == host.cell_end;
    const bool candidates_match =
        cuda_candidates.pairs == host_candidates;
    bool events_match = cuda_events.events.size() == host_events.size();
    if (events_match) {
        for (std::size_t index = 0; index < host_events.size(); ++index) {
            const particles::CollisionEvent& host_event = host_events[index];
            const particles::CollisionEvent& cuda_event =
                cuda_events.events[index];
            if (
                host_event.first != cuda_event.first ||
                host_event.second != cuda_event.second ||
                std::abs(host_event.time - cuda_event.time) > 1.0e-11 ||
                host_event.image_shift != cuda_event.image_shift
            ) {
                events_match = false;
                break;
            }
        }
    }
    const auto event_lists_match = [](
        const std::vector<particles::CollisionEvent>& left,
        const std::vector<particles::CollisionEvent>& right
    ) {
        if (left.size() != right.size()) {
            return false;
        }
        for (std::size_t index = 0; index < left.size(); ++index) {
            if (
                left[index].first != right[index].first ||
                left[index].second != right[index].second ||
                std::abs(left[index].time - right[index].time) > 1.0e-11 ||
                left[index].image_shift != right[index].image_shift
            ) {
                return false;
            }
        }
        return true;
    };
    const bool earliest_time_matches =
        (
            std::isinf(host_batch.earliest_time) &&
            std::isinf(cuda_batch.earliest_time)
        ) ||
        std::abs(
            host_batch.earliest_time - cuda_batch.earliest_time
        ) <= 1.0e-11;
    const bool collision_batch_matches =
        earliest_time_matches &&
        event_lists_match(
            host_independent_events,
            cuda_independent_events
        ) &&
        event_lists_match(
            host_conflicting_events,
            cuda_conflicting_events
        );

    particles::CollisionBatchAdvanceResult host_advance_result{};
    particles::CollisionBatchAdvanceResult cuda_advance_result{};
    std::vector<particles::Particle> host_advanced =
        initialized.particles;
    std::vector<particles::Particle> cuda_advanced =
        initialized.particles;
    if (!host_events.empty()) {
        host_advance_result =
            particles::advance_host_earliest_collision_batch(
                host_advanced,
                host_events,
                grid,
                config.particle_diameter
            );
        cuda_advance_result =
            device.advance_earliest_collision_batch(
                config.particle_diameter
            );
        cuda_advanced = device.download_particles();
    }

    bool advanced_state_matches =
        host_advanced.size() == cuda_advanced.size() &&
        std::abs(
            host_advance_result.delta_time -
            cuda_advance_result.delta_time
        ) <= 1.0e-11 &&
        host_advance_result.resolved_event_count ==
            cuda_advance_result.resolved_event_count &&
        host_advance_result.serialized_conflict_event_count ==
            cuda_advance_result.serialized_conflict_event_count &&
        host_advance_result.deferred_conflicting_event_count ==
            cuda_advance_result.deferred_conflicting_event_count;
    if (advanced_state_matches) {
        for (std::size_t index = 0; index < host_advanced.size(); ++index) {
            const particles::Particle& host_particle =
                host_advanced[index];
            const particles::Particle& cuda_particle =
                cuda_advanced[index];
            if (
                std::abs(
                    host_particle.position.x -
                    cuda_particle.position.x
                ) > 1.0e-11 ||
                std::abs(
                    host_particle.position.y -
                    cuda_particle.position.y
                ) > 1.0e-11 ||
                std::abs(
                    host_particle.position.z -
                    cuda_particle.position.z
                ) > 1.0e-11 ||
                std::abs(
                    host_particle.velocity.x -
                    cuda_particle.velocity.x
                ) > 1.0e-11 ||
                std::abs(
                    host_particle.velocity.y -
                    cuda_particle.velocity.y
                ) > 1.0e-11 ||
                std::abs(
                    host_particle.velocity.z -
                    cuda_particle.velocity.z
                ) > 1.0e-11
            ) {
                advanced_state_matches = false;
                break;
            }
        }
    }

    const auto conserved_quantities = [](
        const std::vector<particles::Particle>& particles
    ) {
        particles::Vec3 total_momentum{};
        double total_energy = 0.0;
        for (const particles::Particle& particle : particles) {
            total_momentum += particles::momentum(particle);
            total_energy += particles::kinetic_energy(particle);
        }
        return std::pair{total_momentum, total_energy};
    };
    const auto [momentum_before, energy_before] =
        conserved_quantities(initialized.particles);
    const auto [momentum_after, energy_after] =
        conserved_quantities(cuda_advanced);
    const double momentum_drift =
        particles::norm(momentum_after - momentum_before);
    const double energy_drift = std::abs(energy_after - energy_before);
    const double conservation_scale =
        std::max(1.0, std::abs(energy_before));
    const bool batch_conservation_passes =
        momentum_drift <= 1.0e-11 &&
        energy_drift <= 1.0e-11 * conservation_scale;

    double minimum_surface_gap = std::numeric_limits<double>::infinity();
    for (const particles::CandidatePair pair : host_candidates) {
        particles::Vec3 displacement =
            cuda_advanced[pair.first].position -
            cuda_advanced[pair.second].position;
        if (grid.boundary[0] == particles::BoundaryKind::periodic) {
            displacement.x -=
                std::round(displacement.x / grid.extent.x) *
                grid.extent.x;
        }
        if (grid.boundary[1] == particles::BoundaryKind::periodic) {
            displacement.y -=
                std::round(displacement.y / grid.extent.y) *
                grid.extent.y;
        }
        if (grid.boundary[2] == particles::BoundaryKind::periodic) {
            displacement.z -=
                std::round(displacement.z / grid.extent.z) *
                grid.extent.z;
        }
        minimum_surface_gap = std::min(
            minimum_surface_gap,
            particles::norm(displacement) -
                config.particle_diameter
        );
    }
    const bool no_penetration =
        std::isinf(minimum_surface_gap) ||
        minimum_surface_gap >= -1.0e-10;

    std::vector<particles::Particle> host_window_state =
        initialized.particles;
    const particles::TimeWindowAdvanceResult host_window_result =
        particles::advance_host_time_window(
            host_window_state,
            grid,
            config.diagnostics.sample_interval,
            config.particle_diameter,
            100'000
        );
    particles::cuda::ParticleGrid window_device{
        initialized.particles.size()
    };
    window_device.upload(initialized.particles);
    const particles::TimeWindowAdvanceResult cuda_window_result =
        window_device.advance_time_window(
            grid,
            config.diagnostics.sample_interval,
            config.particle_diameter,
            100'000
        );
    const std::vector<particles::Particle> cuda_window_state =
        window_device.download_particles();

    bool window_state_matches =
        host_window_state.size() == cuda_window_state.size();
    if (window_state_matches) {
        for (std::size_t index = 0;
             index < host_window_state.size();
             ++index) {
            const particles::Particle& host_particle =
                host_window_state[index];
            const particles::Particle& cuda_particle =
                cuda_window_state[index];
            if (
                std::abs(
                    host_particle.position.x -
                    cuda_particle.position.x
                ) > 1.0e-10 ||
                std::abs(
                    host_particle.position.y -
                    cuda_particle.position.y
                ) > 1.0e-10 ||
                std::abs(
                    host_particle.position.z -
                    cuda_particle.position.z
                ) > 1.0e-10 ||
                std::abs(
                    host_particle.velocity.x -
                    cuda_particle.velocity.x
                ) > 1.0e-10 ||
                std::abs(
                    host_particle.velocity.y -
                    cuda_particle.velocity.y
                ) > 1.0e-10 ||
                std::abs(
                    host_particle.velocity.z -
                    cuda_particle.velocity.z
                ) > 1.0e-10
            ) {
                window_state_matches = false;
                break;
            }
        }
    }
    const bool window_result_matches =
        std::abs(
            host_window_result.advanced_time -
            cuda_window_result.advanced_time
        ) <= 1.0e-10 &&
        host_window_result.batch_count ==
            cuda_window_result.batch_count &&
        host_window_result.candidate_rebuild_count ==
            cuda_window_result.candidate_rebuild_count &&
        host_window_result.resolved_collision_count ==
            cuda_window_result.resolved_collision_count &&
        host_window_result.serialized_conflict_resolution_count ==
            cuda_window_result.serialized_conflict_resolution_count &&
        host_window_result.reflected_particle_count ==
            cuda_window_result.reflected_particle_count &&
        host_window_result.reflected_axis_count ==
            cuda_window_result.reflected_axis_count &&
        host_window_result.deferred_conflicting_event_count ==
            cuda_window_result.deferred_conflicting_event_count &&
        host_window_result.stop_reason ==
            cuda_window_result.stop_reason;

    const auto [window_momentum, window_energy] =
        conserved_quantities(cuda_window_state);
    const double window_momentum_drift =
        particles::norm(window_momentum - momentum_before);
    const double window_energy_drift =
        std::abs(window_energy - energy_before);
    const bool all_boundaries_periodic =
        grid.boundary[0] == particles::BoundaryKind::periodic &&
        grid.boundary[1] == particles::BoundaryKind::periodic &&
        grid.boundary[2] == particles::BoundaryKind::periodic;
    const bool window_conservation_passes =
        (
            !all_boundaries_periodic ||
            window_momentum_drift <= 1.0e-10
        ) &&
        window_energy_drift <= 1.0e-10 * conservation_scale;

    const std::vector<particles::CandidatePair> window_candidates =
        particles::build_host_candidate_pairs(
            cuda_window_state,
            grid
        );
    double window_minimum_surface_gap =
        std::numeric_limits<double>::infinity();
    for (const particles::CandidatePair pair : window_candidates) {
        particles::Vec3 displacement =
            cuda_window_state[pair.first].position -
            cuda_window_state[pair.second].position;
        if (grid.boundary[0] == particles::BoundaryKind::periodic) {
            displacement.x -=
                std::round(displacement.x / grid.extent.x) *
                grid.extent.x;
        }
        if (grid.boundary[1] == particles::BoundaryKind::periodic) {
            displacement.y -=
                std::round(displacement.y / grid.extent.y) *
                grid.extent.y;
        }
        if (grid.boundary[2] == particles::BoundaryKind::periodic) {
            displacement.z -=
                std::round(displacement.z / grid.extent.z) *
                grid.extent.z;
        }
        window_minimum_surface_gap = std::min(
            window_minimum_surface_gap,
            particles::norm(displacement) -
                config.particle_diameter
        );
    }
    const bool window_no_penetration =
        std::isinf(window_minimum_surface_gap) ||
        window_minimum_surface_gap >= -1.0e-10;
    const auto stop_reason_name = [](
        particles::TimeWindowStopReason reason
    ) -> std::string_view {
        switch (reason) {
            case particles::TimeWindowStopReason::completed:
                return "completed";
            case particles::TimeWindowStopReason::collision_conflict:
                return "collision_conflict";
            case particles::TimeWindowStopReason::mixed_event_tie:
                return "mixed_event_tie";
            case particles::TimeWindowStopReason::batch_limit:
                return "batch_limit";
        }
        return "unknown";
    };

    const std::size_t occupied_cells = static_cast<std::size_t>(
        std::count_if(
            host.cell_start.begin(),
            host.cell_start.end(),
            [](std::uint32_t start) {
                return start != particles::empty_cell;
            }
        )
    );

    std::cout << "CUDA grid check\n"
              << "  particles: " << initialized.particles.size() << '\n'
              << "  dimensions: "
              << grid.dimensions[0] << " x "
              << grid.dimensions[1] << " x "
              << grid.dimensions[2] << '\n'
              << "  total cells: " << grid.cell_count << '\n'
              << "  occupied cells: " << occupied_cells << '\n'
              << "  candidate pairs: " << host_candidates.size() << '\n'
              << "  candidates per particle: "
              << (
                     2.0 *
                     static_cast<double>(host_candidates.size()) /
                     static_cast<double>(initialized.particles.size())
                 ) << '\n'
              << "  conservative event horizon: "
              << event_horizon << '\n'
              << "  valid collision events: " << host_events.size() << '\n'
              << "  earliest collision time: "
              << host_batch.earliest_time << '\n'
              << "  earliest batch events: "
              << host_batch.batch_size() << '\n'
              << "  parallel-safe events: "
              << host_batch.independent_event_indices.size() << '\n'
              << "  conflicting events: "
              << host_batch.conflicting_event_indices.size() << '\n'
              << "  batch advance time: "
              << host_advance_result.delta_time << '\n'
              << "  resolved batch events: "
              << host_advance_result.resolved_event_count << '\n'
              << "  serialized conflict events: "
              << host_advance_result.serialized_conflict_event_count
              << '\n'
              << "  deferred conflict events: "
              << host_advance_result.deferred_conflicting_event_count << '\n'
              << "  momentum drift: " << momentum_drift << '\n'
              << "  kinetic-energy drift: " << energy_drift << '\n'
              << "  minimum surface gap: " << minimum_surface_gap << '\n'
              << "  cell-list match: "
              << (cell_list_matches ? "yes" : "no") << '\n'
              << "  candidate-set match: "
              << (candidates_match ? "yes" : "no") << '\n'
              << "  collision-event match: "
              << (events_match ? "yes" : "no") << '\n'
              << "  collision-batch match: "
              << (collision_batch_matches ? "yes" : "no") << '\n'
              << "  advanced-state match: "
              << (advanced_state_matches ? "yes" : "no") << '\n'
              << "  batch conservation: "
              << (batch_conservation_passes ? "yes" : "no") << '\n'
              << "  no penetration: "
              << (no_penetration ? "yes" : "no") << '\n'
              << "CUDA time-window check\n"
              << "  requested time: "
              << cuda_window_result.requested_time << '\n'
              << "  advanced time: "
              << cuda_window_result.advanced_time << '\n'
              << "  stop reason: "
              << stop_reason_name(cuda_window_result.stop_reason) << '\n'
              << "  batches: "
              << cuda_window_result.batch_count << '\n'
              << "  candidate rebuilds: "
              << cuda_window_result.candidate_rebuild_count << '\n'
              << "  resolved collisions: "
              << cuda_window_result.resolved_collision_count << '\n'
              << "  serialized conflict resolutions: "
              << cuda_window_result.serialized_conflict_resolution_count
              << '\n'
              << "  reflected particles: "
              << cuda_window_result.reflected_particle_count << '\n'
              << "  reflected axes: "
              << cuda_window_result.reflected_axis_count << '\n'
              << "  deferred events: "
              << cuda_window_result.deferred_conflicting_event_count
              << '\n'
              << "  momentum drift: "
              << window_momentum_drift << '\n'
              << "  kinetic-energy drift: "
              << window_energy_drift << '\n'
              << "  minimum surface gap: "
              << window_minimum_surface_gap << '\n'
              << "  window-result match: "
              << (window_result_matches ? "yes" : "no") << '\n'
              << "  window-state match: "
              << (window_state_matches ? "yes" : "no") << '\n'
              << "  window conservation: "
              << (window_conservation_passes ? "yes" : "no") << '\n'
              << "  window no penetration: "
              << (window_no_penetration ? "yes" : "no") << '\n';

    return (
        cell_list_matches &&
        candidates_match &&
        events_match &&
        collision_batch_matches &&
        advanced_state_matches &&
        batch_conservation_passes &&
        no_penetration &&
        window_result_matches &&
        window_state_matches &&
        window_conservation_passes &&
        window_no_penetration
    ) ? 0 : 1;
}
#endif

}  // namespace

int main(int argc, char** argv) {
    try {
        if (
            argc >= 2 &&
            std::string_view{argv[1]} == "window-convergence"
        ) {
            const std::filesystem::path config_path =
                argc >= 3
                    ? std::filesystem::path{argv[2]}
                    : std::filesystem::path{"configs/equilibrium.json"};
            const std::filesystem::path output_path =
                argc >= 4
                    ? std::filesystem::path{argv[3]}
                    : std::filesystem::path{
                          "output/window-convergence.csv"
                      };
            return run_window_convergence(config_path, output_path);
        }

        if (argc >= 2 && std::string_view{argv[1]} == "equilibrium") {
            const std::filesystem::path config_path =
                argc >= 3
                    ? std::filesystem::path{argv[2]}
                    : std::filesystem::path{"configs/equilibrium.json"};
            const std::filesystem::path output_directory =
                argc >= 4
                    ? std::filesystem::path{argv[3]}
                    : std::filesystem::path{"output/equilibrium"};
            return run_equilibrium(config_path, output_directory);
        }

        if (argc >= 2 && std::string_view{argv[1]} == "cuda-grid") {
#if PARTICLES_HAS_CUDA
            const std::filesystem::path config_path =
                argc >= 3
                    ? std::filesystem::path{argv[2]}
                    : std::filesystem::path{"configs/equilibrium.json"};
            return run_cuda_grid_check(config_path);
#else
            throw std::runtime_error(
                "CUDA grid support was disabled at configure time."
            );
#endif
        }

        return run_smoke_test();
    } catch (const std::exception& exception) {
        std::cerr << "particles_cli: " << exception.what() << '\n';
        return 1;
    }
}
