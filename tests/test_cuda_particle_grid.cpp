#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <vector>

#include "particles/cuda_particle_grid.hpp"
#include "particles/cuda_support.hpp"
#include "particles/uniform_grid.hpp"

int main() {
    if (particles::cuda::device_count() == 0) {
        std::cout << "CUDA particle-grid test skipped: no CUDA device.\n";
        return 77;
    }

    try {
        const particles::DomainConfig domain{
            .extent = {4.0, 2.0, 2.0},
            .boundary = {
                particles::BoundaryKind::periodic,
                particles::BoundaryKind::periodic,
                particles::BoundaryKind::periodic,
            },
        };
        const particles::UniformGrid grid =
            particles::make_uniform_grid(domain, 1.0);
        std::vector<particles::Particle> particles{
            particles::Particle{
                .position = {0.1, 0.1, 0.1},
                .velocity = {1.0, 2.0, 3.0},
            },
            particles::Particle{
                .position = {3.9, 1.9, 1.9},
                .velocity = {-1.0, -2.0, -3.0},
            },
            particles::Particle{
                .position = {1.1, 0.1, 0.1},
                .velocity = {4.0, 5.0, 6.0},
            },
            particles::Particle{
                .position = {0.2, 0.2, 0.2},
                .velocity = {7.0, 8.0, 9.0},
            },
        };

        particles::cuda::ParticleGrid device{particles.size()};
        device.upload(particles);

        std::vector<particles::Particle> downloaded =
            device.download_particles();
        if (downloaded.size() != particles.size()) {
            std::cerr << "Downloaded particle count differs.\n";
            return 1;
        }
        for (std::size_t index = 0; index < particles.size(); ++index) {
            if (
                downloaded[index].position.x != particles[index].position.x ||
                downloaded[index].position.y != particles[index].position.y ||
                downloaded[index].position.z != particles[index].position.z ||
                downloaded[index].velocity.x != particles[index].velocity.x ||
                downloaded[index].velocity.y != particles[index].velocity.y ||
                downloaded[index].velocity.z != particles[index].velocity.z
            ) {
                std::cerr << "CUDA SoA round trip differs at particle "
                          << index << ".\n";
                return 1;
            }
        }

        constexpr double delta_time = 0.125;
        for (particles::Particle& particle : particles) {
            particles::advance(particle, delta_time);
        }
        device.advance_free_flight(delta_time);
        downloaded = device.download_particles();

        for (std::size_t index = 0; index < particles.size(); ++index) {
            if (
                std::abs(
                    downloaded[index].position.x -
                    particles[index].position.x
                ) > 1.0e-14 ||
                std::abs(
                    downloaded[index].position.y -
                    particles[index].position.y
                ) > 1.0e-14 ||
                std::abs(
                    downloaded[index].position.z -
                    particles[index].position.z
                ) > 1.0e-14
            ) {
                std::cerr << "CUDA free flight differs at particle "
                          << index << ".\n";
                return 1;
            }
        }

        const particles::HostCellList host =
            particles::build_host_cell_list(particles, grid);
        const std::vector<particles::CandidatePair> host_candidates =
            particles::build_host_candidate_pairs(particles, grid);
        device.build_cell_list(grid);
        const particles::cuda::CellListSnapshot snapshot =
            device.download_cell_list();

        if (
            snapshot.sorted_cell_ids != host.sorted_cell_ids ||
            snapshot.particle_indices != host.particle_indices ||
            snapshot.cell_start != host.cell_start ||
            snapshot.cell_end != host.cell_end
        ) {
            std::cerr << "CUDA cell list differs from the CPU reference.\n";
            return 1;
        }

        device.build_candidate_pairs();
        particles::cuda::CandidatePairSnapshot cuda_candidates =
            device.download_candidate_pairs();
        std::sort(
            cuda_candidates.pairs.begin(),
            cuda_candidates.pairs.end()
        );

        if (
            std::adjacent_find(
                cuda_candidates.pairs.begin(),
                cuda_candidates.pairs.end()
            ) != cuda_candidates.pairs.end()
        ) {
            std::cerr << "CUDA candidate list contains duplicate pairs.\n";
            return 1;
        }
        if (cuda_candidates.pairs != host_candidates) {
            std::cerr << "CUDA candidate pairs differ from the CPU reference.\n";
            return 1;
        }

        const std::vector<particles::Particle> collision_particles{
            particles::Particle{
                .position = {0.3, 1.0, 1.0},
                .velocity = {-0.1, 0.0, 0.0},
            },
            particles::Particle{
                .position = {3.7, 1.0, 1.0},
                .velocity = {0.1, 0.0, 0.0},
            },
            particles::Particle{
                .position = {3.9, 0.2, 0.2},
                .velocity = {1.0, 0.0, 0.0},
            },
        };
        constexpr double collision_diameter = 0.2;
        constexpr double collision_horizon = 3.0;
        const std::vector<particles::CandidatePair> collision_candidates =
            particles::build_host_candidate_pairs(collision_particles, grid);
        const std::vector<particles::CollisionEvent> host_events =
            particles::evaluate_host_collision_events(
                collision_particles,
                collision_candidates,
                grid,
                collision_diameter,
                collision_horizon
            );

        particles::cuda::ParticleGrid collision_device{
            collision_particles.size()
        };
        collision_device.upload(collision_particles);
        collision_device.build_cell_list(grid);
        collision_device.build_candidate_pairs();
        collision_device.evaluate_collision_events(
            collision_diameter,
            collision_horizon
        );
        const particles::cuda::CollisionEventSnapshot cuda_events =
            collision_device.download_collision_events();

        if (
            host_events.size() != 1 ||
            cuda_events.events.size() != host_events.size()
        ) {
            std::cerr << "CUDA collision-event count differs from CPU.\n";
            return 1;
        }

        const particles::CollisionEvent& host_event = host_events[0];
        const particles::CollisionEvent& cuda_event = cuda_events.events[0];
        if (
            cuda_event.first != host_event.first ||
            cuda_event.second != host_event.second ||
            std::abs(cuda_event.time - host_event.time) > 1.0e-11 ||
            cuda_event.image_shift != host_event.image_shift
        ) {
            std::cerr << "CUDA collision event differs from CPU reference.\n";
            return 1;
        }

        const particles::CollisionBatchAnalysis host_batch =
            particles::analyze_earliest_collision_batch(
                host_events,
                collision_particles.size()
            );
        collision_device.analyze_earliest_collision_batch();
        const particles::cuda::CollisionBatchSnapshot cuda_batch =
            collision_device.download_collision_batch();
        if (
            std::abs(cuda_batch.earliest_time - host_batch.earliest_time) >
                1.0e-11 ||
            cuda_batch.independent_event_indices !=
                host_batch.independent_event_indices ||
            cuda_batch.conflicting_event_indices !=
                host_batch.conflicting_event_indices
        ) {
            std::cerr
                << "CUDA independent collision batch differs from CPU.\n";
            return 1;
        }

        std::vector<particles::Particle> host_collision_advanced =
            collision_particles;
        const particles::CollisionBatchAdvanceResult host_advance_result =
            particles::advance_host_earliest_collision_batch(
                host_collision_advanced,
                host_events,
                grid,
                collision_diameter
            );
        const particles::CollisionBatchAdvanceResult cuda_advance_result =
            collision_device.advance_earliest_collision_batch(
                collision_diameter
            );
        const std::vector<particles::Particle> cuda_collision_advanced =
            collision_device.download_particles();
        if (
            std::abs(
                cuda_advance_result.delta_time -
                host_advance_result.delta_time
            ) > 1.0e-11 ||
            cuda_advance_result.resolved_event_count !=
                host_advance_result.resolved_event_count ||
            cuda_advance_result.serialized_conflict_event_count !=
                host_advance_result.serialized_conflict_event_count ||
            cuda_advance_result.deferred_conflicting_event_count !=
                host_advance_result.deferred_conflicting_event_count ||
            cuda_collision_advanced.size() !=
                host_collision_advanced.size()
        ) {
            std::cerr
                << "CUDA independent collision advance metadata differs "
                   "from CPU.\n";
            return 1;
        }
        for (std::size_t index = 0;
             index < host_collision_advanced.size();
             ++index) {
            const particles::Particle& host_particle =
                host_collision_advanced[index];
            const particles::Particle& cuda_particle =
                cuda_collision_advanced[index];
            if (
                std::abs(
                    cuda_particle.position.x -
                    host_particle.position.x
                ) > 1.0e-11 ||
                std::abs(
                    cuda_particle.position.y -
                    host_particle.position.y
                ) > 1.0e-11 ||
                std::abs(
                    cuda_particle.position.z -
                    host_particle.position.z
                ) > 1.0e-11 ||
                std::abs(
                    cuda_particle.velocity.x -
                    host_particle.velocity.x
                ) > 1.0e-11 ||
                std::abs(
                    cuda_particle.velocity.y -
                    host_particle.velocity.y
                ) > 1.0e-11 ||
                std::abs(
                    cuda_particle.velocity.z -
                    host_particle.velocity.z
                ) > 1.0e-11
            ) {
                std::cerr
                    << "CUDA independent collision state differs at particle "
                    << index << ".\n";
                return 1;
            }
        }
        if (
            std::abs(cuda_collision_advanced[2].position.x - 1.9) >
                1.0e-11
        ) {
            std::cerr
                << "CUDA batch advance did not wrap a periodic crosser.\n";
            return 1;
        }

        const particles::DomainConfig conflict_domain{
            .extent = {4.0, 4.0, 4.0},
            .boundary = {
                particles::BoundaryKind::periodic,
                particles::BoundaryKind::periodic,
                particles::BoundaryKind::periodic,
            },
        };
        const particles::UniformGrid conflict_grid =
            particles::make_uniform_grid(conflict_domain, 2.0);
        const std::vector<particles::Particle> conflict_particles{
            particles::Particle{
                .position = {0.8, 1.0, 1.0},
                .velocity = {1.0, 0.0, 0.0},
            },
            particles::Particle{
                .position = {2.0, 1.0, 1.0},
                .velocity = {0.0, 0.0, 0.0},
            },
            particles::Particle{
                .position = {3.2, 1.0, 1.0},
                .velocity = {-1.0, 0.0, 0.0},
            },
            particles::Particle{
                .position = {1.0, 2.4, 3.0},
                .velocity = {0.0, 0.5, 0.0},
            },
            particles::Particle{
                .position = {1.0, 3.6, 3.0},
                .velocity = {0.0, -0.5, 0.0},
            },
        };
        constexpr double conflict_diameter = 0.4;
        constexpr double conflict_horizon = 0.8;
        const std::vector<particles::CandidatePair> conflict_candidates =
            particles::build_host_candidate_pairs(
                conflict_particles,
                conflict_grid
            );
        const std::vector<particles::CollisionEvent> conflict_host_events =
            particles::evaluate_host_collision_events(
                conflict_particles,
                conflict_candidates,
                conflict_grid,
                conflict_diameter,
                conflict_horizon
            );
        const particles::CollisionBatchAnalysis conflict_host_batch =
            particles::analyze_earliest_collision_batch(
                conflict_host_events,
                conflict_particles.size()
            );

        particles::cuda::ParticleGrid conflict_device{
            conflict_particles.size()
        };
        conflict_device.upload(conflict_particles);
        conflict_device.build_cell_list(conflict_grid);
        conflict_device.build_candidate_pairs();
        conflict_device.evaluate_collision_events(
            conflict_diameter,
            conflict_horizon
        );
        conflict_device.analyze_earliest_collision_batch();
        const particles::cuda::CollisionEventSnapshot conflict_cuda_events =
            conflict_device.download_collision_events();
        const particles::cuda::CollisionBatchSnapshot conflict_cuda_batch =
            conflict_device.download_collision_batch();

        const auto selected_pairs = [](
            const std::vector<particles::CollisionEvent>& events,
            const std::vector<std::uint32_t>& indices
        ) {
            std::vector<particles::CandidatePair> pairs;
            pairs.reserve(indices.size());
            for (const std::uint32_t index : indices) {
                pairs.push_back(particles::CandidatePair{
                    .first = events[index].first,
                    .second = events[index].second,
                });
            }
            std::sort(pairs.begin(), pairs.end());
            return pairs;
        };

        if (
            conflict_host_events.size() != 3 ||
            conflict_host_batch.independent_event_indices.size() != 1 ||
            conflict_host_batch.conflicting_event_indices.size() != 2 ||
            conflict_cuda_batch.batch_size() !=
                conflict_host_batch.batch_size() ||
            std::abs(
                conflict_cuda_batch.earliest_time -
                conflict_host_batch.earliest_time
            ) > 1.0e-11 ||
            selected_pairs(
                conflict_cuda_events.events,
                conflict_cuda_batch.independent_event_indices
            ) != selected_pairs(
                conflict_host_events,
                conflict_host_batch.independent_event_indices
            ) ||
            selected_pairs(
                conflict_cuda_events.events,
                conflict_cuda_batch.conflicting_event_indices
            ) != selected_pairs(
                conflict_host_events,
                conflict_host_batch.conflicting_event_indices
            )
        ) {
            std::cerr
                << "CUDA collision-chain conflict analysis differs from CPU.\n";
            return 1;
        }

        std::vector<particles::Particle> conflict_host_advanced =
            conflict_particles;
        const particles::CollisionBatchAdvanceResult
            conflict_host_advance_result =
                particles::advance_host_earliest_collision_batch(
                    conflict_host_advanced,
                    conflict_host_events,
                    conflict_grid,
                    conflict_diameter
                );
        const particles::CollisionBatchAdvanceResult
            conflict_cuda_advance_result =
                conflict_device.advance_earliest_collision_batch(
                    conflict_diameter
                );
        const std::vector<particles::Particle> conflict_cuda_advanced =
            conflict_device.download_particles();
        if (
            conflict_cuda_advance_result.resolved_event_count != 2 ||
            conflict_cuda_advance_result.serialized_conflict_event_count !=
                1 ||
            conflict_cuda_advance_result.deferred_conflicting_event_count !=
                0 ||
            conflict_cuda_advance_result.resolved_event_count !=
                conflict_host_advance_result.resolved_event_count ||
            conflict_cuda_advance_result.serialized_conflict_event_count !=
                conflict_host_advance_result.serialized_conflict_event_count ||
            conflict_cuda_advance_result.deferred_conflicting_event_count !=
                conflict_host_advance_result.deferred_conflicting_event_count
        ) {
            std::cerr
                << "CUDA collision-chain advance metadata differs from CPU.\n";
            return 1;
        }
        for (std::size_t index = 0;
             index < conflict_host_advanced.size();
             ++index) {
            const particles::Particle& host_particle =
                conflict_host_advanced[index];
            const particles::Particle& cuda_particle =
                conflict_cuda_advanced[index];
            if (
                std::abs(
                    cuda_particle.position.x -
                    host_particle.position.x
                ) > 1.0e-11 ||
                std::abs(
                    cuda_particle.position.y -
                    host_particle.position.y
                ) > 1.0e-11 ||
                std::abs(
                    cuda_particle.position.z -
                    host_particle.position.z
                ) > 1.0e-11 ||
                std::abs(
                    cuda_particle.velocity.x -
                    host_particle.velocity.x
                ) > 1.0e-11 ||
                std::abs(
                    cuda_particle.velocity.y -
                    host_particle.velocity.y
                ) > 1.0e-11 ||
                std::abs(
                    cuda_particle.velocity.z -
                    host_particle.velocity.z
                ) > 1.0e-11
            ) {
                std::cerr
                    << "CUDA collision-chain state differs at particle "
                    << index << ".\n";
                return 1;
            }
        }

        const auto states_match = [](
            const std::vector<particles::Particle>& host_state,
            const std::vector<particles::Particle>& cuda_state
        ) {
            if (host_state.size() != cuda_state.size()) {
                return false;
            }
            for (std::size_t index = 0; index < host_state.size(); ++index) {
                const particles::Particle& host_particle =
                    host_state[index];
                const particles::Particle& cuda_particle =
                    cuda_state[index];
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
                    return false;
                }
            }
            return true;
        };
        const auto window_results_match = [](
            const particles::TimeWindowAdvanceResult& host_result,
            const particles::TimeWindowAdvanceResult& cuda_result
        ) {
            return
                std::abs(
                    host_result.requested_time -
                    cuda_result.requested_time
                ) <= 1.0e-11 &&
                std::abs(
                    host_result.advanced_time -
                    cuda_result.advanced_time
                ) <= 1.0e-11 &&
                host_result.batch_count == cuda_result.batch_count &&
                host_result.candidate_rebuild_count ==
                    cuda_result.candidate_rebuild_count &&
                host_result.resolved_collision_count ==
                    cuda_result.resolved_collision_count &&
                host_result.serialized_conflict_resolution_count ==
                    cuda_result.serialized_conflict_resolution_count &&
                host_result.reflected_particle_count ==
                    cuda_result.reflected_particle_count &&
                host_result.reflected_axis_count ==
                    cuda_result.reflected_axis_count &&
                host_result.deferred_conflicting_event_count ==
                    cuda_result.deferred_conflicting_event_count &&
                host_result.stop_reason == cuda_result.stop_reason;
        };
        const auto parallel_step_results_match = [](
            const particles::ParallelWindowStepResult& host_result,
            const particles::ParallelWindowStepResult& cuda_result
        ) {
            const bool gap_matches =
                (
                    std::isinf(host_result.minimum_surface_gap) &&
                    std::isinf(cuda_result.minimum_surface_gap)
                ) ||
                std::abs(
                    host_result.minimum_surface_gap -
                    cuda_result.minimum_surface_gap
                ) <= 1.0e-11;
            return
                std::abs(
                    host_result.requested_window -
                    cuda_result.requested_window
                ) <= 1.0e-11 &&
                std::abs(
                    host_result.advanced_time -
                    cuda_result.advanced_time
                ) <= 1.0e-11 &&
                std::abs(
                    host_result.candidate_horizon -
                    cuda_result.candidate_horizon
                ) <= 1.0e-11 &&
                host_result.candidate_pair_count ==
                    cuda_result.candidate_pair_count &&
                host_result.detected_collision_count ==
                    cuda_result.detected_collision_count &&
                host_result.accepted_collision_count ==
                    cuda_result.accepted_collision_count &&
                host_result.contended_collision_count ==
                    cuda_result.contended_collision_count &&
                host_result.overlap_pair_count ==
                    cuda_result.overlap_pair_count &&
                host_result.limited_by_candidate_horizon ==
                    cuda_result.limited_by_candidate_horizon &&
                gap_matches;
        };
        const auto parallel_run_results_match = [](
            const particles::ParallelWindowRunResult& host_result,
            const particles::ParallelWindowRunResult& cuda_result
        ) {
            const bool gap_matches =
                (
                    std::isinf(host_result.minimum_surface_gap) &&
                    std::isinf(cuda_result.minimum_surface_gap)
                ) ||
                std::abs(
                    host_result.minimum_surface_gap -
                    cuda_result.minimum_surface_gap
                ) <= 1.0e-11;
            return
                std::abs(
                    host_result.requested_time -
                    cuda_result.requested_time
                ) <= 1.0e-11 &&
                std::abs(
                    host_result.advanced_time -
                    cuda_result.advanced_time
                ) <= 1.0e-11 &&
                std::abs(
                    host_result.maximum_window -
                    cuda_result.maximum_window
                ) <= 1.0e-11 &&
                host_result.step_count == cuda_result.step_count &&
                host_result.candidate_rebuild_count ==
                    cuda_result.candidate_rebuild_count &&
                host_result.horizon_limited_step_count ==
                    cuda_result.horizon_limited_step_count &&
                host_result.candidate_pair_count ==
                    cuda_result.candidate_pair_count &&
                host_result.detected_collision_count ==
                    cuda_result.detected_collision_count &&
                host_result.accepted_collision_count ==
                    cuda_result.accepted_collision_count &&
                host_result.contended_collision_count ==
                    cuda_result.contended_collision_count &&
                host_result.overlap_pair_count ==
                    cuda_result.overlap_pair_count &&
                host_result.stop_reason == cuda_result.stop_reason &&
                gap_matches;
        };
        const auto adaptive_run_results_match = [](
            const particles::AdaptiveParallelWindowRunResult& host_result,
            const particles::AdaptiveParallelWindowRunResult& cuda_result
        ) {
            const auto scalar_matches = [](double left, double right) {
                return
                    (std::isinf(left) && std::isinf(right)) ||
                    std::abs(left - right) <= 1.0e-11;
            };
            return
                scalar_matches(
                    host_result.requested_time,
                    cuda_result.requested_time
                ) &&
                scalar_matches(
                    host_result.advanced_time,
                    cuda_result.advanced_time
                ) &&
                scalar_matches(
                    host_result.maximum_window,
                    cuda_result.maximum_window
                ) &&
                scalar_matches(
                    host_result.minimum_window,
                    cuda_result.minimum_window
                ) &&
                scalar_matches(
                    host_result.growth_factor,
                    cuda_result.growth_factor
                ) &&
                scalar_matches(
                    host_result.minimum_attempted_window,
                    cuda_result.minimum_attempted_window
                ) &&
                scalar_matches(
                    host_result.minimum_accepted_window,
                    cuda_result.minimum_accepted_window
                ) &&
                host_result.accepted_step_count ==
                    cuda_result.accepted_step_count &&
                host_result.attempt_count ==
                    cuda_result.attempt_count &&
                host_result.rollback_count ==
                    cuda_result.rollback_count &&
                host_result.maximum_refinement_depth ==
                    cuda_result.maximum_refinement_depth &&
                host_result.successful_steps_before_growth ==
                    cuda_result.successful_steps_before_growth &&
                host_result.window_growth_count ==
                    cuda_result.window_growth_count &&
                host_result.candidate_rebuild_count ==
                    cuda_result.candidate_rebuild_count &&
                host_result.horizon_limited_attempt_count ==
                    cuda_result.horizon_limited_attempt_count &&
                host_result.candidate_pair_count ==
                    cuda_result.candidate_pair_count &&
                host_result.detected_collision_count ==
                    cuda_result.detected_collision_count &&
                host_result.accepted_collision_count ==
                    cuda_result.accepted_collision_count &&
                host_result.contended_collision_count ==
                    cuda_result.contended_collision_count &&
                host_result.rejected_candidate_pair_count ==
                    cuda_result.rejected_candidate_pair_count &&
                host_result.rejected_detected_collision_count ==
                    cuda_result.rejected_detected_collision_count &&
                host_result.rejected_accepted_collision_count ==
                    cuda_result.rejected_accepted_collision_count &&
                host_result.rejected_contended_collision_count ==
                    cuda_result.rejected_contended_collision_count &&
                host_result.rejected_overlap_pair_count ==
                    cuda_result.rejected_overlap_pair_count &&
                host_result.exact_recovery_count ==
                    cuda_result.exact_recovery_count &&
                host_result.exact_recovery_batch_count ==
                    cuda_result.exact_recovery_batch_count &&
                host_result.exact_recovery_candidate_rebuild_count ==
                    cuda_result.exact_recovery_candidate_rebuild_count &&
                host_result.exact_recovery_collision_count ==
                    cuda_result.exact_recovery_collision_count &&
                host_result.
                    exact_recovery_serialized_conflict_resolution_count ==
                    cuda_result.
                        exact_recovery_serialized_conflict_resolution_count &&
                host_result.exact_recovery_reflected_particle_count ==
                    cuda_result.exact_recovery_reflected_particle_count &&
                host_result.exact_recovery_reflected_axis_count ==
                    cuda_result.exact_recovery_reflected_axis_count &&
                scalar_matches(
                    host_result.minimum_surface_gap,
                    cuda_result.minimum_surface_gap
                ) &&
                scalar_matches(
                    host_result.minimum_rejected_surface_gap,
                    cuda_result.minimum_rejected_surface_gap
                ) &&
                host_result.stop_reason == cuda_result.stop_reason;
        };

        const particles::DomainConfig corner_domain{
            .extent = {4.0, 4.0, 4.0},
            .boundary = {
                particles::BoundaryKind::reflecting,
                particles::BoundaryKind::reflecting,
                particles::BoundaryKind::reflecting,
            },
        };
        const particles::UniformGrid corner_grid =
            particles::make_uniform_grid(corner_domain, 2.0);
        const std::vector<particles::Particle> corner_initial{
            particles::Particle{
                .position = {3.0, 3.0, 2.0},
                .velocity = {1.0, 1.0, 0.0},
            },
        };
        std::vector<particles::Particle> corner_host = corner_initial;
        const particles::ReflectingWallBatch corner_host_batch =
            particles::find_host_reflecting_wall_batch(
                corner_host,
                corner_grid,
                1.0,
                1.0
            );
        particles::cuda::ParticleGrid corner_device{
            corner_initial.size()
        };
        corner_device.upload(corner_initial);
        corner_device.build_cell_list(corner_grid);
        corner_device.evaluate_reflecting_wall_batch(1.0, 1.0);
        const particles::cuda::ReflectingWallBatchSnapshot
            corner_cuda_batch =
                corner_device.download_reflecting_wall_batch();
        if (
            corner_cuda_batch.events != corner_host_batch.events ||
            std::abs(
                corner_cuda_batch.earliest_time -
                corner_host_batch.earliest_time
            ) > 1.0e-11
        ) {
            std::cerr
                << "CUDA reflecting-wall corner batch differs from CPU.\n";
            return 1;
        }
        const particles::ReflectingWallAdvanceResult
            corner_host_result =
                particles::advance_host_reflecting_wall_batch(
                    corner_host,
                    corner_host_batch,
                    corner_grid,
                    1.0
                );
        const particles::ReflectingWallAdvanceResult
            corner_cuda_result =
                corner_device.advance_reflecting_wall_batch(1.0);
        const std::vector<particles::Particle> corner_cuda =
            corner_device.download_particles();
        if (
            std::abs(
                corner_host_result.delta_time -
                corner_cuda_result.delta_time
            ) > 1.0e-11 ||
            corner_host_result.reflected_particle_count !=
                corner_cuda_result.reflected_particle_count ||
            corner_host_result.reflected_axis_count !=
                corner_cuda_result.reflected_axis_count ||
            !states_match(corner_host, corner_cuda)
        ) {
            std::cerr
                << "CUDA reflecting-wall corner advance differs from CPU.\n";
            return 1;
        }

        const particles::DomainConfig wall_domain{
            .extent = {4.0, 2.0, 2.0},
            .boundary = {
                particles::BoundaryKind::reflecting,
                particles::BoundaryKind::periodic,
                particles::BoundaryKind::periodic,
            },
        };
        const particles::UniformGrid wall_grid =
            particles::make_uniform_grid(wall_domain, 2.0);
        const std::vector<particles::Particle> wall_initial{
            particles::Particle{
                .position = {1.0, 1.0, 1.0},
                .velocity = {1.0, 0.0, 0.0},
            },
        };
        std::vector<particles::Particle> wall_host = wall_initial;
        const particles::TimeWindowAdvanceResult wall_host_result =
            particles::advance_host_time_window(
                wall_host,
                wall_grid,
                6.0,
                1.0,
                100
            );
        particles::cuda::ParticleGrid wall_device{wall_initial.size()};
        wall_device.upload(wall_initial);
        const particles::TimeWindowAdvanceResult wall_cuda_result =
            wall_device.advance_time_window(
                wall_grid,
                6.0,
                1.0,
                100
            );
        const std::vector<particles::Particle> wall_cuda =
            wall_device.download_particles();
        if (
            !window_results_match(
                wall_host_result,
                wall_cuda_result
            ) ||
            !states_match(wall_host, wall_cuda)
        ) {
            std::cerr
                << "CUDA repeated reflecting-wall window differs from CPU.\n";
            return 1;
        }

        const std::vector<particles::Particle> window_collision_initial{
            particles::Particle{
                .position = {1.0, 1.0, 1.0},
                .velocity = {0.1, 0.0, 0.0},
            },
            particles::Particle{
                .position = {2.2, 1.0, 1.0},
                .velocity = {-0.1, 0.0, 0.0},
            },
        };
        const particles::UniformGrid window_collision_grid =
            particles::make_uniform_grid(domain, 2.0);
        std::vector<particles::Particle> window_collision_host =
            window_collision_initial;
        const particles::TimeWindowAdvanceResult
            window_collision_host_result =
                particles::advance_host_time_window(
                    window_collision_host,
                    window_collision_grid,
                    2.0,
                    1.0,
                    100
                );
        particles::cuda::ParticleGrid window_collision_device{
            window_collision_initial.size()
        };
        window_collision_device.upload(window_collision_initial);
        const particles::TimeWindowAdvanceResult
            window_collision_cuda_result =
                window_collision_device.advance_time_window(
                    window_collision_grid,
                    2.0,
                    1.0,
                    100
                );
        const std::vector<particles::Particle> window_collision_cuda =
            window_collision_device.download_particles();
        if (
            !window_results_match(
                window_collision_host_result,
                window_collision_cuda_result
            ) ||
            !states_match(
                window_collision_host,
                window_collision_cuda
            )
        ) {
            std::cerr
                << "CUDA repeated collision window differs from CPU.\n";
            return 1;
        }

        std::vector<particles::Particle> window_conflict_host =
            conflict_particles;
        const particles::TimeWindowAdvanceResult
            window_conflict_host_result =
                particles::advance_host_time_window(
                    window_conflict_host,
                    conflict_grid,
                    1.0,
                    conflict_diameter,
                    10
                );
        particles::cuda::ParticleGrid window_conflict_device{
            conflict_particles.size()
        };
        window_conflict_device.upload(conflict_particles);
        const particles::TimeWindowAdvanceResult
            window_conflict_cuda_result =
                window_conflict_device.advance_time_window(
                    conflict_grid,
                    1.0,
                    conflict_diameter,
                    10
                );
        const std::vector<particles::Particle> window_conflict_cuda =
            window_conflict_device.download_particles();
        if (
            !window_results_match(
                window_conflict_host_result,
                window_conflict_cuda_result
            ) ||
            !states_match(
                window_conflict_host,
                window_conflict_cuda
            )
        ) {
            std::cerr
                << "CUDA collision-conflict window differs from CPU.\n";
            return 1;
        }

        const particles::DomainConfig mixed_domain{
            .extent = {6.0, 4.0, 4.0},
            .boundary = {
                particles::BoundaryKind::reflecting,
                particles::BoundaryKind::periodic,
                particles::BoundaryKind::periodic,
            },
        };
        const particles::UniformGrid mixed_grid =
            particles::make_uniform_grid(mixed_domain, 2.0);
        const std::vector<particles::Particle> mixed_initial{
            particles::Particle{
                .position = {1.0, 1.0, 1.0},
                .velocity = {0.5, 0.0, 0.0},
            },
            particles::Particle{
                .position = {3.0, 1.0, 1.0},
                .velocity = {-0.5, 0.0, 0.0},
            },
            particles::Particle{
                .position = {4.0, 3.0, 3.0},
                .velocity = {1.5, 0.0, 0.0},
            },
        };
        std::vector<particles::Particle> mixed_host = mixed_initial;
        const particles::TimeWindowAdvanceResult mixed_host_result =
            particles::advance_host_time_window(
                mixed_host,
                mixed_grid,
                2.0,
                1.0,
                10
            );
        particles::cuda::ParticleGrid mixed_device{mixed_initial.size()};
        mixed_device.upload(mixed_initial);
        const particles::TimeWindowAdvanceResult mixed_cuda_result =
            mixed_device.advance_time_window(
                mixed_grid,
                2.0,
                1.0,
                10
            );
        const std::vector<particles::Particle> mixed_cuda =
            mixed_device.download_particles();
        if (
            !window_results_match(mixed_host_result, mixed_cuda_result) ||
            !states_match(mixed_host, mixed_cuda)
        ) {
            std::cerr
                << "CUDA mixed wall-collision tie differs from CPU.\n";
            return 1;
        }

        const particles::DomainConfig parallel_domain{
            .extent = {12.0, 4.0, 4.0},
            .boundary = {
                particles::BoundaryKind::periodic,
                particles::BoundaryKind::periodic,
                particles::BoundaryKind::periodic,
            },
        };
        const particles::UniformGrid parallel_grid =
            particles::make_uniform_grid(parallel_domain, 2.0);
        const std::vector<particles::Particle> parallel_initial{
            particles::Particle{
                .position = {1.0, 1.0, 1.0},
                .velocity = {0.5, 0.0, 0.0},
            },
            particles::Particle{
                .position = {2.2, 1.0, 1.0},
                .velocity = {-0.5, 0.0, 0.0},
            },
            particles::Particle{
                .position = {7.0, 3.0, 3.0},
                .velocity = {0.5, 0.0, 0.0},
            },
            particles::Particle{
                .position = {8.2, 3.0, 3.0},
                .velocity = {-0.5, 0.0, 0.0},
            },
        };
        std::vector<particles::Particle> parallel_host =
            parallel_initial;
        const particles::ParallelWindowStepResult parallel_host_result =
            particles::advance_host_parallel_window_step(
                parallel_host,
                parallel_grid,
                0.5,
                1.0
            );
        particles::cuda::ParticleGrid parallel_device{
            parallel_initial.size()
        };
        parallel_device.upload(parallel_initial);
        const particles::ParallelWindowStepResult parallel_cuda_result =
            parallel_device.advance_parallel_window_step(
                parallel_grid,
                0.5,
                1.0
            );
        const std::vector<particles::Particle> parallel_cuda =
            parallel_device.download_particles();
        if (
            !parallel_step_results_match(
                parallel_host_result,
                parallel_cuda_result
            ) ||
            !states_match(parallel_host, parallel_cuda)
        ) {
            std::cerr
                << "CUDA disjoint parallel window differs from CPU.\n";
            return 1;
        }

        const particles::DomainConfig reflecting_parallel_domain{
            .extent = {4.0, 2.0, 2.0},
            .boundary = {
                particles::BoundaryKind::reflecting,
                particles::BoundaryKind::periodic,
                particles::BoundaryKind::periodic,
            },
        };
        const particles::UniformGrid reflecting_parallel_grid =
            particles::make_uniform_grid(
                reflecting_parallel_domain,
                1.0
            );
        const std::vector<particles::Particle>
            reflecting_parallel_initial{
                particles::Particle{
                    .position = {0.35, 1.0, 1.0},
                    .velocity = {-1.0, 0.0, 0.0},
                },
            };
        std::vector<particles::Particle> reflecting_parallel_host =
            reflecting_parallel_initial;
        const particles::ParallelWindowRunResult
            reflecting_parallel_host_result =
                particles::advance_host_parallel_windows(
                    reflecting_parallel_host,
                    reflecting_parallel_grid,
                    0.2,
                    0.2,
                    0.5
                );
        particles::cuda::ParticleGrid reflecting_parallel_device{
            reflecting_parallel_initial.size()
        };
        reflecting_parallel_device.upload(
            reflecting_parallel_initial
        );
        const particles::ParallelWindowRunResult
            reflecting_parallel_cuda_result =
                reflecting_parallel_device.advance_parallel_windows(
                    reflecting_parallel_grid,
                    0.2,
                    0.2,
                    0.5
                );
        const std::vector<particles::Particle>
            reflecting_parallel_cuda =
                reflecting_parallel_device.download_particles();
        if (
            !parallel_run_results_match(
                reflecting_parallel_host_result,
                reflecting_parallel_cuda_result
            ) ||
            !states_match(
                reflecting_parallel_host,
                reflecting_parallel_cuda
            )
        ) {
            std::cerr
                << "CUDA reflecting parallel window differs from CPU.\n";
            return 1;
        }

        const particles::UniformGrid
            reflecting_collision_grid =
                particles::make_uniform_grid(
                    reflecting_parallel_domain,
                    2.0
                );
        const std::vector<particles::Particle>
            reflecting_collision_initial{
                particles::Particle{
                    .position = {0.6, 1.0, 1.0},
                    .velocity = {0.0, 0.0, 0.0},
                },
                particles::Particle{
                    .position = {1.4, 1.0, 1.0},
                    .velocity = {-2.0, 0.0, 0.0},
                },
            };
        std::vector<particles::Particle>
            reflecting_collision_host =
                reflecting_collision_initial;
        const particles::ParallelWindowStepResult
            reflecting_collision_host_result =
                particles::advance_host_parallel_window_step(
                    reflecting_collision_host,
                    reflecting_collision_grid,
                    0.375,
                    0.5
                );
        particles::cuda::ParticleGrid
            reflecting_collision_device{
                reflecting_collision_initial.size()
            };
        reflecting_collision_device.upload(
            reflecting_collision_initial
        );
        const particles::ParallelWindowStepResult
            reflecting_collision_cuda_result =
                reflecting_collision_device.
                    advance_parallel_window_step(
                        reflecting_collision_grid,
                        0.375,
                        0.5
                    );
        const std::vector<particles::Particle>
            reflecting_collision_cuda =
                reflecting_collision_device.download_particles();
        if (
            !parallel_step_results_match(
                reflecting_collision_host_result,
                reflecting_collision_cuda_result
            ) ||
            !states_match(
                reflecting_collision_host,
                reflecting_collision_cuda
            )
        ) {
            std::cerr
                << "CUDA collision-then-reflection window differs "
                   "from CPU.\n";
            return 1;
        }

        const std::vector<particles::Particle> parallel_chain_initial{
            particles::Particle{
                .position = {1.0, 2.0, 2.0},
                .velocity = {1.0, 0.0, 0.0},
            },
            particles::Particle{
                .position = {2.2, 2.0, 2.0},
                .velocity = {0.0, 0.0, 0.0},
            },
            particles::Particle{
                .position = {3.4, 2.0, 2.0},
                .velocity = {-1.0, 0.0, 0.0},
            },
        };
        std::vector<particles::Particle> parallel_chain_host =
            parallel_chain_initial;
        const particles::ParallelWindowStepResult
            parallel_chain_host_result =
                particles::advance_host_parallel_window_step(
                    parallel_chain_host,
                    parallel_grid,
                    0.5,
                    1.0
                );
        particles::cuda::ParticleGrid parallel_chain_device{
            parallel_chain_initial.size()
        };
        parallel_chain_device.upload(parallel_chain_initial);
        const particles::ParallelWindowStepResult
            parallel_chain_cuda_result =
                parallel_chain_device.advance_parallel_window_step(
                    parallel_grid,
                    0.5,
                    1.0
                );
        const std::vector<particles::Particle> parallel_chain_cuda =
            parallel_chain_device.download_particles();
        if (
            !parallel_step_results_match(
                parallel_chain_host_result,
                parallel_chain_cuda_result
            ) ||
            !states_match(
                parallel_chain_host,
                parallel_chain_cuda
            ) ||
            parallel_chain_cuda_result.contended_collision_count != 1 ||
            parallel_chain_cuda_result.overlap_pair_count != 1
        ) {
            std::cerr
                << "CUDA contended parallel window differs from CPU.\n";
            return 1;
        }

        const std::vector<particles::Particle> parallel_run_initial{
            particles::Particle{
                .position = {1.0, 2.0, 2.0},
                .velocity = {1.0, 0.0, 0.0},
            },
            particles::Particle{
                .position = {2.4, 2.0, 2.0},
                .velocity = {-1.0, 0.0, 0.0},
            },
        };
        std::vector<particles::Particle> parallel_run_host =
            parallel_run_initial;
        const particles::ParallelWindowRunResult
            parallel_run_host_result =
                particles::advance_host_parallel_windows(
                    parallel_run_host,
                    parallel_grid,
                    0.6,
                    0.3,
                    1.0
                );
        particles::cuda::ParticleGrid parallel_run_device{
            parallel_run_initial.size()
        };
        parallel_run_device.upload(parallel_run_initial);
        const particles::ParallelWindowRunResult
            parallel_run_cuda_result =
                parallel_run_device.advance_parallel_windows(
                    parallel_grid,
                    0.6,
                    0.3,
                    1.0
                );
        const std::vector<particles::Particle> parallel_run_cuda =
            parallel_run_device.download_particles();
        if (
            !parallel_run_results_match(
                parallel_run_host_result,
                parallel_run_cuda_result
            ) ||
            !states_match(parallel_run_host, parallel_run_cuda)
        ) {
            std::cerr
                << "CUDA repeated parallel windows differ from CPU.\n";
            return 1;
        }

        const std::vector<particles::Particle> causal_chain_initial{
            particles::Particle{
                .position = {1.0, 2.0, 2.0},
                .velocity = {1.0, 0.0, 0.0},
            },
            particles::Particle{
                .position = {2.2, 2.0, 2.0},
                .velocity = {0.0, 0.0, 0.0},
            },
            particles::Particle{
                .position = {3.4, 2.0, 2.0},
                .velocity = {0.0, 0.0, 0.0},
            },
        };
        for (const double maximum_window : {0.5, 0.25, 0.125}) {
            std::vector<particles::Particle> causal_chain_host =
                causal_chain_initial;
            const particles::ParallelWindowRunResult
                causal_chain_host_result =
                    particles::advance_host_parallel_windows(
                        causal_chain_host,
                        parallel_grid,
                        0.8,
                        maximum_window,
                        1.0
                    );
            particles::cuda::ParticleGrid causal_chain_device{
                causal_chain_initial.size()
            };
            causal_chain_device.upload(causal_chain_initial);
            const particles::ParallelWindowRunResult
                causal_chain_cuda_result =
                    causal_chain_device.advance_parallel_windows(
                        parallel_grid,
                        0.8,
                        maximum_window,
                        1.0
                    );
            const std::vector<particles::Particle> causal_chain_cuda =
                causal_chain_device.download_particles();
            if (
                !parallel_run_results_match(
                    causal_chain_host_result,
                    causal_chain_cuda_result
                ) ||
                !states_match(
                    causal_chain_host,
                    causal_chain_cuda
                )
            ) {
                std::cerr
                    << "CUDA causal-chain convergence differs from CPU "
                    << "at maximum window " << maximum_window << ".\n";
                return 1;
            }
        }

        std::vector<particles::Particle> adaptive_chain_host =
            causal_chain_initial;
        const particles::AdaptiveParallelWindowRunResult
            adaptive_chain_host_result =
                particles::advance_host_adaptive_parallel_windows(
                    adaptive_chain_host,
                    parallel_grid,
                    0.8,
                    0.5,
                    0.03125,
                    1.0,
                    1'000'000,
                    1.0e-12,
                    2,
                    2.0
                );
        particles::cuda::ParticleGrid adaptive_chain_device{
            causal_chain_initial.size()
        };
        adaptive_chain_device.upload(causal_chain_initial);
        const particles::AdaptiveParallelWindowRunResult
            adaptive_chain_cuda_result =
                adaptive_chain_device.advance_adaptive_parallel_windows(
                    parallel_grid,
                    0.8,
                    0.5,
                    0.03125,
                    1.0,
                    1'000'000,
                    1.0e-12,
                    2,
                    2.0
                );
        const std::vector<particles::Particle> adaptive_chain_cuda =
            adaptive_chain_device.download_particles();
        if (
            !adaptive_run_results_match(
                adaptive_chain_host_result,
                adaptive_chain_cuda_result
            ) ||
            !states_match(
                adaptive_chain_host,
                adaptive_chain_cuda
            )
        ) {
            std::cerr
                << "CUDA adaptive causal-chain rollback differs from "
                   "CPU.\n";
            return 1;
        }

        std::vector<particles::Particle> recovered_chain_host =
            causal_chain_initial;
        const particles::AdaptiveParallelWindowRunResult
            recovered_chain_host_result =
                particles::advance_host_adaptive_parallel_windows(
                    recovered_chain_host,
                    parallel_grid,
                    0.8,
                    0.5,
                    0.5,
                    1.0,
                    1'000'000,
                    1.0e-12,
                    2,
                    2.0,
                    100
                );
        particles::cuda::ParticleGrid recovered_chain_device{
            causal_chain_initial.size()
        };
        recovered_chain_device.upload(causal_chain_initial);
        const particles::AdaptiveParallelWindowRunResult
            recovered_chain_cuda_result =
                recovered_chain_device.
                    advance_adaptive_parallel_windows(
                        parallel_grid,
                        0.8,
                        0.5,
                        0.5,
                        1.0,
                        1'000'000,
                        1.0e-12,
                        2,
                        2.0,
                        100
                    );
        const std::vector<particles::Particle>
            recovered_chain_cuda =
                recovered_chain_device.download_particles();
        if (
            !adaptive_run_results_match(
                recovered_chain_host_result,
                recovered_chain_cuda_result
            ) ||
            !states_match(
                recovered_chain_host,
                recovered_chain_cuda
            ) ||
            recovered_chain_cuda_result.exact_recovery_count != 1
        ) {
            std::cerr
                << "CUDA exact recovery from minimum-window exhaustion "
                   "differs from CPU.\n";
            return 1;
        }

        std::cout
            << "CUDA SoA, free flight, cell list, and candidate pairs "
               "match the CPU reference; event timing and conflict analysis "
               "also match; fixed and adaptive approximate windows match "
               "too.\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "CUDA particle-grid test failed: "
                  << exception.what() << '\n';
        return 1;
    }
}
