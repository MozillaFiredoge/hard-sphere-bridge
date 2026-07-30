#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "particles/collision_events.hpp"

namespace {

class CollisionEventTestFailure final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void expect_true(bool condition, std::string_view message) {
    if (!condition) {
        throw CollisionEventTestFailure(std::string(message));
    }
}

void expect_near(double actual, double expected, double tolerance, std::string_view message) {
    if (std::abs(actual - expected) > tolerance) {
        throw CollisionEventTestFailure(
            std::string(message) + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
        );
    }
}

void expect_vec_near(
    const particles::Vec3& actual,
    const particles::Vec3& expected,
    double tolerance,
    std::string_view message
) {
    expect_near(actual.x, expected.x, tolerance, message);
    expect_near(actual.y, expected.y, tolerance, message);
    expect_near(actual.z, expected.z, tolerance, message);
}

[[nodiscard]] particles::Vec3 total_momentum(
    const std::vector<particles::Particle>& particles
) {
    particles::Vec3 result{};
    for (const particles::Particle& particle : particles) {
        result += particles::momentum(particle);
    }
    return result;
}

[[nodiscard]] double total_kinetic_energy(
    const std::vector<particles::Particle>& particles
) {
    double result = 0.0;
    for (const particles::Particle& particle : particles) {
        result += particles::kinetic_energy(particle);
    }
    return result;
}

[[nodiscard]] particles::DomainConfig periodic_domain() {
    return particles::DomainConfig{
        .extent = {4.0, 2.0, 2.0},
        .boundary = {
            particles::BoundaryKind::periodic,
            particles::BoundaryKind::periodic,
            particles::BoundaryKind::periodic,
        },
    };
}

void test_direct_collision_time_from_candidates() {
    particles::DomainConfig domain = periodic_domain();
    domain.boundary = {
        particles::BoundaryKind::reflecting,
        particles::BoundaryKind::reflecting,
        particles::BoundaryKind::reflecting,
    };
    const particles::UniformGrid grid =
        particles::make_uniform_grid(domain, 2.0);
    const std::vector<particles::Particle> particles{
        particles::Particle{
            .position = {1.0, 1.0, 1.0},
            .velocity = {0.1, 0.0, 0.0},
        },
        particles::Particle{
            .position = {2.2, 1.0, 1.0},
            .velocity = {-0.1, 0.0, 0.0},
        },
    };
    const std::vector<particles::CandidatePair> candidates =
        particles::build_host_candidate_pairs(particles, grid);
    const std::vector<particles::CollisionEvent> events =
        particles::evaluate_host_collision_events(
            particles,
            candidates,
            grid,
            1.0,
            2.0
        );

    expect_true(candidates.size() == 1, "one direct candidate");
    expect_true(events.size() == 1, "one direct collision event");
    expect_near(events[0].time, 1.0, 1.0e-12, "direct collision time");
    expect_true(
        events[0].image_shift == particles::PeriodicImageShift{},
        "direct collision image shift"
    );
}

void test_periodic_image_collision_time() {
    const particles::DomainConfig domain = periodic_domain();
    const particles::UniformGrid grid =
        particles::make_uniform_grid(domain, 1.0);
    const std::vector<particles::Particle> particles{
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
    const std::vector<particles::CandidatePair> candidates =
        particles::build_host_candidate_pairs(particles, grid);
    const std::vector<particles::CollisionEvent> events =
        particles::evaluate_host_collision_events(
            particles,
            candidates,
            grid,
            0.2,
            3.0
        );

    expect_true(events.size() == 1, "one periodic collision event");
    expect_near(events[0].time, 2.0, 1.0e-11, "periodic collision time");
    expect_true(
        events[0].image_shift ==
            particles::PeriodicImageShift{.x = -1, .y = 0, .z = 0},
        "periodic collision image shift"
    );
}

void test_diverging_candidate_has_no_event() {
    const particles::DomainConfig domain = periodic_domain();
    const particles::UniformGrid grid =
        particles::make_uniform_grid(domain, 1.0);
    const std::vector<particles::Particle> particles{
        particles::Particle{
            .position = {1.0, 1.0, 1.0},
            .velocity = {-0.1, 0.0, 0.0},
        },
        particles::Particle{
            .position = {1.5, 1.0, 1.0},
            .velocity = {0.1, 0.0, 0.0},
        },
    };
    const std::vector<particles::CandidatePair> candidates =
        particles::build_host_candidate_pairs(particles, grid);
    const std::vector<particles::CollisionEvent> events =
        particles::evaluate_host_collision_events(
            particles,
            candidates,
            grid,
            0.2,
            1.0
        );

    expect_true(candidates.size() == 1, "one diverging candidate");
    expect_true(events.empty(), "diverging pair must not produce an event");
}

void test_conservative_horizon_uses_motion_and_wall_bounds() {
    particles::DomainConfig domain = periodic_domain();
    domain.boundary = {
        particles::BoundaryKind::reflecting,
        particles::BoundaryKind::periodic,
        particles::BoundaryKind::periodic,
    };
    const particles::UniformGrid grid =
        particles::make_uniform_grid(domain, 2.0);

    const std::vector<particles::Particle> central{
        particles::Particle{
            .position = {1.0, 1.0, 1.0},
            .velocity = {1.0, 0.0, 0.0},
        },
    };
    expect_near(
        particles::conservative_candidate_horizon(central, grid, 1.0),
        0.5,
        1.0e-12,
        "cell-motion horizon"
    );

    const std::vector<particles::Particle> near_wall{
        particles::Particle{
            .position = {3.4, 1.0, 1.0},
            .velocity = {1.0, 0.0, 0.0},
        },
    };
    expect_near(
        particles::conservative_candidate_horizon(near_wall, grid, 1.0),
        0.1,
        1.0e-12,
        "reflecting-wall horizon"
    );
}

void test_earliest_batch_selects_disjoint_events() {
    const std::vector<particles::CollisionEvent> events{
        particles::CollisionEvent{
            .first = 0,
            .second = 1,
            .time = 1.0,
        },
        particles::CollisionEvent{
            .first = 2,
            .second = 3,
            .time = 1.0 + 0.5e-12,
        },
        particles::CollisionEvent{
            .first = 4,
            .second = 5,
            .time = 1.0 + 2.0e-12,
        },
    };

    const particles::CollisionBatchAnalysis analysis =
        particles::analyze_earliest_collision_batch(
            events,
            6,
            1.0e-12
        );

    expect_near(analysis.earliest_time, 1.0, 1.0e-15, "earliest batch time");
    expect_true(analysis.batch_size() == 2, "anchored earliest batch size");
    expect_true(
        analysis.independent_event_indices ==
            std::vector<std::uint32_t>{0, 1},
        "disjoint events are independent"
    );
    expect_true(
        analysis.conflicting_event_indices.empty(),
        "disjoint batch has no conflicts"
    );
}

void test_earliest_batch_quarantines_collision_chain() {
    const std::vector<particles::CollisionEvent> events{
        particles::CollisionEvent{
            .first = 0,
            .second = 1,
            .time = 0.75,
        },
        particles::CollisionEvent{
            .first = 1,
            .second = 2,
            .time = 0.75,
        },
        particles::CollisionEvent{
            .first = 3,
            .second = 4,
            .time = 0.75,
        },
    };

    const particles::CollisionBatchAnalysis analysis =
        particles::analyze_earliest_collision_batch(events, 5);

    expect_true(analysis.batch_size() == 3, "chain batch size");
    expect_true(
        analysis.independent_event_indices ==
            std::vector<std::uint32_t>{2},
        "disjoint event remains parallel-safe"
    );
    expect_true(
        analysis.conflicting_event_indices ==
            std::vector<std::uint32_t>{0, 1},
        "shared-particle chain is quarantined"
    );
}

void test_empty_collision_batch() {
    const particles::CollisionBatchAnalysis analysis =
        particles::analyze_earliest_collision_batch({}, 0);

    expect_true(
        std::isinf(analysis.earliest_time),
        "empty batch has infinite earliest time"
    );
    expect_true(analysis.batch_size() == 0, "empty batch size");
}

void test_advance_independent_collision_batch() {
    particles::DomainConfig domain = periodic_domain();
    domain.boundary = {
        particles::BoundaryKind::reflecting,
        particles::BoundaryKind::reflecting,
        particles::BoundaryKind::reflecting,
    };
    const particles::UniformGrid grid =
        particles::make_uniform_grid(domain, 2.0);
    std::vector<particles::Particle> particles{
        particles::Particle{
            .position = {1.0, 1.0, 1.0},
            .velocity = {0.1, 0.0, 0.0},
        },
        particles::Particle{
            .position = {2.2, 1.0, 1.0},
            .velocity = {-0.1, 0.0, 0.0},
        },
    };
    const std::vector<particles::CandidatePair> candidates =
        particles::build_host_candidate_pairs(particles, grid);
    const std::vector<particles::CollisionEvent> events =
        particles::evaluate_host_collision_events(
            particles,
            candidates,
            grid,
            1.0,
            2.0
        );
    const particles::Vec3 momentum_before = total_momentum(particles);
    const double energy_before = total_kinetic_energy(particles);

    const particles::CollisionBatchAdvanceResult result =
        particles::advance_host_earliest_collision_batch(
            particles,
            events,
            grid,
            1.0
        );

    expect_near(result.delta_time, 1.0, 1.0e-12, "batch advance time");
    expect_true(result.resolved_event_count == 1, "one resolved event");
    expect_true(!result.blocked_by_conflicts(), "independent batch is not blocked");
    expect_near(particles[0].position.x, 1.1, 1.0e-12, "first contact position");
    expect_near(particles[1].position.x, 2.1, 1.0e-12, "second contact position");
    expect_near(particles[0].velocity.x, -0.1, 1.0e-12, "first outgoing velocity");
    expect_near(particles[1].velocity.x, 0.1, 1.0e-12, "second outgoing velocity");
    expect_vec_near(
        total_momentum(particles),
        momentum_before,
        1.0e-12,
        "batch momentum conservation"
    );
    expect_near(
        total_kinetic_energy(particles),
        energy_before,
        1.0e-12,
        "batch energy conservation"
    );
}

void test_advance_periodic_collision_batch() {
    const particles::DomainConfig domain = periodic_domain();
    const particles::UniformGrid grid =
        particles::make_uniform_grid(domain, 1.0);
    std::vector<particles::Particle> particles{
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
    const std::vector<particles::CandidatePair> candidates =
        particles::build_host_candidate_pairs(particles, grid);
    const std::vector<particles::CollisionEvent> events =
        particles::evaluate_host_collision_events(
            particles,
            candidates,
            grid,
            0.2,
            3.0
        );

    const particles::CollisionBatchAdvanceResult result =
        particles::advance_host_earliest_collision_batch(
            particles,
            events,
            grid,
            0.2
    );

    expect_true(events.size() == 1, "one periodic event with crosser");
    expect_near(result.delta_time, 2.0, 1.0e-11, "periodic batch time");
    expect_near(particles[0].position.x, 0.1, 1.0e-12, "periodic first position");
    expect_near(particles[1].position.x, 3.9, 1.0e-12, "periodic second position");
    expect_near(particles[0].velocity.x, 0.1, 1.0e-12, "periodic first velocity");
    expect_near(particles[1].velocity.x, -0.1, 1.0e-12, "periodic second velocity");
    expect_near(particles[2].position.x, 1.9, 1.0e-12, "periodic crosser wrapped");
}

void test_advance_batch_serializes_shared_particle_chain() {
    const particles::DomainConfig domain{
        .extent = {4.0, 4.0, 4.0},
        .boundary = {
            particles::BoundaryKind::periodic,
            particles::BoundaryKind::periodic,
            particles::BoundaryKind::periodic,
        },
    };
    const particles::UniformGrid grid =
        particles::make_uniform_grid(domain, 2.0);
    std::vector<particles::Particle> particles{
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
    const std::vector<particles::CandidatePair> candidates =
        particles::build_host_candidate_pairs(particles, grid);
    const std::vector<particles::CollisionEvent> events =
        particles::evaluate_host_collision_events(
            particles,
            candidates,
            grid,
            0.4,
            0.8
        );
    const particles::Vec3 momentum_before = total_momentum(particles);
    const double energy_before = total_kinetic_energy(particles);

    const particles::CollisionBatchAdvanceResult result =
        particles::advance_host_earliest_collision_batch(
            particles,
            events,
            grid,
            0.4
        );

    expect_true(result.resolved_event_count == 2, "disjoint and one chain event resolved");
    expect_true(
        result.serialized_conflict_event_count == 1,
        "one chain event serialized"
    );
    expect_true(
        result.deferred_conflicting_event_count == 0,
        "serialized chain leaves no stale deferred event"
    );
    expect_true(!result.blocked_by_conflicts(), "serialized chain can rebuild");
    expect_near(particles[0].velocity.x, 0.0, 1.0e-12, "chain first serialized");
    expect_near(particles[1].velocity.x, 1.0, 1.0e-12, "chain middle serialized");
    expect_near(particles[2].velocity.x, -1.0, 1.0e-12, "chain last unchanged");
    expect_near(particles[3].velocity.y, -0.5, 1.0e-12, "disjoint first resolved");
    expect_near(particles[4].velocity.y, 0.5, 1.0e-12, "disjoint second resolved");
    expect_vec_near(
        total_momentum(particles),
        momentum_before,
        1.0e-12,
        "mixed batch momentum conservation"
    );
    expect_near(
        total_kinetic_energy(particles),
        energy_before,
        1.0e-12,
        "mixed batch energy conservation"
    );
}

void test_conflict_serialization_ignores_input_order() {
    const particles::DomainConfig domain = periodic_domain();
    const particles::UniformGrid grid =
        particles::make_uniform_grid(domain, 2.0);
    std::vector<particles::Particle> particles{
        particles::Particle{
            .position = {0.5, 1.0, 1.0},
            .velocity = {1.0, 0.0, 0.0},
        },
        particles::Particle{
            .position = {1.5, 1.0, 1.0},
            .velocity = {0.0, 0.0, 0.0},
        },
        particles::Particle{
            .position = {2.5, 1.0, 1.0},
            .velocity = {-1.0, 0.0, 0.0},
        },
    };
    const std::vector<particles::CollisionEvent> reverse_order_events{
        particles::CollisionEvent{
            .first = 1,
            .second = 2,
            .time = 0.0,
        },
        particles::CollisionEvent{
            .first = 0,
            .second = 1,
            .time = 0.0,
        },
    };

    const particles::CollisionBatchAdvanceResult result =
        particles::advance_host_earliest_collision_batch(
            particles,
            reverse_order_events,
            grid,
            1.0
        );

    expect_true(
        result.serialized_conflict_event_count == 1,
        "one reverse-order conflict serialized"
    );
    expect_near(particles[0].velocity.x, 0.0, 1.0e-12, "minimum pair first");
    expect_near(particles[1].velocity.x, 1.0, 1.0e-12, "minimum pair second");
    expect_near(particles[2].velocity.x, -1.0, 1.0e-12, "larger pair deferred");
}

void test_reflecting_wall_corner_batch() {
    const particles::DomainConfig domain{
        .extent = {4.0, 4.0, 4.0},
        .boundary = {
            particles::BoundaryKind::reflecting,
            particles::BoundaryKind::reflecting,
            particles::BoundaryKind::reflecting,
        },
    };
    const particles::UniformGrid grid =
        particles::make_uniform_grid(domain, 2.0);
    std::vector<particles::Particle> particles{
        particles::Particle{
            .position = {3.0, 3.0, 2.0},
            .velocity = {1.0, 1.0, 0.0},
        },
    };

    const particles::ReflectingWallBatch batch =
        particles::find_host_reflecting_wall_batch(
            particles,
            grid,
            1.0,
            1.0
        );
    expect_true(batch.events.size() == 1, "one corner wall event");
    expect_true(batch.events[0].axis_mask == 0x03U, "corner axis mask");
    expect_near(batch.earliest_time, 0.5, 1.0e-12, "corner wall time");

    const particles::ReflectingWallAdvanceResult result =
        particles::advance_host_reflecting_wall_batch(
            particles,
            batch,
            grid,
            1.0
        );
    expect_true(result.reflected_particle_count == 1, "one reflected particle");
    expect_true(result.reflected_axis_count == 2, "two reflected axes");
    expect_near(particles[0].position.x, 3.5, 1.0e-12, "corner x position");
    expect_near(particles[0].position.y, 3.5, 1.0e-12, "corner y position");
    expect_near(particles[0].velocity.x, -1.0, 1.0e-12, "corner x velocity");
    expect_near(particles[0].velocity.y, -1.0, 1.0e-12, "corner y velocity");
}

void test_host_time_window_repeats_wall_batches() {
    const particles::DomainConfig domain{
        .extent = {4.0, 2.0, 2.0},
        .boundary = {
            particles::BoundaryKind::reflecting,
            particles::BoundaryKind::periodic,
            particles::BoundaryKind::periodic,
        },
    };
    const particles::UniformGrid grid =
        particles::make_uniform_grid(domain, 2.0);
    std::vector<particles::Particle> particles{
        particles::Particle{
            .position = {1.0, 1.0, 1.0},
            .velocity = {1.0, 0.0, 0.0},
        },
    };

    const particles::TimeWindowAdvanceResult result =
        particles::advance_host_time_window(
            particles,
            grid,
            6.0,
            1.0,
            100
        );

    expect_true(result.completed(), "wall time window completed");
    expect_near(result.advanced_time, 6.0, 1.0e-12, "wall window time");
    expect_true(
        result.reflected_particle_count == 2,
        "two wall batches reflected"
    );
    expect_true(result.reflected_axis_count == 2, "two wall axes reflected");
    expect_near(particles[0].position.x, 1.0, 1.0e-12, "wall round-trip position");
    expect_near(particles[0].velocity.x, 1.0, 1.0e-12, "wall round-trip velocity");
}

void test_host_time_window_rebuilds_after_collision() {
    const particles::DomainConfig domain = periodic_domain();
    const particles::UniformGrid grid =
        particles::make_uniform_grid(domain, 2.0);
    std::vector<particles::Particle> particles{
        particles::Particle{
            .position = {1.0, 1.0, 1.0},
            .velocity = {0.1, 0.0, 0.0},
        },
        particles::Particle{
            .position = {2.2, 1.0, 1.0},
            .velocity = {-0.1, 0.0, 0.0},
        },
    };
    const particles::Vec3 momentum_before = total_momentum(particles);
    const double energy_before = total_kinetic_energy(particles);

    const particles::TimeWindowAdvanceResult result =
        particles::advance_host_time_window(
            particles,
            grid,
            2.0,
            1.0,
            100
        );

    expect_true(result.completed(), "collision time window completed");
    expect_true(result.resolved_collision_count == 1, "one window collision");
    expect_true(
        result.candidate_rebuild_count >= 2,
        "events rebuilt after collision"
    );
    expect_near(particles[0].position.x, 1.0, 1.0e-12, "window first position");
    expect_near(particles[1].position.x, 2.2, 1.0e-12, "window second position");
    expect_near(particles[0].velocity.x, -0.1, 1.0e-12, "window first velocity");
    expect_near(particles[1].velocity.x, 0.1, 1.0e-12, "window second velocity");
    expect_vec_near(
        total_momentum(particles),
        momentum_before,
        1.0e-12,
        "window momentum conservation"
    );
    expect_near(
        total_kinetic_energy(particles),
        energy_before,
        1.0e-12,
        "window energy conservation"
    );
}

void test_host_time_window_resolves_collision_chain() {
    const particles::DomainConfig domain{
        .extent = {4.0, 4.0, 4.0},
        .boundary = {
            particles::BoundaryKind::periodic,
            particles::BoundaryKind::periodic,
            particles::BoundaryKind::periodic,
        },
    };
    const particles::UniformGrid grid =
        particles::make_uniform_grid(domain, 2.0);
    std::vector<particles::Particle> particles{
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
    };

    const particles::TimeWindowAdvanceResult result =
        particles::advance_host_time_window(
            particles,
            grid,
            1.0,
            0.4,
            10
        );

    expect_true(result.completed(), "collision chain window completes");
    expect_near(result.advanced_time, 1.0, 1.0e-12, "chain window time");
    expect_true(
        result.resolved_collision_count == 3,
        "three binary impulses resolve chain"
    );
    expect_true(
        result.serialized_conflict_resolution_count == 1,
        "one conflict round serialized"
    );
    expect_true(
        result.deferred_conflicting_event_count == 0,
        "chain leaves no deferred events"
    );
    expect_near(particles[0].velocity.x, -1.0, 1.0e-12, "chain first reversed");
    expect_near(particles[1].velocity.x, 0.0, 1.0e-12, "chain middle stopped");
    expect_near(particles[2].velocity.x, 1.0, 1.0e-12, "chain last reversed");
}

void test_host_time_window_reports_mixed_event_tie() {
    const particles::DomainConfig domain{
        .extent = {6.0, 4.0, 4.0},
        .boundary = {
            particles::BoundaryKind::reflecting,
            particles::BoundaryKind::periodic,
            particles::BoundaryKind::periodic,
        },
    };
    const particles::UniformGrid grid =
        particles::make_uniform_grid(domain, 2.0);
    std::vector<particles::Particle> particles{
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
    const particles::TimeWindowAdvanceResult result =
        particles::advance_host_time_window(
            particles,
            grid,
            2.0,
            1.0,
            10
        );

    expect_true(!result.completed(), "mixed tie blocks window");
    expect_true(
        result.stop_reason ==
            particles::TimeWindowStopReason::mixed_event_tie,
        "mixed tie stop reason"
    );
    expect_near(result.advanced_time, 1.0, 1.0e-12, "mixed tie stop time");
    expect_near(particles[0].position.x, 1.5, 1.0e-12, "mixed tie first contact");
    expect_near(particles[1].position.x, 2.5, 1.0e-12, "mixed tie second contact");
    expect_near(particles[2].position.x, 5.5, 1.0e-12, "mixed tie wall contact");
    expect_near(particles[0].velocity.x, 0.5, 1.0e-12, "mixed tie first velocity");
    expect_near(particles[1].velocity.x, -0.5, 1.0e-12, "mixed tie second velocity");
    expect_near(particles[2].velocity.x, 1.5, 1.0e-12, "mixed tie wall velocity");
}

}  // namespace

void run_collision_event_tests() {
    using Test = std::pair<std::string_view, std::function<void()>>;

    const std::vector<Test> tests{
        {"candidate direct collision time", test_direct_collision_time_from_candidates},
        {"candidate periodic collision time", test_periodic_image_collision_time},
        {"candidate diverging pair", test_diverging_candidate_has_no_event},
        {
            "conservative candidate horizon",
            test_conservative_horizon_uses_motion_and_wall_bounds,
        },
        {
            "earliest disjoint collision batch",
            test_earliest_batch_selects_disjoint_events,
        },
        {
            "earliest collision-chain conflict",
            test_earliest_batch_quarantines_collision_chain,
        },
        {"empty collision batch", test_empty_collision_batch},
        {
            "advance independent collision batch",
            test_advance_independent_collision_batch,
        },
        {
            "advance periodic collision batch",
            test_advance_periodic_collision_batch,
        },
        {
            "advance batch serializes collision chain",
            test_advance_batch_serializes_shared_particle_chain,
        },
        {
            "conflict serialization input order",
            test_conflict_serialization_ignores_input_order,
        },
        {
            "reflecting wall corner batch",
            test_reflecting_wall_corner_batch,
        },
        {
            "host repeated wall batches",
            test_host_time_window_repeats_wall_batches,
        },
        {
            "host collision-window rebuild",
            test_host_time_window_rebuilds_after_collision,
        },
        {
            "host collision-window chain resolution",
            test_host_time_window_resolves_collision_chain,
        },
        {
            "host mixed wall-collision tie",
            test_host_time_window_reports_mixed_event_tie,
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
        throw CollisionEventTestFailure(
            std::to_string(failures) + " collision-event test(s) failed"
        );
    }
}
