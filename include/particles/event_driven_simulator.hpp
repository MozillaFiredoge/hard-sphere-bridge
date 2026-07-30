#pragma once

#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

#include "particles/config.hpp"
#include "particles/hard_sphere.hpp"

namespace particles {

enum class EventKind {
    particle_collision,
    reflecting_wall,
    periodic_wrap,
};

inline constexpr std::size_t no_particle =
    std::numeric_limits<std::size_t>::max();
inline constexpr std::size_t no_axis =
    std::numeric_limits<std::size_t>::max();

struct SimulationEvent {
    EventKind kind{EventKind::particle_collision};
    double time{};
    double delta_time{};
    std::size_t first{no_particle};
    std::size_t second{no_particle};
    std::size_t axis{no_axis};
    Vec3 periodic_image_shift{};
};

struct EventStatistics {
    std::size_t particle_collisions{};
    std::size_t reflecting_wall_collisions{};
    std::size_t periodic_wraps{};

    [[nodiscard]] constexpr std::size_t total() const noexcept {
        return particle_collisions +
               reflecting_wall_collisions +
               periodic_wraps;
    }
};

// A deliberately simple O(N^2) event-driven simulator.
//
// This implementation is the small-scale correctness reference for future
// parallel algorithms. It recomputes all candidate events after every event,
// favoring deterministic behavior and inspectability over throughput.
class EventDrivenSimulator {
public:
    EventDrivenSimulator(
        std::vector<Particle> particles,
        double diameter,
        DomainConfig domain,
        double tolerance = 1.0e-12
    );

    [[nodiscard]] const std::vector<Particle>& particles() const noexcept {
        return particles_;
    }

    [[nodiscard]] double time() const noexcept {
        return time_;
    }

    [[nodiscard]] double diameter() const noexcept {
        return diameter_;
    }

    [[nodiscard]] const DomainConfig& domain() const noexcept {
        return domain_;
    }

    [[nodiscard]] const EventStatistics& statistics() const noexcept {
        return statistics_;
    }

    // Advances to and resolves the earliest event in [time, time + horizon].
    // If no event occurs, advances by the full horizon and returns nullopt.
    [[nodiscard]] std::optional<SimulationEvent> advance_to_next_event(
        double horizon
    );

    // Advances until end_time, resolving all events along the way.
    // The event limit protects tests and diagnostics from zero-time loops.
    void run_until(
        double end_time,
        std::size_t maximum_events = 1'000'000
    );

private:
    struct EventCandidate;

    [[nodiscard]] std::optional<EventCandidate> find_next_event(
        double horizon
    ) const;
    void advance_all(double delta_time);
    void resolve(const EventCandidate& candidate);
    void validate_initial_state() const;

    std::vector<Particle> particles_;
    double diameter_{};
    DomainConfig domain_{};
    double tolerance_{};
    double time_{};
    EventStatistics statistics_{};
};

}  // namespace particles

