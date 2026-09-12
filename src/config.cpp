#include "particles/config.hpp"

#include <boost/json/src.hpp>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace particles {
namespace {

namespace json = boost::json;

[[nodiscard]] std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw ConfigError("Cannot open configuration file: " + path.string());
    }

    std::ostringstream contents;
    contents << stream.rdbuf();

    if (!stream.good() && !stream.eof()) {
        throw ConfigError("Failed while reading configuration file: " + path.string());
    }

    return contents.str();
}

[[nodiscard]] json::object parse_root(const std::filesystem::path& path) {
    json::error_code error;
    json::value document = json::parse(read_text_file(path), error);

    if (error) {
        throw ConfigError(
            "Invalid JSON in " + path.string() + ": " + error.message()
        );
    }
    if (!document.is_object()) {
        throw ConfigError("Configuration root must be a JSON object.");
    }

    return std::move(document).as_object();
}

[[nodiscard]] const json::value& required(
    const json::object& object,
    std::string_view key,
    std::string_view path
) {
    const auto found = object.find(key);
    if (found == object.end()) {
        throw ConfigError(
            "Missing required configuration field: " +
            std::string(path) + "." + std::string(key)
        );
    }
    return found->value();
}

[[nodiscard]] const json::object& required_object(
    const json::object& object,
    std::string_view key,
    std::string_view path
) {
    const json::value& value = required(object, key, path);
    if (!value.is_object()) {
        throw ConfigError(
            std::string(path) + "." + std::string(key) + " must be an object."
        );
    }
    return value.as_object();
}

[[nodiscard]] std::string required_string(
    const json::object& object,
    std::string_view key,
    std::string_view path
) {
    const json::value& value = required(object, key, path);
    if (!value.is_string()) {
        throw ConfigError(
            std::string(path) + "." + std::string(key) + " must be a string."
        );
    }

    const json::string& text = value.as_string();
    return {text.data(), text.size()};
}

[[nodiscard]] double required_number(
    const json::object& object,
    std::string_view key,
    std::string_view path
) {
    const json::value& value = required(object, key, path);
    if (!value.is_number()) {
        throw ConfigError(
            std::string(path) + "." + std::string(key) + " must be a number."
        );
    }

    const double number = value.to_number<double>();
    if (!std::isfinite(number)) {
        throw ConfigError(
            std::string(path) + "." + std::string(key) + " must be finite."
        );
    }
    return number;
}

[[nodiscard]] double optional_number(
    const json::object& object,
    std::string_view key,
    std::string_view path,
    double fallback
) {
    if (object.find(key) == object.end()) {
        return fallback;
    }
    return required_number(object, key, path);
}

[[nodiscard]] std::uint64_t required_uint64(
    const json::object& object,
    std::string_view key,
    std::string_view path
) {
    const json::value& value = required(object, key, path);

    if (value.is_uint64()) {
        return value.as_uint64();
    }
    if (value.is_int64() && value.as_int64() >= 0) {
        return static_cast<std::uint64_t>(value.as_int64());
    }

    throw ConfigError(
        std::string(path) + "." + std::string(key) +
        " must be a non-negative integer."
    );
}

[[nodiscard]] std::size_t required_size(
    const json::object& object,
    std::string_view key,
    std::string_view path
) {
    const std::uint64_t value = required_uint64(object, key, path);
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw ConfigError(
            std::string(path) + "." + std::string(key) +
            " does not fit into size_t."
        );
    }
    return static_cast<std::size_t>(value);
}

[[nodiscard]] Vec3 required_vec3(
    const json::object& object,
    std::string_view key,
    std::string_view path
) {
    const json::value& value = required(object, key, path);
    if (!value.is_array() || value.as_array().size() != 3) {
        throw ConfigError(
            std::string(path) + "." + std::string(key) +
            " must be an array of three numbers."
        );
    }

    Vec3 result{};
    double* components[3]{&result.x, &result.y, &result.z};
    const json::array& array = value.as_array();

    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (!array[axis].is_number()) {
            throw ConfigError(
                std::string(path) + "." + std::string(key) +
                " must contain only numbers."
            );
        }
        *components[axis] = array[axis].to_number<double>();
        if (!std::isfinite(*components[axis])) {
            throw ConfigError(
                std::string(path) + "." + std::string(key) +
                " must contain only finite numbers."
            );
        }
    }

    return result;
}

[[nodiscard]] BoundaryKind parse_boundary(
    const json::object& object,
    std::string_view key,
    std::string_view path
) {
    const std::string value = required_string(object, key, path);
    if (value == "periodic") {
        return BoundaryKind::periodic;
    }
    if (value == "reflecting") {
        return BoundaryKind::reflecting;
    }

    throw ConfigError(
        std::string(path) + "." + std::string(key) +
        " must be \"periodic\" or \"reflecting\"."
    );
}

[[nodiscard]] DomainConfig parse_domain(const json::object& root) {
    const json::object& domain = required_object(root, "domain", "root");
    const json::object& boundary =
        required_object(domain, "boundary", "root.domain");

    return DomainConfig{
        .extent = required_vec3(domain, "extent", "root.domain"),
        .boundary = {
            parse_boundary(boundary, "x", "root.domain.boundary"),
            parse_boundary(boundary, "y", "root.domain.boundary"),
            parse_boundary(boundary, "z", "root.domain.boundary"),
        },
    };
}

[[nodiscard]] WindowConvergenceConfig parse_window_convergence(
    const json::object& root
) {
    const auto found = root.find("window_convergence");
    if (found == root.end()) {
        return {};
    }
    if (!found->value().is_object()) {
        throw ConfigError(
            "root.window_convergence must be an object."
        );
    }
    const json::object& window = found->value().as_object();
    return WindowConvergenceConfig{
        .duration = required_number(
            window,
            "duration",
            "root.window_convergence"
        ),
        .maximum_window = required_number(
            window,
            "maximum_window",
            "root.window_convergence"
        ),
        .refinement_levels = required_size(
            window,
            "refinement_levels",
            "root.window_convergence"
        ),
        .maximum_steps = required_size(
            window,
            "maximum_steps",
            "root.window_convergence"
        ),
    };
}

[[nodiscard]] AdaptiveWindowConfig parse_adaptive_window(
    const json::object& root
) {
    const json::object& adaptive =
        required_object(root, "adaptive_window", "root");
    return AdaptiveWindowConfig{
        .minimum_window = required_number(
            adaptive,
            "minimum_window",
            "root.adaptive_window"
        ),
        .maximum_attempts = required_size(
            adaptive,
            "maximum_attempts",
            "root.adaptive_window"
        ),
        .successful_steps_before_growth = required_size(
            adaptive,
            "successful_steps_before_growth",
            "root.adaptive_window"
        ),
        .growth_factor = required_number(
            adaptive,
            "growth_factor",
            "root.adaptive_window"
        ),
    };
}

void validate_domain(const DomainConfig& domain, double diameter) {
    const double extents[3]{
        domain.extent.x,
        domain.extent.y,
        domain.extent.z,
    };

    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(extents[axis]) || extents[axis] <= diameter) {
            throw ConfigError(
                "Every domain extent must be finite and exceed the particle diameter."
            );
        }
        if (
            domain.boundary[axis] == BoundaryKind::periodic &&
            extents[axis] <= 2.0 * diameter
        ) {
            throw ConfigError(
                "Every periodic extent must exceed two particle diameters."
            );
        }
    }
}

void validate_units(double mass, double boltzmann_constant) {
    if (!std::isfinite(mass) || mass <= 0.0) {
        throw ConfigError("Particle mass must be finite and positive.");
    }
    if (!std::isfinite(boltzmann_constant) || boltzmann_constant <= 0.0) {
        throw ConfigError("Boltzmann constant must be finite and positive.");
    }
}

void validate_window_convergence(
    const WindowConvergenceConfig& config
) {
    if (!std::isfinite(config.duration) || config.duration < 0.0) {
        throw ConfigError(
            "Window-convergence duration must be finite and non-negative."
        );
    }
    if (
        !std::isfinite(config.maximum_window) ||
        config.maximum_window <= 0.0
    ) {
        throw ConfigError(
            "Window-convergence maximum window must be finite and positive."
        );
    }
    if (
        config.refinement_levels < 3 ||
        config.refinement_levels > 64
    ) {
        throw ConfigError(
            "Window convergence requires between 3 and 64 refinement levels."
        );
    }
    if (config.maximum_steps == 0) {
        throw ConfigError(
            "Window-convergence step limit must be positive."
        );
    }
}

void validate_adaptive_window(
    const AdaptiveWindowConfig& config,
    double maximum_window
) {
    if (
        !std::isfinite(config.minimum_window) ||
        config.minimum_window <= 0.0 ||
        config.minimum_window > maximum_window
    ) {
        throw ConfigError(
            "Adaptive minimum window must be positive and no larger than "
            "the experiment maximum window."
        );
    }
    if (config.maximum_attempts == 0) {
        throw ConfigError(
            "Adaptive parallel-window attempt limit must be positive."
        );
    }
    if (config.successful_steps_before_growth == 0) {
        throw ConfigError(
            "Adaptive window growth requires at least one successful step."
        );
    }
    if (
        !std::isfinite(config.growth_factor) ||
        config.growth_factor <= 1.0
    ) {
        throw ConfigError(
            "Adaptive window growth factor must be finite and exceed one."
        );
    }
}

void expect_experiment(const json::object& root, std::string_view expected) {
    const std::string experiment = required_string(root, "experiment", "root");
    if (experiment != expected) {
        throw ConfigError(
            "Expected experiment \"" + std::string(expected) +
            "\", but configuration contains \"" + experiment + "\"."
        );
    }
}

}  // namespace

void validate(const ShockTubeConfig& config) {
    if (config.schema_version != 1) {
        throw ConfigError("Unsupported shock-tube schema version.");
    }
    validate_units(config.particle_mass, config.boltzmann_constant);

    if (
        !std::isfinite(config.reference_temperature) ||
        config.reference_temperature <= 0.0
    ) {
        throw ConfigError("Reference temperature must be finite and positive.");
    }
    if (config.particle_count == 0) {
        throw ConfigError("Particle count must be positive.");
    }
    if (
        !std::isfinite(config.particle_diameter) ||
        config.particle_diameter <= 0.0
    ) {
        throw ConfigError("Particle diameter must be finite and positive.");
    }

    validate_domain(config.domain, config.particle_diameter);

    if (
        !std::isfinite(config.interface_x) ||
        config.interface_x <= 0.0 ||
        config.interface_x >= config.domain.extent.x
    ) {
        throw ConfigError("Shock-tube interface must lie inside the x domain.");
    }
    if (
        !std::isfinite(config.transition_width) ||
        config.transition_width <= 0.0
    ) {
        throw ConfigError("Shock-tube transition width must be finite and positive.");
    }

    const MacroscopicState states[2]{config.left, config.right};
    for (const MacroscopicState& state : states) {
        if (
            !std::isfinite(state.relative_number_density) ||
            state.relative_number_density <= 0.0
        ) {
            throw ConfigError("Relative number densities must be finite and positive.");
        }
        if (!std::isfinite(state.temperature) || state.temperature <= 0.0) {
            throw ConfigError("Temperatures must be finite and positive.");
        }
        if (
            !std::isfinite(state.bulk_velocity.x) ||
            !std::isfinite(state.bulk_velocity.y) ||
            !std::isfinite(state.bulk_velocity.z)
        ) {
            throw ConfigError("Bulk velocities must be finite.");
        }
    }

    if (
        config.diagnostics.microscopic_particle_samples == 0 ||
        config.diagnostics.phase_space_x_bins == 0 ||
        config.diagnostics.phase_space_vx_bins == 0 ||
        config.diagnostics.macroscopic_x_slabs == 0
    ) {
        throw ConfigError("All diagnostic bin counts must be positive.");
    }
    if (
        !std::isfinite(
            config.diagnostics.phase_space_velocity_x_min
        ) ||
        !std::isfinite(
            config.diagnostics.phase_space_velocity_x_max
        ) ||
        config.diagnostics.phase_space_velocity_x_max <=
            config.diagnostics.phase_space_velocity_x_min
    ) {
        throw ConfigError(
            "Phase-space velocity maximum must exceed its finite minimum."
        );
    }
    if (
        !std::isfinite(config.evolution.end_time) ||
        config.evolution.end_time < 0.0
    ) {
        throw ConfigError(
            "Shock-tube end time must be finite and non-negative."
        );
    }
    if (
        !std::isfinite(config.evolution.frame_interval) ||
        config.evolution.frame_interval <= 0.0
    ) {
        throw ConfigError(
            "Shock-tube frame interval must be finite and positive."
        );
    }
    if (
        !std::isfinite(config.evolution.cell_width_factor) ||
        config.evolution.cell_width_factor <= 1.0
    ) {
        throw ConfigError(
            "Shock-tube cell-width factor must be finite and exceed one."
        );
    }
    if (
        !std::isfinite(config.evolution.maximum_window) ||
        config.evolution.maximum_window <= 0.0
    ) {
        throw ConfigError(
            "Shock-tube maximum window must be finite and positive."
        );
    }
    if (config.evolution.exact_recovery_batch_limit == 0) {
        throw ConfigError(
            "Shock-tube exact-recovery batch limit must be positive."
        );
    }
    validate_adaptive_window(
        config.evolution.adaptive_window,
        config.evolution.maximum_window
    );
}

void validate(const EquilibriumConfig& config) {
    if (config.schema_version != 1) {
        throw ConfigError("Unsupported equilibrium schema version.");
    }
    validate_units(config.particle_mass, config.boltzmann_constant);

    if (config.particle_count == 0) {
        throw ConfigError("Particle count must be positive.");
    }
    if (
        !std::isfinite(config.particle_diameter) ||
        config.particle_diameter <= 0.0
    ) {
        throw ConfigError("Particle diameter must be finite and positive.");
    }

    validate_domain(config.domain, config.particle_diameter);

    if (!std::isfinite(config.temperature) || config.temperature <= 0.0) {
        throw ConfigError("Equilibrium temperature must be finite and positive.");
    }
    if (
        !std::isfinite(config.bulk_velocity.x) ||
        !std::isfinite(config.bulk_velocity.y) ||
        !std::isfinite(config.bulk_velocity.z)
    ) {
        throw ConfigError("Bulk velocity must be finite.");
    }
    if (!std::isfinite(config.end_time) || config.end_time < 0.0) {
        throw ConfigError("Equilibrium end time must be finite and non-negative.");
    }
    if (
        !std::isfinite(config.diagnostics.sample_interval) ||
        config.diagnostics.sample_interval <= 0.0
    ) {
        throw ConfigError(
            "Equilibrium diagnostic sample interval must be finite and positive."
        );
    }
    if (config.diagnostics.velocity_x_bins == 0) {
        throw ConfigError("Velocity histogram bin count must be positive.");
    }
    if (
        !std::isfinite(config.diagnostics.velocity_x_min) ||
        !std::isfinite(config.diagnostics.velocity_x_max) ||
        config.diagnostics.velocity_x_max <=
            config.diagnostics.velocity_x_min
    ) {
        throw ConfigError(
            "Velocity histogram maximum must exceed its finite minimum."
        );
    }
    validate_window_convergence(config.window_convergence);
}

void validate(const ParallelWindowScenarioConfig& config) {
    if (config.schema_version != 1) {
        throw ConfigError(
            "Unsupported parallel-window scenario schema version."
        );
    }
    if (config.scenario.empty()) {
        throw ConfigError(
            "Parallel-window scenario name must not be empty."
        );
    }
    if (
        !std::isfinite(config.particle_mass) ||
        config.particle_mass <= 0.0
    ) {
        throw ConfigError(
            "Scenario particle mass must be finite and positive."
        );
    }
    if (
        !std::isfinite(config.particle_diameter) ||
        config.particle_diameter <= 0.0
    ) {
        throw ConfigError(
            "Scenario particle diameter must be finite and positive."
        );
    }
    validate_domain(config.domain, config.particle_diameter);
    validate_window_convergence(config.window_convergence);

    for (const BoundaryKind boundary : config.domain.boundary) {
        if (boundary != BoundaryKind::periodic) {
            throw ConfigError(
                "Parallel-window scenarios currently require periodic "
                "boundaries on all axes."
            );
        }
    }
    if (config.particles.empty()) {
        throw ConfigError(
            "Parallel-window scenario must contain at least one particle."
        );
    }
    if (
        config.particles.size() >
        static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max()
        )
    ) {
        throw ConfigError(
            "Parallel-window scenario contains too many particles."
        );
    }

    const double extents[3]{
        config.domain.extent.x,
        config.domain.extent.y,
        config.domain.extent.z,
    };
    for (const ParticleStateConfig& particle : config.particles) {
        const double positions[3]{
            particle.position.x,
            particle.position.y,
            particle.position.z,
        };
        const double velocities[3]{
            particle.velocity.x,
            particle.velocity.y,
            particle.velocity.z,
        };
        for (std::size_t axis = 0; axis < 3; ++axis) {
            if (
                !std::isfinite(positions[axis]) ||
                positions[axis] < 0.0 ||
                positions[axis] >= extents[axis]
            ) {
                throw ConfigError(
                    "Scenario particle positions must lie in the periodic "
                    "domain."
                );
            }
            if (!std::isfinite(velocities[axis])) {
                throw ConfigError(
                    "Scenario particle velocities must be finite."
                );
            }
        }
    }

    const double diameter_squared =
        config.particle_diameter * config.particle_diameter;
    for (std::size_t first = 0;
         first < config.particles.size();
         ++first) {
        for (std::size_t second = first + 1;
             second < config.particles.size();
             ++second) {
            Vec3 displacement =
                config.particles[first].position -
                config.particles[second].position;
            double* values[3]{
                &displacement.x,
                &displacement.y,
                &displacement.z,
            };
            for (std::size_t axis = 0; axis < 3; ++axis) {
                *values[axis] -=
                    std::round(*values[axis] / extents[axis]) *
                    extents[axis];
            }
            if (squared_norm(displacement) < diameter_squared) {
                throw ConfigError(
                    "Parallel-window scenario contains overlapping "
                    "particles."
                );
            }
        }
    }
}

void validate(const ParallelWindowStressConfig& config) {
    if (config.schema_version != 1) {
        throw ConfigError(
            "Unsupported parallel-window stress schema version."
        );
    }
    validate_units(
        config.particle_mass,
        config.boltzmann_constant
    );
    if (config.seed_count == 0 || config.seed_count > 1'000) {
        throw ConfigError(
            "Parallel-window stress seed count must be between 1 and 1000."
        );
    }
    if (
        config.seed_count - 1 >
        std::numeric_limits<std::uint64_t>::max() -
            config.base_random_seed
    ) {
        throw ConfigError(
            "Parallel-window stress seed range exceeds uint64_t."
        );
    }
    if (config.particle_count == 0) {
        throw ConfigError(
            "Parallel-window stress particle count must be positive."
        );
    }
    if (
        !std::isfinite(config.particle_diameter) ||
        config.particle_diameter <= 0.0
    ) {
        throw ConfigError(
            "Stress particle diameter must be finite and positive."
        );
    }
    validate_domain(config.domain, config.particle_diameter);
    for (const BoundaryKind boundary : config.domain.boundary) {
        if (boundary != BoundaryKind::periodic) {
            throw ConfigError(
                "Parallel-window stress experiments currently require "
                "periodic boundaries on all axes."
            );
        }
    }
    if (!std::isfinite(config.temperature) || config.temperature <= 0.0) {
        throw ConfigError(
            "Stress temperature must be finite and positive."
        );
    }
    if (
        !std::isfinite(config.bulk_velocity.x) ||
        !std::isfinite(config.bulk_velocity.y) ||
        !std::isfinite(config.bulk_velocity.z)
    ) {
        throw ConfigError(
            "Stress bulk velocity must be finite."
        );
    }
    validate_window_convergence(config.window_convergence);
    validate_adaptive_window(
        config.adaptive_window,
        config.window_convergence.maximum_window
    );
}

void validate(const CudaScalingConfig& config) {
    if (config.schema_version != 1) {
        throw ConfigError(
            "Unsupported CUDA-scaling schema version."
        );
    }
    validate_units(config.particle_mass, config.boltzmann_constant);
    if (config.base_particle_count == 0) {
        throw ConfigError(
            "CUDA-scaling base particle count must be positive."
        );
    }
    if (config.level_count == 0 || config.level_count > 31) {
        throw ConfigError(
            "CUDA scaling requires between 1 and 31 levels."
        );
    }
    std::size_t maximum_particle_count =
        config.base_particle_count;
    for (std::size_t level = 1;
         level < config.level_count;
         ++level) {
        if (
            maximum_particle_count >
            static_cast<std::size_t>(
                std::numeric_limits<std::uint32_t>::max()
            ) / 2U
        ) {
            throw ConfigError(
                "CUDA-scaling particle levels exceed uint32_t."
            );
        }
        maximum_particle_count *= 2U;
    }
    if (
        maximum_particle_count >=
        static_cast<std::size_t>(
            std::numeric_limits<std::int32_t>::max()
        )
    ) {
        throw ConfigError(
            "CUDA scaling requires fewer than INT_MAX particles."
        );
    }
    if (
        !std::isfinite(config.number_density) ||
        config.number_density <= 0.0
    ) {
        throw ConfigError(
            "CUDA-scaling number density must be finite and positive."
        );
    }
    if (
        !std::isfinite(config.particle_diameter) ||
        config.particle_diameter <= 0.0
    ) {
        throw ConfigError(
            "CUDA-scaling diameter must be finite and positive."
        );
    }
    const double smallest_extent = std::cbrt(
        static_cast<double>(config.base_particle_count) /
        config.number_density
    );
    if (
        !std::isfinite(smallest_extent) ||
        smallest_extent <= 2.0 * config.particle_diameter
    ) {
        throw ConfigError(
            "CUDA-scaling base domain must exceed two diameters."
        );
    }
    if (!std::isfinite(config.temperature) || config.temperature <= 0.0) {
        throw ConfigError(
            "CUDA-scaling temperature must be finite and positive."
        );
    }
    if (
        !std::isfinite(config.bulk_velocity.x) ||
        !std::isfinite(config.bulk_velocity.y) ||
        !std::isfinite(config.bulk_velocity.z)
    ) {
        throw ConfigError(
            "CUDA-scaling bulk velocity must be finite."
        );
    }
    if (
        !std::isfinite(config.cell_width_factor) ||
        config.cell_width_factor <= 1.0
    ) {
        throw ConfigError(
            "CUDA-scaling cell-width factor must exceed one."
        );
    }
    if (
        !std::isfinite(config.warmup_duration) ||
        config.warmup_duration < 0.0
    ) {
        throw ConfigError(
            "CUDA-scaling warmup duration must be finite and non-negative."
        );
    }
    if (
        !std::isfinite(config.measured_duration) ||
        config.measured_duration <= 0.0
    ) {
        throw ConfigError(
            "CUDA-scaling measured duration must be finite and positive."
        );
    }
    if (
        !std::isfinite(config.maximum_window) ||
        config.maximum_window <= 0.0
    ) {
        throw ConfigError(
            "CUDA-scaling maximum window must be finite and positive."
        );
    }
    validate_adaptive_window(
        config.adaptive_window,
        config.maximum_window
    );
    if (config.maximum_placement_attempts_per_particle == 0) {
        throw ConfigError(
            "CUDA-scaling placement-attempt limit must be positive."
        );
    }
}

ShockTubeConfig load_shock_tube_config(const std::filesystem::path& path) {
    const json::object root = parse_root(path);
    expect_experiment(root, "shock_tube");

    const json::object& units = required_object(root, "units", "root");
    const json::object& particle = required_object(root, "particles", "root");
    const json::object& initial =
        required_object(root, "initial_condition", "root");
    const json::object& left =
        required_object(initial, "left", "root.initial_condition");
    const json::object& right =
        required_object(initial, "right", "root.initial_condition");
    const json::object& diagnostics =
        required_object(root, "diagnostics", "root");
    const json::object& simulation =
        required_object(root, "simulation", "root");

    ShockTubeConfig config{
        .schema_version = static_cast<std::uint32_t>(
            required_uint64(root, "schema_version", "root")
        ),
        .random_seed = required_uint64(root, "random_seed", "root"),
        .particle_mass = required_number(units, "particle_mass", "root.units"),
        .boltzmann_constant =
            required_number(units, "boltzmann_constant", "root.units"),
        .reference_temperature =
            required_number(units, "reference_temperature", "root.units"),
        .domain = parse_domain(root),
        .particle_count = required_size(particle, "count", "root.particles"),
        .particle_diameter =
            required_number(particle, "diameter", "root.particles"),
        .interface_x =
            required_number(initial, "interface_x", "root.initial_condition"),
        .transition_width =
            required_number(initial, "transition_width", "root.initial_condition"),
        .left = MacroscopicState{
            .relative_number_density = required_number(
                left,
                "relative_number_density",
                "root.initial_condition.left"
            ),
            .temperature = required_number(
                left,
                "temperature",
                "root.initial_condition.left"
            ),
            .bulk_velocity = required_vec3(
                left,
                "bulk_velocity",
                "root.initial_condition.left"
            ),
        },
        .right = MacroscopicState{
            .relative_number_density = required_number(
                right,
                "relative_number_density",
                "root.initial_condition.right"
            ),
            .temperature = required_number(
                right,
                "temperature",
                "root.initial_condition.right"
            ),
            .bulk_velocity = required_vec3(
                right,
                "bulk_velocity",
                "root.initial_condition.right"
            ),
        },
        .transverse_shear_amplitude = optional_number(
            initial,
            "transverse_shear_amplitude",
            "root.initial_condition",
            0.0
        ),
        .transverse_shear_mode = static_cast<std::size_t>(
            optional_number(
                initial,
                "transverse_shear_mode",
                "root.initial_condition",
                1.0
            )
        ),
        .diagnostics = DiagnosticsConfig{
            .microscopic_particle_samples = required_size(
                diagnostics,
                "microscopic_particle_samples",
                "root.diagnostics"
            ),
            .phase_space_x_bins = required_size(
                diagnostics,
                "phase_space_x_bins",
                "root.diagnostics"
            ),
            .phase_space_vx_bins = required_size(
                diagnostics,
                "phase_space_vx_bins",
                "root.diagnostics"
            ),
            .phase_space_velocity_x_min = required_number(
                diagnostics,
                "phase_space_velocity_x_min",
                "root.diagnostics"
            ),
            .phase_space_velocity_x_max = required_number(
                diagnostics,
                "phase_space_velocity_x_max",
                "root.diagnostics"
            ),
            .macroscopic_x_slabs = required_size(
                diagnostics,
                "macroscopic_x_slabs",
                "root.diagnostics"
            ),
        },
        .evolution = ShockTubeEvolutionConfig{
            .end_time = required_number(
                simulation,
                "end_time",
                "root.simulation"
            ),
            .frame_interval = required_number(
                simulation,
                "frame_interval",
                "root.simulation"
            ),
            .cell_width_factor = required_number(
                simulation,
                "cell_width_factor",
                "root.simulation"
            ),
            .maximum_window = required_number(
                simulation,
                "maximum_window",
                "root.simulation"
            ),
            .exact_recovery_batch_limit = required_size(
                simulation,
                "exact_recovery_batch_limit",
                "root.simulation"
            ),
            .adaptive_window = parse_adaptive_window(root),
        },
    };

    validate(config);
    return config;
}

EquilibriumConfig load_equilibrium_config(const std::filesystem::path& path) {
    const json::object root = parse_root(path);
    expect_experiment(root, "equilibrium");

    const json::object& units = required_object(root, "units", "root");
    const json::object& particle = required_object(root, "particles", "root");
    const json::object& initial =
        required_object(root, "initial_condition", "root");
    const json::object& run = required_object(root, "run", "root");
    const json::object& diagnostics =
        required_object(root, "diagnostics", "root");

    EquilibriumConfig config{
        .schema_version = static_cast<std::uint32_t>(
            required_uint64(root, "schema_version", "root")
        ),
        .random_seed = required_uint64(root, "random_seed", "root"),
        .particle_mass = required_number(units, "particle_mass", "root.units"),
        .boltzmann_constant =
            required_number(units, "boltzmann_constant", "root.units"),
        .domain = parse_domain(root),
        .particle_count = required_size(particle, "count", "root.particles"),
        .particle_diameter =
            required_number(particle, "diameter", "root.particles"),
        .temperature =
            required_number(initial, "temperature", "root.initial_condition"),
        .bulk_velocity =
            required_vec3(initial, "bulk_velocity", "root.initial_condition"),
        .end_time = required_number(run, "end_time", "root.run"),
        .diagnostics = EquilibriumDiagnosticsConfig{
            .sample_interval = required_number(
                diagnostics,
                "sample_interval",
                "root.diagnostics"
            ),
            .velocity_x_bins = required_size(
                diagnostics,
                "velocity_x_bins",
                "root.diagnostics"
            ),
            .velocity_x_min = required_number(
                diagnostics,
                "velocity_x_min",
                "root.diagnostics"
            ),
            .velocity_x_max = required_number(
                diagnostics,
                "velocity_x_max",
                "root.diagnostics"
            ),
        },
        .window_convergence = parse_window_convergence(root),
    };

    validate(config);
    return config;
}

ParallelWindowScenarioConfig load_parallel_window_scenario_config(
    const std::filesystem::path& path
) {
    const json::object root = parse_root(path);
    expect_experiment(root, "parallel_window_scenario");

    const json::object& units = required_object(root, "units", "root");
    const json::object& particle =
        required_object(root, "particles", "root");
    const json::value& states_value = required(
        particle,
        "states",
        "root.particles"
    );
    if (!states_value.is_array()) {
        throw ConfigError(
            "root.particles.states must be an array."
        );
    }

    std::vector<ParticleStateConfig> states;
    const json::array& state_array = states_value.as_array();
    states.reserve(state_array.size());
    for (std::size_t index = 0; index < state_array.size(); ++index) {
        if (!state_array[index].is_object()) {
            throw ConfigError(
                "Every root.particles.states entry must be an object."
            );
        }
        const json::object& state = state_array[index].as_object();
        const std::string state_path =
            "root.particles.states[" + std::to_string(index) + "]";
        states.push_back(ParticleStateConfig{
            .position = required_vec3(
                state,
                "position",
                state_path
            ),
            .velocity = required_vec3(
                state,
                "velocity",
                state_path
            ),
        });
    }

    ParallelWindowScenarioConfig config{
        .schema_version = static_cast<std::uint32_t>(
            required_uint64(root, "schema_version", "root")
        ),
        .scenario = required_string(root, "scenario", "root"),
        .particle_mass = required_number(
            units,
            "particle_mass",
            "root.units"
        ),
        .domain = parse_domain(root),
        .particle_diameter = required_number(
            particle,
            "diameter",
            "root.particles"
        ),
        .particles = std::move(states),
        .window_convergence = parse_window_convergence(root),
    };

    validate(config);
    return config;
}

ParallelWindowStressConfig load_parallel_window_stress_config(
    const std::filesystem::path& path
) {
    const json::object root = parse_root(path);
    expect_experiment(root, "parallel_window_stress");

    const json::object& units = required_object(root, "units", "root");
    const json::object& particle =
        required_object(root, "particles", "root");
    const json::object& initial =
        required_object(root, "initial_condition", "root");

    ParallelWindowStressConfig config{
        .schema_version = static_cast<std::uint32_t>(
            required_uint64(root, "schema_version", "root")
        ),
        .base_random_seed =
            required_uint64(root, "random_seed", "root"),
        .seed_count = required_size(root, "seed_count", "root"),
        .particle_mass = required_number(
            units,
            "particle_mass",
            "root.units"
        ),
        .boltzmann_constant = required_number(
            units,
            "boltzmann_constant",
            "root.units"
        ),
        .domain = parse_domain(root),
        .particle_count = required_size(
            particle,
            "count",
            "root.particles"
        ),
        .particle_diameter = required_number(
            particle,
            "diameter",
            "root.particles"
        ),
        .temperature = required_number(
            initial,
            "temperature",
            "root.initial_condition"
        ),
        .bulk_velocity = required_vec3(
            initial,
            "bulk_velocity",
            "root.initial_condition"
        ),
        .window_convergence = parse_window_convergence(root),
        .adaptive_window = parse_adaptive_window(root),
    };

    validate(config);
    return config;
}

CudaScalingConfig load_cuda_scaling_config(
    const std::filesystem::path& path
) {
    const json::object root = parse_root(path);
    expect_experiment(root, "cuda_scaling");

    const json::object& units = required_object(root, "units", "root");
    const json::object& particle =
        required_object(root, "particles", "root");
    const json::object& initial =
        required_object(root, "initial_condition", "root");
    const json::object& grid =
        required_object(root, "grid", "root");
    const json::object& benchmark =
        required_object(root, "benchmark", "root");

    CudaScalingConfig config{
        .schema_version = static_cast<std::uint32_t>(
            required_uint64(root, "schema_version", "root")
        ),
        .random_seed =
            required_uint64(root, "random_seed", "root"),
        .particle_mass = required_number(
            units,
            "particle_mass",
            "root.units"
        ),
        .boltzmann_constant = required_number(
            units,
            "boltzmann_constant",
            "root.units"
        ),
        .base_particle_count = required_size(
            particle,
            "base_count",
            "root.particles"
        ),
        .level_count = required_size(
            particle,
            "level_count",
            "root.particles"
        ),
        .number_density = required_number(
            particle,
            "number_density",
            "root.particles"
        ),
        .particle_diameter = required_number(
            particle,
            "diameter",
            "root.particles"
        ),
        .temperature = required_number(
            initial,
            "temperature",
            "root.initial_condition"
        ),
        .bulk_velocity = required_vec3(
            initial,
            "bulk_velocity",
            "root.initial_condition"
        ),
        .cell_width_factor = required_number(
            grid,
            "cell_width_factor",
            "root.grid"
        ),
        .warmup_duration = required_number(
            benchmark,
            "warmup_duration",
            "root.benchmark"
        ),
        .measured_duration = required_number(
            benchmark,
            "measured_duration",
            "root.benchmark"
        ),
        .maximum_window = required_number(
            benchmark,
            "maximum_window",
            "root.benchmark"
        ),
        .adaptive_window = parse_adaptive_window(root),
        .maximum_placement_attempts_per_particle = required_size(
            initial,
            "maximum_placement_attempts_per_particle",
            "root.initial_condition"
        ),
    };

    validate(config);
    return config;
}

}  // namespace particles
