#include "particles/config.hpp"

#include <boost/json/src.hpp>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>

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
        config.diagnostics.phase_space_x_bins == 0 ||
        config.diagnostics.phase_space_vx_bins == 0 ||
        config.diagnostics.macroscopic_x_slabs == 0
    ) {
        throw ConfigError("All diagnostic bin counts must be positive.");
    }
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
    if (
        !std::isfinite(config.window_convergence.duration) ||
        config.window_convergence.duration < 0.0
    ) {
        throw ConfigError(
            "Window-convergence duration must be finite and non-negative."
        );
    }
    if (
        !std::isfinite(config.window_convergence.maximum_window) ||
        config.window_convergence.maximum_window <= 0.0
    ) {
        throw ConfigError(
            "Window-convergence maximum window must be finite and positive."
        );
    }
    if (
        config.window_convergence.refinement_levels < 3 ||
        config.window_convergence.refinement_levels > 64
    ) {
        throw ConfigError(
            "Window convergence requires between 3 and 64 refinement levels."
        );
    }
    if (config.window_convergence.maximum_steps == 0) {
        throw ConfigError(
            "Window-convergence step limit must be positive."
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
        .diagnostics = DiagnosticsConfig{
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
            .macroscopic_x_slabs = required_size(
                diagnostics,
                "macroscopic_x_slabs",
                "root.diagnostics"
            ),
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

}  // namespace particles
