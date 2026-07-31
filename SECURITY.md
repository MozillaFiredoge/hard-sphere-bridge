# Security Policy

## Supported versions

Security fixes are applied to the latest tagged release and the `main`
branch. Older development snapshots are not supported.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use
[GitHub private vulnerability reporting](https://github.com/MozillaFiredoge/hard-sphere-bridge/security/advisories/new)
and include:

- the affected commit or release;
- the platform and build configuration;
- a minimal reproduction;
- the expected impact;
- whether the issue involves untrusted configuration files or output paths.

The project is a local scientific simulator and does not expose a network
service. Security reports should nevertheless consider malformed JSON,
integer-size overflow, unsafe output paths, dependency issues, and denial of
service through pathological simulation parameters.

General numerical disagreements and failed convergence checks belong in the
scientific-validation issue template rather than the private security channel.
