#pragma once

#include <cmath>

namespace particles {

struct Vec3 {
    double x{};
    double y{};
    double z{};

    constexpr Vec3& operator+=(const Vec3& other) noexcept {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    constexpr Vec3& operator-=(const Vec3& other) noexcept {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    constexpr Vec3& operator*=(double scalar) noexcept {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }
};

[[nodiscard]] constexpr Vec3 operator+(Vec3 left, const Vec3& right) noexcept {
    return left += right;
}

[[nodiscard]] constexpr Vec3 operator-(Vec3 left, const Vec3& right) noexcept {
    return left -= right;
}

[[nodiscard]] constexpr Vec3 operator-(const Vec3& value) noexcept {
    return {-value.x, -value.y, -value.z};
}

[[nodiscard]] constexpr Vec3 operator*(Vec3 value, double scalar) noexcept {
    return value *= scalar;
}

[[nodiscard]] constexpr Vec3 operator*(double scalar, Vec3 value) noexcept {
    return value *= scalar;
}

[[nodiscard]] constexpr Vec3 operator/(Vec3 value, double scalar) noexcept {
    value *= 1.0 / scalar;
    return value;
}

[[nodiscard]] constexpr double dot(const Vec3& left, const Vec3& right) noexcept {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

[[nodiscard]] constexpr double squared_norm(const Vec3& value) noexcept {
    return dot(value, value);
}

[[nodiscard]] inline double norm(const Vec3& value) noexcept {
    return std::sqrt(squared_norm(value));
}

}  // namespace particles

