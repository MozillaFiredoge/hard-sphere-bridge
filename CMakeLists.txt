cmake_minimum_required(VERSION 3.24)

project(
    particles
    VERSION 0.1.0
    DESCRIPTION "Newtonian hard-sphere simulation from microscopic to macroscopic scales"
    LANGUAGES CXX
)

option(PARTICLES_ENABLE_CUDA "Build CUDA components when a CUDA compiler is available" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CTest)

add_library(particles_core STATIC
    src/cpu_reference/hard_sphere.cpp
)

target_include_directories(particles_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(particles_core PUBLIC cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(particles_core PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
    )
elseif(MSVC)
    target_compile_options(particles_core PRIVATE /W4 /permissive-)
endif()

set(PARTICLES_HAS_CUDA OFF)

if(PARTICLES_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            # Keep configuration working in headless build environments where
            # nvcc cannot inspect a local GPU. A single architecture also emits
            # PTX that newer devices can JIT. Production builds should override
            # this with their native architecture for best performance.
            set(CMAKE_CUDA_ARCHITECTURES 75)
        endif()

        enable_language(CUDA)
        set(PARTICLES_HAS_CUDA ON)

        add_library(particles_cuda STATIC
            src/cuda/cuda_support.cu
        )

        target_include_directories(particles_cuda
            PUBLIC
                ${PROJECT_SOURCE_DIR}/include
        )

        set_target_properties(particles_cuda PROPERTIES
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
            CUDA_EXTENSIONS OFF
            CUDA_SEPARABLE_COMPILATION ON
        )
    else()
        message(WARNING
            "PARTICLES_ENABLE_CUDA is ON, but no CUDA compiler was found. "
            "Only CPU components will be built."
        )
    endif()
endif()

add_executable(particles_cli
    src/app/main.cpp
)

target_link_libraries(particles_cli PRIVATE particles_core)

if(PARTICLES_HAS_CUDA)
    target_link_libraries(particles_cli PRIVATE particles_cuda)
    target_compile_definitions(particles_cli PRIVATE PARTICLES_HAS_CUDA=1)
else()
    target_compile_definitions(particles_cli PRIVATE PARTICLES_HAS_CUDA=0)
endif()

if(BUILD_TESTING)
    add_executable(particles_tests
        tests/test_hard_sphere.cpp
    )

    target_link_libraries(particles_tests PRIVATE particles_core)

    add_test(
        NAME hard_sphere_unit_tests
        COMMAND particles_tests
    )
endif()

message(STATUS "Particles CUDA support: ${PARTICLES_HAS_CUDA}")
