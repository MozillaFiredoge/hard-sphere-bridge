cmake_minimum_required(VERSION 3.24)

project(
    particles
    VERSION 0.1.0
    DESCRIPTION "Newtonian hard-sphere simulation from microscopic to macroscopic scales"
    LANGUAGES CXX
)

option(PARTICLES_ENABLE_CUDA "Build CUDA components when a CUDA compiler is available" ON)

set(PARTICLES_GIT_COMMIT "unknown")
set(PARTICLES_GIT_DIRTY 0)
find_package(Git QUIET)
if(Git_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        RESULT_VARIABLE PARTICLES_GIT_RESULT
        OUTPUT_VARIABLE PARTICLES_GIT_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(PARTICLES_GIT_RESULT EQUAL 0)
        set(PARTICLES_GIT_COMMIT "${PARTICLES_GIT_OUTPUT}")
    endif()
    execute_process(
        COMMAND ${GIT_EXECUTABLE} status --porcelain
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_VARIABLE PARTICLES_GIT_STATUS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT PARTICLES_GIT_STATUS STREQUAL "")
        set(PARTICLES_GIT_DIRTY 1)
    endif()
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CTest)

find_package(Boost 1.75 REQUIRED)

add_library(particles_core STATIC
    src/collision_events.cpp
    src/config.cpp
    src/cpu_reference/event_driven_simulator.cpp
    src/diagnostics.cpp
    src/cpu_reference/hard_sphere.cpp
    src/initialization.cpp
    src/parallel_window.cpp
    src/uniform_grid.cpp
)

target_include_directories(particles_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
    SYSTEM PRIVATE
        ${Boost_INCLUDE_DIRS}
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
            src/cuda/cuda_particle_grid.cu
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

        target_link_libraries(particles_cuda PUBLIC particles_core)
    else()
        message(WARNING
            "PARTICLES_ENABLE_CUDA is ON, but no CUDA compiler was found. "
            "Only CPU components will be built."
        )
    endif()
endif()

add_executable(particles_cli
    src/app/cuda_scaling.cpp
    src/app/main.cpp
    src/app/multiscale_snapshot.cpp
    src/app/window_stress.cpp
)

target_link_libraries(particles_cli PRIVATE particles_core)
target_compile_definitions(particles_cli PRIVATE
    PARTICLES_VERSION="${PROJECT_VERSION}"
    PARTICLES_BUILD_TYPE="$<CONFIG>"
    PARTICLES_CXX_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
    PARTICLES_CXX_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
    PARTICLES_GIT_COMMIT="${PARTICLES_GIT_COMMIT}"
    PARTICLES_GIT_DIRTY=${PARTICLES_GIT_DIRTY}
)

if(PARTICLES_HAS_CUDA)
    target_link_libraries(particles_cli PRIVATE particles_cuda)
    target_compile_definitions(particles_cli PRIVATE
        PARTICLES_HAS_CUDA=1
        PARTICLES_CUDA_COMPILER_VERSION="${CMAKE_CUDA_COMPILER_VERSION}"
    )
else()
    target_compile_definitions(particles_cli PRIVATE
        PARTICLES_HAS_CUDA=0
        PARTICLES_CUDA_COMPILER_VERSION="disabled"
    )
endif()

include(GNUInstallDirs)
set(
    PARTICLES_INSTALL_DOCDIR
    "${CMAKE_INSTALL_DATADIR}/doc/hard-sphere-bridge"
)

install(
    TARGETS particles_cli
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(
    DIRECTORY configs/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/hard-sphere-bridge/configs
)
install(
    PROGRAMS
        tools/analyze_n_kn_scan.py
        tools/analyze_multiseed_fields.py
        tools/analyze_wave_kinematics.py
        tools/euler_reference.py
        tools/run_n_kn_scan.py
        tools/run_multiseed_shock_tube.py
        tools/visualize_shock_tube.py
    DESTINATION ${CMAKE_INSTALL_DATADIR}/hard-sphere-bridge/tools
)
install(
    FILES
        LICENSE
        README.md
        CHANGELOG.md
        CITATION.cff
        CODE_OF_CONDUCT.md
        CONTRIBUTING.md
        RELEASE_CHECKLIST.md
        SECURITY.md
    DESTINATION ${PARTICLES_INSTALL_DOCDIR}
)
install(
    DIRECTORY docs/
    DESTINATION ${PARTICLES_INSTALL_DOCDIR}/docs
)

if(BUILD_TESTING)
    add_executable(particles_tests
        tests/test_collision_events.cpp
        tests/test_diagnostics.cpp
        tests/test_event_driven_simulator.cpp
        tests/test_hard_sphere.cpp
        tests/test_initialization.cpp
        tests/test_parallel_window.cpp
        tests/test_uniform_grid.cpp
    )

    target_link_libraries(particles_tests PRIVATE particles_core)
    target_compile_definitions(particles_tests PRIVATE
        PARTICLES_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    )

    add_test(
        NAME hard_sphere_unit_tests
        COMMAND particles_tests
    )

    add_test(
        NAME particles_cli_help
        COMMAND particles_cli --help
    )
    set_tests_properties(particles_cli_help PROPERTIES
        PASS_REGULAR_EXPRESSION "Usage: particles_cli"
    )

    add_test(
        NAME particles_cli_version
        COMMAND particles_cli --version
    )
    set_tests_properties(particles_cli_version PROPERTIES
        PASS_REGULAR_EXPRESSION "particles_cli ${PROJECT_VERSION}"
    )

    add_test(
        NAME particles_cli_rejects_unknown_command
        COMMAND particles_cli definitely-not-a-command
    )
    set_tests_properties(particles_cli_rejects_unknown_command PROPERTIES
        WILL_FAIL TRUE
    )

    if(PARTICLES_HAS_CUDA)
        add_executable(particles_cuda_tests
            tests/test_cuda_particle_grid.cpp
        )
        target_link_libraries(particles_cuda_tests PRIVATE particles_cuda)

        add_test(
            NAME cuda_particle_grid_tests
            COMMAND particles_cuda_tests
        )
        set_tests_properties(cuda_particle_grid_tests PROPERTIES
            SKIP_RETURN_CODE 77
        )
    endif()
endif()

message(STATUS "Particles CUDA support: ${PARTICLES_HAS_CUDA}")
