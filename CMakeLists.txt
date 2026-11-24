cmake_minimum_required(VERSION 3.20)
project(Lenia LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(lenia_core src/convolve.cpp src/step.cpp src/kernel.cpp src/kernel_shape.cpp src/pattern.cpp
                       src/environment.cpp src/node_graph.cpp src/project.cpp src/rule.cpp src/display_graph.cpp
                       src/snapshot.cpp src/fft_convolve.cpp src/world.cpp)
target_include_directories(lenia_core PUBLIC include)

enable_testing()

add_executable(test_convolve tests/test_convolve.cpp)
target_link_libraries(test_convolve PRIVATE lenia_core)
add_test(NAME convolve_impulse COMMAND test_convolve)

add_executable(test_step tests/test_step.cpp)
target_link_libraries(test_step PRIVATE lenia_core)
add_test(NAME lenia_step COMMAND test_step)

add_executable(test_ring_kernel tests/test_ring_kernel.cpp)
target_link_libraries(test_ring_kernel PRIVATE lenia_core)
add_test(NAME ring_kernel COMMAND test_ring_kernel)

add_executable(test_kernel_shape tests/test_kernel_shape.cpp)
target_link_libraries(test_kernel_shape PRIVATE lenia_core)
add_test(NAME kernel_shape COMMAND test_kernel_shape)

# Bundled creature data (patterns/README.md); each file's license ships with it.
set(LENIA_PATTERN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/patterns)
set(LENIA_PATTERN_FILES ${LENIA_PATTERN_DIR}/orbium.lenia ${LENIA_PATTERN_DIR}/ORBIUM-LICENSE.txt)

add_executable(test_orbium tests/test_orbium.cpp)
target_link_libraries(test_orbium PRIVATE lenia_core)
target_compile_definitions(test_orbium PRIVATE LENIA_PATTERN_DIR="${LENIA_PATTERN_DIR}")
add_test(NAME orbium COMMAND test_orbium)

add_executable(test_node_graph tests/test_node_graph.cpp)
target_link_libraries(test_node_graph PRIVATE lenia_core)
target_compile_definitions(test_node_graph PRIVATE LENIA_PATTERN_DIR="${LENIA_PATTERN_DIR}")
add_test(NAME node_graph COMMAND test_node_graph)

add_executable(test_project tests/test_project.cpp)
target_link_libraries(test_project PRIVATE lenia_core)
target_compile_definitions(test_project PRIVATE LENIA_PATTERN_DIR="${LENIA_PATTERN_DIR}")
add_test(NAME project COMMAND test_project)

add_executable(test_rule tests/test_rule.cpp)
target_link_libraries(test_rule PRIVATE lenia_core)
target_compile_definitions(test_rule PRIVATE LENIA_PATTERN_DIR="${LENIA_PATTERN_DIR}")
add_test(NAME rule COMMAND test_rule)

# Two state fields (Density + Nutrient): seeding, simultaneous update, the trail demo, files.
add_executable(test_state_fields tests/test_state_fields.cpp)
target_link_libraries(test_state_fields PRIVATE lenia_core)
target_compile_definitions(test_state_fields PRIVATE LENIA_PATTERN_DIR="${LENIA_PATTERN_DIR}")
add_test(NAME state_fields COMMAND test_state_fields)

# Reusable kernels: Kernel K and Convolve(Field, Kernel) nodes, independent kernels per field.
add_executable(test_convolve_node tests/test_convolve_node.cpp)
target_link_libraries(test_convolve_node PRIVATE lenia_core)
add_test(NAME convolve_node COMMAND test_convolve_node)

add_executable(test_display_graph tests/test_display_graph.cpp)
target_link_libraries(test_display_graph PRIVATE lenia_core)
add_test(NAME display_graph COMMAND test_display_graph)

add_executable(test_snapshot tests/test_snapshot.cpp)
target_link_libraries(test_snapshot PRIVATE lenia_core)
target_compile_definitions(test_snapshot PRIVATE LENIA_PATTERN_DIR="${LENIA_PATTERN_DIR}")
add_test(NAME snapshot COMMAND test_snapshot)

# CPU reference of the GPU's FFT convolution (shaders/fft_convolve.comp) against convolve.
add_executable(test_fft_convolve tests/test_fft_convolve.cpp)
target_link_libraries(test_fft_convolve PRIVATE lenia_core)
target_compile_definitions(test_fft_convolve PRIVATE LENIA_PATTERN_DIR="${LENIA_PATTERN_DIR}")
add_test(NAME fft_convolve COMMAND test_fft_convolve)

# Optional Vulkan compute backend. Needs the Vulkan SDK (headers, loader import library and
# glslc or glslangValidator); the CPU-only build above does not.
option(LENIA_ENABLE_VULKAN "Build the Vulkan compute backend and its CPU/GPU parity test" OFF)

# Optional Qt6 Widgets kernel workbench (apps/workbench). Needs LENIA_ENABLE_VULKAN and a Qt 6
# build with Vulkan support; set CMAKE_PREFIX_PATH to the Qt kit, e.g. C:/Qt/6.8.3/msvc2022_64.
option(LENIA_BUILD_WORKBENCH "Build the Qt6 Vulkan kernel workbench" OFF)

if(LENIA_BUILD_WORKBENCH AND NOT LENIA_ENABLE_VULKAN)
    message(FATAL_ERROR "LENIA_BUILD_WORKBENCH needs LENIA_ENABLE_VULKAN=ON")
endif()

if(LENIA_ENABLE_VULKAN)
    find_package(Vulkan REQUIRED)

    set(LENIA_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
    set(LENIA_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)

    if(NOT Vulkan_GLSLC_EXECUTABLE AND NOT Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
        message(FATAL_ERROR "LENIA_ENABLE_VULKAN needs glslc or glslangValidator (part of the Vulkan SDK)")
    endif()

    # Compiles shaders/<source> to ${LENIA_GENERATED_DIR}/<output>: SPIR-V as comma-separated
    # 32-bit words, #included into a uint32_t array. The stage comes from the file extension.
    # Every shader depends on lenia_math.glsl and lenia_rule.glsl, the shared #includes.
    function(lenia_add_shader source output)
        set(src ${LENIA_SHADER_DIR}/${source})
        set(out ${LENIA_GENERATED_DIR}/${output})
        if(Vulkan_GLSLC_EXECUTABLE)
            set(compile ${Vulkan_GLSLC_EXECUTABLE} -mfmt=num -o ${out} ${src})
        else()
            set(compile ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V -x -o ${out} ${src})
        endif()
        add_custom_command(
            OUTPUT ${out}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${LENIA_GENERATED_DIR}
            COMMAND ${compile}
            DEPENDS ${src} ${LENIA_SHADER_DIR}/lenia_math.glsl ${LENIA_SHADER_DIR}/lenia_rule.glsl
            COMMENT "Compiling ${source} to SPIR-V"
            VERBATIM)
    endfunction()

    lenia_add_shader(lenia_step.comp lenia_step.spv.inc)
    lenia_add_shader(rule_step.comp rule_step.spv.inc)
    lenia_add_shader(fft_convolve.comp fft_convolve.spv.inc)

    add_library(lenia_vulkan src/vulkan_step.cpp ${LENIA_GENERATED_DIR}/lenia_step.spv.inc
                             ${LENIA_GENERATED_DIR}/rule_step.spv.inc ${LENIA_GENERATED_DIR}/fft_convolve.spv.inc)
    target_include_directories(lenia_vulkan PRIVATE ${LENIA_GENERATED_DIR})
    target_link_libraries(lenia_vulkan PUBLIC lenia_core PRIVATE Vulkan::Vulkan)

    add_executable(test_vulkan_parity tests/test_vulkan_parity.cpp)
    target_link_libraries(test_vulkan_parity PRIVATE lenia_vulkan)
    target_compile_definitions(test_vulkan_parity PRIVATE LENIA_PATTERN_DIR="${LENIA_PATTERN_DIR}")
    add_test(NAME vulkan_parity COMMAND test_vulkan_parity)
    # Reported as skipped, not failed, when no Vulkan device is available at runtime.
    set_tests_properties(vulkan_parity PROPERTIES SKIP_RETURN_CODE 77)

    # Headless GPU benchmark (docs/benchmark.md). Not a test: run it by hand, in Release.
    add_executable(lenia_bench apps/bench/main.cpp)
    target_link_libraries(lenia_bench PRIVATE lenia_vulkan)
    target_compile_definitions(lenia_bench PRIVATE LENIA_PATTERN_DIR="${LENIA_PATTERN_DIR}")
endif()

if(LENIA_BUILD_WORKBENCH)
    find_package(Qt6 6.2 REQUIRED COMPONENTS Gui Widgets)

    lenia_add_shader(workbench_step.comp workbench_step.spv.inc)
    lenia_add_shader(workbench_view.vert workbench_view.vert.spv.inc)
    lenia_add_shader(workbench_view.frag workbench_view.frag.spv.inc)

    # Everything but main.cpp, shared by the application and its Qt/Vulkan smoke test.
    set(LENIA_WORKBENCH_SOURCES
        apps/workbench/main_window.cpp
        apps/workbench/main_window.hpp
        apps/workbench/edit_history.cpp
        apps/workbench/edit_history.hpp
        apps/workbench/lenia_view.cpp
        apps/workbench/lenia_view.hpp
        apps/workbench/kernel_editor.cpp
        apps/workbench/kernel_editor.hpp
        apps/workbench/node_editor.cpp
        apps/workbench/node_editor.hpp
        apps/workbench/palettes.cpp
        apps/workbench/palettes.hpp
        ${LENIA_GENERATED_DIR}/workbench_step.spv.inc
        ${LENIA_GENERATED_DIR}/fft_convolve.spv.inc
        ${LENIA_GENERATED_DIR}/workbench_view.vert.spv.inc
        ${LENIA_GENERATED_DIR}/workbench_view.frag.spv.inc)
    # Vulkan calls go through Qt's QVulkanFunctions tables; only the headers are needed.
    if(TARGET Vulkan::Headers)
        set(LENIA_VULKAN_HEADERS Vulkan::Headers)
    else()
        set(LENIA_VULKAN_HEADERS Vulkan::Vulkan)
    endif()

    # Compiles the workbench sources into `target`, with the bundled patterns as the Qt
    # resource `resource_name` (one per target: the generated files are named after it).
    function(lenia_add_workbench_sources target resource_name)
        target_sources(${target} PRIVATE ${LENIA_WORKBENCH_SOURCES})
        set_target_properties(${target} PROPERTIES AUTOMOC ON)
        target_include_directories(${target} PRIVATE ${LENIA_GENERATED_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/apps/workbench)
        # UI strings use non-ASCII characters (μ, σ, ×).
        target_compile_options(${target} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
        target_link_libraries(${target} PRIVATE lenia_core Qt6::Gui Qt6::Widgets ${LENIA_VULKAN_HEADERS})
        # The default Orbium preset is compiled in (":/patterns/orbium.lenia"), together with
        # its license, so it loads regardless of the working directory.
        qt_add_resources(${target} ${resource_name}
            PREFIX /patterns
            BASE ${LENIA_PATTERN_DIR}
            FILES ${LENIA_PATTERN_FILES})
    endfunction()

    add_executable(lenia_workbench WIN32 apps/workbench/main.cpp)
    lenia_add_workbench_sources(lenia_workbench lenia_patterns)

    # Real Qt + Vulkan smoke test: opens the workbench window, checks that a display graph edit
    # recolors the viewport without touching the world, and that a snapshot continues exactly.
    # Needs a desktop session and a Vulkan device; skipped (77) without them. It is built into
    # the same output directory as lenia_workbench, after it, so it finds the Qt runtime that
    # windeployqt put there.
    add_executable(test_workbench_smoke tests/test_workbench_smoke.cpp)
    lenia_add_workbench_sources(test_workbench_smoke lenia_patterns_smoke)
    target_compile_definitions(test_workbench_smoke PRIVATE LENIA_PATTERN_DIR="${LENIA_PATTERN_DIR}")
    add_dependencies(test_workbench_smoke lenia_workbench)
    add_test(NAME workbench_smoke COMMAND test_workbench_smoke)
    set_tests_properties(workbench_smoke PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180 LABELS gui)
    # Undo / redo in the real window (docs/undo.md): mouse gestures on the node canvases and the
    # kernel canvas, inspector and panel edits, Edit menu state, Ctrl+Z from the viewport, and
    # that undo never touches the world, its generation or play / pause. Same requirements.
    add_executable(test_workbench_undo tests/test_workbench_undo.cpp)
    lenia_add_workbench_sources(test_workbench_undo lenia_patterns_undo)
    target_compile_definitions(test_workbench_undo PRIVATE LENIA_PATTERN_DIR="${LENIA_PATTERN_DIR}")
    add_dependencies(test_workbench_undo lenia_workbench)
    add_test(NAME workbench_undo COMMAND test_workbench_undo)
    set_tests_properties(workbench_undo PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180 LABELS gui)

    # The undo history's checkpoint, merge and reset rules on their own: no window, no Vulkan.
    add_executable(test_edit_history tests/test_edit_history.cpp apps/workbench/edit_history.cpp
                                     apps/workbench/edit_history.hpp)
    set_target_properties(test_edit_history PROPERTIES AUTOMOC ON)
    target_include_directories(test_edit_history PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/apps/workbench)
    target_link_libraries(test_edit_history PRIVATE lenia_core Qt6::Gui)
    add_dependencies(test_edit_history lenia_workbench)
    add_test(NAME edit_history COMMAND test_edit_history)

    if(WIN32)
        # Without windeployqt (LENIA_WORKBENCH_DEPLOY=OFF) the Qt DLLs come from the kit.
        set_tests_properties(workbench_smoke workbench_undo edit_history PROPERTIES
            ENVIRONMENT_MODIFICATION "PATH=path_list_append:$<TARGET_FILE_DIR:Qt6::Core>")
    endif()

    # The pattern files and license also sit next to the executable, in patterns/, for the
    # "Load pattern file…" dialog and so a copied build directory carries the license.
    add_custom_command(TARGET lenia_workbench POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:lenia_workbench>/patterns
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LENIA_PATTERN_FILES}
                $<TARGET_FILE_DIR:lenia_workbench>/patterns
        COMMENT "Copying bundled patterns next to lenia_workbench"
        VERBATIM)

    # Copies the Qt runtime next to the executable so it starts from the build tree.
    option(LENIA_WORKBENCH_DEPLOY "Run windeployqt on lenia_workbench after each build (Windows)" ON)
    if(WIN32 AND LENIA_WORKBENCH_DEPLOY AND TARGET Qt6::windeployqt)
        add_custom_command(TARGET lenia_workbench POST_BUILD
            COMMAND Qt6::windeployqt --no-translations $<TARGET_FILE:lenia_workbench>
            COMMENT "Deploying Qt runtime for lenia_workbench"
            VERBATIM)
    endif()

    # Distribution: cmake --install <build> --config Release --prefix <dir> gives
    # <dir>/bin/lenia_workbench(.exe), <dir>/bin/patterns/ and, on Windows, the Qt runtime.
    include(GNUInstallDirs)
    install(TARGETS lenia_workbench RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(FILES ${LENIA_PATTERN_FILES} DESTINATION ${CMAKE_INSTALL_BINDIR}/patterns)
    if(WIN32 AND TARGET Qt6::windeployqt)
        install(CODE "
            execute_process(
                COMMAND \"$<TARGET_FILE:Qt6::windeployqt>\" --no-translations
                        \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/$<TARGET_FILE_NAME:lenia_workbench>\"
                COMMAND_ERROR_IS_FATAL ANY)")
    endif()
endif()
