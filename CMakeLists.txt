cmake_minimum_required(VERSION 3.16)
project(nanorag VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(NANORAG_BUILD_TESTS "Build nanorag unit tests" ON)

# ---------------------------------------------------------------------------
# Dependencies (owned libraries; no third-party ML runtimes)
# ---------------------------------------------------------------------------
set(TINYANN_BUILD_CLI OFF CACHE BOOL "Build tinyann CLI" FORCE)
set(TINYANN_BUILD_TESTS OFF CACHE BOOL "Build tinyann tests" FORCE)
set(NANOLLM_BUILD_CLI OFF CACHE BOOL "Build nanollm CLI" FORCE)
set(NANOLLM_BUILD_TESTS OFF CACHE BOOL "Build nanollm tests" FORCE)

add_subdirectory(third_party/tinyann EXCLUDE_FROM_ALL)
add_subdirectory(third_party/nanollm EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# nanorag library (header-heavy Phase 0; sources for CLI-free core later)
# ---------------------------------------------------------------------------
add_library(nanorag INTERFACE)
add_library(nanorag::nanorag ALIAS nanorag)
target_include_directories(nanorag INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(nanorag INTERFACE tinyann::tinyann nanollm::nanollm)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
add_executable(nanorag_cli src/main.cpp)
target_link_libraries(nanorag_cli PRIVATE nanorag)
set_target_properties(nanorag_cli PROPERTIES OUTPUT_NAME nanorag)
if(NOT MSVC)
    target_compile_options(nanorag_cli PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(NANORAG_BUILD_TESTS)
    enable_testing()
    add_executable(nanorag_tests tests/test_embedder_store.cpp)
    target_link_libraries(nanorag_tests PRIVATE nanorag)
    if(NOT MSVC)
        target_compile_options(nanorag_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME nanorag_tests COMMAND nanorag_tests)
endif()

install(TARGETS nanorag_cli RUNTIME DESTINATION bin)
install(DIRECTORY include/nanorag DESTINATION include)
