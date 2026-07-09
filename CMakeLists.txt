cmake_minimum_required(VERSION 3.16)
project(nanorag VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(NANORAG_BUILD_TESTS "Build nanorag unit tests" ON)
option(NANORAG_BUILD_CLI "Build nanorag CLI binary" ON)
option(NANORAG_INSTALL "Generate install/export rules" ON)
option(NANORAG_USE_SUBMODULES
  "Use third_party/tinyann and third_party/nanollm submodules (default). OFF uses find_package."
  ON)

# ---------------------------------------------------------------------------
# Dependencies: submodules (dev default) or installed packages
# ---------------------------------------------------------------------------
if(NANORAG_USE_SUBMODULES)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tinyann/CMakeLists.txt")
    message(FATAL_ERROR
      "third_party/tinyann missing. Run: git submodule update --init --recursive\n"
      "Or configure with -DNANORAG_USE_SUBMODULES=OFF after installing tinyann.")
  endif()
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nanollm/CMakeLists.txt")
    message(FATAL_ERROR
      "third_party/nanollm missing. Run: git submodule update --init --recursive\n"
      "Or configure with -DNANORAG_USE_SUBMODULES=OFF after installing nanollm.")
  endif()

  set(TINYANN_BUILD_CLI OFF CACHE BOOL "Build tinyann CLI" FORCE)
  set(TINYANN_BUILD_TESTS OFF CACHE BOOL "Build tinyann tests" FORCE)
  set(TINYANN_INSTALL OFF CACHE BOOL "Skip nested tinyann install; parent may re-export" FORCE)
  set(NANOLLM_BUILD_CLI OFF CACHE BOOL "Build nanollm CLI" FORCE)
  set(NANOLLM_BUILD_TESTS OFF CACHE BOOL "Build nanollm tests" FORCE)
  set(NANOLLM_INSTALL OFF CACHE BOOL "Skip nested nanollm install; parent may re-export" FORCE)

  add_subdirectory(third_party/tinyann EXCLUDE_FROM_ALL)
  add_subdirectory(third_party/nanollm EXCLUDE_FROM_ALL)
else()
  find_package(tinyann 0.1 CONFIG REQUIRED)
  find_package(nanollm 0.4 CONFIG REQUIRED)
endif()

# ---------------------------------------------------------------------------
# nanorag library (header-only interface)
# ---------------------------------------------------------------------------
add_library(nanorag INTERFACE)
add_library(nanorag::nanorag ALIAS nanorag)
target_include_directories(nanorag INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(nanorag INTERFACE tinyann::tinyann nanollm::nanollm)
target_compile_features(nanorag INTERFACE cxx_std_17)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if(NANORAG_BUILD_CLI)
  add_executable(nanorag_cli src/main.cpp)
  target_link_libraries(nanorag_cli PRIVATE nanorag)
  set_target_properties(nanorag_cli PROPERTIES OUTPUT_NAME nanorag)
  if(NOT MSVC)
    target_compile_options(nanorag_cli PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(NANORAG_BUILD_TESTS)
  enable_testing()
  add_executable(nanorag_tests tests/test_embedder_store.cpp)
  target_link_libraries(nanorag_tests PRIVATE nanorag)
  add_executable(nanorag_grounding_tests tests/test_grounding.cpp)
  target_link_libraries(nanorag_grounding_tests PRIVATE nanorag)
  add_executable(nanorag_eval_harness tests/test_eval_harness.cpp)
  target_link_libraries(nanorag_eval_harness PRIVATE nanorag)
  if(NOT MSVC)
    target_compile_options(nanorag_tests PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(nanorag_grounding_tests PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(nanorag_eval_harness PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  add_test(NAME nanorag_tests COMMAND nanorag_tests)
  add_test(NAME nanorag_grounding_tests COMMAND nanorag_grounding_tests)
  # Eval harness needs data/demo; run from source root
  add_test(
    NAME nanorag_eval_harness
    COMMAND nanorag_eval_harness
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  )
  set_tests_properties(nanorag_eval_harness PROPERTIES
    ENVIRONMENT "NANORAG_DEMO_DATA=${CMAKE_SOURCE_DIR}/data/demo"
  )
endif()

# ---------------------------------------------------------------------------
# Install / export
# ---------------------------------------------------------------------------
if(NANORAG_INSTALL)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  # Self-contained prefix when built with submodules: install deps under their own namespaces.
  if(NANORAG_USE_SUBMODULES)
    install(TARGETS tinyann
      EXPORT tinyannTargets
      INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(DIRECTORY third_party/tinyann/include/tinyann
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(EXPORT tinyannTargets
      FILE tinyannTargets.cmake
      NAMESPACE tinyann::
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tinyann
    )
    configure_package_config_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tinyann/cmake/tinyannConfig.cmake.in
      ${CMAKE_CURRENT_BINARY_DIR}/tinyannConfig.cmake
      INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tinyann
    )
    write_basic_package_version_file(
      ${CMAKE_CURRENT_BINARY_DIR}/tinyannConfigVersion.cmake
      VERSION 0.1.0
      COMPATIBILITY SameMajorVersion
    )
    install(FILES
      ${CMAKE_CURRENT_BINARY_DIR}/tinyannConfig.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/tinyannConfigVersion.cmake
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tinyann
    )

    install(TARGETS nanollm
      EXPORT nanollmTargets
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
      INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(DIRECTORY third_party/nanollm/include/nanollm
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(EXPORT nanollmTargets
      FILE nanollmTargets.cmake
      NAMESPACE nanollm::
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nanollm
    )
    configure_package_config_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nanollm/cmake/nanollmConfig.cmake.in
      ${CMAKE_CURRENT_BINARY_DIR}/nanollmConfig.cmake
      INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nanollm
    )
    write_basic_package_version_file(
      ${CMAKE_CURRENT_BINARY_DIR}/nanollmConfigVersion.cmake
      VERSION 0.4.0
      COMPATIBILITY SameMajorVersion
    )
    install(FILES
      ${CMAKE_CURRENT_BINARY_DIR}/nanollmConfig.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/nanollmConfigVersion.cmake
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nanollm
    )
  endif()

  install(TARGETS nanorag
    EXPORT nanoragTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
  install(DIRECTORY include/nanorag
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
  if(NANORAG_BUILD_CLI)
    install(TARGETS nanorag_cli
      EXPORT nanoragTargets
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
  endif()

  install(EXPORT nanoragTargets
    FILE nanoragTargets.cmake
    NAMESPACE nanorag::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nanorag
  )

  configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/nanoragConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/nanoragConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nanorag
  )
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/nanoragConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
  )
  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/nanoragConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/nanoragConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nanorag
  )

  install(FILES VERSION COMPATIBILITY.md LICENSE
    DESTINATION ${CMAKE_INSTALL_DATADIR}/nanorag
    OPTIONAL
  )
endif()
