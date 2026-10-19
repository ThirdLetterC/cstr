cmake_minimum_required(VERSION 3.21)

# Read the public version so the version workflow remains the single updater.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS cstr.h)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/cstr.h" cstr_version_line
    REGEX "^#define CS_VERSION_STRING \"[0-9]+\\.[0-9]+\\.[0-9]+\"$")
if(NOT cstr_version_line)
    message(FATAL_ERROR "Cannot read CS_VERSION_STRING from cstr.h")
endif()
string(REGEX REPLACE ".*\"([0-9]+\\.[0-9]+\\.[0-9]+)\".*" "\\1"
    cstr_version "${cstr_version_line}")
project(cstr VERSION "${cstr_version}" LANGUAGES C)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(BUILD_TESTING "Build the cstr test suite" ${PROJECT_IS_TOP_LEVEL})
option(CS_BUILD_BENCHMARKS "Build the cstr benchmark" ${PROJECT_IS_TOP_LEVEL})
option(CS_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(CS_BUILD_FUZZERS "Build libFuzzer targets (requires Clang)" OFF)

add_library(cstr cstr.c)
add_library(cstr::cstr ALIAS cstr)
target_compile_features(cstr PUBLIC c_std_11)
set_target_properties(cstr PROPERTIES
    C_EXTENSIONS OFF
    VERSION "${PROJECT_VERSION}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"
    WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_include_directories(cstr PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if(UNIX)
    target_link_libraries(cstr PUBLIC m)
elseif(WIN32)
    target_link_libraries(cstr PUBLIC bcrypt)
endif()

if(CS_ENABLE_SANITIZERS)
    if(NOT CMAKE_C_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$" OR MSVC)
        message(FATAL_ERROR "CS_ENABLE_SANITIZERS requires GCC or Clang with a Unix-style driver")
    endif()
    target_compile_options(cstr PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(cstr PUBLIC -fsanitize=address,undefined)
endif()

set(cstr_build_targets cstr)
if(BUILD_TESTING)
    include(CTest)
    add_executable(test_cstr test_cstr.c)
    target_link_libraries(test_cstr PRIVATE cstr::cstr)
    add_test(NAME cstr COMMAND test_cstr)
    list(APPEND cstr_build_targets test_cstr)
endif()

if(CS_BUILD_BENCHMARKS)
    add_executable(bench_cstr bench_cstr.c)
    target_link_libraries(bench_cstr PRIVATE cstr::cstr)
    add_custom_target(bench
        COMMAND $<TARGET_FILE:bench_cstr>
        DEPENDS bench_cstr
        USES_TERMINAL)
    list(APPEND cstr_build_targets bench_cstr)
endif()

foreach(cstr_target IN LISTS cstr_build_targets)
    set_target_properties(${cstr_target} PROPERTIES C_EXTENSIONS OFF)
    if(MSVC)
        target_compile_options(${cstr_target} PRIVATE /W4)
    elseif(CMAKE_C_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
        target_compile_options(${cstr_target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endforeach()

if(CS_BUILD_FUZZERS)
    add_subdirectory(fuzz)
endif()

install(TARGETS cstr EXPORT cstrTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES cstr.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT cstrTargets NAMESPACE cstr::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cstr)
configure_package_config_file(cmake/cstrConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/cstrConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cstr)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/cstrConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}" COMPATIBILITY SameMajorVersion)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cstrConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/cstrConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cstr)
