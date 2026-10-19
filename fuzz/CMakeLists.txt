if(NOT CMAKE_C_COMPILER_ID MATCHES "^(Clang|AppleClang)$" OR MSVC)
    message(FATAL_ERROR "Fuzzing requires Clang with the libFuzzer runtime; set CMAKE_C_COMPILER")
endif()
if(NOT UNIX)
    message(FATAL_ERROR "The fuzz runner currently supports Linux and macOS")
endif()

include(CheckCSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-fsanitize=fuzzer,address,undefined")
check_c_source_compiles("
    #include <stddef.h>
    int LLVMFuzzerTestOneInput(const unsigned char *data, size_t size) {
        (void)data; (void)size; return 0;
    }" CS_HAS_LIBFUZZER)
unset(CMAKE_REQUIRED_FLAGS)
if(NOT CS_HAS_LIBFUZZER)
    message(FATAL_ERROR "Clang cannot link libFuzzer/ASan/UBSan. On macOS use Homebrew LLVM's clang.")
endif()

# Instrument the library itself, independently of the normal cstr target.
add_library(cstr_fuzz OBJECT ../cstr.c)
target_include_directories(cstr_fuzz PUBLIC "${PROJECT_SOURCE_DIR}")
target_compile_features(cstr_fuzz PUBLIC c_std_11)
target_compile_options(cstr_fuzz PRIVATE
    -O1 -g -fno-omit-frame-pointer -fsanitize=fuzzer-no-link,address,undefined
    -fno-sanitize-recover=undefined -Wall -Wextra -Wpedantic)
set_target_properties(cstr_fuzz PROPERTIES C_EXTENSIONS OFF)

foreach(name core unicode codecs)
    add_executable(fuzz_${name} fuzz_${name}.c)
    target_link_libraries(fuzz_${name} PRIVATE cstr_fuzz m)
    target_compile_options(fuzz_${name} PRIVATE
        -O1 -g -fno-omit-frame-pointer -fsanitize=fuzzer,address,undefined
        -fno-sanitize-recover=undefined -Wall -Wextra -Wpedantic)
    target_link_options(fuzz_${name} PRIVATE -fsanitize=fuzzer,address,undefined)
    set_target_properties(fuzz_${name} PROPERTIES C_EXTENSIONS OFF)
endforeach()
