# Build and test

[Back to README](../README.md) · [API guide](api.md)

## Add cstr to a project

Copy [cstr.h](../cstr.h) and [cstr.c](../cstr.c) into your project. Include `cstr.h` where you use the API, and compile `cstr.c` with your sources. C++ callers use the header's `extern "C"` declarations.

The library supports ISO C11, C17, and C23. The commands below use C11 and GCC. Run them from the project root. They link the system math library with `-lm`, as the Makefile does.

## Build with CMake

CMake 3.21 or newer is supported alongside the Makefile. From the project root:

```sh
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --parallel
ctest --test-dir build/release --output-on-failure
```

This builds the `cstr` library, `test_cstr`, and `bench_cstr`. Run the benchmark with:

```sh
cmake --build build/release --target bench
```

For sanitizer tests with GCC or Clang:

```sh
cmake -S . -B build/asan -DCMAKE_BUILD_TYPE=Debug -DCS_ENABLE_SANITIZERS=ON
cmake --build build/asan --parallel
ctest --test-dir build/asan --output-on-failure
```

Use `-DBUILD_TESTING=OFF` to skip tests, `-DCS_BUILD_BENCHMARKS=OFF` to skip
benchmarks, or `-DBUILD_SHARED_LIBS=ON` to build a shared library instead of a
static library. For generators with multiple configurations, add `--config
Release` to build commands and `-C Release` to CTest commands.

To embed the source in another CMake project:

```cmake
add_subdirectory(path/to/cstr)
target_link_libraries(my_app PRIVATE cstr::cstr)
```

Tests and benchmarks default to off when cstr is included as a subproject.
The target supplies the header path and required system libraries.

You can also install the library:

```sh
cmake --install build/release --prefix /path/to/install
```

Consumers can configure with `-DCMAKE_PREFIX_PATH=/path/to/install`, then use:

```cmake
find_package(cstr CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE cstr::cstr)
```

CMake reads the library version from `cstr.h`, so version updates do not need
another version field in the build files.

## Run tests

With Make:

```sh
make test
```

Or compile and run directly:

```sh
gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic cstr.c test_cstr.c -lm -o test_cstr
./test_cstr
```

## Run tests with sanitizers

AddressSanitizer checks memory access. UndefinedBehaviorSanitizer checks for undefined behavior.

```sh
make test-asan
```

Or run:

```sh
gcc -std=c11 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Wall -Wextra -Werror -pedantic cstr.c test_cstr.c -lm -o test_cstr_asan
./test_cstr_asan
```

## Run benchmarks

To use the optimization level from the [recorded benchmarks](benchmarks.md):

```sh
make bench CFLAGS='-std=c11 -O3 -Wall -Wextra -Werror -pedantic'
```

Make only rebuilds a binary when its dependencies change. If you already built `bench_cstr` with different flags, remove it first or use the direct command below to rebuild it.

```sh
gcc -std=c11 -O3 -Wall -Wextra -Werror -pedantic cstr.c bench_cstr.c -lm -o bench_cstr
./bench_cstr
```

## Continuous integration

GitHub Actions runs tests and benchmarks on pushes and pull requests that change
their inputs. You can also start either workflow manually from the Actions tab.

- `test.yml` runs unit tests and AddressSanitizer/UndefinedBehaviorSanitizer on
  Ubuntu 26.04 with GCC and Clang, and macOS 15 (Apple Silicon) with Clang.
  It also builds the library, tests, and benchmark with CMake and runs CTest.
- `benchmark.yml` runs optimized (`-O3`) benchmarks on Ubuntu 26.04 with GCC and
  macOS 15 with Clang. Each job saves a `benchmark.txt` artifact for 30 days,
  including the commit, system, compiler, flags, and results. Available backends
  include NEON on Apple Silicon; unsupported backends are reported as `N/A`.

Both workflows watch `cstr.c`, `cstr.h`, `Makefile`, and their own workflow file.
Tests also watch `test_cstr.c`, `bench_cstr.c`, `CMakeLists.txt`, and `cmake/**`;
benchmarks also watch `bench_cstr.c`.
Documentation-only changes do not trigger either workflow.

Benchmark results are measurements from shared runners. They are useful for
inspection, but do not set pass/fail performance thresholds. The README benchmark
badge reports whether the workflow succeeded.

## Update the library version

After `version.yml` is on the default branch, open **Actions → Update version →
Run workflow** and enter a version such as `1.1.0`.

The workflow accepts a `MAJOR.MINOR.PATCH` version greater than the current one.
It updates the version macros in `cstr.h`, the release number in `README.md`,
and the project version in `pyproject.toml` and `uv.lock`. It leaves the Unicode
data version unchanged.

It builds the library and benchmark, runs unit and sanitizer tests, then commits
and pushes the four updated files to the default branch using `GITHUB_TOKEN`.
Repository permissions and branch rules must allow this push. A concurrent
change to the branch causes the push to fail rather than overwrite that change.

The workflow runs its own checks because a push made with `GITHUB_TOKEN` does
not trigger the separate push workflows. It does not create a tag or GitHub
release. Adding the workflow does not change the current version.

## Format C code

Install `clang-format`, then run:

```sh
make format
```

This formats `cstr.c`, `cstr.h`, `test_cstr.c`, and `bench_cstr.c` in place using
the settings in `.clang-format`. The style uses four-space indentation and a
100-column line limit.

To use a specific executable, run `make format CLANG_FORMAT=clang-format-23`.

## Generate Unicode tables

Normal builds already include the Unicode tables. You only need this step when regenerating them.

```sh
python3 gen_unicode_tables.py unicode_tables.inc
```

The generator is tied to a specific Linux setup. It reads ICU 78 / Unicode 17.0.0 data from `/usr/lib/chromium/chromium` and `/usr/lib/chromium/icudtl.dat`, using fixed offsets. It also loads `/usr/lib/x86_64-linux-gnu/libicuuc.so.72.1`.

Those files must match what the script expects. Python alone is not enough to run it on another system. The command writes `unicode_tables.inc`; it does not automatically replace the tables embedded in `cstr.c`.

See [Unicode data and binary size](unicode.md) for the table contents and reported sizes.
