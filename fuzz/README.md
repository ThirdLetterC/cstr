# Fuzzing cstr

These targets use LLVM libFuzzer, AddressSanitizer (ASan), and UndefinedBehaviorSanitizer
(UBSan). Both the harness and `cstr.c` are instrumented. Checks remain active in
release builds, and UBSan findings stop the run.

You need CMake 3.21 or newer and Clang with the libFuzzer, ASan, and UBSan runtimes.
On macOS, use Homebrew LLVM rather than Apple's system Clang if the system compiler
cannot link libFuzzer.

## Build and run

From the repository root on Linux:

```sh
make fuzz-smoke FUZZ_CC=clang FUZZ_SECONDS=30
```

On Apple Silicon with Homebrew LLVM:

```sh
make fuzz-smoke FUZZ_CC="$(brew --prefix llvm)/bin/clang" FUZZ_SECONDS=30
```

This builds all three targets and runs each for roughly 30 seconds. The script
stops on the first crash, failed check, sanitizer finding, or timeout. Each input
is limited to 4,096 bytes, with a 10-second timeout and a 2,048 MiB RSS limit.
A short run is a smoke check, not proof that the library is free of bugs.

`make fuzz` builds without running. You can also use CMake directly:

```sh
cmake -S . -B build/fuzz -DCMAKE_C_COMPILER=clang \
  -DCS_BUILD_FUZZERS=ON -DBUILD_TESTING=OFF -DCS_BUILD_BENCHMARKS=OFF
cmake --build build/fuzz --target fuzz_core fuzz_unicode fuzz_codecs --parallel
./fuzz/run.sh build/fuzz/fuzz 30
```

Use a new build directory when changing compilers. With a multi-configuration
generator, pass `--config Debug` when building and give the script the executable
directory, such as `build/fuzz/fuzz/Debug`.

## Targets and inputs

| Target | Checks |
| :--- | :--- |
| `fuzz_core` | Append, prepend, insert, erase, replace, assignment, self-append, clone, and reserve against a separate byte-buffer model; exact search against a simple reference scan. |
| `fuzz_unicode` | Repair arbitrary UTF-8, validate repaired text, decode/encode round trips, grapheme iterator progress and ranges, normalization idempotence, and UTF-16 byte round trips in both byte orders. |
| `fuzz_codecs` | Hex, Base64, and URL round trips; Base64 streaming encoder chunk boundaries; malformed decoder input and unchanged destinations on error. |

The tracked `seeds/` directory includes ASCII, embedded NULs, Unicode text,
combining marks, emoji, malformed UTF-8, escaped input, Base64, and all byte values.
The runner copies these seeds into writable corpora under `build/fuzz/fuzz/corpus/`.
New discoveries stay in that ignored build directory. It reuses those corpora on
later runs; the checked-in seeds are not modified.

## Longer runs and reproducing failures

After a smoke run has created the corpus directories, run one target longer:

```sh
build/fuzz/fuzz/fuzz_unicode build/fuzz/fuzz/corpus/unicode \
  -max_total_time=3600 -max_len=4096 -timeout=10 -rss_limit_mb=2048 \
  -artifact_prefix=build/fuzz/fuzz/artifacts/unicode/
```

Crash artifacts are saved under `build/fuzz/fuzz/artifacts/<target>/`. Reproduce
one with the same executable, for example:

```sh
build/fuzz/fuzz/fuzz_unicode build/fuzz/fuzz/artifacts/unicode/crash-HASH -runs=1
```

Keep useful minimized reproducers as named seeds and add a focused unit regression
test when fixing a library bug. The fuzz targets cover selected APIs; they do not
currently cover regex, formatting, every allocator failure, or every backend.

## CI

The `Fuzz` workflow runs each target for 30 seconds on Ubuntu 26.04 when the
library, fuzz files, build files, or workflow changes. It can also be started
manually. It uploads the evolved corpora and any crash artifacts for 30 days.
See [LLVM's libFuzzer documentation](https://llvm.org/docs/LibFuzzer.html) for more
runtime options.
