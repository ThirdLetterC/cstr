# cstr

[![Tests](https://github.com/ThirdLetterC/cstr/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/ThirdLetterC/cstr/actions/workflows/test.yml)
[![Benchmarks](https://github.com/ThirdLetterC/cstr/actions/workflows/benchmark.yml/badge.svg?branch=main)](https://github.com/ThirdLetterC/cstr/actions/workflows/benchmark.yml)
[![Fuzz](https://github.com/ThirdLetterC/cstr/actions/workflows/fuzz.yml/badge.svg?branch=main)](https://github.com/ThirdLetterC/cstr/actions/workflows/fuzz.yml)

`cstr` is a self-contained string library for ISO C11, C17, and C23.

This release is **1.3.0**.

Copy [cstr.h](cstr.h) and [cstr.c](cstr.c) into your project to use it. The default build needs no external libraries or data files at runtime.

The library provides owned strings, fixed buffers, string views, Unicode text handling, search, formatting, codecs, regular expressions, and hashing.

## Supported operating systems

| OS | Compiler | CI environment |
| --- | --- | --- |
| Linux | GCC, Clang | Ubuntu 26.04 |
| macOS | Apple Clang | macOS 15 (Apple Silicon) |
| Windows | MSVC | Windows Server 2022, x64 |
| FreeBSD | GCC, Clang | FreeBSD 14.1 VM |
| OpenBSD | Clang | OpenBSD 7.9 VM |

See the [build guide](docs/building.md) for platform-specific commands. Use
CMake on Windows and GNU Make (`gmake`) or CMake on FreeBSD and OpenBSD.

## Documentation

- [Build and test](docs/building.md): add the library to a project, run tests and benchmarks, and generate Unicode tables.
- [Memory and API contracts](docs/contracts.md): ownership, buffer sizes, errors, and thread safety.
- [API guide](docs/api.md): function groups, units, and performance costs.
- [Unicode data](docs/unicode.md): supported Unicode features, table layout, and reported binary sizes.
- [Benchmarks](docs/benchmarks.md): recorded results and the test environment.
- [Fuzzing](fuzz/README.md): build and run sanitizer-backed libFuzzer targets.

## License

The library code and algorithms use the MIT license. The bundled Unicode data uses Unicode License v3 (`Unicode-3.0`). See the license notices in [cstr.h](cstr.h) and [cstr.c](cstr.c).
