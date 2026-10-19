# Benchmarks

[Back to README](../README.md) · [Build and test](building.md)

These results were recorded in the original README. They have not been rerun as part of this documentation update.

## Test environment

- Linux x86-64 with two Intel KVM virtual CPUs.
- GCC 12.2.0.
- Compiler flags: `-std=c11 -O3 -Wall -Wextra -Werror -pedantic`.
- Automatically selected backend: `avx2`.

MiB/s measures throughput in 1,048,576-byte units per second; higher is better. Times are per operation; lower is better. `ns` means nanoseconds and `us` means microseconds.

## Scan and search results

| Operation and input | Size | `scalar` | `libc` | `sse2` | `avx2` | AVX2 speedup over scalar |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `is_ascii`, ASCII | 256 B | 1,599.8 MiB/s (152.6 ns) | 1,165.1 MiB/s (209.5 ns) | 8,290.1 MiB/s (29.4 ns) | 17,387.2 MiB/s (14.0 ns) | 10.9× |
| `is_ascii`, ASCII | 64 KiB | 1,469.8 MiB/s (42.5 us) | 1,699.2 MiB/s (36.8 us) | 9,805.7 MiB/s (6.4 us) | 36,581.8 MiB/s (1.71 us) | 24.9× |
| `is_ascii`, ASCII | 1 MiB | 1,587.0 MiB/s (630.1 us) | 1,478.8 MiB/s (676.2 us) | 6,067.1 MiB/s (164.8 us) | 29,207.1 MiB/s (34.2 us) | 18.4× |
| `validate_utf8`, ASCII | 4 KiB | 1,166.4 MiB/s (3,349.1 ns) | 903.9 MiB/s (4,321.3 ns) | 16,893.2 MiB/s (231.2 ns) | 31,894.7 MiB/s (122.5 ns) | 27.3× |
| `validate_utf8`, ASCII | 1 MiB | 979.2 MiB/s (1,021.3 us) | 1,040.7 MiB/s (960.9 us) | 17,728.7 MiB/s (56.4 us) | 35,589.3 MiB/s (28.1 us) | 36.3× |
| `validate_utf8`, Ukrainian UTF-8 | 64 KiB | 420.6 MiB/s (148.6 us) | 412.8 MiB/s (151.4 us) | 590.0 MiB/s (105.9 us) | 573.5 MiB/s (109.0 us) | 1.36× |
| `find_byte`, target byte at the end | 4 KiB | 2,175.8 MiB/s (1,795.3 ns) | 62,446.0 MiB/s (62.6 ns) | 16,210.7 MiB/s (241.0 ns) | 29,926.0 MiB/s (130.5 ns) | 13.8× |
| `find_substr`, Two-Way + SIMD | 64 KiB | 863.5 MiB/s (72.4 us) | 3,219.4 MiB/s (19.4 us) | 4,437.9 MiB/s (14.1 us) | 3,548.1 MiB/s (17.6 us) | 4.11× |

For the last three rows, SSE2 is 1.40× faster than scalar for Ukrainian UTF-8, libc is 28.7× faster for `find_byte`, and SSE2 is 5.14× faster for `find_substr`. The original README put the SSE2 speedups for Ukrainian UTF-8 and substring search in the AVX2 column; the table above uses the AVX2 throughput ratios.

## Append results

The append test used 10,000 fragments totaling 340 KiB.

| Method | Time per operation |
| :--- | ---: |
| Grow capacity as needed | 202.76 ns |
| Call `cs_reserve` before appending | 24.60 ns |

Reserving space ahead of time was about 8.2× faster in this run.

Run [the benchmark commands](building.md#run-benchmarks) to measure your own system.
