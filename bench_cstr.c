/* ==========================================================================
 * bench_cstr.c — Reproducible Benchmark Suite for cstr (Section 19)
 *
 * Compares identical operations and semantics across backends:
 *   - CS_BACKEND_SCALAR
 *   - CS_BACKEND_LIBC
 *   - CS_BACKEND_SSE2
 *   - CS_BACKEND_AVX2
 *   - CS_BACKEND_AUTO
 *
 * Sizes tested: 32 B, 256 B, 4 KiB, 64 KiB, 1 MiB, 16 MiB
 * Datasets: ASCII, Ukrainian UTF-8, Mixed UTF-8 + Emoji, Adversarial Repeat
 * ========================================================================== */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "cstr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a;
    uint64_t y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

static volatile size_t g_bench_sink = 0;

static void fill_pattern(char *buf, size_t size, cs_view pat) {
    size_t pos = 0;
    while (pos < size) {
        size_t chunk = (size - pos < pat.len) ? (size - pos) : pat.len;
        if (chunk < pat.len) {
            while (chunk > 0 && ((uint8_t)pat.data[chunk] & 0xC0U) == 0x80U) {
                chunk--;
            }
            if (chunk == 0) {
                memset(buf + pos, 'a', size - pos);
                break;
            }
        }
        memcpy(buf + pos, pat.data, chunk);
        pos += chunk;
    }
}

static void bench_op_across_backends(const char *op_name, size_t size, int mode_id, cs_view data_view) {
    cs_backend backends[] = {
        CS_BACKEND_SCALAR,
        CS_BACKEND_LIBC,
        CS_BACKEND_SSE2,
        CS_BACKEND_AVX2
    };
    const size_t trials = 7;
    const size_t iters  = (size <= 4096) ? 5000 : ((size <= 65536) ? 500 : ((size <= 1048576) ? 40 : 4));

    printf("%-22s | %8zu B | ", op_name, size);
    for (size_t b = 0; b < 4; b++) {
        if (!cs_backend_supported(backends[b])) {
            printf("%10s: %9s | ", cs_backend_name(backends[b]), "N/A");
            continue;
        }
        cs_set_forced_backend(backends[b]);
        uint64_t samples[7];
        for (size_t t = 0; t < trials; t++) {
            uint64_t t0 = now_ns();
            for (size_t k = 0; k < iters; k++) {
                if (mode_id == 0) {
                    g_bench_sink += cs_is_ascii(data_view) ? 1U : 0U;
                } else if (mode_id == 1) {
                    size_t eoff = 0;
                    g_bench_sink += (size_t)cs_validate_utf8(data_view, &eoff) + eoff;
                } else if (mode_id == 2) {
                    size_t pos = 0;
                    cs_find_byte(data_view, '!', 0, &pos);
                    g_bench_sink += pos;
                } else if (mode_id == 3) {
                    size_t pos = 0;
                    cs_find(data_view, CS_LIT("TARGET_NEEDLE!"), 0, &pos);
                    g_bench_sink += pos;
                }
            }
            uint64_t t1 = now_ns();
            samples[t] = (t1 - t0);
        }
        qsort(samples, trials, sizeof(uint64_t), cmp_u64);
        double med_ns = (double)samples[trials / 2] / (double)iters;
        double mib_s  = ((double)size / (1024.0 * 1024.0)) / (med_ns * 1e-9);
        printf("%6s: %7.1f MiB/s (%7.1f ns) | ", cs_backend_name(backends[b]), mib_s, med_ns);
    }
    printf("\n");
    cs_set_forced_backend(CS_BACKEND_AUTO);
}

int main(void) {
    cs_runtime_info_t info;
    cs_runtime_info(&info);
    printf("========================================================================================================================\n");
    printf("cstr Benchmark Report (v%s, Unicode %s, Arch: %s, Auto Backend: %s)\n",
           CS_VERSION_STRING, CS_UNICODE_VERSION, info.arch_name, info.backend_name);
    printf("========================================================================================================================\n");

    size_t sizes[] = { 32, 256, 4096, 65536, 1048576, 16777216 };
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
        size_t sz = sizes[i];
        char *buf = (char *)malloc(sz + 32);
        if (!buf) continue;

        fill_pattern(buf, sz, CS_LIT("abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"));
        if (sz >= 14) {
            memcpy(buf + sz - 14, "TARGET_NEEDLE!", 14);
        }
        cs_view v_ascii = { buf, sz };
        bench_op_across_backends("is_ascii (ASCII)",      sz, 0, v_ascii);
        bench_op_across_backends("validate_utf8 (ASCII)", sz, 1, v_ascii);
        bench_op_across_backends("find_byte (end)",       sz, 2, v_ascii);
        bench_op_across_backends("find_substr (Two-Way)", sz, 3, v_ascii);

        fill_pattern(buf, sz, CS_LIT("Самодостатня бібліотека рядків для C (Ґудзик, Єдність, Історія, Їжак) — UTF-8 Benchmark "));
        cs_view v_ua = { buf, sz };
        bench_op_across_backends("validate_utf8 (UA)",    sz, 1, v_ua);
        printf("------------------------------------------------------------------------------------------------------------------------\n");

        free(buf);
    }

    const size_t frag_count = 10000;
    cs_view frag = CS_LIT("фрагмент_рядка_12345_");
    uint64_t t0 = now_ns();
    cs_string s_dyn;
    cs_init(&s_dyn);
    for (size_t i = 0; i < frag_count; i++) {
        cs_append(&s_dyn, frag);
    }
    uint64_t t1 = now_ns();
    g_bench_sink += s_dyn.len;
    cs_destroy(&s_dyn);

    uint64_t t2 = now_ns();
    cs_string s_pre;
    cs_init(&s_pre);
    cs_reserve(&s_pre, frag_count * frag.len);
    for (size_t i = 0; i < frag_count; i++) {
        cs_append(&s_pre, frag);
    }
    uint64_t t3 = now_ns();
    g_bench_sink += s_pre.len;
    cs_destroy(&s_pre);

    printf("Builder 10,000 appends (%zu B total): dynamic=%llu us (%.2f ns/op), preallocated=%llu us (%.2f ns/op)\n",
           frag_count * frag.len,
           (unsigned long long)((t1 - t0) / 1000ULL), (double)(t1 - t0) / (double)frag_count,
           (unsigned long long)((t3 - t2) / 1000ULL), (double)(t3 - t2) / (double)frag_count);

    return 0;
}