/* ==========================================================================
 * cstr.c — Self-Contained String & Unicode Library for ISO C11/C17/C23
 *
 * SPDX-License-Identifier: MIT AND Unicode-3.0
 * Copyright (c) 2026 cstr Contributors
 *
 * Includes embedded Unicode 17.0.0 Character Database tables under the
 * Unicode License v3 (see full notice in the Unicode Tables section below).
 * ========================================================================== */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "cstr.h"

#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <bcrypt.h>
#elif defined(__linux__) || defined(__APPLE__) || defined(__unix__)
#include <locale.h>
#if defined(__APPLE__)
#include <xlocale.h>
#endif
#include <fcntl.h>
#include <unistd.h>
#endif

/* ==========================================================================
 * Architecture & SIMD Intrinsics Headers (Guarded for Baseline Compilation)
 * ========================================================================== */

#if CS_ENABLE_SIMD && !CS_FORCE_SCALAR
#if defined(__x86_64__) || defined(_M_X64)
#define CS_ARCH_X86_64 1
#if defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#include <immintrin.h>
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
#define CS_ARCH_ARM64 1
#include <arm_neon.h>
#if defined(__linux__)
#include <sys/auxv.h>
#endif
#endif
#endif

#ifndef CS_ARCH_X86_64
#define CS_ARCH_X86_64 0
#endif
#ifndef CS_ARCH_ARM64
#define CS_ARCH_ARM64 0
#endif

#if (defined(__GNUC__) || defined(__clang__)) && CS_ARCH_X86_64
#define CS_TARGET_SSE2   __attribute__((target("sse2")))
#define CS_TARGET_AVX2   __attribute__((target("avx2")))
#define CS_TARGET_AVX512 __attribute__((target("avx512bw,avx512vl")))
#else
#define CS_TARGET_SSE2
#define CS_TARGET_AVX2
#define CS_TARGET_AVX512
#endif

/* ==========================================================================
 * Internal Overflow-Checked Arithmetic Helpers
 * ========================================================================== */

static inline bool cs_add_overflow_sz(size_t a, size_t b, size_t *out) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_add_overflow(a, b, out);
#else
    if (SIZE_MAX - a < b) return true;
    *out = a + b;
    return false;
#endif
}

static inline bool cs_mul_overflow_sz(size_t a, size_t b, size_t *out) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_mul_overflow(a, b, out);
#else
    if (a != 0 && SIZE_MAX / a < b) return true;
    *out = a * b;
    return false;
#endif
}

#if defined(__cplusplus)
#define CS_ALIGNOF(T) alignof(T)
#else
#define CS_ALIGNOF(T) _Alignof(T)
#endif

/* Internal static empty NUL sentinel so cs_c_str() is always non-NULL */
static const char cs_empty_cstr_sentinel[1] = { '\0' };

/* ==========================================================================
 * Section 15: Runtime CPU Dispatch & Backend Selection
 * ========================================================================== */

static atomic_int cs_g_init_state = 0; /* 0 = uninit, 1 = in-progress, 2 = done */
static uint32_t   cs_g_supported_mask = (1U << CS_BACKEND_SCALAR) | (1U << CS_BACKEND_LIBC);
static cs_backend cs_g_auto_backend   = CS_BACKEND_SCALAR;
static atomic_int cs_g_forced_backend = CS_BACKEND_AUTO;

#if CS_ARCH_X86_64 && CS_ENABLE_SIMD && !CS_FORCE_SCALAR
static inline uint64_t cs_xgetbv_u64(uint32_t idx) {
#if defined(_MSC_VER)
    return _xgetbv(idx);
#else
    uint32_t eax = 0, edx = 0;
    __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(idx));
    return ((uint64_t)edx << 32) | eax;
#endif
}

static void cs_detect_x86_features(uint32_t *out_mask, cs_backend *out_best) {
    uint32_t mask = (1U << CS_BACKEND_SCALAR) | (1U << CS_BACKEND_LIBC) | (1U << CS_BACKEND_SSE2);
    cs_backend best = CS_BACKEND_SSE2;

#if defined(__GNUC__) || defined(__clang__)
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    unsigned int max_leaf = __get_cpuid_max(0, NULL);
    if (max_leaf >= 7) {
        __cpuid_count(1, 0, eax, ebx, ecx, edx);
        /* Bit 27 = OSXSAVE, Bit 28 = AVX */
        bool osxsave = (ecx & (1U << 27)) != 0;
        bool avx     = (ecx & (1U << 28)) != 0;
        if (osxsave && avx) {
            uint64_t xcr0 = cs_xgetbv_u64(0);
            /* XMM (bit 1) and YMM (bit 2) state must be enabled by OS */
            if ((xcr0 & 0x6ULL) == 0x6ULL) {
                unsigned int eax7 = 0, ebx7 = 0, ecx7 = 0, edx7 = 0;
                __cpuid_count(7, 0, eax7, ebx7, ecx7, edx7);
                /* EBX bit 5 = AVX2 */
                if ((ebx7 & (1U << 5)) != 0) {
                    mask |= (1U << CS_BACKEND_AVX2);
                    best = CS_BACKEND_AVX2;
                }
                /* Check AVX-512BW (bit 30) + AVX-512VL (bit 31) + AVX-512F (bit 16) + ZMM OS state (bits 5..7) */
                if ((xcr0 & 0xE6ULL) == 0xE6ULL) {
                    const uint32_t avx512_req = (1U << 16) | (1U << 30) | (1U << 31);
                    if ((ebx7 & avx512_req) == avx512_req) {
                        mask |= (1U << CS_BACKEND_AVX512);
                        /* Per Section 15: AVX-512 is optional and not automatically preferred over AVX2 */
                    }
                }
            }
        }
    }
#elif defined(_MSC_VER)
    int info[4] = {0};
    __cpuid(info, 0);
    if (info[0] >= 7) {
        __cpuid(info, 1);
        bool osxsave = (info[2] & (1 << 27)) != 0;
        bool avx     = (info[2] & (1 << 28)) != 0;
        if (osxsave && avx) {
            uint64_t xcr0 = cs_xgetbv_u64(0);
            if ((xcr0 & 0x6ULL) == 0x6ULL) {
                int info7[4] = {0};
                __cpuidex(info7, 7, 0);
                if ((info7[1] & (1 << 5)) != 0) {
                    mask |= (1U << CS_BACKEND_AVX2);
                    best = CS_BACKEND_AVX2;
                }
            }
        }
    }
#endif
    *out_mask = mask;
    *out_best = best;
}
#endif

static void cs_ensure_dispatch_init(void) {
    int state = atomic_load_explicit(&cs_g_init_state, memory_order_acquire);
    if (state == 2) return;
    int expected = 0;
    if (atomic_compare_exchange_strong_explicit(&cs_g_init_state, &expected, 1,
                                                memory_order_acq_rel, memory_order_acquire)) {
        uint32_t mask = (1U << CS_BACKEND_SCALAR) | (1U << CS_BACKEND_LIBC);
        cs_backend best = CS_BACKEND_LIBC;
#if CS_FORCE_SCALAR || !CS_ENABLE_SIMD
        mask = (1U << CS_BACKEND_SCALAR) | (1U << CS_BACKEND_LIBC);
        best = CS_BACKEND_SCALAR;
#elif CS_ARCH_X86_64
        cs_detect_x86_features(&mask, &best);
#elif CS_ARCH_ARM64
        mask |= (1U << CS_BACKEND_NEON);
        best = CS_BACKEND_NEON;
#endif
        cs_g_supported_mask = mask;
        cs_g_auto_backend   = best;
        atomic_store_explicit(&cs_g_init_state, 2, memory_order_release);
    } else {
        while (atomic_load_explicit(&cs_g_init_state, memory_order_acquire) != 2) {
            /* Spin briefly until initialization finishes */
        }
    }
}

static inline cs_backend cs_active_backend(void) {
    cs_ensure_dispatch_init();
    cs_backend forced = (cs_backend)atomic_load_explicit(&cs_g_forced_backend, memory_order_relaxed);
    return (forced != CS_BACKEND_AUTO) ? forced : cs_g_auto_backend;
}

const char *cs_backend_name(cs_backend backend) {
    switch (backend) {
        case CS_BACKEND_AUTO:   return "auto";
        case CS_BACKEND_SCALAR: return "scalar";
        case CS_BACKEND_LIBC:   return "libc";
        case CS_BACKEND_SSE2:   return "sse2";
        case CS_BACKEND_AVX2:   return "avx2";
        case CS_BACKEND_AVX512: return "avx512bw+vl";
        case CS_BACKEND_NEON:   return "neon";
        default:                return "unknown";
    }
}

void cs_runtime_info(cs_runtime_info_t *out_info) {
    if (!out_info) return;
    cs_ensure_dispatch_init();
    cs_backend active = cs_active_backend();
    out_info->active_backend = active;
    out_info->forced_backend = (cs_backend)atomic_load_explicit(&cs_g_forced_backend, memory_order_relaxed);
    out_info->supported_mask = cs_g_supported_mask;
#if CS_ARCH_X86_64
    out_info->arch_name = "x86_64";
#elif CS_ARCH_ARM64
    out_info->arch_name = "aarch64";
#else
    out_info->arch_name = "generic";
#endif
    out_info->backend_name = cs_backend_name(active);
}

bool cs_backend_supported(cs_backend backend) {
    cs_ensure_dispatch_init();
    if (backend == CS_BACKEND_AUTO) return true;
    if ((int)backend < 0 || (int)backend > 6) return false;
#if CS_FORCE_SCALAR
    if (backend != CS_BACKEND_SCALAR) return false;
#endif
    return (cs_g_supported_mask & (1U << (uint32_t)backend)) != 0;
}

cs_status cs_set_forced_backend(cs_backend backend) {
    cs_ensure_dispatch_init();
    if (backend != CS_BACKEND_AUTO && !cs_backend_supported(backend)) {
        return CS_EUNSUPPORTED;
    }
    atomic_store_explicit(&cs_g_forced_backend, (int)backend, memory_order_relaxed);
    return CS_OK;
}

const char *cs_status_str(cs_status status) {
    switch (status) {
        case CS_OK:           return "CS_OK: success";
        case CS_END:          return "CS_END: end of iteration";
        case CS_NOT_FOUND:    return "CS_NOT_FOUND: pattern not found";
        case CS_EINVAL:       return "CS_EINVAL: invalid argument";
        case CS_ENOMEM:       return "CS_ENOMEM: out of memory";
        case CS_ERANGE:       return "CS_ERANGE: out of range";
        case CS_EOVERFLOW:    return "CS_EOVERFLOW: arithmetic overflow";
        case CS_EUTF8:        return "CS_EUTF8: invalid UTF-8 sequence";
        case CS_EBUFFER:      return "CS_EBUFFER: fixed buffer too small";
        case CS_EUNSUPPORTED: return "CS_EUNSUPPORTED: unsupported operation or backend";
        case CS_ELIMIT:       return "CS_ELIMIT: resource limit exceeded";
        case CS_ESYNTAX:      return "CS_ESYNTAX: syntax error";
        default:              return "CS_UNKNOWN: unknown status";
    }
}

/* ==========================================================================
 * Safe Scalar & SIMD Primitives (Bounds-Checked, Guard-Page Safe)
 * ========================================================================== */

static bool cs_is_ascii_scalar(const uint8_t *p, size_t len) {
    for (size_t i = 0; i < len; i++) {
        if (p[i] & 0x80U) return false;
    }
    return true;
}

static size_t cs_find_byte_scalar(const uint8_t *p, size_t len, uint8_t needle) {
    for (size_t i = 0; i < len; i++) {
        if (p[i] == needle) return i;
    }
    return SIZE_MAX;
}

static size_t cs_rfind_byte_scalar(const uint8_t *p, size_t len, uint8_t needle) {
    for (size_t i = len; i > 0; i--) {
        if (p[i - 1] == needle) return i - 1;
    }
    return SIZE_MAX;
}

static bool cs_equal_scalar(const uint8_t *a, const uint8_t *b, size_t len) {
    for (size_t i = 0; i < len; i++) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

static void cs_ascii_case_convert_scalar(uint8_t *dst, const uint8_t *src, size_t len, bool upper) {
    if (upper) {
        for (size_t i = 0; i < len; i++) {
            uint8_t c = src[i];
            dst[i] = (c >= 'a' && c <= 'z') ? (uint8_t)(c - 32U) : c;
        }
    } else {
        for (size_t i = 0; i < len; i++) {
            uint8_t c = src[i];
            dst[i] = (c >= 'A' && c <= 'Z') ? (uint8_t)(c + 32U) : c;
        }
    }
}

#if CS_ARCH_X86_64 && CS_ENABLE_SIMD && !CS_FORCE_SCALAR
CS_TARGET_SSE2
static bool cs_is_ascii_sse2(const uint8_t *p, size_t len) {
    size_t i = 0;
    while (i + 16 <= len) {
        __m128i v = _mm_loadu_si128((const __m128i *)(const void *)(p + i));
        if (_mm_movemask_epi8(v) != 0) return false;
        i += 16;
    }
    return cs_is_ascii_scalar(p + i, len - i);
}

CS_TARGET_AVX2
static bool cs_is_ascii_avx2(const uint8_t *p, size_t len) {
    size_t i = 0;
    while (i + 32 <= len) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)(p + i));
        if (_mm256_movemask_epi8(v) != 0) return false;
        i += 32;
    }
    return cs_is_ascii_sse2(p + i, len - i);
}

CS_TARGET_SSE2
static size_t cs_find_byte_sse2(const uint8_t *p, size_t len, uint8_t needle) {
    size_t i = 0;
    __m128i vn = _mm_set1_epi8((char)needle);
    while (i + 16 <= len) {
        __m128i v = _mm_loadu_si128((const __m128i *)(const void *)(p + i));
        __m128i eq = _mm_cmpeq_epi8(v, vn);
        int mask = _mm_movemask_epi8(eq);
        if (mask != 0) {
            return i + (size_t)__builtin_ctz((unsigned int)mask);
        }
        i += 16;
    }
    size_t tail = cs_find_byte_scalar(p + i, len - i, needle);
    return (tail == SIZE_MAX) ? SIZE_MAX : (i + tail);
}

CS_TARGET_AVX2
static size_t cs_find_byte_avx2(const uint8_t *p, size_t len, uint8_t needle) {
    size_t i = 0;
    __m256i vn = _mm256_set1_epi8((char)needle);
    while (i + 32 <= len) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)(p + i));
        __m256i eq = _mm256_cmpeq_epi8(v, vn);
        int mask = _mm256_movemask_epi8(eq);
        if (mask != 0) {
            return i + (size_t)__builtin_ctz((unsigned int)mask);
        }
        i += 32;
    }
    size_t tail = cs_find_byte_sse2(p + i, len - i, needle);
    return (tail == SIZE_MAX) ? SIZE_MAX : (i + tail);
}

CS_TARGET_SSE2
static size_t cs_rfind_byte_sse2(const uint8_t *p, size_t len, uint8_t needle) {
    size_t i = len;
    __m128i vn = _mm_set1_epi8((char)needle);
    while (i >= 16) {
        i -= 16;
        __m128i v = _mm_loadu_si128((const __m128i *)(const void *)(p + i));
        __m128i eq = _mm_cmpeq_epi8(v, vn);
        unsigned int mask = (unsigned int)_mm_movemask_epi8(eq);
        if (mask != 0) {
            return i + (size_t)(31 - __builtin_clz(mask));
        }
    }
    return cs_rfind_byte_scalar(p, i, needle);
}

CS_TARGET_AVX2
static size_t cs_rfind_byte_avx2(const uint8_t *p, size_t len, uint8_t needle) {
    size_t i = len;
    __m256i vn = _mm256_set1_epi8((char)needle);
    while (i >= 32) {
        i -= 32;
        __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)(p + i));
        __m256i eq = _mm256_cmpeq_epi8(v, vn);
        unsigned int mask = (unsigned int)_mm256_movemask_epi8(eq);
        if (mask != 0) {
            return i + (size_t)(31 - __builtin_clz(mask));
        }
    }
    return cs_rfind_byte_sse2(p, i, needle);
}

CS_TARGET_SSE2
static bool cs_equal_sse2(const uint8_t *a, const uint8_t *b, size_t len) {
    size_t i = 0;
    while (i + 16 <= len) {
        __m128i va = _mm_loadu_si128((const __m128i *)(const void *)(a + i));
        __m128i vb = _mm_loadu_si128((const __m128i *)(const void *)(b + i));
        __m128i eq = _mm_cmpeq_epi8(va, vb);
        if (_mm_movemask_epi8(eq) != 0xFFFF) return false;
        i += 16;
    }
    return cs_equal_scalar(a + i, b + i, len - i);
}

CS_TARGET_AVX2
static bool cs_equal_avx2(const uint8_t *a, const uint8_t *b, size_t len) {
    size_t i = 0;
    while (i + 32 <= len) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(const void *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(const void *)(b + i));
        __m256i eq = _mm256_cmpeq_epi8(va, vb);
        if ((uint32_t)_mm256_movemask_epi8(eq) != 0xFFFFFFFFU) return false;
        i += 32;
    }
    return cs_equal_sse2(a + i, b + i, len - i);
}
#endif /* CS_ARCH_X86_64 */

#if CS_ARCH_ARM64 && CS_ENABLE_SIMD && !CS_FORCE_SCALAR
static bool cs_is_ascii_neon(const uint8_t *p, size_t len) {
    size_t i = 0;
    while (i + 16 <= len) {
        uint8x16_t v = vld1q_u8(p + i);
        if (vmaxvq_u8(v) >= 0x80U) return false;
        i += 16;
    }
    return cs_is_ascii_scalar(p + i, len - i);
}

static size_t cs_find_byte_neon(const uint8_t *p, size_t len, uint8_t needle) {
    size_t i = 0;
    uint8x16_t vn = vdupq_n_u8(needle);
    while (i + 16 <= len) {
        uint8x16_t v = vld1q_u8(p + i);
        uint8x16_t eq = vceqq_u8(v, vn);
        if (vmaxvq_u8(eq) != 0) {
            for (size_t k = 0; k < 16; k++) {
                if (p[i + k] == needle) return i + k;
            }
        }
        i += 16;
    }
    size_t tail = cs_find_byte_scalar(p + i, len - i, needle);
    return (tail == SIZE_MAX) ? SIZE_MAX : (i + tail);
}
#endif /* CS_ARCH_ARM64 */

static bool cs_dispatch_is_ascii(const uint8_t *p, size_t len) {
    if (len == 0) return true;
    cs_backend b = cs_active_backend();
    if (b == CS_BACKEND_SCALAR || len < 16) {
        return cs_is_ascii_scalar(p, len);
    }
#if CS_ARCH_X86_64 && CS_ENABLE_SIMD && !CS_FORCE_SCALAR
    if ((b == CS_BACKEND_AVX2 || b == CS_BACKEND_AVX512) && len >= 32) {
        return cs_is_ascii_avx2(p, len);
    }
    if (b == CS_BACKEND_SSE2 || b == CS_BACKEND_AVX2 || b == CS_BACKEND_AVX512) {
        return cs_is_ascii_sse2(p, len);
    }
#elif CS_ARCH_ARM64 && CS_ENABLE_SIMD && !CS_FORCE_SCALAR
    if (b == CS_BACKEND_NEON) {
        return cs_is_ascii_neon(p, len);
    }
#endif
    return cs_is_ascii_scalar(p, len);
}

static size_t cs_dispatch_find_byte(const uint8_t *p, size_t len, uint8_t needle) {
    if (len == 0) return SIZE_MAX;
    cs_backend b = cs_active_backend();
    if (b == CS_BACKEND_SCALAR) {
        return cs_find_byte_scalar(p, len, needle);
    }
    if (b == CS_BACKEND_LIBC) {
        const void *found = memchr(p, needle, len);
        return found ? (size_t)((const uint8_t *)found - p) : SIZE_MAX;
    }
#if CS_ARCH_X86_64 && CS_ENABLE_SIMD && !CS_FORCE_SCALAR
    if (b == CS_BACKEND_AVX2 || b == CS_BACKEND_AVX512) {
        return cs_find_byte_avx2(p, len, needle);
    }
    if (b == CS_BACKEND_SSE2) {
        return cs_find_byte_sse2(p, len, needle);
    }
#elif CS_ARCH_ARM64 && CS_ENABLE_SIMD && !CS_FORCE_SCALAR
    if (b == CS_BACKEND_NEON) {
        return cs_find_byte_neon(p, len, needle);
    }
#endif
    const void *found = memchr(p, needle, len);
    return found ? (size_t)((const uint8_t *)found - p) : SIZE_MAX;
}

static size_t cs_dispatch_rfind_byte(const uint8_t *p, size_t len, uint8_t needle) {
    if (len == 0) return SIZE_MAX;
    cs_backend b = cs_active_backend();
    if (b == CS_BACKEND_SCALAR) {
        return cs_rfind_byte_scalar(p, len, needle);
    }
#if defined(__GLIBC__)
    if (b == CS_BACKEND_LIBC) {
        const void *found = memrchr(p, needle, len);
        return found ? (size_t)((const uint8_t *)found - p) : SIZE_MAX;
    }
#endif
#if CS_ARCH_X86_64 && CS_ENABLE_SIMD && !CS_FORCE_SCALAR
    if (b == CS_BACKEND_AVX2 || b == CS_BACKEND_AVX512) {
        return cs_rfind_byte_avx2(p, len, needle);
    }
    if (b == CS_BACKEND_SSE2) {
        return cs_rfind_byte_sse2(p, len, needle);
    }
#endif
    return cs_rfind_byte_scalar(p, len, needle);
}

static bool cs_dispatch_equal(const uint8_t *a, const uint8_t *b, size_t len) {
    if (len == 0) return true;
    if (a == b) return true;
    cs_backend bk = cs_active_backend();
    if (bk == CS_BACKEND_SCALAR) {
        return cs_equal_scalar(a, b, len);
    }
    if (bk == CS_BACKEND_LIBC) {
        return memcmp(a, b, len) == 0;
    }
#if CS_ARCH_X86_64 && CS_ENABLE_SIMD && !CS_FORCE_SCALAR
    if (bk == CS_BACKEND_AVX2 || bk == CS_BACKEND_AVX512) {
        return cs_equal_avx2(a, b, len);
    }
    if (bk == CS_BACKEND_SSE2) {
        return cs_equal_sse2(a, b, len);
    }
#endif
    return memcmp(a, b, len) == 0;
}

/* ==========================================================================
 * Section 2 & 5: Allocator, View, String, Buffer & Builder Core
 * ========================================================================== */

#if !CS_NO_HEAP
#if CS_USE_MIMALLOC
#include <mimalloc.h>

static void *cs_mi_alloc(void *ctx, size_t size, size_t align) {
    if (size == 0) size = 1;
    if (align == 0) align = 1;
    if (ctx) {
        return mi_heap_malloc_aligned((mi_heap_t *)ctx, size, align);
    }
    return mi_malloc_aligned(size, align);
}

static void *cs_mi_realloc(void *ctx, void *ptr, size_t old_size, size_t new_size, size_t align) {
    if (align == 0) align = 1;
    if (new_size == 0) {
        if (ptr) mi_free_size_aligned(ptr, old_size, align);
        return NULL;
    }
    if (ctx) {
        return mi_heap_realloc_aligned((mi_heap_t *)ctx, ptr, new_size, align);
    }
    return mi_realloc_aligned(ptr, new_size, align);
}

static void cs_mi_free(void *ctx, void *ptr, size_t size, size_t align) {
    (void)ctx;
    if (!ptr) return;
    if (align == 0) align = 1;
    mi_free_size_aligned(ptr, size, align);
}

cs_allocator cs_mimalloc_allocator(void *mi_heap) {
    cs_allocator a;
    a.alloc   = cs_mi_alloc;
    a.realloc = cs_mi_realloc;
    a.free    = cs_mi_free;
    a.ctx     = mi_heap;
    return a;
}
#else /* !CS_USE_MIMALLOC */

static void *cs_std_alloc(void *ctx, size_t size, size_t align) {
    (void)ctx;
    if (size == 0) size = 1;
    if (align > CS_ALIGNOF(max_align_t)) {
#if defined(_MSC_VER)
        return _aligned_malloc(size, align);
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
        size_t mask = align - 1U;
        if ((align & mask) == 0U) {
            size_t rounded = (size + mask) & ~mask;
            return aligned_alloc(align, rounded);
        }
#endif
    }
    return malloc(size);
}

static void *cs_std_realloc(void *ctx, void *ptr, size_t old_size, size_t new_size, size_t align) {
    (void)ctx;
    (void)old_size;
    (void)align;
    if (new_size == 0) {
        free(ptr);
        return NULL;
    }
    return realloc(ptr, new_size);
}

static void cs_std_free(void *ctx, void *ptr, size_t size, size_t align) {
    (void)ctx;
    (void)size;
#if defined(_MSC_VER)
    if (align > CS_ALIGNOF(max_align_t)) {
        _aligned_free(ptr);
        return;
    }
#else
    (void)align;
#endif
    free(ptr);
}
#endif /* CS_USE_MIMALLOC */
#else
static void *cs_null_alloc(void *ctx, size_t size, size_t align) {
    (void)ctx; (void)size; (void)align;
    return NULL;
}

static void *cs_null_realloc(void *ctx, void *ptr, size_t old_size, size_t new_size, size_t align) {
    (void)ctx; (void)ptr; (void)old_size; (void)new_size; (void)align;
    return NULL;
}

static void cs_null_free(void *ctx, void *ptr, size_t size, size_t align) {
    (void)ctx; (void)ptr; (void)size; (void)align;
}
#endif

static atomic_int   cs_g_custom_alloc_set = 0;
static cs_allocator cs_g_custom_alloc     = { NULL, NULL, NULL, NULL };

static cs_allocator cs_builtin_default_allocator(void) {
    cs_allocator a;
#if CS_NO_HEAP
    a.alloc   = cs_null_alloc;
    a.realloc = cs_null_realloc;
    a.free    = cs_null_free;
    a.ctx     = NULL;
#elif CS_USE_MIMALLOC
    a.alloc   = cs_mi_alloc;
    a.realloc = cs_mi_realloc;
    a.free    = cs_mi_free;
    a.ctx     = NULL;
#else
    a.alloc   = cs_std_alloc;
    a.realloc = cs_std_realloc;
    a.free    = cs_std_free;
    a.ctx     = NULL;
#endif
    return a;
}

cs_allocator cs_default_allocator(void) {
    if (atomic_load_explicit(&cs_g_custom_alloc_set, memory_order_acquire) != 0) {
        return cs_g_custom_alloc;
    }
    return cs_builtin_default_allocator();
}

cs_status cs_set_default_allocator(cs_allocator alloc) {
    if (!alloc.alloc || !alloc.free) return CS_EINVAL;
    cs_g_custom_alloc = alloc;
    atomic_store_explicit(&cs_g_custom_alloc_set, 1, memory_order_release);
    return CS_OK;
}

void cs_reset_default_allocator(void) {
    atomic_store_explicit(&cs_g_custom_alloc_set, 0, memory_order_release);
}

void cs_workspace_init(cs_workspace *ws, void *mem, size_t size) {
    if (!ws) return;
    ws->mem  = mem;
    ws->size = mem ? size : 0;
    ws->used = 0;
}

void cs_workspace_reset(cs_workspace *ws) {
    if (!ws) return;
    ws->used = 0;
}

#if CS_ENABLE_UNICODE
static void *cs_ws_alloc(cs_workspace *ws, size_t size, size_t align) {
    if (!ws || !ws->mem) return NULL;
    if (align == 0) align = sizeof(void *);
    size_t mask = align - 1;
    size_t aligned_used = (ws->used + mask) & ~mask;
    size_t next_used = 0;
    if (cs_add_overflow_sz(aligned_used, size, &next_used) || next_used > ws->size) {
        return NULL;
    }
    void *ptr = (uint8_t *)ws->mem + aligned_used;
    ws->used = next_used;
    return ptr;
}
#endif

cs_view cs_sv(const char *cstr) {
    if (!cstr) return cs_empty_view();
    cs_view v = { cstr, strlen(cstr) };
    return v;
}

cs_view cs_sv_from_bytes(const char *data, size_t len) {
    if (!data || len == 0) {
        cs_view v = { data, 0 };
        return v;
    }
    cs_view v = { data, len };
    return v;
}

cs_status cs_view_from_cstr(const char *cstr, cs_view *out) {
    if (!out) return CS_EINVAL;
    if (!cstr) {
        *out = cs_empty_view();
        return CS_OK;
    }
    out->data = cstr;
    out->len  = strlen(cstr);
    return CS_OK;
}

cs_status cs_view_from_cstr_n(const char *cstr, size_t max_bytes, cs_view *out) {
    if (!out) return CS_EINVAL;
    if (!cstr) {
        if (max_bytes == 0) {
            *out = cs_empty_view();
            return CS_OK;
        }
        return CS_EINVAL;
    }
    const void *nul = memchr(cstr, '\0', max_bytes);
    if (!nul) {
        return CS_ERANGE;
    }
    out->data = cstr;
    out->len  = (size_t)((const char *)nul - cstr);
    return CS_OK;
}

cs_status cs_view_from_bytes(const char *data, size_t len, cs_view *out) {
    if (!out) return CS_EINVAL;
    if (!data && len > 0) return CS_EINVAL;
    out->data = data;
    out->len  = len;
    return CS_OK;
}

void cs_init(cs_string *s) {
    if (!s) return;
    s->data       = (char *)(uintptr_t)cs_empty_cstr_sentinel;
    s->len        = 0;
    s->cap        = 0;
    s->alloc      = cs_default_allocator();
    s->generation = 1;
    s->flags      = 0;
}

cs_status cs_init_with_allocator(cs_string *s, cs_allocator alloc) {
    if (!s) return CS_EINVAL;
    if (!alloc.alloc || !alloc.free) return CS_EINVAL;
    s->data       = (char *)(uintptr_t)cs_empty_cstr_sentinel;
    s->len        = 0;
    s->cap        = 0;
    s->alloc      = alloc;
    s->generation = 1;
    s->flags      = 0;
    return CS_OK;
}

static inline bool cs_string_is_heap(const cs_string *s) {
    return s && s->cap > 0 && s->data != NULL && s->data != cs_empty_cstr_sentinel;
}

cs_status cs_reserve(cs_string *s, size_t min_cap) {
    if (!s) return CS_EINVAL;
    if (!s->alloc.alloc) s->alloc = cs_default_allocator();
    if (min_cap <= s->cap) return CS_OK;

    /* Geometric growth with overflow check */
    size_t new_cap = s->cap;
    if (new_cap == 0) {
        new_cap = 15;
    }
    while (new_cap < min_cap) {
        size_t grown = 0;
        if (cs_add_overflow_sz(new_cap, (new_cap >> 1) + 8U, &grown) || grown < min_cap) {
            new_cap = min_cap;
            break;
        }
        new_cap = grown;
    }

    size_t alloc_bytes = 0;
    if (cs_add_overflow_sz(new_cap, 1U, &alloc_bytes)) {
        return CS_EOVERFLOW;
    }

    char *new_data = NULL;
    if (cs_string_is_heap(s)) {
        size_t old_bytes = s->cap + 1U;
        if (s->alloc.realloc) {
            new_data = (char *)s->alloc.realloc(s->alloc.ctx, s->data, old_bytes, alloc_bytes, 1U);
        } else {
            new_data = (char *)s->alloc.alloc(s->alloc.ctx, alloc_bytes, 1U);
            if (new_data) {
                memcpy(new_data, s->data, s->len + 1U);
                s->alloc.free(s->alloc.ctx, s->data, old_bytes, 1U);
            }
        }
    } else {
        new_data = (char *)s->alloc.alloc(s->alloc.ctx, alloc_bytes, 1U);
        if (new_data) {
            new_data[0] = '\0';
        }
    }

    if (!new_data) {
        return CS_ENOMEM;
    }

    s->data = new_data;
    s->cap  = new_cap;
    s->generation++;
    return CS_OK;
}

cs_status cs_from_view(cs_string *s, cs_view v, const cs_allocator *alloc) {
    if (!s || !cs_view_is_valid(v)) return CS_EINVAL;
    if (alloc) {
        cs_status st = cs_init_with_allocator(s, *alloc);
        if (st != CS_OK) return st;
    } else {
        cs_init(s);
    }
    if (v.len == 0) return CS_OK;
    cs_status st = cs_reserve(s, v.len);
    if (st != CS_OK) return st;
    memcpy(s->data, v.data, v.len);
    s->len = v.len;
    s->data[s->len] = '\0';
    return CS_OK;
}

cs_status cs_from_cstr(cs_string *s, const char *cstr, const cs_allocator *alloc) {
    if (!s || !cstr) return CS_EINVAL;
    return cs_from_view(s, cs_sv(cstr), alloc);
}

cs_status cs_from_bytes(cs_string *s, const char *data, size_t len, const cs_allocator *alloc) {
    if (!data && len > 0) return CS_EINVAL;
    cs_view v = { data, len };
    return cs_from_view(s, v, alloc);
}

cs_status cs_clone(cs_string *dst, const cs_string *src) {
    if (!dst || !src || dst == src) return CS_EINVAL;
    return cs_from_view(dst, cs_as_view(src), &src->alloc);
}

void cs_move(cs_string *dst, cs_string *src) {
    if (!dst || !src) return;
    if (dst == src) return;
    *dst = *src;
    dst->generation++;
    src->data       = (char *)(uintptr_t)cs_empty_cstr_sentinel;
    src->len        = 0;
    src->cap        = 0;
    src->generation++;
}

void cs_destroy(cs_string *s) {
    if (!s) return;
    if (cs_string_is_heap(s) && s->alloc.free) {
        s->alloc.free(s->alloc.ctx, s->data, s->cap + 1U, 1U);
    }
    s->data = (char *)(uintptr_t)cs_empty_cstr_sentinel;
    s->len  = 0;
    s->cap  = 0;
    s->generation++;
}

void cs_clear(cs_string *s) {
    if (!s) return;
    s->len = 0;
    if (cs_string_is_heap(s)) {
        s->data[0] = '\0';
    } else {
        s->data = (char *)(uintptr_t)cs_empty_cstr_sentinel;
    }
    s->generation++;
}

void cs_reset(cs_string *s) {
    cs_destroy(s);
}

cs_status cs_shrink_to_fit(cs_string *s) {
    if (!s) return CS_EINVAL;
    if (!cs_string_is_heap(s) || s->cap == s->len) return CS_OK;
    if (s->len == 0) {
        cs_destroy(s);
        return CS_OK;
    }
    size_t new_bytes = s->len + 1U;
    size_t old_bytes = s->cap + 1U;
    char *new_data = NULL;
    if (s->alloc.realloc) {
        new_data = (char *)s->alloc.realloc(s->alloc.ctx, s->data, old_bytes, new_bytes, 1U);
    } else {
        new_data = (char *)s->alloc.alloc(s->alloc.ctx, new_bytes, 1U);
        if (new_data) {
            memcpy(new_data, s->data, new_bytes);
            s->alloc.free(s->alloc.ctx, s->data, old_bytes, 1U);
        }
    }
    if (!new_data) return CS_ENOMEM;
    s->data = new_data;
    s->cap  = s->len;
    s->generation++;
    return CS_OK;
}

void cs_swap(cs_string *a, cs_string *b) {
    if (!a || !b || a == b) return;
    cs_string tmp = *a;
    *a = *b;
    *b = tmp;
    a->generation++;
    b->generation++;
}

char *cs_detach(cs_string *s, size_t *out_len, size_t *out_cap) {
    if (!s) return NULL;
    if (!cs_string_is_heap(s)) {
        /* Allocate a 1-byte NUL buffer so caller always gets an owned heap buffer if possible */
        if (s->alloc.alloc) {
            char *p = (char *)s->alloc.alloc(s->alloc.ctx, 1U, 1U);
            if (p) {
                p[0] = '\0';
                if (out_len) *out_len = 0;
                if (out_cap) *out_cap = 0;
                s->generation++;
                return p;
            }
        }
        return NULL;
    }
    char *ptr = s->data;
    if (out_len) *out_len = s->len;
    if (out_cap) *out_cap = s->cap;
    s->data = (char *)(uintptr_t)cs_empty_cstr_sentinel;
    s->len  = 0;
    s->cap  = 0;
    s->generation++;
    return ptr;
}

cs_status cs_adopt(cs_string *s, char *buf, size_t len, size_t cap, cs_allocator alloc) {
    if (!s || !buf || len > cap || !alloc.alloc || !alloc.free) return CS_EINVAL;
    cs_destroy(s);
    buf[len] = '\0';
    s->data  = buf;
    s->len   = len;
    s->cap   = cap;
    s->alloc = alloc;
    s->generation++;
    return CS_OK;
}

/* Check if view `v` aliases `[s->data, s->data + s->cap]` */
static inline bool cs_view_aliases_string(const cs_string *s, cs_view v, size_t *out_offset) {
    if (!cs_string_is_heap(s) || !v.data || v.len == 0) return false;
    uintptr_t s_begin = (uintptr_t)s->data;
    uintptr_t s_end   = s_begin + s->cap + 1U;
    uintptr_t v_begin = (uintptr_t)v.data;
    if (v_begin >= s_begin && v_begin < s_end) {
        if (out_offset) *out_offset = (size_t)(v_begin - s_begin);
        return true;
    }
    return false;
}

cs_status cs_assign(cs_string *s, cs_view v) {
    if (!s || !cs_view_is_valid(v)) return CS_EINVAL;
    if (v.len == 0) {
        cs_clear(s);
        return CS_OK;
    }
    size_t alias_off = 0;
    bool aliases = cs_view_aliases_string(s, v, &alias_off);
    cs_status st = cs_reserve(s, v.len);
    if (st != CS_OK) return st;
    const char *src = aliases ? (s->data + alias_off) : v.data;
    memmove(s->data, src, v.len);
    s->len = v.len;
    s->data[s->len] = '\0';
    s->generation++;
    return CS_OK;
}

cs_status cs_assign_cstr(cs_string *s, const char *cstr) {
    if (!s || !cstr) return CS_EINVAL;
    return cs_assign(s, cs_sv(cstr));
}

cs_status cs_append(cs_string *s, cs_view v) {
    if (!s || !cs_view_is_valid(v)) return CS_EINVAL;
    if (v.len == 0) return CS_OK;

    size_t new_len = 0;
    if (cs_add_overflow_sz(s->len, v.len, &new_len)) return CS_EOVERFLOW;

    size_t alias_off = 0;
    bool aliases = cs_view_aliases_string(s, v, &alias_off);

    cs_status st = cs_reserve(s, new_len);
    if (st != CS_OK) return st;

    const char *src = aliases ? (s->data + alias_off) : v.data;
    memmove(s->data + s->len, src, v.len);
    s->len = new_len;
    s->data[s->len] = '\0';
    s->generation++;
    return CS_OK;
}

cs_status cs_append_cstr(cs_string *s, const char *cstr) {
    if (!s || !cstr) return CS_EINVAL;
    return cs_append(s, cs_sv(cstr));
}

cs_status cs_append_byte(cs_string *s, uint8_t byte_val) {
    char c = (char)byte_val;
    cs_view v = { &c, 1U };
    return cs_append(s, v);
}

cs_status cs_append_many(cs_string *s, const cs_view *views, size_t count) {
    if (!s) return CS_EINVAL;
    if (count == 0) return CS_OK;
    if (!views) return CS_EINVAL;

    size_t add_len = 0;
    for (size_t i = 0; i < count; i++) {
        if (!cs_view_is_valid(views[i])) return CS_EINVAL;
        if (cs_add_overflow_sz(add_len, views[i].len, &add_len)) return CS_EOVERFLOW;
    }
    if (add_len == 0) return CS_OK;

    size_t new_len = 0;
    if (cs_add_overflow_sz(s->len, add_len, &new_len)) return CS_EOVERFLOW;

    /* Check if any view aliases s */
    bool any_alias = false;
    for (size_t i = 0; i < count; i++) {
        if (cs_view_aliases_string(s, views[i], NULL)) {
            any_alias = true;
            break;
        }
    }

    if (!any_alias) {
        cs_status st = cs_reserve(s, new_len);
        if (st != CS_OK) return st;
        size_t pos = s->len;
        for (size_t i = 0; i < count; i++) {
            if (views[i].len > 0) {
                memcpy(s->data + pos, views[i].data, views[i].len);
                pos += views[i].len;
            }
        }
        s->len = new_len;
        s->data[s->len] = '\0';
        s->generation++;
        return CS_OK;
    }

    /* Self-aliasing fallback: append into temporary string then swap */
    cs_string tmp;
    cs_status st = cs_clone(&tmp, s);
    if (st != CS_OK) return st;
    st = cs_reserve(&tmp, new_len);
    if (st != CS_OK) {
        cs_destroy(&tmp);
        return st;
    }
    for (size_t i = 0; i < count; i++) {
        if (views[i].len > 0) {
            memcpy(tmp.data + tmp.len, views[i].data, views[i].len);
            tmp.len += views[i].len;
        }
    }
    tmp.data[tmp.len] = '\0';
    cs_swap(s, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
}

cs_status cs_replace_range_bytes(cs_string *s, size_t begin, size_t end, cs_view replacement) {
    if (!s || !cs_view_is_valid(replacement)) return CS_EINVAL;
    if (begin > end || end > s->len) return CS_ERANGE;

    size_t removed_len = end - begin;
    size_t tail_len    = s->len - end;
    size_t base_len    = s->len - removed_len;
    size_t new_len     = 0;
    if (cs_add_overflow_sz(base_len, replacement.len, &new_len)) return CS_EOVERFLOW;

    size_t alias_off = 0;
    bool aliases = cs_view_aliases_string(s, replacement, &alias_off);

    if (aliases) {
        /* Copy replacement to temporary stack or heap buffer if it overlaps with mutated range */
        uint8_t stack_buf[256];
        char *temp_rep = NULL;
        if (replacement.len <= sizeof(stack_buf)) {
            if (replacement.len > 0) memcpy(stack_buf, replacement.data, replacement.len);
            temp_rep = (char *)stack_buf;
        } else {
            temp_rep = (char *)s->alloc.alloc(s->alloc.ctx, replacement.len, 1U);
            if (!temp_rep) return CS_ENOMEM;
            memcpy(temp_rep, replacement.data, replacement.len);
        }
        cs_status st = cs_reserve(s, new_len);
        if (st != CS_OK) {
            if (temp_rep != (char *)stack_buf) s->alloc.free(s->alloc.ctx, temp_rep, replacement.len, 1U);
            return st;
        }
        if (tail_len > 0 && begin + replacement.len != end) {
            memmove(s->data + begin + replacement.len, s->data + end, tail_len);
        }
        if (replacement.len > 0) {
            memcpy(s->data + begin, temp_rep, replacement.len);
        }
        if (temp_rep != (char *)stack_buf) s->alloc.free(s->alloc.ctx, temp_rep, replacement.len, 1U);
    } else {
        cs_status st = cs_reserve(s, new_len);
        if (st != CS_OK) return st;
        if (tail_len > 0 && begin + replacement.len != end) {
            memmove(s->data + begin + replacement.len, s->data + end, tail_len);
        }
        if (replacement.len > 0) {
            memcpy(s->data + begin, replacement.data, replacement.len);
        }
    }

    s->len = new_len;
    if (cs_string_is_heap(s) || new_len > 0) {
        s->data[s->len] = '\0';
    }
    s->generation++;
    return CS_OK;
}

cs_status cs_insert_bytes(cs_string *s, size_t pos, cs_view v) {
    return cs_replace_range_bytes(s, pos, pos, v);
}

cs_status cs_prepend(cs_string *s, cs_view v) {
    return cs_replace_range_bytes(s, 0, 0, v);
}

cs_status cs_erase_bytes(cs_string *s, size_t begin, size_t end) {
    return cs_replace_range_bytes(s, begin, end, cs_empty_view());
}

cs_status cs_truncate_bytes(cs_string *s, size_t max_bytes) {
    if (!s) return CS_EINVAL;
    if (max_bytes >= s->len) return CS_OK;
    s->len = max_bytes;
    s->data[s->len] = '\0';
    s->generation++;
    return CS_OK;
}

cs_status cs_resize_bytes(cs_string *s, size_t new_len, uint8_t fill_byte) {
    if (!s) return CS_EINVAL;
    if (new_len <= s->len) {
        return cs_truncate_bytes(s, new_len);
    }
    cs_status st = cs_reserve(s, new_len);
    if (st != CS_OK) return st;
    memset(s->data + s->len, fill_byte, new_len - s->len);
    s->len = new_len;
    s->data[s->len] = '\0';
    s->generation++;
    return CS_OK;
}

#define CS_MAX_REPEAT_BYTES (256ULL * 1024ULL * 1024ULL)

cs_status cs_repeat(cs_string *dst, cs_view v, size_t count) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
    if (count == 0 || v.len == 0) {
        cs_clear(dst);
        return CS_OK;
    }
    size_t total = 0;
    if (cs_mul_overflow_sz(v.len, count, &total)) return CS_EOVERFLOW;
    if (total > CS_MAX_REPEAT_BYTES) return CS_ELIMIT;

    size_t alias_off = 0;
    bool aliases = cs_view_aliases_string(dst, v, &alias_off);

    cs_status st = cs_reserve(dst, total);
    if (st != CS_OK) return st;

    const char *src = aliases ? (dst->data + alias_off) : v.data;
    memmove(dst->data, src, v.len);
    /* Doubling copy for O(total) memcpy speed */
    size_t filled = v.len;
    while (filled < total) {
        size_t chunk = (total - filled < filled) ? (total - filled) : filled;
        memcpy(dst->data + filled, dst->data, chunk);
        filled += chunk;
    }
    dst->len = total;
    dst->data[dst->len] = '\0';
    dst->generation++;
    return CS_OK;
}

cs_status cs_repeat_buffer(cs_buffer *dst, cs_view v, size_t count, size_t *out_required) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
    size_t total = 0;
    if (v.len > 0 && count > 0) {
        if (cs_mul_overflow_sz(v.len, count, &total)) return CS_EOVERFLOW;
        if (total > CS_MAX_REPEAT_BYTES) return CS_ELIMIT;
    }
    size_t req_with_nul = 0;
    if (cs_add_overflow_sz(total, 1U, &req_with_nul)) return CS_EOVERFLOW;
    if (out_required) *out_required = req_with_nul;
    if (total > dst->cap) return CS_EBUFFER;

    for (size_t i = 0; i < count; i++) {
        if (v.len > 0) memcpy(dst->data + i * v.len, v.data, v.len);
    }
    dst->len = total;
    dst->data[dst->len] = '\0';
    return CS_OK;
}

/* Fixed-capacity cs_buffer implementation */
cs_status cs_buffer_init(cs_buffer *b, char *storage, size_t storage_size) {
    if (!b || !storage || storage_size == 0) return CS_EINVAL;
    b->data    = storage;
    b->len     = 0;
    b->cap     = storage_size - 1U;
    b->data[0] = '\0';
    return CS_OK;
}

void cs_buffer_clear(cs_buffer *b) {
    if (!b || !b->data) return;
    b->len     = 0;
    b->data[0] = '\0';
}

cs_view cs_buffer_as_view(const cs_buffer *b) {
    if (!b || !b->data) return cs_empty_view();
    cs_view v = { b->data, b->len };
    return v;
}

const char *cs_buffer_c_str(const cs_buffer *b) {
    if (!b || !b->data) return cs_empty_cstr_sentinel;
    return b->data;
}

cs_status cs_buffer_append(cs_buffer *b, cs_view v, size_t *out_required) {
    if (!b || !b->data || !cs_view_is_valid(v)) return CS_EINVAL;
    size_t new_len = 0;
    if (cs_add_overflow_sz(b->len, v.len, &new_len)) return CS_EOVERFLOW;
    size_t req_with_nul = 0;
    if (cs_add_overflow_sz(new_len, 1U, &req_with_nul)) return CS_EOVERFLOW;
    if (out_required) *out_required = req_with_nul;
    if (new_len > b->cap) {
        /* Leave destination buffer untouched on CS_EBUFFER */
        return CS_EBUFFER;
    }
    if (v.len > 0) {
        memmove(b->data + b->len, v.data, v.len);
    }
    b->len = new_len;
    b->data[b->len] = '\0';
    return CS_OK;
}

cs_status cs_buffer_append_byte(cs_buffer *b, uint8_t byte_val, size_t *out_required) {
    char c = (char)byte_val;
    cs_view v = { &c, 1U };
    return cs_buffer_append(b, v, out_required);
}

/* Builder API */
void cs_builder_init_string(cs_builder *b, cs_string *s) {
    if (!b) return;
    b->str_target     = s;
    b->buf_target     = NULL;
    b->sticky_err     = s ? CS_OK : CS_EINVAL;
    b->required_bytes = s ? (s->len + 1U) : 1U;
}

void cs_builder_init_buffer(cs_builder *b, cs_buffer *buf) {
    if (!b) return;
    b->str_target     = NULL;
    b->buf_target     = buf;
    b->sticky_err     = (buf && buf->data) ? CS_OK : CS_EINVAL;
    b->required_bytes = (buf && buf->data) ? (buf->len + 1U) : 1U;
}

cs_status cs_builder_append(cs_builder *b, cs_view v) {
    if (!b) return CS_EINVAL;
    if (!cs_view_is_valid(v)) {
        b->sticky_err = CS_EINVAL;
        return CS_EINVAL;
    }
    size_t next_req = 0;
    if (!cs_add_overflow_sz(b->required_bytes, v.len, &next_req)) {
        b->required_bytes = next_req;
    } else {
        b->sticky_err = CS_EOVERFLOW;
        return CS_EOVERFLOW;
    }
    if (b->sticky_err != CS_OK) return b->sticky_err;

    if (b->str_target) {
        cs_status st = cs_append(b->str_target, v);
        if (st != CS_OK) b->sticky_err = st;
        return st;
    } else if (b->buf_target) {
        cs_status st = cs_buffer_append(b->buf_target, v, NULL);
        if (st != CS_OK) b->sticky_err = st;
        return st;
    }
    return CS_EINVAL;
}

cs_status cs_builder_append_cstr(cs_builder *b, const char *cstr) {
    if (!cstr) {
        if (b) b->sticky_err = CS_EINVAL;
        return CS_EINVAL;
    }
    return cs_builder_append(b, cs_sv(cstr));
}

cs_status cs_builder_append_byte(cs_builder *b, uint8_t byte_val) {
    char c = (char)byte_val;
    cs_view v = { &c, 1U };
    return cs_builder_append(b, v);
}

cs_status cs_builder_finish(cs_builder *b, cs_view *out_view, size_t *out_required) {
    if (!b) return CS_EINVAL;
    if (out_required) *out_required = b->required_bytes;
    if (b->sticky_err != CS_OK) return b->sticky_err;
    if (out_view) {
        if (b->str_target) *out_view = cs_as_view(b->str_target);
        else if (b->buf_target) *out_view = cs_buffer_as_view(b->buf_target);
    }
    return CS_OK;
}

/* Basic Accessors */
bool cs_is_empty(cs_view v) {
    return v.len == 0;
}

size_t cs_len_bytes(cs_view v) {
    return v.len;
}

size_t cs_capacity(const cs_string *s) {
    return s ? s->cap : 0;
}

const char *cs_data(const cs_string *s) {
    return (s && s->data) ? s->data : cs_empty_cstr_sentinel;
}

const char *cs_c_str(const cs_string *s) {
    return (s && s->data) ? s->data : cs_empty_cstr_sentinel;
}

cs_view cs_as_view(const cs_string *s) {
    if (!s || !s->data) return cs_empty_view();
    cs_view v = { s->data, s->len };
    return v;
}

cs_status cs_byte_at(cs_view v, size_t idx, uint8_t *out_byte) {
    if (!cs_view_is_valid(v) || !out_byte) return CS_EINVAL;
    if (idx >= v.len) return CS_ERANGE;
    *out_byte = (uint8_t)v.data[idx];
    return CS_OK;
}

cs_status cs_front_byte(cs_view v, uint8_t *out_byte) {
    return cs_byte_at(v, 0, out_byte);
}

cs_status cs_back_byte(cs_view v, uint8_t *out_byte) {
    if (!cs_view_is_valid(v) || !out_byte) return CS_EINVAL;
    if (v.len == 0) return CS_ERANGE;
    *out_byte = (uint8_t)v.data[v.len - 1U];
    return CS_OK;
}

cs_status cs_slice_bytes(cs_view v, size_t begin, size_t end, cs_view *out) {
    if (!cs_view_is_valid(v) || !out) return CS_EINVAL;
    if (begin > end || end > v.len) return CS_ERANGE;
    out->data = v.data ? (v.data + begin) : NULL;
    out->len  = end - begin;
    return CS_OK;
}

cs_view cs_slice_bytes_clamped(cs_view v, size_t begin, size_t end) {
    if (!cs_view_is_valid(v)) return cs_empty_view();
    if (begin > v.len) begin = v.len;
    if (end > v.len)   end   = v.len;
    if (begin > end)   begin = end;
    cs_view res = { v.data ? (v.data + begin) : NULL, end - begin };
    return res;
}

cs_status cs_take_bytes(cs_view v, size_t count, cs_view *out) {
    return cs_slice_bytes(v, 0, count, out);
}

cs_status cs_drop_bytes(cs_view v, size_t count, cs_view *out) {
    return cs_slice_bytes(v, count, v.len, out);
}

void cs_byte_iter_init(cs_byte_iter *it, cs_view v) {
    if (!it) return;
    it->v   = v;
    it->pos = 0;
}

cs_status cs_byte_iter_next(cs_byte_iter *it, uint8_t *out_byte, size_t *out_offset) {
    if (!it || !cs_view_is_valid(it->v)) return CS_EINVAL;
    if (it->pos >= it->v.len) return CS_END;
    if (out_offset) *out_offset = it->pos;
    if (out_byte)   *out_byte   = (uint8_t)it->v.data[it->pos];
    it->pos++;
    return CS_OK;
}

cs_status cs_byte_iter_prev(cs_byte_iter *it, uint8_t *out_byte, size_t *out_offset) {
    if (!it || !cs_view_is_valid(it->v)) return CS_EINVAL;
    if (it->pos == 0 || it->pos > it->v.len) return CS_END;
    it->pos--;
    if (out_offset) *out_offset = it->pos;
    if (out_byte)   *out_byte   = (uint8_t)it->v.data[it->pos];
    return CS_OK;
}

/* ==========================================================================
 * Part 2: Unicode Properties, UTF-8/16/32 Codecs, UAX #29 Segmentation,
 *         Access/Slices & UAX #15 Normalization
 * ========================================================================== */

#if CS_ENABLE_UNICODE
/* Note: unicode_tables.inc will be spliced right here during cstr.c assembly */
/* ==========================================================================
 * AUTO-GENERATED UNICODE 17.0.0 TABLES (gen_unicode_tables.py v2.0.0)
 * DO NOT EDIT MANUALLY.
 *
 * Source UCD SHA-256 Checksums (ICU 78 / Unicode 17.0.0):
 *   icudt78l/propsTrie         : a957092c13862b8e3c61d210baf3f939d81e7daf0f6994fa83daa5a87944d825
 *   icudt78l/propsVecTrie      : 7adf440558d01c3f9bacd4e3664fd7112356aa3863ffb3dc96ebfeab343fff97
 *   icudt78l/propsVectors      : 5bb509c433a7b9b7facb5d07f1a525416186d0a6b7cfa613c9b5d337dee283ce
 *   icudt78l/ucase             : f149022cbfb0d593bebc435ffe681d99a2fd0645fc3b4b562e6bf34b517ad7d2
 *   icudt78l/nfc.nrm           : 00507c5a69f4f6dbd52722c7f392d7db3bb8f30b4d96e0e853df02952594453b
 *   icudt78l/nfkc.nrm          : 38e8721d93e531da5c76538807cd7a4b42ddc409138a9b6f62613be1ce4d3790
 *   icudt78l/uemoji.icu        : ebf60beb2a2fe55dbc31b7cbd3d3ce8d349ca040c7a35bf13fd5d2c2a07b1fc3
 *   icudt78l/brkitr/char.brk   : 5daa5d70b011e407f0c4ba06498792b224852f911991e14f0d88a4bef19feaa1
 *
 * UNICODE LICENSE V3 - DATA FILES AND SOFTWARE
 * Copyright (c) 1991-2025 Unicode, Inc. All rights reserved.
 * Distributed under the Terms of Use in https://www.unicode.org/copyright.html.
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of the Unicode data files and any associated documentation
 * (the "Data Files") or Unicode software and any associated documentation
 * (the "Software") to deal in the Data Files or Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, and/or sell copies of the Data
 * Files or Software, and to permit persons to whom the Data Files or
 * Software are furnished to do so, provided that either (a) this
 * copyright and permission notice appear with all copies of the Data
 * Files or Software, or (b) this copyright and permission notice appear
 * in associated Documentation.
 * ========================================================================== */

typedef struct {
    uint8_t gc;
    uint8_t gcb;
    uint8_t wb;
    uint8_t sb;
    uint8_t flags;
    uint8_t ccc;
    uint8_t qc;
    uint8_t incb;
} cs_uprop_record;

static const cs_uprop_record cs_uprop_palette[260] = {
    {0, 0, 0, 0, 0x00U, 0, 0x00U, 0}, /* 0 */
    {26, 3, 0, 0, 0x00U, 0, 0x00U, 0}, /* 1 */
    {26, 3, 0, 6, 0x02U, 0, 0x00U, 0}, /* 2 */
    {26, 2, 2, 2, 0x02U, 0, 0x00U, 0}, /* 3 */
    {26, 3, 3, 6, 0x02U, 0, 0x00U, 0}, /* 4 */
    {26, 1, 1, 1, 0x02U, 0, 0x00U, 0}, /* 5 */
    {23, 0, 18, 6, 0x02U, 0, 0x00U, 0}, /* 6 */
    {18, 0, 0, 12, 0x00U, 0, 0x00U, 0}, /* 7 */
    {18, 0, 12, 13, 0x00U, 0, 0x00U, 0}, /* 8 */
    {18, 0, 0, 0, 0x00U, 0, 0x00U, 0}, /* 9 */
    {20, 0, 0, 0, 0x00U, 0, 0x00U, 0}, /* 10 */
    {18, 0, 11, 13, 0x20U, 0, 0x00U, 0}, /* 11 */
    {14, 0, 0, 13, 0x00U, 0, 0x00U, 0}, /* 12 */
    {15, 0, 0, 13, 0x00U, 0, 0x00U, 0}, /* 13 */
    {19, 0, 0, 0, 0x00U, 0, 0x00U, 0}, /* 14 */
    {18, 0, 15, 14, 0x00U, 0, 0x00U, 0}, /* 15 */
    {13, 0, 0, 14, 0x00U, 0, 0x00U, 0}, /* 16 */
    {18, 0, 13, 11, 0x20U, 0, 0x00U, 0}, /* 17 */
    {9, 0, 16, 10, 0x00U, 0, 0x00U, 0}, /* 18 */
    {18, 0, 14, 14, 0x20U, 0, 0x00U, 0}, /* 19 */
    {1, 0, 10, 8, 0x15U, 0, 0x00U, 0}, /* 20 */
    {21, 0, 0, 0, 0x20U, 0, 0x00U, 0}, /* 21 */
    {12, 0, 17, 0, 0x00U, 0, 0x00U, 0}, /* 22 */
    {2, 0, 10, 7, 0x19U, 0, 0x00U, 0}, /* 23 */
    {2, 0, 10, 7, 0x59U, 0, 0x00U, 0}, /* 24 */
    {26, 3, 3, 4, 0x02U, 0, 0x00U, 0}, /* 25 */
    {23, 0, 0, 6, 0x02U, 0, 0x50U, 0}, /* 26 */
    {22, 0, 0, 0, 0x00U, 0, 0x00U, 0}, /* 27 */
    {21, 0, 0, 0, 0x20U, 0, 0x50U, 0}, /* 28 */
    {22, 14, 0, 0, 0x80U, 0, 0x00U, 0}, /* 29 */
    {5, 0, 10, 7, 0x19U, 0, 0x50U, 0}, /* 30 */
    {16, 0, 0, 13, 0x00U, 0, 0x00U, 0}, /* 31 */
    {27, 3, 7, 5, 0x20U, 0, 0x00U, 0}, /* 32 */
    {11, 0, 0, 0, 0x00U, 0, 0x50U, 0}, /* 33 */
    {2, 0, 10, 7, 0x19U, 0, 0x50U, 0}, /* 34 */
    {18, 0, 14, 0, 0x20U, 0, 0x00U, 0}, /* 35 */
    {21, 0, 10, 0, 0x20U, 0, 0x50U, 0}, /* 36 */
    {17, 0, 0, 13, 0x00U, 0, 0x00U, 0}, /* 37 */
    {1, 0, 10, 8, 0x15U, 0, 0x44U, 0}, /* 38 */
    {2, 0, 10, 7, 0x19U, 0, 0x44U, 0}, /* 39 */
    {2, 0, 10, 7, 0x59U, 0, 0x44U, 0}, /* 40 */
    {1, 0, 10, 8, 0x15U, 0, 0x50U, 0}, /* 41 */
    {5, 0, 10, 9, 0x01U, 0, 0x00U, 0}, /* 42 */
    {3, 0, 10, 8, 0x11U, 0, 0x50U, 0}, /* 43 */
    {4, 0, 10, 7, 0x39U, 0, 0x50U, 0}, /* 44 */
    {4, 0, 10, 7, 0x79U, 0, 0x50U, 0}, /* 45 */
    {4, 0, 10, 9, 0x21U, 0, 0x00U, 0}, /* 46 */
    {4, 0, 10, 7, 0x39U, 0, 0x00U, 0}, /* 47 */
    {21, 0, 10, 0, 0x20U, 0, 0x00U, 0}, /* 48 */
    {6, 4, 4, 3, 0x20U, 230, 0x22U, 3}, /* 49 */
    {6, 4, 4, 3, 0x20U, 230, 0x00U, 3}, /* 50 */
    {6, 4, 4, 3, 0x20U, 232, 0x00U, 3}, /* 51 */
    {6, 4, 4, 3, 0x20U, 220, 0x00U, 3}, /* 52 */
    {6, 4, 4, 3, 0x20U, 216, 0x22U, 3}, /* 53 */
    {6, 4, 4, 3, 0x20U, 202, 0x00U, 3}, /* 54 */
    {6, 4, 4, 3, 0x20U, 220, 0x22U, 3}, /* 55 */
    {6, 4, 4, 3, 0x20U, 202, 0x22U, 3}, /* 56 */
    {6, 4, 4, 3, 0x20U, 1, 0x00U, 3}, /* 57 */
    {6, 4, 4, 3, 0x20U, 1, 0x22U, 3}, /* 58 */
    {6, 4, 4, 3, 0x20U, 230, 0x55U, 3}, /* 59 */
    {6, 4, 4, 3, 0x39U, 240, 0x22U, 3}, /* 60 */
    {6, 4, 4, 3, 0x20U, 0, 0x00U, 3}, /* 61 */
    {6, 4, 4, 3, 0x20U, 233, 0x00U, 3}, /* 62 */
    {6, 4, 4, 3, 0x20U, 234, 0x00U, 3}, /* 63 */
    {6, 4, 4, 3, 0x21U, 230, 0x00U, 3}, /* 64 */
    {4, 0, 10, 9, 0x21U, 0, 0x55U, 0}, /* 65 */
    {18, 0, 15, 14, 0x00U, 0, 0x55U, 0}, /* 66 */
    {21, 0, 0, 0, 0x20U, 0, 0x54U, 0}, /* 67 */
    {18, 0, 14, 0, 0x20U, 0, 0x55U, 0}, /* 68 */
    {1, 0, 10, 8, 0x15U, 0, 0x54U, 0}, /* 69 */
    {8, 4, 4, 3, 0x20U, 0, 0x00U, 3}, /* 70 */
    {18, 0, 10, 0, 0x00U, 0, 0x00U, 0}, /* 71 */
    {18, 0, 0, 14, 0x00U, 0, 0x00U, 0}, /* 72 */
    {18, 0, 15, 12, 0x00U, 0, 0x00U, 0}, /* 73 */
    {13, 0, 10, 0, 0x00U, 0, 0x00U, 0}, /* 74 */
    {6, 4, 4, 3, 0x20U, 222, 0x00U, 3}, /* 75 */
    {6, 4, 4, 3, 0x20U, 228, 0x00U, 3}, /* 76 */
    {6, 4, 4, 3, 0x21U, 10, 0x00U, 3}, /* 77 */
    {6, 4, 4, 3, 0x21U, 11, 0x00U, 3}, /* 78 */
    {6, 4, 4, 3, 0x21U, 12, 0x00U, 3}, /* 79 */
    {6, 4, 4, 3, 0x21U, 13, 0x00U, 3}, /* 80 */
    {6, 4, 4, 3, 0x21U, 14, 0x00U, 3}, /* 81 */
    {6, 4, 4, 3, 0x21U, 15, 0x00U, 3}, /* 82 */
    {6, 4, 4, 3, 0x21U, 16, 0x00U, 3}, /* 83 */
    {6, 4, 4, 3, 0x21U, 17, 0x00U, 3}, /* 84 */
    {6, 4, 4, 3, 0x21U, 18, 0x00U, 3}, /* 85 */
    {6, 4, 4, 3, 0x21U, 19, 0x00U, 3}, /* 86 */
    {6, 4, 4, 3, 0x21U, 20, 0x00U, 3}, /* 87 */
    {6, 4, 4, 3, 0x21U, 21, 0x00U, 3}, /* 88 */
    {6, 4, 4, 3, 0x21U, 22, 0x00U, 3}, /* 89 */
    {13, 0, 0, 0, 0x00U, 0, 0x00U, 0}, /* 90 */
    {6, 4, 4, 3, 0x21U, 23, 0x00U, 3}, /* 91 */
    {6, 4, 4, 3, 0x21U, 24, 0x00U, 3}, /* 92 */
    {6, 4, 4, 3, 0x21U, 25, 0x00U, 3}, /* 93 */
    {6, 4, 4, 3, 0x21U, 220, 0x00U, 3}, /* 94 */
    {5, 0, 9, 9, 0x01U, 0, 0x00U, 0}, /* 95 */
    {18, 0, 10, 9, 0x00U, 0, 0x00U, 0}, /* 96 */
    {27, 7, 16, 10, 0x20U, 0, 0x00U, 0}, /* 97 */
    {6, 4, 4, 3, 0x21U, 30, 0x00U, 3}, /* 98 */
    {6, 4, 4, 3, 0x21U, 31, 0x00U, 3}, /* 99 */
    {6, 4, 4, 3, 0x21U, 32, 0x00U, 3}, /* 100 */
    {5, 0, 10, 9, 0x01U, 0, 0x44U, 0}, /* 101 */
    {6, 4, 4, 3, 0x21U, 27, 0x00U, 3}, /* 102 */
    {6, 4, 4, 3, 0x21U, 28, 0x00U, 3}, /* 103 */
    {6, 4, 4, 3, 0x21U, 29, 0x00U, 3}, /* 104 */
    {6, 4, 4, 3, 0x21U, 33, 0x00U, 3}, /* 105 */
    {6, 4, 4, 3, 0x21U, 34, 0x00U, 3}, /* 106 */
    {6, 4, 4, 3, 0x21U, 230, 0x22U, 3}, /* 107 */
    {6, 4, 4, 3, 0x21U, 220, 0x22U, 3}, /* 108 */
    {18, 0, 16, 10, 0x00U, 0, 0x00U, 0}, /* 109 */
    {18, 0, 15, 10, 0x00U, 0, 0x00U, 0}, /* 110 */
    {6, 4, 4, 3, 0x21U, 35, 0x00U, 3}, /* 111 */
    {5, 0, 10, 9, 0x01U, 0, 0x50U, 0}, /* 112 */
    {27, 7, 10, 5, 0x20U, 0, 0x00U, 0}, /* 113 */
    {6, 4, 4, 3, 0x21U, 36, 0x00U, 3}, /* 114 */
    {6, 4, 4, 3, 0x21U, 0, 0x00U, 3}, /* 115 */
    {7, 8, 4, 3, 0x01U, 0, 0x00U, 0}, /* 116 */
    {5, 0, 10, 9, 0x01U, 0, 0x00U, 1}, /* 117 */
    {5, 0, 10, 9, 0x01U, 0, 0x44U, 1}, /* 118 */
    {6, 4, 4, 3, 0x20U, 7, 0x22U, 3}, /* 119 */
    {6, 4, 4, 3, 0x20U, 9, 0x00U, 2}, /* 120 */
    {5, 0, 10, 9, 0x01U, 0, 0x55U, 1}, /* 121 */
    {6, 4, 4, 3, 0x20U, 7, 0x00U, 3}, /* 122 */
    {7, 4, 4, 3, 0x01U, 0, 0x22U, 3}, /* 123 */
    {7, 8, 4, 3, 0x01U, 0, 0x44U, 0}, /* 124 */
    {11, 0, 0, 0, 0x00U, 0, 0x00U, 0}, /* 125 */
    {5, 0, 10, 9, 0x01U, 0, 0x55U, 0}, /* 126 */
    {6, 4, 4, 3, 0x20U, 9, 0x00U, 3}, /* 127 */
    {6, 4, 4, 3, 0x21U, 0, 0x22U, 3}, /* 128 */
    {6, 4, 4, 3, 0x21U, 0, 0x44U, 3}, /* 129 */
    {6, 4, 4, 3, 0x21U, 84, 0x00U, 3}, /* 130 */
    {6, 4, 4, 3, 0x21U, 91, 0x22U, 3}, /* 131 */
    {7, 4, 4, 3, 0x01U, 0, 0x44U, 3}, /* 132 */
    {5, 7, 10, 9, 0x01U, 0, 0x00U, 0}, /* 133 */
    {6, 4, 4, 3, 0x20U, 9, 0x22U, 3}, /* 134 */
    {5, 0, 0, 9, 0x01U, 0, 0x00U, 0}, /* 135 */
    {5, 8, 0, 9, 0x01U, 0, 0x50U, 0}, /* 136 */
    {6, 4, 4, 3, 0x21U, 103, 0x00U, 3}, /* 137 */
    {6, 4, 4, 3, 0x21U, 9, 0x00U, 3}, /* 138 */
    {4, 0, 0, 9, 0x21U, 0, 0x00U, 0}, /* 139 */
    {6, 4, 4, 3, 0x20U, 107, 0x00U, 3}, /* 140 */
    {6, 4, 4, 3, 0x21U, 118, 0x00U, 3}, /* 141 */
    {6, 4, 4, 3, 0x20U, 122, 0x00U, 3}, /* 142 */
    {5, 0, 0, 9, 0x01U, 0, 0x50U, 0}, /* 143 */
    {18, 0, 0, 0, 0x00U, 0, 0x50U, 0}, /* 144 */
    {6, 4, 4, 3, 0x20U, 216, 0x00U, 3}, /* 145 */
    {7, 8, 4, 3, 0x00U, 0, 0x00U, 0}, /* 146 */
    {6, 4, 4, 3, 0x21U, 129, 0x00U, 3}, /* 147 */
    {6, 4, 4, 3, 0x21U, 130, 0x00U, 3}, /* 148 */
    {6, 4, 4, 3, 0x21U, 0, 0x55U, 3}, /* 149 */
    {6, 4, 4, 3, 0x21U, 132, 0x00U, 3}, /* 150 */
    {6, 4, 4, 3, 0x21U, 0, 0x50U, 3}, /* 151 */
    {5, 0, 0, 9, 0x01U, 0, 0x00U, 1}, /* 152 */
    {5, 0, 0, 9, 0x01U, 0, 0x44U, 1}, /* 153 */
    {7, 0, 4, 3, 0x01U, 0, 0x00U, 0}, /* 154 */
    {2, 0, 10, 9, 0x19U, 0, 0x00U, 0}, /* 155 */
    {5, 9, 10, 9, 0x01U, 0, 0x00U, 0}, /* 156 */
    {5, 10, 10, 9, 0x01U, 0, 0x00U, 0}, /* 157 */
    {5, 10, 10, 9, 0x01U, 0, 0x22U, 0}, /* 158 */
    {5, 11, 10, 9, 0x01U, 0, 0x22U, 0}, /* 159 */
    {5, 11, 10, 9, 0x01U, 0, 0x00U, 0}, /* 160 */
    {10, 0, 10, 9, 0x01U, 0, 0x00U, 0}, /* 161 */
    {7, 4, 4, 3, 0x00U, 9, 0x00U, 3}, /* 162 */
    {6, 4, 4, 3, 0x21U, 228, 0x00U, 3}, /* 163 */
    {11, 0, 16, 10, 0x00U, 0, 0x00U, 0}, /* 164 */
    {7, 4, 4, 3, 0x00U, 9, 0x00U, 2}, /* 165 */
    {1, 0, 10, 9, 0x15U, 0, 0x00U, 0}, /* 166 */
    {6, 4, 4, 3, 0x20U, 214, 0x00U, 3}, /* 167 */
    {6, 4, 4, 3, 0x20U, 218, 0x00U, 3}, /* 168 */
    {2, 0, 10, 7, 0x19U, 0, 0x54U, 0}, /* 169 */
    {2, 0, 10, 7, 0x19U, 0, 0x55U, 0}, /* 170 */
    {3, 0, 10, 8, 0x11U, 0, 0x44U, 0}, /* 171 */
    {1, 0, 10, 8, 0x15U, 0, 0x55U, 0}, /* 172 */
    {21, 0, 0, 0, 0x20U, 0, 0x55U, 0}, /* 173 */
    {23, 0, 18, 6, 0x02U, 0, 0x55U, 0}, /* 174 */
    {23, 0, 18, 6, 0x02U, 0, 0x50U, 0}, /* 175 */
    {27, 3, 0, 5, 0x20U, 0, 0x00U, 0}, /* 176 */
    {27, 4, 4, 3, 0x20U, 0, 0x00U, 0}, /* 177 */
    {27, 5, 5, 3, 0x20U, 0, 0x00U, 3}, /* 178 */
    {13, 0, 0, 0, 0x00U, 0, 0x50U, 0}, /* 179 */
    {16, 0, 13, 13, 0x20U, 0, 0x00U, 0}, /* 180 */
    {17, 0, 13, 13, 0x20U, 0, 0x00U, 0}, /* 181 */
    {18, 0, 13, 11, 0x20U, 0, 0x50U, 0}, /* 182 */
    {24, 3, 3, 4, 0x02U, 0, 0x00U, 0}, /* 183 */
    {25, 3, 3, 4, 0x02U, 0, 0x00U, 0}, /* 184 */
    {23, 0, 17, 6, 0x02U, 0, 0x50U, 0}, /* 185 */
    {18, 14, 0, 12, 0x80U, 0, 0x50U, 0}, /* 186 */
    {19, 0, 15, 0, 0x00U, 0, 0x00U, 0}, /* 187 */
    {18, 0, 0, 12, 0x00U, 0, 0x50U, 0}, /* 188 */
    {0, 3, 0, 0, 0x00U, 0, 0x00U, 0}, /* 189 */
    {19, 0, 0, 0, 0x00U, 0, 0x50U, 0}, /* 190 */
    {14, 0, 0, 13, 0x00U, 0, 0x50U, 0}, /* 191 */
    {15, 0, 0, 13, 0x00U, 0, 0x50U, 0}, /* 192 */
    {20, 0, 0, 0, 0x00U, 0, 0x50U, 0}, /* 193 */
    {22, 0, 0, 0, 0x00U, 0, 0x50U, 0}, /* 194 */
    {22, 14, 0, 0, 0x80U, 0, 0x50U, 0}, /* 195 */
    {2, 14, 10, 7, 0x99U, 0, 0x50U, 0}, /* 196 */
    {2, 0, 10, 7, 0x59U, 0, 0x50U, 0}, /* 197 */
    {10, 0, 10, 8, 0x15U, 0, 0x50U, 0}, /* 198 */
    {10, 0, 10, 7, 0x19U, 0, 0x50U, 0}, /* 199 */
    {19, 14, 0, 0, 0x80U, 0, 0x00U, 0}, /* 200 */
    {19, 0, 0, 0, 0x00U, 0, 0x44U, 0}, /* 201 */
    {22, 0, 0, 0, 0x00U, 0, 0x44U, 0}, /* 202 */
    {14, 0, 0, 13, 0x00U, 0, 0x55U, 0}, /* 203 */
    {15, 0, 0, 13, 0x00U, 0, 0x55U, 0}, /* 204 */
    {22, 0, 10, 8, 0x15U, 0, 0x50U, 0}, /* 205 */
    {22, 14, 10, 8, 0x95U, 0, 0x50U, 0}, /* 206 */
    {22, 0, 10, 7, 0x19U, 0, 0x50U, 0}, /* 207 */
    {22, 0, 0, 13, 0x00U, 0, 0x00U, 0}, /* 208 */
    {19, 0, 0, 0, 0x00U, 0, 0x55U, 0}, /* 209 */
    {4, 0, 10, 9, 0x21U, 0, 0x50U, 0}, /* 210 */
    {18, 0, 0, 13, 0x00U, 0, 0x00U, 0}, /* 211 */
    {10, 0, 0, 9, 0x01U, 0, 0x00U, 0}, /* 212 */
    {7, 4, 4, 3, 0x00U, 224, 0x00U, 3}, /* 213 */
    {13, 14, 0, 0, 0x80U, 0, 0x00U, 0}, /* 214 */
    {4, 0, 8, 9, 0x21U, 0, 0x00U, 0}, /* 215 */
    {10, 0, 0, 9, 0x01U, 0, 0x50U, 0}, /* 216 */
    {18, 14, 0, 0, 0x80U, 0, 0x00U, 0}, /* 217 */
    {5, 0, 0, 9, 0x01U, 0, 0x44U, 0}, /* 218 */
    {6, 4, 4, 3, 0x20U, 8, 0x22U, 3}, /* 219 */
    {21, 0, 8, 0, 0x20U, 0, 0x50U, 0}, /* 220 */
    {4, 0, 0, 9, 0x21U, 0, 0x44U, 0}, /* 221 */
    {13, 0, 8, 0, 0x00U, 0, 0x00U, 0}, /* 222 */
    {5, 0, 8, 9, 0x01U, 0, 0x00U, 0}, /* 223 */
    {5, 0, 8, 9, 0x01U, 0, 0x44U, 0}, /* 224 */
    {4, 0, 8, 9, 0x21U, 0, 0x44U, 0}, /* 225 */
    {5, 0, 8, 9, 0x01U, 0, 0x50U, 0}, /* 226 */
    {22, 0, 8, 0, 0x00U, 0, 0x50U, 0}, /* 227 */
    {5, 12, 10, 9, 0x01U, 0, 0x44U, 0}, /* 228 */
    {5, 13, 10, 9, 0x01U, 0, 0x44U, 0}, /* 229 */
    {28, 0, 0, 0, 0x00U, 0, 0x00U, 0}, /* 230 */
    {29, 0, 0, 0, 0x00U, 0, 0x00U, 0}, /* 231 */
    {5, 0, 0, 9, 0x01U, 0, 0x55U, 0}, /* 232 */
    {5, 0, 9, 9, 0x01U, 0, 0x55U, 0}, /* 233 */
    {6, 4, 4, 3, 0x21U, 26, 0x00U, 3}, /* 234 */
    {5, 0, 9, 9, 0x01U, 0, 0x50U, 0}, /* 235 */
    {18, 0, 0, 14, 0x00U, 0, 0x50U, 0}, /* 236 */
    {18, 0, 14, 14, 0x20U, 0, 0x50U, 0}, /* 237 */
    {13, 0, 0, 14, 0x00U, 0, 0x50U, 0}, /* 238 */
    {12, 0, 17, 0, 0x00U, 0, 0x50U, 0}, /* 239 */
    {18, 0, 15, 14, 0x00U, 0, 0x50U, 0}, /* 240 */
    {18, 0, 13, 0, 0x20U, 0, 0x50U, 0}, /* 241 */
    {9, 0, 16, 10, 0x00U, 0, 0x50U, 0}, /* 242 */
    {4, 0, 8, 9, 0x21U, 0, 0x50U, 0}, /* 243 */
    {4, 4, 4, 3, 0x21U, 0, 0x50U, 3}, /* 244 */
    {7, 4, 4, 3, 0x01U, 0, 0x66U, 3}, /* 245 */
    {6, 4, 4, 3, 0x21U, 0, 0x66U, 3}, /* 246 */
    {5, 10, 10, 9, 0x01U, 0, 0x66U, 0}, /* 247 */
    {5, 10, 10, 9, 0x01U, 0, 0x44U, 0}, /* 248 */
    {7, 4, 4, 3, 0x01U, 6, 0x00U, 3}, /* 249 */
    {6, 4, 4, 3, 0x21U, 1, 0x00U, 3}, /* 250 */
    {22, 0, 0, 0, 0x00U, 0, 0x55U, 0}, /* 251 */
    {7, 4, 4, 3, 0x00U, 216, 0x00U, 3}, /* 252 */
    {7, 4, 4, 3, 0x00U, 226, 0x00U, 3}, /* 253 */
    {0, 14, 0, 0, 0x80U, 0, 0x00U, 0}, /* 254 */
    {22, 0, 10, 8, 0x15U, 0, 0x00U, 0}, /* 255 */
    {22, 14, 10, 8, 0x95U, 0, 0x00U, 0}, /* 256 */
    {22, 6, 6, 0, 0x00U, 0, 0x00U, 0}, /* 257 */
    {21, 4, 4, 0, 0x20U, 0, 0x00U, 3}, /* 258 */
    {27, 4, 4, 3, 0x20U, 0, 0x00U, 3}, /* 259 */
};

static const uint8_t cs_uprop_stage1[272] = {
    0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 6, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16, 17, 18, 5, 19, 17, 20, 21, 22, 23, 24, 25,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 26, 27, 28, 5, 29, 30,
    5, 31, 5, 32, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
    17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
    17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
    17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
    17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
    17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
    17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
    17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
    17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
    17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
    17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
    33, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 34,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 34
};

static const uint16_t cs_uprop_stage2[2240] = {
    0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704,
    768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472,
    1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 2048, 2112, 2176, 2240,
    2304, 2368, 2432, 2496, 2560, 2624, 2688, 2752, 2816, 2880, 2944, 3008,
    3072, 3136, 3200, 3264, 3328, 3392, 3456, 3520, 3584, 3648, 3712, 3776,
    3840, 3904, 3968, 4032, 4096, 4160, 4224, 4288, 4352, 4416, 4480, 4544,
    1664, 4608, 4672, 4736, 4800, 4864, 4928, 4992, 5056, 1664, 1664, 1664,
    1664, 1664, 1664, 1664, 1664, 5120, 5184, 5248, 5312, 5376, 5440, 5504,
    5568, 5632, 5696, 5760, 5824, 5888, 5952, 6016, 6080, 6144, 6208, 6272,
    6336, 6400, 6464, 6528, 6592, 6656, 6720, 6784, 6848, 6912, 6976, 7040,
    7104, 7168, 7232, 7296, 7360, 7424, 7488, 7552, 7616, 7680, 7744, 7808,
    7872, 7936, 8000, 8064, 8128, 8192, 8256, 8320, 8384, 8448, 8512, 8576,
    8640, 8704, 8768, 8832, 8896, 8896, 8960, 9024, 9088, 9152, 9216, 9280,
    9344, 9408, 9472, 9536, 8896, 8896, 8896, 8896, 9600, 9664, 9728, 9792,
    9856, 9920, 9664, 9984, 10048, 10112, 8896, 8896, 10176, 10240, 10304, 10368,
    10432, 10496, 10560, 10624, 10688, 10752, 10816, 10880, 10944, 10944, 10944, 11008,
    11072, 11136, 11200, 11264, 11328, 11392, 11456, 11520, 11584, 11648, 11712, 11776,
    11840, 11904, 10944, 10944, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 8896,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    12032, 1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664,
    1664, 1664, 1664, 1664, 1664, 1664, 12096, 12160, 1664, 1664, 1664, 1664,
    12224, 12288, 12352, 12416, 12480, 12544, 12608, 12672, 12736, 12800, 12864, 12928,
    12992, 13056, 13120, 13184, 13248, 13312, 13376, 13440, 13504, 13568, 13632, 13696,
    13760, 13824, 13888, 13952, 14016, 14080, 14144, 13760, 13824, 13888, 13952, 14016,
    14080, 14144, 13760, 13824, 13888, 13952, 14016, 14080, 14144, 13760, 13824, 13888,
    13952, 14016, 14080, 14144, 13760, 13824, 13888, 13952, 14016, 14080, 14144, 13760,
    13824, 13888, 13952, 14016, 14080, 14144, 13760, 13824, 13888, 13952, 14016, 14080,
    14144, 13760, 13824, 13888, 13952, 14016, 14080, 14144, 13760, 13824, 13888, 13952,
    14016, 14080, 14144, 13760, 13824, 13888, 13952, 14016, 14080, 14144, 13760, 13824,
    13888, 13952, 14016, 14080, 14144, 13760, 13824, 13888, 13952, 14016, 14080, 14144,
    13760, 13824, 13888, 13952, 14016, 14080, 14144, 13760, 13824, 13888, 13952, 14016,
    14080, 14144, 13760, 13824, 13888, 13952, 14016, 14080, 14144, 13760, 13824, 13888,
    13952, 14016, 14080, 14144, 13760, 13824, 13888, 13952, 14016, 14080, 14144, 13760,
    13824, 13888, 13952, 14016, 14080, 14144, 13760, 13824, 13888, 13952, 14016, 14080,
    14144, 13760, 13824, 13888, 13952, 14016, 14080, 14144, 13760, 13824, 13888, 13952,
    14016, 14080, 14144, 13760, 13824, 13888, 13952, 14016, 14080, 14144, 13760, 13824,
    13888, 13952, 14016, 14080, 14144, 13760, 13824, 13888, 13952, 14016, 14080, 14144,
    13760, 13824, 13888, 13952, 14016, 14080, 14208, 14272, 14336, 14336, 14336, 14336,
    14336, 14336, 14336, 14336, 14336, 14336, 14336, 14336, 14336, 14336, 14336, 14336,
    14336, 14336, 14336, 14336, 14336, 14336, 14336, 14336, 14336, 14336, 14336, 14336,
    14336, 14336, 14336, 14336, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400,
    14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400,
    14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400,
    14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400,
    14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400,
    14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400,
    14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400,
    14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400,
    14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14464, 14464, 14464, 14464,
    14528, 14592, 14464, 14656, 14720, 14784, 14848, 14912, 11392, 11392, 11392, 11392,
    14976, 15040, 15104, 15168, 15232, 15296, 11392, 15360, 15424, 15488, 15552, 15616,
    15680, 15744, 1664, 15808, 15872, 15936, 16000, 16064, 16128, 16128, 16192, 16256,
    16320, 16384, 16448, 16512, 16576, 16640, 16704, 16768, 16832, 16896, 16960, 17024,
    1664, 1664, 1664, 1664, 17088, 17152, 17216, 16128, 17280, 17344, 17408, 17472,
    17536, 17600, 17664, 17728, 17792, 17856, 17920, 17984, 18048, 18112, 18176, 16128,
    1664, 18240, 18304, 18368, 18432, 18496, 18560, 16128, 16128, 18624, 18688, 18752,
    18816, 18880, 18944, 19008, 19072, 19136, 19200, 19264, 19328, 19392, 19456, 19520,
    19584, 19648, 19712, 19776, 19840, 19904, 19968, 20032, 20096, 20160, 20224, 20288,
    16128, 16128, 20352, 20416, 20480, 20544, 20608, 20672, 20736, 20800, 16128, 16128,
    20864, 16128, 20928, 20992, 21056, 21120, 21184, 21248, 21312, 21376, 21440, 21504,
    21568, 21632, 16128, 21696, 21760, 21824, 21888, 16128, 21952, 22016, 22080, 22144,
    16128, 16128, 16128, 22208, 22272, 22336, 22400, 22464, 1664, 1664, 1664, 1664,
    1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664, 17600, 16128,
    22528, 22592, 1664, 1664, 1664, 22656, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 22720, 22784,
    1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664,
    1664, 1664, 1664, 1664, 22848, 22912, 1664, 1664, 1664, 1664, 1664, 1664,
    1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664,
    1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664,
    1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664,
    1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664, 1664,
    1664, 1664, 1664, 1664, 1664, 1664, 1664, 15808, 1664, 1664, 1664, 1664,
    1664, 1664, 1664, 1664, 1664, 22976, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 23040, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 1664, 1664, 1664, 1664,
    1664, 1664, 1664, 1664, 21504, 23104, 23168, 23232, 23296, 23360, 23424, 16128,
    16128, 16128, 16128, 16128, 16128, 23488, 16128, 16128, 16128, 23552, 23616, 23680,
    1664, 23744, 23808, 23872, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 23936, 24000, 16128, 11968, 24064,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 24128,
    24192, 11968, 11968, 11968, 24256, 24320, 11968, 11968, 11968, 11968, 11968, 24384,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    1664, 24448, 24512, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 8896, 8896, 8896, 24576, 8896, 8896, 8896, 8896,
    8896, 8896, 24640, 24704, 24768, 24832, 8896, 24896, 8896, 8896, 8896, 24960,
    25024, 25088, 25152, 25216, 8896, 25280, 16128, 25344, 8896, 25408, 16128, 16128,
    25472, 25536, 25600, 25664, 25728, 25792, 25856, 25920, 25984, 26048, 26112, 26176,
    26240, 26304, 26368, 26432, 8896, 8896, 8896, 8896, 8896, 8896, 8896, 8896,
    26496, 26560, 26624, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 26688, 16128, 16128, 16128,
    26752, 26816, 26880, 16128, 26944, 27008, 16128, 16128, 16128, 16128, 27072, 27136,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 27200, 16128, 16128, 16128, 27264,
    16128, 16128, 16128, 27328, 16128, 16128, 16128, 27392, 1664, 1664, 1664, 27456,
    27520, 27584, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 27648, 27712, 16128, 27776, 16128, 16128, 16128, 27840, 27904, 27968, 28032,
    16128, 16128, 16128, 16128, 28096, 8896, 28160, 28224, 28288, 28352, 28416, 28480,
    28544, 28608, 28672, 28672, 28736, 28800, 28864, 28928, 28800, 28800, 28800, 28992,
    29056, 29120, 29184, 29248, 28800, 29312, 28800, 29376, 8896, 8896, 8896, 29440,
    29504, 29568, 29632, 29696, 29760, 29824, 28800, 28800, 8896, 29888, 29952, 30016,
    8896, 8896, 30080, 30144, 28672, 28672, 28672, 28672, 28672, 28672, 28672, 28672,
    28672, 28672, 28672, 28672, 28672, 28672, 28672, 30208, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 30272,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 30336, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 30400, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 30464, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 30528, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 14464, 14464, 14464, 14464,
    14464, 14464, 14464, 14464, 30592, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 30656, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968, 11968,
    11968, 30720, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128, 16128,
    30784, 30848, 30912, 30912, 30976, 30976, 30976, 31040, 30912, 30912, 30912, 30912,
    30912, 30912, 30912, 30912, 30912, 30912, 30912, 30912, 30912, 30912, 30912, 30912,
    30912, 30912, 30912, 30912, 30912, 30912, 30912, 30912, 30912, 30912, 30912, 30912,
    30912, 30912, 30912, 30912, 30912, 30912, 30912, 30912, 30912, 30912, 30912, 30912,
    30912, 30912, 30912, 30912, 30912, 30912, 30912, 30912, 30912, 30912, 30912, 30912,
    30912, 30912, 30912, 30912, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400,
    14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400,
    14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400,
    14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400,
    14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400, 14400,
    14400, 14400, 14400, 14400, 14400, 14400, 14400, 31104
};

static const uint16_t cs_uprop_stage3[31168] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 4, 5, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    6, 7, 8, 9, 10, 9, 9, 11, 12, 13, 9, 14, 15, 16, 17, 9,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 15, 14, 14, 14, 7,
    9, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 12, 9, 13, 21, 22,
    21, 23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 12, 14, 13, 14, 1,
    1, 1, 1, 1, 1, 25, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    26, 9, 10, 10, 10, 10, 27, 9, 28, 29, 30, 31, 14, 32, 29, 28,
    27, 14, 33, 33, 28, 34, 9, 35, 36, 33, 30, 37, 33, 33, 33, 9,
    38, 38, 38, 38, 38, 38, 20, 38, 38, 38, 38, 38, 38, 38, 38, 38,
    20, 38, 38, 38, 38, 38, 38, 14, 20, 38, 38, 38, 38, 38, 20, 23,
    39, 39, 39, 39, 39, 39, 23, 39, 39, 39, 39, 39, 39, 39, 39, 39,
    23, 39, 39, 39, 39, 39, 39, 14, 23, 39, 39, 39, 39, 39, 23, 39,
    38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39,
    20, 23, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39,
    38, 39, 38, 39, 38, 39, 20, 23, 38, 39, 38, 39, 38, 39, 38, 40,
    38, 23, 41, 34, 38, 39, 38, 39, 23, 38, 39, 38, 39, 38, 39, 41,
    34, 20, 23, 38, 39, 38, 39, 38, 39, 34, 20, 23, 38, 39, 38, 39,
    38, 39, 20, 23, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39,
    38, 39, 38, 39, 38, 39, 20, 23, 38, 39, 38, 39, 38, 39, 38, 39,
    38, 39, 38, 39, 38, 39, 38, 39, 38, 38, 39, 38, 39, 38, 39, 34,
    23, 20, 20, 23, 20, 23, 20, 20, 23, 20, 20, 20, 23, 23, 20, 20,
    20, 20, 23, 20, 20, 23, 20, 20, 20, 23, 23, 23, 20, 20, 23, 20,
    38, 39, 20, 23, 20, 23, 20, 20, 23, 20, 23, 23, 20, 23, 20, 38,
    39, 20, 20, 20, 23, 20, 23, 20, 20, 23, 23, 42, 20, 23, 23, 23,
    42, 42, 42, 42, 41, 43, 34, 41, 43, 34, 41, 43, 34, 38, 39, 38,
    39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 23, 38, 39,
    38, 39, 38, 39, 20, 23, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39,
    39, 41, 43, 34, 38, 39, 20, 20, 38, 39, 38, 39, 38, 39, 38, 39,
    38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39,
    38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 20, 23, 38, 39,
    20, 23, 20, 23, 20, 23, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39,
    38, 39, 38, 39, 23, 23, 23, 23, 23, 23, 20, 20, 23, 20, 20, 23,
    23, 20, 23, 20, 20, 20, 20, 23, 20, 24, 20, 23, 20, 23, 20, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 24, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 42, 42, 23, 23, 23, 23, 23, 23, 23, 24, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    44, 44, 45, 44, 44, 44, 44, 44, 44, 46, 46, 46, 46, 46, 46, 46,
    47, 47, 48, 48, 48, 48, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46,
    46, 46, 48, 48, 48, 48, 48, 48, 28, 28, 28, 28, 28, 28, 48, 48,
    44, 44, 44, 44, 44, 48, 48, 48, 48, 48, 48, 48, 46, 48, 46, 48,
    48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
    49, 49, 49, 49, 49, 50, 49, 49, 49, 49, 49, 49, 49, 50, 50, 49,
    50, 49, 50, 49, 49, 51, 52, 52, 52, 52, 51, 53, 52, 52, 52, 52,
    52, 54, 54, 55, 55, 55, 55, 56, 56, 52, 52, 52, 52, 55, 55, 52,
    55, 55, 52, 52, 57, 57, 57, 57, 58, 52, 52, 52, 52, 50, 50, 50,
    59, 59, 49, 59, 59, 60, 50, 52, 52, 52, 50, 50, 50, 52, 52, 61,
    50, 50, 50, 52, 52, 52, 52, 50, 51, 52, 52, 50, 62, 63, 63, 62,
    63, 63, 62, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    20, 23, 20, 23, 65, 21, 20, 23, 0, 0, 44, 23, 23, 23, 66, 20,
    0, 0, 0, 0, 28, 67, 38, 68, 38, 38, 38, 0, 38, 0, 38, 38,
    39, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 0, 20, 20, 20, 20, 20, 20, 20, 38, 38, 39, 39, 39, 39,
    39, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 39, 39, 39, 39, 39, 20,
    34, 34, 41, 69, 69, 34, 34, 23, 20, 23, 20, 23, 20, 23, 20, 23,
    20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23,
    34, 34, 34, 24, 41, 34, 14, 20, 23, 41, 20, 23, 23, 20, 20, 20,
    38, 38, 20, 38, 20, 20, 20, 38, 20, 20, 20, 20, 38, 38, 38, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 38, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 39, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    39, 39, 23, 39, 23, 23, 24, 39, 24, 23, 23, 23, 39, 39, 39, 23,
    20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23,
    20, 23, 20, 23, 20, 23, 38, 39, 20, 23, 20, 23, 20, 23, 20, 23,
    20, 23, 27, 50, 50, 50, 50, 50, 70, 70, 20, 23, 20, 23, 20, 23,
    20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23,
    20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23,
    20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23,
    20, 38, 39, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 23,
    38, 39, 38, 39, 20, 23, 38, 39, 20, 23, 38, 39, 38, 39, 38, 39,
    20, 23, 38, 39, 38, 39, 38, 39, 20, 23, 38, 39, 38, 39, 38, 39,
    38, 39, 38, 39, 38, 39, 20, 23, 38, 39, 20, 23, 20, 23, 20, 23,
    20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23,
    20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23,
    20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23,
    0, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 0, 0, 46, 71, 71, 71, 72, 71, 35,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 34, 23, 73, 74, 0, 0, 27, 27, 10,
    0, 52, 50, 50, 50, 50, 52, 50, 50, 50, 75, 52, 50, 50, 50, 50,
    50, 50, 52, 52, 52, 52, 52, 52, 50, 50, 52, 50, 50, 75, 76, 50,
    77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 86, 87, 88, 89, 90, 91,
    9, 92, 93, 9, 64, 94, 9, 85, 0, 0, 0, 0, 0, 0, 0, 0,
    95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95,
    95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 0, 0, 0, 0, 95,
    95, 95, 95, 96, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    97, 97, 97, 97, 97, 97, 14, 14, 14, 9, 9, 10, 15, 15, 27, 27,
    64, 64, 64, 64, 64, 64, 64, 64, 98, 99, 100, 9, 32, 7, 7, 7,
    42, 42, 101, 101, 101, 101, 101, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    46, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 102, 103, 104, 98, 99,
    100, 105, 106, 107, 107, 108, 94, 64, 50, 64, 64, 64, 94, 64, 64, 94,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 9, 109, 110, 9, 42, 42,
    111, 42, 42, 42, 42, 112, 112, 112, 112, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    101, 42, 101, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 101, 7, 42, 64, 64, 64, 64, 64, 64, 64, 97, 27, 50,
    50, 64, 64, 94, 64, 46, 46, 64, 64, 27, 52, 50, 50, 94, 42, 42,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 42, 42, 42, 27, 27, 42,
    7, 7, 7, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0, 113,
    42, 114, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    64, 94, 64, 64, 94, 64, 64, 94, 94, 94, 64, 94, 94, 64, 94, 64,
    50, 50, 52, 50, 52, 50, 52, 50, 52, 50, 50, 0, 0, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 115, 115, 115, 115, 115, 115, 115, 115, 115, 115,
    115, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 50, 50, 50, 50, 50,
    50, 50, 52, 50, 46, 46, 27, 9, 15, 7, 46, 0, 0, 52, 10, 10,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 64, 64, 50, 50, 46, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 46, 64, 64, 64, 46, 64, 64, 64, 64, 50, 0, 0,
    9, 9, 9, 9, 9, 9, 9, 7, 9, 7, 9, 9, 9, 7, 7, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 52, 52, 52, 0, 0, 9, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 21, 42, 42, 42, 42, 42, 42, 42,
    97, 97, 0, 0, 0, 0, 0, 64, 50, 52, 52, 52, 50, 50, 50, 50,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 46, 50, 50, 50, 50, 50, 52,
    52, 52, 52, 52, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    50, 50, 97, 94, 64, 64, 94, 64, 64, 94, 50, 50, 50, 52, 52, 52,
    102, 103, 104, 64, 64, 64, 94, 64, 64, 94, 94, 64, 64, 64, 64, 64,
    115, 115, 115, 116, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117,
    117, 117, 117, 117, 117, 117, 117, 117, 117, 118, 117, 117, 117, 117, 117, 117,
    117, 118, 117, 117, 118, 117, 117, 117, 117, 117, 115, 116, 119, 42, 116, 116,
    116, 115, 115, 115, 115, 115, 115, 115, 115, 116, 116, 116, 116, 120, 116, 116,
    42, 50, 52, 50, 50, 115, 115, 115, 121, 121, 121, 121, 121, 121, 121, 121,
    42, 42, 115, 115, 7, 7, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
    9, 46, 42, 42, 42, 42, 42, 42, 117, 117, 117, 117, 117, 117, 117, 117,
    42, 115, 116, 116, 0, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 42,
    42, 0, 0, 42, 42, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117,
    117, 117, 117, 117, 117, 117, 117, 117, 117, 0, 117, 117, 117, 117, 117, 117,
    117, 0, 117, 0, 0, 0, 117, 117, 117, 117, 0, 0, 122, 42, 123, 116,
    116, 115, 115, 115, 115, 0, 0, 116, 116, 0, 0, 124, 124, 120, 42, 0,
    0, 0, 0, 0, 0, 0, 0, 123, 0, 0, 0, 0, 121, 121, 0, 121,
    42, 42, 115, 115, 0, 0, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
    117, 117, 10, 10, 125, 125, 125, 125, 125, 125, 27, 10, 42, 9, 50, 0,
    0, 115, 115, 116, 0, 42, 42, 42, 42, 42, 42, 0, 0, 0, 0, 42,
    42, 0, 0, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 42, 42, 42, 42, 42, 42,
    42, 0, 42, 126, 0, 42, 126, 0, 42, 42, 0, 0, 122, 0, 116, 116,
    116, 115, 115, 0, 0, 0, 0, 115, 115, 0, 0, 115, 115, 127, 0, 0,
    0, 115, 0, 0, 0, 0, 0, 0, 0, 126, 126, 126, 42, 0, 126, 0,
    0, 0, 0, 0, 0, 0, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
    115, 115, 42, 42, 42, 115, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 115, 115, 116, 0, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 42,
    42, 42, 0, 42, 42, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117,
    117, 117, 117, 117, 117, 117, 117, 117, 117, 0, 117, 117, 117, 117, 117, 117,
    117, 0, 117, 117, 0, 117, 117, 117, 117, 117, 0, 0, 122, 42, 116, 116,
    116, 115, 115, 115, 115, 115, 0, 115, 115, 116, 0, 116, 116, 120, 0, 0,
    42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 115, 115, 0, 0, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
    9, 10, 0, 0, 0, 0, 0, 0, 0, 117, 115, 115, 115, 61, 61, 61,
    0, 115, 116, 116, 0, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 42,
    42, 0, 0, 42, 42, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117,
    117, 117, 117, 117, 117, 117, 117, 117, 117, 0, 117, 117, 117, 117, 117, 117,
    117, 0, 117, 117, 0, 117, 117, 117, 117, 117, 0, 0, 122, 42, 123, 115,
    116, 115, 115, 115, 115, 0, 0, 116, 124, 0, 0, 124, 124, 120, 0, 0,
    0, 0, 0, 0, 0, 61, 128, 123, 0, 0, 0, 0, 121, 121, 0, 117,
    42, 42, 115, 115, 0, 0, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
    27, 117, 125, 125, 125, 125, 125, 125, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 115, 42, 0, 42, 42, 42, 42, 42, 42, 0, 0, 0, 42, 42,
    42, 0, 42, 42, 101, 42, 0, 0, 0, 42, 42, 0, 42, 0, 42, 42,
    0, 0, 0, 42, 42, 0, 0, 0, 42, 42, 42, 0, 0, 0, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 0, 0, 123, 116,
    115, 116, 116, 0, 0, 0, 116, 116, 116, 0, 124, 124, 124, 127, 0, 0,
    42, 0, 0, 0, 0, 0, 0, 123, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
    125, 125, 125, 27, 27, 27, 27, 27, 27, 10, 27, 0, 0, 0, 0, 0,
    115, 116, 116, 116, 115, 42, 42, 42, 42, 42, 42, 42, 42, 0, 42, 42,
    42, 0, 42, 42, 42, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117,
    117, 117, 117, 117, 117, 117, 117, 117, 117, 0, 117, 117, 117, 117, 117, 117,
    117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 0, 0, 122, 42, 115, 115,
    115, 116, 116, 116, 116, 0, 115, 115, 129, 0, 115, 115, 115, 120, 0, 0,
    0, 0, 0, 0, 0, 130, 131, 0, 117, 117, 117, 0, 42, 42, 0, 0,
    42, 42, 115, 115, 0, 0, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
    0, 0, 0, 0, 0, 0, 0, 9, 125, 125, 125, 125, 125, 125, 125, 27,
    42, 115, 116, 116, 9, 42, 42, 42, 42, 42, 42, 42, 42, 0, 42, 42,
    42, 0, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 0, 42, 42, 42, 42, 42, 0, 0, 122, 42, 116, 115,
    132, 116, 123, 116, 116, 0, 115, 132, 132, 0, 132, 132, 115, 127, 0, 0,
    0, 0, 0, 0, 0, 123, 123, 0, 0, 0, 0, 0, 42, 42, 42, 0,
    42, 42, 115, 115, 0, 0, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
    0, 42, 42, 116, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    115, 115, 116, 116, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 42, 42,
    42, 0, 42, 42, 42, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117,
    117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117,
    117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 127, 127, 42, 123, 116,
    116, 115, 115, 115, 115, 0, 116, 116, 116, 0, 124, 124, 124, 120, 133, 27,
    0, 0, 0, 0, 42, 42, 42, 123, 125, 125, 125, 125, 125, 125, 125, 42,
    42, 42, 115, 115, 0, 0, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
    125, 125, 125, 125, 125, 125, 125, 125, 125, 27, 42, 42, 42, 42, 42, 42,
    0, 115, 116, 116, 0, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 0, 0, 0, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 0, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 42, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 0, 0, 0, 134, 0, 0, 0, 0, 123,
    116, 116, 115, 115, 115, 0, 115, 0, 116, 116, 124, 116, 124, 124, 124, 123,
    0, 0, 0, 0, 0, 0, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
    0, 0, 116, 116, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 115, 135, 136, 115, 115, 115, 115, 137, 137, 138, 0, 0, 0, 0, 10,
    135, 135, 135, 135, 135, 135, 139, 61, 140, 140, 140, 140, 61, 115, 61, 9,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 9, 9, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 135, 135, 0, 135, 0, 135, 135, 135, 135, 135, 0, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 0, 135, 0, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 115, 135, 136, 115, 115, 115, 115, 141, 141, 127, 115, 115, 135, 0, 0,
    135, 135, 135, 135, 135, 0, 139, 0, 142, 142, 142, 142, 61, 115, 61, 0,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 143, 143, 135, 135,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 27, 27, 27, 9, 9, 9, 9, 9, 9, 9, 9, 144, 9, 9, 9,
    9, 9, 9, 27, 9, 27, 27, 27, 52, 52, 27, 27, 27, 27, 27, 27,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 125, 125, 125, 125, 125, 125,
    125, 125, 125, 125, 27, 52, 27, 52, 27, 145, 12, 13, 12, 13, 146, 146,
    42, 42, 42, 126, 42, 42, 42, 42, 0, 42, 42, 42, 42, 126, 42, 42,
    42, 42, 126, 42, 42, 42, 42, 126, 42, 42, 42, 42, 126, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 126, 42, 42, 42, 0, 0, 0,
    0, 147, 148, 149, 150, 149, 149, 151, 149, 151, 148, 148, 148, 148, 115, 116,
    148, 149, 64, 64, 127, 9, 50, 50, 42, 42, 42, 42, 42, 115, 115, 115,
    115, 115, 115, 149, 115, 115, 115, 115, 0, 115, 115, 115, 115, 149, 115, 115,
    115, 115, 149, 115, 115, 115, 115, 149, 115, 115, 115, 115, 149, 115, 115, 115,
    115, 115, 115, 115, 115, 115, 115, 115, 115, 149, 115, 115, 115, 0, 27, 27,
    27, 27, 27, 27, 27, 27, 52, 27, 27, 27, 27, 27, 27, 0, 27, 27,
    9, 9, 9, 9, 9, 27, 27, 27, 27, 9, 9, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152,
    152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152,
    152, 152, 152, 152, 152, 152, 153, 152, 152, 152, 152, 154, 154, 115, 128, 115,
    115, 116, 115, 115, 115, 115, 115, 122, 154, 120, 127, 116, 116, 115, 115, 152,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 7, 7, 9, 9, 9, 9,
    152, 152, 152, 152, 152, 152, 116, 116, 115, 115, 152, 152, 152, 152, 115, 115,
    115, 152, 154, 154, 154, 152, 152, 154, 154, 154, 154, 154, 154, 154, 152, 152,
    152, 115, 115, 115, 115, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152,
    152, 152, 115, 154, 116, 115, 115, 154, 154, 154, 154, 154, 154, 94, 152, 154,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 154, 154, 154, 115, 27, 27,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 0, 20, 0, 0, 0, 0, 0, 20, 0, 0,
    155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155,
    155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155,
    155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 9, 44, 155, 155, 155,
    156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156,
    156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156,
    156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156,
    156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156,
    156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156,
    156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156,
    157, 158, 158, 158, 158, 158, 158, 158, 158, 158, 158, 158, 158, 158, 158, 158,
    158, 158, 158, 158, 158, 158, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157,
    157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157,
    157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157,
    157, 157, 157, 157, 157, 157, 157, 157, 159, 159, 159, 159, 159, 159, 159, 159,
    159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159,
    159, 159, 159, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160,
    160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160,
    160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160,
    160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 42, 42, 42, 42, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 0, 42, 0, 42, 42, 42, 42, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 42, 42, 42, 42, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 0, 42, 42, 42, 42, 0, 0, 42, 42, 42, 42, 42, 42, 42, 0,
    42, 0, 42, 42, 42, 42, 0, 0, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 0, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 0, 42, 42, 42, 42, 0, 0, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 50, 50, 50,
    9, 9, 7, 9, 9, 9, 9, 7, 7, 125, 125, 125, 125, 125, 125, 125,
    125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 0, 0, 0, 0, 0, 0,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 0, 0, 23, 23, 23, 23, 23, 23, 0, 0,
    90, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 27, 7, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    6, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 12, 13, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 9, 9, 9, 161, 161,
    161, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 115, 115, 127, 162, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 115, 115, 162, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 115, 115, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 42, 42,
    42, 0, 115, 115, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152,
    152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152,
    152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152,
    152, 152, 152, 152, 61, 61, 116, 115, 115, 115, 115, 115, 115, 115, 116, 116,
    116, 116, 116, 116, 116, 116, 115, 116, 116, 61, 61, 61, 61, 61, 61, 61,
    61, 61, 120, 61, 7, 7, 9, 139, 9, 9, 9, 10, 135, 50, 0, 0,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 0, 0, 0, 0,
    125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 0, 0, 0, 0, 0, 0,
    9, 9, 72, 7, 9, 9, 90, 9, 72, 7, 9, 61, 61, 61, 32, 61,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 46, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 115, 115, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 163, 42, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0,
    115, 115, 115, 116, 116, 116, 116, 115, 115, 116, 116, 116, 0, 0, 0, 0,
    116, 116, 115, 116, 116, 116, 116, 116, 116, 75, 50, 52, 0, 0, 0, 0,
    27, 0, 0, 0, 7, 7, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 0, 0,
    135, 135, 135, 135, 135, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 0, 0, 0, 0,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 0, 0, 0, 0, 0, 0,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 164, 0, 0, 0, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 64, 94, 116, 116, 115, 0, 0, 9, 9,
    152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152,
    152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152,
    152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152,
    152, 152, 152, 152, 152, 116, 115, 116, 115, 115, 115, 115, 115, 115, 115, 0,
    120, 154, 115, 154, 154, 115, 115, 115, 115, 115, 115, 115, 115, 116, 116, 116,
    116, 116, 116, 115, 115, 50, 50, 50, 50, 50, 50, 50, 50, 0, 0, 52,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 0, 0, 0, 0,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 0, 0, 0, 0,
    9, 9, 9, 9, 9, 9, 9, 139, 7, 7, 7, 7, 9, 9, 0, 0,
    50, 50, 50, 50, 50, 52, 52, 52, 52, 52, 52, 50, 50, 52, 70, 94,
    94, 50, 50, 52, 52, 50, 50, 50, 50, 50, 52, 50, 64, 64, 64, 50,
    50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 52, 0, 0,
    50, 50, 50, 50, 50, 50, 52, 50, 50, 50, 50, 63, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    115, 115, 115, 115, 116, 42, 101, 42, 101, 42, 101, 117, 118, 42, 101, 42,
    42, 42, 101, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117,
    117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117,
    117, 117, 117, 117, 122, 123, 115, 115, 115, 115, 115, 132, 115, 132, 116, 116,
    124, 124, 115, 132, 165, 117, 117, 117, 117, 117, 117, 117, 117, 0, 7, 7,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 7, 7, 9, 9, 7, 7,
    9, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 50, 52, 50, 50, 50,
    50, 50, 50, 50, 27, 27, 27, 27, 27, 27, 27, 27, 27, 7, 7, 7,
    115, 115, 116, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117,
    117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117,
    117, 116, 115, 115, 115, 115, 116, 116, 115, 115, 162, 120, 115, 115, 117, 117,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 42, 117, 117, 117, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 122, 116, 115, 115, 116, 116, 116, 115, 116, 115,
    115, 115, 162, 162, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 116, 116, 116, 116, 116, 116, 116, 116, 115, 115, 115, 115,
    115, 115, 115, 115, 116, 116, 115, 122, 0, 0, 0, 7, 7, 9, 9, 9,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 0, 42, 42, 42,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 46, 46, 46, 46, 46, 46, 7, 7,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 20, 23, 0, 0, 0, 0, 0,
    166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166,
    166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166,
    166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 0, 0, 166, 166, 166,
    9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 9, 57, 52, 52, 52, 52, 52, 50, 50, 52, 52, 52, 52,
    50, 146, 57, 57, 57, 57, 57, 57, 57, 42, 42, 42, 42, 52, 42, 42,
    42, 42, 42, 42, 50, 42, 42, 146, 50, 50, 42, 0, 0, 0, 0, 0,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 44, 44, 44, 47,
    44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 47, 44, 44, 44, 44,
    44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 47, 44,
    44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44,
    44, 44, 45, 44, 44, 44, 44, 44, 44, 44, 44, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 44, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 24, 23, 23, 23, 23, 44, 44, 44, 44, 44,
    44, 44, 44, 44, 45, 44, 44, 44, 45, 44, 44, 44, 44, 44, 44, 44,
    44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44,
    50, 50, 52, 50, 50, 50, 50, 50, 50, 50, 52, 50, 50, 63, 167, 52,
    54, 50, 50, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 50, 51, 76, 76, 52, 168, 50, 62, 52, 50, 52,
    38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39,
    38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39,
    38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 40, 38, 39,
    38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39,
    38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39,
    38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39,
    38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39,
    38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39,
    38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39,
    38, 39, 38, 39, 38, 39, 39, 39, 39, 39, 34, 169, 23, 23, 20, 23,
    38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39,
    38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39,
    38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 40, 38, 39, 38, 39,
    38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39,
    38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39,
    38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 20, 23, 20, 23, 20, 23,
    39, 39, 39, 39, 39, 39, 39, 39, 38, 38, 38, 38, 38, 38, 38, 38,
    39, 39, 39, 39, 39, 39, 0, 0, 38, 38, 38, 38, 38, 38, 0, 0,
    39, 39, 39, 39, 39, 39, 39, 39, 38, 38, 38, 38, 38, 38, 38, 38,
    39, 39, 39, 39, 39, 39, 39, 39, 38, 38, 38, 38, 38, 38, 38, 38,
    39, 39, 39, 39, 39, 39, 0, 0, 38, 38, 38, 38, 38, 38, 0, 0,
    39, 39, 39, 39, 39, 39, 39, 39, 0, 38, 0, 38, 0, 38, 0, 38,
    39, 39, 39, 39, 39, 39, 39, 39, 38, 38, 38, 38, 38, 38, 38, 38,
    39, 170, 39, 170, 39, 170, 39, 170, 39, 170, 39, 170, 39, 170, 0, 0,
    39, 39, 39, 39, 39, 39, 39, 39, 171, 171, 171, 171, 171, 171, 171, 171,
    39, 39, 39, 39, 39, 39, 39, 39, 171, 171, 171, 171, 171, 171, 171, 171,
    39, 39, 39, 39, 39, 39, 39, 39, 171, 171, 171, 171, 171, 171, 171, 171,
    39, 39, 39, 39, 39, 0, 39, 39, 38, 38, 38, 172, 171, 28, 170, 28,
    28, 67, 39, 39, 39, 0, 39, 39, 38, 172, 38, 172, 171, 67, 67, 67,
    39, 39, 39, 170, 0, 0, 39, 39, 38, 38, 38, 172, 0, 67, 67, 67,
    39, 39, 39, 170, 39, 39, 39, 39, 38, 38, 38, 172, 38, 67, 173, 173,
    0, 0, 39, 39, 39, 0, 39, 39, 38, 172, 38, 172, 171, 173, 28, 0,
    174, 174, 175, 175, 175, 175, 175, 26, 175, 175, 175, 176, 177, 178, 32, 32,
    90, 179, 90, 16, 16, 90, 9, 144, 180, 181, 12, 31, 31, 37, 12, 31,
    9, 9, 9, 9, 182, 144, 144, 35, 183, 184, 32, 32, 32, 32, 32, 185,
    9, 9, 9, 144, 144, 9, 144, 144, 9, 31, 37, 9, 186, 7, 144, 22,
    22, 9, 9, 9, 187, 12, 13, 188, 188, 186, 9, 9, 9, 9, 9, 9,
    9, 9, 14, 9, 22, 9, 9, 144, 9, 9, 9, 9, 9, 9, 9, 175,
    32, 32, 32, 32, 32, 189, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
    33, 45, 0, 0, 33, 33, 33, 33, 33, 33, 190, 190, 190, 191, 192, 44,
    33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 190, 190, 190, 191, 192, 0,
    44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 0, 0, 0,
    10, 10, 10, 10, 10, 10, 10, 10, 193, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 57, 57, 50, 50, 50, 50, 57, 57, 57, 50, 50, 70, 70, 70,
    70, 50, 70, 70, 70, 57, 57, 50, 52, 50, 57, 57, 52, 52, 52, 52,
    50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    194, 194, 41, 194, 27, 194, 194, 41, 27, 194, 34, 41, 41, 41, 34, 34,
    41, 41, 41, 34, 27, 41, 194, 27, 14, 41, 41, 41, 41, 41, 27, 27,
    194, 194, 195, 27, 41, 27, 172, 27, 41, 27, 172, 172, 41, 41, 27, 34,
    41, 41, 20, 41, 34, 112, 112, 112, 112, 196, 27, 194, 34, 34, 41, 41,
    190, 14, 14, 14, 14, 41, 34, 34, 197, 197, 27, 14, 27, 27, 23, 27,
    33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33,
    198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198,
    199, 199, 199, 199, 199, 199, 199, 199, 199, 199, 199, 199, 199, 199, 199, 199,
    161, 161, 161, 20, 23, 161, 161, 161, 161, 33, 27, 27, 0, 0, 0, 0,
    14, 14, 14, 14, 200, 29, 29, 29, 29, 29, 201, 201, 27, 27, 27, 27,
    14, 27, 27, 14, 27, 27, 14, 27, 27, 29, 29, 27, 27, 27, 201, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 202, 201, 201,
    27, 27, 14, 27, 14, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 201, 14, 14, 14, 14, 201, 14, 14, 201, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 201, 14, 201, 14, 14, 14, 14, 14, 190, 190, 14, 190,
    190, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 201, 14, 14, 201, 14, 14, 201, 14, 201, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    201, 14, 201, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 201, 201, 201,
    201, 201, 14, 14, 201, 201, 14, 14, 201, 201, 14, 14, 14, 14, 14, 14,
    201, 201, 14, 14, 201, 201, 14, 14, 201, 201, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 201, 201, 201, 201,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    201, 201, 201, 201, 14, 14, 14, 14, 14, 14, 201, 201, 201, 201, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    27, 27, 27, 27, 27, 27, 27, 27, 12, 13, 12, 13, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 29, 29, 27, 27, 27, 27,
    14, 14, 27, 27, 27, 27, 27, 27, 29, 203, 204, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 14, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 29,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 14, 14, 14, 14,
    14, 14, 27, 27, 27, 27, 27, 27, 27, 29, 29, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 27, 27, 27, 27, 29, 29, 29, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33,
    33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33,
    33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33,
    33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 194, 194, 194, 194,
    194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
    194, 194, 194, 194, 194, 194, 205, 205, 205, 205, 205, 205, 205, 205, 205, 205,
    205, 205, 206, 205, 205, 205, 205, 205, 205, 205, 205, 205, 205, 205, 205, 205,
    207, 207, 207, 207, 207, 207, 207, 207, 207, 207, 207, 207, 207, 207, 207, 207,
    207, 207, 207, 207, 207, 207, 207, 207, 207, 207, 33, 125, 125, 125, 125, 125,
    125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 29, 29, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 29, 14, 27, 27, 27, 27, 27, 27, 27, 27,
    29, 14, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 14, 14, 14, 200, 200, 200, 200, 14,
    29, 29, 29, 29, 29, 27, 27, 27, 27, 27, 27, 27, 27, 27, 29, 27,
    27, 29, 27, 27, 29, 29, 27, 27, 29, 27, 27, 27, 27, 29, 27, 27,
    29, 27, 29, 29, 27, 27, 29, 27, 27, 27, 29, 27, 27, 27, 29, 29,
    27, 27, 27, 27, 27, 27, 27, 27, 29, 29, 29, 27, 27, 27, 27, 27,
    29, 27, 29, 27, 27, 27, 27, 27, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 29,
    29, 27, 27, 29, 27, 29, 29, 27, 29, 27, 27, 27, 27, 27, 27, 14,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 29, 27, 27, 29, 29,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 29, 29, 29, 29, 29, 29, 27, 29, 27, 29, 29, 27, 27, 27,
    29, 29, 27, 27, 27, 27, 27, 29, 27, 27, 29, 29, 27, 27, 27, 27,
    29, 29, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 29, 29, 27,
    27, 27, 27, 27, 29, 29, 27, 27, 29, 27, 27, 27, 27, 27, 29, 29,
    27, 29, 27, 29, 29, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 29, 29, 27, 27, 27, 27, 27,
    29, 29, 29, 29, 29, 29, 27, 29, 29, 29, 29, 27, 27, 29, 27, 27,
    27, 27, 29, 27, 27, 29, 27, 27, 29, 29, 29, 29, 29, 29, 27, 29,
    27, 27, 29, 27, 29, 27, 29, 27, 27, 27, 27, 27, 27, 29, 27, 27,
    27, 29, 27, 27, 27, 27, 27, 27, 29, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 29, 29, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 29, 27, 27, 29, 27, 27, 27, 27, 29, 27, 29, 27,
    27, 27, 27, 29, 29, 29, 27, 29, 27, 27, 27, 208, 208, 208, 208, 208,
    208, 27, 27, 29, 29, 27, 27, 27, 12, 13, 12, 13, 12, 13, 12, 13,
    12, 13, 12, 13, 12, 13, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    125, 125, 125, 125, 27, 29, 29, 29, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 29, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    29, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 29,
    14, 14, 14, 14, 14, 12, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 200, 200, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12,
    13, 12, 13, 12, 13, 12, 13, 12, 13, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 12, 13, 12, 13, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 12, 13, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 190, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 190, 190, 190, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 209, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    27, 27, 27, 27, 27, 29, 29, 29, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 29, 29, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 27, 27, 14, 14, 14, 14, 14, 14, 27, 27, 27,
    29, 27, 27, 27, 27, 29, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 0, 0, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    20, 23, 20, 20, 20, 23, 23, 20, 23, 20, 23, 20, 23, 20, 20, 20,
    20, 23, 20, 23, 23, 20, 23, 23, 23, 23, 23, 23, 45, 44, 20, 20,
    20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23,
    20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23,
    20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23,
    20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23,
    20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23,
    20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23,
    20, 23, 20, 23, 23, 27, 27, 27, 27, 27, 27, 20, 23, 20, 23, 50,
    50, 50, 20, 23, 0, 0, 0, 0, 0, 7, 7, 7, 9, 125, 9, 9,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 0, 23, 0, 0, 0, 0, 0, 23, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 0, 0, 0, 0, 0, 210,
    9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 0, 42, 42, 42, 42, 42, 42, 42, 0,
    42, 42, 42, 42, 42, 42, 42, 0, 42, 42, 42, 42, 42, 42, 42, 0,
    42, 42, 42, 42, 42, 42, 42, 0, 42, 42, 42, 42, 42, 42, 42, 0,
    42, 42, 42, 42, 42, 42, 42, 0, 42, 42, 42, 42, 42, 42, 42, 0,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    211, 211, 31, 37, 31, 37, 211, 211, 211, 31, 37, 211, 31, 37, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 90, 9, 9, 90, 9, 31, 37, 9, 9,
    31, 37, 12, 13, 12, 13, 12, 13, 12, 13, 9, 9, 9, 9, 7, 46,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 90, 90, 7, 9, 9, 9,
    90, 9, 12, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    27, 27, 9, 7, 7, 12, 13, 12, 13, 12, 13, 12, 13, 90, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 0, 27, 27, 27, 27, 194,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 194, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
    194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
    194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
    194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
    194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
    194, 194, 194, 194, 194, 194, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    175, 72, 7, 9, 27, 46, 135, 212, 12, 13, 12, 13, 12, 13, 12, 13,
    12, 13, 27, 27, 12, 13, 12, 13, 12, 13, 12, 13, 90, 12, 13, 13,
    27, 212, 212, 212, 212, 212, 212, 212, 212, 212, 168, 76, 51, 75, 213, 213,
    214, 215, 215, 215, 215, 215, 194, 27, 216, 216, 216, 46, 42, 217, 27, 27,
    0, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 218, 135, 218, 135,
    218, 135, 218, 135, 218, 135, 218, 135, 218, 135, 218, 135, 218, 135, 218, 135,
    218, 135, 218, 135, 135, 218, 135, 218, 135, 218, 135, 135, 135, 135, 135, 135,
    218, 218, 135, 218, 218, 135, 218, 218, 135, 218, 218, 135, 218, 218, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 218, 135, 135, 0, 0, 219, 219, 220, 220, 139, 221, 143,
    222, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 224, 223, 224, 223,
    224, 223, 224, 223, 224, 223, 224, 223, 224, 223, 224, 223, 224, 223, 224, 223,
    224, 223, 224, 223, 223, 224, 223, 224, 223, 224, 223, 223, 223, 223, 223, 223,
    224, 224, 223, 224, 224, 223, 224, 224, 223, 224, 224, 223, 224, 224, 223, 223,
    223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223,
    223, 223, 223, 223, 224, 223, 223, 224, 224, 224, 224, 9, 215, 215, 225, 226,
    0, 0, 0, 0, 0, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    0, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 0,
    27, 27, 33, 33, 33, 33, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27,
    223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223,
    194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
    194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 0,
    33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 194, 194, 194, 194, 194, 194,
    194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
    194, 194, 194, 194, 194, 194, 194, 194, 125, 125, 125, 125, 125, 125, 125, 125,
    194, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33,
    194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
    194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 27,
    33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 194, 194, 194, 194, 194, 194,
    194, 194, 194, 194, 194, 194, 194, 195, 194, 195, 194, 194, 194, 194, 194, 194,
    194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
    194, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33,
    194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
    227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227,
    227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227,
    227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 194,
    227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227,
    227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227,
    227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227,
    227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227,
    227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227,
    227, 227, 227, 227, 227, 227, 227, 227, 194, 194, 194, 194, 194, 194, 194, 194,
    194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
    194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 46, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 0,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 46, 46, 46, 46, 46, 46, 9, 7,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 46, 9, 7, 7,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 42, 42, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23,
    20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23,
    20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 42, 50,
    70, 70, 70, 9, 64, 64, 64, 64, 64, 64, 64, 64, 50, 50, 9, 46,
    20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23,
    20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 44, 44, 64, 64,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161,
    50, 50, 9, 7, 9, 9, 9, 7, 0, 0, 0, 0, 0, 0, 0, 0,
    21, 21, 21, 21, 21, 21, 21, 21, 48, 48, 48, 48, 48, 48, 48, 48,
    48, 48, 48, 48, 48, 48, 48, 46, 46, 46, 46, 46, 46, 46, 46, 46,
    48, 48, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23,
    23, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23,
    20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23,
    20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23,
    20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23,
    44, 23, 23, 23, 23, 23, 23, 23, 23, 20, 23, 20, 23, 20, 20, 23,
    20, 23, 20, 23, 20, 23, 20, 23, 46, 48, 48, 20, 23, 20, 23, 42,
    20, 23, 20, 23, 23, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23,
    20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 20, 20, 20, 20, 23,
    20, 20, 20, 20, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23,
    20, 23, 20, 23, 20, 20, 20, 20, 23, 20, 23, 20, 20, 23, 20, 23,
    20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 23, 20, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 44, 44, 44, 44, 20, 23, 42, 44, 44, 23, 42, 42, 42, 42, 42,
    42, 42, 115, 42, 42, 42, 127, 42, 42, 42, 42, 115, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 116, 116, 115, 115, 116, 27, 27, 27, 27, 127, 0, 0, 0,
    125, 125, 125, 125, 125, 125, 27, 27, 10, 27, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 9, 9, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0,
    116, 116, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116,
    116, 116, 116, 116, 127, 115, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
    50, 50, 42, 42, 42, 42, 42, 42, 9, 9, 9, 42, 9, 42, 42, 115,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 115, 115, 115, 115, 115, 52, 52, 52, 9, 7,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 115, 115, 115, 115, 115, 115, 115, 115, 115,
    115, 115, 116, 162, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9,
    156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156,
    156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 0, 0, 0,
    115, 115, 115, 116, 42, 42, 42, 42, 42, 117, 117, 117, 42, 42, 42, 117,
    117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117,
    117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117,
    117, 117, 117, 122, 116, 116, 115, 115, 115, 115, 116, 116, 115, 115, 116, 116,
    165, 9, 9, 9, 9, 9, 9, 9, 7, 7, 9, 9, 9, 9, 0, 46,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 0, 0, 9, 9,
    152, 152, 152, 152, 152, 115, 139, 152, 152, 152, 152, 152, 152, 152, 152, 152,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 152, 152, 152, 152, 152, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 115, 115, 115, 115, 115, 115, 116,
    116, 115, 115, 116, 116, 115, 115, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 115, 42, 42, 42, 42, 42, 42, 42, 42, 115, 116, 0, 0,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 9, 7, 7, 7,
    152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152,
    139, 152, 152, 152, 135, 135, 135, 27, 27, 27, 152, 154, 115, 154, 152, 152,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    64, 135, 64, 64, 94, 135, 135, 64, 64, 135, 135, 135, 135, 135, 64, 50,
    135, 50, 135, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 135, 135, 139, 9, 9,
    117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 116, 115, 115, 116, 116,
    7, 7, 42, 46, 46, 116, 120, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 42, 42, 42, 42, 42, 42, 0, 0, 42, 42, 42, 42, 42, 42, 0,
    0, 42, 42, 42, 42, 42, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 0, 42, 42, 42, 42, 42, 42, 42, 0,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 48, 44, 44, 44, 44,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 44, 21, 21, 0, 0, 0, 0,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117,
    117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 42, 42, 42, 42, 42,
    42, 42, 42, 116, 116, 115, 116, 116, 115, 116, 116, 7, 146, 127, 0, 0,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 0, 0, 0, 0,
    228, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229,
    229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 228, 229, 229, 229,
    229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229,
    229, 229, 229, 229, 229, 229, 229, 229, 228, 229, 229, 229, 229, 229, 229, 229,
    229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229,
    229, 229, 229, 229, 228, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229,
    229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229,
    228, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229,
    229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 228, 229, 229, 229,
    229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229,
    229, 229, 229, 229, 229, 229, 229, 229, 228, 229, 229, 229, 229, 229, 229, 229,
    229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229,
    229, 229, 229, 229, 228, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229,
    229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229,
    228, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229,
    229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 228, 229, 229, 229,
    229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229,
    229, 229, 229, 229, 229, 229, 229, 229, 228, 229, 229, 229, 229, 229, 229, 229,
    229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229,
    229, 229, 229, 229, 228, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229,
    229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229,
    228, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229,
    229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 228, 229, 229, 229,
    229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229,
    229, 229, 229, 229, 229, 229, 229, 229, 228, 229, 229, 229, 229, 229, 229, 229,
    229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229,
    229, 229, 229, 229, 228, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229,
    229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229,
    229, 229, 229, 229, 229, 229, 229, 229, 228, 229, 229, 229, 229, 229, 229, 229,
    229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229,
    229, 229, 229, 229, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157,
    157, 157, 157, 157, 157, 157, 157, 0, 0, 0, 0, 160, 160, 160, 160, 160,
    160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160,
    160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160,
    160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 0, 0, 0, 0,
    230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230,
    230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230,
    230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230,
    230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230,
    231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231,
    231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231,
    231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231,
    231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231,
    232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232,
    232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232,
    232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232,
    232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232,
    232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 135, 135,
    232, 135, 232, 135, 135, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 135,
    232, 135, 232, 135, 135, 232, 232, 135, 135, 135, 232, 232, 232, 232, 232, 232,
    232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232,
    232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232,
    232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232,
    232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 0, 0,
    232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232,
    232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232,
    232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    34, 34, 34, 34, 34, 34, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 34, 34, 34, 34, 34, 0, 0, 0, 0, 0, 233, 234, 233,
    235, 235, 235, 235, 235, 235, 235, 235, 235, 190, 233, 233, 233, 233, 233, 233,
    233, 233, 233, 233, 233, 233, 233, 0, 233, 233, 233, 233, 233, 0, 233, 0,
    233, 233, 0, 233, 233, 0, 233, 233, 233, 233, 233, 233, 233, 233, 233, 235,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    112, 112, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
    21, 21, 21, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 13, 12,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    27, 27, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    112, 112, 112, 112, 112, 112, 112, 112, 27, 27, 27, 27, 27, 27, 27, 27,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 193, 27, 27, 27,
    61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
    236, 236, 188, 237, 236, 188, 188, 191, 192, 144, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 52, 52, 52, 52, 52, 52, 52, 50, 50,
    144, 238, 238, 239, 239, 191, 192, 191, 192, 191, 192, 191, 192, 191, 192, 191,
    192, 191, 192, 191, 192, 9, 9, 191, 192, 144, 144, 144, 144, 239, 239, 239,
    240, 236, 182, 0, 240, 237, 188, 188, 238, 191, 192, 191, 192, 191, 192, 144,
    144, 144, 190, 238, 190, 190, 190, 0, 144, 193, 144, 144, 0, 0, 0, 0,
    112, 112, 112, 42, 112, 0, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 0, 0, 32,
    0, 188, 144, 144, 193, 144, 144, 241, 191, 192, 144, 190, 240, 238, 182, 144,
    242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 237, 240, 190, 190, 190, 188,
    144, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
    41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 191, 144, 192, 28, 239,
    28, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34,
    34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 191, 190, 192, 190, 191,
    192, 188, 191, 192, 236, 144, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226,
    243, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226,
    226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226,
    226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 244, 244,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 0,
    0, 0, 112, 112, 112, 112, 112, 112, 0, 0, 112, 112, 112, 112, 112, 112,
    0, 0, 112, 112, 112, 112, 112, 112, 0, 0, 112, 112, 112, 0, 0, 0,
    193, 193, 190, 28, 194, 193, 193, 0, 194, 190, 190, 190, 190, 194, 194, 0,
    189, 189, 189, 189, 189, 189, 189, 189, 189, 32, 32, 32, 27, 27, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 0, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 42, 42, 0, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 0, 0, 0,
    9, 9, 9, 0, 0, 0, 0, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    125, 125, 125, 125, 0, 0, 0, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161,
    161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161,
    161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161,
    161, 161, 161, 161, 161, 125, 125, 125, 125, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 125, 125, 27, 27, 27, 0,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 0, 0, 0,
    27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 52, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    52, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    125, 125, 125, 125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 161, 42, 42, 42, 42, 42, 42, 42, 42, 161, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 64, 64, 64, 64, 64, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 9,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 0, 0, 0, 0, 42, 42, 42, 42, 42, 42, 42, 42,
    9, 161, 161, 161, 161, 161, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 0, 0, 0, 0,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 0, 0, 0, 0, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 0, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 0, 20, 20, 20, 20,
    20, 20, 20, 0, 20, 20, 0, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 0, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 0, 23, 23, 23, 23, 23, 23, 23, 0, 23, 23, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 101, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 101, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    47, 210, 210, 44, 44, 44, 0, 44, 44, 44, 44, 44, 44, 44, 44, 44,
    44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44,
    44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44,
    44, 0, 44, 44, 44, 44, 44, 44, 44, 44, 44, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 0, 0, 42, 0, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 0, 42, 42, 0, 0, 0, 42, 0, 0, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 0, 9, 125, 125, 125, 125, 125, 125, 125, 125,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 27, 27, 125, 125, 125, 125, 125, 125, 125,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0,
    0, 0, 0, 0, 0, 0, 0, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 0, 42, 42, 0, 0, 0, 0, 0, 125, 125, 125, 125, 125,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 125, 125, 125, 125, 125, 125, 0, 0, 0, 9,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 0, 0, 0, 9,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 0, 0, 125, 125, 42, 42,
    125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    0, 0, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    117, 115, 115, 115, 0, 115, 115, 0, 0, 0, 0, 0, 115, 94, 115, 64,
    117, 117, 117, 117, 0, 117, 117, 117, 0, 117, 117, 117, 117, 117, 117, 117,
    117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117,
    117, 117, 117, 117, 117, 117, 0, 0, 50, 57, 52, 0, 0, 0, 0, 120,
    125, 125, 125, 125, 125, 125, 125, 125, 125, 0, 0, 0, 0, 0, 0, 0,
    9, 9, 9, 9, 9, 9, 7, 7, 9, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 125, 125, 9,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 125, 125, 125,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 27, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 50, 52, 0, 0, 0, 0, 125, 125, 125, 125, 125,
    9, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 0, 0, 0, 9, 9, 9, 9, 9, 9, 9,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 0, 0, 125, 125, 125, 125, 125, 125, 125, 125,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 0, 0, 0, 0, 0, 125, 125, 125, 125, 125, 125, 125, 125,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 125, 125, 125, 125, 125, 125, 125,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 0, 0, 0, 0, 0, 0, 0, 125, 125, 125, 125, 125, 125,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 64, 64, 64, 64, 0, 0, 0, 0, 0, 0, 0, 0,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 0, 0, 0, 0,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 42, 42, 42, 42, 46, 42,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 0, 0, 0, 64, 50, 50, 50, 50, 90, 46,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 0, 0, 0, 0, 0, 0, 0, 0, 14, 14,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 64, 64, 90, 0, 0,
    42, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 42, 42, 42, 46, 42, 42, 0, 0, 0, 0, 0, 0, 0, 0,
    9, 27, 27, 27, 27, 27, 27, 27, 27, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 94, 94, 115, 52, 52, 52,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 125, 125, 125,
    125, 125, 125, 125, 125, 125, 125, 42, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 52, 52, 50, 50, 50, 52, 50, 52, 52, 52,
    52, 125, 125, 125, 125, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 50, 52, 50, 52, 7, 7, 7, 7, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 125, 125, 125, 125, 125, 125, 125, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    116, 115, 116, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 115, 115, 115, 115, 115, 115, 115, 115,
    115, 115, 115, 115, 115, 115, 127, 7, 7, 9, 9, 9, 9, 9, 0, 0,
    0, 0, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    125, 125, 125, 125, 125, 125, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
    127, 42, 42, 115, 115, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127,
    115, 115, 116, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 101, 42, 101, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 101, 42, 42, 42, 42,
    116, 116, 116, 115, 115, 115, 115, 116, 116, 127, 119, 9, 9, 97, 7, 7,
    7, 7, 115, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 97, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 0, 0, 0, 0, 0,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 0, 0, 0, 0,
    64, 64, 64, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117,
    117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117,
    117, 117, 117, 117, 117, 117, 117, 128, 115, 115, 115, 115, 116, 115, 129, 129,
    115, 115, 115, 120, 127, 0, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
    9, 7, 7, 7, 117, 116, 116, 117, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 122, 9, 9, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    115, 115, 116, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 116, 116, 116, 115, 115, 115, 115, 115, 115, 115, 115, 115, 116,
    162, 42, 133, 133, 42, 7, 7, 9, 9, 61, 122, 61, 61, 7, 116, 115,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 42, 9, 42, 9, 7, 7,
    0, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    125, 125, 125, 125, 125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 0, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 116, 116, 116, 115,
    115, 115, 116, 116, 115, 162, 122, 115, 7, 7, 9, 7, 7, 9, 115, 42,
    42, 115, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 0, 42, 0, 42, 42, 42, 42, 0, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 7, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 115,
    116, 116, 116, 115, 115, 115, 115, 115, 115, 122, 127, 0, 0, 0, 0, 0,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 0, 0, 0, 0,
    115, 115, 116, 116, 0, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 42,
    42, 0, 0, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 42, 42, 42, 42, 42, 42,
    42, 0, 42, 42, 0, 42, 42, 42, 42, 42, 0, 122, 122, 42, 123, 116,
    115, 116, 116, 116, 116, 0, 0, 116, 116, 0, 0, 124, 124, 162, 0, 0,
    42, 0, 0, 0, 0, 0, 0, 123, 0, 0, 0, 0, 0, 42, 42, 42,
    42, 42, 116, 116, 0, 0, 50, 50, 50, 50, 50, 50, 50, 0, 0, 0,
    50, 50, 50, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    117, 117, 117, 118, 117, 118, 117, 117, 117, 117, 0, 117, 0, 0, 118, 0,
    117, 118, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117,
    117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117,
    117, 117, 117, 117, 117, 117, 0, 42, 123, 116, 116, 128, 115, 115, 115, 115,
    115, 0, 123, 0, 0, 245, 0, 245, 245, 123, 116, 0, 116, 116, 127, 162,
    120, 133, 61, 42, 7, 7, 0, 9, 9, 0, 0, 0, 0, 0, 0, 0,
    0, 61, 61, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 116, 116, 116, 115, 115, 115, 115, 115, 115, 115, 115,
    116, 116, 127, 115, 115, 116, 122, 42, 42, 42, 42, 7, 7, 9, 9, 9,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 9, 9, 0, 9, 50, 42,
    42, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    123, 116, 116, 115, 115, 115, 115, 115, 115, 116, 128, 124, 124, 123, 124, 115,
    115, 116, 127, 122, 42, 42, 9, 42, 0, 0, 0, 0, 0, 0, 0, 0,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 123,
    116, 116, 115, 115, 115, 115, 0, 0, 116, 116, 124, 124, 115, 115, 116, 127,
    122, 9, 7, 7, 9, 9, 9, 9, 9, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 42, 42, 42, 42, 115, 115, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    116, 116, 116, 115, 115, 115, 115, 115, 115, 115, 115, 116, 116, 115, 116, 127,
    115, 7, 7, 9, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 0, 0, 0, 0,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 115, 116, 115, 116, 116,
    115, 115, 115, 115, 115, 115, 162, 122, 42, 9, 0, 0, 0, 0, 0, 0,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 0, 0, 0, 0,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
    18, 18, 18, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 0, 0, 115, 116, 115,
    154, 154, 115, 115, 115, 115, 116, 115, 115, 115, 115, 127, 0, 0, 0, 0,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 125, 125, 7, 7, 7, 27,
    135, 135, 135, 135, 135, 135, 135, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 116, 116, 116, 115,
    115, 115, 115, 115, 115, 115, 115, 115, 116, 127, 122, 9, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 125, 125, 125, 125, 125, 125,
    125, 125, 125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42,
    117, 117, 117, 117, 117, 117, 117, 0, 0, 117, 0, 0, 117, 117, 117, 117,
    117, 117, 117, 117, 0, 117, 117, 0, 117, 117, 117, 117, 117, 117, 117, 117,
    117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117,
    123, 116, 116, 116, 116, 116, 0, 116, 124, 0, 0, 115, 115, 162, 120, 133,
    116, 133, 116, 122, 7, 9, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 116, 116, 116, 115, 115, 115, 115, 0, 0, 115, 115, 116, 116, 116, 116,
    127, 42, 9, 42, 116, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    117, 115, 115, 115, 115, 115, 115, 115, 115, 115, 115, 117, 117, 117, 117, 117,
    117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117,
    117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117,
    117, 117, 117, 61, 127, 115, 115, 115, 115, 116, 42, 115, 115, 115, 115, 9,
    9, 9, 7, 7, 9, 9, 9, 120, 0, 0, 0, 0, 0, 0, 0, 0,
    117, 115, 115, 115, 115, 115, 115, 116, 116, 115, 115, 115, 117, 117, 117, 117,
    117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117,
    117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117,
    117, 117, 117, 117, 133, 133, 133, 133, 133, 133, 115, 115, 115, 115, 115, 115,
    115, 115, 115, 115, 115, 115, 115, 116, 61, 120, 9, 7, 7, 42, 9, 9,
    9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 0, 0, 0, 0, 0,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    115, 116, 115, 115, 115, 116, 115, 116, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 116,
    115, 115, 115, 115, 115, 115, 115, 0, 115, 115, 115, 115, 115, 115, 116, 127,
    42, 7, 7, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 125, 125, 125, 125, 125, 125,
    125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 0, 0, 0,
    9, 9, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    0, 0, 115, 115, 115, 115, 115, 115, 115, 115, 115, 115, 115, 115, 115, 115,
    115, 115, 115, 115, 115, 115, 115, 115, 0, 116, 115, 115, 115, 115, 115, 115,
    115, 116, 115, 115, 116, 115, 115, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 0, 42, 42, 0, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 115, 115, 115, 115, 115, 115, 0, 0, 0, 115, 0, 115, 115, 0, 115,
    115, 115, 122, 115, 127, 127, 133, 115, 0, 0, 0, 0, 0, 0, 0, 0,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 0, 42, 42, 0, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 116, 116, 116, 116, 116, 0,
    115, 115, 0, 116, 116, 115, 116, 127, 42, 0, 0, 0, 0, 0, 0, 0,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 46, 42, 42, 0, 0, 0, 0,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 115, 115, 116, 116, 7, 7, 0, 0, 0, 0, 0, 0, 0,
    115, 115, 133, 116, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117,
    117, 0, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117,
    117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117,
    117, 117, 117, 117, 116, 116, 115, 115, 115, 115, 115, 0, 0, 0, 116, 116,
    115, 162, 120, 7, 7, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 61, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    125, 125, 125, 125, 125, 27, 27, 27, 27, 27, 27, 27, 27, 10, 10, 10,
    10, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9,
    161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161,
    161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161,
    161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161,
    161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161,
    161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161,
    161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161,
    161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 0,
    9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
    61, 42, 42, 42, 42, 42, 42, 61, 61, 61, 61, 61, 61, 61, 61, 61,
    61, 61, 61, 61, 61, 61, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 128, 128,
    128, 246, 246, 246, 246, 246, 246, 246, 246, 128, 116, 116, 116, 115, 115, 127,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 0, 0, 7, 7,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0,
    57, 57, 57, 57, 57, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    50, 50, 50, 50, 50, 50, 50, 7, 7, 9, 9, 9, 27, 27, 27, 27,
    46, 46, 46, 46, 7, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 125, 125, 125, 125, 125,
    125, 125, 0, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 0, 0, 0, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    46, 46, 46, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 157, 42, 42, 42, 158, 247, 248, 248, 46, 46, 9, 7, 7,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 0, 0, 0, 0,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    125, 125, 125, 125, 125, 125, 125, 9, 7, 9, 9, 0, 0, 0, 0, 0,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 0, 0, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 0, 0, 115,
    42, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116,
    116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116,
    116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116,
    116, 116, 116, 116, 116, 116, 116, 116, 0, 0, 0, 0, 0, 0, 0, 115,
    115, 115, 115, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    46, 46, 9, 46, 61, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    249, 249, 139, 139, 212, 212, 212, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    215, 215, 215, 215, 0, 215, 215, 215, 215, 215, 215, 215, 0, 215, 215, 0,
    223, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    223, 223, 223, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 135, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    135, 135, 135, 0, 0, 223, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 223, 223, 223, 223, 0, 0, 0, 0, 0, 0, 0, 0,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 27, 61, 250, 7,
    32, 32, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
    194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
    242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 27, 27, 27, 0, 0, 0,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 0, 0, 0, 0, 0, 0, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
    61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
    61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 0, 0,
    61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
    61, 61, 61, 61, 61, 61, 61, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 0, 0, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 251, 251,
    251, 251, 251, 251, 251, 252, 252, 57, 57, 57, 27, 27, 27, 253, 252, 252,
    252, 252, 252, 32, 32, 32, 32, 32, 32, 32, 32, 52, 52, 52, 52, 52,
    52, 52, 52, 27, 27, 50, 50, 50, 50, 50, 52, 52, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 50, 50, 50, 50, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 251, 251, 251, 251, 251,
    251, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    27, 27, 50, 50, 50, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    125, 125, 125, 125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    125, 125, 125, 125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    125, 125, 125, 125, 125, 125, 125, 125, 125, 0, 0, 0, 0, 0, 0, 0,
    41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
    41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 34, 34, 34, 34, 34, 34,
    34, 34, 197, 197, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34,
    34, 34, 34, 34, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
    41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 34, 34,
    34, 34, 34, 34, 34, 0, 197, 197, 34, 34, 34, 34, 34, 34, 34, 34,
    34, 34, 34, 34, 34, 34, 34, 34, 41, 41, 41, 41, 41, 41, 41, 41,
    41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
    41, 41, 34, 34, 34, 34, 34, 34, 34, 34, 197, 197, 34, 34, 34, 34,
    34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 41, 0, 41, 41,
    0, 0, 41, 0, 0, 41, 41, 0, 0, 41, 41, 41, 41, 0, 41, 41,
    41, 41, 41, 41, 41, 41, 34, 34, 34, 34, 0, 34, 0, 34, 197, 197,
    34, 34, 34, 34, 0, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34,
    41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
    41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 34, 34, 34, 34, 34, 34,
    34, 34, 197, 197, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34,
    34, 34, 34, 34, 41, 41, 0, 41, 41, 41, 41, 0, 0, 41, 41, 41,
    41, 41, 41, 41, 41, 0, 41, 41, 41, 41, 41, 41, 41, 0, 34, 34,
    34, 34, 34, 34, 34, 34, 197, 197, 34, 34, 34, 34, 34, 34, 34, 34,
    34, 34, 34, 34, 34, 34, 34, 34, 41, 41, 0, 41, 41, 41, 41, 0,
    41, 41, 41, 41, 41, 0, 41, 0, 0, 0, 41, 41, 41, 41, 41, 41,
    41, 0, 34, 34, 34, 34, 34, 34, 34, 34, 197, 197, 34, 34, 34, 34,
    34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 41, 41, 41, 41,
    41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
    41, 41, 41, 41, 41, 41, 34, 34, 34, 34, 34, 34, 34, 34, 197, 197,
    34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34,
    41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
    41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 34, 34, 34, 34, 34, 34,
    34, 34, 197, 197, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34,
    34, 34, 34, 34, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
    41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 34, 34,
    34, 34, 34, 34, 34, 34, 197, 197, 34, 34, 34, 34, 34, 34, 34, 34,
    34, 34, 34, 34, 34, 34, 34, 34, 41, 41, 41, 41, 41, 41, 41, 41,
    41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
    41, 41, 34, 34, 34, 34, 34, 34, 34, 34, 197, 197, 34, 34, 34, 34,
    34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 41, 41, 41, 41,
    41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
    41, 41, 41, 41, 41, 41, 34, 34, 34, 34, 34, 34, 34, 34, 197, 197,
    34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34,
    41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
    41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 34, 34, 34, 34, 34, 34,
    34, 34, 197, 197, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34,
    34, 34, 34, 34, 34, 34, 0, 0, 41, 41, 41, 41, 41, 41, 41, 41,
    41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
    41, 190, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34,
    34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 190, 34, 34, 34, 34,
    34, 34, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
    41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 190, 34, 34, 34, 34,
    34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34,
    34, 34, 34, 34, 34, 190, 34, 34, 34, 34, 34, 34, 41, 41, 41, 41,
    41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
    41, 41, 41, 41, 41, 190, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34,
    34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 190,
    34, 34, 34, 34, 34, 34, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
    41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 190,
    34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34,
    34, 34, 34, 34, 34, 34, 34, 34, 34, 190, 34, 34, 34, 34, 34, 34,
    41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
    41, 41, 41, 41, 41, 41, 41, 41, 41, 190, 34, 34, 34, 34, 34, 34,
    34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34,
    34, 34, 34, 190, 34, 34, 34, 34, 34, 34, 41, 34, 0, 0, 242, 242,
    242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242,
    242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242,
    242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242,
    61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
    61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
    61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
    61, 61, 61, 61, 61, 61, 61, 27, 27, 27, 27, 61, 61, 61, 61, 61,
    61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
    61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
    61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 27, 27, 27,
    27, 27, 27, 27, 27, 61, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 61, 27, 27, 9, 7, 9, 9, 9, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 61, 61, 61, 61, 61,
    0, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 42, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 24, 23, 23, 23, 23, 0,
    0, 0, 0, 0, 0, 23, 23, 23, 23, 23, 23, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    64, 64, 64, 64, 64, 64, 64, 0, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 0, 0, 64, 64, 64, 64, 64,
    64, 64, 0, 64, 64, 0, 64, 64, 64, 64, 64, 0, 0, 0, 0, 0,
    44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44,
    44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 45, 45, 44, 44,
    44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44,
    44, 44, 44, 44, 44, 44, 44, 44, 45, 44, 44, 44, 44, 44, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 46, 46, 46, 46, 46, 46, 46, 0, 0,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 0, 0, 42, 27,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 50, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 50, 50, 50, 50,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 0, 0, 0, 10,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 46, 51, 51, 52, 50,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 50, 52,
    42, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 0, 0, 9,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0,
    42, 42, 42, 64, 42, 42, 64, 42, 42, 42, 42, 42, 42, 42, 64, 64,
    42, 42, 42, 42, 42, 64, 0, 0, 0, 0, 0, 0, 0, 0, 42, 46,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    42, 42, 42, 42, 42, 42, 42, 0, 42, 42, 42, 42, 0, 42, 42, 0,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0,
    42, 42, 42, 42, 42, 0, 0, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    52, 52, 52, 52, 52, 52, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 50, 50, 50, 64, 50, 50, 122, 46, 0, 0, 0, 0,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 0, 0, 9, 9,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 27, 125, 125, 125,
    10, 125, 125, 125, 125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
    125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 27, 125,
    125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 0, 0,
    112, 112, 112, 112, 0, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    0, 112, 112, 0, 112, 0, 0, 112, 0, 112, 112, 112, 112, 112, 112, 112,
    112, 112, 112, 0, 112, 112, 112, 112, 0, 112, 0, 112, 0, 0, 0, 0,
    0, 0, 112, 0, 0, 0, 0, 112, 0, 112, 0, 112, 0, 112, 112, 112,
    0, 112, 112, 0, 112, 0, 0, 112, 0, 112, 0, 112, 0, 112, 0, 112,
    0, 112, 112, 0, 112, 0, 0, 112, 112, 112, 112, 0, 112, 112, 112, 112,
    112, 112, 112, 0, 112, 112, 112, 112, 0, 112, 112, 112, 112, 0, 112, 0,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 0, 112, 112, 112, 112, 112,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 0, 0, 0, 0,
    0, 112, 112, 112, 0, 112, 112, 112, 112, 112, 0, 112, 112, 112, 112, 112,
    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    14, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    27, 27, 27, 27, 29, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 254, 254, 254, 254,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 254,
    254, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    254, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 29,
    254, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 125, 125, 27, 27, 27,
    194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
    194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 27,
    205, 205, 205, 205, 205, 205, 205, 205, 205, 205, 205, 205, 205, 205, 205, 205,
    205, 205, 205, 205, 205, 205, 205, 205, 205, 205, 194, 194, 194, 194, 194, 194,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 194, 194, 194, 27, 27, 27,
    256, 256, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 256, 256,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 27, 27, 27, 27, 29, 27,
    194, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 254, 254,
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    254, 254, 254, 254, 254, 254, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257,
    257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257,
    194, 195, 195, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 195, 194, 194, 194, 194, 194,
    194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 195,
    194, 194, 195, 195, 195, 195, 195, 195, 195, 195, 195, 194, 254, 254, 254, 254,
    194, 194, 194, 194, 194, 194, 194, 194, 194, 254, 254, 254, 254, 254, 254, 254,
    195, 195, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    27, 27, 27, 27, 27, 27, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 29, 27, 27, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 27, 27, 29, 29, 27, 29, 29, 29, 27, 27, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 27, 27, 29, 29, 29, 27, 29, 29, 29, 29, 258, 258, 258, 258, 258,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 27, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 29, 29, 29, 29, 29, 29, 27,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 27, 27, 27, 27, 27, 27, 27, 29,
    29, 27, 27, 29, 29, 29, 29, 29, 29, 29, 29, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 29, 27, 27, 29, 29, 29, 29, 27, 27,
    29, 27, 27, 27, 27, 29, 29, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 29, 29, 27, 27, 29, 27, 27, 27, 27, 27, 27, 27,
    27, 29, 29, 27, 27, 27, 27, 27, 27, 27, 27, 27, 29, 27, 27, 27,
    27, 27, 29, 29, 29, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 29, 29, 29, 27, 27, 27, 27, 27, 27, 27, 27, 29, 29, 29, 27,
    27, 29, 27, 29, 27, 27, 27, 27, 29, 27, 27, 27, 27, 27, 27, 29,
    27, 27, 27, 29, 27, 27, 27, 27, 27, 27, 29, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 208, 208, 208, 27, 27, 27, 27, 27, 27, 27,
    29, 29, 29, 29, 29, 29, 27, 27, 27, 27, 27, 29, 29, 29, 29, 29,
    29, 29, 29, 27, 27, 29, 29, 29, 29, 254, 254, 254, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 27, 27, 27, 29, 27, 29, 29, 254, 254, 254,
    29, 27, 27, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 254, 254, 254,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 254, 254, 254, 254, 254, 254,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 254, 254, 254, 254,
    29, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 254, 254, 254, 254,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 254, 254, 254, 254, 254, 254, 254, 254,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 254, 254, 254, 254, 254, 254,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 254, 254, 254, 254, 254, 254, 254, 254,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 254, 254,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 254, 254, 254, 254,
    27, 27, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 254, 254, 254, 254, 254, 254, 254,
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 27, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 27, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 254, 254, 254, 254, 254, 254, 254, 254,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 254, 254,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 254, 254, 254,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 254, 254, 254, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 254, 29, 254, 254, 254, 254, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 254, 254, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 254, 254, 254, 254, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 254, 254, 254, 254, 254, 254, 254,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 0, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 27, 0, 0, 0, 0, 0,
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 0, 0,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 0, 0,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 0, 0,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232,
    232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 0, 0, 0, 0, 0,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
    135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 0, 0, 0, 0, 0, 0,
    189, 32, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189,
    189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189,
    259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259,
    259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259,
    259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259,
    259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259,
    259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259,
    259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259, 259,
    189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189,
    189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189,
    189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189,
    189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189,
    61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
    61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
    61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
    61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
    61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
    61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
    61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
    189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189,
    231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231,
    231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231,
    231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231,
    231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 0, 0
};

typedef struct {
    uint32_t cp;
    int32_t  lc;
    int32_t  uc;
    int32_t  tc;
    int32_t  cf;
    int32_t  scf;
} cs_ucase_entry;

#define CS_UCASE_COUNT 3037U
static const cs_ucase_entry cs_ucase_table[3037] = {
    {0x0041U, 97, 65, 65, 97, 97},
    {0x0042U, 98, 66, 66, 98, 98},
    {0x0043U, 99, 67, 67, 99, 99},
    {0x0044U, 100, 68, 68, 100, 100},
    {0x0045U, 101, 69, 69, 101, 101},
    {0x0046U, 102, 70, 70, 102, 102},
    {0x0047U, 103, 71, 71, 103, 103},
    {0x0048U, 104, 72, 72, 104, 104},
    {0x0049U, 105, 73, 73, 105, 105},
    {0x004AU, 106, 74, 74, 106, 106},
    {0x004BU, 107, 75, 75, 107, 107},
    {0x004CU, 108, 76, 76, 108, 108},
    {0x004DU, 109, 77, 77, 109, 109},
    {0x004EU, 110, 78, 78, 110, 110},
    {0x004FU, 111, 79, 79, 111, 111},
    {0x0050U, 112, 80, 80, 112, 112},
    {0x0051U, 113, 81, 81, 113, 113},
    {0x0052U, 114, 82, 82, 114, 114},
    {0x0053U, 115, 83, 83, 115, 115},
    {0x0054U, 116, 84, 84, 116, 116},
    {0x0055U, 117, 85, 85, 117, 117},
    {0x0056U, 118, 86, 86, 118, 118},
    {0x0057U, 119, 87, 87, 119, 119},
    {0x0058U, 120, 88, 88, 120, 120},
    {0x0059U, 121, 89, 89, 121, 121},
    {0x005AU, 122, 90, 90, 122, 122},
    {0x0061U, 97, 65, 65, 97, 97},
    {0x0062U, 98, 66, 66, 98, 98},
    {0x0063U, 99, 67, 67, 99, 99},
    {0x0064U, 100, 68, 68, 100, 100},
    {0x0065U, 101, 69, 69, 101, 101},
    {0x0066U, 102, 70, 70, 102, 102},
    {0x0067U, 103, 71, 71, 103, 103},
    {0x0068U, 104, 72, 72, 104, 104},
    {0x0069U, 105, 73, 73, 105, 105},
    {0x006AU, 106, 74, 74, 106, 106},
    {0x006BU, 107, 75, 75, 107, 107},
    {0x006CU, 108, 76, 76, 108, 108},
    {0x006DU, 109, 77, 77, 109, 109},
    {0x006EU, 110, 78, 78, 110, 110},
    {0x006FU, 111, 79, 79, 111, 111},
    {0x0070U, 112, 80, 80, 112, 112},
    {0x0071U, 113, 81, 81, 113, 113},
    {0x0072U, 114, 82, 82, 114, 114},
    {0x0073U, 115, 83, 83, 115, 115},
    {0x0074U, 116, 84, 84, 116, 116},
    {0x0075U, 117, 85, 85, 117, 117},
    {0x0076U, 118, 86, 86, 118, 118},
    {0x0077U, 119, 87, 87, 119, 119},
    {0x0078U, 120, 88, 88, 120, 120},
    {0x0079U, 121, 89, 89, 121, 121},
    {0x007AU, 122, 90, 90, 122, 122},
    {0x00B5U, 181, 924, 924, 956, 956},
    {0x00C0U, 224, 192, 192, 224, 224},
    {0x00C1U, 225, 193, 193, 225, 225},
    {0x00C2U, 226, 194, 194, 226, 226},
    {0x00C3U, 227, 195, 195, 227, 227},
    {0x00C4U, 228, 196, 196, 228, 228},
    {0x00C5U, 229, 197, 197, 229, 229},
    {0x00C6U, 230, 198, 198, 230, 230},
    {0x00C7U, 231, 199, 199, 231, 231},
    {0x00C8U, 232, 200, 200, 232, 232},
    {0x00C9U, 233, 201, 201, 233, 233},
    {0x00CAU, 234, 202, 202, 234, 234},
    {0x00CBU, 235, 203, 203, 235, 235},
    {0x00CCU, 236, 204, 204, 236, 236},
    {0x00CDU, 237, 205, 205, 237, 237},
    {0x00CEU, 238, 206, 206, 238, 238},
    {0x00CFU, 239, 207, 207, 239, 239},
    {0x00D0U, 240, 208, 208, 240, 240},
    {0x00D1U, 241, 209, 209, 241, 241},
    {0x00D2U, 242, 210, 210, 242, 242},
    {0x00D3U, 243, 211, 211, 243, 243},
    {0x00D4U, 244, 212, 212, 244, 244},
    {0x00D5U, 245, 213, 213, 245, 245},
    {0x00D6U, 246, 214, 214, 246, 246},
    {0x00D8U, 248, 216, 216, 248, 248},
    {0x00D9U, 249, 217, 217, 249, 249},
    {0x00DAU, 250, 218, 218, 250, 250},
    {0x00DBU, 251, 219, 219, 251, 251},
    {0x00DCU, 252, 220, 220, 252, 252},
    {0x00DDU, 253, 221, 221, 253, 253},
    {0x00DEU, 254, 222, 222, 254, 254},
    {0x00DFU, 223, -2, -18, -34, 223},
    {0x00E0U, 224, 192, 192, 224, 224},
    {0x00E1U, 225, 193, 193, 225, 225},
    {0x00E2U, 226, 194, 194, 226, 226},
    {0x00E3U, 227, 195, 195, 227, 227},
    {0x00E4U, 228, 196, 196, 228, 228},
    {0x00E5U, 229, 197, 197, 229, 229},
    {0x00E6U, 230, 198, 198, 230, 230},
    {0x00E7U, 231, 199, 199, 231, 231},
    {0x00E8U, 232, 200, 200, 232, 232},
    {0x00E9U, 233, 201, 201, 233, 233},
    {0x00EAU, 234, 202, 202, 234, 234},
    {0x00EBU, 235, 203, 203, 235, 235},
    {0x00ECU, 236, 204, 204, 236, 236},
    {0x00EDU, 237, 205, 205, 237, 237},
    {0x00EEU, 238, 206, 206, 238, 238},
    {0x00EFU, 239, 207, 207, 239, 239},
    {0x00F0U, 240, 208, 208, 240, 240},
    {0x00F1U, 241, 209, 209, 241, 241},
    {0x00F2U, 242, 210, 210, 242, 242},
    {0x00F3U, 243, 211, 211, 243, 243},
    {0x00F4U, 244, 212, 212, 244, 244},
    {0x00F5U, 245, 213, 213, 245, 245},
    {0x00F6U, 246, 214, 214, 246, 246},
    {0x00F8U, 248, 216, 216, 248, 248},
    {0x00F9U, 249, 217, 217, 249, 249},
    {0x00FAU, 250, 218, 218, 250, 250},
    {0x00FBU, 251, 219, 219, 251, 251},
    {0x00FCU, 252, 220, 220, 252, 252},
    {0x00FDU, 253, 221, 221, 253, 253},
    {0x00FEU, 254, 222, 222, 254, 254},
    {0x00FFU, 255, 376, 376, 255, 255},
    {0x0100U, 257, 256, 256, 257, 257},
    {0x0101U, 257, 256, 256, 257, 257},
    {0x0102U, 259, 258, 258, 259, 259},
    {0x0103U, 259, 258, 258, 259, 259},
    {0x0104U, 261, 260, 260, 261, 261},
    {0x0105U, 261, 260, 260, 261, 261},
    {0x0106U, 263, 262, 262, 263, 263},
    {0x0107U, 263, 262, 262, 263, 263},
    {0x0108U, 265, 264, 264, 265, 265},
    {0x0109U, 265, 264, 264, 265, 265},
    {0x010AU, 267, 266, 266, 267, 267},
    {0x010BU, 267, 266, 266, 267, 267},
    {0x010CU, 269, 268, 268, 269, 269},
    {0x010DU, 269, 268, 268, 269, 269},
    {0x010EU, 271, 270, 270, 271, 271},
    {0x010FU, 271, 270, 270, 271, 271},
    {0x0110U, 273, 272, 272, 273, 273},
    {0x0111U, 273, 272, 272, 273, 273},
    {0x0112U, 275, 274, 274, 275, 275},
    {0x0113U, 275, 274, 274, 275, 275},
    {0x0114U, 277, 276, 276, 277, 277},
    {0x0115U, 277, 276, 276, 277, 277},
    {0x0116U, 279, 278, 278, 279, 279},
    {0x0117U, 279, 278, 278, 279, 279},
    {0x0118U, 281, 280, 280, 281, 281},
    {0x0119U, 281, 280, 280, 281, 281},
    {0x011AU, 283, 282, 282, 283, 283},
    {0x011BU, 283, 282, 282, 283, 283},
    {0x011CU, 285, 284, 284, 285, 285},
    {0x011DU, 285, 284, 284, 285, 285},
    {0x011EU, 287, 286, 286, 287, 287},
    {0x011FU, 287, 286, 286, 287, 287},
    {0x0120U, 289, 288, 288, 289, 289},
    {0x0121U, 289, 288, 288, 289, 289},
    {0x0122U, 291, 290, 290, 291, 291},
    {0x0123U, 291, 290, 290, 291, 291},
    {0x0124U, 293, 292, 292, 293, 293},
    {0x0125U, 293, 292, 292, 293, 293},
    {0x0126U, 295, 294, 294, 295, 295},
    {0x0127U, 295, 294, 294, 295, 295},
    {0x0128U, 297, 296, 296, 297, 297},
    {0x0129U, 297, 296, 296, 297, 297},
    {0x012AU, 299, 298, 298, 299, 299},
    {0x012BU, 299, 298, 298, 299, 299},
    {0x012CU, 301, 300, 300, 301, 301},
    {0x012DU, 301, 300, 300, 301, 301},
    {0x012EU, 303, 302, 302, 303, 303},
    {0x012FU, 303, 302, 302, 303, 303},
    {0x0130U, -50, 304, 304, -50, 304},
    {0x0131U, 305, 73, 73, 305, 305},
    {0x0132U, 307, 306, 306, 307, 307},
    {0x0133U, 307, 306, 306, 307, 307},
    {0x0134U, 309, 308, 308, 309, 309},
    {0x0135U, 309, 308, 308, 309, 309},
    {0x0136U, 311, 310, 310, 311, 311},
    {0x0137U, 311, 310, 310, 311, 311},
    {0x0139U, 314, 313, 313, 314, 314},
    {0x013AU, 314, 313, 313, 314, 314},
    {0x013BU, 316, 315, 315, 316, 316},
    {0x013CU, 316, 315, 315, 316, 316},
    {0x013DU, 318, 317, 317, 318, 318},
    {0x013EU, 318, 317, 317, 318, 318},
    {0x013FU, 320, 319, 319, 320, 320},
    {0x0140U, 320, 319, 319, 320, 320},
    {0x0141U, 322, 321, 321, 322, 322},
    {0x0142U, 322, 321, 321, 322, 322},
    {0x0143U, 324, 323, 323, 324, 324},
    {0x0144U, 324, 323, 323, 324, 324},
    {0x0145U, 326, 325, 325, 326, 326},
    {0x0146U, 326, 325, 325, 326, 326},
    {0x0147U, 328, 327, 327, 328, 328},
    {0x0148U, 328, 327, 327, 328, 328},
    {0x0149U, 329, -66, -66, -82, 329},
    {0x014AU, 331, 330, 330, 331, 331},
    {0x014BU, 331, 330, 330, 331, 331},
    {0x014CU, 333, 332, 332, 333, 333},
    {0x014DU, 333, 332, 332, 333, 333},
    {0x014EU, 335, 334, 334, 335, 335},
    {0x014FU, 335, 334, 334, 335, 335},
    {0x0150U, 337, 336, 336, 337, 337},
    {0x0151U, 337, 336, 336, 337, 337},
    {0x0152U, 339, 338, 338, 339, 339},
    {0x0153U, 339, 338, 338, 339, 339},
    {0x0154U, 341, 340, 340, 341, 341},
    {0x0155U, 341, 340, 340, 341, 341},
    {0x0156U, 343, 342, 342, 343, 343},
    {0x0157U, 343, 342, 342, 343, 343},
    {0x0158U, 345, 344, 344, 345, 345},
    {0x0159U, 345, 344, 344, 345, 345},
    {0x015AU, 347, 346, 346, 347, 347},
    {0x015BU, 347, 346, 346, 347, 347},
    {0x015CU, 349, 348, 348, 349, 349},
    {0x015DU, 349, 348, 348, 349, 349},
    {0x015EU, 351, 350, 350, 351, 351},
    {0x015FU, 351, 350, 350, 351, 351},
    {0x0160U, 353, 352, 352, 353, 353},
    {0x0161U, 353, 352, 352, 353, 353},
    {0x0162U, 355, 354, 354, 355, 355},
    {0x0163U, 355, 354, 354, 355, 355},
    {0x0164U, 357, 356, 356, 357, 357},
    {0x0165U, 357, 356, 356, 357, 357},
    {0x0166U, 359, 358, 358, 359, 359},
    {0x0167U, 359, 358, 358, 359, 359},
    {0x0168U, 361, 360, 360, 361, 361},
    {0x0169U, 361, 360, 360, 361, 361},
    {0x016AU, 363, 362, 362, 363, 363},
    {0x016BU, 363, 362, 362, 363, 363},
    {0x016CU, 365, 364, 364, 365, 365},
    {0x016DU, 365, 364, 364, 365, 365},
    {0x016EU, 367, 366, 366, 367, 367},
    {0x016FU, 367, 366, 366, 367, 367},
    {0x0170U, 369, 368, 368, 369, 369},
    {0x0171U, 369, 368, 368, 369, 369},
    {0x0172U, 371, 370, 370, 371, 371},
    {0x0173U, 371, 370, 370, 371, 371},
    {0x0174U, 373, 372, 372, 373, 373},
    {0x0175U, 373, 372, 372, 373, 373},
    {0x0176U, 375, 374, 374, 375, 375},
    {0x0177U, 375, 374, 374, 375, 375},
    {0x0178U, 255, 376, 376, 255, 255},
    {0x0179U, 378, 377, 377, 378, 378},
    {0x017AU, 378, 377, 377, 378, 378},
    {0x017BU, 380, 379, 379, 380, 380},
    {0x017CU, 380, 379, 379, 380, 380},
    {0x017DU, 382, 381, 381, 382, 382},
    {0x017EU, 382, 381, 381, 382, 382},
    {0x017FU, 383, 83, 83, 115, 115},
    {0x0180U, 384, 579, 579, 384, 384},
    {0x0181U, 595, 385, 385, 595, 595},
    {0x0182U, 387, 386, 386, 387, 387},
    {0x0183U, 387, 386, 386, 387, 387},
    {0x0184U, 389, 388, 388, 389, 389},
    {0x0185U, 389, 388, 388, 389, 389},
    {0x0186U, 596, 390, 390, 596, 596},
    {0x0187U, 392, 391, 391, 392, 392},
    {0x0188U, 392, 391, 391, 392, 392},
    {0x0189U, 598, 393, 393, 598, 598},
    {0x018AU, 599, 394, 394, 599, 599},
    {0x018BU, 396, 395, 395, 396, 396},
    {0x018CU, 396, 395, 395, 396, 396},
    {0x018EU, 477, 398, 398, 477, 477},
    {0x018FU, 601, 399, 399, 601, 601},
    {0x0190U, 603, 400, 400, 603, 603},
    {0x0191U, 402, 401, 401, 402, 402},
    {0x0192U, 402, 401, 401, 402, 402},
    {0x0193U, 608, 403, 403, 608, 608},
    {0x0194U, 611, 404, 404, 611, 611},
    {0x0195U, 405, 502, 502, 405, 405},
    {0x0196U, 617, 406, 406, 617, 617},
    {0x0197U, 616, 407, 407, 616, 616},
    {0x0198U, 409, 408, 408, 409, 409},
    {0x0199U, 409, 408, 408, 409, 409},
    {0x019AU, 410, 573, 573, 410, 410},
    {0x019BU, 411, 42972, 42972, 411, 411},
    {0x019CU, 623, 412, 412, 623, 623},
    {0x019DU, 626, 413, 413, 626, 626},
    {0x019EU, 414, 544, 544, 414, 414},
    {0x019FU, 629, 415, 415, 629, 629},
    {0x01A0U, 417, 416, 416, 417, 417},
    {0x01A1U, 417, 416, 416, 417, 417},
    {0x01A2U, 419, 418, 418, 419, 419},
    {0x01A3U, 419, 418, 418, 419, 419},
    {0x01A4U, 421, 420, 420, 421, 421},
    {0x01A5U, 421, 420, 420, 421, 421},
    {0x01A6U, 640, 422, 422, 640, 640},
    {0x01A7U, 424, 423, 423, 424, 424},
    {0x01A8U, 424, 423, 423, 424, 424},
    {0x01A9U, 643, 425, 425, 643, 643},
    {0x01ACU, 429, 428, 428, 429, 429},
    {0x01ADU, 429, 428, 428, 429, 429},
    {0x01AEU, 648, 430, 430, 648, 648},
    {0x01AFU, 432, 431, 431, 432, 432},
    {0x01B0U, 432, 431, 431, 432, 432},
    {0x01B1U, 650, 433, 433, 650, 650},
    {0x01B2U, 651, 434, 434, 651, 651},
    {0x01B3U, 436, 435, 435, 436, 436},
    {0x01B4U, 436, 435, 435, 436, 436},
    {0x01B5U, 438, 437, 437, 438, 438},
    {0x01B6U, 438, 437, 437, 438, 438},
    {0x01B7U, 658, 439, 439, 658, 658},
    {0x01B8U, 441, 440, 440, 441, 441},
    {0x01B9U, 441, 440, 440, 441, 441},
    {0x01BCU, 445, 444, 444, 445, 445},
    {0x01BDU, 445, 444, 444, 445, 445},
    {0x01BFU, 447, 503, 503, 447, 447},
    {0x01C4U, 454, 452, 453, 454, 454},
    {0x01C5U, 454, 452, 453, 454, 454},
    {0x01C6U, 454, 452, 453, 454, 454},
    {0x01C7U, 457, 455, 456, 457, 457},
    {0x01C8U, 457, 455, 456, 457, 457},
    {0x01C9U, 457, 455, 456, 457, 457},
    {0x01CAU, 460, 458, 459, 460, 460},
    {0x01CBU, 460, 458, 459, 460, 460},
    {0x01CCU, 460, 458, 459, 460, 460},
    {0x01CDU, 462, 461, 461, 462, 462},
    {0x01CEU, 462, 461, 461, 462, 462},
    {0x01CFU, 464, 463, 463, 464, 464},
    {0x01D0U, 464, 463, 463, 464, 464},
    {0x01D1U, 466, 465, 465, 466, 466},
    {0x01D2U, 466, 465, 465, 466, 466},
    {0x01D3U, 468, 467, 467, 468, 468},
    {0x01D4U, 468, 467, 467, 468, 468},
    {0x01D5U, 470, 469, 469, 470, 470},
    {0x01D6U, 470, 469, 469, 470, 470},
    {0x01D7U, 472, 471, 471, 472, 472},
    {0x01D8U, 472, 471, 471, 472, 472},
    {0x01D9U, 474, 473, 473, 474, 474},
    {0x01DAU, 474, 473, 473, 474, 474},
    {0x01DBU, 476, 475, 475, 476, 476},
    {0x01DCU, 476, 475, 475, 476, 476},
    {0x01DDU, 477, 398, 398, 477, 477},
    {0x01DEU, 479, 478, 478, 479, 479},
    {0x01DFU, 479, 478, 478, 479, 479},
    {0x01E0U, 481, 480, 480, 481, 481},
    {0x01E1U, 481, 480, 480, 481, 481},
    {0x01E2U, 483, 482, 482, 483, 483},
    {0x01E3U, 483, 482, 482, 483, 483},
    {0x01E4U, 485, 484, 484, 485, 485},
    {0x01E5U, 485, 484, 484, 485, 485},
    {0x01E6U, 487, 486, 486, 487, 487},
    {0x01E7U, 487, 486, 486, 487, 487},
    {0x01E8U, 489, 488, 488, 489, 489},
    {0x01E9U, 489, 488, 488, 489, 489},
    {0x01EAU, 491, 490, 490, 491, 491},
    {0x01EBU, 491, 490, 490, 491, 491},
    {0x01ECU, 493, 492, 492, 493, 493},
    {0x01EDU, 493, 492, 492, 493, 493},
    {0x01EEU, 495, 494, 494, 495, 495},
    {0x01EFU, 495, 494, 494, 495, 495},
    {0x01F0U, 496, -98, -98, -114, 496},
    {0x01F1U, 499, 497, 498, 499, 499},
    {0x01F2U, 499, 497, 498, 499, 499},
    {0x01F3U, 499, 497, 498, 499, 499},
    {0x01F4U, 501, 500, 500, 501, 501},
    {0x01F5U, 501, 500, 500, 501, 501},
    {0x01F6U, 405, 502, 502, 405, 405},
    {0x01F7U, 447, 503, 503, 447, 447},
    {0x01F8U, 505, 504, 504, 505, 505},
    {0x01F9U, 505, 504, 504, 505, 505},
    {0x01FAU, 507, 506, 506, 507, 507},
    {0x01FBU, 507, 506, 506, 507, 507},
    {0x01FCU, 509, 508, 508, 509, 509},
    {0x01FDU, 509, 508, 508, 509, 509},
    {0x01FEU, 511, 510, 510, 511, 511},
    {0x01FFU, 511, 510, 510, 511, 511},
    {0x0200U, 513, 512, 512, 513, 513},
    {0x0201U, 513, 512, 512, 513, 513},
    {0x0202U, 515, 514, 514, 515, 515},
    {0x0203U, 515, 514, 514, 515, 515},
    {0x0204U, 517, 516, 516, 517, 517},
    {0x0205U, 517, 516, 516, 517, 517},
    {0x0206U, 519, 518, 518, 519, 519},
    {0x0207U, 519, 518, 518, 519, 519},
    {0x0208U, 521, 520, 520, 521, 521},
    {0x0209U, 521, 520, 520, 521, 521},
    {0x020AU, 523, 522, 522, 523, 523},
    {0x020BU, 523, 522, 522, 523, 523},
    {0x020CU, 525, 524, 524, 525, 525},
    {0x020DU, 525, 524, 524, 525, 525},
    {0x020EU, 527, 526, 526, 527, 527},
    {0x020FU, 527, 526, 526, 527, 527},
    {0x0210U, 529, 528, 528, 529, 529},
    {0x0211U, 529, 528, 528, 529, 529},
    {0x0212U, 531, 530, 530, 531, 531},
    {0x0213U, 531, 530, 530, 531, 531},
    {0x0214U, 533, 532, 532, 533, 533},
    {0x0215U, 533, 532, 532, 533, 533},
    {0x0216U, 535, 534, 534, 535, 535},
    {0x0217U, 535, 534, 534, 535, 535},
    {0x0218U, 537, 536, 536, 537, 537},
    {0x0219U, 537, 536, 536, 537, 537},
    {0x021AU, 539, 538, 538, 539, 539},
    {0x021BU, 539, 538, 538, 539, 539},
    {0x021CU, 541, 540, 540, 541, 541},
    {0x021DU, 541, 540, 540, 541, 541},
    {0x021EU, 543, 542, 542, 543, 543},
    {0x021FU, 543, 542, 542, 543, 543},
    {0x0220U, 414, 544, 544, 414, 414},
    {0x0222U, 547, 546, 546, 547, 547},
    {0x0223U, 547, 546, 546, 547, 547},
    {0x0224U, 549, 548, 548, 549, 549},
    {0x0225U, 549, 548, 548, 549, 549},
    {0x0226U, 551, 550, 550, 551, 551},
    {0x0227U, 551, 550, 550, 551, 551},
    {0x0228U, 553, 552, 552, 553, 553},
    {0x0229U, 553, 552, 552, 553, 553},
    {0x022AU, 555, 554, 554, 555, 555},
    {0x022BU, 555, 554, 554, 555, 555},
    {0x022CU, 557, 556, 556, 557, 557},
    {0x022DU, 557, 556, 556, 557, 557},
    {0x022EU, 559, 558, 558, 559, 559},
    {0x022FU, 559, 558, 558, 559, 559},
    {0x0230U, 561, 560, 560, 561, 561},
    {0x0231U, 561, 560, 560, 561, 561},
    {0x0232U, 563, 562, 562, 563, 563},
    {0x0233U, 563, 562, 562, 563, 563},
    {0x023AU, 11365, 570, 570, 11365, 11365},
    {0x023BU, 572, 571, 571, 572, 572},
    {0x023CU, 572, 571, 571, 572, 572},
    {0x023DU, 410, 573, 573, 410, 410},
    {0x023EU, 11366, 574, 574, 11366, 11366},
    {0x023FU, 575, 11390, 11390, 575, 575},
    {0x0240U, 576, 11391, 11391, 576, 576},
    {0x0241U, 578, 577, 577, 578, 578},
    {0x0242U, 578, 577, 577, 578, 578},
    {0x0243U, 384, 579, 579, 384, 384},
    {0x0244U, 649, 580, 580, 649, 649},
    {0x0245U, 652, 581, 581, 652, 652},
    {0x0246U, 583, 582, 582, 583, 583},
    {0x0247U, 583, 582, 582, 583, 583},
    {0x0248U, 585, 584, 584, 585, 585},
    {0x0249U, 585, 584, 584, 585, 585},
    {0x024AU, 587, 586, 586, 587, 587},
    {0x024BU, 587, 586, 586, 587, 587},
    {0x024CU, 589, 588, 588, 589, 589},
    {0x024DU, 589, 588, 588, 589, 589},
    {0x024EU, 591, 590, 590, 591, 591},
    {0x024FU, 591, 590, 590, 591, 591},
    {0x0250U, 592, 11375, 11375, 592, 592},
    {0x0251U, 593, 11373, 11373, 593, 593},
    {0x0252U, 594, 11376, 11376, 594, 594},
    {0x0253U, 595, 385, 385, 595, 595},
    {0x0254U, 596, 390, 390, 596, 596},
    {0x0256U, 598, 393, 393, 598, 598},
    {0x0257U, 599, 394, 394, 599, 599},
    {0x0259U, 601, 399, 399, 601, 601},
    {0x025BU, 603, 400, 400, 603, 603},
    {0x025CU, 604, 42923, 42923, 604, 604},
    {0x0260U, 608, 403, 403, 608, 608},
    {0x0261U, 609, 42924, 42924, 609, 609},
    {0x0263U, 611, 404, 404, 611, 611},
    {0x0264U, 612, 42955, 42955, 612, 612},
    {0x0265U, 613, 42893, 42893, 613, 613},
    {0x0266U, 614, 42922, 42922, 614, 614},
    {0x0268U, 616, 407, 407, 616, 616},
    {0x0269U, 617, 406, 406, 617, 617},
    {0x026AU, 618, 42926, 42926, 618, 618},
    {0x026BU, 619, 11362, 11362, 619, 619},
    {0x026CU, 620, 42925, 42925, 620, 620},
    {0x026FU, 623, 412, 412, 623, 623},
    {0x0271U, 625, 11374, 11374, 625, 625},
    {0x0272U, 626, 413, 413, 626, 626},
    {0x0275U, 629, 415, 415, 629, 629},
    {0x027DU, 637, 11364, 11364, 637, 637},
    {0x0280U, 640, 422, 422, 640, 640},
    {0x0282U, 642, 42949, 42949, 642, 642},
    {0x0283U, 643, 425, 425, 643, 643},
    {0x0287U, 647, 42929, 42929, 647, 647},
    {0x0288U, 648, 430, 430, 648, 648},
    {0x0289U, 649, 580, 580, 649, 649},
    {0x028AU, 650, 433, 433, 650, 650},
    {0x028BU, 651, 434, 434, 651, 651},
    {0x028CU, 652, 581, 581, 652, 652},
    {0x0292U, 658, 439, 439, 658, 658},
    {0x029DU, 669, 42930, 42930, 669, 669},
    {0x029EU, 670, 42928, 42928, 670, 670},
    {0x0345U, 837, 921, 921, 953, 953},
    {0x0370U, 881, 880, 880, 881, 881},
    {0x0371U, 881, 880, 880, 881, 881},
    {0x0372U, 883, 882, 882, 883, 883},
    {0x0373U, 883, 882, 882, 883, 883},
    {0x0376U, 887, 886, 886, 887, 887},
    {0x0377U, 887, 886, 886, 887, 887},
    {0x037BU, 891, 1021, 1021, 891, 891},
    {0x037CU, 892, 1022, 1022, 892, 892},
    {0x037DU, 893, 1023, 1023, 893, 893},
    {0x037FU, 1011, 895, 895, 1011, 1011},
    {0x0386U, 940, 902, 902, 940, 940},
    {0x0388U, 941, 904, 904, 941, 941},
    {0x0389U, 942, 905, 905, 942, 942},
    {0x038AU, 943, 906, 906, 943, 943},
    {0x038CU, 972, 908, 908, 972, 972},
    {0x038EU, 973, 910, 910, 973, 973},
    {0x038FU, 974, 911, 911, 974, 974},
    {0x0390U, 912, -131, -131, -155, 912},
    {0x0391U, 945, 913, 913, 945, 945},
    {0x0392U, 946, 914, 914, 946, 946},
    {0x0393U, 947, 915, 915, 947, 947},
    {0x0394U, 948, 916, 916, 948, 948},
    {0x0395U, 949, 917, 917, 949, 949},
    {0x0396U, 950, 918, 918, 950, 950},
    {0x0397U, 951, 919, 919, 951, 951},
    {0x0398U, 952, 920, 920, 952, 952},
    {0x0399U, 953, 921, 921, 953, 953},
    {0x039AU, 954, 922, 922, 954, 954},
    {0x039BU, 955, 923, 923, 955, 955},
    {0x039CU, 956, 924, 924, 956, 956},
    {0x039DU, 957, 925, 925, 957, 957},
    {0x039EU, 958, 926, 926, 958, 958},
    {0x039FU, 959, 927, 927, 959, 959},
    {0x03A0U, 960, 928, 928, 960, 960},
    {0x03A1U, 961, 929, 929, 961, 961},
    {0x03A3U, 963, 931, 931, 963, 963},
    {0x03A4U, 964, 932, 932, 964, 964},
    {0x03A5U, 965, 933, 933, 965, 965},
    {0x03A6U, 966, 934, 934, 966, 966},
    {0x03A7U, 967, 935, 935, 967, 967},
    {0x03A8U, 968, 936, 936, 968, 968},
    {0x03A9U, 969, 937, 937, 969, 969},
    {0x03AAU, 970, 938, 938, 970, 970},
    {0x03ABU, 971, 939, 939, 971, 971},
    {0x03ACU, 940, 902, 902, 940, 940},
    {0x03ADU, 941, 904, 904, 941, 941},
    {0x03AEU, 942, 905, 905, 942, 942},
    {0x03AFU, 943, 906, 906, 943, 943},
    {0x03B0U, 944, -179, -179, -203, 944},
    {0x03B1U, 945, 913, 913, 945, 945},
    {0x03B2U, 946, 914, 914, 946, 946},
    {0x03B3U, 947, 915, 915, 947, 947},
    {0x03B4U, 948, 916, 916, 948, 948},
    {0x03B5U, 949, 917, 917, 949, 949},
    {0x03B6U, 950, 918, 918, 950, 950},
    {0x03B7U, 951, 919, 919, 951, 951},
    {0x03B8U, 952, 920, 920, 952, 952},
    {0x03B9U, 953, 921, 921, 953, 953},
    {0x03BAU, 954, 922, 922, 954, 954},
    {0x03BBU, 955, 923, 923, 955, 955},
    {0x03BCU, 956, 924, 924, 956, 956},
    {0x03BDU, 957, 925, 925, 957, 957},
    {0x03BEU, 958, 926, 926, 958, 958},
    {0x03BFU, 959, 927, 927, 959, 959},
    {0x03C0U, 960, 928, 928, 960, 960},
    {0x03C1U, 961, 929, 929, 961, 961},
    {0x03C2U, 962, 931, 931, 963, 963},
    {0x03C3U, 963, 931, 931, 963, 963},
    {0x03C4U, 964, 932, 932, 964, 964},
    {0x03C5U, 965, 933, 933, 965, 965},
    {0x03C6U, 966, 934, 934, 966, 966},
    {0x03C7U, 967, 935, 935, 967, 967},
    {0x03C8U, 968, 936, 936, 968, 968},
    {0x03C9U, 969, 937, 937, 969, 969},
    {0x03CAU, 970, 938, 938, 970, 970},
    {0x03CBU, 971, 939, 939, 971, 971},
    {0x03CCU, 972, 908, 908, 972, 972},
    {0x03CDU, 973, 910, 910, 973, 973},
    {0x03CEU, 974, 911, 911, 974, 974},
    {0x03CFU, 983, 975, 975, 983, 983},
    {0x03D0U, 976, 914, 914, 946, 946},
    {0x03D1U, 977, 920, 920, 952, 952},
    {0x03D5U, 981, 934, 934, 966, 966},
    {0x03D6U, 982, 928, 928, 960, 960},
    {0x03D7U, 983, 975, 975, 983, 983},
    {0x03D8U, 985, 984, 984, 985, 985},
    {0x03D9U, 985, 984, 984, 985, 985},
    {0x03DAU, 987, 986, 986, 987, 987},
    {0x03DBU, 987, 986, 986, 987, 987},
    {0x03DCU, 989, 988, 988, 989, 989},
    {0x03DDU, 989, 988, 988, 989, 989},
    {0x03DEU, 991, 990, 990, 991, 991},
    {0x03DFU, 991, 990, 990, 991, 991},
    {0x03E0U, 993, 992, 992, 993, 993},
    {0x03E1U, 993, 992, 992, 993, 993},
    {0x03E2U, 995, 994, 994, 995, 995},
    {0x03E3U, 995, 994, 994, 995, 995},
    {0x03E4U, 997, 996, 996, 997, 997},
    {0x03E5U, 997, 996, 996, 997, 997},
    {0x03E6U, 999, 998, 998, 999, 999},
    {0x03E7U, 999, 998, 998, 999, 999},
    {0x03E8U, 1001, 1000, 1000, 1001, 1001},
    {0x03E9U, 1001, 1000, 1000, 1001, 1001},
    {0x03EAU, 1003, 1002, 1002, 1003, 1003},
    {0x03EBU, 1003, 1002, 1002, 1003, 1003},
    {0x03ECU, 1005, 1004, 1004, 1005, 1005},
    {0x03EDU, 1005, 1004, 1004, 1005, 1005},
    {0x03EEU, 1007, 1006, 1006, 1007, 1007},
    {0x03EFU, 1007, 1006, 1006, 1007, 1007},
    {0x03F0U, 1008, 922, 922, 954, 954},
    {0x03F1U, 1009, 929, 929, 961, 961},
    {0x03F2U, 1010, 1017, 1017, 1010, 1010},
    {0x03F3U, 1011, 895, 895, 1011, 1011},
    {0x03F4U, 952, 1012, 1012, 952, 952},
    {0x03F5U, 1013, 917, 917, 949, 949},
    {0x03F7U, 1016, 1015, 1015, 1016, 1016},
    {0x03F8U, 1016, 1015, 1015, 1016, 1016},
    {0x03F9U, 1010, 1017, 1017, 1010, 1010},
    {0x03FAU, 1019, 1018, 1018, 1019, 1019},
    {0x03FBU, 1019, 1018, 1018, 1019, 1019},
    {0x03FDU, 891, 1021, 1021, 891, 891},
    {0x03FEU, 892, 1022, 1022, 892, 892},
    {0x03FFU, 893, 1023, 1023, 893, 893},
    {0x0400U, 1104, 1024, 1024, 1104, 1104},
    {0x0401U, 1105, 1025, 1025, 1105, 1105},
    {0x0402U, 1106, 1026, 1026, 1106, 1106},
    {0x0403U, 1107, 1027, 1027, 1107, 1107},
    {0x0404U, 1108, 1028, 1028, 1108, 1108},
    {0x0405U, 1109, 1029, 1029, 1109, 1109},
    {0x0406U, 1110, 1030, 1030, 1110, 1110},
    {0x0407U, 1111, 1031, 1031, 1111, 1111},
    {0x0408U, 1112, 1032, 1032, 1112, 1112},
    {0x0409U, 1113, 1033, 1033, 1113, 1113},
    {0x040AU, 1114, 1034, 1034, 1114, 1114},
    {0x040BU, 1115, 1035, 1035, 1115, 1115},
    {0x040CU, 1116, 1036, 1036, 1116, 1116},
    {0x040DU, 1117, 1037, 1037, 1117, 1117},
    {0x040EU, 1118, 1038, 1038, 1118, 1118},
    {0x040FU, 1119, 1039, 1039, 1119, 1119},
    {0x0410U, 1072, 1040, 1040, 1072, 1072},
    {0x0411U, 1073, 1041, 1041, 1073, 1073},
    {0x0412U, 1074, 1042, 1042, 1074, 1074},
    {0x0413U, 1075, 1043, 1043, 1075, 1075},
    {0x0414U, 1076, 1044, 1044, 1076, 1076},
    {0x0415U, 1077, 1045, 1045, 1077, 1077},
    {0x0416U, 1078, 1046, 1046, 1078, 1078},
    {0x0417U, 1079, 1047, 1047, 1079, 1079},
    {0x0418U, 1080, 1048, 1048, 1080, 1080},
    {0x0419U, 1081, 1049, 1049, 1081, 1081},
    {0x041AU, 1082, 1050, 1050, 1082, 1082},
    {0x041BU, 1083, 1051, 1051, 1083, 1083},
    {0x041CU, 1084, 1052, 1052, 1084, 1084},
    {0x041DU, 1085, 1053, 1053, 1085, 1085},
    {0x041EU, 1086, 1054, 1054, 1086, 1086},
    {0x041FU, 1087, 1055, 1055, 1087, 1087},
    {0x0420U, 1088, 1056, 1056, 1088, 1088},
    {0x0421U, 1089, 1057, 1057, 1089, 1089},
    {0x0422U, 1090, 1058, 1058, 1090, 1090},
    {0x0423U, 1091, 1059, 1059, 1091, 1091},
    {0x0424U, 1092, 1060, 1060, 1092, 1092},
    {0x0425U, 1093, 1061, 1061, 1093, 1093},
    {0x0426U, 1094, 1062, 1062, 1094, 1094},
    {0x0427U, 1095, 1063, 1063, 1095, 1095},
    {0x0428U, 1096, 1064, 1064, 1096, 1096},
    {0x0429U, 1097, 1065, 1065, 1097, 1097},
    {0x042AU, 1098, 1066, 1066, 1098, 1098},
    {0x042BU, 1099, 1067, 1067, 1099, 1099},
    {0x042CU, 1100, 1068, 1068, 1100, 1100},
    {0x042DU, 1101, 1069, 1069, 1101, 1101},
    {0x042EU, 1102, 1070, 1070, 1102, 1102},
    {0x042FU, 1103, 1071, 1071, 1103, 1103},
    {0x0430U, 1072, 1040, 1040, 1072, 1072},
    {0x0431U, 1073, 1041, 1041, 1073, 1073},
    {0x0432U, 1074, 1042, 1042, 1074, 1074},
    {0x0433U, 1075, 1043, 1043, 1075, 1075},
    {0x0434U, 1076, 1044, 1044, 1076, 1076},
    {0x0435U, 1077, 1045, 1045, 1077, 1077},
    {0x0436U, 1078, 1046, 1046, 1078, 1078},
    {0x0437U, 1079, 1047, 1047, 1079, 1079},
    {0x0438U, 1080, 1048, 1048, 1080, 1080},
    {0x0439U, 1081, 1049, 1049, 1081, 1081},
    {0x043AU, 1082, 1050, 1050, 1082, 1082},
    {0x043BU, 1083, 1051, 1051, 1083, 1083},
    {0x043CU, 1084, 1052, 1052, 1084, 1084},
    {0x043DU, 1085, 1053, 1053, 1085, 1085},
    {0x043EU, 1086, 1054, 1054, 1086, 1086},
    {0x043FU, 1087, 1055, 1055, 1087, 1087},
    {0x0440U, 1088, 1056, 1056, 1088, 1088},
    {0x0441U, 1089, 1057, 1057, 1089, 1089},
    {0x0442U, 1090, 1058, 1058, 1090, 1090},
    {0x0443U, 1091, 1059, 1059, 1091, 1091},
    {0x0444U, 1092, 1060, 1060, 1092, 1092},
    {0x0445U, 1093, 1061, 1061, 1093, 1093},
    {0x0446U, 1094, 1062, 1062, 1094, 1094},
    {0x0447U, 1095, 1063, 1063, 1095, 1095},
    {0x0448U, 1096, 1064, 1064, 1096, 1096},
    {0x0449U, 1097, 1065, 1065, 1097, 1097},
    {0x044AU, 1098, 1066, 1066, 1098, 1098},
    {0x044BU, 1099, 1067, 1067, 1099, 1099},
    {0x044CU, 1100, 1068, 1068, 1100, 1100},
    {0x044DU, 1101, 1069, 1069, 1101, 1101},
    {0x044EU, 1102, 1070, 1070, 1102, 1102},
    {0x044FU, 1103, 1071, 1071, 1103, 1103},
    {0x0450U, 1104, 1024, 1024, 1104, 1104},
    {0x0451U, 1105, 1025, 1025, 1105, 1105},
    {0x0452U, 1106, 1026, 1026, 1106, 1106},
    {0x0453U, 1107, 1027, 1027, 1107, 1107},
    {0x0454U, 1108, 1028, 1028, 1108, 1108},
    {0x0455U, 1109, 1029, 1029, 1109, 1109},
    {0x0456U, 1110, 1030, 1030, 1110, 1110},
    {0x0457U, 1111, 1031, 1031, 1111, 1111},
    {0x0458U, 1112, 1032, 1032, 1112, 1112},
    {0x0459U, 1113, 1033, 1033, 1113, 1113},
    {0x045AU, 1114, 1034, 1034, 1114, 1114},
    {0x045BU, 1115, 1035, 1035, 1115, 1115},
    {0x045CU, 1116, 1036, 1036, 1116, 1116},
    {0x045DU, 1117, 1037, 1037, 1117, 1117},
    {0x045EU, 1118, 1038, 1038, 1118, 1118},
    {0x045FU, 1119, 1039, 1039, 1119, 1119},
    {0x0460U, 1121, 1120, 1120, 1121, 1121},
    {0x0461U, 1121, 1120, 1120, 1121, 1121},
    {0x0462U, 1123, 1122, 1122, 1123, 1123},
    {0x0463U, 1123, 1122, 1122, 1123, 1123},
    {0x0464U, 1125, 1124, 1124, 1125, 1125},
    {0x0465U, 1125, 1124, 1124, 1125, 1125},
    {0x0466U, 1127, 1126, 1126, 1127, 1127},
    {0x0467U, 1127, 1126, 1126, 1127, 1127},
    {0x0468U, 1129, 1128, 1128, 1129, 1129},
    {0x0469U, 1129, 1128, 1128, 1129, 1129},
    {0x046AU, 1131, 1130, 1130, 1131, 1131},
    {0x046BU, 1131, 1130, 1130, 1131, 1131},
    {0x046CU, 1133, 1132, 1132, 1133, 1133},
    {0x046DU, 1133, 1132, 1132, 1133, 1133},
    {0x046EU, 1135, 1134, 1134, 1135, 1135},
    {0x046FU, 1135, 1134, 1134, 1135, 1135},
    {0x0470U, 1137, 1136, 1136, 1137, 1137},
    {0x0471U, 1137, 1136, 1136, 1137, 1137},
    {0x0472U, 1139, 1138, 1138, 1139, 1139},
    {0x0473U, 1139, 1138, 1138, 1139, 1139},
    {0x0474U, 1141, 1140, 1140, 1141, 1141},
    {0x0475U, 1141, 1140, 1140, 1141, 1141},
    {0x0476U, 1143, 1142, 1142, 1143, 1143},
    {0x0477U, 1143, 1142, 1142, 1143, 1143},
    {0x0478U, 1145, 1144, 1144, 1145, 1145},
    {0x0479U, 1145, 1144, 1144, 1145, 1145},
    {0x047AU, 1147, 1146, 1146, 1147, 1147},
    {0x047BU, 1147, 1146, 1146, 1147, 1147},
    {0x047CU, 1149, 1148, 1148, 1149, 1149},
    {0x047DU, 1149, 1148, 1148, 1149, 1149},
    {0x047EU, 1151, 1150, 1150, 1151, 1151},
    {0x047FU, 1151, 1150, 1150, 1151, 1151},
    {0x0480U, 1153, 1152, 1152, 1153, 1153},
    {0x0481U, 1153, 1152, 1152, 1153, 1153},
    {0x048AU, 1163, 1162, 1162, 1163, 1163},
    {0x048BU, 1163, 1162, 1162, 1163, 1163},
    {0x048CU, 1165, 1164, 1164, 1165, 1165},
    {0x048DU, 1165, 1164, 1164, 1165, 1165},
    {0x048EU, 1167, 1166, 1166, 1167, 1167},
    {0x048FU, 1167, 1166, 1166, 1167, 1167},
    {0x0490U, 1169, 1168, 1168, 1169, 1169},
    {0x0491U, 1169, 1168, 1168, 1169, 1169},
    {0x0492U, 1171, 1170, 1170, 1171, 1171},
    {0x0493U, 1171, 1170, 1170, 1171, 1171},
    {0x0494U, 1173, 1172, 1172, 1173, 1173},
    {0x0495U, 1173, 1172, 1172, 1173, 1173},
    {0x0496U, 1175, 1174, 1174, 1175, 1175},
    {0x0497U, 1175, 1174, 1174, 1175, 1175},
    {0x0498U, 1177, 1176, 1176, 1177, 1177},
    {0x0499U, 1177, 1176, 1176, 1177, 1177},
    {0x049AU, 1179, 1178, 1178, 1179, 1179},
    {0x049BU, 1179, 1178, 1178, 1179, 1179},
    {0x049CU, 1181, 1180, 1180, 1181, 1181},
    {0x049DU, 1181, 1180, 1180, 1181, 1181},
    {0x049EU, 1183, 1182, 1182, 1183, 1183},
    {0x049FU, 1183, 1182, 1182, 1183, 1183},
    {0x04A0U, 1185, 1184, 1184, 1185, 1185},
    {0x04A1U, 1185, 1184, 1184, 1185, 1185},
    {0x04A2U, 1187, 1186, 1186, 1187, 1187},
    {0x04A3U, 1187, 1186, 1186, 1187, 1187},
    {0x04A4U, 1189, 1188, 1188, 1189, 1189},
    {0x04A5U, 1189, 1188, 1188, 1189, 1189},
    {0x04A6U, 1191, 1190, 1190, 1191, 1191},
    {0x04A7U, 1191, 1190, 1190, 1191, 1191},
    {0x04A8U, 1193, 1192, 1192, 1193, 1193},
    {0x04A9U, 1193, 1192, 1192, 1193, 1193},
    {0x04AAU, 1195, 1194, 1194, 1195, 1195},
    {0x04ABU, 1195, 1194, 1194, 1195, 1195},
    {0x04ACU, 1197, 1196, 1196, 1197, 1197},
    {0x04ADU, 1197, 1196, 1196, 1197, 1197},
    {0x04AEU, 1199, 1198, 1198, 1199, 1199},
    {0x04AFU, 1199, 1198, 1198, 1199, 1199},
    {0x04B0U, 1201, 1200, 1200, 1201, 1201},
    {0x04B1U, 1201, 1200, 1200, 1201, 1201},
    {0x04B2U, 1203, 1202, 1202, 1203, 1203},
    {0x04B3U, 1203, 1202, 1202, 1203, 1203},
    {0x04B4U, 1205, 1204, 1204, 1205, 1205},
    {0x04B5U, 1205, 1204, 1204, 1205, 1205},
    {0x04B6U, 1207, 1206, 1206, 1207, 1207},
    {0x04B7U, 1207, 1206, 1206, 1207, 1207},
    {0x04B8U, 1209, 1208, 1208, 1209, 1209},
    {0x04B9U, 1209, 1208, 1208, 1209, 1209},
    {0x04BAU, 1211, 1210, 1210, 1211, 1211},
    {0x04BBU, 1211, 1210, 1210, 1211, 1211},
    {0x04BCU, 1213, 1212, 1212, 1213, 1213},
    {0x04BDU, 1213, 1212, 1212, 1213, 1213},
    {0x04BEU, 1215, 1214, 1214, 1215, 1215},
    {0x04BFU, 1215, 1214, 1214, 1215, 1215},
    {0x04C0U, 1231, 1216, 1216, 1231, 1231},
    {0x04C1U, 1218, 1217, 1217, 1218, 1218},
    {0x04C2U, 1218, 1217, 1217, 1218, 1218},
    {0x04C3U, 1220, 1219, 1219, 1220, 1220},
    {0x04C4U, 1220, 1219, 1219, 1220, 1220},
    {0x04C5U, 1222, 1221, 1221, 1222, 1222},
    {0x04C6U, 1222, 1221, 1221, 1222, 1222},
    {0x04C7U, 1224, 1223, 1223, 1224, 1224},
    {0x04C8U, 1224, 1223, 1223, 1224, 1224},
    {0x04C9U, 1226, 1225, 1225, 1226, 1226},
    {0x04CAU, 1226, 1225, 1225, 1226, 1226},
    {0x04CBU, 1228, 1227, 1227, 1228, 1228},
    {0x04CCU, 1228, 1227, 1227, 1228, 1228},
    {0x04CDU, 1230, 1229, 1229, 1230, 1230},
    {0x04CEU, 1230, 1229, 1229, 1230, 1230},
    {0x04CFU, 1231, 1216, 1216, 1231, 1231},
    {0x04D0U, 1233, 1232, 1232, 1233, 1233},
    {0x04D1U, 1233, 1232, 1232, 1233, 1233},
    {0x04D2U, 1235, 1234, 1234, 1235, 1235},
    {0x04D3U, 1235, 1234, 1234, 1235, 1235},
    {0x04D4U, 1237, 1236, 1236, 1237, 1237},
    {0x04D5U, 1237, 1236, 1236, 1237, 1237},
    {0x04D6U, 1239, 1238, 1238, 1239, 1239},
    {0x04D7U, 1239, 1238, 1238, 1239, 1239},
    {0x04D8U, 1241, 1240, 1240, 1241, 1241},
    {0x04D9U, 1241, 1240, 1240, 1241, 1241},
    {0x04DAU, 1243, 1242, 1242, 1243, 1243},
    {0x04DBU, 1243, 1242, 1242, 1243, 1243},
    {0x04DCU, 1245, 1244, 1244, 1245, 1245},
    {0x04DDU, 1245, 1244, 1244, 1245, 1245},
    {0x04DEU, 1247, 1246, 1246, 1247, 1247},
    {0x04DFU, 1247, 1246, 1246, 1247, 1247},
    {0x04E0U, 1249, 1248, 1248, 1249, 1249},
    {0x04E1U, 1249, 1248, 1248, 1249, 1249},
    {0x04E2U, 1251, 1250, 1250, 1251, 1251},
    {0x04E3U, 1251, 1250, 1250, 1251, 1251},
    {0x04E4U, 1253, 1252, 1252, 1253, 1253},
    {0x04E5U, 1253, 1252, 1252, 1253, 1253},
    {0x04E6U, 1255, 1254, 1254, 1255, 1255},
    {0x04E7U, 1255, 1254, 1254, 1255, 1255},
    {0x04E8U, 1257, 1256, 1256, 1257, 1257},
    {0x04E9U, 1257, 1256, 1256, 1257, 1257},
    {0x04EAU, 1259, 1258, 1258, 1259, 1259},
    {0x04EBU, 1259, 1258, 1258, 1259, 1259},
    {0x04ECU, 1261, 1260, 1260, 1261, 1261},
    {0x04EDU, 1261, 1260, 1260, 1261, 1261},
    {0x04EEU, 1263, 1262, 1262, 1263, 1263},
    {0x04EFU, 1263, 1262, 1262, 1263, 1263},
    {0x04F0U, 1265, 1264, 1264, 1265, 1265},
    {0x04F1U, 1265, 1264, 1264, 1265, 1265},
    {0x04F2U, 1267, 1266, 1266, 1267, 1267},
    {0x04F3U, 1267, 1266, 1266, 1267, 1267},
    {0x04F4U, 1269, 1268, 1268, 1269, 1269},
    {0x04F5U, 1269, 1268, 1268, 1269, 1269},
    {0x04F6U, 1271, 1270, 1270, 1271, 1271},
    {0x04F7U, 1271, 1270, 1270, 1271, 1271},
    {0x04F8U, 1273, 1272, 1272, 1273, 1273},
    {0x04F9U, 1273, 1272, 1272, 1273, 1273},
    {0x04FAU, 1275, 1274, 1274, 1275, 1275},
    {0x04FBU, 1275, 1274, 1274, 1275, 1275},
    {0x04FCU, 1277, 1276, 1276, 1277, 1277},
    {0x04FDU, 1277, 1276, 1276, 1277, 1277},
    {0x04FEU, 1279, 1278, 1278, 1279, 1279},
    {0x04FFU, 1279, 1278, 1278, 1279, 1279},
    {0x0500U, 1281, 1280, 1280, 1281, 1281},
    {0x0501U, 1281, 1280, 1280, 1281, 1281},
    {0x0502U, 1283, 1282, 1282, 1283, 1283},
    {0x0503U, 1283, 1282, 1282, 1283, 1283},
    {0x0504U, 1285, 1284, 1284, 1285, 1285},
    {0x0505U, 1285, 1284, 1284, 1285, 1285},
    {0x0506U, 1287, 1286, 1286, 1287, 1287},
    {0x0507U, 1287, 1286, 1286, 1287, 1287},
    {0x0508U, 1289, 1288, 1288, 1289, 1289},
    {0x0509U, 1289, 1288, 1288, 1289, 1289},
    {0x050AU, 1291, 1290, 1290, 1291, 1291},
    {0x050BU, 1291, 1290, 1290, 1291, 1291},
    {0x050CU, 1293, 1292, 1292, 1293, 1293},
    {0x050DU, 1293, 1292, 1292, 1293, 1293},
    {0x050EU, 1295, 1294, 1294, 1295, 1295},
    {0x050FU, 1295, 1294, 1294, 1295, 1295},
    {0x0510U, 1297, 1296, 1296, 1297, 1297},
    {0x0511U, 1297, 1296, 1296, 1297, 1297},
    {0x0512U, 1299, 1298, 1298, 1299, 1299},
    {0x0513U, 1299, 1298, 1298, 1299, 1299},
    {0x0514U, 1301, 1300, 1300, 1301, 1301},
    {0x0515U, 1301, 1300, 1300, 1301, 1301},
    {0x0516U, 1303, 1302, 1302, 1303, 1303},
    {0x0517U, 1303, 1302, 1302, 1303, 1303},
    {0x0518U, 1305, 1304, 1304, 1305, 1305},
    {0x0519U, 1305, 1304, 1304, 1305, 1305},
    {0x051AU, 1307, 1306, 1306, 1307, 1307},
    {0x051BU, 1307, 1306, 1306, 1307, 1307},
    {0x051CU, 1309, 1308, 1308, 1309, 1309},
    {0x051DU, 1309, 1308, 1308, 1309, 1309},
    {0x051EU, 1311, 1310, 1310, 1311, 1311},
    {0x051FU, 1311, 1310, 1310, 1311, 1311},
    {0x0520U, 1313, 1312, 1312, 1313, 1313},
    {0x0521U, 1313, 1312, 1312, 1313, 1313},
    {0x0522U, 1315, 1314, 1314, 1315, 1315},
    {0x0523U, 1315, 1314, 1314, 1315, 1315},
    {0x0524U, 1317, 1316, 1316, 1317, 1317},
    {0x0525U, 1317, 1316, 1316, 1317, 1317},
    {0x0526U, 1319, 1318, 1318, 1319, 1319},
    {0x0527U, 1319, 1318, 1318, 1319, 1319},
    {0x0528U, 1321, 1320, 1320, 1321, 1321},
    {0x0529U, 1321, 1320, 1320, 1321, 1321},
    {0x052AU, 1323, 1322, 1322, 1323, 1323},
    {0x052BU, 1323, 1322, 1322, 1323, 1323},
    {0x052CU, 1325, 1324, 1324, 1325, 1325},
    {0x052DU, 1325, 1324, 1324, 1325, 1325},
    {0x052EU, 1327, 1326, 1326, 1327, 1327},
    {0x052FU, 1327, 1326, 1326, 1327, 1327},
    {0x0531U, 1377, 1329, 1329, 1377, 1377},
    {0x0532U, 1378, 1330, 1330, 1378, 1378},
    {0x0533U, 1379, 1331, 1331, 1379, 1379},
    {0x0534U, 1380, 1332, 1332, 1380, 1380},
    {0x0535U, 1381, 1333, 1333, 1381, 1381},
    {0x0536U, 1382, 1334, 1334, 1382, 1382},
    {0x0537U, 1383, 1335, 1335, 1383, 1383},
    {0x0538U, 1384, 1336, 1336, 1384, 1384},
    {0x0539U, 1385, 1337, 1337, 1385, 1385},
    {0x053AU, 1386, 1338, 1338, 1386, 1386},
    {0x053BU, 1387, 1339, 1339, 1387, 1387},
    {0x053CU, 1388, 1340, 1340, 1388, 1388},
    {0x053DU, 1389, 1341, 1341, 1389, 1389},
    {0x053EU, 1390, 1342, 1342, 1390, 1390},
    {0x053FU, 1391, 1343, 1343, 1391, 1391},
    {0x0540U, 1392, 1344, 1344, 1392, 1392},
    {0x0541U, 1393, 1345, 1345, 1393, 1393},
    {0x0542U, 1394, 1346, 1346, 1394, 1394},
    {0x0543U, 1395, 1347, 1347, 1395, 1395},
    {0x0544U, 1396, 1348, 1348, 1396, 1396},
    {0x0545U, 1397, 1349, 1349, 1397, 1397},
    {0x0546U, 1398, 1350, 1350, 1398, 1398},
    {0x0547U, 1399, 1351, 1351, 1399, 1399},
    {0x0548U, 1400, 1352, 1352, 1400, 1400},
    {0x0549U, 1401, 1353, 1353, 1401, 1401},
    {0x054AU, 1402, 1354, 1354, 1402, 1402},
    {0x054BU, 1403, 1355, 1355, 1403, 1403},
    {0x054CU, 1404, 1356, 1356, 1404, 1404},
    {0x054DU, 1405, 1357, 1357, 1405, 1405},
    {0x054EU, 1406, 1358, 1358, 1406, 1406},
    {0x054FU, 1407, 1359, 1359, 1407, 1407},
    {0x0550U, 1408, 1360, 1360, 1408, 1408},
    {0x0551U, 1409, 1361, 1361, 1409, 1409},
    {0x0552U, 1410, 1362, 1362, 1410, 1410},
    {0x0553U, 1411, 1363, 1363, 1411, 1411},
    {0x0554U, 1412, 1364, 1364, 1412, 1412},
    {0x0555U, 1413, 1365, 1365, 1413, 1413},
    {0x0556U, 1414, 1366, 1366, 1414, 1414},
    {0x0561U, 1377, 1329, 1329, 1377, 1377},
    {0x0562U, 1378, 1330, 1330, 1378, 1378},
    {0x0563U, 1379, 1331, 1331, 1379, 1379},
    {0x0564U, 1380, 1332, 1332, 1380, 1380},
    {0x0565U, 1381, 1333, 1333, 1381, 1381},
    {0x0566U, 1382, 1334, 1334, 1382, 1382},
    {0x0567U, 1383, 1335, 1335, 1383, 1383},
    {0x0568U, 1384, 1336, 1336, 1384, 1384},
    {0x0569U, 1385, 1337, 1337, 1385, 1385},
    {0x056AU, 1386, 1338, 1338, 1386, 1386},
    {0x056BU, 1387, 1339, 1339, 1387, 1387},
    {0x056CU, 1388, 1340, 1340, 1388, 1388},
    {0x056DU, 1389, 1341, 1341, 1389, 1389},
    {0x056EU, 1390, 1342, 1342, 1390, 1390},
    {0x056FU, 1391, 1343, 1343, 1391, 1391},
    {0x0570U, 1392, 1344, 1344, 1392, 1392},
    {0x0571U, 1393, 1345, 1345, 1393, 1393},
    {0x0572U, 1394, 1346, 1346, 1394, 1394},
    {0x0573U, 1395, 1347, 1347, 1395, 1395},
    {0x0574U, 1396, 1348, 1348, 1396, 1396},
    {0x0575U, 1397, 1349, 1349, 1397, 1397},
    {0x0576U, 1398, 1350, 1350, 1398, 1398},
    {0x0577U, 1399, 1351, 1351, 1399, 1399},
    {0x0578U, 1400, 1352, 1352, 1400, 1400},
    {0x0579U, 1401, 1353, 1353, 1401, 1401},
    {0x057AU, 1402, 1354, 1354, 1402, 1402},
    {0x057BU, 1403, 1355, 1355, 1403, 1403},
    {0x057CU, 1404, 1356, 1356, 1404, 1404},
    {0x057DU, 1405, 1357, 1357, 1405, 1405},
    {0x057EU, 1406, 1358, 1358, 1406, 1406},
    {0x057FU, 1407, 1359, 1359, 1407, 1407},
    {0x0580U, 1408, 1360, 1360, 1408, 1408},
    {0x0581U, 1409, 1361, 1361, 1409, 1409},
    {0x0582U, 1410, 1362, 1362, 1410, 1410},
    {0x0583U, 1411, 1363, 1363, 1411, 1411},
    {0x0584U, 1412, 1364, 1364, 1412, 1412},
    {0x0585U, 1413, 1365, 1365, 1413, 1413},
    {0x0586U, 1414, 1366, 1366, 1414, 1414},
    {0x0587U, 1415, -226, -242, -258, 1415},
    {0x10A0U, 11520, 4256, 4256, 11520, 11520},
    {0x10A1U, 11521, 4257, 4257, 11521, 11521},
    {0x10A2U, 11522, 4258, 4258, 11522, 11522},
    {0x10A3U, 11523, 4259, 4259, 11523, 11523},
    {0x10A4U, 11524, 4260, 4260, 11524, 11524},
    {0x10A5U, 11525, 4261, 4261, 11525, 11525},
    {0x10A6U, 11526, 4262, 4262, 11526, 11526},
    {0x10A7U, 11527, 4263, 4263, 11527, 11527},
    {0x10A8U, 11528, 4264, 4264, 11528, 11528},
    {0x10A9U, 11529, 4265, 4265, 11529, 11529},
    {0x10AAU, 11530, 4266, 4266, 11530, 11530},
    {0x10ABU, 11531, 4267, 4267, 11531, 11531},
    {0x10ACU, 11532, 4268, 4268, 11532, 11532},
    {0x10ADU, 11533, 4269, 4269, 11533, 11533},
    {0x10AEU, 11534, 4270, 4270, 11534, 11534},
    {0x10AFU, 11535, 4271, 4271, 11535, 11535},
    {0x10B0U, 11536, 4272, 4272, 11536, 11536},
    {0x10B1U, 11537, 4273, 4273, 11537, 11537},
    {0x10B2U, 11538, 4274, 4274, 11538, 11538},
    {0x10B3U, 11539, 4275, 4275, 11539, 11539},
    {0x10B4U, 11540, 4276, 4276, 11540, 11540},
    {0x10B5U, 11541, 4277, 4277, 11541, 11541},
    {0x10B6U, 11542, 4278, 4278, 11542, 11542},
    {0x10B7U, 11543, 4279, 4279, 11543, 11543},
    {0x10B8U, 11544, 4280, 4280, 11544, 11544},
    {0x10B9U, 11545, 4281, 4281, 11545, 11545},
    {0x10BAU, 11546, 4282, 4282, 11546, 11546},
    {0x10BBU, 11547, 4283, 4283, 11547, 11547},
    {0x10BCU, 11548, 4284, 4284, 11548, 11548},
    {0x10BDU, 11549, 4285, 4285, 11549, 11549},
    {0x10BEU, 11550, 4286, 4286, 11550, 11550},
    {0x10BFU, 11551, 4287, 4287, 11551, 11551},
    {0x10C0U, 11552, 4288, 4288, 11552, 11552},
    {0x10C1U, 11553, 4289, 4289, 11553, 11553},
    {0x10C2U, 11554, 4290, 4290, 11554, 11554},
    {0x10C3U, 11555, 4291, 4291, 11555, 11555},
    {0x10C4U, 11556, 4292, 4292, 11556, 11556},
    {0x10C5U, 11557, 4293, 4293, 11557, 11557},
    {0x10C7U, 11559, 4295, 4295, 11559, 11559},
    {0x10CDU, 11565, 4301, 4301, 11565, 11565},
    {0x10D0U, 4304, 7312, 4304, 4304, 4304},
    {0x10D1U, 4305, 7313, 4305, 4305, 4305},
    {0x10D2U, 4306, 7314, 4306, 4306, 4306},
    {0x10D3U, 4307, 7315, 4307, 4307, 4307},
    {0x10D4U, 4308, 7316, 4308, 4308, 4308},
    {0x10D5U, 4309, 7317, 4309, 4309, 4309},
    {0x10D6U, 4310, 7318, 4310, 4310, 4310},
    {0x10D7U, 4311, 7319, 4311, 4311, 4311},
    {0x10D8U, 4312, 7320, 4312, 4312, 4312},
    {0x10D9U, 4313, 7321, 4313, 4313, 4313},
    {0x10DAU, 4314, 7322, 4314, 4314, 4314},
    {0x10DBU, 4315, 7323, 4315, 4315, 4315},
    {0x10DCU, 4316, 7324, 4316, 4316, 4316},
    {0x10DDU, 4317, 7325, 4317, 4317, 4317},
    {0x10DEU, 4318, 7326, 4318, 4318, 4318},
    {0x10DFU, 4319, 7327, 4319, 4319, 4319},
    {0x10E0U, 4320, 7328, 4320, 4320, 4320},
    {0x10E1U, 4321, 7329, 4321, 4321, 4321},
    {0x10E2U, 4322, 7330, 4322, 4322, 4322},
    {0x10E3U, 4323, 7331, 4323, 4323, 4323},
    {0x10E4U, 4324, 7332, 4324, 4324, 4324},
    {0x10E5U, 4325, 7333, 4325, 4325, 4325},
    {0x10E6U, 4326, 7334, 4326, 4326, 4326},
    {0x10E7U, 4327, 7335, 4327, 4327, 4327},
    {0x10E8U, 4328, 7336, 4328, 4328, 4328},
    {0x10E9U, 4329, 7337, 4329, 4329, 4329},
    {0x10EAU, 4330, 7338, 4330, 4330, 4330},
    {0x10EBU, 4331, 7339, 4331, 4331, 4331},
    {0x10ECU, 4332, 7340, 4332, 4332, 4332},
    {0x10EDU, 4333, 7341, 4333, 4333, 4333},
    {0x10EEU, 4334, 7342, 4334, 4334, 4334},
    {0x10EFU, 4335, 7343, 4335, 4335, 4335},
    {0x10F0U, 4336, 7344, 4336, 4336, 4336},
    {0x10F1U, 4337, 7345, 4337, 4337, 4337},
    {0x10F2U, 4338, 7346, 4338, 4338, 4338},
    {0x10F3U, 4339, 7347, 4339, 4339, 4339},
    {0x10F4U, 4340, 7348, 4340, 4340, 4340},
    {0x10F5U, 4341, 7349, 4341, 4341, 4341},
    {0x10F6U, 4342, 7350, 4342, 4342, 4342},
    {0x10F7U, 4343, 7351, 4343, 4343, 4343},
    {0x10F8U, 4344, 7352, 4344, 4344, 4344},
    {0x10F9U, 4345, 7353, 4345, 4345, 4345},
    {0x10FAU, 4346, 7354, 4346, 4346, 4346},
    {0x10FDU, 4349, 7357, 4349, 4349, 4349},
    {0x10FEU, 4350, 7358, 4350, 4350, 4350},
    {0x10FFU, 4351, 7359, 4351, 4351, 4351},
    {0x13A0U, 43888, 5024, 5024, 5024, 5024},
    {0x13A1U, 43889, 5025, 5025, 5025, 5025},
    {0x13A2U, 43890, 5026, 5026, 5026, 5026},
    {0x13A3U, 43891, 5027, 5027, 5027, 5027},
    {0x13A4U, 43892, 5028, 5028, 5028, 5028},
    {0x13A5U, 43893, 5029, 5029, 5029, 5029},
    {0x13A6U, 43894, 5030, 5030, 5030, 5030},
    {0x13A7U, 43895, 5031, 5031, 5031, 5031},
    {0x13A8U, 43896, 5032, 5032, 5032, 5032},
    {0x13A9U, 43897, 5033, 5033, 5033, 5033},
    {0x13AAU, 43898, 5034, 5034, 5034, 5034},
    {0x13ABU, 43899, 5035, 5035, 5035, 5035},
    {0x13ACU, 43900, 5036, 5036, 5036, 5036},
    {0x13ADU, 43901, 5037, 5037, 5037, 5037},
    {0x13AEU, 43902, 5038, 5038, 5038, 5038},
    {0x13AFU, 43903, 5039, 5039, 5039, 5039},
    {0x13B0U, 43904, 5040, 5040, 5040, 5040},
    {0x13B1U, 43905, 5041, 5041, 5041, 5041},
    {0x13B2U, 43906, 5042, 5042, 5042, 5042},
    {0x13B3U, 43907, 5043, 5043, 5043, 5043},
    {0x13B4U, 43908, 5044, 5044, 5044, 5044},
    {0x13B5U, 43909, 5045, 5045, 5045, 5045},
    {0x13B6U, 43910, 5046, 5046, 5046, 5046},
    {0x13B7U, 43911, 5047, 5047, 5047, 5047},
    {0x13B8U, 43912, 5048, 5048, 5048, 5048},
    {0x13B9U, 43913, 5049, 5049, 5049, 5049},
    {0x13BAU, 43914, 5050, 5050, 5050, 5050},
    {0x13BBU, 43915, 5051, 5051, 5051, 5051},
    {0x13BCU, 43916, 5052, 5052, 5052, 5052},
    {0x13BDU, 43917, 5053, 5053, 5053, 5053},
    {0x13BEU, 43918, 5054, 5054, 5054, 5054},
    {0x13BFU, 43919, 5055, 5055, 5055, 5055},
    {0x13C0U, 43920, 5056, 5056, 5056, 5056},
    {0x13C1U, 43921, 5057, 5057, 5057, 5057},
    {0x13C2U, 43922, 5058, 5058, 5058, 5058},
    {0x13C3U, 43923, 5059, 5059, 5059, 5059},
    {0x13C4U, 43924, 5060, 5060, 5060, 5060},
    {0x13C5U, 43925, 5061, 5061, 5061, 5061},
    {0x13C6U, 43926, 5062, 5062, 5062, 5062},
    {0x13C7U, 43927, 5063, 5063, 5063, 5063},
    {0x13C8U, 43928, 5064, 5064, 5064, 5064},
    {0x13C9U, 43929, 5065, 5065, 5065, 5065},
    {0x13CAU, 43930, 5066, 5066, 5066, 5066},
    {0x13CBU, 43931, 5067, 5067, 5067, 5067},
    {0x13CCU, 43932, 5068, 5068, 5068, 5068},
    {0x13CDU, 43933, 5069, 5069, 5069, 5069},
    {0x13CEU, 43934, 5070, 5070, 5070, 5070},
    {0x13CFU, 43935, 5071, 5071, 5071, 5071},
    {0x13D0U, 43936, 5072, 5072, 5072, 5072},
    {0x13D1U, 43937, 5073, 5073, 5073, 5073},
    {0x13D2U, 43938, 5074, 5074, 5074, 5074},
    {0x13D3U, 43939, 5075, 5075, 5075, 5075},
    {0x13D4U, 43940, 5076, 5076, 5076, 5076},
    {0x13D5U, 43941, 5077, 5077, 5077, 5077},
    {0x13D6U, 43942, 5078, 5078, 5078, 5078},
    {0x13D7U, 43943, 5079, 5079, 5079, 5079},
    {0x13D8U, 43944, 5080, 5080, 5080, 5080},
    {0x13D9U, 43945, 5081, 5081, 5081, 5081},
    {0x13DAU, 43946, 5082, 5082, 5082, 5082},
    {0x13DBU, 43947, 5083, 5083, 5083, 5083},
    {0x13DCU, 43948, 5084, 5084, 5084, 5084},
    {0x13DDU, 43949, 5085, 5085, 5085, 5085},
    {0x13DEU, 43950, 5086, 5086, 5086, 5086},
    {0x13DFU, 43951, 5087, 5087, 5087, 5087},
    {0x13E0U, 43952, 5088, 5088, 5088, 5088},
    {0x13E1U, 43953, 5089, 5089, 5089, 5089},
    {0x13E2U, 43954, 5090, 5090, 5090, 5090},
    {0x13E3U, 43955, 5091, 5091, 5091, 5091},
    {0x13E4U, 43956, 5092, 5092, 5092, 5092},
    {0x13E5U, 43957, 5093, 5093, 5093, 5093},
    {0x13E6U, 43958, 5094, 5094, 5094, 5094},
    {0x13E7U, 43959, 5095, 5095, 5095, 5095},
    {0x13E8U, 43960, 5096, 5096, 5096, 5096},
    {0x13E9U, 43961, 5097, 5097, 5097, 5097},
    {0x13EAU, 43962, 5098, 5098, 5098, 5098},
    {0x13EBU, 43963, 5099, 5099, 5099, 5099},
    {0x13ECU, 43964, 5100, 5100, 5100, 5100},
    {0x13EDU, 43965, 5101, 5101, 5101, 5101},
    {0x13EEU, 43966, 5102, 5102, 5102, 5102},
    {0x13EFU, 43967, 5103, 5103, 5103, 5103},
    {0x13F0U, 5112, 5104, 5104, 5104, 5104},
    {0x13F1U, 5113, 5105, 5105, 5105, 5105},
    {0x13F2U, 5114, 5106, 5106, 5106, 5106},
    {0x13F3U, 5115, 5107, 5107, 5107, 5107},
    {0x13F4U, 5116, 5108, 5108, 5108, 5108},
    {0x13F5U, 5117, 5109, 5109, 5109, 5109},
    {0x13F8U, 5112, 5104, 5104, 5104, 5104},
    {0x13F9U, 5113, 5105, 5105, 5105, 5105},
    {0x13FAU, 5114, 5106, 5106, 5106, 5106},
    {0x13FBU, 5115, 5107, 5107, 5107, 5107},
    {0x13FCU, 5116, 5108, 5108, 5108, 5108},
    {0x13FDU, 5117, 5109, 5109, 5109, 5109},
    {0x1C80U, 7296, 1042, 1042, 1074, 1074},
    {0x1C81U, 7297, 1044, 1044, 1076, 1076},
    {0x1C82U, 7298, 1054, 1054, 1086, 1086},
    {0x1C83U, 7299, 1057, 1057, 1089, 1089},
    {0x1C84U, 7300, 1058, 1058, 1090, 1090},
    {0x1C85U, 7301, 1058, 1058, 1090, 1090},
    {0x1C86U, 7302, 1066, 1066, 1098, 1098},
    {0x1C87U, 7303, 1122, 1122, 1123, 1123},
    {0x1C88U, 7304, 42570, 42570, 42571, 42571},
    {0x1C89U, 7306, 7305, 7305, 7306, 7306},
    {0x1C8AU, 7306, 7305, 7305, 7306, 7306},
    {0x1C90U, 4304, 7312, 7312, 4304, 4304},
    {0x1C91U, 4305, 7313, 7313, 4305, 4305},
    {0x1C92U, 4306, 7314, 7314, 4306, 4306},
    {0x1C93U, 4307, 7315, 7315, 4307, 4307},
    {0x1C94U, 4308, 7316, 7316, 4308, 4308},
    {0x1C95U, 4309, 7317, 7317, 4309, 4309},
    {0x1C96U, 4310, 7318, 7318, 4310, 4310},
    {0x1C97U, 4311, 7319, 7319, 4311, 4311},
    {0x1C98U, 4312, 7320, 7320, 4312, 4312},
    {0x1C99U, 4313, 7321, 7321, 4313, 4313},
    {0x1C9AU, 4314, 7322, 7322, 4314, 4314},
    {0x1C9BU, 4315, 7323, 7323, 4315, 4315},
    {0x1C9CU, 4316, 7324, 7324, 4316, 4316},
    {0x1C9DU, 4317, 7325, 7325, 4317, 4317},
    {0x1C9EU, 4318, 7326, 7326, 4318, 4318},
    {0x1C9FU, 4319, 7327, 7327, 4319, 4319},
    {0x1CA0U, 4320, 7328, 7328, 4320, 4320},
    {0x1CA1U, 4321, 7329, 7329, 4321, 4321},
    {0x1CA2U, 4322, 7330, 7330, 4322, 4322},
    {0x1CA3U, 4323, 7331, 7331, 4323, 4323},
    {0x1CA4U, 4324, 7332, 7332, 4324, 4324},
    {0x1CA5U, 4325, 7333, 7333, 4325, 4325},
    {0x1CA6U, 4326, 7334, 7334, 4326, 4326},
    {0x1CA7U, 4327, 7335, 7335, 4327, 4327},
    {0x1CA8U, 4328, 7336, 7336, 4328, 4328},
    {0x1CA9U, 4329, 7337, 7337, 4329, 4329},
    {0x1CAAU, 4330, 7338, 7338, 4330, 4330},
    {0x1CABU, 4331, 7339, 7339, 4331, 4331},
    {0x1CACU, 4332, 7340, 7340, 4332, 4332},
    {0x1CADU, 4333, 7341, 7341, 4333, 4333},
    {0x1CAEU, 4334, 7342, 7342, 4334, 4334},
    {0x1CAFU, 4335, 7343, 7343, 4335, 4335},
    {0x1CB0U, 4336, 7344, 7344, 4336, 4336},
    {0x1CB1U, 4337, 7345, 7345, 4337, 4337},
    {0x1CB2U, 4338, 7346, 7346, 4338, 4338},
    {0x1CB3U, 4339, 7347, 7347, 4339, 4339},
    {0x1CB4U, 4340, 7348, 7348, 4340, 4340},
    {0x1CB5U, 4341, 7349, 7349, 4341, 4341},
    {0x1CB6U, 4342, 7350, 7350, 4342, 4342},
    {0x1CB7U, 4343, 7351, 7351, 4343, 4343},
    {0x1CB8U, 4344, 7352, 7352, 4344, 4344},
    {0x1CB9U, 4345, 7353, 7353, 4345, 4345},
    {0x1CBAU, 4346, 7354, 7354, 4346, 4346},
    {0x1CBDU, 4349, 7357, 7357, 4349, 4349},
    {0x1CBEU, 4350, 7358, 7358, 4350, 4350},
    {0x1CBFU, 4351, 7359, 7359, 4351, 4351},
    {0x1D79U, 7545, 42877, 42877, 7545, 7545},
    {0x1D7DU, 7549, 11363, 11363, 7549, 7549},
    {0x1D8EU, 7566, 42950, 42950, 7566, 7566},
    {0x1E00U, 7681, 7680, 7680, 7681, 7681},
    {0x1E01U, 7681, 7680, 7680, 7681, 7681},
    {0x1E02U, 7683, 7682, 7682, 7683, 7683},
    {0x1E03U, 7683, 7682, 7682, 7683, 7683},
    {0x1E04U, 7685, 7684, 7684, 7685, 7685},
    {0x1E05U, 7685, 7684, 7684, 7685, 7685},
    {0x1E06U, 7687, 7686, 7686, 7687, 7687},
    {0x1E07U, 7687, 7686, 7686, 7687, 7687},
    {0x1E08U, 7689, 7688, 7688, 7689, 7689},
    {0x1E09U, 7689, 7688, 7688, 7689, 7689},
    {0x1E0AU, 7691, 7690, 7690, 7691, 7691},
    {0x1E0BU, 7691, 7690, 7690, 7691, 7691},
    {0x1E0CU, 7693, 7692, 7692, 7693, 7693},
    {0x1E0DU, 7693, 7692, 7692, 7693, 7693},
    {0x1E0EU, 7695, 7694, 7694, 7695, 7695},
    {0x1E0FU, 7695, 7694, 7694, 7695, 7695},
    {0x1E10U, 7697, 7696, 7696, 7697, 7697},
    {0x1E11U, 7697, 7696, 7696, 7697, 7697},
    {0x1E12U, 7699, 7698, 7698, 7699, 7699},
    {0x1E13U, 7699, 7698, 7698, 7699, 7699},
    {0x1E14U, 7701, 7700, 7700, 7701, 7701},
    {0x1E15U, 7701, 7700, 7700, 7701, 7701},
    {0x1E16U, 7703, 7702, 7702, 7703, 7703},
    {0x1E17U, 7703, 7702, 7702, 7703, 7703},
    {0x1E18U, 7705, 7704, 7704, 7705, 7705},
    {0x1E19U, 7705, 7704, 7704, 7705, 7705},
    {0x1E1AU, 7707, 7706, 7706, 7707, 7707},
    {0x1E1BU, 7707, 7706, 7706, 7707, 7707},
    {0x1E1CU, 7709, 7708, 7708, 7709, 7709},
    {0x1E1DU, 7709, 7708, 7708, 7709, 7709},
    {0x1E1EU, 7711, 7710, 7710, 7711, 7711},
    {0x1E1FU, 7711, 7710, 7710, 7711, 7711},
    {0x1E20U, 7713, 7712, 7712, 7713, 7713},
    {0x1E21U, 7713, 7712, 7712, 7713, 7713},
    {0x1E22U, 7715, 7714, 7714, 7715, 7715},
    {0x1E23U, 7715, 7714, 7714, 7715, 7715},
    {0x1E24U, 7717, 7716, 7716, 7717, 7717},
    {0x1E25U, 7717, 7716, 7716, 7717, 7717},
    {0x1E26U, 7719, 7718, 7718, 7719, 7719},
    {0x1E27U, 7719, 7718, 7718, 7719, 7719},
    {0x1E28U, 7721, 7720, 7720, 7721, 7721},
    {0x1E29U, 7721, 7720, 7720, 7721, 7721},
    {0x1E2AU, 7723, 7722, 7722, 7723, 7723},
    {0x1E2BU, 7723, 7722, 7722, 7723, 7723},
    {0x1E2CU, 7725, 7724, 7724, 7725, 7725},
    {0x1E2DU, 7725, 7724, 7724, 7725, 7725},
    {0x1E2EU, 7727, 7726, 7726, 7727, 7727},
    {0x1E2FU, 7727, 7726, 7726, 7727, 7727},
    {0x1E30U, 7729, 7728, 7728, 7729, 7729},
    {0x1E31U, 7729, 7728, 7728, 7729, 7729},
    {0x1E32U, 7731, 7730, 7730, 7731, 7731},
    {0x1E33U, 7731, 7730, 7730, 7731, 7731},
    {0x1E34U, 7733, 7732, 7732, 7733, 7733},
    {0x1E35U, 7733, 7732, 7732, 7733, 7733},
    {0x1E36U, 7735, 7734, 7734, 7735, 7735},
    {0x1E37U, 7735, 7734, 7734, 7735, 7735},
    {0x1E38U, 7737, 7736, 7736, 7737, 7737},
    {0x1E39U, 7737, 7736, 7736, 7737, 7737},
    {0x1E3AU, 7739, 7738, 7738, 7739, 7739},
    {0x1E3BU, 7739, 7738, 7738, 7739, 7739},
    {0x1E3CU, 7741, 7740, 7740, 7741, 7741},
    {0x1E3DU, 7741, 7740, 7740, 7741, 7741},
    {0x1E3EU, 7743, 7742, 7742, 7743, 7743},
    {0x1E3FU, 7743, 7742, 7742, 7743, 7743},
    {0x1E40U, 7745, 7744, 7744, 7745, 7745},
    {0x1E41U, 7745, 7744, 7744, 7745, 7745},
    {0x1E42U, 7747, 7746, 7746, 7747, 7747},
    {0x1E43U, 7747, 7746, 7746, 7747, 7747},
    {0x1E44U, 7749, 7748, 7748, 7749, 7749},
    {0x1E45U, 7749, 7748, 7748, 7749, 7749},
    {0x1E46U, 7751, 7750, 7750, 7751, 7751},
    {0x1E47U, 7751, 7750, 7750, 7751, 7751},
    {0x1E48U, 7753, 7752, 7752, 7753, 7753},
    {0x1E49U, 7753, 7752, 7752, 7753, 7753},
    {0x1E4AU, 7755, 7754, 7754, 7755, 7755},
    {0x1E4BU, 7755, 7754, 7754, 7755, 7755},
    {0x1E4CU, 7757, 7756, 7756, 7757, 7757},
    {0x1E4DU, 7757, 7756, 7756, 7757, 7757},
    {0x1E4EU, 7759, 7758, 7758, 7759, 7759},
    {0x1E4FU, 7759, 7758, 7758, 7759, 7759},
    {0x1E50U, 7761, 7760, 7760, 7761, 7761},
    {0x1E51U, 7761, 7760, 7760, 7761, 7761},
    {0x1E52U, 7763, 7762, 7762, 7763, 7763},
    {0x1E53U, 7763, 7762, 7762, 7763, 7763},
    {0x1E54U, 7765, 7764, 7764, 7765, 7765},
    {0x1E55U, 7765, 7764, 7764, 7765, 7765},
    {0x1E56U, 7767, 7766, 7766, 7767, 7767},
    {0x1E57U, 7767, 7766, 7766, 7767, 7767},
    {0x1E58U, 7769, 7768, 7768, 7769, 7769},
    {0x1E59U, 7769, 7768, 7768, 7769, 7769},
    {0x1E5AU, 7771, 7770, 7770, 7771, 7771},
    {0x1E5BU, 7771, 7770, 7770, 7771, 7771},
    {0x1E5CU, 7773, 7772, 7772, 7773, 7773},
    {0x1E5DU, 7773, 7772, 7772, 7773, 7773},
    {0x1E5EU, 7775, 7774, 7774, 7775, 7775},
    {0x1E5FU, 7775, 7774, 7774, 7775, 7775},
    {0x1E60U, 7777, 7776, 7776, 7777, 7777},
    {0x1E61U, 7777, 7776, 7776, 7777, 7777},
    {0x1E62U, 7779, 7778, 7778, 7779, 7779},
    {0x1E63U, 7779, 7778, 7778, 7779, 7779},
    {0x1E64U, 7781, 7780, 7780, 7781, 7781},
    {0x1E65U, 7781, 7780, 7780, 7781, 7781},
    {0x1E66U, 7783, 7782, 7782, 7783, 7783},
    {0x1E67U, 7783, 7782, 7782, 7783, 7783},
    {0x1E68U, 7785, 7784, 7784, 7785, 7785},
    {0x1E69U, 7785, 7784, 7784, 7785, 7785},
    {0x1E6AU, 7787, 7786, 7786, 7787, 7787},
    {0x1E6BU, 7787, 7786, 7786, 7787, 7787},
    {0x1E6CU, 7789, 7788, 7788, 7789, 7789},
    {0x1E6DU, 7789, 7788, 7788, 7789, 7789},
    {0x1E6EU, 7791, 7790, 7790, 7791, 7791},
    {0x1E6FU, 7791, 7790, 7790, 7791, 7791},
    {0x1E70U, 7793, 7792, 7792, 7793, 7793},
    {0x1E71U, 7793, 7792, 7792, 7793, 7793},
    {0x1E72U, 7795, 7794, 7794, 7795, 7795},
    {0x1E73U, 7795, 7794, 7794, 7795, 7795},
    {0x1E74U, 7797, 7796, 7796, 7797, 7797},
    {0x1E75U, 7797, 7796, 7796, 7797, 7797},
    {0x1E76U, 7799, 7798, 7798, 7799, 7799},
    {0x1E77U, 7799, 7798, 7798, 7799, 7799},
    {0x1E78U, 7801, 7800, 7800, 7801, 7801},
    {0x1E79U, 7801, 7800, 7800, 7801, 7801},
    {0x1E7AU, 7803, 7802, 7802, 7803, 7803},
    {0x1E7BU, 7803, 7802, 7802, 7803, 7803},
    {0x1E7CU, 7805, 7804, 7804, 7805, 7805},
    {0x1E7DU, 7805, 7804, 7804, 7805, 7805},
    {0x1E7EU, 7807, 7806, 7806, 7807, 7807},
    {0x1E7FU, 7807, 7806, 7806, 7807, 7807},
    {0x1E80U, 7809, 7808, 7808, 7809, 7809},
    {0x1E81U, 7809, 7808, 7808, 7809, 7809},
    {0x1E82U, 7811, 7810, 7810, 7811, 7811},
    {0x1E83U, 7811, 7810, 7810, 7811, 7811},
    {0x1E84U, 7813, 7812, 7812, 7813, 7813},
    {0x1E85U, 7813, 7812, 7812, 7813, 7813},
    {0x1E86U, 7815, 7814, 7814, 7815, 7815},
    {0x1E87U, 7815, 7814, 7814, 7815, 7815},
    {0x1E88U, 7817, 7816, 7816, 7817, 7817},
    {0x1E89U, 7817, 7816, 7816, 7817, 7817},
    {0x1E8AU, 7819, 7818, 7818, 7819, 7819},
    {0x1E8BU, 7819, 7818, 7818, 7819, 7819},
    {0x1E8CU, 7821, 7820, 7820, 7821, 7821},
    {0x1E8DU, 7821, 7820, 7820, 7821, 7821},
    {0x1E8EU, 7823, 7822, 7822, 7823, 7823},
    {0x1E8FU, 7823, 7822, 7822, 7823, 7823},
    {0x1E90U, 7825, 7824, 7824, 7825, 7825},
    {0x1E91U, 7825, 7824, 7824, 7825, 7825},
    {0x1E92U, 7827, 7826, 7826, 7827, 7827},
    {0x1E93U, 7827, 7826, 7826, 7827, 7827},
    {0x1E94U, 7829, 7828, 7828, 7829, 7829},
    {0x1E95U, 7829, 7828, 7828, 7829, 7829},
    {0x1E96U, 7830, -274, -274, -290, 7830},
    {0x1E97U, 7831, -306, -306, -322, 7831},
    {0x1E98U, 7832, -338, -338, -354, 7832},
    {0x1E99U, 7833, -370, -370, -386, 7833},
    {0x1E9AU, 7834, -402, -402, -418, 7834},
    {0x1E9BU, 7835, 7776, 7776, 7777, 7777},
    {0x1E9EU, 223, 7838, 7838, -34, 223},
    {0x1EA0U, 7841, 7840, 7840, 7841, 7841},
    {0x1EA1U, 7841, 7840, 7840, 7841, 7841},
    {0x1EA2U, 7843, 7842, 7842, 7843, 7843},
    {0x1EA3U, 7843, 7842, 7842, 7843, 7843},
    {0x1EA4U, 7845, 7844, 7844, 7845, 7845},
    {0x1EA5U, 7845, 7844, 7844, 7845, 7845},
    {0x1EA6U, 7847, 7846, 7846, 7847, 7847},
    {0x1EA7U, 7847, 7846, 7846, 7847, 7847},
    {0x1EA8U, 7849, 7848, 7848, 7849, 7849},
    {0x1EA9U, 7849, 7848, 7848, 7849, 7849},
    {0x1EAAU, 7851, 7850, 7850, 7851, 7851},
    {0x1EABU, 7851, 7850, 7850, 7851, 7851},
    {0x1EACU, 7853, 7852, 7852, 7853, 7853},
    {0x1EADU, 7853, 7852, 7852, 7853, 7853},
    {0x1EAEU, 7855, 7854, 7854, 7855, 7855},
    {0x1EAFU, 7855, 7854, 7854, 7855, 7855},
    {0x1EB0U, 7857, 7856, 7856, 7857, 7857},
    {0x1EB1U, 7857, 7856, 7856, 7857, 7857},
    {0x1EB2U, 7859, 7858, 7858, 7859, 7859},
    {0x1EB3U, 7859, 7858, 7858, 7859, 7859},
    {0x1EB4U, 7861, 7860, 7860, 7861, 7861},
    {0x1EB5U, 7861, 7860, 7860, 7861, 7861},
    {0x1EB6U, 7863, 7862, 7862, 7863, 7863},
    {0x1EB7U, 7863, 7862, 7862, 7863, 7863},
    {0x1EB8U, 7865, 7864, 7864, 7865, 7865},
    {0x1EB9U, 7865, 7864, 7864, 7865, 7865},
    {0x1EBAU, 7867, 7866, 7866, 7867, 7867},
    {0x1EBBU, 7867, 7866, 7866, 7867, 7867},
    {0x1EBCU, 7869, 7868, 7868, 7869, 7869},
    {0x1EBDU, 7869, 7868, 7868, 7869, 7869},
    {0x1EBEU, 7871, 7870, 7870, 7871, 7871},
    {0x1EBFU, 7871, 7870, 7870, 7871, 7871},
    {0x1EC0U, 7873, 7872, 7872, 7873, 7873},
    {0x1EC1U, 7873, 7872, 7872, 7873, 7873},
    {0x1EC2U, 7875, 7874, 7874, 7875, 7875},
    {0x1EC3U, 7875, 7874, 7874, 7875, 7875},
    {0x1EC4U, 7877, 7876, 7876, 7877, 7877},
    {0x1EC5U, 7877, 7876, 7876, 7877, 7877},
    {0x1EC6U, 7879, 7878, 7878, 7879, 7879},
    {0x1EC7U, 7879, 7878, 7878, 7879, 7879},
    {0x1EC8U, 7881, 7880, 7880, 7881, 7881},
    {0x1EC9U, 7881, 7880, 7880, 7881, 7881},
    {0x1ECAU, 7883, 7882, 7882, 7883, 7883},
    {0x1ECBU, 7883, 7882, 7882, 7883, 7883},
    {0x1ECCU, 7885, 7884, 7884, 7885, 7885},
    {0x1ECDU, 7885, 7884, 7884, 7885, 7885},
    {0x1ECEU, 7887, 7886, 7886, 7887, 7887},
    {0x1ECFU, 7887, 7886, 7886, 7887, 7887},
    {0x1ED0U, 7889, 7888, 7888, 7889, 7889},
    {0x1ED1U, 7889, 7888, 7888, 7889, 7889},
    {0x1ED2U, 7891, 7890, 7890, 7891, 7891},
    {0x1ED3U, 7891, 7890, 7890, 7891, 7891},
    {0x1ED4U, 7893, 7892, 7892, 7893, 7893},
    {0x1ED5U, 7893, 7892, 7892, 7893, 7893},
    {0x1ED6U, 7895, 7894, 7894, 7895, 7895},
    {0x1ED7U, 7895, 7894, 7894, 7895, 7895},
    {0x1ED8U, 7897, 7896, 7896, 7897, 7897},
    {0x1ED9U, 7897, 7896, 7896, 7897, 7897},
    {0x1EDAU, 7899, 7898, 7898, 7899, 7899},
    {0x1EDBU, 7899, 7898, 7898, 7899, 7899},
    {0x1EDCU, 7901, 7900, 7900, 7901, 7901},
    {0x1EDDU, 7901, 7900, 7900, 7901, 7901},
    {0x1EDEU, 7903, 7902, 7902, 7903, 7903},
    {0x1EDFU, 7903, 7902, 7902, 7903, 7903},
    {0x1EE0U, 7905, 7904, 7904, 7905, 7905},
    {0x1EE1U, 7905, 7904, 7904, 7905, 7905},
    {0x1EE2U, 7907, 7906, 7906, 7907, 7907},
    {0x1EE3U, 7907, 7906, 7906, 7907, 7907},
    {0x1EE4U, 7909, 7908, 7908, 7909, 7909},
    {0x1EE5U, 7909, 7908, 7908, 7909, 7909},
    {0x1EE6U, 7911, 7910, 7910, 7911, 7911},
    {0x1EE7U, 7911, 7910, 7910, 7911, 7911},
    {0x1EE8U, 7913, 7912, 7912, 7913, 7913},
    {0x1EE9U, 7913, 7912, 7912, 7913, 7913},
    {0x1EEAU, 7915, 7914, 7914, 7915, 7915},
    {0x1EEBU, 7915, 7914, 7914, 7915, 7915},
    {0x1EECU, 7917, 7916, 7916, 7917, 7917},
    {0x1EEDU, 7917, 7916, 7916, 7917, 7917},
    {0x1EEEU, 7919, 7918, 7918, 7919, 7919},
    {0x1EEFU, 7919, 7918, 7918, 7919, 7919},
    {0x1EF0U, 7921, 7920, 7920, 7921, 7921},
    {0x1EF1U, 7921, 7920, 7920, 7921, 7921},
    {0x1EF2U, 7923, 7922, 7922, 7923, 7923},
    {0x1EF3U, 7923, 7922, 7922, 7923, 7923},
    {0x1EF4U, 7925, 7924, 7924, 7925, 7925},
    {0x1EF5U, 7925, 7924, 7924, 7925, 7925},
    {0x1EF6U, 7927, 7926, 7926, 7927, 7927},
    {0x1EF7U, 7927, 7926, 7926, 7927, 7927},
    {0x1EF8U, 7929, 7928, 7928, 7929, 7929},
    {0x1EF9U, 7929, 7928, 7928, 7929, 7929},
    {0x1EFAU, 7931, 7930, 7930, 7931, 7931},
    {0x1EFBU, 7931, 7930, 7930, 7931, 7931},
    {0x1EFCU, 7933, 7932, 7932, 7933, 7933},
    {0x1EFDU, 7933, 7932, 7932, 7933, 7933},
    {0x1EFEU, 7935, 7934, 7934, 7935, 7935},
    {0x1EFFU, 7935, 7934, 7934, 7935, 7935},
    {0x1F00U, 7936, 7944, 7944, 7936, 7936},
    {0x1F01U, 7937, 7945, 7945, 7937, 7937},
    {0x1F02U, 7938, 7946, 7946, 7938, 7938},
    {0x1F03U, 7939, 7947, 7947, 7939, 7939},
    {0x1F04U, 7940, 7948, 7948, 7940, 7940},
    {0x1F05U, 7941, 7949, 7949, 7941, 7941},
    {0x1F06U, 7942, 7950, 7950, 7942, 7942},
    {0x1F07U, 7943, 7951, 7951, 7943, 7943},
    {0x1F08U, 7936, 7944, 7944, 7936, 7936},
    {0x1F09U, 7937, 7945, 7945, 7937, 7937},
    {0x1F0AU, 7938, 7946, 7946, 7938, 7938},
    {0x1F0BU, 7939, 7947, 7947, 7939, 7939},
    {0x1F0CU, 7940, 7948, 7948, 7940, 7940},
    {0x1F0DU, 7941, 7949, 7949, 7941, 7941},
    {0x1F0EU, 7942, 7950, 7950, 7942, 7942},
    {0x1F0FU, 7943, 7951, 7951, 7943, 7943},
    {0x1F10U, 7952, 7960, 7960, 7952, 7952},
    {0x1F11U, 7953, 7961, 7961, 7953, 7953},
    {0x1F12U, 7954, 7962, 7962, 7954, 7954},
    {0x1F13U, 7955, 7963, 7963, 7955, 7955},
    {0x1F14U, 7956, 7964, 7964, 7956, 7956},
    {0x1F15U, 7957, 7965, 7965, 7957, 7957},
    {0x1F18U, 7952, 7960, 7960, 7952, 7952},
    {0x1F19U, 7953, 7961, 7961, 7953, 7953},
    {0x1F1AU, 7954, 7962, 7962, 7954, 7954},
    {0x1F1BU, 7955, 7963, 7963, 7955, 7955},
    {0x1F1CU, 7956, 7964, 7964, 7956, 7956},
    {0x1F1DU, 7957, 7965, 7965, 7957, 7957},
    {0x1F20U, 7968, 7976, 7976, 7968, 7968},
    {0x1F21U, 7969, 7977, 7977, 7969, 7969},
    {0x1F22U, 7970, 7978, 7978, 7970, 7970},
    {0x1F23U, 7971, 7979, 7979, 7971, 7971},
    {0x1F24U, 7972, 7980, 7980, 7972, 7972},
    {0x1F25U, 7973, 7981, 7981, 7973, 7973},
    {0x1F26U, 7974, 7982, 7982, 7974, 7974},
    {0x1F27U, 7975, 7983, 7983, 7975, 7975},
    {0x1F28U, 7968, 7976, 7976, 7968, 7968},
    {0x1F29U, 7969, 7977, 7977, 7969, 7969},
    {0x1F2AU, 7970, 7978, 7978, 7970, 7970},
    {0x1F2BU, 7971, 7979, 7979, 7971, 7971},
    {0x1F2CU, 7972, 7980, 7980, 7972, 7972},
    {0x1F2DU, 7973, 7981, 7981, 7973, 7973},
    {0x1F2EU, 7974, 7982, 7982, 7974, 7974},
    {0x1F2FU, 7975, 7983, 7983, 7975, 7975},
    {0x1F30U, 7984, 7992, 7992, 7984, 7984},
    {0x1F31U, 7985, 7993, 7993, 7985, 7985},
    {0x1F32U, 7986, 7994, 7994, 7986, 7986},
    {0x1F33U, 7987, 7995, 7995, 7987, 7987},
    {0x1F34U, 7988, 7996, 7996, 7988, 7988},
    {0x1F35U, 7989, 7997, 7997, 7989, 7989},
    {0x1F36U, 7990, 7998, 7998, 7990, 7990},
    {0x1F37U, 7991, 7999, 7999, 7991, 7991},
    {0x1F38U, 7984, 7992, 7992, 7984, 7984},
    {0x1F39U, 7985, 7993, 7993, 7985, 7985},
    {0x1F3AU, 7986, 7994, 7994, 7986, 7986},
    {0x1F3BU, 7987, 7995, 7995, 7987, 7987},
    {0x1F3CU, 7988, 7996, 7996, 7988, 7988},
    {0x1F3DU, 7989, 7997, 7997, 7989, 7989},
    {0x1F3EU, 7990, 7998, 7998, 7990, 7990},
    {0x1F3FU, 7991, 7999, 7999, 7991, 7991},
    {0x1F40U, 8000, 8008, 8008, 8000, 8000},
    {0x1F41U, 8001, 8009, 8009, 8001, 8001},
    {0x1F42U, 8002, 8010, 8010, 8002, 8002},
    {0x1F43U, 8003, 8011, 8011, 8003, 8003},
    {0x1F44U, 8004, 8012, 8012, 8004, 8004},
    {0x1F45U, 8005, 8013, 8013, 8005, 8005},
    {0x1F48U, 8000, 8008, 8008, 8000, 8000},
    {0x1F49U, 8001, 8009, 8009, 8001, 8001},
    {0x1F4AU, 8002, 8010, 8010, 8002, 8002},
    {0x1F4BU, 8003, 8011, 8011, 8003, 8003},
    {0x1F4CU, 8004, 8012, 8012, 8004, 8004},
    {0x1F4DU, 8005, 8013, 8013, 8005, 8005},
    {0x1F50U, 8016, -434, -434, -450, 8016},
    {0x1F51U, 8017, 8025, 8025, 8017, 8017},
    {0x1F52U, 8018, -467, -467, -491, 8018},
    {0x1F53U, 8019, 8027, 8027, 8019, 8019},
    {0x1F54U, 8020, -515, -515, -539, 8020},
    {0x1F55U, 8021, 8029, 8029, 8021, 8021},
    {0x1F56U, 8022, -563, -563, -587, 8022},
    {0x1F57U, 8023, 8031, 8031, 8023, 8023},
    {0x1F59U, 8017, 8025, 8025, 8017, 8017},
    {0x1F5BU, 8019, 8027, 8027, 8019, 8019},
    {0x1F5DU, 8021, 8029, 8029, 8021, 8021},
    {0x1F5FU, 8023, 8031, 8031, 8023, 8023},
    {0x1F60U, 8032, 8040, 8040, 8032, 8032},
    {0x1F61U, 8033, 8041, 8041, 8033, 8033},
    {0x1F62U, 8034, 8042, 8042, 8034, 8034},
    {0x1F63U, 8035, 8043, 8043, 8035, 8035},
    {0x1F64U, 8036, 8044, 8044, 8036, 8036},
    {0x1F65U, 8037, 8045, 8045, 8037, 8037},
    {0x1F66U, 8038, 8046, 8046, 8038, 8038},
    {0x1F67U, 8039, 8047, 8047, 8039, 8039},
    {0x1F68U, 8032, 8040, 8040, 8032, 8032},
    {0x1F69U, 8033, 8041, 8041, 8033, 8033},
    {0x1F6AU, 8034, 8042, 8042, 8034, 8034},
    {0x1F6BU, 8035, 8043, 8043, 8035, 8035},
    {0x1F6CU, 8036, 8044, 8044, 8036, 8036},
    {0x1F6DU, 8037, 8045, 8045, 8037, 8037},
    {0x1F6EU, 8038, 8046, 8046, 8038, 8038},
    {0x1F6FU, 8039, 8047, 8047, 8039, 8039},
    {0x1F70U, 8048, 8122, 8122, 8048, 8048},
    {0x1F71U, 8049, 8123, 8123, 8049, 8049},
    {0x1F72U, 8050, 8136, 8136, 8050, 8050},
    {0x1F73U, 8051, 8137, 8137, 8051, 8051},
    {0x1F74U, 8052, 8138, 8138, 8052, 8052},
    {0x1F75U, 8053, 8139, 8139, 8053, 8053},
    {0x1F76U, 8054, 8154, 8154, 8054, 8054},
    {0x1F77U, 8055, 8155, 8155, 8055, 8055},
    {0x1F78U, 8056, 8184, 8184, 8056, 8056},
    {0x1F79U, 8057, 8185, 8185, 8057, 8057},
    {0x1F7AU, 8058, 8170, 8170, 8058, 8058},
    {0x1F7BU, 8059, 8171, 8171, 8059, 8059},
    {0x1F7CU, 8060, 8186, 8186, 8060, 8060},
    {0x1F7DU, 8061, 8187, 8187, 8061, 8061},
    {0x1F80U, 8064, -610, 8072, -626, 8064},
    {0x1F81U, 8065, -642, 8073, -658, 8065},
    {0x1F82U, 8066, -674, 8074, -690, 8066},
    {0x1F83U, 8067, -706, 8075, -722, 8067},
    {0x1F84U, 8068, -738, 8076, -754, 8068},
    {0x1F85U, 8069, -770, 8077, -786, 8069},
    {0x1F86U, 8070, -802, 8078, -818, 8070},
    {0x1F87U, 8071, -834, 8079, -850, 8071},
    {0x1F88U, 8064, -610, 8072, -626, 8064},
    {0x1F89U, 8065, -642, 8073, -658, 8065},
    {0x1F8AU, 8066, -674, 8074, -690, 8066},
    {0x1F8BU, 8067, -706, 8075, -722, 8067},
    {0x1F8CU, 8068, -738, 8076, -754, 8068},
    {0x1F8DU, 8069, -770, 8077, -786, 8069},
    {0x1F8EU, 8070, -802, 8078, -818, 8070},
    {0x1F8FU, 8071, -834, 8079, -850, 8071},
    {0x1F90U, 8080, -866, 8088, -882, 8080},
    {0x1F91U, 8081, -898, 8089, -914, 8081},
    {0x1F92U, 8082, -930, 8090, -946, 8082},
    {0x1F93U, 8083, -962, 8091, -978, 8083},
    {0x1F94U, 8084, -994, 8092, -1010, 8084},
    {0x1F95U, 8085, -1026, 8093, -1042, 8085},
    {0x1F96U, 8086, -1058, 8094, -1074, 8086},
    {0x1F97U, 8087, -1090, 8095, -1106, 8087},
    {0x1F98U, 8080, -866, 8088, -882, 8080},
    {0x1F99U, 8081, -898, 8089, -914, 8081},
    {0x1F9AU, 8082, -930, 8090, -946, 8082},
    {0x1F9BU, 8083, -962, 8091, -978, 8083},
    {0x1F9CU, 8084, -994, 8092, -1010, 8084},
    {0x1F9DU, 8085, -1026, 8093, -1042, 8085},
    {0x1F9EU, 8086, -1058, 8094, -1074, 8086},
    {0x1F9FU, 8087, -1090, 8095, -1106, 8087},
    {0x1FA0U, 8096, -1122, 8104, -1138, 8096},
    {0x1FA1U, 8097, -1154, 8105, -1170, 8097},
    {0x1FA2U, 8098, -1186, 8106, -1202, 8098},
    {0x1FA3U, 8099, -1218, 8107, -1234, 8099},
    {0x1FA4U, 8100, -1250, 8108, -1266, 8100},
    {0x1FA5U, 8101, -1282, 8109, -1298, 8101},
    {0x1FA6U, 8102, -1314, 8110, -1330, 8102},
    {0x1FA7U, 8103, -1346, 8111, -1362, 8103},
    {0x1FA8U, 8096, -1122, 8104, -1138, 8096},
    {0x1FA9U, 8097, -1154, 8105, -1170, 8097},
    {0x1FAAU, 8098, -1186, 8106, -1202, 8098},
    {0x1FABU, 8099, -1218, 8107, -1234, 8099},
    {0x1FACU, 8100, -1250, 8108, -1266, 8100},
    {0x1FADU, 8101, -1282, 8109, -1298, 8101},
    {0x1FAEU, 8102, -1314, 8110, -1330, 8102},
    {0x1FAFU, 8103, -1346, 8111, -1362, 8103},
    {0x1FB0U, 8112, 8120, 8120, 8112, 8112},
    {0x1FB1U, 8113, 8121, 8121, 8113, 8113},
    {0x1FB2U, 8114, -1378, -1394, -1410, 8114},
    {0x1FB3U, 8115, -1426, 8124, -1442, 8115},
    {0x1FB4U, 8116, -1458, -1474, -1490, 8116},
    {0x1FB6U, 8118, -1506, -1506, -1522, 8118},
    {0x1FB7U, 8119, -1539, -1563, -1587, 8119},
    {0x1FB8U, 8112, 8120, 8120, 8112, 8112},
    {0x1FB9U, 8113, 8121, 8121, 8113, 8113},
    {0x1FBAU, 8048, 8122, 8122, 8048, 8048},
    {0x1FBBU, 8049, 8123, 8123, 8049, 8049},
    {0x1FBCU, 8115, -1426, 8124, -1442, 8115},
    {0x1FBEU, 8126, 921, 921, 953, 953},
    {0x1FC2U, 8130, -1610, -1626, -1642, 8130},
    {0x1FC3U, 8131, -1658, 8140, -1674, 8131},
    {0x1FC4U, 8132, -1690, -1706, -1722, 8132},
    {0x1FC6U, 8134, -1738, -1738, -1754, 8134},
    {0x1FC7U, 8135, -1771, -1795, -1819, 8135},
    {0x1FC8U, 8050, 8136, 8136, 8050, 8050},
    {0x1FC9U, 8051, 8137, 8137, 8051, 8051},
    {0x1FCAU, 8052, 8138, 8138, 8052, 8052},
    {0x1FCBU, 8053, 8139, 8139, 8053, 8053},
    {0x1FCCU, 8131, -1658, 8140, -1674, 8131},
    {0x1FD0U, 8144, 8152, 8152, 8144, 8144},
    {0x1FD1U, 8145, 8153, 8153, 8145, 8145},
    {0x1FD2U, 8146, -1843, -1843, -1867, 8146},
    {0x1FD3U, 8147, -131, -131, -155, 912},
    {0x1FD6U, 8150, -1890, -1890, -1906, 8150},
    {0x1FD7U, 8151, -1923, -1923, -1947, 8151},
    {0x1FD8U, 8144, 8152, 8152, 8144, 8144},
    {0x1FD9U, 8145, 8153, 8153, 8145, 8145},
    {0x1FDAU, 8054, 8154, 8154, 8054, 8054},
    {0x1FDBU, 8055, 8155, 8155, 8055, 8055},
    {0x1FE0U, 8160, 8168, 8168, 8160, 8160},
    {0x1FE1U, 8161, 8169, 8169, 8161, 8161},
    {0x1FE2U, 8162, -1971, -1971, -1995, 8162},
    {0x1FE3U, 8163, -179, -179, -203, 944},
    {0x1FE4U, 8164, -2018, -2018, -2034, 8164},
    {0x1FE5U, 8165, 8172, 8172, 8165, 8165},
    {0x1FE6U, 8166, -2050, -2050, -2066, 8166},
    {0x1FE7U, 8167, -2083, -2083, -2107, 8167},
    {0x1FE8U, 8160, 8168, 8168, 8160, 8160},
    {0x1FE9U, 8161, 8169, 8169, 8161, 8161},
    {0x1FEAU, 8058, 8170, 8170, 8058, 8058},
    {0x1FEBU, 8059, 8171, 8171, 8059, 8059},
    {0x1FECU, 8165, 8172, 8172, 8165, 8165},
    {0x1FF2U, 8178, -2130, -2146, -2162, 8178},
    {0x1FF3U, 8179, -2178, 8188, -2194, 8179},
    {0x1FF4U, 8180, -2210, -2226, -2242, 8180},
    {0x1FF6U, 8182, -2258, -2258, -2274, 8182},
    {0x1FF7U, 8183, -2291, -2315, -2339, 8183},
    {0x1FF8U, 8056, 8184, 8184, 8056, 8056},
    {0x1FF9U, 8057, 8185, 8185, 8057, 8057},
    {0x1FFAU, 8060, 8186, 8186, 8060, 8060},
    {0x1FFBU, 8061, 8187, 8187, 8061, 8061},
    {0x1FFCU, 8179, -2178, 8188, -2194, 8179},
    {0x2126U, 969, 8486, 8486, 969, 969},
    {0x212AU, 107, 8490, 8490, 107, 107},
    {0x212BU, 229, 8491, 8491, 229, 229},
    {0x2132U, 8526, 8498, 8498, 8526, 8526},
    {0x214EU, 8526, 8498, 8498, 8526, 8526},
    {0x2160U, 8560, 8544, 8544, 8560, 8560},
    {0x2161U, 8561, 8545, 8545, 8561, 8561},
    {0x2162U, 8562, 8546, 8546, 8562, 8562},
    {0x2163U, 8563, 8547, 8547, 8563, 8563},
    {0x2164U, 8564, 8548, 8548, 8564, 8564},
    {0x2165U, 8565, 8549, 8549, 8565, 8565},
    {0x2166U, 8566, 8550, 8550, 8566, 8566},
    {0x2167U, 8567, 8551, 8551, 8567, 8567},
    {0x2168U, 8568, 8552, 8552, 8568, 8568},
    {0x2169U, 8569, 8553, 8553, 8569, 8569},
    {0x216AU, 8570, 8554, 8554, 8570, 8570},
    {0x216BU, 8571, 8555, 8555, 8571, 8571},
    {0x216CU, 8572, 8556, 8556, 8572, 8572},
    {0x216DU, 8573, 8557, 8557, 8573, 8573},
    {0x216EU, 8574, 8558, 8558, 8574, 8574},
    {0x216FU, 8575, 8559, 8559, 8575, 8575},
    {0x2170U, 8560, 8544, 8544, 8560, 8560},
    {0x2171U, 8561, 8545, 8545, 8561, 8561},
    {0x2172U, 8562, 8546, 8546, 8562, 8562},
    {0x2173U, 8563, 8547, 8547, 8563, 8563},
    {0x2174U, 8564, 8548, 8548, 8564, 8564},
    {0x2175U, 8565, 8549, 8549, 8565, 8565},
    {0x2176U, 8566, 8550, 8550, 8566, 8566},
    {0x2177U, 8567, 8551, 8551, 8567, 8567},
    {0x2178U, 8568, 8552, 8552, 8568, 8568},
    {0x2179U, 8569, 8553, 8553, 8569, 8569},
    {0x217AU, 8570, 8554, 8554, 8570, 8570},
    {0x217BU, 8571, 8555, 8555, 8571, 8571},
    {0x217CU, 8572, 8556, 8556, 8572, 8572},
    {0x217DU, 8573, 8557, 8557, 8573, 8573},
    {0x217EU, 8574, 8558, 8558, 8574, 8574},
    {0x217FU, 8575, 8559, 8559, 8575, 8575},
    {0x2183U, 8580, 8579, 8579, 8580, 8580},
    {0x2184U, 8580, 8579, 8579, 8580, 8580},
    {0x24B6U, 9424, 9398, 9398, 9424, 9424},
    {0x24B7U, 9425, 9399, 9399, 9425, 9425},
    {0x24B8U, 9426, 9400, 9400, 9426, 9426},
    {0x24B9U, 9427, 9401, 9401, 9427, 9427},
    {0x24BAU, 9428, 9402, 9402, 9428, 9428},
    {0x24BBU, 9429, 9403, 9403, 9429, 9429},
    {0x24BCU, 9430, 9404, 9404, 9430, 9430},
    {0x24BDU, 9431, 9405, 9405, 9431, 9431},
    {0x24BEU, 9432, 9406, 9406, 9432, 9432},
    {0x24BFU, 9433, 9407, 9407, 9433, 9433},
    {0x24C0U, 9434, 9408, 9408, 9434, 9434},
    {0x24C1U, 9435, 9409, 9409, 9435, 9435},
    {0x24C2U, 9436, 9410, 9410, 9436, 9436},
    {0x24C3U, 9437, 9411, 9411, 9437, 9437},
    {0x24C4U, 9438, 9412, 9412, 9438, 9438},
    {0x24C5U, 9439, 9413, 9413, 9439, 9439},
    {0x24C6U, 9440, 9414, 9414, 9440, 9440},
    {0x24C7U, 9441, 9415, 9415, 9441, 9441},
    {0x24C8U, 9442, 9416, 9416, 9442, 9442},
    {0x24C9U, 9443, 9417, 9417, 9443, 9443},
    {0x24CAU, 9444, 9418, 9418, 9444, 9444},
    {0x24CBU, 9445, 9419, 9419, 9445, 9445},
    {0x24CCU, 9446, 9420, 9420, 9446, 9446},
    {0x24CDU, 9447, 9421, 9421, 9447, 9447},
    {0x24CEU, 9448, 9422, 9422, 9448, 9448},
    {0x24CFU, 9449, 9423, 9423, 9449, 9449},
    {0x24D0U, 9424, 9398, 9398, 9424, 9424},
    {0x24D1U, 9425, 9399, 9399, 9425, 9425},
    {0x24D2U, 9426, 9400, 9400, 9426, 9426},
    {0x24D3U, 9427, 9401, 9401, 9427, 9427},
    {0x24D4U, 9428, 9402, 9402, 9428, 9428},
    {0x24D5U, 9429, 9403, 9403, 9429, 9429},
    {0x24D6U, 9430, 9404, 9404, 9430, 9430},
    {0x24D7U, 9431, 9405, 9405, 9431, 9431},
    {0x24D8U, 9432, 9406, 9406, 9432, 9432},
    {0x24D9U, 9433, 9407, 9407, 9433, 9433},
    {0x24DAU, 9434, 9408, 9408, 9434, 9434},
    {0x24DBU, 9435, 9409, 9409, 9435, 9435},
    {0x24DCU, 9436, 9410, 9410, 9436, 9436},
    {0x24DDU, 9437, 9411, 9411, 9437, 9437},
    {0x24DEU, 9438, 9412, 9412, 9438, 9438},
    {0x24DFU, 9439, 9413, 9413, 9439, 9439},
    {0x24E0U, 9440, 9414, 9414, 9440, 9440},
    {0x24E1U, 9441, 9415, 9415, 9441, 9441},
    {0x24E2U, 9442, 9416, 9416, 9442, 9442},
    {0x24E3U, 9443, 9417, 9417, 9443, 9443},
    {0x24E4U, 9444, 9418, 9418, 9444, 9444},
    {0x24E5U, 9445, 9419, 9419, 9445, 9445},
    {0x24E6U, 9446, 9420, 9420, 9446, 9446},
    {0x24E7U, 9447, 9421, 9421, 9447, 9447},
    {0x24E8U, 9448, 9422, 9422, 9448, 9448},
    {0x24E9U, 9449, 9423, 9423, 9449, 9449},
    {0x2C00U, 11312, 11264, 11264, 11312, 11312},
    {0x2C01U, 11313, 11265, 11265, 11313, 11313},
    {0x2C02U, 11314, 11266, 11266, 11314, 11314},
    {0x2C03U, 11315, 11267, 11267, 11315, 11315},
    {0x2C04U, 11316, 11268, 11268, 11316, 11316},
    {0x2C05U, 11317, 11269, 11269, 11317, 11317},
    {0x2C06U, 11318, 11270, 11270, 11318, 11318},
    {0x2C07U, 11319, 11271, 11271, 11319, 11319},
    {0x2C08U, 11320, 11272, 11272, 11320, 11320},
    {0x2C09U, 11321, 11273, 11273, 11321, 11321},
    {0x2C0AU, 11322, 11274, 11274, 11322, 11322},
    {0x2C0BU, 11323, 11275, 11275, 11323, 11323},
    {0x2C0CU, 11324, 11276, 11276, 11324, 11324},
    {0x2C0DU, 11325, 11277, 11277, 11325, 11325},
    {0x2C0EU, 11326, 11278, 11278, 11326, 11326},
    {0x2C0FU, 11327, 11279, 11279, 11327, 11327},
    {0x2C10U, 11328, 11280, 11280, 11328, 11328},
    {0x2C11U, 11329, 11281, 11281, 11329, 11329},
    {0x2C12U, 11330, 11282, 11282, 11330, 11330},
    {0x2C13U, 11331, 11283, 11283, 11331, 11331},
    {0x2C14U, 11332, 11284, 11284, 11332, 11332},
    {0x2C15U, 11333, 11285, 11285, 11333, 11333},
    {0x2C16U, 11334, 11286, 11286, 11334, 11334},
    {0x2C17U, 11335, 11287, 11287, 11335, 11335},
    {0x2C18U, 11336, 11288, 11288, 11336, 11336},
    {0x2C19U, 11337, 11289, 11289, 11337, 11337},
    {0x2C1AU, 11338, 11290, 11290, 11338, 11338},
    {0x2C1BU, 11339, 11291, 11291, 11339, 11339},
    {0x2C1CU, 11340, 11292, 11292, 11340, 11340},
    {0x2C1DU, 11341, 11293, 11293, 11341, 11341},
    {0x2C1EU, 11342, 11294, 11294, 11342, 11342},
    {0x2C1FU, 11343, 11295, 11295, 11343, 11343},
    {0x2C20U, 11344, 11296, 11296, 11344, 11344},
    {0x2C21U, 11345, 11297, 11297, 11345, 11345},
    {0x2C22U, 11346, 11298, 11298, 11346, 11346},
    {0x2C23U, 11347, 11299, 11299, 11347, 11347},
    {0x2C24U, 11348, 11300, 11300, 11348, 11348},
    {0x2C25U, 11349, 11301, 11301, 11349, 11349},
    {0x2C26U, 11350, 11302, 11302, 11350, 11350},
    {0x2C27U, 11351, 11303, 11303, 11351, 11351},
    {0x2C28U, 11352, 11304, 11304, 11352, 11352},
    {0x2C29U, 11353, 11305, 11305, 11353, 11353},
    {0x2C2AU, 11354, 11306, 11306, 11354, 11354},
    {0x2C2BU, 11355, 11307, 11307, 11355, 11355},
    {0x2C2CU, 11356, 11308, 11308, 11356, 11356},
    {0x2C2DU, 11357, 11309, 11309, 11357, 11357},
    {0x2C2EU, 11358, 11310, 11310, 11358, 11358},
    {0x2C2FU, 11359, 11311, 11311, 11359, 11359},
    {0x2C30U, 11312, 11264, 11264, 11312, 11312},
    {0x2C31U, 11313, 11265, 11265, 11313, 11313},
    {0x2C32U, 11314, 11266, 11266, 11314, 11314},
    {0x2C33U, 11315, 11267, 11267, 11315, 11315},
    {0x2C34U, 11316, 11268, 11268, 11316, 11316},
    {0x2C35U, 11317, 11269, 11269, 11317, 11317},
    {0x2C36U, 11318, 11270, 11270, 11318, 11318},
    {0x2C37U, 11319, 11271, 11271, 11319, 11319},
    {0x2C38U, 11320, 11272, 11272, 11320, 11320},
    {0x2C39U, 11321, 11273, 11273, 11321, 11321},
    {0x2C3AU, 11322, 11274, 11274, 11322, 11322},
    {0x2C3BU, 11323, 11275, 11275, 11323, 11323},
    {0x2C3CU, 11324, 11276, 11276, 11324, 11324},
    {0x2C3DU, 11325, 11277, 11277, 11325, 11325},
    {0x2C3EU, 11326, 11278, 11278, 11326, 11326},
    {0x2C3FU, 11327, 11279, 11279, 11327, 11327},
    {0x2C40U, 11328, 11280, 11280, 11328, 11328},
    {0x2C41U, 11329, 11281, 11281, 11329, 11329},
    {0x2C42U, 11330, 11282, 11282, 11330, 11330},
    {0x2C43U, 11331, 11283, 11283, 11331, 11331},
    {0x2C44U, 11332, 11284, 11284, 11332, 11332},
    {0x2C45U, 11333, 11285, 11285, 11333, 11333},
    {0x2C46U, 11334, 11286, 11286, 11334, 11334},
    {0x2C47U, 11335, 11287, 11287, 11335, 11335},
    {0x2C48U, 11336, 11288, 11288, 11336, 11336},
    {0x2C49U, 11337, 11289, 11289, 11337, 11337},
    {0x2C4AU, 11338, 11290, 11290, 11338, 11338},
    {0x2C4BU, 11339, 11291, 11291, 11339, 11339},
    {0x2C4CU, 11340, 11292, 11292, 11340, 11340},
    {0x2C4DU, 11341, 11293, 11293, 11341, 11341},
    {0x2C4EU, 11342, 11294, 11294, 11342, 11342},
    {0x2C4FU, 11343, 11295, 11295, 11343, 11343},
    {0x2C50U, 11344, 11296, 11296, 11344, 11344},
    {0x2C51U, 11345, 11297, 11297, 11345, 11345},
    {0x2C52U, 11346, 11298, 11298, 11346, 11346},
    {0x2C53U, 11347, 11299, 11299, 11347, 11347},
    {0x2C54U, 11348, 11300, 11300, 11348, 11348},
    {0x2C55U, 11349, 11301, 11301, 11349, 11349},
    {0x2C56U, 11350, 11302, 11302, 11350, 11350},
    {0x2C57U, 11351, 11303, 11303, 11351, 11351},
    {0x2C58U, 11352, 11304, 11304, 11352, 11352},
    {0x2C59U, 11353, 11305, 11305, 11353, 11353},
    {0x2C5AU, 11354, 11306, 11306, 11354, 11354},
    {0x2C5BU, 11355, 11307, 11307, 11355, 11355},
    {0x2C5CU, 11356, 11308, 11308, 11356, 11356},
    {0x2C5DU, 11357, 11309, 11309, 11357, 11357},
    {0x2C5EU, 11358, 11310, 11310, 11358, 11358},
    {0x2C5FU, 11359, 11311, 11311, 11359, 11359},
    {0x2C60U, 11361, 11360, 11360, 11361, 11361},
    {0x2C61U, 11361, 11360, 11360, 11361, 11361},
    {0x2C62U, 619, 11362, 11362, 619, 619},
    {0x2C63U, 7549, 11363, 11363, 7549, 7549},
    {0x2C64U, 637, 11364, 11364, 637, 637},
    {0x2C65U, 11365, 570, 570, 11365, 11365},
    {0x2C66U, 11366, 574, 574, 11366, 11366},
    {0x2C67U, 11368, 11367, 11367, 11368, 11368},
    {0x2C68U, 11368, 11367, 11367, 11368, 11368},
    {0x2C69U, 11370, 11369, 11369, 11370, 11370},
    {0x2C6AU, 11370, 11369, 11369, 11370, 11370},
    {0x2C6BU, 11372, 11371, 11371, 11372, 11372},
    {0x2C6CU, 11372, 11371, 11371, 11372, 11372},
    {0x2C6DU, 593, 11373, 11373, 593, 593},
    {0x2C6EU, 625, 11374, 11374, 625, 625},
    {0x2C6FU, 592, 11375, 11375, 592, 592},
    {0x2C70U, 594, 11376, 11376, 594, 594},
    {0x2C72U, 11379, 11378, 11378, 11379, 11379},
    {0x2C73U, 11379, 11378, 11378, 11379, 11379},
    {0x2C75U, 11382, 11381, 11381, 11382, 11382},
    {0x2C76U, 11382, 11381, 11381, 11382, 11382},
    {0x2C7EU, 575, 11390, 11390, 575, 575},
    {0x2C7FU, 576, 11391, 11391, 576, 576},
    {0x2C80U, 11393, 11392, 11392, 11393, 11393},
    {0x2C81U, 11393, 11392, 11392, 11393, 11393},
    {0x2C82U, 11395, 11394, 11394, 11395, 11395},
    {0x2C83U, 11395, 11394, 11394, 11395, 11395},
    {0x2C84U, 11397, 11396, 11396, 11397, 11397},
    {0x2C85U, 11397, 11396, 11396, 11397, 11397},
    {0x2C86U, 11399, 11398, 11398, 11399, 11399},
    {0x2C87U, 11399, 11398, 11398, 11399, 11399},
    {0x2C88U, 11401, 11400, 11400, 11401, 11401},
    {0x2C89U, 11401, 11400, 11400, 11401, 11401},
    {0x2C8AU, 11403, 11402, 11402, 11403, 11403},
    {0x2C8BU, 11403, 11402, 11402, 11403, 11403},
    {0x2C8CU, 11405, 11404, 11404, 11405, 11405},
    {0x2C8DU, 11405, 11404, 11404, 11405, 11405},
    {0x2C8EU, 11407, 11406, 11406, 11407, 11407},
    {0x2C8FU, 11407, 11406, 11406, 11407, 11407},
    {0x2C90U, 11409, 11408, 11408, 11409, 11409},
    {0x2C91U, 11409, 11408, 11408, 11409, 11409},
    {0x2C92U, 11411, 11410, 11410, 11411, 11411},
    {0x2C93U, 11411, 11410, 11410, 11411, 11411},
    {0x2C94U, 11413, 11412, 11412, 11413, 11413},
    {0x2C95U, 11413, 11412, 11412, 11413, 11413},
    {0x2C96U, 11415, 11414, 11414, 11415, 11415},
    {0x2C97U, 11415, 11414, 11414, 11415, 11415},
    {0x2C98U, 11417, 11416, 11416, 11417, 11417},
    {0x2C99U, 11417, 11416, 11416, 11417, 11417},
    {0x2C9AU, 11419, 11418, 11418, 11419, 11419},
    {0x2C9BU, 11419, 11418, 11418, 11419, 11419},
    {0x2C9CU, 11421, 11420, 11420, 11421, 11421},
    {0x2C9DU, 11421, 11420, 11420, 11421, 11421},
    {0x2C9EU, 11423, 11422, 11422, 11423, 11423},
    {0x2C9FU, 11423, 11422, 11422, 11423, 11423},
    {0x2CA0U, 11425, 11424, 11424, 11425, 11425},
    {0x2CA1U, 11425, 11424, 11424, 11425, 11425},
    {0x2CA2U, 11427, 11426, 11426, 11427, 11427},
    {0x2CA3U, 11427, 11426, 11426, 11427, 11427},
    {0x2CA4U, 11429, 11428, 11428, 11429, 11429},
    {0x2CA5U, 11429, 11428, 11428, 11429, 11429},
    {0x2CA6U, 11431, 11430, 11430, 11431, 11431},
    {0x2CA7U, 11431, 11430, 11430, 11431, 11431},
    {0x2CA8U, 11433, 11432, 11432, 11433, 11433},
    {0x2CA9U, 11433, 11432, 11432, 11433, 11433},
    {0x2CAAU, 11435, 11434, 11434, 11435, 11435},
    {0x2CABU, 11435, 11434, 11434, 11435, 11435},
    {0x2CACU, 11437, 11436, 11436, 11437, 11437},
    {0x2CADU, 11437, 11436, 11436, 11437, 11437},
    {0x2CAEU, 11439, 11438, 11438, 11439, 11439},
    {0x2CAFU, 11439, 11438, 11438, 11439, 11439},
    {0x2CB0U, 11441, 11440, 11440, 11441, 11441},
    {0x2CB1U, 11441, 11440, 11440, 11441, 11441},
    {0x2CB2U, 11443, 11442, 11442, 11443, 11443},
    {0x2CB3U, 11443, 11442, 11442, 11443, 11443},
    {0x2CB4U, 11445, 11444, 11444, 11445, 11445},
    {0x2CB5U, 11445, 11444, 11444, 11445, 11445},
    {0x2CB6U, 11447, 11446, 11446, 11447, 11447},
    {0x2CB7U, 11447, 11446, 11446, 11447, 11447},
    {0x2CB8U, 11449, 11448, 11448, 11449, 11449},
    {0x2CB9U, 11449, 11448, 11448, 11449, 11449},
    {0x2CBAU, 11451, 11450, 11450, 11451, 11451},
    {0x2CBBU, 11451, 11450, 11450, 11451, 11451},
    {0x2CBCU, 11453, 11452, 11452, 11453, 11453},
    {0x2CBDU, 11453, 11452, 11452, 11453, 11453},
    {0x2CBEU, 11455, 11454, 11454, 11455, 11455},
    {0x2CBFU, 11455, 11454, 11454, 11455, 11455},
    {0x2CC0U, 11457, 11456, 11456, 11457, 11457},
    {0x2CC1U, 11457, 11456, 11456, 11457, 11457},
    {0x2CC2U, 11459, 11458, 11458, 11459, 11459},
    {0x2CC3U, 11459, 11458, 11458, 11459, 11459},
    {0x2CC4U, 11461, 11460, 11460, 11461, 11461},
    {0x2CC5U, 11461, 11460, 11460, 11461, 11461},
    {0x2CC6U, 11463, 11462, 11462, 11463, 11463},
    {0x2CC7U, 11463, 11462, 11462, 11463, 11463},
    {0x2CC8U, 11465, 11464, 11464, 11465, 11465},
    {0x2CC9U, 11465, 11464, 11464, 11465, 11465},
    {0x2CCAU, 11467, 11466, 11466, 11467, 11467},
    {0x2CCBU, 11467, 11466, 11466, 11467, 11467},
    {0x2CCCU, 11469, 11468, 11468, 11469, 11469},
    {0x2CCDU, 11469, 11468, 11468, 11469, 11469},
    {0x2CCEU, 11471, 11470, 11470, 11471, 11471},
    {0x2CCFU, 11471, 11470, 11470, 11471, 11471},
    {0x2CD0U, 11473, 11472, 11472, 11473, 11473},
    {0x2CD1U, 11473, 11472, 11472, 11473, 11473},
    {0x2CD2U, 11475, 11474, 11474, 11475, 11475},
    {0x2CD3U, 11475, 11474, 11474, 11475, 11475},
    {0x2CD4U, 11477, 11476, 11476, 11477, 11477},
    {0x2CD5U, 11477, 11476, 11476, 11477, 11477},
    {0x2CD6U, 11479, 11478, 11478, 11479, 11479},
    {0x2CD7U, 11479, 11478, 11478, 11479, 11479},
    {0x2CD8U, 11481, 11480, 11480, 11481, 11481},
    {0x2CD9U, 11481, 11480, 11480, 11481, 11481},
    {0x2CDAU, 11483, 11482, 11482, 11483, 11483},
    {0x2CDBU, 11483, 11482, 11482, 11483, 11483},
    {0x2CDCU, 11485, 11484, 11484, 11485, 11485},
    {0x2CDDU, 11485, 11484, 11484, 11485, 11485},
    {0x2CDEU, 11487, 11486, 11486, 11487, 11487},
    {0x2CDFU, 11487, 11486, 11486, 11487, 11487},
    {0x2CE0U, 11489, 11488, 11488, 11489, 11489},
    {0x2CE1U, 11489, 11488, 11488, 11489, 11489},
    {0x2CE2U, 11491, 11490, 11490, 11491, 11491},
    {0x2CE3U, 11491, 11490, 11490, 11491, 11491},
    {0x2CEBU, 11500, 11499, 11499, 11500, 11500},
    {0x2CECU, 11500, 11499, 11499, 11500, 11500},
    {0x2CEDU, 11502, 11501, 11501, 11502, 11502},
    {0x2CEEU, 11502, 11501, 11501, 11502, 11502},
    {0x2CF2U, 11507, 11506, 11506, 11507, 11507},
    {0x2CF3U, 11507, 11506, 11506, 11507, 11507},
    {0x2D00U, 11520, 4256, 4256, 11520, 11520},
    {0x2D01U, 11521, 4257, 4257, 11521, 11521},
    {0x2D02U, 11522, 4258, 4258, 11522, 11522},
    {0x2D03U, 11523, 4259, 4259, 11523, 11523},
    {0x2D04U, 11524, 4260, 4260, 11524, 11524},
    {0x2D05U, 11525, 4261, 4261, 11525, 11525},
    {0x2D06U, 11526, 4262, 4262, 11526, 11526},
    {0x2D07U, 11527, 4263, 4263, 11527, 11527},
    {0x2D08U, 11528, 4264, 4264, 11528, 11528},
    {0x2D09U, 11529, 4265, 4265, 11529, 11529},
    {0x2D0AU, 11530, 4266, 4266, 11530, 11530},
    {0x2D0BU, 11531, 4267, 4267, 11531, 11531},
    {0x2D0CU, 11532, 4268, 4268, 11532, 11532},
    {0x2D0DU, 11533, 4269, 4269, 11533, 11533},
    {0x2D0EU, 11534, 4270, 4270, 11534, 11534},
    {0x2D0FU, 11535, 4271, 4271, 11535, 11535},
    {0x2D10U, 11536, 4272, 4272, 11536, 11536},
    {0x2D11U, 11537, 4273, 4273, 11537, 11537},
    {0x2D12U, 11538, 4274, 4274, 11538, 11538},
    {0x2D13U, 11539, 4275, 4275, 11539, 11539},
    {0x2D14U, 11540, 4276, 4276, 11540, 11540},
    {0x2D15U, 11541, 4277, 4277, 11541, 11541},
    {0x2D16U, 11542, 4278, 4278, 11542, 11542},
    {0x2D17U, 11543, 4279, 4279, 11543, 11543},
    {0x2D18U, 11544, 4280, 4280, 11544, 11544},
    {0x2D19U, 11545, 4281, 4281, 11545, 11545},
    {0x2D1AU, 11546, 4282, 4282, 11546, 11546},
    {0x2D1BU, 11547, 4283, 4283, 11547, 11547},
    {0x2D1CU, 11548, 4284, 4284, 11548, 11548},
    {0x2D1DU, 11549, 4285, 4285, 11549, 11549},
    {0x2D1EU, 11550, 4286, 4286, 11550, 11550},
    {0x2D1FU, 11551, 4287, 4287, 11551, 11551},
    {0x2D20U, 11552, 4288, 4288, 11552, 11552},
    {0x2D21U, 11553, 4289, 4289, 11553, 11553},
    {0x2D22U, 11554, 4290, 4290, 11554, 11554},
    {0x2D23U, 11555, 4291, 4291, 11555, 11555},
    {0x2D24U, 11556, 4292, 4292, 11556, 11556},
    {0x2D25U, 11557, 4293, 4293, 11557, 11557},
    {0x2D27U, 11559, 4295, 4295, 11559, 11559},
    {0x2D2DU, 11565, 4301, 4301, 11565, 11565},
    {0xA640U, 42561, 42560, 42560, 42561, 42561},
    {0xA641U, 42561, 42560, 42560, 42561, 42561},
    {0xA642U, 42563, 42562, 42562, 42563, 42563},
    {0xA643U, 42563, 42562, 42562, 42563, 42563},
    {0xA644U, 42565, 42564, 42564, 42565, 42565},
    {0xA645U, 42565, 42564, 42564, 42565, 42565},
    {0xA646U, 42567, 42566, 42566, 42567, 42567},
    {0xA647U, 42567, 42566, 42566, 42567, 42567},
    {0xA648U, 42569, 42568, 42568, 42569, 42569},
    {0xA649U, 42569, 42568, 42568, 42569, 42569},
    {0xA64AU, 42571, 42570, 42570, 42571, 42571},
    {0xA64BU, 42571, 42570, 42570, 42571, 42571},
    {0xA64CU, 42573, 42572, 42572, 42573, 42573},
    {0xA64DU, 42573, 42572, 42572, 42573, 42573},
    {0xA64EU, 42575, 42574, 42574, 42575, 42575},
    {0xA64FU, 42575, 42574, 42574, 42575, 42575},
    {0xA650U, 42577, 42576, 42576, 42577, 42577},
    {0xA651U, 42577, 42576, 42576, 42577, 42577},
    {0xA652U, 42579, 42578, 42578, 42579, 42579},
    {0xA653U, 42579, 42578, 42578, 42579, 42579},
    {0xA654U, 42581, 42580, 42580, 42581, 42581},
    {0xA655U, 42581, 42580, 42580, 42581, 42581},
    {0xA656U, 42583, 42582, 42582, 42583, 42583},
    {0xA657U, 42583, 42582, 42582, 42583, 42583},
    {0xA658U, 42585, 42584, 42584, 42585, 42585},
    {0xA659U, 42585, 42584, 42584, 42585, 42585},
    {0xA65AU, 42587, 42586, 42586, 42587, 42587},
    {0xA65BU, 42587, 42586, 42586, 42587, 42587},
    {0xA65CU, 42589, 42588, 42588, 42589, 42589},
    {0xA65DU, 42589, 42588, 42588, 42589, 42589},
    {0xA65EU, 42591, 42590, 42590, 42591, 42591},
    {0xA65FU, 42591, 42590, 42590, 42591, 42591},
    {0xA660U, 42593, 42592, 42592, 42593, 42593},
    {0xA661U, 42593, 42592, 42592, 42593, 42593},
    {0xA662U, 42595, 42594, 42594, 42595, 42595},
    {0xA663U, 42595, 42594, 42594, 42595, 42595},
    {0xA664U, 42597, 42596, 42596, 42597, 42597},
    {0xA665U, 42597, 42596, 42596, 42597, 42597},
    {0xA666U, 42599, 42598, 42598, 42599, 42599},
    {0xA667U, 42599, 42598, 42598, 42599, 42599},
    {0xA668U, 42601, 42600, 42600, 42601, 42601},
    {0xA669U, 42601, 42600, 42600, 42601, 42601},
    {0xA66AU, 42603, 42602, 42602, 42603, 42603},
    {0xA66BU, 42603, 42602, 42602, 42603, 42603},
    {0xA66CU, 42605, 42604, 42604, 42605, 42605},
    {0xA66DU, 42605, 42604, 42604, 42605, 42605},
    {0xA680U, 42625, 42624, 42624, 42625, 42625},
    {0xA681U, 42625, 42624, 42624, 42625, 42625},
    {0xA682U, 42627, 42626, 42626, 42627, 42627},
    {0xA683U, 42627, 42626, 42626, 42627, 42627},
    {0xA684U, 42629, 42628, 42628, 42629, 42629},
    {0xA685U, 42629, 42628, 42628, 42629, 42629},
    {0xA686U, 42631, 42630, 42630, 42631, 42631},
    {0xA687U, 42631, 42630, 42630, 42631, 42631},
    {0xA688U, 42633, 42632, 42632, 42633, 42633},
    {0xA689U, 42633, 42632, 42632, 42633, 42633},
    {0xA68AU, 42635, 42634, 42634, 42635, 42635},
    {0xA68BU, 42635, 42634, 42634, 42635, 42635},
    {0xA68CU, 42637, 42636, 42636, 42637, 42637},
    {0xA68DU, 42637, 42636, 42636, 42637, 42637},
    {0xA68EU, 42639, 42638, 42638, 42639, 42639},
    {0xA68FU, 42639, 42638, 42638, 42639, 42639},
    {0xA690U, 42641, 42640, 42640, 42641, 42641},
    {0xA691U, 42641, 42640, 42640, 42641, 42641},
    {0xA692U, 42643, 42642, 42642, 42643, 42643},
    {0xA693U, 42643, 42642, 42642, 42643, 42643},
    {0xA694U, 42645, 42644, 42644, 42645, 42645},
    {0xA695U, 42645, 42644, 42644, 42645, 42645},
    {0xA696U, 42647, 42646, 42646, 42647, 42647},
    {0xA697U, 42647, 42646, 42646, 42647, 42647},
    {0xA698U, 42649, 42648, 42648, 42649, 42649},
    {0xA699U, 42649, 42648, 42648, 42649, 42649},
    {0xA69AU, 42651, 42650, 42650, 42651, 42651},
    {0xA69BU, 42651, 42650, 42650, 42651, 42651},
    {0xA722U, 42787, 42786, 42786, 42787, 42787},
    {0xA723U, 42787, 42786, 42786, 42787, 42787},
    {0xA724U, 42789, 42788, 42788, 42789, 42789},
    {0xA725U, 42789, 42788, 42788, 42789, 42789},
    {0xA726U, 42791, 42790, 42790, 42791, 42791},
    {0xA727U, 42791, 42790, 42790, 42791, 42791},
    {0xA728U, 42793, 42792, 42792, 42793, 42793},
    {0xA729U, 42793, 42792, 42792, 42793, 42793},
    {0xA72AU, 42795, 42794, 42794, 42795, 42795},
    {0xA72BU, 42795, 42794, 42794, 42795, 42795},
    {0xA72CU, 42797, 42796, 42796, 42797, 42797},
    {0xA72DU, 42797, 42796, 42796, 42797, 42797},
    {0xA72EU, 42799, 42798, 42798, 42799, 42799},
    {0xA72FU, 42799, 42798, 42798, 42799, 42799},
    {0xA732U, 42803, 42802, 42802, 42803, 42803},
    {0xA733U, 42803, 42802, 42802, 42803, 42803},
    {0xA734U, 42805, 42804, 42804, 42805, 42805},
    {0xA735U, 42805, 42804, 42804, 42805, 42805},
    {0xA736U, 42807, 42806, 42806, 42807, 42807},
    {0xA737U, 42807, 42806, 42806, 42807, 42807},
    {0xA738U, 42809, 42808, 42808, 42809, 42809},
    {0xA739U, 42809, 42808, 42808, 42809, 42809},
    {0xA73AU, 42811, 42810, 42810, 42811, 42811},
    {0xA73BU, 42811, 42810, 42810, 42811, 42811},
    {0xA73CU, 42813, 42812, 42812, 42813, 42813},
    {0xA73DU, 42813, 42812, 42812, 42813, 42813},
    {0xA73EU, 42815, 42814, 42814, 42815, 42815},
    {0xA73FU, 42815, 42814, 42814, 42815, 42815},
    {0xA740U, 42817, 42816, 42816, 42817, 42817},
    {0xA741U, 42817, 42816, 42816, 42817, 42817},
    {0xA742U, 42819, 42818, 42818, 42819, 42819},
    {0xA743U, 42819, 42818, 42818, 42819, 42819},
    {0xA744U, 42821, 42820, 42820, 42821, 42821},
    {0xA745U, 42821, 42820, 42820, 42821, 42821},
    {0xA746U, 42823, 42822, 42822, 42823, 42823},
    {0xA747U, 42823, 42822, 42822, 42823, 42823},
    {0xA748U, 42825, 42824, 42824, 42825, 42825},
    {0xA749U, 42825, 42824, 42824, 42825, 42825},
    {0xA74AU, 42827, 42826, 42826, 42827, 42827},
    {0xA74BU, 42827, 42826, 42826, 42827, 42827},
    {0xA74CU, 42829, 42828, 42828, 42829, 42829},
    {0xA74DU, 42829, 42828, 42828, 42829, 42829},
    {0xA74EU, 42831, 42830, 42830, 42831, 42831},
    {0xA74FU, 42831, 42830, 42830, 42831, 42831},
    {0xA750U, 42833, 42832, 42832, 42833, 42833},
    {0xA751U, 42833, 42832, 42832, 42833, 42833},
    {0xA752U, 42835, 42834, 42834, 42835, 42835},
    {0xA753U, 42835, 42834, 42834, 42835, 42835},
    {0xA754U, 42837, 42836, 42836, 42837, 42837},
    {0xA755U, 42837, 42836, 42836, 42837, 42837},
    {0xA756U, 42839, 42838, 42838, 42839, 42839},
    {0xA757U, 42839, 42838, 42838, 42839, 42839},
    {0xA758U, 42841, 42840, 42840, 42841, 42841},
    {0xA759U, 42841, 42840, 42840, 42841, 42841},
    {0xA75AU, 42843, 42842, 42842, 42843, 42843},
    {0xA75BU, 42843, 42842, 42842, 42843, 42843},
    {0xA75CU, 42845, 42844, 42844, 42845, 42845},
    {0xA75DU, 42845, 42844, 42844, 42845, 42845},
    {0xA75EU, 42847, 42846, 42846, 42847, 42847},
    {0xA75FU, 42847, 42846, 42846, 42847, 42847},
    {0xA760U, 42849, 42848, 42848, 42849, 42849},
    {0xA761U, 42849, 42848, 42848, 42849, 42849},
    {0xA762U, 42851, 42850, 42850, 42851, 42851},
    {0xA763U, 42851, 42850, 42850, 42851, 42851},
    {0xA764U, 42853, 42852, 42852, 42853, 42853},
    {0xA765U, 42853, 42852, 42852, 42853, 42853},
    {0xA766U, 42855, 42854, 42854, 42855, 42855},
    {0xA767U, 42855, 42854, 42854, 42855, 42855},
    {0xA768U, 42857, 42856, 42856, 42857, 42857},
    {0xA769U, 42857, 42856, 42856, 42857, 42857},
    {0xA76AU, 42859, 42858, 42858, 42859, 42859},
    {0xA76BU, 42859, 42858, 42858, 42859, 42859},
    {0xA76CU, 42861, 42860, 42860, 42861, 42861},
    {0xA76DU, 42861, 42860, 42860, 42861, 42861},
    {0xA76EU, 42863, 42862, 42862, 42863, 42863},
    {0xA76FU, 42863, 42862, 42862, 42863, 42863},
    {0xA779U, 42874, 42873, 42873, 42874, 42874},
    {0xA77AU, 42874, 42873, 42873, 42874, 42874},
    {0xA77BU, 42876, 42875, 42875, 42876, 42876},
    {0xA77CU, 42876, 42875, 42875, 42876, 42876},
    {0xA77DU, 7545, 42877, 42877, 7545, 7545},
    {0xA77EU, 42879, 42878, 42878, 42879, 42879},
    {0xA77FU, 42879, 42878, 42878, 42879, 42879},
    {0xA780U, 42881, 42880, 42880, 42881, 42881},
    {0xA781U, 42881, 42880, 42880, 42881, 42881},
    {0xA782U, 42883, 42882, 42882, 42883, 42883},
    {0xA783U, 42883, 42882, 42882, 42883, 42883},
    {0xA784U, 42885, 42884, 42884, 42885, 42885},
    {0xA785U, 42885, 42884, 42884, 42885, 42885},
    {0xA786U, 42887, 42886, 42886, 42887, 42887},
    {0xA787U, 42887, 42886, 42886, 42887, 42887},
    {0xA78BU, 42892, 42891, 42891, 42892, 42892},
    {0xA78CU, 42892, 42891, 42891, 42892, 42892},
    {0xA78DU, 613, 42893, 42893, 613, 613},
    {0xA790U, 42897, 42896, 42896, 42897, 42897},
    {0xA791U, 42897, 42896, 42896, 42897, 42897},
    {0xA792U, 42899, 42898, 42898, 42899, 42899},
    {0xA793U, 42899, 42898, 42898, 42899, 42899},
    {0xA794U, 42900, 42948, 42948, 42900, 42900},
    {0xA796U, 42903, 42902, 42902, 42903, 42903},
    {0xA797U, 42903, 42902, 42902, 42903, 42903},
    {0xA798U, 42905, 42904, 42904, 42905, 42905},
    {0xA799U, 42905, 42904, 42904, 42905, 42905},
    {0xA79AU, 42907, 42906, 42906, 42907, 42907},
    {0xA79BU, 42907, 42906, 42906, 42907, 42907},
    {0xA79CU, 42909, 42908, 42908, 42909, 42909},
    {0xA79DU, 42909, 42908, 42908, 42909, 42909},
    {0xA79EU, 42911, 42910, 42910, 42911, 42911},
    {0xA79FU, 42911, 42910, 42910, 42911, 42911},
    {0xA7A0U, 42913, 42912, 42912, 42913, 42913},
    {0xA7A1U, 42913, 42912, 42912, 42913, 42913},
    {0xA7A2U, 42915, 42914, 42914, 42915, 42915},
    {0xA7A3U, 42915, 42914, 42914, 42915, 42915},
    {0xA7A4U, 42917, 42916, 42916, 42917, 42917},
    {0xA7A5U, 42917, 42916, 42916, 42917, 42917},
    {0xA7A6U, 42919, 42918, 42918, 42919, 42919},
    {0xA7A7U, 42919, 42918, 42918, 42919, 42919},
    {0xA7A8U, 42921, 42920, 42920, 42921, 42921},
    {0xA7A9U, 42921, 42920, 42920, 42921, 42921},
    {0xA7AAU, 614, 42922, 42922, 614, 614},
    {0xA7ABU, 604, 42923, 42923, 604, 604},
    {0xA7ACU, 609, 42924, 42924, 609, 609},
    {0xA7ADU, 620, 42925, 42925, 620, 620},
    {0xA7AEU, 618, 42926, 42926, 618, 618},
    {0xA7B0U, 670, 42928, 42928, 670, 670},
    {0xA7B1U, 647, 42929, 42929, 647, 647},
    {0xA7B2U, 669, 42930, 42930, 669, 669},
    {0xA7B3U, 43859, 42931, 42931, 43859, 43859},
    {0xA7B4U, 42933, 42932, 42932, 42933, 42933},
    {0xA7B5U, 42933, 42932, 42932, 42933, 42933},
    {0xA7B6U, 42935, 42934, 42934, 42935, 42935},
    {0xA7B7U, 42935, 42934, 42934, 42935, 42935},
    {0xA7B8U, 42937, 42936, 42936, 42937, 42937},
    {0xA7B9U, 42937, 42936, 42936, 42937, 42937},
    {0xA7BAU, 42939, 42938, 42938, 42939, 42939},
    {0xA7BBU, 42939, 42938, 42938, 42939, 42939},
    {0xA7BCU, 42941, 42940, 42940, 42941, 42941},
    {0xA7BDU, 42941, 42940, 42940, 42941, 42941},
    {0xA7BEU, 42943, 42942, 42942, 42943, 42943},
    {0xA7BFU, 42943, 42942, 42942, 42943, 42943},
    {0xA7C0U, 42945, 42944, 42944, 42945, 42945},
    {0xA7C1U, 42945, 42944, 42944, 42945, 42945},
    {0xA7C2U, 42947, 42946, 42946, 42947, 42947},
    {0xA7C3U, 42947, 42946, 42946, 42947, 42947},
    {0xA7C4U, 42900, 42948, 42948, 42900, 42900},
    {0xA7C5U, 642, 42949, 42949, 642, 642},
    {0xA7C6U, 7566, 42950, 42950, 7566, 7566},
    {0xA7C7U, 42952, 42951, 42951, 42952, 42952},
    {0xA7C8U, 42952, 42951, 42951, 42952, 42952},
    {0xA7C9U, 42954, 42953, 42953, 42954, 42954},
    {0xA7CAU, 42954, 42953, 42953, 42954, 42954},
    {0xA7CBU, 612, 42955, 42955, 612, 612},
    {0xA7CCU, 42957, 42956, 42956, 42957, 42957},
    {0xA7CDU, 42957, 42956, 42956, 42957, 42957},
    {0xA7CEU, 42959, 42958, 42958, 42959, 42959},
    {0xA7CFU, 42959, 42958, 42958, 42959, 42959},
    {0xA7D0U, 42961, 42960, 42960, 42961, 42961},
    {0xA7D1U, 42961, 42960, 42960, 42961, 42961},
    {0xA7D2U, 42963, 42962, 42962, 42963, 42963},
    {0xA7D3U, 42963, 42962, 42962, 42963, 42963},
    {0xA7D4U, 42965, 42964, 42964, 42965, 42965},
    {0xA7D5U, 42965, 42964, 42964, 42965, 42965},
    {0xA7D6U, 42967, 42966, 42966, 42967, 42967},
    {0xA7D7U, 42967, 42966, 42966, 42967, 42967},
    {0xA7D8U, 42969, 42968, 42968, 42969, 42969},
    {0xA7D9U, 42969, 42968, 42968, 42969, 42969},
    {0xA7DAU, 42971, 42970, 42970, 42971, 42971},
    {0xA7DBU, 42971, 42970, 42970, 42971, 42971},
    {0xA7DCU, 411, 42972, 42972, 411, 411},
    {0xA7F5U, 42998, 42997, 42997, 42998, 42998},
    {0xA7F6U, 42998, 42997, 42997, 42998, 42998},
    {0xAB53U, 43859, 42931, 42931, 43859, 43859},
    {0xAB70U, 43888, 5024, 5024, 5024, 5024},
    {0xAB71U, 43889, 5025, 5025, 5025, 5025},
    {0xAB72U, 43890, 5026, 5026, 5026, 5026},
    {0xAB73U, 43891, 5027, 5027, 5027, 5027},
    {0xAB74U, 43892, 5028, 5028, 5028, 5028},
    {0xAB75U, 43893, 5029, 5029, 5029, 5029},
    {0xAB76U, 43894, 5030, 5030, 5030, 5030},
    {0xAB77U, 43895, 5031, 5031, 5031, 5031},
    {0xAB78U, 43896, 5032, 5032, 5032, 5032},
    {0xAB79U, 43897, 5033, 5033, 5033, 5033},
    {0xAB7AU, 43898, 5034, 5034, 5034, 5034},
    {0xAB7BU, 43899, 5035, 5035, 5035, 5035},
    {0xAB7CU, 43900, 5036, 5036, 5036, 5036},
    {0xAB7DU, 43901, 5037, 5037, 5037, 5037},
    {0xAB7EU, 43902, 5038, 5038, 5038, 5038},
    {0xAB7FU, 43903, 5039, 5039, 5039, 5039},
    {0xAB80U, 43904, 5040, 5040, 5040, 5040},
    {0xAB81U, 43905, 5041, 5041, 5041, 5041},
    {0xAB82U, 43906, 5042, 5042, 5042, 5042},
    {0xAB83U, 43907, 5043, 5043, 5043, 5043},
    {0xAB84U, 43908, 5044, 5044, 5044, 5044},
    {0xAB85U, 43909, 5045, 5045, 5045, 5045},
    {0xAB86U, 43910, 5046, 5046, 5046, 5046},
    {0xAB87U, 43911, 5047, 5047, 5047, 5047},
    {0xAB88U, 43912, 5048, 5048, 5048, 5048},
    {0xAB89U, 43913, 5049, 5049, 5049, 5049},
    {0xAB8AU, 43914, 5050, 5050, 5050, 5050},
    {0xAB8BU, 43915, 5051, 5051, 5051, 5051},
    {0xAB8CU, 43916, 5052, 5052, 5052, 5052},
    {0xAB8DU, 43917, 5053, 5053, 5053, 5053},
    {0xAB8EU, 43918, 5054, 5054, 5054, 5054},
    {0xAB8FU, 43919, 5055, 5055, 5055, 5055},
    {0xAB90U, 43920, 5056, 5056, 5056, 5056},
    {0xAB91U, 43921, 5057, 5057, 5057, 5057},
    {0xAB92U, 43922, 5058, 5058, 5058, 5058},
    {0xAB93U, 43923, 5059, 5059, 5059, 5059},
    {0xAB94U, 43924, 5060, 5060, 5060, 5060},
    {0xAB95U, 43925, 5061, 5061, 5061, 5061},
    {0xAB96U, 43926, 5062, 5062, 5062, 5062},
    {0xAB97U, 43927, 5063, 5063, 5063, 5063},
    {0xAB98U, 43928, 5064, 5064, 5064, 5064},
    {0xAB99U, 43929, 5065, 5065, 5065, 5065},
    {0xAB9AU, 43930, 5066, 5066, 5066, 5066},
    {0xAB9BU, 43931, 5067, 5067, 5067, 5067},
    {0xAB9CU, 43932, 5068, 5068, 5068, 5068},
    {0xAB9DU, 43933, 5069, 5069, 5069, 5069},
    {0xAB9EU, 43934, 5070, 5070, 5070, 5070},
    {0xAB9FU, 43935, 5071, 5071, 5071, 5071},
    {0xABA0U, 43936, 5072, 5072, 5072, 5072},
    {0xABA1U, 43937, 5073, 5073, 5073, 5073},
    {0xABA2U, 43938, 5074, 5074, 5074, 5074},
    {0xABA3U, 43939, 5075, 5075, 5075, 5075},
    {0xABA4U, 43940, 5076, 5076, 5076, 5076},
    {0xABA5U, 43941, 5077, 5077, 5077, 5077},
    {0xABA6U, 43942, 5078, 5078, 5078, 5078},
    {0xABA7U, 43943, 5079, 5079, 5079, 5079},
    {0xABA8U, 43944, 5080, 5080, 5080, 5080},
    {0xABA9U, 43945, 5081, 5081, 5081, 5081},
    {0xABAAU, 43946, 5082, 5082, 5082, 5082},
    {0xABABU, 43947, 5083, 5083, 5083, 5083},
    {0xABACU, 43948, 5084, 5084, 5084, 5084},
    {0xABADU, 43949, 5085, 5085, 5085, 5085},
    {0xABAEU, 43950, 5086, 5086, 5086, 5086},
    {0xABAFU, 43951, 5087, 5087, 5087, 5087},
    {0xABB0U, 43952, 5088, 5088, 5088, 5088},
    {0xABB1U, 43953, 5089, 5089, 5089, 5089},
    {0xABB2U, 43954, 5090, 5090, 5090, 5090},
    {0xABB3U, 43955, 5091, 5091, 5091, 5091},
    {0xABB4U, 43956, 5092, 5092, 5092, 5092},
    {0xABB5U, 43957, 5093, 5093, 5093, 5093},
    {0xABB6U, 43958, 5094, 5094, 5094, 5094},
    {0xABB7U, 43959, 5095, 5095, 5095, 5095},
    {0xABB8U, 43960, 5096, 5096, 5096, 5096},
    {0xABB9U, 43961, 5097, 5097, 5097, 5097},
    {0xABBAU, 43962, 5098, 5098, 5098, 5098},
    {0xABBBU, 43963, 5099, 5099, 5099, 5099},
    {0xABBCU, 43964, 5100, 5100, 5100, 5100},
    {0xABBDU, 43965, 5101, 5101, 5101, 5101},
    {0xABBEU, 43966, 5102, 5102, 5102, 5102},
    {0xABBFU, 43967, 5103, 5103, 5103, 5103},
    {0xFB00U, 64256, -2362, -2378, -2394, 64256},
    {0xFB01U, 64257, -2410, -2426, -2442, 64257},
    {0xFB02U, 64258, -2458, -2474, -2490, 64258},
    {0xFB03U, 64259, -2507, -2531, -2555, 64259},
    {0xFB04U, 64260, -2579, -2603, -2627, 64260},
    {0xFB05U, 64261, -2650, -2666, -2682, 64262},
    {0xFB06U, 64262, -2650, -2666, -2682, 64262},
    {0xFB13U, 64275, -2698, -2714, -2730, 64275},
    {0xFB14U, 64276, -2746, -2762, -2778, 64276},
    {0xFB15U, 64277, -2794, -2810, -2826, 64277},
    {0xFB16U, 64278, -2842, -2858, -2874, 64278},
    {0xFB17U, 64279, -2890, -2906, -2922, 64279},
    {0xFF21U, 65345, 65313, 65313, 65345, 65345},
    {0xFF22U, 65346, 65314, 65314, 65346, 65346},
    {0xFF23U, 65347, 65315, 65315, 65347, 65347},
    {0xFF24U, 65348, 65316, 65316, 65348, 65348},
    {0xFF25U, 65349, 65317, 65317, 65349, 65349},
    {0xFF26U, 65350, 65318, 65318, 65350, 65350},
    {0xFF27U, 65351, 65319, 65319, 65351, 65351},
    {0xFF28U, 65352, 65320, 65320, 65352, 65352},
    {0xFF29U, 65353, 65321, 65321, 65353, 65353},
    {0xFF2AU, 65354, 65322, 65322, 65354, 65354},
    {0xFF2BU, 65355, 65323, 65323, 65355, 65355},
    {0xFF2CU, 65356, 65324, 65324, 65356, 65356},
    {0xFF2DU, 65357, 65325, 65325, 65357, 65357},
    {0xFF2EU, 65358, 65326, 65326, 65358, 65358},
    {0xFF2FU, 65359, 65327, 65327, 65359, 65359},
    {0xFF30U, 65360, 65328, 65328, 65360, 65360},
    {0xFF31U, 65361, 65329, 65329, 65361, 65361},
    {0xFF32U, 65362, 65330, 65330, 65362, 65362},
    {0xFF33U, 65363, 65331, 65331, 65363, 65363},
    {0xFF34U, 65364, 65332, 65332, 65364, 65364},
    {0xFF35U, 65365, 65333, 65333, 65365, 65365},
    {0xFF36U, 65366, 65334, 65334, 65366, 65366},
    {0xFF37U, 65367, 65335, 65335, 65367, 65367},
    {0xFF38U, 65368, 65336, 65336, 65368, 65368},
    {0xFF39U, 65369, 65337, 65337, 65369, 65369},
    {0xFF3AU, 65370, 65338, 65338, 65370, 65370},
    {0xFF41U, 65345, 65313, 65313, 65345, 65345},
    {0xFF42U, 65346, 65314, 65314, 65346, 65346},
    {0xFF43U, 65347, 65315, 65315, 65347, 65347},
    {0xFF44U, 65348, 65316, 65316, 65348, 65348},
    {0xFF45U, 65349, 65317, 65317, 65349, 65349},
    {0xFF46U, 65350, 65318, 65318, 65350, 65350},
    {0xFF47U, 65351, 65319, 65319, 65351, 65351},
    {0xFF48U, 65352, 65320, 65320, 65352, 65352},
    {0xFF49U, 65353, 65321, 65321, 65353, 65353},
    {0xFF4AU, 65354, 65322, 65322, 65354, 65354},
    {0xFF4BU, 65355, 65323, 65323, 65355, 65355},
    {0xFF4CU, 65356, 65324, 65324, 65356, 65356},
    {0xFF4DU, 65357, 65325, 65325, 65357, 65357},
    {0xFF4EU, 65358, 65326, 65326, 65358, 65358},
    {0xFF4FU, 65359, 65327, 65327, 65359, 65359},
    {0xFF50U, 65360, 65328, 65328, 65360, 65360},
    {0xFF51U, 65361, 65329, 65329, 65361, 65361},
    {0xFF52U, 65362, 65330, 65330, 65362, 65362},
    {0xFF53U, 65363, 65331, 65331, 65363, 65363},
    {0xFF54U, 65364, 65332, 65332, 65364, 65364},
    {0xFF55U, 65365, 65333, 65333, 65365, 65365},
    {0xFF56U, 65366, 65334, 65334, 65366, 65366},
    {0xFF57U, 65367, 65335, 65335, 65367, 65367},
    {0xFF58U, 65368, 65336, 65336, 65368, 65368},
    {0xFF59U, 65369, 65337, 65337, 65369, 65369},
    {0xFF5AU, 65370, 65338, 65338, 65370, 65370},
    {0x10400U, 66600, 66560, 66560, 66600, 66600},
    {0x10401U, 66601, 66561, 66561, 66601, 66601},
    {0x10402U, 66602, 66562, 66562, 66602, 66602},
    {0x10403U, 66603, 66563, 66563, 66603, 66603},
    {0x10404U, 66604, 66564, 66564, 66604, 66604},
    {0x10405U, 66605, 66565, 66565, 66605, 66605},
    {0x10406U, 66606, 66566, 66566, 66606, 66606},
    {0x10407U, 66607, 66567, 66567, 66607, 66607},
    {0x10408U, 66608, 66568, 66568, 66608, 66608},
    {0x10409U, 66609, 66569, 66569, 66609, 66609},
    {0x1040AU, 66610, 66570, 66570, 66610, 66610},
    {0x1040BU, 66611, 66571, 66571, 66611, 66611},
    {0x1040CU, 66612, 66572, 66572, 66612, 66612},
    {0x1040DU, 66613, 66573, 66573, 66613, 66613},
    {0x1040EU, 66614, 66574, 66574, 66614, 66614},
    {0x1040FU, 66615, 66575, 66575, 66615, 66615},
    {0x10410U, 66616, 66576, 66576, 66616, 66616},
    {0x10411U, 66617, 66577, 66577, 66617, 66617},
    {0x10412U, 66618, 66578, 66578, 66618, 66618},
    {0x10413U, 66619, 66579, 66579, 66619, 66619},
    {0x10414U, 66620, 66580, 66580, 66620, 66620},
    {0x10415U, 66621, 66581, 66581, 66621, 66621},
    {0x10416U, 66622, 66582, 66582, 66622, 66622},
    {0x10417U, 66623, 66583, 66583, 66623, 66623},
    {0x10418U, 66624, 66584, 66584, 66624, 66624},
    {0x10419U, 66625, 66585, 66585, 66625, 66625},
    {0x1041AU, 66626, 66586, 66586, 66626, 66626},
    {0x1041BU, 66627, 66587, 66587, 66627, 66627},
    {0x1041CU, 66628, 66588, 66588, 66628, 66628},
    {0x1041DU, 66629, 66589, 66589, 66629, 66629},
    {0x1041EU, 66630, 66590, 66590, 66630, 66630},
    {0x1041FU, 66631, 66591, 66591, 66631, 66631},
    {0x10420U, 66632, 66592, 66592, 66632, 66632},
    {0x10421U, 66633, 66593, 66593, 66633, 66633},
    {0x10422U, 66634, 66594, 66594, 66634, 66634},
    {0x10423U, 66635, 66595, 66595, 66635, 66635},
    {0x10424U, 66636, 66596, 66596, 66636, 66636},
    {0x10425U, 66637, 66597, 66597, 66637, 66637},
    {0x10426U, 66638, 66598, 66598, 66638, 66638},
    {0x10427U, 66639, 66599, 66599, 66639, 66639},
    {0x10428U, 66600, 66560, 66560, 66600, 66600},
    {0x10429U, 66601, 66561, 66561, 66601, 66601},
    {0x1042AU, 66602, 66562, 66562, 66602, 66602},
    {0x1042BU, 66603, 66563, 66563, 66603, 66603},
    {0x1042CU, 66604, 66564, 66564, 66604, 66604},
    {0x1042DU, 66605, 66565, 66565, 66605, 66605},
    {0x1042EU, 66606, 66566, 66566, 66606, 66606},
    {0x1042FU, 66607, 66567, 66567, 66607, 66607},
    {0x10430U, 66608, 66568, 66568, 66608, 66608},
    {0x10431U, 66609, 66569, 66569, 66609, 66609},
    {0x10432U, 66610, 66570, 66570, 66610, 66610},
    {0x10433U, 66611, 66571, 66571, 66611, 66611},
    {0x10434U, 66612, 66572, 66572, 66612, 66612},
    {0x10435U, 66613, 66573, 66573, 66613, 66613},
    {0x10436U, 66614, 66574, 66574, 66614, 66614},
    {0x10437U, 66615, 66575, 66575, 66615, 66615},
    {0x10438U, 66616, 66576, 66576, 66616, 66616},
    {0x10439U, 66617, 66577, 66577, 66617, 66617},
    {0x1043AU, 66618, 66578, 66578, 66618, 66618},
    {0x1043BU, 66619, 66579, 66579, 66619, 66619},
    {0x1043CU, 66620, 66580, 66580, 66620, 66620},
    {0x1043DU, 66621, 66581, 66581, 66621, 66621},
    {0x1043EU, 66622, 66582, 66582, 66622, 66622},
    {0x1043FU, 66623, 66583, 66583, 66623, 66623},
    {0x10440U, 66624, 66584, 66584, 66624, 66624},
    {0x10441U, 66625, 66585, 66585, 66625, 66625},
    {0x10442U, 66626, 66586, 66586, 66626, 66626},
    {0x10443U, 66627, 66587, 66587, 66627, 66627},
    {0x10444U, 66628, 66588, 66588, 66628, 66628},
    {0x10445U, 66629, 66589, 66589, 66629, 66629},
    {0x10446U, 66630, 66590, 66590, 66630, 66630},
    {0x10447U, 66631, 66591, 66591, 66631, 66631},
    {0x10448U, 66632, 66592, 66592, 66632, 66632},
    {0x10449U, 66633, 66593, 66593, 66633, 66633},
    {0x1044AU, 66634, 66594, 66594, 66634, 66634},
    {0x1044BU, 66635, 66595, 66595, 66635, 66635},
    {0x1044CU, 66636, 66596, 66596, 66636, 66636},
    {0x1044DU, 66637, 66597, 66597, 66637, 66637},
    {0x1044EU, 66638, 66598, 66598, 66638, 66638},
    {0x1044FU, 66639, 66599, 66599, 66639, 66639},
    {0x104B0U, 66776, 66736, 66736, 66776, 66776},
    {0x104B1U, 66777, 66737, 66737, 66777, 66777},
    {0x104B2U, 66778, 66738, 66738, 66778, 66778},
    {0x104B3U, 66779, 66739, 66739, 66779, 66779},
    {0x104B4U, 66780, 66740, 66740, 66780, 66780},
    {0x104B5U, 66781, 66741, 66741, 66781, 66781},
    {0x104B6U, 66782, 66742, 66742, 66782, 66782},
    {0x104B7U, 66783, 66743, 66743, 66783, 66783},
    {0x104B8U, 66784, 66744, 66744, 66784, 66784},
    {0x104B9U, 66785, 66745, 66745, 66785, 66785},
    {0x104BAU, 66786, 66746, 66746, 66786, 66786},
    {0x104BBU, 66787, 66747, 66747, 66787, 66787},
    {0x104BCU, 66788, 66748, 66748, 66788, 66788},
    {0x104BDU, 66789, 66749, 66749, 66789, 66789},
    {0x104BEU, 66790, 66750, 66750, 66790, 66790},
    {0x104BFU, 66791, 66751, 66751, 66791, 66791},
    {0x104C0U, 66792, 66752, 66752, 66792, 66792},
    {0x104C1U, 66793, 66753, 66753, 66793, 66793},
    {0x104C2U, 66794, 66754, 66754, 66794, 66794},
    {0x104C3U, 66795, 66755, 66755, 66795, 66795},
    {0x104C4U, 66796, 66756, 66756, 66796, 66796},
    {0x104C5U, 66797, 66757, 66757, 66797, 66797},
    {0x104C6U, 66798, 66758, 66758, 66798, 66798},
    {0x104C7U, 66799, 66759, 66759, 66799, 66799},
    {0x104C8U, 66800, 66760, 66760, 66800, 66800},
    {0x104C9U, 66801, 66761, 66761, 66801, 66801},
    {0x104CAU, 66802, 66762, 66762, 66802, 66802},
    {0x104CBU, 66803, 66763, 66763, 66803, 66803},
    {0x104CCU, 66804, 66764, 66764, 66804, 66804},
    {0x104CDU, 66805, 66765, 66765, 66805, 66805},
    {0x104CEU, 66806, 66766, 66766, 66806, 66806},
    {0x104CFU, 66807, 66767, 66767, 66807, 66807},
    {0x104D0U, 66808, 66768, 66768, 66808, 66808},
    {0x104D1U, 66809, 66769, 66769, 66809, 66809},
    {0x104D2U, 66810, 66770, 66770, 66810, 66810},
    {0x104D3U, 66811, 66771, 66771, 66811, 66811},
    {0x104D8U, 66776, 66736, 66736, 66776, 66776},
    {0x104D9U, 66777, 66737, 66737, 66777, 66777},
    {0x104DAU, 66778, 66738, 66738, 66778, 66778},
    {0x104DBU, 66779, 66739, 66739, 66779, 66779},
    {0x104DCU, 66780, 66740, 66740, 66780, 66780},
    {0x104DDU, 66781, 66741, 66741, 66781, 66781},
    {0x104DEU, 66782, 66742, 66742, 66782, 66782},
    {0x104DFU, 66783, 66743, 66743, 66783, 66783},
    {0x104E0U, 66784, 66744, 66744, 66784, 66784},
    {0x104E1U, 66785, 66745, 66745, 66785, 66785},
    {0x104E2U, 66786, 66746, 66746, 66786, 66786},
    {0x104E3U, 66787, 66747, 66747, 66787, 66787},
    {0x104E4U, 66788, 66748, 66748, 66788, 66788},
    {0x104E5U, 66789, 66749, 66749, 66789, 66789},
    {0x104E6U, 66790, 66750, 66750, 66790, 66790},
    {0x104E7U, 66791, 66751, 66751, 66791, 66791},
    {0x104E8U, 66792, 66752, 66752, 66792, 66792},
    {0x104E9U, 66793, 66753, 66753, 66793, 66793},
    {0x104EAU, 66794, 66754, 66754, 66794, 66794},
    {0x104EBU, 66795, 66755, 66755, 66795, 66795},
    {0x104ECU, 66796, 66756, 66756, 66796, 66796},
    {0x104EDU, 66797, 66757, 66757, 66797, 66797},
    {0x104EEU, 66798, 66758, 66758, 66798, 66798},
    {0x104EFU, 66799, 66759, 66759, 66799, 66799},
    {0x104F0U, 66800, 66760, 66760, 66800, 66800},
    {0x104F1U, 66801, 66761, 66761, 66801, 66801},
    {0x104F2U, 66802, 66762, 66762, 66802, 66802},
    {0x104F3U, 66803, 66763, 66763, 66803, 66803},
    {0x104F4U, 66804, 66764, 66764, 66804, 66804},
    {0x104F5U, 66805, 66765, 66765, 66805, 66805},
    {0x104F6U, 66806, 66766, 66766, 66806, 66806},
    {0x104F7U, 66807, 66767, 66767, 66807, 66807},
    {0x104F8U, 66808, 66768, 66768, 66808, 66808},
    {0x104F9U, 66809, 66769, 66769, 66809, 66809},
    {0x104FAU, 66810, 66770, 66770, 66810, 66810},
    {0x104FBU, 66811, 66771, 66771, 66811, 66811},
    {0x10570U, 66967, 66928, 66928, 66967, 66967},
    {0x10571U, 66968, 66929, 66929, 66968, 66968},
    {0x10572U, 66969, 66930, 66930, 66969, 66969},
    {0x10573U, 66970, 66931, 66931, 66970, 66970},
    {0x10574U, 66971, 66932, 66932, 66971, 66971},
    {0x10575U, 66972, 66933, 66933, 66972, 66972},
    {0x10576U, 66973, 66934, 66934, 66973, 66973},
    {0x10577U, 66974, 66935, 66935, 66974, 66974},
    {0x10578U, 66975, 66936, 66936, 66975, 66975},
    {0x10579U, 66976, 66937, 66937, 66976, 66976},
    {0x1057AU, 66977, 66938, 66938, 66977, 66977},
    {0x1057CU, 66979, 66940, 66940, 66979, 66979},
    {0x1057DU, 66980, 66941, 66941, 66980, 66980},
    {0x1057EU, 66981, 66942, 66942, 66981, 66981},
    {0x1057FU, 66982, 66943, 66943, 66982, 66982},
    {0x10580U, 66983, 66944, 66944, 66983, 66983},
    {0x10581U, 66984, 66945, 66945, 66984, 66984},
    {0x10582U, 66985, 66946, 66946, 66985, 66985},
    {0x10583U, 66986, 66947, 66947, 66986, 66986},
    {0x10584U, 66987, 66948, 66948, 66987, 66987},
    {0x10585U, 66988, 66949, 66949, 66988, 66988},
    {0x10586U, 66989, 66950, 66950, 66989, 66989},
    {0x10587U, 66990, 66951, 66951, 66990, 66990},
    {0x10588U, 66991, 66952, 66952, 66991, 66991},
    {0x10589U, 66992, 66953, 66953, 66992, 66992},
    {0x1058AU, 66993, 66954, 66954, 66993, 66993},
    {0x1058CU, 66995, 66956, 66956, 66995, 66995},
    {0x1058DU, 66996, 66957, 66957, 66996, 66996},
    {0x1058EU, 66997, 66958, 66958, 66997, 66997},
    {0x1058FU, 66998, 66959, 66959, 66998, 66998},
    {0x10590U, 66999, 66960, 66960, 66999, 66999},
    {0x10591U, 67000, 66961, 66961, 67000, 67000},
    {0x10592U, 67001, 66962, 66962, 67001, 67001},
    {0x10594U, 67003, 66964, 66964, 67003, 67003},
    {0x10595U, 67004, 66965, 66965, 67004, 67004},
    {0x10597U, 66967, 66928, 66928, 66967, 66967},
    {0x10598U, 66968, 66929, 66929, 66968, 66968},
    {0x10599U, 66969, 66930, 66930, 66969, 66969},
    {0x1059AU, 66970, 66931, 66931, 66970, 66970},
    {0x1059BU, 66971, 66932, 66932, 66971, 66971},
    {0x1059CU, 66972, 66933, 66933, 66972, 66972},
    {0x1059DU, 66973, 66934, 66934, 66973, 66973},
    {0x1059EU, 66974, 66935, 66935, 66974, 66974},
    {0x1059FU, 66975, 66936, 66936, 66975, 66975},
    {0x105A0U, 66976, 66937, 66937, 66976, 66976},
    {0x105A1U, 66977, 66938, 66938, 66977, 66977},
    {0x105A3U, 66979, 66940, 66940, 66979, 66979},
    {0x105A4U, 66980, 66941, 66941, 66980, 66980},
    {0x105A5U, 66981, 66942, 66942, 66981, 66981},
    {0x105A6U, 66982, 66943, 66943, 66982, 66982},
    {0x105A7U, 66983, 66944, 66944, 66983, 66983},
    {0x105A8U, 66984, 66945, 66945, 66984, 66984},
    {0x105A9U, 66985, 66946, 66946, 66985, 66985},
    {0x105AAU, 66986, 66947, 66947, 66986, 66986},
    {0x105ABU, 66987, 66948, 66948, 66987, 66987},
    {0x105ACU, 66988, 66949, 66949, 66988, 66988},
    {0x105ADU, 66989, 66950, 66950, 66989, 66989},
    {0x105AEU, 66990, 66951, 66951, 66990, 66990},
    {0x105AFU, 66991, 66952, 66952, 66991, 66991},
    {0x105B0U, 66992, 66953, 66953, 66992, 66992},
    {0x105B1U, 66993, 66954, 66954, 66993, 66993},
    {0x105B3U, 66995, 66956, 66956, 66995, 66995},
    {0x105B4U, 66996, 66957, 66957, 66996, 66996},
    {0x105B5U, 66997, 66958, 66958, 66997, 66997},
    {0x105B6U, 66998, 66959, 66959, 66998, 66998},
    {0x105B7U, 66999, 66960, 66960, 66999, 66999},
    {0x105B8U, 67000, 66961, 66961, 67000, 67000},
    {0x105B9U, 67001, 66962, 66962, 67001, 67001},
    {0x105BBU, 67003, 66964, 66964, 67003, 67003},
    {0x105BCU, 67004, 66965, 66965, 67004, 67004},
    {0x10C80U, 68800, 68736, 68736, 68800, 68800},
    {0x10C81U, 68801, 68737, 68737, 68801, 68801},
    {0x10C82U, 68802, 68738, 68738, 68802, 68802},
    {0x10C83U, 68803, 68739, 68739, 68803, 68803},
    {0x10C84U, 68804, 68740, 68740, 68804, 68804},
    {0x10C85U, 68805, 68741, 68741, 68805, 68805},
    {0x10C86U, 68806, 68742, 68742, 68806, 68806},
    {0x10C87U, 68807, 68743, 68743, 68807, 68807},
    {0x10C88U, 68808, 68744, 68744, 68808, 68808},
    {0x10C89U, 68809, 68745, 68745, 68809, 68809},
    {0x10C8AU, 68810, 68746, 68746, 68810, 68810},
    {0x10C8BU, 68811, 68747, 68747, 68811, 68811},
    {0x10C8CU, 68812, 68748, 68748, 68812, 68812},
    {0x10C8DU, 68813, 68749, 68749, 68813, 68813},
    {0x10C8EU, 68814, 68750, 68750, 68814, 68814},
    {0x10C8FU, 68815, 68751, 68751, 68815, 68815},
    {0x10C90U, 68816, 68752, 68752, 68816, 68816},
    {0x10C91U, 68817, 68753, 68753, 68817, 68817},
    {0x10C92U, 68818, 68754, 68754, 68818, 68818},
    {0x10C93U, 68819, 68755, 68755, 68819, 68819},
    {0x10C94U, 68820, 68756, 68756, 68820, 68820},
    {0x10C95U, 68821, 68757, 68757, 68821, 68821},
    {0x10C96U, 68822, 68758, 68758, 68822, 68822},
    {0x10C97U, 68823, 68759, 68759, 68823, 68823},
    {0x10C98U, 68824, 68760, 68760, 68824, 68824},
    {0x10C99U, 68825, 68761, 68761, 68825, 68825},
    {0x10C9AU, 68826, 68762, 68762, 68826, 68826},
    {0x10C9BU, 68827, 68763, 68763, 68827, 68827},
    {0x10C9CU, 68828, 68764, 68764, 68828, 68828},
    {0x10C9DU, 68829, 68765, 68765, 68829, 68829},
    {0x10C9EU, 68830, 68766, 68766, 68830, 68830},
    {0x10C9FU, 68831, 68767, 68767, 68831, 68831},
    {0x10CA0U, 68832, 68768, 68768, 68832, 68832},
    {0x10CA1U, 68833, 68769, 68769, 68833, 68833},
    {0x10CA2U, 68834, 68770, 68770, 68834, 68834},
    {0x10CA3U, 68835, 68771, 68771, 68835, 68835},
    {0x10CA4U, 68836, 68772, 68772, 68836, 68836},
    {0x10CA5U, 68837, 68773, 68773, 68837, 68837},
    {0x10CA6U, 68838, 68774, 68774, 68838, 68838},
    {0x10CA7U, 68839, 68775, 68775, 68839, 68839},
    {0x10CA8U, 68840, 68776, 68776, 68840, 68840},
    {0x10CA9U, 68841, 68777, 68777, 68841, 68841},
    {0x10CAAU, 68842, 68778, 68778, 68842, 68842},
    {0x10CABU, 68843, 68779, 68779, 68843, 68843},
    {0x10CACU, 68844, 68780, 68780, 68844, 68844},
    {0x10CADU, 68845, 68781, 68781, 68845, 68845},
    {0x10CAEU, 68846, 68782, 68782, 68846, 68846},
    {0x10CAFU, 68847, 68783, 68783, 68847, 68847},
    {0x10CB0U, 68848, 68784, 68784, 68848, 68848},
    {0x10CB1U, 68849, 68785, 68785, 68849, 68849},
    {0x10CB2U, 68850, 68786, 68786, 68850, 68850},
    {0x10CC0U, 68800, 68736, 68736, 68800, 68800},
    {0x10CC1U, 68801, 68737, 68737, 68801, 68801},
    {0x10CC2U, 68802, 68738, 68738, 68802, 68802},
    {0x10CC3U, 68803, 68739, 68739, 68803, 68803},
    {0x10CC4U, 68804, 68740, 68740, 68804, 68804},
    {0x10CC5U, 68805, 68741, 68741, 68805, 68805},
    {0x10CC6U, 68806, 68742, 68742, 68806, 68806},
    {0x10CC7U, 68807, 68743, 68743, 68807, 68807},
    {0x10CC8U, 68808, 68744, 68744, 68808, 68808},
    {0x10CC9U, 68809, 68745, 68745, 68809, 68809},
    {0x10CCAU, 68810, 68746, 68746, 68810, 68810},
    {0x10CCBU, 68811, 68747, 68747, 68811, 68811},
    {0x10CCCU, 68812, 68748, 68748, 68812, 68812},
    {0x10CCDU, 68813, 68749, 68749, 68813, 68813},
    {0x10CCEU, 68814, 68750, 68750, 68814, 68814},
    {0x10CCFU, 68815, 68751, 68751, 68815, 68815},
    {0x10CD0U, 68816, 68752, 68752, 68816, 68816},
    {0x10CD1U, 68817, 68753, 68753, 68817, 68817},
    {0x10CD2U, 68818, 68754, 68754, 68818, 68818},
    {0x10CD3U, 68819, 68755, 68755, 68819, 68819},
    {0x10CD4U, 68820, 68756, 68756, 68820, 68820},
    {0x10CD5U, 68821, 68757, 68757, 68821, 68821},
    {0x10CD6U, 68822, 68758, 68758, 68822, 68822},
    {0x10CD7U, 68823, 68759, 68759, 68823, 68823},
    {0x10CD8U, 68824, 68760, 68760, 68824, 68824},
    {0x10CD9U, 68825, 68761, 68761, 68825, 68825},
    {0x10CDAU, 68826, 68762, 68762, 68826, 68826},
    {0x10CDBU, 68827, 68763, 68763, 68827, 68827},
    {0x10CDCU, 68828, 68764, 68764, 68828, 68828},
    {0x10CDDU, 68829, 68765, 68765, 68829, 68829},
    {0x10CDEU, 68830, 68766, 68766, 68830, 68830},
    {0x10CDFU, 68831, 68767, 68767, 68831, 68831},
    {0x10CE0U, 68832, 68768, 68768, 68832, 68832},
    {0x10CE1U, 68833, 68769, 68769, 68833, 68833},
    {0x10CE2U, 68834, 68770, 68770, 68834, 68834},
    {0x10CE3U, 68835, 68771, 68771, 68835, 68835},
    {0x10CE4U, 68836, 68772, 68772, 68836, 68836},
    {0x10CE5U, 68837, 68773, 68773, 68837, 68837},
    {0x10CE6U, 68838, 68774, 68774, 68838, 68838},
    {0x10CE7U, 68839, 68775, 68775, 68839, 68839},
    {0x10CE8U, 68840, 68776, 68776, 68840, 68840},
    {0x10CE9U, 68841, 68777, 68777, 68841, 68841},
    {0x10CEAU, 68842, 68778, 68778, 68842, 68842},
    {0x10CEBU, 68843, 68779, 68779, 68843, 68843},
    {0x10CECU, 68844, 68780, 68780, 68844, 68844},
    {0x10CEDU, 68845, 68781, 68781, 68845, 68845},
    {0x10CEEU, 68846, 68782, 68782, 68846, 68846},
    {0x10CEFU, 68847, 68783, 68783, 68847, 68847},
    {0x10CF0U, 68848, 68784, 68784, 68848, 68848},
    {0x10CF1U, 68849, 68785, 68785, 68849, 68849},
    {0x10CF2U, 68850, 68786, 68786, 68850, 68850},
    {0x10D50U, 68976, 68944, 68944, 68976, 68976},
    {0x10D51U, 68977, 68945, 68945, 68977, 68977},
    {0x10D52U, 68978, 68946, 68946, 68978, 68978},
    {0x10D53U, 68979, 68947, 68947, 68979, 68979},
    {0x10D54U, 68980, 68948, 68948, 68980, 68980},
    {0x10D55U, 68981, 68949, 68949, 68981, 68981},
    {0x10D56U, 68982, 68950, 68950, 68982, 68982},
    {0x10D57U, 68983, 68951, 68951, 68983, 68983},
    {0x10D58U, 68984, 68952, 68952, 68984, 68984},
    {0x10D59U, 68985, 68953, 68953, 68985, 68985},
    {0x10D5AU, 68986, 68954, 68954, 68986, 68986},
    {0x10D5BU, 68987, 68955, 68955, 68987, 68987},
    {0x10D5CU, 68988, 68956, 68956, 68988, 68988},
    {0x10D5DU, 68989, 68957, 68957, 68989, 68989},
    {0x10D5EU, 68990, 68958, 68958, 68990, 68990},
    {0x10D5FU, 68991, 68959, 68959, 68991, 68991},
    {0x10D60U, 68992, 68960, 68960, 68992, 68992},
    {0x10D61U, 68993, 68961, 68961, 68993, 68993},
    {0x10D62U, 68994, 68962, 68962, 68994, 68994},
    {0x10D63U, 68995, 68963, 68963, 68995, 68995},
    {0x10D64U, 68996, 68964, 68964, 68996, 68996},
    {0x10D65U, 68997, 68965, 68965, 68997, 68997},
    {0x10D70U, 68976, 68944, 68944, 68976, 68976},
    {0x10D71U, 68977, 68945, 68945, 68977, 68977},
    {0x10D72U, 68978, 68946, 68946, 68978, 68978},
    {0x10D73U, 68979, 68947, 68947, 68979, 68979},
    {0x10D74U, 68980, 68948, 68948, 68980, 68980},
    {0x10D75U, 68981, 68949, 68949, 68981, 68981},
    {0x10D76U, 68982, 68950, 68950, 68982, 68982},
    {0x10D77U, 68983, 68951, 68951, 68983, 68983},
    {0x10D78U, 68984, 68952, 68952, 68984, 68984},
    {0x10D79U, 68985, 68953, 68953, 68985, 68985},
    {0x10D7AU, 68986, 68954, 68954, 68986, 68986},
    {0x10D7BU, 68987, 68955, 68955, 68987, 68987},
    {0x10D7CU, 68988, 68956, 68956, 68988, 68988},
    {0x10D7DU, 68989, 68957, 68957, 68989, 68989},
    {0x10D7EU, 68990, 68958, 68958, 68990, 68990},
    {0x10D7FU, 68991, 68959, 68959, 68991, 68991},
    {0x10D80U, 68992, 68960, 68960, 68992, 68992},
    {0x10D81U, 68993, 68961, 68961, 68993, 68993},
    {0x10D82U, 68994, 68962, 68962, 68994, 68994},
    {0x10D83U, 68995, 68963, 68963, 68995, 68995},
    {0x10D84U, 68996, 68964, 68964, 68996, 68996},
    {0x10D85U, 68997, 68965, 68965, 68997, 68997},
    {0x118A0U, 71872, 71840, 71840, 71872, 71872},
    {0x118A1U, 71873, 71841, 71841, 71873, 71873},
    {0x118A2U, 71874, 71842, 71842, 71874, 71874},
    {0x118A3U, 71875, 71843, 71843, 71875, 71875},
    {0x118A4U, 71876, 71844, 71844, 71876, 71876},
    {0x118A5U, 71877, 71845, 71845, 71877, 71877},
    {0x118A6U, 71878, 71846, 71846, 71878, 71878},
    {0x118A7U, 71879, 71847, 71847, 71879, 71879},
    {0x118A8U, 71880, 71848, 71848, 71880, 71880},
    {0x118A9U, 71881, 71849, 71849, 71881, 71881},
    {0x118AAU, 71882, 71850, 71850, 71882, 71882},
    {0x118ABU, 71883, 71851, 71851, 71883, 71883},
    {0x118ACU, 71884, 71852, 71852, 71884, 71884},
    {0x118ADU, 71885, 71853, 71853, 71885, 71885},
    {0x118AEU, 71886, 71854, 71854, 71886, 71886},
    {0x118AFU, 71887, 71855, 71855, 71887, 71887},
    {0x118B0U, 71888, 71856, 71856, 71888, 71888},
    {0x118B1U, 71889, 71857, 71857, 71889, 71889},
    {0x118B2U, 71890, 71858, 71858, 71890, 71890},
    {0x118B3U, 71891, 71859, 71859, 71891, 71891},
    {0x118B4U, 71892, 71860, 71860, 71892, 71892},
    {0x118B5U, 71893, 71861, 71861, 71893, 71893},
    {0x118B6U, 71894, 71862, 71862, 71894, 71894},
    {0x118B7U, 71895, 71863, 71863, 71895, 71895},
    {0x118B8U, 71896, 71864, 71864, 71896, 71896},
    {0x118B9U, 71897, 71865, 71865, 71897, 71897},
    {0x118BAU, 71898, 71866, 71866, 71898, 71898},
    {0x118BBU, 71899, 71867, 71867, 71899, 71899},
    {0x118BCU, 71900, 71868, 71868, 71900, 71900},
    {0x118BDU, 71901, 71869, 71869, 71901, 71901},
    {0x118BEU, 71902, 71870, 71870, 71902, 71902},
    {0x118BFU, 71903, 71871, 71871, 71903, 71903},
    {0x118C0U, 71872, 71840, 71840, 71872, 71872},
    {0x118C1U, 71873, 71841, 71841, 71873, 71873},
    {0x118C2U, 71874, 71842, 71842, 71874, 71874},
    {0x118C3U, 71875, 71843, 71843, 71875, 71875},
    {0x118C4U, 71876, 71844, 71844, 71876, 71876},
    {0x118C5U, 71877, 71845, 71845, 71877, 71877},
    {0x118C6U, 71878, 71846, 71846, 71878, 71878},
    {0x118C7U, 71879, 71847, 71847, 71879, 71879},
    {0x118C8U, 71880, 71848, 71848, 71880, 71880},
    {0x118C9U, 71881, 71849, 71849, 71881, 71881},
    {0x118CAU, 71882, 71850, 71850, 71882, 71882},
    {0x118CBU, 71883, 71851, 71851, 71883, 71883},
    {0x118CCU, 71884, 71852, 71852, 71884, 71884},
    {0x118CDU, 71885, 71853, 71853, 71885, 71885},
    {0x118CEU, 71886, 71854, 71854, 71886, 71886},
    {0x118CFU, 71887, 71855, 71855, 71887, 71887},
    {0x118D0U, 71888, 71856, 71856, 71888, 71888},
    {0x118D1U, 71889, 71857, 71857, 71889, 71889},
    {0x118D2U, 71890, 71858, 71858, 71890, 71890},
    {0x118D3U, 71891, 71859, 71859, 71891, 71891},
    {0x118D4U, 71892, 71860, 71860, 71892, 71892},
    {0x118D5U, 71893, 71861, 71861, 71893, 71893},
    {0x118D6U, 71894, 71862, 71862, 71894, 71894},
    {0x118D7U, 71895, 71863, 71863, 71895, 71895},
    {0x118D8U, 71896, 71864, 71864, 71896, 71896},
    {0x118D9U, 71897, 71865, 71865, 71897, 71897},
    {0x118DAU, 71898, 71866, 71866, 71898, 71898},
    {0x118DBU, 71899, 71867, 71867, 71899, 71899},
    {0x118DCU, 71900, 71868, 71868, 71900, 71900},
    {0x118DDU, 71901, 71869, 71869, 71901, 71901},
    {0x118DEU, 71902, 71870, 71870, 71902, 71902},
    {0x118DFU, 71903, 71871, 71871, 71903, 71903},
    {0x16E40U, 93792, 93760, 93760, 93792, 93792},
    {0x16E41U, 93793, 93761, 93761, 93793, 93793},
    {0x16E42U, 93794, 93762, 93762, 93794, 93794},
    {0x16E43U, 93795, 93763, 93763, 93795, 93795},
    {0x16E44U, 93796, 93764, 93764, 93796, 93796},
    {0x16E45U, 93797, 93765, 93765, 93797, 93797},
    {0x16E46U, 93798, 93766, 93766, 93798, 93798},
    {0x16E47U, 93799, 93767, 93767, 93799, 93799},
    {0x16E48U, 93800, 93768, 93768, 93800, 93800},
    {0x16E49U, 93801, 93769, 93769, 93801, 93801},
    {0x16E4AU, 93802, 93770, 93770, 93802, 93802},
    {0x16E4BU, 93803, 93771, 93771, 93803, 93803},
    {0x16E4CU, 93804, 93772, 93772, 93804, 93804},
    {0x16E4DU, 93805, 93773, 93773, 93805, 93805},
    {0x16E4EU, 93806, 93774, 93774, 93806, 93806},
    {0x16E4FU, 93807, 93775, 93775, 93807, 93807},
    {0x16E50U, 93808, 93776, 93776, 93808, 93808},
    {0x16E51U, 93809, 93777, 93777, 93809, 93809},
    {0x16E52U, 93810, 93778, 93778, 93810, 93810},
    {0x16E53U, 93811, 93779, 93779, 93811, 93811},
    {0x16E54U, 93812, 93780, 93780, 93812, 93812},
    {0x16E55U, 93813, 93781, 93781, 93813, 93813},
    {0x16E56U, 93814, 93782, 93782, 93814, 93814},
    {0x16E57U, 93815, 93783, 93783, 93815, 93815},
    {0x16E58U, 93816, 93784, 93784, 93816, 93816},
    {0x16E59U, 93817, 93785, 93785, 93817, 93817},
    {0x16E5AU, 93818, 93786, 93786, 93818, 93818},
    {0x16E5BU, 93819, 93787, 93787, 93819, 93819},
    {0x16E5CU, 93820, 93788, 93788, 93820, 93820},
    {0x16E5DU, 93821, 93789, 93789, 93821, 93821},
    {0x16E5EU, 93822, 93790, 93790, 93822, 93822},
    {0x16E5FU, 93823, 93791, 93791, 93823, 93823},
    {0x16E60U, 93792, 93760, 93760, 93792, 93792},
    {0x16E61U, 93793, 93761, 93761, 93793, 93793},
    {0x16E62U, 93794, 93762, 93762, 93794, 93794},
    {0x16E63U, 93795, 93763, 93763, 93795, 93795},
    {0x16E64U, 93796, 93764, 93764, 93796, 93796},
    {0x16E65U, 93797, 93765, 93765, 93797, 93797},
    {0x16E66U, 93798, 93766, 93766, 93798, 93798},
    {0x16E67U, 93799, 93767, 93767, 93799, 93799},
    {0x16E68U, 93800, 93768, 93768, 93800, 93800},
    {0x16E69U, 93801, 93769, 93769, 93801, 93801},
    {0x16E6AU, 93802, 93770, 93770, 93802, 93802},
    {0x16E6BU, 93803, 93771, 93771, 93803, 93803},
    {0x16E6CU, 93804, 93772, 93772, 93804, 93804},
    {0x16E6DU, 93805, 93773, 93773, 93805, 93805},
    {0x16E6EU, 93806, 93774, 93774, 93806, 93806},
    {0x16E6FU, 93807, 93775, 93775, 93807, 93807},
    {0x16E70U, 93808, 93776, 93776, 93808, 93808},
    {0x16E71U, 93809, 93777, 93777, 93809, 93809},
    {0x16E72U, 93810, 93778, 93778, 93810, 93810},
    {0x16E73U, 93811, 93779, 93779, 93811, 93811},
    {0x16E74U, 93812, 93780, 93780, 93812, 93812},
    {0x16E75U, 93813, 93781, 93781, 93813, 93813},
    {0x16E76U, 93814, 93782, 93782, 93814, 93814},
    {0x16E77U, 93815, 93783, 93783, 93815, 93815},
    {0x16E78U, 93816, 93784, 93784, 93816, 93816},
    {0x16E79U, 93817, 93785, 93785, 93817, 93817},
    {0x16E7AU, 93818, 93786, 93786, 93818, 93818},
    {0x16E7BU, 93819, 93787, 93787, 93819, 93819},
    {0x16E7CU, 93820, 93788, 93788, 93820, 93820},
    {0x16E7DU, 93821, 93789, 93789, 93821, 93821},
    {0x16E7EU, 93822, 93790, 93790, 93822, 93822},
    {0x16E7FU, 93823, 93791, 93791, 93823, 93823},
    {0x16EA0U, 93883, 93856, 93856, 93883, 93883},
    {0x16EA1U, 93884, 93857, 93857, 93884, 93884},
    {0x16EA2U, 93885, 93858, 93858, 93885, 93885},
    {0x16EA3U, 93886, 93859, 93859, 93886, 93886},
    {0x16EA4U, 93887, 93860, 93860, 93887, 93887},
    {0x16EA5U, 93888, 93861, 93861, 93888, 93888},
    {0x16EA6U, 93889, 93862, 93862, 93889, 93889},
    {0x16EA7U, 93890, 93863, 93863, 93890, 93890},
    {0x16EA8U, 93891, 93864, 93864, 93891, 93891},
    {0x16EA9U, 93892, 93865, 93865, 93892, 93892},
    {0x16EAAU, 93893, 93866, 93866, 93893, 93893},
    {0x16EABU, 93894, 93867, 93867, 93894, 93894},
    {0x16EACU, 93895, 93868, 93868, 93895, 93895},
    {0x16EADU, 93896, 93869, 93869, 93896, 93896},
    {0x16EAEU, 93897, 93870, 93870, 93897, 93897},
    {0x16EAFU, 93898, 93871, 93871, 93898, 93898},
    {0x16EB0U, 93899, 93872, 93872, 93899, 93899},
    {0x16EB1U, 93900, 93873, 93873, 93900, 93900},
    {0x16EB2U, 93901, 93874, 93874, 93901, 93901},
    {0x16EB3U, 93902, 93875, 93875, 93902, 93902},
    {0x16EB4U, 93903, 93876, 93876, 93903, 93903},
    {0x16EB5U, 93904, 93877, 93877, 93904, 93904},
    {0x16EB6U, 93905, 93878, 93878, 93905, 93905},
    {0x16EB7U, 93906, 93879, 93879, 93906, 93906},
    {0x16EB8U, 93907, 93880, 93880, 93907, 93907},
    {0x16EBBU, 93883, 93856, 93856, 93883, 93883},
    {0x16EBCU, 93884, 93857, 93857, 93884, 93884},
    {0x16EBDU, 93885, 93858, 93858, 93885, 93885},
    {0x16EBEU, 93886, 93859, 93859, 93886, 93886},
    {0x16EBFU, 93887, 93860, 93860, 93887, 93887},
    {0x16EC0U, 93888, 93861, 93861, 93888, 93888},
    {0x16EC1U, 93889, 93862, 93862, 93889, 93889},
    {0x16EC2U, 93890, 93863, 93863, 93890, 93890},
    {0x16EC3U, 93891, 93864, 93864, 93891, 93891},
    {0x16EC4U, 93892, 93865, 93865, 93892, 93892},
    {0x16EC5U, 93893, 93866, 93866, 93893, 93893},
    {0x16EC6U, 93894, 93867, 93867, 93894, 93894},
    {0x16EC7U, 93895, 93868, 93868, 93895, 93895},
    {0x16EC8U, 93896, 93869, 93869, 93896, 93896},
    {0x16EC9U, 93897, 93870, 93870, 93897, 93897},
    {0x16ECAU, 93898, 93871, 93871, 93898, 93898},
    {0x16ECBU, 93899, 93872, 93872, 93899, 93899},
    {0x16ECCU, 93900, 93873, 93873, 93900, 93900},
    {0x16ECDU, 93901, 93874, 93874, 93901, 93901},
    {0x16ECEU, 93902, 93875, 93875, 93902, 93902},
    {0x16ECFU, 93903, 93876, 93876, 93903, 93903},
    {0x16ED0U, 93904, 93877, 93877, 93904, 93904},
    {0x16ED1U, 93905, 93878, 93878, 93905, 93905},
    {0x16ED2U, 93906, 93879, 93879, 93906, 93906},
    {0x16ED3U, 93907, 93880, 93880, 93907, 93907},
    {0x1E900U, 125218, 125184, 125184, 125218, 125218},
    {0x1E901U, 125219, 125185, 125185, 125219, 125219},
    {0x1E902U, 125220, 125186, 125186, 125220, 125220},
    {0x1E903U, 125221, 125187, 125187, 125221, 125221},
    {0x1E904U, 125222, 125188, 125188, 125222, 125222},
    {0x1E905U, 125223, 125189, 125189, 125223, 125223},
    {0x1E906U, 125224, 125190, 125190, 125224, 125224},
    {0x1E907U, 125225, 125191, 125191, 125225, 125225},
    {0x1E908U, 125226, 125192, 125192, 125226, 125226},
    {0x1E909U, 125227, 125193, 125193, 125227, 125227},
    {0x1E90AU, 125228, 125194, 125194, 125228, 125228},
    {0x1E90BU, 125229, 125195, 125195, 125229, 125229},
    {0x1E90CU, 125230, 125196, 125196, 125230, 125230},
    {0x1E90DU, 125231, 125197, 125197, 125231, 125231},
    {0x1E90EU, 125232, 125198, 125198, 125232, 125232},
    {0x1E90FU, 125233, 125199, 125199, 125233, 125233},
    {0x1E910U, 125234, 125200, 125200, 125234, 125234},
    {0x1E911U, 125235, 125201, 125201, 125235, 125235},
    {0x1E912U, 125236, 125202, 125202, 125236, 125236},
    {0x1E913U, 125237, 125203, 125203, 125237, 125237},
    {0x1E914U, 125238, 125204, 125204, 125238, 125238},
    {0x1E915U, 125239, 125205, 125205, 125239, 125239},
    {0x1E916U, 125240, 125206, 125206, 125240, 125240},
    {0x1E917U, 125241, 125207, 125207, 125241, 125241},
    {0x1E918U, 125242, 125208, 125208, 125242, 125242},
    {0x1E919U, 125243, 125209, 125209, 125243, 125243},
    {0x1E91AU, 125244, 125210, 125210, 125244, 125244},
    {0x1E91BU, 125245, 125211, 125211, 125245, 125245},
    {0x1E91CU, 125246, 125212, 125212, 125246, 125246},
    {0x1E91DU, 125247, 125213, 125213, 125247, 125247},
    {0x1E91EU, 125248, 125214, 125214, 125248, 125248},
    {0x1E91FU, 125249, 125215, 125215, 125249, 125249},
    {0x1E920U, 125250, 125216, 125216, 125250, 125250},
    {0x1E921U, 125251, 125217, 125217, 125251, 125251},
    {0x1E922U, 125218, 125184, 125184, 125218, 125218},
    {0x1E923U, 125219, 125185, 125185, 125219, 125219},
    {0x1E924U, 125220, 125186, 125186, 125220, 125220},
    {0x1E925U, 125221, 125187, 125187, 125221, 125221},
    {0x1E926U, 125222, 125188, 125188, 125222, 125222},
    {0x1E927U, 125223, 125189, 125189, 125223, 125223},
    {0x1E928U, 125224, 125190, 125190, 125224, 125224},
    {0x1E929U, 125225, 125191, 125191, 125225, 125225},
    {0x1E92AU, 125226, 125192, 125192, 125226, 125226},
    {0x1E92BU, 125227, 125193, 125193, 125227, 125227},
    {0x1E92CU, 125228, 125194, 125194, 125228, 125228},
    {0x1E92DU, 125229, 125195, 125195, 125229, 125229},
    {0x1E92EU, 125230, 125196, 125196, 125230, 125230},
    {0x1E92FU, 125231, 125197, 125197, 125231, 125231},
    {0x1E930U, 125232, 125198, 125198, 125232, 125232},
    {0x1E931U, 125233, 125199, 125199, 125233, 125233},
    {0x1E932U, 125234, 125200, 125200, 125234, 125234},
    {0x1E933U, 125235, 125201, 125201, 125235, 125235},
    {0x1E934U, 125236, 125202, 125202, 125236, 125236},
    {0x1E935U, 125237, 125203, 125203, 125237, 125237},
    {0x1E936U, 125238, 125204, 125204, 125238, 125238},
    {0x1E937U, 125239, 125205, 125205, 125239, 125239},
    {0x1E938U, 125240, 125206, 125206, 125240, 125240},
    {0x1E939U, 125241, 125207, 125207, 125241, 125241},
    {0x1E93AU, 125242, 125208, 125208, 125242, 125242},
    {0x1E93BU, 125243, 125209, 125209, 125243, 125243},
    {0x1E93CU, 125244, 125210, 125210, 125244, 125244},
    {0x1E93DU, 125245, 125211, 125211, 125245, 125245},
    {0x1E93EU, 125246, 125212, 125212, 125246, 125246},
    {0x1E93FU, 125247, 125213, 125213, 125247, 125247},
    {0x1E940U, 125248, 125214, 125214, 125248, 125248},
    {0x1E941U, 125249, 125215, 125215, 125249, 125249},
    {0x1E942U, 125250, 125216, 125216, 125250, 125250},
    {0x1E943U, 125251, 125217, 125217, 125251, 125251},
};

static const uint32_t cs_ucase_multi_pool[367] = {
    0x53U, 0x53U, 0x53U, 0x73U, 0x73U, 0x73U, 0x69U, 0x307U, 0x2BCU, 0x4EU, 0x2BCU, 0x6EU,
    0x4AU, 0x30CU, 0x6AU, 0x30CU, 0x399U, 0x308U, 0x301U, 0x3B9U, 0x308U, 0x301U, 0x3A5U, 0x308U,
    0x301U, 0x3C5U, 0x308U, 0x301U, 0x535U, 0x552U, 0x535U, 0x582U, 0x565U, 0x582U, 0x48U, 0x331U,
    0x68U, 0x331U, 0x54U, 0x308U, 0x74U, 0x308U, 0x57U, 0x30AU, 0x77U, 0x30AU, 0x59U, 0x30AU,
    0x79U, 0x30AU, 0x41U, 0x2BEU, 0x61U, 0x2BEU, 0x3A5U, 0x313U, 0x3C5U, 0x313U, 0x3A5U, 0x313U,
    0x300U, 0x3C5U, 0x313U, 0x300U, 0x3A5U, 0x313U, 0x301U, 0x3C5U, 0x313U, 0x301U, 0x3A5U, 0x313U,
    0x342U, 0x3C5U, 0x313U, 0x342U, 0x1F08U, 0x399U, 0x1F00U, 0x3B9U, 0x1F09U, 0x399U, 0x1F01U, 0x3B9U,
    0x1F0AU, 0x399U, 0x1F02U, 0x3B9U, 0x1F0BU, 0x399U, 0x1F03U, 0x3B9U, 0x1F0CU, 0x399U, 0x1F04U, 0x3B9U,
    0x1F0DU, 0x399U, 0x1F05U, 0x3B9U, 0x1F0EU, 0x399U, 0x1F06U, 0x3B9U, 0x1F0FU, 0x399U, 0x1F07U, 0x3B9U,
    0x1F28U, 0x399U, 0x1F20U, 0x3B9U, 0x1F29U, 0x399U, 0x1F21U, 0x3B9U, 0x1F2AU, 0x399U, 0x1F22U, 0x3B9U,
    0x1F2BU, 0x399U, 0x1F23U, 0x3B9U, 0x1F2CU, 0x399U, 0x1F24U, 0x3B9U, 0x1F2DU, 0x399U, 0x1F25U, 0x3B9U,
    0x1F2EU, 0x399U, 0x1F26U, 0x3B9U, 0x1F2FU, 0x399U, 0x1F27U, 0x3B9U, 0x1F68U, 0x399U, 0x1F60U, 0x3B9U,
    0x1F69U, 0x399U, 0x1F61U, 0x3B9U, 0x1F6AU, 0x399U, 0x1F62U, 0x3B9U, 0x1F6BU, 0x399U, 0x1F63U, 0x3B9U,
    0x1F6CU, 0x399U, 0x1F64U, 0x3B9U, 0x1F6DU, 0x399U, 0x1F65U, 0x3B9U, 0x1F6EU, 0x399U, 0x1F66U, 0x3B9U,
    0x1F6FU, 0x399U, 0x1F67U, 0x3B9U, 0x1FBAU, 0x399U, 0x1FBAU, 0x345U, 0x1F70U, 0x3B9U, 0x391U, 0x399U,
    0x3B1U, 0x3B9U, 0x386U, 0x399U, 0x386U, 0x345U, 0x3ACU, 0x3B9U, 0x391U, 0x342U, 0x3B1U, 0x342U,
    0x391U, 0x342U, 0x399U, 0x391U, 0x342U, 0x345U, 0x3B1U, 0x342U, 0x3B9U, 0x1FCAU, 0x399U, 0x1FCAU,
    0x345U, 0x1F74U, 0x3B9U, 0x397U, 0x399U, 0x3B7U, 0x3B9U, 0x389U, 0x399U, 0x389U, 0x345U, 0x3AEU,
    0x3B9U, 0x397U, 0x342U, 0x3B7U, 0x342U, 0x397U, 0x342U, 0x399U, 0x397U, 0x342U, 0x345U, 0x3B7U,
    0x342U, 0x3B9U, 0x399U, 0x308U, 0x300U, 0x3B9U, 0x308U, 0x300U, 0x399U, 0x342U, 0x3B9U, 0x342U,
    0x399U, 0x308U, 0x342U, 0x3B9U, 0x308U, 0x342U, 0x3A5U, 0x308U, 0x300U, 0x3C5U, 0x308U, 0x300U,
    0x3A1U, 0x313U, 0x3C1U, 0x313U, 0x3A5U, 0x342U, 0x3C5U, 0x342U, 0x3A5U, 0x308U, 0x342U, 0x3C5U,
    0x308U, 0x342U, 0x1FFAU, 0x399U, 0x1FFAU, 0x345U, 0x1F7CU, 0x3B9U, 0x3A9U, 0x399U, 0x3C9U, 0x3B9U,
    0x38FU, 0x399U, 0x38FU, 0x345U, 0x3CEU, 0x3B9U, 0x3A9U, 0x342U, 0x3C9U, 0x342U, 0x3A9U, 0x342U,
    0x399U, 0x3A9U, 0x342U, 0x345U, 0x3C9U, 0x342U, 0x3B9U, 0x46U, 0x46U, 0x46U, 0x66U, 0x66U,
    0x66U, 0x46U, 0x49U, 0x46U, 0x69U, 0x66U, 0x69U, 0x46U, 0x4CU, 0x46U, 0x6CU, 0x66U,
    0x6CU, 0x46U, 0x46U, 0x49U, 0x46U, 0x66U, 0x69U, 0x66U, 0x66U, 0x69U, 0x46U, 0x46U,
    0x4CU, 0x46U, 0x66U, 0x6CU, 0x66U, 0x66U, 0x6CU, 0x53U, 0x54U, 0x53U, 0x74U, 0x73U,
    0x74U, 0x544U, 0x546U, 0x544U, 0x576U, 0x574U, 0x576U, 0x544U, 0x535U, 0x544U, 0x565U, 0x574U,
    0x565U, 0x544U, 0x53BU, 0x544U, 0x56BU, 0x574U, 0x56BU, 0x54EU, 0x546U, 0x54EU, 0x576U, 0x57EU,
    0x576U, 0x544U, 0x53DU, 0x544U, 0x56DU, 0x574U, 0x56DU
};

typedef struct {
    uint32_t cp;
    uint32_t info;
} cs_udecomp_entry;

#define CS_UDECOMP_COUNT 5914U
static const cs_udecomp_entry cs_udecomp_table[5914] = {
    {0x00A0U, 0x81000000U},
    {0x00A8U, 0x82000001U},
    {0x00AAU, 0x81000003U},
    {0x00AFU, 0x82000004U},
    {0x00B2U, 0x81000006U},
    {0x00B3U, 0x81000007U},
    {0x00B4U, 0x82000008U},
    {0x00B5U, 0x8100000AU},
    {0x00B8U, 0x8200000BU},
    {0x00B9U, 0x8100000DU},
    {0x00BAU, 0x8100000EU},
    {0x00BCU, 0x8300000FU},
    {0x00BDU, 0x83000012U},
    {0x00BEU, 0x83000015U},
    {0x00C0U, 0x02000018U},
    {0x00C1U, 0x0200001AU},
    {0x00C2U, 0x0200001CU},
    {0x00C3U, 0x0200001EU},
    {0x00C4U, 0x02000020U},
    {0x00C5U, 0x02000022U},
    {0x00C7U, 0x02000024U},
    {0x00C8U, 0x02000026U},
    {0x00C9U, 0x02000028U},
    {0x00CAU, 0x0200002AU},
    {0x00CBU, 0x0200002CU},
    {0x00CCU, 0x0200002EU},
    {0x00CDU, 0x02000030U},
    {0x00CEU, 0x02000032U},
    {0x00CFU, 0x02000034U},
    {0x00D1U, 0x02000036U},
    {0x00D2U, 0x02000038U},
    {0x00D3U, 0x0200003AU},
    {0x00D4U, 0x0200003CU},
    {0x00D5U, 0x0200003EU},
    {0x00D6U, 0x02000040U},
    {0x00D9U, 0x02000042U},
    {0x00DAU, 0x02000044U},
    {0x00DBU, 0x02000046U},
    {0x00DCU, 0x02000048U},
    {0x00DDU, 0x0200004AU},
    {0x00E0U, 0x0200004CU},
    {0x00E1U, 0x0200004EU},
    {0x00E2U, 0x02000050U},
    {0x00E3U, 0x02000052U},
    {0x00E4U, 0x02000054U},
    {0x00E5U, 0x02000056U},
    {0x00E7U, 0x02000058U},
    {0x00E8U, 0x0200005AU},
    {0x00E9U, 0x0200005CU},
    {0x00EAU, 0x0200005EU},
    {0x00EBU, 0x02000060U},
    {0x00ECU, 0x02000062U},
    {0x00EDU, 0x02000064U},
    {0x00EEU, 0x02000066U},
    {0x00EFU, 0x02000068U},
    {0x00F1U, 0x0200006AU},
    {0x00F2U, 0x0200006CU},
    {0x00F3U, 0x0200006EU},
    {0x00F4U, 0x02000070U},
    {0x00F5U, 0x02000072U},
    {0x00F6U, 0x02000074U},
    {0x00F9U, 0x02000076U},
    {0x00FAU, 0x02000078U},
    {0x00FBU, 0x0200007AU},
    {0x00FCU, 0x0200007CU},
    {0x00FDU, 0x0200007EU},
    {0x00FFU, 0x02000080U},
    {0x0100U, 0x02000082U},
    {0x0101U, 0x02000084U},
    {0x0102U, 0x02000086U},
    {0x0103U, 0x02000088U},
    {0x0104U, 0x0200008AU},
    {0x0105U, 0x0200008CU},
    {0x0106U, 0x0200008EU},
    {0x0107U, 0x02000090U},
    {0x0108U, 0x02000092U},
    {0x0109U, 0x02000094U},
    {0x010AU, 0x02000096U},
    {0x010BU, 0x02000098U},
    {0x010CU, 0x0200009AU},
    {0x010DU, 0x0200009CU},
    {0x010EU, 0x0200009EU},
    {0x010FU, 0x020000A0U},
    {0x0112U, 0x020000A2U},
    {0x0113U, 0x020000A4U},
    {0x0114U, 0x020000A6U},
    {0x0115U, 0x020000A8U},
    {0x0116U, 0x020000AAU},
    {0x0117U, 0x020000ACU},
    {0x0118U, 0x020000AEU},
    {0x0119U, 0x020000B0U},
    {0x011AU, 0x020000B2U},
    {0x011BU, 0x020000B4U},
    {0x011CU, 0x020000B6U},
    {0x011DU, 0x020000B8U},
    {0x011EU, 0x020000BAU},
    {0x011FU, 0x020000BCU},
    {0x0120U, 0x020000BEU},
    {0x0121U, 0x020000C0U},
    {0x0122U, 0x020000C2U},
    {0x0123U, 0x020000C4U},
    {0x0124U, 0x020000C6U},
    {0x0125U, 0x020000C8U},
    {0x0128U, 0x020000CAU},
    {0x0129U, 0x020000CCU},
    {0x012AU, 0x020000CEU},
    {0x012BU, 0x020000D0U},
    {0x012CU, 0x020000D2U},
    {0x012DU, 0x020000D4U},
    {0x012EU, 0x020000D6U},
    {0x012FU, 0x020000D8U},
    {0x0130U, 0x020000DAU},
    {0x0132U, 0x820000DCU},
    {0x0133U, 0x820000DEU},
    {0x0134U, 0x020000E0U},
    {0x0135U, 0x020000E2U},
    {0x0136U, 0x020000E4U},
    {0x0137U, 0x020000E6U},
    {0x0139U, 0x020000E8U},
    {0x013AU, 0x020000EAU},
    {0x013BU, 0x020000ECU},
    {0x013CU, 0x020000EEU},
    {0x013DU, 0x020000F0U},
    {0x013EU, 0x020000F2U},
    {0x013FU, 0x820000F4U},
    {0x0140U, 0x820000F6U},
    {0x0143U, 0x020000F8U},
    {0x0144U, 0x020000FAU},
    {0x0145U, 0x020000FCU},
    {0x0146U, 0x020000FEU},
    {0x0147U, 0x02000100U},
    {0x0148U, 0x02000102U},
    {0x0149U, 0x82000104U},
    {0x014CU, 0x02000106U},
    {0x014DU, 0x02000108U},
    {0x014EU, 0x0200010AU},
    {0x014FU, 0x0200010CU},
    {0x0150U, 0x0200010EU},
    {0x0151U, 0x02000110U},
    {0x0154U, 0x02000112U},
    {0x0155U, 0x02000114U},
    {0x0156U, 0x02000116U},
    {0x0157U, 0x02000118U},
    {0x0158U, 0x0200011AU},
    {0x0159U, 0x0200011CU},
    {0x015AU, 0x0200011EU},
    {0x015BU, 0x02000120U},
    {0x015CU, 0x02000122U},
    {0x015DU, 0x02000124U},
    {0x015EU, 0x02000126U},
    {0x015FU, 0x02000128U},
    {0x0160U, 0x0200012AU},
    {0x0161U, 0x0200012CU},
    {0x0162U, 0x0200012EU},
    {0x0163U, 0x02000130U},
    {0x0164U, 0x02000132U},
    {0x0165U, 0x02000134U},
    {0x0168U, 0x02000136U},
    {0x0169U, 0x02000138U},
    {0x016AU, 0x0200013AU},
    {0x016BU, 0x0200013CU},
    {0x016CU, 0x0200013EU},
    {0x016DU, 0x02000140U},
    {0x016EU, 0x02000142U},
    {0x016FU, 0x02000144U},
    {0x0170U, 0x02000146U},
    {0x0171U, 0x02000148U},
    {0x0172U, 0x0200014AU},
    {0x0173U, 0x0200014CU},
    {0x0174U, 0x0200014EU},
    {0x0175U, 0x02000150U},
    {0x0176U, 0x02000152U},
    {0x0177U, 0x02000154U},
    {0x0178U, 0x02000156U},
    {0x0179U, 0x02000158U},
    {0x017AU, 0x0200015AU},
    {0x017BU, 0x0200015CU},
    {0x017CU, 0x0200015EU},
    {0x017DU, 0x02000160U},
    {0x017EU, 0x02000162U},
    {0x017FU, 0x81000164U},
    {0x01A0U, 0x02000165U},
    {0x01A1U, 0x02000167U},
    {0x01AFU, 0x02000169U},
    {0x01B0U, 0x0200016BU},
    {0x01C4U, 0x8200016DU},
    {0x01C5U, 0x8200016FU},
    {0x01C6U, 0x82000171U},
    {0x01C7U, 0x82000173U},
    {0x01C8U, 0x82000175U},
    {0x01C9U, 0x82000177U},
    {0x01CAU, 0x82000179U},
    {0x01CBU, 0x8200017BU},
    {0x01CCU, 0x8200017DU},
    {0x01CDU, 0x0200017FU},
    {0x01CEU, 0x02000181U},
    {0x01CFU, 0x02000183U},
    {0x01D0U, 0x02000185U},
    {0x01D1U, 0x02000187U},
    {0x01D2U, 0x02000189U},
    {0x01D3U, 0x0200018BU},
    {0x01D4U, 0x0200018DU},
    {0x01D5U, 0x0200018FU},
    {0x01D6U, 0x02000191U},
    {0x01D7U, 0x02000193U},
    {0x01D8U, 0x02000195U},
    {0x01D9U, 0x02000197U},
    {0x01DAU, 0x02000199U},
    {0x01DBU, 0x0200019BU},
    {0x01DCU, 0x0200019DU},
    {0x01DEU, 0x0200019FU},
    {0x01DFU, 0x020001A1U},
    {0x01E0U, 0x020001A3U},
    {0x01E1U, 0x020001A5U},
    {0x01E2U, 0x020001A7U},
    {0x01E3U, 0x020001A9U},
    {0x01E6U, 0x020001ABU},
    {0x01E7U, 0x020001ADU},
    {0x01E8U, 0x020001AFU},
    {0x01E9U, 0x020001B1U},
    {0x01EAU, 0x020001B3U},
    {0x01EBU, 0x020001B5U},
    {0x01ECU, 0x020001B7U},
    {0x01EDU, 0x020001B9U},
    {0x01EEU, 0x020001BBU},
    {0x01EFU, 0x020001BDU},
    {0x01F0U, 0x020001BFU},
    {0x01F1U, 0x820001C1U},
    {0x01F2U, 0x820001C3U},
    {0x01F3U, 0x820001C5U},
    {0x01F4U, 0x020001C7U},
    {0x01F5U, 0x020001C9U},
    {0x01F8U, 0x020001CBU},
    {0x01F9U, 0x020001CDU},
    {0x01FAU, 0x020001CFU},
    {0x01FBU, 0x020001D1U},
    {0x01FCU, 0x020001D3U},
    {0x01FDU, 0x020001D5U},
    {0x01FEU, 0x020001D7U},
    {0x01FFU, 0x020001D9U},
    {0x0200U, 0x020001DBU},
    {0x0201U, 0x020001DDU},
    {0x0202U, 0x020001DFU},
    {0x0203U, 0x020001E1U},
    {0x0204U, 0x020001E3U},
    {0x0205U, 0x020001E5U},
    {0x0206U, 0x020001E7U},
    {0x0207U, 0x020001E9U},
    {0x0208U, 0x020001EBU},
    {0x0209U, 0x020001EDU},
    {0x020AU, 0x020001EFU},
    {0x020BU, 0x020001F1U},
    {0x020CU, 0x020001F3U},
    {0x020DU, 0x020001F5U},
    {0x020EU, 0x020001F7U},
    {0x020FU, 0x020001F9U},
    {0x0210U, 0x020001FBU},
    {0x0211U, 0x020001FDU},
    {0x0212U, 0x020001FFU},
    {0x0213U, 0x02000201U},
    {0x0214U, 0x02000203U},
    {0x0215U, 0x02000205U},
    {0x0216U, 0x02000207U},
    {0x0217U, 0x02000209U},
    {0x0218U, 0x0200020BU},
    {0x0219U, 0x0200020DU},
    {0x021AU, 0x0200020FU},
    {0x021BU, 0x02000211U},
    {0x021EU, 0x02000213U},
    {0x021FU, 0x02000215U},
    {0x0226U, 0x02000217U},
    {0x0227U, 0x02000219U},
    {0x0228U, 0x0200021BU},
    {0x0229U, 0x0200021DU},
    {0x022AU, 0x0200021FU},
    {0x022BU, 0x02000221U},
    {0x022CU, 0x02000223U},
    {0x022DU, 0x02000225U},
    {0x022EU, 0x02000227U},
    {0x022FU, 0x02000229U},
    {0x0230U, 0x0200022BU},
    {0x0231U, 0x0200022DU},
    {0x0232U, 0x0200022FU},
    {0x0233U, 0x02000231U},
    {0x02B0U, 0x81000233U},
    {0x02B1U, 0x81000234U},
    {0x02B2U, 0x81000235U},
    {0x02B3U, 0x81000236U},
    {0x02B4U, 0x81000237U},
    {0x02B5U, 0x81000238U},
    {0x02B6U, 0x81000239U},
    {0x02B7U, 0x8100023AU},
    {0x02B8U, 0x8100023BU},
    {0x02D8U, 0x8200023CU},
    {0x02D9U, 0x8200023EU},
    {0x02DAU, 0x82000240U},
    {0x02DBU, 0x82000242U},
    {0x02DCU, 0x82000244U},
    {0x02DDU, 0x82000246U},
    {0x02E0U, 0x81000248U},
    {0x02E1U, 0x81000249U},
    {0x02E2U, 0x81000164U},
    {0x02E3U, 0x8100024AU},
    {0x02E4U, 0x8100024BU},
    {0x0340U, 0x0100024CU},
    {0x0341U, 0x0100024DU},
    {0x0343U, 0x0100024EU},
    {0x0344U, 0x0200024FU},
    {0x0374U, 0x01000251U},
    {0x037AU, 0x82000252U},
    {0x037EU, 0x01000254U},
    {0x0384U, 0x82000008U},
    {0x0385U, 0x02000255U},
    {0x0386U, 0x02000257U},
    {0x0387U, 0x01000259U},
    {0x0388U, 0x0200025AU},
    {0x0389U, 0x0200025CU},
    {0x038AU, 0x0200025EU},
    {0x038CU, 0x02000260U},
    {0x038EU, 0x02000262U},
    {0x038FU, 0x02000264U},
    {0x0390U, 0x02000266U},
    {0x03AAU, 0x02000268U},
    {0x03ABU, 0x0200026AU},
    {0x03ACU, 0x0200026CU},
    {0x03ADU, 0x0200026EU},
    {0x03AEU, 0x02000270U},
    {0x03AFU, 0x02000272U},
    {0x03B0U, 0x02000274U},
    {0x03CAU, 0x02000276U},
    {0x03CBU, 0x02000278U},
    {0x03CCU, 0x0200027AU},
    {0x03CDU, 0x0200027CU},
    {0x03CEU, 0x0200027EU},
    {0x03D0U, 0x81000280U},
    {0x03D1U, 0x81000281U},
    {0x03D2U, 0x81000282U},
    {0x03D3U, 0x02000283U},
    {0x03D4U, 0x02000285U},
    {0x03D5U, 0x81000287U},
    {0x03D6U, 0x81000288U},
    {0x03F0U, 0x81000289U},
    {0x03F1U, 0x8100028AU},
    {0x03F2U, 0x8100028BU},
    {0x03F4U, 0x8100028CU},
    {0x03F5U, 0x8100028DU},
    {0x03F9U, 0x8100028EU},
    {0x0400U, 0x0200028FU},
    {0x0401U, 0x02000291U},
    {0x0403U, 0x02000293U},
    {0x0407U, 0x02000295U},
    {0x040CU, 0x02000297U},
    {0x040DU, 0x02000299U},
    {0x040EU, 0x0200029BU},
    {0x0419U, 0x0200029DU},
    {0x0439U, 0x0200029FU},
    {0x0450U, 0x020002A1U},
    {0x0451U, 0x020002A3U},
    {0x0453U, 0x020002A5U},
    {0x0457U, 0x020002A7U},
    {0x045CU, 0x020002A9U},
    {0x045DU, 0x020002ABU},
    {0x045EU, 0x020002ADU},
    {0x0476U, 0x020002AFU},
    {0x0477U, 0x020002B1U},
    {0x04C1U, 0x020002B3U},
    {0x04C2U, 0x020002B5U},
    {0x04D0U, 0x020002B7U},
    {0x04D1U, 0x020002B9U},
    {0x04D2U, 0x020002BBU},
    {0x04D3U, 0x020002BDU},
    {0x04D6U, 0x020002BFU},
    {0x04D7U, 0x020002C1U},
    {0x04DAU, 0x020002C3U},
    {0x04DBU, 0x020002C5U},
    {0x04DCU, 0x020002C7U},
    {0x04DDU, 0x020002C9U},
    {0x04DEU, 0x020002CBU},
    {0x04DFU, 0x020002CDU},
    {0x04E2U, 0x020002CFU},
    {0x04E3U, 0x020002D1U},
    {0x04E4U, 0x020002D3U},
    {0x04E5U, 0x020002D5U},
    {0x04E6U, 0x020002D7U},
    {0x04E7U, 0x020002D9U},
    {0x04EAU, 0x020002DBU},
    {0x04EBU, 0x020002DDU},
    {0x04ECU, 0x020002DFU},
    {0x04EDU, 0x020002E1U},
    {0x04EEU, 0x020002E3U},
    {0x04EFU, 0x020002E5U},
    {0x04F0U, 0x020002E7U},
    {0x04F1U, 0x020002E9U},
    {0x04F2U, 0x020002EBU},
    {0x04F3U, 0x020002EDU},
    {0x04F4U, 0x020002EFU},
    {0x04F5U, 0x020002F1U},
    {0x04F8U, 0x020002F3U},
    {0x04F9U, 0x020002F5U},
    {0x0587U, 0x820002F7U},
    {0x0622U, 0x020002F9U},
    {0x0623U, 0x020002FBU},
    {0x0624U, 0x020002FDU},
    {0x0625U, 0x020002FFU},
    {0x0626U, 0x02000301U},
    {0x0675U, 0x82000303U},
    {0x0676U, 0x82000305U},
    {0x0677U, 0x82000307U},
    {0x0678U, 0x82000309U},
    {0x06C0U, 0x0200030BU},
    {0x06C2U, 0x0200030DU},
    {0x06D3U, 0x0200030FU},
    {0x0929U, 0x02000311U},
    {0x0931U, 0x02000313U},
    {0x0934U, 0x02000315U},
    {0x0958U, 0x02000317U},
    {0x0959U, 0x02000319U},
    {0x095AU, 0x0200031BU},
    {0x095BU, 0x0200031DU},
    {0x095CU, 0x0200031FU},
    {0x095DU, 0x02000321U},
    {0x095EU, 0x02000323U},
    {0x095FU, 0x02000325U},
    {0x09CBU, 0x02000327U},
    {0x09CCU, 0x02000329U},
    {0x09DCU, 0x0200032BU},
    {0x09DDU, 0x0200032DU},
    {0x09DFU, 0x0200032FU},
    {0x0A33U, 0x02000331U},
    {0x0A36U, 0x02000333U},
    {0x0A59U, 0x02000335U},
    {0x0A5AU, 0x02000337U},
    {0x0A5BU, 0x02000339U},
    {0x0A5EU, 0x0200033BU},
    {0x0B48U, 0x0200033DU},
    {0x0B4BU, 0x0200033FU},
    {0x0B4CU, 0x02000341U},
    {0x0B5CU, 0x02000343U},
    {0x0B5DU, 0x02000345U},
    {0x0B94U, 0x02000347U},
    {0x0BCAU, 0x02000349U},
    {0x0BCBU, 0x0200034BU},
    {0x0BCCU, 0x0200034DU},
    {0x0C48U, 0x0200034FU},
    {0x0CC0U, 0x02000351U},
    {0x0CC7U, 0x02000353U},
    {0x0CC8U, 0x02000355U},
    {0x0CCAU, 0x02000357U},
    {0x0CCBU, 0x02000359U},
    {0x0D4AU, 0x0200035BU},
    {0x0D4BU, 0x0200035DU},
    {0x0D4CU, 0x0200035FU},
    {0x0DDAU, 0x02000361U},
    {0x0DDCU, 0x02000363U},
    {0x0DDDU, 0x02000365U},
    {0x0DDEU, 0x02000367U},
    {0x0E33U, 0x82000369U},
    {0x0EB3U, 0x8200036BU},
    {0x0EDCU, 0x8200036DU},
    {0x0EDDU, 0x8200036FU},
    {0x0F0CU, 0x81000371U},
    {0x0F43U, 0x02000372U},
    {0x0F4DU, 0x02000374U},
    {0x0F52U, 0x02000376U},
    {0x0F57U, 0x02000378U},
    {0x0F5CU, 0x0200037AU},
    {0x0F69U, 0x0200037CU},
    {0x0F73U, 0x0200037EU},
    {0x0F75U, 0x02000380U},
    {0x0F76U, 0x02000382U},
    {0x0F77U, 0x82000384U},
    {0x0F78U, 0x02000386U},
    {0x0F79U, 0x82000388U},
    {0x0F81U, 0x0200038AU},
    {0x0F93U, 0x0200038CU},
    {0x0F9DU, 0x0200038EU},
    {0x0FA2U, 0x02000390U},
    {0x0FA7U, 0x02000392U},
    {0x0FACU, 0x02000394U},
    {0x0FB9U, 0x02000396U},
    {0x1026U, 0x02000398U},
    {0x10FCU, 0x8100039AU},
    {0x1B06U, 0x0200039BU},
    {0x1B08U, 0x0200039DU},
    {0x1B0AU, 0x0200039FU},
    {0x1B0CU, 0x020003A1U},
    {0x1B0EU, 0x020003A3U},
    {0x1B12U, 0x020003A5U},
    {0x1B3BU, 0x020003A7U},
    {0x1B3DU, 0x020003A9U},
    {0x1B40U, 0x020003ABU},
    {0x1B41U, 0x020003ADU},
    {0x1B43U, 0x020003AFU},
    {0x1D2CU, 0x810003B1U},
    {0x1D2DU, 0x810003B2U},
    {0x1D2EU, 0x810003B3U},
    {0x1D30U, 0x810003B4U},
    {0x1D31U, 0x810003B5U},
    {0x1D32U, 0x810003B6U},
    {0x1D33U, 0x810003B7U},
    {0x1D34U, 0x810003B8U},
    {0x1D35U, 0x810003B9U},
    {0x1D36U, 0x810003BAU},
    {0x1D37U, 0x810003BBU},
    {0x1D38U, 0x810003BCU},
    {0x1D39U, 0x810003BDU},
    {0x1D3AU, 0x810003BEU},
    {0x1D3CU, 0x810003BFU},
    {0x1D3DU, 0x810003C0U},
    {0x1D3EU, 0x810003C1U},
    {0x1D3FU, 0x810003C2U},
    {0x1D40U, 0x810003C3U},
    {0x1D41U, 0x810003C4U},
    {0x1D42U, 0x810003C5U},
    {0x1D43U, 0x81000003U},
    {0x1D44U, 0x810003C6U},
    {0x1D45U, 0x810003C7U},
    {0x1D46U, 0x810003C8U},
    {0x1D47U, 0x810003C9U},
    {0x1D48U, 0x810003CAU},
    {0x1D49U, 0x810003CBU},
    {0x1D4AU, 0x810003CCU},
    {0x1D4BU, 0x810003CDU},
    {0x1D4CU, 0x810003CEU},
    {0x1D4DU, 0x810003CFU},
    {0x1D4FU, 0x810003D0U},
    {0x1D50U, 0x810003D1U},
    {0x1D51U, 0x810003D2U},
    {0x1D52U, 0x8100000EU},
    {0x1D53U, 0x810003D3U},
    {0x1D54U, 0x810003D4U},
    {0x1D55U, 0x810003D5U},
    {0x1D56U, 0x810003D6U},
    {0x1D57U, 0x810003D7U},
    {0x1D58U, 0x810003D8U},
    {0x1D59U, 0x810003D9U},
    {0x1D5AU, 0x810003DAU},
    {0x1D5BU, 0x810003DBU},
    {0x1D5CU, 0x810003DCU},
    {0x1D5DU, 0x81000280U},
    {0x1D5EU, 0x810003DDU},
    {0x1D5FU, 0x810003DEU},
    {0x1D60U, 0x81000287U},
    {0x1D61U, 0x810003DFU},
    {0x1D62U, 0x810003E0U},
    {0x1D63U, 0x81000236U},
    {0x1D64U, 0x810003D8U},
    {0x1D65U, 0x810003DBU},
    {0x1D66U, 0x81000280U},
    {0x1D67U, 0x810003DDU},
    {0x1D68U, 0x8100028AU},
    {0x1D69U, 0x81000287U},
    {0x1D6AU, 0x810003DFU},
    {0x1D78U, 0x810003E1U},
    {0x1D9BU, 0x810003E2U},
    {0x1D9CU, 0x810003E3U},
    {0x1D9DU, 0x810003E4U},
    {0x1D9EU, 0x810003E5U},
    {0x1D9FU, 0x810003CEU},
    {0x1DA0U, 0x810003E6U},
    {0x1DA1U, 0x810003E7U},
    {0x1DA2U, 0x810003E8U},
    {0x1DA3U, 0x810003E9U},
    {0x1DA4U, 0x810003EAU},
    {0x1DA5U, 0x810003EBU},
    {0x1DA6U, 0x810003ECU},
    {0x1DA7U, 0x810003EDU},
    {0x1DA8U, 0x810003EEU},
    {0x1DA9U, 0x810003EFU},
    {0x1DAAU, 0x810003F0U},
    {0x1DABU, 0x810003F1U},
    {0x1DACU, 0x810003F2U},
    {0x1DADU, 0x810003F3U},
    {0x1DAEU, 0x810003F4U},
    {0x1DAFU, 0x810003F5U},
    {0x1DB0U, 0x810003F6U},
    {0x1DB1U, 0x810003F7U},
    {0x1DB2U, 0x810003F8U},
    {0x1DB3U, 0x810003F9U},
    {0x1DB4U, 0x810003FAU},
    {0x1DB5U, 0x810003FBU},
    {0x1DB6U, 0x810003FCU},
    {0x1DB7U, 0x810003FDU},
    {0x1DB8U, 0x810003FEU},
    {0x1DB9U, 0x810003FFU},
    {0x1DBAU, 0x81000400U},
    {0x1DBBU, 0x81000401U},
    {0x1DBCU, 0x81000402U},
    {0x1DBDU, 0x81000403U},
    {0x1DBEU, 0x81000404U},
    {0x1DBFU, 0x81000281U},
    {0x1E00U, 0x02000405U},
    {0x1E01U, 0x02000407U},
    {0x1E02U, 0x02000409U},
    {0x1E03U, 0x0200040BU},
    {0x1E04U, 0x0200040DU},
    {0x1E05U, 0x0200040FU},
    {0x1E06U, 0x02000411U},
    {0x1E07U, 0x02000413U},
    {0x1E08U, 0x02000415U},
    {0x1E09U, 0x02000417U},
    {0x1E0AU, 0x02000419U},
    {0x1E0BU, 0x0200041BU},
    {0x1E0CU, 0x0200041DU},
    {0x1E0DU, 0x0200041FU},
    {0x1E0EU, 0x02000421U},
    {0x1E0FU, 0x02000423U},
    {0x1E10U, 0x02000425U},
    {0x1E11U, 0x02000427U},
    {0x1E12U, 0x02000429U},
    {0x1E13U, 0x0200042BU},
    {0x1E14U, 0x0200042DU},
    {0x1E15U, 0x0200042FU},
    {0x1E16U, 0x02000431U},
    {0x1E17U, 0x02000433U},
    {0x1E18U, 0x02000435U},
    {0x1E19U, 0x02000437U},
    {0x1E1AU, 0x02000439U},
    {0x1E1BU, 0x0200043BU},
    {0x1E1CU, 0x0200043DU},
    {0x1E1DU, 0x0200043FU},
    {0x1E1EU, 0x02000441U},
    {0x1E1FU, 0x02000443U},
    {0x1E20U, 0x02000445U},
    {0x1E21U, 0x02000447U},
    {0x1E22U, 0x02000449U},
    {0x1E23U, 0x0200044BU},
    {0x1E24U, 0x0200044DU},
    {0x1E25U, 0x0200044FU},
    {0x1E26U, 0x02000451U},
    {0x1E27U, 0x02000453U},
    {0x1E28U, 0x02000455U},
    {0x1E29U, 0x02000457U},
    {0x1E2AU, 0x02000459U},
    {0x1E2BU, 0x0200045BU},
    {0x1E2CU, 0x0200045DU},
    {0x1E2DU, 0x0200045FU},
    {0x1E2EU, 0x02000461U},
    {0x1E2FU, 0x02000463U},
    {0x1E30U, 0x02000465U},
    {0x1E31U, 0x02000467U},
    {0x1E32U, 0x02000469U},
    {0x1E33U, 0x0200046BU},
    {0x1E34U, 0x0200046DU},
    {0x1E35U, 0x0200046FU},
    {0x1E36U, 0x02000471U},
    {0x1E37U, 0x02000473U},
    {0x1E38U, 0x02000475U},
    {0x1E39U, 0x02000477U},
    {0x1E3AU, 0x02000479U},
    {0x1E3BU, 0x0200047BU},
    {0x1E3CU, 0x0200047DU},
    {0x1E3DU, 0x0200047FU},
    {0x1E3EU, 0x02000481U},
    {0x1E3FU, 0x02000483U},
    {0x1E40U, 0x02000485U},
    {0x1E41U, 0x02000487U},
    {0x1E42U, 0x02000489U},
    {0x1E43U, 0x0200048BU},
    {0x1E44U, 0x0200048DU},
    {0x1E45U, 0x0200048FU},
    {0x1E46U, 0x02000491U},
    {0x1E47U, 0x02000493U},
    {0x1E48U, 0x02000495U},
    {0x1E49U, 0x02000497U},
    {0x1E4AU, 0x02000499U},
    {0x1E4BU, 0x0200049BU},
    {0x1E4CU, 0x0200049DU},
    {0x1E4DU, 0x0200049FU},
    {0x1E4EU, 0x020004A1U},
    {0x1E4FU, 0x020004A3U},
    {0x1E50U, 0x020004A5U},
    {0x1E51U, 0x020004A7U},
    {0x1E52U, 0x020004A9U},
    {0x1E53U, 0x020004ABU},
    {0x1E54U, 0x020004ADU},
    {0x1E55U, 0x020004AFU},
    {0x1E56U, 0x020004B1U},
    {0x1E57U, 0x020004B3U},
    {0x1E58U, 0x020004B5U},
    {0x1E59U, 0x020004B7U},
    {0x1E5AU, 0x020004B9U},
    {0x1E5BU, 0x020004BBU},
    {0x1E5CU, 0x020004BDU},
    {0x1E5DU, 0x020004BFU},
    {0x1E5EU, 0x020004C1U},
    {0x1E5FU, 0x020004C3U},
    {0x1E60U, 0x020004C5U},
    {0x1E61U, 0x020004C7U},
    {0x1E62U, 0x020004C9U},
    {0x1E63U, 0x020004CBU},
    {0x1E64U, 0x020004CDU},
    {0x1E65U, 0x020004CFU},
    {0x1E66U, 0x020004D1U},
    {0x1E67U, 0x020004D3U},
    {0x1E68U, 0x020004D5U},
    {0x1E69U, 0x020004D7U},
    {0x1E6AU, 0x020004D9U},
    {0x1E6BU, 0x020004DBU},
    {0x1E6CU, 0x020004DDU},
    {0x1E6DU, 0x020004DFU},
    {0x1E6EU, 0x020004E1U},
    {0x1E6FU, 0x020004E3U},
    {0x1E70U, 0x020004E5U},
    {0x1E71U, 0x020004E7U},
    {0x1E72U, 0x020004E9U},
    {0x1E73U, 0x020004EBU},
    {0x1E74U, 0x020004EDU},
    {0x1E75U, 0x020004EFU},
    {0x1E76U, 0x020004F1U},
    {0x1E77U, 0x020004F3U},
    {0x1E78U, 0x020004F5U},
    {0x1E79U, 0x020004F7U},
    {0x1E7AU, 0x020004F9U},
    {0x1E7BU, 0x020004FBU},
    {0x1E7CU, 0x020004FDU},
    {0x1E7DU, 0x020004FFU},
    {0x1E7EU, 0x02000501U},
    {0x1E7FU, 0x02000503U},
    {0x1E80U, 0x02000505U},
    {0x1E81U, 0x02000507U},
    {0x1E82U, 0x02000509U},
    {0x1E83U, 0x0200050BU},
    {0x1E84U, 0x0200050DU},
    {0x1E85U, 0x0200050FU},
    {0x1E86U, 0x02000511U},
    {0x1E87U, 0x02000513U},
    {0x1E88U, 0x02000515U},
    {0x1E89U, 0x02000517U},
    {0x1E8AU, 0x02000519U},
    {0x1E8BU, 0x0200051BU},
    {0x1E8CU, 0x0200051DU},
    {0x1E8DU, 0x0200051FU},
    {0x1E8EU, 0x02000521U},
    {0x1E8FU, 0x02000523U},
    {0x1E90U, 0x02000525U},
    {0x1E91U, 0x02000527U},
    {0x1E92U, 0x02000529U},
    {0x1E93U, 0x0200052BU},
    {0x1E94U, 0x0200052DU},
    {0x1E95U, 0x0200052FU},
    {0x1E96U, 0x02000531U},
    {0x1E97U, 0x02000533U},
    {0x1E98U, 0x02000535U},
    {0x1E99U, 0x02000537U},
    {0x1E9AU, 0x82000539U},
    {0x1E9BU, 0x0200053BU},
    {0x1EA0U, 0x0200053DU},
    {0x1EA1U, 0x0200053FU},
    {0x1EA2U, 0x02000541U},
    {0x1EA3U, 0x02000543U},
    {0x1EA4U, 0x02000545U},
    {0x1EA5U, 0x02000547U},
    {0x1EA6U, 0x02000549U},
    {0x1EA7U, 0x0200054BU},
    {0x1EA8U, 0x0200054DU},
    {0x1EA9U, 0x0200054FU},
    {0x1EAAU, 0x02000551U},
    {0x1EABU, 0x02000553U},
    {0x1EACU, 0x02000555U},
    {0x1EADU, 0x02000557U},
    {0x1EAEU, 0x02000559U},
    {0x1EAFU, 0x0200055BU},
    {0x1EB0U, 0x0200055DU},
    {0x1EB1U, 0x0200055FU},
    {0x1EB2U, 0x02000561U},
    {0x1EB3U, 0x02000563U},
    {0x1EB4U, 0x02000565U},
    {0x1EB5U, 0x02000567U},
    {0x1EB6U, 0x02000569U},
    {0x1EB7U, 0x0200056BU},
    {0x1EB8U, 0x0200056DU},
    {0x1EB9U, 0x0200056FU},
    {0x1EBAU, 0x02000571U},
    {0x1EBBU, 0x02000573U},
    {0x1EBCU, 0x02000575U},
    {0x1EBDU, 0x02000577U},
    {0x1EBEU, 0x02000579U},
    {0x1EBFU, 0x0200057BU},
    {0x1EC0U, 0x0200057DU},
    {0x1EC1U, 0x0200057FU},
    {0x1EC2U, 0x02000581U},
    {0x1EC3U, 0x02000583U},
    {0x1EC4U, 0x02000585U},
    {0x1EC5U, 0x02000587U},
    {0x1EC6U, 0x02000589U},
    {0x1EC7U, 0x0200058BU},
    {0x1EC8U, 0x0200058DU},
    {0x1EC9U, 0x0200058FU},
    {0x1ECAU, 0x02000591U},
    {0x1ECBU, 0x02000593U},
    {0x1ECCU, 0x02000595U},
    {0x1ECDU, 0x02000597U},
    {0x1ECEU, 0x02000599U},
    {0x1ECFU, 0x0200059BU},
    {0x1ED0U, 0x0200059DU},
    {0x1ED1U, 0x0200059FU},
    {0x1ED2U, 0x020005A1U},
    {0x1ED3U, 0x020005A3U},
    {0x1ED4U, 0x020005A5U},
    {0x1ED5U, 0x020005A7U},
    {0x1ED6U, 0x020005A9U},
    {0x1ED7U, 0x020005ABU},
    {0x1ED8U, 0x020005ADU},
    {0x1ED9U, 0x020005AFU},
    {0x1EDAU, 0x020005B1U},
    {0x1EDBU, 0x020005B3U},
    {0x1EDCU, 0x020005B5U},
    {0x1EDDU, 0x020005B7U},
    {0x1EDEU, 0x020005B9U},
    {0x1EDFU, 0x020005BBU},
    {0x1EE0U, 0x020005BDU},
    {0x1EE1U, 0x020005BFU},
    {0x1EE2U, 0x020005C1U},
    {0x1EE3U, 0x020005C3U},
    {0x1EE4U, 0x020005C5U},
    {0x1EE5U, 0x020005C7U},
    {0x1EE6U, 0x020005C9U},
    {0x1EE7U, 0x020005CBU},
    {0x1EE8U, 0x020005CDU},
    {0x1EE9U, 0x020005CFU},
    {0x1EEAU, 0x020005D1U},
    {0x1EEBU, 0x020005D3U},
    {0x1EECU, 0x020005D5U},
    {0x1EEDU, 0x020005D7U},
    {0x1EEEU, 0x020005D9U},
    {0x1EEFU, 0x020005DBU},
    {0x1EF0U, 0x020005DDU},
    {0x1EF1U, 0x020005DFU},
    {0x1EF2U, 0x020005E1U},
    {0x1EF3U, 0x020005E3U},
    {0x1EF4U, 0x020005E5U},
    {0x1EF5U, 0x020005E7U},
    {0x1EF6U, 0x020005E9U},
    {0x1EF7U, 0x020005EBU},
    {0x1EF8U, 0x020005EDU},
    {0x1EF9U, 0x020005EFU},
    {0x1F00U, 0x020005F1U},
    {0x1F01U, 0x020005F3U},
    {0x1F02U, 0x020005F5U},
    {0x1F03U, 0x020005F7U},
    {0x1F04U, 0x020005F9U},
    {0x1F05U, 0x020005FBU},
    {0x1F06U, 0x020005FDU},
    {0x1F07U, 0x020005FFU},
    {0x1F08U, 0x02000601U},
    {0x1F09U, 0x02000603U},
    {0x1F0AU, 0x02000605U},
    {0x1F0BU, 0x02000607U},
    {0x1F0CU, 0x02000609U},
    {0x1F0DU, 0x0200060BU},
    {0x1F0EU, 0x0200060DU},
    {0x1F0FU, 0x0200060FU},
    {0x1F10U, 0x02000611U},
    {0x1F11U, 0x02000613U},
    {0x1F12U, 0x02000615U},
    {0x1F13U, 0x02000617U},
    {0x1F14U, 0x02000619U},
    {0x1F15U, 0x0200061BU},
    {0x1F18U, 0x0200061DU},
    {0x1F19U, 0x0200061FU},
    {0x1F1AU, 0x02000621U},
    {0x1F1BU, 0x02000623U},
    {0x1F1CU, 0x02000625U},
    {0x1F1DU, 0x02000627U},
    {0x1F20U, 0x02000629U},
    {0x1F21U, 0x0200062BU},
    {0x1F22U, 0x0200062DU},
    {0x1F23U, 0x0200062FU},
    {0x1F24U, 0x02000631U},
    {0x1F25U, 0x02000633U},
    {0x1F26U, 0x02000635U},
    {0x1F27U, 0x02000637U},
    {0x1F28U, 0x02000639U},
    {0x1F29U, 0x0200063BU},
    {0x1F2AU, 0x0200063DU},
    {0x1F2BU, 0x0200063FU},
    {0x1F2CU, 0x02000641U},
    {0x1F2DU, 0x02000643U},
    {0x1F2EU, 0x02000645U},
    {0x1F2FU, 0x02000647U},
    {0x1F30U, 0x02000649U},
    {0x1F31U, 0x0200064BU},
    {0x1F32U, 0x0200064DU},
    {0x1F33U, 0x0200064FU},
    {0x1F34U, 0x02000651U},
    {0x1F35U, 0x02000653U},
    {0x1F36U, 0x02000655U},
    {0x1F37U, 0x02000657U},
    {0x1F38U, 0x02000659U},
    {0x1F39U, 0x0200065BU},
    {0x1F3AU, 0x0200065DU},
    {0x1F3BU, 0x0200065FU},
    {0x1F3CU, 0x02000661U},
    {0x1F3DU, 0x02000663U},
    {0x1F3EU, 0x02000665U},
    {0x1F3FU, 0x02000667U},
    {0x1F40U, 0x02000669U},
    {0x1F41U, 0x0200066BU},
    {0x1F42U, 0x0200066DU},
    {0x1F43U, 0x0200066FU},
    {0x1F44U, 0x02000671U},
    {0x1F45U, 0x02000673U},
    {0x1F48U, 0x02000675U},
    {0x1F49U, 0x02000677U},
    {0x1F4AU, 0x02000679U},
    {0x1F4BU, 0x0200067BU},
    {0x1F4CU, 0x0200067DU},
    {0x1F4DU, 0x0200067FU},
    {0x1F50U, 0x02000681U},
    {0x1F51U, 0x02000683U},
    {0x1F52U, 0x02000685U},
    {0x1F53U, 0x02000687U},
    {0x1F54U, 0x02000689U},
    {0x1F55U, 0x0200068BU},
    {0x1F56U, 0x0200068DU},
    {0x1F57U, 0x0200068FU},
    {0x1F59U, 0x02000691U},
    {0x1F5BU, 0x02000693U},
    {0x1F5DU, 0x02000695U},
    {0x1F5FU, 0x02000697U},
    {0x1F60U, 0x02000699U},
    {0x1F61U, 0x0200069BU},
    {0x1F62U, 0x0200069DU},
    {0x1F63U, 0x0200069FU},
    {0x1F64U, 0x020006A1U},
    {0x1F65U, 0x020006A3U},
    {0x1F66U, 0x020006A5U},
    {0x1F67U, 0x020006A7U},
    {0x1F68U, 0x020006A9U},
    {0x1F69U, 0x020006ABU},
    {0x1F6AU, 0x020006ADU},
    {0x1F6BU, 0x020006AFU},
    {0x1F6CU, 0x020006B1U},
    {0x1F6DU, 0x020006B3U},
    {0x1F6EU, 0x020006B5U},
    {0x1F6FU, 0x020006B7U},
    {0x1F70U, 0x020006B9U},
    {0x1F71U, 0x010006BBU},
    {0x1F72U, 0x020006BCU},
    {0x1F73U, 0x010006BEU},
    {0x1F74U, 0x020006BFU},
    {0x1F75U, 0x010006C1U},
    {0x1F76U, 0x020006C2U},
    {0x1F77U, 0x010006C4U},
    {0x1F78U, 0x020006C5U},
    {0x1F79U, 0x010006C7U},
    {0x1F7AU, 0x020006C8U},
    {0x1F7BU, 0x010006CAU},
    {0x1F7CU, 0x020006CBU},
    {0x1F7DU, 0x010006CDU},
    {0x1F80U, 0x020006CEU},
    {0x1F81U, 0x020006D0U},
    {0x1F82U, 0x020006D2U},
    {0x1F83U, 0x020006D4U},
    {0x1F84U, 0x020006D6U},
    {0x1F85U, 0x020006D8U},
    {0x1F86U, 0x020006DAU},
    {0x1F87U, 0x020006DCU},
    {0x1F88U, 0x020006DEU},
    {0x1F89U, 0x020006E0U},
    {0x1F8AU, 0x020006E2U},
    {0x1F8BU, 0x020006E4U},
    {0x1F8CU, 0x020006E6U},
    {0x1F8DU, 0x020006E8U},
    {0x1F8EU, 0x020006EAU},
    {0x1F8FU, 0x020006ECU},
    {0x1F90U, 0x020006EEU},
    {0x1F91U, 0x020006F0U},
    {0x1F92U, 0x020006F2U},
    {0x1F93U, 0x020006F4U},
    {0x1F94U, 0x020006F6U},
    {0x1F95U, 0x020006F8U},
    {0x1F96U, 0x020006FAU},
    {0x1F97U, 0x020006FCU},
    {0x1F98U, 0x020006FEU},
    {0x1F99U, 0x02000700U},
    {0x1F9AU, 0x02000702U},
    {0x1F9BU, 0x02000704U},
    {0x1F9CU, 0x02000706U},
    {0x1F9DU, 0x02000708U},
    {0x1F9EU, 0x0200070AU},
    {0x1F9FU, 0x0200070CU},
    {0x1FA0U, 0x0200070EU},
    {0x1FA1U, 0x02000710U},
    {0x1FA2U, 0x02000712U},
    {0x1FA3U, 0x02000714U},
    {0x1FA4U, 0x02000716U},
    {0x1FA5U, 0x02000718U},
    {0x1FA6U, 0x0200071AU},
    {0x1FA7U, 0x0200071CU},
    {0x1FA8U, 0x0200071EU},
    {0x1FA9U, 0x02000720U},
    {0x1FAAU, 0x02000722U},
    {0x1FABU, 0x02000724U},
    {0x1FACU, 0x02000726U},
    {0x1FADU, 0x02000728U},
    {0x1FAEU, 0x0200072AU},
    {0x1FAFU, 0x0200072CU},
    {0x1FB0U, 0x0200072EU},
    {0x1FB1U, 0x02000730U},
    {0x1FB2U, 0x02000732U},
    {0x1FB3U, 0x02000734U},
    {0x1FB4U, 0x02000736U},
    {0x1FB6U, 0x02000738U},
    {0x1FB7U, 0x0200073AU},
    {0x1FB8U, 0x0200073CU},
    {0x1FB9U, 0x0200073EU},
    {0x1FBAU, 0x02000740U},
    {0x1FBBU, 0x01000742U},
    {0x1FBCU, 0x02000743U},
    {0x1FBDU, 0x82000745U},
    {0x1FBEU, 0x01000747U},
    {0x1FBFU, 0x82000745U},
    {0x1FC0U, 0x82000748U},
    {0x1FC1U, 0x0200074AU},
    {0x1FC2U, 0x0200074CU},
    {0x1FC3U, 0x0200074EU},
    {0x1FC4U, 0x02000750U},
    {0x1FC6U, 0x02000752U},
    {0x1FC7U, 0x02000754U},
    {0x1FC8U, 0x02000756U},
    {0x1FC9U, 0x01000758U},
    {0x1FCAU, 0x02000759U},
    {0x1FCBU, 0x0100075BU},
    {0x1FCCU, 0x0200075CU},
    {0x1FCDU, 0x0200075EU},
    {0x1FCEU, 0x02000760U},
    {0x1FCFU, 0x02000762U},
    {0x1FD0U, 0x02000764U},
    {0x1FD1U, 0x02000766U},
    {0x1FD2U, 0x02000768U},
    {0x1FD3U, 0x0100076AU},
    {0x1FD6U, 0x0200076BU},
    {0x1FD7U, 0x0200076DU},
    {0x1FD8U, 0x0200076FU},
    {0x1FD9U, 0x02000771U},
    {0x1FDAU, 0x02000773U},
    {0x1FDBU, 0x01000775U},
    {0x1FDDU, 0x02000776U},
    {0x1FDEU, 0x02000778U},
    {0x1FDFU, 0x0200077AU},
    {0x1FE0U, 0x0200077CU},
    {0x1FE1U, 0x0200077EU},
    {0x1FE2U, 0x02000780U},
    {0x1FE3U, 0x01000782U},
    {0x1FE4U, 0x02000783U},
    {0x1FE5U, 0x02000785U},
    {0x1FE6U, 0x02000787U},
    {0x1FE7U, 0x02000789U},
    {0x1FE8U, 0x0200078BU},
    {0x1FE9U, 0x0200078DU},
    {0x1FEAU, 0x0200078FU},
    {0x1FEBU, 0x01000791U},
    {0x1FECU, 0x02000792U},
    {0x1FEDU, 0x02000794U},
    {0x1FEEU, 0x01000796U},
    {0x1FEFU, 0x01000797U},
    {0x1FF2U, 0x02000798U},
    {0x1FF3U, 0x0200079AU},
    {0x1FF4U, 0x0200079CU},
    {0x1FF6U, 0x0200079EU},
    {0x1FF7U, 0x020007A0U},
    {0x1FF8U, 0x020007A2U},
    {0x1FF9U, 0x010007A4U},
    {0x1FFAU, 0x020007A5U},
    {0x1FFBU, 0x010007A7U},
    {0x1FFCU, 0x020007A8U},
    {0x1FFDU, 0x010007AAU},
    {0x1FFEU, 0x820007ABU},
    {0x2000U, 0x010007ADU},
    {0x2001U, 0x010007AEU},
    {0x2002U, 0x81000000U},
    {0x2003U, 0x81000000U},
    {0x2004U, 0x81000000U},
    {0x2005U, 0x81000000U},
    {0x2006U, 0x81000000U},
    {0x2007U, 0x81000000U},
    {0x2008U, 0x81000000U},
    {0x2009U, 0x81000000U},
    {0x200AU, 0x81000000U},
    {0x2011U, 0x810007AFU},
    {0x2017U, 0x820007B0U},
    {0x2024U, 0x810007B2U},
    {0x2025U, 0x820007B3U},
    {0x2026U, 0x830007B5U},
    {0x202FU, 0x81000000U},
    {0x2033U, 0x820007B8U},
    {0x2034U, 0x830007BAU},
    {0x2036U, 0x820007BDU},
    {0x2037U, 0x830007BFU},
    {0x203CU, 0x820007C2U},
    {0x203EU, 0x820007C4U},
    {0x2047U, 0x820007C6U},
    {0x2048U, 0x820007C8U},
    {0x2049U, 0x820007CAU},
    {0x2057U, 0x840007CCU},
    {0x205FU, 0x81000000U},
    {0x2070U, 0x810007D0U},
    {0x2071U, 0x810003E0U},
    {0x2074U, 0x810007D1U},
    {0x2075U, 0x810007D2U},
    {0x2076U, 0x810007D3U},
    {0x2077U, 0x810007D4U},
    {0x2078U, 0x810007D5U},
    {0x2079U, 0x810007D6U},
    {0x207AU, 0x810007D7U},
    {0x207BU, 0x810007D8U},
    {0x207CU, 0x810007D9U},
    {0x207DU, 0x810007DAU},
    {0x207EU, 0x810007DBU},
    {0x207FU, 0x810007DCU},
    {0x2080U, 0x810007D0U},
    {0x2081U, 0x8100000DU},
    {0x2082U, 0x81000006U},
    {0x2083U, 0x81000007U},
    {0x2084U, 0x810007D1U},
    {0x2085U, 0x810007D2U},
    {0x2086U, 0x810007D3U},
    {0x2087U, 0x810007D4U},
    {0x2088U, 0x810007D5U},
    {0x2089U, 0x810007D6U},
    {0x208AU, 0x810007D7U},
    {0x208BU, 0x810007D8U},
    {0x208CU, 0x810007D9U},
    {0x208DU, 0x810007DAU},
    {0x208EU, 0x810007DBU},
    {0x2090U, 0x81000003U},
    {0x2091U, 0x810003CBU},
    {0x2092U, 0x8100000EU},
    {0x2093U, 0x8100024AU},
    {0x2094U, 0x810003CCU},
    {0x2095U, 0x81000233U},
    {0x2096U, 0x810003D0U},
    {0x2097U, 0x81000249U},
    {0x2098U, 0x810003D1U},
    {0x2099U, 0x810007DCU},
    {0x209AU, 0x810003D6U},
    {0x209BU, 0x81000164U},
    {0x209CU, 0x810003D7U},
    {0x20A8U, 0x820007DDU},
    {0x2100U, 0x830007DFU},
    {0x2101U, 0x830007E2U},
    {0x2102U, 0x810007E5U},
    {0x2103U, 0x820007E6U},
    {0x2105U, 0x830007E8U},
    {0x2106U, 0x830007EBU},
    {0x2107U, 0x810007EEU},
    {0x2109U, 0x820007EFU},
    {0x210AU, 0x810003CFU},
    {0x210BU, 0x810003B8U},
    {0x210CU, 0x810003B8U},
    {0x210DU, 0x810003B8U},
    {0x210EU, 0x81000233U},
    {0x210FU, 0x810007F1U},
    {0x2110U, 0x810003B9U},
    {0x2111U, 0x810003B9U},
    {0x2112U, 0x810003BCU},
    {0x2113U, 0x81000249U},
    {0x2115U, 0x810003BEU},
    {0x2116U, 0x820007F2U},
    {0x2119U, 0x810003C1U},
    {0x211AU, 0x810007F4U},
    {0x211BU, 0x810003C2U},
    {0x211CU, 0x810003C2U},
    {0x211DU, 0x810003C2U},
    {0x2120U, 0x820007F5U},
    {0x2121U, 0x830007F7U},
    {0x2122U, 0x820007FAU},
    {0x2124U, 0x810007FCU},
    {0x2126U, 0x010007FDU},
    {0x2128U, 0x810007FCU},
    {0x212AU, 0x010003BBU},
    {0x212BU, 0x010007FEU},
    {0x212CU, 0x810003B3U},
    {0x212DU, 0x810007E5U},
    {0x212FU, 0x810003CBU},
    {0x2130U, 0x810003B5U},
    {0x2131U, 0x810007FFU},
    {0x2133U, 0x810003BDU},
    {0x2134U, 0x8100000EU},
    {0x2135U, 0x81000800U},
    {0x2136U, 0x81000801U},
    {0x2137U, 0x81000802U},
    {0x2138U, 0x81000803U},
    {0x2139U, 0x810003E0U},
    {0x213BU, 0x83000804U},
    {0x213CU, 0x81000288U},
    {0x213DU, 0x810003DDU},
    {0x213EU, 0x81000807U},
    {0x213FU, 0x81000808U},
    {0x2140U, 0x81000809U},
    {0x2145U, 0x810003B4U},
    {0x2146U, 0x810003CAU},
    {0x2147U, 0x810003CBU},
    {0x2148U, 0x810003E0U},
    {0x2149U, 0x81000235U},
    {0x2150U, 0x8300080AU},
    {0x2151U, 0x8300080DU},
    {0x2152U, 0x84000810U},
    {0x2153U, 0x83000814U},
    {0x2154U, 0x83000817U},
    {0x2155U, 0x8300081AU},
    {0x2156U, 0x8300081DU},
    {0x2157U, 0x83000820U},
    {0x2158U, 0x83000823U},
    {0x2159U, 0x83000826U},
    {0x215AU, 0x83000829U},
    {0x215BU, 0x8300082CU},
    {0x215CU, 0x8300082FU},
    {0x215DU, 0x83000832U},
    {0x215EU, 0x83000835U},
    {0x215FU, 0x82000838U},
    {0x2160U, 0x810003B9U},
    {0x2161U, 0x8200083AU},
    {0x2162U, 0x8300083CU},
    {0x2163U, 0x8200083FU},
    {0x2164U, 0x81000841U},
    {0x2165U, 0x82000842U},
    {0x2166U, 0x83000844U},
    {0x2167U, 0x84000847U},
    {0x2168U, 0x8200084BU},
    {0x2169U, 0x8100084DU},
    {0x216AU, 0x8200084EU},
    {0x216BU, 0x83000850U},
    {0x216CU, 0x810003BCU},
    {0x216DU, 0x810007E5U},
    {0x216EU, 0x810003B4U},
    {0x216FU, 0x810003BDU},
    {0x2170U, 0x810003E0U},
    {0x2171U, 0x82000853U},
    {0x2172U, 0x83000855U},
    {0x2173U, 0x82000858U},
    {0x2174U, 0x810003DBU},
    {0x2175U, 0x8200085AU},
    {0x2176U, 0x8300085CU},
    {0x2177U, 0x8400085FU},
    {0x2178U, 0x82000863U},
    {0x2179U, 0x8100024AU},
    {0x217AU, 0x82000865U},
    {0x217BU, 0x83000867U},
    {0x217CU, 0x81000249U},
    {0x217DU, 0x810003E3U},
    {0x217EU, 0x810003CAU},
    {0x217FU, 0x810003D1U},
    {0x2189U, 0x8300086AU},
    {0x219AU, 0x0200086DU},
    {0x219BU, 0x0200086FU},
    {0x21AEU, 0x02000871U},
    {0x21CDU, 0x02000873U},
    {0x21CEU, 0x02000875U},
    {0x21CFU, 0x02000877U},
    {0x2204U, 0x02000879U},
    {0x2209U, 0x0200087BU},
    {0x220CU, 0x0200087DU},
    {0x2224U, 0x0200087FU},
    {0x2226U, 0x02000881U},
    {0x222CU, 0x82000883U},
    {0x222DU, 0x83000885U},
    {0x222FU, 0x82000888U},
    {0x2230U, 0x8300088AU},
    {0x2241U, 0x0200088DU},
    {0x2244U, 0x0200088FU},
    {0x2247U, 0x02000891U},
    {0x2249U, 0x02000893U},
    {0x2260U, 0x02000895U},
    {0x2262U, 0x02000897U},
    {0x226DU, 0x02000899U},
    {0x226EU, 0x0200089BU},
    {0x226FU, 0x0200089DU},
    {0x2270U, 0x0200089FU},
    {0x2271U, 0x020008A1U},
    {0x2274U, 0x020008A3U},
    {0x2275U, 0x020008A5U},
    {0x2278U, 0x020008A7U},
    {0x2279U, 0x020008A9U},
    {0x2280U, 0x020008ABU},
    {0x2281U, 0x020008ADU},
    {0x2284U, 0x020008AFU},
    {0x2285U, 0x020008B1U},
    {0x2288U, 0x020008B3U},
    {0x2289U, 0x020008B5U},
    {0x22ACU, 0x020008B7U},
    {0x22ADU, 0x020008B9U},
    {0x22AEU, 0x020008BBU},
    {0x22AFU, 0x020008BDU},
    {0x22E0U, 0x020008BFU},
    {0x22E1U, 0x020008C1U},
    {0x22E2U, 0x020008C3U},
    {0x22E3U, 0x020008C5U},
    {0x22EAU, 0x020008C7U},
    {0x22EBU, 0x020008C9U},
    {0x22ECU, 0x020008CBU},
    {0x22EDU, 0x020008CDU},
    {0x2329U, 0x010008CFU},
    {0x232AU, 0x010008D0U},
    {0x2460U, 0x8100000DU},
    {0x2461U, 0x81000006U},
    {0x2462U, 0x81000007U},
    {0x2463U, 0x810007D1U},
    {0x2464U, 0x810007D2U},
    {0x2465U, 0x810007D3U},
    {0x2466U, 0x810007D4U},
    {0x2467U, 0x810007D5U},
    {0x2468U, 0x810007D6U},
    {0x2469U, 0x820008D1U},
    {0x246AU, 0x820008D3U},
    {0x246BU, 0x820008D5U},
    {0x246CU, 0x820008D7U},
    {0x246DU, 0x820008D9U},
    {0x246EU, 0x820008DBU},
    {0x246FU, 0x820008DDU},
    {0x2470U, 0x820008DFU},
    {0x2471U, 0x820008E1U},
    {0x2472U, 0x820008E3U},
    {0x2473U, 0x820008E5U},
    {0x2474U, 0x830008E7U},
    {0x2475U, 0x830008EAU},
    {0x2476U, 0x830008EDU},
    {0x2477U, 0x830008F0U},
    {0x2478U, 0x830008F3U},
    {0x2479U, 0x830008F6U},
    {0x247AU, 0x830008F9U},
    {0x247BU, 0x830008FCU},
    {0x247CU, 0x830008FFU},
    {0x247DU, 0x84000902U},
    {0x247EU, 0x84000906U},
    {0x247FU, 0x8400090AU},
    {0x2480U, 0x8400090EU},
    {0x2481U, 0x84000912U},
    {0x2482U, 0x84000916U},
    {0x2483U, 0x8400091AU},
    {0x2484U, 0x8400091EU},
    {0x2485U, 0x84000922U},
    {0x2486U, 0x84000926U},
    {0x2487U, 0x8400092AU},
    {0x2488U, 0x8200092EU},
    {0x2489U, 0x82000930U},
    {0x248AU, 0x82000932U},
    {0x248BU, 0x82000934U},
    {0x248CU, 0x82000936U},
    {0x248DU, 0x82000938U},
    {0x248EU, 0x8200093AU},
    {0x248FU, 0x8200093CU},
    {0x2490U, 0x8200093EU},
    {0x2491U, 0x83000940U},
    {0x2492U, 0x83000943U},
    {0x2493U, 0x83000946U},
    {0x2494U, 0x83000949U},
    {0x2495U, 0x8300094CU},
    {0x2496U, 0x8300094FU},
    {0x2497U, 0x83000952U},
    {0x2498U, 0x83000955U},
    {0x2499U, 0x83000958U},
    {0x249AU, 0x8300095BU},
    {0x249BU, 0x8300095EU},
    {0x249CU, 0x83000961U},
    {0x249DU, 0x83000964U},
    {0x249EU, 0x83000967U},
    {0x249FU, 0x8300096AU},
    {0x24A0U, 0x8300096DU},
    {0x24A1U, 0x83000970U},
    {0x24A2U, 0x83000973U},
    {0x24A3U, 0x83000976U},
    {0x24A4U, 0x83000979U},
    {0x24A5U, 0x8300097CU},
    {0x24A6U, 0x8300097FU},
    {0x24A7U, 0x83000982U},
    {0x24A8U, 0x83000985U},
    {0x24A9U, 0x83000988U},
    {0x24AAU, 0x8300098BU},
    {0x24ABU, 0x8300098EU},
    {0x24ACU, 0x83000991U},
    {0x24ADU, 0x83000994U},
    {0x24AEU, 0x83000997U},
    {0x24AFU, 0x8300099AU},
    {0x24B0U, 0x8300099DU},
    {0x24B1U, 0x830009A0U},
    {0x24B2U, 0x830009A3U},
    {0x24B3U, 0x830009A6U},
    {0x24B4U, 0x830009A9U},
    {0x24B5U, 0x830009ACU},
    {0x24B6U, 0x810003B1U},
    {0x24B7U, 0x810003B3U},
    {0x24B8U, 0x810007E5U},
    {0x24B9U, 0x810003B4U},
    {0x24BAU, 0x810003B5U},
    {0x24BBU, 0x810007FFU},
    {0x24BCU, 0x810003B7U},
    {0x24BDU, 0x810003B8U},
    {0x24BEU, 0x810003B9U},
    {0x24BFU, 0x810003BAU},
    {0x24C0U, 0x810003BBU},
    {0x24C1U, 0x810003BCU},
    {0x24C2U, 0x810003BDU},
    {0x24C3U, 0x810003BEU},
    {0x24C4U, 0x810003BFU},
    {0x24C5U, 0x810003C1U},
    {0x24C6U, 0x810007F4U},
    {0x24C7U, 0x810003C2U},
    {0x24C8U, 0x810009AFU},
    {0x24C9U, 0x810003C3U},
    {0x24CAU, 0x810003C4U},
    {0x24CBU, 0x81000841U},
    {0x24CCU, 0x810003C5U},
    {0x24CDU, 0x8100084DU},
    {0x24CEU, 0x810009B0U},
    {0x24CFU, 0x810007FCU},
    {0x24D0U, 0x81000003U},
    {0x24D1U, 0x810003C9U},
    {0x24D2U, 0x810003E3U},
    {0x24D3U, 0x810003CAU},
    {0x24D4U, 0x810003CBU},
    {0x24D5U, 0x810003E6U},
    {0x24D6U, 0x810003CFU},
    {0x24D7U, 0x81000233U},
    {0x24D8U, 0x810003E0U},
    {0x24D9U, 0x81000235U},
    {0x24DAU, 0x810003D0U},
    {0x24DBU, 0x81000249U},
    {0x24DCU, 0x810003D1U},
    {0x24DDU, 0x810007DCU},
    {0x24DEU, 0x8100000EU},
    {0x24DFU, 0x810003D6U},
    {0x24E0U, 0x810009B1U},
    {0x24E1U, 0x81000236U},
    {0x24E2U, 0x81000164U},
    {0x24E3U, 0x810003D7U},
    {0x24E4U, 0x810003D8U},
    {0x24E5U, 0x810003DBU},
    {0x24E6U, 0x8100023AU},
    {0x24E7U, 0x8100024AU},
    {0x24E8U, 0x8100023BU},
    {0x24E9U, 0x81000401U},
    {0x24EAU, 0x810007D0U},
    {0x2A0CU, 0x840009B2U},
    {0x2A74U, 0x830009B6U},
    {0x2A75U, 0x820009B9U},
    {0x2A76U, 0x830009BBU},
    {0x2ADCU, 0x020009BEU},
    {0x2C7CU, 0x81000235U},
    {0x2C7DU, 0x81000841U},
    {0x2D6FU, 0x810009C0U},
    {0x2E9FU, 0x810009C1U},
    {0x2EF3U, 0x810009C2U},
    {0x2F00U, 0x810009C3U},
    {0x2F01U, 0x810009C4U},
    {0x2F02U, 0x810009C5U},
    {0x2F03U, 0x810009C6U},
    {0x2F04U, 0x810009C7U},
    {0x2F05U, 0x810009C8U},
    {0x2F06U, 0x810009C9U},
    {0x2F07U, 0x810009CAU},
    {0x2F08U, 0x810009CBU},
    {0x2F09U, 0x810009CCU},
    {0x2F0AU, 0x810009CDU},
    {0x2F0BU, 0x810009CEU},
    {0x2F0CU, 0x810009CFU},
    {0x2F0DU, 0x810009D0U},
    {0x2F0EU, 0x810009D1U},
    {0x2F0FU, 0x810009D2U},
    {0x2F10U, 0x810009D3U},
    {0x2F11U, 0x810009D4U},
    {0x2F12U, 0x810009D5U},
    {0x2F13U, 0x810009D6U},
    {0x2F14U, 0x810009D7U},
    {0x2F15U, 0x810009D8U},
    {0x2F16U, 0x810009D9U},
    {0x2F17U, 0x810009DAU},
    {0x2F18U, 0x810009DBU},
    {0x2F19U, 0x810009DCU},
    {0x2F1AU, 0x810009DDU},
    {0x2F1BU, 0x810009DEU},
    {0x2F1CU, 0x810009DFU},
    {0x2F1DU, 0x810009E0U},
    {0x2F1EU, 0x810009E1U},
    {0x2F1FU, 0x810009E2U},
    {0x2F20U, 0x810009E3U},
    {0x2F21U, 0x810009E4U},
    {0x2F22U, 0x810009E5U},
    {0x2F23U, 0x810009E6U},
    {0x2F24U, 0x810009E7U},
    {0x2F25U, 0x810009E8U},
    {0x2F26U, 0x810009E9U},
    {0x2F27U, 0x810009EAU},
    {0x2F28U, 0x810009EBU},
    {0x2F29U, 0x810009ECU},
    {0x2F2AU, 0x810009EDU},
    {0x2F2BU, 0x810009EEU},
    {0x2F2CU, 0x810009EFU},
    {0x2F2DU, 0x810009F0U},
    {0x2F2EU, 0x810009F1U},
    {0x2F2FU, 0x810009F2U},
    {0x2F30U, 0x810009F3U},
    {0x2F31U, 0x810009F4U},
    {0x2F32U, 0x810009F5U},
    {0x2F33U, 0x810009F6U},
    {0x2F34U, 0x810009F7U},
    {0x2F35U, 0x810009F8U},
    {0x2F36U, 0x810009F9U},
    {0x2F37U, 0x810009FAU},
    {0x2F38U, 0x810009FBU},
    {0x2F39U, 0x810009FCU},
    {0x2F3AU, 0x810009FDU},
    {0x2F3BU, 0x810009FEU},
    {0x2F3CU, 0x810009FFU},
    {0x2F3DU, 0x81000A00U},
    {0x2F3EU, 0x81000A01U},
    {0x2F3FU, 0x81000A02U},
    {0x2F40U, 0x81000A03U},
    {0x2F41U, 0x81000A04U},
    {0x2F42U, 0x81000A05U},
    {0x2F43U, 0x81000A06U},
    {0x2F44U, 0x81000A07U},
    {0x2F45U, 0x81000A08U},
    {0x2F46U, 0x81000A09U},
    {0x2F47U, 0x81000A0AU},
    {0x2F48U, 0x81000A0BU},
    {0x2F49U, 0x81000A0CU},
    {0x2F4AU, 0x81000A0DU},
    {0x2F4BU, 0x81000A0EU},
    {0x2F4CU, 0x81000A0FU},
    {0x2F4DU, 0x81000A10U},
    {0x2F4EU, 0x81000A11U},
    {0x2F4FU, 0x81000A12U},
    {0x2F50U, 0x81000A13U},
    {0x2F51U, 0x81000A14U},
    {0x2F52U, 0x81000A15U},
    {0x2F53U, 0x81000A16U},
    {0x2F54U, 0x81000A17U},
    {0x2F55U, 0x81000A18U},
    {0x2F56U, 0x81000A19U},
    {0x2F57U, 0x81000A1AU},
    {0x2F58U, 0x81000A1BU},
    {0x2F59U, 0x81000A1CU},
    {0x2F5AU, 0x81000A1DU},
    {0x2F5BU, 0x81000A1EU},
    {0x2F5CU, 0x81000A1FU},
    {0x2F5DU, 0x81000A20U},
    {0x2F5EU, 0x81000A21U},
    {0x2F5FU, 0x81000A22U},
    {0x2F60U, 0x81000A23U},
    {0x2F61U, 0x81000A24U},
    {0x2F62U, 0x81000A25U},
    {0x2F63U, 0x81000A26U},
    {0x2F64U, 0x81000A27U},
    {0x2F65U, 0x81000A28U},
    {0x2F66U, 0x81000A29U},
    {0x2F67U, 0x81000A2AU},
    {0x2F68U, 0x81000A2BU},
    {0x2F69U, 0x81000A2CU},
    {0x2F6AU, 0x81000A2DU},
    {0x2F6BU, 0x81000A2EU},
    {0x2F6CU, 0x81000A2FU},
    {0x2F6DU, 0x81000A30U},
    {0x2F6EU, 0x81000A31U},
    {0x2F6FU, 0x81000A32U},
    {0x2F70U, 0x81000A33U},
    {0x2F71U, 0x81000A34U},
    {0x2F72U, 0x81000A35U},
    {0x2F73U, 0x81000A36U},
    {0x2F74U, 0x81000A37U},
    {0x2F75U, 0x81000A38U},
    {0x2F76U, 0x81000A39U},
    {0x2F77U, 0x81000A3AU},
    {0x2F78U, 0x81000A3BU},
    {0x2F79U, 0x81000A3CU},
    {0x2F7AU, 0x81000A3DU},
    {0x2F7BU, 0x81000A3EU},
    {0x2F7CU, 0x81000A3FU},
    {0x2F7DU, 0x81000A40U},
    {0x2F7EU, 0x81000A41U},
    {0x2F7FU, 0x81000A42U},
    {0x2F80U, 0x81000A43U},
    {0x2F81U, 0x81000A44U},
    {0x2F82U, 0x81000A45U},
    {0x2F83U, 0x81000A46U},
    {0x2F84U, 0x81000A47U},
    {0x2F85U, 0x81000A48U},
    {0x2F86U, 0x81000A49U},
    {0x2F87U, 0x81000A4AU},
    {0x2F88U, 0x81000A4BU},
    {0x2F89U, 0x81000A4CU},
    {0x2F8AU, 0x81000A4DU},
    {0x2F8BU, 0x81000A4EU},
    {0x2F8CU, 0x81000A4FU},
    {0x2F8DU, 0x81000A50U},
    {0x2F8EU, 0x81000A51U},
    {0x2F8FU, 0x81000A52U},
    {0x2F90U, 0x81000A53U},
    {0x2F91U, 0x81000A54U},
    {0x2F92U, 0x81000A55U},
    {0x2F93U, 0x81000A56U},
    {0x2F94U, 0x81000A57U},
    {0x2F95U, 0x81000A58U},
    {0x2F96U, 0x81000A59U},
    {0x2F97U, 0x81000A5AU},
    {0x2F98U, 0x81000A5BU},
    {0x2F99U, 0x81000A5CU},
    {0x2F9AU, 0x81000A5DU},
    {0x2F9BU, 0x81000A5EU},
    {0x2F9CU, 0x81000A5FU},
    {0x2F9DU, 0x81000A60U},
    {0x2F9EU, 0x81000A61U},
    {0x2F9FU, 0x81000A62U},
    {0x2FA0U, 0x81000A63U},
    {0x2FA1U, 0x81000A64U},
    {0x2FA2U, 0x81000A65U},
    {0x2FA3U, 0x81000A66U},
    {0x2FA4U, 0x81000A67U},
    {0x2FA5U, 0x81000A68U},
    {0x2FA6U, 0x81000A69U},
    {0x2FA7U, 0x81000A6AU},
    {0x2FA8U, 0x81000A6BU},
    {0x2FA9U, 0x81000A6CU},
    {0x2FAAU, 0x81000A6DU},
    {0x2FABU, 0x81000A6EU},
    {0x2FACU, 0x81000A6FU},
    {0x2FADU, 0x81000A70U},
    {0x2FAEU, 0x81000A71U},
    {0x2FAFU, 0x81000A72U},
    {0x2FB0U, 0x81000A73U},
    {0x2FB1U, 0x81000A74U},
    {0x2FB2U, 0x81000A75U},
    {0x2FB3U, 0x81000A76U},
    {0x2FB4U, 0x81000A77U},
    {0x2FB5U, 0x81000A78U},
    {0x2FB6U, 0x81000A79U},
    {0x2FB7U, 0x81000A7AU},
    {0x2FB8U, 0x81000A7BU},
    {0x2FB9U, 0x81000A7CU},
    {0x2FBAU, 0x81000A7DU},
    {0x2FBBU, 0x81000A7EU},
    {0x2FBCU, 0x81000A7FU},
    {0x2FBDU, 0x81000A80U},
    {0x2FBEU, 0x81000A81U},
    {0x2FBFU, 0x81000A82U},
    {0x2FC0U, 0x81000A83U},
    {0x2FC1U, 0x81000A84U},
    {0x2FC2U, 0x81000A85U},
    {0x2FC3U, 0x81000A86U},
    {0x2FC4U, 0x81000A87U},
    {0x2FC5U, 0x81000A88U},
    {0x2FC6U, 0x81000A89U},
    {0x2FC7U, 0x81000A8AU},
    {0x2FC8U, 0x81000A8BU},
    {0x2FC9U, 0x81000A8CU},
    {0x2FCAU, 0x81000A8DU},
    {0x2FCBU, 0x81000A8EU},
    {0x2FCCU, 0x81000A8FU},
    {0x2FCDU, 0x81000A90U},
    {0x2FCEU, 0x81000A91U},
    {0x2FCFU, 0x81000A92U},
    {0x2FD0U, 0x81000A93U},
    {0x2FD1U, 0x81000A94U},
    {0x2FD2U, 0x81000A95U},
    {0x2FD3U, 0x81000A96U},
    {0x2FD4U, 0x81000A97U},
    {0x2FD5U, 0x81000A98U},
    {0x3000U, 0x81000000U},
    {0x3036U, 0x81000A99U},
    {0x3038U, 0x810009DAU},
    {0x3039U, 0x81000A9AU},
    {0x303AU, 0x81000A9BU},
    {0x304CU, 0x02000A9CU},
    {0x304EU, 0x02000A9EU},
    {0x3050U, 0x02000AA0U},
    {0x3052U, 0x02000AA2U},
    {0x3054U, 0x02000AA4U},
    {0x3056U, 0x02000AA6U},
    {0x3058U, 0x02000AA8U},
    {0x305AU, 0x02000AAAU},
    {0x305CU, 0x02000AACU},
    {0x305EU, 0x02000AAEU},
    {0x3060U, 0x02000AB0U},
    {0x3062U, 0x02000AB2U},
    {0x3065U, 0x02000AB4U},
    {0x3067U, 0x02000AB6U},
    {0x3069U, 0x02000AB8U},
    {0x3070U, 0x02000ABAU},
    {0x3071U, 0x02000ABCU},
    {0x3073U, 0x02000ABEU},
    {0x3074U, 0x02000AC0U},
    {0x3076U, 0x02000AC2U},
    {0x3077U, 0x02000AC4U},
    {0x3079U, 0x02000AC6U},
    {0x307AU, 0x02000AC8U},
    {0x307CU, 0x02000ACAU},
    {0x307DU, 0x02000ACCU},
    {0x3094U, 0x02000ACEU},
    {0x309BU, 0x82000AD0U},
    {0x309CU, 0x82000AD2U},
    {0x309EU, 0x02000AD4U},
    {0x309FU, 0x82000AD6U},
    {0x30ACU, 0x02000AD8U},
    {0x30AEU, 0x02000ADAU},
    {0x30B0U, 0x02000ADCU},
    {0x30B2U, 0x02000ADEU},
    {0x30B4U, 0x02000AE0U},
    {0x30B6U, 0x02000AE2U},
    {0x30B8U, 0x02000AE4U},
    {0x30BAU, 0x02000AE6U},
    {0x30BCU, 0x02000AE8U},
    {0x30BEU, 0x02000AEAU},
    {0x30C0U, 0x02000AECU},
    {0x30C2U, 0x02000AEEU},
    {0x30C5U, 0x02000AF0U},
    {0x30C7U, 0x02000AF2U},
    {0x30C9U, 0x02000AF4U},
    {0x30D0U, 0x02000AF6U},
    {0x30D1U, 0x02000AF8U},
    {0x30D3U, 0x02000AFAU},
    {0x30D4U, 0x02000AFCU},
    {0x30D6U, 0x02000AFEU},
    {0x30D7U, 0x02000B00U},
    {0x30D9U, 0x02000B02U},
    {0x30DAU, 0x02000B04U},
    {0x30DCU, 0x02000B06U},
    {0x30DDU, 0x02000B08U},
    {0x30F4U, 0x02000B0AU},
    {0x30F7U, 0x02000B0CU},
    {0x30F8U, 0x02000B0EU},
    {0x30F9U, 0x02000B10U},
    {0x30FAU, 0x02000B12U},
    {0x30FEU, 0x02000B14U},
    {0x30FFU, 0x82000B16U},
    {0x3131U, 0x81000B18U},
    {0x3132U, 0x81000B19U},
    {0x3133U, 0x81000B1AU},
    {0x3134U, 0x81000B1BU},
    {0x3135U, 0x81000B1CU},
    {0x3136U, 0x81000B1DU},
    {0x3137U, 0x81000B1EU},
    {0x3138U, 0x81000B1FU},
    {0x3139U, 0x81000B20U},
    {0x313AU, 0x81000B21U},
    {0x313BU, 0x81000B22U},
    {0x313CU, 0x81000B23U},
    {0x313DU, 0x81000B24U},
    {0x313EU, 0x81000B25U},
    {0x313FU, 0x81000B26U},
    {0x3140U, 0x81000B27U},
    {0x3141U, 0x81000B28U},
    {0x3142U, 0x81000B29U},
    {0x3143U, 0x81000B2AU},
    {0x3144U, 0x81000B2BU},
    {0x3145U, 0x81000B2CU},
    {0x3146U, 0x81000B2DU},
    {0x3147U, 0x81000B2EU},
    {0x3148U, 0x81000B2FU},
    {0x3149U, 0x81000B30U},
    {0x314AU, 0x81000B31U},
    {0x314BU, 0x81000B32U},
    {0x314CU, 0x81000B33U},
    {0x314DU, 0x81000B34U},
    {0x314EU, 0x81000B35U},
    {0x314FU, 0x81000B36U},
    {0x3150U, 0x81000B37U},
    {0x3151U, 0x81000B38U},
    {0x3152U, 0x81000B39U},
    {0x3153U, 0x81000B3AU},
    {0x3154U, 0x81000B3BU},
    {0x3155U, 0x81000B3CU},
    {0x3156U, 0x81000B3DU},
    {0x3157U, 0x81000B3EU},
    {0x3158U, 0x81000B3FU},
    {0x3159U, 0x81000B40U},
    {0x315AU, 0x81000B41U},
    {0x315BU, 0x81000B42U},
    {0x315CU, 0x81000B43U},
    {0x315DU, 0x81000B44U},
    {0x315EU, 0x81000B45U},
    {0x315FU, 0x81000B46U},
    {0x3160U, 0x81000B47U},
    {0x3161U, 0x81000B48U},
    {0x3162U, 0x81000B49U},
    {0x3163U, 0x81000B4AU},
    {0x3164U, 0x81000B4BU},
    {0x3165U, 0x81000B4CU},
    {0x3166U, 0x81000B4DU},
    {0x3167U, 0x81000B4EU},
    {0x3168U, 0x81000B4FU},
    {0x3169U, 0x81000B50U},
    {0x316AU, 0x81000B51U},
    {0x316BU, 0x81000B52U},
    {0x316CU, 0x81000B53U},
    {0x316DU, 0x81000B54U},
    {0x316EU, 0x81000B55U},
    {0x316FU, 0x81000B56U},
    {0x3170U, 0x81000B57U},
    {0x3171U, 0x81000B58U},
    {0x3172U, 0x81000B59U},
    {0x3173U, 0x81000B5AU},
    {0x3174U, 0x81000B5BU},
    {0x3175U, 0x81000B5CU},
    {0x3176U, 0x81000B5DU},
    {0x3177U, 0x81000B5EU},
    {0x3178U, 0x81000B5FU},
    {0x3179U, 0x81000B60U},
    {0x317AU, 0x81000B61U},
    {0x317BU, 0x81000B62U},
    {0x317CU, 0x81000B63U},
    {0x317DU, 0x81000B64U},
    {0x317EU, 0x81000B65U},
    {0x317FU, 0x81000B66U},
    {0x3180U, 0x81000B67U},
    {0x3181U, 0x81000B68U},
    {0x3182U, 0x81000B69U},
    {0x3183U, 0x81000B6AU},
    {0x3184U, 0x81000B6BU},
    {0x3185U, 0x81000B6CU},
    {0x3186U, 0x81000B6DU},
    {0x3187U, 0x81000B6EU},
    {0x3188U, 0x81000B6FU},
    {0x3189U, 0x81000B70U},
    {0x318AU, 0x81000B71U},
    {0x318BU, 0x81000B72U},
    {0x318CU, 0x81000B73U},
    {0x318DU, 0x81000B74U},
    {0x318EU, 0x81000B75U},
    {0x3192U, 0x810009C3U},
    {0x3193U, 0x810009C9U},
    {0x3194U, 0x81000B76U},
    {0x3195U, 0x81000B77U},
    {0x3196U, 0x81000B78U},
    {0x3197U, 0x81000B79U},
    {0x3198U, 0x81000B7AU},
    {0x3199U, 0x81000B7BU},
    {0x319AU, 0x810009C7U},
    {0x319BU, 0x81000B7CU},
    {0x319CU, 0x81000B7DU},
    {0x319DU, 0x81000B7EU},
    {0x319EU, 0x81000B7FU},
    {0x319FU, 0x810009CBU},
    {0x3200U, 0x83000B80U},
    {0x3201U, 0x83000B83U},
    {0x3202U, 0x83000B86U},
    {0x3203U, 0x83000B89U},
    {0x3204U, 0x83000B8CU},
    {0x3205U, 0x83000B8FU},
    {0x3206U, 0x83000B92U},
    {0x3207U, 0x83000B95U},
    {0x3208U, 0x83000B98U},
    {0x3209U, 0x83000B9BU},
    {0x320AU, 0x83000B9EU},
    {0x320BU, 0x83000BA1U},
    {0x320CU, 0x83000BA4U},
    {0x320DU, 0x83000BA7U},
    {0x320EU, 0x84000BAAU},
    {0x320FU, 0x84000BAEU},
    {0x3210U, 0x84000BB2U},
    {0x3211U, 0x84000BB6U},
    {0x3212U, 0x84000BBAU},
    {0x3213U, 0x84000BBEU},
    {0x3214U, 0x84000BC2U},
    {0x3215U, 0x84000BC6U},
    {0x3216U, 0x84000BCAU},
    {0x3217U, 0x84000BCEU},
    {0x3218U, 0x84000BD2U},
    {0x3219U, 0x84000BD6U},
    {0x321AU, 0x84000BDAU},
    {0x321BU, 0x84000BDEU},
    {0x321CU, 0x84000BE2U},
    {0x321DU, 0x87000BE6U},
    {0x321EU, 0x86000BEDU},
    {0x3220U, 0x83000BF3U},
    {0x3221U, 0x83000BF6U},
    {0x3222U, 0x83000BF9U},
    {0x3223U, 0x83000BFCU},
    {0x3224U, 0x83000BFFU},
    {0x3225U, 0x83000C02U},
    {0x3226U, 0x83000C05U},
    {0x3227U, 0x83000C08U},
    {0x3228U, 0x83000C0BU},
    {0x3229U, 0x83000C0EU},
    {0x322AU, 0x83000C11U},
    {0x322BU, 0x83000C14U},
    {0x322CU, 0x83000C17U},
    {0x322DU, 0x83000C1AU},
    {0x322EU, 0x83000C1DU},
    {0x322FU, 0x83000C20U},
    {0x3230U, 0x83000C23U},
    {0x3231U, 0x83000C26U},
    {0x3232U, 0x83000C29U},
    {0x3233U, 0x83000C2CU},
    {0x3234U, 0x83000C2FU},
    {0x3235U, 0x83000C32U},
    {0x3236U, 0x83000C35U},
    {0x3237U, 0x83000C38U},
    {0x3238U, 0x83000C3BU},
    {0x3239U, 0x83000C3EU},
    {0x323AU, 0x83000C41U},
    {0x323BU, 0x83000C44U},
    {0x323CU, 0x83000C47U},
    {0x323DU, 0x83000C4AU},
    {0x323EU, 0x83000C4DU},
    {0x323FU, 0x83000C50U},
    {0x3240U, 0x83000C53U},
    {0x3241U, 0x83000C56U},
    {0x3242U, 0x83000C59U},
    {0x3243U, 0x83000C5CU},
    {0x3244U, 0x81000C5FU},
    {0x3245U, 0x81000C60U},
    {0x3246U, 0x81000A05U},
    {0x3247U, 0x81000C61U},
    {0x3250U, 0x83000C62U},
    {0x3251U, 0x82000C65U},
    {0x3252U, 0x82000C67U},
    {0x3253U, 0x82000C69U},
    {0x3254U, 0x82000C6BU},
    {0x3255U, 0x82000C6DU},
    {0x3256U, 0x82000C6FU},
    {0x3257U, 0x82000C71U},
    {0x3258U, 0x82000C73U},
    {0x3259U, 0x82000C75U},
    {0x325AU, 0x82000C77U},
    {0x325BU, 0x82000C79U},
    {0x325CU, 0x82000C7BU},
    {0x325DU, 0x82000C7DU},
    {0x325EU, 0x82000C7FU},
    {0x325FU, 0x82000C81U},
    {0x3260U, 0x81000B18U},
    {0x3261U, 0x81000B1BU},
    {0x3262U, 0x81000B1EU},
    {0x3263U, 0x81000B20U},
    {0x3264U, 0x81000B28U},
    {0x3265U, 0x81000B29U},
    {0x3266U, 0x81000B2CU},
    {0x3267U, 0x81000B2EU},
    {0x3268U, 0x81000B2FU},
    {0x3269U, 0x81000B31U},
    {0x326AU, 0x81000B32U},
    {0x326BU, 0x81000B33U},
    {0x326CU, 0x81000B34U},
    {0x326DU, 0x81000B35U},
    {0x326EU, 0x82000C83U},
    {0x326FU, 0x82000C85U},
    {0x3270U, 0x82000C87U},
    {0x3271U, 0x82000C89U},
    {0x3272U, 0x82000C8BU},
    {0x3273U, 0x82000C8DU},
    {0x3274U, 0x82000C8FU},
    {0x3275U, 0x82000C91U},
    {0x3276U, 0x82000C93U},
    {0x3277U, 0x82000C95U},
    {0x3278U, 0x82000C97U},
    {0x3279U, 0x82000C99U},
    {0x327AU, 0x82000C9BU},
    {0x327BU, 0x82000C9DU},
    {0x327CU, 0x85000C9FU},
    {0x327DU, 0x84000CA4U},
    {0x327EU, 0x82000CA8U},
    {0x3280U, 0x810009C3U},
    {0x3281U, 0x810009C9U},
    {0x3282U, 0x81000B76U},
    {0x3283U, 0x81000B77U},
    {0x3284U, 0x81000CAAU},
    {0x3285U, 0x81000CABU},
    {0x3286U, 0x81000CACU},
    {0x3287U, 0x810009CEU},
    {0x3288U, 0x81000CADU},
    {0x3289U, 0x810009DAU},
    {0x328AU, 0x81000A0CU},
    {0x328BU, 0x81000A18U},
    {0x328CU, 0x81000A17U},
    {0x328DU, 0x81000A0DU},
    {0x328EU, 0x81000A69U},
    {0x328FU, 0x810009E2U},
    {0x3290U, 0x81000A0AU},
    {0x3291U, 0x81000CAEU},
    {0x3292U, 0x81000CAFU},
    {0x3293U, 0x81000CB0U},
    {0x3294U, 0x81000CB1U},
    {0x3295U, 0x81000CB2U},
    {0x3296U, 0x81000CB3U},
    {0x3297U, 0x81000CB4U},
    {0x3298U, 0x81000CB5U},
    {0x3299U, 0x81000CB6U},
    {0x329AU, 0x81000CB7U},
    {0x329BU, 0x810009E8U},
    {0x329CU, 0x81000CB8U},
    {0x329DU, 0x81000CB9U},
    {0x329EU, 0x81000CBAU},
    {0x329FU, 0x81000CBBU},
    {0x32A0U, 0x81000CBCU},
    {0x32A1U, 0x81000CBDU},
    {0x32A2U, 0x81000CBEU},
    {0x32A3U, 0x81000CBFU},
    {0x32A4U, 0x81000B78U},
    {0x32A5U, 0x81000B79U},
    {0x32A6U, 0x81000B7AU},
    {0x32A7U, 0x81000CC0U},
    {0x32A8U, 0x81000CC1U},
    {0x32A9U, 0x81000CC2U},
    {0x32AAU, 0x81000CC3U},
    {0x32ABU, 0x81000CC4U},
    {0x32ACU, 0x81000CC5U},
    {0x32ADU, 0x81000CC6U},
    {0x32AEU, 0x81000CC7U},
    {0x32AFU, 0x81000CC8U},
    {0x32B0U, 0x81000CC9U},
    {0x32B1U, 0x82000CCAU},
    {0x32B2U, 0x82000CCCU},
    {0x32B3U, 0x82000CCEU},
    {0x32B4U, 0x82000CD0U},
    {0x32B5U, 0x82000CD2U},
    {0x32B6U, 0x82000CD4U},
    {0x32B7U, 0x82000CD6U},
    {0x32B8U, 0x82000CD8U},
    {0x32B9U, 0x82000CDAU},
    {0x32BAU, 0x82000CDCU},
    {0x32BBU, 0x82000CDEU},
    {0x32BCU, 0x82000CE0U},
    {0x32BDU, 0x82000CE2U},
    {0x32BEU, 0x82000CE4U},
    {0x32BFU, 0x82000CE6U},
    {0x32C0U, 0x82000CE8U},
    {0x32C1U, 0x82000CEAU},
    {0x32C2U, 0x82000CECU},
    {0x32C3U, 0x82000CEEU},
    {0x32C4U, 0x82000CF0U},
    {0x32C5U, 0x82000CF2U},
    {0x32C6U, 0x82000CF4U},
    {0x32C7U, 0x82000CF6U},
    {0x32C8U, 0x82000CF8U},
    {0x32C9U, 0x83000CFAU},
    {0x32CAU, 0x83000CFDU},
    {0x32CBU, 0x83000D00U},
    {0x32CCU, 0x82000D03U},
    {0x32CDU, 0x83000D05U},
    {0x32CEU, 0x82000D08U},
    {0x32CFU, 0x83000D0AU},
    {0x32D0U, 0x81000D0DU},
    {0x32D1U, 0x81000D0EU},
    {0x32D2U, 0x81000D0FU},
    {0x32D3U, 0x81000D10U},
    {0x32D4U, 0x81000D11U},
    {0x32D5U, 0x81000D12U},
    {0x32D6U, 0x81000D13U},
    {0x32D7U, 0x81000D14U},
    {0x32D8U, 0x81000D15U},
    {0x32D9U, 0x81000D16U},
    {0x32DAU, 0x81000D17U},
    {0x32DBU, 0x81000D18U},
    {0x32DCU, 0x81000D19U},
    {0x32DDU, 0x81000D1AU},
    {0x32DEU, 0x81000D1BU},
    {0x32DFU, 0x81000D1CU},
    {0x32E0U, 0x81000D1DU},
    {0x32E1U, 0x81000D1EU},
    {0x32E2U, 0x81000D1FU},
    {0x32E3U, 0x81000D20U},
    {0x32E4U, 0x81000D21U},
    {0x32E5U, 0x81000D22U},
    {0x32E6U, 0x81000D23U},
    {0x32E7U, 0x81000D24U},
    {0x32E8U, 0x81000D25U},
    {0x32E9U, 0x81000D26U},
    {0x32EAU, 0x81000D27U},
    {0x32EBU, 0x81000D28U},
    {0x32ECU, 0x81000D29U},
    {0x32EDU, 0x81000D2AU},
    {0x32EEU, 0x81000D2BU},
    {0x32EFU, 0x81000D2CU},
    {0x32F0U, 0x81000D2DU},
    {0x32F1U, 0x81000D2EU},
    {0x32F2U, 0x81000D2FU},
    {0x32F3U, 0x81000D30U},
    {0x32F4U, 0x81000D31U},
    {0x32F5U, 0x81000D32U},
    {0x32F6U, 0x81000D33U},
    {0x32F7U, 0x81000D34U},
    {0x32F8U, 0x81000D35U},
    {0x32F9U, 0x81000D36U},
    {0x32FAU, 0x81000D37U},
    {0x32FBU, 0x81000D38U},
    {0x32FCU, 0x81000D39U},
    {0x32FDU, 0x81000D3AU},
    {0x32FEU, 0x81000D3BU},
    {0x32FFU, 0x82000D3CU},
    {0x3300U, 0x84000D3EU},
    {0x3301U, 0x84000D42U},
    {0x3302U, 0x84000D46U},
    {0x3303U, 0x83000D4AU},
    {0x3304U, 0x84000D4DU},
    {0x3305U, 0x83000D51U},
    {0x3306U, 0x83000D54U},
    {0x3307U, 0x85000D57U},
    {0x3308U, 0x84000D5CU},
    {0x3309U, 0x83000D60U},
    {0x330AU, 0x83000D63U},
    {0x330BU, 0x83000D66U},
    {0x330CU, 0x84000D69U},
    {0x330DU, 0x84000D6DU},
    {0x330EU, 0x83000D71U},
    {0x330FU, 0x83000D74U},
    {0x3310U, 0x82000D77U},
    {0x3311U, 0x83000D79U},
    {0x3312U, 0x84000D7CU},
    {0x3313U, 0x84000D80U},
    {0x3314U, 0x82000D84U},
    {0x3315U, 0x85000D86U},
    {0x3316U, 0x86000D8BU},
    {0x3317U, 0x85000D91U},
    {0x3318U, 0x83000D96U},
    {0x3319U, 0x85000D99U},
    {0x331AU, 0x85000D9EU},
    {0x331BU, 0x84000DA3U},
    {0x331CU, 0x83000DA7U},
    {0x331DU, 0x83000DAAU},
    {0x331EU, 0x83000DADU},
    {0x331FU, 0x84000DB0U},
    {0x3320U, 0x85000DB4U},
    {0x3321U, 0x84000DB9U},
    {0x3322U, 0x83000DBDU},
    {0x3323U, 0x83000DC0U},
    {0x3324U, 0x83000DC3U},
    {0x3325U, 0x82000DC6U},
    {0x3326U, 0x82000DC8U},
    {0x3327U, 0x82000DCAU},
    {0x3328U, 0x82000DCCU},
    {0x3329U, 0x83000DCEU},
    {0x332AU, 0x83000DD1U},
    {0x332BU, 0x85000DD4U},
    {0x332CU, 0x83000DD9U},
    {0x332DU, 0x84000DDCU},
    {0x332EU, 0x85000DE0U},
    {0x332FU, 0x83000DE5U},
    {0x3330U, 0x82000DE8U},
    {0x3331U, 0x82000DEAU},
    {0x3332U, 0x85000DECU},
    {0x3333U, 0x84000DF1U},
    {0x3334U, 0x85000DF5U},
    {0x3335U, 0x83000DFAU},
    {0x3336U, 0x85000DFDU},
    {0x3337U, 0x82000E02U},
    {0x3338U, 0x83000E04U},
    {0x3339U, 0x83000E07U},
    {0x333AU, 0x83000E0AU},
    {0x333BU, 0x83000E0DU},
    {0x333CU, 0x83000E10U},
    {0x333DU, 0x84000E13U},
    {0x333EU, 0x83000E17U},
    {0x333FU, 0x82000E1AU},
    {0x3340U, 0x83000E1CU},
    {0x3341U, 0x83000E1FU},
    {0x3342U, 0x83000E22U},
    {0x3343U, 0x84000E25U},
    {0x3344U, 0x83000E29U},
    {0x3345U, 0x83000E2CU},
    {0x3346U, 0x83000E2FU},
    {0x3347U, 0x85000E32U},
    {0x3348U, 0x84000E37U},
    {0x3349U, 0x82000E3BU},
    {0x334AU, 0x85000E3DU},
    {0x334BU, 0x82000E42U},
    {0x334CU, 0x84000E44U},
    {0x334DU, 0x84000E48U},
    {0x334EU, 0x83000E4CU},
    {0x334FU, 0x83000E4FU},
    {0x3350U, 0x83000E52U},
    {0x3351U, 0x84000E55U},
    {0x3352U, 0x82000E59U},
    {0x3353U, 0x83000E5BU},
    {0x3354U, 0x84000E5EU},
    {0x3355U, 0x82000E62U},
    {0x3356U, 0x85000E64U},
    {0x3357U, 0x83000E69U},
    {0x3358U, 0x82000E6CU},
    {0x3359U, 0x82000E6EU},
    {0x335AU, 0x82000E70U},
    {0x335BU, 0x82000E72U},
    {0x335CU, 0x82000E74U},
    {0x335DU, 0x82000E76U},
    {0x335EU, 0x82000E78U},
    {0x335FU, 0x82000E7AU},
    {0x3360U, 0x82000E7CU},
    {0x3361U, 0x82000E7EU},
    {0x3362U, 0x83000E80U},
    {0x3363U, 0x83000E83U},
    {0x3364U, 0x83000E86U},
    {0x3365U, 0x83000E89U},
    {0x3366U, 0x83000E8CU},
    {0x3367U, 0x83000E8FU},
    {0x3368U, 0x83000E92U},
    {0x3369U, 0x83000E95U},
    {0x336AU, 0x83000E98U},
    {0x336BU, 0x83000E9BU},
    {0x336CU, 0x83000E9EU},
    {0x336DU, 0x83000EA1U},
    {0x336EU, 0x83000EA4U},
    {0x336FU, 0x83000EA7U},
    {0x3370U, 0x83000EAAU},
    {0x3371U, 0x83000EADU},
    {0x3372U, 0x82000EB0U},
    {0x3373U, 0x82000EB2U},
    {0x3374U, 0x83000EB4U},
    {0x3375U, 0x82000EB7U},
    {0x3376U, 0x82000EB9U},
    {0x3377U, 0x82000EBBU},
    {0x3378U, 0x83000EBDU},
    {0x3379U, 0x83000EC0U},
    {0x337AU, 0x82000EC3U},
    {0x337BU, 0x82000EC5U},
    {0x337CU, 0x82000EC7U},
    {0x337DU, 0x82000EC9U},
    {0x337EU, 0x82000ECBU},
    {0x337FU, 0x84000ECDU},
    {0x3380U, 0x82000ED1U},
    {0x3381U, 0x82000ED3U},
    {0x3382U, 0x82000ED5U},
    {0x3383U, 0x82000ED7U},
    {0x3384U, 0x82000ED9U},
    {0x3385U, 0x82000EDBU},
    {0x3386U, 0x82000EDDU},
    {0x3387U, 0x82000EDFU},
    {0x3388U, 0x83000EE1U},
    {0x3389U, 0x84000EE4U},
    {0x338AU, 0x82000EE8U},
    {0x338BU, 0x82000EEAU},
    {0x338CU, 0x82000EECU},
    {0x338DU, 0x82000EEEU},
    {0x338EU, 0x82000EF0U},
    {0x338FU, 0x82000EF2U},
    {0x3390U, 0x82000EF4U},
    {0x3391U, 0x83000EF6U},
    {0x3392U, 0x83000EF9U},
    {0x3393U, 0x83000EFCU},
    {0x3394U, 0x83000EFFU},
    {0x3395U, 0x82000F02U},
    {0x3396U, 0x82000F04U},
    {0x3397U, 0x82000F06U},
    {0x3398U, 0x82000F08U},
    {0x3399U, 0x82000F0AU},
    {0x339AU, 0x82000F0CU},
    {0x339BU, 0x82000F0EU},
    {0x339CU, 0x82000F10U},
    {0x339DU, 0x82000F12U},
    {0x339EU, 0x82000F14U},
    {0x339FU, 0x83000F16U},
    {0x33A0U, 0x83000F19U},
    {0x33A1U, 0x82000F1CU},
    {0x33A2U, 0x83000F1EU},
    {0x33A3U, 0x83000F21U},
    {0x33A4U, 0x83000F24U},
    {0x33A5U, 0x82000F27U},
    {0x33A6U, 0x83000F29U},
    {0x33A7U, 0x83000F2CU},
    {0x33A8U, 0x84000F2FU},
    {0x33A9U, 0x82000F33U},
    {0x33AAU, 0x83000F35U},
    {0x33ABU, 0x83000F38U},
    {0x33ACU, 0x83000F3BU},
    {0x33ADU, 0x83000F3EU},
    {0x33AEU, 0x85000F41U},
    {0x33AFU, 0x86000F46U},
    {0x33B0U, 0x82000F4CU},
    {0x33B1U, 0x82000F4EU},
    {0x33B2U, 0x82000F50U},
    {0x33B3U, 0x82000F52U},
    {0x33B4U, 0x82000F54U},
    {0x33B5U, 0x82000F56U},
    {0x33B6U, 0x82000F58U},
    {0x33B7U, 0x82000F5AU},
    {0x33B8U, 0x82000F5CU},
    {0x33B9U, 0x82000F5EU},
    {0x33BAU, 0x82000F60U},
    {0x33BBU, 0x82000F62U},
    {0x33BCU, 0x82000F64U},
    {0x33BDU, 0x82000F66U},
    {0x33BEU, 0x82000F68U},
    {0x33BFU, 0x82000F6AU},
    {0x33C0U, 0x82000F6CU},
    {0x33C1U, 0x82000F6EU},
    {0x33C2U, 0x84000F70U},
    {0x33C3U, 0x82000F74U},
    {0x33C4U, 0x82000F76U},
    {0x33C5U, 0x82000F78U},
    {0x33C6U, 0x84000F7AU},
    {0x33C7U, 0x83000F7EU},
    {0x33C8U, 0x82000F81U},
    {0x33C9U, 0x82000F83U},
    {0x33CAU, 0x82000F85U},
    {0x33CBU, 0x82000F87U},
    {0x33CCU, 0x82000F89U},
    {0x33CDU, 0x82000F8BU},
    {0x33CEU, 0x82000F8DU},
    {0x33CFU, 0x82000F8FU},
    {0x33D0U, 0x82000F91U},
    {0x33D1U, 0x82000F93U},
    {0x33D2U, 0x83000F95U},
    {0x33D3U, 0x82000F98U},
    {0x33D4U, 0x82000F9AU},
    {0x33D5U, 0x83000F9CU},
    {0x33D6U, 0x83000F9FU},
    {0x33D7U, 0x82000FA2U},
    {0x33D8U, 0x84000FA4U},
    {0x33D9U, 0x83000FA8U},
    {0x33DAU, 0x82000FABU},
    {0x33DBU, 0x82000FADU},
    {0x33DCU, 0x82000FAFU},
    {0x33DDU, 0x82000FB1U},
    {0x33DEU, 0x83000FB3U},
    {0x33DFU, 0x83000FB6U},
    {0x33E0U, 0x82000FB9U},
    {0x33E1U, 0x82000FBBU},
    {0x33E2U, 0x82000FBDU},
    {0x33E3U, 0x82000FBFU},
    {0x33E4U, 0x82000FC1U},
    {0x33E5U, 0x82000FC3U},
    {0x33E6U, 0x82000FC5U},
    {0x33E7U, 0x82000FC7U},
    {0x33E8U, 0x82000FC9U},
    {0x33E9U, 0x83000FCBU},
    {0x33EAU, 0x83000FCEU},
    {0x33EBU, 0x83000FD1U},
    {0x33ECU, 0x83000FD4U},
    {0x33EDU, 0x83000FD7U},
    {0x33EEU, 0x83000FDAU},
    {0x33EFU, 0x83000FDDU},
    {0x33F0U, 0x83000FE0U},
    {0x33F1U, 0x83000FE3U},
    {0x33F2U, 0x83000FE6U},
    {0x33F3U, 0x83000FE9U},
    {0x33F4U, 0x83000FECU},
    {0x33F5U, 0x83000FEFU},
    {0x33F6U, 0x83000FF2U},
    {0x33F7U, 0x83000FF5U},
    {0x33F8U, 0x83000FF8U},
    {0x33F9U, 0x83000FFBU},
    {0x33FAU, 0x83000FFEU},
    {0x33FBU, 0x83001001U},
    {0x33FCU, 0x83001004U},
    {0x33FDU, 0x83001007U},
    {0x33FEU, 0x8300100AU},
    {0x33FFU, 0x8300100DU},
    {0xA69CU, 0x81001010U},
    {0xA69DU, 0x81001011U},
    {0xA770U, 0x81001012U},
    {0xA7F1U, 0x810009AFU},
    {0xA7F2U, 0x810007E5U},
    {0xA7F3U, 0x810007FFU},
    {0xA7F4U, 0x810007F4U},
    {0xA7F8U, 0x81001013U},
    {0xA7F9U, 0x81001014U},
    {0xAB5CU, 0x81001015U},
    {0xAB5DU, 0x81001016U},
    {0xAB5EU, 0x81001017U},
    {0xAB5FU, 0x81001018U},
    {0xAB69U, 0x81001019U},
    {0xF900U, 0x0100101AU},
    {0xF901U, 0x0100101BU},
    {0xF902U, 0x01000A61U},
    {0xF903U, 0x0100101CU},
    {0xF904U, 0x0100101DU},
    {0xF905U, 0x0100101EU},
    {0xF906U, 0x0100101FU},
    {0xF907U, 0x01000A97U},
    {0xF908U, 0x01000A97U},
    {0xF909U, 0x01001020U},
    {0xF90AU, 0x01000A69U},
    {0xF90BU, 0x01001021U},
    {0xF90CU, 0x01001022U},
    {0xF90DU, 0x01001023U},
    {0xF90EU, 0x01001024U},
    {0xF90FU, 0x01001025U},
    {0xF910U, 0x01001026U},
    {0xF911U, 0x01001027U},
    {0xF912U, 0x01001028U},
    {0xF913U, 0x01001029U},
    {0xF914U, 0x0100102AU},
    {0xF915U, 0x0100102BU},
    {0xF916U, 0x0100102CU},
    {0xF917U, 0x0100102DU},
    {0xF918U, 0x0100102EU},
    {0xF919U, 0x0100102FU},
    {0xF91AU, 0x01001030U},
    {0xF91BU, 0x01001031U},
    {0xF91CU, 0x01001032U},
    {0xF91DU, 0x01001033U},
    {0xF91EU, 0x01001034U},
    {0xF91FU, 0x01001035U},
    {0xF920U, 0x01001036U},
    {0xF921U, 0x01001037U},
    {0xF922U, 0x01001038U},
    {0xF923U, 0x01001039U},
    {0xF924U, 0x0100103AU},
    {0xF925U, 0x0100103BU},
    {0xF926U, 0x0100103CU},
    {0xF927U, 0x0100103DU},
    {0xF928U, 0x0100103EU},
    {0xF929U, 0x0100103FU},
    {0xF92AU, 0x01001040U},
    {0xF92BU, 0x01001041U},
    {0xF92CU, 0x01001042U},
    {0xF92DU, 0x01001043U},
    {0xF92EU, 0x01001044U},
    {0xF92FU, 0x01001045U},
    {0xF930U, 0x01001046U},
    {0xF931U, 0x01001047U},
    {0xF932U, 0x01001048U},
    {0xF933U, 0x01001049U},
    {0xF934U, 0x01000A3FU},
    {0xF935U, 0x0100104AU},
    {0xF936U, 0x0100104BU},
    {0xF937U, 0x0100104CU},
    {0xF938U, 0x0100104DU},
    {0xF939U, 0x0100104EU},
    {0xF93AU, 0x0100104FU},
    {0xF93BU, 0x01001050U},
    {0xF93CU, 0x01001051U},
    {0xF93DU, 0x01001052U},
    {0xF93EU, 0x01001053U},
    {0xF93FU, 0x01001054U},
    {0xF940U, 0x01000A88U},
    {0xF941U, 0x01001055U},
    {0xF942U, 0x01001056U},
    {0xF943U, 0x01001057U},
    {0xF944U, 0x01001058U},
    {0xF945U, 0x01001059U},
    {0xF946U, 0x0100105AU},
    {0xF947U, 0x0100105BU},
    {0xF948U, 0x0100105CU},
    {0xF949U, 0x0100105DU},
    {0xF94AU, 0x0100105EU},
    {0xF94BU, 0x0100105FU},
    {0xF94CU, 0x01001060U},
    {0xF94DU, 0x01001061U},
    {0xF94EU, 0x01001062U},
    {0xF94FU, 0x01001063U},
    {0xF950U, 0x01001064U},
    {0xF951U, 0x01001065U},
    {0xF952U, 0x01001066U},
    {0xF953U, 0x01001067U},
    {0xF954U, 0x01001068U},
    {0xF955U, 0x01001069U},
    {0xF956U, 0x0100106AU},
    {0xF957U, 0x0100106BU},
    {0xF958U, 0x0100106CU},
    {0xF959U, 0x0100106DU},
    {0xF95AU, 0x0100106EU},
    {0xF95BU, 0x0100106FU},
    {0xF95CU, 0x0100102AU},
    {0xF95DU, 0x01001070U},
    {0xF95EU, 0x01001071U},
    {0xF95FU, 0x01001072U},
    {0xF960U, 0x01001073U},
    {0xF961U, 0x01001074U},
    {0xF962U, 0x01001075U},
    {0xF963U, 0x01001076U},
    {0xF964U, 0x01001077U},
    {0xF965U, 0x01001078U},
    {0xF966U, 0x01001079U},
    {0xF967U, 0x0100107AU},
    {0xF968U, 0x0100107BU},
    {0xF969U, 0x0100107CU},
    {0xF96AU, 0x0100107DU},
    {0xF96BU, 0x0100107EU},
    {0xF96CU, 0x0100107FU},
    {0xF96DU, 0x01001080U},
    {0xF96EU, 0x01001081U},
    {0xF96FU, 0x01001082U},
    {0xF970U, 0x01001083U},
    {0xF971U, 0x01000A63U},
    {0xF972U, 0x01001084U},
    {0xF973U, 0x01001085U},
    {0xF974U, 0x01001086U},
    {0xF975U, 0x01001087U},
    {0xF976U, 0x01001088U},
    {0xF977U, 0x01001089U},
    {0xF978U, 0x0100108AU},
    {0xF979U, 0x0100108BU},
    {0xF97AU, 0x0100108CU},
    {0xF97BU, 0x0100108DU},
    {0xF97CU, 0x0100108EU},
    {0xF97DU, 0x0100108FU},
    {0xF97EU, 0x01001090U},
    {0xF97FU, 0x01001091U},
    {0xF980U, 0x01001092U},
    {0xF981U, 0x010009E8U},
    {0xF982U, 0x01001093U},
    {0xF983U, 0x01001094U},
    {0xF984U, 0x01001095U},
    {0xF985U, 0x01001096U},
    {0xF986U, 0x01001097U},
    {0xF987U, 0x01001098U},
    {0xF988U, 0x01001099U},
    {0xF989U, 0x0100109AU},
    {0xF98AU, 0x010009D5U},
    {0xF98BU, 0x0100109BU},
    {0xF98CU, 0x0100109CU},
    {0xF98DU, 0x0100109DU},
    {0xF98EU, 0x0100109EU},
    {0xF98FU, 0x0100109FU},
    {0xF990U, 0x010010A0U},
    {0xF991U, 0x010010A1U},
    {0xF992U, 0x010010A2U},
    {0xF993U, 0x010010A3U},
    {0xF994U, 0x010010A4U},
    {0xF995U, 0x010010A5U},
    {0xF996U, 0x010010A6U},
    {0xF997U, 0x010010A7U},
    {0xF998U, 0x010010A8U},
    {0xF999U, 0x010010A9U},
    {0xF99AU, 0x010010AAU},
    {0xF99BU, 0x010010ABU},
    {0xF99CU, 0x010010ACU},
    {0xF99DU, 0x010010ADU},
    {0xF99EU, 0x010010AEU},
    {0xF99FU, 0x010010AFU},
    {0xF9A0U, 0x010010B0U},
    {0xF9A1U, 0x01001082U},
    {0xF9A2U, 0x010010B1U},
    {0xF9A3U, 0x010010B2U},
    {0xF9A4U, 0x010010B3U},
    {0xF9A5U, 0x010010B4U},
    {0xF9A6U, 0x010010B5U},
    {0xF9A7U, 0x010010B6U},
    {0xF9A8U, 0x010010B7U},
    {0xF9A9U, 0x010010B8U},
    {0xF9AAU, 0x01001072U},
    {0xF9ABU, 0x010010B9U},
    {0xF9ACU, 0x010010BAU},
    {0xF9ADU, 0x010010BBU},
    {0xF9AEU, 0x010010BCU},
    {0xF9AFU, 0x010010BDU},
    {0xF9B0U, 0x010010BEU},
    {0xF9B1U, 0x010010BFU},
    {0xF9B2U, 0x010010C0U},
    {0xF9B3U, 0x010010C1U},
    {0xF9B4U, 0x010010C2U},
    {0xF9B5U, 0x010010C3U},
    {0xF9B6U, 0x010010C4U},
    {0xF9B7U, 0x010010C5U},
    {0xF9B8U, 0x010010C6U},
    {0xF9B9U, 0x010010C7U},
    {0xF9BAU, 0x010010C8U},
    {0xF9BBU, 0x010010C9U},
    {0xF9BCU, 0x010010CAU},
    {0xF9BDU, 0x010010CBU},
    {0xF9BEU, 0x010010CCU},
    {0xF9BFU, 0x0100102AU},
    {0xF9C0U, 0x010010CDU},
    {0xF9C1U, 0x010010CEU},
    {0xF9C2U, 0x010010CFU},
    {0xF9C3U, 0x010010D0U},
    {0xF9C4U, 0x01000A96U},
    {0xF9C5U, 0x010010D1U},
    {0xF9C6U, 0x010010D2U},
    {0xF9C7U, 0x010010D3U},
    {0xF9C8U, 0x010010D4U},
    {0xF9C9U, 0x010010D5U},
    {0xF9CAU, 0x010010D6U},
    {0xF9CBU, 0x010010D7U},
    {0xF9CCU, 0x010010D8U},
    {0xF9CDU, 0x010010D9U},
    {0xF9CEU, 0x010010DAU},
    {0xF9CFU, 0x010010DBU},
    {0xF9D0U, 0x010010DCU},
    {0xF9D1U, 0x01000CABU},
    {0xF9D2U, 0x010010DDU},
    {0xF9D3U, 0x010010DEU},
    {0xF9D4U, 0x010010DFU},
    {0xF9D5U, 0x010010E0U},
    {0xF9D6U, 0x010010E1U},
    {0xF9D7U, 0x010010E2U},
    {0xF9D8U, 0x010010E3U},
    {0xF9D9U, 0x010010E4U},
    {0xF9DAU, 0x010010E5U},
    {0xF9DBU, 0x01001074U},
    {0xF9DCU, 0x010010E6U},
    {0xF9DDU, 0x010010E7U},
    {0xF9DEU, 0x010010E8U},
    {0xF9DFU, 0x010010E9U},
    {0xF9E0U, 0x010010EAU},
    {0xF9E1U, 0x010010EBU},
    {0xF9E2U, 0x010010ECU},
    {0xF9E3U, 0x010010EDU},
    {0xF9E4U, 0x010010EEU},
    {0xF9E5U, 0x010010EFU},
    {0xF9E6U, 0x010010F0U},
    {0xF9E7U, 0x010010F1U},
    {0xF9E8U, 0x010010F2U},
    {0xF9E9U, 0x01000A68U},
    {0xF9EAU, 0x010010F3U},
    {0xF9EBU, 0x010010F4U},
    {0xF9ECU, 0x010010F5U},
    {0xF9EDU, 0x010010F6U},
    {0xF9EEU, 0x010010F7U},
    {0xF9EFU, 0x010010F8U},
    {0xF9F0U, 0x010010F9U},
    {0xF9F1U, 0x010010FAU},
    {0xF9F2U, 0x010010FBU},
    {0xF9F3U, 0x010010FCU},
    {0xF9F4U, 0x010010FDU},
    {0xF9F5U, 0x010010FEU},
    {0xF9F6U, 0x010010FFU},
    {0xF9F7U, 0x01000A37U},
    {0xF9F8U, 0x01001100U},
    {0xF9F9U, 0x01001101U},
    {0xF9FAU, 0x01001102U},
    {0xF9FBU, 0x01001103U},
    {0xF9FCU, 0x01001104U},
    {0xF9FDU, 0x01001105U},
    {0xF9FEU, 0x01001106U},
    {0xF9FFU, 0x01001107U},
    {0xFA00U, 0x01001108U},
    {0xFA01U, 0x01001109U},
    {0xFA02U, 0x0100110AU},
    {0xFA03U, 0x0100110BU},
    {0xFA04U, 0x0100110CU},
    {0xFA05U, 0x0100110DU},
    {0xFA06U, 0x0100110EU},
    {0xFA07U, 0x0100110FU},
    {0xFA08U, 0x01000A52U},
    {0xFA09U, 0x01001110U},
    {0xFA0AU, 0x01000A55U},
    {0xFA0BU, 0x01001111U},
    {0xFA0CU, 0x01001112U},
    {0xFA0DU, 0x01001113U},
    {0xFA10U, 0x01001114U},
    {0xFA12U, 0x01001115U},
    {0xFA15U, 0x01001116U},
    {0xFA16U, 0x01001117U},
    {0xFA17U, 0x01001118U},
    {0xFA18U, 0x01001119U},
    {0xFA19U, 0x0100111AU},
    {0xFA1AU, 0x0100111BU},
    {0xFA1BU, 0x0100111CU},
    {0xFA1CU, 0x0100111DU},
    {0xFA1DU, 0x0100111EU},
    {0xFA1EU, 0x01000A3EU},
    {0xFA20U, 0x0100111FU},
    {0xFA22U, 0x01001120U},
    {0xFA25U, 0x01001121U},
    {0xFA26U, 0x01001122U},
    {0xFA2AU, 0x01001123U},
    {0xFA2BU, 0x01001124U},
    {0xFA2CU, 0x01001125U},
    {0xFA2DU, 0x01001126U},
    {0xFA2EU, 0x01001127U},
    {0xFA2FU, 0x01001128U},
    {0xFA30U, 0x01001129U},
    {0xFA31U, 0x0100112AU},
    {0xFA32U, 0x0100112BU},
    {0xFA33U, 0x0100112CU},
    {0xFA34U, 0x0100112DU},
    {0xFA35U, 0x0100112EU},
    {0xFA36U, 0x0100112FU},
    {0xFA37U, 0x01001130U},
    {0xFA38U, 0x01001131U},
    {0xFA39U, 0x01001132U},
    {0xFA3AU, 0x01001133U},
    {0xFA3BU, 0x01001134U},
    {0xFA3CU, 0x010009EFU},
    {0xFA3DU, 0x01001135U},
    {0xFA3EU, 0x01001136U},
    {0xFA3FU, 0x01001137U},
    {0xFA40U, 0x01001138U},
    {0xFA41U, 0x01001139U},
    {0xFA42U, 0x0100113AU},
    {0xFA43U, 0x0100113BU},
    {0xFA44U, 0x0100113CU},
    {0xFA45U, 0x0100113DU},
    {0xFA46U, 0x0100113EU},
    {0xFA47U, 0x0100113FU},
    {0xFA48U, 0x01001140U},
    {0xFA49U, 0x01001141U},
    {0xFA4AU, 0x01001142U},
    {0xFA4BU, 0x01001143U},
    {0xFA4CU, 0x01000CB0U},
    {0xFA4DU, 0x01001144U},
    {0xFA4EU, 0x01001145U},
    {0xFA4FU, 0x01001146U},
    {0xFA50U, 0x01001147U},
    {0xFA51U, 0x01000CB4U},
    {0xFA52U, 0x01001148U},
    {0xFA53U, 0x01001149U},
    {0xFA54U, 0x0100114AU},
    {0xFA55U, 0x0100114BU},
    {0xFA56U, 0x0100114CU},
    {0xFA57U, 0x010010A6U},
    {0xFA58U, 0x0100114DU},
    {0xFA59U, 0x0100114EU},
    {0xFA5AU, 0x0100114FU},
    {0xFA5BU, 0x01001150U},
    {0xFA5CU, 0x01001151U},
    {0xFA5DU, 0x01001152U},
    {0xFA5EU, 0x01001152U},
    {0xFA5FU, 0x01001153U},
    {0xFA60U, 0x01001154U},
    {0xFA61U, 0x01001155U},
    {0xFA62U, 0x01001156U},
    {0xFA63U, 0x01001157U},
    {0xFA64U, 0x01001158U},
    {0xFA65U, 0x01001159U},
    {0xFA66U, 0x0100115AU},
    {0xFA67U, 0x01001121U},
    {0xFA68U, 0x0100115BU},
    {0xFA69U, 0x0100115CU},
    {0xFA6AU, 0x0100115DU},
    {0xFA6BU, 0x0100115EU},
    {0xFA6CU, 0x0100115FU},
    {0xFA6DU, 0x01001160U},
    {0xFA70U, 0x01001161U},
    {0xFA71U, 0x01001162U},
    {0xFA72U, 0x01001163U},
    {0xFA73U, 0x01001164U},
    {0xFA74U, 0x01001165U},
    {0xFA75U, 0x01001166U},
    {0xFA76U, 0x01001167U},
    {0xFA77U, 0x01001168U},
    {0xFA78U, 0x0100112FU},
    {0xFA79U, 0x01001169U},
    {0xFA7AU, 0x0100116AU},
    {0xFA7BU, 0x0100116BU},
    {0xFA7CU, 0x01001114U},
    {0xFA7DU, 0x0100116CU},
    {0xFA7EU, 0x0100116DU},
    {0xFA7FU, 0x0100116EU},
    {0xFA80U, 0x0100116FU},
    {0xFA81U, 0x01001170U},
    {0xFA82U, 0x01001171U},
    {0xFA83U, 0x01001172U},
    {0xFA84U, 0x01001173U},
    {0xFA85U, 0x01001174U},
    {0xFA86U, 0x01001175U},
    {0xFA87U, 0x01001176U},
    {0xFA88U, 0x01001177U},
    {0xFA89U, 0x01001137U},
    {0xFA8AU, 0x01001178U},
    {0xFA8BU, 0x01001138U},
    {0xFA8CU, 0x01001179U},
    {0xFA8DU, 0x0100117AU},
    {0xFA8EU, 0x0100117BU},
    {0xFA8FU, 0x0100117CU},
    {0xFA90U, 0x0100117DU},
    {0xFA91U, 0x01001115U},
    {0xFA92U, 0x0100103FU},
    {0xFA93U, 0x0100117EU},
    {0xFA94U, 0x0100117FU},
    {0xFA95U, 0x01000A10U},
    {0xFA96U, 0x01001083U},
    {0xFA97U, 0x010010D6U},
    {0xFA98U, 0x01001180U},
    {0xFA99U, 0x01001181U},
    {0xFA9AU, 0x0100113FU},
    {0xFA9BU, 0x01001182U},
    {0xFA9CU, 0x01001140U},
    {0xFA9DU, 0x01001183U},
    {0xFA9EU, 0x01001184U},
    {0xFA9FU, 0x01001185U},
    {0xFAA0U, 0x01001117U},
    {0xFAA1U, 0x01001186U},
    {0xFAA2U, 0x01001187U},
    {0xFAA3U, 0x01001188U},
    {0xFAA4U, 0x01001189U},
    {0xFAA5U, 0x0100118AU},
    {0xFAA6U, 0x01001118U},
    {0xFAA7U, 0x0100118BU},
    {0xFAA8U, 0x0100118CU},
    {0xFAA9U, 0x0100118DU},
    {0xFAAAU, 0x0100118EU},
    {0xFAABU, 0x0100118FU},
    {0xFAACU, 0x01001190U},
    {0xFAADU, 0x0100114CU},
    {0xFAAEU, 0x01001191U},
    {0xFAAFU, 0x01001192U},
    {0xFAB0U, 0x010010A6U},
    {0xFAB1U, 0x01001193U},
    {0xFAB2U, 0x01001150U},
    {0xFAB3U, 0x01001194U},
    {0xFAB4U, 0x01001195U},
    {0xFAB5U, 0x01001196U},
    {0xFAB6U, 0x01001197U},
    {0xFAB7U, 0x01001198U},
    {0xFAB8U, 0x01001155U},
    {0xFAB9U, 0x01001199U},
    {0xFABAU, 0x01001120U},
    {0xFABBU, 0x0100119AU},
    {0xFABCU, 0x01001156U},
    {0xFABDU, 0x01001070U},
    {0xFABEU, 0x0100119BU},
    {0xFABFU, 0x01001157U},
    {0xFAC0U, 0x0100119CU},
    {0xFAC1U, 0x01001159U},
    {0xFAC2U, 0x0100119DU},
    {0xFAC3U, 0x0100119EU},
    {0xFAC4U, 0x0100119FU},
    {0xFAC5U, 0x010011A0U},
    {0xFAC6U, 0x010011A1U},
    {0xFAC7U, 0x0100115BU},
    {0xFAC8U, 0x0100111DU},
    {0xFAC9U, 0x010011A2U},
    {0xFACAU, 0x0100115CU},
    {0xFACBU, 0x010011A3U},
    {0xFACCU, 0x0100115DU},
    {0xFACDU, 0x010011A4U},
    {0xFACEU, 0x01000A97U},
    {0xFACFU, 0x010011A5U},
    {0xFAD0U, 0x010011A6U},
    {0xFAD1U, 0x010011A7U},
    {0xFAD2U, 0x010011A8U},
    {0xFAD3U, 0x010011A9U},
    {0xFAD4U, 0x010011AAU},
    {0xFAD5U, 0x010011ABU},
    {0xFAD6U, 0x010011ACU},
    {0xFAD7U, 0x010011ADU},
    {0xFAD8U, 0x010011AEU},
    {0xFAD9U, 0x010011AFU},
    {0xFB00U, 0x820011B0U},
    {0xFB01U, 0x820011B2U},
    {0xFB02U, 0x820011B4U},
    {0xFB03U, 0x830011B6U},
    {0xFB04U, 0x830011B9U},
    {0xFB05U, 0x820011BCU},
    {0xFB06U, 0x820011BEU},
    {0xFB13U, 0x820011C0U},
    {0xFB14U, 0x820011C2U},
    {0xFB15U, 0x820011C4U},
    {0xFB16U, 0x820011C6U},
    {0xFB17U, 0x820011C8U},
    {0xFB1DU, 0x020011CAU},
    {0xFB1FU, 0x020011CCU},
    {0xFB20U, 0x810011CEU},
    {0xFB21U, 0x81000800U},
    {0xFB22U, 0x81000803U},
    {0xFB23U, 0x810011CFU},
    {0xFB24U, 0x810011D0U},
    {0xFB25U, 0x810011D1U},
    {0xFB26U, 0x810011D2U},
    {0xFB27U, 0x810011D3U},
    {0xFB28U, 0x810011D4U},
    {0xFB29U, 0x810007D7U},
    {0xFB2AU, 0x020011D5U},
    {0xFB2BU, 0x020011D7U},
    {0xFB2CU, 0x020011D9U},
    {0xFB2DU, 0x020011DBU},
    {0xFB2EU, 0x020011DDU},
    {0xFB2FU, 0x020011DFU},
    {0xFB30U, 0x020011E1U},
    {0xFB31U, 0x020011E3U},
    {0xFB32U, 0x020011E5U},
    {0xFB33U, 0x020011E7U},
    {0xFB34U, 0x020011E9U},
    {0xFB35U, 0x020011EBU},
    {0xFB36U, 0x020011EDU},
    {0xFB38U, 0x020011EFU},
    {0xFB39U, 0x020011F1U},
    {0xFB3AU, 0x020011F3U},
    {0xFB3BU, 0x020011F5U},
    {0xFB3CU, 0x020011F7U},
    {0xFB3EU, 0x020011F9U},
    {0xFB40U, 0x020011FBU},
    {0xFB41U, 0x020011FDU},
    {0xFB43U, 0x020011FFU},
    {0xFB44U, 0x02001201U},
    {0xFB46U, 0x02001203U},
    {0xFB47U, 0x02001205U},
    {0xFB48U, 0x02001207U},
    {0xFB49U, 0x02001209U},
    {0xFB4AU, 0x0200120BU},
    {0xFB4BU, 0x0200120DU},
    {0xFB4CU, 0x0200120FU},
    {0xFB4DU, 0x02001211U},
    {0xFB4EU, 0x02001213U},
    {0xFB4FU, 0x82001215U},
    {0xFB50U, 0x81001217U},
    {0xFB51U, 0x81001217U},
    {0xFB52U, 0x81001218U},
    {0xFB53U, 0x81001218U},
    {0xFB54U, 0x81001218U},
    {0xFB55U, 0x81001218U},
    {0xFB56U, 0x81001219U},
    {0xFB57U, 0x81001219U},
    {0xFB58U, 0x81001219U},
    {0xFB59U, 0x81001219U},
    {0xFB5AU, 0x8100121AU},
    {0xFB5BU, 0x8100121AU},
    {0xFB5CU, 0x8100121AU},
    {0xFB5DU, 0x8100121AU},
    {0xFB5EU, 0x8100121BU},
    {0xFB5FU, 0x8100121BU},
    {0xFB60U, 0x8100121BU},
    {0xFB61U, 0x8100121BU},
    {0xFB62U, 0x8100121CU},
    {0xFB63U, 0x8100121CU},
    {0xFB64U, 0x8100121CU},
    {0xFB65U, 0x8100121CU},
    {0xFB66U, 0x8100121DU},
    {0xFB67U, 0x8100121DU},
    {0xFB68U, 0x8100121DU},
    {0xFB69U, 0x8100121DU},
    {0xFB6AU, 0x8100121EU},
    {0xFB6BU, 0x8100121EU},
    {0xFB6CU, 0x8100121EU},
    {0xFB6DU, 0x8100121EU},
    {0xFB6EU, 0x8100121FU},
    {0xFB6FU, 0x8100121FU},
    {0xFB70U, 0x8100121FU},
    {0xFB71U, 0x8100121FU},
    {0xFB72U, 0x81001220U},
    {0xFB73U, 0x81001220U},
    {0xFB74U, 0x81001220U},
    {0xFB75U, 0x81001220U},
    {0xFB76U, 0x81001221U},
    {0xFB77U, 0x81001221U},
    {0xFB78U, 0x81001221U},
    {0xFB79U, 0x81001221U},
    {0xFB7AU, 0x81001222U},
    {0xFB7BU, 0x81001222U},
    {0xFB7CU, 0x81001222U},
    {0xFB7DU, 0x81001222U},
    {0xFB7EU, 0x81001223U},
    {0xFB7FU, 0x81001223U},
    {0xFB80U, 0x81001223U},
    {0xFB81U, 0x81001223U},
    {0xFB82U, 0x81001224U},
    {0xFB83U, 0x81001224U},
    {0xFB84U, 0x81001225U},
    {0xFB85U, 0x81001225U},
    {0xFB86U, 0x81001226U},
    {0xFB87U, 0x81001226U},
    {0xFB88U, 0x81001227U},
    {0xFB89U, 0x81001227U},
    {0xFB8AU, 0x81001228U},
    {0xFB8BU, 0x81001228U},
    {0xFB8CU, 0x81001229U},
    {0xFB8DU, 0x81001229U},
    {0xFB8EU, 0x8100122AU},
    {0xFB8FU, 0x8100122AU},
    {0xFB90U, 0x8100122AU},
    {0xFB91U, 0x8100122AU},
    {0xFB92U, 0x8100122BU},
    {0xFB93U, 0x8100122BU},
    {0xFB94U, 0x8100122BU},
    {0xFB95U, 0x8100122BU},
    {0xFB96U, 0x8100122CU},
    {0xFB97U, 0x8100122CU},
    {0xFB98U, 0x8100122CU},
    {0xFB99U, 0x8100122CU},
    {0xFB9AU, 0x8100122DU},
    {0xFB9BU, 0x8100122DU},
    {0xFB9CU, 0x8100122DU},
    {0xFB9DU, 0x8100122DU},
    {0xFB9EU, 0x8100122EU},
    {0xFB9FU, 0x8100122EU},
    {0xFBA0U, 0x8100122FU},
    {0xFBA1U, 0x8100122FU},
    {0xFBA2U, 0x8100122FU},
    {0xFBA3U, 0x8100122FU},
    {0xFBA4U, 0x81001230U},
    {0xFBA5U, 0x81001230U},
    {0xFBA6U, 0x81001231U},
    {0xFBA7U, 0x81001231U},
    {0xFBA8U, 0x81001231U},
    {0xFBA9U, 0x81001231U},
    {0xFBAAU, 0x81001232U},
    {0xFBABU, 0x81001232U},
    {0xFBACU, 0x81001232U},
    {0xFBADU, 0x81001232U},
    {0xFBAEU, 0x81001233U},
    {0xFBAFU, 0x81001233U},
    {0xFBB0U, 0x81001234U},
    {0xFBB1U, 0x81001234U},
    {0xFBD3U, 0x81001235U},
    {0xFBD4U, 0x81001235U},
    {0xFBD5U, 0x81001235U},
    {0xFBD6U, 0x81001235U},
    {0xFBD7U, 0x81001236U},
    {0xFBD8U, 0x81001236U},
    {0xFBD9U, 0x81001237U},
    {0xFBDAU, 0x81001237U},
    {0xFBDBU, 0x81001238U},
    {0xFBDCU, 0x81001238U},
    {0xFBDDU, 0x81001239U},
    {0xFBDEU, 0x8100123AU},
    {0xFBDFU, 0x8100123AU},
    {0xFBE0U, 0x8100123BU},
    {0xFBE1U, 0x8100123BU},
    {0xFBE2U, 0x8100123CU},
    {0xFBE3U, 0x8100123CU},
    {0xFBE4U, 0x8100123DU},
    {0xFBE5U, 0x8100123DU},
    {0xFBE6U, 0x8100123DU},
    {0xFBE7U, 0x8100123DU},
    {0xFBE8U, 0x8100123EU},
    {0xFBE9U, 0x8100123EU},
    {0xFBEAU, 0x8200123FU},
    {0xFBEBU, 0x8200123FU},
    {0xFBECU, 0x82001241U},
    {0xFBEDU, 0x82001241U},
    {0xFBEEU, 0x82001243U},
    {0xFBEFU, 0x82001243U},
    {0xFBF0U, 0x82001245U},
    {0xFBF1U, 0x82001245U},
    {0xFBF2U, 0x82001247U},
    {0xFBF3U, 0x82001247U},
    {0xFBF4U, 0x82001249U},
    {0xFBF5U, 0x82001249U},
    {0xFBF6U, 0x8200124BU},
    {0xFBF7U, 0x8200124BU},
    {0xFBF8U, 0x8200124BU},
    {0xFBF9U, 0x8200124DU},
    {0xFBFAU, 0x8200124DU},
    {0xFBFBU, 0x8200124DU},
    {0xFBFCU, 0x8100124FU},
    {0xFBFDU, 0x8100124FU},
    {0xFBFEU, 0x8100124FU},
    {0xFBFFU, 0x8100124FU},
    {0xFC00U, 0x82001250U},
    {0xFC01U, 0x82001252U},
    {0xFC02U, 0x82001254U},
    {0xFC03U, 0x8200124DU},
    {0xFC04U, 0x82001256U},
    {0xFC05U, 0x82001258U},
    {0xFC06U, 0x8200125AU},
    {0xFC07U, 0x8200125CU},
    {0xFC08U, 0x8200125EU},
    {0xFC09U, 0x82001260U},
    {0xFC0AU, 0x82001262U},
    {0xFC0BU, 0x82001264U},
    {0xFC0CU, 0x82001266U},
    {0xFC0DU, 0x82001268U},
    {0xFC0EU, 0x8200126AU},
    {0xFC0FU, 0x8200126CU},
    {0xFC10U, 0x8200126EU},
    {0xFC11U, 0x82001270U},
    {0xFC12U, 0x82001272U},
    {0xFC13U, 0x82001274U},
    {0xFC14U, 0x82001276U},
    {0xFC15U, 0x82001278U},
    {0xFC16U, 0x8200127AU},
    {0xFC17U, 0x8200127CU},
    {0xFC18U, 0x8200127EU},
    {0xFC19U, 0x82001280U},
    {0xFC1AU, 0x82001282U},
    {0xFC1BU, 0x82001284U},
    {0xFC1CU, 0x82001286U},
    {0xFC1DU, 0x82001288U},
    {0xFC1EU, 0x8200128AU},
    {0xFC1FU, 0x8200128CU},
    {0xFC20U, 0x8200128EU},
    {0xFC21U, 0x82001290U},
    {0xFC22U, 0x82001292U},
    {0xFC23U, 0x82001294U},
    {0xFC24U, 0x82001296U},
    {0xFC25U, 0x82001298U},
    {0xFC26U, 0x8200129AU},
    {0xFC27U, 0x8200129CU},
    {0xFC28U, 0x8200129EU},
    {0xFC29U, 0x820012A0U},
    {0xFC2AU, 0x820012A2U},
    {0xFC2BU, 0x820012A4U},
    {0xFC2CU, 0x820012A6U},
    {0xFC2DU, 0x820012A8U},
    {0xFC2EU, 0x820012AAU},
    {0xFC2FU, 0x820012ACU},
    {0xFC30U, 0x820012AEU},
    {0xFC31U, 0x820012B0U},
    {0xFC32U, 0x820012B2U},
    {0xFC33U, 0x820012B4U},
    {0xFC34U, 0x820012B6U},
    {0xFC35U, 0x820012B8U},
    {0xFC36U, 0x820012BAU},
    {0xFC37U, 0x820012BCU},
    {0xFC38U, 0x820012BEU},
    {0xFC39U, 0x820012C0U},
    {0xFC3AU, 0x820012C2U},
    {0xFC3BU, 0x820012C4U},
    {0xFC3CU, 0x820012C6U},
    {0xFC3DU, 0x820012C8U},
    {0xFC3EU, 0x820012CAU},
    {0xFC3FU, 0x820012CCU},
    {0xFC40U, 0x820012CEU},
    {0xFC41U, 0x820012D0U},
    {0xFC42U, 0x820012D2U},
    {0xFC43U, 0x820012D4U},
    {0xFC44U, 0x820012D6U},
    {0xFC45U, 0x820012D8U},
    {0xFC46U, 0x820012DAU},
    {0xFC47U, 0x820012DCU},
    {0xFC48U, 0x820012DEU},
    {0xFC49U, 0x820012E0U},
    {0xFC4AU, 0x820012E2U},
    {0xFC4BU, 0x820012E4U},
    {0xFC4CU, 0x820012E6U},
    {0xFC4DU, 0x820012E8U},
    {0xFC4EU, 0x820012EAU},
    {0xFC4FU, 0x820012ECU},
    {0xFC50U, 0x820012EEU},
    {0xFC51U, 0x820012F0U},
    {0xFC52U, 0x820012F2U},
    {0xFC53U, 0x820012F4U},
    {0xFC54U, 0x820012F6U},
    {0xFC55U, 0x820012F8U},
    {0xFC56U, 0x820012FAU},
    {0xFC57U, 0x820012FCU},
    {0xFC58U, 0x820012FEU},
    {0xFC59U, 0x82001300U},
    {0xFC5AU, 0x82001302U},
    {0xFC5BU, 0x82001304U},
    {0xFC5CU, 0x82001306U},
    {0xFC5DU, 0x82001308U},
    {0xFC5EU, 0x8300130AU},
    {0xFC5FU, 0x8300130DU},
    {0xFC60U, 0x83001310U},
    {0xFC61U, 0x83001313U},
    {0xFC62U, 0x83001316U},
    {0xFC63U, 0x83001319U},
    {0xFC64U, 0x8200131CU},
    {0xFC65U, 0x8200131EU},
    {0xFC66U, 0x82001254U},
    {0xFC67U, 0x82001320U},
    {0xFC68U, 0x8200124DU},
    {0xFC69U, 0x82001256U},
    {0xFC6AU, 0x82001322U},
    {0xFC6BU, 0x82001324U},
    {0xFC6CU, 0x8200125EU},
    {0xFC6DU, 0x82001326U},
    {0xFC6EU, 0x82001260U},
    {0xFC6FU, 0x82001262U},
    {0xFC70U, 0x82001328U},
    {0xFC71U, 0x8200132AU},
    {0xFC72U, 0x8200126AU},
    {0xFC73U, 0x8200132CU},
    {0xFC74U, 0x8200126CU},
    {0xFC75U, 0x8200126EU},
    {0xFC76U, 0x8200132EU},
    {0xFC77U, 0x82001330U},
    {0xFC78U, 0x82001272U},
    {0xFC79U, 0x82001332U},
    {0xFC7AU, 0x82001274U},
    {0xFC7BU, 0x82001276U},
    {0xFC7CU, 0x820012B0U},
    {0xFC7DU, 0x820012B2U},
    {0xFC7EU, 0x820012B8U},
    {0xFC7FU, 0x820012BAU},
    {0xFC80U, 0x820012BCU},
    {0xFC81U, 0x820012C4U},
    {0xFC82U, 0x820012C6U},
    {0xFC83U, 0x820012C8U},
    {0xFC84U, 0x820012CAU},
    {0xFC85U, 0x820012D2U},
    {0xFC86U, 0x820012D4U},
    {0xFC87U, 0x820012D6U},
    {0xFC88U, 0x82001334U},
    {0xFC89U, 0x820012DEU},
    {0xFC8AU, 0x82001336U},
    {0xFC8BU, 0x82001338U},
    {0xFC8CU, 0x820012EAU},
    {0xFC8DU, 0x8200133AU},
    {0xFC8EU, 0x820012ECU},
    {0xFC8FU, 0x820012EEU},
    {0xFC90U, 0x82001308U},
    {0xFC91U, 0x8200133CU},
    {0xFC92U, 0x8200133EU},
    {0xFC93U, 0x820012FEU},
    {0xFC94U, 0x82001340U},
    {0xFC95U, 0x82001300U},
    {0xFC96U, 0x82001302U},
    {0xFC97U, 0x82001250U},
    {0xFC98U, 0x82001252U},
    {0xFC99U, 0x82001342U},
    {0xFC9AU, 0x82001254U},
    {0xFC9BU, 0x82001344U},
    {0xFC9CU, 0x82001258U},
    {0xFC9DU, 0x8200125AU},
    {0xFC9EU, 0x8200125CU},
    {0xFC9FU, 0x8200125EU},
    {0xFCA0U, 0x82001346U},
    {0xFCA1U, 0x82001264U},
    {0xFCA2U, 0x82001266U},
    {0xFCA3U, 0x82001268U},
    {0xFCA4U, 0x8200126AU},
    {0xFCA5U, 0x82001348U},
    {0xFCA6U, 0x82001272U},
    {0xFCA7U, 0x82001278U},
    {0xFCA8U, 0x8200127AU},
    {0xFCA9U, 0x8200127CU},
    {0xFCAAU, 0x8200127EU},
    {0xFCABU, 0x82001280U},
    {0xFCACU, 0x82001284U},
    {0xFCADU, 0x82001286U},
    {0xFCAEU, 0x82001288U},
    {0xFCAFU, 0x8200128AU},
    {0xFCB0U, 0x8200128CU},
    {0xFCB1U, 0x8200128EU},
    {0xFCB2U, 0x8200134AU},
    {0xFCB3U, 0x82001290U},
    {0xFCB4U, 0x82001292U},
    {0xFCB5U, 0x82001294U},
    {0xFCB6U, 0x82001296U},
    {0xFCB7U, 0x82001298U},
    {0xFCB8U, 0x8200129AU},
    {0xFCB9U, 0x8200129EU},
    {0xFCBAU, 0x820012A0U},
    {0xFCBBU, 0x820012A2U},
    {0xFCBCU, 0x820012A4U},
    {0xFCBDU, 0x820012A6U},
    {0xFCBEU, 0x820012A8U},
    {0xFCBFU, 0x820012AAU},
    {0xFCC0U, 0x820012ACU},
    {0xFCC1U, 0x820012AEU},
    {0xFCC2U, 0x820012B4U},
    {0xFCC3U, 0x820012B6U},
    {0xFCC4U, 0x820012BEU},
    {0xFCC5U, 0x820012C0U},
    {0xFCC6U, 0x820012C2U},
    {0xFCC7U, 0x820012C4U},
    {0xFCC8U, 0x820012C6U},
    {0xFCC9U, 0x820012CCU},
    {0xFCCAU, 0x820012CEU},
    {0xFCCBU, 0x820012D0U},
    {0xFCCCU, 0x820012D2U},
    {0xFCCDU, 0x8200134CU},
    {0xFCCEU, 0x820012D8U},
    {0xFCCFU, 0x820012DAU},
    {0xFCD0U, 0x820012DCU},
    {0xFCD1U, 0x820012DEU},
    {0xFCD2U, 0x820012E4U},
    {0xFCD3U, 0x820012E6U},
    {0xFCD4U, 0x820012E8U},
    {0xFCD5U, 0x820012EAU},
    {0xFCD6U, 0x8200134EU},
    {0xFCD7U, 0x820012F0U},
    {0xFCD8U, 0x820012F2U},
    {0xFCD9U, 0x82001350U},
    {0xFCDAU, 0x820012F8U},
    {0xFCDBU, 0x820012FAU},
    {0xFCDCU, 0x820012FCU},
    {0xFCDDU, 0x820012FEU},
    {0xFCDEU, 0x82001352U},
    {0xFCDFU, 0x82001254U},
    {0xFCE0U, 0x82001344U},
    {0xFCE1U, 0x8200125EU},
    {0xFCE2U, 0x82001346U},
    {0xFCE3U, 0x8200126AU},
    {0xFCE4U, 0x82001348U},
    {0xFCE5U, 0x82001272U},
    {0xFCE6U, 0x82001354U},
    {0xFCE7U, 0x8200128CU},
    {0xFCE8U, 0x82001356U},
    {0xFCE9U, 0x82001358U},
    {0xFCEAU, 0x8200135AU},
    {0xFCEBU, 0x820012C4U},
    {0xFCECU, 0x820012C6U},
    {0xFCEDU, 0x820012D2U},
    {0xFCEEU, 0x820012EAU},
    {0xFCEFU, 0x8200134EU},
    {0xFCF0U, 0x820012FEU},
    {0xFCF1U, 0x82001352U},
    {0xFCF2U, 0x8300135CU},
    {0xFCF3U, 0x8300135FU},
    {0xFCF4U, 0x83001362U},
    {0xFCF5U, 0x82001365U},
    {0xFCF6U, 0x82001367U},
    {0xFCF7U, 0x82001369U},
    {0xFCF8U, 0x8200136BU},
    {0xFCF9U, 0x8200136DU},
    {0xFCFAU, 0x8200136FU},
    {0xFCFBU, 0x82001371U},
    {0xFCFCU, 0x82001373U},
    {0xFCFDU, 0x82001375U},
    {0xFCFEU, 0x82001377U},
    {0xFCFFU, 0x82001379U},
    {0xFD00U, 0x8200137BU},
    {0xFD01U, 0x8200137DU},
    {0xFD02U, 0x8200137FU},
    {0xFD03U, 0x82001381U},
    {0xFD04U, 0x82001383U},
    {0xFD05U, 0x82001385U},
    {0xFD06U, 0x82001387U},
    {0xFD07U, 0x82001389U},
    {0xFD08U, 0x8200138BU},
    {0xFD09U, 0x8200138DU},
    {0xFD0AU, 0x8200138FU},
    {0xFD0BU, 0x82001391U},
    {0xFD0CU, 0x82001358U},
    {0xFD0DU, 0x82001393U},
    {0xFD0EU, 0x82001395U},
    {0xFD0FU, 0x82001397U},
    {0xFD10U, 0x82001399U},
    {0xFD11U, 0x82001365U},
    {0xFD12U, 0x82001367U},
    {0xFD13U, 0x82001369U},
    {0xFD14U, 0x8200136BU},
    {0xFD15U, 0x8200136DU},
    {0xFD16U, 0x8200136FU},
    {0xFD17U, 0x82001371U},
    {0xFD18U, 0x82001373U},
    {0xFD19U, 0x82001375U},
    {0xFD1AU, 0x82001377U},
    {0xFD1BU, 0x82001379U},
    {0xFD1CU, 0x8200137BU},
    {0xFD1DU, 0x8200137DU},
    {0xFD1EU, 0x8200137FU},
    {0xFD1FU, 0x82001381U},
    {0xFD20U, 0x82001383U},
    {0xFD21U, 0x82001385U},
    {0xFD22U, 0x82001387U},
    {0xFD23U, 0x82001389U},
    {0xFD24U, 0x8200138BU},
    {0xFD25U, 0x8200138DU},
    {0xFD26U, 0x8200138FU},
    {0xFD27U, 0x82001391U},
    {0xFD28U, 0x82001358U},
    {0xFD29U, 0x82001393U},
    {0xFD2AU, 0x82001395U},
    {0xFD2BU, 0x82001397U},
    {0xFD2CU, 0x82001399U},
    {0xFD2DU, 0x8200138DU},
    {0xFD2EU, 0x8200138FU},
    {0xFD2FU, 0x82001391U},
    {0xFD30U, 0x82001358U},
    {0xFD31U, 0x82001356U},
    {0xFD32U, 0x8200135AU},
    {0xFD33U, 0x8200129CU},
    {0xFD34U, 0x82001286U},
    {0xFD35U, 0x82001288U},
    {0xFD36U, 0x8200128AU},
    {0xFD37U, 0x8200138DU},
    {0xFD38U, 0x8200138FU},
    {0xFD39U, 0x82001391U},
    {0xFD3AU, 0x8200129CU},
    {0xFD3BU, 0x8200129EU},
    {0xFD3CU, 0x8200139BU},
    {0xFD3DU, 0x8200139BU},
    {0xFD50U, 0x8300139DU},
    {0xFD51U, 0x830013A0U},
    {0xFD52U, 0x830013A0U},
    {0xFD53U, 0x830013A3U},
    {0xFD54U, 0x830013A6U},
    {0xFD55U, 0x830013A9U},
    {0xFD56U, 0x830013ACU},
    {0xFD57U, 0x830013AFU},
    {0xFD58U, 0x830013B2U},
    {0xFD59U, 0x830013B2U},
    {0xFD5AU, 0x830013B5U},
    {0xFD5BU, 0x830013B8U},
    {0xFD5CU, 0x830013BBU},
    {0xFD5DU, 0x830013BEU},
    {0xFD5EU, 0x830013C1U},
    {0xFD5FU, 0x830013C4U},
    {0xFD60U, 0x830013C4U},
    {0xFD61U, 0x830013C7U},
    {0xFD62U, 0x830013CAU},
    {0xFD63U, 0x830013CAU},
    {0xFD64U, 0x830013CDU},
    {0xFD65U, 0x830013CDU},
    {0xFD66U, 0x830013D0U},
    {0xFD67U, 0x830013D3U},
    {0xFD68U, 0x830013D3U},
    {0xFD69U, 0x830013D6U},
    {0xFD6AU, 0x830013D9U},
    {0xFD6BU, 0x830013D9U},
    {0xFD6CU, 0x830013DCU},
    {0xFD6DU, 0x830013DCU},
    {0xFD6EU, 0x830013DFU},
    {0xFD6FU, 0x830013E2U},
    {0xFD70U, 0x830013E2U},
    {0xFD71U, 0x830013E5U},
    {0xFD72U, 0x830013E5U},
    {0xFD73U, 0x830013E8U},
    {0xFD74U, 0x830013EBU},
    {0xFD75U, 0x830013EEU},
    {0xFD76U, 0x830013F1U},
    {0xFD77U, 0x830013F1U},
    {0xFD78U, 0x830013F4U},
    {0xFD79U, 0x830013F7U},
    {0xFD7AU, 0x830013FAU},
    {0xFD7BU, 0x830013FDU},
    {0xFD7CU, 0x83001400U},
    {0xFD7DU, 0x83001400U},
    {0xFD7EU, 0x83001403U},
    {0xFD7FU, 0x83001406U},
    {0xFD80U, 0x83001409U},
    {0xFD81U, 0x8300140CU},
    {0xFD82U, 0x8300140FU},
    {0xFD83U, 0x83001412U},
    {0xFD84U, 0x83001412U},
    {0xFD85U, 0x83001415U},
    {0xFD86U, 0x83001415U},
    {0xFD87U, 0x83001418U},
    {0xFD88U, 0x83001418U},
    {0xFD89U, 0x8300141BU},
    {0xFD8AU, 0x8300141EU},
    {0xFD8BU, 0x83001421U},
    {0xFD8CU, 0x83001424U},
    {0xFD8DU, 0x83001427U},
    {0xFD8EU, 0x8300142AU},
    {0xFD8FU, 0x8300142DU},
    {0xFD92U, 0x83001430U},
    {0xFD93U, 0x83001433U},
    {0xFD94U, 0x83001436U},
    {0xFD95U, 0x83001439U},
    {0xFD96U, 0x8300143CU},
    {0xFD97U, 0x8300143FU},
    {0xFD98U, 0x8300143FU},
    {0xFD99U, 0x83001442U},
    {0xFD9AU, 0x83001445U},
    {0xFD9BU, 0x83001448U},
    {0xFD9CU, 0x8300144BU},
    {0xFD9DU, 0x8300144BU},
    {0xFD9EU, 0x8300144EU},
    {0xFD9FU, 0x83001451U},
    {0xFDA0U, 0x83001454U},
    {0xFDA1U, 0x83001457U},
    {0xFDA2U, 0x8300145AU},
    {0xFDA3U, 0x8300145DU},
    {0xFDA4U, 0x83001460U},
    {0xFDA5U, 0x83001463U},
    {0xFDA6U, 0x83001466U},
    {0xFDA7U, 0x83001469U},
    {0xFDA8U, 0x8300146CU},
    {0xFDA9U, 0x8300146FU},
    {0xFDAAU, 0x83001472U},
    {0xFDABU, 0x83001475U},
    {0xFDACU, 0x83001478U},
    {0xFDADU, 0x8300147BU},
    {0xFDAEU, 0x8300147EU},
    {0xFDAFU, 0x83001481U},
    {0xFDB0U, 0x83001484U},
    {0xFDB1U, 0x83001487U},
    {0xFDB2U, 0x8300148AU},
    {0xFDB3U, 0x8300148DU},
    {0xFDB4U, 0x83001403U},
    {0xFDB5U, 0x83001409U},
    {0xFDB6U, 0x83001490U},
    {0xFDB7U, 0x83001493U},
    {0xFDB8U, 0x83001496U},
    {0xFDB9U, 0x83001499U},
    {0xFDBAU, 0x8300149CU},
    {0xFDBBU, 0x8300149FU},
    {0xFDBCU, 0x8300149CU},
    {0xFDBDU, 0x83001496U},
    {0xFDBEU, 0x830014A2U},
    {0xFDBFU, 0x830014A5U},
    {0xFDC0U, 0x830014A8U},
    {0xFDC1U, 0x830014ABU},
    {0xFDC2U, 0x830014AEU},
    {0xFDC3U, 0x8300149FU},
    {0xFDC4U, 0x830013EEU},
    {0xFDC5U, 0x830013D0U},
    {0xFDC6U, 0x830014B1U},
    {0xFDC7U, 0x830014B4U},
    {0xFDF0U, 0x830014B7U},
    {0xFDF1U, 0x830014BAU},
    {0xFDF2U, 0x840014BDU},
    {0xFDF3U, 0x840014C1U},
    {0xFDF4U, 0x840014C5U},
    {0xFDF5U, 0x840014C9U},
    {0xFDF6U, 0x840014CDU},
    {0xFDF7U, 0x840014D1U},
    {0xFDF8U, 0x840014D5U},
    {0xFDF9U, 0x830014D9U},
    {0xFDFAU, 0x920014DCU},
    {0xFDFBU, 0x880014EEU},
    {0xFDFCU, 0x840014F6U},
    {0xFE10U, 0x810014FAU},
    {0xFE11U, 0x810014FBU},
    {0xFE12U, 0x810014FCU},
    {0xFE13U, 0x810014FDU},
    {0xFE14U, 0x81000254U},
    {0xFE15U, 0x810014FEU},
    {0xFE16U, 0x810014FFU},
    {0xFE17U, 0x81001500U},
    {0xFE18U, 0x81001501U},
    {0xFE19U, 0x81001502U},
    {0xFE30U, 0x81001503U},
    {0xFE31U, 0x81001504U},
    {0xFE32U, 0x81001505U},
    {0xFE33U, 0x81001506U},
    {0xFE34U, 0x81001506U},
    {0xFE35U, 0x810007DAU},
    {0xFE36U, 0x810007DBU},
    {0xFE37U, 0x81001507U},
    {0xFE38U, 0x81001508U},
    {0xFE39U, 0x81001509U},
    {0xFE3AU, 0x8100150AU},
    {0xFE3BU, 0x8100150BU},
    {0xFE3CU, 0x8100150CU},
    {0xFE3DU, 0x8100150DU},
    {0xFE3EU, 0x8100150EU},
    {0xFE3FU, 0x810008CFU},
    {0xFE40U, 0x810008D0U},
    {0xFE41U, 0x8100150FU},
    {0xFE42U, 0x81001510U},
    {0xFE43U, 0x81001511U},
    {0xFE44U, 0x81001512U},
    {0xFE47U, 0x81001513U},
    {0xFE48U, 0x81001514U},
    {0xFE49U, 0x81001515U},
    {0xFE4AU, 0x81001515U},
    {0xFE4BU, 0x81001515U},
    {0xFE4CU, 0x81001515U},
    {0xFE4DU, 0x81001506U},
    {0xFE4EU, 0x81001506U},
    {0xFE4FU, 0x81001506U},
    {0xFE50U, 0x810014FAU},
    {0xFE51U, 0x810014FBU},
    {0xFE52U, 0x810007B2U},
    {0xFE54U, 0x81000254U},
    {0xFE55U, 0x810014FDU},
    {0xFE56U, 0x810014FFU},
    {0xFE57U, 0x810014FEU},
    {0xFE58U, 0x81001504U},
    {0xFE59U, 0x810007DAU},
    {0xFE5AU, 0x810007DBU},
    {0xFE5BU, 0x81001507U},
    {0xFE5CU, 0x81001508U},
    {0xFE5DU, 0x81001509U},
    {0xFE5EU, 0x8100150AU},
    {0xFE5FU, 0x81001516U},
    {0xFE60U, 0x81001517U},
    {0xFE61U, 0x81001518U},
    {0xFE62U, 0x810007D7U},
    {0xFE63U, 0x81001519U},
    {0xFE64U, 0x8100151AU},
    {0xFE65U, 0x8100151BU},
    {0xFE66U, 0x810007D9U},
    {0xFE68U, 0x8100151CU},
    {0xFE69U, 0x8100151DU},
    {0xFE6AU, 0x8100151EU},
    {0xFE6BU, 0x8100151FU},
    {0xFE70U, 0x82001520U},
    {0xFE71U, 0x82001522U},
    {0xFE72U, 0x82001524U},
    {0xFE74U, 0x82001526U},
    {0xFE76U, 0x82001528U},
    {0xFE77U, 0x8200152AU},
    {0xFE78U, 0x8200152CU},
    {0xFE79U, 0x8200152EU},
    {0xFE7AU, 0x82001530U},
    {0xFE7BU, 0x82001532U},
    {0xFE7CU, 0x82001534U},
    {0xFE7DU, 0x82001536U},
    {0xFE7EU, 0x82001538U},
    {0xFE7FU, 0x8200153AU},
    {0xFE80U, 0x8100153CU},
    {0xFE81U, 0x8100153DU},
    {0xFE82U, 0x8100153DU},
    {0xFE83U, 0x8100153EU},
    {0xFE84U, 0x8100153EU},
    {0xFE85U, 0x8100153FU},
    {0xFE86U, 0x8100153FU},
    {0xFE87U, 0x81001540U},
    {0xFE88U, 0x81001540U},
    {0xFE89U, 0x81001541U},
    {0xFE8AU, 0x81001541U},
    {0xFE8BU, 0x81001541U},
    {0xFE8CU, 0x81001541U},
    {0xFE8DU, 0x81001542U},
    {0xFE8EU, 0x81001542U},
    {0xFE8FU, 0x81001543U},
    {0xFE90U, 0x81001543U},
    {0xFE91U, 0x81001543U},
    {0xFE92U, 0x81001543U},
    {0xFE93U, 0x81001544U},
    {0xFE94U, 0x81001544U},
    {0xFE95U, 0x81001545U},
    {0xFE96U, 0x81001545U},
    {0xFE97U, 0x81001545U},
    {0xFE98U, 0x81001545U},
    {0xFE99U, 0x81001546U},
    {0xFE9AU, 0x81001546U},
    {0xFE9BU, 0x81001546U},
    {0xFE9CU, 0x81001546U},
    {0xFE9DU, 0x81001547U},
    {0xFE9EU, 0x81001547U},
    {0xFE9FU, 0x81001547U},
    {0xFEA0U, 0x81001547U},
    {0xFEA1U, 0x81001548U},
    {0xFEA2U, 0x81001548U},
    {0xFEA3U, 0x81001548U},
    {0xFEA4U, 0x81001548U},
    {0xFEA5U, 0x81001549U},
    {0xFEA6U, 0x81001549U},
    {0xFEA7U, 0x81001549U},
    {0xFEA8U, 0x81001549U},
    {0xFEA9U, 0x8100154AU},
    {0xFEAAU, 0x8100154AU},
    {0xFEABU, 0x8100154BU},
    {0xFEACU, 0x8100154BU},
    {0xFEADU, 0x8100154CU},
    {0xFEAEU, 0x8100154CU},
    {0xFEAFU, 0x8100154DU},
    {0xFEB0U, 0x8100154DU},
    {0xFEB1U, 0x8100154EU},
    {0xFEB2U, 0x8100154EU},
    {0xFEB3U, 0x8100154EU},
    {0xFEB4U, 0x8100154EU},
    {0xFEB5U, 0x8100154FU},
    {0xFEB6U, 0x8100154FU},
    {0xFEB7U, 0x8100154FU},
    {0xFEB8U, 0x8100154FU},
    {0xFEB9U, 0x81001550U},
    {0xFEBAU, 0x81001550U},
    {0xFEBBU, 0x81001550U},
    {0xFEBCU, 0x81001550U},
    {0xFEBDU, 0x81001551U},
    {0xFEBEU, 0x81001551U},
    {0xFEBFU, 0x81001551U},
    {0xFEC0U, 0x81001551U},
    {0xFEC1U, 0x81001552U},
    {0xFEC2U, 0x81001552U},
    {0xFEC3U, 0x81001552U},
    {0xFEC4U, 0x81001552U},
    {0xFEC5U, 0x81001553U},
    {0xFEC6U, 0x81001553U},
    {0xFEC7U, 0x81001553U},
    {0xFEC8U, 0x81001553U},
    {0xFEC9U, 0x81001554U},
    {0xFECAU, 0x81001554U},
    {0xFECBU, 0x81001554U},
    {0xFECCU, 0x81001554U},
    {0xFECDU, 0x81001555U},
    {0xFECEU, 0x81001555U},
    {0xFECFU, 0x81001555U},
    {0xFED0U, 0x81001555U},
    {0xFED1U, 0x81001556U},
    {0xFED2U, 0x81001556U},
    {0xFED3U, 0x81001556U},
    {0xFED4U, 0x81001556U},
    {0xFED5U, 0x81001557U},
    {0xFED6U, 0x81001557U},
    {0xFED7U, 0x81001557U},
    {0xFED8U, 0x81001557U},
    {0xFED9U, 0x81001558U},
    {0xFEDAU, 0x81001558U},
    {0xFEDBU, 0x81001558U},
    {0xFEDCU, 0x81001558U},
    {0xFEDDU, 0x81001559U},
    {0xFEDEU, 0x81001559U},
    {0xFEDFU, 0x81001559U},
    {0xFEE0U, 0x81001559U},
    {0xFEE1U, 0x8100155AU},
    {0xFEE2U, 0x8100155AU},
    {0xFEE3U, 0x8100155AU},
    {0xFEE4U, 0x8100155AU},
    {0xFEE5U, 0x8100155BU},
    {0xFEE6U, 0x8100155BU},
    {0xFEE7U, 0x8100155BU},
    {0xFEE8U, 0x8100155BU},
    {0xFEE9U, 0x8100155CU},
    {0xFEEAU, 0x8100155CU},
    {0xFEEBU, 0x8100155CU},
    {0xFEECU, 0x8100155CU},
    {0xFEEDU, 0x8100155DU},
    {0xFEEEU, 0x8100155DU},
    {0xFEEFU, 0x8100123EU},
    {0xFEF0U, 0x8100123EU},
    {0xFEF1U, 0x8100155EU},
    {0xFEF2U, 0x8100155EU},
    {0xFEF3U, 0x8100155EU},
    {0xFEF4U, 0x8100155EU},
    {0xFEF5U, 0x8200155FU},
    {0xFEF6U, 0x8200155FU},
    {0xFEF7U, 0x82001561U},
    {0xFEF8U, 0x82001561U},
    {0xFEF9U, 0x82001563U},
    {0xFEFAU, 0x82001563U},
    {0xFEFBU, 0x82001565U},
    {0xFEFCU, 0x82001565U},
    {0xFF01U, 0x810014FEU},
    {0xFF02U, 0x81001567U},
    {0xFF03U, 0x81001516U},
    {0xFF04U, 0x8100151DU},
    {0xFF05U, 0x8100151EU},
    {0xFF06U, 0x81001517U},
    {0xFF07U, 0x81001568U},
    {0xFF08U, 0x810007DAU},
    {0xFF09U, 0x810007DBU},
    {0xFF0AU, 0x81001518U},
    {0xFF0BU, 0x810007D7U},
    {0xFF0CU, 0x810014FAU},
    {0xFF0DU, 0x81001519U},
    {0xFF0EU, 0x810007B2U},
    {0xFF0FU, 0x81001569U},
    {0xFF10U, 0x810007D0U},
    {0xFF11U, 0x8100000DU},
    {0xFF12U, 0x81000006U},
    {0xFF13U, 0x81000007U},
    {0xFF14U, 0x810007D1U},
    {0xFF15U, 0x810007D2U},
    {0xFF16U, 0x810007D3U},
    {0xFF17U, 0x810007D4U},
    {0xFF18U, 0x810007D5U},
    {0xFF19U, 0x810007D6U},
    {0xFF1AU, 0x810014FDU},
    {0xFF1BU, 0x81000254U},
    {0xFF1CU, 0x8100151AU},
    {0xFF1DU, 0x810007D9U},
    {0xFF1EU, 0x8100151BU},
    {0xFF1FU, 0x810014FFU},
    {0xFF20U, 0x8100151FU},
    {0xFF21U, 0x810003B1U},
    {0xFF22U, 0x810003B3U},
    {0xFF23U, 0x810007E5U},
    {0xFF24U, 0x810003B4U},
    {0xFF25U, 0x810003B5U},
    {0xFF26U, 0x810007FFU},
    {0xFF27U, 0x810003B7U},
    {0xFF28U, 0x810003B8U},
    {0xFF29U, 0x810003B9U},
    {0xFF2AU, 0x810003BAU},
    {0xFF2BU, 0x810003BBU},
    {0xFF2CU, 0x810003BCU},
    {0xFF2DU, 0x810003BDU},
    {0xFF2EU, 0x810003BEU},
    {0xFF2FU, 0x810003BFU},
    {0xFF30U, 0x810003C1U},
    {0xFF31U, 0x810007F4U},
    {0xFF32U, 0x810003C2U},
    {0xFF33U, 0x810009AFU},
    {0xFF34U, 0x810003C3U},
    {0xFF35U, 0x810003C4U},
    {0xFF36U, 0x81000841U},
    {0xFF37U, 0x810003C5U},
    {0xFF38U, 0x8100084DU},
    {0xFF39U, 0x810009B0U},
    {0xFF3AU, 0x810007FCU},
    {0xFF3BU, 0x81001513U},
    {0xFF3CU, 0x8100151CU},
    {0xFF3DU, 0x81001514U},
    {0xFF3EU, 0x8100156AU},
    {0xFF3FU, 0x81001506U},
    {0xFF40U, 0x81000797U},
    {0xFF41U, 0x81000003U},
    {0xFF42U, 0x810003C9U},
    {0xFF43U, 0x810003E3U},
    {0xFF44U, 0x810003CAU},
    {0xFF45U, 0x810003CBU},
    {0xFF46U, 0x810003E6U},
    {0xFF47U, 0x810003CFU},
    {0xFF48U, 0x81000233U},
    {0xFF49U, 0x810003E0U},
    {0xFF4AU, 0x81000235U},
    {0xFF4BU, 0x810003D0U},
    {0xFF4CU, 0x81000249U},
    {0xFF4DU, 0x810003D1U},
    {0xFF4EU, 0x810007DCU},
    {0xFF4FU, 0x8100000EU},
    {0xFF50U, 0x810003D6U},
    {0xFF51U, 0x810009B1U},
    {0xFF52U, 0x81000236U},
    {0xFF53U, 0x81000164U},
    {0xFF54U, 0x810003D7U},
    {0xFF55U, 0x810003D8U},
    {0xFF56U, 0x810003DBU},
    {0xFF57U, 0x8100023AU},
    {0xFF58U, 0x8100024AU},
    {0xFF59U, 0x8100023BU},
    {0xFF5AU, 0x81000401U},
    {0xFF5BU, 0x81001507U},
    {0xFF5CU, 0x8100156BU},
    {0xFF5DU, 0x81001508U},
    {0xFF5EU, 0x8100156CU},
    {0xFF5FU, 0x8100156DU},
    {0xFF60U, 0x8100156EU},
    {0xFF61U, 0x810014FCU},
    {0xFF62U, 0x8100150FU},
    {0xFF63U, 0x81001510U},
    {0xFF64U, 0x810014FBU},
    {0xFF65U, 0x8100156FU},
    {0xFF66U, 0x81000D3BU},
    {0xFF67U, 0x81001570U},
    {0xFF68U, 0x81001571U},
    {0xFF69U, 0x81001572U},
    {0xFF6AU, 0x81001573U},
    {0xFF6BU, 0x81001574U},
    {0xFF6CU, 0x81001575U},
    {0xFF6DU, 0x81001576U},
    {0xFF6EU, 0x81001577U},
    {0xFF6FU, 0x81001578U},
    {0xFF70U, 0x81001579U},
    {0xFF71U, 0x81000D0DU},
    {0xFF72U, 0x81000D0EU},
    {0xFF73U, 0x81000D0FU},
    {0xFF74U, 0x81000D10U},
    {0xFF75U, 0x81000D11U},
    {0xFF76U, 0x81000D12U},
    {0xFF77U, 0x81000D13U},
    {0xFF78U, 0x81000D14U},
    {0xFF79U, 0x81000D15U},
    {0xFF7AU, 0x81000D16U},
    {0xFF7BU, 0x81000D17U},
    {0xFF7CU, 0x81000D18U},
    {0xFF7DU, 0x81000D19U},
    {0xFF7EU, 0x81000D1AU},
    {0xFF7FU, 0x81000D1BU},
    {0xFF80U, 0x81000D1CU},
    {0xFF81U, 0x81000D1DU},
    {0xFF82U, 0x81000D1EU},
    {0xFF83U, 0x81000D1FU},
    {0xFF84U, 0x81000D20U},
    {0xFF85U, 0x81000D21U},
    {0xFF86U, 0x81000D22U},
    {0xFF87U, 0x81000D23U},
    {0xFF88U, 0x81000D24U},
    {0xFF89U, 0x81000D25U},
    {0xFF8AU, 0x81000D26U},
    {0xFF8BU, 0x81000D27U},
    {0xFF8CU, 0x81000D28U},
    {0xFF8DU, 0x81000D29U},
    {0xFF8EU, 0x81000D2AU},
    {0xFF8FU, 0x81000D2BU},
    {0xFF90U, 0x81000D2CU},
    {0xFF91U, 0x81000D2DU},
    {0xFF92U, 0x81000D2EU},
    {0xFF93U, 0x81000D2FU},
    {0xFF94U, 0x81000D30U},
    {0xFF95U, 0x81000D31U},
    {0xFF96U, 0x81000D32U},
    {0xFF97U, 0x81000D33U},
    {0xFF98U, 0x81000D34U},
    {0xFF99U, 0x81000D35U},
    {0xFF9AU, 0x81000D36U},
    {0xFF9BU, 0x81000D37U},
    {0xFF9CU, 0x81000D38U},
    {0xFF9DU, 0x8100157AU},
    {0xFF9EU, 0x8100157BU},
    {0xFF9FU, 0x8100157CU},
    {0xFFA0U, 0x8100157DU},
    {0xFFA1U, 0x8100157EU},
    {0xFFA2U, 0x8100157FU},
    {0xFFA3U, 0x81001580U},
    {0xFFA4U, 0x81001581U},
    {0xFFA5U, 0x81001582U},
    {0xFFA6U, 0x81001583U},
    {0xFFA7U, 0x81001584U},
    {0xFFA8U, 0x81001585U},
    {0xFFA9U, 0x81001586U},
    {0xFFAAU, 0x81001587U},
    {0xFFABU, 0x81001588U},
    {0xFFACU, 0x81001589U},
    {0xFFADU, 0x8100158AU},
    {0xFFAEU, 0x8100158BU},
    {0xFFAFU, 0x8100158CU},
    {0xFFB0U, 0x8100158DU},
    {0xFFB1U, 0x8100158EU},
    {0xFFB2U, 0x8100158FU},
    {0xFFB3U, 0x81001590U},
    {0xFFB4U, 0x81001591U},
    {0xFFB5U, 0x81001592U},
    {0xFFB6U, 0x81001593U},
    {0xFFB7U, 0x81001594U},
    {0xFFB8U, 0x81001595U},
    {0xFFB9U, 0x81001596U},
    {0xFFBAU, 0x81001597U},
    {0xFFBBU, 0x81001598U},
    {0xFFBCU, 0x81001599U},
    {0xFFBDU, 0x8100159AU},
    {0xFFBEU, 0x8100159BU},
    {0xFFC2U, 0x8100159CU},
    {0xFFC3U, 0x8100159DU},
    {0xFFC4U, 0x8100159EU},
    {0xFFC5U, 0x8100159FU},
    {0xFFC6U, 0x810015A0U},
    {0xFFC7U, 0x810015A1U},
    {0xFFCAU, 0x810015A2U},
    {0xFFCBU, 0x810015A3U},
    {0xFFCCU, 0x810015A4U},
    {0xFFCDU, 0x810015A5U},
    {0xFFCEU, 0x810015A6U},
    {0xFFCFU, 0x810015A7U},
    {0xFFD2U, 0x810015A8U},
    {0xFFD3U, 0x810015A9U},
    {0xFFD4U, 0x810015AAU},
    {0xFFD5U, 0x810015ABU},
    {0xFFD6U, 0x810015ACU},
    {0xFFD7U, 0x810015ADU},
    {0xFFDAU, 0x810015AEU},
    {0xFFDBU, 0x810015AFU},
    {0xFFDCU, 0x810015B0U},
    {0xFFE0U, 0x810015B1U},
    {0xFFE1U, 0x810015B2U},
    {0xFFE2U, 0x810015B3U},
    {0xFFE3U, 0x810015B4U},
    {0xFFE4U, 0x810015B5U},
    {0xFFE5U, 0x810015B6U},
    {0xFFE6U, 0x810015B7U},
    {0xFFE8U, 0x810015B8U},
    {0xFFE9U, 0x810015B9U},
    {0xFFEAU, 0x810015BAU},
    {0xFFEBU, 0x810015BBU},
    {0xFFECU, 0x810015BCU},
    {0xFFEDU, 0x810015BDU},
    {0xFFEEU, 0x810015BEU},
    {0x105C9U, 0x020015BFU},
    {0x105E4U, 0x020015C1U},
    {0x10781U, 0x810015C3U},
    {0x10782U, 0x810015C4U},
    {0x10783U, 0x810015C5U},
    {0x10784U, 0x810015C6U},
    {0x10785U, 0x810015C7U},
    {0x10787U, 0x810015C8U},
    {0x10788U, 0x810015C9U},
    {0x10789U, 0x810015CAU},
    {0x1078AU, 0x810015CBU},
    {0x1078BU, 0x810015CCU},
    {0x1078CU, 0x810015CDU},
    {0x1078DU, 0x810015CEU},
    {0x1078EU, 0x810015CFU},
    {0x1078FU, 0x810015D0U},
    {0x10790U, 0x810015D1U},
    {0x10791U, 0x810015D2U},
    {0x10792U, 0x810015D3U},
    {0x10793U, 0x810015D4U},
    {0x10794U, 0x810015D5U},
    {0x10795U, 0x810007F1U},
    {0x10796U, 0x810015D6U},
    {0x10797U, 0x810015D7U},
    {0x10798U, 0x810015D8U},
    {0x10799U, 0x810015D9U},
    {0x1079AU, 0x810015DAU},
    {0x1079BU, 0x810015DBU},
    {0x1079CU, 0x810015DCU},
    {0x1079DU, 0x810015DDU},
    {0x1079EU, 0x810015DEU},
    {0x1079FU, 0x810015DFU},
    {0x107A0U, 0x810015E0U},
    {0x107A1U, 0x810015E1U},
    {0x107A2U, 0x810015E2U},
    {0x107A3U, 0x810015E3U},
    {0x107A4U, 0x810015E4U},
    {0x107A5U, 0x810009B1U},
    {0x107A6U, 0x810015E5U},
    {0x107A7U, 0x810015E6U},
    {0x107A8U, 0x810015E7U},
    {0x107A9U, 0x810015E8U},
    {0x107AAU, 0x810015E9U},
    {0x107ABU, 0x810015EAU},
    {0x107ACU, 0x810015EBU},
    {0x107ADU, 0x810015ECU},
    {0x107AEU, 0x810015EDU},
    {0x107AFU, 0x810015EEU},
    {0x107B0U, 0x810015EFU},
    {0x107B2U, 0x810015F0U},
    {0x107B3U, 0x810015F1U},
    {0x107B4U, 0x810015F2U},
    {0x107B5U, 0x810015F3U},
    {0x107B6U, 0x810015F4U},
    {0x107B7U, 0x810015F5U},
    {0x107B8U, 0x810015F6U},
    {0x107B9U, 0x810015F7U},
    {0x107BAU, 0x810015F8U},
    {0x1109AU, 0x020015F9U},
    {0x1109CU, 0x020015FBU},
    {0x110ABU, 0x020015FDU},
    {0x1112EU, 0x020015FFU},
    {0x1112FU, 0x02001601U},
    {0x1134BU, 0x02001603U},
    {0x1134CU, 0x02001605U},
    {0x11383U, 0x02001607U},
    {0x11385U, 0x02001609U},
    {0x1138EU, 0x0200160BU},
    {0x11391U, 0x0200160DU},
    {0x113C5U, 0x0200160FU},
    {0x113C7U, 0x02001611U},
    {0x113C8U, 0x02001613U},
    {0x114BBU, 0x02001615U},
    {0x114BCU, 0x02001617U},
    {0x114BEU, 0x02001619U},
    {0x115BAU, 0x0200161BU},
    {0x115BBU, 0x0200161DU},
    {0x11938U, 0x0200161FU},
    {0x16121U, 0x02001621U},
    {0x16122U, 0x02001623U},
    {0x16123U, 0x02001625U},
    {0x16124U, 0x02001627U},
    {0x16125U, 0x02001629U},
    {0x16126U, 0x0200162BU},
    {0x16127U, 0x0200162DU},
    {0x16128U, 0x0200162FU},
    {0x16D68U, 0x02001631U},
    {0x16D69U, 0x02001633U},
    {0x16D6AU, 0x02001635U},
    {0x1CCD6U, 0x810003B1U},
    {0x1CCD7U, 0x810003B3U},
    {0x1CCD8U, 0x810007E5U},
    {0x1CCD9U, 0x810003B4U},
    {0x1CCDAU, 0x810003B5U},
    {0x1CCDBU, 0x810007FFU},
    {0x1CCDCU, 0x810003B7U},
    {0x1CCDDU, 0x810003B8U},
    {0x1CCDEU, 0x810003B9U},
    {0x1CCDFU, 0x810003BAU},
    {0x1CCE0U, 0x810003BBU},
    {0x1CCE1U, 0x810003BCU},
    {0x1CCE2U, 0x810003BDU},
    {0x1CCE3U, 0x810003BEU},
    {0x1CCE4U, 0x810003BFU},
    {0x1CCE5U, 0x810003C1U},
    {0x1CCE6U, 0x810007F4U},
    {0x1CCE7U, 0x810003C2U},
    {0x1CCE8U, 0x810009AFU},
    {0x1CCE9U, 0x810003C3U},
    {0x1CCEAU, 0x810003C4U},
    {0x1CCEBU, 0x81000841U},
    {0x1CCECU, 0x810003C5U},
    {0x1CCEDU, 0x8100084DU},
    {0x1CCEEU, 0x810009B0U},
    {0x1CCEFU, 0x810007FCU},
    {0x1CCF0U, 0x810007D0U},
    {0x1CCF1U, 0x8100000DU},
    {0x1CCF2U, 0x81000006U},
    {0x1CCF3U, 0x81000007U},
    {0x1CCF4U, 0x810007D1U},
    {0x1CCF5U, 0x810007D2U},
    {0x1CCF6U, 0x810007D3U},
    {0x1CCF7U, 0x810007D4U},
    {0x1CCF8U, 0x810007D5U},
    {0x1CCF9U, 0x810007D6U},
    {0x1D15EU, 0x02001637U},
    {0x1D15FU, 0x02001639U},
    {0x1D160U, 0x0200163BU},
    {0x1D161U, 0x0200163DU},
    {0x1D162U, 0x0200163FU},
    {0x1D163U, 0x02001641U},
    {0x1D164U, 0x02001643U},
    {0x1D1BBU, 0x02001645U},
    {0x1D1BCU, 0x02001647U},
    {0x1D1BDU, 0x02001649U},
    {0x1D1BEU, 0x0200164BU},
    {0x1D1BFU, 0x0200164DU},
    {0x1D1C0U, 0x0200164FU},
    {0x1D400U, 0x810003B1U},
    {0x1D401U, 0x810003B3U},
    {0x1D402U, 0x810007E5U},
    {0x1D403U, 0x810003B4U},
    {0x1D404U, 0x810003B5U},
    {0x1D405U, 0x810007FFU},
    {0x1D406U, 0x810003B7U},
    {0x1D407U, 0x810003B8U},
    {0x1D408U, 0x810003B9U},
    {0x1D409U, 0x810003BAU},
    {0x1D40AU, 0x810003BBU},
    {0x1D40BU, 0x810003BCU},
    {0x1D40CU, 0x810003BDU},
    {0x1D40DU, 0x810003BEU},
    {0x1D40EU, 0x810003BFU},
    {0x1D40FU, 0x810003C1U},
    {0x1D410U, 0x810007F4U},
    {0x1D411U, 0x810003C2U},
    {0x1D412U, 0x810009AFU},
    {0x1D413U, 0x810003C3U},
    {0x1D414U, 0x810003C4U},
    {0x1D415U, 0x81000841U},
    {0x1D416U, 0x810003C5U},
    {0x1D417U, 0x8100084DU},
    {0x1D418U, 0x810009B0U},
    {0x1D419U, 0x810007FCU},
    {0x1D41AU, 0x81000003U},
    {0x1D41BU, 0x810003C9U},
    {0x1D41CU, 0x810003E3U},
    {0x1D41DU, 0x810003CAU},
    {0x1D41EU, 0x810003CBU},
    {0x1D41FU, 0x810003E6U},
    {0x1D420U, 0x810003CFU},
    {0x1D421U, 0x81000233U},
    {0x1D422U, 0x810003E0U},
    {0x1D423U, 0x81000235U},
    {0x1D424U, 0x810003D0U},
    {0x1D425U, 0x81000249U},
    {0x1D426U, 0x810003D1U},
    {0x1D427U, 0x810007DCU},
    {0x1D428U, 0x8100000EU},
    {0x1D429U, 0x810003D6U},
    {0x1D42AU, 0x810009B1U},
    {0x1D42BU, 0x81000236U},
    {0x1D42CU, 0x81000164U},
    {0x1D42DU, 0x810003D7U},
    {0x1D42EU, 0x810003D8U},
    {0x1D42FU, 0x810003DBU},
    {0x1D430U, 0x8100023AU},
    {0x1D431U, 0x8100024AU},
    {0x1D432U, 0x8100023BU},
    {0x1D433U, 0x81000401U},
    {0x1D434U, 0x810003B1U},
    {0x1D435U, 0x810003B3U},
    {0x1D436U, 0x810007E5U},
    {0x1D437U, 0x810003B4U},
    {0x1D438U, 0x810003B5U},
    {0x1D439U, 0x810007FFU},
    {0x1D43AU, 0x810003B7U},
    {0x1D43BU, 0x810003B8U},
    {0x1D43CU, 0x810003B9U},
    {0x1D43DU, 0x810003BAU},
    {0x1D43EU, 0x810003BBU},
    {0x1D43FU, 0x810003BCU},
    {0x1D440U, 0x810003BDU},
    {0x1D441U, 0x810003BEU},
    {0x1D442U, 0x810003BFU},
    {0x1D443U, 0x810003C1U},
    {0x1D444U, 0x810007F4U},
    {0x1D445U, 0x810003C2U},
    {0x1D446U, 0x810009AFU},
    {0x1D447U, 0x810003C3U},
    {0x1D448U, 0x810003C4U},
    {0x1D449U, 0x81000841U},
    {0x1D44AU, 0x810003C5U},
    {0x1D44BU, 0x8100084DU},
    {0x1D44CU, 0x810009B0U},
    {0x1D44DU, 0x810007FCU},
    {0x1D44EU, 0x81000003U},
    {0x1D44FU, 0x810003C9U},
    {0x1D450U, 0x810003E3U},
    {0x1D451U, 0x810003CAU},
    {0x1D452U, 0x810003CBU},
    {0x1D453U, 0x810003E6U},
    {0x1D454U, 0x810003CFU},
    {0x1D456U, 0x810003E0U},
    {0x1D457U, 0x81000235U},
    {0x1D458U, 0x810003D0U},
    {0x1D459U, 0x81000249U},
    {0x1D45AU, 0x810003D1U},
    {0x1D45BU, 0x810007DCU},
    {0x1D45CU, 0x8100000EU},
    {0x1D45DU, 0x810003D6U},
    {0x1D45EU, 0x810009B1U},
    {0x1D45FU, 0x81000236U},
    {0x1D460U, 0x81000164U},
    {0x1D461U, 0x810003D7U},
    {0x1D462U, 0x810003D8U},
    {0x1D463U, 0x810003DBU},
    {0x1D464U, 0x8100023AU},
    {0x1D465U, 0x8100024AU},
    {0x1D466U, 0x8100023BU},
    {0x1D467U, 0x81000401U},
    {0x1D468U, 0x810003B1U},
    {0x1D469U, 0x810003B3U},
    {0x1D46AU, 0x810007E5U},
    {0x1D46BU, 0x810003B4U},
    {0x1D46CU, 0x810003B5U},
    {0x1D46DU, 0x810007FFU},
    {0x1D46EU, 0x810003B7U},
    {0x1D46FU, 0x810003B8U},
    {0x1D470U, 0x810003B9U},
    {0x1D471U, 0x810003BAU},
    {0x1D472U, 0x810003BBU},
    {0x1D473U, 0x810003BCU},
    {0x1D474U, 0x810003BDU},
    {0x1D475U, 0x810003BEU},
    {0x1D476U, 0x810003BFU},
    {0x1D477U, 0x810003C1U},
    {0x1D478U, 0x810007F4U},
    {0x1D479U, 0x810003C2U},
    {0x1D47AU, 0x810009AFU},
    {0x1D47BU, 0x810003C3U},
    {0x1D47CU, 0x810003C4U},
    {0x1D47DU, 0x81000841U},
    {0x1D47EU, 0x810003C5U},
    {0x1D47FU, 0x8100084DU},
    {0x1D480U, 0x810009B0U},
    {0x1D481U, 0x810007FCU},
    {0x1D482U, 0x81000003U},
    {0x1D483U, 0x810003C9U},
    {0x1D484U, 0x810003E3U},
    {0x1D485U, 0x810003CAU},
    {0x1D486U, 0x810003CBU},
    {0x1D487U, 0x810003E6U},
    {0x1D488U, 0x810003CFU},
    {0x1D489U, 0x81000233U},
    {0x1D48AU, 0x810003E0U},
    {0x1D48BU, 0x81000235U},
    {0x1D48CU, 0x810003D0U},
    {0x1D48DU, 0x81000249U},
    {0x1D48EU, 0x810003D1U},
    {0x1D48FU, 0x810007DCU},
    {0x1D490U, 0x8100000EU},
    {0x1D491U, 0x810003D6U},
    {0x1D492U, 0x810009B1U},
    {0x1D493U, 0x81000236U},
    {0x1D494U, 0x81000164U},
    {0x1D495U, 0x810003D7U},
    {0x1D496U, 0x810003D8U},
    {0x1D497U, 0x810003DBU},
    {0x1D498U, 0x8100023AU},
    {0x1D499U, 0x8100024AU},
    {0x1D49AU, 0x8100023BU},
    {0x1D49BU, 0x81000401U},
    {0x1D49CU, 0x810003B1U},
    {0x1D49EU, 0x810007E5U},
    {0x1D49FU, 0x810003B4U},
    {0x1D4A2U, 0x810003B7U},
    {0x1D4A5U, 0x810003BAU},
    {0x1D4A6U, 0x810003BBU},
    {0x1D4A9U, 0x810003BEU},
    {0x1D4AAU, 0x810003BFU},
    {0x1D4ABU, 0x810003C1U},
    {0x1D4ACU, 0x810007F4U},
    {0x1D4AEU, 0x810009AFU},
    {0x1D4AFU, 0x810003C3U},
    {0x1D4B0U, 0x810003C4U},
    {0x1D4B1U, 0x81000841U},
    {0x1D4B2U, 0x810003C5U},
    {0x1D4B3U, 0x8100084DU},
    {0x1D4B4U, 0x810009B0U},
    {0x1D4B5U, 0x810007FCU},
    {0x1D4B6U, 0x81000003U},
    {0x1D4B7U, 0x810003C9U},
    {0x1D4B8U, 0x810003E3U},
    {0x1D4B9U, 0x810003CAU},
    {0x1D4BBU, 0x810003E6U},
    {0x1D4BDU, 0x81000233U},
    {0x1D4BEU, 0x810003E0U},
    {0x1D4BFU, 0x81000235U},
    {0x1D4C0U, 0x810003D0U},
    {0x1D4C1U, 0x81000249U},
    {0x1D4C2U, 0x810003D1U},
    {0x1D4C3U, 0x810007DCU},
    {0x1D4C5U, 0x810003D6U},
    {0x1D4C6U, 0x810009B1U},
    {0x1D4C7U, 0x81000236U},
    {0x1D4C8U, 0x81000164U},
    {0x1D4C9U, 0x810003D7U},
    {0x1D4CAU, 0x810003D8U},
    {0x1D4CBU, 0x810003DBU},
    {0x1D4CCU, 0x8100023AU},
    {0x1D4CDU, 0x8100024AU},
    {0x1D4CEU, 0x8100023BU},
    {0x1D4CFU, 0x81000401U},
    {0x1D4D0U, 0x810003B1U},
    {0x1D4D1U, 0x810003B3U},
    {0x1D4D2U, 0x810007E5U},
    {0x1D4D3U, 0x810003B4U},
    {0x1D4D4U, 0x810003B5U},
    {0x1D4D5U, 0x810007FFU},
    {0x1D4D6U, 0x810003B7U},
    {0x1D4D7U, 0x810003B8U},
    {0x1D4D8U, 0x810003B9U},
    {0x1D4D9U, 0x810003BAU},
    {0x1D4DAU, 0x810003BBU},
    {0x1D4DBU, 0x810003BCU},
    {0x1D4DCU, 0x810003BDU},
    {0x1D4DDU, 0x810003BEU},
    {0x1D4DEU, 0x810003BFU},
    {0x1D4DFU, 0x810003C1U},
    {0x1D4E0U, 0x810007F4U},
    {0x1D4E1U, 0x810003C2U},
    {0x1D4E2U, 0x810009AFU},
    {0x1D4E3U, 0x810003C3U},
    {0x1D4E4U, 0x810003C4U},
    {0x1D4E5U, 0x81000841U},
    {0x1D4E6U, 0x810003C5U},
    {0x1D4E7U, 0x8100084DU},
    {0x1D4E8U, 0x810009B0U},
    {0x1D4E9U, 0x810007FCU},
    {0x1D4EAU, 0x81000003U},
    {0x1D4EBU, 0x810003C9U},
    {0x1D4ECU, 0x810003E3U},
    {0x1D4EDU, 0x810003CAU},
    {0x1D4EEU, 0x810003CBU},
    {0x1D4EFU, 0x810003E6U},
    {0x1D4F0U, 0x810003CFU},
    {0x1D4F1U, 0x81000233U},
    {0x1D4F2U, 0x810003E0U},
    {0x1D4F3U, 0x81000235U},
    {0x1D4F4U, 0x810003D0U},
    {0x1D4F5U, 0x81000249U},
    {0x1D4F6U, 0x810003D1U},
    {0x1D4F7U, 0x810007DCU},
    {0x1D4F8U, 0x8100000EU},
    {0x1D4F9U, 0x810003D6U},
    {0x1D4FAU, 0x810009B1U},
    {0x1D4FBU, 0x81000236U},
    {0x1D4FCU, 0x81000164U},
    {0x1D4FDU, 0x810003D7U},
    {0x1D4FEU, 0x810003D8U},
    {0x1D4FFU, 0x810003DBU},
    {0x1D500U, 0x8100023AU},
    {0x1D501U, 0x8100024AU},
    {0x1D502U, 0x8100023BU},
    {0x1D503U, 0x81000401U},
    {0x1D504U, 0x810003B1U},
    {0x1D505U, 0x810003B3U},
    {0x1D507U, 0x810003B4U},
    {0x1D508U, 0x810003B5U},
    {0x1D509U, 0x810007FFU},
    {0x1D50AU, 0x810003B7U},
    {0x1D50DU, 0x810003BAU},
    {0x1D50EU, 0x810003BBU},
    {0x1D50FU, 0x810003BCU},
    {0x1D510U, 0x810003BDU},
    {0x1D511U, 0x810003BEU},
    {0x1D512U, 0x810003BFU},
    {0x1D513U, 0x810003C1U},
    {0x1D514U, 0x810007F4U},
    {0x1D516U, 0x810009AFU},
    {0x1D517U, 0x810003C3U},
    {0x1D518U, 0x810003C4U},
    {0x1D519U, 0x81000841U},
    {0x1D51AU, 0x810003C5U},
    {0x1D51BU, 0x8100084DU},
    {0x1D51CU, 0x810009B0U},
    {0x1D51EU, 0x81000003U},
    {0x1D51FU, 0x810003C9U},
    {0x1D520U, 0x810003E3U},
    {0x1D521U, 0x810003CAU},
    {0x1D522U, 0x810003CBU},
    {0x1D523U, 0x810003E6U},
    {0x1D524U, 0x810003CFU},
    {0x1D525U, 0x81000233U},
    {0x1D526U, 0x810003E0U},
    {0x1D527U, 0x81000235U},
    {0x1D528U, 0x810003D0U},
    {0x1D529U, 0x81000249U},
    {0x1D52AU, 0x810003D1U},
    {0x1D52BU, 0x810007DCU},
    {0x1D52CU, 0x8100000EU},
    {0x1D52DU, 0x810003D6U},
    {0x1D52EU, 0x810009B1U},
    {0x1D52FU, 0x81000236U},
    {0x1D530U, 0x81000164U},
    {0x1D531U, 0x810003D7U},
    {0x1D532U, 0x810003D8U},
    {0x1D533U, 0x810003DBU},
    {0x1D534U, 0x8100023AU},
    {0x1D535U, 0x8100024AU},
    {0x1D536U, 0x8100023BU},
    {0x1D537U, 0x81000401U},
    {0x1D538U, 0x810003B1U},
    {0x1D539U, 0x810003B3U},
    {0x1D53BU, 0x810003B4U},
    {0x1D53CU, 0x810003B5U},
    {0x1D53DU, 0x810007FFU},
    {0x1D53EU, 0x810003B7U},
    {0x1D540U, 0x810003B9U},
    {0x1D541U, 0x810003BAU},
    {0x1D542U, 0x810003BBU},
    {0x1D543U, 0x810003BCU},
    {0x1D544U, 0x810003BDU},
    {0x1D546U, 0x810003BFU},
    {0x1D54AU, 0x810009AFU},
    {0x1D54BU, 0x810003C3U},
    {0x1D54CU, 0x810003C4U},
    {0x1D54DU, 0x81000841U},
    {0x1D54EU, 0x810003C5U},
    {0x1D54FU, 0x8100084DU},
    {0x1D550U, 0x810009B0U},
    {0x1D552U, 0x81000003U},
    {0x1D553U, 0x810003C9U},
    {0x1D554U, 0x810003E3U},
    {0x1D555U, 0x810003CAU},
    {0x1D556U, 0x810003CBU},
    {0x1D557U, 0x810003E6U},
    {0x1D558U, 0x810003CFU},
    {0x1D559U, 0x81000233U},
    {0x1D55AU, 0x810003E0U},
    {0x1D55BU, 0x81000235U},
    {0x1D55CU, 0x810003D0U},
    {0x1D55DU, 0x81000249U},
    {0x1D55EU, 0x810003D1U},
    {0x1D55FU, 0x810007DCU},
    {0x1D560U, 0x8100000EU},
    {0x1D561U, 0x810003D6U},
    {0x1D562U, 0x810009B1U},
    {0x1D563U, 0x81000236U},
    {0x1D564U, 0x81000164U},
    {0x1D565U, 0x810003D7U},
    {0x1D566U, 0x810003D8U},
    {0x1D567U, 0x810003DBU},
    {0x1D568U, 0x8100023AU},
    {0x1D569U, 0x8100024AU},
    {0x1D56AU, 0x8100023BU},
    {0x1D56BU, 0x81000401U},
    {0x1D56CU, 0x810003B1U},
    {0x1D56DU, 0x810003B3U},
    {0x1D56EU, 0x810007E5U},
    {0x1D56FU, 0x810003B4U},
    {0x1D570U, 0x810003B5U},
    {0x1D571U, 0x810007FFU},
    {0x1D572U, 0x810003B7U},
    {0x1D573U, 0x810003B8U},
    {0x1D574U, 0x810003B9U},
    {0x1D575U, 0x810003BAU},
    {0x1D576U, 0x810003BBU},
    {0x1D577U, 0x810003BCU},
    {0x1D578U, 0x810003BDU},
    {0x1D579U, 0x810003BEU},
    {0x1D57AU, 0x810003BFU},
    {0x1D57BU, 0x810003C1U},
    {0x1D57CU, 0x810007F4U},
    {0x1D57DU, 0x810003C2U},
    {0x1D57EU, 0x810009AFU},
    {0x1D57FU, 0x810003C3U},
    {0x1D580U, 0x810003C4U},
    {0x1D581U, 0x81000841U},
    {0x1D582U, 0x810003C5U},
    {0x1D583U, 0x8100084DU},
    {0x1D584U, 0x810009B0U},
    {0x1D585U, 0x810007FCU},
    {0x1D586U, 0x81000003U},
    {0x1D587U, 0x810003C9U},
    {0x1D588U, 0x810003E3U},
    {0x1D589U, 0x810003CAU},
    {0x1D58AU, 0x810003CBU},
    {0x1D58BU, 0x810003E6U},
    {0x1D58CU, 0x810003CFU},
    {0x1D58DU, 0x81000233U},
    {0x1D58EU, 0x810003E0U},
    {0x1D58FU, 0x81000235U},
    {0x1D590U, 0x810003D0U},
    {0x1D591U, 0x81000249U},
    {0x1D592U, 0x810003D1U},
    {0x1D593U, 0x810007DCU},
    {0x1D594U, 0x8100000EU},
    {0x1D595U, 0x810003D6U},
    {0x1D596U, 0x810009B1U},
    {0x1D597U, 0x81000236U},
    {0x1D598U, 0x81000164U},
    {0x1D599U, 0x810003D7U},
    {0x1D59AU, 0x810003D8U},
    {0x1D59BU, 0x810003DBU},
    {0x1D59CU, 0x8100023AU},
    {0x1D59DU, 0x8100024AU},
    {0x1D59EU, 0x8100023BU},
    {0x1D59FU, 0x81000401U},
    {0x1D5A0U, 0x810003B1U},
    {0x1D5A1U, 0x810003B3U},
    {0x1D5A2U, 0x810007E5U},
    {0x1D5A3U, 0x810003B4U},
    {0x1D5A4U, 0x810003B5U},
    {0x1D5A5U, 0x810007FFU},
    {0x1D5A6U, 0x810003B7U},
    {0x1D5A7U, 0x810003B8U},
    {0x1D5A8U, 0x810003B9U},
    {0x1D5A9U, 0x810003BAU},
    {0x1D5AAU, 0x810003BBU},
    {0x1D5ABU, 0x810003BCU},
    {0x1D5ACU, 0x810003BDU},
    {0x1D5ADU, 0x810003BEU},
    {0x1D5AEU, 0x810003BFU},
    {0x1D5AFU, 0x810003C1U},
    {0x1D5B0U, 0x810007F4U},
    {0x1D5B1U, 0x810003C2U},
    {0x1D5B2U, 0x810009AFU},
    {0x1D5B3U, 0x810003C3U},
    {0x1D5B4U, 0x810003C4U},
    {0x1D5B5U, 0x81000841U},
    {0x1D5B6U, 0x810003C5U},
    {0x1D5B7U, 0x8100084DU},
    {0x1D5B8U, 0x810009B0U},
    {0x1D5B9U, 0x810007FCU},
    {0x1D5BAU, 0x81000003U},
    {0x1D5BBU, 0x810003C9U},
    {0x1D5BCU, 0x810003E3U},
    {0x1D5BDU, 0x810003CAU},
    {0x1D5BEU, 0x810003CBU},
    {0x1D5BFU, 0x810003E6U},
    {0x1D5C0U, 0x810003CFU},
    {0x1D5C1U, 0x81000233U},
    {0x1D5C2U, 0x810003E0U},
    {0x1D5C3U, 0x81000235U},
    {0x1D5C4U, 0x810003D0U},
    {0x1D5C5U, 0x81000249U},
    {0x1D5C6U, 0x810003D1U},
    {0x1D5C7U, 0x810007DCU},
    {0x1D5C8U, 0x8100000EU},
    {0x1D5C9U, 0x810003D6U},
    {0x1D5CAU, 0x810009B1U},
    {0x1D5CBU, 0x81000236U},
    {0x1D5CCU, 0x81000164U},
    {0x1D5CDU, 0x810003D7U},
    {0x1D5CEU, 0x810003D8U},
    {0x1D5CFU, 0x810003DBU},
    {0x1D5D0U, 0x8100023AU},
    {0x1D5D1U, 0x8100024AU},
    {0x1D5D2U, 0x8100023BU},
    {0x1D5D3U, 0x81000401U},
    {0x1D5D4U, 0x810003B1U},
    {0x1D5D5U, 0x810003B3U},
    {0x1D5D6U, 0x810007E5U},
    {0x1D5D7U, 0x810003B4U},
    {0x1D5D8U, 0x810003B5U},
    {0x1D5D9U, 0x810007FFU},
    {0x1D5DAU, 0x810003B7U},
    {0x1D5DBU, 0x810003B8U},
    {0x1D5DCU, 0x810003B9U},
    {0x1D5DDU, 0x810003BAU},
    {0x1D5DEU, 0x810003BBU},
    {0x1D5DFU, 0x810003BCU},
    {0x1D5E0U, 0x810003BDU},
    {0x1D5E1U, 0x810003BEU},
    {0x1D5E2U, 0x810003BFU},
    {0x1D5E3U, 0x810003C1U},
    {0x1D5E4U, 0x810007F4U},
    {0x1D5E5U, 0x810003C2U},
    {0x1D5E6U, 0x810009AFU},
    {0x1D5E7U, 0x810003C3U},
    {0x1D5E8U, 0x810003C4U},
    {0x1D5E9U, 0x81000841U},
    {0x1D5EAU, 0x810003C5U},
    {0x1D5EBU, 0x8100084DU},
    {0x1D5ECU, 0x810009B0U},
    {0x1D5EDU, 0x810007FCU},
    {0x1D5EEU, 0x81000003U},
    {0x1D5EFU, 0x810003C9U},
    {0x1D5F0U, 0x810003E3U},
    {0x1D5F1U, 0x810003CAU},
    {0x1D5F2U, 0x810003CBU},
    {0x1D5F3U, 0x810003E6U},
    {0x1D5F4U, 0x810003CFU},
    {0x1D5F5U, 0x81000233U},
    {0x1D5F6U, 0x810003E0U},
    {0x1D5F7U, 0x81000235U},
    {0x1D5F8U, 0x810003D0U},
    {0x1D5F9U, 0x81000249U},
    {0x1D5FAU, 0x810003D1U},
    {0x1D5FBU, 0x810007DCU},
    {0x1D5FCU, 0x8100000EU},
    {0x1D5FDU, 0x810003D6U},
    {0x1D5FEU, 0x810009B1U},
    {0x1D5FFU, 0x81000236U},
    {0x1D600U, 0x81000164U},
    {0x1D601U, 0x810003D7U},
    {0x1D602U, 0x810003D8U},
    {0x1D603U, 0x810003DBU},
    {0x1D604U, 0x8100023AU},
    {0x1D605U, 0x8100024AU},
    {0x1D606U, 0x8100023BU},
    {0x1D607U, 0x81000401U},
    {0x1D608U, 0x810003B1U},
    {0x1D609U, 0x810003B3U},
    {0x1D60AU, 0x810007E5U},
    {0x1D60BU, 0x810003B4U},
    {0x1D60CU, 0x810003B5U},
    {0x1D60DU, 0x810007FFU},
    {0x1D60EU, 0x810003B7U},
    {0x1D60FU, 0x810003B8U},
    {0x1D610U, 0x810003B9U},
    {0x1D611U, 0x810003BAU},
    {0x1D612U, 0x810003BBU},
    {0x1D613U, 0x810003BCU},
    {0x1D614U, 0x810003BDU},
    {0x1D615U, 0x810003BEU},
    {0x1D616U, 0x810003BFU},
    {0x1D617U, 0x810003C1U},
    {0x1D618U, 0x810007F4U},
    {0x1D619U, 0x810003C2U},
    {0x1D61AU, 0x810009AFU},
    {0x1D61BU, 0x810003C3U},
    {0x1D61CU, 0x810003C4U},
    {0x1D61DU, 0x81000841U},
    {0x1D61EU, 0x810003C5U},
    {0x1D61FU, 0x8100084DU},
    {0x1D620U, 0x810009B0U},
    {0x1D621U, 0x810007FCU},
    {0x1D622U, 0x81000003U},
    {0x1D623U, 0x810003C9U},
    {0x1D624U, 0x810003E3U},
    {0x1D625U, 0x810003CAU},
    {0x1D626U, 0x810003CBU},
    {0x1D627U, 0x810003E6U},
    {0x1D628U, 0x810003CFU},
    {0x1D629U, 0x81000233U},
    {0x1D62AU, 0x810003E0U},
    {0x1D62BU, 0x81000235U},
    {0x1D62CU, 0x810003D0U},
    {0x1D62DU, 0x81000249U},
    {0x1D62EU, 0x810003D1U},
    {0x1D62FU, 0x810007DCU},
    {0x1D630U, 0x8100000EU},
    {0x1D631U, 0x810003D6U},
    {0x1D632U, 0x810009B1U},
    {0x1D633U, 0x81000236U},
    {0x1D634U, 0x81000164U},
    {0x1D635U, 0x810003D7U},
    {0x1D636U, 0x810003D8U},
    {0x1D637U, 0x810003DBU},
    {0x1D638U, 0x8100023AU},
    {0x1D639U, 0x8100024AU},
    {0x1D63AU, 0x8100023BU},
    {0x1D63BU, 0x81000401U},
    {0x1D63CU, 0x810003B1U},
    {0x1D63DU, 0x810003B3U},
    {0x1D63EU, 0x810007E5U},
    {0x1D63FU, 0x810003B4U},
    {0x1D640U, 0x810003B5U},
    {0x1D641U, 0x810007FFU},
    {0x1D642U, 0x810003B7U},
    {0x1D643U, 0x810003B8U},
    {0x1D644U, 0x810003B9U},
    {0x1D645U, 0x810003BAU},
    {0x1D646U, 0x810003BBU},
    {0x1D647U, 0x810003BCU},
    {0x1D648U, 0x810003BDU},
    {0x1D649U, 0x810003BEU},
    {0x1D64AU, 0x810003BFU},
    {0x1D64BU, 0x810003C1U},
    {0x1D64CU, 0x810007F4U},
    {0x1D64DU, 0x810003C2U},
    {0x1D64EU, 0x810009AFU},
    {0x1D64FU, 0x810003C3U},
    {0x1D650U, 0x810003C4U},
    {0x1D651U, 0x81000841U},
    {0x1D652U, 0x810003C5U},
    {0x1D653U, 0x8100084DU},
    {0x1D654U, 0x810009B0U},
    {0x1D655U, 0x810007FCU},
    {0x1D656U, 0x81000003U},
    {0x1D657U, 0x810003C9U},
    {0x1D658U, 0x810003E3U},
    {0x1D659U, 0x810003CAU},
    {0x1D65AU, 0x810003CBU},
    {0x1D65BU, 0x810003E6U},
    {0x1D65CU, 0x810003CFU},
    {0x1D65DU, 0x81000233U},
    {0x1D65EU, 0x810003E0U},
    {0x1D65FU, 0x81000235U},
    {0x1D660U, 0x810003D0U},
    {0x1D661U, 0x81000249U},
    {0x1D662U, 0x810003D1U},
    {0x1D663U, 0x810007DCU},
    {0x1D664U, 0x8100000EU},
    {0x1D665U, 0x810003D6U},
    {0x1D666U, 0x810009B1U},
    {0x1D667U, 0x81000236U},
    {0x1D668U, 0x81000164U},
    {0x1D669U, 0x810003D7U},
    {0x1D66AU, 0x810003D8U},
    {0x1D66BU, 0x810003DBU},
    {0x1D66CU, 0x8100023AU},
    {0x1D66DU, 0x8100024AU},
    {0x1D66EU, 0x8100023BU},
    {0x1D66FU, 0x81000401U},
    {0x1D670U, 0x810003B1U},
    {0x1D671U, 0x810003B3U},
    {0x1D672U, 0x810007E5U},
    {0x1D673U, 0x810003B4U},
    {0x1D674U, 0x810003B5U},
    {0x1D675U, 0x810007FFU},
    {0x1D676U, 0x810003B7U},
    {0x1D677U, 0x810003B8U},
    {0x1D678U, 0x810003B9U},
    {0x1D679U, 0x810003BAU},
    {0x1D67AU, 0x810003BBU},
    {0x1D67BU, 0x810003BCU},
    {0x1D67CU, 0x810003BDU},
    {0x1D67DU, 0x810003BEU},
    {0x1D67EU, 0x810003BFU},
    {0x1D67FU, 0x810003C1U},
    {0x1D680U, 0x810007F4U},
    {0x1D681U, 0x810003C2U},
    {0x1D682U, 0x810009AFU},
    {0x1D683U, 0x810003C3U},
    {0x1D684U, 0x810003C4U},
    {0x1D685U, 0x81000841U},
    {0x1D686U, 0x810003C5U},
    {0x1D687U, 0x8100084DU},
    {0x1D688U, 0x810009B0U},
    {0x1D689U, 0x810007FCU},
    {0x1D68AU, 0x81000003U},
    {0x1D68BU, 0x810003C9U},
    {0x1D68CU, 0x810003E3U},
    {0x1D68DU, 0x810003CAU},
    {0x1D68EU, 0x810003CBU},
    {0x1D68FU, 0x810003E6U},
    {0x1D690U, 0x810003CFU},
    {0x1D691U, 0x81000233U},
    {0x1D692U, 0x810003E0U},
    {0x1D693U, 0x81000235U},
    {0x1D694U, 0x810003D0U},
    {0x1D695U, 0x81000249U},
    {0x1D696U, 0x810003D1U},
    {0x1D697U, 0x810007DCU},
    {0x1D698U, 0x8100000EU},
    {0x1D699U, 0x810003D6U},
    {0x1D69AU, 0x810009B1U},
    {0x1D69BU, 0x81000236U},
    {0x1D69CU, 0x81000164U},
    {0x1D69DU, 0x810003D7U},
    {0x1D69EU, 0x810003D8U},
    {0x1D69FU, 0x810003DBU},
    {0x1D6A0U, 0x8100023AU},
    {0x1D6A1U, 0x8100024AU},
    {0x1D6A2U, 0x8100023BU},
    {0x1D6A3U, 0x81000401U},
    {0x1D6A4U, 0x81001651U},
    {0x1D6A5U, 0x81001652U},
    {0x1D6A8U, 0x81001653U},
    {0x1D6A9U, 0x81001654U},
    {0x1D6AAU, 0x81000807U},
    {0x1D6ABU, 0x81001655U},
    {0x1D6ACU, 0x81001656U},
    {0x1D6ADU, 0x81001657U},
    {0x1D6AEU, 0x81001658U},
    {0x1D6AFU, 0x8100028CU},
    {0x1D6B0U, 0x81001659U},
    {0x1D6B1U, 0x8100165AU},
    {0x1D6B2U, 0x8100165BU},
    {0x1D6B3U, 0x8100165CU},
    {0x1D6B4U, 0x8100165DU},
    {0x1D6B5U, 0x8100165EU},
    {0x1D6B6U, 0x8100165FU},
    {0x1D6B7U, 0x81000808U},
    {0x1D6B8U, 0x81001660U},
    {0x1D6B9U, 0x81001661U},
    {0x1D6BAU, 0x8100028EU},
    {0x1D6BBU, 0x81001662U},
    {0x1D6BCU, 0x81000282U},
    {0x1D6BDU, 0x81001663U},
    {0x1D6BEU, 0x81001664U},
    {0x1D6BFU, 0x81001665U},
    {0x1D6C0U, 0x810007FDU},
    {0x1D6C1U, 0x81001666U},
    {0x1D6C2U, 0x81001667U},
    {0x1D6C3U, 0x81000280U},
    {0x1D6C4U, 0x810003DDU},
    {0x1D6C5U, 0x810003DEU},
    {0x1D6C6U, 0x8100028DU},
    {0x1D6C7U, 0x81001668U},
    {0x1D6C8U, 0x81001669U},
    {0x1D6C9U, 0x81000281U},
    {0x1D6CAU, 0x81000747U},
    {0x1D6CBU, 0x81000289U},
    {0x1D6CCU, 0x8100166AU},
    {0x1D6CDU, 0x8100000AU},
    {0x1D6CEU, 0x8100166BU},
    {0x1D6CFU, 0x8100166CU},
    {0x1D6D0U, 0x8100166DU},
    {0x1D6D1U, 0x81000288U},
    {0x1D6D2U, 0x8100028AU},
    {0x1D6D3U, 0x8100028BU},
    {0x1D6D4U, 0x8100166EU},
    {0x1D6D5U, 0x8100166FU},
    {0x1D6D6U, 0x81001670U},
    {0x1D6D7U, 0x81000287U},
    {0x1D6D8U, 0x810003DFU},
    {0x1D6D9U, 0x81001671U},
    {0x1D6DAU, 0x81001672U},
    {0x1D6DBU, 0x81001673U},
    {0x1D6DCU, 0x81001674U},
    {0x1D6DDU, 0x81001675U},
    {0x1D6DEU, 0x81001676U},
    {0x1D6DFU, 0x81001677U},
    {0x1D6E0U, 0x81001678U},
    {0x1D6E1U, 0x81001679U},
    {0x1D6E2U, 0x81001653U},
    {0x1D6E3U, 0x81001654U},
    {0x1D6E4U, 0x81000807U},
    {0x1D6E5U, 0x81001655U},
    {0x1D6E6U, 0x81001656U},
    {0x1D6E7U, 0x81001657U},
    {0x1D6E8U, 0x81001658U},
    {0x1D6E9U, 0x8100028CU},
    {0x1D6EAU, 0x81001659U},
    {0x1D6EBU, 0x8100165AU},
    {0x1D6ECU, 0x8100165BU},
    {0x1D6EDU, 0x8100165CU},
    {0x1D6EEU, 0x8100165DU},
    {0x1D6EFU, 0x8100165EU},
    {0x1D6F0U, 0x8100165FU},
    {0x1D6F1U, 0x81000808U},
    {0x1D6F2U, 0x81001660U},
    {0x1D6F3U, 0x81001661U},
    {0x1D6F4U, 0x8100028EU},
    {0x1D6F5U, 0x81001662U},
    {0x1D6F6U, 0x81000282U},
    {0x1D6F7U, 0x81001663U},
    {0x1D6F8U, 0x81001664U},
    {0x1D6F9U, 0x81001665U},
    {0x1D6FAU, 0x810007FDU},
    {0x1D6FBU, 0x81001666U},
    {0x1D6FCU, 0x81001667U},
    {0x1D6FDU, 0x81000280U},
    {0x1D6FEU, 0x810003DDU},
    {0x1D6FFU, 0x810003DEU},
    {0x1D700U, 0x8100028DU},
    {0x1D701U, 0x81001668U},
    {0x1D702U, 0x81001669U},
    {0x1D703U, 0x81000281U},
    {0x1D704U, 0x81000747U},
    {0x1D705U, 0x81000289U},
    {0x1D706U, 0x8100166AU},
    {0x1D707U, 0x8100000AU},
    {0x1D708U, 0x8100166BU},
    {0x1D709U, 0x8100166CU},
    {0x1D70AU, 0x8100166DU},
    {0x1D70BU, 0x81000288U},
    {0x1D70CU, 0x8100028AU},
    {0x1D70DU, 0x8100028BU},
    {0x1D70EU, 0x8100166EU},
    {0x1D70FU, 0x8100166FU},
    {0x1D710U, 0x81001670U},
    {0x1D711U, 0x81000287U},
    {0x1D712U, 0x810003DFU},
    {0x1D713U, 0x81001671U},
    {0x1D714U, 0x81001672U},
    {0x1D715U, 0x81001673U},
    {0x1D716U, 0x81001674U},
    {0x1D717U, 0x81001675U},
    {0x1D718U, 0x81001676U},
    {0x1D719U, 0x81001677U},
    {0x1D71AU, 0x81001678U},
    {0x1D71BU, 0x81001679U},
    {0x1D71CU, 0x81001653U},
    {0x1D71DU, 0x81001654U},
    {0x1D71EU, 0x81000807U},
    {0x1D71FU, 0x81001655U},
    {0x1D720U, 0x81001656U},
    {0x1D721U, 0x81001657U},
    {0x1D722U, 0x81001658U},
    {0x1D723U, 0x8100028CU},
    {0x1D724U, 0x81001659U},
    {0x1D725U, 0x8100165AU},
    {0x1D726U, 0x8100165BU},
    {0x1D727U, 0x8100165CU},
    {0x1D728U, 0x8100165DU},
    {0x1D729U, 0x8100165EU},
    {0x1D72AU, 0x8100165FU},
    {0x1D72BU, 0x81000808U},
    {0x1D72CU, 0x81001660U},
    {0x1D72DU, 0x81001661U},
    {0x1D72EU, 0x8100028EU},
    {0x1D72FU, 0x81001662U},
    {0x1D730U, 0x81000282U},
    {0x1D731U, 0x81001663U},
    {0x1D732U, 0x81001664U},
    {0x1D733U, 0x81001665U},
    {0x1D734U, 0x810007FDU},
    {0x1D735U, 0x81001666U},
    {0x1D736U, 0x81001667U},
    {0x1D737U, 0x81000280U},
    {0x1D738U, 0x810003DDU},
    {0x1D739U, 0x810003DEU},
    {0x1D73AU, 0x8100028DU},
    {0x1D73BU, 0x81001668U},
    {0x1D73CU, 0x81001669U},
    {0x1D73DU, 0x81000281U},
    {0x1D73EU, 0x81000747U},
    {0x1D73FU, 0x81000289U},
    {0x1D740U, 0x8100166AU},
    {0x1D741U, 0x8100000AU},
    {0x1D742U, 0x8100166BU},
    {0x1D743U, 0x8100166CU},
    {0x1D744U, 0x8100166DU},
    {0x1D745U, 0x81000288U},
    {0x1D746U, 0x8100028AU},
    {0x1D747U, 0x8100028BU},
    {0x1D748U, 0x8100166EU},
    {0x1D749U, 0x8100166FU},
    {0x1D74AU, 0x81001670U},
    {0x1D74BU, 0x81000287U},
    {0x1D74CU, 0x810003DFU},
    {0x1D74DU, 0x81001671U},
    {0x1D74EU, 0x81001672U},
    {0x1D74FU, 0x81001673U},
    {0x1D750U, 0x81001674U},
    {0x1D751U, 0x81001675U},
    {0x1D752U, 0x81001676U},
    {0x1D753U, 0x81001677U},
    {0x1D754U, 0x81001678U},
    {0x1D755U, 0x81001679U},
    {0x1D756U, 0x81001653U},
    {0x1D757U, 0x81001654U},
    {0x1D758U, 0x81000807U},
    {0x1D759U, 0x81001655U},
    {0x1D75AU, 0x81001656U},
    {0x1D75BU, 0x81001657U},
    {0x1D75CU, 0x81001658U},
    {0x1D75DU, 0x8100028CU},
    {0x1D75EU, 0x81001659U},
    {0x1D75FU, 0x8100165AU},
    {0x1D760U, 0x8100165BU},
    {0x1D761U, 0x8100165CU},
    {0x1D762U, 0x8100165DU},
    {0x1D763U, 0x8100165EU},
    {0x1D764U, 0x8100165FU},
    {0x1D765U, 0x81000808U},
    {0x1D766U, 0x81001660U},
    {0x1D767U, 0x81001661U},
    {0x1D768U, 0x8100028EU},
    {0x1D769U, 0x81001662U},
    {0x1D76AU, 0x81000282U},
    {0x1D76BU, 0x81001663U},
    {0x1D76CU, 0x81001664U},
    {0x1D76DU, 0x81001665U},
    {0x1D76EU, 0x810007FDU},
    {0x1D76FU, 0x81001666U},
    {0x1D770U, 0x81001667U},
    {0x1D771U, 0x81000280U},
    {0x1D772U, 0x810003DDU},
    {0x1D773U, 0x810003DEU},
    {0x1D774U, 0x8100028DU},
    {0x1D775U, 0x81001668U},
    {0x1D776U, 0x81001669U},
    {0x1D777U, 0x81000281U},
    {0x1D778U, 0x81000747U},
    {0x1D779U, 0x81000289U},
    {0x1D77AU, 0x8100166AU},
    {0x1D77BU, 0x8100000AU},
    {0x1D77CU, 0x8100166BU},
    {0x1D77DU, 0x8100166CU},
    {0x1D77EU, 0x8100166DU},
    {0x1D77FU, 0x81000288U},
    {0x1D780U, 0x8100028AU},
    {0x1D781U, 0x8100028BU},
    {0x1D782U, 0x8100166EU},
    {0x1D783U, 0x8100166FU},
    {0x1D784U, 0x81001670U},
    {0x1D785U, 0x81000287U},
    {0x1D786U, 0x810003DFU},
    {0x1D787U, 0x81001671U},
    {0x1D788U, 0x81001672U},
    {0x1D789U, 0x81001673U},
    {0x1D78AU, 0x81001674U},
    {0x1D78BU, 0x81001675U},
    {0x1D78CU, 0x81001676U},
    {0x1D78DU, 0x81001677U},
    {0x1D78EU, 0x81001678U},
    {0x1D78FU, 0x81001679U},
    {0x1D790U, 0x81001653U},
    {0x1D791U, 0x81001654U},
    {0x1D792U, 0x81000807U},
    {0x1D793U, 0x81001655U},
    {0x1D794U, 0x81001656U},
    {0x1D795U, 0x81001657U},
    {0x1D796U, 0x81001658U},
    {0x1D797U, 0x8100028CU},
    {0x1D798U, 0x81001659U},
    {0x1D799U, 0x8100165AU},
    {0x1D79AU, 0x8100165BU},
    {0x1D79BU, 0x8100165CU},
    {0x1D79CU, 0x8100165DU},
    {0x1D79DU, 0x8100165EU},
    {0x1D79EU, 0x8100165FU},
    {0x1D79FU, 0x81000808U},
    {0x1D7A0U, 0x81001660U},
    {0x1D7A1U, 0x81001661U},
    {0x1D7A2U, 0x8100028EU},
    {0x1D7A3U, 0x81001662U},
    {0x1D7A4U, 0x81000282U},
    {0x1D7A5U, 0x81001663U},
    {0x1D7A6U, 0x81001664U},
    {0x1D7A7U, 0x81001665U},
    {0x1D7A8U, 0x810007FDU},
    {0x1D7A9U, 0x81001666U},
    {0x1D7AAU, 0x81001667U},
    {0x1D7ABU, 0x81000280U},
    {0x1D7ACU, 0x810003DDU},
    {0x1D7ADU, 0x810003DEU},
    {0x1D7AEU, 0x8100028DU},
    {0x1D7AFU, 0x81001668U},
    {0x1D7B0U, 0x81001669U},
    {0x1D7B1U, 0x81000281U},
    {0x1D7B2U, 0x81000747U},
    {0x1D7B3U, 0x81000289U},
    {0x1D7B4U, 0x8100166AU},
    {0x1D7B5U, 0x8100000AU},
    {0x1D7B6U, 0x8100166BU},
    {0x1D7B7U, 0x8100166CU},
    {0x1D7B8U, 0x8100166DU},
    {0x1D7B9U, 0x81000288U},
    {0x1D7BAU, 0x8100028AU},
    {0x1D7BBU, 0x8100028BU},
    {0x1D7BCU, 0x8100166EU},
    {0x1D7BDU, 0x8100166FU},
    {0x1D7BEU, 0x81001670U},
    {0x1D7BFU, 0x81000287U},
    {0x1D7C0U, 0x810003DFU},
    {0x1D7C1U, 0x81001671U},
    {0x1D7C2U, 0x81001672U},
    {0x1D7C3U, 0x81001673U},
    {0x1D7C4U, 0x81001674U},
    {0x1D7C5U, 0x81001675U},
    {0x1D7C6U, 0x81001676U},
    {0x1D7C7U, 0x81001677U},
    {0x1D7C8U, 0x81001678U},
    {0x1D7C9U, 0x81001679U},
    {0x1D7CAU, 0x8100167AU},
    {0x1D7CBU, 0x8100167BU},
    {0x1D7CEU, 0x810007D0U},
    {0x1D7CFU, 0x8100000DU},
    {0x1D7D0U, 0x81000006U},
    {0x1D7D1U, 0x81000007U},
    {0x1D7D2U, 0x810007D1U},
    {0x1D7D3U, 0x810007D2U},
    {0x1D7D4U, 0x810007D3U},
    {0x1D7D5U, 0x810007D4U},
    {0x1D7D6U, 0x810007D5U},
    {0x1D7D7U, 0x810007D6U},
    {0x1D7D8U, 0x810007D0U},
    {0x1D7D9U, 0x8100000DU},
    {0x1D7DAU, 0x81000006U},
    {0x1D7DBU, 0x81000007U},
    {0x1D7DCU, 0x810007D1U},
    {0x1D7DDU, 0x810007D2U},
    {0x1D7DEU, 0x810007D3U},
    {0x1D7DFU, 0x810007D4U},
    {0x1D7E0U, 0x810007D5U},
    {0x1D7E1U, 0x810007D6U},
    {0x1D7E2U, 0x810007D0U},
    {0x1D7E3U, 0x8100000DU},
    {0x1D7E4U, 0x81000006U},
    {0x1D7E5U, 0x81000007U},
    {0x1D7E6U, 0x810007D1U},
    {0x1D7E7U, 0x810007D2U},
    {0x1D7E8U, 0x810007D3U},
    {0x1D7E9U, 0x810007D4U},
    {0x1D7EAU, 0x810007D5U},
    {0x1D7EBU, 0x810007D6U},
    {0x1D7ECU, 0x810007D0U},
    {0x1D7EDU, 0x8100000DU},
    {0x1D7EEU, 0x81000006U},
    {0x1D7EFU, 0x81000007U},
    {0x1D7F0U, 0x810007D1U},
    {0x1D7F1U, 0x810007D2U},
    {0x1D7F2U, 0x810007D3U},
    {0x1D7F3U, 0x810007D4U},
    {0x1D7F4U, 0x810007D5U},
    {0x1D7F5U, 0x810007D6U},
    {0x1D7F6U, 0x810007D0U},
    {0x1D7F7U, 0x8100000DU},
    {0x1D7F8U, 0x81000006U},
    {0x1D7F9U, 0x81000007U},
    {0x1D7FAU, 0x810007D1U},
    {0x1D7FBU, 0x810007D2U},
    {0x1D7FCU, 0x810007D3U},
    {0x1D7FDU, 0x810007D4U},
    {0x1D7FEU, 0x810007D5U},
    {0x1D7FFU, 0x810007D6U},
    {0x1E030U, 0x8100167CU},
    {0x1E031U, 0x8100167DU},
    {0x1E032U, 0x8100167EU},
    {0x1E033U, 0x8100167FU},
    {0x1E034U, 0x81001680U},
    {0x1E035U, 0x81001681U},
    {0x1E036U, 0x81001682U},
    {0x1E037U, 0x81001683U},
    {0x1E038U, 0x81001684U},
    {0x1E039U, 0x81001685U},
    {0x1E03AU, 0x81001686U},
    {0x1E03BU, 0x81001687U},
    {0x1E03CU, 0x81001688U},
    {0x1E03DU, 0x81001689U},
    {0x1E03EU, 0x8100168AU},
    {0x1E03FU, 0x8100168BU},
    {0x1E040U, 0x8100168CU},
    {0x1E041U, 0x8100168DU},
    {0x1E042U, 0x8100168EU},
    {0x1E043U, 0x8100168FU},
    {0x1E044U, 0x81001690U},
    {0x1E045U, 0x81001691U},
    {0x1E046U, 0x81001692U},
    {0x1E047U, 0x81001693U},
    {0x1E048U, 0x81001694U},
    {0x1E049U, 0x81001695U},
    {0x1E04AU, 0x81001696U},
    {0x1E04BU, 0x81001697U},
    {0x1E04CU, 0x81001698U},
    {0x1E04DU, 0x81001699U},
    {0x1E04EU, 0x8100169AU},
    {0x1E04FU, 0x8100169BU},
    {0x1E050U, 0x8100169CU},
    {0x1E051U, 0x8100167CU},
    {0x1E052U, 0x8100167DU},
    {0x1E053U, 0x8100167EU},
    {0x1E054U, 0x8100167FU},
    {0x1E055U, 0x81001680U},
    {0x1E056U, 0x81001681U},
    {0x1E057U, 0x81001682U},
    {0x1E058U, 0x81001683U},
    {0x1E059U, 0x81001684U},
    {0x1E05AU, 0x81001685U},
    {0x1E05BU, 0x81001686U},
    {0x1E05CU, 0x81001688U},
    {0x1E05DU, 0x81001689U},
    {0x1E05EU, 0x8100168BU},
    {0x1E05FU, 0x8100168DU},
    {0x1E060U, 0x8100168EU},
    {0x1E061U, 0x8100168FU},
    {0x1E062U, 0x81001690U},
    {0x1E063U, 0x81001691U},
    {0x1E064U, 0x81001692U},
    {0x1E065U, 0x81001010U},
    {0x1E066U, 0x81001693U},
    {0x1E067U, 0x8100169DU},
    {0x1E068U, 0x81001698U},
    {0x1E069U, 0x8100169EU},
    {0x1E06AU, 0x8100169FU},
    {0x1E06BU, 0x810016A0U},
    {0x1E06CU, 0x810016A1U},
    {0x1E06DU, 0x810016A2U},
    {0x1EE00U, 0x81001542U},
    {0x1EE01U, 0x81001543U},
    {0x1EE02U, 0x81001547U},
    {0x1EE03U, 0x8100154AU},
    {0x1EE05U, 0x8100155DU},
    {0x1EE06U, 0x8100154DU},
    {0x1EE07U, 0x81001548U},
    {0x1EE08U, 0x81001552U},
    {0x1EE09U, 0x8100155EU},
    {0x1EE0AU, 0x81001558U},
    {0x1EE0BU, 0x81001559U},
    {0x1EE0CU, 0x8100155AU},
    {0x1EE0DU, 0x8100155BU},
    {0x1EE0EU, 0x8100154EU},
    {0x1EE0FU, 0x81001554U},
    {0x1EE10U, 0x81001556U},
    {0x1EE11U, 0x81001550U},
    {0x1EE12U, 0x81001557U},
    {0x1EE13U, 0x8100154CU},
    {0x1EE14U, 0x8100154FU},
    {0x1EE15U, 0x81001545U},
    {0x1EE16U, 0x81001546U},
    {0x1EE17U, 0x81001549U},
    {0x1EE18U, 0x8100154BU},
    {0x1EE19U, 0x81001551U},
    {0x1EE1AU, 0x81001553U},
    {0x1EE1BU, 0x81001555U},
    {0x1EE1CU, 0x810016A3U},
    {0x1EE1DU, 0x8100122EU},
    {0x1EE1EU, 0x810016A4U},
    {0x1EE1FU, 0x810016A5U},
    {0x1EE21U, 0x81001543U},
    {0x1EE22U, 0x81001547U},
    {0x1EE24U, 0x8100155CU},
    {0x1EE27U, 0x81001548U},
    {0x1EE29U, 0x8100155EU},
    {0x1EE2AU, 0x81001558U},
    {0x1EE2BU, 0x81001559U},
    {0x1EE2CU, 0x8100155AU},
    {0x1EE2DU, 0x8100155BU},
    {0x1EE2EU, 0x8100154EU},
    {0x1EE2FU, 0x81001554U},
    {0x1EE30U, 0x81001556U},
    {0x1EE31U, 0x81001550U},
    {0x1EE32U, 0x81001557U},
    {0x1EE34U, 0x8100154FU},
    {0x1EE35U, 0x81001545U},
    {0x1EE36U, 0x81001546U},
    {0x1EE37U, 0x81001549U},
    {0x1EE39U, 0x81001551U},
    {0x1EE3BU, 0x81001555U},
    {0x1EE42U, 0x81001547U},
    {0x1EE47U, 0x81001548U},
    {0x1EE49U, 0x8100155EU},
    {0x1EE4BU, 0x81001559U},
    {0x1EE4DU, 0x8100155BU},
    {0x1EE4EU, 0x8100154EU},
    {0x1EE4FU, 0x81001554U},
    {0x1EE51U, 0x81001550U},
    {0x1EE52U, 0x81001557U},
    {0x1EE54U, 0x8100154FU},
    {0x1EE57U, 0x81001549U},
    {0x1EE59U, 0x81001551U},
    {0x1EE5BU, 0x81001555U},
    {0x1EE5DU, 0x8100122EU},
    {0x1EE5FU, 0x810016A5U},
    {0x1EE61U, 0x81001543U},
    {0x1EE62U, 0x81001547U},
    {0x1EE64U, 0x8100155CU},
    {0x1EE67U, 0x81001548U},
    {0x1EE68U, 0x81001552U},
    {0x1EE69U, 0x8100155EU},
    {0x1EE6AU, 0x81001558U},
    {0x1EE6CU, 0x8100155AU},
    {0x1EE6DU, 0x8100155BU},
    {0x1EE6EU, 0x8100154EU},
    {0x1EE6FU, 0x81001554U},
    {0x1EE70U, 0x81001556U},
    {0x1EE71U, 0x81001550U},
    {0x1EE72U, 0x81001557U},
    {0x1EE74U, 0x8100154FU},
    {0x1EE75U, 0x81001545U},
    {0x1EE76U, 0x81001546U},
    {0x1EE77U, 0x81001549U},
    {0x1EE79U, 0x81001551U},
    {0x1EE7AU, 0x81001553U},
    {0x1EE7BU, 0x81001555U},
    {0x1EE7CU, 0x810016A3U},
    {0x1EE7EU, 0x810016A4U},
    {0x1EE80U, 0x81001542U},
    {0x1EE81U, 0x81001543U},
    {0x1EE82U, 0x81001547U},
    {0x1EE83U, 0x8100154AU},
    {0x1EE84U, 0x8100155CU},
    {0x1EE85U, 0x8100155DU},
    {0x1EE86U, 0x8100154DU},
    {0x1EE87U, 0x81001548U},
    {0x1EE88U, 0x81001552U},
    {0x1EE89U, 0x8100155EU},
    {0x1EE8BU, 0x81001559U},
    {0x1EE8CU, 0x8100155AU},
    {0x1EE8DU, 0x8100155BU},
    {0x1EE8EU, 0x8100154EU},
    {0x1EE8FU, 0x81001554U},
    {0x1EE90U, 0x81001556U},
    {0x1EE91U, 0x81001550U},
    {0x1EE92U, 0x81001557U},
    {0x1EE93U, 0x8100154CU},
    {0x1EE94U, 0x8100154FU},
    {0x1EE95U, 0x81001545U},
    {0x1EE96U, 0x81001546U},
    {0x1EE97U, 0x81001549U},
    {0x1EE98U, 0x8100154BU},
    {0x1EE99U, 0x81001551U},
    {0x1EE9AU, 0x81001553U},
    {0x1EE9BU, 0x81001555U},
    {0x1EEA1U, 0x81001543U},
    {0x1EEA2U, 0x81001547U},
    {0x1EEA3U, 0x8100154AU},
    {0x1EEA5U, 0x8100155DU},
    {0x1EEA6U, 0x8100154DU},
    {0x1EEA7U, 0x81001548U},
    {0x1EEA8U, 0x81001552U},
    {0x1EEA9U, 0x8100155EU},
    {0x1EEABU, 0x81001559U},
    {0x1EEACU, 0x8100155AU},
    {0x1EEADU, 0x8100155BU},
    {0x1EEAEU, 0x8100154EU},
    {0x1EEAFU, 0x81001554U},
    {0x1EEB0U, 0x81001556U},
    {0x1EEB1U, 0x81001550U},
    {0x1EEB2U, 0x81001557U},
    {0x1EEB3U, 0x8100154CU},
    {0x1EEB4U, 0x8100154FU},
    {0x1EEB5U, 0x81001545U},
    {0x1EEB6U, 0x81001546U},
    {0x1EEB7U, 0x81001549U},
    {0x1EEB8U, 0x8100154BU},
    {0x1EEB9U, 0x81001551U},
    {0x1EEBAU, 0x81001553U},
    {0x1EEBBU, 0x81001555U},
    {0x1F100U, 0x820016A6U},
    {0x1F101U, 0x820016A8U},
    {0x1F102U, 0x820016AAU},
    {0x1F103U, 0x820016ACU},
    {0x1F104U, 0x820016AEU},
    {0x1F105U, 0x820016B0U},
    {0x1F106U, 0x820016B2U},
    {0x1F107U, 0x820016B4U},
    {0x1F108U, 0x820016B6U},
    {0x1F109U, 0x820016B8U},
    {0x1F10AU, 0x820016BAU},
    {0x1F110U, 0x830016BCU},
    {0x1F111U, 0x830016BFU},
    {0x1F112U, 0x830016C2U},
    {0x1F113U, 0x830016C5U},
    {0x1F114U, 0x830016C8U},
    {0x1F115U, 0x830016CBU},
    {0x1F116U, 0x830016CEU},
    {0x1F117U, 0x830016D1U},
    {0x1F118U, 0x830016D4U},
    {0x1F119U, 0x830016D7U},
    {0x1F11AU, 0x830016DAU},
    {0x1F11BU, 0x830016DDU},
    {0x1F11CU, 0x830016E0U},
    {0x1F11DU, 0x830016E3U},
    {0x1F11EU, 0x830016E6U},
    {0x1F11FU, 0x830016E9U},
    {0x1F120U, 0x830016ECU},
    {0x1F121U, 0x830016EFU},
    {0x1F122U, 0x830016F2U},
    {0x1F123U, 0x830016F5U},
    {0x1F124U, 0x830016F8U},
    {0x1F125U, 0x830016FBU},
    {0x1F126U, 0x830016FEU},
    {0x1F127U, 0x83001701U},
    {0x1F128U, 0x83001704U},
    {0x1F129U, 0x83001707U},
    {0x1F12AU, 0x8300170AU},
    {0x1F12BU, 0x810007E5U},
    {0x1F12CU, 0x810003C2U},
    {0x1F12DU, 0x8200170DU},
    {0x1F12EU, 0x8200170FU},
    {0x1F130U, 0x810003B1U},
    {0x1F131U, 0x810003B3U},
    {0x1F132U, 0x810007E5U},
    {0x1F133U, 0x810003B4U},
    {0x1F134U, 0x810003B5U},
    {0x1F135U, 0x810007FFU},
    {0x1F136U, 0x810003B7U},
    {0x1F137U, 0x810003B8U},
    {0x1F138U, 0x810003B9U},
    {0x1F139U, 0x810003BAU},
    {0x1F13AU, 0x810003BBU},
    {0x1F13BU, 0x810003BCU},
    {0x1F13CU, 0x810003BDU},
    {0x1F13DU, 0x810003BEU},
    {0x1F13EU, 0x810003BFU},
    {0x1F13FU, 0x810003C1U},
    {0x1F140U, 0x810007F4U},
    {0x1F141U, 0x810003C2U},
    {0x1F142U, 0x810009AFU},
    {0x1F143U, 0x810003C3U},
    {0x1F144U, 0x810003C4U},
    {0x1F145U, 0x81000841U},
    {0x1F146U, 0x810003C5U},
    {0x1F147U, 0x8100084DU},
    {0x1F148U, 0x810009B0U},
    {0x1F149U, 0x810007FCU},
    {0x1F14AU, 0x82001711U},
    {0x1F14BU, 0x82000F5EU},
    {0x1F14CU, 0x82001713U},
    {0x1F14DU, 0x82001715U},
    {0x1F14EU, 0x83001717U},
    {0x1F14FU, 0x8200171AU},
    {0x1F16AU, 0x8200171CU},
    {0x1F16BU, 0x8200171EU},
    {0x1F16CU, 0x82001720U},
    {0x1F190U, 0x82001722U},
    {0x1F200U, 0x82001724U},
    {0x1F201U, 0x82001726U},
    {0x1F202U, 0x81000D17U},
    {0x1F210U, 0x81000A02U},
    {0x1F211U, 0x81001728U},
    {0x1F212U, 0x81001729U},
    {0x1F213U, 0x8100172AU},
    {0x1F214U, 0x810009C9U},
    {0x1F215U, 0x8100172BU},
    {0x1F216U, 0x8100172CU},
    {0x1F217U, 0x81000B7EU},
    {0x1F218U, 0x8100172DU},
    {0x1F219U, 0x8100172EU},
    {0x1F21AU, 0x8100172FU},
    {0x1F21BU, 0x810010CCU},
    {0x1F21CU, 0x81001730U},
    {0x1F21DU, 0x81001731U},
    {0x1F21EU, 0x81001732U},
    {0x1F21FU, 0x81001733U},
    {0x1F220U, 0x81001734U},
    {0x1F221U, 0x81001735U},
    {0x1F222U, 0x81000A26U},
    {0x1F223U, 0x81001736U},
    {0x1F224U, 0x81001737U},
    {0x1F225U, 0x81001738U},
    {0x1F226U, 0x81001739U},
    {0x1F227U, 0x8100173AU},
    {0x1F228U, 0x8100173BU},
    {0x1F229U, 0x810009C3U},
    {0x1F22AU, 0x81000B76U},
    {0x1F22BU, 0x8100173CU},
    {0x1F22CU, 0x81000CC0U},
    {0x1F22DU, 0x81000B79U},
    {0x1F22EU, 0x81000CC1U},
    {0x1F22FU, 0x8100173DU},
    {0x1F230U, 0x81000A5EU},
    {0x1F231U, 0x8100173EU},
    {0x1F232U, 0x8100173FU},
    {0x1F233U, 0x81001740U},
    {0x1F234U, 0x81001741U},
    {0x1F235U, 0x81001742U},
    {0x1F236U, 0x81000CAFU},
    {0x1F237U, 0x81000A0CU},
    {0x1F238U, 0x81001743U},
    {0x1F239U, 0x81001744U},
    {0x1F23AU, 0x81001745U},
    {0x1F23BU, 0x81001746U},
    {0x1F240U, 0x83001747U},
    {0x1F241U, 0x8300174AU},
    {0x1F242U, 0x8300174DU},
    {0x1F243U, 0x83001750U},
    {0x1F244U, 0x83001753U},
    {0x1F245U, 0x83001756U},
    {0x1F246U, 0x83001759U},
    {0x1F247U, 0x8300175CU},
    {0x1F248U, 0x8300175FU},
    {0x1F250U, 0x81001762U},
    {0x1F251U, 0x81001763U},
    {0x1FBF0U, 0x810007D0U},
    {0x1FBF1U, 0x8100000DU},
    {0x1FBF2U, 0x81000006U},
    {0x1FBF3U, 0x81000007U},
    {0x1FBF4U, 0x810007D1U},
    {0x1FBF5U, 0x810007D2U},
    {0x1FBF6U, 0x810007D3U},
    {0x1FBF7U, 0x810007D4U},
    {0x1FBF8U, 0x810007D5U},
    {0x1FBF9U, 0x810007D6U},
    {0x2F800U, 0x01001764U},
    {0x2F801U, 0x01001765U},
    {0x2F802U, 0x01001766U},
    {0x2F803U, 0x01001767U},
    {0x2F804U, 0x01001768U},
    {0x2F805U, 0x01001129U},
    {0x2F806U, 0x01001769U},
    {0x2F807U, 0x0100176AU},
    {0x2F808U, 0x0100176BU},
    {0x2F809U, 0x0100176CU},
    {0x2F80AU, 0x0100112AU},
    {0x2F80BU, 0x0100176DU},
    {0x2F80CU, 0x0100176EU},
    {0x2F80DU, 0x0100176FU},
    {0x2F80EU, 0x0100112BU},
    {0x2F80FU, 0x01001770U},
    {0x2F810U, 0x01001771U},
    {0x2F811U, 0x01001772U},
    {0x2F812U, 0x01001773U},
    {0x2F813U, 0x01001774U},
    {0x2F814U, 0x01001775U},
    {0x2F815U, 0x01001732U},
    {0x2F816U, 0x01001776U},
    {0x2F817U, 0x01001777U},
    {0x2F818U, 0x01001778U},
    {0x2F819U, 0x01001779U},
    {0x2F81AU, 0x0100177AU},
    {0x2F81BU, 0x01001162U},
    {0x2F81CU, 0x0100177BU},
    {0x2F81DU, 0x010009D3U},
    {0x2F81EU, 0x0100177CU},
    {0x2F81FU, 0x0100177DU},
    {0x2F820U, 0x0100177EU},
    {0x2F821U, 0x0100177FU},
    {0x2F822U, 0x01001744U},
    {0x2F823U, 0x01001780U},
    {0x2F824U, 0x01001781U},
    {0x2F825U, 0x01001167U},
    {0x2F826U, 0x0100112CU},
    {0x2F827U, 0x0100112DU},
    {0x2F828U, 0x01001168U},
    {0x2F829U, 0x01001782U},
    {0x2F82AU, 0x01001783U},
    {0x2F82BU, 0x01001076U},
    {0x2F82CU, 0x01001784U},
    {0x2F82DU, 0x0100112EU},
    {0x2F82EU, 0x01001785U},
    {0x2F82FU, 0x01001786U},
    {0x2F830U, 0x01001787U},
    {0x2F831U, 0x01001788U},
    {0x2F832U, 0x01001788U},
    {0x2F833U, 0x01001788U},
    {0x2F834U, 0x01001789U},
    {0x2F835U, 0x0100178AU},
    {0x2F836U, 0x0100178BU},
    {0x2F837U, 0x0100178CU},
    {0x2F838U, 0x0100178DU},
    {0x2F839U, 0x0100178EU},
    {0x2F83AU, 0x0100178FU},
    {0x2F83BU, 0x01001790U},
    {0x2F83CU, 0x01001791U},
    {0x2F83DU, 0x01001792U},
    {0x2F83EU, 0x01001793U},
    {0x2F83FU, 0x01001794U},
    {0x2F840U, 0x01001795U},
    {0x2F841U, 0x01001796U},
    {0x2F842U, 0x01001797U},
    {0x2F843U, 0x01001798U},
    {0x2F844U, 0x01001799U},
    {0x2F845U, 0x0100179AU},
    {0x2F846U, 0x0100179AU},
    {0x2F847U, 0x0100116AU},
    {0x2F848U, 0x0100179BU},
    {0x2F849U, 0x0100179CU},
    {0x2F84AU, 0x0100179DU},
    {0x2F84BU, 0x0100179EU},
    {0x2F84CU, 0x01001130U},
    {0x2F84DU, 0x0100179FU},
    {0x2F84EU, 0x010017A0U},
    {0x2F84FU, 0x010017A1U},
    {0x2F850U, 0x01001108U},
    {0x2F851U, 0x010017A2U},
    {0x2F852U, 0x010017A3U},
    {0x2F853U, 0x010017A4U},
    {0x2F854U, 0x010017A5U},
    {0x2F855U, 0x010017A6U},
    {0x2F856U, 0x010017A7U},
    {0x2F857U, 0x010017A8U},
    {0x2F858U, 0x010017A9U},
    {0x2F859U, 0x010017AAU},
    {0x2F85AU, 0x010017ABU},
    {0x2F85BU, 0x010017ACU},
    {0x2F85CU, 0x010017ADU},
    {0x2F85DU, 0x0100172BU},
    {0x2F85EU, 0x010017AEU},
    {0x2F85FU, 0x010017AFU},
    {0x2F860U, 0x010017B0U},
    {0x2F861U, 0x010017B1U},
    {0x2F862U, 0x010017B2U},
    {0x2F863U, 0x010017B3U},
    {0x2F864U, 0x010017B4U},
    {0x2F865U, 0x010017B5U},
    {0x2F866U, 0x010017B6U},
    {0x2F867U, 0x010017B7U},
    {0x2F868U, 0x010017B8U},
    {0x2F869U, 0x010017B9U},
    {0x2F86AU, 0x010017BAU},
    {0x2F86BU, 0x010017BAU},
    {0x2F86CU, 0x010017BBU},
    {0x2F86DU, 0x010017BCU},
    {0x2F86EU, 0x010017BDU},
    {0x2F86FU, 0x01001072U},
    {0x2F870U, 0x010017BEU},
    {0x2F871U, 0x010017BFU},
    {0x2F872U, 0x010017C0U},
    {0x2F873U, 0x010017C1U},
    {0x2F874U, 0x010017C2U},
    {0x2F875U, 0x010009EDU},
    {0x2F876U, 0x010017C3U},
    {0x2F877U, 0x010017C4U},
    {0x2F878U, 0x010009EFU},
    {0x2F879U, 0x010017C5U},
    {0x2F87AU, 0x010017C6U},
    {0x2F87BU, 0x010017C7U},
    {0x2F87CU, 0x010017C8U},
    {0x2F87DU, 0x010017C9U},
    {0x2F87EU, 0x010017CAU},
    {0x2F87FU, 0x010017CBU},
    {0x2F880U, 0x010017CCU},
    {0x2F881U, 0x010017CDU},
    {0x2F882U, 0x010017CEU},
    {0x2F883U, 0x010017CFU},
    {0x2F884U, 0x010017D0U},
    {0x2F885U, 0x010017D1U},
    {0x2F886U, 0x010017D2U},
    {0x2F887U, 0x010017D3U},
    {0x2F888U, 0x010017D4U},
    {0x2F889U, 0x010017D5U},
    {0x2F88AU, 0x010017D6U},
    {0x2F88BU, 0x010017D7U},
    {0x2F88CU, 0x010017D8U},
    {0x2F88DU, 0x010017D9U},
    {0x2F88EU, 0x0100103EU},
    {0x2F88FU, 0x010017DAU},
    {0x2F890U, 0x010009F9U},
    {0x2F891U, 0x010017DBU},
    {0x2F892U, 0x010017DBU},
    {0x2F893U, 0x010017DCU},
    {0x2F894U, 0x010017DDU},
    {0x2F895U, 0x010017DDU},
    {0x2F896U, 0x010017DEU},
    {0x2F897U, 0x010017DFU},
    {0x2F898U, 0x010017E0U},
    {0x2F899U, 0x010017E1U},
    {0x2F89AU, 0x010017E2U},
    {0x2F89BU, 0x010017E3U},
    {0x2F89CU, 0x010017E4U},
    {0x2F89DU, 0x010017E5U},
    {0x2F89EU, 0x010017E6U},
    {0x2F89FU, 0x010017E7U},
    {0x2F8A0U, 0x010017E8U},
    {0x2F8A1U, 0x010017E9U},
    {0x2F8A2U, 0x010017EAU},
    {0x2F8A3U, 0x01001135U},
    {0x2F8A4U, 0x010017EBU},
    {0x2F8A5U, 0x010017ECU},
    {0x2F8A6U, 0x010017EDU},
    {0x2F8A7U, 0x010017EEU},
    {0x2F8A8U, 0x01001176U},
    {0x2F8A9U, 0x010017EEU},
    {0x2F8AAU, 0x010017EFU},
    {0x2F8ABU, 0x01001137U},
    {0x2F8ACU, 0x010017F0U},
    {0x2F8ADU, 0x010017F1U},
    {0x2F8AEU, 0x010017F2U},
    {0x2F8AFU, 0x010017F3U},
    {0x2F8B0U, 0x01001138U},
    {0x2F8B1U, 0x01001023U},
    {0x2F8B2U, 0x010017F4U},
    {0x2F8B3U, 0x010017F5U},
    {0x2F8B4U, 0x010017F6U},
    {0x2F8B5U, 0x010017F7U},
    {0x2F8B6U, 0x010017F8U},
    {0x2F8B7U, 0x010017F9U},
    {0x2F8B8U, 0x010017FAU},
    {0x2F8B9U, 0x010017FBU},
    {0x2F8BAU, 0x010017FCU},
    {0x2F8BBU, 0x010017FDU},
    {0x2F8BCU, 0x010017FEU},
    {0x2F8BDU, 0x010017FFU},
    {0x2F8BEU, 0x01001800U},
    {0x2F8BFU, 0x01001801U},
    {0x2F8C0U, 0x01001802U},
    {0x2F8C1U, 0x01001803U},
    {0x2F8C2U, 0x01001804U},
    {0x2F8C3U, 0x01001805U},
    {0x2F8C4U, 0x01001806U},
    {0x2F8C5U, 0x01001807U},
    {0x2F8C6U, 0x01001808U},
    {0x2F8C7U, 0x01001809U},
    {0x2F8C8U, 0x01001139U},
    {0x2F8C9U, 0x0100180AU},
    {0x2F8CAU, 0x0100180BU},
    {0x2F8CBU, 0x0100180CU},
    {0x2F8CCU, 0x0100180DU},
    {0x2F8CDU, 0x0100180EU},
    {0x2F8CEU, 0x0100180FU},
    {0x2F8CFU, 0x0100113BU},
    {0x2F8D0U, 0x01001810U},
    {0x2F8D1U, 0x01001811U},
    {0x2F8D2U, 0x01001812U},
    {0x2F8D3U, 0x01001813U},
    {0x2F8D4U, 0x01001814U},
    {0x2F8D5U, 0x01001815U},
    {0x2F8D6U, 0x01001816U},
    {0x2F8D7U, 0x01001817U},
    {0x2F8D8U, 0x0100103FU},
    {0x2F8D9U, 0x0100117EU},
    {0x2F8DAU, 0x01001818U},
    {0x2F8DBU, 0x01001819U},
    {0x2F8DCU, 0x0100181AU},
    {0x2F8DDU, 0x0100181BU},
    {0x2F8DEU, 0x0100181CU},
    {0x2F8DFU, 0x0100181DU},
    {0x2F8E0U, 0x0100181EU},
    {0x2F8E1U, 0x0100181FU},
    {0x2F8E2U, 0x0100113CU},
    {0x2F8E3U, 0x01001820U},
    {0x2F8E4U, 0x01001821U},
    {0x2F8E5U, 0x01001822U},
    {0x2F8E6U, 0x01001823U},
    {0x2F8E7U, 0x010011A8U},
    {0x2F8E8U, 0x01001824U},
    {0x2F8E9U, 0x01001825U},
    {0x2F8EAU, 0x01001826U},
    {0x2F8EBU, 0x01001827U},
    {0x2F8ECU, 0x01001828U},
    {0x2F8EDU, 0x01001829U},
    {0x2F8EEU, 0x0100182AU},
    {0x2F8EFU, 0x0100182BU},
    {0x2F8F0U, 0x0100182CU},
    {0x2F8F1U, 0x0100182DU},
    {0x2F8F2U, 0x0100182EU},
    {0x2F8F3U, 0x0100182FU},
    {0x2F8F4U, 0x01001830U},
    {0x2F8F5U, 0x01001083U},
    {0x2F8F6U, 0x01001831U},
    {0x2F8F7U, 0x01001832U},
    {0x2F8F8U, 0x01001833U},
    {0x2F8F9U, 0x01001834U},
    {0x2F8FAU, 0x01001835U},
    {0x2F8FBU, 0x01001836U},
    {0x2F8FCU, 0x01001837U},
    {0x2F8FDU, 0x01001838U},
    {0x2F8FEU, 0x01001839U},
    {0x2F8FFU, 0x0100183AU},
    {0x2F900U, 0x0100183BU},
    {0x2F901U, 0x0100113DU},
    {0x2F902U, 0x010010D6U},
    {0x2F903U, 0x0100183CU},
    {0x2F904U, 0x0100183DU},
    {0x2F905U, 0x0100183EU},
    {0x2F906U, 0x0100183FU},
    {0x2F907U, 0x01001840U},
    {0x2F908U, 0x01001841U},
    {0x2F909U, 0x01001842U},
    {0x2F90AU, 0x01001843U},
    {0x2F90BU, 0x01001181U},
    {0x2F90CU, 0x01001844U},
    {0x2F90DU, 0x01001845U},
    {0x2F90EU, 0x01001846U},
    {0x2F90FU, 0x01001847U},
    {0x2F910U, 0x01001848U},
    {0x2F911U, 0x01001849U},
    {0x2F912U, 0x0100184AU},
    {0x2F913U, 0x0100184BU},
    {0x2F914U, 0x01001182U},
    {0x2F915U, 0x0100184CU},
    {0x2F916U, 0x0100184DU},
    {0x2F917U, 0x0100184EU},
    {0x2F918U, 0x0100184FU},
    {0x2F919U, 0x01001850U},
    {0x2F91AU, 0x01001851U},
    {0x2F91BU, 0x01001852U},
    {0x2F91CU, 0x01001853U},
    {0x2F91DU, 0x01001854U},
    {0x2F91EU, 0x01001855U},
    {0x2F91FU, 0x01001856U},
    {0x2F920U, 0x01001857U},
    {0x2F921U, 0x01001184U},
    {0x2F922U, 0x01001858U},
    {0x2F923U, 0x01001859U},
    {0x2F924U, 0x0100185AU},
    {0x2F925U, 0x0100185BU},
    {0x2F926U, 0x0100185CU},
    {0x2F927U, 0x0100185DU},
    {0x2F928U, 0x0100185EU},
    {0x2F929U, 0x0100185FU},
    {0x2F92AU, 0x01001860U},
    {0x2F92BU, 0x01001861U},
    {0x2F92CU, 0x01001862U},
    {0x2F92DU, 0x01001862U},
    {0x2F92EU, 0x01001863U},
    {0x2F92FU, 0x01001864U},
    {0x2F930U, 0x01001186U},
    {0x2F931U, 0x01001865U},
    {0x2F932U, 0x01001866U},
    {0x2F933U, 0x01001867U},
    {0x2F934U, 0x01001868U},
    {0x2F935U, 0x01001869U},
    {0x2F936U, 0x0100186AU},
    {0x2F937U, 0x0100186BU},
    {0x2F938U, 0x01001075U},
    {0x2F939U, 0x0100186CU},
    {0x2F93AU, 0x0100186DU},
    {0x2F93BU, 0x0100186EU},
    {0x2F93CU, 0x0100186FU},
    {0x2F93DU, 0x01001870U},
    {0x2F93EU, 0x01001871U},
    {0x2F93FU, 0x01001872U},
    {0x2F940U, 0x0100118CU},
    {0x2F941U, 0x01001873U},
    {0x2F942U, 0x01001874U},
    {0x2F943U, 0x01001875U},
    {0x2F944U, 0x01001876U},
    {0x2F945U, 0x01001877U},
    {0x2F946U, 0x01001878U},
    {0x2F947U, 0x01001878U},
    {0x2F948U, 0x0100118DU},
    {0x2F949U, 0x010011AAU},
    {0x2F94AU, 0x01001879U},
    {0x2F94BU, 0x0100187AU},
    {0x2F94CU, 0x0100187BU},
    {0x2F94DU, 0x0100187CU},
    {0x2F94EU, 0x0100187DU},
    {0x2F94FU, 0x01001050U},
    {0x2F950U, 0x0100118FU},
    {0x2F951U, 0x0100187EU},
    {0x2F952U, 0x0100187FU},
    {0x2F953U, 0x01001147U},
    {0x2F954U, 0x01001880U},
    {0x2F955U, 0x01001881U},
    {0x2F956U, 0x0100111CU},
    {0x2F957U, 0x01001882U},
    {0x2F958U, 0x01001883U},
    {0x2F959U, 0x0100114AU},
    {0x2F95AU, 0x01001884U},
    {0x2F95BU, 0x01001885U},
    {0x2F95CU, 0x01001886U},
    {0x2F95DU, 0x01001887U},
    {0x2F95EU, 0x01001887U},
    {0x2F95FU, 0x01001888U},
    {0x2F960U, 0x01001889U},
    {0x2F961U, 0x0100188AU},
    {0x2F962U, 0x0100188BU},
    {0x2F963U, 0x0100188CU},
    {0x2F964U, 0x0100188DU},
    {0x2F965U, 0x0100188EU},
    {0x2F966U, 0x0100188FU},
    {0x2F967U, 0x01001890U},
    {0x2F968U, 0x01001891U},
    {0x2F969U, 0x01001892U},
    {0x2F96AU, 0x01001893U},
    {0x2F96BU, 0x01001894U},
    {0x2F96CU, 0x01001895U},
    {0x2F96DU, 0x01001896U},
    {0x2F96EU, 0x01001897U},
    {0x2F96FU, 0x01001898U},
    {0x2F970U, 0x01001899U},
    {0x2F971U, 0x0100189AU},
    {0x2F972U, 0x0100189BU},
    {0x2F973U, 0x0100189CU},
    {0x2F974U, 0x0100189DU},
    {0x2F975U, 0x0100189EU},
    {0x2F976U, 0x0100189FU},
    {0x2F977U, 0x010018A0U},
    {0x2F978U, 0x010018A1U},
    {0x2F979U, 0x010018A2U},
    {0x2F97AU, 0x01001150U},
    {0x2F97BU, 0x010018A3U},
    {0x2F97CU, 0x010018A4U},
    {0x2F97DU, 0x010018A5U},
    {0x2F97EU, 0x010018A6U},
    {0x2F97FU, 0x010018A7U},
    {0x2F980U, 0x010018A8U},
    {0x2F981U, 0x010018A9U},
    {0x2F982U, 0x010018AAU},
    {0x2F983U, 0x010018ABU},
    {0x2F984U, 0x010018ACU},
    {0x2F985U, 0x010018ADU},
    {0x2F986U, 0x010018AEU},
    {0x2F987U, 0x010018AFU},
    {0x2F988U, 0x010018B0U},
    {0x2F989U, 0x010018B1U},
    {0x2F98AU, 0x010018B2U},
    {0x2F98BU, 0x010017DCU},
    {0x2F98CU, 0x010018B3U},
    {0x2F98DU, 0x010018B4U},
    {0x2F98EU, 0x010018B5U},
    {0x2F98FU, 0x010018B6U},
    {0x2F990U, 0x010018B7U},
    {0x2F991U, 0x010018B8U},
    {0x2F992U, 0x010018B9U},
    {0x2F993U, 0x010018BAU},
    {0x2F994U, 0x010018BBU},
    {0x2F995U, 0x010018BCU},
    {0x2F996U, 0x010018BDU},
    {0x2F997U, 0x010018BEU},
    {0x2F998U, 0x01001086U},
    {0x2F999U, 0x010018BFU},
    {0x2F99AU, 0x010018C0U},
    {0x2F99BU, 0x010018C1U},
    {0x2F99CU, 0x010018C2U},
    {0x2F99DU, 0x010018C3U},
    {0x2F99EU, 0x010018C4U},
    {0x2F99FU, 0x01001153U},
    {0x2F9A0U, 0x010018C5U},
    {0x2F9A1U, 0x010018C6U},
    {0x2F9A2U, 0x010018C7U},
    {0x2F9A3U, 0x010018C8U},
    {0x2F9A4U, 0x010018C9U},
    {0x2F9A5U, 0x010018CAU},
    {0x2F9A6U, 0x010018CBU},
    {0x2F9A7U, 0x010018CCU},
    {0x2F9A8U, 0x010018CDU},
    {0x2F9A9U, 0x010018CEU},
    {0x2F9AAU, 0x010018CFU},
    {0x2F9ABU, 0x010018D0U},
    {0x2F9ACU, 0x010018D1U},
    {0x2F9ADU, 0x010018D2U},
    {0x2F9AEU, 0x010018D3U},
    {0x2F9AFU, 0x010018D4U},
    {0x2F9B0U, 0x010018D5U},
    {0x2F9B1U, 0x010018D6U},
    {0x2F9B2U, 0x010018D7U},
    {0x2F9B3U, 0x010018D8U},
    {0x2F9B4U, 0x0100104BU},
    {0x2F9B5U, 0x010018D9U},
    {0x2F9B6U, 0x010018DAU},
    {0x2F9B7U, 0x010018DBU},
    {0x2F9B8U, 0x010018DCU},
    {0x2F9B9U, 0x010018DDU},
    {0x2F9BAU, 0x010018DEU},
    {0x2F9BBU, 0x01001196U},
    {0x2F9BCU, 0x010018DFU},
    {0x2F9BDU, 0x010018E0U},
    {0x2F9BEU, 0x010018E1U},
    {0x2F9BFU, 0x010018E2U},
    {0x2F9C0U, 0x010018E3U},
    {0x2F9C1U, 0x010018E4U},
    {0x2F9C2U, 0x010018E5U},
    {0x2F9C3U, 0x010018E6U},
    {0x2F9C4U, 0x01000A53U},
    {0x2F9C5U, 0x010018E7U},
    {0x2F9C6U, 0x010018E8U},
    {0x2F9C7U, 0x010018E9U},
    {0x2F9C8U, 0x010018EAU},
    {0x2F9C9U, 0x010018EBU},
    {0x2F9CAU, 0x010018ECU},
    {0x2F9CBU, 0x010018EDU},
    {0x2F9CCU, 0x010018EEU},
    {0x2F9CDU, 0x010018EFU},
    {0x2F9CEU, 0x010018F0U},
    {0x2F9CFU, 0x010018F1U},
    {0x2F9D0U, 0x0100119BU},
    {0x2F9D1U, 0x0100119CU},
    {0x2F9D2U, 0x01000A5AU},
    {0x2F9D3U, 0x010018F2U},
    {0x2F9D4U, 0x010018F3U},
    {0x2F9D5U, 0x010018F4U},
    {0x2F9D6U, 0x010018F5U},
    {0x2F9D7U, 0x010018F6U},
    {0x2F9D8U, 0x010018F7U},
    {0x2F9D9U, 0x010018F8U},
    {0x2F9DAU, 0x010018F9U},
    {0x2F9DBU, 0x010018FAU},
    {0x2F9DCU, 0x010018FBU},
    {0x2F9DDU, 0x010018FCU},
    {0x2F9DEU, 0x010018FDU},
    {0x2F9DFU, 0x0100119DU},
    {0x2F9E0U, 0x010018FEU},
    {0x2F9E1U, 0x010018FFU},
    {0x2F9E2U, 0x01001900U},
    {0x2F9E3U, 0x01001901U},
    {0x2F9E4U, 0x01001902U},
    {0x2F9E5U, 0x01001903U},
    {0x2F9E6U, 0x01001904U},
    {0x2F9E7U, 0x01001905U},
    {0x2F9E8U, 0x01001906U},
    {0x2F9E9U, 0x01001907U},
    {0x2F9EAU, 0x01001908U},
    {0x2F9EBU, 0x01001909U},
    {0x2F9ECU, 0x0100190AU},
    {0x2F9EDU, 0x0100190BU},
    {0x2F9EEU, 0x0100190CU},
    {0x2F9EFU, 0x0100190DU},
    {0x2F9F0U, 0x0100190EU},
    {0x2F9F1U, 0x0100190FU},
    {0x2F9F2U, 0x01001910U},
    {0x2F9F3U, 0x01001911U},
    {0x2F9F4U, 0x01001912U},
    {0x2F9F5U, 0x01001913U},
    {0x2F9F6U, 0x01001914U},
    {0x2F9F7U, 0x01001915U},
    {0x2F9F8U, 0x01001916U},
    {0x2F9F9U, 0x01001917U},
    {0x2F9FAU, 0x01001918U},
    {0x2F9FBU, 0x01001919U},
    {0x2F9FCU, 0x0100191AU},
    {0x2F9FDU, 0x0100191BU},
    {0x2F9FEU, 0x010011A3U},
    {0x2F9FFU, 0x010011A3U},
    {0x2FA00U, 0x0100191CU},
    {0x2FA01U, 0x0100191DU},
    {0x2FA02U, 0x0100191EU},
    {0x2FA03U, 0x0100191FU},
    {0x2FA04U, 0x01001920U},
    {0x2FA05U, 0x01001921U},
    {0x2FA06U, 0x01001922U},
    {0x2FA07U, 0x01001923U},
    {0x2FA08U, 0x01001924U},
    {0x2FA09U, 0x01001925U},
    {0x2FA0AU, 0x010011A4U},
    {0x2FA0BU, 0x01001926U},
    {0x2FA0CU, 0x01001927U},
    {0x2FA0DU, 0x01001928U},
    {0x2FA0EU, 0x01001929U},
    {0x2FA0FU, 0x0100192AU},
    {0x2FA10U, 0x0100192BU},
    {0x2FA11U, 0x0100192CU},
    {0x2FA12U, 0x0100192DU},
    {0x2FA13U, 0x0100192EU},
    {0x2FA14U, 0x0100192FU},
    {0x2FA15U, 0x01000A8AU},
    {0x2FA16U, 0x01001930U},
    {0x2FA17U, 0x01000A8EU},
    {0x2FA18U, 0x01001931U},
    {0x2FA19U, 0x01001932U},
    {0x2FA1AU, 0x01001933U},
    {0x2FA1BU, 0x01001934U},
    {0x2FA1CU, 0x01000A93U},
    {0x2FA1DU, 0x01001935U},
};

static const uint32_t cs_udecomp_pool[6454] = {
    0x20U, 0x20U, 0x308U, 0x61U, 0x20U, 0x304U, 0x32U, 0x33U, 0x20U, 0x301U, 0x3BCU, 0x20U,
    0x327U, 0x31U, 0x6FU, 0x31U, 0x2044U, 0x34U, 0x31U, 0x2044U, 0x32U, 0x33U, 0x2044U, 0x34U,
    0x41U, 0x300U, 0x41U, 0x301U, 0x41U, 0x302U, 0x41U, 0x303U, 0x41U, 0x308U, 0x41U, 0x30AU,
    0x43U, 0x327U, 0x45U, 0x300U, 0x45U, 0x301U, 0x45U, 0x302U, 0x45U, 0x308U, 0x49U, 0x300U,
    0x49U, 0x301U, 0x49U, 0x302U, 0x49U, 0x308U, 0x4EU, 0x303U, 0x4FU, 0x300U, 0x4FU, 0x301U,
    0x4FU, 0x302U, 0x4FU, 0x303U, 0x4FU, 0x308U, 0x55U, 0x300U, 0x55U, 0x301U, 0x55U, 0x302U,
    0x55U, 0x308U, 0x59U, 0x301U, 0x61U, 0x300U, 0x61U, 0x301U, 0x61U, 0x302U, 0x61U, 0x303U,
    0x61U, 0x308U, 0x61U, 0x30AU, 0x63U, 0x327U, 0x65U, 0x300U, 0x65U, 0x301U, 0x65U, 0x302U,
    0x65U, 0x308U, 0x69U, 0x300U, 0x69U, 0x301U, 0x69U, 0x302U, 0x69U, 0x308U, 0x6EU, 0x303U,
    0x6FU, 0x300U, 0x6FU, 0x301U, 0x6FU, 0x302U, 0x6FU, 0x303U, 0x6FU, 0x308U, 0x75U, 0x300U,
    0x75U, 0x301U, 0x75U, 0x302U, 0x75U, 0x308U, 0x79U, 0x301U, 0x79U, 0x308U, 0x41U, 0x304U,
    0x61U, 0x304U, 0x41U, 0x306U, 0x61U, 0x306U, 0x41U, 0x328U, 0x61U, 0x328U, 0x43U, 0x301U,
    0x63U, 0x301U, 0x43U, 0x302U, 0x63U, 0x302U, 0x43U, 0x307U, 0x63U, 0x307U, 0x43U, 0x30CU,
    0x63U, 0x30CU, 0x44U, 0x30CU, 0x64U, 0x30CU, 0x45U, 0x304U, 0x65U, 0x304U, 0x45U, 0x306U,
    0x65U, 0x306U, 0x45U, 0x307U, 0x65U, 0x307U, 0x45U, 0x328U, 0x65U, 0x328U, 0x45U, 0x30CU,
    0x65U, 0x30CU, 0x47U, 0x302U, 0x67U, 0x302U, 0x47U, 0x306U, 0x67U, 0x306U, 0x47U, 0x307U,
    0x67U, 0x307U, 0x47U, 0x327U, 0x67U, 0x327U, 0x48U, 0x302U, 0x68U, 0x302U, 0x49U, 0x303U,
    0x69U, 0x303U, 0x49U, 0x304U, 0x69U, 0x304U, 0x49U, 0x306U, 0x69U, 0x306U, 0x49U, 0x328U,
    0x69U, 0x328U, 0x49U, 0x307U, 0x49U, 0x4AU, 0x69U, 0x6AU, 0x4AU, 0x302U, 0x6AU, 0x302U,
    0x4BU, 0x327U, 0x6BU, 0x327U, 0x4CU, 0x301U, 0x6CU, 0x301U, 0x4CU, 0x327U, 0x6CU, 0x327U,
    0x4CU, 0x30CU, 0x6CU, 0x30CU, 0x4CU, 0xB7U, 0x6CU, 0xB7U, 0x4EU, 0x301U, 0x6EU, 0x301U,
    0x4EU, 0x327U, 0x6EU, 0x327U, 0x4EU, 0x30CU, 0x6EU, 0x30CU, 0x2BCU, 0x6EU, 0x4FU, 0x304U,
    0x6FU, 0x304U, 0x4FU, 0x306U, 0x6FU, 0x306U, 0x4FU, 0x30BU, 0x6FU, 0x30BU, 0x52U, 0x301U,
    0x72U, 0x301U, 0x52U, 0x327U, 0x72U, 0x327U, 0x52U, 0x30CU, 0x72U, 0x30CU, 0x53U, 0x301U,
    0x73U, 0x301U, 0x53U, 0x302U, 0x73U, 0x302U, 0x53U, 0x327U, 0x73U, 0x327U, 0x53U, 0x30CU,
    0x73U, 0x30CU, 0x54U, 0x327U, 0x74U, 0x327U, 0x54U, 0x30CU, 0x74U, 0x30CU, 0x55U, 0x303U,
    0x75U, 0x303U, 0x55U, 0x304U, 0x75U, 0x304U, 0x55U, 0x306U, 0x75U, 0x306U, 0x55U, 0x30AU,
    0x75U, 0x30AU, 0x55U, 0x30BU, 0x75U, 0x30BU, 0x55U, 0x328U, 0x75U, 0x328U, 0x57U, 0x302U,
    0x77U, 0x302U, 0x59U, 0x302U, 0x79U, 0x302U, 0x59U, 0x308U, 0x5AU, 0x301U, 0x7AU, 0x301U,
    0x5AU, 0x307U, 0x7AU, 0x307U, 0x5AU, 0x30CU, 0x7AU, 0x30CU, 0x73U, 0x4FU, 0x31BU, 0x6FU,
    0x31BU, 0x55U, 0x31BU, 0x75U, 0x31BU, 0x44U, 0x17DU, 0x44U, 0x17EU, 0x64U, 0x17EU, 0x4CU,
    0x4AU, 0x4CU, 0x6AU, 0x6CU, 0x6AU, 0x4EU, 0x4AU, 0x4EU, 0x6AU, 0x6EU, 0x6AU, 0x41U,
    0x30CU, 0x61U, 0x30CU, 0x49U, 0x30CU, 0x69U, 0x30CU, 0x4FU, 0x30CU, 0x6FU, 0x30CU, 0x55U,
    0x30CU, 0x75U, 0x30CU, 0xDCU, 0x304U, 0xFCU, 0x304U, 0xDCU, 0x301U, 0xFCU, 0x301U, 0xDCU,
    0x30CU, 0xFCU, 0x30CU, 0xDCU, 0x300U, 0xFCU, 0x300U, 0xC4U, 0x304U, 0xE4U, 0x304U, 0x226U,
    0x304U, 0x227U, 0x304U, 0xC6U, 0x304U, 0xE6U, 0x304U, 0x47U, 0x30CU, 0x67U, 0x30CU, 0x4BU,
    0x30CU, 0x6BU, 0x30CU, 0x4FU, 0x328U, 0x6FU, 0x328U, 0x1EAU, 0x304U, 0x1EBU, 0x304U, 0x1B7U,
    0x30CU, 0x292U, 0x30CU, 0x6AU, 0x30CU, 0x44U, 0x5AU, 0x44U, 0x7AU, 0x64U, 0x7AU, 0x47U,
    0x301U, 0x67U, 0x301U, 0x4EU, 0x300U, 0x6EU, 0x300U, 0xC5U, 0x301U, 0xE5U, 0x301U, 0xC6U,
    0x301U, 0xE6U, 0x301U, 0xD8U, 0x301U, 0xF8U, 0x301U, 0x41U, 0x30FU, 0x61U, 0x30FU, 0x41U,
    0x311U, 0x61U, 0x311U, 0x45U, 0x30FU, 0x65U, 0x30FU, 0x45U, 0x311U, 0x65U, 0x311U, 0x49U,
    0x30FU, 0x69U, 0x30FU, 0x49U, 0x311U, 0x69U, 0x311U, 0x4FU, 0x30FU, 0x6FU, 0x30FU, 0x4FU,
    0x311U, 0x6FU, 0x311U, 0x52U, 0x30FU, 0x72U, 0x30FU, 0x52U, 0x311U, 0x72U, 0x311U, 0x55U,
    0x30FU, 0x75U, 0x30FU, 0x55U, 0x311U, 0x75U, 0x311U, 0x53U, 0x326U, 0x73U, 0x326U, 0x54U,
    0x326U, 0x74U, 0x326U, 0x48U, 0x30CU, 0x68U, 0x30CU, 0x41U, 0x307U, 0x61U, 0x307U, 0x45U,
    0x327U, 0x65U, 0x327U, 0xD6U, 0x304U, 0xF6U, 0x304U, 0xD5U, 0x304U, 0xF5U, 0x304U, 0x4FU,
    0x307U, 0x6FU, 0x307U, 0x22EU, 0x304U, 0x22FU, 0x304U, 0x59U, 0x304U, 0x79U, 0x304U, 0x68U,
    0x266U, 0x6AU, 0x72U, 0x279U, 0x27BU, 0x281U, 0x77U, 0x79U, 0x20U, 0x306U, 0x20U, 0x307U,
    0x20U, 0x30AU, 0x20U, 0x328U, 0x20U, 0x303U, 0x20U, 0x30BU, 0x263U, 0x6CU, 0x78U, 0x295U,
    0x300U, 0x301U, 0x313U, 0x308U, 0x301U, 0x2B9U, 0x20U, 0x345U, 0x3BU, 0xA8U, 0x301U, 0x391U,
    0x301U, 0xB7U, 0x395U, 0x301U, 0x397U, 0x301U, 0x399U, 0x301U, 0x39FU, 0x301U, 0x3A5U, 0x301U,
    0x3A9U, 0x301U, 0x3CAU, 0x301U, 0x399U, 0x308U, 0x3A5U, 0x308U, 0x3B1U, 0x301U, 0x3B5U, 0x301U,
    0x3B7U, 0x301U, 0x3B9U, 0x301U, 0x3CBU, 0x301U, 0x3B9U, 0x308U, 0x3C5U, 0x308U, 0x3BFU, 0x301U,
    0x3C5U, 0x301U, 0x3C9U, 0x301U, 0x3B2U, 0x3B8U, 0x3A5U, 0x3D2U, 0x301U, 0x3D2U, 0x308U, 0x3C6U,
    0x3C0U, 0x3BAU, 0x3C1U, 0x3C2U, 0x398U, 0x3B5U, 0x3A3U, 0x415U, 0x300U, 0x415U, 0x308U, 0x413U,
    0x301U, 0x406U, 0x308U, 0x41AU, 0x301U, 0x418U, 0x300U, 0x423U, 0x306U, 0x418U, 0x306U, 0x438U,
    0x306U, 0x435U, 0x300U, 0x435U, 0x308U, 0x433U, 0x301U, 0x456U, 0x308U, 0x43AU, 0x301U, 0x438U,
    0x300U, 0x443U, 0x306U, 0x474U, 0x30FU, 0x475U, 0x30FU, 0x416U, 0x306U, 0x436U, 0x306U, 0x410U,
    0x306U, 0x430U, 0x306U, 0x410U, 0x308U, 0x430U, 0x308U, 0x415U, 0x306U, 0x435U, 0x306U, 0x4D8U,
    0x308U, 0x4D9U, 0x308U, 0x416U, 0x308U, 0x436U, 0x308U, 0x417U, 0x308U, 0x437U, 0x308U, 0x418U,
    0x304U, 0x438U, 0x304U, 0x418U, 0x308U, 0x438U, 0x308U, 0x41EU, 0x308U, 0x43EU, 0x308U, 0x4E8U,
    0x308U, 0x4E9U, 0x308U, 0x42DU, 0x308U, 0x44DU, 0x308U, 0x423U, 0x304U, 0x443U, 0x304U, 0x423U,
    0x308U, 0x443U, 0x308U, 0x423U, 0x30BU, 0x443U, 0x30BU, 0x427U, 0x308U, 0x447U, 0x308U, 0x42BU,
    0x308U, 0x44BU, 0x308U, 0x565U, 0x582U, 0x627U, 0x653U, 0x627U, 0x654U, 0x648U, 0x654U, 0x627U,
    0x655U, 0x64AU, 0x654U, 0x627U, 0x674U, 0x648U, 0x674U, 0x6C7U, 0x674U, 0x64AU, 0x674U, 0x6D5U,
    0x654U, 0x6C1U, 0x654U, 0x6D2U, 0x654U, 0x928U, 0x93CU, 0x930U, 0x93CU, 0x933U, 0x93CU, 0x915U,
    0x93CU, 0x916U, 0x93CU, 0x917U, 0x93CU, 0x91CU, 0x93CU, 0x921U, 0x93CU, 0x922U, 0x93CU, 0x92BU,
    0x93CU, 0x92FU, 0x93CU, 0x9C7U, 0x9BEU, 0x9C7U, 0x9D7U, 0x9A1U, 0x9BCU, 0x9A2U, 0x9BCU, 0x9AFU,
    0x9BCU, 0xA32U, 0xA3CU, 0xA38U, 0xA3CU, 0xA16U, 0xA3CU, 0xA17U, 0xA3CU, 0xA1CU, 0xA3CU, 0xA2BU,
    0xA3CU, 0xB47U, 0xB56U, 0xB47U, 0xB3EU, 0xB47U, 0xB57U, 0xB21U, 0xB3CU, 0xB22U, 0xB3CU, 0xB92U,
    0xBD7U, 0xBC6U, 0xBBEU, 0xBC7U, 0xBBEU, 0xBC6U, 0xBD7U, 0xC46U, 0xC56U, 0xCBFU, 0xCD5U, 0xCC6U,
    0xCD5U, 0xCC6U, 0xCD6U, 0xCC6U, 0xCC2U, 0xCCAU, 0xCD5U, 0xD46U, 0xD3EU, 0xD47U, 0xD3EU, 0xD46U,
    0xD57U, 0xDD9U, 0xDCAU, 0xDD9U, 0xDCFU, 0xDDCU, 0xDCAU, 0xDD9U, 0xDDFU, 0xE4DU, 0xE32U, 0xECDU,
    0xEB2U, 0xEABU, 0xE99U, 0xEABU, 0xEA1U, 0xF0BU, 0xF42U, 0xFB7U, 0xF4CU, 0xFB7U, 0xF51U, 0xFB7U,
    0xF56U, 0xFB7U, 0xF5BU, 0xFB7U, 0xF40U, 0xFB5U, 0xF71U, 0xF72U, 0xF71U, 0xF74U, 0xFB2U, 0xF80U,
    0xFB2U, 0xF81U, 0xFB3U, 0xF80U, 0xFB3U, 0xF81U, 0xF71U, 0xF80U, 0xF92U, 0xFB7U, 0xF9CU, 0xFB7U,
    0xFA1U, 0xFB7U, 0xFA6U, 0xFB7U, 0xFABU, 0xFB7U, 0xF90U, 0xFB5U, 0x1025U, 0x102EU, 0x10DCU, 0x1B05U,
    0x1B35U, 0x1B07U, 0x1B35U, 0x1B09U, 0x1B35U, 0x1B0BU, 0x1B35U, 0x1B0DU, 0x1B35U, 0x1B11U, 0x1B35U, 0x1B3AU,
    0x1B35U, 0x1B3CU, 0x1B35U, 0x1B3EU, 0x1B35U, 0x1B3FU, 0x1B35U, 0x1B42U, 0x1B35U, 0x41U, 0xC6U, 0x42U,
    0x44U, 0x45U, 0x18EU, 0x47U, 0x48U, 0x49U, 0x4AU, 0x4BU, 0x4CU, 0x4DU, 0x4EU, 0x4FU,
    0x222U, 0x50U, 0x52U, 0x54U, 0x55U, 0x57U, 0x250U, 0x251U, 0x1D02U, 0x62U, 0x64U, 0x65U,
    0x259U, 0x25BU, 0x25CU, 0x67U, 0x6BU, 0x6DU, 0x14BU, 0x254U, 0x1D16U, 0x1D17U, 0x70U, 0x74U,
    0x75U, 0x1D1DU, 0x26FU, 0x76U, 0x1D25U, 0x3B3U, 0x3B4U, 0x3C7U, 0x69U, 0x43DU, 0x252U, 0x63U,
    0x255U, 0xF0U, 0x66U, 0x25FU, 0x261U, 0x265U, 0x268U, 0x269U, 0x26AU, 0x1D7BU, 0x29DU, 0x26DU,
    0x1D85U, 0x29FU, 0x271U, 0x270U, 0x272U, 0x273U, 0x274U, 0x275U, 0x278U, 0x282U, 0x283U, 0x1ABU,
    0x289U, 0x28AU, 0x1D1CU, 0x28BU, 0x28CU, 0x7AU, 0x290U, 0x291U, 0x292U, 0x41U, 0x325U, 0x61U,
    0x325U, 0x42U, 0x307U, 0x62U, 0x307U, 0x42U, 0x323U, 0x62U, 0x323U, 0x42U, 0x331U, 0x62U,
    0x331U, 0xC7U, 0x301U, 0xE7U, 0x301U, 0x44U, 0x307U, 0x64U, 0x307U, 0x44U, 0x323U, 0x64U,
    0x323U, 0x44U, 0x331U, 0x64U, 0x331U, 0x44U, 0x327U, 0x64U, 0x327U, 0x44U, 0x32DU, 0x64U,
    0x32DU, 0x112U, 0x300U, 0x113U, 0x300U, 0x112U, 0x301U, 0x113U, 0x301U, 0x45U, 0x32DU, 0x65U,
    0x32DU, 0x45U, 0x330U, 0x65U, 0x330U, 0x228U, 0x306U, 0x229U, 0x306U, 0x46U, 0x307U, 0x66U,
    0x307U, 0x47U, 0x304U, 0x67U, 0x304U, 0x48U, 0x307U, 0x68U, 0x307U, 0x48U, 0x323U, 0x68U,
    0x323U, 0x48U, 0x308U, 0x68U, 0x308U, 0x48U, 0x327U, 0x68U, 0x327U, 0x48U, 0x32EU, 0x68U,
    0x32EU, 0x49U, 0x330U, 0x69U, 0x330U, 0xCFU, 0x301U, 0xEFU, 0x301U, 0x4BU, 0x301U, 0x6BU,
    0x301U, 0x4BU, 0x323U, 0x6BU, 0x323U, 0x4BU, 0x331U, 0x6BU, 0x331U, 0x4CU, 0x323U, 0x6CU,
    0x323U, 0x1E36U, 0x304U, 0x1E37U, 0x304U, 0x4CU, 0x331U, 0x6CU, 0x331U, 0x4CU, 0x32DU, 0x6CU,
    0x32DU, 0x4DU, 0x301U, 0x6DU, 0x301U, 0x4DU, 0x307U, 0x6DU, 0x307U, 0x4DU, 0x323U, 0x6DU,
    0x323U, 0x4EU, 0x307U, 0x6EU, 0x307U, 0x4EU, 0x323U, 0x6EU, 0x323U, 0x4EU, 0x331U, 0x6EU,
    0x331U, 0x4EU, 0x32DU, 0x6EU, 0x32DU, 0xD5U, 0x301U, 0xF5U, 0x301U, 0xD5U, 0x308U, 0xF5U,
    0x308U, 0x14CU, 0x300U, 0x14DU, 0x300U, 0x14CU, 0x301U, 0x14DU, 0x301U, 0x50U, 0x301U, 0x70U,
    0x301U, 0x50U, 0x307U, 0x70U, 0x307U, 0x52U, 0x307U, 0x72U, 0x307U, 0x52U, 0x323U, 0x72U,
    0x323U, 0x1E5AU, 0x304U, 0x1E5BU, 0x304U, 0x52U, 0x331U, 0x72U, 0x331U, 0x53U, 0x307U, 0x73U,
    0x307U, 0x53U, 0x323U, 0x73U, 0x323U, 0x15AU, 0x307U, 0x15BU, 0x307U, 0x160U, 0x307U, 0x161U,
    0x307U, 0x1E62U, 0x307U, 0x1E63U, 0x307U, 0x54U, 0x307U, 0x74U, 0x307U, 0x54U, 0x323U, 0x74U,
    0x323U, 0x54U, 0x331U, 0x74U, 0x331U, 0x54U, 0x32DU, 0x74U, 0x32DU, 0x55U, 0x324U, 0x75U,
    0x324U, 0x55U, 0x330U, 0x75U, 0x330U, 0x55U, 0x32DU, 0x75U, 0x32DU, 0x168U, 0x301U, 0x169U,
    0x301U, 0x16AU, 0x308U, 0x16BU, 0x308U, 0x56U, 0x303U, 0x76U, 0x303U, 0x56U, 0x323U, 0x76U,
    0x323U, 0x57U, 0x300U, 0x77U, 0x300U, 0x57U, 0x301U, 0x77U, 0x301U, 0x57U, 0x308U, 0x77U,
    0x308U, 0x57U, 0x307U, 0x77U, 0x307U, 0x57U, 0x323U, 0x77U, 0x323U, 0x58U, 0x307U, 0x78U,
    0x307U, 0x58U, 0x308U, 0x78U, 0x308U, 0x59U, 0x307U, 0x79U, 0x307U, 0x5AU, 0x302U, 0x7AU,
    0x302U, 0x5AU, 0x323U, 0x7AU, 0x323U, 0x5AU, 0x331U, 0x7AU, 0x331U, 0x68U, 0x331U, 0x74U,
    0x308U, 0x77U, 0x30AU, 0x79U, 0x30AU, 0x61U, 0x2BEU, 0x17FU, 0x307U, 0x41U, 0x323U, 0x61U,
    0x323U, 0x41U, 0x309U, 0x61U, 0x309U, 0xC2U, 0x301U, 0xE2U, 0x301U, 0xC2U, 0x300U, 0xE2U,
    0x300U, 0xC2U, 0x309U, 0xE2U, 0x309U, 0xC2U, 0x303U, 0xE2U, 0x303U, 0x1EA0U, 0x302U, 0x1EA1U,
    0x302U, 0x102U, 0x301U, 0x103U, 0x301U, 0x102U, 0x300U, 0x103U, 0x300U, 0x102U, 0x309U, 0x103U,
    0x309U, 0x102U, 0x303U, 0x103U, 0x303U, 0x1EA0U, 0x306U, 0x1EA1U, 0x306U, 0x45U, 0x323U, 0x65U,
    0x323U, 0x45U, 0x309U, 0x65U, 0x309U, 0x45U, 0x303U, 0x65U, 0x303U, 0xCAU, 0x301U, 0xEAU,
    0x301U, 0xCAU, 0x300U, 0xEAU, 0x300U, 0xCAU, 0x309U, 0xEAU, 0x309U, 0xCAU, 0x303U, 0xEAU,
    0x303U, 0x1EB8U, 0x302U, 0x1EB9U, 0x302U, 0x49U, 0x309U, 0x69U, 0x309U, 0x49U, 0x323U, 0x69U,
    0x323U, 0x4FU, 0x323U, 0x6FU, 0x323U, 0x4FU, 0x309U, 0x6FU, 0x309U, 0xD4U, 0x301U, 0xF4U,
    0x301U, 0xD4U, 0x300U, 0xF4U, 0x300U, 0xD4U, 0x309U, 0xF4U, 0x309U, 0xD4U, 0x303U, 0xF4U,
    0x303U, 0x1ECCU, 0x302U, 0x1ECDU, 0x302U, 0x1A0U, 0x301U, 0x1A1U, 0x301U, 0x1A0U, 0x300U, 0x1A1U,
    0x300U, 0x1A0U, 0x309U, 0x1A1U, 0x309U, 0x1A0U, 0x303U, 0x1A1U, 0x303U, 0x1A0U, 0x323U, 0x1A1U,
    0x323U, 0x55U, 0x323U, 0x75U, 0x323U, 0x55U, 0x309U, 0x75U, 0x309U, 0x1AFU, 0x301U, 0x1B0U,
    0x301U, 0x1AFU, 0x300U, 0x1B0U, 0x300U, 0x1AFU, 0x309U, 0x1B0U, 0x309U, 0x1AFU, 0x303U, 0x1B0U,
    0x303U, 0x1AFU, 0x323U, 0x1B0U, 0x323U, 0x59U, 0x300U, 0x79U, 0x300U, 0x59U, 0x323U, 0x79U,
    0x323U, 0x59U, 0x309U, 0x79U, 0x309U, 0x59U, 0x303U, 0x79U, 0x303U, 0x3B1U, 0x313U, 0x3B1U,
    0x314U, 0x1F00U, 0x300U, 0x1F01U, 0x300U, 0x1F00U, 0x301U, 0x1F01U, 0x301U, 0x1F00U, 0x342U, 0x1F01U,
    0x342U, 0x391U, 0x313U, 0x391U, 0x314U, 0x1F08U, 0x300U, 0x1F09U, 0x300U, 0x1F08U, 0x301U, 0x1F09U,
    0x301U, 0x1F08U, 0x342U, 0x1F09U, 0x342U, 0x3B5U, 0x313U, 0x3B5U, 0x314U, 0x1F10U, 0x300U, 0x1F11U,
    0x300U, 0x1F10U, 0x301U, 0x1F11U, 0x301U, 0x395U, 0x313U, 0x395U, 0x314U, 0x1F18U, 0x300U, 0x1F19U,
    0x300U, 0x1F18U, 0x301U, 0x1F19U, 0x301U, 0x3B7U, 0x313U, 0x3B7U, 0x314U, 0x1F20U, 0x300U, 0x1F21U,
    0x300U, 0x1F20U, 0x301U, 0x1F21U, 0x301U, 0x1F20U, 0x342U, 0x1F21U, 0x342U, 0x397U, 0x313U, 0x397U,
    0x314U, 0x1F28U, 0x300U, 0x1F29U, 0x300U, 0x1F28U, 0x301U, 0x1F29U, 0x301U, 0x1F28U, 0x342U, 0x1F29U,
    0x342U, 0x3B9U, 0x313U, 0x3B9U, 0x314U, 0x1F30U, 0x300U, 0x1F31U, 0x300U, 0x1F30U, 0x301U, 0x1F31U,
    0x301U, 0x1F30U, 0x342U, 0x1F31U, 0x342U, 0x399U, 0x313U, 0x399U, 0x314U, 0x1F38U, 0x300U, 0x1F39U,
    0x300U, 0x1F38U, 0x301U, 0x1F39U, 0x301U, 0x1F38U, 0x342U, 0x1F39U, 0x342U, 0x3BFU, 0x313U, 0x3BFU,
    0x314U, 0x1F40U, 0x300U, 0x1F41U, 0x300U, 0x1F40U, 0x301U, 0x1F41U, 0x301U, 0x39FU, 0x313U, 0x39FU,
    0x314U, 0x1F48U, 0x300U, 0x1F49U, 0x300U, 0x1F48U, 0x301U, 0x1F49U, 0x301U, 0x3C5U, 0x313U, 0x3C5U,
    0x314U, 0x1F50U, 0x300U, 0x1F51U, 0x300U, 0x1F50U, 0x301U, 0x1F51U, 0x301U, 0x1F50U, 0x342U, 0x1F51U,
    0x342U, 0x3A5U, 0x314U, 0x1F59U, 0x300U, 0x1F59U, 0x301U, 0x1F59U, 0x342U, 0x3C9U, 0x313U, 0x3C9U,
    0x314U, 0x1F60U, 0x300U, 0x1F61U, 0x300U, 0x1F60U, 0x301U, 0x1F61U, 0x301U, 0x1F60U, 0x342U, 0x1F61U,
    0x342U, 0x3A9U, 0x313U, 0x3A9U, 0x314U, 0x1F68U, 0x300U, 0x1F69U, 0x300U, 0x1F68U, 0x301U, 0x1F69U,
    0x301U, 0x1F68U, 0x342U, 0x1F69U, 0x342U, 0x3B1U, 0x300U, 0x3ACU, 0x3B5U, 0x300U, 0x3ADU, 0x3B7U,
    0x300U, 0x3AEU, 0x3B9U, 0x300U, 0x3AFU, 0x3BFU, 0x300U, 0x3CCU, 0x3C5U, 0x300U, 0x3CDU, 0x3C9U,
    0x300U, 0x3CEU, 0x1F00U, 0x345U, 0x1F01U, 0x345U, 0x1F02U, 0x345U, 0x1F03U, 0x345U, 0x1F04U, 0x345U,
    0x1F05U, 0x345U, 0x1F06U, 0x345U, 0x1F07U, 0x345U, 0x1F08U, 0x345U, 0x1F09U, 0x345U, 0x1F0AU, 0x345U,
    0x1F0BU, 0x345U, 0x1F0CU, 0x345U, 0x1F0DU, 0x345U, 0x1F0EU, 0x345U, 0x1F0FU, 0x345U, 0x1F20U, 0x345U,
    0x1F21U, 0x345U, 0x1F22U, 0x345U, 0x1F23U, 0x345U, 0x1F24U, 0x345U, 0x1F25U, 0x345U, 0x1F26U, 0x345U,
    0x1F27U, 0x345U, 0x1F28U, 0x345U, 0x1F29U, 0x345U, 0x1F2AU, 0x345U, 0x1F2BU, 0x345U, 0x1F2CU, 0x345U,
    0x1F2DU, 0x345U, 0x1F2EU, 0x345U, 0x1F2FU, 0x345U, 0x1F60U, 0x345U, 0x1F61U, 0x345U, 0x1F62U, 0x345U,
    0x1F63U, 0x345U, 0x1F64U, 0x345U, 0x1F65U, 0x345U, 0x1F66U, 0x345U, 0x1F67U, 0x345U, 0x1F68U, 0x345U,
    0x1F69U, 0x345U, 0x1F6AU, 0x345U, 0x1F6BU, 0x345U, 0x1F6CU, 0x345U, 0x1F6DU, 0x345U, 0x1F6EU, 0x345U,
    0x1F6FU, 0x345U, 0x3B1U, 0x306U, 0x3B1U, 0x304U, 0x1F70U, 0x345U, 0x3B1U, 0x345U, 0x3ACU, 0x345U,
    0x3B1U, 0x342U, 0x1FB6U, 0x345U, 0x391U, 0x306U, 0x391U, 0x304U, 0x391U, 0x300U, 0x386U, 0x391U,
    0x345U, 0x20U, 0x313U, 0x3B9U, 0x20U, 0x342U, 0xA8U, 0x342U, 0x1F74U, 0x345U, 0x3B7U, 0x345U,
    0x3AEU, 0x345U, 0x3B7U, 0x342U, 0x1FC6U, 0x345U, 0x395U, 0x300U, 0x388U, 0x397U, 0x300U, 0x389U,
    0x397U, 0x345U, 0x1FBFU, 0x300U, 0x1FBFU, 0x301U, 0x1FBFU, 0x342U, 0x3B9U, 0x306U, 0x3B9U, 0x304U,
    0x3CAU, 0x300U, 0x390U, 0x3B9U, 0x342U, 0x3CAU, 0x342U, 0x399U, 0x306U, 0x399U, 0x304U, 0x399U,
    0x300U, 0x38AU, 0x1FFEU, 0x300U, 0x1FFEU, 0x301U, 0x1FFEU, 0x342U, 0x3C5U, 0x306U, 0x3C5U, 0x304U,
    0x3CBU, 0x300U, 0x3B0U, 0x3C1U, 0x313U, 0x3C1U, 0x314U, 0x3C5U, 0x342U, 0x3CBU, 0x342U, 0x3A5U,
    0x306U, 0x3A5U, 0x304U, 0x3A5U, 0x300U, 0x38EU, 0x3A1U, 0x314U, 0xA8U, 0x300U, 0x385U, 0x60U,
    0x1F7CU, 0x345U, 0x3C9U, 0x345U, 0x3CEU, 0x345U, 0x3C9U, 0x342U, 0x1FF6U, 0x345U, 0x39FU, 0x300U,
    0x38CU, 0x3A9U, 0x300U, 0x38FU, 0x3A9U, 0x345U, 0xB4U, 0x20U, 0x314U, 0x2002U, 0x2003U, 0x2010U,
    0x20U, 0x333U, 0x2EU, 0x2EU, 0x2EU, 0x2EU, 0x2EU, 0x2EU, 0x2032U, 0x2032U, 0x2032U, 0x2032U,
    0x2032U, 0x2035U, 0x2035U, 0x2035U, 0x2035U, 0x2035U, 0x21U, 0x21U, 0x20U, 0x305U, 0x3FU, 0x3FU,
    0x3FU, 0x21U, 0x21U, 0x3FU, 0x2032U, 0x2032U, 0x2032U, 0x2032U, 0x30U, 0x34U, 0x35U, 0x36U,
    0x37U, 0x38U, 0x39U, 0x2BU, 0x2212U, 0x3DU, 0x28U, 0x29U, 0x6EU, 0x52U, 0x73U, 0x61U,
    0x2FU, 0x63U, 0x61U, 0x2FU, 0x73U, 0x43U, 0xB0U, 0x43U, 0x63U, 0x2FU, 0x6FU, 0x63U,
    0x2FU, 0x75U, 0x190U, 0xB0U, 0x46U, 0x127U, 0x4EU, 0x6FU, 0x51U, 0x53U, 0x4DU, 0x54U,
    0x45U, 0x4CU, 0x54U, 0x4DU, 0x5AU, 0x3A9U, 0xC5U, 0x46U, 0x5D0U, 0x5D1U, 0x5D2U, 0x5D3U,
    0x46U, 0x41U, 0x58U, 0x393U, 0x3A0U, 0x2211U, 0x31U, 0x2044U, 0x37U, 0x31U, 0x2044U, 0x39U,
    0x31U, 0x2044U, 0x31U, 0x30U, 0x31U, 0x2044U, 0x33U, 0x32U, 0x2044U, 0x33U, 0x31U, 0x2044U,
    0x35U, 0x32U, 0x2044U, 0x35U, 0x33U, 0x2044U, 0x35U, 0x34U, 0x2044U, 0x35U, 0x31U, 0x2044U,
    0x36U, 0x35U, 0x2044U, 0x36U, 0x31U, 0x2044U, 0x38U, 0x33U, 0x2044U, 0x38U, 0x35U, 0x2044U,
    0x38U, 0x37U, 0x2044U, 0x38U, 0x31U, 0x2044U, 0x49U, 0x49U, 0x49U, 0x49U, 0x49U, 0x49U,
    0x56U, 0x56U, 0x56U, 0x49U, 0x56U, 0x49U, 0x49U, 0x56U, 0x49U, 0x49U, 0x49U, 0x49U,
    0x58U, 0x58U, 0x58U, 0x49U, 0x58U, 0x49U, 0x49U, 0x69U, 0x69U, 0x69U, 0x69U, 0x69U,
    0x69U, 0x76U, 0x76U, 0x69U, 0x76U, 0x69U, 0x69U, 0x76U, 0x69U, 0x69U, 0x69U, 0x69U,
    0x78U, 0x78U, 0x69U, 0x78U, 0x69U, 0x69U, 0x30U, 0x2044U, 0x33U, 0x2190U, 0x338U, 0x2192U,
    0x338U, 0x2194U, 0x338U, 0x21D0U, 0x338U, 0x21D4U, 0x338U, 0x21D2U, 0x338U, 0x2203U, 0x338U, 0x2208U,
    0x338U, 0x220BU, 0x338U, 0x2223U, 0x338U, 0x2225U, 0x338U, 0x222BU, 0x222BU, 0x222BU, 0x222BU, 0x222BU,
    0x222EU, 0x222EU, 0x222EU, 0x222EU, 0x222EU, 0x223CU, 0x338U, 0x2243U, 0x338U, 0x2245U, 0x338U, 0x2248U,
    0x338U, 0x3DU, 0x338U, 0x2261U, 0x338U, 0x224DU, 0x338U, 0x3CU, 0x338U, 0x3EU, 0x338U, 0x2264U,
    0x338U, 0x2265U, 0x338U, 0x2272U, 0x338U, 0x2273U, 0x338U, 0x2276U, 0x338U, 0x2277U, 0x338U, 0x227AU,
    0x338U, 0x227BU, 0x338U, 0x2282U, 0x338U, 0x2283U, 0x338U, 0x2286U, 0x338U, 0x2287U, 0x338U, 0x22A2U,
    0x338U, 0x22A8U, 0x338U, 0x22A9U, 0x338U, 0x22ABU, 0x338U, 0x227CU, 0x338U, 0x227DU, 0x338U, 0x2291U,
    0x338U, 0x2292U, 0x338U, 0x22B2U, 0x338U, 0x22B3U, 0x338U, 0x22B4U, 0x338U, 0x22B5U, 0x338U, 0x3008U,
    0x3009U, 0x31U, 0x30U, 0x31U, 0x31U, 0x31U, 0x32U, 0x31U, 0x33U, 0x31U, 0x34U, 0x31U,
    0x35U, 0x31U, 0x36U, 0x31U, 0x37U, 0x31U, 0x38U, 0x31U, 0x39U, 0x32U, 0x30U, 0x28U,
    0x31U, 0x29U, 0x28U, 0x32U, 0x29U, 0x28U, 0x33U, 0x29U, 0x28U, 0x34U, 0x29U, 0x28U,
    0x35U, 0x29U, 0x28U, 0x36U, 0x29U, 0x28U, 0x37U, 0x29U, 0x28U, 0x38U, 0x29U, 0x28U,
    0x39U, 0x29U, 0x28U, 0x31U, 0x30U, 0x29U, 0x28U, 0x31U, 0x31U, 0x29U, 0x28U, 0x31U,
    0x32U, 0x29U, 0x28U, 0x31U, 0x33U, 0x29U, 0x28U, 0x31U, 0x34U, 0x29U, 0x28U, 0x31U,
    0x35U, 0x29U, 0x28U, 0x31U, 0x36U, 0x29U, 0x28U, 0x31U, 0x37U, 0x29U, 0x28U, 0x31U,
    0x38U, 0x29U, 0x28U, 0x31U, 0x39U, 0x29U, 0x28U, 0x32U, 0x30U, 0x29U, 0x31U, 0x2EU,
    0x32U, 0x2EU, 0x33U, 0x2EU, 0x34U, 0x2EU, 0x35U, 0x2EU, 0x36U, 0x2EU, 0x37U, 0x2EU,
    0x38U, 0x2EU, 0x39U, 0x2EU, 0x31U, 0x30U, 0x2EU, 0x31U, 0x31U, 0x2EU, 0x31U, 0x32U,
    0x2EU, 0x31U, 0x33U, 0x2EU, 0x31U, 0x34U, 0x2EU, 0x31U, 0x35U, 0x2EU, 0x31U, 0x36U,
    0x2EU, 0x31U, 0x37U, 0x2EU, 0x31U, 0x38U, 0x2EU, 0x31U, 0x39U, 0x2EU, 0x32U, 0x30U,
    0x2EU, 0x28U, 0x61U, 0x29U, 0x28U, 0x62U, 0x29U, 0x28U, 0x63U, 0x29U, 0x28U, 0x64U,
    0x29U, 0x28U, 0x65U, 0x29U, 0x28U, 0x66U, 0x29U, 0x28U, 0x67U, 0x29U, 0x28U, 0x68U,
    0x29U, 0x28U, 0x69U, 0x29U, 0x28U, 0x6AU, 0x29U, 0x28U, 0x6BU, 0x29U, 0x28U, 0x6CU,
    0x29U, 0x28U, 0x6DU, 0x29U, 0x28U, 0x6EU, 0x29U, 0x28U, 0x6FU, 0x29U, 0x28U, 0x70U,
    0x29U, 0x28U, 0x71U, 0x29U, 0x28U, 0x72U, 0x29U, 0x28U, 0x73U, 0x29U, 0x28U, 0x74U,
    0x29U, 0x28U, 0x75U, 0x29U, 0x28U, 0x76U, 0x29U, 0x28U, 0x77U, 0x29U, 0x28U, 0x78U,
    0x29U, 0x28U, 0x79U, 0x29U, 0x28U, 0x7AU, 0x29U, 0x53U, 0x59U, 0x71U, 0x222BU, 0x222BU,
    0x222BU, 0x222BU, 0x3AU, 0x3AU, 0x3DU, 0x3DU, 0x3DU, 0x3DU, 0x3DU, 0x3DU, 0x2ADDU, 0x338U,
    0x2D61U, 0x6BCDU, 0x9F9FU, 0x4E00U, 0x4E28U, 0x4E36U, 0x4E3FU, 0x4E59U, 0x4E85U, 0x4E8CU, 0x4EA0U, 0x4EBAU,
    0x513FU, 0x5165U, 0x516BU, 0x5182U, 0x5196U, 0x51ABU, 0x51E0U, 0x51F5U, 0x5200U, 0x529BU, 0x52F9U, 0x5315U,
    0x531AU, 0x5338U, 0x5341U, 0x535CU, 0x5369U, 0x5382U, 0x53B6U, 0x53C8U, 0x53E3U, 0x56D7U, 0x571FU, 0x58EBU,
    0x5902U, 0x590AU, 0x5915U, 0x5927U, 0x5973U, 0x5B50U, 0x5B80U, 0x5BF8U, 0x5C0FU, 0x5C22U, 0x5C38U, 0x5C6EU,
    0x5C71U, 0x5DDBU, 0x5DE5U, 0x5DF1U, 0x5DFEU, 0x5E72U, 0x5E7AU, 0x5E7FU, 0x5EF4U, 0x5EFEU, 0x5F0BU, 0x5F13U,
    0x5F50U, 0x5F61U, 0x5F73U, 0x5FC3U, 0x6208U, 0x6236U, 0x624BU, 0x652FU, 0x6534U, 0x6587U, 0x6597U, 0x65A4U,
    0x65B9U, 0x65E0U, 0x65E5U, 0x66F0U, 0x6708U, 0x6728U, 0x6B20U, 0x6B62U, 0x6B79U, 0x6BB3U, 0x6BCBU, 0x6BD4U,
    0x6BDBU, 0x6C0FU, 0x6C14U, 0x6C34U, 0x706BU, 0x722AU, 0x7236U, 0x723BU, 0x723FU, 0x7247U, 0x7259U, 0x725BU,
    0x72ACU, 0x7384U, 0x7389U, 0x74DCU, 0x74E6U, 0x7518U, 0x751FU, 0x7528U, 0x7530U, 0x758BU, 0x7592U, 0x7676U,
    0x767DU, 0x76AEU, 0x76BFU, 0x76EEU, 0x77DBU, 0x77E2U, 0x77F3U, 0x793AU, 0x79B8U, 0x79BEU, 0x7A74U, 0x7ACBU,
    0x7AF9U, 0x7C73U, 0x7CF8U, 0x7F36U, 0x7F51U, 0x7F8AU, 0x7FBDU, 0x8001U, 0x800CU, 0x8012U, 0x8033U, 0x807FU,
    0x8089U, 0x81E3U, 0x81EAU, 0x81F3U, 0x81FCU, 0x820CU, 0x821BU, 0x821FU, 0x826EU, 0x8272U, 0x8278U, 0x864DU,
    0x866BU, 0x8840U, 0x884CU, 0x8863U, 0x897EU, 0x898BU, 0x89D2U, 0x8A00U, 0x8C37U, 0x8C46U, 0x8C55U, 0x8C78U,
    0x8C9DU, 0x8D64U, 0x8D70U, 0x8DB3U, 0x8EABU, 0x8ECAU, 0x8F9BU, 0x8FB0U, 0x8FB5U, 0x9091U, 0x9149U, 0x91C6U,
    0x91CCU, 0x91D1U, 0x9577U, 0x9580U, 0x961CU, 0x96B6U, 0x96B9U, 0x96E8U, 0x9751U, 0x975EU, 0x9762U, 0x9769U,
    0x97CBU, 0x97EDU, 0x97F3U, 0x9801U, 0x98A8U, 0x98DBU, 0x98DFU, 0x9996U, 0x9999U, 0x99ACU, 0x9AA8U, 0x9AD8U,
    0x9ADFU, 0x9B25U, 0x9B2FU, 0x9B32U, 0x9B3CU, 0x9B5AU, 0x9CE5U, 0x9E75U, 0x9E7FU, 0x9EA5U, 0x9EBBU, 0x9EC3U,
    0x9ECDU, 0x9ED1U, 0x9EF9U, 0x9EFDU, 0x9F0EU, 0x9F13U, 0x9F20U, 0x9F3BU, 0x9F4AU, 0x9F52U, 0x9F8DU, 0x9F9CU,
    0x9FA0U, 0x3012U, 0x5344U, 0x5345U, 0x304BU, 0x3099U, 0x304DU, 0x3099U, 0x304FU, 0x3099U, 0x3051U, 0x3099U,
    0x3053U, 0x3099U, 0x3055U, 0x3099U, 0x3057U, 0x3099U, 0x3059U, 0x3099U, 0x305BU, 0x3099U, 0x305DU, 0x3099U,
    0x305FU, 0x3099U, 0x3061U, 0x3099U, 0x3064U, 0x3099U, 0x3066U, 0x3099U, 0x3068U, 0x3099U, 0x306FU, 0x3099U,
    0x306FU, 0x309AU, 0x3072U, 0x3099U, 0x3072U, 0x309AU, 0x3075U, 0x3099U, 0x3075U, 0x309AU, 0x3078U, 0x3099U,
    0x3078U, 0x309AU, 0x307BU, 0x3099U, 0x307BU, 0x309AU, 0x3046U, 0x3099U, 0x20U, 0x3099U, 0x20U, 0x309AU,
    0x309DU, 0x3099U, 0x3088U, 0x308AU, 0x30ABU, 0x3099U, 0x30ADU, 0x3099U, 0x30AFU, 0x3099U, 0x30B1U, 0x3099U,
    0x30B3U, 0x3099U, 0x30B5U, 0x3099U, 0x30B7U, 0x3099U, 0x30B9U, 0x3099U, 0x30BBU, 0x3099U, 0x30BDU, 0x3099U,
    0x30BFU, 0x3099U, 0x30C1U, 0x3099U, 0x30C4U, 0x3099U, 0x30C6U, 0x3099U, 0x30C8U, 0x3099U, 0x30CFU, 0x3099U,
    0x30CFU, 0x309AU, 0x30D2U, 0x3099U, 0x30D2U, 0x309AU, 0x30D5U, 0x3099U, 0x30D5U, 0x309AU, 0x30D8U, 0x3099U,
    0x30D8U, 0x309AU, 0x30DBU, 0x3099U, 0x30DBU, 0x309AU, 0x30A6U, 0x3099U, 0x30EFU, 0x3099U, 0x30F0U, 0x3099U,
    0x30F1U, 0x3099U, 0x30F2U, 0x3099U, 0x30FDU, 0x3099U, 0x30B3U, 0x30C8U, 0x1100U, 0x1101U, 0x11AAU, 0x1102U,
    0x11ACU, 0x11ADU, 0x1103U, 0x1104U, 0x1105U, 0x11B0U, 0x11B1U, 0x11B2U, 0x11B3U, 0x11B4U, 0x11B5U, 0x111AU,
    0x1106U, 0x1107U, 0x1108U, 0x1121U, 0x1109U, 0x110AU, 0x110BU, 0x110CU, 0x110DU, 0x110EU, 0x110FU, 0x1110U,
    0x1111U, 0x1112U, 0x1161U, 0x1162U, 0x1163U, 0x1164U, 0x1165U, 0x1166U, 0x1167U, 0x1168U, 0x1169U, 0x116AU,
    0x116BU, 0x116CU, 0x116DU, 0x116EU, 0x116FU, 0x1170U, 0x1171U, 0x1172U, 0x1173U, 0x1174U, 0x1175U, 0x1160U,
    0x1114U, 0x1115U, 0x11C7U, 0x11C8U, 0x11CCU, 0x11CEU, 0x11D3U, 0x11D7U, 0x11D9U, 0x111CU, 0x11DDU, 0x11DFU,
    0x111DU, 0x111EU, 0x1120U, 0x1122U, 0x1123U, 0x1127U, 0x1129U, 0x112BU, 0x112CU, 0x112DU, 0x112EU, 0x112FU,
    0x1132U, 0x1136U, 0x1140U, 0x1147U, 0x114CU, 0x11F1U, 0x11F2U, 0x1157U, 0x1158U, 0x1159U, 0x1184U, 0x1185U,
    0x1188U, 0x1191U, 0x1192U, 0x1194U, 0x119EU, 0x11A1U, 0x4E09U, 0x56DBU, 0x4E0AU, 0x4E2DU, 0x4E0BU, 0x7532U,
    0x4E19U, 0x4E01U, 0x5929U, 0x5730U, 0x28U, 0x1100U, 0x29U, 0x28U, 0x1102U, 0x29U, 0x28U, 0x1103U,
    0x29U, 0x28U, 0x1105U, 0x29U, 0x28U, 0x1106U, 0x29U, 0x28U, 0x1107U, 0x29U, 0x28U, 0x1109U,
    0x29U, 0x28U, 0x110BU, 0x29U, 0x28U, 0x110CU, 0x29U, 0x28U, 0x110EU, 0x29U, 0x28U, 0x110FU,
    0x29U, 0x28U, 0x1110U, 0x29U, 0x28U, 0x1111U, 0x29U, 0x28U, 0x1112U, 0x29U, 0x28U, 0x1100U,
    0x1161U, 0x29U, 0x28U, 0x1102U, 0x1161U, 0x29U, 0x28U, 0x1103U, 0x1161U, 0x29U, 0x28U, 0x1105U,
    0x1161U, 0x29U, 0x28U, 0x1106U, 0x1161U, 0x29U, 0x28U, 0x1107U, 0x1161U, 0x29U, 0x28U, 0x1109U,
    0x1161U, 0x29U, 0x28U, 0x110BU, 0x1161U, 0x29U, 0x28U, 0x110CU, 0x1161U, 0x29U, 0x28U, 0x110EU,
    0x1161U, 0x29U, 0x28U, 0x110FU, 0x1161U, 0x29U, 0x28U, 0x1110U, 0x1161U, 0x29U, 0x28U, 0x1111U,
    0x1161U, 0x29U, 0x28U, 0x1112U, 0x1161U, 0x29U, 0x28U, 0x110CU, 0x116EU, 0x29U, 0x28U, 0x110BU,
    0x1169U, 0x110CU, 0x1165U, 0x11ABU, 0x29U, 0x28U, 0x110BU, 0x1169U, 0x1112U, 0x116EU, 0x29U, 0x28U,
    0x4E00U, 0x29U, 0x28U, 0x4E8CU, 0x29U, 0x28U, 0x4E09U, 0x29U, 0x28U, 0x56DBU, 0x29U, 0x28U,
    0x4E94U, 0x29U, 0x28U, 0x516DU, 0x29U, 0x28U, 0x4E03U, 0x29U, 0x28U, 0x516BU, 0x29U, 0x28U,
    0x4E5DU, 0x29U, 0x28U, 0x5341U, 0x29U, 0x28U, 0x6708U, 0x29U, 0x28U, 0x706BU, 0x29U, 0x28U,
    0x6C34U, 0x29U, 0x28U, 0x6728U, 0x29U, 0x28U, 0x91D1U, 0x29U, 0x28U, 0x571FU, 0x29U, 0x28U,
    0x65E5U, 0x29U, 0x28U, 0x682AU, 0x29U, 0x28U, 0x6709U, 0x29U, 0x28U, 0x793EU, 0x29U, 0x28U,
    0x540DU, 0x29U, 0x28U, 0x7279U, 0x29U, 0x28U, 0x8CA1U, 0x29U, 0x28U, 0x795DU, 0x29U, 0x28U,
    0x52B4U, 0x29U, 0x28U, 0x4EE3U, 0x29U, 0x28U, 0x547CU, 0x29U, 0x28U, 0x5B66U, 0x29U, 0x28U,
    0x76E3U, 0x29U, 0x28U, 0x4F01U, 0x29U, 0x28U, 0x8CC7U, 0x29U, 0x28U, 0x5354U, 0x29U, 0x28U,
    0x796DU, 0x29U, 0x28U, 0x4F11U, 0x29U, 0x28U, 0x81EAU, 0x29U, 0x28U, 0x81F3U, 0x29U, 0x554FU,
    0x5E7CU, 0x7B8FU, 0x50U, 0x54U, 0x45U, 0x32U, 0x31U, 0x32U, 0x32U, 0x32U, 0x33U, 0x32U,
    0x34U, 0x32U, 0x35U, 0x32U, 0x36U, 0x32U, 0x37U, 0x32U, 0x38U, 0x32U, 0x39U, 0x33U,
    0x30U, 0x33U, 0x31U, 0x33U, 0x32U, 0x33U, 0x33U, 0x33U, 0x34U, 0x33U, 0x35U, 0x1100U,
    0x1161U, 0x1102U, 0x1161U, 0x1103U, 0x1161U, 0x1105U, 0x1161U, 0x1106U, 0x1161U, 0x1107U, 0x1161U, 0x1109U,
    0x1161U, 0x110BU, 0x1161U, 0x110CU, 0x1161U, 0x110EU, 0x1161U, 0x110FU, 0x1161U, 0x1110U, 0x1161U, 0x1111U,
    0x1161U, 0x1112U, 0x1161U, 0x110EU, 0x1161U, 0x11B7U, 0x1100U, 0x1169U, 0x110CU, 0x116EU, 0x110BU, 0x1174U,
    0x110BU, 0x116EU, 0x4E94U, 0x516DU, 0x4E03U, 0x4E5DU, 0x682AU, 0x6709U, 0x793EU, 0x540DU, 0x7279U, 0x8CA1U,
    0x795DU, 0x52B4U, 0x79D8U, 0x7537U, 0x9069U, 0x512AU, 0x5370U, 0x6CE8U, 0x9805U, 0x4F11U, 0x5199U, 0x6B63U,
    0x5DE6U, 0x53F3U, 0x533BU, 0x5B97U, 0x5B66U, 0x76E3U, 0x4F01U, 0x8CC7U, 0x5354U, 0x591CU, 0x33U, 0x36U,
    0x33U, 0x37U, 0x33U, 0x38U, 0x33U, 0x39U, 0x34U, 0x30U, 0x34U, 0x31U, 0x34U, 0x32U,
    0x34U, 0x33U, 0x34U, 0x34U, 0x34U, 0x35U, 0x34U, 0x36U, 0x34U, 0x37U, 0x34U, 0x38U,
    0x34U, 0x39U, 0x35U, 0x30U, 0x31U, 0x6708U, 0x32U, 0x6708U, 0x33U, 0x6708U, 0x34U, 0x6708U,
    0x35U, 0x6708U, 0x36U, 0x6708U, 0x37U, 0x6708U, 0x38U, 0x6708U, 0x39U, 0x6708U, 0x31U, 0x30U,
    0x6708U, 0x31U, 0x31U, 0x6708U, 0x31U, 0x32U, 0x6708U, 0x48U, 0x67U, 0x65U, 0x72U, 0x67U,
    0x65U, 0x56U, 0x4CU, 0x54U, 0x44U, 0x30A2U, 0x30A4U, 0x30A6U, 0x30A8U, 0x30AAU, 0x30ABU, 0x30ADU,
    0x30AFU, 0x30B1U, 0x30B3U, 0x30B5U, 0x30B7U, 0x30B9U, 0x30BBU, 0x30BDU, 0x30BFU, 0x30C1U, 0x30C4U, 0x30C6U,
    0x30C8U, 0x30CAU, 0x30CBU, 0x30CCU, 0x30CDU, 0x30CEU, 0x30CFU, 0x30D2U, 0x30D5U, 0x30D8U, 0x30DBU, 0x30DEU,
    0x30DFU, 0x30E0U, 0x30E1U, 0x30E2U, 0x30E4U, 0x30E6U, 0x30E8U, 0x30E9U, 0x30EAU, 0x30EBU, 0x30ECU, 0x30EDU,
    0x30EFU, 0x30F0U, 0x30F1U, 0x30F2U, 0x4EE4U, 0x548CU, 0x30A2U, 0x30D1U, 0x30FCU, 0x30C8U, 0x30A2U, 0x30EBU,
    0x30D5U, 0x30A1U, 0x30A2U, 0x30F3U, 0x30DAU, 0x30A2U, 0x30A2U, 0x30FCU, 0x30EBU, 0x30A4U, 0x30CBU, 0x30F3U,
    0x30B0U, 0x30A4U, 0x30F3U, 0x30C1U, 0x30A6U, 0x30A9U, 0x30F3U, 0x30A8U, 0x30B9U, 0x30AFU, 0x30FCU, 0x30C9U,
    0x30A8U, 0x30FCU, 0x30ABU, 0x30FCU, 0x30AAU, 0x30F3U, 0x30B9U, 0x30AAU, 0x30FCU, 0x30E0U, 0x30ABU, 0x30A4U,
    0x30EAU, 0x30ABU, 0x30E9U, 0x30C3U, 0x30C8U, 0x30ABU, 0x30EDU, 0x30EAU, 0x30FCU, 0x30ACU, 0x30EDU, 0x30F3U,
    0x30ACU, 0x30F3U, 0x30DEU, 0x30AEU, 0x30ACU, 0x30AEU, 0x30CBU, 0x30FCU, 0x30ADU, 0x30E5U, 0x30EAU, 0x30FCU,
    0x30AEU, 0x30EBU, 0x30C0U, 0x30FCU, 0x30ADU, 0x30EDU, 0x30ADU, 0x30EDU, 0x30B0U, 0x30E9U, 0x30E0U, 0x30ADU,
    0x30EDU, 0x30E1U, 0x30FCU, 0x30C8U, 0x30EBU, 0x30ADU, 0x30EDU, 0x30EFU, 0x30C3U, 0x30C8U, 0x30B0U, 0x30E9U,
    0x30E0U, 0x30B0U, 0x30E9U, 0x30E0U, 0x30C8U, 0x30F3U, 0x30AFU, 0x30EBU, 0x30BCU, 0x30A4U, 0x30EDU, 0x30AFU,
    0x30EDU, 0x30FCU, 0x30CDU, 0x30B1U, 0x30FCU, 0x30B9U, 0x30B3U, 0x30EBU, 0x30CAU, 0x30B3U, 0x30FCU, 0x30DDU,
    0x30B5U, 0x30A4U, 0x30AFU, 0x30EBU, 0x30B5U, 0x30F3U, 0x30C1U, 0x30FCU, 0x30E0U, 0x30B7U, 0x30EAU, 0x30F3U,
    0x30B0U, 0x30BBU, 0x30F3U, 0x30C1U, 0x30BBU, 0x30F3U, 0x30C8U, 0x30C0U, 0x30FCU, 0x30B9U, 0x30C7U, 0x30B7U,
    0x30C9U, 0x30EBU, 0x30C8U, 0x30F3U, 0x30CAU, 0x30CEU, 0x30CEU, 0x30C3U, 0x30C8U, 0x30CFU, 0x30A4U, 0x30C4U,
    0x30D1U, 0x30FCU, 0x30BBU, 0x30F3U, 0x30C8U, 0x30D1U, 0x30FCU, 0x30C4U, 0x30D0U, 0x30FCU, 0x30ECU, 0x30EBU,
    0x30D4U, 0x30A2U, 0x30B9U, 0x30C8U, 0x30EBU, 0x30D4U, 0x30AFU, 0x30EBU, 0x30D4U, 0x30B3U, 0x30D3U, 0x30EBU,
    0x30D5U, 0x30A1U, 0x30E9U, 0x30C3U, 0x30C9U, 0x30D5U, 0x30A3U, 0x30FCU, 0x30C8U, 0x30D6U, 0x30C3U, 0x30B7U,
    0x30A7U, 0x30EBU, 0x30D5U, 0x30E9U, 0x30F3U, 0x30D8U, 0x30AFU, 0x30BFU, 0x30FCU, 0x30EBU, 0x30DAU, 0x30BDU,
    0x30DAU, 0x30CBU, 0x30D2U, 0x30D8U, 0x30EBU, 0x30C4U, 0x30DAU, 0x30F3U, 0x30B9U, 0x30DAU, 0x30FCU, 0x30B8U,
    0x30D9U, 0x30FCU, 0x30BFU, 0x30DDU, 0x30A4U, 0x30F3U, 0x30C8U, 0x30DCU, 0x30EBU, 0x30C8U, 0x30DBU, 0x30F3U,
    0x30DDU, 0x30F3U, 0x30C9U, 0x30DBU, 0x30FCU, 0x30EBU, 0x30DBU, 0x30FCU, 0x30F3U, 0x30DEU, 0x30A4U, 0x30AFU,
    0x30EDU, 0x30DEU, 0x30A4U, 0x30EBU, 0x30DEU, 0x30C3U, 0x30CFU, 0x30DEU, 0x30EBU, 0x30AFU, 0x30DEU, 0x30F3U,
    0x30B7U, 0x30E7U, 0x30F3U, 0x30DFU, 0x30AFU, 0x30EDU, 0x30F3U, 0x30DFU, 0x30EAU, 0x30DFU, 0x30EAU, 0x30D0U,
    0x30FCU, 0x30EBU, 0x30E1U, 0x30ACU, 0x30E1U, 0x30ACU, 0x30C8U, 0x30F3U, 0x30E1U, 0x30FCU, 0x30C8U, 0x30EBU,
    0x30E4U, 0x30FCU, 0x30C9U, 0x30E4U, 0x30FCU, 0x30EBU, 0x30E6U, 0x30A2U, 0x30F3U, 0x30EAU, 0x30C3U, 0x30C8U,
    0x30EBU, 0x30EAU, 0x30E9U, 0x30EBU, 0x30D4U, 0x30FCU, 0x30EBU, 0x30FCU, 0x30D6U, 0x30EBU, 0x30ECU, 0x30E0U,
    0x30ECU, 0x30F3U, 0x30C8U, 0x30B2U, 0x30F3U, 0x30EFU, 0x30C3U, 0x30C8U, 0x30U, 0x70B9U, 0x31U, 0x70B9U,
    0x32U, 0x70B9U, 0x33U, 0x70B9U, 0x34U, 0x70B9U, 0x35U, 0x70B9U, 0x36U, 0x70B9U, 0x37U, 0x70B9U,
    0x38U, 0x70B9U, 0x39U, 0x70B9U, 0x31U, 0x30U, 0x70B9U, 0x31U, 0x31U, 0x70B9U, 0x31U, 0x32U,
    0x70B9U, 0x31U, 0x33U, 0x70B9U, 0x31U, 0x34U, 0x70B9U, 0x31U, 0x35U, 0x70B9U, 0x31U, 0x36U,
    0x70B9U, 0x31U, 0x37U, 0x70B9U, 0x31U, 0x38U, 0x70B9U, 0x31U, 0x39U, 0x70B9U, 0x32U, 0x30U,
    0x70B9U, 0x32U, 0x31U, 0x70B9U, 0x32U, 0x32U, 0x70B9U, 0x32U, 0x33U, 0x70B9U, 0x32U, 0x34U,
    0x70B9U, 0x68U, 0x50U, 0x61U, 0x64U, 0x61U, 0x41U, 0x55U, 0x62U, 0x61U, 0x72U, 0x6FU,
    0x56U, 0x70U, 0x63U, 0x64U, 0x6DU, 0x64U, 0x6DU, 0xB2U, 0x64U, 0x6DU, 0xB3U, 0x49U,
    0x55U, 0x5E73U, 0x6210U, 0x662DU, 0x548CU, 0x5927U, 0x6B63U, 0x660EU, 0x6CBBU, 0x682AU, 0x5F0FU, 0x4F1AU,
    0x793EU, 0x70U, 0x41U, 0x6EU, 0x41U, 0x3BCU, 0x41U, 0x6DU, 0x41U, 0x6BU, 0x41U, 0x4BU,
    0x42U, 0x4DU, 0x42U, 0x47U, 0x42U, 0x63U, 0x61U, 0x6CU, 0x6BU, 0x63U, 0x61U, 0x6CU,
    0x70U, 0x46U, 0x6EU, 0x46U, 0x3BCU, 0x46U, 0x3BCU, 0x67U, 0x6DU, 0x67U, 0x6BU, 0x67U,
    0x48U, 0x7AU, 0x6BU, 0x48U, 0x7AU, 0x4DU, 0x48U, 0x7AU, 0x47U, 0x48U, 0x7AU, 0x54U,
    0x48U, 0x7AU, 0x3BCU, 0x2113U, 0x6DU, 0x2113U, 0x64U, 0x2113U, 0x6BU, 0x2113U, 0x66U, 0x6DU,
    0x6EU, 0x6DU, 0x3BCU, 0x6DU, 0x6DU, 0x6DU, 0x63U, 0x6DU, 0x6BU, 0x6DU, 0x6DU, 0x6DU,
    0xB2U, 0x63U, 0x6DU, 0xB2U, 0x6DU, 0xB2U, 0x6BU, 0x6DU, 0xB2U, 0x6DU, 0x6DU, 0xB3U,
    0x63U, 0x6DU, 0xB3U, 0x6DU, 0xB3U, 0x6BU, 0x6DU, 0xB3U, 0x6DU, 0x2215U, 0x73U, 0x6DU,
    0x2215U, 0x73U, 0xB2U, 0x50U, 0x61U, 0x6BU, 0x50U, 0x61U, 0x4DU, 0x50U, 0x61U, 0x47U,
    0x50U, 0x61U, 0x72U, 0x61U, 0x64U, 0x72U, 0x61U, 0x64U, 0x2215U, 0x73U, 0x72U, 0x61U,
    0x64U, 0x2215U, 0x73U, 0xB2U, 0x70U, 0x73U, 0x6EU, 0x73U, 0x3BCU, 0x73U, 0x6DU, 0x73U,
    0x70U, 0x56U, 0x6EU, 0x56U, 0x3BCU, 0x56U, 0x6DU, 0x56U, 0x6BU, 0x56U, 0x4DU, 0x56U,
    0x70U, 0x57U, 0x6EU, 0x57U, 0x3BCU, 0x57U, 0x6DU, 0x57U, 0x6BU, 0x57U, 0x4DU, 0x57U,
    0x6BU, 0x3A9U, 0x4DU, 0x3A9U, 0x61U, 0x2EU, 0x6DU, 0x2EU, 0x42U, 0x71U, 0x63U, 0x63U,
    0x63U, 0x64U, 0x43U, 0x2215U, 0x6BU, 0x67U, 0x43U, 0x6FU, 0x2EU, 0x64U, 0x42U, 0x47U,
    0x79U, 0x68U, 0x61U, 0x48U, 0x50U, 0x69U, 0x6EU, 0x4BU, 0x4BU, 0x4BU, 0x4DU, 0x6BU,
    0x74U, 0x6CU, 0x6DU, 0x6CU, 0x6EU, 0x6CU, 0x6FU, 0x67U, 0x6CU, 0x78U, 0x6DU, 0x62U,
    0x6DU, 0x69U, 0x6CU, 0x6DU, 0x6FU, 0x6CU, 0x50U, 0x48U, 0x70U, 0x2EU, 0x6DU, 0x2EU,
    0x50U, 0x50U, 0x4DU, 0x50U, 0x52U, 0x73U, 0x72U, 0x53U, 0x76U, 0x57U, 0x62U, 0x56U,
    0x2215U, 0x6DU, 0x41U, 0x2215U, 0x6DU, 0x31U, 0x65E5U, 0x32U, 0x65E5U, 0x33U, 0x65E5U, 0x34U,
    0x65E5U, 0x35U, 0x65E5U, 0x36U, 0x65E5U, 0x37U, 0x65E5U, 0x38U, 0x65E5U, 0x39U, 0x65E5U, 0x31U,
    0x30U, 0x65E5U, 0x31U, 0x31U, 0x65E5U, 0x31U, 0x32U, 0x65E5U, 0x31U, 0x33U, 0x65E5U, 0x31U,
    0x34U, 0x65E5U, 0x31U, 0x35U, 0x65E5U, 0x31U, 0x36U, 0x65E5U, 0x31U, 0x37U, 0x65E5U, 0x31U,
    0x38U, 0x65E5U, 0x31U, 0x39U, 0x65E5U, 0x32U, 0x30U, 0x65E5U, 0x32U, 0x31U, 0x65E5U, 0x32U,
    0x32U, 0x65E5U, 0x32U, 0x33U, 0x65E5U, 0x32U, 0x34U, 0x65E5U, 0x32U, 0x35U, 0x65E5U, 0x32U,
    0x36U, 0x65E5U, 0x32U, 0x37U, 0x65E5U, 0x32U, 0x38U, 0x65E5U, 0x32U, 0x39U, 0x65E5U, 0x33U,
    0x30U, 0x65E5U, 0x33U, 0x31U, 0x65E5U, 0x67U, 0x61U, 0x6CU, 0x44AU, 0x44CU, 0xA76FU, 0x126U,
    0x153U, 0xA727U, 0xAB37U, 0x26BU, 0xAB52U, 0x28DU, 0x8C48U, 0x66F4U, 0x8CC8U, 0x6ED1U, 0x4E32U, 0x53E5U,
    0x5951U, 0x5587U, 0x5948U, 0x61F6U, 0x7669U, 0x7F85U, 0x863FU, 0x87BAU, 0x88F8U, 0x908FU, 0x6A02U, 0x6D1BU,
    0x70D9U, 0x73DEU, 0x843DU, 0x916AU, 0x99F1U, 0x4E82U, 0x5375U, 0x6B04U, 0x721BU, 0x862DU, 0x9E1EU, 0x5D50U,
    0x6FEBU, 0x85CDU, 0x8964U, 0x62C9U, 0x81D8U, 0x881FU, 0x5ECAU, 0x6717U, 0x6D6AU, 0x72FCU, 0x90CEU, 0x4F86U,
    0x51B7U, 0x52DEU, 0x64C4U, 0x6AD3U, 0x7210U, 0x76E7U, 0x8606U, 0x865CU, 0x8DEFU, 0x9732U, 0x9B6FU, 0x9DFAU,
    0x788CU, 0x797FU, 0x7DA0U, 0x83C9U, 0x9304U, 0x8AD6U, 0x58DFU, 0x5F04U, 0x7C60U, 0x807EU, 0x7262U, 0x78CAU,
    0x8CC2U, 0x96F7U, 0x58D8U, 0x5C62U, 0x6A13U, 0x6DDAU, 0x6F0FU, 0x7D2FU, 0x7E37U, 0x964BU, 0x52D2U, 0x808BU,
    0x51DCU, 0x51CCU, 0x7A1CU, 0x7DBEU, 0x83F1U, 0x9675U, 0x8B80U, 0x62CFU, 0x8AFEU, 0x4E39U, 0x5BE7U, 0x6012U,
    0x7387U, 0x7570U, 0x5317U, 0x78FBU, 0x4FBFU, 0x5FA9U, 0x4E0DU, 0x6CCCU, 0x6578U, 0x7D22U, 0x53C3U, 0x585EU,
    0x7701U, 0x8449U, 0x8AAAU, 0x6BBAU, 0x6C88U, 0x62FEU, 0x82E5U, 0x63A0U, 0x7565U, 0x4EAEU, 0x5169U, 0x51C9U,
    0x6881U, 0x7CE7U, 0x826FU, 0x8AD2U, 0x91CFU, 0x52F5U, 0x5442U, 0x5EECU, 0x65C5U, 0x6FFEU, 0x792AU, 0x95ADU,
    0x9A6AU, 0x9E97U, 0x9ECEU, 0x66C6U, 0x6B77U, 0x8F62U, 0x5E74U, 0x6190U, 0x6200U, 0x649AU, 0x6F23U, 0x7149U,
    0x7489U, 0x79CAU, 0x7DF4U, 0x806FU, 0x8F26U, 0x84EEU, 0x9023U, 0x934AU, 0x5217U, 0x52A3U, 0x54BDU, 0x70C8U,
    0x88C2U, 0x5EC9U, 0x5FF5U, 0x637BU, 0x6BAEU, 0x7C3EU, 0x7375U, 0x4EE4U, 0x56F9U, 0x5DBAU, 0x601CU, 0x73B2U,
    0x7469U, 0x7F9AU, 0x8046U, 0x9234U, 0x96F6U, 0x9748U, 0x9818U, 0x4F8BU, 0x79AEU, 0x91B4U, 0x96B8U, 0x60E1U,
    0x4E86U, 0x50DAU, 0x5BEEU, 0x5C3FU, 0x6599U, 0x71CEU, 0x7642U, 0x84FCU, 0x907CU, 0x6688U, 0x962EU, 0x5289U,
    0x677BU, 0x67F3U, 0x6D41U, 0x6E9CU, 0x7409U, 0x7559U, 0x786BU, 0x7D10U, 0x985EU, 0x622EU, 0x9678U, 0x502BU,
    0x5D19U, 0x6DEAU, 0x8F2AU, 0x5F8BU, 0x6144U, 0x6817U, 0x9686U, 0x5229U, 0x540FU, 0x5C65U, 0x6613U, 0x674EU,
    0x68A8U, 0x6CE5U, 0x7406U, 0x75E2U, 0x7F79U, 0x88CFU, 0x88E1U, 0x96E2U, 0x533FU, 0x6EBAU, 0x541DU, 0x71D0U,
    0x7498U, 0x85FAU, 0x96A3U, 0x9C57U, 0x9E9FU, 0x6797U, 0x6DCBU, 0x81E8U, 0x7B20U, 0x7C92U, 0x72C0U, 0x7099U,
    0x8B58U, 0x4EC0U, 0x8336U, 0x523AU, 0x5207U, 0x5EA6U, 0x62D3U, 0x7CD6U, 0x5B85U, 0x6D1EU, 0x66B4U, 0x8F3BU,
    0x964DU, 0x5ED3U, 0x5140U, 0x55C0U, 0x585AU, 0x6674U, 0x51DEU, 0x732AU, 0x76CAU, 0x793CU, 0x795EU, 0x7965U,
    0x798FU, 0x9756U, 0x7CBEU, 0x8612U, 0x8AF8U, 0x9038U, 0x90FDU, 0x98EFU, 0x98FCU, 0x9928U, 0x9DB4U, 0x90DEU,
    0x96B7U, 0x4FAEU, 0x50E7U, 0x514DU, 0x52C9U, 0x52E4U, 0x5351U, 0x559DU, 0x5606U, 0x5668U, 0x5840U, 0x58A8U,
    0x5C64U, 0x6094U, 0x6168U, 0x618EU, 0x61F2U, 0x654FU, 0x65E2U, 0x6691U, 0x6885U, 0x6D77U, 0x6E1AU, 0x6F22U,
    0x716EU, 0x722BU, 0x7422U, 0x7891U, 0x7949U, 0x7948U, 0x7950U, 0x7956U, 0x798DU, 0x798EU, 0x7A40U, 0x7A81U,
    0x7BC0U, 0x7E09U, 0x7E41U, 0x7F72U, 0x8005U, 0x81EDU, 0x8279U, 0x8457U, 0x8910U, 0x8996U, 0x8B01U, 0x8B39U,
    0x8CD3U, 0x8D08U, 0x8FB6U, 0x96E3U, 0x97FFU, 0x983BU, 0x6075U, 0x242EEU, 0x8218U, 0x4E26U, 0x51B5U, 0x5168U,
    0x4F80U, 0x5145U, 0x5180U, 0x52C7U, 0x52FAU, 0x5555U, 0x5599U, 0x55E2U, 0x58B3U, 0x5944U, 0x5954U, 0x5A62U,
    0x5B28U, 0x5ED2U, 0x5ED9U, 0x5F69U, 0x5FADU, 0x60D8U, 0x614EU, 0x6108U, 0x6160U, 0x6234U, 0x63C4U, 0x641CU,
    0x6452U, 0x6556U, 0x671BU, 0x6756U, 0x6EDBU, 0x6ECBU, 0x701EU, 0x77A7U, 0x7235U, 0x72AFU, 0x7471U, 0x7506U,
    0x753BU, 0x761DU, 0x761FU, 0x76DBU, 0x76F4U, 0x774AU, 0x7740U, 0x78CCU, 0x7AB1U, 0x7C7BU, 0x7D5BU, 0x7F3EU,
    0x8352U, 0x83EFU, 0x8779U, 0x8941U, 0x8986U, 0x8ABFU, 0x8ACBU, 0x8AEDU, 0x8B8AU, 0x8F38U, 0x9072U, 0x9199U,
    0x9276U, 0x967CU, 0x97DBU, 0x980BU, 0x9B12U, 0x2284AU, 0x22844U, 0x233D5U, 0x3B9DU, 0x4018U, 0x4039U, 0x25249U,
    0x25CD0U, 0x27ED3U, 0x9F43U, 0x9F8EU, 0x66U, 0x66U, 0x66U, 0x69U, 0x66U, 0x6CU, 0x66U, 0x66U,
    0x69U, 0x66U, 0x66U, 0x6CU, 0x17FU, 0x74U, 0x73U, 0x74U, 0x574U, 0x576U, 0x574U, 0x565U,
    0x574U, 0x56BU, 0x57EU, 0x576U, 0x574U, 0x56DU, 0x5D9U, 0x5B4U, 0x5F2U, 0x5B7U, 0x5E2U, 0x5D4U,
    0x5DBU, 0x5DCU, 0x5DDU, 0x5E8U, 0x5EAU, 0x5E9U, 0x5C1U, 0x5E9U, 0x5C2U, 0xFB49U, 0x5C1U, 0xFB49U,
    0x5C2U, 0x5D0U, 0x5B7U, 0x5D0U, 0x5B8U, 0x5D0U, 0x5BCU, 0x5D1U, 0x5BCU, 0x5D2U, 0x5BCU, 0x5D3U,
    0x5BCU, 0x5D4U, 0x5BCU, 0x5D5U, 0x5BCU, 0x5D6U, 0x5BCU, 0x5D8U, 0x5BCU, 0x5D9U, 0x5BCU, 0x5DAU,
    0x5BCU, 0x5DBU, 0x5BCU, 0x5DCU, 0x5BCU, 0x5DEU, 0x5BCU, 0x5E0U, 0x5BCU, 0x5E1U, 0x5BCU, 0x5E3U,
    0x5BCU, 0x5E4U, 0x5BCU, 0x5E6U, 0x5BCU, 0x5E7U, 0x5BCU, 0x5E8U, 0x5BCU, 0x5E9U, 0x5BCU, 0x5EAU,
    0x5BCU, 0x5D5U, 0x5B9U, 0x5D1U, 0x5BFU, 0x5DBU, 0x5BFU, 0x5E4U, 0x5BFU, 0x5D0U, 0x5DCU, 0x671U,
    0x67BU, 0x67EU, 0x680U, 0x67AU, 0x67FU, 0x679U, 0x6A4U, 0x6A6U, 0x684U, 0x683U, 0x686U, 0x687U,
    0x68DU, 0x68CU, 0x68EU, 0x688U, 0x698U, 0x691U, 0x6A9U, 0x6AFU, 0x6B3U, 0x6B1U, 0x6BAU, 0x6BBU,
    0x6C0U, 0x6C1U, 0x6BEU, 0x6D2U, 0x6D3U, 0x6ADU, 0x6C7U, 0x6C6U, 0x6C8U, 0x677U, 0x6CBU, 0x6C5U,
    0x6C9U, 0x6D0U, 0x649U, 0x626U, 0x627U, 0x626U, 0x6D5U, 0x626U, 0x648U, 0x626U, 0x6C7U, 0x626U,
    0x6C6U, 0x626U, 0x6C8U, 0x626U, 0x6D0U, 0x626U, 0x649U, 0x6CCU, 0x626U, 0x62CU, 0x626U, 0x62DU,
    0x626U, 0x645U, 0x626U, 0x64AU, 0x628U, 0x62CU, 0x628U, 0x62DU, 0x628U, 0x62EU, 0x628U, 0x645U,
    0x628U, 0x649U, 0x628U, 0x64AU, 0x62AU, 0x62CU, 0x62AU, 0x62DU, 0x62AU, 0x62EU, 0x62AU, 0x645U,
    0x62AU, 0x649U, 0x62AU, 0x64AU, 0x62BU, 0x62CU, 0x62BU, 0x645U, 0x62BU, 0x649U, 0x62BU, 0x64AU,
    0x62CU, 0x62DU, 0x62CU, 0x645U, 0x62DU, 0x62CU, 0x62DU, 0x645U, 0x62EU, 0x62CU, 0x62EU, 0x62DU,
    0x62EU, 0x645U, 0x633U, 0x62CU, 0x633U, 0x62DU, 0x633U, 0x62EU, 0x633U, 0x645U, 0x635U, 0x62DU,
    0x635U, 0x645U, 0x636U, 0x62CU, 0x636U, 0x62DU, 0x636U, 0x62EU, 0x636U, 0x645U, 0x637U, 0x62DU,
    0x637U, 0x645U, 0x638U, 0x645U, 0x639U, 0x62CU, 0x639U, 0x645U, 0x63AU, 0x62CU, 0x63AU, 0x645U,
    0x641U, 0x62CU, 0x641U, 0x62DU, 0x641U, 0x62EU, 0x641U, 0x645U, 0x641U, 0x649U, 0x641U, 0x64AU,
    0x642U, 0x62DU, 0x642U, 0x645U, 0x642U, 0x649U, 0x642U, 0x64AU, 0x643U, 0x627U, 0x643U, 0x62CU,
    0x643U, 0x62DU, 0x643U, 0x62EU, 0x643U, 0x644U, 0x643U, 0x645U, 0x643U, 0x649U, 0x643U, 0x64AU,
    0x644U, 0x62CU, 0x644U, 0x62DU, 0x644U, 0x62EU, 0x644U, 0x645U, 0x644U, 0x649U, 0x644U, 0x64AU,
    0x645U, 0x62CU, 0x645U, 0x62DU, 0x645U, 0x62EU, 0x645U, 0x645U, 0x645U, 0x649U, 0x645U, 0x64AU,
    0x646U, 0x62CU, 0x646U, 0x62DU, 0x646U, 0x62EU, 0x646U, 0x645U, 0x646U, 0x649U, 0x646U, 0x64AU,
    0x647U, 0x62CU, 0x647U, 0x645U, 0x647U, 0x649U, 0x647U, 0x64AU, 0x64AU, 0x62CU, 0x64AU, 0x62DU,
    0x64AU, 0x62EU, 0x64AU, 0x645U, 0x64AU, 0x649U, 0x64AU, 0x64AU, 0x630U, 0x670U, 0x631U, 0x670U,
    0x649U, 0x670U, 0x20U, 0x64CU, 0x651U, 0x20U, 0x64DU, 0x651U, 0x20U, 0x64EU, 0x651U, 0x20U,
    0x64FU, 0x651U, 0x20U, 0x650U, 0x651U, 0x20U, 0x651U, 0x670U, 0x626U, 0x631U, 0x626U, 0x632U,
    0x626U, 0x646U, 0x628U, 0x631U, 0x628U, 0x632U, 0x628U, 0x646U, 0x62AU, 0x631U, 0x62AU, 0x632U,
    0x62AU, 0x646U, 0x62BU, 0x631U, 0x62BU, 0x632U, 0x62BU, 0x646U, 0x645U, 0x627U, 0x646U, 0x631U,
    0x646U, 0x632U, 0x646U, 0x646U, 0x64AU, 0x631U, 0x64AU, 0x632U, 0x64AU, 0x646U, 0x626U, 0x62EU,
    0x626U, 0x647U, 0x628U, 0x647U, 0x62AU, 0x647U, 0x635U, 0x62EU, 0x644U, 0x647U, 0x646U, 0x647U,
    0x647U, 0x670U, 0x64AU, 0x647U, 0x62BU, 0x647U, 0x633U, 0x647U, 0x634U, 0x645U, 0x634U, 0x647U,
    0x640U, 0x64EU, 0x651U, 0x640U, 0x64FU, 0x651U, 0x640U, 0x650U, 0x651U, 0x637U, 0x649U, 0x637U,
    0x64AU, 0x639U, 0x649U, 0x639U, 0x64AU, 0x63AU, 0x649U, 0x63AU, 0x64AU, 0x633U, 0x649U, 0x633U,
    0x64AU, 0x634U, 0x649U, 0x634U, 0x64AU, 0x62DU, 0x649U, 0x62DU, 0x64AU, 0x62CU, 0x649U, 0x62CU,
    0x64AU, 0x62EU, 0x649U, 0x62EU, 0x64AU, 0x635U, 0x649U, 0x635U, 0x64AU, 0x636U, 0x649U, 0x636U,
    0x64AU, 0x634U, 0x62CU, 0x634U, 0x62DU, 0x634U, 0x62EU, 0x634U, 0x631U, 0x633U, 0x631U, 0x635U,
    0x631U, 0x636U, 0x631U, 0x627U, 0x64BU, 0x62AU, 0x62CU, 0x645U, 0x62AU, 0x62DU, 0x62CU, 0x62AU,
    0x62DU, 0x645U, 0x62AU, 0x62EU, 0x645U, 0x62AU, 0x645U, 0x62CU, 0x62AU, 0x645U, 0x62DU, 0x62AU,
    0x645U, 0x62EU, 0x62CU, 0x645U, 0x62DU, 0x62DU, 0x645U, 0x64AU, 0x62DU, 0x645U, 0x649U, 0x633U,
    0x62DU, 0x62CU, 0x633U, 0x62CU, 0x62DU, 0x633U, 0x62CU, 0x649U, 0x633U, 0x645U, 0x62DU, 0x633U,
    0x645U, 0x62CU, 0x633U, 0x645U, 0x645U, 0x635U, 0x62DU, 0x62DU, 0x635U, 0x645U, 0x645U, 0x634U,
    0x62DU, 0x645U, 0x634U, 0x62CU, 0x64AU, 0x634U, 0x645U, 0x62EU, 0x634U, 0x645U, 0x645U, 0x636U,
    0x62DU, 0x649U, 0x636U, 0x62EU, 0x645U, 0x637U, 0x645U, 0x62DU, 0x637U, 0x645U, 0x645U, 0x637U,
    0x645U, 0x64AU, 0x639U, 0x62CU, 0x645U, 0x639U, 0x645U, 0x645U, 0x639U, 0x645U, 0x649U, 0x63AU,
    0x645U, 0x645U, 0x63AU, 0x645U, 0x64AU, 0x63AU, 0x645U, 0x649U, 0x641U, 0x62EU, 0x645U, 0x642U,
    0x645U, 0x62DU, 0x642U, 0x645U, 0x645U, 0x644U, 0x62DU, 0x645U, 0x644U, 0x62DU, 0x64AU, 0x644U,
    0x62DU, 0x649U, 0x644U, 0x62CU, 0x62CU, 0x644U, 0x62EU, 0x645U, 0x644U, 0x645U, 0x62DU, 0x645U,
    0x62DU, 0x62CU, 0x645U, 0x62DU, 0x645U, 0x645U, 0x62DU, 0x64AU, 0x645U, 0x62CU, 0x62DU, 0x645U,
    0x62CU, 0x645U, 0x645U, 0x62EU, 0x62CU, 0x645U, 0x62EU, 0x645U, 0x645U, 0x62CU, 0x62EU, 0x647U,
    0x645U, 0x62CU, 0x647U, 0x645U, 0x645U, 0x646U, 0x62DU, 0x645U, 0x646U, 0x62DU, 0x649U, 0x646U,
    0x62CU, 0x645U, 0x646U, 0x62CU, 0x649U, 0x646U, 0x645U, 0x64AU, 0x646U, 0x645U, 0x649U, 0x64AU,
    0x645U, 0x645U, 0x628U, 0x62EU, 0x64AU, 0x62AU, 0x62CU, 0x64AU, 0x62AU, 0x62CU, 0x649U, 0x62AU,
    0x62EU, 0x64AU, 0x62AU, 0x62EU, 0x649U, 0x62AU, 0x645U, 0x64AU, 0x62AU, 0x645U, 0x649U, 0x62CU,
    0x645U, 0x64AU, 0x62CU, 0x62DU, 0x649U, 0x62CU, 0x645U, 0x649U, 0x633U, 0x62EU, 0x649U, 0x635U,
    0x62DU, 0x64AU, 0x634U, 0x62DU, 0x64AU, 0x636U, 0x62DU, 0x64AU, 0x644U, 0x62CU, 0x64AU, 0x644U,
    0x645U, 0x64AU, 0x64AU, 0x62DU, 0x64AU, 0x64AU, 0x62CU, 0x64AU, 0x64AU, 0x645U, 0x64AU, 0x645U,
    0x645U, 0x64AU, 0x642U, 0x645U, 0x64AU, 0x646U, 0x62DU, 0x64AU, 0x639U, 0x645U, 0x64AU, 0x643U,
    0x645U, 0x64AU, 0x646U, 0x62CU, 0x62DU, 0x645U, 0x62EU, 0x64AU, 0x644U, 0x62CU, 0x645U, 0x643U,
    0x645U, 0x645U, 0x62CU, 0x62DU, 0x64AU, 0x62DU, 0x62CU, 0x64AU, 0x645U, 0x62CU, 0x64AU, 0x641U,
    0x645U, 0x64AU, 0x628U, 0x62DU, 0x64AU, 0x633U, 0x62EU, 0x64AU, 0x646U, 0x62CU, 0x64AU, 0x635U,
    0x644U, 0x6D2U, 0x642U, 0x644U, 0x6D2U, 0x627U, 0x644U, 0x644U, 0x647U, 0x627U, 0x643U, 0x628U,
    0x631U, 0x645U, 0x62DU, 0x645U, 0x62FU, 0x635U, 0x644U, 0x639U, 0x645U, 0x631U, 0x633U, 0x648U,
    0x644U, 0x639U, 0x644U, 0x64AU, 0x647U, 0x648U, 0x633U, 0x644U, 0x645U, 0x635U, 0x644U, 0x649U,
    0x635U, 0x644U, 0x649U, 0x20U, 0x627U, 0x644U, 0x644U, 0x647U, 0x20U, 0x639U, 0x644U, 0x64AU,
    0x647U, 0x20U, 0x648U, 0x633U, 0x644U, 0x645U, 0x62CU, 0x644U, 0x20U, 0x62CU, 0x644U, 0x627U,
    0x644U, 0x647U, 0x631U, 0x6CCU, 0x627U, 0x644U, 0x2CU, 0x3001U, 0x3002U, 0x3AU, 0x21U, 0x3FU,
    0x3016U, 0x3017U, 0x2026U, 0x2025U, 0x2014U, 0x2013U, 0x5FU, 0x7BU, 0x7DU, 0x3014U, 0x3015U, 0x3010U,
    0x3011U, 0x300AU, 0x300BU, 0x300CU, 0x300DU, 0x300EU, 0x300FU, 0x5BU, 0x5DU, 0x203EU, 0x23U, 0x26U,
    0x2AU, 0x2DU, 0x3CU, 0x3EU, 0x5CU, 0x24U, 0x25U, 0x40U, 0x20U, 0x64BU, 0x640U, 0x64BU,
    0x20U, 0x64CU, 0x20U, 0x64DU, 0x20U, 0x64EU, 0x640U, 0x64EU, 0x20U, 0x64FU, 0x640U, 0x64FU,
    0x20U, 0x650U, 0x640U, 0x650U, 0x20U, 0x651U, 0x640U, 0x651U, 0x20U, 0x652U, 0x640U, 0x652U,
    0x621U, 0x622U, 0x623U, 0x624U, 0x625U, 0x626U, 0x627U, 0x628U, 0x629U, 0x62AU, 0x62BU, 0x62CU,
    0x62DU, 0x62EU, 0x62FU, 0x630U, 0x631U, 0x632U, 0x633U, 0x634U, 0x635U, 0x636U, 0x637U, 0x638U,
    0x639U, 0x63AU, 0x641U, 0x642U, 0x643U, 0x644U, 0x645U, 0x646U, 0x647U, 0x648U, 0x64AU, 0x644U,
    0x622U, 0x644U, 0x623U, 0x644U, 0x625U, 0x644U, 0x627U, 0x22U, 0x27U, 0x2FU, 0x5EU, 0x7CU,
    0x7EU, 0x2985U, 0x2986U, 0x30FBU, 0x30A1U, 0x30A3U, 0x30A5U, 0x30A7U, 0x30A9U, 0x30E3U, 0x30E5U, 0x30E7U,
    0x30C3U, 0x30FCU, 0x30F3U, 0x3099U, 0x309AU, 0x3164U, 0x3131U, 0x3132U, 0x3133U, 0x3134U, 0x3135U, 0x3136U,
    0x3137U, 0x3138U, 0x3139U, 0x313AU, 0x313BU, 0x313CU, 0x313DU, 0x313EU, 0x313FU, 0x3140U, 0x3141U, 0x3142U,
    0x3143U, 0x3144U, 0x3145U, 0x3146U, 0x3147U, 0x3148U, 0x3149U, 0x314AU, 0x314BU, 0x314CU, 0x314DU, 0x314EU,
    0x314FU, 0x3150U, 0x3151U, 0x3152U, 0x3153U, 0x3154U, 0x3155U, 0x3156U, 0x3157U, 0x3158U, 0x3159U, 0x315AU,
    0x315BU, 0x315CU, 0x315DU, 0x315EU, 0x315FU, 0x3160U, 0x3161U, 0x3162U, 0x3163U, 0xA2U, 0xA3U, 0xACU,
    0xAFU, 0xA6U, 0xA5U, 0x20A9U, 0x2502U, 0x2190U, 0x2191U, 0x2192U, 0x2193U, 0x25A0U, 0x25CBU, 0x105D2U,
    0x307U, 0x105DAU, 0x307U, 0x2D0U, 0x2D1U, 0xE6U, 0x299U, 0x253U, 0x2A3U, 0xAB66U, 0x2A5U, 0x2A4U,
    0x256U, 0x257U, 0x1D91U, 0x258U, 0x25EU, 0x2A9U, 0x264U, 0x262U, 0x260U, 0x29BU, 0x29CU, 0x267U,
    0x284U, 0x2AAU, 0x2ABU, 0x26CU, 0x1DF04U, 0xA78EU, 0x26EU, 0x1DF05U, 0x28EU, 0x1DF06U, 0xF8U, 0x276U,
    0x277U, 0x27AU, 0x1DF08U, 0x27DU, 0x27EU, 0x280U, 0x2A8U, 0x2A6U, 0xAB67U, 0x2A7U, 0x288U, 0x2C71U,
    0x28FU, 0x2A1U, 0x2A2U, 0x298U, 0x1C0U, 0x1C1U, 0x1C2U, 0x1DF0AU, 0x1DF1EU, 0x11099U, 0x110BAU, 0x1109BU,
    0x110BAU, 0x110A5U, 0x110BAU, 0x11131U, 0x11127U, 0x11132U, 0x11127U, 0x11347U, 0x1133EU, 0x11347U, 0x11357U, 0x11382U,
    0x113C9U, 0x11384U, 0x113BBU, 0x1138BU, 0x113C2U, 0x11390U, 0x113C9U, 0x113C2U, 0x113C2U, 0x113C2U, 0x113B8U, 0x113C2U,
    0x113C9U, 0x114B9U, 0x114BAU, 0x114B9U, 0x114B0U, 0x114B9U, 0x114BDU, 0x115B8U, 0x115AFU, 0x115B9U, 0x115AFU, 0x11935U,
    0x11930U, 0x1611EU, 0x1611EU, 0x1611EU, 0x16129U, 0x1611EU, 0x1611FU, 0x16129U, 0x1611FU, 0x1611EU, 0x16120U, 0x16121U,
    0x1611FU, 0x16122U, 0x1611FU, 0x16121U, 0x16120U, 0x16D67U, 0x16D67U, 0x16D63U, 0x16D67U, 0x16D69U, 0x16D67U, 0x1D157U,
    0x1D165U, 0x1D158U, 0x1D165U, 0x1D15FU, 0x1D16EU, 0x1D15FU, 0x1D16FU, 0x1D15FU, 0x1D170U, 0x1D15FU, 0x1D171U, 0x1D15FU,
    0x1D172U, 0x1D1B9U, 0x1D165U, 0x1D1BAU, 0x1D165U, 0x1D1BBU, 0x1D16EU, 0x1D1BCU, 0x1D16EU, 0x1D1BBU, 0x1D16FU, 0x1D1BCU,
    0x1D16FU, 0x131U, 0x237U, 0x391U, 0x392U, 0x394U, 0x395U, 0x396U, 0x397U, 0x399U, 0x39AU, 0x39BU,
    0x39CU, 0x39DU, 0x39EU, 0x39FU, 0x3A1U, 0x3F4U, 0x3A4U, 0x3A6U, 0x3A7U, 0x3A8U, 0x2207U, 0x3B1U,
    0x3B6U, 0x3B7U, 0x3BBU, 0x3BDU, 0x3BEU, 0x3BFU, 0x3C3U, 0x3C4U, 0x3C5U, 0x3C8U, 0x3C9U, 0x2202U,
    0x3F5U, 0x3D1U, 0x3F0U, 0x3D5U, 0x3F1U, 0x3D6U, 0x3DCU, 0x3DDU, 0x430U, 0x431U, 0x432U, 0x433U,
    0x434U, 0x435U, 0x436U, 0x437U, 0x438U, 0x43AU, 0x43BU, 0x43CU, 0x43EU, 0x43FU, 0x440U, 0x441U,
    0x442U, 0x443U, 0x444U, 0x445U, 0x446U, 0x447U, 0x448U, 0x44BU, 0x44DU, 0x44EU, 0xA689U, 0x4D9U,
    0x456U, 0x458U, 0x4E9U, 0x4AFU, 0x4CFU, 0x491U, 0x455U, 0x45FU, 0x4ABU, 0xA651U, 0x4B1U, 0x66EU,
    0x6A1U, 0x66FU, 0x30U, 0x2EU, 0x30U, 0x2CU, 0x31U, 0x2CU, 0x32U, 0x2CU, 0x33U, 0x2CU,
    0x34U, 0x2CU, 0x35U, 0x2CU, 0x36U, 0x2CU, 0x37U, 0x2CU, 0x38U, 0x2CU, 0x39U, 0x2CU,
    0x28U, 0x41U, 0x29U, 0x28U, 0x42U, 0x29U, 0x28U, 0x43U, 0x29U, 0x28U, 0x44U, 0x29U,
    0x28U, 0x45U, 0x29U, 0x28U, 0x46U, 0x29U, 0x28U, 0x47U, 0x29U, 0x28U, 0x48U, 0x29U,
    0x28U, 0x49U, 0x29U, 0x28U, 0x4AU, 0x29U, 0x28U, 0x4BU, 0x29U, 0x28U, 0x4CU, 0x29U,
    0x28U, 0x4DU, 0x29U, 0x28U, 0x4EU, 0x29U, 0x28U, 0x4FU, 0x29U, 0x28U, 0x50U, 0x29U,
    0x28U, 0x51U, 0x29U, 0x28U, 0x52U, 0x29U, 0x28U, 0x53U, 0x29U, 0x28U, 0x54U, 0x29U,
    0x28U, 0x55U, 0x29U, 0x28U, 0x56U, 0x29U, 0x28U, 0x57U, 0x29U, 0x28U, 0x58U, 0x29U,
    0x28U, 0x59U, 0x29U, 0x28U, 0x5AU, 0x29U, 0x3014U, 0x53U, 0x3015U, 0x43U, 0x44U, 0x57U,
    0x5AU, 0x48U, 0x56U, 0x53U, 0x44U, 0x53U, 0x53U, 0x50U, 0x50U, 0x56U, 0x57U, 0x43U,
    0x4DU, 0x43U, 0x4DU, 0x44U, 0x4DU, 0x52U, 0x44U, 0x4AU, 0x307BU, 0x304BU, 0x30B3U, 0x30B3U,
    0x5B57U, 0x53CCU, 0x30C7U, 0x591AU, 0x89E3U, 0x4EA4U, 0x6620U, 0x7121U, 0x524DU, 0x5F8CU, 0x518DU, 0x65B0U,
    0x521DU, 0x7D42U, 0x8CA9U, 0x58F0U, 0x5439U, 0x6F14U, 0x6295U, 0x6355U, 0x904AU, 0x6307U, 0x6253U, 0x7981U,
    0x7A7AU, 0x5408U, 0x6E80U, 0x7533U, 0x5272U, 0x55B6U, 0x914DU, 0x3014U, 0x672CU, 0x3015U, 0x3014U, 0x4E09U,
    0x3015U, 0x3014U, 0x4E8CU, 0x3015U, 0x3014U, 0x5B89U, 0x3015U, 0x3014U, 0x70B9U, 0x3015U, 0x3014U, 0x6253U,
    0x3015U, 0x3014U, 0x76D7U, 0x3015U, 0x3014U, 0x52DDU, 0x3015U, 0x3014U, 0x6557U, 0x3015U, 0x5F97U, 0x53EFU,
    0x4E3DU, 0x4E38U, 0x4E41U, 0x20122U, 0x4F60U, 0x4FBBU, 0x5002U, 0x507AU, 0x5099U, 0x50CFU, 0x349EU, 0x2063AU,
    0x5154U, 0x5164U, 0x5177U, 0x2051CU, 0x34B9U, 0x5167U, 0x2054BU, 0x5197U, 0x51A4U, 0x4ECCU, 0x51ACU, 0x291DFU,
    0x5203U, 0x34DFU, 0x523BU, 0x5246U, 0x5277U, 0x3515U, 0x5305U, 0x5306U, 0x5349U, 0x535AU, 0x5373U, 0x537DU,
    0x537FU, 0x20A2CU, 0x7070U, 0x53CAU, 0x53DFU, 0x20B63U, 0x53EBU, 0x53F1U, 0x5406U, 0x549EU, 0x5438U, 0x5448U,
    0x5468U, 0x54A2U, 0x54F6U, 0x5510U, 0x5553U, 0x5563U, 0x5584U, 0x55ABU, 0x55B3U, 0x55C2U, 0x5716U, 0x5717U,
    0x5651U, 0x5674U, 0x58EEU, 0x57CEU, 0x57F4U, 0x580DU, 0x578BU, 0x5832U, 0x5831U, 0x58ACU, 0x214E4U, 0x58F2U,
    0x58F7U, 0x5906U, 0x5922U, 0x5962U, 0x216A8U, 0x216EAU, 0x59ECU, 0x5A1BU, 0x5A27U, 0x59D8U, 0x5A66U, 0x36EEU,
    0x36FCU, 0x5B08U, 0x5B3EU, 0x219C8U, 0x5BC3U, 0x5BD8U, 0x5BF3U, 0x21B18U, 0x5BFFU, 0x5C06U, 0x5F53U, 0x3781U,
    0x5C60U, 0x5CC0U, 0x5C8DU, 0x21DE4U, 0x5D43U, 0x21DE6U, 0x5D6EU, 0x5D6BU, 0x5D7CU, 0x5DE1U, 0x5DE2U, 0x382FU,
    0x5DFDU, 0x5E28U, 0x5E3DU, 0x5E69U, 0x3862U, 0x22183U, 0x387CU, 0x5EB0U, 0x5EB3U, 0x5EB6U, 0x2A392U, 0x22331U,
    0x8201U, 0x5F22U, 0x38C7U, 0x232B8U, 0x261DAU, 0x5F62U, 0x5F6BU, 0x38E3U, 0x5F9AU, 0x5FCDU, 0x5FD7U, 0x5FF9U,
    0x6081U, 0x393AU, 0x391CU, 0x226D4U, 0x60C7U, 0x6148U, 0x614CU, 0x617AU, 0x61B2U, 0x61A4U, 0x61AFU, 0x61DEU,
    0x6210U, 0x621BU, 0x625DU, 0x62B1U, 0x62D4U, 0x6350U, 0x22B0CU, 0x633DU, 0x62FCU, 0x6368U, 0x6383U, 0x63E4U,
    0x22BF1U, 0x6422U, 0x63C5U, 0x63A9U, 0x3A2EU, 0x6469U, 0x647EU, 0x649DU, 0x6477U, 0x3A6CU, 0x656CU, 0x2300AU,
    0x65E3U, 0x66F8U, 0x6649U, 0x3B19U, 0x3B08U, 0x3AE4U, 0x5192U, 0x5195U, 0x6700U, 0x669CU, 0x80ADU, 0x43D9U,
    0x6721U, 0x675EU, 0x6753U, 0x233C3U, 0x3B49U, 0x67FAU, 0x6785U, 0x6852U, 0x2346DU, 0x688EU, 0x681FU, 0x6914U,
    0x6942U, 0x69A3U, 0x69EAU, 0x6AA8U, 0x236A3U, 0x6ADBU, 0x3C18U, 0x6B21U, 0x238A7U, 0x6B54U, 0x3C4EU, 0x6B72U,
    0x6B9FU, 0x6BBBU, 0x23A8DU, 0x21D0BU, 0x23AFAU, 0x6C4EU, 0x23CBCU, 0x6CBFU, 0x6CCDU, 0x6C67U, 0x6D16U, 0x6D3EU,
    0x6D69U, 0x6D78U, 0x6D85U, 0x23D1EU, 0x6D34U, 0x6E2FU, 0x6E6EU, 0x3D33U, 0x6EC7U, 0x23ED1U, 0x6DF9U, 0x6F6EU,
    0x23F5EU, 0x23F8EU, 0x6FC6U, 0x7039U, 0x701BU, 0x3D96U, 0x704AU, 0x707DU, 0x7077U, 0x70ADU, 0x20525U, 0x7145U,
    0x24263U, 0x719CU, 0x243ABU, 0x7228U, 0x7250U, 0x24608U, 0x7280U, 0x7295U, 0x24735U, 0x24814U, 0x737AU, 0x738BU,
    0x3EACU, 0x73A5U, 0x3EB8U, 0x7447U, 0x745CU, 0x7485U, 0x74CAU, 0x3F1BU, 0x7524U, 0x24C36U, 0x753EU, 0x24C92U,
    0x2219FU, 0x7610U, 0x24FA1U, 0x24FB8U, 0x25044U, 0x3FFCU, 0x4008U, 0x250F3U, 0x250F2U, 0x25119U, 0x25133U, 0x771EU,
    0x771FU, 0x778BU, 0x4046U, 0x4096U, 0x2541DU, 0x784EU, 0x40E3U, 0x25626U, 0x2569AU, 0x256C5U, 0x79EBU, 0x412FU,
    0x7A4AU, 0x7A4FU, 0x2597CU, 0x25AA7U, 0x7AEEU, 0x4202U, 0x25BABU, 0x7BC6U, 0x7BC9U, 0x4227U, 0x25C80U, 0x7CD2U,
    0x42A0U, 0x7CE8U, 0x7CE3U, 0x7D00U, 0x25F86U, 0x7D63U, 0x4301U, 0x7DC7U, 0x7E02U, 0x7E45U, 0x4334U, 0x26228U,
    0x26247U, 0x4359U, 0x262D9U, 0x7F7AU, 0x2633EU, 0x7F95U, 0x7FFAU, 0x264DAU, 0x26523U, 0x8060U, 0x265A8U, 0x8070U,
    0x2335FU, 0x43D5U, 0x80B2U, 0x8103U, 0x440BU, 0x813EU, 0x5AB5U, 0x267A7U, 0x267B5U, 0x23393U, 0x2339CU, 0x8204U,
    0x8F9EU, 0x446BU, 0x8291U, 0x828BU, 0x829DU, 0x52B3U, 0x82B1U, 0x82B3U, 0x82BDU, 0x82E6U, 0x26B3CU, 0x831DU,
    0x8363U, 0x83ADU, 0x8323U, 0x83BDU, 0x83E7U, 0x8353U, 0x83CAU, 0x83CCU, 0x83DCU, 0x26C36U, 0x26D6BU, 0x26CD5U,
    0x452BU, 0x84F1U, 0x84F3U, 0x8516U, 0x273CAU, 0x8564U, 0x26F2CU, 0x455DU, 0x4561U, 0x26FB1U, 0x270D2U, 0x456BU,
    0x8650U, 0x8667U, 0x8669U, 0x86A9U, 0x8688U, 0x870EU, 0x86E2U, 0x8728U, 0x876BU, 0x8786U, 0x45D7U, 0x87E1U,
    0x8801U, 0x45F9U, 0x8860U, 0x27667U, 0x88D7U, 0x88DEU, 0x4635U, 0x88FAU, 0x34BBU, 0x278AEU, 0x27966U, 0x46BEU,
    0x46C7U, 0x8AA0U, 0x27CA8U, 0x8CABU, 0x8CC1U, 0x8D1BU, 0x8D77U, 0x27F2FU, 0x20804U, 0x8DCBU, 0x8DBCU, 0x8DF0U,
    0x208DEU, 0x8ED4U, 0x285D2U, 0x285EDU, 0x9094U, 0x90F1U, 0x9111U, 0x2872EU, 0x911BU, 0x9238U, 0x92D7U, 0x92D8U,
    0x927CU, 0x93F9U, 0x9415U, 0x28BFAU, 0x958BU, 0x4995U, 0x95B7U, 0x28D77U, 0x49E6U, 0x96C3U, 0x5DB2U, 0x9723U,
    0x29145U, 0x2921AU, 0x4A6EU, 0x4A76U, 0x97E0U, 0x2940AU, 0x4AB2U, 0x29496U, 0x9829U, 0x295B6U, 0x98E2U, 0x4B33U,
    0x9929U, 0x99A7U, 0x99C2U, 0x99FEU, 0x4BCEU, 0x29B30U, 0x9C40U, 0x9CFDU, 0x4CCEU, 0x4CEDU, 0x9D67U, 0x2A0CEU,
    0x4CF8U, 0x2A105U, 0x2A20EU, 0x2A291U, 0x4D56U, 0x9EFEU, 0x9F05U, 0x9F0FU, 0x9F16U, 0x2A600U
};

typedef struct {
    uint64_t pair;
    uint32_t composed;
} cs_ucomp_entry;

#define CS_UCOMP_COUNT 949U
static const cs_ucomp_entry cs_ucomp_table[949] = {
    {0x0000003C00000338ULL, 0x226EU},
    {0x0000003D00000338ULL, 0x2260U},
    {0x0000003E00000338ULL, 0x226FU},
    {0x0000004100000300ULL, 0x00C0U},
    {0x0000004100000301ULL, 0x00C1U},
    {0x0000004100000302ULL, 0x00C2U},
    {0x0000004100000303ULL, 0x00C3U},
    {0x0000004100000304ULL, 0x0100U},
    {0x0000004100000306ULL, 0x0102U},
    {0x0000004100000307ULL, 0x0226U},
    {0x0000004100000308ULL, 0x00C4U},
    {0x0000004100000309ULL, 0x1EA2U},
    {0x000000410000030AULL, 0x00C5U},
    {0x000000410000030CULL, 0x01CDU},
    {0x000000410000030FULL, 0x0200U},
    {0x0000004100000311ULL, 0x0202U},
    {0x0000004100000323ULL, 0x1EA0U},
    {0x0000004100000325ULL, 0x1E00U},
    {0x0000004100000328ULL, 0x0104U},
    {0x0000004200000307ULL, 0x1E02U},
    {0x0000004200000323ULL, 0x1E04U},
    {0x0000004200000331ULL, 0x1E06U},
    {0x0000004300000301ULL, 0x0106U},
    {0x0000004300000302ULL, 0x0108U},
    {0x0000004300000307ULL, 0x010AU},
    {0x000000430000030CULL, 0x010CU},
    {0x0000004300000327ULL, 0x00C7U},
    {0x0000004400000307ULL, 0x1E0AU},
    {0x000000440000030CULL, 0x010EU},
    {0x0000004400000323ULL, 0x1E0CU},
    {0x0000004400000327ULL, 0x1E10U},
    {0x000000440000032DULL, 0x1E12U},
    {0x0000004400000331ULL, 0x1E0EU},
    {0x0000004500000300ULL, 0x00C8U},
    {0x0000004500000301ULL, 0x00C9U},
    {0x0000004500000302ULL, 0x00CAU},
    {0x0000004500000303ULL, 0x1EBCU},
    {0x0000004500000304ULL, 0x0112U},
    {0x0000004500000306ULL, 0x0114U},
    {0x0000004500000307ULL, 0x0116U},
    {0x0000004500000308ULL, 0x00CBU},
    {0x0000004500000309ULL, 0x1EBAU},
    {0x000000450000030CULL, 0x011AU},
    {0x000000450000030FULL, 0x0204U},
    {0x0000004500000311ULL, 0x0206U},
    {0x0000004500000323ULL, 0x1EB8U},
    {0x0000004500000327ULL, 0x0228U},
    {0x0000004500000328ULL, 0x0118U},
    {0x000000450000032DULL, 0x1E18U},
    {0x0000004500000330ULL, 0x1E1AU},
    {0x0000004600000307ULL, 0x1E1EU},
    {0x0000004700000301ULL, 0x01F4U},
    {0x0000004700000302ULL, 0x011CU},
    {0x0000004700000304ULL, 0x1E20U},
    {0x0000004700000306ULL, 0x011EU},
    {0x0000004700000307ULL, 0x0120U},
    {0x000000470000030CULL, 0x01E6U},
    {0x0000004700000327ULL, 0x0122U},
    {0x0000004800000302ULL, 0x0124U},
    {0x0000004800000307ULL, 0x1E22U},
    {0x0000004800000308ULL, 0x1E26U},
    {0x000000480000030CULL, 0x021EU},
    {0x0000004800000323ULL, 0x1E24U},
    {0x0000004800000327ULL, 0x1E28U},
    {0x000000480000032EULL, 0x1E2AU},
    {0x0000004900000300ULL, 0x00CCU},
    {0x0000004900000301ULL, 0x00CDU},
    {0x0000004900000302ULL, 0x00CEU},
    {0x0000004900000303ULL, 0x0128U},
    {0x0000004900000304ULL, 0x012AU},
    {0x0000004900000306ULL, 0x012CU},
    {0x0000004900000307ULL, 0x0130U},
    {0x0000004900000308ULL, 0x00CFU},
    {0x0000004900000309ULL, 0x1EC8U},
    {0x000000490000030CULL, 0x01CFU},
    {0x000000490000030FULL, 0x0208U},
    {0x0000004900000311ULL, 0x020AU},
    {0x0000004900000323ULL, 0x1ECAU},
    {0x0000004900000328ULL, 0x012EU},
    {0x0000004900000330ULL, 0x1E2CU},
    {0x0000004A00000302ULL, 0x0134U},
    {0x0000004B00000301ULL, 0x1E30U},
    {0x0000004B0000030CULL, 0x01E8U},
    {0x0000004B00000323ULL, 0x1E32U},
    {0x0000004B00000327ULL, 0x0136U},
    {0x0000004B00000331ULL, 0x1E34U},
    {0x0000004C00000301ULL, 0x0139U},
    {0x0000004C0000030CULL, 0x013DU},
    {0x0000004C00000323ULL, 0x1E36U},
    {0x0000004C00000327ULL, 0x013BU},
    {0x0000004C0000032DULL, 0x1E3CU},
    {0x0000004C00000331ULL, 0x1E3AU},
    {0x0000004D00000301ULL, 0x1E3EU},
    {0x0000004D00000307ULL, 0x1E40U},
    {0x0000004D00000323ULL, 0x1E42U},
    {0x0000004E00000300ULL, 0x01F8U},
    {0x0000004E00000301ULL, 0x0143U},
    {0x0000004E00000303ULL, 0x00D1U},
    {0x0000004E00000307ULL, 0x1E44U},
    {0x0000004E0000030CULL, 0x0147U},
    {0x0000004E00000323ULL, 0x1E46U},
    {0x0000004E00000327ULL, 0x0145U},
    {0x0000004E0000032DULL, 0x1E4AU},
    {0x0000004E00000331ULL, 0x1E48U},
    {0x0000004F00000300ULL, 0x00D2U},
    {0x0000004F00000301ULL, 0x00D3U},
    {0x0000004F00000302ULL, 0x00D4U},
    {0x0000004F00000303ULL, 0x00D5U},
    {0x0000004F00000304ULL, 0x014CU},
    {0x0000004F00000306ULL, 0x014EU},
    {0x0000004F00000307ULL, 0x022EU},
    {0x0000004F00000308ULL, 0x00D6U},
    {0x0000004F00000309ULL, 0x1ECEU},
    {0x0000004F0000030BULL, 0x0150U},
    {0x0000004F0000030CULL, 0x01D1U},
    {0x0000004F0000030FULL, 0x020CU},
    {0x0000004F00000311ULL, 0x020EU},
    {0x0000004F0000031BULL, 0x01A0U},
    {0x0000004F00000323ULL, 0x1ECCU},
    {0x0000004F00000328ULL, 0x01EAU},
    {0x0000005000000301ULL, 0x1E54U},
    {0x0000005000000307ULL, 0x1E56U},
    {0x0000005200000301ULL, 0x0154U},
    {0x0000005200000307ULL, 0x1E58U},
    {0x000000520000030CULL, 0x0158U},
    {0x000000520000030FULL, 0x0210U},
    {0x0000005200000311ULL, 0x0212U},
    {0x0000005200000323ULL, 0x1E5AU},
    {0x0000005200000327ULL, 0x0156U},
    {0x0000005200000331ULL, 0x1E5EU},
    {0x0000005300000301ULL, 0x015AU},
    {0x0000005300000302ULL, 0x015CU},
    {0x0000005300000307ULL, 0x1E60U},
    {0x000000530000030CULL, 0x0160U},
    {0x0000005300000323ULL, 0x1E62U},
    {0x0000005300000326ULL, 0x0218U},
    {0x0000005300000327ULL, 0x015EU},
    {0x0000005400000307ULL, 0x1E6AU},
    {0x000000540000030CULL, 0x0164U},
    {0x0000005400000323ULL, 0x1E6CU},
    {0x0000005400000326ULL, 0x021AU},
    {0x0000005400000327ULL, 0x0162U},
    {0x000000540000032DULL, 0x1E70U},
    {0x0000005400000331ULL, 0x1E6EU},
    {0x0000005500000300ULL, 0x00D9U},
    {0x0000005500000301ULL, 0x00DAU},
    {0x0000005500000302ULL, 0x00DBU},
    {0x0000005500000303ULL, 0x0168U},
    {0x0000005500000304ULL, 0x016AU},
    {0x0000005500000306ULL, 0x016CU},
    {0x0000005500000308ULL, 0x00DCU},
    {0x0000005500000309ULL, 0x1EE6U},
    {0x000000550000030AULL, 0x016EU},
    {0x000000550000030BULL, 0x0170U},
    {0x000000550000030CULL, 0x01D3U},
    {0x000000550000030FULL, 0x0214U},
    {0x0000005500000311ULL, 0x0216U},
    {0x000000550000031BULL, 0x01AFU},
    {0x0000005500000323ULL, 0x1EE4U},
    {0x0000005500000324ULL, 0x1E72U},
    {0x0000005500000328ULL, 0x0172U},
    {0x000000550000032DULL, 0x1E76U},
    {0x0000005500000330ULL, 0x1E74U},
    {0x0000005600000303ULL, 0x1E7CU},
    {0x0000005600000323ULL, 0x1E7EU},
    {0x0000005700000300ULL, 0x1E80U},
    {0x0000005700000301ULL, 0x1E82U},
    {0x0000005700000302ULL, 0x0174U},
    {0x0000005700000307ULL, 0x1E86U},
    {0x0000005700000308ULL, 0x1E84U},
    {0x0000005700000323ULL, 0x1E88U},
    {0x0000005800000307ULL, 0x1E8AU},
    {0x0000005800000308ULL, 0x1E8CU},
    {0x0000005900000300ULL, 0x1EF2U},
    {0x0000005900000301ULL, 0x00DDU},
    {0x0000005900000302ULL, 0x0176U},
    {0x0000005900000303ULL, 0x1EF8U},
    {0x0000005900000304ULL, 0x0232U},
    {0x0000005900000307ULL, 0x1E8EU},
    {0x0000005900000308ULL, 0x0178U},
    {0x0000005900000309ULL, 0x1EF6U},
    {0x0000005900000323ULL, 0x1EF4U},
    {0x0000005A00000301ULL, 0x0179U},
    {0x0000005A00000302ULL, 0x1E90U},
    {0x0000005A00000307ULL, 0x017BU},
    {0x0000005A0000030CULL, 0x017DU},
    {0x0000005A00000323ULL, 0x1E92U},
    {0x0000005A00000331ULL, 0x1E94U},
    {0x0000006100000300ULL, 0x00E0U},
    {0x0000006100000301ULL, 0x00E1U},
    {0x0000006100000302ULL, 0x00E2U},
    {0x0000006100000303ULL, 0x00E3U},
    {0x0000006100000304ULL, 0x0101U},
    {0x0000006100000306ULL, 0x0103U},
    {0x0000006100000307ULL, 0x0227U},
    {0x0000006100000308ULL, 0x00E4U},
    {0x0000006100000309ULL, 0x1EA3U},
    {0x000000610000030AULL, 0x00E5U},
    {0x000000610000030CULL, 0x01CEU},
    {0x000000610000030FULL, 0x0201U},
    {0x0000006100000311ULL, 0x0203U},
    {0x0000006100000323ULL, 0x1EA1U},
    {0x0000006100000325ULL, 0x1E01U},
    {0x0000006100000328ULL, 0x0105U},
    {0x0000006200000307ULL, 0x1E03U},
    {0x0000006200000323ULL, 0x1E05U},
    {0x0000006200000331ULL, 0x1E07U},
    {0x0000006300000301ULL, 0x0107U},
    {0x0000006300000302ULL, 0x0109U},
    {0x0000006300000307ULL, 0x010BU},
    {0x000000630000030CULL, 0x010DU},
    {0x0000006300000327ULL, 0x00E7U},
    {0x0000006400000307ULL, 0x1E0BU},
    {0x000000640000030CULL, 0x010FU},
    {0x0000006400000323ULL, 0x1E0DU},
    {0x0000006400000327ULL, 0x1E11U},
    {0x000000640000032DULL, 0x1E13U},
    {0x0000006400000331ULL, 0x1E0FU},
    {0x0000006500000300ULL, 0x00E8U},
    {0x0000006500000301ULL, 0x00E9U},
    {0x0000006500000302ULL, 0x00EAU},
    {0x0000006500000303ULL, 0x1EBDU},
    {0x0000006500000304ULL, 0x0113U},
    {0x0000006500000306ULL, 0x0115U},
    {0x0000006500000307ULL, 0x0117U},
    {0x0000006500000308ULL, 0x00EBU},
    {0x0000006500000309ULL, 0x1EBBU},
    {0x000000650000030CULL, 0x011BU},
    {0x000000650000030FULL, 0x0205U},
    {0x0000006500000311ULL, 0x0207U},
    {0x0000006500000323ULL, 0x1EB9U},
    {0x0000006500000327ULL, 0x0229U},
    {0x0000006500000328ULL, 0x0119U},
    {0x000000650000032DULL, 0x1E19U},
    {0x0000006500000330ULL, 0x1E1BU},
    {0x0000006600000307ULL, 0x1E1FU},
    {0x0000006700000301ULL, 0x01F5U},
    {0x0000006700000302ULL, 0x011DU},
    {0x0000006700000304ULL, 0x1E21U},
    {0x0000006700000306ULL, 0x011FU},
    {0x0000006700000307ULL, 0x0121U},
    {0x000000670000030CULL, 0x01E7U},
    {0x0000006700000327ULL, 0x0123U},
    {0x0000006800000302ULL, 0x0125U},
    {0x0000006800000307ULL, 0x1E23U},
    {0x0000006800000308ULL, 0x1E27U},
    {0x000000680000030CULL, 0x021FU},
    {0x0000006800000323ULL, 0x1E25U},
    {0x0000006800000327ULL, 0x1E29U},
    {0x000000680000032EULL, 0x1E2BU},
    {0x0000006800000331ULL, 0x1E96U},
    {0x0000006900000300ULL, 0x00ECU},
    {0x0000006900000301ULL, 0x00EDU},
    {0x0000006900000302ULL, 0x00EEU},
    {0x0000006900000303ULL, 0x0129U},
    {0x0000006900000304ULL, 0x012BU},
    {0x0000006900000306ULL, 0x012DU},
    {0x0000006900000308ULL, 0x00EFU},
    {0x0000006900000309ULL, 0x1EC9U},
    {0x000000690000030CULL, 0x01D0U},
    {0x000000690000030FULL, 0x0209U},
    {0x0000006900000311ULL, 0x020BU},
    {0x0000006900000323ULL, 0x1ECBU},
    {0x0000006900000328ULL, 0x012FU},
    {0x0000006900000330ULL, 0x1E2DU},
    {0x0000006A00000302ULL, 0x0135U},
    {0x0000006A0000030CULL, 0x01F0U},
    {0x0000006B00000301ULL, 0x1E31U},
    {0x0000006B0000030CULL, 0x01E9U},
    {0x0000006B00000323ULL, 0x1E33U},
    {0x0000006B00000327ULL, 0x0137U},
    {0x0000006B00000331ULL, 0x1E35U},
    {0x0000006C00000301ULL, 0x013AU},
    {0x0000006C0000030CULL, 0x013EU},
    {0x0000006C00000323ULL, 0x1E37U},
    {0x0000006C00000327ULL, 0x013CU},
    {0x0000006C0000032DULL, 0x1E3DU},
    {0x0000006C00000331ULL, 0x1E3BU},
    {0x0000006D00000301ULL, 0x1E3FU},
    {0x0000006D00000307ULL, 0x1E41U},
    {0x0000006D00000323ULL, 0x1E43U},
    {0x0000006E00000300ULL, 0x01F9U},
    {0x0000006E00000301ULL, 0x0144U},
    {0x0000006E00000303ULL, 0x00F1U},
    {0x0000006E00000307ULL, 0x1E45U},
    {0x0000006E0000030CULL, 0x0148U},
    {0x0000006E00000323ULL, 0x1E47U},
    {0x0000006E00000327ULL, 0x0146U},
    {0x0000006E0000032DULL, 0x1E4BU},
    {0x0000006E00000331ULL, 0x1E49U},
    {0x0000006F00000300ULL, 0x00F2U},
    {0x0000006F00000301ULL, 0x00F3U},
    {0x0000006F00000302ULL, 0x00F4U},
    {0x0000006F00000303ULL, 0x00F5U},
    {0x0000006F00000304ULL, 0x014DU},
    {0x0000006F00000306ULL, 0x014FU},
    {0x0000006F00000307ULL, 0x022FU},
    {0x0000006F00000308ULL, 0x00F6U},
    {0x0000006F00000309ULL, 0x1ECFU},
    {0x0000006F0000030BULL, 0x0151U},
    {0x0000006F0000030CULL, 0x01D2U},
    {0x0000006F0000030FULL, 0x020DU},
    {0x0000006F00000311ULL, 0x020FU},
    {0x0000006F0000031BULL, 0x01A1U},
    {0x0000006F00000323ULL, 0x1ECDU},
    {0x0000006F00000328ULL, 0x01EBU},
    {0x0000007000000301ULL, 0x1E55U},
    {0x0000007000000307ULL, 0x1E57U},
    {0x0000007200000301ULL, 0x0155U},
    {0x0000007200000307ULL, 0x1E59U},
    {0x000000720000030CULL, 0x0159U},
    {0x000000720000030FULL, 0x0211U},
    {0x0000007200000311ULL, 0x0213U},
    {0x0000007200000323ULL, 0x1E5BU},
    {0x0000007200000327ULL, 0x0157U},
    {0x0000007200000331ULL, 0x1E5FU},
    {0x0000007300000301ULL, 0x015BU},
    {0x0000007300000302ULL, 0x015DU},
    {0x0000007300000307ULL, 0x1E61U},
    {0x000000730000030CULL, 0x0161U},
    {0x0000007300000323ULL, 0x1E63U},
    {0x0000007300000326ULL, 0x0219U},
    {0x0000007300000327ULL, 0x015FU},
    {0x0000007400000307ULL, 0x1E6BU},
    {0x0000007400000308ULL, 0x1E97U},
    {0x000000740000030CULL, 0x0165U},
    {0x0000007400000323ULL, 0x1E6DU},
    {0x0000007400000326ULL, 0x021BU},
    {0x0000007400000327ULL, 0x0163U},
    {0x000000740000032DULL, 0x1E71U},
    {0x0000007400000331ULL, 0x1E6FU},
    {0x0000007500000300ULL, 0x00F9U},
    {0x0000007500000301ULL, 0x00FAU},
    {0x0000007500000302ULL, 0x00FBU},
    {0x0000007500000303ULL, 0x0169U},
    {0x0000007500000304ULL, 0x016BU},
    {0x0000007500000306ULL, 0x016DU},
    {0x0000007500000308ULL, 0x00FCU},
    {0x0000007500000309ULL, 0x1EE7U},
    {0x000000750000030AULL, 0x016FU},
    {0x000000750000030BULL, 0x0171U},
    {0x000000750000030CULL, 0x01D4U},
    {0x000000750000030FULL, 0x0215U},
    {0x0000007500000311ULL, 0x0217U},
    {0x000000750000031BULL, 0x01B0U},
    {0x0000007500000323ULL, 0x1EE5U},
    {0x0000007500000324ULL, 0x1E73U},
    {0x0000007500000328ULL, 0x0173U},
    {0x000000750000032DULL, 0x1E77U},
    {0x0000007500000330ULL, 0x1E75U},
    {0x0000007600000303ULL, 0x1E7DU},
    {0x0000007600000323ULL, 0x1E7FU},
    {0x0000007700000300ULL, 0x1E81U},
    {0x0000007700000301ULL, 0x1E83U},
    {0x0000007700000302ULL, 0x0175U},
    {0x0000007700000307ULL, 0x1E87U},
    {0x0000007700000308ULL, 0x1E85U},
    {0x000000770000030AULL, 0x1E98U},
    {0x0000007700000323ULL, 0x1E89U},
    {0x0000007800000307ULL, 0x1E8BU},
    {0x0000007800000308ULL, 0x1E8DU},
    {0x0000007900000300ULL, 0x1EF3U},
    {0x0000007900000301ULL, 0x00FDU},
    {0x0000007900000302ULL, 0x0177U},
    {0x0000007900000303ULL, 0x1EF9U},
    {0x0000007900000304ULL, 0x0233U},
    {0x0000007900000307ULL, 0x1E8FU},
    {0x0000007900000308ULL, 0x00FFU},
    {0x0000007900000309ULL, 0x1EF7U},
    {0x000000790000030AULL, 0x1E99U},
    {0x0000007900000323ULL, 0x1EF5U},
    {0x0000007A00000301ULL, 0x017AU},
    {0x0000007A00000302ULL, 0x1E91U},
    {0x0000007A00000307ULL, 0x017CU},
    {0x0000007A0000030CULL, 0x017EU},
    {0x0000007A00000323ULL, 0x1E93U},
    {0x0000007A00000331ULL, 0x1E95U},
    {0x000000A800000300ULL, 0x1FEDU},
    {0x000000A800000301ULL, 0x0385U},
    {0x000000A800000342ULL, 0x1FC1U},
    {0x000000C200000300ULL, 0x1EA6U},
    {0x000000C200000301ULL, 0x1EA4U},
    {0x000000C200000303ULL, 0x1EAAU},
    {0x000000C200000309ULL, 0x1EA8U},
    {0x000000C400000304ULL, 0x01DEU},
    {0x000000C500000301ULL, 0x01FAU},
    {0x000000C600000301ULL, 0x01FCU},
    {0x000000C600000304ULL, 0x01E2U},
    {0x000000C700000301ULL, 0x1E08U},
    {0x000000CA00000300ULL, 0x1EC0U},
    {0x000000CA00000301ULL, 0x1EBEU},
    {0x000000CA00000303ULL, 0x1EC4U},
    {0x000000CA00000309ULL, 0x1EC2U},
    {0x000000CF00000301ULL, 0x1E2EU},
    {0x000000D400000300ULL, 0x1ED2U},
    {0x000000D400000301ULL, 0x1ED0U},
    {0x000000D400000303ULL, 0x1ED6U},
    {0x000000D400000309ULL, 0x1ED4U},
    {0x000000D500000301ULL, 0x1E4CU},
    {0x000000D500000304ULL, 0x022CU},
    {0x000000D500000308ULL, 0x1E4EU},
    {0x000000D600000304ULL, 0x022AU},
    {0x000000D800000301ULL, 0x01FEU},
    {0x000000DC00000300ULL, 0x01DBU},
    {0x000000DC00000301ULL, 0x01D7U},
    {0x000000DC00000304ULL, 0x01D5U},
    {0x000000DC0000030CULL, 0x01D9U},
    {0x000000E200000300ULL, 0x1EA7U},
    {0x000000E200000301ULL, 0x1EA5U},
    {0x000000E200000303ULL, 0x1EABU},
    {0x000000E200000309ULL, 0x1EA9U},
    {0x000000E400000304ULL, 0x01DFU},
    {0x000000E500000301ULL, 0x01FBU},
    {0x000000E600000301ULL, 0x01FDU},
    {0x000000E600000304ULL, 0x01E3U},
    {0x000000E700000301ULL, 0x1E09U},
    {0x000000EA00000300ULL, 0x1EC1U},
    {0x000000EA00000301ULL, 0x1EBFU},
    {0x000000EA00000303ULL, 0x1EC5U},
    {0x000000EA00000309ULL, 0x1EC3U},
    {0x000000EF00000301ULL, 0x1E2FU},
    {0x000000F400000300ULL, 0x1ED3U},
    {0x000000F400000301ULL, 0x1ED1U},
    {0x000000F400000303ULL, 0x1ED7U},
    {0x000000F400000309ULL, 0x1ED5U},
    {0x000000F500000301ULL, 0x1E4DU},
    {0x000000F500000304ULL, 0x022DU},
    {0x000000F500000308ULL, 0x1E4FU},
    {0x000000F600000304ULL, 0x022BU},
    {0x000000F800000301ULL, 0x01FFU},
    {0x000000FC00000300ULL, 0x01DCU},
    {0x000000FC00000301ULL, 0x01D8U},
    {0x000000FC00000304ULL, 0x01D6U},
    {0x000000FC0000030CULL, 0x01DAU},
    {0x0000010200000300ULL, 0x1EB0U},
    {0x0000010200000301ULL, 0x1EAEU},
    {0x0000010200000303ULL, 0x1EB4U},
    {0x0000010200000309ULL, 0x1EB2U},
    {0x0000010300000300ULL, 0x1EB1U},
    {0x0000010300000301ULL, 0x1EAFU},
    {0x0000010300000303ULL, 0x1EB5U},
    {0x0000010300000309ULL, 0x1EB3U},
    {0x0000011200000300ULL, 0x1E14U},
    {0x0000011200000301ULL, 0x1E16U},
    {0x0000011300000300ULL, 0x1E15U},
    {0x0000011300000301ULL, 0x1E17U},
    {0x0000014C00000300ULL, 0x1E50U},
    {0x0000014C00000301ULL, 0x1E52U},
    {0x0000014D00000300ULL, 0x1E51U},
    {0x0000014D00000301ULL, 0x1E53U},
    {0x0000015A00000307ULL, 0x1E64U},
    {0x0000015B00000307ULL, 0x1E65U},
    {0x0000016000000307ULL, 0x1E66U},
    {0x0000016100000307ULL, 0x1E67U},
    {0x0000016800000301ULL, 0x1E78U},
    {0x0000016900000301ULL, 0x1E79U},
    {0x0000016A00000308ULL, 0x1E7AU},
    {0x0000016B00000308ULL, 0x1E7BU},
    {0x0000017F00000307ULL, 0x1E9BU},
    {0x000001A000000300ULL, 0x1EDCU},
    {0x000001A000000301ULL, 0x1EDAU},
    {0x000001A000000303ULL, 0x1EE0U},
    {0x000001A000000309ULL, 0x1EDEU},
    {0x000001A000000323ULL, 0x1EE2U},
    {0x000001A100000300ULL, 0x1EDDU},
    {0x000001A100000301ULL, 0x1EDBU},
    {0x000001A100000303ULL, 0x1EE1U},
    {0x000001A100000309ULL, 0x1EDFU},
    {0x000001A100000323ULL, 0x1EE3U},
    {0x000001AF00000300ULL, 0x1EEAU},
    {0x000001AF00000301ULL, 0x1EE8U},
    {0x000001AF00000303ULL, 0x1EEEU},
    {0x000001AF00000309ULL, 0x1EECU},
    {0x000001AF00000323ULL, 0x1EF0U},
    {0x000001B000000300ULL, 0x1EEBU},
    {0x000001B000000301ULL, 0x1EE9U},
    {0x000001B000000303ULL, 0x1EEFU},
    {0x000001B000000309ULL, 0x1EEDU},
    {0x000001B000000323ULL, 0x1EF1U},
    {0x000001B70000030CULL, 0x01EEU},
    {0x000001EA00000304ULL, 0x01ECU},
    {0x000001EB00000304ULL, 0x01EDU},
    {0x0000022600000304ULL, 0x01E0U},
    {0x0000022700000304ULL, 0x01E1U},
    {0x0000022800000306ULL, 0x1E1CU},
    {0x0000022900000306ULL, 0x1E1DU},
    {0x0000022E00000304ULL, 0x0230U},
    {0x0000022F00000304ULL, 0x0231U},
    {0x000002920000030CULL, 0x01EFU},
    {0x0000039100000300ULL, 0x1FBAU},
    {0x0000039100000301ULL, 0x0386U},
    {0x0000039100000304ULL, 0x1FB9U},
    {0x0000039100000306ULL, 0x1FB8U},
    {0x0000039100000313ULL, 0x1F08U},
    {0x0000039100000314ULL, 0x1F09U},
    {0x0000039100000345ULL, 0x1FBCU},
    {0x0000039500000300ULL, 0x1FC8U},
    {0x0000039500000301ULL, 0x0388U},
    {0x0000039500000313ULL, 0x1F18U},
    {0x0000039500000314ULL, 0x1F19U},
    {0x0000039700000300ULL, 0x1FCAU},
    {0x0000039700000301ULL, 0x0389U},
    {0x0000039700000313ULL, 0x1F28U},
    {0x0000039700000314ULL, 0x1F29U},
    {0x0000039700000345ULL, 0x1FCCU},
    {0x0000039900000300ULL, 0x1FDAU},
    {0x0000039900000301ULL, 0x038AU},
    {0x0000039900000304ULL, 0x1FD9U},
    {0x0000039900000306ULL, 0x1FD8U},
    {0x0000039900000308ULL, 0x03AAU},
    {0x0000039900000313ULL, 0x1F38U},
    {0x0000039900000314ULL, 0x1F39U},
    {0x0000039F00000300ULL, 0x1FF8U},
    {0x0000039F00000301ULL, 0x038CU},
    {0x0000039F00000313ULL, 0x1F48U},
    {0x0000039F00000314ULL, 0x1F49U},
    {0x000003A100000314ULL, 0x1FECU},
    {0x000003A500000300ULL, 0x1FEAU},
    {0x000003A500000301ULL, 0x038EU},
    {0x000003A500000304ULL, 0x1FE9U},
    {0x000003A500000306ULL, 0x1FE8U},
    {0x000003A500000308ULL, 0x03ABU},
    {0x000003A500000314ULL, 0x1F59U},
    {0x000003A900000300ULL, 0x1FFAU},
    {0x000003A900000301ULL, 0x038FU},
    {0x000003A900000313ULL, 0x1F68U},
    {0x000003A900000314ULL, 0x1F69U},
    {0x000003A900000345ULL, 0x1FFCU},
    {0x000003AC00000345ULL, 0x1FB4U},
    {0x000003AE00000345ULL, 0x1FC4U},
    {0x000003B100000300ULL, 0x1F70U},
    {0x000003B100000301ULL, 0x03ACU},
    {0x000003B100000304ULL, 0x1FB1U},
    {0x000003B100000306ULL, 0x1FB0U},
    {0x000003B100000313ULL, 0x1F00U},
    {0x000003B100000314ULL, 0x1F01U},
    {0x000003B100000342ULL, 0x1FB6U},
    {0x000003B100000345ULL, 0x1FB3U},
    {0x000003B500000300ULL, 0x1F72U},
    {0x000003B500000301ULL, 0x03ADU},
    {0x000003B500000313ULL, 0x1F10U},
    {0x000003B500000314ULL, 0x1F11U},
    {0x000003B700000300ULL, 0x1F74U},
    {0x000003B700000301ULL, 0x03AEU},
    {0x000003B700000313ULL, 0x1F20U},
    {0x000003B700000314ULL, 0x1F21U},
    {0x000003B700000342ULL, 0x1FC6U},
    {0x000003B700000345ULL, 0x1FC3U},
    {0x000003B900000300ULL, 0x1F76U},
    {0x000003B900000301ULL, 0x03AFU},
    {0x000003B900000304ULL, 0x1FD1U},
    {0x000003B900000306ULL, 0x1FD0U},
    {0x000003B900000308ULL, 0x03CAU},
    {0x000003B900000313ULL, 0x1F30U},
    {0x000003B900000314ULL, 0x1F31U},
    {0x000003B900000342ULL, 0x1FD6U},
    {0x000003BF00000300ULL, 0x1F78U},
    {0x000003BF00000301ULL, 0x03CCU},
    {0x000003BF00000313ULL, 0x1F40U},
    {0x000003BF00000314ULL, 0x1F41U},
    {0x000003C100000313ULL, 0x1FE4U},
    {0x000003C100000314ULL, 0x1FE5U},
    {0x000003C500000300ULL, 0x1F7AU},
    {0x000003C500000301ULL, 0x03CDU},
    {0x000003C500000304ULL, 0x1FE1U},
    {0x000003C500000306ULL, 0x1FE0U},
    {0x000003C500000308ULL, 0x03CBU},
    {0x000003C500000313ULL, 0x1F50U},
    {0x000003C500000314ULL, 0x1F51U},
    {0x000003C500000342ULL, 0x1FE6U},
    {0x000003C900000300ULL, 0x1F7CU},
    {0x000003C900000301ULL, 0x03CEU},
    {0x000003C900000313ULL, 0x1F60U},
    {0x000003C900000314ULL, 0x1F61U},
    {0x000003C900000342ULL, 0x1FF6U},
    {0x000003C900000345ULL, 0x1FF3U},
    {0x000003CA00000300ULL, 0x1FD2U},
    {0x000003CA00000301ULL, 0x0390U},
    {0x000003CA00000342ULL, 0x1FD7U},
    {0x000003CB00000300ULL, 0x1FE2U},
    {0x000003CB00000301ULL, 0x03B0U},
    {0x000003CB00000342ULL, 0x1FE7U},
    {0x000003CE00000345ULL, 0x1FF4U},
    {0x000003D200000301ULL, 0x03D3U},
    {0x000003D200000308ULL, 0x03D4U},
    {0x0000040600000308ULL, 0x0407U},
    {0x0000041000000306ULL, 0x04D0U},
    {0x0000041000000308ULL, 0x04D2U},
    {0x0000041300000301ULL, 0x0403U},
    {0x0000041500000300ULL, 0x0400U},
    {0x0000041500000306ULL, 0x04D6U},
    {0x0000041500000308ULL, 0x0401U},
    {0x0000041600000306ULL, 0x04C1U},
    {0x0000041600000308ULL, 0x04DCU},
    {0x0000041700000308ULL, 0x04DEU},
    {0x0000041800000300ULL, 0x040DU},
    {0x0000041800000304ULL, 0x04E2U},
    {0x0000041800000306ULL, 0x0419U},
    {0x0000041800000308ULL, 0x04E4U},
    {0x0000041A00000301ULL, 0x040CU},
    {0x0000041E00000308ULL, 0x04E6U},
    {0x0000042300000304ULL, 0x04EEU},
    {0x0000042300000306ULL, 0x040EU},
    {0x0000042300000308ULL, 0x04F0U},
    {0x000004230000030BULL, 0x04F2U},
    {0x0000042700000308ULL, 0x04F4U},
    {0x0000042B00000308ULL, 0x04F8U},
    {0x0000042D00000308ULL, 0x04ECU},
    {0x0000043000000306ULL, 0x04D1U},
    {0x0000043000000308ULL, 0x04D3U},
    {0x0000043300000301ULL, 0x0453U},
    {0x0000043500000300ULL, 0x0450U},
    {0x0000043500000306ULL, 0x04D7U},
    {0x0000043500000308ULL, 0x0451U},
    {0x0000043600000306ULL, 0x04C2U},
    {0x0000043600000308ULL, 0x04DDU},
    {0x0000043700000308ULL, 0x04DFU},
    {0x0000043800000300ULL, 0x045DU},
    {0x0000043800000304ULL, 0x04E3U},
    {0x0000043800000306ULL, 0x0439U},
    {0x0000043800000308ULL, 0x04E5U},
    {0x0000043A00000301ULL, 0x045CU},
    {0x0000043E00000308ULL, 0x04E7U},
    {0x0000044300000304ULL, 0x04EFU},
    {0x0000044300000306ULL, 0x045EU},
    {0x0000044300000308ULL, 0x04F1U},
    {0x000004430000030BULL, 0x04F3U},
    {0x0000044700000308ULL, 0x04F5U},
    {0x0000044B00000308ULL, 0x04F9U},
    {0x0000044D00000308ULL, 0x04EDU},
    {0x0000045600000308ULL, 0x0457U},
    {0x000004740000030FULL, 0x0476U},
    {0x000004750000030FULL, 0x0477U},
    {0x000004D800000308ULL, 0x04DAU},
    {0x000004D900000308ULL, 0x04DBU},
    {0x000004E800000308ULL, 0x04EAU},
    {0x000004E900000308ULL, 0x04EBU},
    {0x0000062700000653ULL, 0x0622U},
    {0x0000062700000654ULL, 0x0623U},
    {0x0000062700000655ULL, 0x0625U},
    {0x0000064800000654ULL, 0x0624U},
    {0x0000064A00000654ULL, 0x0626U},
    {0x000006C100000654ULL, 0x06C2U},
    {0x000006D200000654ULL, 0x06D3U},
    {0x000006D500000654ULL, 0x06C0U},
    {0x000009280000093CULL, 0x0929U},
    {0x000009300000093CULL, 0x0931U},
    {0x000009330000093CULL, 0x0934U},
    {0x000009C7000009BEULL, 0x09CBU},
    {0x000009C7000009D7ULL, 0x09CCU},
    {0x00000B4700000B3EULL, 0x0B4BU},
    {0x00000B4700000B56ULL, 0x0B48U},
    {0x00000B4700000B57ULL, 0x0B4CU},
    {0x00000B9200000BD7ULL, 0x0B94U},
    {0x00000BC600000BBEULL, 0x0BCAU},
    {0x00000BC600000BD7ULL, 0x0BCCU},
    {0x00000BC700000BBEULL, 0x0BCBU},
    {0x00000C4600000C56ULL, 0x0C48U},
    {0x00000CBF00000CD5ULL, 0x0CC0U},
    {0x00000CC600000CC2ULL, 0x0CCAU},
    {0x00000CC600000CD5ULL, 0x0CC7U},
    {0x00000CC600000CD6ULL, 0x0CC8U},
    {0x00000CCA00000CD5ULL, 0x0CCBU},
    {0x00000D4600000D3EULL, 0x0D4AU},
    {0x00000D4600000D57ULL, 0x0D4CU},
    {0x00000D4700000D3EULL, 0x0D4BU},
    {0x00000DD900000DCAULL, 0x0DDAU},
    {0x00000DD900000DCFULL, 0x0DDCU},
    {0x00000DD900000DDFULL, 0x0DDEU},
    {0x00000DDC00000DCAULL, 0x0DDDU},
    {0x000010250000102EULL, 0x1026U},
    {0x00001B0500001B35ULL, 0x1B06U},
    {0x00001B0700001B35ULL, 0x1B08U},
    {0x00001B0900001B35ULL, 0x1B0AU},
    {0x00001B0B00001B35ULL, 0x1B0CU},
    {0x00001B0D00001B35ULL, 0x1B0EU},
    {0x00001B1100001B35ULL, 0x1B12U},
    {0x00001B3A00001B35ULL, 0x1B3BU},
    {0x00001B3C00001B35ULL, 0x1B3DU},
    {0x00001B3E00001B35ULL, 0x1B40U},
    {0x00001B3F00001B35ULL, 0x1B41U},
    {0x00001B4200001B35ULL, 0x1B43U},
    {0x00001E3600000304ULL, 0x1E38U},
    {0x00001E3700000304ULL, 0x1E39U},
    {0x00001E5A00000304ULL, 0x1E5CU},
    {0x00001E5B00000304ULL, 0x1E5DU},
    {0x00001E6200000307ULL, 0x1E68U},
    {0x00001E6300000307ULL, 0x1E69U},
    {0x00001EA000000302ULL, 0x1EACU},
    {0x00001EA000000306ULL, 0x1EB6U},
    {0x00001EA100000302ULL, 0x1EADU},
    {0x00001EA100000306ULL, 0x1EB7U},
    {0x00001EB800000302ULL, 0x1EC6U},
    {0x00001EB900000302ULL, 0x1EC7U},
    {0x00001ECC00000302ULL, 0x1ED8U},
    {0x00001ECD00000302ULL, 0x1ED9U},
    {0x00001F0000000300ULL, 0x1F02U},
    {0x00001F0000000301ULL, 0x1F04U},
    {0x00001F0000000342ULL, 0x1F06U},
    {0x00001F0000000345ULL, 0x1F80U},
    {0x00001F0100000300ULL, 0x1F03U},
    {0x00001F0100000301ULL, 0x1F05U},
    {0x00001F0100000342ULL, 0x1F07U},
    {0x00001F0100000345ULL, 0x1F81U},
    {0x00001F0200000345ULL, 0x1F82U},
    {0x00001F0300000345ULL, 0x1F83U},
    {0x00001F0400000345ULL, 0x1F84U},
    {0x00001F0500000345ULL, 0x1F85U},
    {0x00001F0600000345ULL, 0x1F86U},
    {0x00001F0700000345ULL, 0x1F87U},
    {0x00001F0800000300ULL, 0x1F0AU},
    {0x00001F0800000301ULL, 0x1F0CU},
    {0x00001F0800000342ULL, 0x1F0EU},
    {0x00001F0800000345ULL, 0x1F88U},
    {0x00001F0900000300ULL, 0x1F0BU},
    {0x00001F0900000301ULL, 0x1F0DU},
    {0x00001F0900000342ULL, 0x1F0FU},
    {0x00001F0900000345ULL, 0x1F89U},
    {0x00001F0A00000345ULL, 0x1F8AU},
    {0x00001F0B00000345ULL, 0x1F8BU},
    {0x00001F0C00000345ULL, 0x1F8CU},
    {0x00001F0D00000345ULL, 0x1F8DU},
    {0x00001F0E00000345ULL, 0x1F8EU},
    {0x00001F0F00000345ULL, 0x1F8FU},
    {0x00001F1000000300ULL, 0x1F12U},
    {0x00001F1000000301ULL, 0x1F14U},
    {0x00001F1100000300ULL, 0x1F13U},
    {0x00001F1100000301ULL, 0x1F15U},
    {0x00001F1800000300ULL, 0x1F1AU},
    {0x00001F1800000301ULL, 0x1F1CU},
    {0x00001F1900000300ULL, 0x1F1BU},
    {0x00001F1900000301ULL, 0x1F1DU},
    {0x00001F2000000300ULL, 0x1F22U},
    {0x00001F2000000301ULL, 0x1F24U},
    {0x00001F2000000342ULL, 0x1F26U},
    {0x00001F2000000345ULL, 0x1F90U},
    {0x00001F2100000300ULL, 0x1F23U},
    {0x00001F2100000301ULL, 0x1F25U},
    {0x00001F2100000342ULL, 0x1F27U},
    {0x00001F2100000345ULL, 0x1F91U},
    {0x00001F2200000345ULL, 0x1F92U},
    {0x00001F2300000345ULL, 0x1F93U},
    {0x00001F2400000345ULL, 0x1F94U},
    {0x00001F2500000345ULL, 0x1F95U},
    {0x00001F2600000345ULL, 0x1F96U},
    {0x00001F2700000345ULL, 0x1F97U},
    {0x00001F2800000300ULL, 0x1F2AU},
    {0x00001F2800000301ULL, 0x1F2CU},
    {0x00001F2800000342ULL, 0x1F2EU},
    {0x00001F2800000345ULL, 0x1F98U},
    {0x00001F2900000300ULL, 0x1F2BU},
    {0x00001F2900000301ULL, 0x1F2DU},
    {0x00001F2900000342ULL, 0x1F2FU},
    {0x00001F2900000345ULL, 0x1F99U},
    {0x00001F2A00000345ULL, 0x1F9AU},
    {0x00001F2B00000345ULL, 0x1F9BU},
    {0x00001F2C00000345ULL, 0x1F9CU},
    {0x00001F2D00000345ULL, 0x1F9DU},
    {0x00001F2E00000345ULL, 0x1F9EU},
    {0x00001F2F00000345ULL, 0x1F9FU},
    {0x00001F3000000300ULL, 0x1F32U},
    {0x00001F3000000301ULL, 0x1F34U},
    {0x00001F3000000342ULL, 0x1F36U},
    {0x00001F3100000300ULL, 0x1F33U},
    {0x00001F3100000301ULL, 0x1F35U},
    {0x00001F3100000342ULL, 0x1F37U},
    {0x00001F3800000300ULL, 0x1F3AU},
    {0x00001F3800000301ULL, 0x1F3CU},
    {0x00001F3800000342ULL, 0x1F3EU},
    {0x00001F3900000300ULL, 0x1F3BU},
    {0x00001F3900000301ULL, 0x1F3DU},
    {0x00001F3900000342ULL, 0x1F3FU},
    {0x00001F4000000300ULL, 0x1F42U},
    {0x00001F4000000301ULL, 0x1F44U},
    {0x00001F4100000300ULL, 0x1F43U},
    {0x00001F4100000301ULL, 0x1F45U},
    {0x00001F4800000300ULL, 0x1F4AU},
    {0x00001F4800000301ULL, 0x1F4CU},
    {0x00001F4900000300ULL, 0x1F4BU},
    {0x00001F4900000301ULL, 0x1F4DU},
    {0x00001F5000000300ULL, 0x1F52U},
    {0x00001F5000000301ULL, 0x1F54U},
    {0x00001F5000000342ULL, 0x1F56U},
    {0x00001F5100000300ULL, 0x1F53U},
    {0x00001F5100000301ULL, 0x1F55U},
    {0x00001F5100000342ULL, 0x1F57U},
    {0x00001F5900000300ULL, 0x1F5BU},
    {0x00001F5900000301ULL, 0x1F5DU},
    {0x00001F5900000342ULL, 0x1F5FU},
    {0x00001F6000000300ULL, 0x1F62U},
    {0x00001F6000000301ULL, 0x1F64U},
    {0x00001F6000000342ULL, 0x1F66U},
    {0x00001F6000000345ULL, 0x1FA0U},
    {0x00001F6100000300ULL, 0x1F63U},
    {0x00001F6100000301ULL, 0x1F65U},
    {0x00001F6100000342ULL, 0x1F67U},
    {0x00001F6100000345ULL, 0x1FA1U},
    {0x00001F6200000345ULL, 0x1FA2U},
    {0x00001F6300000345ULL, 0x1FA3U},
    {0x00001F6400000345ULL, 0x1FA4U},
    {0x00001F6500000345ULL, 0x1FA5U},
    {0x00001F6600000345ULL, 0x1FA6U},
    {0x00001F6700000345ULL, 0x1FA7U},
    {0x00001F6800000300ULL, 0x1F6AU},
    {0x00001F6800000301ULL, 0x1F6CU},
    {0x00001F6800000342ULL, 0x1F6EU},
    {0x00001F6800000345ULL, 0x1FA8U},
    {0x00001F6900000300ULL, 0x1F6BU},
    {0x00001F6900000301ULL, 0x1F6DU},
    {0x00001F6900000342ULL, 0x1F6FU},
    {0x00001F6900000345ULL, 0x1FA9U},
    {0x00001F6A00000345ULL, 0x1FAAU},
    {0x00001F6B00000345ULL, 0x1FABU},
    {0x00001F6C00000345ULL, 0x1FACU},
    {0x00001F6D00000345ULL, 0x1FADU},
    {0x00001F6E00000345ULL, 0x1FAEU},
    {0x00001F6F00000345ULL, 0x1FAFU},
    {0x00001F7000000345ULL, 0x1FB2U},
    {0x00001F7400000345ULL, 0x1FC2U},
    {0x00001F7C00000345ULL, 0x1FF2U},
    {0x00001FB600000345ULL, 0x1FB7U},
    {0x00001FBF00000300ULL, 0x1FCDU},
    {0x00001FBF00000301ULL, 0x1FCEU},
    {0x00001FBF00000342ULL, 0x1FCFU},
    {0x00001FC600000345ULL, 0x1FC7U},
    {0x00001FF600000345ULL, 0x1FF7U},
    {0x00001FFE00000300ULL, 0x1FDDU},
    {0x00001FFE00000301ULL, 0x1FDEU},
    {0x00001FFE00000342ULL, 0x1FDFU},
    {0x0000219000000338ULL, 0x219AU},
    {0x0000219200000338ULL, 0x219BU},
    {0x0000219400000338ULL, 0x21AEU},
    {0x000021D000000338ULL, 0x21CDU},
    {0x000021D200000338ULL, 0x21CFU},
    {0x000021D400000338ULL, 0x21CEU},
    {0x0000220300000338ULL, 0x2204U},
    {0x0000220800000338ULL, 0x2209U},
    {0x0000220B00000338ULL, 0x220CU},
    {0x0000222300000338ULL, 0x2224U},
    {0x0000222500000338ULL, 0x2226U},
    {0x0000223C00000338ULL, 0x2241U},
    {0x0000224300000338ULL, 0x2244U},
    {0x0000224500000338ULL, 0x2247U},
    {0x0000224800000338ULL, 0x2249U},
    {0x0000224D00000338ULL, 0x226DU},
    {0x0000226100000338ULL, 0x2262U},
    {0x0000226400000338ULL, 0x2270U},
    {0x0000226500000338ULL, 0x2271U},
    {0x0000227200000338ULL, 0x2274U},
    {0x0000227300000338ULL, 0x2275U},
    {0x0000227600000338ULL, 0x2278U},
    {0x0000227700000338ULL, 0x2279U},
    {0x0000227A00000338ULL, 0x2280U},
    {0x0000227B00000338ULL, 0x2281U},
    {0x0000227C00000338ULL, 0x22E0U},
    {0x0000227D00000338ULL, 0x22E1U},
    {0x0000228200000338ULL, 0x2284U},
    {0x0000228300000338ULL, 0x2285U},
    {0x0000228600000338ULL, 0x2288U},
    {0x0000228700000338ULL, 0x2289U},
    {0x0000229100000338ULL, 0x22E2U},
    {0x0000229200000338ULL, 0x22E3U},
    {0x000022A200000338ULL, 0x22ACU},
    {0x000022A800000338ULL, 0x22ADU},
    {0x000022A900000338ULL, 0x22AEU},
    {0x000022AB00000338ULL, 0x22AFU},
    {0x000022B200000338ULL, 0x22EAU},
    {0x000022B300000338ULL, 0x22EBU},
    {0x000022B400000338ULL, 0x22ECU},
    {0x000022B500000338ULL, 0x22EDU},
    {0x0000304600003099ULL, 0x3094U},
    {0x0000304B00003099ULL, 0x304CU},
    {0x0000304D00003099ULL, 0x304EU},
    {0x0000304F00003099ULL, 0x3050U},
    {0x0000305100003099ULL, 0x3052U},
    {0x0000305300003099ULL, 0x3054U},
    {0x0000305500003099ULL, 0x3056U},
    {0x0000305700003099ULL, 0x3058U},
    {0x0000305900003099ULL, 0x305AU},
    {0x0000305B00003099ULL, 0x305CU},
    {0x0000305D00003099ULL, 0x305EU},
    {0x0000305F00003099ULL, 0x3060U},
    {0x0000306100003099ULL, 0x3062U},
    {0x0000306400003099ULL, 0x3065U},
    {0x0000306600003099ULL, 0x3067U},
    {0x0000306800003099ULL, 0x3069U},
    {0x0000306F00003099ULL, 0x3070U},
    {0x0000306F0000309AULL, 0x3071U},
    {0x0000307200003099ULL, 0x3073U},
    {0x000030720000309AULL, 0x3074U},
    {0x0000307500003099ULL, 0x3076U},
    {0x000030750000309AULL, 0x3077U},
    {0x0000307800003099ULL, 0x3079U},
    {0x000030780000309AULL, 0x307AU},
    {0x0000307B00003099ULL, 0x307CU},
    {0x0000307B0000309AULL, 0x307DU},
    {0x0000309D00003099ULL, 0x309EU},
    {0x000030A600003099ULL, 0x30F4U},
    {0x000030AB00003099ULL, 0x30ACU},
    {0x000030AD00003099ULL, 0x30AEU},
    {0x000030AF00003099ULL, 0x30B0U},
    {0x000030B100003099ULL, 0x30B2U},
    {0x000030B300003099ULL, 0x30B4U},
    {0x000030B500003099ULL, 0x30B6U},
    {0x000030B700003099ULL, 0x30B8U},
    {0x000030B900003099ULL, 0x30BAU},
    {0x000030BB00003099ULL, 0x30BCU},
    {0x000030BD00003099ULL, 0x30BEU},
    {0x000030BF00003099ULL, 0x30C0U},
    {0x000030C100003099ULL, 0x30C2U},
    {0x000030C400003099ULL, 0x30C5U},
    {0x000030C600003099ULL, 0x30C7U},
    {0x000030C800003099ULL, 0x30C9U},
    {0x000030CF00003099ULL, 0x30D0U},
    {0x000030CF0000309AULL, 0x30D1U},
    {0x000030D200003099ULL, 0x30D3U},
    {0x000030D20000309AULL, 0x30D4U},
    {0x000030D500003099ULL, 0x30D6U},
    {0x000030D50000309AULL, 0x30D7U},
    {0x000030D800003099ULL, 0x30D9U},
    {0x000030D80000309AULL, 0x30DAU},
    {0x000030DB00003099ULL, 0x30DCU},
    {0x000030DB0000309AULL, 0x30DDU},
    {0x000030EF00003099ULL, 0x30F7U},
    {0x000030F000003099ULL, 0x30F8U},
    {0x000030F100003099ULL, 0x30F9U},
    {0x000030F200003099ULL, 0x30FAU},
    {0x000030FD00003099ULL, 0x30FEU},
    {0x000105D200000307ULL, 0x105C9U},
    {0x000105DA00000307ULL, 0x105E4U},
    {0x00011099000110BAULL, 0x1109AU},
    {0x0001109B000110BAULL, 0x1109CU},
    {0x000110A5000110BAULL, 0x110ABU},
    {0x0001113100011127ULL, 0x1112EU},
    {0x0001113200011127ULL, 0x1112FU},
    {0x000113470001133EULL, 0x1134BU},
    {0x0001134700011357ULL, 0x1134CU},
    {0x00011382000113C9ULL, 0x11383U},
    {0x00011384000113BBULL, 0x11385U},
    {0x0001138B000113C2ULL, 0x1138EU},
    {0x00011390000113C9ULL, 0x11391U},
    {0x000114B9000114B0ULL, 0x114BCU},
    {0x000114B9000114BAULL, 0x114BBU},
    {0x000114B9000114BDULL, 0x114BEU},
    {0x000115B8000115AFULL, 0x115BAU},
    {0x000115B9000115AFULL, 0x115BBU},
    {0x0001193500011930ULL, 0x11938U},
    {0x00016D6300016D67ULL, 0x16D69U},
    {0x00016D6900016D67ULL, 0x16D6AU},
};


#define CS_UF_ALPHA    (1U << 0)
#define CS_UF_WSPACE   (1U << 1)
#define CS_UF_UPPER    (1U << 2)
#define CS_UF_LOWER    (1U << 3)
#define CS_UF_CASED    (1U << 4)
#define CS_UF_CI       (1U << 5)
#define CS_UF_SD       (1U << 6)
#define CS_UF_EXTPICT  (1U << 7)

enum {
    CS_GCB_OTHER = 0,
    CS_GCB_CR = 1,
    CS_GCB_LF = 2,
    CS_GCB_CONTROL = 3,
    CS_GCB_EXTEND = 4,
    CS_GCB_ZWJ = 5,
    CS_GCB_RI = 6,
    CS_GCB_PREPEND = 7,
    CS_GCB_SPACINGMARK = 8,
    CS_GCB_L = 9,
    CS_GCB_V = 10,
    CS_GCB_T = 11,
    CS_GCB_LV = 12,
    CS_GCB_LVT = 13,
    CS_GCB_EXTPICT = 14
};

enum {
    CS_WB_OTHER = 0,
    CS_WB_CR = 1,
    CS_WB_LF = 2,
    CS_WB_NEWLINE = 3,
    CS_WB_EXTEND = 4,
    CS_WB_ZWJ = 5,
    CS_WB_RI = 6,
    CS_WB_FORMAT = 7,
    CS_WB_KATAKANA = 8,
    CS_WB_HEBREW_LETTER = 9,
    CS_WB_ALETTER = 10,
    CS_WB_SINGLE_QUOTE = 11,
    CS_WB_DOUBLE_QUOTE = 12,
    CS_WB_MIDNUMLET = 13,
    CS_WB_MIDLETTER = 14,
    CS_WB_MIDNUM = 15,
    CS_WB_NUMERIC = 16,
    CS_WB_EXTENDNUMLET = 17,
    CS_WB_WSEGSPACE = 18
};

enum {
    CS_SB_OTHER = 0,
    CS_SB_CR = 1,
    CS_SB_LF = 2,
    CS_SB_EXTEND = 3,
    CS_SB_SEP = 4,
    CS_SB_FORMAT = 5,
    CS_SB_SP = 6,
    CS_SB_LOWER = 7,
    CS_SB_UPPER = 8,
    CS_SB_OLETTER = 9,
    CS_SB_NUMERIC = 10,
    CS_SB_ATERM = 11,
    CS_SB_STERM = 12,
    CS_SB_CLOSE = 13,
    CS_SB_SCONTINUE = 14
};

enum {
    CS_INCB_NONE      = 0,
    CS_INCB_CONSONANT = 1,
    CS_INCB_LINKER    = 2,
    CS_INCB_EXTEND    = 3
};

static inline const cs_uprop_record *cs_uprop_lookup(uint32_t cp) {
    if (cp >= 0x110000U) {
        return &cs_uprop_palette[0];
    }
    uint32_t s1  = cs_uprop_stage1[cp >> 12];
    uint32_t s2  = cs_uprop_stage2[(s1 << 6) | ((cp >> 6) & 0x3FU)];
    uint16_t idx = cs_uprop_stage3[s2 + (cp & 0x3FU)];
    return &cs_uprop_palette[idx];
}

static const cs_ucase_entry *cs_ucase_find(uint32_t cp) {
    size_t lo = 0, hi = CS_UCASE_COUNT;
    while (lo < hi) {
        size_t mid = lo + ((hi - lo) >> 1);
        uint32_t mcp = cs_ucase_table[mid].cp;
        if (mcp == cp) return &cs_ucase_table[mid];
        if (mcp < cp) lo = mid + 1;
        else hi = mid;
    }
    return NULL;
}

static size_t cs_ucase_decode_target(uint32_t cp, int32_t val, uint32_t out_cps[3]) {
    if (val >= 0) {
        out_cps[0] = (uint32_t)val;
        return 1U;
    }
    uint32_t packed = (uint32_t)(-val);
    uint32_t len    = packed & 0x7U;
    uint32_t off    = packed >> 3;
    for (uint32_t i = 0; i < len; i++) {
        out_cps[i] = cs_ucase_multi_pool[off + i];
    }
    (void)cp;
    return (size_t)len;
}

static const cs_udecomp_entry *cs_udecomp_find(uint32_t cp) {
    size_t lo = 0, hi = CS_UDECOMP_COUNT;
    while (lo < hi) {
        size_t mid = lo + ((hi - lo) >> 1);
        uint32_t mcp = cs_udecomp_table[mid].cp;
        if (mcp == cp) return &cs_udecomp_table[mid];
        if (mcp < cp) lo = mid + 1;
        else hi = mid;
    }
    return NULL;
}

static uint32_t cs_ucomp_find(uint32_t starter, uint32_t combiner) {
    /* Hangul algorithmic composition (UAX #15) */
    if (starter >= 0x1100U && starter <= 0x1112U && combiner >= 0x1161U && combiner <= 0x1175U) {
        uint32_t l_idx = starter - 0x1100U;
        uint32_t v_idx = combiner - 0x1161U;
        return 0xAC00U + (l_idx * 21U + v_idx) * 28U;
    }
    if (starter >= 0xAC00U && starter <= 0xD7A3U && ((starter - 0xAC00U) % 28U == 0U) &&
        combiner >= 0x11A8U && combiner <= 0x11C2U) {
        return starter + (combiner - 0x11A7U);
    }

    uint64_t key = ((uint64_t)starter << 32) | (uint64_t)combiner;
    size_t lo = 0, hi = CS_UCOMP_COUNT;
    while (lo < hi) {
        size_t mid = lo + ((hi - lo) >> 1);
        uint64_t mk = cs_ucomp_table[mid].pair;
        if (mk == key) return cs_ucomp_table[mid].composed;
        if (mk < key) lo = mid + 1;
        else hi = mid;
    }
    return 0;
}
#endif /* CS_ENABLE_UNICODE */

/* ==========================================================================
 * UTF-8 Validation, Decoding, Encoding & Maximal Subpart Repair (Section 10)
 * ========================================================================== */

/* Strict UTF-8 single sequence decoder + maximal subpart length calculator.
 * Returns true if valid sequence decoded at `p[0..avail-1]`.
 * On false, sets `*out_err_subpart_len` to the Unicode Standard Table 3-7
 * maximal subpart length (1..3) so CS_DECODE_REPLACE consumes it accurately. */
static bool cs_utf8_decode_one_raw(const uint8_t *p, size_t avail, uint32_t *out_cp,
                                   size_t *out_seq_len, size_t *out_err_subpart_len) {
    uint8_t b0 = p[0];
    if (b0 < 0x80U) {
        *out_cp = b0;
        *out_seq_len = 1U;
        return true;
    }
    if (b0 < 0xC2U || b0 > 0xF4U) {
        *out_err_subpart_len = 1U;
        return false;
    }
    if (b0 <= 0xDFU) {
        /* 2-byte sequence */
        if (avail < 2U) {
            *out_err_subpart_len = 1U;
            return false;
        }
        uint8_t b1 = p[1];
        if ((b1 & 0xC0U) != 0x80U) {
            *out_err_subpart_len = 1U;
            return false;
        }
        *out_cp = ((uint32_t)(b0 & 0x1FU) << 6) | (uint32_t)(b1 & 0x3FU);
        *out_seq_len = 2U;
        return true;
    }
    if (b0 <= 0xEFU) {
        /* 3-byte sequence */
        uint8_t b1_min = (b0 == 0xE0U) ? 0xA0U : 0x80U;
        uint8_t b1_max = (b0 == 0xEDU) ? 0x9FU : 0xBFU;
        if (avail < 2U) {
            *out_err_subpart_len = 1U;
            return false;
        }
        uint8_t b1 = p[1];
        if (b1 < b1_min || b1 > b1_max) {
            *out_err_subpart_len = 1U;
            return false;
        }
        if (avail < 3U) {
            *out_err_subpart_len = 2U;
            return false;
        }
        uint8_t b2 = p[2];
        if ((b2 & 0xC0U) != 0x80U) {
            *out_err_subpart_len = 2U;
            return false;
        }
        *out_cp = ((uint32_t)(b0 & 0x0FU) << 12) |
                  ((uint32_t)(b1 & 0x3FU) << 6)  |
                  (uint32_t)(b2 & 0x3FU);
        *out_seq_len = 3U;
        return true;
    }
    /* 4-byte sequence: F0..F4 */
    uint8_t b1_min = (b0 == 0xF0U) ? 0x90U : 0x80U;
    uint8_t b1_max = (b0 == 0xF4U) ? 0x8FU : 0xBFU;
    if (avail < 2U) {
        *out_err_subpart_len = 1U;
        return false;
    }
    uint8_t b1 = p[1];
    if (b1 < b1_min || b1 > b1_max) {
        *out_err_subpart_len = 1U;
        return false;
    }
    if (avail < 3U) {
        *out_err_subpart_len = 2U;
        return false;
    }
    uint8_t b2 = p[2];
    if ((b2 & 0xC0U) != 0x80U) {
        *out_err_subpart_len = 2U;
        return false;
    }
    if (avail < 4U) {
        *out_err_subpart_len = 3U;
        return false;
    }
    uint8_t b3 = p[3];
    if ((b3 & 0xC0U) != 0x80U) {
        *out_err_subpart_len = 3U;
        return false;
    }
    *out_cp = ((uint32_t)(b0 & 0x07U) << 18) |
              ((uint32_t)(b1 & 0x3FU) << 12) |
              ((uint32_t)(b2 & 0x3FU) << 6)  |
              (uint32_t)(b3 & 0x3FU);
    *out_seq_len = 4U;
    return true;
}

#if CS_ARCH_X86_64 && CS_ENABLE_SIMD && !CS_FORCE_SCALAR
CS_TARGET_SSE2
static cs_status cs_validate_utf8_sse2(const uint8_t *p, size_t len, size_t *out_error_offset) {
    size_t i = 0;
    while (i < len) {
        if (p[i] < 0x80U) {
            while (i + 16 <= len) {
                __m128i v = _mm_loadu_si128((const __m128i *)(const void *)(p + i));
                if (_mm_movemask_epi8(v) != 0) break;
                i += 16;
            }
            while (i < len && p[i] < 0x80U) i++;
            if (i >= len) break;
        }
        uint32_t cp = 0;
        size_t seq_len = 0, err_len = 0;
        if (!cs_utf8_decode_one_raw(p + i, len - i, &cp, &seq_len, &err_len)) {
            if (out_error_offset) *out_error_offset = i;
            return CS_EUTF8;
        }
        i += seq_len;
    }
    if (out_error_offset) *out_error_offset = len;
    return CS_OK;
}

CS_TARGET_AVX2
static cs_status cs_validate_utf8_avx2(const uint8_t *p, size_t len, size_t *out_error_offset) {
    size_t i = 0;
    while (i < len) {
        if (p[i] < 0x80U) {
            while (i + 32 <= len) {
                __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)(p + i));
                if (_mm256_movemask_epi8(v) != 0) break;
                i += 32;
            }
            while (i < len && p[i] < 0x80U) i++;
            if (i >= len) break;
        }
        uint32_t cp = 0;
        size_t seq_len = 0, err_len = 0;
        if (!cs_utf8_decode_one_raw(p + i, len - i, &cp, &seq_len, &err_len)) {
            if (out_error_offset) *out_error_offset = i;
            return CS_EUTF8;
        }
        i += seq_len;
    }
    if (out_error_offset) *out_error_offset = len;
    return CS_OK;
}
#endif

cs_status cs_validate_utf8(cs_view v, size_t *out_error_offset) {
    if (!cs_view_is_valid(v)) return CS_EINVAL;
    const uint8_t *p = (const uint8_t *)v.data;
    size_t len = v.len;

    cs_backend bk = cs_active_backend();
    (void)bk;
#if CS_ARCH_X86_64 && CS_ENABLE_SIMD && !CS_FORCE_SCALAR
    if ((bk == CS_BACKEND_AVX2 || bk == CS_BACKEND_AVX512) && len >= 32) {
        return cs_validate_utf8_avx2(p, len, out_error_offset);
    }
    if ((bk == CS_BACKEND_SSE2 || bk == CS_BACKEND_AVX2 || bk == CS_BACKEND_AVX512) && len >= 16) {
        return cs_validate_utf8_sse2(p, len, out_error_offset);
    }
#endif
    size_t i = 0;
    while (i < len) {
        if (p[i] < 0x80U) {
            i++;
            continue;
        }
        uint32_t cp = 0;
        size_t seq_len = 0, err_len = 0;
        if (!cs_utf8_decode_one_raw(p + i, len - i, &cp, &seq_len, &err_len)) {
            if (out_error_offset) *out_error_offset = i;
            return CS_EUTF8;
        }
        i += seq_len;
    }
    if (out_error_offset) *out_error_offset = len;
    return CS_OK;
}

cs_status cs_decode_utf8_next(cs_view v, size_t offset, uint32_t *out_cp, size_t *out_next_offset) {
    if (!cs_view_is_valid(v)) return CS_EINVAL;
    if (offset > v.len) return CS_ERANGE;
    if (offset == v.len) return CS_END;

    const uint8_t *p = (const uint8_t *)v.data + offset;
    uint32_t cp = 0;
    size_t seq_len = 0, err_len = 0;
    if (!cs_utf8_decode_one_raw(p, v.len - offset, &cp, &seq_len, &err_len)) {
        return CS_EUTF8;
    }
    if (out_cp) *out_cp = cp;
    if (out_next_offset) *out_next_offset = offset + seq_len;
    return CS_OK;
}

cs_status cs_decode_utf8_prev(cs_view v, size_t offset, uint32_t *out_cp, size_t *out_prev_offset) {
    if (!cs_view_is_valid(v)) return CS_EINVAL;
    if (offset > v.len) return CS_ERANGE;
    if (offset == 0) return CS_END;

    const uint8_t *base = (const uint8_t *)v.data;
    /* If offset < v.len and base[offset] is a continuation byte (0x80..0xBF),
     * then `offset` was inside the middle of a multi-byte sequence! */
    if (offset < v.len && (base[offset] & 0xC0U) == 0x80U) {
        /* Verify if there is a valid lead byte before offset that spans across offset */
        size_t probe = offset;
        size_t back = 0;
        while (probe > 0 && back < 3U && (base[probe - 1U] & 0xC0U) == 0x80U) {
            probe--;
            back++;
        }
        if (probe > 0) {
            uint32_t tmp_cp = 0;
            size_t tmp_len = 0, tmp_err = 0;
            if (cs_utf8_decode_one_raw(base + probe - 1U, v.len - (probe - 1U), &tmp_cp, &tmp_len, &tmp_err)) {
                if ((probe - 1U) + tmp_len > offset) {
                    return CS_EINVAL; /* Caller passed mid-codepoint offset */
                }
            }
        }
    }

    /* Step back up to 4 bytes to find the lead byte */
    size_t start = offset - 1U;
    size_t steps = 0;
    while (start > 0 && steps < 3U && (base[start] & 0xC0U) == 0x80U) {
        start--;
        steps++;
    }

    uint32_t cp = 0;
    size_t seq_len = 0, err_len = 0;
    if (!cs_utf8_decode_one_raw(base + start, offset - start, &cp, &seq_len, &err_len) ||
        start + seq_len != offset) {
        return CS_EUTF8;
    }
    if (out_cp) *out_cp = cp;
    if (out_prev_offset) *out_prev_offset = start;
    return CS_OK;
}

cs_status cs_encode_utf8_codepoint(uint32_t cp, char out_buf[4], size_t *out_len) {
    if (!out_buf) return CS_EINVAL;
    if (cp > 0x10FFFFU || (cp >= 0xD800U && cp <= 0xDFFFU)) {
        return CS_EUTF8;
    }
    if (cp < 0x80U) {
        out_buf[0] = (char)cp;
        if (out_len) *out_len = 1U;
    } else if (cp < 0x800U) {
        out_buf[0] = (char)(0xC0U | (cp >> 6));
        out_buf[1] = (char)(0x80U | (cp & 0x3FU));
        if (out_len) *out_len = 2U;
    } else if (cp < 0x10000U) {
        out_buf[0] = (char)(0xE0U | (cp >> 12));
        out_buf[1] = (char)(0x80U | ((cp >> 6) & 0x3FU));
        out_buf[2] = (char)(0x80U | (cp & 0x3FU));
        if (out_len) *out_len = 3U;
    } else {
        out_buf[0] = (char)(0xF0U | (cp >> 18));
        out_buf[1] = (char)(0x80U | ((cp >> 12) & 0x3FU));
        out_buf[2] = (char)(0x80U | ((cp >> 6) & 0x3FU));
        out_buf[3] = (char)(0x80U | (cp & 0x3FU));
        if (out_len) *out_len = 4U;
    }
    return CS_OK;
}

cs_status cs_append_codepoint(cs_string *s, uint32_t cp) {
    if (!s) return CS_EINVAL;
    char buf[4];
    size_t len = 0;
    cs_status st = cs_encode_utf8_codepoint(cp, buf, &len);
    if (st != CS_OK) return st;
    cs_view v = { buf, len };
    return cs_append(s, v);
}

cs_status cs_buffer_append_codepoint(cs_buffer *b, uint32_t cp, size_t *out_required) {
    if (!b) return CS_EINVAL;
    char buf[4];
    size_t len = 0;
    cs_status st = cs_encode_utf8_codepoint(cp, buf, &len);
    if (st != CS_OK) return st;
    cs_view v = { buf, len };
    return cs_buffer_append(b, v, out_required);
}

cs_status cs_builder_append_codepoint(cs_builder *b, uint32_t cp) {
    if (!b) return CS_EINVAL;
    char buf[4];
    size_t len = 0;
    cs_status st = cs_encode_utf8_codepoint(cp, buf, &len);
    if (st != CS_OK) {
        b->sticky_err = st;
        return st;
    }
    cs_view v = { buf, len };
    return cs_builder_append(b, v);
}

/* UTF-8 Repair (replaces each maximal subpart with U+FFFD = EF BF BD) */
cs_status cs_repair_utf8(cs_string *dst, cs_view v) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;

    const uint8_t *p = (const uint8_t *)v.data;
    size_t i = 0;
    while (i < v.len) {
        uint32_t cp = 0;
        size_t seq_len = 0, err_len = 0;
        if (cs_utf8_decode_one_raw(p + i, v.len - i, &cp, &seq_len, &err_len)) {
            cs_view seg = { (const char *)(p + i), seq_len };
            st = cs_append(&tmp, seg);
            if (st != CS_OK) { cs_destroy(&tmp); return st; }
            i += seq_len;
        } else {
            st = cs_append(&tmp, CS_LIT("\xEF\xBF\xBD"));
            if (st != CS_OK) { cs_destroy(&tmp); return st; }
            i += err_len;
        }
    }
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
}

cs_status cs_repair_utf8_buffer(cs_buffer *dst, cs_view v, size_t *out_required) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
    const uint8_t *p = (const uint8_t *)v.data;
    size_t i = 0;
    size_t total_out = 0;
    while (i < v.len) {
        uint32_t cp = 0;
        size_t seq_len = 0, err_len = 0;
        if (cs_utf8_decode_one_raw(p + i, v.len - i, &cp, &seq_len, &err_len)) {
            if (cs_add_overflow_sz(total_out, seq_len, &total_out)) return CS_EOVERFLOW;
            i += seq_len;
        } else {
            if (cs_add_overflow_sz(total_out, 3U, &total_out)) return CS_EOVERFLOW;
            i += err_len;
        }
    }
    size_t req_with_nul = 0;
    if (cs_add_overflow_sz(total_out, 1U, &req_with_nul)) return CS_EOVERFLOW;
    if (out_required) *out_required = req_with_nul;
    if (total_out > dst->cap) return CS_EBUFFER;

    size_t out_pos = 0;
    i = 0;
    while (i < v.len) {
        uint32_t cp = 0;
        size_t seq_len = 0, err_len = 0;
        if (cs_utf8_decode_one_raw(p + i, v.len - i, &cp, &seq_len, &err_len)) {
            memcpy(dst->data + out_pos, p + i, seq_len);
            out_pos += seq_len;
            i += seq_len;
        } else {
            memcpy(dst->data + out_pos, "\xEF\xBF\xBD", 3U);
            out_pos += 3U;
            i += err_len;
        }
    }
    dst->len = out_pos;
    dst->data[dst->len] = '\0';
    return CS_OK;
}

/* ==========================================================================
 * Code Point Iterator (`cs_cp_iter`)
 * ========================================================================== */

void cs_cp_iter_init(cs_cp_iter *it, cs_view v, cs_decode_mode mode) {
    if (!it) return;
    it->v    = v;
    it->pos  = 0;
    it->mode = mode;
}

cs_status cs_cp_iter_next(cs_cp_iter *it, uint32_t *out_cp, size_t *out_start, size_t *out_end) {
    if (!it || !cs_view_is_valid(it->v)) return CS_EINVAL;
    if (it->pos >= it->v.len) return CS_END;

    const uint8_t *p = (const uint8_t *)it->v.data + it->pos;
    size_t avail = it->v.len - it->pos;
    uint32_t cp = 0;
    size_t seq_len = 0, err_len = 0;
    if (cs_utf8_decode_one_raw(p, avail, &cp, &seq_len, &err_len)) {
        size_t start = it->pos;
        it->pos += seq_len;
        if (out_cp)    *out_cp    = cp;
        if (out_start) *out_start = start;
        if (out_end)   *out_end   = it->pos;
        return CS_OK;
    }
    if (it->mode == CS_DECODE_REPLACE) {
        size_t start = it->pos;
        it->pos += err_len;
        if (out_cp)    *out_cp    = 0xFFFDU;
        if (out_start) *out_start = start;
        if (out_end)   *out_end   = it->pos;
        return CS_OK;
    }
    return CS_EUTF8;
}

cs_status cs_cp_iter_prev(cs_cp_iter *it, uint32_t *out_cp, size_t *out_start, size_t *out_end) {
    if (!it || !cs_view_is_valid(it->v)) return CS_EINVAL;
    if (it->pos == 0 || it->pos > it->v.len) return CS_END;

    size_t end_pos = it->pos;
    uint32_t cp = 0;
    size_t prev_off = 0;
    cs_status st = cs_decode_utf8_prev(it->v, end_pos, &cp, &prev_off);
    if (st == CS_OK) {
        it->pos = prev_off;
        if (out_cp)    *out_cp    = cp;
        if (out_start) *out_start = prev_off;
        if (out_end)   *out_end   = end_pos;
        return CS_OK;
    }
    if (it->mode == CS_DECODE_REPLACE) {
        it->pos = end_pos - 1U;
        if (out_cp)    *out_cp    = 0xFFFDU;
        if (out_start) *out_start = it->pos;
        if (out_end)   *out_end   = end_pos;
        return CS_OK;
    }
    return st;
}

/* ==========================================================================
 * UTF-8 <-> UTF-16 & UTF-32 Conversions (Section 10)
 * ========================================================================== */

cs_status cs_utf8_to_utf16(cs_view v, uint16_t *out_units, size_t max_units, size_t *out_len) {
    if (!cs_view_is_valid(v)) return CS_EINVAL;
    size_t needed = 0;
    cs_cp_iter it;
    cs_cp_iter_init(&it, v, CS_DECODE_STRICT);
    uint32_t cp = 0;
    cs_status st;
    while ((st = cs_cp_iter_next(&it, &cp, NULL, NULL)) == CS_OK) {
        size_t ulen = (cp > 0xFFFFU) ? 2U : 1U;
        if (cs_add_overflow_sz(needed, ulen, &needed)) return CS_EOVERFLOW;
    }
    if (st != CS_END) return st;
    if (out_len) *out_len = needed;
    if (!out_units) return CS_OK;
    if (needed > max_units) return CS_EBUFFER;

    size_t pos = 0;
    cs_cp_iter_init(&it, v, CS_DECODE_STRICT);
    while (cs_cp_iter_next(&it, &cp, NULL, NULL) == CS_OK) {
        if (cp <= 0xFFFFU) {
            out_units[pos++] = (uint16_t)cp;
        } else {
            uint32_t u = cp - 0x10000U;
            out_units[pos++] = (uint16_t)(0xD800U | (u >> 10));
            out_units[pos++] = (uint16_t)(0xDC00U | (u & 0x3FFU));
        }
    }
    return CS_OK;
}

cs_status cs_utf16_to_utf8(const uint16_t *units, size_t unit_len, cs_string *dst) {
    if (!dst || (!units && unit_len > 0)) return CS_EINVAL;
    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;

    size_t i = 0;
    while (i < unit_len) {
        uint16_t u1 = units[i++];
        uint32_t cp = u1;
        if (u1 >= 0xD800U && u1 <= 0xDBFFU) {
            if (i >= unit_len) { cs_destroy(&tmp); return CS_EUTF8; }
            uint16_t u2 = units[i++];
            if (u2 < 0xDC00U || u2 > 0xDFFFU) { cs_destroy(&tmp); return CS_EUTF8; }
            cp = 0x10000U + (((uint32_t)(u1 - 0xD800U) << 10) | (uint32_t)(u2 - 0xDC00U));
        } else if (u1 >= 0xDC00U && u1 <= 0xDFFFU) {
            cs_destroy(&tmp);
            return CS_EUTF8;
        }
        st = cs_append_codepoint(&tmp, cp);
        if (st != CS_OK) { cs_destroy(&tmp); return st; }
    }
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
}

static inline void cs_write_u16_endian(uint8_t out[2], uint16_t v, cs_endian endian) {
    if (endian == CS_ENDIAN_LITTLE) {
        out[0] = (uint8_t)(v & 0xFFU);
        out[1] = (uint8_t)(v >> 8);
    } else {
        out[0] = (uint8_t)(v >> 8);
        out[1] = (uint8_t)(v & 0xFFU);
    }
}

static inline uint16_t cs_read_u16_endian(const uint8_t in[2], cs_endian endian) {
    if (endian == CS_ENDIAN_LITTLE) {
        return (uint16_t)in[0] | ((uint16_t)in[1] << 8);
    } else {
        return ((uint16_t)in[0] << 8) | (uint16_t)in[1];
    }
}

cs_status cs_utf8_to_utf16_bytes(cs_view v, cs_endian endian, cs_bom_policy bom, cs_string *dst_bytes) {
    if (!dst_bytes || !cs_view_is_valid(v)) return CS_EINVAL;
    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst_bytes->alloc);
    if (st != CS_OK) return st;

    if (bom == CS_BOM_EMIT) {
        uint8_t b[2];
        cs_write_u16_endian(b, 0xFEFFU, endian);
        st = cs_append(&tmp, cs_sv_from_bytes((const char *)b, 2U));
        if (st != CS_OK) { cs_destroy(&tmp); return st; }
    }

    cs_cp_iter it;
    cs_cp_iter_init(&it, v, CS_DECODE_STRICT);
    uint32_t cp = 0;
    while ((st = cs_cp_iter_next(&it, &cp, NULL, NULL)) == CS_OK) {
        if (cp <= 0xFFFFU) {
            uint8_t b[2];
            cs_write_u16_endian(b, (uint16_t)cp, endian);
            st = cs_append(&tmp, cs_sv_from_bytes((const char *)b, 2U));
            if (st != CS_OK) { cs_destroy(&tmp); return st; }
        } else {
            uint32_t u = cp - 0x10000U;
            uint8_t b[4];
            cs_write_u16_endian(b,     (uint16_t)(0xD800U | (u >> 10)),    endian);
            cs_write_u16_endian(b + 2, (uint16_t)(0xDC00U | (u & 0x3FFU)), endian);
            st = cs_append(&tmp, cs_sv_from_bytes((const char *)b, 4U));
            if (st != CS_OK) { cs_destroy(&tmp); return st; }
        }
    }
    if (st != CS_END) { cs_destroy(&tmp); return st; }
    cs_swap(dst_bytes, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
}

cs_status cs_utf16_bytes_to_utf8(cs_view raw_bytes, cs_endian default_endian, cs_bom_policy bom, cs_string *dst) {
    if (!dst || !cs_view_is_valid(raw_bytes)) return CS_EINVAL;
    if ((raw_bytes.len & 1U) != 0U) return CS_EUTF8;

    const uint8_t *p = (const uint8_t *)raw_bytes.data;
    size_t len = raw_bytes.len;
    cs_endian endian = default_endian;

    if (bom == CS_BOM_STRIP_AUTO && len >= 2U) {
        if (p[0] == 0xFFU && p[1] == 0xFEU) {
            endian = CS_ENDIAN_LITTLE;
            p += 2;
            len -= 2;
        } else if (p[0] == 0xFEU && p[1] == 0xFFU) {
            endian = CS_ENDIAN_BIG;
            p += 2;
            len -= 2;
        }
    }

    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;

    size_t i = 0;
    while (i < len) {
        uint16_t u1 = cs_read_u16_endian(p + i, endian);
        i += 2;
        uint32_t cp = u1;
        if (u1 >= 0xD800U && u1 <= 0xDBFFU) {
            if (i >= len) { cs_destroy(&tmp); return CS_EUTF8; }
            uint16_t u2 = cs_read_u16_endian(p + i, endian);
            i += 2;
            if (u2 < 0xDC00U || u2 > 0xDFFFU) { cs_destroy(&tmp); return CS_EUTF8; }
            cp = 0x10000U + (((uint32_t)(u1 - 0xD800U) << 10) | (uint32_t)(u2 - 0xDC00U));
        } else if (u1 >= 0xDC00U && u1 <= 0xDFFFU) {
            cs_destroy(&tmp);
            return CS_EUTF8;
        }
        st = cs_append_codepoint(&tmp, cp);
        if (st != CS_OK) { cs_destroy(&tmp); return st; }
    }
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
}

cs_status cs_utf8_to_utf32(cs_view v, uint32_t *out_cps, size_t max_cps, size_t *out_len) {
    if (!cs_view_is_valid(v)) return CS_EINVAL;
    size_t count = 0;
    cs_status st = cs_len_codepoints(v, &count);
    if (st != CS_OK) return st;
    if (out_len) *out_len = count;
    if (!out_cps) return CS_OK;
    if (count > max_cps) return CS_EBUFFER;

    size_t pos = 0;
    cs_cp_iter it;
    cs_cp_iter_init(&it, v, CS_DECODE_STRICT);
    uint32_t cp = 0;
    while (cs_cp_iter_next(&it, &cp, NULL, NULL) == CS_OK) {
        out_cps[pos++] = cp;
    }
    return CS_OK;
}

cs_status cs_utf32_to_utf8(const uint32_t *cps, size_t cp_len, cs_string *dst) {
    if (!dst || (!cps && cp_len > 0)) return CS_EINVAL;
    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;

    for (size_t i = 0; i < cp_len; i++) {
        st = cs_append_codepoint(&tmp, cps[i]);
        if (st != CS_OK) { cs_destroy(&tmp); return st; }
    }
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
}

static inline void cs_write_u32_endian(uint8_t out[4], uint32_t v, cs_endian endian) {
    if (endian == CS_ENDIAN_LITTLE) {
        out[0] = (uint8_t)(v & 0xFFU);
        out[1] = (uint8_t)((v >> 8) & 0xFFU);
        out[2] = (uint8_t)((v >> 16) & 0xFFU);
        out[3] = (uint8_t)((v >> 24) & 0xFFU);
    } else {
        out[0] = (uint8_t)((v >> 24) & 0xFFU);
        out[1] = (uint8_t)((v >> 16) & 0xFFU);
        out[2] = (uint8_t)((v >> 8) & 0xFFU);
        out[3] = (uint8_t)(v & 0xFFU);
    }
}

static inline uint32_t cs_read_u32_endian(const uint8_t in[4], cs_endian endian) {
    if (endian == CS_ENDIAN_LITTLE) {
        return (uint32_t)in[0] | ((uint32_t)in[1] << 8) | ((uint32_t)in[2] << 16) | ((uint32_t)in[3] << 24);
    } else {
        return ((uint32_t)in[0] << 24) | ((uint32_t)in[1] << 16) | ((uint32_t)in[2] << 8) | (uint32_t)in[3];
    }
}

cs_status cs_utf8_to_utf32_bytes(cs_view v, cs_endian endian, cs_bom_policy bom, cs_string *dst_bytes) {
    if (!dst_bytes || !cs_view_is_valid(v)) return CS_EINVAL;
    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst_bytes->alloc);
    if (st != CS_OK) return st;

    if (bom == CS_BOM_EMIT) {
        uint8_t b[4];
        cs_write_u32_endian(b, 0x0000FEFFU, endian);
        st = cs_append(&tmp, cs_sv_from_bytes((const char *)b, 4U));
        if (st != CS_OK) { cs_destroy(&tmp); return st; }
    }

    cs_cp_iter it;
    cs_cp_iter_init(&it, v, CS_DECODE_STRICT);
    uint32_t cp = 0;
    while ((st = cs_cp_iter_next(&it, &cp, NULL, NULL)) == CS_OK) {
        uint8_t b[4];
        cs_write_u32_endian(b, cp, endian);
        st = cs_append(&tmp, cs_sv_from_bytes((const char *)b, 4U));
        if (st != CS_OK) { cs_destroy(&tmp); return st; }
    }
    if (st != CS_END) { cs_destroy(&tmp); return st; }
    cs_swap(dst_bytes, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
}

cs_status cs_utf32_bytes_to_utf8(cs_view raw_bytes, cs_endian default_endian, cs_bom_policy bom, cs_string *dst) {
    if (!dst || !cs_view_is_valid(raw_bytes)) return CS_EINVAL;
    if ((raw_bytes.len & 3U) != 0U) return CS_EUTF8;

    const uint8_t *p = (const uint8_t *)raw_bytes.data;
    size_t len = raw_bytes.len;
    cs_endian endian = default_endian;

    if (bom == CS_BOM_STRIP_AUTO && len >= 4U) {
        if (p[0] == 0xFFU && p[1] == 0xFEU && p[2] == 0x00U && p[3] == 0x00U) {
            endian = CS_ENDIAN_LITTLE;
            p += 4;
            len -= 4;
        } else if (p[0] == 0x00U && p[1] == 0x00U && p[2] == 0xFEU && p[3] == 0xFFU) {
            endian = CS_ENDIAN_BIG;
            p += 4;
            len -= 4;
        }
    }

    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;

    for (size_t i = 0; i < len; i += 4U) {
        uint32_t cp = cs_read_u32_endian(p + i, endian);
        st = cs_append_codepoint(&tmp, cp);
        if (st != CS_OK) { cs_destroy(&tmp); return st; }
    }
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
}

#ifdef _WIN32
cs_status cs_from_win32_wide(cs_string *dst, const wchar_t *wstr, size_t wlen) {
    return cs_utf16_to_utf8((const uint16_t *)wstr, wlen, dst);
}

cs_status cs_to_win32_wide(cs_view v, wchar_t *out_wbuf, size_t max_wchars, size_t *out_wlen) {
    return cs_utf8_to_utf16(v, (uint16_t *)out_wbuf, max_wchars, out_wlen);
}
#endif

/* ==========================================================================
 * Code Point & String Property Predicates (Section 10)
 * ========================================================================== */

cs_general_category cs_cp_category(uint32_t cp) {
#if CS_ENABLE_UNICODE
    if (cp > 0x10FFFFU) return CS_GC_CN;
    return (cs_general_category)cs_uprop_lookup(cp)->gc;
#else
    if (cp >= 'A' && cp <= 'Z') return CS_GC_LU;
    if (cp >= 'a' && cp <= 'z') return CS_GC_LL;
    if (cp >= '0' && cp <= '9') return CS_GC_ND;
    if (cp == ' ') return CS_GC_ZS;
    if (cp < 0x20U || cp == 0x7FU) return CS_GC_CC;
    return CS_GC_CN;
#endif
}

bool cs_cp_is_ascii(uint32_t cp) {
    return cp < 0x80U;
}

bool cs_cp_is_alphabetic(uint32_t cp) {
#if CS_ENABLE_UNICODE
    if (cp > 0x10FFFFU) return false;
    return (cs_uprop_lookup(cp)->flags & CS_UF_ALPHA) != 0;
#else
    return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
#endif
}

bool cs_cp_is_digit(uint32_t cp) {
    return cs_cp_category(cp) == CS_GC_ND;
}

bool cs_cp_is_whitespace(uint32_t cp) {
#if CS_ENABLE_UNICODE
    if (cp > 0x10FFFFU) return false;
    return (cs_uprop_lookup(cp)->flags & CS_UF_WSPACE) != 0;
#else
    return cp == ' ' || (cp >= 0x09U && cp <= 0x0DU);
#endif
}

bool cs_cp_is_upper(uint32_t cp) {
#if CS_ENABLE_UNICODE
    if (cp > 0x10FFFFU) return false;
    return (cs_uprop_lookup(cp)->flags & CS_UF_UPPER) != 0;
#else
    return cp >= 'A' && cp <= 'Z';
#endif
}

bool cs_cp_is_lower(uint32_t cp) {
#if CS_ENABLE_UNICODE
    if (cp > 0x10FFFFU) return false;
    return (cs_uprop_lookup(cp)->flags & CS_UF_LOWER) != 0;
#else
    return cp >= 'a' && cp <= 'z';
#endif
}

bool cs_cp_is_title(uint32_t cp) {
    return cs_cp_category(cp) == CS_GC_LT;
}

bool cs_cp_is_alnum(uint32_t cp) {
    if (cs_cp_is_alphabetic(cp)) return true;
    cs_general_category gc = cs_cp_category(cp);
    return gc == CS_GC_ND || gc == CS_GC_NL || gc == CS_GC_NO;
}

bool cs_cp_is_control(uint32_t cp) {
    return cs_cp_category(cp) == CS_GC_CC;
}

bool cs_cp_is_printable(uint32_t cp) {
    if (cp == 0x20U) return true;
    cs_general_category gc = cs_cp_category(cp);
    return (gc >= CS_GC_LU && gc <= CS_GC_SO);
}

bool cs_is_ascii(cs_view v) {
    if (!cs_view_is_valid(v)) return false;
    if (v.len == 0) return true;
    return cs_dispatch_is_ascii((const uint8_t *)v.data, v.len);
}

static cs_status cs_eval_string_pred(cs_view v, bool (*pred)(uint32_t), bool *out_result) {
    if (!cs_view_is_valid(v) || !out_result) return CS_EINVAL;
    if (v.len == 0) {
        *out_result = false;
        return CS_OK;
    }
    cs_cp_iter it;
    cs_cp_iter_init(&it, v, CS_DECODE_STRICT);
    uint32_t cp = 0;
    cs_status st;
    bool all_ok = true;
    while ((st = cs_cp_iter_next(&it, &cp, NULL, NULL)) == CS_OK) {
        if (!pred(cp)) {
            all_ok = false;
            /* Continue scanning to validate remaining UTF-8 strictly */
        }
    }
    if (st != CS_END) return st;
    *out_result = all_ok;
    return CS_OK;
}

cs_status cs_is_alphabetic(cs_view v, bool *out_result) { return cs_eval_string_pred(v, cs_cp_is_alphabetic, out_result); }
cs_status cs_is_digit(cs_view v, bool *out_result)      { return cs_eval_string_pred(v, cs_cp_is_digit, out_result); }
cs_status cs_is_whitespace(cs_view v, bool *out_result) { return cs_eval_string_pred(v, cs_cp_is_whitespace, out_result); }
cs_status cs_is_alnum(cs_view v, bool *out_result)      { return cs_eval_string_pred(v, cs_cp_is_alnum, out_result); }
cs_status cs_is_upper(cs_view v, bool *out_result)      { return cs_eval_string_pred(v, cs_cp_is_upper, out_result); }
cs_status cs_is_lower(cs_view v, bool *out_result)      { return cs_eval_string_pred(v, cs_cp_is_lower, out_result); }
cs_status cs_is_printable(cs_view v, bool *out_result)  { return cs_eval_string_pred(v, cs_cp_is_printable, out_result); }

/* ==========================================================================
 * UAX #29 Segmentation: Grapheme, Word, and Sentence Iterators
 * ========================================================================== */

#if CS_ENABLE_UNICODE
static inline uint8_t cs_get_gcb_incb(uint32_t cp, bool *out_extpict, uint8_t *out_incb) {
    const cs_uprop_record *r = cs_uprop_lookup(cp);
    if (out_extpict) *out_extpict = (r->flags & CS_UF_EXTPICT) != 0;
    if (out_incb)    *out_incb    = r->incb;
    return r->gcb;
}

/* Determine if a grapheme break occurs between prev (with state) and next code point */
static bool cs_is_grapheme_break(uint8_t prev_gcb, bool prev_extpict_zwj, bool prev_in_extpict,
                                 uint8_t incb_state, size_t ri_run_count,
                                 uint8_t next_gcb, bool next_extpict, uint8_t next_incb) {
    /* GB3: CR x LF */
    if (prev_gcb == CS_GCB_CR && next_gcb == CS_GCB_LF) return false;
    /* GB4: (Control | CR | LF) ÷ */
    if (prev_gcb == CS_GCB_CONTROL || prev_gcb == CS_GCB_CR || prev_gcb == CS_GCB_LF) return true;
    /* GB5: ÷ (Control | CR | LF) */
    if (next_gcb == CS_GCB_CONTROL || next_gcb == CS_GCB_CR || next_gcb == CS_GCB_LF) return true;
    /* GB6: L x (L | V | LV | LVT) */
    if (prev_gcb == CS_GCB_L &&
        (next_gcb == CS_GCB_L || next_gcb == CS_GCB_V || next_gcb == CS_GCB_LV || next_gcb == CS_GCB_LVT)) {
        return false;
    }
    /* GB7: (LV | V) x (V | T) */
    if ((prev_gcb == CS_GCB_LV || prev_gcb == CS_GCB_V) &&
        (next_gcb == CS_GCB_V || next_gcb == CS_GCB_T)) {
        return false;
    }
    /* GB8: (LVT | T) x T */
    if ((prev_gcb == CS_GCB_LVT || prev_gcb == CS_GCB_T) && next_gcb == CS_GCB_T) {
        return false;
    }
    /* GB9: x (Extend | ZWJ) */
    if (next_gcb == CS_GCB_EXTEND || next_gcb == CS_GCB_ZWJ) return false;
    /* GB9a: x SpacingMark */
    if (next_gcb == CS_GCB_SPACINGMARK) return false;
    /* GB9b: Prepend x */
    if (prev_gcb == CS_GCB_PREPEND) return false;
    /* GB9c: \p{InCB=Consonant} [\p{InCB=Extend}\p{InCB=Linker}]* \p{InCB=Linker} [\p{InCB=Extend}\p{InCB=Linker}]* x \p{InCB=Consonant} */
    if (incb_state == 2U && next_incb == CS_INCB_CONSONANT) return false;
    /* GB11: \p{Extended_Pictographic} Extend* ZWJ x \p{Extended_Pictographic} */
    if (prev_extpict_zwj && next_extpict) return false;
    /* GB12 / GB13: sot (RI RI)* RI x RI */
    if (prev_gcb == CS_GCB_RI && next_gcb == CS_GCB_RI && (ri_run_count & 1U) == 1U) {
        return false;
    }
    (void)prev_in_extpict;
    /* GB999: Any ÷ Any */
    return true;
}
#endif /* CS_ENABLE_UNICODE */

void cs_grapheme_iter_init(cs_grapheme_iter *it, cs_view v, cs_decode_mode mode) {
    if (!it) return;
    it->v    = v;
    it->pos  = 0;
    it->mode = mode;
}

cs_status cs_grapheme_iter_next(cs_grapheme_iter *it, cs_view *out_cluster, size_t *out_start, size_t *out_end) {
    if (!it || !cs_view_is_valid(it->v)) return CS_EINVAL;
    if (it->pos >= it->v.len) return CS_END;

    size_t start_pos = it->pos;
    cs_cp_iter cpi;
    cpi.v    = it->v;
    cpi.pos  = it->pos;
    cpi.mode = it->mode;

    uint32_t cp = 0;
    size_t cp_start = 0, cp_end = 0;
    cs_status st = cs_cp_iter_next(&cpi, &cp, &cp_start, &cp_end);
    if (st != CS_OK) return st;

#if CS_ENABLE_UNICODE
    bool extpict = false;
    uint8_t first_incb = 0;
    uint8_t prev_gcb = cs_get_gcb_incb(cp, &extpict, &first_incb);
    bool in_extpict = extpict;
    bool extpict_zwj = false;
    uint8_t incb_state = (first_incb == CS_INCB_CONSONANT) ? 1U : 0U;
    size_t ri_run = (prev_gcb == CS_GCB_RI) ? 1U : 0U;

    size_t cluster_end = cp_end;
    while (cpi.pos < cpi.v.len) {
        size_t save_pos = cpi.pos;
        uint32_t next_cp = 0;
        size_t n_start = 0, n_end = 0;
        st = cs_cp_iter_next(&cpi, &next_cp, &n_start, &n_end);
        if (st != CS_OK) return st;

        bool next_extpict = false;
        uint8_t next_incb = 0;
        uint8_t next_gcb = cs_get_gcb_incb(next_cp, &next_extpict, &next_incb);

        if (cs_is_grapheme_break(prev_gcb, extpict_zwj, in_extpict, incb_state, ri_run,
                                 next_gcb, next_extpict, next_incb)) {
            cpi.pos = save_pos;
            break;
        }

        cluster_end = n_end;
        /* Update GB9c (InCB) state */
        if (next_incb == CS_INCB_CONSONANT) {
            incb_state = 1U;
        } else if (next_incb == CS_INCB_LINKER) {
            if (incb_state >= 1U) incb_state = 2U;
        } else if (next_incb == CS_INCB_EXTEND) {
            /* incb_state stays 1U or 2U if >= 1U, else 0U */
        } else {
            incb_state = 0U;
        }
        /* Update GB11 & GB12/13 state */
        if (next_gcb == CS_GCB_EXTEND) {
            /* in_extpict stays true if already in_extpict; extpict_zwj clears */
            extpict_zwj = false;
        } else if (next_gcb == CS_GCB_ZWJ) {
            extpict_zwj = in_extpict;
        } else {
            in_extpict  = next_extpict;
            extpict_zwj = false;
        }
        if (next_gcb == CS_GCB_RI) {
            ri_run++;
        } else {
            ri_run = 0;
        }
        prev_gcb = next_gcb;
    }
    it->pos = cluster_end;
#else
    it->pos = cp_end;
#endif

    if (out_cluster) {
        out_cluster->data = it->v.data + start_pos;
        out_cluster->len  = it->pos - start_pos;
    }
    if (out_start) *out_start = start_pos;
    if (out_end)   *out_end   = it->pos;
    return CS_OK;
}

cs_status cs_grapheme_iter_prev(cs_grapheme_iter *it, cs_view *out_cluster, size_t *out_start, size_t *out_end) {
    if (!it || !cs_view_is_valid(it->v)) return CS_EINVAL;
    if (it->pos == 0 || it->pos > it->v.len) return CS_END;

    /* Scan forward over the prefix [0, it->pos) to find the exact last grapheme boundary
     * (guarantees 100% consistency with forward GB11/GB12/GB13 state rules). */
    cs_grapheme_iter fwd;
    cs_view prefix = { it->v.data, it->pos };
    cs_grapheme_iter_init(&fwd, prefix, it->mode);
    cs_view last_cl = cs_empty_view();
    size_t last_s = 0, last_e = 0;
    cs_status st;
    while ((st = cs_grapheme_iter_next(&fwd, &last_cl, &last_s, &last_e)) == CS_OK) {
        /* advance */
    }
    if (st != CS_END) return st;
    it->pos = last_s;
    if (out_cluster) *out_cluster = last_cl;
    if (out_start)   *out_start   = last_s;
    if (out_end)     *out_end     = last_e;
    return CS_OK;
}

/* Word Segmentation Iterator (UAX #29 WB1..WB999) */
void cs_word_iter_init(cs_word_iter *it, cs_view v, cs_decode_mode mode) {
    if (!it) return;
    it->v    = v;
    it->pos  = 0;
    it->mode = mode;
}

#if CS_ENABLE_UNICODE
static inline bool cs_wb_is_ahletter(uint8_t wb) {
    return wb == CS_WB_ALETTER || wb == CS_WB_HEBREW_LETTER;
}
static inline bool cs_wb_is_midnumletq(uint8_t wb) {
    return wb == CS_WB_MIDNUMLET || wb == CS_WB_SINGLE_QUOTE;
}
static inline bool cs_wb_is_ignore(uint8_t wb) {
    return wb == CS_WB_EXTEND || wb == CS_WB_FORMAT || wb == CS_WB_ZWJ;
}

/* Peek next non-ignored WB property after byte offset `after_pos` */
static uint8_t cs_wb_peek_next_non_ignored(cs_view v, size_t after_pos, cs_decode_mode mode) {
    cs_cp_iter it;
    it.v    = v;
    it.pos  = after_pos;
    it.mode = mode;
    uint32_t cp = 0;
    while (cs_cp_iter_next(&it, &cp, NULL, NULL) == CS_OK) {
        uint8_t wb = cs_uprop_lookup(cp)->wb;
        if (!cs_wb_is_ignore(wb)) return wb;
    }
    return CS_WB_OTHER;
}
#endif

cs_status cs_word_iter_next(cs_word_iter *it, cs_view *out_segment, size_t *out_start, size_t *out_end, bool *out_is_word_like) {
    if (!it || !cs_view_is_valid(it->v)) return CS_EINVAL;
    if (it->pos >= it->v.len) return CS_END;

    size_t start_pos = it->pos;
    cs_cp_iter cpi;
    cpi.v    = it->v;
    cpi.pos  = it->pos;
    cpi.mode = it->mode;

    uint32_t first_cp = 0;
    size_t cp_start = 0, cp_end = 0;
    cs_status st = cs_cp_iter_next(&cpi, &first_cp, &cp_start, &cp_end);
    if (st != CS_OK) return st;

#if CS_ENABLE_UNICODE
    const cs_uprop_record *first_rec = cs_uprop_lookup(first_cp);
    uint8_t raw_prev_wb = first_rec->wb;
    uint8_t eff_prev_wb = raw_prev_wb;
    size_t  ri_run      = (eff_prev_wb == CS_WB_RI) ? 1U : 0U;
    bool    word_like   = cs_cp_is_alnum(first_cp) || first_rec->wb == CS_WB_KATAKANA;
    size_t  seg_end     = cp_end;

    while (cpi.pos < cpi.v.len) {
        size_t save_pos = cpi.pos;
        uint32_t next_cp = 0;
        size_t n_start = 0, n_end = 0;
        st = cs_cp_iter_next(&cpi, &next_cp, &n_start, &n_end);
        if (st != CS_OK) return st;

        const cs_uprop_record *next_rec = cs_uprop_lookup(next_cp);
        uint8_t next_wb = next_rec->wb;
        bool next_extpict = (next_rec->flags & CS_UF_EXTPICT) != 0;

        /* WB3: CR x LF */
        if (raw_prev_wb == CS_WB_CR && next_wb == CS_WB_LF) {
            seg_end = n_end;
            raw_prev_wb = next_wb;
            eff_prev_wb = next_wb;
            continue;
        }
        /* WB3a: (Newline | CR | LF) ÷ */
        if (raw_prev_wb == CS_WB_NEWLINE || raw_prev_wb == CS_WB_CR || raw_prev_wb == CS_WB_LF) {
            cpi.pos = save_pos;
            break;
        }
        /* WB3b: ÷ (Newline | CR | LF) */
        if (next_wb == CS_WB_NEWLINE || next_wb == CS_WB_CR || next_wb == CS_WB_LF) {
            cpi.pos = save_pos;
            break;
        }
        /* WB3c: ZWJ x \p{Extended_Pictographic} */
        if (raw_prev_wb == CS_WB_ZWJ && next_extpict) {
            seg_end = n_end;
            raw_prev_wb = next_wb;
            if (!cs_wb_is_ignore(next_wb)) eff_prev_wb = next_wb;
            continue;
        }
        /* WB3d: WSegSpace x WSegSpace */
        if (raw_prev_wb == CS_WB_WSEGSPACE && next_wb == CS_WB_WSEGSPACE) {
            seg_end = n_end;
            raw_prev_wb = next_wb;
            eff_prev_wb = next_wb;
            continue;
        }
        /* WB4: X (Extend | Format | ZWJ)* -> X */
        if (cs_wb_is_ignore(next_wb)) {
            seg_end = n_end;
            raw_prev_wb = next_wb;
            continue;
        }

        bool no_break = false;
        /* WB5: AHLetter x AHLetter */
        if (cs_wb_is_ahletter(eff_prev_wb) && cs_wb_is_ahletter(next_wb)) {
            no_break = true;
        }
        /* WB6: AHLetter x (MidLetter | MidNumLetQ) AHLetter */
        else if (cs_wb_is_ahletter(eff_prev_wb) &&
                 (next_wb == CS_WB_MIDLETTER || cs_wb_is_midnumletq(next_wb))) {
            uint8_t look = cs_wb_peek_next_non_ignored(it->v, n_end, it->mode);
            if (cs_wb_is_ahletter(look)) {
                no_break = true;
            }
        }
        /* WB7a: Hebrew_Letter x Single_Quote */
        if (!no_break && eff_prev_wb == CS_WB_HEBREW_LETTER && next_wb == CS_WB_SINGLE_QUOTE) {
            no_break = true;
        }
        /* WB7b: Hebrew_Letter x Double_Quote Hebrew_Letter */
        if (!no_break && eff_prev_wb == CS_WB_HEBREW_LETTER && next_wb == CS_WB_DOUBLE_QUOTE) {
            uint8_t look = cs_wb_peek_next_non_ignored(it->v, n_end, it->mode);
            if (look == CS_WB_HEBREW_LETTER) {
                no_break = true;
            }
        }
        /* WB7 / WB7c are handled because when WB6 / WB7b matched, we advance through MidLetter/Quote
         * and keep track of whether we bridge to the following AHLetter/Hebrew_Letter! */
        if (!no_break && (eff_prev_wb == CS_WB_MIDLETTER || cs_wb_is_midnumletq(eff_prev_wb)) &&
            cs_wb_is_ahletter(next_wb)) {
            no_break = true;
        }
        if (!no_break && eff_prev_wb == CS_WB_DOUBLE_QUOTE && next_wb == CS_WB_HEBREW_LETTER) {
            no_break = true;
        }
        /* WB8: Numeric x Numeric */
        if (!no_break && eff_prev_wb == CS_WB_NUMERIC && next_wb == CS_WB_NUMERIC) {
            no_break = true;
        }
        /* WB9: AHLetter x Numeric */
        if (!no_break && cs_wb_is_ahletter(eff_prev_wb) && next_wb == CS_WB_NUMERIC) {
            no_break = true;
        }
        /* WB10: Numeric x AHLetter */
        if (!no_break && eff_prev_wb == CS_WB_NUMERIC && cs_wb_is_ahletter(next_wb)) {
            no_break = true;
        }
        /* WB11 & WB12: Numeric (MidNum | MidNumLetQ) x Numeric */
        if (!no_break && eff_prev_wb == CS_WB_NUMERIC &&
            (next_wb == CS_WB_MIDNUM || cs_wb_is_midnumletq(next_wb))) {
            uint8_t look = cs_wb_peek_next_non_ignored(it->v, n_end, it->mode);
            if (look == CS_WB_NUMERIC) {
                no_break = true;
            }
        }
        if (!no_break && (eff_prev_wb == CS_WB_MIDNUM || cs_wb_is_midnumletq(eff_prev_wb)) &&
            next_wb == CS_WB_NUMERIC) {
            no_break = true;
        }
        /* WB13: Katakana x Katakana */
        if (!no_break && eff_prev_wb == CS_WB_KATAKANA && next_wb == CS_WB_KATAKANA) {
            no_break = true;
        }
        /* WB13a: (AHLetter | Numeric | Katakana | ExtendNumLet) x ExtendNumLet */
        if (!no_break && next_wb == CS_WB_EXTENDNUMLET &&
            (cs_wb_is_ahletter(eff_prev_wb) || eff_prev_wb == CS_WB_NUMERIC ||
             eff_prev_wb == CS_WB_KATAKANA || eff_prev_wb == CS_WB_EXTENDNUMLET)) {
            no_break = true;
        }
        /* WB13b: ExtendNumLet x (AHLetter | Numeric | Katakana) */
        if (!no_break && eff_prev_wb == CS_WB_EXTENDNUMLET &&
            (cs_wb_is_ahletter(next_wb) || next_wb == CS_WB_NUMERIC || next_wb == CS_WB_KATAKANA)) {
            no_break = true;
        }
        /* WB15 / WB16: Regional_Indicator pairs */
        if (!no_break && eff_prev_wb == CS_WB_RI && next_wb == CS_WB_RI && (ri_run & 1U) == 1U) {
            no_break = true;
        }

        if (!no_break) {
            cpi.pos = save_pos;
            break;
        }

        seg_end = n_end;
        if (cs_cp_is_alnum(next_cp) || next_wb == CS_WB_KATAKANA) {
            word_like = true;
        }
        if (next_wb == CS_WB_RI) ri_run++;
        else ri_run = 0;
        raw_prev_wb = next_wb;
        eff_prev_wb = next_wb;
    }
    it->pos = seg_end;
    if (out_is_word_like) *out_is_word_like = word_like;
#else
    it->pos = cp_end;
    if (out_is_word_like) *out_is_word_like = cs_cp_is_alnum(first_cp);
#endif

    if (out_segment) {
        out_segment->data = it->v.data + start_pos;
        out_segment->len  = it->pos - start_pos;
    }
    if (out_start) *out_start = start_pos;
    if (out_end)   *out_end   = it->pos;
    return CS_OK;
}

/* Sentence Segmentation Iterator (UAX #29 SB1..SB998) */
void cs_sentence_iter_init(cs_sentence_iter *it, cs_view v, cs_decode_mode mode) {
    if (!it) return;
    it->v    = v;
    it->pos  = 0;
    it->mode = mode;
}

cs_status cs_sentence_iter_next(cs_sentence_iter *it, cs_view *out_sentence, size_t *out_start, size_t *out_end) {
    if (!it || !cs_view_is_valid(it->v)) return CS_EINVAL;
    if (it->pos >= it->v.len) return CS_END;

    size_t start_pos = it->pos;
    cs_cp_iter cpi;
    cpi.v    = it->v;
    cpi.pos  = it->pos;
    cpi.mode = it->mode;

    uint32_t cp = 0;
    size_t cp_start = 0, cp_end = 0;
    cs_status st = cs_cp_iter_next(&cpi, &cp, &cp_start, &cp_end);
    if (st != CS_OK) return st;

#if CS_ENABLE_UNICODE
    uint8_t prev_sb = cs_uprop_lookup(cp)->sb;
    uint8_t prev_prev_sb = CS_SB_OTHER;
    bool seen_saterm = (prev_sb == CS_SB_ATERM || prev_sb == CS_SB_STERM);
    bool seen_aterm  = (prev_sb == CS_SB_ATERM);
    bool seen_sp     = false;
    bool seen_sep    = (prev_sb == CS_SB_SEP || prev_sb == CS_SB_CR || prev_sb == CS_SB_LF);
    size_t sent_end  = cp_end;

    while (cpi.pos < cpi.v.len) {
        size_t save_pos = cpi.pos;
        uint32_t next_cp = 0;
        size_t n_start = 0, n_end = 0;
        st = cs_cp_iter_next(&cpi, &next_cp, &n_start, &n_end);
        if (st != CS_OK) return st;

        uint8_t next_sb = cs_uprop_lookup(next_cp)->sb;

        /* SB3: CR x LF */
        if (prev_sb == CS_SB_CR && next_sb == CS_SB_LF) {
            sent_end = n_end;
            it->pos  = sent_end;
            goto done;
        }
        /* SB4: (Sep | CR | LF) ÷ */
        if (seen_sep) {
            cpi.pos = save_pos;
            break;
        }
        /* SB5: Ignore Extend and Format */
        if (next_sb == CS_SB_EXTEND || next_sb == CS_SB_FORMAT) {
            sent_end = n_end;
            continue;
        }

        if (seen_saterm) {
            /* SB6: ATerm x Numeric */
            if (seen_aterm && !seen_sp && next_sb == CS_SB_NUMERIC) {
                seen_saterm = false;
                seen_aterm  = false;
            }
            /* SB7: (Upper | Lower) ATerm x Upper */
            else if (seen_aterm && !seen_sp &&
                     (prev_prev_sb == CS_SB_UPPER || prev_prev_sb == CS_SB_LOWER) &&
                     next_sb == CS_SB_UPPER) {
                seen_saterm = false;
                seen_aterm  = false;
            }
            /* SB8a: SATerm Close* Sp* x (SContinue | SATerm) */
            else if (next_sb == CS_SB_SCONTINUE || next_sb == CS_SB_ATERM || next_sb == CS_SB_STERM) {
                seen_saterm = (next_sb == CS_SB_ATERM || next_sb == CS_SB_STERM);
                seen_aterm  = (next_sb == CS_SB_ATERM);
                seen_sp     = false;
            }
            /* SB9: SATerm Close* x (Close | Sp | Sep | CR | LF) */
            else if (!seen_sp && next_sb == CS_SB_CLOSE) {
                /* remain in seen_saterm */
            }
            /* SB10: SATerm Close* Sp* x (Sp | Sep | CR | LF) */
            else if (next_sb == CS_SB_SP) {
                seen_sp = true;
            }
            else if (next_sb == CS_SB_SEP || next_sb == CS_SB_CR || next_sb == CS_SB_LF) {
                seen_sep = true;
            }
            else {
                /* SB8: ATerm Close* Sp* x (!(OLetter | Upper | Lower | Sep | CR | LF | STerm | ATerm))* Lower */
                bool sb8_match = false;
                if (seen_aterm) {
                    if (next_sb == CS_SB_LOWER) {
                        sb8_match = true;
                    } else if (next_sb != CS_SB_OLETTER && next_sb != CS_SB_UPPER &&
                               next_sb != CS_SB_SEP && next_sb != CS_SB_CR && next_sb != CS_SB_LF &&
                               next_sb != CS_SB_STERM && next_sb != CS_SB_ATERM) {
                        /* Lookahead for Lower */
                        cs_cp_iter look = cpi;
                        uint32_t lcp = 0;
                        while (cs_cp_iter_next(&look, &lcp, NULL, NULL) == CS_OK) {
                            uint8_t lsb = cs_uprop_lookup(lcp)->sb;
                            if (lsb == CS_SB_EXTEND || lsb == CS_SB_FORMAT) continue;
                            if (lsb == CS_SB_LOWER) { sb8_match = true; break; }
                            if (lsb == CS_SB_OLETTER || lsb == CS_SB_UPPER ||
                                lsb == CS_SB_SEP || lsb == CS_SB_CR || lsb == CS_SB_LF ||
                                lsb == CS_SB_STERM || lsb == CS_SB_ATERM) {
                                break;
                            }
                        }
                    }
                }
                if (sb8_match) {
                    seen_saterm = false;
                    seen_aterm  = false;
                    seen_sp     = false;
                } else {
                    /* SB11: Break after SATerm Close* Sp* */
                    cpi.pos = save_pos;
                    break;
                }
            }
        } else {
            if (next_sb == CS_SB_ATERM || next_sb == CS_SB_STERM) {
                seen_saterm = true;
                seen_aterm  = (next_sb == CS_SB_ATERM);
                seen_sp     = false;
            } else if (next_sb == CS_SB_SEP || next_sb == CS_SB_CR || next_sb == CS_SB_LF) {
                seen_sep = true;
            }
        }

        sent_end = n_end;
        prev_prev_sb = prev_sb;
        prev_sb = next_sb;
    }
    it->pos = sent_end;
done:
#else
    it->pos = it->v.len;
#endif
    if (out_sentence) {
        out_sentence->data = it->v.data + start_pos;
        out_sentence->len  = it->pos - start_pos;
    }
    if (out_start) *out_start = start_pos;
    if (out_end)   *out_end   = it->pos;
    return CS_OK;
}

/* ==========================================================================
 * Section 6: Access, Slices, Coordinate Conversions & Boundary Index
 * ========================================================================== */

cs_status cs_len_codepoints(cs_view v, size_t *out_count) {
    if (!cs_view_is_valid(v) || !out_count) return CS_EINVAL;
    size_t count = 0;
    cs_cp_iter it;
    cs_cp_iter_init(&it, v, CS_DECODE_STRICT);
    uint32_t cp = 0;
    cs_status st;
    while ((st = cs_cp_iter_next(&it, &cp, NULL, NULL)) == CS_OK) {
        count++;
    }
    if (st != CS_END) return st;
    *out_count = count;
    return CS_OK;
}

cs_status cs_len_graphemes(cs_view v, size_t *out_count) {
    if (!cs_view_is_valid(v) || !out_count) return CS_EINVAL;
    size_t count = 0;
    cs_grapheme_iter it;
    cs_grapheme_iter_init(&it, v, CS_DECODE_STRICT);
    cs_status st;
    while ((st = cs_grapheme_iter_next(&it, NULL, NULL, NULL)) == CS_OK) {
        count++;
    }
    if (st != CS_END) return st;
    *out_count = count;
    return CS_OK;
}

cs_status cs_codepoint_at(cs_view v, size_t cp_idx, uint32_t *out_cp) {
    if (!cs_view_is_valid(v) || !out_cp) return CS_EINVAL;
    cs_cp_iter it;
    cs_cp_iter_init(&it, v, CS_DECODE_STRICT);
    uint32_t cp = 0;
    size_t idx = 0;
    cs_status st;
    while ((st = cs_cp_iter_next(&it, &cp, NULL, NULL)) == CS_OK) {
        if (idx == cp_idx) {
            *out_cp = cp;
            return CS_OK;
        }
        idx++;
    }
    return (st == CS_END) ? CS_ERANGE : st;
}

cs_status cs_grapheme_at(cs_view v, size_t graph_idx, cs_view *out_grapheme) {
    if (!cs_view_is_valid(v) || !out_grapheme) return CS_EINVAL;
    cs_grapheme_iter it;
    cs_grapheme_iter_init(&it, v, CS_DECODE_STRICT);
    cs_view cl = cs_empty_view();
    size_t idx = 0;
    cs_status st;
    while ((st = cs_grapheme_iter_next(&it, &cl, NULL, NULL)) == CS_OK) {
        if (idx == graph_idx) {
            *out_grapheme = cl;
            return CS_OK;
        }
        idx++;
    }
    return (st == CS_END) ? CS_ERANGE : st;
}

cs_status cs_front_codepoint(cs_view v, uint32_t *out_cp) {
    if (!cs_view_is_valid(v) || !out_cp) return CS_EINVAL;
    if (v.len == 0) return CS_ERANGE;
    return cs_decode_utf8_next(v, 0, out_cp, NULL);
}

cs_status cs_back_codepoint(cs_view v, uint32_t *out_cp) {
    if (!cs_view_is_valid(v) || !out_cp) return CS_EINVAL;
    if (v.len == 0) return CS_ERANGE;
    return cs_decode_utf8_prev(v, v.len, out_cp, NULL);
}

cs_status cs_front_grapheme(cs_view v, cs_view *out_grapheme) {
    if (!cs_view_is_valid(v) || !out_grapheme) return CS_EINVAL;
    if (v.len == 0) return CS_ERANGE;
    cs_grapheme_iter it;
    cs_grapheme_iter_init(&it, v, CS_DECODE_STRICT);
    return cs_grapheme_iter_next(&it, out_grapheme, NULL, NULL);
}

cs_status cs_back_grapheme(cs_view v, cs_view *out_grapheme) {
    if (!cs_view_is_valid(v) || !out_grapheme) return CS_EINVAL;
    if (v.len == 0) return CS_ERANGE;
    cs_grapheme_iter it;
    cs_grapheme_iter_init(&it, v, CS_DECODE_STRICT);
    it.pos = v.len;
    return cs_grapheme_iter_prev(&it, out_grapheme, NULL, NULL);
}

cs_status cs_slice_codepoints(cs_view v, size_t begin_cp, size_t end_cp, cs_view *out) {
    if (!cs_view_is_valid(v) || !out) return CS_EINVAL;
    if (begin_cp > end_cp) return CS_ERANGE;

    cs_cp_iter it;
    cs_cp_iter_init(&it, v, CS_DECODE_STRICT);
    size_t idx = 0;
    size_t start_byte = v.len, end_byte = v.len;
    bool found_start = false, found_end = false;

    uint32_t cp = 0;
    size_t s = 0, e = 0;
    cs_status st;
    while ((st = cs_cp_iter_next(&it, &cp, &s, &e)) == CS_OK) {
        if (idx == begin_cp) { start_byte = s; found_start = true; }
        if (idx == end_cp)   { end_byte   = s; found_end   = true; }
        idx++;
    }
    if (st != CS_END) return st;
    if (begin_cp == idx) { start_byte = v.len; found_start = true; }
    if (end_cp   == idx) { end_byte   = v.len; found_end   = true; }
    if (!found_start || !found_end) return CS_ERANGE;

    out->data = v.data ? (v.data + start_byte) : NULL;
    out->len  = end_byte - start_byte;
    return CS_OK;
}

cs_status cs_slice_graphemes(cs_view v, size_t begin_gr, size_t end_gr, cs_view *out) {
    if (!cs_view_is_valid(v) || !out) return CS_EINVAL;
    if (begin_gr > end_gr) return CS_ERANGE;

    cs_grapheme_iter it;
    cs_grapheme_iter_init(&it, v, CS_DECODE_STRICT);
    size_t idx = 0;
    size_t start_byte = v.len, end_byte = v.len;
    bool found_start = false, found_end = false;

    size_t s = 0, e = 0;
    cs_status st;
    while ((st = cs_grapheme_iter_next(&it, NULL, &s, &e)) == CS_OK) {
        if (idx == begin_gr) { start_byte = s; found_start = true; }
        if (idx == end_gr)   { end_byte   = s; found_end   = true; }
        idx++;
    }
    if (st != CS_END) return st;
    if (begin_gr == idx) { start_byte = v.len; found_start = true; }
    if (end_gr   == idx) { end_byte   = v.len; found_end   = true; }
    if (!found_start || !found_end) return CS_ERANGE;

    out->data = v.data ? (v.data + start_byte) : NULL;
    out->len  = end_byte - start_byte;
    return CS_OK;
}

cs_status cs_take_codepoints(cs_view v, size_t count, cs_view *out) {
    return cs_slice_codepoints(v, 0, count, out);
}

cs_status cs_drop_codepoints(cs_view v, size_t count, cs_view *out) {
    if (!cs_view_is_valid(v) || !out) return CS_EINVAL;
    size_t byte_off = 0;
    cs_status st = cs_cp_index_to_offset(v, count, &byte_off);
    if (st != CS_OK) return st;
    /* Also validate the remainder of the string */
    cs_view tail = { v.data ? (v.data + byte_off) : NULL, v.len - byte_off };
    st = cs_validate_utf8(tail, NULL);
    if (st != CS_OK) return st;
    *out = tail;
    return CS_OK;
}

cs_status cs_take_graphemes(cs_view v, size_t count, cs_view *out) {
    return cs_slice_graphemes(v, 0, count, out);
}

cs_status cs_drop_graphemes(cs_view v, size_t count, cs_view *out) {
    if (!cs_view_is_valid(v) || !out) return CS_EINVAL;
    size_t byte_off = 0;
    cs_status st = cs_grapheme_index_to_offset(v, count, &byte_off);
    if (st != CS_OK) return st;
    cs_view tail = { v.data ? (v.data + byte_off) : NULL, v.len - byte_off };
    st = cs_validate_utf8(tail, NULL);
    if (st != CS_OK) return st;
    *out = tail;
    return CS_OK;
}

cs_status cs_truncate_codepoints(cs_string *s, size_t max_codepoints) {
    if (!s) return CS_EINVAL;
    cs_view v = cs_as_view(s);
    cs_cp_iter it;
    cs_cp_iter_init(&it, v, CS_DECODE_STRICT);
    size_t idx = 0;
    size_t cut_byte = s->len;
    bool cut_found = false;
    uint32_t cp = 0;
    size_t start_b = 0, end_b = 0;
    cs_status st;
    while ((st = cs_cp_iter_next(&it, &cp, &start_b, &end_b)) == CS_OK) {
        if (idx == max_codepoints) {
            cut_byte = start_b;
            cut_found = true;
            break;
        }
        idx++;
    }
    if (st != CS_END && st != CS_OK) return st;
    if (!cut_found) return CS_OK;
    return cs_truncate_bytes(s, cut_byte);
}

cs_status cs_truncate_graphemes(cs_string *s, size_t max_graphemes) {
    if (!s) return CS_EINVAL;
    cs_view v = cs_as_view(s);
    cs_grapheme_iter it;
    cs_grapheme_iter_init(&it, v, CS_DECODE_STRICT);
    size_t idx = 0;
    size_t cut_byte = s->len;
    bool cut_found = false;
    size_t start_b = 0, end_b = 0;
    cs_status st;
    while ((st = cs_grapheme_iter_next(&it, NULL, &start_b, &end_b)) == CS_OK) {
        if (idx == max_graphemes) {
            cut_byte = start_b;
            cut_found = true;
            break;
        }
        idx++;
    }
    if (st != CS_END && st != CS_OK) return st;
    if (!cut_found) return CS_OK;
    return cs_truncate_bytes(s, cut_byte);
}

cs_status cs_offset_to_cp_index(cs_view v, size_t byte_offset, size_t *out_cp_idx) {
    if (!cs_view_is_valid(v) || !out_cp_idx) return CS_EINVAL;
    if (byte_offset > v.len) return CS_ERANGE;
    cs_cp_iter it;
    cs_cp_iter_init(&it, v, CS_DECODE_STRICT);
    size_t idx = 0;
    uint32_t cp = 0;
    size_t s = 0, e = 0;
    cs_status st;
    while ((st = cs_cp_iter_next(&it, &cp, &s, &e)) == CS_OK) {
        if (s == byte_offset) {
            *out_cp_idx = idx;
            return CS_OK;
        }
        if (s > byte_offset) {
            return CS_EINVAL; /* byte_offset was not on a code point boundary */
        }
        idx++;
    }
    if (st != CS_END) return st;
    if (byte_offset == v.len) {
        *out_cp_idx = idx;
        return CS_OK;
    }
    return CS_EINVAL;
}

cs_status cs_cp_index_to_offset(cs_view v, size_t cp_idx, size_t *out_byte_offset) {
    if (!cs_view_is_valid(v) || !out_byte_offset) return CS_EINVAL;
    cs_cp_iter it;
    cs_cp_iter_init(&it, v, CS_DECODE_STRICT);
    size_t idx = 0;
    uint32_t cp = 0;
    size_t s = 0, e = 0;
    cs_status st;
    while ((st = cs_cp_iter_next(&it, &cp, &s, &e)) == CS_OK) {
        if (idx == cp_idx) {
            *out_byte_offset = s;
            return CS_OK;
        }
        idx++;
    }
    if (st != CS_END) return st;
    if (cp_idx == idx) {
        *out_byte_offset = v.len;
        return CS_OK;
    }
    return CS_ERANGE;
}

cs_status cs_offset_to_grapheme_index(cs_view v, size_t byte_offset, size_t *out_gr_idx) {
    if (!cs_view_is_valid(v) || !out_gr_idx) return CS_EINVAL;
    if (byte_offset > v.len) return CS_ERANGE;
    cs_grapheme_iter it;
    cs_grapheme_iter_init(&it, v, CS_DECODE_STRICT);
    size_t idx = 0;
    size_t s = 0, e = 0;
    cs_status st;
    while ((st = cs_grapheme_iter_next(&it, NULL, &s, &e)) == CS_OK) {
        if (s == byte_offset) {
            *out_gr_idx = idx;
            return CS_OK;
        }
        if (s > byte_offset) {
            return CS_EINVAL; /* byte_offset was not on a grapheme cluster boundary */
        }
        idx++;
    }
    if (st != CS_END) return st;
    if (byte_offset == v.len) {
        *out_gr_idx = idx;
        return CS_OK;
    }
    return CS_EINVAL;
}

cs_status cs_grapheme_index_to_offset(cs_view v, size_t gr_idx, size_t *out_byte_offset) {
    if (!cs_view_is_valid(v) || !out_byte_offset) return CS_EINVAL;
    cs_grapheme_iter it;
    cs_grapheme_iter_init(&it, v, CS_DECODE_STRICT);
    size_t idx = 0;
    size_t s = 0, e = 0;
    cs_status st;
    while ((st = cs_grapheme_iter_next(&it, NULL, &s, &e)) == CS_OK) {
        if (idx == gr_idx) {
            *out_byte_offset = s;
            return CS_OK;
        }
        idx++;
    }
    if (st != CS_END) return st;
    if (gr_idx == idx) {
        *out_byte_offset = v.len;
        return CS_OK;
    }
    return CS_ERANGE;
}

/* Optional O(1) Boundary Index with Owner Mutation Invalidation */
cs_status cs_index_build(cs_index *idx, const cs_string *owner, cs_view v, const cs_allocator *alloc) {
    if (!idx) return CS_EINVAL;
    memset(idx, 0, sizeof(*idx));
    if (owner) v = cs_as_view(owner);
    if (!cs_view_is_valid(v)) return CS_EINVAL;

    cs_allocator a = alloc ? *alloc : (owner ? owner->alloc : cs_default_allocator());
    if (!a.alloc || !a.free) return CS_EINVAL;

    size_t cp_cnt = 0, gr_cnt = 0;
    cs_status st = cs_len_codepoints(v, &cp_cnt);
    if (st != CS_OK) return st;
    st = cs_len_graphemes(v, &gr_cnt);
    if (st != CS_OK) return st;

    size_t *cp_offs = (size_t *)a.alloc(a.ctx, (cp_cnt + 1U) * sizeof(size_t), CS_ALIGNOF(size_t));
    if (!cp_offs) return CS_ENOMEM;
    size_t *gr_offs = (size_t *)a.alloc(a.ctx, (gr_cnt + 1U) * sizeof(size_t), CS_ALIGNOF(size_t));
    if (!gr_offs) {
        a.free(a.ctx, cp_offs, (cp_cnt + 1U) * sizeof(size_t), CS_ALIGNOF(size_t));
        return CS_ENOMEM;
    }

    cs_cp_iter cpi;
    cs_cp_iter_init(&cpi, v, CS_DECODE_STRICT);
    size_t i = 0, s = 0, e = 0;
    uint32_t cp = 0;
    while (cs_cp_iter_next(&cpi, &cp, &s, &e) == CS_OK) {
        cp_offs[i++] = s;
    }
    cp_offs[cp_cnt] = v.len;

    cs_grapheme_iter gri;
    cs_grapheme_iter_init(&gri, v, CS_DECODE_STRICT);
    i = 0;
    while (cs_grapheme_iter_next(&gri, NULL, &s, &e) == CS_OK) {
        gr_offs[i++] = s;
    }
    gr_offs[gr_cnt] = v.len;

    idx->owner            = owner;
    idx->bound_data       = v.data;
    idx->bound_len        = v.len;
    idx->owner_generation = owner ? owner->generation : 0;
    idx->cp_offsets       = cp_offs;
    idx->cp_count         = cp_cnt;
    idx->gr_offsets       = gr_offs;
    idx->gr_count         = gr_cnt;
    idx->alloc            = a;
    return CS_OK;
}

void cs_index_destroy(cs_index *idx) {
    if (!idx) return;
    if (idx->cp_offsets && idx->alloc.free) {
        idx->alloc.free(idx->alloc.ctx, idx->cp_offsets, (idx->cp_count + 1U) * sizeof(size_t), CS_ALIGNOF(size_t));
    }
    if (idx->gr_offsets && idx->alloc.free) {
        idx->alloc.free(idx->alloc.ctx, idx->gr_offsets, (idx->gr_count + 1U) * sizeof(size_t), CS_ALIGNOF(size_t));
    }
    memset(idx, 0, sizeof(*idx));
}

bool cs_index_is_valid(const cs_index *idx) {
    if (!idx || !idx->cp_offsets || !idx->gr_offsets) return false;
    if (idx->owner) {
        if (idx->owner->generation != idx->owner_generation) return false;
        if (idx->owner->data != idx->bound_data || idx->owner->len != idx->bound_len) return false;
    }
    return true;
}

cs_status cs_index_cp_count(const cs_index *idx, size_t *out_count) {
    if (!idx || !out_count) return CS_EINVAL;
    if (!cs_index_is_valid(idx)) return CS_EINVAL;
    *out_count = idx->cp_count;
    return CS_OK;
}

cs_status cs_index_grapheme_count(const cs_index *idx, size_t *out_count) {
    if (!idx || !out_count) return CS_EINVAL;
    if (!cs_index_is_valid(idx)) return CS_EINVAL;
    *out_count = idx->gr_count;
    return CS_OK;
}

cs_status cs_index_cp_offset(const cs_index *idx, size_t cp_idx, size_t *out_offset) {
    if (!idx || !out_offset) return CS_EINVAL;
    if (!cs_index_is_valid(idx)) return CS_EINVAL;
    if (cp_idx > idx->cp_count) return CS_ERANGE;
    *out_offset = idx->cp_offsets[cp_idx];
    return CS_OK;
}

cs_status cs_index_grapheme_span(const cs_index *idx, size_t gr_idx, cs_view *out_span) {
    if (!idx || !out_span) return CS_EINVAL;
    if (!cs_index_is_valid(idx)) return CS_EINVAL;
    if (gr_idx >= idx->gr_count) return CS_ERANGE;
    size_t s = idx->gr_offsets[gr_idx];
    size_t e = idx->gr_offsets[gr_idx + 1U];
    out_span->data = idx->bound_data ? (idx->bound_data + s) : NULL;
    out_span->len  = e - s;
    return CS_OK;
}

/* ==========================================================================
 * UAX #15 Normalization (NFC, NFD, NFKC, NFKD) & Quick_Check (Section 9)
 * ========================================================================== */

#if CS_ENABLE_UNICODE
#define CS_MAX_COMBINING_RUN 1024U

static void cs_decompose_cp_recursive(uint32_t cp, bool compat, uint32_t **buf, size_t *len, size_t *cap,
                                      cs_allocator *alloc, cs_workspace *ws, cs_status *err) {
    if (*err != CS_OK) return;

    /* Hangul algorithmic decomposition (UAX #15) */
    if (cp >= 0xAC00U && cp <= 0xD7A3U) {
        uint32_t s_idx = cp - 0xAC00U;
        uint32_t l_cp  = 0x1100U + (s_idx / 588U);
        uint32_t v_cp  = 0x1161U + ((s_idx % 588U) / 28U);
        uint32_t t_cp  = 0x11A7U + (s_idx % 28U);
        cs_decompose_cp_recursive(l_cp, compat, buf, len, cap, alloc, ws, err);
        cs_decompose_cp_recursive(v_cp, compat, buf, len, cap, alloc, ws, err);
        if (t_cp != 0x11A7U) {
            cs_decompose_cp_recursive(t_cp, compat, buf, len, cap, alloc, ws, err);
        }
        return;
    }

    const cs_udecomp_entry *de = cs_udecomp_find(cp);
    if (de) {
        bool is_compat = (de->info & 0x80000000U) != 0;
        if (!is_compat || compat) {
            uint32_t dlen = (de->info >> 24) & 0x7FU;
            uint32_t doff = de->info & 0x00FFFFFFU;
            for (uint32_t i = 0; i < dlen; i++) {
                cs_decompose_cp_recursive(cs_udecomp_pool[doff + i], compat, buf, len, cap, alloc, ws, err);
            }
            return;
        }
    }

    if (*len >= *cap) {
        size_t new_cap = (*cap == 0) ? 64U : (*cap * 2U);
        uint32_t *new_buf = NULL;
        if (ws) {
            new_buf = (uint32_t *)cs_ws_alloc(ws, new_cap * sizeof(uint32_t), CS_ALIGNOF(uint32_t));
            if (new_buf && *buf) {
                memcpy(new_buf, *buf, (*len) * sizeof(uint32_t));
            }
        } else if (alloc && alloc->alloc) {
            if (alloc->realloc && *buf) {
                new_buf = (uint32_t *)alloc->realloc(alloc->ctx, *buf, (*cap) * sizeof(uint32_t),
                                                     new_cap * sizeof(uint32_t), CS_ALIGNOF(uint32_t));
            } else {
                new_buf = (uint32_t *)alloc->alloc(alloc->ctx, new_cap * sizeof(uint32_t), CS_ALIGNOF(uint32_t));
                if (new_buf && *buf) {
                    memcpy(new_buf, *buf, (*len) * sizeof(uint32_t));
                    alloc->free(alloc->ctx, *buf, (*cap) * sizeof(uint32_t), CS_ALIGNOF(uint32_t));
                }
            }
        }
        if (!new_buf) {
            *err = ws ? CS_EBUFFER : CS_ENOMEM;
            return;
        }
        *buf = new_buf;
        *cap = new_cap;
    }
    (*buf)[(*len)++] = cp;
}

/* Canonical Ordering Algorithm: stable sort of non-starter sequences by CCC */
static cs_status cs_canonical_order(uint32_t *cps, size_t len) {
    size_t i = 0;
    while (i < len) {
        uint8_t ccc = cs_uprop_lookup(cps[i])->ccc;
        if (ccc == 0) {
            i++;
            continue;
        }
        size_t start = i;
        while (i < len && cs_uprop_lookup(cps[i])->ccc != 0) {
            i++;
        }
        size_t run_len = i - start;
        if (run_len > CS_MAX_COMBINING_RUN) {
            return CS_ELIMIT;
        }
        /* Insertion sort preserves stability on equal CCC */
        for (size_t j = start + 1U; j < i; j++) {
            uint32_t key_cp  = cps[j];
            uint8_t  key_ccc = cs_uprop_lookup(key_cp)->ccc;
            size_t   k = j;
            while (k > start && cs_uprop_lookup(cps[k - 1U])->ccc > key_ccc) {
                cps[k] = cps[k - 1U];
                k--;
            }
            cps[k] = key_cp;
        }
    }
    return CS_OK;
}

/* Canonical Composition Algorithm (UAX #15) */
static size_t cs_canonical_compose(uint32_t *cps, size_t len) {
    if (len <= 1U) return len;
    size_t starter_idx = 0;
    uint32_t starter_cp = cps[0];
    uint8_t starter_ccc = cs_uprop_lookup(starter_cp)->ccc;
    int last_ccc = (starter_ccc == 0) ? -1 : 256;

    size_t out_idx = 1U;
    for (size_t i = 1U; i < len; i++) {
        uint32_t ch = cps[i];
        uint8_t ch_ccc = cs_uprop_lookup(ch)->ccc;
        uint32_t composed = 0;
        if (last_ccc < (int)ch_ccc) {
            composed = cs_ucomp_find(starter_cp, ch);
        }
        if (composed != 0) {
            starter_cp = composed;
            cps[starter_idx] = composed;
        } else if (ch_ccc == 0) {
            starter_idx = out_idx;
            starter_cp  = ch;
            last_ccc    = -1;
            cps[out_idx++] = ch;
        } else {
            last_ccc = (int)ch_ccc;
            cps[out_idx++] = ch;
        }
    }
    return out_idx;
}
#endif /* CS_ENABLE_UNICODE */

cs_status cs_norm_quick_check(cs_view v, cs_norm_form form, cs_quick_check_result *out_qc) {
    if (!cs_view_is_valid(v) || !out_qc) return CS_EINVAL;
#if !CS_ENABLE_UNICODE
    (void)form;
    return CS_EUNSUPPORTED;
#else
    unsigned int shift = 0;
    switch (form) {
        case CS_NORM_NFC:  shift = 0; break;
        case CS_NORM_NFD:  shift = 2; break;
        case CS_NORM_NFKC: shift = 4; break;
        case CS_NORM_NFKD: shift = 6; break;
        default: return CS_EINVAL;
    }

    cs_quick_check_result result = CS_QC_YES;
    uint8_t last_ccc = 0;
    size_t combine_run = 0;

    cs_cp_iter it;
    cs_cp_iter_init(&it, v, CS_DECODE_STRICT);
    uint32_t cp = 0;
    cs_status st;
    while ((st = cs_cp_iter_next(&it, &cp, NULL, NULL)) == CS_OK) {
        const cs_uprop_record *rec = cs_uprop_lookup(cp);
        uint8_t ccc = rec->ccc;
        if (ccc != 0) {
            combine_run++;
            if (combine_run > CS_MAX_COMBINING_RUN) return CS_ELIMIT;
            if (last_ccc > ccc) {
                *out_qc = CS_QC_NO;
                return CS_OK;
            }
        } else {
            combine_run = 0;
        }
        last_ccc = ccc;

        uint8_t qc_val = (rec->qc >> shift) & 0x3U;
        if (qc_val == 1U) {
            *out_qc = CS_QC_NO;
            return CS_OK;
        }
        if (qc_val == 2U) {
            result = CS_QC_MAYBE;
        }
    }
    if (st != CS_END) return st;
    *out_qc = result;
    return CS_OK;
#endif
}

cs_status cs_normalize(cs_string *dst, cs_view v, cs_norm_form form) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
#if !CS_ENABLE_UNICODE
    (void)form;
    return CS_EUNSUPPORTED;
#else
    if ((int)form < 0 || (int)form > 3) return CS_EINVAL;
    cs_quick_check_result qc = CS_QC_NO;
    cs_status st = cs_norm_quick_check(v, form, &qc);
    if (st != CS_OK) return st;
    if (qc == CS_QC_YES) {
        return cs_assign(dst, v);
    }

    bool compat  = (form == CS_NORM_NFKC || form == CS_NORM_NFKD);
    bool compose = (form == CS_NORM_NFC  || form == CS_NORM_NFKC);

    uint32_t *cps = NULL;
    size_t len = 0, cap = 0;
    cs_allocator alloc = dst->alloc.alloc ? dst->alloc : cs_default_allocator();

    cs_cp_iter it;
    cs_cp_iter_init(&it, v, CS_DECODE_STRICT);
    uint32_t cp = 0;
    while ((st = cs_cp_iter_next(&it, &cp, NULL, NULL)) == CS_OK) {
        cs_status derr = CS_OK;
        cs_decompose_cp_recursive(cp, compat, &cps, &len, &cap, &alloc, NULL, &derr);
        if (derr != CS_OK) {
            if (cps) alloc.free(alloc.ctx, cps, cap * sizeof(uint32_t), CS_ALIGNOF(uint32_t));
            return derr;
        }
    }
    if (st != CS_END) {
        if (cps) alloc.free(alloc.ctx, cps, cap * sizeof(uint32_t), CS_ALIGNOF(uint32_t));
        return st;
    }

    st = cs_canonical_order(cps, len);
    if (st != CS_OK) {
        if (cps) alloc.free(alloc.ctx, cps, cap * sizeof(uint32_t), CS_ALIGNOF(uint32_t));
        return st;
    }

    if (compose) {
        len = cs_canonical_compose(cps, len);
    }

    st = cs_utf32_to_utf8(cps, len, dst);
    if (cps) alloc.free(alloc.ctx, cps, cap * sizeof(uint32_t), CS_ALIGNOF(uint32_t));
    return st;
#endif
}

cs_status cs_normalize_buffer(cs_buffer *dst, cs_view v, cs_norm_form form, cs_workspace *ws, size_t *out_required) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
#if !CS_ENABLE_UNICODE
    (void)form; (void)ws; (void)out_required;
    return CS_EUNSUPPORTED;
#else
    if ((int)form < 0 || (int)form > 3) return CS_EINVAL;
    cs_quick_check_result qc = CS_QC_NO;
    cs_status st = cs_norm_quick_check(v, form, &qc);
    if (st != CS_OK) return st;
    if (qc == CS_QC_YES) {
        cs_buffer_clear(dst);
        return cs_buffer_append(dst, v, out_required);
    }
    if (!ws || !ws->mem) return CS_EBUFFER;

    size_t saved_ws = ws->used;
    bool compat  = (form == CS_NORM_NFKC || form == CS_NORM_NFKD);
    bool compose = (form == CS_NORM_NFC  || form == CS_NORM_NFKC);

    uint32_t *cps = NULL;
    size_t len = 0, cap = 0;

    cs_cp_iter it;
    cs_cp_iter_init(&it, v, CS_DECODE_STRICT);
    uint32_t cp = 0;
    while ((st = cs_cp_iter_next(&it, &cp, NULL, NULL)) == CS_OK) {
        cs_status derr = CS_OK;
        cs_decompose_cp_recursive(cp, compat, &cps, &len, &cap, NULL, ws, &derr);
        if (derr != CS_OK) { ws->used = saved_ws; return derr; }
    }
    if (st != CS_END) { ws->used = saved_ws; return st; }

    st = cs_canonical_order(cps, len);
    if (st != CS_OK) { ws->used = saved_ws; return st; }
    if (compose) {
        len = cs_canonical_compose(cps, len);
    }

    size_t total_bytes = 0;
    for (size_t i = 0; i < len; i++) {
        char b[4];
        size_t blen = 0;
        cs_encode_utf8_codepoint(cps[i], b, &blen);
        total_bytes += blen;
    }
    if (out_required) *out_required = total_bytes + 1U;
    if (total_bytes > dst->cap) {
        ws->used = saved_ws;
        return CS_EBUFFER;
    }

    dst->len = 0;
    for (size_t i = 0; i < len; i++) {
        cs_buffer_append_codepoint(dst, cps[i], NULL);
    }
    ws->used = saved_ws;
    return CS_OK;
#endif
}

cs_status cs_is_normalized(cs_view v, cs_norm_form form, bool *out_normalized) {
    if (!cs_view_is_valid(v) || !out_normalized) return CS_EINVAL;
    cs_quick_check_result qc = CS_QC_NO;
    cs_status st = cs_norm_quick_check(v, form, &qc);
    if (st != CS_OK) return st;
    if (qc == CS_QC_YES) {
        *out_normalized = true;
        return CS_OK;
    }
    if (qc == CS_QC_NO) {
        *out_normalized = false;
        return CS_OK;
    }
    /* QC == MAYBE: try stack buffer + workspace fast path first, then heap fallback */
    if (v.len <= 256U) {
        char stack_out[512];
        uint64_t stack_ws_mem[256];
        cs_buffer sbuf;
        cs_workspace sws;
        cs_buffer_init(&sbuf, stack_out, sizeof(stack_out));
        cs_workspace_init(&sws, stack_ws_mem, sizeof(stack_ws_mem));
        if (cs_normalize_buffer(&sbuf, v, form, &sws, NULL) == CS_OK) {
            *out_normalized = cs_equal(v, cs_buffer_as_view(&sbuf));
            return CS_OK;
        }
    }
    cs_string tmp;
    cs_init(&tmp);
    st = cs_normalize(&tmp, v, form);
    if (st != CS_OK) {
        cs_destroy(&tmp);
        return st;
    }
    *out_normalized = cs_equal(v, cs_as_view(&tmp));
    cs_destroy(&tmp);
    return CS_OK;
}

cs_status cs_strip_diacritics(cs_string *dst, cs_view v) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
#if !CS_ENABLE_UNICODE
    return CS_EUNSUPPORTED;
#else
    cs_string nfd;
    cs_status st = cs_init_with_allocator(&nfd, dst->alloc);
    if (st != CS_OK) return st;
    st = cs_normalize(&nfd, v, CS_NORM_NFD);
    if (st != CS_OK) { cs_destroy(&nfd); return st; }

    cs_string out;
    st = cs_init_with_allocator(&out, dst->alloc);
    if (st != CS_OK) { cs_destroy(&nfd); return st; }

    cs_cp_iter it;
    cs_cp_iter_init(&it, cs_as_view(&nfd), CS_DECODE_STRICT);
    uint32_t cp = 0;
    while (cs_cp_iter_next(&it, &cp, NULL, NULL) == CS_OK) {
        cs_general_category gc = cs_cp_category(cp);
        if (gc == CS_GC_MN || gc == CS_GC_MC || gc == CS_GC_ME) {
            continue;
        }
        st = cs_append_codepoint(&out, cp);
        if (st != CS_OK) { cs_destroy(&nfd); cs_destroy(&out); return st; }
    }
    cs_destroy(&nfd);
    st = cs_normalize(dst, cs_as_view(&out), CS_NORM_NFC);
    cs_destroy(&out);
    return st;
#endif
}

/* ==========================================================================
 * Part 3: Search & Comparison (Two-Way O(n+m), Unicode Casefold, Natural),
 *         Split/Join/Tokenization, and Text Transformations
 * ========================================================================== */

static inline uint8_t cs_ascii_fold_byte(uint8_t b) {
    return (b >= 'A' && b <= 'Z') ? (uint8_t)(b + 32U) : b;
}

#if CS_ENABLE_UNICODE
static size_t cs_fold_cp_full(uint32_t cp, cs_case_locale locale, uint32_t out_cps[3]) {
    if (locale == CS_LOCALE_TR || locale == CS_LOCALE_AZ) {
        if (cp == 0x0049U) { out_cps[0] = 0x0131U; return 1U; }
        if (cp == 0x0130U) { out_cps[0] = 0x0069U; return 1U; }
    }
#if CS_ENABLE_UNICODE
    const cs_ucase_entry *ce = cs_ucase_find(cp);
    if (ce) {
        return cs_ucase_decode_target(cp, ce->cf, out_cps);
    }
#else
    if (cp >= 'A' && cp <= 'Z') {
        out_cps[0] = cp + 32U;
        return 1U;
    }
#endif
    out_cps[0] = cp;
    return 1U;
}
#endif

static uint32_t cs_fold_cp_simple(uint32_t cp) {
#if CS_ENABLE_UNICODE
    const cs_ucase_entry *ce = cs_ucase_find(cp);
    if (ce && ce->scf >= 0) {
        return (uint32_t)ce->scf;
    }
#else
    if (cp >= 'A' && cp <= 'Z') return cp + 32U;
#endif
    return cp;
}

/* ==========================================================================
 * Section 7: Comparison (Exact, ASCII Case-Insensitive, Unicode Casefold, Natural)
 * ========================================================================== */

bool cs_equal(cs_view a, cs_view b) {
    if (!cs_view_is_valid(a) || !cs_view_is_valid(b)) return false;
    if (a.len != b.len) return false;
    return cs_dispatch_equal((const uint8_t *)a.data, (const uint8_t *)b.data, a.len);
}

int cs_compare(cs_view a, cs_view b) {
    if (a.len == 0 && b.len == 0) return 0;
    size_t min_len = (a.len < b.len) ? a.len : b.len;
    if (min_len > 0 && a.data != b.data) {
        int cmp = memcmp(a.data, b.data, min_len);
        if (cmp != 0) return (cmp < 0) ? -1 : 1;
    }
    if (a.len < b.len) return -1;
    if (a.len > b.len) return 1;
    return 0;
}

cs_status cs_compare_mode(cs_view a, cs_view b, cs_match_mode mode, int *out_cmp) {
    if (!cs_view_is_valid(a) || !cs_view_is_valid(b) || !out_cmp) return CS_EINVAL;
    if (mode == CS_MATCH_EXACT) {
        *out_cmp = cs_compare(a, b);
        return CS_OK;
    }
    if (mode == CS_MATCH_ASCII_CASE_INSENSITIVE) {
        size_t min_len = (a.len < b.len) ? a.len : b.len;
        const uint8_t *pa = (const uint8_t *)a.data;
        const uint8_t *pb = (const uint8_t *)b.data;
        for (size_t i = 0; i < min_len; i++) {
            uint8_t ca = cs_ascii_fold_byte(pa[i]);
            uint8_t cb = cs_ascii_fold_byte(pb[i]);
            if (ca != cb) {
                *out_cmp = (ca < cb) ? -1 : 1;
                return CS_OK;
            }
        }
        *out_cmp = (a.len < b.len) ? -1 : ((a.len > b.len) ? 1 : 0);
        return CS_OK;
    }
    if (mode == CS_MATCH_UNICODE_CASEFOLD) {
#if !CS_ENABLE_UNICODE
        return CS_EUNSUPPORTED;
#else
        cs_cp_iter ita, itb;
        cs_cp_iter_init(&ita, a, CS_DECODE_STRICT);
        cs_cp_iter_init(&itb, b, CS_DECODE_STRICT);

        uint32_t qa[4], qb[4];
        size_t ha = 0, ta = 0, hb = 0, tb = 0;
        bool end_a = false, end_b = false;

        for (;;) {
            if (ha == ta && !end_a) {
                ha = ta = 0;
                uint32_t cp = 0;
                cs_status st = cs_cp_iter_next(&ita, &cp, NULL, NULL);
                if (st == CS_OK) {
                    ta = cs_fold_cp_full(cp, CS_LOCALE_ROOT, qa);
                } else if (st == CS_END) {
                    end_a = true;
                } else {
                    return st;
                }
            }
            if (hb == tb && !end_b) {
                hb = tb = 0;
                uint32_t cp = 0;
                cs_status st = cs_cp_iter_next(&itb, &cp, NULL, NULL);
                if (st == CS_OK) {
                    tb = cs_fold_cp_full(cp, CS_LOCALE_ROOT, qb);
                } else if (st == CS_END) {
                    end_b = true;
                } else {
                    return st;
                }
            }
            if (ha == ta && hb == tb) {
                *out_cmp = 0;
                return CS_OK;
            }
            if (ha == ta) {
                /* Drain remaining b to validate its UTF-8 */
                uint32_t cp = 0;
                cs_status st;
                while ((st = cs_cp_iter_next(&itb, &cp, NULL, NULL)) == CS_OK) {}
                if (st != CS_END) return st;
                *out_cmp = -1;
                return CS_OK;
            }
            if (hb == tb) {
                uint32_t cp = 0;
                cs_status st;
                while ((st = cs_cp_iter_next(&ita, &cp, NULL, NULL)) == CS_OK) {}
                if (st != CS_END) return st;
                *out_cmp = 1;
                return CS_OK;
            }
            uint32_t ca = qa[ha++];
            uint32_t cb = qb[hb++];
            if (ca != cb) {
                /* Validate remaining UTF-8 of both strings */
                uint32_t cp = 0;
                cs_status st;
                while ((st = cs_cp_iter_next(&ita, &cp, NULL, NULL)) == CS_OK) {}
                if (st != CS_END) return st;
                while ((st = cs_cp_iter_next(&itb, &cp, NULL, NULL)) == CS_OK) {}
                if (st != CS_END) return st;
                *out_cmp = (ca < cb) ? -1 : 1;
                return CS_OK;
            }
        }
#endif
    }
    return CS_EINVAL;
}

cs_status cs_equal_mode(cs_view a, cs_view b, cs_match_mode mode, bool *out_equal) {
    if (!out_equal) return CS_EINVAL;
    if (mode == CS_MATCH_EXACT) {
        if (!cs_view_is_valid(a) || !cs_view_is_valid(b)) return CS_EINVAL;
        *out_equal = cs_equal(a, b);
        return CS_OK;
    }
    int cmp = 0;
    cs_status st = cs_compare_mode(a, b, mode, &cmp);
    if (st != CS_OK) return st;
    *out_equal = (cmp == 0);
    return CS_OK;
}

/* Natural comparison: compares arbitrary-length ASCII digit runs numerically
 * without integer conversion. Leading-zero difference is used as a deterministic
 * tie-breaker when numeric values are identical (fewer leading zeros < more leading zeros,
 * or if full strings match except leading zeros, first non-zero zero-count diff wins). */
int cs_compare_natural(cs_view a, cs_view b) {
    const uint8_t *pa = (const uint8_t *)a.data;
    const uint8_t *pb = (const uint8_t *)b.data;
    size_t ia = 0, ib = 0;
    int zero_tie_breaker = 0;

    while (ia < a.len && ib < b.len) {
        uint8_t ca = pa[ia];
        uint8_t cb = pb[ib];
        bool da = (ca >= '0' && ca <= '9');
        bool db = (cb >= '0' && cb <= '9');

        if (da && db) {
            size_t za_start = ia;
            while (ia < a.len && pa[ia] == '0') ia++;
            size_t za_count = ia - za_start;
            size_t sig_a = ia;
            while (ia < a.len && pa[ia] >= '0' && pa[ia] <= '9') ia++;
            size_t len_a = ia - sig_a;

            size_t zb_start = ib;
            while (ib < b.len && pb[ib] == '0') ib++;
            size_t zb_count = ib - zb_start;
            size_t sig_b = ib;
            while (ib < b.len && pb[ib] >= '0' && pb[ib] <= '9') ib++;
            size_t len_b = ib - sig_b;

            if (len_a != len_b) {
                return (len_a < len_b) ? -1 : 1;
            }
            if (len_a > 0) {
                int m = memcmp(pa + sig_a, pb + sig_b, len_a);
                if (m != 0) return (m < 0) ? -1 : 1;
            }
            if (zero_tie_breaker == 0 && za_count != zb_count) {
                zero_tie_breaker = (za_count < zb_count) ? -1 : 1;
            }
        } else {
            if (ca != cb) {
                return (ca < cb) ? -1 : 1;
            }
            ia++;
            ib++;
        }
    }
    if (ia < a.len) return 1;
    if (ib < b.len) return -1;
    return zero_tie_breaker;
}

/* Optional OS-backed localized collation */
cs_status cs_compare_collation(cs_view a, cs_view b, const char *locale_name, int *out_cmp) {
    if (!cs_view_is_valid(a) || !cs_view_is_valid(b) || !locale_name || !out_cmp) return CS_EINVAL;
    /* Reject embedded NULs if backend uses C-strings */
    if ((a.len > 0 && memchr(a.data, '\0', a.len) != NULL) ||
        (b.len > 0 && memchr(b.data, '\0', b.len) != NULL)) {
        return CS_EINVAL;
    }
#if defined(__linux__) || defined(__APPLE__)
    locale_t loc = newlocale(LC_COLLATE_MASK, locale_name, (locale_t)0);
    if (!loc) return CS_EUNSUPPORTED;
    cs_string sa, sb;
    cs_status st = cs_from_view(&sa, a, NULL);
    if (st != CS_OK) { freelocale(loc); return st; }
    st = cs_from_view(&sb, b, NULL);
    if (st != CS_OK) { cs_destroy(&sa); freelocale(loc); return st; }
    int r = strcoll_l(cs_c_str(&sa), cs_c_str(&sb), loc);
    *out_cmp = (r < 0) ? -1 : ((r > 0) ? 1 : 0);
    cs_destroy(&sa);
    cs_destroy(&sb);
    freelocale(loc);
    return CS_OK;
#else
    (void)locale_name;
    return CS_EUNSUPPORTED;
#endif
}

/* ==========================================================================
 * Two-Way O(n + m) Substring Search & Unicode Casefold Whole-Codepoint Search
 * ========================================================================== */

static void cs_two_way_factorize(const uint8_t *needle, size_t m, cs_match_mode mode,
                                 size_t *out_crit, size_t *out_period, bool *out_periodic) {
    /* Compute maximal suffix under <= and >= orders */
    size_t ms1 = SIZE_MAX, j1 = 0, k1 = 1, p1 = 1;
    while (j1 + k1 < m) {
        uint8_t a = (mode == CS_MATCH_ASCII_CASE_INSENSITIVE) ? cs_ascii_fold_byte(needle[j1 + k1]) : needle[j1 + k1];
        uint8_t b = (mode == CS_MATCH_ASCII_CASE_INSENSITIVE) ? cs_ascii_fold_byte(needle[(ms1 == SIZE_MAX ? 0 : ms1 + 1) + k1 - 1])
                                                              : needle[(ms1 == SIZE_MAX ? 0 : ms1 + 1) + k1 - 1];
        if (a < b) {
            j1 += k1;
            k1 = 1;
            p1 = j1 - ms1;
        } else if (a == b) {
            if (k1 != p1) k1++;
            else { j1 += p1; k1 = 1; }
        } else {
            ms1 = j1;
            j1 = ms1 + 1;
            k1 = p1 = 1;
        }
    }

    size_t ms2 = SIZE_MAX, j2 = 0, k2 = 1, p2 = 1;
    while (j2 + k2 < m) {
        uint8_t a = (mode == CS_MATCH_ASCII_CASE_INSENSITIVE) ? cs_ascii_fold_byte(needle[j2 + k2]) : needle[j2 + k2];
        uint8_t b = (mode == CS_MATCH_ASCII_CASE_INSENSITIVE) ? cs_ascii_fold_byte(needle[(ms2 == SIZE_MAX ? 0 : ms2 + 1) + k2 - 1])
                                                              : needle[(ms2 == SIZE_MAX ? 0 : ms2 + 1) + k2 - 1];
        if (b < a) {
            j2 += k2;
            k2 = 1;
            p2 = j2 - ms2;
        } else if (a == b) {
            if (k2 != p2) k2++;
            else { j2 += p2; k2 = 1; }
        } else {
            ms2 = j2;
            j2 = ms2 + 1;
            k2 = p2 = 1;
        }
    }

    size_t crit = 0, per = 1;
    if ((ms1 == SIZE_MAX ? 0 : ms1 + 1) >= (ms2 == SIZE_MAX ? 0 : ms2 + 1)) {
        crit = (ms1 == SIZE_MAX) ? 0 : (ms1 + 1);
        per  = p1;
    } else {
        crit = (ms2 == SIZE_MAX) ? 0 : (ms2 + 1);
        per  = p2;
    }

    bool periodic = true;
    for (size_t i = 0; i < crit; i++) {
        uint8_t c1 = (mode == CS_MATCH_ASCII_CASE_INSENSITIVE) ? cs_ascii_fold_byte(needle[i]) : needle[i];
        uint8_t c2 = (mode == CS_MATCH_ASCII_CASE_INSENSITIVE) ? cs_ascii_fold_byte(needle[i + per]) : needle[i + per];
        if (c1 != c2) {
            periodic = false;
            break;
        }
    }
    if (!periodic) {
        size_t left = crit;
        size_t right = m - crit;
        per = ((left > right) ? left : right) + 1U;
    }
    *out_crit     = crit;
    *out_period   = per;
    *out_periodic = periodic;
}

static size_t cs_two_way_search_bytes(const uint8_t *hay, size_t n, const uint8_t *needle, size_t m,
                                      size_t crit, size_t period, bool periodic, cs_match_mode mode) {
    if (m == 0) return 0;
    if (n < m) return SIZE_MAX;
    if (m == 1) {
        if (mode == CS_MATCH_EXACT) {
            return cs_dispatch_find_byte(hay, n, needle[0]);
        }
        uint8_t nb = cs_ascii_fold_byte(needle[0]);
        for (size_t i = 0; i < n; i++) {
            if (cs_ascii_fold_byte(hay[i]) == nb) return i;
        }
        return SIZE_MAX;
    }

    size_t pos = 0;
    size_t memory = 0;
    while (pos <= n - m) {
        /* SIMD prefilter on first byte only when memory == 0 and mode == EXACT */
        if (memory == 0 && mode == CS_MATCH_EXACT) {
            size_t skip = cs_dispatch_find_byte(hay + pos, (n - m + 1U) - pos, needle[0]);
            if (skip == SIZE_MAX) return SIZE_MAX;
            pos += skip;
        }

        /* Scan right part [max(crit, memory) .. m - 1] */
        size_t i = (crit > memory) ? crit : memory;
        while (i < m) {
            uint8_t hc = (mode == CS_MATCH_ASCII_CASE_INSENSITIVE) ? cs_ascii_fold_byte(hay[pos + i]) : hay[pos + i];
            uint8_t nc = (mode == CS_MATCH_ASCII_CASE_INSENSITIVE) ? cs_ascii_fold_byte(needle[i])      : needle[i];
            if (hc != nc) break;
            i++;
        }
        if (i < m) {
            pos += (i - crit + 1U);
            memory = 0;
            continue;
        }

        /* Scan left part [crit - 1 down to memory] */
        size_t j = crit;
        while (j > memory) {
            uint8_t hc = (mode == CS_MATCH_ASCII_CASE_INSENSITIVE) ? cs_ascii_fold_byte(hay[pos + j - 1U]) : hay[pos + j - 1U];
            uint8_t nc = (mode == CS_MATCH_ASCII_CASE_INSENSITIVE) ? cs_ascii_fold_byte(needle[j - 1U])      : needle[j - 1U];
            if (hc != nc) break;
            j--;
        }
        if (j <= memory) {
            return pos;
        }
        pos += period;
        memory = periodic ? (m - period) : 0;
    }
    return SIZE_MAX;
}

#if CS_ENABLE_UNICODE
/* Unicode Casefold Search that strictly matches whole source code point boundaries
 * and returns the exact byte span [out_start, out_end) in `text`. */
static cs_status cs_find_unicode_casefold(cs_view text, cs_view needle, size_t start_offset,
                                          bool reverse, size_t limit_offset,
                                          const cs_allocator *custom_alloc,
                                          size_t *out_start, size_t *out_end) {
    /* Validate start/limit offset alignment */
    size_t dummy_idx = 0;
    if (!reverse) {
        cs_status st = cs_offset_to_cp_index(text, start_offset, &dummy_idx);
        if (st != CS_OK) return st;
    } else {
        cs_status st = cs_offset_to_cp_index(text, limit_offset, &dummy_idx);
        if (st != CS_OK) return st;
    }

    if (needle.len == 0) {
        /* Validate remainder of text as well */
        cs_status st = cs_validate_utf8(text, NULL);
        if (st != CS_OK) return st;
        size_t p = reverse ? limit_offset : start_offset;
        if (out_start) *out_start = p;
        if (out_end)   *out_end   = p;
        return CS_OK;
    }

    /* Fold needle into UTF-32 buffer */
    uint32_t stack_needle[128];
    uint32_t *n_cps = stack_needle;
    size_t n_cap = 128U, n_len = 0;
    cs_allocator alloc = (custom_alloc && custom_alloc->alloc && custom_alloc->free) ? *custom_alloc : cs_default_allocator();

    cs_cp_iter nit;
    cs_cp_iter_init(&nit, needle, CS_DECODE_STRICT);
    uint32_t cp = 0;
    cs_status st;
    while ((st = cs_cp_iter_next(&nit, &cp, NULL, NULL)) == CS_OK) {
        uint32_t f[3];
        size_t flen = cs_fold_cp_full(cp, CS_LOCALE_ROOT, f);
        if (n_len + flen > n_cap) {
            size_t new_cap = n_cap * 2U;
            uint32_t *new_buf = (uint32_t *)alloc.alloc(alloc.ctx, new_cap * sizeof(uint32_t), CS_ALIGNOF(uint32_t));
            if (!new_buf) {
                if (n_cps != stack_needle) alloc.free(alloc.ctx, n_cps, n_cap * sizeof(uint32_t), CS_ALIGNOF(uint32_t));
                return CS_ENOMEM;
            }
            memcpy(new_buf, n_cps, n_len * sizeof(uint32_t));
            if (n_cps != stack_needle) alloc.free(alloc.ctx, n_cps, n_cap * sizeof(uint32_t), CS_ALIGNOF(uint32_t));
            n_cps = new_buf;
            n_cap = new_cap;
        }
        for (size_t k = 0; k < flen; k++) n_cps[n_len++] = f[k];
    }
    if (st != CS_END) {
        if (n_cps != stack_needle) alloc.free(alloc.ctx, n_cps, n_cap * sizeof(uint32_t), CS_ALIGNOF(uint32_t));
        return st;
    }

    /* Validate full text UTF-8 first */
    st = cs_validate_utf8(text, NULL);
    if (st != CS_OK) {
        if (n_cps != stack_needle) alloc.free(alloc.ctx, n_cps, n_cap * sizeof(uint32_t), CS_ALIGNOF(uint32_t));
        return st;
    }

    cs_cp_iter tit;
    cs_cp_iter_init(&tit, text, CS_DECODE_STRICT);
    tit.pos = reverse ? 0 : start_offset;
    size_t scan_limit = reverse ? limit_offset : text.len;

    bool found = false;
    size_t best_start = 0, best_end = 0;

    while (tit.pos < scan_limit) {
        size_t cand_start = tit.pos;
        cs_cp_iter sub = tit;
        size_t n_idx = 0;
        size_t cand_end = cand_start;
        bool ok = true;

        while (n_idx < n_len) {
            if (sub.pos >= scan_limit) { ok = false; break; }
            uint32_t tcp = 0;
            size_t ts = 0, te = 0;
            cs_cp_iter_next(&sub, &tcp, &ts, &te);
            uint32_t tf[3];
            size_t tflen = cs_fold_cp_full(tcp, CS_LOCALE_ROOT, tf);
            if (n_idx + tflen > n_len) {
                /* Expansion of source code point overflows remaining needle -> half-match forbidden! */
                ok = false;
                break;
            }
            for (size_t k = 0; k < tflen; k++) {
                if (tf[k] != n_cps[n_idx + k]) {
                    ok = false;
                    break;
                }
            }
            if (!ok) break;
            n_idx += tflen;
            cand_end = te;
        }

        if (ok && n_idx == n_len && cand_end <= scan_limit) {
            found = true;
            best_start = cand_start;
            best_end   = cand_end;
            if (!reverse) break;
        }
        /* Advance outer iterator by one code point */
        uint32_t dummy_cp = 0;
        cs_cp_iter_next(&tit, &dummy_cp, NULL, NULL);
    }

    if (n_cps != stack_needle) alloc.free(alloc.ctx, n_cps, n_cap * sizeof(uint32_t), CS_ALIGNOF(uint32_t));
    if (!found) return CS_NOT_FOUND;
    if (out_start) *out_start = best_start;
    if (out_end)   *out_end   = best_end;
    return CS_OK;
}
#endif /* CS_ENABLE_UNICODE */

cs_status cs_searcher_init(cs_searcher *sr, cs_view needle, cs_match_mode mode, const cs_allocator *alloc) {
    if (!sr || !cs_view_is_valid(needle)) return CS_EINVAL;
    memset(sr, 0, sizeof(*sr));
    sr->needle = needle;
    sr->mode   = mode;
    if (alloc) cs_init_with_allocator(&sr->owned_folded, *alloc);
    else cs_init(&sr->owned_folded);

    if (mode == CS_MATCH_UNICODE_CASEFOLD) {
#if !CS_ENABLE_UNICODE
        return CS_EUNSUPPORTED;
#else
        return cs_validate_utf8(needle, NULL);
#endif
    }
    if (needle.len > 0) {
        cs_two_way_factorize((const uint8_t *)needle.data, needle.len, mode,
                             &sr->crit_pos, &sr->period, &sr->periodic);
        sr->first_byte = (uint8_t)needle.data[0];
        sr->last_byte  = (uint8_t)needle.data[needle.len - 1U];
    }
    return CS_OK;
}

void cs_searcher_destroy(cs_searcher *sr) {
    if (!sr) return;
    cs_destroy(&sr->owned_folded);
    memset(sr, 0, sizeof(*sr));
}

cs_status cs_searcher_find(const cs_searcher *sr, cs_view text, size_t start_offset, size_t *out_start, size_t *out_end) {
    if (!sr || !cs_view_is_valid(text)) return CS_EINVAL;
    if (start_offset > text.len) return CS_ERANGE;

    if (sr->mode == CS_MATCH_UNICODE_CASEFOLD) {
#if !CS_ENABLE_UNICODE
        return CS_EUNSUPPORTED;
#else
        return cs_find_unicode_casefold(text, sr->needle, start_offset, false, text.len, &sr->owned_folded.alloc, out_start, out_end);
#endif
    }

    if (sr->needle.len == 0) {
        if (out_start) *out_start = start_offset;
        if (out_end)   *out_end   = start_offset;
        return CS_OK;
    }
    if (text.len - start_offset < sr->needle.len) return CS_NOT_FOUND;

    size_t rel = cs_two_way_search_bytes((const uint8_t *)text.data + start_offset,
                                         text.len - start_offset,
                                         (const uint8_t *)sr->needle.data,
                                         sr->needle.len,
                                         sr->crit_pos, sr->period, sr->periodic, sr->mode);
    if (rel == SIZE_MAX) return CS_NOT_FOUND;
    size_t pos = start_offset + rel;
    if (out_start) *out_start = pos;
    if (out_end)   *out_end   = pos + sr->needle.len;
    return CS_OK;
}

cs_status cs_searcher_rfind(const cs_searcher *sr, cs_view text, size_t end_offset, size_t *out_start, size_t *out_end) {
    if (!sr || !cs_view_is_valid(text)) return CS_EINVAL;
    if (end_offset > text.len) return CS_ERANGE;

    if (sr->mode == CS_MATCH_UNICODE_CASEFOLD) {
#if !CS_ENABLE_UNICODE
        return CS_EUNSUPPORTED;
#else
        return cs_find_unicode_casefold(text, sr->needle, 0, true, end_offset, &sr->owned_folded.alloc, out_start, out_end);
#endif
    }

    if (sr->needle.len == 0) {
        if (out_start) *out_start = end_offset;
        if (out_end)   *out_end   = end_offset;
        return CS_OK;
    }
    if (end_offset < sr->needle.len) return CS_NOT_FOUND;

    const uint8_t *hay = (const uint8_t *)text.data;
    const uint8_t *ndl = (const uint8_t *)sr->needle.data;
    size_t m = sr->needle.len;

    if (sr->mode == CS_MATCH_EXACT && m == 1U) {
        size_t pos = cs_dispatch_rfind_byte(hay, end_offset, ndl[0]);
        if (pos == SIZE_MAX) return CS_NOT_FOUND;
        if (out_start) *out_start = pos;
        if (out_end)   *out_end   = pos + 1U;
        return CS_OK;
    }

    for (size_t pos = end_offset - m + 1U; pos > 0; pos--) {
        size_t idx = pos - 1U;
        if (sr->mode == CS_MATCH_EXACT) {
            if (hay[idx] == ndl[0] && hay[idx + m - 1U] == ndl[m - 1U] &&
                cs_dispatch_equal(hay + idx, ndl, m)) {
                if (out_start) *out_start = idx;
                if (out_end)   *out_end   = idx + m;
                return CS_OK;
            }
        } else {
            bool ok = true;
            for (size_t k = 0; k < m; k++) {
                if (cs_ascii_fold_byte(hay[idx + k]) != cs_ascii_fold_byte(ndl[k])) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                if (out_start) *out_start = idx;
                if (out_end)   *out_end   = idx + m;
                return CS_OK;
            }
        }
    }
    return CS_NOT_FOUND;
}

cs_status cs_find(cs_view text, cs_view needle, size_t start_offset, size_t *out_pos) {
    return cs_find_mode(text, needle, start_offset, CS_MATCH_EXACT, out_pos, NULL);
}

cs_status cs_find_mode(cs_view text, cs_view needle, size_t start_offset, cs_match_mode mode, size_t *out_start, size_t *out_end) {
    cs_searcher sr;
    cs_status st = cs_searcher_init(&sr, needle, mode, NULL);
    if (st != CS_OK) return st;
    st = cs_searcher_find(&sr, text, start_offset, out_start, out_end);
    cs_searcher_destroy(&sr);
    return st;
}

cs_status cs_rfind(cs_view text, cs_view needle, size_t end_offset, size_t *out_pos) {
    return cs_rfind_mode(text, needle, end_offset, CS_MATCH_EXACT, out_pos, NULL);
}

cs_status cs_rfind_mode(cs_view text, cs_view needle, size_t end_offset, cs_match_mode mode, size_t *out_start, size_t *out_end) {
    cs_searcher sr;
    cs_status st = cs_searcher_init(&sr, needle, mode, NULL);
    if (st != CS_OK) return st;
    st = cs_searcher_rfind(&sr, text, end_offset, out_start, out_end);
    cs_searcher_destroy(&sr);
    return st;
}

cs_status cs_find_byte(cs_view text, uint8_t byte_val, size_t start_offset, size_t *out_pos) {
    if (!cs_view_is_valid(text)) return CS_EINVAL;
    if (start_offset > text.len) return CS_ERANGE;
    if (start_offset == text.len) return CS_NOT_FOUND;
    size_t rel = cs_dispatch_find_byte((const uint8_t *)text.data + start_offset, text.len - start_offset, byte_val);
    if (rel == SIZE_MAX) return CS_NOT_FOUND;
    if (out_pos) *out_pos = start_offset + rel;
    return CS_OK;
}

cs_status cs_rfind_byte(cs_view text, uint8_t byte_val, size_t end_offset, size_t *out_pos) {
    if (!cs_view_is_valid(text)) return CS_EINVAL;
    if (end_offset > text.len) return CS_ERANGE;
    if (end_offset == 0) return CS_NOT_FOUND;
    size_t pos = cs_dispatch_rfind_byte((const uint8_t *)text.data, end_offset, byte_val);
    if (pos == SIZE_MAX) return CS_NOT_FOUND;
    if (out_pos) *out_pos = pos;
    return CS_OK;
}

cs_status cs_find_any_byte(cs_view text, cs_view byte_set, size_t start_offset, size_t *out_pos) {
    if (!cs_view_is_valid(text) || !cs_view_is_valid(byte_set)) return CS_EINVAL;
    if (start_offset > text.len) return CS_ERANGE;
    if (byte_set.len == 0 || start_offset == text.len) return CS_NOT_FOUND;

    uint64_t bitmap[4] = {0, 0, 0, 0};
    for (size_t i = 0; i < byte_set.len; i++) {
        uint8_t b = (uint8_t)byte_set.data[i];
        bitmap[b >> 6] |= (1ULL << (b & 63U));
    }
    const uint8_t *p = (const uint8_t *)text.data;
    for (size_t i = start_offset; i < text.len; i++) {
        uint8_t b = p[i];
        if ((bitmap[b >> 6] & (1ULL << (b & 63U))) != 0) {
            if (out_pos) *out_pos = i;
            return CS_OK;
        }
    }
    return CS_NOT_FOUND;
}

cs_status cs_find_not_any_byte(cs_view text, cs_view byte_set, size_t start_offset, size_t *out_pos) {
    if (!cs_view_is_valid(text) || !cs_view_is_valid(byte_set)) return CS_EINVAL;
    if (start_offset > text.len) return CS_ERANGE;
    if (start_offset == text.len) return CS_NOT_FOUND;

    uint64_t bitmap[4] = {0, 0, 0, 0};
    for (size_t i = 0; i < byte_set.len; i++) {
        uint8_t b = (uint8_t)byte_set.data[i];
        bitmap[b >> 6] |= (1ULL << (b & 63U));
    }
    const uint8_t *p = (const uint8_t *)text.data;
    for (size_t i = start_offset; i < text.len; i++) {
        uint8_t b = p[i];
        if ((bitmap[b >> 6] & (1ULL << (b & 63U))) == 0) {
            if (out_pos) *out_pos = i;
            return CS_OK;
        }
    }
    return CS_NOT_FOUND;
}

bool cs_starts_with(cs_view text, cs_view prefix) {
    if (!cs_view_is_valid(text) || !cs_view_is_valid(prefix)) return false;
    if (prefix.len > text.len) return false;
    return cs_dispatch_equal((const uint8_t *)text.data, (const uint8_t *)prefix.data, prefix.len);
}

cs_status cs_starts_with_mode(cs_view text, cs_view prefix, cs_match_mode mode, bool *out_result, size_t *out_match_end) {
    if (!cs_view_is_valid(text) || !cs_view_is_valid(prefix) || !out_result) return CS_EINVAL;
    if (mode != CS_MATCH_UNICODE_CASEFOLD) {
        if (prefix.len > text.len) {
            *out_result = false;
            return CS_OK;
        }
        cs_view head = { text.data, prefix.len };
        cs_status st = cs_equal_mode(head, prefix, mode, out_result);
        if (st == CS_OK && *out_result && out_match_end) *out_match_end = prefix.len;
        return st;
    }
    size_t s = 0, e = 0;
    cs_status st = cs_find_mode(text, prefix, 0, CS_MATCH_UNICODE_CASEFOLD, &s, &e);
    if (st == CS_OK && s == 0) {
        *out_result = true;
        if (out_match_end) *out_match_end = e;
        return CS_OK;
    }
    if (st == CS_OK || st == CS_NOT_FOUND) {
        *out_result = false;
        return CS_OK;
    }
    return st;
}

bool cs_ends_with(cs_view text, cs_view suffix) {
    if (!cs_view_is_valid(text) || !cs_view_is_valid(suffix)) return false;
    if (suffix.len > text.len) return false;
    return cs_dispatch_equal((const uint8_t *)text.data + (text.len - suffix.len),
                             (const uint8_t *)suffix.data, suffix.len);
}

cs_status cs_ends_with_mode(cs_view text, cs_view suffix, cs_match_mode mode, bool *out_result, size_t *out_match_start) {
    if (!cs_view_is_valid(text) || !cs_view_is_valid(suffix) || !out_result) return CS_EINVAL;
    if (mode != CS_MATCH_UNICODE_CASEFOLD) {
        if (suffix.len > text.len) {
            *out_result = false;
            return CS_OK;
        }
        size_t start = text.len - suffix.len;
        cs_view tail = { text.data ? (text.data + start) : NULL, suffix.len };
        cs_status st = cs_equal_mode(tail, suffix, mode, out_result);
        if (st == CS_OK && *out_result && out_match_start) *out_match_start = start;
        return st;
    }
    size_t s = 0, e = 0;
    cs_status st = cs_rfind_mode(text, suffix, text.len, CS_MATCH_UNICODE_CASEFOLD, &s, &e);
    if (st == CS_OK && e == text.len) {
        *out_result = true;
        if (out_match_start) *out_match_start = s;
        return CS_OK;
    }
    if (st == CS_OK || st == CS_NOT_FOUND) {
        *out_result = false;
        return CS_OK;
    }
    return st;
}

bool cs_contains(cs_view text, cs_view needle) {
    size_t pos = 0;
    return cs_find(text, needle, 0, &pos) == CS_OK;
}

cs_status cs_contains_mode(cs_view text, cs_view needle, cs_match_mode mode, bool *out_result) {
    if (!out_result) return CS_EINVAL;
    size_t s = 0, e = 0;
    cs_status st = cs_find_mode(text, needle, 0, mode, &s, &e);
    if (st == CS_OK) { *out_result = true; return CS_OK; }
    if (st == CS_NOT_FOUND) { *out_result = false; return CS_OK; }
    return st;
}

cs_status cs_find_all(cs_view text, cs_view needle, cs_match_mode mode, cs_overlap_mode overlap,
                      size_t *out_starts, size_t *out_ends, size_t max_matches, size_t *out_total_count) {
    if (!cs_view_is_valid(text) || !cs_view_is_valid(needle) || !out_total_count) return CS_EINVAL;
    /* Section 7 contract: empty pattern in count/find_all returns CS_EINVAL */
    if (needle.len == 0) return CS_EINVAL;

    cs_searcher sr;
    cs_status st = cs_searcher_init(&sr, needle, mode, NULL);
    if (st != CS_OK) return st;

    size_t count = 0;
    size_t pos   = 0;
    while (pos <= text.len) {
        size_t ms = 0, me = 0;
        st = cs_searcher_find(&sr, text, pos, &ms, &me);
        if (st == CS_NOT_FOUND) {
            st = CS_OK;
            break;
        }
        if (st != CS_OK) break;

        if (count < max_matches) {
            if (out_starts) out_starts[count] = ms;
            if (out_ends)   out_ends[count]   = me;
        }
        count++;

        if (overlap == CS_OVERLAP_DISALLOW) {
            pos = me;
        } else {
            if (mode == CS_MATCH_UNICODE_CASEFOLD) {
                uint32_t dummy = 0;
                size_t next_off = ms + 1U;
                cs_decode_utf8_next(text, ms, &dummy, &next_off);
                pos = next_off;
            } else {
                pos = ms + 1U;
            }
        }
    }
    cs_searcher_destroy(&sr);
    if (st != CS_OK) return st;
    *out_total_count = count;
    return CS_OK;
}

cs_status cs_count(cs_view text, cs_view needle, cs_overlap_mode overlap, size_t *out_count) {
    return cs_find_all(text, needle, CS_MATCH_EXACT, overlap, NULL, NULL, 0, out_count);
}

cs_status cs_count_mode(cs_view text, cs_view needle, cs_match_mode mode, cs_overlap_mode overlap, size_t *out_count) {
    return cs_find_all(text, needle, mode, overlap, NULL, NULL, 0, out_count);
}

cs_view cs_remove_prefix(cs_view text, cs_view prefix) {
    if (cs_starts_with(text, prefix)) {
        cs_view res = { text.data ? (text.data + prefix.len) : NULL, text.len - prefix.len };
        return res;
    }
    return text;
}

cs_view cs_remove_suffix(cs_view text, cs_view suffix) {
    if (cs_ends_with(text, suffix)) {
        cs_view res = { text.data, text.len - suffix.len };
        return res;
    }
    return text;
}

cs_view cs_common_prefix_bytes(cs_view a, cs_view b) {
    if (!cs_view_is_valid(a) || !cs_view_is_valid(b)) return cs_empty_view();
    size_t min_len = (a.len < b.len) ? a.len : b.len;
    size_t i = 0;
    while (i < min_len && a.data[i] == b.data[i]) i++;
    cs_view res = { a.data, i };
    return res;
}

cs_view cs_common_suffix_bytes(cs_view a, cs_view b) {
    if (!cs_view_is_valid(a) || !cs_view_is_valid(b)) return cs_empty_view();
    size_t min_len = (a.len < b.len) ? a.len : b.len;
    size_t i = 0;
    while (i < min_len && a.data[a.len - 1U - i] == b.data[b.len - 1U - i]) i++;
    cs_view res = { a.data ? (a.data + (a.len - i)) : NULL, i };
    return res;
}

cs_status cs_common_prefix_codepoints(cs_view a, cs_view b, cs_view *out) {
    if (!cs_view_is_valid(a) || !cs_view_is_valid(b) || !out) return CS_EINVAL;
    cs_status st = cs_validate_utf8(a, NULL);
    if (st != CS_OK) return st;
    st = cs_validate_utf8(b, NULL);
    if (st != CS_OK) return st;

    cs_cp_iter ita, itb;
    cs_cp_iter_init(&ita, a, CS_DECODE_STRICT);
    cs_cp_iter_init(&itb, b, CS_DECODE_STRICT);
    size_t end_off = 0;
    for (;;) {
        uint32_t ca = 0, cb = 0;
        size_t ea = 0, eb = 0;
        cs_status sa = cs_cp_iter_next(&ita, &ca, NULL, &ea);
        cs_status sb = cs_cp_iter_next(&itb, &cb, NULL, &eb);
        if (sa != CS_OK || sb != CS_OK || ca != cb) break;
        end_off = ea;
    }
    out->data = a.data;
    out->len  = end_off;
    return CS_OK;
}

cs_status cs_common_suffix_codepoints(cs_view a, cs_view b, cs_view *out) {
    if (!cs_view_is_valid(a) || !cs_view_is_valid(b) || !out) return CS_EINVAL;
    cs_status st = cs_validate_utf8(a, NULL);
    if (st != CS_OK) return st;
    st = cs_validate_utf8(b, NULL);
    if (st != CS_OK) return st;

    cs_cp_iter ita, itb;
    cs_cp_iter_init(&ita, a, CS_DECODE_STRICT);
    cs_cp_iter_init(&itb, b, CS_DECODE_STRICT);
    ita.pos = a.len;
    itb.pos = b.len;
    size_t start_off = a.len;
    for (;;) {
        uint32_t ca = 0, cb = 0;
        size_t sa_off = 0, sb_off = 0;
        cs_status sa = cs_cp_iter_prev(&ita, &ca, &sa_off, NULL);
        cs_status sb = cs_cp_iter_prev(&itb, &cb, &sb_off, NULL);
        if (sa != CS_OK || sb != CS_OK || ca != cb) break;
        start_off = sa_off;
    }
    out->data = a.data ? (a.data + start_off) : NULL;
    out->len  = a.len - start_off;
    return CS_OK;
}

cs_status cs_common_prefix_graphemes(cs_view a, cs_view b, cs_view *out) {
    if (!cs_view_is_valid(a) || !cs_view_is_valid(b) || !out) return CS_EINVAL;
    cs_status st = cs_validate_utf8(a, NULL);
    if (st != CS_OK) return st;
    st = cs_validate_utf8(b, NULL);
    if (st != CS_OK) return st;

    cs_grapheme_iter ita, itb;
    cs_grapheme_iter_init(&ita, a, CS_DECODE_STRICT);
    cs_grapheme_iter_init(&itb, b, CS_DECODE_STRICT);
    size_t end_off = 0;
    for (;;) {
        cs_view ga = cs_empty_view(), gb = cs_empty_view();
        size_t ea = 0, eb = 0;
        cs_status sa = cs_grapheme_iter_next(&ita, &ga, NULL, &ea);
        cs_status sb = cs_grapheme_iter_next(&itb, &gb, NULL, &eb);
        if (sa != CS_OK || sb != CS_OK || !cs_equal(ga, gb)) break;
        end_off = ea;
    }
    out->data = a.data;
    out->len  = end_off;
    return CS_OK;
}

cs_status cs_common_suffix_graphemes(cs_view a, cs_view b, cs_view *out) {
    if (!cs_view_is_valid(a) || !cs_view_is_valid(b) || !out) return CS_EINVAL;
    cs_status st = cs_validate_utf8(a, NULL);
    if (st != CS_OK) return st;
    st = cs_validate_utf8(b, NULL);
    if (st != CS_OK) return st;

    cs_grapheme_iter ita, itb;
    cs_grapheme_iter_init(&ita, a, CS_DECODE_STRICT);
    cs_grapheme_iter_init(&itb, b, CS_DECODE_STRICT);
    ita.pos = a.len;
    itb.pos = b.len;
    size_t start_off = a.len;
    for (;;) {
        cs_view ga = cs_empty_view(), gb = cs_empty_view();
        size_t sa_off = 0, sb_off = 0;
        cs_status sa = cs_grapheme_iter_prev(&ita, &ga, &sa_off, NULL);
        cs_status sb = cs_grapheme_iter_prev(&itb, &gb, &sb_off, NULL);
        if (sa != CS_OK || sb != CS_OK || !cs_equal(ga, gb)) break;
        start_off = sa_off;
    }
    out->data = a.data ? (a.data + start_off) : NULL;
    out->len  = a.len - start_off;
    return CS_OK;
}

/* ==========================================================================
 * Section 8: Split, Join & Line/Whitespace Tokenization
 * ========================================================================== */

cs_split_options cs_default_split_options(void) {
    cs_split_options o;
    o.keep_empty  = true;
    o.max_splits  = CS_UNLIMITED_SPLITS;
    o.strip_outer = false;
    return o;
}

cs_status cs_split_iter_init(cs_split_iter *it, cs_view text, cs_view delim, cs_split_options opts) {
    if (!it || !cs_view_is_valid(text) || !cs_view_is_valid(delim)) return CS_EINVAL;
    /* Section 7/8 contract: split with empty pattern returns CS_EINVAL */
    if (delim.len == 0) return CS_EINVAL;

    if (opts.strip_outer) {
        if (cs_starts_with(text, delim)) {
            text.data += delim.len;
            text.len  -= delim.len;
        }
        if (cs_ends_with(text, delim)) {
            text.len -= delim.len;
        }
    }

    memset(it, 0, sizeof(*it));
    it->text = text;
    it->delim = delim;
    it->opts = opts;
    it->kind = 0;
    it->finished = false;
    return CS_OK;
}

cs_status cs_split_ascii_whitespace_iter_init(cs_split_iter *it, cs_view text, size_t max_splits) {
    if (!it || !cs_view_is_valid(text)) return CS_EINVAL;
    memset(it, 0, sizeof(*it));
    it->text = text;
    it->opts.keep_empty = false;
    it->opts.max_splits = max_splits;
    it->kind = 1;
    return CS_OK;
}

cs_status cs_split_unicode_whitespace_iter_init(cs_split_iter *it, cs_view text, size_t max_splits) {
    if (!it || !cs_view_is_valid(text)) return CS_EINVAL;
    cs_status st = cs_validate_utf8(text, NULL);
    if (st != CS_OK) return st;
    memset(it, 0, sizeof(*it));
    it->text = text;
    it->opts.keep_empty = false;
    it->opts.max_splits = max_splits;
    it->kind = 2;
    return CS_OK;
}

cs_status cs_split_lines_iter_init(cs_split_iter *it, cs_view text, cs_line_mode mode, bool keep_endings) {
    if (!it || !cs_view_is_valid(text)) return CS_EINVAL;
    if (mode == CS_LINES_UNICODE) {
        cs_status st = cs_validate_utf8(text, NULL);
        if (st != CS_OK) return st;
    }
    memset(it, 0, sizeof(*it));
    it->text = text;
    it->kind = 3;
    it->line_mode = mode;
    it->keep_line_endings = keep_endings;
    it->finished = (text.len == 0);
    return CS_OK;
}

static inline bool cs_is_ascii_ws_byte(uint8_t b) {
    return b == ' ' || b == '\t' || b == '\n' || b == '\r' || b == '\v' || b == '\f';
}

/* Match line break at text[pos..]. Returns length of line break in bytes (1, 2, or 3), or 0 if not a line break. */
static size_t cs_match_line_sep(cs_view text, size_t pos, cs_line_mode mode) {
    if (pos >= text.len) return 0;
    const uint8_t *p = (const uint8_t *)text.data + pos;
    size_t rem = text.len - pos;
    if (p[0] == '\r') {
        return (rem >= 2U && p[1] == '\n') ? 2U : 1U;
    }
    if (p[0] == '\n') return 1U;
    if (mode == CS_LINES_UNICODE) {
        if (p[0] == 0x0BU || p[0] == 0x0CU) return 1U; /* VT, FF */
        if (rem >= 2U && p[0] == 0xC2U && p[1] == 0x85U) return 2U; /* NEL U+0085 */
        if (rem >= 3U && p[0] == 0xE2U && p[1] == 0x80U && (p[2] == 0xA8U || p[2] == 0xA9U)) {
            return 3U; /* LS U+2028, PS U+2029 */
        }
    }
    return 0;
}

cs_status cs_split_iter_next(cs_split_iter *it, cs_view *out_part) {
    if (!it || !out_part) return CS_EINVAL;
    if (it->finished) return CS_END;

    if (it->kind == 0) {
        /* Delimiter split */
        while (!it->finished) {
            if (it->splits_done >= it->opts.max_splits) {
                cs_view tail = { it->text.data ? (it->text.data + it->pos) : NULL, it->text.len - it->pos };
                it->finished = true;
                if (!it->opts.keep_empty && tail.len == 0) return CS_END;
                *out_part = tail;
                return CS_OK;
            }
            size_t found_pos = 0;
            cs_status st = cs_find(it->text, it->delim, it->pos, &found_pos);
            if (st == CS_NOT_FOUND) {
                cs_view tail = { it->text.data ? (it->text.data + it->pos) : NULL, it->text.len - it->pos };
                it->finished = true;
                if (!it->opts.keep_empty && tail.len == 0) return CS_END;
                *out_part = tail;
                return CS_OK;
            }
            cs_view part = { it->text.data + it->pos, found_pos - it->pos };
            it->pos = found_pos + it->delim.len;
            it->splits_done++;
            if (!it->opts.keep_empty && part.len == 0) {
                continue;
            }
            *out_part = part;
            return CS_OK;
        }
        return CS_END;
    }

    if (it->kind == 1) {
        /* ASCII whitespace split */
        const uint8_t *p = (const uint8_t *)it->text.data;
        while (it->pos < it->text.len && cs_is_ascii_ws_byte(p[it->pos])) {
            it->pos++;
        }
        if (it->pos >= it->text.len) {
            it->finished = true;
            return CS_END;
        }
        if (it->splits_done >= it->opts.max_splits) {
            size_t end = it->text.len;
            while (end > it->pos && cs_is_ascii_ws_byte(p[end - 1U])) end--;
            *out_part = (cs_view){ it->text.data + it->pos, end - it->pos };
            it->finished = true;
            return CS_OK;
        }
        size_t start = it->pos;
        while (it->pos < it->text.len && !cs_is_ascii_ws_byte(p[it->pos])) {
            it->pos++;
        }
        it->splits_done++;
        *out_part = (cs_view){ it->text.data + start, it->pos - start };
        return CS_OK;
    }

    if (it->kind == 2) {
        /* Unicode whitespace split */
        cs_cp_iter cpi;
        cpi.v    = it->text;
        cpi.pos  = it->pos;
        cpi.mode = CS_DECODE_STRICT;
        uint32_t cp = 0;
        size_t s = 0, e = 0;
        while (cs_cp_iter_next(&cpi, &cp, &s, &e) == CS_OK) {
            if (!cs_cp_is_whitespace(cp)) {
                it->pos = s;
                goto found_u_start;
            }
        }
        it->finished = true;
        return CS_END;

found_u_start:
        if (it->splits_done >= it->opts.max_splits) {
            /* Trim trailing Unicode whitespace */
            cs_view rem = { it->text.data + it->pos, it->text.len - it->pos };
            cs_trim_end_unicode(rem, out_part);
            it->finished = true;
            return CS_OK;
        }
        size_t start = it->pos;
        cpi.pos = start;
        size_t word_end = start;
        while (cs_cp_iter_next(&cpi, &cp, &s, &e) == CS_OK) {
            if (cs_cp_is_whitespace(cp)) {
                it->pos = e;
                it->splits_done++;
                *out_part = (cs_view){ it->text.data + start, word_end - start };
                return CS_OK;
            }
            word_end = e;
        }
        it->pos = it->text.len;
        it->finished = true;
        *out_part = (cs_view){ it->text.data + start, it->text.len - start };
        return CS_OK;
    }

    if (it->kind == 3) {
        /* Lines split */
        if (it->pos >= it->text.len) {
            it->finished = true;
            return CS_END;
        }
        size_t start = it->pos;
        size_t scan  = start;
        while (scan < it->text.len) {
            size_t sep_len = cs_match_line_sep(it->text, scan, it->line_mode);
            if (sep_len > 0) {
                size_t line_end = it->keep_line_endings ? (scan + sep_len) : scan;
                *out_part = (cs_view){ it->text.data + start, line_end - start };
                it->pos = scan + sep_len;
                if (it->pos == it->text.len) {
                    it->finished = true;
                }
                return CS_OK;
            }
            scan++;
        }
        *out_part = (cs_view){ it->text.data + start, it->text.len - start };
        it->pos = it->text.len;
        it->finished = true;
        return CS_OK;
    }

    return CS_EINVAL;
}

cs_status cs_split_into(cs_view text, cs_view delim, cs_split_options opts,
                        cs_view *out_views, size_t max_views, size_t *out_count) {
    if (!out_count) return CS_EINVAL;
    cs_split_iter it;
    cs_status st = cs_split_iter_init(&it, text, delim, opts);
    if (st != CS_OK) return st;

    size_t count = 0;
    cs_view part = cs_empty_view();
    while ((st = cs_split_iter_next(&it, &part)) == CS_OK) {
        if (out_views && count < max_views) {
            out_views[count] = part;
        }
        count++;
    }
    if (st != CS_END) return st;
    *out_count = count;
    if (out_views && count > max_views) return CS_EBUFFER;
    return CS_OK;
}

cs_status cs_rsplit_into(cs_view text, cs_view delim, cs_split_options opts,
                         cs_view *out_views, size_t max_views, size_t *out_count) {
    if (!cs_view_is_valid(text) || !cs_view_is_valid(delim) || !out_count) return CS_EINVAL;
    if (delim.len == 0) return CS_EINVAL;

    if (opts.strip_outer) {
        if (cs_starts_with(text, delim)) {
            text.data += delim.len;
            text.len  -= delim.len;
        }
        if (cs_ends_with(text, delim)) {
            text.len -= delim.len;
        }
    }

    /* Count & fill from right to left, then reverse in-place so order is left-to-right */
    size_t end_pos = text.len;
    size_t splits_done = 0;
    size_t count = 0;

    for (;;) {
        if (splits_done >= opts.max_splits) {
            cs_view head = { text.data, end_pos };
            if (opts.keep_empty || head.len > 0) {
                if (out_views && count < max_views) out_views[count] = head;
                count++;
            }
            break;
        }
        size_t found_pos = 0;
        cs_status st = cs_rfind(text, delim, end_pos, &found_pos);
        if (st == CS_NOT_FOUND) {
            cs_view head = { text.data, end_pos };
            if (opts.keep_empty || head.len > 0) {
                if (out_views && count < max_views) out_views[count] = head;
                count++;
            }
            break;
        }
        cs_view part = { text.data + found_pos + delim.len, end_pos - (found_pos + delim.len) };
        end_pos = found_pos;
        splits_done++;
        if (!opts.keep_empty && part.len == 0) continue;
        if (out_views && count < max_views) out_views[count] = part;
        count++;
    }

    *out_count = count;
    if (out_views) {
        if (count > max_views) return CS_EBUFFER;
        for (size_t i = 0; i < count / 2U; i++) {
            cs_view tmp = out_views[i];
            out_views[i] = out_views[count - 1U - i];
            out_views[count - 1U - i] = tmp;
        }
    }
    return CS_OK;
}

cs_status cs_split_once(cs_view text, cs_view delim, cs_view *out_before, cs_view *out_after, bool *out_found) {
    cs_view sep = cs_empty_view();
    return cs_partition(text, delim, out_before, &sep, out_after, out_found);
}

cs_status cs_rsplit_once(cs_view text, cs_view delim, cs_view *out_before, cs_view *out_after, bool *out_found) {
    cs_view sep = cs_empty_view();
    return cs_rpartition(text, delim, out_before, &sep, out_after, out_found);
}

cs_status cs_partition(cs_view text, cs_view delim, cs_view *out_before, cs_view *out_sep, cs_view *out_after, bool *out_found) {
    if (!cs_view_is_valid(text) || !cs_view_is_valid(delim) || !out_before || !out_sep || !out_after) return CS_EINVAL;
    if (delim.len == 0) return CS_EINVAL;

    size_t pos = 0;
    cs_status st = cs_find(text, delim, 0, &pos);
    if (st == CS_OK) {
        *out_before = (cs_view){ text.data, pos };
        *out_sep    = (cs_view){ text.data + pos, delim.len };
        *out_after  = (cs_view){ text.data + pos + delim.len, text.len - (pos + delim.len) };
        if (out_found) *out_found = true;
        return CS_OK;
    }
    if (st == CS_NOT_FOUND) {
        *out_before = text;
        *out_sep    = cs_empty_view();
        *out_after  = cs_empty_view();
        if (out_found) *out_found = false;
        return CS_OK;
    }
    return st;
}

cs_status cs_rpartition(cs_view text, cs_view delim, cs_view *out_before, cs_view *out_sep, cs_view *out_after, bool *out_found) {
    if (!cs_view_is_valid(text) || !cs_view_is_valid(delim) || !out_before || !out_sep || !out_after) return CS_EINVAL;
    if (delim.len == 0) return CS_EINVAL;

    size_t pos = 0;
    cs_status st = cs_rfind(text, delim, text.len, &pos);
    if (st == CS_OK) {
        *out_before = (cs_view){ text.data, pos };
        *out_sep    = (cs_view){ text.data + pos, delim.len };
        *out_after  = (cs_view){ text.data + pos + delim.len, text.len - (pos + delim.len) };
        if (out_found) *out_found = true;
        return CS_OK;
    }
    if (st == CS_NOT_FOUND) {
        *out_before = cs_empty_view();
        *out_sep    = cs_empty_view();
        *out_after  = text;
        if (out_found) *out_found = false;
        return CS_OK;
    }
    return st;
}

cs_status cs_join_into(cs_string *dst, const cs_view *parts, size_t count, cs_view sep) {
    if (!dst || !cs_view_is_valid(sep) || (!parts && count > 0)) return CS_EINVAL;
    if (count == 0) return CS_OK;

    size_t add_len = 0;
    for (size_t i = 0; i < count; i++) {
        if (!cs_view_is_valid(parts[i])) return CS_EINVAL;
        if (cs_add_overflow_sz(add_len, parts[i].len, &add_len)) return CS_EOVERFLOW;
        if (i + 1U < count) {
            if (cs_add_overflow_sz(add_len, sep.len, &add_len)) return CS_EOVERFLOW;
        }
    }

    cs_string tmp;
    cs_status st = cs_clone(&tmp, dst);
    if (st != CS_OK) return st;
    size_t target_len = 0;
    if (cs_add_overflow_sz(tmp.len, add_len, &target_len)) { cs_destroy(&tmp); return CS_EOVERFLOW; }
    st = cs_reserve(&tmp, target_len);
    if (st != CS_OK) { cs_destroy(&tmp); return st; }

    for (size_t i = 0; i < count; i++) {
        if (parts[i].len > 0) {
            memcpy(tmp.data + tmp.len, parts[i].data, parts[i].len);
            tmp.len += parts[i].len;
        }
        if (i + 1U < count && sep.len > 0) {
            memcpy(tmp.data + tmp.len, sep.data, sep.len);
            tmp.len += sep.len;
        }
    }
    tmp.data[tmp.len] = '\0';
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
}

cs_status cs_join(cs_string *dst, const cs_view *parts, size_t count, cs_view sep) {
    if (!dst) return CS_EINVAL;
    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;
    st = cs_join_into(&tmp, parts, count, sep);
    if (st != CS_OK) { cs_destroy(&tmp); return st; }
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
}

cs_status cs_join_buffer(cs_buffer *dst, const cs_view *parts, size_t count, cs_view sep, size_t *out_required) {
    if (!dst || !cs_view_is_valid(sep) || (!parts && count > 0)) return CS_EINVAL;
    size_t total = 0;
    for (size_t i = 0; i < count; i++) {
        if (!cs_view_is_valid(parts[i])) return CS_EINVAL;
        if (cs_add_overflow_sz(total, parts[i].len, &total)) return CS_EOVERFLOW;
        if (i + 1U < count) {
            if (cs_add_overflow_sz(total, sep.len, &total)) return CS_EOVERFLOW;
        }
    }
    size_t req_with_nul = 0;
    if (cs_add_overflow_sz(total, 1U, &req_with_nul)) return CS_EOVERFLOW;
    if (out_required) *out_required = req_with_nul;
    if (total > dst->cap) return CS_EBUFFER;

    size_t pos = 0;
    for (size_t i = 0; i < count; i++) {
        if (parts[i].len > 0) {
            memcpy(dst->data + pos, parts[i].data, parts[i].len);
            pos += parts[i].len;
        }
        if (i + 1U < count && sep.len > 0) {
            memcpy(dst->data + pos, sep.data, sep.len);
            pos += sep.len;
        }
    }
    dst->len = pos;
    dst->data[dst->len] = '\0';
    return CS_OK;
}

/* ==========================================================================
 * Section 9: Text Transformations (Trim, Replace, Case, Pad, Reverse, Lines)
 * ========================================================================== */

cs_view cs_trim_start_ascii(cs_view v) {
    if (!cs_view_is_valid(v)) return cs_empty_view();
    size_t i = 0;
    while (i < v.len && cs_is_ascii_ws_byte((uint8_t)v.data[i])) i++;
    cs_view res = { v.data ? (v.data + i) : NULL, v.len - i };
    return res;
}

cs_view cs_trim_end_ascii(cs_view v) {
    if (!cs_view_is_valid(v)) return cs_empty_view();
    size_t end = v.len;
    while (end > 0 && cs_is_ascii_ws_byte((uint8_t)v.data[end - 1U])) end--;
    cs_view res = { v.data, end };
    return res;
}

cs_view cs_trim_ascii(cs_view v) {
    return cs_trim_end_ascii(cs_trim_start_ascii(v));
}

cs_status cs_trim_start_unicode(cs_view v, cs_view *out) {
    if (!cs_view_is_valid(v) || !out) return CS_EINVAL;
    cs_cp_iter it;
    cs_cp_iter_init(&it, v, CS_DECODE_STRICT);
    size_t start_off = v.len;
    bool found_non_ws = false;
    uint32_t cp = 0;
    size_t s = 0, e = 0;
    cs_status st;
    while ((st = cs_cp_iter_next(&it, &cp, &s, &e)) == CS_OK) {
        if (!found_non_ws && !cs_cp_is_whitespace(cp)) {
            start_off = s;
            found_non_ws = true;
            /* Continue scanning to validate rest of UTF-8 */
        }
    }
    if (st != CS_END) return st;
    out->data = v.data ? (v.data + start_off) : NULL;
    out->len  = v.len - start_off;
    return CS_OK;
}

cs_status cs_trim_end_unicode(cs_view v, cs_view *out) {
    if (!cs_view_is_valid(v) || !out) return CS_EINVAL;
    cs_cp_iter it;
    cs_cp_iter_init(&it, v, CS_DECODE_STRICT);
    size_t end_off = 0;
    uint32_t cp = 0;
    size_t s = 0, e = 0;
    cs_status st;
    while ((st = cs_cp_iter_next(&it, &cp, &s, &e)) == CS_OK) {
        if (!cs_cp_is_whitespace(cp)) {
            end_off = e;
        }
    }
    if (st != CS_END) return st;
    out->data = v.data;
    out->len  = end_off;
    return CS_OK;
}

cs_status cs_trim_unicode(cs_view v, cs_view *out) {
    cs_view tmp = cs_empty_view();
    cs_status st = cs_trim_start_unicode(v, &tmp);
    if (st != CS_OK) return st;
    return cs_trim_end_unicode(tmp, out);
}

static bool cs_cp_in_cutset(uint32_t cp, cs_view cutset_utf8) {
    cs_cp_iter it;
    cs_cp_iter_init(&it, cutset_utf8, CS_DECODE_STRICT);
    uint32_t c = 0;
    while (cs_cp_iter_next(&it, &c, NULL, NULL) == CS_OK) {
        if (c == cp) return true;
    }
    return false;
}

cs_status cs_trim_start_codepoints(cs_view v, cs_view cutset_utf8, cs_view *out) {
    if (!cs_view_is_valid(v) || !cs_view_is_valid(cutset_utf8) || !out) return CS_EINVAL;
    cs_status st = cs_validate_utf8(cutset_utf8, NULL);
    if (st != CS_OK) return st;
    cs_cp_iter it;
    cs_cp_iter_init(&it, v, CS_DECODE_STRICT);
    size_t start_off = v.len;
    bool found = false;
    uint32_t cp = 0;
    size_t s = 0, e = 0;
    while ((st = cs_cp_iter_next(&it, &cp, &s, &e)) == CS_OK) {
        if (!found && !cs_cp_in_cutset(cp, cutset_utf8)) {
            start_off = s;
            found = true;
        }
    }
    if (st != CS_END) return st;
    out->data = v.data ? (v.data + start_off) : NULL;
    out->len  = v.len - start_off;
    return CS_OK;
}

cs_status cs_trim_end_codepoints(cs_view v, cs_view cutset_utf8, cs_view *out) {
    if (!cs_view_is_valid(v) || !cs_view_is_valid(cutset_utf8) || !out) return CS_EINVAL;
    cs_status st = cs_validate_utf8(cutset_utf8, NULL);
    if (st != CS_OK) return st;
    cs_cp_iter it;
    cs_cp_iter_init(&it, v, CS_DECODE_STRICT);
    size_t end_off = 0;
    uint32_t cp = 0;
    size_t s = 0, e = 0;
    while ((st = cs_cp_iter_next(&it, &cp, &s, &e)) == CS_OK) {
        if (!cs_cp_in_cutset(cp, cutset_utf8)) {
            end_off = e;
        }
    }
    if (st != CS_END) return st;
    out->data = v.data;
    out->len  = end_off;
    return CS_OK;
}

cs_status cs_trim_codepoints(cs_view v, cs_view cutset_utf8, cs_view *out) {
    cs_view tmp = cs_empty_view();
    cs_status st = cs_trim_start_codepoints(v, cutset_utf8, &tmp);
    if (st != CS_OK) return st;
    return cs_trim_end_codepoints(tmp, cutset_utf8, out);
}

/* Replace & Translate */
cs_status cs_replace_first(cs_string *dst, cs_view text, cs_view needle, cs_view replacement) {
    if (!dst || !cs_view_is_valid(text) || !cs_view_is_valid(needle) || !cs_view_is_valid(replacement)) return CS_EINVAL;
    if (needle.len == 0) return CS_EINVAL;

    size_t pos = 0;
    cs_status st = cs_find(text, needle, 0, &pos);
    if (st == CS_NOT_FOUND) {
        return cs_assign(dst, text);
    }
    if (st != CS_OK) return st;

    cs_string tmp;
    st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;
    cs_view parts[3] = {
        { text.data, pos },
        replacement,
        { text.data + pos + needle.len, text.len - (pos + needle.len) }
    };
    st = cs_append_many(&tmp, parts, 3U);
    if (st != CS_OK) { cs_destroy(&tmp); return st; }
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
}

cs_status cs_replace_all(cs_string *dst, cs_view text, cs_view needle, cs_view replacement) {
    if (!dst || !cs_view_is_valid(text) || !cs_view_is_valid(needle) || !cs_view_is_valid(replacement)) return CS_EINVAL;
    if (needle.len == 0) return CS_EINVAL;

    cs_searcher sr;
    cs_status st = cs_searcher_init(&sr, needle, CS_MATCH_EXACT, NULL);
    if (st != CS_OK) return st;

    cs_string tmp;
    st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) { cs_searcher_destroy(&sr); return st; }

    size_t pos = 0;
    while (pos <= text.len) {
        size_t ms = 0, me = 0;
        st = cs_searcher_find(&sr, text, pos, &ms, &me);
        if (st == CS_NOT_FOUND) {
            cs_view tail = { text.data ? (text.data + pos) : NULL, text.len - pos };
            st = cs_append(&tmp, tail);
            break;
        }
        if (st != CS_OK) break;
        cs_view seg = { text.data + pos, ms - pos };
        st = cs_append(&tmp, seg);
        if (st != CS_OK) break;
        st = cs_append(&tmp, replacement);
        if (st != CS_OK) break;
        pos = me;
    }
    cs_searcher_destroy(&sr);
    if (st != CS_OK) { cs_destroy(&tmp); return st; }
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
}

cs_status cs_replace_all_buffer(cs_buffer *dst, cs_view text, cs_view needle, cs_view replacement, size_t *out_required) {
    if (!dst || !cs_view_is_valid(text) || !cs_view_is_valid(needle) || !cs_view_is_valid(replacement)) return CS_EINVAL;
    if (needle.len == 0) return CS_EINVAL;

    size_t match_count = 0;
    cs_status st = cs_count(text, needle, CS_OVERLAP_DISALLOW, &match_count);
    if (st != CS_OK) return st;

    size_t rem_bytes = text.len - (match_count * needle.len);
    size_t rep_bytes = 0;
    if (cs_mul_overflow_sz(match_count, replacement.len, &rep_bytes)) return CS_EOVERFLOW;
    size_t total_len = 0;
    if (cs_add_overflow_sz(rem_bytes, rep_bytes, &total_len)) return CS_EOVERFLOW;
    size_t req_with_nul = 0;
    if (cs_add_overflow_sz(total_len, 1U, &req_with_nul)) return CS_EOVERFLOW;
    if (out_required) *out_required = req_with_nul;
    if (total_len > dst->cap) return CS_EBUFFER;

    cs_searcher sr;
    st = cs_searcher_init(&sr, needle, CS_MATCH_EXACT, NULL);
    if (st != CS_OK) return st;

    size_t pos = 0, out_pos = 0;
    while (pos <= text.len) {
        size_t ms = 0, me = 0;
        st = cs_searcher_find(&sr, text, pos, &ms, &me);
        if (st == CS_NOT_FOUND) {
            size_t tlen = text.len - pos;
            if (tlen > 0) memcpy(dst->data + out_pos, text.data + pos, tlen);
            out_pos += tlen;
            break;
        }
        size_t slen = ms - pos;
        if (slen > 0) memcpy(dst->data + out_pos, text.data + pos, slen);
        out_pos += slen;
        if (replacement.len > 0) memcpy(dst->data + out_pos, replacement.data, replacement.len);
        out_pos += replacement.len;
        pos = me;
    }
    cs_searcher_destroy(&sr);
    dst->len = out_pos;
    dst->data[dst->len] = '\0';
    return CS_OK;
}

cs_status cs_remove_all(cs_string *dst, cs_view text, cs_view needle) {
    return cs_replace_all(dst, text, needle, cs_empty_view());
}

cs_status cs_translate_codepoints(cs_string *dst, cs_view text, const cs_cp_mapping *map, size_t map_len) {
    if (!dst || !cs_view_is_valid(text) || (!map && map_len > 0)) return CS_EINVAL;
    for (size_t i = 0; i < map_len; i++) {
        if (!cs_view_is_valid(map[i].to_utf8)) return CS_EINVAL;
        cs_status st = cs_validate_utf8(map[i].to_utf8, NULL);
        if (st != CS_OK) return st;
    }

    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;

    cs_cp_iter it;
    cs_cp_iter_init(&it, text, CS_DECODE_STRICT);
    uint32_t cp = 0;
    size_t s = 0, e = 0;
    while ((st = cs_cp_iter_next(&it, &cp, &s, &e)) == CS_OK) {
        bool mapped = false;
        for (size_t i = 0; i < map_len; i++) {
            if (map[i].from_cp == cp) {
                st = cs_append(&tmp, map[i].to_utf8);
                mapped = true;
                break;
            }
        }
        if (!mapped) {
            cs_view orig = { text.data + s, e - s };
            st = cs_append(&tmp, orig);
        }
        if (st != CS_OK) { cs_destroy(&tmp); return st; }
    }
    if (st != CS_END) { cs_destroy(&tmp); return st; }
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
}

/* ==========================================================================
 * ASCII & Full Context-Sensitive / Locale-Sensitive Unicode Case Conversions
 * ========================================================================== */

cs_status cs_to_lower_ascii(cs_string *dst, cs_view v) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
    cs_status st = cs_assign(dst, v);
    if (st != CS_OK) return st;
    cs_ascii_case_convert_scalar((uint8_t *)dst->data, (const uint8_t *)dst->data, dst->len, false);
    return CS_OK;
}

cs_status cs_to_upper_ascii(cs_string *dst, cs_view v) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
    cs_status st = cs_assign(dst, v);
    if (st != CS_OK) return st;
    cs_ascii_case_convert_scalar((uint8_t *)dst->data, (const uint8_t *)dst->data, dst->len, true);
    return CS_OK;
}

cs_status cs_to_lower_ascii_buffer(cs_buffer *dst, cs_view v, size_t *out_required) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
    dst->len = 0;
    cs_status st = cs_buffer_append(dst, v, out_required);
    if (st != CS_OK) return st;
    cs_ascii_case_convert_scalar((uint8_t *)dst->data, (const uint8_t *)dst->data, dst->len, false);
    return CS_OK;
}

cs_status cs_to_upper_ascii_buffer(cs_buffer *dst, cs_view v, size_t *out_required) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
    dst->len = 0;
    cs_status st = cs_buffer_append(dst, v, out_required);
    if (st != CS_OK) return st;
    cs_ascii_case_convert_scalar((uint8_t *)dst->data, (const uint8_t *)dst->data, dst->len, true);
    return CS_OK;
}

#if CS_ENABLE_UNICODE
/* Context helpers for SpecialCasing (UAX #44 / Unicode Chapter 3.13):
 * - Final_Sigma: preceded by Cased (ignoring Case_Ignorable), not followed by Cased (ignoring Case_Ignorable)
 * - After_Soft_Dotted (lt): preceded by Soft_Dotted with no intervening CCC == 0 or CCC == 230
 * - More_Above (lt): followed by a character of CCC == 230 with no intervening CCC == 0 or CCC == 230
 * - After_I (tr/az): preceded by U+0049 with no intervening CCC == 0 or CCC == 230
 * - Before_Dot (tr/az): followed by U+0307 with no intervening CCC == 0 or CCC == 230 */
static bool cs_ctx_followed_by_cased(cs_view v, size_t after_off) {
    cs_cp_iter it;
    cs_cp_iter_init(&it, v, CS_DECODE_STRICT);
    it.pos = after_off;
    uint32_t cp = 0;
    while (cs_cp_iter_next(&it, &cp, NULL, NULL) == CS_OK) {
        uint8_t flags = cs_uprop_lookup(cp)->flags;
        if (flags & CS_UF_CASED) return true;
        if (flags & CS_UF_CI)    continue;
        return false;
    }
    return false;
}

static bool cs_ctx_more_above(cs_view v, size_t after_off) {
    cs_cp_iter it;
    cs_cp_iter_init(&it, v, CS_DECODE_STRICT);
    it.pos = after_off;
    uint32_t cp = 0;
    while (cs_cp_iter_next(&it, &cp, NULL, NULL) == CS_OK) {
        uint8_t ccc = cs_uprop_lookup(cp)->ccc;
        if (ccc == 230U) return true;
        if (ccc == 0U)   return false;
    }
    return false;
}

static bool cs_ctx_before_dot(cs_view v, size_t after_off) {
    cs_cp_iter it;
    cs_cp_iter_init(&it, v, CS_DECODE_STRICT);
    it.pos = after_off;
    uint32_t cp = 0;
    while (cs_cp_iter_next(&it, &cp, NULL, NULL) == CS_OK) {
        if (cp == 0x0307U) return true;
        uint8_t ccc = cs_uprop_lookup(cp)->ccc;
        if (ccc == 0U || ccc == 230U) return false;
    }
    return false;
}

/* Mode: 0 = lower, 1 = upper, 2 = title (single char), 3 = casefold */
static size_t cs_map_single_cp_with_ctx(uint32_t cp, int mode, cs_case_locale locale,
                                        bool preceded_by_cased, bool after_I, bool after_soft_dotted,
                                        cs_view full_text, size_t next_off, uint32_t out_cps[3]) {
    if (mode == 3) {
        return cs_fold_cp_full(cp, locale, out_cps);
    }

    /* Locale-specific rules: Turkish / Azeri */
    if (locale == CS_LOCALE_TR || locale == CS_LOCALE_AZ) {
        if (mode == 0) {
            if (cp == 0x0130U) { out_cps[0] = 0x0069U; return 1U; }
            if (cp == 0x0307U && after_I) { return 0U; /* delete combining dot after I */ }
            if (cp == 0x0049U) {
                out_cps[0] = cs_ctx_before_dot(full_text, next_off) ? 0x0069U : 0x0131U;
                return 1U;
            }
        } else {
            if (cp == 0x0069U) { out_cps[0] = 0x0130U; return 1U; }
        }
    }

    /* Locale-specific rules: Lithuanian */
    if (locale == CS_LOCALE_LT) {
        if (mode == 0) {
            if (cp == 0x00CCU) { out_cps[0] = 0x0069U; out_cps[1] = 0x0307U; out_cps[2] = 0x0300U; return 3U; }
            if (cp == 0x00CDU) { out_cps[0] = 0x0069U; out_cps[1] = 0x0307U; out_cps[2] = 0x0301U; return 3U; }
            if (cp == 0x0128U) { out_cps[0] = 0x0069U; out_cps[1] = 0x0307U; out_cps[2] = 0x0303U; return 3U; }
            if ((cp == 0x0049U || cp == 0x004AU || cp == 0x012EU) && cs_ctx_more_above(full_text, next_off)) {
                out_cps[0] = (cp == 0x0049U) ? 0x0069U : ((cp == 0x004AU) ? 0x006AU : 0x012FU);
                out_cps[1] = 0x0307U;
                return 2U;
            }
        } else {
            if (cp == 0x0307U && after_soft_dotted) {
                return 0U; /* delete combining dot above after soft-dotted */
            }
        }
    }

    /* Context-sensitive Final_Sigma (Greek Capital Sigma U+03A3 in lowercase) */
    if (mode == 0 && cp == 0x03A3U) {
        if (preceded_by_cased && !cs_ctx_followed_by_cased(full_text, next_off)) {
            out_cps[0] = 0x03C2U;
            return 1U;
        }
        out_cps[0] = 0x03C3U;
        return 1U;
    }

    const cs_ucase_entry *ce = cs_ucase_find(cp);
    if (!ce) {
        out_cps[0] = cp;
        return 1U;
    }
    int32_t target = (mode == 0) ? ce->lc : ((mode == 1) ? ce->uc : ce->tc);
    return cs_ucase_decode_target(cp, target, out_cps);
}

static cs_status cs_case_transform_impl(cs_builder *b, cs_view v, int mode, cs_case_locale locale) {
    cs_cp_iter it;
    cs_cp_iter_init(&it, v, CS_DECODE_STRICT);

    bool preceded_by_cased = false;
    bool after_I           = false;
    bool after_soft_dotted = false;

    uint32_t cp = 0;
    size_t s = 0, e = 0;
    cs_status st;
    while ((st = cs_cp_iter_next(&it, &cp, &s, &e)) == CS_OK) {
        uint32_t mapped[3];
        size_t mlen = cs_map_single_cp_with_ctx(cp, mode, locale,
                                                preceded_by_cased, after_I, after_soft_dotted,
                                                v, e, mapped);
        for (size_t k = 0; k < mlen; k++) {
            cs_builder_append_codepoint(b, mapped[k]);
        }

        /* Update context state */
        const cs_uprop_record *rec = cs_uprop_lookup(cp);
        uint8_t flags = rec->flags;
        uint8_t ccc   = rec->ccc;

        if (flags & CS_UF_CASED) preceded_by_cased = true;
        else if (!(flags & CS_UF_CI)) preceded_by_cased = false;

        if (cp == 0x0049U) after_I = true;
        else if (ccc == 0U || ccc == 230U) after_I = false;

        if (flags & CS_UF_SD) after_soft_dotted = true;
        else if (ccc == 0U || ccc == 230U) after_soft_dotted = false;
    }
    if (st != CS_END) return st;
    return b->sticky_err;
}
#endif /* CS_ENABLE_UNICODE */

cs_status cs_to_lower_unicode(cs_string *dst, cs_view v, cs_case_locale locale) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
#if !CS_ENABLE_UNICODE
    (void)locale;
    return CS_EUNSUPPORTED;
#else
    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;
    cs_builder b;
    cs_builder_init_string(&b, &tmp);
    st = cs_case_transform_impl(&b, v, 0, locale);
    if (st != CS_OK) { cs_destroy(&tmp); return st; }
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
#endif
}

cs_status cs_to_upper_unicode(cs_string *dst, cs_view v, cs_case_locale locale) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
#if !CS_ENABLE_UNICODE
    (void)locale;
    return CS_EUNSUPPORTED;
#else
    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;
    cs_builder b;
    cs_builder_init_string(&b, &tmp);
    st = cs_case_transform_impl(&b, v, 1, locale);
    if (st != CS_OK) { cs_destroy(&tmp); return st; }
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
#endif
}

cs_status cs_to_casefold_unicode(cs_string *dst, cs_view v, cs_case_locale locale) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
#if !CS_ENABLE_UNICODE
    (void)locale;
    return CS_EUNSUPPORTED;
#else
    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;
    cs_builder b;
    cs_builder_init_string(&b, &tmp);
    st = cs_case_transform_impl(&b, v, 3, locale);
    if (st != CS_OK) { cs_destroy(&tmp); return st; }
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
#endif
}

cs_status cs_to_lower_unicode_buffer(cs_buffer *dst, cs_view v, cs_case_locale locale, size_t *out_required) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
#if !CS_ENABLE_UNICODE
    (void)locale; (void)out_required;
    return CS_EUNSUPPORTED;
#else
    cs_buffer_clear(dst);
    cs_builder b;
    cs_builder_init_buffer(&b, dst);
    cs_status st = cs_case_transform_impl(&b, v, 0, locale);
    if (st != CS_OK && st != CS_EBUFFER) return st;
    return cs_builder_finish(&b, NULL, out_required);
#endif
}

cs_status cs_to_upper_unicode_buffer(cs_buffer *dst, cs_view v, cs_case_locale locale, size_t *out_required) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
#if !CS_ENABLE_UNICODE
    (void)locale; (void)out_required;
    return CS_EUNSUPPORTED;
#else
    cs_buffer_clear(dst);
    cs_builder b;
    cs_builder_init_buffer(&b, dst);
    cs_status st = cs_case_transform_impl(&b, v, 1, locale);
    if (st != CS_OK && st != CS_EBUFFER) return st;
    return cs_builder_finish(&b, NULL, out_required);
#endif
}

cs_status cs_to_casefold_unicode_buffer(cs_buffer *dst, cs_view v, cs_case_locale locale, size_t *out_required) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
#if !CS_ENABLE_UNICODE
    (void)locale; (void)out_required;
    return CS_EUNSUPPORTED;
#else
    cs_buffer_clear(dst);
    cs_builder b;
    cs_builder_init_buffer(&b, dst);
    cs_status st = cs_case_transform_impl(&b, v, 3, locale);
    if (st != CS_OK && st != CS_EBUFFER) return st;
    return cs_builder_finish(&b, NULL, out_required);
#endif
}

/* Titlecase along default UAX #29 word boundaries (Section 9):
 * For each word segment, the first Cased code point is converted with Titlecase_Mapping,
 * and all subsequent code points in that word segment are converted with Lowercase_Mapping. */
cs_status cs_to_title_unicode(cs_string *dst, cs_view v, cs_case_locale locale) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
#if !CS_ENABLE_UNICODE
    (void)locale;
    return CS_EUNSUPPORTED;
#else
    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;
    cs_builder b;
    cs_builder_init_string(&b, &tmp);

    cs_word_iter wit;
    cs_word_iter_init(&wit, v, CS_DECODE_STRICT);
    cs_view wseg = cs_empty_view();
    size_t ws = 0, we = 0;
    bool is_word = false;

    while ((st = cs_word_iter_next(&wit, &wseg, &ws, &we, &is_word)) == CS_OK) {
        cs_cp_iter cpi;
        cs_cp_iter_init(&cpi, wseg, CS_DECODE_STRICT);
        bool first_cased_done = false;
        bool preceded_by_cased = false;
        bool after_I = false;
        bool after_soft_dotted = false;

        uint32_t cp = 0;
        size_t cs_off = 0, ce_off = 0;
        while (cs_cp_iter_next(&cpi, &cp, &cs_off, &ce_off) == CS_OK) {
            const cs_uprop_record *rec = cs_uprop_lookup(cp);
            bool is_cased = (rec->flags & CS_UF_CASED) != 0;
            int mode = 0; /* lower */
            if (!first_cased_done && is_cased) {
                mode = 2; /* titlecase */
                first_cased_done = true;
            }
            uint32_t mapped[3];
            size_t mlen = cs_map_single_cp_with_ctx(cp, mode, locale,
                                                    preceded_by_cased, after_I, after_soft_dotted,
                                                    wseg, ce_off, mapped);
            for (size_t k = 0; k < mlen; k++) {
                cs_builder_append_codepoint(&b, mapped[k]);
            }

            uint8_t flags = rec->flags;
            uint8_t ccc   = rec->ccc;
            if (flags & CS_UF_CASED) preceded_by_cased = true;
            else if (!(flags & CS_UF_CI)) preceded_by_cased = false;
            if (cp == 0x0049U) after_I = true;
            else if (ccc == 0U || ccc == 230U) after_I = false;
            if (flags & CS_UF_SD) after_soft_dotted = true;
            else if (ccc == 0U || ccc == 230U) after_soft_dotted = false;
        }
    }
    if (st != CS_END) { cs_destroy(&tmp); return st; }
    if (b.sticky_err != CS_OK) { cs_destroy(&tmp); return b.sticky_err; }
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
#endif
}

/* Capitalize: first Cased code point in the entire string is converted to Titlecase;
 * remaining code points are either converted to Lowercase (`lower_rest == true`) or left unchanged. */
cs_status cs_capitalize_unicode(cs_string *dst, cs_view v, cs_case_locale locale, bool lower_rest) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
#if !CS_ENABLE_UNICODE
    (void)locale; (void)lower_rest;
    return CS_EUNSUPPORTED;
#else
    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;
    cs_builder b;
    cs_builder_init_string(&b, &tmp);

    cs_cp_iter cpi;
    cs_cp_iter_init(&cpi, v, CS_DECODE_STRICT);
    bool first_cased_done = false;
    bool preceded_by_cased = false;
    bool after_I = false;
    bool after_soft_dotted = false;

    uint32_t cp = 0;
    size_t cs_off = 0, ce_off = 0;
    while ((st = cs_cp_iter_next(&cpi, &cp, &cs_off, &ce_off)) == CS_OK) {
        const cs_uprop_record *rec = cs_uprop_lookup(cp);
        bool is_cased = (rec->flags & CS_UF_CASED) != 0;

        if (!first_cased_done && is_cased) {
            first_cased_done = true;
            uint32_t mapped[3];
            size_t mlen = cs_map_single_cp_with_ctx(cp, 2, locale,
                                                    preceded_by_cased, after_I, after_soft_dotted,
                                                    v, ce_off, mapped);
            for (size_t k = 0; k < mlen; k++) cs_builder_append_codepoint(&b, mapped[k]);
        } else if (lower_rest) {
            uint32_t mapped[3];
            size_t mlen = cs_map_single_cp_with_ctx(cp, 0, locale,
                                                    preceded_by_cased, after_I, after_soft_dotted,
                                                    v, ce_off, mapped);
            for (size_t k = 0; k < mlen; k++) cs_builder_append_codepoint(&b, mapped[k]);
        } else {
            cs_view orig = { v.data + cs_off, ce_off - cs_off };
            cs_builder_append(&b, orig);
        }

        uint8_t flags = rec->flags;
        uint8_t ccc   = rec->ccc;
        if (flags & CS_UF_CASED) preceded_by_cased = true;
        else if (!(flags & CS_UF_CI)) preceded_by_cased = false;
        if (cp == 0x0049U) after_I = true;
        else if (ccc == 0U || ccc == 230U) after_I = false;
        if (flags & CS_UF_SD) after_soft_dotted = true;
        else if (ccc == 0U || ccc == 230U) after_soft_dotted = false;
    }
    if (st != CS_END) { cs_destroy(&tmp); return st; }
    if (b.sticky_err != CS_OK) { cs_destroy(&tmp); return b.sticky_err; }
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
#endif
}

/* ==========================================================================
 * Padding, Reversal, Tabs, Indentation & Newlines (Section 9)
 * ========================================================================== */

static cs_status cs_pad_common_impl(cs_string *dst, cs_view v, size_t left_count, size_t right_count, cs_view fill_unit) {
    size_t left_bytes = 0, right_bytes = 0;
    if (cs_mul_overflow_sz(left_count, fill_unit.len, &left_bytes)) return CS_EOVERFLOW;
    if (cs_mul_overflow_sz(right_count, fill_unit.len, &right_bytes)) return CS_EOVERFLOW;
    if (left_bytes + right_bytes > CS_MAX_REPEAT_BYTES) return CS_ELIMIT;

    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;

    size_t total = 0;
    if (cs_add_overflow_sz(left_bytes, v.len, &total) ||
        cs_add_overflow_sz(total, right_bytes, &total)) {
        cs_destroy(&tmp);
        return CS_EOVERFLOW;
    }
    st = cs_reserve(&tmp, total);
    if (st != CS_OK) { cs_destroy(&tmp); return st; }

    for (size_t i = 0; i < left_count; i++) cs_append(&tmp, fill_unit);
    cs_append(&tmp, v);
    for (size_t i = 0; i < right_count; i++) cs_append(&tmp, fill_unit);

    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
}

cs_status cs_pad_start_bytes(cs_string *dst, cs_view v, size_t target_len, uint8_t fill_byte) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
    if (v.len >= target_len) return cs_assign(dst, v);
    char c = (char)fill_byte;
    cs_view fu = { &c, 1U };
    return cs_pad_common_impl(dst, v, target_len - v.len, 0, fu);
}

cs_status cs_pad_end_bytes(cs_string *dst, cs_view v, size_t target_len, uint8_t fill_byte) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
    if (v.len >= target_len) return cs_assign(dst, v);
    char c = (char)fill_byte;
    cs_view fu = { &c, 1U };
    return cs_pad_common_impl(dst, v, 0, target_len - v.len, fu);
}

cs_status cs_center_bytes(cs_string *dst, cs_view v, size_t target_len, uint8_t fill_byte) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
    if (v.len >= target_len) return cs_assign(dst, v);
    size_t pad = target_len - v.len;
    char c = (char)fill_byte;
    cs_view fu = { &c, 1U };
    return cs_pad_common_impl(dst, v, pad / 2U, pad - (pad / 2U), fu);
}

cs_status cs_pad_start_codepoints(cs_string *dst, cs_view v, size_t target_cps, uint32_t fill_cp) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
    char fbuf[4];
    size_t flen = 0;
    cs_status st = cs_encode_utf8_codepoint(fill_cp, fbuf, &flen);
    if (st != CS_OK) return st;
    size_t cur_cps = 0;
    st = cs_len_codepoints(v, &cur_cps);
    if (st != CS_OK) return st;
    if (cur_cps >= target_cps) return cs_assign(dst, v);
    cs_view fu = { fbuf, flen };
    return cs_pad_common_impl(dst, v, target_cps - cur_cps, 0, fu);
}

cs_status cs_pad_end_codepoints(cs_string *dst, cs_view v, size_t target_cps, uint32_t fill_cp) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
    char fbuf[4];
    size_t flen = 0;
    cs_status st = cs_encode_utf8_codepoint(fill_cp, fbuf, &flen);
    if (st != CS_OK) return st;
    size_t cur_cps = 0;
    st = cs_len_codepoints(v, &cur_cps);
    if (st != CS_OK) return st;
    if (cur_cps >= target_cps) return cs_assign(dst, v);
    cs_view fu = { fbuf, flen };
    return cs_pad_common_impl(dst, v, 0, target_cps - cur_cps, fu);
}

cs_status cs_center_codepoints(cs_string *dst, cs_view v, size_t target_cps, uint32_t fill_cp) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
    char fbuf[4];
    size_t flen = 0;
    cs_status st = cs_encode_utf8_codepoint(fill_cp, fbuf, &flen);
    if (st != CS_OK) return st;
    size_t cur_cps = 0;
    st = cs_len_codepoints(v, &cur_cps);
    if (st != CS_OK) return st;
    if (cur_cps >= target_cps) return cs_assign(dst, v);
    size_t pad = target_cps - cur_cps;
    cs_view fu = { fbuf, flen };
    return cs_pad_common_impl(dst, v, pad / 2U, pad - (pad / 2U), fu);
}

static cs_status cs_validate_single_grapheme(cs_view fill_grapheme) {
    if (!cs_view_is_valid(fill_grapheme) || fill_grapheme.len == 0) return CS_EINVAL;
    size_t gcnt = 0;
    cs_status st = cs_len_graphemes(fill_grapheme, &gcnt);
    if (st != CS_OK) return st;
    return (gcnt == 1U) ? CS_OK : CS_EINVAL;
}

cs_status cs_pad_start_graphemes(cs_string *dst, cs_view v, size_t target_grs, cs_view fill_grapheme) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
    cs_status st = cs_validate_single_grapheme(fill_grapheme);
    if (st != CS_OK) return st;
    size_t cur_grs = 0;
    st = cs_len_graphemes(v, &cur_grs);
    if (st != CS_OK) return st;
    if (cur_grs >= target_grs) return cs_assign(dst, v);
    return cs_pad_common_impl(dst, v, target_grs - cur_grs, 0, fill_grapheme);
}

cs_status cs_pad_end_graphemes(cs_string *dst, cs_view v, size_t target_grs, cs_view fill_grapheme) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
    cs_status st = cs_validate_single_grapheme(fill_grapheme);
    if (st != CS_OK) return st;
    size_t cur_grs = 0;
    st = cs_len_graphemes(v, &cur_grs);
    if (st != CS_OK) return st;
    if (cur_grs >= target_grs) return cs_assign(dst, v);
    return cs_pad_common_impl(dst, v, 0, target_grs - cur_grs, fill_grapheme);
}

cs_status cs_center_graphemes(cs_string *dst, cs_view v, size_t target_grs, cs_view fill_grapheme) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
    cs_status st = cs_validate_single_grapheme(fill_grapheme);
    if (st != CS_OK) return st;
    size_t cur_grs = 0;
    st = cs_len_graphemes(v, &cur_grs);
    if (st != CS_OK) return st;
    if (cur_grs >= target_grs) return cs_assign(dst, v);
    size_t pad = target_grs - cur_grs;
    return cs_pad_common_impl(dst, v, pad / 2U, pad - (pad / 2U), fill_grapheme);
}

/* Reversal (Section 9) */
cs_status cs_reverse_bytes(cs_string *dst, cs_view v) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
    cs_status st = cs_assign(dst, v);
    if (st != CS_OK) return st;
    for (size_t i = 0; i < dst->len / 2U; i++) {
        char tmp = dst->data[i];
        dst->data[i] = dst->data[dst->len - 1U - i];
        dst->data[dst->len - 1U - i] = tmp;
    }
    return CS_OK;
}

cs_status cs_reverse_codepoints(cs_string *dst, cs_view v) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
    cs_status st = cs_validate_utf8(v, NULL);
    if (st != CS_OK) return st;

    cs_string tmp;
    st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;
    st = cs_reserve(&tmp, v.len);
    if (st != CS_OK) { cs_destroy(&tmp); return st; }

    cs_cp_iter it;
    cs_cp_iter_init(&it, v, CS_DECODE_STRICT);
    it.pos = v.len;
    uint32_t cp = 0;
    size_t s = 0, e = 0;
    while (cs_cp_iter_prev(&it, &cp, &s, &e) == CS_OK) {
        memcpy(tmp.data + tmp.len, v.data + s, e - s);
        tmp.len += (e - s);
    }
    tmp.data[tmp.len] = '\0';
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
}

cs_status cs_reverse_graphemes(cs_string *dst, cs_view v) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
    cs_status st = cs_validate_utf8(v, NULL);
    if (st != CS_OK) return st;

    cs_index idx;
    st = cs_index_build(&idx, NULL, v, &dst->alloc);
    if (st != CS_OK) return st;

    cs_string tmp;
    st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) { cs_index_destroy(&idx); return st; }
    st = cs_reserve(&tmp, v.len);
    if (st != CS_OK) { cs_index_destroy(&idx); cs_destroy(&tmp); return st; }

    for (size_t i = idx.gr_count; i > 0; i--) {
        cs_view span = cs_empty_view();
        cs_index_grapheme_span(&idx, i - 1U, &span);
        if (span.len > 0) {
            memcpy(tmp.data + tmp.len, span.data, span.len);
            tmp.len += span.len;
        }
    }
    tmp.data[tmp.len] = '\0';
    cs_index_destroy(&idx);
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
}

/* Expand tabs, indent, dedent, normalize_newlines */
cs_status cs_expand_tabs(cs_string *dst, cs_view v, size_t tab_stop, cs_column_unit unit) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
    if (unit != CS_COL_BYTES) {
        cs_status st = cs_validate_utf8(v, NULL);
        if (st != CS_OK) return st;
    }

    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;

    size_t col = 0;
    if (unit == CS_COL_GRAPHEMES) {
        cs_grapheme_iter it;
        cs_grapheme_iter_init(&it, v, CS_DECODE_STRICT);
        cs_view gr = cs_empty_view();
        while ((st = cs_grapheme_iter_next(&it, &gr, NULL, NULL)) == CS_OK) {
            if (gr.len == 1U && gr.data[0] == '\t') {
                size_t spaces = (tab_stop == 0) ? 0 : (tab_stop - (col % tab_stop));
                for (size_t k = 0; k < spaces; k++) {
                    st = cs_append_byte(&tmp, ' ');
                    if (st != CS_OK) { cs_destroy(&tmp); return st; }
                }
                col += spaces;
            } else {
                st = cs_append(&tmp, gr);
                if (st != CS_OK) { cs_destroy(&tmp); return st; }
                if ((gr.len == 1U && (gr.data[0] == '\n' || gr.data[0] == '\r')) ||
                    (gr.len == 2U && gr.data[0] == '\r' && gr.data[1] == '\n')) {
                    col = 0;
                } else {
                    col++;
                }
            }
        }
    } else if (unit == CS_COL_CODEPOINTS) {
        cs_cp_iter it;
        cs_cp_iter_init(&it, v, CS_DECODE_STRICT);
        uint32_t cp = 0;
        size_t s = 0, e = 0;
        while ((st = cs_cp_iter_next(&it, &cp, &s, &e)) == CS_OK) {
            if (cp == '\t') {
                size_t spaces = (tab_stop == 0) ? 0 : (tab_stop - (col % tab_stop));
                for (size_t k = 0; k < spaces; k++) {
                    st = cs_append_byte(&tmp, ' ');
                    if (st != CS_OK) { cs_destroy(&tmp); return st; }
                }
                col += spaces;
            } else {
                cs_view seg = { v.data + s, e - s };
                st = cs_append(&tmp, seg);
                if (st != CS_OK) { cs_destroy(&tmp); return st; }
                if (cp == '\n' || cp == '\r') col = 0;
                else col++;
            }
        }
    } else {
        for (size_t i = 0; i < v.len; i++) {
            uint8_t b = (uint8_t)v.data[i];
            if (b == '\t') {
                size_t spaces = (tab_stop == 0) ? 0 : (tab_stop - (col % tab_stop));
                for (size_t k = 0; k < spaces; k++) {
                    st = cs_append_byte(&tmp, ' ');
                    if (st != CS_OK) { cs_destroy(&tmp); return st; }
                }
                col += spaces;
            } else {
                st = cs_append_byte(&tmp, b);
                if (st != CS_OK) { cs_destroy(&tmp); return st; }
                if (b == '\n' || b == '\r') col = 0;
                else col++;
            }
        }
    }

    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
}

cs_status cs_indent(cs_string *dst, cs_view v, cs_view prefix) {
    if (!dst || !cs_view_is_valid(v) || !cs_view_is_valid(prefix)) return CS_EINVAL;
    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;

    cs_split_iter it;
    cs_split_lines_iter_init(&it, v, CS_LINES_ASCII, true);
    cs_view line = cs_empty_view();
    while (cs_split_iter_next(&it, &line) == CS_OK) {
        /* Indent non-blank lines */
        cs_view trimmed = cs_trim_ascii(line);
        if (trimmed.len > 0) {
            st = cs_append(&tmp, prefix);
            if (st != CS_OK) { cs_destroy(&tmp); return st; }
        }
        st = cs_append(&tmp, line);
        if (st != CS_OK) { cs_destroy(&tmp); return st; }
    }
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
}

cs_status cs_dedent(cs_string *dst, cs_view v) {
    if (!dst || !cs_view_is_valid(v)) return CS_EINVAL;
    if (v.len == 0) {
        cs_clear(dst);
        return CS_OK;
    }

    /* Find longest common leading whitespace across all non-blank lines */
    cs_view common_ws = cs_empty_view();
    bool has_common = false;

    cs_split_iter it;
    cs_split_lines_iter_init(&it, v, CS_LINES_ASCII, true);
    cs_view line = cs_empty_view();
    while (cs_split_iter_next(&it, &line) == CS_OK) {
        size_t ws_len = 0;
        while (ws_len < line.len && (line.data[ws_len] == ' ' || line.data[ws_len] == '\t')) {
            ws_len++;
        }
        /* Check if remainder of line is only CR/LF (blank line) */
        size_t rem = ws_len;
        while (rem < line.len && (line.data[rem] == '\r' || line.data[rem] == '\n')) rem++;
        if (rem == line.len) continue; /* Ignore blank line */

        cs_view lws = { line.data, ws_len };
        if (!has_common) {
            common_ws = lws;
            has_common = true;
        } else {
            common_ws = cs_common_prefix_bytes(common_ws, lws);
        }
    }

    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;

    cs_split_lines_iter_init(&it, v, CS_LINES_ASCII, true);
    while (cs_split_iter_next(&it, &line) == CS_OK) {
        size_t ws_len = 0;
        while (ws_len < line.len && (line.data[ws_len] == ' ' || line.data[ws_len] == '\t')) ws_len++;
        size_t rem = ws_len;
        while (rem < line.len && (line.data[rem] == '\r' || line.data[rem] == '\n')) rem++;
        if (rem == line.len) {
            /* Strip whitespace on blank line up to min(ws_len, common_ws.len) */
            size_t strip = (has_common && ws_len > common_ws.len) ? common_ws.len : ws_len;
            cs_view out_l = { line.data + strip, line.len - strip };
            st = cs_append(&tmp, out_l);
        } else {
            size_t strip = has_common ? common_ws.len : 0;
            cs_view out_l = { line.data + strip, line.len - strip };
            st = cs_append(&tmp, out_l);
        }
        if (st != CS_OK) { cs_destroy(&tmp); return st; }
    }
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
}

cs_status cs_normalize_newlines(cs_string *dst, cs_view v, cs_line_mode mode, cs_view target_newline) {
    if (!dst || !cs_view_is_valid(v) || !cs_view_is_valid(target_newline)) return CS_EINVAL;
    if (mode == CS_LINES_UNICODE) {
        cs_status st = cs_validate_utf8(v, NULL);
        if (st != CS_OK) return st;
    }
    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;

    size_t i = 0;
    size_t seg_start = 0;
    while (i < v.len) {
        size_t sep_len = cs_match_line_sep(v, i, mode);
        if (sep_len > 0) {
            if (i > seg_start) {
                cs_view seg = { v.data + seg_start, i - seg_start };
                st = cs_append(&tmp, seg);
                if (st != CS_OK) { cs_destroy(&tmp); return st; }
            }
            st = cs_append(&tmp, target_newline);
            if (st != CS_OK) { cs_destroy(&tmp); return st; }
            i += sep_len;
            seg_start = i;
        } else {
            i++;
        }
    }
    if (seg_start < v.len) {
        cs_view tail = { v.data + seg_start, v.len - seg_start };
        st = cs_append(&tmp, tail);
        if (st != CS_OK) { cs_destroy(&tmp); return st; }
    }
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
}

/* ==========================================================================
 * Part 4: Formatting & Deterministic Numbers (Section 11),
 *         Escaping, Data Codecs & Streaming (Section 12),
 *         Wildcard/Glob & Bounded PikeVM Regex (Section 13),
 *         and xxHash64 / SipHash-2-4 Hashing (Section 14)
 * ========================================================================== */

/* ==========================================================================
 * Section 11: Deterministic Numeric Parsing & Formatting
 * ========================================================================== */

cs_num_parse_options cs_default_num_parse_options(void) {
    cs_num_parse_options o;
    o.base              = 10;
    o.allow_whitespace  = false;
    o.allow_underscores = false;
    o.allow_base_prefix = false;
    o.allow_special_fp  = true;
    return o;
}

static inline int cs_char_to_digit(uint8_t c) {
    if (c >= '0' && c <= '9') return (int)(c - '0');
    if (c >= 'a' && c <= 'z') return (int)(c - 'a') + 10;
    if (c >= 'A' && c <= 'Z') return (int)(c - 'A') + 10;
    return -1;
}

static cs_status cs_parse_u64_common(cs_view v, cs_num_parse_options opts, bool is_signed,
                                     bool *out_neg, uint64_t *out_mag, size_t *out_consumed, bool require_exact) {
    if (!cs_view_is_valid(v) || !out_mag) return CS_EINVAL;
    if (opts.base != 0 && (opts.base < 2 || opts.base > 36)) return CS_EINVAL;

    const uint8_t *p = (const uint8_t *)v.data;
    size_t n = v.len;
    size_t i = 0;

    if (opts.allow_whitespace) {
        while (i < n && cs_is_ascii_ws_byte(p[i])) i++;
    }
    if (i >= n) return CS_ESYNTAX;

    bool neg = false;
    if (p[i] == '+') {
        i++;
    } else if (p[i] == '-') {
        if (!is_signed) return CS_ERANGE;
        neg = true;
        i++;
    }
    if (i >= n) return CS_ESYNTAX;

    int base = opts.base;
    if ((base == 0 || opts.allow_base_prefix) && p[i] == '0' && i + 1U < n) {
        uint8_t next = p[i + 1U];
        if ((next == 'x' || next == 'X') && (base == 0 || base == 16)) {
            base = 16;
            i += 2U;
        } else if ((next == 'b' || next == 'B') && (base == 0 || base == 2)) {
            base = 2;
            i += 2U;
        } else if ((next == 'o' || next == 'O') && (base == 0 || base == 8)) {
            base = 8;
            i += 2U;
        }
    }
    if (base == 0) base = 10;

    uint64_t acc = 0;
    uint64_t ubase = (uint64_t)base;
    size_t digits = 0;
    bool prev_underscore = false;
    bool overflowed = false;

    while (i < n) {
        uint8_t c = p[i];
        if (c == '_' && opts.allow_underscores) {
            if (digits == 0 || prev_underscore) return CS_ESYNTAX;
            prev_underscore = true;
            i++;
            continue;
        }
        int d = cs_char_to_digit(c);
        if (d < 0 || d >= base) break;
        prev_underscore = false;
        digits++;
        if (!overflowed) {
            if (acc > (UINT64_MAX - (uint64_t)d) / ubase) {
                overflowed = true;
            } else {
                acc = acc * ubase + (uint64_t)d;
            }
        }
        i++;
    }

    if (prev_underscore || digits == 0) return CS_ESYNTAX;

    size_t end_num = i;
    if (opts.allow_whitespace && require_exact) {
        while (i < n && cs_is_ascii_ws_byte(p[i])) i++;
    }
    if (require_exact && i != n) return CS_ESYNTAX;
    if (overflowed) return CS_ERANGE;

    if (out_neg) *out_neg = neg;
    *out_mag = acc;
    if (out_consumed) *out_consumed = require_exact ? i : end_num;
    return CS_OK;
}

cs_status cs_parse_u64_prefix(cs_view v, cs_num_parse_options opts, uint64_t *out_val, size_t *out_consumed) {
    bool neg = false;
    return cs_parse_u64_common(v, opts, false, &neg, out_val, out_consumed, false);
}

cs_status cs_parse_u64_exact(cs_view v, cs_num_parse_options opts, uint64_t *out_val) {
    bool neg = false;
    return cs_parse_u64_common(v, opts, false, &neg, out_val, NULL, true);
}

cs_status cs_parse_i64_prefix(cs_view v, cs_num_parse_options opts, int64_t *out_val, size_t *out_consumed) {
    if (!out_val) return CS_EINVAL;
    bool neg = false;
    uint64_t mag = 0;
    cs_status st = cs_parse_u64_common(v, opts, true, &neg, &mag, out_consumed, false);
    if (st != CS_OK) return st;
    if (!neg) {
        if (mag > (uint64_t)INT64_MAX) return CS_ERANGE;
        *out_val = (int64_t)mag;
    } else {
        uint64_t max_neg = (uint64_t)INT64_MAX + 1ULL;
        if (mag > max_neg) return CS_ERANGE;
        if (mag == max_neg) *out_val = INT64_MIN;
        else *out_val = -(int64_t)mag;
    }
    return CS_OK;
}

cs_status cs_parse_i64_exact(cs_view v, cs_num_parse_options opts, int64_t *out_val) {
    if (!out_val) return CS_EINVAL;
    bool neg = false;
    uint64_t mag = 0;
    cs_status st = cs_parse_u64_common(v, opts, true, &neg, &mag, NULL, true);
    if (st != CS_OK) return st;
    if (!neg) {
        if (mag > (uint64_t)INT64_MAX) return CS_ERANGE;
        *out_val = (int64_t)mag;
    } else {
        uint64_t max_neg = (uint64_t)INT64_MAX + 1ULL;
        if (mag > max_neg) return CS_ERANGE;
        if (mag == max_neg) *out_val = INT64_MIN;
        else *out_val = -(int64_t)mag;
    }
    return CS_OK;
}

cs_status cs_parse_bool_prefix(cs_view v, bool *out_val, size_t *out_consumed) {
    if (!cs_view_is_valid(v) || !out_val) return CS_EINVAL;
    if (v.len >= 4U) {
        cs_view h4 = { v.data, 4U };
        bool eq = false;
        cs_equal_mode(h4, CS_LIT("true"), CS_MATCH_ASCII_CASE_INSENSITIVE, &eq);
        if (eq) {
            *out_val = true;
            if (out_consumed) *out_consumed = 4U;
            return CS_OK;
        }
    }
    if (v.len >= 5U) {
        cs_view h5 = { v.data, 5U };
        bool eq = false;
        cs_equal_mode(h5, CS_LIT("false"), CS_MATCH_ASCII_CASE_INSENSITIVE, &eq);
        if (eq) {
            *out_val = false;
            if (out_consumed) *out_consumed = 5U;
            return CS_OK;
        }
    }
    if (v.len >= 1U && (v.data[0] == '1' || v.data[0] == '0')) {
        *out_val = (v.data[0] == '1');
        if (out_consumed) *out_consumed = 1U;
        return CS_OK;
    }
    return CS_ESYNTAX;
}

cs_status cs_parse_bool_exact(cs_view v, bool *out_val) {
    size_t consumed = 0;
    cs_status st = cs_parse_bool_prefix(v, out_val, &consumed);
    if (st != CS_OK) return st;
    return (consumed == v.len) ? CS_OK : CS_ESYNTAX;
}

/* Locale-independent IEEE-754 double parser */
static cs_status cs_parse_f64_common(cs_view v, cs_num_parse_options opts, double *out_val,
                                     size_t *out_consumed, bool require_exact) {
    if (!cs_view_is_valid(v) || !out_val) return CS_EINVAL;
    const uint8_t *p = (const uint8_t *)v.data;
    size_t n = v.len;
    size_t i = 0;

    if (opts.allow_whitespace) {
        while (i < n && cs_is_ascii_ws_byte(p[i])) i++;
    }
    if (i >= n) return CS_ESYNTAX;

    bool neg = false;
    if (p[i] == '+') i++;
    else if (p[i] == '-') { neg = true; i++; }
    if (i >= n) return CS_ESYNTAX;

    /* Check special values: inf, infinity, nan */
    if (opts.allow_special_fp) {
        size_t rem = n - i;
        if (rem >= 3U) {
            cs_view s3 = { (const char *)(p + i), 3U };
            bool is_inf = false, is_nan = false;
            cs_equal_mode(s3, CS_LIT("inf"), CS_MATCH_ASCII_CASE_INSENSITIVE, &is_inf);
            cs_equal_mode(s3, CS_LIT("nan"), CS_MATCH_ASCII_CASE_INSENSITIVE, &is_nan);
            if (is_inf) {
                i += 3U;
                if (n - i >= 5U) {
                    cs_view s5 = { (const char *)(p + i), 5U };
                    bool is_inity = false;
                    cs_equal_mode(s5, CS_LIT("inity"), CS_MATCH_ASCII_CASE_INSENSITIVE, &is_inity);
                    if (is_inity) i += 5U;
                }
                size_t end_tok = i;
                if (opts.allow_whitespace && require_exact) {
                    while (i < n && cs_is_ascii_ws_byte(p[i])) i++;
                }
                if (require_exact && i != n) return CS_ESYNTAX;
                *out_val = neg ? -INFINITY : INFINITY;
                if (out_consumed) *out_consumed = require_exact ? i : end_tok;
                return CS_OK;
            }
            if (is_nan) {
                i += 3U;
                size_t end_tok = i;
                if (opts.allow_whitespace && require_exact) {
                    while (i < n && cs_is_ascii_ws_byte(p[i])) i++;
                }
                if (require_exact && i != n) return CS_ESYNTAX;
                *out_val = NAN;
                if (out_consumed) *out_consumed = require_exact ? i : end_tok;
                return CS_OK;
            }
        }
    }

    /* Validate grammar and build clean ASCII buffer with '.' (never ',') */
    char clean[512];
    size_t clen = 0;
    if (neg) clean[clen++] = '-';

    size_t int_digits = 0, frac_digits = 0;
    bool prev_us = false;
    while (i < n) {
        uint8_t c = p[i];
        if (c == '_' && opts.allow_underscores) {
            if (int_digits == 0 || prev_us) return CS_ESYNTAX;
            prev_us = true;
            i++;
            continue;
        }
        if (c >= '0' && c <= '9') {
            prev_us = false;
            int_digits++;
            if (clen + 2U < sizeof(clean)) clean[clen++] = (char)c;
            i++;
        } else {
            break;
        }
    }
    if (prev_us) return CS_ESYNTAX;

    if (i < n && p[i] == '.') {
        if (clen + 2U < sizeof(clean)) clean[clen++] = '.';
        i++;
        while (i < n) {
            uint8_t c = p[i];
            if (c == '_' && opts.allow_underscores) {
                if (frac_digits == 0 || prev_us) return CS_ESYNTAX;
                prev_us = true;
                i++;
                continue;
            }
            if (c >= '0' && c <= '9') {
                prev_us = false;
                frac_digits++;
                if (clen + 2U < sizeof(clean)) clean[clen++] = (char)c;
                i++;
            } else {
                break;
            }
        }
        if (prev_us) return CS_ESYNTAX;
    }

    if (int_digits == 0 && frac_digits == 0) return CS_ESYNTAX;

    if (i < n && (p[i] == 'e' || p[i] == 'E')) {
        size_t save_e = i;
        i++;
        char exp_sign = 0;
        if (i < n && (p[i] == '+' || p[i] == '-')) {
            exp_sign = (char)p[i];
            i++;
        }
        size_t exp_digits = 0;
        size_t exp_start = i;
        while (i < n) {
            uint8_t c = p[i];
            if (c == '_' && opts.allow_underscores) {
                if (exp_digits == 0 || prev_us) return CS_ESYNTAX;
                prev_us = true;
                i++;
                continue;
            }
            if (c >= '0' && c <= '9') {
                prev_us = false;
                exp_digits++;
                i++;
            } else {
                break;
            }
        }
        if (prev_us) return CS_ESYNTAX;
        if (exp_digits == 0) {
            if (require_exact) return CS_ESYNTAX;
            i = save_e; /* Prefix stops before incomplete exponent */
        } else {
            if (clen + 2U < sizeof(clean)) clean[clen++] = 'e';
            if (exp_sign && clen + 2U < sizeof(clean)) clean[clen++] = exp_sign;
            for (size_t k = exp_start; k < i; k++) {
                if (p[k] != '_' && clen + 2U < sizeof(clean)) clean[clen++] = (char)p[k];
            }
        }
    }

    size_t end_num = i;
    if (opts.allow_whitespace && require_exact) {
        while (i < n && cs_is_ascii_ws_byte(p[i])) i++;
    }
    if (require_exact && i != n) return CS_ESYNTAX;
    clean[clen] = '\0';

    /* Parse via C locale strtod_l (or decimal replacement if locale uses comma) */
    errno = 0;
    char *endptr = NULL;
#if defined(__linux__) || defined(__APPLE__)
    locale_t cloc = newlocale(LC_NUMERIC_MASK, "C", (locale_t)0);
    double val = cloc ? strtod_l(clean, &endptr, cloc) : strtod(clean, &endptr);
    if (cloc) freelocale(cloc);
#else
    double val = strtod(clean, &endptr);
#endif
    if (errno == ERANGE) return CS_ERANGE;
    *out_val = val;
    if (out_consumed) *out_consumed = require_exact ? i : end_num;
    return CS_OK;
}

cs_status cs_parse_f64_prefix(cs_view v, cs_num_parse_options opts, double *out_val, size_t *out_consumed) {
    return cs_parse_f64_common(v, opts, out_val, out_consumed, false);
}

cs_status cs_parse_f64_exact(cs_view v, cs_num_parse_options opts, double *out_val) {
    return cs_parse_f64_common(v, opts, out_val, NULL, true);
}

static size_t cs_u64_to_chars(uint64_t val, int base, bool uppercase, char out_buf[68]) {
    const char *digits = uppercase ? "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                   : "0123456789abcdefghijklmnopqrstuvwxyz";
    char rev[68];
    size_t len = 0;
    uint64_t ubase = (uint64_t)base;
    if (val == 0) {
        out_buf[0] = '0';
        return 1U;
    }
    while (val > 0) {
        rev[len++] = digits[val % ubase];
        val /= ubase;
    }
    for (size_t i = 0; i < len; i++) {
        out_buf[i] = rev[len - 1U - i];
    }
    return len;
}

cs_status cs_format_u64_buffer(cs_buffer *dst, uint64_t val, int base, bool uppercase, size_t *out_required) {
    if (!dst || base < 2 || base > 36) return CS_EINVAL;
    char tmp[68];
    size_t len = cs_u64_to_chars(val, base, uppercase, tmp);
    cs_buffer_clear(dst);
    return cs_buffer_append(dst, (cs_view){ tmp, len }, out_required);
}

cs_status cs_format_i64_buffer(cs_buffer *dst, int64_t val, int base, bool uppercase, size_t *out_required) {
    if (!dst || base < 2 || base > 36) return CS_EINVAL;
    char tmp[70];
    size_t pos = 0;
    uint64_t mag = 0;
    if (val < 0) {
        tmp[pos++] = '-';
        mag = (val == INT64_MIN) ? ((uint64_t)INT64_MAX + 1ULL) : (uint64_t)(-val);
    } else {
        mag = (uint64_t)val;
    }
    pos += cs_u64_to_chars(mag, base, uppercase, tmp + pos);
    cs_buffer_clear(dst);
    return cs_buffer_append(dst, (cs_view){ tmp, pos }, out_required);
}

cs_status cs_format_u64(cs_string *dst, uint64_t val, int base, bool uppercase) {
    if (!dst || base < 2 || base > 36) return CS_EINVAL;
    char tmp[68];
    size_t len = cs_u64_to_chars(val, base, uppercase, tmp);
    return cs_assign(dst, (cs_view){ tmp, len });
}

cs_status cs_format_i64(cs_string *dst, int64_t val, int base, bool uppercase) {
    if (!dst || base < 2 || base > 36) return CS_EINVAL;
    char tmp[70];
    size_t pos = 0;
    uint64_t mag = 0;
    if (val < 0) {
        tmp[pos++] = '-';
        mag = (val == INT64_MIN) ? ((uint64_t)INT64_MAX + 1ULL) : (uint64_t)(-val);
    } else {
        mag = (uint64_t)val;
    }
    pos += cs_u64_to_chars(mag, base, uppercase, tmp + pos);
    return cs_assign(dst, (cs_view){ tmp, pos });
}

cs_status cs_format_bool(cs_string *dst, bool val) {
    return cs_assign(dst, val ? CS_LIT("true") : CS_LIT("false"));
}

/* Locale-independent IEEE-754 binary64 float formatting with shortest round-trip */
static size_t cs_format_f64_raw(double val, cs_float_fmt_mode mode, int precision, bool uppercase, char out_buf[128]) {
    if (isnan(val)) {
        const char *s = uppercase ? "NAN" : "nan";
        memcpy(out_buf, s, 4);
        return 3U;
    }
    if (isinf(val)) {
        if (signbit(val)) {
            const char *s = uppercase ? "-INF" : "-inf";
            memcpy(out_buf, s, 5);
            return 4U;
        } else {
            const char *s = uppercase ? "INF" : "inf";
            memcpy(out_buf, s, 4);
            return 3U;
        }
    }

    if (mode == CS_FLOAT_SHORTEST) {
        /* Try precisions 1..17 with %.precg until round-trip matches exact binary64 bits */
        uint64_t orig_bits = 0;
        memcpy(&orig_bits, &val, sizeof(double));
        for (int p = 1; p <= 17; p++) {
            int n = snprintf(out_buf, 128, uppercase ? "%.*G" : "%.*g", p, val);
            if (n <= 0 || n >= 128) continue;
            /* Ensure decimal point is '.' regardless of locale */
            for (int i = 0; i < n; i++) {
                if (out_buf[i] == ',') out_buf[i] = '.';
            }
            if (p < 17) {
                double check = 0.0;
                cs_num_parse_options popts = cs_default_num_parse_options();
                if (cs_parse_f64_exact((cs_view){ out_buf, (size_t)n }, popts, &check) == CS_OK) {
                    uint64_t check_bits = 0;
                    memcpy(&check_bits, &check, sizeof(double));
                    if (check_bits == orig_bits) {
                        return (size_t)n;
                    }
                }
            } else {
                return (size_t)n;
            }
        }
    }

    if (precision < 0) precision = 6;
    if (precision > 60) precision = 60;
    const char *fmt = (mode == CS_FLOAT_FIXED)
                      ? (uppercase ? "%.*F" : "%.*f")
                      : (uppercase ? "%.*E" : "%.*e");
    int n = snprintf(out_buf, 128, fmt, precision, val);
    if (n < 0) n = 0;
    if (n >= 128) n = 127;
    for (int i = 0; i < n; i++) {
        if (out_buf[i] == ',') out_buf[i] = '.';
    }
    return (size_t)n;
}

cs_status cs_format_f64_buffer(cs_buffer *dst, double val, cs_float_fmt_mode mode, int precision, bool uppercase, size_t *out_required) {
    if (!dst) return CS_EINVAL;
    char buf[128];
    size_t len = cs_format_f64_raw(val, mode, precision, uppercase, buf);
    cs_buffer_clear(dst);
    return cs_buffer_append(dst, (cs_view){ buf, len }, out_required);
}

cs_status cs_format_f64(cs_string *dst, double val, cs_float_fmt_mode mode, int precision, bool uppercase) {
    if (!dst) return CS_EINVAL;
    char buf[128];
    size_t len = cs_format_f64_raw(val, mode, precision, uppercase, buf);
    return cs_assign(dst, (cs_view){ buf, len });
}

/* ==========================================================================
 * printf Wrappers & Deterministic Typed Formatter (`cs_format_args`)
 * ========================================================================== */

cs_status cs_vappend_printf(cs_string *dst, const char *fmt, va_list ap) {
    if (!dst || !fmt) return CS_EINVAL;
#if !CS_ENABLE_FORMAT
    (void)ap;
    return CS_EUNSUPPORTED;
#else
    va_list ap_copy;
    va_copy(ap_copy, ap);
    int needed = vsnprintf(NULL, 0, fmt, ap_copy);
    va_end(ap_copy);
    if (needed < 0) return CS_ESYNTAX;

    size_t new_len = 0;
    if (cs_add_overflow_sz(dst->len, (size_t)needed, &new_len)) return CS_EOVERFLOW;
    cs_status st = cs_reserve(dst, new_len);
    if (st != CS_OK) return st;

    vsnprintf(dst->data + dst->len, (size_t)needed + 1U, fmt, ap);
    dst->len = new_len;
    dst->generation++;
    return CS_OK;
#endif
}

cs_status cs_vformat_printf(cs_string *dst, const char *fmt, va_list ap) {
    if (!dst || !fmt) return CS_EINVAL;
    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;
    st = cs_vappend_printf(&tmp, fmt, ap);
    if (st != CS_OK) { cs_destroy(&tmp); return st; }
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
}

cs_status cs_format_printf(cs_string *dst, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    cs_status st = cs_vformat_printf(dst, fmt, ap);
    va_end(ap);
    return st;
}

cs_status cs_append_printf(cs_string *dst, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    cs_status st = cs_vappend_printf(dst, fmt, ap);
    va_end(ap);
    return st;
}

cs_status cs_buffer_format_printf(cs_buffer *dst, size_t *out_required, const char *fmt, ...) {
    if (!dst || !fmt) return CS_EINVAL;
#if !CS_ENABLE_FORMAT
    (void)out_required;
    return CS_EUNSUPPORTED;
#else
    va_list ap, ap_copy;
    va_start(ap, fmt);
    va_copy(ap_copy, ap);
    int needed = vsnprintf(NULL, 0, fmt, ap_copy);
    va_end(ap_copy);
    if (needed < 0) { va_end(ap); return CS_ESYNTAX; }

    size_t req = (size_t)needed + 1U;
    if (out_required) *out_required = req;
    if ((size_t)needed > dst->cap) {
        va_end(ap);
        return CS_EBUFFER;
    }
    vsnprintf(dst->data, req, fmt, ap);
    dst->len = (size_t)needed;
    va_end(ap);
    return CS_OK;
#endif
}

/* Typed Formatter Implementation */
static cs_status cs_format_args_to_builder(cs_builder *b, cs_view fmt, const cs_fmt_arg *args, size_t arg_count) {
#if !CS_ENABLE_FORMAT
    (void)b; (void)fmt; (void)args; (void)arg_count;
    return CS_EUNSUPPORTED;
#else
    if (!cs_view_is_valid(fmt) || (!args && arg_count > 0)) return CS_EINVAL;

    const char *p = fmt.data;
    size_t n = fmt.len;
    size_t i = 0;
    size_t auto_idx = 0;
    int indexing_mode = 0; /* 0 = undecided, 1 = auto {}, 2 = manual {0} */

    while (i < n) {
        if (p[i] == '{') {
            if (i + 1U < n && p[i + 1U] == '{') {
                cs_builder_append_byte(b, '{');
                i += 2U;
                continue;
            }
            i++; /* skip '{' */
            if (i >= n) return CS_ESYNTAX;

            size_t arg_idx = 0;
            if (p[i] >= '0' && p[i] <= '9') {
                if (indexing_mode == 1) return CS_ESYNTAX;
                indexing_mode = 2;
                while (i < n && p[i] >= '0' && p[i] <= '9') {
                    arg_idx = arg_idx * 10U + (size_t)(p[i] - '0');
                    i++;
                }
            } else {
                if (indexing_mode == 2) return CS_ESYNTAX;
                indexing_mode = 1;
                arg_idx = auto_idx++;
            }
            if (arg_idx >= arg_count) return CS_EINVAL;

            /* Parse optional format spec `:[[fill]align][sign][#][0][width][.prec][type]` */
            char fill_char = ' ';
            char align_mode = 0; /* '<', '>', '^' */
            char sign_mode  = 0; /* '+', '-', ' ' */
            bool alt_form   = false;
            bool zero_pad   = false;
            size_t width    = 0;
            int precision   = -1;
            char type_spec  = 0;

            if (i < n && p[i] == ':') {
                i++;
                if (i + 1U < n && (p[i + 1U] == '<' || p[i + 1U] == '>' || p[i + 1U] == '^')) {
                    fill_char  = p[i];
                    align_mode = p[i + 1U];
                    i += 2U;
                } else if (i < n && (p[i] == '<' || p[i] == '>' || p[i] == '^')) {
                    align_mode = p[i];
                    i++;
                }
                if (i < n && (p[i] == '+' || p[i] == '-' || p[i] == ' ')) {
                    sign_mode = p[i];
                    i++;
                }
                if (i < n && p[i] == '#') {
                    alt_form = true;
                    i++;
                }
                if (i < n && p[i] == '0') {
                    zero_pad = true;
                    if (!align_mode) fill_char = '0';
                    i++;
                }
                while (i < n && p[i] >= '0' && p[i] <= '9') {
                    width = width * 10U + (size_t)(p[i] - '0');
                    i++;
                }
                if (i < n && p[i] == '.') {
                    i++;
                    precision = 0;
                    if (i >= n || p[i] < '0' || p[i] > '9') return CS_ESYNTAX;
                    while (i < n && p[i] >= '0' && p[i] <= '9') {
                        precision = precision * 10 + (int)(p[i] - '0');
                        i++;
                    }
                }
                if (i < n && p[i] != '}') {
                    type_spec = p[i];
                    i++;
                }
            }

            if (i >= n || p[i] != '}') return CS_ESYNTAX;
            i++; /* skip '}' */

            const cs_fmt_arg *arg = &args[arg_idx];
            char num_buf[160];
            cs_view payload = cs_empty_view();
            char prefix_buf[4];
            size_t prefix_len = 0;

            if (arg->kind == CS_FMT_VIEW) {
                if (type_spec != 0 && type_spec != 's') return CS_ESYNTAX;
                payload = arg->val.v;
                if (precision >= 0 && (size_t)precision < payload.len) {
                    payload.len = (size_t)precision;
                }
                if (!align_mode) align_mode = '<';
            } else if (arg->kind == CS_FMT_BOOL) {
                if (type_spec != 0 && type_spec != 's') return CS_ESYNTAX;
                payload = arg->val.b ? CS_LIT("true") : CS_LIT("false");
                if (!align_mode) align_mode = '<';
            } else if (arg->kind == CS_FMT_CP) {
                if (type_spec != 0 && type_spec != 'c') return CS_ESYNTAX;
                size_t clen = 0;
                cs_status cst = cs_encode_utf8_codepoint(arg->val.cp, num_buf, &clen);
                if (cst != CS_OK) return cst;
                payload = (cs_view){ num_buf, clen };
                if (!align_mode) align_mode = '<';
            } else if (arg->kind == CS_FMT_I64 || arg->kind == CS_FMT_U64) {
                int base = 10;
                bool upper = false;
                if (type_spec == 0 || type_spec == 'd') { base = 10; }
                else if (type_spec == 'b') { base = 2;  if (alt_form) { prefix_buf[0] = '0'; prefix_buf[1] = 'b'; prefix_len = 2; } }
                else if (type_spec == 'o') { base = 8;  if (alt_form) { prefix_buf[0] = '0'; prefix_buf[1] = 'o'; prefix_len = 2; } }
                else if (type_spec == 'x') { base = 16; if (alt_form) { prefix_buf[0] = '0'; prefix_buf[1] = 'x'; prefix_len = 2; } }
                else if (type_spec == 'X') { base = 16; upper = true; if (alt_form) { prefix_buf[0] = '0'; prefix_buf[1] = 'X'; prefix_len = 2; } }
                else return CS_ESYNTAX;

                uint64_t mag = 0;
                bool is_neg = false;
                if (arg->kind == CS_FMT_I64) {
                    int64_t v = arg->val.i64;
                    if (v < 0) {
                        is_neg = true;
                        mag = (v == INT64_MIN) ? ((uint64_t)INT64_MAX + 1ULL) : (uint64_t)(-v);
                    } else {
                        mag = (uint64_t)v;
                    }
                } else {
                    mag = arg->val.u64;
                }
                if (is_neg) {
                    memmove(prefix_buf + 1, prefix_buf, prefix_len);
                    prefix_buf[0] = '-';
                    prefix_len++;
                } else if (sign_mode == '+' || sign_mode == ' ') {
                    memmove(prefix_buf + 1, prefix_buf, prefix_len);
                    prefix_buf[0] = sign_mode;
                    prefix_len++;
                }
                size_t dlen = cs_u64_to_chars(mag, base, upper, num_buf);
                payload = (cs_view){ num_buf, dlen };
                if (!align_mode) align_mode = '>';
            } else if (arg->kind == CS_FMT_F64) {
                cs_float_fmt_mode fmode = CS_FLOAT_SHORTEST;
                bool upper = false;
                if (type_spec == 0 || type_spec == 'g') { fmode = CS_FLOAT_SHORTEST; }
                else if (type_spec == 'G') { fmode = CS_FLOAT_SHORTEST; upper = true; }
                else if (type_spec == 'f') { fmode = CS_FLOAT_FIXED; }
                else if (type_spec == 'F') { fmode = CS_FLOAT_FIXED; upper = true; }
                else if (type_spec == 'e') { fmode = CS_FLOAT_SCIENTIFIC; }
                else if (type_spec == 'E') { fmode = CS_FLOAT_SCIENTIFIC; upper = true; }
                else return CS_ESYNTAX;

                double fv = arg->val.f64;
                if (!isnan(fv) && signbit(fv)) {
                    prefix_buf[0] = '-';
                    prefix_len = 1;
                    fv = -fv;
                } else if (!isnan(fv) && (sign_mode == '+' || sign_mode == ' ')) {
                    prefix_buf[0] = sign_mode;
                    prefix_len = 1;
                }
                size_t flen = cs_format_f64_raw(fv, fmode, precision, upper, num_buf);
                payload = (cs_view){ num_buf, flen };
                if (!align_mode) align_mode = '>';
            }

            size_t content_len = prefix_len + payload.len;
            size_t pad_len = (width > content_len) ? (width - content_len) : 0;
            size_t left_pad = 0, right_pad = 0;
            if (align_mode == '<') {
                right_pad = pad_len;
            } else if (align_mode == '^') {
                left_pad  = pad_len / 2U;
                right_pad = pad_len - left_pad;
            } else {
                left_pad = pad_len;
            }

            if (zero_pad && align_mode == '>' && fill_char == '0') {
                if (prefix_len > 0) cs_builder_append(b, (cs_view){ prefix_buf, prefix_len });
                for (size_t k = 0; k < left_pad; k++) cs_builder_append_byte(b, '0');
                cs_builder_append(b, payload);
            } else {
                for (size_t k = 0; k < left_pad; k++) cs_builder_append_byte(b, (uint8_t)fill_char);
                if (prefix_len > 0) cs_builder_append(b, (cs_view){ prefix_buf, prefix_len });
                cs_builder_append(b, payload);
                for (size_t k = 0; k < right_pad; k++) cs_builder_append_byte(b, (uint8_t)fill_char);
            }
        } else if (p[i] == '}') {
            if (i + 1U < n && p[i + 1U] == '}') {
                cs_builder_append_byte(b, '}');
                i += 2U;
                continue;
            }
            return CS_ESYNTAX;
        } else {
            size_t start = i;
            while (i < n && p[i] != '{' && p[i] != '}') i++;
            cs_builder_append(b, (cs_view){ p + start, i - start });
        }
    }
    return b->sticky_err;
#endif
}

cs_status cs_append_format_args(cs_string *dst, cs_view fmt, const cs_fmt_arg *args, size_t arg_count) {
    if (!dst) return CS_EINVAL;
    size_t orig_len = dst->len;
    cs_builder b;
    cs_builder_init_string(&b, dst);
    cs_status st = cs_format_args_to_builder(&b, fmt, args, arg_count);
    if (st != CS_OK) {
        cs_truncate_bytes(dst, orig_len);
        return st;
    }
    return CS_OK;
}

cs_status cs_format_args(cs_string *dst, cs_view fmt, const cs_fmt_arg *args, size_t arg_count) {
    if (!dst) return CS_EINVAL;
    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;
    st = cs_append_format_args(&tmp, fmt, args, arg_count);
    if (st != CS_OK) { cs_destroy(&tmp); return st; }
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
}

cs_status cs_buffer_format_args(cs_buffer *dst, cs_view fmt, const cs_fmt_arg *args, size_t arg_count, size_t *out_required) {
    if (!dst) return CS_EINVAL;
    cs_buffer_clear(dst);
    cs_builder b;
    cs_builder_init_buffer(&b, dst);
    cs_status st = cs_format_args_to_builder(&b, fmt, args, arg_count);
    if (st != CS_OK && st != CS_EBUFFER) {
        cs_buffer_clear(dst);
        return st;
    }
    return cs_builder_finish(&b, NULL, out_required);
}

/* ==========================================================================
 * Section 12: Escaping, Data Codecs & Streaming State Machines
 * ========================================================================== */

/* 1. Hex Encode / Decode */
cs_status cs_hex_encode_buffer(cs_buffer *dst, cs_view data, bool uppercase, size_t *out_required) {
    if (!dst || !cs_view_is_valid(data)) return CS_EINVAL;
#if !CS_ENABLE_CODECS
    (void)uppercase; (void)out_required;
    return CS_EUNSUPPORTED;
#else
    size_t out_len = 0;
    if (cs_mul_overflow_sz(data.len, 2U, &out_len)) return CS_EOVERFLOW;
    size_t req = 0;
    if (cs_add_overflow_sz(out_len, 1U, &req)) return CS_EOVERFLOW;
    if (out_required) *out_required = req;
    if (out_len > dst->cap) return CS_EBUFFER;

    const char *hex = uppercase ? "0123456789ABCDEF" : "0123456789abcdef";
    const uint8_t *p = (const uint8_t *)data.data;
    for (size_t i = 0; i < data.len; i++) {
        dst->data[i * 2U]      = hex[p[i] >> 4];
        dst->data[i * 2U + 1U] = hex[p[i] & 0x0FU];
    }
    dst->len = out_len;
    dst->data[dst->len] = '\0';
    return CS_OK;
#endif
}

cs_status cs_hex_encode(cs_string *dst, cs_view data, bool uppercase) {
    if (!dst || !cs_view_is_valid(data)) return CS_EINVAL;
#if !CS_ENABLE_CODECS
    (void)uppercase;
    return CS_EUNSUPPORTED;
#else
    size_t out_len = 0;
    if (cs_mul_overflow_sz(data.len, 2U, &out_len)) return CS_EOVERFLOW;
    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;
    st = cs_reserve(&tmp, out_len);
    if (st != CS_OK) { cs_destroy(&tmp); return st; }

    const char *hex = uppercase ? "0123456789ABCDEF" : "0123456789abcdef";
    const uint8_t *p = (const uint8_t *)data.data;
    for (size_t i = 0; i < data.len; i++) {
        tmp.data[i * 2U]      = hex[p[i] >> 4];
        tmp.data[i * 2U + 1U] = hex[p[i] & 0x0FU];
    }
    tmp.len = out_len;
    if (out_len > 0 || cs_string_is_heap(&tmp)) tmp.data[tmp.len] = '\0';
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
#endif
}

static inline int cs_hex_nibble(uint8_t c) {
    if (c >= '0' && c <= '9') return (int)(c - '0');
    if (c >= 'a' && c <= 'f') return (int)(c - 'a') + 10;
    if (c >= 'A' && c <= 'F') return (int)(c - 'A') + 10;
    return -1;
}

cs_status cs_hex_decode_buffer(cs_buffer *dst, cs_view hex_str, size_t *out_required) {
    if (!dst || !cs_view_is_valid(hex_str)) return CS_EINVAL;
#if !CS_ENABLE_CODECS
    (void)out_required;
    return CS_EUNSUPPORTED;
#else
    if ((hex_str.len & 1U) != 0U) return CS_ESYNTAX;
    size_t out_len = hex_str.len / 2U;
    if (out_required) *out_required = out_len + 1U;
    const uint8_t *p = (const uint8_t *)hex_str.data;
    /* Validate syntax before touching dst */
    for (size_t i = 0; i < hex_str.len; i++) {
        if (cs_hex_nibble(p[i]) < 0) return CS_ESYNTAX;
    }
    if (out_len > dst->cap) return CS_EBUFFER;

    for (size_t i = 0; i < out_len; i++) {
        int hi = cs_hex_nibble(p[i * 2U]);
        int lo = cs_hex_nibble(p[i * 2U + 1U]);
        dst->data[i] = (char)((hi << 4) | lo);
    }
    dst->len = out_len;
    dst->data[dst->len] = '\0';
    return CS_OK;
#endif
}

cs_status cs_hex_decode(cs_string *dst, cs_view hex_str) {
    if (!dst || !cs_view_is_valid(hex_str)) return CS_EINVAL;
#if !CS_ENABLE_CODECS
    return CS_EUNSUPPORTED;
#else
    if ((hex_str.len & 1U) != 0U) return CS_ESYNTAX;
    size_t out_len = hex_str.len / 2U;
    const uint8_t *p = (const uint8_t *)hex_str.data;
    for (size_t i = 0; i < hex_str.len; i++) {
        if (cs_hex_nibble(p[i]) < 0) return CS_ESYNTAX;
    }
    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;
    st = cs_reserve(&tmp, out_len);
    if (st != CS_OK) { cs_destroy(&tmp); return st; }

    for (size_t i = 0; i < out_len; i++) {
        int hi = cs_hex_nibble(p[i * 2U]);
        int lo = cs_hex_nibble(p[i * 2U + 1U]);
        tmp.data[i] = (char)((hi << 4) | lo);
    }
    tmp.len = out_len;
    if (out_len > 0 || cs_string_is_heap(&tmp)) tmp.data[tmp.len] = '\0';
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
#endif
}

/* 2. Base64 & Base64URL (RFC 4648) + Streaming */
#if CS_ENABLE_CODECS
static const char cs_b64_std_alpha[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static const char cs_b64_url_alpha[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

static inline int cs_b64_decode_char(uint8_t c, cs_base64_variant variant) {
    if (c >= 'A' && c <= 'Z') return (int)(c - 'A');
    if (c >= 'a' && c <= 'z') return (int)(c - 'a') + 26;
    if (c >= '0' && c <= '9') return (int)(c - '0') + 52;
    if (variant == CS_BASE64_STANDARD) {
        if (c == '+') return 62;
        if (c == '/') return 63;
    } else {
        if (c == '-') return 62;
        if (c == '_') return 63;
    }
    return -1;
}
#endif

void cs_base64_enc_stream_init(cs_base64_enc_stream *st, cs_base64_variant variant, cs_base64_pad_policy pad) {
    if (!st) return;
    memset(st, 0, sizeof(*st));
    st->variant = variant;
    st->pad     = pad;
}

cs_status cs_base64_enc_stream_feed(cs_base64_enc_stream *st, cs_view chunk, cs_string *dst) {
    if (!st || !dst || !cs_view_is_valid(chunk)) return CS_EINVAL;
#if !CS_ENABLE_CODECS
    return CS_EUNSUPPORTED;
#else
    const char *alpha = (st->variant == CS_BASE64_URL) ? cs_b64_url_alpha : cs_b64_std_alpha;
    const uint8_t *p = (const uint8_t *)chunk.data;
    size_t i = 0;

    while (i < chunk.len) {
        st->rem[st->rem_len++] = p[i++];
        if (st->rem_len == 3U) {
            uint32_t v = ((uint32_t)st->rem[0] << 16) | ((uint32_t)st->rem[1] << 8) | (uint32_t)st->rem[2];
            char out[4] = {
                alpha[(v >> 18) & 0x3FU],
                alpha[(v >> 12) & 0x3FU],
                alpha[(v >> 6)  & 0x3FU],
                alpha[v & 0x3FU]
            };
            cs_status err = cs_append(dst, (cs_view){ out, 4U });
            if (err != CS_OK) return err;
            st->rem_len = 0;
        }
    }
    return CS_OK;
#endif
}

cs_status cs_base64_enc_stream_finish(cs_base64_enc_stream *st, cs_string *dst) {
    if (!st || !dst) return CS_EINVAL;
#if !CS_ENABLE_CODECS
    return CS_EUNSUPPORTED;
#else
    const char *alpha = (st->variant == CS_BASE64_URL) ? cs_b64_url_alpha : cs_b64_std_alpha;
    bool emit_pad = (st->pad != CS_B64_PAD_OMIT);

    if (st->rem_len == 1U) {
        uint32_t v = (uint32_t)st->rem[0] << 16;
        char out[4] = {
            alpha[(v >> 18) & 0x3FU],
            alpha[(v >> 12) & 0x3FU],
            '=',
            '='
        };
        st->rem_len = 0;
        return cs_append(dst, (cs_view){ out, emit_pad ? 4U : 2U });
    } else if (st->rem_len == 2U) {
        uint32_t v = ((uint32_t)st->rem[0] << 16) | ((uint32_t)st->rem[1] << 8);
        char out[4] = {
            alpha[(v >> 18) & 0x3FU],
            alpha[(v >> 12) & 0x3FU],
            alpha[(v >> 6)  & 0x3FU],
            '='
        };
        st->rem_len = 0;
        return cs_append(dst, (cs_view){ out, emit_pad ? 4U : 3U });
    }
    return CS_OK;
#endif
}

cs_status cs_base64_encode(cs_string *dst, cs_view data, cs_base64_variant variant, cs_base64_pad_policy pad) {
    if (!dst || !cs_view_is_valid(data)) return CS_EINVAL;
    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;

    cs_base64_enc_stream est;
    cs_base64_enc_stream_init(&est, variant, pad);
    st = cs_base64_enc_stream_feed(&est, data, &tmp);
    if (st == CS_OK) st = cs_base64_enc_stream_finish(&est, &tmp);
    if (st != CS_OK) { cs_destroy(&tmp); return st; }
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
}

cs_status cs_base64_encode_buffer(cs_buffer *dst, cs_view data, cs_base64_variant variant, cs_base64_pad_policy pad, size_t *out_required) {
    if (!dst || !cs_view_is_valid(data)) return CS_EINVAL;
#if !CS_ENABLE_CODECS
    (void)variant; (void)pad; (void)out_required;
    return CS_EUNSUPPORTED;
#else
    bool emit_pad = (pad != CS_B64_PAD_OMIT);
    size_t full_triples = data.len / 3U;
    size_t rem = data.len % 3U;
    size_t out_len = 0;
    if (cs_mul_overflow_sz(full_triples, 4U, &out_len)) return CS_EOVERFLOW;
    if (rem > 0) {
        size_t tail = emit_pad ? 4U : (rem + 1U);
        if (cs_add_overflow_sz(out_len, tail, &out_len)) return CS_EOVERFLOW;
    }
    size_t req_with_nul = 0;
    if (cs_add_overflow_sz(out_len, 1U, &req_with_nul)) return CS_EOVERFLOW;
    if (out_required) *out_required = req_with_nul;
    if (out_len > dst->cap) return CS_EBUFFER;

    const char *alpha = (variant == CS_BASE64_URL) ? cs_b64_url_alpha : cs_b64_std_alpha;
    const uint8_t *p = (const uint8_t *)data.data;
    size_t in_pos = 0, out_pos = 0;
    while (in_pos + 3U <= data.len) {
        uint32_t v = ((uint32_t)p[in_pos] << 16) | ((uint32_t)p[in_pos + 1U] << 8) | (uint32_t)p[in_pos + 2U];
        dst->data[out_pos++] = alpha[(v >> 18) & 0x3FU];
        dst->data[out_pos++] = alpha[(v >> 12) & 0x3FU];
        dst->data[out_pos++] = alpha[(v >> 6)  & 0x3FU];
        dst->data[out_pos++] = alpha[v & 0x3FU];
        in_pos += 3U;
    }
    if (rem == 1U) {
        uint32_t v = (uint32_t)p[in_pos] << 16;
        dst->data[out_pos++] = alpha[(v >> 18) & 0x3FU];
        dst->data[out_pos++] = alpha[(v >> 12) & 0x3FU];
        if (emit_pad) {
            dst->data[out_pos++] = '=';
            dst->data[out_pos++] = '=';
        }
    } else if (rem == 2U) {
        uint32_t v = ((uint32_t)p[in_pos] << 16) | ((uint32_t)p[in_pos + 1U] << 8);
        dst->data[out_pos++] = alpha[(v >> 18) & 0x3FU];
        dst->data[out_pos++] = alpha[(v >> 12) & 0x3FU];
        dst->data[out_pos++] = alpha[(v >> 6)  & 0x3FU];
        if (emit_pad) {
            dst->data[out_pos++] = '=';
        }
    }
    dst->len = out_pos;
    dst->data[dst->len] = '\0';
    return CS_OK;
#endif
}

void cs_base64_dec_stream_init(cs_base64_dec_stream *st, cs_base64_variant variant, cs_base64_pad_policy pad) {
    if (!st) return;
    memset(st, 0, sizeof(*st));
    st->variant = variant;
    st->pad     = pad;
}

cs_status cs_base64_dec_stream_feed(cs_base64_dec_stream *st, cs_view chunk, cs_string *dst) {
    if (!st || !dst || !cs_view_is_valid(chunk)) return CS_EINVAL;
#if !CS_ENABLE_CODECS
    return CS_EUNSUPPORTED;
#else
    const uint8_t *p = (const uint8_t *)chunk.data;
    for (size_t i = 0; i < chunk.len; i++) {
        uint8_t c = p[i];
        if (c == '=') {
            if (st->pad == CS_B64_PAD_OMIT) return CS_ESYNTAX;
            if (st->quad_len < 2U) return CS_ESYNTAX;
            st->quad[st->quad_len++] = 0xFFU;
            st->pad_seen++;
            if (st->pad_seen > 2U) return CS_ESYNTAX;
        } else {
            if (st->pad_seen > 0) return CS_ESYNTAX; /* Non-pad character after '=' */
            int val = cs_b64_decode_char(c, st->variant);
            if (val < 0) return CS_ESYNTAX;
            st->quad[st->quad_len++] = (uint8_t)val;
        }

        if (st->quad_len == 4U) {
            if (st->quad[2] == 0xFFU && st->quad[3] != 0xFFU) return CS_ESYNTAX;
            if (st->quad[2] == 0xFFU) {
                /* 2 pad chars -> 1 byte; check canonical zero bits in quad[1] (bottom 4 bits must be 0) */
                if ((st->quad[1] & 0x0FU) != 0U) return CS_ESYNTAX;
                uint8_t out1 = (uint8_t)((st->quad[0] << 2) | (st->quad[1] >> 4));
                cs_status err = cs_append_byte(dst, out1);
                if (err != CS_OK) return err;
            } else if (st->quad[3] == 0xFFU) {
                /* 1 pad char -> 2 bytes; check canonical zero bits in quad[2] (bottom 2 bits must be 0) */
                if ((st->quad[2] & 0x03U) != 0U) return CS_ESYNTAX;
                char out2[2] = {
                    (char)((st->quad[0] << 2) | (st->quad[1] >> 4)),
                    (char)(((st->quad[1] & 0x0FU) << 4) | (st->quad[2] >> 2))
                };
                cs_status err = cs_append(dst, (cs_view){ out2, 2U });
                if (err != CS_OK) return err;
            } else {
                uint32_t v = ((uint32_t)st->quad[0] << 18) |
                             ((uint32_t)st->quad[1] << 12) |
                             ((uint32_t)st->quad[2] << 6)  |
                             (uint32_t)st->quad[3];
                char out3[3] = {
                    (char)((v >> 16) & 0xFFU),
                    (char)((v >> 8)  & 0xFFU),
                    (char)(v & 0xFFU)
                };
                cs_status err = cs_append(dst, (cs_view){ out3, 3U });
                if (err != CS_OK) return err;
            }
            st->quad_len = 0;
        }
    }
    return CS_OK;
#endif
}

cs_status cs_base64_dec_stream_finish(cs_base64_dec_stream *st, cs_string *dst) {
    if (!st || !dst) return CS_EINVAL;
#if !CS_ENABLE_CODECS
    return CS_EUNSUPPORTED;
#else
    if (st->quad_len == 0) return CS_OK;
    if (st->pad == CS_B64_PAD_REQUIRED || st->pad_seen > 0) return CS_ESYNTAX;
    if (st->quad_len == 1U) return CS_ESYNTAX;
    if (st->quad_len == 2U) {
        if ((st->quad[1] & 0x0FU) != 0U) return CS_ESYNTAX;
        uint8_t out1 = (uint8_t)((st->quad[0] << 2) | (st->quad[1] >> 4));
        st->quad_len = 0;
        return cs_append_byte(dst, out1);
    }
    if (st->quad_len == 3U) {
        if ((st->quad[2] & 0x03U) != 0U) return CS_ESYNTAX;
        char out2[2] = {
            (char)((st->quad[0] << 2) | (st->quad[1] >> 4)),
            (char)(((st->quad[1] & 0x0FU) << 4) | (st->quad[2] >> 2))
        };
        st->quad_len = 0;
        return cs_append(dst, (cs_view){ out2, 2U });
    }
    return CS_ESYNTAX;
#endif
}

cs_status cs_base64_decode(cs_string *dst, cs_view b64_str, cs_base64_variant variant, cs_base64_pad_policy pad) {
    if (!dst || !cs_view_is_valid(b64_str)) return CS_EINVAL;
    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;

    cs_base64_dec_stream dstm;
    cs_base64_dec_stream_init(&dstm, variant, pad);
    st = cs_base64_dec_stream_feed(&dstm, b64_str, &tmp);
    if (st == CS_OK) st = cs_base64_dec_stream_finish(&dstm, &tmp);
    if (st != CS_OK) { cs_destroy(&tmp); return st; }
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
}

cs_status cs_base64_decode_buffer(cs_buffer *dst, cs_view b64_str, cs_base64_variant variant, cs_base64_pad_policy pad, size_t *out_required) {
    if (!dst || !cs_view_is_valid(b64_str)) return CS_EINVAL;
#if !CS_ENABLE_CODECS
    (void)variant; (void)pad; (void)out_required;
    return CS_EUNSUPPORTED;
#else
    const uint8_t *p = (const uint8_t *)b64_str.data;
    uint8_t quad[4];
    size_t quad_len = 0, pad_seen = 0, out_len = 0;

    /* Pass 1: validate syntax and compute exact decoded byte length without touching dst */
    for (size_t i = 0; i < b64_str.len; i++) {
        uint8_t c = p[i];
        if (c == '=') {
            if (pad == CS_B64_PAD_OMIT || quad_len < 2U) return CS_ESYNTAX;
            quad[quad_len++] = 0xFFU;
            if (++pad_seen > 2U) return CS_ESYNTAX;
        } else {
            if (pad_seen > 0) return CS_ESYNTAX;
            int val = cs_b64_decode_char(c, variant);
            if (val < 0) return CS_ESYNTAX;
            quad[quad_len++] = (uint8_t)val;
        }
        if (quad_len == 4U) {
            if (quad[2] == 0xFFU && quad[3] != 0xFFU) return CS_ESYNTAX;
            if (quad[2] == 0xFFU) {
                if ((quad[1] & 0x0FU) != 0U) return CS_ESYNTAX;
                out_len += 1U;
            } else if (quad[3] == 0xFFU) {
                if ((quad[2] & 0x03U) != 0U) return CS_ESYNTAX;
                out_len += 2U;
            } else {
                out_len += 3U;
            }
            quad_len = 0;
        }
    }
    if (quad_len != 0) {
        if (pad == CS_B64_PAD_REQUIRED || pad_seen > 0 || quad_len == 1U) return CS_ESYNTAX;
        if (quad_len == 2U) {
            if ((quad[1] & 0x0FU) != 0U) return CS_ESYNTAX;
            out_len += 1U;
        } else if (quad_len == 3U) {
            if ((quad[2] & 0x03U) != 0U) return CS_ESYNTAX;
            out_len += 2U;
        }
    }

    if (out_required) *out_required = out_len + 1U;
    if (out_len > dst->cap) return CS_EBUFFER;

    /* Pass 2: write decoded bytes directly into dst->data */
    size_t out_pos = 0;
    quad_len = 0;
    for (size_t i = 0; i < b64_str.len; i++) {
        uint8_t c = p[i];
        quad[quad_len++] = (c == '=') ? 0xFFU : (uint8_t)cs_b64_decode_char(c, variant);
        if (quad_len == 4U) {
            if (quad[2] == 0xFFU) {
                dst->data[out_pos++] = (char)((quad[0] << 2) | (quad[1] >> 4));
            } else if (quad[3] == 0xFFU) {
                dst->data[out_pos++] = (char)((quad[0] << 2) | (quad[1] >> 4));
                dst->data[out_pos++] = (char)(((quad[1] & 0x0FU) << 4) | (quad[2] >> 2));
            } else {
                uint32_t v = ((uint32_t)quad[0] << 18) | ((uint32_t)quad[1] << 12) | ((uint32_t)quad[2] << 6) | (uint32_t)quad[3];
                dst->data[out_pos++] = (char)((v >> 16) & 0xFFU);
                dst->data[out_pos++] = (char)((v >> 8)  & 0xFFU);
                dst->data[out_pos++] = (char)(v & 0xFFU);
            }
            quad_len = 0;
        }
    }
    if (quad_len == 2U) {
        dst->data[out_pos++] = (char)((quad[0] << 2) | (quad[1] >> 4));
    } else if (quad_len == 3U) {
        dst->data[out_pos++] = (char)((quad[0] << 2) | (quad[1] >> 4));
        dst->data[out_pos++] = (char)(((quad[1] & 0x0FU) << 4) | (quad[2] >> 2));
    }
    dst->len = out_pos;
    dst->data[dst->len] = '\0';
    return CS_OK;
#endif
}

/* 3. Streaming UTF-8 Validator / Repair State Machine */
void cs_utf8_stream_init(cs_utf8_stream *st, cs_decode_mode mode) {
    if (!st) return;
    memset(st, 0, sizeof(*st));
    st->mode = mode;
}

cs_status cs_utf8_stream_feed(cs_utf8_stream *st, cs_view chunk, cs_string *dst, size_t *out_chunk_consumed) {
    if (!st || !dst || !cs_view_is_valid(chunk)) return CS_EINVAL;
    size_t consumed = 0;
    const uint8_t *p = (const uint8_t *)chunk.data;

    while (consumed < chunk.len) {
        /* If we have pending bytes from the previous chunk, top them up one byte at a time */
        if (st->pending_len > 0) {
            st->pending[st->pending_len++] = p[consumed++];
            uint32_t cp = 0;
            size_t seq_len = 0, err_len = 0;
            if (cs_utf8_decode_one_raw(st->pending, st->pending_len, &cp, &seq_len, &err_len)) {
                cs_status err = cs_append(dst, (cs_view){ (const char *)st->pending, seq_len });
                if (err != CS_OK) return err;
                if (seq_len < st->pending_len) {
                    memmove(st->pending, st->pending + seq_len, st->pending_len - seq_len);
                }
                st->pending_len -= seq_len;
                st->total_consumed += seq_len;
            } else {
                /* Check if st->pending is still a valid prefix of a longer UTF-8 sequence */
                uint8_t b0 = st->pending[0];
                size_t needed = (b0 >= 0xF0U && b0 <= 0xF4U) ? 4U :
                                ((b0 >= 0xE0U && b0 <= 0xEFU) ? 3U :
                                ((b0 >= 0xC2U && b0 <= 0xDFU) ? 2U : 1U));
                if (st->pending_len < needed && err_len == st->pending_len) {
                    /* Still waiting for more continuation bytes! */
                    continue;
                }
                /* Definitely ill-formed! */
                if (st->mode == CS_DECODE_STRICT) {
                    /* Rollback the newly added byte if err_len < pending_len */
                    if (out_chunk_consumed) *out_chunk_consumed = consumed - (st->pending_len - err_len);
                    return CS_EUTF8;
                }
                cs_status err = cs_append(dst, CS_LIT("\xEF\xBF\xBD"));
                if (err != CS_OK) return err;
                if (err_len < st->pending_len) {
                    memmove(st->pending, st->pending + err_len, st->pending_len - err_len);
                }
                st->pending_len -= err_len;
                st->total_consumed += err_len;
            }
            continue;
        }

        /* Fast path when pending_len == 0 */
        size_t rem = chunk.len - consumed;
        uint32_t cp = 0;
        size_t seq_len = 0, err_len = 0;
        if (cs_utf8_decode_one_raw(p + consumed, rem, &cp, &seq_len, &err_len)) {
            cs_status err = cs_append(dst, (cs_view){ (const char *)(p + consumed), seq_len });
            if (err != CS_OK) return err;
            consumed += seq_len;
            st->total_consumed += seq_len;
        } else {
            uint8_t b0 = p[consumed];
            size_t needed = (b0 >= 0xF0U && b0 <= 0xF4U) ? 4U :
                            ((b0 >= 0xE0U && b0 <= 0xEFU) ? 3U :
                            ((b0 >= 0xC2U && b0 <= 0xDFU) ? 2U : 1U));
            if (rem < needed && err_len == rem) {
                /* Incomplete sequence at the end of chunk: buffer in st->pending */
                memcpy(st->pending, p + consumed, rem);
                st->pending_len = rem;
                consumed += rem;
                break;
            }
            if (st->mode == CS_DECODE_STRICT) {
                if (out_chunk_consumed) *out_chunk_consumed = consumed;
                return CS_EUTF8;
            }
            cs_status err = cs_append(dst, CS_LIT("\xEF\xBF\xBD"));
            if (err != CS_OK) return err;
            consumed += err_len;
            st->total_consumed += err_len;
        }
    }

    if (out_chunk_consumed) *out_chunk_consumed = consumed;
    return CS_OK;
}

cs_status cs_utf8_stream_finish(cs_utf8_stream *st, cs_string *dst) {
    if (!st || !dst) return CS_EINVAL;
    while (st->pending_len > 0) {
        if (st->mode == CS_DECODE_STRICT) {
            return CS_EUTF8;
        }
        uint32_t cp = 0;
        size_t seq_len = 0, err_len = 0;
        cs_utf8_decode_one_raw(st->pending, st->pending_len, &cp, &seq_len, &err_len);
        cs_status err = cs_append(dst, CS_LIT("\xEF\xBF\xBD"));
        if (err != CS_OK) return err;
        if (err_len < st->pending_len) {
            memmove(st->pending, st->pending + err_len, st->pending_len - err_len);
        }
        st->pending_len -= err_len;
        st->total_consumed += err_len;
    }
    return CS_OK;
}

/* 4. JSON String Escape / Unescape (RFC 8259) */
cs_status cs_json_escape(cs_string *dst, cs_view utf8_str, bool escape_non_ascii) {
    if (!dst || !cs_view_is_valid(utf8_str)) return CS_EINVAL;
#if !CS_ENABLE_CODECS
    (void)escape_non_ascii;
    return CS_EUNSUPPORTED;
#else
    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;

    cs_cp_iter it;
    cs_cp_iter_init(&it, utf8_str, CS_DECODE_STRICT);
    uint32_t cp = 0;
    size_t s = 0, e = 0;
    const char *hex = "0123456789abcdef";

    while ((st = cs_cp_iter_next(&it, &cp, &s, &e)) == CS_OK) {
        switch (cp) {
            case '\"': st = cs_append(&tmp, CS_LIT("\\\"")); break;
            case '\\': st = cs_append(&tmp, CS_LIT("\\\\")); break;
            case '\b': st = cs_append(&tmp, CS_LIT("\\b"));  break;
            case '\f': st = cs_append(&tmp, CS_LIT("\\f"));  break;
            case '\n': st = cs_append(&tmp, CS_LIT("\\n"));  break;
            case '\r': st = cs_append(&tmp, CS_LIT("\\r"));  break;
            case '\t': st = cs_append(&tmp, CS_LIT("\\t"));  break;
            default:
                if (cp < 0x20U || (escape_non_ascii && cp >= 0x80U && cp <= 0xFFFFU)) {
                    char u[6] = { '\\', 'u',
                                  hex[(cp >> 12) & 0xFU], hex[(cp >> 8) & 0xFU],
                                  hex[(cp >> 4)  & 0xFU], hex[cp & 0xFU] };
                    st = cs_append(&tmp, (cs_view){ u, 6U });
                } else if (escape_non_ascii && cp > 0xFFFFU) {
                    uint32_t u = cp - 0x10000U;
                    uint16_t hi = (uint16_t)(0xD800U | (u >> 10));
                    uint16_t lo = (uint16_t)(0xDC00U | (u & 0x3FFU));
                    char pair[12] = {
                        '\\', 'u', hex[(hi >> 12) & 0xFU], hex[(hi >> 8) & 0xFU], hex[(hi >> 4) & 0xFU], hex[hi & 0xFU],
                        '\\', 'u', hex[(lo >> 12) & 0xFU], hex[(lo >> 8) & 0xFU], hex[(lo >> 4) & 0xFU], hex[lo & 0xFU]
                    };
                    st = cs_append(&tmp, (cs_view){ pair, 12U });
                } else {
                    st = cs_append(&tmp, (cs_view){ utf8_str.data + s, e - s });
                }
                break;
        }
        if (st != CS_OK) { cs_destroy(&tmp); return st; }
    }
    if (st != CS_END) { cs_destroy(&tmp); return st; }
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
#endif
}

#if CS_ENABLE_CODECS
static bool cs_parse_hex4(const uint8_t *p, uint16_t *out) {
    uint16_t v = 0;
    for (int i = 0; i < 4; i++) {
        int n = cs_hex_nibble(p[i]);
        if (n < 0) return false;
        v = (uint16_t)((v << 4) | (uint16_t)n);
    }
    *out = v;
    return true;
}
#endif

cs_status cs_json_unescape(cs_string *dst, cs_view escaped_str) {
    if (!dst || !cs_view_is_valid(escaped_str)) return CS_EINVAL;
#if !CS_ENABLE_CODECS
    return CS_EUNSUPPORTED;
#else
    cs_status st = cs_validate_utf8(escaped_str, NULL);
    if (st != CS_OK) return st;

    cs_string tmp;
    st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;

    const uint8_t *p = (const uint8_t *)escaped_str.data;
    size_t n = escaped_str.len;
    size_t i = 0;

    while (i < n) {
        uint8_t c = p[i];
        /* RFC 8259: unescaped control characters U+0000..U+001F are invalid */
        if (c < 0x20U) {
            cs_destroy(&tmp);
            return CS_ESYNTAX;
        }
        if (c != '\\') {
            st = cs_append_byte(&tmp, c);
            if (st != CS_OK) { cs_destroy(&tmp); return st; }
            i++;
            continue;
        }
        i++;
        if (i >= n) { cs_destroy(&tmp); return CS_ESYNTAX; }
        uint8_t esc = p[i++];
        switch (esc) {
            case '\"': st = cs_append_byte(&tmp, '\"'); break;
            case '\\': st = cs_append_byte(&tmp, '\\'); break;
            case '/':  st = cs_append_byte(&tmp, '/');  break;
            case 'b':  st = cs_append_byte(&tmp, '\b'); break;
            case 'f':  st = cs_append_byte(&tmp, '\f'); break;
            case 'n':  st = cs_append_byte(&tmp, '\n'); break;
            case 'r':  st = cs_append_byte(&tmp, '\r'); break;
            case 't':  st = cs_append_byte(&tmp, '\t'); break;
            case 'u': {
                if (i + 4U > n) { cs_destroy(&tmp); return CS_ESYNTAX; }
                uint16_t u1 = 0;
                if (!cs_parse_hex4(p + i, &u1)) { cs_destroy(&tmp); return CS_ESYNTAX; }
                i += 4U;
                uint32_t cp = u1;
                if (u1 >= 0xD800U && u1 <= 0xDBFFU) {
                    if (i + 6U > n || p[i] != '\\' || p[i + 1U] != 'u') {
                        cs_destroy(&tmp);
                        return CS_ESYNTAX;
                    }
                    uint16_t u2 = 0;
                    if (!cs_parse_hex4(p + i + 2U, &u2) || u2 < 0xDC00U || u2 > 0xDFFFU) {
                        cs_destroy(&tmp);
                        return CS_ESYNTAX;
                    }
                    i += 6U;
                    cp = 0x10000U + (((uint32_t)(u1 - 0xD800U) << 10) | (uint32_t)(u2 - 0xDC00U));
                } else if (u1 >= 0xDC00U && u1 <= 0xDFFFU) {
                    cs_destroy(&tmp);
                    return CS_ESYNTAX;
                }
                st = cs_append_codepoint(&tmp, cp);
                break;
            }
            default:
                cs_destroy(&tmp);
                return CS_ESYNTAX;
        }
        if (st != CS_OK) { cs_destroy(&tmp); return st; }
    }
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
#endif
}

/* 5. URL Percent Encode / Decode (RFC 3986 + form-urlencoded) */
static inline bool cs_url_is_unreserved(uint8_t b) {
    return (b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z') ||
           (b >= '0' && b <= '9') || b == '-' || b == '.' || b == '_' || b == '~';
}

static inline bool cs_url_is_path_safe(uint8_t b) {
    if (cs_url_is_unreserved(b)) return true;
    /* RFC 3986 pchar / path: unreserved / sub-delims / ":" / "@" / "/" */
    switch (b) {
        case '!': case '$': case '&': case '\'': case '(': case ')':
        case '*': case '+': case ',': case ';':  case '=': case ':':
        case '@': case '/':
            return true;
        default:
            return false;
    }
}

cs_status cs_url_encode(cs_string *dst, cs_view data, cs_url_mode mode) {
    if (!dst || !cs_view_is_valid(data)) return CS_EINVAL;
#if !CS_ENABLE_CODECS
    (void)mode;
    return CS_EUNSUPPORTED;
#else
    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;

    const char *hex = "0123456789ABCDEF";
    const uint8_t *p = (const uint8_t *)data.data;
    for (size_t i = 0; i < data.len; i++) {
        uint8_t b = p[i];
        if (mode == CS_URL_FORM_URLENCODED && b == ' ') {
            st = cs_append_byte(&tmp, '+');
        } else if ((mode == CS_URL_PATH && cs_url_is_path_safe(b)) ||
                   (mode != CS_URL_PATH && cs_url_is_unreserved(b))) {
            st = cs_append_byte(&tmp, b);
        } else {
            char pct[3] = { '%', hex[b >> 4], hex[b & 0x0FU] };
            st = cs_append(&tmp, (cs_view){ pct, 3U });
        }
        if (st != CS_OK) { cs_destroy(&tmp); return st; }
    }
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
#endif
}

cs_status cs_url_decode(cs_string *dst, cs_view encoded, cs_url_mode mode) {
    if (!dst || !cs_view_is_valid(encoded)) return CS_EINVAL;
#if !CS_ENABLE_CODECS
    (void)mode;
    return CS_EUNSUPPORTED;
#else
    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;

    const uint8_t *p = (const uint8_t *)encoded.data;
    size_t i = 0;
    while (i < encoded.len) {
        uint8_t b = p[i];
        if (b == '%') {
            if (i + 3U > encoded.len) { cs_destroy(&tmp); return CS_ESYNTAX; }
            int hi = cs_hex_nibble(p[i + 1U]);
            int lo = cs_hex_nibble(p[i + 2U]);
            if (hi < 0 || lo < 0) { cs_destroy(&tmp); return CS_ESYNTAX; }
            st = cs_append_byte(&tmp, (uint8_t)((hi << 4) | lo));
            i += 3U;
        } else if (b == '+' && mode == CS_URL_FORM_URLENCODED) {
            st = cs_append_byte(&tmp, ' ');
            i++;
        } else {
            st = cs_append_byte(&tmp, b);
            i++;
        }
        if (st != CS_OK) { cs_destroy(&tmp); return st; }
    }
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
#endif
}

/* 6. HTML Escape / Unescape (5 basic entities: & < > " ') */
cs_status cs_html_escape(cs_string *dst, cs_view text) {
    if (!dst || !cs_view_is_valid(text)) return CS_EINVAL;
#if !CS_ENABLE_CODECS
    return CS_EUNSUPPORTED;
#else
    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;

    for (size_t i = 0; i < text.len; i++) {
        char c = text.data[i];
        switch (c) {
            case '&':  st = cs_append(&tmp, CS_LIT("&amp;"));  break;
            case '<':  st = cs_append(&tmp, CS_LIT("&lt;"));   break;
            case '>':  st = cs_append(&tmp, CS_LIT("&gt;"));   break;
            case '\"': st = cs_append(&tmp, CS_LIT("&quot;")); break;
            case '\'': st = cs_append(&tmp, CS_LIT("&#39;"));  break;
            default:   st = cs_append_byte(&tmp, (uint8_t)c);  break;
        }
        if (st != CS_OK) { cs_destroy(&tmp); return st; }
    }
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
#endif
}

cs_status cs_html_unescape(cs_string *dst, cs_view escaped) {
    if (!dst || !cs_view_is_valid(escaped)) return CS_EINVAL;
#if !CS_ENABLE_CODECS
    return CS_EUNSUPPORTED;
#else
    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;

    size_t i = 0;
    while (i < escaped.len) {
        if (escaped.data[i] == '&') {
            cs_view rem = { escaped.data + i, escaped.len - i };
            if (cs_starts_with(rem, CS_LIT("&amp;")))       { st = cs_append_byte(&tmp, '&');  i += 5U; }
            else if (cs_starts_with(rem, CS_LIT("&lt;")))   { st = cs_append_byte(&tmp, '<');  i += 4U; }
            else if (cs_starts_with(rem, CS_LIT("&gt;")))   { st = cs_append_byte(&tmp, '>');  i += 4U; }
            else if (cs_starts_with(rem, CS_LIT("&quot;"))) { st = cs_append_byte(&tmp, '\"'); i += 6U; }
            else if (cs_starts_with(rem, CS_LIT("&#39;")))  { st = cs_append_byte(&tmp, '\''); i += 5U; }
            else if (cs_starts_with(rem, CS_LIT("&#x27;"))) { st = cs_append_byte(&tmp, '\''); i += 6U; }
            else if (cs_starts_with(rem, CS_LIT("&apos;"))) { st = cs_append_byte(&tmp, '\''); i += 6U; }
            else { st = cs_append_byte(&tmp, '&'); i++; }
        } else {
            st = cs_append_byte(&tmp, (uint8_t)escaped.data[i++]);
        }
        if (st != CS_OK) { cs_destroy(&tmp); return st; }
    }
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
#endif
}

/* 7. C String Escape / Unescape */
cs_status cs_c_escape(cs_string *dst, cs_view data) {
    if (!dst || !cs_view_is_valid(data)) return CS_EINVAL;
#if !CS_ENABLE_CODECS
    return CS_EUNSUPPORTED;
#else
    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;

    const char *hex = "0123456789abcdef";
    const uint8_t *p = (const uint8_t *)data.data;
    for (size_t i = 0; i < data.len; i++) {
        uint8_t b = p[i];
        switch (b) {
            case '\a': st = cs_append(&tmp, CS_LIT("\\a"));  break;
            case '\b': st = cs_append(&tmp, CS_LIT("\\b"));  break;
            case '\f': st = cs_append(&tmp, CS_LIT("\\f"));  break;
            case '\n': st = cs_append(&tmp, CS_LIT("\\n"));  break;
            case '\r': st = cs_append(&tmp, CS_LIT("\\r"));  break;
            case '\t': st = cs_append(&tmp, CS_LIT("\\t"));  break;
            case '\v': st = cs_append(&tmp, CS_LIT("\\v"));  break;
            case '\\': st = cs_append(&tmp, CS_LIT("\\\\")); break;
            case '\"': st = cs_append(&tmp, CS_LIT("\\\"")); break;
            case '\'': st = cs_append(&tmp, CS_LIT("\\\'")); break;
            default:
                if (b >= 0x20U && b <= 0x7EU) {
                    st = cs_append_byte(&tmp, b);
                } else {
                    /* Use octal if followed by a hex digit to avoid C '\x' greedy hex ambiguity, else \xHH */
                    bool next_is_hex = (i + 1U < data.len && cs_hex_nibble(p[i + 1U]) >= 0);
                    if (next_is_hex) {
                        char oct[4] = { '\\', (char)('0' + ((b >> 6) & 7U)), (char)('0' + ((b >> 3) & 7U)), (char)('0' + (b & 7U)) };
                        st = cs_append(&tmp, (cs_view){ oct, 4U });
                    } else {
                        char hx[4] = { '\\', 'x', hex[b >> 4], hex[b & 0x0FU] };
                        st = cs_append(&tmp, (cs_view){ hx, 4U });
                    }
                }
                break;
        }
        if (st != CS_OK) { cs_destroy(&tmp); return st; }
    }
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
#endif
}

cs_status cs_c_unescape(cs_string *dst, cs_view escaped) {
    if (!dst || !cs_view_is_valid(escaped)) return CS_EINVAL;
#if !CS_ENABLE_CODECS
    return CS_EUNSUPPORTED;
#else
    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;

    const uint8_t *p = (const uint8_t *)escaped.data;
    size_t n = escaped.len;
    size_t i = 0;

    while (i < n) {
        uint8_t c = p[i++];
        if (c != '\\') {
            st = cs_append_byte(&tmp, c);
            if (st != CS_OK) { cs_destroy(&tmp); return st; }
            continue;
        }
        if (i >= n) { cs_destroy(&tmp); return CS_ESYNTAX; }
        uint8_t e = p[i++];
        switch (e) {
            case 'a':  st = cs_append_byte(&tmp, '\a'); break;
            case 'b':  st = cs_append_byte(&tmp, '\b'); break;
            case 'f':  st = cs_append_byte(&tmp, '\f'); break;
            case 'n':  st = cs_append_byte(&tmp, '\n'); break;
            case 'r':  st = cs_append_byte(&tmp, '\r'); break;
            case 't':  st = cs_append_byte(&tmp, '\t'); break;
            case 'v':  st = cs_append_byte(&tmp, '\v'); break;
            case '\\': st = cs_append_byte(&tmp, '\\'); break;
            case '\"': st = cs_append_byte(&tmp, '\"'); break;
            case '\'': st = cs_append_byte(&tmp, '\''); break;
            case '?':  st = cs_append_byte(&tmp, '?');  break;
            case 'x': {
                if (i + 2U > n) { cs_destroy(&tmp); return CS_ESYNTAX; }
                int hi = cs_hex_nibble(p[i]);
                int lo = cs_hex_nibble(p[i + 1U]);
                if (hi < 0 || lo < 0) { cs_destroy(&tmp); return CS_ESYNTAX; }
                i += 2U;
                st = cs_append_byte(&tmp, (uint8_t)((hi << 4) | lo));
                break;
            }
            case 'u': {
                if (i + 4U > n) { cs_destroy(&tmp); return CS_ESYNTAX; }
                uint16_t u16 = 0;
                if (!cs_parse_hex4(p + i, &u16)) { cs_destroy(&tmp); return CS_ESYNTAX; }
                i += 4U;
                st = cs_append_codepoint(&tmp, (uint32_t)u16);
                break;
            }
            case 'U': {
                if (i + 8U > n) { cs_destroy(&tmp); return CS_ESYNTAX; }
                uint16_t hi16 = 0, lo16 = 0;
                if (!cs_parse_hex4(p + i, &hi16) || !cs_parse_hex4(p + i + 4U, &lo16)) {
                    cs_destroy(&tmp);
                    return CS_ESYNTAX;
                }
                i += 8U;
                uint32_t cp = ((uint32_t)hi16 << 16) | (uint32_t)lo16;
                st = cs_append_codepoint(&tmp, cp);
                break;
            }
            default:
                if (e >= '0' && e <= '7') {
                    uint32_t val = (uint32_t)(e - '0');
                    for (int k = 0; k < 2 && i < n && p[i] >= '0' && p[i] <= '7'; k++) {
                        val = (val << 3) | (uint32_t)(p[i++] - '0');
                    }
                    if (val > 0xFFU) { cs_destroy(&tmp); return CS_ESYNTAX; }
                    st = cs_append_byte(&tmp, (uint8_t)val);
                } else {
                    cs_destroy(&tmp);
                    return CS_ESYNTAX;
                }
                break;
        }
        if (st != CS_OK) { cs_destroy(&tmp); return st; }
    }
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
#endif
}

/* ==========================================================================
 * Section 13: Wildcard / Glob & Bounded PikeVM / Thompson NFA Regex
 * ========================================================================== */

static inline uint32_t cs_norm_cp_for_match(uint32_t cp, cs_match_mode mode) {
    if (mode == CS_MATCH_ASCII_CASE_INSENSITIVE) {
        return (cp >= 'A' && cp <= 'Z') ? (cp + 32U) : cp;
    }
    if (mode == CS_MATCH_UNICODE_CASEFOLD) {
        return cs_fold_cp_simple(cp);
    }
    return cp;
}

static cs_status cs_glob_match_class(cs_view pat, size_t *io_pos, uint32_t text_cp, cs_match_mode mode, bool *out_ok) {
    size_t pos = *io_pos;
    bool negated = false;
    if (pos < pat.len && (pat.data[pos] == '!' || pat.data[pos] == '^')) {
        negated = true;
        pos++;
    }
    bool matched = false;
    bool has_items = false;
    uint32_t tcp = cs_norm_cp_for_match(text_cp, mode);

    while (pos < pat.len) {
        if (pat.data[pos] == ']' && has_items) {
            *io_pos = pos + 1U;
            *out_ok = negated ? !matched : matched;
            return CS_OK;
        }
        has_items = true;
        uint32_t c1 = 0;
        size_t next_p = 0;
        if (pat.data[pos] == '\\') {
            pos++;
            if (pos >= pat.len) return CS_ESYNTAX;
        }
        cs_status st = cs_decode_utf8_next(pat, pos, &c1, &next_p);
        if (st != CS_OK) return st;
        pos = next_p;

        if (pos + 1U < pat.len && pat.data[pos] == '-' && pat.data[pos + 1U] != ']') {
            pos++; /* skip '-' */
            if (pat.data[pos] == '\\') {
                pos++;
                if (pos >= pat.len) return CS_ESYNTAX;
            }
            uint32_t c2 = 0;
            st = cs_decode_utf8_next(pat, pos, &c2, &next_p);
            if (st != CS_OK) return st;
            pos = next_p;
            uint32_t n1 = cs_norm_cp_for_match(c1, mode);
            uint32_t n2 = cs_norm_cp_for_match(c2, mode);
            if (n1 > n2) { uint32_t t = n1; n1 = n2; n2 = t; }
            if (tcp >= n1 && tcp <= n2) matched = true;
        } else {
            if (tcp == cs_norm_cp_for_match(c1, mode)) matched = true;
        }
    }
    return CS_ESYNTAX; /* Unclosed '[' */
}

/* Non-catastrophic O(n) / single-star backtracking glob matcher */
cs_status cs_glob_match(cs_view pattern, cs_view text, cs_match_mode mode, bool *out_matched) {
    if (!cs_view_is_valid(pattern) || !cs_view_is_valid(text) || !out_matched) return CS_EINVAL;
    cs_status st = cs_validate_utf8(pattern, NULL);
    if (st != CS_OK) return st;
    st = cs_validate_utf8(text, NULL);
    if (st != CS_OK) return st;

    /* Pre-validate any '[' character classes in pattern so syntax errors are caught even if not reached */
    for (size_t k = 0; k < pattern.len; ) {
        if (pattern.data[k] == '\\') {
            k++;
            if (k >= pattern.len) return CS_ESYNTAX;
            uint32_t d = 0;
            cs_decode_utf8_next(pattern, k, &d, &k);
        } else if (pattern.data[k] == '[') {
            k++;
            bool ok = false;
            st = cs_glob_match_class(pattern, &k, 0, mode, &ok);
            if (st != CS_OK) return st;
        } else {
            uint32_t d = 0;
            cs_decode_utf8_next(pattern, k, &d, &k);
        }
    }

    size_t p_pos = 0, t_pos = 0;
    size_t star_p = SIZE_MAX, star_t = 0;

    while (t_pos < text.len) {
        uint32_t tcp = 0;
        size_t next_t = 0;
        cs_decode_utf8_next(text, t_pos, &tcp, &next_t);

        if (p_pos < pattern.len && pattern.data[p_pos] == '*') {
            while (p_pos < pattern.len && pattern.data[p_pos] == '*') p_pos++;
            star_p = p_pos;
            star_t = t_pos;
            continue;
        }

        bool step_ok = false;
        size_t next_p = p_pos;
        if (p_pos < pattern.len) {
            char pc = pattern.data[p_pos];
            if (pc == '?') {
                step_ok = true;
                next_p = p_pos + 1U;
            } else if (pc == '[') {
                next_p = p_pos + 1U;
                cs_glob_match_class(pattern, &next_p, tcp, mode, &step_ok);
            } else {
                if (pc == '\\') next_p++;
                uint32_t pcp = 0;
                cs_decode_utf8_next(pattern, next_p, &pcp, &next_p);
                step_ok = (cs_norm_cp_for_match(pcp, mode) == cs_norm_cp_for_match(tcp, mode));
            }
        }

        if (step_ok) {
            p_pos = next_p;
            t_pos = next_t;
        } else if (star_p != SIZE_MAX) {
            p_pos = star_p;
            uint32_t d = 0;
            cs_decode_utf8_next(text, star_t, &d, &star_t);
            t_pos = star_t;
        } else {
            *out_matched = false;
            return CS_OK;
        }
    }

    while (p_pos < pattern.len && pattern.data[p_pos] == '*') p_pos++;
    *out_matched = (p_pos == pattern.len);
    return CS_OK;
}

/* ==========================================================================
 * Bounded Thompson NFA / PikeVM Regex Engine (Section 13)
 * ========================================================================== */

enum {
    CS_RE_OP_CHAR          = 1, /* Match literal code point `cp` */
    CS_RE_OP_ANY           = 2, /* Match '.' */
    CS_RE_OP_CLASS         = 3, /* Match character class [ranges + builtin mask] */
    CS_RE_OP_BUILTIN_CLASS = 4, /* Match \d, \D, \w, \W, \s, \S */
    CS_RE_OP_BOL           = 5, /* ^ */
    CS_RE_OP_EOL           = 6, /* $ */
    CS_RE_OP_SAVE          = 7, /* Save current byte offset into slot `arg` */
    CS_RE_OP_JMP           = 8, /* Unconditional jump to `x` */
    CS_RE_OP_SPLIT         = 9, /* Priority fork to `x` (first priority) and `y` (second priority) */
    CS_RE_OP_MATCH         = 10 /* Accept state */
};

struct cs_regex_inst {
    uint8_t  op;
    uint8_t  negated;
    uint16_t arg;
    uint32_t cp;
    uint32_t x;
    uint32_t y;
};

cs_regex_limits cs_default_regex_limits(void) {
    cs_regex_limits lim;
    lim.max_instructions = 1024U;
    lim.max_repetition   = 256U;
    lim.max_captures     = CS_REGEX_MAX_CAPTURES_LIMIT;
    lim.max_input_bytes  = 16ULL * 1024ULL * 1024ULL;
    return lim;
}

#if CS_ENABLE_REGEX
typedef struct {
    cs_regex *re;
    cs_view   pat;
    size_t    pos;
    cs_status err;
} cs_re_compiler;

static uint32_t cs_re_emit(cs_re_compiler *c, uint8_t op, uint32_t cp, uint32_t x, uint32_t y, uint16_t arg, uint8_t negated) {
    if (c->err != CS_OK) return 0;
    if (c->re->prog_len >= c->re->limits.max_instructions) {
        c->err = CS_ELIMIT;
        return 0;
    }
    if (c->re->prog_len >= c->re->prog_cap) {
        size_t new_cap = (c->re->prog_cap == 0) ? 64U : (c->re->prog_cap * 2U);
        if (new_cap > c->re->limits.max_instructions) new_cap = c->re->limits.max_instructions;
        cs_regex_inst *np = (cs_regex_inst *)c->re->alloc.alloc(c->re->alloc.ctx,
                                                                new_cap * sizeof(cs_regex_inst),
                                                                CS_ALIGNOF(cs_regex_inst));
        if (!np) { c->err = CS_ENOMEM; return 0; }
        if (c->re->prog) {
            memcpy(np, c->re->prog, c->re->prog_len * sizeof(cs_regex_inst));
            c->re->alloc.free(c->re->alloc.ctx, c->re->prog, c->re->prog_cap * sizeof(cs_regex_inst), CS_ALIGNOF(cs_regex_inst));
        }
        c->re->prog     = np;
        c->re->prog_cap = new_cap;
    }
    uint32_t idx = (uint32_t)c->re->prog_len++;
    cs_regex_inst *inst = &c->re->prog[idx];
    inst->op      = op;
    inst->negated = negated;
    inst->arg     = arg;
    inst->cp      = cp;
    inst->x       = x;
    inst->y       = y;
    return idx;
}

static void cs_re_insert_at(cs_re_compiler *c, uint32_t at_pc, size_t count) {
    if (c->err != CS_OK || count == 0) return;
    if (c->re->prog_len + count > c->re->limits.max_instructions) {
        c->err = CS_ELIMIT;
        return;
    }
    while (c->re->prog_len + count > c->re->prog_cap) {
        size_t new_cap = (c->re->prog_cap == 0) ? 64U : (c->re->prog_cap * 2U);
        cs_regex_inst *np = (cs_regex_inst *)c->re->alloc.alloc(c->re->alloc.ctx,
                                                                new_cap * sizeof(cs_regex_inst),
                                                                CS_ALIGNOF(cs_regex_inst));
        if (!np) { c->err = CS_ENOMEM; return; }
        if (c->re->prog) {
            memcpy(np, c->re->prog, c->re->prog_len * sizeof(cs_regex_inst));
            c->re->alloc.free(c->re->alloc.ctx, c->re->prog, c->re->prog_cap * sizeof(cs_regex_inst), CS_ALIGNOF(cs_regex_inst));
        }
        c->re->prog     = np;
        c->re->prog_cap = new_cap;
    }

    /* Shift instructions [at_pc .. prog_len - 1] right by `count` and fix internal jump targets >= at_pc */
    for (size_t i = c->re->prog_len; i > at_pc; i--) {
        cs_regex_inst inst = c->re->prog[i - 1U];
        if (inst.op == CS_RE_OP_JMP || inst.op == CS_RE_OP_SPLIT) {
            if (inst.x >= at_pc) inst.x += (uint32_t)count;
            if (inst.op == CS_RE_OP_SPLIT && inst.y >= at_pc) inst.y += (uint32_t)count;
        }
        c->re->prog[i - 1U + count] = inst;
    }
    c->re->prog_len += count;
}

static void cs_re_clone_subprog(cs_re_compiler *c, uint32_t src_start, uint32_t src_end) {
    if (c->err != CS_OK) return;
    uint32_t len = src_end - src_start;
    uint32_t dst_base = (uint32_t)c->re->prog_len;
    for (uint32_t i = 0; i < len; i++) {
        cs_regex_inst inst = c->re->prog[src_start + i];
        if (inst.op == CS_RE_OP_JMP || inst.op == CS_RE_OP_SPLIT) {
            if (inst.x >= src_start && inst.x <= src_end) inst.x = dst_base + (inst.x - src_start);
            if (inst.op == CS_RE_OP_SPLIT && inst.y >= src_start && inst.y <= src_end) {
                inst.y = dst_base + (inst.y - src_start);
            }
        }
        cs_re_emit(c, inst.op, inst.cp, inst.x, inst.y, inst.arg, inst.negated);
        if (c->err != CS_OK) return;
    }
}

static void cs_re_push_class_pair(cs_re_compiler *c, uint32_t lo, uint32_t hi) {
    if (c->err != CS_OK) return;
    if (c->re->class_pool_len + 2U > c->re->class_pool_cap) {
        size_t new_cap = (c->re->class_pool_cap == 0) ? 64U : (c->re->class_pool_cap * 2U);
        uint32_t *np = (uint32_t *)c->re->alloc.alloc(c->re->alloc.ctx, new_cap * sizeof(uint32_t), CS_ALIGNOF(uint32_t));
        if (!np) { c->err = CS_ENOMEM; return; }
        if (c->re->class_pool) {
            memcpy(np, c->re->class_pool, c->re->class_pool_len * sizeof(uint32_t));
            c->re->alloc.free(c->re->alloc.ctx, c->re->class_pool, c->re->class_pool_cap * sizeof(uint32_t), CS_ALIGNOF(uint32_t));
        }
        c->re->class_pool     = np;
        c->re->class_pool_cap = new_cap;
    }
    c->re->class_pool[c->re->class_pool_len++] = lo;
    c->re->class_pool[c->re->class_pool_len++] = hi;
}

static void cs_re_parse_alt(cs_re_compiler *c);

static uint32_t cs_re_parse_escape_cp(cs_re_compiler *c, uint8_t *out_builtin_kind) {
    *out_builtin_kind = 0;
    if (c->pos >= c->pat.len) { c->err = CS_ESYNTAX; return 0; }
    uint32_t cp = 0;
    size_t next_p = 0;
    cs_decode_utf8_next(c->pat, c->pos, &cp, &next_p);
    c->pos = next_p;

    switch (cp) {
        case 'd': *out_builtin_kind = 'd'; return 0;
        case 'D': *out_builtin_kind = 'D'; return 0;
        case 'w': *out_builtin_kind = 'w'; return 0;
        case 'W': *out_builtin_kind = 'W'; return 0;
        case 's': *out_builtin_kind = 's'; return 0;
        case 'S': *out_builtin_kind = 'S'; return 0;
        case 'n': return '\n';
        case 'r': return '\r';
        case 't': return '\t';
        case 'f': return '\f';
        case 'v': return '\v';
        case '\\': case '.': case '*': case '+': case '?':
        case '(': case ')': case '[': case ']': case '{': case '}':
        case '^': case '$': case '|': case '/': case '-':
            return cp;
        default:
            /* Reject backreferences \1..\9 and unsupported escapes */
            c->err = CS_ESYNTAX;
            return 0;
    }
}

static void cs_re_parse_char_class(cs_re_compiler *c) {
    bool negated = false;
    if (c->pos < c->pat.len && c->pat.data[c->pos] == '^') {
        negated = true;
        c->pos++;
    }
    uint32_t pool_start = (uint32_t)c->re->class_pool_len;
    uint16_t builtin_mask = 0;
    bool has_items = false;

    while (c->pos < c->pat.len) {
        if (c->pat.data[c->pos] == ']' && has_items) {
            c->pos++;
            uint32_t pair_count = (uint32_t)(c->re->class_pool_len - pool_start) / 2U;
            cs_re_emit(c, CS_RE_OP_CLASS, 0, pool_start, pair_count, builtin_mask, negated ? 1U : 0U);
            return;
        }
        has_items = true;
        uint8_t bkind = 0;
        uint32_t c1 = 0;
        if (c->pat.data[c->pos] == '\\') {
            c->pos++;
            c1 = cs_re_parse_escape_cp(c, &bkind);
            if (c->err != CS_OK) return;
            if (bkind != 0) {
                if (bkind == 'd') builtin_mask |= (1U << 0);
                else if (bkind == 'D') builtin_mask |= (1U << 1);
                else if (bkind == 'w') builtin_mask |= (1U << 2);
                else if (bkind == 'W') builtin_mask |= (1U << 3);
                else if (bkind == 's') builtin_mask |= (1U << 4);
                else if (bkind == 'S') builtin_mask |= (1U << 5);
                continue;
            }
        } else {
            size_t np = 0;
            cs_decode_utf8_next(c->pat, c->pos, &c1, &np);
            c->pos = np;
        }

        if (c->pos + 1U < c->pat.len && c->pat.data[c->pos] == '-' && c->pat.data[c->pos + 1U] != ']') {
            c->pos++; /* skip '-' */
            uint32_t c2 = 0;
            uint8_t bkind2 = 0;
            if (c->pat.data[c->pos] == '\\') {
                c->pos++;
                c2 = cs_re_parse_escape_cp(c, &bkind2);
                if (c->err != CS_OK || bkind2 != 0) { c->err = CS_ESYNTAX; return; }
            } else {
                size_t np = 0;
                cs_decode_utf8_next(c->pat, c->pos, &c2, &np);
                c->pos = np;
            }
            if (c1 > c2) { c->err = CS_ESYNTAX; return; }
            if (c->re->flags & CS_REGEX_CASE_INSENSITIVE) {
                for (uint32_t ch = c1; ch <= c2; ch++) {
                    uint32_t fc = cs_fold_cp_simple(ch);
                    cs_re_push_class_pair(c, fc, fc);
                }
            } else {
                cs_re_push_class_pair(c, c1, c2);
            }
        } else {
            uint32_t n1 = (c->re->flags & CS_REGEX_CASE_INSENSITIVE) ? cs_fold_cp_simple(c1) : c1;
            cs_re_push_class_pair(c, n1, n1);
        }
    }
    c->err = CS_ESYNTAX; /* Unclosed '[' */
}

static void cs_re_parse_atom(cs_re_compiler *c) {
    if (c->err != CS_OK || c->pos >= c->pat.len) return;
    char ch = c->pat.data[c->pos];

    if (ch == '(') {
        c->pos++;
        bool capturing = true;
        size_t cap_idx = 0;
        if (c->pos < c->pat.len && c->pat.data[c->pos] == '?') {
            c->pos++;
            if (c->pos < c->pat.len && c->pat.data[c->pos] == ':') {
                c->pos++;
                capturing = false;
            } else {
                /* Reject lookahead/lookbehind/named/other unsupported (?...) syntax */
                c->err = CS_ESYNTAX;
                return;
            }
        }
        if (capturing) {
            cap_idx = c->re->capture_count++;
            if (cap_idx >= c->re->limits.max_captures || cap_idx >= CS_REGEX_MAX_CAPTURES_LIMIT) {
                c->err = CS_ELIMIT;
                return;
            }
            cs_re_emit(c, CS_RE_OP_SAVE, 0, 0, 0, (uint16_t)(cap_idx * 2U), 0);
        }
        cs_re_parse_alt(c);
        if (c->err != CS_OK) return;
        if (c->pos >= c->pat.len || c->pat.data[c->pos] != ')') {
            c->err = CS_ESYNTAX;
            return;
        }
        c->pos++;
        if (capturing) {
            cs_re_emit(c, CS_RE_OP_SAVE, 0, 0, 0, (uint16_t)(cap_idx * 2U + 1U), 0);
        }
        return;
    }

    if (ch == '[') {
        c->pos++;
        cs_re_parse_char_class(c);
        return;
    }

    if (ch == '.') {
        c->pos++;
        cs_re_emit(c, CS_RE_OP_ANY, 0, 0, 0, 0, 0);
        return;
    }

    if (ch == '^') {
        c->pos++;
        cs_re_emit(c, CS_RE_OP_BOL, 0, 0, 0, 0, 0);
        return;
    }

    if (ch == '$') {
        c->pos++;
        cs_re_emit(c, CS_RE_OP_EOL, 0, 0, 0, 0, 0);
        return;
    }

    if (ch == '\\') {
        c->pos++;
        uint8_t bkind = 0;
        uint32_t cp = cs_re_parse_escape_cp(c, &bkind);
        if (c->err != CS_OK) return;
        if (bkind != 0) {
            cs_re_emit(c, CS_RE_OP_BUILTIN_CLASS, 0, 0, 0, bkind, 0);
        } else {
            if (c->re->flags & CS_REGEX_CASE_INSENSITIVE) cp = cs_fold_cp_simple(cp);
            cs_re_emit(c, CS_RE_OP_CHAR, cp, 0, 0, 0, 0);
        }
        return;
    }

    if (ch == '*' || ch == '+' || ch == '?' || ch == '{' || ch == ')') {
        c->err = CS_ESYNTAX;
        return;
    }

    uint32_t cp = 0;
    size_t np = 0;
    cs_decode_utf8_next(c->pat, c->pos, &cp, &np);
    c->pos = np;
    if (c->re->flags & CS_REGEX_CASE_INSENSITIVE) cp = cs_fold_cp_simple(cp);
    cs_re_emit(c, CS_RE_OP_CHAR, cp, 0, 0, 0, 0);
}

static void cs_re_apply_zero_or_one(cs_re_compiler *c, uint32_t atom_start, bool lazy) {
    cs_re_insert_at(c, atom_start, 1U);
    if (c->err != CS_OK) return;
    uint32_t after = (uint32_t)c->re->prog_len;
    c->re->prog[atom_start].op      = CS_RE_OP_SPLIT;
    c->re->prog[atom_start].x       = lazy ? after : (atom_start + 1U);
    c->re->prog[atom_start].y       = lazy ? (atom_start + 1U) : after;
    c->re->prog[atom_start].cp      = 0;
    c->re->prog[atom_start].arg     = 0;
    c->re->prog[atom_start].negated = 0;
}

static void cs_re_apply_star(cs_re_compiler *c, uint32_t atom_start, bool lazy) {
    cs_re_insert_at(c, atom_start, 1U);
    if (c->err != CS_OK) return;
    uint32_t jmp_pc = cs_re_emit(c, CS_RE_OP_JMP, 0, atom_start, 0, 0, 0);
    if (c->err != CS_OK) return;
    uint32_t after = jmp_pc + 1U;
    c->re->prog[atom_start].op      = CS_RE_OP_SPLIT;
    c->re->prog[atom_start].x       = lazy ? after : (atom_start + 1U);
    c->re->prog[atom_start].y       = lazy ? (atom_start + 1U) : after;
    c->re->prog[atom_start].cp      = 0;
    c->re->prog[atom_start].arg     = 0;
    c->re->prog[atom_start].negated = 0;
}

static void cs_re_apply_plus(cs_re_compiler *c, uint32_t atom_start, bool lazy) {
    uint32_t split_pc = cs_re_emit(c, CS_RE_OP_SPLIT, 0, 0, 0, 0, 0);
    if (c->err != CS_OK) return;
    uint32_t after = split_pc + 1U;
    c->re->prog[split_pc].x = lazy ? after : atom_start;
    c->re->prog[split_pc].y = lazy ? atom_start : after;
}

static void cs_re_parse_concat(cs_re_compiler *c) {
    while (c->err == CS_OK && c->pos < c->pat.len) {
        char ch = c->pat.data[c->pos];
        if (ch == '|' || ch == ')') break;

        uint32_t atom_start = (uint32_t)c->re->prog_len;
        cs_re_parse_atom(c);
        if (c->err != CS_OK) return;
        uint32_t atom_end = (uint32_t)c->re->prog_len;

        if (c->pos < c->pat.len) {
            char q = c->pat.data[c->pos];
            if (q == '*' || q == '+' || q == '?') {
                c->pos++;
                bool lazy = false;
                if (c->pos < c->pat.len && c->pat.data[c->pos] == '?') {
                    lazy = true;
                    c->pos++;
                }
                /* Reject stacked quantifiers like a** */
                if (c->pos < c->pat.len &&
                    (c->pat.data[c->pos] == '*' || c->pat.data[c->pos] == '+' ||
                     c->pat.data[c->pos] == '?' || c->pat.data[c->pos] == '{')) {
                    c->err = CS_ESYNTAX;
                    return;
                }
                if (q == '?') cs_re_apply_zero_or_one(c, atom_start, lazy);
                else if (q == '*') cs_re_apply_star(c, atom_start, lazy);
                else cs_re_apply_plus(c, atom_start, lazy);
            } else if (q == '{') {
                c->pos++;
                if (c->pos >= c->pat.len || c->pat.data[c->pos] < '0' || c->pat.data[c->pos] > '9') {
                    c->err = CS_ESYNTAX;
                    return;
                }
                size_t m_rep = 0;
                while (c->pos < c->pat.len && c->pat.data[c->pos] >= '0' && c->pat.data[c->pos] <= '9') {
                    m_rep = m_rep * 10U + (size_t)(c->pat.data[c->pos] - '0');
                    if (m_rep > c->re->limits.max_repetition) { c->err = CS_ELIMIT; return; }
                    c->pos++;
                }
                size_t n_rep = m_rep;
                bool unbounded = false;
                if (c->pos < c->pat.len && c->pat.data[c->pos] == ',') {
                    c->pos++;
                    if (c->pos < c->pat.len && c->pat.data[c->pos] == '}') {
                        unbounded = true;
                    } else {
                        if (c->pos >= c->pat.len || c->pat.data[c->pos] < '0' || c->pat.data[c->pos] > '9') {
                            c->err = CS_ESYNTAX;
                            return;
                        }
                        n_rep = 0;
                        while (c->pos < c->pat.len && c->pat.data[c->pos] >= '0' && c->pat.data[c->pos] <= '9') {
                            n_rep = n_rep * 10U + (size_t)(c->pat.data[c->pos] - '0');
                            if (n_rep > c->re->limits.max_repetition) { c->err = CS_ELIMIT; return; }
                            c->pos++;
                        }
                    }
                }
                if (c->pos >= c->pat.len || c->pat.data[c->pos] != '}') {
                    c->err = CS_ESYNTAX;
                    return;
                }
                c->pos++;
                if (!unbounded && n_rep < m_rep) {
                    c->err = CS_ESYNTAX;
                    return;
                }
                bool lazy = false;
                if (c->pos < c->pat.len && c->pat.data[c->pos] == '?') {
                    lazy = true;
                    c->pos++;
                }

                if (m_rep == 0 && !unbounded && n_rep == 0) {
                    /* {0,0}: erase atom */
                    c->re->prog_len = atom_start;
                } else {
                    /* Emit m_rep mandatory copies (1st is already at [atom_start..atom_end)) */
                    uint32_t last_copy_start = atom_start;
                    for (size_t k = 1U; k < m_rep; k++) {
                        last_copy_start = (uint32_t)c->re->prog_len;
                        cs_re_clone_subprog(c, atom_start, atom_end);
                        if (c->err != CS_OK) return;
                    }
                    if (unbounded) {
                        if (m_rep == 0) {
                            cs_re_apply_star(c, atom_start, lazy);
                        } else {
                            cs_re_apply_plus(c, last_copy_start, lazy);
                        }
                    } else {
                        size_t opt_copies = (m_rep == 0) ? (n_rep - 1U) : (n_rep - m_rep);
                        if (m_rep == 0) {
                            cs_re_apply_zero_or_one(c, atom_start, lazy);
                        }
                        for (size_t k = 0; k < opt_copies; k++) {
                            uint32_t opt_start = (uint32_t)c->re->prog_len;
                            /* Clone from original atom (`m_rep > 0 ? atom_start : atom_start + 1`) */
                            uint32_t src_s = (m_rep == 0) ? (atom_start + 1U) : atom_start;
                            uint32_t src_e = (m_rep == 0) ? (atom_end + 1U)   : atom_end;
                            cs_re_clone_subprog(c, src_s, src_e);
                            if (c->err != CS_OK) return;
                            cs_re_apply_zero_or_one(c, opt_start, lazy);
                            if (c->err != CS_OK) return;
                        }
                    }
                }
            }
        }
    }
}

static void cs_re_parse_alt(cs_re_compiler *c) {
    uint32_t lhs_start = (uint32_t)c->re->prog_len;
    cs_re_parse_concat(c);
    if (c->err != CS_OK) return;

    while (c->pos < c->pat.len && c->pat.data[c->pos] == '|') {
        c->pos++; /* skip '|' */
        cs_re_insert_at(c, lhs_start, 1U);
        if (c->err != CS_OK) return;
        uint32_t jmp_pc = cs_re_emit(c, CS_RE_OP_JMP, 0, 0, 0, 0, 0);
        if (c->err != CS_OK) return;
        uint32_t rhs_start = (uint32_t)c->re->prog_len;
        cs_re_parse_concat(c);
        if (c->err != CS_OK) return;
        uint32_t after = (uint32_t)c->re->prog_len;

        c->re->prog[lhs_start].op      = CS_RE_OP_SPLIT;
        c->re->prog[lhs_start].x       = lhs_start + 1U;
        c->re->prog[lhs_start].y       = rhs_start;
        c->re->prog[lhs_start].cp      = 0;
        c->re->prog[lhs_start].arg     = 0;
        c->re->prog[lhs_start].negated = 0;
        c->re->prog[jmp_pc].x          = after;
    }
}
#endif /* CS_ENABLE_REGEX */

cs_status cs_regex_compile(cs_regex *re, cs_view pattern, uint32_t flags,
                           const cs_regex_limits *limits, const cs_allocator *alloc) {
    if (!re || !cs_view_is_valid(pattern)) return CS_EINVAL;
    memset(re, 0, sizeof(*re));
#if !CS_ENABLE_REGEX
    (void)flags; (void)limits; (void)alloc;
    return CS_EUNSUPPORTED;
#else
    cs_status st = cs_validate_utf8(pattern, NULL);
    if (st != CS_OK) return st;

    re->alloc  = alloc ? *alloc : cs_default_allocator();
    re->limits = limits ? *limits : cs_default_regex_limits();
    re->flags  = flags;
    re->capture_count = 1U; /* Group 0 is whole match */

    cs_re_compiler comp;
    comp.re  = re;
    comp.pat = pattern;
    comp.pos = 0;
    comp.err = CS_OK;

    /* Preamble for unanchored search: `*?` dot-star before group 0 save:
     * PC 0: SPLIT(3, 1)  (try starting match at current pos first, else advance 1 char)
     * PC 1: ANY (matches any code point including '\n')
     * PC 2: JMP(0)
     * PC 3: SAVE(0)
     * ... body ...
     * SAVE(1)
     * MATCH */
    cs_re_emit(&comp, CS_RE_OP_SPLIT, 0, 3U, 1U, 0, 0);
    cs_re_emit(&comp, CS_RE_OP_ANY,   0, 0,  0,  1U /* unconditional any */, 0);
    cs_re_emit(&comp, CS_RE_OP_JMP,   0, 0U, 0,  0, 0);
    cs_re_emit(&comp, CS_RE_OP_SAVE,  0, 0,  0,  0U, 0);

    cs_re_parse_alt(&comp);
    if (comp.err == CS_OK && comp.pos != pattern.len) {
        comp.err = CS_ESYNTAX;
    }
    if (comp.err == CS_OK) {
        cs_re_emit(&comp, CS_RE_OP_SAVE,  0, 0, 0, 1U, 0);
        cs_re_emit(&comp, CS_RE_OP_MATCH, 0, 0, 0, 0,  0);
    }
    if (comp.err != CS_OK) {
        cs_regex_destroy(re);
        return comp.err;
    }
    return CS_OK;
#endif
}

void cs_regex_destroy(cs_regex *re) {
    if (!re) return;
    if (re->prog && re->alloc.free) {
        re->alloc.free(re->alloc.ctx, re->prog, re->prog_cap * sizeof(cs_regex_inst), CS_ALIGNOF(cs_regex_inst));
    }
    if (re->class_pool && re->alloc.free) {
        re->alloc.free(re->alloc.ctx, re->class_pool, re->class_pool_cap * sizeof(uint32_t), CS_ALIGNOF(uint32_t));
    }
    memset(re, 0, sizeof(*re));
}

#if CS_ENABLE_REGEX
static bool cs_re_match_builtin(uint8_t kind, uint32_t cp, bool unicode_mode) {
    bool res = false;
    switch (kind) {
        case 'd': case 'D':
            res = unicode_mode ? cs_cp_is_digit(cp) : (cp >= '0' && cp <= '9');
            return (kind == 'D') ? !res : res;
        case 'w': case 'W':
            res = unicode_mode ? (cs_cp_is_alnum(cp) || cp == '_')
                               : ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') || (cp >= '0' && cp <= '9') || cp == '_');
            return (kind == 'W') ? !res : res;
        case 's': case 'S':
            res = unicode_mode ? cs_cp_is_whitespace(cp)
                               : (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\v' || cp == '\f');
            return (kind == 'S') ? !res : res;
        default:
            return false;
    }
}

typedef struct {
    uint32_t pc;
    size_t   slots[CS_REGEX_MAX_CAPTURES_LIMIT * 2U];
} cs_pike_thread;

typedef struct {
    cs_pike_thread *threads;
    size_t          count;
} cs_pike_list;

/* Recursively follow epsilon transitions (JMP, SPLIT, SAVE, BOL, EOL) in priority order
 * with O(1) visited[pc] deduplication so each PC is added at most once per step! */
static void cs_pike_add_thread(const cs_regex *re, cs_pike_list *list, uint32_t *visited, uint32_t step_id,
                               uint32_t pc, const size_t *slots, cs_view text, size_t byte_pos) {
    if (visited[pc] == step_id) return;
    visited[pc] = step_id;

    const cs_regex_inst *inst = &re->prog[pc];
    switch (inst->op) {
        case CS_RE_OP_JMP:
            cs_pike_add_thread(re, list, visited, step_id, inst->x, slots, text, byte_pos);
            return;
        case CS_RE_OP_SPLIT:
            cs_pike_add_thread(re, list, visited, step_id, inst->x, slots, text, byte_pos);
            cs_pike_add_thread(re, list, visited, step_id, inst->y, slots, text, byte_pos);
            return;
        case CS_RE_OP_SAVE: {
            size_t next_slots[CS_REGEX_MAX_CAPTURES_LIMIT * 2U];
            memcpy(next_slots, slots, re->capture_count * 2U * sizeof(size_t));
            next_slots[inst->arg] = byte_pos;
            cs_pike_add_thread(re, list, visited, step_id, pc + 1U, next_slots, text, byte_pos);
            return;
        }
        case CS_RE_OP_BOL: {
            bool ok = (byte_pos == 0);
            if (!ok && (re->flags & CS_REGEX_MULTILINE) && byte_pos > 0 && text.data[byte_pos - 1U] == '\n') {
                ok = true;
            }
            if (ok) cs_pike_add_thread(re, list, visited, step_id, pc + 1U, slots, text, byte_pos);
            return;
        }
        case CS_RE_OP_EOL: {
            bool ok = (byte_pos == text.len);
            if (!ok && (re->flags & CS_REGEX_MULTILINE) && byte_pos < text.len &&
                (text.data[byte_pos] == '\n' || (text.data[byte_pos] == '\r' && byte_pos + 1U < text.len && text.data[byte_pos + 1U] == '\n'))) {
                ok = true;
            }
            if (ok) cs_pike_add_thread(re, list, visited, step_id, pc + 1U, slots, text, byte_pos);
            return;
        }
        default:
            list->threads[list->count].pc = pc;
            memcpy(list->threads[list->count].slots, slots, re->capture_count * 2U * sizeof(size_t));
            list->count++;
            return;
    }
}
#endif /* CS_ENABLE_REGEX */

cs_status cs_regex_find(const cs_regex *re, cs_view text, size_t start_offset, cs_regex_match *out_match) {
    if (!re || !re->prog || !cs_view_is_valid(text)) return CS_EINVAL;
#if !CS_ENABLE_REGEX
    (void)start_offset; (void)out_match;
    return CS_EUNSUPPORTED;
#else
    if (start_offset > text.len) return CS_ERANGE;
    if (text.len > re->limits.max_input_bytes) return CS_ELIMIT;

    /* Validate UTF-8 and start_offset boundary alignment */
    size_t dummy_cp_idx = 0;
    cs_status st = cs_offset_to_cp_index(text, start_offset, &dummy_cp_idx);
    if (st != CS_OK) return st;
    st = cs_validate_utf8(text, NULL);
    if (st != CS_OK) return st;

    size_t n_inst = re->prog_len;
    size_t alloc_bytes = n_inst * 2U * sizeof(cs_pike_thread) + n_inst * sizeof(uint32_t);
    void *mem = re->alloc.alloc(re->alloc.ctx, alloc_bytes, CS_ALIGNOF(cs_pike_thread));
    if (!mem) return CS_ENOMEM;

    cs_pike_thread *tbuf1 = (cs_pike_thread *)mem;
    cs_pike_thread *tbuf2 = tbuf1 + n_inst;
    uint32_t *visited = (uint32_t *)(void *)(tbuf2 + n_inst);
    memset(visited, 0, n_inst * sizeof(uint32_t));

    cs_pike_list clist = { tbuf1, 0 };
    cs_pike_list nlist = { tbuf2, 0 };
    uint32_t step_id = 1U;

    size_t init_slots[CS_REGEX_MAX_CAPTURES_LIMIT * 2U];
    for (size_t i = 0; i < CS_REGEX_MAX_CAPTURES_LIMIT * 2U; i++) init_slots[i] = SIZE_MAX;

    cs_pike_add_thread(re, &clist, visited, step_id++, 0U, init_slots, text, start_offset);

    bool matched = false;
    size_t best_slots[CS_REGEX_MAX_CAPTURES_LIMIT * 2U];

    size_t pos = start_offset;
    while (clist.count > 0) {
        uint32_t cp = 0;
        size_t next_pos = pos;
        bool has_cp = false;
        if (pos < text.len) {
            cs_decode_utf8_next(text, pos, &cp, &next_pos);
            has_cp = true;
        }
        uint32_t cmp_cp = (re->flags & CS_REGEX_CASE_INSENSITIVE) ? cs_fold_cp_simple(cp) : cp;
        bool u_classes  = (re->flags & CS_REGEX_UNICODE_CLASSES) != 0;

        nlist.count = 0;
        uint32_t next_step_id = step_id++;

        for (size_t i = 0; i < clist.count; i++) {
            cs_pike_thread *t = &clist.threads[i];
            const cs_regex_inst *inst = &re->prog[t->pc];

            if (inst->op == CS_RE_OP_MATCH) {
                /* Found highest-priority match for the earliest start_pos!
                 * Clear lower-priority threads in clist so only higher-priority threads
                 * can continue extending a greedy match. */
                matched = true;
                memcpy(best_slots, t->slots, re->capture_count * 2U * sizeof(size_t));
                break;
            }

            if (!has_cp) continue;
            /* Do not allow the preamble's dot-star (PC 1) to advance once a match has already been found */
            if (matched && t->pc == 1U) continue;

            bool step_ok = false;
            switch (inst->op) {
                case CS_RE_OP_CHAR:
                    step_ok = (cmp_cp == inst->cp);
                    break;
                case CS_RE_OP_ANY:
                    if (inst->arg == 1U) {
                        step_ok = true; /* Preamble unconditional any */
                    } else {
                        step_ok = (re->flags & CS_REGEX_DOTALL) ? true : (cp != '\n');
                    }
                    break;
                case CS_RE_OP_BUILTIN_CLASS:
                    step_ok = cs_re_match_builtin((uint8_t)inst->arg, cp, u_classes);
                    break;
                case CS_RE_OP_CLASS: {
                    bool in_class = false;
                    uint32_t p_start = inst->x;
                    uint32_t p_cnt   = inst->y;
                    for (uint32_t k = 0; k < p_cnt; k++) {
                        uint32_t lo = re->class_pool[p_start + k * 2U];
                        uint32_t hi = re->class_pool[p_start + k * 2U + 1U];
                        if (cmp_cp >= lo && cmp_cp <= hi) { in_class = true; break; }
                    }
                    if (!in_class && inst->arg != 0) {
                        if ((inst->arg & (1U << 0)) && cs_re_match_builtin('d', cp, u_classes)) in_class = true;
                        if ((inst->arg & (1U << 1)) && cs_re_match_builtin('D', cp, u_classes)) in_class = true;
                        if ((inst->arg & (1U << 2)) && cs_re_match_builtin('w', cp, u_classes)) in_class = true;
                        if ((inst->arg & (1U << 3)) && cs_re_match_builtin('W', cp, u_classes)) in_class = true;
                        if ((inst->arg & (1U << 4)) && cs_re_match_builtin('s', cp, u_classes)) in_class = true;
                        if ((inst->arg & (1U << 5)) && cs_re_match_builtin('S', cp, u_classes)) in_class = true;
                    }
                    step_ok = inst->negated ? !in_class : in_class;
                    break;
                }
                default:
                    break;
            }

            if (step_ok) {
                cs_pike_add_thread(re, &nlist, visited, next_step_id, t->pc + 1U, t->slots, text, next_pos);
            }
        }

        if (!has_cp) break;
        pos = next_pos;
        cs_pike_list swap_tmp = clist;
        clist = nlist;
        nlist = swap_tmp;
    }

    re->alloc.free(re->alloc.ctx, mem, alloc_bytes, CS_ALIGNOF(cs_pike_thread));
    if (!matched) return CS_NOT_FOUND;

    if (out_match) {
        memset(out_match, 0, sizeof(*out_match));
        out_match->group_count = re->capture_count;
        for (size_t g = 0; g < re->capture_count; g++) {
            size_t s = best_slots[g * 2U];
            size_t e = best_slots[g * 2U + 1U];
            if (s != SIZE_MAX && e != SIZE_MAX) {
                out_match->groups[g].start   = s;
                out_match->groups[g].end     = e;
                out_match->groups[g].matched = true;
            }
        }
    }
    return CS_OK;
#endif
}

cs_status cs_regex_find_all(const cs_regex *re, cs_view text, cs_regex_match *out_matches,
                            size_t max_matches, size_t *out_total_count) {
    if (!re || !cs_view_is_valid(text) || !out_total_count) return CS_EINVAL;
    size_t count = 0;
    size_t pos   = 0;
    while (pos <= text.len) {
        cs_regex_match m;
        cs_status st = cs_regex_find(re, text, pos, &m);
        if (st == CS_NOT_FOUND) break;
        if (st != CS_OK) return st;

        if (out_matches && count < max_matches) {
            out_matches[count] = m;
        }
        count++;

        if (m.groups[0].end > m.groups[0].start) {
            pos = m.groups[0].end;
        } else {
            if (pos >= text.len) break;
            uint32_t d = 0;
            cs_decode_utf8_next(text, pos, &d, &pos);
        }
    }
    *out_total_count = count;
    return CS_OK;
}

cs_status cs_regex_split(const cs_regex *re, cs_view text, size_t max_splits,
                         cs_view *out_parts, size_t max_parts, size_t *out_count) {
    if (!re || !cs_view_is_valid(text) || !out_count) return CS_EINVAL;
    size_t count = 0;
    size_t splits_done = 0;
    size_t last_end = 0;
    size_t search_pos = 0;

    while (search_pos <= text.len && splits_done < max_splits) {
        cs_regex_match m;
        cs_status st = cs_regex_find(re, text, search_pos, &m);
        if (st == CS_NOT_FOUND) break;
        if (st != CS_OK) return st;

        /* Skip zero-length match at the very beginning of the current segment */
        if (m.groups[0].start == m.groups[0].end && m.groups[0].start == last_end) {
            if (search_pos >= text.len) break;
            uint32_t d = 0;
            cs_decode_utf8_next(text, search_pos, &d, &search_pos);
            continue;
        }

        if (out_parts && count < max_parts) {
            out_parts[count] = (cs_view){ text.data + last_end, m.groups[0].start - last_end };
        }
        count++;
        splits_done++;
        last_end   = m.groups[0].end;
        search_pos = m.groups[0].end;
    }

    if (out_parts && count < max_parts) {
        out_parts[count] = (cs_view){ text.data ? (text.data + last_end) : NULL, text.len - last_end };
    }
    count++;
    *out_count = count;
    if (out_parts && count > max_parts) return CS_EBUFFER;
    return CS_OK;
}

static cs_status cs_regex_expand_replacement(cs_string *dst, cs_view text, const cs_regex_match *m, cs_view tpl) {
    size_t i = 0;
    while (i < tpl.len) {
        if (tpl.data[i] != '$') {
            cs_status st = cs_append_byte(dst, (uint8_t)tpl.data[i++]);
            if (st != CS_OK) return st;
            continue;
        }
        i++;
        if (i >= tpl.len) return CS_ESYNTAX;
        if (tpl.data[i] == '$') {
            cs_status st = cs_append_byte(dst, '$');
            if (st != CS_OK) return st;
            i++;
            continue;
        }
        size_t gidx = 0;
        if (tpl.data[i] == '{') {
            i++;
            if (i >= tpl.len || tpl.data[i] < '0' || tpl.data[i] > '9') return CS_ESYNTAX;
            while (i < tpl.len && tpl.data[i] >= '0' && tpl.data[i] <= '9') {
                gidx = gidx * 10U + (size_t)(tpl.data[i++] - '0');
            }
            if (i >= tpl.len || tpl.data[i] != '}') return CS_ESYNTAX;
            i++;
        } else if (tpl.data[i] >= '0' && tpl.data[i] <= '9') {
            gidx = (size_t)(tpl.data[i++] - '0');
        } else {
            return CS_ESYNTAX;
        }
        if (gidx >= m->group_count) return CS_ERANGE;
        if (m->groups[gidx].matched) {
            cs_view gspan = { text.data + m->groups[gidx].start, m->groups[gidx].end - m->groups[gidx].start };
            cs_status st = cs_append(dst, gspan);
            if (st != CS_OK) return st;
        }
    }
    return CS_OK;
}

cs_status cs_regex_replace(const cs_regex *re, cs_string *dst, cs_view text,
                           cs_view replacement_template, bool replace_all) {
    if (!re || !dst || !cs_view_is_valid(text) || !cs_view_is_valid(replacement_template)) return CS_EINVAL;
    cs_string tmp;
    cs_status st = cs_init_with_allocator(&tmp, dst->alloc);
    if (st != CS_OK) return st;

    size_t last_end = 0;
    size_t search_pos = 0;
    while (search_pos <= text.len) {
        cs_regex_match m;
        st = cs_regex_find(re, text, search_pos, &m);
        if (st == CS_NOT_FOUND) {
            st = CS_OK;
            break;
        }
        if (st != CS_OK) { cs_destroy(&tmp); return st; }

        if (m.groups[0].start > last_end) {
            cs_view prefix = { text.data + last_end, m.groups[0].start - last_end };
            st = cs_append(&tmp, prefix);
            if (st != CS_OK) { cs_destroy(&tmp); return st; }
        }
        st = cs_regex_expand_replacement(&tmp, text, &m, replacement_template);
        if (st != CS_OK) { cs_destroy(&tmp); return st; }

        last_end = m.groups[0].end;
        if (!replace_all) break;

        if (m.groups[0].end > m.groups[0].start) {
            search_pos = m.groups[0].end;
        } else {
            if (m.groups[0].end >= text.len) break;
            uint32_t d = 0;
            cs_decode_utf8_next(text, m.groups[0].end, &d, &search_pos);
            /* Copy the skipped character so it isn't lost */
            cs_view skipped = { text.data + last_end, search_pos - last_end };
            st = cs_append(&tmp, skipped);
            if (st != CS_OK) { cs_destroy(&tmp); return st; }
            last_end = search_pos;
        }
    }

    if (last_end < text.len) {
        cs_view tail = { text.data + last_end, text.len - last_end };
        st = cs_append(&tmp, tail);
        if (st != CS_OK) { cs_destroy(&tmp); return st; }
    }
    cs_swap(dst, &tmp);
    cs_destroy(&tmp);
    return CS_OK;
}

/* ==========================================================================
 * Section 14: Deterministic xxHash64 & Keyed SipHash-2-4
 * ========================================================================== */

#define CS_XXH_PRIME64_1 0x9E3779B185EBCA87ULL
#define CS_XXH_PRIME64_2 0xC2B2AE3D27D4EB4FULL
#define CS_XXH_PRIME64_3 0x165667B19E3779F9ULL
#define CS_XXH_PRIME64_4 0x85EBCA77C2B2AE63ULL
#define CS_XXH_PRIME64_5 0x27D4EB2F165667C5ULL

static inline uint64_t cs_rotl64(uint64_t x, unsigned int r) {
    return (x << r) | (x >> (64U - r));
}

static inline uint64_t cs_read_le64(const uint8_t *p) {
    return (uint64_t)p[0]
         | ((uint64_t)p[1] << 8)
         | ((uint64_t)p[2] << 16)
         | ((uint64_t)p[3] << 24)
         | ((uint64_t)p[4] << 32)
         | ((uint64_t)p[5] << 40)
         | ((uint64_t)p[6] << 48)
         | ((uint64_t)p[7] << 56);
}

static inline uint32_t cs_read_le32(const uint8_t *p) {
    return (uint32_t)p[0]
         | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
}

static inline uint64_t cs_xxh64_round(uint64_t acc, uint64_t input) {
    acc += input * CS_XXH_PRIME64_2;
    acc  = cs_rotl64(acc, 31);
    acc *= CS_XXH_PRIME64_1;
    return acc;
}

static inline uint64_t cs_xxh64_merge_round(uint64_t acc, uint64_t val) {
    val  = cs_xxh64_round(0, val);
    acc ^= val;
    acc  = acc * CS_XXH_PRIME64_1 + CS_XXH_PRIME64_4;
    return acc;
}

uint64_t cs_hash64_seed(cs_view v, uint64_t seed) {
    const uint8_t *p = (const uint8_t *)v.data;
    size_t len = v.len;
    uint64_t h64 = 0;

    if (len >= 32U) {
        const uint8_t *limit = p + len - 32U;
        uint64_t v1 = seed + CS_XXH_PRIME64_1 + CS_XXH_PRIME64_2;
        uint64_t v2 = seed + CS_XXH_PRIME64_2;
        uint64_t v3 = seed + 0;
        uint64_t v4 = seed - CS_XXH_PRIME64_1;

        do {
            v1 = cs_xxh64_round(v1, cs_read_le64(p));      p += 8;
            v2 = cs_xxh64_round(v2, cs_read_le64(p));      p += 8;
            v3 = cs_xxh64_round(v3, cs_read_le64(p));      p += 8;
            v4 = cs_xxh64_round(v4, cs_read_le64(p));      p += 8;
        } while (p <= limit);

        h64 = cs_rotl64(v1, 1) + cs_rotl64(v2, 7) + cs_rotl64(v3, 12) + cs_rotl64(v4, 18);
        h64 = cs_xxh64_merge_round(h64, v1);
        h64 = cs_xxh64_merge_round(h64, v2);
        h64 = cs_xxh64_merge_round(h64, v3);
        h64 = cs_xxh64_merge_round(h64, v4);
    } else {
        h64 = seed + CS_XXH_PRIME64_5;
    }

    h64 += (uint64_t)len;
    const uint8_t *end = (const uint8_t *)v.data + len;
    while (p + 8 <= end) {
        uint64_t k1 = cs_xxh64_round(0, cs_read_le64(p));
        h64 ^= k1;
        h64  = cs_rotl64(h64, 27) * CS_XXH_PRIME64_1 + CS_XXH_PRIME64_4;
        p += 8;
    }
    if (p + 4 <= end) {
        h64 ^= (uint64_t)cs_read_le32(p) * CS_XXH_PRIME64_1;
        h64  = cs_rotl64(h64, 23) * CS_XXH_PRIME64_2 + CS_XXH_PRIME64_3;
        p += 4;
    }
    while (p < end) {
        h64 ^= (uint64_t)(*p) * CS_XXH_PRIME64_5;
        h64  = cs_rotl64(h64, 11) * CS_XXH_PRIME64_1;
        p++;
    }

    h64 ^= h64 >> 33;
    h64 *= CS_XXH_PRIME64_2;
    h64 ^= h64 >> 29;
    h64 *= CS_XXH_PRIME64_3;
    h64 ^= h64 >> 32;
    return h64;
}

uint64_t cs_hash64(cs_view v) {
    return cs_hash64_seed(v, 0ULL);
}

/* Keyed SipHash-2-4 (RFC / Aumasson & Bernstein) */
#define CS_SIPROUND()            \
    do {                         \
        v0 += v1;                \
        v1 = cs_rotl64(v1, 13);  \
        v1 ^= v0;                \
        v0 = cs_rotl64(v0, 32);  \
        v2 += v3;                \
        v3 = cs_rotl64(v3, 16);  \
        v3 ^= v2;                \
        v0 += v3;                \
        v3 = cs_rotl64(v3, 21);  \
        v3 ^= v0;                \
        v2 += v1;                \
        v1 = cs_rotl64(v1, 17);  \
        v1 ^= v2;                \
        v2 = cs_rotl64(v2, 32);  \
    } while (0)

uint64_t cs_hash_keyed(cs_view v, const uint8_t key[16]) {
    if (!key) return 0;
    uint64_t k0 = cs_read_le64(key);
    uint64_t k1 = cs_read_le64(key + 8);

    uint64_t v0 = 0x736f6d6570736575ULL ^ k0;
    uint64_t v1 = 0x646f72616e646f6dULL ^ k1;
    uint64_t v2 = 0x6c7967656e657261ULL ^ k0;
    uint64_t v3 = 0x7465646279746573ULL ^ k1;

    const uint8_t *in = (const uint8_t *)v.data;
    size_t inlen = v.len;
    const uint8_t *end = in + (inlen - (inlen & 7U));
    uint64_t b = ((uint64_t)inlen) << 56;

    for (; in != end; in += 8) {
        uint64_t m = cs_read_le64(in);
        v3 ^= m;
        CS_SIPROUND();
        CS_SIPROUND();
        v0 ^= m;
    }

    switch (inlen & 7U) {
        case 7: b |= ((uint64_t)in[6]) << 48; /* fallthrough */
        case 6: b |= ((uint64_t)in[5]) << 40; /* fallthrough */
        case 5: b |= ((uint64_t)in[4]) << 32; /* fallthrough */
        case 4: b |= ((uint64_t)in[3]) << 24; /* fallthrough */
        case 3: b |= ((uint64_t)in[2]) << 16; /* fallthrough */
        case 2: b |= ((uint64_t)in[1]) << 8;  /* fallthrough */
        case 1: b |= ((uint64_t)in[0]);       break;
        case 0: break;
    }

    v3 ^= b;
    CS_SIPROUND();
    CS_SIPROUND();
    v0 ^= b;

    v2 ^= 0xFFULL;
    CS_SIPROUND();
    CS_SIPROUND();
    CS_SIPROUND();
    CS_SIPROUND();
    return v0 ^ v1 ^ v2 ^ v3;
}

cs_status cs_hash_seed_from_os(uint8_t out_key[16]) {
    if (!out_key) return CS_EINVAL;
#if defined(_WIN32)
    if (BCryptGenRandom(NULL, out_key, 16, BCRYPT_USE_SYSTEM_PREFERRED_RNG) == 0) {
        return CS_OK;
    }
    return CS_EUNSUPPORTED;
#elif defined(__linux__) || defined(__APPLE__) || defined(__unix__)
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd < 0) return CS_EUNSUPPORTED;
    size_t got = 0;
    while (got < 16U) {
        ssize_t r = read(fd, out_key + got, 16U - got);
        if (r <= 0) { close(fd); return CS_EUNSUPPORTED; }
        got += (size_t)r;
    }
    close(fd);
    return CS_OK;
#else
    return CS_EUNSUPPORTED;
#endif
}
