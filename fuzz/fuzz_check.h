#ifndef CSTR_FUZZ_CHECK_H
#define CSTR_FUZZ_CHECK_H

#include "cstr.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Keep checks active in release builds, where assert() may be disabled. */
#define FUZZ_CHECK(condition)                                                                      \
    do {                                                                                           \
        if (!(condition))                                                                          \
            abort();                                                                               \
    } while (0)

static inline void fuzz_check_string(const cs_string *s) {
    FUZZ_CHECK(s->data != NULL);
    FUZZ_CHECK(s->len <= s->cap);
    FUZZ_CHECK(s->data[s->len] == '\0');
}

static inline void fuzz_check_equal(cs_view actual, cs_view expected) {
    FUZZ_CHECK(actual.len == expected.len);
    FUZZ_CHECK(actual.len == 0 || memcmp(actual.data, expected.data, actual.len) == 0);
}

#endif
