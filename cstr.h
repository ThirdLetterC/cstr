/* ==========================================================================
 * cstr.h — Self-Contained String & Unicode Library for ISO C11/C17/C23
 *
 * SPDX-License-Identifier: MIT AND Unicode-3.0
 * Copyright (c) 2026 Yehor Smoliakov
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the Unicode
 * data license notice in cstr.c) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * ========================================================================== */

#ifndef CSTR_H_INCLUDED
#define CSTR_H_INCLUDED

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* ==========================================================================
 * 1. Configuration & Feature Flags (Section 17)
 * ========================================================================== */

#ifndef CS_ENABLE_UNICODE
#define CS_ENABLE_UNICODE 1
#endif

#ifndef CS_ENABLE_REGEX
#define CS_ENABLE_REGEX 1
#endif

#ifndef CS_ENABLE_FORMAT
#define CS_ENABLE_FORMAT 1
#endif

#ifndef CS_ENABLE_CODECS
#define CS_ENABLE_CODECS 1
#endif

#ifndef CS_ENABLE_SIMD
#define CS_ENABLE_SIMD 1
#endif

#ifndef CS_FORCE_SCALAR
#define CS_FORCE_SCALAR 0
#endif

#ifndef CS_NO_HEAP
#define CS_NO_HEAP 0
#endif

#ifndef CS_USE_MIMALLOC
#define CS_USE_MIMALLOC 0
#endif

#define CS_VERSION_MAJOR 1
#define CS_VERSION_MINOR 0
#define CS_VERSION_PATCH 0
#define CS_VERSION_STRING "1.0.0"
#define CS_UNICODE_VERSION "17.0.0"

#if defined(__GNUC__) || defined(__clang__)
#define CS_PRINTF_ATTR(fmt_idx, arg_idx) __attribute__((format(printf, fmt_idx, arg_idx)))
#define CS_NODISCARD __attribute__((warn_unused_result))
#else
#define CS_PRINTF_ATTR(fmt_idx, arg_idx)
#define CS_NODISCARD
#endif

/* ==========================================================================
 * 2. Status Codes & Error Guarantees (Section 4)
 * ========================================================================== */

typedef enum cs_status {
    CS_OK = 0,        /* Success */
    CS_END = 1,       /* Normal end of iterator stream */
    CS_NOT_FOUND = 2, /* Substring/pattern not found (normal search miss) */
    CS_EINVAL =
        -1, /* Invalid argument, NULL with len>0, unaligned UTF-8 offset, empty split needle */
    CS_ENOMEM = -2,    /* Memory allocation failed (destination remains unchanged) */
    CS_ERANGE = -3,    /* Index/slice out of bounds or numeric value out of target range */
    CS_EOVERFLOW = -4, /* size_t / capacity arithmetic overflow */
    CS_EUTF8 = -5,     /* Malformed UTF-8 sequence encountered in strict mode */
    CS_EBUFFER =
        -6, /* Fixed user buffer too small (destination unchanged, required size reported) */
    CS_EUNSUPPORTED =
        -7, /* Feature disabled at compile time or backend/locale unsupported by OS/CPU */
    CS_ELIMIT =
        -8, /* Configurable resource limit exceeded (regex states, repeat, combining marks) */
    CS_ESYNTAX = -9 /* Syntax error in format string, regex pattern, glob, or escape sequence */
} cs_status;

const char *cs_status_str(cs_status status);

/* ==========================================================================
 * 3. Core String Model & Memory Types (Section 2)
 * ========================================================================== */

/* Non-owning immutable byte view. Does not require trailing NUL.
 * {NULL, 0} is a valid empty view; {NULL, len > 0} is invalid (CS_EINVAL). */
typedef struct cs_view {
    const char *data;
    size_t len;
} cs_view;

/* Custom allocator with user context.
 * Must outlive all objects initialized with it. */
typedef struct cs_allocator {
    void *(*alloc)(void *ctx, size_t size, size_t align);
    void *(*realloc)(void *ctx, void *ptr, size_t old_size, size_t new_size, size_t align);
    void (*free)(void *ctx, void *ptr, size_t size, size_t align);
    void *ctx;
} cs_allocator;

/* Owning mutable string with dynamic capacity.
 * - Supports embedded NUL bytes; length is tracked explicitly in `len`.
 * - `cap` is the maximum payload bytes excluding the mandatory trailing NUL.
 * - For any initialized cs_string, `data` is non-NULL and `data[len] == '\0'`.
 * - Empty initialized strings point `data` to an internal static read-only ""
 *   sentinel with `cap == 0`.
 * - Implicit bitwise struct copying is forbidden; use cs_clone() or cs_move(). */
typedef struct cs_string {
    char *data;
    size_t len;
    size_t cap;
    cs_allocator alloc;
    uint32_t generation; /* Incremented on every mutation/destroy to detect stale indices */
    uint32_t flags;      /* Reserved ABI flags */
} cs_string;

/* Fixed-capacity user buffer (never allocates on the heap).
 * `cap` is maximum payload bytes excluding the mandatory trailing NUL. */
typedef struct cs_buffer {
    char *data;
    size_t len;
    size_t cap;
} cs_buffer;

/* Reusable scratch memory workspace for complex no-heap operations. */
typedef struct cs_workspace {
    void *mem;
    size_t size;
    size_t used;
} cs_workspace;

/* Incremental string builder wrapping either cs_string or cs_buffer. */
typedef struct cs_builder {
    cs_string *str_target;
    cs_buffer *buf_target;
    cs_status sticky_err;
    size_t required_bytes;
} cs_builder;

/* Convenience view constructors */
#define CS_LIT(literal) ((cs_view){(literal), sizeof(literal) - 1U})

static inline cs_view cs_empty_view(void) {
    cs_view v = {NULL, 0};
    return v;
}

static inline bool cs_view_is_valid(cs_view v) { return (v.data != NULL) || (v.len == 0); }

cs_view cs_sv(const char *cstr);
cs_view cs_sv_from_bytes(const char *data, size_t len);
cs_status cs_view_from_cstr(const char *cstr, cs_view *out);
cs_status cs_view_from_cstr_n(const char *cstr, size_t max_bytes, cs_view *out);
cs_status cs_view_from_bytes(const char *data, size_t len, cs_view *out);

cs_allocator cs_default_allocator(void);
cs_status cs_set_default_allocator(cs_allocator alloc);
void cs_reset_default_allocator(void);
#if CS_USE_MIMALLOC && !CS_NO_HEAP
cs_allocator cs_mimalloc_allocator(void *mi_heap);
#endif
void cs_workspace_init(cs_workspace *ws, void *mem, size_t size);
void cs_workspace_reset(cs_workspace *ws);

/* ==========================================================================
 * 4. Creation, Lifecycle, Memory & Building (Section 5)
 * ========================================================================== */

void cs_init(cs_string *s);
cs_status cs_init_with_allocator(cs_string *s, cs_allocator alloc);
cs_status cs_from_view(cs_string *s, cs_view v, const cs_allocator *alloc);
cs_status cs_from_cstr(cs_string *s, const char *cstr, const cs_allocator *alloc);
cs_status cs_from_bytes(cs_string *s, const char *data, size_t len, const cs_allocator *alloc);
cs_status cs_clone(cs_string *dst, const cs_string *src);
void cs_move(cs_string *dst, cs_string *src);
void cs_destroy(cs_string *s);
void cs_clear(cs_string *s);
void cs_reset(cs_string *s); /* Frees allocation and resets to empty state */

cs_status cs_reserve(cs_string *s, size_t min_cap);
cs_status cs_shrink_to_fit(cs_string *s);
cs_status cs_assign(cs_string *s, cs_view v);
cs_status cs_assign_cstr(cs_string *s, const char *cstr);
void cs_swap(cs_string *a, cs_string *b);
char *cs_detach(cs_string *s, size_t *out_len, size_t *out_cap);
cs_status cs_adopt(cs_string *s, char *buf, size_t len, size_t cap, cs_allocator alloc);

/* Fixed-buffer lifecycle */
cs_status cs_buffer_init(cs_buffer *b, char *storage, size_t storage_size);
void cs_buffer_clear(cs_buffer *b);
cs_view cs_buffer_as_view(const cs_buffer *b);
const char *cs_buffer_c_str(const cs_buffer *b);

/* String mutations (strong guarantee: unchanged on error; self-aliasing safe) */
cs_status cs_append(cs_string *s, cs_view v);
cs_status cs_append_cstr(cs_string *s, const char *cstr);
cs_status cs_append_byte(cs_string *s, uint8_t byte_val);
cs_status cs_append_codepoint(cs_string *s, uint32_t cp);
cs_status cs_append_many(cs_string *s, const cs_view *views, size_t count);
cs_status cs_prepend(cs_string *s, cs_view v);
cs_status cs_insert_bytes(cs_string *s, size_t pos, cs_view v);
cs_status cs_erase_bytes(cs_string *s, size_t begin, size_t end);
cs_status cs_replace_range_bytes(cs_string *s, size_t begin, size_t end, cs_view replacement);
cs_status cs_truncate_bytes(cs_string *s, size_t max_bytes);
cs_status cs_truncate_codepoints(cs_string *s, size_t max_codepoints);
cs_status cs_truncate_graphemes(cs_string *s, size_t max_graphemes);
cs_status cs_resize_bytes(cs_string *s, size_t new_len, uint8_t fill_byte);
cs_status cs_repeat(cs_string *dst, cs_view v, size_t count);
cs_status cs_repeat_buffer(cs_buffer *dst, cs_view v, size_t count, size_t *out_required);

/* Buffer mutations */
cs_status cs_buffer_append(cs_buffer *b, cs_view v, size_t *out_required);
cs_status cs_buffer_append_byte(cs_buffer *b, uint8_t byte_val, size_t *out_required);
cs_status cs_buffer_append_codepoint(cs_buffer *b, uint32_t cp, size_t *out_required);

/* Builder API */
void cs_builder_init_string(cs_builder *b, cs_string *s);
void cs_builder_init_buffer(cs_builder *b, cs_buffer *buf);
cs_status cs_builder_append(cs_builder *b, cs_view v);
cs_status cs_builder_append_cstr(cs_builder *b, const char *cstr);
cs_status cs_builder_append_byte(cs_builder *b, uint8_t byte_val);
cs_status cs_builder_append_codepoint(cs_builder *b, uint32_t cp);
cs_status cs_builder_finish(cs_builder *b, cs_view *out_view, size_t *out_required);

/* ==========================================================================
 * 5. Access, Slices, Units & Boundary Index (Sections 3 & 6)
 * ========================================================================== */

bool cs_is_empty(cs_view v);
size_t cs_len_bytes(cs_view v);
cs_status cs_len_codepoints(cs_view v, size_t *out_count);
cs_status cs_len_graphemes(cs_view v, size_t *out_count);
size_t cs_capacity(const cs_string *s);
const char *cs_data(const cs_string *s);
const char *cs_c_str(const cs_string *s);
cs_view cs_as_view(const cs_string *s);

cs_status cs_byte_at(cs_view v, size_t idx, uint8_t *out_byte);
cs_status cs_codepoint_at(cs_view v, size_t cp_idx, uint32_t *out_cp);
cs_status cs_grapheme_at(cs_view v, size_t graph_idx, cs_view *out_grapheme);

cs_status cs_front_byte(cs_view v, uint8_t *out_byte);
cs_status cs_back_byte(cs_view v, uint8_t *out_byte);
cs_status cs_front_codepoint(cs_view v, uint32_t *out_cp);
cs_status cs_back_codepoint(cs_view v, uint32_t *out_cp);
cs_status cs_front_grapheme(cs_view v, cs_view *out_grapheme);
cs_status cs_back_grapheme(cs_view v, cs_view *out_grapheme);

/* Half-open [begin, end) slices (strictly bounds-checked, no implicit clamping) */
cs_status cs_slice_bytes(cs_view v, size_t begin, size_t end, cs_view *out);
cs_view cs_slice_bytes_clamped(cs_view v, size_t begin, size_t end);
cs_status cs_slice_codepoints(cs_view v, size_t begin_cp, size_t end_cp, cs_view *out);
cs_status cs_slice_graphemes(cs_view v, size_t begin_gr, size_t end_gr, cs_view *out);

cs_status cs_take_bytes(cs_view v, size_t count, cs_view *out);
cs_status cs_drop_bytes(cs_view v, size_t count, cs_view *out);
cs_status cs_take_codepoints(cs_view v, size_t count, cs_view *out);
cs_status cs_drop_codepoints(cs_view v, size_t count, cs_view *out);
cs_status cs_take_graphemes(cs_view v, size_t count, cs_view *out);
cs_status cs_drop_graphemes(cs_view v, size_t count, cs_view *out);

/* Coordinate conversions */
cs_status cs_offset_to_cp_index(cs_view v, size_t byte_offset, size_t *out_cp_idx);
cs_status cs_cp_index_to_offset(cs_view v, size_t cp_idx, size_t *out_byte_offset);
cs_status cs_offset_to_grapheme_index(cs_view v, size_t byte_offset, size_t *out_gr_idx);
cs_status cs_grapheme_index_to_offset(cs_view v, size_t gr_idx, size_t *out_byte_offset);

/* Optional O(1) boundary index with mutation invalidation tracking */
typedef struct cs_index {
    const cs_string *owner;
    const char *bound_data;
    size_t bound_len;
    uint32_t owner_generation;
    size_t *cp_offsets;
    size_t cp_count;
    size_t *gr_offsets;
    size_t gr_count;
    cs_allocator alloc;
} cs_index;

cs_status cs_index_build(cs_index *idx, const cs_string *owner, cs_view v,
                         const cs_allocator *alloc);
void cs_index_destroy(cs_index *idx);
bool cs_index_is_valid(const cs_index *idx);
cs_status cs_index_cp_count(const cs_index *idx, size_t *out_count);
cs_status cs_index_grapheme_count(const cs_index *idx, size_t *out_count);
cs_status cs_index_cp_offset(const cs_index *idx, size_t cp_idx, size_t *out_offset);
cs_status cs_index_grapheme_span(const cs_index *idx, size_t gr_idx, cs_view *out_span);

/* ==========================================================================
 * 6. Iterators: Bytes, Code Points, Graphemes, Words, Sentences (Sections 2, 8, 10)
 * ========================================================================== */

typedef struct cs_byte_iter {
    cs_view v;
    size_t pos;
} cs_byte_iter;

void cs_byte_iter_init(cs_byte_iter *it, cs_view v);
cs_status cs_byte_iter_next(cs_byte_iter *it, uint8_t *out_byte, size_t *out_offset);
cs_status cs_byte_iter_prev(cs_byte_iter *it, uint8_t *out_byte, size_t *out_offset);

typedef enum cs_decode_mode {
    CS_DECODE_STRICT = 0, /* Return CS_EUTF8 on malformed sequence */
    CS_DECODE_REPLACE = 1 /* Emit U+FFFD for each maximal subpart of ill-formed sequence */
} cs_decode_mode;

typedef struct cs_cp_iter {
    cs_view v;
    size_t pos;
    cs_decode_mode mode;
} cs_cp_iter;

void cs_cp_iter_init(cs_cp_iter *it, cs_view v, cs_decode_mode mode);
cs_status cs_cp_iter_next(cs_cp_iter *it, uint32_t *out_cp, size_t *out_start, size_t *out_end);
cs_status cs_cp_iter_prev(cs_cp_iter *it, uint32_t *out_cp, size_t *out_start, size_t *out_end);

typedef struct cs_grapheme_iter {
    cs_view v;
    size_t pos;
    cs_decode_mode mode;
} cs_grapheme_iter;

void cs_grapheme_iter_init(cs_grapheme_iter *it, cs_view v, cs_decode_mode mode);
cs_status cs_grapheme_iter_next(cs_grapheme_iter *it, cs_view *out_cluster, size_t *out_start,
                                size_t *out_end);
cs_status cs_grapheme_iter_prev(cs_grapheme_iter *it, cs_view *out_cluster, size_t *out_start,
                                size_t *out_end);

typedef struct cs_word_iter {
    cs_view v;
    size_t pos;
    cs_decode_mode mode;
} cs_word_iter;

void cs_word_iter_init(cs_word_iter *it, cs_view v, cs_decode_mode mode);
cs_status cs_word_iter_next(cs_word_iter *it, cs_view *out_segment, size_t *out_start,
                            size_t *out_end, bool *out_is_word_like);

typedef struct cs_sentence_iter {
    cs_view v;
    size_t pos;
    cs_decode_mode mode;
} cs_sentence_iter;

void cs_sentence_iter_init(cs_sentence_iter *it, cs_view v, cs_decode_mode mode);
cs_status cs_sentence_iter_next(cs_sentence_iter *it, cs_view *out_sentence, size_t *out_start,
                                size_t *out_end);

/* ==========================================================================
 * 7. Search & Comparison (Section 7)
 * ========================================================================== */

typedef enum cs_match_mode {
    CS_MATCH_EXACT = 0,                  /* Exact byte comparison */
    CS_MATCH_ASCII_CASE_INSENSITIVE = 1, /* ASCII A-Z == a-z only */
    CS_MATCH_UNICODE_CASEFOLD = 2        /* Full Unicode CaseFolding (e.g. "ß" == "ss") */
} cs_match_mode;

typedef enum cs_overlap_mode {
    CS_OVERLAP_DISALLOW = 0, /* Advance past full match */
    CS_OVERLAP_ALLOW = 1     /* Advance by 1 unit (byte or code point) after match start */
} cs_overlap_mode;

bool cs_equal(cs_view a, cs_view b);
cs_status cs_equal_mode(cs_view a, cs_view b, cs_match_mode mode, bool *out_equal);
int cs_compare(cs_view a, cs_view b);
cs_status cs_compare_mode(cs_view a, cs_view b, cs_match_mode mode, int *out_cmp);
int cs_compare_natural(cs_view a, cs_view b);
cs_status cs_compare_collation(cs_view a, cs_view b, const char *locale_name, int *out_cmp);

bool cs_starts_with(cs_view text, cs_view prefix);
cs_status cs_starts_with_mode(cs_view text, cs_view prefix, cs_match_mode mode, bool *out_result,
                              size_t *out_match_end);
bool cs_ends_with(cs_view text, cs_view suffix);
cs_status cs_ends_with_mode(cs_view text, cs_view suffix, cs_match_mode mode, bool *out_result,
                            size_t *out_match_start);
bool cs_contains(cs_view text, cs_view needle);
cs_status cs_contains_mode(cs_view text, cs_view needle, cs_match_mode mode, bool *out_result);

cs_status cs_find(cs_view text, cs_view needle, size_t start_offset, size_t *out_pos);
cs_status cs_find_mode(cs_view text, cs_view needle, size_t start_offset, cs_match_mode mode,
                       size_t *out_start, size_t *out_end);
cs_status cs_rfind(cs_view text, cs_view needle, size_t end_offset, size_t *out_pos);
cs_status cs_rfind_mode(cs_view text, cs_view needle, size_t end_offset, cs_match_mode mode,
                        size_t *out_start, size_t *out_end);

cs_status cs_find_byte(cs_view text, uint8_t byte_val, size_t start_offset, size_t *out_pos);
cs_status cs_rfind_byte(cs_view text, uint8_t byte_val, size_t end_offset, size_t *out_pos);
cs_status cs_find_any_byte(cs_view text, cs_view byte_set, size_t start_offset, size_t *out_pos);
cs_status cs_find_not_any_byte(cs_view text, cs_view byte_set, size_t start_offset,
                               size_t *out_pos);

cs_status cs_count(cs_view text, cs_view needle, cs_overlap_mode overlap, size_t *out_count);
cs_status cs_count_mode(cs_view text, cs_view needle, cs_match_mode mode, cs_overlap_mode overlap,
                        size_t *out_count);
cs_status cs_find_all(cs_view text, cs_view needle, cs_match_mode mode, cs_overlap_mode overlap,
                      size_t *out_starts, size_t *out_ends, size_t max_matches,
                      size_t *out_total_count);

cs_view cs_remove_prefix(cs_view text, cs_view prefix);
cs_view cs_remove_suffix(cs_view text, cs_view suffix);
cs_view cs_common_prefix_bytes(cs_view a, cs_view b);
cs_view cs_common_suffix_bytes(cs_view a, cs_view b);
cs_status cs_common_prefix_codepoints(cs_view a, cs_view b, cs_view *out);
cs_status cs_common_suffix_codepoints(cs_view a, cs_view b, cs_view *out);
cs_status cs_common_prefix_graphemes(cs_view a, cs_view b, cs_view *out);
cs_status cs_common_suffix_graphemes(cs_view a, cs_view b, cs_view *out);

/* Prepared reusable pattern searcher (Two-Way O(n+m) worst-case) */
typedef struct cs_searcher {
    cs_view needle;
    cs_string owned_folded;
    cs_match_mode mode;
    size_t crit_pos;
    size_t period;
    bool periodic;
    uint8_t first_byte;
    uint8_t last_byte;
} cs_searcher;

cs_status cs_searcher_init(cs_searcher *sr, cs_view needle, cs_match_mode mode,
                           const cs_allocator *alloc);
void cs_searcher_destroy(cs_searcher *sr);
cs_status cs_searcher_find(const cs_searcher *sr, cs_view text, size_t start_offset,
                           size_t *out_start, size_t *out_end);
cs_status cs_searcher_rfind(const cs_searcher *sr, cs_view text, size_t end_offset,
                            size_t *out_start, size_t *out_end);

/* ==========================================================================
 * 8. Split, Join & Tokenization (Section 8)
 * ========================================================================== */

#define CS_UNLIMITED_SPLITS ((size_t)-1)

typedef struct cs_split_options {
    bool keep_empty;   /* True: emit empty views between adjacent delimiters */
    size_t max_splits; /* 0 = no splits (1 whole item); CS_UNLIMITED_SPLITS = unlimited */
    bool strip_outer;  /* True: strip one leading/trailing delimiter before splitting */
} cs_split_options;

typedef enum cs_line_mode {
    CS_LINES_ASCII = 0, /* LF (\n), CR (\r), CRLF (\r\n) */
    CS_LINES_UNICODE =
        1 /* ASCII + NEL (U+0085), VT (U+000B), FF (U+000C), LS (U+2028), PS (U+2029) */
} cs_line_mode;

typedef struct cs_split_iter {
    cs_view text;
    cs_view delim;
    cs_split_options opts;
    size_t pos;
    size_t splits_done;
    int kind; /* 0=delim, 1=ascii_ws, 2=unicode_ws, 3=lines */
    cs_line_mode line_mode;
    bool keep_line_endings;
    bool finished;
} cs_split_iter;

cs_split_options cs_default_split_options(void);

cs_status cs_split_iter_init(cs_split_iter *it, cs_view text, cs_view delim, cs_split_options opts);
cs_status cs_split_ascii_whitespace_iter_init(cs_split_iter *it, cs_view text, size_t max_splits);
cs_status cs_split_unicode_whitespace_iter_init(cs_split_iter *it, cs_view text, size_t max_splits);
cs_status cs_split_lines_iter_init(cs_split_iter *it, cs_view text, cs_line_mode mode,
                                   bool keep_endings);
cs_status cs_split_iter_next(cs_split_iter *it, cs_view *out_part);

cs_status cs_split_into(cs_view text, cs_view delim, cs_split_options opts, cs_view *out_views,
                        size_t max_views, size_t *out_count);
cs_status cs_rsplit_into(cs_view text, cs_view delim, cs_split_options opts, cs_view *out_views,
                         size_t max_views, size_t *out_count);

cs_status cs_split_once(cs_view text, cs_view delim, cs_view *out_before, cs_view *out_after,
                        bool *out_found);
cs_status cs_rsplit_once(cs_view text, cs_view delim, cs_view *out_before, cs_view *out_after,
                         bool *out_found);
cs_status cs_partition(cs_view text, cs_view delim, cs_view *out_before, cs_view *out_sep,
                       cs_view *out_after, bool *out_found);
cs_status cs_rpartition(cs_view text, cs_view delim, cs_view *out_before, cs_view *out_sep,
                        cs_view *out_after, bool *out_found);

cs_status cs_join(cs_string *dst, const cs_view *parts, size_t count, cs_view sep);
cs_status cs_join_into(cs_string *dst, const cs_view *parts, size_t count,
                       cs_view sep); /* Appends to existing dst */
cs_status cs_join_buffer(cs_buffer *dst, const cs_view *parts, size_t count, cs_view sep,
                         size_t *out_required);

/* ==========================================================================
 * 9. Text Transformations & Normalization (Section 9)
 * ========================================================================== */

/* Trim (non-allocating view operations) */
cs_view cs_trim_ascii(cs_view v);
cs_view cs_trim_start_ascii(cs_view v);
cs_view cs_trim_end_ascii(cs_view v);

cs_status cs_trim_unicode(cs_view v, cs_view *out);
cs_status cs_trim_start_unicode(cs_view v, cs_view *out);
cs_status cs_trim_end_unicode(cs_view v, cs_view *out);

cs_status cs_trim_codepoints(cs_view v, cs_view cutset_utf8, cs_view *out);
cs_status cs_trim_start_codepoints(cs_view v, cs_view cutset_utf8, cs_view *out);
cs_status cs_trim_end_codepoints(cs_view v, cs_view cutset_utf8, cs_view *out);

/* Replace & Translate */
cs_status cs_replace_first(cs_string *dst, cs_view text, cs_view needle, cs_view replacement);
cs_status cs_replace_all(cs_string *dst, cs_view text, cs_view needle, cs_view replacement);
cs_status cs_replace_all_buffer(cs_buffer *dst, cs_view text, cs_view needle, cs_view replacement,
                                size_t *out_required);
cs_status cs_remove_all(cs_string *dst, cs_view text, cs_view needle);

typedef struct cs_cp_mapping {
    uint32_t from_cp;
    cs_view to_utf8; /* Empty view deletes the code point */
} cs_cp_mapping;

cs_status cs_translate_codepoints(cs_string *dst, cs_view text, const cs_cp_mapping *map,
                                  size_t map_len);

/* Case conversions (ASCII and full context/locale-sensitive Unicode) */
typedef enum cs_case_locale {
    CS_LOCALE_ROOT = 0, /* Default root Unicode rules (including Greek Final_Sigma) */
    CS_LOCALE_TR = 1,   /* Turkish (İ/ı rules) */
    CS_LOCALE_AZ = 2,   /* Azerbaijani (İ/ı rules) */
    CS_LOCALE_LT = 3    /* Lithuanian (Soft_Dotted + More_Above rules) */
} cs_case_locale;

cs_status cs_to_lower_ascii(cs_string *dst, cs_view v);
cs_status cs_to_upper_ascii(cs_string *dst, cs_view v);
cs_status cs_to_lower_ascii_buffer(cs_buffer *dst, cs_view v, size_t *out_required);
cs_status cs_to_upper_ascii_buffer(cs_buffer *dst, cs_view v, size_t *out_required);

cs_status cs_to_lower_unicode(cs_string *dst, cs_view v, cs_case_locale locale);
cs_status cs_to_upper_unicode(cs_string *dst, cs_view v, cs_case_locale locale);
cs_status cs_to_casefold_unicode(cs_string *dst, cs_view v, cs_case_locale locale);
cs_status cs_to_title_unicode(cs_string *dst, cs_view v, cs_case_locale locale);
cs_status cs_capitalize_unicode(cs_string *dst, cs_view v, cs_case_locale locale, bool lower_rest);

cs_status cs_to_lower_unicode_buffer(cs_buffer *dst, cs_view v, cs_case_locale locale,
                                     size_t *out_required);
cs_status cs_to_upper_unicode_buffer(cs_buffer *dst, cs_view v, cs_case_locale locale,
                                     size_t *out_required);
cs_status cs_to_casefold_unicode_buffer(cs_buffer *dst, cs_view v, cs_case_locale locale,
                                        size_t *out_required);

/* Padding & Alignment */
cs_status cs_pad_start_bytes(cs_string *dst, cs_view v, size_t target_len, uint8_t fill_byte);
cs_status cs_pad_end_bytes(cs_string *dst, cs_view v, size_t target_len, uint8_t fill_byte);
cs_status cs_center_bytes(cs_string *dst, cs_view v, size_t target_len, uint8_t fill_byte);

cs_status cs_pad_start_codepoints(cs_string *dst, cs_view v, size_t target_cps, uint32_t fill_cp);
cs_status cs_pad_end_codepoints(cs_string *dst, cs_view v, size_t target_cps, uint32_t fill_cp);
cs_status cs_center_codepoints(cs_string *dst, cs_view v, size_t target_cps, uint32_t fill_cp);

cs_status cs_pad_start_graphemes(cs_string *dst, cs_view v, size_t target_grs,
                                 cs_view fill_grapheme);
cs_status cs_pad_end_graphemes(cs_string *dst, cs_view v, size_t target_grs, cs_view fill_grapheme);
cs_status cs_center_graphemes(cs_string *dst, cs_view v, size_t target_grs, cs_view fill_grapheme);

/* Reversal */
cs_status cs_reverse_bytes(cs_string *dst, cs_view v);
cs_status cs_reverse_codepoints(cs_string *dst, cs_view v);
cs_status cs_reverse_graphemes(cs_string *dst, cs_view v);

/* Tabs, Indentation & Newlines */
typedef enum cs_column_unit {
    CS_COL_BYTES = 0,
    CS_COL_CODEPOINTS = 1,
    CS_COL_GRAPHEMES = 2
} cs_column_unit;

cs_status cs_expand_tabs(cs_string *dst, cs_view v, size_t tab_stop, cs_column_unit unit);
cs_status cs_indent(cs_string *dst, cs_view v, cs_view prefix);
cs_status cs_dedent(cs_string *dst, cs_view v);
cs_status cs_normalize_newlines(cs_string *dst, cs_view v, cs_line_mode mode,
                                cs_view target_newline);

/* Unicode Normalization (UAX #15) */
typedef enum cs_norm_form {
    CS_NORM_NFC = 0,
    CS_NORM_NFD = 1,
    CS_NORM_NFKC = 2,
    CS_NORM_NFKD = 3
} cs_norm_form;

typedef enum cs_quick_check_result {
    CS_QC_YES = 0,
    CS_QC_NO = 1,
    CS_QC_MAYBE = 2
} cs_quick_check_result;

cs_status cs_normalize(cs_string *dst, cs_view v, cs_norm_form form);
cs_status cs_normalize_buffer(cs_buffer *dst, cs_view v, cs_norm_form form, cs_workspace *ws,
                              size_t *out_required);
cs_status cs_is_normalized(cs_view v, cs_norm_form form, bool *out_normalized);
cs_status cs_norm_quick_check(cs_view v, cs_norm_form form, cs_quick_check_result *out_qc);

/* Separate opt-in diacritic stripping (strips General_Category Mn/Mc/Me after NFD decomposition) */
cs_status cs_strip_diacritics(cs_string *dst, cs_view v);

/* ==========================================================================
 * 10. Unicode Validation, Codecs & Properties (Section 10)
 * ========================================================================== */

cs_status cs_validate_utf8(cs_view v, size_t *out_error_offset);
cs_status cs_decode_utf8_next(cs_view v, size_t offset, uint32_t *out_cp, size_t *out_next_offset);
cs_status cs_decode_utf8_prev(cs_view v, size_t offset, uint32_t *out_cp, size_t *out_prev_offset);
cs_status cs_encode_utf8_codepoint(uint32_t cp, char out_buf[4], size_t *out_len);
cs_status cs_repair_utf8(cs_string *dst, cs_view v);
cs_status cs_repair_utf8_buffer(cs_buffer *dst, cs_view v, size_t *out_required);

typedef enum cs_endian { CS_ENDIAN_LITTLE = 0, CS_ENDIAN_BIG = 1 } cs_endian;

typedef enum cs_bom_policy {
    CS_BOM_NONE = 0,       /* Do not strip or emit BOM */
    CS_BOM_STRIP_AUTO = 1, /* Detect BOM on input to set endianness and strip it */
    CS_BOM_EMIT = 2        /* Prepend U+FEFF BOM on output */
} cs_bom_policy;

cs_status cs_utf8_to_utf16(cs_view v, uint16_t *out_units, size_t max_units, size_t *out_len);
cs_status cs_utf16_to_utf8(const uint16_t *units, size_t unit_len, cs_string *dst);
cs_status cs_utf8_to_utf16_bytes(cs_view v, cs_endian endian, cs_bom_policy bom,
                                 cs_string *dst_bytes);
cs_status cs_utf16_bytes_to_utf8(cs_view raw_bytes, cs_endian default_endian, cs_bom_policy bom,
                                 cs_string *dst);

cs_status cs_utf8_to_utf32(cs_view v, uint32_t *out_cps, size_t max_cps, size_t *out_len);
cs_status cs_utf32_to_utf8(const uint32_t *cps, size_t cp_len, cs_string *dst);
cs_status cs_utf8_to_utf32_bytes(cs_view v, cs_endian endian, cs_bom_policy bom,
                                 cs_string *dst_bytes);
cs_status cs_utf32_bytes_to_utf8(cs_view raw_bytes, cs_endian default_endian, cs_bom_policy bom,
                                 cs_string *dst);

#ifdef _WIN32
cs_status cs_from_win32_wide(cs_string *dst, const wchar_t *wstr, size_t wlen);
cs_status cs_to_win32_wide(cs_view v, wchar_t *out_wbuf, size_t max_wchars, size_t *out_wlen);
#endif

/* General_Category (matches Unicode UAX #44) */
typedef enum cs_general_category {
    CS_GC_CN = 0,  /* Unassigned */
    CS_GC_LU = 1,  /* Uppercase_Letter */
    CS_GC_LL = 2,  /* Lowercase_Letter */
    CS_GC_LT = 3,  /* Titlecase_Letter */
    CS_GC_LM = 4,  /* Modifier_Letter */
    CS_GC_LO = 5,  /* Other_Letter */
    CS_GC_MN = 6,  /* Nonspacing_Mark */
    CS_GC_MC = 7,  /* Spacing_Mark */
    CS_GC_ME = 8,  /* Enclosing_Mark */
    CS_GC_ND = 9,  /* Decimal_Number */
    CS_GC_NL = 10, /* Letter_Number */
    CS_GC_NO = 11, /* Other_Number */
    CS_GC_PC = 12, /* Connector_Punctuation */
    CS_GC_PD = 13, /* Dash_Punctuation */
    CS_GC_PS = 14, /* Open_Punctuation */
    CS_GC_PE = 15, /* Close_Punctuation */
    CS_GC_PI = 16, /* Initial_Punctuation */
    CS_GC_PF = 17, /* Final_Punctuation */
    CS_GC_PO = 18, /* Other_Punctuation */
    CS_GC_SM = 19, /* Math_Symbol */
    CS_GC_SC = 20, /* Currency_Symbol */
    CS_GC_SK = 21, /* Modifier_Symbol */
    CS_GC_SO = 22, /* Other_Symbol */
    CS_GC_ZS = 23, /* Space_Separator */
    CS_GC_ZL = 24, /* Line_Separator */
    CS_GC_ZP = 25, /* Paragraph_Separator */
    CS_GC_CC = 26, /* Control */
    CS_GC_CF = 27, /* Format */
    CS_GC_CS = 28, /* Surrogate */
    CS_GC_CO = 29  /* Private_Use */
} cs_general_category;

cs_general_category cs_cp_category(uint32_t cp);
bool cs_cp_is_ascii(uint32_t cp);
bool cs_cp_is_alphabetic(uint32_t cp);
bool cs_cp_is_digit(uint32_t cp);
bool cs_cp_is_whitespace(uint32_t cp);
bool cs_cp_is_upper(uint32_t cp);
bool cs_cp_is_lower(uint32_t cp);
bool cs_cp_is_title(uint32_t cp);
bool cs_cp_is_alnum(uint32_t cp);
bool cs_cp_is_control(uint32_t cp);
/* Printable: defined as any valid code point in General_Category L, M, N, P, S,
 * or U+0020 (ASCII SPACE), excluding Zl, Zp, Cc, Cf, Cs, Co, Cn. */
bool cs_cp_is_printable(uint32_t cp);

/* String predicates:
 * - cs_is_ascii(empty) == true
 * - All other predicates return false on empty string and CS_EUTF8 on invalid UTF-8. */
bool cs_is_ascii(cs_view v);
cs_status cs_is_alphabetic(cs_view v, bool *out_result);
cs_status cs_is_digit(cs_view v, bool *out_result);
cs_status cs_is_whitespace(cs_view v, bool *out_result);
cs_status cs_is_alnum(cs_view v, bool *out_result);
cs_status cs_is_upper(cs_view v, bool *out_result);
cs_status cs_is_lower(cs_view v, bool *out_result);
cs_status cs_is_printable(cs_view v, bool *out_result);

/* ==========================================================================
 * 11. Formatting & Deterministic Numbers (Section 11)
 * ========================================================================== */

/* libc printf wrappers (caller guarantees format string safety and vararg types;
 * inherits system libc locale behavior for %f). */
cs_status cs_format_printf(cs_string *dst, const char *fmt, ...) CS_PRINTF_ATTR(2, 3);
cs_status cs_append_printf(cs_string *dst, const char *fmt, ...) CS_PRINTF_ATTR(2, 3);
cs_status cs_vformat_printf(cs_string *dst, const char *fmt, va_list ap);
cs_status cs_vappend_printf(cs_string *dst, const char *fmt, va_list ap);
cs_status cs_buffer_format_printf(cs_buffer *dst, size_t *out_required, const char *fmt, ...)
    CS_PRINTF_ATTR(3, 4);

/* Deterministic locale-independent typed formatter:
 * Grammar: `{` [index] [`:` [[fill]align] [sign] [`#`] [`0`] [width] [`.` precision] [type]] `}`
 * Escaped braces: `{{` and `}}`.
 * Types: `s` (string/view, preserves embedded NULs), `d` (decimal), `b` (binary),
 *        `o` (octal), `x`/`X` (hex), `f`/`F` (fixed float), `e`/`E` (scientific),
 *        `g`/`G` (general/shortest float), `c` (Unicode code point encoded as UTF-8). */
typedef enum cs_fmt_arg_kind {
    CS_FMT_VIEW = 0,
    CS_FMT_I64 = 1,
    CS_FMT_U64 = 2,
    CS_FMT_F64 = 3,
    CS_FMT_BOOL = 4,
    CS_FMT_CP = 5
} cs_fmt_arg_kind;

typedef struct cs_fmt_arg {
    cs_fmt_arg_kind kind;
    union {
        cs_view v;
        int64_t i64;
        uint64_t u64;
        double f64;
        bool b;
        uint32_t cp;
    } val;
} cs_fmt_arg;

static inline cs_fmt_arg cs_arg_view(cs_view v) {
    cs_fmt_arg a;
    a.kind = CS_FMT_VIEW;
    a.val.v = v;
    return a;
}
static inline cs_fmt_arg cs_arg_cstr(const char *s) {
    cs_fmt_arg a;
    a.kind = CS_FMT_VIEW;
    a.val.v = cs_sv(s);
    return a;
}
static inline cs_fmt_arg cs_arg_i64(int64_t x) {
    cs_fmt_arg a;
    a.kind = CS_FMT_I64;
    a.val.i64 = x;
    return a;
}
static inline cs_fmt_arg cs_arg_u64(uint64_t x) {
    cs_fmt_arg a;
    a.kind = CS_FMT_U64;
    a.val.u64 = x;
    return a;
}
static inline cs_fmt_arg cs_arg_f64(double x) {
    cs_fmt_arg a;
    a.kind = CS_FMT_F64;
    a.val.f64 = x;
    return a;
}
static inline cs_fmt_arg cs_arg_bool(bool x) {
    cs_fmt_arg a;
    a.kind = CS_FMT_BOOL;
    a.val.b = x;
    return a;
}
static inline cs_fmt_arg cs_arg_cp(uint32_t cp) {
    cs_fmt_arg a;
    a.kind = CS_FMT_CP;
    a.val.cp = cp;
    return a;
}

cs_status cs_format_args(cs_string *dst, cs_view fmt, const cs_fmt_arg *args, size_t arg_count);
cs_status cs_append_format_args(cs_string *dst, cs_view fmt, const cs_fmt_arg *args,
                                size_t arg_count);
cs_status cs_buffer_format_args(cs_buffer *dst, cs_view fmt, const cs_fmt_arg *args,
                                size_t arg_count, size_t *out_required);

/* Deterministic locale-independent numeric parsing & formatting */
typedef struct cs_num_parse_options {
    int base;               /* 0 = auto-detect via prefix if allow_base_prefix, else 10; or 2..36 */
    bool allow_whitespace;  /* Allow leading/trailing ASCII whitespace */
    bool allow_underscores; /* Allow '_' between digits */
    bool allow_base_prefix; /* Allow 0x/0X, 0b/0B, 0o/0O */
    bool allow_special_fp;  /* Allow NaN, Inf, Infinity (case-insensitive) for float */
} cs_num_parse_options;

cs_num_parse_options cs_default_num_parse_options(void);

cs_status cs_parse_i64_exact(cs_view v, cs_num_parse_options opts, int64_t *out_val);
cs_status cs_parse_i64_prefix(cs_view v, cs_num_parse_options opts, int64_t *out_val,
                              size_t *out_consumed);
cs_status cs_parse_u64_exact(cs_view v, cs_num_parse_options opts, uint64_t *out_val);
cs_status cs_parse_u64_prefix(cs_view v, cs_num_parse_options opts, uint64_t *out_val,
                              size_t *out_consumed);
cs_status cs_parse_f64_exact(cs_view v, cs_num_parse_options opts, double *out_val);
cs_status cs_parse_f64_prefix(cs_view v, cs_num_parse_options opts, double *out_val,
                              size_t *out_consumed);
cs_status cs_parse_bool_exact(cs_view v, bool *out_val);
cs_status cs_parse_bool_prefix(cs_view v, bool *out_val, size_t *out_consumed);

typedef enum cs_float_fmt_mode {
    CS_FLOAT_SHORTEST = 0,  /* Shortest round-trip representation for IEEE-754 binary64 */
    CS_FLOAT_FIXED = 1,     /* Fixed-point ddd.ddd with given precision */
    CS_FLOAT_SCIENTIFIC = 2 /* Scientific d.ddde+dd with given precision */
} cs_float_fmt_mode;

cs_status cs_format_i64(cs_string *dst, int64_t val, int base, bool uppercase);
cs_status cs_format_u64(cs_string *dst, uint64_t val, int base, bool uppercase);
cs_status cs_format_f64(cs_string *dst, double val, cs_float_fmt_mode mode, int precision,
                        bool uppercase);
cs_status cs_format_bool(cs_string *dst, bool val);

cs_status cs_format_i64_buffer(cs_buffer *dst, int64_t val, int base, bool uppercase,
                               size_t *out_required);
cs_status cs_format_u64_buffer(cs_buffer *dst, uint64_t val, int base, bool uppercase,
                               size_t *out_required);
cs_status cs_format_f64_buffer(cs_buffer *dst, double val, cs_float_fmt_mode mode, int precision,
                               bool uppercase, size_t *out_required);

/* ==========================================================================
 * 12. Escaping, Data Codecs & Streaming (Section 12)
 * ========================================================================== */

cs_status cs_hex_encode(cs_string *dst, cs_view data, bool uppercase);
cs_status cs_hex_encode_buffer(cs_buffer *dst, cs_view data, bool uppercase, size_t *out_required);
cs_status cs_hex_decode(cs_string *dst, cs_view hex_str);
cs_status cs_hex_decode_buffer(cs_buffer *dst, cs_view hex_str, size_t *out_required);

typedef enum cs_base64_variant {
    CS_BASE64_STANDARD = 0, /* RFC 4648 Section 4 (+ and /) */
    CS_BASE64_URL = 1       /* RFC 4648 Section 5 (- and _) */
} cs_base64_variant;

typedef enum cs_base64_pad_policy {
    CS_B64_PAD_REQUIRED = 0, /* Emit '=' on encode; require '=' on decode */
    CS_B64_PAD_OMIT = 1,     /* Omit '=' on encode; reject '=' on decode */
    CS_B64_PAD_OPTIONAL = 2  /* Accept either padded or unpadded on decode */
} cs_base64_pad_policy;

cs_status cs_base64_encode(cs_string *dst, cs_view data, cs_base64_variant variant,
                           cs_base64_pad_policy pad);
cs_status cs_base64_encode_buffer(cs_buffer *dst, cs_view data, cs_base64_variant variant,
                                  cs_base64_pad_policy pad, size_t *out_required);
cs_status cs_base64_decode(cs_string *dst, cs_view b64_str, cs_base64_variant variant,
                           cs_base64_pad_policy pad);
cs_status cs_base64_decode_buffer(cs_buffer *dst, cs_view b64_str, cs_base64_variant variant,
                                  cs_base64_pad_policy pad, size_t *out_required);

/* JSON String Escape/Unescape (RFC 8259, without outer quotes) */
cs_status cs_json_escape(cs_string *dst, cs_view utf8_str, bool escape_non_ascii);
cs_status cs_json_unescape(cs_string *dst, cs_view escaped_str);

/* URI / URL Percent Encoding (RFC 3986 + application/x-www-form-urlencoded) */
typedef enum cs_url_mode {
    CS_URL_PATH = 0,           /* RFC 3986 path segment (unreserved + sub-delims + ':' '@') */
    CS_URL_QUERY = 1,          /* Strict RFC 3986 query component (space -> %20, '+' -> %2B) */
    CS_URL_FORM_URLENCODED = 2 /* HTML form-urlencoded (space <-> '+', strict unreserved) */
} cs_url_mode;

cs_status cs_url_encode(cs_string *dst, cs_view data, cs_url_mode mode);
cs_status cs_url_decode(cs_string *dst, cs_view encoded, cs_url_mode mode);

/* HTML 5 Basic Entities (&amp; &lt; &gt; &quot; &#39;) */
cs_status cs_html_escape(cs_string *dst, cs_view text);
cs_status cs_html_unescape(cs_string *dst, cs_view escaped);

/* C String Literal Escape/Unescape */
cs_status cs_c_escape(cs_string *dst, cs_view data);
cs_status cs_c_unescape(cs_string *dst, cs_view escaped);

/* Streaming UTF-8 Validator / Repair State Machine */
typedef struct cs_utf8_stream {
    uint8_t pending[4];
    size_t pending_len;
    cs_decode_mode mode;
    size_t total_consumed;
} cs_utf8_stream;

void cs_utf8_stream_init(cs_utf8_stream *st, cs_decode_mode mode);
cs_status cs_utf8_stream_feed(cs_utf8_stream *st, cs_view chunk, cs_string *dst,
                              size_t *out_chunk_consumed);
cs_status cs_utf8_stream_finish(cs_utf8_stream *st, cs_string *dst);

/* Streaming Base64 Encoder & Decoder */
typedef struct cs_base64_enc_stream {
    uint8_t rem[3];
    size_t rem_len;
    cs_base64_variant variant;
    cs_base64_pad_policy pad;
} cs_base64_enc_stream;

void cs_base64_enc_stream_init(cs_base64_enc_stream *st, cs_base64_variant variant,
                               cs_base64_pad_policy pad);
cs_status cs_base64_enc_stream_feed(cs_base64_enc_stream *st, cs_view chunk, cs_string *dst);
cs_status cs_base64_enc_stream_finish(cs_base64_enc_stream *st, cs_string *dst);

typedef struct cs_base64_dec_stream {
    uint8_t quad[4];
    size_t quad_len;
    size_t pad_seen;
    cs_base64_variant variant;
    cs_base64_pad_policy pad;
} cs_base64_dec_stream;

void cs_base64_dec_stream_init(cs_base64_dec_stream *st, cs_base64_variant variant,
                               cs_base64_pad_policy pad);
cs_status cs_base64_dec_stream_feed(cs_base64_dec_stream *st, cs_view chunk, cs_string *dst);
cs_status cs_base64_dec_stream_finish(cs_base64_dec_stream *st, cs_string *dst);

/* ==========================================================================
 * 13. Pattern Matching: Wildcard/Glob & Bounded Thompson NFA Regex (Section 13)
 * ========================================================================== */

cs_status cs_glob_match(cs_view pattern, cs_view text, cs_match_mode mode, bool *out_matched);

#define CS_REGEX_MAX_CAPTURES_LIMIT 16U

typedef enum cs_regex_flags {
    CS_REGEX_NONE = 0,
    CS_REGEX_CASE_INSENSITIVE = (1 << 0), /* Uses Unicode Simple_Case_Folding */
    CS_REGEX_UNICODE_CLASSES = (1 << 1),  /* \d, \w, \s use Unicode properties instead of ASCII */
    CS_REGEX_DOTALL = (1 << 2),           /* '.' matches '\n' as well */
    CS_REGEX_MULTILINE = (1 << 3)         /* '^' and '$' match start/end of lines */
} cs_regex_flags;

typedef struct cs_regex_limits {
    size_t max_instructions; /* Default: 1024 */
    size_t max_repetition;   /* Default: 256 */
    size_t max_captures;     /* Default: 16 (including group 0) */
    size_t max_input_bytes;  /* Default: 16 MiB */
} cs_regex_limits;

typedef struct cs_regex_span {
    size_t start;
    size_t end;
    bool matched;
} cs_regex_span;

typedef struct cs_regex_match {
    cs_regex_span groups[CS_REGEX_MAX_CAPTURES_LIMIT];
    size_t group_count;
} cs_regex_match;

typedef struct cs_regex_inst cs_regex_inst;

typedef struct cs_regex {
    cs_regex_inst *prog;
    size_t prog_len;
    size_t prog_cap;
    uint32_t *class_pool;
    size_t class_pool_len;
    size_t class_pool_cap;
    size_t capture_count;
    uint32_t flags;
    cs_regex_limits limits;
    cs_allocator alloc;
} cs_regex;

cs_regex_limits cs_default_regex_limits(void);

cs_status cs_regex_compile(cs_regex *re, cs_view pattern, uint32_t flags,
                           const cs_regex_limits *limits, const cs_allocator *alloc);
void cs_regex_destroy(cs_regex *re);

cs_status cs_regex_find(const cs_regex *re, cs_view text, size_t start_offset,
                        cs_regex_match *out_match);
cs_status cs_regex_find_all(const cs_regex *re, cs_view text, cs_regex_match *out_matches,
                            size_t max_matches, size_t *out_total_count);
cs_status cs_regex_split(const cs_regex *re, cs_view text, size_t max_splits, cs_view *out_parts,
                         size_t max_parts, size_t *out_count);
cs_status cs_regex_replace(const cs_regex *re, cs_string *dst, cs_view text,
                           cs_view replacement_template, bool replace_all);

/* ==========================================================================
 * 14. Deterministic & Keyed Hashing (Section 14)
 * ========================================================================== */

/* Deterministic non-cryptographic hash: xxHash64 (v1, seed = 0).
 * Identical across all architectures, endiannesses, and SIMD/scalar backends. */
uint64_t cs_hash64(cs_view v);
uint64_t cs_hash64_seed(cs_view v, uint64_t seed);
static inline uint64_t cs_hash_bytes(cs_view v) { return cs_hash64(v); }

/* Keyed PRF hash for hash tables with untrusted keys: SipHash-2-4 (128-bit key). */
uint64_t cs_hash_keyed(cs_view v, const uint8_t key[16]);

/* Explicit OS entropy helper (never called implicitly during hashing). */
cs_status cs_hash_seed_from_os(uint8_t out_key[16]);

/* ==========================================================================
 * 15. Runtime CPU Dispatch & Backend Inspection (Section 15)
 * ========================================================================== */

typedef enum cs_backend {
    CS_BACKEND_AUTO = 0,
    CS_BACKEND_SCALAR = 1,
    CS_BACKEND_LIBC = 2,
    CS_BACKEND_SSE2 = 3,
    CS_BACKEND_AVX2 = 4,
    CS_BACKEND_AVX512 = 5,
    CS_BACKEND_NEON = 6
} cs_backend;

typedef struct cs_runtime_info_t {
    cs_backend active_backend;
    cs_backend forced_backend;
    uint32_t supported_mask; /* Bitmask of (1U << cs_backend) */
    const char *arch_name;
    const char *backend_name;
} cs_runtime_info_t;

void cs_runtime_info(cs_runtime_info_t *out_info);
bool cs_backend_supported(cs_backend backend);
cs_status cs_set_forced_backend(cs_backend backend);
const char *cs_backend_name(cs_backend backend);

#endif /* CSTR_H_INCLUDED */
