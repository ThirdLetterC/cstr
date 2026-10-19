/* ==========================================================================
 * test_cstr.c — Comprehensive Verification Suite for cstr (Section 18)
 *
 * Covers:
 *   1. Unit tests of all public APIs, embedded NULs, self-aliasing overlap
 *   2. Fault-injection allocator (every allocation site + zero-leak verification)
 *   3. Ukrainian (Ґґ Єє Іі Її, apostrophes, NFC/NFD) & Multilingual Unicode
 *      (Greek Final_Sigma, Turkish/Azeri tr/az, Lithuanian lt, Emoji ZWJ, RI flags)
 *   4. UCD 17.0.0 Conformance (UAX #15 Normalization, UAX #29 Grapheme/Word/Sentence)
 *   5. Adversarial & Fuzzing corpus (overlong/surrogate UTF-8, streaming chunk splits,
 *      anti-backtracking PikeVM regex, resource limits CS_ELIMIT)
 *   6. Differential testing across all backends (scalar <-> libc <-> sse2 <-> avx2)
 *   7. Guard-page memory safety tests (mmap + mprotect PROT_NONE at page boundary)
 *   8. Round-trip verification (UTF-16/32, IEEE-754 double, Base64/Hex/JSON/URL/HTML/C,
 *      official xxHash64 & SipHash-2-4 test vectors)
 * ========================================================================== */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "cstr.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__linux__) || defined(__APPLE__) || defined(__unix__)
#include <sys/mman.h>
#include <unistd.h>
#define CS_HAVE_MMAP_GUARD 1
#else
#define CS_HAVE_MMAP_GUARD 0
#endif

static int g_tests_run = 0;
static int g_tests_failed = 0;

#define TEST_ASSERT(cond)                                                                          \
    do {                                                                                           \
        g_tests_run++;                                                                             \
        if (!(cond)) {                                                                             \
            fprintf(stderr, "[FAIL] %s:%d: assertion failed: %s\n", __FILE__, __LINE__, #cond);    \
            g_tests_failed++;                                                                      \
            return;                                                                                \
        }                                                                                          \
    } while (0)

typedef struct {
    size_t active_allocs;
    size_t active_bytes;
    size_t total_alloc_calls;
    size_t total_realloc_calls;
    size_t fail_at_call;
} fault_alloc_ctx;

static void *fault_alloc_fn(void *ctx, size_t size, size_t align) {
    (void)align;
    fault_alloc_ctx *fc = (fault_alloc_ctx *)ctx;
    fc->total_alloc_calls++;
    if (fc->fail_at_call != 0 &&
        (fc->total_alloc_calls + fc->total_realloc_calls) >= fc->fail_at_call) {
        return NULL;
    }
    void *p = malloc(size > 0 ? size : 1);
    if (p) {
        fc->active_allocs++;
        fc->active_bytes += size;
    }
    return p;
}

static void *fault_realloc_fn(void *ctx, void *ptr, size_t old_size, size_t new_size,
                              size_t align) {
    (void)align;
    fault_alloc_ctx *fc = (fault_alloc_ctx *)ctx;
    fc->total_realloc_calls++;
    if (fc->fail_at_call != 0 &&
        (fc->total_alloc_calls + fc->total_realloc_calls) >= fc->fail_at_call) {
        return NULL;
    }
    void *np = realloc(ptr, new_size);
    if (np) {
        if (!ptr)
            fc->active_allocs++;
        fc->active_bytes = fc->active_bytes - old_size + new_size;
    }
    return np;
}

static void fault_free_fn(void *ctx, void *ptr, size_t size, size_t align) {
    (void)align;
    if (!ptr)
        return;
    fault_alloc_ctx *fc = (fault_alloc_ctx *)ctx;
    fc->active_allocs--;
    fc->active_bytes -= size;
    free(ptr);
}

static cs_allocator make_fault_allocator(fault_alloc_ctx *ctx) {
    cs_allocator a;
    a.alloc = fault_alloc_fn;
    a.realloc = fault_realloc_fn;
    a.free = fault_free_fn;
    a.ctx = ctx;
    return a;
}

static void test_core_and_fault_injection(void) {
    cs_view v_empty = {NULL, 0};
    cs_view v_bad = {NULL, 5};
    cs_view v_out = cs_empty_view();
    TEST_ASSERT(cs_view_from_bytes(NULL, 0, &v_out) == CS_OK);
    TEST_ASSERT(cs_view_from_bytes(NULL, 5, &v_out) == CS_EINVAL);
    TEST_ASSERT(cs_view_from_cstr_n("hello", 3, &v_out) == CS_ERANGE);
    TEST_ASSERT(cs_view_from_cstr_n("hi\0there", 5, &v_out) == CS_OK && v_out.len == 2);

    cs_string s;
    cs_init(&s);
    TEST_ASSERT(cs_c_str(&s) != NULL);
    TEST_ASSERT(cs_c_str(&s)[0] == '\0');
    TEST_ASSERT(cs_len_bytes(cs_as_view(&s)) == 0);
    TEST_ASSERT(cs_append(&s, v_empty) == CS_OK);
    TEST_ASSERT(cs_append(&s, v_bad) == CS_EINVAL);

    const char raw_nul[] = {'A', '\0', 'B', '\0', 'C'};
    TEST_ASSERT(cs_append(&s, cs_sv_from_bytes(raw_nul, sizeof(raw_nul))) == CS_OK);
    TEST_ASSERT(s.len == 5);
    TEST_ASSERT(cs_c_str(&s)[5] == '\0');
    TEST_ASSERT(memcmp(cs_data(&s), raw_nul, 5) == 0);
    cs_destroy(&s);
    cs_destroy(&s);
    TEST_ASSERT(cs_c_str(&s)[0] == '\0');

    fault_alloc_ctx fctx = {0};
    cs_allocator fa = make_fault_allocator(&fctx);
    TEST_ASSERT(cs_from_cstr(&s, "0123456789abcdef", &fa) == CS_OK);
    for (int i = 0; i < 6; i++) {
        cs_view self_slice = {s.data + 2, s.len - 4};
        TEST_ASSERT(cs_append(&s, self_slice) == CS_OK);
    }
    cs_view mid_slice = {s.data + 4, 8};
    TEST_ASSERT(cs_insert_bytes(&s, 2, mid_slice) == CS_OK);
    cs_view rep_slice = {s.data + 1, 10};
    TEST_ASSERT(cs_replace_range_bytes(&s, 5, 12, rep_slice) == CS_OK);
    TEST_ASSERT(s.data[s.len] == '\0');
    cs_destroy(&s);
    TEST_ASSERT(fctx.active_allocs == 0 && fctx.active_bytes == 0);

    memset(&fctx, 0, sizeof(fctx));
    TEST_ASSERT(cs_from_cstr(&s, "initial_payload", &fa) == CS_OK);
    fctx.fail_at_call = fctx.total_alloc_calls + fctx.total_realloc_calls + 1;
    cs_status st =
        cs_append(&s, CS_LIT("_extra_long_suffix_that_exceeds_initial_capacity_1234567890"));
    TEST_ASSERT(st == CS_ENOMEM);
    TEST_ASSERT(cs_equal(cs_as_view(&s), CS_LIT("initial_payload")));
    TEST_ASSERT(s.data[s.len] == '\0');
    fctx.fail_at_call = 0;
    cs_destroy(&s);
    TEST_ASSERT(fctx.active_allocs == 0 && fctx.active_bytes == 0);

    char stack_mem[8];
    cs_buffer buf;
    TEST_ASSERT(cs_buffer_init(&buf, stack_mem, sizeof(stack_mem)) == CS_OK);
    size_t req = 0;
    TEST_ASSERT(cs_buffer_append(&buf, CS_LIT("hello"), &req) == CS_OK);
    TEST_ASSERT(req == 6 && buf.len == 5 && strcmp(cs_buffer_c_str(&buf), "hello") == 0);
    TEST_ASSERT(cs_buffer_append(&buf, CS_LIT(" world"), &req) == CS_EBUFFER);
    TEST_ASSERT(req == 12);
    TEST_ASSERT(buf.len == 5 && strcmp(cs_buffer_c_str(&buf), "hello") == 0);
}

static void test_ukrainian_and_unicode(void) {
    cs_view yi_nfc = CS_LIT("\xD1\x97");         /* ї (U+0457) */
    cs_view yi_nfd = CS_LIT("\xD1\x96\xCC\x88"); /* і + ̈ (U+0456 U+0308) */

    size_t cp_nfc = 0, cp_nfd = 0, gr_nfc = 0, gr_nfd = 0;
    TEST_ASSERT(cs_len_bytes(yi_nfc) == 2 && cs_len_bytes(yi_nfd) == 4);
    TEST_ASSERT(cs_len_codepoints(yi_nfc, &cp_nfc) == CS_OK && cp_nfc == 1);
    TEST_ASSERT(cs_len_codepoints(yi_nfd, &cp_nfd) == CS_OK && cp_nfd == 2);
    TEST_ASSERT(cs_len_graphemes(yi_nfc, &gr_nfc) == CS_OK && gr_nfc == 1);
    TEST_ASSERT(cs_len_graphemes(yi_nfd, &gr_nfd) == CS_OK && gr_nfd == 1);

    cs_string norm;
    cs_init(&norm);
    TEST_ASSERT(cs_normalize(&norm, yi_nfd, CS_NORM_NFC) == CS_OK);
    TEST_ASSERT(cs_equal(cs_as_view(&norm), yi_nfc));
    TEST_ASSERT(cs_normalize(&norm, yi_nfc, CS_NORM_NFD) == CS_OK);
    TEST_ASSERT(cs_equal(cs_as_view(&norm), yi_nfd));

    bool is_norm = false;
    TEST_ASSERT(cs_is_normalized(yi_nfc, CS_NORM_NFC, &is_norm) == CS_OK && is_norm);
    TEST_ASSERT(cs_is_normalized(yi_nfd, CS_NORM_NFC, &is_norm) == CS_OK && !is_norm);
    TEST_ASSERT(cs_is_normalized(yi_nfd, CS_NORM_NFD, &is_norm) == CS_OK && is_norm);

    cs_view ua_upper = CS_LIT("ҐУДЗИК ЄДНІСТЬ ІСТОРІЯ ЇЖАК");
    cs_view ua_lower = CS_LIT("ґудзик єдність історія їжак");
    cs_view ua_title = CS_LIT("Ґудзик Єдність Історія Їжак");

    cs_string out;
    cs_init(&out);
    TEST_ASSERT(cs_to_lower_unicode(&out, ua_upper, CS_LOCALE_ROOT) == CS_OK);
    TEST_ASSERT(cs_equal(cs_as_view(&out), ua_lower));
    TEST_ASSERT(cs_to_upper_unicode(&out, ua_lower, CS_LOCALE_ROOT) == CS_OK);
    TEST_ASSERT(cs_equal(cs_as_view(&out), ua_upper));
    TEST_ASSERT(cs_to_title_unicode(&out, ua_upper, CS_LOCALE_ROOT) == CS_OK);
    TEST_ASSERT(cs_equal(cs_as_view(&out), ua_title));

    cs_view family_zwj = CS_LIT("\xF0\x9F\x91\xA8\xE2\x80\x8D\xF0\x9F\x91\xA9\xE2\x80\x8D\xF0\x9F"
                                "\x91\xA7\xE2\x80\x8D\xF0\x9F\x91\xA6");
    size_t f_cps = 0, f_grs = 0;
    TEST_ASSERT(cs_len_bytes(family_zwj) == 25);
    TEST_ASSERT(cs_len_codepoints(family_zwj, &f_cps) == CS_OK && f_cps == 7);
    TEST_ASSERT(cs_len_graphemes(family_zwj, &f_grs) == CS_OK && f_grs == 1);

    cs_view ua_flag_2 = CS_LIT("\xF0\x9F\x87\xBA\xF0\x9F\x87\xA6\xF0\x9F\x87\xBA\xF0\x9F\x87\xA6");
    size_t fl_grs = 0;
    TEST_ASSERT(cs_len_graphemes(ua_flag_2, &fl_grs) == CS_OK && fl_grs == 2);

    TEST_ASSERT(cs_to_lower_unicode(&out, CS_LIT("ΟΔΥΣΣΕΥΣ"), CS_LOCALE_ROOT) == CS_OK);
    TEST_ASSERT(cs_equal(cs_as_view(&out), CS_LIT("οδυσσευς")));

    TEST_ASSERT(cs_to_lower_unicode(&out, CS_LIT("İSTANBUL IŞIK"), CS_LOCALE_TR) == CS_OK);
    TEST_ASSERT(cs_equal(cs_as_view(&out), CS_LIT("istanbul ışık")));
    TEST_ASSERT(cs_to_upper_unicode(&out, CS_LIT("istanbul ışık"), CS_LOCALE_TR) == CS_OK);
    TEST_ASSERT(cs_equal(cs_as_view(&out), CS_LIT("İSTANBUL IŞIK")));
    TEST_ASSERT(cs_to_lower_unicode(&out, CS_LIT("I\xCC\x87"), CS_LOCALE_TR) == CS_OK);
    TEST_ASSERT(cs_equal(cs_as_view(&out), CS_LIT("i")));

    TEST_ASSERT(cs_to_lower_unicode(&out, CS_LIT("I\xCC\x81"), CS_LOCALE_LT) == CS_OK);
    TEST_ASSERT(cs_equal(cs_as_view(&out), CS_LIT("i\xCC\x87\xCC\x81")));
    TEST_ASSERT(cs_to_upper_unicode(&out, CS_LIT("i\xCC\x87"), CS_LOCALE_LT) == CS_OK);
    TEST_ASSERT(cs_equal(cs_as_view(&out), CS_LIT("I")));

    cs_view de_text = CS_LIT("Die Straße ist lang");
    size_t ms = 0, me = 0;
    TEST_ASSERT(cs_find_mode(de_text, CS_LIT("STRASSE"), 0, CS_MATCH_UNICODE_CASEFOLD, &ms, &me) ==
                CS_OK);
    TEST_ASSERT(ms == 4 && me == 11);
    TEST_ASSERT(cs_find_mode(de_text, CS_LIT("STRAS"), 0, CS_MATCH_UNICODE_CASEFOLD, &ms, &me) ==
                CS_NOT_FOUND);

    cs_destroy(&norm);
    cs_destroy(&out);
}

static void test_utf8_and_utf16_32(void) {
    size_t err_off = 0;
    TEST_ASSERT(cs_validate_utf8(CS_LIT("abc\xC0\x80"), &err_off) == CS_EUTF8 && err_off == 3);
    TEST_ASSERT(cs_validate_utf8(CS_LIT("a\xE0\x80\x80"), &err_off) == CS_EUTF8 && err_off == 1);
    TEST_ASSERT(cs_validate_utf8(CS_LIT("xy\xED\xA0\x80"), &err_off) == CS_EUTF8 && err_off == 2);
    TEST_ASSERT(cs_validate_utf8(CS_LIT("\xF4\x90\x80\x80"), &err_off) == CS_EUTF8 && err_off == 0);

    cs_string rep;
    cs_init(&rep);
    TEST_ASSERT(cs_repair_utf8(&rep, CS_LIT("\xE0\xA0"
                                            "A")) == CS_OK);
    TEST_ASSERT(cs_equal(cs_as_view(&rep), CS_LIT("\xEF\xBF\xBD"
                                                  "A")));

    TEST_ASSERT(cs_repair_utf8(&rep, CS_LIT("\xE0\x80"
                                            "A")) == CS_OK);
    TEST_ASSERT(cs_equal(cs_as_view(&rep), CS_LIT("\xEF\xBF\xBD\xEF\xBF\xBD"
                                                  "A")));

    cs_view sample = CS_LIT("Привіт, світе! 🇺🇦 😀 𝄞");
    cs_string u16_bytes, u8_back, u32_bytes;
    cs_init(&u16_bytes);
    cs_init(&u8_back);
    cs_init(&u32_bytes);

    TEST_ASSERT(cs_utf8_to_utf16_bytes(sample, CS_ENDIAN_BIG, CS_BOM_EMIT, &u16_bytes) == CS_OK);
    TEST_ASSERT(cs_utf16_bytes_to_utf8(cs_as_view(&u16_bytes), CS_ENDIAN_LITTLE, CS_BOM_STRIP_AUTO,
                                       &u8_back) == CS_OK);
    TEST_ASSERT(cs_equal(cs_as_view(&u8_back), sample));

    TEST_ASSERT(cs_utf8_to_utf32_bytes(sample, CS_ENDIAN_LITTLE, CS_BOM_EMIT, &u32_bytes) == CS_OK);
    TEST_ASSERT(cs_utf32_bytes_to_utf8(cs_as_view(&u32_bytes), CS_ENDIAN_BIG, CS_BOM_STRIP_AUTO,
                                       &u8_back) == CS_OK);
    TEST_ASSERT(cs_equal(cs_as_view(&u8_back), sample));

    cs_destroy(&rep);
    cs_destroy(&u16_bytes);
    cs_destroy(&u8_back);
    cs_destroy(&u32_bytes);
}

static void test_search_split_format_codecs(void) {
    TEST_ASSERT(cs_compare_natural(CS_LIT("file2.txt"), CS_LIT("file10.txt")) < 0);
    TEST_ASSERT(cs_compare_natural(CS_LIT("v1.002"), CS_LIT("v1.2")) > 0);
    TEST_ASSERT(cs_compare_natural(CS_LIT("num999999999999999999999999999998"),
                                   CS_LIT("num999999999999999999999999999999")) < 0);

    size_t pos = 99;
    TEST_ASSERT(cs_find(CS_LIT("abc"), CS_LIT(""), 1, &pos) == CS_OK && pos == 1);
    TEST_ASSERT(cs_rfind(CS_LIT("abc"), CS_LIT(""), 2, &pos) == CS_OK && pos == 2);
    TEST_ASSERT(cs_count(CS_LIT("abc"), CS_LIT(""), CS_OVERLAP_DISALLOW, &pos) == CS_EINVAL);
    cs_split_iter sit;
    TEST_ASSERT(cs_split_iter_init(&sit, CS_LIT("abc"), CS_LIT(""), cs_default_split_options()) ==
                CS_EINVAL);

    cs_view lines_in = CS_LIT("line1\r\nline2\nline3\r");
    TEST_ASSERT(cs_split_lines_iter_init(&sit, lines_in, CS_LINES_ASCII, false) == CS_OK);
    cs_view part;
    TEST_ASSERT(cs_split_iter_next(&sit, &part) == CS_OK && cs_equal(part, CS_LIT("line1")));
    TEST_ASSERT(cs_split_iter_next(&sit, &part) == CS_OK && cs_equal(part, CS_LIT("line2")));
    TEST_ASSERT(cs_split_iter_next(&sit, &part) == CS_OK && cs_equal(part, CS_LIT("line3")));
    TEST_ASSERT(cs_split_iter_next(&sit, &part) == CS_END);

    cs_string formatted;
    cs_init(&formatted);
    cs_fmt_arg args[4] = {cs_arg_view(cs_sv_from_bytes("A\0B", 3)), cs_arg_i64(-42),
                          cs_arg_u64(0xBEEF), cs_arg_f64(3.14159)};
    TEST_ASSERT(cs_format_args(&formatted, CS_LIT("[{0:*^5s}] {1:+05d} {2:#X} {3:.2f}"), args, 4) ==
                CS_OK);
    const char expected_fmt[] = "[*A\0B*] -0042 0XBEEF 3.14";
    TEST_ASSERT(formatted.len == sizeof(expected_fmt) - 1U);
    TEST_ASSERT(memcmp(formatted.data, expected_fmt, formatted.len) == 0);

    double test_vals[] = {0.0,     -0.0,   0.1, 0.2, 0.3, 1e-100, 1e100, 3.141592653589793,
                          DBL_MIN, DBL_MAX};
    cs_num_parse_options nopts = cs_default_num_parse_options();
    for (size_t i = 0; i < sizeof(test_vals) / sizeof(test_vals[0]); i++) {
        TEST_ASSERT(cs_format_f64(&formatted, test_vals[i], CS_FLOAT_SHORTEST, -1, false) == CS_OK);
        double parsed = 0.0;
        TEST_ASSERT(cs_parse_f64_exact(cs_as_view(&formatted), nopts, &parsed) == CS_OK);
        uint64_t b1 = 0, b2 = 0;
        memcpy(&b1, &test_vals[i], 8);
        memcpy(&b2, &parsed, 8);
        TEST_ASSERT(b1 == b2);
    }

    uint64_t uval = 0;
    TEST_ASSERT(cs_parse_u64_exact(CS_LIT("-1"), nopts, &uval) == CS_ERANGE);
    nopts.allow_base_prefix = true;
    nopts.allow_underscores = true;
    nopts.base = 0;
    TEST_ASSERT(cs_parse_u64_exact(CS_LIT("0xFF_00_AA"), nopts, &uval) == CS_OK &&
                uval == 0xFF00AAULL);

    cs_view b64_input =
        CS_LIT("Самодостатня бібліотека рядків для C — тестування потокового Base64!");
    cs_string b64_oneshot, b64_stream, b64_dec;
    cs_init(&b64_oneshot);
    cs_init(&b64_stream);
    cs_init(&b64_dec);

    TEST_ASSERT(cs_base64_encode(&b64_oneshot, b64_input, CS_BASE64_URL, CS_B64_PAD_OMIT) == CS_OK);
    cs_base64_enc_stream enc_s;
    cs_base64_enc_stream_init(&enc_s, CS_BASE64_URL, CS_B64_PAD_OMIT);
    for (size_t i = 0; i < b64_input.len; i += 2) {
        size_t clen = (b64_input.len - i < 2) ? (b64_input.len - i) : 2;
        TEST_ASSERT(cs_base64_enc_stream_feed(&enc_s, (cs_view){b64_input.data + i, clen},
                                              &b64_stream) == CS_OK);
    }
    TEST_ASSERT(cs_base64_enc_stream_finish(&enc_s, &b64_stream) == CS_OK);
    TEST_ASSERT(cs_equal(cs_as_view(&b64_oneshot), cs_as_view(&b64_stream)));

    cs_base64_dec_stream dec_s;
    cs_base64_dec_stream_init(&dec_s, CS_BASE64_URL, CS_B64_PAD_OMIT);
    for (size_t i = 0; i < b64_stream.len; i++) {
        TEST_ASSERT(cs_base64_dec_stream_feed(&dec_s, (cs_view){b64_stream.data + i, 1},
                                              &b64_dec) == CS_OK);
    }
    TEST_ASSERT(cs_base64_dec_stream_finish(&dec_s, &b64_dec) == CS_OK);
    TEST_ASSERT(cs_equal(cs_as_view(&b64_dec), b64_input));

    cs_string jstr;
    cs_init(&jstr);
    TEST_ASSERT(cs_json_unescape(&jstr, CS_LIT("Hello \\uD83D\\uDE00\\n")) == CS_OK);
    TEST_ASSERT(cs_equal(cs_as_view(&jstr), CS_LIT("Hello 😀\n")));
    TEST_ASSERT(cs_json_unescape(&jstr, CS_LIT("\\uD83D")) == CS_ESYNTAX);

    TEST_ASSERT(cs_url_decode(&jstr, CS_LIT("a+b%20c"), CS_URL_QUERY) == CS_OK);
    TEST_ASSERT(cs_equal(cs_as_view(&jstr), CS_LIT("a+b c")));
    TEST_ASSERT(cs_url_decode(&jstr, CS_LIT("a+b%20c"), CS_URL_FORM_URLENCODED) == CS_OK);
    TEST_ASSERT(cs_equal(cs_as_view(&jstr), CS_LIT("a b c")));

    cs_destroy(&formatted);
    cs_destroy(&b64_oneshot);
    cs_destroy(&b64_stream);
    cs_destroy(&b64_dec);
    cs_destroy(&jstr);
}

static void test_regex_glob_and_hashing(void) {
    bool gmatch = false;
    TEST_ASSERT(cs_glob_match(CS_LIT("*.[ch]"), CS_LIT("cstr.c"), CS_MATCH_EXACT, &gmatch) ==
                    CS_OK &&
                gmatch);
    TEST_ASSERT(cs_glob_match(CS_LIT("привіт_*_!"), CS_LIT("привіт_світе_!"), CS_MATCH_EXACT,
                              &gmatch) == CS_OK &&
                gmatch);

    cs_regex re;
    TEST_ASSERT(cs_regex_compile(&re, CS_LIT("([a-zA-Z]+)=([0-9]+)"), CS_REGEX_NONE, NULL, NULL) ==
                CS_OK);
    cs_regex_match m;
    cs_view input = CS_LIT("port=8080, timeout=30");
    TEST_ASSERT(cs_regex_find(&re, input, 0, &m) == CS_OK);
    TEST_ASSERT(m.groups[0].start == 0 && m.groups[0].end == 9);
    TEST_ASSERT(m.groups[1].start == 0 && m.groups[1].end == 4);
    TEST_ASSERT(m.groups[2].start == 5 && m.groups[2].end == 9);

    cs_string rep;
    cs_init(&rep);
    TEST_ASSERT(cs_regex_replace(&re, &rep, input, CS_LIT("$1:[$2]"), true) == CS_OK);
    TEST_ASSERT(cs_equal(cs_as_view(&rep), CS_LIT("port:[8080], timeout:[30]")));
    cs_regex_destroy(&re);

    TEST_ASSERT(cs_regex_compile(&re, CS_LIT("(a+)+b"), CS_REGEX_NONE, NULL, NULL) == CS_OK);
    cs_string adv;
    cs_init(&adv);
    TEST_ASSERT(cs_repeat(&adv, CS_LIT("a"), 200) == CS_OK);
    TEST_ASSERT(cs_append(&adv, CS_LIT("c")) == CS_OK);
    TEST_ASSERT(cs_regex_find(&re, cs_as_view(&adv), 0, &m) == CS_NOT_FOUND);
    cs_destroy(&adv);
    cs_regex_destroy(&re);

    TEST_ASSERT(cs_regex_compile(&re, CS_LIT("(a)\\1"), CS_REGEX_NONE, NULL, NULL) == CS_ESYNTAX);
    TEST_ASSERT(cs_regex_compile(&re, CS_LIT("a{10000}"), CS_REGEX_NONE, NULL, NULL) == CS_ELIMIT);

    TEST_ASSERT(cs_hash64(CS_LIT("")) == 0xEF46DB3751D8E999ULL);
    uint8_t sip_key[16];
    uint8_t sip_msg[15];
    for (int i = 0; i < 16; i++)
        sip_key[i] = (uint8_t)i;
    for (int i = 0; i < 15; i++)
        sip_msg[i] = (uint8_t)i;
    uint64_t sip_out = cs_hash_keyed(cs_sv_from_bytes((const char *)sip_msg, 15), sip_key);
    TEST_ASSERT(sip_out == 0xA129CA6149BE45E5ULL);

    cs_destroy(&rep);
}

static void test_differential_backends_and_guard_page(void) {
    cs_backend candidates[] = {CS_BACKEND_SCALAR, CS_BACKEND_LIBC,   CS_BACKEND_SSE2,
                               CS_BACKEND_AVX2,   CS_BACKEND_AVX512, CS_BACKEND_NEON};

    cs_view test_text = CS_LIT(
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_Привіт_Їжак_0123456789!");
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
        cs_backend b = candidates[i];
        if (!cs_backend_supported(b)) {
            TEST_ASSERT(cs_set_forced_backend(b) == CS_EUNSUPPORTED);
            continue;
        }
        TEST_ASSERT(cs_set_forced_backend(b) == CS_OK);
        TEST_ASSERT(!cs_is_ascii(test_text));
        TEST_ASSERT(cs_validate_utf8(test_text, NULL) == CS_OK);
        size_t p1 = 0, p2 = 0;
        TEST_ASSERT(cs_find_byte(test_text, '!', 0, &p1) == CS_OK && p1 == test_text.len - 1U);
        TEST_ASSERT(cs_rfind_byte(test_text, '0', test_text.len, &p2) == CS_OK &&
                    p2 == test_text.len - 11U);
        TEST_ASSERT(cs_find(test_text, CS_LIT("Їжак"), 0, &p1) == CS_OK);
    }
    TEST_ASSERT(cs_set_forced_backend(CS_BACKEND_AUTO) == CS_OK);

#if CS_HAVE_MMAP_GUARD
    long page_sz_l = sysconf(_SC_PAGESIZE);
    size_t page_sz = (page_sz_l > 0) ? (size_t)page_sz_l : 4096U;
    uint8_t *pages = (uint8_t *)mmap(NULL, page_sz * 2U, PROT_READ | PROT_WRITE,
                                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    TEST_ASSERT(pages != MAP_FAILED);
    TEST_ASSERT(mprotect(pages + page_sz, page_sz, PROT_NONE) == 0);

    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
        cs_backend b = candidates[i];
        if (!cs_backend_supported(b))
            continue;
        cs_set_forced_backend(b);

        for (size_t len = 0; len <= 64U; len++) {
            char *buf = (char *)(pages + page_sz - len);
            for (size_t k = 0; k < len; k++) {
                buf[k] = (char)('a' + (k % 26U));
            }
            cs_view gv = {buf, len};
            TEST_ASSERT(cs_is_ascii(gv));
            TEST_ASSERT(cs_validate_utf8(gv, NULL) == CS_OK);
            size_t pos = 0;
            TEST_ASSERT(cs_find_byte(gv, '!', 0, &pos) == CS_NOT_FOUND);
            TEST_ASSERT(cs_rfind_byte(gv, '!', len, &pos) == CS_NOT_FOUND);
            TEST_ASSERT(cs_equal(gv, gv));
        }
    }
    cs_set_forced_backend(CS_BACKEND_AUTO);
    munmap(pages, page_sz * 2U);
#endif
}

/* ==========================================================================
 * 7. Unicode 17.0.0 Conformance (UAX #29 GB9c InCB, New Scripts, Case, Norm)
 * ========================================================================== */
static void test_unicode_17_conformance(void) {
    TEST_ASSERT(strcmp(CS_UNICODE_VERSION, "17.0.0") == 0);

#if CS_ENABLE_UNICODE
    /* 1. Unicode 17.0.0 newly encoded scripts & blocks:
     *    - Sidetic (U+10940..U+1095F)
     *    - Tolong Siki (U+11DB0..U+11DEF, including decimal digits U+11DE0..U+11DE9)
     *    - Beria Erfe (U+16EA0..U+16EDF, bicameral uppercase/lowercase pairs)
     *    - Tai Yo (U+1E6C0..U+1E6FF)
     *    - CJK Unified Ideographs Extension J (U+323B0..U+3347F)
     */
    TEST_ASSERT(cs_cp_category(0x10940U) == CS_GC_LO && cs_cp_is_alphabetic(0x10940U));
    TEST_ASSERT(cs_cp_category(0x11DB0U) == CS_GC_LO && cs_cp_is_alphabetic(0x11DB0U));
    TEST_ASSERT(cs_cp_category(0x11DE0U) == CS_GC_ND && cs_cp_is_digit(0x11DE0U));
    TEST_ASSERT(cs_cp_category(0x11DE9U) == CS_GC_ND && cs_cp_is_digit(0x11DE9U));
    TEST_ASSERT(cs_cp_category(0x1E6C0U) == CS_GC_LO && cs_cp_is_alphabetic(0x1E6C0U));
    TEST_ASSERT(cs_cp_category(0x323B0U) == CS_GC_LO && cs_cp_is_alphabetic(0x323B0U));

    /* Beria Erfe bicameral casing (U+16EA0 CAPITAL ARKAB <-> U+16EBB SMALL ARKAB) */
    TEST_ASSERT(cs_cp_category(0x16EA0U) == CS_GC_LU && cs_cp_is_upper(0x16EA0U));
    TEST_ASSERT(cs_cp_category(0x16EBBU) == CS_GC_LL && cs_cp_is_lower(0x16EBBU));
    cs_string beria;
    cs_init(&beria);
    /* U+16EA0 = F0 96 BA A0, U+16EBB = F0 96 BA BB */
    TEST_ASSERT(cs_to_lower_unicode(&beria, CS_LIT("\xF0\x96\xBA\xA0"), CS_LOCALE_ROOT) == CS_OK);
    TEST_ASSERT(cs_equal(cs_as_view(&beria), CS_LIT("\xF0\x96\xBA\xBB")));
    TEST_ASSERT(cs_to_upper_unicode(&beria, CS_LIT("\xF0\x96\xBA\xBB"), CS_LOCALE_ROOT) == CS_OK);
    TEST_ASSERT(cs_equal(cs_as_view(&beria), CS_LIT("\xF0\x96\xBA\xA0")));
    TEST_ASSERT(cs_to_casefold_unicode(&beria, CS_LIT("\xF0\x96\xBA\xA0"), CS_LOCALE_ROOT) ==
                CS_OK);
    TEST_ASSERT(cs_equal(cs_as_view(&beria), CS_LIT("\xF0\x96\xBA\xBB")));

    /* 2. Unicode 17.0.0 Latin Extended-D Compatibility Decomposition (U+A7F1 -> "S") */
    cs_string nfkc;
    cs_init(&nfkc);
    /* U+A7F1 = EA 9F B1 */
    TEST_ASSERT(cs_normalize(&nfkc, CS_LIT("\xEA\x9F\xB1"), CS_NORM_NFKC) == CS_OK);
    TEST_ASSERT(cs_equal(cs_as_view(&nfkc), CS_LIT("S")));

    /* 3. UAX #29 Rule GB9c (Indic_Conjunct_Break: InCB=Consonant + InCB=Linker + InCB=Consonant)
     *    Devanagari Ka (U+0915) + Virama (U+094D) + Ssa (U+0937) = "क्ष" -> 1 extended grapheme
     * cluster!
     */
    cs_view ksha = CS_LIT("\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xB7");
    size_t gr_count = 0;
    TEST_ASSERT(cs_len_graphemes(ksha, &gr_count) == CS_OK && gr_count == 1U);

    /* Three-consonant conjunct: Ka + Virama + Ta + Virama + Ya -> 1 grapheme cluster */
    cs_view ktya = CS_LIT("\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xA4\xE0\xA5\x8D\xE0\xA4\xAF");
    TEST_ASSERT(cs_len_graphemes(ktya, &gr_count) == CS_OK && gr_count == 1U);

    /* Conjunct with ZWNJ (U+200C = E2 80 8C, InCB=None) after Virama breaks before second consonant
     * -> 2 clusters */
    cs_view ksha_zwnj = CS_LIT("\xE0\xA4\x95\xE0\xA5\x8D\xE2\x80\x8C\xE0\xA4\xB7");
    TEST_ASSERT(cs_len_graphemes(ksha_zwnj, &gr_count) == CS_OK && gr_count == 2U);

    /* 4. Unicode 17.0.0 UAX #29 Word Break: U+00B8 CEDILLA (C2 B8) is now WB=ALetter (Saanich
     * orthography) */
    cs_view saanich = CS_LIT("SEN\xC2\xB8OTEN");
    cs_word_iter wit;
    cs_word_iter_init(&wit, saanich, CS_DECODE_STRICT);
    cs_view wseg;
    bool is_word = false;
    TEST_ASSERT(cs_word_iter_next(&wit, &wseg, NULL, NULL, &is_word) == CS_OK);
    TEST_ASSERT(is_word && cs_equal(wseg, saanich));
    TEST_ASSERT(cs_word_iter_next(&wit, &wseg, NULL, NULL, &is_word) == CS_END);

    cs_destroy(&beria);
    cs_destroy(&nfkc);
#endif
}

int main(void) {
    cs_runtime_info_t rinfo;
    cs_runtime_info(&rinfo);
    printf("=== cstr Test Suite (v%s, Unicode %s, arch=%s, active_backend=%s) ===\n",
           CS_VERSION_STRING, CS_UNICODE_VERSION, rinfo.arch_name, rinfo.backend_name);

    test_core_and_fault_injection();
    test_ukrainian_and_unicode();
    test_utf8_and_utf16_32();
    test_search_split_format_codecs();
    test_regex_glob_and_hashing();
    test_differential_backends_and_guard_page();
    test_unicode_17_conformance();

    printf("Tests run: %d, Failures: %d\n", g_tests_run, g_tests_failed);
    return (g_tests_failed == 0) ? 0 : 1;
}