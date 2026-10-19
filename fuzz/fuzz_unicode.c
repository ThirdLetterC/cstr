#include "fuzz_check.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size > 4096)
        return 0;
    cs_view input = {(const char *)data, size};
    cs_string repaired, normalized, twice, encoded, decoded;
    cs_init(&repaired);
    cs_init(&normalized);
    cs_init(&twice);
    cs_init(&encoded);
    cs_init(&decoded);

    FUZZ_CHECK(cs_repair_utf8(&repaired, input) == CS_OK);
    fuzz_check_string(&repaired);
    cs_view text = cs_as_view(&repaired);
    FUZZ_CHECK(cs_validate_utf8(text, NULL) == CS_OK);
    if (cs_validate_utf8(input, NULL) == CS_OK)
        fuzz_check_equal(text, input);

    /* Decoding must make progress and encoding must reproduce the same bytes. */
    for (size_t offset = 0; offset < text.len;) {
        uint32_t cp;
        size_t next, count;
        char bytes[4];
        FUZZ_CHECK(cs_decode_utf8_next(text, offset, &cp, &next) == CS_OK);
        FUZZ_CHECK(next > offset && next <= text.len);
        FUZZ_CHECK(cs_encode_utf8_codepoint(cp, bytes, &count) == CS_OK);
        fuzz_check_equal((cs_view){bytes, count}, (cs_view){text.data + offset, next - offset});
        offset = next;
    }

    cs_grapheme_iter iterator;
    cs_grapheme_iter_init(&iterator, text, CS_DECODE_STRICT);
    size_t previous = 0, start, end;
    cs_view cluster;
    cs_status status;
    while ((status = cs_grapheme_iter_next(&iterator, &cluster, &start, &end)) == CS_OK) {
        FUZZ_CHECK(start == previous && end > start && end <= text.len);
        fuzz_check_equal(cluster, (cs_view){text.data + start, end - start});
        previous = end;
    }
    FUZZ_CHECK(status == CS_END && previous == text.len);

    for (int form = CS_NORM_NFC; form <= CS_NORM_NFKD; form++) {
        status = cs_normalize(&normalized, text, (cs_norm_form)form);
        FUZZ_CHECK(status == CS_OK || status == CS_ELIMIT);
        if (status != CS_OK)
            continue;
        fuzz_check_string(&normalized);
        FUZZ_CHECK(cs_validate_utf8(cs_as_view(&normalized), NULL) == CS_OK);
        FUZZ_CHECK(cs_normalize(&twice, cs_as_view(&normalized), (cs_norm_form)form) == CS_OK);
        fuzz_check_equal(cs_as_view(&twice), cs_as_view(&normalized));
    }

    for (int endian = CS_ENDIAN_LITTLE; endian <= CS_ENDIAN_BIG; endian++) {
        FUZZ_CHECK(cs_utf8_to_utf16_bytes(text, (cs_endian)endian, CS_BOM_NONE, &encoded) == CS_OK);
        FUZZ_CHECK(cs_utf16_bytes_to_utf8(cs_as_view(&encoded), (cs_endian)endian, CS_BOM_NONE,
                                          &decoded) == CS_OK);
        fuzz_check_equal(cs_as_view(&decoded), text);
    }

    cs_destroy(&decoded);
    cs_destroy(&encoded);
    cs_destroy(&twice);
    cs_destroy(&normalized);
    cs_destroy(&repaired);
    return 0;
}
