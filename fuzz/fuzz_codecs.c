#include "fuzz_check.h"

static void check_decode(cs_string *dst, cs_status status) {
    fuzz_check_string(dst);
    if (status != CS_OK)
        fuzz_check_equal(cs_as_view(dst), CS_LIT("unchanged"));
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size > 4096)
        return 0;
    cs_view input = {(const char *)data, size};
    cs_string encoded, decoded, streamed;
    cs_init(&encoded);
    cs_init(&decoded);
    cs_init(&streamed);

    FUZZ_CHECK(cs_hex_encode(&encoded, input, false) == CS_OK);
    FUZZ_CHECK(cs_hex_decode(&decoded, cs_as_view(&encoded)) == CS_OK);
    fuzz_check_equal(cs_as_view(&decoded), input);

    for (int variant = CS_BASE64_STANDARD; variant <= CS_BASE64_URL; variant++) {
        for (int pad = CS_B64_PAD_REQUIRED; pad <= CS_B64_PAD_OMIT; pad++) {
            FUZZ_CHECK(cs_base64_encode(&encoded, input, (cs_base64_variant)variant,
                                        (cs_base64_pad_policy)pad) == CS_OK);
            FUZZ_CHECK(cs_base64_decode(&decoded, cs_as_view(&encoded), (cs_base64_variant)variant,
                                        (cs_base64_pad_policy)pad) == CS_OK);
            fuzz_check_equal(cs_as_view(&decoded), input);
            cs_clear(&streamed);
            cs_base64_enc_stream state;
            cs_base64_enc_stream_init(&state, (cs_base64_variant)variant,
                                      (cs_base64_pad_policy)pad);
            size_t split = size == 0 ? 0 : data[0] % (size + 1);
            FUZZ_CHECK(cs_base64_enc_stream_feed(&state, (cs_view){input.data, split}, &streamed) ==
                       CS_OK);
            FUZZ_CHECK(cs_base64_enc_stream_feed(&state,
                                                 (cs_view){input.data + split, size - split},
                                                 &streamed) == CS_OK);
            FUZZ_CHECK(cs_base64_enc_stream_finish(&state, &streamed) == CS_OK);
            fuzz_check_equal(cs_as_view(&streamed), cs_as_view(&encoded));
        }
    }

    for (int mode = CS_URL_PATH; mode <= CS_URL_FORM_URLENCODED; mode++) {
        FUZZ_CHECK(cs_url_encode(&encoded, input, (cs_url_mode)mode) == CS_OK);
        FUZZ_CHECK(cs_url_decode(&decoded, cs_as_view(&encoded), (cs_url_mode)mode) == CS_OK);
        fuzz_check_equal(cs_as_view(&decoded), input);
        FUZZ_CHECK(cs_assign(&decoded, CS_LIT("unchanged")) == CS_OK);
        check_decode(&decoded, cs_url_decode(&decoded, input, (cs_url_mode)mode));
    }

    FUZZ_CHECK(cs_assign(&decoded, CS_LIT("unchanged")) == CS_OK);
    check_decode(&decoded, cs_hex_decode(&decoded, input));
    FUZZ_CHECK(cs_assign(&decoded, CS_LIT("unchanged")) == CS_OK);
    check_decode(&decoded, cs_json_unescape(&decoded, input));
    FUZZ_CHECK(cs_assign(&decoded, CS_LIT("unchanged")) == CS_OK);
    check_decode(&decoded,
                 cs_base64_decode(&decoded, input, CS_BASE64_STANDARD, CS_B64_PAD_OPTIONAL));
    fuzz_check_string(&encoded);
    fuzz_check_string(&streamed);
    cs_destroy(&streamed);
    cs_destroy(&decoded);
    cs_destroy(&encoded);
    return 0;
}
