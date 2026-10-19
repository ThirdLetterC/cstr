#include "fuzz_check.h"

/* Compare byte mutations against a separate fixed-size reference buffer. */
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size > 4096)
        return 0;
    char model[8192];
    size_t len = 0;
    cs_string s;
    cs_init(&s);

    for (size_t i = 0; i < size;) {
        uint8_t op = data[i++];
        size_t count = (size - i < 16) ? size - i : 16;
        cs_view part = {(const char *)data + i, count};
        size_t pos = len == 0 ? 0 : op % (len + 1);
        switch (op % 8) {
        case 0:
            if (len + count > sizeof(model))
                break;
            FUZZ_CHECK(cs_append(&s, part) == CS_OK);
            memcpy(model + len, part.data, count);
            len += count;
            break;
        case 1:
            if (len + count > sizeof(model))
                break;
            FUZZ_CHECK(cs_prepend(&s, part) == CS_OK);
            memmove(model + count, model, len);
            memcpy(model, part.data, count);
            len += count;
            break;
        case 2:
            if (len + count > sizeof(model))
                break;
            FUZZ_CHECK(cs_insert_bytes(&s, pos, part) == CS_OK);
            memmove(model + pos + count, model + pos, len - pos);
            memcpy(model + pos, part.data, count);
            len += count;
            break;
        case 3: {
            size_t end = pos + ((len - pos < count) ? len - pos : count);
            FUZZ_CHECK(cs_erase_bytes(&s, pos, end) == CS_OK);
            memmove(model + pos, model + end, len - end);
            len -= end - pos;
            break;
        }
        case 4: {
            size_t end = pos + ((len - pos < count) ? len - pos : count);
            if (len - (end - pos) + count > sizeof(model))
                break;
            FUZZ_CHECK(cs_replace_range_bytes(&s, pos, end, part) == CS_OK);
            memmove(model + pos + count, model + end, len - end);
            memcpy(model + pos, part.data, count);
            len = len - (end - pos) + count;
            break;
        }
        case 5: {
            size_t take = (len - pos < count) ? len - pos : count;
            if (len + take > sizeof(model))
                break;
            cs_view alias = {s.data + pos, take};
            FUZZ_CHECK(cs_append(&s, alias) == CS_OK);
            memcpy(model + len, model + pos, take);
            len += take;
            break;
        }
        case 6: {
            cs_string copy;
            cs_init(&copy);
            FUZZ_CHECK(cs_clone(&copy, &s) == CS_OK);
            fuzz_check_equal(cs_as_view(&copy), (cs_view){model, len});
            cs_destroy(&copy);
            FUZZ_CHECK(cs_reserve(&s, len + count) == CS_OK);
            break;
        }
        case 7:
            FUZZ_CHECK(cs_assign(&s, part) == CS_OK);
            memcpy(model, part.data, count);
            len = count;
            break;
        }
        i += count;
        fuzz_check_string(&s);
        fuzz_check_equal(cs_as_view(&s), (cs_view){model, len});
    }

    /* Compare exact search with a straightforward byte-by-byte reference. */
    cs_view needle = {(const char *)data, size < 8 ? size : 8};
    size_t expected = SIZE_MAX;
    if (needle.len <= len) {
        for (size_t i = 0; i <= len - needle.len; i++) {
            if (needle.len == 0 || memcmp(model + i, needle.data, needle.len) == 0) {
                expected = i;
                break;
            }
        }
    }
    size_t found = SIZE_MAX;
    cs_status status = cs_find(cs_as_view(&s), needle, 0, &found);
    FUZZ_CHECK(status == (expected == SIZE_MAX ? CS_NOT_FOUND : CS_OK));
    if (status == CS_OK)
        FUZZ_CHECK(found == expected);
    cs_destroy(&s);
    return 0;
}
