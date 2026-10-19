# API guide

[Back to README](../README.md) · [Memory and API contracts](contracts.md)

This guide lists the main function groups. See [cstr.h](../cstr.h) for full declarations and argument types.

## Units and ranges

Lengths and capacities are in bytes unless a function name says otherwise. A code point is a Unicode value. A grapheme cluster is a group of code points that usually appears as one visible character.

Ranges use `[begin, end)`: they include `begin` and exclude `end`. Search results and iterator positions are byte offsets in the original text, even when matching uses Unicode rules.

In the cost notes below, `n` is the input length, `m` is the added text or search pattern length, and `p` is the regex pattern size. `O(n)` means work grows with input length; `O(1)` means constant work or storage.

## Lifecycle and memory

Functions: `cs_init`, `cs_init_with_allocator`, `cs_from_view`, `cs_from_cstr`, `cs_from_bytes`, `cs_clone`, `cs_move`, `cs_destroy`, `cs_clear`, `cs_reset`, `cs_reserve`, `cs_shrink_to_fit`, `cs_assign`, `cs_swap`, `cs_detach`, `cs_adopt`.

Sizes are in bytes (`size_t`). Copying takes `O(n)` time. Capacity grows by about 1.5 times, which spreads allocation costs across repeated appends.

## Building and changing strings

Functions: `cs_append`, `cs_append_byte`, `cs_append_codepoint`, `cs_append_many`, `cs_prepend`, `cs_insert_bytes`, `cs_erase_bytes`, `cs_replace_range_bytes`, `cs_truncate_bytes/codepoints/graphemes`, `cs_resize_bytes`, `cs_repeat`, `cs_builder_*`.

Suffixes such as `_bytes`, `_codepoints`, and `_graphemes` identify the unit. Operations that copy or shift content generally take `O(n + m)` time. `cs_repeat` copies blocks with `memcpy`, doubling the filled block as it goes.

## Access, slices, and indexes

Functions: `cs_len_bytes/codepoints/graphemes`, `cs_byte_at`, `cs_codepoint_at`, `cs_grapheme_at`, `cs_front_*`, `cs_back_*`, `cs_slice_bytes/codepoints/graphemes`, `cs_slice_bytes_clamped`, `cs_take_*`, `cs_drop_*`, `cs_offset_to_*`, `cs_index_*`.

Slices use half-open ranges in the named unit. Coordinate conversions use byte offsets. Byte access takes `O(1)` time. Code point and grapheme access can take `O(n)` without an index; a prebuilt `cs_index` provides `O(1)` access to stored boundaries.

## Iterators

Functions: `cs_byte_iter_*`, `cs_cp_iter_*`, `cs_grapheme_iter_*`, `cs_word_iter_*`, `cs_sentence_iter_*`, `cs_split_iter_*`.

Iterators return an item and its byte range, `[out_start, out_end)`, in the source. They use constant storage without allocations and scan the input in `O(n)` time.

Unicode segmentation follows UAX #29. Grapheme iteration supports GB9c for Indic conjuncts (`Indic_Conjunct_Break`) and GB11 for emoji sequences joined by a zero-width joiner (ZWJ).

## Search and comparison

Functions: `cs_equal[_mode]`, `cs_compare[_mode]`, `cs_compare_natural`, `cs_compare_collation`, `cs_starts_with[_mode]`, `cs_ends_with[_mode]`, `cs_contains[_mode]`, `cs_find[_mode]`, `cs_rfind[_mode]`, `cs_find_byte`, `cs_rfind_byte`, `cs_find_any_byte`, `cs_find_not_any_byte`, `cs_count[_mode]`, `cs_find_all`, `cs_searcher_*`.

Search positions are byte offsets in the original text. The Two-Way substring search algorithm takes `O(n + m)` time and constant extra storage.

## Splitting and joining

Functions: `cs_split_into`, `cs_rsplit_into`, `cs_split_once`, `cs_rsplit_once`, `cs_partition`, `cs_rpartition`, `cs_split_ascii_whitespace_*`, `cs_split_unicode_whitespace_*`, `cs_split_lines_*`, `cs_join`, `cs_join_into`, `cs_join_buffer`.

Split results are `cs_view` slices of the source, so they do not copy the text. Line splitting treats CRLF as one separator. Splitting takes `O(n)` time. Joining calculates the required size before copying.

## Text changes and normalization

Functions: `cs_trim_*`, `cs_replace_first/all`, `cs_remove_all`, `cs_translate_codepoints`, `cs_to_lower/upper/casefold/title_unicode`, `cs_capitalize_unicode`, `cs_pad_start/end_*`, `cs_center_*`, `cs_reverse_bytes/codepoints/graphemes`, `cs_expand_tabs`, `cs_indent`, `cs_dedent`, `cs_normalize_newlines`, `cs_normalize`, `cs_is_normalized`, `cs_norm_quick_check`, `cs_strip_diacritics`.

Case conversion supports `CS_LOCALE_ROOT`, `CS_LOCALE_TR` (Turkish), `CS_LOCALE_AZ` (Azerbaijani), and `CS_LOCALE_LT` (Lithuanian), plus the Greek final sigma rule.

Normalization supports NFC, NFD, NFKC, and NFKD under UAX #15. Processing takes linear time with canonical reordering bounded by `CS_MAX_COMBINING_RUN = 1024`.

## Unicode and encodings

Functions: `cs_validate_utf8`, `cs_decode_utf8_next/prev`, `cs_encode_utf8_codepoint`, `cs_repair_utf8`, `cs_utf8_to_utf16[_bytes]`, `cs_utf16[_bytes]_to_utf8`, `cs_utf8_to_utf32[_bytes]`, `cs_utf32[_bytes]_to_utf8`, `cs_cp_*`, `cs_is_*`.

`CS_DECODE_STRICT` rejects invalid input. `CS_DECODE_REPLACE` replaces each invalid maximal subpart with `U+FFFD`, the replacement character. Conversion supports little-endian (LE) and big-endian (BE) data and byte order marks (BOM).

These operations take `O(n)` time. SIMD speeds up scanning through ASCII bytes.

## Formatting and numbers

Functions: `cs_format_printf`, `cs_append_printf`, `cs_format_args`, `cs_parse_i64/u64/f64/bool_{exact,prefix}`, `cs_format_i64/u64/f64/bool`.

Number handling uses `.` as the decimal separator regardless of the process locale. Integer bases range from 2 to 36. Double formatting produces the shortest text that parses back to the same value. Work grows with the length of the number or format.

## Codecs and streaming

Functions: `cs_hex_encode/decode[_buffer]`, `cs_base64_encode/decode[_buffer]`, `cs_json_escape/unescape`, `cs_url_encode/decode`, `cs_html_escape/unescape`, `cs_c_escape/unescape`, `cs_utf8_stream_*`, `cs_base64_enc/dec_stream_*`.

Base64 supports standard and URL-safe alphabets and configurable `=` padding. JSON decoding handles surrogate pairs such as `\uD83D\uDE00`. URL encoding supports `PATH`, `QUERY`, and `FORM_URLENCODED` modes.

Codecs take `O(n)` time. Streaming APIs keep a fixed amount of state between chunks, including pending input bytes and bookkeeping fields.

## Patterns and hashing

Functions: `cs_glob_match`, `cs_regex_compile/find/find_all/split/replace`, `cs_hash64`, `cs_hash64_seed`, `cs_hash_keyed`, `cs_hash_seed_from_os`.

Regex matching uses a Pike VM / Thompson NFA to avoid catastrophic backtracking. It takes `O(n × p)` time and uses limits set through `cs_regex_limits`.

Hashing takes `O(n)` time. The library provides deterministic xxHash64 and keyed SipHash-2-4.

Names containing `*`, bracketed suffixes, or slash-separated endings above are shorthand for related functions, not literal C identifiers.
