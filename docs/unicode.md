# Unicode data and binary size

[Back to README](../README.md) · [Build and test](building.md)

## Unicode support

`cstr` includes Unicode **17.0.0** tables covering 159,801 graphic and format characters. These include Sidetic, Tolong Siki, Beria Erfe, Tai Yo, and CJK Unified Ideographs Extension J.

The tables support character properties, case conversion, normalization, and text boundaries. Grapheme rules include UAX #29 GB9c for Indic conjuncts.

The tables are embedded in [cstr.c](../cstr.c), so normal builds do not need Unicode data files at runtime. [gen_unicode_tables.py](../gen_unicode_tables.py), version 2.0.0, generates tables from synchronized Unicode 17.0.0 data in ICU 78 (`icudt78l`). See the [generator requirements](building.md#generate-unicode-tables) before running it.

## Embedded tables

These are the sizes reported with the original documentation. KiB means 1,024 bytes.

| Table in read-only storage (`.rodata`) | Contents | Reported size |
| :--- | :--- | ---: |
| Property trie: `cs_uprop_stage1/2/3` and a palette of 260 unique property sets | A three-level lookup for all `0x110000` code points. Includes general category; `Alpha`, `WSpace`, `Upper`, `Lower`, `Cased`, `CI`, `SD`, and `ExtPict` flags; grapheme, word, sentence, and Indic conjunct break properties; canonical combining class; and NFC/NFD/NFKC/NFKD quick checks. Indic conjunct values are `None`, `Consonant`, `Linker`, and `Extend`. | 67.5 KiB |
| Case mappings: `cs_ucase_table` and `cs_ucase_multi_pool` | Full and simple lowercase, uppercase, titlecase, and case-fold mappings (`lc`, `uc`, `tc`, `cf`, `scf`). Contains 3,037 entries and expansions from one code point to two or three. Includes Beria Erfe and Unicode 17.0.0 case-fold updates. | 72.6 KiB |
| Decomposition: `cs_udecomp_table` and `cs_udecomp_pool` | 5,914 canonical and compatibility decomposition entries for UAX #15. Hangul decomposition uses an algorithm instead of table entries. | 71.4 KiB |
| Composition: `cs_ucomp_table` | 949 canonical composition pairs, respecting `Full_Composition_Exclusion`. | 14.8 KiB |
| **All Unicode tables with `CS_ENABLE_UNICODE=1`** | Full Unicode profile. | **226.4 KiB (231,836 bytes)** |

## Reported object sizes

These figures describe the original x86-64 build. Compiler, platform, and feature settings can change the result.

| Build | Included features | Reported size |
| :--- | :--- | :--- |
| Full `cstr.o`, `-O2`, x86-64 | Code, SIMD, regex, codecs, and Unicode 17.0.0 tables. | 357.8 KiB: 366,351 bytes in the reported `.text` total, 8 bytes `.data`, and 44 bytes `.bss`. |
| Minimal profile, `-Os`, Unicode/regex/SIMD disabled | Strings, buffers, scalar search, formatting, codecs, and hashing. | 69.9 KiB: 71,627 bytes in the reported `.text` total, 8 bytes `.data`, and 44 bytes `.bss`. |
