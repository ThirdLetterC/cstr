#!/usr/bin/env python3
"""
gen_unicode_tables.py — Offline Unicode 17.0.0 Table Generator for cstr (cstr.c)

Extracts synchronized Unicode 17.0.0 UCD properties from the system's ICU 78
(icudt78l, Unicode 17.0.0) data tables in:
  - /usr/lib/chromium/chromium (propsTrie, propsVecTrie, propsVectors, ucase, nfc)
  - /usr/lib/chromium/icudtl.dat (nfkc.nrm, uemoji.icu, brkitr/char.brk)

Generates compact static const C11 tables for:
  - General_Category (30 categories, including all 4,803 new Unicode 17.0.0 characters
    bringing total graphic/format characters to 159,801)
  - Binary properties (Alphabetic, White_Space, Uppercase, Lowercase, Cased,
    Case_Ignorable, Soft_Dotted, Extended_Pictographic)
  - UAX #29 Segmentation properties (Grapheme_Cluster_Break, Word_Break, Sentence_Break,
    and Indic_Conjunct_Break InCB=None/Consonant/Linker/Extend for Rule GB9c)
  - UAX #15 Normalization properties (Canonical_Combining_Class, NFC/NFD/NFKC/NFKD Quick_Check,
    Canonical & Compatibility Decompositions, Canonical Composition pairs excluding
    Full_Composition_Exclusion)
  - Full & Simple Case Mappings (Lowercase, Uppercase, Titlecase, CaseFolding, SpecialCasing,
    including Unicode 17.0.0 CaseFolding.txt updates for U+1FD3, U+1FE3, and U+FB05)
"""

import ctypes
import hashlib
import struct
import sys

MAX_CP = 0x110000

# 1. General Category mapping (matches cs_general_category enum in cstr.h)
GC_MAP = {
    "Cn": 0,
    "Lu": 1,
    "Ll": 2,
    "Lt": 3,
    "Lm": 4,
    "Lo": 5,
    "Mn": 6,
    "Mc": 7,
    "Me": 8,
    "Nd": 9,
    "Nl": 10,
    "No": 11,
    "Pc": 12,
    "Pd": 13,
    "Ps": 14,
    "Pe": 15,
    "Pi": 16,
    "Pf": 17,
    "Po": 18,
    "Sm": 19,
    "Sc": 20,
    "Sk": 21,
    "So": 22,
    "Zs": 23,
    "Zl": 24,
    "Zp": 25,
    "Cc": 26,
    "Cf": 27,
    "Cs": 28,
    "Co": 29,
}

# ICU UCharCategory (0..29) -> cs_general_category (0..29)
ICU_TO_CSTR_GC = {
    0: 0,  # Cn
    1: 1,  # Lu
    2: 2,  # Ll
    3: 3,  # Lt
    4: 4,  # Lm
    5: 5,  # Lo
    6: 6,  # Mn
    7: 8,  # Me -> 8
    8: 7,  # Mc -> 7
    9: 9,  # Nd
    10: 10,  # Nl
    11: 11,  # No
    12: 23,  # Zs
    13: 24,  # Zl
    14: 25,  # Zp
    15: 26,  # Cc
    16: 27,  # Cf
    17: 29,  # Co
    18: 28,  # Cs
    19: 13,  # Pd
    20: 14,  # Ps
    21: 15,  # Pe
    22: 12,  # Pc
    23: 18,  # Po
    24: 19,  # Sm
    25: 20,  # Sc
    26: 21,  # Sk
    27: 22,  # So
    28: 16,  # Pi
    29: 17,  # Pf
}

# ICU UGraphemeClusterBreak (0..17) -> CS_GCB_* (0..14)
ICU_TO_CSTR_GCB = {
    0: 0,  # Other
    1: 3,  # Control
    2: 1,  # CR
    3: 4,  # Extend
    4: 9,  # L
    5: 2,  # LF
    6: 12,  # LV
    7: 13,  # LVT
    8: 11,  # T
    9: 10,  # V
    10: 8,  # SpacingMark
    11: 7,  # Prepend
    12: 6,  # Regional_Indicator
    13: 0,  # E_Base (obsolete -> Other)
    14: 0,  # E_Base_GAZ (obsolete -> Other)
    15: 0,  # E_Modifier (obsolete -> Other)
    16: 0,  # Glue_After_Zwj (obsolete -> Other)
    17: 5,  # ZWJ
}
CS_GCB_EXTPICT = 14

# ICU UWordBreakValues (0..22) -> CS_WB_* (0..18)
ICU_TO_CSTR_WB = {
    0: 0,  # Other
    1: 10,  # ALetter
    2: 7,  # Format
    3: 8,  # Katakana
    4: 14,  # MidLetter
    5: 15,  # MidNum
    6: 16,  # Numeric
    7: 17,  # ExtendNumLet
    8: 1,  # CR
    9: 4,  # Extend
    10: 2,  # LF
    11: 13,  # MidNumLet
    12: 3,  # Newline
    13: 6,  # Regional_Indicator
    14: 9,  # Hebrew_Letter
    15: 11,  # Single_Quote
    16: 12,  # Double_Quote
    17: 0,  # E_Base (obsolete -> Other)
    18: 0,  # E_Base_GAZ (obsolete -> Other)
    19: 0,  # E_Modifier (obsolete -> Other)
    20: 0,  # Glue_After_Zwj (obsolete -> Other)
    21: 5,  # ZWJ
    22: 18,  # WSegSpace
}

# ICU USentenceBreak (0..14) -> CS_SB_* (0..14)
ICU_TO_CSTR_SB = {
    0: 0,  # Other
    1: 11,  # ATerm
    2: 13,  # Close
    3: 5,  # Format
    4: 7,  # Lower
    5: 10,  # Numeric
    6: 9,  # OLetter
    7: 4,  # Sep
    8: 6,  # Sp
    9: 12,  # STerm
    10: 8,  # Upper
    11: 1,  # CR
    12: 3,  # Extend
    13: 2,  # LF
    14: 14,  # SContinue
}


def utf16_to_cps(u16_list):
    cps = []
    i = 0
    while i < len(u16_list):
        u = u16_list[i]
        i += 1
        if (
            0xD800 <= u <= 0xDBFF
            and i < len(u16_list)
            and 0xDC00 <= u16_list[i] <= 0xDFFF
        ):
            cps.append(0x10000 + (((u - 0xD800) << 10) | (u16_list[i] - 0xDC00)))
            i += 1
        else:
            cps.append(u)
    return tuple(cps)


def ucptrie_fast16_get(index_u16, data_u16, high_start, cp):
    if cp <= 0xFFFF:
        return data_u16[index_u16[cp >> 6] + (cp & 0x3F)]
    if cp >= high_start:
        return data_u16[len(data_u16) - 2]
    i1 = (0x10000 >> 6) + ((cp - 0x10000) >> 14)
    i2 = index_u16[index_u16[i1] + ((cp >> 9) & 0x1F)]
    if (i2 & 0x8000) == 0:
        i3 = index_u16[i2 + ((cp >> 4) & 0x1F)]
    else:
        idx3 = (cp >> 4) & 0x1F
        block = (i2 & 0x7FFF) + (idx3 * 9) // 8
        sub = idx3 & 7
        i3 = index_u16[block + 1] | (
            (index_u16[block - sub] << (2 + 2 * sub)) & 0x30000
        )
    return data_u16[i3 + (cp & 0xF)]


class Norm2Decoder78:
    """Decodes ICU 78 (formatVersion 5, Unicode 17.0.0) Normalizer2 tries & extraData."""

    def __init__(self, indexes, index_u16, data_u16, extra_u16, high_start):
        self.indexes = indexes
        self.index_u16 = index_u16
        self.data_u16 = data_u16
        self.extra_u16 = extra_u16
        self.high_start = high_start
        self.minDecompNoCP = indexes[8]
        self.minCompNoMaybeCP = indexes[9]
        self.minYesNo = indexes[10]
        self.minNoNo = indexes[11]
        self.limitNoNo = indexes[12]
        self.minMaybeYes = indexes[13]
        self.minYesNoMappingsOnly = indexes[14]
        self.minNoNoCompBoundaryBefore = indexes[15]
        self.minNoNoCompNoMaybeCC = indexes[16]
        self.minNoNoEmpty = indexes[17]
        self.minLcccCP = indexes[18]
        self.minMaybeNo = indexes[20] if indexes[0] >= 88 else self.minMaybeYes
        self.minMaybeNoCombinesFwd = (
            indexes[21] if indexes[0] >= 88 else self.minMaybeYes
        )
        self.centerNoNoDelta = (self.minMaybeNo >> 3) - 65

    def get_norm16(self, cp):
        if 0xD800 <= cp <= 0xDFFF:
            return 1
        return ucptrie_fast16_get(self.index_u16, self.data_u16, self.high_start, cp)

    def get_ccc(self, cp):
        if cp < self.minLcccCP or (0xD800 <= cp <= 0xDFFF):
            return 0
        norm16 = self.get_norm16(cp)
        if norm16 >= 0xFC00:
            return (norm16 >> 1) & 0xFF
        if norm16 < self.minNoNo or self.limitNoNo <= norm16 < 0xFC00:
            return 0
        idx = norm16 >> 1
        first_unit = self.extra_u16[idx]
        if first_unit & 0x80:
            return self.extra_u16[idx - 1] & 0xFF
        return 0

    def get_raw_decomposition(self, cp):
        if cp < self.minDecompNoCP or (0xD800 <= cp <= 0xDFFF):
            return None
        norm16 = self.get_norm16(cp)
        if norm16 < self.minYesNo or self.minMaybeYes <= norm16:
            return None
        if self.limitNoNo <= norm16 < self.minMaybeNo:
            return (cp + ((norm16 >> 3) - self.centerNoNoDelta),)
        if norm16 == self.minYesNo:
            return None
        if self.minMaybeNo <= norm16 < self.minMaybeYes:
            idx = (self.limitNoNo >> 1) + ((norm16 - self.minMaybeNo) >> 1)
        else:
            idx = norm16 >> 1
        first_unit = self.extra_u16[idx]
        length = first_unit & 0x1F
        if first_unit & 0x40:
            raw_pos = idx - ((first_unit >> 7) & 1) - 1
            rm0 = self.extra_u16[raw_pos]
            if rm0 <= 0x1F:
                return utf16_to_cps(self.extra_u16[raw_pos - rm0 : raw_pos])
            else:
                return utf16_to_cps((rm0,) + self.extra_u16[idx + 3 : idx + 1 + length])
        else:
            return utf16_to_cps(self.extra_u16[idx + 1 : idx + 1 + length])

    def get_quick_check(self, cp):
        if 0xD800 <= cp <= 0xDFFF:
            return 0, 0
        norm16 = self.get_norm16(cp)
        decomp_qc = (
            0
            if (
                cp < self.minDecompNoCP
                or norm16 < self.minYesNo
                or norm16 >= self.minMaybeYes
            )
            else 1
        )
        if cp < self.minCompNoMaybeCP or norm16 < self.minNoNo or norm16 > 0xFE00:
            comp_qc = 0
        elif self.minMaybeNo <= norm16 <= 0xFE00:
            comp_qc = 2
        else:
            comp_qc = 1
        return comp_qc, decomp_qc


def load_nrm_file(buf, off):
    hsize = struct.unpack_from("<H", buf, off)[0]
    base = off + hsize
    ix = struct.unpack_from("<22i", buf, base)
    trie_off = base + ix[0]
    extra_off = base + ix[1]
    _sig, opts, idx_len, data_len_lo, _i3n, _dn_lo, sh_hi = struct.unpack_from(
        "<IHHHHHH", buf, trie_off
    )
    data_len = ((opts & 0xF000) << 4) | data_len_lo
    t_idx = struct.unpack_from(f"<{idx_len}H", buf, trie_off + 16)
    t_dat = struct.unpack_from(f"<{data_len}H", buf, trie_off + 16 + idx_len * 2)
    t_ext = struct.unpack_from(f"<{(ix[2] - ix[1]) // 2}H", buf, extra_off)
    return Norm2Decoder78(ix, t_idx, t_dat, t_ext, sh_hi << 9)


def decode_ucase_cp(icu, trie_ptr, exc_arr, cp):
    props = icu.utrie2_get32_72(trie_ptr, cp)
    utype = props & 3
    is_lower = 1 if utype == 1 else 0
    is_upper = 1 if utype == 2 else 0
    is_cased = 1 if utype != 0 else 0
    is_ci = 1 if (props & 4) != 0 else 0

    if (props & 8) == 0:
        is_sd = 1 if (props & 0x60) == 0x20 else 0
        delta = (props >> 7) if (props < 0x8000) else ((props >> 7) - 512)
        lc = uc = tc = cf = scf = (cp,)
        if utype == 1:
            if delta != 0:
                uc = tc = (cp + delta,)
        elif utype >= 2 and delta != 0:
            lc = cf = scf = (cp + delta,)
        return is_upper, is_lower, is_cased, is_ci, is_sd, lc, uc, tc, cf, scf

    pe = props >> 4
    exc_word = exc_arr[pe]
    pe += 1
    is_sd = 1 if ((exc_word >> 7) & 0x60) == 0x20 else 0
    double_slots = (exc_word & 0x100) != 0

    def get_slot(slot_bit_idx):
        mask = (1 << slot_bit_idx) - 1
        prev_bits = (exc_word & mask).bit_count()
        if double_slots:
            off = pe + prev_bits * 2
            val = (exc_arr[off] << 16) | exc_arr[off + 1]
        else:
            off = pe + prev_bits
            val = exc_arr[off]
        return val, off

    slc = suc = stc = scf_val = cp
    if exc_word & 0x10:
        delta_val, _ = get_slot(4)
        if exc_word & 0x400:
            delta_val = -delta_val
        if utype == 1:
            suc = stc = cp + delta_val
        else:
            slc = scf_val = cp + delta_val

    if exc_word & 0x01:
        slc, _ = get_slot(0)
    if exc_word & 0x04:
        suc, _ = get_slot(2)
    if exc_word & 0x08:
        stc, _ = get_slot(3)
    elif exc_word & 0x04:
        stc = suc

    if exc_word & 0x200:
        scf_val = cp
    elif exc_word & 0x02:
        scf_val, _ = get_slot(1)
    elif exc_word & 0x01:
        scf_val = slc

    lc = (slc,)
    uc = (suc,)
    tc = (stc,)
    cf = (scf_val,)
    scf = (scf_val,)

    if exc_word & 0x80:
        _, full_off = get_slot(7)
        full_len_val = exc_arr[full_off]
        str_off = full_off + (2 if double_slots else 1)
        l_len = full_len_val & 0xF
        f_len = (full_len_val >> 4) & 0xF
        u_len = (full_len_val >> 8) & 0xF
        t_len = (full_len_val >> 12) & 0xF
        if l_len > 0:
            lc = utf16_to_cps(exc_arr[str_off : str_off + l_len])
        str_off += l_len
        if f_len > 0:
            cf = utf16_to_cps(exc_arr[str_off : str_off + f_len])
        str_off += f_len
        if u_len > 0:
            uc = utf16_to_cps(exc_arr[str_off : str_off + u_len])
        str_off += u_len
        if t_len > 0:
            tc = utf16_to_cps(exc_arr[str_off : str_off + t_len])

    # Conditional special casing in Root locale (SpecialCasing.txt / CaseFolding.txt)
    if cp == 0x0130:
        lc = (0x0069, 0x0307)
        cf = (0x0069, 0x0307)
    elif cp == 0x0587:
        uc = (0x0535, 0x0552)
        tc = (0x0535, 0x0582)

    return is_upper, is_lower, is_cased, is_ci, is_sd, lc, uc, tc, cf, scf


def format_int_array(arr, per_line=12, hex_fmt=False):
    lines = []
    for i in range(0, len(arr), per_line):
        chunk = arr[i : i + per_line]
        if hex_fmt:
            s = ", ".join(f"0x{x:X}U" for x in chunk)
        else:
            s = ", ".join(str(x) for x in chunk)
        lines.append("    " + s)
    return ",\n".join(lines)


def generate(out_path):
    icu = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libicuuc.so.72.1")
    icu.utrie2_get32_72.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    icu.utrie2_get32_72.restype = ctypes.c_uint32
    icu.ucptrie_openFromBinary_72.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    icu.ucptrie_openFromBinary_72.restype = ctypes.c_void_p
    icu.ucptrie_get_72.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    icu.ucptrie_get_72.restype = ctypes.c_uint32

    with open("/usr/lib/chromium/chromium", "rb") as f:
        cr = f.read()
    with open("/usr/lib/chromium/icudtl.dat", "rb") as f:
        icudtl = f.read()

    def make_utrie2(raw_bytes, idx_len, data_len, i2null, dnull, hstart, hv_idx):
        buf = ctypes.create_string_buffer(raw_bytes)
        idx_p = ctypes.addressof(buf)
        d16_p = idx_p + idx_len * 2
        st = ctypes.create_string_buffer(64)
        struct.pack_into(
            "<QQQiiHHIIii",
            st,
            0,
            idx_p,
            d16_p,
            0,
            idx_len,
            data_len,
            i2null,
            dnull,
            0,
            0,
            hstart,
            hv_idx,
        )
        return buf, st

    # Extract ICU 78 (Unicode 17.0.0) data blocks
    pt_raw = cr[0x25839E0 : 0x25839E0 + (4628 + 19480) * 2]
    pv_raw = cr[0x258F640 : 0x258F640 + (5924 + 37208) * 2]
    pvec_raw = cr[0x257CF60:0x25839E0]
    uc_raw = cr[0x2576580 : 0x2576580 + (3468 + 10076) * 2]
    exc_raw = cr[0x2575580:0x2576290]
    nfc_raw = cr[0x25581D0 : 0x255D080 + 8162 * 2]
    nfkc_raw = icudtl[7798000 : 7798000 + 56128]
    uemoji_raw = icudtl[9006464 : 9006464 + 14400]
    cbrk_raw = icudtl[536512 : 536512 + 14048]

    _pt_buf, pt_st = make_utrie2(pt_raw, 4628, 19480, 2624, 4756, 0x110000, 24104)
    _pv_buf, pv_st = make_utrie2(pv_raw, 5924, 37208, 2624, 6052, 0x110000, 43128)
    _uc_buf, uc_st = make_utrie2(uc_raw, 3468, 10076, 392, 3592, 0xE0800, 13540)
    exc78_u16 = struct.unpack_from(f"<{len(exc_raw) // 2}H", exc_raw, 0)

    nfc_ix = list(struct.unpack_from("<22i", cr, 0x25581D0))
    nfc_dec78 = Norm2Decoder78(
        nfc_ix,
        struct.unpack_from("<1888H", cr, 0x255C1C0),
        struct.unpack_from("<8162H", cr, 0x255D080),
        struct.unpack_from(f"<{(nfc_ix[2] - nfc_ix[1]) // 2}H", cr, 0x2558230),
        195584,
    )
    nfkc_dec78 = load_nrm_file(icudtl, 7798000)

    err = ctypes.c_int32(0)
    actual = ctypes.c_int32(0)
    e_hsize = struct.unpack_from("<H", uemoji_raw, 0)[0]
    e_ix = struct.unpack_from("<16i", uemoji_raw, e_hsize)
    e_buf = ctypes.create_string_buffer(uemoji_raw[e_hsize + e_ix[0] :])
    e_trie = icu.ucptrie_openFromBinary_72(
        -1, -1, e_buf, len(e_buf), ctypes.byref(actual), ctypes.byref(err)
    )

    t_idx = cbrk_raw.find(b"3irT")
    cb_buf = ctypes.create_string_buffer(cbrk_raw[t_idx:])
    cb_trie = icu.ucptrie_openFromBinary_72(
        -1, -1, cb_buf, len(cbrk_raw) - t_idx, ctypes.byref(actual), ctypes.byref(err)
    )

    # Build 3-stage property trie, case table, and normalization tables in a single pass
    palette = []
    palette_map = {}

    # Seed index 0 with unassigned record
    unassigned_rec = (0, 0, 0, 0, 0, 0, 0, 0)
    palette.append(unassigned_rec)
    palette_map[unassigned_rec] = 0

    cp_palette_idx = [0] * MAX_CP

    case_multi_pool = []
    case_multi_map = {}

    def encode_case_target(cp, seq):
        if seq is None or seq == (cp,):
            return cp
        if len(seq) == 1:
            return seq[0]
        if seq not in case_multi_map:
            case_multi_map[seq] = len(case_multi_pool)
            case_multi_pool.extend(seq)
        offset = case_multi_map[seq]
        length = len(seq)
        assert length <= 7
        return -((offset << 3) | length)

    case_rows = []
    decomp_rows = []
    decomp_pool = []
    decomp_pool_map = {}
    comp_pairs = []

    for cp in range(MAX_CP):
        pt = icu.utrie2_get32_72(pt_st, cp)
        gc = ICU_TO_CSTR_GC[pt & 0x1F]
        pv = icu.utrie2_get32_72(pv_st, cp)
        w1, w2 = struct.unpack_from("<2I", pvec_raw, pv * 4 + 4)
        gcb = ICU_TO_CSTR_GCB[(w2 >> 5) & 0x1F]
        wb = ICU_TO_CSTR_WB[(w2 >> 10) & 0x1F]
        sb = ICU_TO_CSTR_SB[(w2 >> 15) & 0x1F]

        cb = icu.ucptrie_get_72(cb_trie, cp)
        # InCB: 0=None, 1=Consonant (char.brk=11), 2=Linker (char.brk=12), 3=Extend (char.brk=8 or 17)
        incb = 1 if cb == 11 else (2 if cb == 12 else (3 if cb in (8, 17) else 0))

        extpict = 1 if (icu.ucptrie_get_72(e_trie, cp) & 0x20) else 0
        if extpict and gcb == 0:
            gcb = CS_GCB_EXTPICT

        if 0xD800 <= cp <= 0xDFFF:
            u = l = c = ci = sd = 0
            lc = uc = tc = cf = scf = (cp,)
        else:
            u, l, c, ci, sd, lc, uc, tc, cf, scf = decode_ucase_cp(
                icu, uc_st, exc78_u16, cp
            )

        alpha = 1 if (w1 & 0x100) else 0
        wspace = 1 if (w1 & 0x1) else 0
        flags = (
            alpha
            | (wspace << 1)
            | (u << 2)
            | (l << 3)
            | (c << 4)
            | (ci << 5)
            | (sd << 6)
            | (extpict << 7)
        )

        ccc = nfc_dec78.get_ccc(cp)
        nfc_qc, nfd_qc = nfc_dec78.get_quick_check(cp)
        nfkc_qc, nfkd_qc = nfkc_dec78.get_quick_check(cp)
        qc = nfc_qc | (nfd_qc << 2) | (nfkc_qc << 4) | (nfkd_qc << 6)

        rec = (gc, gcb, wb, sb, flags, ccc, qc, incb)
        idx = palette_map.get(rec)
        if idx is None:
            idx = len(palette)
            palette.append(rec)
            palette_map[rec] = idx
        cp_palette_idx[cp] = idx

        # Case mapping row
        if not (0xD800 <= cp <= 0xDFFF) and (
            lc != (cp,) or uc != (cp,) or tc != (cp,) or cf != (cp,) or scf != (cp,)
        ):
            case_rows.append(
                (
                    cp,
                    encode_case_target(cp, lc),
                    encode_case_target(cp, uc),
                    encode_case_target(cp, tc),
                    encode_case_target(cp, cf),
                    encode_case_target(cp, scf),
                )
            )

        # Normalization decomposition & canonical composition rows
        if not (0xAC00 <= cp <= 0xD7A3 or 0xD800 <= cp <= 0xDFFF):
            raw_c = nfc_dec78.get_raw_decomposition(cp)
            if raw_c is not None:
                is_compat = 0
                seq = raw_c
                norm16 = nfc_dec78.get_norm16(cp)
                if nfc_dec78.minYesNo < norm16 < nfc_dec78.minNoNo and len(seq) == 2:
                    pair_key = (seq[0] << 32) | seq[1]
                    comp_pairs.append((pair_key, cp))
            else:
                raw_k = nfkc_dec78.get_raw_decomposition(cp)
                if raw_k is not None:
                    is_compat = 1
                    seq = raw_k
                else:
                    seq = None

            if seq is not None:
                if seq not in decomp_pool_map:
                    decomp_pool_map[seq] = len(decomp_pool)
                    decomp_pool.extend(seq)
                offset = decomp_pool_map[seq]
                length = len(seq)
                assert length < 32
                info = (is_compat << 31) | (length << 24) | offset
                decomp_rows.append((cp, info))

    comp_pairs.sort(key=lambda x: x[0])

    # Build 3-stage trie:
    # Stage 3: blocks of 64 code points (cp & 0x3F)
    # Stage 2: chunks of 64 blocks = 4096 code points ((cp >> 6) & 0x3F)
    # Stage 1: 272 top-level entries (cp >> 12)
    stage3_blocks = []
    stage3_map = {}
    stage2_raw = []

    for blk_idx in range(MAX_CP // 64):
        blk = tuple(cp_palette_idx[blk_idx * 64 : (blk_idx + 1) * 64])
        if blk not in stage3_map:
            stage3_map[blk] = len(stage3_blocks)
            stage3_blocks.extend(blk)
        stage2_raw.append(stage3_map[blk])

    stage2_chunks = []
    stage2_map = {}
    stage1 = []

    for chk_idx in range(len(stage2_raw) // 64):
        chk = tuple(stage2_raw[chk_idx * 64 : (chk_idx + 1) * 64])
        if chk not in stage2_map:
            stage2_map[chk] = len(stage2_chunks) // 64
            stage2_chunks.extend(chk)
        stage1.append(stage2_map[chk])

    assert len(stage1) == 272
    assert len(stage2_chunks) // 64 <= 256

    # Compute SHA-256 digests of ICU 78 (Unicode 17.0.0) UCD data blocks
    ucd_blocks = [
        ("icudt78l/propsTrie", pt_raw),
        ("icudt78l/propsVecTrie", pv_raw),
        ("icudt78l/propsVectors", pvec_raw),
        ("icudt78l/ucase", uc_raw + exc_raw),
        ("icudt78l/nfc.nrm", nfc_raw),
        ("icudt78l/nfkc.nrm", nfkc_raw),
        ("icudt78l/uemoji.icu", uemoji_raw),
        ("icudt78l/brkitr/char.brk", cbrk_raw),
    ]
    checksums = [(name, hashlib.sha256(data).hexdigest()) for name, data in ucd_blocks]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "/* ==========================================================================\n"
        )
        f.write(
            " * AUTO-GENERATED UNICODE 17.0.0 TABLES (gen_unicode_tables.py v2.0.0)\n"
        )
        f.write(" * DO NOT EDIT MANUALLY.\n")
        f.write(" *\n")
        f.write(" * Source UCD SHA-256 Checksums (ICU 78 / Unicode 17.0.0):\n")
        f.writelines(f" *   {name:<26} : {digest}\n" for name, digest in checksums)
        f.write(" *\n")
        f.write(" * UNICODE LICENSE V3 - DATA FILES AND SOFTWARE\n")
        f.write(" * Copyright (c) 1991-2025 Unicode, Inc. All rights reserved.\n")
        f.write(
            " * Distributed under the Terms of Use in https://www.unicode.org/copyright.html.\n"
        )
        f.write(
            " * Permission is hereby granted, free of charge, to any person obtaining\n"
        )
        f.write(
            " * a copy of the Unicode data files and any associated documentation\n"
        )
        f.write(
            ' * (the "Data Files") or Unicode software and any associated documentation\n'
        )
        f.write(' * (the "Software") to deal in the Data Files or Software without\n')
        f.write(
            " * restriction, including without limitation the rights to use, copy,\n"
        )
        f.write(
            " * modify, merge, publish, distribute, and/or sell copies of the Data\n"
        )
        f.write(
            " * Files or Software, and to permit persons to whom the Data Files or\n"
        )
        f.write(" * Software are furnished to do so, provided that either (a) this\n")
        f.write(
            " * copyright and permission notice appear with all copies of the Data\n"
        )
        f.write(
            " * Files or Software, or (b) this copyright and permission notice appear\n"
        )
        f.write(" * in associated Documentation.\n")
        f.write(
            " * ========================================================================== */\n\n"
        )

        f.write("typedef struct {\n")
        f.write("    uint8_t gc;\n")
        f.write("    uint8_t gcb;\n")
        f.write("    uint8_t wb;\n")
        f.write("    uint8_t sb;\n")
        f.write("    uint8_t flags;\n")
        f.write("    uint8_t ccc;\n")
        f.write("    uint8_t qc;\n")
        f.write("    uint8_t incb;\n")
        f.write("} cs_uprop_record;\n\n")

        f.write(f"static const cs_uprop_record cs_uprop_palette[{len(palette)}] = {{\n")
        f.writelines(f"    {{{r[0]}, {r[1]}, {r[2]}, {r[3]}, 0x{r[4]:02X}U, {r[5]}, 0x{r[6]:02X}U, {r[7]}}}, /* {i} */\n" for i, r in enumerate(palette))
        f.write("};\n\n")

        f.write(f"static const uint8_t cs_uprop_stage1[{len(stage1)}] = {{\n")
        f.write(format_int_array(stage1, 16))
        f.write("\n};\n\n")

        f.write(f"static const uint16_t cs_uprop_stage2[{len(stage2_chunks)}] = {{\n")
        f.write(format_int_array(stage2_chunks, 12))
        f.write("\n};\n\n")

        f.write(f"static const uint16_t cs_uprop_stage3[{len(stage3_blocks)}] = {{\n")
        f.write(format_int_array(stage3_blocks, 16))
        f.write("\n};\n\n")

        f.write("typedef struct {\n")
        f.write("    uint32_t cp;\n")
        f.write("    int32_t  lc;\n")
        f.write("    int32_t  uc;\n")
        f.write("    int32_t  tc;\n")
        f.write("    int32_t  cf;\n")
        f.write("    int32_t  scf;\n")
        f.write("} cs_ucase_entry;\n\n")

        f.write(f"#define CS_UCASE_COUNT {len(case_rows)}U\n")
        f.write(f"static const cs_ucase_entry cs_ucase_table[{len(case_rows)}] = {{\n")
        f.writelines(f"    {{0x{r[0]:04X}U, {r[1]}, {r[2]}, {r[3]}, {r[4]}, {r[5]}}},\n" for r in case_rows)
        f.write("};\n\n")

        f.write(
            f"static const uint32_t cs_ucase_multi_pool[{len(case_multi_pool)}] = {{\n"
        )
        f.write(format_int_array(case_multi_pool, 12, hex_fmt=True))
        f.write("\n};\n\n")

        f.write("typedef struct {\n")
        f.write("    uint32_t cp;\n")
        f.write("    uint32_t info;\n")
        f.write("} cs_udecomp_entry;\n\n")

        f.write(f"#define CS_UDECOMP_COUNT {len(decomp_rows)}U\n")
        f.write(
            f"static const cs_udecomp_entry cs_udecomp_table[{len(decomp_rows)}] = {{\n"
        )
        for cp, info in decomp_rows:
            f.write(f"    {{0x{cp:04X}U, 0x{info:08X}U}},\n")
        f.write("};\n\n")

        f.write(f"static const uint32_t cs_udecomp_pool[{len(decomp_pool)}] = {{\n")
        f.write(format_int_array(decomp_pool, 12, hex_fmt=True))
        f.write("\n};\n\n")

        f.write("typedef struct {\n")
        f.write("    uint64_t pair;\n")
        f.write("    uint32_t composed;\n")
        f.write("} cs_ucomp_entry;\n\n")

        f.write(f"#define CS_UCOMP_COUNT {len(comp_pairs)}U\n")
        f.write(f"static const cs_ucomp_entry cs_ucomp_table[{len(comp_pairs)}] = {{\n")
        for pair_key, comp_cp in comp_pairs:
            f.write(f"    {{0x{pair_key:016X}ULL, 0x{comp_cp:04X}U}},\n")
        f.write("};\n")

    b_palette = len(palette) * 8
    b_s1 = len(stage1) * 1
    b_s2 = len(stage2_chunks) * 2
    b_s3 = len(stage3_blocks) * 2
    b_case = len(case_rows) * 24 + len(case_multi_pool) * 4
    b_decomp = len(decomp_rows) * 8 + len(decomp_pool) * 4
    b_comp = len(comp_pairs) * 16
    total_b = b_palette + b_s1 + b_s2 + b_s3 + b_case + b_decomp + b_comp
    print(f"Generated {out_path} (Unicode 17.0.0) successfully.")
    print(
        f"  Property trie + palette : {(b_palette + b_s1 + b_s2 + b_s3) / 1024:.1f} KiB ({len(palette)} palette entries)"
    )
    print(
        f"  Case mappings + pool    : {b_case / 1024:.1f} KiB ({len(case_rows)} entries)"
    )
    print(
        f"  Decompositions + pool   : {b_decomp / 1024:.1f} KiB ({len(decomp_rows)} entries)"
    )
    print(
        f"  Canonical compositions  : {b_comp / 1024:.1f} KiB ({len(comp_pairs)} entries)"
    )
    print(f"  Total binary .rodata    : {total_b / 1024:.1f} KiB ({total_b} bytes)")


if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "unicode_tables.inc"
    generate(out_file)
