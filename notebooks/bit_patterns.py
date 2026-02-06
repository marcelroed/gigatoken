# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "ty==0.0.1a15",
# ]
# ///

# To run, install `uv` and do `uvx marimo edit bit_patterns.py`

import marimo

__generated_with = "0.17.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from __future__ import annotations

    from dataclasses import dataclass, field

    class Sentence:
        s: str
        bits: dict[str, list[bool]]

        def __init__(self, s: str):
            object.__setattr__(self, "s", s)
            object.__setattr__(self, "bits", {})

        def __setattr__(self, name: str, bits: list[bool]):
            self.bits[name] = bits

        def __getattr__(self, name):
            return self.bits[name]

    class Bits:
        def __init__(self, bits: list[bool]):
            self.bits = bits

        def __iter__(self):
            return iter(self.bits)

        def __or__(self, other: Bits) -> Bits:
            assert len(self) == len(other)
            return Bits([a or b for a, b in zip(self, other)])

        def __and__(self, other: Bits) -> Bits:
            assert len(self) == len(other)
            return Bits([a and b for a, b in zip(self, other)])

        def __invert__(self) -> Bits:
            return Bits([not a for a in self])

        def __getitem__(self, idx):
            return self.bits[idx]

        def __setitem__(self, idx, value):
            self.bits[idx] = value

        def __lshift__(self, other: int) -> Bits:
            return Bits(self.bits[other:] + [False] * other)

        def shl(self, count: int, fill: bool) -> Bits:
            return Bits(self.bits[count:] + [fill] * count)

        def __rshift__(self, other: int) -> Bits:
            return Bits([False] * other + self.bits[:-other])

        def shr(self, count: int, fill: bool) -> Bits:
            return Bits([fill] * count + self.bits[:-count])

        def __len__(self):
            return len(self.bits)

        @classmethod
        def from_condition(cls, condition, sequence) -> Bits:
            return cls([condition(e) for e in sequence])

    class Lanes:
        def __init__(self, lanes: list[int]):
            self.lanes = lanes

        def __iter__(self):
            return iter(self.lanes)

        def __getitem__(self, idx):
            return self.lanes[idx]

        def __setitem__(self, idx, value):
            self.lanes[idx] = value

        def __lshift__(self, other: int) -> Lanes:
            return Lanes(self.lanes[other:] + [0] * other)

        def shl(self, count: int, fill: bool) -> Lanes:
            return Lanes(self.lanes[count:] + [fill] * count)

        def __rshift__(self, other: int) -> Lanes:
            return Lanes([0] * other + self.lanes[:-other])

        def shr(self, count: int, fill: bool) -> Lanes:
            return Lanes([fill] * count + self.lanes[:-count])

        def apply(self, f) -> Lanes:
            return Lanes([f(b) for b in self.lanes])

        def __len__(self):
            return len(self.lanes)
        def __str__(self):
            return ' '.join(f'{b:02X}' for b in self.lanes)
    return Bits, Lanes, Sentence


@app.cell
def _():
    strings = [
        "What'lls that be?  'll",
        "I have888 of   \"'()'ll them",
    ]
    return (strings,)


@app.cell
def _(
    Sentence,
    class_pat,
    colored_string,
    contraction,
    mo,
    show_sentence,
    strings,
):
    def process_sentence(string: str):
        s = Sentence(string)
        s.space = class_pat(lambda x: x == " ", s.s)
        s.prev_space = s.space >> 1
        s.whitespace = class_pat(str.isspace, s.s)
        s.prev_whitespace = s.whitespace >> 1

        s.letter = class_pat(str.isalpha, s.s)
        s.prev_letter = s.letter >> 1

        s.number = class_pat(str.isnumeric, s.s)
        s.prev_number = s.number >> 1

        s.other = ~(s.letter | s.whitespace | s.number)
        s.prev_other = s.other >> 1

        s.contraction = contraction(s.s) & ~s.prev_space & ~s.prev_other
        s.contraction_end = (
            s.contraction >> 3
        )  # TODO(marcelroed): Also handle the length 2 case

        s.letter_start_naive = s.letter & ~s.prev_letter & ~(s.contraction >> 1) | (
            s.contraction_end & s.letter
        )

        s.letter_start_preceded_by_space = s.prev_space & s.letter_start_naive
        s.letter_start_not_preceded_by_space = (
            s.letter_start_naive & ~s.letter_start_preceded_by_space
        )

        s.letter_start = (
            s.letter_start_preceded_by_space << 1
        ) | s.letter_start_not_preceded_by_space
        s.letter_end = s.prev_letter & ~s.letter
        process_numbers_other(s)

        return mo.vstack(
            [
                show_sentence(s),
                mo.Html(colored_string(s.s, s.section_start, s.section_end)),
            ]
        )

    def process_numbers_other(s: Sentence):
        s.number_start_naive = s.number & ~s.prev_number
        s.number_end = s.prev_number & ~s.number
        s.number_start_preceded_by_space = s.prev_space & s.number_start_naive
        s.number_start_not_preceded_by_space = (
            s.number_start_naive & ~s.number_start_preceded_by_space
        )
        s.number_start = (
            s.number_start_preceded_by_space << 1
        ) | s.number_start_not_preceded_by_space

        s.other_start_naive = s.other & ~s.prev_other
        s.other_end = s.prev_other & ~s.other & ~(s.contraction >> 1)
        s.other_start_preceded_by_space = s.prev_space & s.other_start_naive
        s.other_start_not_preceded_by_space = (
            s.other_start_naive & ~s.other_start_preceded_by_space
        )
        s.other_start = (
            s.other_start_preceded_by_space << 1
        ) | s.other_start_not_preceded_by_space

        s.whitespace_start = s.whitespace & ~s.prev_whitespace
        s.whitespace_end = (s.prev_whitespace & ~s.whitespace) << 1

        s.section_start = (
            s.letter_start
            | s.number_start
            | s.other_start
            | s.whitespace_start
            | s.contraction
        )
        s.section_start[0] = True
        s.section_end = s.letter_end | s.number_end | s.other_end | s.whitespace_end
        # s.section_end[-1] = True

    mo.hstack([process_sentence(s) for s in strings], justify="center")
    return


@app.cell
def _(Bits):
    def class_pat(f, s: str):
        return Bits.from_condition(f, s)

    def contraction(s: str):
        bits = [0] * len(s)
        i = 0
        while (location := s.find("'ll", i)) != -1:
            bits[location] = True
            i = location + 1
        return Bits(bits)

    # def shift_left(bits: list[bool], fill=False):
    #     return bits[1:] + [fill]
    # def shift_right(bits: list[bool], fill=False):
    #     return [fill] + bits[:-1]
    # def bitand(a: list[bool], b: list[bool]) -> list[bool]:
    #     return [ae and be for ae, be in zip(a, b)]
    # def bitnot(a: list[bool]) -> list[bool]:
    #     return [not ae for ae in a]
    # def bitor(a: list[bool], b: list[bool]) -> list[bool]:
    #     return [ae or be for ae, be in zip(a, b)]
    return class_pat, contraction


@app.cell
def _(Bits, Sentence, mo):
    def colored_string(s: str, starts: Bits, ends: Bits):
        colors = ["red", "green", "blue"]
        color_i = 0
        s_list = ['<code style="white-space:pre;text-align: right;">']
        for c, start, end in zip(s, starts, ends):
            if end:
                s_list.append("</span>")
            if start:
                s_list.append(f'<span style="background: {colors[color_i]};">')
                color_i += 1
                color_i %= len(colors)
            s_list.append(c)
        s_list.append("</code>")
        return "".join(s_list)

    def show_sentence(sentence: Sentence):
        max_name_length = max([len(name) for name in sentence.bits.keys()], default=0)
        out_str = [" " * (max_name_length + 2) + sentence.s]
        for name, bits in sentence.bits.items():
            bitstring = "".join("1" if bit else "0" for bit in bits)
            out_str.append(f"{name: >{max_name_length}}: {bitstring}")
        return mo.md(f"```\n{'\n'.join(out_str + [out_str[0]])}\n```")
    return colored_string, show_sentence


@app.cell
def _(mo):
    mo.md(r"""
    ### Notes
    It seems like we need 3 tokens from the past, and 3 tokens into the future, both for finding contractions.
    Maybe then we just load 8 tokens extra and slap 4 onto each side?
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### New Notes
    It seems like we can handle the unicode case with not too much of a hassle.
    If we enumerate all the unicode code points in utf-32, they break down into ~910 ranges of Letter/Number/Other general category groups. We can handle this size of classification within SIMDJson style of processing.
    """)
    return


@app.cell
def _():
    # def length(b: int):
    #     if b < 0x80:  # 0yyyzzzz
    #         return 0
    #     if (b & 0b11100000) == 0b1100000:  # 110xxxyy
    #         return 1
    #     if (b & 0b11110000) == 0b1110000:  # 1110wwww
    #         return 2
    #     if (b & 0b11111000) == 0b11110000:  # 11110uvv
    #         return 3
    #     return 4 # (invalid)
    # len_tbl = [
    #     length(i)
    #     for i in range(256)
    # ]
    return


@app.cell
def _(Lanes, len_tbl, resolve_lead_u8):
    def classify_bytes():
        s = 'Here is some tex2t tࠀhat uses ²ünîcøde 𐍅 brrr'
        bytes = s.encode('utf-8')
        b0 = Lanes([e for e in bytes])

        b1 = b0 << 1
        b2 = b0 << 2
        b3 = b0 << 3

        i0 = b0.apply(lambda b: b >> 2)
        i1 = b1.apply(lambda b: b >> 2)
        # kind = Lanes([unpack_primary(primary[h0][h1])[0] for h0, h1 in zip(i0, i1)])
        # ident = Lanes([unpack_primary(primary[h0][h1])[1] for h0, h1 in zip(i0, i1)])
        lens = b0.apply(lambda b: len_tbl[b])
        print(b0)
        print(b1)
        print(b2)
        print(f'{b0   = !s}')
        print(f'{b1   = !s}')
        print(f'{i0   = !s}')
        print(f'{i1   = !s}')
        text = ''.join([f'{c:<{len(c.encode('utf-8') * 3)}}' for c in s])
        print(f'{text = !s}')
        # print(f'{kind = !s}')
        # print(f'{ident= !s}')
        fcls = Lanes([resolve_lead_u8(b0e, b1e, b2e, b3e) for b0e, b1e, b2e, b3e in zip(b0, b1, b2, b3)])
        print(f'{fcls = !s}')
        print(f'{lens = !s}')

    classify_bytes()
    return


@app.cell(hide_code=True)
def _():
    # # Generates SIMD-friendly tables for classifying UTF-8 bytes into L/N/Z/O
    # # using a primary table keyed by (b0>>2, b1>>2) and small refinement tables.
    # #
    # # Output:
    # #  - len_tbl[256]: 0=continuation, 1..4=sequence length if lead, 5=illegal lead
    # #  - ascii_class[128]: 0=L,1=N,2=O
    # #  - primary[64][64]: packed u16 entries (kind + id/class)
    # #  - ref2_rows, ref3_rows, ref3low_rows, ref4_rows, ref4low_rows
    # 
    # from unicodedata import category
    # 
    # # ---------- Config / encoding for table entries ----------
    # 
    # CLASS_L, CLASS_N, CLASS_Z, CLASS_O = 0, 1, 2, 3
    # 
    # KIND_CLASS   = 0  # primary entry is a final class (id=CLASS_*)
    # KIND_REF2    = 1  # points to ref2_rows[id], 4-way keyed by (b1&3)
    # KIND_REF3    = 2  # points to ref3_rows[id], 16-way keyed by (b2>>4)
    # KIND_REF3LOW = 3  # points to ref3low_rows[id], 16-way keyed by (b2&0xF)
    # KIND_REF4    = 4  # points to ref4_rows[id], 16-way keyed by (b3>>4)
    # KIND_REF4LOW = 5  # points to ref4low_rows[id], 16-way keyed by (b3&0xF)
    # 
    # # Pack a primary cell: 3-bit kind, 13-bit id/class
    # def pack_primary(kind, ident):
    #     assert 0 <= kind <= 7 and 0 <= ident <= 0x1FFF
    #     return (kind & 7) | (ident << 3)
    # 
    # def get_kind_ident(primary_value):
    #     kind = primary_value & 7
    #     ident = primary_value >> 3
    #     return kind, ident
    # 
    # # Utility to classify by Unicode general category
    # def cp_class(cp):
    #     if 0xD800 <= cp <= 0xDFFF:  # surrogate
    #         return CLASS_O
    #     cat = category(chr(cp))
    #     if cat[0] == 'L':
    #         return CLASS_L
    #     elif cat[0] == 'N':
    #         return CLASS_N
    #     elif cat[0] == 'Z':
    #         return CLASS_Z
    #     else:
    #         return CLASS_O  # includes Cn (unassigned), marks, punctuation, symbols, controls
    # 
    # # ---------- Build the UTF-8 sequence → class map ----------
    # 
    # # seq_map: keys are tuples of bytes, values are CLASS_*
    # seq_map = {}
    # 
    # for cp in range(0x110000):
    #     cls = cp_class(cp)
    #     # Encode to UTF-8 (Python guarantees legal UTF-8 for scalar values)
    #     try:
    #         b = chr(cp).encode('utf-8', 'strict')
    #         seq_map[tuple(b)] = cls
    #     except UnicodeEncodeError:
    #         pass
    # 
    # # ---------- ASCII table (fast path) ----------
    # ascii_class = [CLASS_O]*128
    # for b in range(0x00, 0x80):
    #     # one-byte UTF-8: (b,) must exist in seq_map
    #     cls = seq_map.get((b,), CLASS_O)
    #     ascii_class[b] = cls
    # 
    # # ---------- len_tbl for every first byte ----------
    # # 0=cont, 1..4=lead length, 5=illegal lead
    # len_tbl = [5]*256
    # for b in range(256):
    #     if 0x80 <= b <= 0xBF:
    #         len_tbl[b] = 0  # continuation
    #     elif b <= 0x7F:
    #         len_tbl[b] = 1
    #     elif 0xC2 <= b <= 0xDF:
    #         len_tbl[b] = 2
    #     elif 0xE0 <= b <= 0xEF:
    #         len_tbl[b] = 3
    #     elif 0xF0 <= b <= 0xF4:
    #         len_tbl[b] = 4
    #     else:
    #         len_tbl[b] = 5  # illegal lead: C0,C1,F5..FF
    # 
    # # ---------- Raw buckets for primary cells ----------
    # # We index primary cells by quarter-bytes: i0 = b0>>2, i1 = b1>>2  (64 x 64)
    # # For each cell we accumulate per-length raw data so we can compress later.
    # 
    # from collections import defaultdict
    # 
    # # For len=2: cell -> dict[low2 (0..3) -> class]
    # raw2 = defaultdict(lambda: {k: None for k in range(4)})
    # 
    # # For len=3: cell -> dict[low2 -> dict[b2 -> class]]
    # raw3 = defaultdict(lambda: {k: defaultdict(lambda: None) for k in range(4)})
    # 
    # # For len=4: cell -> dict[low2 -> dict[b2 -> dict[b3 -> class]]]
    # raw4 = defaultdict(lambda: {k: defaultdict(lambda: defaultdict(lambda: None)) for k in range(4)})
    # 
    # def cell_index(b0, b1):
    #     return (b0 >> 2, b1 >> 2)  # 0..63 each
    # 
    # # Feed the raw structures from seq_map
    # for seq, cls in seq_map.items():
    #     if len(seq) == 1:
    #         continue  # handled in ascii
    #     b0, b1 = seq[0], seq[1]
    #     i = cell_index(b0, b1)
    #     low2 = b1 & 0x3
    #     if len(seq) == 2:
    #         if (b0, b1) == (0xc2, 0xb2):
    #             breakpoint()
    #         if i == (48, 44):
    #             print(f'({b0=:8b}, {b1=:8b}), {i=}, {low2=}')
    #         raw2[i][low2] = cls
    #     elif len(seq) == 3:
    #         b2 = seq[2]
    #         raw3[i][low2][b2] = cls
    #     else:  # len == 4
    #         b2, b3 = seq[2], seq[3]
    #         raw4[i][low2][b2][b3] = cls
    # 
    # # Helper: fill defaults to CLASS_O for missing legal tuples.
    # # Because we treat any "not L/N/Z" as O, it's safe to default to O for gaps.
    # def fill_defaults_len2(i):
    #     # for each low2, there is exactly one exact b1 inside (i1<<2)|low2
    #     # If missing -> O.
    #     return [ (raw2[i][k] if raw2[i][k] is not None else CLASS_O) for k in range(4) ]
    # 
    # def fill_defaults_len3(i):
    #     # Produce dict low2 -> array[256] of b2->class (default O)
    #     out = {}
    #     for low2 in range(4):
    #         arr = [CLASS_O]*256
    #         for b2, cls in raw3[i][low2].items():
    #             arr[b2] = cls if cls is not None else CLASS_O
    #         out[low2] = arr
    #     return out
    # 
    # def fill_defaults_len4(i):
    #     # Produce dict low2 -> array[256][256] of b2,b3->class (default O)
    #     out = {}
    #     for low2 in range(4):
    #         grid = [[CLASS_O]*256 for _ in range(256)]
    #         for b2, m in raw4[i][low2].items():
    #             row = grid[b2]
    #             for b3, cls in m.items():
    #                 row[b3] = cls if cls is not None else CLASS_O
    #         out[low2] = grid
    #     return out
    # 
    # # ---------- Dedup pools for refinement rows ----------
    # def intern_row(pool, key_tup):
    #     """Deduplicate rows: pool: dict key_tup->id, plus list storage on the side."""
    #     if key_tup in pool['map']:
    #         return pool['map'][key_tup]
    #     idx = len(pool['rows'])
    #     pool['rows'].append(list(key_tup))
    #     pool['map'][key_tup] = idx
    #     return idx
    # 
    # ref2_pool     = {'rows': [], 'map': {}}
    # ref3_pool     = {'rows': [], 'map': {}}
    # ref3low_pool  = {'rows': [], 'map': {}}
    # ref4_pool     = {'rows': [], 'map': {}}
    # ref4low_pool  = {'rows': [], 'map': {}}
    # 
    # # Tokens used inside refinement rows:
    # # - Final classes: 0,1,2 as-is
    # # - Pointers are encoded as (KIND_* << 8) | id   (fits in u16)
    # def tok_class(c):
    #     return c  # 0..2
    # 
    # def tok_ptr(kind, idx):
    #     return ((kind & 0xFF) << 8) | (idx & 0xFF)  # u16 token
    # 
    # def is_tok_ptr(t): return (t >> 8) != 0
    # def tok_kind(t):   return (t >> 8) & 0xFF
    # def tok_id(t):     return t & 0xFF
    # 
    # # ---------- Build primary + refinements ----------
    # # Primary is 64x64 u16 entries (packed: 3-bit kind + 13-bit id/class)
    # primary = [[pack_primary(KIND_CLASS, CLASS_O) for _ in range(64)] for __ in range(64)]
    # 
    # def all_equal(lst):
    #     return all(x == lst[0] for x in lst)
    # 
    # # 2-byte regions: b0 in [C2..DF]
    # for i0 in range(64):
    #     for i1 in range(64):
    #         if (i0, i1) == (0x30, 0x2c):
    #             breakpoint()
    #         # Decide what length this primary cell corresponds to by b0 quarter
    #         b0_min = i0 << 2
    #         b0_max = b0_min | 3
    #         # If the whole quarter is outside 2-byte lead range, skip here
    #         if b0_max < 0xC2 or b0_min > 0xDF:
    #             continue
    #         i = (i0, i1)
    #         row = fill_defaults_len2(i)  # 4 entries
    #         if all_equal(row):
    #             primary[i0][i1] = pack_primary(KIND_CLASS, row[0])
    #         else:
    #             rid = intern_row(ref2_pool, tuple(row))
    #             primary[i0][i1] = pack_primary(KIND_REF2, rid)
    # 
    # # 3-byte regions: b0 in [E0..EF]
    # for i0 in range(64):
    #     b0_min = i0 << 2
    #     b0_max = b0_min | 3
    #     if b0_max < 0xE0 or b0_min > 0xEF:
    #         continue
    #     for i1 in range(64):
    #         i = (i0, i1)
    #         grids = fill_defaults_len3(i)  # low2 -> b2[256]
    #         # Check if completely uniform across all low2 and b2
    #         flat = [v for low2 in range(4) for v in grids[low2]]
    #         if all_equal(flat):
    #             primary[i0][i1] = pack_primary(KIND_CLASS, flat[0])
    #             continue
    # 
    #         # Try to collapse by b2>>4 (hi nibble), optionally b2&0xF (low)
    #         # For each low2, build a 16-entry token row (class or ref3low pointer)
    #         ref2_row = []
    #         identical_ref2_entry = None
    #         all_same = True
    # 
    #         for low2 in range(4):
    #             b2arr = grids[low2]
    #             # First see if uniform across all b2
    #             if all_equal(b2arr):
    #                 token = tok_class(b2arr[0])
    #             else:
    #                 # build hi-nibble row of 16 tokens
    #                 hi_tokens = []
    #                 for hi in range(16):
    #                     rng = b2arr[hi<<4:(hi<<4)+16]
    #                     if all_equal(rng):
    #                         hi_tokens.append(tok_class(rng[0]))
    #                     else:
    #                         # need a low-nibble row for this hi
    #                         low_row = tuple(rng)  # 16 entries
    #                         lrid = intern_row(ref3low_pool, low_row)
    #                         hi_tokens.append(tok_ptr(KIND_REF3LOW, lrid))
    #                 # dedup and intern this 16-entry row into ref3
    #                 rid3 = intern_row(ref3_pool, tuple(hi_tokens))
    #                 token = tok_ptr(KIND_REF3, rid3)
    #             ref2_row.append(token)
    #             if identical_ref2_entry is None:
    #                 identical_ref2_entry = token
    #             elif token != identical_ref2_entry:
    #                 all_same = False
    # 
    #         if all_same:
    #             # All four entries identical; collapse primary
    #             t = identical_ref2_entry
    #             if is_tok_ptr(t):
    #                 primary[i0][i1] = pack_primary(tok_kind(t), tok_id(t))
    #             else:
    #                 primary[i0][i1] = pack_primary(KIND_CLASS, t)
    #         else:
    #             # Keep a ref2 row (4 entries)
    #             rid2 = intern_row(ref2_pool, tuple(ref2_row))
    #             primary[i0][i1] = pack_primary(KIND_REF2, rid2)
    # 
    # # 4-byte regions: b0 in [F0..F4]
    # for i0 in range(64):
    #     b0_min = i0 << 2
    #     b0_max = b0_min | 3
    #     if b0_max < 0xF0 or b0_min > 0xF4:
    #         continue
    #     for i1 in range(64):
    #         i = (i0, i1)
    #         grids = fill_defaults_len4(i)  # low2 -> b2[256][256]
    #         # Check fully uniform
    #         all_vals = []
    #         for low2 in range(4):
    #             for b2 in range(256):
    #                 all_vals.extend(grids[low2][b2])
    #         if all_equal(all_vals):
    #             primary[i0][i1] = pack_primary(KIND_CLASS, all_vals[0])
    #             continue
    # 
    #         ref2_row = []
    #         identical_ref2_entry = None
    #         all_same = True
    # 
    #         for low2 in range(4):
    #             # For this low2, make a ref3 row keyed by b2>>4,
    #             # whose entries are class or a pointer to a "per-b2-hi" detail.
    #             hi_tokens = []
    #             for hi2 in range(16):
    #                 # Collect over all b2 whose hi nibble == hi2
    #                 # We want, for each b3, the class to be independent of (b2 low nibble)
    #                 # If not, we split by b2 low nibble (ref3low). Inside that, we may still
    #                 # need a ref4 row keyed by b3>>4.
    #                 # Step 1: compute for each low b2 nibble a 256-entry vector of classes over b3
    #                 per_low = []
    #                 for lo2 in range(16):
    #                     b2 = (hi2<<4) | lo2
    #                     per_low.append(grids[low2][b2][:])  # copy 256 list
    #                 # Try to see if all 16 low-b2 vectors are identical
    #                 if all(all_equal([per_low[0][b3], per_low[l][b3]])
    #                        for l in range(16) for b3 in range(256)):
    #                     # They are identical; collapse across b2 low nibble
    #                     b3arr = per_low[0]
    #                     # Summarize by b3>>4 with possible low rows
    #                     hi4_tokens = []
    #                     for hi3 in range(16):
    #                         seg = b3arr[hi3<<4:(hi3<<4)+16]
    #                         if all_equal(seg):
    #                             hi4_tokens.append(tok_class(seg[0]))
    #                         else:
    #                             # Need a low nibble row over b3
    #                             rid4low = intern_row(ref4low_pool, tuple(seg))
    #                             hi4_tokens.append(tok_ptr(KIND_REF4LOW, rid4low))
    #                     rid4 = intern_row(ref4_pool, tuple(hi4_tokens))
    #                     hi_tokens.append(tok_ptr(KIND_REF4, rid4))
    #                 else:
    #                     # Not identical across b2 low nibble → make ref3low with 16 entries, each of which
    #                     # is summarized by b3>>4 (ref4) or b3 low-nibble (ref4low) if needed.
    #                     ref3low_tokens = []
    #                     for lo2 in range(16):
    #                         b3arr = per_low[lo2]
    #                         hi4_tokens = []
    #                         for hi3 in range(16):
    #                             seg = b3arr[hi3<<4:(hi3<<4)+16]
    #                             if all_equal(seg):
    #                                 hi4_tokens.append(tok_class(seg[0]))
    #                             else:
    #                                 rid4low = intern_row(ref4low_pool, tuple(seg))
    #                                 hi4_tokens.append(tok_ptr(KIND_REF4LOW, rid4low))
    #                         rid4 = intern_row(ref4_pool, tuple(hi4_tokens))
    #                         ref3low_tokens.append(tok_ptr(KIND_REF4, rid4))
    #                     # Intern this 16-entry row in ref3low
    #                     rid3low = intern_row(ref3low_pool, tuple(ref3low_tokens))
    #                     hi_tokens.append(tok_ptr(KIND_REF3LOW, rid3low))
    #             # dedup and intern this 16-entry row into ref3
    #             rid3 = intern_row(ref3_pool, tuple(hi_tokens))
    #             token = tok_ptr(KIND_REF3, rid3)
    #             ref2_row.append(token)
    #             if identical_ref2_entry is None:
    #                 identical_ref2_entry = token
    #             elif token != identical_ref2_entry:
    #                 all_same = False
    # 
    #         if all_same:
    #             t = identical_ref2_entry
    #             primary[i0][i1] = pack_primary(tok_kind(t), tok_id(t))
    #         else:
    #             rid2 = intern_row(ref2_pool, tuple(ref2_row))
    #             primary[i0][i1] = pack_primary(KIND_REF2, rid2)
    # 
    # # ---------- Pretty-print / serialize ----------
    # def as_c_array_u8(name, arr):
    #     vals = ','.join(str(int(x) & 0xFF) for x in arr)
    #     return f'static const uint8_t {name}[{len(arr)}] = {{{vals}}};\n'
    # 
    # def as_c_array_u16_2d(name, grid):
    #     h = len(grid); w = len(grid[0])
    #     flat = []
    #     for r in grid:
    #         flat.extend(r)
    #     vals = ','.join(str(int(x) & 0xFFFF) for x in flat)
    #     return f'static const uint16_t {name}[{h}][{w}] = {{{vals}}};\n'
    # 
    # def dump_tables():
    #     print("// === Length table ===")
    #     print(as_c_array_u8("len_tbl", len_tbl))
    #     print("// === ASCII class (0=L,1=N,2=Z,3=O) ===")
    #     print(as_c_array_u8("ascii_class", ascii_class))
    #     print("// === Primary 64x64 packed (3-bit kind | 13-bit id/class) ===")
    #     print(as_c_array_u16_2d("primary", primary))
    # 
    #     def dump_pool(name, pool):
    #         rows = pool['rows']
    #         print(f"// === {name} ({len(rows)} rows) ===")
    #         for i, row in enumerate(rows):
    #             vals = ','.join(str(int(x) & 0xFFFF) for x in row)
    #             print(f"static const uint16_t {name}_{i}[{len(row)}] = {{{vals}}};")
    #         # Also emit a pointer table for convenience
    #         if rows:
    #             print(f"static const uint16_t* {name}[] = {{")
    #             for i in range(len(rows)):
    #                 print(f"  {name}_{i},")
    #             print("};\n")
    #         else:
    #             print(f"// no rows\n")
    # 
    #     dump_pool("ref2", ref2_pool)
    #     dump_pool("ref3", ref3_pool)
    #     dump_pool("ref3low", ref3low_pool)
    #     dump_pool("ref4", ref4_pool)
    #     dump_pool("ref4low", ref4low_pool)
    # 
    # # Uncomment to print C-style tables:
    # # dump_tables()
    # 
    # # If you prefer Python artifacts (e.g. pickle/json), you can export
    # # len_tbl, ascii_class, primary, and each pool's rows list directly.
    return


@app.cell
def _():


    from unicodedata import category
    from collections import defaultdict

    # -------------------- Class encodings --------------------
    CLASS_L, CLASS_N, CLASS_Z, CLASS_O = 0, 1, 2, 3  # stored directly as u8 0..2

    def cp_class(cp: int) -> int:
        if 0xD800 <= cp <= 0xDFFF:  # surrogates
            return CLASS_O
        c0 = category(chr(cp))[0]
        if c0 == 'L': return CLASS_L
        if c0 == 'N': return CLASS_N
        if c0 == 'Z': return CLASS_Z
        return CLASS_O

    # -------------------- Byte-length table + ASCII --------------------
    len_tbl = [5]*256  # 0=cont, 1..4=len, 5=illegal lead
    for b in range(256):
        if 0x80 <= b <= 0xBF: len_tbl[b] = 0
        elif b <= 0x7F:       len_tbl[b] = 1
        elif 0xC2 <= b <= 0xDF: len_tbl[b] = 2
        elif 0xE0 <= b <= 0xEF: len_tbl[b] = 3
        elif 0xF0 <= b <= 0xF4: len_tbl[b] = 4

    ascii_class = [CLASS_O]*128

    # -------------------- Build byte-sequence map --------------------
    seq_map = {}
    for cp in range(0x110000):
        # try:
        b = chr(cp).encode('utf-8', 'surrogatepass')
        # except Exception:
        #     continue
        seq_map[tuple(b)] = cp_class(cp)

    for b in range(128):
        ascii_class[b] = seq_map.get((b,), CLASS_O)

    # -------------------- Helpers --------------------
    def cell_index(b0, b1): return (b0 >> 2, b1 >> 2)                      # 64×64
    def k16(b0, b1):        return ((b0 & 3) << 2) | (b1 & 3)              # 0..15
    def all_equal(seq):     return not seq or seq.count(seq[0]) == len(seq)

    # Row pools (dedup) — each entry stored as u8
    def make_pool():
        return {'rows': [], 'map': {}}

    def intern_row(pool, key_bytes):  # key_bytes: bytes (length 16)
        m = pool['map']
        if key_bytes in m: return m[key_bytes]
        idx = len(pool['rows'])
        pool['rows'].append(bytearray(key_bytes))  # store as bytearray for easy C dump
        m[key_bytes] = idx
        # u8 pointer space allows 0..252 rows (id 0..252 → encoded as 3..255)
        assert idx <= 252, f"Pool overflow (>253 rows). Consider banking or switch this pool to u16. size={idx+1}"
        return idx

    # Encode entry: class (0..2) or pointer to next stage (3+id)
    def enc_class(c): return c  # 0..2
    def enc_ptr(id_): return 3 + id_  # 3..255  (id in 0..252)

    # -------------------- Raw accumulation @ quarter-byte granularity --------------------
    raw2 = defaultdict(lambda: [None]*16)                                     # (i0,i1)->[16]
    raw3 = defaultdict(lambda: [defaultdict(lambda: None) for _ in range(16)])# ->[16][b2]
    raw4 = defaultdict(lambda: [defaultdict(lambda: defaultdict(lambda: None)) for _ in range(16)])  # ->[16][b2][b3]

    for seq, c in seq_map.items():
        if len(seq) == 1: continue
        b0, b1 = seq[0], seq[1]
        i = cell_index(b0, b1)
        k = k16(b0, b1)
        if len(seq) == 2:
            raw2[i][k] = c
        elif len(seq) == 3:
            raw3[i][k][seq[2]] = c
        else:
            raw4[i][k][seq[2]][seq[3]] = c

    def fill_len2(i):
        row = raw2[i]
        return [CLASS_O if v is None else v for v in row]  # 16 ints

    def fill_len3(i):
        out = {}
        rows = raw3[i]
        for kk in range(16):
            arr = [CLASS_O]*256
            for b2, c in rows[kk].items():
                arr[b2] = CLASS_O if c is None else c
            out[kk] = arr  # 256 ints
        return out

    def fill_len4(i):
        out = {}
        rows = raw4[i]
        for kk in range(16):
            grid = [[CLASS_O]*256 for _ in range(256)]
            for b2, m in rows[kk].items():
                row = grid[b2]
                for b3, c in m.items():
                    row[b3] = CLASS_O if c is None else c
            out[kk] = grid  # 256×256 ints
        return out

    # -------------------- Pools --------------------
    ref2_pool    = make_pool()  # 16 entries per row, u8
    ref3_pool    = make_pool()  # 16 entries per row, u8
    ref3low_pool = make_pool()  # 16 entries per row, u8
    ref4_pool    = make_pool()  # 16 entries per row, u8
    ref4low_pool = make_pool()  # 16 entries per row, u8

    # -------------------- Primary grid (u8) --------------------
    # primary[i0][i1] = class (0..2) or ptr to ref2 row (3+id)
    primary = [[enc_class(CLASS_O) for _ in range(64)] for __ in range(64)]

    # 2-byte region: primary -> class or ref2
    for i0 in range(64):
        b0_min = i0<<2; b0_max = b0_min|3
        if b0_max < 0xC2 or b0_min > 0xDF: continue
        for i1 in range(64):
            row16 = fill_len2((i0,i1))
            if all_equal(row16):
                primary[i0][i1] = enc_class(row16[0])
            else:
                rid = intern_row(ref2_pool, bytes(enc_class(x) for x in row16))
                primary[i0][i1] = enc_ptr(rid)

    # 3-byte region: primary -> class or ref2; ref2[k] -> class or ref3
    for i0 in range(64):
        b0_min = i0<<2; b0_max = b0_min|3
        if b0_max < 0xE0 or b0_min > 0xEF: continue
        for i1 in range(64):
            k2b2 = fill_len3((i0,i1))   # dict k->[256]
            flat = []
            for kk in range(16): flat.extend(k2b2[kk])
            if all_equal(flat):
                primary[i0][i1] = enc_class(flat[0]); continue

            row = []
            same = True; first = None
            for kk in range(16):
                b2arr = k2b2[kk]
                if all_equal(b2arr):
                    t = enc_class(b2arr[0])
                else:
                    # summarize by b2>>4 with possible ref3low
                    hi_tokens = []
                    for hi in range(16):
                        chunk = b2arr[hi<<4:(hi<<4)+16]
                        if all_equal(chunk):
                            hi_tokens.append(enc_class(chunk[0]))
                        else:
                            rid_low = intern_row(ref3low_pool, bytes(enc_class(x) for x in chunk))
                            hi_tokens.append(enc_ptr(rid_low))  # -> ref3low
                    rid3 = intern_row(ref3_pool, bytes(hi_tokens))  # 16 bytes
                    t = enc_ptr(rid3)  # -> ref3
                row.append(t)
                if first is None: first = t
                elif t != first:   same = False

            # Even if all entries equal and are pointers, we still encode primary->ref2,
            # to keep the "next-stage-only" rule. (We could have primary->ref3 directly,
            # but that would require a mixed-kind encoding.)
            rid2 = intern_row(ref2_pool, bytes(row))
            primary[i0][i1] = enc_ptr(rid2)

    # 4-byte region: primary -> class or ref2; ref2[k] -> ref3; ref3 -> ref3low/ref4; etc.
    for i0 in range(64):
        b0_min = i0<<2; b0_max = b0_min|3
        if b0_max < 0xF0 or b0_min > 0xF4: continue
        for i1 in range(64):
            k2grid = fill_len4((i0,i1))  # dict k -> 256×256
            # global uniform?
            sample = None; uniform = True
            for kk in range(16):
                for b2 in range(256):
                    row = k2grid[kk][b2]
                    if sample is None: sample = row[0]
                    if not all(x == sample for x in row):
                        uniform = False; break
                if not uniform: break
            if uniform:
                primary[i0][i1] = enc_class(sample); continue

            row_ref2 = []
            for kk in range(16):
                hi2_tokens = []
                for hi2 in range(16):
                    per_low = [k2grid[kk][(hi2<<4)|lo] for lo in range(16)]
                    identical = all(per_low[lo] == per_low[0] for lo in range(1,16))
                    if identical:
                        b3arr = per_low[0]
                        hi3_tokens = []
                        for hi3 in range(16):
                            seg = b3arr[hi3<<4:(hi3<<4)+16]
                            if all_equal(seg):
                                hi3_tokens.append(enc_class(seg[0]))
                            else:
                                rid4low = intern_row(ref4low_pool, bytes(enc_class(x) for x in seg))
                                hi3_tokens.append(enc_ptr(rid4low))  # -> ref4low
                        rid4 = intern_row(ref4_pool, bytes(hi3_tokens))
                        hi2_tokens.append(enc_ptr(rid4))  # -> ref4
                    else:
                        r3low_tokens = []
                        for lo2 in range(16):
                            b3arr = per_low[lo2]
                            hi3_tokens = []
                            for hi3 in range(16):
                                seg = b3arr[hi3<<4:(hi3<<4)+16]
                                if all_equal(seg):
                                    hi3_tokens.append(enc_class(seg[0]))
                                else:
                                    rid4low = intern_row(ref4low_pool, bytes(enc_class(x) for x in seg))
                                    hi3_tokens.append(enc_ptr(rid4low))
                            rid4 = intern_row(ref4_pool, bytes(hi3_tokens))
                            r3low_tokens.append(enc_ptr(rid4))  # -> ref4
                        rid3low = intern_row(ref3low_pool, bytes(r3low_tokens))
                        hi2_tokens.append(enc_ptr(rid3low))       # -> ref3low
                rid3 = intern_row(ref3_pool, bytes(hi2_tokens))
                row_ref2.append(enc_ptr(rid3))                    # -> ref3

            rid2 = intern_row(ref2_pool, bytes(row_ref2))
            primary[i0][i1] = enc_ptr(rid2)

    # Export tables (all u8 arrays)
    tables_u8 = {
        "len_tbl":    bytearray(len_tbl),
        "ascii_class":bytearray(ascii_class),
        "primary":    [bytearray(row) for row in primary],  # 64 rows × 64 bytes
        "ref2":       ref2_pool["rows"],     # list[bytearray(16)]
        "ref3":       ref3_pool["rows"],     # list[bytearray(16)]
        "ref3low":    ref3low_pool["rows"],  # list[bytearray(16)]
        "ref4":       ref4_pool["rows"],     # list[bytearray(16)]
        "ref4low":    ref4low_pool["rows"],  # list[bytearray(16)]
    }

    CLASS_TO_CHAR = ('L','N', 'Z','O')

    def resolve_lead_u8(b0, b1, b2, b3, T=tables_u8) -> int:
        """
        Return class code (0=L,1=N,2=O) for a lead given next bytes, using u8 tables.
        Each table returns either a class (<=2) or a pointer (>=3) to the next stage.
        """
        if b0 < 128:
            return tables_u8['ascii_class'][b0]
        # primary
        e = T["primary"][b0 >> 2][b1 >> 2]
        if e <= 2: return e
        id2 = e - 3  # -> ref2 row

        # ref2
        k = ((b0 & 3) << 2) | (b1 & 3)
        e = T["ref2"][id2][k]
        if e <= 2: return e
        id3 = e - 3  # -> ref3 row

        # ref3
        e = T["ref3"][id3][b2 >> 4]
        if e <= 2: return e
        id3l = e - 3  # -> ref3low row

        # ref3low
        e = T["ref3low"][id3l][b2 & 0x0F]
        if e <= 2: return e
        id4 = e - 3  # -> ref4 row

        # ref4
        e = T["ref4"][id4][b3 >> 4]
        if e <= 2: return e
        id4l = e - 3  # -> ref4low row

        # ref4low (terminal)
        return T["ref4low"][id4l][b3 & 0x0F]

    def classify_utf8_lno_u8(data: bytes, T) -> list[str]:
        """
        Scalar reference classification using u8 tables.
        """
        n = len(data)
        out = ['O'] * n
        i = 0
        while i < n:
            b0 = data[i]
            tag = T["len_tbl"][b0]

            if tag == 0:
                # continuation: will be painted by previous lead if any
                i += 1
                continue
            if tag == 1:
                c = T["ascii_class"][b0 & 0x7F]
                out[i] = CLASS_TO_CHAR[c]
                i += 1
                continue
            if tag in (2,3,4):
                b1 = data[i+1] if i+1 < n else 0
                b2 = data[i+2] if i+2 < n else 0
                b3 = data[i+3] if i+3 < n else 0
                c = resolve_lead_u8(b0, b1, b2, b3, T)
                ch = CLASS_TO_CHAR[c]
                out[i] = ch
                need = tag - 1
                j = i + 1
                while need > 0 and j < n and T["len_tbl"][data[j]] == 0:
                    out[j] = ch
                    need -= 1
                    j += 1
                i += 1
                continue
            # illegal lead
            i += 1

        # optional: ensure all continuation bytes get painted (handles stray conts len_tbl==0 not preceded by a valid lead)
        carry = None; remain = 0
        for idx, b in enumerate(data):
            tag = T["len_tbl"][b]
            if tag == 0:
                if remain > 0:
                    out[idx] = carry; remain -= 1
            else:
                if out[idx] in ('L','N','O'):
                    carry = out[idx]; remain = tag - 1
                else:
                    carry = None; remain = 0
        return out
    from unicodedata import category
    from collections import defaultdict

    # -------------------- Class encodings (u8) --------------------
    CLASS_L, CLASS_N, CLASS_Z, CLASS_O = 0, 1, 2, 3  # stored directly as u8 0..3

    def cp_class(cp: int) -> int:
        if 0xD800 <= cp <= 0xDFFF:  # surrogates
            return CLASS_O
        c0 = category(chr(cp))[0]
        if c0 == 'L': return CLASS_L
        if c0 == 'N': return CLASS_N
        if c0 == 'Z': return CLASS_Z
        return CLASS_O  # everything else, including Cn/illegal in this pass

    # -------------------- Byte-length table + ASCII --------------------
    len_tbl = [5]*256  # 0=cont, 1..4=len, 5=illegal lead
    for b in range(256):
        if 0x80 <= b <= 0xBF: len_tbl[b] = 0
        elif b <= 0x7F:       len_tbl[b] = 1
        elif 0xC2 <= b <= 0xDF: len_tbl[b] = 2
        elif 0xE0 <= b <= 0xEF: len_tbl[b] = 3
        elif 0xF0 <= b <= 0xF4: len_tbl[b] = 4

    ascii_class = [CLASS_O]*128

    # -------------------- Build byte-sequence map --------------------
    seq_map = {}
    for cp in range(0x110000):
        b = chr(cp).encode('utf-8', 'surrogatepass')
        seq_map[tuple(b)] = cp_class(cp)

    for b in range(128):
        ascii_class[b] = seq_map.get((b,), CLASS_O)

    # -------------------- Helpers --------------------
    def cell_index(b0, b1): return (b0 >> 2, b1 >> 2)                  # 64×64
    def k16(b0, b1):        return ((b0 & 3) << 2) | (b1 & 3)          # 0..15
    def all_equal(seq):     return not seq or seq.count(seq[0]) == len(seq)

    # Row pools (dedup) — each entry stored as u8
    def make_pool():
        return {'rows': [], 'map': {}}

    def intern_row(pool, key_bytes):  # key_bytes: bytes length 16
        m = pool['map']
        if key_bytes in m: return m[key_bytes]
        idx = len(pool['rows'])
        pool['rows'].append(bytearray(key_bytes))
        m[key_bytes] = idx
        # u8 pointer space allows 252 rows: ids 0..251 encoded as 4..255
        assert idx <= 251, f"Pool overflow (>252 rows). Consider banking or promote this pool. size={idx+1}"
        return idx

    # Encode entry: class (0..3) or pointer to next stage (4+id)
    def enc_class(c): return c               # 0..3
    def enc_ptr(id_): return 4 + id_         # 4..255  (id in 0..251)

    # -------------------- Raw accumulation @ quarter-byte granularity --------------------
    raw2 = defaultdict(lambda: [None]*16)                                     # (i0,i1)->[16]
    raw3 = defaultdict(lambda: [defaultdict(lambda: None) for _ in range(16)])# ->[16][b2]
    raw4 = defaultdict(lambda: [defaultdict(lambda: defaultdict(lambda: None)) for _ in range(16)])  # ->[16][b2][b3]

    for seq, c in seq_map.items():
        if len(seq) == 1: continue
        b0, b1 = seq[0], seq[1]
        i = cell_index(b0, b1)
        kk = k16(b0, b1)
        if len(seq) == 2:
            raw2[i][kk] = c
        elif len(seq) == 3:
            raw3[i][kk][seq[2]] = c
        else:
            raw4[i][kk][seq[2]][seq[3]] = c

    def fill_len2(i):
        row = raw2[i]
        return [CLASS_O if v is None else v for v in row]  # 16 ints

    def fill_len3(i):
        out = {}
        rows = raw3[i]
        for kk in range(16):
            arr = [CLASS_O]*256
            for b2, c in rows[kk].items():
                arr[b2] = CLASS_O if c is None else c
            out[kk] = arr
        return out

    def fill_len4(i):
        out = {}
        rows = raw4[i]
        for kk in range(16):
            grid = [[CLASS_O]*256 for _ in range(256)]
            for b2, m in rows[kk].items():
                row = grid[b2]
                for b3, c in m.items():
                    row[b3] = CLASS_O if c is None else c
            out[kk] = grid
        return out

    # -------------------- Pools --------------------
    ref2_pool    = make_pool()  # 16 u8 entries
    ref3_pool    = make_pool()  # 16 u8 entries
    ref3low_pool = make_pool()  # 16 u8 entries
    ref4_pool    = make_pool()  # 16 u8 entries
    ref4low_pool = make_pool()  # 16 u8 entries

    # -------------------- Primary grid (u8) --------------------
    # primary[i0][i1] = class (0..3) or ptr to ref2 row (4+id)
    primary = [[enc_class(CLASS_O) for _ in range(64)] for __ in range(64)]

    # 2-byte region
    for i0 in range(64):
        b0_min = i0<<2; b0_max = b0_min|3
        if b0_max < 0xC2 or b0_min > 0xDF: continue
        for i1 in range(64):
            row16 = fill_len2((i0,i1))
            if all_equal(row16):
                primary[i0][i1] = enc_class(row16[0])
            else:
                rid = intern_row(ref2_pool, bytes(enc_class(x) for x in row16))
                primary[i0][i1] = enc_ptr(rid)

    # 3-byte region
    for i0 in range(64):
        b0_min = i0<<2; b0_max = b0_min|3
        if b0_max < 0xE0 or b0_min > 0xEF: continue
        for i1 in range(64):
            k2b2 = fill_len3((i0,i1))   # dict kk->[256]
            flat = []
            for kk in range(16): flat.extend(k2b2[kk])
            if all_equal(flat):
                primary[i0][i1] = enc_class(flat[0]); continue

            row = []
            for kk in range(16):
                b2arr = k2b2[kk]
                if all_equal(b2arr):
                    t = enc_class(b2arr[0])
                else:
                    hi_tokens = []
                    for hi in range(16):
                        chunk = b2arr[hi<<4:(hi<<4)+16]
                        if all_equal(chunk):
                            hi_tokens.append(enc_class(chunk[0]))
                        else:
                            rid_low = intern_row(ref3low_pool, bytes(enc_class(x) for x in chunk))
                            hi_tokens.append(enc_ptr(rid_low))  # -> ref3low
                    rid3 = intern_row(ref3_pool, bytes(hi_tokens))
                    t = enc_ptr(rid3)  # -> ref3
                row.append(t)

            rid2 = intern_row(ref2_pool, bytes(row))
            primary[i0][i1] = enc_ptr(rid2)

    # 4-byte region
    for i0 in range(64):
        b0_min = i0<<2; b0_max = b0_min|3
        if b0_max < 0xF0 or b0_min > 0xF4: continue
        for i1 in range(64):
            k2grid = fill_len4((i0,i1))  # dict kk -> 256×256
            # global uniform?
            sample = None; uniform = True
            for kk in range(16):
                for b2 in range(256):
                    row = k2grid[kk][b2]
                    if sample is None: sample = row[0]
                    if not all(x == sample for x in row):
                        uniform = False; break
                if not uniform: break
            if uniform:
                primary[i0][i1] = enc_class(sample); continue


            row_ref2 = []
            for kk in range(16):
                hi2_tokens = []
                for hi2 in range(16):
                    per_low = [k2grid[kk][(hi2<<4)|lo] for lo in range(16)]
                    identical = all(per_low[lo] == per_low[0] for lo in range(1,16))
                    if identical:
                        b3arr = per_low[0]
                        hi3_tokens = []
                        for hi3 in range(16):
                            seg = b3arr[hi3<<4:(hi3<<4)+16]
                            if all_equal(seg):
                                hi3_tokens.append(enc_class(seg[0]))
                            else:
                                rid4low = intern_row(ref4low_pool, bytes(enc_class(x) for x in seg))
                                hi3_tokens.append(enc_ptr(rid4low))  # -> ref4low
                        rid4 = intern_row(ref4_pool, bytes(hi3_tokens))
                        hi2_tokens.append(enc_ptr(rid4))  # -> ref4
                    else:
                        r3low_tokens = []
                        for lo2 in range(16):
                            b3arr = per_low[lo2]
                            hi3_tokens = []
                            for hi3 in range(16):
                                seg = b3arr[hi3<<4:(hi3<<4)+16]
                                if all_equal(seg):
                                    hi3_tokens.append(enc_class(seg[0]))
                                else:
                                    rid4low = intern_row(ref4low_pool, bytes(enc_class(x) for x in seg))
                                    hi3_tokens.append(enc_ptr(rid4low))
                            rid4 = intern_row(ref4_pool, bytes(hi3_tokens))
                            r3low_tokens.append(enc_ptr(rid4))  # -> ref4
                        rid3low = intern_row(ref3low_pool, bytes(r3low_tokens))
                        hi2_tokens.append(enc_ptr(rid3low))       # -> ref3low
                rid3 = intern_row(ref3_pool, bytes(hi2_tokens))
                row_ref2.append(enc_ptr(rid3))                    # -> ref3

            rid2 = intern_row(ref2_pool, bytes(row_ref2))
            primary[i0][i1] = enc_ptr(rid2)



    # ---------- 4-byte region builder (no stage skipping, u8 tables, L/N/Z/O) ----------
    # Assumes the following exist in scope:
    #   - enc_class(c: int) -> u8          # 0..3 (L,N,Z,O)
    #   - enc_ptr(id: int) -> u8           # 4+id  (points to NEXT stage only)
    #   - intern_row(pool, key_bytes) -> int   # dedups 16-byte rows; returns row id (0..251)
    #   - all_equal(list[int]) -> bool
    #   - fill_len4((i0,i1)) -> dict[k(0..15)] -> 256x256 int grid of classes for (b2,b3)
    #   - primary: 64x64 u8 matrix to fill
    #   - ref2_pool, ref3_pool, ref3low_pool, ref4_pool, ref4low_pool

    for i0 in range(64):
        b0_min = i0 << 2
        b0_max = b0_min | 3
        # 4-byte UTF-8 leads are F0..F4
        if b0_max < 0xF0 or b0_min > 0xF4:
            continue

        for i1 in range(64):
            # k2grid[k] is a 256x256 grid over (b2,b3) for this primary cell & k16
            k2grid = fill_len4((i0, i1))

            # 1) If the entire cell is uniform across k, b2, b3 → store class directly in primary.
            sample = None
            uniform = True
            for kk in range(16):
                for b2 in range(256):
                    row = k2grid[kk][b2]
                    if sample is None:
                        sample = row[0]
                    # fast uniform check for this row
                    if not all_equal(row) or row[0] != sample:
                        uniform = False
                        break
                if not uniform:
                    break
            if uniform:
                primary[i0][i1] = enc_class(sample)
                continue

            # 2) Otherwise, build a ref2 row with 16 entries (keyed by k16 = (b0&3)<<2 | (b1&3)).
            row_ref2 = []

            for kk in range(16):
                # Build a ref3 row keyed by b2>>4 (16 entries), each entry points to ref3low.
                hi2_tokens = []

                for hi2 in range(16):
                    # Collect the 16 b2-low-nibble rows of length 256 over b3.
                    per_low = [k2grid[kk][(hi2 << 4) | lo] for lo in range(16)]

                    # If all 16 low-nibble rows are IDENTICAL, we still MUST go through ref3low.
                    # We'll create a degenerate ref3low row that points to the same ref4 for all 16 lows.
                    identical = all(per_low[lo] == per_low[0] for lo in range(1, 16))

                    if identical:
                        # Summarize that single b3 array by b3>>4 into a ref4 row (with possible ref4low).
                        b3arr = per_low[0]
                        hi3_tokens = []
                        for hi3 in range(16):
                            seg = b3arr[(hi3 << 4):(hi3 << 4) + 16]
                            if all_equal(seg):
                                hi3_tokens.append(enc_class(seg[0]))
                            else:
                                rid4low = intern_row(ref4low_pool, bytes(enc_class(x) for x in seg))
                                hi3_tokens.append(enc_ptr(rid4low))  # ref4 -> ref4low

                        rid4 = intern_row(ref4_pool, bytes(hi3_tokens))

                        # *** NO SKIP: create a degenerate ref3low row that points to this ref4 for all 16 lows.
                        r3low_degenerate = bytes(enc_ptr(rid4) for _ in range(16))
                        rid3low = intern_row(ref3low_pool, r3low_degenerate)

                        # ref3 entry points to ref3low (next stage)
                        hi2_tokens.append(enc_ptr(rid3low))

                    else:
                        # Need a real ref3low row: for each b2 low-nibble, build a ref4 row.
                        r3low_tokens = []
                        for lo2 in range(16):
                            b3arr = per_low[lo2]
                            hi3_tokens = []
                            for hi3 in range(16):
                                seg = b3arr[(hi3 << 4):(hi3 << 4) + 16]
                                if all_equal(seg):
                                    hi3_tokens.append(enc_class(seg[0]))
                                else:
                                    rid4low = intern_row(ref4low_pool, bytes(enc_class(x) for x in seg))
                                    hi3_tokens.append(enc_ptr(rid4low))  # ref4 -> ref4low
                            rid4 = intern_row(ref4_pool, bytes(hi3_tokens))
                            r3low_tokens.append(enc_ptr(rid4))  # ref3low -> ref4

                        rid3low = intern_row(ref3low_pool, bytes(r3low_tokens))
                        hi2_tokens.append(enc_ptr(rid3low))  # ref3 -> ref3low

                # Dedup and intern the ref3 row (16 entries), each pointing to ref3low.
                rid3 = intern_row(ref3_pool, bytes(hi2_tokens))
                row_ref2.append(enc_ptr(rid3))  # ref2 -> ref3

            # Dedup and store the ref2 row (16 entries), each pointing to ref3.
            rid2 = intern_row(ref2_pool, bytes(row_ref2))
            primary[i0][i1] = enc_ptr(rid2)  # primary -> ref2


    tables_u8 = {
        "len_tbl":     bytearray(len_tbl),
        "ascii_class": bytearray(ascii_class),
        "primary":     [bytearray(row) for row in primary],  # 64×64
        "ref2":        ref2_pool["rows"],     # each 16 bytes
        "ref3":        ref3_pool["rows"],
        "ref3low":     ref3low_pool["rows"],
        "ref4":        ref4_pool["rows"],
        "ref4low":     ref4low_pool["rows"],
    }

    CLASS_TO_CHAR = ('L', 'N', 'Z', 'O')

    def resolve_lead_u8(b0, b1, b2, b3, T=tables_u8) -> int:
        """
        Return class code (0=L,1=N,2=Z,3=O) for a lead given next bytes, using u8 tables.
        Each table returns either a class (<=3) or a pointer (>=4) to the next stage.
        """

        if b0 < 128:
            return tables_u8['ascii_class'][b0]

        # primary
        e = T["primary"][b0 >> 2][b1 >> 2]
        if e <= 3: return e
        id2 = e - 4  # -> ref2

        # ref2
        k = ((b0 & 3) << 2) | (b1 & 3)
        e = T["ref2"][id2][k]
        if e <= 3: return e
        id3 = e - 4  # -> ref3

        # ref3
        e = T["ref3"][id3][b2 >> 4]
        if e <= 3: return e
        id3l = e - 4  # -> ref3low

        # ref3low
        e = T["ref3low"][id3l][b2 & 0x0F]
        if e <= 3: return e
        id4 = e - 4  # -> ref4

        # ref4
        e = T["ref4"][id4][b3 >> 4]
        if e <= 3: return e
        id4l = e - 4  # -> ref4low

        # ref4low (terminal)
        return T["ref4low"][id4l][b3 & 0x0F]

    def classify_utf8_lnoz_u8(data: bytes, T) -> list[str]:
        """
        Scalar reference classification using u8 tables (L/N/Z/O).
        """
        n = len(data)
        out = ['O'] * n
        i = 0
        while i < n:
            b0 = data[i]
            tag = T["len_tbl"][b0]

            if tag == 0:
                i += 1
                continue
            if tag == 1:
                c = T["ascii_class"][b0 & 0x7F]
                out[i] = CLASS_TO_CHAR[c]
                i += 1
                continue
            if tag in (2,3,4):
                b1 = data[i+1] if i+1 < n else 0
                b2 = data[i+2] if i+2 < n else 0
                b3 = data[i+3] if i+3 < n else 0
                c = resolve_lead_u8(b0, b1, b2, b3, T)
                ch = CLASS_TO_CHAR[c]
                out[i] = ch
                need = tag - 1
                j = i + 1
                while need > 0 and j < n and T["len_tbl"][data[j]] == 0:
                    out[j] = ch
                    need -= 1
                    j += 1
                i += 1
                continue

            # illegal lead -> O by policy
            i += 1

        # Optional second pass to paint any dangling continuation bytes
        carry = None; remain = 0
        for idx, b in enumerate(data):
            tag = T["len_tbl"][b]
            if tag == 0:
                if remain > 0:
                    out[idx] = carry; remain -= 1
            else:
                if out[idx] in ('L','N','Z','O'):
                    carry = out[idx]; remain = tag - 1
                else:
                    carry = None; remain = 0

        return out
    return (
        ascii_class,
        cp_class,
        len_tbl,
        primary,
        resolve_lead_u8,
        seq_map,
        tables_u8,
    )


@app.cell(hide_code=True)
def _():
    # # Generates SIMD-friendly tables for classifying UTF-8 bytes into L/N/O
    # # Primary grid is keyed by (b0>>2, b1>>2). The first refinement ("ref2")
    # # is 16-way, keyed by ((b0&3)<<2)|(b1&3) to retain the low bits lost by quarter-bucketing.
    # 
    # from unicodedata import category
    # from collections import defaultdict
    # 
    # # ----- Class & kind encodings -----
    # CLASS_L, CLASS_N, CLASS_O = 0, 1, 2
    # 
    # KIND_CLASS   = 0  # primary entry is a final class (id = CLASS_*)
    # KIND_REF2    = 1  # -> ref2_rows[id][ ((b0&3)<<2)|(b1&3) ]
    # KIND_REF3    = 2  # -> ref3_rows[id][ b2>>4 ]
    # KIND_REF3LOW = 3  # -> ref3low_rows[id][ b2&0xF ]
    # KIND_REF4    = 4  # -> ref4_rows[id][ b3>>4 ]
    # KIND_REF4LOW = 5  # -> ref4low_rows[id][ b3&0xF ]
    # 
    # def pack_primary(kind, ident):
    #     assert 0 <= kind <= 7 and 0 <= ident <= 0x1FFF
    #     return (kind & 7) | (ident << 3)
    # 
    # def unpack_primary(primary):
    #     kind = primary & 7
    #     ident = primary >> 3
    #     return kind, ident
    # 
    # def cp_class(cp):
    #     if 0xD800 <= cp <= 0xDFFF:
    #         return CLASS_O
    #     c = category(chr(cp))[0]
    #     if c == 'L': return CLASS_L
    #     if c == 'N': return CLASS_N
    #     return CLASS_O
    # 
    # # ----- Build sequence map -----
    # seq_map = {}
    # for cp in range(0x110000):
    #     cls = cp_class(cp)
    #     try:
    #         b = chr(cp).encode('utf-8', 'strict')
    #     except Exception:
    #         continue
    #     seq_map[tuple(b)] = cls
    # 
    # # ----- ASCII table -----
    # ascii_class = [CLASS_O]*128
    # for b in range(0x80):
    #     ascii_class[b] = seq_map.get((b,), CLASS_O)
    # 
    # # ----- Length table -----
    # len_tbl = [5]*256
    # for b in range(256):
    #     if 0x80 <= b <= 0xBF: len_tbl[b] = 0
    #     elif b <= 0x7F:       len_tbl[b] = 1
    #     elif 0xC2 <= b <= 0xDF: len_tbl[b] = 2
    #     elif 0xE0 <= b <= 0xEF: len_tbl[b] = 3
    #     elif 0xF0 <= b <= 0xF4: len_tbl[b] = 4
    #     else: len_tbl[b] = 5
    # 
    # # ----- Helpers -----
    # def cell_index(b0, b1): return (b0>>2, b1>>2)  # 0..63 each
    # def idx16(b0, b1): return ((b0 & 3) << 2) | (b1 & 3)  # 0..15
    # 
    # def all_equal(seq):
    #     it = iter(seq)
    #     try:
    #         first = next(it)
    #     except StopIteration:
    #         return True
    #     return all(x == first for x in it)
    # 
    # # ----- Raw accumulation structures -----
    # # For each primary cell (i0,i1), we keep:
    # # - len=2:       raw2[(i0,i1)][16]              -> final class
    # # - len=3:       raw3[(i0,i1)][16][b2]          -> class
    # # - len=4:       raw4[(i0,i1)][16][b2][b3]      -> class
    # 
    # raw2 = defaultdict(lambda: [None]*16)
    # raw3 = defaultdict(lambda: [defaultdict(lambda: None) for _ in range(16)])
    # raw4 = defaultdict(lambda: [defaultdict(lambda: defaultdict(lambda: None)) for _ in range(16)])
    # 
    # for seq, cls in seq_map.items():
    #     if len(seq) == 1:
    #         continue
    #     b0, b1 = seq[0], seq[1]
    #     i = cell_index(b0, b1)
    #     k = idx16(b0, b1)
    #     if len(seq) == 2:
    #         raw2[i][k] = cls
    #     elif len(seq) == 3:
    #         b2 = seq[2]
    #         raw3[i][k][b2] = cls
    #     else:
    #         b2, b3 = seq[2], seq[3]
    #         raw4[i][k][b2][b3] = cls
    # 
    # # ----- Default filling -----
    # def fill_defaults_len2(i):
    #     row = raw2[i]
    #     return [ (c if c is not None else CLASS_O) for c in row ]  # 16 entries
    # 
    # def fill_defaults_len3(i):
    #     # return dict k(0..15)-> list[256] of b2->class
    #     out = {}
    #     rows = raw3[i]
    #     for k in range(16):
    #         arr = [CLASS_O]*256
    #         for b2, c in rows[k].items():
    #             arr[b2] = CLASS_O if c is None else c
    #         out[k] = arr
    #     return out
    # 
    # def fill_defaults_len4(i):
    #     # return dict k(0..15)-> list[256][256] of b2,b3->class
    #     out = {}
    #     rows = raw4[i]
    #     for k in range(16):
    #         grid = [[CLASS_O]*256 for _ in range(256)]
    #         for b2, m in rows[k].items():
    #             row = grid[b2]
    #             for b3, c in m.items():
    #                 row[b3] = CLASS_O if c is None else c
    #         out[k] = grid
    #     return out
    # 
    # # ----- Dedup pools -----
    # def intern_row(pool, key):
    #     m = pool['map']; rows = pool['rows']
    #     if key in m: return m[key]
    #     idx = len(rows); rows.append(list(key)); m[key] = idx; return idx
    # 
    # ref2_pool     = {'rows': [], 'map': {}}
    # ref3_pool     = {'rows': [], 'map': {}}
    # ref3low_pool  = {'rows': [], 'map': {}}
    # ref4_pool     = {'rows': [], 'map': {}}
    # ref4low_pool  = {'rows': [], 'map': {}}
    # 
    # def tok_class(c): return c & 0xFFFF
    # def tok_ptr(kind, idx): return ((kind & 0xFF) << 8) | (idx & 0xFF)
    # def is_tok_ptr(t): return (t >> 8) != 0
    # def tok_kind(t): return (t >> 8) & 0xFF
    # def tok_id(t): return t & 0xFF
    # 
    # # ----- Build primary + refinements -----
    # primary = [[pack_primary(KIND_CLASS, CLASS_O) for _ in range(64)] for __ in range(64)]
    # 
    # # 2-byte
    # for i0 in range(64):
    #     b0_min = i0<<2; b0_max = b0_min|3
    #     if b0_max < 0xC2 or b0_min > 0xDF: continue
    #     for i1 in range(64):
    #         i = (i0,i1)
    #         row16 = fill_defaults_len2(i)
    #         if all_equal(row16):
    #             primary[i0][i1] = pack_primary(KIND_CLASS, row16[0])
    #         else:
    #             rid = intern_row(ref2_pool, tuple(row16))
    #             primary[i0][i1] = pack_primary(KIND_REF2, rid)
    # 
    # # 3-byte
    # for i0 in range(64):
    #     b0_min = i0<<2; b0_max = b0_min|3
    #     if b0_max < 0xE0 or b0_min > 0xEF: continue
    #     for i1 in range(64):
    #         i = (i0,i1)
    #         k_to_b2 = fill_defaults_len3(i)  # dict k-> [256]
    #         # If all 16×256 same:
    #         flat = []
    #         for k in range(16): flat.extend(k_to_b2[k])
    #         if all_equal(flat):
    #             primary[i0][i1] = pack_primary(KIND_CLASS, flat[0]); continue
    # 
    #         # Build ref2 row (16 tokens), each either class or a ref3 pointer
    #         ref2_row = []
    #         all_same = True
    #         first_tok = None
    #         for k in range(16):
    #             b2arr = k_to_b2[k]
    #             if all_equal(b2arr):
    #                 t = tok_class(b2arr[0])
    #             else:
    #                 # summarize by b2>>4 with optional b2&0xF
    #                 hi_tokens = []
    #                 for hi in range(16):
    #                     chunk = b2arr[hi<<4:(hi<<4)+16]
    #                     if all_equal(chunk):
    #                         hi_tokens.append(tok_class(chunk[0]))
    #                     else:
    #                         rid_low = intern_row(ref3low_pool, tuple(chunk))
    #                         hi_tokens.append(tok_ptr(KIND_REF3LOW, rid_low))
    #                 rid3 = intern_row(ref3_pool, tuple(hi_tokens))
    #                 t = tok_ptr(KIND_REF3, rid3)
    #             ref2_row.append(t)
    #             if first_tok is None: first_tok = t
    #             elif t != first_tok: all_same = False
    # 
    #         if all_same:
    #             t = first_tok
    #             if is_tok_ptr(t):
    #                 primary[i0][i1] = pack_primary(tok_kind(t), tok_id(t))
    #             else:
    #                 primary[i0][i1] = pack_primary(KIND_CLASS, t)
    #         else:
    #             rid2 = intern_row(ref2_pool, tuple(ref2_row))
    #             primary[i0][i1] = pack_primary(KIND_REF2, rid2)
    # 
    # # 4-byte
    # for i0 in range(64):
    #     b0_min = i0<<2; b0_max = b0_min|3
    #     if b0_max < 0xF0 or b0_min > 0xF4: continue
    #     for i1 in range(64):
    #         i = (i0,i1)
    #         k_to_b2b3 = fill_defaults_len4(i)  # dict k-> 256x256
    #         # global uniform check (coarse but cheap)
    #         sample = None; uniform = True
    #         for k in range(16):
    #             for b2 in range(256):
    #                 row = k_to_b2b3[k][b2]
    #                 if sample is None: sample = row[0]
    #                 if not all(x == sample for x in row):
    #                     uniform = False; break
    #             if not uniform: break
    #         if uniform:
    #             primary[i0][i1] = pack_primary(KIND_CLASS, sample); continue
    # 
    #         ref2_row = []
    #         all_same = True; first_tok = None
    #         for k in range(16):
    #             # summarize across b2
    #             # try to see if all b2 rows identical
    #             equal_across_b2 = True
    #             first_row = None
    #             for b2 in range(256):
    #                 row = k_to_b2b3[k][b2]
    #                 if first_row is None: first_row = row
    #                 elif row != first_row: equal_across_b2 = False; break
    #             if equal_across_b2:
    #                 # summarize that single row by b3>>4 (with lows if needed)
    #                 hi_tokens = []
    #                 b3arr = first_row
    #                 for hi3 in range(16):
    #                     seg = b3arr[hi3<<4:(hi3<<4)+16]
    #                     if all_equal(seg):
    #                         hi_tokens.append(tok_class(seg[0]))
    #                     else:
    #                         rid4low = intern_row(ref4low_pool, tuple(seg))
    #                         hi_tokens.append(tok_ptr(KIND_REF4LOW, rid4low))
    #                 rid4 = intern_row(ref4_pool, tuple(hi_tokens))
    #                 t = tok_ptr(KIND_REF4, rid4)
    #             else:
    #                 # not identical across b2: we could split by b2>>4; do that
    #                 hi2_tokens = []
    #                 for hi2 in range(16):
    #                     # combine 16 low-b2 rows; if they are identical, collapse
    #                     per_low = [k_to_b2b3[k][(hi2<<4)|lo] for lo in range(16)]
    #                     identical = all(per_low[lo] == per_low[0] for lo in range(1,16))
    #                     if identical:
    #                         b3arr = per_low[0]
    #                         hi4_tokens = []
    #                         for hi3 in range(16):
    #                             seg = b3arr[hi3<<4:(hi3<<4)+16]
    #                             if all_equal(seg):
    #                                 hi4_tokens.append(tok_class(seg[0]))
    #                             else:
    #                                 rid4low = intern_row(ref4low_pool, tuple(seg))
    #                                 hi4_tokens.append(tok_ptr(KIND_REF4LOW, rid4low))
    #                         rid4 = intern_row(ref4_pool, tuple(hi4_tokens))
    #                         hi2_tokens.append(tok_ptr(KIND_REF4, rid4))
    #                     else:
    #                         # fallback: make a ref4 per low-b2 and store 16 of them in a ref3low row
    #                         ref3low_tokens = []
    #                         for lo2 in range(16):
    #                             b3arr = k_to_b2b3[k][(hi2<<4)|lo2]
    #                             hi4_tokens = []
    #                             for hi3 in range(16):
    #                                 seg = b3arr[hi3<<4:(hi3<<4)+16]
    #                                 if all_equal(seg):
    #                                     hi4_tokens.append(tok_class(seg[0]))
    #                                 else:
    #                                     rid4low = intern_row(ref4low_pool, tuple(seg))
    #                                     hi4_tokens.append(tok_ptr(KIND_REF4LOW, rid4low))
    #                             rid4 = intern_row(ref4_pool, tuple(hi4_tokens))
    #                             ref3low_tokens.append(tok_ptr(KIND_REF4, rid4))
    #                         rid3low = intern_row(ref3low_pool, tuple(ref3low_tokens))
    #                         hi2_tokens.append(tok_ptr(KIND_REF3LOW, rid3low))
    #                 rid3 = intern_row(ref3_pool, tuple(hi2_tokens))
    #                 t = tok_ptr(KIND_REF3, rid3)
    # 
    #             ref2_row.append(t)
    #             if first_tok is None: first_tok = t
    #             elif t != first_tok: all_same = False
    # 
    #         if all_same:
    #             t = first_tok
    #             primary[i0][i1] = pack_primary(tok_kind(t), tok_id(t)) if is_tok_ptr(t) else pack_primary(KIND_CLASS, t)
    #         else:
    #             rid2 = intern_row(ref2_pool, tuple(ref2_row))
    #             primary[i0][i1] = pack_primary(KIND_REF2, rid2)
    # 
    # # ----- (Optional) emit C arrays -----
    # def as_c_array_u8(name, arr):
    #     vals = ','.join(str(int(x)&0xFF) for x in arr)
    #     return f'static const uint8_t {name}[{len(arr)}]={{ {vals} }};\n'
    # 
    # def as_c_array_u16_2d(name, grid):
    #     h=len(grid); w=len(grid[0]); flat=[x for row in grid for x in row]
    #     vals=','.join(str(int(x)&0xFFFF) for x in flat)
    #     return f'static const uint16_t {name}[{h}][{w}]={{ {vals} }};\n'
    # 
    # def dump_pool(name, pool):
    #     rows = pool['rows']
    #     print(f"// {name}: {len(rows)} rows")
    #     for i,row in enumerate(rows):
    #         vals=','.join(str(int(x)&0xFFFF) for x in row)
    #         print(f"static const uint16_t {name}_{i}[{len(row)}]={{ {vals} }};")
    #     if rows:
    #         print(f"static const uint16_t* {name}[] = {{")
    #         for i in range(len(rows)): print(f"  {name}_{i},")
    #         print("};\n")
    # 
    # def dump_tables():
    #     print(as_c_array_u8("len_tbl", len_tbl))
    #     print(as_c_array_u8("ascii_class", ascii_class))
    #     print(as_c_array_u16_2d("primary", primary))
    #     dump_pool("ref2", ref2_pool)
    #     dump_pool("ref3", ref3_pool)
    #     dump_pool("ref3low", ref3low_pool)
    #     dump_pool("ref4", ref4_pool)
    #     dump_pool("ref4low", ref4low_pool)
    # 
    # # dump_tables()
    # 
    return


@app.cell
def _(cp_class, resolve_lead_u8):
    for ccp in range(0x110000):
        bb = chr(ccp).encode('utf-8', 'surrogatepass')
        bb0, bb1, bb2, bb3 = list(bb) + [0] * (4 - len(bb))
        real_class = cp_class(ccp)
        my_class = resolve_lead_u8(bb0, bb1, bb2, bb3)
        assert real_class == my_class, f"{ccp=:4X}, {real_class=}, {my_class=}, {bb=}, {chr(ccp)=}"
    return


@app.cell
def _(tables_u8):
    tables_u8
    return


@app.cell
def _(len_tbl):
    len_tbl
    return


@app.cell
def _(ascii_class):
    ascii_class
    return


@app.cell
def _(primary):
    primary
    return


@app.cell
def _(get_kind_ident, primary):
    from collections import Counter
    counter = Counter(get_kind_ident(e)[0] for l in primary for e in l)
    counter
    return


@app.cell
def _(seq_map):
    seq_map[tuple('²'.encode('utf-8'))]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
