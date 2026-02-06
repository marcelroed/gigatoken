import marimo

__generated_with = "0.17.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from dataclasses import dataclass
    from enum import IntEnum
    from unicodedata import category
    from collections import defaultdict

    class CLASS(IntEnum):
        L = 0
        N = 1
        Z = 2
        O = 3
        def __str__(self):
            return self.name


    @dataclass(eq=False)
    class Lanes:
        l: list[int]
        max_digits: int = 1

        @classmethod
        def zero(cls, n: int) -> Lanes:
            return cls([0] * n)

        def shr(self, n: int) -> Lanes:
            return Lanes([None] * n + self.l[:-n])
        
        def shl(self, n: int) -> Lanes:
            return Lanes(self.l[n:] + [None] * n)

        def __repr__(self):
            return f"[{' '.join(f'{d:<{self.max_digits}}' for d in self.l)}]"

        def __getitem__(self, idx):
            return self.l[idx]

        def __eq__(self, other):
            return Lanes([e1 == e2 for e1, e2 in zip(self.l, other.l)])

        def __ne__(self, other):
            return Lanes([e1 != e2 for e1, e2 in zip(self.l, other.l)])

        def __len__(self):
            return len(self.l)

        @classmethod
        def map(cls, f, *laness) -> Lanes:
            return Lanes([f(*lane) for lane in zip(*laness)])

    def cp_class(cp: int) -> int:
        if cp == ord('\n'):
            return CLASS.Z
        if 0xD800 <= cp <= 0xDFFF:  # surrogates
            return CLASS.O
        c0 = category(chr(cp))[0]
        if c0 == 'L': return CLASS.L
        if c0 == 'N': return CLASS.N
        if c0 == 'Z': return CLASS.Z
        return CLASS.O

    print(Lanes.zero(8))
    return CLASS, Lanes, category, cp_class, dataclass, defaultdict, mo


@app.cell
def _(Lanes, cp_class, dataclass, defaultdict):
    @dataclass
    class OrganizedBytes:
        source_str: str
        bytewise_data: dict[str, list]
        def __getitem__(self, key):
            return self.bytewise_data[key]
        def __setitem__(self, key, item):
            assert len(item) == len(next(iter(self.bytewise_data.values())))
            self.bytewise_data[key] = item
        def __str__(self):
            bytewise_idx = 0
            bytewise_printed = defaultdict(list[str])
            longest = 2
        
            for c in self.source_str:
                # bytes = self.bytewise_data['bytes']
                local_bytes = c.encode('utf-8')
            
                bytewise_printed['char'].append(repr(c)[1:-1])
                bytewise_printed['char'].extend([''] * (len(local_bytes) - 1))
            
                for i, b in enumerate(local_bytes):
                    idx = bytewise_idx + i
                    # bytewise_printed['bytes'].append(f"{b:02X}")
                    for k, v in self.bytewise_data.items():
                        value = v[idx]
                        match k:
                            case _ if (k[0] == 'b' and k[1].isnumeric()) or k == 'bytes':
                                str_rep = f"{value:02X}" if value is not None else "NaN"
                            case _:
                                if isinstance(value, bool):
                                    value = int(value)
                                str_rep = str(value) if value is not None else "NaN"
                        bytewise_printed[k].append(str_rep)
                        longest = max(longest, len(str_rep))
                bytewise_idx += len(local_bytes)

            longest_key_name = max(len(k) for k in bytewise_printed.keys())
            output = []
            for k, v in bytewise_printed.items():
                output.append(f"{k:<{longest_key_name}}   ")
                for item in v:
                    output.append(f"{item:<{longest}} ")
                output.append('\n')
            return ''.join(output)
            

    def classify_bytes(s: str):
        bytes, classes = [], []
        for c in s:
            cp = ord(c)
            b = c.encode('utf-8')
            bytes.extend(b)
            cl = cp_class(cp)
            classes.extend([cl] * len(b))
        return OrganizedBytes(s, bytewise_data={
            'bytes': Lanes(bytes),
            'class': Lanes(classes),
        })
    classified = classify_bytes("a  test've\n\nwith aål123 and t's")
    print(classified)
    return OrganizedBytes, classified


@app.cell
def _(CLASS, Lanes, OrganizedBytes, classified):
    from copy import deepcopy

    def simd_boundaries(o: OrganizedBytes) -> OrganizedBytes:
        o = deepcopy(o)
        b0 = o['bytes']
        b1 = b0.shl(1)
        c0 = o['class']
        c1 = c0.shl(1)

        # There is a class boundary between current and next character
        class_boundary01 = c0 != c1

        # Backtrack once for boundaries that exist on a whitespace
        class_boundary_whitespace01 = Lanes.map(lambda cb, c: cb and (c == CLASS.Z), class_boundary01, c0)

        # Spaces that should be merged into the next token
        merge_space = Lanes.map(lambda cbw, b: cbw and b == 0x20, class_boundary_whitespace01, b0)

        # Otherwise the last whitespace character should stay separate
        con2, con3 = handle_contraction(b0, c0)
    
    

    
        o['c1'] = c1
        o['class_boundary01'] = class_boundary01
        o['class_boundary_whitespace01'] = class_boundary_whitespace01
        o['merge_space'] = merge_space
        o['con2'] = con2
        o['con3'] = con3
    
    
        print(o)

    def handle_contraction(b0, c0) -> tuple[Lanes, Lanes]:
        b1 = b0.shl(1)
        b2 = b0.shl(2)
        b3 = b0.shl(3)

        # Cannot be preceded by whitespace or other
        def valid2(c0, b1, b2):
            if any(e is None for e in (c0, b1, b2)):
                return None
            if c0 == CLASS.Z or c0 == CLASS.O:
                return False
            is_valid = b1 == ord("'") and chr(b2) in 'sdmt'
            if is_valid:
                print(locals())
            return is_valid

        def valid3(c0, b1, b2, b3):
            if any(e is None for e in (c0, b1, b2, b3)):
                return None
            if c0 == CLASS.Z or c0 == CLASS.O:
                return False
            if chr(b1) != "'":
                return False
            pair = f'{chr(b2)}{chr(b3)}'
            is_valid = pair in ('ve', 're', 'll')
            if is_valid:
                print(locals())
            return is_valid

        matched2 = Lanes.map(valid2, c0, b1, b2).shr(1)
        matched3 = Lanes.map(valid3, c0, b1, b2, b3).shr(1)

        return matched2, matched3

    

    simd_boundaries(classified)
    return


@app.cell
def _(category):
    category('\n')
    return


@app.cell
def _(mo):
    mo.md(r"""
    From oniguruma:

    The `\s` metacharacter in Unicode is exactly equivalent to the character class
    `[\t\n\v\f\r \x85\xA0\x1680\x2000-\x200A\x2028-\x2029\x202F\x205F\x3000]` — that is, it matches
    the same as ASCII, plus U+0085 (next line), U+00A0 (nonbreaking space), U+1680 (Ogham space mark),
    U+2000 (en quad) through U+200A (hair space) (this range includes several widths of Unicode spaces),
    U+2028 (line separator) through U+2029 (paragraph separator),
    U+202F (narrow no-break space), U+205F (medium mathematical space), and U+3000 (CJK ideographic space).
    """)
    return


if __name__ == "__main__":
    app.run()
