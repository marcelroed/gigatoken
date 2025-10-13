# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "ty==0.0.1a15",
# ]
# ///

# To run, install `uv` and do `uvx marimo edit bit_patterns.py`

import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


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
    return Bits, Sentence


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
    mo.md(
        r"""
    ### Notes
    It seems like we need 3 tokens from the past, and 3 tokens into the future, both for finding contractions.
    Maybe then we just load 8 tokens extra and slap 4 onto each side?
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
