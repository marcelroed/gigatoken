import marimo

__generated_with = "0.17.4"
app = marimo.App(width="full")


@app.cell
def _(CLASS_L, CLASS_N, CLASS_O, CLASS_Z):
    from __future__ import annotations
    from functools import lru_cache
    from dataclasses import dataclass
    from enum import IntEnum
    from unicodedata import category
    from collections import defaultdict
    from tqdm import tqdm, trange
    from functools import lru_cache
    from emoji import is_emoji

    class CLASS(IntEnum):
        """2 bit enum representing the character class of a code point"""
        L = 0
        N = 1
        Z = 2
        O = 3

        def __repr__(self):
            return self.name

        def as_bits(self):
            return tuple([int(b) for b in f'{self.value:02b}'])

    def cp_class(cp: int) -> int:
        if 0xD800 <= cp <= 0xDFFF:  # surrogates
            return CLASS.O
        c0 = category(chr(cp))[0]
        if c0 == 'L': return CLASS.L
        if c0 == 'N': return CLASS.N
        if c0 == 'Z': return CLASS.Z
        return CLASS.O


    def get_bits(b):
        return [int(i) for i in f"{b:08b}"]

    def get_bits_utf32(cp):
        return [int(i) for i in f"{cp:021b}"]

    @dataclass(slots=True)
    class Codepoint:
        cp: int
        bytes: tuple[int]
        cls: CLASS

        @classmethod
        def from_utf32(cls, cp: int) -> Codepoint:
            bytes = tuple(chr(cp).encode('utf-8', 'surrogatepass'))
            clss = cp_class(cp)
            return cls(cp, bytes, clss)

        def as_sequence(self) -> tuple[int | CLASS]:
            return (*self.bytes, cls)

        @lru_cache(maxsize=None)
        def as_bit_sequence(self, binary: CLASS | None = None, reverse: bool = False) -> tuple[int | CLASS]:
            cls = self.cls
            if binary is not None:
                if cls != binary:
                    cls = CLASS.O
            if reverse:
                return (*list(bit for b in self.bytes for bit in reversed(get_bits(b))), cls)
            return (*(bit for b in self.bytes for bit in get_bits(b)), cls)

        def as_utf32_bit_sequence(self, binary: CLASS | None = None, reverse: bool = False) -> tuple[int | CLASS]:
            if reverse:
                raise ValueError("Not supported")
            cls = self.cls
            if binary is not None:
                if cls != binary:
                    cls = CLASS.O
            return (*get_bits_utf32(self.cp), cls)

        def __repr__(self):
            bytes_str = ' '.join(f'{b:02X}' for b in self.bytes)
            return f'{chr(self.cp)}[U+{self.cp:X}, {bytes_str}, {self.cls.name}]'

        def __str__(self):
            return repr(self)

        def __eq__(self, other):
            return self.cp == other.cp

        def __hash__(self):
            return hash(self.cp)


    def _codepoints():
        res = []
        for cp in range(0x110000):
            # if category(chr(cp)) in ('Cs', 'Co', 'Cn'):
            #     continue
            res.append(Codepoint.from_utf32(cp))

        by_length: list[list[Codepoint]] = [[] for _ in range(5)]
        for cp in res:
            by_length[len(cp.bytes)].append(cp)

        return res, by_length
    codepoints, cp_by_length = _codepoints()
    codepoints3_with_emoji = [cp for cp in codepoints if len(cp.bytes) <= 3 or is_emoji(chr(cp.cp))]

    def cp_class(cp: int) -> int:
        if 0xD800 <= cp <= 0xDFFF:  # surrogates
            return CLASS_O
        c0 = category(chr(cp))[0]
        if c0 == 'L': return CLASS_L
        if c0 == 'N': return CLASS_N
        if c0 == 'Z': return CLASS_Z
        return CLASS_O


    print(len(codepoints))
    print(len(codepoints3_with_emoji))
    return (
        CLASS,
        IntEnum,
        codepoints,
        codepoints3_with_emoji,
        cp_by_length,
        dataclass,
        defaultdict,
        tqdm,
        trange,
    )


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compressed Byte DFA

    We want to scan byte by byte through each 4-tuple of the text, (comparing lane shifted `b0, b1, b2, b3`), producing the class of the sequence that starts at `b0`.

    For each byte, we need to fetch either
    * What is the output type of the bytes I've processed (we're done)
    * What is the minimum state I need to disambiguate between byte sequences that share the next few bytes

    The table lookups need to be in

    ### How do we construct this state machine?
    To begin with, we have to store all the necessary information about previous bytes (and no more!).
    If we do this with each byte processed, we need to do lookups with the byte _and_ the carried context, which is more than we can look up using SIMD.
    Instead we want to process each nibble along with a state nibble that we carry along with us (4 * 2 = 8) lookups.
    We can layout our sequences in bit strings + class.
    """)
    return


@app.cell
def _(CLASS, codepoints, defaultdict, mo):
    def general_merge(to_bits: int, from_bits: int = 0, limit_to_length: int | None = None, binary: CLASS | None = None, codepoints=codepoints):
        if from_bits != 0:
            # Not sure if this is needed yet
            raise NotImplemented()

        # Group code points by their first n_bits
        by_first_section = defaultdict(list)
        for cp in codepoints:
            if limit_to_length is not None and len(cp.bytes) != limit_to_length:
                continue
            # seq = cp.as_utf32_bit_sequence(binary=binary, reverse=False)
            seq = cp.as_bit_sequence(binary=binary, reverse=False)
            # seq = tuple([*seq[:-1][::-1], seq[-1]])
            if limit_to_length:
                assert len(seq) == 8 * limit_to_length + 1 #, f'Expected length , got {len(seq)}'
            by_first_section[seq[:to_bits]].append(seq[to_bits:])

        # We now have (first section) -> [last section]
        # We build a map from {last section} -> [first section],
        # where we can input one case of all the elements in the last section,
        # and get all the first sections that give us that set of last section results.

        # The number of unique entries in this map is the number of states we need to represent the first n bits.

        # Find out how many groups of first sections you would need to represent all last sections
        merged = defaultdict(list)
        for first_section, s in by_first_section.items():
            merged[frozenset(s)].append(first_section)
        # print(f'\x1b[1A\x1b[2KYou need {len(merged)} values for the intermediate state at {to_bits}', end='')
        # print(merged)
        return len(merged)

    #        start      end         0           8          16          24          32
    # 4-byte U+010000	U+10FFFF	11110uvv	10vvwwww	10xxxxyy	10yyzzzz
    limit_to_length = 3
    for i in mo.status.progress_bar(range(0, 34, 1), show_eta=True, show_rate=True):
        # Top at 141 with binary for L
        # Top at 164 values with no binary
        # print(f'{i = }', end='')
        n_states = general_merge(to_bits=i, limit_to_length=None, codepoints=codepoints)  # You need 164 values for the intermediate state
        print(f'You need {n_states} values for the intermediate state at {i}')
        # print('-'*30)
    # general_merge(to_bits=26, limit_to_length=4, binary=None)  # You need 164 values for the intermediate state
    # general_merge(to_bits=32, limit_to_length=4, binary=CLASS.L)  # You need 164 values for the intermediate state
    return (general_merge,)


@app.cell
def _(general_merge):
    general_merge(to_bits=26, limit_to_length=4, binary=None)  # You need 164 values for the intermediate state
    return


@app.cell
def _():
    # Find the best bits to distinguish on
    return


@app.cell
def _(cp_by_length, defaultdict, tqdm, trange):
    def merge_2len():
        # All code points with their classes grouped by the first byte
        by_first_byte = [set() for _ in range(256)]
        for cp in cp_by_length[2]:
            by_first_byte[cp.bytes[0]].add((cp.bytes[1], cp.cls))
        by_first_byte = [frozenset(s) for s in by_first_byte]

        # Find all first bytes that give you unique sets of outputs for the full range of second bytes
        merged = defaultdict(list)
        for b0, s in enumerate(by_first_byte):
            if s:
                merged[s].append(b0)

        print(len(merged))  # 21 groups
        print(merged)

    def merge_3len():
        by_first_two_bytes = [[set() for __ in range(256)] for _ in range(256)]
        for cp in cp_by_length[3]:
            by_first_two_bytes[cp.bytes[0]][cp.bytes[1]].add((cp.bytes[2], cp.cls))
        by_first_two_bytes = [[frozenset(s) for s in t] for t in by_first_two_bytes]
        merged = defaultdict(list)
        for b0, t in enumerate(by_first_two_bytes):
            for b1, s in enumerate(t):
                if s:
                    merged[s].append((b0, b1))
        print(len(merged))  # 143 groups
        # print(merged)

    def merge_4len():
        by_first_three_bytes = [[[list() for ___ in range(256)] for __ in range(256)] for _ in trange(256)]
        for cp in tqdm(cp_by_length[4]):
            by_first_three_bytes[cp.bytes[0]][cp.bytes[1]][cp.bytes[2]].append((cp.bytes[3], cp.cls))
        # by_first_three_bytes = [[[frozenset(s) for s in t] for t in y] for y in tqdm(by_first_three_bytes)]
        merged = defaultdict(list)
        for b0, t in enumerate(tqdm(by_first_three_bytes)):
            for b1, u in enumerate(t):
                for b2, s in enumerate(u):
                    if s:
                        merged[frozenset(s)].append((b0, b1, b2))
        print(len(merged))  # 165 groups
        # print(merged)

    def merge_4lenby2():
        by_first_two_bytes = [[list() for _ in range(256)] for _ in range(256)]
        for cp in tqdm(cp_by_length[4]):
            # print(cp.bytes[:2])
            by_first_two_bytes[cp.bytes[0]][cp.bytes[1]].append((cp.bytes[2], cp.bytes[3], cp.cls))
        # by_first_two_bytes = [[frozenset(s) for s in t] for t in tqdm(by_first_two_bytes)]
        merged = defaultdict(list)
        for b0, t in enumerate(tqdm(by_first_two_bytes)):
            for b1, s in enumerate(t):
                if s:
                    merged[frozenset(s)].append((b0, b1))
        print(len(merged))  # 22 groups
        # print(merged.values())
        # print([f'{v[0] * 16 + v[1]:2X}' for v in merged.values()])
        # print(merged)

    # merge_4lenby2()
    merge_4lenby2()
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


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(codepoints):
    # Build ranges
    def run_ranges():
        previous_class = None
        range_count = 1
        ranges = []
        last_range_start = 0
        for idx, cp in enumerate(codepoints):
            cls = cp.cls
            # if cls != CLASS.L:
            #     cls = CLASS.O
            if cls != previous_class:
                ranges.append((idx - last_range_start, previous_class))
                if idx - last_range_start < 5:
                    print(f'Range {[codepoints[l] for l in range(last_range_start, idx)]}')
                last_range_start = idx
                range_count += 1
                previous_class = cls
        print(range_count)
        print(ranges)
    # run_ranges()
    return


@app.cell
def _(mo):
    mo.md(
        r"""### Code to construct real tables from the merges we hypothesized above"""
    )
    return


@app.cell
def _(codepoints, defaultdict):
    def construct_tables():
        # Build tables to map between constructed states

        states_by_length = []

        by_prefix_length = []
        with_shared_prefix = defaultdict(list)
        for prefix_len in range(0, 33):
            for cp in codepoints:
                seq = cp.as_bit_sequence()  # (*bits, class)
                prefix = seq[:prefix_len]
                suffix = seq[prefix_len:]
                with_shared_prefix[prefix].append(suffix)
            for prefix, suffixes in with_shared_prefix.items():
                frozen_suffixes = frozenset(suffixes)
                previous_prefix = prefix[:-1]
    # construct_tables()
    return


@app.cell
def _(CLASS, IntEnum, codepoints3_with_emoji, dataclass, defaultdict):
    from typing import Dict, List, Optional, Sequence, Tuple, Iterable, Set

    # ---- DFA data structure ----
    @dataclass(frozen=True)
    class MinimizedDFA:
        start: int
        transitions: Dict[Tuple[int, int], int]  # (state, symbol) -> next_state
        outputs: Dict[int, Optional[CLASS]]      # state -> CLASS (or None for non-terminal)
        alphabet: Tuple[int, ...] = (0, 1)

        @property
        def states(self) -> Set[int]:
            s = {self.start}
            for (q, _), r in self.transitions.items():
                s.add(q); s.add(r)
            return s

        def classify(self, bits: Sequence[int]) -> Optional[CLASS]:
            """Return the CLASS for an exact sequence, or None if not in the set."""
            q = self.start
            for b in bits:
                if b not in self.alphabet:
                    raise ValueError(f"Bit {b} not in alphabet {self.alphabet}")
                q = self.transitions[(q, b)]
            return self.outputs.get(q)

        def describe(self) -> str:
            lines = []
            lines.append(f"States: {len(self.states)} | Transitions: {len(self.transitions)}")
            lines.append(f"Start: {self.start}")
            for q in sorted(self.states):
                out = self.outputs.get(q)
                lines.append(f"State {q}: output={out.name if isinstance(out, IntEnum) else out}")
                for a in self.alphabet:
                    lines.append(f"  --{a}--> {self.transitions[(q, a)]}")
            return "\n".join(lines)

    def _hopcroft_minimize(
        start: int,
        transitions: Dict[Tuple[int, int], int],
        outputs: Dict[int, Optional[CLASS]],
        alphabet: Tuple[int, ...]
    ) -> MinimizedDFA:
        # Build state set
        states: Set[int] = set()
        states.add(start)
        for (q, _), r in transitions.items():
            states.add(q); states.add(r)

        # Initial partition: by output label (including None)
        by_label: Dict[Optional[CLASS], Set[int]] = defaultdict(set)
        for q in states:
            by_label[outputs.get(q)].add(q)
        P: List[Set[int]] = [s for s in by_label.values() if s]  # blocks
        W: List[Set[int]] = [set(block) for block in P]          # worklist

        # Precompute predecessors for each symbol
        pred: Dict[int, Dict[int, Set[int]]] = {a: defaultdict(set) for a in alphabet}
        for (q, a), r in transitions.items():
            pred[a][r].add(q)

        while W:
            A = W.pop()
            for a in alphabet:
                # Predecessors that lead into A with symbol a
                X = set()
                for r in A:
                    X |= pred[a].get(r, set())
                if not X:
                    continue
                newP = []
                for Y in P:
                    inter = Y & X
                    diff = Y - X
                    if inter and diff:
                        newP.extend([inter, diff])
                        # Maintain W
                        if Y in W:
                            W.remove(Y)
                            W.extend([inter, diff])
                        else:
                            # add the smaller part to W (classic optimization)
                            W.append(inter if len(inter) <= len(diff) else diff)
                    else:
                        newP.append(Y)
                P = newP

        # Map each old state to a block index (new state)
        block_index: Dict[int, int] = {}
        for i, block in enumerate(P):
            for q in block:
                block_index[q] = i

        new_start = block_index[start]
        new_outputs: Dict[int, Optional[CLASS]] = {}
        new_transitions: Dict[Tuple[int, int], int] = {}

        # For each block, choose a representative to define transitions/output
        for i, block in enumerate(P):
            rep = next(iter(block))
            new_outputs[i] = outputs.get(rep)
            for a in alphabet:
                tgt_old = transitions[(rep, a)]
                new_transitions[(i, a)] = block_index[tgt_old]

        return MinimizedDFA(
            start=new_start,
            transitions=new_transitions,
            outputs=new_outputs,
            alphabet=alphabet
        )

    # ---- Builder & Minimizer ----
    def build_minimized_classifier(
        samples: Iterable[Tuple[Tuple[int, ...], CLASS]],
        alphabet: Tuple[int, ...] = (0, 1)
    ) -> MinimizedDFA:
        """
        Build a minimal DFA (Moore machine) that recognizes exactly the given sequences
        and outputs their CLASS at terminal states. All other sequences go to a sink.
        Minimization uses Hopcroft's algorithm generalized to preserve per-state outputs.
        """

        # Step 1: Build a prefix-trie DFA (incomplete transitions initially)
        next_state = 0
        start = 0
        transitions: Dict[Tuple[int, int], int] = {}
        outputs: Dict[int, Optional[CLASS]] = defaultdict(lambda: None)

        def new_state() -> int:
            nonlocal next_state
            s = next_state
            next_state += 1
            return s

        new_state()  # allocate state 0 as start

        # Insert each sequence, marking terminal output
        for bits, label in samples:
            q = start
            for b in bits:
                if b not in alphabet:
                    raise ValueError(f"Bit {b} not in alphabet {alphabet}")
                key = (q, b)
                if key not in transitions:
                    transitions[key] = new_state()
                q = transitions[key]
            # Conflict detection: same sequence tagged with different class
            if outputs[q] is not None and outputs[q] != label:
                raise ValueError(f"Conflicting labels for sequence {bits}: {outputs[q]} vs {label}")
            outputs[q] = label

        # Step 2: Add sink state and complete the DFA (total on alphabet)
        sink = new_state()
        outputs[sink] = None
        # Ensure every state has both transitions; add missing edges to sink
        all_states = set([start] + [s for _, s in transitions.items()])
        # we may add states while enumerating; keep expanding until stable
        changed = True
        while changed:
            changed = False
            for q in list(all_states):
                for a in alphabet:
                    if (q, a) not in transitions:
                        transitions[(q, a)] = sink
                        changed = True
                        all_states.add(sink)
            # also ensure sink loops to itself
            for a in alphabet:
                transitions[(sink, a)] = sink

            # expand states discovered after filling
            for (_, _), r in list(transitions.items()):
                if r not in all_states:
                    all_states.add(r)
                    changed = True

        # Step 3: Hopcroft minimization (respecting outputs as distinguishing feature)
        return _hopcroft_minimize(start, transitions, outputs, alphabet)

    # ---- Convenience: build from your format and show a quick summary ----
    def optimize_classifier(
        data: Iterable[Tuple[Tuple[int, ...], CLASS]],
        alphabet: Tuple[int, ...] = (0, 1),
    ) -> MinimizedDFA:
        """
        Public entry point. Pass your [(bits_tuple, CLASS), ...].
        Returns a minimized DFA you can use to classify sequences and inspect structure.
        """
        dfa = build_minimized_classifier(data, alphabet=alphabet)
        return dfa

    # ---- Example usage (remove or adapt in your codebase) ----
    # Toy set
    bit_sequences = [
        cp.as_bit_sequence()
        for cp in codepoints3_with_emoji
    ]
    samples = [
        (bits[:-1], bits[-1])  # (bit sequence, CLASS)
        for bits in bit_sequences
    ]
    dfa = optimize_classifier(samples)
    print(dfa.describe())
    print("Classify (1,0):", dfa.classify((1,0)))
    print("Classify (1,1):", dfa.classify((1,1)))  # -> None (not in set)
    return


if __name__ == "__main__":
    app.run()
