import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
def _():
    from __future__ import annotations
    from functools import lru_cache
    from dataclasses import dataclass
    from enum import IntEnum
    from unicodedata import category
    from collections import defaultdict
    from functools import lru_cache
    from pathlib import Path
    import marimo as mo
    from tqdm import tqdm

    class CLASS(IntEnum):
        """2 bit enum representing the character class of a code point"""
        L = 0
        N = 1
        Z = 2
        O = 3
        I = 4

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

    @dataclass(slots=True)
    class Codepoint:
        cp: int
        bytes: tuple[int]
        cls: CLASS

        @classmethod
        def from_utf32(cls, cp: int, valid_set: set | None = None) -> Codepoint:
            bytes = tuple(chr(cp).encode('utf-8', 'surrogatepass'))
            clss = cp_class(cp)
            if valid_set is not None and cp not in valid_set:
                clss = CLASS.I
            return cls(cp, bytes, clss)

        def as_sequence(self) -> tuple[int | CLASS]:
            return (*self.bytes, cls)

        @lru_cache(maxsize=None)
        def as_bit_sequence(self) -> tuple[int | CLASS, ...]:
            cls = self.cls
            return (*(bit for b in self.bytes for bit in get_bits(b)), cls)

        @lru_cache(maxsize=None)
        def as_u32_bit_sequence(self) -> tuple[int | CLASS, ...]:
            s = map(int, bin(self.cp)[2:])
            return (*s, self.cls)

        def __repr__(self):
            bytes_str = ' '.join(f'{b:02X}' for b in self.bytes)
            return f'{chr(self.cp)}[U+{self.cp:X}, {bytes_str}, {self.cls.name}]'

        def __str__(self):
            return repr(self)

        def __eq__(self, other):
            return self.cp == other.cp

        def __hash__(self):
            return hash(self.cp)

    def get_codepoints(filename='/Users/marcel/data/TinyStoriesV2-GPT4-train.txt'):
        text = Path(filename).read_text()
        # for c in tqdm(Path(filename).read_text()):
        #     s.add(ord(c))
        s = set(text)
    
        return sorted((ord(c) for c in s))
    

    def most_frequent_codepoints(prop: float):
        # Load from file
        from json import load
        with open("/Users/marcel/data/cc_codepoint_counts.json", "r") as f:
            counter = load(f)
        by_freq = sorted([(int(k), v) for k, v in counter.items()], key=lambda x: x[1], reverse=True)
        total_count = len(by_freq)
        codepoints = [cp for cp, count in by_freq[:int(total_count * prop)]]
        cp_set = set(codepoints)
        return [Codepoint.from_utf32(cp, cp_set) for cp in range(0x110000)]


    def _codepoints() -> tuple[list[Codepoint], list[list[Codepoint]]]:
        """Enumerate all Unicode code points"""
        res = []
        for cp in range(0x110000):
        # for cp in get_codepoints(filename="/Users/marcel/data/owt_valid.txt"):
            res.append(Codepoint.from_utf32(cp))

        by_length: list[list[Codepoint]] = [[] for _ in range(5)]
        for cp in res:
            by_length[len(cp.bytes)].append(cp)

        return res, by_length
    codepoints, cp_by_length = _codepoints()

    # def cp_class(cp: int) -> int:
    #     if 0xD800 <= cp <= 0xDFFF:  # surrogates
    #         return CLASS_O
    #     c0 = category(chr(cp))[0]
    #     if c0 == 'L': return CLASS_L
    #     if c0 == 'N': return CLASS_N
    #     if c0 == 'Z': return CLASS_Z
    #     return CLASS_O


    # codepoints = most_frequent_codepoints(0.2)
    print(f'{len(codepoints)=}')
    return CLASS, Codepoint, codepoints, dataclass, defaultdict, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figuring out DFA states
    After we've processed $i$ bits, we want to compress the processed information into as few states as possible.
    These states are found, by merging together all states that have the suffix.
    We represent a full codepoint entry as
    ```
    (bit_0, bit_1, ..., bit_(N-1), CLASS)
    ```
    This way, we can split it up to bit $i$, giving us a prefix and a suffix:

    ```
    prefix = (bit_0, bit_1, ..., bit_i)
    suffix = (bit_(i+1), ..., bit_(N-1), CLASS)
    ```
    We then merge all prefixes share the same set of suffixes into equivalence classes.
    This leads to the following relation

    Prefix $A$ leads to the same state as prefix $B \iff$ the set of suffixes that follow $A$ in the codepoint entry dataset is equal to the set of suffixes that follow $B$.

    The code follows:
    """)
    return


@app.cell
def _(CLASS, Codepoint, codepoints, defaultdict):
    class MergeEntry:
        def __init__(self, full_entry: tuple[int | CLASS, ...]):
            self.pairs: tuple[tuple[int, ...], CLASS] = [full_entry[:-1], full_entry[-1]]



    def general_merge(to_bits: int, codepoints: list[Codepoint]=codepoints):
        # Group code points by their first n_bits
        by_first_section = defaultdict(list)
        for cp in codepoints:
            # if len(cp.bytes) > 3:
            #     continue
            # seq = (bit0, bit1, bit2, ..., bitN, CLASS)
            # where CLASS in {0, 1, 2, 3}
            seq = cp.as_bit_sequence()
            prefix = seq[:to_bits]
            suffix = seq[to_bits:-1]
            clss = seq[-1]
            by_first_section[prefix].append((suffix, clss))

        # We want to merge all histories (prefixes) that have the same behavior (overlapping suffixes result in the same class)
        merged = defaultdict(list)
        partition_lengths = []
        for first_section, s in by_first_section.items():
            distinct_class_suffixes = defaultdict(set)
            for suffix, clss in s:
                distinct_class_suffixes[clss].add(suffix)

            n_partitions = max(len(suffix_set) for suffix_set in distinct_class_suffixes.values())
            partition_lengths.append(n_partitions)
        
        
            merged[frozenset(s)].append(first_section)

        # return len(merged)
        return max(partition_lengths)

    # 0         8          16          24          32
    # 11110uvv	10vvwwww	10xxxxyy	10yyzzzz
    # for i in mo.status.progress_bar(range(0, 34, 1), show_eta=True, show_rate=True):
    #     n_states = general_merge(to_bits=i, codepoints=codepoints)
    #     print(f'You need {n_states:<3} states for the DFA after processing {i:<2} bits')
    return


@app.cell
def _(codepoints, dataclass):
    from typing import Any, Dict, Hashable, Iterable, List, Optional, Tuple, Set


    @dataclass(slots=True)
    class Node:
        # Children are indices into `nodes` list; -1 means missing.
        child0: int = -1
        child1: int = -1
        # terminal_class is the class label if some string ends exactly here; else None.
        terminal_class: Optional[Hashable] = None
        depth: int = 0


    def build_trie(strings_and_classes: Iterable[Tuple[str, Hashable]]) -> List[Node]:
        """
        Build a binary trie from (bitstring, class).
        bitstring must be a string of '0'/'1' with variable length.
        If the same bitstring appears multiple times, their classes must match.
        """
        nodes: List[Node] = [Node(depth=0)]  # root at index 0

        for bitstr, cls in strings_and_classes:
            cur = 0
            d = 0
            for ch in bitstr:
                d += 1
                if ch == "0" or ch == 0:
                    nxt = nodes[cur].child0
                    if nxt == -1:
                        nxt = len(nodes)
                        nodes[cur].child0 = nxt
                        nodes.append(Node(depth=d))
                    cur = nxt
                elif ch == "1" or ch == 1:
                    nxt = nodes[cur].child1
                    if nxt == -1:
                        nxt = len(nodes)
                        nodes[cur].child1 = nxt
                        nodes.append(Node(depth=d))
                    cur = nxt
                else:
                    raise ValueError(f"Non-bit character {ch!r} in {bitstr!r}")

            # Mark terminal class
            if nodes[cur].terminal_class is None:
                nodes[cur].terminal_class = cls
            else:
                if nodes[cur].terminal_class != cls:
                    raise ValueError(
                        f"Conflicting classes for bitstring {bitstr!r}: "
                        f"{nodes[cur].terminal_class!r} vs {cls!r}"
                    )

        return nodes


    def minimize_suffix_subtrees(nodes: List[Node]) -> Tuple[List[int], Dict[Tuple[Any, int, int], int]]:
        """
        Merge equivalent suffix subtrees bottom-up via hashing signatures.

        Returns:
          state_id: list mapping node_index -> minimal_state_id
          sig2id: mapping signature -> id (mainly for inspection/debug)
        """
        # Process nodes in decreasing depth (bottom-up).
        order = sorted(range(len(nodes)), key=lambda i: nodes[i].depth, reverse=True)

        sig2id: Dict[Tuple[Any, int, int], int] = {}
        state_id: List[int] = [-1] * len(nodes)

        for i in order:
            n = nodes[i]
            c0 = -1 if n.child0 == -1 else state_id[n.child0]
            c1 = -1 if n.child1 == -1 else state_id[n.child1]
            sig = (n.terminal_class, c0, c1)

            sid = sig2id.get(sig)
            if sid is None:
                sid = len(sig2id)
                sig2id[sig] = sid
            state_id[i] = sid

        return state_id, sig2id


    def min_states_after_k_bits(strings_and_classes: Iterable[Tuple[str, Hashable]], L: int) -> Dict[int, int]:
        """
        Compute S_k = minimum number of states needed after reading exactly k bits,
        for all k in [0, L], given a finite set of labeled bitstrings of length <= L.

        Returns:
          dict k -> S_k
        """
        nodes = build_trie(strings_and_classes)
        state_id, _ = minimize_suffix_subtrees(nodes)

        # Collect reachable trie nodes per depth (only those that exist in the trie).
        depth_to_states: Dict[int, Set[int]] = {k: set() for k in range(L + 1)}
        for idx, node in enumerate(nodes):
            d = node.depth
            if 0 <= d <= L:
                depth_to_states[d].add(state_id[idx])

        return {k: len(depth_to_states[k]) for k in range(L + 1)}


    def min_states_at_k(strings_and_classes: Iterable[Tuple[str, Hashable]], k: int) -> int:
        """
        Convenience wrapper to compute only S_k for a specific k.
        """
        # L can be inferred as max length; but allowing k up to a given bound is fine.
        pairs = list(strings_and_classes)
        L = max((len(s) for s, _ in pairs), default=0)
        if k > L:
            return 0
        return min_states_after_k_bits(pairs, L)[k]


    # Example:
    data = [
        (seq[:-1] + tuple([0] * (32 - len(seq[:-1]))), seq[-1])
        for seq in [cp.as_bit_sequence() for cp in codepoints]
        if len(seq[:-1]) < 32
    ]
    L = 32
    print(set(d[1] for d in data))

    Sk = min_states_after_k_bits(data, L)
    for k in range(L + 1):
        print(f"k={k}: S_k={Sk[k]}")

    return


@app.cell
def _():
    from __future__ import annotations

    import random
    from collections import defaultdict
    from dataclasses import dataclass
    from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Tuple


    # -----------------------------
    # Bit utilities
    # -----------------------------

    def bits_to_int(bitstr: str) -> int:
        """bitstr is '0'/'1' length L, MSB is bitstr[0]."""
        return int(bitstr, 2)


    def project_bits(x: int, positions: Sequence[int], L: int) -> int:
        """
        Extract bits of x at given positions (0..L-1, where 0 is MSB) and pack
        them into an integer in the same order as positions.
        """
        out = 0
        for p in positions:
            out = (out << 1) | ((x >> (L - 1 - p)) & 1)
        return out


    def choose_random_partition(L: int, b1: int, b2: int, rng: random.Random) -> Tuple[List[int], List[int], List[int]]:
        """
        Randomly split bit positions [0..L-1] into three disjoint groups
        of sizes b1, b2, and L-b1-b2.
        """
        pos = list(range(L))
        rng.shuffle(pos)
        g1 = sorted(pos[:b1])
        g2 = sorted(pos[b1:b1 + b2])
        g3 = sorted(pos[b1 + b2:])
        return g1, g2, g3


    # -----------------------------
    # Conflict construction + greedy coloring
    # -----------------------------

    def lower_bound_colors_from_remainders(items: Iterable[Tuple[int, int, Hashable]]) -> int:
        """
        Each item is (a, r, cls). For a fixed remainder r, all different classes among items
        that share r must occupy different colors.
        Lower bound: max over r of (# distinct classes seen at r).
        """
        seen: Dict[int, set] = defaultdict(set)
        for _, r, c in items:
            seen[r].add(c)
        return max((len(s) for s in seen.values()), default=0)


    def build_conflict_adjacency(items: Iterable[Tuple[int, int, Hashable]]) -> Dict[int, set]:
        """
        Build adjacency for vertices 'a' with edges between a1 and a2 if there exists
        a remainder r such that (a1, r) and (a2, r) appear with different classes.

        items: (a, r, cls)
        returns: adj[a] = set(neighbors)
        """
        # Group by remainder r, then by class.
        by_r: Dict[int, Dict[Hashable, List[int]]] = defaultdict(lambda: defaultdict(list))
        vertices = set()

        for a, r, c in items:
            by_r[r][c].append(a)
            vertices.add(a)

        adj: Dict[int, set] = {a: set() for a in vertices}

        # For each r, connect all a's across different classes (complete multipartite).
        for r, class_to_as in by_r.items():
            if len(class_to_as) <= 1:
                continue
            classes = list(class_to_as.keys())
            for i in range(len(classes)):
                Ai = class_to_as[classes[i]]
                for j in range(i + 1, len(classes)):
                    Aj = class_to_as[classes[j]]
                    # Add edges between every vertex in Ai and every vertex in Aj
                    for u in Ai:
                        nu = adj[u]
                        for v in Aj:
                            nu.add(v)
                            adj[v].add(u)
        return adj


    def greedy_color_with_cap(adj: Dict[int, set], max_colors: int) -> Optional[Dict[int, int]]:
        """
        Greedy coloring (degree-descending) with early abort if colors exceed max_colors.
        Returns mapping vertex->color, or None if it exceeded max_colors.
        """
        # Order by descending degree.
        order = sorted(adj.keys(), key=lambda v: len(adj[v]), reverse=True)
        color: Dict[int, int] = {}

        for v in order:
            used = set()
            for u in adj[v]:
                cu = color.get(u)
                if cu is not None:
                    used.add(cu)
            # Smallest nonnegative color not in used
            c = 0
            while c in used:
                c += 1
                if c >= max_colors:
                    return None
            color[v] = c
        return color


    # -----------------------------
    # Two-iteration feasibility + search for minimal B
    # -----------------------------

    @dataclass(frozen=True)
    class TwoPassSolution:
        B: int
        first_bits: List[int]          # positions of bits used in pass 1 (size B)
        state_bits: int                # s = 2B - L
        state_count: int               # <= 2^s
        # pass1_state_of_a: mapping from a (pass1 pattern) -> state id (color)
        pass1_state_of_a: Dict[int, int]


    def try_two_pass_for_subset(
        xs: Sequence[int],
        cs: Sequence[Hashable],
        L: int,
        B: int,
        first_bits: Sequence[int],
    ) -> Optional[TwoPassSolution]:
        """
        Fix pass1 bit positions (size B). Pass 2 uses all remaining bits plus s state bits
        where s = 2B - L. Feasible if we can map pass1 patterns 'a' to <= 2^s states
        such that (state, remainder) -> class is well-defined.
        """
        if len(first_bits) != B:
            raise ValueError("first_bits must have size B")
        s = 2 * B - L
        if s < 0:
            return None
        max_states = 1 << s

        # Complement positions for remainder
        first_set = set(first_bits)
        rem_bits = [p for p in range(L) if p not in first_set]

        items = []
        for x, c in zip(xs, cs):
            a = project_bits(x, first_bits, L)
            r = project_bits(x, rem_bits, L)
            items.append((a, r, c))

        lb = lower_bound_colors_from_remainders(items)
        if lb > max_states:
            return None

        adj = build_conflict_adjacency(items)
        coloring = greedy_color_with_cap(adj, max_states)
        if coloring is None:
            return None

        used_states = 1 + max(coloring.values(), default=-1)
        if used_states > max_states:
            return None

        return TwoPassSolution(
            B=B,
            first_bits=list(first_bits),
            state_bits=s,
            state_count=used_states,
            pass1_state_of_a=coloring,
        )


    def find_min_B_two_pass(
        bitstrings_and_classes: Sequence[Tuple[str, Hashable]],
        L: int,
        trials_per_B: int = 2000,
        seed: int = 0,
    ) -> Optional[TwoPassSolution]:
        """
        Heuristic search for the smallest B such that a 2-iteration scheme exists.

        Assumptions:
          - All bitstrings have exactly length L.
          - Pass 1 uses exactly B input bits (using fewer only makes pass 2 harder).
          - Pass 2 uses all remaining L-B bits + s state bits, where s = 2B-L.

        Returns a concrete solution (subset + pass1 state mapping) if found, else None.
        """
        xs = [bits_to_int(s) for s, _ in bitstrings_and_classes]
        cs = [c for _, c in bitstrings_and_classes]
        rng = random.Random(seed)

        for B in range((L + 1) // 2, L + 1):
            best: Optional[TwoPassSolution] = None

            for _ in range(trials_per_B):
                first_bits = sorted(rng.sample(range(L), B))
                sol = try_two_pass_for_subset(xs, cs, L, B, first_bits)
                if sol is not None:
                    # Prefer fewer intermediate states (smaller memory)
                    if best is None or sol.state_count < best.state_count:
                        best = sol
                        # Early exit: if we hit the theoretical minimum 1 state, can't improve
                        if best.state_count == 1:
                            return best

            if best is not None:
                return best

        return None


    # -----------------------------
    # Three-iteration feasibility + search for minimal B
    # -----------------------------

    @dataclass(frozen=True)
    class ThreePassSolution:
        B: int
        g1: List[int]     # pass1 input bit positions (size b1)
        g2: List[int]     # pass2 input bit positions (size b2)
        g3: List[int]     # pass3 input bit positions (size b3 = L-b1-b2)
        s1: int           # pass1 state bits = B - b2
        s2: int           # pass2 state bits = B - b3
        states1: int      # used states at stage1
        states2: int      # used states at stage2
        pass1_state_of_a1: Dict[int, int]     # a1 -> state1
        pass2_state_of_u: Dict[int, int]      # u=(state1,g2pattern) -> state2


    def try_three_pass_for_partition(
        xs: Sequence[int],
        cs: Sequence[Hashable],
        L: int,
        B: int,
        g1: Sequence[int],
        g2: Sequence[int],
        g3: Sequence[int],
    ) -> Optional[ThreePassSolution]:
        """
        Constructive (greedy) feasibility check for 3 iterations:

          Pass1 reads g1 bits (b1=len(g1)) -> state1 (<=2^s1), where s1 = B - b2
          Pass2 reads g2 bits + state1 bits -> state2 (<=2^s2), where s2 = B - b3
          Pass3 reads g3 bits + state2 bits -> class

        Necessary constraints:
          - b1 <= B
          - b2 + s1 <= B  => s1 <= B - b2  (we use s1 = B - b2 as the maximum allowed)
          - b3 + s2 <= B  => s2 <= B - b3  (we use s2 = B - b3 as the maximum allowed)

        This function tries to find:
          - a1(g1pattern) -> state1 via greedy coloring on conflicts over (g2,g3)
          - u=(state1,g2pattern) -> state2 via greedy coloring on conflicts over g3
        """
        b1, b2, b3 = len(g1), len(g2), len(g3)
        if b1 > B:
            return None
        s1 = B - b2
        s2 = B - b3
        if s1 < 0 or s2 < 0:
            return None

        max_states1 = 1 << s1
        max_states2 = 1 << s2

        # Stage 1 items: (a1, r23, class) where r23 is pattern on (g2,g3)
        items1 = []
        for x, c in zip(xs, cs):
            a1 = project_bits(x, g1, L)
            r23 = (project_bits(x, g2, L) << b3) | project_bits(x, g3, L)
            items1.append((a1, r23, c))

        lb1 = lower_bound_colors_from_remainders(items1)
        if lb1 > max_states1:
            return None

        adj1 = build_conflict_adjacency(items1)
        color1 = greedy_color_with_cap(adj1, max_states1)
        if color1 is None:
            return None
        used1 = 1 + max(color1.values(), default=-1)
        if used1 > max_states1:
            return None

        # Stage 2 items: (u, r3, class) where u = (state1 << b2) | g2pattern, r3 = g3pattern
        items2 = []
        for x, c in zip(xs, cs):
            a1 = project_bits(x, g1, L)
            st1 = color1[a1]
            g2p = project_bits(x, g2, L)
            u = (st1 << b2) | g2p
            r3 = project_bits(x, g3, L)
            items2.append((u, r3, c))

        lb2 = lower_bound_colors_from_remainders(items2)
        if lb2 > max_states2:
            return None

        adj2 = build_conflict_adjacency(items2)
        color2 = greedy_color_with_cap(adj2, max_states2)
        if color2 is None:
            return None
        used2 = 1 + max(color2.values(), default=-1)
        if used2 > max_states2:
            return None

        return ThreePassSolution(
            B=B,
            g1=list(g1),
            g2=list(g2),
            g3=list(g3),
            s1=s1,
            s2=s2,
            states1=used1,
            states2=used2,
            pass1_state_of_a1=color1,
            pass2_state_of_u=color2,
        )


    def find_min_B_three_pass(
        bitstrings_and_classes: Sequence[Tuple[str, Hashable]],
        L: int,
        trials_per_config: int = 2000,
        seed: int = 0,
    ) -> Optional[ThreePassSolution]:
        """
        Heuristic search for smallest B such that a 3-iteration scheme exists.

        We search over:
          - B (increasing)
          - (b1, b2) with b1<=B, b3=L-b1-b2 >= 0, and b3<=B (so pass3 can index)
          - random partitions of bit positions into groups of sizes (b1,b2,b3)

        Returns a concrete solution if found, else None.
        """
        xs = [bits_to_int(s) for s, _ in bitstrings_and_classes]
        cs = [c for _, c in bitstrings_and_classes]
        rng = random.Random(seed)

        for B in range(1, L + 1):
            # Enumerate plausible (b1,b2); b3 determined.
            # b2 must be < = B (and practically < B to leave s1>0, but allow equality).
            for b1 in range(0, min(B, L) + 1):
                for b2 in range(0, min(B, L - b1) + 1):
                    b3 = L - b1 - b2
                    if b3 < 0:
                        continue
                    if b3 > B:
                        continue
                    # Need some room for state bits; allow s1 or s2 to be 0 but that’s very restrictive.
                    s1 = B - b2
                    s2 = B - b3
                    if s1 < 0 or s2 < 0:
                        continue

                    best: Optional[ThreePassSolution] = None

                    for _ in range(trials_per_config):
                        g1, g2, g3 = choose_random_partition(L, b1, b2, rng)
                        sol = try_three_pass_for_partition(xs, cs, L, B, g1, g2, g3)
                        if sol is not None:
                            if best is None or (sol.states1, sol.states2) < (best.states1, best.states2):
                                best = sol
                                # Good enough to stop early for this B if both stages are tiny
                                if best.states1 == 1 and best.states2 == 1:
                                    return best

                    if best is not None:
                        return best

        return None


    # -----------------------------
    # Example usage (replace with your data)
    # -----------------------------

    # Example: list of (bitstring, class). Must all be length L.
    L = 32
    data: List[Tuple[str, str]] = [
        ("0" * 32, "A"),
        ("1" * 32, "B"),
        ("01" * 16, "C"),
        ("10" * 16, "D"),
        # ... add your 10,000 items ...
    ]

    two = find_min_B_two_pass(data, L, trials_per_B=500, seed=1)
    print("Two-pass:", two)

    three = find_min_B_three_pass(data, L, trials_per_config=200, seed=2)
    print("Three-pass:", three)

    return dataclass, defaultdict


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The following overview shows much state we need after processing each bit:

    ```plaintext
    bitidx     0   1   2   3   4   5   6   7      8   9   10  11  12  13  14  15    16  17  18  19  20  21  22  23    24  25  26  27  28  29  30  31
    states     2   3   5   9   14  21  28  37     34  34  44  52  72  78  108 169   165 165 210 203 170 121 128 169   165 165 176 142 97  35  10  4
    state bits 1   2   3   4   4   5   5   6      6   6   6   6   7   7   7   8     8   8   8   8   8   7   7   8     8   8   8   8   7   6   4   2
    ```

    The state bits are the state needed _after_ processing the bit at the given index.
    We see that we at most require 210 unique states when processing all 32 bits left to right.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Using the DFA
    We can process the first 7 bits with a single [VPERMI2B](https://uops.info/table.html?search=vpermi2b&cb_lat=on&cb_tp=on&cb_uops=on&cb_ports=on&cb_ZEN4=on&cb_measurements=on&cb_doc=on&cb_base=on&cb_avx512=on), leading to 28 states, which fit within $\lceil\log_2(28)\rceil = 5$ bits.

    We then consume another 2 bits, (7 and 10, skipping bits 8 and 9, since these are always `1 0`), leading to 44 states = 6 state bits after processing 11 bits total.

    For bit 11, we do a VPERMI2B with the 6 state bits and 1 read bit, leading to 52 states.
    For bits 12 and 13, VPERMI2B twice, using the 6 state bits and 2 read bits to lookup to get a 78 element, 7-bit state.

    Here is an overview of the initial DFA process, showing the stage in which we process each bit, along with how many states result after processing those bits:
    ```plaintext
    bitidx     0                         8                         16                       24
    stage      1  1  1  1  1  1  1  2    -  -  2  3  4  4  x  x    -  -  x  x  x  x  x  x   -  -  x  x  x  x  x  x
    states     (       28        )  (    44    )  52 (78)
    state bits          5                6        6   7
    ```

    We are left with a 7-bit state, and 14 bits left to consume.
    Can split this into 14-bit lookup then 15-bit lookup (-2 bits with packing)?

    We can now continue using either a bunch more small VPERMI2B ops, or by using large, much more expensive [VPGATHERDD](https://uops.info/table.html?search=VPGATHERDD&cb_lat=on&cb_tp=on&cb_uops=on&cb_ports=on&cb_ZEN4=on&cb_measurements=on&cb_doc=on&cb_base=on&cb_avx512=on) ops to process many bits.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Notes
    Any time processing a single bit leads to at least a bit increase in the state size, we should instead process more bits at a time.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Reducing state size
    **Note:** It should be possible to gather statistics of the most used characters, then limit the algorithm based on these statistics to only recognize characters that show up frequently.

    We can take the $N$ most frequent code points, then build the DFA only for those codepoints.
    In this case we still need an indicator for if we missed this subset.

    How can we form a DFA that works for a subset of codepoints?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Table Construction
    In order to construct the tables, we perform the same computation as the state estimation, but give a value to
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
