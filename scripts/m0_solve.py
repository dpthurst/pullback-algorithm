#!/usr/bin/env python3
"""Exploratory script from the M0 spike, kept as a record.

NOTE (2026-10-23): the "spec" below encodes a *misreading* — it demands
phi image walks that match the pi-portrait.  In fact the portrait comes
from the covering pi, while phi's boundary images have the embedding
pattern (3 base faces once each + the central quad collapsed), which
the Harmonic data already satisfies.  The search below is therefore
answering the wrong question; its zero Moore-matching hits do not imply
any incompatibility in the paper's data.

Original spec: the portrait constraints from [PChar] Fig. 3a/spines.tex, with
face assignments from repr.mp:
  Gamma_0 faces:  (p r) = (a,b) contains inf;  (r q) = (b,c) contains 1;
                  (p q) = (a,c) contains -1.
  Gamma_1:  bigons (p0 r0), (p1 r1) contain {-1, 1};
            quad (q0 p1 q1 p0) = (a,c,a,c) exterior, contains inf;
            quad (r0 q1 r1 q0) = (b,c,b,c) central, contains 0.
  Portrait: bigons -> (a,b) face, deg 1;  exterior -> (a,c) face, deg 2;
            central -> (b,c) face, deg 2.
Search chains up to length 3 per edge, then report the solutions and
their Moore labels (rose classes after collapsing c = q).
"""

import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pullback.ribbons import RibbonGraph, F, B, rev, fmt_dart
from pullback.covers import monodromy_cover
from pullback.portrait import tighten

base = RibbonGraph({"a": [("p", F), ("q", F), ("r", F)],
                    "b": [("r", B), ("q", B), ("p", B)]})
cover = monodromy_cover(base, sheets=(0, 1), monodromy={"q": {0: 1, 1: 0}})

DARTS = [("p", F), ("p", B), ("q", F), ("q", B), ("r", F), ("r", B)]


def all_chains(maxlen=3):
    """All composable chains of length 0..maxlen, grouped by (src, tgt)."""
    out = {}
    def add(ch):
        if not ch:
            return
        s, t = base.src(ch[0]), base.tgt(ch[-1])
        out.setdefault((s, t), []).append(ch)
    for d in DARTS:
        add((d,))
    for length in (2, 3):
        for ch in product(DARTS, repeat=length):
            ok = all(base.tgt(ch[i]) == base.src(ch[i + 1]) for i in range(length - 1))
            if ok:
                add(tuple(ch))
    # empty chain: allowed between any equal endpoints
    return out


CHAINS = all_chains()

VERTS = [("a", 0), ("a", 1), ("b", 0), ("b", 1)]
EDGES = [("p", 0), ("p", 1), ("r", 0), ("r", 1), ("q", 0), ("q", 1)]


def cyclic_eq(w1, w2):
    if len(w1) != len(w2):
        return False
    n = len(w1)
    return any(w1 == w2[i:] + w2[:i] for i in range(n))


FACE_AB = [("p", F), ("r", B)]   # [p, R] and rotations; also reversed
FACE_BC = [("r", F), ("q", B)]
FACE_AC = [("p", F), ("q", B)]


def covers(walk, face, k):
    """Is tightened `walk` the k-th power of `face` (up to rotation,
    either global direction)?"""
    w = tighten(walk)
    if len(w) != k * len(face):
        return False
    rface = [rev(d) for d in reversed(face)]
    for f in (face, rface):
        target = tuple(f) * k
        if any(w == target[i:] + target[:i] for i in range(len(target))):
            return True
    return False


solutions = []
for vim_tuple in product("ab", repeat=4):
    vim = dict(zip(VERTS, vim_tuple))
    # chain candidates per edge, from vertex images
    cand = {}
    ok = True
    for e in EDGES:
        s, t = cover.edges[e]
        si, ti = vim[s], vim[t]
        opts = []
        if si == ti:
            opts.append(())
        for ch in CHAINS.get((si, ti), []):
            if len(ch) <= 3:
                opts.append(ch)
        # cap: skip chains with immediate backtracks
        opts = [c for c in opts if tighten(c) == c]
        if not opts:
            ok = False
            break
        cand[e] = opts
    if not ok:
        continue
    # B1/B2 first: bigon constraints involve p's and r's
    for p0c in cand[("p", 0)]:
        for r0c in cand[("r", 0)]:
            if not covers(p0c + tuple(rev(d) for d in reversed(r0c)), FACE_AB, 1):
                continue
            for p1c in cand[("p", 1)]:
                for r1c in cand[("r", 1)]:
                    if not covers(p1c + tuple(rev(d) for d in reversed(r1c)), FACE_AB, 1):
                        continue
                    for q0c in cand[("q", 0)]:
                        for q1c in cand[("q", 1)]:
                            ext = (q0c
                                   + tuple(rev(d) for d in reversed(p1c))
                                   + q1c
                                   + tuple(rev(d) for d in reversed(p0c)))
                            if not covers(ext, FACE_AC, 2):
                                continue
                            cent = (r0c
                                    + tuple(rev(d) for d in reversed(q1c))
                                    + r1c
                                    + tuple(rev(d) for d in reversed(q0c)))
                            if not covers(cent, FACE_BC, 2):
                                continue
                            solutions.append({
                                "vim": dict(vim),
                                ("p", 0): p0c, ("p", 1): p1c,
                                ("r", 0): r0c, ("r", 1): r1c,
                                ("q", 0): q0c, ("q", 1): q1c,
                            })

print(f"{len(solutions)} solutions\n")


def fmt_ch(ch):
    return "".join(fmt_dart(d) for d in ch) if ch else "(const)"


def rose(ch):
    """Collapse q: keep p, r letters only, then freely reduce."""
    return tighten(tuple(d for d in ch if d[0] != "q"))


# Filter: Moore labels must match Fig. 3d —
#   p0 -> A, p1 -> b, r0 -> 1, r1 -> 1, q's -> 1 in the rose.
P_ = (("p", B),)
R_ = (("r", F),)
moore_ok = [
    s for s in solutions
    if rose(s[("p", 0)]) == P_ and rose(s[("p", 1)]) == R_
    and rose(s[("r", 0)]) == () and rose(s[("r", 1)]) == ()
    and rose(s[("q", 0)]) == () and rose(s[("q", 1)]) == ()
]
print(f"{len(moore_ok)} solutions with Moore labels as in Fig. 3d\n")
for s in moore_ok[:12]:
    print("vertex images:", {str(k): v for k, v in s["vim"].items()})
    for e in EDGES:
        ch = s[e]
        print(f"  phi{str(e):10s} = {fmt_ch(ch):12s}  rose: {fmt_ch(rose(ch))}")
    print()
