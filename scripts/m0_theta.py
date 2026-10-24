#!/usr/bin/env python3
"""M0 spike: the theta example (1+z^2)/(1-z^2), end to end from the
JSON session: data in -> faces -> portrait (via pi) -> Moore diagram.

Expected ([PChar] Fig. 3 + spines.tex): portrait degrees (1,1,2,2);
Moore labels matching Fig. 3d up to the sheet relabeling.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pullback.ribbons import fmt_dart
from pullback.portrait import face_map_pi, tighten
from pullback.harmonic_io import load_session

base, cover, phi = load_session(ROOT / "sessions" / "theta.json")


def fmt_walk(walk):
    return "[" + " ".join(fmt_dart(d) for d in walk) + "]"


print("== base ==")
print(base)
print("faces:", [fmt_walk(f) for f in base.faces()])
print()
print("== cover ==")
print(cover)
print("faces:", [fmt_walk(f) for f in cover.faces()])
print()
print("== vertex images under phi ==")
for v in sorted(phi.vertex_images, key=str):
    print(f"  {v} -> {phi.vertex_images[v]}")
print()
print("== portrait (via the covering pi) ==")
report, base_walks = face_map_pi(cover, base)
for row in report:
    tgt = row["target"]
    tgt_s = fmt_walk(base_walks[tgt]) if tgt is not None else "NOT A FACE (data error)"
    print(f"  face {fmt_walk(row['face'])}")
    print(f"    projection  {fmt_walk(row['projection'])}")
    print(f"    -> {tgt_s}  degree {row['degree']}")

print()
print("== Moore diagram (rose collapse of c = q) ==")
parent = {}


def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(x, y):
    parent[find(x)] = find(y)


for e in cover.edge_list():
    if e[0] == "q":
        s, t = cover.edges[e]
        union(s, t)
        union(t, s)
rose_verts = {}
for v in cover.vertices():
    rose_verts.setdefault(find(v), len(rose_verts))
print("R1 vertices:", {str(k): v for k, v in rose_verts.items()})
for e in cover.edge_list():
    if e[0] == "q":
        continue
    s, t = cover.edges[e]
    ch = phi.chains[e]
    rose_class = fmt_walk(tighten(tuple(d for d in ch if d[0] != "q")))
    print(f"  edge {str(e):10s} ({fmt_dart((e[0], 1))}) "
          f"{rose_verts[find(s)]} -> {rose_verts[find(t)]}   "
          f"label ({e[0]}, {rose_class})")
