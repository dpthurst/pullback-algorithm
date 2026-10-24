"""Importer for the Harmonic text format (GraphIO.lhs).

Format (see .scratch/Harmonic/GraphIO.lhs):
  graph: flowers — "apqr bRQP": vertex letter then its incident darts in
    cyclic order; lowercase dart = forward (out of the vertex),
    uppercase = backward.
  cover: monodromy per edge — "q01" or "rabc|def": edge letter then
    disjoint cycles, cycles separated by '|'; sheets are single
    characters; edges not mentioned get the identity.
  map:   one token per cover edge — "p0P": (edge p, sheet 0) maps to the
    edge path "P" (a string of darts); unlisted edges are constant.
"""

from __future__ import annotations

import json

from .ribbons import RibbonGraph, F, B, parse_dart
from .covers import monodromy_cover
from .maps import make_graph_map


def parse_graph(text: str) -> RibbonGraph:
    flowers = {}
    for word in text.split():
        v, darts = word[0], word[1:]
        flowers[v] = [parse_dart(c) for c in darts]
    return RibbonGraph(flowers)


def _parse_perm(cycles: str) -> dict:
    """'ab|cd' -> {a: b, b: a, c: d, d: c} style mapping (single chars)."""
    perm = {}
    for cyc in cycles.split("|"):
        for i, c in enumerate(cyc):
            perm[c] = cyc[(i + 1) % len(cyc)]
    return perm


def parse_cover(base: RibbonGraph, text: str):
    monodromy = {}
    sheets = set()
    for word in text.split():
        e, perm = word[0], _parse_perm(word[1:])
        monodromy[e] = {_sheet_key(k): _sheet_key(v) for k, v in perm.items()}
        sheets.update(monodromy[e])
        sheets.update(monodromy[e].values())
    sheets = tuple(sorted(sheets))
    return monodromy_cover(base, sheets, monodromy), sheets


def parse_cmap(cover, base, text: str):
    """Tokens: edge letter, sheet letter, then the image path (possibly
    empty is not allowed — omit the edge instead).  Duplicate edge
    tokens are an error (the Harmonic code merged them silently;
    e.g. Examples.lhs rabbitMap has "b0a c0b b0c", a typo for
    "b0a c0b b1c" — see scripts/build_library.py)."""
    chains = {}
    for word in text.split():
        e, sheet, path = word[0], word[1], word[2:]
        key = (e, _sheet_key(sheet))
        if key in chains:
            raise ValueError(f"duplicate map token for {key}")
        chains[key] = [parse_dart(c) for c in path]
    return make_graph_map(cover, base, chains)


def _sheet_key(s):
    return int(s) if s.isdigit() else s


def load_session(path):
    """Load a JSON session: keys graph, cover, phi (Harmonic-format
    strings).  Returns (base, cover, phi)."""
    with open(path) as f:
        sess = json.load(f)
    base = parse_graph(sess["graph"])
    cover, _sheets = parse_cover(base, sess["cover"])
    phi = parse_cmap(cover, base, sess["phi"])
    return base, cover, phi
