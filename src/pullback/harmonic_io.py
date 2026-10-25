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


# Session fields beyond the data model (graph/cover/phi), with their meaning
# for M3.  `marked` and `orbifold` are independent: a graph may carry both.
#   marked:   base vertices whose lifts are pinned (empty = free class)
#   orbifold: RESERVED, not read yet -- [{"vertex": v, "order": n}] cone points
SESSION_FIELDS = {
    "marked": "list of base vertex names whose lifts are pinned (homotopy class)",
    "orbifold": "reserved: list of {vertex, order} cone points (orbigraphs)",
}


def check_session_fields(sess):
    """Validate the non-data session fields.  Raises ValueError on a
    malformed `marked` or `orbifold`; unknown fields are ignored (they are
    documentation, e.g. `expected_from_output_tex`)."""
    marked = sess.get("marked", [])
    if not isinstance(marked, list) or not all(isinstance(v, str) for v in marked):
        raise ValueError("session field 'marked' must be a list of vertex names")
    orb = sess.get("orbifold", [])
    if not isinstance(orb, list):
        raise ValueError("session field 'orbifold' must be a list")
    for cp in orb:
        if (not isinstance(cp, dict) or "vertex" not in cp or "order" not in cp
                or not isinstance(cp["order"], int) or cp["order"] < 2):
            raise ValueError("orbifold cone points must be {vertex, order} with "
                             "integer order >= 2, got %r" % (cp,))
    return sess


def load_session(path):
    """Load a JSON session: keys graph, cover, phi (Harmonic-format
    strings).  Returns (base, cover, phi)."""
    with open(path) as f:
        sess = json.load(f)
    check_session_fields(sess)
    base = parse_graph(sess["graph"])
    cover, _sheets = parse_cover(base, sess["cover"])
    phi = parse_cmap(cover, base, sess["phi"])
    marked = sess.get("marked", [])
    bad = [v for v in marked if v not in base.vertices()]
    if bad:
        raise ValueError(f"marked vertices {bad} are not vertices of the base graph")
    return base, cover, phi
