"""Monodromy covers of ribbon graphs (Harmonic's permCover convention).

Given a base ribbon graph, sheets 0..d-1, and a permutation per edge
(identity by default), the lift of edge e with start sheet s runs
(e, s): (src e, s) -> (tgt e, sigma_e(s)).  The ribbon structure pulls
back: at (v, s), the cyclic order is the lift of the order at v.
"""

from __future__ import annotations

from .ribbons import RibbonGraph, F, B


def monodromy_cover(base: RibbonGraph, sheets, monodromy: dict):
    """sheets: iterable of sheet labels.  monodromy: {edge: {s: s'}}.
    Returns (cover, pi) with pi mapping cover vertices/edges/darts to
    the base."""
    sheets = tuple(sheets)
    perms = {}
    for e in base.edge_list():
        p = dict(monodromy.get(e, {}))
        for s in sheets:
            p.setdefault(s, s)
        perms[e] = p

    flowers = {}
    for v in base.vertices():
        for s in sheets:
            lifts = []
            for (label, ori) in base.flowers[v]:
                if ori == F:
                    s_at = s
                else:
                    # the backward dart (label, B) at v = tgt label lifts
                    # to the dart whose source sheet satisfies sigma(src)=s
                    inv = {t2: t1 for t1, t2 in perms[label].items()}
                    s_at = inv[s]
                lifts.append(((label, s_at), ori))
            flowers[(v, s)] = lifts
    return RibbonGraph(flowers)


def cover_map_down(dart):
    """Project a cover dart to the base dart."""
    (label, _sheet), ori = dart
    return (label, ori)


def expand_generator_cover(base, tree, generators):
    """Item 60's user-facing form: a spanning tree T0 is fixed, the
    remaining edges generate pi_1, and the user gives one permutation
    per generator.  Tree edges must carry no permutation.  Returns the
    per-edge monodromy dict ready for monodromy_cover."""
    tree = set(tree)
    on_tree = [e for e in generators if e in tree]
    if on_tree:
        raise ValueError(f"permutations given on tree edges {on_tree}")
    return dict(generators)


def check_generator_cover(base, tree, monodromy):
    """Validate a per-generator specification: every edge outside the
    tree is a generator; tree edges must be trivial."""
    tree = set(tree)
    gens = [e for e in base.edge_list() if e not in tree]
    for e in tree:
        if monodromy.get(e):
            raise ValueError(f"tree edge {e} carries a nontrivial permutation")
    return gens
