"""The rose normal form of a graph virtual endomorphism (plan item 130).

Choose a spanning tree T0 of Gamma_0, collapse it to get the rose R0
(one vertex, g = |E0|-|V0|+1 loops); collapse pi^{-1}(T0) to get R1;
homotop phi so vertices go to the vertex.  The result is an automaton:

  states  = the free generators of pi_1(Gamma_0)  (the non-tree edges)
  letters = the sheets of the cover
  sigma_g = the sheet permutation of the lifts of g
  g|_i    = the phi-image of the lift of g at sheet i, as a word in the
            generators (base-tree letters deleted)

This generalizes automaton.RoseAutomaton (which assumed the base was
already a rose) to an arbitrary spine.
"""

from __future__ import annotations

from itertools import combinations

from .ribbons import F, B, reduce_chain
from .validate import _tree_path
from .automaton import Automaton


def spanning_trees(base, cap=20000):
    """All spanning trees of a connected base graph, as frozensets of
    edge labels, in a fixed order (by size, then sorted labels).

    A base with one vertex is a rose: its only spanning tree is empty
    (a loop is not a tree, so {loop} must not be offered as one).
    """
    edges = sorted(base.edge_list())
    verts = base.vertices()
    out = []
    for k in range(len(verts)):
        for combo in combinations(edges, k):
            parent = {v: v for v in verts}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            ok = True
            for e in combo:
                u, v = base.edges[e]
                ru, rv = find(u), find(v)
                if ru == rv:
                    ok = False
                    break
                parent[ru] = rv
            if ok and len({find(v) for v in verts}) == 1:
                out.append(frozenset(combo))
                if len(out) > cap:
                    raise ValueError(
                        f"more than {cap} spanning trees; enumerate by hand")
    return out


def is_admissible_tree(base, cover, phi, tree):
    """Can the source tree pi^{-1}(T0) be collapsed under phi?

    Collapsing pi^{-1}(T0) inside Gamma_1 and T0 inside Gamma_0 gives the
    virtual endomorphism R1 -> R0 of [PChar, fig:repr-rose] only if phi
    sends every lift of a tree edge to a POINT of R0, i.e. to a word that
    is trivial once the tree is contracted.  (This is the condition the
    paper states for its own example: "the two c edges of Gamma_1 map
    with a constant map to the two vertices of Gamma_0", with c the tree.)

    Measured 2026-10-24, and the reason this is a gate rather than a
    footnote: over the 58 spanning trees of the 14 library sessions this
    condition holds for exactly 14, and exactly on those 14 do the two
    natural readings of "the section" -- phi of the whole lifted loop
    tau.g.tau^{-1}, or phi of the generator's lift alone (the tau-parts
    lie in the collapsed tree, so they are points of R1) -- give
    identical data.  On the other 44 they disagree, and the automaton is
    not determined by the data: for theta's tree {p} the two readings
    give level-7 group orders 2^26 and 2^4 where the true value is 2^25,
    and the section closure under {p} does not terminate (5105 distinct
    level-5 actions by depth 40, and growing).
    """
    tree = set(tree)
    tree_set = tree
    sheets = {s for v in cover.vertices() for (_, s) in [v]}
    for e in tree:
        for s in sheets:
            ch = reduce_chain(tuple(d for d in phi.image_dart(((e, s), F))
                                    if d[0] not in tree_set))
            if ch:
                return False
    return True


def admissible_trees(base, cover, phi):
    """Every spanning tree on which the collapse is legitimate."""
    return [t for t in spanning_trees(base)
            if is_admissible_tree(base, cover, phi, t)]


def normal_form(base, cover, phi, tree=None, require_admissible=True) -> Automaton:
    """The rose normal form of a graph virtual endomorphism.

    With `tree=None` the first admissible spanning tree is used, so the
    answer does not depend on the arbitrary tree `validate.spanning_tree`
    happens to produce (for theta that one was {p}, the inadmissible
    one).  `require_admissible=False` computes the automaton anyway, for
    inspection: an inadmissible tree gives data that is not well defined,
    see `is_admissible_tree`.
    """
    if tree is None:
        cands = admissible_trees(base, cover, phi) if require_admissible \
            else spanning_trees(base)
        if not cands:
            raise ValueError(
                "no admissible spanning tree: phi does not collapse the "
                "lifts of any spanning tree, so the data do not define a "
                "virtual endomorphism of a rose (see is_admissible_tree)")
        tree0 = sorted(cands[0])
    else:
        tree0 = list(tree)
        if not (set(tree0) & set(base.edge_list())) == set(tree0):
            raise ValueError(f"unknown edge in tree {tree0}")
        if require_admissible and not is_admissible_tree(base, cover, phi, tree0):
            raise ValueError(
                f"tree {sorted(tree0)} is not admissible: phi does not send "
                "the lifts of its edges to points of R0 (see "
                "is_admissible_tree); the automaton would not be well defined")
    gens0 = [e for e in base.edge_list() if e not in set(tree0)]
    tree0set = set(tree0)
    # a base path is lifted through the cover; its phi-image, with
    # base-tree letters deleted, is a word in the generators
    def lift_path(path, start_vertex):
        """Lift a path of base darts from a cover vertex; returns
        (lifted darts, end cover vertex)."""
        v = start_vertex
        lifted = []
        for d in path:
            # the unique lift of d starting at v: dart (d[0], sheet) with
            # matching sheet and orientation, starting at v
            label, ori = d
            v_sheet = v[1]
            if ori == F:
                s = v_sheet
            else:
                # the lift of (label, B) starting at (v, s) is the dart
                # (label, s') with s' such that sigma(s') = s
                s = v_sheet
            # find the cover dart at v over d
            found = None
            for dd in cover.flowers[v]:
                if dd[0][0] == label and dd[1] == ori:
                    found = dd
                    break
            if found is None:
                raise ValueError(f"no lift of dart {d} at {v}")
            lifted.append(found)
            v = cover.tgt(found)
        return tuple(lifted), v

    index = {e: i for i, e in enumerate(gens0)}

    # the "sheets" of the automaton are the vertex classes of
    # R1 = Gamma_1 / pi^{-1}(T0): merge cover vertices along the lifts
    # of the tree edges
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for e in tree0:
        for s in {s for v in cover.vertices() for (_, s) in [v]}:
            a_v, b_v = cover.edges[(e, s)]
            union(a_v, b_v)
    classes = {}
    for v in cover.vertices():
        classes.setdefault(find(v), []).append(v)
    reps = {c: min(vs, key=str) for c, vs in classes.items()}
    class_of = {v: c for c, vs in classes.items() for v in vs}
    sheet_list = sorted(classes, key=lambda c: str(reps[c]))
    sheet_index = {c: i for i, c in enumerate(sheet_list)}

    # basepoint: the loop for generator g is
    # tau(b0 -> src g) . g . tau(tgt g -> b0),  lifted from each class.
    b0 = base.vertices()[0]

    sigma = {}
    sections = {}
    for g in gens0:
        u, v = base.edges[g]
        loop = (_tree_path(base, tree0, b0, u) + ((g, F),)
                + _tree_path(base, tree0, v, b0))
        sig = []
        secs = []
        for c in sheet_list:
            # the class is a tree; take its vertex over the basepoint
            start = [w for w in classes[c] if w[0] == b0][0]
            lifted, end = lift_path(loop, start)
            word = []
            for d in lifted:
                word.extend(phi.image_dart(d))
            word = reduce_chain(tuple(x for x in word if x[0] not in tree0set))
            sig.append(sheet_index[class_of[end]])
            secs.append(tuple((index[d[0]], d[1]) for d in word))
        sigma[g] = sig
        sections[g] = secs
    return Automaton(tuple(gens0), tuple(range(len(sheet_list))), sigma, sections)
