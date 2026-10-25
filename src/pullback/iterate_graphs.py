"""The explicit iterate graphs G_n of a graph virtual endomorphism, built
as graphs so that curves and the minimiser can run on them.  Plan items
190/210; extends src/pullback/iterates.py (which computes the filling of
phi^n_0 without building G_n).

G_n = { (x_1, ..., x_n) : pi(x_i) = phi(x_{i+1}) for i < n }, an n-fold
iterate of the pullback relation; phi^n_0(x_1..x_n) = phi(x_1) and
pi^n_0(x_1..x_n) = pi(x_n), so pi^n_0 is a covering of degree deg(pi)^n
(the author's statement, AGENTS.md A0.1: |V(G_n)| = deg(pi)^n |V_0| and
|E(G_n)| = deg(pi)^n |E_0|).

Multi-dart chains are handled by refining the source first: a chain of k
darts crosses k-1 target vertices in its interior, so subdividing the source
edge at those preimages turns it into k pieces each mapping to a single
dart.  The refined map is the same homotopy class, so Emb[phi^n_0] is
unchanged (it is an infimum over the class, intro.tex eq:emb-homotopy).
Note the counts then differ from the author's covering statement
(|V(G_n)| = deg(pi)^n |V_0|, |E(G_n)| = deg(pi)^n |E_0|, AGENTS.md A0.1):
that statement is for the UNrefined tower.  A refined source has extra
edges (one per piece) and extra vertices (one per interior split point);
the bouquet's chains are single-dart, so no refinement happens and the
counts match exactly.
"""

from __future__ import annotations

from fractions import Fraction as F

from .refined import Graph, GraphMap, inv, free_reduce, subdivide, piece_label

__all__ = ["iterate_graph", "iterate_map", "refine_to_single_darts",
           "session_iterate"]


def refine_to_single_darts(src, tgt, chains):
    """Refine the source graph so every chain has at most one dart.

    A chain of k darts crosses k-1 target vertices in its interior (the dart
    joins); subdividing the source edge at those preimages gives k pieces,
    each mapping to a whole target dart.  Returns (refined source Graph,
    {piece edge: (base edge, piece index)}, {refined edge: single-dart
    chain}).  Constant edges are untouched.
    """
    splits = {}
    for e in sorted(src.edges, key=str):
        ch = free_reduce(tuple(chains.get(e, ())))
        if len(ch) > 1:
            step = src.alpha[e] / len(ch)
            splits[e] = [step] * len(ch)
    if not splits:
        return src, {}, {e: free_reduce(tuple(chains.get(e, ())))
                         for e in sorted(src.edges, key=str)}
    rsrc, back = subdivide(src, splits, tag="s")
    rchains = {}
    for e in sorted(src.edges, key=str):
        ch = free_reduce(tuple(chains.get(e, ())))
        if len(ch) <= 1:
            # unsplit edge: the refined label is (e, 0)
            rchains[piece_label(e, 0)] = ch
            continue
        for i, d in enumerate(ch):
            rchains[piece_label(e, i)] = (d,)
    return rsrc, back, rchains


def iterate_graph(base, cover, pi, phi, n):
    """Build G_n explicitly.  base: Gamma_0, cover: Gamma_1, pi: Gamma_1 ->
    Gamma_0 (a covering), phi: Gamma_1 -> Gamma_0 with chains of <= 1 dart
    (use refine_to_single_darts first for multi-dart chains).

    An edge of G_n is a tuple (d_1, ..., d_n) where d_n is a dart of the
    cover and, for i < n, d_i ranges over the lifts of phi(d_{i+1}) (a dart),
    or over the points of pi^{-1} of a vertex when phi is constant there.
    """
    for e in sorted(cover.edges, key=str):
        if len(pi.chains[e]) != 1:
            raise NotImplementedError("pi must be a covering by single darts")
        if len(phi.chains[e]) > 1:
            raise NotImplementedError("phi must map each edge to <= 1 dart")

    # lifts of a dart of the base, as darts of the cover
    lift = {}
    for e in sorted(cover.edges, key=str):
        l, sgn = pi.chains[e][0]
        lift.setdefault(l, {"+": [], "-": []})
        lift[l]["+" if sgn == 1 else "-"].append((e, 1))
        lift[l]["-" if sgn == 1 else "+"].append((e, -1))
    # points of the cover over each vertex of the base
    over = {}
    for u in sorted(cover.deg, key=str):
        over.setdefault(pi.vmap[u], []).append((u,))

    def lifts_of(x):
        """Points of the cover over the point x of the base: over a dart, the
        lifts of that dart; over a vertex, the points above it."""
        if len(x) == 1:                       # a vertex, in our 1-tuple form
            return list(over.get(x[0], []))
        return list(lift.get(x[0], {"+": [], "-": []})["+" if x[1] == 1 else "-"])

    def image_of(d):
        """phi's image of a coordinate value: a dart, or a vertex (1-tuple)."""
        if len(d) == 1:
            return phi.vmap[d[0]]
        ch = phi.chains[d[0]]
        if not ch:
            return (phi.vmap[cover.tail(d)],)
        return ch[0] if d[1] == 1 else inv(ch[0])

    def endpoints(tup):
        tl, hd = [], []
        for d in tup:
            if len(d) == 1:
                tl.append(d)
                hd.append(d)
            else:
                tl.append((cover.tail(d),))
                hd.append((cover.head(d),))
        return tuple(tl), tuple(hd)

    def canon(tup):
        rev = tuple(d if len(d) == 1 else (d[0], -d[1]) for d in tup)
        return min(tup, rev)

    def build(i, prefix):
        if i == 0:
            edges.setdefault(canon(prefix), endpoints(canon(prefix)))
            return
        for d in lifts_of(image_of(prefix[0])):
            build(i - 1, (d,) + prefix)

    edges = {}
    for e in sorted(cover.edges, key=str):
        for s in (1, -1):
            build(n - 1, ((e, s),))

    alpha = {tup: cover.alpha[tup[-1][0]] for tup in edges}
    return Graph(edges, alpha, marked=[], name=f"G{n}")


def iterate_map(base, cover, pi, phi, n, Gn):
    """phi^n_0: G_n -> Gamma_0, (x_1..x_n) |-> phi(x_1)."""
    chains = {}
    for tup in Gn.edges:
        d = tup[0]
        if len(d) == 2:
            ch = phi.chains[d[0]]
            chains[tup] = ch if d[1] == 1 else tuple(inv(x) for x in reversed(ch))
        else:
            chains[tup] = ()
    return GraphMap(Gn, base, chains, name=f"phi^{n}_0")


def session_iterate(path, n, marked_base=()):
    """Load a session and build (G_n, phi^n_0, base) with single-dart chains.

    The source map is refined to single darts first (refine_to_single_darts),
    so multi-dart sessions work; the refined map is the same homotopy class.
    `marked_base` pins the lifts of the given base vertices (the session's
    `marked` field is read as well).
    """
    import json
    from .harmonic_io import load_session
    from .refined import session_map
    with open(path) as f:
        sess = json.load(f)
    marked_base = set(marked_base) | set(sess.get("marked", []))
    base_rib, cover_rib, phi_rib = load_session(path)

    # base and pi as refined.Graph maps
    from .refined import graph_from_ribbon
    G2 = graph_from_ribbon(base_rib)
    G2.marked = frozenset()
    G1 = graph_from_ribbon(cover_rib)
    G1.marked = frozenset({u for u in cover_rib.vertices() if u[0] in marked_base})
    chains1 = {e: tuple(phi_rib.chains.get(e, ())) for e in G1.edges}
    pi_chains = {}
    # pi: the cover projection, as a GraphMap G1 -> G2
    for e in G1.edges:
        # the cover edge e = (label, sheet) projects to the base edge label
        pi_chains[e] = ((e[0], 1),)
    pi = GraphMap(G1, G2, pi_chains, name="pi")
    phi = GraphMap(G1, G2, chains1, name="phi")

    # refine the source so every phi-chain is a single dart
    rsrc, back, rchains = refine_to_single_darts(G1, G2, chains1)
    # pi on the refined source: a piece of edge e projects to e's base edge
    rpi_chains = {}
    for e in rsrc.edges:
        base_e = back.get(e, e)
        rpi_chains[e] = ((base_e[0] if isinstance(base_e, tuple) else base_e, 1),)
    rpi = GraphMap(rsrc, G2, rpi_chains, name="pi~")
    rphi = GraphMap(rsrc, G2, rchains, name="phi~")
    Gn = iterate_graph(G2, rsrc, rpi, rphi, n)
    # marked vertices of G_n: tuples of cover vertices whose coordinates are pinned
    marked_n = set()
    for v in Gn.deg:
        if all(len(c) == 1 and c[0] in G1.marked for c in v):
            marked_n.add(v)
    Gn.marked = frozenset(marked_n)
    phin = iterate_map(G2, rsrc, rpi, rphi, n, Gn)
    return Gn, phin, G2
