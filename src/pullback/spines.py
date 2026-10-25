"""Spine changes for a branched self-cover: ribbon-graph Whitehead moves and
general graph pullbacks.

The engine models a branched self-cover of the sphere by a pair (G_0, G_1):
a spine G_0, a covering pi: G_1 -> G_0 and a graph map phi: G_1 -> G_0 (the
retraction on the preimage of the spine).  Two operations move between such
pairs, and both are the SAME construction -- the fibre product of graph maps:

  * the iterate.  G_{n+1} = G_1 x_{G_0} G_n, the covering pi pulled back
    along phi^n_0: G_n -> G_0, with phi^{n+1}_0 = phi . proj_{G_1}
    (iterate_graphs.iterate_graph builds this tuple by tuple);

  * the spine change.  A Whitehead move on an edge of G_0 -- collapse it,
    expand the merged vertex along a split -- is a homotopy equivalence
    s = h . c': G_0' -> G_0 (c' the collapse of the new edge, h the inverse
    collapse); pulling pi back along s gives G_1' = G_1 x_{G_0} G_0' and
    phi' = k . phi . proj_{G_1}, where k = h' . c: G_0 -> G_0' is the other
    expansion (a homotopy inverse of s).  The pair (G_0', G_1') presents the
    same surface map on a different spine.

`whitehead_move` implements the ribbon-preserving move.  The collapse merges
the two flowers across the collapsed darts (successors a -> d and c -> b);
the expansion splits the merged flower into two cyclic arcs, and the move
that keeps every edge's two darts on one side -- so loops stay loops -- is
the one that preserves the ribbon structure; it is unique when it exists.
On the theta graph's middle edge it gives exactly the barbell of the
sessions (flowers (p, p^-1, q') and (r^-1, r, q'^-1)).

`pullback` is the general fibre product of two graph maps with a common
target: both are refined to single-dart chains (constant edges stay
constant) and the product graph is built from the matching edge pairs and
the degenerate (constant edge, vertex) pairs.  It is the graph-level
construction BLMW's tree lifting uses (curves and trees pulled back along a
map), and what `transport_pair` uses to move a whole session across a
Whitehead move.
"""

from fractions import Fraction as F

from .ribbons import RibbonGraph, F as FORWARD, B as BACKWARD
from .refined import Graph, GraphMap, inv, graph_from_ribbon
from .iterate_graphs import refine_to_single_darts

__all__ = ["collapse_edge", "expand_vertex", "whitehead_move",
           "pullback", "compose", "transport_pair"]


# --------------------------------------------------------------- rotations
def _rotate(ds, start, count):
    n = len(ds)
    return tuple(ds[(start + i) % n] for i in range(count))


def _sigma_inv(graph, dart):
    ds = graph.flowers[graph.src(dart)]
    return ds[ds.index(dart) - 1]


def _merged_label(u, v, edge):
    if u == v:
        return u
    return ("~",) + tuple(sorted((str(u), str(v)))) + (edge,)


def _merged_flower(graph, edge, u, v):
    """(merged cyclic order, the tail's arc, the head's arc).

    The arcs are the darts at the tail (after the collapsed dart) and at the
    head (after its reverse); their concatenation is the merged order.
    """
    d, e = (edge, FORWARD), (edge, BACKWARD)
    sigma = graph.sigma
    A, B = [], []
    if u != v:
        cur = sigma(d)
        while cur != d:
            A.append(cur)
            cur = sigma(cur)
        cur = sigma(e)
        while cur != e:
            B.append(cur)
            cur = sigma(cur)
    else:
        pred_d, pred_e = _sigma_inv(graph, d), _sigma_inv(graph, e)
        cur = sigma(pred_e)                # succ(e+), around to pred(e-)
        while cur != pred_e:
            A.append(cur)
            cur = sigma(cur)
        cur = sigma(pred_d)                # succ(e-), around to pred(e+)
        while cur != pred_d:
            B.append(cur)
            cur = sigma(cur)
    return tuple(A) + tuple(B), tuple(A), tuple(B)


def collapse_edge(graph, edge):
    """Collapse `edge`; return (R, chains) with chains the collapse map
    graph -> R (each surviving edge to its own dart, the collapsed edge to
    the empty chain)."""
    u, v = graph.src((edge, FORWARD)), graph.tgt((edge, FORWARD))
    merged, _A, _B = _merged_flower(graph, edge, u, v)
    flowers = {w: ds for w, ds in graph.flowers.items() if w != u and w != v}
    flowers[_merged_label(u, v, edge)] = merged
    R = RibbonGraph(flowers)
    chains = {e: ((e, FORWARD),) for e in R.edge_list()}
    chains[edge] = ()
    return R, chains


def expand_vertex(graph, vertex, arcs, new_edge):
    """Expand `vertex` along the pair of arcs `arcs` = (A, B).

    A and B are disjoint cyclic arcs of the flower covering it; the new edge
    `new_edge` joins A's vertex (its forward dart there) to B's.

    Returns (G', chains) with chains the expansion map graph -> G': the
    merged vertex maps to A's vertex; an edge with both darts in A maps to
    itself, with both in B to new_edge . e . new_edge^-1, and a crossing
    edge to its own dart out of A's vertex followed by the new edge back.
    """
    A, B = tuple(arcs[0]), tuple(arcs[1])
    ds = graph.flowers[vertex]
    if sorted(d for d in A + B) != sorted(ds):
        raise ValueError("the arcs must partition the flower")
    vA, vB = (vertex, 0), (vertex, 1)
    flowers = {w: d for w, d in graph.flowers.items() if w != vertex}
    flowers[vA] = A + ((new_edge, FORWARD),)
    flowers[vB] = B + ((new_edge, BACKWARD),)
    Gp = RibbonGraph(flowers)
    side = {d: 0 for d in A}
    side.update({d: 1 for d in B})
    chains = {}
    for e in graph.edge_list():
        fwd, bwd = (e, FORWARD), (e, BACKWARD)
        sf, sb = side.get(fwd, -1), side.get(bwd, -1)
        if sf == 0 and sb == 0:
            chains[e] = (fwd,)
        elif sf == 1 and sb == 1:
            chains[e] = ((new_edge, FORWARD), fwd, (new_edge, BACKWARD))
        elif sf == 0:
            # crossing: closed path from A's vertex through the edge and back
            chains[e] = (fwd, (new_edge, BACKWARD))
        else:
            chains[e] = ((e, BACKWARD), (new_edge, BACKWARD))
    return Gp, chains


def _loop_closed_splits(graph, vertex):
    """The cyclic splits of `vertex` whose arcs are unions of whole edges."""
    ds = graph.flowers[vertex]
    n = len(ds)
    pos = {}
    for i, d in enumerate(ds):
        pos.setdefault(d[0], set()).add(i)
    out = []
    for start in range(n):
        for count in range(1, n):
            idx = {(start + i) % n for i in range(count)}
            if all(p <= idx or not (p & idx) for p in pos.values()):
                split = (start, count)
                other = ((start + count) % n, n - count)
                if split <= other:
                    out.append(split)
    return out


def whitehead_move(graph, edge, new_edge=None):
    """The ribbon-preserving Whitehead move on `edge`.

    Returns (R, c, moved, exp, exp2):
      R      the intermediate graph,
      c      the collapse graph -> R (chains),
      moved  the moved RibbonGraph (the collapsed edge replaced by the new
             edge `new_edge`, default: the same label),
      exp    the loop-preserving expansion R -> moved (a homotopy inverse
             of the collapse of the new edge),
      exp2   the inverse collapse R -> graph (a homotopy inverse of c).
    Raises if the loop-preserving split is not unique.
    """
    if new_edge is None:
        new_edge = edge
    u, v = graph.src((edge, FORWARD)), graph.tgt((edge, FORWARD))
    R, c = collapse_edge(graph, edge)
    w = _merged_label(u, v, edge)
    _merged, arcA, arcB = _merged_flower(graph, edge, u, v)
    splits = _loop_closed_splits(R, w)
    if len(splits) != 1:
        raise ValueError(
            f"the Whitehead move on {edge!r} is not unique: the merged vertex "
            f"has {len(splits)} loop-preserving splits {splits}")
    start, count = splits[0]
    ds = R.flowers[w]
    moved, exp = expand_vertex(R, w, (_rotate(ds, start, count),
                                      _rotate(ds, start + count, len(ds) - count)),
                               new_edge)
    back, exp2 = expand_vertex(R, w, (arcA, arcB), edge)
    return R, c, moved, exp, exp2


# ------------------------------------------------------------ graph pullback
def compose(g, f):
    """The composite map g . f (chains wired through g)."""
    src = f.src
    edges = src.edge_list() if hasattr(src, "edge_list") else list(src.edges)
    chains = {e: g.image_word(f.chains[e]) for e in edges}
    return GraphMap(f.src, g.tgt, chains)


def pullback(pi, f):
    """The fibre product of two graph maps with a common target.

    pi: C -> B and f: A -> B.  Both are refined to single-dart chains (a
    constant edge stays constant).  Returns (P, proj_C, proj_A): the product
    graph and the two projections, with pi . proj_C = f . proj_A exactly.

    Edges of P: (a) pairs of edges (ec, ea) whose images are the same dart
    up to orientation; (b) (ec, a) with ec constant, a a vertex of A over
    its image; (c) (c, ea) with ea constant and c a vertex of C over its
    image.  A pair that is constant on BOTH sides would give a 2-cell and
    is refused (it does not occur with a covering on one side).
    """
    C, back_c, ch_c = refine_to_single_darts(pi.src, pi.tgt, pi.chains)
    A, back_a, ch_a = refine_to_single_darts(f.src, f.tgt, f.chains)
    B = pi.tgt
    pic = GraphMap(C, B, ch_c, name=f"refined({pi.name})")
    fia = GraphMap(A, B, ch_a, name=f"refined({f.name})")
    vc, va = pic.vmap, fia.vmap
    vset = {(c, a) for c in C.deg for a in A.deg
            if vc.get(c) is not None and vc.get(c) == va.get(a)}
    edges, pc, pa, alpha = {}, {}, {}, {}
    Ce, Ae = sorted(C.edges, key=str), sorted(A.edges, key=str)
    # (a) both edges non-constant: the images must be the same dart
    for ec in Ce:
        dc = pic.chains[ec]
        if not dc:
            continue
        for ea in Ae:
            da = fia.chains[ea]
            if not da:
                continue
            if dc[0] == da[0]:
                u = (C.tail((ec, FORWARD)), A.tail((ea, FORWARD)))
                w = (C.head((ec, FORWARD)), A.head((ea, FORWARD)))
                lab = ("E", ec, ea, 0)
                pc[lab], pa[lab] = ((ec, FORWARD),), ((ea, FORWARD),)
            elif dc[0] == inv(da[0]):
                u = (C.tail((ec, FORWARD)), A.head((ea, FORWARD)))
                w = (C.head((ec, FORWARD)), A.tail((ea, FORWARD)))
                lab = ("E", ec, ea, 1)
                pc[lab], pa[lab] = ((ec, FORWARD),), ((ea, BACKWARD),)
            else:
                continue
            if u not in vset or w not in vset:
                raise ValueError(f"product edge {lab} misses the vertex set")
            edges[lab] = (u, w)
            alpha[lab] = A.alpha[ea]
    # (b) a constant edge of A: {c} x ea for each vertex c of C over its image
    for ea in Ae:
        if fia.chains[ea]:
            continue
        p = va.get(A.tail((ea, FORWARD)))
        for c in sorted(C.deg, key=str):
            if vc.get(c) != p:
                continue
            u = (c, A.tail((ea, FORWARD)))
            w = (c, A.head((ea, FORWARD)))
            lab = ("A", c, ea)
            if u in vset and w in vset:
                edges[lab] = (u, w)
                alpha[lab] = A.alpha[ea]
                pc[lab], pa[lab] = (), ((ea, FORWARD),)
    # (c) a constant edge of C: ec x {a} for each vertex a of A over its image
    for ec in Ce:
        if pic.chains[ec]:
            continue
        p = vc.get(C.tail((ec, FORWARD)))
        for a in sorted(A.deg, key=str):
            if va.get(a) != p:
                continue
            u = (C.tail((ec, FORWARD)), a)
            w = (C.head((ec, FORWARD)), a)
            lab = ("C", ec, a)
            if u in vset and w in vset:
                edges[lab] = (u, w)
                alpha[lab] = C.alpha[ec]
                pc[lab], pa[lab] = ((ec, FORWARD),), ()
    P = Graph(edges, alpha=alpha, name=f"pullback({pi.name},{f.name})")
    mp_c = GraphMap(P, C, pc, name="proj_C")
    mp_a = GraphMap(P, A, pa, name="proj_A")
    return P, mp_c, mp_a


# ------------------------------------------------------------- the transport
def transport_pair(base, cover, phi_chains, edge, new_edge=None):
    """Move the spine `base` across the Whitehead move on `edge`.

    base, cover: ribbons.RibbonGraph (the session's base and cover); the
    covering pi: cover -> base is by labels; phi_chains: {cover edge: chain}
    on the base.

    Returns (base2, cover2, pi2, phi2, s, k):
      base2   the moved RibbonGraph,
      cover2  the pulled-back cover (a refined.Graph),
      pi2     the new covering cover2 -> base2 (a GraphMap),
      phi2    the new map cover2 -> base2 (a GraphMap),
      s, k    the two halves of the move: s = h . c': G_0' -> G_0 and
              k = h' . c: G_0 -> G_0' (GraphMaps, used for checking).
    The pair presents the same surface map on the moved spine.
    """
    G0 = graph_from_ribbon(base)
    G1 = graph_from_ribbon(cover)
    pi = GraphMap(G1, G0, {e: ((e[0], FORWARD),) for e in G1.edges}, name="pi")
    phi = GraphMap(G1, G0, {e: tuple(phi_chains.get(e, ())) for e in G1.edges},
                   name="phi")
    R, c, moved, exp, exp2 = whitehead_move(base, edge, new_edge)
    if new_edge is None:
        new_edge = edge
    GR = graph_from_ribbon(R)
    G0p = graph_from_ribbon(moved)
    cap = GraphMap(GR, G0, {e: tuple(v) for e, v in exp2.items()}, name="h")
    cp_chains = {e: ((e, FORWARD),) for e in R.edge_list()}
    cp_chains[new_edge] = ()
    cpr = GraphMap(G0p, GR, cp_chains, name="c'")
    s = compose(cap, cpr)                       # G_0' -> G_0
    hp = GraphMap(GR, G0p, {e: tuple(v) for e, v in exp.items()}, name="h'")
    cg = GraphMap(G0, GR, {e: tuple(v) for e, v in c.items()}, name="c")
    k = compose(hp, cg)                         # G_0 -> G_0'
    P, proj_c, proj_p = pullback(pi, s)
    # the pullback's A-projection lands on the refined G_0'; compose with the
    # refinement's back map to land on the unrefined base2
    _A = proj_p.tgt
    _back = {}
    for e in _A.edges:
        orig = e[0] if isinstance(e, tuple) else e
        _back[e] = ((orig, FORWARD),)
    unref = GraphMap(_A, G0p, _back, name="unrefine")
    pi2 = compose(unref, proj_p)
    phi2 = compose(k, compose(phi, proj_c))
    return moved, P, pi2, phi2, s, k
