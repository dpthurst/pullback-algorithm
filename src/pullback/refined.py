"""Marked elastic graphs, PL maps and refined (constant-derivative) maps.

The data model of the M3 energy minimiser (plan items 160-210).  A map
between elastic graphs is given by chains of target darts per source edge
with vertices mapping to vertices -- exactly the session data model of
`pullback.energies`.  A *constant-derivative* map in [Elast]'s sense
(lipschitz.tex def:const-deriv: edge-reduced and |f'| constant on each edge
of the source, the source being refined at the preimages of the target's
vertices) is instead data on a refined source and target; that is
`RefinedMap` here, and it is what a minimiser returns: [Elast] Example 1.17's
minimiser sends a vertex to 2/5 of the way along a target edge, which the
chain model cannot express (the unrefined reading of the same map gives
14/15, not the published 3/5 -- see tests/test_energies.py TestEx117).

The class [phi] is part of the input: `marked` is the set of source
vertices whose images are pinned.  Marked and free homotopy classes give
different answers (the paper's examples are marked; the repo's sessions are
free -- see AGENTS.md "M3 readiness" and the session schema).  A third
ingredient, orbifold (cone) points, is reserved in the session schema
(`orbifold`: [{vertex, order}]) and is independent of `marked`: a graph may
carry both.  Nothing here uses cone points yet.

Sources: [Elast] = arXiv:1607.00340 (intro.tex eq:emb, eq:emb-homotopy,
def:curve, def:el-curve; lipschitz.tex def:const-deriv).
"""

from __future__ import annotations

from fractions import Fraction as F

__all__ = [
    "Graph", "GraphMap", "RefinedMap", "subdivide", "piece_label",
    "inv", "free_reduce", "emb_of_map", "same_class", "fundamental_loops",
    "graph_from_ribbon", "session_map",
]


# ---------------------------------------------------------------- chains
def inv(dart):
    """The inverse dart."""
    return (dart[0], -dart[1])


def free_reduce(word):
    """Free reduction: cancel adjacent inverse darts.  Never changes |f'|."""
    out = []
    for d in word:
        if out and out[-1] == (d[0], -d[1]):
            out.pop()
        else:
            out.append(d)
    return tuple(out)


# ---------------------------------------------------------------- graphs
class Graph:
    """A marked elastic graph: edges with two endpoints, a positive rational
    q-length alpha on each edge, and a set of marked vertices (whose images
    the homotopy class pins).

    edges: {edge label: (tail, head)}; the dart (e, +1) runs tail -> head.
    alpha: a dict {edge: Fraction} or a single Fraction for all edges.
    marked: vertices whose image is pinned; defaults to the 1-valent
    vertices ([Elast] basic.tex: marked points are 1-valent vertices).
    """

    def __init__(self, edges, alpha=None, marked=None, name=""):
        self.edges = {e: (u, v) for e, (u, v) in edges.items()}
        self.name = name
        if alpha is None:
            alpha = F(1)
        if isinstance(alpha, dict):
            missing = [e for e in self.edges if e not in alpha]
            if missing:
                raise ValueError(f"no q-length for edges {missing}")
            self.alpha = {e: F(alpha[e]) for e in self.edges}
        else:
            self.alpha = {e: F(alpha) for e in self.edges}
        for e, a in self.alpha.items():
            if a <= 0:
                raise ValueError(f"q-length must be positive, edge {e}: {a}")
        self.deg = {}
        for e in sorted(self.edges, key=str):
            u, v = self.edges[e]
            if u == v:
                self.deg[u] = self.deg.get(u, 0) + 2
            else:
                self.deg[u] = self.deg.get(u, 0) + 1
                self.deg[v] = self.deg.get(v, 0) + 1
        if marked is None:
            marked = [v for v in self.deg if self.deg[v] == 1]
        self.marked = frozenset(marked)
        for v in self.marked:
            if v not in self.deg:
                raise ValueError(f"marked vertex {v!r} is not a vertex")

    # -- dart arithmetic
    def tail(self, dart):
        u, v = self.edges[dart[0]]
        return u if dart[1] == 1 else v

    def head(self, dart):
        u, v = self.edges[dart[0]]
        return v if dart[1] == 1 else u

    def darts(self):
        """All darts, deterministically ordered."""
        return [(e, s) for e in sorted(self.edges, key=str) for s in (1, -1)]

    def out_darts(self, v):
        return [d for d in self.darts() if self.tail(d) == v]

    def vertices(self):
        return sorted(self.deg, key=str)

    def check_word(self, word, closed=None):
        """Check that `word` is a dart path in this graph."""
        for a, b in zip(word, word[1:]):
            if self.head(a) != self.tail(b):
                return False
        if closed is True and (not word or self.tail(word[0]) != self.head(word[-1])):
            return False
        return True

    def volume(self):
        return sum(self.alpha.values(), F(0))

    def __repr__(self):
        return f"Graph({self.name!r}, {len(self.edges)} edges, marked={sorted(self.marked, key=str)})"


# ---------------------------------------------------------------- maps
class GraphMap:
    """A PL graph map given by its chains: {source edge: tuple of target darts}.

    Vertices map to vertices (the session data model); a map sending a
    vertex to the interior of a target edge is a RefinedMap instead.
    """

    def __init__(self, src, tgt, chains, name=""):
        self.src, self.tgt, self.name = src, tgt, name
        ch = {}
        for e, seq in chains.items():
            if e not in src.edges:
                raise ValueError(f"chain for unknown source edge {e!r}")
            ch[e] = free_reduce(tuple(seq))
        for e in sorted(src.edges, key=str):
            ch.setdefault(e, ())
        self.chains = ch
        self.vmap = self._vertex_images()
        for e in sorted(src.edges, key=str):
            u, v = src.edges[e]
            seq = ch[e]
            a = self.vmap[u] if not seq else self.tgt.tail(seq[0])
            b = self.vmap[v] if not seq else self.tgt.head(seq[-1])
            if self.vmap[u] != a or self.vmap[v] != b:
                raise ValueError(f"chain of edge {e!r} does not join the images "
                                 f"of its endpoints ({self.vmap[u]},{self.vmap[v]}) "
                                 f"vs ({a},{b})")

    def _vertex_images(self):
        """{source vertex: image vertex}, or None where the map is constant on
        every incident chain and no image is forced."""
        out = {}
        for e in sorted(self.src.edges, key=str):
            seq = self.chains[e]
            if not seq:
                continue
            u, v = self.src.edges[e]
            for vert, img in ((u, self.tgt.tail(seq[0])), (v, self.tgt.head(seq[-1]))):
                if vert in out and out[vert] != img:
                    raise ValueError(f"vertex {vert!r} has two images: "
                                     f"{out[vert]!r} and {img!r}")
                out[vert] = img
        changed = True
        while changed:
            changed = False
            for e in sorted(self.src.edges, key=str):
                u, v = self.src.edges[e]
                if self.chains[e]:
                    continue
                if u in out and v not in out:
                    out[v] = out[u]
                    changed = True
                elif v in out and u not in out:
                    out[u] = out[v]
                    changed = True
                elif u in out and v in out and out[u] != out[v]:
                    raise ValueError(f"constant edge {e!r} maps to a point but "
                                     f"its endpoints map to {out[u]!r} and {out[v]!r}")
        for v in self.src.deg:
            out.setdefault(v, None)
        return out

    def image_word(self, word):
        """The image of a dart word under the map, freely reduced."""
        out = []
        for e, s in word:
            seq = self.chains[e]
            out.extend(seq if s == 1 else [inv(d) for d in reversed(seq)])
        return free_reduce(out)

    def emb(self):
        """Emb of this explicit representative, exact (intro.tex eq:emb)."""
        return emb_of_map(self.chains, self.src, self.tgt)

    def __repr__(self):
        return f"GraphMap({self.name!r}, {len(self.chains)} edges)"


def emb_of_map(chains, src, tgt):
    """Emb of an explicit representative, exactly (a Fraction).

    Fill^2 on a target edge d is the sum over source edges mapping over d
    of |phi'(e)|, with |phi'| = (length of the image chain)/alpha_src(e);
    the essential supremum is the max over target edges (images of vertices
    carry no measure, intro.tex eq:emb).
    """
    fills = {}
    for e in sorted(chains, key=str):
        seq = chains[e]
        if not seq:
            continue
        length = sum((tgt.alpha[d[0]] for d in seq), F(0))
        dil = length / src.alpha[e]
        for d in seq:
            fills[d[0]] = fills.get(d[0], F(0)) + dil
    return max(fills.values()) if fills else F(0)


# ------------------------------------------------------- refined maps
def subdivide(graph, splits, tag=""):
    """Subdivide every edge of `graph` into the given rational pieces.

    splits: {edge: [length, length, ...]} summing to alpha(edge).  Returns
    (refined graph, {refined edge: base edge}).  Piece i of edge e is the
    refined edge (e, i); the pieces are joined in order along e.
    """
    edges, alpha, back = {}, {}, {}
    for e in sorted(graph.edges, key=str):
        pieces = [F(x) for x in splits.get(e, [graph.alpha[e]])]
        if sum(pieces, F(0)) != graph.alpha[e]:
            raise ValueError(f"pieces of edge {e!r} sum to {sum(pieces, F(0))}, "
                             f"not alpha={graph.alpha[e]}")
        if any(p <= 0 for p in pieces):
            raise ValueError(f"pieces of edge {e!r} must be positive")
        u, v = graph.edges[e]
        prev = u
        for i, p in enumerate(pieces):
            nxt = v if i == len(pieces) - 1 else (tag or "s", e, i)
            lab = (e, i)
            edges[lab] = (prev, nxt)
            alpha[lab] = p
            back[lab] = e
            prev = nxt
    rg = Graph(edges, alpha, marked=graph.marked, name=f"{graph.name}{tag}~sub")
    return rg, back


def piece_label(edge, i):
    """The label of piece i of `edge` in a subdivided graph."""
    return (edge, i)


class RefinedMap:
    """A constant-derivative representative: a base map plus subdivisions of
    the source (at the preimages of the target's vertices) and of the target
    (at the images of the source's vertices), and one chain per refined
    source edge (lipschitz.tex def:const-deriv).

    src_splits / tgt_splits: {base edge: [rational lengths]}; chains:
    {refined source edge: tuple of refined target darts}.
    """

    def __init__(self, base, src_splits, tgt_splits, chains, name=""):
        self.base = base
        self.name = name
        self.src_sub, self.src_back = subdivide(base.src, src_splits, tag="s")
        self.tgt_sub, self.tgt_back = subdivide(base.tgt, tgt_splits, tag="t")
        self.rmap = GraphMap(self.src_sub, self.tgt_sub, chains,
                             name=name or base.name)

    def emb(self):
        """Emb of this representative: exact."""
        return self.rmap.emb()

    def expand(self, word):
        """Lift a word of *base* source darts to a word of refined source
        darts (each base dart runs over its pieces in order)."""
        out = []
        for e, s in word:
            idxs = []
            i = 0
            while piece_label(e, i) in self.src_sub.edges:
                idxs.append(i)
                i += 1
            if not idxs:
                raise ValueError(f"base source edge {e!r} has no pieces")
            for i in (idxs if s == 1 else list(reversed(idxs))):
                out.append((piece_label(e, i), s))
        return tuple(out)

    def merged_base_word(self, word):
        """Forget the target subdivision: consecutive refinements of one base
        dart are merged into that dart."""
        out = []
        for d in word:
            e = self.tgt_back[d[0]]
            s = int(d[1])
            idx = d[0][1]
            if out and out[-1][0] == e and out[-1][1] == s:
                prev = out[-1][2]
                if (s == 1 and idx == prev + 1) or (s == -1 and idx == prev - 1):
                    out[-1] = (e, s, idx)
                    continue
            out.append((e, s, idx))
        return free_reduce([(e, s) for e, s, _ in out])

    def collapse(self):
        """The base-level chains of this representative: forget both
        subdivisions."""
        chains = {}
        for e in sorted(self.base.src.edges, key=str):
            chains[e] = self.merged_base_word(
                self.rmap.image_word(self.expand(((e, 1),))))
        return chains

    def in_class(self):
        """Sound certificate that this representative lies in [phi].

        Basepoint: the first marked source vertex (else the first vertex).
        Required: (1) every marked source vertex maps to the same point of
        the *base* target as under phi; (2) the same for the basepoint; and
        (3) for every loop in a free basis of pi_1 of the source, the
        induced element of pi_1 of the target agrees with phi's.  For a tree
        source pi_1 is trivial and (1)-(2) are the whole condition, which is
        exactly right: a map of a tree with fixed marked images is
        determined up to homotopy.
        """
        base = self.base
        root = _basepoint(base.src)
        wroot = self.rmap.vmap.get(root)
        if wroot is None or wroot not in base.tgt.deg:
            return False
        if wroot != base.vmap.get(root):
            return False
        for v in sorted(base.src.marked, key=str):
            img = self.rmap.vmap.get(v)
            if img is None or img not in base.tgt.deg:
                return False
            if img != base.vmap.get(v):
                return False
        _r, _parent, basis = fundamental_loops(base.src, root)
        _, tparent, _ = fundamental_loops(base.tgt)

        def pi1(img_of):
            out = {}
            for g in basis:
                words = []
                for d in img_of(g):
                    words.extend(loop_word(base.tgt, d, tparent, base.vmap[root]))
                out[tuple(g)] = free_reduce(tuple(words))
            return out

        here = pi1(lambda g: self.merged_base_word(self.rmap.image_word(self.expand(g))))
        there = pi1(lambda g: base.image_word(g))
        return here == there


# --------------------------------------------------------------- pi_1 data
def _tree(graph, root):
    """A BFS spanning tree: {vertex: (parent vertex, dart from parent)}."""
    parent = {root: None}
    order = [root]
    i = 0
    while i < len(order):
        v = order[i]
        i += 1
        for d in graph.out_darts(v):
            w = graph.head(d)
            if w not in parent:
                parent[w] = (v, d)
                order.append(w)
    if len(parent) != len(graph.deg):
        raise ValueError(f"graph {graph.name!r} is not connected")
    return parent


def _tree_path(parent, v):
    """The reduced dart word from the root to v inside the spanning tree."""
    out = []
    while parent[v] is not None:
        v, d = parent[v]
        out.append(d)
    out.reverse()
    return tuple(out)


def _basepoint(graph):
    if graph.marked:
        return sorted(graph.marked, key=str)[0]
    return sorted(graph.deg, key=str)[0]


def loop_word(graph, dribble, parent, root):
    """The word of a dart of `graph` as a loop at `root`, reduced."""
    d = (dribble[0], int(dribble[1]))
    out = list(_tree_path(parent, graph.tail(d)))
    out.append(d)
    out.extend(inv(x) for x in reversed(_tree_path(parent, graph.head(d))))
    return free_reduce(out)


def fundamental_loops(graph, root=None):
    """A free basis of pi_1: for each edge outside a spanning tree, the loop
    word at the root (deterministic order)."""
    if root is None:
        root = _basepoint(graph)
    parent = _tree(graph, root)
    tree_edges = {d[0] for v, pd in parent.items() if pd is not None for d in [pd[1]]}
    basis = []
    for e in sorted(graph.edges, key=str):
        if e in tree_edges:
            continue
        u, v = graph.edges[e]
        basis.append(loop_word(graph, (e, 1), parent, root))
    return root, parent, basis


def _pi1_image(m, root, parent, basis):
    """{basis loop: its image as a reduced word in the target's basis darts}."""
    troot, tparent, _ = fundamental_loops(m.tgt)
    out = {}
    for g in basis:
        img = m.image_word(g)
        words = []
        for d in img:
            words.extend(loop_word(m.tgt, d, tparent, m.vmap[root]))
        out[tuple(g)] = free_reduce(tuple(words))
    return out


def same_class(m1, m2):
    """Sound check that two maps between the same marked graphs are
    homotopic: same image of the basepoint, same images of the marked
    vertices, and the same induced map on pi_1 at a common basepoint."""
    if (sorted(m1.src.edges, key=str) != sorted(m2.src.edges, key=str)
            or sorted(m1.tgt.edges, key=str) != sorted(m2.tgt.edges, key=str)):
        raise ValueError("same_class: the two maps have different graphs")
    root = _basepoint(m1.src)
    if m1.vmap.get(root) is None or m2.vmap.get(root) is None:
        raise ValueError("same_class: basepoint image is not determined")
    if m1.vmap[root] != m2.vmap[root]:
        return False
    for v in sorted(m1.src.marked, key=str):
        if m1.vmap.get(v) != m2.vmap.get(v):
            return False
    r, parent, basis = fundamental_loops(m1.src, root)
    return _pi1_image(m1, r, parent, basis) == _pi1_image(m2, r, parent, basis)


# ------------------------------------------------------- session bridge
def graph_from_ribbon(R):
    """A Graph in this module's sense from a pullback.ribbons.RibbonGraph:
    every vertex marked (the applet's marked graph is the graph itself)."""
    edges = {}
    for d in R.darts():
        if d[1] == 1:
            edges[d[0]] = (R.src(d), R.tgt(d))
    return Graph(edges, marked=set(R.vertices()), name=R.name if hasattr(R, "name") else "")


def session_map(path, src_root=None, marked_base=()):
    """Read a session as a (source Graph, GraphMap, target Graph).

    The source is the cover with alpha_1 the lift of alpha_0 (unit by
    default), the target is the base graph with the same elastic constants.
    `marked_base` lists BASE vertices whose lifts are pinned (a marked base
    vertex v requires every lift of v to map to v); the default is the free
    homotopy class, matching the published values for the bouquet (the
    sessions' convention, plan item 90/320).
    """
    import json
    from .harmonic_io import load_session
    if src_root is not None:  # pragma: no cover - convenience for scratch use
        import sys
        if str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))
    with open(path) as f:
        sess = json.load(f)
    marked_base = set(marked_base) | set(sess.get("marked", []))
    base, cover, phi = load_session(path)
    G2 = graph_from_ribbon(base)
    G1 = graph_from_ribbon(cover)
    G2.marked = frozenset()
    for v in sorted(marked_base, key=str):
        if v not in base.vertices():
            raise ValueError(f"marked vertex {v!r} is not a vertex of the base graph")
    # pin the lifts of the marked base vertices
    pinned = {u for u in cover.vertices() if u[0] in marked_base}
    G1.marked = frozenset(pinned)
    chains = {}
    for e in G1.edges:
        ch = phi.chains.get(e, ())
        chains[e] = tuple((lab, ori) for lab, ori in ch)
    alpha1 = {e: F(1) for e in G1.edges}
    alpha2 = {e: F(1) for e in G2.edges}
    if sess.get("lengths"):
        alpha2 = {e: F(x) for e, x in zip(sorted(G2.edges, key=str), sess["lengths"])}
    if sess.get("elastics"):
        alpha1 = {e: F(x) for e, x in zip(sorted(G1.edges, key=str), sess["elastics"])}
    m = GraphMap(G1, G2, chains, name=sess.get("name", "session"))
    return G1, m, G2
