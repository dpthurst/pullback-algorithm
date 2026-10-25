"""Lower bounds on Emb[phi] from the extremal-length side of [Elast]
intro.tex thm:emb-sf:  Emb[phi] = SF_EL[phi], where

    SF_EL[phi] = sup over marked multi-curves c: C -> G1 of
                 EL[phi o c] / EL[c],

and EL(c) = sum_e alpha(e) n_c(e)^2 (intro.tex def:el-curve, eq:sf-el),
n_c(e) the number of preimages of a regular point of e (weighted by the
components' weights).  So *every* reduced marked multi-curve whose image
is also reduced gives a rigorous lower bound EL[phi o c]/EL[c] <= Emb[phi],
computed here in exact rational arithmetic: no harmonic maps, no length
iteration, no floating point.

The search is a search (its bounds are inputs); every number it reports is
exact and certified.  This is the lower-bound half of the applet's
"undecided" state (plan items 180/200): together with an explicit
representative (an upper bound) it gives the certified bracket of plan D5.
"""

from fractions import Fraction as F
from itertools import combinations

from .refined import Graph, GraphMap, free_reduce, inv

__all__ = ["Curve", "enumerate_components", "all_weightings", "lower_bound",
           "curve_ratio", "bracket"]


class Curve:
    """A weighted marked multi-curve on `graph`.

    components: [(weight, dart word)], each word reduced; a closed word is a
    loop (its vertices must avoid the marked points), an open word is an arc
    between two marked vertices with no marked vertex in its interior
    ([Elast] intro.tex def:curve).
    """

    def __init__(self, graph, components, name=""):
        self.graph = graph
        self.name = name
        comp = []
        for w, word in components:
            w = int(w)
            if w <= 0:
                raise ValueError(f"weights must be positive, got {w}")
            word = tuple((d[0], int(d[1])) for d in word)
            if free_reduce(word) != word:
                raise ValueError(f"component {word} is not reduced")
            if not word:
                raise ValueError("empty component")
            if not graph.check_word(word):
                raise ValueError(f"component {word} is not a path in {graph!r}")
            closed = graph.tail(word[0]) == graph.head(word[-1])
            if closed:
                bad = [v for v in {graph.tail(d) for d in word} if v in graph.marked]
                if bad:
                    raise ValueError(f"loop {word} passes through marked vertices {bad}")
            else:
                if graph.tail(word[0]) not in graph.marked:
                    raise ValueError(f"arc {word} does not start at a marked vertex")
                if graph.head(word[-1]) not in graph.marked:
                    raise ValueError(f"arc {word} does not end at a marked vertex")
                inner = [graph.head(d) for d in word[:-1]]
                bad = [v for v in inner if v in graph.marked]
                if bad:
                    raise ValueError(f"arc {word} passes through marked vertices {bad}")
            comp.append((w, word))
        self.components = tuple(sorted(comp))

    def counts(self):
        """n_c: {edge: weighted number of preimages} on the source graph."""
        out = {}
        for w, word in self.components:
            for d in word:
                out[d[0]] = out.get(d[0], 0) + w
        return out

    def EL(self):
        """The extremal length of the curve, exact: sum alpha(e) n_c(e)^2."""
        c = self.counts()
        return sum((self.graph.alpha[e] * c[e] * c[e] for e in c), F(0))

    def image(self, m):
        """phi o c as a multi-curve on the target (each component freely
        reduced), with the same weights."""
        comp = []
        for w, word in self.components:
            img = m.image_word(word)
            if not img:
                continue  # a component that collapses to a point carries nothing
            comp.append((w, img))
        if not comp:
            return None
        return Curve(m.tgt, comp, name=f"{self.name}|phi")

    def __repr__(self):
        return f"Curve({self.name!r}, {[(w, len(c)) for w, c in self.components]})"


def curve_ratio(m, c):
    """EL[phi o c] / EL[c] <= Emb[phi], exactly; None if the image vanishes."""
    img = c.image(m)
    if img is None:
        return None
    return img.EL() / c.EL()


def enumerate_components(graph, max_darts=6):
    """All reduced marked arcs and loops of `graph` with at most max_darts
    darts, up to rotation (loops) and reversal (arcs).  Deterministic order.

    An arc runs between two marked vertices with no marked vertex in its
    interior; a loop is a closed reduced walk avoiding the marked vertices
    ([Elast] intro.tex def:curve).
    """
    arcs, loops = {}, {}

    def canon_path(word):
        rev = tuple(inv(d) for d in reversed(word))
        return min(word, rev)

    def canon_loop(word):
        n = len(word)
        cands = [tuple(word[i:] + word[:i]) for i in range(n)]
        rev = tuple(inv(d) for d in reversed(word))
        cands += [tuple(rev[i:] + rev[:i]) for i in range(n)]
        return min(cands)

    for start in sorted(graph.marked, key=str):
        stack = [(start, ())]
        while stack:
            v, word = stack.pop()
            for d in graph.out_darts(v):
                word2 = word + (d,)
                if free_reduce(word2) != word2:
                    continue
                w = graph.head(d)
                if w in graph.marked:
                    arcs.setdefault(canon_path(word2), word2)
                    continue
                if len(word2) < max_darts:
                    stack.append((w, word2))
    for start in sorted(graph.deg, key=str):
        if start in graph.marked:
            continue
        stack = [(start, ())]
        while stack:
            v, word = stack.pop()
            if len(word) >= max_darts:
                continue
            for d in graph.out_darts(v):
                w = graph.head(d)
                if w in graph.marked:
                    continue
                word2 = word + (d,)
                if free_reduce(word2) != word2:
                    continue
                if w == start:
                    if len(word2) >= 2:
                        loops.setdefault(canon_loop(word2), word2)
                    continue
                stack.append((w, word2))
    return ([arcs[k] for k in sorted(arcs, key=str)]
            + [loops[k] for k in sorted(loops, key=str)])


def all_weightings(count, max_parts, max_weight):
    """Weight vectors over `count` components: up to max_parts distinct
    components with weights in 1..max_weight, in a deterministic order."""
    for k in range(1, min(max_parts, count) + 1):
        for support in combinations(range(count), k):
            yield from _weights(support, max_weight)


def _weights(support, max_weight):
    if not support:
        yield ()
        return
    head, rest = support[0], support[1:]
    for w in range(1, max_weight + 1):
        for tail in _weights(rest, max_weight):
            yield ((head, w),) + tail


def lower_bound(m, max_darts=6, top_k=8, max_parts=3, max_weight=3,
                extra=()):
    """The best certified lower bound on Emb from curves: returns
    (ratio, Curve) -- the curve with the largest exact EL ratio found among
    the marked multi-curves of the source with <= max_darts darts per
    component, combining up to max_parts of the `top_k` best single
    components with weights <= max_weight, plus any curves in `extra`."""
    comps = enumerate_components(m.src, max_darts=max_darts)
    scored = []
    for word in comps:
        try:
            c = Curve(m.src, [(1, word)])
        except ValueError:
            continue
        r = curve_ratio(m, c)
        if r is None or r <= 0:
            continue
        scored.append((r, word))
    scored.sort(key=lambda t: (-t[0], str(t[1])))
    pool = [w for _, w in scored[:top_k]]
    best = (scored[0][0], Curve(m.src, [(1, scored[0][1])], name="lb")) if scored else None
    for x in all_weightings(len(pool), max_parts, max_weight):
        if not x:
            continue
        comp = [(w, pool[i]) for i, w in x]
        try:
            c = Curve(m.src, comp, name="lb")
            r = curve_ratio(m, c)
        except ValueError:
            continue
        if r is not None and (best is None or r > best[0]):
            best = (r, c)
    for c in extra:
        r = curve_ratio(m, c)
        if r is not None and (best is None or r > best[0]):
            best = (r, c)
    return best


def bracket(m, witnesses=(), **kw):
    """(lower, upper, exact?) : lower from lower_bound, upper from the
    best explicit representative given (any object with .emb())."""
    low = lower_bound(m, **kw)
    lows = low[0] if low is not None else F(0)
    ups = [w.emb() for w in witnesses]
    ups.append(m.emb())
    up = min(ups)
    return lows, up, lows == up
