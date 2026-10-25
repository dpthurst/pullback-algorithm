"""The exact homotopy minimiser by the CONVEX formulation in the universal
cover (plan item 170) -- the route that removes the type search entirely.

WHY.  The harmonic step H_[phi](l) minimises Dir(g) = sum_e l(g(e))^2/alpha1(e)
over the homotopy class.  In the "positions + chains" chart (fastmin.py) that
minimisation is a convex least-squares problem only *for a fixed combinatorial
type*, so the type has to be searched -- and the type graph has barriers
(moving a vertex from inside one target edge into a neighbouring one must pass
through their common vertex, where Dir is higher), which is how the search can
stop short of the harmonic map.  Lifted to the universal cover T of the target
the energy

    F(x) = sum_{e=(u,v)} d_T(x_u, tau_e x_v)^2 / alpha1(e)

is convex -- T is a tree, hence CAT(0), and the squared distance to a fixed
point is convex along geodesics -- so there is no barrier and no type to search:
the minimiser is unique up to the global deck translation, and the type (which
target edges each image crosses) is read off the solution.  This is the
principled fix for the bouquet failure of fastmin.py.

NO COVER IS BUILT.  A lifted point of T is a *reduced word* -- a path in the
target from a root -- plus an offset inside its last edge:

    (word, None, 0)      a lifted vertex (the endpoint of `word`)
    (word, dart, t)      0 < t < l(dart): inside the lift of `dart`

and a deck element is a reduced loop word at the root.  Everything is exact
`fractions.Fraction`; the words stay short because they are reduced.

THE BLOCK MINIMISATION IS EXACT.  At a vertex v, with the other positions
fixed, the objective is sum_i w_i d(x, z_i)^2 over x in T.  Its minimiser lies
in the Steiner tree of the z_i, which is covered by the atomic single-edge
segments of the pairwise geodesics; on each such segment every distance is
affine in the segment parameter, so the objective is a convex quadratic whose
exact rational minimiser is one computation (`minimise_one`).  Coordinate
descent over the vertices with this exact block solver converges to the
global minimum, and the directional tension gaps it returns are exactly the
balance condition of harmonic.tex cor:harmonic-triangle -- so a converged run
carries its own certificate.

Ported from the accepted swarm member m1 (`.ultramath/smvna7iev/m1/
homotopy.py`), with the inexact Newton polish dropped: the coordinate descent
with exact block minimisation is the whole harmonic step.  Cross-checked
against `pullback.minimise` (the relaxed-polytope QP) where both run.
"""

from fractions import Fraction as Q

F, B = 1, -1


def rev(d):
    return (d[0], -d[1])


def red(w):
    out = []
    for d in w:
        if out and out[-1] == rev(d):
            out.pop()
        else:
            out.append(d)
    return tuple(out)


class G:
    """A metric graph: edges {label: (v0, v1)}, dart (label, F) runs v0 -> v1.

    lengths: {label: positive Fraction}.  Vertex names are strings.
    """

    def __init__(self, edges, lengths, root=None):
        self.edges = {k: (str(a), str(b)) for k, (a, b) in edges.items()}
        self.len = {k: Q(v) for k, v in lengths.items()}
        for k in self.edges:
            if k not in self.len:
                raise KeyError(f"no length for edge {k!r}")
            if self.len[k] <= 0:
                raise ValueError(f"edge {k!r} must be positive")
        self.root = root

    def labels(self):
        return sorted(self.edges, key=str)

    def src(self, d):
        v0, v1 = self.edges[d[0]]
        return v0 if d[1] == F else v1

    def tgt(self, d):
        return self.src(rev(d))

    def ell(self, d):
        return self.len[d[0]]

    def vertices(self):
        vs = set()
        for v0, v1 in self.edges.values():
            vs.add(v0)
            vs.add(v1)
        return sorted(vs)

    def darts(self):
        return [(e, F) for e in self.labels()] + [(e, B) for e in self.labels()]

    def out(self, v):
        return [d for d in self.darts() if self.src(d) == v]

    def incident(self, v):
        return [e for e in self.labels() if v in self.edges[e]]

    def with_lengths(self, ell):
        return G(self.edges, ell, self.root)

    def __repr__(self):
        return f"G({self.edges}, {self.len}, root={self.root})"


    @classmethod
    def from_refined(cls, graph, root=None):
        """From a pullback.refined.Graph (same edge/vertex conventions)."""
        return cls({e: graph.edges[e] for e in graph.edges},
                   {e: graph.alpha[e] for e in graph.edges}, root=root)


def lp_vertex(w):
    return (red(w), None, Q(0))


def lp_inside(w, d, t, g):
    """Canonical lifted point at distance t along dart d from the end of w."""
    w = red(w)
    if t <= 0:
        return (w, None, Q(0))
    if t >= g.ell(d):
        return (red(w + (d,)), None, Q(0))
    if w and w[-1] == rev(d):
        return lp_inside(w[:-1], rev(d), g.ell(d) - t, g)
    return (w, d, t)


def lp_deck(deck, p, g):
    """Apply the deck element `deck` (a reduced loop word at the root)."""
    w, d, t = p
    if d is None:
        return (red(deck + w), None, Q(0))
    return lp_inside(deck + w, d, t, g)


def lp_letters(p, g):
    w, d, t = p
    out = [(x, g.ell(x)) for x in w]
    if d is not None:
        out.append((d, t))
    return out


def lp_vertex_of(p, g):
    """The vertex of G under a lifted *vertex* point."""
    w, d, t = p
    if d is not None:
        raise ValueError("not a vertex point")
    if w:
        return g.tgt(w[-1])
    return g.root


def _walk_letters(p, g):
    """The signed traversals from the root to p: (dart, a, b), a <= b."""
    w, d, t = p
    out = [(x, Q(0), g.ell(x)) for x in w]
    if d is not None:
        out.append((d, Q(0), t))
    return out


def _flip(tr, g):
    d, a, b = tr
    return (rev(d), g.ell(d) - b, g.ell(d) - a)


def lp_path(p, r, g):
    """(distance, up, down): the tree geodesic from p to r.

    `up` / `down` are signed traversals (dart, a, b) with a <= b, from p up to
    the branch point and from there down to r.
    """
    a, b = _walk_letters(p, g), _walk_letters(r, g)
    i = 0
    while True:
        if i >= len(a) and i >= len(b):
            return (Q(0), [], [])
        if i < len(a) and i < len(b) and a[i][0] == b[i][0]:
            if i == len(a) - 1 or i == len(b) - 1:
                d = a[i][0]
                la, lb = a[i][2], b[i][2]
                ra = [a[j] for j in range(i + 1, len(a))]
                rb = [b[j] for j in range(i + 1, len(b))]
                up = [_flip(x, g) for x in reversed(ra)]
                down = list(rb)
                if not ra and not rb:
                    if la == lb:
                        return (Q(0), [], [])
                    if la > lb:
                        # p is further along d than r: the way up from p walks
                        # d BACKWARD, so the traversal is the flip of (d, lb, la)
                        return (la - lb, [_flip((d, lb, la), g)], [])
                    return (lb - la, [], [(d, la, lb)])
                if not ra:
                    back = g.ell(d) - la
                    if back > 0:
                        up = [(d, la, g.ell(d))]
                    return (back + sum((x[2] - x[1] for x in rb), Q(0)), up, down)
                if not rb:
                    back = g.ell(d) - lb
                    if back > 0:
                        # from the end of d the way down to r walks d BACKWARD
                        down = [_flip((d, lb, g.ell(d)), g)]
                    return (back + sum((x[2] - x[1] for x in ra), Q(0)), up, down)
            i += 1
            continue
        up = [_flip(a[j], g) for j in range(len(a) - 1, i - 1, -1)]
        down = [b[j] for j in range(i, len(b))]
        dist = sum((x[2] - x[1] for x in a[i:]), Q(0)) + \
            sum((x[2] - x[1] for x in b[i:]), Q(0))
        return (dist, up, down)


def lp_dist(p, r, g):
    return lp_path(p, r, g)[0]


def lp_lca(p, r, g):
    """The branch point (tree meet) of two lifted points."""
    a, b = lp_letters(p, g), lp_letters(r, g)
    i = 0
    while i < len(a) and i < len(b) and a[i][0] == b[i][0]:
        if i == len(a) - 1 or i == len(b) - 1:
            ra = (i == len(a) - 1)
            rb = (i == len(b) - 1)
            if ra and rb:
                return p if a[i][1] <= b[i][1] else r
            d = a[i][0]
            if ra:
                if a[i][1] >= g.ell(d):
                    return p
                return (tuple(x for x, _ in a[:i]) + (d,), None, Q(0))
            if b[i][1] >= g.ell(d):
                return r
            return (tuple(x for x, _ in b[:i]) + (d,), None, Q(0))
        i += 1
    return (tuple(x for x, _ in a[:i]), None, Q(0))


def _first_motion(up, down):
    tr = up[0] if up else (down[0] if down else None)
    if tr is None:
        return None
    d, a, b = tr
    return d if b > a else rev(d)


def lp_direction(p, r, g):
    """The dart walked first from p towards r (None if p == r)."""
    dist, up, down = lp_path(p, r, g)
    if dist == 0:
        return None
    return _first_motion(up, down)


def lp_step(p, d, eps, g):
    """Move from p by eps > 0 in the lifted direction d."""
    w, dd, t = p
    if dd is None:
        return lp_inside(w, d, eps, g)
    if d == dd:
        return lp_inside(w, dd, t + eps, g)
    return lp_inside(w, dd, t - eps, g)


def lp_directions(p, g):
    w, d, t = p
    if d is None:
        return g.out(lp_vertex_of(p, g))
    return [d, rev(d)]


def lp_first_step(p, d, g):
    """How far one can walk from p in direction d before the first vertex."""
    w, dd, t = p
    if dd is None:
        return g.ell(d)
    return (g.ell(dd) - t) if d == dd else t


# --------------------------------------------------------- the harmonic step
def _fval(terms, x, g):
    return sum((w * lp_dist(x, y, g) ** 2 for y, w in terms), Q(0))


def _eps_star(terms, a, d, g):
    """The exact one-direction optimum from a, and the tension gap."""
    ahead, behind, tot = Q(0), Q(0), Q(0)
    for y, w in terms:
        dist, up, down = lp_path(a, y, g)
        dd = _first_motion(up, down)
        tot += w
        if dd == d:
            ahead += w * dist
        else:
            behind += w * dist
    if tot == 0:
        return Q(0), Q(0)
    return (ahead - behind) / tot, ahead - behind


def _atomic_segments(p, q, g):
    """Split the tree geodesic p -> q at every vertex into single-edge pieces."""
    dist, up, down = lp_path(p, q, g)
    segs = []
    cur = p
    for tr in up + down:
        d = _first_motion([tr], [])
        rem = tr[2] - tr[1]
        while rem > 0:
            step = lp_first_step(cur, d, g)
            if step <= 0:
                raise RuntimeError("atomic walk stuck")
            step = min(step, rem)
            nxt = lp_step(cur, d, step, g)
            segs.append((cur, nxt))
            cur = nxt
            rem -= step
    return segs


def minimise_one(terms, g, rounds=1):
    """Exact minimiser of  sum_i w_i * dist(x, y_i)^2  over x in the tree.

    The optimum lies in the Steiner tree of the y_i.  That tree is covered by
    the atomic single-edge segments of the pairwise geodesics; on each such
    segment every dist(x, y_i) is affine in the segment parameter, so the
    objective is a convex quadratic and its exact minimiser is one rational
    computation.  Returns (x, gaps); gaps are the directional tension gaps at
    x, all <= 0 at the optimum -- that *is* the balance condition of
    cor:harmonic-triangle.
    """
    pts = [y for y, _ in terms]
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            lca = lp_lca(terms[i][0], terms[j][0], g)
            if lca not in pts:
                pts.append(lca)
    segs = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            segs.extend(_atomic_segments(pts[i], pts[j], g))
    best, bestv = None, None
    for x in pts:
        v = _fval(terms, x, g)
        if bestv is None or v < bestv:
            best, bestv = x, v
    for (a, b) in segs:
        L = lp_dist(a, b, g)
        if L <= 0:
            continue
        d = lp_direction(a, b, g)
        num, tot = Q(0), Q(0)
        for y, w in terms:
            dy = lp_dist(a, y, g)
            if dy > 0 and lp_direction(a, y, g) == d:
                c = dy
            else:
                c = -dy
            num += w * c
            tot += w
        if tot == 0:
            continue
        t = num / tot
        if t < 0:
            t = Q(0)
        if t > L:
            t = L
        x = lp_step(a, d, t, g)
        v = _fval(terms, x, g)
        if v < bestv:
            best, bestv = x, v
    gaps = []
    for d in lp_directions(best, g):
        _eps, gap = _eps_star(terms, best, d, g)
        gaps.append((d, gap))
    return best, gaps


# ------------------------------------------------------- the homotopy class
def _canon(tr, ell):
    """A signed traversal (dart, a, b) in the canonical frame of its edge:
    (label, lo, hi) with lo <= hi, parameters along the dart (label, F)."""
    d, a, b = tr
    lo, hi = (a, b) if a <= b else (b, a)
    if d[1] == F:
        return (d[0], lo, hi)
    L = ell[d[0]]
    return (d[0], L - hi, L - lo)


def deck_inv(g):
    return tuple(rev(x) for x in reversed(g))


def image_dart(cls, dart):
    ch = cls.chains[dart[0]]
    if dart[1] == F:
        return tuple(ch)
    return tuple(rev(x) for x in reversed(ch))

class Cls:
    """A homotopy class of marked graph maps src -> tgt, given by chains.

    `tgt.len` is the target's *elastic* structure alpha2 (fixed); the target
    metric ell is supplied separately, so the same class can be evaluated at
    every length vector of alg:iter.
    """

    def __init__(self, src, chains, tgt, marked=(), root=None):
        self.src = src
        self.tgt = tgt
        self.alpha2 = dict(tgt.len)
        self.chains = {e: red(chains.get(e, ())) for e in src.labels()}
        self.marked = set(marked)
        self._setup(root)

    # -- class data -----------------------------------------------------
    def _setup(self, root):
        img = {}
        for e in self.src.labels():
            ch = self.chains[e]
            if not ch:
                continue
            v0, v1 = self.src.edges[e]
            for v, w, which in ((v0, self.tgt.src(ch[0]), "tail"),
                                (v1, self.tgt.tgt(ch[-1]), "head")):
                if v in img and img[v] != w:
                    raise ValueError(
                        f"inconsistent vertex images at {v}: {img[v]} vs {w} ({which} of {e})")
                img[v] = w
        self.img = img
        verts = self.src.vertices()
        if root is not None and root in img:
            base = root
        elif self.marked & set(img):
            base = sorted(self.marked & set(img), key=str)[0]
        elif img:
            base = sorted(img, key=str)[0]
        elif self.marked:
            base = sorted(self.marked, key=str)[0]
        else:
            base = verts[0]
        if base not in img:
            raise ValueError(f"base vertex {base} has no determined image")
        self.base = base
        self.b2 = img[base]
        self.tgt.root = self.b2
        # spanning tree of the source, rooted at base
        tree = {base: ()}
        order = [base]
        qi = 0
        while qi < len(order):
            u = order[qi]
            qi += 1
            for d in self.src.out(u):
                v = self.src.tgt(d)
                if v in tree:
                    continue
                tree[v] = tree[u] + (d,)
                order.append(v)
        if len(tree) != len(verts):
            raise ValueError("source graph is not connected")
        self.tree = tree
        # lifted reference words and deck elements
        W = {}
        for v in tree:
            w = ()
            for d in tree[v]:
                w = w + image_dart(self, d)
            W[v] = red(w)
        self.W = W
        self.tau = {}
        for d in self.src.darts():
            u, v = self.src.src(d), self.src.tgt(d)
            self.tau[d] = red(W[u] + image_dart(self, d) + deck_inv(W[v]))
        for d in self.src.darts():   # consistency: tau_rev = tau^-1
            if red(self.tau[d] + self.tau[rev(d)]) != ():
                raise ValueError(f"deck element not inverse-consistent on {d}")
        self.X0 = {v: lp_vertex(W[v]) for v in verts}

    # -- energies -------------------------------------------------------
    def _term(self, d, X, ell):
        """(landing lifted point, weight) of source dart d at positions X."""
        g = self.tgt.with_lengths(ell)
        u, v = self.src.src(d), self.src.tgt(d)
        land = lp_deck(self.tau[d], X[v], g)
        return X[u], land, Q(1) / self.src.ell(d)

    def energy(self, X, ell):
        g = self.tgt.with_lengths(ell)
        tot = Q(0)
        for e in self.src.labels():
            d = (e, F)
            u, v = self.src.src(d), self.src.tgt(d)
            tot += lp_dist(X[u], lp_deck(self.tau[d], X[v], g), g) ** 2 / self.src.ell(d)
        return tot

    def pieces(self, X, ell):
        """{source edge: [(target edge, a, b), ...]}: the geodesic images, as
        signed traversals with parameters along the target dart (a <= b)."""
        g = self.tgt.with_lengths(ell)
        out = {}
        for e in self.src.labels():
            d = (e, F)
            u, v = self.src.src(d), self.src.tgt(d)
            dist, up, down = lp_path(X[u], lp_deck(self.tau[d], X[v], g), g)
            out[e] = [_canon(t, ell) for t in up + down]
        return out

    def dir_energy(self, X, ell):
        g = self.tgt.with_lengths(ell)
        return self.energy(X, ell)

    def multiplicities(self, X, ell, scale=None):
        """(v, spread): v(f) is the pointwise multiplicity  sum_x w(x)  on the
        target edge f (eq:weights-sum, filling.tex:991), measured with the
        metric `scale` on the target (default: ell); `spread` records, per
        edge, the set of constant values found on its open sub-intervals (a
        taut/harmonic map has exactly one, so the map is a Wgt(Γ2))."""
        pcs = self.pieces(X, ell)
        sc = dict(ell) if scale is None else dict(scale)
        per = {f: [] for f in self.tgt.labels()}
        for e in self.src.labels():
            ps = pcs[e]
            tot = sum((b - a for (_lab, a, b) in ps), Q(0))
            w = tot / self.src.len[e]     # |g'| on the source edge, ell metric
            for (lab, a, b) in ps:
                k = sc[lab] / ell[lab]
                per[lab].append((w * k, a * k, b * k))
        v, spread = {}, {}
        for f in self.tgt.labels():
            items = per[f]
            cuts = sorted({Q(0), sc[f]} | {a for _, a, _ in items} | {b for _, _, b in items})
            vals = set()
            for i in range(len(cuts) - 1):
                lo, hi = cuts[i], cuts[i + 1]
                if hi <= lo:
                    continue
                mid = (lo + hi) / 2
                vals.add(sum((w for w, a, b in items if a <= mid <= b), Q(0)))
            spread[f] = vals
            v[f] = max(vals) if vals else Q(0)
        return v, spread

    def fill(self, X, ell):
        """v = N_[phi](w) on the target edges (eq:weights-sum, filling.tex:991)."""
        return self.multiplicities(X, ell)[0]

    def emb(self, X, ell):
        """Emb of the representative: the essential supremum over the target
        of the pointwise multiplicity (intro.tex eq:emb), with the target
        measured in its elastic metric alpha2.  Target vertices have measure
        zero, so the ess sup is the max of the per-edge multiplicities."""
        v = self.multiplicities(X, ell, scale=self.alpha2)[0]
        return max(v.values()) if v else Q(0)

    # -- the harmonic step ---------------------------------------------
    def harmonic(self, ell, X=None, rounds=40, tol=Q(1, 10 ** 9), trace=False):
        """A Dir-minimal representative of the class at target lengths ell.

        Coordinate descent in the universal cover with the exact block
        minimiser (`minimise_one`) at every vertex; returns (X, gaps) where
        gaps collects (vertex, direction, tension gap) at the end, all <= 0
        when the map is balanced, i.e. harmonic (cor:harmonic-triangle).

        The descent converges geometrically, so it is stopped when a sweep
        improves Dir by less than `tol` and the offsets are then snapped to
        rationals with small denominators -- otherwise the exact arithmetic
        accumulates denominators of size 2^48 per sweep and the iteration
        crawls.  Snapping only moves a point inside its edge, so the
        combinatorial type (and hence the answer, to ~1e-9) is unchanged.
        """
        g = self.tgt.with_lengths(ell)
        X = dict(self.X0 if X is None else X)
        order = [v for v in self.src.vertices() if v not in self.marked]
        prev = None
        for _ in range(rounds):
            moved = False
            for v in order:
                terms = []
                for d in self.src.darts():
                    if self.src.src(d) == v:
                        w = Q(1) / self.src.ell(d)
                        land = lp_deck(self.tau[d], X[self.src.tgt(d)], g)
                        terms.append((land, w))
                    elif self.src.tgt(d) == v:
                        w = Q(1) / self.src.ell(d)
                        land = lp_deck(deck_inv(self.tau[d]), X[self.src.src(d)], g)
                        terms.append((land, w))
                if not terms:
                    continue
                x, gaps = minimise_one(terms, g)
                if x != X[v]:
                    X[v] = x
                    moved = True
            e = self.energy(X, ell)
            if not moved or (prev is not None and prev - e <= tol * max(e, Q(1))):
                break
            prev = e
        gaps_out = []
        for v in order:
            terms = []
            for d in self.src.darts():
                if self.src.src(d) == v:
                    terms.append((lp_deck(self.tau[d], X[self.src.tgt(d)], g),
                                  Q(1) / self.src.ell(d)))
                elif self.src.tgt(d) == v:
                    terms.append((lp_deck(deck_inv(self.tau[d]), X[self.src.src(d)], g),
                                  Q(1) / self.src.ell(d)))
            if terms:
                _x, gaps = minimise_one(terms, g)
                for dd, gap in gaps:
                    gaps_out.append((v, dd, gap))
        # the balance certificate above is computed on the descent's own
        # output; only then are the offsets snapped (snapping perturbs
        # the map by ~1e-9 and would turn the gaps slightly positive)
        X = {v: _snap_point(x) for v, x in X.items()}
        return X, gaps_out

    # -- algorithm 6.22 -------------------------------------------------
    def dr(self, ell, X=None, tol=Q(1, 10 ** 9)):
        """(Dir, Dir_id, DR, v, X) at target lengths ell.

        `tol` is the descent's stopping tolerance (see `harmonic`); alg:iter
        passes a looseness that decays towards it in the early steps."""
        X, gaps = self.harmonic(ell, X, tol=tol)
        Dir = self.energy(X, ell)
        Dir_id = sum((ell[f] ** 2 / self.alpha2[f] for f in self.tgt.labels()), Q(0))
        v = self.fill(X, ell)
        return Dir, Dir_id, Dir / Dir_id, v, X

    def emb_of_input(self, ell):
        """Emb of the input representative itself (no minimisation)."""
        return self.emb(self.X0, ell)

    def step(self, ell, X=None):
        """One step of alg:iter: returns (ell_next, DR, v, X)."""
        Dir, Dir_id, ratio, v, X = self.dr(ell, X)
        nxt = {f: self.alpha2[f] * v[f] for f in self.tgt.labels()}
        return nxt, ratio, v, X


# --------------------------------------------- the fixed point of alg:iter
def signature(cls, ell, X=None):
    """A canonical, hashable description of the combinatorial type of the
    harmonic representative at ell (which target edges each source edge
    crosses, in order).  Constant on the cone of linearity of Iter."""
    if X is None:
        X, _gaps = cls.harmonic(ell)
    pcs = cls.pieces(X, ell)
    return tuple((e, tuple(lab for (lab, _a, _b) in pcs[e])) for e in cls.src.labels())



def _snap_point(x, max_den=10 ** 7):
    """Snap a lifted point's offset to a small-denominator rational (the word
    is already short and exact; only the offset accumulates denominators).
    Floats pass through unchanged."""
    w, d, t = x
    if d is None:
        return (w, None, 0 if isinstance(t, float) else Q(0))
    if isinstance(t, float):
        return (w, d, t)
    return (w, d, Q(t).limit_denominator(max_den))


# ------------------------------------------------------------- alg:iter
def alg_iter(cls, ell0=None, steps=40, tol=None, htol_loose=None,
             verbose=False):
    """[Elast] alg:iter (filling.tex:522) on the class `cls`, exactly.

    Returns (lambdas, ell, cls, X, gaps): lambdas[i] is the multiplicative
    factor of step i (the last estimates SF[phi]); gaps is the balance
    certificate of the final harmonic map (empty = harmonic).

    The iteration stops when the factor repeats exactly (the exact fixed
    point), or -- when `tol` is given -- when it changes by at most a
    relative `tol` (a bracketed value; the geometric convergence reaches
    1e-12 in roughly a quarter of the default 40 steps, so a large `tol`
    like 1e-12 is a substantial saving with the fixed point still certified
    by the gaps).

    `htol_loose` gives the early steps' harmonic descents a loose stopping
    tolerance, decaying geometrically (a factor 10 per step) to the default
    Q(1,10**9); the early steps only need to get the metric roughly right,
    and the final steps -- which produce the reported factor -- keep the
    tight descent.  The final certificate call is always tight.
    """
    labs = cls.tgt.labels()
    ell = {f: Q(1) for f in labs} if ell0 is None else dict(ell0)
    X = None
    out = []
    base_tol = Q(1, 10 ** 9)
    for i in range(steps):
        ht = base_tol if htol_loose is None else max(
            base_tol, htol_loose * Q(1, 10) ** (-i))
        Dir, Dir_id, ratio, v, X = cls.dr(ell, X, tol=ht)
        out.append(ratio)
        if verbose:
            print(f"  iter {i}: DR = {ratio} = {float(ratio):.9f}")
        nxt = {f: cls.alpha2[f] * v[f] for f in labs}
        s = max(nxt.values())
        if s == 0:
            break
        ell = {f: nxt[f] / s for f in labs}
        if len(out) >= 2 and out[-1] == out[-2]:
            break
        if (tol is not None and len(out) >= 2
                and abs(out[-1] - out[-2]) <= tol * max(abs(out[-1]), Q(1))):
            break
    _X, gaps = cls.harmonic(ell, X)
    return out, ell, cls, X, gaps


def session_class(path, marked_base=()):
    """Load a session as a Cls: source = the cover with alpha1 the lift of
    alpha0, target = the base graph with alpha2 = alpha0.  alpha0 is the
    session's "elastics" -- a list over the base's edges in sorted order
    (the theta session's [4,1,4] is (p,q,r), the paper's assignment); unit
    when absent."""
    import json
    from .harmonic_io import load_session
    from .refined import graph_from_ribbon
    with open(path) as f:
        sess = json.load(f)
    marked_base = set(marked_base) | set(sess.get("marked", []))
    base, cover, phi = load_session(path)
    G2 = G.from_refined(graph_from_ribbon(base))
    G1 = G.from_refined(graph_from_ribbon(cover))
    elast = sess.get("elastics")
    if elast is not None:
        elast = dict(zip(sorted(G2.edges, key=str), elast))
        for e in G1.edges:
            G1.len[e] = Q(elast[e[0]])
        for e in G2.edges:
            G2.len[e] = Q(elast[e])
    marked = {u for u in cover.vertices() if u[0] in marked_base}
    chains = {e: tuple(phi.chains.get(e, ())) for e in G1.edges}
    return Cls(G1, chains, G2, marked=marked)
