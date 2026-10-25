"""The fast (floating-point) homotopy minimiser: [Elast] alg:iter with the
harmonic step computed by the *position* linear algebra, following the
author's Harmonic code (`.scratch/Harmonic/Harmonic.lhs`, `Filling.lhs`).

WHY THIS EXISTS.  `pullback.minimise` solves the harmonic step exactly, as a
QP over the relaxed polytope cut out by *all* simple marked curves of the
source.  That is exact and certifiable, but the number of curves grows
exponentially with |E(G_n)|: on the theta example it is 3 curves at n = 1,
23 at n = 2, 97 at n = 3 and 5378 at n = 4 (a minute), and n >= 5 is out of
reach.  The author's Haskell code reaches n = 8 because it never enumerates
curves: for a fixed combinatorial type the harmonic map is an ordinary
least-squares problem in the *positions* of the source vertices, of size
|V(G_n)|.

Both implementations compute the same objects (H_[phi](l), then
v = N(w), l' = alpha2 v), so they cross-check against each other where both
run: see `tests/test_fastmin.py` and `scripts/figure2_fast.py`.

THE POSITION MODEL (the author's conventions, Harmonic.lhs)
----------------------------------------------------------
A map is (chains, vloc):

  chains[e] = the image path of source edge e, a tuple of target darts
              (label, +-1), *including* the endpoint darts when an endpoint
              sits in the interior of a target edge;
  vloc[v]   = ("V", target vertex) or ("E", target edge, p) with p the
              *centred* position: the point is at l(E)/2 + p from the tail,
              p in [-l(E)/2, l(E)/2].

For a source edge e = (u, v) the image length is

  l(img_e) = sum_{d in chain(e)} l(d)
             - sum over endpoints of e in an edge E of l(E)/2
             + sum over endpoints of e in an edge of coeff * p

with coeff = -1 if the chain leaves that endpoint along the dart E forward,
+1 if backward (this is `lengthMap` + `vertexMap` of Harmonic.lhs).  In
matrix form l(img) = L l2 + V p, and minimising

  Dir = sum_e l(img_e)^2 / alpha1(e)

over p is the weighted least-squares problem  min || W^{1/2}(L l2 + V p) ||,
W = diag(1/alpha1), which `numpy.linalg.lstsq` solves, returning the
minimal-norm solution when the system is degenerate (the author's `findPos`).

The combinatorial type is then repaired as in his `pushPull`: a position that
leaves its edge is *pushed* onto the vertex it crossed, and a vertex at a
target vertex whose tension triangle inequality fails is *pulled* into the
offending edge; each move edits the incident chains by one dart.

The normal matrix has at most four nonzeros per row (an edge touches two
vertices), so it is assembled as a sparse CSR matrix and solved by a sparse
LU, whose factorization is cached while the combinatorial type is unchanged.
On the theta example that takes n = 8 from 50 s to 36 s.

KNOWN GAP.  The repair is a local search over types and can stop in a type
that is locally optimal but not harmonic: there is no move that takes a
vertex *already inside* an edge into a *different* edge (it must first be
pushed out to a vertex, and the push only fires when the position leaves the
edge).  On the bouquet example (whose chains are constant on the lifts of b)
that happens: at n = 2 the map found has Dir = 1.625 where the exact
harmonic map has 1.5, and the iteration converges to 0.820 instead of 3/4.
Theta -- the [Rubber] Figure 2 example -- is unaffected, and is checked to
~1e-7 at n = 1..8 against the paper.

PRECISION.  Everything here is float64.  This is the scaling path, not the
certification path: `pullback.minimise` provides exact rational brackets on
the small cases, and the two agree to ~1e-9 on the theta and bouquet examples
where both run.  A value from here is computed, not proved: label it so.
"""

from __future__ import annotations

import numpy as np


def _inv(d):
    return (d[0], -d[1])


def _rev(word):
    return tuple((d[0], -d[1]) for d in reversed(tuple(word)))


class FastMap:
    """A map with vertex positions: chains + vloc (the author's PosMap)."""

    def __init__(self, src, tgt, chains, vloc=None):
        self.src, self.tgt = src, tgt
        self.chains = {e: tuple(chains[e]) for e in src.edges}
        self.vloc = dict(vloc) if vloc is not None else _start_positions(src, tgt, self.chains)
        self._lu = None
        self._lu_key = None

    # -- chains oriented away from a vertex ------------------------------
    def chain_from(self, v, e):
        a, b = self.src.edges[e]
        if a == v:
            return self.chains[e]
        if b == v:
            return _rev(self.chains[e])
        raise ValueError(f"vertex {v!r} is not an endpoint of edge {e!r}")

    def _set_chain_from(self, v, e, c):
        a, b = self.src.edges[e]
        self.chains[e] = tuple(c) if a == v else _rev(tuple(c))

    def edges_at(self, v):
        return [e for e in sorted(self.src.edges, key=str) if v in self.src.edges[e]]

    def _type_key(self, free):
        """A canonical signature of the combinatorial type (so a cached
        factorization can be reused until a push/pull changes it)."""
        return (tuple(free),
                tuple(sorted((str(e), self.chains[e]) for e in self.chains)),
                tuple(sorted((str(v), self.vloc[v][0],
                              self.vloc[v][1] if len(self.vloc[v]) > 1 else None)
                             for v in self.vloc)))

    def free_vertices(self):
        return [v for v in sorted(self.vloc, key=str) if self.vloc[v][0] == "E"]

    # -- the linear algebra ----------------------------------------------
    def matrices(self, ell2, free):
        """(e1, e2, L, V): l(img) = L l2 + V p for the current type."""
        e1 = sorted(self.chains, key=str)
        e2 = sorted(self.tgt.edges, key=str)
        i2 = {e: k for k, e in enumerate(e2)}
        i_free = {v: k for k, v in enumerate(free)}
        L = np.zeros((len(e1), len(e2)))
        V = np.zeros((len(e1), len(free)))
        for i, e in enumerate(e1):
            ch = self.chains[e]
            for d in ch:
                L[i, i2[d[0]]] += 1.0
            u, v = self.src.edges[e]
            for end in (u, v):
                loc = self.vloc[end]
                if loc[0] != "E":
                    continue
                E = loc[1]
                c = self.chain_from(end, e)
                if not c or c[0][0] != E:
                    raise ValueError(f"chain of {e!r} does not meet the "
                                     f"position of {end!r} in {E!r}")
                L[i, i2[E]] -= 0.5
                V[i, i_free[end]] += -1.0 if c[0][1] == 1 else +1.0
        return e1, e2, L, V

    def image_lengths(self, ell2):
        """{source edge: l(img)} from the current positions."""
        free = self.free_vertices()
        e1, e2, L, V = self.matrices(ell2, free)
        l2 = np.array([ell2[e] for e in e2])
        p = np.array([self.vloc[v][2] for v in free]) if free else np.zeros(0)
        vals = L @ l2 + V @ p
        # A piece whose image lies inside one target edge has length
        # |p_v - p_u|: the sign only records which way round the two
        # endpoints sit, and the Dirichlet energy is blind to it (it squares),
        # but the pushforward N(w) and the reported lengths are not.
        return {e: abs(float(x)) for e, x in zip(e1, vals)}

    def dirichlet(self, alpha1, ell2):
        w = self.image_lengths(ell2)
        return float(sum(w[e] * w[e] / alpha1[e] for e in w))

    # -- one least-squares solve for the positions -----------------------
    def find_positions(self, alpha1, ell2):
        free = self.free_vertices()
        if not free:
            return
        e1, e2, L, V = self.matrices(ell2, free)
        l2 = np.array([ell2[e] for e in e2])
        w = np.array([1.0 / alpha1[e] for e in e1])
        # normal equations (A^T W A) p = -A^T W (L l2), with a small ridge for
        # the degenerate directions (a whole component may be free to slide).
        # This is the author's findPos; the ridge replaces his minimal-norm
        # choice of solution and is 10x faster than an SVD least-squares fit.
        A = V
        rhs = -((A * w[:, None]).T @ (L @ l2))
        # A has at most two nonzeros per row, so A^T W A is sparse (<= 4
        # nonzeros per row); solve it by a sparse LU, cached while the
        # combinatorial type is unchanged, falling back to a dense solve.
        key = self._type_key(free)
        if self._lu is None or self._lu_key != key:
            self._lu = None
            self._lu_key = key
            try:
                import scipy.sparse as sp
                import scipy.sparse.linalg as spla
                rows, cols, vals = [], [], []
                for i in range(A.shape[0]):
                    nz = np.nonzero(A[i])[0]
                    for j1 in nz:
                        for j2 in nz:
                            rows.append(j1)
                            cols.append(j2)
                            vals.append(A[i, j1] * A[i, j2] * w[i])
                M = sp.csr_matrix((vals, (rows, cols)), shape=(len(free), len(free)))
                M = M + 1e-12 * sp.identity(len(free), format="csr")
                self._lu = spla.splu(M.tocsc())
            except Exception:
                self._lu = None
        if self._lu is not None:
            p = self._lu.solve(rhs)
        else:
            M = (A * w[:, None]).T @ A
            try:
                p = np.linalg.solve(M + 1e-12 * np.eye(len(free)), rhs)
            except np.linalg.LinAlgError:
                p, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None],
                                        -(L @ l2) * np.sqrt(w), rcond=None)
        for j, v in enumerate(free):
            self.vloc[v] = ("E", self.vloc[v][1], float(p[j]))

    # -- the type repairs (the author's pushPull) ------------------------
    def push(self, ell2, eps=1e-9):
        """Move ONE vertex whose position left its edge onto the vertex it
        crossed.  One move per call: a batch would move both ends of an edge
        at once, and the incremental chain edits are only correct when a
        single vertex moves."""
        moved = False
        for v in list(self.free_vertices()):
            E = self.vloc[v][1]
            p = self.vloc[v][2]
            half = 0.5 * ell2[E]
            if p > half + eps:
                self._push(v, E, +1)
                moved = True
            elif p < -half - eps:
                self._push(v, E, -1)
                moved = True
        return moved

    def _push(self, v, E, direction):
        """Move v from inside E onto its head (direction +1) or tail (-1)."""
        a, b = self.tgt.edges[E]
        newv = b if direction == 1 else a
        ahead = (E, 1 if direction == 1 else -1)   # the dart v was travelling along
        for e in self.edges_at(v):
            c = list(self.chain_from(v, e))
            w = self.src.edges[e][0] if self.src.edges[e][1] == v else self.src.edges[e][1]
            if not c:
                # the map was constant on e; now v is at newv.  If the other
                # endpoint is there too the edge stays constant, otherwise the
                # image is the piece of E from newv back to w's position.
                c = [] if self.vloc[w] == ("V", newv) else [_inv(ahead)]
            elif c[0] == ahead:
                c = c[1:]                # the image used exactly that dart
                if not c and self.vloc[w] != ("V", newv):
                    # ... and the other endpoint is still in E: the image is
                    # the piece of E from newv back to that point
                    c = [_inv(ahead)]
            elif c[0] == _inv(ahead):
                pass                     # the image runs back into E
            else:
                raise ValueError("chain does not leave along the edge of its end")
            self._set_chain_from(v, e, c)
        self.vloc[v] = ("V", newv)

    def tensions(self, alpha1, ell2):
        """{vertex: {oriented dart: tension}}: the tension of a direction is
        the sum of w = l(img)/alpha1 over the incident edges whose image
        leaves the vertex along it."""
        w = self.image_lengths(ell2)
        out = {}
        for e in self.chains:
            u, v = self.src.edges[e]
            we = w[e] / alpha1[e]
            for end in (u, v):
                c = self.chain_from(end, e)
                if not c:
                    continue
                out.setdefault(end, {})
                out[end][c[0]] = out[end].get(c[0], 0.0) + we
        return out

    def pull(self, alpha1, ell2, eps=1e-9):
        """Pull one vertex at a target vertex into an edge along which its
        tension dominates (the author's pullVertices)."""
        T = self.tensions(alpha1, ell2)
        for v in sorted(self.vloc, key=str):
            if self.vloc[v][0] != "V":
                continue
            dirs = T.get(v, {})
            if not dirs:
                continue
            total = sum(dirs.values())
            for d, t in sorted(dirs.items(), key=lambda kv: (-kv[1], str(kv[0]))):
                if 2 * t > total + eps * max(total, 1.0):
                    self._pull(v, d)
                    return True
        return False

    def _pull(self, v, d):
        """v (at the tail of the oriented dart d) moves to the centre of d's
        edge; the incident chains gain the back dart (the author's pullVertex)."""
        for e in self.edges_at(v):
            c = list(self.chain_from(v, e))
            if c and c[0] in (d, _inv(d)):
                pass
            else:
                c = [_inv(d)] + c
            self._set_chain_from(v, e, c)
        self.vloc[v] = ("E", d[0], 0.0)

    def validate(self):
        """Raise unless every chain is a genuine path in the target graph that
        starts and ends at its endpoints' positions (a continuous PL map)."""
        for e in sorted(self.chains, key=str):
            ch = self.chains[e]
            for d, nxt in zip(ch, ch[1:]):
                if _head_of(self.tgt, d) != _tail_of(self.tgt, nxt):
                    raise ValueError(
                        f"chain of {e!r} is not a path at {d!r} -> {nxt!r}")
            u, v = self.src.edges[e]
            if not ch:
                if not _same_point(self.tgt, self.vloc[u], self.vloc[v]):
                    raise ValueError(f"chain of {e!r} is empty but its ends "
                                     f"{self.vloc[u]} and {self.vloc[v]} differ")
                continue
            for end, d, which in ((u, ch[0], "tail"), (v, ch[-1], "head")):
                loc = self.vloc[end]
                if loc[0] == "V":
                    p = (_tail_of(self.tgt, d) if which == "tail"
                         else _head_of(self.tgt, d))
                    if p != loc[1]:
                        raise ValueError(
                            f"chain of {e!r} starts/ends at {p!r} but {end!r} "
                            f"is at vertex {loc[1]!r}")
                else:
                    if d[0] != loc[1]:
                        raise ValueError(
                            f"chain of {e!r} does not meet {end!r} in edge {loc[1]!r}")

    # -- the harmonic step ------------------------------------------------
    def harmonic(self, alpha1, ell2, max_steps=2000, verbose=False):
        """Solve-and-repair until the type is stable: the harmonic map."""
        for step in range(max_steps):
            self.find_positions(alpha1, ell2)
            if self.push(ell2):
                continue
            if self.pull(alpha1, ell2):
                continue
            return step
        raise RuntimeError("harmonic type search did not stabilise")

    # -- the pushforward and alg:iter -------------------------------------
    def pushforward(self, alpha1, ell2):
        """v = L^T w with w = l(img)/alpha1 (alg:iter step 3)."""
        free = self.free_vertices()
        e1, e2, L, V = self.matrices(ell2, free)
        img = self.image_lengths(ell2)
        w = np.array([img[e] / alpha1[e] for e in e1])
        return {e: float(x) for e, x in zip(e2, L.T @ w)}


def _start_positions(src, tgt, chains):
    """Every source vertex at a target vertex: the chain map's vertex images."""
    out = {}
    for e in sorted(src.edges, key=str):
        ch = chains[e]
        if not ch:
            continue
        u, v = src.edges[e]
        out[u] = _tail_of(tgt, ch[0])
        out[v] = _head_of(tgt, ch[-1])
    changed = True
    while changed:
        changed = False
        for e in sorted(src.edges, key=str):
            if chains[e]:
                continue
            u, v = src.edges[e]
            if u in out and v not in out:
                out[v] = out[u]
                changed = True
            elif v in out and u not in out:
                out[u] = out[v]
                changed = True
    missing = [v for v in _vertices(src) if v not in out]
    if missing:
        raise ValueError(f"no image for vertices {missing}")
    return {v: ("V", tv) for v, tv in out.items()}


def _same_point(tgt, loc1, loc2, eps=1e-12):
    """Do two vertex locations denote the same point of the target?"""
    if loc1[0] == "V" and loc2[0] == "V":
        return loc1[1] == loc2[1]
    if loc1[0] == "E" and loc2[0] == "E":
        return loc1[1] == loc2[1] and abs(loc1[2] - loc2[2]) < eps
    return False


def _tail_of(tgt, dart):
    a, b = tgt.edges[dart[0]]
    return a if dart[1] == 1 else b


def _head_of(tgt, dart):
    a, b = tgt.edges[dart[0]]
    return b if dart[1] == 1 else a


def _vertices(g):
    out = set()
    for (a, b) in g.edges.values():
        out.add(a)
        out.add(b)
    return out


def alg_iter(src, tgt, chains, alpha1, alpha2, steps=200, tol=1e-13, verbose=False):
    """[Elast] alg:iter on the class given by `chains`, starting from unit
    target lengths.  Returns (lambdas, ell2, map): lambdas[i] is the
    multiplicative factor of step i, and the last estimates SF[phi]."""
    m = FastMap(src, tgt, chains)
    ell2 = {e: 1.0 for e in alpha2}
    out = []
    for i in range(steps):
        m.harmonic(alpha1, ell2)
        v = m.pushforward(alpha1, ell2)
        nxt = {e: alpha2[e] * v[e] for e in alpha2}
        norm = float(np.sqrt(sum(x * x for x in nxt.values())))
        cur = float(np.sqrt(sum(x * x for x in ell2.values())))
        lam = norm / cur
        out.append(lam)
        ell2 = {e: nxt[e] / norm for e in alpha2}
        if verbose:
            print(f"  iter {i}: lambda = {lam:.12f}")
        if len(out) >= 2 and abs(out[-1] - out[-2]) < tol * max(1.0, abs(out[-1])):
            break
    return out, ell2, m
