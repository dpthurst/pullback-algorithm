"""The homotopy minimiser for Emb[phi]: [Elast] Algorithm 6.22 (alg:iter),
the length iteration, with an exact rational harmonic step and an exact
fixed-point solve.  Plan item 170.

    Emb[phi] = inf over psi in [phi] of Emb(psi)        (intro.tex eq:emb-homotopy)

The harmonic step H_[phi](l) is the unique minimiser of
Dir^{a1}(r) = sum_e r(e)^2/alpha1(e) over the relaxed polytope

    Rlx_l[phi] = { r >= 0 : sum_e n_c(e) r(e) >= l[phi . c]
                   for every marked curve c on G1 crossing each edge
                   at most twice }                    (harmonic.tex def:relaxed,
                                                  lem:relaxed (relaxed-simple))

so the polytope is cut out by finitely many half-spaces, each linear in l.
That is the max-flow/min-cut side of taut.tex thm:maxflow-mincut: the primal
variable r is a length on G1, the dual y >= 0 weights the curves, and

    r = D1(w),  w = A^T y,  v = N_[phi](w) = R^T y,  Dir = y^T R l,

the last identity being rem:Dir-deriv (d Dir_[phi]/d l = 2 N(w)) with the
envelope theorem.  So the push-forward multiplicity v -- the only thing
steps (2)-(3) of alg:iter produce -- is read off the *dual*; no explicit
map is reconstructed.

Then Iter(l) = D2(v) = alpha2 * v, and on each cone of linearity of
H_[phi] (a fixed QP active set S) Iter is a *linear* map

    M = diag(alpha2) . R_S^T . (A_S D1 A_S^T)^-1 . R_S .

M need NOT be entrywise nonnegative (at the tripod fixed point it has
negative entries), so Perron-Frobenius is NOT invoked: the fixed point is
found from the exact characteristic polynomial and verified against the
*real* iteration, Iter(l*) = lambda l* (verified_fixed_points).  An interior
fixed point with multiplier lambda has Emb[phi] = lambda
(prop:fixed-lambda + prop:lambda-filling-emb, filling.tex:716/169); any
fixed point gives DR(l) = lambda as a certified LOWER bound, since
DR(l) = Dir^l[phi]/Dir^l[id] <= SF_Dir[phi] = Emb[phi] (thm:emb-sf,
filling.tex:516).

The output is a certified bracket (plan D5: exact certificates + bracketed
numerics, NO interval arithmetic):

    LB = max DR over the exactly verified fixed points   (thm:emb-sf)
    UB = Emb of an explicit representative in the class  (lem:filling-fill)

which closes when the two agree.  Boundary fixed points (some l(e) = 0)
are detected and reported; the escape procedure of Algorithm 6.49
(alg:Emb, filling.tex:1239) is NOT implemented -- on every test case so far
(including the paper's examp:partial-filling) the bracket closes without
it, via an exhibited representative.  Note Emb can be irrational on rational
data ([Elast] alg:iter's eigenvalue on a cone), so a bracket, not a value,
is the right output in general.

Everything is exact `fractions.Fraction`; nothing printed depends on set or
dict iteration order (sorted iteration throughout; diff PYTHONHASHSEED=0
against 1).
"""

from __future__ import annotations

from fractions import Fraction as F

from .refined import Graph, GraphMap, inv, free_reduce

__all__ = ["Minimiser", "minimise", "curve_rows", "qp_active_set",
           "lin_solve", "charpoly", "rational_roots", "null_vector"]


# ------------------------------------------------- the relaxed polytope
def _chain_counts(chain, labels):
    """n_c: traversals per edge label, ignoring direction."""
    n = {}
    for (e, _s) in chain:
        n[e] = n.get(e, 0) + 1
    return n


def _cyclic_red(chain):
    """Cyclic reduction: free reduction, then drop a seam backtrack
    (valid for closed walks only)."""
    w = free_reduce(chain)
    while len(w) >= 2 and w[0] == (w[-1][0], -w[-1][1]):
        w = w[1:-1]
    return w


def curve_rows(G, chains, tgt, max_crossings=1):
    """The finitely many curve constraints of Rlx_l[phi].

    A *marked curve* is (intro.tex def:curve; harmonic.tex lem:relaxed
    condition (relaxed-simple)) either an arc between two marked vertices
    with no marked vertex in its interior, or a closed loop avoiding the
    marked vertices -- each reduced (no immediate backtrack, and a closed
    curve's seam must not backtrack) and crossing each edge at most
    `max_crossings` times.  The default of ONE crossing per edge is the
    complete set for the graphs in the library (every edge of a simple
    curve is crossed once) and is what the tests use; the paper allows
    two.  Each curve gives

        sum_e n_c(e) r(e) >= l[phi . c]                      (eq:relaxed)

    with l[phi.c] the length of the geodesic of the image class: free
    reduction for an arc, cyclic reduction for a closed curve.

    The marked set is the source graph's: it determines BOTH the curve
    family and (through [phi]) the class.  A free class (marked = {}) is
    all loops, no arcs.

    Returns a dict {(n_tuple, R_tuple): (n, R, closed)} where n is the
    multiplicity vector of the walk over sorted(G.edges) and R the
    multiplicity vector of the reduced image over sorted(tgt.edges).
    """
    labels1 = sorted(G.edges, key=str)
    labels2 = sorted(tgt.edges, key=str)
    out = {}

    def record(walk, closed):
        img = []
        for d in walk:
            ch = chains[d[0]]
            if d[1] == -1:
                ch = tuple(inv(x) for x in reversed(ch))
            img.extend(ch)
        img = _cyclic_red(img) if closed else free_reduce(img)
        if not img:
            return
        n = _chain_counts(walk, labels1)
        R = _chain_counts(img, labels2)
        key = (tuple(n.get(e, 0) for e in labels1),
               tuple(R.get(e, 0) for e in labels2))
        out.setdefault(key, (key[0], key[1], closed))

    # (a) arcs: reduced walks from a marked vertex to a marked vertex,
    # including closed ones (an arc from v to v is a closed walk through
    # the marked point -- [Elast] def:curve allows the two boundary points
    # of an arc to map to the same marked point)
    def walk(start, v, sofar, counts):
        if sofar and v in G.marked:
            record(sofar, v == start)
        for d in G.out_darts(v):
            e = d[0]
            if counts.get(e, 0) >= max_crossings:
                continue
            if sofar and sofar[-1] == (e, -d[1]):
                continue
            c2 = dict(counts)
            c2[e] = c2.get(e, 0) + 1
            walk(start, G.head(d), sofar + (d,), c2)

    for s in sorted(G.marked, key=str):
        walk(s, s, (), {})
    # Closed loops avoiding marked vertices, based at any vertex.  With
    # max_crossings = 1 each edge is crossed at most once, so a loop is a
    # SIMPLE cycle of G: no repeated vertex.  The DFS therefore stops at the
    # first revisit of any vertex (other than the start, which closes the
    # loop), which keeps the enumeration at the number of simple cycles --
    # finite and small for the library's graphs.  (Allowing two crossings
    # per edge, the paper's bound, would explore non-simple closed walks
    # and is exponentially larger; it is not the default.)
    for s in sorted(G.vertices(), key=str):
        if s in G.marked:
            continue
        stack = [(s, (), frozenset({s}))]
        while stack:
            v, sofar, seen = stack.pop()
            for d in G.out_darts(v):
                e = d[0]
                if sofar and sofar[-1] == (e, -d[1]):
                    continue
                if e in seen:
                    continue
                w = G.head(d)
                if w in G.marked:
                    continue
                word2 = sofar + (d,)
                if w == s and len(word2) >= 2:
                    # record once per loop: the first dart must be the
                    # minimum over rotations and reversals (string keys,
                    # since edge labels may not be comparable)
                    rv = tuple(inv(x) for x in reversed(word2))
                    rots = [word2[i:] + word2[:i] for i in range(len(word2))]
                    rots += [rv[i:] + rv[:i] for i in range(len(rv))]
                    if min(rots, key=repr) == word2:
                        record(word2, True)
                if w not in seen:
                    stack.append((w, word2, seen | {e, w}))
    return {k: out[k] for k in sorted(out)}


# ------------------------------------------------------------- exact QP
def lin_solve(Q, b):
    """Solve Q y = b exactly; None if Q is singular (Fractions)."""
    n = len(b)
    if n == 0:
        return []
    M = [[F(Q[i][j]) for j in range(n)] + [F(b[i])] for i in range(n)]
    for col in range(n):
        piv = None
        for r in range(col, n):
            if M[r][col] != 0:
                piv = r
                break
        if piv is None:
            return None
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [x / pv for x in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0:
                fac = M[r][col]
                M[r] = [a - fac * c for a, c in zip(M[r], M[col])]
    return [M[i][n] for i in range(n)]


def qp_active_set(rows, b, D, max_iter=2000):
    """min  sum_e r(e)^2 / D(e)   s.t.  rows . r >= b,  r >= 0  (exact).

    rows: list of coefficient tuples (same length as D).
    b:    list of Fractions, b >= 0.

    Returns (r, [(index, multiplier)], S) with the KKT conditions
    verified exactly: r feasible, multipliers >= 0 on S, r = D A_S^T y.

    Primal active-set method.  The r >= 0 rows are appended to `rows`
    with right-hand side 0, so the feasible set is a polyhedron and the
    minimiser exists.
    """
    n = len(D)
    rows = [tuple(F(x) for x in row) for row in rows]
    b = [F(x) for x in b]
    ncurve = len(rows)
    for e in range(n):
        unit = tuple(F(1) if i == e else F(0) for i in range(n))
        rows.append(unit)
        b.append(F(0))
    A = rows
    m = len(A)
    # feasible start: r = T * 1 with T large enough
    T = F(1)
    for i in range(ncurve):
        s = sum(A[i], F(0))
        if s > 0:
            T = max(T, b[i] / s)
    T = max(T, F(1))
    r = [T] * n
    S = []
    for _ in range(max_iter):
        if S:
            k = len(S)
            Q = [[sum((A[i][e] * A[j][e] * D[e] for e in range(n)), F(0))
                  for j in S] for i in S]
            bs = [b[i] for i in S]
            mu = lin_solve(Q, bs)
            if mu is None:
                S = S[:-1]
                continue
            rs = [sum((D[e] * A[i][e] * mu[k] for k, i in enumerate(S)), F(0))
                  for e in range(n)]
        else:
            mu = []
            rs = [F(0)] * n
        if rs == r:
            bad = [k for k in range(len(S)) if mu[k] < 0]
            if not bad:
                for i in range(ncurve):
                    if sum((A[i][e] * r[e] for e in range(n)), F(0)) < b[i]:
                        raise RuntimeError("infeasible point reported optimal")
                return (tuple(r), [(S[k], mu[k]) for k in range(len(S))],
                        tuple(S))
            k = min(bad, key=lambda k: (mu[k], -k))
            del S[k]
            continue
        p = [rs[e] - r[e] for e in range(n)]
        alpha, block = F(1), None
        for i in range(m):
            if i in S:
                continue
            ap = sum((A[i][e] * p[e] for e in range(n)), F(0))
            if ap < 0:
                t = (b[i] - sum((A[i][e] * r[e] for e in range(n)), F(0))) / ap
                if t < alpha:
                    alpha, block = t, i
        if alpha <= 0:
            if block is not None and block not in S:
                S.append(block)
                continue
            raise RuntimeError("active set stalled")
        r = [r[e] + alpha * p[e] for e in range(n)]
        if block is not None:
            S.append(block)
    raise RuntimeError("QP active set did not terminate")


# ------------------------------------------- the harmonic step + alg:iter
class Minimiser:
    """A homotopy class of maps G1 -> G2, with alg:iter on it.

    G1, G2: refined.Graph (marked elastic).  The class is the marked
    homotopy class: G1.marked lists the source vertices whose images are
    pinned (empty = free class, the sessions' convention).
    """

    def __init__(self, G1, G2, chains, name="", max_crossings=1):
        self.G1, self.G2 = G1, G2
        self.name = name
        self.chains = {e: (free_reduce(tuple(chains[e])) if e in chains else ())
                       for e in G1.edges}
        rows = curve_rows(G1, self.chains, G2, max_crossings=max_crossings)
        # keep only constraints with a nonzero right-hand side class
        self.keys = [k for k in rows if any(k[1])]
        self.A = [list(k[0]) for k in self.keys]
        self.R = [list(k[1]) for k in self.keys]
        self.D = [self.G1.alpha[e] for e in sorted(G1.edges, key=str)]
        self.labels1 = sorted(G1.edges, key=str)
        self.labels2 = sorted(G2.edges, key=str)

    # -- one harmonic step: l -> (m, w, v, Dir, DR, l_next)
    def harmonic(self, l):
        l = {e: F(l[e]) for e in self.labels2}
        b = [sum((F(Ri[j]) * l[e] for j, e in enumerate(self.labels2)), F(0))
             for Ri in self.R]
        r, ySall, S = qp_active_set(self.A, b, self.D)
        nc = len(self.A)
        yS = [(i, yi) for i, yi in ySall if i < nc]   # curve rows only
        S = tuple(i for i in S if i < nc)
        w = [F(0)] * len(self.D)
        v = [F(0)] * len(self.labels2)
        for i, yi in yS:
            for e in range(len(self.D)):
                w[e] += F(self.A[i][e]) * yi
            for j in range(len(self.labels2)):
                v[j] += F(self.R[i][j]) * yi
        Dir = sum((yi * b[i] for i, yi in yS), F(0))
        Dir_id = sum((l[e] ** 2 / self.G2.alpha[e] for e in self.labels2), F(0))
        m = {e: self.D[i] * w[i] for i, e in enumerate(self.labels1)}
        return {
            "l": l,
            "m": m,
            "w": w,
            "v": {e: v[j] for j, e in enumerate(self.labels2)},
            "Dir": Dir,
            "Dir_id": Dir_id,
            "DR": Dir / Dir_id if Dir_id else None,
            "active": S,
            "l_next": {e: self.G2.alpha[e] * v[j]
                       for j, e in enumerate(self.labels2)},
        }

    def iterate(self, l0, steps=40, verbose=False):
        """alg:iter, projectively (normalise l each step)."""
        l = {e: F(l0[e]) for e in self.labels2}
        hist = []
        for i in range(steps):
            h = self.harmonic(l)
            hist.append(h)
            if verbose:
                print(i, "l=", l, "DR=", h["DR"], "act=", h["active"])
            nxt = h["l_next"]
            tot = sum(nxt.values(), F(0))
            if tot == 0:
                break
            l = {e: nxt[e] / tot for e in self.labels2}
        return hist

    def cone_matrix(self, active):
        """M = diag(a2) R_S^T (A_S D A_S^T)^-1 R_S : the linear map Iter
        on the cone of l whose active set is exactly `active`.  NOT
        assumed nonnegative."""
        S = list(active)
        n2 = len(self.labels2)
        if not S:
            return [[F(0)] * n2 for _ in range(n2)]
        Q = [[sum((F(self.A[i][e]) * F(self.A[j][e]) * F(self.D[e])
                   for e in range(len(self.D))), F(0)) for j in S] for i in S]
        inv = []
        for k in range(len(S)):
            unit = [F(1) if i == k else F(0) for i in range(len(S))]
            col = lin_solve(Q, unit)
            if col is None:
                return None
            inv.append(col)
        M = [[F(0)] * n2 for _ in range(n2)]
        for i in range(n2):
            for j in range(n2):
                tot = F(0)
                for a in range(len(S)):
                    for bb in range(len(S)):
                        tot += F(self.R[S[a]][j]) * inv[a][bb] * F(self.R[S[bb]][i])
                M[i][j] = F(self.G2.alpha[self.labels2[i]]) * tot
        return M


# ------------------------------------------- exact linear algebra
def matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[sum((A[i][k] * B[k][j] for k in range(m)), F(0)) for j in range(p)]
            for i in range(n)]


def charpoly(M):
    """Characteristic polynomial of a square Fraction matrix: list of
    coefficients, highest degree first (no divisions: Leverrier)."""
    n = len(M)
    traces = []
    cur = [[F(1) if i == j else F(0) for j in range(n)] for i in range(n)]
    for k in range(1, n + 1):
        cur = matmul(cur, M)
        traces.append(sum((cur[i][i] for i in range(n)), F(0)))
    c = [F(1)]
    for k in range(1, n + 1):
        s = traces[k - 1]
        for i in range(1, k):
            s += c[i] * traces[k - i - 1]
        c.append(-s / k)
    return c


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _divisors(n):
    n = abs(int(n))
    if n == 0:
        return [1]
    out = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            out.append(i)
            if i != n // i:
                out.append(n // i)
        i += 1
    return sorted(out)


def rational_roots(coeffs):
    """Rational roots of a rational polynomial given highest degree first.

    Clear denominators, then the rational root theorem over the integers:
    p/q with p | a_0 and q | a_n (a_n the leading coefficient).
    """
    import math
    c = [F(x) for x in coeffs]
    while c and c[0] == 0:
        c.pop(0)
    while len(c) >= 2 and c[-1] == 0:      # roots at 0: ignored (lam > 0)
        c.pop()
    if len(c) <= 1:
        return []
    lcm = 1
    for x in c:
        lcm = lcm * x.denominator // math.gcd(lcm, x.denominator)
    ints = [int(x * lcm) for x in c]
    lead, const = ints[0], ints[-1]
    out = []
    for p in _divisors(const):
        for q in _divisors(lead):
            for sign in (1, -1):
                r = F(sign * p, q)
                if r == 0:
                    continue
                val = F(0)
                for coef in c:
                    val = val * r + coef
                if val == 0:
                    out.append(r)
                    break
    return [(r, 1) for r in sorted(set(out), reverse=True)]


def null_vector(A):
    """A nonzero vector in the kernel of A over Q, or None (exact)."""
    n = len(A)
    m = len(A[0]) if n else 0
    M = [[F(x) for x in row] for row in A]
    piv_cols = []
    r = 0
    for col in range(m):
        piv = None
        for i in range(r, n):
            if M[i][col] != 0:
                piv = i
                break
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        pv = M[r][col]
        M[r] = [x / pv for x in M[r]]
        for i in range(n):
            if i != r and M[i][col] != 0:
                fac = M[i][col]
                M[i] = [a - fac * b for a, b in zip(M[i], M[r])]
        piv_cols.append(col)
        r += 1
        if r == n:
            break
    free = [c for c in range(m) if c not in piv_cols]
    if not free:
        return None
    f = free[0]
    x = [F(0)] * m
    x[f] = F(1)
    for i, c in enumerate(piv_cols):
        x[c] = -M[i][f]
    return x


def verified_fixed_points(C, M):
    """All (lam, l, h) with lam rational, l >= 0, l != 0, that satisfy the
    *actual* iteration Iter(l) = lam l exactly (not just the linearised
    cone map).  Sorted by lam descending."""
    out = []
    n = len(M)
    poly = charpoly(M)
    for lam, _m in sorted(rational_roots(poly), reverse=True):
        if lam <= 0:
            continue
        A = [[M[i][j] - (lam if i == j else 0) for j in range(n)] for i in range(n)]
        vec = null_vector(A)
        if vec is None:
            continue
        if all(x < 0 for x in vec):
            vec = [-x for x in vec]
        if any(x < 0 for x in vec):
            continue
        tot = sum(vec, F(0))
        if tot == 0:
            continue
        l = {e: vec[i] / tot for i, e in enumerate(C.labels2)}
        h = C.harmonic(l)
        if all(h["l_next"][e] == lam * l[e] for e in C.labels2):
            out.append((lam, l, h))
    out.sort(key=lambda t: (-t[0], tuple(sorted(t[1].items()))))
    return out


def minimise(C, l0=None, max_steps=120):
    """alg:iter to find the cone, then the exact fixed point on it.

    Returns a dict with the iteration history, every exactly verified
    fixed point, and the certified lower bound for Emb[phi]:
        LB = max DR over the verified fixed points  (thm:emb-sf: DR <= Emb)
        interior = the fixed points with l > 0 (there Emb = lam exactly,
                   prop:fixed-lambda + prop:lambda-filling-emb)
    The certified bracket is [LB, Emb(psi)] for any explicit psi in [phi].
    """
    labels2 = C.labels2
    if l0 is None:
        l0 = {e: F(1) + F(i + 1, 97) for i, e in enumerate(labels2)}
    l = {e: F(l0[e]) for e in labels2}
    hist = []
    prev, stable = None, 0
    for _ in range(max_steps):
        h = C.harmonic(l)
        hist.append((dict(l), h))
        if h["active"] == prev:
            stable += 1
        else:
            stable, prev = 0, h["active"]
        nxt = h["l_next"]
        tot = sum(nxt.values(), F(0))
        if tot == 0:
            break
        newl = {e: nxt[e] / tot for e in labels2}
        if newl == l or stable >= 2:
            break
        l = newl
    S = hist[-1][1]["active"]
    M = C.cone_matrix(S)
    fps = verified_fixed_points(C, M) if M is not None else []
    if not fps:
        # the last active set may be transient: try the ones before it
        for k in range(len(hist) - 2, -1, -1):
            M = C.cone_matrix(hist[k][1]["active"])
            if M is None:
                continue
            fps = verified_fixed_points(C, M)
            if fps:
                break
    lb = max((t[2]["DR"] for t in fps if t[2]["DR"] is not None), default=None)
    interior = [t for t in fps if all(t[1][e] > 0 for e in labels2)]
    return {
        "history": hist,
        "fixed_points": fps,
        "LB": lb,
        "interior": [(t[0], t[1]) for t in interior],
        "M": M,
    }
