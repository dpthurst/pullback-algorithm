"""Spherical CEPS uniformizer for a PL sphere: the special-vertex device.

Conventions, sources, root causes of the four gate failures and the measured
convergence orders are documented in
``.ultramath/smwqhd2vy/m5/report.md``.  Primary sources:

  * M. Gillespie, B. Springborn, K. Crane, *Discrete Conformal Equivalence of
    Polyhedral Surfaces*, ACM TOG 40(4) 103 (2021) -- ``refs/``; sections 4 and
    7: the variational principle, the modified (peacock) flip algorithm, the
    special vertex, the constraints and the flat-disk layout;
  * B. Springborn, *Discrete conformal maps and ideal hyperbolic polyhedra*,
    arXiv:1707.06848 -- the triangle function f, the energy H_Θ, the Delaunay
    criterion in horocyclic-arc form, delta_{Δ,λ} and Prop. 5.16 / 7.17.

Conventions borrowed from the MIT reference implementation
https://github.com/MarkGillespie/CEPS at commit
b734d2cdc2afad7dc6afe9d609a108bc9a68c24f (MIT licence, (c) 2021 Mark
Gillespie) are marked at the point of use: the clamped-cosine corner angle
(``geometry-central`` ``EdgeLengthGeometry::cornerAngle``), the dropping of
cotan weights on zero-area faces (``PartiallyDecoratedTriangulation::
partialCotanLaplacian``), the objective/gradient signs of
``SphericalUniformization.cpp`` (``sphericalObjective``), the rule that the
objective is ``inf`` on a triangle with no angles, and the clamp-to-the-
constraints step of ``PetscWrapper::cleanUpTriangulation``.  No C++ code is
copied; the equations are re-derived and cross-checked in
``tests/test_ceps_spherical_controls.py``.
"""

"""Core Penner / ideal-hyperbolic algebra for spherical CEPS.

Conventions (fixed once, used everywhere; see report.md):

  A triangulated sphere carries positive *chordal* length data
  ``ell[i,j]`` = Euclidean chord ``|x_i - x_j|`` of the configuration the
  metric is meant to represent.  Penner coordinates are

      lambda_ij = 2 log ell_ij                       (GSC eq. (6)/eq. (17))

  Discrete conformal vertex scaling is ``ell'_ij = e^{(u_i+u_j)/2} ell_ij``,
  i.e. ``lambda'_ij = lambda_ij + u_i + u_j``      (Penner scaling, Prop. 3.2).

  For a triangle with vertices (i,j,k) the three *horocyclic arc lengths* are

      alpha_i = ell_jk / (ell_ki ell_ij)            (GSC eq. (17))

  and the Euclidean angle at i is the angle opposite the side ``ell_jk``.

  The ideal (hyperbolic) Delaunay condition at an edge ij with opposite
  vertices k, l is

      alpha_k + alpha_l <= alpha_i + alpha_j        (GSC eq. (18), Spr. (13))

  written with the *scaled* arc lengths; scaling commutes with the Ptolemy
  flip, so the check can be done directly with ell' = exp(lambda'/2).
"""

import math
from dataclasses import dataclass, field

import numpy as np

LOG2 = math.log(2.0)


# --------------------------------------------------------------------------
# elementary triangle algebra
# --------------------------------------------------------------------------
def tri_angles(l1, l2, l3):
    """Angles opposite l1, l2, l3; the *clamped* cosine convention.

    Returns (a1, a2, a3) with a1 opposite l1 etc.  The law-of-cosines ratio is
    clamped into [-1, 1], so a triangle whose sides violate the (strict)
    triangle inequality yields the degenerate limit: the angle opposite the
    longest side is pi and the other two are 0.  This is exactly the convention
    of the MIT CEPS reference implementation (``geometry-central``,
    ``EdgeLengthGeometry::cornerAngle``: ``q = clamp(q, -1.0, 1.0);
    angle = acos(q)``), and it is the value of the analytic continuation of the
    triangle function f used in Springborn 1707.06848 Remark 7.6 / Springborn-
    Schroeder-Pinkall 2008 on the region where f is extended off its domain A.
    Clamping makes E a continuous (C^1) function of u on the whole of R^V
    instead of raising: f -> pi * x_max there, whose gradient is (pi, 0, 0),
    matching the limit of the gradient from inside A.

    Do not "fix" this back into an exception: the energy of a scaled metric is
    needed at trial points that are not realizable Euclidean cone metrics.
    """
    def _ang(opp, u, v):
        q = (u * u + v * v - opp * opp) / (2.0 * u * v)
        return math.acos(-1.0 if q <= -1.0 else (1.0 if q >= 1.0 else q))
    a1 = _ang(l1, l2, l3)
    a2 = _ang(l2, l3, l1)
    a3 = _ang(l3, l1, l2)
    return a1, a2, a3


def arc_lengths(lij, ljk, lki):
    """(alpha_i, alpha_j, alpha_k) = horocyclic arc lengths (GSC eq. (17))."""
    return (ljk / (lki * lij), lki / (lij * ljk), lij / (ljk * lki))


def ptolemy(lij, lki, llj, ljk, lli):
    """Ptolemy flip: length of the new diagonal kl (GSC eq. (8))."""
    return (lki * llj + ljk * lli) / lij


def penner_to_len(lam):
    return math.exp(0.5 * lam)


def len_to_penner(l):
    return 2.0 * math.log(l)


def lobachevsky(alpha):
    """Milnor's Lobachevsky function L(alpha) = -int_0^alpha log|2 sin t| dt.

    pi-periodic and odd; computed on [0, pi/2] by integration by parts plus
    fixed-order Gauss-Legendre (the integrand t*cot(t) is analytic there).
    """
    # reduce by periodicity / oddness into [0, pi/2]
    x = float(alpha)
    sign = 1.0
    if x < 0.0:
        x, sign = -x, -sign
    x = x % math.pi                      # L is pi-periodic
    if x > 0.5 * math.pi:                # L(pi - x) = -L(x) (oddness + period)
        x = math.pi - x
        sign = -sign
    if x < 1e-14:
        return 0.0
    # L(x) = -x log(2 sin x) + int_0^x t cot(t) dt
    nodes, weights = np.polynomial.legendre.leggauss(48)
    t = 0.5 * x * (nodes + 1.0)
    w = 0.5 * x * weights
    f = np.ones_like(t)
    nz = t != 0.0
    f[nz] = t[nz] / np.tan(t[nz])
    integ = float(np.dot(w, f))
    val = -x * math.log(2.0 * math.sin(x)) + integ
    return sign * val


# --------------------------------------------------------------------------
# triangulated surface / disk bookkeeping
# --------------------------------------------------------------------------
def canonical_edge(i, j):
    return (i, j) if i < j else (j, i)


@dataclass
class Disk:
    """A triangulated disk: oriented faces, edge -> adjacent (face, apex).

    ``faces`` is a list of 3-tuples, consistently counter-clockwise oriented.
    ``adj[e]`` is the list of (face_index, apex_vertex) with ``e`` in that face.
    Interior edges have two entries, boundary edges one.
    """

    faces: list
    adj: dict = field(default_factory=dict)

    @staticmethod
    def build(faces):
        adj = {}
        for fi, f in enumerate(faces):
            a, b, c = f
            for (i, j), k in ((( a, b), c), ((b, c), a), ((c, a), b)):
                adj.setdefault(canonical_edge(i, j), []).append((fi, k))
        return Disk(list(faces), adj)

    def edges(self):
        return list(self.adj.keys())

    def boundary_edges(self):
        return [e for e, l in self.adj.items() if len(l) == 1]

    def interior_edges(self):
        return [e for e, l in self.adj.items() if len(l) == 2]

    def boundary_loop(self):
        """Ordered boundary cycle of vertices (requires a simple boundary)."""
        bd = self.boundary_edges()
        nxt = {}
        for (i, j) in bd:
            nxt.setdefault(i, []).append(j)
            nxt.setdefault(j, []).append(i)
        if not bd:
            raise ValueError("no boundary")
        if any(len(v) != 2 for v in nxt.values()):
            raise ValueError("boundary is not a simple cycle")
        start = bd[0][0]
        loop = [start]
        prev, cur = None, start
        while True:
            a, b = nxt[cur]
            nxt_v = b if a == prev else a
            prev, cur = cur, nxt_v
            if cur == start:
                break
            loop.append(cur)
        return loop

    def vertex_edges(self):
        ve = {}
        for (i, j) in self.adj:
            ve.setdefault(i, []).append(canonical_edge(i, j))
            ve.setdefault(j, []).append(canonical_edge(i, j))
        return ve

    def vertex_faces(self):
        vf = {}
        for fi, f in enumerate(self.faces):
            for v in f:
                vf.setdefault(v, []).append(fi)
        return vf


def sphere_edges(faces):
    out = {}
    for fi, f in enumerate(faces):
        a, b, c = f
        for (i, j), k in ((( a, b), c), ((b, c), a), ((c, a), b)):
            out.setdefault(canonical_edge(i, j), []).append((fi, k))
    return out


def check_sphere(faces, n):
    """Minimal validation: 3 distinct vertices per face, closed, edge-manifold,
    Euler characteristic 2, connected."""
    ed = sphere_edges(faces)
    for f in faces:
        if len(set(f)) != 3:
            raise ValueError(f"degenerate face {f}")
    for e, l in ed.items():
        if len(l) != 2:
            raise ValueError(f"edge {e} has {len(l)} incident faces (not a closed manifold)")
    v = n
    e = len(ed)
    f = len(faces)
    if v - e + f != 2:
        raise ValueError(f"Euler characteristic {v-e+f} != 2")
    # connectivity of the 1-skeleton
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for (i, j) in ed:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    if len({find(i) for i in range(n)}) != 1:
        raise ValueError("1-skeleton is not connected")
    return ed


def disk_of_sphere(faces, edge_adj, v_inf):
    """Faces not containing v_inf, oriented consistently (inherited)."""
    out = [f for f in faces if v_inf not in f]
    return Disk.build(out)


def oriented_faces_sphere(faces, n):
    """Try to orient the sphere faces consistently; returns faces (orientation
    irrelevant to the intrinsic algebra, but kept for layout of the disk)."""
    return [tuple(f) for f in faces]


"""Spherical CEPS uniformizer with the special-vertex (GSC sec. 7) device.

Everything here is implemented from the primary sources:

  * retriangulation: the modified flip algorithm of GSC sec. 7.1 and
    Springborn Prop. 5.10; the Delaunay test is written with horocyclic arc
    lengths (GSC eq. (18) / Spr. eq. (13)), which needs no triangle inequality
    and is exact algebra in the Penner coordinates;
  * delta_{Delta,lambda}(v, v_inf): Springborn Prop. 5.16 (decorate with
    u(v_inf) = 0, u(v) = +inf for v != v_inf, flip, read the horocycle
    distance on the edge joining v to v_inf).  Prop. 5.11 lets us use a large
    finite T in place of +inf, so only genuine triangulations occur;
  * energy: the modified spherical energy of GSC sec. 7.2 / Springborn
    Def. 7.16 eq. (55) with (48),  E(u) = 2 pi sum u + sum_t 2 f - pi sum_e
    lambda_tilde, analytic gradient dE/du_v = 2 pi - theta_v + pi(deg_F -
    deg_E) and cotan-Laplacian Hessian; the scaling identity
    E(u + h 1) = E(u) + 2 pi h (Springborn Prop. 7.17) is an executable test;
  * minimisation: min E(u) s.t. u_v >= delta_v, damped active-set Newton with
    KKT reporting (no faked convergence: failures raise);
  * layout, stereographic reinsertion of v_inf at the pole, explicit Moebius
    gauge, convex-inscribed positions and their supporting-plane certificate.
"""

import math
import time

import numpy as np
from scipy.sparse import csr_matrix

_clock = time.perf_counter


TWO_PI = 2.0 * math.pi


class CEPSFailure(RuntimeError):
    """Raised instead of returning a non-solution."""


# --------------------------------------------------------------------------
# Penner state on a triangulated sphere or disk
# --------------------------------------------------------------------------
class PennerState:
    """Triangulated surface + *transported* base Penner coordinates ``lam``."""

    def __init__(self, faces, lam, closed=True):
        self.closed = closed
        self.faces = [tuple(f) for f in faces]
        self.lam = dict(lam)
        self.flips = 0
        self.stuck = []
        self._sync()

    def _sync(self):
        self.disk = Disk.build(self.faces)
        self.vs = sorted({v for f in self.faces for v in f})
        self.es = sorted(self.disk.adj.keys())
        self.eidx = {e: k for k, e in enumerate(self.es)}
        for e in list(self.lam):
            if e not in self.eidx:
                del self.lam[e]

    def copy(self):
        st = PennerState(self.faces, self.lam, self.closed)
        st.flips = self.flips
        st.stuck = list(self.stuck)
        return st

    def lam_t(self, e, u):
        i, j = e
        return self.lam[e] + u[i] + u[j]

    def len_t(self, e, u):
        return math.exp(0.5 * self.lam_t(e, u))

    def arc_at(self, f, u, v):
        """alpha_v = exp((lam_opp - lam_o1 - lam_o2)/2); always defined."""
        a, b, c = f
        l_a = self.lam_t(canonical_edge(b, c), u)
        l_b = self.lam_t(canonical_edge(c, a), u)
        l_c = self.lam_t(canonical_edge(a, b), u)
        if v == a:
            return math.exp(0.5 * (l_a - l_b - l_c))
        if v == b:
            return math.exp(0.5 * (l_b - l_c - l_a))
        if v == c:
            return math.exp(0.5 * (l_c - l_a - l_b))
        raise KeyError(v)

    def violates(self, e, u, tol=1e-13):
        """Ideal-Delaunay violation at an interior edge (GSC eq. (18))."""
        inc = self.disk.adj[e]
        if len(inc) != 2:
            return False
        (f1, k), (f2, l) = inc
        i, j = e
        if len({i, j, k, l}) != 4:
            return False
        A1, A2 = self.faces[f1], self.faces[f2]
        lhs = self.arc_at(A1, u, k) + self.arc_at(A2, u, l)
        rhs = (self.arc_at(A1, u, i) + self.arc_at(A1, u, j)
               + self.arc_at(A2, u, i) + self.arc_at(A2, u, j))
        scale = max(lhs, rhs, 1e-300)
        return lhs > rhs + tol * scale

    def flip(self, e, u):
        """Ptolemy flip; returns False (and records) if not possible."""
        inc = self.disk.adj[e]
        (f1, k), (f2, l) = inc
        i, j = e
        if len({i, j, k, l}) != 4:
            self.stuck.append((e, "shared vertex"))
            return False
        new_e = canonical_edge(k, l)
        if new_e in self.eidx:
            self.stuck.append((e, "duplicate"))
            return False
        lkl = ptolemy(self.len_t(e, u),
                      self.len_t(canonical_edge(k, i), u),
                      self.len_t(canonical_edge(l, j), u),
                      self.len_t(canonical_edge(j, k), u),
                      self.len_t(canonical_edge(l, i), u))
        if not (lkl > 0.0 and math.isfinite(lkl)):
            self.stuck.append((e, "nonpositive"))
            return False
        self.lam[new_e] = 2.0 * math.log(lkl) - u[k] - u[l]
        self.faces[f1] = (k, i, l)
        self.faces[f2] = (k, l, j)
        del self.lam[e]
        self.flips += 1
        self._sync()
        return True

    def retriangulate(self, u, maxflips=20000, tol=1e-13):
        """Modified flip algorithm: flip until ideal-Delaunay (GSC sec. 7.1).

        Returns the list of *locked* violations: interior edges that violate
        the ideal Delaunay condition (GSC eq. (18)) for which the Ptolemy flip
        is not a legal move, because the new diagonal is already an edge of the
        triangulation (the four vertices of the quad span K4).  Such an edge is
        locally rigid -- no sequence of flips changes it -- so the chart is as
        Delaunay as the combinatorics allow, and the energy is evaluated on it.
        They are *reported*, not swallowed: a non-empty list means the chart is
        not the Delaunay chart of the metric and the caller decides what to do.
        """
        n = 0
        blocked = set()
        while True:
            target = None
            for e in self.disk.interior_edges():
                if e in blocked:
                    continue
                if self.violates(e, u, tol):
                    target = e
                    break
            if target is None:
                return sorted(blocked)
            if not self.flip(target, u):
                blocked.add(target)
                continue
            n += 1
            if n > maxflips:
                raise CEPSFailure("flip algorithm exceeded its budget")


# --------------------------------------------------------------------------
# delta by Prop. 5.16
# --------------------------------------------------------------------------
def peacock(faces, lam, v_inf, T=40.0, maxflips=10000):
    """Adjusted Delaunay (peacock) chart for the partial decoration
    u(v_inf) = 0, u(v) = +inf (v != v_inf), plus the signed horocycle
    distances delta(v, v_inf)  (Springborn Prop. 5.10 / 5.16, GSC sec. 7.1).

    Prop. 5.16 decorates only v_inf, so by Prop. 5.7 every punctured Delaunay
    face is a monogon with peripheral vertex v_inf, and the adjusted
    triangulation refines each monogon by the arc joining its central vertex to
    v_inf.  Accordingly the modified flip algorithm of Prop. 5.10 is run with
    the local condition (13) evaluated at a large finite T in place of +inf
    (Prop. 5.11: the adjusted triangulation is Delaunay for every sufficiently
    large finite decoration).  In that limit exactly the edges of the link of
    v_inf violate (13); flipping one replaces it by a new edge (v_inf, c) and
    enlarges the link, so the process terminates because deg(v_inf) increases.

    A flip whose new edge already exists cannot occur in a simplicial
    triangulation; that is the degenerate monogon case of the theory (the
    second phase of Prop. 5.10).  Its arc value is still well defined by the
    Ptolemy relation, and delta is the *minimum* over arcs (Def. 5.14), so the
    value is accumulated.  Every step is recorded in ``info``.
    """
    n = max(max(f) for f in faces) + 1
    u = {v: T for v in range(n)}
    u[v_inf] = 0.0
    st = PennerState(faces, lam, closed=True)
    special = {}
    for e in st.eidx:
        if v_inf in e:
            v = e[0] if e[1] == v_inf else e[1]
            special[v] = st.lam[e]
    blocked = []
    degenerate = []

    def neigh():
        return {j for e in st.disk.adj if v_inf in e for j in e if j != v_inf}

    def link_edges():
        out = []
        for e in st.disk.adj:
            if v_inf in e:
                continue
            inc = st.disk.adj[e]
            if len(inc) != 2:
                continue
            if not any(v_inf in st.faces[fi] for fi, _ in inc):
                continue
            c = None
            for fi, apex in inc:
                if apex != v_inf:
                    c = apex
                    break
            if c is not None:
                out.append((e, c))
        return out

    for _ in range(maxflips):
        nb = neigh()
        target = None
        for (e, c) in link_edges():
            if e in blocked:
                continue
            if not st.violates(e, u):
                continue
            if c in nb:
                # degenerate flip: the arc (v_inf,c) exists already; its value
                # is still given by Ptolemy, and delta is the minimum over arcs
                a, b = e
                lkl = ptolemy(st.len_t(e, u),
                              st.len_t(canonical_edge(v_inf, a), u),
                              st.len_t(canonical_edge(c, b), u),
                              st.len_t(canonical_edge(b, v_inf), u),
                              st.len_t(canonical_edge(c, a), u))
                if lkl is not None and lkl > 0 and math.isfinite(lkl):
                    val = 2.0 * math.log(lkl) - u[v_inf] - u[c]
                    degenerate.append((e, c, val))
                    special[c] = min(special.get(c, math.inf), val)
                blocked.append(e)
                continue
            target = e
            break
        if target is None:
            break
        if not st.flip(target, u):
            blocked.append(target)
            continue
        for ee in st.eidx:
            if v_inf in ee:
                v = ee[0] if ee[1] == v_inf else ee[1]
                special[v] = min(special.get(v, math.inf), st.lam[ee])
    missing = [v for v in range(n) if v != v_inf and v not in special]
    info = {"flips": st.flips, "blocked": blocked, "missing": missing,
            "deg_v_inf": len(neigh()), "degenerate": degenerate,
            "chart_faces": list(st.faces), "chart_lam": dict(st.lam)}
    delta = {v: special[v] for v in special}
    return st, delta, info


def delta_bounds(faces, lam, v_inf, T=40.0):
    """Signed shortest horocycle distances delta(v, v_inf) (Prop. 5.16)."""
    st, delta, info = peacock(faces, lam, v_inf, T=T)
    if info["missing"]:
        raise CEPSFailure(
            "peacock chart did not connect the special vertex to "
            f"{info['missing']} (degenerate case not handled)")
    return delta, info


# --------------------------------------------------------------------------
# energy / gradient / Hessian on the punctured disk
# --------------------------------------------------------------------------
def energy_grad(u, st, vs):
    """E(u) and its gradient on the current (Delaunay) chart of the disk."""
    E = TWO_PI * sum(u[v] for v in vs)
    g = {v: TWO_PI for v in vs}
    for f in st.faces:
        a, b, c = f
        lab = st.len_t(canonical_edge(a, b), u)
        lbc = st.len_t(canonical_edge(b, c), u)
        lca = st.len_t(canonical_edge(c, a), u)
        aa, ab, ac = tri_angles(lbc, lca, lab)     # angles at a, b, c
        x1, x2, x3 = math.log(lbc), math.log(lca), math.log(lab)
        E += 2.0 * (aa * x1 + ab * x2 + ac * x3
                    + lobachevsky(aa) + lobachevsky(ab) + lobachevsky(ac))
        g[a] += math.pi - aa
        g[b] += math.pi - ab
        g[c] += math.pi - ac
    for e in st.es:
        i, j = e
        E -= math.pi * st.lam_t(e, u)
        g[i] -= math.pi
        g[j] -= math.pi
    return E, g


def hessian(u, st, vs):
    """Cotan-Laplacian (GSC 4.2.3, Spr. eq. (34)); removed faces contribute 0."""
    idx = {v: k for k, v in enumerate(vs)}
    n = len(vs)
    rows, cols, vals = [], [], []
    for f in st.faces:
        a, b, c = f
        lab = st.len_t(canonical_edge(a, b), u)
        lbc = st.len_t(canonical_edge(b, c), u)
        lca = st.len_t(canonical_edge(c, a), u)
        # Degenerate (area 0) triangles contribute no cotan weight: this is the
        # MIT CEPS convention (PartiallyDecoratedTriangulation::
        # partialCotanLaplacian skips faces with area <= 0), and it keeps the
        # clamped-angle convention of tri_angles from producing infinities.
        s = 0.5 * (lab + lbc + lca)
        if not (s > lab and s > lbc and s > lca):
            continue
        aa, ab, ac = tri_angles(lbc, lca, lab)
        for (i, j), ang in (((a, b), ac), ((b, c), aa), ((c, a), ab)):
            tan = math.tan(ang)
            if tan == 0.0:
                continue
            # D^2 of ONE triangle term of *this* energy.  E contains 2 f, and
            # x_i = (lambda_i + u_j + u_k)/2, so with D^2 f_x = sum cot(alpha_i)
            # (d x_j - d x_k)^2 (Springborn 1707.06848 Prop. 7.2(iii)) the
            # coefficient of (du_i - du_j)^2 for the corner opposite ij is
            # 2 * cot(alpha) * (1/2)^2 = cot(alpha)/2.  The factor 1/4 that one
            # might read off Prop. 7.7(iii) is the Hessian of sum f, i.e. of
            # half of E, and gives a *half* Newton step: the iteration then
            # converges linearly, not quadratically (measured; see report.md).
            w = 0.5 / tan
            # a nearly right-degenerate corner has cot ~ 0; a nearly flat one
            # has |cot| -> inf.  Cap so that a single bad face cannot make the
            # Newton matrix numerically singular in a way the line search
            # cannot recover from; the cap is far outside any physical weight.
            w = max(-1e6, min(1e6, w))
            idx_i, idx_j = idx[i], idx[j]
            rows += [idx_i, idx_j, idx_i, idx_j]
            cols += [idx_i, idx_j, idx_j, idx_i]
            vals += [w, w, -w, -w]
    return csr_matrix((vals, (rows, cols)), shape=(n, n))


def realizable(st, u, tol=0.0):
    """True iff every triangle of the chart is a strict Euclidean triangle.

    The energy, its gradient and its Hessian are those of the discrete
    conformal metric only on this set (Springborn 1707.06848 Def. 7.1 ``A`` /
    Def. 7.5).  Off it, ``energy_grad`` still returns the value of the convex
    extension (clamped angles, Springborn Remark 7.6), but the *iteration* must
    not go there: the extension is only C^1 and its cotan weights blow up.  The
    MIT reference implementation makes the same choice by returning
    ``inf`` from its objective on a triangle with no angles.
    """
    for f in st.faces:
        a, b, c = f
        lab = st.len_t(canonical_edge(a, b), u)
        lbc = st.len_t(canonical_edge(b, c), u)
        lca = st.len_t(canonical_edge(c, a), u)
        s = 0.5 * (lab + lbc + lca)
        m = max(lab, lbc, lca)
        if not s > m + tol * s:
            return False
    return True


def disk_state(faces, lam, v_inf):
    """Input sphere chart -> punctured disk chart (star of v_inf removed)."""
    ed = sphere_edges(faces)
    disk = disk_of_sphere(faces, ed, v_inf)
    lam_d = {e: lam[e] for e in disk.adj}
    st = PennerState(disk.faces, lam_d, closed=False)
    if v_inf in st.vs:
        raise CEPSFailure("special vertex survived in the disk")
    return st


class Result:
    pass


def solve(faces, ell, v_inf, maxit=80, tol=1e-11, margin=None,
          delta_T=40.0, verbose=False, polish=True, polish_iters=None,
          tight=1e-9):
    """Full spherical CEPS solve (GSC sec. 7).  Returns a Result or raises.

    Algorithm (see report.md of the swarm run for the measurements):

    ``delta``  the peacock / modified-flip chart and the bounds
               ``u_v >= -delta_v`` (Springborn Prop. 5.16, GSC sec. 7.1, 7.3).
    ``pin``    the vertices adjacent to the special vertex ``v_inf`` are set
               exactly to their bounds ``u_v = -delta_v``.  This is not a
               heuristic: GSC sec. 7.3 states that at a minimizer the
               inequalities are satisfied with equality for exactly the
               vertices adjacent to ``v_inf`` (Springborn 2019, Prop. 5.16), and
               the MIT reference implementation does the same thing in
               ``PetscWrapper::cleanUpTriangulation`` (clamp the boundary to the
               constraints, then run rounds of CETM with fixed boundary
               conditions).  The *only* thing that must be checked afterwards is
               the KKT condition at the pinned vertices, ``-grad_v <= 0``; if it
               fails, this solve raises instead of returning a non-answer.
    ``newton`` damped Newton on the remaining (interior) vertices.  At every
               trial point the chart is the ideal-Delaunay chart of the scaled
               metric (GSC sec. 7.1/7.2), the energy is the modified spherical
               energy, the gradient the cone-angle defect and the Hessian the
               cotan Laplacian of the *removed-face* convention.  The step is
               accepted on a decrease of E, and -- when E is already at the
               resolution ``eps_m |E|`` of a float -- on a decrease of the
               gradient norm on the *same* chart.  Without the second branch an
               Armijo test alone accepts zero-length steps at |grad| ~ 1e-8 and
               reports a "converged" solve whose KKT residual is 6e-8; that is
               exactly the failure this module used to have.

    All the extra outputs (positions, certificate, timings) are produced by
    :func:`run`.

    ``margin`` is accepted for backwards compatibility with the earlier
    "u = -delta + margin" starting heuristic and is ignored: the starting point
    is determined by the constraints themselves, so there is nothing to tune
    (``tests/test_ceps_spherical_controls.py`` checks that the answer does not
    depend on it, nor on the peacock limit T = ``delta_T``).
    """
    n = max(max(f) for f in faces) + 1
    check_sphere(faces, n)
    ed = sphere_edges(faces)
    lam = {e: 2.0 * math.log(ell[e]) for e in ed}

    t_phase = {"delta": 0.0, "pin": 0.0, "newton": 0.0, "polish": 0.0,
               "eval": 0.0}
    t0 = _clock()
    delta, dinfo = delta_bounds(faces, lam, v_inf, T=delta_T)
    t_phase["delta"] = _clock() - t0

    t0 = _clock()
    base = disk_state(faces, lam, v_inf)
    vs = base.vs
    bd = sorted({j for e in base.disk.adj if len(base.disk.adj[e]) == 1
                 for j in e})
    pin = [v for v in vs if v in set(bd)]
    free = [v for v in vs if v not in set(bd)]
    lo = np.array([-delta[v] if v in set(bd) else -np.inf for v in vs])
    lomap = {v: -delta[v] for v in pin}
    c0 = (sum(lomap.values()) / len(lomap)) if lomap else 0.0
    u = {v: (lomap[v] if v in lomap else c0) for v in vs}
    ifree = np.array([vs.index(v) for v in free], dtype=int)
    t_phase["pin"] = _clock() - t0
    n_eval = [0]
    t0 = _clock()

    def evaluate(uu):
        """(chart, E, grad, locked) at uu on the ideal-Delaunay chart."""
        st = base.copy()
        locked = st.retriangulate(uu)
        E, g = energy_grad(uu, st, vs)
        n_eval[0] += 1
        return st, E, g, locked

    t_phase["eval"] = 0.0
    def _eval_timed(uu):
        t = _clock()
        out = evaluate(uu)
        t_phase["eval"] += _clock() - t
        return out

    def to_vec(uu):
        return np.array([uu[v] for v in vs])

    def to_dict(vec):
        return {v: float(vec[k]) for k, v in enumerate(vs)}

    def kkt_of(uu, g):
        k = 0.0
        for v in vs:
            if v in lomap:
                k = max(k, -g[v])
            else:
                k = max(k, abs(g[v]))
        return float(k)

    # ------------------------------------------------------------------ newton
    t_newton = _clock()
    log = []
    nit = 0
    gmax_free = math.inf
    for nit in range(1, maxit + 1):
        st, E, g, locked = _eval_timed(u)
        gv = to_vec(g)
        gfree = float(np.max(np.abs(gv[ifree]))) if ifree.size else 0.0
        kkt = kkt_of(u, g)
        log.append({"it": nit, "E": E, "kkt": kkt, "g_free": gfree,
                    "flips": st.flips, "locked": len(locked),
                    "realizable": realizable(st, u)})
        if verbose:
            print(log[-1])
        gmax_free = gfree
        if gfree <= tol or ifree.size == 0:
            break
        H = hessian(u, st, vs).toarray()
        Hf = H[np.ix_(ifree, ifree)]
        gf = gv[ifree]
        d, *_ = np.linalg.lstsq(Hf, -gf, rcond=None)
        d = np.where(np.isfinite(d), d, 0.0)
        if float(gf @ d) >= 0.0:
            ev = np.linalg.eigvalsh(Hf)
            Hr = Hf + (max(1e-10, -float(ev.min()) + 1e-8)
                       + 1e-8 * float(np.max(np.abs(ev)))) * np.eye(Hf.shape[0])
            d2, *_ = np.linalg.lstsq(Hr, -gf, rcond=None)
            d2 = np.where(np.isfinite(d2), d2, 0.0)
            d = d2 if float(gf @ d2) < 0.0 else -gf
        pred = -float(gf @ d)
        uv = to_vec(u)
        faces_now = sorted(st.faces)
        epsE = 1e-15 * (1.0 + abs(E))
        moved = False
        t = 1.0
        for _ in range(60):
            un = uv.copy()
            un[ifree] += t * d
            u_new = to_dict(un)
            try:
                st_try, E_try, g_try, _lk = _eval_timed(u_new)
            except (ValueError, CEPSFailure, OverflowError, ZeroDivisionError):
                t *= 0.5
                continue
            gf_try = float(np.max(np.abs(to_vec(g_try)[ifree])))
            same = sorted(st_try.faces) == faces_now
            if E_try <= E - max(1e-4 * t * pred, epsE):
                moved = True
                break
            if same and E_try <= E + epsE and gf_try < gfree:
                moved = True
                break
            t *= 0.5
        if not moved:
            log[-1]["stopped"] = "no decrease"
            break
        u = u_new
    t_phase["newton"] = _clock() - t_newton

    # ------------------------------------------------- optional extra polish
    t0 = _clock()
    st, E, g, locked = _eval_timed(u)
    kkt = kkt_of(u, g)
    polish_log = []
    if polish and kkt > tol and ifree.size:
        # identical iteration, but the chart is frozen once it stops changing,
        # so that last-digit steps cannot be rejected because the flip pattern
        # oscillates.  This is the phase that takes |grad| from ~1e-9 to 1e-15.
        n_pol = polish_iters if polish_iters is not None else 40
        frozen = None
        for k in range(n_pol):
            gv = to_vec(g)
            gf = gv[ifree]
            gfree = float(np.max(np.abs(gf)))
            polish_log.append({"k": k, "g_free": gfree, "E": E})
            if gfree <= 1e-14:
                break
            if frozen is None or sorted(frozen.faces) != sorted(st.faces):
                frozen = st
            H = hessian(u, frozen, vs).toarray()
            Hf = H[np.ix_(ifree, ifree)]
            d, *_ = np.linalg.lstsq(Hf, -gf, rcond=None)
            d = np.where(np.isfinite(d), d, 0.0)
            if not np.any(d) or float(gf @ d) >= 0.0:
                d = -gf
            uv = to_vec(u)
            t = 1.0
            moved = False
            for _ in range(40):
                un = uv.copy()
                un[ifree] += t * d
                u_new = to_dict(un)
                try:
                    E_new, g_new = energy_grad(u_new, frozen, vs)
                except (ValueError, OverflowError, ZeroDivisionError):
                    t *= 0.5
                    continue
                g_new_free = float(np.max(np.abs(to_vec(g_new)[ifree])))
                if math.isfinite(E_new) and g_new_free < gfree:
                    u = u_new
                    moved = True
                    break
                t *= 0.5
            if not moved:
                break
            st, E, g, locked = _eval_timed(u)
        st, E, g, locked = _eval_timed(u)
    t_phase["polish"] = _clock() - t0
    kkt = kkt_of(u, g)

    if kkt > 1e-6:
        raise CEPSFailure(
            f"did not reach a KKT point (KKT residual {kkt:.3e}, free gradient "
            f"{gmax_free:.3e} after {nit} Newton steps); no result is returned")

    res = Result()
    res.u = u
    res.pinned = pin
    res.free = free
    res.boundary = bd
    res.delta = delta
    res.delta_info = dinfo
    res.energy = E
    res.grad = g
    res.iters = nit
    res.flips = st.flips
    res.kkt = kkt
    res.locked = list(locked)
    res.realizable = realizable(st, u)
    res.active = [v for k, v in enumerate(vs)
                  if np.isfinite(lo[k]) and abs(u[v] - lo[k]) <= 1e-9]
    res.theta = vertex_angles(u, st, vs)
    res.state = st
    res.vs = vs
    res.log = log
    res.polish_log = polish_log
    res.stuck = list(st.stuck)
    res.margin = margin
    res.timing = t_phase
    res.n_eval = n_eval[0]
    return res


def _kkt(u, g, lo, vs):
    kkt = 0.0
    for k, v in enumerate(vs):
        if not np.isfinite(lo[k]):
            kkt = max(kkt, abs(g[v]))
            continue
        if u[v] <= lo[k] + 1e-9:
            kkt = max(kkt, -g[v])
        else:
            kkt = max(kkt, abs(g[v]))
    return float(kkt)


def vertex_angles(u, st, vs):
    th = {v: 0.0 for v in vs}
    for f in st.faces:
        a, b, c = f
        lab = st.len_t(canonical_edge(a, b), u)
        lbc = st.len_t(canonical_edge(b, c), u)
        lca = st.len_t(canonical_edge(c, a), u)
        aa, ab, ac = tri_angles(lbc, lca, lab)
        th[a] += aa
        th[b] += ab
        th[c] += ac
    return th


# --------------------------------------------------------------------------
# layout, reinsertion, gauge, certificate
# --------------------------------------------------------------------------
def flat_layout(u, st, vs):
    """Develop the flat disk in R^2 triangle by triangle.

    Each new triangle is placed on the *opposite* side of the shared edge from
    the apex of the already placed triangle, which is the unique consistent
    development of a flat triangulated region (no orientation bookkeeping of
    the abstract faces is needed, and no embedding choice is left free).
    """
    faces = st.faces
    pos = {}
    placed = [False] * len(faces)

    def len_of(i, j):
        return st.len_t(canonical_edge(i, j), u)

    def third(fi, e):
        for v in faces[fi]:
            if v not in e:
                return v
        raise KeyError

    f0 = faces[0]
    a, b, c = f0
    lab, lbc, lca = len_of(a, b), len_of(b, c), len_of(c, a)
    pos[a] = np.array([0.0, 0.0])
    pos[b] = np.array([lab, 0.0])
    ca = (lab * lab + lca * lca - lbc * lbc) / (2.0 * lab)
    # North-pole stereographic reinsertion reverses the orientation of the
    # flat coordinate chart relative to an outward-oriented sphere.  Develop
    # the first (oriented) face clockwise so the complete sphere has the input
    # orientation, not its mirror.  Distances and the intrinsic energy are
    # unaffected; subsequent faces are placed on opposite sides as before.
    sa = -math.sqrt(max(0.0, lca * lca - ca * ca))
    pos[c] = np.array([ca, sa])
    placed[0] = True
    frontier = [0]
    while frontier:
        nxt = []
        for fi in frontier:
            verts = list(faces[fi])
            for kk in range(3):
                i, j = verts[kk], verts[(kk + 1) % 3]
                e = canonical_edge(i, j)
                for (fj, apex) in st.disk.adj[e]:
                    if placed[fj]:
                        continue
                    w = third(fi, e)          # apex of the placed face
                    l_ij, l_ia, l_ja = len_of(i, j), len_of(i, apex), len_of(j, apex)
                    cos_i = (l_ij * l_ij + l_ia * l_ia - l_ja * l_ja) / (2.0 * l_ij * l_ia)
                    ang = math.acos(max(-1.0, min(1.0, cos_i)))
                    d = pos[j] - pos[i]
                    d = d / np.linalg.norm(d)
                    nl = np.array([-d[1], d[0]])
                    q1 = pos[i] + l_ia * (math.cos(ang) * d + math.sin(ang) * nl)
                    q2 = pos[i] + l_ia * (math.cos(ang) * d - math.sin(ang) * nl)
                    def cross2(p, q):
                        return float(p[0] * q[1] - p[1] * q[0])
                    s1 = cross2(d, q1 - pos[i])
                    sw = cross2(d, pos[w] - pos[i])
                    pos[apex] = q1 if s1 * sw < 0 else q2
                    placed[fj] = True
                    nxt.append(fj)
        frontier = nxt
    if not all(placed):
        raise CEPSFailure("layout did not reach every face")
    return {v: pos[v].copy() for v in vs}


def mobius_normalise(pts):
    """Explicit gauge: planar barycentre to 0 and RMS radius to 1."""
    keys = sorted(pts)
    P = np.array([pts[k] for k in keys])
    c = P.mean(axis=0)
    P = P - c
    s = math.sqrt(float(np.mean(np.sum(P * P, axis=1))))
    if s <= 0.0:
        raise CEPSFailure("degenerate layout")
    P /= s
    return {k: P[i] for i, k in enumerate(keys)}


def stereo_reinsert(pts, v_inf, n):
    """Inverse stereographic projection; v_inf re-inserted at the pole."""
    X = np.zeros((n, 3))
    for k, p in pts.items():
        x, y = float(p[0]), float(p[1])
        r2 = x * x + y * y
        X[k] = (2.0 * x / (1.0 + r2), 2.0 * y / (1.0 + r2), (r2 - 1.0) / (1.0 + r2))
    X[v_inf] = np.array([0.0, 0.0, 1.0])
    return X


def oriented_closed_surface(faces):
    """True iff every directed edge of ``faces`` occurs exactly once."""
    seen = {}
    for f in faces:
        a, b, c = f
        for (i, j) in ((a, b), (b, c), (c, a)):
            key = (i, j)
            if key in seen:
                return False
            seen[key] = True
    for (i, j) in seen:
        if (j, i) not in seen:
            return False
    return True


def convexity_certificate(X, faces_final, ell_tilde=None, n_expected=None,
                          extra_vertices=()):
    """Distinctness, sphere containment, supporting-plane test per face.

    ``ell_tilde`` (optional) maps every edge of ``faces_final`` to the scaled
    length ``exp((lambda + u_i + u_j)/2)`` of the flat chart; when it is given
    the certificate also reports the *intrinsic scaling consistency*
    ``max_e |log |p_i - p_j| - log ell_tilde_e - c|`` of the laid-out disk,
    which is the independent check that the layout really is the conformal
    flattening of the solved metric and not merely a set of KKT numbers.
    """
    n = X.shape[0]
    norms = np.linalg.norm(X, axis=1)
    sphere_err = float(np.max(np.abs(norms - 1.0)))
    d2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    mind = float(np.sqrt(d2.min()))
    worst = math.inf
    orient = 0
    for f in faces_final:
        a, b, c = f
        nrm = np.cross(X[b] - X[a], X[c] - X[a])
        ln = np.linalg.norm(nrm)
        if ln == 0.0:
            worst = -math.inf
            continue
        nrm = nrm / ln
        if float(nrm @ X[a]) < 0:
            nrm = -nrm
            orient += 1
        vals = X @ nrm
        others = [k for k in range(n) if k not in (a, b, c)]
        if others:
            margin = float(vals[a] - np.max(vals[others]))
            worst = min(worst, margin)
    used = sorted({v for f in faces_final for v in f} | set(extra_vertices))
    n_exp = n if n_expected is None else n_expected
    out = {"sphere_err": sphere_err, "min_pair_distance": mind,
           "worst_support_margin": worst, "flipped_faces": orient,
           "orientation_consistent": orient == 0,
           "oriented_closed_surface": oriented_closed_surface(faces_final),
           "n_vertices_used": len(used),
           "all_labelled": used == list(range(n_exp))}
    if ell_tilde is not None:
        rs = []
        for f in faces_final:
            a, b, c = f
            for (i, j) in ((a, b), (b, c), (c, a)):
                e = canonical_edge(i, j)
                if e in ell_tilde:
                    rs.append(math.log(float(np.linalg.norm(X[i] - X[j])))
                              - math.log(ell_tilde[e]))
        if rs:
            out["scaling_residual"] = float(np.max(np.abs(
                np.array(rs) - np.mean(rs))))
    return out


def run(faces, ell, v_inf, **kw):
    """Solve + flat layout + stereographic reinsertion + certificate."""
    t_layout = _clock()
    res = solve(faces, ell, v_inf, **kw)
    st, u, vs = res.state, res.u, res.vs
    flat = flat_layout(u, st, vs)
    flat = mobius_normalise(flat)
    n = max(max(f) for f in faces) + 1
    X = stereo_reinsert(flat, v_inf, n)
    disk_faces = [tuple(f) for f in st.faces]
    ell_tilde = {e: st.len_t(e, u) for e in st.es}
    # re-inserting the special vertex means re-inserting the faces of its star
    # too: one face (v_inf, j1, j2) for each boundary edge j1-j2 of the final
    # disk chart.  Without them the output is a punctured polyhedron and no
    # orientation/closedness certificate can be stated.
    bd_edges = sorted(st.disk.boundary_edges())
    star_faces = [(v_inf, j1, j2) for (j1, j2) in bd_edges]
    faces_final = float_and_fix_orientation(X, disk_faces + star_faces)
    cert = convexity_certificate(X, faces_final, ell_tilde=None,
                                 n_expected=n, extra_vertices=(v_inf,))
    # the flat disk (before reinsertion) is the conformal flattening, so its
    # chord lengths are proportional to the solved scaled lengths.
    rs = [math.log(max(1e-300, float(np.linalg.norm(flat[e[0]] - flat[e[1]]))))
          - math.log(ell_tilde[e]) for e in st.es]
    cert["flat_scaling_residual"] = (float(np.max(np.abs(
        np.array(rs) - np.mean(rs)))) if rs else 0.0)
    res.positions = X
    res.faces_final = faces_final
    res.disk_faces = disk_faces
    res.star_faces = star_faces
    res.flat = flat
    res.ell_tilde = ell_tilde
    res.certificate = cert
    res.timing["layout"] = _clock() - t_layout - sum(
        res.timing.get(k, 0.0) for k in ("delta", "newton", "polish"))
    return res


# --------------------------------------------------------------------------
# public helpers for tests and for the lift driver
# --------------------------------------------------------------------------
def chordal_lengths(faces, X):
    """Input chordal lengths of a configuration on S^2."""
    ed = sphere_edges(faces)
    return {e: float(np.linalg.norm(np.asarray(X)[e[0]] - np.asarray(X)[e[1]]))
            for e in ed}


def energy_at(faces, ell, v_inf, u):
    """E(u) on the (deterministically) retriangulated punctured disk."""
    ed = sphere_edges(faces)
    lam = {e: 2.0 * math.log(ell[e]) for e in ed}
    st = disk_state(faces, lam, v_inf)
    vs = st.vs
    st.retriangulate(u)
    E, g = energy_grad(u, st, vs)
    return E, g, st


def float_and_fix_orientation(X, faces):
    """Consistent outward orientation of hull faces for a configuration on S^2."""
    X = np.asarray(X, float)
    out = []
    for f in faces:
        a, b, c = f
        nrm = np.cross(X[b] - X[a], X[c] - X[a])
        out.append((a, b, c) if float(nrm @ X[a]) > 0 else (a, c, b))
    return out
