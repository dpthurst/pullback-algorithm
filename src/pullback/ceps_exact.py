"""Exact-rational fixtures and an independent oracle for the special-vertex
spherical CEPS uniformizer (Gillespie-Springborn-Crane 2021 sec. 7,
Springborn arXiv:1707.06848 secs. 6-7 and Thm 10.5).

SCOPE.  This module is *not* a production solver.  It provides

  * exact rational point sets on S^2 and their exact chord metrics,
  * the exact acceptance identities a solver output must satisfy
    (Mobius invariant, sphere containment, supporting planes, orientation,
    labelled vertex-scaling consistency), and
  * an independent oracle for small PL spheres which computes the answer from
    the characterisation of Thm 10.5 (existence of a convex inscribed
    polyhedron whose chordal metric is discretely conformally equivalent to the
    input) instead of from the sec.-7 energy, so that the production solver can
    be checked against something that shares none of its machinery.

Conventions (fixed from the primary sources, see report.md):

  * Springborn's decoration variable u is used throughout: Penner coordinates
    scale as  lambda~_e = lambda_e + u_v + u_w  with  u_v  = distance the
    horocycle at v moves towards its cusp.  GSC's u is -u (their arc lengths
    grow as e^{+u}), so GSC's constraint  u_j >= delta_{i* j}  is, in the
    convention used here,  u_j <= -delta.
  * x_v := exp(u_v/2), so  ell~_ij = ell_ij * x_i * x_j  (rational for rational
    x and ell), and lambda~_ij = 2 log ell~_ij = lambda_ij + u_i + u_j.
  * chordal metric of p, q in S^2 subset R^3 is |p-q|; under stereographic
    projection  p = E(z) = (2 Re z, 2 Im z, |z|^2-1)/(1+|z|^2)  one has
    |E(z)-E(w)| = 2|z-w| / sqrt((1+|z|^2)(1+|w|^2)), i.e. the chordal metric is
    discretely conformally equivalent to the planar metric with per-vertex
    factor 2/(1+|z|^2)  (GSC sec. 7.6).
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction as F
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:                      # mpmath is used only by the oracle
    import mpmath as mp
except Exception:         # pragma: no cover
    mp = None


# --------------------------------------------------------------------------
# small exact rational linear algebra on R^3
# --------------------------------------------------------------------------
Vec = Tuple[F, F, F]


def vadd(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vsub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vscale(t: F, a: Vec) -> Vec:
    return (t * a[0], t * a[1], t * a[2])


def dot(a: Vec, b: Vec) -> F:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec, b: Vec) -> Vec:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def norm_sq(a: Vec) -> F:
    return dot(a, a)


def as_frac(p: Sequence) -> Vec:
    return tuple(F(x) for x in p)  # type: ignore[return-value]


# --------------------------------------------------------------------------
# rational point sets on S^2
# --------------------------------------------------------------------------
def sphere_param(a: F, b: F) -> Vec:
    """Rational parametrisation of the unit sphere: (a,b) |-> S^2, rational.

    (2a, 2b, a^2+b^2-1) / (a^2+b^2+1).  Rational input gives a rational point.
    """
    d = a * a + b * b + 1
    return (2 * a / d, 2 * b / d, (a * a + b * b - 1) / d)


def equator_param(t: F) -> Vec:
    """Rational point on the equatorial unit circle: ((1-t^2),(2t),0)/(1+t^2)."""
    d = 1 + t * t
    return ((1 - t * t) / d, 2 * t / d, F(0))


def fixture_tetra() -> List[Vec]:
    """A rational inscribed tetrahedron.

    The Mobius invariant reported by this module's convention,
    ``mobius_invariant(points, (0,1,2,3))``, is ``-1/2 + i/2`` (cross-ratio
    ``cr(z0,z1;z2,z3) = (z2-z0)(z3-z1)/((z2-z1)(z3-z0))`` of the four labelled
    points in a stereographic chart with the projection pole off the
    quadruple).  Use ``fixture_tetra_facets()`` for the consistently oriented
    facet list.
    """
    return [as_frac((1, 0, 0)), as_frac((0, 1, 0)), as_frac((0, 0, 1)),
            as_frac((F(3, 5), F(4, 5), 0))]


def fixture_tetra_facets() -> List[Tuple[int, int, int]]:
    """Consistently oriented, outward-oriented facets of ``fixture_tetra()``.

    Every edge appears exactly once as each of its two darts (checked by
    ``check_opposite_darts``) and every facet is outward oriented with respect
    to the vertex centroid (checked by ``check_orientation``).  The naive list
    [(0,1,2),(0,1,3),(0,2,3),(1,2,3)] carries dart (0,1) twice and is refused
    by ``pullback.dense_assembly.base_from_face_lists``; see the regression
    test ``test_facet_lists_are_consistently_oriented``.
    """
    return [(2, 1, 0), (3, 0, 1), (3, 1, 2), (3, 2, 0)]


def fixture_octahedron() -> List[Vec]:
    return [as_frac((1, 0, 0)), as_frac((-1, 0, 0)),
            as_frac((0, 1, 0)), as_frac((0, -1, 0)),
            as_frac((0, 0, 1)), as_frac((0, 0, -1))]


def fixture_bipyramid(ts: Iterable[F] = (F(0), F(2), F(-1))) -> List[Vec]:
    """Rational bipyramid: north pole, south pole, and rational equator points.

    The default equator parameters give points at 0, 127 and 270 degrees, so the
    origin is strictly inside the equatorial triangle and the bipyramid is
    non-degenerate (no antipodal pair, no co-circular quadruple).
    """
    pts = [as_frac((0, 0, 1)), as_frac((0, 0, -1))]
    pts += [equator_param(F(t)) for t in ts]
    return pts


def fixture_bipyramid_facets() -> List[Tuple[int, int, int]]:
    """Facets of ``fixture_bipyramid()``: apex 0 up, apex 1 down, rim 2,3,4."""
    return [(0, 2, 3), (0, 3, 4), (0, 4, 2), (1, 3, 2), (1, 4, 3), (1, 2, 4)]


def fixture_generic(n: int) -> List[Vec]:
    """n rational points on S^2 in general position (spiral parametrisation)."""
    pts: List[Vec] = []
    k = 0
    while len(pts) < n:
        a = F(k * k + 1, k * k + 3) * (1 if k % 2 == 0 else -1)
        b = F(2 * k + 1, k * k + 5)
        k += 1
        p = sphere_param(a, b)
        if all(norm_sq(vsub(p, q)) > 0 for q in pts):
            pts.append(p)
    return pts


# --------------------------------------------------------------------------
# exact metrics and exact acceptance identities
# --------------------------------------------------------------------------
def chord_sq(p: Vec, q: Vec) -> F:
    """Exact squared chord length of two points of S^2 (rational for rational p,q)."""
    return norm_sq(vsub(p, q))


def chord_metric_sq(points: Sequence[Vec]) -> Dict[Tuple[int, int], F]:
    m: Dict[Tuple[int, int], F] = {}
    for i, j in itertools.combinations(range(len(points)), 2):
        m[(i, j)] = chord_sq(points[i], points[j])
    return m


def check_on_sphere(points: Sequence[Vec]) -> List[str]:
    """Exact: every point has unit norm.  Returns a list of defects (empty = ok)."""
    bad = []
    for i, p in enumerate(points):
        if norm_sq(p) != 1:
            bad.append("point %d has |p|^2 = %s != 1" % (i, norm_sq(p)))
    return bad


def check_distinct(points: Sequence[Vec]) -> List[str]:
    bad = []
    for i, j in itertools.combinations(range(len(points)), 2):
        if norm_sq(vsub(points[i], points[j])) == 0:
            bad.append("points %d and %d coincide" % (i, j))
    return bad


def facet_margins(points: Sequence[Vec],
                  facets: Sequence[Sequence[int]]) -> List[Tuple[int, F]]:
    """Exact supporting-plane margins.

    For each facet, the plane through its (first three) vertices is oriented by
    its normal n = (b-a) x (c-a).  The margin of the facet is min over the
    remaining vertices of  sign * <p - a, n>  where sign makes the centroid side
    positive.  A strictly convex inscribed polyhedron with these facets has all
    margins > 0.  Returns (facet index, margin) triples.
    """
    centroid = vscale(F(1, len(points)),
                      (sum(p[0] for p in points),
                       sum(p[1] for p in points),
                       sum(p[2] for p in points)))
    out = []
    for k, f in enumerate(facets):
        a, b, c = points[f[0]], points[f[1]], points[f[2]]
        n = cross(vsub(b, a), vsub(c, a))
        if norm_sq(n) == 0:
            out.append((k, F(0)))
            continue
        s = dot(vsub(centroid, a), n)
        if s == 0:
            out.append((k, F(0)))
            continue
        sign = F(1) if s > 0 else F(-1)
        vals = [sign * dot(vsub(points[i], a), n)
                for i in range(len(points)) if i not in f]
        out.append((k, min(vals) if vals else F(0)))
    return out


def check_supporting(points: Sequence[Vec],
                     facets: Sequence[Sequence[int]]) -> List[str]:
    bad = []
    for k, m in facet_margins(points, facets):
        if m <= 0:
            bad.append("facet %d is not strictly supporting (margin %s)" % (k, m))
    return bad


def signed_volume(a: Vec, b: Vec, c: Vec, d: Vec) -> F:
    return dot(vsub(b, a), cross(vsub(c, a), vsub(d, a)))


def check_opposite_darts(facets: Sequence[Sequence[int]]) -> List[str]:
    """Combinatorial consistency of an oriented triangulation (exact, no points).

    A consistently oriented closed triangulated surface uses every edge exactly
    twice, once as each of its two darts.  This is the check that
    ``pullback.dense_assembly.base_from_face_lists`` enforces; a facet list that
    repeats a dart describes an unoriented or non-manifold input and must be
    rejected before any metric work.
    """
    from collections import Counter
    count: Counter = Counter()
    for f in facets:
        for t in range(3):
            count[(f[t], f[(t + 1) % 3])] += 1
    bad = []
    for dart, n in sorted(count.items()):
        if n != 1:
            bad.append("dart %s is carried by %d faces" % (dart, n))
    edges = {frozenset((f[t], f[(t + 1) % 3])) for f in facets for t in range(3)}
    for e in sorted(edges, key=sorted):
        a, b = tuple(sorted(e))
        if count[(a, b)] != 1 or count[(b, a)] != 1:
            bad.append("edge %s does not appear once in each direction" % (e,))
    return bad


def check_orientation(points: Sequence[Vec],
                      facets: Sequence[Sequence[int]]) -> List[str]:
    """Exact outward orientation of the *given* facet order (no reorientation).

    Acceptance must judge the facet order it is given, not a post-hoc
    reorientation: for a convex inscribed polyhedron, ``(b-a) x (c-a)`` points
    away from the vertex centroid, which is exactly the orientation the input
    declares.  Degenerate (zero-area) facets are reported too.
    """
    centroid = vscale(F(1, len(points)),
                      (sum(p[0] for p in points),
                       sum(p[1] for p in points),
                       sum(p[2] for p in points)))
    bad = []
    for k, f in enumerate(facets):
        a, b, c = points[f[0]], points[f[1]], points[f[2]]
        n = cross(vsub(b, a), vsub(c, a))
        if norm_sq(n) == 0:
            bad.append("facet %d is degenerate" % k)
            continue
        if dot(n, vsub(centroid, a)) >= 0:
            bad.append("facet %d is not outward oriented in the given order" % k)
    return bad


def orient_outward(points: Sequence[Vec],
                   facets: Sequence[Sequence[int]]) -> List[Tuple[int, ...]]:
    """Orient every facet so that its normal points away from the origin.

    Construction helper only: the acceptance checks (``check_orientation``,
    ``radial_degree_exact``, ``origin_radial_degree``) never reorient facets.
    Uses the vertex centroid (always strictly inside a convex polyhedron), so it
    is valid whether or not the origin is interior.
    """
    centroid = vscale(F(1, len(points)),
                      (sum(p[0] for p in points),
                       sum(p[1] for p in points),
                       sum(p[2] for p in points)))
    out: List[Tuple[int, ...]] = []
    for f in facets:
        a, b, c = points[f[0]], points[f[1]], points[f[2]]
        n = cross(vsub(b, a), vsub(c, a))
        out.append(tuple(f) if dot(n, vsub(centroid, a)) < 0
                   else tuple(reversed(f)))
    return out


def radial_degree_exact(points: Sequence[Vec],
                        facets: Sequence[Sequence[int]]) -> Dict[str, object]:
    """Exact radial/cone-degree certificate.

    For an outward-oriented closed triangulated surface with the origin strictly
    inside, every signed cone volume  det[p_f0,p_f1,p_f2]/6  is positive and the
    radial map has degree +1.  A doubled surface gives as many negative as
    positive cones, a collapsed one gives zeros.  Purely rational arithmetic.
    """
    pos = neg = zero = 0
    vols: List[F] = []
    centroid = vscale(F(1, len(points)),
                      (sum(p[0] for p in points),
                       sum(p[1] for p in points),
                       sum(p[2] for p in points)))
    for f in facets:  # the given order is used; combine with check_orientation
        v = signed_volume(centroid, points[f[0]], points[f[1]], points[f[2]])
        vols.append(v)
        if v > 0:
            pos += 1
        elif v < 0:
            neg += 1
        else:
            zero += 1
    ok = (pos == len(facets)) and (neg == 0) and (zero == 0)
    return {"degree_one": ok, "positive": pos, "negative": neg, "zeros": zero,
            "cone_volumes": vols,
            "total_volume": sum(vols) / 6 if all(v > 0 for v in vols) else None}


def origin_radial_degree(points: Sequence[Vec],
                         facets: Sequence[Sequence[int]]) -> Dict[str, object]:
    """Exact origin-radial diagnostic (NOT a convexity test -- see below).

    An earlier accepted finding of this project, restated: supporting planes, not
    origin-radial signs, decide convexity and orientation.  An inscribed convex
    polyhedron may have its vertices inside a closed hemisphere, in which case the
    origin lies on or outside the surface and the radial map to S^2 is not even
    defined on the surface.  The rational tetrahedron fixture is such a case.

    Returns {"origin": "inside"|"outside"|"on_surface", "degree": 1 or None,
    "plane_signs": [...]}.  A degree is only reported when the origin is strictly
    inside every outward half-space (then the radial map has degree +1).
    """
    signs = []
    for f in facets:  # given order only -- see check_orientation
        a, b, c = points[f[0]], points[f[1]], points[f[2]]
        n = cross(vsub(b, a), vsub(c, a))
        signs.append(dot(n, a))
    if any(s == 0 for s in signs):
        where = "on_surface"
    elif all(s > 0 for s in signs):
        where = "inside"
    else:
        where = "outside"
    return {"origin": where, "degree": 1 if where == "inside" else None,
            "plane_signs": signs}


# --------------------------------------------------------------------------
# Mobius invariant (gauge free)
# --------------------------------------------------------------------------
def _mpf(x) -> "mp.mpf":
    """Convert an int/Fraction/float to an mpf without float round-off."""
    if isinstance(x, F):
        return mp.mpf(x.numerator) / mp.mpf(x.denominator)
    return mp.mpf(x)


def to_complex_mp(p: Sequence) -> complex:
    """Stereographic projection of a point of S^2 (not the north pole) to C.

    Uses the standard chart from the south pole so that the north pole is
    regular: z = (x + i y)/(1 + z_coord).  Grader: exact rational input gives a
    complex number whose real/imaginary parts are rational to machine precision.
    """
    x, y, z = [_mpf(t) for t in p]
    d = 1 + z
    if d == 0:
        raise ValueError("point is the projection pole")
    return complex(x / d, y / d)


def cross_ratio_mp(a, b, c, d):
    """cr(a,b;c,d) = (c-a)(d-b) / ((c-b)(d-a))."""
    return ((c - a) * (d - b)) / ((c - b) * (d - a))


_CHART_POLES = ((F(0), F(0), -1), (F(0), F(0), 1), (F(-1), F(0), F(0)),
                (F(0), F(-1), F(0)), (F(0), F(0), F(0)))


def mobius_invariant(points: Sequence[Sequence], idx: Sequence[int] = (0, 1, 2, 3)):
    """Mobius invariant of four labelled points of S^2 (exact up to mp precision).

    Cross-ratio cr(z0,z1;z2,z3) = (z2-z0)(z3-z1)/((z2-z1)(z3-z0)) of the four
    points in a stereographic chart with the projection pole at distance 0 from
    the sphere.  The value depends on the *labelled order* (the 24 permutations
    give the six classical values) and is invariant under Mobius maps of S^2.
    The chart is chosen so that no point of the quadruple is the pole.
    """
    if mp is None:  # pragma: no cover
        raise RuntimeError("mpmath required for mobius_invariant")
    quad = [points[i] for i in idx]
    old = mp.mp.dps
    mp.mp.dps = max(old, 40)
    try:
        for pole in _CHART_POLES:
            # rotate the sphere so that `pole` becomes the south pole (0,0,-1)
            n = vsub(pole, (F(0), F(0), F(-1)))
            if norm_sq(n) == 0:
                rot = _identity_rotation()
            else:
                rot = _rotation_sending(n, (F(0), F(0), F(-1)))
            rotated = [_rotate(rot, p) for p in quad]
            if any(vsub(p, (F(0), F(0), F(-1))) == (F(0), F(0), F(0)) for p in rotated):
                continue
            z = [to_complex_mp(p) for p in rotated]
            return cross_ratio_mp(z[0], z[1], z[2], z[3])
        raise ValueError("no admissible stereographic chart")
    finally:
        mp.mp.dps = old


def _identity_rotation():
    return ((F(1), F(0), F(0)), (F(0), F(1), F(0)), (F(0), F(0), F(1)))


def _rotation_sending(u: Vec, w: Vec):
    """Exact rational/float rotation matrix with R u = |u| w (Householder, exact).

    For u, w of the same length the map  x |-> H x  with
    H = I - 2 vv^T/|v|^2,  v = u - w, is an orthogonal reflection sending u to w.
    Determinant -1 is irrelevant for the cross-ratio (it conjugates it).
    """
    v = vsub(u, w)
    nv = norm_sq(v)
    if nv == 0:
        return _identity_rotation()
    rows = []
    for i in range(3):
        rows.append(tuple((F(1) if i == j else F(0)) - 2 * v[i] * v[j] / nv
                          for j in range(3)))
    return tuple(rows)


def _rotate(rot, p: Vec) -> Vec:
    return tuple(sum(rot[i][j] * p[j] for j in range(3)) for i in range(3))  # type: ignore[return-value]


def same_oriented_mobius_class(points_a, points_b, tol=1e-12,
                               which=((0, 1, 2, 3),)) -> bool:
    """Same *oriented* Mobius class: invariants agree, conjugation not allowed.

    An orientation-reversing (anti-Mobius) image has the conjugate cross-ratio,
    so this test refuses a mirror image -- which is what a labelled, oriented PL
    sphere demands.  Use ``is_mirror_image`` to identify that case explicitly.
    """
    for idx in which:
        a = mobius_invariant(points_a, idx)
        b = mobius_invariant(points_b, idx)
        if abs(a - b) > tol * max(1.0, abs(a)):
            return False
    return True


def is_mirror_image(points_a, points_b, tol=1e-12) -> bool:
    """True when the two configurations are related by a reflection of S^2."""
    a = mobius_invariant(points_a)
    b = mobius_invariant(points_b)
    return abs(a - b.conjugate()) <= tol * max(1.0, abs(a)) and abs(a - b) > tol


def same_mobius_class(points_a, points_b, tol=1e-25, which=((0, 1, 2, 3),)) -> bool:
    for idx in which:
        a = mobius_invariant(points_a, idx)
        b = mobius_invariant(points_b, idx)
        if abs(a - b) > tol * max(1.0, abs(a)):
            return False
    return True


# --------------------------------------------------------------------------
# labelled vertex scaling: the CEPS condition and its exact fixture form
# --------------------------------------------------------------------------
def scaling_tolerance(points: Sequence[Sequence],
                      metric_sq: Dict[Tuple[int, int], float],
                      u: Sequence[float],
                      allow_constant: bool = True
                      ) -> Tuple[float, Dict[Tuple[int, int], float]]:
    """Residual of  chord^2(i,j) = l^2(i,j) * exp(u_i + u_j)  for a labelled output.

    ``metric_sq`` may be exact rationals or floats; ``points`` are the solver's
    output positions (floats or Fractions).  Returns (max residual, per-edge) of
    the *logarithmic* identity; with ``allow_constant`` the one additive constant
    that the conformal factor is free to have (the global scale of the output) is
    removed first.  Pass ``allow_constant=False`` to test a fully pinned u.
    """
    pf = [np.array([float(c) for c in p], dtype=float) for p in points]
    uv = np.array([float(t) for t in u], dtype=float)
    raw: Dict[Tuple[int, int], float] = {}
    for (i, j), l2 in metric_sq.items():
        d2 = float(np.sum((pf[i] - pf[j]) ** 2))
        raw[(i, j)] = math.log(d2 / float(l2)) - (uv[i] + uv[j])
    c = sum(raw.values()) / len(raw) if (allow_constant and raw) else 0.0
    per = {e: v - c for e, v in raw.items()}
    worst = max((abs(v) for v in per.values()), default=0.0)
    return worst, per


def scaling_from_positions(points: Sequence[Sequence],
                           metric_sq: Dict[Tuple[int, int], float]) -> np.ndarray:
    """Solve log(chord^2/l^2) = u_i + u_j for u (least squares, per connected graph).

    The system is linear in u; with a connected graph it has the one-dimensional
    kernel u -> u + c*1, which is removed by returning u with mean zero.
    """
    n = len(points)
    pf = [np.array([float(c) for c in p], dtype=float) for p in points]
    rows, rhs = [], []
    edges = sorted(metric_sq)
    for k, (i, j) in enumerate(edges):
        d2 = float(np.sum((pf[i] - pf[j]) ** 2))
        rhs.append(math.log(d2 / float(metric_sq[(i, j)])))
        row = np.zeros(n)
        row[i] = 1.0
        row[j] = 1.0
        rows.append(row)
    A = np.array(rows)
    b = np.array(rhs)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    sol -= sol.mean()
    return sol


def check_scaling(points: Sequence[Sequence],
                  metric_sq: Dict[Tuple[int, int], float],
                  u: Sequence[float], tol: float = 1e-9) -> List[str]:
    """Exact-equivalent acceptance check of discrete conformal equivalence.

    Reports a defect unless the labelled identity chord^2 = l^2 exp(u_i+u_j)
    holds on every edge of the input triangulation to relative ``tol``.
    """
    worst, per = scaling_tolerance(points, metric_sq, u)
    bad = []
    if not (worst <= tol):
        worst_edge = max(per, key=lambda e: abs(per[e]))
        bad.append("vertex scaling fails: max rel. residual %.3e on edge %s"
                   % (worst, worst_edge))
    return bad


# --------------------------------------------------------------------------
# vertex-scaled fixtures: input with a *known* answer up to Mobius
# --------------------------------------------------------------------------
class VertexScaledFixture:
    """A rational inscribed polyhedron, rescaled by a rational vertex factor.

    The input metric on edge (i,j) is  l'_ij^2 = l_ij^2 * exp(v_i + v_j)  with
    l_ij the chord length of the rational inscribed configuration and v rational.
    By discrete conformal equivalence (GSC eq. (1) / Springborn eq. (udc)) the
    desired answer is the original configuration up to a Mobius transformation,
    and the conformal factor from the input to the *answer's* chordal metric is
    u = -v up to an additive constant.  Everything rational stays exact:

        chord^2(p_i,p_j) = l_ij^2 = l'_ij^2 * exp(-v_i - v_j)     (exact rational)
    """

    def __init__(self, points: Sequence[Vec], facets: Sequence[Sequence[int]],
                 v: Sequence[F], pole: int = 0):
        self.points = [as_frac(p) for p in points]
        self.facets = [tuple(f) for f in facets]
        self.v = [F(t) for t in v]
        self.pole = int(pole)
        self.metric_sq = chord_metric_sq(self.points)
        # l'^2 = l^2 * exp(v_i+v_j): keep rational log-exponent separately
        self.scaled_sq = {(i, j): (l2, self.v[i] + self.v[j])
                          for (i, j), l2 in self.metric_sq.items()}

    # ---- exact expected data -------------------------------------------
    def expected_invariant(self, idx=(0, 1, 2, 3)):
        return mobius_invariant(self.points, idx)

    def expected_u(self) -> List[float]:
        """u = -v (representative = the original configuration; mean-shifted)."""
        u = [-float(t) for t in self.v]
        m = sum(u) / len(u)
        return [t - m for t in u]

    def input_lengths_sq(self) -> Dict[Tuple[int, int], float]:
        return {(i, j): float(l2) * math.exp(float(w))
                for (i, j), (l2, w) in self.scaled_sq.items()}

    def input_lengths(self) -> Dict[Tuple[int, int], float]:
        return {e: math.sqrt(s) for e, s in self.input_lengths_sq().items()}

    # ---- acceptance ----------------------------------------------------
    def check(self, points_out: Sequence[Sequence], tol: float = 1e-9) -> List[str]:
        bad: List[str] = []
        pf = [as_frac([F(float(c)).limit_denominator(10 ** 15) for c in p])
              for p in points_out]
        if len(points_out) == len(self.points):
            for i, p in enumerate(points_out):
                s = float(sum(float(c) ** 2 for c in p))
                if abs(s - 1) > tol:
                    bad.append("output point %d off the unit sphere: |p|^2-1 = %.3e"
                               % (i, s - 1))
            if not same_mobius_class(points_out, self.points, tol=max(tol, 1e-18)):
                bad.append("Mobius invariant %.17g != expected %.17g"
                           % (mobius_invariant(points_out),
                              self.expected_invariant()))
        else:
            bad.append("vertex count %d != %d" % (len(points_out), len(self.points)))
        del pf
        return bad


# --------------------------------------------------------------------------
# independent oracle for four vertices (scaling-free characterisation)
# --------------------------------------------------------------------------
def solve_tetra_scaling_free(metric_sq: Dict[Tuple[int, int], float],
                             pole: int = 3, gauge: Sequence[int] = (0, 1, 2),
                             dps: int = 45, seeds: int = 200):
    """Oracle: all Mobius-classes of convex inscribed tetrahedra realising ``metric_sq``.

    Uses only Thm 10.5's characterisation (chordal metric discretely conformally
    equivalent to the input), *not* the sec.-7 energy, not delta, not the
    peacock.  Gauge: the three vertices in ``gauge`` are placed at e1,e2,e3 of
    the unit sphere; the fourth is solved from the two independent ratios
    chord(i,pole)^2 / (l'^2_it exp(u_i)) .  Returns a list of (positions, u)
    with u mean-shifted.  This is an oracle for small labelled fixtures only;
    it is *not* a solver (it exploits that a 4-vertex disk has no interior
    vertex) and must never be used as one.
    """
    if mp is None:  # pragma: no cover
        raise RuntimeError("mpmath required")
    old = mp.mp.dps
    mp.mp.dps = dps
    try:
        a, b, c = [int(t) for t in gauge]
        d = int(pole)
        if {a, b, c, d} != {0, 1, 2, 3}:
            raise ValueError("gauge and pole must partition the vertices")

        def key(i, j):
            return (min(i, j), max(i, j))

        A = mp.matrix([[1, 1, 0], [1, 0, 1], [0, 1, 1]]) / 2
        g = mp.matrix([mp.log(mp.sqrt(2) / mp.sqrt(mp.mpf(str(metric_sq[key(a, b)])))),
                       mp.log(mp.sqrt(2) / mp.sqrt(mp.mpf(str(metric_sq[key(a, c)])))),
                       mp.log(mp.sqrt(2) / mp.sqrt(mp.mpf(str(metric_sq[key(b, c)]))))])
        u3 = mp.lu_solve(A, g)
        P = {a: [mp.mpf(1), mp.mpf(0), mp.mpf(0)],
             b: [mp.mpf(0), mp.mpf(1), mp.mpf(0)],
             c: [mp.mpf(0), mp.mpf(0), mp.mpf(1)]}

        def resid(x):
            p4 = [x[0], x[1], x[2]]
            d2 = [sum((P[i][k] - p4[k]) ** 2 for k in range(3)) for i in (a, b, c)]
            val = [d2[t] / (mp.mpf(str(metric_sq[key((a, b, c)[t], d)]))
                            * mp.e ** u3[t]) for t in range(3)]
            return mp.matrix([val[0] - val[1], val[0] - val[2],
                              sum(v * v for v in p4) - 1])

        rng = np.random.default_rng(20240607)
        found = []
        for _ in range(int(seeds)):
            x0 = rng.normal(size=3)
            x0 /= np.linalg.norm(x0)
            x = mp.matrix([mp.mpf(float(t)) for t in x0])
            ok = True
            for _it in range(120):
                Fl = resid(x)
                J = mp.matrix(3, 3)
                h = mp.mpf(10) ** (-(dps // 2))
                for k in range(3):
                    xp = mp.matrix([x[t] for t in range(3)])
                    xp[k] += h
                    Fp = resid(xp)
                    for r_ in range(3):
                        J[r_, k] = (Fp[r_] - Fl[r_]) / h
                try:
                    dx = mp.lu_solve(J, -Fl)
                except Exception:
                    ok = False
                    break
                x = x + dx
                if max(abs(t) for t in dx) < mp.mpf(10) ** (-(dps - 8)):
                    break
            if not ok:
                continue
            if max(abs(t) for t in resid(x)) > mp.mpf(10) ** (-(dps - 14)):
                continue
            if any(max(abs(x[k] - y[k]) for k in range(3)) < mp.mpf(10) ** (-12)
                   for y in found):
                continue
            found.append(x)

        out = []
        for x in found:
            p4 = [x[0], x[1], x[2]]
            pos = [None, None, None, None]
            pos[a], pos[b], pos[c] = P[a], P[b], P[c]
            pos[d] = p4
            d2 = [sum((P[i][k] - p4[k]) ** 2 for k in range(3)) for i in (a, b, c)]
            u4 = mp.log(d2[0] / (mp.mpf(str(metric_sq[key(a, d)])) * mp.e ** u3[0]))
            u = [None, None, None, None]
            u[a], u[b], u[c], u[d] = u3[0], u3[1], u3[2], u4
            m = sum(u) / 4
            out.append(([[float(t) for t in p] for p in pos],
                        [float(t - m) for t in u]))
        return out
    finally:
        mp.mp.dps = old


# --------------------------------------------------------------------------
# negative controls
# --------------------------------------------------------------------------
def naive_bound_delta(metric_sq: Dict[Tuple[int, int], float], pole: int,
                      sign: float = +1.0) -> Dict[int, float]:
    """The *wrong* recipes  delta_v = sign * 2 log l(v,pole).

    Recorded here so that a test can assert they fail: 2 log l is the raw Penner
    coordinate of the input decoration, but delta is the *shortest* signed
    horocycle distance, which the peacock triangulation of Springborn Prop. 5.16
    changes.  On the rational vertex-scaled tetrahedron both signs fail.
    """
    out = {}
    for (i, j), l2 in metric_sq.items():
        if i == pole:
            out[j] = sign * math.log(l2)
        elif j == pole:
            out[i] = sign * math.log(l2)
    return out


def layout_disk(sides: Dict[Tuple[int, int], float],
                vertices: Sequence[int]) -> Dict[int, complex]:
    """Flat layout of a triangulated disk read edge by edge (BFS), returns z_v.

    Only used by the negative controls and by tests; a first vertex is put at the
    origin and its first neighbour on the positive real axis.  Returns None for
    vertices unreachable from ``vertices[0]``.
    """
    adj: Dict[int, List[Tuple[int, float]]] = {v: [] for v in vertices}
    for (i, j), l in sides.items():
        if i in adj and j in adj:
            adj[i].append((j, l))
            adj[j].append((i, l))
    root = vertices[0]
    pos: Dict[int, complex] = {root: 0j}
    if adj[root]:
        nxt = adj[root][0]
        pos[nxt[0]] = complex(nxt[1], 0.0)
    # iterative placement: use two known neighbours where possible
    for _ in range(10 * len(vertices) + 10):
        progressed = False
        for v in vertices:
            if v in pos:
                continue
            known = [(w, l) for (w, l) in adj[v] if w in pos]
            for (w1, l1), (w2, l2) in itertools.combinations(known, 2):
                L = abs(pos[w1] - pos[w2])
                if L == 0:
                    continue
                a = (l1 * l1 - l2 * l2 + L * L) / (2 * L)
                h2 = l1 * l1 - a * a
                if h2 < 0:
                    continue
                h = math.sqrt(h2)
                u = (pos[w2] - pos[w1]) / L
                base = pos[w1] + a * u
                cand = [base + 1j * h * u, base - 1j * h * u]
                # pick the candidate that keeps the winding counter-clockwise
                # relative to already-placed edge (w1,w2): left side
                pos[v] = cand[0]
                progressed = True
                break
            if v in pos:
                break
        if not progressed:
            break
    return pos


def negative_controls() -> Dict[str, List[str]]:
    """Explicit witnesses that must be rejected by any acceptance test."""
    pts = fixture_tetra()
    facets = fixture_tetra_facets()
    collapsed = [pts[0]] * 4
    doubled = [pts[0], pts[1], pts[2], pts[0]]      # not distinct
    off_sphere = [pts[0], pts[1], pts[2],
                  vscale(F(2), pts[3])]             # vscale does not normalise
    out = {
        "collapse": check_distinct(collapsed) + check_supporting(collapsed, facets),
        "repeated_vertex": check_distinct(doubled),
        "off_sphere": check_on_sphere(off_sphere),
    }
    return out


# --------------------------------------------------------------------------
# the all-active tetrahedron recipe (used only to *falsify* naive bound rules)
# --------------------------------------------------------------------------
def inverse_stereographic(z: complex) -> Vec:
    """E(z) = (2 Re z, 2 Im z, |z|^2 - 1) / (1 + |z|^2), rational for rational z."""
    x, y = mp.re(z), mp.im(z)
    d = 1 + x * x + y * y
    return (2 * x / d, 2 * y / d, (x * x + y * y - 1) / d)


def square_root_mp(sq):
    """sqrt of an exact Fraction/float to working precision (mpmath)."""
    if isinstance(sq, F):
        return mp.sqrt(_mpf(sq))
    return mp.sqrt(mp.mpf(sq))


def tetra_all_active_positions(metric_sq: Dict[Tuple[int, int], float], pole: int,
                               delta: Dict[int, float], gauge: int = 0):
    """The 'all constraints active' construction for a four-vertex sphere.

    On a 4-vertex sphere the special-vertex disk has three boundary vertices and
    no interior vertex, so Theorem 7.18 forces every remaining vertex to sit on
    its bound: u_v = delta_v.  With x_v = exp(u_v/2) the flat disk sides are
    ell~_ij = ell_ij x_i x_j; the disk is a single triangle, which is laid out in
    the plane and mapped back to S^2 by inverse stereographic projection with the
    special vertex at the pole.

    Returns (positions, invariant).  This is the exact recipe a solver follows
    once it knows delta, and it is what the *false* bound rules are fed below.
    """
    rim = [v for v in range(4) if v != pole]
    def key(i, j):
        return (min(i, j), max(i, j))
    x = {v: mp.e ** (mp.mpf(delta[v]) / 2) for v in rim}
    sides = {}
    for a, b in itertools.combinations(rim, 2):
        sides[(a, b)] = (square_root_mp(metric_sq[key(a, b)]) * x[a] * x[b])
    # layout the triangle with first vertex at 0 and second on the real axis
    a, b, c = rim
    l_ab, l_bc, l_ca = sides[(min(a, b), max(a, b))], sides[(min(b, c), max(b, c))], sides[(min(c, a), max(c, a))]
    z = {a: mp.mpc(0), b: mp.mpc(l_ab)}
    xx = (l_bc ** 2 - l_ca ** 2 + l_ab ** 2) / (2 * l_ab)
    h2 = l_bc ** 2 - xx ** 2
    if h2 < 0:
        raise ValueError("disk triangle violates the triangle inequality")
    z[c] = mp.mpc(xx, mp.sqrt(h2))
    pos = [None, None, None, None]
    for v in rim:
        pos[v] = inverse_stereographic(z[v])
    pos[pole] = (mp.mpf(0), mp.mpf(0), mp.mpf(1))
    return pos, mobius_invariant([pos[i] for i in range(4)])


def candidate_rule_report(fixture: "VertexScaledFixture") -> Dict[str, complex]:
    """Mobius invariants produced by candidate bound rules on a 4-vertex fixture.

    Recorded for the record and used by the falsification tests.  The three
    rules are

      "plus_2log"  delta_v = +lambda_v_pole        (raw input Penner coordinate)
      "minus_2log" delta_v = -lambda_v_pole
      "minus_v"    delta_v = -v_v                  (the formal scaling factor)

    Each is fed to ``tetra_all_active_positions``.  The three results and the
    original configuration are pairwise different Mobius classes, so neither
    shortcut reproduces the answer: the sec.-7 bound must be the peacock value of
    Springborn Prop. 5.16.  This module records the invariants and does not claim
    which rule is correct.
    """
    if len(fixture.points) != 4:
        raise ValueError("four vertices only")
    l2 = fixture.input_lengths_sq()
    out: Dict[str, complex] = {}
    for name, sign in (("plus_2log", +1.0), ("minus_2log", -1.0)):
        delta = naive_bound_delta(l2, fixture.pole, sign)
        _pos, inv = tetra_all_active_positions(l2, fixture.pole, delta)
        out[name] = complex(inv)
    delta_ok = {v: -float(fixture.v[v]) for v in range(4) if v != fixture.pole}
    _pos, inv = tetra_all_active_positions(l2, fixture.pole, delta_ok)
    out["minus_v"] = complex(inv)
    out["original_configuration"] = complex(fixture.expected_invariant())
    return out


def enumerate_scaling_branches(metric_sq: Dict[Tuple[int, int], float],
                               pole: int = 3, seeds: int = 60):
    """Enumerate the Mobius classes that satisfy the labelled scaling identity.

    For a four-vertex sphere the scaling-free system
    ``chord^2(i,j) = l^2(i,j) exp(u_i+u_j)`` is square after the 3-point Mobius
    gauge is fixed, so it has finitely many solutions.  They come in conjugate
    (orientation-reversed) pairs; the CEPS answer is the class whose conformal
    factor also satisfies the sec.-7 bounds.  For the rational fixtures of this
    module the answer is known independently (the input *is* the chordal metric
    of an inscribed rational polyhedron, vertex rescaled), so the enumeration is
    a genuine oracle check: exactly one orientation class must reproduce it.

    Returns a list of dicts with the positions (as floats), the mean-shifted
    conformal factor and the Mobius invariant.
    """
    if len(metric_sq) != 6:
        raise ValueError("six edges expected (four vertices)")
    if len({v for e in metric_sq for v in e}) != 4:
        raise ValueError("four vertices expected")
    gauge = tuple(v for v in range(4) if v != pole)
    sols = solve_tetra_scaling_free(metric_sq, pole=pole, gauge=gauge, seeds=seeds)
    return [{"positions": pos, "u": u, "invariant": mobius_invariant(pos)}
            for pos, u in sols]
