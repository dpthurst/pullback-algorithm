"""Special-vertex spherical CEPS: energy, gradient, Hessian and their exact structure.

Everything here transcribes *explicit published equations* rather than a numerical
recipe.  Sources (local copies under ``refs/`` of the project repo):

* Gillespie, Springborn, Crane, "Discrete Conformal Equivalence of Polyhedral
  Surfaces", ACM TOG 40(4) 103 (2021) -- cited below as GSC.
* B. Springborn, "Ideal hyperbolic polyhedra and discrete uniformization",
  arXiv:1707.06848 -- cited as Spr.

Notation.  ``Delta`` is a triangulation of the closed surface, ``v_inf`` the
special vertex, ``Delta^o`` the *disk* obtained by deleting ``v_inf`` together with
its incident edges and faces (Spr Def. 6.1, eqs. (15)-(17)).  ``lam`` are the input
Penner coordinates ``lam_e = 2 log ell_e`` (Spr eq. (21)), ``u`` the log scale
factors, and the scaled coordinates are

    lam_tilde_e = lam_e + u_i + u_j            (Spr Def. 5.3 / GSC eq. (1))

with ``lam_tilde_e = +inf`` on edges incident with ``v_inf``.  The pair
``(Delta_tilde, lam_tilde) = Del(Delta, lam, u|u_v_inf = +inf)`` is the *adjusted*
Delaunay triangulation (Spr Prop. 5.10).  On the disk ``Delta_tilde^o`` the flat
metric is ``ell_tilde_e = exp(lam_tilde_e/2)`` (Spr eq. (21)), and realizable
coordinates are characterized by (Spr Def. 6.1 (r2b))

    interior vertex:  theta_v = 2 pi        (Spr eq. (18))
    boundary vertex:  theta_v <= pi         (Spr eq. (19))

where theta_v is the angle sum in that flat metric.

The u-dependent part of the energy (Spr Def. 7.16, eq. (56); the decoration
constant -2 pi sum_v log c_v(Delta,lam) is u-independent and dropped -- it affects
no derivative, no minimizer and not the scaling identity):

    E(u) = sum_{t in Delta_tilde^o} 2 f(x_1,x_2,x_3)
           - pi sum_{e in E(Delta_tilde^o)} (u_i + u_j)
           + 2 pi sum_{v in V^o} u_v ,          x_k = lam_tilde_{e_k(t)}/2 ,

with the triangle function (Spr Def. 7.1, eq. (22))

    f(x1,x2,x3) = sum_k alpha_k x_k + sum_k L(alpha_k),
    L(a) = - int_0^a log(2 sin t) dt = Cl_2(2a)/2      (Milnor's Lobachevsky)

and alpha_k the angle opposite the side exp(x_k).

Derivatives (Spr Prop. 7.17 (iii),(iv); eqs. (58),(59)):

    dE/du_v = 2 pi - theta_v + pi (deg2(v) - deg1(v))                  (58)
    D^2 E   = (1/4) sum_{e in E(Delta^o)} w_e (du_{v1} - du_{v2})^2    (59)
    w_e     = cot(alpha_e) + cot(alpha'_e)     (one or two opposite angles)

*** The most error-prone convention in (58) ***: ``deg1``/``deg2`` are the edge- and
triangle-degrees *inside the disk* ``Delta^o`` (GSC §7.2 counts "edges and faces of
T containing j", where their T is the mesh after removing i*).  At a disk boundary
vertex (two boundary edges) the disk degrees differ by exactly one, so the target
angle sum there is pi; at an interior vertex it is 2 pi.  Counting instead in the
*closed* complex shifts every boundary gradient by +pi (target 2 pi rather than pi)
and yields a smooth-looking but wrong "solution" whose disk boundary is not convex.
``gradient`` implements the disk convention, ``gradient_closed_convention`` the
wrong one, so the difference is tested rather than assumed.

Consequences used by the certification layer:

* Summing (58) over V^o gives exactly 2 pi (``gradient_sum_defect``), i.e.
  Prop. 7.17 (v), eq. (60):    E(u + h 1) = E(u) + 2 pi h .
  So E is unbounded below along h -> -inf and the *unconstrained* problem has no
  minimum: the Hessian (59) has the constants in its kernel and 1 is a direction
  of linear decrease, not a gauge direction that may be pinned away.
* The bounds ``u_v >= delta_v`` (Spr eq. (61)) cut that ray off.  At the minimum at
  least one bound is active (an interior optimizer would need dE = 0, which (60)
  forbids), and GSC §7.3 states the active set is exactly the vertices adjacent to
  v_inf, with multiplier grad_v = pi - theta_v >= 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp

# --------------------------------------------------------------------------------
# Milnor's Lobachevsky L(a) = -int_0^a log(2 sin t) dt = Cl_2(2a)/2
# --------------------------------------------------------------------------------
# L is odd and pi-periodic, so L(pi-a) = -L(a); on [0, pi/2]
#     L(a) = -a log(2a) + a + a^3 G(a),   G analytic, G(0) = 1/18.
# G is tabulated once on [0, pi/2] by Chebyshev interpolation.
_GA, _GB = 0.0, 0.5 * np.pi
_GDEG = 40
_GCOEF: np.ndarray | None = None


def lobachevsky_mp(a):
    """Milnor's Lobachevsky function at high precision (mpmath)."""
    from mpmath import clsin, mpf

    return clsin(2, 2 * mpf(a)) / 2


def _build_g_table() -> np.ndarray:
    from mpmath import mp, mpf, log

    mp.dps = 40
    n = _GDEG + 1
    k = np.arange(n)
    nodes = np.cos(np.pi * (k + 0.5) / n)
    a = 0.5 * (_GB - _GA) * nodes + 0.5 * (_GB + _GA)
    vals = np.empty(n)
    for t, ai in enumerate(a):
        ai = mpf(float(ai))
        vals[t] = float((lobachevsky_mp(ai) + ai * log(2 * ai) - ai) / ai**3)
    coef = np.array(
        [(2.0 / n) * float(np.sum(vals * np.cos(np.pi * j * (k + 0.5) / n))) for j in range(n)]
    )
    coef[0] *= 0.5
    return coef


def _g_eval(a: np.ndarray) -> np.ndarray:
    global _GCOEF
    if _GCOEF is None:
        _GCOEF = _build_g_table()
    t = np.clip((2.0 * a - (_GA + _GB)) / (_GB - _GA), -1.0, 1.0)
    n = _GCOEF.size
    b1 = np.zeros_like(t)
    b2 = np.zeros_like(t)
    for j in range(n - 1, 0, -1):
        b0 = 2.0 * t * b1 - b2 + _GCOEF[j]
        b2, b1 = b1, b0
    return t * b1 - b2 + _GCOEF[0]


def lobachevsky(a):
    """Milnor's Lobachevsky function, vectorised; odd and pi-periodic."""
    a = np.asarray(a, float)
    r = np.mod(a, np.pi)
    flip = r > 0.5 * np.pi
    r = np.where(flip, np.pi - r, r)
    pos = r > 1e-300
    rr = np.where(pos, r, 1.0)
    val = np.where(pos, -rr * np.log(2.0 * rr) + rr + rr**3 * _g_eval(rr), 0.0)
    return np.where(flip, -val, val)


def triangle_angles_from_log_sides(x1, x2, x3):
    """Angles opposite the sides exp(x_k) of Euclidean triangles.

    Returns ``(angles, ok)`` with ``angles[..., k]`` opposite ``exp(x_k)`` and
    ``ok`` flagging a strict triangle inequality; uses
    ``tan(alpha_i/2) = sqrt(((s-b)(s-c))/(s(s-a)))``, stable for thin triangles.
    """
    x1, x2, x3 = np.broadcast_arrays(np.asarray(x1, float), np.asarray(x2, float), np.asarray(x3, float))
    return _angles_impl(x1, x2, x3)


def _angles_impl(x1, x2, x3):
    with np.errstate(all="ignore"):
        return _angles_body(x1, x2, x3)


def _angles_body(x1, x2, x3):
    ell = tuple(np.exp(np.clip(x, -700, 700)) for x in (x1, x2, x3))
    ang = np.empty(x1.shape + (3,))
    ok = np.ones(x1.shape, bool)
    for i in range(3):
        a, b, c = ell[i], ell[(i + 1) % 3], ell[(i + 2) % 3]
        s = 0.5 * (a + b + c)
        num = (s - b) * (s - c)
        den = s * (s - a)
        good = (num > 0) & (den > 0) & np.isfinite(num) & np.isfinite(den)
        ratio = np.where(good, num / np.where(den > 0, den, 1.0), 1.0)
        ok = ok & good
        ang[..., i] = 2.0 * np.arctan(np.sqrt(np.clip(ratio, 0.0, None)))
    return ang, ok


# --------------------------------------------------------------------------------
# Disk complex
# --------------------------------------------------------------------------------
class DiskError(ValueError):
    """The supplied faces do not triangulate a closed disk with one boundary cycle."""


@dataclass
class Disk:
    """The triangulated closed disk ``Delta^o = Delta \\ {v_inf}``."""

    n: int
    faces: tuple[tuple[int, int, int], ...]
    edges: tuple[tuple[int, int], ...]
    edge_faces: dict[tuple[int, int], tuple[tuple[int, int], ...]]
    boundary_cycle: tuple[int, ...]
    boundary_edges: frozenset[tuple[int, int]]
    interior_vertices: tuple[int, ...]
    boundary_vertices: tuple[int, ...]
    deg1: np.ndarray
    deg2: np.ndarray
    deg1_closed: np.ndarray
    deg2_closed: np.ndarray
    face_edges: tuple[tuple[tuple[int, int], ...], ...]

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def euler_characteristic(self) -> int:
        return self.n - self.n_edges + self.n_faces

    def interior_edges(self) -> tuple[tuple[int, int], ...]:
        return tuple(e for e in self.edges if len(self.edge_faces[e]) == 2)


def build_disk(n: int, faces: Iterable[Sequence[int]]) -> Disk:
    """Validate and index a triangulated closed disk on vertices 0..n-1."""
    faces = tuple(tuple(int(v) for v in t) for t in faces)
    for t in faces:
        if len(t) != 3 or len(set(t)) != 3:
            raise DiskError(f"face {t} is not a non-degenerate triangle")
        if min(t) < 0 or max(t) >= n:
            raise DiskError(f"face {t} out of range")
    if len(set(faces)) != len(faces):
        raise DiskError("duplicate face")

    edge_faces: dict[tuple[int, int], list[tuple[int, int]]] = {}
    directed: set[tuple[int, int]] = set()
    for fi, t in enumerate(faces):
        for c in range(3):
            i, j = t[(c + 1) % 3], t[(c + 2) % 3]
            edge_faces.setdefault((min(i, j), max(i, j)), []).append((fi, c))
        for c in range(3):
            d = (t[c], t[(c + 1) % 3])
            if d in directed:
                raise DiskError(f"directed edge {d} repeats: faces are not consistently oriented")
            directed.add(d)
    for e, lst in edge_faces.items():
        if len(lst) > 2:
            raise DiskError(f"edge {e} is incident with {len(lst)} faces (not a surface)")
    edges = tuple(sorted(edge_faces))
    boundary_edges = frozenset(e for e, lst in edge_faces.items() if len(lst) == 1)

    deg1 = np.zeros(n, dtype=int)
    deg2 = np.zeros(n, dtype=int)
    for (i, j) in edges:
        deg1[i] += 1
        deg1[j] += 1
    for t in faces:
        for v in t:
            deg2[v] += 1

    bnd_be = np.zeros(n, dtype=int)
    for (i, j) in boundary_edges:
        bnd_be[i] += 1
        bnd_be[j] += 1
    boundary_vertices = tuple(int(v) for v in np.nonzero(bnd_be)[0])
    interior_vertices = tuple(int(v) for v in np.nonzero(bnd_be == 0)[0])
    for v in boundary_vertices:
        if bnd_be[v] != 2:
            raise DiskError(f"boundary vertex {v} has {bnd_be[v]} boundary edges (want 2)")
    if not boundary_vertices:
        raise DiskError("no boundary: this is a closed surface, not a disk")

    nxt: dict[int, list[int]] = {}
    for (i, j) in boundary_edges:
        nxt.setdefault(i, []).append(j)
        nxt.setdefault(j, []).append(i)
    start = min(boundary_vertices)
    cycle = [start]
    prev, cur = None, start
    while True:
        cand = [w for w in nxt[cur] if w != prev]
        if not cand:
            raise DiskError("broken boundary cycle")
        w = cand[0]
        if w == start:
            break
        cycle.append(w)
        prev, cur = cur, w
        if len(cycle) > len(boundary_vertices):
            raise DiskError("boundary walk does not close on itself")
    if len(cycle) != len(boundary_vertices):
        raise DiskError("boundary is disconnected")

    deg1_closed = deg1 + 1
    deg2_closed = deg2.copy()
    for v in boundary_vertices:
        deg2_closed[v] += 2

    d = Disk(
        n=n,
        faces=faces,
        edges=edges,
        edge_faces={e: tuple(lst) for e, lst in edge_faces.items()},
        boundary_cycle=tuple(cycle),
        boundary_edges=boundary_edges,
        interior_vertices=interior_vertices,
        boundary_vertices=boundary_vertices,
        deg1=deg1,
        deg2=deg2,
        deg1_closed=deg1_closed,
        deg2_closed=deg2_closed,
        face_edges=tuple(
            tuple(
                (min(t[(c + 1) % 3], t[(c + 2) % 3]), max(t[(c + 1) % 3], t[(c + 2) % 3]))
                for c in range(3)
            )
            for t in faces
        ),
    )
    if d.euler_characteristic != 1:
        raise DiskError(
            f"Euler characteristic V - E + F = {d.euler_characteristic}, expected 1 for a disk"
        )
    return d


# --------------------------------------------------------------------------------
# Metric quantities
# --------------------------------------------------------------------------------
@dataclass
class Metric:
    """Flat disk metric data and the Eq. (58)/(59) quantities."""

    corner_angles: np.ndarray  # (n_faces, 3): corner c of face fi is at faces[fi][c]
    theta: np.ndarray  # (n,)
    triangle_ok: np.ndarray  # (n_faces,)
    min_angle: float
    min_triangle_slack: float


def _lam_get(lam, e):
    if not isinstance(lam, dict):
        raise TypeError("lam must be a dict keyed by sorted vertex pairs")
    return lam[e]


def scale_lambda(disk: Disk, lam, u) -> dict[tuple[int, int], float]:
    """``lam_tilde_e = lam_e + u_i + u_j`` on the disk (Penner scaling, GSC eq. (1))."""
    u = np.asarray(u, float)
    return {(i, j): _lam_get(lam, (i, j)) + u[i] + u[j] for (i, j) in disk.edges}


def metric(disk: Disk, lam) -> Metric:
    """Angles, angle sums and triangle validity for scaled Penner coordinates."""
    lam = {e: _lam_get(lam, e) for e in disk.edges}
    xs = np.empty((disk.n_faces, 3))
    for fi in range(disk.n_faces):
        for c in range(3):
            i, j = disk.face_edges[fi][c]
            xs[fi, c] = 0.5 * lam[(i, j)]
    ang, ok = triangle_angles_from_log_sides(xs[:, 0], xs[:, 1], xs[:, 2])
    ang = np.asarray(ang, float).reshape(disk.n_faces, 3)
    theta = np.zeros(disk.n)
    for fi, t in enumerate(disk.faces):
        for c in range(3):
            theta[t[c]] += ang[fi, c]
    slack = np.empty(disk.n_faces)
    for fi in range(disk.n_faces):
        ll = np.sort(np.exp(np.clip([0.5 * lam[disk.face_edges[fi][c]] for c in range(3)], -700, 700)))
        slack[fi] = np.log(ll[0] + ll[1]) - np.log(ll[2])
    return Metric(
        corner_angles=ang,
        theta=theta,
        triangle_ok=np.asarray(ok, bool).reshape(disk.n_faces),
        min_angle=float(np.min(ang)),
        min_triangle_slack=float(np.min(slack)),
    )


def gradient(disk: Disk, theta: np.ndarray) -> np.ndarray:
    """Eq. (58) with the degrees taken *in the disk* Delta^o (correct convention)."""
    return 2.0 * np.pi - theta + np.pi * (disk.deg2 - disk.deg1).astype(float)


def gradient_closed_convention(disk: Disk, theta: np.ndarray) -> np.ndarray:
    """Eq. (58) with degrees taken in the *closed* complex: the wrong disk gradient.

    It adds exactly +pi at every boundary vertex.  Kept so a test can reject it.
    """
    return 2.0 * np.pi - theta + np.pi * (disk.deg2_closed - disk.deg1_closed).astype(float)


def gradient_sum_defect(disk: Disk, theta: np.ndarray) -> float:
    """``sum_v dE/du_v`` must equal exactly 2 pi (derivative of Spr eq. (60))."""
    return float(np.sum(gradient(disk, theta)) - 2.0 * np.pi)


def cotan_weights(disk: Disk, m: Metric) -> np.ndarray:
    """``w_e = cot(alpha_e) + cot(alpha'_e)``, the (one or two) angles opposite ``e``."""
    idx = {e: k for k, e in enumerate(disk.edges)}
    w = np.zeros(disk.n_edges)
    for fi in range(disk.n_faces):
        for c in range(3):
            w[idx[disk.face_edges[fi][c]]] += 1.0 / np.tan(m.corner_angles[fi, c])
    return w


def hessian(disk: Disk, m: Metric) -> sp.csr_matrix:
    """The verified second derivative of Eq. (56).

        D^2 E = (1/2) sum_{e in E(Delta^o)} w_e (du_{v1} - du_{v2})^2 ,
        w_e   = cot(alpha_e) + cot(alpha'_e) .

    The printed normalisation in Spr Prop. 7.17 (iv), eq. (59), is ``1/4``.  That
    value is *not* consistent with eq. (56) together with Prop. 7.2 (iii): the
    finite-difference Hessian of Eq. (56) is exactly twice the ``1/4`` form (see
    ``tests/test_ceps_hessian.py::test_hessian_matches_finite_differences`` and the
    factor-2 table in ``report.md``).  Eq. (58) and the scaling identity (60) *are*
    consistent with (56), so the discrepancy is isolated to the printed 1/4.  The
    kernel and the convexity conclusion are unaffected.
    """
    w = cotan_weights(disk, m)
    rows, cols, vals = [], [], []
    for k, (i, j) in enumerate(disk.edges):
        a = 0.5 * w[k]
        rows.extend((i, i, j, j))
        cols.extend((i, j, i, j))
        vals.extend((a, -a, -a, a))
    return sp.coo_matrix((vals, (rows, cols)), shape=(disk.n, disk.n)).tocsr()


def hessian_dense(disk: Disk, m: Metric) -> np.ndarray:
    return np.asarray(hessian(disk, m).todense())


# --------------------------------------------------------------------------------
# Energy: u-dependent part of Spr eq. (56)
# --------------------------------------------------------------------------------
def _face_log_sides(disk: Disk, lam) -> np.ndarray:
    xs = np.empty((disk.n_faces, 3))
    for fi in range(disk.n_faces):
        for c in range(3):
            i, j = disk.face_edges[fi][c]
            xs[fi, c] = 0.5 * _lam_get(lam, (i, j))
    return xs


def triangle_function(x1, x2, x3):
    """Springborn Def. 7.1, eq. (22): ``f(x) = sum a_k x_k + sum L(a_k)``.

    ``a_k`` is the angle opposite the side ``exp(x_k)``.  Returns ``(f, angles, ok)``.
    """
    xs = np.stack(np.broadcast_arrays(np.asarray(x1, float), np.asarray(x2, float), np.asarray(x3, float)), axis=-1)
    ang, ok = triangle_angles_from_log_sides(xs[..., 0], xs[..., 1], xs[..., 2])
    ang = np.asarray(ang, float).reshape(xs.shape)
    f = np.sum(ang * xs, axis=-1) + np.sum(lobachevsky(ang), axis=-1)
    return f, ang, ok


def energy_np(disk: Disk, lam, u: np.ndarray) -> float:
    """Eq. (56), u-dependent part (the decoration constant is dropped)."""
    u = np.asarray(u, float)
    xs = _face_log_sides(disk, scale_lambda(disk, lam, u))
    f, ang, ok = triangle_function(xs[:, 0], xs[:, 1], xs[:, 2])
    if not np.all(ok):
        return float("inf")
    ang = np.asarray(ang, float).reshape(disk.n_faces, 3)
    tot = 2.0 * float(np.sum(f))
    for (i, j) in disk.edges:
        tot -= np.pi * (_lam_get(lam, (i, j)) + u[i] + u[j])
    tot += 2.0 * np.pi * float(np.sum(u))
    return tot


def energy_mp(disk: Disk, lam, u: np.ndarray):
    """Eq. (56), u-dependent part, in mpmath (independent reference implementation)."""
    from mpmath import mp, mpf, acos, exp as mexp, log as mlog, pi as mpi, clsin

    mp.dps = 30
    uu = [mpf(float(v)) for v in u]
    tot = mpf(0)
    for fi in range(disk.n_faces):
        ell = []
        for c in range(3):
            i, j = disk.face_edges[fi][c]
            ell.append(mexp(0.5 * (mpf(float(_lam_get(lam, (i, j)))) + uu[i] + uu[j])))
        al = []
        for i in range(3):
            j, k = (i + 1) % 3, (i + 2) % 3
            if not (ell[i] < ell[j] + ell[k]):
                return mp.inf  # not a Euclidean triangle: the energy is undefined
            c = (ell[j] ** 2 + ell[k] ** 2 - ell[i] ** 2) / (2 * ell[j] * ell[k])
            al.append(acos(min(mp.mpf(1), max(mp.mpf(-1), c))))
        xs = [mlog(e) for e in ell]
        tot += 2 * (sum(al[i] * xs[i] for i in range(3)) + sum(clsin(2, 2 * al[i]) / 2 for i in range(3)))
    for (i, j) in disk.edges:
        tot -= mpi * (mpf(float(_lam_get(lam, (i, j)))) + uu[i] + uu[j])
    for v in range(disk.n):
        tot += 2 * mpi * uu[v]
    return tot


def scaling_defect(disk: Disk, lam, u: np.ndarray, h: float) -> float:
    """Prop. 7.17 (v), eq. (60): ``E(u + h 1) - E(u) - 2 pi h`` must vanish."""
    u = np.asarray(u, float)
    return energy_np(disk, lam, u + h) - energy_np(disk, lam, u) - 2.0 * np.pi * h
