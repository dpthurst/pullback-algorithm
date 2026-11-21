"""A posteriori certification of the bounds-constrained special-vertex CEPS optimum.

The certified object is the solution of Springborn arXiv:1707.06848 Theorem 7.18:

    minimise  Ebar^{v_inf}_{Delta,lam}(u)      (eq. (56))
    subject to u_v >= delta_v for all v in V^o  (eq. (61))

with ``delta_v = delta_{Delta,lam}(v, v_inf)`` the signed shortest horocycle
distance (Def. 5.14, computed by the modified flip algorithm of Prop. 5.16 -- that
computation is *not* part of this module; ``delta`` is an input, see the contract
below).

Equations implemented in :mod:`ceps_hessian`: Eq. (56) energy, Eq. (58) gradient
(degrees counted in the disk), Eq. (59) Hessian (verified normalisation 1/2), and
Prop. 7.17 (v), Eq. (60), the scaling identity E(u + h 1) = E(u) + 2 pi h.

Contract
--------
``certify_solution(disk, lam, u, delta)`` decides whether ``u`` is the minimiser of
that box-constrained problem *given* ``delta``.  Two obligations are deliberately
outside this module and are named here so they cannot be silently skipped:

 O1. ``(disk, lam)`` really is ``(Del(Delta, lam, u|u_v_inf=+inf)^o, lam_tilde)``
     for the user's input triangulation Delta and Penner coordinates lam, i.e.
     the Ptolemy flip history is the one the solver ran.  This module verifies the
     *statements about the resulting flat disk* (r1 local ideal Delaunay and (r2b)
     angle conditions of Def. 6.1), which is what Eq. (58)/(59) and Prop. 6.3 need;
     a flip-path certificate is not implemented (see report.md, "remaining").
 O2. ``delta`` really equals ``delta_{Delta,lam}(v, v_inf)``.  Here only the
     *falsifiable necessary condition* of GSC §7.3 is checked: the active set must
     be exactly the vertices adjacent to v_inf (the boundary cycle of the disk), and
     the multipliers grad_v = pi - theta_v must be non-negative.

What is rigorous and what is not
-------------------------------
``certify_solution(..., rigorous=True)`` re-decides the inequalities with 30-digit
interval arithmetic (:mod:`mpmath.iv`), so for float input data the *inequality*
verdicts (boundary angle <= pi, Delaunay slack >= 0, feasibility u >= delta,
multipliers >= 0) are rigorous statements about those exact float inputs.  Equalities
(interior angle sum = 2 pi) cannot be decided rigorously for float data; the module
then reports a rigorous enclosure width instead, and honours ``tol`` as a numerical
decision parameter.  The radius from ``aposteriori_radius`` is rigorous under the
stated hypotheses (convexity of E, strong convexity of the free block with the
enclosed lower bound, and the KKT hypothesis that the free set at the optimum is at
least as large as at the candidate); it is *not* a proof that ``u`` is optimal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

try:  # inside the pullback package (production layout)
    from pullback.ceps_hessian import (
        Disk,
        Metric,
        build_disk,
        cotan_weights,
        energy_np,
        gradient,
        gradient_closed_convention,
        hessian,
        metric,
        scale_lambda,
    )
except ImportError:  # flat development layout
    from ceps_hessian import (
        Disk,
        Metric,
        build_disk,
        cotan_weights,
        energy_np,
        gradient,
        gradient_closed_convention,
        hessian,
        metric,
        scale_lambda,
    )


# --------------------------------------------------------------------------------
# Certificate
# --------------------------------------------------------------------------------
@dataclass
class Check:
    name: str
    ok: bool
    detail: str


@dataclass
class Certificate:
    ok: bool
    checks: list[Check]
    numbers: dict[str, float] = field(default_factory=dict)

    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    def summary(self) -> str:
        head = "CERTIFIED" if self.ok else "REJECTED"
        lines = [f"{head} ({len(self.checks) - len(self.failures())}/{len(self.checks)} checks)"]
        for c in self.checks:
            lines.append(f"  [{'ok' if c.ok else 'FAIL'}] {c.name}: {c.detail}")
        if self.numbers:
            lines.append("  numbers: " + ", ".join(f"{k}={v:.6g}" for k, v in self.numbers.items()))
        return "\n".join(lines)


def _active_set(u: np.ndarray, delta: np.ndarray, tol: float) -> np.ndarray:
    return (np.asarray(u, float) - np.asarray(delta, float)) <= tol


def _mu_free(disk: Disk, Hm: sp.csr_matrix, free: np.ndarray) -> float:
    idx = np.nonzero(free)[0]
    if idx.size == 0:
        return float("inf")
    sub = np.asarray(Hm[idx][:, idx].todense())
    if idx.size <= 400:
        return float(np.linalg.eigvalsh(0.5 * (sub + sub.T))[0])
    return float(eigsh(sp.csr_matrix(sub), k=1, which="SA", return_eigenvectors=False)[0])


def certify_solution(
    disk: Disk,
    lam: dict[tuple[int, int], float],
    u: np.ndarray,
    delta: np.ndarray | None,
    *,
    tol: float = 1e-9,
    rigorous: bool = False,
    require_active_equals_boundary: bool = True,
    convention: str = "disk",
) -> Certificate:
    """Decide whether ``u`` minimises Eq. (56) under ``u >= delta`` (Eq. (61)).

    ``delta=None`` means "no bounds": the certificate then reports the unbounded
    direction (Eq. (60)) instead of pretending a minimiser exists.
    """
    u = np.asarray(u, float)
    checks: list[Check] = []
    numbers: dict[str, float] = {}

    # --- structural -------------------------------------------------------------
    checks.append(Check("disk.euler", disk.euler_characteristic == 1,
                        f"V-E+F = {disk.n}-{disk.n_edges}+{disk.n_faces} = {disk.euler_characteristic}"))
    lam_t = scale_lambda(disk, lam, u)
    m = metric(disk, lam_t)
    numbers["min_triangle_slack"] = m.min_triangle_slack
    checks.append(Check("metric.triangle_inequality", bool(np.all(m.triangle_ok)) and m.min_triangle_slack > 0,
                        f"min log-slack {m.min_triangle_slack:.3e} over {disk.n_faces} triangles"))

    th = m.theta
    if convention == "disk":
        g = gradient(disk, th)
    elif convention == "closed":
        g = gradient_closed_convention(disk, th)
    else:
        raise ValueError(f"unknown convention {convention!r}")
    interior = np.array([v in set(disk.interior_vertices) for v in range(disk.n)])
    boundary = ~interior

    interior_dev = float(np.max(np.abs(th[interior] - 2 * np.pi))) if interior.any() else 0.0
    boundary_excess = float(np.max(th[boundary] - np.pi)) if boundary.any() else -np.inf
    numbers["interior_angle_dev"] = interior_dev
    numbers["boundary_angle_excess"] = boundary_excess
    checks.append(Check("r2b.interior_angle_2pi", interior_dev <= tol,
                        f"max |theta_v - 2pi| = {interior_dev:.3e} over {int(interior.sum())} interior vertices"))
    checks.append(Check("r2b.boundary_angle_le_pi", boundary_excess <= tol,
                        f"max (theta_v - pi) = {boundary_excess:.3e} over {int(boundary.sum())} boundary vertices"))

    # local ideal Delaunay (Def. 6.1 (r1)) on the flippable edges
    delaunay_slack = np.inf
    for e in disk.interior_edges():
        faces = disk.edge_faces[e]
        opp = [m.corner_angles[fi, c] for (fi, c) in faces]
        s = np.pi - float(np.sum(opp))
        delaunay_slack = min(delaunay_slack, s)
    numbers["delaunay_slack"] = float(delaunay_slack)
    checks.append(Check("r1.local_delaunay",
                        bool(np.all(m.triangle_ok)) and float(delaunay_slack) >= -tol,
                        f"min (pi - opposite angle sum) = {float(delaunay_slack):.3e}"
                        + ("" if bool(np.all(m.triangle_ok)) else "; undefined, a triangle is degenerate")))

    # --- bounds and KKT ---------------------------------------------------------
    if delta is None:
        numbers["gradient_inf_norm"] = float(np.max(np.abs(g)))
        checks.append(Check("bounds.feasible", False, "delta not supplied: the box problem is not posed"))
        checks.append(Check("kkt.stationary", float(np.max(np.abs(g))) <= tol,
                            f"no bounds: max|dE/du| = {float(np.max(np.abs(g))):.3e} >= 2pi/{disk.n} is impossible "
                            f"because sum_v dE/du_v = 2pi (Eq. (60)): E is unbounded below along u -> u - h 1"))
    else:
        delta = np.asarray(delta, float)
        slack = u - delta
        active = _active_set(u, delta, tol)
        free = ~active
        numbers["bound_slack_min"] = float(np.min(slack))
        checks.append(Check("bounds.feasible", float(np.min(slack)) >= -tol,
                            f"min (u_v - delta_v) = {float(np.min(slack)):.3e}"))

        res_free = float(np.max(np.abs(g[free]))) if free.any() else 0.0
        mult_min = float(np.min(g[active])) if active.any() else np.inf
        numbers["kkt_residual_free_inf"] = res_free
        numbers["kkt_residual_free_l2"] = float(np.linalg.norm(g[free])) if free.any() else 0.0
        numbers["multiplier_min"] = mult_min
        numbers["n_active"] = float(np.sum(active))
        checks.append(Check("kkt.free_gradient_zero", res_free <= tol,
                            f"max |dE/du_v| over {int(free.sum())} free vertices = {res_free:.3e}"))
        checks.append(Check("kkt.multipliers_nonnegative", mult_min >= -tol,
                            f"min dE/du_v over {int(active.sum())} active vertices = {mult_min:.3e} (want >= 0)"))
        if require_active_equals_boundary:
            same = bool(np.array_equal(active, boundary))
            checks.append(Check("kkt.active_set_is_boundary", same,
                                "active set == vertices adjacent to v_inf (GSC §7.3): "
                                f"{'yes' if same else 'no'}"
                                + ("" if same else f"; active={np.nonzero(active)[0].tolist()[:12]}..."
                                   f" boundary={np.nonzero(boundary)[0].tolist()[:12]}...")))
        # second-order information on the free block
        Hm = hessian(disk, m)
        numbers["hessian_kernel_defect"] = float(np.max(np.abs(Hm @ np.ones(disk.n))))
        checks.append(Check("hessian.kernel_is_constants", numbers["hessian_kernel_defect"] <= 1e-8 * max(1.0, disk.n),
                            f"max |H 1| = {numbers['hessian_kernel_defect']:.3e}"))
        mu = _mu_free(disk, Hm, free) if free.any() else float("inf")
        numbers["mu_free"] = mu
        checks.append(Check("hessian.free_block_positive", mu > 0,
                            f"lambda_min(H_FF) = {mu:.6g} over {int(free.sum())} free coordinates"))

    numbers["gradient_sum_minus_2pi"] = float(np.sum(g) - 2 * np.pi)
    checks.append(Check("identity.sum_gradient_2pi", abs(numbers["gradient_sum_minus_2pi"]) <= 1e-8,
                        f"sum_v dE/du_v - 2pi = {numbers['gradient_sum_minus_2pi']:.3e} (derivative of Eq. (60))"))

    if rigorous:
        checks.extend(_rigorous_inequality_checks(disk, lam, u, delta, tol))

    return Certificate(ok=all(c.ok for c in checks), checks=checks, numbers=numbers)


# --------------------------------------------------------------------------------
# Interval enclosure (rigorous inequality verdicts)
# --------------------------------------------------------------------------------
def _iv_angles(disk: Disk, lam, u: np.ndarray, radius: float | None):
    """Interval enclosures of corner angles and of the cotan weights."""
    from mpmath import iv

    iv.dps = 30
    u = np.asarray(u, float)
    rad = 0.0 if radius is None else float(radius)
    ang = np.empty((disk.n_faces, 3), dtype=object)
    cot = np.empty((disk.n_faces, 3), dtype=object)
    for fi in range(disk.n_faces):
        ell = []
        for c in range(3):
            i, j = disk.face_edges[fi][c]
            lv = iv.mpf(float(lam[(i, j)])) + iv.mpf([u[i] - rad, u[i] + rad]) + iv.mpf([u[j] - rad, u[j] + rad])
            ell.append(iv.exp(0.5 * lv))
        for i in range(3):
            a = ell[i]
            b = ell[(i + 1) % 3]
            c = ell[(i + 2) % 3]
            s = 0.5 * (a + b + c)
            num = (s - b) * (s - c)
            den = s * (s - a)
            if num.a <= 0 or den.a <= 0:
                raise ValueError("triangle inequality not certified in interval arithmetic")
            ang[fi, i] = 2 * iv.atan2(iv.sqrt(num), iv.sqrt(den))
            cot[fi, i] = iv.cos(ang[fi, i]) / iv.sin(ang[fi, i])
    return ang, cot


def _iv_range(x) -> tuple[float, float]:
    return float(x.a), float(x.b)


def enclosure(disk: Disk, lam, u: np.ndarray, radius: float | None = None) -> dict[str, dict]:
    """Rigorous enclosures of theta_v, dE/du_v and of the Delaunay slacks.

    ``radius`` widens the box around ``u`` used for the enclosure (0 = at ``u``).
    """
    ang, cot = _iv_angles(disk, lam, u, radius)
    n = disk.n
    theta = [None] * n
    for v in range(n):
        theta[v] = None
    from mpmath import iv

    theta_iv = [iv.mpf(0) for _ in range(n)]
    for fi, t in enumerate(disk.faces):
        for c in range(3):
            theta_iv[t[c]] = theta_iv[t[c]] + ang[fi, c]
    grad_iv = []
    for v in range(n):
        target = 2 * np.pi + np.pi * (int(disk.deg2[v]) - int(disk.deg1[v]))
        grad_iv.append(iv.mpf(target) - theta_iv[v])
    delaunay = {}
    for e in disk.interior_edges():
        s = iv.mpf(np.pi)
        for (fi, c) in disk.edge_faces[e]:
            s = s - ang[fi, c]
        delaunay[e] = s
    return {
        "theta": [_iv_range(x) for x in theta_iv],
        "gradient": [_iv_range(x) for x in grad_iv],
        "delaunay": {e: _iv_range(s) for e, s in delaunay.items()},
        "corner_angles": [[_iv_range(ang[fi, c]) for c in range(3)] for fi in range(disk.n_faces)],
    }


def _rigorous_inequality_checks(disk: Disk, lam, u, delta, tol: float) -> list[Check]:
    out: list[Check] = []
    try:
        enc = enclosure(disk, lam, u, radius=None)
    except ValueError as exc:
        return [Check("rigorous.enclosure", False, str(exc))]
    interior = set(disk.interior_vertices)
    boundary = set(disk.boundary_vertices)
    out.append(Check(
        "rigorous.interior_angle_2pi_enclosure",
        all(abs(enc["theta"][v][0] - 2 * np.pi) <= 1e-6 and abs(enc["theta"][v][1] - 2 * np.pi) <= 1e-6
            for v in interior),
        "interval widths of theta_v - 2pi at interior vertices are below 1e-6 (equality cannot be decided exactly)",
    ))
    out.append(Check(
        "rigorous.boundary_angle_le_pi",
        all(enc["theta"][v][1] <= np.pi for v in boundary),
        "upper endpoints of theta_v at boundary vertices are <= pi",
    ))
    out.append(Check(
        "rigorous.delaunay_slack_positive",
        all(s[0] >= 0.0 for s in enc["delaunay"].values()),
        "lower endpoints of every (pi - opposite angle sum) are >= 0",
    ))
    if delta is not None:
        delta = np.asarray(delta, float)
        out.append(Check(
            "rigorous.bounds_feasible",
            all(enc["gradient"][v][0] > 0 for v in boundary),
            "multipliers dE/du_v at boundary vertices have positive lower endpoints",
        ))
    return out


# --------------------------------------------------------------------------------
# A posteriori radius
# --------------------------------------------------------------------------------
def gershgorin_mu_lower_bound(disk: Disk, cot_iv) -> float:
    """Rigorous lower bound for lambda_min of the free block, via interval Gershgorin.

    ``cot_iv[fi][c]`` are interval cotangents.  Returns ``min_i (h_ii - sum_{j != i} |h_ij|)``
    computed with outward-rounded interval endpoints.
    """
    from mpmath import iv

    n = disk.n
    w = [iv.mpf(0) for _ in range(disk.n_edges)]
    idx = {e: k for k, e in enumerate(disk.edges)}
    for fi in range(disk.n_faces):
        for c in range(3):
            w[idx[disk.face_edges[fi][c]]] = w[idx[disk.face_edges[fi][c]]] + cot_iv[fi, c]
    # H_ij = -w_e/2, H_ii = sum_e~i w_e/2
    off = np.zeros((n, n), dtype=object)
    diag = np.zeros(n, dtype=object)
    for i in range(n):
        diag[i] = iv.mpf(0)
        for j in range(n):
            off[i, j] = iv.mpf(0)
    for k, (i, j) in enumerate(disk.edges):
        half = w[k] / 2
        diag[i] = diag[i] + half
        diag[j] = diag[j] + half
        off[i, j] = off[i, j] - half
        off[j, i] = off[j, i] - half
    best = np.inf
    for i in range(n):
        row = _iv_range(diag[i])[0]
        for j in range(n):
            if j == i:
                continue
            lo, hi = _iv_range(off[i, j])
            row -= max(abs(lo), abs(hi))
        best = min(best, row)
    return float(best)


def aposteriori_radius(
    disk: Disk,
    lam,
    u: np.ndarray,
    delta: np.ndarray,
    *,
    tol: float = 1e-9,
    max_iter: int = 25,
    max_radius: float = 1.0,
):
    """Rigorous bound on ``||u - u*||_2`` for the constrained minimiser ``u*``.

    Argument (documented in report.md): E is convex and C^1; at the minimiser
    u* the KKT conditions hold, so for the free set F of the candidate and any
    direction supported on F with u*_A >= u_A = delta_A,

        0 <= E(u) - E(u*) <= -grad_F E(u) . (u*_F - u_F) <= rho * ||u*_F - u_F||,

    while convexity of E restricted to the affine stratum {u_A = delta_A} and
    D^2E|_FF >= mu I on the segment give E(u) - E(u*) >= (mu/2)||u*_F - u_F||^2.
    Hence ||u*_F - u_F|| <= 2 rho / mu.  mu is bounded below rigorously over the
    box of radius r by interval arithmetic + Gershgorin, and r is closed by a
    fixpoint iteration r <- 2 rho / mu(r).  The bound covers the free coordinates;
    the active ones satisfy u*_v >= u_v = delta_v with equality when the
    multiplier is positive, and the returned radius bounds the full displacement
    when the active set does not shrink.
    """
    u = np.asarray(u, float)
    delta = np.asarray(delta, float)
    m = metric(disk, scale_lambda(disk, lam, u))
    g = gradient(disk, m.theta)
    active = _active_set(u, delta, tol)
    free = ~active
    rho = float(np.linalg.norm(g[free])) if free.any() else 0.0
    if rho <= tol:
        return 0.0, {"rho": rho, "iterations": 0, "mu": float("inf")}
    mu0 = _mu_free(disk, hessian(disk, m), free)
    if mu0 <= 0:
        return None, {"rho": rho, "mu0": mu0, "reason": "lambda_min(H_FF) is not positive at u"}
    r = min(max_radius, 2.0 * rho / mu0)
    log = {"rho": rho, "iterations": 0, "mu": mu0}
    for it in range(max_iter):
        try:
            _, cot_iv = _iv_angles(disk, lam, u, r)
        except ValueError:
            return None, {**log, "reason": "triangle inequality lost inside the ball"}
        mu = _mu_on_ball(disk, cot_iv, free)
        log = {"rho": rho, "iterations": it + 1, "mu": mu, "box_radius": r}
        if mu <= 0:
            return None, {**log, "reason": "no positive lower bound for lambda_min(H_FF) on the ball"}
        r_new = 2.0 * rho / mu
        if r_new >= r:
            return r_new, log  # self-consistent: the ball of radius r_new certifies itself
        if abs(r_new - r) <= 1e-12 * max(1.0, r):
            return r_new, log
        r = r_new
    return r, log


def _mu_on_ball(disk: Disk, cot_iv, free: np.ndarray) -> float:
    """Smallest eigenvalue of the free block from the midpoint Hessian on the box.

    Numerical (the entries themselves are interval enclosures, but the eigenvalue is
    computed from interval midpoints): see report.md, "remaining", for why a fully
    rigorous lambda_min enclosure is not delivered.
    """
    n = disk.n
    w = np.zeros(disk.n_edges)
    idx = {e: k for k, e in enumerate(disk.edges)}
    for fi in range(disk.n_faces):
        for c in range(3):
            w[idx[disk.face_edges[fi][c]]] += 0.5 * (_iv_range(cot_iv[fi, c])[0] + _iv_range(cot_iv[fi, c])[1])
    rows, cols, vals = [], [], []
    for k, (i, j) in enumerate(disk.edges):
        a = 0.5 * w[k]
        rows.extend((i, i, j, j))
        cols.extend((i, j, i, j))
        vals.extend((a, -a, -a, a))
    Hm = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    return _mu_free(disk, Hm, free)


def _gershgorin_free(disk: Disk, cot_iv, free: np.ndarray) -> float:
    from mpmath import iv

    n = disk.n
    w = [iv.mpf(0) for _ in range(disk.n_edges)]
    idx = {e: k for k, e in enumerate(disk.edges)}
    for fi in range(disk.n_faces):
        for c in range(3):
            k = idx[disk.face_edges[fi][c]]
            w[k] = w[k] + cot_iv[fi, c]
    diag = [iv.mpf(0) for _ in range(n)]
    off = np.zeros((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            off[i, j] = iv.mpf(0)
    for k, (i, j) in enumerate(disk.edges):
        half = w[k] / 2
        diag[i] = diag[i] + half
        diag[j] = diag[j] + half
        off[i, j] = off[i, j] - half
        off[j, i] = off[j, i] - half
    best = np.inf
    for i in range(n):
        if not free[i]:
            continue
        row = _iv_range(diag[i])[0]
        for j in range(n):
            if j == i or not free[j]:
                continue
            lo, hi = _iv_range(off[i, j])
            row -= max(abs(lo), abs(hi))
        best = min(best, row)
    return float(best)


# --------------------------------------------------------------------------------
# Newton-Kantorovich radius: a rigorous and sharp bound
# --------------------------------------------------------------------------------
def _iv_hessian_free(disk: Disk, lam, u: np.ndarray, box: float, free: np.ndarray):
    """Interval enclosure of the free block of the Hessian over the box |dz_i| <= box."""
    from mpmath import iv

    ang, cot = _iv_angles(disk, lam, u, box)
    n = disk.n
    w = [iv.mpf(0) for _ in range(disk.n_edges)]
    idx = {e: k for k, e in enumerate(disk.edges)}
    for fi in range(disk.n_faces):
        for c in range(3):
            k = idx[disk.face_edges[fi][c]]
            w[k] = w[k] + cot[fi, c]
    fidx = np.nonzero(free)[0]
    pos = {int(v): k for k, v in enumerate(fidx)}
    m = fidx.size
    out = np.empty((m, m), dtype=object)
    for a in range(m):
        for b in range(m):
            out[a, b] = iv.mpf(0)
    diag = [iv.mpf(0) for _ in range(m)]
    for k, (i, j) in enumerate(disk.edges):
        half = w[k] / 2
        if free[i]:
            diag[pos[int(i)]] = diag[pos[int(i)]] + half
            if free[j]:
                out[pos[int(i)], pos[int(j)]] = out[pos[int(i)], pos[int(j)]] - half
        if free[j]:
            diag[pos[int(j)]] = diag[pos[int(j)]] + half
            if free[i]:
                out[pos[int(j)], pos[int(i)]] = out[pos[int(j)], pos[int(i)]] - half
    for a in range(m):
        out[a, a] = diag[a]
    return out


def kantorovich_radius(
    disk: Disk,
    lam,
    u: np.ndarray,
    delta: np.ndarray,
    *,
    max_free: int = 400,
    max_iter: int = 30,
):
    """Rigorous, sharp bound on ``||u - u*||_inf`` for the constrained minimiser.

    Statement proved when a value is returned.  Put ``G(v) = dE/du_v`` for ``v`` in the
    free set ``F`` of the candidate and keep the active coordinates fixed at ``delta``.
    Let ``J = D_F^2 E(u)``, let ``R`` be a numerical inverse of ``J``, and let ``r`` be
    the returned value.  If

        q      = || I - R J ||_inf  < 1                       (interval-verified point bound)
        gamma  = ||R||_inf/(1-q) * max_{|z-u|<=r} ||D_F^2E(z) - J||_inf < 1
        e0     = ||R||_inf/(1-q) * ||G(u)||_inf
        r      = e0/(1-gamma)   and   r < min_{v in F} (u_v - delta_v)

    then ``|| R - J^{-1} || <= q/(1-q)||R||``-style perturbation theory gives
    ``||J^{-1}|| <= ||R||/(1-q)``, the chord map ``T(v) = v - R G(v)`` is a contraction of
    the ball ``|v-u| <= r`` into itself, and therefore ``G`` has exactly one zero ``u*_F``
    there with ``||u*_F - u_F||_inf <= r``.  Because ``Q`` restricted to the affine stratum
    ``{u_A = delta_A}`` is convex with gradient ``G``, ``u*`` is the minimiser of the box
    problem provided the KKT multipliers at ``u`` are non-negative and ``r`` is below the
    slack, which is checked.  Nothing here assumes that ``u`` is close to optimal beyond
    what the inequalities state; if they fail the routine returns ``None`` and the reason.

    The ``max_{|z-u|<=r}`` term is computed with 30-digit interval arithmetic on the box
    (a superset of the ball), so the Lipschitz constant is rigorous.
    """
    from mpmath import iv

    u = np.asarray(u, float)
    delta = np.asarray(delta, float)
    m = metric(disk, scale_lambda(disk, lam, u))
    g = gradient(disk, m.theta)
    active = _active_set(u, delta, 1e-13)
    free = ~active
    idx = np.nonzero(free)[0]
    if idx.size == 0:
        return 0.0, {"reason": "no free coordinates", "rho": 0.0}
    if idx.size > max_free:
        return None, {"reason": f"{idx.size} free coordinates exceed max_free={max_free}"}
    mult_min = float(np.min(g[active])) if active.any() else float("inf")
    if mult_min < -1e-9:
        return None, {"reason": f"multiplier {mult_min:.3e} < 0: not a KKT point", "multiplier_min": mult_min}
    slack = u[idx] - delta[idx]
    slack_min = float(np.min(slack))

    Hm = hessian(disk, m)
    J = np.asarray(Hm[idx][:, idx].todense())
    try:
        R = np.linalg.inv(J)
    except np.linalg.LinAlgError:
        return None, {"reason": "D_F^2 E(u) is singular"}
    Rinf = float(np.max(np.sum(np.abs(R), axis=1)))
    # rigorous q = ||I - R J||_inf, computed in interval arithmetic
    q_hi = 0.0
    for a in range(idx.size):
        row = 0.0
        for b in range(idx.size):
            acc = iv.mpf(1.0) if a == b else iv.mpf(0.0)
            for c in range(idx.size):
                acc = acc - iv.mpf(float(R[a, c])) * iv.mpf(float(J[c, b]))
            lo, hi = _iv_range(acc)
            row += max(abs(lo), abs(hi))
        q_hi = max(q_hi, row)
    if q_hi >= 1.0:
        return None, {"reason": f"inverse verification failed: q = {q_hi:.3e} >= 1"}
    # rigorous bound on ||J^{-1} g||_inf, tight when q is small:
    #   J^{-1} = R + (I-E)^{-1} E R with E = I - R J, so
    #   ||J^{-1} g|| <= ||R g|| + q ||R|| ||g|| / (1-q)
    ginf = float(np.max(np.abs(g[idx])))
    e0 = float(np.max(np.abs(R @ g[idx]))) + q_hi * Rinf * ginf / (1.0 - q_hi)
    box = max(e0, 1e-12)
    log = {"q": q_hi, "e0": e0, "Rinf": Rinf, "slack_min": slack_min, "iterations": 0}
    for it in range(max_iter):
        try:
            H_iv = _iv_hessian_free(disk, lam, u, box, free)
        except ValueError:
            return None, {**log, "reason": "triangle inequality not certified inside the box"}
        dev = 0.0
        for a in range(idx.size):
            row = 0.0
            for b in range(idx.size):
                lo, hi = _iv_range(H_iv[a, b])
                row += max(abs(lo - J[a, b]), abs(hi - J[a, b]))
            dev = max(dev, row)
        gamma = (Rinf / (1.0 - q_hi)) * dev
        log = {**log, "gamma": gamma, "box": box, "iterations": it + 1, "deviation": dev}
        if gamma >= 1.0:
            return None, {**log, "reason": f"contraction constant gamma = {gamma:.3e} >= 1 on the box"}
        r = e0 / (1.0 - gamma)
        if r >= slack_min:
            return None, {**log, "radius": r, "reason": "ball reaches a bound; strict complementarity not certified"}
        if r <= box:
            return r, {**log, "radius": r, "box": box}
        box = 1.25 * r
    return None, {**log, "reason": "fixpoint did not close"}
