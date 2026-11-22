"""Discrete uniformisation of a triangulated sphere: the round realisation.

Given a triangulation of S^2 with n vertices and positive edge lengths
satisfying the triangle inequalities, `round_realise` looks for positions
x_i on the unit sphere whose *chordal* metric is discretely conformally
equivalent to the given lengths, in the sense of Springborn (arXiv:1707.06848,
Def. 10.1): two metrics on a fixed triangulation are discretely conformally
equivalent iff

    log chord_ij - log ell_ij = (u_i + u_j)/2      for every edge ij,

for some vertex scaling u.  Springborn's Theorem 10.5 guarantees such a realisation
for EVERY piecewise-Euclidean sphere metric, without any restriction that the
input cone angles be below 2*pi.  Convexity belongs to the inscribed OUTPUT;
discrete conformal vertex scaling changes triangle angles and can move an input
angle above 2*pi back below that threshold.  (BDS's prescribed-angle, hyper-ideal
circle-pattern setup is a different framework.)

The theorem allows Delaunay changes; this solver works on a fixed triangulation
and is a local numerical method, so its success is measured by the residual and
is not inferred from the theorem.

The criterion used here is exact and cheap: for a candidate configuration the
numbers a_ij = log chord_ij - log ell_ij must lie in the image of the edge-
vertex incidence map A (A[(ij),k] = 1/2 if k in {i,j}).  So the residual

    r(x) = (I - A A^+) a(x)

vanishes exactly when x is a round realisation, and its norm is the natural
objective to minimise over x.  `conformal_residual` computes it;
`round_realise` minimises it by Gauss-Newton on the sphere (normalising after
each step), with an analytic Jacobian -- small meshes only, which is the regime
the |P| = 4 work lives in.

Every function here is a *testable* object: a round metric has residual zero at
its own configuration and at every vertex-scaling of it.  In particular, a
vertex-scaled metric whose cone angles cross 2*pi remains a valid test case.
"""

from __future__ import annotations

import numpy as np


def edges_of(faces):
    """The edge list of a triangulation, as sorted pairs with no repeats."""
    seen = set()
    for f in faces:
        n = len(f)
        for k in range(n):
            i, j = f[k], f[(k + 1) % n]
            seen.add((i, j) if i < j else (j, i))
    return sorted(seen)


def chordal_lengths(x):
    """Matrix of distances |x_i - x_j| for points on the sphere."""
    x = np.asarray(x, dtype=float)
    sq = np.sum(x * x, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * x @ x.T
    return np.sqrt(np.clip(d2, 0.0, None))


def incidence(faces, n):
    """A: edges x vertices, with 1/2 in the two slots of each edge."""
    E = edges_of(faces)
    A = np.zeros((len(E), n))
    for row, (i, j) in enumerate(E):
        A[row, i] = 0.5
        A[row, j] = 0.5
    return A, E


_PROJECTOR_CACHE = {}


def projector(faces, n):
    """The orthogonal projector P = I - A A^+ onto the complement of the image
    of the edge-vertex incidence, cached.

    The conformal residual is P a with a_ij = log chord_ij - log ell_ij, and the
    least-squares solve that used to sit inside every residual evaluation is the
    same one every time -- A depends only on the combinatorics.  Computing P once
    turns each residual into a matvec (measured: 1.49 ms -> 0.15 ms at 336 edges)
    and, more importantly, makes an analytic Jacobian cheap to build.
    """
    key = (tuple(tuple(f) for f in faces), int(n))
    if key not in _PROJECTOR_CACHE:
        A, E = incidence(faces, n)
        P = np.eye(len(E)) - A @ np.linalg.pinv(A)
        # edges incident with each vertex, for the analytic Jacobian
        rows = [[] for _ in range(n)]
        for e, (i, j) in enumerate(E):
            rows[i].append(e)
            rows[j].append(e)
        _PROJECTOR_CACHE[key] = (P, E, rows)
    return _PROJECTOR_CACHE[key]


def conformal_residual(faces, lengths, x):
    """r = (I - A A^+) (log chord - log ell), and the projected part's norm.

    `lengths` is a dict {(i, j): ell}.  Returns (residual per edge, edges,
    the norm of the part that no vertex scaling can explain).
    """
    x = np.asarray(x)
    P, E, _rows = projector(faces, len(x))
    chord = chordal_lengths(x)
    a = np.array([np.log(chord[i, j]) - np.log(lengths[(i, j)]) for (i, j) in E])
    r = P @ a
    return r, E, float(np.linalg.norm(r))


def conformal_jacobian(faces, x):
    """The derivative of the conformal residual with respect to the positions.

    d r / d x = P J_a, and J_a is sparse: a_ij = log|x_i - x_j| - log ell_ij, so
    d a_ij/d x_i = u and d a_ij/d x_j = -u with u = (x_i - x_j)/|x_i - x_j|^2.
    Built vertex by vertex -- column block k comes from the edges *incident with
    k* only -- so the cost is O(E^2) in total rather than the E * 3n of a dense
    product, and there is no finite-difference error.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    P, E, rows = projector(faces, n)
    J = np.zeros((len(E), 3 * n))
    for k in range(n):
        inc = rows[k]
        if not inc:
            continue
        B = np.zeros((len(inc), 3))
        for slot, e in enumerate(inc):
            i, j = E[e]
            other = j if i == k else i
            d = x[k] - x[other]
            d2 = float(d @ d)
            if d2 <= 0.0:
                continue
            # d a_e / d x_k = (x_k - x_other)/|x_k - x_other|^2, with the SAME
            # sign at either endpoint (the derivative at the other end brings
            # its own minus from d(x_i - x_j)/dx_j)
            B[slot] = d / d2
        J[:, 3 * k:3 * k + 3] = P[:, inc] @ B
    return J


# `conformal_residual` is defined once, above, using `projector`.  An accidental
# second copy of the old least-squares implementation here used to override it;
# besides wasting a solve on every residual evaluation, that made the analytic
# Jacobian and the runtime residual take different numerical paths.


def round_realise(faces, lengths, x0=None, maxit=120, tol=1e-13, step=1.0,
                  seed=0):
    """Positions on S^2 with chordal metric discretely conformal to `lengths`.

    Gauss-Newton on the residual of `conformal_residual`, with the ANALYTIC
    Jacobian of `conformal_jacobian` and a backtracking step; the configuration is
    renormalised onto the sphere after each step.  Returns (x, residual_norm,
    iterations), where the iteration count is the true one -- the loop stops when
    the residual stops improving, not at the cap.

    The analytic Jacobian replaced a finite-difference one (measured 2026-10-28:
    0.002 s against 0.466 s at 336 edges, 296x, and no difference error).  It also
    fixes the floor: the finite-difference Jacobian stalls at a residual about
    2.4e-12 regardless of the tolerance asked for, because the step it computes is
    only accurate to the differencing.  The Möbius/rotation gauge is not fixed --
    `numpy.linalg.lstsq` takes the minimum-norm step through it.

    It is a local method: start it from a configuration with the right
    combinatorics (the caller's initial guess), which is exactly how the
    pullback iteration will use it, one step at a time.
    """
    n = len(faces) and (max(max(f) for f in faces) + 1)
    if x0 is None:
        rng = np.random.RandomState(seed)
        x = rng.normal(size=(n, 3))
        x /= np.linalg.norm(x, axis=1)[:, None]
    else:
        x = np.array(x0, dtype=float)
        x /= np.linalg.norm(x, axis=1)[:, None]

    def residual(xx):
        return conformal_residual(faces, lengths, xx)[0]

    best = float(np.linalg.norm(residual(x)))
    stalls = 0
    it = 0
    for it in range(maxit):
        r0 = residual(x)
        norm0 = float(np.linalg.norm(r0))
        if norm0 < tol:
            return x, norm0, it
        J = conformal_jacobian(faces, x)
        delta, *_ = np.linalg.lstsq(J, -r0, rcond=None)
        accepted = False
        for _ in range(20):
            xt = (x.reshape(-1) + step * delta).reshape(x.shape)
            xt /= np.linalg.norm(xt, axis=1)[:, None]
            normt = float(np.linalg.norm(residual(xt)))
            if normt < norm0:
                progress = (norm0 - normt) / max(norm0, 1e-300)
                x, best, accepted = xt, normt, True
                break
            step *= 0.5
        if not accepted:
            break
        # stop when the residual has stopped improving: without this the loop
        # burns its whole budget chasing a floor it cannot reach (measured: the
        # numerical-Jacobian version stalled at 2.35e-12 and reported `maxit`
        # every time, which made the iteration count meaningless)
        stalls = stalls + 1 if progress < 1e-10 else 0
        if stalls >= 3:
            break
        step = min(1.0, step * 1.5)
    return x, best, it
