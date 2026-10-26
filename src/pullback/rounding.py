"""Discrete uniformisation of a triangulated sphere: the round realisation.

Given a triangulation of S^2 with n vertices and positive edge lengths
satisfying the triangle inequalities, `round_realise` looks for positions
x_i on the unit sphere whose *chordal* metric is discretely conformally
equivalent to the given lengths, in the sense of Springborn (arXiv:1707.06848,
Def. 10.1): two metrics on a fixed triangulation are discretely conformally
equivalent iff

    log chord_ij - log ell_ij = (u_i + u_j)/2      for every edge ij,

for some vertex scaling u.  Theorem 10.5 guarantees the round realisation
exists and is unique up to Möbius whenever the given metric is a genuine
piecewise-Euclidean sphere metric (positive curvature at every vertex, which
is what a convex inscribed polyhedron forces).

The criterion used here is exact and cheap: for a candidate configuration the
numbers a_ij = log chord_ij - log ell_ij must lie in the image of the edge-
vertex incidence map A (A[(ij),k] = 1/2 if k in {i,j}).  So the residual

    r(x) = (I - A A^+) a(x)

vanishes exactly when x is a round realisation, and its norm is the natural
objective to minimise over x.  `conformal_residual` computes it;
`round_realise` minimises it by Gauss-Newton on the sphere (normalising after
each step), with a numerical Jacobian -- small meshes only, which is the
regime the |P| = 4 work lives in.

Every function here is a *testable* object: a round metric has residual zero
at its own configuration and at every vertex-scaling of it, and a metric with
a non-positive angle defect has no configuration with residual zero.
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


def conformal_residual(faces, lengths, x):
    """r = (I - A A^+) (log chord - log ell), and the projected part's norm.

    `lengths` is a dict {(i, j): ell}.  Returns (residual per edge, edges,
    the norm of the part that no vertex scaling can explain).
    """
    A, E = incidence(faces, len(np.asarray(x)))
    chord = chordal_lengths(x)
    a = np.array([np.log(chord[i, j]) - np.log(lengths[(i, j)]) for (i, j) in E])
    sol, *_ = np.linalg.lstsq(A, a, rcond=None)
    r = a - A @ sol
    return r, E, float(np.linalg.norm(r))


def round_realise(faces, lengths, x0=None, maxit=120, tol=1e-13, step=1.0,
                  seed=0):
    """Positions on S^2 with chordal metric discretely conformal to `lengths`.

    Gauss-Newton on the residual of `conformal_residual`, with a numerical
    Jacobian and a backtracking step; the configuration is renormalised onto
    the sphere after each step.  Returns (x, residual_norm, iterations).
    The Möbius/rotation gauge is not fixed -- `numpy.linalg.lstsq` takes the
    minimum-norm step through it.

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
    for it in range(maxit):
        r0 = residual(x)
        norm0 = float(np.linalg.norm(r0))
        if norm0 < tol:
            return x, norm0, it
        # numerical Jacobian of the residual with respect to x
        J = np.zeros((r0.size, 3 * n))
        h = 1e-7
        for k in range(3 * n):
            xp = x.copy()
            xp.reshape(-1)[k] += h
            xp /= np.linalg.norm(xp, axis=1)[:, None]
            J[:, k] = (residual(xp) - r0) / h
        delta, *_ = np.linalg.lstsq(J, -r0, rcond=None)
        accepted = False
        for _ in range(20):
            xt = (x.reshape(-1) + step * delta).reshape(x.shape)
            xt /= np.linalg.norm(xt, axis=1)[:, None]
            normt = float(np.linalg.norm(residual(xt)))
            if normt < norm0:
                x, best, accepted = xt, normt, True
                break
            step *= 0.5
        if not accepted:
            break
        step = min(1.0, step * 1.5)
    return x, best, maxit
