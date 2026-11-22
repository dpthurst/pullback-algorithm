"""Möbius-equivariant Springborn centration in Lorentz coordinates.

For three or more distinct points on ``S^d``, Springborn's theorem (stated as
Theorem 2 in Z. Langi, *Centering Koebe polyhedra via Möbius transformations*,
Groups Geom. Dyn. 15 (2021), 197--221, p.198) gives a Möbius transformation
carrying their barycentre to the origin, unique up to an orthogonal map.  This
is the intrinsic central placement used here.

It is *not* obtained by summing the coordinate-normalised null lifts ``(1,q)``:
those lifts acquire point-dependent scalings under a Möbius map, so that sum
is not Möbius-equivariant.  Instead solve the coupled equation for a point on
the positive hyperboloid, apply its inverse Lorentz boost to all input points,
and require zero barycentre.  Uniqueness up to rotation then gives the desired
equivariance.  The same transformation is applied to source and output, so
empty-circle Delaunay combinatorics is unchanged.  The remaining great-circle
source representative is a named convention, not itself a Möbius invariant.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares, root

__all__ = ["LORENTZ", "null_lift", "lorentz_centre", "centred_coordinates"]


LORENTZ = np.diag([-1.0, 1.0, 1.0, 1.0])
_E0 = np.array([1.0, 0.0, 0.0, 0.0])


def null_lift(points):
    """Lift unit sphere vectors to future null vectors in ``R^(1,3)``."""
    X = np.asarray(points, dtype=float)
    if X.size == 0:
        return np.empty((0, 4), dtype=float)
    X = np.atleast_2d(X)
    if X.shape[1] != 3 or not np.all(np.isfinite(X)):
        raise ValueError("expected finite three-dimensional sphere vectors")
    norms = np.linalg.norm(X, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("cannot lift a zero sphere vector")
    X = X / norms[:, None]
    return np.column_stack([np.ones(len(X)), X])


def _springborn_placement(points, tol=1e-12):
    """Solve for the inverse boost making the point barycentre zero."""
    Xi = null_lift(points)
    if len(Xi) < 3:
        raise ValueError("Springborn centration needs at least three points")

    def transformed_mean(r):
        centre = np.concatenate(([np.sqrt(1.0 + r @ r)], r))
        Z = (_lorentz_boost(centre) @ Xi.T).T
        return (Z[:, 1:] / Z[:, :1]).sum(axis=0)

    sol = root(transformed_mean, np.zeros(3), method="hybr", options={
        "maxfev": 1000, "xtol": 1e-13})
    residual = float(np.linalg.norm(sol.fun))
    if not np.isfinite(residual) or residual > tol:
        # A second Newton method can escape premature hybrid termination;
        # use a measured fallback, never accept a merely stationary iterate.
        ls = least_squares(transformed_mean, np.zeros(3), x_scale="jac",
                           max_nfev=1000, ftol=1e-14, xtol=1e-14, gtol=1e-14)
        if np.isfinite(ls.cost) and float(np.linalg.norm(ls.fun)) < residual:
            sol, residual = ls, float(np.linalg.norm(ls.fun))
    if not np.isfinite(residual) or residual > tol:
        raise ValueError(
            f"no Springborn zero-barycentre placement: {sol.message}; "
            f"residual {residual:.3e}")
    r = np.asarray(sol.x, dtype=float)
    centre = np.concatenate(([np.sqrt(1.0 + r @ r)], r))
    return centre, {"residual": residual, "message": str(sol.message),
                    "evaluations": int(sol.nfev),
                    "reported_success": bool(sol.success)}


def _lorentz_boost(centre):
    """Orthochronous Lorentz boost taking ``centre`` to the origin.

    Write ``centre = (cosh(t), sinh(t) n)``.  Apply the inverse rapidity
    boost in the ``(e0, n)`` plane.  Unlike a reflection about an affine
    midpoint, this preserves every future null vector, not just the centre.
    """
    if np.linalg.norm(centre - _E0) <= 1e-15:
        return np.eye(4)
    c0 = float(centre[0])
    r = np.asarray(centre[1:])
    rho = float(np.linalg.norm(r))
    if c0 <= 0.0 or rho <= 1e-15:
        raise ValueError("invalid hyperboloid centre")
    n = r / rho
    B = np.eye(4)
    B[0] = np.concatenate(([c0], -r))
    B[1:, 0] = -r
    B[1:, 1:] += (c0 - 1.0) * np.outer(n, n)
    return B


def centred_coordinates(source_points, output_points):
    """Apply the intrinsic Lorentz boost to source and output vectors.

    Returns ``(Y_source, Y_output, record)``.  ``record['boost']`` is the linear
    isometry used before independent null-vector normalisation.  ``record``
    contains numerical certificates of the null cone, Lorentz preservation,
    centre transport, and hull weights; these are invalidations of obvious
    formula errors, not numerical proofs about borderline Delaunay decisions.
    """
    Xi_q = null_lift(output_points)
    centre, solver = _springborn_placement(output_points)
    B = _lorentz_boost(centre)
    if not np.allclose(B @ centre, _E0, atol=1e-11, rtol=0.0):
        raise ValueError("the hyperbolic centre did not reach the origin")
    if not np.allclose(B @ LORENTZ @ B.T, LORENTZ, atol=1e-11, rtol=0.0):
        raise ValueError("the centration matrix is not Lorentz orthogonal")
    if np.linalg.det(B) < 0.0 or B[0, 0] < 0.0:
        raise ValueError("centration is not proper and orthochronous")

    Xi_x = null_lift(source_points)
    moved_q = (B @ Xi_q.T).T
    moved_x = (B @ Xi_x.T).T
    if np.any(moved_q[:, 0] <= 0.0) or np.any(moved_x[:, 0] <= 0.0):
        raise ValueError("centration left the future null cone")
    Yq = moved_q[:, 1:] / moved_q[:, :1]
    Yx = moved_x[:, 1:] / moved_x[:, :1]
    residual_q = np.abs((moved_q @ LORENTZ * moved_q).sum(axis=1))
    residual_x = np.abs((moved_x @ LORENTZ * moved_x).sum(axis=1))
    record = {
        "boost": B,
        "centre": centre,
        "solver": solver,
        "centred_barycentre": Yq.sum(axis=0),
        "ordinary_barycentre": Yq.sum(axis=0),
        "boost_centre_residual": float(np.linalg.norm(B @ centre - _E0)),
        "lorentz_residual": float(np.max(np.abs(B @ LORENTZ @ B.T - LORENTZ))),
        "null_residual": float(max(float(np.max(residual_q)),
                                   float(np.max(residual_x)) if residual_x.size else 0.0)),
        "boosted_sum": moved_q.sum(axis=0),
        "construction": "Springborn zero-barycentre Lorentz placement",
    }
    return Yx, Yq, record


def lorentz_centre(points):
    """Return ``Y, record`` for an ideal configuration alone."""
    Xi = null_lift(points)
    Y, _centre, record = centred_coordinates(Xi[:, 1:], Xi[:, 1:])
    return Y, record
