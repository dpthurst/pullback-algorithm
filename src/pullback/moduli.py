"""The moduli chart for four points, and Möbius normalisation.

For a Thurston map with |P| = 4, Teichmüller space Teich(S^2 - P) is one
complex dimension (the author's decision of 2026-10-25: work with |P| = 4
first).  A configuration of four points on the Riemann sphere is recorded by
its cross-ratio, up to the six classical substitutions, and PSL(2, C) acts by
fractional linear maps; this module is the coordinate system the pullback
iteration will live in.

Conventions: for four distinct points,

    cr(a, b, c, d) = (a - c)(b - d) / ((a - d)(b - c)),

so the standard normalisation holds in the author's form: cr(inf, 0; 1, z) = z
(check the slot order -- with 0 and 1 first and infinity third the same
formula gives (z - 1)/z, which is one of the six substitutes and describes
the same configuration), and the four points are concyclic (or collinear)
iff the cross-ratio is real.  A Möbius map changes the value by one of the
six substitutions z, 1/z, 1-z, 1/(1-z), z/(z-1), (z-1)/z, so the chart is the
quotient of C - {0, 1} by S_3: two cross-ratios related by one of these
describe the same configuration.

Infinite entries are handled by the limits, not by floating point: a point of
P is frequently infinity (every polynomial has it), and `(inf - c)` is nan.
"""

from __future__ import annotations

import numpy as np

SIX = (
    lambda z: z,
    lambda z: 1.0 / z,
    lambda z: 1.0 - z,
    lambda z: 1.0 / (1.0 - z),
    lambda z: z / (z - 1.0),
    lambda z: (z - 1.0) / z,
)


def is_inf(z):
    z = complex(z)
    return bool(np.isinf(z.real) or np.isinf(z.imag))


def cross_ratio(a, b, c, d):
    """cr(a, b, c, d) = (a-c)(b-d)/((a-d)(b-c)); at most one entry may be infinite."""
    a, b, c, d = complex(a), complex(b), complex(c), complex(d)
    infs = [is_inf(v) for v in (a, b, c, d)]
    if sum(infs) > 1:
        raise ValueError("cross-ratio needs at most one infinite point")
    if infs[0]:                       # a = inf
        return (b - d) / (b - c)
    if infs[1]:                       # b = inf
        return (a - c) / (a - d)
    if infs[2]:                       # c = inf
        return (b - d) / (a - d)
    if infs[3]:                       # d = inf
        return (a - c) / (b - c)
    return ((a - c) * (b - d)) / ((a - d) * (b - c))


def normalise(points):
    """The fourth point after the Möbius map sending the first three to 0, 1, inf."""
    p0, p1, p2, p3 = points
    return cross_ratio(p0, p1, p2, p3)


def apply_mobius(m, z):
    """(a z + b)/(c z + d) for m = (a, b, c, d), with infinity respected."""
    a, b, c, d = m
    if is_inf(z):
        return complex(np.inf) if c == 0 else a / c
    z = complex(z)
    den = c * z + d
    return complex(np.inf) if den == 0 else (a * z + b) / den


def orbit(z):
    """The six cross-ratio values belonging to one configuration."""
    return [f(z) for f in SIX]


def same_configuration(z, w, tol=1e-9):
    """Do two cross-ratios describe the same four points up to Möbius?"""
    if abs(z) < tol or abs(z - 1.0) < tol:
        raise ValueError("degenerate cross-ratio")
    for v in orbit(z):
        if abs(v - w) < tol:
            return True
        if abs(v) > 1 / tol and abs(w) > 1 / tol and abs(1 / v - 1 / w) < tol:
            return True
    return False
