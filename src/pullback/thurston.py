"""M4: numeric postcritical configurations, starting with the rabbit.

At |P| = 4 the state of Thurston's pullback iteration is a cross-ratio
(`moduli.py`), and the iteration needs a target: the configuration of a known
rational map, which is a fixed point of the pullback map up to Möbius.  This
module computes such configurations from a map's coefficients and checks them
against the combinatorial portrait of a session.

The Douady rabbit is the workhorse.  Its parameter is the root of
c^3 + 2c^2 + c + 1 = 0 with positive imaginary part -- from the critical
orbit 0 -> c -> c^2+c -> 0, which gives (c^2+c)^2 + c = 0.  Its postcritical
set is {0, c, c^2+c, infinity}, four points, so Teich is one complex
dimension.

Vertex count of the pullback mesh (the author's correction, 2026-10-25): it
is d|P| *counting multiplicity*, but the distinct preimages are fewer,
because the critical values contribute one preimage each.  For the rabbit:
over 0 two simple preimages, over c one double (0), over c^2+c two simple,
over infinity one double (infinity) -- six distinct, not eight.  The session's
own portrait says the same thing: six cover faces with degrees
[1, 1, 1, 1, 2, 2].
"""

from __future__ import annotations

import mpmath as mp
import numpy as np

from .moduli import cross_ratio, is_inf
from .rational import RationalMap

mp.mp.dps = 40


def rabbit_parameter():
    """The Douady rabbit's c (Im > 0), as an mpmath mpf/mpc."""
    roots = mp.polyroots([mp.mpf(1), mp.mpf(2), mp.mpf(1), mp.mpf(1)],
                         maxsteps=200, extraprec=400)
    c = max(roots, key=lambda r: mp.im(r))
    if abs(c ** 3 + 2 * c ** 2 + c + 1) > mp.mpf("1e-30"):
        raise ArithmeticError("rabbit parameter failed its defining cubic")
    return c


def rabbit_map():
    c = rabbit_parameter()
    return RationalMap([mp.mpf(1), mp.mpf(0), c], [mp.mpf(1)])


def postcritical_list(m, depth=64, tol=mp.mpf("1e-25")):
    """The postcritical set as a list of finite mpmath points, plus infinity."""
    pts, has_inf, _tol = m.postcritical_set(depth=depth, tol=tol)
    out = [mp.mpc(p) for p in pts]
    if has_inf:
        out.append(mp.inf)
    return out


def preimage_profile(m, points):
    """Sorted local degrees of f over the points of P (counting multiplicities)."""
    degs = []
    for p in points:
        degs.extend(d for (_x, d) in m.preimage_with_degrees(p))
    return sorted(degs)


def distinct_preimages(m, points):
    """How many distinct points f^{-1}(P) has (the pullback mesh's vertices)."""
    return len(preimage_profile(m, points))


def _order_key(z):
    """Finite points first by (re, im), infinity last."""
    if not mp.isfinite(z):
        return (1, 0.0, 0.0)
    return (0, float(mp.re(z)), float(mp.im(z)))


def modulus(points):
    """Cross-ratio of four points, in a canonical order, at double precision.

    The order is canonical so that repeated calls agree; the value is only
    defined up to the six substitutions anyway (`moduli.same_configuration`).
    """
    if len(points) != 4:
        raise ValueError(f"need |P| = 4 for this chart, got {len(points)}")
    pts = sorted(points, key=_order_key)
    as_complex = [complex(np.inf) if not mp.isfinite(p) else complex(p) for p in pts]
    return cross_ratio(*as_complex)
