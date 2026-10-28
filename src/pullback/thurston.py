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

from .moduli import cross_ratio
from .rational import RationalMap, mobius_conjugate

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

def period_three_centres():
    """The three period-3 superattracting centres: rabbit, corabbit, airplane.

    All three are roots of c^3 + 2c^2 + c + 1 = 0 (the same cubic that defines
    the rabbit parameter): each has a critical orbit of period 3 through 0.
    Returned as (name, c) sorted by argument, so the names are stable:
    rabbit and corabbit are the conjugate pair with Im != 0, the airplane the
    real one.
    """
    roots = mp.polyroots([mp.mpf(1), mp.mpf(2), mp.mpf(1), mp.mpf(1)],
                         maxsteps=200, extraprec=400)
    real = [r for r in roots if abs(mp.im(r)) < mp.mpf("1e-30")]
    other = sorted([r for r in roots if abs(mp.im(r)) >= mp.mpf("1e-30")],
                   key=lambda r: -mp.im(r))
    if len(real) != 1 or len(other) != 2:
        raise ArithmeticError("the period-3 cubic should have one real and two complex roots")
    return [("rabbit", other[0]), ("corabbit", other[1]), ("airplane", real[0])]


def centre_modulus(c):
    """The modulus of the postcritical configuration of z^2 + c."""
    f = RationalMap([mp.mpf(1), mp.mpf(0), c], [mp.mpf(1)])
    return modulus(postcritical_list(f))


def centre_table():
    """[(name, c, modulus)] for the three period-3 centres."""
    out = []
    for name, c in period_three_centres():
        out.append((name, c, centre_modulus(c)))
    return out


# ------------------------------------------------ strictly rational examples
#
# Every example above is a polynomial, and a polynomial's infinity is fixed and
# totally ramified: infinity is in P, it is a critical point, and it is its own
# critical value.  That is a special geometry -- the mesh always has a vertex at
# infinity, the preimage of P always contains infinity, and the cover always has
# a ramified sheet over a fixed point -- and a discretisation that only ever
# meets it is not tested.  The two maps below are proper rational maps (not
# polynomials), with exact data.


def strictly_rational_rabbit():
    """A proper rational map with the rabbit's portrait: M o (z^2 + c) o M^-1
    for M(z) = 1/(z + 1).

    It is not a polynomial (f(infinity) = 1/(c + 2), finite), and no point of P
    is infinity: P = {1, 1/(c+1), 1/(c^2+c+1), 0}.  So the mesh has no vertex at
    infinity, the cover has no fixed ramified point, and every statement about
    the pullback mesh has to be made without the polynomial's crutch.  Its
    portrait degrees are the rabbit's, [1, 1, 1, 1, 2, 2], which the tests check
    against the map rather than assume.
    """
    cc = rabbit_parameter()
    f = mobius_conjugate(RationalMap([mp.mpf(1), mp.mpf(0), cc], [mp.mpf(1)]),
                         mp.mpf(1), mp.mpf(0), mp.mpf(1), mp.mpf(1))
    return {"name": "rabbit-by-mobius", "map": f, "parameter": cc,
            "mobius": (mp.mpf(1), mp.mpf(0), mp.mpf(1), mp.mpf(1))}


def slit_rational_map():
    """(2 z^2 - 4)/z^2 = 2 - 4/z^2: strictly rational, exact rational
    coefficients, and genuinely not a polynomial.

    Postcritical set {2, 1, -2, infinity} -- four points: the critical points are
    0 and infinity, and their orbits are 0 -> infinity -> 2 -> 1 -> -2 -> 1.  So
    |P| = 4, the portrait degrees are [1, 1, 1, 1, 2, 2], and |f^{-1}(P)| = 6
    (two double preimages: 0 and infinity).  Neither critical point is fixed, so
    no Moebius conjugation makes it a polynomial.

    The cycle 1 <-> -2 contains no critical point, so its type is "cycle without
    branch point" -- z^2 + i's type, and the type recorded for julia-4442.  The
    pullback therefore does not contract on pi_1 and this map is *not* a
    convergence target; it is the strictly rational geometry test, with the
    property that every coefficient and every point of P is exact, and that
    infinity is a pole which is also a critical point.

    (For the record, an exactly known *hyperbolic* strictly rational map with
    |P| = 4 is not available here: that is the twisted-rabbit question, and its
    configuration is what the pullback is supposed to find.)
    """
    return RationalMap([mp.mpf(2), mp.mpf(0), mp.mpf(-4)],
                       [mp.mpf(1), mp.mpf(0), mp.mpf(0)])
