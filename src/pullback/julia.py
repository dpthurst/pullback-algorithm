"""Escape-time pictures of a rational map on the Riemann sphere.

The kernel behind the viewer, deliberately toolkit-free: a grid of points on
S^2 (theta from the north pole, phi around) is carried to C by stereographic
projection, iterated, and the number of steps until it leaves a large disk is
recorded.  The sphere view then samples this texture through one analytic
ray-sphere intersection per pixel -- there is no ray tracing, no reflection
and no shading.

Conventions: z = cot(theta/2) * exp(i*phi), i.e. stereographic projection from
the north pole, which is sent to infinity; theta is therefore in the OPEN
interval (0, pi) and the two poles are not sampled (they are two points on a
grid of 10^4 -- the picture loses nothing, and the map need not extend over
them).

Numerics: numpy complex128 throughout.  This is a picture, not a
certificate; every number the app certifies comes from the exact engine, not
from here.
"""

from __future__ import annotations

import numpy as np

# A point is "escaped" once |f^k(z)| exceeds this.  For a polynomial of
# degree d >= 2 this is far outside the filled Julia set for the library's
# maps; for a rational map it is a display convention, not an invariant.
ESCAPE_RADIUS = 1e6


def parse_map(text, z=None):
    """Coefficient lists (num, den) of a rational map written as text.

    Accepts the session `map` strings as written: `^` for powers, implicit
    multiplication (`2z^2`, `4/27 (...)`) and `i` for the imaginary unit.
    Returns lists of complex coefficients, highest degree first.

    Raises ValueError for text that is not a rational function of z -- e.g.
    the rabbit session's `Douady rabbit z^2+c`, whose c is not a number.
    """
    from sympy import I, Poly, together
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )
    from sympy import symbols

    z = symbols("z") if z is None else z
    text = text.strip()
    if not text:
        raise ValueError("empty map")
    transforms = standard_transformations + (implicit_multiplication_application,
                                             convert_xor)
    try:
        expr = parse_expr(text, transformations=transforms,
                          local_dict={"z": z, "i": I}, evaluate=True)
    except Exception as exc:                                # noqa: BLE001
        raise ValueError(f"cannot read {text!r} as a function of z: {exc}") from exc
    if z not in expr.free_symbols:
        raise ValueError(f"{text!r} has no z in it")
    try:
        num, den = together(expr).as_numer_denom()
        num_c = [complex(c) for c in Poly(num, z).all_coeffs()]
        den_c = [complex(c) for c in Poly(den, z).all_coeffs()]
    except Exception as exc:                                # noqa: BLE001
        raise ValueError(f"{text!r} is not a rational function with numeric "
                         f"coefficients: {exc}") from exc
    if all(c == 0 for c in num_c):
        raise ValueError(f"{text!r} is identically zero")
    return num_c, den_c


def _polyval(coeffs, w):
    acc = np.zeros_like(w)
    for c in coeffs:
        acc = acc * w + c
    return acc


def _eval_rational(num, den, zz):
    """N(zz)/D(zz), with the limit at infinity taken by leading coefficients.

    Necessary because the Mobius chart sends ordinary points to infinity:
    polynomial evaluation there gives inf/inf = nan, which would be read as
    an escape (this made every theta cell "escape" before the fix).
    """
    zz = np.asarray(zz, dtype=complex)
    out = np.full(zz.shape, np.nan, dtype=complex)
    finite = np.isfinite(zz)
    if finite.any():
        w = zz[finite]
        n = _polyval(num, w)
        d = _polyval(den, w)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[finite] = np.where(d != 0, n / d, np.inf)
    if (~finite).any():
        if len(num) > len(den):
            lim = np.inf
        elif len(num) < len(den):
            lim = 0.0
        else:
            lim = complex(num[0]) / complex(den[0])
        out[~finite] = lim
    return out


def escape_counts(num, den, z, maxiter=200, radius=ESCAPE_RADIUS):
    """Steps until |f^k(z)| exceeds `radius`, `maxiter` where it never does.

    `num`, `den` are coefficient lists highest-first; `z` an array of complex
    points.  A vanishing denominator counts as escaped (the value is
    infinite), so a pole of f is drawn with the exterior.
    """
    z = np.asarray(z, dtype=complex)
    out = np.full(z.shape, maxiter, dtype=np.int32)
    alive = np.ones(z.shape, dtype=bool)
    w = z.copy()
    for k in range(maxiter):
        if not alive.any():
            break
        w = np.where(alive, w, 0)
        n = _polyval(num, w)
        d = _polyval(den, w)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            w = np.where(d != 0, n / d, np.inf)
            big = (~np.isfinite(w)) | (np.abs(w) > radius)
        hit = alive & big
        out[hit] = k + 1
        alive &= ~hit
    return out


def critical_points(num, den):
    """Finite critical points: roots of N' D - N D' (as numpy complex)."""
    N, D = np.array(num, dtype=complex), np.array(den, dtype=complex)
    coeffs = np.polysub(np.polymul(np.polyder(N), D), np.polymul(N, np.polyder(D)))
    if coeffs.size <= 1:
        return np.array([], dtype=complex)
    return np.roots(coeffs)


def postcritical(num, den, depth=40):
    """The finite points met by the critical orbits (capped at `depth`)."""
    out = []
    for c in critical_points(num, den):
        z = complex(c)
        for _ in range(depth):
            if not np.isfinite(z) or abs(z) > 1e12:
                break
            out.append(z)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                d = np.polyval(den, z)
                z = np.polyval(num, z) / d if d != 0 else complex(np.inf)
    return out


def mobius_centre(num, den):
    """A point a outside the postcritical set, for w = 1/(z - a).

    Escape time is a polynomial criterion: for a rational map whose
    attracting cycle contains infinity (theta's (1+z^2)/(1-z^2) has the
    superattracting 2-cycle infinity <-> -1), every Fatou point eventually
    passes near infinity and any radius test calls it "escaped".  Conjugating
    by w = 1/(z - a) moves the postcritical set -- hence the cycle -- into a
    bounded region, where the criterion is the usual one.
    """
    P = postcritical(num, den)
    finite = [abs(p) for p in P if np.isfinite(p)]
    return (max(finite) + 1.0) if finite else 1.0


def escape_counts_conjugated(num, den, z, a=None, maxiter=200,
                             radius=ESCAPE_RADIUS):
    """`escape_counts` after the Mobius change w = 1/(z - a).

    The Julia set is carried to the w-chart by the conjugacy, so the counts
    answer "does the orbit stay bounded" -- the question that is meaningful
    for a rational map -- rather than "does it leave a disk", which a map
    with an attracting infinity does not distinguish.
    """
    if a is None:
        a = mobius_centre(num, den)
    z = np.asarray(z, dtype=complex)
    out = np.full(z.shape, maxiter, dtype=np.int32)
    alive = np.ones(z.shape, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        w = 1.0 / (z - a)
    for k in range(maxiter):
        if not alive.any():
            break
        w = np.where(alive, w, 0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            zz = a + 1.0 / w
            f = _eval_rational(num, den, zz)
            wn = 1.0 / (f - a)
            big = (~np.isfinite(wn)) | (np.abs(wn) > radius)
        hit = alive & big
        out[hit] = k + 1
        alive &= ~hit
        w = wn
    return out


def attracting_cycles(num, den, maxiter=200, eps=1e-9, period_max=12):
    """Finite attracting cycles found from the critical orbits.

    A critical orbit of a hyperbolic map lands on an attracting cycle, so
    the cycles are read off the tail of each critical orbit (period <=
    `period_max`, two consecutive near-repetitions).  A cycle that contains
    infinity is not returned: that basin is the radius test in
    `basin_counts`.
    """
    cycles = []
    for c in critical_points(num, den):
        z = complex(c)
        orbit = []
        for _ in range(maxiter):
            if not np.isfinite(z) or abs(z) > 1e12:
                break
            orbit.append(z)
            z = complex(_eval_rational(num, den, np.array([z]))[0])
        else:
            for p in range(1, min(period_max, len(orbit) - 1) + 1):
                if (abs(orbit[-1] - orbit[-1 - p]) < eps
                        and abs(orbit[-2] - orbit[-2 - p]) < eps):
                    cyc = tuple(orbit[-p:])
                    key = tuple(sorted((round(v.real, 6), round(v.imag, 6))
                                       for v in cyc))
                    if key not in [tuple(sorted((round(v.real, 6), round(v.imag, 6))
                                               for v in other)) for other in cycles]:
                        cycles.append(cyc)
                    break
    return cycles


def basin_counts(num, den, z, maxiter=200, eps=1e-9, radius=ESCAPE_RADIUS,
                 cycles=None):
    """Steps until the orbit settles: onto an attracting cycle, or away.

    The kernel the viewer uses.  Escape time alone is a polynomial
    criterion: for a rational map whose attracting cycle contains infinity
    (theta's (1+z^2)/(1-z^2) has the superattracting 2-cycle
    infinity <-> -1) every Fatou point eventually passes near infinity, so a
    radius test calls the whole sphere "exterior"; and for a hyperbolic map
    with one attracting cycle the Julia set is a measure-zero boundary, so
    "did it escape" carries no shape at all.

    A point settles when it comes within `eps` of a point of an attracting
    cycle (`attracting_cycles`, computed once by the caller when passing a
    texture) or when |f^k(z)| exceeds `radius` -- an attracting infinity, or
    a pole.  What is still unsettled at `maxiter` is the Julia set; the
    number of steps is the depth of the basin, which is the picture.

    Settling is judged against *attracting* cycles on purpose: a bare
    near-repetition also holds at every repelling periodic point, which is
    the opposite of what the picture wants (`z = 1` is a repelling fixed
    point of z^2 and must stay unsettled).
    """
    z = np.asarray(z, dtype=complex)
    if cycles is None:
        cycles = attracting_cycles(num, den, maxiter=maxiter, eps=eps)
    pts = [complex(c) for cyc in cycles for c in cyc]
    out = np.full(z.shape, maxiter, dtype=np.int32)
    alive = np.ones(z.shape, dtype=bool)
    w = z.copy()
    for k in range(maxiter):
        if not alive.any():
            break
        w = _eval_rational(num, den, np.where(alive, w, 0))
        settled = (~np.isfinite(w)) | (np.abs(w) > radius)
        for c in pts:
            settled |= np.abs(w - c) < eps
        hit = alive & settled
        out[hit] = k + 1
        alive &= ~hit
        w = np.where(alive & np.isfinite(w), w, 0)
    return out


def sphere_grid(n_theta=128, n_phi=256):
    """Cell centres of an equiangular grid on S^2, and their z = cot(t/2)e^{iphi}."""
    theta = (np.arange(n_theta) + 0.5) * (np.pi / n_theta)
    phi = (np.arange(n_phi) + 0.5) * (2 * np.pi / n_phi)
    T, P = np.meshgrid(theta, phi, indexing="ij")
    return theta, phi, (1.0 / np.tan(T / 2.0)) * np.exp(1j * P)


def sphere_texture(num, den, n_theta=128, n_phi=256, maxiter=200):
    """Basin depths on the sphere grid: shape (n_theta, n_phi)."""
    _theta, _phi, Z = sphere_grid(n_theta, n_phi)
    return basin_counts(num, den, Z, maxiter=maxiter)


def plane_grid(xlim, ylim, nx=480, ny=480):
    """Cell centres of a rectangular grid in C."""
    x = np.linspace(xlim[0], xlim[1], nx)
    y = np.linspace(ylim[0], ylim[1], ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    return X + 1j * Y


def colourise(counts, maxiter=200, interior=(12, 12, 38)):
    """RGB image of a depth array: slow (deep, near the Julia set) is dark,
    quick is bright, and what never settled is the interior colour."""
    counts = np.asarray(counts)
    t = 1.0 - np.log1p(np.minimum(counts, maxiter)) / np.log1p(maxiter)
    rgb = np.empty(counts.shape + (3,), dtype=np.uint8)
    rgb[..., 0] = np.clip(255 * t ** 0.65, 0, 255)
    rgb[..., 1] = np.clip(190 * t ** 1.6, 0, 255)
    rgb[..., 2] = np.clip(70 + 150 * (1 - t) ** 2, 0, 255)
    inside = counts >= maxiter
    rgb[inside] = interior
    return rgb
