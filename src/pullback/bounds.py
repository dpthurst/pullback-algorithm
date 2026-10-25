"""Upper bounds on the asymptotic energy ^E E^p_p by tuning the
conformal structure, and the p-critical value they give.

[PChar, Prop. 5.7] / [ConfDim, Prop. "energy"]: the asymptotic energy
^E E^p_p[pi, phi] does not depend on the choice of q-conformal structure
on G_0; E^p_p(phi^n_0; alpha) does.  Hence for *any* alpha > 0

    ^E E^p_p  <=  E^p_p(phi^n_0; alpha)^(1/n)          (plan item 180),

and tuning alpha can only improve the bound.  This module searches over
alpha by coordinate descent in log alpha (E^p_p is invariant under
scaling all of alpha, so one direction is flat), then re-evaluates at
the rational point it found, so the reported number is exact for
integral p (energies.Energy).  Every candidate alpha gives a valid
bound, so a failed or truncated search costs sharpness, never
correctness; the untuned structure is always among the candidates.

What this does NOT do is minimise over the homotopy class of phi, which
is what [Elast] §6 does for p = 2 and what makes [ConfDim]'s numbers for
the carpets possible.  Until that layer exists, these bounds are honest
but weak (see AGENTS.md A7).

The p-critical value of a bound is the largest p at which it is still
< 1: ^E E^p_p is non-increasing in p ([PChar, Prop. 6.10]), so a bound
crossing 1 at p_0 gives ARCdim <= p_0.  A p-obstruction, by contrast,
gives ARCdim >= p; that is a different module.
"""

from __future__ import annotations

from fractions import Fraction

from .energies import exponent
from .iterates import base_edges, fill_matrix, iterate_energy


def float_matrix(cover, phi, p, beta):
    """M as floats, for the search (the exact one is fill_matrix)."""
    M = fill_matrix(cover, phi, p, beta)
    return {l: {l2: float(a) for l2, a in row.items()} for l, row in M.items()}


def fill_max(cover, phi, n, p, beta):
    """max over base edges of Fill^p(phi^n_0), in floats."""
    M = float_matrix(cover, phi, p, beta)
    v = {l: 1.0 for l in M}
    for _ in range(n):
        v = {l: sum(a * v[l2] for l2, a in row.items()) for l, row in M.items()}
    return max(v.values()) if v else 0.0


FACTORS = (0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 0.96, 1.0, 1.04, 1.11, 1.25,
           1.43, 1.7, 2.0, 2.9)


def optimize_beta(cover, phi, n, p, beta0=None, sweeps=40, factors=FACTORS,
                  tol=1e-13):
    """Coordinate descent on log beta; returns (best_value, beta)."""
    if beta0 is None:
        beta0 = {e: 1.0 for e in base_edges(cover)}
    beta = dict(beta0)
    best = fill_max(cover, phi, n, p, beta)
    for _ in range(sweeps):
        improved = False
        for e in beta:
            cur = beta[e]
            for f in factors:
                beta[e] = cur * f
                v = fill_max(cover, phi, n, p, beta)
                if v < best * (1 - tol):
                    best, cur, improved = v, beta[e], True
            beta[e] = cur
        if not improved:
            break
    return best, beta


def rationalize(beta, denom=10 ** 6):
    """The search's floats as exact positive rationals (same ratios to
    1e-6).  Normalised to median 1, since only ratios matter."""
    out = {e: Fraction(float(v)).limit_denominator(denom) for e, v in beta.items()}
    vals = sorted(out.values())
    scale = vals[len(vals) // 2]
    if scale > 0:
        out = {e: v / scale for e, v in out.items()}
    return out


def bounding_lengths(cover, phi, n, p, beta0=None, sweeps=40):
    """(Energy, beta_exact): the best of the untuned structure and the
    rational rounding of the optimiser's point."""
    beta0 = rationalize({e: 1.0 for e in base_edges(cover)} if beta0 is None
                        else beta0)
    _, beta = optimize_beta(cover, phi, n, p, beta0, sweeps)
    best_e = iterate_energy(cover, phi, n, p, base_lengths=beta0)
    best_b = dict(beta0)
    cand = iterate_energy(cover, phi, n, p, base_lengths=rationalize(beta))
    if cand.value < best_e.value:
        best_e, best_b = cand, rationalize(beta)
    return best_e, best_b


def upper_bound(cover, phi, p, n_max, beta0=None):
    """[(n, Energy E, Energy E^(1/n), beta)] for n = 1..n_max."""
    rows = []
    for n in range(1, n_max + 1):
        e, beta = bounding_lengths(cover, phi, n, p, beta0)
        rows.append((n, e, e.pow(Fraction(1, n)), beta))
        if _is_one(e):
            break
    return rows


def _is_one(e):
    v = e.exact_value() if e.exact else None
    return (v == 1) if v is not None else (e.value == 1)


def best_bound(cover, phi, p, n_max, beta0=None):
    """The sharpest row of upper_bound."""
    rows = upper_bound(cover, phi, p, n_max, beta0)
    return min(rows, key=lambda r: r[2].value)


def critical_p(cover, phi, p_lo, p_hi, n_max, beta0=None, tol=Fraction(1, 64)):
    """Bisect for the largest p at which the tuned bound is still < 1.

    Returns (p_star, row).  p_star is the first p found (to within tol)
    where the bound is < 1, i.e. ARCdim <= p_star is certified by it."""
    lo, hi = exponent(p_lo), exponent(p_hi)
    if not _below_one(best_bound(cover, phi, hi, n_max, beta0)[2]):
        return hi, best_bound(cover, phi, hi, n_max, beta0)
    if _below_one(best_bound(cover, phi, lo, n_max, beta0)[2]):
        return lo, best_bound(cover, phi, lo, n_max, beta0)
    while hi - lo > tol:
        mid = (lo + hi) / 2
        row = best_bound(cover, phi, mid, n_max, beta0)
        if _below_one(row[2]):
            hi = mid
        else:
            lo = mid
    return hi, best_bound(cover, phi, hi, n_max, beta0)


def _below_one(e):
    v = e.exact_value() if e.exact else None
    if v is not None:
        return v < 1
    return e.value < 1
