"""The exact dynamical orbifold of a graph virtual endomorphism.

WHAT THIS COMPUTES
------------------
For a Thurston map -- a postcritically finite branched self-cover
f : S^2 -> S^2 of degree d >= 2 -- Thurston attaches an *orbifold*
O_f = (S^2, nu_f).  The definition is [DH, Section 1]:

    nu_f is the smallest function nu : S^2 -> N* u {infty} such that
      (1) nu(x) = 1 when x is not postcritical, and
      (2) nu(x) is a multiple of nu(y) * deg_y(f) for every y with f(y) = x,

and the Euler characteristic is

    chi(O_f) = 2 - sum_{x in S^2} (1 - 1/nu(x)) = 2 - sum_{x in P} (1 - 1/nu(x)).

[DH] prove chi(O_f) <= 0 for a critically finite branched mapping and
classify the non-hyperbolic cases separately (DH Section 9); Milnor's
Cor. 4.5/4.8 give the same list from the flat-metric side.  Two
consequences worth stating, because they are easy to get wrong:

  * chi = 0 is *not* the same as "Lattes".  Summing 1/nu_x = n - 2 over the
    n = |P| terms gives exactly six signatures (see
    `test_orbifold.TestFlatSignatureEnumeration`): the four of [Mi,
    Cor. 4.5] -- (2,2,2,2), (2,4,4), (3,3,3), (2,3,6) -- and (2,2,inf) and
    (inf,inf), which are the Chebyshev-like and power maps of [Mi, Section
    6], not Lattes maps.
  * a computed chi > 0 is evidence that the portrait is *incomplete* -- P
    is missing a critical value, which for a polynomial means leaving out
    the critical fixed point at infinity.  The Riemann-Hurwitz certificate
    below catches the same fault independently, so the two never disagree;
    z^2+i with P = {0,i,-1+i,-i} fails both.

The "multiple of" in (2) is an lcm in practice, and since nu = 1 off P the
least solution is the fixed point of

    nu(x) = lcm { nu(y) * deg_y(f) : y in f^{-1}(x) },   nu = 1 off P.   (*)

Equivalently ([DH]; also Milnor's framing, and the "Ramification
Portraits" formulation) nu(x) is the lcm of deg_y(f^k) over all k >= 1 and
all y with f^k(y) = x.  Two facts make the finite computation exact (and the
second implementation in `test_orbifold.py` checks them against depth-lcm's):

  * nu(x) = infty  <=>  x lies on a cycle of f|_P containing a critical
    point.  (Iterating (*) around such a cycle multiplies nu by
    deg >= 2 each lap; and a chain ending at x can only go round a cycle
    that x is itself on, because a cycle point has a unique image.)
  * the finite coordinates never see an infinite one as a preimage, so the
    recursion restricted to the finite set is monotone and stabilises.

NO ANALYTIC DATA IS USED.  Everything below is read from the ribbon data
(base, cover, phi) through the portrait of [PChar] Section 2:

  * the punctures of the base are the faces of Gamma_0 -- the labels of P;
  * the punctures of the cover are the faces of Gamma_1, and the covering
    portrait `face_map_pi` gives, for each cover face y, the base face
    x = pi(y) and the local degree deg_y(f);
  * the marking P -> f^{-1}(P) is supplied by `face_map_phi`: the covered
    base face whose phi-image wraps base face C is the point C itself
    sitting in the cover, so its pi-target is f(C) and its pi-degree is
    the local degree deg_C(f).  This is exactly the read-off
    `classify.type_classification` already performs.

This is the same computation by which the author's Harmonic code labelled
the flower examples: session julia-4422 (quadratic, four postcritical
points, critical point preperiodic) comes out (4,4,2,2), the label in its
own `source` field, and julia-4442's claimed (4,4,4,2).

WHAT IS *NOT* DETERMINED BY A GRAPH VE
--------------------------------------
1. The marking.  Step 3 above needs each base face to have exactly one
   cover face whose phi-boundary wraps it.  When that fails -- the shipped
   sessions `grigorchuk` (base face 0 wrapped twice, base face 1 not at
   all) and `julia-4442` (no face wrapped at all) -- the ribbon data does
   not determine which cover face is "the point C", so the dynamics on P,
   and with it the orbifold, are not determined either.  This module
   refuses with the witness list instead of guessing.
2. The analytic Lattes structure, and flexible versus rigid.  The portrait
   decides *Lattes*: [Mi, Cor. 4.8] a four-point portrait is a Lattes one
   iff the four conditions in `lattes_report` hold.  It does **not** decide
   flexible versus rigid.  [Mi, Thm 1.1] writes every Lattes map as
   Theta o L o Theta^{-1} with G_n acting by rotation, n in {2,3,4,6};
   n >= 3 forces the torus ([Mi, Section 4]), hence rigid, but n = 2 allows
   any torus, and [Mi, Lemma 5.5] calls the map flexible exactly when the
   affine derivative alpha is an integer.  alpha = 1 + i on the square torus
   is a rigid Lattes map of degree 2 with the same (2,2,2,2) portrait as the
   flexible ones.  alpha (equivalently the homology matrix A_f of the torus
   lift, Douady-Hubbard Section 9) is not supplied by the portrait alone.
   The full topological class determines the flexible/rigid alternative via
   A_f, but this module does not yet compute that lift, so it reports
   `flexible = "undetermined"` exactly where it is needed.  The torus
   modulus is not combinatorial, and nothing here constructs a rational map.

WHAT THIS IS NOT
----------------
* It is not the "cone order from the valence of the spine" numerology: the
  flat angle a regular-dual realisation gives at a corner is
  pi(1 - 2/k) per incident k-gon (`ve_initial.obstruction_profile`
  computes those), a property of the chosen metric, not of f.  The
  orbifold order nu(x) is a ramification index of the dynamics and is
  computed from degrees along orbits.
* It is not the "local degree".  deg_x(f) is one arrow of the portrait;
  nu(x) is an lcm over the whole backward orbit.  For julia-4422 the two
  are (1,2,1,1) and (4,2,4,2) on the same four points.
* It is not the [PChar] "type".  That is the condition "each cycle in P
  contains a critical point" ([Thurston20:Characterize], quoted in
  [PChar] intro.tex), which is strictly stronger than chi(O_f) < 0 on the
  library: julia-4422 and twisted-z2i are "cycle without branch point"
  yet have chi = -1/2 < 0.  `orbifold()` reports both, side by side.

REFERENCES
----------
[DH]    A. Douady, J. H. Hubbard, *A proof of Thurston's topological
        characterization of rational functions*, Acta Math. 171 (1993);
        Section 1 (definition of O_f and chi(O_f)), Section 9
        (classification of the non-hyperbolic cases).
[Mi]    J. Milnor, *On Lattes maps*, arXiv:math/0402147: Thm 1.1
        (Theta o L o Theta^{-1} with G_n, n in {2,3,4,6}), Cor. 3.2/4.5
        (flat ramification indices are {2,2,2,2},{3,3,3},{2,4,4},
        {2,3,6}), Thm 4.1 with Cor. 4.8 (four postcritical points: Lattes
        iff every critical point is simple and none is postcritical;
        three postcritical points: type {3,3,3} iff every critical point
        has local degree 3 and none is postcritical), Lemma 5.5 (flexible
        iff the affine derivative alpha is an integer, so the portrait does
        not decide flexible versus rigid), and the remark that
        "Lattes maps (and also the Chebyshev-like maps of Section 6) are
        the ones for which this metric has curvature zero".
[PChar] D. Thurston et al., sources in `refs/confdim-src`, intro.tex
        (the hyperbolic-type hypothesis) and Section 1.2 (type
        classification; `classify.py`).
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd, inf, isqrt

from .portrait import face_map_phi, face_map_pi

__all__ = [
    "OrbifoldError", "portrait", "portrait_data", "orbifold",
    "lattes_report", "orbifold_of_session", "orbifold_guard", "guard_verdict",
    "describe", "LATTES_2222", "RIGID_LATTES",
]

LATTES_2222 = (2, 2, 2, 2)
#: [Mi, Cor. 4.5]: the four flat signatures.  (2,2,2,2) is the four-point
#: Lattes portrait, G_2 acting on a torus.  It is *not* the flexible family:
#: G_2 acts on any torus, but the affine map L need not be an integer
#: multiple of the identity, and [Mi, Lemma 5.5] makes a map flexible
#: exactly when n = 2 *and* the affine derivative alpha is an integer.  With
#: alpha non-real, e.g. alpha = 1 + i on the square torus (degree
#: |alpha|^2 = 2), the map is rigid and has the same (2,2,2,2) portrait.
#: So the portrait decides "Lattes portrait", not "flexible".  The full
#: topological class determines A_f, but that lift is not implemented here.
#: Signatures with n >= 3 are rigid: [Mi, Section 4] shows the torus is then
#: uniquely determined by n, so there is nothing to deform.
RIGID_LATTES = ((2, 4, 4), (3, 3, 3), (2, 3, 6))


class OrbifoldError(ValueError):
    """The ribbon data does not determine the dynamics on P."""


# ---------------------------------------------------------------------------
# 1. the portrait, with the marking
# ---------------------------------------------------------------------------

def portrait(base, cover, phi):
    """The ramification portrait of (pi, phi) on the punctures, with marking.

    Returned (all indices are face indices, i.e. puncture labels):

    ``n``          number of punctures of the base = |P|;
    ``preimages``  x -> list of (degree of f at y, y's base label or None),
                   one entry per face y of the cover over x; the entry
                   carries the base label exactly when y *is* a point of P
                   (the marked face of `phi`);
    ``marked``     C -> the cover face that is the point C;
    ``dynamics``   C -> f(C), read off the marked face of C;
    ``local_degree`` C -> deg_C(f), likewise;
    ``degree``     d, the covering degree (checked to be x-independent);
    ``certificates`` the exact checks listed at the end of this function.

    Raises `OrbifoldError` when the marking is not determined by the data.
    """
    prows, bwalks = face_map_pi(cover, base)
    frows, _ = face_map_phi(phi)
    n = len(bwalks)

    over = {}
    for i, r in enumerate(frows):
        if r["target"] is not None:
            over.setdefault(r["target"], []).append(i)
    if sorted(over) != list(range(n)) or any(len(v) != 1 for v in over.values()):
        counts = {C: len(v) for C, v in sorted(over.items())}
        raise OrbifoldError(
            "the marked face of each puncture is not singled out by phi: "
            f"base face -> number of wrapped cover faces {counts} "
            f"(want one for each of 0..{n - 1}); "
            "the ribbon data does not determine the dynamics on P")

    marked = {C: v[0] for C, v in over.items()}
    on_cover = {f: C for C, f in marked.items()}

    preimages, totals = {}, {}
    for i, r in enumerate(prows):
        if r["target"] is None:
            raise OrbifoldError(
                f"cover face {i} does not project to a power of a base face: "
                "not a covering (data error, not a convention)")
        preimages.setdefault(r["target"], []).append((r["degree"], on_cover.get(i)))
        totals[r["target"]] = totals.get(r["target"], 0) + r["degree"]
    if sorted(preimages) != list(range(n)):
        raise OrbifoldError(f"not every puncture has a preimage: {sorted(preimages)}")

    degrees = set(totals.values())
    if len(degrees) != 1:
        raise OrbifoldError(f"the covering degree is not constant over P: {totals}")
    d = degrees.pop()

    dynamics = {C: prows[marked[C]]["target"] for C in range(n)}
    local_degree = {C: prows[marked[C]]["degree"] for C in range(n)}

    # Riemann-Hurwitz for f itself, read on level 1: every critical point of
    # f lies over P (P contains the critical values), so
    #   sum over f^{-1}(P) of (deg - 1) = sum over Crit of (deg - 1) = 2d - 2.
    rh_lhs = sum(deg - 1 for x in preimages for (deg, _) in preimages[x])
    n_pre = sum(len(v) for v in preimages.values())
    cert = {
        "degree": d,
        "degree_constant_over_P": True,
        "sum_deg_minus_one": rh_lhs,
        "riemann_hurwitz": rh_lhs == 2 * d - 2,
        "n_preimages": n_pre,
        # The *number* of preimages is forced too: f restricts to a genuine
        # (unbranched) covering of the punctured sphere S^2 \ P, of degree d,
        # so 2 - |f^{-1}P| = d(2 - n), i.e. |f^{-1}P| = d(n-2) + 2 for n >= 3.
        # Milnor states the same count as |f^{-1}P \ P| = (n-2)(d-1).
        "forced_n_preimages": d * (n - 2) + 2 if n >= 3 else None,
        "preimage_count": (n < 3) or n_pre == d * (n - 2) + 2,
        "n_critical_preimages": sum(1 for x in preimages
                                    for (deg, _) in preimages[x] if deg >= 2),
        "max_local_degree": max(deg for x in preimages
                                for (deg, _) in preimages[x]),
        "marking_injective": len(set(marked.values())) == n,
    }
    if not cert["riemann_hurwitz"]:
        raise OrbifoldError(
            "portrait fails Riemann-Hurwitz: sum(deg-1) = "
            f"{rh_lhs} != 2d-2 = {2 * d - 2}.  Either some critical value is "
            "not a puncture, or the cover data is wrong.")
    if not cert["preimage_count"]:
        raise OrbifoldError(
            f"portrait has {n_pre} preimages of P but a degree-{d} cover of "
            f"the {n}-punctured sphere has d(n-2)+2 = {d * (n - 2) + 2}")
    return {
        "n": n, "d": d, "preimages": preimages, "marked": marked,
        "dynamics": dynamics, "local_degree": local_degree,
        "certificates": cert, "base_walks": [tuple(dd[0] for dd in w)
                                             for w in bwalks],
    }


def portrait_data(n, dynamics, local_degree, preimages, check=False):
    """A portrait written out by hand, for tests and for worked examples.

    `preimages[x]` is the list of (degree, base label or None) over x, in
    the format `portrait` returns.  `check=True` runs the same exact
    checks (degree, Riemann-Hurwitz) as `portrait`.
    """
    dynamics = dict(dynamics)
    local_degree = dict(local_degree)
    pre = {x: list(v) for x, v in preimages.items()}
    if sorted(pre) != list(range(n)):
        raise OrbifoldError(f"preimages must be given for every puncture 0..{n - 1}")
    totals = {x: sum(deg for (deg, _) in pre[x]) for x in pre}
    degrees = set(totals.values())
    if len(degrees) != 1:
        raise OrbifoldError(f"degree not constant over P: {totals}")
    d = degrees.pop()
    out = {
        "n": n, "d": d, "preimages": pre,
        "marked": {C: None for C in range(n)},
        "dynamics": dynamics, "local_degree": local_degree,
        "base_walks": [],
        "certificates": {"degree": d, "degree_constant_over_P": True,
                         "sum_deg_minus_one":
                             sum(deg - 1 for x in pre for (deg, _) in pre[x]),
                         "riemann_hurwitz": None, "n_preimages":
                             sum(len(v) for v in pre.values()),
                         "forced_n_preimages": d * (n - 2) + 2 if n >= 3 else None,
                         "preimage_count": None,
                         "n_critical_preimages":
                             sum(1 for x in pre for (deg, _) in pre[x] if deg >= 2),
                         "max_local_degree": max(deg for x in pre for (deg, _) in pre[x]),
                         "marking_injective": None},
    }
    cert = out["certificates"]
    cert["riemann_hurwitz"] = cert["sum_deg_minus_one"] == 2 * d - 2
    cert["preimage_count"] = (n < 3) or (cert["n_preimages"]
                                        == cert["forced_n_preimages"])
    if check and not cert["riemann_hurwitz"]:
        raise OrbifoldError("hand portrait fails Riemann-Hurwitz: "
                            f"{cert['sum_deg_minus_one']} != {2 * d - 2}")
    if check and not cert["preimage_count"]:
        raise OrbifoldError("hand portrait has the wrong number of preimages: "
                            f"{cert['n_preimages']} != {cert['forced_n_preimages']}")
    return out


# ---------------------------------------------------------------------------
# 2. the orbifold
# ---------------------------------------------------------------------------

def _cycles(dyn, n):
    """The cycles of the functional graph dyn on 0..n-1, each a list."""
    cycles, state = [], [0] * n          # 0 unvisited, 1 on stack, 2 done
    for start in range(n):
        if state[start]:
            continue
        path, pos = [], {}
        x = start
        while state[x] == 0:
            state[x] = 1
            pos[x] = len(path)
            path.append(x)
            x = dyn[x]
        if x in pos:
            cycles.append(path[pos[x]:])
        for y in path:
            state[y] = 2
    return cycles


def orbifold(p):
    """Orders nu, signature, Euler characteristics and the class of O_f.

    Exact throughout.  `inf` (a float) is used only as the marker for an
    infinite order; every finite order is a Python int, and every
    characteristic a `Fraction`.
    """
    n, dyn, pre, locdeg = p["n"], p["dynamics"], p["preimages"], p["local_degree"]

    cycles = _cycles(dyn, n)
    infinite = set()
    for cyc in cycles:
        if any(locdeg[x] >= 2 for x in cyc):
            infinite.update(cyc)
    unbranched = [cyc for cyc in cycles if not any(locdeg[x] >= 2 for x in cyc)]

    nu = {x: (inf if x in infinite else 1) for x in range(n)}
    finite = [x for x in range(n) if x not in infinite]
    for _ in range(4 * n + 8):
        new = {}
        for x in finite:
            m = 1
            for (deg, C) in pre[x]:
                w = 1 if C is None else nu[C]
                if w is inf:
                    raise AssertionError(
                        f"point {x} has the infinite preimage {C}: impossible, "
                        "the infinite set is closed under f")
                m = m * w * deg // gcd(m, w * deg)
            new[x] = m
        if all(new[x] == nu[x] for x in finite):
            break
        nu.update(new)
    else:                                    # pragma: no cover - see notes
        raise AssertionError("the orbifold recursion did not stabilise")

    chi = Fraction(2) - sum((Fraction(1) if v is inf else 1 - Fraction(1, v))
                            for v in nu.values())
    # The repository's earlier bookkeeping (notes/open-items.md: "cone orders
    # 2,2,1,1, chi_orb = -1") treats the order-1 postcritical points as
    # punctures, i.e. sums 1/nu rather than 1 - 1/nu.  Reported for
    # reconciliation only: the curvature criterion uses `chi`.
    chi_p = Fraction(2) - sum(Fraction(0) if v is inf else Fraction(1, v)
                              for v in nu.values())
    signature = tuple(sorted((v for v in nu.values() if v != 1),
                             key=lambda v: (v is inf, v)))

    if chi > 0:
        kind = "spherical"
    elif chi == 0:
        kind = "parabolic"
    else:
        kind = "hyperbolic"
    return {
        "orders": nu, "signature": signature,
        "infinite": sorted(infinite),
        "chi": chi, "chi_punctured": chi_p, "kind": kind,
        "cycles": cycles, "unbranched_cycles": unbranched,
        "all_cycles_branched": not unbranched,
        "n": n, "d": p["d"],
        "certificates": dict(p["certificates"],
                             recursion_stable=True,
                             euler_identity=(
                                 chi == Fraction(4 - n) - chi_p)),
    }


# ---------------------------------------------------------------------------
# 3. the Lattes tests
# ---------------------------------------------------------------------------

def lattes_report(p, orb=None):
    """The exact combinatorial Lattes verdict -- and what it does *not* say.

    [Mi, Cor. 4.8] (with Thm 4.1): a map with exactly four postcritical
    points is Lattes iff its critical points are all simple (local degree
    two) and none is postcritical.  Read on the portrait that is four
    conditions, each checked separately here, on both sides of the
    equivalence:

    ``four_postcritical``      n = 4;
    ``all_critical_simple``    every local degree of f is at most 2, i.e.
                               no critical point has local degree > 2
                               (together with Riemann-Hurwitz this says
                               every critical point has local degree 2);
    ``no_critical_postcritical`` deg_x(f) = 1 for every x in P;
    ``preimage_decomposition`` f^{-1}(P) = Crit u P, i.e. every preimage of
                               a postcritical point is critical or
                               postcritical -- the characterisation Milnor
                               states with Thm 4.1 and Cor. 4.8, and uses for
                               nu >= 4 postcritical points;
    and then the two consequences that must follow, and do:

    ``orbifold_all_two``       every order is 2, i.e. signature (2,2,2,2);
    ``euler_zero``             chi(O_f) = 0.

    `lattes_portrait` is the conjunction of the first four: this is the
    four-point *Lattes portrait*, i.e. the Thurston class of a Lattes map.
    The signature test alone is *not* the verdict -- an orbifold can be
    (2,2,2,2) with a critical postcritical point only if that point's order
    is infinite, so the two agree for n = 4 -- but the conditions say which
    one failed, which is the difference between "not Lattes" and "not
    Lattes, and here is the critical point that is postcritical".

    `flexible` is False unless the four-point Lattes portrait holds.  When it
    holds, a non-square degree also rules flexibility out: [Mi, Lemma 5.5]
    requires an integer affine derivative alpha, hence d = |alpha|^2 = k^2.
    A square-degree four-point portrait returns "undetermined": a non-real
    alpha can give a rigid Lattes map with the same portrait, while an
    integer alpha gives the flexible family.  Their split is the homology
    matrix A_f of the torus lift (Douady-Hubbard Section 9), which this
    module does not yet compute.

    `rigid` is True for the three signatures with n >= 3 ([Mi, Section 4]:
    the torus is then uniquely determined, so the map cannot deform) and
    None for (2,2,2,2) and for everything else -- undetermined, not False.
    [Mi, Cor. 4.8] gives the {3,3,3} local-degree condition (every critical
    point of local degree 3, none postcritical); the {2,4,4} and {2,3,6}
    cases are reported from the signature alone, with `source_condition`
    None, because no local-degree characterisation of them is quoted here.
    """
    orb = orb or orbifold(p)
    cert = p["certificates"]
    checks = {
        "four_postcritical": p["n"] == 4,
        "all_critical_simple": cert["max_local_degree"] <= 2,
        "no_critical_postcritical": all(x == 1 for x in p["local_degree"].values()),
        "preimage_decomposition":
            cert["n_preimages"] == p["n"] + cert["n_critical_preimages"],
    }
    portrait_lattes = all(checks.values())
    sig = orb["signature"]
    rigid = True if sig in RIGID_LATTES else None
    square = isqrt(p["d"]) ** 2 == p["d"]
    if not portrait_lattes:
        flexible = False
        flexible_requires = "not the four-point Lattes portrait"
    elif not square:
        flexible = False
        flexible_requires = (
            f"degree {p['d']} is not a square; an integer affine derivative "
            "would require d = k^2")
    else:
        flexible = "undetermined"
        flexible_requires = (
            "the torus-lift homology matrix A_f: integer alpha is flexible, "
            "non-real alpha is rigid (not computed here)")
    return {
        "checks": checks,
        "lattes_portrait": portrait_lattes,
        "signature_is_2222": sig == LATTES_2222,
        "flexible": flexible,
        "flexible_requires": flexible_requires,
        "rigid": rigid,
        "source_condition": (
            "every critical point of local degree 3 and none postcritical"
            if sig == (3, 3, 3) else None),
        "orbifold_all_two": all(v == 2 for v in orb["orders"].values()),
        "euler_zero": orb["chi"] == 0,
        "kind": orb["kind"],
    }


# ---------------------------------------------------------------------------
# 4. the iteration gate
# ---------------------------------------------------------------------------

def orbifold_of_session(session):
    """(portrait, orbifold, lattes) for a `pullback.Session`."""
    p = portrait(session.base, session.cover, session.phi)
    orb = orbifold(p)
    return p, orb, lattes_report(p, orb)


def orbifold_guard(session):
    """The computed replacement for the declared class in `convergence.py`.

    ``orbifold_ok`` requires chi(O_f) < 0.  Parabolic orbifolds (including
    the four-point Lattes portrait) and spherical portraits are excluded,
    with the exact signature recorded.  chi > 0 should not occur on a
    complete portrait and is treated as a data fault.

    ``branched_cycles_ok`` and ``unbranched_cycles`` are additional dynamical
    information, NOT an exclusion: Thurston pullback still makes sense with
    an unbranched cycle.  Such a map may be obstructed, or its contraction
    behaviour may need a different analysis.  This is distinct from the
    [PChar] hypothesis that every cycle of P contain a critical point, which
    applies to the energy theorem rather than defining whether pullback
    exists.  julia-4422 and twisted-z2i therefore remain candidates.

    ``allowed`` means only that the hyperbolic-orbifold question may be
    asked.  It certifies neither convergence, a fixed point, nor absence of
    a Thurston obstruction.
    """
    try:
        p, orb, lat = orbifold_of_session(session)
    except OrbifoldError as exc:
        return {"allowed": False, "status": "undetermined",
                "reason": f"the orbifold is not determined by the ribbon data: {exc}"}
    return guard_verdict(p, orb, lat)


def guard_verdict(p, orb, lat):
    """The gate's verdict from a portrait and its orbifold (pure)."""
    orbifold_ok = orb["kind"] == "hyperbolic"
    branched_ok = orb["all_cycles_branched"]
    out = {
        "allowed": orbifold_ok,
        "orbifold_ok": orbifold_ok,
        "branched_cycles_ok": branched_ok,
        "status": orb["kind"],
        "signature": orb["signature"],
        "orders": orb["orders"],
        "chi": orb["chi"],
        "n": orb["n"],
        "degree": p["d"],
        "cycles": orb["cycles"],
        "all_cycles_branched": orb["all_cycles_branched"],
        "unbranched_cycles": orb["unbranched_cycles"],
        "lattes_portrait": lat["lattes_portrait"],
        "flexible": lat["flexible"],
        "lattes_checks": lat["checks"],
    }
    if orb["kind"] == "parabolic":
        out["reason"] = (
            f"parabolic orbifold, signature {_sig(orb['signature'])}, chi = 0: "
            "excluded from the contraction experiments")
        if lat["lattes_portrait"]:
            out["reason"] += (
                "; the four-point Lattes portrait holds (not by itself a "
                "rational-realisation certificate); flexible = "
                f"{lat['flexible']}: " + lat["flexible_requires"])
        elif orb["signature"] in RIGID_LATTES:
            out["reason"] += (
                "; rigid affine-quotient signature, distinct from the narrow "
                "four-point Lattes class")
        else:
            out["reason"] += (
                "; power/Chebyshev parabolic signature, not the narrow "
                "four-point Lattes class")
    elif orb["kind"] == "spherical":
        out["reason"] = (
            f"spherical orbifold, signature {_sig(orb['signature'])}, chi = "
            f"{orb['chi']} > 0: not the Thurston setting (and a complete "
            "portrait should not give chi > 0)")
    else:
        out["reason"] = (
            f"hyperbolic orbifold, signature {_sig(orb['signature'])}, chi = "
            f"{orb['chi']} < 0: the classical question applies")
    if not branched_ok:
        note = (
            f"P has a cycle with no branch point {orb['unbranched_cycles']}: "
            "Thurston pullback still makes sense, but the [PChar] "
            "hyperbolic-type hypothesis fails; rationality, obstruction and "
            "convergence are not decided by this gate")
        out["unbranched_cycle_note"] = note
        out["reason"] += "; note " + note
    return out


# ---------------------------------------------------------------------------
# 5. display
# ---------------------------------------------------------------------------

def _sig(sig):
    return "(" + ",".join("inf" if v is inf else str(v) for v in sig) + ")"


def describe(orb, lat=None):
    """One line: signature, chi, class, Lattes verdict."""
    bits = [f"|P|={orb['n']}", f"d={orb['d']}",
            f"signature={_sig(orb['signature'])}",
            f"chi={orb['chi']}", orb["kind"]]
    if lat is not None:
        if lat["lattes_portrait"]:
            bits.append("LATTES-4POINT (flexible undetermined)")
        elif lat["rigid"]:
            bits.append("rigid-Lattes-type")
        else:
            failed = [k for k, v in lat["checks"].items() if not v]
            bits.append("not-Lattes" + (f" (fails {','.join(failed)})" if failed else ""))
    if not orb["all_cycles_branched"]:
        bits.append("unbranched cycle")
    return " ".join(bits)
