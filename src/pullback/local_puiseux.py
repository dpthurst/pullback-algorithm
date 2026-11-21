"""Local Puiseux expansions for chosen inverse branches of a rational map.

The only dependency is ``mpmath``.  Coefficient lists use the
``pullback.rational.RationalMap`` convention (highest power first), so pass
``m.num, m.den`` directly.  Building and evaluating an expansion restores the
caller's mpmath precision.

Fix ``b = f(p)`` and let ``k`` be the local degree at ``p``.  In the target
chart ``A(w) = w-b`` (finite ``b``) or ``A(w) = 1/w`` (infinite ``b``), and in
the source chart ``u = z-p`` (or ``u = 1/z`` at ``p = infinity``), write

    A(f(p+u)) = u**k * h(u),       h(0) != 0,
    tau = u * h(u)**(1/k),         tau**k = A(w).

Lagrange inversion gives the selected local inverse without Newton iteration:

    u = Phi(tau) = sum(c_n * tau**n, n >= 1),
    c_n = (1/n) * [u**(n-1)] h(u)**(-n/k).

For any ``1 <= k <= deg(f)``, :func:`seeds` enumerates the ``k`` branches by
rotating ``tau`` through all roots of unity.  At total ramification
``k = deg(f)`` these form the complete fibre.  For smaller ``k`` they form only
the cluster attached to this chosen centre; a whole nearby fibre requires the
other centres or another method.  Off ``b`` these are simple roots, so later
Newton polishing must declare multiplicity one, not ``k``.

Conservative checked domain
---------------------------
Critical values can bound an inverse branch, but they need not be singular on
the *selected* sheet.  Similarly, reaching ``f(infinity)`` need not send this
branch to source infinity: another finite preimage may be reached instead.
Consequently :func:`validity_radius` deliberately excludes every candidate
singularity and returns a **conservative lower bound** on the selected
branch's true radius, not an exact radius:

    S = { critical values c : c != b }  union  { f(q) },
    q = infinity for finite p, and q = 0 for p = infinity,
    R_tau = ( min over c in S of |A(c)| ) ** (1/k).

Here ``f(q)`` is the value at the pole of the source coordinate: source
infinity for finite ``p``, and source zero for ``p = infinity``.  A critical
value equal to ``b`` is omitted because the expansion is analytic at its
centre.  Exact zeros in the coefficient arithmetic are dropped; a small
nonzero candidate triggers one radius retry with 40 additional digits, and
persistent ambiguity is refused rather than rounded to zero.  Inside this
bound no candidate singularity occurs.  Outside it the series may converge or
diverge: refusal means only "not certified by this sufficient domain", never
an assertion of divergence.  ``radius_from_coefficients`` is an independent
estimate of the actual branch radius and can be larger.

Checked evaluation refuses an unavailable/unrequested bound and refuses
outside ``margin * R_tau`` with ``0 < margin <= 1``.  The named
``allow_unknown_radius=True`` bypass is for diagnostics only and records
``radius_checked=False``.  A one-term expansion corresponds to an affine
conjugacy to ``z**d``; a merely Moebius-conjugate two-branch map generally has
an infinite series and a finite escape bound.

This is a local analytic tool, not a global atlas, an unknown-cover
uniformizer, or a CEPS seed.  Equality and local-degree tests are numerical at
the working precision, not symbolic or interval certificates.
"""
from __future__ import annotations

import mpmath as mp

_INF = mp.inf
DEFAULT_DPS = 40


def _workdps(dps):
    """``mp.workdps`` that never lowers the caller's precision."""
    want = DEFAULT_DPS if dps is None else max(int(dps), 1)
    return mp.workdps(max(want, mp.mp.dps))


class LocalSeriesError(ValueError):
    """The local start does not exist, or must not be trusted."""


class AmbiguousCriticalValue(LocalSeriesError):
    """A nonzero candidate is too close to the centre for this precision."""


# ---------------------------------------------------------------------------
# truncated power series, LOWEST order first, held at length M+1
# ---------------------------------------------------------------------------
def _fit(a, M):
    a = list(a)[:M + 1]
    return a + [mp.mpc(0)] * (M + 1 - len(a))


def smul(a, b, M):
    out = [mp.mpc(0)] * (M + 1)
    for i, x in enumerate(a[:M + 1]):
        if x == 0:
            continue
        for j, y in enumerate(b[:M + 1 - i]):
            if y != 0:
                out[i + j] += x * y
    return out


def sdiv(a, b, M):
    """a/b truncated at M, low first; b[0] != 0."""
    if b[0] == 0:
        raise ZeroDivisionError("series division by a vanishing constant term")
    a = _fit(a, M)
    out = [mp.mpc(0)] * (M + 1)
    inv0 = 1 / b[0]
    for n in range(M + 1):
        s = a[n]
        for k in range(1, n + 1):
            if k < len(b):
                s -= b[k] * out[n - k]
        out[n] = s * inv0
    return out


def binom(E, alpha, M):
    """(1+E)^alpha for E[0] == 0, truncated at M."""
    E = _fit(E, M)
    total = _fit([mp.mpc(1)], M)
    pw = _fit([mp.mpc(1)], M)
    coef = mp.mpc(1)
    for n in range(1, M + 1):
        coef = coef * (alpha - (n - 1)) / n
        pw = smul(pw, E, M)
        for i in range(M + 1):
            if pw[i] != 0:
                total[i] += coef * pw[i]
    return total


def shift_poly(coef_high, p, n):
    """Low-first coefficients of q(p+u) up to degree n, q given highest first."""
    c = [mp.mpc(x) for x in reversed(coef_high)]          # low first, q(u)
    acc = [mp.mpc(0)] * (n + 1)
    row = [mp.mpc(1)]                                      # (u+p)^i, low first
    for i, ei in enumerate(c):
        if ei != 0:
            for j, rj in enumerate(row):
                acc[j] += ei * rj
        if i == len(c) - 1:
            break
        new = [mp.mpc(0)] * min(n + 1, len(row) + 1)
        for j, rj in enumerate(row):
            if rj == 0:
                continue
            new[j] += rj * p
            if j + 1 <= n:
                new[j + 1] += rj
        row = new
    return acc


def _order0(c):
    """(e, c shifted down by e) with (c/u^e)(0) != 0."""
    e = 0
    while e < len(c) and c[e] == 0:
        e += 1
    if e >= len(c):
        raise LocalSeriesError("coefficient series vanishes identically: raise "
                               "the truncation order")
    return e, c[e:]


# ---------------------------------------------------------------------------
# f(p+u) = u^E * a(u)/b(u),  a(0), b(0) != 0
# ---------------------------------------------------------------------------
def laurent_pair(num, den, p, M):
    """(E, a, b) with f(p+u) = u^E a(u)/b(u); a and b are low-first series."""
    num = [mp.mpc(x) for x in num]
    den = [mp.mpc(x) for x in den]
    if p == _INF:
        # f(1/u) = u^(degD-degN) N~(u)/D~(u) with N~(u) = u^degN N(1/u),
        # whose LOW-first coefficient list is the highest-first list of N.
        E0 = (len(den) - 1) - (len(num) - 1)
        A = [mp.mpc(x) for x in num]
        B = [mp.mpc(x) for x in den]
    else:
        E0 = 0
        A = shift_poly(num, p, M + 4)
        B = shift_poly(den, p, M + 4)
    ea, a = _order0(A)
    eb, b = _order0(B)
    return E0 + ea - eb, a, b


def local_degree_of(num, den, p, dps=None):
    """Local degree of f at p from the Laurent pair (no derivative needed)."""
    with _workdps(dps):
        p = _INF if p == _INF else mp.mpc(p)
        E, a, b = laurent_pair(num, den, p, 24)
        if E != 0:
            return abs(E)
        d = [a[i] - a[0] / b[0] * b[i] for i in range(min(len(a), len(b)))]
        k = 0
        while k < len(d) and d[k] == 0:
            k += 1
        return max(k, 1)


# ---------------------------------------------------------------------------
# the local series
# ---------------------------------------------------------------------------
def local_series(num, den, p, order, *, tol_digits=18, k=None, dps=None,
                 check_radius=True, chart="auto"):
    """Closed-form branch series at ``p``; see the module docstring.

    ``k`` may be supplied when the caller knows the local degree exactly; a
    mismatch with the computed leading order raises instead of returning
    coefficients for a point that is not exactly critical at this precision.

    ``chart``: ``"auto"`` uses the affine chart ``w = b`` unless ``p`` is
    exactly a pole (``D(p) = 0`` to the working precision of the coefficient
    arithmetic), ``"affine"`` forces ``w - b`` and ``"reciprocal"`` forces
    ``1/w``.  A float position for a pole never makes ``D(p)`` vanish exactly,
    so a caller who knows the point is a pole must say so.
    """
    with _workdps(dps):
        return _local_series(num, den, p, order, tol_digits, k, check_radius,
                             chart)


def _local_series(num, den, p, order, tol_digits, k_given, check_radius, chart):
    p = _INF if p == _INF else mp.mpc(p)
    M = order + 2
    E, a, b = laurent_pair(num, den, p, M + 4)
    if chart == "reciprocal" and E >= 0:
        reciprocal = True
    elif chart == "affine" and E < 0:
        raise LocalSeriesError("affine chart requested at a pole: f(p) is "
                               "infinite, use chart='reciprocal'")
    else:
        reciprocal = E < 0
    if reciprocal:
        # A o f = 1/f: one computation on the swapped pair.  b is infinity.
        E, a, b = laurent_pair(den, num, p, M + 4)
        b_sub = mp.mpc(0)
    else:
        b_sub = mp.mpc(0) if E > 0 else a[0] / b[0]
    num_f = [mp.mpc(0)] * (M + 1)
    for i, x in enumerate(a[:M + 1 - E] if E else a[:M + 1]):
        num_f[i + E] += x
    for i in range(min(len(b), M + 1)):
        num_f[i] -= b_sub * b[i]
    if E < 0:
        raise LocalSeriesError("internal: negative Laurent order after the "
                               "chart choice; raise the truncation order")
    scale = max([abs(x) for x in num_f] + [abs(x) for x in b[:M + 1]]
                + [mp.mpf(1)])
    thr_k = scale * mp.mpf(10) ** (-tol_digits)
    kk = 0
    while kk <= M and abs(num_f[kk]) <= thr_k:
        kk += 1
    if kk > M:
        raise LocalSeriesError(
            "no local degree found: A(f(p+u)) vanishes below "
            f"10^-{tol_digits} of the coefficient scale over the whole "
            "truncation order")
    leakage = (max([abs(num_f[j]) for j in range(kk)] or [mp.mpf(0)])
               / abs(num_f[kk]))
    h = sdiv(num_f[kk:], b, M + 1)
    b_val = _INF if reciprocal else b_sub
    k = int(kk)
    if k < 1:
        raise LocalSeriesError(f"local degree {k} < 1")
    if k_given is not None and int(k_given) != k:
        raise LocalSeriesError(
            f"declared local degree {k_given} != leading order {k} of "
            f"A(f(p+u)): p is not exactly a critical point at this precision")
    h0 = h[0]
    root = mp.exp(mp.log(h0) / k)               # principal k-th root
    Eser = [mp.mpc(0)] + [h[i] / h0 for i in range(1, M + 2)]
    P = [x / root for x in binom(Eser, mp.mpf(-1) / k, M)]
    coeffs = [mp.mpc(0)] * (order + 1)
    pw = _fit([mp.mpc(1)], M)
    for n in range(1, order + 1):
        pw = smul(pw, P, M)
        coeffs[n] = pw[n - 1] / n
    keep = coeffs[1:order + 1]
    scale = max([abs(x) for x in keep] or [mp.mpf(1)]) or mp.mpf(1)
    thr = scale * mp.mpf(10) ** (-tol_digits)
    tail = max([abs(x) for x in keep[1:]] or [mp.mpf(0)])
    out = {"p": p, "b": b_val, "k": k, "order": order, "coeffs": keep,
           "h": h[:order + 2], "h0": h0, "c1": keep[0],
           "tail_max": tail, "tail_threshold": thr,
           "exact": bool(tail <= thr),
           "leakage": leakage, "tol_digits": tol_digits, "dps": mp.mp.dps,
           "chart": "1/w" if b_val == _INF else "w-b",
           "b_is_infinite": b_val == _INF,
           "radius_kind": "conservative_lower_bound"}
    if check_radius:
        try:
            retries = 0
            try:
                R, info = validity_radius(num, den, p, b_val, k, dps=mp.mp.dps)
            except AmbiguousCriticalValue:
                # Repeated roots can leave artificial nonzero residue, while a
                # genuinely close critical value remains nonzero.  Resolve the
                # distinction with one bounded retry rather than guessing.
                retries = 1
                R, info = validity_radius(num, den, p, b_val, k,
                                          dps=mp.mp.dps + 40)
            out["radius_tau"] = R
            out["critical"] = info["rows"]
            out["radius_all_tau"] = info["R_all_tau"]
            out["radius_note"] = info["note"]
            out["radius_status"] = "known"
            out["radius_retries"] = retries
            out["radius_dps"] = info["dps"]
        except Exception as exc:                  # root finding can stall
            out["radius_tau"] = None
            out["critical"] = []
            out["radius_all_tau"] = None
            out["radius_status"] = "unavailable"
            out["radius_retries"] = retries
            out["radius_dps"] = None
            out["radius_note"] = (f"radius unavailable ({type(exc).__name__}: "
                                 f"{exc}); coefficients are unaffected, checked "
                                 f"evaluation refuses them")
    else:
        out["radius_tau"] = None
        out["critical"] = []
        out["radius_all_tau"] = None
        out["radius_status"] = "not_requested"
        out["radius_retries"] = 0
        out["radius_dps"] = None
        out["radius_note"] = "not computed"
    return out


# ---------------------------------------------------------------------------
# the validity radius from the critical values
# ---------------------------------------------------------------------------
def _polyder(c):
    n = len(c) - 1
    return [c[i] * (n - i) for i in range(n)] if n > 0 else [mp.mpc(0)]


def _polymul(a, b):
    out = [mp.mpc(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x == 0:
            continue
        for j, y in enumerate(b):
            if y != 0:
                out[i + j] += x * y
    return out


def _polysub(a, b):
    n = max(len(a), len(b))
    a = [mp.mpc(0)] * (n - len(a)) + list(a)
    b = [mp.mpc(0)] * (n - len(b)) + list(b)
    return [x - y for x, y in zip(a, b)]


def _polyval(c, z):
    out = mp.mpc(0)
    for x in c:
        out = out * z + x
    return out


def critical_points(num, den, dps=None):
    """Finite critical points (roots of N'D - N D') plus infinity if critical."""
    with _workdps(dps):
        num = [mp.mpc(x) for x in num]
        den = [mp.mpc(x) for x in den]
        wn = _polysub(_polymul(_polyder(num), den),
                      _polymul(num, _polyder(den)))
        out = []
        while len(wn) > 1 and wn[0] == 0:
            wn = wn[1:]
        if len(wn) > 1:
            try:
                roots = mp.polyroots(wn, maxsteps=4000, extraprec=200)
            except mp.libmp.NoConvergence:
                # A perfect power (two totally ramified points, i.e. the
                # two-branch case) stalls the iteration; retry at low precision
                # the way pullback.rational._polyroots does.
                with mp.workdps(min(mp.mp.dps, 30)):
                    roots = mp.polyroots(wn, maxsteps=4000, extraprec=200)
            out = [mp.mpc(z) for z in roots]
        if local_degree_of(num, den, _INF, dps=mp.mp.dps) >= 2:
            out.append(_INF)
        return out


def _pole_value(num, den):
    """f(infinity) as a point of the sphere."""
    if len(num) > len(den):
        return _INF
    if len(num) < len(den):
        return mp.mpc(0)
    return num[0] / den[0]


def _zero_value(num, den):
    """f(0) as a point of the sphere, including removable Laurent factors."""
    E, a, b = laurent_pair(num, den, mp.mpc(0), 6)
    if E < 0:
        return _INF
    if E > 0:
        return mp.mpc(0)
    return a[0] / b[0]


def _A_of_point(num, den, q, b):
    """A(f(q)) as a difference of polynomials, inf-safe.

    Finite b:  (N(q) - b D(q)) / D(q)
    b = inf :  D(q) / N(q)
    Returns None when the value cannot be resolved (D(q) = 0 with finite b).
    """
    num = [mp.mpc(x) for x in num]
    den = [mp.mpc(x) for x in den]
    if q == _INF:
        c0 = _pole_value(num, den)
        if b == _INF:
            if c0 == _INF:
                return mp.mpc(0)
            if c0 == 0:
                return _INF
            return 1 / c0
        return _INF if c0 == _INF else c0 - b
    if b == _INF:
        nq = _polyval(num, q)
        if nq == 0:
            return _INF
        return _polyval(den, q) / nq
    dq = _polyval(den, q)
    if dq == 0:
        return None
    return (_polyval(num, q) - b * dq) / dq


def validity_radius(num, den, p, b=None, k=None, dps=None, tol=None):
    """Conservative lower bound on the selected branch's convergence radius.

    Returns ``(R_tau, info)``; ``info["rows"]`` lists
    ``(critical point, chart value A(f(q)), |A|, label)``, ``R_all_tau`` is the
    same minimum *including* the dropped near-zero terms, and the coincident
    list records critical points whose value is ``b``.

    The returned disk excludes every candidate branch point and every possible
    escape pole, so it is a sufficient analytic-continuation domain.  Some
    candidates can be regular on the selected sheet: this bound need not be
    sharp, and exceeding it does **not** imply divergence.

    A critical value exactly equal to ``b`` in the coefficient arithmetic is
    dropped: it would put the candidate at ``tau = 0``, where this expansion is
    analytic.  A small but NONZERO difference is never rounded down.  If it is
    below the working-precision threshold, :class:`AmbiguousCriticalValue`
    marks it inconclusive instead of overstating it. :func:`local_series`
    retries once with 40 additional digits; persistent ambiguity still results
    in an unavailable radius rather than a guessed domain.
    """
    with _workdps(dps):
        p = _INF if p == _INF else mp.mpc(p)
        if b is None:
            E, a, bb = laurent_pair(num, den, p, 6)
            b = _INF if E < 0 else (mp.mpc(0) if E > 0 else a[0] / bb[0])
        if k is None:
            k = local_degree_of(num, den, p, dps=mp.mp.dps)
        scale = max([abs(mp.mpc(x)) for x in list(num) + list(den)] + [mp.mpf(1)])
        tol = tol if tol is not None else scale * mp.mpf(10) ** (-(mp.mp.dps - 18))
        rows = []
        R = _INF
        R_all = _INF
        coincident = []
        ambiguous = []
        for q in critical_points(num, den, dps=mp.mp.dps):
            A = _A_of_point(num, den, q, b)
            if A is None:                     # pole: |A| = inf in this chart
                rows.append((q, _INF, _INF, "unresolved"))
                continue
            mod = _INF if A == _INF else abs(A)
            R_all = min(R_all, mod)
            if mod <= tol:
                if mod == 0:
                    coincident.append((q, A))
                    rows.append((q, A, mod, "equals-b exactly (tau=0, dropped)"))
                else:
                    # A small NONZERO value difference is not a point error
                    # and cannot be rounded down: doing so raises the bound.
                    # Mark the whole domain inconclusive below.  Raising dps
                    # resolves the candidate honestly when it is distinct.
                    ambiguous.append((q, A, mod))
                    rows.append((q, A, mod, "near-b (nonzero, inconclusive)"))
                continue
            rows.append((q, A, mod, "other"))
            if mod < R:
                R = mod
        # A branch CAN have a pole at the pole of the SOURCE chart: f(infinity)
        # for finite p, or f(0) for p = infinity where u = 1/z.  Exclude the
        # corresponding target value to get a sufficient domain even though
        # this particular branch may instead reach a regular finite preimage
        # and remain analytic.
        source_pole = mp.mpc(0) if p == _INF else _INF
        c0 = _zero_value(num, den) if p == _INF else _pole_value(num, den)
        if c0 != _INF:
            if b == _INF:
                escape_A = 1 / c0 if c0 != 0 else _INF
            else:
                escape_A = c0 - b
            mod = abs(escape_A)
            if 0 < mod <= tol:
                ambiguous.append((source_pole, c0, mod))
                rows.append((source_pole, escape_A, mod,
                             "near-b escape (nonzero, inconclusive)"))
            else:
                if b != _INF and mod == 0:
                    mod = _INF
                R_all = min(R_all, mod)
                label = ("equals-b exactly (tau=0, dropped)" if mod == _INF
                         else "escape-to-source-pole")
                rows.append((source_pole, escape_A, mod, label))
                if mod < R:
                    R = mod
        if ambiguous:
            q, value, distance = ambiguous[0]
            raise AmbiguousCriticalValue(
                "nonzero critical/escape value difference "
                f"{mp.nstr(distance, 6)} at {q} (value {value}) is below the "
                f"equality threshold {mp.nstr(tol, 6)} at dps {mp.mp.dps}; "
                "cannot distinguish it from b without overstating the "
                "conservative radius. Increase dps")
        R_tau = _INF if R == _INF else R ** (mp.mpf(1) / k)
        R_all_tau = _INF if R_all == _INF else R_all ** (mp.mpf(1) / k)
        note = ("infinite: no candidate singularity remains in this chart"
                if R == _INF else "")
        info = {"rows": rows, "coincident": coincident, "k": k, "b": b,
                "radius_kind": "conservative_lower_bound",
                "R_w": R, "R_tau": R_tau, "R_all_w": R_all,
                "R_all_tau": R_all_tau, "tol": tol, "dps": mp.mp.dps,
                "note": note}
        return R_tau, info


# ---------------------------------------------------------------------------
# using the series
# ---------------------------------------------------------------------------
def chart_value(expansion, w):
    """A(w) = w-b (finite b) or 1/w (b = inf)."""
    b = expansion["b"]
    if b == _INF:
        if w == _INF:
            return mp.mpc(0)
        if w == 0:
            raise LocalSeriesError("target zero with infinite b: A(w) is "
                                   "infinite, outside every convergence radius")
        return 1 / mp.mpc(w)
    if w == _INF:
        raise LocalSeriesError("target at infinity with finite b: A(w) is "
                               "infinite, outside every convergence radius")
    return mp.mpc(w) - b


def evaluate(expansion, tau, n=None):
    """Phi(tau) with the first n coefficients (default: all)."""
    cs = expansion["coeffs"] if n is None else expansion["coeffs"][:n]
    u = mp.mpc(0)
    pw = mp.mpc(1)
    for c in cs:
        pw *= tau
        u += c * pw
    return u


def preimage(expansion, u):
    p = expansion["p"]
    if p == _INF:
        return _INF if u == 0 else 1 / u
    return p + u


def seeds(expansion, w, *, check_radius=True, margin=mp.mpf("0.5"), n=None,
          allow_unknown_radius=False):
    """The k preimages of ``w`` from a *built* expansion.

    Checked evaluation.  ``R_tau`` is a **conservative lower bound** on the
    selected branch's actual convergence radius, not an exact location of its
    first singularity.  ``check_radius=True`` (the default) raises
    ``LocalSeriesError`` in either of these cases:

    * ``radius_status != "known"`` -- the Wronskian root find failed, or the
      expansion was built with ``check_radius=False``.  No sufficient domain
      was computed, so returning positions would be an unverified claim.
    * ``|A(w)|^(1/k) > margin * R_tau`` -- outside the sufficient checked
      domain.  The series can still converge there; this refusal does not
      assert divergence and does not estimate the actual branch radius.

    ``margin`` must lie in ``(0, 1]``.  ``allow_unknown_radius=True`` is the
    only unknown-bound bypass.  It is a separate, greppable name because it
    disables a refusal; it records ``radius_checked=False`` on the expansion.
    ``check_radius=False`` is likewise an explicit diagnostic bypass, not a
    domain certificate.  Prefer a sufficient domain or an a posteriori
    residual/position certificate for production answers.
    """
    with _workdps(expansion.get("dps")):
        return _seeds(expansion, w, check_radius, margin, n, allow_unknown_radius)


def _seeds(expansion, w, check_radius, margin, n, allow_unknown_radius):
    if not 0 < margin <= 1:
        raise LocalSeriesError("margin must lie in (0, 1] for checked evaluation")
    k = expansion["k"]
    A = chart_value(expansion, w)
    T = mp.mpc(0) if A == 0 else mp.exp(mp.log(mp.mpc(A)) / k)
    checked = False
    if check_radius:
        status = expansion.get("radius_status")
        R = expansion.get("radius_tau")
        if status != "known" or R is None:
            if not allow_unknown_radius:
                raise LocalSeriesError(
                    f"conservative radius lower bound is {status or 'unknown'}, not known: "
                    f"checked evaluation refuses to place preimages. Rebuild "
                    f"with check_radius=True and a working critical-point solve, "
                    f"or pass allow_unknown_radius=True to skip the check "
                    f"explicitly")
            checked = False
        else:
            if R != _INF and abs(T) > margin * R:
                raise LocalSeriesError(
                    f"|A(w)|^(1/k) = {mp.nstr(abs(T), 6)} exceeds {margin} x "
                    f"the conservative convergence lower bound {mp.nstr(R, 6)}: "
                    f"outside the checked sufficient domain; this is not a "
                    f"divergence assertion")
            checked = True
    out = []
    for j in range(k):
        zeta = mp.exp(mp.mpc(0, 2 * mp.pi * j / k))
        out.append(preimage(expansion, evaluate(expansion, zeta * T, n=n)))
    expansion["radius_checked"] = checked
    return out, T


def leading_residual(expansion, n=None):
    """Leading coefficient of ``f(z_n) - w`` in units of ``T^(n+k)``.

    With ``Phi_n`` the truncation after n terms,
    ``tau(Phi_n) = tau + delta_{n+1} tau^(n+1) + ...`` and therefore
    ``f(z_n) - w = k delta_{n+1} T^(n+k) + O(T^(n+k+1))``.
    """
    k = expansion["k"]
    N = len(expansion["coeffs"]) if n is None else n
    h = expansion["h"]
    h0 = h[0]
    root = mp.exp(mp.log(h0) / k)
    Eser = [mp.mpc(0)] + [h[i] / h0 for i in range(1, N + 3)]
    g = [root * x for x in binom(Eser, mp.mpf(1) / k, N + 1)]
    phi = _fit([mp.mpc(0)] + expansion["coeffs"][:N], N + 1)
    acc = [mp.mpc(0)] * (N + 2)
    acc[0] = g[N] if N < len(g) else mp.mpc(0)      # sum_i g_{i-1} phi^i
    for i in range(N, 0, -1):
        acc = smul(acc, phi, N + 1)
        acc[0] += g[i - 1]
    acc = smul(acc, phi, N + 1)
    return k * acc[N + 1]


# ---------------------------------------------------------------------------
# target metric: the series lives in the chart, tolerances may not
# ---------------------------------------------------------------------------
def chordal(a, b):
    if a == _INF and b == _INF:
        return mp.mpf(0)
    if a == _INF:
        return 2 / mp.sqrt(1 + abs(b) ** 2)
    if b == _INF:
        return 2 / mp.sqrt(1 + abs(a) ** 2)
    return 2 * abs(a - b) / mp.sqrt((1 + abs(a) ** 2) * (1 + abs(b) ** 2))


def euclid_from_chordal(b, d):
    """Exact largest |w-b| over the chordal disk of radius ``d`` around finite b."""
    if b == _INF:
        if d <= 0:
            return _INF
        return mp.mpf(2) / d * mp.sqrt(max(mp.mpf(0), 1 - d ** 2 / 4))
    b = mp.mpc(b)

    def f(r):
        return 2 * r / mp.sqrt((1 + abs(b) ** 2) * (1 + (abs(b) + r) ** 2))

    lo, hi = mp.mpf(0), mp.mpf(1)
    while f(hi) < d:
        hi *= 2
        if hi > mp.mpf(10) ** 40:
            return _INF
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(mid) < d:
            lo = mid
        else:
            hi = mid
    return hi


def chordal_budget(b, R):
    """A chordal radius ``d`` whose whole disk stays inside ``|w-b| < R``."""
    b = mp.mpc(b)
    return 2 * R / mp.sqrt((1 + abs(b) ** 2) * (1 + (abs(b) + R) ** 2))


# ---------------------------------------------------------------------------
# a posteriori diagnostics
# ---------------------------------------------------------------------------
def resolvable(cs, floor=20):
    """Indices of coefficients that are not numerical zeros.

    An exact zero of the series is only noise-level in floating point
    (~10^-40 at dps 40), so terms far below *both* neighbours are dropped.
    A global threshold is wrong here: for a small radius the magnitudes
    legitimately span hundreds of orders of magnitude across the order.
    """
    out = []
    n = len(cs)
    for i, c in enumerate(cs):
        nb = max([abs(cs[j]) for j in (i - 1, i + 1) if 0 <= j < n] or [0])
        if abs(c) > nb * mp.mpf(10) ** (-floor):
            out.append(i)
    return out


def ratio_radius(expansion, floor=20):
    """``|c_n|^(-k/n)``: a radius estimate from the coefficients alone.

    Independent of the conservative candidate-value bound.  It estimates the
    selected branch's actual radius and can legitimately exceed ``radius_tau``.
    """
    k = expansion["k"]
    cs = expansion["coeffs"]
    keep = set(resolvable(cs, floor))
    return [_INF if i not in keep else abs(c) ** (-mp.mpf(k) / (i + 1))
            for i, c in enumerate(cs)]


def radius_from_coefficients(expansion, floor=20):
    """Radius estimated by the log-linear decay of the coefficients.

    ``|c_n| ~ C n^(-1-1/e) R_actual^(-n)`` for a nearest critical point of
    ramification ``e``, so the log-linear slope of ``log|c_n|`` against ``n``
    estimates ``-log R_actual`` up to ``O(1/n)``; the estimate is biased high
    by about ``exp((1+1/e)/n_avg)``.  This is a diagnostic, not a certified
    replacement for the conservative bound.  Returns ``(R_estimate, slope,
    points)``.
    """
    cs = expansion["coeffs"]
    keep = set(resolvable(cs, floor))
    pts = [(i + 1, mp.log(abs(c))) for i, c in enumerate(cs) if i in keep]
    pts = pts[len(pts) // 2:]
    if len(pts) < 4:
        return _INF, mp.mpf(0), 0
    nn = mp.mpf(len(pts))
    sx = sum(mp.mpf(n) for n, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(mp.mpf(n) ** 2 for n, _ in pts)
    sxy = sum(mp.mpf(n) * y for n, y in pts)
    slope = (nn * sxy - sx * sy) / (nn * sxx - sx * sx)
    return mp.exp(-slope), slope, len(pts)


def residual(num, den, w, z):
    """|f(z) - w| (inf-safe, not chordal)."""
    num = [mp.mpc(x) for x in num]
    den = [mp.mpc(x) for x in den]
    if z == _INF:
        fz = _INF if len(num) > len(den) else (
            mp.mpc(0) if len(num) < len(den) else num[0] / den[0])
    else:
        dq = _polyval(den, mp.mpc(z))
        fz = _INF if dq == 0 else _polyval(num, mp.mpc(z)) / dq
    if w == _INF:
        return mp.mpf(0) if fz == _INF else mp.mpf(2)
    if fz == _INF:
        return mp.mpf(2)
    return abs(fz - mp.mpc(w))
