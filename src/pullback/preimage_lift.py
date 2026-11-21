"""Uniformize an assembled sphere cover and polish every labelled preimage.

``uniformize_cover`` is the actual CEPS interface.  The independent rational
map is *not* an input to that solve; it enters only afterwards as an optional
oracle for polishing and convergence measurements.

Chart convention
----------------
Positions are unit vectors in R^3.  Their chart coordinate is

    z = (x + i*y) / (1 - Z),

which is the inverse of :func:`spherical_ceps.stereo_reinsert`; the north pole
is ``mp.inf``.  A Möbius gauge must be fitted before comparing a CEPS result to
an independently given rational map.  ``lift_and_polish`` therefore accepts an
explicit ``gauge_map`` and never guesses that the two charts already agree.

Polishing uses the declared local degree as the root multiplicity.  At a root
of multiplicity k the step z-k*g/g' is quadratically convergent, while plain
Newton is only linear.  Finite and infinite targets use polynomial numerator
equations in direct or reciprocal coordinates, so poles and preimages at
infinity do not need a second algorithm.
"""
from __future__ import annotations

import math
import time

import mpmath as mp
import numpy as np

from . import spherical_ceps as sc


class BasinError(RuntimeError):
    """A guarded polish failed or its travel left the declared basin."""


def _cover_geometry(lift, lengths):
    cert = lift.get("cert")
    if cert is not None:
        if not cert.get("is_sphere", False):
            raise ValueError("CEPS adapter requires a connected spherical cover")
        if not cert.get("cover_ok", False) or not cert.get("fibre_ok", False):
            raise ValueError("cover/fibre certificate does not match the declared degree")
    faces = [tuple(v) for v, _labels in lift["tris"]]
    labels = lift["edges"]
    base_of = lift["base_of"]
    proj = lift["proj"]
    if sorted(proj) != list(range(len(proj))):
        raise ValueError("CEPS adapter expects compact vertex ids 0..n-1")
    ell = {}
    for lab, (i, j) in labels.items():
        be = tuple(base_of[lab])
        if be not in lengths:
            raise ValueError(f"no base length for lifted edge {lab!r}: {be!r}")
        value = float(lengths[be])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"invalid base length for {be!r}: {value!r}")
        e = sc.canonical_edge(i, j)
        previous = ell.get(e, value)
        if abs(previous - value) > 1e-12 * max(1.0, abs(value)):
            raise ValueError(f"inconsistent inherited length over {be!r}")
        ell[e] = previous
    return faces, ell


def _realisation_error(res, chart_tol):
    """Why a CEPS return is not an accepted spherical realisation, or ``None``."""
    if not getattr(res, "realizable", False):
        return "CEPS stopped on the clamped energy extension, not a realizable triangle metric"
    cert = res.certificate
    if not np.isfinite(res.kkt):
        return f"non-finite KKT residual {res.kkt}"
    if cert["sphere_err"] > 1e-9:
        return f"positions leave the unit sphere: {cert['sphere_err']}"
    if cert["min_pair_distance"] <= 0.0:
        return "collapsed lifted vertices"
    if cert["worst_support_margin"] <= 0.0:
        return f"no strictly convex supporting polyhedron: margin {cert['worst_support_margin']}"
    if not cert["orientation_consistent"] or not cert["oriented_closed_surface"]:
        return "final faces are not a consistently oriented closed surface"
    if not cert["all_labelled"]:
        return "the final chart does not use every lifted vertex"
    if (not math.isfinite(cert["flat_scaling_residual"])
            or cert["flat_scaling_residual"] > chart_tol):
        return (f"flat-chart scaling residual {cert['flat_scaling_residual']} "
                f"exceeds {chart_tol}")
    return None


def uniformize_cover(lift, lengths, *, base=None, base_positions=None,
                     special_vertex=None, chart_tol=1e-8, shortcuts=True, **options):
    """Run real CEPS and return every lifted vertex, grouped in its fibre.

    Each ``preimages[p]`` row is ``(lift_vertex, declared_local_degree, xyz)``.
    The analytic map used for an optional later oracle comparison plays no role
    here.  A low-KKT point is not accepted by itself: strict Euclidean
    realizability, finite on-sphere distinct positions, a strictly supporting
    polyhedron, closed oriented labelled faces, and a flat-chart scaling
    residual at most ``chart_tol`` are required too.

    Supplying ``base`` and ``base_positions`` first enables the exact two-branch
    root shortcut.  It is used only when the declared cyclic monodromy is
    compatible, with ``shortcut_error`` retained if certification fails and
    CEPS retrieves the case.  Set ``shortcuts=False`` to force a CEPS control;
    use :func:`shortcut_two_branch` to require the closed form.

    With ``special_vertex=None`` (the default), vertex ids are tried in order;
    both ``CEPSFailure`` and a returned invalid realisation are logged as failed
    attempts.  An explicit vertex preserves strict single-choice refusal.
    Extremely small positive margins are reported, not promoted to exact
    certificates.  CEPS and polish timings remain separate.
    """
    faces, ell = _cover_geometry(lift, lengths)
    n = len(lift["proj"])
    options = dict(options)
    shortcut_error = None
    if shortcuts and base is not None and base_positions is not None:
        from . import two_branch as tb
        try:
            return shortcut_two_branch(lift, base, base_positions, lengths)
        except tb.NotTwoPoint:
            pass
        except tb.ShortcutFailure as exc:
            shortcut_error = str(exc)
    candidates = list(range(n)) if special_vertex is None else [int(special_vertex)]
    t0 = time.perf_counter()
    attempts = []
    res = None
    for special in candidates:
        started = time.perf_counter()
        try:
            candidate = sc.run(faces, ell, special, **options)
        except sc.CEPSFailure as exc:
            attempts.append({"special_vertex": special, "ok": False,
                             "seconds": time.perf_counter() - started,
                             "error": str(exc)})
            continue
        issue = _realisation_error(candidate, chart_tol)
        if issue is not None:
            attempts.append({"special_vertex": special, "ok": False,
                             "seconds": time.perf_counter() - started,
                             "error": issue})
            continue
        res = candidate
        attempts.append({"special_vertex": special, "ok": True,
                         "seconds": time.perf_counter() - started})
        break
    elapsed = time.perf_counter() - t0
    if res is None:
        detail = "; ".join(f"{a['special_vertex']}: {a['error']}" for a in attempts)
        raise sc.CEPSFailure(f"no special vertex succeeded ({detail})")
    P = np.asarray(res.positions, dtype=float)
    if P.shape != (n, 3):
        raise ValueError(f"CEPS returned {P.shape}, expected {(n, 3)}")
    if not np.all(np.isfinite(P)):
        raise ValueError("CEPS returned non-finite positions")
    proj = lift["proj"]
    if sorted(res.vs) != [v for v in range(n) if v != special]:
        raise ValueError("unexpected CEPS disk vertex list")
    fibres = {}
    for u, p in proj.items():
        fibres.setdefault(p, []).append((int(u), int(lift["ld"][u]), P[u].copy()))
    out = {"positions": P, "preimages": fibres, "res": res,
           "faces_final": list(res.faces_final), "face_lists": faces,
           "lengths": ell, "uniformize_seconds": elapsed,
           "ceps_seconds": elapsed, "ceps_phase_seconds": dict(res.timing),
           "special_vertex": special, "special_attempts": attempts,
           "method": "ceps", "used_shortcut": False}
    if shortcut_error is not None:
        out["shortcut_error"] = shortcut_error
    return out


def shortcut_two_branch(lift, base, base_positions, lengths):
    """Exact ``w^d`` uniformisation when exactly two branch values are present.

    Riemann--Hurwitz forces total ramification at both values for any such
    sphere cover.  The declared local monodromies are checked to be inverse
    full cycles and the Galois deck action is transported through the declared
    lifted edges; profiles alone are never used to infer that structure.

    The return has the same positions/preimages convention as
    :func:`uniformize_cover`.  Geometry of the *inherited face list* is
    reported separately: correct roots can coexist with a wrapped or
    non-simplicial lift, so convexity of that input chart is not embedded as
    an algebraic acceptance gate.  Projection residual certifies membership
    of each root in its declared fibre, not the root labelling; the latter
    comes from the transported monodromy.  No synthetic CEPS ``res`` record is
    returned, because its chart/KKT/schema have different meanings here.
    """
    from . import two_branch as tb

    t0 = time.perf_counter()
    built = tb.exact_positions(base, lift["d"], lift, base_positions)
    certificate = tb.certificate(built, base, lift, base_positions)
    residual = certificate["projection_residual"]
    if not math.isfinite(residual) or residual > 1e-8:
        raise tb.ShortcutFailure(f"root projection residual {residual} exceeds 1e-8")
    if not certificate["fibre_ok"]:
        raise tb.ShortcutFailure("closed-form fibres do not satisfy the declared degrees")
    faces, ell = _cover_geometry(lift, lengths)
    P = np.asarray(built["positions"], dtype=float)
    n = len(lift["proj"])
    if P.shape != (n, 3) or not np.all(np.isfinite(P)):
        raise tb.ShortcutFailure("closed-form returned invalid positions")
    fibres = {}
    for u, p in lift["proj"].items():
        fibres.setdefault(p, []).append((int(u), int(lift["ld"][u]), P[u].copy()))
    elapsed = time.perf_counter() - t0
    result = {"positions": P, "preimages": fibres, "method": "two-branch-root",
              "faces_final": [tuple(v) for v, _labels in lift["tris"]],
              "face_lists": faces, "lengths": ell, "root_indices": built["index"],
              "certificate": certificate, "shortcut": built,
              "used_shortcut": True, "uniformize_seconds": elapsed,
              "ceps_seconds": elapsed, "ceps_phase_seconds": {"root_shortcut": elapsed}}
    return result


def _chart(p):
    """North-pole chart of a sphere vector; scalar chart values pass through."""
    if np.ndim(p) == 0:
        return mp.mpc(p)
    if len(p) != 3:
        raise ValueError("a sphere position must have three coordinates")
    x, y, z = (float(v) for v in p)
    if z > 0.0 and (x != 0.0 or y != 0.0):
        # (x+i*y)/(1-z) == (1+z)/(x-i*y) on the unit sphere.
        # The second form avoids cancelling 1-z near the north pole.
        return mp.mpc(1.0 + z) / mp.mpc(x, -y)
    den = 1.0 - z
    if den == 0.0:
        return mp.inf
    return mp.mpc(x / den, y / den)


def _position(z):
    """Unit-sphere point with north-pole chart ``z`` (the exact chart inverse)."""
    z = mp.mpc(z)
    if z == mp.inf:
        return np.array([0.0, 0.0, 1.0])
    r2 = abs(z) ** 2
    den = 1.0 + r2
    return np.array([float(2.0 * z.real / den),
                     float(2.0 * z.imag / den),
                     float((r2 - 1.0) / den)])


def _image(rational_map, z):
    return rational_map.f_infinity() if z == mp.inf else rational_map(z)


def _chordal(z, w):
    """Chordal distance between two chart values on the Riemann sphere."""
    if z == mp.inf and w == mp.inf:
        return mp.mpf(0)
    if z == mp.inf:
        return 2 / mp.sqrt(1 + abs(w) ** 2)
    if w == mp.inf:
        return 2 / mp.sqrt(1 + abs(z) ** 2)
    return (2 * abs(z - w) /
            mp.sqrt((1 + abs(z) ** 2) * (1 + abs(w) ** 2)))


def _residual(rational_map, w, target):
    """Chart-independent chordal image residual, including infinite values."""
    return _chordal(_image(rational_map, w), target)


def _largest_error(rational_map, positions, targets):
    return max((_residual(rational_map, _chart(p), targets[i])
                for i, p in enumerate(positions)), default=mp.mpf(0))


def _error_order(errors, *, max_pairs=4):
    """Fit log(e[k+1]) = log(C) + q*log(e[k]) on an asymptotic tail.

    The last at most ``max_pairs`` pairs with first error below 0.1 are used;
    at least three pairs are required.  Very small nonzero values are retained
    (they are the asymptotic data), and an intercept is fitted.  A one-step
    exact solve therefore reports no order rather than inventing one.
    """
    pairs = [(mp.mpf(errors[i]), mp.mpf(errors[i + 1]))
             for i in range(len(errors) - 1)
             if errors[i] > 0 and errors[i + 1] > 0 and errors[i] < mp.mpf("0.1")]
    pairs = pairs[-max_pairs:]
    if len(pairs) < 3:
        return None, {"order": None, "pairs": len(pairs), "reliable": False,
                      "reason": "fewer than three asymptotic nonzero pairs"}
    x = np.array([float(mp.log(a)) for a, _b in pairs])
    y = np.array([float(mp.log(b)) for _a, b in pairs])
    A = np.column_stack((np.ones(len(x)), x))
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    fit = A @ coef
    rms = float(np.sqrt(np.mean((y - fit) ** 2)))
    order = float(coef[1])
    info = {"order": order, "log_constant": float(coef[0]),
            "rms_log_residual": rms, "pairs": len(pairs),
            # A large log-residual is an explicit precision-floor warning,
            # not a licence to quote the fitted slope as an asymptotic order.
            "reliable": bool(rms <= 0.25 and 0.25 <= order <= 4.0),
            "errors": [[float(a), float(b)] for a, b in pairs]}
    return order, info


def _deriv(coeffs):
    n = len(coeffs) - 1
    return [c * (n - i) for i, c in enumerate(coeffs[:-1])]


def _eval(coeffs, z):
    out = mp.mpc(0)
    for c in coeffs:
        out = out * z + c
    return out


def _times_power(coeffs, power):
    return list(coeffs) + [mp.mpc(0)] * power


def _equation_coefficients(rational_map, target, reciprocal):
    """Polynomial G=A/B whose zeros are the requested preimages.

    In reciprocal coordinates x=1/z, f(1/x) is written without negative
    powers before differentiating.  At target infinity G=1/f, whose zeros
    are the finite poles; the reciprocal formula also covers z=infinity.
    """
    N, D = list(rational_map.num), list(rational_map.den)
    n, m = len(N) - 1, len(D) - 1
    t = mp.mpc(target)
    if not reciprocal:
        if target == mp.inf:
            return D, N
        return _polysub(N, [t * c for c in D]), D
    Nr, Dr = list(reversed(N)), list(reversed(D))
    q = m - n
    if target == mp.inf:
        # G=1/f(1/x)=x^(n-m)*Dr/Nr.
        if -q >= 0:
            return _times_power(Dr, -q), Nr
        return Dr, _times_power(Nr, q)
    if q >= 0:
        A = _polysub(_times_power(Nr, q), [t * c for c in Dr])
        return A, Dr
    A = _polysub(Nr, [t * c for c in _times_power(Dr, -q)])
    return A, _times_power(Dr, -q)


def _polysub(a, b):
    n = max(len(a), len(b))
    aa = [mp.mpc(0)] * (n - len(a)) + list(a)
    bb = [mp.mpc(0)] * (n - len(b)) + list(b)
    return [x - y for x, y in zip(aa, bb)]


def _equation(rational_map, z, target):
    reciprocal = z == mp.inf or (z != mp.inf and abs(z) > 1)
    x = mp.mpf(0) if z == mp.inf else (1 / z if reciprocal else z)
    A, B = _equation_coefficients(rational_map, target, reciprocal)
    av, bv = _eval(A, x), _eval(B, x)
    ap, bp = _eval(_deriv(A), x), _eval(_deriv(B), x)
    if bv == 0:
        g = mp.mpc(0) if av == 0 else mp.inf
        gp = mp.nan
    else:
        g = av / bv
        gp = (ap * bv - av * bp) / (bv * bv)
    return reciprocal, x, g, gp


def polish_preimages(rational_map, initial_positions, targets, local_degrees=None,
                     *, maxit=32, tol=1e-12, position_tol=1e-12,
                     basin_radius=0.25, backtracks=40, labels=None):
    """Per-vertex, declared-multiplicity Newton polish with a travel guard.

    ``basin_radius`` bounds the *sum of chordal displacements* travelled by
    each vertex, not merely its final residual.  It defaults to 0.25 and may be
    a scalar, a per-vertex sequence, or ``None`` to disable that additional
    guard.  Residual decrease alone cannot distinguish two correctly unramified
    preimages; callers must also retain the lifted adjacency/labels.  The bound
    rejects far-basin crossings, not every imaginable nearby sheet swap.

    The returned ``history`` records every iteration, per-vertex residual,
    local degree, accepted backtracking scale, rejected trial count, travel,
    and elapsed time.  A non-converged or floored vertex raises ``BasinError``.
    """
    targets = [mp.inf if t == mp.inf else mp.mpc(t) for t in targets]
    n = len(initial_positions)
    if n == 0 or n != len(targets):
        raise ValueError("positions and targets must have the same nonzero length")
    w = [_chart(p) for p in initial_positions]
    if local_degrees is None:
        degrees = [1] * n
    else:
        degrees = [int(k) for k in local_degrees]
        if len(degrees) != n or any(k < 1 for k in degrees):
            raise ValueError("local degrees must be positive integers per vertex")
    if labels is None:
        labels = list(range(n))
    if len(labels) != n:
        raise ValueError("labels must have the same length as positions")
    if basin_radius is None:
        radii = [mp.inf] * n
    elif np.ndim(basin_radius) == 0:
        radii = [mp.mpf(basin_radius)] * n
    else:
        radii = [mp.mpf(r) for r in basin_radius]
        if len(radii) != n:
            raise ValueError("basin_radius must be scalar or per-vertex")
    if any(r <= 0 for r in radii):
        raise ValueError("basin radii must be positive")

    old_dps = mp.mp.dps
    needed = int(math.ceil(-math.log10(float(position_tol)))) * max(degrees) + 25
    mp.mp.dps = max(old_dps, 50, needed)
    t0 = time.perf_counter()
    try:
        per_errors = [[_residual(rational_map, w[i], targets[i])] for i in range(n)]
        errors = [max(row[-1] for row in per_errors)]
        state = ["open"] * n
        last_step = [None] * n
        travel = [mp.mpf(0)] * n
        history = []
        for it in range(maxit + 1):
            for i in range(n):
                if state[i] != "open":
                    continue
                if per_errors[i][-1] == 0:
                    state[i] = "converged"
                elif (per_errors[i][-1] <= tol and last_step[i] is not None
                      and last_step[i] <= position_tol):
                    state[i] = "converged"
            if all(s == "converged" for s in state):
                # The batch maximum is a progress scalar, not a mathematical
                # error sequence: its argmax can change vertices.  Fit each
                # continuous per-vertex history, then quote the best-supported
                # one together with all the others.
                fits = [_error_order(row) for row in per_errors]
                candidates = [(i, order, fit) for i, (order, fit) in enumerate(fits)
                              if order is not None]
                if candidates:
                    i, order, fit = max(
                        candidates,
                        key=lambda row: (row[2]["reliable"], row[2]["pairs"],
                                         -row[2]["rms_log_residual"]))
                    fit = dict(fit, vertex=i,
                               per_vertex=[item[1] for item in fits])
                else:
                    order = None
                    fit = {"order": None, "pairs": 0, "reliable": False,
                           "vertex": None, "reason": "no vertex has three pairs",
                           "per_vertex": [item[1] for item in fits]}
                return {"positions": np.array([_position(z) for z in w]),
                        "chart_values": list(w), "residuals": errors,
                        "per_vertex_residuals": per_errors,
                        "iterations": len(history),
                        "polish_seconds": time.perf_counter() - t0,
                        "order": order, "order_fit": fit, "history": history,
                        "local_degrees": degrees, "travel": travel,
                        "vertex_state": state, "status": "converged"}
            if it == maxit:
                break
            rows = []
            for i, z in enumerate(w):
                if state[i] != "open":
                    rows.append({"vertex": i, "label": labels[i],
                                 "local_degree": degrees[i],
                                 "residual": per_errors[i][-1], "skipped": True})
                    continue
                reciprocal, x, g, gp = _equation(rational_map, z, targets[i])
                if not (mp.isfinite(g) and mp.isfinite(gp)) or gp == 0:
                    raise BasinError(
                        f"singular Newton equation at vertex {i} (residual {per_errors[i][-1]})")
                move = degrees[i] * g / gp
                if not mp.isfinite(move):
                    raise BasinError(f"non-finite Newton step at vertex {i}")
                accepted = None
                rejected = 0
                for trial in range(backtracks):
                    scale = mp.mpf(2) ** (-trial)
                    xn = x - scale * move
                    zn = mp.inf if reciprocal and xn == 0 else (1 / xn if reciprocal else xn)
                    if (reciprocal and abs(xn) <= position_tol
                            and _residual(rational_map, mp.inf, targets[i]) == 0):
                        # The desired preimage is the pole itself.  Preserve
                        # that exact algebraic label instead of returning a
                        # merely enormous finite chart coordinate after the
                        # image evaluation has reached its precision floor.
                        zn = mp.inf
                    en = _residual(rational_map, zn, targets[i])
                    dn = _chordal(z, zn)
                    if en < per_errors[i][-1] and travel[i] + dn <= radii[i]:
                        accepted = (zn, en, dn, scale)
                        break
                    rejected += 1
                if accepted is None:
                    raise BasinError(
                        f"line search or basin guard failed at iteration {it}, vertex {i}; "
                        f"residual {per_errors[i][-1]}, travel {travel[i]}, radius {radii[i]}")
                zn, en, dn, scale = accepted
                w[i] = zn
                per_errors[i].append(en)
                last_step[i] = dn
                travel[i] += dn
                rows.append({"vertex": i, "label": labels[i],
                             "local_degree": degrees[i], "residual": en,
                             "step": dn, "travel": travel[i], "scale": scale,
                             "rejected_trials": rejected, "skipped": False})
            errors.append(max(row[-1] for row in per_errors))
            history.append({"iteration": it + 1, "max_residual": errors[-1],
                            "seconds": time.perf_counter() - t0, "vertices": rows})
        raise BasinError(f"no convergence after {maxit} iterations; "
                         f"residual {errors[-1]}, states {state}")
    finally:
        mp.mp.dps = old_dps


def lift_and_polish(rational_map, lift, lengths, targets, local_degrees,
                    *, special_vertex=None, gauge_map=None, maxit=32,
                    tol=1e-12, position_tol=1e-12, basin_radius=0.25,
                    base=None, base_positions=None, shortcuts=True,
                    **solve_options):
    """Run CEPS, optionally apply an explicitly fitted gauge, then polish."""
    solved = uniformize_cover(lift, lengths, base=base,
                              base_positions=base_positions,
                              special_vertex=special_vertex,
                              shortcuts=shortcuts, **solve_options)
    initial = (solved["positions"] if gauge_map is None
               else np.asarray(gauge_map.apply(solved["positions"])))
    polished = polish_preimages(
        rational_map, initial, targets, local_degrees=local_degrees,
        maxit=maxit, tol=tol, position_tol=position_tol,
        basin_radius=basin_radius)
    out = dict(solved)
    out.update(polished)
    out["ceps_positions"] = solved["positions"]
    out["positions"] = polished["positions"]
    return out
