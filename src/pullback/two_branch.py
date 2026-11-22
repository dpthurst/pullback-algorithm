"""Exact shortcut for a degree-d cover with exactly two critical values.

Mathematical content
====================
Let ``pi : Sigma -> S^2`` be a degree-``d > 0`` cover of the sphere with exactly
two critical values (branch values) ``b0, b1``.  Riemann-Hurwitz with only two
critical values ``P, Q`` reads

    2d - 2 = (d - |P|) + (d - |Q|)     hence   |P| + |Q| = 2,

so *both* fibres are single points: "exactly two critical values" already forces
total ramification at both, and no other profile is possible (a [2,2]/[4]
degree-4 example cannot exist, for the same reason).  The monodromy around
``b0`` is then a d-cycle ``g`` and the monodromy around ``b1`` is ``g^-1`` (the
product of the two generators is trivial on the sphere with two punctures), so
the monodromy group is ``<g> = Z/d``, the cover is Galois cyclic, and the
quotient map is conjugate to ``w -> w^d``:

    pi = mu^-1 o (z -> z^d) o nu^-1 ,   mu(b0) = 0, mu(b1) = oo.

Hence every preimage is a d-th root: with ``nu`` absorbed into the free source
gauge,

    P_u = position( omega^{j(u)} * mu(z_{proj u})^{1/d} ),   omega = e^{2 pi i/d}

and the exact answer is a finite set of roots per fibre.  The only real content
is the labelling ``j(u)``, which must be *torsor compatible* with the cover's
monodromy.  Branch profiles do not determine it; and the assembler's own vertex
order is not the compatible numbering, which is why the naive family
"index order + per-base-vertex shift" misses the answer as soon as d >= 3.

Labelling rule (two independent pieces)
  * argument unwinding: on the cut base (1-skeleton minus a path from ``b0`` to
    ``b1``, and minus the two branch vertices themselves) the argument of
    ``mu(z_p)`` unwinds continuously; its winding ``s_p`` against the principal
    argument is an integer and only ``s_p mod d`` matters;
  * fibre transport: numbering each unramified fibre by the exponent ``i`` with
    ``g^i(x0) = x`` makes ``g`` the cyclic shift, which is the numbering in
    which the deck action is a rotation of the root index.

What this module measures and certifies
  * exact projection ``mu(P_u^d) = b_{proj u}`` in the chordal metric;
  * fibre completeness and the declared local degrees, including the two
    totally ramified preimages;
  * the cyclic/Galois deck structure (``g1 = g0^-1`` as permutations, group
    order d, orbits equal to fibres) -- strictly more than matching profiles;
  * numerically resolved integral edge windings and path-independent lifted
    transport, with ambiguous samples refused rather than silently rounded;
  * positive face orientation, total solid angle and supporting margins,
    reported separately.  ``uniformize_from_base`` requires convex inscription;
    the dict-returning algebraic adapter deliberately does not.

Projection residual proves fibre membership, **not root identity**: every root
of one base value leaves it unchanged.  Root labels come from the declared
monodromy and lifted transport, and can additionally be diagnosed by the
orientation data.  The shortcut delivers a Moebius class, not an isometry;
downstream lifting code must measure its own vertex-scaling relation.
"""
from __future__ import annotations

import math

import mpmath as mp
import numpy as np

from pullback import dense_assembly as A
from pullback import preimage_lift as PL


class NotTwoPoint(ValueError):
    """The lift is not a two-critical-value / total-ramification cover."""


class ShortcutFailure(RuntimeError):
    """The two-point structure is present but the closed form did not certify."""


# --------------------------------------------------------------- chart algebra
_chart = PL._chart
_position = PL._position


def _wrap(x):
    """Principal value in (-pi, pi]."""
    y = (x + mp.pi) % (2 * mp.pi) - mp.pi
    return mp.pi if y == -mp.pi else y


def _hom(z):
    return (mp.mpc(1), mp.mpc(0)) if z == mp.inf else (mp.mpc(z), mp.mpc(1))


def mobius_matrix(z0, z1):
    """Projective matrix of the chart Mobius map with z0 -> 0, z1 -> infinity."""
    a0, b0 = _hom(z0)
    a1, b1 = _hom(z1)
    det = a1 * b0 - a0 * b1
    if det == 0:
        raise ValueError("degenerate normalisation: equal branch values")
    return (b0 / det, -a0 / det, -b1 / det, a1 / det)


def mobius_eval(M, z):
    m00, m01, m10, m11 = M
    if z == mp.inf:
        num, den = m00, m10
    else:
        num, den = m00 * z + m01, m10 * z + m11
    if den == 0:
        return mp.inf
    return num / den


def mobius_to_zero_infinity(z, z0, z1):
    """Chart Mobius map sending z0 -> 0 and z1 -> infinity (infinity safe)."""
    return mobius_eval(mobius_matrix(z0, z1), z)


def root_family(z, d):
    """The d principal d-th roots of z; the d roots coincide at 0 and oo."""
    if z == mp.inf or z == 0:
        return [z] * d
    r = mp.mpf(abs(z)) ** (mp.mpf(1) / d)
    th = mp.arg(z) / d
    return [r * mp.exp(1j * (th + 2 * mp.pi * k / d)) for k in range(d)]


# ------------------------------------------------------------------- structure
def _cycle_lengths(p):
    seen, lengths = set(), []
    for start in range(len(p)):
        if start in seen:
            continue
        n, x = 0, start
        while x not in seen:
            seen.add(x)
            x = p[x]
            n += 1
        lengths.append(n)
    return sorted(lengths, reverse=True)


def corner_sheets(base, lift):
    """One sheet label per lifted vertex, read at a *fixed* incident base face.

    ``lift["sheets"][u]`` is the union over the whole flower and can be larger
    than the local degree at a cut vertex (``dense_assembly`` documents this).
    The per-face corner sets ``lift["per_face"][u][k]`` are the correct object:
    over a fixed face ``k`` they partition Z/d, so a lifted vertex of local
    degree 1 has exactly one sheet label there.
    """
    faces_at = {}
    for k, (verts, _labs) in enumerate(base.tris):
        for v in verts:
            faces_at.setdefault(v, []).append(k)
    k_of = {p: min(faces_at[p]) for p in faces_at}
    out = {}
    for u, p in lift["proj"].items():
        corner = lift["per_face"][u].get(k_of[p])
        out[u] = tuple(corner) if corner is not None else tuple(lift["sheets"][u])
    return out, k_of


def monodromy_group(g, d):
    ident, group, x = tuple(range(d)), {tuple(range(d))}, tuple(g)
    while x != ident:
        group.add(x)
        x = A.perm_compose(g, x)
    return group


def analyse(base, d, lift):
    """Exact structure of a two-critical-value cover, or raise ``NotTwoPoint``."""
    if d < 2:
        raise NotTwoPoint(f"degree {d} is not a nontrivial cover")
    profile = A.critical_profile(base, d, lift)
    branch = sorted(p for p, prof in profile.items() if prof != [1] * d)
    cert = lift["cert"]
    out = {"d": d, "profile": profile, "branch_values": branch,
           "n_branch": len(branch), "rh_deficit": cert["rh_deficit"],
           "rh_expected": cert["rh_expected"], "is_sphere": cert["is_sphere"],
           "connected": cert["connected"], "fibre_ok": cert["fibre_ok"],
           "counts": cert["counts"]}
    if len(branch) != 2:
        raise NotTwoPoint(f"{len(branch)} critical values over base vertices "
                          f"{branch}; the root shortcut needs exactly 2")
    if not cert["is_sphere"]:
        raise NotTwoPoint("the lift is not a connected combinatorial sphere")
    for p in branch:
        if profile[p] != [d]:
            raise NotTwoPoint(f"critical value {p} has profile {profile[p]}, "
                              f"not [{d}]: Riemann-Hurwitz forbids this")
    if out["rh_deficit"] != out["rh_expected"]:
        raise NotTwoPoint(f"Riemann-Hurwitz deficit {out['rh_deficit']} != "
                          f"{out['rh_expected']}")
    local = dict(cert.get("local_realised")
                 or A.local_perms(base, lift["edge_perms"], d))
    g0, g1 = tuple(local[branch[0]]), tuple(local[branch[1]])
    inv = [0] * d
    for i, j in enumerate(g0):
        inv[j] = i
    inv = tuple(inv)
    group = monodromy_group(g0, d)
    corner, _k_of = corner_sheets(base, lift)
    fib_ok, orbits = True, {}
    for p in sorted(profile):
        idx = lift["index"][p]
        by_sheet = {}
        for u in idx:
            for s in corner[u]:
                by_sheet[s] = u
        # deck action g: u -> the unique lifted vertex over p whose sheets
        # contain g(sheets[u]).  A Galois cover has exactly one orbit.
        seen, n_orb = set(), 0
        for u in idx:
            if u in seen:
                continue
            n_orb += 1
            for g in group:
                image = {by_sheet.get(g[s]) for s in corner[u]}
                if None in image or len(image) != 1:
                    fib_ok = False
                else:
                    seen.add(image.pop())
        orbits[str(p)] = n_orb
        if n_orb != 1:
            fib_ok = False
    out.update({
        "g0": g0, "g1": g1, "g0_cycle_type": _cycle_lengths(g0),
        "g1_cycle_type": _cycle_lengths(g1),
        "g0_is_d_cycle": _cycle_lengths(g0) == [d],
        "g1_is_d_cycle": _cycle_lengths(g1) == [d],
        "g1_equals_g0_inverse": g1 == inv, "g1_equals_g0": g1 == g0,
        "deck_group_order": len(group),
        "deck_group_is_cyclic_d": len(group) == d and _cycle_lengths(g0) == [d],
        "galois": len(group) == d, "deck_orbits": orbits,
        "deck_orbits_match_fibres": fib_ok,
    })
    out["compatible"] = bool(out["g0_is_d_cycle"] and out["g1_is_d_cycle"]
                             and out["g1_equals_g0_inverse"] and fib_ok
                             and out["deck_group_is_cyclic_d"])
    if not out["compatible"]:
        raise NotTwoPoint(
            "the declared local monodromies are not compatible with a cyclic "
            f"two-point cover: g0={g0} (type {out['g0_cycle_type']}), "
            f"g1={g1} (type {out['g1_cycle_type']}), g1 == g0^-1: "
            f"{out['g1_equals_g0_inverse']}, |<g0>| = {len(group)}, "
            f"orbits/fibres {orbits}")
    return out


# ------------------------------------------------------------------ label logic
def edge_windings(base, positions, z0, z1, *, samples=64):
    """Integer winding ``delta(p,q)`` of the normalised base argument per edge.

    ``delta`` is the number of turns the *continuous* argument of
    ``mu(b(t))`` gains along the base edge ``(p,q)`` when the principal part
    ``arg(mu(b_q)) - arg(mu(b_p))`` is subtracted:

        delta = (Delta_arg_along_the_edge - (arg v_q - arg v_p)) / 2 pi

    It is computed from the geometry by sampling the chord, not from a chart
    convention: the chord and the spherical arc between the same base points are
    homotopic in ``S^2`` minus the two branch values, so they wind identically.
    Edges incident to a branch value are not assigned (their target argument is
    undefined); the constructor never needs them.
    """
    z = {p: _chart(positions[p]) for p in range(len(positions))}
    M = mobius_matrix(z0, z1)
    v = {p: mobius_eval(M, z[p]) for p in z}
    delta, degenerate, sampled = {}, [], {}
    for e in base.edges:
        p, q = e
        if v[p] in (0, mp.inf) or v[q] in (0, mp.inf):
            continue
        bp, bq = np.asarray(positions[p], float), np.asarray(positions[q], float)
        prev, total = None, mp.mpf(0)
        for i in range(samples + 1):
            t = float(i) / samples
            pt = (1.0 - t) * bp + t * bq
            pt = pt / np.linalg.norm(pt)
            val = mobius_eval(M, _chart(pt))
            if val == 0 or val == mp.inf:
                degenerate.append((p, q, t))
                break
            if prev is not None:
                total += _wrap(mp.arg(val) - mp.arg(prev))
            prev = val
        else:
            principal = mp.arg(v[q]) - mp.arg(v[p])
            turns = (total - principal) / (2 * mp.pi)
            nearest = mp.nint(turns)
            delta[(p, q)] = int(nearest)
            sampled[(p, q)] = (float(total), float(principal), float(turns),
                               float(abs(turns - nearest)))
    return {"delta": delta, "degenerate": degenerate, "turns": sampled}


def monodromy_index(g, d):
    """``i(x)`` with ``g^i(0) = x``, for the cycle of ``g`` through 0."""
    order, x, x0 = [], 0, 0
    while True:
        order.append(x)
        x = g[x]
        if x == x0:
            break
    if len(order) != d:
        raise ShortcutFailure(f"local monodromy {g} is not a d-cycle")
    return {v: i for i, v in enumerate(order)}


def lift_propagation(base, d, lift, delta, info, corner, eps=1):
    """Root index of every lifted vertex, by transport plus deck symmetry.

    Two structural facts are used, both verified by the returned residuals.

    * Transport.  Crossing a lifted edge over the base edge ``(p,q)`` adds
      ``delta(p,q)`` to the root index.  Edges whose endpoint projects to a
      critical value carry no equation: every root coincides there, so the
      index at a totally ramified vertex is free.  This is why the offset
      between the d sheets has to come from somewhere else.
    * Deck symmetry.  The answer is Galois cyclic, so the deck group acts by
      multiplying positions with ``omega``; in the monodromy numbering
      ``g0^i(0) = sheet`` the deck element that moves sheet ``0`` to sheet
      ``i`` adds exactly ``i * eps`` to the root index.  The assembler
      trivialises the cover over a spanning tree, so the lifted graph minus the
      two critical preimages can fall apart into the d sheets; the transport is
      therefore carried out on one sheet and extended by the deck action.

    A disagreement modulo d on a closed lifted walk would be a genuine
    inconsistency (a closed walk has monodromy-fixed sheet, hence winding
    divisible by d), so ``conflict`` must stay ``None``.
    """
    p0, p1 = info["branch_values"]
    idx = monodromy_index(info["g0"], d)
    ramified = [u for u, p in lift["proj"].items() if p in (p0, p1)]
    sheet_of = {}
    for u, p in lift["proj"].items():
        if u in ramified:
            continue
        if len(corner[u]) != 1:
            raise ShortcutFailure(f"lifted vertex {u} over {p} is not of local "
                                  f"degree 1 in the fixed corner")
        sheet_of[u] = idx[corner[u][0]]
    adj = {}
    for lab, (i, j) in lift["edges"].items():
        p, q = lift["proj"][i], lift["proj"][j]
        if i in ramified or j in ramified:
            continue
        w = delta.get((p, q))
        sign = 1
        if w is None:
            w = delta.get((q, p))
            sign = -1
        if w is None:
            continue
        adj.setdefault(i, []).append((j, sign * w))
        adj.setdefault(j, []).append((i, -sign * w))
    if not adj:
        raise ShortcutFailure("no lifted edge carries a winding equation")
    seeds = sorted(u for u in adj if sheet_of[u] == 0)
    if not seeds:
        raise ShortcutFailure("no lifted vertex sits in sheet 0")
    start = seeds[0]
    index = {start: 0}
    queue = [start]
    conflict = None
    while queue:
        u = queue.pop(0)
        for v, w in adj[u]:
            expected = (index[u] + eps * w) % d
            if v in index:
                if index[v] != expected and conflict is None:
                    conflict = {"from": u, "to": v, "have": index[v],
                                "want": expected, "raw": w}
                continue
            index[v] = expected
            queue.append(v)
    # extend to every other sheet by the deck action
    by_sheet = {}
    for u, s_ in sheet_of.items():
        by_sheet.setdefault((lift["proj"][u], s_), []).append(u)
    missing, deck_checks = [], []
    for u, s_ in sheet_of.items():
        if u in index:
            continue
        partner = by_sheet.get((lift["proj"][u], 0))
        if not partner:
            missing.append(u)
            continue
        j = index.get(partner[0])
        if j is None:
            missing.append(u)
            continue
        index[u] = (j + eps * s_) % d
        deck_checks.append((u, partner[0], s_))
    for u in ramified:
        index[u] = 0
    missing = [u for u in missing if u not in index]
    return {"index": index, "conflict": conflict, "missing": missing,
            "reachable": len(index) - len(ramified),
            "deck_seeded": len(deck_checks), "seed": start,
            "sheet_of": sheet_of}


def exact_positions(base, d, lift, base_positions, *, eps=1, swap=False,
                    samples=64, winding_tol=1e-6):
    """Closed-form positions of every lifted vertex.

    ``base_positions[p]`` is the unit vector of base vertex ``p``.  Returns one
    representative of the exact Moebius class plus every intermediate quantity
    (Mobius normalisation, per-edge windings, per-vertex root indices).
    ``eps = +-1`` and ``swap`` select the two cyclic conventions.  The
    orientation certificate distinguishes them when the inherited chart is
    convex; the separate algebraic adapter does not pretend that projection
    residual alone can do so.  Sampled windings must be integral within
    ``winding_tol`` and edges must not pass through a branch value.
    """
    info = analyse(base, d, lift)
    p0, p1 = info["branch_values"]
    if swap:
        p0, p1 = p1, p0
    n = len(base_positions)
    z_of = {p: _chart(base_positions[p]) for p in range(n)}
    if z_of[p0] == z_of[p1]:
        raise ShortcutFailure("the two branch values coincide on the sphere")
    z0, z1 = z_of[p0], z_of[p1]
    M = mobius_matrix(z0, z1)
    zvals = {p: mobius_eval(M, z_of[p]) for p in z_of}
    wind = edge_windings(base, base_positions, z0, z1, samples=samples)
    if wind["degenerate"]:
        raise ShortcutFailure(f"a base edge passes through a branch value: "
                              f"{wind['degenerate'][:3]}")
    unresolved = [(edge, row[3]) for edge, row in wind["turns"].items()
                  if not math.isfinite(row[3]) or row[3] > winding_tol]
    if unresolved:
        raise ShortcutFailure(f"sampled winding is not integral within "
                              f"{winding_tol}: {unresolved[:3]}; increase samples")
    corner, _k = corner_sheets(base, lift)
    prop = lift_propagation(base, d, lift, wind["delta"], info, corner, eps=eps)
    if prop["conflict"] is not None:
        raise ShortcutFailure(f"lifted transport is not path independent: "
                              f"{prop['conflict']}")
    if prop["missing"]:
        raise ShortcutFailure(f"lifted vertices unreachable by transport: "
                              f"{prop['missing'][:4]}")
    fam = {}
    positions = {}
    for u in sorted(lift["proj"]):
        p = lift["proj"][u]
        if p in (info["branch_values"]):
            positions[u] = _position(zvals[p])      # 0 or infinity
            continue
        if p not in fam:
            fam[p] = root_family(zvals[p], d)
        j = prop["index"][u] % d
        positions[u] = _position(fam[p][j])
    P = np.array([positions[u] for u in sorted(lift["proj"])])
    return {"positions": P, "index": prop["index"], "eps": eps, "swap": swap,
            "winding": wind, "info": info, "zvals": zvals,
            "mu_branch": (z0, z1), "p0": p0, "p1": p1,
            "reachable": prop["reachable"]}


# ----------------------------------------------------------------- certificates
def solid_angle_sum(P, tris):
    """Signed solid angle sum of the lifted faces over 4*pi (+1 = positive)."""
    total = 0.0
    for verts, _labels in tris:
        a, b, c = (P[i] for i in verts)
        num = float(a @ np.cross(b, c))
        den = 1.0 + float(a @ b) + float(b @ c) + float(c @ a)
        total += 2.0 * math.atan2(num, den)
    return total / (4.0 * math.pi)


def orientation_votes(P, tris):
    """Faces with positive vs negative signed area on the sphere."""
    pos = neg = 0
    for verts, _labels in tris:
        a, b, c = (P[i] for i in verts)
        num = float(a @ np.cross(b, c))
        den = 1.0 + float(a @ b) + float(b @ c) + float(c @ a)
        ang = 2.0 * math.atan2(num, den)
        if ang > 0:
            pos += 1
        elif ang < 0:
            neg += 1
    return pos, neg


def support_margin(P, tris):
    """Smallest margin by which every face plane supports the vertex set."""
    worst = np.inf
    for verts, _l in tris:
        a, b, c = (P[i] for i in verts)
        n = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(n))
        if norm < 1e-300:
            return -np.inf
        n = n / norm
        other = [-(P[i] - a) @ n for i in range(len(P)) if i not in verts]
        if other:
            worst = min(worst, float(np.min(other)))
    return float(worst)


def projection_residual(P, base, lift, base_positions, d=None):
    """Chordal residual of ``mu(P_u^d)`` against the base value over ``u``.

    This is the defining property of the answer: the normalised covering map
    ``mu^-1 o (.)^d`` sends ``P_u`` to the base position projected below it.
    """
    B = (np.array([base_positions[p] for p in sorted(base_positions)])
         if isinstance(base_positions, dict) else np.asarray(base_positions, float))
    dd = lift["d"] if d is None else d
    info = analyse(base, dd, lift)
    p0, p1 = info["branch_values"]
    z_of = {p: _chart(B[p]) for p in range(len(B))}
    z0, z1 = z_of[p0], z_of[p1]
    worst = mp.mpf(0)
    for u in sorted(lift["proj"]):
        p = lift["proj"][u]
        w = _chart(P[u])
        img = mp.inf if w == mp.inf else w ** dd
        target = mobius_to_zero_infinity(z_of[p], z0, z1)
        worst = max(worst, PL._chordal(img, target))
    return worst


# ---------------------------------------------------------------- certificates
def certificate(built, base, lift, base_positions):
    """Every number a caller needs to accept or refuse the shortcut answer."""
    P = built["positions"]
    ld = lift["ld"]
    tris = lift["tris"]
    out = {"eps": built["eps"], "swap": built["swap"]}
    out["projection_residual"] = float(projection_residual(P, base, lift,
                                                           base_positions))
    pos, neg = orientation_votes(P, tris)
    out["faces_positive"], out["faces_negative"] = pos, neg
    out["solid_angle_sum"] = solid_angle_sum(P, tris)
    out["support_margin"] = support_margin(P, tris)
    out["oriented"] = bool(neg == 0 and pos > 0)
    out["convex_inscribed"] = bool(out["oriented"]
                                   and out["solid_angle_sum"] > 0.5
                                   and out["support_margin"] > 0.0)
    # fibres: completeness against the declared local degrees
    worst_fibre, counts = np.inf, {}
    for p, idx in sorted(lift["index"].items()):
        rows = sorted(ld[u] for u in idx)
        counts[str(p)] = rows
        if sum(rows) != lift["d"]:
            worst_fibre = -1.0
        if len(idx) > 1:
            for a in idx:
                for b in idx:
                    if a < b:
                        worst_fibre = min(worst_fibre,
                                          float(np.linalg.norm(P[a] - P[b])))
    out["fibre_ok"] = bool(all(sum(v) == lift["d"] for v in counts.values()))
    out["fibre_counts"] = counts
    out["min_distinct_in_fibre"] = (None if worst_fibre == np.inf
                                    else float(worst_fibre))
    out["min_pair_global"] = float(min(
        np.linalg.norm(P[a] - P[b]) for a in range(len(P))
        for b in range(a + 1, len(P))) if len(P) > 1 else np.inf)
    shape = (float(np.max(np.abs(np.linalg.norm(P, axis=1) - 1.0)))
             if len(P) else 0.0)
    out["max_sphere_error"] = shape
    out["windings"] = {"degenerate_edges": built["winding"]["degenerate"],
                       "delta": {f"{k}": v for k, v in
                                 sorted(built["winding"]["delta"].items())}}
    return out


def uniformize_from_base(base, d, lift, base_positions, *, eps=None,
                         swap=False):
    """Build a ``uniformize(tris, edges, lengths, vertices)`` callable.

    ``lift_one_step`` calls ``uniformize(tris, edges, lengths, vertices)`` and
    expects an ``(n, 3)`` array in ``vertices`` order (or a dict).  This
    replaces the CEPS solve by the closed form; with ``eps = None`` both cyclic
    conventions are tried and the positively oriented one is returned, so the
    certificate decides rather than the caller.
    """
    def uniformize(tris, edges, lengths, vertices):
        n = len(vertices)
        if n != len(lift["proj"]):
            raise ShortcutFailure(f"the shortcut was built for {len(lift['proj'])} "
                                  f"lifted vertices, got {n}")
        order = list(vertices)
        if sorted(order) != sorted(lift["proj"]):
            raise ShortcutFailure("vertex ids do not match the assembled lift")
        choices = (eps,) if eps is not None else (1, -1)
        last = None
        for e in choices:
            built = exact_positions(base, d, lift, base_positions, eps=e,
                                    swap=swap)
            last = built
            if certificate(built, base, lift, base_positions)[
                    "convex_inscribed"]:
                P = built["positions"]
                return {u: P[u] for u in order}
        raise ShortcutFailure(
            "the root construction did not certify as a convex inscribed "
            "sphere: " + str(certificate(last, base, lift, base_positions)))
    return uniformize
