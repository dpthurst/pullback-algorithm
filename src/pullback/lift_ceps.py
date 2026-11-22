"""One-step lift from branching data through a pluggable spherical uniformizer.

The lift itself is combinatorial: a base triangulation with per-edge monodromy
permutations assembles a degree-``d`` cover (:mod:`pullback.dense_assembly`),
the cover inherits the base edge lengths, and a spherical CEPS uniformizer turns
the inherited *piecewise-Euclidean* metric into vertex positions on ``S^2``
(Springborn 2019, Theorem 7.18 / 10.5).  This module is the seam between the
two: it owns the lifting itself, the Möbius gauge of the answer, and the
integration-level checks; it does **not** contain an energy, a gradient or a
Newton solver.  Those enter through ``uniformize``, which is a black box with
the contract written out in :func:`lift_one_step`.

Why the seam needs a gauge policy
---------------------------------
The uniformizer's answer is unique only up to a Möbius transformation of the
sphere, so a lift is an *orbit*.  Two consequences are handled here:

* the representative is pinned by an explicit gauge
  (:func:`pullback.mobius_gauge.pin_gauge` on three lifted vertices of the
  largest-area lifted face), so that two runs of the same pipeline, or two
  different solvers, are comparable coordinate by coordinate;
* when an independent oracle realisation is supplied the lift is compared to it
  through :func:`pullback.mobius_gauge.register_on_labels`, never by
  coordinates, and the reported error is the registration residual.

Why a lift is not a metric isometry
----------------------------------
The inherited metric has cone angle ``ld(u) * Theta_base`` at a lifted vertex of
local degree ``ld``; the CEPS uniformizer instead produces angle sum ``2pi`` at
every interior vertex (GSC Section 7.2), which is what makes the realisation a
convex inscribed polyhedron.  The discrete conformal factor ``u`` is therefore
not zero in general, and this module reports the fitted ``u`` and its residual
rather than pretending the lift is isometric to the inherited metric.

Degrees that must not be confused
---------------------------------
``d`` (the covering degree, from the assembly certificate), the intrinsic
orientation of the inscribed convex sphere (read from its supporting planes,
never from radial projection through the origin) and the per-vertex local
degrees ``ld`` are different things.  Origin-radial winding is only a gauge
diagnostic: it can vanish when a correct convex configuration is Möbius moved
so that the origin leaves its hull.  A collapsed realisation can still have
``d = 2`` as a map to the base; it is rejected by distinctness, not by residual.
"""

from __future__ import annotations

import numpy as np

from . import dense_assembly, mobius_gauge

__all__ = ["LiftError", "lift_one_step", "inspect_lift"]


class LiftError(ValueError):
    """The branching data or the uniformizer output is not usable as a lift."""


# --------------------------------------------------------------------- helpers
def _edge_length_table(base, lengths):
    """Normalise ``lengths`` to ``{canonical edge: positive float}``."""
    if isinstance(lengths, dict):
        table = {}
        for e, value in lengths.items():
            key = dense_assembly.canon(int(e[0]), int(e[1]))
            table[key] = float(value)
    else:
        values = list(lengths)
        if len(values) != len(base.edges):
            raise LiftError(f"{len(values)} lengths for {len(base.edges)} base edges")
        table = {e: float(v) for e, v in zip(sorted(base.edges), values)}
    missing = [e for e in base.edges if e not in table]
    if missing:
        raise LiftError(f"no base length given for {len(missing)} edges, e.g. {missing[:3]}")
    bad = [e for e, v in table.items() if not (np.isfinite(v) and v > 0.0)]
    if bad:
        raise LiftError(f"non-positive or non-finite base length on {bad[:3]}")
    return table


def _lifted_metric(lift, length_table):
    """The inherited (pullback) metric on the cover, edge by edge."""
    edges, base_of = lift["edges"], lift["base_of"]
    lengths = {}
    for label, (i, j) in edges.items():
        e = base_of[label]
        lengths[label] = length_table[dense_assembly.canon(*e)]
    # the inherited metric really is a pullback: parallel labels over the same
    # base edge must carry the same length (they do by construction; this is
    # the guard that a future "per-sheet length" feature cannot slip through
    # unnoticed).
    by_base = {}
    for label, value in lengths.items():
        by_base.setdefault(dense_assembly.canon(*base_of[label]), set()).add(round(value, 15))
    for e, vals in by_base.items():
        if len(vals) != 1:
            raise LiftError(f"parallel labels over {e} carry different lengths {sorted(vals)}")
    return lengths


def _as_position_array(lift, positions, n):
    """Accept the solver's array or label->point dict and return an (n,3) array."""
    verts = sorted(lift["proj"])
    if isinstance(positions, dict):
        missing = [v for v in verts if v not in positions and str(v) not in positions]
        if missing:
            raise LiftError(f"the uniformizer returned no position for {len(missing)} "
                            f"lifted vertices, e.g. {missing[:3]}")
        P = np.array([positions.get(v, positions.get(str(v))) for v in verts], dtype=float)
    else:
        P = np.array(positions, dtype=float)
        if P.shape != (n, 3):
            raise LiftError(f"uniformizer returned shape {P.shape}, expected {(n, 3)}")
    if not np.all(np.isfinite(P)):
        raise LiftError("the uniformizer returned a non-finite position")
    return verts, P


def _largest_area_lifted_face(lift, lengths):
    """The lifted face whose three lifted edge lengths give the largest area.

    A deterministic choice of gauge anchors: it depends only on the lifted
    metric, so two solvers that agree on the metric pin to the same frame.
    """
    best, best_area = None, -1.0
    for k, (verts, labels) in enumerate(lift["tris"]):
        a, b, c = (lengths[l] for l in labels)
        s = 0.5 * (a + b + c)
        rad = s * (s - a) * (s - b) * (s - c)
        if rad > best_area:
            best, best_area = (k, verts), rad
    if best is None or best_area <= 0.0:
        raise LiftError("no lifted face has a positive heron area; the inherited "
                        "metric is degenerate")
    return best


# ---------------------------------------------------------------------- checks
def _solid_angle_sum(P, tris):
    """Signed solid angles of the lifted faces seen from the origin / 4 pi."""
    total = 0.0
    for verts, _labels in tris:
        a, b, c = (P[i] for i in verts)
        num = float(a @ np.cross(b, c))
        den = (1.0 + float(a @ b) + float(b @ c) + float(c @ a))
        total += 2.0 * np.arctan2(num, den)
    return total / (4.0 * np.pi)


def _supporting_margin(P, tris, sample=None):
    """Smallest margin by which every face plane supports the vertex set.

    Positive means the plane through a face has all other vertices strictly on
    the inner side, which for a positively oriented triangulation is exactly the
    statement that the faces are the boundary of the convex hull.
    """
    order = range(len(tris)) if sample is None else sample
    worst = np.inf
    for k in order:
        verts, _l = tris[k]
        a, b, c = (P[i] for i in verts)
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-300:
            return -np.inf
        n = n / norm
        # the face's own vertices lie on the plane; a supporting plane needs
        # every *other* vertex strictly on the inner side.  Report the smallest
        # such margin (positive for a strict convex hull, negative for a
        # fold-over), which is the number a certificate can quote.
        other = [-(P[i] - a) @ n for i in range(len(P)) if i not in verts]
        if not other:
            continue
        worst = min(worst, float(np.min(other)))
    return float(worst)


def _scaling_fit(P, tris, lengths):
    """Least-squares vertex scale factors:  log|p_i-p_j| = log l_ij + (u_i+u_j)/2.

    The solution is unique (the incidence matrix has full column rank on a
    connected complex), so the reported ``u`` is a definite number.  The residual is
    the max absolute deviation in log length and is the number that decides
    whether the candidate is discretely conformal to the inherited metric.
    """
    rows = []
    for verts, labels in tris:
        for t in range(3):
            i, j = verts[t], verts[(t + 1) % 3]
            ell = lengths[labels[t]]
            chord = float(np.linalg.norm(P[i] - P[j]))
            if chord <= 0.0:
                return {"ok": False, "reason": "coincident lifted edge",
                        "residual": float("inf"), "u": None}
            rows.append((i, j, np.log(chord) - np.log(ell)))
    n = len(P)
    A = np.zeros((len(rows), n))
    b = np.zeros(len(rows))
    for k, (i, j, value) in enumerate(rows):
        A[k, i] = 0.5
        A[k, j] = 0.5
        b[k] = value
    # ``u`` is unique for a connected complex (every column of ``A`` is used),
    # so no gauge may be removed from it: adding a constant to ``u`` changes the
    # predicted log-lengths by that constant, which is a different metric.
    u, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = A @ u - b
    return {"ok": True, "u": u, "residual": float(np.max(np.abs(resid))),
            "rms": float(np.sqrt(np.mean(resid ** 2))),
            "rank": int(np.linalg.matrix_rank(A))}


# ---------------------------------------------------------------- the lift step
def lift_one_step(base, d, edge_perms, lengths, uniformize, *, gauge="rotate",
                  oracle=None, oracle_labels=None, tol=1e-9):
    """Assemble the cover, uniformize it, gauge the answer and inspect the lift.

    Parameters
    ----------
    base : ``dense_assembly.Base``
        Consistently oriented base triangulation.
    d : int
        Covering degree.
    edge_perms : dict
        Per-edge permutation datum, as in :mod:`pullback.dense_assembly`.
    lengths : dict or sequence
        Base edge lengths, keyed by the canonical edge (or aligned with
        ``sorted(base.edges)``).  These are the lengths of the base *PL
        metric*; the cover inherits them by pullback.
    uniformize : callable
        ``uniformize(tris, edges, lengths, vertices)`` where

        * ``tris`` is a list of ``((i, j, k), (label_ij, label_jk, label_ki))``
          for the lifted complex,
        * ``edges`` is ``{label: (i, j)}`` for the lifted edges,
        * ``lengths`` is ``{label: positive float}``, the inherited metric,
        * ``vertices`` is the sorted list of lifted vertex ids.

        It must return an ``(n, 3)`` array of unit vectors in the order of
        ``vertices`` (or a ``{vertex: xyz}`` dict).  The callable is the black
        box: a CEPS solver, an oracle, or a stub in a test.
    gauge : {"rotate", "mobius", "none"}
        ``rotate`` (default) fixes the rotational part of the gauge with two
        labelled rotations (heaviest vertex to the pole, farthest vertex onto
        the meridian; see :func:`pullback.mobius_gauge.rotation_gauge`).  Rotations are isometries, so every geometric property of the
        answer -- in particular supporting-plane convexity of the flat-faced
        polyhedron -- is preserved, and the representative is reproducible.
        ``mobius`` pins the full six-dimensional Möbius freedom by sending the
        three vertices of the largest-area lifted face to a fixed equilateral
        frame; that is the right choice for *comparing* two answers.  A proper
        Möbius map preserves the supporting-cap incidence of each face circle,
        so supporting-plane convexity remains an intrinsic check.  Its margin
        magnitude is gauge dependent and is diagnostic only.  ``none`` leaves
        the solver's own gauge untouched.
    oracle : optional
        An independent realisation (array in ``vertices`` order, or a
        ``{vertex: xyz}`` dict) to compare against up to Möbius.
    tol : float
        Two-sided threshold used for the numeric verdicts; the checks report
        their measured numbers, so a caller can apply a stricter test.

    Returns
    -------
    dict with the (possibly gauged) positions, the vertex order, the gauge map,
    the fitted discrete scale factors, the assembly certificates, the
    registration against the oracle (if any) and explicit verdicts.
    """
    if gauge not in ("rotate", "mobius", "none"):
        raise LiftError(f"unknown gauge policy {gauge!r}")
    if int(d) < 1:
        raise LiftError("the covering degree must be positive")

    # 1. the combinatorial lift
    lift = dense_assembly.assemble(base, int(d), edge_perms)
    cert = lift["cert"]
    if not cert["is_sphere"] or not cert["cover_ok"]:
        raise LiftError("the branching data do not assemble a connected spherical "
                        f"cover: is_sphere={cert['is_sphere']} cover_ok={cert['cover_ok']} "
                        f"components={cert['components']} chi={cert['chi_lift']}")

    # 2. the inherited metric
    length_table = _edge_length_table(base, lengths)
    lifted_lengths = _lifted_metric(lift, length_table)

    # 3. the uniformizer
    verts = sorted(lift["proj"])
    call = {"tris": lift["tris"], "edges": lift["edges"],
            "lengths": dict(lifted_lengths), "vertices": list(verts)}
    returned = uniformize(call["tris"], call["edges"], call["lengths"],
                          call["vertices"])
    verts, P = _as_position_array(lift, returned, len(verts))
    norms = np.linalg.norm(P, axis=1)
    if np.any(np.abs(norms - 1.0) > max(tol, 1e-9)):
        raise LiftError("the uniformizer's positions are not on the unit sphere: "
                        f"max | |p| - 1 | = {np.max(np.abs(norms - 1.0)):.3e}")

    # 4. the gauge
    gauge_info = {"policy": gauge, "mobius": None,
                  "preserves_flat_convexity": True}
    if gauge == "rotate":
        mob, P = mobius_gauge.rotation_gauge(P)
        gauge_info.update(mobius=mob)
    elif gauge == "mobius":
        k, anchors = _largest_area_lifted_face(lift, lifted_lengths)
        mob, P, pin = mobius_gauge.pin_gauge(P, anchors)
        gauge_info.update(anchors=list(anchors), face=k, pin=pin, mobius=mob)

    # 5. inspection
    report = inspect_lift(lift, P, lifted_lengths, tol=tol,
                          convexity_applicable=gauge_info["preserves_flat_convexity"])

    # 6. comparison against an independent realisation, up to Möbius
    if oracle is not None:
        o_labels = list(verts) if oracle_labels is None else [str(l) for l in oracle_labels]
        if isinstance(oracle, dict):
            O = np.array([oracle[v] for v in verts], dtype=float)
        else:
            O = np.array(oracle, dtype=float)
            if O.shape != P.shape:
                raise LiftError(f"oracle shape {O.shape} does not match {P.shape}")
        mob, info = mobius_gauge.register_on_labels(P, verts, O, o_labels)
        report["oracle"] = dict(info, registration=mob)
        report["verdicts"]["matches_oracle"] = info["max_angle"] <= max(tol, 1e-9)

    report["lift"] = lift
    report["vertices"] = verts
    report["positions"] = P
    report["gauge"] = gauge_info
    report["lengths"] = lifted_lengths
    report["base_lengths"] = length_table
    return report


def inspect_lift(lift, P, lengths, *, tol=1e-9, support_sample=None,
                 convexity_applicable=True):
    """Integration-level, gauge-covariant checks on one candidate lift.

    ``convexity_applicable`` is retained for API compatibility and ignored:
    proper Möbius maps always preserve supporting-cap incidence.  The support
    *margin magnitude* is not invariant and remains a diagnostic.  Origin
    radial winding is likewise reported only as ``origin_radial_degree_raw``;
    it is not an intrinsic orientation test.  Exact/incircle certification is
    still a separate layer, and near-degenerate floating signs are not claimed
    as exact certificates.
    """
    tris = lift["tris"]
    verts = sorted(lift["proj"])
    if len(P) != len(verts):
        raise LiftError(f"{len(P)} positions for {len(verts)} lifted vertices")
    threshold = max(tol, 1e-12)
    verdicts = {}

    dmat = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    np.fill_diagonal(dmat, np.inf)
    min_sep = float(np.min(dmat))
    verdicts["distinct"] = bool(min_sep > threshold * 10.0)
    verdicts["on_sphere"] = bool(np.max(np.abs(np.linalg.norm(P, axis=1) - 1.0))
                                 <= max(tol, 1e-9))

    # Combinatorial orientation, independent of where the origin happens to be.
    directed = set()
    consistent = True
    degenerate = 0
    for face, _labels in tris:
        a, b, c = face
        if np.linalg.norm(np.cross(P[b] - P[a], P[c] - P[a])) < 1e-14:
            degenerate += 1
        for dart in ((a, b), (b, c), (c, a)):
            if dart in directed:
                consistent = False
            directed.add(dart)
    for a, b in directed:
        if (b, a) not in directed:
            consistent = False
            break
    verdicts["orientation_consistent"] = bool(consistent and degenerate == 0)
    verdicts["degenerate_faces"] = int(degenerate)

    if support_sample is None:
        margin = _supporting_margin(P, tris)
    else:
        margin = _supporting_margin(P, tris, sample=support_sample)
    reversed_tris = [((f[0], f[2], f[1]), labels) for f, labels in tris]
    reverse_margin = _supporting_margin(P, reversed_tris)
    verdicts["supporting_margin"] = float(margin)
    verdicts["reverse_supporting_margin"] = float(reverse_margin)
    verdicts["convexity_applicable"] = True
    verdicts["convex"] = bool(max(margin, reverse_margin) >= -threshold)

    # For an embedded convex inscribed sphere, supporting planes determine the
    # intrinsic orientation without any ambient radial projection.
    if margin >= -threshold and margin >= reverse_margin:
        orientation = 1
    elif reverse_margin >= -threshold:
        orientation = -1
    else:
        orientation = 0
    verdicts["orientation"] = orientation
    verdicts["map_degree_raw"] = float(orientation)
    verdicts["map_degree"] = orientation
    verdicts["map_degree_integer"] = bool(orientation != 0)
    origin_winding = _solid_angle_sum(P, tris)
    verdicts["origin_radial_degree_raw"] = float(origin_winding)
    verdicts["cover_degree"] = int(lift["d"])

    scaling = _scaling_fit(P, tris, lengths)
    verdicts["scaling_residual"] = float(scaling.get("residual", np.inf))
    verdicts["scaling_consistent"] = bool(scaling["ok"]
                                          and scaling["residual"] <= max(tol, 1e-9))
    verdicts["collapse"] = bool(not verdicts["distinct"])
    verdicts["orientation_ok"] = bool(orientation == 1)
    ok = (verdicts["distinct"] and verdicts["on_sphere"]
          and verdicts["orientation_consistent"] and verdicts["orientation_ok"]
          and verdicts["convex"] and verdicts["map_degree_integer"]
          and verdicts["scaling_consistent"])
    verdicts["is_lift"] = bool(ok)
    try:
        signature = mobius_gauge.signature(P)
    except mobius_gauge.GaugeError:
        signature = None
    return {"verdicts": verdicts, "scaling": scaling, "signature": signature}
