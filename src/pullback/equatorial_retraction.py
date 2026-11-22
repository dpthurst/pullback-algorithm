"""Retraction onto a two-cap cocircular Delaunay spine.

For points that lie on one circle of the sphere the empty-circle decomposition
has two cells with the same cyclic boundary: one for each cap.  Their dual is
two vertices joined by parallel edges, not a triangulated convex hull.  This
module reads source great-circle arcs in that geometry.  It never chooses a
triangulation, a nearest chordal-metric Voronoi cell, or a guessed equatorial
order when the data do not decide one.

Chart contract
--------------
The chart is the *circle's own plane*: the unit normal ``normal`` and offset
``offset`` with ``q[i] @ normal == offset`` for every puncture ``i``, and an
orthonormal in-plane frame ``(u, w)`` centred on the circle.  ``offset == 0``
is a great circle, ``offset != 0`` a small one; both are supported, so the
caller does not have to present the data in a great-circle gauge.

Two facts make that safe, and they are what the code relies on:

* a source arc is a great-circle arc of the sphere, so it meets the boundary
  circle exactly where it meets the plane -- ``_event_position`` is an
  incidence test against ``x @ normal - offset``, not a great-circle
  assumption;
* a cap is a side of that plane, so the cell of a sample is the sign of
  ``x @ normal - offset``, again for a small circle as much as a great one.

Which consecutive punctures a crossing lies between is decided by its *angular*
position in ``(u, w)`` (``_boundary_pair``).  For ``offset == 0`` that is the
same incidence as the affine positive-cone test ``x = a q[i] + b q[j]``,
``a, b >= 0``, which the tests check it against; the affine test is invalid
once ``offset != 0`` because the boundary arc is then not a great-circle arc,
and using it there misclassifies genuine crossings.

Acceptance policy
-----------------
One measure decides the route: ``plane_thickness``, the largest distance of the
punctures from their best-fit plane.  ``<= DEFAULT_TOLERANCE`` is the equatorial
route, ``<= NEAR_FLAT_UNDECIDED`` is a declared undecided band (refused), above
that the generic central-projection route.  The equatorial route additionally
certifies the carrier with a *second, independent* measure -- the largest
distance from the supporting plane picked by ``mesh.delaunay_faces`` -- and
refuses when the two disagree instead of silently switching route.  The band is
policy, not a theorem: inside it the empty-circle decomposition is decided by
comparisons as small as the numbers themselves, and a two-resolution stability
check cannot see a misassignment shared by both resolutions.
"""

from __future__ import annotations

import numpy as np

from . import mesh


DEFAULT_TOLERANCE = 1e-9
#: Declared undecided band above DEFAULT_TOLERANCE; see the module docstring.
NEAR_FLAT_UNDECIDED = 1e-4
#: Smallest admissible angular gap between consecutive boundary punctures.
MIN_ANGULAR_GAP = 1e-8


class EquatorialError(ValueError):
    """The equatorial carrier or the arc's transition is not decided."""


def plane_thickness(q):
    """Largest distance of the punctures from their best-fit plane.

    This is the one geometric measure the route policy of this module uses.  It
    is a distance, so it is directly comparable with the ``coplanarity_residual``
    that :func:`layout` measures from the supporting plane independently.
    """
    Q = np.asarray(q, dtype=float)
    centred = Q - Q.mean(axis=0)
    normal = np.linalg.svd(centred, full_matrices=False)[2][-1]
    return float(np.max(np.abs(centred @ normal)))


def layout(q, faces, source=None, tol=DEFAULT_TOLERANCE):
    """Return the plane, cyclic boundary, and puncture-pair darts of a two-cap.

    ``faces`` is the output of :func:`mesh.delaunay_faces`.  Anything which is
    not two polygons with exactly the same corners is not in this module's
    scope; it returns ``None``.  At this boundary the caller deliberately does
    not force a three-dimensional convex-hull chart.
    """
    if len(faces) != 2 or len(faces[0]) != len(faces[1]):
        return None
    boundary, other = tuple(faces[0]), tuple(faces[1])
    if len(boundary) < 3 or set(boundary) != set(other) or other != (
            (boundary[0],) + tuple(reversed(boundary[1:]))):
        return None
    if set(boundary) != set(range(len(q))):
        raise EquatorialError("equatorial carrier does not cover all output punctures")
    q = np.asarray(q, dtype=float)
    tri = q[list(boundary[:3])]
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    norm = float(np.linalg.norm(normal))
    if norm <= tol:
        raise EquatorialError("equatorial boundary has no spanning plane normal")
    normal /= norm
    offsets = q @ normal
    centre = float(np.mean(offsets))
    residual = float(np.max(np.abs(offsets - centre)))
    if residual > tol:
        raise EquatorialError(
            f"equatorial plane is undecided: coplanarity residual {residual:.3e}")

    radius = float(np.linalg.norm(tri[0] - centre * normal))
    if radius <= tol:
        raise EquatorialError("equatorial circle has no radius; the punctures coincide")
    u = (tri[0] - centre * normal) / radius
    w = np.cross(normal, u)
    raw_angles = np.array([
        float(np.arctan2(float(q[i] @ w), float(q[i] @ u)))
        for i in boundary])
    # Lift the cyclic order by cumulative positive turns.  np.unwrap leaves an
    # exact -pi jump in place, which can falsely reject a perpendicular pair.
    steps = np.mod(np.diff(np.concatenate(
        [raw_angles, raw_angles[:1] + 2 * np.pi])), 2 * np.pi)
    angles = raw_angles[0] + np.r_[0.0, np.cumsum(steps[:-1])]
    gaps = np.diff(np.concatenate([angles, angles[:1] + 2 * np.pi]))
    if abs(float(steps.sum()) - 2 * np.pi) > 1e-12:
        raise EquatorialError(
            "equatorial boundary is not one positively oriented cyclic order")
    if np.min(gaps) <= MIN_ANGULAR_GAP:
        raise EquatorialError(
            f"equatorial point order is undecided: smallest gap {np.min(gaps):.3e}")

    # A transition may only carry a label the target really has; ``source`` is
    # the target spine here.  No second label map: the primal pair IS the label.
    labels = None if source is None else set(source.edge_list())
    pairs = [mesh.edge_key(boundary[i], boundary[(i + 1) % len(boundary)])
             for i in range(len(boundary))]
    return {
        "normal": normal, "offset": centre, "boundary": boundary,
        "pairs": pairs, "labels": labels, "cells": (0, 1),
        "coplanarity_residual": residual, "min_angular_gap": float(np.min(gaps)),
        "radius": radius, "angles": tuple(float(a) for a in angles),
        "u": u, "w": w,
    }


def require_real_fourfold(q, tol=DEFAULT_TOLERANCE):
    """Reality certificate for a concyclic four-point set.

    A four-puncture configuration is called globally equatorial only when its
    one cross-ratio is real.  The certificate exists for ``len(q) == 4``; a
    larger concyclic block has no single fourfold invariant and is certified by
    the two independent coplanarity measures instead (see the module docstring).
    """
    z = mesh.labelled_cross_ratio(np.asarray(q, dtype=float))
    if not np.isfinite(z) or abs(complex(z).imag) > tol:
        raise EquatorialError(
            f"globally equatorial, but the fourfold cross-ratio is not close "
            f"to real: {z}; the equatorial route is undecided")
    return complex(z)


def equatorial_route(q, faces, source=None, tol=DEFAULT_TOLERANCE,
                     undecided=NEAR_FLAT_UNDECIDED):
    """Decide the carrier and return ``(layout_or_None, record)``.

    ``record["route"]`` is ``"equatorial"`` or ``"generic"``; the declared
    undecided band and any disagreement between the two coplanarity measures
    raise :class:`EquatorialError` rather than picking a carrier.
    """
    q = np.asarray(q, dtype=float)
    thickness = plane_thickness(q)
    record = {"plane_thickness": thickness}
    if thickness <= tol:
        geo = layout(q, faces, source=source, tol=tol)
        if geo is None:
            raise EquatorialError(
                "the output is concyclic to within "
                f"{thickness:.3e} but its Delaunay decomposition is not the "
                "two-cap one; refusing rather than choosing a carrier")
        record["route"] = "equatorial"
        if len(q) == 4:
            record["cross_ratio"] = require_real_fourfold(q, tol=tol)
        return geo, record
    if thickness <= undecided:
        raise EquatorialError(
            f"near-equatorial ambiguity: plane thickness {thickness:.3e} is "
            f"above the equatorial tolerance {tol:.0e} and below the "
            f"generic-route limit {undecided:.0e}; the carrier is undecided")
    geo = layout(q, faces, source=source, tol=tol)
    if geo is not None:
        raise EquatorialError(
            "the Delaunay decomposition is two-cap but the plane thickness "
            f"{thickness:.3e} is above {tol:.0e}; refusing the inconsistent "
            "carrier")
    record["route"] = "generic"
    return None, record


def _point_on_boundary_arc(a, b, p, tol=DEFAULT_TOLERANCE):
    """Does ``p`` lie in the affine positive cone of ``a, b``?

    This is the great-circle incidence test: valid only when the boundary
    circle passes through the origin, because then the boundary arc from ``a``
    to ``b`` is the short great-circle arc.  It is kept as the independent
    control for :func:`_angular_boundary_pair`, not as the production test.
    """
    c = float(np.clip(a @ b, -1.0, 1.0))
    den = 1.0 - c * c
    if den <= 1e-22:
        return None                      # antipodal pair: not a cone at all
    pa, pb = float(a @ p), float(b @ p)
    x = (pa - c * pb) / den
    y = (pb - c * pa) / den
    residual = float(np.linalg.norm(p - x * a - y * b))
    return residual <= tol and x >= -tol and y >= -tol


def _affine_boundary_pair(x, geo, tol=DEFAULT_TOLERANCE):
    """The ``_point_on_boundary_arc`` pair containing ``x``, or ``None``."""
    q, boundary = geo["q"], geo["boundary"]
    found = []
    for k, (a, b) in enumerate(zip(boundary, boundary[1:] + boundary[:1])):
        i, j = sorted((a, b))
        hit = _point_on_boundary_arc(q[i], q[j], x, tol=tol)
        if hit is None:
            return None                  # antipodal boundary arc
        if hit:
            found.append(geo["pairs"][k])
    return found[0] if len(found) == 1 else None


def _angular_boundary_pair(x, geo, tol=DEFAULT_TOLERANCE):
    """The consecutive boundary pair whose circle arc contains ``x``.

    Incidence is angular in the circle's own plane, so this is valid for a
    small circle as well as a great one.  ``layout`` has already put the
    boundary punctures in (unwrapped) cyclic order with a positive gap.
    """
    normal, offset = geo["normal"], geo["offset"]
    if abs(float(x @ normal) - offset) > tol:
        raise EquatorialError(
            "a boundary event is not on the equatorial plane: offset residual "
            f"{float(x @ normal) - offset:.3e}")
    angles = geo["angles"]
    u, w = geo["u"], geo["w"]
    theta = float(np.arctan2(float(x @ w), float(x @ u)))
    theta = (theta - angles[0]) % (2.0 * np.pi) + angles[0]
    for k in range(len(angles)):
        low = angles[k]
        high = angles[k + 1] if k + 1 < len(angles) else angles[0] + 2 * np.pi
        if low <= theta < high:
            return geo["pairs"][k]
    raise EquatorialError(
        "a crossing of the equatorial boundary is not between any declared "
        "consecutive punctures; the cyclic order is undecided")


def _boundary_pair(x, geo, tol=DEFAULT_TOLERANCE):
    """The puncture pair a boundary crossing belongs to."""
    pair = _angular_boundary_pair(x, geo, tol=tol)
    if abs(geo["offset"]) <= tol:
        # On a great circle both incidences apply; check that the angular test
        # has not drifted away from the affine one rather than assume it.
        affine = _affine_boundary_pair(x, geo, tol=tol)
        if affine is not None and affine != pair:
            raise EquatorialError(
                "equatorial incidence tests disagree: angular "
                f"{pair}, affine {affine}")
    return pair


def _event_position(arc, sample, indices, normal, offset, tol):
    """Localise one boundary event within its adjacent sample interval."""
    for i in indices:
        if i >= len(arc) - 1:
            continue
        sa = float(arc[i] @ normal - offset)
        sb = float(arc[i + 1] @ normal - offset)
        if abs(sa) <= tol and i > 0:
            sa = float(arc[i - 1] @ normal - offset)
        if sa * sb <= 0.0:
            lo, hi = arc[i], arc[i + 1]
            slo = sa
            for _ in range(48):
                mid = (lo + hi) / 2.0
                mid /= np.linalg.norm(mid)
                smid = float(mid @ normal - offset)
                if smid * (slo if abs(slo) > tol else -1.0) >= 0.0:
                    lo, slo = mid, smid
                else:
                    hi = mid
            point = (lo + hi) / 2.0
            # The bracket can exhaust double precision a few ulps short of the
            # plane.  Enforce the incidence it was just computed to localise,
            # rather than discarding that measurement into a wider tolerance.
            point -= (float(point @ normal - offset)) * normal
            size = float(np.linalg.norm(point))
            if size <= tol:
                raise EquatorialError("equatorial crossing has no finite location")
            return point / size
    return arc[sample]


def cell_sequence(a, b, q, faces, target, steps=256,
                  tol=DEFAULT_TOLERANCE, geo=None):
    """Retract one arc via two cap cells and explicit boundary transitions.

    Returns ``(cells, margin, clearance, labels)`` with the same meaning as
    :func:`pullback.geometric_retraction.cell_sequence`: ``margin`` is the
    smallest bisector margin, i.e. the smallest gap between the two nearest
    puncture distances over all arc samples, and ``clearance`` the smallest
    nearest-puncture distance.  Both are measurements; neither is a constant.

    A boundary event carries the actual target edge label.  This resolves the
    otherwise ambiguous fact that every parallel edge joins the same two cap
    vertices.  Arcs along the equatorial boundary, or arcs whose tangency to it
    cannot be distinguished from a crossing, are refused.
    """
    from pullback.geometric_retraction import arc_points

    arc = arc_points(a, b, steps)
    if geo is None:
        geo = layout(q, faces, source=target, tol=tol)
    if geo is None:
        raise EquatorialError("output is not a two-cap Delaunay configuration")
    geo["q"] = np.asarray(q, dtype=float)
    normal, offset = geo["normal"], geo["offset"]
    hits, clearances, margins = [], [], []
    for x in arc:
        signed = float(x @ normal - offset)
        chord = np.sort(np.linalg.norm(geo["q"] - x, axis=1))
        clearances.append(float(chord[0]))
        margins.append(float(chord[1] - chord[0]))
        hits.append(geo["cells"] if abs(signed) <= tol
                    else (geo["cells"][0] if signed > 0 else geo["cells"][1],))
    if max(abs(float(x @ normal) - offset) for x in arc) <= tol:
        raise EquatorialError(
            "an entire source arc lies on the equatorial boundary; its cap "
            "side and homotopy representative are undecided")
    if min(clearances) <= 1e-10:
        raise EquatorialError("a source arc passes through an output puncture")

    # A sample exactly on the wall is evidence for the one cap transition,
    # not two successive transitions into and out of the candidate pair.  Skip
    # it when assembling cap runs; its index still brackets the event.  If the
    # arc returns to the same cap instead of crossing, it was a tangency.
    rows, indices = [], []
    last_single = None
    contact_from_single = False
    for k, row in enumerate(hits):
        if len(row) != 1:
            if last_single is None and not rows:
                # An equatorial source vertex has two admissible cap images;
                # retain them for the shared-vertex constraint in `_word`.
                rows.append(row)
                indices.append(k)
            elif last_single is not None:
                contact_from_single = True
            continue
        cell = row[0]
        last_single = cell
        if not rows or cell != rows[-1]:
            rows.append(cell)
            indices.append(k)
    if contact_from_single and len(rows) == 1:
        raise EquatorialError(
            "the source arc touches the equatorial boundary without crossing it")
    labels = []
    for j, (left, right) in enumerate(zip(rows, rows[1:])):
        if left == right:
            labels.append(None)
            continue
        point = _event_position(arc, indices[j + 1],
                                range(max(0, indices[j] - 1),
                                      min(len(arc) - 1, indices[j + 1] + 1)),
                                normal, offset, tol)
        pair = _boundary_pair(point, geo, tol=tol)
        if geo["labels"] is not None and pair not in geo["labels"]:
            raise EquatorialError(f"no dual edge for equatorial pair {pair}")
        labels.append(pair)

    return tuple(rows), float(min(margins)), float(min(clearances)), labels
