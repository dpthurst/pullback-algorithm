"""Compute a geometric output comparison ``upsilon`` as a graph map.

This is the small geometric primitive required by ``spine-reformulation.md``.
It interpolates each lifted spine edge by its *declared* short-great-circle
representative, follows that arc through the output Delaunay cells, and reads
the dual Voronoi chain from the cell transitions.  A source vertex landing on
a Delaunay edge is slid consistently to one dual endpoint before its incident
chains are built.  No transport or homotopy census enters.

The numerical realiser supplies vertex positions only, not lifted arcs.  The
arc used here is therefore a named representative, not a claim that the solver
actually integrated a curve (it does not).  Antipodal endpoints have no short
geodesic.  Degenerate configurations and cell paths that do not stabilise are
refused rather than silently turned into a guessed homotopy class.
"""
from __future__ import annotations

import numpy as np

from pullback import mesh
from pullback import refined_cover as rr
from pullback.hyperbolic_centre import centred_coordinates

try:                                     # the two-cap carrier component
    from pullback import equatorial_retraction as eq
except ImportError as exc:               # tracked-only retention fallback
    if "equatorial_retraction" not in str(exc):
        raise
    # Deliberately not fatal.  Without that module the two-cap carrier is
    # simply unavailable and this module behaves exactly as it did before the
    # carrier existed: a concyclic output is refused by _origin_inside_hull.
    # The module and the equatorial hunks of this file are ONE retention unit
    # (see the component note); this fallback only keeps the package
    # importable if the unit is ever split by mistake.
    eq = None

#: Empty without the two-cap module, so an ``except`` clause never has to test.
_EQUATORIAL_ERRORS = () if eq is None else (eq.EquatorialError,)
from pullback.maps import make_graph_map
from pullback.ribbons import B, F, reduce_chain

__all__ = [
    "GeometricRetractionError", "arc_points", "cell_sequence", "geometric_upsilon",
    "refined_spine", "refined_vertex_positions", "refined_upsilon",
]


class GeometricRetractionError(ValueError):
    """The geometric retraction is not defined on these data."""


def _unit(x):
    x = np.asarray(x, dtype=float)
    n = float(np.linalg.norm(x))
    if not np.isfinite(n) or n == 0.0:
        raise GeometricRetractionError("zero/non-finite vector in source data")
    return x / n


def _output_vectors(points):
    """Output puncture positions as unit vectors, rejecting duplicates."""
    v = np.array([_unit(mesh.to_vector(p) if np.ndim(p) == 0 else p)
                  for p in points], dtype=float)
    if len(v) < 4:
        raise GeometricRetractionError("at least four output punctures are required")
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            if np.linalg.norm(v[i] - v[j]) <= 1e-10:
                raise GeometricRetractionError(
                    f"output punctures {i} and {j} coincide or nearly coincide")
    return v


def arc_points(a, b, samples):
    """Points of the unique short great-circle arc from ``a`` to ``b``.

    ``samples`` counts segments, so there are ``samples + 1`` points.  The two
    endpoints are included.  Antipodal and coincident endpoints are refused.
    """
    a, b = _unit(a), _unit(b)
    if np.linalg.norm(a - b) <= 1e-14:
        raise GeometricRetractionError("an edge has coincident endpoints")
    if np.linalg.norm(a + b) <= 1e-14:
        raise GeometricRetractionError("an edge has antipodal endpoints")
    if samples < 1:
        raise ValueError("samples must be positive")
    theta = float(np.arccos(np.clip(float(a @ b), -1.0, 1.0))
                   )
    if theta < 1e-14:
        raise GeometricRetractionError("an edge has coincident endpoints")
    t = np.linspace(0.0, 1.0, int(samples) + 1)[:, None]
    return ((np.sin((1.0 - t) * theta) * a + np.sin(t * theta) * b)
            / np.sin(theta))


def _on_short_arc(a, b, p, tol=1e-12):
    """Whether unit vector ``p`` lies on the minor arc from ``a`` to ``b``.

    A point of that arc is a positive combination of its endpoints.  Solving
    the two dot-product equations gives a direct coplanarity/on-segment test;
    unlike sampling, it cannot miss a puncture between two sample points.
    ``arc_points`` validates the non-antipodal, non-coincident case first.
    """
    c = float(np.clip(a @ b, -1.0, 1.0))
    den = 1.0 - c * c
    if den <= 1e-28:
        return False                       # arc_points will issue the refusal
    pa, pb = float(a @ p), float(b @ p)
    alpha = (pa - c * pb) / den
    beta = (pb - c * pa) / den
    residual = float(np.linalg.norm(p - alpha * a - beta * b))
    return residual <= tol and alpha >= -tol and beta >= -tol


def _nearest(q, arc):
    """Return the nearest puncture row of each arc sample and its margin."""
    chord = np.linalg.norm(arc[:, None, :] - q[None, :, :], axis=2)
    cells = np.argmin(chord, axis=1)
    ordered = np.partition(chord, 1, axis=1)
    return cells, ordered[:, 1] - ordered[:, 0], ordered[:, 0]


def _face_hits(x, q, faces, tol=1e-10):
    """Indices of Delaunay faces whose spherical facet contains ``x``.

    Cocircular Delaunay blocks are polygons.  Test a fan of triangles within
    each polygon and stop at its first hit, so the diagonal of the fan does not
    become an artificial extra cell transition.
    """
    x = _unit(x)
    hits = []
    for i, f in enumerate(faces):
        for k in range(1, len(f) - 1):
            A, B, C = q[f[0]], q[f[k]], q[f[k + 1]]
            U, V = B - A, C - A
            normal = np.cross(U, V)
            if np.linalg.norm(normal) <= 1e-14:
                continue
            den = float(x @ normal)
            if abs(den) <= 1e-14:
                continue
            scale = float(A @ normal) / den
            if scale <= 0.0 or abs(scale) > 1e12:
                continue
            y = scale * x
            M = np.column_stack((U, V))
            uv, *_ = np.linalg.lstsq(M, y - A, rcond=None)
            u, v = float(uv[0]), float(uv[1])
            if u >= -tol and v >= -tol and u + v <= 1.0 + tol:
                hits.append(i)
                break
    return hits


def _origin_inside_hull(q, faces, tol=1e-12):
    """Whether the ray origin is strictly inside the output Delaunay hull.

    The facet-path construction below uses central projection.  Outside the
    hull a ray can meet two nonadjacent facets, so the data do not determine a
    single Delaunay cell path.  A Möbius re-chart is then required; silently
    choosing one hit would not define a retraction of the punctured sphere.
    """
    for f in faces:
        A, B, C = q[f[0]], q[f[1]], q[f[2]]
        normal = np.cross(B - A, C - A)
        size = float(np.linalg.norm(normal))
        if size <= 1e-14:
            return False
        normal /= size
        off = float(A @ normal)
        signed = q @ normal - off
        if np.all(signed >= -tol):
            margin = -off                    # origin's value on the same side
        elif np.all(signed <= tol):
            margin = off
        else:
            return False                     # not a supporting hull facet
        if margin <= tol:
            return False                     # outside, or on a facet plane
    return True


def _equatorial_route(q, faces, target):
    """Carrier decision from the two-cap module, or the generic route."""
    if eq is None:
        return None, {"route": "unavailable", "plane_thickness": None}
    return eq.equatorial_route(q, faces, source=target)


def cell_sequence(a, b, q, faces, steps=256):
    """Collapse a great-circle arc to its ordered Delaunay-face indices."""
    arc = arc_points(a, b, steps)
    margin = np.empty(len(arc))
    vertex_distance = np.empty(len(arc))
    rows = []
    for k, x in enumerate(arc):
        hits = _face_hits(x, q, faces)
        if not hits:
            raise GeometricRetractionError(
                f"arc sample {k}/{steps} lies outside the output Delaunay cells")
        # Membership ambiguity is resolved only at a face boundary, and is
        # handled consistently for all incident edges below.  Most arc samples
        # have exactly one hit; isolated boundary hits are allowed and recorded.
        rows.append(tuple(hits))
        chord = np.linalg.norm(q - x, axis=1)
        order = np.sort(chord)
        margin[k] = order[1] - order[0]
        vertex_distance[k] = order[0]
    out = []
    for row in rows:
        cell = row[0] if len(row) == 1 else row
        if not out or cell != out[-1]:
            out.append(cell)
    return tuple(out), float(margin.min()), float(vertex_distance.min())


def _word(cells, target, start_cell=None, end_cell=None,
          transition_labels=None):
    """Convert sampled Delaunay memberships to an oriented Voronoi chain.

    Candidates are handled as candidate faces, not as if a tuple were itself a
    face index.  A shortest adjacency path is chosen exactly by (edge count,
    lexicographic chain), so no homotopy or transport census is involved.
    Returns the chain and its first/last Delaunay-face images.
    """
    if not cells:
        raise GeometricRetractionError("empty cell path")
    rows = ((c,) if isinstance(c, int) else tuple(c) for c in cells)
    rows = list(rows)
    initial = rows[0]
    if start_cell is not None:
        if start_cell not in initial:
            raise GeometricRetractionError(
                f"declared start cell {start_cell} is not in {initial}")
        initial = (start_cell,)
    if end_cell is not None and end_cell not in rows[-1]:
        raise GeometricRetractionError(
            f"declared end cell {end_cell} is not in {rows[-1]}")
    paths = {v: (0, (), v) for v in initial}
    for step, candidates in enumerate(rows[1:]):
        nxt = {}
        for a, (cost, word, start) in paths.items():
            for b in candidates:
                if a == b:
                    nc, nw = cost, word
                else:
                    if transition_labels is not None:
                        label = transition_labels[step]
                        if label is None:
                            raise GeometricRetractionError(
                                f"unresolved equatorial transition: {cells}")
                        shared = [label]
                    else:
                        shared = [e for e, incident in target._voronoi_faces.items()
                                  if a in incident and b in incident]
                    if len(shared) != 1:
                        continue
                    nc, nw = cost + 1, word + ((shared[0], F if a < b else B),)
                key = (nc, tuple(str(x) for x in nw))
                old = nxt.get(b)
                if old is None or key < (old[0], tuple(str(x) for x in old[1])):
                    nxt[b] = (nc, nw, start)
        if not nxt:
            raise GeometricRetractionError(
                f"cell walk has no Delaunay edge: {cells} at {candidates}")
        paths = nxt
    if end_cell is not None:
        if end_cell not in paths:
            raise GeometricRetractionError(
                f"cell walk cannot end at declared cell {end_cell}: {cells}")
        paths = {end_cell: paths[end_cell]}
    chosen = min(paths, key=lambda b: (
        paths[b][0], tuple(str(x) for x in paths[b][1]), b))
    _cost, word, start = paths[chosen]
    return reduce_chain(tuple(word)), start, chosen


def _target_metadata(output_points):
    """Recover the Delaunay triangulation behind a Voronoi spine."""
    faces = mesh.delaunay_faces(output_points)
    edge_faces = {}
    voronoi_faces = {}
    for i, f in enumerate(faces):
        for k in range(len(f)):
            e = rr.canon(f[k], f[(k + 1) % len(f)])
            voronoi_faces.setdefault(e, set()).add(i)
            edge_faces[e] = rr.canon(f[k], f[(k + 1) % len(f)])
    return faces, edge_faces, voronoi_faces


def geometric_upsilon(source, source_positions, output_points, target=None,
                      steps=(256, 512), centre="hyperbolic"):
    """Compute ``upsilon : source -> Voronoi(output_points)`` as a graph map.

    ``source`` is the lifted spine as a RibbonGraph.  ``source_positions[i]``
    is the realised position of ``source.vertices()[i]``; for the refined
    wrapper these are face centres of the *dual* Delaunay-block picture.
    Every edge in ``source.edge_list()`` is assigned a chain by direct cell
    retraction, and vertex images are shared by every incident edge.  A
    two-resolution certificate is required per edge.  ``output_points`` may be
    chart scalars or three-dimensional chordal vectors.  By default the source
    and output configurations are transformed by their Lorentz-hyperbolic
    centroid, so central membership remains meaningful in any hemisphere.
    ``centre="origin"`` retains the earlier construction for negative controls.
    Any non-stabilising, degenerate, or antipodal data raise
    GeometricRetractionError.

    A concyclic (cocircular) output is not refused: it is retracted onto the two-cap
    carrier of :mod:`pullback.equatorial_retraction`, which owns the chart, the
    acceptance policy and the cap transitions.  The route is decided there, by
    one measure, and reported as ``"equatorial_route"``; a near-flat or
    self-inconsistent configuration is a typed refusal, never a guess between
    the two carriers.  Nothing here chooses a facet in a zero-thickness hull,
    and no carrier change is called a completed pullback or a uniformisation.

    The returned dict contains the actual ``GraphMap`` as ``map``, the output
    ``target``, the cell and dart path of every edge, the two sampling levels,
    and the nearest-puncture clearance and bisector-margin samples.  The
    chordal metric is used only for this retraction, never to define a
    chain-indexed pullback metric.
    """
    q = _output_vectors(output_points)
    if source_positions is None and source is None:
        # Permit target-only diagnostic calls; a real graph map still needs the
        # source as soon as target construction and centration have succeeded.
        x = np.empty((0, 3), dtype=float)
    else:
        x = np.asarray(source_positions, dtype=float)
    expected_shape = (0 if source is None else len(source.vertices()), 3)
    if x.shape != expected_shape:
        raise GeometricRetractionError(
            f"expected {expected_shape} source positions, got {x.shape}")
    x = np.array([_unit(v) for v in x])
    try:
        if centre == "hyperbolic":
            x, q, gauge = centred_coordinates(x, q)
        elif centre == "origin":
            gauge = {"construction": "unmodified sphere origin"}
        else:
            raise ValueError("centre must be 'hyperbolic' or 'origin'")
        # Use the centred coordinates consistently for Delaunay, Voronoi, and
        # every source arc.  Chordal-vector callers get the same normalisation
        # as chart-scalar callers; the raw arrays are not mixed back in.
        chart_points = [mesh.to_point(v) for v in q]
        _faces, edge_faces, voronoi_faces = _target_metadata(chart_points)
        # One decision, taken where the two-cap chart and its acceptance policy
        # live: either the equatorial carrier with its own plane and cap cells,
        # or the generic central-projection route.  The two must agree with the
        # combinatorial decomposition; a disagreement is a typed refusal there.
        equatorial, equatorial_route = _equatorial_route(q, _faces, target)
        if target is None:
            target = mesh.voronoi_spine(chart_points)
        if equatorial is None and not _origin_inside_hull(q, _faces):
            raise GeometricRetractionError(
                "the sphere centre is not strictly inside the output Delaunay "
                "hull; re-chart before using the central-cell retraction")
    except GeometricRetractionError:
        raise
    except _EQUATORIAL_ERRORS as exc:
        raise GeometricRetractionError(str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise GeometricRetractionError(
            f"degenerate centred Delaunay/Voronoi data: {exc}") from exc
    # Delaunay-face and edge labels are independent of numerical refinement.
    target._edge_faces = edge_faces
    target._voronoi_faces = voronoi_faces
    if source is None:
        raise GeometricRetractionError("a source graph and source positions are required")
    index = {v: i for i, v in enumerate(source.vertices())}
    # A vertex whose radial projection lies on a Delaunay edge retracts to the
    # interior of the dual Voronoi edge, not to one of its endpoints.  GraphMap
    # needs a vertex image, so slide that image to one endpoint and constrain
    # every incident edge to the same slide.  Doing this once per source
    # vertex is essential: independent per-edge choices change the endpoint
    # images and can silently alter the induced pi_1 homomorphism.
    vertex_images = {}
    vertex_cells = {}
    for v in source.vertices():
        p = x[index[v]]
        if float(np.min(np.linalg.norm(q - p, axis=1))) <= 1e-10:
            raise GeometricRetractionError(
                f"source vertex {v} coincides with an output puncture")
        if equatorial is not None:
            # A concyclic output is classified in its circle's plane, not by
            # projecting a three-dimensional convex hull from the origin.
            signed = float(p @ equatorial["normal"] - equatorial["offset"])
            cells = equatorial["cells"]
            hits = (cells if abs(signed) <= eq.DEFAULT_TOLERANCE
                    else (cells[0] if signed > 0 else cells[1],))
        else:
            hits = tuple(_face_hits(p, q, _faces))
        if not hits:
            raise GeometricRetractionError(
                f"source vertex {v} lies outside the output Delaunay cells")
        if len(hits) > 2:
            raise GeometricRetractionError(
                f"source vertex {v} has ambiguous Delaunay cells {hits}")
        if len(hits) == 2 and equatorial is None:
            shared = [e for e, incident in voronoi_faces.items()
                      if set(incident) == set(hits)]
            if len(shared) != 1:
                raise GeometricRetractionError(
                    f"source vertex {v}: adjacent cells {hits} do not share "
                    "one Delaunay edge")
        vertex_cells[v] = hits
        vertex_images[v] = min(hits)
    cell_rows, chains = {}, {}
    try:
        for label in source.edge_list():
            a, b = source.edges[label]
            pa, pb = x[index[a]], x[index[b]]
            arc_points(pa, pb, 1)           # validate the arc before predicates
            for puncture, p in enumerate(q):
                if _on_short_arc(pa, pb, p):
                    raise GeometricRetractionError(
                        f"edge {label!r}: the arc passes through output "
                        f"puncture {puncture}")
            if equatorial is not None:
                first = eq.cell_sequence(pa, pb, q, _faces, target,
                                         steps=steps[0], geo=equatorial)
                second = eq.cell_sequence(pa, pb, q, _faces, target,
                                          steps=steps[1], geo=equatorial)
            else:
                first = (*cell_sequence(pa, pb, q, _faces, steps[0]), None)
                second = (*cell_sequence(pa, pb, q, _faces, steps[1]), None)
            if first[0] != second[0]:
                raise GeometricRetractionError(
                    f"edge {label!r}: unstable nearest-cell sequence "
                    f"{first[0]} != {second[0]}; the arc may be ambiguous")
            if min(first[2], second[2]) <= 1e-10:
                raise GeometricRetractionError(
                    f"edge {label!r}: the arc meets an output puncture")
            if first[3] != second[3]:
                raise GeometricRetractionError(
                    f"edge {label!r}: unstable equatorial transition labels")
            word, start_cell, end_cell = _word(
                first[0], target, vertex_images[a], vertex_images[b], first[3])
            second_word, second_start, second_end = _word(
                second[0], target, vertex_images[a], vertex_images[b], second[3])
            if (word, start_cell, end_cell) != (second_word, second_start, second_end):
                raise GeometricRetractionError(
                    f"edge {label!r}: unresolved cell choices do not stabilise")
            cell_rows[label] = {"cells": first[0], "darts": word,
                                "transition_labels": first[3],
                                "nearest_margin": min(first[1], second[1]),
                                "closest_puncture": min(first[2], second[2])}
            chains[label] = word
            if a not in vertex_images:
                vertex_images[a] = start_cell
            elif vertex_images[a] != start_cell:
                raise GeometricRetractionError(
                    f"inconsistent start cell at source vertex {a}")
            if b not in vertex_images:
                vertex_images[b] = end_cell
            elif vertex_images[b] != end_cell:
                raise GeometricRetractionError(
                    f"inconsistent end cell at source vertex {b}")
        result = make_graph_map(source, target, chains, vertex_images)
    except _EQUATORIAL_ERRORS as exc:
        raise GeometricRetractionError(str(exc)) from exc
    except KeyError as exc:
        raise GeometricRetractionError(f"missing source-edge geometry: {exc}") from exc
    return {
        "map": result,
        "upsilon": result,
        "target": target,
        "source": source,
        "paths": cell_rows,
        "sampling_levels": tuple(steps),
        "realisation": "short great-circle arcs between realised endpoints",
        "vertex_images": dict(vertex_images),
        "vertex_cells": vertex_cells,
        "target_face_points": mesh.voronoi_face_points(target, len(q)),
        "equatorial": equatorial is not None,
        "equatorial_route": equatorial_route,
        "output_points": list(output_points),
        "centred_output": q.copy(),
        "centred_source": x.copy(),
        "gauge": gauge,
    }


def refined_spine(refined):
    """The realised lifted spine of ``RefinedVE.K2`` as a RibbonGraph."""
    return rr.dual_of(refined.K2["tris"], refined.K2["edges"])


def refined_vertex_positions(refined, positions):
    """Realised Delaunay-face centres of ``RefinedVE.K2``.

    A lifted spine's vertices are its Delaunay faces, not its punctures.  The
    realisation supplies the lifted puncture coordinates, so each source-spine
    vertex position is the normalised centroid of its realised triangle.  Rows
    follow ``refined_spine(refined).vertices()``; this is the canonical dual
    face order, not generally ``K2['tris']`` order.  A zero-sum centroid is
    refused instead of normalised to a meaningless point.
    """
    X = np.asarray(positions, dtype=float)
    tris = refined.K2["tris"]
    graph = rr.tri_graph(tris, refined.K2["edges"])
    source = refined_spine(refined)
    index = {v: i for i, v in enumerate(source.vertices())}
    # RibbonGraph.dual numbers its vertices by graph.faces(), whose canonical
    # order is generally not the input triangle order.  Match the actual face
    # geometry instead of attaching centroid k to dual vertex k.
    rows = [None] * len(source.vertices())
    for v, walk in enumerate(graph.faces()):
        vertices = set()
        for dart in walk:
            vertices.update(refined.K2["edges"][dart[0]])
        hits = [k for k, tr in enumerate(tris) if set(tr[0]) == vertices]
        if len(hits) != 1:
            raise GeometricRetractionError(
                f"canonical lifted face {tuple(sorted(vertices))} matches "
                f"{len(hits)} triangles")
        tr = tris[hits[0]]
        w = X[list(tr[0])].sum(axis=0)
        norm = float(np.linalg.norm(w))
        if norm <= 1e-12 or not np.isfinite(norm):
            raise GeometricRetractionError(
                f"degenerate lifted Delaunay face {tr[0]}: zero centroid")
        rows[index[v]] = w / norm
    return np.asarray(rows)


def refined_upsilon(refined, cert, steps=(256, 512)):
    """Compute the retraction for a ``RefinedVE.step`` certificate.

    The output configuration is the four active punctures returned by the real
    step itself; non-active source vertices remain part of the source spine.
    """
    source = refined_spine(refined)
    positions = refined_vertex_positions(refined, cert["x"])
    output = [mesh.to_point(v) for v in cert["config"]]
    return geometric_upsilon(source, positions, output, steps=steps)
