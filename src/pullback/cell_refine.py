"""Uniform, label-aware 1 -> 4 subdivision of lifted cell complexes.

These are STRUCTURAL PL operations.  Each parallel edge label receives its own
midpoint, and each Euclidean child is congruent to its parent at half scale.
The helpers verify simpliciality, edge incidences and the same length-derived
angle sums at the old vertices.  They deliberately do not pretend to solve for
an embedded configuration: a refinement can still be folded and must be checked
by pullback.embedding and by the rank/excess certificate.  No numerical step
family or continuation claim is exported here.
"""

from __future__ import annotations

import numpy as np

from pullback.rounding import edges_of

__all__ = ["face_edge_labels", "subdivide_cells", "subdivide_simplicial",
           "coarse_lengths", "coarse_angle_sums", "refined_cone_angles"]


def _key(i, j):
    return (i, j) if i < j else (j, i)


def face_edge_labels(cells):
    """Per face: the polygon and the coarse edge *label* at each corner.

    ``cells["corners"][k]`` is the boundary walk of face k as dual darts and
    ``cells["polygons"][k]`` is the source vertex of each dart, so corner j of
    face k is the dart running from ``polygons[k][j]`` to
    ``polygons[k][j + 1]`` and the edge label it traverses is ``dart[0]``.
    """
    out = []
    for k, (poly, walk) in enumerate(zip(cells["polygons"], cells["corners"])):
        if len(walk) != len(poly):
            raise ValueError("corner walk and polygon disagree in length")
        for j, dart in enumerate(walk):
            got = (cells["dual"].src(dart), cells["dual"].tgt(dart))
            want = (poly[j], poly[(j + 1) % len(poly)])
            if got != want:
                raise ValueError(
                    f"corner dart {dart} of face {k} does not run from its "
                    f"polygon vertex to the next ({got} against {want})")
        out.append((tuple(poly), tuple(dart[0] for dart in walk)))
    return out


def _check_simplicial(faces, label_ends, n):
    """Refuse anything the vertex-pair representation could silently lose."""
    for face in faces:
        if len(face) != 3 or len(set(face)) != 3:
            raise ValueError(f"refined face {face} is not a triangle")
    by_pair = {}
    for label, pair in label_ends.items():
        by_pair.setdefault(pair, []).append(label)
    parallel = {p: ls for p, ls in by_pair.items() if len(ls) > 1}
    if parallel:
        raise ValueError(f"refined complex still has parallel edges: {parallel}")
    uses = {}
    for face in faces:
        for k in range(3):
            pair = _key(face[k], face[(k + 1) % 3])
            uses[pair] = uses.get(pair, 0) + 1
    bad = {p: c for p, c in uses.items() if c != 2}
    if bad:
        raise ValueError(f"refined edges not in exactly two faces: {bad}")
    if set(uses) != set(by_pair):
        raise ValueError(
            "refined face walk and refined edge labels disagree on "
            f"{sorted(set(uses) ^ set(by_pair), key=str)}")
    if set(uses) != set(edges_of(faces)):
        raise ValueError("edges_of does not see the refined edge set")
    chi = n - len(label_ends) + len(faces)
    if chi != 2:
        raise ValueError(f"refined complex has Euler characteristic {chi}")


def subdivide_cells(cells, metric):
    """Level 0 -> 1: the label-aware uniform 1 -> 4 similarity subdivision.

    Returns a dict with

    ``faces``           refined triangles (the coarse vertex ids are kept)
    ``lengths``         sorted vertex pair -> length
    ``labels``          the refined edge labels, one per refined edge
    ``label_ends``      label -> sorted vertex pair (checked injective)
    ``label_lengths``   label -> length
    ``midpoint``        coarse edge label -> its new vertex
    ``parent``          refined vertex -> the coarse edge label it sits on, or
                        None for a coarse vertex
    ``coarse_vertices`` the coarse vertices are indices ``0 .. that - 1``
    ``counts``          V, E, F of the refined complex

    The checks are the point of the construction: they fail loudly if a
    refinement would still have two edges through one vertex pair, if a face
    became degenerate, or if the Euler characteristic moved.
    """
    dual = cells["dual"]
    vertices = list(cells["vertices"])
    coarse_edges = list(cells["edges"])
    faces = face_edge_labels(cells)

    for e in coarse_edges:
        if e not in metric["lengths"]:
            raise ValueError(f"coarse edge {e} has no lifted length")
    mid = {e: len(vertices) + i for i, e in enumerate(coarse_edges)}

    refined_faces = []
    labels = []
    label_ends = {}
    label_lengths = {}

    def add(label, i, j, length):
        if label in label_ends:
            raise ValueError(f"refined label {label} emitted twice")
        if i == j:
            raise ValueError(f"refined edge {label} is a self loop at {i}")
        if not length > 0.0:
            raise ValueError(f"refined edge {label} has length {length}")
        label_ends[label] = _key(i, j)
        label_lengths[label] = float(length)
        labels.append(label)

    for f_index, (poly, elabels) in enumerate(faces):
        if len(poly) != 3:
            raise ValueError(
                f"face {f_index} has {len(poly)} corners; the 1 -> 4 refinement "
                "is defined for triangles, and the rabbit's lifted complex is "
                "already a triangulation")
        m = [mid[e] for e in elabels]
        for k in range(3):
            v = poly[k]
            e_next, e_prev = elabels[k], elabels[k - 1]
            add(("half", e_next, v), v, mid[e_next],
                0.5 * metric["lengths"][e_next])
            refined_faces.append((v, mid[e_next], mid[e_prev]))
        # the midpoint of e_k to the midpoint of e_{k+1} is parallel to the
        # third edge of this face and half its length
        for k in range(3):
            third = metric["lengths"][elabels[(k + 2) % 3]]
            add(("spoke", f_index, k), m[k], m[(k + 1) % 3], 0.5 * third)
        refined_faces.append((m[0], m[1], m[2]))

    _check_simplicial(refined_faces, label_ends, len(vertices) + len(mid))
    lengths = {pair: label_lengths[label] for label, pair in label_ends.items()}
    return {
        "faces": tuple(refined_faces),
        "lengths": lengths,
        "labels": tuple(labels),
        "label_ends": label_ends,
        "label_lengths": label_lengths,
        "midpoint": mid,
        "parent": {v: None for v in vertices} | {m: e for e, m in mid.items()},
        "coarse_vertices": len(vertices),
        "counts": {"V": len(vertices) + len(mid), "E": len(labels),
                   "F": len(refined_faces)},
        "coarse_counts": dict(cells["counts"]),
    }


def subdivide_simplicial(faces, lengths):
    """The same 1 -> 4 rule for a simplicial complex with pair-keyed lengths.

    Levels >= 1 have no parallel edges, so a sorted vertex pair is a faithful
    edge label; this is the level-by-level engine behind ``step_family``.
    """
    n = 1 + max(max(f) for f in faces)
    for face in faces:
        for k in range(3):
            pair = _key(face[k], face[(k + 1) % 3])
            if pair not in lengths:
                raise ValueError(f"edge {pair} of face {face} has no length")
    mid = {pair: n + i for i, pair in enumerate(sorted(lengths))}
    out_faces = []
    out_lengths = {}
    for f_index, face in enumerate(faces):
        pairs = [_key(face[k], face[(k + 1) % 3]) for k in range(3)]
        m = [mid[p] for p in pairs]
        for k in range(3):
            v = face[k]
            out_lengths[_key(v, m[k])] = 0.5 * lengths[pairs[k]]
            out_faces.append((v, m[k], m[k - 1]))
        for k in range(3):
            out_lengths[_key(m[k], m[(k + 1) % 3])] = 0.5 * lengths[
                pairs[(k + 2) % 3]]
        out_faces.append((m[0], m[1], m[2]))
    label_ends = {pair: pair for pair in out_lengths}
    _check_simplicial(out_faces, label_ends, n + len(mid))
    return {"faces": tuple(out_faces), "lengths": out_lengths,
            "midpoint": mid, "coarse_vertices": n,
            "counts": {"V": n + len(mid), "E": len(out_lengths),
                       "F": len(out_faces)}}


def coarse_lengths(cells, metric):
    """Coarse lengths keyed by vertex pair, refusing a non-representable class.

    The coarse complex does have a parallel class, so this keying is faithful
    only when the members of each class carry the same length -- which they do
    here, both being lifts of one base Delaunay edge.  A residual computed this
    way is the *merged* coarse system, which for these lengths is the multigraph
    system too; if the lengths differed this function refuses rather than
    pretending.
    """
    from pullback.lift import parallel_edge_pairs

    dual = cells["dual"]
    lengths = {}
    for e in cells["edges"]:
        lengths[_key(*dual.edges[e])] = float(metric["lengths"][e])
    for pair, es in parallel_edge_pairs(cells).items():
        values = {round(float(metric["lengths"][e]), 12) for e in es}
        if len(values) != 1:
            raise ValueError(
                f"parallel class {pair} carries different lengths {values}; the "
                "merged coarse system is not the multigraph system, so a "
                "vertex-pair residual cannot certify it")
    return lengths


def coarse_angle_sums(cells, metric):
    """The angle sum at each coarse vertex of the metric the lengths define.

    Computed from the lengths alone, so it is independent of
    ``lifted_delaunay_metric``'s portrait argument.  On the rabbit it agrees
    with ``metric["cone_angles"]`` to 1e-13 at every vertex, which is the check
    that `local degree x base cone angle` really is the angle sum of the lifted
    piecewise-Euclidean metric and not a separate prescription.  (Vertex 0 is
    then genuinely hyper-ideal *as an input angle*, 8.4902 > 2*pi -- which is
    allowed and is not a realisation obstruction.)
    """
    lengths = coarse_lengths(cells, metric)
    total = {v: 0.0 for v in cells["vertices"]}
    for poly in cells["polygons"]:
        for k in range(len(poly)):
            i, j, l = poly[k], poly[(k + 1) % len(poly)], poly[k - 1]
            a, b, c = lengths[_key(i, j)], lengths[_key(i, l)], lengths[_key(l, j)]
            total[i] += float(np.arccos(np.clip(
                (a * a + b * b - c * c) / (2.0 * a * b), -1.0, 1.0)))
    return total


def refined_cone_angles(subdivision):
    """Angle sums of the refined *input* metric, plus the triangle-inequality
    check.  Every midpoint must come out exactly flat: that is the certificate
    that the refined datum is the coarse surface triangulated more finely."""
    faces, lengths = subdivision["faces"], subdivision["lengths"]
    n = max(max(f) for f in faces) + 1
    total = {v: 0.0 for v in range(n)}
    for face in faces:
        for k in range(3):
            i, j, m = face[k], face[(k + 1) % 3], face[(k + 2) % 3]
            a, b, c = lengths[_key(i, j)], lengths[_key(i, m)], lengths[_key(j, m)]
            if min(a, b, c) <= 0 or not (abs(a - b) < c < a + b):
                raise ValueError(
                    f"refined face {face} violates the triangle inequality: "
                    f"{a}, {b}, {c}")
            total[i] += float(np.arccos(np.clip(
                (a * a + b * b - c * c) / (2.0 * a * b), -1.0, 1.0)))
    return total
