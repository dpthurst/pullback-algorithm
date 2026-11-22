"""Initial geometric state from an abstract ribbon virtual endomorphism.

The constructor takes ONLY the session's ribbon data -- the spine Gamma_0, the
cover pi : Gamma_1 -> Gamma_0 and the map phi -- and produces

  * puncture positions in C-hat, each carrying the label of the spine face whose
    disc it is the centre of,
  * a combinatorial realisation of Gamma_0 that identifies it (the full-cell
    quotient: one polygon per spine vertex, sides paired by darts, faces = the
    punctures), together with the exactly specified subdivision used to make it
    realisable,
  * a frozen lifted cover of that realisation through (pi, phi), with the degree
    profile and the ribbon face correspondence,
  * a step that turns a configuration of the base punctures into a configuration
    of the lifted punctures -- the discrete Thurston pullback.

No analytic rational map, no critical values and no preimage computation occurs
anywhere in this module.  `check_no_analytic_data` in the tests asserts that.

WHY THE FULL-CELL QUOTIENT AND NOT A STAR MESH
----------------------------------------------
The dual of a spine is a cell decomposition D of S^2 whose *vertices are the
punctures*: one 2-cell per spine vertex v (a k_v-gon, k_v the valence), one
1-cell per spine edge, one 0-cell per spine face.  The obvious way to give D a
geometry is a star mesh: put a point inside each polygon and join it to the
corners.  That fails whenever a polygon of D meets a vertex more than once (a
face of the *map* that is not a simple polygon), and for a spine that is a
bouquet it always does: a bouquet of three loops has V=1, so D has ONE face,
its boundary walk is C,A,C,B,C,E, and the vertex C occurs three times.  A star
mesh then produces two triangles with the same three corners -- not a cell
decomposition, and `Mesh.validate` refuses it (directed edge used twice).

The full-cell quotient keeps the polygon and its side pairing: corners are
*darts*, never repeated, so the repeated vertex is a cone angle, not a mesh
defect.  Subdivision then has to be stated exactly.  `fan_subdivision` below is
that statement: one new centre vertex per face, one new spoke *per corner
position* (so the two spokes at two positions that happen to share a corner
vertex stay distinct EDGES, and are reported as a parallel class), and one
triangle (centre, corner, next corner) per side.  It always passes the labelled
Euler check V + F - E = 2.  It is deliberately an EDGE-LABELLED multigraph: the
rabbit's base needs V=5, E=9, F=6 with a parallel class of size 3, and
collapsing that class onto a vertex pair is exactly the mistake
`rounding_multi` was written to avoid.
"""
from __future__ import annotations

import math

import numpy as np

from pullback.lift import dual_cells, parallel_edge_pairs
from pullback import rounding_multi as mg
from pullback.portrait import face_map_pi, face_map_phi

__all__ = [
    "full_cell_quotient", "obstruction_profile", "fan_subdivision",
    "regular_lengths", "arbitrary_lengths", "realise", "initial_state",
    "frozen_cover", "step", "iterate",
]


# ---------------------------------------------------------------------------
# 1. the full-cell quotient
# ---------------------------------------------------------------------------

def full_cell_quotient(spine):
    """The cell decomposition D dual to `spine`, as a quotient of polygons.

    Returns a dict.  `polygons[k]` is a face of D (a spine vertex) as a cyclic
    list of dual vertices (= punctures), `corners[k][j]` is the dart of
    `spine` that is side j of that polygon, `edge_of_corner[k][j]` its edge
    label.  `puncture_faces` maps a spine face index (the puncture label) to the
    dual vertex index; it is the identity by `dual_cells`' construction, and it
    is returned explicitly because every later label goes through it.
    """
    cells = dual_cells(spine)
    edge_of_corner = []
    for k, walk in enumerate(cells["corners"]):
        edge_of_corner.append(tuple(dart[0] for dart in walk))
    counts = cells["counts"]
    if counts["V"] - counts["E"] + counts["F"] != 2:
        raise ValueError(f"dual cell decomposition is not a sphere: {counts}")
    return {
        "spine": spine,
        "cells": cells,
        "vertices": tuple(cells["vertices"]),         # = the punctures
        "edges": tuple(cells["edges"]),               # = the spine's edges
        "polygons": tuple(tuple(p) for p in cells["polygons"]),
        "corners": tuple(tuple(w) for w in cells["corners"]),
        "edge_of_corner": tuple(edge_of_corner),
        "counts": dict(counts),
        "puncture_faces": {i: i for i in cells["vertices"]},
    }


def obstruction_profile(q, tol=1e-12):
    """What a regular-dual (star-mesh) realisation of `q` would have to survive.

    Reported, each with a witness:

    `vertex_repeats`   per face, the dual vertices occurring more than once.
                       A star mesh at such a face makes two triangles with the
                       same corners.  THIS is the failure the task names.
    `edge_repeats`     per face, the edge labels occurring twice -- a bridge of
                       the spine, whose dual is a loop of D.  A star mesh
                       cannot assign such a face a single length twice.
    `smooth_vertices`  dual vertices whose regular-metric cone angle is 2*pi.
                       They are not cone points of the flat surface, so the
                       vertex set of D is strictly finer than the singular set:
                       a labelling artefact, not a geometry.
    `star_realisable`  no repeats, and E == 3V - 6, i.e. D is already a
                       simplicial triangulation of S^2 with the punctures as
                       its vertex set.  Then the equilateral metric is a
                       legitimate regular-dual metric with excess 0.
    """
    vertex_repeats, edge_repeats = {}, {}
    for k, poly in enumerate(q["polygons"]):
        seen, rep = {}, {}
        for v in poly:
            seen[v] = seen.get(v, 0) + 1
        for v, n in seen.items():
            if n > 1:
                rep[v] = n
        if rep:
            vertex_repeats[k] = rep
        eseen, erep = {}, {}
        for e in q["edge_of_corner"][k]:
            eseen[e] = eseen.get(e, 0) + 1
        for e, n in eseen.items():
            if n > 1:
                erep[e] = n
        if erep:
            edge_repeats[k] = erep
    angle = {}
    for k, poly in enumerate(q["polygons"]):
        kk = len(poly)
        a = math.pi * (1.0 - 2.0 / kk)       # corner angle of a regular k-gon
        for v in poly:
            angle[v] = angle.get(v, 0.0) + a
    smooth = sorted(v for v, a in angle.items() if abs(a - 2 * math.pi) < 1e-9)
    V, E = q["counts"]["V"], q["counts"]["E"]
    return {
        "vertex_repeats": vertex_repeats,
        "edge_repeats": edge_repeats,
        "cone_angles": angle,
        "smooth_vertices": smooth,
        "star_realisable": not vertex_repeats and not edge_repeats and E == 3 * V - 6,
        "V": V, "E": E, "F": q["counts"]["F"],
    }


def _describe_obstruction(q, prof):
    bits = []
    for k, rep in prof["vertex_repeats"].items():
        bits.append(f"face {k} {q['polygons'][k]} meets vertices {rep}")
    for k, rep in prof["edge_repeats"].items():
        bits.append(f"face {k} uses edges {rep} twice (bridge of the spine)")
    if not bits and prof["E"] != 3 * prof["V"] - 6:
        bits.append(f"E={prof['E']} but a triangulation of S^2 on V={prof['V']} "
                    f"vertices has 3V-6={3 * prof['V'] - 6} edges")
    return "; ".join(bits) or "none"


# ---------------------------------------------------------------------------
# 2. the exactly specified subdivision
# ---------------------------------------------------------------------------

def fan_subdivision(q, tag=""):
    """The 1 -> k fan subdivision of the full-cell quotient, edge-labelled.

    Statement of the construction, exactly:

    * one new vertex ``("c", k)`` per face k of q (the polygon's interior point);
    * the coarse edges keep their labels (so the base/lift correspondence is
      inherited, not recomputed);
    * one new spoke ``("s", tag, k, j)`` per CORNER POSITION j of face k, joining
      the centre to ``polygons[k][j]``.  Two corner positions that share a
      corner vertex give two distinct spoke labels on the same vertex pair --
      a parallel class, reported, never merged;
    * one triangle ``(centre, polygons[k][j], polygons[k][j+1])`` per side j.

    Invariants (checked, not assumed): every triangle has three distinct
    vertices, every label occurs in exactly two triangles, every coarse label
    occurs in two triangles, and V + F - E = 2 on the labelled multigraph.
    """
    polys, corners, eoc = q["polygons"], q["corners"], q["edge_of_corner"]
    ncoarse = len(q["vertices"])
    centre = {k: ncoarse + k for k in range(len(polys))}
    spokes = {}
    for k, poly in enumerate(polys):
        spokes[k] = []
        for j in range(len(poly)):
            spokes[k].append(("s", tag, k, j))
    faces = []
    ends = {}
    use = {}
    for k, poly in enumerate(polys):
        n = len(poly)
        c = centre[k]
        for j in range(n):
            a, b = poly[j], poly[(j + 1) % n]
            s_a, s_b = spokes[k][j], spokes[k][(j + 1) % n]
            e = eoc[k][j]
            # orientation: the polygon's walk is the face boundary of q, so the
            # fan triangle inherits it as (centre, a, b)
            faces.append((c, a, b))
            for lab, (i, jj) in ((s_a, (c, a)), (s_b, (b, c)), (e, (a, b))):
                if lab in ends and ends[lab] != tuple(sorted((i, jj))):
                    raise ValueError(f"label {lab} got two vertex pairs")
                ends[lab] = tuple(sorted((i, jj)))
                use[lab] = use.get(lab, 0) + 1
    for f in faces:
        if len(set(f)) != 3:
            raise ValueError(f"fan triangle {f} repeats a corner")
    bad = {lab: n for lab, n in use.items() if n != 2}
    if bad:
        raise ValueError(f"labels not used by exactly two triangles: {bad}")
    n = ncoarse + len(polys)
    E = len(ends)
    if n - E + len(faces) != 2:
        raise ValueError(f"fan subdivision has Euler characteristic "
                         f"{n - E + len(faces)}")
    # the edge-label -> vertex-pair map, with labels (not pairs) as keys
    edges = {}
    for lab, pair in ends.items():
        edges[lab] = (int(pair[0]), int(pair[1]))
    if set(edges) != set(use):
        raise ValueError("edge label set and use set disagree")
    return {
        "faces": tuple(faces),
        "edges": edges,
        "centre": centre,
        "spoke": {lab: pair for lab, pair in ends.items()
                  if isinstance(lab, tuple) and lab[:2] == ("s", tag)},
        "coarse_vertices": ncoarse,
        "n_vertices": n,
        "counts": {"V": n, "E": E, "F": len(faces)},
        "coarse_counts": dict(q["counts"]),
        "parallel": parallel_edge_pairs({"dual": _PairGraph(edges),
                                         "edges": list(edges)}),
    }


class _PairGraph:
    """Minimal `dual` stand-in: edge label -> (src, tgt), for parallel_edge_pairs."""

    def __init__(self, edges):
        self.edges = {lab: pair for lab, pair in edges.items()}


def regular_lengths(q, fan):
    """The regular-dual metric: every coarse edge 1, spokes the circumradius.

    For a k-gon face of side 1 the centre-to-corner distance is 1/(2 sin(pi/k)),
    so the fan triangle at that corner is the isoceles triangle with apex angle
    2*pi/k.  For k = 3 this is the equilateral triangle: the regular-dual
    realisation of a trivalent spine is equilateral, which is what the task
    calls 'often equilateral triangles' -- and it is NOT imposed on a bouquet,
    whose single face has k = 6 and cone angle 2*pi/3 at three of its corners.
    """
    lengths = {}
    for lab in fan["edges"]:
        if isinstance(lab, tuple) and lab[:2] == ("s", ""):
            k = lab[2]
            lengths[lab] = 1.0 / (2.0 * math.sin(math.pi / len(q["polygons"][k])))
        else:
            lengths[lab] = 1.0
    return lengths


def arbitrary_lengths(q, fan, seed=0, spread=0.35):
    """A deliberately different initial dual metric, same complex.

    Each coarse edge gets 1 + spread*U, each spoke the corresponding
    circumradius scaled by the geometric mean of its face's two side lengths --
    a rule stated here so the realisation is reproducible, and chosen only to be
    positive and to satisfy the triangle inequalities (checked by `realise`).
    No property of an analytic map enters: `seed` is the only input.
    Validity is measured only by `realise`; the generator does not assert that
    every seeded collection belongs to a valid embedded polyhedron.
    """
    rng = np.random.RandomState(seed)
    coarse = {}
    for lab in q["edges"]:
        coarse[lab] = 1.0 + spread * float(rng.uniform())
    lengths = {}
    for k, poly in enumerate(q["polygons"]):
        n = len(poly)
        sides = [coarse[q["edge_of_corner"][k][j]] for j in range(n)]
        for j in range(n):
            s = math.sqrt(sides[j] * sides[(j + 1) % n])
            lengths[("s", "", k, j)] = s / (2.0 * math.sin(math.pi / n))
    lengths.update(coarse)
    return lengths


# ---------------------------------------------------------------------------
# 3. realisation
# ---------------------------------------------------------------------------

def realise(fan, lengths, x0=None, maxit=400, tol=1e-13):
    """Realise the fan-subdivided dual complex as points of C-hat.

    The criterion is the edge-labelled one of `rounding_multi`: the chordal
    lengths of the output are discretely conformally equivalent to `lengths`
    (log chord - log ell = (u_i + u_j)/2 for a vertex scaling u).  That is the
    'uniformize' step of the regular-dual recipe, and it is the only numerical
    step in this module.  Returns the positions, the punctures first, plus the
    rank/excess certificate of the complex -- `excess > 0` means the complex
    does not determine the conformal class, so the realisation is one of a
    family and the number of moduli is reported rather than hidden.
    """
    edges = fan["edges"]
    n = fan["n_vertices"]
    if x0 is None:
        rng = np.random.RandomState(12345)
        x0 = rng.normal(size=(n, 3))
        x0 = x0 / np.linalg.norm(x0, axis=1)[:, None]
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (n, 3):
        raise ValueError(f"x0 has shape {x0.shape}, expected {(n, 3)}")
    info = mg.structure(edges, x0)
    x, residual, iters = mg.round_realise(edges, lengths, x0=x0,
                                          maxit=maxit, tol=tol)
    out = dict(info)
    out.update({"residual": float(residual), "iterations": int(iters),
                "x": x, "lengths": dict(lengths)})
    if not np.isfinite(residual) or residual > max(tol, 1e-12) * 100:
        out["converged"] = False
    else:
        out["converged"] = True
    nc = fan["coarse_vertices"]
    out["puncture_points"] = [x[i] for i in range(nc)]
    return out


def initial_state(spine, kind="regular", seed=0, x0=None, maxit=400, tol=1e-13):
    """The initial geometric state of a graph virtual endomorphism's base.

    `spine` is the RibbonGraph (only its ribbon structure is read).  Returns

    ``quotient``    the full-cell quotient of the spine,
    ``obstructions``  `obstruction_profile` -- what a star mesh would have to
                      survive, with witnesses,
    ``subdivision`` the fan subdivision (the exactly specified one),
    ``metric``      the dual metric chosen: 'regular' (all coarse edges 1,
                    spokes the circumradius) or 'arbitrary' (seeded),
    ``positions``   the puncture positions in C-hat, indexed by the spine's
                    FACE index -- the label, not a geometric guess,
    ``labels``      face index -> the spine's own boundary walk (darts), so the
                    puncture label is reconstructible from the ribbon data,
    ``realisation`` the rank/excess/residual record; an actual embedded spine
                  drawing and an embedding certificate are not yet produced.
    """
    q = full_cell_quotient(spine)
    prof = obstruction_profile(q)
    fan = fan_subdivision(q, tag="")
    if kind == "regular":
        lengths = regular_lengths(q, fan)
    elif kind == "arbitrary":
        lengths = arbitrary_lengths(q, fan, seed=seed)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    real = realise(fan, lengths, x0=x0, maxit=maxit, tol=tol)
    faces = spine.faces()
    return {
        "quotient": q,
        "obstructions": prof,
        "obstruction_text": _describe_obstruction(q, prof),
        "subdivision": fan,
        "metric": lengths,
        "positions": real["puncture_points"],
        "labels": {i: tuple(d[0] for d in w) for i, w in enumerate(faces)},
        "realisation": real,
        "kind": kind,
    }


# ---------------------------------------------------------------------------
# 4. the frozen lifted cover
# ---------------------------------------------------------------------------

def frozen_cover(session):
    """The cover of the base realisation through (pi, phi), as frozen data.

    Nothing here looks at positions, critical values or preimages.  The lifted
    complex is the full-cell quotient of the COVER spine, whose edge labels are
    `(base_edge, sheet)` -- so every lifted edge knows which base edge it lies
    over, including the two copies of a doubled edge.

    Returned:
      ``lifted``          the cover's full-cell quotient + fan subdivision,
      ``base_of_edge``    lifted edge label -> base edge label,
      ``degree_profile``  per lifted puncture: (base puncture, local degree),
      ``active``          lifted punctures whose boundary walk does not collapse
                          under phi, with their base label -- the P_0' of the
                          author's reformulation, computed from the portrait,
      ``face_check``      whether each cover face walk projects to a power of a
                          base face walk (the ribbon face correspondence),
    """
    qb = full_cell_quotient(session.base)
    qc = full_cell_quotient(session.cover)
    fb = fan_subdivision(qb, tag="b")
    fc = fan_subdivision(qc, tag="c")

    base_edges = set(session.base.edge_list())

    def _spoke(lab):
        return isinstance(lab, tuple) and len(lab) == 4 and lab[0] == "s"

    base_of_edge = {}
    for lab in fc["edges"]:
        if _spoke(lab):
            continue                       # fan spokes are internal to a face
        if lab in base_edges:
            base_of_edge[lab] = lab
            continue
        if isinstance(lab, tuple) and len(lab) == 2 and lab[0] in base_edges:
            base_of_edge[lab] = lab[0]
            continue
        raise ValueError(f"lifted edge {lab!r} carries no base edge label")

    rows, base_walks = face_map_pi(session.cover, session.base)
    degree_profile = {}
    for i, row in enumerate(rows):
        degree_profile[i] = {"base_face": row["target"], "degree": row["degree"],
                             "projection": tuple(row["projection"])
                             if row["projection"] is not None else None}
    phi_rows, _ = face_map_phi(session.phi)
    active, collapsed = {}, {}
    for i, row in enumerate(phi_rows):
        if row["tight"] and row["target"] is not None:
            active[i] = {"base_face": row["target"], "degree": row["degree"]}
        else:
            collapsed[i] = {"target": row["target"], "tight": row["tight"]}
    ok = all(r["target"] is not None for r in rows)
    return {
        "base": qb, "lifted": qc,
        "base_fan": fb, "lifted_fan": fc,
        "base_of_edge": base_of_edge,
        "degree_profile": degree_profile,
        "active": active,
        "collapsed": collapsed,
        "face_check": {"projects_to_base_face": ok,
                       "base_walks": [tuple(d[0] for d in w) for w in base_walks]},
        "n_base_punctures": qb["counts"]["V"],
        "n_lifted_punctures": qc["counts"]["V"],
    }


# ---------------------------------------------------------------------------
# 5. one Thurston step on the frozen cover
# ---------------------------------------------------------------------------

def _chordal(x, i, j):
    return float(np.linalg.norm(np.asarray(x[i]) - np.asarray(x[j])))


def step(frozen, positions, x0=None, maxit=400, tol=1e-13):
    """One pullback step on the frozen lifted cover.

    The base metric at the configuration is read off the configuration itself:
    every coarse edge of the base quotient is a pair of punctures (the dual
    edge of a spine edge), so its length is the chordal distance of the two
    punctures' positions.  Each spoke keeps the ratio the frozen regular metric
    gave it, rescaled by the geometric mean of its face's current side lengths,
    so the base complex is a function of the configuration and not of a
    representative.  Lifted edges inherit their base edge's length; the lifted
    complex is then realised and the new base configuration is read at the
    active lifted punctures (labelled by phi's target face).

    Möbius covariance of the STEP is NOT asserted: chordal lengths need not be
    related by one vertex potential after a general Möbius map, and the spoke
    rule above is only a convenient positive choice.  Acceptance therefore
    compares labelled configuration coordinates, not per-vertex positions.
    """
    qb, fb = frozen["base"], frozen["base_fan"]
    base_len = {}
    for lab, (i, j) in fb["edges"].items():
        if isinstance(lab, tuple) and lab[:2] == ("s", "b"):
            continue
        base_len[lab] = _chordal(positions, i, j)
    # spokes: frozen ratio to the face's two sides
    for k, poly in enumerate(qb["polygons"]):
        n = len(poly)
        sides = [base_len.get(qb["edge_of_corner"][k][j], None) for j in range(n)]
        sides = [s if s and s > 0 else 1.0 for s in sides]
        for j in range(n):
            lab = ("s", "b", k, j)
            cur = math.sqrt(sides[j] * sides[(j + 1) % n])
            base_len[lab] = cur / (2.0 * math.sin(math.pi / n))
    qc = frozen["lifted"]
    lifted = {}
    for lab in frozen["lifted_fan"]["edges"]:
        if isinstance(lab, tuple) and len(lab) == 4 and lab[0] == "s":
            continue                       # spokes are filled in below
        base = frozen["base_of_edge"].get(lab)
        if base is None:
            raise ValueError(f"lifted edge {lab!r} has no base length")
        if base not in base_len:
            raise ValueError(f"base edge {base!r} missing from the step's metric")
        lifted[lab] = base_len[base]
    for k, poly in enumerate(qc["polygons"]):
        n = len(poly)
        sides = [lifted[qc["edge_of_corner"][k][j]] for j in range(n)]
        for j in range(n):
            lifted[("s", "c", k, j)] = (math.sqrt(sides[j] * sides[(j + 1) % n])
                                        / (2.0 * math.sin(math.pi / n)))
    real = realise(frozen["lifted_fan"], lifted, x0=x0, maxit=maxit, tol=tol)
    x = real["x"]
    new = {}
    for i, row in frozen["active"].items():
        bf = row["base_face"]
        if bf in new:
            raise ValueError(f"two active lifted punctures share base face {bf}")
        new[bf] = x[i]
    out = {"positions": [new[i] for i in sorted(new)],
           "base_faces": sorted(new),
           "realisation": real, "metric": lifted,
           "converged": real["converged"], "excess": real["excess"],
           "residual": real["residual"]}
    out["complete"] = (len(new) == frozen["n_base_punctures"]
                       and out["converged"])
    return out


def iterate(frozen, positions, n, **kw):
    """`step` n times, returning the trajectory of configurations."""
    cur = list(positions)
    hist = []
    for _ in range(int(n)):
        r = step(frozen, cur, **kw)
        hist.append(r)
        if not r["complete"]:
            break
        cur = r["positions"]
    return hist
