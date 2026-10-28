"""The discrete pullback step, and its refinement knob.

One step, in the labelled Teichmüller chart:

    tau  ->  points of P on S^2
         ->  a triangulated base mesh with those points as vertices, metric =
             chordal lengths (an inscribed polyhedron)
         ->  the congruent lift through f  (`mesh.lift`)
         ->  `rounding.round_realise`  -- the discrete uniformisation
         ->  read the diagonal copy of P inside f^{-1}(P)  ->  tau'

The state is a *labelled* configuration (`moduli.Configuration`): the preimage
of a point of P is labelled by it, so the labels travel through f|_P, and a
fixed point of the step is the labelled statement sigma(tau) = tau.

Two things to keep apart when reading the numbers this returns.

* `residual` is `round_realise`'s optimum value: how far the pulled-back flat
  metric is from being exactly discretely conformally equivalent to an
  inscribed polyhedron *on this mesh*.  It measures the discretisation, and it
  is expected to fall as the mesh refines -- that is the convergence the study
  measures, and it is not the same question as whether the step's fixed point
  converges to the analytic one.
* `step_error` is the distance between the returned chart coordinate and the
  input's.  At a genuine solution of sigma(tau) = tau this is the quantity
  that has to vanish; away from one it is just the size of one step.

The refinement families are `mesh.subdivide`, which puts an edge midpoint (on
the sphere) into every edge.  The old vertices keep their positions, and the
polyhedron's geometry converges to the sphere's; it is *not* the same mesh
refined by the pullback tower, and the study reports both.
"""

from __future__ import annotations

import mpmath as mp
import numpy as np

from . import mesh as msh
from .moduli import cross_ratio
from .rounding import round_realise
from .thurston import _order_key


def pcf_data(f, depth=64, tol=mp.mpf("1e-25")):
    """The labelled postcritical configuration of a critically finite map.

    Labels are integers in a canonical order (finite points by (re, im), then
    infinity), and `dynamics` sends a label to the label of its image -- the
    labelled marking the Teichmüller chart needs.  Nothing here is read from a
    session, so the same object serves the analytic examples and, later, the
    engine's own data.
    """
    pts, has_inf, _tol = f.postcritical_set(depth=depth, tol=tol)
    points = [mp.mpc(p) for p in pts] + ([mp.inf] if has_inf else [])
    order = sorted(range(len(points)), key=lambda i: _order_key(points[i]))
    points = [points[i] for i in order]
    dynamics = {}
    for i, p in enumerate(points):
        img = f(mp.mpc(p)) if mp.isfinite(p) else f.f_infinity()
        best, bestd = None, None
        for j, q in enumerate(points):
            if not mp.isfinite(q) and not mp.isfinite(img):
                d = mp.mpf(0)
            elif mp.isfinite(q) and mp.isfinite(img):
                d = abs(mp.mpc(img) - mp.mpc(q))
            else:
                d = mp.inf
            if bestd is None or d < bestd:
                best, bestd = j, d
        if bestd > mp.mpf("1e-15"):
            raise ArithmeticError(
                f"the image of point {i} is not in P (distance {mp.nstr(bestd, 5)}):"
                " the map is not postcritically finite")
        dynamics[i] = best
    return {"points": points, "dynamics": dynamics,
            "degrees": f.portrait_degrees()}


def chart_coordinate(vectors):
    """The Teichmüller chart on the labelled mesh: pin the first three labelled
    points to (infinity, z1, z2) and read the rest.

    A cross-ratio for |P| = 4 (`labelled_cross_ratio`), a tuple of them beyond
    that.  Gauge-free: it depends only on the Möbius class of the input, so the
    solver's arbitrary normalisation does not enter.
    """
    if len(vectors) == 4:
        return msh.labelled_cross_ratio(vectors)
    vs = [np.asarray(v, dtype=float) for v in vectors]
    p0 = vs[0] / np.linalg.norm(vs[0])
    north = np.array([0.0, 0.0, 1.0])
    d = p0 - north
    if np.linalg.norm(d) < 1e-15:
        R = np.eye(3)
    elif np.linalg.norm(p0 + north) < 1e-15:
        R = np.diag([1.0, -1.0, -1.0])
    else:
        u = d / np.linalg.norm(d)
        R = np.eye(3) - 2.0 * np.outer(u, u)
    pts = [msh.to_point(R @ v) for v in vs]
    return tuple(cross_ratio(pts[0], pts[1], pts[2], pts[k])
                 for k in range(3, len(pts)))


def single_preimage_vertices(points, f):
    """Base vertices with exactly one distinct preimage -- the critical values.

    An edge joining two of them lifts to a *bigon*: the preimage of the open
    edge is d arcs and all of them join the same two points, so the lifted
    faces are not triangles and the vertex-index representation cannot hold
    them.  This is the generic case, not a corner: a degree-2 map with four
    postcritical points has two critical values, and the tetrahedron on four
    points joins every pair.
    """
    return {i for i, p in enumerate(points)
            if len(f.preimage_with_degrees(p)) == 1}


def base_mesh(points, f, refine=0, faces=None, max_insert=40, spine=None,
              marking=None):
    """The base mesh: `points` as the first vertices, a triangulated sphere,
    then `refine` face-centre refinements.

    Three things stop the coarsest mesh from being usable, and all three are
    properties of the data rather than numerical accidents:

    * **The hull may not be a triangulation.**  Convex position is not enough:
      if the origin is not strictly inside the hull the point set lies in a
      closed hemisphere, no triangle is a face, and there is no triangulation at
      all (three points on a great circle is the usual cause, and *every*
      polynomial has c^2+c, c, 0 and infinity on the real line if its parameter
      is real).  A generic extra vertex fixes it.
    * **A critical-value edge lifts to a bigon** (`single_preimage_vertices`),
      so an extra vertex goes inside a face containing it.
    * **A face with an antipodal pair is a hemisphere** and has no usable
      interior point; `mesh.face_sample` copes, but the construction is happier
      without one.

    Vertices are inserted until the mesh validates and has no critical-value
    edge; `inserted` records how many.  The loop is a heuristic: it gives up
    with a message rather than looping, and a caller that needs a specific mesh
    can pass `faces`.
    """
    pts = [mp.mpc(p) if mp.isfinite(p) else mp.inf for p in points]
    single = single_preimage_vertices(points, f)
    inserted = 0
    if faces is None:
        extra = iter(msh.spread_points(points, max_insert + 4))
        faces = None
        while inserted <= max_insert:
            if faces is not None:
                try:
                    msh.Mesh(pts, faces).validate()
                except ValueError:
                    msh.Mesh(pts, faces)          # raises with its own message
            try:
                faces = msh.sphere_triangulation(pts)
                msh.Mesh(pts, faces).validate()
            except ValueError:
                pts.append(mp.mpc(next(extra)))
                inserted += 1
                continue
            E = {msh.edge_key(fc[k], fc[(k + 1) % 3])
                 for fc in faces for k in range(3)}
            diameters = {e for e in E if np.linalg.norm(
                msh.to_vector(pts[e[0]]) + msh.to_vector(pts[e[1]])) < 1e-6}
            bad = sorted(({e for e in E if e[0] in single and e[1] in single})
                         | diameters)
            if not bad:
                break
            e = bad[0]
            face = next(fc for fc in faces if e[0] in fc and e[1] in fc)
            vecs = np.array([msh.to_vector(p) for p in pts])
            pts.append(mp.mpc(msh.to_point(msh.face_sample(vecs, face))))
            inserted += 1
            faces = None
        else:
            raise ValueError(
                f"could not build a liftable mesh in {max_insert} insertions")
    m = msh.Mesh(pts, faces, name="base")
    m.validate()
    for _ in range(refine):
        m = msh.subdivide_face_centres(m)
    m.inserted = inserted
    # The mesh is a ribbon graph: it carries the rotation system, and the
    # refinement above preserved it.  When the spine is given, check the mesh's
    # topological order against the ribbon structure -- the order at the punctures
    # is the spine's face-boundary order, reversed, and the mesh's rotation must
    # contain it.
    m.ribbon_order = m.rotation()
    m.ribbon_consistent = None
    if spine is not None:
        if marking is None:
            markings = msh.consistent_markings(m, spine)
            m.ribbon_consistent = bool(markings)
            m.consistent_markings = markings
        else:
            m.ribbon_consistent = msh.marking_consistent_with_ribbon(
                m, spine, marking)
            if not m.ribbon_consistent:
                raise ValueError(
                    "the mesh's topological order does not agree with the"
                    " ribbon structure under this marking")
    return m


def _diagonal_vertex(lifted, base_index, wanted):
    """The lifted vertex over base vertex `base_index` whose position is
    `wanted` -- the diagonal copy of P in f^{-1}(P)."""
    best, bestd = None, None
    for i, (b, _k) in enumerate(lifted.source):
        if b != base_index:
            continue
        p = lifted.points[i]
        if not mp.isfinite(p) and not mp.isfinite(wanted):
            d = 0.0
        elif mp.isfinite(p) and mp.isfinite(wanted):
            d = float(abs(mp.mpc(p) - mp.mpc(wanted)))
        else:
            continue
        if bestd is None or d < bestd:
            best, bestd = i, d
    if best is None or bestd is None or bestd > 1e-10:
        raise ArithmeticError(
            f"no lifted vertex over base vertex {base_index} at the diagonal"
            f" preimage (best distance {bestd})")
    return best


def pullback_step(points, dynamics, f, refine=0, faces=None, maxit=200,
                  tol=1e-13, x0_mode="analytic", seed=0):
    """One discrete pullback step of a labelled configuration.

    `points` are the labelled positions of P (label i at points[i]), `dynamics`
    the map on the labels (label -> label of the image).  Returns a dict with
    the new positions (`points`), the chart coordinates before and after
    (`coordinate`, `coordinate_out`), the uniformisation residual, and the mesh
    sizes.  `dynamics` is used only to identify the diagonal copy of P: for
    label i the preimage used is the point labelled j with dynamics[j] == i.

    `x0_mode="analytic"` starts the solve at the analytic preimages, which have
    the right combinatorics by construction; `"random"` is the control that
    tests whether the initial guess is doing the work.
    """
    base = base_mesh(points, f, refine=refine, faces=faces)
    # This is where the step currently stops: the congruent lift needs the
    # cyclic order of the edge-ends at each lifted vertex, and `mesh.lift_by_edges`
    # does not have it yet (see notes/thurston-pullback.md, "the discrete step,
    # attempted").  The error is a DegenerateLift, and it is deliberate: no
    # number from this function should be trusted until the lift is a
    # triangulation.
    lifted, stats = msh.lift_by_edges(base, f)
    lengths = msh.congruent_lengths(base, lifted)
    if x0_mode == "analytic":
        x0 = lifted.vectors()
    elif x0_mode == "random":
        x0 = None
    else:
        raise ValueError(f"unknown x0_mode {x0_mode!r}")
    x, residual, iters = round_realise(lifted.faces, lengths, x0=x0,
                                       maxit=maxit, tol=tol, seed=seed)
    # the diagonal copy: for label i, the point j with f(p_j) = p_i
    # The diagonal copy of P in f^{-1}(P), and the labelling convention that
    # makes the fixed-point statement come out right.
    #
    # The new configuration's label p sits at the position of the *source* point
    # that is p -- the point of the source sphere marked p, which lies over
    # f(p), not over p.  (Reading it the other way round -- label p at the
    # preimage of p -- returns the input configuration relabelled by f|_P, a
    # quantity independent of the mesh, which is how the mistake was caught:
    # the step error came out identical to 12 digits at every refinement.)
    #
    # This needs f|_P to be a permutation of P -- a Thurston marking.  It is not
    # for every map: for the slit map f(1) = f(-2) = 1, so 1 has two preimages
    # inside P and 2 has none, and there is no diagonal to read.
    images = [dynamics[lab] for lab in sorted(dynamics)]
    if sorted(images) != sorted(dynamics):
        raise ValueError(
            "P is not a Thurston marking for this map: f|_P is not a"
            " permutation of P, so the diagonal copy of P in f^{-1}(P) is"
            " ambiguous and the labelled fixed-point statement is not defined")
    diag = []
    for i in range(len(points)):
        diag.append(_diagonal_vertex(lifted, dynamics[i], points[i]))
    out = [x[v] for v in diag]
    coord_in = chart_coordinate([msh.to_vector(p) for p in points])
    coord_out = chart_coordinate(out)
    if isinstance(coord_in, tuple):
        step_error = max(abs(a - b) for a, b in zip(coord_in, coord_out))
    else:
        step_error = abs(coord_in - coord_out)
    return {
        "points": [msh.to_point(v) for v in out],
        "vectors": out,
        "coordinate": coord_in,
        "coordinate_out": coord_out,
        "step_error": step_error,
        "residual": residual,
        "iterations": iters,
        "base_faces": len(base.faces),
        "base_vertices": len(base.points),
        "inserted_vertices": getattr(base, "inserted", 0),
        "lifted_vertices": stats["vertices"],
        "lifted_faces": stats["faces"],
        "discretisation_error": stats["discretisation_error"],
    }


def iterate(points, dynamics, f, n, refine=0, faces=None, **kw):
    """Iterate the discrete step and report the trajectory of the chart
    coordinate, so a reader can see whether the iteration settles."""
    out = []
    cur = list(points)
    for k in range(n):
        res = pullback_step(cur, dynamics, f, refine=refine, faces=faces, **kw)
        out.append({"n": k + 1, "coordinate": res["coordinate_out"],
                    "step_error": res["step_error"],
                    "residual": res["residual"],
                    "points": res["points"]})
        cur = res["points"]
    return out
