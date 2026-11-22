"""Frozen-cover Thurston pullback for a known analytic rational map.

The combinatorial cover is built once at the analytic postcritical marking,
then held fixed.  Auxiliary base positions move by the Moebius map determined
by three finite marks; source labels remain vertices in that fixed cover rather
than being guessed geometrically.  This is the validated known-map counterpart
of the graph-VE construction.  It is not yet a constructor for an abstract
session without an analytic marking/local cover.
"""
from __future__ import annotations

import mpmath as mp
import numpy as np

from pullback import mesh as msh
from pullback.discrete import _diagonal_vertex, chart_coordinate
from pullback.rounding import round_realise

mp.mp.dps = 40


def critical_values(f):
    """The critical values of f (finite ones and infinity), as a list."""
    out = []
    for c in f.critical_points():
        v = f(mp.mpc(c)) if mp.isfinite(c) else f.f_infinity()
        if not any(_close(v, w) for w in out):
            out.append(v)
    if f.local_degree_infinity() > 1:
        v = f.f_infinity()
        if not any(_close(v, w) for w in out):
            out.append(v)
    return out


def _close(a, b, tol=mp.mpf("1e-20")):
    if not mp.isfinite(a) and not mp.isfinite(b):
        return True
    if not mp.isfinite(a) or not mp.isfinite(b):
        return False
    return abs(a - b) < tol


def single_preimage_vertices(points, f):
    """Base vertices with exactly one distinct preimage (the critical values)."""
    return {i for i, p in enumerate(points)
            if len(f.preimage_with_degrees(p)) == 1}


def base_mesh_with_critical_values(points, f, max_insert=40, extra=(), refine=0):
    """A liftable base mesh: `points`, plus the critical values of f that are
    missing, plus inserted points until the mesh validates and has no
    critical-value edge and no diameter."""
    pts = [mp.mpc(p) if mp.isfinite(p) else mp.inf for p in points]
    for v in critical_values(f):
        if not any(_close(v, mp.mpc(p) if mp.isfinite(p) else mp.inf) for p in pts):
            pts.append(mp.mpc(v) if mp.isfinite(v) else mp.inf)
    for v in extra:
        pts.append(v)
    n_given = len(pts)
    spare = iter(msh.spread_points(points, max_insert + 4))
    inserted = 0
    faces = None
    while inserted <= max_insert:
        try:
            faces = msh.sphere_triangulation(pts)
            msh.Mesh(pts, faces).validate()
        except ValueError:
            pts.append(mp.mpc(next(spare)))
            inserted += 1
            faces = None
            continue
        E = {msh.edge_key(fc[k], fc[(k + 1) % 3])
             for fc in faces for k in range(3)}
        single = single_preimage_vertices(pts, f)
        vecs = np.array([msh.to_vector(p) for p in pts])
        diameters = {e for e in E if np.linalg.norm(vecs[e[0]] + vecs[e[1]]) < 1e-6}
        bad = sorted(({e for e in E if e[0] in single and e[1] in single})
                     | diameters)
        if not bad:
            break
        e = bad[0]
        face = next(fc for fc in faces if e[0] in fc and e[1] in fc)
        pts.append(mp.mpc(msh.to_point(msh.face_sample(vecs, face))))
        inserted += 1
        faces = None
    else:
        raise ValueError("could not build a liftable mesh")
    m = msh.Mesh(pts, faces, name="base")
    m.validate()
    for _ in range(refine):
        from pullback.discrete import refine_mesh
        m = refine_mesh(m)
    m.validate()
    m.inserted = inserted
    m.n_configuration = len(points)
    m.single = single
    return m


class FrozenStep:
    """A pullback step whose combinatorics are fixed at the analytic marking.

    `mesh.lift_by_edges` builds the lifted mesh from the *positions*: it
    computes preimages of the current positions, pairs the arcs, and infers the
    local degrees.  That is well defined at the analytic marking, where the
    critical values are among the marked positions, and nowhere else -- which is
    why the second iterate raises DegenerateLift, and why the labels of the two
    sheets over a base vertex are only pinned by a positional rule that ties.

    But sigma does not need the lift to be rebuilt at all.  The combinatorial
    cover of the base mesh by f is a single finite datum: over base vertex j the
    source vertices are the preimages of the *abstract* point p_j, with the local
    degrees f has there; every base edge lifts to deg f arcs; every base face to
    deg f triangles.  All of it is computed once, at the analytic marking, where
    `lift_by_edges` succeeds, and can then be held fixed.  A step at any
    configuration is then: move the base marked vertices to the new positions,
    carry auxiliary vertices by the Moebius map matching three finite marks,
    give every source edge its base edge's chordal length (the congruent lift is
    a length assignment, not a positional one), and uniformise.

    The labelled configuration is read at *frozen* source vertices: the ones
    identified at the analytic marking with the marked points p_i.  There is no
    preimage list to order, so there is no sheet to choose: the sheet is which
    source vertex it is, and that is combinatorial data of the cover.
    """

    def __init__(self, points, f, refine=0, maxit=200, tol=1e-13):
        self.f = f
        self.points = [mp.mpc(p) if mp.isfinite(p) else mp.inf for p in points]
        base = base_mesh_with_critical_values(self.points, f, refine=refine)
        lifted, stats = msh.lift_by_edges(base, f)
        self.base = base
        self.lifted = lifted
        self.stats = stats
        self.refine = refine
        self.maxit, self.tol = maxit, tol
        # the frozen label vertices: at the analytic marking the source point p_i
        # sits at points[i] over base vertex dynamics[i], and that lifted vertex
        # IS p_i.  Read once, here, and never again.
        self.marks = [i for i in range(len(self.points))]
        # indices of the base vertices that are marked points
        self.base_marks = list(range(len(points)))
        self.label_vertices = None          # set by `identify`
        self.x = lifted.vectors()           # a start for the first solve

    def identify(self, dynamics):
        self.dynamics = dynamics
        lab = []
        for i in range(len(self.points)):
            lab.append(_diagonal_vertex(self.lifted, dynamics[i], self.points[i]))
        self.label_vertices = lab
        return lab

    def base_at(self, points):
        """The frozen base mesh with the marked vertices moved to `points`.

        The auxiliary vertices are moved by the Moebius map that carries the
        three finite marked points of the base marking to their new positions.
        That is the criterion the auxiliary vertices were missing: they are not
        fixed points of the sphere (which would crowd a marked point as soon as
        it moved and would never refine away), they are *functions of the
        configuration*, carried by the same Moebius map as the marked points.
        With it the base structure at every configuration is the Moebius image
        of the base structure at the marking, so the discretisation is a
        function of the labelled Teich point rather than of a representative.
        """
        pts = [mp.mpc(p) if mp.isfinite(p) else mp.inf for p in points]
        aux = list(self.base.points[len(self.points):])
        triple = [i for i in range(len(points)) if mp.isfinite(self.points[i])
                  and mp.isfinite(pts[i])][:3]
        if len(triple) == 3 and not (
                triple == [0, 1, 2] and all(self.points[i] == pts[i]
                                            for i in triple)):
            m = msh._mobius_through([(self.points[i], pts[i]) for i in triple])
            aux = [msh._mp_apply(m, z) for z in aux]
        return msh.Mesh(pts + aux, self.base.faces, name="frozen-base")

    def step(self, points, x0=None):
        base = self.base_at(points)
        lengths = msh.congruent_lengths(base, self.lifted)
        x, residual, iters = round_realise(
            self.lifted.faces, lengths,
            x0=self.x if x0 is None else x0, maxit=self.maxit, tol=self.tol)
        if self.label_vertices is None:
            raise ValueError("call identify(dynamics) before stepping")
        vectors = [x[v] for v in self.label_vertices]
        out = [msh.to_point(v) for v in vectors]
        self.x = x
        return {"points": out, "vectors": vectors,
                "coordinate": chart_coordinate(vectors),
                "residual": residual, "iterations": iters,
                "base_faces": len(base.faces), "x": x, "base": base,
                "label_vertices": list(self.label_vertices),
                "refine": self.refine}


    def iterate(self, points, n):
        """Apply `step` n times with the frozen cover and fixed source labels.

        Returned records include the labelled coordinate at each iterate.  The
        fixed source marks, not proximity, determine the sheet, so the solver's
        previous realisation can be used safely to continue the local basin.
        """
        cur = list(points)
        out = []
        for _ in range(int(n)):
            result = self.step(cur)
            cur = result["points"]
            out.append(result)
        return out
