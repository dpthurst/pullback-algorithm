"""The mesh machinery, and the obstruction that stops the discrete step.

Everything here is checkable without sigma, and each check has a case on which
it fails: `validate` on a corrupted face, `orient_faces` on a mesh lying in a
hemisphere (the case that caught the outward-normal rule), the bigon over a
base edge joining two critical values, the diameter that `subdivide` cannot cut,
and the hemisphere face whose centroid collapses onto a vertex.
"""

import sys
import unittest

import mpmath as mp
import numpy as np

sys.path.insert(0, "src")

from pullback import mesh as msh  # noqa: E402
from pullback import thurston as th  # noqa: E402
from pullback.discrete import (base_mesh, pcf_data,  # noqa: E402
                               single_preimage_vertices)
from pullback.moduli import Configuration  # noqa: E402
from pullback.session import Session  # noqa: E402

mp.mp.dps = 40


def setUpModule():
    """Pin the working precision for this module.

    mpmath's precision is global and other test modules set it to their own
    value, so a module-level assignment at import time is not enough: the test
    that compares the derivative with a finite difference failed at 3.9e-12
    because a *previous* module had left the precision at 30 digits, where a
    step of 1e-20 loses ten digits to cancellation.
    """
    mp.mp.dps = 40


def octahedron():
    pts = [0.0, 1.0j, -1.0, -1.0j, complex(np.inf), 0.0]
    pts = [0.0, 1.0j, -1.0, -1.0j, complex(np.inf)]
    # five points are not an octahedron; build the real one
    pts = [1.0, -1.0, 1.0j, -1.0j, 0.0, complex(np.inf)]
    return msh.Mesh(pts, msh.sphere_triangulation(pts), name="octahedron")


class TestChart(unittest.TestCase):
    def test_vector_roundtrip_including_infinity(self):
        for z in [0.0, 1.0, -2.5 + 0.7j, complex(np.inf)]:
            w = msh.to_point(msh.to_vector(z))
            if mp.isfinite(mp.mpc(z)):
                self.assertLess(abs(w - complex(z)), 1e-12)
            else:
                self.assertFalse(np.isfinite(np.real(w)))

    def test_labelled_cross_ratio_agrees_with_the_configuration_chart(self):
        pts = [0.0, 1.0, complex(np.inf), 2.0]
        want = Configuration(["p0", "p1", "p2", "p3"], pts).coordinate()
        got = msh.labelled_cross_ratio([msh.to_vector(p) for p in pts])
        self.assertLess(abs(want - got), 1e-12)

    def test_labelled_cross_ratio_is_invariant_under_a_rotation(self):
        rng = np.random.RandomState(3)
        q, _r = np.linalg.qr(rng.normal(size=(3, 3)))
        pts = [0.3 + 0.2j, -1.0 + 0.9j, 2.0 - 0.4j, 0.5j]
        v = [msh.to_vector(p) for p in pts]
        a = msh.labelled_cross_ratio(v)
        b = msh.labelled_cross_ratio([q @ x for x in v])
        self.assertLess(abs(a - b), 1e-12)


class TestValidation(unittest.TestCase):
    def test_the_octahedron_and_the_tetrahedron_validate(self):
        self.assertEqual(octahedron().validate(), {"V": 6, "E": 12, "F": 8})
        pts = [0.0, 1.0, 1.0j, 2.0 + 0.5j]
        m = msh.Mesh(pts, msh.sphere_triangulation(pts))
        self.assertEqual(m.validate(), {"V": 4, "E": 6, "F": 4})

    def test_a_flipped_face_is_refused(self):
        m = octahedron()
        faces = list(m.faces)
        faces[0] = (faces[0][0], faces[0][2], faces[0][1])
        with self.assertRaises(ValueError):
            msh.Mesh(m.points, faces).validate()

    def test_a_repeated_vertex_is_refused(self):
        m = octahedron()
        faces = list(m.faces)
        faces[0] = (faces[0][0], faces[0][0], faces[0][1])
        with self.assertRaises(ValueError):
            msh.Mesh(m.points, faces).validate()

    def test_a_missing_face_is_refused(self):
        m = octahedron()
        with self.assertRaises(ValueError):
            msh.Mesh(m.points, m.faces[:-1]).validate()


class TestOrientation(unittest.TestCase):
    def test_scrambled_orientations_are_repaired(self):
        m = octahedron()
        scrambled = []
        for k, f in enumerate(m.faces):
            scrambled.append(f if k % 3 else (f[0], f[2], f[1]))
        fixed = msh.orient_faces(m.points, scrambled)
        self.assertEqual(msh.Mesh(m.points, fixed).validate(), {"V": 6, "E": 12, "F": 8})

    def test_a_mesh_in_a_hemisphere_orientates(self):
        # the outward-normal rule needs the origin inside the mesh and fails
        # here; the two-colouring does not.  Four points of the rabbit's
        # postcritical set are not in a hemisphere, so use four clustered points.
        pts = [0.0, 0.2, 0.2j, 0.15 + 0.15j]
        faces = msh.sphere_triangulation(pts)
        self.assertEqual(msh.Mesh(pts, faces).validate()["F"], 4)


class TestRefinement(unittest.TestCase):
    def test_face_centre_refinement_counts(self):
        pts = [0.0, 1.0, 1.0j, 2.0 + 0.5j]
        m = msh.Mesh(pts, msh.sphere_triangulation(pts))
        r = msh.subdivide_face_centres(m)
        self.assertEqual(r.validate(), {"V": 8, "E": 18, "F": 12})

    def test_edge_midpoint_refinement_refuses_a_diameter(self):
        # the rabbit's postcritical set contains 0 and infinity, which are
        # antipodal: the midpoint of that chord is the origin.  The faces are
        # written out because `sphere_triangulation` cannot handle this point
        # set -- with an antipodal pair most quadruples are coplanar with the
        # origin and it finds no face at all, which is itself worth knowing.
        pts = [0.0, complex(np.inf), 1.0, -1.0]
        faces = msh.orient_faces(pts, [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
        m = msh.Mesh(pts, faces)
        m.validate()
        with self.assertRaises(ValueError):
            msh.subdivide(m)

    def test_antipodal_pairs_are_chart_independent(self):
        pts = [0.0, complex(np.inf), 1.0, -2.0]
        pairs = msh.antipodal_pairs(pts)
        self.assertEqual(pairs, [(0, 1)])
        _M, _Minv, moved = msh.pole_free_chart(pts)
        self.assertEqual(msh.antipodal_pairs([mp.mpc(z) for z in moved]), [(0, 1)])

    def test_pole_free_chart_preserves_the_metric(self):
        pts = [0.0, complex(np.inf), 0.6 + 0.3j, -1.2 + 0.4j]
        faces = msh.orient_faces(pts, [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
        m = msh.Mesh(pts, faces)
        before = m.chordal_lengths()
        _M, _Minv, moved = msh.pole_free_chart(pts)
        after = msh.Mesh(moved, m.faces).chordal_lengths()
        for e in before:
            self.assertLess(abs(before[e] - after[e]), 1e-10)
        for z in moved:
            self.assertTrue(mp.isfinite(mp.mpc(z)))


class TestFaceSample(unittest.TestCase):
    def test_a_hemisphere_face_has_no_usable_centroid(self):
        # a face with an antipodal pair: the normalised centroid is a vertex,
        # so a path lift from it would start on the critical value
        vectors = np.array([msh.to_vector(p) for p in
                            [0.0, complex(np.inf), 0.5 + 0.5j]])
        centroid = vectors.sum(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        self.assertLess(min(np.linalg.norm(centroid - v) for v in vectors), 1e-6)
        sample = msh.face_sample(vectors, (0, 1, 2))
        self.assertGreater(min(np.linalg.norm(sample - v) for v in vectors), 1e-3)


class TestBaseMeshAndTheObstruction(unittest.TestCase):
    def setUp(self):
        self.f = th.rabbit_map()
        self.d = pcf_data(self.f)

    def test_the_rabbit_tetrahedron_cannot_be_lifted(self):
        """The measured obstruction.

        A degree-2 map with four postcritical points has two critical values,
        and the tetrahedron on four points joins every pair -- so its edge
        between them lifts to a bigon.  Separately, 0 and infinity are
        antipodal, so two of its faces are hemispheres.  So the coarsest mesh is
        unusable, and it is not a numerical accident.
        """
        pts = [mp.mpc(p) if mp.isfinite(p) else mp.inf for p in self.d["points"]]
        self.assertEqual(single_preimage_vertices(self.d["points"], self.f), {1, 3})
        faces = msh.sphere_triangulation(pts)
        m = msh.Mesh(pts, faces, name="tetrahedron")
        m.validate()
        with self.assertRaises(msh.DegenerateLift) as ctx:
            msh.lift_by_edges(m, self.f)
        self.assertIn("single-preimage", str(ctx.exception).split(":")[0])
        bad_faces, bad_edges = msh.mesh_defects(pts, faces)
        self.assertTrue(bad_faces and bad_edges)

    def test_a_five_vertex_bipyramid_still_has_a_bad_edge(self):
        # the bipyramid with the two critical values at the poles breaks the
        # bigon but puts infinity next to 0, which is antipodal to it
        pts = [mp.mpc(p) if mp.isfinite(p) else mp.inf for p in self.d["points"]]
        pts = pts + [mp.mpc(msh.spread_points(self.d["points"], 1)[0])]
        faces = msh.bipyramid_mesh(pts, (1, 3))
        m = msh.Mesh(pts, faces, name="bipyramid")
        m.validate()
        bad_faces, _bad_edges = msh.mesh_defects(pts, faces)
        self.assertTrue(bad_faces)

    def test_base_mesh_breaks_critical_value_edges(self):
        # both maps have two critical values in P, so the tetrahedron always has
        # a bigon edge; the builder inserts vertices until it does not
        for f in (th.slit_rational_map(), self.f):
            d = pcf_data(f)
            m = base_mesh(d["points"], f)
            self.assertGreaterEqual(m.inserted, 1)
            E = {msh.edge_key(fc[k], fc[(k + 1) % 3])
                 for fc in m.faces for k in range(3)}
            pairs = msh.antipodal_pairs(m.points)
            singles = single_preimage_vertices(
                [mp.mpc(p) if mp.isfinite(p) else mp.inf for p in m.points], f)
            for (i, j) in E:
                self.assertFalse(i in singles and j in singles,
                                 f"edge {(i, j)} still joins two critical values")
                self.assertNotIn((min(i, j), max(i, j)),
                                 [(min(x, y), max(x, y)) for (x, y) in pairs],
                                 f"edge {(i, j)} is a diameter")

    def test_a_point_set_on_a_great_circle_has_no_triangulation(self):
        # the slit map's postcritical set {2, 1, -2, infinity} lies on the real
        # line: four points on a great circle, so no triangle is a face
        pts = [mp.mpf(2), mp.mpf(1), mp.mpf(-2), mp.inf]
        with self.assertRaises(ValueError):
            msh.sphere_triangulation(pts)


def cube():
    """The cube: a valid *polygonal* cell decomposition of S^2 (six quads)."""
    V = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
         (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    Q = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
         (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3)]
    pts = [msh.to_point(np.array(v, dtype=float)) for v in V]
    return msh.Mesh(pts, msh.orient_faces(pts, Q), name="cube")


class TestDelaunayAndVoronoiSpine(unittest.TestCase):
    """The spine adapted to a configuration, and why it is not a triangulation.

    The author's reformulation (2026-10-28) builds the spine from the *current*
    positions: Delaunay-decompose the punctures, dualise, and use the result as
    the spine Gamma_0'.  Two things follow that the tests pin: the Delaunay
    decomposition is not always a triangulation (a cocircular block is one
    polygonal face, and choosing a diagonal inside it would be non-canonical), and
    the dual is a genuine spine -- one face per puncture -- unlike the ribbon-graph
    dual of a session's spine, whose faces are that spine's vertices.
    """

    def rabbit_points(self):
        from pullback.discrete import pcf_data
        d = pcf_data(th.rabbit_map())
        return [mp.mpc(p) if mp.isfinite(p) else mp.inf for p in d["points"]]

    def test_four_points_in_general_position_give_the_tetrahedron(self):
        faces = msh.delaunay_faces(self.rabbit_points())
        self.assertEqual(sorted(len(f) for f in faces), [3, 3, 3, 3])

    def test_a_cocircular_block_is_one_polygon(self):
        # four points coplanar at z = 0.5 with everything else below: their plane
        # is supporting, so the block is the quad -- not two triangles chosen by
        # a diagonal
        ring = [msh.to_point(np.array([0.6 * np.cos(2 * np.pi * k / 4),
                                       0.6 * np.sin(2 * np.pi * k / 4), 0.5]))
                for k in range(4)]
        pts = ring + [msh.to_point(np.array([0.0, 0.0, -1.0]))]
        faces = msh.delaunay_faces(pts)
        self.assertEqual(sorted(len(f) for f in faces), [3, 3, 3, 3, 4])

    def test_points_in_one_hemisphere_are_not_special(self):
        """The author's point (2026-10-28): Delaunay is conformally invariant.

        Formulated with empty circles, the condition is unchanged by a Mobius map,
        and then "all the points lie in a hemisphere" is not a degenerate case at
        all -- it is a position, not a property.  This was a degenerate case in the
        chordal/hull picture (a great circle is a plane through the origin), and
        the test pins that it is not one now.
        """
        great = [mp.mpc(1), mp.mpc(1j), mp.mpc(-1), mp.mpc(-1j),
                 mp.mpc("0.2") + mp.mpc("0.3") * 1j]
        faces = msh.delaunay_faces(great)
        # one polygonal block (the four points on the great circle) plus triangles
        self.assertEqual(sorted(len(f) for f in faces), [3, 3, 3, 3, 4])
        self.assertEqual(msh.voronoi_spine(great).faces() and
                         len(msh.voronoi_spine(great).faces()), 5)

    def test_the_decomposition_is_mobius_invariant(self):
        """The check that can fail, and would have failed for a chordal version.

        A random Mobius map moves the points; the empty-circle condition does not
        change, so the Delaunay decomposition must come back combinatorially the
        same -- same set of faces, as sets of labels.
        """
        from pullback.moduli import apply_mobius
        M = (mp.mpc("0.6") + mp.mpc("0.2") * 1j, mp.mpc("0.3") - mp.mpc("0.1") * 1j,
             mp.mpc("-0.2") + mp.mpc("0.4") * 1j, mp.mpc("0.9") + mp.mpc("0.1") * 1j)
        cases = [
            [mp.mpc(0), mp.mpc(1), mp.mpc(1j), mp.mpc("0.3") + mp.mpc("0.4") * 1j,
             mp.mpc(-1)],
            [mp.mpc(0), mp.mpc(1), mp.mpc(1j), mp.mpc("0.3") + mp.mpc("0.4") * 1j,
             mp.mpc(-1), mp.mpc("-0.5") + mp.mpc("0.8") * 1j],
            self.rabbit_points(),
            [mp.mpc(1), mp.mpc(1j), mp.mpc(-1), mp.mpc(-1j),
             mp.mpc("0.2") + mp.mpc("0.3") * 1j],
        ]
        for pts in cases:
            before = {frozenset(f) for f in msh.delaunay_faces(pts)}
            moved = [mp.mpc(apply_mobius(M, p if mp.isfinite(p) else mp.inf))
                     for p in pts]
            after = {frozenset(f) for f in msh.delaunay_faces(moved)}
            self.assertEqual(before, after,
                             msg="not Mobius invariant for %d points" % len(pts))

    def test_the_voronoi_spine_of_the_rabbit_is_a_spine(self):
        pts = self.rabbit_points()
        sp = msh.voronoi_spine(pts)
        V, E, F = len(sp.vertices()), len(sp.edge_list()), len(sp.faces())
        self.assertEqual((V, E, F), (4, 6, 4))
        self.assertEqual(V - E + F, 2)
        # one face per puncture, and each of the punctures has valence 3 here
        self.assertEqual(sorted(len(w) for w in sp.faces()), [3, 3, 3, 3])

    def test_voronoi_edge_labels_are_primal_delaunay_edges(self):
        """A dual edge carries the base edge whose length it must inherit.

        The old code labelled it by its two dual endpoints instead.  Those are
        both pairs of integers on the tetrahedron, so the defect is invisible
        until the labels are used to attach metric data.
        """
        pts = self.rabbit_points()
        faces = msh.delaunay_faces(pts)
        sp = msh.voronoi_spine(pts)
        expected = {}
        for i, face in enumerate(faces):
            for k in range(len(face)):
                edge = msh.edge_key(face[k], face[(k + 1) % len(face)])
                other = next(j for j, g in enumerate(faces)
                             if j != i and edge[0] in g and edge[1] in g)
                expected[edge] = {i, other}
        self.assertEqual(set(sp.edge_list()), set(expected))
        for edge, endpoints in expected.items():
            self.assertEqual(set(sp.edges[edge]), endpoints, msg=str(edge))
        self.assertEqual(msh.voronoi_face_points(sp, len(pts)),
                         {0: 1, 1: 0, 2: 2, 3: 3})

    def test_the_spine_dualises_back_to_the_delaunay_decomposition(self):
        pts = self.rabbit_points()
        faces = msh.delaunay_faces(pts)
        sp = msh.voronoi_spine(pts)
        # the spine's vertices are the Delaunay faces, so its vertex count is the
        # face count of the decomposition
        self.assertEqual(len(sp.vertices()), len(faces))

    def test_the_ribbon_dual_of_a_session_spine_is_a_different_object(self):
        # for contrast, and because it caused a confusion worth pinning: the
        # ribbon-graph dual of the rabbit's bouquet spine has ONE face (the
        # bouquet's single vertex), so it is a spine of the sphere minus one
        # point, not minus P
        s = Session.load("sessions/rabbit.json")
        d = s.base.dual()
        self.assertEqual(len(d.vertices()), 4)
        self.assertEqual(len(d.edge_list()), 3)
        self.assertEqual(len(d.faces()), 1)


class TestPolygonalFaces(unittest.TestCase):
    """Faces of any size, because the mesh dual to a spine is polygonal.

    The data this is for: the dual of a spine has one face per spine vertex, so a
    valence-k spine vertex gives a k-gon; and the *canonical* Delaunay
    decomposition is not always a triangulation -- a cocircular block is a
    polygon, and choosing a diagonal inside it is a non-canonical act.
    """

    def test_a_polygonal_sphere_validates(self):
        c = cube()
        self.assertEqual(c.validate(), {"V": 8, "E": 12, "F": 6})
        self.assertEqual(sorted(len(f) for f in c.faces), [4] * 6)
        # and the pillow: three vertices, two triangles -- the theta spine's dual
        pts = [0.0, 1.0, 1.0j]
        pillow = msh.Mesh(pts, msh.orient_faces(pts, [(0, 1, 2), (0, 2, 1)]))
        self.assertEqual(pillow.validate(), {"V": 3, "E": 3, "F": 2})

    def test_a_polygonal_face_visiting_a_vertex_twice_is_refused(self):
        # the dual of a spine with a LOOP has such a face (the rabbit's bouquet
        # gives a hexagonal face whose walk doubles back), and it is not a disc
        # with distinct corners, so no flat metric lives on it
        pts = [0.0, 1.0, 1.0j, 2.0]
        bad = msh.Mesh(pts, [(0, 1, 0, 2)])
        with self.assertRaises(ValueError):
            bad.validate()

    def test_round_realise_works_on_polygonal_faces(self):
        from pullback.rounding import round_realise
        c = cube()
        lens = c.chordal_lengths()
        rng = np.random.RandomState(0)
        x0 = c.vectors() + 0.08 * rng.normal(size=(8, 3))
        x0 /= np.linalg.norm(x0, axis=1)[:, None]
        _x, res, iters = round_realise(c.faces, lens, x0=x0, maxit=200, tol=1e-13)
        self.assertLess(res, 1e-9)
        self.assertLess(iters, 200)

    def test_the_star_triangulates_and_preserves_the_rotation(self):
        c = cube()
        star = msh.subdivide_face_centres(c)
        self.assertEqual(star.validate(), {"V": 14, "E": 36, "F": 24})
        self.assertTrue(all(len(f) == 3 for f in star.faces))
        for v in c.rotation():
            old = c.rotation()[v]
            got = [x for x in star.rotation()[v] if x in set(old)]
            self.assertEqual(len(got), len(old))
            k = got.index(old[0])
            self.assertEqual(got[k:] + got[:k], list(old),
                             msg="vertex %d: %s -> %s" % (v, old, got))

    def test_refine_mesh_triangulates_before_it_subdivides(self):
        from pullback.discrete import refine_mesh
        m = refine_mesh(cube())
        self.assertTrue(all(len(f) == 3 for f in m.faces))
        self.assertEqual(m.validate(), {"V": 50, "E": 144, "F": 96})
        m2 = refine_mesh(m)
        self.assertEqual(m2.validate(), {"V": 194, "E": 576, "F": 384})

    def test_edge_midpoint_subdivision_refuses_a_polygon_and_says_what_to_do(self):
        with self.assertRaises(ValueError) as ctx:
            msh.subdivide(cube())
        self.assertIn("subdivide_face_centres", str(ctx.exception))


class TestRefinementConverges(unittest.TestCase):
    """The refinement must converge to the round metric -- the check that was missing.

    A vertex scaling l~_ij = exp((u_i+u_j)/2) l_ij changes every triangle angle,
    so cone angles are *not* invariants of discrete conformal equivalence (that is
    what the theory is for: choose u to uniformize them).  What must converge under
    refinement is the cone angle at each vertex, to the value 2*pi of the smooth
    limit.  An earlier refinement in this module (face centres at skewed sample
    points) made the puncture angles *diverge* -- 4.53, 7.07, 9.61, 12.08 rad --
    and every conclusion drawn from it was void.  These tests are that failure's
    regression test.
    """

    def _cone_angles(self, mesh):
        v = np.array([msh.to_vector(p) for p in mesh.points])
        tot = np.zeros(len(mesh.points))
        for face in mesh.faces:
            for k in range(3):
                i, j, l = face[k], face[(k + 1) % 3], face[(k + 2) % 3]
                a, b = v[j] - v[i], v[l] - v[i]
                cos = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
                tot[i] += np.arccos(max(-1.0, min(1.0, cos)))
        return tot

    def test_vertex_scaling_changes_triangle_angles(self):
        # the fact that the wrong claim got backwards, computed directly
        import math
        ell = {(0, 1): 1.0, (1, 2): 1.3, (0, 2): 1.7}
        u = {0: 0.0, 1: 0.9, 2: -0.4}

        def angles(e):
            a, b, c = e[(1, 2)], e[(0, 2)], e[(0, 1)]
            A = lambda x, y, z: math.acos((y * y + z * z - x * x) / (2 * y * z))
            return A(a, b, c), A(b, c, a), A(c, a, b)
        before = angles(ell)
        after = angles({k: math.exp((u[k[0]] + u[k[1]]) / 2) * v
                        for k, v in ell.items()})
        # the angles change by tens of degrees, and their sum stays pi
        self.assertGreater(max(abs(a - b) for a, b in zip(before, after)), 0.3)
        self.assertAlmostEqual(sum(before), sum(after), places=12)

    def test_the_puncture_angles_converge_to_two_pi(self):
        from pullback.discrete import base_mesh, refine_mesh
        f = th.rabbit_map()
        d = pcf_data(f)
        m = base_mesh(d["points"], f, refine=0)
        deficits = []
        for _ in range(4):
            ca = self._cone_angles(m)
            deficits.append([abs(2 * np.pi - ca[i]) for i in range(4)])
            m = refine_mesh(m)
        for i in range(4):
            col = [d[i] for d in deficits]
            # strictly decreasing, and by a factor near 4 each level (O(h^2))
            for k in range(len(col) - 1):
                self.assertLess(col[k + 1], col[k] * 0.6,
                                msg="puncture %d: deficits %s do not fall" % (i, col))

    def test_the_face_centre_refinement_diverges(self):
        """The failure, kept: the skew in `face_sample` compounds.

        This is not a claim that face-centre refinement is wrong in general -- it
        is that *this* one, with skewed interior points, diverges, and that a
        refinement study built on it measures nothing.
        """
        from pullback.discrete import base_mesh
        f = th.rabbit_map()
        d = pcf_data(f)
        m = base_mesh(d["points"], f, refine=0)
        for _ in range(3):
            m = msh.subdivide_face_centres(m)
        ca = self._cone_angles(m)
        # at least one puncture angle is far above 2*pi -- the divergence
        self.assertGreater(max(ca[:4]), 2 * np.pi + 1.0)


class TestWhereTheOrderCannotComeFrom(unittest.TestCase):
    def test_the_round_geometry_does_not_give_the_cyclic_order(self):
        """The measured reason the faces are built combinatorially.

        At a lifted vertex of local degree 1 the cyclic order of the edge-ends
        is forced -- it is the base order at the vertex it lies over -- but the
        round positions do not show it: the round metric is not the pulled-back
        flat metric, so the lifted triangles are not drawn as non-crossing
        triangles.  Measured on the rabbit with a two-cap base mesh, at the
        lifted vertex over c^2+c, whose five base edges are indexed 3:infinity,
        4:rho1, 5:rho2, 1:c, 7:rho4: the tangent order of the analytic preimages
        is 3, 7, 4, 5, 1 while the combinatorial order is 3, 4, 1, 7, 5 -- and the
        pair they disagree about, 4 and 7, is 1.18 degrees apart, so no tolerance
        separates them.
        """
        f = th.rabbit_map()
        d = pcf_data(f)
        pts = list(d["points"]) + [mp.mpc(p)
                                   for p in msh.spread_points(d["points"], 4)]
        faces = msh.two_cap_mesh(pts, 3, 2, [0, 4, 5], [1, 6, 7])
        m = msh.Mesh(pts, faces, name="twocap")
        m.validate()
        # this mesh has no diameter and no critical-value edge, so the lift gets
        # as far as the faces -- and the geometry-based order would have been
        # read there, wrongly, from the tangent directions
        with self.assertRaises(msh.DegenerateLift):
            msh.lift_by_edges(m, f)

    def test_congruent_lengths_agrees_with_the_lifted_faces(self):
        # a hand-built two-face "lift": the length of an edge must not depend on
        # which of its two faces supplied it
        base = msh.Mesh([0.0, 1.0, 1.0j], [(0, 1, 2)])
        lifted_points = [0.0, 1.0, 1.0j, -1.0, -1.0j]
        lifted = msh.Mesh(lifted_points, [(0, 1, 2), (0, 3, 4)],
                          source=[(0, 0), (1, 0), (2, 0), (1, 0), (2, 0)])
        lens = msh.congruent_lengths(base, lifted)
        # every lifted edge carries its base edge's length, and the two lifted
        # edges over one base edge must agree -- the assertion that fails when
        # a lift is not congruent
        self.assertEqual(len(lens), 6)
        self.assertAlmostEqual(lens[(0, 1)], lens[(0, 3)], places=12)
        self.assertAlmostEqual(lens[(0, 2)], lens[(0, 4)], places=12)
        self.assertAlmostEqual(lens[(1, 2)], lens[(3, 4)], places=12)


if __name__ == "__main__":
    unittest.main()
