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
