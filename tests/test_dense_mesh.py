import unittest

import numpy as np

from pullback import dense_assembly as A
from pullback.dense_mesh import (DenseMeshError, branch_ring, dense_base,
                                 subdivide_triangles)


class TestDenseMesh(unittest.TestCase):
    def tetrahedron(self):
        phi = np.sqrt(3.0)
        return [np.array(new).astype(float) / phi for new in
                [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]]

    def test_ring_respects_local_tangent_frame(self):
        centre = np.array([0.4, -0.3, np.sqrt(0.75)])
        centre /= np.linalg.norm(centre)
        points = branch_ring(centre, radius=0.03, count=8, phase=0.13)
        for p in points:
            self.assertAlmostEqual(float(np.dot(p, centre)), np.cos(0.03), places=13)
            self.assertAlmostEqual(float(np.linalg.norm(p)), 1.0, places=13)
        self.assertEqual(len(points), 8)
        with self.assertRaises(ValueError):
            branch_ring(centre, radius=0.0)

    def test_dense_base_and_degree_three_sheet_certificates(self):
        p = self.tetrahedron()
        base = dense_base(p, level=1,
                          rings=[{"point": p[0], "count": 12, "radius": 0.03},
                                {"point": p[3], "count": 12, "radius": 0.045}])
        self.assertEqual(base.certificate()["chi"], 2)
        c3 = A.perm_from_cycles(3, [(0, 1, 2)])
        i = A.perm_identity(3)
        local = {v: c3 if v == 0 else
                   A.perm_inverse(c3) if v == 1 else i
                   for v in base.as_assembly_base().vertices}
        perm = A.perms_from_local(base.as_assembly_base(), 3, local)
        lift = A.assemble(base.as_assembly_base(), 3, perm)
        self.assertTrue(lift["cert"]["cover_ok"])
        self.assertTrue(lift["cert"]["orientation_consistent"])
        self.assertTrue(lift["cert"]["ribbon_ok"])
        self.assertTrue(lift["cert"]["chi_lift_ok"])
        self.assertTrue(lift["cert"]["rh_ok"])
        self.assertEqual(lift["d"], 3)

    def test_feature_subdivision_preserves_labels_and_incidence(self):
        X0 = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                       [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
        F0 = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
              (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
        X1, F1 = subdivide_triangles(X0, F0)
        self.assertEqual(len(X1), 18)
        self.assertEqual(len(F1), 32)
        np.testing.assert_allclose(X1[:6], X0, atol=1e-15, rtol=0)
        base = A.base_from_face_lists(F1)
        self.assertEqual((base.n_vertices(), base.n_edges(), base.n_faces()),
                         (18, 48, 32))
        self.assertEqual(base.euler(), 2)
        for v in base.vertices:
            self.assertEqual(len(base.flower(v)), base.face_degree[v])
        X2, F2 = subdivide_triangles(X0, F0, levels=2)
        self.assertEqual((len(X2), len(F2)), (66, 128))
        self.assertEqual(A.base_from_face_lists(F2).euler(), 2)

    def test_feature_subdivision_rejects_bad_faces_and_midpoints(self):
        X = np.array([[1, 0, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)
        with self.assertRaises(DenseMeshError):
            subdivide_triangles(X, [(0, 1, 1)])
        with self.assertRaisesRegex(DenseMeshError, "antipodal midpoint"):
            subdivide_triangles(X, [(0, 1, 2)], validate_hull=False)

    def test_bad_or_degenerate_input_is_not_silently_repaired(self):
        with self.assertRaises(ValueError):
            dense_base(self.tetrahedron()[:3], level=1)
        with self.assertRaises(DenseMeshError):
            dense_base(self.tetrahedron() + [self.tetrahedron()[0]])
        with self.assertRaises(ValueError):
            dense_base([np.array([1.0, 0, 0]) * k for k in range(4)])


if __name__ == "__main__":
    unittest.main()
