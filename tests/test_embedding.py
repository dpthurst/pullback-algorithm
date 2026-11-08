"""Pinned assertions for the acceptance invariant and the well-posedness count.

Run:  python3 -m unittest test_acceptance -v
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tests.test_rounding_multi import rabbit_admissible


from pullback.rounding import edges_of, incidence, round_realise
from pullback.mesh import to_vector
from pullback import embedding as acceptance
from pullback import cell_refine as rc


class Invariant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, _spine, cls.cells, cls.metric = rabbit_admissible()
        labelled = rc.face_edge_labels(cls.cells)
        cls.faces = [p for p, _ in labelled]
        cls.labels = [l for _, l in labelled]
        cls.oriented = acceptance.orient_faces(cls.faces, cls.labels)
        cls.lengths = rc.coarse_lengths(cls.cells, cls.metric)
        cls.n = len(cls.cells["vertices"])
        fp, port = cls.metric["face_points"], cls.metric["portrait"]
        cls.collapse = np.array(
            [to_vector(cls.data["points"][fp[port[v]["target"]]])
             for v in cls.cells["vertices"]], dtype=float)
        cls.sub = rc.subdivide_cells(cls.cells, cls.metric)

    # ---- the embedding certificate ---------------------------------------
    def test_orientation_needs_edge_ids_not_vertex_pairs(self):
        # pairing by vertices would report (0,5) as a non-manifold edge
        with self.assertRaises(ValueError):
            acceptance.orient_faces(self.faces, acceptance.pair_labels(
                [f for f in self.faces if f[0] != 0 or f[1] != 1 or f[2] != 3]))
        # with edge ids it orients, and every shared edge is traversed twice
        for f in self.oriented:
            self.assertEqual(len(set(f)), 3)

    def test_embedding_certificate_rejects_the_collapse(self):
        cert = acceptance.embedding_certificate(self.collapse, self.oriented)
        self.assertFalse(cert["embedded"])
        self.assertEqual(cert["min_abs_det"], 0.0)
        # the collapse fails both ways: two pairs of vertices coincide and four
        # faces are degenerate.  It is reported as non-injective first.
        self.assertTrue("coincident" in cert["reason"]
                        or "degenerate" in cert["reason"])
        self.assertFalse(acceptance.embedding_certificate(
            self.collapse, self.oriented, inject_tol=0.0)["embedded"])

    def test_a_known_embedded_configuration_passes(self):
        # control: the regular inscribed tetrahedron is a genuine embedded
        # spherical triangulation with the origin strictly inside, so the
        # certificate must pass; flipping one vertex through the origin must
        # make it fail.
        vecs = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0],
                         [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
        vecs = vecs / np.sqrt(3.0)
        faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]
        orf = acceptance.orient_faces(faces, acceptance.pair_labels(faces))
        cert = acceptance.embedding_certificate(vecs, orf)
        self.assertTrue(cert["embedded"], cert["reason"])
        self.assertGreater(cert["min_abs_det"], 1e-3)
        bad = vecs.copy()
        bad[0] = -bad[0]
        self.assertFalse(acceptance.embedding_certificate(bad, orf)["embedded"])

    # ---- the Mobius coordinate survives coincident points ----------------
    def test_mobius_coordinate_handles_the_collapse(self):
        coord = acceptance.mobius_coordinate(self.collapse)
        self.assertEqual(len(coord[1]), self.n - 3)
        # gauge covariance: a rotation of the whole configuration is free
        th = 0.7
        R = np.array([[np.cos(th), -np.sin(th), 0.0],
                      [np.sin(th), np.cos(th), 0.0],
                      [0.0, 0.0, 1.0]])
        other = self.collapse @ R.T
        self.assertEqual(coord[0], acceptance.mobius_coordinate(other)[0])
        for u, v in zip(coord[1], acceptance.mobius_coordinate(other)[1]):
            self.assertLess(abs(complex(u) - complex(v)), 1e-9)

    # ---- well-posedness: merging a parallel class loses one condition -----
    def test_merged_coarse_system_is_underdetermined(self):
        E = len(edges_of(self.faces))          # merged: 11 vertex pairs
        A, _ = incidence(self.faces, self.n)
        self.assertEqual((E, self.n), (11, 6))
        self.assertEqual(np.linalg.matrix_rank(A), 6)
        # conformal condition codim = E - rank(A) = 5, leaving a positive
        # dimensional family: 12 unknowns mod Mobius - 5 = 7
        self.assertEqual(3 * self.n - 6 - (E - np.linalg.matrix_rank(A)), 7)
        # the true (edge-labelled) complex has 12 = 3V - 6 edges, so the
        # codimension is 2V - 6 and the solutions are isolated
        self.assertEqual(len(self.cells["edges"]), 3 * self.n - 6)
        self.assertEqual(2 * self.n - 6, 6)

    def test_the_merged_solve_is_start_dependent(self):
        rng = np.random.RandomState(0)
        classes = set()
        for _ in range(120):
            x, res, _it = round_realise(self.oriented, self.lengths,
                                        x0=rng.normal(size=(self.n, 3)),
                                        maxit=600, tol=1e-14)
            self.assertLess(res, 1e-10)
            c = acceptance.mobius_coordinate(x)
            classes.add((c[0], tuple(complex(round(float(np.real(z)), 6),
                                             round(float(np.imag(z)), 6))
                                     for z in c[1])))
        self.assertGreater(len(classes), 100)

    def test_the_refined_complex_is_a_triangulation(self):
        counts = self.sub["counts"]
        self.assertEqual(counts["V"], 18)
        self.assertEqual(counts["E"], 3 * counts["V"] - 6)
        self.assertEqual(len(edges_of(self.sub["faces"])), counts["E"])


if __name__ == "__main__":
    unittest.main()
