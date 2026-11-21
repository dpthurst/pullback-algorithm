"""Per-edge, arbitrary-degree branched-cover assembly certificates."""
import random
import unittest

import numpy as np

from pullback import dense_assembly as A
from pullback import embedding, rounding_multi as mg


TETRA = [(0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)]


def assert_certificate(test, lift, d, rh=None):
    c = lift["cert"]
    test.assertTrue(c["cover_ok"], c)
    test.assertTrue(c["edge_in_two_faces"], c)
    test.assertTrue(c["orientation_consistent"], c)
    test.assertTrue(c["ribbon_ok"], c)
    test.assertTrue(c["fibre_ok"], c)
    test.assertEqual(c["rh_deficit"], 2 * d - 2 if rh is None else rh)
    test.assertEqual(c["chi_lift"], 2)
    test.assertTrue(c["connected"], c)
    test.assertTrue(c["is_sphere"], c)
    return c


class TestDenseAssembly(unittest.TestCase):
    def test_degree_two_and_three_with_different_branching(self):
        ident2, sw = (0, 1), (1, 0)
        base = A.base_from_face_lists(TETRA)
        perms = A.perms_from_local(base, 2, {0: sw, 1: ident2,
                                           2: ident2, 3: sw})
        c = assert_certificate(self, A.assemble(base, 2, perms), 2)
        self.assertEqual(c["counts"]["lift"], [6, 12, 8])
        self.assertEqual(c["fibres"], {"0": [2], "1": [1, 1],
                                        "2": [1, 1], "3": [2]})
        self.assertEqual(c["chi_lift"], 2)

        c3 = A.perm_from_cycles(3, [(0, 1, 2)])
        ident3 = A.perm_identity(3)
        perms = A.perms_from_local(base, 3, {0: c3, 1: A.perm_inverse(c3),
                                           2: ident3, 3: ident3})
        c = assert_certificate(self, A.assemble(base, 3, perms), 3)
        self.assertEqual(c["counts"]["lift"], [8, 18, 12])
        self.assertEqual(c["fibres"], {"0": [3], "1": [3],
                                        "2": [1, 1, 1], "3": [1, 1, 1]})
        self.assertEqual(c["chi_lift"], 2)

    def test_nontrivial_global_monodromy_is_not_reduced_to_degrees(self):
        d = 4
        local = [(3, 0, 1, 2), (0, 1, 2, 3),
                 (1, 2, 3, 0), A.perm_identity(d)]
        base = A.base_from_face_lists(TETRA)
        perms = A.perms_from_local(base, d, dict(zip(range(4), local)))
        c = assert_certificate(self, A.assemble(base, d, perms), d)
        self.assertEqual(c["counts"]["lift"], [10, 24, 16])
        self.assertEqual(c["fibres"], {"0": [4], "1": [1, 1, 1, 1],
                                        "2": [4], "3": [1, 1, 1, 1]})
        self.assertEqual(c["chi_lift"], 2)

    def test_cut_tree_independence_and_arbitrary_datum(self):
        # Changing the cut changes presentation, not lift or ramification datum.
        base = A.base_from_face_lists(TETRA)
        sw, ident = (1, 0), (0, 1)
        local = {0: sw, 1: ident, 2: ident, 3: sw}
        results = []
        for root in base.vertices:
            perms = A.perms_from_local(base, 2, local, root=root)
            lift = A.assemble(base, 2, perms)
            assert_certificate(self, lift, 2)
            results.append((lift["cert"]["counts"]["lift"], lift["cert"]["fibres"]))
            self.assertEqual(lift["cert"]["chi_lift"], 2)
        self.assertTrue(all(row == results[0] for row in results[1:]))
        self.assertEqual(results[0][0], [6, 12, 8])

        # An arbitrary per-edge datum need not yield a sphere.  The kernel of
        # the assembly must nevertheless close, remain oriented and satisfy
        # the exact covering/Gauss--Bonnet identities for every random datum.
        rng = random.Random(11)
        for d in (2, 3, 5, 7):
            for _ in range(3):
                perms = {}
                for edge in base.edges:
                    p = list(range(d))
                    rng.shuffle(p)
                    perms[edge] = tuple(p)
                lift = A.assemble(base, d, perms)
                c = lift["cert"]
                self.assertTrue(c["cover_ok"] and c["edge_in_two_faces"])
                self.assertTrue(c["orientation_consistent"])
                self.assertTrue(c["ribbon_ok"])
                self.assertEqual(c["chi_lift"], 2 * d - c["rh_deficit"])

    def test_chi_two_and_single_links_do_not_imply_connectedness(self):
        # The fixed third sheet is an unbranched sphere.  The other two sheets
        # form a torus double cover with four simple branch points.  Every old
        # count passes, including chi=2 and Riemann--Hurwitz, but this is not a
        # connected spherical lift and must be reported as such.
        base = A.base_from_face_lists(TETRA)
        transposition = (1, 0, 2)
        perms = A.perms_from_local(
            base, 3, {v: transposition for v in base.vertices})
        c = A.assemble(base, 3, perms)["cert"]
        self.assertTrue(c["cover_ok"], c)
        self.assertTrue(c["orientation_consistent"], c)
        self.assertTrue(c["ribbon_ok"], c)
        self.assertTrue(c["fibre_ok"], c)
        self.assertTrue(c["rh_ok"], c)
        self.assertEqual(c["chi_lift"], 2)
        self.assertEqual(c["components"], 2)
        self.assertFalse(c["connected"])
        self.assertFalse(c["is_sphere"])

    def test_permutation_inverse_composition_and_local_realisation(self):
        p = A.perm_from_cycles(5, [(0, 2, 4), (1, 3)])
        q = A.perm_inverse(p)
        self.assertEqual(A.perm_compose(p, q), A.perm_identity(5))
        self.assertEqual(A.perm_order(p), 6)
        self.assertEqual(A.perm_cycle_type(p), (3, 2))
        base = A.base_from_face_lists(TETRA)
        perms = dict.fromkeys(base.edges, tuple(range(5)))
        perms[next(iter(base.edges))] = p
        lift = A.assemble(base, 5, perms)
        self.assertEqual(A.local_perms(base, perms, 5),
                         lift["cert"]["local_realised"])


    def test_collapse_is_a_degree_two_control_not_a_lift(self):
        # Exact independent-font statistic: use a one-sign radial decomposition
        # of the round tetrahedron, not a hand-built "near-flat" approximation.
        base = A.base_from_face_lists(TETRA)
        X = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
                     dtype=float) / np.sqrt(3.0)
        perms = A.perms_from_local(base, 2,
                                   {0: (1, 0), 1: (0, 1), 2: (0, 1), 3: (1, 0)})
        lift = A.assemble(base, 2, perms)
        collapsed = X[[lift["proj"][u] for u in range(lift["cert"]["counts"]["lift"][0])]]
        lengths = {label: float(np.linalg.norm(X[label[0][0]] - X[label[0][1]]))
                   for label in lift["edges"]}
        residual = mg.conformal_residual(lift["edges"], lengths, collapsed)[2]
        self.assertLess(float(residual), 1e-11)
        orient = embedding.orient_faces(
            [vertices for vertices, _labels in lift["tris"]],
            [labels for _vertices, labels in lift["tris"]])
        c = embedding.classify(collapsed, orient)
        self.assertFalse(c["embedded"])
        self.assertAlmostEqual(abs(c["degree"]), 2.0)
        self.assertTrue(all(c["det"] < 0.0), c)
        self.assertLess(len(np.unique(np.round(collapsed, 12), axis=0)),
                        len(collapsed),
                        "a lift must not collapse the two sheets at a basepoint")


if __name__ == "__main__":
    unittest.main()
