import unittest

import numpy as np

from pullback.analytic_metrics import edge_metrics, rational_certificate
from pullback.mesh import Mesh
from pullback.rational import RationalMap


class View:
    def __init__(self, points, source, faces):
        self.mesh = Mesh(points, faces, source=source)

    @property
    def points(self):
        return self.mesh.points

    @property
    def source(self):
        return self.mesh.source

    def edges(self):
        return self.mesh.edges()


class TestAnalyticMetrics(unittest.TestCase):
    def test_rabbit_analytic_lift_two_refinement_stages(self):
        # The same production lift that later solved the analytic rabbit is
        # also the reference for a 58-vertex dense base.  Do not infer it from
        # a conformal-lift formula or from a small sphere alone.
        from pullback import discrete, mesh, thurston
        f = thurston.rabbit_map()
        data = discrete.pcf_data(f)
        base = discrete.base_mesh(data["points"], f, refine=1, max_insert=40)
        lifted, _info = mesh.lift_by_edges(base, f)
        cert = rational_certificate(f, lifted, base, tol=1e-18)
        self.assertEqual((len(base.points), len(lifted.points), len(lifted.faces)),
                         (58, 114, 224))
        self.assertEqual(len(lifted.points) - len(lifted.edges()) + len(lifted.faces), 2)
        self.assertEqual(cert["ramified_vertices"], [2, 5])
        product = edge_metrics(f, lifted, base, family="graph_product")
        self.assertLess(min(product.values()), 0.035)
        self.assertGreater(min(product.values()), 0.034)
        self.assertLess(max(product.values()), 1.53)

    def test_degree_three_critical_points_map_to_critical_values(self):
        # f(z)=z^3-3z: over -2, 2 there is one double and one simple
        # preimage; over infinity there is a degree-three preimage.  This
        # intentionally exercises general ramification, not just a square root.
        f = RationalMap([1, 0, -3, 0], [1])
        # z=0,+/-sqrt(3) are the three simple preimages of 0,
        # z=+/-1 are double preimages of -2,+2, and z=+i maps to -4i.
        base = Mesh([0, -4j, -2, 2, np.inf], [(0, 1, 2), (0, 2, 3), (0, 3, 4),
                                           (0, 4, 1)])
        source = [(0, "root0"), (0, "root1"), (2, 0), (3, 0),
                  (1, 0), (0, "root2"), (4, 0)]
        positions = [0, np.sqrt(3), 1, -1, 1j, -np.sqrt(3), np.inf]
        lifted = View(positions, source, [(0, 1, 2), (3, 4, 5), (5, 6, 0)])
        cert = rational_certificate(f, lifted, base)
        self.assertEqual(sorted(cert["ramified_vertices"]), [2, 3, 6])
        self.assertEqual([r["local_degree"] for r in cert["vertices"]],
                         [1, 1, 2, 2, 1, 1, 3])
        metrics = edge_metrics(f, lifted, base, family="graph_product")
        self.assertEqual(len(metrics), 9)
        self.assertTrue(all(v > 0 for v in metrics.values()))

    def test_graph_affine_is_nonsingular_at_a_ramified_point(self):
        f = RationalMap([1, 0, 0], [1])       # z^2
        base = Mesh([0, 1], [(0, 1, 1)])
        lifted = View([0, 1, -1], [(0, 0), (1, "+"), (1, "-")],
                      [(0, 1, 2), (0, 2, 1)])
        rational_certificate(f, lifted, base)
        metrics = edge_metrics(f, lifted, base, family="graph_affine")
        self.assertAlmostEqual(metrics[(0, 1)], np.sqrt(2.0))
        self.assertGreater(metrics[(0, 1)], metrics[(0, 2)] * 0.0)

    def test_wrong_sheet_is_refused_not_merged(self):
        f = RationalMap([1, 0, 0], [1])
        base = Mesh([0, 1], [(0, 1, 1)])
        lifted = View([0, 2, -1], [(0, 0), (1, "wrong"), (1, "-")],
                      [(0, 1, 2), (0, 2, 1)])
        with self.assertRaisesRegex(ValueError, "not over base vertex"):
            rational_certificate(f, lifted, base)

    def test_product_metric_is_finite_at_a_pole(self):
        f = RationalMap([1], [1, 0])     # 1/z
        base = Mesh([1, np.inf, 0], [(0, 1, 2), (0, 2, 1)])
        lifted = View([1, 0, np.inf], [(0, 0), (1, 0), (2, 0)],
                      [(0, 1, 2), (0, 2, 1)])
        rational_certificate(f, lifted, base)
        self.assertTrue(edge_metrics(f, lifted, base, family="graph_product")[(0, 1)] > 0.0)
        with self.assertRaisesRegex(ValueError, "finite chart"):
            edge_metrics(f, lifted, base, family="graph_affine")


if __name__ == "__main__":
    unittest.main()
