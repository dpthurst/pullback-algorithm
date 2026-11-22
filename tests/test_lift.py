"""Lifted Delaunay cells, including multigraph edges and cone angles.

The coarse rabbit lift is deliberately NOT a `Mesh`: a base Delaunay edge joins
its two critical values, so its two lifts join the same pair of vertices.  These
tests pin the native ribbon-dual representation rather than hiding that edge by
choosing a diagonal or by dropping one of the two edge labels.
"""

import glob
import sys
import unittest

import mpmath as mp
import numpy as np

sys.path.insert(0, "src")

from pullback import discrete, mesh, thurston as th  # noqa: E402
from pullback.homotopy import active_faces, basis_transport  # noqa: E402
from pullback.portrait import face_map_pi  # noqa: E402
from pullback.validate import is_pi1_surjective  # noqa: E402
from pullback.lift import (active_puncture_labels, dual_cells, dual_polygons,  # noqa: E402
                           lifted_delaunay_metric, parallel_edge_pairs,
                           pullback_ve, round_realise_cells)
from pullback.maps import make_graph_map  # noqa: E402
from pullback.ribbons import F, B, RibbonGraph, fmt_dart  # noqa: E402
from pullback.session import Session  # noqa: E402


class TestRabbitLiftedCells(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mp.mp.dps = 40
        cls.session = Session.load("sessions/rabbit.json")
        cls.data = discrete.pcf_data(th.rabbit_map())
        cls.spine = mesh.voronoi_spine(cls.data["points"])
        # This arbitrary basis transport is sufficient for testing the generic
        # native-dual and metric code.  It is NOT certified as the geometric h;
        # the peripheral-structure test lives with the homotopy code.
        rho = basis_transport(cls.spine, cls.session.base)
        cls.P, cls.pi_prime, cls.phi_prime, _pc = pullback_ve(
            cls.session.base, cls.session.cover, cls.session.phi, cls.spine, rho,
            validate_h=False)  # explicitly a negative structural fixture
        cls.cells = dual_cells(cls.P)
        cls.metric = lifted_delaunay_metric(
            cls.cells, cls.spine, cls.data["points"])

    def test_graph_map_aliases_have_the_right_target(self):
        self.assertIs(self.phi_prime.target, self.spine)
        self.assertIs(self.phi_prime.tgt, self.spine)
        self.assertIs(self.pi_prime.tgt, self.spine)

    def test_tuple_labelled_darts_print_their_orientation(self):
        forward = (((0, 1), 0), F)
        backward = (((0, 1), 0), B)
        self.assertNotEqual(fmt_dart(forward), fmt_dart(backward))
        self.assertIn("^-1", fmt_dart(backward))
        self.assertIn("^-1", repr(RibbonGraph({
            0: (forward,), 1: (backward,),
        })))

    def test_native_dual_has_the_lifted_delaunay_cells(self):
        self.assertEqual(self.cells["counts"], {"V": 6, "E": 12, "F": 8})
        self.assertTrue(all(len(p) == 3 for p in self.cells["polygons"]))
        expected = {
            (0, 1, 3), (1, 0, 5), (2, 1, 4), (1, 2, 5),
            (3, 1, 5), (4, 1, 5), (0, 3, 5), (2, 4, 5),
        }
        self.assertEqual(set(self.cells["polygons"]), expected)
        primal, polygons = dual_polygons(self.P)
        self.assertEqual(primal, self.P.faces())
        self.assertEqual(polygons, self.cells["polygons"])

    def test_the_doubled_edge_keeps_two_distinct_labels(self):
        parallel = parallel_edge_pairs(self.cells)
        self.assertEqual(list(parallel), [(1, 5)])
        labels = parallel[(1, 5)]
        self.assertEqual(len(labels), 2)
        self.assertEqual(len(set(labels)), 2)
        self.assertEqual(labels[0][0], labels[1][0])
        # Vertex-pair Mesh keys cannot represent this cell decomposition.
        with self.assertRaises(ValueError):
            mesh.Mesh([0] * 6, self.cells["polygons"]).validate()

    def test_every_lifted_edge_inherits_its_base_length(self):
        lengths = self.metric["lengths"]
        self.assertEqual(len(lengths), 12)
        base = self.metric["base_lengths"]
        for edge, length in lengths.items():
            self.assertAlmostEqual(length, base[tuple(sorted(edge[0]))], places=14)
        # In particular, neither copy of the doubled edge was discarded.
        doubled = parallel_edge_pairs(self.cells)[(1, 5)]
        self.assertEqual(len({lengths[e] for e in doubled}), 1)
        self.assertEqual(len({e for e in lengths if e in doubled}), 2)

    def test_cone_angles_are_base_angles_times_local_degree(self):
        cone = self.metric["cone_angles"]
        base = self.metric["base_cone_angles"]
        portrait = self.metric["portrait"]
        self.assertAlmostEqual(sum(base.values()), 4 * np.pi, places=12)
        for vertex in self.cells["vertices"]:
            row = portrait[vertex]
            point = self.metric["face_points"][row["target"]]
            self.assertAlmostEqual(
                cone[vertex], row["degree"] * base[point], places=14)
        self.assertEqual(self.metric["face_points"], {0: 1, 1: 0, 2: 2, 3: 3})
        hyper = self.metric["hyperideal"]
        self.assertEqual(len(hyper), 1)
        self.assertGreater(cone[hyper[0]], 2 * np.pi)
        self.assertAlmostEqual(cone[hyper[0]], 8.381992527641588, places=12)


class TestSelfTransport(unittest.TestCase):
    def test_pullback_preserves_the_portrait_on_every_original_spine(self):
        for path in sorted(glob.glob("sessions/*.json")):
            session = Session.load(path)
            with self.subTest(session=session.name):
                rho = basis_transport(session.base, session.base)
                P, pi_prime, phi_prime, _pc = pullback_ve(
                    session.base, session.cover, session.phi, session.base, rho)
                cells = dual_cells(P)

                def profile(cover, base):
                    out = {}
                    for row in face_map_pi(cover, base)[0]:
                        out.setdefault(row["target"], []).append(row["degree"])
                    return {k: sorted(v) for k, v in out.items()}

                self.assertEqual(profile(P, session.base),
                                 profile(session.cover, session.base))
                self.assertTrue(is_pi1_surjective(phi_prime, session.base))
                self.assertEqual(len(active_faces(phi_prime)),
                                 len(active_faces(session.phi)))
                self.assertEqual(cells["counts"]["V"], len(session.cover.faces()))
                # Non-simple boundary walks are retained, not silently rejected.
                self.assertEqual(
                    cells["counts"]["V"] - cells["counts"]["E"]
                    + cells["counts"]["F"], 2)


class TestAdmissibleRabbitCandidate(unittest.TestCase):
    """One short-word candidate passing all available geometric certificates.

    This is a known-answer engine fixture, not a claim that the unmarked session
    determines a unique isotopy class.  Keeping the candidate here makes the
    transported VE, labels and analytic coarse metric executable rather than
    merely counting faces of an arbitrary basis transport.
    """

    @classmethod
    def setUpClass(cls):
        mp.mp.dps = 40
        cls.session = Session.load("sessions/rabbit.json")
        cls.data = discrete.pcf_data(th.rabbit_map())
        cls.spine = mesh.voronoi_spine(cls.data["points"])
        chains = {
            (0, 1): (), (1, 2): (), (0, 2): (),
            (0, 3): (("b", F),),
            (1, 3): (("a", B),),
            (2, 3): (("c", F),),
        }
        rho = make_graph_map(
            cls.spine, cls.session.base, chains,
            {v: cls.session.base.vertices()[0] for v in cls.spine.vertices()})
        cls.P, cls.pi_prime, cls.phi_prime, _pc = pullback_ve(
            cls.session.base, cls.session.cover, cls.session.phi, cls.spine, rho)
        cls.cells = dual_cells(cls.P)
        cls.metric = lifted_delaunay_metric(
            cls.cells, cls.spine, cls.data["points"])

    def test_cover_profile_and_doubled_critical_value_edge(self):
        profile = {}
        for row in self.metric["portrait"]:
            point = self.metric["face_points"][row["target"]]
            profile.setdefault(point, []).append(row["degree"])
        self.assertEqual(profile, {0: [1, 1], 1: [2], 2: [1, 1], 3: [2]})
        parallel = parallel_edge_pairs(self.cells)
        self.assertEqual(list(parallel), [(0, 5)])
        self.assertEqual({e[0] for e in parallel[(0, 5)]}, {(1, 3)})
        self.assertEqual(self.cells["counts"], {"V": 6, "E": 12, "F": 8})
        self.assertTrue(all(len(p) == 3 for p in self.cells["polygons"]))

    def test_active_labels_satisfy_both_portrait_legs(self):
        labels = active_puncture_labels(
            self.phi_prime, self.metric["face_points"], self.pi_prime,
            self.data["dynamics"])
        self.assertEqual(labels, {0: 3, 1: 1, 2: 0, 3: 5})

    def test_parallel_coarse_cells_report_the_missing_conformal_modulus(self):
        self.assertEqual(self.metric["hyperideal"], [0])
        self.assertAlmostEqual(
            self.metric["cone_angles"][0], 8.490218538108175, places=12)
        out = round_realise_cells(self.cells, self.metric, maxit=200, seed=0)
        self.assertLess(out["residual"], 1e-12)
        self.assertEqual(out["redundant"], 1)
        self.assertEqual(out["excess"], 1)
        self.assertFalse(out["canonical"])
        with self.assertRaisesRegex(ValueError, "excess 1"):
            round_realise_cells(
                self.cells, self.metric, require_canonical=True, seed=0)


if __name__ == "__main__":
    unittest.main()
