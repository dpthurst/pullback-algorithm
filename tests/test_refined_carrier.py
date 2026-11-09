"""Certified refined carrier and comparison-preserving marked iteration."""
import os
import unittest

from pullback.geometric_retraction import GeometricRetractionError
from pullback.marked_iteration import continue_refined, iterate_marked, marked_step
from pullback.session import Session

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestRefinedCarrier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = Session.load(os.path.join(REPO, "sessions", "rabbit.json"))
        cls.initial = cls.session.refined_geometry(seed=0)
        cls.orbit = iterate_marked(cls.initial, steps=3)
        cls.first, cls.second, cls.third = cls.orbit["steps"]
        cls.carrier = cls.first["carrier"]

    def test_carrier_has_the_expected_graphs_and_portrait(self):
        c = self.carrier
        source, spine = c["lifted"], c["spine"]
        self.assertIs(spine, self.initial.init.spine)
        self.assertIs(c["pi_prime"].source, source)
        self.assertIs(c["phi_prime"].source, source)
        self.assertIs(c["pi_prime"].target, spine)
        self.assertIs(c["phi_prime"].target, spine)
        self.assertEqual((len(source.vertices()), len(source.edge_list()),
                          len(source.faces())), (12, 18, 8))
        self.assertEqual((len(c["refined_base"].vertices()),
                          len(c["refined_base"].edge_list()),
                          len(c["refined_base"].faces())), (6, 9, 5))
        cert = c["certificates"]
        self.assertEqual(cert["commuting_edges"], 18)
        self.assertTrue(cert["commuting_vertex_images"])
        self.assertTrue(cert["pi1_surjective"])
        self.assertTrue(cert["cover2"]["ok"])
        self.assertTrue(cert["ribbon2_ok"])
        self.assertEqual(cert["peripheral_images"],
                         [None, 2, None, None, None, 1, 0, 3])
        self.assertEqual(cert["active_source_faces"], {1: 2, 5: 0, 6: 1, 7: 3})
        self.assertEqual(cert["phi_degrees"], [None, 1, None, None, None, 1, 1, 1])
        self.assertEqual(cert["auxiliary_source_faces"], [0, 3])
        self.assertEqual(cert["inactive_source_faces"], [0, 2, 3, 4])
        self.assertEqual(len(cert["auxiliary_base_faces"]), 1)

    def test_cover_and_inclusion_commute_on_every_edge(self):
        c = self.carrier
        pi, collapsed = c["pi_prime"], c["collapsed_cover"]
        self.assertEqual(pi.vertex_images, collapsed.vertex_images)
        for edge in pi.source.edge_list():
            self.assertEqual(pi.chains[edge], collapsed.chains[edge], edge)

    def test_one_marked_step_preserves_both_comparison_directions(self):
        step = self.first
        self.assertEqual(step["identities"], {"to_then_from": True,
                                             "from_then_to": True})
        self.assertTrue(all(step["identities"].values()))
        basepoints = step["upsilon"]["basepoints"]
        self.assertEqual(basepoints["current"], 0)
        self.assertEqual(basepoints["lift"], 0)
        self.assertEqual(basepoints["output"], 3)
        self.assertEqual(step["comparison"]["new_spine"].vertices()[0], 3)
        output = step["comparison"]
        self.assertEqual((len(output["P"].vertices()), len(output["P"].edge_list()),
                          len(output["P"].faces())), (8, 12, 6))

    def test_three_numerical_steps_are_embedded_and_repeat_the_portrait(self):
        for step in (self.first, self.second, self.third):
            cert = step["numerical"]
            self.assertTrue(cert["embedded"], cert["reason"])
            self.assertTrue(cert["injective"])
            self.assertEqual(cert["excess"], 0)
            self.assertLess(cert["residual"], 1e-12)
            self.assertAlmostEqual(abs(cert["degree"]), 1.0, places=9)
            self.assertEqual(step["carrier"]["certificates"]["peripheral_images"],
                             [None, 2, None, None, None, 1, 0, 3])
        z1 = self.first["numerical"]["chart"][1][0]
        z2 = self.second["numerical"]["chart"][1][0]
        self.assertNotAlmostEqual(z1, z2, places=8)

    def test_reconstruction_preserves_the_exact_marked_carrier(self):
        for earlier, later in ((self.first, self.second), (self.second, self.third)):
            previous = earlier["comparison"]
            nxt = later["refined"]
            self.assertIsNone(nxt.init.transports)
            self.assertEqual(nxt.init.spine.serialization(),
                             previous["new_spine"].serialization())
            self.assertEqual(nxt.init.P.serialization(), previous["P"].serialization())
            self.assertEqual(nxt.init.h.chains, previous["to_original"].chains)
            self.assertEqual(nxt.init.h.vertex_images,
                             previous["to_original"].vertex_images)
            self.assertEqual(nxt.init.phi_prime.chains,
                             previous["phi_prime"].chains)
            self.assertEqual(nxt.init.phi_prime.vertex_images,
                             previous["phi_prime"].vertex_images)
            # Repeating the public continuation is deterministic as well.
            self.assertEqual(continue_refined(earlier).init.phi_prime.chains,
                             previous["phi_prime"].chains)

    def test_numerical_continuation_uses_the_previous_realisation(self):
        self.assertFalse(self.first["x0_was_supplied"])
        self.assertTrue(self.second["x0_was_supplied"])
        self.assertTrue(self.third["x0_was_supplied"])
        for step in (self.first, self.second, self.third):
            self.assertEqual(step["next_seed"], "previous_realisation")
        self.assertIsNone(self.orbit["terminal"].init.transports)

    def test_fourth_step_stops_at_the_geometric_rechart_boundary(self):
        fourth = continue_refined(self.third)
        with self.assertRaisesRegex(GeometricRetractionError, "sphere centre"):
            marked_step(fourth, x0=self.third["numerical"]["x"])

    def test_nonembedded_realisation_is_refused_before_graph_work(self):
        with self.assertRaisesRegex(ValueError, "embedded"):
            marked_step(self.initial, cert={
                "embedded": False, "injective": False,
                "degree": 0.0, "excess": 1, "reason": "folded control"})


if __name__ == "__main__":
    unittest.main()
