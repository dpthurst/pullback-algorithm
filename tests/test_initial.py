"""Tests for the initial geometric state built from a session alone.

Run from the repository root:

    PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_initial.py' -q

What is claimed here, and what is measured
------------------------------------------
Claimed (tests that must pass without any analytic map):

* `puncture_data` recovers the punctures, their labels, the diagonal (which
  face of the cover IS the puncture), the dynamics and the local degrees of a
  session from (pi, phi) alone, with uniqueness and degree certificates, on
  the rabbit plus five other sessions;
* `frozen_cover` is a covering of edge-LABELLED cell complexes: every lifted
  edge covers a base edge, d per base edge, the duals are spheres, and a dual
  face that meets a puncture twice keeps both corners (the rabbit's base dual
  is one hexagon over four punctures);
* a step from an arbitrary labelled configuration solves (residual ~1e-16)
  and is exactly Moebius covariant.

Measured, and NOT claimed: the step does not converge.  The certificate of
`rounding_multi.structure` explains why -- the spine dual has too few
independent length constraints -- and the iteration is a 3-cycle.  See
`test_the_measured_boundary_of_the_spine_dual` and report.md.
"""

import unittest
from pathlib import Path

import mpmath as mp
import numpy as np

from pullback import Session
from pullback import mesh as msh
from pullback import rounding_multi as mg
from pullback import initial as ist

mp.mp.dps = 30

SESSIONS = ("rabbit", "theta", "barbell", "tetrahedral", "census-3.2",
            "julia-4422")


def mobius(positions):
    M = (mp.mpc(0.7, 0.3), mp.mpc(0.2, -0.4), mp.mpc(-0.1, 0.5), mp.mpc(0.9, 0.2))
    return [msh._mp_apply(M, mp.mpc(p) if mp.isfinite(p) else mp.inf)
            for p in positions]


def load(name):
    return Session.load(f"sessions/{name}.json")


class TestTheConstructorUsesNoAnalyticMap(unittest.TestCase):
    def test_the_module_never_touches_a_rational_map_or_preimages(self):
        import ast
        tree = ast.parse(Path(ist.__file__).read_text())
        used = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Attribute):
                used.add(node.attr)
            elif isinstance(node, ast.ImportFrom):
                used.add(node.module or "")
                used.update(a.name for a in node.names)
        for forbidden in ("pcf_data", "lift_by_edges", "preimage_with_degrees",
                          "critical_values", "pullback.rational",
                          "pullback.thurston", "rabbit_map", "load",
                          "postcritical_set", "f_infinity"):
            self.assertNotIn(forbidden, used,
                             f"{forbidden!r} is used by the constructor")

    def test_its_only_inputs_are_a_session_and_a_seed(self):
        import inspect
        sig = inspect.signature(ist.InitialState.__init__)
        self.assertEqual(list(sig.parameters), ["self", "session", "seed",
                                                "positions"])


class TestPunctureData(unittest.TestCase):
    def test_the_rabbit_diagonal_and_dynamics(self):
        d = ist.puncture_data(load("rabbit"))
        # the 3-cycle of the postcritical set plus the fixed totally ramified
        # point: 3 -> 0 -> 2 -> 3 and 1 -> 1, critical exactly at 1 and 3
        self.assertEqual(d.dynamics, {3: 0, 0: 2, 2: 3, 1: 1})
        self.assertEqual(d.critical, {3: True, 0: False, 2: False, 1: True})
        self.assertEqual(d.degree_profile, {0: [2], 1: [2], 2: [1, 1], 3: [1, 1]})
        self.assertEqual(d.degree, 2)
        self.assertEqual(len(d.labels), 4)
        self.assertEqual(len(d.lifted_labels), 6)
        self.assertEqual(sorted(d.inactive), [3, 5])

    def test_the_rabbit_diagonal_agrees_with_the_one_analytic_fact_we_have(self):
        """The critical point of z^2 + c has local degree 2 and its image has a
        single preimage; that pattern is the fingerprint of the labelling, and
        the constructor reproduces it without ever seeing the map."""
        d = ist.puncture_data(load("rabbit"))
        crit = [j for j in d.labels if d.local_degrees[j] == 2]
        self.assertEqual(sorted(crit), [1, 3])
        for j in crit:                       # a critical value, one preimage
            self.assertEqual(d.degree_profile[d.dynamics[j]], [2])
        self.assertEqual(d.dynamics[1], 1)   # the fixed totally ramified point

    def test_certificates_on_six_sessions(self):
        for name in SESSIONS:
            d = ist.puncture_data(load(name))
            c = d.certificates
            with self.subTest(session=name):
                self.assertTrue(c["cover_degree_ok"], c["cover_degree_sums"])
                self.assertTrue(c["diagonal_distinct"])
                self.assertTrue(c["diagonal_sits_over_its_image"])
                self.assertTrue(c["phi_wraps_its_own_label"])
                self.assertEqual(c["euler_base_dual"], 2)
                self.assertEqual(c["euler_lifted_dual"], 2)
                # Riemann-Hurwitz for the sphere: sum(local degree - 1) = 2d-2
                self.assertEqual(c["riemann_hurwitz"], 2 * d.degree - 2)

    def test_an_unknown_diagonal_is_refused_not_guessed(self):
        """A session whose phi kills two faces of a four-fold cover cannot pin
        the diagonal; the constructor must say so."""
        base = load("rabbit").base
        cover = load("rabbit").cover
        with self.assertRaises(ValueError):
            # phi with every chain collapsed: no face wraps anything
            from pullback.maps import make_graph_map
            trivial = make_graph_map(cover, base, {}, {v: "x" for v in cover.vertices()})
            ist.PunctureData(base, cover, trivial)


class TestTheFrozenCover(unittest.TestCase):
    def test_the_rabbit_lifted_cover_is_an_edge_labelled_covering(self):
        s = load("rabbit")
        d = ist.puncture_data(s)
        fc = ist.frozen_cover(d)
        c = fc["certificates"]
        self.assertEqual((c["base_counts"], c["lift_counts"]),
                         ({"V": 4, "E": 3, "F": 1}, {"V": 6, "E": 6, "F": 2}))
        self.assertEqual(c["base_euler"], 2)
        self.assertEqual(c["lift_euler"], 2)
        self.assertTrue(c["edges_over_base_ok"])
        self.assertTrue(c["corners_ok"])
        # three base edges, two lifted edges each (degree 2)
        self.assertEqual(set(c["edges_over_base"].values()), {2})
        # the label vertices are four distinct lifted punctures
        self.assertEqual(len(set(fc["label_vertices"].values())), 4)

    def test_a_dual_face_that_meets_a_puncture_twice_keeps_both_corners(self):
        """The failure the star mesh had: the rabbit's base dual is one hexagon
        whose six corners sit on only four punctures.  As a corner-slot walk it
        is exact, and the *side* labels still name every copy of a doubled
        edge."""
        fc = ist.frozen_cover(ist.puncture_data(load("rabbit")))
        base = fc["base"]
        self.assertEqual(len(base["polygons"]), 1)
        poly = base["polygons"][0]
        self.assertEqual(len(poly), 6)
        self.assertLess(len(set(poly)), 6)        # repeated corners: expected
        self.assertIn(0, [k for k in base["non_simple_faces"]])
        # ... and the sides still distinguish the two copies of each loop
        sides = [d[0] for d in base["corners"][0]]
        self.assertEqual(len(set(sides)), 3)
        self.assertEqual(len(sides), 6)

    def test_every_session_gives_a_covering_with_matching_degrees(self):
        for name in SESSIONS:
            d = ist.puncture_data(load(name))
            fc = ist.frozen_cover(d)
            with self.subTest(session=name):
                self.assertTrue(fc["certificates"]["edges_over_base_ok"])
                self.assertEqual(fc["certificates"]["lift_euler"], 2)
                self.assertEqual(fc["certificates"]["base_euler"], 2)


class TestTheStep(unittest.TestCase):
    def test_a_step_from_an_arbitrary_configuration_solves(self):
        st = ist.InitialState(load("rabbit"), seed=0)
        r = st.step()
        self.assertLess(r["residual"], 1e-12)
        self.assertEqual(len(r["points"]), 4)
        self.assertEqual(len(set(r["labels"])), 4)
        # the labelled punctures land at distinct points
        v = np.array(r["vectors"])
        d = mg.chordal_lengths(v) + np.eye(4)
        self.assertGreater(d.min(), 1e-3)

    def test_the_step_is_moebius_covariant(self):
        for name in ("rabbit", "census-3.2"):
            base = load(name)
            a = ist.EnrichedState(base, seed=1)
            b = ist.EnrichedState(base, seed=1)
            ra = a.step(a.positions)
            rb = b.step(mobius(a.positions))
            with self.subTest(session=name):
                self.assertAlmostEqual(
                    abs(complex(ra["coordinate"]) - complex(rb["coordinate"])),
                    0.0, places=9)

    def test_the_same_ve_from_two_different_realisations(self):
        """Two arbitrary initial realisations must be the same *labelled*
        object: the chart of a step depends on the configuration, not on the
        representative, and a Moebius-moved start gives the Moebius-moved
        answer."""
        s = load("rabbit")
        a = ist.EnrichedState(s, seed=3)
        r0 = a.step(a.positions)
        r1 = a.step(mobius(a.positions))
        self.assertAlmostEqual(abs(complex(r0["coordinate"])
                                   - complex(r1["coordinate"])), 0.0, places=9)

    def test_the_theta_bouquet_is_not_asked_to_be_equilateral(self):
        """|P| = 3 sessions have no Teichmuller modulus; the constructor still
        returns a concrete labelled configuration and a frozen cover, and it
        makes no claim that the spine dual is built of equilateral triangles."""
        d = ist.puncture_data(load("theta"))
        st = ist.InitialState(load("theta"), seed=0)
        r = st.step()
        self.assertLess(r["residual"], 1e-12)
        lens = np.array(sorted(r["base_lengths"].values()))
        self.assertGreater(lens.max() / lens.min(), 1.0 + 1e-6)

    def test_a_dual_self_loop_is_refused_before_the_solver(self):
        # A bridge of the spine is a self-loop of its dual.  Its length is not
        # a chord between two labelled points, so returning a silent zero here
        # used to defer the real problem to round_realise's positivity check.
        for constructor in (ist.InitialState, ist.EnrichedState):
            with self.subTest(constructor=constructor.__name__):
                st = constructor(load("barbell"), seed=0)
                with self.assertRaisesRegex(ValueError, "self-loop"):
                    st.step()


class TestCombinatorialUpdates(unittest.TestCase):
    def test_identity_update_preserves_the_state(self):
        from pullback.homotopy import basis_transport
        s = load("rabbit")
        st = ist.InitialState(s, seed=0)
        up = st.update_combinatorics(s.base, basis_transport(s.base, s.base))
        self.assertIsNot(up, st)
        self.assertIs(st.data.base, s.base)       # the parent was not mutated
        self.assertEqual(up.data_labels, [0, 1, 2, 3])
        self.assertEqual(up.data.dynamics, st.data.dynamics)
        self.assertEqual(up.positions, st.positions)
        self.assertLess(up.step()["residual"], 1e-12)

    def test_nontrivial_update_preserves_inherited_external_labels(self):
        from pullback import thurston as th
        from pullback import discrete
        from pullback.maps import make_graph_map
        from pullback.ribbons import B, F

        s = load("rabbit")
        data = discrete.pcf_data(th.rabbit_map())
        spine = msh.voronoi_spine(data["points"])
        chains = {
            (0, 1): (), (1, 2): (), (0, 2): (),
            (0, 3): (("b", F),), (1, 3): (("a", B),),
            (2, 3): (("c", F),),
        }
        comparison = make_graph_map(
            spine, s.base, chains,
            {v: s.base.vertices()[0] for v in spine.vertices()})
        st = ist.InitialState(s, positions=data["points"])
        up = st.update_combinatorics(spine, comparison)

        self.assertNotEqual(up.data_labels, list(range(4)))
        self.assertEqual(sorted(up.data_labels), list(range(4)))
        self.assertEqual(up.positions, st.positions)
        self.assertEqual(st.positions, data["points"])
        for new_label in up.data.labels:
            old_label = up.data_labels[new_label]
            image = up.data.dynamics[new_label]
            self.assertEqual(up.data_labels[image], st.data.dynamics[old_label])
            self.assertEqual(up.data.local_degrees[new_label],
                             st.data.local_degrees[old_label])
        self.assertEqual(up.comparison["external_to_data"],
                         {label: j for j, label in enumerate(up.data_labels)})

        # The returned configuration is still in the original external order,
        # not the permuted face order of the new spine.  Only the explicit call
        # above changed combinatorics; stepping does not rebuild them.
        out = up.step()
        self.assertLess(out["residual"], 1e-12)
        self.assertEqual(sorted(out["label_vertices"]), list(range(4)))
        self.assertEqual(len(out["points"]), 4)
        self.assertAlmostEqual(abs(out["coordinate"] - st.chart(out["points"])),
                               0.0, places=12)


class TestTheMeasuredBoundaryOfTheSpineDual(unittest.TestCase):
    """The honest measurement: the spine dual under-determines the conformal
    class, so the iteration is a cycle, not a contraction."""

    def test_the_spine_dual_has_far_too_few_independent_lengths(self):
        st = ist.InitialState(load("rabbit"), seed=0)
        edges = st.cover["edges"]
        x = st.step()["x"]
        info = mg.structure(edges, x)
        self.assertEqual(info["V"], 6)
        self.assertEqual(info["E"], 6)
        # 2V-6 = 6 real moduli are needed; the complex supplies
        # E - rank(A) - redundant = 0, so the deficit is the whole 6
        self.assertEqual(info["rank_A"], 5)                 # a bipartite 1-skeleton
        self.assertEqual(st.cover["certificates"]["free_dimension"], 0)
        self.assertEqual(st.cover["certificates"]["deficit"], 6)

    def test_chords_close_the_rank_but_not_to_zero(self):
        st = ist.EnrichedState(load("rabbit"), seed=0)
        self.assertEqual(len(st.pack["added_chords"]), 3)
        self.assertEqual(st.pack["free_dimension"], 5)
        self.assertEqual(st.pack["target_free_dimension"], 6)
        self.assertFalse(st.pack["canonical"])

    def test_the_iteration_is_a_three_cycle_not_a_contraction(self):
        st = ist.EnrichedState(load("rabbit"), seed=0)
        ws = [st.chart()]
        cur = st.positions
        for _ in range(6):
            cur = st.step(cur)["points"]
            ws.append(st.chart(cur))
        gaps = [abs(b - a) for a, b in zip(ws, ws[1:])]
        self.assertGreater(gaps[-1], 0.5 * gaps[0])          # no contraction
        self.assertAlmostEqual(abs(ws[3] - ws[0]), 0.0, places=9)   # period 3
        self.assertAlmostEqual(abs(ws[4] - ws[1]), 0.0, places=9)

    def test_no_analytic_limit_is_claimed_anywhere_in_the_constructor(self):
        src = Path(ist.__file__).read_text()
        for word in ("archive", "the limit is", "converges to the",
                     "attracting"):
            self.assertNotIn(word, src.lower())


if __name__ == "__main__":
    unittest.main()
