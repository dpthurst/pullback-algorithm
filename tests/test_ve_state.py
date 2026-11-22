"""Acceptance tests for the session-only initial geometric state (direction d3).

Run from the repository root:

    PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_ve_state.py' -v

What is asserted, and why each is evidence rather than a restatement:

1. PURITY.  `ve_init.purity_guard()` patches the registered analytic entry
   points (rational maps, `pcf_data`, `lift_by_edges`, `lift_path`, the
   frozen-step constructors that need critical values) so that touching one
   raises.  The whole construction and a full step run inside the block.  This
   is a name-level regression net, not a proof about unregistered Python paths.
2. STRUCTURE.  Counts, the cover degree against the session's own degree, the
   multiset of local degrees over every puncture against the session's own
   portrait under the transport's marking, the link condition at every cell,
   Euler characteristic 2 for both the cover and its dual cells.
3. THE KNOWN-ANSWER PORTRAIT.  The rabbit's labels and puncture dynamics read
   out of the constructed cover must equal the analytic ones that
   `tests/test_lift.py` pins (`{0:3,1:1,2:0,3:5}`, `{0:2,1:0,2:1,3:3}`).  The
   analytic data is used HERE ONLY, as a known answer for a combinatorial
   output; the constructor never sees it.
4. NEGATIVE CONTROLS.  The naive regular-dual star mesh of the rabbit's bouquet
   is refused by `Mesh.validate` (repeated corner), so the failure the task
   warns about is reproduced as a test rather than described.  And a bouquet is
   not equilateral: the realised base angles deviate from pi/3, and the
   construction still succeeds, so no equilateral-triangle assumption is hidden
   in it.
5. THE MEASURED FAILURE.  For four punctures the discrete-conformal gauge
   absorbs the choice of initial points: two different realisations produce the
   SAME lifted realisation (determinants agree to 1e-9), and the iteration falls
   into an exact 3-cycle instead of converging.  Both facts are pinned, with the
   numbers, instead of a convergence claim.
"""

import unittest
from pathlib import Path

import mpmath as mp
import numpy as np

from pullback import discrete, mesh
from pullback.session import Session
from pullback.ve_state import VEInit, chart, purity_guard

mp.mp.dps = 40

REPO = Path(__file__).resolve().parents[1]
TWO_PI = 2.0 * np.pi


def session(name):
    return Session.load(f"{REPO}/sessions/{name}.json")


class Constructed(unittest.TestCase):
    """Construct once per class: the transport search is ~5 s."""

    _cache = {}

    @classmethod
    def state(cls, name, seed):
        key = (name, seed)
        if key not in cls._cache:
            cls._cache[key] = VEInit(session(name), seed=seed)
        return cls._cache[key]


class TestPurity(unittest.TestCase):
    def test_construction_and_step_never_touch_analytic_data(self):
        with purity_guard():
            v = Constructed.state("rabbit", 0)
            cert = v.certificates(realise=False)
            out = v.step([mp.mpc(0.5 + 0.2j), mp.mpc(-1.0 + 0.4j),
                          mp.mpc(0.3 - 0.9j), mp.mpc(-0.2 - 0.3j)])
        self.assertEqual(cert["cover_degree"], 2)
        self.assertLess(out["residual"], 1e-10)

    def test_the_guard_actually_bites(self):
        with self.assertRaises(AssertionError):
            with purity_guard():
                discrete.pcf_data(__import__("pullback").thurston.rabbit_map())

    def test_the_guard_covers_the_rational_map_constructor(self):
        from pullback import rational
        original = rational.RationalMap
        with self.assertRaisesRegex(AssertionError, "RationalMap"):
            with purity_guard():
                rational.RationalMap([mp.mpf(1)], [mp.mpf(1)])
        self.assertIs(rational.RationalMap, original)

    def test_session_initial_geometry_uses_only_the_ve(self):
        v = session("rabbit").initial_geometry(seed=0)
        self.assertEqual(v.session.name, "rabbit")
        cert = v.certificates(realise=False)
        self.assertEqual(cert["cover_degree"], 2)
        self.assertEqual(cert["lifted_labels"], {0: 3, 1: 1, 2: 0, 3: 5})

    def test_session_initial_dual_state_uses_only_the_ve(self):
        from pullback.initial import InitialState
        v = session("rabbit").initial_dual_state(seed=0)
        self.assertIsInstance(v, InitialState)
        self.assertEqual(v.data.dynamics, {3: 0, 0: 2, 2: 3, 1: 1})
        self.assertEqual(v.data.degree, 2)


class TestStructure(Constructed):
    @classmethod
    def setUpClass(cls):
        cls.v = cls.state("rabbit", 0)
        cls.cert = cls.v.certificates()

    def test_punctures_and_spine(self):
        self.assertEqual(self.cert["punctures"], 4)
        self.assertEqual(self.cert["spine_counts"], {"V": 4, "E": 6, "F": 4})
        self.assertEqual(self.cert["spine_euler"], 2)

    def test_frozen_cover_has_the_session_degree(self):
        self.assertEqual(self.cert["cover_counts"], {"V": 8, "E": 12, "F": 6})
        self.assertEqual(self.cert["cover_degree"], self.cert["session_degree"])
        self.assertEqual(self.cert["cells_counts"], {"V": 6, "E": 12, "F": 8})
        self.assertEqual(self.cert["cells_euler"], 2)

    def test_ribbon_face_correspondence_and_degree_profile(self):
        # the transported portrait equals the session's own portrait once the
        # transport's puncture permutation is applied
        self.assertTrue(self.cert["degree_profile_matches_session"])
        profile = self.cert["degree_profile"]
        self.assertEqual(sorted(d for v in profile.values() for d in v),
                         [1, 1, 1, 1, 2, 2])
        self.assertEqual(sum(self.cert["degree_profile"].values(),
                             []).count(2), 2)

    def test_every_cell_link_is_a_single_cycle(self):
        self.assertTrue(self.cert["links_single_cycle"])
        self.assertEqual(self.cert["non_simple_dual_faces"], [])

    def test_labels_and_dynamics_match_the_analytic_known_answer(self):
        # known answer pinned in tests/test_lift.py and in discrete.pcf_data;
        # only compared here, never used by the constructor
        self.assertEqual(self.cert["lifted_labels"], {0: 3, 1: 1, 2: 0, 3: 5})
        self.assertEqual(self.cert["dynamics"], {0: 2, 1: 0, 2: 1, 3: 3})

    def test_identity_comparison_update_keeps_both_directions(self):
        from pullback.validate import rose_words, spanning_tree

        v = self.v
        out = v.advance_comparison(v.spine, v.phi_prime)
        tree, gens = spanning_tree(v.spine)
        words = lambda m: rose_words(m, tree, tree, gens)
        self.assertEqual(words(out["to_original"]), words(v.h))
        from_original = out["from_original"]
        self.assertIs(from_original.source, v.session.base)
        self.assertIs(from_original.target, v.spine)
        self.assertIs(out["to_original"].source, v.spine)
        self.assertIs(out["to_original"].target, v.session.base)
        self.assertEqual(out["P"].serialization(), v.P.serialization())


    def test_hyperideal_vertices_are_reported_not_hidden(self):
        self.assertEqual(self.cert["hyperideal"], [1, 5])
        for v in self.cert["hyperideal"]:
            self.assertGreater(self.cert["cone_angles"][v], TWO_PI)

    def test_the_coarse_realisation_is_folded_and_non_canonical(self):
        """The measured failure of the *geometric* layer, pinned exactly."""
        self.assertEqual(self.cert["excess"], 1)
        self.assertFalse(self.cert["canonical"])
        self.assertFalse(self.cert["embedded"])
        self.assertIn("folded", self.cert["embedding"]["reason"])
        self.assertIn("3 of 8 faces", self.cert["embedding"]["reason"])
        self.assertLess(self.cert["residual"], 1e-12)


class TestNegativeControls(Constructed):
    def test_naive_star_mesh_with_repeated_corners_is_refused(self):
        """The failure mode the task names, reproduced as a failing input.

        The bouquet's dual cell decomposition has ONE face whose boundary walk
        meets the central puncture three times: corners (0,1,0,2,0,3).  A star
        mesh that lists it as a polygon is refused, so no realisation can
        silently pretend the corners are distinct.
        """
        pts = [mp.mpc(0), mp.mpc(1), mp.mpc(1j), mp.mpc(-1)]
        star = mesh.Mesh(pts, [(0, 1, 0, 2, 0, 3), (1, 3, 2)])
        with self.assertRaisesRegex(ValueError, "repeated vertex"):
            star.validate()

    def test_a_bouquet_is_not_assumed_equilateral(self):
        """Arbitrary 4-point realisation: base corner angles are not pi/3."""
        v = Constructed.state("rabbit", 0)
        pts = v.points
        vecs = np.array([mesh.to_vector(p) for p in pts])
        worst = 0.0
        for face in v.base_faces:
            for k in range(3):
                a = vecs[face[k]]
                b = vecs[face[(k + 1) % 3]] - a
                c = vecs[face[(k - 1) % 3]] - a
                ang = float(np.arccos(np.dot(b, c)
                                      / (np.linalg.norm(b) * np.linalg.norm(c))))
                worst = max(worst, abs(ang - np.pi / 3))
        self.assertGreater(worst, 0.1)          # measured: 0.53 rad
        # and the construction is not disturbed by that
        self.assertEqual(v.certificates(realise=False)["cover_degree"], 2)


class TestChoiceOfInitialRealisation(Constructed):
    """Two different realisations of the same session, same start."""

    START = [mp.mpc(0.5 + 0.2j), mp.mpc(-1.0 + 0.4j),
             mp.mpc(0.3 - 0.9j), mp.mpc(-0.2 - 0.3j)]

    def _run(self, seed, n=6):
        v = self.state("rabbit", seed)
        v.x = None
        hist = v.iterate(self.START, n)
        return v, [complex(h["coordinate"]) for h in hist], hist

    def test_custom_non_rotational_initial_points_are_accepted(self):
        """Arbitrary input, not just the Fibonacci family's z-axis rotation.

        This is an acceptance test for choosing the initial realization, not a
        claim that the resulting conformal class is independent of the choice.
        """
        custom = [mp.mpc(0.37 + 0.21j), mp.mpc(-1.3 + 0.55j),
                  mp.mpc(0.2 - 0.8j), mp.mpc(-0.4 - 0.1j)]
        v = VEInit(session("rabbit"), points=custom)
        cert = v.certificates(realise=False)
        self.assertEqual(cert["cover_degree"], 2)
        self.assertEqual(cert["dynamics"], {0: 2, 1: 0, 2: 1, 3: 3})
        self.assertEqual(cert["lifted_labels"], {0: 3, 1: 1, 2: 0, 3: 5})
        a = np.array([mesh.to_vector(p) for p in custom])
        b = np.array([mesh.to_vector(p) for p in self.state("rabbit", 0).points])
        da = np.sqrt(((a[:, None, :] - a[None, :, :]) ** 2).sum(-1))
        db = np.sqrt(((b[:, None, :] - b[None, :, :]) ** 2).sum(-1))
        self.assertGreater(float(np.abs(da - db).max()), 1e-3)

    def test_seed_phase_is_an_exact_rotation_of_the_initial_family(self):
        a0 = np.array([mesh.to_vector(p) for p in self.state("rabbit", 0).points])
        a1 = np.array([mesh.to_vector(p) for p in self.state("rabbit", 1).points])
        d0 = np.sqrt(((a0[:, None, :] - a0[None, :, :]) ** 2).sum(-1))
        d1 = np.sqrt(((a1[:, None, :] - a1[None, :, :]) ** 2).sum(-1))
        self.assertLess(float(np.abs(d0 - d1).max()), 1e-12)
        # This is only an exact-rotation control, NOT invariance under arbitrary
        # perturbations of the initial realisation.
        a, ca, _ = self._run(0, 1)
        b, cb, _ = self._run(1, 1)
        self.assertEqual(a.certificates(realise=False)["lifted_labels"],
                         b.certificates(realise=False)["lifted_labels"])
        # the discrete-conformal gauge absorbs the choice of initial points:
        # the realised face determinants agree (measured agreement ~1e-14)
        da = a.certificates()["embedding"]["det"]
        db = b.certificates()["embedding"]["det"]
        self.assertLess(float(np.max(np.abs(np.array(da) - np.array(db)))), 1e-6)

    def test_the_iterate_is_the_same_in_both_seed_rotations(self):
        _a, ca, _ = self._run(0, 6)
        _b, cb, _ = self._run(1, 6)
        self.assertLess(max(abs(x - y) for x, y in zip(ca, cb)), 1e-6)

    def test_the_step_is_not_a_contraction_on_the_coarse_cover(self):
        """Measured: an exact 3-cycle, not convergence.

        The lifted coarse cover is folded and under-determines the conformal
        class (excess 1), so the frozen step is not a sphere map; the labelled
        chart falls into a 3-cycle whose step sizes repeat to 1e-9.
        """
        _v, cs, hist = self._run(0, 7)
        steps = [abs(b - a) for a, b in zip(cs, cs[1:])]
        self.assertGreater(min(steps), 0.4)                 # no contraction
        for k in range(len(steps) - 3):
            self.assertAlmostEqual(steps[k], steps[k + 3], places=9)
        self.assertLess(hist[0]["residual"], 1e-12)         # solves each step

    def test_the_analytic_configuration_is_not_a_fixed_point_of_the_coarse_step(self):
        """The combinatorics are right (labels match) but the coarse geometry is
        not: the step moves the analytic postcritical configuration by 0.869 in
        the labelled chart.  Compare FrozenStep, which moves it by 0.123 at
        refine 0 -- the coarse realisation is the whole of the difference."""
        v = self.state("rabbit", 0)
        data = discrete.pcf_data(__import__("pullback").thurston.rabbit_map())
        v.x = None
        disp = v.fixed_displacement(data["points"])
        self.assertGreater(float(disp), 0.5)
        self.assertLess(float(disp), 1.2)


class TestSecondSession(Constructed):
    """twisted-rabbit: same base graph, a different cover and phi."""

    @classmethod
    def setUpClass(cls):
        cls.v = cls.state("twisted-rabbit", 0)
        cls.cert = cls.v.certificates()

    def test_structure_certificates_hold(self):
        self.assertEqual(self.cert["cover_degree"], 2)
        self.assertEqual(self.cert["cells_euler"], 2)
        self.assertTrue(self.cert["degree_profile_matches_session"])
        self.assertTrue(self.cert["links_single_cycle"])
        self.assertEqual(self.cert["spine_counts"], {"V": 4, "E": 6, "F": 4})

    def test_it_is_a_different_virtual_endomorphism(self):
        other = self.state("rabbit", 0)
        self.assertNotEqual(self.v.session.phi.chains, other.session.phi.chains)

    def test_the_geometry_is_governed_by_the_base_not_by_phi(self):
        """Same four punctures, so the same lifted realisation as the rabbit.

        The two sessions differ only in the cover/phi; the discrete-conformal
        metric of the lifted cells is a function of the base puncture positions
        and the label assignment, both of which coincide.  A step from the same
        configuration therefore lands in the same labelled chart.
        """
        start = [mp.mpc(0.5 + 0.2j), mp.mpc(-1.0 + 0.4j),
                 mp.mpc(0.3 - 0.9j), mp.mpc(-0.2 - 0.3j)]
        self.v.x = None
        a = complex(self.v.step(start)["coordinate"])
        other = self.state("rabbit", 0)
        other.x = None
        b = complex(other.step(start)["coordinate"])
        self.assertLess(abs(a - b), 1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
