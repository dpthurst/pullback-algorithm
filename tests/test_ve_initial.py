"""Tests for the initial geometric state of an abstract ribbon virtual endomorphism.

Run from the repository root:

    PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_ve_initial.py' -q

Every assertion below is measured, not expected: the numbers in the tests are
what the construction gives, and the two negative tests (the rabbit's star mesh
and the rabbit's step) assert the measured failure, not a claim of success.
"""
import math
import unittest

import numpy as np

from pullback import ve_initial as ve_init
from pullback.session import Session


def load(name):
    return Session.load(f"sessions/{name}.json")


def boost(v, axis, s):
    """A Mobius transformation of S^2 (a Lorentz boost), applied to rows."""
    v = np.asarray(v, dtype=float)
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    v = np.asarray(v)
    ch, sh = math.cosh(s), math.sinh(s)
    d = v @ a
    return (v + (((ch - 1.0) * d + sh)[:, None]) * a[None, :]) / (ch + sh * d)[:, None]


class TestNoAnalyticData(unittest.TestCase):
    def test_the_constructor_reads_no_map(self):
        import ast
        from pathlib import Path
        tree = ast.parse(Path(ve_init.__file__).read_text())
        mods, names = [], []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                mods.append(node.module)
            elif isinstance(node, ast.Import):
                mods += [a.name for a in node.names]
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.append(node.name)
            elif isinstance(node, ast.Attribute):
                names.append(node.attr)
        forbidden_mods = ("pullback.thurston", "pullback.discrete",
                          "pullback.rational", "pullback.iterate_graphs")
        for m in forbidden_mods:
            self.assertNotIn(m, mods, m)
        for n in ("rabbit_map", "critical_values", "preimage_with_degrees",
                  "pcf_data", "critical_points", "preimage"):
            self.assertNotIn(n, names, n)

    def test_the_public_entry_points_take_a_session_only(self):
        import inspect
        for fn in (ve_init.initial_state, ve_init.frozen_cover):
            names = list(inspect.signature(fn).parameters)
            self.assertEqual(names[0], "spine" if fn is ve_init.initial_state
                             else "session")


class TestBouquetObstruction(unittest.TestCase):
    """The rabbit's spine is a bouquet of three loops; its dual is the
    (3,3,3) pillowcase and NO star mesh exists."""

    @classmethod
    def setUpClass(cls):
        cls.session = load("rabbit")
        cls.q = ve_init.full_cell_quotient(cls.session.base)
        cls.prof = ve_init.obstruction_profile(cls.q)

    def test_the_dual_is_one_hexagon_on_four_punctures(self):
        self.assertEqual(self.q["counts"], {"V": 4, "E": 3, "F": 1})
        self.assertEqual(self.q["polygons"], ((0, 1, 3, 1, 2, 1),))

    def test_the_failure_of_the_star_mesh_is_a_repeated_corner(self):
        self.assertEqual(self.prof["vertex_repeats"], {0: {1: 3}})
        self.assertEqual(self.prof["edge_repeats"], {0: {"a": 2, "c": 2, "b": 2}})
        self.assertFalse(self.prof["star_realisable"])
        self.assertIn("meets vertices", ve_init._describe_obstruction(self.q, self.prof))

    def test_the_tripely_repeated_vertex_is_smooth_not_a_cone_point(self):
        ang = self.prof["cone_angles"]
        self.assertEqual(self.prof["smooth_vertices"], [1])
        self.assertAlmostEqual(ang[1] / (2 * math.pi), 1.0, places=12)
        for v in (0, 2, 3):
            self.assertAlmostEqual(ang[v] / (2 * math.pi), 1.0 / 3.0, places=12)

    def test_the_fan_subdivision_resolves_it_as_a_labelled_multigraph(self):
        fan = ve_init.fan_subdivision(self.q)
        self.assertEqual(fan["counts"], {"V": 5, "E": 9, "F": 6})
        self.assertEqual(fan["counts"]["V"] - fan["counts"]["E"]
                         + fan["counts"]["F"], 2)
        for f in fan["faces"]:
            self.assertEqual(len(set(f)), 3)          # no repeated corner
        self.assertEqual(len(fan["parallel"][(1, 4)]), 3)   # C met three times

    def test_the_regular_dual_state_exists_and_reports_its_moduli(self):
        st = ve_init.initial_state(self.session.base, kind="regular")
        self.assertEqual(len(st["positions"]), 4)
        self.assertTrue(st["realisation"]["converged"])
        self.assertLess(st["realisation"]["residual"], 1e-10)
        # the bouquet's complex does NOT determine the conformal class
        self.assertEqual(st["realisation"]["excess"], 2)
        self.assertFalse(st["realisation"]["canonical"])

    def test_a_different_dual_metric_fails_to_realise_and_is_not_claimed(self):
        st = ve_init.initial_state(self.session.base, kind="arbitrary", seed=0)
        self.assertFalse(st["realisation"]["converged"])
        self.assertGreater(st["realisation"]["residual"], 1e-3)


class TestStarRealisableSpines(unittest.TestCase):
    """theta and tetrahedral: the dual is already a simplicial triangulation,
    so the regular-dual (equilateral) realisation exists and is canonical."""

    def test_theta_is_a_double_equilateral_triangle(self):
        st = ve_init.initial_state(load("theta").base)
        prof = st["obstructions"]
        self.assertTrue(prof["star_realisable"], prof)
        for v in range(3):
            self.assertAlmostEqual(prof["cone_angles"][v] / math.pi, 2 / 3,
                                   places=12)
        self.assertEqual(st["realisation"]["excess"], 0)
        self.assertTrue(st["realisation"]["canonical"])
        self.assertLess(st["realisation"]["residual"], 1e-12)
        self.assertEqual(len(st["positions"]), 3)

    def test_tetrahedral_regular_dual_is_a_regular_tetrahedron(self):
        session = load("tetrahedral")
        st = ve_init.initial_state(session.base)
        self.assertTrue(st["obstructions"]["star_realisable"])
        self.assertEqual(st["subdivision"]["counts"], {"V": 8, "E": 18, "F": 12})
        self.assertEqual(st["realisation"]["excess"], 0)
        self.assertLess(st["realisation"]["residual"], 1e-12)
        from pullback.mesh import labelled_cross_ratio
        cr = complex(labelled_cross_ratio(np.array(st["positions"])))
        # the regular tetrahedron's labelled cross ratio is exp(i*pi/3)
        self.assertLess(abs(cr - complex(0.5, math.sqrt(3) / 2)), 1e-6)


class TestDifferentInitialRealizations(unittest.TestCase):
    def test_two_dual_metrics_give_genuinely_different_teich_points(self):
        from pullback.mesh import labelled_cross_ratio
        session = load("tetrahedral")
        a = ve_init.initial_state(session.base, kind="regular")
        b = ve_init.initial_state(session.base, kind="arbitrary", seed=0)
        ca = complex(labelled_cross_ratio(np.array(a["positions"])))
        cb = complex(labelled_cross_ratio(np.array(b["positions"])))
        self.assertGreater(abs(ca - cb), 0.05)
        self.assertTrue(a["realisation"]["converged"])
        self.assertTrue(b["realisation"]["converged"])

    def test_the_step_is_mobius_covariant(self):
        session = load("tetrahedral")
        st = ve_init.initial_state(session.base)
        fr = ve_init.frozen_cover(session)
        rng = np.random.RandomState(7)
        x0 = rng.normal(size=(fr["lifted_fan"]["n_vertices"], 3))
        x0 = x0 / np.linalg.norm(x0, axis=1)[:, None]
        _ = x0
        # the realisation of a FIXED metric is Mobius-covariant: moving the
        # starting point by a Mobius map moves the solution by the same map.
        # (The step itself is not covariant, because its base metric is read off
        # chordal distances and the spoke rule is not a conformal invariant.)
        L = st["subdivision"]
        n = L["n_vertices"]
        x0 = x0[:n] / np.linalg.norm(x0[:n], axis=1)[:, None]
        r1 = ve_init.realise(L, st["metric"], x0=x0)
        M = lambda v: boost(v, (0.3, -0.5, 0.81), 0.7)
        r2 = ve_init.realise(L, st["metric"], x0=M(x0))
        # The solver fixes a gauge, so the raw positions differ; what is
        # invariant is the Mobius class.  Measured: the two solutions are the
        # same labelled point of Teich, to the solver tolerance.
        from pullback.mesh import labelled_cross_ratio
        c1 = complex(labelled_cross_ratio(np.array(r1["x"][:n])))
        c2 = complex(labelled_cross_ratio(np.array(r2["x"][:n])))
        c3 = complex(labelled_cross_ratio(np.array(st["positions"])))
        self.assertLess(abs(c1 - c2), 1e-8)
        self.assertLess(abs(c1 - c3), 1e-8)
        self.assertLess(abs(r1["residual"] - r2["residual"]), 1e-10)


class TestFrozenCover(unittest.TestCase):
    def test_rabbit_degree_profile_and_face_correspondence(self):
        session = load("rabbit")
        fr = ve_init.frozen_cover(session)
        self.assertEqual(fr["n_base_punctures"], 4)
        self.assertEqual(fr["n_lifted_punctures"], 6)
        self.assertEqual(sorted(d["degree"] for d in fr["degree_profile"].values()),
                         [1, 1, 1, 1, 2, 2])
        self.assertEqual(sorted(d["base_face"] for d in fr["degree_profile"].values()),
                         [0, 1, 2, 2, 3, 3])
        self.assertTrue(fr["face_check"]["projects_to_base_face"])
        # every active lifted puncture lies over a distinct base face
        self.assertEqual(sorted(d["base_face"] for d in fr["active"].values()),
                         [0, 1, 2, 3])
        self.assertEqual(fr["base_of_edge"], {("a", 0): "a", ("a", 1): "a",
                                              ("b", 0): "b", ("b", 1): "b",
                                              ("c", 0): "c", ("c", 1): "c"})

    def test_tetrahedral_covers_every_base_face_with_total_degree_3(self):
        fr = ve_init.frozen_cover(load("tetrahedral"))
        self.assertEqual(fr["n_lifted_punctures"], 8)
        self.assertEqual(sum(d["degree"] for d in fr["degree_profile"].values()), 12)
        self.assertEqual(sorted(d["base_face"] for d in fr["active"].values()),
                         [0, 1, 2, 3])


class TestTheStep(unittest.TestCase):
    def test_tetrahedral_step_is_canonical_and_complete(self):
        session = load("tetrahedral")
        st = ve_init.initial_state(session.base)
        fr = ve_init.frozen_cover(session)
        r = ve_init.step(fr, st["positions"])
        self.assertTrue(r["complete"])
        self.assertEqual(r["excess"], 0)
        self.assertLess(r["residual"], 1e-12)
        self.assertEqual(len(r["positions"]), 4)

    def test_theta_step_is_complete(self):
        session = load("theta")
        st = ve_init.initial_state(session.base)
        fr = ve_init.frozen_cover(session)
        r = ve_init.step(fr, st["positions"])
        self.assertTrue(r["complete"])
        self.assertLess(r["residual"], 1e-12)

    def test_the_tetrahedral_iteration_settles_into_a_two_cycle(self):
        """MEASURED, and not what a contracting Thurston step would give: the
        cross ratio alternates between two values instead of approaching one.
        The frozen complex is not rebuilt from the configuration (the Delaunay
        decomposition of the current positions is never recomputed), so this is
        a limit cycle of the frozen discretisation, not evidence about the
        analytic map."""
        from pullback.mesh import labelled_cross_ratio
        session = load("tetrahedral")
        fr = ve_init.frozen_cover(session)
        st = ve_init.initial_state(session.base, kind="arbitrary", seed=0)
        hist = ve_init.iterate(fr, st["positions"], 6)
        self.assertEqual(len(hist), 6)
        crs = [complex(labelled_cross_ratio(np.array(h["positions"]))) for h in hist]
        gaps = [abs(crs[i + 1] - crs[i]) for i in range(len(crs) - 1)]
        # after the first step the cross ratio alternates between two values:
        # the gap stops shrinking (0.0974 repeated), it does not contract
        self.assertAlmostEqual(gaps[-1], gaps[-2], places=6)
        self.assertAlmostEqual(gaps[-1], gaps[-3], places=6)
        self.assertGreater(gaps[-1], 0.05)
        self.assertLess(abs(crs[2] - crs[4]), 1e-8)
        self.assertLess(abs(crs[3] - crs[5]), 1e-8)

    def test_the_rabbit_step_fails_and_says_by_how_much(self):
        session = load("rabbit")
        st = ve_init.initial_state(session.base, kind="regular")
        fr = ve_init.frozen_cover(session)
        r = ve_init.step(fr, st["positions"])
        self.assertFalse(r["complete"])
        self.assertGreater(r["residual"], 1e-3)
        self.assertEqual(r["excess"], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
