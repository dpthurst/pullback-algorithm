"""Computed dynamical gates: no labels or convergence-by-class assertion."""

import copy
from fractions import Fraction
from math import inf
import unittest

from pullback.convergence import (ConvergenceError, convergence_guard,
                                  require_convergence_allowed)
from pullback.initial import InitialState
from pullback.orbifold import (guard_verdict, lattes_report, orbifold,
                               portrait_data)
from pullback.session import Session


def load(name):
    return Session.load(f"sessions/{name}.json")


def four_point_lattes_portrait():
    # Degree four, sixteen simple critical points, none postcritical.
    pre = {3: [(1, 0), (1, 1), (1, 2), (1, 3)],
           0: [(2, None), (2, None)],
           1: [(2, None), (2, None)],
           2: [(2, None), (2, None)]}
    return portrait_data(4, {x: 3 for x in range(4)},
                         {x: 1 for x in range(4)}, pre, check=True)


class TestConvergenceGuard(unittest.TestCase):
    def test_lattes_sounding_names_do_not_override_the_computation(self):
        for name in ("slit-lattes", "blown-up-lattes"):
            session = load(name)
            self.assertNotIn("dynamical_class", session.raw)
            info = convergence_guard(session)
            self.assertTrue(info["allowed"], name)
            self.assertTrue(info["candidate"], name)
            self.assertEqual(info["status"], "candidate")
            self.assertEqual(info["orbifold_status"], "hyperbolic")
            self.assertEqual(info["signature"], (3, 6, inf))
            self.assertEqual(info["chi"], Fraction(-1, 2))
            self.assertFalse(info["lattes_portrait"])
            self.assertFalse(info["flexible"])
            self.assertTrue(require_convergence_allowed(session)["allowed"])

    def test_injected_lattes_metadata_does_not_override_the_ve(self):
        data = copy.deepcopy(load("rabbit").raw)
        data["name"] = "not-a-contracting-map"
        data["dynamical_class"] = "flexible_lattes"
        info = convergence_guard(Session.from_dict(data))
        self.assertEqual(info["status"], "candidate")
        self.assertEqual(info["orbifold_status"], "hyperbolic")
        self.assertTrue(info["allowed"])
        self.assertFalse(info["lattes_portrait"])

    def test_unbranched_cycles_remain_candidates_with_a_warning(self):
        for name in ("julia-4422", "twisted-z2i"):
            info = convergence_guard(load(name))
            self.assertEqual(info["status"], "candidate", name)
            self.assertEqual(info["type"], "cycle without branch point")
            self.assertEqual(info["orbifold_status"], "hyperbolic", name)
            self.assertEqual(info["chi"], Fraction(-1, 2))
            self.assertTrue(info["orbifold_ok"])
            self.assertFalse(info["branched_cycles_ok"])
            self.assertTrue(info["allowed"])
            self.assertTrue(info["candidate"])
            self.assertTrue(info["unbranched_cycles"])
            self.assertIn("Thurston pullback still makes sense", info["reason"])
            self.assertIn("obstruction", info["reason"])
            self.assertIn("conditional", info["reason"])
            self.assertTrue(require_convergence_allowed(load(name))["allowed"])

    def test_hyperbolic_type_is_a_candidate_not_a_certificate(self):
        for name in ("rabbit", "tetrahedral"):
            info = convergence_guard(load(name))
            self.assertTrue(info["allowed"], name)
            self.assertTrue(info["candidate"], name)
            self.assertEqual(info["status"], "candidate")
            self.assertEqual(info["orbifold_status"], "hyperbolic")
            self.assertEqual(info["signature"], (inf, inf, inf, inf))
            self.assertEqual(info["chi"], Fraction(-2))
            self.assertIn("conditional", info["reason"])
            self.assertNotIn("certified unique fixed point", info["reason"])
            self.assertTrue(info["cycles"])

    def test_session_exposes_the_exact_dynamical_orbifold(self):
        result = load("julia-4422").dynamical_orbifold()
        self.assertEqual(result["orbifold"]["signature"], (2, 2, 4, 4))
        self.assertEqual(result["orbifold"]["chi"], Fraction(-1, 2))
        self.assertFalse(result["lattes"]["lattes_portrait"])
        self.assertFalse(result["lattes"]["flexible"])

    def test_four_point_lattes_portrait_is_not_called_flexible(self):
        p = four_point_lattes_portrait()
        orb = orbifold(p)
        lat = lattes_report(p, orb)
        self.assertEqual(orb["signature"], (2, 2, 2, 2))
        self.assertEqual(orb["chi"], 0)
        self.assertTrue(lat["lattes_portrait"])
        self.assertEqual(lat["flexible"], "undetermined")
        info = guard_verdict(p, orb, lat)
        self.assertFalse(info["allowed"])
        self.assertEqual(info["status"], "parabolic")
        self.assertIn("not by itself a rational-realisation certificate",
                      info["reason"])
        self.assertIn("undetermined", info["reason"])
        self.assertIn("torus-lift", info["reason"])
        self.assertNotIn("even though the dynamical orbifold is hyperbolic",
                         info["reason"])

    def test_undetermined_markings_are_not_permission(self):
        for name in ("grigorchuk", "julia-4442"):
            info = convergence_guard(load(name))
            self.assertFalse(info["allowed"], name)
            self.assertFalse(info["candidate"], name)
            self.assertEqual(info["status"], "undetermined")
            self.assertIn("not determined", info["reason"])
            with self.assertRaises(ConvergenceError):
                require_convergence_allowed(load(name))

    def test_iteration_gate_does_not_decide_obstruction(self):
        # The dynamical gate no longer blocks a potentially obstructed map.
        # This does not validate the present coarse discretisation: its rank
        # and embedding defects are independent and still have to pass before
        # an actual numerical step can be accepted.
        state = load("julia-4422").initial_dual_state()
        self.assertTrue(state.convergence["allowed"])
        self.assertTrue(state.convergence["unbranched_cycles"])
        self.assertTrue(state.cover["certificates"]["deficit"] > 0)
        other = InitialState(load("julia-4422"))
        self.assertTrue(other.convergence["allowed"])
        self.assertTrue(require_convergence_allowed(other.session)["allowed"])


if __name__ == "__main__":
    unittest.main()
