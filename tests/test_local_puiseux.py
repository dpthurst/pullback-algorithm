"""Local Puiseux series: corrected domains, branch mechanics and polish rules."""
import unittest
from unittest import mock

import mpmath as mp

from pullback import local_puiseux as LP
from pullback import preimage_lift as PL
from pullback.rational import RationalMap


class TestLocalPuiseux(unittest.TestCase):
    def setUp(self):
        self.old_dps = mp.mp.dps
        mp.mp.dps = 60

    def tearDown(self):
        mp.mp.dps = self.old_dps

    def test_affine_power_has_one_term_and_infinite_radius(self):
        for d in (2, 3, 4):
            with self.subTest(degree=d):
                expansion = LP.local_series(
                    [1] + [0] * d, [1], 0, 8, k=d, dps=60)
                self.assertEqual(expansion["k"], d)
                self.assertTrue(expansion["exact"])
                self.assertEqual(expansion["radius_tau"], mp.inf)
                self.assertLess(abs(expansion["coeffs"][0] - 1), mp.mpf("1e-45"))
                self.assertLess(max(map(abs, expansion["coeffs"][1:])),
                                mp.mpf("1e-40"))
                got, _T = LP.seeds(expansion, mp.mpc("0.3", "-0.2"))
                self.assertEqual(len(got), d)
                self.assertLess(max(LP.residual(
                    [1] + [0] * d, [1], mp.mpc("0.3", "-0.2"), z)
                    for z in got), mp.mpf("1e-45"))

    def test_coincident_critical_value_does_not_zero_the_radius(self):
        # f = z^2(z-1)^2. Critical values 0 and 1/16; at p=0 the
        # other critical preimage of b=0 must not be treated as a singularity
        # at tau=0.  R_w=1/16, hence R_tau=1/4.
        num, den = [1, -2, 1, 0, 0], [1]
        expansion = LP.local_series(num, den, 0, 8, k=2, dps=60)
        self.assertLess(abs(expansion["radius_tau"] - mp.mpf(1) / 4),
                        mp.mpf("1e-45"))
        expected = [1, 1, 2, 5]
        for coefficient, target in zip(expansion["coeffs"], expected):
            self.assertLess(abs(coefficient - target), mp.mpf("1e-45"))
        got, _T = LP.seeds(expansion, mp.mpf("1e-8"))
        self.assertLess(max(LP.residual(num, den, mp.mpf("1e-8"), z)
                            for z in got), mp.mpf("1e-25"))

    def test_escape_value_limits_mobius_conjugate_and_mobius_maps(self):
        # There are only two critical values, but the source chart u=z-1 has
        # a pole when the inverse reaches z=infinity at w=f(infinity)=1.
        num = [1, -3, 3, -1]
        den = [1, 3, 3, 1]
        expansion = LP.local_series(num, den, 1, 8, k=3, dps=60)
        self.assertLess(abs(expansion["radius_tau"] - 1), mp.mpf("1e-45"))
        self.assertFalse(expansion["exact"])
        for coefficient in expansion["coeffs"][:5]:
            self.assertLess(abs(coefficient - 2), mp.mpf("1e-40"))

        # Even a degree-one inverse can have a finite pole radius.
        mobius = LP.local_series([1], [1, 0], 1, 6, k=1, dps=60)
        self.assertLess(abs(mobius["radius_tau"] - 1), mp.mpf("1e-45"))

    def test_infinite_centre_uses_the_zero_source_escape(self):
        # f = z/(z-1).  At p=infinity, u=1/z and the inverse is
        # u=tau/(1+tau) with tau=w-1.  Its pole is at w=f(0)=0, so R=1;
        # omitting this source-zero escape would incorrectly give infinity.
        expansion = LP.local_series([1, 0], [1, -1], mp.inf, 20, k=1, dps=60)
        self.assertEqual(expansion["k"], 1)
        self.assertLess(abs(expansion["radius_tau"] - 1), mp.mpf("1e-45"))
        escape = [row for row in expansion["critical"]
                  if row[3] == "escape-to-source-pole"]
        self.assertEqual(len(escape), 1)
        self.assertEqual(escape[0][0], 0)
        self.assertLess(abs(escape[0][1] + 1), mp.mpf("1e-45"))
        for index, coefficient in enumerate(expansion["coeffs"][:5]):
            self.assertLess(abs(coefficient - (-1) ** index), mp.mpf("1e-45"))
        got, _T = LP.seeds(expansion, mp.mpf("0.9"))
        self.assertLess(abs(got[0] + 9), mp.mpf("1e-15"))

    def test_infinity_critical_row_uses_the_actual_sphere_value(self):
        # f=1+1/z^2 at p=1 has b=2 and a critical point at infinity with
        # f(infinity)=1. Its chart value is A=-1, never the old placeholder 0.
        expansion = LP.local_series([1, 0, 1], [1, 0, 0], 1, 8, k=1, dps=60)
        rows = [row for row in expansion["critical"]
                if row[0] == mp.inf and row[3] == "other"]
        self.assertTrue(rows, expansion["critical"])
        self.assertLess(abs(rows[0][1] + 1), mp.mpf("1e-45"))
        self.assertLess(abs(rows[0][2] - 1), mp.mpf("1e-45"))
        self.assertLess(abs(expansion["radius_all_tau"] - 1), mp.mpf("1e-45"))
        self.assertLess(abs(expansion["radius_tau"] - 1), mp.mpf("1e-45"))

    def test_checked_evaluation_refuses_outside_the_lower_bound(self):
        expansion = LP.local_series([1, 1, 0, 0], [1], 0, 12, k=2, dps=60)
        radius = expansion["radius_tau"]
        self.assertEqual(expansion["radius_kind"], "conservative_lower_bound")
        target = (mp.mpf("1.5") * radius) ** 2
        with self.assertRaisesRegex(LP.LocalSeriesError, "lower bound"):
            LP.seeds(expansion, target)
        with self.assertRaisesRegex(LP.LocalSeriesError, "margin"):
            LP.seeds(expansion, mp.mpf("1e-8"), margin=2)

    def test_escape_candidate_can_be_regular_on_the_selected_branch(self):
        # f = z^2/(z^2-3z+1), at a simple centre near the finite preimage
        # z=1/3 of f(infinity)=1.  The escape value bounds a sufficient
        # domain, but this branch goes to 1/3 there rather than to infinity.
        num, den = [1, 0, 0], [1, -3, 1]
        with mp.workdps(70):
            expansion = LP.local_series(num, den, mp.mpf("0.34"), 50,
                                        k=1, dps=70)
            b, radius = expansion["b"], expansion["radius_tau"]
            self.assertGreater(LP.radius_from_coefficients(expansion)[0],
                               4 * radius)
            target = b + 2 * radius
            with self.assertRaisesRegex(LP.LocalSeriesError, "lower bound"):
                LP.seeds(expansion, target)
            for factor in (2, 4):
                target = b + factor * radius
                got, _T = LP.seeds(expansion, target, check_radius=False)
                roots = mp.polyroots([1 - target, 3 * target, -target],
                                     maxsteps=2000, extraprec=100)
                nearest = min(abs(got[0] - root) for root in roots)
                self.assertLess(nearest, mp.mpf("1e-9"))
                self.assertLess(LP.residual(num, den, target, got[0]),
                                mp.mpf("1e-8"))
            got, _T = LP.seeds(expansion, 1, check_radius=False)
            self.assertLess(abs(got[0] - mp.mpf(1) / 3), mp.mpf("1e-30"))

    def test_other_sheet_critical_value_does_not_force_divergence(self):
        # f = z^3-3z.  At critical value -2 the selected local branch is the
        # SIMPLE preimage near z=-2; the other two sheets collide at z=1.
        # Including -2 gives a safe lower bound but not the selected radius.
        num, den = [1, 0, -3, 0], [1]
        with mp.workdps(70):
            expansion = LP.local_series(num, den, mp.mpf("-1.99"), 50,
                                        k=1, dps=70)
            b, radius = expansion["b"], expansion["radius_tau"]
            self.assertLess(abs(radius - abs(-2 - b)), mp.mpf("1e-50"))
            self.assertGreater(LP.radius_from_coefficients(expansion)[0],
                               4 * radius)
            with self.assertRaisesRegex(LP.LocalSeriesError, "lower bound"):
                LP.seeds(expansion, b + 2 * radius)
            for factor in (2, 4):
                target = b + factor * radius
                got, _T = LP.seeds(expansion, target, check_radius=False)
                roots = mp.polyroots([1, 0, -3, -target],
                                     maxsteps=2000, extraprec=100)
                nearest = min(abs(got[0] - root) for root in roots)
                self.assertLess(nearest, mp.mpf("1e-40"))
                self.assertLess(LP.residual(num, den, target, got[0]),
                                mp.mpf("1e-35"))

    def test_near_coincident_nonzero_value_triggers_precision_refinement(self):
        # At 40 digits f(1)-f(p)=2.7e-23 is below the candidate threshold but
        # nonzero. The build must retry at 80 digits and report the small bound,
        # never drop it and report R=4 (which allowed catastrophic extrapolation).
        num, den = [1, 0, -3, 0], [1]
        p = 1 + mp.mpf("3e-12")
        with mp.workdps(40):
            expansion = LP.local_series(num, den, p, 20, k=1, dps=40)
            self.assertEqual(expansion["radius_status"], "known")
            self.assertEqual(expansion["radius_retries"], 1)
            self.assertEqual(expansion["radius_dps"], 80)
            self.assertLess(expansion["radius_tau"], mp.mpf("1e-20"))
            self.assertLessEqual(expansion["radius_tau"],
                                 10 * 3 * mp.mpf("3e-12") ** 2)
            with self.assertRaisesRegex(LP.LocalSeriesError, "lower bound"):
                LP.seeds(expansion, expansion["b"] + mp.mpf("1e-3"))

    def test_radius_unavailable_is_not_an_implicit_domain_certificate(self):
        with mock.patch.object(LP, "validity_radius",
                               side_effect=mp.libmp.NoConvergence("control")):
            expansion = LP.local_series(
                [1, 1, 0, 0], [1], 0, 8, k=2, dps=60)
        self.assertIsNone(expansion["radius_tau"])
        self.assertIn("unavailable", expansion["radius_note"])
        with self.assertRaisesRegex(LP.LocalSeriesError, "unavailable"):
            LP.seeds(expansion, mp.mpf("1e-8"))

    def test_residual_exponent_and_leading_constant(self):
        num, den = [1, 1, 0, 0], [1]
        expansion = LP.local_series(num, den, 0, 16, k=2, dps=70)
        for order, exponent in ((2, 2), (4, 3), (6, 4)):
            with self.subTest(order=order):
                residuals = []
                for eps in (mp.mpf("1e-4"), mp.mpf("1e-6")):
                    got, T = LP.seeds(expansion, eps, n=order)
                    residual = max(LP.residual(num, den, eps, z) for z in got)
                    predicted = (abs(LP.leading_residual(expansion, n=order))
                                 * abs(T) ** (order + expansion["k"]))
                    self.assertLess(abs(residual / predicted - 1),
                                    mp.mpf("0.02"))
                    residuals.append(residual)
                measured = mp.log(residuals[0] / residuals[1]) / mp.log(100)
                self.assertLess(abs(measured - exponent), mp.mpf("0.02"))

    def test_off_vertex_targets_are_polished_as_simple_roots(self):
        num, den = [1, 1, 0, 0, 0], [1]
        expansion = LP.local_series(num, den, 0, 4, k=3, dps=60)
        target = mp.mpf("1e-4")
        initial, _T = LP.seeds(expansion, target)
        rational = RationalMap(num, den)
        common = dict(targets=[target] * 3, maxit=64, tol=mp.mpf("1e-30"),
                      position_tol=mp.mpf("1e-30"), basin_radius=None)
        polished = PL.polish_preimages(
            rational, initial, local_degrees=[1] * 3, **common)
        self.assertEqual(polished["vertex_state"], ["converged"] * 3)
        with self.assertRaises(PL.BasinError):
            PL.polish_preimages(
                rational, initial, local_degrees=[3] * 3, **common)

    def test_chordal_and_chart_tolerances_are_not_interchanged(self):
        b, radius = mp.mpf(1000), mp.mpf(1) / 16
        budget = LP.chordal_budget(b, radius)
        round_trip = LP.euclid_from_chordal(b, budget)
        self.assertLess(abs(round_trip / radius - 1), mp.mpf("1e-35"))
        far = LP.euclid_from_chordal(b, mp.mpf("1e-3"))
        expected = mp.mpf("1000.00125000032812510351565875245")
        self.assertLess(abs(far - expected), mp.mpf("1e-20"))
        self.assertGreater(far, 10000 * radius)

    def test_infinity_poles_and_existing_rational_map_coefficients(self):
        for d in (2, 3):
            with self.subTest(point="infinity", degree=d):
                num = [1] + [0] * d
                expansion = LP.local_series(num, [1], mp.inf, 6, k=d, dps=60)
                self.assertEqual(expansion["b"], mp.inf)
                got, _T = LP.seeds(expansion, mp.mpc("0.3", "0.2"))
                self.assertLess(max(LP.residual(
                    num, [1], mp.mpc("0.3", "0.2"), z) for z in got),
                    mp.mpf("1e-45"))

        expansion = LP.local_series([1], [1, 0, 0], 0, 6, k=2, dps=60)
        self.assertEqual(expansion["b"], mp.inf)
        got, _T = LP.seeds(expansion, mp.mpf("2.5"))
        self.assertLess(max(LP.residual([1], [1, 0, 0], mp.mpf("2.5"), z)
                            for z in got), mp.mpf("1e-45"))

        rational = RationalMap([53, -48, 6, 6], [24, -9, -27, 23])
        # This fixture deliberately tests an approximate centre.  A float 1/3
        # cannot establish the exact critical value, even with declared k=3,
        # so it is a coefficient-mechanics check, not a checked-domain claim.
        expansion = LP.local_series(
            rational.num, rational.den, mp.mpf(1) / 3, 12, k=3, dps=60,
            check_radius=False)
        self.assertEqual(expansion["k"], 3)
        target = expansion["b"] + mp.mpf("1e-8")
        got, _T = LP.seeds(expansion, target, check_radius=False)
        self.assertLess(max(LP.residual(rational.num, rational.den, target, z)
                            for z in got), mp.mpf("1e-25"))

    def test_branch_value_at_infinity_and_empty_chart_target(self):
        # f = 1/z^2: p=infinity is a double zero, so w=b=0 returns that
        # point twice; p=0 is the double pole, so w=b=infinity returns zero
        # twice.  Both need an explicit u=0 case, not reciprocal division.
        num, den = [1], [1, 0, 0]
        at_infinity = LP.local_series(num, den, mp.inf, 6, k=2, dps=60)
        got, T = LP.seeds(at_infinity, 0)
        self.assertEqual(T, 0)
        self.assertEqual(got, [mp.inf, mp.inf])

        at_zero = LP.local_series(num, den, 0, 6, k=2, dps=60)
        got, T = LP.seeds(at_zero, mp.inf)
        self.assertEqual(T, 0)
        self.assertEqual(got, [0, 0])
        with self.assertRaisesRegex(LP.LocalSeriesError, "outside"):
            LP.seeds(at_zero, 0)
        with self.assertRaisesRegex(LP.LocalSeriesError, "outside"):
            LP.seeds(at_infinity, mp.inf)

    def test_precision_is_restored_after_a_build(self):
        before = mp.mp.dps
        LP.local_series([1, 1, 0, 0], [1], 0, 5, k=2, dps=80,
                        check_radius=False)
        self.assertEqual(mp.mp.dps, before)


if __name__ == "__main__":
    unittest.main()
