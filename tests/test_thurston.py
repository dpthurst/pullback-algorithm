"""The rabbit's numeric postcritical configuration, and its portrait.

Known answers, independent of the engine's combinatorial data:
- the rabbit parameter satisfies c^3 + 2c^2 + c + 1 = 0 (from the critical
  orbit 0 -> c -> c^2+c -> 0), has positive imaginary part and |c| < 1;
- the postcritical set is {0, c, c^2+c, infinity}: four points;
- the local degrees of f over P are [1, 1, 1, 1, 2, 2] -- and the session's
  combinatorial portrait says the same, which is the cross-check that the
  analytic and the ribbon-graph portraits describe one map;
- so the pullback mesh has SIX distinct vertices, not d|P| = eight (author's
  correction, 2026-10-25): the two critical values contribute one preimage
  each, of multiplicity two.
"""

import json
import sys
import unittest
from pathlib import Path

import mpmath as mp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pullback.moduli import same_configuration  # noqa: E402
from pullback.thurston import (  # noqa: E402
    centre_table,
    distinct_preimages,
    modulus,
    period_three_centres,
    postcritical_list,
    preimage_profile,
    rabbit_map,
    rabbit_parameter,
)

mp.mp.dps = 40


class TestRabbitParameter(unittest.TestCase):
    def test_defining_cubic(self):
        c = rabbit_parameter()
        self.assertLess(abs(c ** 3 + 2 * c ** 2 + c + 1), mp.mpf("1e-28"))
        self.assertGreater(mp.im(c), 0)
        self.assertLess(abs(c), 1)

    def test_critical_orbit_is_a_three_cycle(self):
        f, c = rabbit_map(), rabbit_parameter()
        z0 = f(mp.mpc(0))
        z1 = f(z0)
        z2 = f(z1)
        self.assertLess(abs(z0 - c), mp.mpf("1e-30"))
        self.assertLess(abs(z2), mp.mpf("1e-28"))          # back to 0
        self.assertGreater(abs(z1 - z0), mp.mpf("1e-3"))   # three distinct


class TestPostcriticalConfiguration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = rabbit_map()
        cls.P = postcritical_list(cls.f)

    def test_four_points_including_infinity(self):
        self.assertEqual(len(self.P), 4)
        self.assertEqual(sum(1 for p in self.P if not mp.isfinite(p)), 1)

    def test_portrait_degrees_match_the_session(self):
        degs = preimage_profile(self.f, self.P)
        self.assertEqual(degs, [1, 1, 1, 1, 2, 2])
        want = json.loads((ROOT / "tests" / "library_fixtures.json").read_text())
        self.assertEqual(degs, sorted(want["rabbit"]["portrait_degrees"]))

    def test_six_distinct_preimages_not_eight(self):
        self.assertEqual(distinct_preimages(self.f, self.P), 6)
        # counting multiplicity it is d|P| = 8, which is the trap
        self.assertEqual(sum(preimage_profile(self.f, self.P)), 8)

    def test_modulus_is_nonreal_and_stable(self):
        tau = modulus(self.P)
        self.assertGreater(abs(tau.imag), 1e-6)
        self.assertTrue(same_configuration(tau, modulus(self.P)))
        mp.mp.dps = 60
        try:
            self.assertLess(abs(modulus(postcritical_list(rabbit_map())) - tau), 1e-12)
        finally:
            mp.mp.dps = 40


class TestPeriodThreeCentres(unittest.TestCase):
    """The three maps a twist of the rabbit can be equivalent to.

    All three have a critical 3-cycle, so all three are roots of
    c^3 + 2c^2 + c + 1; their postcritical moduli are distinguishable, which
    is what makes "which map is the twisted rabbit" a testable question.
    """

    def test_one_real_and_a_conjugate_pair(self):
        centres = dict(period_three_centres())
        self.assertEqual(set(centres), {"rabbit", "corabbit", "airplane"})
        self.assertLess(abs(mp.im(centres["airplane"])), mp.mpf("1e-30"))
        self.assertGreater(mp.im(centres["rabbit"]), 0)
        self.assertLess(mp.im(centres["corabbit"]), 0)
        self.assertLess(abs(centres["rabbit"] - mp.conj(centres["corabbit"])),
                        mp.mpf("1e-28"))

    def test_all_satisfy_the_cubic(self):
        for _name, c in period_three_centres():
            self.assertLess(abs(c ** 3 + 2 * c ** 2 + c + 1), mp.mpf("1e-28"))

    def test_moduli_are_pairwise_distinct(self):
        table = centre_table()
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertFalse(same_configuration(table[i][2], table[j][2]),
                                 (table[i][0], table[j][0]))

    def test_airplane_modulus_is_real(self):
        # a real c gives a configuration symmetric under conjugation, hence
        # four concyclic points and a real cross-ratio
        tau = [t for (n, _c, t) in centre_table() if n == "airplane"][0]
        self.assertLess(abs(tau.imag), 1e-9)

    def test_rabbit_matches_the_earlier_parameter(self):
        tau = [t for (n, _c, t) in centre_table() if n == "rabbit"][0]
        self.assertLess(abs(tau - modulus(postcritical_list(rabbit_map()))), 1e-12)


if __name__ == "__main__":
    unittest.main()
