"""Cross-check: sessions' pi-portraits against the analytic maps.

The multiset of local degrees over P, computed from f's coefficients,
must equal the pi-portrait degree multiset of the session that records
the same map.  This is the ground truth an SC importer must reproduce,
and the first module of that importer (rational.py).
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import mpmath as mp
mp.mp.dps = 30

from pullback.rational import RationalMap
import json

I = mp.mpc(0, 1)
MAPS = {
    "theta": ([1, 0, 1], [-1, 0, 1]),                       # (1+z^2)/(1-z^2)
    "julia-4442": ([2, 0, -4], [1, 0, 0]),                  # (2z^2-4)/z^2
    "julia-4422": ([1, 0, 1 - I], [(1 - I), 0, -(1 - I)]),  # 1+(z^2/(1-i)) over 1-z^2
    "slit-lattes": ([1, -1, 0, 0],                          # z^2(z-1)/(z-1/3)^3
                    [1, -1, mp.mpf(1) / 3, -mp.mpf(1) / 27]),
}


class TestPortraitDegrees(unittest.TestCase):
    def test_against_sessions(self):
        fixtures = json.loads((ROOT / "tests" / "library_fixtures.json").read_text())
        for name, (num, den) in MAPS.items():
            with self.subTest(session=name):
                got = RationalMap(num, den).portrait_degrees()
                self.assertEqual(got, fixtures[name]["portrait_degrees"],
                                 f"{name}: analytic portrait vs session")

    def test_degree_counts(self):
        for name, (num, den) in MAPS.items():
            with self.subTest(session=name):
                f = RationalMap(num, den)
                P, has_inf, tol = f.postcritical_set()
                n = len(P) + (1 if has_inf else 0)
                # |f^{-1}(P)| = d * |P| counting multiplicity; the face
                # count of the cover is the DISTINCT count
                degs = f.portrait_degrees()
                self.assertTrue(all(d >= 1 for d in degs))
                self.assertEqual(sum(degs), (len(num) - 1 if len(num) >= len(den) else len(den) - 1) * n)


class TestPostcriticalOrbitThroughInfinity(unittest.TestCase):
    """The forward closure must continue PAST infinity, not stop at it.

    The old `postcritical_set` broke at the first infinite value and threw away
    the rest of every critical orbit that passes through infinity.  For strictly
    rational maps -- whose critical orbits pass through infinity constantly --
    that made perfectly PCF maps look like "not postcritically finite" further
    downstream.  Measured on IMG's table: before the fix, [3,5] gave {0, inf};
    after, {0, 1, inf}.
    """

    def test_an_orbit_through_infinity_is_not_truncated(self):
        # ((z-1)/z)^2: critical points 0 (double pole) and 1 (double zero).
        # The orbit of 0 is 0 -> infinity -> 1 -> 0, so P = {0, 1, infinity}.
        numlen = False
        from pullback.discrete import pcf_data
        den = [1, 0, 0]
        num = [1, -2, 1]
        f = RationalMap(num, den)
        # postcritical_set returns the FINITE points plus a flag for infinity
        P, has_inf, _tol = f.postcritical_set(depth=64)
        self.assertEqual(len(P), 2)
        self.assertTrue(has_inf)
        self.assertIn(mp.mpc(0), P)
        self.assertIn(mp.mpc(1), P)
        d = pcf_data(f)
        # and the dynamics is a 3-cycle, a permutation, not a truncation
        self.assertEqual(sorted(d["dynamics"].values()), [0, 1, 2])

    def test_the_golden_maps_keep_a_complete_four_cycle(self):
        # f(z) = 1/(cz+1)^2 with c = (-3+sqrt 5)/2: P is the 4-cycle
        # 0 -> 1 -> phi^2 -> infinity -> 0, and |f^-1(P)| = 6
        from pullback.discrete import pcf_data
        from pullback import thurston as th
        for name, f, c in th.img_golden_maps():
            d = pcf_data(f)
            self.assertEqual(len(d["points"]), 4, msg=name)
            self.assertEqual(sorted(d["dynamics"].values()), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
