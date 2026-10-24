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


if __name__ == "__main__":
    unittest.main()
