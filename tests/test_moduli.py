"""The four-point moduli chart: known-answer checks.

- cr(0, 1, inf, z) = (z - 1)/z, by the convention stated in the module (the
  other common normalisation differs by z -> 1/(1-z), a symmetry of the six
  values, so nothing distinguishes them mathematically);
- the cross-ratio is invariant under every Möbius transformation (tested on
  a random one, and on the ones that fix or permute the standard points);
- four points are concyclic or collinear iff the cross-ratio is real: the
  test uses a quadruple on the unit circle (real) and a generic one (not);
- the six values of one configuration are closed under the substitutions, so
  two of them describe the same configuration and a third does not.
"""

import sys
import unittest

import numpy as np

sys.path.insert(0, "src")

from pullback.moduli import (
    apply_mobius,
    cross_ratio,
    normalise,
    orbit,
    same_configuration,
)

INF = complex(np.inf)


class TestCrossRatio(unittest.TestCase):
    def test_standard_normalisation(self):
        # cr(0,1,inf,z) = (z-1)/z under this convention, and the model point
        # (0, 1, z) with infinity in the fourth slot gives z/(z-1); the two
        # are reciprocal substitutes, as the six-value orbit requires.
        for z in (0.3 + 0.4j, -2.0, 0.5j, 3 + 1e-3j):
            self.assertAlmostEqual(abs(cross_ratio(0, 1, INF, z) - (z - 1) / z),
                                   0.0, places=12)
            self.assertAlmostEqual(abs(cross_ratio(0, 1, z, INF) - z / (z - 1)),
                                   0.0, places=12)

    def test_infinite_entries_by_limit(self):
        # cr(inf, 0, 1, 2) = (0 - 2)/(0 - 1) = 2
        self.assertAlmostEqual(abs(cross_ratio(INF, 0, 1, 2) - 2.0), 0.0, places=12)
        # and it agrees with the finite limit along a large real value
        big = 1e9
        self.assertAlmostEqual(abs(cross_ratio(big, 0, 1, 2) - 2.0), 0.0, places=6)

    def test_invariance_under_a_random_mobius_map(self):
        rng = np.random.RandomState(7)
        m = tuple(complex(rng.normal(), rng.normal()) for _ in range(4))
        pts = [0.2 + 0.3j, -1.1 + 0.4j, 0.7 - 0.9j, 2.0 + 0.1j]
        moved = [apply_mobius(m, p) for p in pts]
        want = cross_ratio(*pts)
        got = cross_ratio(*moved)
        self.assertAlmostEqual(abs(got - want) / max(1.0, abs(want)), 0.0, places=10)

    def test_mobius_invariance_through_infinity(self):
        pts = [INF, 0.0, 1.0, -2.0]
        m = (1.0, 2.0, 3.0, 4.0)          # sends infinity to 1/3, finite elsewhere
        moved = [apply_mobius(m, p) for p in pts]
        self.assertAlmostEqual(abs(cross_ratio(*moved) - cross_ratio(*pts)), 0.0, places=10)

    def test_real_iff_concyclic(self):
        circle = [np.exp(1j * t) for t in (0.1, 1.0, 2.5, 4.0)]
        self.assertAlmostEqual(cross_ratio(*circle).imag, 0.0, places=12)
        generic = [0.1 + 0.2j, 1.0 - 0.4j, -0.5 + 0.9j, 0.3 - 1.1j]
        self.assertGreater(abs(cross_ratio(*generic).imag), 1e-3)

    def test_normalise_and_orbit(self):
        pts = [0.4 + 0.7j, 2.0, -1.0 + 0.2j, 0.0]
        z = normalise(pts)
        self.assertEqual(same_configuration(z, z), True)
        self.assertEqual(same_configuration(z, 1.0 / z), True)
        self.assertEqual(same_configuration(z, 1.0 - z), True)
        self.assertEqual(same_configuration(z, z + 3.0), False)
        self.assertEqual(len({round(v.real, 9) for v in orbit(0.3 + 0.2j)}), 6)


if __name__ == "__main__":
    unittest.main()
