"""Teichmüller vs moduli: the distinction the iteration depends on.

- Two labelled configurations with the same four points but labels permuted
  have the same moduli (the six-value orbit agrees) and different
  Teichmüller coordinates.  Moduli cannot serve as the state of the pullback
  iteration, because sigma is not well defined there.
- `pullback_labels` transports the labels through f|_P: for the rabbit's
  dynamics (0 -> c -> c^2+c -> 0, infinity fixed) the preimage of each label
  inside P is computed, and a dynamics that is not a permutation of P is
  refused.
"""

import sys
import unittest

import numpy as np

sys.path.insert(0, "src")

from pullback.moduli import Configuration, pullback_labels, same_configuration  # noqa: E402

INF = complex(np.inf)


class TestTeichVsModuli(unittest.TestCase):
    def setUp(self):
        self.a = Configuration(["p0", "p1", "p2", "p3"], [0, 1, INF, 2.0])

    def test_permutation_keeps_moduli_and_changes_the_coordinate(self):
        b = self.a.permuted([1, 0, 2, 3])          # p0 and p1 swap positions
        self.assertTrue(same_configuration(self.a.coordinate(), b.coordinate()))
        self.assertGreater(abs(self.a.coordinate() - b.coordinate()), 1e-6)

    def test_the_coordinate_uses_the_labels_not_the_positions(self):
        # the same point set read in two label orders gives 1/2 and 2
        b = self.a.permuted([1, 0, 2, 3])
        self.assertAlmostEqual(abs(self.a.coordinate() - 0.5), 0.0, places=12)
        self.assertAlmostEqual(abs(b.coordinate() - 2.0), 0.0, places=12)

    def test_a_bad_permutation_is_refused(self):
        with self.assertRaises(ValueError):
            self.a.permuted([0, 0, 1, 2])


class TestLabelTransport(unittest.TestCase):
    def test_rabbit_dynamics_transports_the_labels(self):
        # f(0) = c, f(c) = c^2+c, f(c^2+c) = 0, f(inf) = inf
        dynamics = {"p0": "p1", "p3": "p3", "p1": "p2", "p2": "p0"}
        pre = pullback_labels(["p0", "p1", "p2", "p3"], dynamics)
        self.assertEqual(pre, ["p2", "p0", "p1", "p3"])

    def test_a_non_permutation_is_refused(self):
        dynamics = {"p0": "p1", "p1": "p1", "p2": "p0", "p3": "p3"}
        with self.assertRaises(ValueError):
            pullback_labels(["p0", "p1", "p2", "p3"], dynamics)


if __name__ == "__main__":
    unittest.main()
