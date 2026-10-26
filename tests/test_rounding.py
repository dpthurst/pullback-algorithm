"""The round realisation: known-answer checks on the octahedron.

The criterion is exact, so the tests are strong:
- the regular octahedron is inscribed, so its chordal metric is round: the
  conformal residual vanishes at its own configuration;
- a vertex scaling of a round metric is discretely conformally equivalent to
  it (by definition), so the residual still vanishes -- this is the property
  the solver relies on;
- an arbitrary perturbation of one edge changes the conformal class, so the
  residual is bounded away from zero;
- a metric that is not piecewise Euclidean at all (one edge longer than the
  sum of the other two in a face) has no round realisation, so the solver
  cannot drive the residual to zero -- the check that can fail.  A metric
  with a flat vertex is NOT such a counterexample: sides 1, 1, 2 give
  degenerate triangles, and the degenerate inscribed limit still satisfies
  the conformal equations (measured: residual 2.5e-13, so the first version
  of this test was wrong about its own premise);
- and round_realise recovers a round configuration from a perturbed start.
"""

import sys
import unittest

import numpy as np

sys.path.insert(0, "src")

from pullback.rounding import (  # noqa: E402
    chordal_lengths,
    conformal_residual,
    edges_of,
    round_realise,
)

FACES = [(0, 2, 4), (0, 4, 3), (0, 3, 5), (0, 5, 2),
         (1, 4, 2), (1, 3, 4), (1, 5, 3), (1, 2, 5)]


def octahedron():
    return np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                     [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)


def lengths_from(x, faces=FACES):
    d = chordal_lengths(x)
    return {(i, j): d[i, j] for (i, j) in edges_of(faces)}


class TestCriterion(unittest.TestCase):
    def setUp(self):
        self.x = octahedron()
        self.l = lengths_from(self.x)

    def test_round_metric_has_zero_residual(self):
        _r, _E, norm = conformal_residual(FACES, self.l, self.x)
        self.assertLess(norm, 1e-13)

    def test_vertex_scaling_preserves_the_class(self):
        rng = np.random.RandomState(3)
        u = rng.normal(size=6)
        scaled = {e: self.l[e] * np.exp(0.5 * (u[e[0]] + u[e[1]]))
                  for e in self.l}
        _r, _E, norm = conformal_residual(FACES, scaled, self.x)
        self.assertLess(norm, 1e-13)
        chord = chordal_lengths(self.x)
        E = edges_of(FACES)
        a = np.log([chord[i, j] for (i, j) in E]) - np.log([scaled[e] for e in E])
        self.assertGreater(np.std(a), 1e-3)     # u is not constant: a real test

    def test_one_edge_perturbed_is_not_the_same_class(self):
        bad = dict(self.l)
        key = edges_of(FACES)[0]
        bad[key] *= 1.3
        _r, _E, norm = conformal_residual(FACES, bad, self.x)
        self.assertGreater(norm, 1e-3)


class TestSolver(unittest.TestCase):
    def test_recovers_from_a_perturbed_start(self):
        x = octahedron()
        l = lengths_from(x)
        rng = np.random.RandomState(11)
        x0 = x + 0.15 * rng.normal(size=x.shape)
        xr, norm, _it = round_realise(FACES, l, x0=x0, maxit=60)
        self.assertLess(norm, 1e-10)
        self.assertLess(abs(np.linalg.norm(xr, axis=1) - 1).max(), 1e-12)

    def test_a_non_euclidean_metric_cannot_be_rounded(self):
        # one edge longer than the sum of the other two in its face: no
        # Euclidean triangle exists, so no inscribed polyhedron can realise
        # this metric discretely conformally
        l = {}
        for f in FACES:
            for k in range(3):
                i, j = f[k], f[(k + 1) % 3]
                e = (i, j) if i < j else (j, i)
                l.setdefault(e, 1.0)
        l[(0, 2)] = 5.0
        _xr, norm, _it = round_realise(FACES, l, maxit=80)
        self.assertGreater(norm, 1e-3)


if __name__ == "__main__":
    unittest.main()
