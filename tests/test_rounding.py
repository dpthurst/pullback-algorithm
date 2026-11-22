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
from unittest import mock

import numpy as np

sys.path.insert(0, "src")

from pullback.rounding import (  # noqa: E402
    chordal_lengths,
    conformal_jacobian,
    conformal_residual,
    edges_of,
    incidence,
    projector,
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


    def test_input_cone_angles_above_two_pi_are_allowed(self):
        # Vertex scaling DEMONSTRABLY changes angles; convexity is a property of
        # the OUTPUT in Springborn's theorem, not a restriction on the input.
        x = octahedron()
        l = lengths_from(x)
        u = np.array([
            0.1437470470395749, 0.34935892539897817,
            -0.5204435968520382, 0.04120711577607982,
            0.025425940372266212, 0.33034889916502225,
        ])
        scaled = {e: l[e] * np.exp(0.5 * (u[e[0]] + u[e[1]])) for e in l}

        cone = {i: 0.0 for i in range(6)}
        for tri in FACES:
            for k, a in enumerate(tri):
                b, c = tri[(k + 1) % 3], tri[(k + 2) % 3]
                ab = scaled[tuple(sorted((a, b)))]
                ac = scaled[tuple(sorted((a, c)))]
                bc = scaled[tuple(sorted((b, c)))]
                self.assertLess(ab, ac + bc)  # still a Euclidean triangle
                cone[a] += float(np.arccos(np.clip(
                    (ab * ab + ac * ac - bc * bc) / (2 * ab * ac), -1, 1)))
        self.assertGreater(cone[2], 2 * np.pi)
        self.assertAlmostEqual(cone[2], 6.4436825223649405, places=13)

        rng = np.random.RandomState(11)
        x0 = x + 0.15 * rng.normal(size=x.shape)
        _xr, norm, iters = round_realise(
            FACES, scaled, x0=x0, maxit=100, tol=1e-12)
        self.assertLess(norm, 1e-12)
        self.assertLess(iters, 100)


class TestAnalyticJacobian(unittest.TestCase):
    """The Jacobian that replaced the finite-difference one (2026-10-28).

    Three things are pinned: the projector rewrite leaves the residual bit-identical,
    the analytic Jacobian agrees with finite differences at the rate the *finite
    difference* is wrong (so the disagreement shrinking linearly in h is evidence
    for the analytic one, not against it), and round_realise now reports a true
    iteration count instead of the cap.
    """

    def setUp(self):
        self.faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]
        rng = np.random.RandomState(0)
        x = rng.normal(size=(4, 3))
        x /= np.linalg.norm(x, axis=1)[:, None]
        self.x = x
        chord = chordal_lengths(x)
        self.lengths = {e: float(chord[e[0], e[1]]) * 1.01 for e in edges_of(self.faces)}

    def test_the_projector_residual_is_the_least_squares_residual(self):
        r, E, _n = conformal_residual(self.faces, self.lengths, self.x)
        A, E2 = incidence(self.faces, 4)
        self.assertEqual(E, E2)
        chord = chordal_lengths(self.x)
        a = np.array([np.log(chord[i, j]) - np.log(self.lengths[(i, j)]) for (i, j) in E])
        old = a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
        self.assertLess(np.abs(r - old).max(), 1e-12)

    def test_the_residual_does_not_repeat_a_least_squares_solve(self):
        # Regression: an accidental second definition of conformal_residual
        # used to override the cached-projector implementation.  Populate the
        # projector first (pinv is allowed); the residual itself must not lstsq.
        projector(self.faces, 4)
        with mock.patch("numpy.linalg.lstsq",
                        side_effect=AssertionError("residual called lstsq")):
            conformal_residual(self.faces, self.lengths, self.x)

    def test_the_jacobian_agrees_with_finite_differences_at_the_rate_they_are_wrong(self):
        J = conformal_jacobian(self.faces, self.x)
        r0 = conformal_residual(self.faces, self.lengths, self.x)[0]
        errors = []
        for h in (1e-5, 1e-6, 1e-7):
            Jfd = np.zeros_like(J)
            for k in range(J.shape[1]):
                xp = self.x.copy()
                xp.reshape(-1)[k] += h
                Jfd[:, k] = (conformal_residual(self.faces, self.lengths, xp)[0] - r0) / h
            errors.append(np.abs(J - Jfd).max() / np.abs(Jfd).max())
        # forward differences are O(h) wrong, so the disagreement must fall by ~10
        # for each factor of 10 in h -- that is what shows the analytic side right
        self.assertLess(errors[-1], errors[0] / 5.0,
                        msg="disagreement does not fall with h: %s" % errors)

    def test_round_realise_reports_a_true_iteration_count(self):
        x, norm, iters = round_realise(self.faces, self.lengths, x0=self.x,
                                       maxit=200, tol=1e-13)
        self.assertLess(norm, 1e-6)
        self.assertLess(iters, 200, "the loop ran to the cap: no stall detection")


if __name__ == "__main__":
    unittest.main()
