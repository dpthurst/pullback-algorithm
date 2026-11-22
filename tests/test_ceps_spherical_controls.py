"""Controls for the spherical CEPS uniformizer that the gate did not cover.

Each test here is written against an *independent* statement, not against the
solver's own residual:

* the triangle function and its gradient/Hessian are compared with central
  finite differences of the energy itself (so the signs of the spherical
  modified energy, GSC sec. 7.2 / Springborn Prop. 7.7, are tested, not
  assumed);
* every special vertex of every fixture is solved and the answer is compared
  with the fixture through a Moebius registration (Springborn Thm. 10.5), for
  the *vertex-scaled* inputs as well -- a metric in a discrete conformal class
  must uniformize to the same polyhedron;
* the re-inserted polyhedron is certified independently of u: closed oriented
  surface, every vertex labelled, sphere containment, supporting planes, and
  the intrinsic scaling consistency of the flat layout.

Conventions borrowed from the MIT CEPS reference implementation
(https://github.com/MarkGillespie/CEPS, commit b734d2c, MIT licence) are marked
where they are used: the clamped-cosine corner angle
(``geometry-central``/``EdgeLengthGeometry::cornerAngle``) and the rule that
cotan weights of zero-area faces are dropped
(``PartiallyDecoratedTriangulation::partialCotanLaplacian``).
"""

import math
import unittest

import numpy as np

from pullback import spherical_ceps as sc

from tests.test_ceps_spherical import (OCTA, TETRA, hull_fixture,
                                      mobius_residual)


class TestClampedAngleConvention(unittest.TestCase):
    """The energy exists off the domain of realizable Euclidean triangles."""

    def test_degenerate_triangle_angles_are_the_limit(self):
        # 1.7 > 0.3 + 0.33: not a Euclidean triangle.  The clamped cosine
        # convention gives the degenerate limit (pi, 0, 0).
        a1, a2, a3 = sc.tri_angles(1.7, 0.3, 0.33)
        self.assertAlmostEqual(a1, math.pi, places=12)
        self.assertEqual((a2, a3), (0.0, 0.0))

    def test_energy_is_finite_on_a_non_realizable_metric(self):
        X, F = hull_fixture(5, 0)
        ell = sc.chordal_lengths(F, X)
        rng = np.random.RandomState(3)
        u = {v: float(rng.normal()) for v in range(5) if v != 0}
        E, g, _st = sc.energy_at(F, ell, 0, u)
        self.assertTrue(math.isfinite(E))
        self.assertTrue(all(math.isfinite(g[v]) for v in g))

    def test_scaling_identity_off_the_domain(self):
        """E(u + h 1) = E(u) + 2 pi h holds for *every* u (Prop. 7.17), also
        where the metric is not a realizable Euclidean cone metric."""
        X, F = hull_fixture(5, 0)
        ell = sc.chordal_lengths(F, X)
        rng = np.random.RandomState(3)
        u = {v: float(rng.normal()) for v in range(5) if v != 0}
        E0, _, _ = sc.energy_at(F, ell, 0, u)
        for h in (0.1, -0.3, 1.7):
            uh = {v: u[v] + h for v in u}
            E1, _, _ = sc.energy_at(F, ell, 0, uh)
            self.assertAlmostEqual(E1 - E0, 2 * math.pi * h, places=9)


class TestDerivatives(unittest.TestCase):
    """Gradient and Hessian against finite differences of E itself."""

    def _setup(self, n=6, seed=1, v_inf=4):
        X, F = hull_fixture(n, seed)
        ell = sc.chordal_lengths(F, X)
        res = sc.solve(F, ell, v_inf)
        # a perturbation of the answer: the chart still is the Delaunay one,
        # so E is C^2 there and finite differences are meaningful.
        rng = np.random.RandomState(7)
        u = {v: res.u[v] + 0.05 * float(rng.normal()) for v in res.vs}
        return F, ell, v_inf, res.vs, u

    def test_gradient_matches_central_differences(self):
        F, ell, v_inf, vs, u = self._setup()
        E0, g0, st = sc.energy_at(F, ell, v_inf, u)
        h = 1e-5
        for v in vs:
            up, um = dict(u), dict(u)
            up[v] += h
            um[v] -= h
            Ep = sc.energy_at(F, ell, v_inf, up)[0]
            Em = sc.energy_at(F, ell, v_inf, um)[0]
            fd = (Ep - Em) / (2.0 * h)
            self.assertLess(abs(fd - g0[v]), 1e-5 * max(1.0, abs(g0[v])),
                            f"dE/du_{v}: analytic {g0[v]}, fd {fd}")

    def test_hessian_matches_differences_of_the_gradient(self):
        """cotan Laplacian checked column by column on ONE chart.

        The chart is frozen, so this tests the second derivative of the energy
        on a Penner cell.  (Across a flip E is only C^1 at a cocircular
        configuration, so a difference quotient that crosses a flip measures
        something else; that is why the chart is frozen here.)
        """
        F, ell, v_inf, vs, u = self._setup()
        _E, g0, st = sc.energy_at(F, ell, v_inf, u)
        H = sc.hessian(u, st, vs).toarray()
        h = 1e-6
        worst = 0.0
        for k, v in enumerate(vs):
            up, um = dict(u), dict(u)
            up[v] += h
            um[v] -= h
            gp = sc.energy_grad(up, st, vs)[1]
            gm = sc.energy_grad(um, st, vs)[1]
            col = np.array([(gp[w] - gm[w]) / (2.0 * h) for w in vs])
            worst = max(worst, float(np.max(np.abs(col - H[:, k]))))
        self.assertLess(worst, 1e-6)
        # symmetric with the expected one-dimensional kernel along 1
        self.assertLess(float(np.max(np.abs(H - H.T))), 1e-12)
        self.assertLess(float(np.max(np.abs(H @ np.ones(len(vs))))), 1e-8)


class TestEverySpecialVertex(unittest.TestCase):
    """The gate's own cases plus the four that used to fail, all v_inf."""

    def _certify(self, X, F, v_inf, ell, tol_kkt=1e-10, tol_mob=1e-8):
        res = sc.run(F, ell, v_inf)
        self.assertLessEqual(res.kkt, tol_kkt)
        cert = res.certificate
        self.assertLess(cert["sphere_err"], 1e-12)
        self.assertGreater(cert["min_pair_distance"], 1e-6)
        self.assertGreater(cert["worst_support_margin"], 0.0)
        self.assertTrue(cert["oriented_closed_surface"])
        self.assertTrue(cert["all_labelled"])
        self.assertLess(cert["flat_scaling_residual"], 1e-10)
        self.assertLess(mobius_residual(res.positions, X), tol_mob)
        return res

    def test_regular_polyhedra_all_special_vertices(self):
        for X, F in (TETRA, OCTA):
            ell = sc.chordal_lengths(F, X)
            for v_inf in range(len(X)):
                with self.subTest(v_inf=v_inf):
                    self._certify(X, F, v_inf, ell)

    def test_random_fixtures_all_special_vertices(self):
        for n, seed in ((5, 0), (6, 1), (7, 2)):
            X, F = hull_fixture(n, seed)
            ell = sc.chordal_lengths(F, X)
            for v_inf in range(n):
                with self.subTest(n=n, seed=seed, v_inf=v_inf):
                    self._certify(X, F, v_inf, ell)

    def test_vertex_scaled_inputs_every_special_vertex(self):
        """Ell = l * exp((u_i + u_j)/2) is the same conformal class."""
        for n, seed in ((5, 0), (6, 1), (7, 2)):
            X, F = hull_fixture(n, seed)
            ell = sc.chordal_lengths(F, X)
            rng = np.random.RandomState(seed + 10)
            u = rng.normal(scale=0.7, size=n)
            ell2 = {e: l * math.exp(0.5 * (u[e[0]] + u[e[1]]))
                    for e, l in ell.items()}
            for v_inf in range(n):
                with self.subTest(n=n, seed=seed, v_inf=v_inf):
                    self._certify(X, F, v_inf, ell2)

    def test_strongly_scaled_input(self):
        """scale 2.0 in u is far from the fixture: the bounds must still hold."""
        X, F = hull_fixture(6, 1)
        ell = sc.chordal_lengths(F, X)
        rng = np.random.RandomState(99)
        u = rng.normal(scale=2.0, size=6)
        ell2 = {e: l * math.exp(0.5 * (u[e[0]] + u[e[1]]))
                for e, l in ell.items()}
        for v_inf in range(6):
            with self.subTest(v_inf=v_inf):
                self._certify(X, F, v_inf, ell2)

    def test_answer_does_not_depend_on_the_peacock_limit(self):
        """delta is a limit; the solver must not depend on how it is reached.

        Prop. 5.11 of Springborn 2019 replaces the infinite decoration by a
        sufficiently large finite T; the constraint values and hence the answer
        must be the same for T = 20, 40, 80.
        """
        X, F = hull_fixture(6, 1)
        ell = sc.chordal_lengths(F, X)
        base = sc.solve(F, ell, 4, delta_T=40.0)
        for T in (20.0, 80.0):
            r2 = sc.solve(F, ell, 4, delta_T=T)
            self.assertLessEqual(r2.kkt, 1e-10)
            self.assertLess(max(abs(r2.u[v] - base.u[v]) for v in base.vs),
                            1e-7)

    def test_answer_does_not_depend_on_the_unused_start_margin(self):
        """The starting heuristic is gone; the old knob must be inert."""
        X, F = hull_fixture(6, 1)
        ell = sc.chordal_lengths(F, X)
        r = sc.solve(F, ell, 4)
        for margin in (0.03, 5.0):
            r2 = sc.solve(F, ell, 4, margin=margin)
            self.assertLessEqual(r2.kkt, 1e-10)
            self.assertLess(max(abs(r2.u[v] - r.u[v]) for v in r.vs), 1e-12)


class TestNegativeControls(unittest.TestCase):
    def test_degenerate_input_is_rejected(self):
        X, F = hull_fixture(5, 0)
        ell = sc.chordal_lengths(F, X)
        ell0 = {e: 0.0 for e in ell}
        with self.assertRaises(Exception):
            sc.solve(F, ell0, 0)

    def test_open_surface_is_rejected(self):
        X, F = hull_fixture(5, 0)
        with self.assertRaises(ValueError):
            sc.solve(F[:-1], sc.chordal_lengths(F, X), 0)

    def test_certificate_rejects_a_collapse(self):
        X, F = TETRA
        Xc = np.tile(np.array([[0.0, 0.0, 1.0]]), (4, 1))
        cert = sc.convexity_certificate(Xc, F)
        self.assertLess(cert["min_pair_distance"], 1e-12)
        self.assertLessEqual(cert["worst_support_margin"], 0.0)

    def test_certificate_rejects_a_non_convex_configuration(self):
        """One vertex pushed above a supporting plane: convexity must fail.

        (Note that pushing a vertex *inside* the hull does not break a
        supporting-plane inequality -- every plane still supports the set --
        so the control has to put a vertex on the wrong side of a face.)
        """
        X, F = OCTA
        Y = X.copy()
        Y[5] = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
        cert = sc.convexity_certificate(Y, sc.float_and_fix_orientation(Y, F))
        self.assertLessEqual(cert["worst_support_margin"], 0.0)

    def test_oriented_closed_surface_rejects_a_reversed_face(self):
        X, F = TETRA
        good = sc.float_and_fix_orientation(X, F)
        self.assertTrue(sc.oriented_closed_surface(good))
        a, b, c = good[0]
        bad = [(a, c, b)] + good[1:]
        self.assertFalse(sc.oriented_closed_surface(bad))


if __name__ == "__main__":
    unittest.main()
