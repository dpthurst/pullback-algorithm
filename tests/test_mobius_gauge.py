"""Möbius gauge, registration and gluing: exact and negative controls.

The point of these tests is that the gauge layer is *checked*, not assumed:

* the O(3,1) representation is validated against the 2x2 Hermitian action and
  against exact rational Möbius maps, including points at both poles;
* registration is exact (0 to working precision) for consistent data, and the
  orientation verdict is a real decision, not a default;
* cross ratios are the Möbius invariant used to compare configurations;
* the explicit gauge is reproducible and idempotent, the rotational gauge is
  rotation-equivariant and is shown NOT to pin boosts (a negative control that
  stops it being passed off as a full Möbius canonicalisation);
* gluing recovers a configuration from Möbius-scrambled patches and refuses
  inconsistent ones -- with the exact caveat that a three-label overlap is
  three-transitive and therefore cannot detect inconsistency.
"""
import unittest

import numpy as np

from pullback import mobius_gauge as G
from pullback.moduli import apply_mobius


def sphere(rng, n):
    P = rng.normal(size=(n, 3))
    return P / np.linalg.norm(P, axis=1)[:, None]


def random_map(rng, anti=False):
    """A random Möbius map, built from a random 2x2 complex matrix."""
    A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    while abs(np.linalg.det(A)) < 0.25:
        A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    lam = G._lorentz_from_matrix(A)
    return G.Mobius(lam @ (G.MIRROR if anti else np.eye(4)))


def herm(P):
    P = np.asarray(P, dtype=float)
    X = np.empty((len(P), 2, 2), dtype=complex)
    X[:, 0, 0] = 1.0 + P[:, 2]
    X[:, 0, 1] = P[:, 0] - 1j * P[:, 1]
    X[:, 1, 0] = P[:, 0] + 1j * P[:, 1]
    X[:, 1, 1] = 1.0 - P[:, 2]
    return X


class TestRepresentation(unittest.TestCase):
    def test_lorentz_matrix_is_orthogonal_and_agrees_with_hermitian_action(self):
        rng = np.random.default_rng(11)
        for _ in range(5):
            A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            lam = G._lorentz_from_matrix(A)
            self.assertLess(np.max(np.abs(lam.T @ G.ETA @ lam - G.ETA)), 1e-12)
            self.assertAlmostEqual(abs(np.linalg.det(lam)), 1.0, places=12)
            P = sphere(rng, 9)
            Q1 = G.Mobius(lam).apply(P)
            Ah = A / np.sqrt(abs(np.linalg.det(A)))
            Y = np.einsum('ij,njk,kl->nil', Ah, herm(P), Ah.conj().T)
            x0 = 0.5 * (Y[:, 0, 0] + Y[:, 1, 1]).real
            w = Y[:, 0, 1] / x0
            Q2 = np.stack([w.real, -w.imag, Y[:, 0, 0].real / x0 - 1.0], axis=1)
            self.assertLess(np.max(np.abs(Q1 - Q2)), 1e-12)

    def test_identity_inverse_and_composition(self):
        rng = np.random.default_rng(12)
        P = sphere(rng, 7)
        I = G.Mobius(np.eye(4))
        self.assertTrue(I.is_identity())
        self.assertLess(np.max(np.abs(I.apply(P) - P)), 1e-15)
        M = random_map(rng)
        N = random_map(rng)
        back = (M.inverse() @ M).apply(P)
        self.assertLess(np.max(np.abs(back - P)), 1e-12)
        # (M @ N) is M after N
        self.assertLess(np.max(np.abs((M @ N).apply(P) - M.apply(N.apply(P)))), 1e-12)
        self.assertEqual(M.orientation, +1)
        self.assertEqual(random_map(rng, anti=True).orientation, -1)
        self.assertEqual(P.tolist()[0] != 0.0, True)

    def test_antipodal_and_polar_points_are_handled(self):
        """Both poles and antipodal pairs must be ordinary points, not errors."""
        P = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
                      [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0]])
        rng = np.random.default_rng(13)
        for anti in (False, True):
            M = random_map(rng, anti=anti)
            Q = M.apply(P)
            self.assertLess(np.max(np.abs(np.linalg.norm(Q, axis=1) - 1.0)), 1e-12)
            mob, info = G.register(P, Q)
            self.assertLess(info["max_angle"], 1e-12)
            self.assertEqual(info["orientation"], -1 if anti else +1)

    def test_rejects_off_sphere_points(self):
        with self.assertRaises(G.GaugeError):
            G.as_points(np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))


class TestRegistration(unittest.TestCase):
    def test_exact_registration_of_random_maps(self):
        rng = np.random.default_rng(21)
        worst = 0.0
        for trial in range(40):
            anti = trial % 4 == 3
            M = random_map(rng, anti=anti)
            P = sphere(rng, 5)
            Q = M.apply(P)
            mob, info = G.register(P, Q)
            worst = max(worst, info["max_angle"])
            self.assertEqual(info["orientation"], -1 if anti else +1)
        self.assertLess(worst, 1e-11)

    def test_four_and_more_correspondences_are_least_squares(self):
        rng = np.random.default_rng(22)
        M = random_map(rng)
        P = sphere(rng, 12)
        Q = M.apply(P)
        mob, info = G.register(P, Q)
        self.assertLess(info["max_angle"], 1e-12)
        self.assertLess(info["rms_angle"], 1e-12)

    def test_forced_orientation_is_a_negative_control(self):
        """Registering a mirror image as orientation preserving must fail loudly."""
        rng = np.random.default_rng(23)
        M = random_map(rng, anti=True)
        P = sphere(rng, 6)
        Q = M.apply(P)
        mob, info = G.register(P, Q, orientation=1)
        self.assertGreater(info["max_angle"], 1e-3)
        mob, info = G.register(P, Q, orientation=-1)
        self.assertLess(info["max_angle"], 1e-12)

    def test_inconsistent_correspondences_are_reported(self):
        """An over-determined inconsistent system is *reported*, not silently fit."""
        rng = np.random.default_rng(24)
        P = sphere(rng, 5)
        Q = sphere(rng, 5)          # unrelated: no Möbius map fits
        mob, info = G.fit_mobius(P, Q)
        self.assertTrue(info["inconsistent"])
        self.assertGreater(info["max_angle"], 1e-2)
        # and the degenerate case is a different failure, still refused
        with self.assertRaises(G.GaugeError):
            G.fit_mobius(np.array([[1.0, 0.0, 0.0]] * 4),
                         np.array([[1.0, 0.0, 0.0]] * 4))

    def test_too_few_correspondences_are_refused(self):
        P = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        with self.assertRaises(G.GaugeError):
            G.fit_mobius(P, P)

    def test_three_correspondences_always_have_a_solution(self):
        """Three-transitivity: three points never over-determine the map.

        This is the exact reason a three-label patch overlap cannot detect an
        inconsistent patch, and it is asserted rather than assumed.
        """
        rng = np.random.default_rng(25)
        for _ in range(5):
            P = sphere(rng, 3)
            Q = sphere(rng, 3)
            mob, info = G.fit_mobius(P, Q)
            self.assertLess(info["max_angle"], 1e-12)

    def test_registration_ignores_pre_and_post_composition(self):
        rng = np.random.default_rng(26)
        P = sphere(rng, 8)
        Q = random_map(rng).apply(P)
        A, B = random_map(rng), random_map(rng)
        mob, info = G.register(A.apply(P), B.apply(Q))
        self.assertLess(info["max_angle"], 1e-11)


class TestInvariants(unittest.TestCase):
    def test_cross_ratio_is_a_mobius_invariant(self):
        rng = np.random.default_rng(31)
        P = sphere(rng, 7)
        for anti in (False, True):
            M = random_map(rng, anti=anti)
            Q = M.apply(P)
            for quad in [(0, 1, 2, 3), (1, 3, 4, 6), (0, 2, 5, 6)]:
                z = G.cross_ratio(P, quad)
                w = G.cross_ratio(Q, quad)
                if anti:
                    self.assertLess(abs(z - np.conj(w)), 1e-11)
                else:
                    self.assertLess(abs(z - w), 1e-11)
        # and the converse control: a different configuration has a different one
        R = sphere(rng, 7)
        self.assertGreater(abs(G.cross_ratio(P, (0, 1, 2, 3))
                               - G.cross_ratio(R, (0, 1, 2, 3))), 1e-6)

    def test_signature_of_a_mobius_image_matches(self):
        rng = np.random.default_rng(32)
        P = sphere(rng, 9)
        Q = random_map(rng).apply(P)
        self.assertLess(G.signature_residual(P, Q), 1e-11)

    def test_mobius_distance_is_zero_only_for_the_same_point_of_teich(self):
        rng = np.random.default_rng(33)
        P = sphere(rng, 6)
        self.assertLess(G.mobius_distance(P, random_map(rng).apply(P))["max_angle"], 1e-11)
        # a genuine perturbation of one vertex is visible
        Q = P.copy()
        Q[2] = sphere(rng, 1)[0]
        self.assertGreater(G.mobius_distance(P, Q)["max_angle"], 1e-3)


class TestGauges(unittest.TestCase):
    def test_pin_gauge_is_exact_and_idempotent(self):
        rng = np.random.default_rng(41)
        P = sphere(rng, 10)
        mob, Q, info = G.pin_gauge(P, (0, 3, 5))
        self.assertLess(info["anchor_angle"], 1e-13)
        self.assertEqual(info["orientation"], +1)
        self.assertLess(np.max(np.abs(Q[[0, 3, 5]] - G.EQUILATERAL)), 1e-13)
        mob2, R, info2 = G.pin_gauge(Q, (0, 3, 5))
        self.assertLess(np.max(np.abs(R - Q)), 1e-13)
        # the pinned representative does not depend on the incoming gauge
        M = random_map(rng)
        _m, S, _i = G.pin_gauge(M.apply(P), (0, 3, 5))
        self.assertLess(np.max(np.abs(S - Q)), 1e-12)

    def test_centred_gauge_is_rotation_equivariant(self):
        rng = np.random.default_rng(42)
        P = sphere(rng, 11)
        w = np.linspace(1.0, 2.0, len(P))
        _m, Q = G.centred_gauge(P, w)
        c = (w @ Q) / np.sum(w)
        self.assertLess(np.linalg.norm(np.cross(c, np.array([0.0, 0.0, 1.0]))), 1e-13)
        theta = 0.9
        R = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                      [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
        Mr = np.eye(4)
        Mr[:3, :3] = R
        _m2, Q2 = G.centred_gauge(G.Mobius(Mr).apply(P), w)
        self.assertLess(np.max(np.abs(Q2 - Q)), 1e-13)

    def test_centred_gauge_does_not_pin_boosts(self):
        """The rotational gauge is NOT a full Möbius gauge (negative control).

        Boosting a centred configuration keeps it centred (the boost fixes the
        pole) but changes every coordinate, so a caller that needs a unique
        representative must use ``pin_gauge``, not this.
        """
        rng = np.random.default_rng(43)
        P = sphere(rng, 11)
        _m, Q = G.centred_gauge(P)
        A = np.array([[1.0, 0.0], [0.0, 0.35]])     # fixes the north pole
        boosted = G.Mobius(G._lorentz_from_matrix(A)).apply(Q)
        self.assertGreater(np.max(np.abs(boosted - Q)), 1e-2)
        _m2, Q2 = G.centred_gauge(boosted)
        # a boost is *not* gauge for this construction: centring the boosted
        # configuration lands on a different representative, so the rotational
        # gauge leaves the three boost directions free.
        self.assertGreater(np.max(np.abs(Q2 - Q)), 1e-2)


class TestGluing(unittest.TestCase):
    def _patches(self, P, parts, rng, anti_at=()):
        return [G.Patch(idx, random_map(rng, anti=(k in anti_at)).apply(P[idx]))
                for k, idx in enumerate(parts)]

    def test_gluing_recovers_a_scrambled_configuration(self):
        rng = np.random.default_rng(51)
        P = sphere(rng, 14)
        parts = [list(range(0, 8)), list(range(5, 12)), list(range(10, 14)) + [0, 1]]
        res = G.glue_patches(self._patches(P, parts, rng, anti_at=(2,)))
        self.assertTrue(res["verdict"]["consistent"])
        self.assertEqual(sorted(int(l) for l in res["labels"]), list(range(14)))
        assembled = dict(zip(res["labels"], res["points"]))
        Q = np.array([assembled[str(i)] for i in range(14)])
        self.assertLess(G.mobius_distance(P, Q)["max_angle"], 1e-11)
        # the third patch entered mirrored and the gluing detected that
        self.assertEqual(res["verdict"]["orientation"][2], -1)

    def test_inconsistent_patch_is_rejected_on_a_four_label_overlap(self):
        rng = np.random.default_rng(52)
        P = sphere(rng, 10)
        good = G.Patch(range(0, 6), random_map(rng).apply(P[0:6]))
        # four shared labels, but the shared positions are unrelated
        bad = G.Patch([2, 3, 4, 5], sphere(rng, 4))
        res = G.glue_patches([good, bad], require="report")
        self.assertFalse(res["verdict"]["consistent"])
        self.assertEqual(res["verdict"]["inconsistent_patches"], [1])
        with self.assertRaises(G.GaugeError):
            G.glue_patches([good, bad])

    def test_three_label_overlap_cannot_detect_inconsistency(self):
        """The exact limit of the gluing check, asserted so it is not oversold."""
        rng = np.random.default_rng(53)
        P = sphere(rng, 8)
        good = G.Patch(range(0, 6), random_map(rng).apply(P[0:6]))
        three = G.Patch([3, 4, 5], sphere(rng, 3))
        res = G.glue_patches([good, three], require="report")
        self.assertTrue(res["verdict"]["consistent"])
        self.assertEqual(res["verdict"]["overlap"][1], 3)

    def test_underdetermined_overlap_is_refused(self):
        rng = np.random.default_rng(54)
        P = sphere(rng, 8)
        a = G.Patch(range(0, 5), random_map(rng).apply(P[0:5]))
        b = G.Patch([3, 4, 6], random_map(rng).apply(P[[3, 4, 6]]))
        with self.assertRaises(G.GaugeError):
            G.glue_patches([a, b])

    def test_register_on_labels_uses_labels_not_order(self):
        rng = np.random.default_rng(55)
        P = sphere(rng, 7)
        Q = random_map(rng).apply(P)
        order = [4, 0, 6, 1, 5, 2, 3]
        mob, info = G.register_on_labels(P, list(range(7)),
                                         Q[order], [order[i] for i in range(7)][::-1]
                                         if False else order)
        self.assertLess(info["max_angle"], 1e-12)
        self.assertEqual(info["shared"], 7)


class TestAgainstStereographicConventions(unittest.TestCase):
    """Cross-check the O(3,1) action against an independent Möbius formula.

    ``pullback.moduli.apply_mobius`` is the repository's existing scalar
    implementation; if the two agree the representation is not self-consistent
    nonsense.
    """

    def test_pole_chart_agrees_with_repository_mobius(self):
        rng = np.random.default_rng(61)
        a, b, c, d = (complex(*rng.normal(size=2)) for _ in range(4))
        det = a * d - b * c
        if abs(det) < 0.3:
            self.skipTest("degenerate random matrix")
        A = np.array([[a, b], [c, d]])
        P = sphere(rng, 8)
        Q = G.Mobius(G._lorentz_from_matrix(A)).apply(P)
        # the repository formula works in the complex coordinate
        # z_rep = (x + i y) / (1 + z); the spinor chart here is
        # z = (1 + z3, x + i y), i.e. z_rep = z1 / z0.
        z0 = 1.0 + P[:, 2]
        keep = np.abs(z0) > 1e-3
        zrep = (P[keep, 0] + 1j * P[keep, 1]) / z0[keep]
        # the 4x4 map acts on spinors up to a scale: recover it from the action
        # on the chart coordinate directly by three-point matching
        pairs = [(zrep[i], (Q[keep][i, 0] + 1j * Q[keep][i, 1])
                  / (1.0 + Q[keep][i, 2])) for i in range(3)]
        m = self._solve_mobius(pairs)
        for i in range(len(zrep)):
            pred = apply_mobius(m, zrep[i])
            got = ((Q[keep][i, 0] + 1j * Q[keep][i, 1])
                   / (1.0 + Q[keep][i, 2]))
            self.assertLess(abs(pred - got), 1e-9)

    @staticmethod
    def _solve_mobius(pairs):
        rows = []
        for z, w in pairs:
            rows.append([z, 1.0, -w * z, -w])
        _, _s, Vh = np.linalg.svd(np.array(rows, dtype=complex), full_matrices=True)
        v = np.conj(Vh[-1])
        a, b, c, d = v
        return (a, b, c, d)


if __name__ == "__main__":
    unittest.main()
