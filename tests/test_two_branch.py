"""Exact two-branch shortcut and its local monodromy/geometry boundaries."""
import math
import unittest
from unittest import mock

import numpy as np

from pullback import dense_assembly as A
from pullback import lift_ceps as L
from pullback import preimage_lift as PL
from pullback import two_branch as TB
from tests.test_preimage_lift import (OCTA, FACES, root_cover,
                                four_simple_branch_cover,
                                match_analytic_fibres)


def separated_root_cover(eps, d):
    X = OCTA.copy()
    X[4] = (math.sin(eps), 0.0, math.cos(eps))
    X[5] = (-math.sin(eps), 0.0, math.cos(eps))
    cycle = A.perm_from_cycles(d, [list(range(d))])
    local = {v: A.perm_identity(d) for v in range(len(X))}
    local[4], local[5] = cycle, A.perm_inverse(cycle)
    base = A.base_from_face_lists(FACES)
    lift = A.assemble(base, d, A.perms_from_local(base, d, local, root=0))
    lengths = {(a, b): float(np.linalg.norm(X[a] - X[b])) for a, b in base.edges}
    return base, lift, lengths, X


class TestTwoBranchShortcut(unittest.TestCase):
    def test_root_covers_match_all_exact_fibres(self):
        for d in (2, 3, 4, 5):
            with self.subTest(degree=d):
                lift, lengths = root_cover(d)
                base = A.base_from_face_lists(FACES)
                out = PL.shortcut_two_branch(lift, base, OCTA, lengths)
                self.assertEqual(out["method"], "two-branch-root")
                self.assertTrue(out["used_shortcut"])
                self.assertNotIn("res", out)  # not a fake CEPS-shaped result
                self.assertEqual(len(out["positions"]), 4 * d + 2)
                self.assertLess(out["certificate"]["projection_residual"], 1e-11)
                self.assertTrue(out["shortcut"]["info"]["compatible"])
                for p, rows in out["preimages"].items():
                    self.assertEqual(sum(ld for _u, ld, _xyz in rows), d)
                worst, _mob, _analytic = match_analytic_fibres(
                    out["positions"], lift, d)
                self.assertLess(worst, 1e-10)
                generic = PL.uniformize_cover(
                    lift, lengths, base=base, base_positions=OCTA)
                self.assertTrue(generic["used_shortcut"])
                self.assertEqual(generic["method"], "two-branch-root")

    def test_ineligible_cover_does_not_pretend_to_be_cyclic(self):
        base = A.base_from_face_lists(FACES)
        lift, lengths = four_simple_branch_cover()
        with self.assertRaisesRegex(TB.NotTwoPoint, "exactly 2"):
            PL.shortcut_two_branch(lift, base, OCTA, lengths)

    def test_close_values_return_roots_even_when_ceps_is_hard(self):
        for eps in (0.01, 0.2, 0.35, 0.5, 0.8):
            with self.subTest(eps=eps):
                base, lift, lengths, X = separated_root_cover(eps, 2)
                out = PL.shortcut_two_branch(lift, base, X, lengths)
                self.assertLess(out["certificate"]["projection_residual"], 1e-11)
                self.assertEqual(len(out["preimages"]), len(X))
                # At larger separations the inherited combinatorics can cease
                # to embed convexly. The algebraic answer is not discarded.
                self.assertIn("convex_inscribed", out["certificate"])

    def test_coincident_branch_values_are_refused(self):
        base, lift, lengths, X = separated_root_cover(0.0, 2)
        with self.assertRaisesRegex(TB.ShortcutFailure, "coincide"):
            PL.shortcut_two_branch(lift, base, X, lengths)
        fallback = PL.uniformize_cover(lift, lengths, base=base, base_positions=X)
        self.assertEqual(fallback["method"], "ceps")
        self.assertIn("coincide", fallback["shortcut_error"])

    def test_ambiguous_sampled_winding_is_refused(self):
        lift, _lengths = root_cover(3)
        base = A.base_from_face_lists(FACES)
        real = TB.edge_windings

        def ambiguous(*args, **kwargs):
            out = real(*args, **kwargs)
            edge = next(iter(out["turns"]))
            total, principal, turns, _error = out["turns"][edge]
            out["turns"][edge] = (total, principal, turns + 0.25, 0.25)
            return out

        with mock.patch.object(TB, "edge_windings", side_effect=ambiguous):
            with self.assertRaisesRegex(TB.ShortcutFailure, "not integral"):
                TB.exact_positions(base, 3, lift, OCTA)

    def test_projection_residual_is_not_a_root_labelling_oracle(self):
        lift, lengths = root_cover(3)
        base = A.base_from_face_lists(FACES)
        good = TB.exact_positions(base, 3, lift, OCTA, eps=1)
        mirrored = TB.exact_positions(base, 3, lift, OCTA, eps=-1)
        good_cert = TB.certificate(good, base, lift, OCTA)
        mirrored_cert = TB.certificate(mirrored, base, lift, OCTA)
        # Every point remains a root of the declared target, even when its
        # transported sheet label is wrong.  Orientation supplies a separate
        # diagnostic; projection residual cannot arbitrate the labelling.
        self.assertLess(good_cert["projection_residual"], 1e-11)
        self.assertLess(mirrored_cert["projection_residual"], 1e-11)
        self.assertTrue(good_cert["convex_inscribed"])
        self.assertFalse(mirrored_cert["convex_inscribed"])
        self.assertAlmostEqual(mirrored_cert["solid_angle_sum"], 2.0, places=12)
        selected = PL.shortcut_two_branch(lift, base, OCTA, lengths)
        self.assertTrue(selected["certificate"]["convex_inscribed"])

    def test_existing_lift_entry_point_accepts_the_shortcut_uniformizer(self):
        d = 3
        lift, lengths = root_cover(d)
        base = A.base_from_face_lists(FACES)
        uniformizer = TB.uniformize_from_base(base, d, lift, OCTA)
        out = L.lift_one_step(base, d, lift["edge_perms"], lengths,
                              uniformizer, gauge="mobius")
        self.assertTrue(out["verdicts"]["is_lift"], out["verdicts"])
        self.assertLess(out["verdicts"]["scaling_residual"], 1e-10)


if __name__ == "__main__":
    unittest.main()
