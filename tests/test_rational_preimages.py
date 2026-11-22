"""Analytic full-point-set preimages, kept independent of CEPS."""
import ast
import unittest
from unittest import mock
from pathlib import Path

import mpmath as mp

from pullback import rational_preimages as RP
from pullback.rational import RationalMap


class TestRationalPreimages(unittest.TestCase):
    def test_point_set_preserves_targets_and_complete_multiplicities(self):
        f = RationalMap([1, 0, -1], [1])  # z^2 - 1
        targets = [0, 1, 0, mp.inf]
        fibres = RP.full_preimages(f, targets)
        self.assertEqual([fibre.target for fibre in fibres], targets)
        for fibre in fibres:
            self.assertSumOfLocalDegrees(fibre, 2)
        self.assertEqual({(round(float(mp.re(z)), 12), degree)
                          for z, degree in fibres[0].entries},
                         {(-1.0, 1), (1.0, 1)})
        self.assertEqual(fibres[-1].entries, ((mp.inf, 2),))
        self.assertEqual(fibres[0].positions, fibres[2].positions)

    def test_partial_ramification_returns_each_local_class(self):
        f = RationalMap([1, 0, -3, 0], [1])  # z^3 - 3z
        fibre, = RP.full_preimages(f, [-2], degree_profiles={-2: [2, 1]})
        self.assertSumOfLocalDegrees(fibre, 3)
        self.assertEqual(fibre.declared_profile, (1, 2))
        rows = {(round(float(mp.re(z)), 12), degree) for z, degree in fibre.entries}
        self.assertEqual(rows, {(-2.0, 1), (1.0, 2)})
        self.assertEqual(set(fibre.local_degrees), {1, 2})
        self.assertEqual(len(fibre.positions), 2)

    def test_combinatorial_profile_mismatch_or_omission_is_an_error(self):
        f = RationalMap([1, 0, -3, 0], [1])
        with self.assertRaisesRegex(RP.ProfileMismatch, "numerical profile"):
            RP.full_preimages(f, [-2], degree_profiles={-2: [3]})
        with self.assertRaisesRegex(RP.ProfileMismatch, "no declared"):
            RP.full_preimages(f, [0], degree_profiles={-2: [2, 1]})

    def assertSumOfLocalDegrees(self, fibre, degree):
        self.assertEqual(fibre.degree, degree)
        self.assertEqual(sum(fibre.local_degrees), degree)

    def test_degree_mismatch_is_an_error_not_a_plausible_fibre(self):
        f = RationalMap([1, 0, 0], [1])
        with mock.patch.object(f, "preimage_with_degrees", return_value=[]):
            with self.assertRaisesRegex(RP.IncompleteFibre, "expected 2"):
                RP.full_preimages(f, [0])
        with self.assertRaisesRegex(ValueError, "local degrees"):
            RP.Fibre("bad", ((mp.mpc(0), 0),), 2)

    def test_no_ceps_or_local_series_dependency(self):
        tree = ast.parse(Path(RP.__file__).read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((f"pullback.{node.module}" if node.level
                              else node.module))
        self.assertEqual({name for name in imported if name.startswith("pullback")},
                         {"pullback.rational"})
        for name in ("preimage_lift", "local_puiseux", "spherical_ceps",
                     "dense_assembly", "dense_mesh"):
            self.assertFalse(any(name in item for item in imported))


if __name__ == "__main__":
    unittest.main()
