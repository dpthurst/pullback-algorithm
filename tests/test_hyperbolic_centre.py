"""Independence and degeneracy checks for hyperbolic centration."""
import unittest

import numpy as np

from pullback import mesh
from pullback.geometric_retraction import (
    GeometricRetractionError, geometric_upsilon)
from pullback.hyperbolic_centre import LORENTZ, centred_coordinates
from pullback.ribbons import B, F, RibbonGraph


def _spinor_mobius(points, A):
    """The boundary action of a determinant-one matrix on null spinors."""
    A = np.asarray(A, dtype=complex)
    assert np.linalg.det(A) != 0
    A = A / np.sqrt(np.linalg.det(A))
    out = []
    for x, y, z in points:
        H = np.array([[1 + z, x - 1j * y], [x + 1j * y, 1 - z]])
        H = A @ H @ A.conj().T
        w = np.array([((H[0, 0] + H[1, 1]) / 2).real,
                      H[0, 1].real, -H[0, 1].imag,
                      ((H[0, 0] - H[1, 1]) / 2).real])
        out.append(w[1:] / w[0])
    return np.array(out)


def _spinor_lorentz(A):
    """R^(1,3) Lorentz matrix for the same null-spinor action."""
    def herm(v):
        t, x, y, z = v
        return np.array([[t + z, x - 1j * y], [x + 1j * y, t - z]])
    def unherm(H):
        return np.array([(H[0, 0].real + H[1, 1].real) / 2,
                         H[0, 1].real, -H[0, 1].imag,
                         (H[0, 0].real - H[1, 1].real) / 2])
    A = np.asarray(A, dtype=complex)
    A /= np.sqrt(np.linalg.det(A))
    return np.array([unherm(A @ herm(np.eye(4)[k]) @ A.conj().T)
                     for k in range(4)]).T


class TestHyperbolicCentre(unittest.TestCase):
    def test_regular_tetrahedron_is_already_centred(self):
        q = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
                     dtype=float) / np.sqrt(3.0)
        source = np.array([[1.0, 0.0, 0.0]])
        y_source, y_q, record = centred_coordinates(source, q)
        self.assertLess(record["boost_centre_residual"], 1e-14)
        self.assertLess(record["lorentz_residual"], 1e-14)
        self.assertLess(record["null_residual"], 1e-14)
        np.testing.assert_allclose(q.sum(axis=0), 0, atol=2e-15)
        np.testing.assert_allclose(y_q, q, atol=2e-15)
        np.testing.assert_allclose(y_source, source, atol=2e-15)
        self.assertEqual(record["construction"],
                         "Springborn zero-barycentre Lorentz placement")

    def test_sl2_change_leaves_centred_geometry_up_to_rotation(self):
        rng = np.random.RandomState(914)
        q = rng.normal(size=(4, 3))
        q /= np.linalg.norm(q, axis=1)[:, None]
        source = rng.normal(size=(7, 3))
        source /= np.linalg.norm(source, axis=1)[:, None]
        yy_source, y_q, rec1 = centred_coordinates(source, q)
        A = np.array([[1.2 + 0.35j, 0.4 - 1.1j], [-0.75 + 0.2j, 0.6 + 0.15j]])
        Aq = _spinor_mobius(q, A)
        Ax = _spinor_mobius(source, A)
        _, za_q, rec2 = centred_coordinates(Ax, Aq)
        za_source = centred_coordinates(Ax, Aq)[0]
        # Compose the Lorentz matrices directly: B2.L.B1^-1 must fix e0,
        # hence be a purely spatial orthogonal map on the centred sphere.
        M = rec2["boost"] @ _spinor_lorentz(A) @ np.linalg.inv(rec1["boost"])
        np.testing.assert_allclose(M[0], [1.0, 0.0, 0.0, 0.0], atol=1e-12)
        R = M[1:, 1:]
        self.assertLess(np.linalg.norm(R @ R.T - np.eye(3)), 1e-12)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=12)
        G1 = np.vstack([y_q, yy_source])
        G2 = np.vstack([za_q, za_source])
        self.assertLess(np.linalg.norm(G2 - G1 @ R.T), 1e-12)
        self.assertLess(rec1["boost_centre_residual"], 1e-12)
        self.assertLess(rec2["boost_centre_residual"], 1e-12)
        self.assertLess(rec1["null_residual"], 1e-12)
        self.assertLess(rec2["null_residual"], 1e-12)

    def test_exact_equator_uses_the_two_cap_carrier(self):
        q = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                      [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        points = [mesh.to_point(v) for v in q]
        source = RibbonGraph({0: (("arc", F),), 1: (("arc", B),)})
        a = np.array([0.7, 0.4, 0.5]); a /= np.linalg.norm(a)
        b = np.array([0.8, 0.2, -0.2]); b /= np.linalg.norm(b)
        out = geometric_upsilon(source, np.array([a, b]), points,
                                steps=(32, 64))
        self.assertEqual((len(out["target"].vertices()),
                          len(out["target"].edge_list()),
                          len(out["target"].faces())), (2, 4, 4))
        self.assertTrue(out["equatorial"])
        self.assertNotEqual(out["map"].chains["arc"], ())

    def test_every_positive_punctured_detail_refuses_an_arc_through_a_puncture(self):
        # A valid centred tetrahedron must not be confused with the equatorial
        # case above.  The retraction still refuses an arc through a puncture.
        q = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
                     dtype=float) / np.sqrt(3.0)
        direction = np.array([0.0, 0.0, 1.0])
        u = np.cross(q[0], direction)
        u /= np.linalg.norm(u)
        a = 0.8 * q[0] + 0.6 * u
        b = 0.8 * q[0] - 0.6 * u
        source = RibbonGraph({0: (("e", F),), 1: (("e", B),)})
        with self.assertRaisesRegex(GeometricRetractionError, "puncture 0"):
            geometric_upsilon(source, np.array([a, b]), q, steps=(17, 37))


if __name__ == "__main__":
    unittest.main()
