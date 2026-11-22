"""Near-flat and near-equatorial acceptance: the declared policy, pinned.

The two-cap carrier is chosen by one geometric measure -- the largest distance
of the punctures from their best-fit plane -- with one declared undecided band
above it.  Every branch of that policy is exercised here by an input, at both
ends of the band, so the boundary is neither hidden in a composite of unrelated
tolerances nor free to drift.
"""
import unittest

import mpmath as mp
import numpy as np

from pullback import discrete, equatorial_retraction as eq, mesh, thurston
from pullback.geometric_retraction import GeometricRetractionError, geometric_upsilon
from pullback.rational import RationalMap
from pullback.ribbons import B, F, RibbonGraph
from test_equatorial_retraction import assert_bigon_periphery
from test_hyperbolic_centre import _spinor_mobius


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def _ring(z=0.0, n=4):
    return np.array([_unit((0.6 * np.cos(2 * np.pi * k / n),
                            0.6 * np.sin(2 * np.pi * k / n), z)) for k in range(n)])


def _two_vertices():
    return RibbonGraph({0: (("arc", F),), 1: (("arc", B),)})


def _generic_source():
    return np.array([_unit((0.7, 0.4, 0.5)), _unit((0.8, 0.2, -0.2))])


def _tilted_ring(shift):
    """A ring whose last puncture is lifted out of the plane by ``shift``."""
    q = _ring().copy()
    q[3, 2] += shift
    q[3] = _unit(q[3])
    return q


def _spinor_data():
    return np.array([[1.2 + 0.35j, 0.4 - 1.1j],
                     [-0.75 + 0.2j, 0.6 + 0.15j]])


class TestNearFlatPolicy(unittest.TestCase):
    def test_the_declared_policy_decides_every_branch(self):
        # One measure, two constants, four outcomes.  The shifts below put the
        # input in each of them, and the assertion is re-derived from the
        # measured thickness, so the test states the policy rather than a
        # transscribed table of numbers.
        shifts = (0.0, 1e-12, 6.2e-10, 1.0e-9, 2.0e-9, 8.3e-9, 1e-6, 1e-5,
                  1e-3, 1e-2)
        seen = set()
        for shift in shifts:
            q = _tilted_ring(shift)
            thickness = eq.plane_thickness(q)
            points = [mesh.to_point(v) for v in q]
            try:
                out = geometric_upsilon(_two_vertices(), _generic_source(),
                                        points, steps=(64, 128))
            except GeometricRetractionError as exc:
                message = str(exc)
                if thickness <= eq.DEFAULT_TOLERANCE:
                    self.assertIn("not the two-cap one", message)
                    seen.add("equatorial-carrier-unverified")
                else:
                    self.assertLessEqual(thickness, eq.NEAR_FLAT_UNDECIDED)
                    self.assertIn("near-equatorial ambiguity", message)
                    self.assertIn("plane thickness", message)
                    seen.add("undecided")
                continue
            # The record measures the *centred* configuration, and centring is
            # a Möbius map, so it rescales plane distances; what must agree is
            # the branch, not the last digit.
            self.assertEqual(out["equatorial_route"]["plane_thickness"]
                             <= eq.DEFAULT_TOLERANCE,
                             thickness <= eq.DEFAULT_TOLERANCE)
            if thickness <= eq.DEFAULT_TOLERANCE:
                self.assertEqual(out["equatorial_route"]["route"], "equatorial")
                seen.add("equatorial")
            else:
                self.assertGreater(thickness, eq.NEAR_FLAT_UNDECIDED)
                self.assertEqual(out["equatorial_route"]["route"], "generic")
                self.assertFalse(out["equatorial"])
                seen.add("generic")
        self.assertEqual(seen, {"equatorial", "equatorial-carrier-unverified",
                                "undecided", "generic"})

    def test_the_two_boundaries_are_where_the_policy_says(self):
        # Exact equatorial data and a shift 6.2e-10 are both two-cap; a 1e-9
        # shift reaches the carrier's own coplanarity tolerance and is refused
        # by name rather than retracted through a facet chosen by rounding.
        for shift, route in ((0.0, "equatorial"), (6.2e-10, "equatorial")):
            out = geometric_upsilon(_two_vertices(), _generic_source(),
                                    [mesh.to_point(v) for v in _tilted_ring(shift)],
                                    steps=(64, 128))
            self.assertEqual(out["equatorial_route"]["route"], route)
            self.assertLessEqual(out["equatorial_route"]["plane_thickness"],
                                 eq.DEFAULT_TOLERANCE)
        with self.assertRaisesRegex(GeometricRetractionError,
                                    "concyclic to within"):
            geometric_upsilon(_two_vertices(), _generic_source(),
                              [mesh.to_point(v) for v in _tilted_ring(1e-9)],
                              steps=(64, 128))
        # Above the declared limit the generic route is used again, and it is
        # still a real retraction, not a refusal.
        out = geometric_upsilon(_two_vertices(), _generic_source(),
                                [mesh.to_point(v) for v in _tilted_ring(1e-2)],
                                steps=(64, 128))
        self.assertEqual(out["equatorial_route"]["route"], "generic")
        self.assertEqual((len(out["target"].vertices()),
                          len(out["target"].edge_list()),
                          len(out["target"].faces())), (4, 6, 4))

    def test_real_airplane_uses_two_caps_and_actual_transition_labels(self):
        centre = dict(thurston.period_three_centres())["airplane"]
        f = RationalMap([mp.mpf(1), mp.mpf(0), centre], [mp.mpf(1)])
        points = discrete.pcf_data(f)["points"]
        vectors = np.array([mesh.to_vector(point) for point in points])
        self.assertTrue(np.all(vectors[:, 1] == 0.0))
        faces = mesh.delaunay_faces(points)
        geo = eq.layout(vectors, faces)
        self.assertEqual(geo["coplanarity_residual"], 0.0)
        self.assertAlmostEqual(geo["min_angular_gap"], 1.0358959519612752,
                               places=12)
        source = np.array([_unit(geo["normal"] + 0.27 * geo["u"] + 0.08 * geo["w"]),
                           _unit(-geo["normal"] + 0.18 * geo["u"] + 0.04 * geo["w"])])
        self.assertGreater(float(source[0] @ geo["normal"] - geo["offset"]), 0.0)
        self.assertLess(float(source[1] @ geo["normal"] - geo["offset"]), 0.0)

        out = geometric_upsilon(_two_vertices(), source, points, steps=(64, 128))
        target = out["target"]
        self.assertEqual((len(target.vertices()), len(target.edge_list()),
                          len(target.faces())), (2, 4, 4))
        for edge in target.edge_list():
            self.assertEqual(set(target.edges[edge]), {0, 1})
        # The label is not merely present: it is the pair the crossing is
        # between.  Asserting membership here was the audited weakness.
        self.assertEqual(out["paths"]["arc"]["cells"], (0, 1))
        self.assertEqual(out["paths"]["arc"]["transition_labels"], [(0, 1)])
        self.assertEqual([d[0] for d in out["paths"]["arc"]["darts"]], [(0, 1)])
        self.assertEqual(out["map"].chains["arc"], out["paths"]["arc"]["darts"])
        self.assertGreater(out["paths"]["arc"]["nearest_margin"], 0.0)
        assert_bigon_periphery(self, target)

    def test_nonreal_near_equator_remains_a_named_refusal(self):
        points = [mesh.to_point(v) for v in _tilted_ring(1e-6)]
        with self.assertRaisesRegex(GeometricRetractionError,
                                    "near-equatorial ambiguity") as caught:
            geometric_upsilon(_two_vertices(), _generic_source(), points,
                              steps=(32, 64))
        self.assertIn("plane thickness", str(caught.exception))
        self.assertIn("undecided", str(caught.exception))

    def test_equator_arc_is_refused_before_and_after_mobius_transport(self):
        q = _ring()
        source = np.array([_unit((np.cos(0.25), np.sin(0.25), 0.0)),
                           _unit((np.cos(1.35), np.sin(1.35), 0.0))])
        points = [mesh.to_point(v) for v in q]
        with self.assertRaisesRegex(GeometricRetractionError, "entire source arc"):
            geometric_upsilon(_two_vertices(), source, points, steps=(32, 64))

        A = _spinor_data()
        q_A = _spinor_mobius(q, A)
        source_A = _spinor_mobius(source, A)
        with self.assertRaisesRegex(GeometricRetractionError, "entire source arc"):
            geometric_upsilon(_two_vertices(), source_A,
                              [mesh.to_point(v) for v in q_A], steps=(32, 64))

    def test_nondegenerate_carrier_and_labelled_map_are_mobius_covariant(self):
        # The target's combinatorics is a conformal invariant, so it must be
        # identical after a Möbius transport.  The *chains* are compared as a
        # measured instance at this one configuration: the declared arc is a
        # short great-circle arc, which a Möbius map does not preserve, so
        # equality here is evidence and not a covariance theorem.
        q = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
                     dtype=float) / np.sqrt(3.0)
        points = [mesh.to_point(v) for v in q]
        source = _generic_source()
        original = geometric_upsilon(_two_vertices(), source, points,
                                     steps=(64, 128))
        self.assertFalse(original["equatorial"])
        self.assertEqual(original["equatorial_route"]["route"], "generic")

        A = _spinor_data()
        moved = geometric_upsilon(_two_vertices(), _spinor_mobius(source, A),
                                  [mesh.to_point(v) for v in _spinor_mobius(q, A)],
                                  steps=(64, 128))
        self.assertEqual(original["target"].serialization(),
                         moved["target"].serialization())
        self.assertEqual(original["target_face_points"], moved["target_face_points"])
        self.assertEqual(original["map"].vertex_images, moved["map"].vertex_images)
        self.assertEqual(original["paths"]["arc"]["cells"],
                         moved["paths"]["arc"]["cells"])
        self.assertEqual(original["map"].chains, moved["map"].chains)


if __name__ == "__main__":
    unittest.main()
