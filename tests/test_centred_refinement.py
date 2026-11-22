"""Contract tests for ``RefinedVE.centred_refinement``: the declared gauge only.

Nothing here runs a pullback step: no ``RefinedVE.step``, no ``marked_iteration``,
no ``comparison_map.py``, no iterated map.  The tests pin five things.

1. What each midpoint convention *is*: ``transported`` carries the input datum
   ``(punctures, midpoint)`` by one Möbius map (so base chord ratios are the
   Möbius factors of the input ones), ``recomputed`` re-applies the declared
   midpoint rule after centring.
2. What each one is *covariant* under.  The centred punctures are Möbius
   covariant up to a rotation; ``recomputed`` inherits that covariance, while
   ``transported`` is covariant on pairs only, because the arc-midpoint rule is
   not Möbius covariant.
3. That the archived seed-0 geometry is a *degenerate* control for the
   comparison of the two conventions (they agree there to 4e-16), with the
   measured reason, so a comparison on a Möbius-moved configuration is used
   instead of a vacuous one.
4. That centring presents exactly-cocircular data on a *great* circle, so the
   two-cap carrier's plane precondition is satisfiable in this chart, while it
   does not invent the two-cap branch for generic data.
5. Typed refusals, a measured (never fabricated) record, and determinism.

The three tests whose names mention ``cocircular``, ``airplane`` or ``two_cap``
are the *empirical tie* to the two-cap carrier's plane precondition: they call
``mesh.delaunay_faces`` (whose cocircular branch is a tracked hunk) and
``pullback.equatorial_retraction`` (lazily imported, so the other eight tests run
without it).  They belong to the two-cap retention unit; the rest of the file
needs only HEAD plus ``centred_refinement``.
"""
import unittest

import numpy as np

from pullback import mesh
from pullback.hyperbolic_centre import centred_coordinates
from pullback.rounding_multi import projector
from pullback.session import Session


def _spinor_mobius(points, A):
    """The boundary action of a determinant-one matrix on null spinors.

    A local copy of the helper in ``tests/test_hyperbolic_centre.py`` so this
    file depends on no other test module (both runners then agree).
    """
    A = np.asarray(A, dtype=complex)
    A = A / np.sqrt(np.linalg.det(A))
    out = []
    for x, y, z in points:
        H = np.array([[1 + z, x - 1j * y], [x + 1j * y, 1 - z]])
        H = A @ H @ A.conj().T
        w = np.array([((H[0, 0] + H[1, 1]) / 2).real, H[0, 1].real,
                      -H[0, 1].imag, ((H[0, 0] - H[1, 1]) / 2).real])
        out.append(w[1:] / w[0])
    return np.array(out)


def _rotation_matching(source, target):
    """Least-squares rotation R with ``R @ source[i] ~ target[i]``."""
    U, _s, Vt = np.linalg.svd(source.T @ target)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = Vt.T @ U.T
    return R


def _two_cap(points):
    """The carrier's plane test, imported lazily: see the module docstring."""
    from pullback.equatorial_retraction import layout, require_real_fourfold
    faces = mesh.delaunay_faces([mesh.to_point(v) for v in points])
    geo = layout(points, faces)
    certificate = require_real_fourfold(points) if geo is not None else None
    return faces, geo, certificate


def _plane(positions):
    """Best supporting plane of a concyclic set: (normal, offset, residual)."""
    Y = np.asarray(positions, dtype=float)
    _u, _s, vt = np.linalg.svd(Y - Y.mean(axis=0))
    normal = vt[-1]
    offsets = Y @ normal
    return normal, float(np.mean(offsets)), float(np.ptp(offsets))


def _ring(azimuths, z=0.0, radius=0.6):
    q = np.array([[radius * np.cos(np.radians(a)), radius * np.sin(np.radians(a)), z]
                  for a in azimuths], dtype=float)
    return q / np.linalg.norm(q, axis=1)[:, None]


def _oriented_faces(refined):
    from pullback import embedding
    return embedding.orient_faces([v for v, _labels in refined.T2["tris"]],
                                  [labels for _v, labels in refined.T2["tris"]])


def _log_ratio_residual(refined, before, after):
    """Independent recomputation of the record's ``edge_ratio_residual``."""
    P, order, _rows = projector(refined.edgesT2, refined.m + 1)
    ratios = np.array([np.log(float(np.linalg.norm(after[j] - after[i]))
                              / float(np.linalg.norm(before[j] - before[i])))
                       for _label, i, j in order])
    return float(np.linalg.norm(P @ ratios))


class TestCentredRefinementChart(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = Session.load("sessions/twisted-rabbit.json")
        cls.refined = cls.session.refined_geometry(seed=0)
        cls.m = m = cls.refined.m
        cls.raw = cls.refined.base_positions(cls.refined.points)
        # A genuine SL(2,C) change, not a rotation: the control for everything
        # that the input gauge is not allowed to decide.
        A = np.array([[1.2 + 0.35j, 0.4 - 1.1j], [-0.75 + 0.2j, 0.6 + 0.15j]])
        cls.A = A
        cls.moved = _spinor_mobius(cls.refined.points, A)
        cls.transported, cls.rec_t = cls.refined.centred_refinement(
            cls.refined.points, midpoint="transported")
        cls.recomputed, cls.rec_r = cls.refined.centred_refinement(
            cls.refined.points, midpoint="recomputed")

    # -------------------------------------------------- the two conventions
    def test_transported_is_one_mobius_map_of_the_input_datum(self):
        m = self.m
        moved_source, moved_base, _record = centred_coordinates(
            self.raw[m:], self.raw[:m])
        np.testing.assert_allclose(self.transported[:m], moved_base,
                                   atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(self.transported[m], moved_source.ravel(),
                                   atol=2e-15, rtol=0.0)
        # the chart really is centred: the ordinary barycentre is zero
        self.assertLess(float(np.linalg.norm(self.transported[:m].sum(axis=0))),
                        1e-13)
        # the same finite datum: chord ratios are the Mobius factors, so the
        # conformal residual of the comparison vanishes
        self.assertLess(self.rec_t["edge_ratio_residual"], 1e-12)
        self.assertAlmostEqual(
            self.rec_t["edge_ratio_residual"],
            _log_ratio_residual(self.refined, self.raw, self.transported), places=15)

    def test_recomputed_reapplies_the_named_rule_after_centring(self):
        m = self.m
        # identical punctures: both conventions use the punctures' own boost
        np.testing.assert_allclose(self.recomputed[:m], self.transported[:m],
                                   atol=2e-15, rtol=0.0)
        # the midpoint is the declared rule evaluated in the new gauge
        np.testing.assert_allclose(
            self.recomputed[m],
            self.refined.base_positions(self.recomputed[:m])[m],
            atol=2e-15, rtol=0.0)
        np.testing.assert_allclose(
            self.recomputed[m],
            self.refined.arc_midpoint(self.recomputed[:m]),
            atol=2e-15, rtol=0.0)
        # At this data the residual is zero too, because the two constructions
        # coincide (see the degenerate-control test); a genuinely nonzero value
        # is pinned on the Mobius-moved control.  What is pinned here is that
        # the recorded number is the measured one, not a constant.
        self.assertAlmostEqual(
            self.rec_r["edge_ratio_residual"],
            _log_ratio_residual(self.refined, self.raw, self.recomputed), places=15)
        self.assertEqual(self.rec_t["covariant_as"], "the datum (punctures, midpoint)")
        self.assertEqual(self.rec_r["covariant_as"], "a function of the punctures alone")

    def test_seed_zero_is_a_degenerate_control_with_a_measured_reason(self):
        m = self.m
        # The convention comparison is vacuous at this data: the two midpoints
        # agree far below any tolerance the tests could resolve.
        self.assertLess(float(np.linalg.norm(self.transported[m] - self.recomputed[m])),
                        1e-13)
        # Reason (measured, not asserted by fiat): the ramified punctures are
        # symmetric about the centring direction, so their images have equal
        # time coordinates and the two constructions coincide.
        direction = np.asarray(self.rec_t["centre"], dtype=float)[1:]
        direction = direction / np.linalg.norm(direction)
        a, b = self.refined.ram
        self.assertLess(abs(float(direction @ self.raw[a] - direction @ self.raw[b])),
                        1e-13)
        self.assertGreater(float(np.linalg.norm(
            self.rec_t["centre"][1:])), 1e-6,
            "the control must be a nontrivial boost, not an identity chart")

    def test_mobius_moved_control_separates_the_conventions(self):
        m = self.m
        moved_t, rec_t = self.refined.centred_refinement(self.moved, midpoint="transported")
        moved_r, rec_r = self.refined.centred_refinement(self.moved, midpoint="recomputed")
        np.testing.assert_allclose(moved_t[:m], moved_r[:m], atol=2e-15, rtol=0.0)
        self.assertGreater(float(np.linalg.norm(moved_t[m] - moved_r[m])), 1e-2)
        # transported is the same finite datum (residual ~0); recomputing the
        # midpoint is a different discretisation, and the record says so by the
        # measured, genuinely nonzero residual
        self.assertLess(rec_t["edge_ratio_residual"], 1e-12)
        self.assertGreater(rec_r["edge_ratio_residual"], 1e-2)
        self.assertAlmostEqual(
            rec_r["edge_ratio_residual"],
            _log_ratio_residual(self.refined, self.refined.base_positions(self.moved),
                                moved_r), places=15)
        # Nondegenerate control: the same-sign facet pattern survives, and no
        # facet is close to folding.
        ori = _oriented_faces(self.refined)
        det = np.array([np.linalg.det(np.array([moved_t[a], moved_t[b], moved_t[c]]))
                        for a, b, c in ori])
        self.assertTrue(bool((det > 0).all() or (det < 0).all()))
        self.assertGreater(float(np.abs(det).min()), 1e-2)

    def test_recomputed_chart_is_mobius_covariant_the_arc_midpoint_is_not(self):
        m = self.m
        moved_t, _rec = self.refined.centred_refinement(self.moved, midpoint="transported")
        moved_r, _rec2 = self.refined.centred_refinement(self.moved, midpoint="recomputed")
        R = _rotation_matching(self.transported[:m], moved_t[:m])
        # the punctures are covariant: one rotation carries the whole chart
        self.assertLess(float(np.linalg.norm(R @ R.T - np.eye(3))), 1e-12)
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=12)
        np.testing.assert_allclose(self.transported[:m] @ R.T, moved_t[:m],
                                   atol=3e-13, rtol=0.0)
        # recomputed is a function of the centred punctures, hence covariant
        # including the midpoint
        np.testing.assert_allclose(self.recomputed @ R.T, moved_r,
                                   atol=3e-13, rtol=0.0)
        # transported is covariant on pairs only: with the midpoint re-derived
        # in the moved gauge it is not the rotation image of the original
        self.assertGreater(float(np.linalg.norm(R @ self.transported[m] - moved_t[m])),
                           1e-2)
        self.assertGreater(float(np.linalg.norm(moved_t[m] - moved_r[m])), 1e-2)

    # --------------------------------------- the two-cap precondition, measured
    def test_centring_presents_exactly_cocircular_data_on_a_great_circle(self):
        m = self.m
        small = _ring([0, 50, 120, 200], z=0.5)
        _n, raw_offset, raw_residual = _plane(small)
        self.assertAlmostEqual(raw_offset, 0.64018439966448, places=12)
        self.assertLess(raw_residual, 1e-15)
        for midpoint in ("transported", "recomputed"):
            X, rec = self.refined.centred_refinement(small, midpoint=midpoint)
            # Guard the control itself: an antipodal-pair fixture (0/90/180/270)
            # has zero barycentre, so the boost is the identity and the chart
            # cannot detect a chart error (d1's trap).  A nonzero offset forces
            # a nonzero barycentre, and the boost direction is the check.
            self.assertGreater(float(np.linalg.norm(rec["centre"][1:])), 1e-6)
            _normal, offset, residual = _plane(X[:m])
            self.assertLessEqual(abs(residual), 1e-13)
            self.assertLessEqual(abs(offset), 1e-13,
                                 "centring a concyclic set must put it on a "
                                 "great circle: its barycentre lies in its plane")
            self.assertLessEqual(abs(complex(mesh.labelled_cross_ratio(X[:m])).imag),
                                 1e-13)
            faces, geo, z = _two_cap(X[:m])
            self.assertEqual((len(faces), sorted(map(len, faces))), (2, [4, 4]))
            self.assertIsNotNone(geo)
            self.assertLessEqual(abs(geo["offset"]), 1e-9)
            self.assertLessEqual(geo["coplanarity_residual"], 1e-12)
            self.assertAlmostEqual(abs(complex(z).imag), 0.0, places=12)
        # The uncentred presentation of the same four points is the small-circle
        # chart: that is the chart in which the two-cap plane test sees a small
        # circle, and it is not the centred chart.
        _raw_faces, raw_geo, _raw_z = _two_cap(small)
        self.assertIsNotNone(raw_geo)
        self.assertAlmostEqual(raw_geo["offset"], 0.64018439966448, places=12)

    def test_the_real_airplane_postcritical_set_is_great_after_centring(self):
        # Exact cocircular data from the real period-3 centre: every point is
        # real, so this is structural, not a tolerance case.
        import mpmath as mp
        from pullback import discrete, thurston
        from pullback.rational import RationalMap
        c = dict(thurston.period_three_centres())["airplane"]
        f = RationalMap([mp.mpf(1), mp.mpf(0), c], [mp.mpf(1)])
        vectors = np.array([mesh.to_vector(p) for p in discrete.pcf_data(f)["points"]])
        self.assertTrue(np.all(vectors[:, 1] == 0.0))
        normal, offset, residual = _plane(vectors)
        self.assertEqual((offset, residual), (0.0, 0.0))
        X, rec = self.refined.centred_refinement(vectors, midpoint="transported")
        # Nontrivial boost: the postcritical set is not zero-barycentre, so the
        # identity chart is excluded and this control can detect a chart error.
        self.assertGreater(float(np.linalg.norm(rec["centre"][1:])), 1e-6)
        _normal, offset, residual = _plane(X[:self.m])
        self.assertLessEqual(abs(offset), 1e-13)
        self.assertLessEqual(abs(residual), 1e-13)
        self.assertEqual(abs(complex(mesh.labelled_cross_ratio(X[:self.m])).imag), 0.0)
        faces, geo, _z = _two_cap(X[:self.m])
        self.assertEqual((len(faces), sorted(map(len, faces))), (2, [4, 4]))
        self.assertIsNotNone(geo)
        self.assertLessEqual(abs(geo["offset"]), 1e-9)

    def test_centring_does_not_invent_the_two_cap_branch(self):
        generic = np.random.RandomState(7).normal(size=(self.m, 3))
        generic = generic / np.linalg.norm(generic, axis=1)[:, None]
        _normal, _offset, residual = _plane(generic)
        self.assertGreater(residual, 1e-2)
        X, _rec = self.refined.centred_refinement(generic, midpoint="transported")
        _normal, _offset, moved_residual = _plane(X[:self.m])
        self.assertGreater(moved_residual, 1e-2,
                           "a generic configuration must stay non-cocircular")
        faces, geo, _z = _two_cap(X[:self.m])
        self.assertEqual((len(faces), sorted(map(len, faces))),
                         (4, [3, 3, 3, 3]))
        self.assertIsNone(geo)

    # ------------------------------------------------- refusals and the record
    def test_refusals_are_typed(self):
        with self.assertRaisesRegex(ValueError, "midpoint must be one of"):
            self.refined.centred_refinement(self.refined.points, midpoint="guessed")
        with self.assertRaisesRegex(ValueError, "three-dimensional punctures"):
            self.refined.centred_refinement(self.refined.points[:-1])
        broken = np.array(self.refined.points, dtype=float)
        broken[1] = 0.0
        with self.assertRaisesRegex(ValueError, "finite nonzero"):
            self.refined.centred_refinement(broken)
        a, b = self.refined.ram
        antipodal = np.array(self.refined.points, dtype=float)
        antipodal[b] = -np.asarray(self.refined.points[a], dtype=float)
        with self.assertRaisesRegex(ValueError, "antipodal"):
            self.refined.centred_refinement(antipodal)

    def test_record_has_no_unmeasured_or_duplicated_fields(self):
        expected = {
            "midpoint_convention", "midpoint_rule", "chart", "boost", "centre",
            "centred_barycentre", "boost_centre_residual", "lorentz_residual",
            "null_residual", "solver", "edge_ratio_residual", "covariant_as",
            "scope",
        }
        self.assertEqual(set(self.rec_t), expected)
        # the archived record returned this array twice under two names, which
        # invites reading one copy as an independent measurement
        self.assertNotIn("ordinary_barycentre", self.rec_t)
        self.assertLess(float(np.linalg.norm(self.rec_t["centred_barycentre"])), 1e-12)
        for key in ("boost_centre_residual", "lorentz_residual", "null_residual"):
            self.assertLess(float(self.rec_t[key]), 1e-12)
        self.assertEqual(self.rec_t["chart"],
                         "Springborn zero-barycentre Lorentz placement (up to rotation)")
        self.assertIn("chordal", self.rec_t["midpoint_rule"])
        self.assertIn("no uniformisation", self.rec_t["scope"])
        self.assertIn("iteration", self.rec_t["scope"])
        self.assertEqual(self.rec_t["midpoint_convention"], "transported")

    def test_the_chart_is_deterministic(self):
        again, rec = self.refined.centred_refinement(self.refined.points,
                                                     midpoint="transported")
        self.assertTrue(np.array_equal(again, self.transported))
        self.assertTrue(np.array_equal(rec["boost"], self.rec_t["boost"]))
        self.assertEqual(rec["edge_ratio_residual"], self.rec_t["edge_ratio_residual"])


if __name__ == "__main__":
    unittest.main()
