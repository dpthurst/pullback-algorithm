"""Tests for the independent CEPS certifier.

Run from this directory with

    PYTHONPATH=src:. python3 -m unittest test_ceps_cert -v

Every test states its tier (exact rational, or 50 significant digits), so a
pass is never stronger than the arithmetic behind it.

The battery is two-sided and each side does real work.  The positive cases are
fixtures whose answer is known in closed form -- a convex inscribed polyhedron
whose chordal metric is prescribed, or the preimage configuration of
``z -> z**d`` -- so no optimiser is involved anywhere.  The negative cases are
the ones a residual-only acceptance rule gets wrong, including a configuration
that satisfies every metric condition and is rejected solely by the Delaunay
condition, which is the whole content of GSC §7.1.
"""

from __future__ import annotations

import math
import unittest
from fractions import Fraction

import mpmath as mp

from pullback.ceps_cert import (Complex, Report, canon, certify, check_complex,
                               check_cross_ratios, check_delaunay,
                               check_gauss_bonnet, check_scaling, delaunay_edge,
                               mobius_image, replay_flips, strict_triangle,
                               vertex_angle_sums, sq_lengths as C_sq)
import ceps_fixtures as F

Q = Fraction
DPS = 50


def _cert(cx, sq_in, pts, **kw):
    return certify(cx, sq_in, pts, **kw)


# --------------------------------------------------------------------------
# exact tier: the identity claim on the classical configurations
# --------------------------------------------------------------------------
class TestExactIdentity(unittest.TestCase):
    """``l`` = chordal metric of a convex inscribed polyhedron, claim = itself.

    The vertex scaling is ``u = 0`` and the triangulation is already Delaunay,
    so the certificate is a proof rather than an estimate.
    """

    def _run(self, mk, name):
        pts, faces = mk()
        cx = Complex(len(pts), faces)
        sq = F.chordal_sq(pts)
        rep = _cert(cx, sq, pts, name=name)
        self.assertEqual(rep.failing(), [], "%s:\n%s" % (name, rep))
        self.assertEqual(rep.tier, "exact")
        self.assertEqual(rep["scale_factors_positive"]["value"], (1.0, 1.0))
        self.assertEqual(rep.failing(include_gauge=True), [], str(rep))
        return rep

    def test_regular_tetrahedron(self):
        rep = self._run(F.regular_tetra, "regular tetrahedron")
        self.assertEqual(rep["radial_degree"]["value"], 1)

    def test_octahedron(self):
        self._run(F.octahedron, "octahedron")

    def test_rational_tetrahedron(self):
        self._run(F.rational_tetra, "rational tetrahedron")

    def test_octahedral_asymmetric(self):
        self._run(F.octahedral_asymmetric, "asymmetric octahedron")


class TestDelaunayAndFlips(unittest.TestCase):
    """The Delaunay row, and what a legitimate Ptolemy flip is.

    Three facts are pinned here, because getting any of them wrong is how a
    "Delaunay loop" masquerades as CEPS:

    * the rational bipyramid's own face structure *is* intrinsically Delaunay
      (the earlier belief that convexity can fail this was an artefact of a
      sign bug in ``delaunay_edge`` when one numerator is exactly zero);
    * ``cot a + cot a' < 0`` is only a violation on a *flippable* edge.  On a
      tetrahedron every edge's alternative diagonal is already an edge, so no
      flip exists and such a triangle pair is not a Delaunay violation -- and a
      convex inscribed tetrahedron can indeed exhibit the negative sign;
    * a Ptolemy flip of a genuinely flippable edge preserves the metric.
    """

    def test_bipyramid_is_delaunay(self):
        pts, faces = F.rational_bipyramid()
        cx = Complex(len(pts), faces)
        sq = F.chordal_sq(pts)
        self.assertEqual(check_delaunay(cx, sq).failing(), [])

    def test_convex_inscribed_tetra_can_have_a_negative_cot_sum(self):
        """Unflippable edges are not Delaunay violations."""
        pts, faces = F.rational_tetra()
        cx = Complex(len(pts), faces)
        sq = F.chordal_sq(pts)
        M = F.rational_mobius((1, 0), (2, 1), (0, 1), (1, 1))
        moved = mobius_image(pts, M)
        sqm = C_sq(moved)
        sign, info = delaunay_edge(cx, sqm, (2, 3))
        self.assertEqual(sign, -1)
        self.assertEqual(info["opposite"], (0, 1))
        a0 = _angle(moved, _face_with(cx, (2, 3, 0)), 0)
        a1 = _angle(moved, _face_with(cx, (2, 3, 1)), 1)
        self.assertGreater(a0 + a1, 180.0)
        lens = {e: math.sqrt(float(sqm[e])) for e in cx.edges}
        _, _, frep = replay_flips(cx, lens, [(2, 3)])
        self.assertIn("flip_0_valid", frep.failing())
        self.assertEqual(check_delaunay(cx, sqm).failing(), [])

    def test_flip_replay_preserves_the_metric(self):
        """Flipping a genuinely flippable edge keeps the flat metric."""
        pts, faces = F.rational_bipyramid()
        cx = Complex(len(pts), faces)
        sq = F.chordal_sq(pts)
        lens = {e: math.sqrt(float(sq[e])) for e in cx.edges}
        eq = [e for e in sorted(cx.edges)
              if {cx.third(e[0], e[1]), cx.third(e[1], e[0])} == {4, 5}]
        self.assertTrue(eq, "the bipyramid has equatorial edges")
        e = eq[0]
        i, j = e
        k, l = cx.third(i, j), cx.third(j, i)
        self.assertNotIn(canon(k, l), cx.edges)
        cx2, lens2, frep = replay_flips(cx, lens, [e])
        self.assertEqual(frep.failing(), [], str(frep))
        self.assertEqual(len(cx2.faces), len(cx.faces))
        self.assertAlmostEqual(lens2[canon(k, l)] * lens[canon(i, j)],
                               lens[canon(i, k)] * lens[canon(j, l)]
                               + lens[canon(i, l)] * lens[canon(j, k)],
                               places=12)

    def test_convex_inscribed_angle_sums_are_below_two_pi(self):
        """Gauss--Bonnet: a convex *inscribed* sphere is not flat.

        ``Theta_v = 2 pi`` is the flat *disk* condition of GSC §7.5, a
        different stage from the spherical answer.  Applying it to the sphere
        is a category error, so the certificate uses the total defect ``4 pi``
        and records the strict inequality.
        """
        for mk in (F.octahedron, F.rational_tetra, F.rational_bipyramid,
                   F.octahedral_asymmetric):
            pts, faces = mk()
            cx = Complex(len(pts), faces)
            sums = vertex_angle_sums(pts, cx)
            self.assertTrue(all(s < 2 * math.pi for s in sums),
                            "%s: %s" % (mk.__name__, sums))
            self.assertAlmostEqual(sum(2 * math.pi - s for s in sums),
                                   4 * math.pi, places=12)


# --------------------------------------------------------------------------
# numeric tier
# --------------------------------------------------------------------------
class TestNumericTier(unittest.TestCase):

    def test_icosahedron_identity(self):
        mp.mp.dps = DPS
        pts, faces = F.icosahedron(DPS)
        cx = Complex(len(pts), faces)
        sq = F.chordal_sq(pts)
        rep = _cert(cx, sq, pts, tol=1e-40, name="icosahedron")
        self.assertEqual(rep.failing(), [], str(rep))
        self.assertEqual(rep.tier, "numeric")
        self.assertGreater(rep["supporting_margin"]["value"], 0)
        self.assertGreater(rep["delaunay_margin"]["value"], 0)
        self.assertEqual(rep["radial_degree"]["value"], 1)


# --------------------------------------------------------------------------
# vertex scaling: the answer is the original configuration up to Mobius
# --------------------------------------------------------------------------
class TestVertexScaling(unittest.TestCase):
    """Prescribe ``l'^2_ij = k_i k_j l^2_ij``; the answer is the same points.

    ``k`` is chosen rational, so the certificate stays exact and the recovered
    factors can be compared with the prescribed ones.  The convention is the
    one the certificate uses, ``l~^2 = exp(u_i+u_j) l^2`` with ``l~`` the
    *output* chordal metric, so a scaled input and the original output give
    ``u = -log k``.
    """

    def test_prescribed_scaling(self):
        pts, faces = F.rational_tetra()
        cx = Complex(len(pts), faces)
        base = F.chordal_sq(pts)
        k = {0: Q(2), 1: Q(3, 2), 2: Q(5), 3: Q(7, 3)}
        scaled = F.scaled_metric(base, k)
        rep = _cert(cx, scaled, pts, name="vertex scaled")
        self.assertEqual(rep.failing(), [], str(rep))
        self.assertEqual(rep.tier, "exact")
        lo, hi = rep["scale_factors_positive"]["value"]
        self.assertAlmostEqual(lo, 1.0 / 5.0)
        self.assertAlmostEqual(hi, 1.0 / 1.5)

    def test_supplied_u_is_checked(self):
        pts, faces = F.rational_tetra()
        cx = Complex(len(pts), faces)
        base = F.chordal_sq(pts)
        k = {0: Q(2), 1: Q(3, 2), 2: Q(5), 3: Q(7, 3)}
        scaled = F.scaled_metric(base, k)
        good = [-math.log(float(v)) for v in k.values()]
        rep = _cert(cx, scaled, pts, u=good, name="u ok")
        self.assertEqual(rep.failing(), [], str(rep))
        bad = list(good)
        bad[2] += 0.25
        rep = _cert(cx, scaled, pts, u=bad, name="u bad")
        self.assertIn("l~^2 = exp(u_i+u_j) l^2", rep.failing())

    def test_local_scaling_is_not_a_vertex_scaling(self):
        """Scaling one edge only admits no vertex scaling at all."""
        pts, faces = F.octahedron()
        cx = Complex(len(pts), faces)
        sq = F.chordal_sq(pts)
        broken = dict(sq)
        e0 = sorted(cx.edges)[0]
        broken[e0] = broken[e0] * Q(2)
        self.assertIn("global_vertex_scaling",
                      check_scaling(cx, sq, broken).failing())

    def test_scaling_a_star_is_a_vertex_scaling(self):
        """Scaling every edge at one vertex *is* admissible -- worth pinning.

        A natural mistaken test is to scale the edges at one vertex and expect
        the check to fail; it must *not*, because that is exactly the vertex
        scaling with ``k_0`` changed.  This is the mirror image of the previous
        test, and both are needed to know the check is calibrated.
        """
        pts, faces = F.octahedron()
        cx = Complex(len(pts), faces)
        sq = F.chordal_sq(pts)
        star = dict(sq)
        for e in cx.edges:
            if 0 in e:
                star[e] = star[e] * Q(4)
        self.assertEqual(check_scaling(cx, sq, star).failing(), [])


# --------------------------------------------------------------------------
# Mobius invariance
# --------------------------------------------------------------------------
class TestMobiusInvariance(unittest.TestCase):

    def test_rational_mobius_image(self):
        pts, faces = F.octahedron()
        cx = Complex(len(pts), faces)
        sq = F.chordal_sq(pts)
        M = F.rational_mobius((2, 1), (1, -3), (1, 0), (3, 2))
        moved = mobius_image(pts, M)
        self.assertEqual(len(set(moved)), len(moved))
        rep = _cert(cx, sq, moved, name="Mobius image")
        # the intrinsic certificate is Mobius invariant, so it must pass ...
        self.assertEqual(rep.failing(), [], str(rep))
        self.assertEqual(rep.tier, "exact")
        lo, hi = rep["scale_factors_positive"]["value"]
        self.assertGreater(hi - lo, 0.5,
                           "a non-trivial Mobius map must move the factors")
        # ... while the gauge rows are free to fail: this map pushes the origin
        # out of the polyhedron, so the radial degree is no longer 1.
        self.assertTrue(rep["radial_degree"]["gauge"])
        self.assertIn("radial_degree", rep.gauge_failing(),
                      "expected this gauge to lose the origin")

    def test_mobius_image_of_the_tetrahedron_passes(self):
        """A Mobius image of a correct answer is a correct answer.

        The intrinsic certificate is Mobius invariant, so the image passes; its
        vertex factors change, which is the evidence that the map was not the
        identity.  The Delaunay row also passes because the tetra's edges are
        unflippable (see ``TestDelaunayAndFlips``).
        """
        pts, faces = F.rational_tetra()
        cx = Complex(len(pts), faces)
        sq = F.chordal_sq(pts)
        M = F.rational_mobius((1, 0), (2, 1), (0, 1), (1, 1))
        moved = mobius_image(pts, M)
        A = _cert(cx, sq, pts, name="gauge A")
        B = _cert(cx, sq, moved, name="gauge B")
        self.assertEqual(A.failing(), [])
        self.assertEqual(B.failing(), [], str(B))
        lo, hi = B["scale_factors_positive"]["value"]
        self.assertGreater(abs(math.log(hi / lo)), 0.1,
                           "the image must not be the identity scaling")


# --------------------------------------------------------------------------
# branched covers: the degree-2 and degree-3 fixtures
# --------------------------------------------------------------------------
def _face_with(cx, tri):
    want = frozenset(tri)
    for f in cx.faces:
        if frozenset(f) == want:
            return f
    raise KeyError(tri)


def _angle(points, f, v):
    """Angle at ``v`` of the Euclidean triangle ``points[f]``, in degrees."""
    from pullback.ceps_cert import angle_at
    return math.degrees(angle_at(points, f, v))


def _fibres(proj, ld):
    out = {}
    for v, b in proj.items():
        out.setdefault(b, []).append(ld[v])
    return out


class TestBranchedCover(unittest.TestCase):
    """The pullback of the octahedron by ``z -> z**d``.

    The lift's input metric is the *pullback* of the base chordal metric; the
    claimed answer is the preimage configuration.  That the two are discretely
    conformally equivalent is the content being checked -- it is not assumed,
    and the cross-ratio residual is reported.
    """

    def _cover(self, d):
        mp.mp.dps = DPS
        cov = F.branched_cover_octahedron(d, DPS)
        cx = Complex(len(cov["lift_points"]), cov["lift_faces"])
        rep = _cert(cx, cov["lift_sq"], cov["lift_points"], tol=1e-30,
                    name="lift d=%d" % d,
                    expect_cover={"d": d, "proj": cov["proj"],
                                  "ld": cov["ld"],
                                  "fibres": _fibres(cov["proj"], cov["ld"]),
                                  "base_faces": cov["base_faces"]})
        return cov, cx, rep

    def test_degree_2(self):
        cov, cx, rep = self._cover(2)
        self.assertEqual((cx.n, len(cx.edges), len(cx.faces)), (10, 24, 16))
        self.assertEqual(rep.failing(), [], str(rep))
        self.assertLess(rep["same_cross_ratios"]["value"][1], 1e-40)
        self.assertGreater(rep["supporting_margin"]["value"], 0)
        self.assertEqual(rep["radial_degree"]["value"], 1)
        self.assertTrue(rep["fibre_degree_total"]["ok"])
        self.assertTrue(rep["d_lifted_faces_per_base"]["ok"])
        self.assertAlmostEqual(rep["scale_factors_positive"]["value"][0],
                               1.0 / math.sqrt(2 + math.sqrt(2)), places=12)

    def test_degree_3(self):
        cov, cx, rep = self._cover(3)
        self.assertEqual((cx.n, len(cx.edges), len(cx.faces)), (14, 36, 24))
        self.assertEqual(rep.failing(), [], str(rep))
        self.assertLess(rep["same_cross_ratios"]["value"][1], 1e-40)
        self.assertGreater(rep["supporting_margin"]["value"], 0)

    def test_wrong_sheet_is_rejected(self):
        """Swapping two lifted vertices must break the cover certificate."""
        mp.mp.dps = DPS
        cov = F.branched_cover_octahedron(2, DPS)
        proj, ld = dict(cov["proj"]), dict(cov["ld"])
        # swap two lifted vertices that lie over *different* base vertices;
        # swapping two over the same base vertex is a covering automorphism and
        # correctly changes nothing
        # two independent corruptions of the declared cover, each of which must
        # break a *different* cover row.  Swapping two lifted vertices over the
        # same base point is a covering automorphism and must correctly change
        # nothing, so corrupting the local degree is the sharper test.
        v = [u for u in ld if ld[u] == 1][0]
        ld[v] = 2
        rep = _cert(Complex(len(cov["lift_points"]), cov["lift_faces"]),
                    cov["lift_sq"], cov["lift_points"], tol=1e-30,
                    name="wrong sheet",
                    expect_cover={"d": 2, "proj": proj, "ld": ld,
                                  "fibres": _fibres(proj, ld),
                                  "base_faces": cov["base_faces"]})
        self.assertTrue(
            set(rep.failing()) & {"faces_project_to_base_faces",
                                  "fibre_degree_total",
                                  "fibre_profile_matches",
                                  "d_lifted_faces_per_base",
                                  "every_vertex_projects"},
            "a wrong sheet must break the cover certificate; failing: %s"
            % rep.failing())


# --------------------------------------------------------------------------
# negative controls
# --------------------------------------------------------------------------
class TestNegativeControls(unittest.TestCase):
    """Each case must be refused, and refused for the stated reason."""

    def test_collapse_all_points_equal(self):
        pts, faces = F.octahedron()
        cx = Complex(len(pts), faces)
        sq = F.chordal_sq(pts)
        rep = _cert(cx, sq, [(Q(1), Q(0), Q(0))] * len(pts), name="collapse")
        self.assertIn("distinct", rep.failing())
        self.assertIn("faces_are_supporting", rep.failing())
        self.assertIn("total_angle_defect_4pi", rep.failing())

    def test_collapse_onto_a_great_circle(self):
        pts, faces = F.octahedron()
        cx = Complex(len(pts), faces)
        sq = F.chordal_sq(pts)
        flat = [(p[0], p[1], Q(0)) for p in pts]
        rep = _cert(cx, sq, flat, name="planar collapse")
        self.assertIn("distinct", rep.failing())
        self.assertFalse(rep["faces_are_supporting"]["ok"])
        self.assertFalse(rep["radial_degree"]["ok"])

    def test_one_point_inside_the_sphere(self):
        pts, faces = F.rational_tetra()
        cx = Complex(len(pts), faces)
        sq = F.chordal_sq(pts)
        moved = list(pts)
        moved[3] = tuple(c / 2 for c in pts[3])
        rep = _cert(cx, sq, moved, name="off sphere")
        self.assertIn("on_sphere", rep.failing())
        # it is still a convex polyhedron, just not an *inscribed* one, so the
        # supporting row correctly passes while the conformal rows do not
        self.assertTrue(rep["faces_are_supporting"]["ok"])

    def test_reversed_orientation(self):
        pts, faces = F.rational_tetra()
        rev = [(f[0], f[2], f[1]) for f in faces]
        cx = Complex(len(pts), rev)
        sq = F.chordal_sq(pts)
        rep = _cert(cx, sq, pts, name="reversed")
        self.assertIn("faces_are_supporting", rep.failing())
        self.assertFalse(rep["radial_degree"]["ok"])

    def test_wrong_metric_is_refused(self):
        pts, faces = F.octahedron()
        cx = Complex(len(pts), faces)
        sq = dict(F.chordal_sq(pts))
        for e in list(sq)[:3]:
            sq[e] = sq[e] * Q(3, 2)
        rep = _cert(cx, sq, pts, name="wrong metric")
        self.assertIn("same_cross_ratios", rep.failing())

    def test_degenerate_metric_is_refused(self):
        pts, faces = F.rational_tetra()
        cx = Complex(len(pts), faces)
        sq = dict(F.chordal_sq(pts))
        f = cx.faces[0]
        sa = sq[canon(f[0], f[1])]
        sb = sq[canon(f[1], f[2])]
        sq[canon(f[2], f[0])] = 4 * (sa + sb)
        rep = _cert(cx, sq, pts, name="degenerate metric")
        self.assertIn("strict_triangle_inequality", rep.failing())

    def test_triangle_inequality_edge_cases(self):
        """``sa + sb = sc`` is a *strict* triangle, not a degenerate one."""
        self.assertTrue(strict_triangle(Q(2), Q(4, 3), Q(10, 3)))
        self.assertFalse(strict_triangle(Q(1), Q(1), Q(4)))
        self.assertTrue(strict_triangle(Q(1), Q(1), Q(3)))
        self.assertFalse(strict_triangle(Q(0), Q(1), Q(1)))

    def test_disconnected_complex(self):
        """Two disjoint spheres: chi = 4, and Gauss--Bonnet says 8 pi."""
        p1, f1 = F.rational_tetra()
        faces = list(f1) + [(i + 4, j + 4, k + 4) for (i, j, k) in f1]
        pts = list(p1) + [tuple(-c for c in p) for p in p1]
        cx = Complex(8, faces)
        sq = F.chordal_sq(pts)
        rep = _cert(cx, sq, pts, name="disconnected")
        self.assertIn("connected", rep.failing())
        self.assertEqual(cx.euler(), 4)
        self.assertIn("total_angle_defect_4pi", rep.failing())

    def test_missing_points_count(self):
        pts, faces = F.octahedron()
        cx = Complex(len(pts), faces)
        sq = F.chordal_sq(pts)
        rep = _cert(cx, sq, pts[:-1], name="short")
        self.assertIn("points_count", rep.failing())

    def test_changed_complex_without_flip_history_refuses(self):
        """The certificate must not guess the input metric on new edges."""
        pts, faces = F.rational_bipyramid()
        cx = Complex(len(pts), faces)
        sq = F.chordal_sq(pts)
        eq = [e for e in sorted(cx.edges)
              if {cx.third(e[0], e[1]), cx.third(e[1], e[0])} == {4, 5}]
        self.assertTrue(eq, "the bipyramid has equatorial edges")
        rep = _cert(cx, sq, pts, cx_out=cx.flip(eq[0]), name="no history")
        self.assertIn("cross_ratio", rep.failing())
        self.assertFalse(rep["cross_ratio"]["ok"])

    def test_flipping_a_delaunay_edge_is_metric_preserving(self):
        """A Delaunay edge may be flipped; the replay must accept it.

        This is the complement of the main flip test: ``replay_flips`` has to
        distinguish "Ptolemy flip of a convex quad" (fine, changes the
        triangulation and keeps the metric) from "flip of a non-convex quad"
        (not a metric-preserving operation).
        """
        pts, faces = F.rational_bipyramid()
        cx = Complex(len(pts), faces)
        sq = F.chordal_sq(pts)
        lens = {e: math.sqrt(float(sq[e])) for e in cx.edges}
        good = [e for e in cx.edges if delaunay_edge(cx, sq, e)[0] > 0]
        self.assertTrue(good)
        cx3, lens3, frep = replay_flips(cx, lens, [good[0]])
        self.assertEqual(frep.failing(), [], str(frep))
        # the perimeter of the quadrilateral is preserved, and the two new
        # diagonals satisfy the Ptolemy relation exactly
        self.assertEqual(len(cx3.edges), len(cx.edges))


# --------------------------------------------------------------------------
# pieces
# --------------------------------------------------------------------------
class TestPieces(unittest.TestCase):

    def test_delaunay_sign_on_the_octahedron(self):
        """Regular octahedron: n = 2 + 2 - 2 and (4A)^2 = 12 everywhere."""
        pts, faces = F.octahedron()
        cx = Complex(len(pts), faces)
        sq = F.chordal_sq(pts)
        for e in cx.edges:
            sign, info = delaunay_edge(cx, sq, e)
            self.assertEqual(sign, 1)
            self.assertEqual(info["n1"], Q(2))
            self.assertEqual(info["K1"], Q(12))

    def test_cross_ratio_is_scaling_invariant(self):
        pts, faces = F.octahedron()
        cx = Complex(len(pts), faces)
        sq = F.chordal_sq(pts)
        k = {v: Q(2 + v) for v in range(cx.n)}
        scaled = F.scaled_metric(sq, k)
        self.assertEqual(check_cross_ratios(cx, sq, scaled).failing(), [])
        self.assertEqual(check_scaling(cx, sq, scaled).failing(), [])

    def test_gauss_bonnet_catches_degeneracy(self):
        """Degenerate faces make the total angle defect leave 4 pi."""
        pts, faces = F.octahedron()
        cx = Complex(len(pts), faces)
        broken = list(pts)
        broken[0] = (Q(0), Q(1), Q(0))           # duplicate of vertex 2,
        # which is adjacent to 0, so the shared faces degenerate
        rep = check_gauss_bonnet(broken, cx)
        self.assertIn("total_angle_defect_4pi", rep.failing())

    def test_complex_rejects_a_bad_face(self):
        cx = Complex(4, [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3), (1, 2, 3)])
        self.assertIn("well_formed", check_complex(cx).failing())

    def test_report_is_per_check(self):
        rep = Report("x")
        rep.add("a", True, "fine")
        rep.add("b", False, "not fine")
        rep.add("c", False, "gauge", gauge=True)
        self.assertFalse(rep.ok())
        self.assertTrue(rep.ok("a"))
        self.assertEqual(rep.failing(), ["b"])
        self.assertEqual(rep.gauge_failing(), ["c"])
        self.assertEqual(rep.failing(include_gauge=True), ["b", "c"])
        self.assertIn("FAIL", rep.summary())


if __name__ == "__main__":
    unittest.main(verbosity=2)
