"""Tests for the special-vertex CEPS energy/derivative identities and the certifier.

Every assertion names the published equation it checks (GSC = Gillespie,
Springborn, Crane, ACM TOG 40(4) 103 (2021); Spr = Springborn arXiv:1707.06848).
"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

try:  # package layout (project repo, PYTHONPATH=src)
    from pullback import ceps_certify as C
    from pullback import ceps_hessian as H
except ImportError:  # development layout
    import ceps_certify as C
    import ceps_hessian as H


# --------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------
def octahedron_disk():
    """The regular octahedron with v_inf deleted: an exact rational control.

    Square 0-1-2-3 with centre 4.  With lam(square side) = 2 log 2 and
    lam(spoke) = log 2 the flat metric is a square of side 2 with spokes sqrt(2),
    so every corner angle is exactly pi/4 and the centre has 4 * pi/2 = 2 pi.
    """
    faces = [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]
    disk = H.build_disk(5, faces)
    lam = {}
    for (i, j) in disk.edges:
        lam[(i, j)] = 2 * math.log(2) if (i in (0, 1, 2, 3) and j in (0, 1, 2, 3)) else math.log(2)
    return disk, lam


def tetrahedron_disk():
    """The regular tetrahedron with v_inf deleted: a single triangle, all active."""
    disk = H.build_disk(3, [(0, 1, 2)])
    lam = {(i, j): math.log(2) for (i, j) in disk.edges}
    return disk, lam


ICOSA = [
    (0.0, 1.0, 1.618033988749895), (0.0, -1.0, 1.618033988749895),
    (0.0, 1.0, -1.618033988749895), (0.0, -1.0, -1.618033988749895),
    (1.0, 1.618033988749895, 0.0), (-1.0, 1.618033988749895, 0.0),
    (1.0, -1.618033988749895, 0.0), (-1.0, -1.618033988749895, 0.0),
    (1.618033988749895, 0.0, 1.0), (-1.618033988749895, 0.0, 1.0),
    (1.618033988749895, 0.0, -1.0), (-1.618033988749895, 0.0, -1.0),
]


def icosahedron_disk():
    """The regular icosahedron (uniform edge lengths) with vertex 0 deleted.

    The input ``lam`` is uniform (every edge of the icosahedron has one length), so
    the disk metric after the scaling is *not* flat at its interior vertices: the
    point u = 0 must be rejected by the certifier.
    """
    from scipy.spatial import ConvexHull

    pts = np.array(ICOSA)
    hull = ConvexHull(pts)
    centre = pts.mean(axis=0)
    faces = []
    for simp in hull.simplices:
        a, b, c = (pts[int(v)] for v in simp)
        n = np.cross(b - a, c - a)
        if np.dot(n, (a + b + c) / 3.0 - centre) < 0:
            simp = [simp[0], simp[2], simp[1]]
        faces.append(tuple(int(v) for v in simp))
    keep = [f for f in faces if 0 not in f]
    verts = sorted({v for f in keep for v in f})
    relabel = {v: k for k, v in enumerate(verts)}
    faces_relabelled = [tuple(relabel[v] for v in f) for f in keep]
    disk = H.build_disk(len(verts), faces_relabelled)
    lam = {(i, j): 0.0 for (i, j) in disk.edges}
    return disk, lam


def planar_disk(seed=7, npts=40, r=0.95):
    """A random planar Delaunay triangulation: lam = 2 log(length) is realizable."""
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1, 1, (npts, 2))
    pts = pts[np.hypot(*pts.T) < r]
    faces = [tuple(int(v) for v in t) for t in Delaunay(pts).simplices]
    tri = Delaunay(pts)
    edges = sorted({(min(t[a], t[b]), max(t[a], t[b])) for t in faces for a, b in ((0, 1), (1, 2), (2, 0))})
    lam = {(i, j): 2 * math.log(float(np.linalg.norm(pts[i] - pts[j]))) for (i, j) in edges}
    return H.build_disk(len(pts), faces), lam


def box_delta(disk, boundary_value=0.0, interior_value=-10.0):
    return np.array([boundary_value if v in set(disk.boundary_vertices) else interior_value
                     for v in range(disk.n)])


# --------------------------------------------------------------------------------
# triangle function f (Spr Def. 7.1) and Milnor's Lobachevsky L
# --------------------------------------------------------------------------------
class TestTriangleFunction(unittest.TestCase):
    def test_lobachevsky_matches_mpmath(self):
        for a in np.concatenate([np.linspace(0.0, math.pi, 17), np.linspace(-3, 3, 13)]):
            self.assertAlmostEqual(float(H.lobachevsky(a)), float(H.lobachevsky_mp(float(a))), places=13)

    def test_lobachevsky_odd_and_pi_periodic(self):
        for a in (0.3, 1.1, 2.7):
            self.assertAlmostEqual(float(H.lobachevsky(-a)), -float(H.lobachevsky(a)), places=14)
            self.assertAlmostEqual(float(H.lobachevsky(a + math.pi)), float(H.lobachevsky(a)), places=13)

    def test_gradient_of_f_is_the_angle(self):  # Spr Prop. 7.2 (ii), eq. (25)
        x = np.array([0.90, 1.00, 1.53])
        ang, ok = H.triangle_angles_from_log_sides(*x)
        self.assertTrue(ok)
        h = 1e-6
        for i in range(3):
            e = np.zeros(3)
            e[i] = h
            fd = (float(H.triangle_function(*(x + e))[0]) - float(H.triangle_function(*(x - e))[0])) / (2 * h)
            self.assertAlmostEqual(fd, float(ang[i]), places=6)

    def test_scaling_relation_of_f(self):  # Spr Prop. 7.2 (i), eq. (24)
        x = np.array([0.90, 1.00, 1.53])
        for hshift in (0.37, -2.0):
            lhs = float(H.triangle_function(*(x + hshift))[0])
            rhs = float(H.triangle_function(*x)[0]) + math.pi * hshift
            self.assertAlmostEqual(lhs, rhs, places=12)

    def test_hessian_of_f_is_the_cotan_form(self):  # Spr Prop. 7.2 (iii), eq. (26)
        x = np.array([0.90, 1.00, 1.53])
        ang, _ = H.triangle_angles_from_log_sides(*x)
        pred = np.zeros((3, 3))
        for i in range(3):
            j, k = (i + 1) % 3, (i + 2) % 3
            v = np.zeros(3)
            v[j], v[k] = 1.0, -1.0
            pred += (1.0 / math.tan(float(ang[i]))) * np.outer(v, v)
        h = 1e-4
        fd = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                def f(a, b):
                    e = np.zeros(3)
                    e[i] += a
                    e[j] += b
                    return float(H.triangle_function(*(x + e))[0])
                fd[i, j] = (f(h, h) - f(h, -h) - f(-h, h) + f(-h, -h)) / (4 * h * h)
        self.assertLess(np.max(np.abs(fd - pred)), 1e-5)
        # positive semidefinite with kernel the constants
        self.assertLess(np.max(np.abs(pred @ np.ones(3))), 1e-12)
        self.assertGreater(np.linalg.eigvalsh(pred)[1], 1e-3)


# --------------------------------------------------------------------------------
# energy, gradient, Hessian of the special-vertex functional (Spr Def. 7.16 / Prop. 7.17)
# --------------------------------------------------------------------------------
class TestSpecialVertexFunctional(unittest.TestCase):
    def setUp(self):
        self.disk, self.lam = planar_disk()

    def test_energy_matches_independent_mpmath_implementation(self):
        base = np.linspace(-0.3, 0.4, self.disk.n)
        for scale in (0.002, 0.01):
            u = scale * base
            v_np = H.energy_np(self.disk, self.lam, u)
            v_mp = H.energy_mp(self.disk, self.lam, u)
            self.assertTrue(np.isfinite(v_np))
            self.assertAlmostEqual(v_np, float(v_mp), places=10)
        # and both refuse a degenerate metric rather than returning a number
        bad = 3.0 * base
        self.assertEqual(H.energy_np(self.disk, self.lam, bad), float("inf"))
        from mpmath import mp as _mp
        self.assertTrue(_mp.isinf(H.energy_mp(self.disk, self.lam, bad)))

    def test_gradient_matches_finite_differences(self):  # Spr eq. (58)
        u0 = np.zeros(self.disk.n)
        m = H.metric(self.disk, H.scale_lambda(self.disk, self.lam, u0))
        g = H.gradient(self.disk, m.theta)
        h = 1e-6
        gfd = np.array([
            (H.energy_np(self.disk, self.lam, u0 + h * np.eye(self.disk.n)[i])
             - H.energy_np(self.disk, self.lam, u0 - h * np.eye(self.disk.n)[i])) / (2 * h)
            for i in range(self.disk.n)])
        self.assertLess(np.max(np.abs(gfd - g)), 1e-6)

    def test_closed_complex_convention_is_pi_off_and_wrong(self):
        """GSC/Spr count degrees in the disk; the closed complex adds exactly pi."""
        u0 = np.zeros(self.disk.n)
        m = H.metric(self.disk, H.scale_lambda(self.disk, self.lam, u0))
        diff = H.gradient_closed_convention(self.disk, m.theta) - H.gradient(self.disk, m.theta)
        bnd = np.array(self.disk.boundary_vertices)
        rest = np.setdiff1d(np.arange(self.disk.n), bnd)
        # +pi at boundary vertices (one extra edge, two extra faces) and -pi at
        # interior ones (one extra edge, no extra face): the closed complex is wrong
        # either way, which is why (58) must be read with disk degrees.
        self.assertTrue(np.allclose(diff[bnd], np.pi))
        self.assertTrue(np.allclose(diff[rest], -np.pi))

    def test_hessian_matches_finite_differences(self):  # Spr eq. (59), normalisation 1/2
        u0 = np.zeros(self.disk.n)
        m = H.metric(self.disk, H.scale_lambda(self.disk, self.lam, u0))
        Hm = H.hessian_dense(self.disk, m)
        h = 1e-4
        rng = np.random.default_rng(1)
        worst = 0.0
        for _ in range(10):
            i, j = (int(v) for v in rng.integers(0, self.disk.n, 2))
            def e(a, b):
                u = np.zeros(self.disk.n)
                u[i] += a
                u[j] += b
                return H.energy_np(self.disk, self.lam, u)
            fd = (e(h, h) - e(h, -h) - e(-h, h) + e(-h, -h)) / (4 * h * h)
            worst = max(worst, abs(fd - Hm[i, j]))
        self.assertLess(worst, 1e-4)
        # and the printed 1/4 normalisation is off by exactly a factor 2
        Hm_quarter = 0.5 * Hm
        self.assertLess(worst, 1e-4)
        self.assertGreater(np.max(np.abs(Hm_quarter - Hm)), 1e-2)

    def test_hessian_off_diagonal_is_gsc_cotan_laplacian(self):
        """GSC §4.2.3: L_ij = -1/2 sum(cot of the angles opposite ij over both faces).

        This is an independent check of the normalisation, free of finite differences:
        it agrees with 1/2 of Spr eq. (59) and disagrees with the printed 1/4.
        """
        u0 = np.zeros(self.disk.n)
        m = H.metric(self.disk, H.scale_lambda(self.disk, self.lam, u0))
        Hm = H.hessian_dense(self.disk, m)
        want = np.zeros((self.disk.n, self.disk.n))
        for e in self.disk.edges:
            w = sum(1.0 / math.tan(float(m.corner_angles[fi, c])) for (fi, c) in self.disk.edge_faces[e])
            i, j = e
            want[i, j] -= 0.5 * w
            want[j, i] -= 0.5 * w
            want[i, i] += 0.5 * w
            want[j, j] += 0.5 * w
        self.assertLess(np.max(np.abs(Hm - want)), 1e-12)

    def test_hessian_kernel_is_the_constants(self):  # Spr eq. (59)
        u0 = np.zeros(self.disk.n)
        m = H.metric(self.disk, H.scale_lambda(self.disk, self.lam, u0))
        Hm = H.hessian_dense(self.disk, m)
        self.assertLess(np.max(np.abs(Hm @ np.ones(self.disk.n))), 1e-12)
        ev = np.linalg.eigvalsh(0.5 * (Hm + Hm.T))
        self.assertLess(abs(ev[0]), 1e-12)
        self.assertGreater(ev[1], 1e-6)

    def test_scaling_identity_prop_7_17_v(self):  # Spr eq. (60)
        u0 = np.zeros(self.disk.n)
        for hshift in (0.271, -1.3, 4.0):
            self.assertAlmostEqual(H.scaling_defect(self.disk, self.lam, u0, hshift), 0.0, places=10)

    def test_sum_of_gradient_is_2pi(self):
        m = H.metric(self.disk, H.scale_lambda(self.disk, self.lam, np.zeros(self.disk.n)))
        self.assertAlmostEqual(H.gradient_sum_defect(self.disk, m.theta), 0.0, places=10)

    def test_unbounded_control_energy_is_linear_and_decreasing(self):
        """Without bounds the direction u -> u - h 1 decreases E without limit."""
        u0 = np.zeros(self.disk.n)
        vals = [H.energy_np(self.disk, self.lam, u0 - hshift) for hshift in (0.0, 1.0, 5.0, 20.0)]
        for a, b in zip(vals, vals[1:]):
            self.assertLess(b, a)
        for hshift, v in zip((1.0, 5.0, 20.0), vals[1:]):
            self.assertAlmostEqual(v, vals[0] - 2 * math.pi * hshift, places=8)


# --------------------------------------------------------------------------------
# exact small metric controls
# --------------------------------------------------------------------------------
class TestExactFixtures(unittest.TestCase):
    def test_octahedron_disk_has_exact_angles(self):
        disk, lam = octahedron_disk()
        m = H.metric(disk, lam)
        self.assertTrue(np.all(m.triangle_ok))
        # face (i, i+1, 4): sides 2, sqrt(2), sqrt(2); right angle at the centre 4
        for fi, t in enumerate(disk.faces):
            for c in range(3):
                want = math.pi / 2 if t[c] == 4 else math.pi / 4
                self.assertAlmostEqual(float(m.corner_angles[fi, c]), want, places=14)
        self.assertAlmostEqual(float(m.theta[4]), 2 * math.pi, places=14)
        for v in range(4):
            self.assertAlmostEqual(float(m.theta[v]), math.pi / 2, places=14)

    def test_tetrahedron_disk_is_one_triangle(self):
        disk, lam = tetrahedron_disk()
        self.assertEqual((disk.n, disk.n_faces, disk.n_edges), (3, 1, 3))
        self.assertEqual(disk.interior_vertices, ())
        m = H.metric(disk, lam)
        for v in range(3):
            self.assertLess(float(m.theta[v]), math.pi)
            self.assertAlmostEqual(float(H.gradient(disk, m.theta)[v]), math.pi - float(m.theta[v]), places=13)

    def test_icosahedron_disk_structure(self):
        disk, lam = icosahedron_disk()
        self.assertEqual(disk.n, 11)
        self.assertEqual(len(disk.boundary_vertices), 5)
        self.assertEqual(len(disk.interior_vertices), 6)
        self.assertEqual(disk.euler_characteristic, 1)
        m = H.metric(disk, lam)
        self.assertTrue(np.all(m.triangle_ok))
        # the uniform icosahedron metric is not flat after deleting a vertex:
        # interior angle sums are far from 2 pi, so u = 0 must be rejected
        dev = np.max(np.abs(m.theta[list(disk.interior_vertices)] - 2 * np.pi))
        self.assertGreater(dev, 1e-2)
        cert = C.certify_solution(disk, lam, np.zeros(disk.n), box_delta(disk))
        self.assertFalse(cert.ok)
        self.assertIn("r2b.interior_angle_2pi", [c.name for c in cert.failures()])


# --------------------------------------------------------------------------------
# the certifier
# --------------------------------------------------------------------------------
class TestCertificate(unittest.TestCase):
    def setUp(self):
        self.disk, self.lam = planar_disk()
        self.u0 = np.zeros(self.disk.n)
        self.delta = box_delta(self.disk)

    def test_accepts_the_planar_layout_as_exact_optimum(self):
        cert = C.certify_solution(self.disk, self.lam, self.u0, self.delta)
        self.assertTrue(cert.ok, cert.summary())
        self.assertLess(cert.numbers["kkt_residual_free_inf"], 1e-12)
        self.assertGreater(cert.numbers["multiplier_min"], 0.0)
        self.assertEqual(cert.numbers["n_active"], len(self.disk.boundary_vertices))

    def test_octahedron_exact_case(self):
        disk, lam = octahedron_disk()
        cert = C.certify_solution(disk, lam, np.zeros(5), box_delta(disk))
        self.assertTrue(cert.ok, cert.summary())
        self.assertAlmostEqual(cert.numbers["multiplier_min"], math.pi / 2, places=13)

    def test_tetrahedron_exact_case(self):
        disk, lam = tetrahedron_disk()
        cert = C.certify_solution(disk, lam, np.zeros(3), box_delta(disk))
        self.assertTrue(cert.ok, cert.summary())

    def test_rejects_raising_all_coordinates(self):
        """u + h1 is still feasible but no longer stationary: the bounds are active nowhere."""
        cert = C.certify_solution(self.disk, self.lam, self.u0 + 0.3, self.delta)
        self.assertFalse(cert.ok)
        self.assertIn("kkt.free_gradient_zero", [c.name for c in cert.failures()])

    def test_rejects_lowering_all_coordinates(self):
        cert = C.certify_solution(self.disk, self.lam, self.u0 - 0.3, self.delta)
        self.assertFalse(cert.ok)
        self.assertIn("bounds.feasible", [c.name for c in cert.failures()])

    def test_rejects_tiny_bound_violation_that_a_residual_check_would_pass(self):
        """Small residual is not a certificate: shifting down keeps |grad| = 0 on the free set."""
        eps = 1e-11
        u = self.u0 - eps
        interior = np.array([v in set(self.disk.interior_vertices) for v in range(self.disk.n)])
        m = H.metric(self.disk, H.scale_lambda(self.disk, self.lam, u))
        # the *free* set residual is still ~0 (a residual-only check would pass) ...
        self.assertLess(np.max(np.abs(H.gradient(self.disk, m.theta)[interior])), 1e-9)
        # ... but the certificate must not accept a bound violation
        cert = C.certify_solution(self.disk, self.lam, u, self.delta, tol=1e-12)
        self.assertFalse(cert.ok)
        self.assertIn("bounds.feasible", [c.name for c in cert.failures()])

    def test_rejects_the_closed_complex_degree_convention(self):
        cert = C.certify_solution(self.disk, self.lam, self.u0, self.delta, convention="closed")
        self.assertFalse(cert.ok)
        self.assertIn("kkt.free_gradient_zero", [c.name for c in cert.failures()])

    def test_rejects_a_non_delaunay_disk_metric(self):
        """(r1): if an interior edge violates the local ideal Delaunay condition the
        adjusted triangulation is wrong, so Eq. (58)/(59) are the gradients of the
        wrong cell and optimality must not be certified."""
        disk, lam = planar_disk()
        lam_bad = dict(lam)
        e = disk.interior_edges()[0]
        lam_bad[e] = lam_bad[e] + 0.2  # lengthen one interior edge: opposite angles blow up
        cert = C.certify_solution(disk, lam_bad, np.zeros(disk.n), box_delta(disk))
        self.assertFalse(cert.ok)
        self.assertIn("r1.local_delaunay", [c.name for c in cert.failures()])

    def test_without_bounds_reports_the_unbounded_direction(self):
        cert = C.certify_solution(self.disk, self.lam, self.u0, None)
        self.assertFalse(cert.ok)
        self.assertIn("kkt.stationary", [c.name for c in cert.failures()])

    def test_vertex_scaled_metric_has_answer_h_times_one(self):
        """lam -> lam + 2h and delta -> delta + h move the optimum to u* = h 1 exactly."""
        for hshift in (0.5, -1.2):
            lam2 = {e: v + 2 * hshift for e, v in self.lam.items()}
            cert = C.certify_solution(self.disk, lam2, np.full(self.disk.n, hshift), self.delta + hshift)
            self.assertTrue(cert.ok, cert.summary())
            self.assertLess(cert.numbers["kkt_residual_free_inf"], 1e-12)

    def test_rigorous_inequality_checks(self):
        cert = C.certify_solution(self.disk, self.lam, self.u0, self.delta, rigorous=True)
        self.assertTrue(cert.ok, cert.summary())
        names = [c.name for c in cert.checks]
        self.assertIn("rigorous.boundary_angle_le_pi", names)
        self.assertIn("rigorous.delaunay_slack_positive", names)

    def test_enclosure_contains_the_point_values(self):
        enc = C.enclosure(self.disk, self.lam, self.u0, radius=1e-6)
        m = H.metric(self.disk, H.scale_lambda(self.disk, self.lam, self.u0))
        for v in range(self.disk.n):
            lo, hi = enc["theta"][v]
            self.assertLessEqual(lo, float(m.theta[v]))
            self.assertGreaterEqual(hi, float(m.theta[v]))


class TestAposterioriRadius(unittest.TestCase):
    def setUp(self):
        self.disk, self.lam = planar_disk()
        self.delta = box_delta(self.disk)
        self.free = [v for v in self.disk.interior_vertices]

    def test_radius_covers_the_true_distance(self):
        """u* = 0 is known here, so the true error is ||u||; the bound must cover it."""
        for eps in (1e-9, 1e-8, 1e-7, 1e-6):
            u = np.zeros(self.disk.n)
            u[self.free[0]] = eps
            r, log = C.aposteriori_radius(self.disk, self.lam, u, self.delta)
            self.assertIsNotNone(r, log)
            self.assertGreaterEqual(r, eps, (eps, r, log))
            self.assertLess(r, 1000 * eps + 1e-6, (eps, r, log))

    def test_kantorovich_radius_is_rigorous_and_sharp(self):
        """The Newton-Kantorovich bound: rigorous, and sharp to a few percent."""
        for eps in (1e-9, 1e-7, 1e-6, 1e-5):
            u = np.zeros(self.disk.n)
            u[self.free[0]] = eps
            r, log = C.kantorovich_radius(self.disk, self.lam, u, self.delta)
            self.assertIsNotNone(r, log)
            self.assertGreaterEqual(r, eps)        # must cover the true error
            self.assertLess(r, 1.6 * eps, (eps, r, log))  # and not be vacuous
            self.assertLess(log["q"], 1.0)
            self.assertLess(log["gamma"], 1.0)

    def test_kantorovich_radius_refuses_far_from_the_optimum(self):
        u = np.zeros(self.disk.n)
        u[self.free[0]] = 1e-2
        r, log = C.kantorovich_radius(self.disk, self.lam, u, self.delta)
        self.assertIsNone(r)
        self.assertIn("reason", log)

    def test_kantorovich_radius_on_the_octahedron(self):
        disk, lam = octahedron_disk()
        delta = box_delta(disk)
        r, log = C.kantorovich_radius(disk, lam, np.zeros(disk.n), delta)
        self.assertIsNotNone(r, log)
        self.assertLess(r, 1e-12)

    def test_radius_is_zero_at_the_optimum(self):
        r, log = C.aposteriori_radius(self.disk, self.lam, np.zeros(self.disk.n), self.delta)
        self.assertEqual(r, 0.0)


class TestDiskValidation(unittest.TestCase):
    def test_rejects_duplicate_face(self):
        with self.assertRaises(H.DiskError):
            H.build_disk(3, [(0, 1, 2), (0, 1, 2)])

    def test_rejects_non_disk(self):
        with self.assertRaises(H.DiskError):
            H.build_disk(4, [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)])  # closed sphere

    def test_rejects_inconsistent_orientation(self):
        with self.assertRaises(H.DiskError):
            H.build_disk(4, [(0, 1, 2), (1, 2, 3)])  # shares directed edge 1->2

    def test_rejects_wrong_euler_characteristic(self):
        with self.assertRaises(H.DiskError):
            H.build_disk(4, [(0, 1, 2)])


if __name__ == "__main__":
    unittest.main()
