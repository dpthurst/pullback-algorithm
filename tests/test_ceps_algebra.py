"""Identity tests for :mod:`pullback.ceps_algebra`.

Every test here is an *identity check* between two independently written
computations of the same source equation, or between a source equation and the
geometry it is supposed to describe.  None of them needs a solver, and none of
them is a residual of an optimisation.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from pullback import ceps_algebra as A


# ---------------------------------------------------------------------------
# test meshes
# ---------------------------------------------------------------------------


def sphere_faces(points):
    """Oriented faces of the convex hull of points on the unit sphere."""
    from scipy.spatial import ConvexHull

    p = np.asarray(points, dtype=float)
    hull = ConvexHull(p)
    faces = []
    for tri in hull.simplices:
        a, b, c = (p[t] for t in tri)
        n = np.cross(b - a, c - a)
        if float(n @ ((a + b + c) / 3.0)) < 0.0:
            tri = (tri[0], tri[2], tri[1])
        faces.append(tuple(int(t) for t in tri))
    return tuple(faces)


def octahedron_points():
    return [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, -1), (0, 0, 1)]


def icosahedron_points():
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    pts = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            pts.append((0.0, s1 * 1.0, s2 * phi))
            pts.append((s1 * 1.0, s2 * phi, 0.0))
            pts.append((s2 * phi, 0.0, s1 * 1.0))
    return pts


def chordal_lambda(points, faces):
    """Penner coordinates ``lam = 2 log |p_i - p_j|`` of the chordal metric."""
    lam = {}
    for f in faces:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            e = (a, b) if a < b else (b, a)
            lam[e] = 2.0 * math.log(float(np.linalg.norm(
                np.asarray(points[e[0]]) - np.asarray(points[e[1]]))))
    return lam


def small_u(disk, scale=0.05, seed=0):
    rng = np.random.default_rng(seed)
    return {v: float(scale * rng.standard_normal()) for v in disk.disk_vertices()}


# ---------------------------------------------------------------------------
# 1. Penner coordinates, scaling, angles, arcs
# ---------------------------------------------------------------------------


class PennerCoordinateTests(unittest.TestCase):

    def test_penner_scaling_is_the_euclidean_conformal_scaling(self):
        # SPR Prop. 3.2 (2): lam~_ij = lam_ij + u_i + u_j, i.e. the side
        # ell_ij is multiplied by e^{(u_i+u_j)/2}.  This is a *similarity*
        # only for u = c 1 (an overall scale); a general conformal change
        # reshapes the triangle, which is the whole point.
        lam = {(0, 1): -0.3, (1, 2): 0.7, (0, 2): 0.1}
        u = {0: 0.2, 1: -0.5, 2: 0.31}
        lam_s = {e: v + u[e[0]] + u[e[1]] for e, v in lam.items()}
        for e in lam:
            self.assertAlmostEqual(A.lambda_length(lam_s[e]) / A.lambda_length(lam[e]),
                                   math.exp(0.5 * (u[e[0]] + u[e[1]])), places=13)
        # the three side ratios multiply to exp(u_0+u_1+u_2) (SPR's fiber scaling)
        ratio = math.prod(A.lambda_length(lam_s[e]) / A.lambda_length(lam[e]) for e in lam)
        self.assertAlmostEqual(ratio, math.exp(u[0] + u[1] + u[2]), places=13)
        # uniform u = c 1 is a similarity: angles are unchanged
        uc = {v: 0.37 for v in u}
        lam_c = {e: v + uc[e[0]] + uc[e[1]] for e, v in lam.items()}
        ang = A.face_angles(lam[(0, 1)], lam[(1, 2)], lam[(0, 2)])
        ang_c = A.face_angles(lam_c[(0, 1)], lam_c[(1, 2)], lam_c[(0, 2)])
        self.assertAlmostEqual(max(abs(a - b) for a, b in zip(ang, ang_c)), 0.0, places=13)

    def test_angles_match_the_law_of_cosines(self):
        rng = np.random.default_rng(3)
        checked = 0
        for _ in range(50):
            lam = rng.uniform(-1.0, 1.0, 3)
            try:
                ang = A.face_angles(*lam)
            except ValueError:
                continue
            checked += 1
            ell = [A.lambda_length(x) for x in lam]
            # ang[0] is at vertex 0, opposite side ell[1] = ell_jk
            for idx, (opp, p, q) in zip((0, 1, 2),
                                        ((ell[1], ell[0], ell[2]),
                                         (ell[2], ell[0], ell[1]),
                                         (ell[0], ell[1], ell[2]))):
                ref = math.acos((p * p + q * q - opp * opp) / (2 * p * q))
                self.assertAlmostEqual(ang[idx], ref, places=12)
            self.assertAlmostEqual(sum(ang), math.pi, places=12)
        self.assertGreater(checked, 5)

    def test_arcs_prop33(self):
        rng = np.random.default_rng(4)
        for _ in range(40):
            lam = rng.uniform(-1.0, 1.0, 3)
            arcs = A.face_arcs(*lam)
            ell = [A.lambda_length(x) for x in lam]
            # GSC (17): a_i^{jk} = ell_jk / (ell_ij ell_ki)
            self.assertAlmostEqual(arcs[0], ell[1] / (ell[0] * ell[2]), places=12)
            self.assertAlmostEqual(arcs[1], ell[2] / (ell[1] * ell[0]), places=12)
            self.assertAlmostEqual(arcs[2], ell[0] / (ell[2] * ell[1]), places=12)
            # SPR Prop. 3.3 (4): the Penner coordinate of the edge opposite
            # vertex i is -log a_j - log a_k; here that edge is lam[1] = lam_jk
            self.assertAlmostEqual(lam[1], -math.log(arcs[1]) - math.log(arcs[2]), places=12)
            self.assertAlmostEqual(lam[2], -math.log(arcs[2]) - math.log(arcs[0]), places=12)
            self.assertAlmostEqual(lam[0], -math.log(arcs[0]) - math.log(arcs[1]), places=12)
            # scaling law (GSC, below (17)): a_v -> exp(-u_v) a_v
            u = rng.uniform(-1, 1, 3)
            lam_s = (lam[0] + u[0] + u[1], lam[1] + u[1] + u[2], lam[2] + u[2] + u[0])
            arcs_s = A.face_arcs(*lam_s)
            for v in range(3):
                self.assertAlmostEqual(arcs_s[v] / arcs[v], math.exp(-u[v]), places=12)

    def test_horocycle_length_prop34(self):
        pts = octahedron_points()
        faces = sphere_faces(pts)
        disk = A.DiskTriangulation.from_faces(5, faces)
        lam = chordal_lambda(pts, faces)
        for v in disk.disk_vertices():
            total = 0.0
            for f in disk.disk_faces():
                if v in f:
                    total += disk.face_arcs_of(lam, f)[f.index(v)]
            self.assertAlmostEqual(disk.horocycle_length(lam, v), total, places=13)
            self.assertGreater(total, 0.0)

    def test_degenerate_face_raises(self):
        lam = (2.0 * math.log(2.0), 0.0, 0.0)
        with self.assertRaises(ValueError):
            A.face_angles(*lam)


# ---------------------------------------------------------------------------
# 2. the triangle function f (SPR Def. 7.1, Prop. 7.2)
# ---------------------------------------------------------------------------


class TriangleFunctionTests(unittest.TestCase):

    def test_grad_is_the_angle_vector(self):
        # SPR (25): df/dx_i = a_i, with a_i opposite the side e^{x_i}.
        for x in ((0.1, 0.2, 0.3), (-0.2, 0.35, 0.3), (0.0, 0.0, 0.0)):
            h = 1e-6
            fd = np.array([
                (A.triangle_f(np.array(x) + h * np.eye(3)[i])
                 - A.triangle_f(np.array(x) - h * np.eye(3)[i])) / (2 * h)
                for i in range(3)])
            g = A.triangle_f_grad(x)
            self.assertLess(np.abs(fd - g).max(), 1e-8)
            self.assertAlmostEqual(float(g.sum()), math.pi, places=12)

    def test_hess_is_the_cotangent_matrix(self):
        # SPR (26)
        for x in ((0.1, 0.2, 0.3), (-0.2, 0.35, 0.3)):
            h = 1e-6
            fd = np.array([
                (A.triangle_f_grad(np.array(x) + h * np.eye(3)[i])
                 - A.triangle_f_grad(np.array(x) - h * np.eye(3)[i])) / (2 * h)
                for i in range(3)]).T
            self.assertLess(np.abs(fd - A.triangle_f_hess(x)).max(), 1e-8)
            ev = np.linalg.eigvalsh(A.triangle_f_hess(x))
            self.assertGreaterEqual(ev[0], -1e-12)
            self.assertLess(abs(ev[0]), 1e-10)
            self.assertTrue(np.allclose(A.triangle_f_hess(x) @ np.ones(3), 0.0, atol=1e-10))

    def test_scaling_relation_24(self):
        x = np.array([0.1, 0.2, 0.3])
        for h in (-1.3, 0.0, 0.4, 2.1):
            self.assertAlmostEqual(A.triangle_f(x + h) - A.triangle_f(x), math.pi * h, places=12)

    def test_lobachevsky_derivative_and_values(self):
        for a in (0.2, 0.7, 1.1, 2.0):
            h = 1e-6
            fd = (A.lobachevsky(a + h) - A.lobachevsky(a - h)) / (2 * h)
            self.assertAlmostEqual(fd, -math.log(abs(2.0 * math.sin(a))), places=5)
        self.assertAlmostEqual(A.lobachevsky(0.0), 0.0, places=14)
        self.assertAlmostEqual(A.lobachevsky(0.3) + A.lobachevsky(-0.3), 0.0, places=13)
        # L(pi/2) = -int_0^{pi/2} log(2 sin t) dt = 0  (classical)
        self.assertAlmostEqual(A.lobachevsky(math.pi / 2), 0.0, places=13)
        # independent check: L(a) = -int_0^a log|2 sin t| dt
        from scipy.integrate import quad
        for a in (0.4, math.pi / 6, 1.2):
            ref, _ = quad(lambda t: -math.log(abs(2.0 * math.sin(t))), 0.0, a,
                          limit=200, epsabs=1e-13)
            self.assertAlmostEqual(A.lobachevsky(a), ref, places=10)


# ---------------------------------------------------------------------------
# 3. the ideal Delaunay condition
# ---------------------------------------------------------------------------


class DelaunayConditionTests(unittest.TestCase):

    def _quad(self, rng, concyclic=False):
        ang = np.sort(rng.uniform(0.0, 2.0 * np.pi, 4))
        rad = np.full(4, 1.0) if concyclic else rng.uniform(0.4, 1.6, 4)
        names = ["i", "k", "j", "l"]  # cyclic order i, k, j, l -> diagonal ij
        pts = {n: np.array([rad[t] * np.cos(ang[t]), rad[t] * np.sin(ang[t])])
               for t, n in enumerate(names)}
        return pts

    @staticmethod
    def _lam_and_arcs(pts, i, j, k, l):
        length = {(a, b): float(np.linalg.norm(pts[a] - pts[b]))
                  for a in pts for b in pts if a < b}
        lam = {e: 2.0 * math.log(v) for e, v in length.items()}
        t1, t2 = (i, k, j), (i, j, l)

        def arc(v, tri):
            o = [w for w in tri if w != v]
            e1 = (v, o[0]) if v < o[0] else (o[0], v)
            e2 = (v, o[1]) if v < o[1] else (o[1], v)
            e3 = (o[0], o[1]) if o[0] < o[1] else (o[1], o[0])
            return math.exp(0.5 * (lam[e3] - lam[e1] - lam[e2]))

        def angle_at(p, q, r):
            u, v = pts[p] - pts[q], pts[r] - pts[q]
            c = float(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))
            return math.acos(max(-1.0, min(1.0, c)))

        ang_opposite = angle_at(i, k, j) + angle_at(i, l, j)
        arcs_opposite = arc(k, t1) + arc(l, t2)
        arcs_endpoints = arc(i, t1) + arc(i, t2) + arc(j, t1) + arc(j, t2)
        return ang_opposite, arcs_opposite, arcs_endpoints

    def test_delaunay_arc_condition_is_the_euclidean_one(self):
        """GSC (18) -- arcs at the vertices *opposite* the edge are smaller than
        the arcs at its endpoints -- is exactly the Euclidean local Delaunay
        condition (sum of the opposite angles <= pi)."""
        rng = np.random.default_rng(11)
        checked = 0
        for _ in range(200):
            pts = self._quad(rng)
            ang, arcs_opp, arcs_end = self._lam_and_arcs(pts, "i", "j", "k", "l")
            if abs(ang - math.pi) < 1e-6:
                continue
            self.assertEqual(arcs_opp < arcs_end, ang < math.pi,
                             msg=f"angles {math.degrees(ang):.6f}, arcs {arcs_opp:.6e} "
                                 f"vs {arcs_end:.6e}")
            checked += 1
        self.assertGreater(checked, 20)

    def test_opposite_pairing_is_not_the_delaunay_condition(self):
        # the reverse pairing fails on non-concyclic quadrilaterals, so the
        # convention used in the module is pinned by a discriminating test
        rng = np.random.default_rng(12)
        disagreements = 0
        checked = 0
        for _ in range(80):
            pts = self._quad(rng)
            ang, arcs_opp, arcs_end = self._lam_and_arcs(pts, "i", "j", "k", "l")
            if abs(ang - math.pi) < 1e-3:
                continue
            checked += 1
            if (arcs_end < arcs_opp) != (ang < math.pi):
                disagreements += 1
        self.assertGreater(checked, 20)
        self.assertEqual(disagreements, checked)

    def test_peacock_limit_condition(self):
        """With ``u = 0`` at ``v_inf`` and ``+inf`` everywhere else, every spoke
        is locally Delaunay and every edge whose two endpoints are shrunk is
        not -- GSC §7.3's peacock mechanism, exactly."""
        pts = octahedron_points()
        faces = sphere_faces(pts)
        disk = A.DiskTriangulation.from_faces(5, faces)
        lam = chordal_lambda(pts, faces)
        u = {v: (0.0 if v == 5 else math.inf) for v in disk.vertices()}
        for e in disk.edges():
            margin = A.local_delaunay_margin(disk, lam, e, u)
            if 5 in e:
                self.assertGreaterEqual(margin, -1e-15, msg=f"spoke {e}")
                continue
            # For a non-spoke edge the local condition is *never* strictly
            # satisfied.  If one of the two opposite vertices is v_inf the
            # violation is strict (GSC §7.3: "an edge connecting two vertices
            # j1, j2 != i* cannot be Delaunay"); if neither is, the margin is
            # exactly zero and the edge is *nonessential* in the sense of SPR
            # Prop. 5.10, which must then be flipped in the second phase
            # ("iteratively flip all nonessential edges opposite an undecorated
            # cusp").  This tie is why the peacock needs that second phase; a
            # plain "no violating edge" loop stops on a non-peacock
            # triangulation.
            fi = disk.edge_faces()[e]
            opposite = {v for f in (disk.faces[fi[0]], disk.faces[fi[1]])
                        for v in f if v not in e}
            if 5 in opposite:
                self.assertLess(margin, -1e-15, msg=f"rim edge {e}")
            else:
                self.assertAlmostEqual(margin, 0.0, places=15,
                                       msg=f"nonessential edge {e}")


# ---------------------------------------------------------------------------
# 4. the Ptolemy flip
# ---------------------------------------------------------------------------


class PtolemyTests(unittest.TestCase):

    def test_ptolemy_is_exact_for_concyclic_quadrilaterals(self):
        rng = np.random.default_rng(21)
        for _ in range(20):
            ang = np.sort(rng.uniform(0, 2 * math.pi, 4))
            p = [np.array([math.cos(t), math.sin(t)]) for t in ang]
            L = lambda x, y: float(np.linalg.norm(p[x] - p[y]))
            lam = {(x, y): 2.0 * math.log(L(x, y))
                   for x in range(4) for y in range(4) if x < y}
            new_lam_bd = A.ptolemy_flip(lam[(0, 1)], lam[(1, 2)], lam[(2, 3)],
                                        lam[(0, 3)], lam[(0, 2)])
            self.assertAlmostEqual(new_lam_bd, lam[(1, 3)], places=12)

    def test_ptolemy_flip_is_an_involution(self):
        rng = np.random.default_rng(22)
        keys = [(0, 1), (0, 3), (1, 2), (2, 3), (0, 2), (1, 3)]
        for _ in range(20):
            lam = dict(zip(keys, rng.uniform(-1.0, 1.0, 6)))
            new_bd = A.ptolemy_flip(lam[(0, 1)], lam[(1, 2)], lam[(2, 3)],
                                    lam[(0, 3)], lam[(0, 2)])
            back_ac = A.ptolemy_flip(lam[(0, 1)], lam[(1, 2)], lam[(2, 3)],
                                     lam[(0, 3)], new_bd)
            self.assertAlmostEqual(back_ac, lam[(0, 2)], places=12)

    def test_ptolemy_commutes_with_penner_scaling(self):
        rng = np.random.default_rng(23)
        u = rng.uniform(-1, 1, 4)
        keys = [(0, 1), (0, 3), (1, 2), (2, 3), (0, 2)]
        lam = dict(zip(keys, rng.uniform(-1, 1, 5)))
        flip = A.ptolemy_flip(lam[(0, 1)], lam[(1, 2)], lam[(2, 3)],
                              lam[(0, 3)], lam[(0, 2)])
        lam_s = {e: v + u[e[0]] + u[e[1]] for e, v in lam.items()}
        flip_s = A.ptolemy_flip(lam_s[(0, 1)], lam_s[(1, 2)], lam_s[(2, 3)],
                                lam_s[(0, 3)], lam_s[(0, 2)])
        # the new diagonal is (1, 3)
        self.assertAlmostEqual(flip_s, flip + u[1] + u[3], places=12)


# ---------------------------------------------------------------------------
# 5. energy, gradient, Hessian (SPR (56), (58), (59); GSC §7.2)
# ---------------------------------------------------------------------------


class EnergyIdentityTests(unittest.TestCase):

    def _disks(self):
        out = []
        pts = octahedron_points()
        faces = sphere_faces(pts)
        out.append(("octahedron", A.DiskTriangulation.from_faces(5, faces),
                    chordal_lambda(pts, faces)))
        pts = icosahedron_points()
        faces = sphere_faces(pts)
        out.append(("icosahedron", A.DiskTriangulation.from_faces(11, faces),
                    chordal_lambda(pts, faces)))
        return out

    def test_disk_cells(self):
        for name, disk, lam in self._disks():
            self.assertEqual(disk.disk_euler(), 1, msg=name)
            self.assertGreater(len(disk.disk_faces()), 0)
            nbrs = {b if a == disk.v_inf else a for (a, b) in disk.edges()
                    if disk.v_inf in (a, b)}
            self.assertEqual(set(disk.boundary_vertices()), nbrs, msg=name)
            for v in disk.disk_vertices():
                if v in nbrs:
                    self.assertEqual(disk.deg2(v), disk.deg1(v) - 1)
                else:
                    self.assertEqual(disk.deg2(v), disk.deg1(v))

    def test_gradient_matches_differentiated_energy(self):
        """SPR (58)/GSC §7.2 vs the exact derivative of SPR (56)."""
        for name, disk, lam in self._disks():
            u = small_u(disk, 0.02, seed=1)
            g_auto = A.energy_grad(disk, lam, u)
            g_closed = A.energy_grad_closed_form(disk, lam, u)
            verts = disk.disk_vertices()
            g_fd = {}
            h = 1e-6
            for v in verts:
                up = dict(u)
                um = dict(u)
                up[v] += h
                um[v] -= h
                g_fd[v] = (A.energy(disk, lam, up) - A.energy(disk, lam, um)) / (2 * h)
            for v in verts:
                self.assertLess(abs(g_auto[v] - g_closed[v]), 1e-11, msg=f"{name} {v}")
                self.assertLess(abs(g_fd[v] - g_closed[v]), 1e-6, msg=f"{name} {v}")

    def test_gradient_boundary_interior_reading(self):
        for name, disk, lam in self._disks():
            u = small_u(disk, 0.02, seed=2)
            theta = disk.total_angle(lam, u)
            g = A.energy_grad_closed_form(disk, lam, u)
            bnd = set(disk.boundary_vertices())
            for v in disk.disk_vertices():
                ref = (math.pi if v in bnd else 2.0 * math.pi) - theta[v]
                self.assertAlmostEqual(g[v], ref, places=11, msg=f"{name} {v}")

    def test_hessian_matches_gradient_jacobian(self):
        """SPR (59) (cotangent weights) vs finite differences of the gradient."""
        for name, disk, lam in self._disks():
            u = small_u(disk, 0.02, seed=3)
            verts, H = A.energy_hess_dense(disk, lam, u)
            h = 1e-6
            fd = np.zeros_like(H)
            g0 = A.energy_grad_closed_form(disk, lam, u)
            for col, v in enumerate(verts):
                up = dict(u)
                um = dict(u)
                up[v] += h
                um[v] -= h
                gp = A.energy_grad_closed_form(disk, lam, up)
                gm = A.energy_grad_closed_form(disk, lam, um)
                fd[:, col] = [(gp[w] - gm[w]) / (2 * h) for w in verts]
            self.assertLess(np.abs(H - fd).max(), 1e-6, msg=name)
            self.assertLess(np.abs(H - H.T).max(), 1e-14)
            ev = np.linalg.eigvalsh(H)
            self.assertGreater(ev[0], -1e-9, msg=name)
            # kernel = constants (the scaling direction of SPR (60))
            self.assertTrue(np.allclose(H @ np.ones(len(verts)), 0.0, atol=1e-9))
            # gradient invariant along the scaling direction
            up = {v: u[v] + 0.01 for v in u}
            g2 = A.energy_grad_closed_form(disk, lam, up)
            for w in verts:
                self.assertAlmostEqual(g2[w], g0[w], places=10)

    def test_scaling_identity_60(self):
        for name, disk, lam in self._disks():
            u = small_u(disk, 0.02, seed=4)
            for variant in ("gsc", "springborn"):
                for h in (-0.7, 0.05, 1.3):
                    self.assertAlmostEqual(
                        A.scaling_defect(disk, lam, u, h, variant=variant), 0.0,
                        places=9, msg=f"{name} {variant} h={h}")

    def test_gsc_and_springborn_differ_by_the_horocycle_constant(self):
        for name, disk, lam in self._disks():
            u = small_u(disk, 0.02, seed=5)
            diff = (A.energy(disk, lam, u, variant="gsc")
                    - A.energy(disk, lam, u, variant="springborn"))
            ref = 2.0 * math.pi * sum(math.log(disk.horocycle_length(lam, v))
                                      for v in disk.disk_vertices())
            self.assertAlmostEqual(diff, ref, places=9, msg=name)

    def test_unbounded_direction_is_visible(self):
        """Control: without the bounds the energy is unbounded below along -1.

        ``E(u - h 1) = E(u) - 2 pi h -> -inf`` at exactly slope ``2 pi``.  A
        solver that reports convergence without the bounds has pinned a gauge,
        not found a minimiser.
        """
        pts = octahedron_points()
        faces = sphere_faces(pts)
        disk = A.DiskTriangulation.from_faces(5, faces)
        lam = chordal_lambda(pts, faces)
        u = {v: 0.0 for v in disk.disk_vertices()}
        e0 = A.energy(disk, lam, u)
        for h in (0.5, 3.0, 12.0):
            e = A.energy(disk, lam, {v: -h for v in u})
            self.assertLess(e, e0)
            self.assertAlmostEqual((e0 - e) / h, 2.0 * math.pi, places=8)


# ---------------------------------------------------------------------------
# 6. delta and the sign of the bound (SPR Def. 5.14, Prop. 5.16, Thm. 7.18)
# ---------------------------------------------------------------------------


class DeltaBoundTests(unittest.TestCase):

    def test_tetrahedron_bound_convention(self):
        """Tetrahedron: no flip is needed, so ``delta(v, v_inf) = lam_{v v_inf}``
        exactly (SPR Prop. 5.16) and ``u = delta 1`` is a bound-active KKT point
        with ``g_v = pi - pi/3 = 2 pi/3 > 0`` (SPR (58), KKT (63)).

        This is the sign test for the constraint: ``u_v = delta_v`` is exactly
        the tangency of the horocycle at ``v`` with the missing horocycle at
        ``v_inf`` (SPR Prop. 3.2 + Def. 5.14), which is the realisability
        geometry; the opposite sign would place it at distance ``2 delta``.
        """
        pts = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
        faces = sphere_faces(pts)
        disk = A.DiskTriangulation.from_faces(3, faces)
        lam = chordal_lambda(pts, faces)
        self.assertLess(max(abs(v - lam[(0, 1)]) for v in lam.values()), 1e-12)
        lam_v = lam[(0, 1)]
        delta = {v: lam_v for v in disk.disk_vertices()}
        u = dict(delta)
        g = A.energy_grad_closed_form(disk, lam, u)
        for v in disk.disk_vertices():
            self.assertAlmostEqual(g[v], 2.0 * math.pi / 3.0, places=11)
            self.assertGreater(g[v], 0.0)          # KKT (63), bound active
            self.assertTrue(A.bound_active(u[v], delta[v]))
            self.assertLessEqual(A.bound_residual(u[v], delta[v]), 1e-15)
        e0 = A.energy(disk, lam, u)
        rng = np.random.default_rng(7)
        verts = disk.disk_vertices()
        for _ in range(50):
            d = rng.uniform(0.0, 1.0, size=3)
            d = d / np.linalg.norm(d)
            for t in (1e-4, 1e-3):
                up = {v: u[v] + t * d[i] for i, v in enumerate(verts)}
                self.assertGreaterEqual(A.energy(disk, lam, up) - e0, -1e-12)
        v0 = verts[0]
        um = dict(u)
        um[v0] -= 1e-3
        self.assertLess(A.energy(disk, lam, um) - e0, -1e-6)

    def test_octahedron_peacock_spokes(self):
        """Octahedron: the spokes are the edges that survive the peacock
        condition, so delta comes from the spokes of the input triangulation."""
        pts = octahedron_points()
        faces = sphere_faces(pts)
        disk = A.DiskTriangulation.from_faces(5, faces)
        lam = chordal_lambda(pts, faces)
        u = {v: (0.0 if v == 5 else math.inf) for v in disk.vertices()}
        spokes = [e for e in disk.edges() if 5 in e]
        self.assertEqual(len(spokes), 4)
        delta = A.delta_from_peacock({e: lam[e] for e in spokes}, 5)
        self.assertEqual(set(delta), {0, 1, 2, 3})
        for e in spokes:
            self.assertTrue(A.is_locally_ideal_delaunay(disk, lam, e, u))
        for v, d in delta.items():
            self.assertTrue(A.bound_active(d, d))
            self.assertAlmostEqual(A.bound_residual(d, d), 0.0)

    def test_delta_from_peacock_rejects_non_spokes(self):
        with self.assertRaises(ValueError):
            A.delta_from_peacock({(0, 1): 0.3}, 5)


# ---------------------------------------------------------------------------
# 7. flip transport and the peacock (SPR Prop. 3.6, 5.10, 5.16)
# ---------------------------------------------------------------------------


class FlipTransportTests(unittest.TestCase):

    def test_flip_round_trip_and_decorated_invariant(self):
        """A flip transports lambda by Ptolemy; the total horocycle lengths
        ``c_v`` (SPR Prop. 3.4) are decoration data and must be invariant."""
        pts = icosahedron_points()
        faces = sphere_faces(pts)
        lam = chordal_lambda(pts, faces)

        def c_v(fs, ll, v):
            total = 0.0
            for f in fs:
                if v in f:
                    i, j, k = f
                    total += A.face_arcs(ll[(i, j) if i < j else (j, i)],
                                         ll[(j, k) if j < k else (k, j)],
                                         ll[(k, i) if k < i else (i, k)])[f.index(v)]
            return total

        def ekey(a, b):
            return (a, b) if a <= b else (b, a)

        base = {v: c_v(faces, lam, v) for v in range(12)}
        fs, ll = tuple(faces), dict(lam)
        # flip three edges of the icosahedron and check the invariant each time
        for e in [(0, 1), (3, 7), (5, 10)]:
            fh = [f for f in fs if e[0] in f and e[1] in f]
            opp = [v for f in fh for v in f if v not in e]
            a, b = opp
            new = A.ptolemy_flip(ll[ekey(*e)], ll[ekey(e[0], a)],
                                 ll[ekey(a, e[1])], ll[ekey(e[1], b)],
                                 ll[ekey(b, e[0])])
            ll.pop(ekey(*e))
            ll[ekey(a, b)] = new
            fs = A.flip_edge(fs, e)
            for v in range(12):
                self.assertAlmostEqual(c_v(fs, ll, v), base[v], places=12)

    def test_peacock_tetrahedron_and_octahedron_exact(self):
        """Exact delta values against hand Ptolemy, no solver involved.

        Tetrahedron: delta = lam = 3 log 2 (edge length 2 sqrt 2).
        Octahedron, v_inf = +z: the four rim vertices keep delta = lam = log 2;
        the flip (x,y) -> (+z,-z) creates the fifth spoke with
        lam = 2 log(2 sqrt 2) = 3 log 2 by Ptolemy with all sides sqrt 2.
        """
        pts = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
        faces = sphere_faces(pts)
        lam = chordal_lambda(pts, faces)
        delta = A.delta_certified(faces, lam, 3)
        self.assertEqual(set(delta), {0, 1, 2})
        for v, d in delta.items():
            self.assertAlmostEqual(d, 3.0 * math.log(2.0), places=12)

        pts = octahedron_points()
        faces = sphere_faces(pts)
        lam = chordal_lambda(pts, faces)
        delta = A.delta_certified(faces, lam, 5)
        self.assertEqual(set(delta), {0, 1, 2, 3, 4})
        for v in (0, 1, 2, 3):
            self.assertAlmostEqual(delta[v], math.log(2.0), places=12)
        self.assertAlmostEqual(delta[4], 3.0 * math.log(2.0), places=12)

    def test_peacock_icosahedron_is_refused_not_faked(self):
        """Negative control: on the icosahedron the local modified Delaunay rule
        does *not* determine the peacock, and ``delta_certified`` must refuse
        rather than return whichever order happened to run.

        Observed (all four deterministic orders, 5-6 flips each): delta varies
        by up to 3.74 in log units and is not constant on the 5-element orbits
        of the stabiliser of ``v_inf``, so every produced answer is wrong.
        """
        pts = icosahedron_points()
        faces = sphere_faces(pts)
        lam = chordal_lambda(pts, faces)
        with self.assertRaises(A.PeacockAmbiguity) as ctx:
            A.delta_certified(faces, lam, 11)
        self.assertIn("order dependent", str(ctx.exception))

    def test_orient_and_flip_keep_a_valid_sphere(self):
        pts = octahedron_points()
        faces = sphere_faces(pts)
        flipped = A.flip_edge(faces, (0, 1))
        disk = A.DiskTriangulation.from_faces(5, flipped)   # validates orientation
        self.assertEqual(len(disk.edges()), 12)   # flip swaps one diagonal for the other
        self.assertIn((4, 5), {tuple(sorted(e)) for e in disk.edges()})
        self.assertNotIn((0, 1), {tuple(sorted(e)) for e in disk.edges()})


class ArcSearchDeltaTests(unittest.TestCase):
    """``delta`` as the minimum over ideal arcs (SPR Def. 5.14 + Prop. 5.16)."""

    def test_tetrahedron_and_octahedron_certified_exact(self):
        pts = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
        faces = sphere_faces(pts)
        res = A.delta_arc_search(faces, chordal_lambda(pts, faces), 3, depth=2)
        self.assertTrue(res["certified"])
        self.assertEqual(set(res["delta"]), {0, 1, 2})
        for v, d in res["delta"].items():
            self.assertAlmostEqual(d, 3.0 * math.log(2.0), places=12)

        pts = octahedron_points()
        faces = sphere_faces(pts)
        res = A.delta_arc_search(faces, chordal_lambda(pts, faces), 5, depth=2)
        self.assertTrue(res["certified"])
        for v in (0, 1, 2, 3):
            self.assertAlmostEqual(res["delta"][v], math.log(2.0), places=12)
        self.assertAlmostEqual(res["delta"][4], 3.0 * math.log(2.0), places=12)

    def test_icosahedron_minimum_is_orbit_constant(self):
        """Evidence, not a certificate: on the icosahedron no complete peacock
        within six flips attains all the arc minima, so ``certified`` is False.
        The minima are nevertheless exactly constant on the orbits of the
        stabiliser of ``v_inf`` in the icosahedral group --
        {3,5,7,9,10} (the neighbours), {0,1,4,6,8} and the antipode {2} -- which
        is what any true ``delta`` must satisfy, and they are 2 log 2, 4 log 2
        and 2 log 10 respectively.  A solver must not treat them as certified.
        """
        pts = icosahedron_points()
        faces = sphere_faces(pts)
        res = A.delta_arc_search(faces, chordal_lambda(pts, faces), 11, depth=4)
        self.assertFalse(res["certified"])
        delta = res["delta"]
        self.assertEqual(set(delta), set(range(11)))
        for orbit, value in (((3, 5, 7, 9, 10), 2.0 * math.log(2.0)),
                             ((0, 1, 4, 6, 8), 4.0 * math.log(2.0)),
                             ((2,), 2.0 * math.log(10.0))):
            for v in orbit:
                self.assertAlmostEqual(delta[v], value, places=9,
                                       msg=f"vertex {v} of orbit {orbit}")


if __name__ == "__main__":
    unittest.main()
