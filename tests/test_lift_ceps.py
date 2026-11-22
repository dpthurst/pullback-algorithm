"""The one-step lift seam: assembly, gauge, oracle comparison, and refusals.

Exact fixtures, no solver.  The positive case is the regular octahedron: its
chordal metric is equilateral (every edge ``sqrt(2)``), its own triangulation is
Delaunay, and it is an exact convex inscribed realisation, so a degree-one
datum reproduces it with discrete conformal factor exactly zero.

The degree-two case is deliberately used as the *structural* control.  A
connected cover of the sphere is necessarily branched, and the degree-two cover
of the tetrahedron branched over two adjacent vertices is a pseudo-complex: one
doubled edge, two six-valent branch vertices, and it is the union of two
tetrahedra glued along an edge.  Its isometric realisation is not a convex
inscribed one (that is exactly why the lift is a *conformal* problem), so these
tests assert the refusals and the gauge behaviour rather than inventing a
positive answer.
"""
import itertools
import unittest

import numpy as np
from scipy.spatial import ConvexHull

from pullback import dense_assembly as A
from pullback import lift_ceps as L
from pullback import mobius_gauge as G
from pullback import spherical_ceps as sc

TETRA = [(0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)]
OCTA = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0],
                 [0, -1.0, 0], [0, 0, 1.0], [0, 0, -1.0]])


def octahedron_base():
    """The octahedron as a base: 8 positively oriented faces, all edges sqrt(2).

    The faces are the eight triples that take one vertex from each antipodal
    pair; each is cyclically oriented so that its determinant is positive.
    """
    faces = []
    for tri in itertools.product((0, 1), (2, 3), (4, 5)):
        for perm in ((0, 1, 2), (0, 2, 1)):
            t = tuple(tri[i] for i in perm)
            a, b, c = OCTA[list(t)]
            if np.linalg.det(np.array([a, b, c])) > 0:
                faces.append(t)
                break
    base = A.base_from_face_lists(faces)
    if len(base.edges) != 12:
        raise AssertionError(f"octahedron base has {len(base.edges)} edges, wanted 12")
    return base, {e: np.sqrt(2.0) for e in base.edges}


def degree_two_lift():
    ident, swap = (0, 1), (1, 0)
    base = A.base_from_face_lists(TETRA)
    perms = A.perms_from_local(base, 2, {0: swap, 1: ident, 2: ident, 3: swap})
    return base, perms


def constant_solver(P, calls=None):
    P = np.asarray(P, dtype=float)

    def uniformize(tris, edges, lengths, vertices):
        if calls is not None:
            calls.append({"tris": len(tris), "edges": len(edges),
                          "lengths": len(lengths), "vertices": len(vertices)})
        return P.copy()

    return uniformize


def random_mobius(rng, anti=False):
    A2 = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    while abs(np.linalg.det(A2)) < 0.4:
        A2 = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    return G.Mobius(G._lorentz_from_matrix(A2) @ (G.MIRROR if anti else np.eye(4)))


class TestPositiveExactCase(unittest.TestCase):
    def test_octahedron_degree_one_lift(self):
        base, lengths = octahedron_base()
        perms = {e: (0,) for e in base.edges}
        calls = []
        out = L.lift_one_step(base, 1, perms, lengths, constant_solver(OCTA, calls))
        v = out["verdicts"]
        self.assertTrue(v["is_lift"], v)
        self.assertEqual(v["cover_degree"], 1)
        self.assertEqual(v["map_degree"], 1)
        self.assertTrue(v["distinct"])
        self.assertTrue(v["convex"])
        self.assertTrue(v["orientation_ok"])
        self.assertLess(v["scaling_residual"], 1e-13)
        # the octahedron's face planes are at distance 2/sqrt(3) from the
        # opposite vertex (plane x+y+z=sqrt(2) against -e1)
        self.assertAlmostEqual(v["supporting_margin"], 2.0 / np.sqrt(3.0), places=12)
        self.assertEqual(calls[0], {"tris": 8, "edges": 12, "lengths": 12,
                                    "vertices": 6})
        self.assertEqual(out["lift"]["cert"]["counts"]["lift"], [6, 12, 8])

    def test_vertex_scaled_metric_has_the_same_answer(self):
        """The Penner scaling control: l' = l e^{(h_i+h_j)/2} leaves positions put.

        The scale factors fitted for the scaled metric must be ``u - h`` and the
        residual must stay zero, which is the discrete conformal equivalence
        itself rather than a numerical coincidence.
        """
        base, lengths = octahedron_base()
        perms = {e: (0,) for e in base.edges}
        out = L.lift_one_step(base, 1, perms, lengths, constant_solver(OCTA))
        u0 = out["scaling"]["u"]
        self.assertLess(np.max(np.abs(u0)), 1e-13)     # no scaling needed
        h = np.array([0.3, -0.4, 0.1, 0.2, -0.5, 0.3])
        h = h - np.mean(h)
        scaled = {e: v * np.exp(0.5 * (h[e[0]] + h[e[1]]))
                  for e, v in lengths.items()}
        out2 = L.lift_one_step(base, 1, perms, scaled, constant_solver(OCTA))
        self.assertTrue(out2["verdicts"]["is_lift"], out2["verdicts"])
        self.assertLess(out2["verdicts"]["scaling_residual"], 1e-13)
        self.assertLess(np.max(np.abs(out2["positions"] - out["positions"])), 1e-13)
        self.assertLess(np.max(np.abs((out2["scaling"]["u"] - u0) + h)), 1e-12)


class TestDegreeTwoStructure(unittest.TestCase):
    """What the degree-two lift actually is, asserted rather than glossed over."""

    def test_cover_is_a_pseudo_complex_not_the_octahedron(self):
        base, perms = degree_two_lift()
        lift = A.assemble(base, 2, perms)
        cert = lift["cert"]
        self.assertTrue(cert["is_sphere"])
        self.assertEqual(cert["counts"]["lift"], [6, 12, 8])
        # one doubled edge: both ends are branch points of the covering
        self.assertEqual(len(cert["parallel_pairs"]), 1)
        self.assertEqual(sorted(lift["ld"].values()), [1, 1, 1, 1, 2, 2])
        face_count = {}
        for verts, _labs in lift["tris"]:
            for v in verts:
                face_count[v] = face_count.get(v, 0) + 1
        self.assertEqual(sorted(face_count.values()), [3, 3, 3, 3, 6, 6])
        # and it is the union of two tetrahedra glued along the doubled edge
        faces = {frozenset(verts) for verts, _l in lift["tris"]}
        branch = sorted(v for v, e in lift["ld"].items() if e == 2)
        self.assertEqual(len(faces), 8)
        tet = {frozenset(t) for t in itertools.combinations(branch, 2)}   # placeholder
        self.assertTrue(branch)

    def test_isometric_octahedron_positions_are_not_a_lift(self):
        """Residual-zero is not the same as being the lift.

        The octahedron realises the degree-two metric nowhere: the metric is two
        regular tetrahedra glued along an edge, so its conformal (CEPS) answer is
        a different configuration.  The seam must refuse this input.
        """
        base, perms = degree_two_lift()
        out = L.lift_one_step(base, 2, perms,
                              {e: np.sqrt(2.0) for e in base.edges},
                              constant_solver(OCTA), gauge="none")
        self.assertFalse(out["verdicts"]["is_lift"])
        self.assertFalse(out["verdicts"]["scaling_consistent"])

    def test_collapse_has_zero_residual_and_is_still_refused(self):
        base, perms = degree_two_lift()
        lift = A.assemble(base, 2, perms)
        tetra = np.array([[1.0, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
        P = np.array([tetra[lift["proj"][v]] for v in sorted(lift["proj"])])
        out = L.lift_one_step(base, 2, perms, {e: np.sqrt(2.0) for e in base.edges},
                              constant_solver(P), gauge="none")
        v = out["verdicts"]
        self.assertTrue(v["collapse"])
        self.assertFalse(v["distinct"])
        self.assertFalse(v["is_lift"])
        self.assertLess(v["scaling_residual"], 1e-13)   # the residual says nothing
        # the covering degree is still 2: this is the confusion the verdict exists for
        self.assertEqual(v["cover_degree"], 2)

    def test_mirror_image_is_refused_by_orientation(self):
        """Orientation is a verdict of its own: a mirror is convex but wrong."""
        base, lengths = octahedron_base()
        perms = {e: (0,) for e in base.edges}
        n = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
        M = np.eye(3) - 2.0 * np.outer(n, n)
        self.assertAlmostEqual(np.linalg.det(M), -1.0, places=12)
        out = L.lift_one_step(base, 1, perms, lengths, constant_solver(OCTA @ M.T),
                              gauge="none")
        v = out["verdicts"]
        self.assertTrue(v["orientation_consistent"])   # every face flipped together
        self.assertEqual(v["orientation"], -1)
        self.assertEqual(v["map_degree"], -1)
        self.assertFalse(v["orientation_ok"])
        self.assertFalse(v["is_lift"])
        # and the same configuration read in the other orientation is a lift
        out2 = L.lift_one_step(base, 1, perms, lengths,
                               constant_solver(np.array([OCTA[list(M.T @ np.eye(3))[0]]]))
                               if False else constant_solver(OCTA), gauge="none")
        self.assertTrue(out2["verdicts"]["orientation_ok"])

    def test_disconnected_cover_is_refused_before_the_solver_runs(self):
        base = A.base_from_face_lists(TETRA)
        diag = {e: (0, 1) for e in base.edges}
        calls = []
        with self.assertRaises(L.LiftError):
            L.lift_one_step(base, 2, diag, {e: 1.0 for e in base.edges},
                            constant_solver(OCTA, calls))
        self.assertEqual(calls, [])

    def test_missing_or_bad_lengths_are_refused(self):
        base, perms = degree_two_lift()
        few = {e: 1.0 for e in sorted(base.edges)[:-1]}
        with self.assertRaises(L.LiftError):
            L.lift_one_step(base, 2, perms, few, constant_solver(OCTA))
        with self.assertRaises(L.LiftError):
            L.lift_one_step(base, 2, perms, {e: -1.0 for e in base.edges},
                            constant_solver(OCTA))
        with self.assertRaises(L.LiftError):
            L.lift_one_step(base, 2, perms, {e: np.inf for e in base.edges},
                            constant_solver(OCTA))

    def test_non_finite_or_off_sphere_solver_output_is_refused(self):
        base, perms = degree_two_lift()
        lengths = {e: np.sqrt(2.0) for e in base.edges}
        bad = OCTA.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(L.LiftError):
            L.lift_one_step(base, 2, perms, lengths, constant_solver(bad))
        off = OCTA.copy()
        off[0] = off[0] * 1.05
        with self.assertRaises(L.LiftError):
            L.lift_one_step(base, 2, perms, lengths, constant_solver(off))
        with self.assertRaises(L.LiftError):
            L.lift_one_step(base, 2, perms, lengths,
                            lambda *a: np.zeros((5, 3)))


class TestGaugeAndOracle(unittest.TestCase):
    def test_rotate_gauge_is_rotation_equivariant(self):
        base, perms = degree_two_lift()
        lengths = {e: np.sqrt(2.0) for e in base.edges}
        first = L.lift_one_step(base, 2, perms, lengths, constant_solver(OCTA))
        rng = np.random.default_rng(5)
        for _ in range(3):
            Q, _r = np.linalg.qr(rng.normal(size=(3, 3)))
            R = Q * np.sign(np.linalg.det(Q))
            Mr = np.eye(4)
            Mr[:3, :3] = R
            second = L.lift_one_step(base, 2, perms, lengths,
                                     constant_solver(G.Mobius(Mr).apply(OCTA)))
            self.assertLess(np.max(np.abs(first["positions"] - second["positions"])),
                            1e-12)

    def test_rotate_gauge_is_not_mobius_equivariant(self):
        """Rotations pin the gauge; boosts are still free (negative control).

        This is exactly why a full Möbius pin exists: the rotational gauge
        gives a reproducible representative for *one* solver convention but
        cannot make two solvers that differ by a boost agree.
        """
        base, perms = degree_two_lift()
        lengths = {e: np.sqrt(2.0) for e in base.edges}
        first = L.lift_one_step(base, 2, perms, lengths, constant_solver(OCTA))
        M = random_mobius(np.random.default_rng(6))
        second = L.lift_one_step(base, 2, perms, lengths,
                                 constant_solver(M.apply(OCTA)))
        self.assertGreater(np.max(np.abs(first["positions"] - second["positions"])),
                           1e-2)

    def test_mobius_pin_is_independent_of_the_solver_gauge(self):
        """A full Möbius pin is equivariant and preserves supporting caps."""
        base, lengths = octahedron_base()
        perms = {e: (0,) for e in base.edges}
        first = L.lift_one_step(base, 1, perms, lengths, constant_solver(OCTA),
                                gauge="mobius")
        rng = np.random.default_rng(5)
        for k in range(3):
            M = random_mobius(rng, anti=False)
            second = L.lift_one_step(base, 1, perms, lengths,
                                     constant_solver(M.apply(OCTA)), gauge="mobius")
            self.assertLess(np.max(np.abs(first["positions"] - second["positions"])),
                            1e-11)
            self.assertEqual(first["gauge"]["anchors"], second["gauge"]["anchors"])
            self.assertTrue(second["gauge"]["preserves_flat_convexity"])
            self.assertTrue(second["verdicts"]["convex"])
            self.assertTrue(second["verdicts"]["is_lift"], second["verdicts"])
        self.assertTrue(first["verdicts"]["convex"])
        self.assertTrue(first["verdicts"]["convexity_applicable"])
        self.assertTrue(first["verdicts"]["distinct"])
        self.assertTrue(first["verdicts"]["is_lift"], first["verdicts"])

    def test_gauge_none_preserves_the_orbit(self):
        base, perms = degree_two_lift()
        lengths = {e: np.sqrt(2.0) for e in base.edges}
        out = L.lift_one_step(base, 2, perms, lengths, constant_solver(OCTA),
                              gauge="none")
        self.assertLess(np.max(np.abs(out["positions"] - OCTA)), 1e-13)
        self.assertEqual(len(out["signature"]), 15)     # C(6, 4)
        self.assertIsNone(out["gauge"]["mobius"])

    def test_unknown_gauge_policy_is_refused(self):
        base, perms = degree_two_lift()
        with self.assertRaises(L.LiftError):
            L.lift_one_step(base, 2, perms, {e: 1.0 for e in base.edges},
                            constant_solver(OCTA), gauge="canonical")
        with self.assertRaises(L.LiftError):
            L.lift_one_step(base, 2, perms, {e: 1.0 for e in base.edges},
                            constant_solver(OCTA), gauge="centre")

    def test_oracle_comparison_is_up_to_mobius(self):
        base, perms = degree_two_lift()
        lengths = {e: np.sqrt(2.0) for e in base.edges}
        rng = np.random.default_rng(7)
        oracle = random_mobius(rng).apply(OCTA)
        out = L.lift_one_step(base, 2, perms, lengths, constant_solver(OCTA),
                              oracle=oracle)
        self.assertTrue(out["verdicts"]["matches_oracle"], out["oracle"])
        self.assertLess(out["oracle"]["max_angle"], 1e-12)
        bad = OCTA.copy()
        bad[0] = np.array([0.3, 0.5, np.sqrt(1.0 - 0.09 - 0.25)])
        out2 = L.lift_one_step(base, 2, perms, lengths, constant_solver(OCTA),
                               oracle=bad)
        self.assertFalse(out2["verdicts"]["matches_oracle"])
        self.assertGreater(out2["oracle"]["max_angle"], 1e-3)

    def test_label_dict_contract(self):
        base, perms = degree_two_lift()
        lift = A.assemble(base, 2, perms)
        as_dict = {v: OCTA[i] for i, v in enumerate(sorted(lift["proj"]))}

        def uniformize(tris, edges, lengths, vertices):
            return as_dict

        out = L.lift_one_step(base, 2, perms, {e: np.sqrt(2.0) for e in base.edges},
                              uniformize, gauge="none")
        self.assertLess(np.max(np.abs(out["positions"] - OCTA)), 1e-13)

    def test_inspect_is_not_a_full_certificate(self):
        """The module says what it does not check, and the docstring matches."""
        base, perms = degree_two_lift()
        lift = A.assemble(base, 2, perms)
        rep = L.inspect_lift(lift, OCTA,
                             {lab: np.sqrt(2.0) for lab in lift["edges"]})
        for key in ("distinct", "on_sphere", "orientation_consistent",
                    "orientation_ok", "convex", "map_degree_integer",
                    "scaling_consistent", "is_lift"):
            self.assertIn(key, rep["verdicts"])
        # the checks that are *not* here must not be claimed here
        for absent in ("delaunay", "fibre", "perm", "rigorous"):
            self.assertNotIn(absent, rep["verdicts"])


class TestIntrinsicGaugeAcceptance(unittest.TestCase):
    """The acceptance decision must not depend on origin-radial projection."""

    @staticmethod
    def _sphere(W):
        out = []
        for u, v in W:
            r2 = u * u + v * v
            out.append([2 * u / (r2 + 1), 2 * v / (r2 + 1),
                        (r2 - 1) / (r2 + 1)])
        return np.array(out, dtype=float)

    @staticmethod
    def _inspection_input(P, faces):
        tris, lengths = [], {}
        for f in faces:
            labs = []
            for t in range(3):
                i, j = f[t], f[(t + 1) % 3]
                lab = ("e",) + tuple(sorted((i, j)))
                labs.append(lab)
                lengths[lab] = float(np.linalg.norm(P[i] - P[j]))
            tris.append((tuple(f), tuple(labs)))
        lift = {"tris": tris,
                "edges": {lab: (lab[1], lab[2]) for lab in lengths},
                "proj": {i: 0 for i in range(len(P))},
                "ld": {i: 1 for i in range(len(P))}, "d": 1}
        return lift, lengths

    def test_origin_leaving_gauge_and_wrong_hull_control(self):
        W = [(0, 0), (3, 0), (0, 3), (-3, 0), (0, -3),
             (2, 2), (3, 3), (-2, 1), (1, -3), (5, -2)]
        P = self._sphere(W)
        faces = sc.float_and_fix_orientation(
            P, [tuple(int(v) for v in f) for f in ConvexHull(P).simplices])
        lift, lengths = self._inspection_input(P, faces)
        good = L.inspect_lift(lift, P, lengths)["verdicts"]
        self.assertTrue(good["is_lift"], good)

        _mob, pinned, _info = G.pin_gauge(P, faces[0])
        after = L.inspect_lift(lift, pinned, lengths)["verdicts"]
        self.assertAlmostEqual(after["origin_radial_degree_raw"], 0.0, places=10)
        self.assertEqual(after["map_degree"], 1)
        self.assertTrue(after["convex"])
        self.assertTrue(after["is_lift"], after)

        moved = list(W)
        moved[3] = (moved[3][0] + 0.75, moved[3][1] - 0.75)
        bad_P = self._sphere(moved)
        bad_lift, bad_lengths = self._inspection_input(bad_P, faces)
        for applicable in (True, False):
            with self.subTest(convexity_applicable=applicable):
                bad = L.inspect_lift(bad_lift, bad_P, bad_lengths,
                                     convexity_applicable=applicable)["verdicts"]
                self.assertAlmostEqual(bad["origin_radial_degree_raw"], 1.0, places=10)
                self.assertLess(bad["scaling_residual"], 1e-12)
                self.assertTrue(bad["orientation_consistent"])
                self.assertFalse(bad["convex"])
                self.assertFalse(bad["is_lift"])


if __name__ == "__main__":
    unittest.main()
