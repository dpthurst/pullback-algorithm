"""Execution tests for the real-CEPS full-preimage adapter and its polish.

The analytic w^d configurations are constructed only *after* the CEPS solve.
No constant solver, rational-map seed, or frozen-configuration oracle enters
``uniformize_cover``.
"""
import itertools
import unittest

import mpmath as mp
import numpy as np

from pullback import dense_assembly as A
from pullback import lift_ceps as L
from pullback import mobius_gauge as G
from pullback import preimage_lift as PL
from pullback import spherical_ceps as sc
from pullback.rational import RationalMap


OCTA = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0],
                 [0, -1.0, 0], [0, 0, 1.0], [0, 0, -1.0]])
FACES = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
         (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]


def root_cover(d):
    """Sphere cover of w^d, from declarations through the production assembler."""
    north, south = 4, 5
    cycle = A.perm_from_cycles(d, [list(range(d))])
    local = {v: A.perm_identity(d) for v in range(len(OCTA))}
    local[north] = cycle
    local[south] = A.perm_inverse(cycle)
    base = A.base_from_face_lists(FACES)
    perms = A.perms_from_local(base, d, local, root=0)
    lift = A.assemble(base, d, perms)
    lengths = {(a, b): float(np.linalg.norm(OCTA[a] - OCTA[b]))
               for a, b in base.edges}
    return lift, lengths


def subdivided_root_cover(d):
    """Once-subdivided octahedron control, used for special-vertex retries."""
    points = [p.copy() for p in OCTA]
    midpoint = {}

    def mid(a, b):
        key = tuple(sorted((a, b)))
        if key not in midpoint:
            q = OCTA[a] + OCTA[b]
            midpoint[key] = len(points)
            points.append(q / np.linalg.norm(q))
        return midpoint[key]

    faces = []
    for a, b, c in FACES:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        faces.extend([(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)])
    X = np.array(points)
    cycle = A.perm_from_cycles(d, [list(range(d))])
    local = {v: A.perm_identity(d) for v in range(len(X))}
    local[4], local[5] = cycle, A.perm_inverse(cycle)
    base = A.base_from_face_lists(faces)
    perms = A.perms_from_local(base, d, local, root=0)
    lift = A.assemble(base, d, perms)
    lengths = {(a, b): float(np.linalg.norm(X[a] - X[b]))
               for a, b in base.edges}
    return lift, lengths


def four_simple_branch_cover():
    """Degree 3: profiles [2,1] at four base vertices."""
    p = (1, 0, 2)
    q = (0, 2, 1)
    local = {0: p, 1: p, 4: q, 5: q, 2: (0, 1, 2), 3: (0, 1, 2)}
    base = A.base_from_face_lists(FACES)
    perms = A.perms_from_local(base, 3, local, root=0)
    lift = A.assemble(base, 3, perms)
    lengths = {(a, b): float(np.linalg.norm(OCTA[a] - OCTA[b]))
               for a, b in base.edges}
    return lift, lengths


def mixed_multiplicity_cover(*, p_at=(4, 5), q_at=(0, 1)):
    """Degree 4: [2,2] at ``p_at`` and [2,1,1] at ``q_at``."""
    p = (1, 0, 3, 2)                 # (0 1)(2 3)
    q = (0, 2, 1, 3)                 # (1 2)
    local = {v: (0, 1, 2, 3) for v in range(6)}
    for v in p_at:
        local[v] = p
    for v in q_at:
        local[v] = q
    base = A.base_from_face_lists(FACES)
    perms = A.perms_from_local(base, 4, local, root=0)
    lift = A.assemble(base, 4, perms)
    lengths = {(a, b): float(np.linalg.norm(OCTA[a] - OCTA[b]))
               for a, b in base.edges}
    return lift, lengths


def surviving_scaling_residual(out, lift, base_lengths):
    """Independent chordal fit on edges surviving the final Delaunay chart."""
    from pullback import spherical_ceps as sc

    inherited = {sc.canonical_edge(i, j): base_lengths[tuple(lift["base_of"][lab])]
                 for lab, (i, j) in lift["edges"].items()}
    final_edges = {sc.canonical_edge(f[t], f[(t + 1) % 3])
                   for f in out["faces_final"] for t in range(3)}
    common = sorted(set(inherited) & final_edges)
    if not common:
        raise AssertionError("no input edge survives the final chart")
    P = out["positions"]
    A_mat = np.zeros((len(common), len(P)))
    b = np.zeros(len(common))
    for row, (i, j) in enumerate(common):
        A_mat[row, i] = A_mat[row, j] = 0.5
        b[row] = np.log(np.linalg.norm(P[i] - P[j])) - np.log(inherited[i, j])
    u, *_ = np.linalg.lstsq(A_mat, b, rcond=None)
    residual = A_mat @ u - b
    return float(np.max(np.abs(residual))), len(common), len(inherited)


def assert_transitive(test, lift, d):
    """Transitivity of the monodromy group is independent of the cover DSU."""
    generators = list({tuple(p) for p in lift["edge_perms"].values()})
    seen = {0}
    changed = True
    while changed:
        changed = False
        for p in generators:
            for image in {p[x] for x in tuple(seen)}:
                if image not in seen:
                    seen.add(image)
                    changed = True
    test.assertEqual(seen, set(range(d)))


def exact_fibres(d):
    """All exact roots of every base value, as arrays of unit vectors."""
    out = []
    for p in OCTA:
        target = PL._chart(p)
        roots = []
        if target == mp.inf:
            values = [mp.inf]
        elif target == 0:
            values = [mp.mpc(0)]
        else:
            radius = abs(target) ** (mp.mpf(1) / d)
            theta = mp.arg(target) / d
            values = [radius * mp.exp(1j * (theta + 2 * mp.pi * k / d))
                      for k in range(d)]
        roots = np.array([PL._position(z) for z in values])
        out.append(roots)
    return out


def match_analytic_fibres(P, lift, d):
    """Match three generic fibres, then score every root of every fibre.

    For d=3, nearest-root continuation down a base spanning tree gives a wrong
    sheet labelling even though every vertex still satisfies w^d=target.  The
    oracle therefore tests all d^3 anchor choices and requires the remaining
    vertices to decide the match.  The solve has already happened.
    """
    fibres = exact_fibres(d)
    anchor_vertices = [0, d, 2 * d]       # over the distinct base vertices 0,1,2
    best = None
    for labels in itertools.product(range(d), repeat=3):
        Q = np.array([fibres[v][labels[j]] for j, v in enumerate((0, 1, 2))])
        try:
            mob, _info = G.fit_mobius(P[anchor_vertices], Q, orientation=1)
        except G.GaugeError:
            continue
        R = mob.apply(P)
        assignment, rows = [], []
        for u in range(len(P)):
            candidates = fibres[lift["proj"][u]]
            distances = np.linalg.norm(candidates - R[u], axis=1)
            k = int(np.argmin(distances))
            assignment.append(k)
            rows.append(float(distances[k]))
        worst = max(rows)
        if best is None or worst < best[0]:
            best = (worst, mob, assignment)
    if best is None:
        raise AssertionError("no analytic Möbius candidate could be fitted")
    worst, mob, assignment = best
    for u in range(len(P)):
        base_vertex = lift["proj"][u]
        candidates = fibres[base_vertex]
        distinct_in_fibre = 1 if base_vertex in (4, 5) else d
        assert len({assignment[v] for v in lift["index"][base_vertex]}) == distinct_in_fibre
        Q = candidates[assignment[u]]
        if u == 0:
            answer = np.empty_like(P)
        answer[u] = Q
    return worst, mob, answer


def inherited_lengths(lift, base_lengths):
    return {lab: base_lengths[tuple(lift["base_of"][lab])] for lab in lift["edges"]}


class TestRealFullPreimageCEPS(unittest.TestCase):
    def test_degree_two_and_three_real_solves(self):
        for d, counts, margin_floor in ((2, [10, 24, 16], 0.27),
                                        (3, [14, 36, 24], 0.08)):
            with self.subTest(degree=d):
                lift, lengths = root_cover(d)
                special = min(u for u in lift["proj"] if lift["ld"][u] == 1)
                out = PL.uniformize_cover(lift, lengths, special_vertex=special)
                P = out["positions"]
                self.assertEqual(lift["cert"]["counts"]["lift"], counts)
                self.assertEqual(P.shape, (counts[0], 3))
                self.assertTrue(np.isfinite(P).all())
                self.assertLess(np.max(np.abs(np.linalg.norm(P, axis=1) - 1)), 1e-13)
                self.assertEqual(set(out["preimages"]), set(range(len(OCTA))))
                for p, rows in out["preimages"].items():
                    expected = d if p in (4, 5) else [1] * d
                    got = [ld for _u, ld, _xyz in rows]
                    self.assertEqual(got, expected if isinstance(expected, list) else [expected])
                    self.assertEqual(sum(got), d)
                    self.assertEqual([u for u, _ld, _xyz in rows], lift["index"][p])
                cert = out["res"].certificate
                self.assertGreater(cert["min_pair_distance"], 0.3)
                self.assertGreater(cert["worst_support_margin"], margin_floor)
                self.assertTrue(cert["orientation_consistent"], cert)
                self.assertTrue(cert["oriented_closed_surface"], cert)
                self.assertTrue(cert["all_labelled"], cert)
                self.assertLess(cert["flat_scaling_residual"], 1e-12)
                fit = L._scaling_fit(P, lift["tris"], inherited_lengths(lift, lengths))
                self.assertLess(fit["residual"], 1e-12)
                worst, _mob, analytic = match_analytic_fibres(P, lift, d)
                self.assertLess(worst, 1e-10)
                mob, info = G.register_on_labels(P, range(len(P)), analytic,
                                                 range(len(P)), orientation=1)
                self.assertLess(info["max_angle"], 1e-10, info)
                self.assertLess(np.max(np.linalg.norm(mob.apply(P) - analytic, axis=1)),
                                1e-10)
                for key in ("delta", "newton", "layout"):
                    self.assertIn(key, out["ceps_phase_seconds"])

    def test_real_degree_two_polish_after_explicit_gauge_fit(self):
        old = mp.mp.dps
        mp.mp.dps = 100
        try:
            d = 2
            lift, lengths = root_cover(d)
            out = PL.uniformize_cover(lift, lengths, special_vertex=0)
            _worst, gauge, analytic = match_analytic_fibres(out["positions"], lift, d)
            registered = gauge.apply(out["positions"])
            targets = [PL._chart(OCTA[lift["proj"][u]]) for u in range(len(analytic))]
            f = RationalMap([1, 0, 0], [1])
            polished = PL.polish_preimages(
                f, registered, targets, [lift["ld"][u] for u in range(len(analytic))],
                tol=mp.mpf("1e-20"), position_tol=mp.mpf("1e-20"),
                basin_radius=mp.mpf("0.1"))
            self.assertEqual(polished["status"], "converged")
            self.assertLess(max(polished["travel"]), mp.mpf("0.1"))
            self.assertLess(max(polished["residuals"][-1:]), mp.mpf("1e-40"))
            self.assertLess(np.max(np.linalg.norm(polished["positions"] - analytic, axis=1)),
                            1e-10)
            self.assertEqual(polished["chart_values"][8], mp.inf)
            self.assertEqual(polished["chart_values"][9], 0)
            combined = PL.lift_and_polish(
                f, lift, lengths, targets,
                [lift["ld"][u] for u in range(len(analytic))],
                special_vertex=0, gauge_map=gauge,
                tol=mp.mpf("1e-20"), position_tol=mp.mpf("1e-20"),
                basin_radius=mp.mpf("0.1"))
            self.assertEqual(combined["status"], "converged")
            self.assertLess(max(combined["residuals"][-1:]), mp.mpf("1e-40"))
            self.assertEqual(combined["positions"].shape, (10, 3))
            self.assertEqual(combined["ceps_positions"].shape, (10, 3))
        finally:
            mp.mp.dps = old

    def test_degree_three_with_four_simple_ramified_preimages(self):
        lift, lengths = four_simple_branch_cover()
        cert = lift["cert"]
        self.assertEqual(cert["counts"]["lift"], [14, 36, 24])
        self.assertEqual(cert["rh_deficit"], 4)
        self.assertTrue(cert["connected"] and cert["is_sphere"], cert)
        self.assertEqual(cert["parallel_pairs"], [])
        self.assertEqual(lift["cert"]["fibres"], {
            "0": [2, 1], "1": [2, 1], "2": [1, 1, 1],
            "3": [1, 1, 1], "4": [2, 1], "5": [2, 1]})
        assert_transitive(self, lift, 3)

        # No analytic map or seed enters this solve. Vertex 0 fails, then the
        # deterministic policy accepts the healthy realisation at vertex 1.
        out = PL.uniformize_cover(lift, lengths)
        self.assertEqual(out["special_vertex"], 1)
        self.assertEqual(len(out["special_attempts"]), 2)
        self.assertFalse(out["special_attempts"][0]["ok"])
        self.assertEqual(len(out["positions"]), 14)
        self.assertTrue(out["res"].realizable)
        self.assertLess(out["res"].kkt, 1e-10)
        self.assertGreater(out["res"].flips, 0)
        cert = out["res"].certificate
        self.assertGreater(cert["worst_support_margin"], 0.04)
        self.assertGreater(cert["min_pair_distance"], 0.22)
        self.assertLess(cert["flat_scaling_residual"], 1e-12)
        surviving, common, total = surviving_scaling_residual(out, lift, lengths)
        self.assertLess(surviving, 1e-12)
        self.assertLess(common, total)  # this is a genuine flip control
        all_edges = L._scaling_fit(out["positions"], lift["tris"],
                                  inherited_lengths(lift, lengths))["residual"]
        self.assertGreater(all_edges, 1e-4)  # not a valid post-flip certificate

    def test_degree_four_with_two_doublefold_fibres(self):
        lift, lengths = mixed_multiplicity_cover()
        cert = lift["cert"]
        self.assertEqual(cert["counts"]["lift"], [18, 48, 32])
        self.assertEqual(cert["rh_deficit"], 6)
        self.assertTrue(cert["connected"] and cert["is_sphere"], cert)
        self.assertEqual(cert["parallel_pairs"], [])
        self.assertEqual(cert["fibres"], {
            "0": [2, 1, 1], "1": [2, 1, 1],
            "2": [1, 1, 1, 1], "3": [1, 1, 1, 1],
            "4": [2, 2], "5": [2, 2]})
        assert_transitive(self, lift, 4)

        out = PL.uniformize_cover(lift, lengths)
        self.assertEqual(out["special_vertex"], 0)
        self.assertTrue(out["res"].realizable)
        self.assertLess(out["res"].kkt, 1e-10)
        self.assertGreater(out["res"].flips, 0)
        cert = out["res"].certificate
        self.assertGreater(cert["worst_support_margin"], 0.004)
        self.assertGreater(cert["min_pair_distance"], 0.07)
        self.assertLess(cert["flat_scaling_residual"], 1e-12)
        for p in (4, 5):
            rows = out["preimages"][p]
            self.assertEqual([ld for _u, ld, _xyz in rows], [2, 2])
            # Two ramified preimages over one value are two distinct points.
            self.assertGreater(np.linalg.norm(rows[0][2] - rows[1][2]), 1.6)
        surviving, common, total = surviving_scaling_residual(out, lift, lengths)
        self.assertLess(surviving, 1e-12)
        self.assertLess(common, total)
        self.assertGreater(L._scaling_fit(
            out["positions"], lift["tris"], inherited_lengths(lift, lengths)
        )["residual"], 1e-4)

    def test_non_realizable_returns_are_refused_then_retried(self):
        # A low-KKT point on the clamped extension is not a solution. In the
        # primary placement special 7 returns one such point.
        lift, lengths = mixed_multiplicity_cover()
        with self.assertRaisesRegex(sc.CEPSFailure, "no special vertex succeeded"):
            PL.uniformize_cover(lift, lengths, special_vertex=7)

        # The alternate placement used to accept special 4 with margin -1e-16
        # and chart residual 0.198, before silently returning it. The tightened
        # policy continues to the valid solve at special 8.
        alternate, alt_lengths = mixed_multiplicity_cover(p_at=(0, 1), q_at=(4, 5))
        out = PL.uniformize_cover(alternate, alt_lengths)
        self.assertEqual(out["special_vertex"], 8)
        self.assertEqual([a["special_vertex"] for a in out["special_attempts"]],
                         list(range(9)))
        self.assertTrue(all(not a["ok"] for a in out["special_attempts"][:-1]))
        self.assertTrue(out["special_attempts"][-1]["ok"])
        self.assertTrue(out["res"].realizable)
        self.assertGreater(out["res"].certificate["worst_support_margin"], 0.004)
        self.assertLess(out["res"].certificate["flat_scaling_residual"], 1e-12)

    def test_default_retries_special_vertex_with_attempt_log(self):
        lift, lengths = subdivided_root_cover(2)
        self.assertEqual(lift["cert"]["counts"]["lift"], [34, 96, 64])
        with self.assertRaisesRegex(sc.CEPSFailure, "peacock chart"):
            PL.uniformize_cover(lift, lengths, special_vertex=0)
        out = PL.uniformize_cover(lift, lengths)  # automatic deterministic retry
        self.assertEqual(out["special_vertex"], 2)
        self.assertEqual(len(out["special_attempts"]), 3)
        self.assertFalse(out["special_attempts"][0]["ok"])
        self.assertFalse(out["special_attempts"][1]["ok"])
        self.assertTrue(out["special_attempts"][2]["ok"])
        self.assertGreater(out["res"].certificate["worst_support_margin"], 0.019)
        self.assertEqual(sum(len(rows) for rows in out["preimages"].values()), 34)

    def test_cover_certificate_refuses_disconnected_input_before_ceps(self):
        base = A.base_from_face_lists(FACES)
        d = 2
        local = {v: A.perm_identity(d) for v in base.vertices}
        perms = A.perms_from_local(base, d, local)
        lift = A.assemble(base, d, perms)
        self.assertFalse(lift["cert"]["is_sphere"])
        with self.assertRaisesRegex(ValueError, "connected spherical"):
            PL.uniformize_cover(lift, {e: np.sqrt(2) for e in base.edges})


class TestGuardedPreimagePolish(unittest.TestCase):
    def setUp(self):
        self.old_dps = mp.mp.dps
        mp.mp.dps = 100
        self.third = mp.mpf(1) / 3

    def tearDown(self):
        mp.mp.dps = self.old_dps

    def test_chart_is_stable_near_the_pole(self):
        z = mp.mpc("1e12", "-2e12")
        self.assertLess(abs(PL._chart(PL._position(z)) - z) / abs(z), 1e-12)
        self.assertEqual(PL._chart(PL._position(mp.inf)), mp.inf)
        for z in (mp.mpc("0.2", "-0.4"), mp.mpc(-3), mp.inf):
            if z == mp.inf:
                continue
            self.assertLess(abs(PL._chart(PL._position(z)) - z), 1e-14)

    def test_simple_fibre_has_measured_second_order_and_full_history(self):
        out = PL.polish_preimages(
            RationalMap([1, 0, 0], [1]), [mp.mpc(".99"), mp.mpc("-1.01")],
            [1, 1], [1, 1], tol=mp.mpf("1e-20"), position_tol=mp.mpf("1e-20"),
            basin_radius=1)
        self.assertEqual(out["status"], "converged")
        self.assertTrue(out["order_fit"]["reliable"], out["order_fit"])
        self.assertAlmostEqual(out["order"], 2.0, delta=0.01)
        self.assertGreaterEqual(out["order_fit"]["pairs"], 3)
        self.assertTrue(out["history"][-1]["seconds"] >= 0)
        rows = out["history"][0]["vertices"]
        for row, degree in zip(rows, (1, 1)):
            self.assertEqual(row["local_degree"], degree)
            for key in ("residual", "step", "travel", "scale", "rejected_trials"):
                self.assertIn(key, row)

    def test_declared_multiplicity_polishes_ramified_points(self):
        cases = [
            (2, RationalMap([19, -16, 6], [12, -18, 13])),
            (3, RationalMap([53, -48, 6, 6], [24, -9, -27, 23])),
        ]
        for k, f in cases:
            with self.subTest(k=k):
                out = PL.polish_preimages(
                    f, [self.third + mp.mpf("1e-3")], [self.third], [k],
                    tol=mp.mpf("1e-20"), position_tol=mp.mpf("1e-20"),
                    basin_radius=1)
                self.assertEqual(out["status"], "converged")
                self.assertLess(abs(out["chart_values"][0] - self.third), mp.mpf("1e-30"))
                self.assertTrue(out["order_fit"]["reliable"], out["order_fit"])
                self.assertAlmostEqual(out["order"], 2.0, delta=0.01)
                self.assertEqual(out["history"][0]["vertices"][0]["local_degree"], k)

    def test_finite_poles_and_an_infinite_preimage(self):
        f = RationalMap([1, 0, 1], [1, 0, -1])
        out = PL.polish_preimages(
            f, [mp.mpc("1.01"), mp.mpc("-0.99")], [mp.inf, mp.inf], [1, 1],
            tol=mp.mpf("1e-20"), position_tol=mp.mpf("1e-20"), basin_radius=1)
        self.assertLess(abs(out["chart_values"][0] - 1), mp.mpf("1e-30"))
        self.assertLess(abs(out["chart_values"][1] + 1), mp.mpf("1e-30"))
        self.assertAlmostEqual(out["order"], 2.0, delta=0.01)
        infinite = PL.polish_preimages(
            f, [mp.mpc(10)], [1], [2], tol=mp.mpf("1e-20"),
            position_tol=mp.mpf("1e-20"), basin_radius=2)
        self.assertEqual(infinite["chart_values"][0], mp.inf)
        self.assertEqual(infinite["residuals"][-1], 0)

    def test_far_basin_raises_instead_of_accepting(self):
        with self.assertRaises(PL.BasinError):
            PL.polish_preimages(RationalMap([1, 0, 0], [1]), [mp.mpc(4)], [1],
                                [1], basin_radius=mp.mpf("0.1"), maxit=12)

    def test_real_ceps_iterate_with_small_perturbation_has_measured_order(self):
        d = 2
        lift, lengths = root_cover(d)
        solved = PL.uniformize_cover(lift, lengths, special_vertex=0)
        _worst, gauge, analytic = match_analytic_fibres(solved["positions"], lift, d)
        registered = gauge.apply(solved["positions"])
        perturbed = []
        for u, p in enumerate(registered):
            z = PL._chart(p)
            if z == mp.inf:
                z = mp.mpc(100 + u, -50)
            else:
                z = z * (1 + mp.mpf("0.001") * (u + 1)) + mp.mpc("0.0002", "-0.0001")
            perturbed.append(PL._position(z))
        targets = [PL._chart(OCTA[lift["proj"][u]]) for u in range(len(analytic))]
        out = PL.polish_preimages(
            RationalMap([1, 0, 0], [1]), perturbed, targets,
            [lift["ld"][u] for u in range(len(analytic))],
            tol=mp.mpf("1e-20"), position_tol=mp.mpf("1e-20"),
            basin_radius=mp.mpf("0.25"), maxit=32)
        self.assertEqual(out["status"], "converged")
        self.assertTrue(out["order_fit"]["reliable"], out["order_fit"])
        self.assertAlmostEqual(out["order"], 2.0, delta=0.05)
        self.assertLess(np.max(np.linalg.norm(out["positions"] - analytic, axis=1)), 1e-10)


if __name__ == "__main__":
    unittest.main()
