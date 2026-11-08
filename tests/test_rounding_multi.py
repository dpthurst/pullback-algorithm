"""Tests for edge-labelled multigraph uniformisation and its public lift API.

Run from the repository root:

    python3 .ultramath/smw83rxe5/m5/test_multigraph_rounding.py

Every assertion below is a statement about the real rabbit data: the base
Delaunay dual (V4/E6/F4, no redundant pair) and the coarse lifted dual of the
admissible rabbit fixture (V6/E12/F8, one redundant pair -- the two lifts of the
base edge (1,3) join the same pair of dual vertices).
"""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "src")

import mpmath as mp  # noqa: E402

from pullback import rounding_multi as mg  # noqa: E402
from pullback import lift_round as proposal_lift  # noqa: E402
from pullback import discrete, mesh, thurston as th  # noqa: E402
from pullback.lift import (round_realise_cells as round_cells, dual_cells, lifted_delaunay_metric,  # noqa: E402
                           parallel_edge_pairs, pullback_ve, _base_geometry)
from pullback.maps import make_graph_map  # noqa: E402
from pullback.ribbons import F, B  # noqa: E402
from pullback.session import Session  # noqa: E402

RABBIT_CR = complex(0.8774388331233464, 0.7448617666197443)   # thurston.centre_table


def rabbit_admissible():
    """The admissible rabbit fixture of tests/test_lift.py."""
    mp.mp.dps = 40
    session = Session.load("sessions/rabbit.json")
    data = discrete.pcf_data(th.rabbit_map())
    spine = mesh.voronoi_spine(data["points"])
    chains = {
        (0, 1): (), (1, 2): (), (0, 2): (),
        (0, 3): (("b", F),),
        (1, 3): (("a", B),),
        (2, 3): (("c", F),),
    }
    rho = make_graph_map(
        spine, session.base, chains,
        {v: session.base.vertices()[0] for v in spine.vertices()})
    P, pi_prime, phi_prime, _pc = pullback_ve(
        session.base, session.cover, session.phi, spine, rho)
    cells = dual_cells(P)
    metric = lifted_delaunay_metric(cells, spine, data["points"])
    return data, spine, cells, metric


def base_complex():
    """The base Delaunay dual: V4/E6/F4, a tetrahedron, no redundant pair."""
    mp.mp.dps = 40
    data = discrete.pcf_data(th.rabbit_map())
    spine = mesh.voronoi_spine(data["points"])
    cells = dual_cells(spine)
    _vectors, lengths, _cone = _base_geometry(
        data["points"], mesh.delaunay_faces(data["points"]))
    edges, metric_lengths = proposal_lift.labelled_cells(cells)
    labels = {label: float(lengths[tuple(sorted(label))])
              for label in edges}
    return data, spine, cells, edges, labels


def sphere_start(n, seed):
    x = np.random.RandomState(seed).normal(size=(n, 3))
    return x / np.linalg.norm(x, axis=1)[:, None]


def mobius_boost(x, v, s):
    """A genuine Mobius transformation of S^2 (a Lorentz boost)."""
    v = np.asarray(v, dtype=float)
    v = v / np.linalg.norm(v)
    ch, sh = np.cosh(s), np.sinh(s)
    dot = np.asarray(x, dtype=float) @ v
    num = np.asarray(x, dtype=float) + (((ch - 1.0) * dot + sh)[:, None]) * v[None, :]
    return num / (ch + sh * dot)[:, None]


def orbit_of(cr):
    """The six values a labelled 4-point cross ratio can take under relabelling."""
    return [cr, 1.0 / cr, 1.0 - cr, 1.0 / (1.0 - cr),
            cr / (cr - 1.0), (cr - 1.0) / cr]


class TestCoarseRabbitLift(unittest.TestCase):
    """The multigraph case: two edges join the same pair of dual vertices."""

    @classmethod
    def setUpClass(cls):
        cls.data, cls.spine, cls.cells, cls.metric = rabbit_admissible()
        cls.edges, cls.lengths = proposal_lift.labelled_cells(cls.cells, cls.metric)
        cls.n = len(cls.cells["vertices"])

    def test_the_lift_is_labelled_12_edges_not_11(self):
        self.assertEqual(self.cells["counts"], {"V": 6, "E": 12, "F": 8})
        self.assertEqual(len(self.edges), 12)
        self.assertEqual(len({tuple(sorted(v)) for v in self.edges.values()}), 11)
        parallel = parallel_edge_pairs(self.cells)
        self.assertEqual(list(parallel), [(0, 5)])
        first, second = parallel[(0, 5)]
        # two distinct labels, same base edge, same dual endpoints
        self.assertNotEqual(first, second)
        self.assertEqual(first[0], second[0])
        self.assertEqual(self.edges[first], self.edges[second])
        self.assertEqual(mg.edge_incidence(self.edges, self.n)[0].shape, (12, 6))
        self.assertEqual(mg.redundant_pairs(self.edges), 1)

    def test_analytic_jacobian_matches_finite_differences(self):
        x = sphere_start(self.n, 3)
        J = mg.conformal_jacobian(self.edges, x)
        self.assertEqual(J.shape, (12, 18))
        h = 1e-6
        Jfd = np.zeros_like(J)
        for k in range(self.n):
            for c in range(3):
                xp = x.copy()
                xm = x.copy()
                xp[k, c] += h
                xm[k, c] -= h
                Jfd[:, 3 * k + c] = (
                    mg.conformal_residual(self.edges, self.lengths, xp)[0]
                    - mg.conformal_residual(self.edges, self.lengths, xm)[0]) / (2 * h)
        # measured 1.3e-10 at h = 1e-6, i.e. the finite-difference error floor
        self.assertLess(np.abs(J - Jfd).max(), 1e-7)

    def test_gauss_newton_rounds_the_coarse_lift(self):
        x, residual, iters = mg.round_realise(
            self.edges, self.lengths, x0=sphere_start(self.n, 0),
            maxit=200, tol=1e-14)
        self.assertLess(residual, 1e-13)
        self.assertLess(iters, 25)
        # post hoc: the lengths differ from the chordal ones by a vertex scaling
        A, order = mg.edge_incidence(self.edges, self.n)
        chord = mg.chordal_lengths(x)
        a = np.array([np.log(chord[i, j]) - np.log(self.lengths[label])
                      for (label, i, j) in order])
        sol, _res, _rank, _sv = np.linalg.lstsq(A, a, rcond=None)
        self.assertLess(np.abs(a - A @ sol).max(), 1e-12)

    def test_parallel_edges_are_independent(self):
        # Two lifts of the same base edge carry two data.  If their lengths
        # disagree, no configuration can satisfy both: the exact floor is
        # |log(ell1/ell2)|/sqrt(2), because (a_e1 - a_e2) is orthogonal to the
        # image of the incidence map and independent of the positions.
        parallel = parallel_edge_pairs(self.cells)[(0, 5)]
        bad = dict(self.lengths)
        bad[parallel[1]] = self.lengths[parallel[1]] * 1.01
        _x, floor, _it = mg.round_realise(
            self.edges, bad, x0=sphere_start(self.n, 0), maxit=300, tol=1e-15)
        expected = abs(np.log(1.01)) / np.sqrt(2.0)
        self.assertGreater(floor, 1e-4)
        self.assertAlmostEqual(floor, expected, places=9)

    def test_the_vertex_pair_collapse_loses_that_constraint(self):
        # The same inconsistent data, collapsed onto vertex pairs (last write
        # wins, which is what a face-derived edge list does): the solver reports
        # success.  This is the silent data loss the labelled form prevents.
        parallel = parallel_edge_pairs(self.cells)[(0, 5)]
        bad = dict(self.lengths)
        bad[parallel[1]] = self.lengths[parallel[1]] * 1.01
        pairs = {}
        for label, (i, j) in self.edges.items():
            pairs[tuple(sorted((i, j)))] = bad[label]
        collapsed = {key: key for key in pairs}
        _x, collapsed_residual, _it = mg.round_realise(
            collapsed, pairs, x0=sphere_start(self.n, 0), maxit=300, tol=1e-15)
        self.assertLess(collapsed_residual, 1e-10)
        self.assertEqual(len(collapsed), 11)

    def test_the_residual_is_mobius_invariant(self):
        x, _r, _it = mg.round_realise(
            self.edges, self.lengths, x0=sphere_start(self.n, 0),
            maxit=200, tol=1e-15)
        before = mg.conformal_residual(self.edges, self.lengths, x)[2]
        moved = mobius_boost(x, [0.2, -0.7, 0.68], 0.9)
        after = mg.conformal_residual(self.edges, self.lengths, moved)[2]
        self.assertLess(before, 1e-13)
        self.assertLess(after, 1e-12)
        self.assertLess(
            np.abs(mg.mobius_invariants(moved) - mg.mobius_invariants(x)).max(),
            1e-10)

    def test_rank_accounting_says_the_realisation_is_not_canonical(self):
        x, _r, _it = mg.round_realise(
            self.edges, self.lengths, x0=sphere_start(self.n, 0),
            maxit=200, tol=1e-15)
        info = mg.structure(self.edges, x)
        self.assertEqual(info["V"], 6)
        self.assertEqual(info["E"], 12)
        self.assertEqual(info["rank_A"], 6)
        self.assertEqual(info["redundant"], 1)
        # measured: 5 = E - rank(A) - redundant, with a clean singular-value gap
        self.assertEqual(info["rank_PJ"], 5)
        sv = info["singular_values"]
        self.assertGreater(sv[4], 0.1)
        self.assertLess(sv[5], 1e-12)
        # solution set 2V - rank = 7 dimensions, one more than the Mobius orbit
        self.assertEqual(2 * info["V"] - info["rank_PJ"], 7)
        self.assertEqual(info["excess"], 1)
        self.assertFalse(info["canonical"])

    def test_independent_starts_land_in_different_mobius_classes(self):
        found = []
        for seed in range(3):
            x, residual, _it = mg.round_realise(
                self.edges, self.lengths, x0=sphere_start(self.n, 300 + seed),
                maxit=300, tol=1e-14)
            self.assertLess(residual, 1e-12)
            found.append(mg.mobius_invariants(x))
        spread = max(np.abs(found[i] - found[0]).max() for i in range(1, 3))
        # measured 14.13 for these three seeds (3.2 to 54 over a wider sweep):
        # the extra modulus is not infinitesimal
        self.assertGreater(spread, 1.0)


class TestBaseDelaunayComplex(unittest.TestCase):
    """The canonical case: V4/E6/F4, no redundant pair, excess 0."""

    @classmethod
    def setUpClass(cls):
        (cls.data, cls.spine, cls.cells,
         cls.edges, cls.lengths) = base_complex()
        cls.n = len(cls.cells["vertices"])

    def test_the_structure_is_a_triangulation(self):
        self.assertEqual(self.cells["counts"], {"V": 4, "E": 6, "F": 4})
        self.assertEqual(mg.redundant_pairs(self.edges), 0)
        info = mg.structure(self.edges)
        self.assertEqual(info["rank_A"], 4)
        self.assertEqual(info["rank_P"], 2)
        self.assertEqual(2 * info["V"] - 6, 2)

    def test_excess_is_zero_so_the_realisation_is_canonical(self):
        x, residual, _it = mg.round_realise(
            self.edges, self.lengths, x0=sphere_start(self.n, 0),
            maxit=200, tol=1e-15)
        self.assertLess(residual, 1e-13)
        info = mg.structure(self.edges, x)
        self.assertEqual(info["rank_PJ"], 2)
        self.assertEqual(info["excess"], 0)
        self.assertTrue(info["canonical"])

    def test_the_round_realisation_recovers_the_rabbit_configuration(self):
        # Known answer: the base Delaunay metric is the chordal metric of the
        # rabbit's own postcritical configuration, so its round realisation must
        # be that configuration up to Mobius and relabelling.  The cross ratio
        # must therefore be one of the six orbit values of thurston's published
        # rabbit cross ratio (or a conjugate -- the chordal metric cannot see
        # orientation, so the mirror image is an equally good realisation).
        x, residual, _it = mg.round_realise(
            self.edges, self.lengths, x0=sphere_start(self.n, 0),
            maxit=200, tol=1e-15)
        self.assertLess(residual, 1e-13)
        found = mg.mobius_invariants(x)[0]
        targets = [z for cr in orbit_of(RABBIT_CR) for z in (cr, np.conj(cr))]
        best = min(abs(found - z) for z in targets)
        self.assertLess(best, 1e-11)
        # the value itself, pinned: 0.6623589786223735 + 0.5622795120623013i
        self.assertAlmostEqual(found.real, 0.6623589786223735, places=11)
        self.assertAlmostEqual(abs(found.imag), 0.5622795120623013, places=11)

    def test_independent_starts_agree_up_to_conjugation(self):
        reference = None
        for seed in range(4):
            x, residual, _it = mg.round_realise(
                self.edges, self.lengths, x0=sphere_start(self.n, 500 + seed),
                maxit=300, tol=1e-15)
            self.assertLess(residual, 1e-13)
            cr = mg.mobius_invariants(x)[0]
            if reference is None:
                reference = cr
                continue
            self.assertLess(
                min(abs(cr - reference), abs(cr - np.conj(reference))), 1e-10)

    def test_vertex_scaling_of_the_input_leaves_the_class_alone(self):
        # The invariance the whole method rests on: ell -> exp((u_i+u_j)/2) ell
        # is the same conformal class, so the same Mobius class comes back.
        u = np.array([0.3, -0.15, 0.22, 0.41])
        scaled = {label: self.lengths[label]
                  * np.exp(0.5 * (u[i] + u[j]))
                  for label, (i, j) in self.edges.items()}
        a, _r, _it = mg.round_realise(
            self.edges, self.lengths, x0=sphere_start(self.n, 700), maxit=300,
            tol=1e-15)
        b, _r, _it = mg.round_realise(
            self.edges, scaled, x0=sphere_start(self.n, 701), maxit=300,
            tol=1e-15)
        ca, cb = mg.mobius_invariants(a)[0], mg.mobius_invariants(b)[0]
        self.assertLess(min(abs(ca - cb), abs(ca - np.conj(cb))), 1e-10)

    def test_perturbing_one_length_changes_the_class_and_stays_unique(self):
        # A check that can fail: the class must move, and must still be unique.
        label = sorted(self.lengths, key=str)[0]
        perturbed = dict(self.lengths)
        perturbed[label] = self.lengths[label] * 1.02
        reference, _r, _it = mg.round_realise(
            self.edges, self.lengths, x0=sphere_start(self.n, 899), maxit=300,
            tol=1e-15)
        cr0 = mg.mobius_invariants(reference)[0]
        values = []
        for seed in range(3):
            x, residual, _it = mg.round_realise(
                self.edges, perturbed, x0=sphere_start(self.n, 900 + seed),
                maxit=300, tol=1e-15)
            self.assertLess(residual, 1e-13)
            values.append(mg.mobius_invariants(x)[0])
        for value in values:
            self.assertLess(
                min(abs(value - values[0]), abs(value - np.conj(values[0]))),
                1e-10)
        moved = min(abs(values[0] - cr0), abs(np.conj(values[0]) - cr0))
        self.assertGreater(moved, 1e-3)   # it really moved


def _shared_vertex(e1, e2):
    common = set(e1) & set(e2)
    return common.pop() if len(common) == 1 else None


def _split_copy(edges, faces, label, tag, next_vertex):
    """Split the two faces through `label` at a fresh midpoint vertex.

    Each copy of a doubled edge gets its OWN midpoint, and each of its two
    incident triangles is cut in two by the diagonal from the midpoint to the
    opposite corner.  That is what restores E = 3V - 6 without creating a
    degree-2 vertex (which would make the count worse, not better).
    """
    a, b = edges.pop(label)
    m = next_vertex
    first, second = (tag, 0), (tag, 1)
    edges[first] = (a, m)
    edges[second] = (m, b)
    out = []
    for face in faces:
        if label not in face:
            out.append(face)
            continue
        k = face.index(label)
        prev, nxt = face[k - 1], face[(k + 1) % len(face)]
        opp = _shared_vertex(edges[prev], edges[nxt])
        if opp is None:
            raise ValueError("the two neighbouring edges do not meet")
        diag = (tag, "diag", len(out))
        edges[diag] = (m, opp)
        out.append((prev, first, diag))
        out.append((second, nxt, diag))
    return edges, out, next_vertex + 1


class TestMinimalTriangulatedRepair(unittest.TestCase):
    """What a complex needs for excess 0: one midpoint per copy of the pair.

    This is the measured handoff to the refined-base direction: splitting one
    copy only (a degree-2 vertex) leaves E - V at 6 while 2V - 6 becomes 8, so
    it makes the excess WORSE.  Both copies must be split, each at its own new
    vertex, and each of the four incident triangles cut by its own diagonal.
    """

    @classmethod
    def setUpClass(cls):
        cls.data, cls.spine, cls.cells, cls.metric = rabbit_admissible()
        cls.edges, cls.lengths = proposal_lift.labelled_cells(cls.cells, cls.metric)
        cls.faces = [tuple(dart[0] for dart in walk)
                     for walk in cls.cells["corners"]]

    def test_splitting_each_copy_gives_a_triangulation_with_excess_zero(self):
        parallel = parallel_edge_pairs(self.cells)[(0, 5)]
        edges = dict(self.edges)
        faces = list(self.faces)
        nxt = len(self.cells["vertices"])
        for k, label in enumerate(parallel):
            edges, faces, nxt = _split_copy(edges, faces, label, ("split", k), nxt)
        info = mg.structure(edges)
        self.assertEqual(info["V"], 8)
        self.assertEqual(info["E"], 18)
        self.assertEqual(len(faces), 12)
        self.assertEqual(info["E"], 3 * info["V"] - 6)
        self.assertEqual(len(faces), 2 * info["V"] - 4)
        self.assertEqual(info["redundant"], 0)
        self.assertEqual(info["rank_P"], 10)
        # rank(PJ) = E - rank(A) - redundant = 10 = 2V - 6, so excess is 0
        self.assertEqual(2 * info["V"] - 6,
                         info["E"] - info["rank_A"] - info["redundant"])
        # every face is a triangle and every edge is used exactly twice
        self.assertTrue(all(len(f) == 3 for f in faces))
        uses = {}
        for face in faces:
            for t in range(3):
                uses[face[t]] = uses.get(face[t], 0) + 1
        self.assertEqual({e: u for e, u in uses.items() if u != 2}, {})

    def test_splitting_one_copy_is_already_a_triangulation(self):
        # The minimal repair: ONE midpoint on one copy plus two diagonals.
        # V 7, E 15, F 10 and 3V - 6 = 15, 2V - 4 = 10: a genuine triangulation
        # with no redundant pair, so the excess is already 0.
        parallel = parallel_edge_pairs(self.cells)[(0, 5)]
        edges, faces, _nxt = _split_copy(dict(self.edges), list(self.faces),
                                        parallel[0], ("split", 0), 6)
        info = mg.structure(edges)
        self.assertEqual((info["V"], info["E"], len(faces)), (7, 15, 10))
        self.assertEqual(info["E"], 3 * info["V"] - 6)
        self.assertEqual(len(faces), 2 * info["V"] - 4)
        self.assertEqual(info["redundant"], 0)
        self.assertEqual(2 * info["V"] - 6,
                         info["E"] - info["rank_A"] - info["redundant"])
        uses = {}
        for face in faces:
            for t in range(3):
                uses[face[t]] = uses.get(face[t], 0) + 1
        self.assertEqual({e: u for e, u in uses.items() if u != 2}, {})


class TestProposedLiftAPI(unittest.TestCase):
    """The boundary a caller uses: cells in, configuration + certificate out."""

    @classmethod
    def setUpClass(cls):
        cls.data, cls.spine, cls.cells, cls.metric = rabbit_admissible()

    def test_the_api_returns_a_configuration_and_a_certificate(self):
        out = round_cells(
            self.cells, self.metric, x0=sphere_start(6, 0), maxit=300, tol=1e-14)
        self.assertEqual(out["x"].shape, (6, 3))
        self.assertLess(out["residual"], 1e-13)
        self.assertEqual(out["V"], 6)
        self.assertEqual(out["E"], 12)
        self.assertEqual(out["redundant"], 1)
        self.assertEqual(out["excess"], 1)
        self.assertFalse(out["canonical"])
        # positions are on the sphere and the cone angles are the output's own
        self.assertLess(
            np.abs(np.linalg.norm(out["x"], axis=1) - 1.0).max(), 1e-14)
        self.assertAlmostEqual(out["cone_angles"].sum() / np.pi, 8.0, places=10)

    def test_requiring_canonical_says_exactly_what_is_missing(self):
        with self.assertRaises(ValueError) as caught:
            round_cells(
                self.cells, self.metric, x0=sphere_start(6, 0), maxit=300,
                tol=1e-14, require_canonical=True)
        message = str(caught.exception)
        self.assertIn("excess 1", message)
        self.assertIn("not unique up to Mobius", message)

    def test_a_parallel_pair_is_reported_not_refused(self):
        # The old boundary raised on 'parallel edge pairs'; this one solves.
        self.assertEqual(list(parallel_edge_pairs(self.cells)), [(0, 5)])
        out = round_cells(
            self.cells, self.metric, x0=sphere_start(6, 1), maxit=300, tol=1e-14)
        self.assertLess(out["residual"], 1e-13)

    def test_the_base_complex_is_canonical_through_the_same_api(self):
        _data, _spine, cells, edges, lengths = base_complex()
        metric = {"lengths": {label: lengths[label] for label in edges}}
        out = round_cells(
            cells, metric, x0=sphere_start(4, 0), maxit=300, tol=1e-15,
            require_canonical=True)
        self.assertTrue(out["canonical"])
        self.assertEqual(out["excess"], 0)
        found = mg.mobius_invariants(out["x"])[0]
        targets = [z for cr in orbit_of(RABBIT_CR) for z in (cr, np.conj(cr))]
        self.assertLess(min(abs(found - z) for z in targets), 1e-10)

    def test_coarse_realisation_is_reported_as_folded_and_can_be_refused(self):
        out = round_cells(
            self.cells, self.metric, x0=sphere_start(6, 0), maxit=300, tol=1e-14)
        self.assertIn("embedding", out)
        self.assertFalse(out["embedding"]["embedded"])
        self.assertIn("opposite orientation", out["embedding"]["reason"])
        with self.assertRaisesRegex(ValueError, "not an embedded sphere"):
            round_cells(self.cells, self.metric, x0=sphere_start(6, 0),
                        maxit=300, tol=1e-14, require_embedded=True)

    def test_a_regular_tetrahedron_is_canonical_and_embedded(self):
        points = [
            mesh.to_point(np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)),
            mesh.to_point(np.array([1.0, -1.0, -1.0]) / np.sqrt(3.0)),
            mesh.to_point(np.array([-1.0, 1.0, -1.0]) / np.sqrt(3.0)),
            mesh.to_point(np.array([-1.0, -1.0, 1.0]) / np.sqrt(3.0)),
        ]
        spine = mesh.voronoi_spine(points)
        cells = dual_cells(spine)
        _vectors, base_lengths, _cone = _base_geometry(
            points, mesh.delaunay_faces(points))
        edges, metric_lengths = proposal_lift.labelled_cells(cells)
        metric = {"lengths": {
            label: base_lengths[tuple(sorted(label))] for label in edges}}
        face_point = mesh.voronoi_face_points(spine, len(points))
        x0 = np.array([mesh.to_vector(points[face_point[v]])
                       for v in cells["vertices"]])
        out = round_cells(cells, metric, x0=x0, maxit=300, tol=1e-15,
                          require_canonical=True, require_embedded=True)
        self.assertTrue(out["canonical"])
        self.assertTrue(out["embedding"]["embedded"])
        self.assertLess(out["residual"], 1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
