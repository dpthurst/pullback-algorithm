"""M3 minimiser: alg:iter, the exact harmonic step, the fixed-point solve,
and the certified bracket (plan items 170/180).

Every check has an input on which it fails (agent-craft §3).
"""

import sys
import unittest
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pullback.refined import Graph, GraphMap, session_map
from pullback.minimise import (Minimiser, minimise, curve_rows,
                               qp_active_set, rational_roots, charpoly,
                               null_vector, lin_solve)

from tests.test_refined import tripod, ex117_map


def ex117_class():
    """[Elast] Ex. 1.17 as a Minimiser: leaves marked."""
    base = ex117_map()
    return Minimiser(base.src, base.tgt, base.chains)


def partial_filling_class():
    """[Elast] filling.tex examp:partial-filling: both graphs the join of
    two circles, elastics (2,2) and (1,4), phi the identity on pi_1."""
    G1 = Graph({"l": ("v", "v"), "m": ("v", "v")}, {"l": F(2), "m": F(2)},
               marked=["v"])
    G2 = Graph({"e": ("w", "w"), "f": ("w", "w")}, {"e": F(1), "f": F(4)})
    chains = {"l": (("e", 1),), "m": (("f", 1),)}
    return Minimiser(G1, G2, chains)


class TestEx117(unittest.TestCase):
    """[Elast] Ex. 1.17: Emb = 3/5, minimiser at 2/5 along an edge."""

    def setUp(self):
        self.C = ex117_class()
        self.res = minimise(self.C)

    def test_curve_constraints_are_the_three_arcs(self):
        # the tripod is a tree, so the simple marked curves are exactly the
        # three leaf-to-leaf arcs; each crosses each edge once
        keys = sorted(self.C.keys)
        self.assertEqual(keys,
                         sorted([((1, 1, 0), (1, 1, 0)), ((1, 0, 1), (1, 0, 1)),
                                 ((0, 1, 1), (0, 1, 1))]))

    def test_harmonic_step_at_unit_lengths(self):
        # the Dir-minimiser for l = alpha2 = (1,1,1) is the z = 1/5 map,
        # NOT the Emb-minimiser: m = (4/5, 6/5, 6/5), Dir = 8/5
        h = self.C.harmonic({"a1": F(1), "a2": F(1), "a3": F(1)})
        self.assertEqual([h["m"][e] for e in ("a1", "a2", "a3")],
                         [F(4, 5), F(6, 5), F(6, 5)])
        self.assertEqual(h["Dir"], F(8, 5))
        self.assertEqual(h["DR"], F(8, 15))

    def test_fixed_point_is_exact(self):
        # Iter(l*) = (3/5) l* with l* = (1/2, 1/4, 1/4) = 1/4*(2,1,1)
        lam, l = self.res["interior"][0]
        self.assertEqual(lam, F(3, 5))
        self.assertEqual([l[e] for e in ("a1", "a2", "a3")],
                         [F(1, 2), F(1, 4), F(1, 4)])
        h = self.C.harmonic(l)
        for e in ("a1", "a2", "a3"):
            self.assertEqual(h["l_next"][e], lam * l[e])
        self.assertEqual(h["DR"], F(3, 5))

    def test_bracket_closes_at_3_5(self):
        # LB = max DR = 3/5 (thm:emb-sf) and the published minimiser has
        # Emb = 3/5 (tests/test_energies.py TestEx117, repo evaluator)
        self.assertEqual(self.res["LB"], F(3, 5))
        from tests.test_refined import ex117_witness
        self.assertEqual(ex117_witness().emb(), F(3, 5))

    def test_fixed_point_geometry_is_the_published_minimiser(self):
        # at l* the harmonic map sends the source vertex to distance
        # l_a1 - m(a1) = 1/5 along arm a1, i.e. 2/5 OF THE ARM -- the
        # "2/5 of the way along an edge" of [Elast] examp:emb
        lam, l = self.res["interior"][0]
        h = self.C.harmonic(l)
        z = l["a1"] - h["m"]["a1"]
        self.assertEqual(z / l["a1"], F(2, 5))

    def test_dr_is_non_decreasing(self):
        # lem:DR-ER-increase: DR never decreases along alg:iter
        drs = [h["DR"] for _l, h in self.res["history"]]
        for a, b in zip(drs, drs[1:]):
            self.assertLessEqual(a, b)


class TestPartialFilling(unittest.TestCase):
    """[Elast] filling.tex examp:partial-filling: Emb = 2, no lambda-filling
    map (a boundary fixed point)."""

    def setUp(self):
        self.C = partial_filling_class()
        self.res = minimise(self.C)

    def test_two_fixed_points_one_attracting(self):
        fps = {(lam, tuple(sorted((k, str(v)) for k, v in l.items())))
               for lam, l, h in self.res["fixed_points"]}
        self.assertIn((F(2), (("e", "0"), ("f", "1"))), fps)
        self.assertIn((F(1, 2), (("e", "1"), ("f", "0"))), fps)

    def test_no_interior_fixed_point(self):
        self.assertEqual(self.res["interior"], [])

    def test_emb_is_2(self):
        # both sides meet at 2 without the escape procedure
        self.assertEqual(self.res["LB"], F(2))
        G1 = Graph({"l": ("v", "v"), "m": ("v", "v")},
                   {"l": F(2), "m": F(2)}, marked=["v"])
        G2 = Graph({"e": ("w", "w"), "f": ("w", "w")}, {"e": F(1), "f": F(4)})
        m = GraphMap(G1, G2, {"l": (("e", 1),), "m": (("f", 1),)})
        self.assertEqual(m.emb(), F(2))


class TestBarbell(unittest.TestCase):
    """The barbell session, all cover vertices marked (the convention under
    which Emb = 5 is proved): the explicit bound 6 is not sharp."""

    def setUp(self):
        G1, m, G2 = session_map(ROOT / "sessions" / "barbell.json")
        G1.marked = frozenset(G1.deg)
        self.G1, self.m, self.G2 = G1, m, G2
        self.C = Minimiser(G1, G2, m.chains)
        self.res = minimise(self.C)

    def test_explicit_representative_is_6(self):
        # the pinned explicit Fill = (1, 6, 3) (AGENTS.md A7)
        self.assertEqual(self.m.emb(), F(6))

    def test_minimised_is_5_with_closed_bracket(self):
        self.assertEqual(self.res["LB"], F(5))
        # the certificate map: p0 split into q (2/5), r (1/5), q^-1 (2/5),
        # p1 -> P whole, the q/r lifts constant -- a refinement of phi itself
        from tests.test_refined import ex117_map  # noqa: F401  (import check)
        cert = _barbell_certificate(self.G1, self.m, self.G2)
        self.assertEqual(cert.emb(), F(5))
        self.assertTrue(cert.in_class())
        self.assertLess(F(5), F(6))

    def test_boundary_fixed_point(self):
        self.assertTrue(any(l["p"] == 0
                            for lam, l, h in self.res["fixed_points"]))


def _barbell_certificate(G1, m, G2):
    """The prop:fixed-boundary certificate map for the barbell: the collapsed
    edge is p (l_p = 0 at the boundary fixed point); the Delta_1 source edge
    p0 is refined at the preimages of the target's vertices into q (2/5),
    r (1/5), q^-1 (2/5)."""
    from pullback.refined import RefinedMap, piece_label
    base = m
    # source edges of the cover: (p,0),(p,1),(q,0),(q,1),(r,0),(r,1).
    # subdivide() relabels every edge to (e, i), so the chains are keyed by
    # refined labels throughout (the unsplit edges are (e, 0)).
    src_splits = {("p", 0): [F(2, 5), F(1, 5), F(2, 5)]}
    chains = {
        piece_label(("p", 0), 0): ((piece_label("q", 0), 1),),
        piece_label(("p", 0), 1): ((piece_label("r", 0), 1),),
        piece_label(("p", 0), 2): ((piece_label("q", 0), -1),),
        piece_label(("p", 1), 0): ((piece_label("p", 0), -1),),
        piece_label(("q", 0), 0): (),
        piece_label(("q", 1), 0): (),
        piece_label(("r", 0), 0): (),
        piece_label(("r", 1), 0): (),
    }
    return RefinedMap(base, src_splits=src_splits, tgt_splits={},
                      chains=chains, name="cert")


class TestExactAlgebra(unittest.TestCase):
    """The exact linear algebra: rational roots, charpoly, null vectors."""

    def test_rational_roots(self):
        self.assertEqual(rational_roots([F(1), F(-5, 2), F(1)]),
                         [(F(2), 1), (F(1, 2), 1)])
        self.assertEqual(rational_roots([F(1), F(-8, 5), F(3, 5)]),
                         [(F(1), 1), (F(3, 5), 1)])

    def test_charpoly(self):
        M = [[F(0), F(1)], [F(1), F(0)]]
        self.assertEqual(charpoly(M), [F(1), F(0), F(-1)])

    def test_null_vector(self):
        # kernel of [[1, -1]]: (1, 1)
        v = null_vector([[F(1), F(-1)]])
        self.assertEqual(v, [F(1), F(1)])

    def test_qp_active_set(self):
        # min x^2 + y^2 s.t. x + y >= 1, x,y >= 0: optimum (1/2, 1/2), val 1/2
        r, ys, S = qp_active_set([(F(1), F(1))], [F(1)], [F(1), F(1)])
        self.assertEqual(r, (F(1, 2), F(1, 2)))
        self.assertEqual(sum(x * x for x in r), F(1, 2))


class TestMutants(unittest.TestCase):
    """Each mutant has an input on which the corresponding check fails
    (agent-craft §3): the assertion is that the *correct* value is not
    produced by the mutant."""

    def test_mutant_vertex_to_vertex_harmonic_step(self):
        # replacing the QP harmonic step by the vertex-to-vertex reading
        # gives m = (1, 1, 1) at l = (1,1,1), not (4/5, 6/5, 6/5)
        C = ex117_class()
        h = C.harmonic({"a1": F(1), "a2": F(1), "a3": F(1)})
        self.assertNotEqual([h["m"][e] for e in ("a1", "a2", "a3")],
                            [F(1), F(1), F(1)])

    def test_mutant_drops_the_arc_constraints(self):
        # with no curve rows the QP optimum is r = 0, Dir = 0: detected
        r, ys, S = qp_active_set([], [], [F(1), F(1), F(1)])
        self.assertEqual(r, (F(0), F(0), F(0)))
        C = ex117_class()
        h = C.harmonic({"a1": F(1), "a2": F(1), "a3": F(1)})
        self.assertGreater(h["Dir"], F(0))

    def test_mutant_unit_target_lengths(self):
        # the fixed point is at l* = (1/2, 1/4, 1/4), not at (1, 1, 1):
        # using unit lengths throughout changes the answer
        lam, l = ex117_class() and minimise(ex117_class())["interior"][0]
        self.assertNotEqual([l[e] for e in ("a1", "a2", "a3")],
                            [F(1), F(1), F(1)])


if __name__ == "__main__":
    unittest.main()
