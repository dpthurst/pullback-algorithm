"""M3 certificates and iterate graphs: the SF_EL lower bounds (plan 180/200)
and the explicit G_n (plan 190), with the bouquet end-to-end test.

Every check has an input on which it fails (agent-craft §3).
"""

import sys
import unittest
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pullback.refined import Graph, GraphMap
from pullback.certificates import (Curve, curve_ratio, lower_bound,
                                   enumerate_components, bracket)
from pullback.iterate_graphs import (iterate_graph, iterate_map,
                                     refine_to_single_darts, session_iterate)
from pullback.iterates import iterate_energy
from pullback.harmonic_io import load_session

from tests.test_refined import tripod, ex117_map


def ex117_curve():
    """The tripod curve of [Elast] intro.tex (the example after thm:emb-sf):
    two arcs ending at the leaf of arm 1, n_c = (2,1,1)."""
    base = ex117_map()
    c = Curve(base.src, [(1, (("a2", -1), ("a1", 1))),
                         (1, (("a1", -1), ("a3", 1)))], name="c")
    return base, c


class TestDualCertificates(unittest.TestCase):
    """The extremal-length lower bounds (thm:emb-sf, eq:sf-el)."""

    def test_el_ratio_is_the_paper_number(self):
        # EL[phi o c]/EL[c] = (4+1+1)/(4+3+3) = 6/10 = 3/5
        base, c = ex117_curve()
        self.assertEqual(c.EL(), F(10))
        self.assertEqual(c.image(base).EL(), F(6))
        self.assertEqual(curve_ratio(base, c), F(3, 5))

    def test_lower_bound_is_3_5(self):
        base, _c = ex117_curve()
        low = lower_bound(base, max_darts=6, top_k=8, max_parts=3, max_weight=3)
        self.assertEqual(low[0], F(3, 5))
        # content: the bound is not 1
        self.assertNotEqual(low[0], F(1))

    def test_lower_bound_moves_with_the_data(self):
        # perturbing an elastic constant moves the bound
        G1 = tripod({"a1": F(1), "a2": F(2), "a3": F(3)}, "G1'")
        G2 = tripod(F(1), "G2")
        other = GraphMap(G1, G2, {"a1": (("a1", 1),), "a2": (("a2", 1),),
                                  "a3": (("a3", 1),)}, name="phi'")
        low2 = lower_bound(other, max_darts=6, top_k=8, max_parts=3,
                           max_weight=3)
        self.assertNotEqual(low2[0], F(3, 5))

    def test_curve_validity_is_enforced(self):
        base, _c = ex117_curve()
        # a non-reduced word is refused
        with self.assertRaises(ValueError):
            Curve(base.src, [(1, (("a1", 1), ("a1", -1), ("a2", 1)))])
        # an arc must start and end at marked vertices
        with self.assertRaises(ValueError):
            Curve(base.src, [(1, (("a1", 1),))])
        # a loop must avoid marked vertices
        with self.assertRaises(ValueError):
            Curve(base.src, [(1, (("a1", 1), ("a2", -1)))])

    def test_partial_filling_dual(self):
        # the right-hand loop: EL 2 -> 4, ratio 2.  The curve family's marked
        # set is empty here (a loop avoids marked points, def:curve); the
        # class convention is a separate input.
        G1 = Graph({"l": ("v", "v"), "m": ("v", "v")},
                   {"l": F(2), "m": F(2)}, marked=[])
        G2 = Graph({"e": ("w", "w"), "f": ("w", "w")}, {"e": F(1), "f": F(4)})
        m = GraphMap(G1, G2, {"l": (("e", 1),), "m": (("f", 1),)})
        loop = Curve(G1, [(1, (("m", 1),))], name="right loop")
        self.assertEqual(curve_ratio(m, loop), F(2))
        self.assertEqual(m.emb(), F(2))
        # the left loop is weaker
        left = Curve(G1, [(1, (("l", 1),))], name="left loop")
        self.assertLess(curve_ratio(m, left), F(2))

    def test_bracket_closes_for_ex117(self):
        from tests.test_refined import ex117_witness
        base, _c = ex117_curve()
        low, up, exact = bracket(base, witnesses=[ex117_witness()],
                                max_darts=6, top_k=8, max_parts=3, max_weight=3)
        self.assertEqual(low, F(3, 5))
        self.assertEqual(up, F(3, 5))
        self.assertTrue(exact)


class TestIterateGraphs(unittest.TestCase):
    """The explicit G_n: counts, agreement with iterates.py, and the bouquet
    end-to-end test against the published Harmonic values."""

    def test_counts_and_explicit_bound(self):
        for n in (1, 2, 3):
            Gn, phin, G2 = session_iterate(ROOT / "sessions" / "bouquet.json", n)
            self.assertEqual(len(Gn.deg), 2 ** n)
            self.assertEqual(len(Gn.edges), 2 ** (n + 1))
            self.assertEqual(phin.emb(), F(1))

    def test_agrees_with_the_ported_iterate_energy(self):
        # the explicit G_n must reproduce src/pullback/iterates.py's
        # E^2_2(phi^n_0)^2 = Emb(phi^n_0)
        base, cover, phi = load_session(ROOT / "sessions" / "bouquet.json")
        for n in (1, 2, 3):
            e = iterate_energy(cover, phi, n, F(2))
            _Gn, phin, _G2 = session_iterate(ROOT / "sessions" / "bouquet.json", n)
            self.assertEqual(e.pow(2).exact_value(), phin.emb())
            self.assertEqual(phin.emb(), F(1))

    def test_dual_reaches_the_published_value_at_n_3(self):
        # the published Emb[f^3] = 5/8 (Harmonic output.tex), reached from
        # the extremal-length side on the explicit G_3
        _Gn, phin, _G2 = session_iterate(ROOT / "sessions" / "bouquet.json", 3)
        SEARCH = [{"max_darts": 6, "top_k": 6, "max_parts": 4, "max_weight": 6},
                  {"max_darts": 8, "top_k": 6, "max_parts": 3, "max_weight": 5}]
        best = None
        for kw in SEARCH:
            low = lower_bound(phin, **kw)
            if low is not None and (best is None or low[0] > best[0]):
                best = low
        self.assertEqual(best[0], F(5, 8))
        self.assertEqual(phin.emb(), F(1))          # the explicit bound is 1
        self.assertNotEqual(best[0], F(1))          # so the test has content

    def test_dual_bounds_are_below_the_published_values(self):
        # n = 4, 5: the search falls short of the published values, so
        # those brackets stay open -- asserted as bounds, not values
        published = {4: F(23, 48), 5: F(187, 480)}
        for n, pub in published.items():
            _Gn, phin, _G2 = session_iterate(ROOT / "sessions" / "bouquet.json", n)
            low = lower_bound(phin, max_darts=6, top_k=6, max_parts=4,
                              max_weight=6)
            if low is not None:
                self.assertLessEqual(low[0], pub)
            self.assertLessEqual(pub, phin.emb())

    def test_minimiser_matches_the_published_values(self):
        # the end-to-end test: the minimiser on the explicit G_n reproduces
        # the published Harmonic values exactly, from the fixed point alone
        # (LB = max DR = Emb[phi^n_0] by prop:fixed-lambda at an interior
        # fixed point).  n = 1..3: tractable (the loop enumeration is the
        # simple cycles of G_n); n = 4 takes ~75 s and n = 5 is out of
        # reach, so they stay script-level (scripts/m3_minimise.py).
        from pullback.minimise import Minimiser, minimise
        published = {1: F(1), 2: F(3, 4), 3: F(5, 8)}
        for n, pub in published.items():
            Gn, phin, G2 = session_iterate(ROOT / "sessions" / "bouquet.json", n)
            C = Minimiser(Gn, G2, phin.chains)
            res = minimise(C)
            self.assertEqual(res["LB"], pub, f"n={n}")
            self.assertLessEqual(pub, phin.emb())   # explicit UB dominates

    def test_refine_to_single_darts(self):
        # a multi-dart chain is refined into single-dart pieces
        G1 = Graph({"e": ("u", "v")}, F(1))
        G2 = Graph({"a": ("w", "x"), "b": ("x", "y")}, F(1))
        chains = {"e": (("a", 1), ("b", 1))}
        rsrc, back, rchains = refine_to_single_darts(G1, G2, chains)
        self.assertEqual(len(rsrc.edges), 2)
        self.assertEqual(rchains, {("e", 0): (("a", 1),), ("e", 1): (("b", 1),)})
        # the refined map has the same Emb as the original
        m = GraphMap(G1, G2, chains)
        rm = GraphMap(rsrc, G2, rchains)
        self.assertEqual(m.emb(), rm.emb())


class TestRubberFigure2(unittest.TestCase):
    """Figure 2 of [Rubber] (arXiv:1502.02561), the log-plot of SF[phi_n] for
    f(z) = (1+z^2)/(1-z^2).  Its (1,1,0) curve is the bouquet -- the exact
    fractions 1, 3/4, 5/8, 23/48, 187/480 at n = 1..5 (the first four are
    TestIterateGraphs's published-value test); its (1,1,1) curve is the theta
    session with unit elastics, where the fixed point is rational at n = 1
    and n = 3 and irrational at n = 2."""

    def test_theta_unit_elastics_is_3_2_at_n1_and_6_7_at_n3(self):
        from pullback.minimise import Minimiser, minimise
        from pullback.iterate_graphs import session_iterate

        def run(n):
            Gn, phin, G2 = session_iterate(ROOT / "sessions" / "theta.json", n)
            C = Minimiser(Gn, G2, phin.chains)      # unit elastics throughout
            return minimise(C)

        self.assertEqual(run(1)["LB"], F(3, 2))
        self.assertEqual(run(3)["LB"], F(6, 7))
        # content: n = 2 is irrational, so the exact fixed-point solve finds
        # nothing there and the value must be reported as a bracket, not a
        # value (the paper's 0.879465^2 = 0.773459 is its limit)
        self.assertIsNone(run(2)["LB"])

    def test_figure_data_are_the_exact_fractions(self):
        # the figure's (1,1,0) entries are the n-th roots of the exact values
        for frac, root, fig in [(F(3, 4), 2, 0.866025), (F(5, 8), 3, 0.854988),
                                (F(23, 48), 4, 0.831997),
                                (F(187, 480), 5, 0.828171)]:
            self.assertAlmostEqual(float(frac) ** (1 / root), fig, places=6)
        self.assertAlmostEqual(float(F(6, 7)) ** (1 / 3), 0.949914, places=6)


class TestDeterminism(unittest.TestCase):
    def test_bouquet_iterate_is_deterministic(self):
        rows = []
        for _ in range(2):
            Gn, phin, _G2 = session_iterate(ROOT / "sessions" / "bouquet.json", 2)
            rows.append((sorted(Gn.deg, key=str),
                         {k: phin.chains[k] for k in sorted(phin.chains, key=str)}))
        self.assertEqual(rows[0], rows[1])


if __name__ == "__main__":
    unittest.main()
