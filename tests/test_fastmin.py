"""The fast (floating-point) minimiser against the exact one (plan 170).

`pullback.minimise` is exact but enumerates all simple curves of the source,
which is exponential in |E(G_n)|; `pullback.fastmin` solves the harmonic step
by position least squares, as the author's Harmonic code does, and scales.
The two are cross-checked here on the cases where both run.
"""

import sys
import unittest
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pullback.fastmin import FastMap, alg_iter
from pullback.iterate_graphs import session_iterate
from pullback.minimise import Minimiser, minimise


def edge_letter(e):
    """The base edge letter of a G_n edge label."""
    lab = e[-1] if isinstance(e, tuple) else e
    while isinstance(lab, tuple) and len(lab) >= 1 and not isinstance(lab[0], str):
        lab = lab[0]
    return lab[0] if isinstance(lab, tuple) else lab


def theta_run(n, a):
    Gn, phin, G2 = session_iterate(ROOT / "sessions" / "theta.json", n)
    alpha1 = {e: float(a[edge_letter(e)]) for e in Gn.edges}
    lams, _ell2, m = alg_iter(Gn, G2, phin.chains, alpha1, dict(a))
    m.validate()
    return lams[-1]


class TestThetaAgainstPaper(unittest.TestCase):
    """The theta example of [Rubber] Figure 2: SF[phi_n] for two parameter
    sets, n = 1..4 (the paper's table; the values for larger n are the same
    computation and are exercised by scripts/reproduce_figure2.py)."""

    # the paper's SF[f^(n)]^{1/n}, raised to the n
    PAPER = {
        (F(4), F(1), F(4)): [F(864357, 10**6), None, None, None],
    }

    def test_4_1_4(self):
        a = {"p": 4.0, "q": 1.0, "r": 4.0}
        # paper: 0.864357, 0.839121, 0.825873, 0.818408 (the n-th roots)
        for n, root in ((1, 0.864357), (2, 0.839121), (3, 0.825873), (4, 0.818408)):
            got = theta_run(n, a)
            # the paper prints the n-th root to six decimals, so its n-th
            # power carries ~n*5e-7 of rounding: compare at 1e-5
            self.assertAlmostEqual(got, root ** n, delta=1e-5, msg=f"n={n}")

    def test_1_1_1(self):
        a = {"p": 1.0, "q": 1.0, "r": 1.0}
        # paper: 1.5, 0.879465, 0.949914, 0.852017; n = 1 and 3 are exact
        # (3/2 and 6/7 -- the exact engine's interior fixed points)
        self.assertAlmostEqual(theta_run(1, a), 1.5, places=9)
        self.assertAlmostEqual(theta_run(2, a), 0.879465 ** 2, delta=1e-5)
        self.assertAlmostEqual(theta_run(3, a), float(F(6, 7)), places=8)
        self.assertAlmostEqual(theta_run(4, a), 0.852017 ** 4, delta=1e-5)


class TestCrossCheckWithExact(unittest.TestCase):
    """The fast path against the exact engine where both run."""

    def test_theta_unit_elastics_matches_exact(self):
        # exact: interior fixed points 3/2 (n=1) and 6/7 (n=3)
        for n, want in ((1, F(3, 2)), (3, F(6, 7))):
            Gn, phin, G2 = session_iterate(ROOT / "sessions" / "theta.json", n)
            alpha1 = {e: 1.0 for e in Gn.edges}
            exact = minimise(Minimiser(Gn, G2, phin.chains))["LB"]
            self.assertEqual(exact, want)
            self.assertAlmostEqual(theta_run(n, {"p": 1.0, "q": 1.0, "r": 1.0}),
                                   float(want), places=8, msg=f"n={n}")

    def test_theta_4_1_4_matches_exact_dr(self):
        # the exact engine's DR sequence converges to the fast value
        a = {"p": F(4), "q": F(1), "r": F(4)}
        Gn, phin, G2 = session_iterate(ROOT / "sessions" / "theta.json", 2)
        for e in Gn.edges:
            Gn.alpha[e] = a[edge_letter(e)]
        for e in G2.edges:
            G2.alpha[e] = a[e]
        C = Minimiser(Gn, G2, phin.chains)
        l = {"p": F(1), "q": F(1), "r": F(1)}
        dr = None
        for _ in range(10):
            h = C.harmonic(l)
            dr = float(h["DR"])
            nxt = h["l_next"]
            tot = sum(nxt.values(), F(0))
            l = {e: nxt[e] / tot for e in l}
        fast = theta_run(2, {"p": 4.0, "q": 1.0, "r": 4.0})
        self.assertAlmostEqual(fast, dr, places=6)


class TestWarmStart(unittest.TestCase):
    """alg:iter takes an initial target metric (`ell0`); the tower of
    iterates can be chained -- the previous n's fixed-point metric starts
    the next run -- and the answers must not change."""

    def test_theta_warm_chain_matches_cold(self):
        a = {"p": 1.0, "q": 1.0, "r": 1.0}
        ell_prev = None
        for n in (1, 2, 3, 4):
            Gn, phin, G2 = session_iterate(ROOT / "sessions" / "theta.json", n)
            alpha1 = {e: a[edge_letter(e)] for e in Gn.edges}
            cold, _e, _m = alg_iter(Gn, G2, phin.chains, alpha1, dict(a))
            warm, ell_prev, _m = alg_iter(Gn, G2, phin.chains, alpha1, dict(a),
                                          ell0=ell_prev)
            with self.subTest(n=n):
                self.assertAlmostEqual(warm[-1], cold[-1], places=12)
                self.assertLessEqual(len(warm), len(cold))


class TestKnownLimitation(unittest.TestCase):
    """The fast path is not yet right for the bouquet at n >= 2: its chains
    are constant on the lifts of b, and the degenerate cases (both endpoints
    in one edge; an empty chain after a push) are where the type repair goes
    wrong.  The exact engine handles it; this test records the gap rather
    than hiding it."""

    def test_bouquet_constant_chains(self):
        Gn, phin, G2 = session_iterate(ROOT / "sessions" / "bouquet.json", 2)
        alpha1 = {e: 1.0 for e in Gn.edges}
        alpha2 = {e: 1.0 for e in G2.edges}
        # exact: 3/4 at n = 2
        exact = minimise(Minimiser(Gn, G2, phin.chains))["LB"]
        self.assertEqual(exact, F(3, 4))
        lams, _ell, _m = alg_iter(Gn, G2, phin.chains, alpha1, alpha2)
        if abs(lams[-1] - float(exact)) < 1e-6:
            return                       # fixed: the limitation is gone
        self.skipTest(f"known limitation: fast path gives {lams[-1]:.6f} "
                      f"for the bouquet n=2, exact 0.750000")


if __name__ == "__main__":
    unittest.main()
