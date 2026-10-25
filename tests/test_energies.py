"""M3 energies: the E^p_q family for all p, and the iterate bounds.

Checks with known answers:
  * the trivial map is the length ratio, for every p;
  * the p-parallel law of [PChar, Prop. 7.7] / [ConfDim] holds exactly at
    rational lengths (and fails for the wrong law: the test has a
    corrupted companion);
  * E^1_1, E^1_q, E^p_inf against hand formulas;
  * sub-multiplicativity E^p_r(psi phi) <= E^p_q(phi) E^q_r(psi);
  * the iterate construction: |V(G_n)| = d^n |V_0| always, and for a
    genuine covering phi the energy is exactly (deg phi)^(1/p) at n = 1
    and (deg phi)^(n/p) at n, which no wrong edge constraint survives.
"""

import sys
import unittest
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pullback.energies import (Energy, E_1_1, E_1_q, E_p_inf, E_p_p,
                               compose_chains, energy, exact_root, fills)
from pullback.harmonic_io import load_session, parse_cmap, parse_cover, parse_graph
from pullback.iterates import (degree, fill_matrix, fill_vector,
                               iterate_energy, perron_root)


def harmonic_sum(a, b, p):
    """a (+) b = (a^(1-p) + b^(1-p))^(1/(1-p)); exact when it is rational."""
    k = p - 1
    s = F(1) / (a ** k) + F(1) / (b ** k)
    r = exact_root(s, k)
    return None if r is None else F(1) / r


class TestTrivialMaps(unittest.TestCase):
    def test_single_edge_is_length_ratio_power(self):
        # One edge of length L onto one edge of length 1: |phi'| = L,
        # Fill = L^(p-1), E^p_p = L^((p-1)/p) — the definition
        # ([ConfDim] E^q_q with q = p), not L.  Note E^2_2 = sqrt(L),
        # which is the plan's E^2_2 = sqrt(Emb).
        # source length 1, image length 7/3, so |phi'| = 7/3
        for p in (F(2), F(3), F(7)):
            e = E_p_p({"x": (("e", 1),)}, {"x": F(1)}, {"e": F(7, 3)}, p)
            self.assertEqual(e.pow(p).exact_value(), F(7, 3) ** (p - 1), f"p={p}")
        # a non-integral p leaves the Fill irrational: numerically then
        p = F(3, 2)
        e = E_p_p({"x": (("e", 1),)}, {"x": F(1)}, {"e": F(7, 3)}, p)
        self.assertAlmostEqual(float(e.pow(p).value), (7 / 3) ** 0.5, places=12)
        # q = 1 is a different animal: alpha_tgt is a *weight*, and
        # E^1_1 = ess sup n(y)/w(y) = 1/(7/3).
        e1 = E_p_p({"x": (("e", 1),)}, {"x": F(1)}, {"e": F(7, 3)}, F(1))
        self.assertEqual(e1.exact_value(), F(3, 7))

    def test_p2_is_embedding_energy(self):
        chains = {"x": (("e", 1),), "y": (("e", 1),), "z": (("f", 1),)}
        e = E_p_p(chains, {"x": F(1), "y": F(3), "z": F(2)}, F(1), F(2))
        # filling on e: 1 + 1/3; on f: 1/2.  E^2_2^2 = max = 4/3.
        self.assertEqual(e.pow(2).exact_value(), F(4, 3))


class TestParallelLaw(unittest.TestCase):
    """Two parallel source edges mapping to the same path may be replaced
    by one edge of the p-harmonic-sum length, leaving the energy alone."""

    def check(self, p, a, b):
        c = harmonic_sum(a, b, p)
        self.assertIsNotNone(c, "test pair must have a rational p-harmonic sum")
        two = {"x": (("e", 1),), "y": (("e", 1),)}
        one = {"z": (("e", 1),)}
        e2 = E_p_p(two, {"x": a, "y": b}, F(1), p)
        e1 = E_p_p(one, {"z": c}, F(1), p)
        self.assertEqual(e2, e1, f"p={p} a={a} b={b} c={c}")
        self.assertTrue(e2.exact)
        return e2

    def test_p2_exact(self):
        # p = 2: harmonic sum of 3 and 6 is 2
        self.check(F(2), F(3), F(6))

    def test_p3_exact(self):
        # p = 3: (1/9 + 1/16)^(-1/2) = 12/5
        self.check(F(3), F(3), F(4))

    def test_p4_numeric(self):
        # p = 4, a = 1, b = 2: the p-harmonic sum (9/8)^(1/3) is irrational,
        # so substitute the exact value and compare the two fillings: the
        # two-edge filling is a^(-3) + b^(-3); the single-edge one is
        # c^(-3) with c = (9/8)^(1/3), i.e. (8/9).
        two = {"x": (("e", 1),), "y": (("e", 1),)}
        e2 = E_p_p(two, {"x": F(1), "y": F(2)}, F(1), F(4))
        self.assertEqual(e2.pow(4).exact_value(), F(1) + F(1, 8))

    def test_wrong_law_is_detected(self):
        # The harmonic sum is not the arithmetic sum: with a = 3, b = 6 the
        # replacement 9 gives a different energy, so the test above has
        # content.  (A check that cannot fail is not a check.)
        two = {"x": (("e", 1),), "y": (("e", 1),)}
        one = {"z": (("e", 1),)}
        e2 = E_p_p(two, {"x": F(3), "y": F(6)}, F(1), F(2))
        wrong = E_p_p(one, {"z": F(9)}, F(1), F(2))
        self.assertNotEqual(e2, wrong)


class TestOtherCases(unittest.TestCase):
    def test_E1_1_is_multiplicity(self):
        chains = {"x": (("e", 1),), "y": (("e", 1), ("f", 1)),
                  "z": (), }
        e = E_1_1(chains, {"x": F(1), "y": F(1), "z": F(1)}, F(1))
        self.assertEqual(e.exact_value(), F(2))  # e has two preimages

    def test_E1_q_hand(self):
        # E^1_q = (sum_d alpha_d n_d^(q/(q-1)))^((q-1)/q); n = 2 on e only.
        # q = 2 makes the exponent integral and the answer exact: base = 5*4.
        chains = {"x": (("e", 1),), "y": (("e", 1),)}
        e = E_1_q(chains, F(1), {"e": F(5)}, F(2))
        self.assertEqual(e.pow(2).exact_value(), F(20))
        # q = 3: base = 5 * 2^(3/2), so only E^3 is exact up to the root
        e3 = E_1_q(chains, F(1), {"e": F(5)}, F(3))
        self.assertEqual(e3.exponent, F(2, 3))
        self.assertAlmostEqual(float(e3.pow(3).value), 200.0, places=10)

    def test_Ep_inf_hand(self):
        # E^p_inf = (sum_e alpha_e^(1-p) l(phi e)^p)^(1/p)
        chains = {"x": (("e", 1),), "y": (("e", 1), ("f", 1))}
        alpha = {"x": F(2), "y": F(4)}
        tg = {"e": F(1), "f": F(3)}
        for p in (F(2), F(3)):
            e = E_p_inf(chains, alpha, tg, p)
            base = F(2) ** (1 - p) * F(1) ** 1 + F(4) ** (1 - p) * F(4) ** p
            self.assertEqual(e.base, base)
            self.assertEqual(e.exponent, F(1, 1) / p)


class TestSubmultiplicativity(unittest.TestCase):
    """E^p_r(psi phi) <= E^p_q(phi) E^q_r(psi) ([Elast, Prop. A.12])."""

    def setUp(self):
        self.base, self.cover, self.phi = load_session(ROOT / "sessions" / "theta.json")
        tg = {e: F(1) for e in self.base.edge_list()}
        self.tg = tg
        # outer map G_0 -> G_0 sending every base edge to the edge p
        self.outer = {e: (("p", 1),) for e in self.base.edge_list()}

    def test_p2_chain(self):
        chains = self.phi.chains
        comp = compose_chains(chains, self.outer)
        for p, q, r in ((F(2), F(2), F(2)), (F(3), F(3), F(3)),
                        (F(2), F(3), F(3)), (F(1), F(2), F(2))):
            lhs = energy(comp, self.tg, p, r, F(1))
            e1 = energy(chains, self.tg, p, q, F(1))
            e2 = energy(self.outer, self.tg, q, r, F(1))
            self.assertLessEqual(lhs.value, e1.value * e2.value * (1 + 1e-25),
                                 f"p={p} q={q} r={r}")


class TestIterates(unittest.TestCase):
    def test_n1_agrees_with_direct(self):
        for name in ("theta", "census-3.2", "barbell", "rabbit"):
            base, cover, phi = load_session(ROOT / "sessions" / f"{name}.json")
            for p in (F(2), F(3)):
                direct = E_p_p(phi.chains, F(1), F(1), p)
                via = iterate_energy(cover, phi, 1, p)
                self.assertEqual(direct.value, via.value, f"{name} p={p}")

    def test_matrix_against_brute_force(self):
        """Fill^p(phi^n_0) = sum over chains of products of dilations.

        Written out independently for n = 2, 3: x_1 runs over the
        preimages of a generic y, x_{i+1} over the preimages of
        pi(x_i), each contributing delta^(p-1)."""
        for name in ("theta", "census-3.2", "barbell"):
            base, cover, phi = load_session(ROOT / "sessions" / f"{name}.json")
            delta = {}
            for f in cover.edges:
                ch = phi.chains[f]
                delta[f] = (F(len(ch)) if ch else F(0)) / F(1)  # unit alpha
            pre = {}   # base edge -> [(cover edge, multiplicity)]
            for l in {f[0] for f in cover.edges}:
                pre[l] = []
            for f in cover.edges:
                for d in phi.chains[f]:
                    pre[d[0]].append(f)
            for n in (2, 3):
                for l, fs in pre.items():
                    tot = 0
                    for f1 in fs:
                        if n == 1:
                            tot += delta[f1]
                        elif n == 2:
                            tot += delta[f1] * sum(delta[f2]
                                                   for f2 in pre[f1[0]])
                        else:
                            tot += delta[f1] * sum(
                                delta[f2] * sum(delta[f3] for f3 in pre[f2[0]])
                                for f2 in pre[f1[0]])
                    got = fill_vector(cover, phi, n, F(2))[l]
                    self.assertAlmostEqual(float(tot), float(got), places=12,
                                           msg=f"{name} n={n} l={l}")

    def test_theta_multiplicity_by_hand(self):
        # theta: Fill^2 of phi is (1, 2, 1) on base edges (p, q, r); at
        # n = 2 the q-preimages each have a single further preimage, so
        # Fill^1_1(phi^2_0) = 2, not 4.
        base, cover, phi = load_session(ROOT / "sessions" / "theta.json")
        v1 = fill_vector(cover, phi, 1, F(1))
        self.assertEqual(v1, {"p": F(1), "q": F(2), "r": F(1)})
        v2 = fill_vector(cover, phi, 2, F(1))
        self.assertEqual(v2, {"p": F(1), "q": F(2), "r": F(1)})
        self.assertEqual(iterate_energy(cover, phi, 2, F(1)).exact_value(), F(2))

    def test_genuine_cover_energy_is_degree(self):
        # base rose <a,b>; double cover a: swap, b: identity; phi a genuine
        # 2-fold covering (a0->a, a1->b, b0->b, b1->a).  Every chain is a
        # single dart, every delta = 1, so Fill = 2^n everywhere.
        base = parse_graph("XaAbB")
        cover, _ = parse_cover(base, "a01")
        phi = parse_cmap(cover, base, "a0a a1b b0b b1a")
        self.assertEqual(degree(cover), 2)
        for n in (1, 2, 3, 4):
            for p in (F(1), F(2), F(3)):
                e = iterate_energy(cover, phi, n, p)
                self.assertEqual(e.pow(p).exact_value(), 2 ** n, f"n={n} p={p}")

    def test_multidart_chains_are_counted(self):
        """The bug this test pins: an enumeration of G_n by tuples with
        "phi(e_{i+1}) is a single dart" misses the orbits over edges
        whose image is a longer chain.  census-3.2 has such an edge, so
        the naive product-only value must differ from the truth."""
        base, cover, phi = load_session(ROOT / "sessions" / "census-3.2.json")
        naive = {}
        for f in cover.edges:
            ch = phi.chains[f]
            naive[f] = (F(len(ch)) if ch else F(0)) / F(1)
        product_only = {}
        for e1 in cover.edges:
            ch = phi.chains[e1]
            for e2 in cover.edges:
                c2 = phi.chains[e2]
                if len(c2) == 1 and c2[0][0] == e1[0]:
                    w = naive[e1]
                    for d in ch:
                        product_only[d[0]] = product_only.get(d[0], 0) + w
        truth = fill_vector(cover, phi, 2, F(2))
        self.assertNotEqual({k: float(v) for k, v in product_only.items()},
                            {k: float(v) for k, v in truth.items()})

    def test_scale_invariance(self):
        base, cover, phi = load_session(ROOT / "sessions" / "theta.json")
        for n in (1, 2):
            beta = {e: F(i + 2, 3) for i, e in enumerate(base.edge_list())}
            scaled = {e: 7 * v for e, v in beta.items()}
            for p in (F(2), F(3)):
                self.assertEqual(
                    iterate_energy(cover, phi, n, p, base_lengths=beta).value,
                    iterate_energy(cover, phi, n, p, base_lengths=scaled).value)

    def test_perron_root_of_theta(self):
        # theta at unit alpha: Fill^2(phi^n_0) = 2 for every n (measured
        # from the matrix), so E^2_2(phi^n_0) = sqrt 2 and the bound is
        # 2^(1/(2n)) -> 1.  The Perron root of the fill matrix is 1.
        base, cover, phi = load_session(ROOT / "sessions" / "theta.json")
        for n in (1, 2, 3):
            self.assertEqual(iterate_energy(cover, phi, n, F(2)).pow(2).exact_value(), F(2))
            self.assertAlmostEqual(float(iterate_energy(cover, phi, n, F(2))
                                         .pow(F(1, n)).value), 2 ** (1 / (2 * n)), places=12)
        self.assertAlmostEqual(perron_root(cover, phi, F(2)), 1.0, places=10)


class TestBounds(unittest.TestCase):
    def setUp(self):
        self.base, self.cover, self.phi = load_session(ROOT / "sessions" / "theta.json")

    def test_tuning_never_worse_and_exact(self):
        from pullback.bounds import bounding_lengths, upper_bound
        for n in (1, 2):
            e_tuned, beta = bounding_lengths(self.cover, self.phi, n, F(2))
            self.assertLessEqual(e_tuned.value,
                                 iterate_energy(self.cover, self.phi, n, F(2)).value)
            self.assertTrue(e_tuned.exact)
            self.assertIsInstance(e_tuned.base, F)
            for v in beta.values():
                self.assertGreater(v, 0)
        rows = upper_bound(self.cover, self.phi, F(2), 2)
        self.assertLessEqual(min(r[2].value for r in rows), rows[0][2].value)

    def test_critical_p_brackets(self):
        # theta is rational, so its asymptotic energy is < 1 at p = 2
        # (2^(-1/3)), but the sample point gives E^2_2(phi) = sqrt 2 > 1,
        # so the bound cannot certify anything at p = 2: the bisection
        # must return p-star > 2.
        from pullback.bounds import critical_p
        p, row = critical_p(self.cover, self.phi, F(3, 2), F(4), 2)
        self.assertGreater(p, 2)


if __name__ == "__main__":
    unittest.main()


class TestBarbellByHand(unittest.TestCase):
    """barbell: base q(a->b), p(b->a), r(loop at b); cover p01 r01;
    phi p0 -> q r Q (3 darts), p1 -> P, r0, r1 constant.  Unit alpha.

    delta(p0) = 3, delta(p1) = 1, delta of the rest 0.  The fill matrix
    has M[q][p] = 6 (q twice in p0's chain), M[r][p] = 3, M[p][p] = 1,
    so Fill^2(phi^n_0) = (1, 6, 3) for every n: E^2_2(phi^n_0) = sqrt 6
    and the bound is 6^(1/(2n))."""

    def test_fill_matrix_and_energies(self):
        base, cover, phi = load_session(ROOT / "sessions" / "barbell.json")
        M = fill_matrix(cover, phi, F(2))
        self.assertEqual(M["q"]["p"], F(6))
        self.assertEqual(M["r"]["p"], F(3))
        self.assertEqual(M["p"]["p"], F(1))
        for n in (1, 2, 3):
            self.assertEqual(fill_vector(cover, phi, n, F(2)),
                             {"p": F(1), "q": F(6), "r": F(3)})
            e = iterate_energy(cover, phi, n, F(2))
            self.assertEqual(e.pow(2).exact_value(), F(6))
            self.assertAlmostEqual(float(e.pow(F(1, n)).value), 6 ** (1 / (2 * n)),
                                   places=12)


class TestTreeSpine(unittest.TestCase):
    """A tree spine is representable, and the two metrics are independent.

    M3's gate example is [Elast, Ex. 1.17]: two tripods with marked ends,
    elastic constants (1,3,3) on the source and (1,1,1) on the target, and
    Emb[phi] = 3/5 attained by a map that is NOT the harmonic one.  What the
    explicit-representative layer can prove is asserted here; the value 3/5 is
    an infimum over the homotopy class and needs the plan-170 minimiser, so it
    is deliberately not asserted.  The published minimiser *is* pinned exactly
    in TestEx117 below — on the subdivided graphs, where its Emb is 3/5.
    """

    def test_tripod_parses(self):
        # centre v with darts a,b,c; the three leaves carry the reverse ends.
        # (Vertex names and dart letters share one alphabet, so the leaves
        # must not be named A/B/C.)
        G = parse_graph("vabc 1A 2B 3C")
        self.assertEqual(sorted(G.vertices()), ["1", "2", "3", "v"])
        faces = G.faces()
        self.assertEqual(len(faces), 1)                  # a tree: one boundary walk
        self.assertEqual(sum(len(f) for f in faces), 6)  # 3 edges, both sides

    def test_two_metrics_are_independent(self):
        # identity map on the arms, source (1,2,4) and target (2,1,3):
        # |phi'| = 2, 1/2, 3/4, so Emb = 2 exactly and E^2_2 = sqrt(2).
        chains = {"a": (("a", 1),), "b": (("b", 1),), "c": (("c", 1),)}
        src = {"a": F(1), "b": F(2), "c": F(4)}
        tgt = {"a": F(2), "b": F(1), "c": F(3)}
        e = E_p_p(chains, src, tgt, F(2))
        self.assertEqual(e.pow(2).exact_value(), F(2))
        # Using one metric for both sides is a different answer (1), so a
        # wrong convention cannot pass this test.
        self.assertEqual(E_p_p(chains, src, src, F(2)).pow(2).exact_value(), F(1))
        self.assertEqual(E_p_p(chains, tgt, tgt, F(2)).pow(2).exact_value(), F(1))

    def test_ex_1_17_vertex_representative(self):
        # The vertex-to-vertex representative of Ex. 1.17's class, arms to
        # arms: |phi'| = 1, 1/3, 1/3, so Emb = 1 exactly (the minimiser is
        # below this; see the class docstring).
        chains = {"a": (("a", 1),), "b": (("b", 1),), "c": (("c", 1),)}
        e = E_p_p(chains, {"a": F(1), "b": F(3), "c": F(3)},
                  {"a": F(1), "b": F(1), "c": F(1)}, F(2))
        self.assertEqual(e.pow(2).exact_value(), F(1))


class TestEx117(unittest.TestCase):
    """[Elast] Example 1.17 (intro.tex, examp:emb), the minimiser, exactly.

    Data, as published and as drawn in energies.mp figures 104 and 11:
      G1 = tripod, elastic constants (1,3,3);  G2 = tripod, lengths (1,1,1);
      the minimiser sends the vertex of G1 to the point 2/5 of the way along
      an edge of G2, and Emb(phi) = 3/5.
    The subtlety the example is about: on the *unsubdivided* source the
      minimiser's derivative is not constant, but it IS constant-derivative in
      the paper's sense (lipschitz.tex, def:const-deriv: edge-reduced and
      |f'| constant on each edge of the source) -- because the source is
      subdivided at the preimages of the target's vertices.  Arms 2 and 3 of
      G1 (constant 3) are subdivided at elastic measure 5/3 from the leaf --
      the figure's mark_bar(z0--z2, 4/9) with the label $5/3$ (5/3 of a total
      3 is 5/9 of the arm, i.e. 4/9 from the centre); that breakpoint is the
      preimage of the target's central vertex.  Each piece then maps into a
      single target edge with constant derivative: the centre-side piece (4/3)
      maps to the inner 2/5 of G2's arm 1 at derivative 3/10; the leaf-side
      piece (5/3) maps to the whole of G2's arm 2 at derivative 3/5.  Arm 1
      (constant 1) maps to the outer 3/5 at derivative 3/5.  Every target edge
      then has Fill exactly 3/5.
    Both graphs are subdivided here so that every image is a chain of whole
    darts, which is what the engine's map model requires.
    Minimality is NOT asserted: that is item 170.  What is asserted is the
    exact energy of the published minimiser, and that the same map with the
    source left unsubdivided gives a different (larger) value.
    """

    # target: arm 1 split at p (2/5 from the centre); arms 2, 3 whole
    TGT = {"a1i": F(2, 5), "a1o": F(3, 5), "a2": F(1), "a3": F(1)}
    # source: arm 1 = 1; arms 2, 3 split at 5/3 from the leaf
    SRC_SUBDIVIDED = {"s1": F(1), "e2c": F(4, 3), "e2l": F(5, 3),
                     "e3c": F(4, 3), "e3l": F(5, 3)}

    def test_minimiser_has_emb_3_5(self):
        chains = {"s1": (("a1o", 1),),
                  "e2c": (("a1i", -1),), "e2l": (("a2", 1),),
                  "e3c": (("a1i", -1),), "e3l": (("a3", 1),)}
        fl = fills(chains, self.SRC_SUBDIVIDED, self.TGT, F(2))
        self.assertEqual(fl, {"a1i": F(3, 5), "a1o": F(3, 5),
                              "a2": F(3, 5), "a3": F(3, 5)})
        e = E_p_p(chains, self.SRC_SUBDIVIDED, self.TGT, F(2))
        self.assertEqual(e.pow(2).exact_value(), F(3, 5))       # Emb = 3/5
        self.assertAlmostEqual(float(e.value), (3 / 5) ** 0.5, places=12)

    def test_unsubdivided_reading_is_larger(self):
        # The same map with the source arms NOT subdivided: arm 2 maps onto
        # (inner 2/5 of arm 1) + (arm 2) as one chain, so |phi'| = 7/15 on it
        # and the inner edge carries 14/15.  This is the reading that is NOT
        # constant-derivative in the paper's sense (the derivative jumps inside
        # the arm); the subdivided map above is.
        src = {"s1": F(1), "s2": F(3), "s3": F(3)}
        chains = {"s1": (("a1o", 1),),
                  "s2": (("a1i", -1), ("a2", 1)),
                  "s3": (("a1i", -1), ("a3", 1))}
        e = E_p_p(chains, src, self.TGT, F(2))
        self.assertEqual(e.pow(2).exact_value(), F(14, 15))

    def test_swapping_the_piece_measures_is_detected(self):
        # If the subdivision point were the other way round (5/3 on the centre
        # side), the fill on the inner edge would not be 3/5.
        src = {"s1": F(1), "e2c": F(5, 3), "e2l": F(4, 3),
               "e3c": F(5, 3), "e3l": F(4, 3)}
        chains = {"s1": (("a1o", 1),),
                  "e2c": (("a1i", -1),), "e2l": (("a2", 1),),
                  "e3c": (("a1i", -1),), "e3l": (("a3", 1),)}
        e = E_p_p(chains, src, self.TGT, F(2))
        self.assertNotEqual(e.pow(2).exact_value(), F(3, 5))
