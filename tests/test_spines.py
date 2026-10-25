"""Spine changes (src/pullback/spines.py): the ribbon-preserving Whitehead
move, the general graph pullback, and the transport of a (G_0, G_1) pair.

The pullback is cross-checked against `iterate_graph` (G_{n+1} = G_1 x_{G_0}
G_n, built tuple by tuple there); the transport is cross-checked against the
library's barbell session (the author's own presentation of the same spine)
and by the invariant it must preserve: the minimised Emb of the class.
"""

import sys
import unittest
from fractions import Fraction as Q
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx

from pullback import covermin as cm
from pullback import spines as sp
from pullback.harmonic_io import load_session, parse_graph
from pullback.iterate_graphs import session_iterate
from pullback.refined import GraphMap, graph_from_ribbon


def cyclic_key(flower):
    """The lexicographically least rotation of a cyclic word of darts."""
    w = tuple(str(d) for d in flower)
    return min(w[i:] + w[:i] for i in range(len(w)))


def flower_multiset(g):
    return sorted(cyclic_key(ds) for ds in g.flowers.values())


def nx_multigraph(G):
    M = nx.MultiGraph()
    for e, (u, v) in G.edges.items():
        M.add_edge(u, v, label=e)
    return M


class TestWhiteheadMove(unittest.TestCase):
    """The theta graph on its middle edge gives exactly the barbell of the
    sessions (up to rotation of the flowers and the vertex labels), and the
    collapse/expansion maps are homotopy inverses."""

    def test_theta_to_barbell(self):
        theta = parse_graph("apqr bRQP")
        barbell = parse_graph("aqpP bQRr")
        R, c, moved, exp, exp2 = sp.whitehead_move(theta, "q")
        self.assertEqual(flower_multiset(moved), flower_multiset(barbell))
        # the intermediate graph is the rose with the two loops
        self.assertEqual(len(R.edge_list()), 2)
        self.assertEqual(len(R.vertices()), 1)

    def test_collapse_inverse(self):
        theta = parse_graph("apqr bRQP")
        R, c, moved, exp, exp2 = sp.whitehead_move(theta, "q")
        GR, G0 = graph_from_ribbon(R), graph_from_ribbon(theta)
        coll = GraphMap(G0, GR, {e: tuple(v) for e, v in c.items()})
        back = GraphMap(GR, G0, {e: tuple(v) for e, v in exp2.items()})
        comp = sp.compose(coll, back)
        for e in GR.edges:
            self.assertEqual(comp.chains[e], ((e, 1),), e)
        # the new edge's collapse composed with the loop-preserving expansion
        cp = GraphMap(graph_from_ribbon(moved), GR,
                      {**{e: ((e, 1),) for e in R.edge_list()}, "q": ()})
        expm = GraphMap(GR, graph_from_ribbon(moved),
                        {e: tuple(v) for e, v in exp.items()})
        comp2 = sp.compose(cp, expm)
        for e in GR.edges:
            self.assertEqual(comp2.chains[e], ((e, 1),), e)


class TestPullbackAgainstIterate(unittest.TestCase):
    """G_{n+1} = the pullback of the covering pi along phi^n_0."""

    def test_theta_and_bouquet(self):
        for name in ("theta", "bouquet"):
            base, cover, phi = load_session(ROOT / "sessions" / f"{name}.json")
            G2 = graph_from_ribbon(base)
            G1 = graph_from_ribbon(cover)
            pi = GraphMap(G1, G2, {e: ((e[0], 1),) for e in G1.edges})
            phim = GraphMap(G1, G2,
                            {e: tuple(phi.chains.get(e, ())) for e in G1.edges})
            P1, pc1, pa1 = sp.pullback(pi, phim)
            nextphi = sp.compose(phim, pc1)
            P2, pc2, pa2 = sp.pullback(pi, nextphi)
            for n, P in ((2, P1), (3, P2)):
                Gn, _phin, _ = session_iterate(
                    ROOT / "sessions" / f"{name}.json", n)
                with self.subTest(session=name, n=n):
                    self.assertEqual(len(P.deg), len(Gn.deg))
                    self.assertEqual(len(P.edges), len(Gn.edges))
                    self.assertTrue(nx.is_isomorphic(nx_multigraph(P),
                                                     nx_multigraph(Gn)))
            # the square commutes chain-wise: pi . proj_C = phi . proj_A
            for lab in P1.edges:
                ic = tuple((d[0][0], 1) for d in pc1.chains[lab])
                ia = phim.image_word(pa1.chains[lab])
                self.assertEqual(ic, ia, lab)


class TestTransport(unittest.TestCase):
    """Transporting the theta pair across the Whitehead move on q gives the
    barbell pair of the library (the author's own presentation), and the
    minimised Emb of the class is unchanged."""

    def setUp(self):
        self.base, self.cover, self.phi = load_session(
            ROOT / "sessions" / "theta.json")
        (self.moved, self.P, self.pi2, self.phi2, self.s,
         self.k) = sp.transport_pair(self.base, self.cover, self.phi.chains,
                                     "q", new_edge="q")

    def test_base_is_the_barbell(self):
        self.assertEqual(flower_multiset(self.moved),
                         flower_multiset(parse_graph("aqpP bQRr")))

    def test_cover_is_a_degree_two_covering(self):
        # 8 vertices over the 4 of the barbell, two per vertex
        self.assertEqual(len(self.P.deg), 8)
        over = {}
        for (c, a) in self.P.deg:
            over[a] = over.get(a, 0) + 1
        for a, n in over.items():
            self.assertEqual(n, 2, a)
        # every edge of the barbell has two lifts per refined piece
        cnt = {}
        for lab, ch in self.pi2.chains.items():
            self.assertTrue(ch)            # a covering has no constant edge
            cnt[ch[0][0]] = cnt.get(ch[0][0], 0) + 1
        self.assertEqual(cnt, {"p": 4, "r": 4, "q": 2})

    def test_map_matches_the_library_barbell(self):
        # the base edge and chain of every non-constant edge of the
        # transported map, as a multiset, is the library barbell's (the
        # sheets may be swapped)
        _b, _c, phib = load_session(ROOT / "sessions" / "barbell.json")
        here = sorted((str(self.pi2.chains[lab][0][0]), tuple(ch))
                      for lab, ch in self.phi2.chains.items() if ch)
        there = []
        for e, ch in phib.chains.items():
            if not ch:
                continue
            base = e[0] if isinstance(e, tuple) else e
            there.append((str(base), tuple(ch)))
        self.assertEqual(here, sorted(there))

    def test_engine_values_match_the_library_barbell(self):
        # the transported pair is the author's barbell pair: the exact QP's
        # certified lower bound and covermin's fixed-point lambda agree
        from pullback.minimise import Minimiser, minimise
        G2p = graph_from_ribbon(self.moved)
        lb_p = minimise(Minimiser(self.P, G2p, self.phi2.chains))["LB"]
        Gn, phin, G2 = session_iterate(ROOT / "sessions" / "barbell.json", 1)
        lb_b = minimise(Minimiser(Gn, G2, phin.chains))["LB"]
        self.assertEqual(lb_p, lb_b)

        def lam(P, phi2, G2):
            cls = cm.Cls(cm.G.from_refined(P),
                         {e: tuple(phi2.chains.get(e, ())) for e in P.edges},
                         cm.G.from_refined(G2), marked=set())
            lams, _ell, _c, _X, _g = cm.alg_iter(cls)
            return lams[-1]
        lam_p = lam(self.P, self.phi2, G2p)
        lam_b = lam(Gn, phin, G2)
        self.assertLessEqual(abs(float(lam_p - lam_b)), 1e-12)
        self.assertLessEqual(abs(float(lam_b - Q(3))), 1e-12)


if __name__ == "__main__":
    unittest.main()
