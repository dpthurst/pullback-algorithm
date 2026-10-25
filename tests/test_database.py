"""Comparison-free iteration invariants on the whole session database.

Every session in sessions/ (n = 1, the iteration capped at 12 steps) must
satisfy the invariants that need no comparison target -- the exact QP cannot
run on most sessions at n >= 2 and there are no published values:

  * the run completes, and the balance certificate holds exactly at the
    final harmonic map (every directional tension gap <= 0);
  * Dir(harmonic) <= Dir(the input representative), i.e. the harmonic step
    really minimises;
  * the multiplicities v >= 0, the factor lambda > 0;
  * lambda <= the explicit bound Emb(phi^0) = max_l (M 1)_l
    (thm:emb-sf: DR <= Emb), and covermin's own Emb of the input
    representative agrees with iterate_energy's matrix power at n = 1.

Two sessions are documented failures, not defects, and are pinned as such:

  * census-3.2 -- session_iterate raises "vertex ... has two images" (the
    exact engine fails identically, pre-existing);
  * census-53 -- the session carries irrational elastics (3+sqrt(105),
    outside the exact-rational engine, D5).

The full check also covered n = 2, the DR monotonicity and the twisted
sessions' limitations; see AGENTS.md (the twisted-z2i n = 1 DR decrease is a
documented exception -- the descent stalls at a start-dependent balanced
point, the free class has several).
"""

import sys
import unittest
from fractions import Fraction as Q
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pullback import covermin as cm
from pullback.harmonic_io import load_session
from pullback.iterate_graphs import session_iterate
from pullback.iterates import iterate_energy
from pullback.refined import graph_from_ribbon

SESSIONS = sorted(p.stem for p in (ROOT / "sessions").glob("*.json"))
KNOWN_FAILURES = {"census-3.2", "census-53"}


def session_elastics(path, G2):
    """{base edge: Fraction} or None (unit); ValueError on irrational data."""
    import json
    with open(path) as f:
        sess = json.load(f)
    elast = sess.get("elastics")
    if elast is None:
        return None
    return {e: Q(x) for e, x in zip(sorted(G2.edges, key=str), elast)}


class TestDatabaseIteration(unittest.TestCase):
    """The comparison-free invariants, session by session."""

    def test_all_sessions(self):
        checked = []
        for name in SESSIONS:
            path = ROOT / "sessions" / f"{name}.json"
            if name in KNOWN_FAILURES:
                continue
            with self.subTest(session=name):
                _b, _c, _p = load_session(path)
                elast = session_elastics(path, graph_from_ribbon(_b))
                Gn, phin, G2 = session_iterate(path, 1)
                if elast is not None:
                    for e in Gn.edges:
                        Gn.alpha[e] = elast[_base_letter(e)]
                    for e in G2.edges:
                        G2.alpha[e] = elast[e]
                cls = cm.Cls(cm.G.from_refined(Gn),
                             {e: tuple(phin.chains.get(e, ()))
                              for e in Gn.edges},
                             cm.G.from_refined(G2),
                             marked=set(Gn.marked))
                ell = {f: Q(1) for f in cls.tgt.labels()}
                dir_in = cls.energy(cls.X0, ell)
                dir_h, _did, _dr, v, X = cls.dr(ell)
                self.assertLessEqual(dir_h, dir_in)
                self.assertTrue(all(x >= 0 for x in v.values()))
                # the input representative's Emb: the piece model against
                # iterate_energy's matrix power
                emb_pieces = cls.emb(cls.X0, ell)
                _base, cover, phi = load_session(path)
                emb_matrix = iterate_energy(cover, phi, 1, Q(2),
                                            base_lengths=elast) \
                    .pow(2).exact_value()
                self.assertEqual(emb_pieces, emb_matrix)
                lams, _ell, _c, _X, gaps = cm.alg_iter(
                    cls, steps=12, tol=Q(1, 10**12))
                self.assertTrue(gaps)
                self.assertEqual([g for g in gaps if g[2] > 0], [])
                self.assertGreater(lams[-1], 0)
                self.assertLessEqual(lams[-1], emb_pieces)
                checked.append(name)
        self.assertGreaterEqual(len(checked), 12)

    def test_census_3_2_is_the_documented_failure(self):
        # session_iterate cannot refine it (a vertex has two images); the
        # exact engine fails identically -- pinned so a fix is noticed
        with self.assertRaises(ValueError) as ctx:
            session_iterate(ROOT / "sessions" / "census-3.2.json", 1)
        self.assertIn("two images", str(ctx.exception))

    def test_census_53_irrational_elastics(self):
        # the session's elastics are irrational; the exact-rational engine
        # cannot represent them (D5: no interval arithmetic) -- pinned
        path = ROOT / "sessions" / "census-53.json"
        _b, _c, _p = load_session(path)
        with self.assertRaises(ValueError) as ctx:
            session_elastics(path, graph_from_ribbon(_b))
        self.assertIn("sqrt", str(ctx.exception))


def _base_letter(e):
    """The base edge label of a G_n edge label."""
    lab = e[-1] if isinstance(e, tuple) else e
    while isinstance(lab, tuple) and not isinstance(lab[0], str):
        lab = lab[0]
    return lab[0] if isinstance(lab, tuple) else lab


if __name__ == "__main__":
    unittest.main()
