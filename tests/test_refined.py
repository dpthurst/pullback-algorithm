"""M3 minimiser: refined maps and exact Emb of representatives (plan 160/170).

Every check has an input on which it fails (agent-craft §3).
"""

import sys
import unittest
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pullback.refined import (Graph, GraphMap, RefinedMap, subdivide,
                              piece_label, emb_of_map, same_class,
                              session_map, free_reduce, inv)


def tripod(alpha, name):
    """A tripod with centre 'c' and leaves p1, p2, p3 (the marked points)."""
    edges = {f"a{i + 1}": ("c", f"p{i + 1}") for i in range(3)}
    return Graph(edges, alpha, marked=["p1", "p2", "p3"], name=name)


def ex117_map():
    """[Elast] intro.tex examp:emb: G1 tripod elastics (1,3,3), G2 tripod
    lengths (1,1,1), phi the vertex-to-vertex map arms to arms."""
    G1 = tripod({"a1": F(1), "a2": F(3), "a3": F(3)}, "G1")
    G2 = tripod(F(1), "G2")
    return GraphMap(G1, G2,
                    {"a1": (("a1", 1),), "a2": (("a2", 1),), "a3": (("a3", 1),)},
                    name="phi")


def ex117_witness():
    """The published minimiser of Ex. 1.17: the centre goes 2/5 along
    target arm 1, so source arms 2, 3 are split at elastic measure 5/3 from
    the leaf (4/3 from the centre) and target arm 1 at 2/5."""
    base = ex117_map()
    return RefinedMap(
        base,
        src_splits={"a2": [F(4, 3), F(5, 3)], "a3": [F(4, 3), F(5, 3)]},
        tgt_splits={"a1": [F(2, 5), F(3, 5)]},
        chains={piece_label("a1", 0): ((piece_label("a1", 1), 1),),
                piece_label("a2", 0): ((piece_label("a1", 0), -1),),
                piece_label("a2", 1): ((piece_label("a2", 0), 1),),
                piece_label("a3", 0): ((piece_label("a1", 0), -1),),
                piece_label("a3", 1): ((piece_label("a3", 0), 1),)},
        name="psi")


class TestRefinedMap(unittest.TestCase):
    """The piece model: exact Fill on cells, the 3/5 vs 14/15 trap, and the
    class certificate."""

    def test_published_minimiser_is_3_5(self):
        self.assertEqual(ex117_witness().emb(), F(3, 5))

    def test_unrefined_representative_is_1(self):
        # the vertex-to-vertex map has Emb = 1, not 3/5: the trap the piece
        # model exists to avoid (AGENTS.md "M3 readiness")
        self.assertEqual(ex117_map().emb(), F(1))

    def test_in_class(self):
        self.assertTrue(ex117_witness().in_class())
        # a witness with a permuted marked image is not in the class
        bad = RefinedMap(
            ex117_map(),
            src_splits={"a2": [F(4, 3), F(5, 3)], "a3": [F(4, 3), F(5, 3)]},
            tgt_splits={"a1": [F(2, 5), F(3, 5)]},
            chains={piece_label("a1", 0): ((piece_label("a1", 1), 1),),
                    piece_label("a2", 0): ((piece_label("a1", 0), -1),),
                    piece_label("a2", 1): ((piece_label("a3", 0), 1),),  # wrong arm
                    piece_label("a3", 0): ((piece_label("a1", 0), -1),),
                    piece_label("a3", 1): ((piece_label("a3", 0), 1),)})
        self.assertFalse(bad.in_class())

    def test_collapse_round_trips(self):
        # forgetting the subdivisions recovers the witness's own base-level
        # chains: a1 -> a1, a2 -> a1^-1 a2, a3 -> a1^-1 a3 (the centre is at
        # 2/5 along arm a1, not at the vertex)
        w = ex117_witness()
        self.assertEqual(w.collapse(),
                         {"a1": (("a1", 1),),
                          "a2": (("a1", -1), ("a2", 1)),
                          "a3": (("a1", -1), ("a3", 1))})
        # the collapsed map's marked images agree with phi's (the centre c
        # maps into arm a1, which GraphMap cannot represent -- that is the
        # point of the piece model -- so the class check is in_class(), above)
        base = ex117_map()
        self.assertEqual(w.rmap.vmap["p1"], base.vmap["p1"])
        self.assertEqual(w.rmap.vmap["p2"], base.vmap["p2"])
        self.assertEqual(w.rmap.vmap["p3"], base.vmap["p3"])

    def test_same_class(self):
        self.assertTrue(same_class(ex117_map(), ex117_map()))
        other = GraphMap(ex117_map().src, ex117_map().tgt,
                         {"a1": (("a2", 1),), "a2": (("a1", 1),), "a3": (("a3", 1),)})
        self.assertFalse(same_class(ex117_map(), other))

    def test_subdivide_rejects_bad_splits(self):
        G = tripod(F(1), "G")
        with self.assertRaises(ValueError):
            subdivide(G, {"a1": [F(1, 2)]})          # does not sum to alpha
        with self.assertRaises(ValueError):
            subdivide(G, {"a1": [F(2), F(-1)]})      # negative piece

    def test_free_reduce(self):
        self.assertEqual(free_reduce((("a", 1), ("a", -1), ("b", 1))),
                         (("b", 1),))
        self.assertEqual(inv(("a", 1)), ("a", -1))


class TestSessionBridge(unittest.TestCase):
    """Sessions load as refined-graph maps; marked pins the lifts."""

    def test_barbell_loads(self):
        G1, m, G2 = session_map(ROOT / "sessions" / "barbell.json")
        self.assertEqual(len(G1.edges), 6)     # the degree-2 cover
        self.assertEqual(len(G2.edges), 3)
        self.assertEqual(m.emb(), F(6))        # the pinned explicit Fill
        self.assertEqual(G1.marked, frozenset())   # free class by default

    def test_marked_pins_the_lifts(self):
        G1, m, G2 = session_map(ROOT / "sessions" / "barbell.json",
                                marked_base=["a"])
        lifts = {u for u in G1.deg if u[0] == "a"}
        self.assertEqual(G1.marked, frozenset(lifts))
        self.assertEqual(len(lifts), 2)

    def test_marked_unknown_vertex_rejected(self):
        with self.assertRaises(ValueError):
            session_map(ROOT / "sessions" / "barbell.json",
                        marked_base=["zzz"])


class TestSessionFields(unittest.TestCase):
    """The session schema: marked and orbifold (reserved) validate."""

    def test_all_sessions_load(self):
        from pullback.harmonic_io import load_session
        for p in sorted((ROOT / "sessions").glob("*.json")):
            base, cover, phi = load_session(p)
            self.assertTrue(base.vertices())

    def test_marked_must_be_vertex_names(self):
        from pullback.harmonic_io import check_session_fields
        with self.assertRaises(ValueError):
            check_session_fields({"marked": [1, 2]})

    def test_orbifold_cone_point_shape(self):
        from pullback.harmonic_io import check_session_fields
        with self.assertRaises(ValueError):
            check_session_fields({"orbifold": [{"vertex": "a", "order": 1}]})
        with self.assertRaises(ValueError):
            check_session_fields({"orbifold": [{"vertex": "a"}]})
        # marked and orbifold coexist
        check_session_fields({"marked": ["a"],
                              "orbifold": [{"vertex": "b", "order": 2}]})

    def test_sessions_carry_the_fields(self):
        import json
        for p in sorted((ROOT / "sessions").glob("*.json")):
            sess = json.load(open(p))
            self.assertIn("marked", sess)
            self.assertIn("orbifold", sess)
            self.assertEqual(sess["marked"], [])
            self.assertEqual(sess["orbifold"], [])


if __name__ == "__main__":
    unittest.main()
