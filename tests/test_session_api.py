"""The scripting facade (`pullback.Session`): known-answer checks.

Every value pinned here is a published or FR-verified number, not a
recording of what the engine happens to print:
- theta: [PChar] Fig. 3's (1+z^2)/(1-z^2), portrait degrees [1,1,2,2], the
  level orders 2,4,8,32,256,16384 (FR cross-check, plan item 150), one
  admissible tree {q};
- the refusals are real too: grigorchuk and julia-4442 have no ribbon
  presentation, and blown-up-lattes and census-53 have no admissible tree --
  the last two must still produce an automaton, by the fallback.
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pullback import Session, library  # noqa: E402

THETA = ROOT / "sessions" / "theta.json"


class TestThetaKnownValues(unittest.TestCase):
    def setUp(self):
        self.s = Session.load(THETA)

    def test_identity_and_validation(self):
        s = self.s
        self.assertEqual(s.name, "theta")
        self.assertEqual(s.degree, 2)
        self.assertEqual(s.validation(),
                         {"cover_connected": True, "base_connected": True,
                          "pi1_surjective": True})
        self.assertEqual(sorted(s.portrait()["degrees"]), [1, 1, 2, 2])

    def test_type(self):
        self.assertEqual(self.s.type_classification()["type"], "hyperbolic type")

    def test_automaton_tree_and_nucleus(self):
        info = self.s.automaton()
        self.assertEqual(info["tree"], ["q"])
        self.assertTrue(info["admissible"])
        self.assertEqual(tuple(info["automaton"].letters), ("p", "r"))
        self.assertEqual(len(info["automaton"].sheets), 2)
        nuc = self.s.nucleus()
        self.assertEqual(nuc["classes"], 5)
        self.assertTrue(nuc["closed"])

    def test_level_orders_match_fr(self):
        # FR cross-check (plan item 150; AGENTS.md A8): re-run, do not quote.
        self.assertEqual([self.s.level_order(n) for n in range(1, 7)],
                         [2, 4, 8, 32, 256, 16384])

    def test_energy_is_the_papers_unit_speed(self):
        # (1+z^2)/(1-z^2) is rational with Emb = 1; the bound must find it.
        rows = self.s.energy_bounds(p=2, n=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["E"], 1.0)
        self.assertEqual(rows[0]["E_root"], 1.0)


class TestLibraryAndRefusals(unittest.TestCase):
    def test_library_loads(self):
        lib = library()
        self.assertEqual(len(lib), 15)
        for name, s in lib.items():
            self.assertEqual(s.name, name)
            for key in ("graph", "cover", "phi"):
                self.assertIn(key, s.raw)

    def test_no_admissible_tree_falls_back(self):
        # Documented: the shortcut is not exact for these, so no admissible
        # tree exists; the automaton is still built, on a plain spanning tree.
        for name in ("blown-up-lattes", "census-53"):
            info = library()[name].automaton()
            self.assertFalse(info["admissible"], name)
            self.assertTrue(info["tree"], name)
            self.assertTrue(len(info["automaton"].letters) >= 1, name)

    def test_non_ribbon_sessions_refuse_the_type(self):
        lib = library()
        for name in ("grigorchuk", "julia-4442"):
            with self.assertRaises(ValueError, msg=name):
                lib[name].type_classification()


class TestBadInputIsReported(unittest.TestCase):
    def test_malformed_json_raises(self):
        scratch = ROOT / ".scratch" / "api-test"
        scratch.mkdir(parents=True, exist_ok=True)
        bad = scratch / "broken.json"
        bad.write_text('{"graph": "apqr bRQP", "cover": ')
        try:
            with self.assertRaises(json.JSONDecodeError):
                Session.load(bad)
        finally:
            bad.unlink()

    def test_duplicate_phi_token_raises(self):
        data = json.loads(THETA.read_text())
        data["phi"] = "p0P p0r r0Q r1q"          # p0 twice
        with self.assertRaises(ValueError):
            Session.from_dict(data)

    # (a disconnected cover is not expressible in this schema: the sheets
    # are declared by the monodromy itself, so any cover the parser accepts
    # is connected by construction)

    def test_inconsistent_cover_is_refused_at_parse(self):
        # With p acting as (0 1) too, phi's p0P and p1r disagree about the
        # image of the lifted p-edge's two ends.  Parseable strings,
        # impossible data -- refused before any check runs.
        data = json.loads(THETA.read_text())
        data["cover"] = "q01 p01"
        with self.assertRaises(ValueError):
            Session.from_dict(data)


if __name__ == "__main__":
    unittest.main()
