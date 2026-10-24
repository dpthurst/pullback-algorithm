"""M1 library suite: every session in sessions/ loads, validates, and
its portrait matches tests/library_fixtures.json; plus the degree-sum
law (over each base face the degrees sum to the cover degree)."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pullback.harmonic_io import load_session, parse_cover, parse_cmap
from pullback.validate import validate
from pullback.portrait import face_map_pi

FIXTURES = json.loads((ROOT / "tests" / "library_fixtures.json").read_text())
SESSIONS = sorted((ROOT / "sessions").glob("*.json"))


def cover_degrees(rows):
    return sorted(r["degree"] for r in rows if r["degree"] is not None)


class TestLibrary(unittest.TestCase):
    def test_sessions_present(self):
        names = {p.stem for p in SESSIONS}
        self.assertEqual(names, set(FIXTURES),
                         "sessions/ and fixtures disagree")

    def test_validate_and_portrait(self):
        for path in SESSIONS:
            name = path.stem
            with self.subTest(session=name):
                base, cover, phi = load_session(path)
                rep = validate(phi, base)
                want = FIXTURES[name]
                self.assertEqual(rep, want["validation"], f"{name} validation")
                rows, _ = face_map_pi(cover, base)
                self.assertEqual(cover_degrees(rows),
                                 want["portrait_degrees"],
                                 f"{name} portrait degrees")
                bad = [r for r in rows if r["degree"] is None]
                self.assertEqual(len(bad),
                                 len(want["portrait_nonface_projections"]),
                                 f"{name} non-face projections")

    def test_spines_are_planar(self):
        """Every spine must be a sphere spine (chi = V-E+F = 2)."""
        known_nonplanar = set()  # empty: julia-4422 fixed 2026-10-24 (flowers reversed)
        for name, want in FIXTURES.items():
            with self.subTest(session=name):
                chi = want["spine_euler"]
                if name in known_nonplanar:
                    self.assertNotEqual(chi, 2)
                else:
                    self.assertEqual(chi, 2, f"{name} spine is not a sphere spine")
                    self.assertEqual(want["cover_euler"], 2,
                                     f"{name} cover is not a sphere")

    def test_mirror_tracking(self):
        """Global reversal (author, 2026-10-24): mirroring gives the
        complex-conjugate map; both are valid, and sessions record which
        orientation they use so comparisons are made in the same one.
        Checks: double mirror is the identity; the canonical-orientation
        flag flips; the portrait degree multiset is orientation-blind;
        and the recorded fixture flag matches the computation."""
        import json
        for path in SESSIONS:
            name = path.stem
            with self.subTest(session=name):
                sess = json.loads(path.read_text())
                base, cover, phi = load_session(path)
                mir = base.mirror()
                self.assertEqual(mir.mirror().serialization(), base.serialization())
                _, flag_self = base.canonical_orientation()
                _, flag_mir = mir.canonical_orientation()
                if base.serialization() != mir.serialization():
                    self.assertNotEqual(flag_self, flag_mir)
                self.assertEqual(flag_self, FIXTURES[name]["spine_mirrored"],
                                 f"{name}: recorded orientation flag")
                # same monodromy and phi on the mirrored spine
                cov2, _ = parse_cover(mir, sess["cover"])
                phi2 = parse_cmap(cov2, mir, sess["phi"])
                rows, _ = face_map_pi(cov2, mir)
                self.assertEqual(cover_degrees(rows),
                                 FIXTURES[name]["portrait_degrees"],
                                 f"{name}: portrait degrees under the mirror")

    def test_degree_sum_law(self):
        """Over each base face, the degrees of the faces above it sum to
        the degree of the cover."""
        for path in SESSIONS:
            name = path.stem
            with self.subTest(session=name):
                base, cover, phi = load_session(path)
                rows, base_walks = face_map_pi(cover, base)
                nverts = len(base.vertices())
                d = len(cover.vertices()) // nverts
                per_face = {}
                for r in rows:
                    if r["degree"] is not None:
                        per_face[r["target"]] = per_face.get(r["target"], 0) + r["degree"]
                for i, total in per_face.items():
                    self.assertEqual(total, d, f"{name} face {i}")


if __name__ == "__main__":
    unittest.main()
