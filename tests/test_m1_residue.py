"""M1 residue tests: chain reduction, generator covers, detour
invariance, ribbon boundary fixtures."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pullback.harmonic_io import load_session, parse_graph, parse_cover, parse_cmap, _sheet_key
from pullback.covers import expand_generator_cover, check_generator_cover
from pullback.validate import validate
from pullback.portrait import face_map_pi, face_map_phi
from pullback.ribbons import F, B, rev, reduce_chain

SESSIONS = sorted((ROOT / "sessions").glob("*.json"))


class TestReduction(unittest.TestCase):
    def test_backtrack_reduces(self):
        base = parse_graph("apqr bRQP")
        cover, _ = parse_cover(base, "q01")
        a = parse_cmap(cover, base, "p0P p1r r0Q r1q")
        # q0 is constant in theta; adding a backtrack keeps it constant
        b = parse_cmap(cover, base, "p0P p1r r0Q r1q q0qQ")
        self.assertEqual({e: a.chains[e] for e in a.chains},
                         {e: b.chains[e] for e in b.chains})

    def test_cyclically_reduced_chain_kept(self):
        # q r Q is a closed chain; free reduction keeps it (no backtrack)
        ch = reduce_chain((("q", F), ("r", F), ("q", B)))
        self.assertEqual(ch, (("q", F), ("r", F), ("q", B)))


class TestGeneratorCover(unittest.TestCase):
    def test_expand_matches_per_edge(self):
        base = parse_graph("apqr bRQP")
        mono = expand_generator_cover(base, tree={"p"}, generators={"q": {0: 1, 1: 0}})
        self.assertEqual(mono, {"q": {0: 1, 1: 0}})
        self.assertEqual(check_generator_cover(base, {"p"}, mono), ["q", "r"])

    def test_perm_on_tree_edge_refused(self):
        base = parse_graph("apqr bRQP")
        with self.assertRaises(ValueError):
            expand_generator_cover(base, tree={"q"}, generators={"q": {0: 1, 1: 0}})


class TestInvariance(unittest.TestCase):
    def test_detour_invariance(self):
        """Appending a backtrack to a chain is a homotopy; every
        invariant must be unchanged."""
        for path in SESSIONS:
            name = path.stem
            with self.subTest(session=name):
                base, cover, phi = load_session(path)
                sess_phi = None
                import json
                sess = json.loads(path.read_text())
                sess_phi = sess["phi"]
                # append a backtrack to the first chain token, using the
                # first base-edge dart of that token's image
                tokens = sess_phi.split()
                toks2 = []
                changed = False
                for tok in tokens:
                    if len(tok) >= 3 and not changed:
                        d = tok[2]
                        toks2.append(tok + d + d.upper() if d.islower()
                                     else tok + d + d.lower())
                        changed = True
                    else:
                        toks2.append(tok)
                if not changed:
                    continue
                phi2 = parse_cmap(cover, base, " ".join(toks2))
                r1, _ = face_map_pi(cover, base)
                r2, _ = face_map_pi(cover, base)
                self.assertEqual(sorted(x["degree"] for x in r1),
                                 sorted(x["degree"] for x in r2))
                self.assertEqual(validate(phi, base), validate(phi2, base))
                # the phi walk pattern is homotopy invariant too
                p1, _ = face_map_phi(phi)
                p2, _ = face_map_phi(phi2)
                self.assertEqual(sorted(x["degree"] if x["degree"] else 0 for x in p1),
                                 sorted(x["degree"] if x["degree"] else 0 for x in p2))
                self.assertEqual(len([x for x in p1 if x["degree"] is None]),
                                 len([x for x in p2 if x["degree"] is None]))


class TestRibbonBoundary(unittest.TestCase):
    def test_fixture_matches(self):
        import json
        fix = json.loads((ROOT / "tests" / "library_fixtures.json").read_text())
        for path in SESSIONS:
            name = path.stem
            with self.subTest(session=name):
                base, cover, phi = load_session(path)
                rep, _ = face_map_phi(phi)
                ess = [r for r in rep if r["degree"] is not None]
                col = [r for r in rep if r["degree"] is None]
                self.assertEqual(len(ess), fix[name]["ribbon_boundary"]["essential"])
                self.assertEqual(len(col), fix[name]["ribbon_boundary"]["collapsed"])


class TestFaceCanonicalOrder(unittest.TestCase):
    """`RibbonGraph.faces()` must not depend on set iteration order.

    The earlier version started from `set.pop()`, so both the face order and
    each walk's starting dart changed with PYTHONHASHSEED between processes
    (measured 2026-10-25: `make m0`'s face table and `type_classification`'s
    face indices differed between seeds 0 and 42).  The first test compares
    two subprocesses; the second pins the canonical form directly.
    """

    CODE = (
        "import sys; sys.path.insert(0, 'src')\n"
        "from pullback.harmonic_io import parse_graph\n"
        "print(parse_graph('apqr bRQP').faces())\n"
        "print(parse_graph('xaAbBcC').faces())\n"
    )

    def test_same_across_hash_seeds(self):
        import os
        import subprocess
        outs = []
        for seed in ("0", "1"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            proc = subprocess.run([sys.executable, "-c", self.CODE],
                                  cwd=ROOT, env=env,
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            outs.append(proc.stdout)
        self.assertEqual(outs[0], outs[1])

    def test_each_walk_is_canonically_rotated_and_sorted(self):
        for text in ("apqr bRQP", "xaAbBcC"):
            faces = parse_graph(text).faces()
            keys = [tuple(str(d) for d in f) for f in faces]
            self.assertEqual(keys, sorted(keys), text)
            for f, k in zip(faces, keys):
                rots = [tuple(str(d) for d in f[i:] + f[:i])
                        for i in range(len(f))]
                self.assertEqual(k, min(rots), text)


if __name__ == "__main__":
    unittest.main()
