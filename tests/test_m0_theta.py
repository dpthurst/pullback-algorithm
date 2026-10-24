"""M0 test suite: the theta example against [PChar] Fig. 3."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pullback.ribbons import F, B
from pullback.portrait import face_map_pi, face_map_phi, tighten
from pullback.harmonic_io import load_session, parse_graph, parse_cover, parse_cmap

base, cover, phi = load_session(ROOT / "sessions" / "theta.json")


def as_set(faces):
    """Faces up to cyclic rotation."""
    out = set()
    for f in faces:
        f = tuple(f)
        n = len(f)
        out.add(min(f[i:] + f[:i] for i in range(n)))
    return out


class TestDataModel(unittest.TestCase):
    def test_base_counts(self):
        self.assertEqual(len(base.vertices()), 2)
        self.assertEqual(len(base.edge_list()), 3)

    def test_cover_counts(self):
        # degree-2 cover: twice the vertices, twice the edges
        self.assertEqual(len(cover.vertices()), 4)
        self.assertEqual(len(cover.edge_list()), 6)

    def test_base_faces(self):
        # three bigons: (p r), (q p), (r q)
        expect = as_set([
            (("p", F), ("r", B)),
            (("q", F), ("p", B)),
            (("r", F), ("q", B)),
        ])
        self.assertEqual(as_set(base.faces()), expect)

    def test_cover_faces(self):
        # two bigons (p,r) and two quadrilaterals
        faces = cover.faces()
        self.assertEqual(len(faces), 4)
        self.assertEqual(sorted(len(f) for f in faces), [2, 2, 4, 4])


class TestPortrait(unittest.TestCase):
    def test_degrees(self):
        report, _ = face_map_pi(cover, base)
        self.assertEqual(sorted(r["degree"] for r in report), [1, 1, 2, 2])

    def test_targets(self):
        report, base_walks = face_map_pi(cover, base)
        base_set = as_set(base_walks)
        for row in report:
            self.assertIsNotNone(row["target"], f"face {row['face']} missed")
        # every base face is covered
        hit = {min(base_walks[r["target"]][i:] + base_walks[r["target"]][:i]
                   for i in range(len(base_walks[r["target"]]))) for r in report}
        self.assertEqual(hit, base_set)

    def test_phi_embedding_pattern(self):
        # The ribbon signature of the theta data.  N Gamma_0 is a
        # three-holed sphere, N Gamma_1 a four-holed sphere; under a
        # ribbon map the four boundary circles map to the three
        # boundary circles of N Gamma_0, each exactly once, and the
        # fourth — the central quad, over the unmarked point 0 —
        # collapses to a contractible loop.  (Hole count forces one
        # collapsed component; cf. "collapsed" boundary components in
        # the proof of [PChar, thm spine-surf].)
        report, base_walks = face_map_phi(phi)
        essential = [r for r in report if r["degree"] is not None]
        collapsed = [r for r in report if r["degree"] is None]
        self.assertEqual(len(collapsed), 1)
        self.assertEqual(len(collapsed[0]["face"]), 4)
        self.assertEqual(sorted(r["degree"] for r in essential), [1, 1, 1])
        # three distinct base faces hit, i.e. all of them, once each
        targets = set()
        for r in essential:
            w = base_walks[r["target"]]
            targets.add(min(w[i:] + w[:i] for i in range(len(w))))
        self.assertEqual(len(targets), len(base_walks))


class TestMooreDiagram(unittest.TestCase):
    """Rose collapse of c = q: labels must match Fig. 3d."""

    def test_moore_labels(self):
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            parent[find(x)] = find(y)

        for e in cover.edge_list():
            if e[0] == "q":
                s, t = cover.edges[e]
                union(s, t)
                union(t, s)
        labels = {}
        for e in cover.edge_list():
            if e[0] == "q":
                continue
            s, t = cover.edges[e]
            rose = tighten(tuple(d for d in phi.chains[e] if d[0] != "q"))
            labels[e] = (find(s) == find(t), rose)  # (is_loop, class)
        # two a-edges with classes P (=A) and r (=b); two b-edges trivial
        classes = sorted(rose for _, rose in labels.values())
        self.assertEqual(classes, [(), (), (("p", B),), (("r", F),)])


if __name__ == "__main__":
    unittest.main()
