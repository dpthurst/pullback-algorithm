"""M2: type classification (item 7) and Schreier graphs."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pullback.harmonic_io import load_session
from pullback.normalform import normal_form
from pullback.classify import type_classification


def load(name):
    return load_session(ROOT / "sessions" / f"{name}.json")


class TestClassification(unittest.TestCase):
    def test_theta_hyperbolic(self):
        r = type_classification(*load("theta"))
        self.assertEqual(r["type"], "hyperbolic type")

    def test_rabbit_hyperbolic(self):
        r = type_classification(*load("rabbit"))
        self.assertEqual(r["type"], "hyperbolic type")

    def test_twisted_rabbit_hyperbolic(self):
        # author's correction 2026-10-25: the twisted rabbit is hR with h
        # fixing P, so its portrait is the rabbit's; hyperbolic.
        r = type_classification(*load("twisted-rabbit"))
        self.assertEqual(r["type"], "hyperbolic type")
        self.assertFalse(r["unbranched_cycles"])

    def test_twisted_z2i_unbranched_cycle(self):
        # the string as Harmonic wrote it; the author believes it is a twist
        # of z^2+i (whose critical orbit i -> -1+i <-> -i has no branch point)
        r = type_classification(*load("twisted-z2i"))
        self.assertEqual(r["type"], "cycle without branch point")
        self.assertTrue(r["unbranched_cycles"])

    def test_julia_4422_unbranched(self):
        r = type_classification(*load("julia-4422"))
        self.assertEqual(r["type"], "cycle without branch point")

    def test_barycentric_hyperbolic(self):
        r = type_classification(*load("barycentric"))
        self.assertEqual(r["type"], "hyperbolic type")

    def test_broken_phi_raises(self):
        # julia-4442 and grigorchuk fail the ribbon boundary pattern; the
        # classifier refuses them with a clear message
        for name in ("julia-4442", "grigorchuk"):
            with self.subTest(session=name):
                with self.assertRaises(ValueError):
                    type_classification(*load(name))


class TestSchreier(unittest.TestCase):
    def test_theta_level1(self):
        base, cover, phi = load("theta")
        aut = normal_form(base, cover, phi, tree={"q"})
        verts, edges = aut.schreier_graph(1)
        self.assertEqual(len(verts), 2)
        self.assertEqual(len(edges), 2 * len(aut.letters))
        # connected
        adj = {v: set() for v in verts}
        for u, g, w in edges:
            adj[u].add(w)
        seen = {verts[0]}
        frontier = [verts[0]]
        while frontier:
            x = frontier.pop()
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    frontier.append(y)
        self.assertEqual(seen, set(verts))

    def test_rabbit_level2_size(self):
        base, cover, phi = load("rabbit")
        aut = normal_form(base, cover, phi)
        verts, edges = aut.schreier_graph(2)
        self.assertEqual(len(verts), 4)
        self.assertEqual(len(edges), 4 * len(aut.letters))


if __name__ == "__main__":
    unittest.main()
