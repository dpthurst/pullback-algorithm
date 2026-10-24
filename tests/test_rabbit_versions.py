"""Partial verification that three presentations of the rabbit are the
same virtual endomorphism.

The three phi strings on the rabbit's rose (graph "xaAbBcC", cover
"a01"):
  1. "b0a c0b a0AB a1cba"  -- the IMG package's machine
     (PoirierExamples(1) = PolynomialSphereMachine(2,[1/7]):
      f1 = sigma(f1^-1 f2^-1, f3 f2 f1), f2 = (f1, 1), f3 = (f2, 1));
  2. "b0a c0b a0c"         -- the author's cleaner presentation;
  3. "b0a c0b a1c"         -- its sheet-swapped mirror.

Check: at every level n up to LEVELS the images of the three state sets
in Sym(X^n) generate the SAME subgroup.  Equality of level-n images for
all n would prove the automata generate the same closure; a finite range
is a necessary condition and strong practical evidence, not a proof --
hence "partially verified".  (Deciding Thurston equivalence of two
machines is exactly the problem the IMG package's ThurstonAlgorithm
solves; this level check is a cheaper proxy.)
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sympy.combinatorics import Permutation, PermutationGroup

from pullback.harmonic_io import parse_graph, parse_cover, parse_cmap
from pullback.automaton import RoseAutomaton

LEVELS = range(1, 7)

VERSIONS = {
    "img": "b0a c0b a0AB a1cba",
    "a0c": "b0a c0b a0c",
    "a1c": "b0a c0b a1c",
}


def build(phi_string):
    base = parse_graph("xaAbBcC")
    cover, _ = parse_cover(base, "a01")
    phi = parse_cmap(cover, base, phi_string)
    return RoseAutomaton(base, cover, phi)


AUTOMATA = {name: build(s) for name, s in VERSIONS.items()}


def level_group(aut, n):
    gens = []
    for g, perm in aut.level_generators(n).items():
        gens.append(Permutation([perm[i] for i in range(len(perm))]))
    return PermutationGroup(gens)


class TestRabbitVersions(unittest.TestCase):
    def test_actions_are_permutations(self):
        for name, aut in AUTOMATA.items():
            for n in LEVELS:
                d = len(aut.sheets) ** n
                for g, perm in aut.level_generators(n).items():
                    with self.subTest(version=name, level=n, state=g):
                        self.assertEqual(sorted(perm.values()), list(range(d)))

    def test_level_groups_agree(self):
        for n in LEVELS:
            g = {name: level_group(aut, n) for name, aut in AUTOMATA.items()}
            with self.subTest(level=n):
                for other in ("a0c", "a1c"):
                    self.assertEqual(g["img"].order(), g[other].order(),
                                     f"orders differ at level {n}: img vs {other}")
                    for gen in g[other].generators:
                        self.assertIn(gen, g["img"],
                                      f"generator missing at level {n}: img vs {other}")


if __name__ == "__main__":
    unittest.main()
