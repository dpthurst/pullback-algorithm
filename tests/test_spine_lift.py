"""Runnable controls for `spine_lift` (prototype; not production code).

Every case is built from `dense_assembly` alone: a triangulated base, a
prescribed local monodromy, `perms_from_local` and `assemble`.  No CEPS solve,
no Newton step, no position and no line search is involved anywhere.
"""
import itertools
import unittest

import numpy as np

from pullback import dense_assembly as A
from pullback import preimage_lift as PL
from tests.test_preimage_lift import OCTA, FACES, root_cover

from pullback import spine_lift as S

TETRA = [(0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)]
# regular octahedron, orient_outward: 4 = +z, 5 = -z, ring 0,2,1,3 ccw
OCTA = [(4, 0, 2), (5, 2, 0), (4, 2, 1), (5, 1, 2),
        (4, 1, 3), (5, 3, 1), (4, 3, 0), (5, 0, 3)]


def build(faces, d, local):
    base = A.base_from_face_lists(faces)
    perms = A.perms_from_local(base, d, local)
    lift = A.assemble(base, d, perms)
    return base, lift


def ident(d, verts):
    return {v: A.perm_identity(d) for v in verts}


def canonical_form(rose):
    """Isomorphism invariant of the labelled rose, up to sheet relabelling."""
    best = None
    for perm in itertools.permutations(range(rose.d)):
        rows = tuple(sorted((e.puncture, perm[e.sheet], perm[e.target],
                             len(e.labels)) for e in rose.edges))
        if best is None or rows < best:
            best = rows
    return best


class TestSpineBase(unittest.TestCase):
    def test_petals_are_the_flowers_and_the_basis_certificate(self):
        for faces, name in ((TETRA, "tetra"), (OCTA, "octa")):
            base = A.base_from_face_lists(faces)
            for punctures in (base.vertices, base.vertices[:2],
                              base.vertices[::2]):
                spine = S.rose_spine(base, punctures)
                rep = S.certify_rose(base, spine)
                self.assertTrue(rep["closed"], (name, punctures, rep))
                self.assertTrue(rep["at_vertex"], (name, punctures, rep))
                self.assertTrue(rep["sum_zero"], (name, punctures, rep))
                self.assertTrue(rep["basis_all_ok"], (name, punctures, rep))
                self.assertTrue(rep["rank"], (name, punctures, rep))
                self.assertTrue(rep["ok"], (name, punctures, rep))
                # petals are exactly the flower walks of the punctures
                for pet in spine.petals:
                    self.assertEqual(pet.wires(), tuple(base.flower(pet.puncture)))

    def test_grouping_is_read_off_the_edges_not_guessed(self):
        base = A.base_from_face_lists(TETRA)
        spine = S.rose_spine(base, (0, 1))
        rep = S.certify_rose(base, spine)
        # the only edge with both ends punctures, and only it, is crossed twice
        self.assertEqual(rep["crossed_twice"], 1)
        self.assertEqual(rep["grouping"][(0, 1)], (0, 1))
        self.assertEqual(rep["crossed_once"], 4)

    def test_rank_identity(self):
        base = A.base_from_face_lists(OCTA)
        for punctures in (base.vertices, (4, 5), (0, 1, 2), (2,)):
            spine = S.rose_spine(base, punctures)
            rep = S.certify_rose(base, spine)
            self.assertEqual(rep["rank_basis"], len(punctures) - 1)
            self.assertEqual(rep["rank_pi1_s2_minus_P"], len(punctures) - 1)


class TestAttach(unittest.TestCase):
    def test_degree_two_with_two_punctures(self):
        sw, id2 = (1, 0), A.perm_identity(2)
        local = {0: sw, 1: id2, 2: id2, 3: sw}
        base, lift = build(TETRA, 2, local)
        spine = S.rose_spine(base, (0, 3))
        att = S.attach_spine(spine, base, lift)
        self.assertTrue(att.report["ok"], att.report)
        self.assertEqual(att.report["loop_lengths"], {0: (6,), 3: (6,)})
        for p in (0, 3):
            self.assertEqual(att.report["preimages"][p],
                             tuple(sorted(lift["index"][p])))
        self.assertEqual(att.report["rh_deficit"], 2)   # 2d - 2 with d = 2
        loop = att.loops(spine.petal_of(0))[0]
        self.assertEqual(loop.sheets, (0, 1))
        self.assertEqual(len(loop.faces), 6)
        self.assertEqual(len(set(loop.labels)), 6)

    def test_degree_three_cycles_at_two_punctures(self):
        c3 = A.perm_from_cycles(3, [(0, 1, 2)])
        local = {0: c3, 1: A.perm_inverse(c3),
                 2: A.perm_identity(3), 3: A.perm_identity(3)}
        base, lift = build(TETRA, 3, local)
        spine = S.rose_spine(base, (0, 1))
        att = S.attach_spine(spine, base, lift)
        self.assertTrue(att.report["ok"], att.report)
        self.assertEqual(att.report["loop_lengths"], {0: (9,), 1: (9,)})
        # the path from sheet s at puncture 0 closes only when s is fixed
        for s in range(3):
            w = att.path(spine.petal_of(0), s)
            self.assertNotEqual(w.start_sheet, w.end_sheet)
            # a 3-cycle has one preimage point: the three sheets are the
            # three corners of one lifted vertex, so the path is a path
            # inside that vertex's star, not a loop
            self.assertEqual(w.start_vertex, w.end_vertex)
            self.assertEqual(len(att.loops(spine.petal_of(0))), 1)
        rose = att.to_rose(face0=0)                     # strict: basis petals
        self.assertTrue(rose.report["is_spine"], rose.report)
        self.assertEqual(rose.report["n_petals"], 1)    # |P| - 1 = 1
        self.assertEqual(rose.report["edges"], 3)       # d * (|P| - 1)
        self.assertEqual(rose.report["chi"], 3 * (2 - 2))
        self.assertEqual(rose.components(), 1)

    def test_degree_four_four_cycles(self):
        d = 4
        local = {0: (3, 0, 1, 2), 1: A.perm_identity(d),
                 2: (1, 2, 3, 0), 3: A.perm_identity(d)}
        base, lift = build(TETRA, d, local)
        spine = S.rose_spine(base, (0, 2))
        att = S.attach_spine(spine, base, lift)
        self.assertTrue(att.report["ok"], att.report)
        self.assertEqual(att.report["loop_lengths"], {0: (12,), 2: (12,)})
        self.assertEqual(att.report["rh_deficit"], 6)
        for p in (0, 2):
            self.assertEqual(att.report["preimages"][p],
                             tuple(sorted(lift["index"][p])))
        self.assertEqual(len(att.report["preimages"][0]), 1)
        self.assertEqual(len(att.report["preimages"][2]), 1)

    def test_three_punctures_and_the_basis_has_teeth(self):
        d = 3
        # consistent: the product of the local monodromies is the identity
        local = {0: (1, 0, 2), 1: (0, 2, 1), 2: (1, 2, 0),
                 3: A.perm_identity(d)}
        build(TETRA, d, local)          # accepted by perms_from_local
        base, lift = build(TETRA, d, local)
        spine = S.rose_spine(base, (0, 1, 2))
        att = S.attach_spine(spine, base, lift)
        self.assertTrue(att.report["ok"], att.report)
        # sigma_0 and sigma_1 are transpositions: two loops each, of lengths
        # |orbit| * |petal| = 6 and 3; sigma_2 is a 3-cycle: one loop of 9
        self.assertEqual(att.report["loop_lengths"], {0: (6, 3), 1: (3, 6),
                                                      2: (9,)})
        rose = att.to_rose(face0=0)
        self.assertTrue(rose.report["is_spine"], rose.report)
        self.assertEqual(rose.report["n_petals"], 2)
        self.assertEqual(rose.report["components"], 1)
        # dropping one more petal really does disconnect the Schreier graph:
        # the check is not vacuous
        only_first = [e for e in rose.edges if e.puncture == 0]
        self.assertEqual(rose.components(only_first), 2)

    def test_known_map_z_to_z_power(self):
        for d in (2, 3):
            cyc = tuple(range(d))
            local = ident(d, range(6))
            local[4] = A.perm_from_cycles(d, [tuple(reversed(cyc))])
            local[5] = A.perm_from_cycles(d, [cyc])
            base, lift = build(OCTA, d, local)
            spine = S.rose_spine(base, (4, 5))
            att = S.attach_spine(spine, base, lift)
            self.assertTrue(att.report["ok"], (d, att.report))
            # z -> z^d: one lifting over each branch point, of local degree d
            for p in (4, 5):
                self.assertEqual(att.report["preimages"][p],
                                 tuple(sorted(lift["index"][p])))
            self.assertEqual(len(att.report["preimages"][4]), 1)
            self.assertEqual(len(att.report["preimages"][5]), 1)
            self.assertEqual(att.report["loop_lengths"], {4: (4 * d,),
                                                          5: (4 * d,)})
            self.assertEqual(att.report["rh_deficit"], 2 * d - 2)
            # a one-petal rose: its preimage is the cycle decomposition of rho
            rose = att.to_rose(face0=0)
            self.assertTrue(rose.report["is_spine"], rose.report)
            self.assertEqual(rose.report["components"], 1)
            self.assertEqual(rose.report["edges"], d * 1)
            rho = rose.rho[4]
            seen, cycles = set(), []
            for s in range(d):
                if s in seen:
                    continue
                n, x = 0, s
                while x not in seen:
                    seen.add(x)
                    x = rho[x]
                    n += 1
                cycles.append(n)
            self.assertEqual(cycles, [d])

    def test_all_vertices_punctured(self):
        d = 4
        local = [(3, 0, 1, 2), (0, 1, 2, 3), (1, 2, 3, 0), (0, 1, 2, 3)]
        base, lift = build(TETRA, d, dict(zip(range(4), local)))
        spine = S.rose_spine(base)
        rep = S.certify_rose(base, spine)
        self.assertTrue(rep["ok"], rep)
        self.assertEqual(rep["rank_basis"], 3)
        att = S.attach_spine(spine, base, lift)
        self.assertTrue(att.report["ok"], att.report)
        rose = att.to_rose(face0=0)
        self.assertTrue(rose.report["is_spine"], rose.report)
        self.assertEqual(rose.report["vertices"], d)
        self.assertEqual(rose.report["edges"], d * 3)   # d * (|P| - 1)
        self.assertEqual(rose.report["chi"], d * (2 - 4))
        allrose = att.to_rose(face0=0, which="all")
        self.assertFalse(allrose.report["is_spine"])
        self.assertEqual(allrose.report["edges"], d * 4)
        self.assertEqual(allrose.report["chi"], d * (1 - 4))
        self.assertFalse(allrose.report["chi_ok"])


class TestEulerCharacteristic(unittest.TestCase):
    """The strict rose has |P| - 1 petals and the right Euler characteristic.

    Base rose: one vertex, |P| - 1 petals, chi = 2 - |P| = chi(S^2 - P).
    Its preimage: d vertices, d*(|P| - 1) edges, chi = d*(2 - |P|), which is
    also 2 - (number of lifted punctures), i.e. Riemann-Hurwitz written as
    Euler characteristics.  The all-petal record is deliberately NOT a spine
    and reports its own (wrong for the homotopy type) characteristic.
    """

    CASES = (
        ("tetra d=2 {0,3}", TETRA, 2,
         {0: (1, 0), 1: (0, 1), 2: (0, 1), 3: (1, 0)}, (0, 3)),
        ("tetra d=3 {0,1}", TETRA, 3,
         {0: (1, 2, 0), 1: (2, 0, 1), 2: (0, 1, 2), 3: (0, 1, 2)}, (0, 1)),
        ("tetra d=3 {0,1,2}", TETRA, 3,
         {0: (1, 0, 2), 1: (0, 2, 1), 2: (1, 2, 0), 3: (0, 1, 2)},
         (0, 1, 2)),
        ("tetra d=4 all", TETRA, 4,
         {0: (3, 0, 1, 2), 1: (0, 1, 2, 3), 2: (1, 2, 3, 0), 3: (0, 1, 2, 3)},
         (0, 1, 2, 3)),
        ("octa d=3 {4,5}", OCTA, 3,
         {0: (0, 1, 2), 1: (0, 1, 2), 2: (0, 1, 2), 3: (0, 1, 2),
          4: (2, 0, 1), 5: (1, 2, 0)}, (4, 5)),
        ("octa d=2 {4,5}", OCTA, 2,
         {0: (0, 1), 1: (0, 1), 2: (0, 1), 3: (0, 1),
          4: (1, 0), 5: (1, 0)}, (4, 5)),
    )

    def test_base_rose_and_lift(self):
        for name, faces, d, local, punct in self.CASES:
            base, lift = build(faces, d, local)
            self.assertTrue(lift["cert"]["connected"], name)
            n = len(punct)
            spine = S.rose_spine(base, punct)
            rep = S.certify_rose(base, spine)
            self.assertEqual(rep["rose_vertices"], 1, name)
            self.assertEqual(rep["rose_edges"], n - 1, name)
            self.assertEqual(rep["rose_chi"], 2 - n, name)
            self.assertTrue(rep["rose_chi_ok"], (name, rep))
            self.assertEqual(rep["redundant_rose_chi"], 1 - n, name)
            att = S.attach_spine(spine, base, lift)
            self.assertTrue(att.report["ok"], (name, att.report))
            self.assertEqual(att.report["chi_punctured_lift"], d * (2 - n), name)
            self.assertTrue(att.report["chi_agree"], (name, att.report))
            rose = att.to_rose(face0=0)
            r = rose.report
            self.assertTrue(r["is_spine"], (name, r))
            self.assertEqual(r["vertices"], d, name)
            self.assertEqual(r["edges"], d * (n - 1), name)
            self.assertEqual(r["chi"], d * (2 - n), name)
            self.assertTrue(r["chi_ok"], (name, r))
            self.assertEqual(r["components"], 1, name)
            self.assertEqual(r["rank"], d * (n - 2) + 1, name)
            self.assertTrue(r["rank_ok"], (name, r))
            # every sheet of every basis petal really is one rose edge
            self.assertEqual(len(r["edge_lengths"]), n - 1, name)
            self.assertEqual(sorted(len(r["edge_lengths"][p]) for p in
                                    r["edge_lengths"]), [d] * (n - 1), name)
            aux = att.to_rose(face0=0, which="all")
            self.assertFalse(aux.report["is_spine"], name)
            self.assertEqual(aux.report["edges"], d * n, name)
            self.assertEqual(aux.report["chi"], d * (1 - n), name)

    def test_wrong_rose_is_refused(self):
        base, lift = build(TETRA, 3, {0: (1, 0, 2), 1: (0, 2, 1),
                                      2: (1, 2, 0), 3: A.perm_identity(3)})
        spine = S.rose_spine(base, (0, 1, 2))
        att = S.attach_spine(spine, base, lift)
        aux = att.to_rose(face0=0, which="all")
        self.assertNotEqual(aux.report["chi"], aux.report["chi_expected"])
        with self.assertRaises(ValueError):
            att.to_rose(face0=0, which="petals")


class TestChecksHaveTeeth(unittest.TestCase):
    """A tampered petal must be refused, not silently accepted."""

    def lift(self):
        c3 = A.perm_from_cycles(3, [(0, 1, 2)])
        local = {0: c3, 1: A.perm_inverse(c3),
                 2: A.perm_identity(3), 3: A.perm_identity(3)}
        return build(TETRA, 3, local)

    def test_reversed_petal_is_refused(self):
        base, lift = self.lift()
        spine = S.rose_spine(base, (0, 1))
        good = spine.petals[0]
        bad = S.Petal(good.puncture, tuple(reversed(good.steps)))
        with self.assertRaises(ValueError):
            S.attach_spine(S.SpineRose(spine.punctures, (bad, spine.petals[1]),
                                       spine.dropped), base, lift)

    def test_dropped_step_is_refused(self):
        base, lift = self.lift()
        spine = S.rose_spine(base, (0, 1))
        good = spine.petals[0]
        bad = S.Petal(good.puncture, good.steps[:-1])
        with self.assertRaises(ValueError):
            S.attach_spine(S.SpineRose(spine.punctures, (bad, spine.petals[1]),
                                       spine.dropped), base, lift)

    def test_non_branching_petal_still_works(self):
        # a vertex with trivial local monodromy is d separate loops, one per
        # sheet; it is not an error, and that is the special-vertex criterion
        base, lift = self.lift()
        spine = S.rose_spine(base, (0, 1, 2))
        att = S.attach_spine(spine, base, lift)
        loops = att.loops(spine.petal_of(2))
        self.assertEqual(len(loops), 3)
        self.assertEqual(sorted(len(l.sheets) for l in loops), [1, 1, 1])
        self.assertEqual(sorted(l.length for l in loops),
                         [len(spine.petal_of(2).steps)] * 3)


class TestNotJustMultiplicities(unittest.TestCase):
    """Covers with identical fibre profiles whose preimages differ.

    Both covers are degree 4, connected, with every tetrahedron vertex a
    puncture, and both have the same fibre profile (2,1,1), (2,1,1), (2,1,1),
    (4) over vertices 0,1,2,3.  They are not conjugate in S_4, so they are not
    isomorphic covers, and their preimage roses are not isomorphic either.
    """

    D = 4
    A_LOCAL = {0: (0, 1, 3, 2), 1: (0, 2, 1, 3),
               2: (1, 0, 2, 3), 3: (1, 3, 0, 2)}
    B_LOCAL = {0: (0, 1, 3, 2), 1: (0, 2, 1, 3),
               2: (2, 1, 0, 3), 3: (2, 3, 1, 0)}

    def build(self, local):
        base, lift = build(TETRA, self.D, local)
        spine = S.rose_spine(base)
        return base, lift, S.attach_spine(spine, base, lift)

    def test_same_profiles_different_preimage(self):
        bA, lA, aA = self.build(self.A_LOCAL)
        bB, lB, aB = self.build(self.B_LOCAL)
        self.assertEqual(lA["cert"]["fibres"], lB["cert"]["fibres"])
        for p in range(4):
            self.assertEqual(sorted(aA.report["loop_lengths"][p]),
                             sorted(aB.report["loop_lengths"][p]))
        self.assertEqual(aA.report["rh_deficit"], aB.report["rh_deficit"])
        fA = canonical_form(aA.to_rose(face0=0))
        fB = canonical_form(aB.to_rose(face0=0))
        self.assertNotEqual(fA, fB)

    def test_conjugate_covers_have_isomorphic_preimages(self):
        # conjugating every local monodromy by the same sheet permutation is
        # an isomorphism of covers; the canonical form must not see it
        base, _lift = build(TETRA, self.D, self.A_LOCAL)
        tau = (2, 0, 3, 1)
        loc = {}
        for v, sig in self.B_LOCAL.items():
            inv = [0] * self.D
            for i, x in enumerate(tau):
                inv[x] = i
            loc[v] = tuple(tau[sig[inv[i]]] for i in range(self.D))
        _b1, _l1, a1 = self.build(self.B_LOCAL)
        _b2, _l2, a2 = self.build(loc)
        self.assertEqual(canonical_form(a1.to_rose(face0=0)),
                         canonical_form(a2.to_rose(face0=0)))

    def test_positions_attach_after_and_cannot_steer(self):
        for local in (self.A_LOCAL, self.B_LOCAL):
            base, lift, att = self.build(local)
            # the ids any position table must be keyed by are the assembled
            # fibre over the puncture -- the same ids uniformize_cover returns
            for p in att.spine.punctures:
                ids = sorted(u for u in lift["proj"] if lift["proj"][u] == p)
                self.assertEqual(ids, sorted(lift["index"][p]))
                self.assertEqual(ids, sorted(l.preimage for l in
                                             att.loops(att.spine.petal_of(p))))
            # no field of the record is numeric: labels are ints and tuples of
            # ints, and no float appears anywhere
            rose = att.to_rose(face0=0)
            for e in rose.edges:
                for x in e.labels:
                    self.assertIsInstance(x, tuple)
                    self.assertIsInstance(x[0], tuple)
                    self.assertIsInstance(x[1], int)


class TestStrongDatumControls(unittest.TestCase):
    """Full lifted paths and connectivity stay ahead of graph shape."""

    C1 = {(0, 1): (3, 1, 0, 2), (0, 2): (0, 2, 3, 1),
          (0, 3): (2, 0, 1, 3), (1, 2): (0, 3, 2, 1),
          (1, 3): (1, 3, 0, 2), (2, 3): (3, 2, 1, 0)}
    C2 = {(0, 1): (0, 1, 2, 3), (0, 2): (0, 2, 3, 1),
          (0, 3): (0, 3, 1, 2), (1, 2): (0, 3, 2, 1),
          (1, 3): (0, 3, 2, 1), (2, 3): (1, 3, 2, 0)}
    DISCONNECTED = {
        (0, 2): (5, 2, 3, 4, 1, 0), (1, 2): (1, 2, 4, 5, 3, 0),
        (0, 1): (5, 1, 3, 2, 0, 4), (1, 3): (0, 1, 4, 3, 5, 2),
        (0, 3): (4, 2, 3, 0, 5, 1), (2, 3): (0, 3, 4, 5, 1, 2)}

    def attach(self, datum):
        base = A.base_from_face_lists(TETRA)
        d = len(next(iter(datum.values())))
        lift = A.assemble(base, d, datum)
        spine = S.rose_spine(base)
        attachment = S.attach_spine(spine, base, lift)
        return base, lift, attachment, attachment.to_rose()

    def test_profile_twins_keep_different_full_label_paths(self):
        b1, l1, a1, r1 = self.attach(self.C1)
        b2, l2, a2, r2 = self.attach(self.C2)
        for key in ("fibres", "counts", "chi_lift", "rh_deficit", "rh_ok",
                    "ribbon_ok", "is_sphere", "cover_ok", "fibre_ok"):
            self.assertEqual(l1["cert"][key], l2["cert"][key], key)
        self.assertTrue(a1.report["ok"] and a2.report["ok"])
        self.assertEqual(r1.report["chi"], r2.report["chi"])
        self.assertEqual(r1.components(), r2.components())
        # The coarse graph shape may happen to agree. The actual full preimage
        # record, including the lifted label paths, is not collapsed to it.
        left = tuple((e.puncture, e.sheet, e.target, e.labels) for e in r1.edges)
        right = tuple((e.puncture, e.sheet, e.target, e.labels) for e in r2.edges)
        self.assertNotEqual(left, right)
        for rose, lift, attachment in ((r1, l1, a1), (r2, l2, a2)):
            for edge in rose.edges:
                for label in edge.labels:
                    self.assertIn(label, lift["edges"])
            self.assertEqual(rose.report["rank_expected"],
                             rose.d * (rose.d - 2) + 1)

    def test_local_transitivity_does_not_fool_the_rose_connectivity_check(self):
        base, lift, attachment, rose = self.attach(self.DISCONNECTED)
        local = A.local_perms(base, lift["edge_perms"], rose.d)
        seen, queue = {0}, [0]
        while queue:
            x = queue.pop()
            for permutation in local.values():
                y = permutation[x]
                if y not in seen:
                    seen.add(y)
                    queue.append(y)
        self.assertEqual(len(seen), rose.d)
        self.assertEqual(lift["cert"]["components"], 2)
        self.assertFalse(lift["cert"]["connected"])
        self.assertGreater(rose.components(), 1)
        self.assertFalse(attachment.report["ok"])


class TestCEPSAttachment(unittest.TestCase):
    def test_ceps_vertex_ids_are_the_loop_preimage_ids(self):
        d = 3
        lift, lengths = root_cover(d)
        base = A.base_from_face_lists(FACES)
        spine = S.rose_spine(base, (4, 5))
        attachment = S.attach_spine(spine, base, lift)
        attachment.to_rose()
        # No base positions, hence no exact two-branch shortcut: real CEPS.
        out = PL.uniformize_cover(lift, lengths, special_vertex=0)
        self.assertEqual(out["method"], "ceps")
        for petal in spine.petals:
            given = {u: xyz for u, _degree, xyz in out["preimages"][petal.puncture]}
            self.assertEqual(set(given), set(lift["index"][petal.puncture]))
            for loop in attachment.loops(petal):
                np.testing.assert_allclose(out["positions"][loop.preimage],
                                           given[loop.preimage], atol=0, rtol=0)
        self.assertEqual(out["special_vertex"], 0)  # chart-only, not branching


if __name__ == "__main__":
    unittest.main(verbosity=2)
