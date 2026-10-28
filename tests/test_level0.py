"""The level-0 map from the spine's dual, and its ribbon-preserving refinement.

The construction (`mesh.level0_map`): the spine-dual's edges join the pairs of
punctures the spine's edges separate -- a tree when the spine has one vertex, as
the rabbit's does -- and the level-0 mesh is a triangulation of the punctures at
their analytic positions that contains those tree edges, with the mirror class
chosen so the mesh's rotation agrees with the ribbon order under some marking.
`mesh.consistent_markings` is the check, and it discriminates: on the rabbit the
hull's mirror class admits no marking and the mirrored one admits 6.

Refinement is `RibbonMap.subdivide_faces`, a ribbon operation: the rotation at
the old vertices is preserved, so the ribbon order survives every level (tested
through four refinements, V = 4 to 164), and the dual is refined by the dual
operation.

One convention worth recording, traced on the tetrahedron: the face walk is
sigma of the *reversed* dart (not alpha of sigma), and the centre's flower is the
face's corner order *reversed* -- the same reversal as the dual's rotation.  The
other choices make the walk cross sectors and the faces come out nine long,
which `validate` catches.
"""

import sys
import unittest

import mpmath as mp

sys.path.insert(0, "src")

from pullback import mesh as msh                     # noqa: E402
from pullback import thurston as th                  # noqa: E402
from pullback.discrete import pcf_data              # noqa: E402
from pullback.session import Session                # noqa: E402

mp.mp.dps = 40


def setUpModule():
    mp.mp.dps = 40


class TestLevel0Map(unittest.TestCase):
    def setUp(self):
        self.session = Session.load("sessions/rabbit.json")
        self.spine = self.session.base
        self.f = th.rabbit_map()
        self.d = pcf_data(self.f)
        self.pts = [mp.mpc(p) if mp.isfinite(p) else mp.inf
                    for p in self.d["points"]]

    def test_the_level0_map_exists_and_is_consistent(self):
        mesh, marking = msh.level0_map(self.pts, self.spine)
        self.assertEqual((len(mesh), len(mesh.edges()), len(mesh.faces)),
                         (4, 6, 4))
        self.assertTrue(msh.marking_consistent_with_ribbon(mesh, self.spine,
                                                           marking))
        self.assertEqual(len(mesh.consistent_markings), 6)

    def test_it_contains_the_spine_duals_edges(self):
        mesh, marking = msh.level0_map(self.pts, self.spine)
        tree = msh.ribbon_order_of_spine(self.spine)
        tree_edges = {msh.edge_key(v, u) for v, ns in tree.items() for u in ns}
        self.assertTrue(tree_edges <= set(mesh.edges()))

    def test_the_mirror_class_is_chosen_by_the_check(self):
        """The hull's orientation admits no marking; the mirrored one admits 6.

        This is the measurement that makes the construction a construction and
        not a default: the two mirror classes of the tetrahedron have the same
        faces and the same edges and differ only in the rotation system, and the
        ribbon-order check -- which reads exactly that datum -- separates them.
        """
        tree = msh.ribbon_order_of_spine(self.spine)
        tree_edges = {msh.edge_key(v, u) for v, ns in tree.items() for u in ns}
        cands = msh.triangulations_containing(tree_edges, self.pts)
        self.assertEqual(len(cands), 2)
        counts = sorted(len(msh.consistent_markings(m, self.spine))
                        for m in cands)
        self.assertEqual(counts, [0, 6])
        mesh, _marking = msh.level0_map(self.pts, self.spine)
        counts2 = sorted(len(msh.consistent_markings(m, self.spine))
                         for m in cands)
        self.assertEqual(counts2, [0, 6])
        # the built mesh is the consistent one
        self.assertEqual(len(msh.consistent_markings(mesh, self.spine)), 6)

    def test_a_mismatched_point_set_is_refused(self):
        with self.assertRaises(ValueError):
            msh.level0_map(self.pts[:3], self.spine)


class TestRibbonRefinement(unittest.TestCase):
    def setUp(self):
        self.session = Session.load("sessions/rabbit.json")
        self.spine = self.session.base
        self.f = th.rabbit_map()
        self.d = pcf_data(self.f)
        self.pts = [mp.mpc(p) if mp.isfinite(p) else mp.inf
                    for p in self.d["points"]]
        self.mesh, self.marking = msh.level0_map(self.pts, self.spine)

    def test_refinement_preserves_the_ribbon_order(self):
        r = self.mesh.as_ribbon_map()
        sizes = [(len(r), len(r.faces()))]
        for _ in range(4):
            r = r.subdivide_faces()
            rm = msh.Mesh(r.points, r.faces())
            rm.validate()
            sizes.append((len(r), len(r.faces())))
            self.assertTrue(msh.marking_consistent_with_ribbon(
                rm, self.spine, self.marking),
                msg="ribbon order lost at V=%d" % len(r))
        self.assertEqual(sizes, [(4, 4), (8, 12), (20, 36), (56, 108),
                                 (164, 324)])

    def test_the_dual_is_refined_by_the_dual_operation(self):
        r = self.mesh.as_ribbon_map().subdivide_faces()
        d = r.dual()
        self.assertEqual(len(d), len(r.faces()))
        # the dual's faces are the stars of the primal vertices: the four
        # original punctures now have degree 6 (three tree edges interleaved
        # with three new ones), the four centres have degree 3
        self.assertEqual(sorted(len(f) for f in d.faces()),
                         [3, 3, 3, 3, 6, 6, 6, 6])
        # and the dual is itself a ribbon graph that refines
        d2 = d.subdivide_faces()
        self.assertTrue(msh.Mesh(d2.points, d2.faces()).validate())

    def test_subdivide_faces_refuses_a_degenerate_face(self):
        # a map whose face list visits a vertex twice is not a simple map, and
        # the operation says so rather than producing a broken mesh
        m = msh.RibbonMap([0.0, 1.0, 2.0], {0: [1], 1: [0, 2], 2: [1]})
        with self.assertRaises(ValueError):
            m.subdivide_faces()


if __name__ == "__main__":
    unittest.main()
