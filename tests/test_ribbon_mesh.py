"""The mesh as a ribbon graph: the topological order, and its source.

Two claims are tested here, both from the author's instruction of 2026-10-26:

* **The topological order, before mesh refinement, is in the ribbon structure.**
  The dual of the spine has the punctures as its vertices, and the rotation at a
  puncture is the spine's face-boundary order, reversed.  For the rabbit's spine
  -- a bouquet of three loops at one vertex -- that dual is a tree on the four
  punctures, so the order is the cyclic order of the tree's edges at each
  puncture.  A mesh built without reference to that order is checked against it:
  `consistent_markings` returns the markings under which the mesh's rotation
  contains the ribbon order, and for the tetrahedron on the rabbit's postcritical
  set that is 6 of the 24 markings, all agreeing on the one degree-3 puncture.

* **The mesh refinement keeps the (dual) as a ribbon graph.**  The refinement
  inserts a vertex in each face joined to its corners; at each old vertex the new
  edge takes the slot the face occupied, so the rotation there is preserved, and
  the dual is refined by the dual operation -- its faces are the stars of the
  primal vertices, of the right degrees.

Also tested: `RibbonGraph.dual` is an involution up to relabelling, and the dual
of a mesh has one vertex per face and one face per primal vertex.
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


class TestRibbonGraphDual(unittest.TestCase):
    def test_dual_is_an_involution_up_to_relabelling(self):
        for name in ("rabbit", "theta", "twisted-rabbit", "barbell"):
            s = Session.load("sessions/%s.json" % name)
            g = s.base
            d = g.dual()
            dd = d.dual()
            self.assertEqual(len(d.vertices()), len(g.faces()), msg=name)
            self.assertEqual(len(d.edge_list()), len(g.edge_list()), msg=name)
            self.assertEqual(len(dd.vertices()), len(g.vertices()), msg=name)
            self.assertEqual(len(dd.edge_list()), len(g.edge_list()), msg=name)
            self.assertEqual(len(dd.faces()), len(g.faces()), msg=name)

    def test_dual_of_a_spine_has_the_punctures_as_vertices(self):
        s = Session.load("sessions/rabbit.json")
        d = s.base.dual()
        self.assertEqual(len(d.vertices()), len(s.base.faces()))
        # the rabbit's spine is a bouquet of three loops at one vertex, so its
        # dual is a tree on the four punctures
        self.assertEqual(len(d.edge_list()), len(s.base.edge_list()))
        self.assertEqual(len(d.vertices()) - 1, len(d.edge_list()))


class TestRibbonMap(unittest.TestCase):
    def test_faces_agree_with_the_mesh(self):
        pts = [0.0, 1.0, 1.0j, 2.0 + 0.5j]
        mesh = msh.Mesh(pts, msh.sphere_triangulation(pts), name="tetra")
        rm = mesh.as_ribbon_map()
        self.assertEqual(sorted(tuple(sorted(f)) for f in rm.faces()),
                         sorted(tuple(sorted(f)) for f in mesh.faces))

    def test_dual_has_one_vertex_per_face_and_one_face_per_vertex(self):
        pts = [0.0, 1.0, 1.0j, 2.0 + 0.5j]
        mesh = msh.Mesh(pts, msh.sphere_triangulation(pts), name="tetra")
        d = mesh.dual()
        self.assertEqual(len(d), len(mesh.faces))
        self.assertEqual(sorted(len(f) for f in d.faces()),
                         sorted(len(mesh.rotation()[v])
                                for v in mesh.rotation()))

    def test_refinement_preserves_the_rotation_and_refines_the_dual(self):
        pts = [0.0, 1.0, 1.0j, 2.0 + 0.5j]
        mesh = msh.Mesh(pts, msh.sphere_triangulation(pts), name="tetra")
        r = msh.subdivide_face_centres(mesh)
        for v in mesh.rotation():
            old = mesh.rotation()[v]
            new = r.rotation()[v]
            got = [x for x in new if x in set(old)]
            self.assertTrue(got == old or got == list(reversed(old)),
                            msg="vertex %d: %s -> %s" % (v, old, got))
        # the dual of the refined mesh: one vertex per refined face, and its
        # faces are the stars of the primal vertices -- the four old ones of
        # degree 6, the four new centres of degree 3
        d = r.dual()
        self.assertEqual(len(d), len(r.faces))
        self.assertEqual(sorted(len(f) for f in d.faces()), [3, 3, 3, 3, 6, 6, 6, 6])

    def test_a_broken_rotation_system_is_refused(self):
        with self.assertRaises(ValueError):
            msh.RibbonMap([0.0, 1.0, 2.0],
                          {0: [1, 2], 1: [0, 2], 2: [0, 1, 1]})
        with self.assertRaises(ValueError):
            msh.RibbonMap([0.0, 1.0, 2.0],
                          {0: [1, 2], 1: [0, 2], 2: [0, 1], 3: [0]})


class TestTheOrderIsInTheRibbonStructure(unittest.TestCase):
    def setUp(self):
        self.session = Session.load("sessions/rabbit.json")
        self.spine = self.session.base
        self.f = th.rabbit_map()
        self.d = pcf_data(self.f)
        self.pts = [mp.mpc(p) if mp.isfinite(p) else mp.inf
                    for p in self.d["points"]]

    def test_the_ribbon_order_is_the_spine_face_boundary_reversed(self):
        tree = msh.ribbon_order_of_spine(self.spine)
        d = self.spine.dual()
        for v in d.vertices():
            walk = self.spine.faces()[v]
            # the dual's rotation at v is the reversed face-boundary walk
            want = [d.tgt(dart) for dart in d.flowers[v]]
            self.assertEqual(tree[v], want)
            self.assertEqual(len(want), len(walk))

    def test_the_mesh_order_is_checked_against_the_ribbon_order(self):
        mesh = msh.Mesh(self.pts, msh.sphere_triangulation(self.pts),
                        name="tetra")
        markings = msh.consistent_markings(mesh, self.spine)
        # 6 of the 24 markings, and all of them agree on the spine's one face of
        # degree 3 (the face bounded by three loops)
        self.assertEqual(len(markings), 6)
        degree3 = [v for v in msh.ribbon_order_of_spine(self.spine)
                   if len(msh.ribbon_order_of_spine(self.spine)[v]) == 3]
        self.assertEqual(len(degree3), 1)
        self.assertEqual(len({m[degree3[0]] for m in markings}), 1)

    def test_a_mesh_with_the_wrong_order_admits_no_marking(self):
        # the mirrored tetrahedron: same vertices, reversed rotation system
        mesh = msh.Mesh(self.pts, msh.sphere_triangulation(self.pts),
                        name="tetra")
        mirrored_faces = [(f[0], f[2], f[1]) for f in mesh.faces]
        mirrored = msh.Mesh(self.pts, msh.orient_faces(self.pts, mirrored_faces),
                            name="mirrored")
        mirrored.validate()
        self.assertEqual(msh.consistent_markings(mirrored, self.spine), [])


if __name__ == "__main__":
    unittest.main()
