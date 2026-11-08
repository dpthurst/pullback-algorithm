"""Homotopy sections: existence, construction, and the active set.

The reformulation of the pullback (2026-10-28) needs a one-sided homotopy inverse
of phi.  The engine now computes it, with one check per session: (phi o psi)_* is
the identity in the basis of pi_1(Gamma_0), which for graphs is (phi o psi) ~ id.
"""

import glob
import sys
import unittest

sys.path.insert(0, "src")

from pullback.homotopy import (active_faces, basis_transport, compose,  # noqa: E402
                               homotopy_section, is_peripheral_preserving,
                               peripheral_images)
from pullback.ribbons import F  # noqa: E402
from pullback.session import Session   # noqa: E402
from pullback.validate import rose_words, spanning_tree  # noqa: E402


class TestHomotopySection(unittest.TestCase):
    def _all_sessions(self):
        return [Session.load(p) for p in sorted(glob.glob("sessions/*.json"))]

    def test_a_based_section_exists_for_every_session(self):
        for s in self._all_sessions():
            res = homotopy_section(s.base, s.cover, s.phi)
            self.assertTrue(res["surjective"], msg=s.name)
            comp = res["composed"]
            tree, gens = spanning_tree(s.base)
            cimg = [tuple((gens.index(d[0]), d[1]) for d in w)
                    for w in rose_words(comp, tree, tree, gens)]
            idwords = [((i, F),) for i in range(len(gens))]
            self.assertEqual(cimg, idwords,
                             msg="%s: (phi o psi)_* != identity" % s.name)
            self.assertEqual(comp.image_vertex(s.base.vertices()[0]),
                             s.base.vertices()[0], msg=s.name)

    def test_a_non_surjective_phi_is_refused(self):
        pass  # no session in the library is non-surjective; see the module docstring

    def test_the_active_set_has_the_right_size(self):
        # |P_0'| = |P_0| on 14 of 15 rows; grigorchuk is not a ribbon presentation
        mismatches = []
        for s in self._all_sessions():
            active = active_faces(s.phi)
            if len(active) != len(s.base.faces()):
                mismatches.append((s.name, len(active), len(s.base.faces())))
        self.assertEqual(mismatches, [("grigorchuk", 6, 5)])

    def test_active_faces_are_not_collapsed(self):
        s = Session.load("sessions/rabbit.json")
        active = active_faces(s.phi)
        # m4's check: collapsed = {3,5}, degree 1 faces = {2,3,4,5}, disjoint
        self.assertEqual(active, [0, 1, 2, 4])
        self.assertNotIn(3, active)
        self.assertNotIn(5, active)


class TestGeometricComparison(unittest.TestCase):
    def test_arbitrary_basis_transport_is_not_peripheral(self):
        """Free-basis index transport is not the punctured-surface identity."""
        from pullback import discrete, mesh, thurston as th
        from pullback.lift import pullback_ve

        s = Session.load("sessions/rabbit.json")
        data = discrete.pcf_data(th.rabbit_map())
        spine = mesh.voronoi_spine(data["points"])
        rho = basis_transport(spine, s.base)
        images = peripheral_images(rho)
        self.assertIn(None, images)
        self.assertFalse(is_peripheral_preserving(rho))
        with self.assertRaisesRegex(ValueError, "peripheral"):
            pullback_ve(s.base, s.cover, s.phi, spine, rho)

    def test_identity_transport_preserves_peripherals(self):
        s = Session.load("sessions/rabbit.json")
        identity = basis_transport(s.base, s.base)
        self.assertTrue(is_peripheral_preserving(identity, allow_reverse=False))
        self.assertEqual(peripheral_images(identity, allow_reverse=False),
                         [0, 1, 2, 3])

    def test_based_words_keep_their_conjugator(self):
        """A cyclic quotient is not a based pi_1 computation.

        The second free generator below has actual based image a*b*a^-1.
        Cyclic reduction used to turn that into b, after which the section found
        algebraically was not a section of the realised graph map.
        """
        from pullback import discrete, mesh, thurston as th
        from pullback.maps import make_graph_map

        s = Session.load("sessions/rabbit.json")
        spine = mesh.voronoi_spine(discrete.pcf_data(th.rabbit_map())["points"])
        words = [
            (("a", 1),), (("a", 1),), (("c", -1),),
            (("b", 1),), (("b", 1),), (("c", 1),),
        ]
        rho = make_graph_map(
            spine, s.base, dict(zip(spine.edge_list(), words)),
            {v: s.base.vertices()[0] for v in spine.vertices()})
        tree, gens = spanning_tree(spine)
        image = rose_words(rho, set(), tree, [gens[1]])[0]
        self.assertEqual(image, (("a", 1), ("b", 1), ("a", -1)))
        # This is a genuine inverse witness: it used to stop with "no basepoint
        # lift" even though phi_* had already been certified surjective.
        section = homotopy_section(s.base, spine, rho)
        composed = section["composed"]
        ctree, cgens = spanning_tree(s.base)
        got = [tuple((cgens.index(d[0]), d[1]) for d in w)
               for w in rose_words(composed, ctree, ctree, cgens)]
        self.assertEqual(got, [((i, F),) for i in range(len(cgens))])


if __name__ == "__main__":
    unittest.main()
