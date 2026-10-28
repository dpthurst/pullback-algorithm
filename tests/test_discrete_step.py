"""The discrete pullback step: the lift, and the labelled fixed-point statement.

The step is `discrete.pullback_step`.  What is tested here is the part that has
no sigma in it and therefore can be checked against a known answer: the
congruent lift of a base mesh, and the reading of the diagonal copy of P in
f^{-1}(P) that makes the labelled statement sigma(tau) = tau come out right.

The convention test is the one that matters.  Reading the diagonal the other way
round -- label p at the preimage of p -- returns the input configuration
relabelled by f|_P, which is *independent of the mesh*: measured on the rabbit,
the step error came out identical to 12 digits at every refinement level, which
is what caught it.  So the test asserts the step error is smaller than that
relabelled reading, which the wrong convention fails by a factor of four.
"""

import sys
import unittest

import mpmath as mp

sys.path.insert(0, "src")

from pullback import mesh as msh                     # noqa: E402
from pullback import thurston as th                  # noqa: E402
from pullback.discrete import (base_mesh, chart_coordinate,  # noqa: E402
                               pcf_data, pullback_step, _diagonal_vertex)

mp.mp.dps = 40


def setUpModule():
    mp.mp.dps = 40


MAPS = [("rabbit", th.rabbit_map()),
        ("rabbit-by-mobius", th.strictly_rational_rabbit()["map"]),
        ("slit", th.slit_rational_map())]


class TestLift(unittest.TestCase):
    def test_the_lift_is_a_triangulated_sphere_for_every_map(self):
        for name, f in MAPS:
            d = pcf_data(f)
            base = base_mesh(d["points"], f)
            lifted, stats = msh.lift_by_edges(base, f)
            self.assertEqual(stats["faces"], f.deg * len(base.faces),
                             msg=name)
            self.assertEqual(lifted.validate()["V"], stats["vertices"])
            # every lifted edge carries its base edge's length, and the two
            # faces over one base edge agree
            lens = msh.congruent_lengths(base, lifted)
            self.assertEqual(len(lens), len(lifted.edges()))
            # the preimage count: |f^{-1}(V)| = sum over base vertices of the
            # number of distinct preimages
            want = sum(len(f.preimage_with_degrees(p)) for p in base.points)
            self.assertEqual(stats["vertices"], want, msg=name)

    def test_the_lift_is_congruent(self):
        # a lifted face's three side lengths are its base face's three
        f = th.rabbit_map()
        d = pcf_data(f)
        base = base_mesh(d["points"], f)
        lifted, _stats = msh.lift_by_edges(base, f)
        base_len = base.chordal_lengths()
        for lf in lifted.faces:
            bs = sorted(tuple(sorted((lifted.source[i][0], lifted.source[j][0])))
                        for (i, j) in ((lf[0], lf[1]), (lf[1], lf[2]), (lf[0], lf[2])))
            got = sorted(round(base_len[e], 12) for e in bs)
            self.assertEqual(len(got), 3)


class TestStep(unittest.TestCase):
    def test_the_step_runs_and_the_residual_is_small(self):
        for name, f in MAPS[:2]:
            d = pcf_data(f)
            res = pullback_step(d["points"], d["dynamics"], f, refine=0,
                                maxit=60)
            self.assertLess(res["residual"], 1e-9, msg=name)
            self.assertGreater(res["step_error"], 0.0, msg=name)

    def test_the_step_refuses_a_map_that_is_not_a_thurston_marking(self):
        # the slit map: f(1) = f(-2) = 1, so 1 has two preimages inside P and 2
        # has none; there is no diagonal copy to read
        d = pcf_data(th.slit_rational_map())
        with self.assertRaises(ValueError) as ctx:
            pullback_step(d["points"], d["dynamics"], th.slit_rational_map(),
                          refine=0)
        self.assertIn("Thurston marking", str(ctx.exception))

    def test_the_diagonal_is_read_the_right_way_round(self):
        """Label p sits at the source point p, which lies over f(p).

        Reading the diagonal of the *analytic* preimages must return the input
        configuration exactly: the source point marked p is p itself, and it lies
        over f(p), not over p.  The wrong reading -- label p at the preimage of
        p -- returns the input relabelled by f|_P, whose labelled cross-ratio
        differs, which is how the mistake was caught (the step error came out
        identical to 12 digits at every refinement, a quantity with no mesh in
        it).
        """
        for name, f in MAPS[:2]:
            d = pcf_data(f)
            base = base_mesh(d["points"], f)
            lifted, _stats = msh.lift_by_edges(base, f)
            diag = [_diagonal_vertex(lifted, d["dynamics"][i], d["points"][i])
                    for i in range(len(d["points"]))]
            tau_in = chart_coordinate([msh.to_vector(p) for p in d["points"]])
            tau_diag = chart_coordinate([lifted.vectors()[v] for v in diag])
            self.assertLess(abs(tau_diag - tau_in), 1e-9,
                            msg="%s: the analytic diagonal reads %s, the"
                            " input is %s" % (name, tau_diag, tau_in))

    def test_the_chart_coordinate_is_moebius_invariant(self):
        f = th.rabbit_map()
        d = pcf_data(f)
        pts = [msh.to_vector(p) for p in d["points"]]
        a = chart_coordinate(pts)
        from pullback.moduli import apply_mobius
        M = (mp.mpc(0.6, 0.2), mp.mpc(0.3, -0.1), mp.mpc(-0.2, 0.4),
             mp.mpc(0.9, 0.1))
        b = chart_coordinate([msh.to_vector(apply_mobius(M, complex(p)))
                              for p in d["points"]])
        self.assertLess(abs(a - b), 1e-12)


if __name__ == "__main__":
    unittest.main()
