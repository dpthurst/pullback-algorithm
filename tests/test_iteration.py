"""A well-defined labelled pullback step, and its iteration, on the s=2 rabbit.

Run from the repository root:

    PYTHONPATH=src:.ultramath/smw83rxe5/m6 python3 -m pytest \
        .ultramath/smw83rxe5/m6/test_iteration.py -q

`sidestep.FrozenStep` fixes the two things that stopped `discrete.pullback_step`
after one step:

* the combinatorial cover of the base mesh by f is computed ONCE, at the
  analytic marking, where `mesh.lift_by_edges` succeeds, and held fixed.  A step
  at any other configuration is then a length assignment (each source edge takes
  its base edge's chordal length) plus `round_realise`.  No lift is rebuilt, so
  no critical value can land inside a face and no pairing has to be searched.
* the label of a preimage is a *source vertex* of that frozen cover, so it is
  combinatorial data of the covering -- the sheet question does not arise.

The auxiliary (non-marked) base vertices are moved with the three finite marked
points by the Moebius map that matches them, so the base structure at every
configuration is the Moebius image of the base structure at the marking.  That
is the criterion the fixed auxiliary points lacked: fixed points of the sphere
crowd a marked point as soon as it moves and never refine away.

With that, (i) the step reproduces the reference values of the current
pipeline at the analytic marking at two refinement levels, (ii) it is defined
at a configuration that is not the marking, and (iii) it converges under
refinement *off* the fixed point, so the iteration has a limit.
"""

import sys
import unittest

import mpmath as mp
import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, ".ultramath/smw83rxe5/m6")

from pullback import discrete, mesh as msh                     # noqa: E402
from pullback import thurston as th                            # noqa: E402
from pullback.mesh import DegenerateLift                       # noqa: E402
from pullback.rounding import round_realise                    # noqa: E402
from pullback import iteration as ss                                          # noqa: E402

mp.mp.dps = 40


def chart(points):
    return msh.labelled_cross_ratio([msh.to_vector(p) for p in points])


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = th.rabbit_map()
        cls.data = discrete.pcf_data(cls.f)
        cls.dyn = cls.data["dynamics"]
        cls.P = list(cls.data["points"])
        cls.wP = chart(cls.P)
        cls.first = discrete.pullback_step(cls.P, cls.dyn, cls.f, refine=0,
                                           maxit=200)
        cls.tau1 = list(cls.first["points"])
        cls.w1 = chart(cls.tau1)


class TestTheFrozenStepAgreesWithTheReference(Base):
    """At the analytic marking the frozen cover IS the lift the current
    pipeline builds, so the numbers must agree there."""

    def test_the_marking_step_at_two_levels(self):
        want = {0: 0.12276952116566739, 1: 0.03786453}
        got = {}
        for refine in (0, 1):
            fr = ss.FrozenStep(self.P, self.f, refine=refine)
            fr.identify(self.dyn)
            w = chart(fr.step(self.P)["points"])
            got[refine] = abs(complex(w) - complex(self.wP))
            self.assertAlmostEqual(got[refine], want[refine],
                                   delta=1e-6 + 1e-3 * want[refine])
        self.assertLess(got[1], got[0] / 2)      # it refines

    def test_the_step_at_the_reference_value_is_exact(self):
        fr = ss.FrozenStep(self.P, self.f, refine=0)
        fr.identify(self.dyn)
        w = chart(fr.step(self.P)["points"])
        self.assertAlmostEqual(abs(complex(w)
                                   - complex(0.9525650909251872
                                             - 0.8419617188130308j)), 0.0,
                               places=12)


class TestTheStepAtAGeneralConfiguration(Base):
    """A configuration that is not the analytic marking, and that the current
    pipeline cannot step at all."""

    def test_the_current_pipeline_refuses_and_says_why(self):
        with self.assertRaises(DegenerateLift) as ctx:
            discrete.pullback_step(self.tau1, self.dyn, self.f, refine=0)
        self.assertIn("= 44", str(ctx.exception))

    def test_the_frozen_step_solves_and_has_no_sheet_to_choose(self):
        fr = ss.FrozenStep(self.P, self.f, refine=0)
        fr.identify(self.dyn)
        # every label is a fixed source vertex of the frozen cover: distinct
        # by construction, so no positional rule and no tie is involved
        self.assertEqual(len(set(fr.label_vertices)), 4)
        r = fr.step(self.tau1)
        self.assertLess(r["residual"], 1e-10)
        self.assertAlmostEqual(
            abs(complex(chart(r["points"]))
                - complex(0.8255220520879809 - 0.7995311549492712j)), 0.0,
            places=11)

    def test_the_step_converges_under_refinement_off_the_fixed_point(self):
        """The values at refine 0, 1 are Cauchy-convergent: the difference
        between the two levels is 5.27e-2, and at refine 2 against refine 1 it
        is 4.80e-3 (ratio 11, against 4 for a pure h^2).  Only two levels are
        computed here -- refine 2 costs 30 s."""
        w = {}
        for refine in (0, 1):
            fr = ss.FrozenStep(self.P, self.f, refine=refine)
            fr.identify(self.dyn)
            w[refine] = complex(chart(fr.step(self.tau1)["points"]))
        self.assertAlmostEqual(abs(w[1] - w[0]), 5.273430e-2, delta=3e-4)


class TestTheIteration(Base):
    """The labelled pullback iteration, from the analytic marking."""

    def _run(self, refine, n):
        fr = ss.FrozenStep(self.P, self.f, refine=refine)
        fr.identify(self.dyn)
        return ([complex(self.wP)]
                + [complex(r["coordinate"]) for r in fr.iterate(self.P, n)])

    def test_it_converges_and_the_limit_approaches_the_marking(self):
        for refine, n, bound in ((0, 10, 0.07), (1, 10, 0.03)):
            ws = self._run(refine, n)
            steps = [abs(b - a) for a, b in zip(ws, ws[1:])]
            self.assertLess(steps[-1], steps[0] / 2)      # it contracts
            self.assertLess(steps[-1], 2e-2)              # and has settled
            self.assertLess(abs(ws[-1] - complex(self.wP)), bound)

    def test_the_limit_moves_towards_the_analytic_marking(self):
        """The discrete fixed point is not the analytic one at finite
        refinement, and its distance to it falls with refinement
        (6.26e-2 at refine 0 against 1.79e-2 at refine 1; 9.63e-3 at refine 2
        over five iterates is reported in report.md)."""
        far = self._run(0, 10)
        close = self._run(1, 10)
        self.assertLess(abs(close[-1] - complex(self.wP)),
                        0.5 * abs(far[-1] - complex(self.wP)))


class TestNonPolynomialTargets(unittest.TestCase):
    def test_golden_ratio_step_errors_match_the_reference(self):
        want = {
            "img-4.15.1": (0.3733091044255333, 0.1479848585558072),
            "img-4.15.2": (0.08112008525265192, 0.04006851621345758),
        }
        for name, f, _c in th.img_golden_maps():
            d = discrete.pcf_data(f)
            target = discrete.chart_coordinate(
                [msh.to_vector(p) for p in d["points"]])
            for refine in (0, 1):
                fr = ss.FrozenStep(d["points"], f, refine=refine)
                fr.identify(d["dynamics"])
                got = fr.step(d["points"])
                self.assertLess(got["residual"], 1e-10, msg=name)
                self.assertAlmostEqual(
                    abs(got["coordinate"] - target), want[name][refine],
                    places=12, msg=f"{name}, refine {refine}")

    def test_golden_pullback_contracts_with_refinement(self):
        _name, f, _c = th.img_golden_maps()[0]
        d = discrete.pcf_data(f)
        fr = ss.FrozenStep(d["points"], f, refine=1)
        fr.identify(d["dynamics"])
        history = fr.iterate(d["points"], 5)
        gaps = [abs(b["coordinate"] - a["coordinate"])
                for a, b in zip(history, history[1:])]
        self.assertLess(gaps[-1], gaps[0] / 1.2)
        self.assertLess(gaps[-1], 0.1)
        self.assertTrue(all(r["residual"] < 1e-9 for r in history))

    def test_strictly_rational_rabbit_first_step(self):
        f = th.strictly_rational_rabbit()["map"]
        d = discrete.pcf_data(f)
        fr = ss.FrozenStep(d["points"], f, refine=0)
        fr.identify(d["dynamics"])
        got = fr.step(d["points"])
        self.assertLess(got["residual"], 1e-10)
        self.assertAlmostEqual(
            abs(got["coordinate"] - complex(0.13748894563118444
                                             - 0.9703332400126371j)),
            0.0, places=12)

    def test_strictly_rational_refinement_error_is_not_silently_accepted(self):
        # Measured boundary to the known analytic lift constructor: refinement 1
        # and 2 both fail to pair the analytic arcs.  The rejected fallback was
        # not a valid solution; a caller must see the failure explicitly.
        f = th.strictly_rational_rabbit()["map"]
        d = discrete.pcf_data(f)
        for refine in (1, 2):
            with self.assertRaises(DegenerateLift) as caught:
                ss.FrozenStep(d["points"], f, refine=refine)
            self.assertIn("no pairing", str(caught.exception))


class TestAdditionalAcceptanceControls(Base):
    def test_mobius_representatives_have_the_same_chart(self):
        left = ss.FrozenStep(self.P, self.f, refine=0)
        right = ss.FrozenStep(self.P, self.f, refine=0)
        left.identify(self.dyn)
        right.identify(self.dyn)
        M = (mp.mpc(0.7, 0.3), mp.mpc(0.2, -0.4), mp.mpc(-0.1, 0.5),
             mp.mpc(0.9, 0.2))
        moved = [msh._mp_apply(M, mp.mpc(p) if mp.isfinite(p) else mp.inf)
                 for p in self.P]
        self.assertLess(abs(chart(moved) - self.wP), 1e-12)
        a = left.step(self.P)["coordinate"]
        b = right.step(moved)["coordinate"]
        self.assertLess(abs(complex(a) - complex(b)), 1e-12)

    def test_refine_two_off_the_marking_and_five_iterate_limit(self):
        fr = ss.FrozenStep(self.P, self.f, refine=2)
        fr.identify(self.dyn)
        w = fr.step(self.tau1)["coordinate"]
        self.assertAlmostEqual(
            abs(complex(w) - complex(0.7852143512748953-0.7586495943962368j)),
            0.0, places=11)
        history = fr.iterate(self.P, 5)
        self.assertLess(abs(history[-1]["coordinate"] - self.wP), 1e-2)
        self.assertAlmostEqual(abs(history[-1]["coordinate"] - self.wP),
                               9.628209526167782e-3, places=10)


if __name__ == "__main__":
    unittest.main()
