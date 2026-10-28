"""Strictly rational test data: a proper rational map, not a polynomial.

Every numeric example the engine had was a polynomial, and a polynomial's
infinity is fixed and totally ramified — in P, a critical point, its own critical
value.  That is a special geometry, and the mesh work of the pullback step is
exactly what it stresses.  Two examples fix it, both with exact data:

* `strictly_rational_rabbit` — the rabbit conjugated by M(z) = z/(z+1).  P has no
  point at infinity and the map is not a polynomial; the dynamics is a
  polynomial's, which is what makes it a *convergence target* with a known answer.
* `slit_rational_map` — (2 z^2 - 4)/z^2, exact rational coefficients, four
  postcritical points, neither critical point fixed, so no Möbius conjugation
  makes it a polynomial.  Its type is julia-4442's ("cycle without branch
  point"), so it is not a convergence target, but it is the strictly rational
  geometry test.

The checks here are the ones that catch a broken Möbius composition, which is a
live hazard:  the composition must factor correctly, or N(w) - x D(w) acquires a
root independent of x and every preimage list gets a spurious extra point.
"""

import sys
import unittest

import mpmath as mp

sys.path.insert(0, "src")

from pullback import thurston as th  # noqa: E402
from pullback.discrete import pcf_data  # noqa: E402
from pullback.moduli import apply_mobius  # noqa: E402
from pullback.rational import RationalMap, mobius_conjugate  # noqa: E402

mp.mp.dps = 40


def setUpModule():
    """Pin the working precision for this module.

    mpmath's precision is global and other test modules set it to their own
    value, so a module-level assignment at import time is not enough: the test
    that compares the derivative with a finite difference failed at 3.9e-12
    because a *previous* module had left the precision at 30 digits, where a
    step of 1e-20 loses ten digits to cancellation.
    """
    mp.mp.dps = 40


def cycles_of(dyn):
    """Lengths of the cycles of a functional graph given as a dict."""
    state, out = {}, []
    for start in dyn:
        if start in state:
            continue
        path, cur = [], start
        while cur not in state:
            state[cur] = "open"
            path.append(cur)
            cur = dyn[cur]
        if state.get(cur) == "open":
            out.append(len(path) - path.index(cur))
        for v in path:
            state[v] = "done"
    return sorted(out)


def by_position(z):
    z = complex(z)
    return (z.real, z.imag)




class TestRationalMapDerivative(unittest.TestCase):
    def test_derivative_matches_a_finite_difference(self):
        for f in (th.rabbit_map(), th.slit_rational_map()):
            for z in (mp.mpc(0.3, 0.4), mp.mpc(-1.1, 0.2), mp.mpc(2.0, -0.5)):
                h = mp.mpf("1e-20")
                fd = (f(z + h) - f(z - h)) / (2 * h)
                self.assertLess(abs(f.derivative(z) - fd), mp.mpf("1e-12"))

    def test_derivative_vanishes_at_a_critical_point(self):
        f = th.rabbit_map()
        self.assertLess(abs(f.derivative(mp.mpc(0))), mp.mpf("1e-25"))


class TestMoebiusConjugation(unittest.TestCase):
    def setUp(self):
        self.M = (mp.mpc(0.6, 0.2), mp.mpc(0.3, -0.1),
                  mp.mpc(-0.2, 0.4), mp.mpc(0.9, 0.1))

    def test_the_composition_identity(self):
        for f in (th.rabbit_map(), th.slit_rational_map()):
            g = mobius_conjugate(f, *self.M)
            for z in (mp.mpc(0), mp.mpc(0.7, 0.3), mp.mpc(-1.2, 0.6)):
                self.assertLess(abs(g(apply_mobius(self.M, z))
                                    - apply_mobius(self.M, f(z))), mp.mpf("1e-15"))

    def test_no_common_factor_is_left_behind(self):
        """The degree identity: over a generic point the local degrees sum to
        deg f, with no spurious root.

        This is the check that caught the composition bug: multiplying numerator
        and denominator by the same power of the chart denominator leaves a
        common factor, and then every preimage list carries an extra point whose
        local degree is 2.  It is invisible in the map's values.
        """
        for f in (th.rabbit_map(), th.slit_rational_map()):
            g = mobius_conjugate(f, *self.M)
            for x in (mp.mpc(0.41, 0.13), mp.mpc(-0.9, 0.2)):
                images = g.preimage_with_degrees(x)
                self.assertEqual(len(images), f.deg)
                self.assertEqual(sum(d for _q, d in images), f.deg)

    def test_conjugation_preserves_the_portrait_degrees(self):
        for f in (th.rabbit_map(), th.slit_rational_map()):
            g = mobius_conjugate(f, *self.M)
            self.assertEqual(g.portrait_degrees(), f.portrait_degrees())


class TestStrictlyRationalRabbit(unittest.TestCase):
    def setUp(self):
        self.data = th.strictly_rational_rabbit()
        self.f = self.data["map"]

    def test_it_is_not_a_polynomial(self):
        self.assertEqual(self.f.degN, self.f.degD)
        self.assertTrue(mp.isfinite(self.f.f_infinity()))

    def test_the_postcritical_set_has_four_finite_points(self):
        P = th.postcritical_list(self.f)
        self.assertEqual(len(P), 4)
        self.assertTrue(all(mp.isfinite(p) for p in P))

    def test_the_portrait_degrees_are_the_rabbits(self):
        self.assertEqual(self.f.portrait_degrees(), [1, 1, 1, 1, 2, 2])
        self.assertEqual(th.distinct_preimages(self.f, th.postcritical_list(self.f)), 6)

    def test_the_postcritical_set_is_the_transport_of_the_rabbits(self):
        # P(f) must be exactly M(P(rabbit)): the conjugation moves the geometry
        # and nothing else
        cc = self.data["parameter"]
        rabbit_P = [mp.mpc(0), cc, cc * cc + cc, mp.inf]
        # the stored Mobius map is (a, b, c, d) = (1, 0, 1, 1), i.e. z/(z+1),
        # which sends infinity to 1 and 0 to 0
        M = self.data["mobius"]
        transport = [apply_mobius(M, p) for p in rabbit_P]
        got = sorted((complex(p) for p in th.postcritical_list(self.f)), key=by_position)
        want = sorted((complex(p) for p in transport), key=by_position)
        for a, b in zip(got, want):
            self.assertLess(abs(a - b), 1e-12)

    def test_the_dynamics_is_a_four_point_permutation(self):
        d = pcf_data(self.f)
        self.assertEqual(len(d["dynamics"]), 4)
        self.assertEqual(sorted(d["dynamics"].values()), [0, 1, 2, 3])
        # the rabbit's cycle structure: a 3-cycle and a fixed point
        self.assertEqual(cycles_of(d["dynamics"]), [1, 3])


class TestSlitRationalMap(unittest.TestCase):
    def setUp(self):
        self.f = th.slit_rational_map()

    def test_it_is_strictly_rational(self):
        self.assertEqual(self.f.degN, self.f.degD)
        self.assertTrue(mp.isfinite(self.f.f_infinity()))
        # neither critical point is fixed, so no Moebius conjugation makes it a
        # polynomial
        self.assertEqual(len(self.f.critical_points()), 1)   # 0; infinity too
        self.assertNotEqual(self.f(mp.mpc(0)), mp.mpc(0))

    def test_the_postcritical_set_is_exactly_four_points(self):
        P = th.postcritical_list(self.f)
        self.assertEqual(len(P), 4)
        finite = sorted((complex(p) for p in P if mp.isfinite(p)), key=by_position)
        self.assertEqual(finite, [complex(-2), complex(1), complex(2)])
        self.assertEqual(sum(1 for p in P if not mp.isfinite(p)), 1)

    def test_the_portrait_degrees(self):
        self.assertEqual(self.f.portrait_degrees(), [1, 1, 1, 1, 2, 2])
        self.assertEqual(th.distinct_preimages(self.f, th.postcritical_list(self.f)), 6)

    def test_the_postcritical_dynamics_is_an_unbranched_cycle(self):
        # 0 -> infinity -> 2 -> 1 -> -2 -> 1: the cycle 1 <-> -2 carries no
        # critical point, which is julia-4442's recorded type
        d = pcf_data(self.f)
        # one 2-cycle (1 <-> -2) with the other two points feeding into it
        self.assertEqual(cycles_of(d["dynamics"]), [2])
        # the two cycles are {2, 1} and {-2, 1} in position; the critical
        # values are infinity and 2, and neither is on the 1 <-> -2 cycle
        pts = [complex(p) if mp.isfinite(p) else complex(float("inf"))
               for p in d["points"]]
        self.assertEqual(len(pts), 4)


if __name__ == "__main__":
    unittest.main()
