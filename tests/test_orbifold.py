"""Tests for `orbifold.py`: the exact dynamical orbifold and the Lattes tests.

Run either way:

    python3 test_orbifold.py              # the suite
    python3 test_orbifold.py --table      # the computed table, as Markdown

They need the repository but not to live in it: `_repo_root()` finds it from
this file's location, or from $PULLBACK_REPO.  Dropped in as
`tests/test_orbifold.py` it works unchanged.

What is pinned, and why it can be: the values are exact integers and
Fractions of them, so there is no tolerance to argue about.  Each library
value below was also checked by hand from the portrait (degree, Riemann-
Hurwitz, the lcm recursion and the infinite-order rule); `notes-orbifold.md`
carries the workings for the ones that are not obvious.
"""

import os
import sys
import unittest
from fractions import Fraction
from math import gcd, inf

HERE = os.path.dirname(os.path.abspath(__file__))


def _repo_root():
    root = os.environ.get("PULLBACK_REPO")
    if root:
        return root
    cand = HERE
    for _ in range(8):
        if os.path.isdir(os.path.join(cand, "src", "pullback")) \
                and os.path.isdir(os.path.join(cand, "sessions")):
            return cand
        cand = os.path.dirname(cand)
    raise RuntimeError("cannot find the pullback repository; set PULLBACK_REPO")


REPO = _repo_root()
for path in (os.path.join(REPO, "src"), HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from pullback import Session                                  # noqa: E402
from pullback.classify import type_classification             # noqa: E402
from pullback.portrait import face_map_phi                     # noqa: E402


def _orbifold_module():
    """`pullback.orbifold` if installed, else the copy sitting next to this file.

    The module is written as part of the package (it imports `.portrait`), so
    the copy in this directory is loaded under the package's name rather than
    as a top-level module.
    """
    local = os.path.join(HERE, "orbifold.py")
    if os.path.exists(local):
        import importlib.util
        spec = importlib.util.spec_from_file_location("pullback.orbifold", local)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["pullback.orbifold"] = mod
        spec.loader.exec_module(mod)
        return mod
    from pullback import orbifold
    return orbifold


O = _orbifold_module()

SESSION = os.path.join(REPO, "sessions")


def load(name):
    return Session.load(os.path.join(SESSION, name + ".json"))


def _lcm(*values):
    out = 1
    for v in values:
        out = out * v // gcd(out, v)
    return out


def computed(name):
    s = load(name)
    p = O.portrait(s.base, s.cover, s.phi)
    orb = O.orbifold(p)
    return p, orb, O.lattes_report(p, orb)


#: Every session whose phi marks exactly one cover face per base face, with
#: |P|, degree, orders, signature, chi and kind.
TABLE = {
    # degree 2, one critical point on a 2-cycle, one simple critical
    # preimage over the third puncture.
    "theta":           (3, 2, (inf, inf, 2), (2, inf, inf), Fraction(-1, 2)),
    "barbell":         (3, 2, (inf, inf, 2), (2, inf, inf), Fraction(-1, 2)),
    "bouquet":         (3, 2, (inf, inf, 2), (2, inf, inf), Fraction(-1, 2)),
    # degree 6 and every puncture critical, one of them fixed: orders 3 and 2
    # come from the critical preimages of the other two.  chi = -1/6, the
    # closest any shipped session comes to the parabolic case.
    "barycentric":     (3, 6, (inf, 3, 2), (2, 3, inf), Fraction(-1, 6)),
    "blown-up-lattes": (3, 3, (3, inf, 6), (3, 6, inf), Fraction(-1, 2)),
    "census-53":       (3, 3, (inf, 3, 6), (3, 6, inf), Fraction(-1, 2)),
    "slit-lattes":     (3, 3, (inf, 3, 6), (3, 6, inf), Fraction(-1, 2)),
    "rabbit":          (4, 2, (inf, inf, inf, inf), (inf,) * 4, Fraction(-2)),
    "twisted-rabbit":  (4, 2, (inf, inf, inf, inf), (inf,) * 4, Fraction(-2)),
    "tetrahedral":     (4, 3, (inf, inf, inf, inf), (inf,) * 4, Fraction(-2)),
    "census-3.2":      (4, 2, (inf, inf, 2, inf), (2, inf, inf, inf),
                        Fraction(-3, 2)),
    # the two sessions the repository calls "cycle without branch point";
    # both have a hyperbolic orbifold, and julia-4422's (4,4,2,2) is the
    # label in its own JSON.
    "julia-4422":      (4, 2, (4, 2, 4, 2), (2, 2, 4, 4), Fraction(-1, 2)),
    "twisted-z2i":     (4, 2, (2, inf, 2, 2), (2, 2, 2, inf), Fraction(-1, 2)),
}

#: Sessions whose data does not determine the marking.
UNMARKED = ("grigorchuk", "julia-4442")


class TestPortrait(unittest.TestCase):
    """The portrait, and the marking it needs."""

    def test_the_portrait_is_the_one_classify_reads(self):
        for name in TABLE:
            s = load(name)
            p = O.portrait(s.base, s.cover, s.phi)
            tc = type_classification(s.base, s.cover, s.phi)
            self.assertEqual(p["dynamics"], tc["dynamics"], name)
            self.assertEqual(
                {C: d >= 2 for C, d in p["local_degree"].items()},
                tc["critical"], name)

    def test_the_marking_is_a_bijection_onto_marked_faces(self):
        for name in TABLE:
            p, _, _ = computed(name)
            self.assertEqual(sorted(p["marked"]), list(range(p["n"])), name)
            self.assertEqual(len(set(p["marked"].values())), p["n"], name)
            for C, face in p["marked"].items():
                hits = [d for (d, y) in p["preimages"][p["dynamics"][C]] if y == C]
                self.assertEqual(hits, [p["local_degree"][C]], name)

    def test_the_three_arithmetic_certificates(self):
        for name in TABLE:
            p, _, _ = computed(name)
            c = p["certificates"]
            n, d = p["n"], p["d"]
            self.assertTrue(c["degree_constant_over_P"], name)
            self.assertEqual(c["sum_deg_minus_one"], 2 * d - 2, name)   # R-H
            self.assertEqual(c["n_preimages"], d * (n - 2) + 2, name)   # cover
            self.assertTrue(c["riemann_hurwitz"] and c["preimage_count"], name)

    def test_the_marking_is_refused_where_it_is_not_determined(self):
        for name in UNMARKED:
            s = load(name)
            with self.assertRaises(O.OrbifoldError) as ctx:
                O.portrait(s.base, s.cover, s.phi)
            self.assertIn("marked", str(ctx.exception), name)

    def test_the_witnesses_for_the_refusals(self):
        # grigorchuk: base face 0 is wrapped twice and base face 1 not at
        # all.  julia-4442: nothing is wrapped.
        rows, _ = face_map_phi(load("grigorchuk").phi)
        over = {}
        for r in rows:
            if r["target"] is not None:
                over[r["target"]] = over.get(r["target"], 0) + 1
        self.assertEqual(over, {0: 2, 2: 1, 3: 1, 4: 1})
        rows, _ = face_map_phi(load("julia-4442").phi)
        self.assertEqual([r for r in rows if r["target"] is not None], [])

    def test_a_missing_critical_value_is_caught(self):
        # z^2+i with P = {0, i, -1+i, -i} forgets the critical fixed point
        # infinity, and the Riemann-Hurwitz certificate notices: only one of
        # the two critical points lies over P.
        pre = {0: [(1, None), (1, None)], 1: [(2, 0)],
               2: [(1, 1), (1, 3)], 3: [(1, 2), (1, None)]}
        p = O.portrait_data(4, {0: 1, 1: 2, 2: 3, 3: 2},
                            {0: 2, 1: 1, 2: 1, 3: 1}, pre)
        self.assertEqual(p["certificates"]["sum_deg_minus_one"], 1)
        self.assertFalse(p["certificates"]["riemann_hurwitz"])
        with self.assertRaises(O.OrbifoldError):
            O.portrait_data(4, {0: 1, 1: 2, 2: 3, 3: 2},
                            {0: 2, 1: 1, 2: 1, 3: 1}, pre, check=True)

    def test_the_marking_decides_the_answer(self):
        # Why the refusal above is load-bearing and not pedantry: fix the
        # *cover* data of julia-4442 and vary only which cover face each
        # puncture is (the marking).  Every choice is a consistent portrait
        # (the degree and preimage-count certificates hold for all of them),
        # and the answers range from a flexible Lattes signature to a
        # hyperbolic one.  The cover data alone cannot decide.
        import itertools
        from pullback.portrait import face_map_pi
        s = load("julia-4442")
        prows, bwalks = face_map_pi(s.cover, s.base)
        n, targets = len(bwalks), [r["target"] for r in prows]
        degrees = [r["degree"] for r in prows]
        got = {}
        for chosen in itertools.permutations(range(len(prows)), n):
            inv = {f: C for C, f in enumerate(chosen)}
            pre = {}
            for i, t in enumerate(targets):
                pre.setdefault(t, []).append((degrees[i], inv.get(i)))
            p = O.portrait_data(
                n, {C: targets[chosen[C]] for C in range(n)},
                {C: degrees[chosen[C]] for C in range(n)}, pre, check=True)
            orb = O.orbifold(p)
            got.setdefault((orb["signature"], orb["chi"]), 0)
            got[(orb["signature"], orb["chi"])] += 1
        # a flexible Lattes signature and a hyperbolic one from one dataset
        self.assertIn(((2, 2, 2, 2), Fraction(0)), got)
        self.assertIn(((inf, inf, inf, inf), Fraction(-2)), got)
        # and the label in the session's own `source` is among them, which is
        # a third, independent confirmation of the convention
        self.assertIn(((2, 4, 4, 4), Fraction(-3, 4)), got)
        self.assertGreaterEqual(len(got), 8)


class TestOrbifold(unittest.TestCase):
    """The exact orders and Euler characteristic, pinned."""

    def test_the_library_table(self):
        for name, (n, d, orders, sig, chi) in TABLE.items():
            p, orb, _ = computed(name)
            self.assertEqual(p["n"], n, name)
            self.assertEqual(p["d"], d, name)
            self.assertEqual(tuple(orb["orders"][x] for x in range(n)), orders, name)
            self.assertEqual(orb["signature"], sig, name)
            self.assertEqual(orb["chi"], chi, name)
            self.assertEqual(orb["kind"], "hyperbolic", name)

    def test_the_euler_identity(self):
        # chi = 4 - n - chi_punctured, exactly, with 1/inf = 0.
        for name in TABLE:
            _, orb, _ = computed(name)
            self.assertEqual(orb["chi"],
                             Fraction(4 - orb["n"]) - orb["chi_punctured"], name)
            self.assertTrue(orb["certificates"]["euler_identity"], name)

    def test_types_of_the_reported_numbers(self):
        for name in TABLE:
            _, orb, _ = computed(name)
            self.assertIsInstance(orb["chi"], Fraction, name)
            self.assertIsInstance(orb["chi_punctured"], Fraction, name)
            for v in orb["orders"].values():
                self.assertTrue(v is inf or (isinstance(v, int) and v >= 1),
                                f"{name}: {v!r}")

    def test_julia_4422_recovers_the_authors_own_label(self):
        _, orb, _ = computed("julia-4422")
        self.assertEqual(orb["signature"], (2, 2, 4, 4))
        self.assertEqual(orb["orders"], {0: 4, 1: 2, 2: 4, 3: 2})
        self.assertEqual(orb["chi"], Fraction(-1, 2))

    def test_an_infinite_order_is_exactly_a_critical_cycle(self):
        for name in TABLE:
            p, orb, _ = computed(name)
            expect = set()
            for cyc in orb["cycles"]:
                if any(p["local_degree"][x] >= 2 for x in cyc):
                    expect.update(cyc)
            self.assertEqual(set(orb["infinite"]), expect, name)

    def test_the_barycentric_near_miss(self):
        # (2,3,inf) with chi = -1/6.  Give the fixed critical point the
        # finite order 6 it would need and chi is exactly 0: this session is
        # one critical cycle away from the parabolic (2,3,6) signature.
        _, orb, _ = computed("barycentric")
        self.assertEqual(orb["signature"], (2, 3, inf))
        self.assertEqual(orb["chi"], Fraction(-1, 6))
        as_if = sorted(6 if v is inf else v for v in orb["orders"].values())
        self.assertEqual(as_if, [2, 3, 6])
        self.assertEqual(2 - sum(1 - Fraction(1, v) for v in as_if), 0)
        self.assertIn((2, 3, 6), O.RIGID_LATTES)

    def test_the_two_notions_of_hyperbolic_differ(self):
        # [PChar]'s hypothesis is "each cycle in P contains a critical point"
        # ([Thurston20:Characterize], quoted in intro.tex).  It is strictly
        # stronger than chi < 0 on this library, and these are the witnesses.
        for name in ("julia-4422", "twisted-z2i"):
            s = load(name)
            tc = type_classification(s.base, s.cover, s.phi)
            _, orb, _ = computed(name)
            self.assertEqual(tc["type"], "cycle without branch point", name)
            self.assertLess(orb["chi"], 0, name)
            self.assertFalse(orb["all_cycles_branched"], name)
        # the direction that does hold on every shipped session
        for name in TABLE:
            s = load(name)
            if type_classification(s.base, s.cover, s.phi)["type"] == "hyperbolic type":
                _, orb, _ = computed(name)
                self.assertLess(orb["chi"], 0, name)

    def test_local_degree_and_orbifold_order_are_different_numbers(self):
        # The confusion this module exists to prevent.
        p, orb, _ = computed("julia-4422")
        self.assertEqual(tuple(p["local_degree"][x] for x in range(4)),
                         (1, 2, 1, 1))
        self.assertEqual(tuple(orb["orders"][x] for x in range(4)),
                         (4, 2, 4, 2))
        self.assertEqual(max(p["local_degree"].values()), 2)

    def test_no_shipped_session_has_a_spherical_orbifold(self):
        # [DH] prove chi <= 0 for a critically finite branched mapping; a
        # positive chi here would mean a portrait whose P misses a critical
        # value, which the Riemann-Hurwitz certificate already refuses.
        for name in TABLE:
            _, orb, _ = computed(name)
            self.assertLessEqual(orb["chi"], 0, name)


class TestHandPortraits(unittest.TestCase):
    """Portraits whose orbifold is known from the literature."""

    @staticmethod
    def lattes_even():
        # The Weierstrass model at n = 2: f(wp z) = wp(2z) has degree 4, its
        # critical points are the six points over the 2-torsion that are not
        # themselves 2-torsion (all simple), f|_P is the constant map to one
        # point, and P is the four 2-torsion values.
        pre = {3: [(1, 0), (1, 1), (1, 2), (1, 3)],
               0: [(2, None), (2, None)],
               1: [(2, None), (2, None)],
               2: [(2, None), (2, None)]}
        return O.portrait_data(4, {x: 3 for x in range(4)},
                               {x: 1 for x in range(4)}, pre, check=True)

    @staticmethod
    def lattes_odd():
        # The same at n = 3: degree 9, sixteen simple critical points, four
        # over each postcritical point, and f|_P the identity.
        pre = {x: [(1, x)] + [(2, None)] * 4 for x in range(4)}
        return O.portrait_data(4, {x: x for x in range(4)},
                               {x: 1 for x in range(4)}, pre, check=True)

    def test_the_four_point_lattes_models(self):
        # The Weierstrass construction gives a Lattes map for every torus and
        # every integer n >= 2; two of them are built here.  Both are the
        # four-point Lattes *portrait*.  Neither is certified flexible: that
        # needs the affine derivative, whose torus-lift matrix is not computed
        # here.  Both degrees are squares, so `flexible` must remain
        # "undetermined" and `rigid` must be None rather than False.
        for p, tag in ((self.lattes_even(), "n=2, d=4"),
                       (self.lattes_odd(), "n=3, d=9")):
            orb = O.orbifold(p)
            lat = O.lattes_report(p, orb)
            self.assertEqual(orb["signature"], (2, 2, 2, 2), tag)
            self.assertEqual(orb["chi"], 0, tag)
            self.assertEqual(orb["kind"], "parabolic", tag)
            self.assertTrue(lat["lattes_portrait"], tag)
            self.assertEqual(lat["flexible"], "undetermined", tag)
            self.assertIsNone(lat["rigid"], tag)
            self.assertTrue(lat["orbifold_all_two"] and lat["euler_zero"], tag)

    def test_chebyshev_z2_minus_2(self):
        # P = {-2, 2, inf}; the critical point 0 maps to -2, 2 is a fixed
        # critical value reached from both +-2, and inf is a critical fixed
        # point.  Orders (2,2,inf), chi = 0: flat, and not Lattes -- it is
        # the quotient of the cylinder by w -> w^2 ([Mi, Section 6]).
        p = O.portrait_data(3, {0: 1, 1: 1, 2: 2}, {0: 1, 1: 1, 2: 2},
                            {0: [(2, None)],           # 0 -> -2
                             1: [(1, 0), (1, 1)],      # -2, 2 -> 2
                             2: [(2, 2)]},             # inf -> inf
                            check=True)
        orb = O.orbifold(p)
        self.assertEqual(orb["signature"], (2, 2, inf))
        self.assertEqual(orb["chi"], 0)
        self.assertEqual(orb["kind"], "parabolic")
        lat = O.lattes_report(p, orb)
        self.assertFalse(lat["lattes_portrait"])
        self.assertIsNone(lat["rigid"])
        self.assertFalse(lat["flexible"])  # Chebyshev, not four-point Lattes

    def test_the_power_map_z2(self):
        p = O.portrait_data(2, {0: 0, 1: 1}, {0: 2, 1: 2},
                            {0: [(2, 0)], 1: [(2, 1)]}, check=True)
        orb = O.orbifold(p)
        self.assertEqual(orb["signature"], (inf, inf))
        self.assertEqual(orb["chi"], 0)
        self.assertEqual(orb["kind"], "parabolic")
        self.assertFalse(O.lattes_report(p, orb)["lattes_portrait"])

    @staticmethod
    def z2_plus_i():
        # z^2 + i with the repository's convention, P = the forward orbit of
        # the critical *values* (DH: P = union over n > 0 of f^n(Omega)), so
        # P = {i, -1+i, -i, inf} and the critical point 0 is NOT in P.
        # Labels: 0 = i, 1 = -1+i, 2 = -i, 3 = inf.  Then i -> -1+i <-> -i
        # is the 2-cycle, inf is a critical fixed point (local degree 2, as
        # for every polynomial of degree >= 2), and f^{-1}(i) = {0} double
        # with 0 outside P.  Orders (2,2,2,inf), chi = -1/2: a hyperbolic
        # orbifold with a cycle of P carrying no critical point --
        # [PChar]'s hypothesis fails, chi < 0 does not.
        return O.portrait_data(
            4,
            {0: 1, 1: 2, 2: 1, 3: 3},
            {0: 1, 1: 1, 2: 1, 3: 2},
            {0: [(2, None)],
             1: [(1, 0), (1, 2)],
             2: [(1, 1), (1, None)],
             3: [(2, 3)]},
            check=True)

    def test_z2_plus_i(self):
        p = self.z2_plus_i()
        orb = O.orbifold(p)
        self.assertEqual(tuple(orb["orders"][x] for x in range(4)),
                         (2, 2, 2, inf))
        self.assertEqual(orb["signature"], (2, 2, 2, inf))
        self.assertEqual(orb["chi"], Fraction(-1, 2))
        self.assertEqual(orb["kind"], "hyperbolic")
        self.assertFalse(orb["all_cycles_branched"])
        self.assertFalse(O.lattes_report(p, orb)["lattes_portrait"])

    def test_the_shipped_twisted_z2i_has_the_z2_plus_i_portrait(self):
        # The author believes twisted-z2i is a twist of z^2+i and has not
        # checked it.  Twisting preserves the portrait, so the portrait is
        # the checkable half of that belief, and it holds: the shipped data
        # is z^2+i's portrait under {0: i, 1: inf, 2: -1+i, 3: -i}.  The
        # critical fixed point labelled 1 is infinity, which z^2+i has too;
        # the other critical point (0) is outside P and preperiodic, which is
        # why the signature is (2,2,2,inf) and has just one infinite entry.
        p, orb, _ = computed("twisted-z2i")
        z = self.z2_plus_i()
        m = {0: 0, 1: 3, 2: 1, 3: 2}
        self.assertEqual({m[a]: m[b] for a, b in p["dynamics"].items()},
                         z["dynamics"])
        self.assertEqual({m[a]: d for a, d in p["local_degree"].items()},
                         z["local_degree"])

        def normalise(port, mp):
            return {mp[x]: sorted((d, (mp[C] if C is not None else -1))
                                  for d, C in lst)
                    for x, lst in port["preimages"].items()}

        self.assertEqual(normalise(p, m), normalise(z, {k: k for k in range(4)}))
        self.assertEqual(orb["signature"], O.orbifold(z)["signature"])
        self.assertEqual(orb["signature"], (2, 2, 2, inf))

    def test_the_rabbit(self):
        # P = {0, c, c^2+c, inf}: both critical points are postcritical, 0 on
        # the 3-cycle and infinity fixed.  Every order is infinite, chi = -2.
        # labels: 0 = c, 1 = infinity, 2 = c^2+c, 3 = 0.  The critical
        # points are 3 (mapping to c with local degree 2) and 1 (fixed, the
        # critical point at infinity); c receives the double point 0, and 0
        # and c^2+c each receive a simple point of P plus one free preimage.
        p = O.portrait_data(
            4, {0: 2, 1: 1, 2: 3, 3: 0}, {0: 1, 1: 2, 2: 1, 3: 2},
            {0: [(2, 3)],
             1: [(2, 1)],
             2: [(1, 0), (1, None)],
             3: [(1, 2), (1, None)]},
            check=True)
        orb = O.orbifold(p)
        self.assertEqual(orb["signature"], (inf, inf, inf, inf))
        self.assertEqual(orb["chi"], -2)

    def test_a_critical_cycle_forces_infinite_order(self):
        for d in (2, 3, 7):
            p = O.portrait_data(1, {0: 0}, {0: d}, {0: [(d, 0)]})
            orb = O.orbifold(p)
            self.assertEqual(orb["orders"], {0: inf})
            self.assertEqual(orb["chi"], 1)      # 2 - (1 - 1/inf)

    def test_the_fixed_point_is_the_lcm_over_all_levels(self):
        # [DH]'s equivalent form: nu(x) = lcm of deg_y(f^k) over all k >= 1
        # and all y with f^k(y) = x.  Implemented level by level, with no
        # fixed-point iteration: A_k(x) = lcm over all preimage chains of
        # length exactly k ending at x, A_0 = 1, and A_{k+1}(x) = lcm over
        # preimages y of x of deg_y * A_k(y).  This is a second code path for
        # the same recurrence, and it decides the two claims independently:
        # the level lcm settles at nu for finite orders, and it grows without
        # bound at the infinite ones.
        def level_lcms(p, K):
            n = p["n"]
            A, running = {x: 1 for x in range(n)}, {x: 1 for x in range(n)}
            for _ in range(K):
                A = {x: _lcm(*[(dg * A[C]) if C is not None else dg
                               for (dg, C) in p["preimages"][x]])
                     for x in range(n)}
                running = {x: _lcm(running[x], A[x]) for x in range(n)}
            return A, running

        for name in TABLE:
            p, orb, _ = computed(name)
            K = 6 * p["n"] + 12
            A, running = level_lcms(p, K)
            for x, v in orb["orders"].items():
                if v is not inf:
                    self.assertEqual(running[x], v, f"{name}:{x}")
                else:
                    # still growing at depth K: 2^K is reached for a
                    # degree-2 critical cycle, so a weaker bound suffices
                    half, _ = level_lcms(p, K // 2)
                    self.assertGreater(A[x], half[x], f"{name}:{x}")
                    self.assertGreaterEqual(A[x], 2 ** (K // 8), f"{name}:{x}")

        for p in (self.lattes_even(), self.lattes_odd()):
            A, running = level_lcms(p, 8)
            self.assertEqual(running, {x: 2 for x in range(4)})


class TestFlatSignatureEnumeration(unittest.TestCase):
    """chi = 0 forces one of six signatures -- by exhaustive arithmetic.

    chi = 2 - sum(1 - 1/nu) = 0 is the diophantine equation
    sum_{x in P} 1/nu_x = n - 2 with 1/inf = 0, and n <= 4 because every
    term is at most 1/2.  For n = 4 every term is exactly 1/2.  For n = 3
    it is 1/a + 1/b + 1/c = 1 with 2 <= a <= b <= c: a <= 3, a = 3 forces
    b = c = 3, and a = 2 gives (b-2)(c-2) = 4, i.e. (3,6), (4,4) and the
    degenerate b = 2, 1/c = 0.  For n = 2 it is 1/a + 1/b = 0, i.e. both
    infinite; n <= 1 is impossible.  So the list below is complete.
    """

    @staticmethod
    def _solutions(max_order):
        vals = list(range(2, max_order + 1)) + [inf]
        out = set()
        for n in (1, 2, 3, 4):
            def rec(chosen, start):
                if len(chosen) == n:
                    total = sum(Fraction(0) if v is inf else Fraction(1, v)
                                for v in chosen)
                    if total == n - 2:
                        out.add(tuple(chosen))
                    return
                for i in range(start, len(vals)):
                    rec(chosen + [vals[i]], i)
            rec([], 0)
        return out

    def test_the_six_flat_signatures(self):
        self.assertEqual(self._solutions(12),
                         {(2, 2, 2, 2), (2, 4, 4), (3, 3, 3), (2, 3, 6),
                          (2, 2, inf), (inf, inf)})

    def test_the_bound_is_a_proof_not_a_search(self):
        # The three-point cases, solved rather than searched: with
        # a <= b <= c and 1/a + 1/b + 1/c = 1, a <= 3.
        found = set()
        for a in range(2, 4):                      # a = 2 or 3
            rest = 1 - Fraction(1, a)
            for b in range(a, 40):
                if Fraction(1, b) * 2 < rest:
                    break                          # b too large: no c >= b
                c = rest - Fraction(1, b)
                if c == 0:
                    found.add((a, b, inf))
                elif c > 0 and (1 / c).denominator == 1 \
                        and (1 / c) >= b:
                    found.add((a, b, int(1 / c)))
        self.assertEqual(found, {(2, 3, 6), (2, 4, 4), (3, 3, 3), (2, 2, inf)})
        # and no a > 3 can work, since 3/a < 1 for a > 3
        for a in range(4, 50):
            self.assertLess(3 * Fraction(1, a), 1)


class TestLattes(unittest.TestCase):
    """The four-point Lattes-portrait conditions and flexibility boundary."""

    def test_no_shipped_session_is_lattes(self):
        for name in TABLE:
            _, orb, lat = computed(name)
            self.assertFalse(lat["lattes_portrait"], name)
            self.assertFalse(lat["flexible"], name)
            self.assertLess(orb["chi"], 0, name)
            self.assertIsNone(lat["rigid"], name)

    def test_which_condition_each_session_fails(self):
        only_postcritical = {"no_critical_postcritical"}
        expect = {
            # four punctures, every local degree at most 2, as many
            # preimages of P as a Lattes map has -- and a critical point
            # among the postcritical ones, which is what rules them out.
            "rabbit":           only_postcritical,
            "twisted-rabbit":   only_postcritical,
            "twisted-z2i":      only_postcritical,
            "census-3.2":       only_postcritical,
            "julia-4422":       only_postcritical,
            "tetrahedral":      only_postcritical,
            # three punctures: local degrees are simple, but a critical
            # point is postcritical and the preimage count is short.
            "theta":  {"four_postcritical", "no_critical_postcritical",
                       "preimage_decomposition"},
            "barbell": {"four_postcritical", "no_critical_postcritical",
                        "preimage_decomposition"},
            "bouquet": {"four_postcritical", "no_critical_postcritical",
                        "preimage_decomposition"},
            # three punctures with a local degree 3: every condition fails.
            "barycentric":      {"four_postcritical", "all_critical_simple",
                                 "no_critical_postcritical",
                                 "preimage_decomposition"},
            "blown-up-lattes":  {"four_postcritical", "all_critical_simple",
                                 "no_critical_postcritical",
                                 "preimage_decomposition"},
            "census-53":        {"four_postcritical", "all_critical_simple",
                                 "no_critical_postcritical",
                                 "preimage_decomposition"},
            "slit-lattes":      {"four_postcritical", "all_critical_simple",
                                 "no_critical_postcritical",
                                 "preimage_decomposition"},
        }
        for name, fails in expect.items():
            _, _, lat = computed(name)
            got = {k for k, v in lat["checks"].items() if not v}
            self.assertEqual(got, fails, name)

    def test_three_postcritical_points_forces_a_higher_local_degree(self):
        # With n = 3, all critical points simple would give
        # |f^{-1}P| = 3 + (2d-2) > d + 2 = |f^{-1}P|.  So each n = 3 session
        # fails "all simple" or "no critical postcritical" -- never neither.
        for name in TABLE:
            p, _, lat = computed(name)
            if p["n"] == 3:
                self.assertFalse(lat["checks"]["all_critical_simple"]
                                 and lat["checks"]["no_critical_postcritical"],
                                 name)

    def test_the_rigid_333_signature(self):
        # [Mi, Cor. 4.8]: degree 4, three critical points of local degree 3,
        # none postcritical, a 3-cycle on P.
        pre = {0: [(1, 2), (3, None)], 1: [(1, 0), (3, None)],
               2: [(1, 1), (3, None)]}
        p = O.portrait_data(3, {0: 1, 1: 2, 2: 0}, {0: 1, 1: 1, 2: 1}, pre,
                            check=True)
        orb = O.orbifold(p)
        lat = O.lattes_report(p, orb)
        self.assertEqual(orb["signature"], (3, 3, 3))
        self.assertEqual(orb["chi"], 0)
        self.assertTrue(lat["rigid"])
        self.assertFalse(lat["lattes_portrait"])
        self.assertFalse(lat["flexible"])
        self.assertEqual(lat["source_condition"],
                         "every critical point of local degree 3 and none "
                         "postcritical")


class TestGuard(unittest.TestCase):
    """The iteration gate: orbifold permission and separate cycle warnings."""

    def test_every_hyperbolic_orbifold_session_is_a_candidate(self):
        for name in TABLE:
            g = O.orbifold_guard(load(name))
            self.assertTrue(g["allowed"], name)
            self.assertTrue(g["orbifold_ok"], name)
            self.assertEqual(g["status"], "hyperbolic", name)
            self.assertLess(g["chi"], 0, name)
            self.assertIn("hyperbolic", g["reason"], name)
            expect_branched = name not in ("julia-4422", "twisted-z2i")
            self.assertEqual(g["branched_cycles_ok"], expect_branched, name)

    def test_unbranched_cycles_warn_without_excluding(self):
        # Pullback still makes sense for twisted z^2+i.  An unbranched cycle
        # is not evidence of an invalid map or of a Thurston obstruction.
        for name in ("julia-4422", "twisted-z2i"):
            g = O.orbifold_guard(load(name))
            self.assertTrue(g["allowed"], name)
            self.assertTrue(g["orbifold_ok"], name)
            self.assertFalse(g["branched_cycles_ok"], name)
            self.assertEqual(g["status"], "hyperbolic", name)
            self.assertIn("pullback still makes sense", g["reason"], name)
            self.assertIn("[PChar]", g["reason"], name)
            self.assertIn("obstruction", g["reason"], name)

    def test_the_unmarked_sessions_are_undetermined_not_allowed(self):
        for name in UNMARKED:
            g = O.orbifold_guard(load(name))
            self.assertFalse(g["allowed"], name)
            self.assertEqual(g["status"], "undetermined", name)
            self.assertIn("not determined", g["reason"], name)

    def test_a_four_point_lattes_portrait_is_refused(self):
        p = TestHandPortraits.lattes_even()
        orb = O.orbifold(p)
        g = O.guard_verdict(p, orb, O.lattes_report(p, orb))
        self.assertFalse(g["allowed"])
        self.assertFalse(g["orbifold_ok"])
        self.assertEqual(g["status"], "parabolic")
        self.assertTrue(g["lattes_portrait"])
        self.assertEqual(g["flexible"], "undetermined")
        self.assertIn("Lattes", g["reason"])
        self.assertIn("undetermined", g["reason"])

    def test_the_public_gate_keeps_the_candidate_and_explains_it(self):
        from pullback.convergence import convergence_guard
        s = load("julia-4422")
        public = convergence_guard(s)
        self.assertTrue(public["allowed"])
        self.assertEqual(public["status"], "candidate")
        self.assertEqual(public["type"], "cycle without branch point")
        g = O.orbifold_guard(s)
        self.assertTrue(g["allowed"])
        self.assertFalse(g["all_cycles_branched"])
        self.assertTrue(g["orbifold_ok"])
        self.assertIn("[PChar]", g["reason"])
        self.assertIn("obstruction", public["reason"])

    def test_describe_is_readable(self):
        p, orb, lat = computed("julia-4422")
        s = O.describe(orb, lat)
        self.assertIn("signature=(2,2,4,4)", s)
        self.assertIn("chi=-1/2", s)
        self.assertIn("not-Lattes", s)
        self.assertIn("unbranched cycle", s)


class TestNoAnalyticData(unittest.TestCase):
    """Nothing analytic is read: pure ribbon combinatorics."""

    def test_no_analytic_entry_point_is_touched(self):
        from pullback.ve_state import purity_guard
        with purity_guard():
            for name in TABLE:
                computed(name)

    def test_the_module_imports_only_the_portrait(self):
        with open(O.__file__) as fh:
            src = fh.read()
        imports = [line.strip() for line in src.splitlines()
                   if line.strip().startswith(("from .", "from pullback"))]
        self.assertEqual(imports, ["from .portrait import face_map_phi, face_map_pi"])

    def test_the_table_is_reproducible(self):
        def readout(name):
            p, orb, lat = computed(name)
            return (orb["signature"], str(orb["chi"]), orb["kind"],
                    tuple(sorted(orb["infinite"])), lat["lattes_portrait"])
        self.assertEqual({n: readout(n) for n in TABLE},
                         {n: readout(n) for n in TABLE})


def markdown_table():
    rows = ["| session | \\|P\\| | d | orders | signature | chi | kind | Lattes |",
            "|---|---|---|---|---|---|---|---|"]
    for name in list(TABLE) + list(UNMARKED):
        try:
            p, orb, lat = computed(name)
        except O.OrbifoldError as exc:
            rows.append(f"| `{name}` | undetermined: {exc} |")
            continue
        orders = [("inf" if v is inf else v) for v in
                  (orb["orders"][i] for i in range(orb["n"]))]
        lat_txt = ("**LATTES-4PT**" if lat["lattes_portrait"]
                   else "rigid" if lat["rigid"] else "-")
        rows.append(f"| `{name}` | {orb['n']} | {orb['d']} | {orders} | "
                    f"{O._sig(orb['signature'])} | {orb['chi']} | "
                    f"{orb['kind']} | {lat_txt} |")
    return "\n".join(rows)


if __name__ == "__main__":
    if "--table" in sys.argv:
        print(markdown_table())
        sys.exit(0)
    unittest.main(verbosity=2)
