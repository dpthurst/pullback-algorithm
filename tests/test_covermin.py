"""The convex-cover minimiser (plan item 170): gadgets, cross-checks against
the exact QP, the published values, the balance certificate, determinism.

`pullback.covermin` minimises Dir over the homotopy class by coordinate
descent in the universal cover of the target, with an exact rational block
solver (`minimise_one`).  Lifting the energy to the cover makes it convex,
so the combinatorial type search of `pullback.fastmin` -- and its bouquet
failure -- is gone.  The engine is numerical, though: the descent converges
geometrically and the returned offsets are snapped to small-denominator
rationals, so its outputs are bracketed rather than exact; the exact QP
(`pullback.minimise`) is the certifier.  Every tolerance below is measured,
not guessed (2026-10-26, the six cases of TestHarmonicStepAgainstExactQP
and the runs of TestPublishedValues):

  harmonic step at unit ell, covermin vs the exact QP (worst of 6 cases):
      0 <= Dir - Dir_QP <= 3.5e-9      (one-sided: Dir_QP is the exact min)
      |v - v_QP| <= 1.9e-5
      Dir_id equal exactly
  fixed-point lambda (= Emb[phi^n]) vs the published values and vs the
  QP's certified LB: |err| <= 5.1e-17
  mutation probe (move any vertex 1/10 along any lifted direction):
      every Dir increment >= 1.5e-2 > 0
  balance certificate: every tension gap <= 0, exactly
"""

import os
import subprocess
import sys
import unittest
from fractions import Fraction as Q
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pullback import covermin as cm
from pullback.iterate_graphs import session_iterate
from pullback.minimise import Minimiser, minimise

UNIT_THETA = {"p": 1, "q": 1, "r": 1}

# the six cross-check cases: (session, iterate n, elastics on the base)
CROSSCHECK_CASES = (("theta", 1, UNIT_THETA), ("theta", 2, UNIT_THETA),
                    ("theta", 3, UNIT_THETA), ("bouquet", 1, None),
                    ("bouquet", 2, None), ("bouquet", 3, None))


def edge_letter(e):
    """The base edge letter of a G_n edge label (a tuple of cover darts or
    vertex 1-tuples; the LAST coordinate is always a cover dart, whose label
    is (base label, sheet))."""
    lab = e[-1] if isinstance(e, tuple) else e
    while isinstance(lab, tuple) and not isinstance(lab[0], str):
        lab = lab[0]
    return lab[0] if isinstance(lab, tuple) else lab


_CACHE = {}


def case(name, n, elastics=None):
    """The n-th iterate class of a session, run through alg:iter once.

    Returns a dict with the Cls, the refined graphs (for the exact-QP
    cross-checks), the lambda sequence, the fixed-point lambda, the final
    balance gaps and the harmonic representative X.  Cached: each run costs
    seconds and several tests share it.
    """
    key = (name, n, tuple(sorted((elastics or {}).items())))
    if key not in _CACHE:
        Gn, phin, G2 = session_iterate(ROOT / "sessions" / f"{name}.json", n)
        if elastics is not None:
            for e in Gn.edges:
                Gn.alpha[e] = Q(elastics[edge_letter(e)])
            for e in G2.edges:
                G2.alpha[e] = Q(elastics[e])
        cls = cm.Cls(cm.G.from_refined(Gn),
                     {e: tuple(phin.chains.get(e, ())) for e in Gn.edges},
                     cm.G.from_refined(G2), marked=set(Gn.marked))
        lams, ell, _cls, X, gaps = cm.alg_iter(cls)
        _CACHE[key] = dict(cls=cls, Gn=Gn, G2=G2, phin=phin, lams=lams,
                           lam=lams[-1], gaps=gaps, X=X, ell=ell)
    return _CACHE[key]


def unit_ell(G2):
    return {f: Q(1) for f in sorted(G2.edges, key=str)}


class TestLiftedPathConventions(unittest.TestCase):
    """Regression gadgets for the lp_path orientation bug (fixed 2026-10-26).

    When both lifted points sit in the same target edge and p is further
    along it than r, lp_path returned the traversal FORWARD along the edge
    -- the direction from r to p -- so lp_direction, the _eps_star tension
    gaps and _atomic_segments were all wrong in that case.  Measured then:
    positive "balance" gaps at the exact harmonic map, and an order-
    dependent minimise_one (the same terms in the other order gave the
    right answer).  The gadgets below pin the corrected conventions; they
    are exact and instant.
    """

    def setUp(self):
        self.g = cm.G({"d": ("A", "B")}, {"d": Q(1)})
        self.pB = cm.lp_vertex((("d", cm.F),))       # lifted vertex at B
        self.q = cm.lp_inside((), ("d", cm.F), Q(1, 2), self.g)

    def test_first_motion_backward_along_shared_edge(self):
        dist, up, down = cm.lp_path(self.pB, self.q, self.g)
        self.assertEqual(dist, Q(1, 2))
        # from B the way to q walks d BACKWARD
        self.assertEqual(cm.lp_direction(self.pB, self.q, self.g),
                         ("d", cm.B))
        self.assertEqual(cm.lp_direction(self.q, self.pB, self.g),
                         ("d", cm.F))
        self.assertEqual(up, [(cm.rev(("d", cm.F)), Q(0), Q(1, 2))])
        self.assertEqual(down, [])

    def test_minimise_one_is_order_independent(self):
        # minimiser of d(x, pB)^2 + d(x, q)^2 is x at 3/4, value 1/8
        for terms in ([(self.pB, Q(1)), (self.q, Q(1))],
                      [(self.q, Q(1)), (self.pB, Q(1))]):
            x, gaps = cm.minimise_one(terms, self.g)
            self.assertEqual(x, cm.lp_inside((), ("d", cm.F), Q(3, 4), self.g))
            self.assertEqual(cm._fval(terms, x, self.g), Q(1, 8))
            self.assertEqual([g for _d, g in gaps], [Q(0), Q(0)])

    def test_atomic_segments_cross_vertices(self):
        g = cm.G({"d": ("A", "B"), "e": ("B", "C")}, {"d": Q(1), "e": Q(1)})
        pC = cm.lp_vertex((("d", cm.F), ("e", cm.F)))
        vB = cm.lp_vertex((("d", cm.F),))
        q = cm.lp_inside((), ("d", cm.F), Q(1, 2), g)
        segs = cm._atomic_segments(pC, q, g)
        self.assertEqual(segs, [(pC, vB), (vB, q)])
        dist, up, down = cm.lp_path(pC, q, g)
        self.assertEqual(dist, Q(3, 2))
        self.assertEqual(up, [(cm.rev(("e", cm.F)), Q(0), Q(1))])
        # from the end of d the way down to q walks d BACKWARD
        self.assertEqual(down, [(cm.rev(("d", cm.F)), Q(0), Q(1, 2))])

    def test_eps_star_matches_exact_slope(self):
        # f(x) = sum w_i d(x, y_i)^2; moving x by eps in direction d changes
        # f by 2*eps*(behind - ahead) + eps^2*tot exactly (while no landing
        # point is crossed).  _eps_star returns gap = ahead - behind, so the
        # measured slope must equal -2*gap + eps*tot.
        g = cm.G({"d": ("A", "B"), "e": ("B", "C")}, {"d": Q(1), "e": Q(1)})
        y1 = cm.lp_inside((), ("d", cm.F), Q(1, 4), g)
        y2 = cm.lp_vertex((("d", cm.F), ("e", cm.F)))
        x0 = cm.lp_vertex((("d", cm.F),))
        terms = [(y1, Q(1)), (y2, Q(2))]
        tot = sum(w for _y, w in terms)
        f0 = cm._fval(terms, x0, g)
        for d in cm.lp_directions(x0, g):
            _eps, gap = cm._eps_star(terms, x0, d, g)
            for eps in (Q(1, 1000), Q(1, 100)):
                x1 = cm.lp_step(x0, d, eps, g)
                slope = (cm._fval(terms, x1, g) - f0) / eps
                self.assertEqual(slope, -2 * gap + eps * tot,
                                 f"direction {d}, eps {eps}")


class TestSessionClass(unittest.TestCase):
    """`session_class` loads the session's own elastic structure: the theta
    session's [4,1,4] is (p,q,r) over the base's sorted edges, the paper's
    assignment, and reproduces its (4,1,4) value."""

    def test_theta_matches_paper_and_iterate_path(self):
        cls = cm.session_class(ROOT / "sessions" / "theta.json")
        self.assertEqual(cls.alpha2,
                         {"p": Q(4), "q": Q(1), "r": Q(4)})
        lams, _ell, _c, _X, gaps = cm.alg_iter(cls)
        # the paper's SF[phi^1] for (p,q,r) = (4,1,4), printed to 6 decimals
        self.assertLessEqual(abs(lams[-1] - Q(864357, 10**6)), Q(5, 10**7))
        # the same class built through the iterate path: identical
        r = case("theta", 1, {"p": 4, "q": 1, "r": 4})
        self.assertEqual(lams[-1], r["lam"])
        self.assertEqual([g for g in gaps if g[2] > 0], [])

    def test_bouquet_unit_elastics(self):
        # the bouquet session has no "elastics": the class is unit
        cls = cm.session_class(ROOT / "sessions" / "bouquet.json")
        self.assertEqual(set(cls.alpha2.values()), {Q(1)})
        lams, _ell, _c, _X, _gaps = cm.alg_iter(cls)
        self.assertEqual(lams[-1], Q(1))


class TestHarmonicStepAgainstExactQP(unittest.TestCase):
    """One harmonic step at unit ell vs the exact QP, six cases.

    Measured (2026-10-26, default tol=1e-9 descent): Dir - Dir_QP lies in
    [0, 3.5e-9] -- one-sided, since Dir_QP is the exact minimum and Dir is
    the energy of an actual representative -- |v - v_QP| <= 1.9e-5, and
    Dir_id agrees exactly.  The tolerances carry >= 5x margin over the
    measured worst case.
    """

    def test_dir_within_measured_tolerance(self):
        for name, n, elast in CROSSCHECK_CASES:
            with self.subTest(name=name, n=n):
                r = case(name, n, elast)
                ell = unit_ell(r["G2"])
                Dir, Dir_id, _DR, _v, _X = r["cls"].dr(ell)
                h = Minimiser(r["Gn"], r["G2"], r["phin"].chains).harmonic(ell)
                self.assertEqual(Dir_id, h["Dir_id"])
                self.assertGreaterEqual(Dir, h["Dir"])
                self.assertLessEqual(Dir - h["Dir"], Q(1, 10**7))

    def test_fill_within_measured_tolerance(self):
        for name, n, elast in CROSSCHECK_CASES:
            with self.subTest(name=name, n=n):
                r = case(name, n, elast)
                ell = unit_ell(r["G2"])
                _Dir, _Dir_id, _DR, v, _X = r["cls"].dr(ell)
                h = Minimiser(r["Gn"], r["G2"], r["phin"].chains).harmonic(ell)
                for f in v:
                    self.assertLessEqual(abs(v[f] - h["v"][f]), Q(1, 10**4),
                                         f"target edge {f}")


class TestPublishedValues(unittest.TestCase):
    """The fixed-point lambda is Emb[phi^n].  Published: the bouquet values
    1, 3/4, 5/8, 23/48 at n = 1..4 (the Harmonic paper's Emb[f^n], also the
    exact engine's certified LBs at n = 1..3); theta (1,1,1) 3/2 and 6/7
    (the exact engine's interior fixed points, [Rubber] Figure 2).  Measured
    |lam - published| <= 5.1e-17 (bouquet n=4); the bracket is 1e-15.
    (bouquet n=1,3 and theta n=1 come out exactly equal today.)"""

    def test_bouquet(self):
        for n, pub in ((1, Q(1)), (2, Q(3, 4)), (3, Q(5, 8)), (4, Q(23, 48))):
            with self.subTest(n=n):
                lam = case("bouquet", n)["lam"]
                self.assertLessEqual(abs(lam - pub), Q(1, 10**15))

    def test_theta_unit_elastics(self):
        for n, pub in ((1, Q(3, 2)), (3, Q(6, 7))):
            with self.subTest(n=n):
                lam = case("theta", n, UNIT_THETA)["lam"]
                self.assertLessEqual(abs(lam - pub), Q(1, 10**15))


class TestFixedPointAgainstExactQP(unittest.TestCase):
    """lambda vs the exact engine's certified LB (max DR over the verified
    fixed points; equal to Emb at an interior fixed point).  theta n=2 is
    excluded -- the QP verifies no fixed point there (LB is None); the
    harmonic-step cross-check covers it.  Measured |lam - LB| <= 8.7e-29."""

    def test_lambda_matches_certified_lower_bound(self):
        for name, n, elast in (("theta", 1, UNIT_THETA),
                               ("theta", 3, UNIT_THETA),
                               ("bouquet", 1, None),
                               ("bouquet", 2, None),
                               ("bouquet", 3, None)):
            with self.subTest(name=name, n=n):
                r = case(name, n, elast)
                lb = minimise(Minimiser(r["Gn"], r["G2"],
                                        r["phin"].chains))["LB"]
                self.assertIsNotNone(lb)
                self.assertLessEqual(abs(r["lam"] - lb), Q(1, 10**15))


class TestBalanceCertificate(unittest.TestCase):
    """The balance certificate: at the harmonic map every directional
    tension gap is <= 0 -- the balance condition of harmonic.tex
    cor:harmonic-triangle, which characterises the harmonic (Dir-minimal)
    map.  The gaps are exact rationals computed on the descent's own output
    before snapping."""

    def test_gaps_nonpositive(self):
        for name, n, elast in CROSSCHECK_CASES:
            with self.subTest(name=name, n=n):
                gaps = case(name, n, elast)["gaps"]
                self.assertTrue(gaps)          # the certificate has teeth
                self.assertEqual([g for g in gaps if g[2] > 0], [])


class TestMutationIncreasesDir(unittest.TestCase):
    """Convention-free optimality probe: moving any vertex by 1/10 along
    any lifted direction must increase Dir -- the harmonic map is the
    minimiser.  Measured worst increment over all (vertex, direction):
    +3.0e-2 (theta n=2), +1.6e-2 (bouquet n=2)."""

    def test_probe(self):
        for name, n, elast in (("theta", 2, UNIT_THETA), ("bouquet", 2, None)):
            with self.subTest(name=name, n=n):
                r = case(name, n, elast)
                ell = unit_ell(r["G2"])
                Dir, _Dir_id, _DR, _v, X = r["cls"].dr(ell)
                g = r["cls"].tgt.with_lengths(ell)
                eps = Q(1, 10)
                worst = None
                for vtx in r["cls"].src.vertices():
                    if vtx in r["cls"].marked:
                        continue
                    for d in cm.lp_directions(X[vtx], g):
                        Xp = dict(X)
                        Xp[vtx] = cm.lp_step(X[vtx], d, eps, g)
                        inc = r["cls"].energy(Xp, ell) - Dir
                        if worst is None or inc < worst:
                            worst = inc
                self.assertGreater(worst, 0)


class TestEmbIsTheEssentialSupremum(unittest.TestCase):
    """`emb` returns the scalar Emb (intro.tex eq:emb): the essential
    supremum of the pointwise multiplicity, i.e. the max of the per-edge
    multiplicities (target vertices have measure zero).  Regression: it
    returned the multiplicity *dict* (the swarm port's latent bug, caught by
    the database-wide iteration check on 2026-10-26)."""

    def test_scalar_and_max_of_fill_at_elastic_metric(self):
        r = case("theta", 1, {"p": 4, "q": 1, "r": 4})
        cls = r["cls"]
        ell = dict(cls.alpha2)          # the elastic metric
        e = cls.emb(cls.X0, ell)
        self.assertIsInstance(e, Q)
        self.assertEqual(e, max(cls.fill(cls.X0, ell).values()))

    def test_barbell_explicit_bound(self):
        # the exact engine's bracket for the barbell at n = 1 is [5, 6]:
        # the input representative's Emb is 6 (covermin's alg:iter converges
        # to the non-maximal fixed point DR = 3 -- the paper's caveat, the
        # escape procedure of Algorithm 6.49 is not implemented)
        r = case("barbell", 1)
        cls = r["cls"]
        ell = {f: Q(1) for f in cls.tgt.labels()}
        self.assertEqual(cls.emb_of_input(ell), Q(6))
        self.assertLessEqual(r["lam"], Q(6))


class TestToleranceStop(unittest.TestCase):
    """alg:iter's tolerance stop (used by the figure/tower drivers) leaves
    the fixed-point value bracketed: the lambda agrees with the exact-
    repeat run to the tolerance and the run takes fewer steps."""

    def test_tol_stop_matches_exact_repeat(self):
        for n in (1, 2):
            r = case("theta", n, UNIT_THETA)
            cls = r["cls"]
            lams_exact, _e, _c, _X, _g = cm.alg_iter(cls)
            lams_tol, _e2, _c2, _X2, _g2 = cm.alg_iter(cls, tol=Q(1, 10**12))
            with self.subTest(n=n):
                self.assertLessEqual(abs(lams_tol[-1] - lams_exact[-1]),
                                     Q(1, 10**11))
                self.assertLessEqual(len(lams_tol), len(lams_exact))


class TestLooseDescentSchedule(unittest.TestCase):
    """`htol_loose` gives the early harmonic descents a loose tolerance
    decaying to the default; the reported factor must not change beyond
    the descent tolerance (it is off by default)."""

    def test_loose_schedule_same_factor(self):
        r = case("theta", 1, UNIT_THETA)
        lams_exact, *_ = cm.alg_iter(r["cls"])
        lams_loose, *_ = cm.alg_iter(r["cls"], htol_loose=Q(1, 10 ** 3))
        self.assertLessEqual(abs(lams_loose[-1] - lams_exact[-1]), Q(1, 10 ** 8))


class TestDeterminism(unittest.TestCase):
    """Nothing may depend on set iteration order (AGENTS.md A3.10): the
    harmonic output must be identical across PYTHONHASHSEED values."""

    CODE = (
        "import sys; sys.path.insert(0, 'src')\n"
        "from fractions import Fraction as Q\n"
        "from pullback import covermin as cm\n"
        "cls = cm.session_class('sessions/theta.json')\n"
        "ell = {f: Q(1) for f in cls.tgt.labels()}\n"
        "X, gaps = cls.harmonic(ell)\n"
        "print(sorted((str(v), str(x)) for v, x in X.items()))\n"
        "print(sorted((str(v), str(d), str(g)) for v, d, g in gaps))\n"
        "print(sorted((str(e), str(ps))\n"
        "             for e, ps in cls.pieces(X, ell).items()))\n"
    )

    def test_same_across_hash_seeds(self):
        outs = []
        for seed in ("0", "1"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            proc = subprocess.run([sys.executable, "-c", self.CODE],
                                  cwd=ROOT, env=env,
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            outs.append(proc.stdout)
        self.assertEqual(outs[0], outs[1])


if __name__ == "__main__":
    unittest.main()
