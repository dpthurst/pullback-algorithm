"""Rational maps as coefficients (the front end of an SC importer).

Given f = N/D by coefficient lists, compute: critical points, the
postcritical set, the PCF check, preimage fibers with multiplicities,
and the *portrait degree multiset* — the multiset of local degrees over
the points of P, which must agree with the pi-portrait of any session
claimed to represent f.

Numerics: mpmath at high precision for root-finding; tolerances are
explicit and reported, per the project's rule that every tolerance
carries its measurement.  Iteration depths here are small (PCF sets of
the example library), well inside the reliability range; this is a
watertight check, not an iteration-count experiment.
"""

from __future__ import annotations

import mpmath as mp

mp.mp.dps = 30

# Clustering tolerance for root deduplication.  MEASURED: mpmath's
# polyroots at dps 30 returns the k roots of an exact k-fold root of the
# slit-lattes denominator spread by about 1e-10; merging below 1e-8 is
# safe for the library's maps (all critical points are separated by
# O(1) at this precision).  Reported here so no tolerance is implicit.
CLUSTER_TOL = mp.mpf("1e-8")


def _polyval(coeffs, z):
    """coeffs highest power first."""
    out = mp.mpc(0)
    for c in coeffs:
        out = out * z + mp.mpc(c)
    return out


def _polyroots(coeffs):
    """Roots of a polynomial (highest first) at current precision,
    converged by mpmath's polyroots."""
    # strip leading zeros
    c = list(coeffs)
    while c and abs(mp.mpc(c[0])) < mp.mpf(10) ** (-mp.mp.dps + 5):
        c.pop(0)
    if len(c) <= 1:
        return []
    return mp.polyroots(c, maxsteps=200, extraprec=200)


class RationalMap:
    def __init__(self, num, den):
        self.num = [mp.mpc(c) for c in num]
        self.den = [mp.mpc(c) for c in den]
        while self.num and self.num[0] == 0:
            self.num.pop(0)
        while self.den and self.den[0] == 0:
            self.den.pop(0)
        self.deg = max(len(self.num), len(self.den)) - 1
        if self.deg < 1:
            raise ValueError("degree must be >= 1")

    def __call__(self, z):
        z = mp.mpc(z)
        n = _polyval(self.num, z)
        d = _polyval(self.den, z)
        if abs(d) < mp.mpf(10) ** (-mp.mp.dps + 8):
            return mp.inf
        return n / d

    def derivative(self, z):
        """f'(z) = (N'D - N D')/D^2, for finite z away from the poles."""
        def deriv(c):
            n = len(c) - 1
            return [c[i] * (n - i) for i in range(n)]
        z = mp.mpc(z)
        N, D = self.num, self.den
        num = _polyval(_polysub(_polymul(deriv(N), D), _polymul(N, deriv(D))), z)
        den = _polyval(D, z)
        if abs(den) < mp.mpf(10) ** (-mp.mp.dps + 8):
            return mp.inf
        return num / (den * den)

    def critical_points(self):
        """Finite critical points: roots of N'D - N D'."""
        def deriv(c):
            n = len(c) - 1
            return [c[i] * (n - i) for i in range(n)]
        N, D = self.num, self.den
        Np, Dp = deriv(N), deriv(D)
        # N' D - N D'
        prod1 = _polymul(Np, D)
        prod2 = _polymul(N, Dp)
        diff = _polysub(prod1, prod2)
        roots = _polyroots(diff)
        out = []
        for z in roots:
            if not any(abs(z - w) < CLUSTER_TOL for w in out):
                out.append(z)
        return out

    # -- infinity bookkeeping (degree-based, exact) -------------------
    @property
    def degN(self):
        return len(self.num) - 1

    @property
    def degD(self):
        return len(self.den) - 1

    def local_degree_infinity(self):
        """Local degree at infinity: |degN - degD| when they differ,
        else the order of vanishing of f(1/w) - f(infinity) at w = 0,
        computed as the number of trailing zeros of
        Delta = Ntilde*Dtilde(0) - Ntilde(0)*Dtilde (highest first)."""
        k = self.degD - self.degN
        if k != 0:
            return abs(k)
        Nd = list(reversed(self.num))
        Dd = list(reversed(self.den))
        n = max(len(Nd), len(Dd))
        Nd = [mp.mpc(0)] * (n - len(Nd)) + Nd
        Dd = [mp.mpc(0)] * (n - len(Dd)) + Dd
        f0 = self.num[0] / self.den[0]
        delta = [Nd[i] * Dd[-1] - Nd[-1] * Dd[i] for i in range(n)]
        scale = max(abs(c) for c in delta) if delta else 1
        order = 0
        for c in reversed(delta):
            if abs(c) > scale * mp.mpf(10) ** (-mp.mp.dps + 8):
                break
            order += 1
        return order if order >= 1 else 1

    def f_infinity(self):
        if self.degN > self.degD:
            return mp.inf
        if self.degN < self.degD:
            return mp.mpc(0)
        return self.num[0] / self.den[0]

    def pole_degree(self, z):
        """Local degree at a pole: the multiplicity of the zero of D at
        z (the order of the pole), computed exactly from derivatives."""
        return _root_multiplicity(self.den, z)

    def local_degree(self, z, tol=mp.mpf(10) ** (-12)):
        """Local degree at a FINITE point: 2 when f' vanishes there (all
        our branch points are simple), 1 otherwise; poles via order."""
        z = mp.mpc(z)
        fz = self(z)
        if not mp.isfinite(fz):
            return self.pole_degree(z)
        eps = mp.mpf("1e-15")
        z1 = z + eps
        z2 = z + 2 * eps * mp.mpc("0.7", "0.7")
        d1 = (mp.mpc(self(z1)) - fz) / (z1 - z)
        d2 = (mp.mpc(self(z2)) - fz) / (z2 - z)
        if abs(d1) < tol and abs(d2) < tol:
            return 2
        return 1

    def preimage_with_degrees(self, x):
        """[(y, local degree)] over every y in f^{-1}(x), x finite or
        mp.inf, including y = infinity when applicable."""
        out = []
        if x == mp.inf:
            # poles: DISTINCT roots of D with pole orders; plus infinity
            roots = []
            for z in _polyroots(self.den):
                if not any(abs(z - w) < CLUSTER_TOL
                           for w in roots):
                    roots.append(z)
            for z in roots:
                out.append((z, self.pole_degree(z)))
            if self.degN > self.degD:
                out.append((mp.inf, self.degN - self.degD))
            return out
        x = mp.mpc(x)
        coeff = _polysub(self.num, [x * c for c in self.den])
        roots = []
        for z in _polyroots(coeff):
            if not any(abs(z - w) < CLUSTER_TOL
                       for w in roots):
                roots.append(z)
        for z in roots:
            out.append((z, self.local_degree(z)))
        if self.degN == self.degD:
            finf = self.num[0] / self.den[0]
            if abs(finf - x) < mp.mpf(10) ** (-20):
                out.append((mp.inf, self.local_degree_infinity()))
        elif self.degN < self.degD:
            if abs(x) < mp.mpf(10) ** (-20):
                out.append((mp.inf, self.local_degree_infinity()))
        return out

    def iterate(self, z, n):
        for _ in range(n):
            z = self(z)
        return z

    def postcritical_set(self, depth=12, tol=mp.mpf("1e-12")):
        """Closure of the critical orbits, including infinity; returns
        (finite points, has_infinity, tolerance used)."""
        pts = []
        has_inf = False
        starts = self.critical_points()
        orbits = []
        for z0 in starts:
            # P is the FORWARD orbit: f^n(z0), n >= 1 (a critical point
            # is in P only if it recurs, e.g. superattracting cycles)
            orbit = []
            z = self(mp.mpc(z0))
            for _ in range(depth):
                if not mp.isfinite(z):
                    orbit.append(mp.inf)
                    break
                orbit.append(mp.mpc(z))
                z = self(z)
            orbits.append(orbit)
        if self.local_degree_infinity() > 1:
            orbit = []
            z = self.f_infinity()
            for _ in range(depth):
                if z == mp.inf:
                    orbit.append(mp.inf)
                    break
                orbit.append(mp.mpc(z))
                z = self(z)
            orbits.append(orbit)
        for orb in orbits:
            for z in orb:
                if z == mp.inf:
                    has_inf = True
                    continue
                if not any(abs(z - y) < tol for y in pts):
                    pts.append(z)
        return pts, has_inf, tol

    def portrait_degrees(self):
        """Multiset of local degrees over the points of P: the degree
        multiset a session's pi-portrait must reproduce."""
        P, has_inf, _ = self.postcritical_set()
        degs = []
        targets = list(P) + ([mp.inf] if has_inf else [])
        for x in targets:
            degs.extend(d for (_, d) in self.preimage_with_degrees(x))
        return sorted(degs)


def _deriv(c):
    n = len(c) - 1
    return [c[i] * (n - i) for i in range(n)]


def _root_multiplicity(coeffs, z):
    """Multiplicity of z as a root of the polynomial (highest first)."""
    c = list(coeffs)
    m = 0
    scale = max(abs(cc) for cc in c) if c else 1
    while len(c) > 1:
        v = _polyval(c, z)
        if abs(v) > scale * mp.mpf(10) ** (-10):
            return m if m > 0 else 1
        m += 1
        c = _deriv(c)
        if not c:
            break
        scale = max(scale, max(abs(cc) for cc in c))
    return m if m > 0 else 1


def _polymul(a, b):
    out = [mp.mpc(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def _polysub(a, b):
    n = max(len(a), len(b))
    a = [0] * (n - len(a)) + list(a)
    b = [0] * (n - len(b)) + list(b)
    return [x - y for x, y in zip(a, b)]


# --------------------------------------------------- Moebius conjugation

def _padd(a, b):
    """Sum of two coefficient lists, highest power first."""
    n = max(len(a), len(b))
    a = [mp.mpc(0)] * (n - len(a)) + [mp.mpc(c) for c in a]
    b = [mp.mpc(0)] * (n - len(b)) + [mp.mpc(c) for c in b]
    return [x + y for x, y in zip(a, b)]


def _pscale(s, a):
    return [mp.mpc(s) * mp.mpc(c) for c in a]


def _pmul(a, b):
    out = [mp.mpc(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += mp.mpc(x) * mp.mpc(y)
    return out


def mobius_conjugate(f, a, b, c, d):
    """M o f o M^-1 for M(z) = (a z + b)/(c z + d), as a RationalMap.

    Conjugation by a Möbius map preserves the critical portrait, the orbifold
    and the type; it moves the geometry.  Applied to a polynomial with M not
    affine it produces a map that is not a polynomial, whose postcritical set
    need not contain infinity -- the strictly rational test cases.
    """
    if a * d - b * c == 0:
        raise ValueError("M is degenerate")
    N, D = [mp.mpc(x) for x in f.num], [mp.mpc(x) for x in f.den]
    # M^-1(w) = (d w - b)/(a - c w)
    N1, D1 = [d, -b], [-c, a]
    # X = f(M^-1(w)) = (N(N1/D1) * D1^deg ... ) -- do it by degree bookkeeping:
    # with p = deg f, f(x) = (sum N_k x^(p-k))/(sum D_k x^(q-k)), so
    # f(N1/D1) = sum_k N_k N1^(p-k) D1^k  /  sum_k D_k N1^(q-k) D1^k
    def compose(poly, n1, d1):
        p = len(poly) - 1
        acc = [mp.mpc(0)]
        for k, ck in enumerate(poly):
            term = _pscale(ck, _pmul(_ppow(n1, p - k), _ppow(d1, k)))
            acc = _padd(acc, term)
        return acc
    A = compose(N, N1, D1)          # N(X) = A / D1^p
    B = compose(D, N1, D1)          # D(X) = B / D1^q
    # f(X) = N(X)/D(X) = D1^(q-p) A/B, so multiply through by the power of D1
    # that clears it.  Multiplying both by a *common* power instead leaves an
    # unfactored common factor, and then N(w) - x D(w) has a root independent of
    # x: every preimage list gets a spurious extra point (measured on the slit
    # map, whose critical values were read as three preimages each).
    p, q = len(N) - 1, len(D) - 1
    if p >= q:
        B = _pmul(B, _ppow(D1, p - q))
    else:
        A = _pmul(A, _ppow(D1, q - p))
    num = _padd(_pscale(a, A), _pscale(b, B))
    den = _padd(_pscale(c, A), _pscale(d, B))
    return RationalMap(num, den)


def _ppow(p, k):
    out = [mp.mpc(1)]
    for _ in range(k):
        out = _pmul(out, p)
    return out
