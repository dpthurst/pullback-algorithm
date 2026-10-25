"""The energies E^p_q of graph maps, for every 1 <= p <= q <= infinity.

Sources: [ConfDim] (arXiv:2112.09041) §2, which reviews [Elast]
(arXiv:1607.00340) Appendix A; the two agree and both are implemented
here.  Plan items 160-210 (M3).  For p = 2 this is the elastic energy:
E^2_2 = sqrt(Emb).  Everything else is what p != 2 needs.

A q-conformal structure on a graph, q > 1, is a positive *q-length*
alpha(e) on each edge; for q = 1 it is a positive *weight* w(e).  For a
PL graph map phi: G -> H, [Elast, (Fillp),(Epq)] sets

    Fill^p(phi)(y) := sum_{x in phi^-1(y)} |phi'(x)|^(p-1)      (p > 1)
                    := sum_{x in phi^-1(y)} w(x)                (p = 1)
    E^p_q(phi)     := (||Fill^p(phi)||_{q/(q-p), H})^(1/p)      (1 < q)

with the cases
  * p = q:        E^p_p(phi) = max_y Fill^p(phi)(y)^(1/p)      ([ConfDim] E^q_q)
  * p = 1, q > 1: E^1_q(phi) = (sum_d alpha(d) n(d)^(q/(q-1)))^((q-1)/q)
  * q = infinity: E^p_inf(phi) = (sum_e alpha(e)^(1-p) l(phi(e))^p)^(1/p)
  * p = q = 1:    E^1_1(phi) = ess sup_y n(y)/w(y)             ([ConfDim] E^1_1)
  * p = q = inf:  E^inf_inf(phi) = Lip(phi)
so that for 1 < p < q < infinity

    E^p_q(phi) = (sum_d alpha(d) Fill^p(phi)(d)^(q/(q-p)))^((q-p)/(p q)).

Conventions of this implementation, matching the session data model:

* A graph map is given by its chains, {source edge: tuple of target
  darts}.  Vertices go to vertices, so every chain is a whole sequence
  of darts and |phi'| is constant on each source edge, equal to
  (total length of the image chain) / (length of the source edge); a
  constant edge has |phi'| = 0.  Free reduction of the chains never
  changes |phi'|, so a reduced representative is the right input; a
  chain that traverses one target edge twice contributes twice to Fill,
  at two distinct parameter values.
* Fill^p is therefore constant on each *open* target edge, and the
  essential supremum is a max over edges.  Images of source vertices are
  finitely many points and carry no measure, as [ConfDim] notes; a
  constant source edge contributes nothing to Fill on an open target
  edge.
* Every value is an `Energy(base, exponent)`, meaning base^exponent with
  `base` an exact `Fraction` whenever the data are rational and the
  exponent p integral (plan D5: exact certificates, numerical brackets).
  In particular E^p_p(phi) < 1 is decided exactly for integral p, since
  base < 1 decides it.

These are the energies of an *explicit representative*: nothing here
minimises over the homotopy class.  That is what gives rigorous upper
bounds on the asymptotic energy ([PChar, Prop. 5.6]: ^E E^p_p =
inf_n E^p_p[phi^n]^(1/n) <= E^p_p(psi)^(1/n) for any explicit psi in
[phi^n]); the infimum itself and the asymptotic energy are
`iterates.py` and the obstruction machinery's business.
"""

from __future__ import annotations

from fractions import Fraction

import mpmath as mp

DPS = 30  # mpmath working precision for irrational values

INF = "inf"  # the exponent q = infinity


class ExponentError(ValueError):
    pass


# ---------------------------------------------------------------- numbers
def to_fraction(x):
    """Exact rational or None (ints, Fractions and dyadic floats are
    exact; other floats are refused rather than silently rounded)."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, float) and x == x and abs(x) != float("inf"):
        f = Fraction(x)
        if abs(float(f) - x) == 0:
            return f
    return None


def is_exact(x) -> bool:
    """Exact: a Fraction or an int (both are used as lengths)."""
    return isinstance(x, (Fraction, int))


def _exactize(x):
    return Fraction(x) if isinstance(x, int) else x


def mpf(x):
    """mpf of a Fraction/int/mpf, rounded from the Fraction correctly."""
    if isinstance(x, Fraction):
        return mp.mpf(x.numerator) / mp.mpf(x.denominator)
    return mp.mpf(x)


def exponent(x):
    """Normalise an exponent: Fraction, or INF for infinity."""
    if x is None:
        return INF
    if isinstance(x, str) and x.lower() in ("inf", "infinity"):
        return INF
    if isinstance(x, float) and x == float("inf"):
        return INF
    f = to_fraction(x)
    if f is None:
        raise ExponentError(f"not a rational exponent: {x!r}")
    return f


def _pow(x, e):
    """x^e, exact when x is exact and e an integer, else mpf."""
    x = _exactize(x)
    if isinstance(e, Fraction) and e.denominator == 1:
        e = int(e)
    if isinstance(x, Fraction) and isinstance(e, int):
        return x ** e
    return mp.power(mpf(x), mpf(e) if isinstance(e, Fraction) else mp.mpf(e))


def _div(a, b):
    a, b = _exactize(a), _exactize(b)
    if isinstance(a, Fraction) and isinstance(b, Fraction):
        return a / b
    return mpf(a) / mpf(b)


def _add(a, b):
    a, b = _exactize(a), _exactize(b)
    if isinstance(a, Fraction) and isinstance(b, Fraction):
        return a + b
    return mpf(a) + mpf(b)


def exact_root(x, k):
    """The exact k-th root of a non-negative Fraction, or None.

    Reporting only: sqrt(9/4) = 3/2, sqrt(3/5) = None."""
    x = to_fraction(x)
    if x is None or x < 0:
        return None
    k = int(k)
    rn, rd = _iroot(x.numerator, k), _iroot(x.denominator, k)
    if rn is None or rd is None:
        return None
    return Fraction(rn, rd)


def _iroot(n, k):
    if n < 0:
        return None
    if n == 0:
        return 0
    r = int(round(n ** (1.0 / k)))
    for cand in (r - 1, r, r + 1):
        if cand >= 0 and cand ** k == n:
            return cand
    return None


class Energy:
    """base^exponent, with `base` exact when the data are rational.

    Every energy of the family has this shape, so exactness is carried by
    the pair and no rounding happens until `.value` is asked for."""

    __slots__ = ("base", "exponent")

    def __init__(self, base, exponent=Fraction(1)):
        e = to_fraction(exponent)
        if e is None:
            raise ExponentError(f"exponent must be rational, got {exponent!r}")
        self.base = Fraction(base) if isinstance(base, int) else base
        self.exponent = e

    @property
    def exact(self) -> bool:
        return isinstance(self.base, Fraction)

    def pow(self, k) -> "Energy":
        """(base^e)^k = base^(e k)."""
        f = to_fraction(k)
        if f is None:
            raise ExponentError(f"exponent must be rational, got {k!r}")
        return Energy(self.base, self.exponent * f)

    def exact_value(self):
        """The exact value when it is rational, else None."""
        if not self.exact:
            return None
        e = self.exponent
        if e.denominator == 1:
            return self.base ** int(e)
        if e.numerator == 1 and e >= 0:
            return exact_root(self.base, e.denominator)
        return None

    @property
    def value(self):
        mp.mp.dps = DPS
        return _pow(self.base, self.exponent)

    def __float__(self):
        return float(self.value)

    def _other(self, other):
        return other.value if isinstance(other, Energy) else mpf(other)

    def __lt__(self, other):
        return self.value < self._other(other)

    def __le__(self, other):
        return self.value <= self._other(other)

    def __gt__(self, other):
        return self.value > self._other(other)

    def __ge__(self, other):
        return self.value >= self._other(other)

    def __eq__(self, other):
        if isinstance(other, Energy):
            ex, ey = self.exact_value(), other.exact_value()
            if ex is not None and ey is not None:
                return ex == ey
            return self.value == other.value
        if isinstance(other, (int, Fraction)):
            exact = self.exact_value()
            return exact is not None and exact == other
        return NotImplemented

    def __repr__(self):
        e = self.exponent
        if e == 1:
            return f"Energy({self.base})"
        return f"Energy(({self.base})^({e}))"


# ---------------------------------------------------------------- energies
def _as_dict(alpha, edges, what):
    if isinstance(alpha, dict):
        missing = [e for e in edges if e not in alpha]
        if missing:
            raise ValueError(f"no {what} for edges {missing}")
        return alpha
    if to_fraction(alpha) is None:
        raise TypeError(f"{what} must be a number or a dict, got {alpha!r}")
    return {e: alpha for e in edges}


def chain_length(chain, alpha):
    """The length of a chain of target darts."""
    tot = 0
    for d in chain:
        tot = _add(tot, alpha[d[0]])
    return tot


def dilations(chains, alpha_src, alpha_tgt):
    """|phi'| on each source edge (0 on a constant edge)."""
    alpha_src = _as_dict(alpha_src, list(chains), "source conformal value")
    out = {}
    for e, chain in chains.items():
        if not chain:
            out[e] = Fraction(0) if is_exact(alpha_src[e]) else mpf(0)
        else:
            out[e] = _div(chain_length(chain, alpha_tgt), alpha_src[e])
    return out


def fills(chains, alpha_src, alpha_tgt, p):
    """Fill^p(phi) as a dict on the target edges, exact when possible.

    p = 1 uses the source weights alpha_src; p > 1 uses the source
    q-lengths alpha_src.  A target edge met by no chain is absent
    (Fill = 0 there)."""
    p = exponent(p)
    if p == INF:
        raise ExponentError("Fill is defined for finite p")
    alpha_src = _as_dict(alpha_src, list(chains), "source conformal value")
    alpha_tgt = _as_dict(alpha_tgt, _target_edges(chains), "target conformal value")
    d_ = dilations(chains, alpha_src, alpha_tgt)
    out = {}
    for e, chain in chains.items():
        if not chain:
            continue
        if p == 1:
            w = alpha_src[e]
        else:
            if d_[e] == 0:
                continue
            w = _pow(d_[e], p - 1)
        for d in chain:
            out[d[0]] = _add(out.get(d[0], 0), w)
    return out


def _target_edges(chains):
    seen = []
    for chain in chains.values():
        for d in chain:
            if d[0] not in seen:
                seen.append(d[0])
    return seen


def energy(chains, alpha_tgt, p, q=None, alpha_src=None):
    """E^p_q of the explicit graph map given by `chains`.

    chains:    {source edge: tuple of target darts}
    alpha_tgt: {target edge: q-length}, or a scalar for every target
               edge met (for q = 1: {target edge: weight})
    alpha_src: {source edge: q-length}, or weights when p = 1
               (default 1 on every edge, i.e. the unweighted graph)
    q:         defaults to p (the energy the app needs, E^p_p)
    """
    p = exponent(p)
    q = p if q is None else exponent(q)
    if p == INF and q != INF:
        raise ExponentError("E^inf_q is not defined for q < infinity")
    if p != INF and q != INF and not (p <= q):
        raise ValueError(f"need p <= q, got p={p}, q={q}")
    if p != INF and p < 1:
        raise ValueError(f"need p >= 1, got p={p}")
    if alpha_src is None:
        alpha_src = Fraction(1)
    src_edges = list(chains)

    if q == 1:
        if p != 1:
            raise ExponentError("only E^1_1 is defined for q = 1")
        ws = _as_dict(alpha_src, src_edges, "source weight")
        tg = _as_dict(alpha_tgt, _target_edges(chains), "target weight")
        f = fills(chains, ws, tg, 1)
        base = 0
        for d, val in f.items():
            r = _div(val, tg[d])
            base = r if r > base else base
        return Energy(base, Fraction(1))

    if q == INF:
        al = _as_dict(alpha_src, src_edges, "source q-length")
        tg = _as_dict(alpha_tgt, _target_edges(chains), "target length")
        if p == INF:
            d_ = dilations(chains, al, tg)
            return Energy(max(d_.values()) if d_ else 0, Fraction(1))
        if p == 1:
            tot = 0
            for e, chain in chains.items():
                tot = _add(tot, al[e] * chain_length(chain, tg))
            return Energy(tot, Fraction(1))
        base = 0
        for e, chain in chains.items():
            ell = chain_length(chain, tg)
            if ell == 0:
                continue
            base = _add(base, _pow(al[e], 1 - p) * _pow(ell, p))
        return Energy(base, Fraction(1) / p)

    # 1 <= p <= q < infinity
    tg = _as_dict(alpha_tgt, _target_edges(chains), "target q-length")
    if p == 1:
        ws = _as_dict(alpha_src, src_edges, "source weight")
        n = fills(chains, ws, tg, 1)
        qt = q / (q - 1)
        base = 0
        for d, val in n.items():
            base = _add(base, tg[d] * _pow(val, qt))
        return Energy(base, Fraction(1) / qt)
    al = _as_dict(alpha_src, src_edges, "source q-length")
    f = fills(chains, al, tg, p)
    if p == q:
        base = 0
        for val in f.values():
            base = val if val > base else base
        return Energy(base, Fraction(1) / p)
    e = q / (q - p)
    base = 0
    for d, val in f.items():
        base = _add(base, tg[d] * _pow(val, e))
    return Energy(base, Fraction(1) / p - Fraction(1) / q)


# ------------------------------------------------- short-hand of the cases
def E_p_p(chains, alpha_src, alpha_tgt, p):
    """E^p_p: the energy the app reports (E^2_2 = sqrt(Emb))."""
    return energy(chains, alpha_tgt, p, p, alpha_src)


def E_1_q(chains, weights_src, alpha_tgt, q):
    return energy(chains, alpha_tgt, Fraction(1), q, weights_src)


def E_p_inf(chains, alpha_src, lengths_tgt, p):
    return energy(chains, lengths_tgt, p, INF, alpha_src)


def E_1_1(chains, weights_src, weights_tgt):
    return energy(chains, weights_tgt, Fraction(1), Fraction(1), weights_src)


def embedding_energy(chains, alpha_src, alpha_tgt):
    """Emb(phi) = max_y sum_{phi(x)=y} |phi'(x)| = E^2_2(phi)^2.

    [Elast]'s embedding energy (plan item 160): the energy whose
    minimiser over the homotopy class is Emb[phi]."""
    return E_p_p(chains, alpha_src, alpha_tgt, Fraction(2)).pow(2)


# ------------------------------------------------------------- composition
def compose_chains(inner, outer):
    """The chains of a composite map.

    inner: {edge of G_0: chain of darts in G_1}
    outer: {edge of G_1: chain of darts in G_2}
    returns {edge of G_0: chain of darts in G_2}, the chain of a dart
    being the outer chain, or its reversal for a backward dart.

    Free reduction is left to the caller (or to make_graph_map); it never
    changes a length, so it never changes an energy."""
    from .ribbons import rev

    out = {}
    for e, chain in inner.items():
        acc = []
        for d in chain:
            ch = outer[d[0]]
            acc.extend(ch if d[1] > 0 else [rev(x) for x in reversed(ch)])
        out[e] = tuple(acc)
    return out
