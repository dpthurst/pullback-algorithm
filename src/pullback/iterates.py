"""Iterates phi^n_0: G_n -> G_0 of a graph virtual endomorphism, and the
upper bounds on the asymptotic energies they give.

The space, and the covering the author asked about.  G_n is the orbit
space of [ConfDim] §"Multi-valued dynamical systems" (= [PChar] §5),

    G_n = { (x_1, ..., x_n) in G_1^n : pi(x_i) = phi(x_{i+1}) ,  i < n },

with phi^n_0(x_1..x_n) = phi(x_1) and pi^n_0(x_1..x_n) = pi(x_n), so that
pi^n_0 = pi o pi^{n-1}_{n-2} o ... and the one-step maps drop the first
(pi) or the last (phi) coordinate.  **pi^{n+1}_n is a covering of degree
deg(pi), hence pi^n_0 is a covering of degree deg(pi)^n**: over a point y
of G_n its fibre is pi^{-1}(phi(first coordinate of y)), which is
deg(pi) points because pi is a covering of G_0 at *every* point of G_0,
vertex or not.  So G_n is a deg(pi)^n-fold cover of G_0, with
|V(G_n)| = deg(pi)^n |V(G_0)| and |E(G_n)| = deg(pi)^n |E(G_0)|, and the
q-conformal structure on G_n is the pullback of the one on G_0 along that
covering.  (An earlier draft of this module enumerated G_n's edges as
tuples of G_1-edges with the "phi(e_{i+1}) is a single dart over e_i"
condition and concluded pi^n_0 was not a covering.  That was an error of
that enumeration, not of the definitions: such tuples are only the edges
on which *every* coordinate stays in one edge.  When phi(e_{i+1}) is a
longer chain, x_i still traverses the lift of that chain, and G_n has an
edge for it.  Those edges are exactly the ones with a factor other than
delta_i = 1 and they change the energies, so they cannot be dropped.)

Why the graph is not built.  The filling at a generic point does not need
it.  Let delta(f) = l_0(phi(f)) / alpha_1(f) be the dilation of phi on
the edge f of G_1 (alpha_1 = the lift of alpha_0, so alpha_1(f) =
alpha_0(pi(f))), and let P(l) be the multiset of edges f of G_1 whose
chain phi(f) traverses base edge l (one entry per traversal).  For a
generic point y of G_0 the preimages under phi^n_0 are the chains
x_1 in phi^{-1}(y), x_2 in phi^{-1}(pi(x_1)), ..., and at each x_i the
dilation of phi^n_0 is the *product* of the dilations of phi at
x_1,...,x_n (telescoping: pi is an isometry on edges, and
l_1(x_i) = l_0(phi(x_{i+1})) pointwise).  Since everything is constant on
edges, summing over the tree of preimages is a matrix power:

    M[l][l'] := sum over f in P(l) with pi(f) = l' of delta(f)^(p-1)
    Fill^p(phi^n_0)(y) = (M^n 1)_l    for y generic in base edge l,

so E^p_p(phi^n_0) = (max_l (M^n 1)_l)^(1/p) and, for p < q < infinity,
E^p_q(phi^n_0) = (sum_l alpha_0(l) (M^n 1)_l^(q/(q-p)))^((q-p)/(p q)).
For p = 1 the same matrix with delta^0 = 1 counts preimages (use the
source weights if they are not all 1).  All exact: Fractions for
rational alpha and integral p.

Sanity: for n = 1, (M 1)_l is sum over f traversing l of delta(f)^(p-1),
which is `energies.fills` — the matrix is a rewriting of the single-step
filling, and the tests check the two agree.

Why the bounds are valid (plan item 180).  [PChar, Prop. 5.6]: the
asymptotic energy ^E E^p_p is the limit, and also the infimum, of
E^p_p[phi^n]^(1/n) over n, the infimum running over homotopy classes.
For any explicit representative psi of [phi^n], E^p_p(psi) >=
E^p_p[phi^n], so ^E E^p_p <= E^p_p(phi^n_0)^(1/n) for every n, with
phi^n_0 the representative built by this module.  No minimisation is
involved: the numbers are upper bounds, exact for integral p.  How sharp
they are depends on the representative; [Elast] §6 minimises within the
class, which is a separate layer.

The q = infinity energy of an iterate is not implemented: it sums over
the edges of G_n with their lifted lengths, which needs the edge
decomposition of the cover, not just the filling.  energies.E_p_inf is
for an explicit map on an explicit graph.
"""

from __future__ import annotations

from fractions import Fraction

from .energies import Energy, ExponentError, exponent, mpf, _add, _pow
from .energies import chain_length, is_exact


def cover_lifts(cover):
    """{base edge: [cover edge, ...]} — the lifts of each base edge."""
    out = {}
    for e in cover.edges:
        out.setdefault(e[0], []).append(e)
    return out


def degree(cover):
    """The covering degree, checked constant over edges."""
    counts = {len(v) for v in cover_lifts(cover).values()}
    if len(counts) != 1:
        raise ValueError(f"not a covering: lift counts {sorted(counts)}")
    return counts.pop()


def base_edges(cover):
    return list(cover_lifts(cover))


def lifted_alpha(cover, base_lengths):
    """alpha_1 on G_1: the pullback of alpha_0 along pi."""
    if isinstance(base_lengths, dict):
        return {f: base_lengths[f[0]] for f in cover.edges}
    return {f: base_lengths for f in cover.edges}


def dilations(cover, phi, base_lengths=None):
    """delta(f) = l_0(phi(f)) / alpha_1(f) on each edge of G_1."""
    if base_lengths is None:
        base_lengths = Fraction(1)
    a1 = lifted_alpha(cover, base_lengths)
    out = {}
    for f in cover.edges:
        ch = phi.chains[f]
        if not ch:
            out[f] = Fraction(0) if is_exact(a1[f]) else mpf(0)
        else:
            ell = chain_length(ch, base_lengths if isinstance(base_lengths, dict)
                               else {d[0]: base_lengths for d in ch})
            out[f] = ell / a1[f] if (is_exact(ell) and is_exact(a1[f])) else mpf(ell) / mpf(a1[f])
    return out


def fill_matrix(cover, phi, p, base_lengths=None, weights=None):
    """M[l][l'] = sum over traversals of base edge l by phi(f), pi(f) = l',
    of delta(f)^(p-1) (of w(f) for p = 1).  Exact when possible."""
    p = exponent(p)
    if p == "inf":
        raise ExponentError("the filling matrix is for finite p")
    delta = dilations(cover, phi, base_lengths)
    M = {}
    for l in base_edges(cover):
        M[l] = {}
    for f in cover.edges:
        l2 = f[0]
        if p == 1:
            w = Fraction(1) if weights is None else weights[f]
        else:
            if delta[f] == 0:
                continue
            w = _pow(delta[f], p - 1)
        for d in phi.chains[f]:
            M[d[0]][l2] = _add(M[d[0]].get(l2, 0), w)
    return M


def fill_vector(cover, phi, n, p, base_lengths=None, weights=None):
    """Fill^p(phi^n_0) on each base edge, as M^n applied to 1."""
    if n < 1:
        raise ValueError("n must be >= 1")
    M = fill_matrix(cover, phi, p, base_lengths, weights)
    v = {l: (Fraction(1) if all(is_exact(x) for x in row.values()) else mpf(1))
         for l, row in M.items()}
    for _ in range(n):
        v = {l: _dot(row, v) for l, row in M.items()}
    return v


def _dot(row, v):
    tot = 0
    for l2, a in row.items():
        tot = _add(tot, a * v[l2] if is_exact(a) and is_exact(v[l2]) else mpf(a) * mpf(v[l2]))
    return tot


def iterate_energy(cover, phi, n, p, q=None, base_lengths=None, weights=None):
    """E^p_q(phi^n_0) for 1 <= p <= q < infinity."""
    p = exponent(p)
    q = p if q is None else exponent(q)
    if q == "inf" or p == "inf":
        raise ExponentError("the iterate energy is implemented for q < infinity")
    if not (1 <= p <= q):
        raise ValueError(f"need 1 <= p <= q, got p={p}, q={q}")
    fill = fill_vector(cover, phi, n, p, base_lengths, weights)
    if q == 1:
        if p != 1:
            raise ExponentError("only E^1_1 is defined for q = 1")
        tg = _weights(base_lengths, fill)
        base = 0
        for l, val in fill.items():
            base = max(base, val / tg[l])
        return Energy(base, Fraction(1))
    tg = _lengths(base_lengths, fill)
    if p == 1:
        qt = q / (q - 1)
        base = 0
        for l, val in fill.items():
            base = _add(base, tg[l] * _pow(val, qt))
        return Energy(base, 1 / qt)
    if p == q:
        base = 0
        for val in fill.values():
            base = val if val > base else base
        return Energy(base, 1 / p)
    e = q / (q - p)
    base = 0
    for l, val in fill.items():
        base = _add(base, tg[l] * _pow(val, e))
    return Energy(base, Fraction(1) / p - Fraction(1) / q)


def _lengths(base_lengths, fill):
    if isinstance(base_lengths, dict):
        return base_lengths
    return {l: base_lengths for l in fill}


def _weights(base_lengths, fill):
    return _lengths(base_lengths if base_lengths is not None else Fraction(1), fill)


def asymptotic_profile(cover, phi, p, n_max, base_lengths=None):
    """[(n, E^p_p(phi^n_0), E^p_p(phi^n_0)^(1/n))] for n = 1..n_max."""
    out = []
    for n in range(1, n_max + 1):
        e = iterate_energy(cover, phi, n, p, base_lengths=base_lengths)
        out.append((n, e, e.pow(Fraction(1, n))))
    return out


def asymptotic_upper_bound(cover, phi, p, n_max, base_lengths=None):
    """(n, E, E^(1/n)) minimising E^(1/n): an upper bound on ^E E^p_p."""
    prof = asymptotic_profile(cover, phi, p, n_max, base_lengths)
    return min(prof, key=lambda row: row[2].value)


def perron_root(cover, phi, p, base_lengths=None, iters=200):
    """The growth rate rho(M)^(1/p), the limit of the bounds as n -> inf.

    M is nonnegative, so the limit of (max_l (M^n 1)_l)^(1/(p n)) is
    rho(M)^(1/p) (Perron-Frobenius); rho is computed by the power
    method on floats.  This is *not* the rigorous bound -- the infimum
    over n is (`asymptotic_upper_bound`); it says where the sequence is
    heading."""
    M = fill_matrix(cover, phi, p, base_lengths)
    v = {l: 1.0 for l in M}
    lam = 0.0
    for _ in range(iters):
        w = {l: sum(float(a) * v[l2] for l2, a in row.items()) for l, row in M.items()}
        s = max(w.values()) if w else 0.0
        if s == 0:
            return 0.0
        v = {l: x / s for l, x in w.items()}
        if abs(s - lam) < 1e-15 * max(1.0, s):
            lam = s
            break
        lam = s
    return lam ** (1.0 / float(p))
