"""Discrete conformal uniformisation on an EDGE-LABELLED multigraph.

Proposed replacement for the incident-vertex-pair representation in
`pullback.rounding`.  The criterion is unchanged -- two metrics on a fixed cell
decomposition are discretely conformally equivalent iff

    log chord_e - log ell_e = (u_i + u_j)/2      for every edge e = (i, j),

for some vertex scaling u (Springborn, arXiv:1707.06848, Def. 10.1) -- but the
edges are identified by LABEL, not by their unordered vertex pair.  That is the
whole difference, and it is what makes a complex with two edges joining the same
pair of vertices representable:

  * the unknowns are edge labels, so two parallel edges give two rows of the
    residual, two rows of the Jacobian and two independent data;
  * the incidence map A has two identical rows for a parallel pair, so its image
    contains only length vectors that agree on the pair.  The projected residual
    therefore contains (a_e1 - a_e2)/sqrt(2) EXACTLY, and a parallel pair whose
    lengths disagree has a positive residual floor no configuration can remove.
    Collapsing the pair onto a vertex-pair key destroys that constraint silently.

Labelling is necessary but NOT sufficient for a canonical round realisation, and
the second half of this module measures why.  Both parallel chords are the same
function of the vertex positions, so (a_e1 - a_e2) is constant in x: that
component is either vacuous (equal lengths) or unsatisfiable.  Every redundant
pair therefore costs one INDEPENDENT equation and the local solution set grows by
one dimension.  `structure` reports the accounting (all three lines measured):

    rank(P J_a) = E - rank(A) - redundant           (generic)
    dim(solution set on the sphere) = 2V - rank(P J_a)
    excess = 2V - 6 - rank(P J_a)                   (= #redundant pairs for a
                                                     triangulated sphere)

`excess` is the number of real moduli of the conformal class that the cell
complex cannot see.  excess == 0 is necessary and sufficient (generically) for
the realisation to be unique up to Mobius; excess > 0 means Gauss-Newton still
converges, but different starts land in genuinely different Mobius classes
(measured on the rabbit lift: 4-point cross-ratio distances of order 1-50).

Nothing here needs faces to be triangles, or simple, or of size >= 3: the
criterion is per edge, so a bigon (two parallel sides of one face) is no
obstacle to the solve.  Faces enter only when reading off the OUTPUT cone
angles, and there they are taken as cyclic sequences of edge labels, so a corner
knows which parallel copy it is measuring.
"""

from __future__ import annotations

import itertools

import numpy as np

__all__ = [
    "edge_order", "edge_incidence", "projector", "chordal_lengths",
    "conformal_residual", "conformal_jacobian", "round_realise",
    "edges_from_faces", "cone_angles", "incidence",
    "redundant_pairs", "sphere_tangent_basis", "structure",
    "mobius_invariants",
]


# --------------------------------------------------------------------------
# edge bookkeeping
# --------------------------------------------------------------------------

def edge_order(edges):
    """Canonical, deterministic order of `{edge_label: (i, j)}`.

    Determinism is by `str` of the label, not by hash: nothing printed may
    depend on set iteration order (the project's A3.10 rule).
    """
    out = []
    for label in sorted(edges, key=str):
        pair = edges[label]
        out.append((label, int(pair[0]), int(pair[1])))
    return tuple(out)


def edge_incidence(edges, n):
    """The edge-vertex incidence map A, one ROW PER EDGE LABEL.

    A[e, i] = A[e, j] = 1/2 for e = (i, j).  Parallel labels give identical
    rows -- that is the point: `image(A)` is what a vertex scaling can produce,
    and it cannot separate two parallel edges.
    """
    order = edge_order(edges)
    A = np.zeros((len(order), int(n)))
    for row, (label, i, j) in enumerate(order):
        if i == j:
            raise ValueError(f"self-loop edge {label!r} at {i}")
        A[row, i] = 0.5
        A[row, j] = 0.5
    return A, order


def incidence(faces, n):
    """Backwards-compatible vertex-pair incidence (as `rounding.incidence`)."""
    edges = edges_from_faces(faces)
    A, order = edge_incidence(edges, n)
    return A, [label for (_l, _i, _j) in order]


def edges_from_faces(faces):
    """`{(i, j): (i, j)}` -- the vertex-pair labelling of a simple complex.

    The degenerate case of the labelled representation: one edge per unordered
    pair, so two faces sharing a pair share the same edge.  Use it only when
    the complex really is simple; on a multigraph it MERGES parallel edges and
    loses constraints (measured: a silently satisfiable residual, below).
    """
    seen = {}
    for f in faces:
        k = len(f)
        for t in range(k):
            i, j = f[t], f[(t + 1) % k]
            key = (i, j) if i < j else (j, i)
            seen[key] = key
    return seen


def redundant_pairs(edges):
    """`E - #distinct vertex pairs` = sum over pairs of (multiplicity - 1).

    Each unit is one dimension of the conformal class the vertex positions
    cannot see; see the module docstring.
    """
    pairs = {}
    for _label, i, j in edge_order(edges):
        key = (i, j) if i < j else (j, i)
        pairs[key] = pairs.get(key, 0) + 1
    return len(edge_order(edges)) - len(pairs)


_PROJECTOR_CACHE = {}


def projector(edges, n):
    """(P, order, incident) with P = I - A A^+ the conformal residual's gauge.

    Cached on the combinatorics: A never changes during a solve, so the pinv is
    paid once.  `incident[k]` lists the edge ROWS incident with vertex k, which
    is what makes the analytic Jacobian O(E^2) rather than O(E*3n).
    """
    key = (edge_order(edges), int(n))
    if key not in _PROJECTOR_CACHE:
        A, order = edge_incidence(edges, n)
        P = np.eye(len(order)) - A @ np.linalg.pinv(A)
        rows = [[] for _ in range(int(n))]
        for row, (_label, i, j) in enumerate(order):
            rows[i].append(row)
            rows[j].append(row)
        _PROJECTOR_CACHE[key] = (P, order, rows)
    return _PROJECTOR_CACHE[key]


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def chordal_lengths(x):
    """Matrix of distances |x_i - x_j| for points on the sphere."""
    x = np.asarray(x, dtype=float)
    sq = np.sum(x * x, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * x @ x.T
    return np.sqrt(np.clip(d2, 0.0, None))


def _length_vector(order, lengths, chord):
    a = np.empty(len(order))
    for row, (label, i, j) in enumerate(order):
        ell = lengths[label]
        if ell <= 0.0:
            raise ValueError(f"edge {label!r} has non-positive length {ell}")
        d = chord[i, j]
        if d <= 0.0:
            # A degenerate configuration is not a metric at all; signal it so a
            # line search rejects the step instead of raising inside the solver.
            return None
        a[row] = np.log(d) - np.log(ell)
    return a


def conformal_residual(edges, lengths, x):
    """r = (I - A A^+) (log chord - log ell); returns (r, order, |r|).

    |r| = 0 exactly when some vertex scaling u satisfies the criterion.  The
    return is `inf` when two vertices of an edge coincide, so a step which
    collapses an edge can never be accepted.
    """
    x = np.asarray(x, dtype=float)
    P, order, _rows = projector(edges, len(x))
    a = _length_vector(order, lengths, chordal_lengths(x))
    if a is None:
        return np.full(len(order), np.inf), order, float("inf")
    r = P @ a
    return r, order, float(np.linalg.norm(r))


def conformal_jacobian(edges, x):
    """d r / d x = (I - A A^+) J_a, with J_a sparse in the edge endpoints.

    a_e = log|x_i - x_j| - log ell_e, so d a_e / d x_k = (x_k - x_other)/|x_k -
    x_other|^2 with the SAME sign at either endpoint.  Built vertex by vertex --
    column block k comes from the edges incident with k only.  A redundant pair
    contributes two rows with identical entries (same endpoints, same formula):
    that is exactly why one equation goes missing, since P then cannot resolve
    the difference.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    P, order, rows = projector(edges, n)
    J = np.zeros((len(order), 3 * n))
    for k in range(n):
        inc = rows[k]
        if not inc:
            continue
        B = np.zeros((len(inc), 3))
        for slot, row in enumerate(inc):
            _label, i, j = order[row]
            other = j if i == k else i
            d = x[k] - x[other]
            d2 = float(d @ d)
            if d2 <= 0.0:
                continue
            B[slot] = d / d2
        J[:, 3 * k:3 * k + 3] = P[:, inc] @ B
    return J


def round_realise(edges, lengths, x0=None, maxit=120, tol=1e-13, step=1.0,
                  seed=0):
    """Positions on S^2 whose chordal metric is conformally equivalent to ell.

    Gauss-Newton on `conformal_residual` with the analytic `conformal_jacobian`,
    a backtracking line search and renormalisation onto the sphere.  Returns
    (x, residual_norm, iterations).  Local method: the caller's initial guess
    fixes the basin, exactly as one pullback step at a time uses it.

    The Mobius gauge and the residual's invariance under vertex scaling are left
    to the minimum-norm step.  `structure` says whether the solution found is
    unique up to Mobius; a residual of 0e0 is not by itself that statement.
    """
    order = edge_order(edges)
    if not order:
        raise ValueError("no edges")
    n = 1 + max(max(i, j) for (_l, i, j) in order)
    if x0 is None:
        rng = np.random.RandomState(seed)
        x = rng.normal(size=(n, 3))
        x /= np.linalg.norm(x, axis=1)[:, None]
    else:
        x = np.array(x0, dtype=float)
        if x.shape != (n, 3):
            raise ValueError(f"x0 has shape {x.shape}, expected {(n, 3)}")
        x /= np.linalg.norm(x, axis=1)[:, None]

    def residual(xx):
        return conformal_residual(edges, lengths, xx)[0]

    best = float(np.linalg.norm(residual(x)))
    stalls = 0
    it = 0
    for it in range(maxit):
        r0 = residual(x)
        norm0 = float(np.linalg.norm(r0))
        if not np.isfinite(norm0):
            break
        if norm0 < tol:
            return x, norm0, it
        J = conformal_jacobian(edges, x)
        delta, *_ = np.linalg.lstsq(J, -r0, rcond=None)
        accepted = False
        for _ in range(40):
            xt = (x.reshape(-1) + step * delta).reshape(x.shape)
            norms = np.linalg.norm(xt, axis=1)
            if np.any(norms == 0.0):
                step *= 0.5
                continue
            xt = xt / norms[:, None]
            normt = float(np.linalg.norm(residual(xt)))
            if normt < norm0:
                progress = (norm0 - normt) / max(norm0, 1e-300)
                x, best, accepted = xt, normt, True
                break
            step *= 0.5
        if not accepted:
            break
        stalls = stalls + 1 if progress < 1e-10 else 0
        if stalls >= 3:
            break
        step = min(1.0, step * 1.5)
    return x, best, it


# --------------------------------------------------------------------------
# the structural certificate
# --------------------------------------------------------------------------

def sphere_tangent_basis(x):
    """An orthonormal basis of the tangent space of (S^2)^n at `x`.

    Returned as a 3n x 2n matrix, so `J @ T` is the derivative restricted to
    configurations that stay on the sphere.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    T = np.zeros((3 * n, 2 * n))
    for k in range(n):
        p = x[k]
        e1 = np.cross(p, [0.0, 0.0, 1.0])
        if np.linalg.norm(e1) < 1e-8:
            e1 = np.cross(p, [1.0, 0.0, 0.0])
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(p, e1)
        T[3 * k:3 * k + 3, 2 * k] = e1
        T[3 * k:3 * k + 3, 2 * k + 1] = e2
    return T


def structure(edges, x=None, rtol=1e-9):
    """The rank accounting that decides whether a round realisation is canonical.

    V, E, rank(A) (= V for a connected non-bipartite graph, V-1 if bipartite),
    `redundant` = E - #distinct vertex pairs, and -- given a configuration --
    rank(P J_a) restricted to the product of spheres, plus

        excess = 2V - 6 - rank(PJ),

    the number of real dimensions of solution set beyond the 6-dim Mobius orbit.
    For a triangulated sphere, excess == redundant.

    `canonical` is the honest reading: True means the round realisation is
    unique up to Mobius (generically), so the returned configuration is a
    well-defined point of Teich(S^2, V).  False means it is one member of a
    family and the cell complex cannot choose.
    """
    order = edge_order(edges)
    if not order:
        raise ValueError("no edges")
    n = 1 + max(max(i, j) for (_l, i, j) in order)
    A, _order = edge_incidence(edges, n)
    rank_a = int(np.linalg.matrix_rank(A))
    out = {
        "V": n,
        "E": len(order),
        "redundant": redundant_pairs(edges),
        "rank_A": rank_a,
        "rank_P": len(order) - rank_a,
        "mobius": 6,
        "canonical": None,
        "excess": None,
        "rank_PJ": None,
        "singular_values": None,
    }
    if x is not None:
        J = conformal_jacobian(edges, np.asarray(x, dtype=float))
        T = sphere_tangent_basis(x)
        sv = np.linalg.svd(J @ T, compute_uv=False)
        ref = float(sv[0]) if len(sv) and sv[0] > 0.0 else 1.0
        rank_pj = int((sv > rtol * ref).sum())
        out["rank_PJ"] = rank_pj
        out["singular_values"] = sv
        out["excess"] = 2 * n - 6 - rank_pj
        out["canonical"] = out["excess"] == 0
    return out


def mobius_invariants(x, pole=(0.3, -0.5, 0.81)):
    """The 4-point cross ratios of a configuration: Mobius-invariant labels.

    Comparing two configurations by position is meaningless on the sphere --
    they are the same point of Teich exactly when all their 4-point cross
    ratios agree.  Returns the C(V,4) values in `itertools.combinations` order.
    """
    x = np.asarray(x, dtype=float)
    pole = np.asarray(pole, dtype=float)
    pole = pole / np.linalg.norm(pole)
    b = np.array([0.0, 0.0, 1.0])
    v = np.cross(pole, b)
    c = float(pole @ b)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    R = np.eye(3) + vx + vx @ vx / (1.0 + c)
    y = x @ R.T
    z = (y[:, 0] + 1j * y[:, 1]) / (1.0 - y[:, 2])
    out = []
    for a, b_, c_, d in itertools.combinations(range(len(x)), 4):
        out.append(((z[a] - z[c_]) * (z[b_] - z[d]))
                   / ((z[a] - z[d]) * (z[b_] - z[c_])))
    return np.array(out)


# --------------------------------------------------------------------------
# output cone angles (needs the labelled faces, not just the edges)
# --------------------------------------------------------------------------

def _angle_at(x, v, p, q):
    a = x[p] - x[v]
    b = x[q] - x[v]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        raise ValueError(f"coincident corners at vertex {v}")
    return float(np.arccos(np.clip(float(a @ b) / denom, -1.0, 1.0)))


def cone_angles(face_edges, edges, x):
    """Sum of corner angles at each vertex of a labelled polygonal complex.

    `face_edges[k]` is the cyclic sequence of edge LABELS around face k, in the
    same vertex order: the corner at index t is between face_edges[k][t-1] and
    face_edges[k][t] at the vertex they share.  Because the labels are explicit,
    a bigon's two sides and a doubled edge's two copies are measured
    separately.  A face with two corners is allowed -- its two sides share both
    endpoints, so both its angles vanish, which is a fact about bigons rather
    than a limitation here.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    total = np.zeros(n)
    for k, cyc in enumerate(face_edges):
        m = len(cyc)
        if m < 2:
            raise ValueError(f"face {k} has {m} edges; need at least 2")
        for t in range(m):
            e_prev = edges[cyc[t - 1]]
            e_next = edges[cyc[t]]
            shared = set(e_prev) & set(e_next)
            if len(shared) != 1:
                raise ValueError(
                    f"face {k}: consecutive edges {cyc[t - 1]!r} and {cyc[t]!r} "
                    f"do not meet in exactly one vertex")
            v = shared.pop()
            p = e_prev[1] if e_prev[0] == v else e_prev[0]
            q = e_next[1] if e_next[0] == v else e_next[0]
            if p == q:
                raise ValueError(f"face {k} pinches at vertex {v}")
            total[v] += _angle_at(x, v, p, q)
    return total
