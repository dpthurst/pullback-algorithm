"""Möbius gauge, registration and gluing for spherical realisations.

Springborn (2019) Theorem 7.18 / 10.5 return the spherical CEPS answer as a
configuration of points on ``S^2`` that is unique only **up to a Möbius
transformation** of the sphere.  Three consequences are load-bearing for the
one-step lift:

* a realisation cannot be compared to an oracle by coordinates; it has to be
  registered against it by the unique Möbius map matching labelled
  correspondences, and the *residual of that registration* is the error that
  may be reported;
* a representative must be pinned by an explicit, reproducible gauge;
* when a realisation is assembled from pieces (per patch, per sheet, or a
  hierarchy of refined solves) each piece carries its own unknown Möbius
  freedom, and the pieces have to be glued by solving for those freedoms on
  the overlaps.  A failed glue is a genuine inconsistency, not noise.

Representation
--------------
A Möbius transformation of ``S^2`` is the projective action of
``O(3,1)`` on the null vectors ``x^ = (p, 1)`` with metric
``eta = diag(1, 1, 1, -1)``:  every point of the sphere is a null direction,
and every conformal (orientation preserving or reversing) automorphism of the
sphere is realised by exactly one projective Lorentz transformation.  A map is
therefore stored as a single real 4x4 matrix ``M`` with ``M^T eta M = eta``;
``det M = +1`` means conformal-orientation preserving, ``det M = -1`` means
reversing (a mirror/anti-Möbius map).  Composition is matrix product, inverse
is ``eta M^T eta``, and neither ever divides by a chart coordinate: points at
either pole are handled exactly like any other point.

Everything is checked in ``tests/test_mobius_gauge.py``.
"""

from __future__ import annotations

import itertools

import numpy as np

ETA = np.diag([1.0, 1.0, 1.0, -1.0])

# reflection in the y axis: p -> (x, -y, z).  Anti-Möbius, and in O(3,1).
MIRROR = np.diag([1.0, -1.0, 1.0, 1.0])

# the basis of Hermitian 2x2 matrices in the coordinate order used everywhere
# here, (x, y, z, x0):  X(p) = x sigma_1 + y sigma_2 + z sigma_3 + x0 sigma_0
_SIGMA = [
    np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
]


class GaugeError(ValueError):
    """A gauge/registration problem that cannot be solved as posed."""


# --------------------------------------------------------------------- points
def as_points(P, tol=1e-9):
    """Validate an ``(n, 3)`` array of unit vectors and return it as float."""
    P = np.array(P, dtype=float, copy=True)
    if P.ndim != 2 or P.shape[1] != 3:
        raise GaugeError(f"expected an (n, 3) array of sphere points, got {P.shape}")
    norms = np.linalg.norm(P, axis=1)
    if np.any(np.abs(norms - 1.0) > tol):
        raise GaugeError("points are not on the unit sphere: max |p| error "
                         f"{np.max(np.abs(norms - 1.0)):.3e}")
    return P / norms[:, None]


def _homog(P):
    return np.concatenate([P, np.ones((len(P), 1))], axis=1)


def _dehomog(X):
    x0 = X[:, 3]
    if np.any(np.abs(x0) < 1e-14):
        raise GaugeError("a point was mapped to infinity; the map is not in O(3,1)")
    return X[:, :3] / x0[:, None]


def _herm(P):
    """Hermitian 2x2 matrices X(p) = [[1+z, x-iy], [x+iy, 1-z]] (rank one)."""
    n = len(P)
    X = np.empty((n, 2, 2), dtype=complex)
    X[:, 0, 0] = 1.0 + P[:, 2]
    X[:, 0, 1] = P[:, 0] - 1j * P[:, 1]
    X[:, 1, 0] = P[:, 0] + 1j * P[:, 1]
    X[:, 1, 1] = 1.0 - P[:, 2]
    return X


def _spin(P):
    """The Veronese spinor z(p) = (1+z, x+iy); zero only at the south pole."""
    return np.stack([1.0 + P[:, 2], P[:, 0] + 1j * P[:, 1]], axis=1)


def _null(Q):
    """A null vector of X(q), i.e. a spanning vector of ker X(q)."""
    return np.stack([-(Q[:, 0] - 1j * Q[:, 1]), 1.0 + Q[:, 2]], axis=1)


def _rotation_taking(m, target=(0.0, 0.0, 1.0)):
    """The rotation R with R m = target (m a unit vector)."""
    m = np.asarray(m, dtype=float)
    m = m / np.linalg.norm(m)
    b = np.asarray(target, dtype=float)
    v = np.cross(m, b)
    c = float(m @ b)
    if c < -1.0 + 1e-12:
        # antipodal: rotate by pi about any axis orthogonal to m
        axis = np.array([1.0, 0.0, 0.0])
        if abs(m[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - (axis @ m) * m
        axis = axis / np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    if c > 1.0 - 1e-15:
        return np.eye(3)
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def _safe_rotation(P):
    """A rotation R such that no point of R P is at the south pole.

    The spinor chart used by the DLT is degenerate at the south pole
    (``z = (1 + p3, p1 + i p2)`` vanishes there).  Every candidate direction
    ``m`` gives a rotation whose third coordinate is ``p . m``, so the south
    pole is reached exactly when ``p . m = -1``; we maximise the smallest
    ``1 + p . m`` over a deterministic candidate set.  Only an *exact* hit is
    fatal (an orthonormal pair of data points makes the optimum small, not
    zero, and small is fine: the chart is polynomial, not rational, in ``p``).
    """
    P = np.asarray(P, dtype=float)
    cands = []
    for sx in (-1.0, 0.0, 1.0):
        for sy in (-1.0, 0.0, 1.0):
            for sz in (-1.0, 0.0, 1.0):
                if sx or sy or sz:
                    cands.append(np.array([sx, sy, sz]))
    mean = P.mean(axis=0)
    if np.linalg.norm(mean) > 1e-9:
        cands.append(mean / np.linalg.norm(mean))
    for p in P:
        if np.linalg.norm(p) > 1e-12:
            cands.append(p / np.linalg.norm(p))
            cands.append(-p / np.linalg.norm(p))
    for i in range(min(len(P), 8)):
        for j in range(i + 1, min(len(P), 8)):
            s = P[i] + P[j]
            if np.linalg.norm(s) > 1e-9:
                cands.append(s / np.linalg.norm(s))
    best, best_score = None, -np.inf
    for m in cands:
        score = float(np.min(1.0 + P @ m))
        if score > best_score + 1e-14:
            best, best_score = m, score
    if best_score <= 1e-13:
        raise GaugeError("no chart frame avoids the south pole exactly; "
                         f"degenerate data (best 1 + p.m = {best_score:.3e})")
    return _rotation_taking(best)


def _lorentz_from_matrix(A):
    """The 4x4 Lorentz matrix of the map X -> A X A^dagger.

    The map is conformal in the Lorentz sense up to the scalar |det A|^2
    (det(A X A^dagger) = |det A|^2 det X), so A is first rescaled to unit
    determinant modulus; the resulting matrix is then exactly in O(3,1).
    """
    A = np.asarray(A, dtype=complex)
    d = abs(np.linalg.det(A))
    if d < 1e-300:
        raise GaugeError("degenerate 2x2 matrix (det = 0) does not define a Möbius map")
    A = A / np.sqrt(d)
    lam = np.empty((4, 4))
    for k, S in enumerate(_SIGMA):
        Y = A @ S @ A.conj().T
        lam[0, k] = Y[0, 1].real
        lam[1, k] = -Y[0, 1].imag
        lam[2, k] = 0.5 * (Y[0, 0] - Y[1, 1]).real
        lam[3, k] = 0.5 * (Y[0, 0] + Y[1, 1]).real
    return lam


# ------------------------------------------------------------------------ map
class Mobius:
    """A Möbius transformation of ``S^2``, as a projective O(3,1) matrix."""

    def __init__(self, matrix):
        M = np.array(matrix, dtype=float, copy=True)
        if M.shape != (4, 4):
            raise GaugeError("a sphere Möbius map is a 4x4 matrix")
        # A conformal automorphism is an O(3,1) matrix only up to a positive
        # scalar (X -> A X A^dagger scales the invariant form by |det A|^2), so
        # fix that scalar first and then check the defining identity.
        g = M.T @ ETA @ M
        s = float(np.trace(ETA @ g)) / 4.0
        if not (s > 0.0):
            raise GaugeError("matrix does not preserve the Minkowski form up to scale")
        M = M / np.sqrt(s)
        err = float(np.max(np.abs(M.T @ ETA @ M - ETA)))
        if err > 1e-8:
            raise GaugeError(f"matrix is not in O(3,1): max |M^T eta M - eta| = {err:.3e}")
        # keep the sign that makes the map act on (p, 1) with positive scale
        if float(M[3, 3]) < 0.0:
            M = -M
        self.matrix = M

    # -- basic facts
    @property
    def det(self):
        return float(np.linalg.det(self.matrix))

    @property
    def orientation(self):
        """+1 for a conformal map, -1 for a mirror (anti-Möbius) map."""
        return 1 if self.det > 0.0 else -1

    def is_identity(self, tol=1e-12):
        return float(np.max(np.abs(self.matrix - np.eye(4)))) < tol

    # -- action
    def apply(self, P):
        P = as_points(P)
        return _dehomog(_homog(P) @ self.matrix.T)

    def __matmul__(self, other):
        """``self @ other`` is ``self`` after ``other``."""
        return Mobius(self.matrix @ other.matrix)

    def inverse(self):
        return Mobius(ETA @ self.matrix.T @ ETA)

    def __repr__(self):
        return (f"Mobius(orientation={self.orientation:+d}, "
                f"det={self.det:+.3e})")


def fit_mobius(P, Q, orientation=1):
    """The Möbius map taking ``P[i]`` to ``Q[i]`` (least squares).

    ``orientation=-1`` fits the mirror family instead.  With three or more
    correspondences in general position the solution is unique (three
    correspondences determine a sphere Möbius map) and the algebraic residual
    is zero to working precision; with more it is the least-squares fit.

    Returns ``(map, info)``; ``info`` carries the singular values of the DLT
    system and the algebraic residual, so a caller can refuse a degenerate fit
    instead of silently returning a bad map.
    """
    P = as_points(P)
    Q = as_points(Q)
    if len(P) != len(Q):
        raise GaugeError(f"{len(P)} source points but {len(Q)} targets")
    if len(P) < 3:
        raise GaugeError(f"a sphere Möbius map needs 3 correspondences, got {len(P)}")
    if orientation not in (1, -1):
        raise GaugeError("orientation must be +1 or -1")
    R = _safe_rotation(np.vstack([P, Q]))
    Pr, Qr = P @ R.T, Q @ R.T
    zp = _spin(Pr)
    if orientation < 0:
        zp = np.conj(zp)
    kq = _null(Qr)
    rows = np.empty((len(P), 4), dtype=complex)
    rows[:, 0] = np.conj(kq[:, 0]) * zp[:, 0]
    rows[:, 1] = np.conj(kq[:, 0]) * zp[:, 1]
    rows[:, 2] = np.conj(kq[:, 1]) * zp[:, 0]
    rows[:, 3] = np.conj(kq[:, 1]) * zp[:, 1]
    # full_matrices=True is required: with exactly three correspondences the
    # system is 3x4 and the null vector is the fourth row of Vh, which
    # full_matrices=False does not return.
    _, sv, Vh = np.linalg.svd(rows, full_matrices=True)
    # Rank 3 of the 4-column system is the numerical statement that the
    # correspondences are non-degenerate (three points in general position).
    # *Consistency* is a different question -- an over-determined inconsistent
    # system still has a least-squares answer -- so it is reported, not raised.
    rank = int(np.sum(sv > 1e-10 * max(sv[0], 1e-300)))
    if rank < 3:
        raise GaugeError("Möbius fit is degenerate: the correspondences span only "
                         f"rank {rank} (duplicate or collinear points), "
                         f"singular values {sv}")
    # the right singular vector for a complex system is the CONJUGATE of the
    # corresponding row of Vh (rows = U S Vh and V = conj(Vh)^T)
    A = np.conj(Vh[-1]).reshape(2, 2)
    lam = _lorentz_from_matrix(A)
    rot = np.eye(4)
    rot[:3, :3] = R
    # q = R p, so a map in the rotated frame is conjugated by diag(R, 1)
    M = rot.T @ lam @ rot
    if orientation < 0:
        # the mirror is the y-reflection expressed in the rotated frame
        M = M @ (rot.T @ MIRROR @ rot)
    mob = Mobius(M)
    dev = geodesic_distance(mob.apply(P), Q)
    inconsistent = bool(len(rows) > 3 and sv[-1] > 1e-10 * max(sv[0], 1e-300))
    info = {"singular_values": sv,
            "algebraic_residual": float(sv[-1]),
            "inconsistent": inconsistent,
            "max_angle": float(np.max(dev)) if len(dev) else 0.0,
            "rms_angle": float(np.sqrt(np.mean(dev ** 2))) if len(dev) else 0.0}
    return mob, info


def geodesic_distance(P, Q):
    """Angular distance between corresponding unit vectors, in radians."""
    P = as_points(P)
    Q = as_points(Q)
    if P.shape != Q.shape:
        raise GaugeError(f"shape mismatch {P.shape} vs {Q.shape}")
    # 2 atan2(|p - q|, |p + q|) rather than arccos(p . q): the arccos form has
    # a square-root singularity at zero angle and reports 1e-8 for an exact
    # match, which would swamp every residual quoted below.
    return 2.0 * np.arctan2(np.linalg.norm(P - Q, axis=1),
                            np.linalg.norm(P + Q, axis=1))


def register(P, Q, orientation="auto"):
    """The Möbius map taking the labelled configuration ``P`` to ``Q``.

    ``orientation="auto"`` tries both families and keeps the better one; the
    verdict reports which was used, because a mirror match is a real fact about
    the input (a stereographic convention, for instance) and not something to
    hide.
    """
    if orientation == "auto":
        best = None
        for o in (1, -1):
            try:
                mob, info = fit_mobius(P, Q, orientation=o)
            except GaugeError:
                continue
            if best is None or info["max_angle"] < best[1]["max_angle"]:
                best = (mob, info, o)
        if best is None:
            raise GaugeError("no Möbius registration could be fitted")
        mob, info, o = best
        info = dict(info, orientation=o)
        return mob, info
    mob, info = fit_mobius(P, Q, orientation=orientation)
    info = dict(info, orientation=orientation)
    return mob, info


def mobius_distance(P, Q, orientation="auto"):
    """A Möbius-invariant distance between two labelled configurations.

    The maximum geodesic displacement that remains after the best Möbius
    registration.  Zero means the two configurations are the same point of
    Teichmüller space.
    """
    _mob, info = register(P, Q, orientation=orientation)
    return info


# --------------------------------------------------------------- invariants
def cross_ratio(P, quad):
    """The cross ratio of a labelled quadruple, invariant under Möbius maps.

    Computed with spinor brackets, so it has no distinguished point and no
    chart singularity.  A mirror map conjugates it.
    """
    P = as_points(P)
    i, j, k, l = quad
    R = _safe_rotation(P)
    z = _spin(P @ R.T)

    def det(a, b):
        return z[a, 0] * z[b, 1] - z[a, 1] * z[b, 0]

    num = det(i, j) * det(k, l)
    den = det(i, l) * det(k, j)
    if abs(den) < 1e-300:
        raise GaugeError(f"degenerate quadruple {quad}")
    return complex(num / den)


def signature(P, quads=None, k=48, seed=0):
    """Möbius-invariant coordinates of a labelled configuration.

    With no ``quads`` argument a deterministic ``k``-sample of quadruples is
    used (a diagnostic, not a complete invariant).
    """
    P = as_points(P)
    n = len(P)
    if quads is None:
        if n < 4:
            return np.zeros(0, dtype=complex)
        if n <= 12:
            quads = list(itertools.combinations(range(n), 4))
        else:
            rng = np.random.default_rng(seed)
            quads = [tuple(sorted(rng.choice(n, size=4, replace=False)))
                     for _ in range(k)]
            quads = sorted(set(quads))
    return np.array([cross_ratio(P, q) for q in quads], dtype=complex)


def signature_residual(P, Q, quads=None):
    """Max |cross ratio difference| between two labelled configurations."""
    sP, sQ = signature(P, quads), signature(Q, quads)
    if sP.shape != sQ.shape:
        raise GaugeError("signatures have different lengths")
    if len(sP) == 0:
        return 0.0
    return float(np.max(np.abs(sP - sQ)))


# -------------------------------------------------------------------- gauges
EQUILATERAL = np.array([
    [0.0, 0.0, 1.0],
    [2.0 * np.sqrt(2.0) / 3.0, 0.0, -1.0 / 3.0],
    [-np.sqrt(2.0) / 3.0, np.sqrt(2.0 / 3.0), -1.0 / 3.0],
])


def pin_gauge(P, anchors, targets=None):
    """Pin the Möbius freedom by sending three labelled vertices to fixed targets.

    This is the explicit gauge: three non-collinear vertices of the realisation
    (by default the three corners of the largest-area face, chosen by the
    caller) go to a fixed equilateral frame.  The map is unique, so the
    representative is reproducible, and the reported deviation says how far the
    pinned configuration is from the input one (zero for a clean pin).
    """
    P = as_points(P)
    if len(anchors) != 3:
        raise GaugeError("pin_gauge needs exactly three anchor vertices")
    if targets is None:
        targets = EQUILATERAL
    targets = as_points(targets)
    mob, info = fit_mobius(P[list(anchors)], targets, orientation=1)
    Q = mob.apply(P)
    dev = geodesic_distance(Q[list(anchors)], targets)
    return mob, Q, {"anchor_angle": float(np.max(dev)),
                    "orientation": mob.orientation,
                    "algebraic_residual": info["algebraic_residual"]}


def centred_gauge(P, weights=None):
    """A rotation gauge: centroid at the north pole, then a fixed meridian.

    Two deterministic rotations (no Möbius boosts), so this pins the rotational
    part of the gauge only; it is exactly equivariant under rotations of the
    input, which is what makes it usable for reporting, and it deliberately
    does **not** claim to pin the full six-dimensional Möbius freedom (use
    ``pin_gauge`` for that).
    """
    P = as_points(P)
    n = len(P)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise GaugeError("weights must have one entry per point")
    w = w / np.sum(w)
    c = w @ P
    if np.linalg.norm(c) < 1e-12:
        raise GaugeError("the weighted centroid is the origin; already centred")
    R1 = _rotation_taking(c / np.linalg.norm(c))
    Q = P @ R1.T
    # meridian: rotate about z so that the heaviest vertex (ties: lowest index)
    # has y = 0 and x > 0
    j = int(np.argmax(w))
    x, y = Q[j, 0], Q[j, 1]
    if abs(x) < 1e-14 and abs(y) < 1e-14:
        order = np.argsort(-w, kind="stable")
        j = next((int(t) for t in order
                  if abs(Q[t, 0]) > 1e-14 or abs(Q[t, 1]) > 1e-14), None)
        if j is None:
            raise GaugeError("every vertex is on the pole axis; meridian undefined")
        x, y = Q[j, 0], Q[j, 1]
    theta = -np.arctan2(y, x)
    ct, st = np.cos(theta), np.sin(theta)
    R2 = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
    M = np.eye(4)
    M[:3, :3] = R2 @ R1
    return Mobius(M), Q @ R2.T


def register_on_labels(P, labels_p, Q, labels_q, orientation="auto"):
    """Register two realisations through their shared vertex labels.

    The labels, not the array order, define the correspondence; this is what
    makes the comparison of a solver output with an oracle (or with the base
    of a lift) meaningful.
    """
    P = as_points(P)
    Q = as_points(Q)
    labels_p = [str(l) for l in labels_p]
    labels_q = [str(l) for l in labels_q]
    if len(set(labels_p)) != len(labels_p) or len(set(labels_q)) != len(labels_q):
        raise GaugeError("labels are not unique")
    if len(labels_p) != len(P) or len(labels_q) != len(Q):
        raise GaugeError("labels and points disagree in length")
    where_q = {l: i for i, l in enumerate(labels_q)}
    shared = [l for l in labels_p if l in where_q]
    if len(shared) < 3:
        raise GaugeError(f"{len(shared)} shared labels; a Möbius registration "
                         "needs three")
    src_idx = [labels_p.index(l) for l in shared]
    dst_idx = [where_q[l] for l in shared]
    mob, info = register(P[src_idx], Q[dst_idx], orientation=orientation)
    info = dict(info, shared=len(shared))
    return mob, info


def rotation_gauge(P, weights=None):
    """A fully general, deterministic rotation gauge.

    Two labelled choices fix the three rotational degrees of freedom:

    1. the vertex with the largest weight (ties: lowest index) is rotated to the
       north pole;
    2. the vertex farthest from it (ties: lowest index) is rotated about the
       pole axis onto the positive x half of the xz-plane.

    Both choices are label-based, so the construction is exactly equivariant
    under rotations of the input: ``rotation_gauge(R P) == rotation_gauge(P)``.
    It is *not* equivariant under boosts -- see ``centred_gauge`` and the
    negative control in the tests -- so it pins at most three of the six Möbius
    degrees of freedom.  Unlike ``centred_gauge`` it never fails on a
    centroid-symmetric configuration (the octahedron, for instance, has its
    centroid at the origin).
    """
    P = as_points(P)
    n = len(P)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise GaugeError("weights must have one entry per point")
    j0 = int(np.argmax(w))
    R1 = _rotation_taking(P[j0])
    Q = P @ R1.T
    dist = geodesic_distance(Q, np.repeat(Q[j0][None, :], n, axis=0))
    # Distances are only grouping keys: a rotation must not change *which*
    # vertex is used, so the vertices are scanned in distance shells (ties
    # broken by index) rather than in raw floating-point order.
    order = sorted((j for j in range(n) if j != j0), key=lambda j: -dist[j])
    shells = []
    for j in order:
        if shells and dist[j] >= shells[-1][0] - 1e-9:
            shells[-1][1].append(j)
        else:
            shells.append([dist[j], [j]])
    for _d, members in shells:
        for j1 in sorted(members):
            x, y = Q[j1, 0], Q[j1, 1]
            if abs(x) < 1e-12 and abs(y) < 1e-12:
                continue                # on the pole axis: gives no meridian
            theta = -np.arctan2(y, x)
            ct, st = np.cos(theta), np.sin(theta)
            R2 = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
            M = np.eye(4)
            M[:3, :3] = R2 @ R1
            return Mobius(M), Q @ R2.T
    raise GaugeError("every vertex lies on the pole axis; no meridian exists")


# -------------------------------------------------------------------- gluing
class Patch:
    """A realisation of a labelled subset of one configuration."""

    def __init__(self, labels, points, weights=None):
        self.labels = [str(l) for l in labels]
        if len(set(self.labels)) != len(self.labels):
            raise GaugeError("patch labels are not unique")
        self.points = as_points(points)
        if len(self.labels) != len(self.points):
            raise GaugeError("labels and points disagree in length")
        self.weights = weights
        self._index = {l: i for i, l in enumerate(self.labels)}

    def __len__(self):
        return len(self.labels)

    def get(self, label):
        return self.points[self._index[label]]


def glue_patches(patches, tol=1e-7, require="all"):
    """Glue Möbius-ambiguous realisations into one configuration.

    Each patch is only known up to a Möbius map, so every patch enters with an
    unknown freedom.  A patch that shares at least three labels with the part
    already assembled determines its freedom by a Möbius fit on the overlap;
    the overlap residual says whether the pieces are consistent.

    Three shared labels determine the freedom *and nothing more*: sphere Möbius
    maps are three-transitive, so a three-point overlap always has a solution
    and an inconsistent patch cannot be detected from it.  Consistency is only
    testable on an overlap of at least four labels (and even then only up to
    the precision of the fit); the verdict reports the overlap size so that a
    vacuous "consistent" cannot be mistaken for a check.

    Returns a dict with the assembled points (by label), the per-patch map and
    residual, and a verdict.  ``require="all"`` raises on an inconsistent
    overlap; ``require="report"`` returns the verdict instead.
    """
    patches = list(patches)
    if not patches:
        raise GaugeError("no patches to glue")
    order = sorted(range(len(patches)), key=lambda i: (-len(patches[i]), i))
    ref = order[0]
    assembled = dict(zip(patches[ref].labels, patches[ref].points))
    mobius = {ref: None}
    residual = {ref: 0.0}
    overlap = {ref: len(patches[ref])}
    bad = []
    for i in order[1:]:
        patch = patches[i]
        shared = [l for l in patch.labels if l in assembled]
        overlap[i] = len(shared)
        if len(shared) < 3:
            raise GaugeError(f"patch {i} shares {len(shared)} labels with the "
                             "assembled part; three are needed to determine its "
                             "Möbius freedom")
        src = np.array([patch.get(l) for l in shared])
        dst = np.array([assembled[l] for l in shared])
        mob, info = register(src, dst, orientation="auto")
        moved = mob.apply(patch.points)
        mobius[i] = mob
        residual[i] = info["max_angle"]
        if info["max_angle"] > tol:
            bad.append(i)
        for l, p in zip(patch.labels, moved):
            assembled[l] = p
    verdict = {
        "consistent": not bad,
        "inconsistent_patches": bad,
        "overlap": overlap,
        "residual": residual,
        "tol": tol,
        "orientation": {i: (mobius[i].orientation if mobius[i] is not None else None)
                        for i in mobius},
    }
    if bad and require == "all":
        raise GaugeError(f"patches {bad} are inconsistent with the assembled "
                         f"configuration (residuals "
                         f"{ {i: residual[i] for i in bad} })")
    labels = [l for i in order for l in patches[i].labels]
    seen, uniq = set(), []
    for l in labels:
        if l not in seen:
            seen.add(l)
            uniq.append(l)
    points = np.array([assembled[l] for l in uniq])
    return {"labels": uniq, "points": points, "mobius": mobius,
            "verdict": verdict, "patch_order": order}
