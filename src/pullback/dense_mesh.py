"""Dense spherical base meshes, with optional ramification-protected rings.

A caller supplies spherical ``points`` (the postcritical set for a pullback)
and optional branch-ring centres; these are arbitrary unit 3-vectors, not
polynomial preimages or square-root-specific data.  Background vertices come
from a rotated icosphere.  Ring vertices are sampled in a local tangent
chart.  The final triangulation is simply the spherical convex hull of the
result.  No sheet assignment, projection, square-root formula, chordal lift,
or rational-map answer is hidden in the construction.

The returned entries have ``(vertices, edge_labels)`` and are directly usable
by :mod:`pullback.dense_assembly`.  Rings are a named choice of discretisation;
there is no assertion that being denser or Möbius-invariant is itself a
correctness guarantee.  Branch-ring intrusion must be diagnosed from the
resulting flowers and metric quality, not hidden by the ring metadata.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull

from . import dense_assembly


class DenseMeshError(ValueError):
    """The requested base is not a genuine nondegenerate spherical hull."""


def icosphere(level=1):
    """Unit vectors of a level-``level`` 1-to-4 icosahedral subdivision."""
    level = int(level)
    if level < 0:
        raise ValueError("level must be nonnegative")
    phi = (1 + 5 ** 0.5) / 2
    V = [(-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
         (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
         (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1)]
    V = [np.array(v) / np.linalg.norm(v) for v in V]
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
         (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
         (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
         (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(level):
        cache, nF = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = V[a] + V[b]
                V.append(m / np.linalg.norm(m))
                cache[key] = len(V) - 1
            return cache[key]
        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nF += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        F = nF
    return np.array(V)


def unit(v):
    v = np.asarray(v, dtype=float)
    if v.shape != (3,) or not np.isfinite(v).all() or np.linalg.norm(v) == 0:
        raise ValueError("point must be a finite nonzero 3-vector of the sphere")
    return v / np.linalg.norm(v)


def _tangent_basis(v):
    v = unit(v)
    seed = np.array([0.0, 0.0, 1.0])
    if abs(seed @ v) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    e1 = seed - (seed @ v) * v
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(v, e1)


def branch_ring(centre, radius, count=12, phase=0.0):
    """Points at ``radius`` radians around ``centre`` in its tangent chart."""
    r = float(radius)
    if not 0.0 < r < np.pi:
        raise ValueError("ring radius must lie strictly between 0 and pi")
    count = int(count)
    if count < 3:
        raise ValueError("a branch ring needs at least three points")
    e1, e2 = _tangent_basis(centre)
    return [np.cos(r) * unit(centre) + np.sin(r) *
            (np.cos(2 * np.pi * k / count + phase) * e1
             + np.sin(2 * np.pi * k / count + phase) * e2)
            for k in range(count)]


@dataclass
class DenseBase:
    points: np.ndarray
    tris: list
    puncture_indices: list
    level: int
    ring_records: list

    def as_assembly_base(self):
        return dense_assembly.Base(self.tris)

    def certificate(self):
        base = self.as_assembly_base()
        if base.euler() != 2:
            raise DenseMeshError(f"base hull has Euler characteristic {base.euler()}")
        for v in base.vertices:
            base.flower(v)
        return {"V": base.n_vertices(), "E": base.n_edges(),
                "F": base.n_faces(), "chi": base.euler(),
                "punctures": list(self.puncture_indices), "level": self.level,
                "rings": self.ring_records}


def dense_base(points, level=1, rings=None, rotation_seed=7):
    """Build a dense oriented spherical base around prescribed points.

    ``rings`` is a list of dictionaries with keys ``point``, ``count``,
    ``radius``. This deliberately expresses arbitrary ramification choices
    independently of the cover degree.  The caller is not required to use
    identical rings or one common radius, so high ramification or close
    branch points can have their own local coarse/fine strategy.
    """
    P = [unit(p) for p in points]
    if len(P) < 4:
        raise ValueError("a spherical base needs at least four prescribed points")
    if len({tuple(np.round(x, 13)) for x in P}) != len(P):
        raise DenseMeshError("prescribed points are coincident at the requested precision")
    if not 0 <= int(level) <= 6:
        raise ValueError("level must be between 0 and 6 for this dense builder")
    rng = np.random.default_rng(rotation_seed)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    candidates = list(icosphere(int(level)) @ Q.T)
    records = []
    for ring in rings or []:
        centre = unit(ring["point"])
        count = int(ring.get("count", 12))
        record = {"point": centre, "count": count,
                  "radius": float(ring["radius"])}
        records.append(record)
        candidates.extend(branch_ring(
            record["point"], record["radius"], record["count"], 0.37 * len(records)))
    # Delete background points too near prescribed vertices, rather than
    # letting them silently displace the ring or a marked puncture.
    clearance = float(records and min(r["radius"] for r in records) / 2
                      or 0.018)
    if not 0.0 < clearance < 0.1:
        raise DenseMeshError(f"invalid background clearance {clearance}")
    keep = [p for p in candidates
            if min(np.linalg.norm(p - q) for q in P) >= clearance]
    X = keep + P
    # Deduplicate only above a deliberate scale, so points genuinely closer
    # than this are retained as separate data and can fail visibly in qhull.
    seen, vertices = set(), []
    for p in X:
        key = tuple(np.round(p, 13))
        if key not in seen:
            seen.add(key)
            vertices.append(p)
    X = np.array(vertices)
    indices = []
    for q in P:
        d = np.linalg.norm(X - q, axis=1)
        i = int(np.argmin(d))
        if d[i] > 1e-11:
            raise DenseMeshError(f"prescribed point is absent at distance {d[i]:.3e}")
        indices.append(i)
    try:
        hull = ConvexHull(X)
    except Exception as exc:
        raise DenseMeshError(f"spherical hull failed for {len(X)} points: {exc}") from exc
    tris = []
    for simplex in hull.simplices:
        a, b, c = (int(i) for i in simplex)
        if abs(np.linalg.det(np.array([X[a], X[b], X[c]]))) <= 1e-11:
            raise DenseMeshError(f"degenerate spherical facet {(a, b, c)}")
        if float(np.cross(X[b] - X[a], X[c] - X[a]) @ X[a]) < 0.0:
            b, c = c, b
        labels = tuple((min(i, j), max(i, j))
                       for i, j in ((a, b), (b, c), (c, a)))
        tris.append(((a, b, c), labels))
    return DenseBase(X, tris, indices, int(level), records)
