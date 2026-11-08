"""A precise acceptance invariant for round realisations on a fixed complex.

A round realisation of a triangulated S^2 is a map x: vertices -> S^2.  Two
independent things have to be certified, and neither is the conformal residual:

  (1) LABELLED COMPARISON.  Two configurations that differ by a Mobius map are
      the same point of the chart.  `pullback.discrete.chart_coordinate` already
      gives a Mobius-invariant labelled coordinate for any number of points
      (cross-ratio for |P| = 4, a tuple of cross-ratios beyond that), so
      configurations are compared there and nowhere else.

  (2) EMBEDDING.  The realisation must be a homeomorphism onto S^2, not a folded
      or collapsed image.  For *chordal* triangles there is an exact, cheap
      certificate.  Take the complex with a consistently oriented face list
      (orient_faces propagates one and refuses a non-orientable complex).  Then

          det(a, b, c) = a . (b x c) = 2 * (signed solid angle numerator)

      is, up to a positive factor, the signed distance of the origin from the
      plane of the face: det = n . a with n = (b-a) x (c-a), so the plane's
      equation is n . y = det and the origin lies on the inner side iff
      det > 0.  Hence

          all det_f > 0   <=>   the origin is strictly inside every supporting
                                half-space   ==>   the polyhedral surface is the
                                boundary of a region containing the origin
                                ==>   the winding number is 1
                                ==>   radial projection to S^2 is a
                                      homeomorphism.

      That is the embedding certificate.  It needs no area formula with an
      atan2 branch and no hemisphere assumption: `det > 0` for every face is
      exactly "the closed polyhedron contains the origin and is not
      self-intersecting".  `min |det|` measures how far from degenerate it is.
      (A configuration can be embedded with every cone angle above 2*pi: the
      cone angle is a sum of *Euclidean* angles in the tangent-ish planes and is
      not bounded by the solid angles.)

Notice what is NOT in the invariant: the conformal residual.  Residual ~ 1e-15
is satisfied by folds and by collapses, and on the rabbit's coarse complex it is
satisfied by a whole family of mutually inequivalent branches (see branch_sweep).
"""

from __future__ import annotations

import itertools

import numpy as np

__all__ = ["orient_faces", "pair_labels", "mobius_coordinate", "face_determinants", "embedding_certificate",
           "classify", "branch_sweep", "spherical_area"]


def orient_faces(faces, labels):
    """A consistently oriented copy of a labelled face list, or a refusal.

    `labels[k][j]` must be the edge id of the edge from `faces[k][j]` to
    `faces[k][j+1]`.  Edge ids, not vertex pairs: the coarse rabbit complex has
    two distinct edges joining vertices 0 and 5, and pairing by vertices would
    make it look non-manifold.  One orientation is propagated across the dual
    graph: two faces sharing an edge must traverse it in opposite directions.
    """
    faces = [tuple(f) for f in faces]
    labels = [tuple(l) for l in labels]
    for f in faces:
        if len(f) != 3 or len(set(f)) != 3:
            raise ValueError(f"face {f} is not a triangle")
    edge_faces = {}
    for k, ls in enumerate(labels):
        for pos, e in enumerate(ls):
            edge_faces.setdefault(e, []).append((k, pos))
    for e, ks in edge_faces.items():
        if len(ks) != 2:
            raise ValueError(f"edge {e} is in {len(ks)} faces, not 2")

    def variants(triple, label_triple):
        """The six (triple, labels) presentations, cycling the SAME way."""
        out = []
        for r in range(3):
            t = triple[r:] + triple[:r]
            l = label_triple[r:] + label_triple[:r]
            out.append((t, l))
        rev = triple[::-1]
        rev_l = (label_triple[2], label_triple[1], label_triple[0])
        for r in range(3):
            t = rev[r:] + rev[:r]
            l = rev_l[r:] + rev_l[:r]
            out.append((t, l))
        return out

    oriented = {0: (faces[0], labels[0])}
    stack = [0]
    while stack:
        k = stack.pop()
        ka, kb, kc = oriented[k][0]
        for pos, e in enumerate(labels[k]):
            for other, _pos in edge_faces[e]:
                if other in oriented:
                    continue
                # face k traverses the shared edge; find its direction there
                triple_k, labels_k = oriented[k]
                j = labels_k.index(e)
                a, b = triple_k[j], triple_k[(j + 1) % 3]
                for t, l in variants(faces[other], labels[other]):
                    if (t[0], t[1]) == (b, a):
                        oriented[other] = (t, l)
                        break
                else:
                    raise ValueError(f"face {other} cannot be oriented opposite "
                                     f"to face {k} across edge {e}")
                stack.append(other)
    if len(oriented) != len(faces):
        raise ValueError("complex is not connected")
    return tuple(oriented[k][0] for k in range(len(faces)))


def pair_labels(faces):
    """Edge ids for a simplicial complex: the sorted vertex pair."""
    return [tuple((f[j], f[(j + 1) % 3]) if f[j] < f[(j + 1) % 3]
                  else (f[(j + 1) % 3], f[j]) for j in range(3))
            for f in faces]


def face_determinants(x, faces):
    x = np.asarray(x, dtype=float)
    return np.array([float(np.linalg.det(np.array([x[a], x[b], x[c]])))
                     for a, b, c in faces])


def spherical_area(a, b, c):
    """Signed solid angle of the chordal triangle, in (-2*pi, 2*pi).

    Oosterom-Strackee with the atan2 branch, i.e. the solid angle of the cone
    over the flat triangle, negative when the origin is on the far side.
    """
    det = float(np.linalg.det(np.array([a, b, c])))
    den = 1.0 + float(a @ b + b @ c + c @ a)
    return float(2.0 * np.arctan2(det, den))


def embedding_certificate(x, oriented_faces, tol=1e-9, inject_tol=1e-7):
    """Is `x` an embedded realisation of the consistently oriented complex?

    Three conditions, all necessary and together sufficient:
      * every vertex position is distinct (injectivity -- two non-adjacent
        vertices may coincide without making any face degenerate, and then the
        signed area sum is 8*pi rather than 4*pi, which is how the omission
        shows up);
      * no face is degenerate (det != 0);
      * all face determinants have one sign, i.e. the origin is strictly inside
        every supporting plane, so the winding number is 1 and radial
        projection to S^2 is a homeomorphism.
    """
    x = np.asarray(x, dtype=float)
    rounded = [tuple(np.round(v, 8)) for v in x]
    if len(set(rounded)) != len(rounded):
        pairs = [(i, j) for i in range(len(x)) for j in range(i + 1, len(x))
                 if float(np.linalg.norm(x[i] - x[j])) <= inject_tol]
        return {"embedded": False, "det": face_determinants(x, oriented_faces),
                "min_abs_det": 0.0, "signs": 0,
                "reason": f"{len(x) - len(set(rounded))} coincident vertex "
                          f"pair(s), e.g. {pairs[:3]} -- not injective"}
    det = face_determinants(x, oriented_faces)
    if np.any(np.abs(det) <= tol):
        return {"embedded": False, "det": det,
                "reason": f"{int((np.abs(det) <= tol).sum())} degenerate face(s)",
                "min_abs_det": float(np.abs(det).min()),
                "signs": int(np.sign(det).sum())}
    positive = bool((det > 0).all())
    negative = bool((det < 0).all())
    return {"embedded": positive or negative,
            "det": det,
            "orientation": "+" if positive else ("-" if negative else "mixed"),
            "min_abs_det": float(np.abs(det).min()),
            "reason": "" if (positive or negative) else
                      f"folded: {int((det < 0).sum())} of {len(det)} faces "
                      "have the opposite orientation"}


def mobius_coordinate(x, tol=1e-8):
    """A gauge-covariant labelled coordinate for a configuration of any size.

    Splits a Mobius map off by sending a triple of distinct points to
    (infinity, 0, 1) and reading the rest as cross-ratios.  The triple is the
    first one (lexicographic) whose three points are pairwise distinct -- a
    Mobius-invariant predicate, since Mobius maps are injective -- so the
    coordinate does not depend on the solver's arbitrary normalisation, and it
    survives coincident points elsewhere in the configuration (the coarse
    complex's collapse has two such pairs, which is why
    discrete.chart_coordinate's fixed (0,1,2) triple divides by zero there).
    """
    from pullback.mesh import to_point
    from pullback.moduli import cross_ratio
    import itertools

    x = np.asarray(x, dtype=float)
    pts = [to_point(v) for v in x]
    n = len(pts)
    for i, j, k in itertools.combinations(range(n), 3):
        if (np.linalg.norm(x[i] - x[j]) > tol
                and np.linalg.norm(x[j] - x[k]) > tol
                and np.linalg.norm(x[i] - x[k]) > tol):
            rest = [l for l in range(n) if l not in (i, j, k)]
            return ((i, j, k), tuple(cross_ratio(pts[i], pts[j], pts[k], pts[l])
                                     for l in rest))
    raise ValueError("configuration has fewer than three distinct points")


def classify(x, oriented_faces, points=None):
    """The full acceptance record for one candidate realisation."""
    from pullback import discrete

    cert = embedding_certificate(x, oriented_faces)
    x = np.asarray(x, dtype=float)
    areas = [spherical_area(x[a], x[b], x[c]) for a, b, c in oriented_faces]
    out = {"embedded": cert["embedded"],
           "min_abs_det": cert["min_abs_det"],
           "det": cert["det"],
           "reason": cert["reason"],
           "signed_area_sum": float(sum(areas)),
           "coordinate": mobius_coordinate(x),
           "distinct_positions": len({tuple(np.round(v, 6)) for v in x})}
    return out


def branch_sweep(faces, lengths, n, starts, labels=None, maxit=600, tol=1e-14,
                 seed=0):
    """Solve from many starts and group the results by Mobius-invariant
    coordinate, then certify each branch.  Empirical, not a proof: it finds the
    branches the local solver can reach, and `embedded` on a branch is exact.
    """
    from pullback.rounding import round_realise

    oriented = (tuple(faces) if labels is None
                else orient_faces(faces, labels))
    rng = np.random.RandomState(seed)
    branches = {}
    tried = failed = 0
    for x0 in starts:
        tried += 1
        x, res, it = round_realise(oriented, lengths, x0=x0, maxit=maxit,
                                   tol=tol)
        if res > 1e-10:
            failed += 1
            continue
        rec = classify(x, oriented)
        key = (rec["coordinate"][0],
               tuple((round(float(np.real(z)), 7), round(float(np.imag(z)), 7))
                     for z in rec["coordinate"][1]))
        if key not in branches:
            rec["count"] = 0
            rec["example"] = x
            rec["residual"] = res
            branches[key] = rec
        branches[key]["count"] += 1
    return {"tried": tried, "failed": failed,
            "branches": sorted(branches.values(),
                               key=lambda r: -r["count"]),
            "n_vertices": n}
