"""Fixtures for the CEPS certifier: exact and high-precision *claims*.

A fixture is a claim, never a solver: points on a sphere, a triangulation, and
(for the input side) a PL metric.  The point of a fixture is that the answer to
the CEPS problem is known in closed form, so a certifier can be checked against
a case whose truth does not depend on any optimiser.

Tiers.  ``regular_tetra``, ``octahedron``, ``rational_tetra``,
``rational_bipyramid`` and anything derived from them by a rational
stereographic Mobius map or by rational vertex scaling live in ``Q^3``: the
points and all squared edge lengths are ``Fraction``, so every sign test in
``ceps_cert`` is a proof.  ``icosahedron`` needs ``sqrt 5`` and the
branched-cover fixtures need the ``d``-th roots of the octahedron's plane
coordinates; both are produced at 50 significant digits with ``mpmath`` and are
numeric tier.

No conformal factor anywhere below is obtained by optimisation.  The vertex
scaling of the "answer is the original configuration" fixture is prescribed in
closed form, and the branched-cover fixtures use the closed-form preimage
points of ``z -> z**d`` together with the *exact* deck transformation
``w -> w * exp(2 pi i / d)``, so an oracle is never an implicit solver.
"""

from __future__ import annotations

import math
from fractions import Fraction

import mpmath as mp

from pullback.ceps_cert import (Complex, canon, mobius_image, sq_lengths,
                                stereo_fwd, stereo_inv, vcross, vdot, vsub)

Q = Fraction


# --------------------------------------------------------------------------
# orientation
# --------------------------------------------------------------------------
def orient_outward(points, faces):
    """Give the complex one coherent orientation, then point it outwards.

    Flipping faces one at a time to make ``n . a > 0`` is *not* enough: the
    reversed list need not have opposite darts on a shared edge.  So the
    orientation is first propagated across the face adjacency graph (a face is
    reversed when its dart on a shared edge runs the same way as the
    neighbour's), and only then is the whole surface reversed if its normals
    point inwards.  For a convex configuration with the origin inside,
    ``n . a > 0`` holds for the outward normal because the three vertices of a
    face have equal norm.
    """
    faces = [tuple(f) for f in faces]
    edge_faces = {}
    for t, f in enumerate(faces):
        for s in range(3):
            edge_faces.setdefault(frozenset((f[s], f[(s + 1) % 3])),
                                  []).append(t)
    for e, ts in edge_faces.items():
        if len(ts) != 2:
            raise ValueError("edge %s is in %d faces" % (tuple(e), len(ts)))
    seen = [False] * len(faces)
    order = [0]
    seen[0] = True
    while order:
        t = order.pop(0)
        f = faces[t]
        for s in range(3):
            a, b = f[s], f[(s + 1) % 3]
            for u in edge_faces[frozenset((a, b))]:
                if u == t or seen[u]:
                    continue
                g = faces[u]
                darts = [(g[i], g[(i + 1) % 3]) for i in range(3)]
                if (a, b) in darts:
                    faces[u] = (g[0], g[2], g[1])
                seen[u] = True
                order.append(u)
    if not all(seen):
        raise ValueError("faces do not form one connected surface")
    # flip everything if the normals point inwards
    inward = 0
    for f in faces:
        a, b, c = (points[i] for i in f)
        nrm = vcross(vsub(b, a), vsub(c, a))
        if vdot(nrm, a) < 0:
            inward += 1
    if inward == len(faces):
        faces = [(f[0], f[2], f[1]) for f in faces]
        flipped = len(faces)
    elif inward == 0:
        flipped = 0
    else:
        raise ValueError("%d of %d faces point inwards: not a convex "
                         "configuration around the origin" % (inward, len(faces)))
    return faces, flipped


def plane_side(points, f, d):
    a, b, c = (points[i] for i in f)
    nrm = vcross(vsub(b, a), vsub(c, a))
    return vdot(nrm, vsub(points[d], a))


# --------------------------------------------------------------------------
# classical configurations
# --------------------------------------------------------------------------
def regular_tetra():
    """Regular tetrahedron, squared edges ``8``, on the sphere of radius ``sqrt 3``."""
    pts = [(Q(1), Q(1), Q(1)), (Q(1), Q(-1), Q(-1)),
           (Q(-1), Q(1), Q(-1)), (Q(-1), Q(-1), Q(1))]
    faces, _ = orient_outward(pts, [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
    return pts, faces


def octahedron():
    """Regular octahedron on the unit sphere; chordal squared edges all ``2``."""
    pts = [(Q(1), Q(0), Q(0)), (Q(-1), Q(0), Q(0)),
           (Q(0), Q(1), Q(0)), (Q(0), Q(-1), Q(0)),
           (Q(0), Q(0), Q(1)), (Q(0), Q(0), Q(-1))]
    ring = [0, 2, 1, 3]                      # counter-clockwise seen from +z
    faces = []
    for t in range(4):
        faces.append((4, ring[t], ring[(t + 1) % 4]))
        faces.append((5, ring[(t + 1) % 4], ring[t]))
    faces, _ = orient_outward(pts, faces)
    return pts, faces


def rational_tetra():
    """Asymmetric tetrahedron with rational vertices on the unit sphere.

    The fourth vertex is ``(-12, -8, -9)/17``, from the rational
    parametrisation of the sphere.  Its three coordinates are negative, which
    is exactly what puts the origin strictly inside the tetrahedron: with the
    other three vertices on the positive axes, the barycentric coefficients are
    ``(12, 8, 9, 17)/46``, all positive.  The obvious candidate ``(2,2,1)/3``
    is the centroid of the other three and is not a vertex at all -- which is
    the kind of thing the supporting-plane test has to catch.
    """
    pts = [(Q(1), Q(0), Q(0)), (Q(0), Q(1), Q(0)), (Q(0), Q(0), Q(1)),
           (Q(-12, 17), Q(-8, 17), Q(-9, 17))]
    faces, _ = orient_outward(pts, [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
    return pts, faces


def sphere_point_rational(u, v):
    """``(2u, 2v, 1 - u^2 - v^2)/(1 + u^2 + v^2)``: rational points on ``S^2``.

    ``(u, v) = (0, 0)`` gives the north pole, which collides with the bipyramid
    apex; the ring below avoids it on purpose.
    """
    u, v = Q(u), Q(v)
    d = 1 + u * u + v * v
    return (2 * u / d, 2 * v / d, (1 - u * u - v * v) / d)


def rational_bipyramid(eq_uv=None):
    """Bipyramid over a rational equatorial polygon: convex, inscribed, exact."""
    if eq_uv is None:
        # a convex rational quadrilateral, in cyclic order around the axis and
        # *not* coplanar, so flipping the equatorial diagonal really changes
        # the metric instead of producing a concyclic tie
        eq_uv = [(1, 0), (1, 2), (-1, 1), (-1, -1)]
    ring = [sphere_point_rational(u, v) for (u, v) in eq_uv]
    nn = len(ring)
    pts = ring + [(Q(0), Q(0), Q(1)), (Q(0), Q(0), Q(-1))]
    north, south = nn, nn + 1
    faces = []
    for t in range(nn):
        faces.append((north, t, (t + 1) % nn))
        faces.append((south, (t + 1) % nn, t))
    faces, _ = orient_outward(pts, faces)
    return pts, faces


def icosahedron(dps=50):
    """Regular icosahedron at ``dps`` digits (needs ``sqrt 5``)."""
    mp.mp.dps = dps
    phi = (1 + mp.sqrt(5)) / 2
    pts = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            pts.append((mp.mpf(0), s1 * mp.mpf(1), s2 * phi))
            pts.append((s1 * mp.mpf(1), s2 * phi, mp.mpf(0)))
            pts.append((s2 * phi, mp.mpf(0), s1 * mp.mpf(1)))
    R = mp.sqrt(sum(c * c for c in pts[0]))
    pts = [tuple(c / R for c in p) for p in pts]
    faces, _ = orient_outward(pts, convex_hull_facets(pts))
    return pts, faces


def convex_hull_facets(pts):
    """Facets of the convex hull for a small point set on a sphere.

    A triple is a facet iff every other point lies strictly on one side of its
    plane.  Quadratic in the number of points, which is fine for the fixtures.
    """
    n = len(pts)
    faces = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = pts[i], pts[j], pts[k]
                nrm = vcross(vsub(b, a), vsub(c, a))
                if vdot(nrm, nrm) == 0:
                    continue
                pos = neg = 0
                for d in range(n):
                    if d in (i, j, k):
                        continue
                    s = vdot(nrm, vsub(pts[d], a))
                    if s > 0:
                        pos += 1
                    elif s < 0:
                        neg += 1
                    else:
                        pos = neg = 1
                        break
                if (pos == 0) != (neg == 0):
                    faces.append((i, j, k))
    return faces


def octahedral_asymmetric():
    """Six rational points in octahedral position with the south pole off axis.

    No four of them are coplanar, so flipping one edge really does change the
    metric instead of producing a concyclic tie.
    """
    pts = [(Q(1), Q(0), Q(0)), (Q(-1), Q(0), Q(0)),
           (Q(0), Q(1), Q(0)), (Q(0), Q(-1), Q(0)),
           (Q(0), Q(0), Q(1)), (Q(4, 9), Q(4, 9), Q(-7, 9))]
    ring = [0, 2, 1, 3]
    faces = []
    for t in range(4):
        faces.append((4, ring[t], ring[(t + 1) % 4]))
        faces.append((5, ring[(t + 1) % 4], ring[t]))
    faces, _ = orient_outward(pts, faces)
    return pts, faces


# --------------------------------------------------------------------------
# derived claims
# --------------------------------------------------------------------------
def chordal_sq(points):
    return sq_lengths(points)


def scaled_metric(sq, k):
    """``l'^2_ij = k_i k_j l^2_ij``: a prescribed vertex scaling."""
    return {e: sq[e] * k[e[0]] * k[e[1]] for e in sq}


def rational_mobius(a, b, c, d):
    """A rational Mobius map of the sphere as a 2x2 complex matrix."""
    return (((Q(a[0]), Q(a[1])), (Q(b[0]), Q(b[1]))),
            ((Q(c[0]), Q(c[1])), (Q(d[0]), Q(d[1]))))


def mobius_claim(points, abcd):
    return mobius_image(points, abcd)


def flipped(points, faces, edge):
    """Same points, ``edge`` flipped: the other diagonal of that quadrilateral."""
    return Complex(len(points), faces).flip(edge).faces


# --------------------------------------------------------------------------
# branched covers of the octahedron by z -> z**d
# --------------------------------------------------------------------------
def branched_cover_octahedron(d, dps=50):
    """The pullback of the octahedron by ``z -> z**d``, ``d = 2`` or ``d >= 2``.

    Returns a dict with

    ``base_points``  the six octahedron vertices on the sphere (high precision);
    ``base_faces``   the eight octahedron faces;
    ``base_sq``      their chordal squared lengths (the *base* PL metric);
    ``lift_points``  all preimages of the base vertices, on the sphere;
    ``lift_faces``   the preimage triangulation;
    ``proj``         lifted vertex -> base vertex;
    ``ld``           lifted vertex -> local degree (1 over the two branch
                     points, ``d`` elsewhere);
    ``lift_sq``      the pullback metric: every lifted edge inherits the base
                     chordal length of the base edge under it.

    The lift has ``2 + 4 d`` vertices and ``8 d`` faces, matching the degree-2
    and degree-3 fixtures of the earlier runs (10/16 and 14/24).

    Construction, with no root tracking and no optimiser:

    * the base is the octahedron in the stereographic chart from the north
      pole, so the north pole is ``inf`` and the south pole is ``0`` -- the two
      branch points of ``z -> z**d`` are base *vertices*, as they must be for
      the preimage of a triangulation to be a triangulation;
    * preimages are computed in closed form, ``w = |z|^(1/d) exp(i arg(z)/d)``
      times the ``d``-th roots of unity;
    * the *combinatorial* cover is assembled by ``dense_assembly`` from the
      local monodromy, which for ``z -> z**d`` is a ``d``-cycle at ``0`` and its
      inverse at ``inf`` and the identity elsewhere;
    * each lifted vertex is then given a position by a breadth-first walk of
      the lifted 1-skeleton, continuing the root along each base edge by the
      closed form ``w' = w (z'/z)^(1/d)`` with the argument reduced to
      ``(-pi/d, pi/d]``.  The walk assigns every lifted vertex exactly once and
      every revisit must agree: a disagreement would mean the assembled cover
      and the geometric preimage complex differ, and would raise.
    """
    mp.mp.dps = dps
    try:
        from pullback.dense_assembly import (assemble, base_from_face_lists,
                                             perms_from_local, perm_from_cycles)
    except ImportError as exc:                       # pragma: no cover
        raise ImportError("the branched-cover fixture needs PYTHONPATH=src: %s"
                          % exc)

    base_pts, base_faces = octahedron()
    # octahedron vertex order: 0=+x, 1=-x, 2=+y, 3=-y, 4=+z (north), 5=-z
    # stereographic chart from the north pole: pole -> inf, -z -> 0
    # the chart sends the north pole to infinity: keep it as ``None``
    finite = []
    for p in base_pts:
        pp = tuple(Q(c) for c in p)
        if pp[2] == 1:
            finite.append(None)
        else:
            z = stereo_fwd(pp)
            finite.append((_to_mpf(z[0]), _to_mpf(z[1])))
    north = [i for i in range(len(base_pts)) if finite[i] is None][0]
    south = [i for i in range(len(base_pts)) if base_pts[i][2] == -1][0]

    base = base_from_face_lists(base_faces)
    cyc = tuple(range(d))
    local = {v: tuple(range(d)) for v in base.vertices}
    local[south] = perm_from_cycles(d, [cyc])
    local[north] = perm_from_cycles(d, [tuple(reversed(cyc))])
    edge_perms = perms_from_local(base, d, local)
    lift = assemble(base, d, edge_perms)

    # positions: closed-form preimages, propagated along the lifted 1-skeleton
    def root(z, k):
        """``k``-th ``d``-th root of the finite plane point ``z`` (a pair)."""
        r = mp.sqrt(z[0] * z[0] + z[1] * z[1])
        th = mp.atan2(mp.mpf(z[1]), mp.mpf(z[0]))
        mag = r ** (mp.mpf(1) / d)
        a = (th + 2 * mp.pi * k) / d
        return (mag * mp.cos(a), mag * mp.sin(a))

    def continue_root(w, zfrom, zto):
        """The root of ``zto`` continuous with ``w`` along the segment.

        ``zto`` must be finite; the argument of ``zto/zfrom`` divided by ``d``
        and reduced to ``(-pi/d, pi/d]`` picks the sheet that does not jump.
        """
        q = _cdiv(zto, zfrom)
        mag = mp.sqrt(q[0] * q[0] + q[1] * q[1]) ** (mp.mpf(1) / d)
        th = mp.atan2(mp.mpf(q[1]), mp.mpf(q[0])) / d
        while th > mp.pi / d:
            th -= 2 * mp.pi / d
        while th <= -mp.pi / d:
            th += 2 * mp.pi / d
        s = (mag * mp.cos(th), mag * mp.sin(th))
        return (w[0] * s[0] - w[1] * s[1], w[0] * s[1] + w[1] * s[0])

    # adjacency of the lifted 1-skeleton, and the base edge under it
    adj = {}
    for (e, s2), (a, b) in lift["edges"].items():
        adj.setdefault(a, []).append((b, e))
        adj.setdefault(b, []).append((a, e))

    proj = lift["proj"]
    pos = {}
    # the two branch vertices are placed in closed form: the south pole maps to
    # 0 and the north pole to infinity (``None``)
    for u in proj:
        if proj[u] == south:
            pos[u] = (mp.mpf(0), mp.mpf(0))
        elif proj[u] == north:
            pos[u] = None
    # every other lifted vertex is reached from a non-branch seed, and the
    # continuation never has to start at a branch point
    unramified = [b for b in range(len(base_pts)) if b not in (north, south)]
    seed = [u for u in proj if proj[u] == unramified[0]][0]
    pos[seed] = root(finite[unramified[0]], 0)
    queue = [seed]
    while queue:
        u = queue.pop(0)
        for (w, e) in adj[u]:
            if w in pos:
                continue
            if proj[w] in (north, south):
                continue
            p, q = (e[0], e[1]) if proj[u] == e[0] else (e[1], e[0])
            pos[w] = continue_root(pos[u], finite[p], finite[q])
            queue.append(w)
    if len(pos) != len(proj):
        raise RuntimeError("lifted complex did not reach every vertex: "
                           "%d of %d" % (len(pos), len(proj)))

    def to_sphere(w):
        if w is None:
            return (mp.mpf(0), mp.mpf(0), mp.mpf(1))
        return stereo_inv(w[0], w[1])

    lift_points = [to_sphere(pos[u]) for u in range(len(proj))]
    base_points = [(mp.mpf(0), mp.mpf(0), mp.mpf(1)) if finite[b] is None
                   else stereo_inv(mp.mpf(finite[b][0]), mp.mpf(finite[b][1]))
                   for b in range(len(base_pts))]
    lift_faces = [tuple(tr[0]) for tr in lift["tris"]]
    base_sq = chordal_sq(base_points)
    base_of = {e: canon(proj[e[0]], proj[e[1]]) for e in lift["edges"].values()}
    lift_sq = {canon(a, b): base_sq[canon(proj[a], proj[b])]
               for a, b in lift["edges"].values()}
    return {"base_points": base_points, "base_faces": base_faces,
            "base_sq": base_sq, "lift_points": lift_points,
            "lift_faces": lift_faces, "proj": proj, "ld": lift["ld"],
            "lift_sq": lift_sq, "d": d, "base": base, "lift": lift,
            "south": south, "north": north, "edge_perms": edge_perms}


def _to_mpf(x):
    """Exact rational to ``mpf`` at the working precision."""
    return mp.mpf(x.numerator) / mp.mpf(x.denominator)


def _cdiv(p, q):
    n = q[0] * q[0] + q[1] * q[1]
    return ((p[0] * q[0] + p[1] * q[1]) / n, (p[1] * q[0] - p[0] * q[1]) / n)
