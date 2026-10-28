"""Sphere meshes, and pulling them back through a rational map.

The discrete pullback step is a purely geometric operation, and this module is
the part of it that needs no convention: a triangulation of S^2 whose vertex
set contains the postcritical points, the *congruent* lift of it through a
rational map, and the labelled cross-ratio read off the lifted mesh.  What the
step still lacks is sigma's definition (`moduli.Configuration` is the state,
`notes/thurston-pullback.md` the diagnosis); everything here is checkable
without it, which is why it is built first.

Two facts drive the design.

* A lift is *congruent*: f is a local isometry from the pulled-back metric to
  the base metric, so every lifted triangle has its base triangle's side
  lengths.  That is what makes the lifted metric a piecewise Euclidean metric
  on (S^2, f^{-1}(V)) with no Hurwitz solve anywhere.
* The cone angle at a lift is multiplied by the local degree: if q lifts p
  with local degree e, the angle at q in the pulled-back metric is e times the
  base angle at p.  For e >= 2 and a base angle near 2*pi that is a
  *hyper-ideal* vertex (angle >= 2*pi), which no inscribed polyhedron has
  (every vertex of one has angle < 2*pi).  This is a real obstruction to
  `rounding.round_realise` at the critical preimages, not a numerical one, and
  it is measured in `scripts/refinement_study.py` rather than asserted.

The lift is computed by *path lifting*, not by matching nearest points: for
each base face a sample point in its interior is taken, then for each of its
preimages the three great-circle arcs from the sample to the face's vertices
are lifted by Newton continuation and the endpoints snapped to the analytically
computed preimage sets.  Base faces are away from the critical values (those
are vertices), so the lift of such a path is unique and well separated, and
the snap distance is a check: a wrong lift shows up as a large snap.

Every construction here has a structural check that can fail -- Euler
characteristic, each edge in exactly two faces with opposite orientations --
and they are run by `Mesh.validate`.
"""

from __future__ import annotations

from collections import defaultdict

import itertools

import mpmath as mp
import numpy as np

from .moduli import apply_mobius, cross_ratio, is_inf
from .rational import mobius_conjugate


# --------------------------------------------------------------- the chart
def to_vector(z):
    """The unit vector of a point of the Riemann sphere (stereographic from
    the north pole, so infinity is the north pole).

    The chart vectors are the *geometry*, and double precision is right for
    them; the positions themselves are kept at working precision, because a
    double-precision rounding of a critical value splits the double root of
    f(w) = t into a pair of simple roots about 1e-8 apart (measured), which the
    lift then cannot snap to.
    """
    if is_inf(z):
        return np.array([0.0, 0.0, 1.0])
    z = complex(z)
    r2 = z.real * z.real + z.imag * z.imag
    return np.array([2 * z.real / (1 + r2), 2 * z.imag / (1 + r2),
                     (r2 - 1) / (1 + r2)])


def to_point(v):
    """The inverse of `to_vector`; the north pole is infinity."""
    v = np.asarray(v, dtype=float)
    v = v / np.linalg.norm(v)
    den = 1.0 - v[2]
    if abs(den) < 1e-15:
        return complex(np.inf)
    return complex(v[0] / den, v[1] / den)


def labelled_cross_ratio(vectors):
    """The cross-ratio of four points of S^2, in the order given.

    Computed intrinsically so that no gauge of the solver leaks in: rotate the
    sphere so the first point is the north pole, then read the cross-ratio
    with infinity in the first slot.  A Moebius map changes the cross-ratio by
    one of the six substitutions, so this agrees with
    `moduli.Configuration.coordinate()` on the same four points read in the
    same order (a test pins that).
    """
    vs = [np.asarray(v, dtype=float) for v in vectors]
    p0 = vs[0] / np.linalg.norm(vs[0])
    # Householder reflection sending p0 to the north pole
    north = np.array([0.0, 0.0, 1.0])
    d = p0 - north
    if np.linalg.norm(d) < 1e-15:
        R = np.eye(3)
    elif np.linalg.norm(p0 + north) < 1e-15:
        R = np.diag([1.0, -1.0, -1.0])
    else:
        u = d / np.linalg.norm(d)
        R = np.eye(3) - 2.0 * np.outer(u, u)
    pts = [to_point(R @ v) for v in vs]
    if not (abs(pts[0]) == np.inf):
        raise ArithmeticError("the rotation did not put the first point at infinity")
    return cross_ratio(pts[0], pts[1], pts[2], pts[3])


# ------------------------------------------------------------------- meshes
def edge_key(i, j):
    return (i, j) if i < j else (j, i)


class DegenerateLift(ArithmeticError):
    """The lift of this mesh is not a triangulation.

    Two ways it happens, both from a coarse base mesh rather than from bad data:
    a base edge whose two endpoints are critical values has a preimage that is
    a bigon -- two arcs with the same pair of endpoints -- and a face with an
    antipodal pair is a hemisphere.  Adding a vertex (a face-centre refinement)
    removes both, which is what `discrete.pullback_step` does on seeing this.
    """


class Mesh:
    """A triangulation of S^2 with positions.

    points[i] is the point of the Riemann sphere at vertex i (complex, or
    complex(inf) for infinity).  faces is a list of triples of vertex indices.
    `source[i]` optionally records where a vertex came from -- for a lifted
    mesh, (base vertex index, index into the analytic preimage list) -- which
    is what identifies the diagonal copy of P in f^{-1}(P) without searching
    by position.
    """

    def __init__(self, points, faces, source=None, name=""):
        self.points = list(points)
        self.faces = [tuple(f) for f in faces]
        self.source = list(source) if source is not None else None
        self.name = name

    def __len__(self):
        return len(self.points)

    def vectors(self):
        return np.array([to_vector(p) for p in self.points])

    def edges(self):
        out = sorted({edge_key(f[k], f[(k + 1) % 3])
                      for f in self.faces for k in range(3)})
        return out

    def rotation(self):
        """The rotation system of this mesh: the cyclic order of the neighbours
        at every vertex, read off the faces.  Cached.  This is the topological
        order, and it is a *ribbon graph* datum -- the mesh together with it is a
        map, and `dual` gives the dual map."""
        if not hasattr(self, "_rot"):
            self._rot = rotation_of_faces(self.points, self.faces)
        return self._rot

    def as_ribbon_map(self):
        """This mesh as a `RibbonMap`: the rotation system is the datum, the
        faces follow from it."""
        return RibbonMap(self.points, self.rotation(), name=self.name)

    def dual(self):
        """The dual map: vertices at the face centroids, rotation reversed from
        the face boundary.  A mesh and its dual are both ribbon graphs, and
        refining one refines the other."""
        return self.as_ribbon_map().dual()

    def chordal_lengths(self):
        """{(i,j): |x_i - x_j|} for the edge set -- the base metric."""
        v = self.vectors()
        return {e: float(np.linalg.norm(v[e[0]] - v[e[1]])) for e in self.edges()}

    def validate(self, tolerance=0.0):
        """Structural checks: a closed orientable triangulated sphere.

        Returns a dict of counts.  Raises ValueError on a defect.  Every
        condition here fails for some corrupted input (the tests corrupt one
        deliberately), so a pass is evidence.
        """
        if len(self.faces) < 4:
            raise ValueError("fewer than four faces cannot triangulate S^2")
        V = len(self.points)
        directed = defaultdict(int)
        for f in self.faces:
            if len(set(f)) != 3:
                raise ValueError(f"degenerate face {f}: repeated vertex")
            for k in range(3):
                directed[(f[k], f[(k + 1) % 3])] += 1
        for (i, j), n in directed.items():
            if n != 1:
                raise ValueError(f"directed edge {(i, j)} used {n} times")
            if (j, i) not in directed:
                raise ValueError(f"edge {(i, j)} has no opposite orientation: "
                                 "the surface is not closed")
        E = len(directed) // 2
        F = len(self.faces)
        if V - E + F != 2:
            raise ValueError(f"Euler characteristic {V - E + F} != 2")
        # vertex links: a single cycle per vertex, else the surface is not
        # a manifold at that vertex
        around = defaultdict(list)
        for f in self.faces:
            for k in range(3):
                around[f[k]].append((f[(k + 1) % 3], f[(k + 2) % 3]))
        for v, link in around.items():
            succ, pred = {}, {}
            for (b, c) in link:
                if b in succ:
                    raise ValueError(f"vertex {v} link repeats {b}: not a fan")
                succ[b], pred[c] = c, b
            if set(succ) != set(pred):
                raise ValueError(f"vertex {v} link is not closed")
            start = next(iter(succ))
            cur, count = start, 0
            while True:
                cur = succ[cur]
                count += 1
                if cur == start:
                    break
                if count > len(link):
                    raise ValueError(f"vertex {v} link does not close")
            if count != len(link):
                raise ValueError(f"vertex {v} link is {count} of {len(link)}"
                                 " -- the link is not a single cycle")
        return {"V": V, "E": E, "F": F}


class RibbonMap:
    """A map (cell decomposition) of the sphere: a rotation system plus positions.

    `rot[v]` is the cyclic list of the neighbours of v -- the *topological order*
    at v, which is the datum the lift needs and the one thing the round geometry
    cannot supply (measured: two neighbours 1.18 degrees apart, with the tangent
    order disagreeing with the combinatorial one).  A map is a ribbon graph, and
    so is its dual, which `dual` returns; `subdivide_faces` is a ribbon-graph
    operation, so both survive refinement.

    The level-0 map of the pullback step is the dual of the spine: the spine's
    faces are the punctures, so the dual's vertices are the punctures, and the
    rotation at a puncture is the spine's face-boundary order, reversed.  That is
    where the topological order lives before any mesh refinement -- it is data,
    not geometry.
    """

    def __init__(self, points, rot, name=""):
        self.points = list(points)
        self.rot = {v: list(ns) for v, ns in rot.items()}
        self.name = name
        seen = set()
        for v, ns in self.rot.items():
            if not ns:
                raise ValueError(f"vertex {v} has no neighbours")
            if len(set(ns)) != len(ns):
                raise ValueError(f"vertex {v} repeats a neighbour")
            for u in ns:
                if u not in self.rot:
                    raise ValueError(f"vertex {v} lists unknown neighbour {u}")
                seen.add((v, u))
        for v, ns in self.rot.items():
            for u in ns:
                if (u, v) not in seen:
                    raise ValueError(f"the rotation is not symmetric at {(v, u)}")

    def __len__(self):
        return len(self.points)

    def edges(self):
        return sorted({edge_key(u, v) for v, ns in self.rot.items() for u in ns})

    def rotation_at(self, v):
        return list(self.rot[v])

    def faces(self):
        """The faces as polygons (lists of vertices): the cycles of sigma . alpha
        on directed edges, where sigma turns a directed edge to the next one at
        its source and alpha reverses it.  The polygon is the sequence of the
        sources of the walk's directed edges."""
        def sigma(u, v):
            ns = self.rot[u]
            return (u, ns[(ns.index(v) + 1) % len(ns)])

        visited = set()
        out = []
        limit = 2 * len(self.edges()) + 2
        for v, ns in self.rot.items():
            for u in ns:
                if (v, u) in visited:
                    continue
                poly = [v, u]
                cur = (v, u)
                while True:
                    visited.add(cur)
                    # the next dart of the face: sigma of the reversed dart
                    nxt = sigma(*reversed(cur))
                    if nxt == (v, u):
                        break
                    poly.append(nxt[1])
                    cur = nxt
                    if len(poly) > limit:
                        raise ValueError("the rotation system does not close")
                out.append(poly[:-1])
        return out

    def dual(self):
        """The dual map: vertices are this map's faces, at their centroids, and
        the rotation at a dual vertex is the face's boundary order reversed."""
        polys = self.faces()
        left = {}
        for i, poly in enumerate(polys):
            n = len(poly)
            for k in range(n):
                left[(poly[k], poly[(k + 1) % n])] = i
        vecs = np.array([to_vector(p) for p in self.points])
        points = []
        for poly in polys:
            c = vecs[poly].sum(axis=0)
            points.append(to_point(c / np.linalg.norm(c)))
        rot = {}
        for i, poly in enumerate(polys):
            n = len(poly)
            nbrs = [left[(poly[(k + 1) % n], poly[k])] for k in range(n)]
            rot[i] = list(reversed(nbrs))
        return RibbonMap(points, rot, name=f"dual({self.name})")

    def add_diagonal(self, face, i, j):
        """Split a face by a diagonal between two of its boundary vertices.

        A ribbon-graph operation: the rotation system is the whole datum, so the
        diagonal is recorded by inserting each endpoint into the other's rotation
        in the slot the face occupied, and the faces are whatever the rotation
        system then gives.  The diagonal must join two distinct vertices that are
        not already adjacent and are not adjacent on the face's boundary.
        """
        poly = list(face)
        n = len(poly)
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError("diagonal endpoints out of range")
        a, b = poly[i], poly[j]
        if a == b:
            raise ValueError("a diagonal must join two distinct vertices")
        if (j - i) % n in (1, n - 1):
            raise ValueError("a diagonal must not be a boundary edge")
        if edge_key(a, b) in self.edges():
            raise ValueError(f"{(a, b)} is already an edge")
        for v, k, other in ((a, i, b), (b, j, a)):
            ns = self.rot[v]
            prev = poly[(k - 1) % n]
            nxt = poly[(k + 1) % n]
            if prev not in ns or nxt not in ns:
                raise ValueError("the face is not a face of the rotation system")
            if ns.index(nxt) != (ns.index(prev) + 1) % len(ns):
                raise ValueError(f"the face is not a slot at vertex {v}")
            ns.insert(ns.index(nxt), other)
        return RibbonMap(self.points, self.rot, name=f"{self.name}+d")

    def triangulate(self):
        """Add diagonals until every face is a triangle.

        The diagonals are chosen by a fan from each face, taking the first pair
        that is not already an edge.  The choice is canonical but arbitrary; the
        point is that it is a *ribbon* operation, so the rotation system -- the
        topological order -- is carried through, and at every vertex that already
        had a rotation it is only extended.
        """
        m = RibbonMap(self.points, self.rot, name=self.name)
        for _ in range(4 * len(self.edges()) + 8):
            polys = m.faces()
            target = next((p for p in polys if len(p) > 3), None)
            if target is None:
                return m
            n = len(target)
            done = False
            for i in range(n):
                for j in range(i + 2, n):
                    if (j - i) % n in (1, n - 1):
                        continue
                    try:
                        m = m.add_diagonal(target, i, j)
                        done = True
                        break
                    except ValueError:
                        continue
                if done:
                    break
            if not done:
                raise ValueError(f"cannot triangulate the face {target}")
        raise ValueError("triangulation did not terminate")

    def subdivide_faces(self):
        """Insert a vertex in each face, joined to the face's corners.

        A ribbon-graph operation: the new vertex's rotation is the face's corner
        order, and at each old corner the new edge takes the slot the face
        occupied, between that face's two edges.  So the rotation at the old
        vertices is *preserved* -- extended, not recomputed -- and the dual is
        refined by the dual operation.  For a k-gon face this also triangulates it
        into k triangles, which is how a polygonal map becomes a mesh.
        """
        vecs = np.array([to_vector(p) for p in self.points])
        points = list(self.points)
        rot = {v: list(ns) for v, ns in self.rot.items()}
        for poly in self.faces():
            n = len(poly)
            if n < 3:
                raise ValueError(f"a face of length {n} cannot be subdivided")
            if len(set(poly)) != len(poly):
                raise ValueError(
                    f"face {poly} visits a vertex twice: `subdivide_faces`"
                    " needs a simple map (no face with a repeated corner)")
            centre = face_sample(vecs, tuple(poly))
            points.append(to_point(centre))
            c = len(points) - 1
            # the centre's flower is the face's corner order REVERSED: the walk
            # sigma.alpha traverses a face against the rotation's direction at
            # each vertex, so the centre sees its corners the other way round
            # (the same reversal as the dual's rotation -- traced on the
            # tetrahedron, where the forward order makes the walk cross sectors
            # and come out nine long)
            rot[c] = list(reversed(poly))
            for k in range(n):
                a = poly[k]
                prev = poly[(k - 1) % n]
                ns = rot[a]
                # the walk enters a from prev and leaves to the next corner, and
                # the face's corner at a is the slot sigma gives from the
                # incoming edge: insert after prev, so sigma(a -> prev) is the
                # new edge to the centre and the walk becomes
                # ... -> prev -> a -> centre -> next -> ...
                # (the slot semantics were checked by tracing the tetrahedron's
                # rotation through the operation; the other slot puts the centre
                # in the *neighbouring* face's sector and the face walks come out
                # nine long, which `validate` and the face lengths catch)
                ns.insert((ns.index(prev) + 1) % len(ns), c)
        return RibbonMap(points, rot, name=f"{self.name}+c")


def rotation_of_faces(points, faces):
    """The rotation system of a face list: the cyclic order of the neighbours at
    every vertex.  Two neighbours are consecutive in the order exactly when some
    face has them as the two neighbours of their common vertex, so the faces give
    the *pairing* (after prev comes next), and the order is the chain of pairs.
    """
    after = {}                                  # (vertex, neighbour) -> next
    for face in faces:
        n = len(face)
        for k in range(n):
            v = face[k]
            prev, nxt = face[(k - 1) % n], face[(k + 1) % n]
            if (v, prev) in after:
                raise ValueError(f"vertex {v}: the face list is not a map")
            after[(v, prev)] = nxt
    vertices = sorted({v for (v, _u) in after})
    rot = {}
    for v in vertices:
        ns = sorted({u for (vv, u) in after if vv == v})
        start = ns[0]
        order, cur = [start], start
        while True:
            if (v, cur) not in after:
                raise ValueError(f"vertex {v}: the rotation does not close")
            cur = after[(v, cur)]
            if cur == start:
                break
            order.append(cur)
            if len(order) > len(ns) + 1:
                raise ValueError(f"vertex {v}: the rotation does not close")
        if len(order) != len(ns):
            raise ValueError(f"vertex {v}: the rotation is not a single cycle")
        rot[v] = order
    return rot


def ribbon_order_of_spine(spine):
    """The topological order at every puncture, out of the ribbon structure.

    The dual of the spine has the punctures as its vertices (the spine's faces),
    and its rotation at a puncture is the spine's face-boundary order, reversed.
    For the rabbit's spine -- a bouquet of three loops at one vertex -- the dual
    is a tree on the four punctures, so this is the cyclic order of the tree's
    edges at each puncture.  That is the order before any mesh refinement, and it
    is data.
    """
    d = spine.dual()
    return {v: [d.tgt(dart) for dart in d.flowers[v]] for v in d.vertices()}


def marking_consistent_with_ribbon(mesh, spine, marking):
    """Does the mesh's topological order agree with the ribbon structure?

    `marking` maps a spine face (a puncture) to a mesh vertex.  The check: at
    every puncture, the tree's edges -- the spine-dual's edges -- appear in the
    mesh's rotation in the ribbon order, as a cyclic subsequence (the mesh has
    more edges than the tree, so the tree's order must be *contained* in the
    mesh's, not equal to it).  This is the check that fails when a mesh has been
    built without reference to the ribbon structure.
    """
    tree = ribbon_order_of_spine(spine)
    rot = mesh.rotation()
    for face_index, vertex in marking.items():
        want = tree[face_index]
        if len(want) < 2:
            continue                      # a leaf: no order to check
        got = rot[vertex]
        # the ribbon order must appear among `got` as a cyclic sequence *of the
        # old neighbours* -- a refined mesh interleaves new neighbours, so the
        # order is read by restricting to the marked vertex's tree neighbours
        # and checking the cyclic order there
        # the refined rotation is cyclic, so the restricted order may start
        # anywhere: rotate it to start at want[0] before comparing
        present = [u for u in got if u in set(want)]
        if want[0] in present:
            i = present.index(want[0])
            if present[i:] + present[:i] == list(want):
                continue
        n, k = len(got), len(want)
        if not any(all(got[(i + r) % n] == want[r] for r in range(k))
                   for i in range(n)):
            return False
    return True


def triangulations_containing(base_edges, points):
    """Every triangulation of the point set whose edge set contains `base_edges`.

    For 4 points in convex position there are exactly two triangulations, by the
    two diagonals of the quadrilateral, and a triple of vertices spans a face of
    one of them iff its three edges are present (the fourth point is then on the
    other side).  That is all this needs to enumerate them, and the enumeration
    is exhaustive rather than a default: the point is to *choose* the
    triangulation by a criterion that can be stated -- here, consistency with the
    ribbon structure -- not to accept the hull's.

    Raises ValueError for more than four points, where the combinatorics is
    bigger than this needs: the pullback step at |P| = 4 is the case in hand, and
    the general enumeration is deferred until a map with more postcritical points
    exists (the Devaney family, plan item 330).
    """
    n = len(points)
    if n != 4:
        raise ValueError(
            f"triangulations_containing is implemented for 4 points, not {n}:"
            " the general enumeration waits for a map with more postcritical"
            " points")
    all_pairs = {edge_key(i, j) for i in range(n) for j in range(i + 1, n)}
    if not (set(base_edges) <= all_pairs):
        raise ValueError("the base edges are not edges of this point set")
    seen, results = set(), []
    # With 4 points there is one triangulation up to mirror class -- the
    # tetrahedron's four faces -- and the two classes differ in the rotation
    # system, which is the datum the ribbon-order check reads.  So both classes
    # are built and `consistent_markings` chooses between them; on the rabbit
    # the hull's class admits no marking and the mirrored one admits 6.
    for tri in ([(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)],
                [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]):
        edges = ({edge_key(0, 1), edge_key(1, 2), edge_key(2, 3), edge_key(0, 3),
                  edge_key(0, 2), edge_key(1, 3)})
        if not (set(base_edges) <= edges):
            continue
        m = Mesh(points, orient_faces(points, tri))
        try:
            m.validate()
        except ValueError:
            continue
        if not (set(base_edges) <= set(m.edges())):
            continue
        # NB: the two mirror classes have the *same* face set and the same edge
        # set -- they differ only in the rotation system, which is the datum the
        # ribbon-order check reads.  So the dedup must be by rotation, not by
        # faces: dedup by faces would keep the first class and silently drop the
        # second, which is exactly what happened (measured: 1 candidate with 0
        # markings, where there are 2 candidates and one has 6).
        key = tuple(tuple(m.rotation()[v]) for v in sorted(m.rotation()))
        if key not in seen:
            seen.add(key)
            results.append(m)
    return results


def level0_map(points, spine, deg=None):
    """The level-0 mesh of the pullback step, built from the spine's dual.

    The spine's faces are the punctures, so the spine-dual's vertices are the
    punctures and its edges join the pairs of punctures the spine's edges
    separate -- a tree when the spine has one vertex (the rabbit: three loops).
    The level-0 map is a triangulation of `points` (the punctures at their
    analytic positions, in the spine's face order) that *contains those tree
    edges*, with the mirror class chosen so that the mesh's topological order
    agrees with the ribbon order under some marking of the punctures.

    Returns (mesh, marking) where `marking` maps a spine face (puncture) to a
    mesh vertex.  Raises ValueError when no triangulation of the point set
    carries the ribbon order -- the check that this construction exists to pass.
    """
    import itertools
    tree = ribbon_order_of_spine(spine)
    tree_edges = {edge_key(v, u) for v, ns in tree.items() for u in ns}
    if len(points) != len(tree):
        raise ValueError(
            f"the point set has {len(points)} points but the spine has"
            f" {len(tree)} faces: the marking between them must be given")
    if deg is not None:
        # the local degrees say which punctures are critical values; a base edge
        # joining two of them lifts to a bigon, so the tree alone is not enough
        # and the mesh needs extra vertices -- not implemented here (see
        # `base_mesh`)
        pass
    candidates = triangulations_containing(tree_edges, points)
    if not candidates:
        raise ValueError(
            "no triangulation of the point set contains the spine-dual's"
            " edges: the punctures are not in the position the tree needs")
    for mesh in candidates:
        markings = consistent_markings(mesh, spine)
        if markings:
            # the marking is a choice among the consistent ones; take the first,
            # and say in the mesh how many there were
            mesh.consistent_markings = markings
            return mesh, markings[0]
    raise ValueError(
        "every triangulation containing the spine-dual's edges has a"
        " topological order inconsistent with the ribbon structure: the"
        " punctures' positions or the spine's presentation is wrong")


def consistent_markings(mesh, spine):
    """Every marking of the spine's faces by the mesh's vertices under which the
    mesh's topological order agrees with the ribbon structure.

    Returns a list of dicts (spine face -> mesh vertex).  For the rabbit's spine
    and the tetrahedron on its postcritical set this is 6 of the 24 possible
    markings, and every one of them puts the spine's one degree-3 face on the same
    mesh vertex -- so the check constrains the marking, which is otherwise the
    author's to fix.  The empty list is the answer when the mesh was built without
    reference to the ribbon structure.
    """
    import itertools
    tree = ribbon_order_of_spine(spine)
    faces = sorted(tree)
    vertices = sorted(mesh.rotation())
    out = []
    # an injective assignment: the mesh may carry vertices beyond the punctures
    for perm in itertools.permutations(vertices, len(faces)):
        marking = dict(zip(faces, perm))
        if marking_consistent_with_ribbon(mesh, spine, marking):
            out.append(marking)
    return out


def ribbon_map_from_spine(spine, marking, points):
    """The level-0 map of the pullback step: the dual of the spine.

    `spine` is the engine's RibbonGraph (the spine of S^2 - P); its faces are the
    punctures, so the dual's vertices are the punctures and the rotation at a
    puncture is the spine's face-boundary order, reversed -- the topological order,
    read out of the ribbon structure rather than out of any geometry.  `marking`
    says which spine face is which point of P (`points`, in the analytic order);
    that correspondence is the marking, and it is the author's to fix.
    """
    d = spine.dual()
    rot = {}
    for v in d.vertices():
        rot[v] = [d.tgt(dart) for dart in d.flowers[v]]
    pts = [None] * len(rot)
    for face_index, point_index in marking.items():
        pts[face_index] = points[point_index]
    if any(p is None for p in pts):
        raise ValueError("the marking does not cover every spine face")
    return RibbonMap(pts, rot, name="dual(spine)")


def sphere_triangulation(points):
    """A triangulation of S^2 with the given points as its vertex set: the
    faces of the convex hull of the corresponding unit vectors.

    Brute force, O(n^4), which is the right trade for meshes of this size and
    removes the dependency on a hull routine.  Points on the sphere are in
    convex position, so every triple whose plane has all the other points
    strictly on one side spans a face of the hull, i.e. a triangle of the
    Delaunay triangulation.
    """
    n = len(points)
    if n < 4:
        raise ValueError("need at least four vertices")
    v = np.array([to_vector(p) for p in points])
    faces = []
    eps = 1e-9
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                normal = np.cross(v[j] - v[i], v[k] - v[i])
                norm = np.linalg.norm(normal)
                if norm < eps:
                    continue                       # collinear: not a face
                normal = normal / norm
                side = (v - v[i]) @ normal
                keep = np.ones(n, dtype=bool)
                keep[[i, j, k]] = False
                s = side[keep]
                if np.all(s > eps):
                    faces.append((i, j, k))
                elif np.all(s < -eps):
                    faces.append((i, k, j))
    if not faces:
        raise ValueError("no faces found; are the points in convex position?")
    return orient_faces(points, faces)


def orient_faces(points, faces):
    """Orient a face list consistently, by propagating round the surface.

    Not by the outward-normal rule: that needs the origin strictly inside the
    mesh, which fails for points lying in a hemisphere -- and the test that
    caught it was a tetrahedron of four points on the sphere where the origin is
    outside.  Instead, two-colour the faces of the adjacency graph: a face
    sharing an edge must traverse it opposite to its neighbour.
    """
    faces = [tuple(f) for f in faces]
    edge_to_faces = defaultdict(list)
    for fi, f in enumerate(faces):
        for k in range(3):
            edge_to_faces[edge_key(f[k], f[(k + 1) % 3])].append(fi)
    for e, fl in edge_to_faces.items():
        if len(fl) > 2:
            raise ValueError(f"edge {e} is in {len(fl)} faces: not a surface")
    def oriented(f, flip):
        return f if not flip else (f[0], f[2], f[1])
    flip = [None] * len(faces)
    for seed in range(len(faces)):
        if flip[seed] is not None:
            continue
        flip[seed] = False
        queue = [seed]
        while queue:
            fi = queue.pop()
            f = oriented(faces[fi], flip[fi])
            for k in range(3):
                i, j = f[k], f[(k + 1) % 3]
                for fj in edge_to_faces[edge_key(i, j)]:
                    if fj == fi or flip[fj] is not None:
                        continue
                    g = faces[fj]
                    directed = [(g[t], g[(t + 1) % 3]) for t in range(3)]
                    flip[fj] = (i, j) in directed
                    queue.append(fj)
    out = [oriented(faces[fi], bool(flip[fi])) for fi in range(len(faces))]
    if any(x is None for x in flip):
        raise ValueError("the face graph is not connected")
    return out


def antipodal_pairs(points, tol=1e-6):
    """Pairs of vertices that are antipodal on the sphere."""
    v = np.array([to_vector(p) for p in points])
    out = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if np.linalg.norm(v[i] + v[j]) < tol:
                out.append((i, j))
    return out


def bipyramid_mesh(points, poles):
    """A mesh separating two antipodal vertices: a bipyramid whose apexes are
    the pair and whose belt carries every other vertex.

    Needed because two points of P are antipodal *in every chart* -- 0 and
    infinity for any polynomial whose critical point is periodic, and the
    tetrahedron on four points puts every pair in a face.  A face with an
    antipodal pair is a hemisphere: it has no well-defined interior point, so
    nothing downstream (the lift's sample, the refinement's midpoint) can use it.
    With the pair at the poles and the rest on a belt, no face contains both.
    """
    a, b = poles
    if len(antipodal_pairs(points)) != 1:
        raise ValueError("bipyramid_mesh handles exactly one antipodal pair")
    belt = [i for i in range(len(points)) if i not in (a, b)]
    v = np.array([to_vector(p) for p in points])
    axis = v[a]
    # order the belt cyclically about the axis
    ref = np.cross(axis, v[belt[0]] - (v[belt[0]] @ axis) * axis)
    ref = ref / np.linalg.norm(ref)
    angs = []
    for i in belt:
        w = v[i] - (v[i] @ axis) * axis
        angs.append((float(np.arctan2(ref @ np.cross(axis, w), ref @ w)), i))
    belt = [i for _t, i in sorted(angs)]
    k = len(belt)
    if k < 3:
        raise ValueError("a bipyramid needs at least three belt vertices")
    faces = []
    for t in range(k):
        i, j = belt[t], belt[(t + 1) % k]
        faces.append((a, i, j))
        faces.append((b, i, j))
    return orient_faces(points, faces)


def two_cap_mesh(points, apex_a, apex_b, belt_a, belt_b):
    """A mesh with two caps joined by a band: apex_a over belt_a, apex_b over
    belt_b, and the two belts stitched together.

    This is the smallest construction that satisfies all three conditions the
    lift needs at once, and each condition is forced by the data:
      * no two vertices with a single preimage (the critical values) are joined,
        or the lift of that edge is a bigon;
      * no face contains an antipodal pair, or it is a hemisphere;
      * the two antipodal points are themselves not joined.

    For a degree-2 map with four postcritical points there are exactly two
    critical values, and 0 and infinity are antipodal, so no mesh with five
    vertices -- in particular no tetrahedron, where every pair is an edge --
    can satisfy them.  Eight vertices do: two apexes, two belts of three.
    """
    faces = []
    for t in range(3):
        i, j = belt_a[t], belt_a[(t + 1) % 3]
        faces.append((apex_a, i, j))
        i, j = belt_b[t], belt_b[(t + 1) % 3]
        faces.append((apex_b, j, i))
    for t in range(3):
        x, x2 = belt_a[t], belt_a[(t + 1) % 3]
        y, y2 = belt_b[t], belt_b[(t + 1) % 3]
        faces.append((x, y, x2))
        faces.append((x2, y, y2))
    return orient_faces(points, faces)


def spread_points(points, count, seed=0):
    """`count` points on the sphere clear of the given ones and of each other,
    in a deterministic order."""
    from .moduli import is_inf
    v = np.array([to_vector(p) for p in points])
    out = []
    rng = np.random.RandomState(seed)
    tries = 0
    while len(out) < count:
        tries += 1
        if tries > 20000:
            raise ValueError("could not place enough extra points")
        w = rng.normal(size=3)
        w /= np.linalg.norm(w)
        if np.min(np.linalg.norm(v - w, axis=1)) < 0.45:
            continue
        if any(np.linalg.norm(w + x) < 1e-3 for x in v):
            continue
        if any(np.linalg.norm(w - x) < 1e-3 for x in out):
            continue
        if any(np.linalg.norm(w + x) < 1e-3 for x in out):
            continue
        out.append(w)
    return [to_point(w) for w in out]


def mesh_defects(points, faces):
    """Faces containing an antipodal pair, and edges that are diameters."""
    v = np.array([to_vector(p) for p in points])
    bad_faces, bad_edges = [], []
    for f in faces:
        for x in range(3):
            for y in range(x + 1, 3):
                if np.linalg.norm(v[f[x]] + v[f[y]]) < 1e-6:
                    bad_faces.append(f)
    E = {edge_key(f[k], f[(k + 1) % 3]) for f in faces for k in range(3)}
    for (i, j) in E:
        if np.linalg.norm(v[i] + v[j]) < 1e-6:
            bad_edges.append((i, j))
    return bad_faces, bad_edges


def face_sample(vectors, face, weights=(0.55, 0.30, 0.15)):
    """A point inside a face, robust to an antipodal pair of vertices.

    The obvious choice -- the normalised sum of the vertex vectors -- is
    degenerate exactly when the face contains an antipodal pair: the pair
    cancels and the direction is the third vertex's.  Two points of P are
    antipodal in every rotation chart, so this happens for real data: the rabbit
    has 0 and infinity in P, so two faces of its tetrahedron are hemispheres and
    the symmetric centroid lands on a vertex.  Positive *distinct* weights fix
    it: the antipodal pair no longer cancels, and the point is still an interior
    point of the flat polygon.
    """
    vs = [vectors[i] for i in face]
    if len(vs) < 3:
        raise ValueError("a face needs at least three corners")
    base = [weights[k % len(weights)] for k in range(len(vs))]
    for scale in (1.0, 1.3, 1.7, 2.5, 4.0):
        w = [b * (scale ** k) for k, b in enumerate(base)]
        p = sum(wi * v for wi, v in zip(w, vs))
        n = np.linalg.norm(p)
        if n < 1e-9:
            continue
        p = p / n
        if min(np.linalg.norm(p - v) for v in vs) > 1e-4:
            return p
    raise ArithmeticError(f"no usable interior point for face {face}")


def subdivide_face_centres(mesh):
    """The 1 -> 3 refinement: one new vertex inside each face, joined to the
    face's three vertices.

    This is the refinement the study uses, because `subdivide` (edge midpoints)
    cannot refine a mesh that has a diameter -- the midpoint of an antipodal pair
    is the origin, which is not a point of the sphere -- and the tetrahedron on
    the rabbit's postcritical set has one.  Inserting face centres involves no
    midpoints, and the interior point is chosen by `face_sample`, which is
    robust to antipodal vertices.
    """
    vectors = mesh.vectors()
    points = list(mesh.points)
    faces = []
    for f in mesh.faces:
        centre = face_sample(vectors, f)
        points.append(to_point(centre))
        k = len(points) - 1
        faces.extend([(f[0], f[1], k), (f[1], f[2], k), (f[2], f[0], k)])
    out = Mesh(points, faces, name=f"{mesh.name}+c")
    out.validate()
    return out


def subdivide(mesh, project=True):
    """The 1 -> 4 refinement: insert an edge midpoint in every edge.

    With `project`, the new vertices are put on the sphere (the normalised
    midpoint of the two unit vectors), which is the refinement of the *mesh*:
    the old vertices keep their positions, so the cone angle at each of them is
    unchanged, while the polyhedron's geometry converges to the sphere's.  That
    is exactly the knob the refinement study needs, and it is worth being
    explicit that it does not touch the angles at the postcritical points.
    """
    mids = {}
    points = list(mesh.points)
    for e in mesh.edges():
        a, b = to_vector(mesh.points[e[0]]), to_vector(mesh.points[e[1]])
        mid = (a + b) / 2.0
        norm = np.linalg.norm(mid)
        if norm < 1e-12:
            raise ValueError(f"edge {e} is a diameter: no midpoint on the sphere")
        mid = mid / norm if project else mid
        mids[e] = len(points)
        points.append(to_point(mid))
    faces = []
    for (i, j, k) in mesh.faces:
        ij, jk, ki = mids[edge_key(i, j)], mids[edge_key(j, k)], mids[edge_key(i, k)]
        faces.extend([(i, ij, ki), (ij, j, jk), (ki, jk, k), (ij, jk, ki)])
    out = Mesh(points, faces, name=f"{mesh.name}+1")
    return out


# -------------------------------------------------------------- the lift
def _great_circle(a, b, steps):
    """Points of the shorter great-circle arc from unit vector a to b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    cos = float(np.clip(a @ b, -1.0, 1.0))
    theta = np.arccos(cos)
    if theta < 1e-12:
        return np.array([a])
    s = np.sin(theta)
    return np.array([(np.sin((1 - t) * theta) * a + np.sin(t * theta) * b) / s
                     for t in np.linspace(0.0, 1.0, steps + 1)])


def lift_path(f, start, base_path, tol=1e-24, steps=12):
    """Lift a path in the base sphere through f, starting from `start`.

    Newton continuation with the previous point as the initial guess.  The
    paths used here are arcs in the interior of a base face and therefore
    avoid the critical values (which are vertices), so the lift is unique.
    Returns (end point, largest Newton correction seen).
    """
    w = mp.mpc(start)
    worst = mp.mpf(0)
    for target in base_path:
        if target == mp.inf:
            raise ValueError("a base path through infinity is not supported")
        t = mp.mpc(target)
        for _ in range(steps):
            fw = f(w)
            if fw == mp.inf:
                raise ArithmeticError("the lifted path hit a pole")
            d = f.derivative(w)
            if abs(d) < mp.mpf(10) ** (-mp.mp.dps + 8):
                # a path ending at a critical value ends at a critical
                # preimage, where the derivative is zero: stop here and let
                # the caller's polish finish (plain Newton converges, only
                # linearly)
                break
            delta = (fw - t) / d
            w = w - delta
            worst = max(worst, abs(delta))
            if abs(delta) < tol:
                break
    return w, worst


def _continue_arc(g, w, v0, v1, depth=0, max_depth=7):
    """Lift the great-circle arc from v0 to v1, starting at a preimage of v0.

    Continuation along the arc, bisecting the segment whenever Newton does not
    converge.  A path that passes near a critical value is ill conditioned there
    and a fixed-step continuation either diverges or lands in the wrong basin --
    measured both ways -- so the step size has to adapt to the path rather than
    be fixed in advance.  The lift of an arc is determined by its start and its
    homotopy class, so bisecting changes nothing about the answer.
    """
    def newton(z, target, steps=150):
        """Newton for g(z) = target, in whichever chart keeps the numbers small.

        An arc may pass through a pole, where the lift goes through infinity; in
        the z chart that is a divergence, in the inverted chart u = 1/z it is an
        ordinary point.  So the chart is chosen per step by the size of the
        numbers, and a step that cannot be taken in either chart is a signal to
        bisect.
        """
        t = mp.mpc(target)
        for _ in range(steps):
            if not mp.isfinite(z):
                # the lift is at infinity: continue in the inverted chart
                z = mp.mpc(0)
                continue
            if max(abs(z), abs(t)) > mp.mpf(10) ** 10:
                # inverted chart: Z = 1/z, F(Z) = 1/g(1/Z), T = 1/t
                if z == 0 or t == 0:
                    return None
                Z, T = 1 / z, 1 / t
                fz = 1 / g(1 / Z)
                d = g.derivative(1 / Z) / (Z * Z * g(1 / Z) ** 2)
                if abs(d) < mp.mpf(10) ** (-mp.mp.dps + 8):
                    return None
                delta = (fz - T) / d
                Z = Z - delta
                if abs(delta) < mp.mpf(10) ** (-mp.mp.dps + 10):
                    return 1 / Z
                z = 1 / Z
            else:
                fw = g(z)
                if fw == mp.inf:
                    z = mp.inf
                    continue
                d = g.derivative(z)
                if abs(d) < mp.mpf(10) ** (-mp.mp.dps + 8):
                    return None       # at a critical point: bisect instead
                delta = (fw - t) / d
                z = z - delta
                if abs(delta) < mp.mpf(10) ** (-mp.mp.dps + 10):
                    return z
        return None

    pts = _great_circle(v0, v1, steps=4)
    z = mp.mpc(w)
    for u in pts[1:]:
        nxt = newton(z, _mp_pt(u))
        if nxt is None:
            break
        z = nxt
    else:
        return z
    if depth >= max_depth:
        raise DegenerateLift("the continuation of an arc did not converge")
    mid = np.array(v0, dtype=float) + np.array(v1, dtype=float)
    mid = mid / np.linalg.norm(mid)
    z1 = _continue_arc(g, z, v0, mid, depth + 1, max_depth)
    return _continue_arc(g, z1, mid, v1, depth + 1, max_depth)


def _polish(f, w, target, steps=120, tol=1e-30):
    """Newton at the end of a path, run long.

    A path ending at a critical *value* ends at a critical preimage, where
    f(w) = target has a multiple root and Newton converges only linearly
    (measured: 5e-5 left after 12 steps).  The multiplicity is not trustworthy
    here -- `RationalMap.local_degree` reads the critical preimage as simple,
    because the analytic preimage itself is only located to about 1e-9 by the
    root finder -- so rather than apply a multiplicity-corrected step, run
    plain Newton to convergence: on a double root the error halves per step, and
    120 steps take it well below the double-precision noise floor.
    """
    t = mp.mpc(target)
    for _ in range(steps):
        fw = f(w)
        if fw == mp.inf:
            break
        d = f.derivative(w)
        if abs(d) < mp.mpf(10) ** (-mp.mp.dps + 8):
            break
        delta = (fw - t) / d
        w = w - delta
        if abs(delta) < tol:
            break
    return w


def _mp_vec(z):
    """The unit vector of a point of the sphere, in mpmath arithmetic."""
    if is_inf(z):
        return [mp.mpf(0), mp.mpf(0), mp.mpf(1)]
    z = mp.mpc(z)
    x, y = mp.re(z), mp.im(z)
    r2 = x * x + y * y
    return [2 * x / (1 + r2), 2 * y / (1 + r2), (r2 - 1) / (1 + r2)]


def _mp_pt(v):
    """The inverse of `_mp_vec`; mpmath throughout."""
    n = mp.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    v = [x / n for x in v]
    den = 1 - v[2]
    if abs(den) < mp.mpf(10) ** (-mp.mp.dps + 8):
        return mp.inf
    return mp.mpc(v[0] / den, v[1] / den)


def _mp_mv(R, v):
    return [sum(R[i][j] * v[j] for j in range(3)) for i in range(3)]


def _rotation_to_north(p):
    """Rodrigues' formula for a rotation taking the north pole to p, in
    mpmath (no trigonometric functions, so no precision is lost)."""
    n = [mp.mpf(0), mp.mpf(0), mp.mpf(1)]
    c = p[2]
    cross = [n[1] * p[2] - n[2] * p[1],
             n[2] * p[0] - n[0] * p[2],
             n[0] * p[1] - n[1] * p[0]]
    s = mp.sqrt(sum(x * x for x in cross))
    if s < mp.mpf(10) ** (-mp.mp.dps + 8):
        if c > 0:
            return [[mp.mpf(int(i == j)) for j in range(3)] for i in range(3)]
        return [[mp.mpf(1), 0, 0], [0, mp.mpf(-1), 0], [0, 0, mp.mpf(-1)]]
    axis = [x / s for x in cross]
    K = [[0, -axis[2], axis[1]],
         [axis[2], 0, -axis[0]],
         [-axis[1], axis[0], 0]]
    K2 = [[sum(K[i][k] * K[k][j] for k in range(3)) for j in range(3)]
          for i in range(3)]
    return [[mp.mpf(int(i == j)) + s * K[i][j] + (1 - c) * K2[i][j]
             for j in range(3)] for i in range(3)]


def _mobius_through(pairs):
    """The Möbius map (a,b,c,d) sending three points to three points, at
    working precision.

    The 3x4 system z a + b - w z c - w d = 0 has a one-dimensional null space;
    it is solved by fixing the coordinate with the largest pivot to 1 and
    solving the remaining 3x3 system exactly in mpmath, then normalising by
    ad - bc = 1.  (The float64 SVD version of this was the source of a
    1e-10 error in the conjugated map's derivative -- enough to split a double
    root into two roots 1e-8 apart.  Do not go back to it.)
    """
    A = []
    for z, w in pairs:
        z, w = mp.mpc(z), mp.mpc(w)
        A.append([z, mp.mpc(1), -w * z, -w])
    for fix in range(4):
        cols = [j for j in range(4) if j != fix]
        B = mp.matrix(3, 3)
        rhs = mp.matrix(3, 1)
        for i in range(3):
            for k, j in enumerate(cols):
                B[i, k] = A[i][j]
            rhs[i] = -A[i][fix]
        if abs(mp.det(B)) < mp.mpf(10) ** (-mp.mp.dps // 2):
            continue
        sol = mp.lu_solve(B, rhs)
        coef = [mp.mpc(0)] * 4
        coef[fix] = mp.mpc(1)
        for k, j in enumerate(cols):
            coef[j] = sol[k]
        resid = max(abs(sum(A[i][j] * coef[j] for j in range(4)))
                    for i in range(3))
        if resid < mp.mpf(10) ** (-mp.mp.dps // 2):
            a, b, c, d = coef
            det = a * d - b * c
            if abs(det) < mp.mpf(10) ** (-mp.mp.dps // 2):
                continue
            z = mp.sqrt(det)
            return a / z, b / z, c / z, d / z
    raise ArithmeticError("degenerate Moebius solve")


def _mp_apply(m, z):
    """(a z + b)/(c z + d) at working precision (moduli.apply_mobius rounds
    to double, which is not enough for the chart)."""
    a, b, c, d = m
    if is_inf(z):
        if c == 0:
            return mp.inf
        return a / c
    z = mp.mpc(z)
    den = c * z + d
    if den == 0:
        return mp.inf
    out = (a * z + b) / den
    # a pole hit only to working precision comes back as a huge finite number
    # (measured: 1.2e20 for the rabbit's infinity after the chart round trip);
    # the chart's cross-ratios take infinity exactly, so snap it
    if abs(out) > mp.mpf(10) ** (mp.mp.dps // 2):
        return mp.inf
    return out


def pole_free_chart(points, probes=(0, 1, -1)):
    """A Möbius chart in which none of `points` is infinity.

    `lift` needs the base vertices finite: a path to infinity cannot be
    continued in the z-chart.  Rather than special-case infinity, rotate the
    sphere so that infinity moves to an interior point of the vertex set (which
    is not a vertex, so no vertex is at infinity afterwards).  The map is a
    rotation, hence chordal lengths -- and so everything geometric -- are
    unchanged, and the Teichmüller chart coordinate is Möbius invariant, so
    nothing observable depends on the choice.

    The whole construction is mpmath: the conjugation has to be accurate to
    working precision or the conjugated map's critical values are wrong at
    1e-16, which splits the double roots of its preimage equations.

    Returns (M, Minv, moved_points) with M = (a,b,c,d) acting as
    z -> (a z + b)/(c z + d).
    """
    vs = [_mp_vec(p) for p in points]
    s = [sum(v[i] for v in vs) for i in range(3)]
    n = mp.sqrt(sum(x * x for x in s))
    if n < mp.mpf(10) ** (-mp.mp.dps + 8):
        raise ValueError("the vertex vectors cancel; no interior point to rotate to")
    s = [x / n for x in s]
    probe_vecs = [_mp_vec(z) for z in probes]
    if min(mp.sqrt(sum((a - b) ** 2 for a, b in zip(pv, s)))
           for pv in probe_vecs) < mp.mpf("1e-6"):
        s = [s[0], s[1], s[2] + mp.mpf("1e-3")]
        n = mp.sqrt(sum(x * x for x in s))
        s = [x / n for x in s]
    R = _rotation_to_north(s)
    images = [_mp_pt(_mp_mv(R, v)) for v in probe_vecs]
    if any(not mp.isfinite(z) for z in images):
        raise ArithmeticError("the probe points are not finite after rotation")
    M = _mobius_through(list(zip(probes, images)))
    det = M[0] * M[3] - M[1] * M[2]
    Minv = (M[3] / det, -M[1] / det, -M[2] / det, M[0] / det)
    moved = [_mp_apply(M, p) for p in points]
    if any(is_inf(z) for z in moved):
        raise ArithmeticError("a vertex stayed at infinity in the rotated chart")
    return M, Minv, [mp.mpc(z) for z in moved]


def _preimage_lists(base, f):
    """[(refined preimage, degree), ...] over each base vertex, in the
    pole-free chart, plus the chart map."""
    M, Minv, moved = pole_free_chart(base.points)
    g = mobius_conjugate(f, *M)
    pre = []
    for p in moved:
        refined = []
        for (q, deg) in g.preimage_with_degrees(mp.mpc(p)):
            if mp.isfinite(q):
                q = _polish(g, mp.mpc(q), mp.mpc(p))
            refined.append((q, deg))
        pre.append(refined)
    return M, Minv, moved, g, pre


def _local_degrees(prelist, deg):
    """The local degree of each preimage, from the reported degrees.

    `RationalMap.local_degree` reads a critical preimage as simple when the
    preimage itself is only located to about 1e-9 (a double root splits under a
    1e-16 perturbation of the target), so the reported degrees are checked
    against their sum and repaired by the only rules that are left: one preimage
    carries the whole degree, and deg f preimages are all simple.
    """
    rep = [d for (_q, d) in prelist]
    if sum(rep) == deg:
        return rep
    if len(prelist) == 1:
        return [deg]
    if len(prelist) == deg:
        return [1] * deg
    raise DegenerateLift(
        f"cannot read the local degrees over a vertex: reported {rep}, which do"
        f" not sum to the degree {deg}")


def _repair_pairing(base, pre, a, b, pairs, deg):
    """Re-pair the lifted arcs over a base edge when the lift mis-assigned one.

    The pairing is constrained: over the base edge (a, b), the preimage k of a is
    an endpoint of exactly its local degree many arcs, and the same on the b
    side.  A path lift can violate it -- measured on the rabbit-by-Moebius map
    (both arcs of one edge landed on a preimage of local degree 1) and on the
    rabbit at the first refinement (both arcs out of infinity landed on the same
    preimage of the far vertex).  When it does, the pairing is re-solved as a
    small assignment problem minimising the discretisation error
    |log chord(preimages) - log chord(base edge)|, which is small only for the
    pairing the cover actually gives.  Whatever comes out is still checked
    downstream: the faces must be consistent and the mesh must validate.
    """
    ea = _local_degrees(pre[a], deg)
    eb = _local_degrees(pre[b], deg)
    req_a = [k for k, e in enumerate(ea) for _ in range(e)]
    req_b = [k for k, e in enumerate(eb) for _ in range(e)]
    if (sorted(ka for ka, _ in pairs) == sorted(req_a)
            and sorted(kb for _, kb in pairs) == sorted(req_b)):
        return pairs
    va = np.array([to_vector(q) for (q, _d) in pre[a]])
    vb = np.array([to_vector(q) for (q, _d) in pre[b]])
    base_chord = float(np.linalg.norm(to_vector(base.points[a])
                                      - to_vector(base.points[b])))

    def cost(ka, kb):
        chord = float(np.linalg.norm(va[ka] - vb[kb]))
        if chord <= 0.0:
            return 1e6
        return abs(np.log(chord) - np.log(base_chord))

    d = len(req_a)
    if d > 6:
        raise DegenerateLift(f"too many arcs over base edge {(a, b)} to re-pair")
    best, best_cost = None, None
    for perm in itertools.permutations(range(d)):
        cand = [(req_a[i], req_b[perm[i]]) for i in range(d)]
        c = sum(cost(ka, kb) for (ka, kb) in cand)
        if best_cost is None or c < best_cost:
            best, best_cost = cand, c
    return best


def lift_by_edges(mesh, f):
    """The congruent lift, built from the lifted *edges*.

    A face-based construction needs an interior sample point per face and a
    shortest-path lift from it to each corner, and both are fragile on a coarse
    mesh (measured on the rabbit's tetrahedron and its refinements: a sample
    point in a big face is close to the wrong preimages, and both the
    shortest-path and the nearest-corner constructions produce surfaces whose
    vertex links are not cycles).  The edges are better conditioned: an arc's interior point is far from the critical values
    at its ends, so its preimages are well separated and each gives one lifted
    edge as a pair of neighbouring preimages.

    The faces then need no guessing and no search.  At a lifted vertex of local
    degree 1 the cyclic order of the edge-ends is forced -- it is the base order
    at the vertex it lies over -- so the sector over any base face is known, and
    the lift of that face sitting in the sector is determined.  A base face with
    two critical corners would have the edge between them, which lifts to a
    bigon and is excluded, so every base face has at least two simple corners and
    all of its deg f lifts are determined by them.  (The round positions cannot
    supply the order at a critical preimage: measured on the rabbit, two
    neighbours are 1.18 degrees apart and the tangent order disagrees with the
    combinatorial one, so an order read off the geometry is wrong there.)

    Requires the base mesh to have no diameter -- the midpoint of an antipodal
    pair is the origin, not a point of the sphere -- and no edge joining two
    single-preimage vertices (a bigon); `discrete.base_mesh` builds meshes that
    have neither, and `DegenerateLift` is raised when they are present.
    """
    M, Minv, moved, g, pre = _preimage_lists(mesh, f)
    base = Mesh(moved, mesh.faces, source=mesh.source, name=mesh.name)
    points, source, lookup = [], [], {}
    for i, plist in enumerate(pre):
        for k, (q, _deg) in enumerate(plist):
            lookup[(i, k)] = len(points)
            points.append(mp.inf if not mp.isfinite(q) else mp.mpc(q))
            source.append((i, k))
    single = {i for i, plist in enumerate(pre) if len(plist) == 1}
    for (a, b) in base.edges():
        if a in single and b in single:
            raise DegenerateLift(
                f"base edge {(a, b)} joins two single-preimage vertices: its"
                " lift is a bigon, so the lifted faces are not triangles")
    edge_pairs = {}
    base_edges = set(base.edges())
    incident = defaultdict(list)
    for (a, b) in base.edges():
        va, vb = _mp_vec(base.points[a]), _mp_vec(base.points[b])
        mid = [(va[i] + vb[i]) / 2 for i in range(3)]
        nrm = mp.sqrt(sum(x * x for x in mid))
        if nrm < mp.mpf(10) ** (-mp.mp.dps + 8):
            raise DegenerateLift(f"base edge {(a, b)} is a diameter")
        midpoint = _mp_pt([x / nrm for x in mid])
        images = g.preimage_with_degrees(midpoint)
        if len(images) != f.deg:
            raise DegenerateLift(
                f"an edge midpoint has {len(images)} preimages, not deg f")
        pairs = []
        for (w, _dg) in images:
            ends = []
            for v, pv in ((a, va), (b, vb)):
                t = _continue_arc(g, w, [x / nrm for x in mid], pv)
                if mp.isfinite(t):
                    t = _polish(g, mp.mpc(t), base.points[v], steps=60)
                ends.append(_nearest(pre[v], t))
            pairs.append((ends[0], ends[1]))
        pairs = _repair_pairing(base, pre, a, b, pairs, f.deg)
        pairs = [(lookup[(a, ka)], lookup[(b, kb)]) for (ka, kb) in pairs]
        if len(set(pairs)) != len(pairs):
            raise DegenerateLift(
                f"the lift of base edge {(a, b)} gave repeated pairs {pairs}:"
                " a bigon, or an arc too long for the nearest preimage to be"
                " the right one")
        edge_pairs[(a, b)] = pairs
        for (u, v) in pairs:
            incident[u].append((v, (a, b)))
            incident[v].append((u, (a, b)))
    # ---- the faces, from the corners that are forced
    #
    # No search is needed, and this is why.  At a lifted vertex q over a with
    # local degree 1 the cyclic order of the edge-ends is forced: it is the base
    # order at a, transported by the bijection between the lifted edges at q and
    # the base edges at a.  So the sector at q over any base face is known, and
    # the lifted face sitting in that sector is
    #
    #     {q, other end over (a, b), other end over (a, c)}
    #
    # for the base face (a, b, c).  Every base face has at most one critical
    # corner -- a face with two would have the edge between them, which
    # `discrete.base_mesh` excludes because it lifts to a bigon -- so every base
    # face has at least two simple corners, and its deg f lifts are all
    # determined by them.  (The round positions cannot supply the order at a
    # critical preimage: measured on the rabbit, two neighbours are 1.18 degrees
    # apart and the tangent order disagrees with the combinatorial one.)
    ends = defaultdict(list)                  # (q, base edge) -> [(other, edge)]
    for q, lst in incident.items():
        for (other, (a, b)) in lst:
            key = edge_key(a, b)                  # edges are stored sorted
            ends[(q, key)].append((other, key))
    local_degree = {}
    for q in incident:
        a = source[q][0]
        counts = {len(ends.get((q, edge_key(a, b)), [])) for b in range(len(base.points))
                  if edge_key(a, b) in base_edges}
        if len(counts) != 1:
            raise DegenerateLift(
                f"lifted vertex {q} over base vertex {a} has {sorted(counts)}"
                " edge-ends per base edge: the edge lift is not a covering")
        local_degree[q] = counts.pop()
        if local_degree[q] > 2:
            raise DegenerateLift(
                f"lifted vertex {q} has local degree {local_degree[q]}; only"
                " local degree 1 and 2 are implemented")
    over = defaultdict(list)               # base vertex -> [lifted vertices]
    for q in incident:
        over[source[q][0]].append(q)
    single = {a for a, lst in over.items()
              if len(lst) == 1 and len(pre[a]) == 1}
    for T in base.faces:
        if sum(1 for v in T if v in single) > 1:
            raise DegenerateLift(
                f"base face {T} has two critical corners, so its edge between"
                " them lifts to a bigon")
    faces = set()
    for T in base.faces:
        for i in range(3):
            v, v1, v2 = T[i], T[(i + 1) % 3], T[(i + 2) % 3]
            if v in single:
                continue
            e1, e2 = edge_key(v, v1), edge_key(v, v2)
            for q in over[v]:
                if local_degree[q] != 1:
                    continue
                r1 = _the_end(ends[(q, e1)])
                r2 = _the_end(ends[(q, e2)])
                if r1 is None or r2 is None or len({q, r1, r2}) != 3:
                    raise DegenerateLift(
                        f"the sector at lifted vertex {q} over base face {T}"
                        " is degenerate")
                faces.add(frozenset((q, r1, r2)))
    expected_faces = f.deg * len(base.faces)
    if len(faces) != expected_faces:
        raise DegenerateLift(
            f"{len(faces)} lifted faces, expected deg f * base faces ="
            f" {expected_faces}")
    fl = orient_faces(points, [tuple(sorted(u)) for u in faces])
    out = Mesh([_mp_apply(Minv, z) for z in points], fl, source=source,
               name=f"liftE({mesh.name})")
    stats = out.validate()
    err = _discretisation_error(base, out)
    return out, {"faces": stats["F"], "vertices": stats["V"],
                 "edges": stats["E"], "chart": M,
                 "discretisation_error": err}


def _the_end(lst):
    """The single other endpoint of a lifted edge at a vertex of local degree 1."""
    if len(lst) != 1:
        return None
    return lst[0][0]


def _discretisation_error(base, lifted):
    """Sum over the lifted faces of |log chord(analytic) - log flat length|.

    The flat length of a lifted edge is its base edge's chordal length (the lift
    is congruent), and the analytic chord is the distance between the analytic
    preimages.  The gap between the two is the discretisation error of the
    scheme -- exactly what the refinement study measures -- and it is the
    number that should fall as the mesh refines.
    """
    flat = congruent_lengths(base, lifted)
    vecs = lifted.vectors()
    total = 0.0
    for (i, j, k) in lifted.faces:
        for (u, w) in ((i, j), (j, k), (i, k)):
            chord = float(np.linalg.norm(vecs[u] - vecs[w]))
            total += abs(np.log(chord) - np.log(flat[edge_key(u, w)]))
    return total


def _nearest(prelist, w):
    """Index of the preimage in `prelist` nearest to w."""
    best, bestd = None, None
    for k, (q, _deg) in enumerate(prelist):
        if not mp.isfinite(q):
            d = mp.mpf(0) if not mp.isfinite(w) else mp.inf
        elif not mp.isfinite(w):
            d = mp.inf
        else:
            d = abs(mp.mpc(q) - mp.mpc(w))
        if bestd is None or d < bestd:
            best, bestd = k, d
    return best


def congruent_lengths(base, lifted):
    """The lifted mesh's metric: every lifted edge carries its base edge's
    length, i.e. every lifted triangle is congruent to its base triangle.

    The base edge of a lifted edge is not well defined by the edge alone (a
    lifted edge lies in two lifted faces, whose base faces are the two base
    faces at the shared base edge -- so both give the same value, and that
    agreement is asserted here).
    """
    base_len = base.chordal_lengths()
    known = {}
    for lf in lifted.faces:
        bs = [lifted.source[i][0] for i in lf]
        # each lifted face is congruent to the base face on its sources; the
        # base face is the one whose vertex set is those three sources
        for k in range(3):
            i, j = lf[k], lf[(k + 1) % 3]
            bi, bj = lifted.source[i][0], lifted.source[j][0]
            ell = base_len[edge_key(bi, bj)]
            key = edge_key(i, j)
            if key in known and abs(known[key] - ell) > 1e-12:
                raise ArithmeticError(
                    f"lifted edge {key} gets two lengths, {known[key]} and {ell}"
                    ": the two faces over it are not congruent")
            known[key] = ell
    missing = set(lifted.edges()) - set(known)
    if missing:
        raise ArithmeticError(f"lifted edges without a length: {sorted(missing)[:4]}")
    return known
