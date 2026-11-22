"""Initial geometric state from an abstract ribbon virtual endomorphism.

Purpose
-------
`pullback.iteration.FrozenStep` needs an analytic rational map: it builds the
lifted cover of a base mesh by *preimages* of an explicit f.  This module builds
the same two-dimensional structure from nothing but a session (base ribbon
graph Gamma_0, cover pi: Gamma_1 -> Gamma_0, graph map phi): no rational map, no
critical values, no preimages.

What the data determines, and how
---------------------------------
The punctures of the base surface are the *faces* of Gamma_0, the punctures of
the source surface are the faces of Gamma_1, and the covering of surfaces is the
covering of graphs pi.  Three things have to be read out of (pi, phi):

* `projection`  face of Gamma_1 -> face of Gamma_0 and the local degree, i.e.
  the branched-cover structure over the punctures.  This is `face_map_pi`
  (regions map to regions, degree = ratio of boundary-cycle lengths);
* `diagonal`    which face of Gamma_1 *is* the puncture p_j, for each base
  label j.  This is `face_map_phi`'s tight (embedding) lift: exactly one face of
  Gamma_1 has a phi-boundary image wrapping face j.  It is the datum the naive
  reading calls "the missing marking": it is not missing, it is computed here,
  and where it is *not* unique the constructor raises instead of guessing.
* `dynamics`    j -> projection(diagonal(j)) and `local_degrees` j -> the local
  degree of f at p_j.  Both follow from the two rows above; nothing is assumed
  about the map.

The initial geometric realisation is then an *initialisation choice*, not an
input the data lacks: any n distinct points of the sphere, with explicit
labels, will do.  Zeros and poles of the discretisation are chosen as functions
of that configuration (see `_aux_positions`), so the discretisation depends on
the labelled configuration and not on a representative.

The frozen cover
----------------
`frozen_cover` returns the dual cell complexes of Gamma_0 and Gamma_1 as
EDGE-LABELLED complexes: vertices are the punctures, one labelled edge per edge
of the spine (so a doubled edge keeps both copies), faces are recorded as cyclic
sequences of edge labels (so a face that meets a puncture twice keeps both
corners).  This is the answer to the self-adjacent dual face that a star mesh
cannot represent: `lift.dual_cells` keeps darts, and `rounding_multi`'s
criterion is per edge label, so no polygon ever has to be a simple polygon.

Every lifted edge answers to a base edge through pi, so the metric is
transferred by label, exactly as `FrozenStep` does it.  `step` then uniformises
the lifted complex and reads the labelled punctures off it.
"""

from __future__ import annotations

import numpy as np

from pullback import mesh as msh
from pullback import rounding_multi as mg
from pullback.discrete import chart_coordinate
from pullback.homotopy import peripheral_images
from pullback.lift import dual_cells, pullback_ve
from pullback.lift_round import labelled_corners
from pullback.portrait import face_map_phi, face_map_pi

__all__ = ["PunctureData", "puncture_data", "frozen_cover", "InitialState",
           "arbitrary_positions", "base_lengths", "lifted_lengths"]


# --------------------------------------------------------------------- labels
class PunctureData:
    """The combinatorial initial state: punctures, labels, cover, dynamics."""

    def __init__(self, base, cover, phi):
        self.base, self.cover, self.phi = base, cover, phi
        self.base_faces = [tuple(w) for w in base.faces()]
        self.cover_faces = [tuple(w) for w in cover.faces()]
        n0, n1 = len(self.base_faces), len(self.cover_faces)
        self.labels = list(range(n0))
        self.lifted_labels = list(range(n1))

        self._degree_clash = []
        prows, _ = face_map_pi(cover, base)
        frows, _ = face_map_phi(phi)
        self.pi_portrait = prows
        self.phi_portrait = frows

        for i, row in enumerate(prows):
            if row["target"] is None:
                raise ValueError(
                    f"cover face {i} does not project onto a power of a base "
                    f"face boundary: pi is not a covering with this data")


        # ---- the diagonal: exactly one tight phi face wraps each base face
        marked = {}
        for C, row in enumerate(frows):
            if row["target"] is not None:
                marked.setdefault(row["target"], []).append(C)
        bad = {j: v for j, v in marked.items() if len(v) != 1}
        if bad or sorted(marked) != self.labels:
            raise ValueError(
                "the diagonal (which face of the cover is the puncture p_j) is "
                "not determined: base faces -> candidate cover faces "
                f"{ {j: sorted(v) for j, v in sorted(marked.items())} }"
                f" of {n0} base labels")
        self.diagonal = {j: v[0] for j, v in marked.items()}

        self.projection = {C: prows[C]["target"] for C in self.lifted_labels}
        self.face_degree = {C: prows[C]["degree"] for C in self.lifted_labels}
        self.dynamics = {j: self.projection[C] for j, C in self.diagonal.items()}
        self.local_degrees = {j: self.face_degree[C] for j, C in self.diagonal.items()}
        self.critical = {j: d >= 2 for j, d in self.local_degrees.items()}
        # a lifted vertex's "over" set: which punctures of the source sit over j
        self.over = {j: [] for j in self.labels}
        for C in self.lifted_labels:
            self.over[self.projection[C]].append(C)
        self.inactive = [C for C in self.lifted_labels
                         if C not in set(self.diagonal.values())]

        self.degree_profile = {j: sorted(self.face_degree[C] for C in self.over[j])
                               for j in self.labels}
        self.degree = len(cover.vertices()) // len(base.vertices())
        self.certificates = self._certificates()

    def _certificates(self):
        d = self.degree
        sums = {j: sum(self.degree_profile[j]) for j in self.labels}
        # the diagonal faces are distinct and each sits over its own image
        distinct = len(set(self.diagonal.values())) == len(self.labels)
        sits_over = all(self.projection[C] == self.dynamics[j]
                        for j, C in self.diagonal.items())
        # Euler for both spine duals
        def euler(graph):
            return (len(graph.faces()) - len(graph.edge_list())
                    + len(graph.vertices()))
        # phi's own wrapping multiplicity on the diagonal faces.  It need not
        # equal the local degree of f: phi represents the virtual endomorphism
        # h = phi_* pi_*^{-1}, not the covering f.  The *target* is what carries
        # the diagonal, and by construction phi(diagonal(j)) wraps j.
        phi_wrap = {j: self.phi_portrait[C]["degree"]
                    for j, C in self.diagonal.items()}
        return {
            "degree": d,
            "cover_degree_sums": sums,
            "cover_degree_ok": all(v == d for v in sums.values()),
            "diagonal_distinct": distinct,
            "diagonal_sits_over_its_image": sits_over,
            "n_base_punctures": len(self.labels),
            "n_lifted_punctures": len(self.lifted_labels),
            "n_inactive": len(self.inactive),
            "inactive_faces": list(self.inactive),
            "local_degrees": dict(self.local_degrees),
            "degree_profile": {j: list(v) for j, v in self.degree_profile.items()},
            "phi_wrap_on_diagonal": phi_wrap,
            "phi_wraps_its_own_label": all(
                self.phi_portrait[C]["target"] == j
                for j, C in self.diagonal.items()),
            "euler_base_dual": euler(self.base),
            "euler_lifted_dual": euler(self.cover),
            # Riemann-Hurwitz for a degree-d self-cover of the sphere: the sum
            # is over ALL punctures of the source, not only the diagonal ones
            "riemann_hurwitz": sum(self.face_degree[C] - 1
                                   for C in self.lifted_labels),
            "free_dimension": None,
        }

    def chart_points(self, positions):
        """The labelled configuration as vectors, in label order (a cheap check
        that the caller's positions are in the same order as the labels)."""
        if len(positions) != len(self.labels):
            raise ValueError(f"{len(positions)} positions for {len(self.labels)} labels")
        return [msh.to_vector(p) for p in positions]

    def __repr__(self):
        return (f"<PunctureData {len(self.labels)} base / "
                f"{len(self.lifted_labels)} lifted punctures, degree {self.degree}>")


def puncture_data(session):
    """`PunctureData` of a `pullback.Session` — reads only graph/cover/phi."""
    return PunctureData(session.base, session.cover, session.phi)


# ------------------------------------------------------------------ geometry
def arbitrary_positions(n, seed=0):
    """An arbitrary initial realisation: n labelled points of the sphere.

    This is the initialisation choice, and nothing about the data prefers one
    choice over another.  Deterministic given `seed` so that a run is
    reproducible; a different seed is a different (equally valid) start.
    """
    if n < 3:
        raise ValueError("a punctured sphere needs at least three punctures")
    rng = np.random.RandomState(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    # keep the points apart: a start with two coincident punctures has no metric
    for _ in range(200):
        d = mg.chordal_lengths(v)
        d = d + np.eye(n)
        if d.min() > 0.15:
            break
        v = rng.normal(size=(n, 3))
        v /= np.linalg.norm(v, axis=1)[:, None]
    return [msh.to_point(x) for x in v]


def base_lengths(cells, positions):
    """{base edge label: chordal length} of a corner-slot dual complex.

    The two endpoints of a dual edge are punctures (dual vertices are faces of
    the spine), so the length is a chord of the current configuration.  The
    dual's own edge labels are kept, so a doubled edge keeps both copies.
    """
    vectors = [msh.to_vector(p) for p in positions]
    out = {}
    dual = cells["dual"]
    for label in cells["edges"]:
        a, b = dual.edges[label]
        if a == b:
            raise ValueError(
                f"dual edge {label!r} is a self-loop at puncture {a}; its "
                "length is not determined by the labelled configuration.  "
                "Use a bridge-free combinatorics before computing a metric")
        out[label] = float(np.linalg.norm(np.asarray(vectors[a])
                                          - np.asarray(vectors[b])))
    return out


def lifted_lengths(cells, base_len):
    """{lifted edge label: length of the base edge it covers}.

    A lifted dual edge label is `(base_edge, sheet)`, so the base edge is its
    first component -- the same reading `lift.lifted_delaunay_metric` uses.
    """
    out = {}
    for label in cells["edges"]:
        try:
            base_edge = label[0]
        except TypeError as exc:                      # pragma: no cover
            raise ValueError(f"lifted edge {label!r} carries no base label") from exc
        if base_edge not in base_len:
            raise ValueError(f"lifted edge {label!r} covers unknown base edge "
                             f"{base_edge!r}")
        out[label] = base_len[base_edge]
    return out


def corner_angles(cells, x, edges):
    """Angle sum at every vertex, corner by corner, from the corner slots.

    `rounding_multi.cone_angles` reads the angle at a corner from the two
    consecutive edge labels, which is ambiguous exactly when those two labels
    are a doubled edge (both sides of one face joining the same pair of
    punctures -- the rabbit has one).  This version takes the corner slots from
    `cells["polygons"]`, so it can *name* the ambiguous corners instead of
    refusing: a doubled edge makes its two corners unmeasurable in the flat
    metric, and the count is reported.  Diagnostic only -- no step uses angles.
    """
    dual = cells["dual"]
    total = np.zeros(len(x))
    ambiguous = []
    for k, (poly, walk) in enumerate(zip(cells["polygons"], cells["corners"])):
        m = len(poly)
        for t in range(m):
            v = poly[t]
            a, b = poly[t - 1], poly[(t + 1) % m]
            if a == b:
                ambiguous.append((k, t, v))
                continue
            u = x[a] - x[v]
            w = x[b] - x[v]
            nu, nw = np.linalg.norm(u), np.linalg.norm(w)
            if nu == 0.0 or nw == 0.0:
                ambiguous.append((k, t, v))
                continue
            total[v] += float(np.arccos(np.clip(np.dot(u, w) / (nu * nw), -1.0, 1.0)))
    return total, ambiguous


def frozen_cover(state):
    """The frozen lifted cover of a `PunctureData`, as labelled complexes.

    Returns a dict with the base cells, the lifted cells, the edge-label
    projection, the corner (face) sequences and the label vertices.  Everything
    here is fixed once and never depends on a configuration.
    """
    base_cells = dual_cells(state.base)
    lift_cells = dual_cells(state.cover)
    edges = {label: tuple(lift_cells["dual"].edges[label])
             for label in lift_cells["edges"]}
    label_vertices = {j: state.diagonal[j] for j in state.labels}
    # the lifted dual's vertices ARE the faces of the cover, so a label vertex
    # is a dual vertex index and nothing has to be searched for
    for j, v in label_vertices.items():
        if v not in lift_cells["vertices"]:
            raise ValueError(f"label {j}: vertex {v} is not a lifted dual vertex")
    return {
        "base": base_cells,
        "lift": lift_cells,
        "edges": edges,
        "corners": labelled_corners(lift_cells),
        "base_corners": labelled_corners(base_cells),
        "label_vertices": label_vertices,
        "projection": dict(state.projection),
        "face_degree": dict(state.face_degree),
        "degree": state.degree,
        "certificates": _cover_certificates(state, base_cells, lift_cells, edges),
    }


def _cover_certificates(state, base_cells, lift_cells, edges):
    """Structural checks: it is a sphere, the labels are a cover, edges lift."""
    def counts(cells):
        return dict(cells["counts"])

    # every lifted edge covers a base edge, and the count over each base edge
    # is the degree (a covering of graphs of degree d, edges counted with
    # multiplicity)
    per_base = {}
    for label in edges:
        per_base.setdefault(label[0], []).append(label)
    cov = {b: len(v) for b, v in per_base.items()}
    deg = state.degree
    bad = {b: c for b, c in cov.items() if c != deg}
    missing = set(base_cells["edges"]) - set(per_base)
    # face walks: a corner slot is a dart, so a repeated puncture is legal and
    # the corner sequence of edge labels carries which copy it is
    def corner_ok(cells, corners):
        return all(len(w) == len(c) and len(c) >= 2
                   for w, c in zip(cells["corners"], corners))
    return {
        "base_counts": counts(base_cells),
        "lift_counts": counts(lift_cells),
        "base_euler": (base_cells["counts"]["V"] - base_cells["counts"]["E"]
                       + base_cells["counts"]["F"]),
        "lift_euler": (lift_cells["counts"]["V"] - lift_cells["counts"]["E"]
                       + lift_cells["counts"]["F"]),
        "edges_over_base": cov,
        "edges_over_base_ok": not bad and not missing,
        "edges_over_base_bad": bad,
        "base_edges_missed": sorted(missing, key=str),
        "non_simple_base_faces": list(base_cells["non_simple_faces"]),
        "non_simple_lift_faces": list(lift_cells["non_simple_faces"]),
        "corners_ok": (corner_ok(base_cells, labelled_corners(base_cells))
                       and corner_ok(lift_cells, labelled_corners(lift_cells))),
        "label_vertices": dict(state.diagonal),
        "free_dimension": _free_dimension(edges),
        "deficit": 2 * lift_cells["counts"]["V"] - 6 - _free_dimension(edges),
    }


# ------------------------------------------------------------- the frozen step
class InitialState:
    """An initial geometric state plus the frozen lifted cover, and its step.

    `InitialState(session, seed=0)` is the whole constructor: it needs the
    session and a seed for the arbitrary initial realisation, nothing else.
    """

    def __init__(self, session, seed=0, positions=None):
        self.session = session
        self.data = puncture_data(session)
        self.cover = frozen_cover(self.data)
        self.positions = (list(positions) if positions is not None
                          else arbitrary_positions(len(self.data.labels), seed=seed))
        self.seed = seed
        n = len(self.data.labels)
        # Public configurations keep these stable external labels.  Initially
        # they are just the base face indices; an explicit combinatorial update
        # replaces this with its comparison-induced permutation.
        self.data_labels = list(range(n))
        self._x0 = None
        self.convergence = session.convergence()
        self._initial_chart = self.chart(self.positions)

    # ------------------------------------------------------------- invariants
    def chart(self, positions=None):
        """The labelled Teichmuller coordinate: Mobius-free, so two
        realisations of the same labelled configuration agree here."""
        pos = self.positions if positions is None else positions
        return chart_coordinate(self.data.chart_points(pos))

    @property
    def n(self):
        return len(self.data.labels)

    def _data_positions(self, positions=None):
        """Internal face order from the stable external puncture labels."""
        pos = self.positions if positions is None else list(positions)
        if len(pos) != self.n:
            raise ValueError(f"{len(pos)} positions for {self.n} labels")
        try:
            return [pos[self.data_labels[j]] for j in self.data.labels]
        except IndexError as exc:
            raise ValueError("positions do not cover the external labels") from exc

    def update_combinatorics(self, new_spine, h_inverse, h=None, positions=None):
        """Return a state on explicitly supplied combinatorics.

        `h_inverse : new_spine -> current base` and its inverse `h` are passed
        to `pullback_ve`; neither is inferred from the configuration.  The
        current object is not mutated.  The new state's frozen cover is built
        once here, and subsequent `step`/`iterate` calls never rebuild it.

        Inherited puncture labels are stable: `positions`, `chart`, step
        `points` and step `coordinate` all use the original external order,
        even when the new spine has a different face order.  `data_labels`
        records new base-face index -> external label, and `comparison` keeps
        the comparison maps and both directions of that correspondence.
        """
        data = self.data
        P, pi_prime, phi_prime, projection = pullback_ve(
            data.base, data.cover, data.phi, new_spine, h_inverse, h=h)
        new_to_old = peripheral_images(h_inverse)
        if sorted(new_to_old) != data.labels:
            raise ValueError(
                "the comparison does not give a complete puncture correspondence")
        data_labels = [self.data_labels[j] for j in new_to_old]
        if sorted(data_labels) != self.data.labels:
            raise ValueError("the inherited external labels are not a permutation")

        new = self.__class__.__new__(self.__class__)
        new.session = self.session
        new.data = PunctureData(new_spine, P, phi_prime)
        new.cover = frozen_cover(new.data)
        new.positions = list(self.positions if positions is None else positions)
        if len(new.positions) != self.n:
            raise ValueError(f"{len(new.positions)} positions for {self.n} labels")
        new.seed = self.seed
        new.data_labels = data_labels
        new._x0 = None
        new.convergence = self.convergence
        new.comparison = {
            "h_inverse": h_inverse,
            "projection_to_cover": projection,
            "new_to_old": dict(enumerate(new_to_old)),
            "data_to_external": dict(enumerate(data_labels)),
            "external_to_data": {label: j for j, label in enumerate(data_labels)},
        }
        # Constructing the portrait above validates pi_prime through P's native
        # cover structure.  Retain both legs explicitly for callers comparing
        # transported VEs rather than only their numerical state.
        new.pi_prime = pi_prime
        new.phi_prime = phi_prime
        new._initial_chart = new.chart()
        return new

    # ------------------------------------------------------------------- step
    def step(self, positions=None, x0=None, tol=1e-12, maxit=200):
        """One discrete Thurston pullback: base config -> lifted realisation ->
        the labelled punctures read off it.

        No analytic map is used: the lengths come from the base chords through
        the covering of dual complexes, and the labels are frozen vertex
        indices of the lifted complex.
        """
        pos = self._data_positions(positions)
        blen = base_lengths(self.cover["base"], pos)
        llen = lifted_lengths(self.cover["lift"], blen)
        x, residual, iters = mg.round_realise(
            self.cover["edges"], llen,
            x0=self._x0 if x0 is None else x0, maxit=maxit, tol=tol)
        structure = mg.structure(self.cover["edges"], x)
        cone, ambiguous = corner_angles(self.cover["lift"], x, self.cover["edges"])
        by_label = {}
        label_vertices = {}
        for data_label, external in enumerate(self.data_labels):
            vertex = self.cover["label_vertices"][data_label]
            by_label[external] = x[vertex]
            label_vertices[external] = vertex
        vectors = [by_label[j] for j in range(self.n)]
        out = [msh.to_point(v) for v in vectors]
        self._x0 = x
        return {
            "points": out,
            "vectors": vectors,
            "coordinate": chart_coordinate(vectors),
            "residual": float(residual),
            "iterations": iters,
            "structure": structure,
            "cone_angles": np.asarray(cone, dtype=float),
            "ambiguous_corners": ambiguous,
            "labels": [label_vertices[j] for j in range(self.n)],
            "label_vertices": label_vertices,
            "base_lengths": blen,
            "x": x,
        }

    def iterate(self, n, positions=None):
        from .convergence import require_convergence_allowed
        require_convergence_allowed(self.session)
        cur = self.positions if positions is None else list(positions)
        out = []
        for _ in range(int(n)):
            r = self.step(cur)
            cur = r["points"]
            out.append(r)
        return out

    def __repr__(self):
        c = self.cover["certificates"]
        return (f"<InitialState {self.session.name}: {self.n} punctures, "
                f"lifted {c['lift_counts']['V']}V/{c['lift_counts']['E']}E>")


# ------------------------------------------------------- closing the moduli
def _proj_dart(dart):
    """A lifted dual dart projected to the base dual dart it covers."""
    return (dart[0][0], dart[1])


def chord_candidates(state, base_cells):
    """Every chord of every base polygon that is not already an edge.

    A chord of the base dual polygon at a spine vertex is an edge *inside* that
    polygon, so it splits the polygon and is a legitimate extra 1-cell.  The
    list is canonically ordered, so the enrichment below is deterministic.
    """
    dual = base_cells["dual"]
    have = {frozenset(map(str, dual.edges[e])) for e in base_cells["edges"]}
    out = []
    for k, (poly, walk) in enumerate(zip(base_cells["polygons"],
                                         base_cells["corners"])):
        m = len(poly)
        for i in range(m):
            for j in range(i + 1, m):
                if j == i + 1 or (i == 0 and j == m - 1):
                    continue
                if poly[i] == poly[j]:
                    continue
                if frozenset((str(poly[i]), str(poly[j]))) in have:
                    continue
                out.append((k, i, j))
    return out


def chord_lifts(state, lift_cells, chord):
    """The lifted copies of one base chord: one per lifted face over its face.

    A base polygon at spine vertex v lifts to the polygons at the cover
    vertices over v.  Corner slots are darts, and a lifted dual dart projects to
    a base dual dart, so corner i of the base polygon has a named corner in each
    lifted polygon.  This is the only covering datum the enrichment needs.
    """
    k, i, j = chord
    base_walk = state.__dict__["_base_corners"][k]
    want = (_proj_dart(base_walk[i]), _proj_dart(base_walk[j]))
    out = []
    for kt, walk in enumerate(state.__dict__["_lift_corners"]):
        pos = {}
        for t, dart in enumerate(walk):
            p = _proj_dart(dart)
            if p in want:
                pos.setdefault(p, []).append(t)
        if len(pos) < 2:
            continue
        if any(len(v) != 1 for v in pos.values()):
            raise ValueError(f"lifted face {kt} meets base corner twice: {pos}")
        out.append((kt, pos[want[0]][0], pos[want[1]][0]))
    return out


def enriched_cover(state, base_cells=None, lift_cells=None, verbose=False):
    """The frozen cover plus enough chords to make the realisation canonical.

    `rounding_multi.structure` measures how many real moduli of the conformal
    class the cell complex cannot see: `excess = 2V - 6 - rank(PJ)`, and a step
    is well defined up to Moebius exactly when excess == 0.  The coarse spine
    dual is far from that (the rabbit's lifted dual has V = 6, E = 6, and a
    tree-like 1-skeleton, so E - rank(A) = 1 against the 6 needed), and no
    amount of *spine* refinement helps: subdividing an edge of the spine adds a
    parallel dual edge, not a new constraint.

    What does help is a chord: an edge inside a dual face, which splits that
    face and (with its d lifted copies) raises E - rank(A) by d - 1 at least.  A
    chord lies in a dual face = at a spine vertex, so its lifted copies are one
    per cover vertex over that spine vertex, and the corner correspondence is
    the covering of dual darts.  The enrichment is greedy in the canonical chord
    order and stops as soon as the count reaches 2V - 6: deterministic, exactly
    specified, and reported.
    """
    if base_cells is None:
        base_cells = dual_cells(state.base)
    if lift_cells is None:
        lift_cells = dual_cells(state.cover)
    state._base_corners = [tuple(w) for w in base_cells["corners"]]
    state._lift_corners = [tuple(w) for w in lift_cells["corners"]]
    edges = {label: tuple(lift_cells["dual"].edges[label])
             for label in lift_cells["edges"]}
    base_edges = {label: tuple(base_cells["dual"].edges[label])
                  for label in base_cells["edges"]}
    n = 1 + max(max(i, j) for (i, j) in edges.values())
    target = 2 * n - 6
    added = []
    for chord in chord_candidates(state, base_cells):
        if _free_dimension(edges) >= target:
            break
        k, i, j = chord
        a, b = base_cells["polygons"][k][i], base_cells["polygons"][k][j]
        base_edges[("chord", k, i, j)] = (a, b)
        for (kt, ti, tj) in chord_lifts(state, lift_cells, chord):
            u = lift_cells["polygons"][kt][ti]
            v = lift_cells["polygons"][kt][tj]
            if u == v:
                raise ValueError(
                    f"chord {chord}: lifted copy in face {kt} is a self-loop "
                    f"at puncture {u}")
            edges[("chord", k, i, j, kt)] = (u, v)
        added.append(chord)
        if verbose:
            print("chord", chord, "free dimension now", _free_dimension(edges))
    free = _free_dimension(edges)
    return {
        "base": base_cells,
        "lift": lift_cells,
        "edges": edges,
        "base_edges": base_edges,
        "added_chords": added,
        "vertices": n,
        "free_dimension": free,
        "target_free_dimension": target,
        "canonical": free == target,
        "corners": [tuple(d[0] for d in w) for w in lift_cells["corners"]],
        "label_vertices": {j: state.diagonal[j] for j in state.labels},
        "base_lengths": _chord_base_lengths,
        "projection": dict(state.projection),
        "degree": state.degree,
    }


def _free_dimension(edges):
    """E - rank(A) - redundant: the number of independent length constraints."""
    order = mg.edge_order(edges)
    n = 1 + max(max(i, j) for (_l, i, j) in order)
    A = mg.edge_incidence(edges, n)[0]
    pairs = {}
    for (label, i, j) in order:
        pairs.setdefault((min(i, j), max(i, j)), []).append(label)
    redundant = sum(len(v) - 1 for v in pairs.values())
    return int(len(order) - np.linalg.matrix_rank(A) - redundant)


def _chord_base_lengths(cells_pack, positions):
    """{edge label: chordal length} including the chords."""
    vectors = [msh.to_vector(p) for p in positions]
    out = {}
    for label, (a, b) in cells_pack["base_edges"].items():
        if a == b:
            raise ValueError(
                f"dual edge {label!r} is a self-loop at puncture {a}; its "
                "length is not determined by the labelled configuration")
        out[label] = float(np.linalg.norm(np.asarray(vectors[a])
                                          - np.asarray(vectors[b])))
    return out


def _chord_lifted_lengths(cells_pack, base_len):
    out = {}
    for label, _pair in cells_pack["edges"].items():
        if label[0] == "chord":
            base_edge = ("chord", label[1], label[2], label[3])
        else:
            base_edge = label[0]
        if base_edge not in base_len:
            raise ValueError(f"lifted edge {label!r} covers no base edge")
        out[label] = base_len[base_edge]
    return out


class EnrichedState:
    """An `InitialState` whose lifted complex has excess zero by construction."""

    def __init__(self, session, seed=0, positions=None, verbose=False):
        self.session = session
        self.data = puncture_data(session)
        self.pack = enriched_cover(self.data, verbose=verbose)
        self.positions = (list(positions) if positions is not None
                          else arbitrary_positions(len(self.data.labels), seed=seed))
        self._x0 = None

    @property
    def n(self):
        return len(self.data.labels)

    def step(self, positions=None, x0=None, tol=1e-12, maxit=400):
        pos = self.positions if positions is None else list(positions)
        blen = _chord_base_lengths(self.pack, pos)
        llen = _chord_lifted_lengths(self.pack, blen)
        x, residual, iters = mg.round_realise(
            self.pack["edges"], llen, x0=self._x0 if x0 is None else x0,
            maxit=maxit, tol=tol)
        structure = mg.structure(self.pack["edges"], x)
        labels = [self.pack["label_vertices"][j] for j in self.data.labels]
        vectors = [x[v] for v in labels]
        self._x0 = x
        return {"points": [msh.to_point(v) for v in vectors],
                "vectors": vectors,
                "coordinate": chart_coordinate(vectors),
                "residual": float(residual), "iterations": iters,
                "structure": structure, "labels": labels, "x": x}

    def iterate(self, n, positions=None):
        from .convergence import require_convergence_allowed
        require_convergence_allowed(self.session)
        cur = self.positions if positions is None else list(positions)
        out = []
        for _ in range(int(n)):
            r = self.step(cur)
            cur = r["points"]
            out.append(r)
        return out

    def chart(self, positions=None):
        pos = self.positions if positions is None else positions
        return chart_coordinate([msh.to_vector(p) for p in pos])
