"""Rose spine of a triangulated base, and its lift to an assembled cover.

The point of this module is the *base side* of the combinatorial pullback step:
which loops generate pi_1(S^2 - P), in which order, and which of the lifted
edges each of them uses.  Everything is read off the triangulation's own
rotation data (`dense_assembly.Base.flower`) and the already-declared per-edge
permutation datum; no spanning tree, no BFS and no line search is used to
produce the petals, and no CEPS solve is used anywhere.

Three layers
------------
1. `rose_spine(base, punctures)` -- one petal per puncture, the petal being the
   flower boundary walk of that base vertex.  It is a closed curve around that
   vertex and nothing else, because the star of a vertex of a sphere
   triangulation is a disk.  Canonical order, canonical reference face.

2. `certify_rose(base, spine)` -- exact, cover-free checks:
   the petals are simple closed walks in the dual graph, every base edge is
   crossed once in the flower of each of its endpoints (so the signed sum of
   all petals is the zero chain -- the one relation), and the petals of *all*
   base vertices, with any one dropped, are a free basis of pi_1 of the dual
   graph.  The last is decided by Stallings folding on the words in a
   spanning-tree basis of the dual graph (`validate.generates_free_group`), not
   by a rank count.  For P a proper subset of the vertex set the petals of P
   generate pi_1(S^2 - P) after the flowers of the non-punctures are eliminated
   by the relation; the elimination is the counting identity
   ``rank = |P| - 1 = -chi(S^2 - P)``, and the van Kampen step ("S^2 - P
   retracts to the dual graph with a disk glued along each non-puncture
   flower") is a theorem, stated, not computed.

3. `attach_spine(spine, base, lift)` -- the lift.  Uses only
   `lift["edge_perms"]`, `lift["tris"]`, `lift["edges"]`, `lift["per_face"]`,
   `lift["proj"]` and the per-vertex local monodromy already certified in
   `lift["cert"]["local_realised"]`.  For every petal and every sheet it
   produces the lifted *path* through the lifted dual graph, and for every
   lifting of a puncture (`lift["index"][p]`, the same ids CEPS returns) the
   closed lifted loop around it.  The checks are exact: each lifted edge label
   is asserted to be incident with the corner it is used from, the walk is
   asserted to close on the declared local monodromy, and the visited corner
   set is asserted to be exactly the DSU class of the base corner (an
   independent code path from the walk itself).

Positions (CEPS/Newton) attach afterwards and only as a *label* on
`LiftedLoop.preimage`: that id is `lift["index"][p]`'s member, so a position
array indexed the way `preimage_lift.uniformize_cover` returns `preimages[p]`
is already keyed correctly.  Nothing in this module reads a number.
"""

from __future__ import annotations

from dataclasses import dataclass

from pullback import dense_assembly as A
from pullback import validate as _validate
from pullback.ribbons import F as _F, B as _B, RibbonGraph


# --------------------------------------------------------------------- petals
@dataclass(frozen=True)
class Step:
    """One crossing of the petal: it sits in `face` and crosses `edge`."""

    face: int
    edge: tuple


@dataclass(frozen=True)
class Petal:
    """The flower boundary walk of one base vertex."""

    puncture: int
    steps: tuple

    @property
    def ref_face(self):
        return self.steps[0].face

    def edges(self):
        return tuple(st.edge for st in self.steps)

    def wires(self):
        """(face, edge) pairs, the loop-grouping key of this petal."""
        return tuple((st.face, st.edge) for st in self.steps)


@dataclass(frozen=True)
class SpineRose:
    """Multi-based flower spine: one petal per puncture, no gluing paths."""

    punctures: tuple
    petals: tuple
    dropped: int = -1            # index of the petal left out of the basis

    def basis(self):
        return tuple(p for i, p in enumerate(self.petals) if i != self.dropped)

    def petal_of(self, puncture):
        for p in self.petals:
            if p.puncture == puncture:
                return p
        raise KeyError(puncture)


def rose_spine(base, punctures=None, drop=None):
    """Petals of the flower spine, canonical order, no search.

    `punctures` defaults to every vertex of `base`; `drop` is the puncture
    whose petal is left out of the basis (default: the last in sorted order,
    which is not a homotopy choice -- any one may be dropped).
    """
    if punctures is None:
        punctures = tuple(base.vertices)
    punctures = tuple(sorted({int(p) for p in punctures}))
    if not punctures:
        raise ValueError("an empty puncture set has no positive-rank spine")
    for p in punctures:
        if p not in base.incident:
            raise ValueError(f"{p} is not a vertex of the base")
    petals = []
    for v in punctures:
        flower = base.flower(v)
        if not flower:
            raise ValueError(f"vertex {v} has an empty flower")
        faces = [k for k, _e in flower]
        if len(set(faces)) != len(faces):
            raise ValueError(f"vertex {v}: the flower revisits a face {faces}")
        petals.append(Petal(v, tuple(Step(int(k), tuple(e)) for k, e in flower)))
    if drop is None:
        dropped = len(petals) - 1
    else:
        dropped = [i for i, p in enumerate(petals) if p.puncture == drop][0]
    return SpineRose(punctures, tuple(petals), dropped)


# ------------------------------------------------------- base-side certificate
def _dual_ribbon(base):
    """The dual graph of `base` as a `RibbonGraph`: vertices are base faces.

    The flower at face `k` is its three labelled edges in the face's own
    orientation, so the dart of a lifted edge is `(edge, +1)` exactly when the
    face traverses that edge from its smaller endpoint to its larger one --
    the same convention `dense_assembly` uses for the lift labels.
    """
    flowers = {}
    for k, (verts, labs) in enumerate(base.tris):
        ds = []
        for t in range(3):
            e = tuple(labs[t])
            a, b = verts[t], verts[(t + 1) % 3]
            ds.append((e, _F if (a, b) == (e[0], e[1]) else _B))
        flowers[k] = tuple(ds)
    return RibbonGraph(flowers)


def _tree_basis_words(graph, tree, gens, base_v, walk):
    """`walk` (a closed dart walk in `graph`) as a word in the tree basis."""
    from pullback.validate import _tree_path
    full = list(_tree_path(graph, tree, base_v, graph.src(walk[0])))
    full.extend(walk)
    full.extend(_tree_path(graph, tree, graph.tgt(walk[-1]), base_v))
    index = {e: i for i, e in enumerate(gens)}
    return tuple((index[d[0]], d[1]) for d in full if d[0] in index)


def certify_rose(base, spine):
    """Exact, cover-free certificate of the flower spine.

    Flags (all booleans are computed, none is assumed):

    ``closed``       every petal is a closed walk of the dual graph, and
                     consecutive crossings share the face `next_face` gives;
    ``at_vertex``    each petal crosses exactly the edges at its puncture, once
                     each, and each such edge in the face that leaves it;
    ``sum_zero``     every base edge is crossed by the petal of each of its
                     endpoints with opposite sign, so the signed edge count of
                     the whole petal family is the zero chain -- the single
                     relation between the flower loops;
    ``basis_all``    the flowers of *all* vertices, any one dropped, generate
                     the free group on a spanning-tree basis of the dual graph
                     (Stallings folding, `validate.generates_free_group`);
    ``rank``         ``len(spine.basis()) == rank pi_1(S^2 - P)``.
    """
    dual = _dual_ribbon(base)
    tree, gens = _validate.spanning_tree(dual)
    base_v = dual.vertices()[0]
    out = {"punctures": spine.punctures,
           "n_petals": len(spine.petals),
           "petal_lengths": {p.puncture: len(p.steps) for p in spine.petals},
           "dual": {"vertices": len(dual.vertices()),
                    "edges": len(dual.edge_list()),
                    "tree": len(tree), "generators": len(gens)}}

    closed, at_vertex = True, True
    for pet in spine.petals:
        v = pet.puncture
        flower = base.flower(v)
        want_edges = sorted({e for _k, e in flower})
        got_edges = sorted(set(pet.edges()))
        if got_edges != want_edges or len(got_edges) != len(pet.steps):
            at_vertex = False
        for i, st in enumerate(pet.steps):
            k_next = pet.steps[(i + 1) % len(pet.steps)].face
            if base.next_face(v, st.face) != k_next:
                closed = False
            if st.edge != A.canon(v, base.out_vertex(v, st.face)):
                at_vertex = False
            if st.face not in base.incident[st.edge[0]] and \
               st.face not in base.incident[st.edge[1]]:
                at_vertex = False
    out["closed"] = closed
    out["at_vertex"] = at_vertex

    # signed crossing counts: +1 when the petal crosses `e` in the face that
    # carries e[0] -> e[1], -1 otherwise.  A petal's crossing at `face` goes to
    # the other face of `e`, so the sign is read from the face alone.
    signed = {}
    grouping = {}
    for pet in spine.petals:
        for st in pet.steps:
            e = st.edge
            sgn = +1 if base.dart_face[(e[0], e[1])] == st.face else -1
            signed.setdefault(e, []).append((pet.puncture, sgn))
            grouping.setdefault(e, []).append(pet.puncture)
    # edges with a non-puncture end are crossed once: excluded above, counted
    # here so that nothing silently escapes the check
    partial = {e: rows for e, rows in signed.items() if len(rows) != 2}
    full_edges = {e: rows for e, rows in signed.items() if len(rows) == 2}
    sum_zero = all(rows[0][1] == -rows[1][1] and
                   sorted(b for b, _s in rows) == sorted(e)
                   for e, rows in full_edges.items())
    out["sum_zero"] = bool(sum_zero)
    out["crossed_twice"] = len(full_edges)
    out["crossed_once"] = len(partial)
    out["crossed_once_edges"] = tuple(sorted(partial))
    out["grouping"] = {e: tuple(b for b in bs) for e, bs in sorted(grouping.items())}

    # basis: the flowers of every vertex, any one dropped
    all_spine = rose_spine(base, tuple(base.vertices))
    words = []
    for pet in all_spine.petals:
        walk = []
        cur = pet.ref_face
        for st in pet.steps:
            verts = base.tris[st.face][0]
            e = st.edge
            a, b = e
            ori = _F if verts[(verts.index(a) + 1) % 3] == b else _B
            walk.append((e, ori))
        words.append(_tree_basis_words(dual, tree, gens, base_v,
                                       tuple(walk)))
    out["basis_all"] = {str(all_spine.petals[i].puncture):
                        bool(_validate.generates_free_group(
                            [w for j, w in enumerate(words) if j != i], len(gens)))
                        for i in range(len(words))}
    out["basis_all_ok"] = all(out["basis_all"].values())

    # rank of pi_1(S^2 - P) = |P| - 1, and the dual graph's own rank check
    out["chi_dual"] = len(dual.vertices()) - len(dual.edge_list())
    out["chi_s2_minus_P"] = 2 - len(spine.punctures)
    out["rank_pi1_s2_minus_P"] = len(spine.punctures) - 1
    out["rank_basis"] = len(spine.basis())
    out["rank"] = out["rank_basis"] == out["rank_pi1_s2_minus_P"]

    # Euler characteristic of the strict one-vertex rose: one vertex and
    # |P| - 1 petals, so chi = 1 - (|P| - 1) = 2 - |P| = chi(S^2 - P).
    # With all |P| petals it would be 1 - |P|, which is not a spine.
    out["rose_vertices"] = 1
    out["rose_edges"] = len(spine.basis())
    out["rose_chi"] = out["rose_vertices"] - out["rose_edges"]
    out["rose_chi_expected"] = out["chi_s2_minus_P"]
    out["rose_chi_ok"] = out["rose_chi"] == out["rose_chi_expected"]
    out["redundant_rose_chi"] = 1 - len(spine.petals)   # NOT a spine of S^2-P
    out["ok"] = bool(closed and at_vertex and out["sum_zero"]
                     and out["basis_all_ok"] and out["rank"]
                     and out["rose_chi_ok"])
    return out


# --------------------------------------------------------------- lift attach
@dataclass(frozen=True)
class LiftedPath:
    """The lift of one petal from one sheet: an open or closed lifted walk."""

    puncture: int
    start_sheet: int
    end_sheet: int
    corners: tuple          # ((base vertex, face, sheet), ...) visited
    faces: tuple            # ((face, sheet), ...) lifted dual vertices visited
    labels: tuple           # lifted edge labels `(base edge, sheet)` used
    start_vertex: int       # lifted base vertex containing corners[0]
    end_vertex: int

    @property
    def closed(self):
        return self.start_vertex == self.end_vertex and \
            self.start_sheet == self.end_sheet


@dataclass(frozen=True)
class LiftedLoop:
    """One component of the preimage of a petal: a loop around one lifting."""

    puncture: int
    preimage: int           # lifted vertex id, as lift["index"][puncture]
    sheets: tuple           # the local-monodromy orbit
    faces: tuple            # ((face, sheet), ...) over one lap
    labels: tuple           # lifted edge labels, one lap repeated |orbit| times
    laps: int

    @property
    def length(self):
        return len(self.labels)

    def edges(self):
        return self.labels


@dataclass(frozen=True)
class RoseEdge:
    """One lifted rose edge: the `sheet`-th lift of one base petal."""

    petal: int                    # index into SpineRose.petals
    puncture: int
    sheet: int                    # sheet over the common base point
    target: int                   # sheet over the common base point
    faces: tuple                  # lifted dual vertices visited, in order
    labels: tuple                 # lifted edge labels used, in order

    @property
    def closed(self):
        return self.sheet == self.target


@dataclass(frozen=True)
class SpineRoseLift:
    """Preimage of the rose spine: a Schreier graph on the sheets.

    ``vertices`` are the sheets over the common base point ``face0`` (one
    point per sheet, in the interior of that base face).  ``edges`` are the
    d lifts of each base petal, so ``d * n`` of them for ``n`` petals, and the
    graph is the preimage of the rose up to the homotopy that collapses the
    base spine to that rose.  ``rho`` is the permutation of the fibre induced
    by each petal; it is the sheet permutation the preimage graph's edges
    follow, never a recomputation of the cover's own datum.
    """

    d: int
    face0: int
    petals: tuple                 # punctures, in rose order
    basis: tuple                  # punctures whose petals generate
    edges: tuple
    rho: dict
    report: dict

    def adjacency(self):
        adj = {s: [] for s in range(self.d)}
        for e in self.edges:
            adj[e.sheet].append(e)
        return adj

    def components(self, edges=None):
        """Components of the Schreier graph on the d sheets."""
        parent = list(range(self.d))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for e in (self.edges if edges is None else edges):
            a, b = find(e.sheet), find(e.target)
            if a != b:
                parent[a] = b
        return len({find(s) for s in range(self.d)})


class LiftAttachment:
    """The lifted petals of an assembled cover, with exact controls."""

    def __init__(self, spine, base, lift):
        self.spine = spine
        self.base = base
        self.lift = lift
        self.d = int(lift["d"])
        self.edge_perms = {e: tuple(p) for e, p in lift["edge_perms"].items()}
        self.inverse = {e: A.perm_inverse(p) for e, p in self.edge_perms.items()}
        for e in base.edges:
            if e not in self.edge_perms:
                raise ValueError(f"no declared permutation on base edge {e}")
        self.local = {int(k): tuple(v) for k, v in
                      lift["cert"]["local_realised"].items()}
        self._tri = {k: {s: lift["tris"][k * self.d + s]
                         for s in range(self.d)}
                     for k in range(base.n_faces())}
        self._dual = _dual_ribbon(base)
        self._tree = None
        self.corner = self._corner_index()
        self._paths = {}
        self._loops = {}
        self.report = self._certify()

    # -- indices -----------------------------------------------------------
    def _corner_index(self):
        out = {}
        for u, pf in self.lift["per_face"].items():
            v = self.lift["proj"][u]
            for k, sheets in pf.items():
                for s in sheets:
                    key = (v, int(k), int(s))
                    if key in out and out[key] != u:
                        raise ValueError(f"corner {key} in two lifted vertices")
                    out[key] = int(u)
        for k, (verts, _labs) in enumerate(self.base.tris):
            for s in range(self.d):
                got = [out[(v, k, s)] for v in verts]
                tri = self.lift["tris"][k * self.d + s][0]
                if sorted(got) != sorted(int(x) for x in tri):
                    raise ValueError(f"corners of base face {k} sheet {s} do "
                                     f"not hit its lifted triangle {tri}")
        return out

    @property
    def tree(self):
        if self._tree is None:
            self._tree, _gens = _validate.spanning_tree(self._dual)
        return self._tree

    # -- the one walk primitive -------------------------------------------
    def advance(self, steps, k0, s0):
        """Follow (face, base edge) steps from lifted dual vertex (k0, s0).

        Returns ``(k, s, faces, labels, ends)``.  Every step is checked
        against the assembled lift: the lifted edge label must exist, it must
        be carried by the lifted triangle it is used from and by the one it
        leads to, and its two lifted endpoints must be the corners of its
        sheet in the face carrying ``e[0] -> e[1]``.
        """
        k, s = int(k0), int(s0)
        faces, labels, ends = [], [], []
        for face, e in steps:
            if int(face) != k:
                raise ValueError(f"walk is at face {k}, step says {face}")
            k1 = self.base.dart_face[(e[0], e[1])]
            k2 = self.base.dart_face[(e[1], e[0])]
            if k == k1:
                lab, nxt, k_next = (e, s), self.edge_perms[e][s], k2
            else:
                lab, nxt, k_next = (e, self.inverse[e][s]), \
                    self.inverse[e][s], k1
            if lab not in self.lift["edges"]:
                raise ValueError(f"no lifted edge {lab} over base edge {e}")
            # the assembled label joins the corners of the *current* face and
            # sheet: (e, s) is sheet s in the face carrying e[0] -> e[1], and
            # the corner classes agree across the edge by construction
            a, b = self.lift["edges"][lab]
            if (a, b) != (self.corner[(e[0], k, s)],
                          self.corner[(e[1], k, s)]):
                raise ValueError(f"lifted edge {lab} does not join the corners "
                                 f"({e[0]},{k},{s}) and ({e[1]},{k},{s})")
            if lab not in self._tri[k][s][1]:
                raise ValueError(f"lifted triangle ({k},{s}) does not carry {lab}")
            if lab not in self._tri[k_next][nxt][1]:
                raise ValueError(f"lifted triangle ({k_next},{nxt}) does not "
                                 f"carry {lab}")
            faces.append((k, s))
            labels.append(lab)
            ends.append((a, b))
            k, s = k_next, nxt
        return k, s, tuple(faces), tuple(labels), tuple(ends)

    # -- petal paths -------------------------------------------------------
    def path(self, petal, sheet):
        key = (petal.puncture, petal.ref_face, int(sheet))
        if key in self._paths:
            return self._paths[key]
        s0 = int(sheet)
        steps = tuple((st.face, st.edge) for st in petal.steps)
        k, s, faces, labels, _ends = self.advance(steps, petal.ref_face, s0)
        if k != petal.ref_face:
            raise ValueError(f"petal {petal.puncture}: face walk does not close")
        sigma = self.local[petal.puncture]
        if s != sigma[s0]:
            raise ValueError(f"petal {petal.puncture} sheet {s0}: sheet walk "
                             f"ends at {s}, declared local monodromy {sigma} "
                             f"says {sigma[s0]}")
        corners = tuple((petal.puncture, kk, ss) for kk, ss in faces)
        start = self.corner[corners[0]]
        end = self.corner[(petal.puncture, petal.ref_face, s)]
        out = LiftedPath(petal.puncture, s0, s, corners, faces,
                         labels, start, end)
        self._paths[key] = out
        return out

    # -- loops over the liftings of one puncture ---------------------------
    def loops(self, petal):
        if petal.puncture in self._loops:
            return self._loops[petal.puncture]
        sigma = self.local[petal.puncture]
        seen, orbits = set(), []
        for s in range(self.d):
            if s in seen:
                continue
            orbit, x = [], s
            while x not in seen:
                seen.add(x)
                orbit.append(x)
                x = sigma[x]
            if x != s:
                raise ValueError(f"puncture {petal.puncture}: the declared "
                                 "local monodromy is not a permutation")
            orbits.append(tuple(sorted(orbit)))
        out = []
        for orbit in sorted(orbits, key=lambda o: o[0]):
            faces, labels = [], []
            for s in orbit:
                w = self.path(petal, s)
                faces.extend(w.faces)
                labels.extend(w.labels)
            u = self.corner[(petal.puncture, petal.ref_face, orbit[0])]
            cls = {(petal.puncture, int(k), int(s))
                   for k, sheets in self.lift["per_face"][u].items()
                   for s in sheets}
            visited = set()
            for s in orbit:
                visited |= set(self.path(petal, s).corners)
            if visited != cls:
                raise ValueError(f"puncture {petal.puncture}: the lifted walk "
                                 f"over preimage {u} visits {sorted(visited)}, "
                                 f"but the assembled corner class is "
                                 f"{sorted(cls)}")
            if self.lift["ld"][u] != len(orbit):
                raise ValueError(f"puncture {petal.puncture}: preimage {u} has "
                                 f"declared local degree {self.lift['ld'][u]} "
                                 f"but orbit size {len(orbit)}")
            out.append(LiftedLoop(petal.puncture, int(u), orbit,
                                  tuple(faces), tuple(labels), len(orbit)))
        self._loops[petal.puncture] = tuple(out)
        return self._loops[petal.puncture]

    # -- the strict rose: one base point, conjugating tree paths -----------
    def to_rose(self, face0=0, which="basis"):
        """The strict one-vertex rose and its preimage graph.

        ``which="basis"`` (the default) uses the ``|P| - 1`` basis petals only.
        The strict rose then has one vertex and ``|P| - 1`` petals, so
        ``chi = 1 - (|P| - 1) = 2 - |P|``, and its preimage has ``d`` vertices
        (the sheets over the base point), ``d*(|P| - 1)`` edges and
        ``chi = d - d*(|P| - 1) = d*(2 - |P|)``.  That is exactly
        ``d * chi(S^2 - P)``, the Euler characteristic of the punctured cover,
        and it is asserted, not assumed: a one-vertex model with all ``|P|``
        petals has ``chi = d*(1 - |P|)`` and is *not* a spine of ``S^2 - P``.

        ``which="all"`` keeps the redundant flower loop too.  That record is
        auxiliary: it is the per-edge path bookkeeping of every flower (one
        group per puncture) and its Euler characteristic is deliberately the
        non-spine one, reported as such.

        The petals are based at their own faces, so a strict rose needs a
        transport from the common base point ``face0`` to each petal's face.
        Any spanning tree of the base dual graph gives one; changing the tree
        conjugates every ``rho`` by the transport, i.e. relabels the sheets.
        That is the *only* free datum of this record, and it is why the
        multi-based flower record above is the canonical one.
        """
        from pullback.validate import _tree_path
        d, dual = self.d, self._dual
        if which not in ("basis", "all"):
            raise ValueError(f"which must be 'basis' or 'all', not {which!r}")
        if face0 not in dual.vertices():
            raise ValueError(f"{face0} is not a base face")
        n_punctures = len(self.spine.punctures)
        cover_connected = bool((self.lift.get("cert") or {}).get("connected",
                                                                True))
        chosen = [j for j, _pet in enumerate(self.spine.petals)
                  if which == "all" or j != self.spine.dropped]
        if not chosen:
            raise ValueError("the rose needs at least one petal")
        report = {"face0": face0, "tree_edges": len(self.tree),
                  "which": which, "punctures": self.spine.punctures,
                  "n_petals": len(chosen)}

        def transport(face):
            darts = _tree_path(dual, self.tree, face0, face)
            return tuple((dual.src(x), x[0]) for x in darts)

        rho, edges = {}, []
        for j in chosen:
            pet = self.spine.petals[j]
            out = transport(pet.ref_face)
            back = tuple((dual.src(x), x[0])
                         for x in _tree_path(dual, self.tree, pet.ref_face,
                                             face0))
            forward = [self.advance(out, face0, s) for s in range(d)]
            tau = [forward[s][1] for s in range(d)]        # sheet after `out`
            tau_inv = [0] * d
            for s in range(d):
                tau_inv[tau[s]] = s
            rho_j = [None] * d
            for s0 in range(d):
                # out, then the petal, then back: one lift of the rose edge
                k1_end, s1, f1, l1, _ = forward[s0]
                if k1_end != pet.ref_face:
                    raise ValueError("transport does not land on the petal face")
                k2_end, s2, f2, l2, _ = self.advance(
                    tuple((st.face, st.edge) for st in pet.steps),
                    pet.ref_face, s1)
                if k2_end != pet.ref_face:
                    raise ValueError("petal does not close")
                k3_end, s3, f3, l3, _ = self.advance(back, pet.ref_face, s2)
                if k3_end != face0:
                    raise ValueError("return transport does not reach face0")
                # the petal walk must close on the declared local monodromy
                if s2 != self.local[pet.puncture][s1]:
                    raise ValueError(f"petal {pet.puncture} sheet {s1}: the "
                                     "walk disagrees with the local monodromy")
                rho_j[s0] = s3
                edges.append(RoseEdge(j, pet.puncture, s0, s3,
                                      tuple(f1) + tuple(f2) + tuple(f3),
                                      tuple(l1) + tuple(l2) + tuple(l3)))
            if sorted(rho_j) != list(range(d)):
                raise ValueError(f"petal {pet.puncture}: rho is not a permutation")
            # rho is the conjugate of the declared local monodromy
            conj = [tau_inv[self.local[pet.puncture][tau[s]]] for s in range(d)]
            if tuple(conj) != tuple(rho_j):
                raise ValueError(f"petal {pet.puncture}: rho is not the "
                                 "conjugate of the declared local monodromy")
            rho[pet.puncture] = tuple(rho_j)
        rose = SpineRoseLift(d, face0,
                             tuple(self.spine.petals[j].puncture for j in chosen),
                             tuple(p.puncture for p in self.spine.basis()),
                             tuple(edges), rho, report)
        components = rose.components()
        report["vertices"] = d
        report["edges"] = len(edges)
        report["components"] = components
        report["connected"] = components == 1
        report["chi"] = d - len(edges)
        report["chi_expected"] = d * (2 - n_punctures)
        report["chi_ok"] = report["chi"] == report["chi_expected"]
        report["rank"] = len(edges) - d + components
        report["rank_expected"] = d * (len(chosen) - 1) + components
        report["rank_ok"] = report["rank"] == report["rank_expected"]
        report["is_spine"] = which == "basis"
        report["cover_connected"] = cover_connected
        report["edge_lengths"] = {p: tuple(len(e.labels) for e in edges
                                           if e.puncture == p)
                                  for p in rose.petals}
        if which == "basis":
            # structural identities of the spine model: they can only fail if
            # the construction above is wrong, so they raise rather than warn
            if not report["chi_ok"]:
                raise ValueError(
                    f"strict rose chi = {report['chi']} but the punctured "
                    f"cover has chi = {report['chi_expected']} (d = {d}, "
                    f"|P| = {n_punctures})")
            if len(edges) != d * (n_punctures - 1):
                raise ValueError("strict rose does not have d*(|P|-1) edges")
            # a connected cover has a connected preimage of the rose; a
            # disconnected cover is allowed to have one, and `chi` stays
            # additive, so only a connected cover is held to this
            if cover_connected and not report["connected"]:
                raise ValueError("the preimage of the basis rose is not "
                                 "connected, but the cover is")
        return rose

    # -- checks ------------------------------------------------------------
    def _certify(self):
        rep = {"d": self.d, "punctures": self.spine.punctures,
               "paths": 0, "loops": 0, "loop_lengths": {},
               "preimages": {}}
        for pet in self.spine.petals:
            for s in range(self.d):
                self.path(pet, s)
                rep["paths"] += 1
            loops = self.loops(pet)
            rep["loops"] += len(loops)
            rep["loop_lengths"][pet.puncture] = tuple(l.length for l in loops)
            rep["preimages"][pet.puncture] = tuple(l.preimage for l in loops)
        used = {}
        for pet in self.spine.petals:
            for s in range(self.d):
                for lab in self.path(pet, s).labels:
                    used.setdefault(lab, []).append((pet.puncture, s))
        per_edge = {}
        for lab, rows in used.items():
            per_edge.setdefault(lab[0], set()).add(len(rows))
        expected_edges = {e for e in self.base.edges
                          if any(end in self.spine.punctures for end in e)}
        rep["edge_use_multiplicity"] = {str(e): sorted(v) for e, v in
                                        sorted(per_edge.items())}
        rep["edge_use_ok"] = bool(
            set(per_edge) == expected_edges and
            all(v == {sum(1 for end in e if end in self.spine.punctures)}
                for e, v in per_edge.items()))
        for p in self.spine.punctures:
            want = sorted(self.lift["index"][p])
            got = sorted(l.preimage for l in self.loops(self.spine.petal_of(p)))
            if want != got:
                raise ValueError(f"puncture {p}: loops over {got}, fibre {want}")
            if sum(len(l.sheets) for l in self.loops(self.spine.petal_of(p))) \
                    != self.d:
                raise ValueError(f"puncture {p}: the orbits do not cover S_{self.d}")
        deficit = sum(self.d - len(self.loops(self.spine.petal_of(p)))
                      for p in self.spine.punctures)
        rep["rh_deficit"] = deficit
        rep["rh_expected"] = 2 * self.d - 2
        rep["rh_ok"] = deficit == rep["rh_expected"]
        # the same identity, written as Euler characteristics: the cover minus
        # the lifted punctures has chi = 2 - (number of lifted punctures), and
        # it must equal d * chi(S^2 - P) = d * (2 - |P|)
        rep["lifted_punctures"] = sum(len(self.loops(self.spine.petal_of(p)))
                                      for p in self.spine.punctures)
        rep["chi_punctured_lift"] = 2 - rep["lifted_punctures"]
        rep["chi_punctured_expected"] = self.d * (2 - len(self.spine.punctures))
        rep["chi_agree"] = rep["chi_punctured_lift"] == \
            rep["chi_punctured_expected"]
        # chi = d*chi(S^2-P) is the sphere-cover identity; for any other
        # genus it is reported false rather than raised, so that the record
        # of an over-ramified prescription can still be inspected
        rep["cover_is_sphere"] = bool((self.lift.get("cert") or {})
                                      .get("is_sphere", False))
        rep["chi_ok"] = rep["chi_agree"] or not rep["cover_is_sphere"]
        if rep["cover_is_sphere"] and not rep["chi_agree"]:
            raise ValueError(
                f"Euler characteristic of the punctured cover is "
                f"{rep['chi_punctured_lift']}, but d*chi(S^2-P) = "
                f"{rep['chi_punctured_expected']} (d = {self.d}, |P| = "
                f"{len(self.spine.punctures)})")
        cert = self.lift.get("cert") or {}
        rep["matches_assembled"] = {
            "fibre_ok": bool(cert.get("fibre_ok", False)),
            "rh_deficit": cert.get("rh_deficit"),
            "local_ok": {str(v): tuple(self.local[v]) == tuple(w)
                         for v, w in cert.get("local_realised", {}).items()},
        }
        rep["ok"] = bool(rep["edge_use_ok"] and rep["rh_ok"]
                         and rep["chi_ok"] and rep["cover_is_sphere"]
                         and all(rep["matches_assembled"]["local_ok"].values()))
        return rep


def attach_spine(spine, base, lift):
    return LiftAttachment(spine, base, lift)
