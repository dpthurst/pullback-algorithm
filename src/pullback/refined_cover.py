"""Deterministic, ribbon-native refinement of a VE's frozen lifted cover.

Derived from corner slots only: a labelled triangle ``(v0,v1,v2)`` with
``labs[t]`` carried by ``v_t -> v_{t+1}``, the rotation system of the base
triangulation, and the Z/2 branch datum.  No census of edge words, sheet
assignments, transports, flips or spanning trees appears anywhere below.

Objects
-------
``T``  base triangulation of the base points.  Its vertices are the punctures
       (plus the auxiliary point ``m`` after the split).
``K``  the lifted triangulation ``pi^{-1}(T)``: the complex whose *dual* is the
       trunk's ``VEInit.cells``.  Vertices carry a projection and a local
       degree.
``S``  the spine ``dual(T)``; a ``RibbonGraph`` whose faces are the base points
       and whose vertices are the triangles of ``T``.
``P``  ``dual(K)``, the trunk's lifted spine.

What the refinement uses
------------------------
The trunk's ``phi' : P -> S`` is a *graph map*, and as a graph map it is only
defined up to homotopy: measured on the rabbit, 7 of its 12 edge images are the
trivial chain.  Its homotopy-invariant content is the **portrait**: for every
lifted vertex (= face of ``P``) the base point it sits over, its local degree,
and the base point its face walk tightens to.  That portrait is derived below
from ``(phi, cover, base)`` plus the marking -- no transport section, no graph
map, hence no search -- and the refinement acts on it exactly.
"""

from __future__ import annotations

import numpy as np

from pullback.ribbons import B, F, RibbonGraph, tighten

__all__ = [
    "canon", "tri_edges", "rotation_system", "tri_graph", "dual_of",
    "lift_triangulation", "star_split", "split_lifted", "subdivide_1to4",
    "face_rows_pi", "face_rows_phi", "portrait", "lifted_labels_from_portrait",
    "ribbon_check", "cover_check", "collapse_map", "base_angles",
    "inherited_metric", "conformal_seed",
]


# ---------------------------------------------------------------- triangles
def canon(i, j):
    return (i, j) if i < j else (j, i)


def tri_edges(tris):
    """``{label: (i, j)}`` for a labelled triangulation.

    Each entry is ``(verts, labs)`` or ``(verts, labs, base triangle)``.
    """
    edges = {}
    for tr in tris:
        verts, labs = tr[0], tr[1]
        for t in range(3):
            a, b = verts[t], verts[(t + 1) % 3]
            lab = labs[t]
            pair = canon(a, b)
            if lab in edges and canon(*edges[lab]) != pair:
                raise ValueError(f"label {lab!r} joins {edges[lab]} and {pair}")
            edges.setdefault(lab, (a, b))
    for lab in edges:
        uses = sum(1 for tr in tris for l in tr[1] if l == lab)
        if uses != 2:
            raise ValueError(f"label {lab!r} used {uses} times, not 2")
    return edges


def _dart_at(edges, lab, v):
    a, b = edges[lab]
    if a == v:
        return (lab, F)
    if b == v:
        return (lab, B)
    raise ValueError(f"{lab!r} is not incident with {v}")


def rotation_system(tris, edges=None):
    """``{vertex: [dart, ...]}``: at corner ``(a,b,c)``, ``a->c`` precedes ``a->b``."""
    if edges is None:
        edges = tri_edges(tris)
    succ, at = {}, {}
    for tr in tris:
        verts, labs = tr[0], tr[1]
        for t in range(3):
            a = verts[t]
            d_in = _dart_at(edges, labs[(t + 2) % 3], a)
            d_out = _dart_at(edges, labs[t], a)
            if d_in in succ and succ[d_in] != d_out:
                raise ValueError(f"rotation conflict at {d_in!r}")
            succ[d_in] = d_out
            at.setdefault(a, set()).update((d_in, d_out))
    flowers = {}
    for v, ds in at.items():
        start = min(ds, key=str)
        cyc, d = [start], succ[start]
        while d != start:
            cyc.append(d)
            d = succ[d]
            if len(cyc) > len(ds):
                raise ValueError(f"vertex {v}: rotation does not close")
        if len(cyc) != len(ds):
            raise ValueError(f"vertex {v}: rotation is not a single cycle")
        flowers[v] = cyc
    return flowers


def tri_graph(tris, edges=None):
    """The ribbon graph of a triangulation; its faces are the triangles."""
    return RibbonGraph(rotation_system(tris, edges))


def dual_of(tris, edges=None):
    """Dual: vertices are the triangles, darts are the triangle edges.

    ``RibbonGraph.dual`` keeps the dart identities and reverses each boundary
    walk, so a dual dart ``(label, ori)`` sits at dual vertex ``t`` exactly when
    the triangle ``t`` carries the dart ``(label, ori)``.  That identity is what
    lets a graph map's chains be read as corner slots.
    """
    return tri_graph(tris, edges).dual()


def _far(dart, edges):
    a, b = edges[dart[0]]
    return (b if dart[1] == F else a)


# ------------------------------------------------------- branched lift (Z/2)
class _DSU:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def lift_triangulation(tris, branch_edges):
    """Corner-slot lift of a triangulation along the Z/2 branch datum.

    Node ``(p, k, s)`` is vertex ``p`` seen from triangle ``k`` on relative sheet
    ``s``; nodes are identified across a shared edge with the sheet flipped by
    ``phi(e) = [e in branch_edges]``.  No global sheet exists unless the datum is
    empty, which is why this is a corner-slot construction and not two copies.

    Returns the lifted labelled triangulation with the projection, fibre index,
    per-vertex monodromy ``eps``, per-vertex relative sheets, and the base
    triangle of every lifted triangle.
    """
    tris = [(tuple(tr[0]), tuple(tr[1])) for tr in tris]
    Fs = {canon(*e) for e in branch_edges}
    edges = tri_edges(tris)
    base_edges = {canon(*p) for p in edges.values()}
    if not Fs <= base_edges:
        raise ValueError(f"branch datum uses non-edges {sorted(Fs - base_edges)}")
    flip = {e: (1 if e in Fs else 0) for e in base_edges}

    eps = {v: 0 for v in {x for verts, _l in tris for x in verts}}
    for (i, j) in Fs:
        eps[i] ^= 1
        eps[j] ^= 1

    sides = {}
    for k, (verts, _labs) in enumerate(tris):
        for t in range(3):
            sides.setdefault(canon(verts[t], verts[(t + 1) % 3]), []).append(k)
    for e, ks in sides.items():
        if len(ks) != 2:
            raise ValueError(f"base edge {e} is in {len(ks)} triangles")

    dsu = _DSU()
    for k, (verts, _l) in enumerate(tris):
        for v in verts:
            for s in (0, 1):
                dsu.find((v, k, s))
    for e, (k1, k2) in sides.items():
        for v in e:
            for s in (0, 1):
                dsu.union((v, k1, s), (v, k2, s ^ flip[e]))

    nodes = {(v, k, s): dsu.find((v, k, s))
             for k, (verts, _l) in enumerate(tris) for v in verts for s in (0, 1)}
    classes = {}
    for node, root in nodes.items():
        classes.setdefault(root, set()).add(node[0])
    for root, vs in classes.items():
        if len(vs) != 1:
            raise ValueError(f"lifted vertex over several base points {vs}")
    first = {}
    for node, root in nodes.items():
        if root not in first or node < first[root]:
            first[root] = node
    order = sorted(classes, key=lambda r: (next(iter(classes[r])), first[r]))
    ren = {r: i for i, r in enumerate(order)}
    proj = {ren[r]: next(iter(classes[r])) for r in classes}
    sheets = {}
    for node, r in nodes.items():
        sheets.setdefault(ren[r], set()).add(node[2])
    fibres = {}
    for u, p in proj.items():
        fibres.setdefault(p, []).append(u)
    index = {p: 2 // len(fs) for p, fs in fibres.items()}

    def lab(e, k, s):
        k0 = sides[e][0]
        return (e, s if k == k0 else s ^ flip[e])

    out_edges, out_tris = {}, []
    for k, (verts, _labs) in enumerate(tris):
        for s in (0, 1):
            vs = tuple(ren[nodes[(verts[t], k, s)]] for t in range(3))
            if len(set(vs)) != 3:
                raise ValueError(f"lifted triangle over {verts} degenerate: {vs}")
            labs = []
            for t in range(3):
                e = canon(verts[t], verts[(t + 1) % 3])
                a, b = vs[t], vs[(t + 1) % 3]
                if a == b:
                    raise ValueError(f"lifted edge over {e} is a loop")
                l = lab(e, k, s)
                if l in out_edges and canon(*out_edges[l]) != canon(a, b):
                    raise ValueError(f"lifted edge {l!r} has two end pairs")
                out_edges[l] = (a, b)
                labs.append(l)
            out_tris.append((vs, tuple(labs), k))
    return {"edges": out_edges, "tris": out_tris, "proj": proj,
            "index": index, "fibres": fibres, "eps": eps, "phi": flip,
            "sheets": sheets, "base_tri": {i: t[2] for i, t in enumerate(out_tris)}}


# ------------------------------------------------------------------ splits
def star_split(tris, e, m):
    """Stellar split of a labelled triangulation along the edge ``e``.

    The edge becomes its two halves through the new vertex ``m`` and each
    adjacent triangle is cut by the chord from ``m`` to its opposite vertex.
    Returns the refined triangulation and the parent maps (``edge_parent`` gives
    the coarse edge a refined edge lies over, ``None`` for a chord).  Nothing is
    chosen: the new labels are functions of the vertex labels.
    """
    e = canon(*e)
    hits = [(k, t) for k, (verts, _l) in enumerate(tris) for t in range(3)
            if canon(verts[t], verts[(t + 1) % 3]) == e]
    if len(hits) != 2:
        raise ValueError(f"edge {e} is in {len(hits)} triangles, not 2")
    split_of = dict(hits)
    edge_parent, vert_parent = {}, {}
    new_tris = []
    for k, (verts, labs) in enumerate(tris):
        if k not in split_of:
            new_tris.append((tuple(verts), tuple(labs)))
            for l in labs:
                edge_parent[l] = l
            continue
        t = split_of[k]
        a, b, r = verts[t], verts[(t + 1) % 3], verts[(t + 2) % 3]
        la, lb, lr = labs[t], labs[(t + 1) % 3], labs[(t + 2) % 3]
        if canon(a, b) != e:
            raise ValueError("internal: wrong corner for the split edge")
        h_a, h_b, c_r = canon(a, m), canon(m, b), canon(m, r)
        edge_parent[h_a] = la
        edge_parent[h_b] = lb
        edge_parent[c_r] = None
        edge_parent[lr] = lr
        new_tris.append(((a, m, r), (h_a, c_r, lr)))
        new_tris.append(((m, b, r), (h_b, lb, c_r)))
    for v in {x for verts, _l in tris for x in verts}:
        vert_parent[v] = v
    vert_parent[m] = None
    return {"tris": new_tris, "edges": tri_edges(new_tris),
            "edge_parent": edge_parent, "vert_parent": vert_parent,
            "new_vertex": m, "split_edge": e,
            "halves": {e: (canon(e[0], m), canon(m, e[1]))}}


def split_lifted(lift, split_edge, m, ld=None):
    """Preimage subdivision of the lifted triangulation over ``split_edge``.

    Every lifted edge over the split base edge is cut at its own midpoint, a new
    vertex over ``m``; old vertices keep their indices and old edge labels.  The
    new midpoints sit over ``m`` with local degree 1, the two halves stay in the
    same lift of the edge, and the chords are labelled
    ``("chord", parent_label, opposite_vertex)``.

    The rule is **forced**: a lifted edge over the split base edge must become a
    path through a midpoint over the new base point (otherwise the refined base's
    fibre multiplicities fail), and the two halves must lie in one lift.
    """
    e = canon(*split_edge)
    edges, tris = lift["edges"], lift["tris"]
    proj = dict(lift["proj"])
    n = 1 + max(max(v) for v in edges.values())
    out_edges = dict(edges)
    out_tris = list(tris)
    base_of = {l: canon(*l[0]) for l in edges}
    splits = {}
    j = 0
    for l in sorted(edges, key=str):
        if base_of[l] != e:
            continue
        mu = n + j
        j += 1
        a, b = edges[l]
        hits = [(k, t) for k, (verts, labs, _b) in enumerate(out_tris)
                for t in range(3) if labs[t] == l]
        if len(hits) != 2:
            raise ValueError(f"lifted edge {l!r} is in {len(hits)} triangles")
        quals = {}
        for k, t in hits:
            verts, labs, _b = out_tris[k]
            if canon(verts[t], verts[(t + 1) % 3]) != canon(a, b):
                raise ValueError(f"lifted edge {l!r} traversed the wrong way")
            quals[k] = verts[(t + 2) % 3]
        if len(set(quals.values())) != 2:
            raise ValueError(f"lifted edge {l!r}: degenerate quad")
        h_a, h_b = ("half", l, a), ("half", l, b)
        proj[mu] = m
        out_edges.pop(l)
        out_edges[h_a] = (a, mu)
        out_edges[h_b] = (mu, b)
        base_of.pop(l)
        base_of[h_a] = canon(proj[a], m)
        base_of[h_b] = canon(proj[b], m)
        new = []
        for k, t in sorted(hits):
            verts, labs, bt = out_tris[k]
            a1, b1, r1 = verts[t], verts[(t + 1) % 3], verts[(t + 2) % 3]
            l1r = labs[(t + 2) % 3]
            ch = ("chord", l, r1)
            out_edges[ch] = (mu, r1)
            base_of[ch] = canon(m, proj[r1])
            new.append(((a1, mu, r1), (h_a if a1 == a else h_b, ch, l1r), bt))
            new.append(((mu, b1, r1), (h_b if b1 == b else h_a, labs[(t + 1) % 3],
                                       ch), bt))
        out_tris = [tr for k, tr in enumerate(out_tris) if k not in dict(hits)]
        out_tris.extend(new)
        splits[l] = {"midpoint": mu, "halves": (h_a, h_b)}
    if ld is None:
        fib = {}
        for u, p in lift["proj"].items():
            fib[p] = fib.get(p, 0) + 1
        vals = set(fib.values())
        if len(vals) != 2:
            raise ValueError("cannot infer local degrees from the fibres alone")
        deg = max(max(fib.values()), 2)
        ld = {u: deg // fib[lift["proj"][u]] for u in lift["proj"]}
    ld = dict(ld)
    for u in proj:
        ld.setdefault(u, 1)
    for u in proj:
        if u >= n:
            ld[u] = 1
    return {"edges": out_edges, "tris": out_tris, "proj": proj, "ld": ld,
            "base_of": base_of, "splits": splits, "old_vertex_count": n}


def subdivide_1to4(tris, base_of, proj=None, base_mid=None):
    """Uniform 1 -> 4 subdivision that keeps parent and base-edge data.

    A half of an edge lies over the same base edge as its parent, and a spoke
    lies over the base edge of the triangle edge opposite its corner, so the
    covering projection and the inherited metric commute with the subdivision by
    construction rather than by assumption.
    """
    edges = tri_edges(tris)
    n = 1 + max(x for tr in tris for x in tr[0])
    mid = {l: n + i for i, l in enumerate(sorted(edges, key=str))}
    bmid = ({canon(*edges[l]): v for l, v in mid.items()}
            if base_mid is None else base_mid)
    pv = (lambda v: v) if proj is None else (lambda v: proj[v])
    out_edges, out_tris, out_base = {}, [], {}
    for fi, tr in enumerate(tris):
        verts, labs = tr[0], tr[1]
        m = [mid[l] for l in labs]
        for k in range(3):
            v = verts[k]
            for lab, mv in ((labs[k], m[k]), (labs[k - 1], m[k - 1])):
                if ("half", lab, v) in out_edges:
                    continue
                out_edges[("half", lab, v)] = (v, mv)
                out_base[("half", lab, v)] = canon(pv(v), bmid[canon(*base_of[lab])])
            out_tris.append(((v, m[k], m[k - 1]),
                             (("half", labs[k], v), ("spoke", fi, (k - 1) % 3),
                              ("half", labs[k - 1], v)), fi))
        for k in range(3):
            lab = ("spoke", fi, k)
            out_edges[lab] = (m[k], m[(k + 1) % 3])
            out_base[lab] = canon(bmid[canon(*base_of[labs[k]])],
                                  bmid[canon(*base_of[labs[(k + 1) % 3]])])
        out_tris.append(((m[0], m[1], m[2]),
                         (("spoke", fi, 0), ("spoke", fi, 1), ("spoke", fi, 2)), fi))
    for verts, labs, _fi in out_tris:
        for t in range(3):
            if canon(*out_edges[labs[t]]) != canon(verts[t], verts[(t + 1) % 3]):
                raise ValueError(f"refined label {labs[t]!r} misaligned at {verts}")
    return {"tris": out_tris, "edges": out_edges, "base_of": out_base,
            "mid": mid, "coarse_vertices": n,
            "base_tri": {i: tr[2] for i, tr in enumerate(out_tris)}}


# ------------------------------------------------------- the portrait read-off
def _rotations(walk):
    n = len(walk)
    for i in range(n):
        yield walk[i:] + walk[:i]


def _match_face(word, base_walks):
    """``(face, degree)`` if the reduced word is a power of a base face walk."""
    if not word:
        return None
    for i, bw in enumerate(base_walks):
        k, rem = divmod(len(word), len(bw))
        if rem or k == 0:
            continue
        if any(tuple(word) == r * k for r in _rotations(bw)):
            return i, k
    return None


def face_rows_pi(cover, base):
    """Each cover face: its projection word and the base face it covers.

    An independent implementation of the portrait's covering half: a cover dart
    projects to its base label with the same orientation, and the face is matched
    against the base's own boundary walks, darts kept distinct so parallel lifted
    edges are handled honestly.
    """
    base_walks = [tuple(f) for f in base.faces()]
    rows = []
    for face in cover.faces():
        proj = tuple((d[0][0], d[1]) for d in face)
        rows.append({"face": face, "projection": proj,
                     "hit": _match_face(tuple(proj), base_walks)})
    return rows, base_walks


def face_rows_phi(phi):
    """Each face of ``phi``'s source: the tightened image of its boundary walk."""
    base_walks = [tuple(f) for f in phi.target.faces()]
    rows = []
    for face in phi.source.faces():
        raw = []
        for d in face:
            raw.extend(phi.image_dart(d))
        tight = tuple(tighten(raw))
        rows.append({"face": face, "raw": tuple(raw), "tight": tight,
                     "hit": _match_face(tight, base_walks)})
    return rows, base_walks


def portrait(cover, base, phi, puncture_of_base_face):
    """The dynamical portrait of the VE, from exact face equations only.

    Returns ``(rows, dyn, active)``.  ``rows[k]`` says, for the ``k``-th face of
    the cover: the puncture it covers, its local degree, and the puncture its
    tightened image bounds (``None`` when the image is null-homotopic: the face is
    collapsed and carries no label).  ``dyn[p] = q``: the face carrying the
    postcritical label ``p`` sits over the puncture ``q``.  ``active[p]`` is that
    face.  These are the two data the trunk reads off ``phi'``; here they come
    from the abstract VE, so they cannot depend on the refinement at all.
    """
    pi_rows, _ = face_rows_pi(cover, base)
    ph_rows, _ = face_rows_phi(phi)
    if len(pi_rows) != len(ph_rows):
        raise ValueError("pi and phi read different face lists")
    rows, dyn, active = [], {}, {}
    for k, (r, s) in enumerate(zip(pi_rows, ph_rows)):
        if r["face"] != s["face"]:
            raise ValueError("face order mismatch between the pi and phi read-offs")
        if r["hit"] is None:
            raise ValueError(f"cover face {r['face']!r} does not project to a "
                             f"base face: {r['projection']}")
        q = puncture_of_base_face[r["hit"][0]]
        if not s["tight"]:
            rows.append({"face": r["face"], "over": q, "ld": r["hit"][1],
                         "target": None})
            continue
        if s["hit"] is None:
            raise ValueError(f"face {s['face']!r} tightens to {s['tight']}, which "
                             "is not a base face walk")
        p = puncture_of_base_face[s["hit"][0]]
        if p in active:
            raise ValueError(f"two cover faces carry postcritical label {p}")
        rows.append({"face": r["face"], "over": q, "ld": r["hit"][1],
                     "target": p})
        dyn[p] = q
        active[p] = r["face"]
    return rows, dict(sorted(dyn.items())), active


def lifted_labels_from_portrait(rows, proj, ld):
    """Which lifted vertex carries which postcritical label, and its target.

    A lifted vertex over a base point is a face of ``P``; the portrait row for
    that face names the label.  The matching uses only ``(projection, local
    degree)``, so it is a read-off of the lift rather than of a stored coarse
    label table -- and it fails loudly if the profile does not agree in both
    directions.
    """
    left = {}
    for u, p in proj.items():
        left.setdefault((p, ld[u]), []).append(u)
    for k, vs in left.items():
        if len(vs) > 2:
            raise ValueError(f"too many lifts over {k}: {vs}")
        vs.sort()
    out, target, used = {}, {}, set()
    for r in rows:
        if r["target"] is None:
            continue
        key = (r["over"], r["ld"])
        if not left.get(key):
            raise ValueError(f"portrait row {r} has no lifted vertex over {key}")
        u = left[key].pop(0)
        if u in used:
            raise ValueError(f"lifted vertex {u} carries two labels")
        used.add(u)
        out[r["target"]] = u
        target[u] = r["target"]
    return dict(sorted(out.items())), target


# --------------------------------------------------------------- certificates
def ribbon_check(T, triT, tris, edges, proj, ld):
    """(R1)/(R2): every lifted flower is a rotation of the base flower, repeated.

    ``ld(u) * len(flower(base point))`` darts projecting to a cyclic rotation of
    the base flower repeated ``ld`` times.  This is the criterion that separates
    "the lift of the refined base" from "a refinement of the lift".
    """
    base_rot = rotation_system(T, triT)
    lift_rot = rotation_system(tris, edges)
    rows, bad = [], []
    for u in sorted(lift_rot, key=str):
        p = proj[u]
        bw = [_far_vertex(d, triT) for d in base_rot[p]]
        lw = [_far_vertex(d, edges) for d in lift_rot[u]]
        lw = [proj[v] for v in lw]
        e = ld[u]
        ok = len(lw) == e * len(bw) and any(
            lw == (bw[r:] + bw[:r]) * e for r in range(len(bw)))
        rows.append({"vertex": u, "over": p, "ld": e, "ok": bool(ok),
                     "degree": len(lw), "required": e * len(bw)})
        if not ok:
            bad.append(rows[-1])
    return {"ok": not bad, "rows": rows, "bad": bad}


def _far_vertex(dart, edges):
    a, b = edges[dart[0]]
    return b if dart[1] == F else a


def cover_check(T, triT, tris, edges, proj, base_of, degree=2):
    """Is this labelled complex a cover of ``T`` with the right data?

    Per base edge and per base triangle counts, no phantom labels, projection
    consistent with the base label of every lifted edge.  Exact, no geometry.
    """
    base_edges = {canon(*p) for p in triT.values()}
    per_edge, phantom = {}, []
    for l, (i, j) in edges.items():
        b = base_of[l]
        if b not in base_edges:
            phantom.append(l)
        per_edge[b] = per_edge.get(b, 0) + 1
        if canon(proj[i], proj[j]) != canon(*b):
            raise ValueError(f"lifted edge {l!r} joins projections "
                             f"{canon(proj[i], proj[j])} but is labelled {b}")
    base_tris = {frozenset(tr[0]) for tr in T}
    per_tri = {}
    for tr in tris:
        verts = tr[0]
        img = frozenset(proj[v] for v in verts)
        if len(img) != 3 or img not in base_tris:
            raise ValueError(f"lifted triangle {verts} projects to {sorted(img)}")
        per_tri[img] = per_tri.get(img, 0) + 1
    ok_edges = (not phantom) and all(per_edge.get(b, 0) == degree for b in base_edges)
    ok_tris = all(per_tri.get(f, 0) == degree for f in base_tris) and set(per_tri) <= base_tris
    return {"ok": bool(ok_edges and ok_tris), "edge_counts": bool(ok_edges),
            "face_counts": bool(ok_tris), "no_phantom_labels": not phantom,
            "phantom": phantom,
            "missing_edges": sorted(str(b) for b in base_edges
                                    if per_edge.get(b, 0) != degree),
            "edge_multiplicities": sorted(set(per_edge.values())),
            "face_multiplicities": sorted(set(per_tri.values())),
            "ligament": len(edges)}


def collapse_map(coarse_proj, coarse_ld, refined):
    """The contraction of the refined lift back onto the coarse one.

    Old vertex indices are kept, so the collapse is the identity there and
    ``None`` on the new midpoints.  Verified: same projections, same local
    degrees, and the new vertices are exactly the ones over the auxiliary point.
    """
    n = refined["old_vertex_count"]
    same = all(coarse_proj[u] == refined["proj"][u] for u in range(n))
    same_ld = all(coarse_ld[u] == refined["ld"][u] for u in range(n))
    return {"vertex": {u: (u if u < n else None) for u in refined["proj"]},
            "old_vertices": n,
            "projection_preserved": bool(same),
            "local_degree_preserved": bool(same_ld),
            "new_vertices": [u for u in sorted(refined["proj"]) if u >= n],
            "new_over": [refined["proj"][u] for u in sorted(refined["proj"])
                         if u >= n]}


# ------------------------------------------------------------------- metric
def base_angles(tris, positions):
    """Angle sum of the base triangulation at every base point (spherical)."""
    X = np.asarray(positions, dtype=float)
    total = {}
    for tr in tris:
        verts = tr[0]
        for k in range(3):
            i, j, mm = verts[k], verts[(k + 1) % 3], verts[(k + 2) % 3]
            a, b = X[j] - X[i], X[mm] - X[i]
            den = float(np.linalg.norm(a) * np.linalg.norm(b))
            if den <= 0.0:
                raise ValueError(f"coincident base corners at {i} of {verts}")
            total[i] = total.get(i, 0.0) + float(
                np.arccos(np.clip(float(a @ b) / den, -1.0, 1.0)))
    return total


def inherited_metric(tris, base_of, proj, ld, positions):
    """Chordal lengths pulled back from the base, and ``ld * base angle`` cones."""
    X = np.asarray(positions, dtype=float)
    lengths = {}
    for l, b in base_of.items():
        i, j = b
        ell = float(np.linalg.norm(X[i] - X[j]))
        if not ell > 0.0:
            raise ValueError(f"base edge {b} has zero chordal length")
        lengths[l] = ell
    ang = base_angles(tris, X)
    cone = {u: ld[u] * ang[proj[u]] for u in proj}
    return {"lengths": lengths, "cone_angles": cone, "base_angles": ang,
            "hyperideal": sorted(u for u in cone if cone[u] >= 2 * np.pi - 1e-12)}


def base_from_faces(faces):
    """A labelled triangulation from oriented vertex triples."""
    tris = []
    for f in faces:
        labs = tuple(canon(f[t], f[(t + 1) % 3]) for t in range(3))
        tris.append((tuple(f), labs))
    return tris


# --------------------------------------------------------- the conformal seed
def _triad(pole):
    n = np.asarray(pole, dtype=float)
    n = n / np.linalg.norm(n)
    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(seed @ n)) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    e1 = seed - float(seed @ n) * n
    e1 = e1 / np.linalg.norm(e1)
    return e1, np.cross(n, e1), n


def _stereo(v, trip):
    e1, e2, n = trip
    v = np.asarray(v, dtype=float)
    return complex(float(v @ e1), float(v @ e2)) / (1.0 - float(v @ n))


def _unstereo(z, trip):
    e1, e2, n = trip
    z = complex(z)
    d = 1.0 + abs(z) ** 2
    return (2.0 * z.real * e1 + 2.0 * z.imag * e2
            + (abs(z) ** 2 - 1.0) * n) / d


def conformal_seed(proj, positions, sheets=None, branch=None):
    """Deterministic seed: the conformal branched lift, declared branch rule.

    A degree-2 branched cover of S^2 with exactly two branch points is
    ``A^{-1} o (z -> z^2) o A`` in the chart ``A`` carrying the branch points to
    ``0`` and ``infinity``, so the conformal lift of a base configuration is
    closed form: ``x_u = A^{-1}(sigma(u) sqrt(A(p)))`` over an unramified point
    and ``x_u = p`` at a branch point.

    ``sigma`` is a declared convention, never a search:

    * with ``sheets`` (the relative sheet of each lifted vertex, from the
      corner-slot DSU) it is ``sigma(u) = +1`` iff sheet 0 -- the sheet rule;
    * ``branch="principal"`` is ``sigma = +1`` everywhere -- the principal rule.

    Which one lands in the embedded branch is measured, not assumed.
    """
    if branch not in (None, "principal"):
        raise ValueError(f"unknown branch convention {branch!r}")
    fib = {}
    for u, p in proj.items():
        fib.setdefault(p, []).append(u)
    ram = sorted([p for p, us in fib.items() if len(us) == 1])
    if len(ram) != 2:
        raise ValueError(f"the conformal formula needs two branch points, got {ram}")
    X = np.asarray(positions, dtype=float)
    c = X.sum(axis=0)
    nrm = float(np.linalg.norm(c))
    pole = -c / nrm if nrm > 1e-12 else np.array([0.0, 0.0, 1.0])
    if float(np.min(np.linalg.norm(X - pole, axis=1))) < 1e-6:
        raise ValueError("stereographic pole coincides with a base point")
    trip = _triad(pole)
    z0, z1 = _stereo(X[ram[0]], trip), _stereo(X[ram[1]], trip)
    if abs(z1) > 1e6:
        raise ValueError("pole too close to the second branch point")
    A = lambda z: (z - z0) / (z - z1)
    Ainv = lambda w: (w * z1 - z0) / (w - 1.0)
    out = np.zeros((1 + max(proj), 3))
    for p, us in fib.items():
        if len(us) == 1:
            out[us[0]] = X[p]
            continue
        w = A(_stereo(X[p], trip))
        if abs(w) <= 1e-14:
            raise ValueError("an unramified base point sits at a branch point")
        zw = np.sqrt(w)
        for u in us:
            s = 1.0
            if sheets is not None:
                sh = sheets.get(u)
                if sh is None or len(sh) != 1:
                    raise ValueError(f"unramified lift {u} has no single sheet")
                s = 1.0 if next(iter(sh)) == 0 else -1.0
            if branch == "principal":
                s = 1.0
            out[u] = _unstereo(Ainv(s * zw), trip)
    if not np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-12):
        raise ValueError("seed points are not on the sphere")
    return out


def orientation_seed(proj, positions, oriented_faces, branch_states=None):
    """The conformal seed whose lifted triangles all have one orientation.

    The only free data in the conformal lift is which of the two preimages of an
    unramified base point goes to which lifted vertex.  Write that choice as a
    bit vector ``s``, one bit per unramified base point.  Then, exactly,

        sign det(y_u, y_v, y_w) = sign det(y_u, y_v, y_w)|_{s=0} * (-1)^{w_t . s}

    with ``w_t`` the incidence of the lifted triangle ``t`` (measured: the
    determinant sign is a character of the sign vector, checked at every
    configuration by ``multiplicative``).  So "every lifted triangle has the
    orientation of the complex" is a **linear system over F2**, solved here
    directly -- the 2^{#unramified} assignments are never enumerated.  The kernel
    contains the deck flip (1,...,1), which is an exact symmetry of the point set,
    so the gauge ``s_0 = 0`` is harmless.

    Returns ``(y, record)``; ``record`` carries the F2 rank, the solution set
    size, and the multiplicativity check.
    """
    if branch_states is None:
        branch_states = {}
    del branch_states
    fib = {}
    for u, p in proj.items():
        fib.setdefault(p, []).append(u)
    unram = sorted(p for p in fib if len(fib[p]) == 2)
    k = len(unram)
    if k == 0:
        return conformal_seed(proj, positions), {"rank": 0, "solutions": 1}

    def seed_of(bits):
        sh = {}
        for p, b in zip(unram, bits):
            u0, u1 = sorted(fib[p])
            sh[u0], sh[u1] = ({0}, {1}) if b == 0 else ({1}, {0})
        return conformal_seed(proj, positions, sheets=sh)

    def signs(y):
        return np.array([1 if np.linalg.det(np.array([y[a], y[b], y[c]])) > 0 else -1
                         for a, b, c in oriented_faces], dtype=int)

    z = tuple([0] * k)
    s0 = signs(seed_of(z))
    target = s0[0]
    e = [0 if v == target else 1 for v in s0]
    W = np.zeros((len(oriented_faces), k), dtype=int)
    for j in range(k):
        bits = list(z)
        bits[j] = 1
        sj = signs(seed_of(tuple(bits)))
        W[:, j] = [0 if a == b else 1 for a, b in zip(sj, s0)]
    # F2 elimination on the augmented system
    A = [row[:] + [e[i]] for i, row in enumerate(W.tolist())]
    rows, ncol = len(A), k
    piv, r = [], 0
    for c in range(ncol):
        p = next((i for i in range(r, rows) if A[i][c] % 2), None)
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        for i in range(rows):
            if i != r and A[i][c] % 2:
                A[i] = [(a + b) % 2 for a, b in zip(A[i], A[r])]
        piv.append(c)
        r += 1
    for i in range(r, rows):
        if A[i][ncol] % 2 and not any(x % 2 for x in A[i][:ncol]):
            raise ValueError("no orientation-consistent conformal seed exists")
    sol = [0] * k
    for i, c in enumerate(piv):
        sol[c] = A[i][ncol] % 2
    # gauge: the deck flip is a symmetry, so fix the first free coordinate
    if sol[0]:
        sol = [1 - x for x in sol]
    y = seed_of(tuple(sol))
    final = signs(y)
    pred = [s0[t] * (-1 if sum(W[t, j] * sol[j] for j in range(k)) % 2 else 1)
            for t in range(len(oriented_faces))]
    return y, {"rank": r, "solutions": 2 ** (k - r), "bits": sol,
               "one_sign": bool(len(set(final.tolist())) == 1),
               "multiplicative": bool(list(final) == pred),
               "seed_sign": int(final[0]) if len(set(final.tolist())) == 1 else 0}
