"""General-degree, per-edge-permutation sheet assembly of a branched cover.

This is the general construction that ``refined_cover.lift_triangulation`` is the
Z/2 specialisation of.  Nothing below assumes d = 2, two branch points, a single
cut path, or Z/2-valued labels.

Data
----
A *base* is a consistently oriented triangulation of a closed surface, given as
a list of ``(verts, labs)``: ``verts[t] -> verts[t+1]`` carries ``labs[t]``.  A
*dart* is an ordered pair ``(a, b)``; because the base is oriented, each dart is
carried by exactly one face.

A *per-edge permutation datum* on a degree-``d`` cover is a map

    edge_perms[canon(a, b)] = pi in S_d,

read as: **the corner of the face carrying the dart ``a -> b`` has sheet ``s``
at both ends of the edge exactly when the corner of the face carrying
``b -> a`` there has sheet ``pi(s)``.**  Off the support of the datum the
permutation is the identity, i.e. the cover is trivialised across that edge.

Assembly rule
-------------
Node ``(v, k, s)`` is "vertex ``v`` seen from face ``k`` on sheet ``s``".  Across
the base edge ``e = {a, b}`` shared by the dart faces ``k1`` (carries ``a -> b``)
and ``k2`` (carries ``b -> a``) we impose

    (a, k1, s) ~ (a, k2, pi(s))        and        (b, k1, s) ~ (b, k2, pi(s)).

The lifted vertices over ``v`` are the resulting classes; the local degree is
the common number of sheets in the class over one fixed incident base face.
(The total number of distinct sheets in the class can be larger at a cut vertex.)
A vertex on the cut therefore opens into as many copies as there are orbits of
its local monodromy, which is the "proper cut-vertex copy" rule -- not a global
choice of sheet per base point.
"""

from __future__ import annotations

from pullback.refined_cover import canon, tri_edges, rotation_system, _far


# --------------------------------------------------------------- permutations
def perm_identity(d):
    return tuple(range(d))


def perm_is_identity(p):
    return all(i == x for i, x in enumerate(p))


def perm_inverse(p):
    inv = [0] * len(p)
    for i, j in enumerate(p):
        inv[j] = i
    return tuple(inv)


def perm_compose(p, q):
    """``p`` after ``q``: ``(p o q)(x) = p(q(x))``."""
    return tuple(p[q[x]] for x in range(len(p)))


def perm_order(p):
    n, x = 1, p
    while not perm_is_identity(x):
        x = perm_compose(p, x)
        n += 1
    return n


def perm_cycle_type(p):
    seen, cyc = set(), []
    for i in range(len(p)):
        if i in seen:
            continue
        c, j = 0, i
        while j not in seen:
            seen.add(j)
            j = p[j]
            c += 1
        cyc.append(c)
    return tuple(sorted(cyc, reverse=True))


def perm_from_cycles(d, cycles):
    p = list(range(d))
    for cyc in cycles:
        for i, x in enumerate(cyc):
            p[x] = cyc[(i + 1) % len(cyc)]
    return tuple(p)


# ------------------------------------------------------------------- the base
class Base:
    """Consistently oriented triangulation plus its dart bookkeeping."""

    def __init__(self, tris, check_orientation=True):
        self.tris = [(tuple(int(x) for x in tr[0]), tuple(tr[1])) for tr in tris]
        self.edges = tri_edges(self.tris)
        self.dart_face = {}
        for k, (verts, _labs) in enumerate(self.tris):
            for t in range(3):
                d = (verts[t], verts[(t + 1) % 3])
                if d in self.dart_face:
                    raise ValueError(f"dart {d} is carried by two faces: "
                                     "the base is not consistently oriented")
                self.dart_face[d] = k
        if check_orientation:
            for (a, b) in self.dart_face:
                if (b, a) not in self.dart_face:
                    raise ValueError(f"dart {a}->{b} has no opposite dart")
        self.incident = {}
        for k, (verts, _l) in enumerate(self.tris):
            for v in verts:
                self.incident.setdefault(v, []).append(k)
        self.vertices = sorted(self.incident)
        self.face_degree = {v: len(self.incident[v]) for v in self.vertices}

    def out_vertex(self, v, k):
        """The other end of the dart of face ``k`` that leaves ``v``."""
        verts = self.tris[k][0]
        t = verts.index(v)
        return verts[(t + 1) % 3]

    def next_face(self, v, k):
        """Face reached from ``k`` across the edge leaving ``v``."""
        w = self.out_vertex(v, k)
        return self.dart_face[(w, v)]

    def flower(self, v):
        """Faces and edges around ``v`` in the positive rotation order."""
        out, k = [], self.incident[v][0]
        for _ in range(len(self.incident[v])):
            out.append((k, canon(v, self.out_vertex(v, k))))
            k = self.next_face(v, k)
        if k != self.incident[v][0]:
            raise ValueError(f"vertex {v}: flower does not close")
        return out

    def n_faces(self):
        return len(self.tris)

    def n_edges(self):
        return len(self.edges)

    def n_vertices(self):
        return len(self.vertices)

    def euler(self):
        return self.n_vertices() - self.n_edges() + self.n_faces()


def base_from_face_lists(faces):
    """``[(a,b,c), ...]`` with canon labels, exactly as ``refined_cover`` does."""
    return Base([(tuple(f), tuple(canon(f[t], f[(t + 1) % 3]) for t in range(3)))
                 for f in faces])


# ------------------------------------------------------------------ the datum
def _dart_perm(edge_perms, v, e):
    """Permutation to apply when crossing ``e`` in the dart leaving ``v``."""
    p = edge_perms.get(e)
    if p is None:
        return None
    return p if v == e[0] else perm_inverse(p)


def _other(e, v):
    return e[1] if v == e[0] else e[0]


def local_perms(base, edge_perms, d):
    """Realised local monodromy at every base vertex, as a permutation of S_d."""
    out = {}
    for v in base.vertices:
        s = perm_identity(d)
        for _k, e in base.flower(v):
            q = _dart_perm(edge_perms, v, e)
            if q is not None:
                s = perm_compose(q, s)
        out[v] = s
    return out


def perms_from_local(base, d, local, root=None, tree=None):
    """Solve for a per-edge datum with the prescribed local monodromies.

    The cut is a spanning tree ``tree`` of the 1-skeleton (by default a BFS tree
    rooted at ``root``); off the cut the datum is the identity, so the base minus
    the cut is a disk and the cover is trivialised over it.  Removing a leaf ``v``
    of the remaining tree leaves exactly one unknown permutation at ``v``, and the
    cyclic product at ``v`` then determines it; the root's equation is the only
    check.  A consistent family is therefore solved for, never guessed.
    """
    verts = base.vertices
    if root is None:
        root = verts[0]
    adj = {}
    for e in base.edges:
        adj.setdefault(e[0], set()).add(e[1])
        adj.setdefault(e[1], set()).add(e[0])
    if tree is None:
        prev = {root: None}
        queue = [root]
        while queue:
            u = queue.pop(0)
            for w in sorted(adj[u]):
                if w not in prev:
                    prev[w] = u
                    queue.append(w)
        if len(prev) != len(verts):
            raise ValueError("the 1-skeleton is not connected")
        tree = {canon(v, prev[v]) for v in verts if prev[v] is not None}
    children = {}
    for e in tree:
        children.setdefault(e[0], []).append(e[1])
        children.setdefault(e[1], []).append(e[0])

    depth = {root: 0}
    queue = [root]
    while queue:
        u = queue.pop(0)
        for w in sorted(children.get(u, [])):
            if w not in depth:
                depth[w] = depth[u] + 1
                queue.append(w)
    order = sorted((v for v in verts if v != root), key=lambda v: -depth[v])

    edge_perms = {e: perm_identity(d) for e in base.edges}
    for v in order:
        flower = base.flower(v)
        cands = [e for _k, e in flower
                 if e in tree and depth.get(_other(e, v)) == depth[v] - 1]
        if len(cands) != 1:
            raise ValueError(f"vertex {v}: {len(cands)} parent edges in the cut tree")
        e_par = cands[0]
        js = [i for i, (_k, e) in enumerate(flower) if e == e_par]
        if len(js) != 1:
            raise ValueError(f"vertex {v}: the parent edge is not in the flower once")
        j = js[0]
        k = len(flower)
        pre = perm_identity(d)     # P_{j-1} o ... o P_0
        for i in range(j):
            pre = perm_compose(_dart_perm(edge_perms, v, flower[i][1]), pre)
        post = perm_identity(d)    # P_{k-1} o ... o P_{j+1}
        for i in range(k - 1, j, -1):
            post = perm_compose(_dart_perm(edge_perms, v, flower[i][1]), post)
        # sigma = post o P_j o pre  =>  P_j = post^{-1} o sigma o pre^{-1}
        pj = perm_compose(perm_inverse(post),
                          perm_compose(local[v], perm_inverse(pre)))
        edge_perms[e_par] = pj if v == e_par[0] else perm_inverse(pj)

    realised = local_perms(base, edge_perms, d)
    for v in verts:
        if realised[v] != tuple(local[v]):
            raise ValueError(
                f"prescribed local monodromy at {v} is not consistent: got "
                f"{realised[v]}, wanted {tuple(local[v])}")
    return edge_perms


# ------------------------------------------------------------------ the lift
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


def assemble(base, d, edge_perms):
    """Lift ``base`` along the datum; return the complex and its certificates."""
    for e, p in edge_perms.items():
        if len(p) != d or sorted(p) != list(range(d)):
            raise ValueError(f"datum on {e} is not a permutation of S_{d}: {p}")
    dsu = _DSU()
    for k, (verts, _labs) in enumerate(base.tris):
        for v in verts:
            for s in range(d):
                dsu.find((v, k, s))
    for e, p in edge_perms.items():
        k1 = base.dart_face[(e[0], e[1])]          # carries min -> max
        k2 = base.dart_face[(e[1], e[0])]
        for v in e:
            for s in range(d):
                dsu.union((v, k1, s), (v, k2, p[s]))

    classes = {}
    for k, (verts, _labs) in enumerate(base.tris):
        for v in verts:
            for s in range(d):
                classes.setdefault(dsu.find((v, k, s)), set()).add((v, k, s))
    roots = sorted(classes, key=lambda r: min(classes[r]))
    ren = {r: i for i, r in enumerate(roots)}
    proj, sheets, ld, per_face = {}, {}, {}, {}
    for r in roots:
        nodes = classes[r]
        vs = {v for v, _k, _s in nodes}
        if len(vs) != 1:
            raise ValueError(f"lifted vertex over several base points {sorted(vs)}")
        v = vs.pop()
        # corner-slots of this lifted vertex over each base face at v: the
        # local degree is the common count.  A genuine cover has the same
        # number over every sector, and we insist on it here.
        counts = {}
        for _v, k, s in nodes:
            counts.setdefault(k, set()).add(s)
        e = {len(ss) for ss in counts.values()}
        if len(e) != 1:
            raise ValueError(f"lifted vertex over {v} has sector counts "
                             f"{ {k: len(s) for k, s in counts.items()} }")
        u = ren[r]
        proj[u] = v
        ld[u] = e.pop()
        sheets[u] = sorted({s for _v, _k, s in nodes})
        per_face[u] = {k: sorted(ss) for k, ss in counts.items()}
    index = {}
    for u in proj:
        index.setdefault(proj[u], []).append(u)

    # lifted edges, keyed by (base edge, sheet on the min->max side)
    lift_edges, base_of = {}, {}
    for e, p in edge_perms.items():
        k1 = base.dart_face[(e[0], e[1])]
        for s in range(d):
            a = ren[dsu.find((e[0], k1, s))]
            b = ren[dsu.find((e[1], k1, s))]
            if a == b:
                raise ValueError(f"lifted edge over {e} in sheet {s} is a loop")
            lift_edges[(e, s)] = (a, b)
            base_of[(e, s)] = e
    lift_tris = []
    for k, (verts, labs) in enumerate(base.tris):
        for s in range(d):
            vs = tuple(ren[dsu.find((verts[t], k, s))] for t in range(3))
            if len(set(vs)) != 3:
                raise ValueError(f"lifted triangle over {verts} degenerate: {vs}")
            ls = []
            for t in range(3):
                e = labs[t]
                a, b = vs[t], vs[(t + 1) % 3]
                lab = (e, s) if base.dart_face[(e[0], e[1])] == k else \
                      (e, perm_inverse(edge_perms[e])[s])
                if canon(*lift_edges[lab]) != canon(a, b):
                    raise ValueError(f"lifted label {lab} misaligned over {verts}")
                ls.append(lab)
            lift_tris.append((vs, tuple(ls)))

    lift = {"edges": lift_edges, "tris": lift_tris, "proj": proj,
            "sheets": sheets, "ld": ld, "base_of": base_of, "index": index,
            "per_face": per_face, "d": d, "edge_perms": dict(edge_perms)}
    lift["cert"] = certify(base, d, lift)
    return lift


def certify(base, d, lift):
    """Exact combinatorial certificates.  No geometry is used anywhere."""
    ce, ct, proj, ld = lift["edges"], lift["tris"], lift["proj"], lift["ld"]
    out = {}
    # 1. every lifted edge is used by exactly two lifted faces, once each way
    used, incident_faces, oriented_ok = {}, {}, True
    for k, (verts, labs) in enumerate(ct):
        for t in range(3):
            lab = labs[t]
            a, b = verts[t], verts[(t + 1) % 3]
            u = used.setdefault(lab, [])
            if u and u[-1][0] == a:
                oriented_ok = False
            u.append((a, b))
            incident_faces.setdefault(lab, []).append(k)
    bad = [lab for lab, u in used.items() if len(u) != 2]
    out["edge_in_two_faces"] = not bad
    out["bad_edge_count"] = len(bad)
    out["orientation_consistent"] = bool(oriented_ok and not bad and all(
        u[0] == (u[1][1], u[1][0]) for u in used.values()))

    # Face connectivity is independent of Euler characteristics: a sphere plus
    # a torus also has chi=2.  Union adjacent faces, including parallel-edge
    # identities, so a disconnected cover cannot masquerade as one sphere.
    parent = list(range(len(ct)))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for fs in incident_faces.values():
        if len(fs) == 2:
            a, b = (root(k) for k in fs)
            if a != b:
                parent[a] = b
    out["components"] = len({root(k) for k in range(len(ct))})
    out["connected"] = bool(ct) and out["components"] == 1
    # 2. it is a cover with multiplicity d, no phantom labels
    base_edges = {canon(*p) for p in base.edges.values()}
    per_edge, phantom = {}, []
    for lab, (i, j) in ce.items():
        b = lift["base_of"][lab]
        if b not in base_edges:
            phantom.append(lab)
        per_edge[b] = per_edge.get(b, 0) + 1
        if canon(proj[i], proj[j]) != canon(*b):
            raise ValueError(f"lifted edge {lab} joins projections "
                             f"{canon(proj[i], proj[j])}, labelled {canon(*b)}")
    base_tris = {frozenset(tr[0]) for tr in base.tris}
    per_tri = {}
    for verts, _labs in ct:
        img = frozenset(proj[v] for v in verts)
        if len(img) != 3 or img not in base_tris:
            raise ValueError(f"lifted triangle {verts} projects to {sorted(img)}")
        per_tri[img] = per_tri.get(img, 0) + 1
    out["no_phantom_labels"] = not phantom
    out["edge_multiplicity"] = sorted(set(per_edge.values()))
    out["face_multiplicity"] = sorted(set(per_tri.values()))
    out["cover_ok"] = bool(not phantom and not bad
                           and all(v == d for v in per_edge.values())
                           and all(v == d for v in per_tri.values()))
    # 3. Euler characteristic and Riemann-Hurwitz
    out["counts"] = {"base": [base.n_vertices(), base.n_edges(), base.n_faces()],
                     "lift": [len(proj), len(ce), len(ct)]}
    out["chi_base"] = base.euler()
    out["chi_lift"] = len(proj) - len(ce) + len(ct)
    out["rh_deficit"] = int(sum(ld[u] - 1 for u in proj))
    out["rh_expected"] = 2 * d - 2
    out["rh_ok"] = out["rh_deficit"] == out["rh_expected"]
    out["chi_lift_ok"] = out["chi_lift"] == 2
    # 4. per-vertex fibre accounting: sum of local degrees over each fibre = d
    fib = {}
    for u, p in proj.items():
        fib[p] = fib.get(p, 0) + ld[u]
    out["fibre_ok"] = all(v == d for v in fib.values())
    out["fibres"] = {str(p): sorted((ld[u] for u in proj if proj[u] == p),
                                    reverse=True) for p in sorted(fib)}
    # 5. the ribbon structure: every lifted flower is the base flower, repeated e
    base_rot = rotation_system(base.tris, base.edges)
    lift_rot = rotation_system(ct, ce)
    rows, bad_v = [], []
    for u in sorted(lift_rot, key=str):
        p = proj[u]
        bw = [_far(dd, base.edges) for dd in base_rot[p]]
        lw = [proj[_far(dd, ce)] for dd in lift_rot[u]]
        e = ld[u]
        ok = len(lw) == e * len(bw) and any(
            lw == (bw[r:] + bw[:r]) * e for r in range(len(bw)))
        if not ok:
            bad_v.append(u)
        rows.append({"u": u, "over": p, "ld": e, "ok": bool(ok),
                     "len": len(lw), "want": e * len(bw)})
    out["ribbon_ok"] = not bad_v
    out["ribbon_bad"] = bad_v
    out["ribbon_rows"] = rows
    out["is_sphere"] = bool(out["connected"] and out["chi_lift"] == 2
                            and out["orientation_consistent"] and out["ribbon_ok"])
    # 6. the datum really realises the declared local monodromies
    out["local_realised"] = local_perms(base, lift["edge_perms"], d)
    out["parallel_pairs"] = _parallel(ce)
    return out


def _parallel(ce):
    pairs = {}
    for lab, (i, j) in ce.items():
        pairs.setdefault(canon(i, j), []).append(lab)
    return sorted([sorted(str(l) for l in ls) for ls in pairs.values() if len(ls) > 1])


def critical_profile(base, d, lift):
    """Per base vertex: local degrees over it, descending."""
    proj, ld = lift["proj"], lift["ld"]
    return {v: sorted((ld[u] for u in proj if proj[u] == v), reverse=True)
            for v in base.vertices}
