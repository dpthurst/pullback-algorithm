"""The production pipeline: (base, cover, phi) + labelled positions -> one step.

Derived, in order:

    coarse   T  base triangulation of the punctures (the Voronoi spine's dual)
             K  = pi^{-1}(T), the trunk's ``VEInit.cells`` as a triangulation
             branch datum, relative sheets, ramified points     [read off K]
    portrait labels, local degrees, dynamics                     [abstract VE]
    split    T' = T with the branch-point edge split at its arc midpoint m
             K' = preimage subdivision of K over that edge (labels inherited)
             branch datum of the refinement, sheets of the new vertices
    metric   chordal lengths pulled back from the base, cones ld * base angle
    step     conformal seed -> Gauss-Newton -> labelled read-off

The object delivered here is a **refined cover + inherited metric + seed + one
numerical step**, not the VE's pullback map.  Two things are *declared inputs*
taken from the trunk's initialisation, and neither is derived here:

* the *transport*, which fixes the geometric marking.  Pass ``chains`` (or use
  ``RefinedVE.from_declared``) to validate a declared transport without a census;
  the legacy default constructs it using ``VEInit``'s exhaustive search, and
* the *postcritical lift table* ``postcritical_lifts`` (default: the trunk's
  ``VEInit.lifted_labels``), i.e. which of the two lifts over an unramified
  puncture carries a postcritical label.  The portrait does **not** determine
  it: both lifts share ``(projection, local degree)``, so the table carries one
  bit per unramified puncture, and only the *global* deck flip of that table is
  a gauge symmetry (machine-checked in ``test_portrait_is_not_enough``).

The portrait (puncture, local degree, image puncture per cover face) is the face
data the step consumes; it is a homotopy-invariant *shadow* of ``phi'``, not a
substitute for its homotopy class: a mapping class can fix the punctures and the
whole portrait and still change the induced map on ``pi_1``, hence the Thurston
pullback.
"""

from __future__ import annotations

import numpy as np

from . import refined_cover as rr
from pullback import embedding
from pullback import rounding_multi as mg
from pullback.mesh import to_vector
from pullback.ve_state import VEInit


# --------------------------------------------------------------- labels/marks
def puncture_of_base_face(init):
    """Base face -> puncture label, transported by the trunk's marking."""
    out = {}
    for k, face in enumerate(init.marking):
        out[face] = init.face_points[k]
    if sorted(out) != sorted(range(len(out))):
        raise ValueError(f"the marking is not a bijection: {init.marking}")
    return out


def propagate_sheets(edges, base_of, proj, branch, root=None):
    """Relative sheet of every unramified lifted vertex, by exact propagation.

    ``sheet(v) = sheet(u) xor branch(f)`` across a lifted edge over the base edge
    ``f``.  The only choice is the root and its partner (the deck flip); every
    other value is forced, and a contradiction raises rather than opening a new
    candidate.  Deterministic: the root is the smallest unramified vertex over the
    smallest base point with a two-point fibre.
    """
    fib = {}
    for u, p in proj.items():
        fib.setdefault(p, []).append(u)
    cand = sorted(p for p in fib if len(fib[p]) == 2)
    sheets = {}
    if root is not None:
        sheets = dict(root)
    elif cand:
        u0, u1 = sorted(fib[cand[0]])
        sheets = {u0: 0, u1: 1}
    todo = list(sheets)
    while todo:
        u = todo.pop()
        for l, (a, b) in edges.items():
            if a == u:
                v = b
            elif b == u:
                v = a
            else:
                continue
            if len(fib[proj[v]]) == 1:
                continue
            want = sheets[u] ^ branch.get(base_of[l], 0)
            if v in sheets:
                if sheets[v] != want:
                    raise ValueError(f"sheet contradiction at {v} over {base_of[l]}")
            else:
                sheets[v] = want
                todo.append(v)
    return sheets, (cand[0] if cand else None)


def derive_branch_datum(edgesT, proj):
    """The Z/2 branch datum of the coarse lift, and the ramified base points.

    ``ramified`` is an exact read-off of the cover: a base point is ramified iff
    its fibre has a single lifted vertex.  The datum itself is **not** determined
    by the cover -- every two cochains with the same ``eps`` give isomorphic
    lifts, because the covering is classified by ``pi_1(S^2 - P) -> Z/2``, which
    is ``eps`` modulo ``sum eps = 0``.  So the datum is taken with a declared
    convention: the shortest path in the base triangulation between the two
    ramified points, by breadth-first search with sorted edges.  For the rabbit
    that is the single edge joining the two branch points.

    Verified, not assumed: the declared datum has odd degree at exactly the
    ramified points, and the lift it produces is the trunk's complex.
    """
    fib = {}
    for u, p in proj.items():
        fib.setdefault(p, []).append(u)
    ram = sorted(p for p in fib if len(fib[p]) == 1)
    if len(ram) != 2:
        raise ValueError(f"expected two ramified base points, got {ram}")
    adj = {}
    for f, (p, q) in edgesT.items():
        adj.setdefault(p, []).append(q)
        adj.setdefault(q, []).append(p)
    a, b = ram
    prev = {a: None}
    queue = [a]
    while queue:
        v = queue.pop(0)
        for w in sorted(adj[v]):
            if w not in prev:
                prev[w] = v
                queue.append(w)
    if b not in prev:
        raise ValueError(f"{b} is unreachable from {a}")
    path, v = [], b
    while prev[v] is not None:
        path.append(rr.canon(v, prev[v]))
        v = prev[v]
    branch = {f: 0 for f in edgesT}
    for f in path:
        branch[f] = 1
    eps = {p: 0 for p in fib}
    for f, val in branch.items():
        if val:
            for p in edgesT[f]:
                eps[p] ^= 1
    want = {p: (1 if len(fib[p]) == 1 else 0) for p in fib}
    if eps != want:
        raise ValueError(f"the declared branch datum gives eps {eps}, not {want}")
    return branch, ram, path


# ------------------------------------------------------------------ the state
class FoldedBase(ValueError):
    """The base complex at this configuration is not an embedded triangulation.

    The stellar split of a base edge at its arc midpoint folds when a triangle at
    that edge is not contained in an open hemisphere.  The inherited metric is
    then the metric of a folded base, and no step is defined.
    """


class RefinedVE:
    """The refined ribbon/cover data, its labels, metric and one frozen step."""

    def __init__(self, session, points=None, seed=0, chains=None, verbose=False,
                 postcritical_lifts=None, init=None):
        self.session = session
        self.init = init = (VEInit(session, points=points, seed=seed, chains=chains)
                            if init is None else init)
        self.points = [to_vector(p) for p in init.points]
        n = len(self.points)

        # ---- coarse complex: the trunk's dual cells, read as a triangulation
        self.T = T = rr.base_from_faces(init.base_faces)
        self.edgesT = rr.tri_edges(T)
        cells = init.cells
        dual = cells["dual"]
        self.K_edges = {l: tuple(int(x) for x in dual.edges[l]) for l in cells["edges"]}
        self.base_of = {l: rr.canon(*l[0]) for l in self.K_edges}
        self.K_tris = [(tuple(int(x) for x in cells["polygons"][k]),
                        tuple(d[0] for d in cells["corners"][k]), None)
                       for k in range(len(cells["polygons"]))]
        self.proj = {u: init.face_points[init.pi_rows[u]["target"]]
                     for u in range(len(init.pi_rows))}
        self.ld = {u: int(init.pi_rows[u]["degree"]) for u in range(len(init.pi_rows))}
        if any(p not in range(n) for p in self.proj.values()):
            raise ValueError("a lifted vertex has no puncture label")

        # ---- the portrait, from the abstract VE (no phi', no transport section)
        pbf = puncture_of_base_face(init)
        self.rows, self.dyn, self.active = rr.portrait(
            session.cover, session.base, session.phi, pbf)
        self.branch, self.ram, self.branch_path = derive_branch_datum(
            self.edgesT, self.proj)
        self.K_lift = rr.lift_triangulation(
            T, sorted(f for f, v in self.branch.items() if v))

        # ---- the split at the branch-point edge
        m = self.m = n
        if len(self.ram) != 2:
            raise ValueError(f"expected two ramified base points, got {self.ram}")
        e = rr.canon(*self.ram)
        if e not in {rr.canon(*p) for p in self.edgesT.values()}:
            raise ValueError(f"the branch points {self.ram} are not joined by a base "
                             "edge, so this repair does not apply")
        self.split_edge = e
        self.T2 = rr.star_split(T, e, m)
        self.edgesT2 = self.T2["edges"]
        self.branch2 = {f: v for f, v in self.branch.items() if f != e}
        for x in e:
            f = rr.canon(x, m)
            self.branch2[f] = 1                    # each branch half carries phi = 1
        coarse = {"edges": self.K_edges, "tris": self.K_tris,
                  "proj": self.proj}
        self.K2 = rr.split_lifted(coarse, e, m, ld=self.ld)

        # ---- labels: the declared postcritical lift table, untouched by the split
        self.labels = dict(postcritical_lifts if postcritical_lifts is not None
                           else init.lifted_labels)
        self.labels_source = ("declared" if postcritical_lifts is not None
                              else "VEInit.lifted_labels (inherited)")
        for p, u in self.labels.items():
            if self.K2["proj"][u] != self.dyn[p]:
                raise ValueError(f"label {p}: lifted vertex {u} sits over "
                                 f"{self.K2['proj'][u]}, but dyn({p}) = {self.dyn[p]}")
        self.target = {u: p for p, u in self.labels.items()}
        if verbose:
            print("   branch datum", sorted(f for f, v in self.branch.items() if v),
                  "| ramified", self.ram, "| split", e, "| aux", m,
                  "| labels", self.labels)

    @classmethod
    def from_declared(cls, session, chains, postcritical_lifts, points=None):
        """Build without a transport census, from explicitly declared inputs.

        ``chains`` and the postcritical lift table are convention inputs, not
        values the portrait can determine.  ``pullback_ve`` validates the
        declared transport; everything after it is derived from corner slots.
        """
        init = VEInit(session, points=points, chains=chains)
        return cls(session, postcritical_lifts=postcritical_lifts, init=init)

    # -------------------------------------------------------------- checks
    def certificates(self):
        ce, ct = self.K2["edges"], self.K2["tris"]
        rb = rr.ribbon_check(self.T2["tris"], self.edgesT2, ct, ce,
                             self.K2["proj"], self.K2["ld"])
        cert = {}
        cert["coarse_counts"] = {"T": [1 + max(max(x) for x in self.edgesT.values()),
                                        len(self.edgesT), len(self.T)],
                                 "K": [len(self.proj), len(self.K_edges),
                                       len(self.K_tris)]}
        cert["refined_counts"] = {
            "T'": [1 + max(max(x) for x in self.edgesT2.values()),
                   len(self.edgesT2), len(self.T2["tris"])],
            "K'": [1 + max(max(x) for x in ce.values()), len(ce), len(ct)]}
        cert["cover2"] = rr.cover_check(self.T2["tris"], self.edgesT2, ct, ce,
                                       self.K2["proj"], self.K2["base_of"])
        cert["ribbon2_ok"] = rb["ok"]
        cert["ribbon2_rows"] = rb["rows"]
        cert["collapse"] = rr.collapse_map(self.proj, self.ld, self.K2)
        cert["branch_datum"] = sorted(f for f, v in self.branch.items() if v)
        cert["branch_datum2"] = sorted(f for f, v in self.branch2.items() if v)
        cert["ramified"] = list(self.ram)
        cert["split_edge"] = self.split_edge
        cert["aux"] = self.m
        cert["aux_vertices"] = sorted(u for u in self.K2["proj"]
                                      if self.K2["proj"][u] == self.m)
        cert["aux_not_postcritical"] = all(u not in self.target
                                           for u in cert["aux_vertices"])
        cert["labels"] = dict(sorted(self.labels.items()))
        cert["dyn"] = dict(self.dyn)
        cert["portrait_rows"] = [{k: v for k, v in r.items() if k != "face"}
                                 for r in self.rows]
        cert["labels_roundtrip"] = all(self.K2["proj"][u] == self.dyn[p]
                                       for p, u in self.labels.items())
        cert["degrees"] = {str(u): int(self.K2["ld"][u]) for u in sorted(self.K2["proj"])}
        cert["proj"] = {str(u): int(self.K2["proj"][u]) for u in sorted(self.K2["proj"])}
        fib = {}
        for u, p in self.K2["proj"].items():
            fib[p] = fib.get(p, 0) + self.K2["ld"][u]
        cert["fibre_degrees"] = {str(p): int(v) for p, v in sorted(fib.items())}
        cert["fibre_degrees_ok"] = all(v == self.session.degree for v in fib.values())
        cert["subdivision"] = self.subdivision_certificates()
        return cert

    def subdivision_certificates(self, levels=2):
        """Uniform 1 -> 4 refinement must commute with the cover.

        Subdivide the refined base and the refined lift with the same rule.  The
        subdivided lift must still be a cover of the subdivided base with the
        same edge/face multiplicities and the same base-edge labels, and the
        counts must match the verified chain V8/E18/F12 -> V26/E72/F48 ->
        V98/E288/F192.
        """
        base_tris = self.T2["tris"]
        bof = {l: rr.canon(*p) for l, p in self.T2["edges"].items()}
        lift = {"tris": self.K2["tris"], "edges": self.K2["edges"],
                "base_of": self.K2["base_of"], "proj": self.K2["proj"],
                "ld": self.K2["ld"]}
        out = []
        for k in range(levels):
            base_tris, bof, lift = self._one_level(base_tris, bof, lift)
            bverts = 1 + max(x for tr in base_tris for x in tr[0])
            lverts = 1 + max(x for tr in lift["tris"] for x in tr[0])
            out.append({"level": k + 1,
                        "base": [bverts, len(bof), len(base_tris)],
                        "lift": [lverts, len(lift["base_of"]),
                                 len(lift["tris"])]})
        return out

    def _one_level(self, base_tris, bof, lift):
        bsub = rr.subdivide_1to4(base_tris, bof)
        bmid = {rr.canon(*bof[l]): v for l, v in bsub["mid"].items()}
        lsub = rr.subdivide_1to4(lift["tris"], lift["base_of"],
                                 proj=lift["proj"], base_mid=bmid)
        proj = dict(lift["proj"])
        ld = dict(lift["ld"])
        for l2, v2 in lsub["mid"].items():
            f2 = lift["base_of"][l2]
            proj[v2] = bmid[f2]
            ld[v2] = 1
        chk = rr.cover_check(bsub["tris"], bsub["edges"], lsub["tris"],
                             lsub["edges"], proj, lsub["base_of"])
        if not chk["ok"]:
            raise ValueError(f"1->4 refinement does not commute with the cover: {chk}")
        return bsub["tris"], {l: rr.canon(*p) for l, p in bsub["edges"].items()}, {
            "tris": lsub["tris"], "edges": lsub["edges"],
            "base_of": lsub["base_of"], "proj": proj, "ld": ld}

    # ------------------------------------------------------------- geometry
    def base_positions(self, config):
        """The base points: the punctures, then the auxiliary arc midpoint."""
        X = np.zeros((self.m + 1, 3))
        for i in range(self.m):
            v = np.asarray(config[i], dtype=float)
            X[i] = v / np.linalg.norm(v)
        a, b = self.ram
        w = X[a] + X[b]
        nrm = float(np.linalg.norm(w))
        if nrm < 1e-9:
            raise ValueError(f"the branch points {self.ram} are antipodal: the arc "
                             "midpoint is not defined by this convention")
        X[self.m] = w / nrm
        return X

    def metric(self, config):
        X = self.base_positions(config)
        return X, rr.inherited_metric(self.T2["tris"], self.K2["base_of"],
                                     self.K2["proj"], self.K2["ld"], X)

    def base_orientation(self, config):
        """Determinant signs of the refined base complex at ``config``.

        ``one_sign`` is the gate: a folded base complex has a mixed pattern and
        the arc-midpoint convention has failed for that configuration.
        """
        X = self.base_positions(config)
        faces = [v for v, _l in self.T2["tris"]]
        labs = [l for _v, l in self.T2["tris"]]
        ori = embedding.orient_faces(faces, labs)
        det = np.array([np.linalg.det(np.array([X[a], X[b], X[c]]))
                        for a, b, c in ori])
        pos, neg = int((det > 0).sum()), int((det < 0).sum())
        return {"pos": pos, "neg": neg, "one_sign": bool(pos == 0 or neg == 0)}

    def oriented_faces(self):
        faces = [v for v, _l, _b in self.K2["tris"]]
        labels = [l for _v, l, _b in self.K2["tris"]]
        return embedding.orient_faces(faces, labels)

    def seed(self, config):
        """The orientation-consistent conformal seed (an F2 solve, not a census)."""
        X, _met = self.metric(config)
        y, rec = rr.orientation_seed(self.K2["proj"], X, self.oriented_faces())
        return {"X": X, "x0": y, "record": rec}

    def step(self, config, x0=None, maxit=600, tol=1e-14):
        """One frozen pullback step: inherited metric, realisation, read-off.

        Refuses to run on a folded base complex: the inherited metric would be
        the metric of a folded base, so the realised lift says nothing.
        """
        base = self.base_orientation(config)
        if not base["one_sign"]:
            raise FoldedBase(
                "the refined base complex is folded at this configuration "
                f"(determinant signs {base['pos']}/{base['neg']}); the "
                "arc-midpoint split convention does not apply")
        X, met = self.metric(config)
        seed_rec = None
        if x0 is None:
            sd = self.seed(config)
            x0, seed_rec = sd["x0"], sd["record"]
        x, res, it = mg.round_realise(self.K2["edges"], met["lengths"], x0=x0,
                                      maxit=maxit, tol=tol)
        cert = {"residual": float(res), "iterations": int(it),
                "seed": seed_rec, "base": base}
        info = mg.structure(self.K2["edges"], x)
        cert.update({k: info[k] for k in ("V", "E", "rank_A", "rank_PJ",
                                          "redundant", "excess")})
        faces = [v for v, _l, _b in self.K2["tris"]]
        labels = [l for _v, l, _b in self.K2["tris"]]
        ori = embedding.orient_faces(faces, labels)
        emb = embedding.classify(x, ori)
        cert.update({"embedded": bool(emb["embedded"]),
                     "degree": float(emb["degree"]),
                     "min_abs_det": float(emb["min_abs_det"]),
                     "reason": emb["reason"], "coordinate": emb["coordinate"]})
        cone = mg.cone_angles(labels, self.K2["edges"], x)
        want = np.array([met["cone_angles"][u] for u in range(len(x))])
        cert["cone_vs_ldbase"] = float(np.abs(cone - want).max())
        cert["cone_vs_2pi"] = float(np.abs(cone - 2 * np.pi).max())
        cert["cone"] = cone
        inj, hits = ambient_injective(x, ori)
        cert["injective"] = bool(inj)
        cert["hit_counts"] = sorted(set(int(h) for h in hits))
        new = [x[self.labels[p]] for p in range(self.m)]
        cert["config"] = new
        cert["chart"] = chart(new)
        cert["x"] = x
        return cert


# ------------------------------------------------------------------ helpers
def chart(config, gauge=None):
    """The labelled coordinate in a declared Mobius gauge.

    ``gauge`` is the ordered triple of labels sent to ``infinity, 0, 1``; the
    default is the first lexicographic triple of pairwise distinct points, which
    is what ``embedding.mobius_coordinate`` uses.  Declaring it makes comparisons
    across configurations explicit rather than implicit.
    """
    x = np.asarray(config, dtype=float)
    if gauge is None:
        return embedding.mobius_coordinate(x)
    from pullback.moduli import cross_ratio
    from pullback.mesh import to_point
    pts = [to_point(v) for v in x]
    i, j, k = gauge
    rest = [l for l in range(len(pts)) if l not in (i, j, k)]
    return ((i, j, k), tuple(cross_ratio(pts[i], pts[j], pts[k], pts[l])
                             for l in rest))


def ambient_injective(x, ori, n=240, seed=1):
    """Sampled ambient ray injectivity: every sampled direction meets once."""
    x = np.asarray(x, dtype=float)
    rng = np.random.RandomState(seed)
    w = rng.normal(size=(n, 3))
    w /= np.linalg.norm(w, axis=1)[:, None]
    hits = np.zeros(n, dtype=int)
    for a, b, c in ori:
        A, B, C = x[a], x[b], x[c]
        e1, e2 = B - A, C - A
        nrm = np.cross(e1, e2)
        den = w @ nrm
        num = float(A @ nrm)
        ok = np.abs(den) > 1e-14
        if not ok.any():
            continue
        idx = np.where(ok)[0]
        t = num / den[idx]
        good = t > 0
        idx, t = idx[good], t[good]
        if not len(idx):
            continue
        P = t[:, None] * w[idx]
        M = np.stack([e1, e2], axis=1)
        sol, *_ = np.linalg.lstsq(M, (P - A).T, rcond=None)
        u, v = sol[0], sol[1]
        inside = (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1 + 1e-12)
        hits[idx[inside]] += 1
    return bool((hits == 1).all()), hits
