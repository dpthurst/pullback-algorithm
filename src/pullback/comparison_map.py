"""The missing combinatorial step: active rose ``B_next`` and ``comparison_map``.

Read-only prototype for the coordinator.  Nothing here edits the repository; it
imports it (``PYTHONPATH=<repo>/src``) and adds exactly the one object the plan
names as missing:

    comparison_map : P_n -> B_next

on the ACTIVE rose ``B_next`` (one vertex, ``|P'| - 1`` petals, ``|P'|`` labelled
faces), together with the per-edge/per-face peripheral description of the
capped and the active loops, and the promoted state produced by
``lift.advance_comparison``.

Why this shape
--------------
``advance_comparison`` consumes ``upsilon : lifted -> new_spine`` and nothing
else from the geometry.  Its checks are on the composite

    comparison = upsilon . psi,        psi a homotopy section of phi_n,

which must be a homotopy equivalence carrying the punctures to the punctures.
So the combinatorial minimum is: choose ``B_next`` (a spine of the SAME surface)
and give ``upsilon`` whose composite with any section is such an equivalence.

The construction here is ``comparison_map := theta . phi_n`` where

    theta : B_n -> B_next

is the collapse of a spanning tree of the current base.  Three facts make this
the right primitive and they are all certified, not assumed:

1. Collapsing a spanning tree is a homotopy equivalence of the *surface*: it
   keeps ``V - E`` hence keeps ``chi`` and the face count.  So ``B_next`` has
   exactly ``|P|`` faces and ``|P| - 1`` petals -- it is a spine of the same
   punctured sphere, and its faces can be labelled by the punctures.
2. ``theta`` is peripheral, orientation-sensitively (checked with
   ``allow_reverse=False``), so the face labels transport as a bijection.
3. ``theta . phi_n`` kills exactly the loops ``phi_n`` already kills.  Those are
   the boundary loops of the COLLAPSED faces of ``P_n``: the extra punctures of
   ``S^2 - phi_n^{-1}(P)``.  So "capping" happens on the source side only, where
   it is free, and never on the target side, where it would change the surface.
   For the group statement, thicken ``P_n`` and attach discs along exactly those
   collapsed boundaries.  The resulting surface has the base's puncture count,
   and ``phi_n`` induces a surjection from its fundamental group onto that of
   the base.  Both are free groups of the same finite rank, so the surjection is
   an isomorphism by Hopfianity.  Hence the kernel of ``phi_n`` is precisely the
   normal closure of the collapsed boundary classes; the homotopy equivalence
   ``theta`` adds nothing to that kernel.  The reported face-index equality is
   only bookkeeping, not a computational proof of this normal-closure fact.

What is combinatorially forced and what is a declared choice
-----------------------------------------------------------
* ``|P'| = |P|`` (the active diagonal is a bijection with degree one) is
  REQUIRED for ``B_next`` to be a spine of the same surface.  If a base face is
  unhit, the strict rose has rank ``|P'| - 1 < |P| - 1`` and the equal-rank gate
  inside ``pullback_ve`` refuses the step.  That refusal is the answer, not a
  bug; the prototype exposes it as ``Refusal("target_petal_unhit")`` and the
  test suite drives it deliberately.
* The spanning tree is a real choice.  Different trees give the same certified
  data (ranks, chi, RH, portrait, dynamics of labels) and different chains.
  ``tree_invariance`` reports exactly that, so nobody mistakes the chains for
  canonical.
* The isotopy class of the comparison is *not* determined by the combinatorics.
  A different choice of tree, or the geometric ``upsilon`` of a realised step,
  differs by a mapping-class element.  The combinatorial step therefore pins
  the label algebra and the type, never the isotopy class.

Naming: the object is ``comparison_map`` throughout.  It is not ``upsilon`` in
the sense of the geometric retraction of a realised spine; it is the
combinatorial comparison that a realised step must agree with on labels.
"""

from __future__ import annotations

from pullback.homotopy import compose, peripheral_images
from pullback.lift import advance_comparison
from pullback.maps import make_graph_map
from pullback.portrait import face_map_phi, face_map_pi
from pullback.ribbons import F, reduce_chain, rev
from pullback.spines import collapse_edge
from pullback.validate import is_connected, is_pi1_surjective, spanning_tree

__all__ = [
    "Refusal", "active_diagonal", "collapse_to_rose", "build_comparison_map",
    "identity_map", "one_step", "iterate", "certify_state", "alternative_trees",
    "tree_invariance",
]


class Refusal(ValueError):
    """The combinatorial step is not defined on these data.

    ``reason`` is a stable code, not prose: the caller decides whether a
    refusal stops the iteration or is merely reported.
    """

    def __init__(self, reason, detail=""):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


# --------------------------------------------------------------- stage (a)
def active_diagonal(phi):
    """Diagnose the face map of ``phi``.

    Distinguishes the three cases:

    * ``collapsed``  the boundary walk tightens to nothing.  The face is an
      extra puncture of the source surface, filled in by the pullback; its loop
      is killed by design and needs no geometric input.
    * ``active``     tightens to a rotation of an actual face walk, with degree
      ``degree``.  Only these carry a puncture label forward.
    * ``non_face``   tightens to a non-empty word that is not any face walk.
      That is not a tighter collapse, it is the wrong comparison map; the
      session is refused.

    ``ok`` requires the active diagonal to be a bijection onto the faces with
    degree one: injective (one face per target), exhaustive (every target), no
    winding above one.
    """
    rows, base_faces = face_map_phi(phi)
    active = [i for i, r in enumerate(rows) if r["tight"]]
    collapsed = [i for i, r in enumerate(rows) if not r["tight"]]
    non_face = [i for i in active if rows[i]["target"] is None]
    targets = [rows[i]["target"] for i in active if rows[i]["target"] is not None]
    degrees = {i: rows[i]["degree"] for i in active if rows[i]["target"] is not None}
    winding = sorted({d for d in degrees.values() if d is not None and d > 1})
    out = {
        "rows": rows,
        "base_faces": [tuple(f) for f in base_faces],
        "n_faces_base": len(base_faces),
        "n_faces_cover": len(rows),
        "active": tuple(active),
        "collapsed": tuple(collapsed),
        "non_face": tuple(non_face),
        "targets": tuple(targets),
        "degrees": degrees,
        "injective": len(set(targets)) == len(targets),
        "exhaustive": sorted(set(targets)) == list(range(len(base_faces))),
        "winding_gt_1": tuple(winding),
        "portrait": {i: (rows[i]["target"], rows[i]["degree"]) for i in active},
    }
    out["ok"] = bool(out["injective"] and out["exhaustive"]
                     and not out["winding_gt_1"] and not non_face)
    if non_face:
        out["reason"] = "non_base_face_word"
    elif out["winding_gt_1"]:
        out["reason"] = "active_winding_gt_1"
    elif not out["injective"]:
        out["reason"] = "active_not_injective"
    elif not out["exhaustive"]:
        out["reason"] = "target_petal_unhit"
    else:
        out["reason"] = None
    return out


# --------------------------------------------------------------- stage (b)
def _require_spanning_tree(base, tree):
    """Refuse an edge set that is not a spanning tree of ``base``."""
    for edge in tree:
        if edge not in base.edges:
            raise Refusal("tree_edge_missing", f"{edge!r} is not in the graph")
    if len(set(tree)) != len(tree):
        raise Refusal("not_a_spanning_tree", f"repeated edge in {tree}")
    vertices = base.vertices()
    if len(tree) != len(vertices) - 1:
        raise Refusal("not_a_spanning_tree",
                      f"{len(tree)} edges for {len(vertices)} vertices")
    chosen = set(tree)
    seen = {vertices[0]}
    stack = [vertices[0]]
    while stack:
        vertex = stack.pop()
        for dart in base.flowers[vertex]:
            target = base.tgt(dart)
            if dart[0] in chosen and target not in seen:
                seen.add(target)
                stack.append(target)
    if len(seen) != len(vertices):
        raise Refusal("not_a_spanning_tree",
                      f"the {len(tree)} given edges span {len(seen)} of "
                      f"{len(vertices)} vertices")


def _chain_after(chains, edge, new_chains):
    """Rewrite every word of ``chains`` through the collapse ``new_chains``."""
    out = {}
    for x, word in chains.items():
        acc = []
        for lab, ori in word:
            img = new_chains[lab]
            if ori == F:
                acc.extend(img)
            else:
                acc.extend(rev(d) for d in reversed(img))
        out[x] = reduce_chain(tuple(acc))
    return out


def collapse_to_rose(base, tree=None):
    """Collapse a spanning tree of ``base``; return the strict rose and theta.

    Returns a dict with ``rose`` (one vertex, ``E - V + 1`` petals),
    ``theta`` : ``base -> rose`` as a ``GraphMap``, the ``tree`` used, and
    ``faces_before`` / ``faces_after`` (they must be equal: this is the
    surface-preserving certificate).
    """
    if tree is None:
        tree, gens = spanning_tree(base)
        tree = sorted(tree, key=str)
    else:
        tree = sorted(tree, key=str)
        _require_spanning_tree(base, tree)
        gens = [e for e in base.edge_list() if e not in set(tree)]
    chains = {e: ((e, F),) for e in base.edge_list()}
    cur = base
    steps = []
    for edge in tree:
        if edge not in cur.edge_list():
            raise Refusal("tree_edge_missing", f"{edge!r} is not in the graph")
        before = len(cur.vertices())
        rose, ch = collapse_edge(cur, edge)
        chains = _chain_after(chains, edge, ch)
        steps.append({"edge": edge, "vertices": (before, len(rose.vertices())),
                      "faces": (len(cur.faces()), len(rose.faces()))})
        cur = rose
    if len(cur.vertices()) != 1:
        raise Refusal("not_a_rose",
                      f"the collapse left {len(cur.vertices())} vertices")
    vertex = cur.vertices()[0]
    theta = make_graph_map(base, cur, chains, {v: vertex for v in base.vertices()})
    theta.name = "theta"
    report = {
        "rose": cur,
        "theta": theta,
        "tree": tuple(tree),
        "generators": tuple(gens),
        "steps": steps,
        "faces_before": [tuple(f) for f in base.faces()],
        "faces_after": [tuple(f) for f in cur.faces()],
        "n_petals": len(cur.edge_list()),
        "rank_before": len(base.edge_list()) - len(base.vertices()) + 1,
        "rank_after": len(cur.edge_list()) - len(cur.vertices()) + 1,
    }
    # collapse keeps chi, hence the face count and the rank
    report["faces_preserved"] = (len(report["faces_before"]) ==
                                 len(report["faces_after"]))
    report["rank_preserved"] = report["rank_before"] == report["rank_after"]
    return report


# --------------------------------------------------------------- stage (c)
def identity_map(graph):
    """The identity graph map of ``graph`` onto itself."""
    m = make_graph_map(graph, graph,
                       {e: ((e, F),) for e in graph.edge_list()},
                       {v: v for v in graph.vertices()})
    m.name = "identity"
    return m


def build_comparison_map(base, cover, phi, tree=None, to_original=None):
    """Build ``B_next`` and ``comparison_map : cover -> B_next``.

    ``base``/``cover``/``phi`` are a state of the VE: ``phi : cover -> base``.
    ``to_original`` is the accumulated comparison ``base -> Gamma_0`` (identity
    when the state is the session itself); it is only used to label the new
    faces with the puncture labels of the *input*, so a step can be checked
    against the labels it started from.

    Refuses (``Refusal``) unless the active diagonal is a bijection of degree
    one, in which case ``B_next`` is a spine of the same surface.  The
    ``comparison_map`` is ``theta . phi``.
    """
    diag = active_diagonal(phi)
    if diag["non_face"]:
        raise Refusal("non_base_face_word",
                      f"cover faces {list(diag['non_face'])} tighten to words "
                      "that are not face walks; that is not a tighter collapse")
    if diag["winding_gt_1"]:
        raise Refusal("active_winding_gt_1", f"degrees {sorted(diag['winding_gt_1'])}")
    if not diag["injective"]:
        raise Refusal("active_not_injective",
                      f"targets {list(diag['targets'])}")
    if not diag["exhaustive"]:
        missing = sorted(set(range(diag["n_faces_base"])) - set(diag["targets"]))
        raise Refusal(
            "target_petal_unhit",
            f"base faces {missing} are not the image of any cover face; the "
            "active rose would have rank |P'|-1 < |P|-1 and is not a spine of "
            "the same punctured sphere, so the petal cannot be capped")

    collapse = collapse_to_rose(base, tree=tree)
    rose, theta = collapse["rose"], collapse["theta"]
    if not (collapse["faces_preserved"] and collapse["rank_preserved"]):
        raise Refusal("collapse_changed_the_surface", str(collapse["steps"]))

    # the face labels: base face -> rose face, orientation-sensitive
    labels = peripheral_images(theta, allow_reverse=False)
    if None in labels or sorted(labels) != list(range(len(rose.faces()))):
        raise Refusal(
            "collapse_not_peripheral",
            f"face images {labels}; a tree collapse is a homeomorphism of the "
            "punctured surface and must permute its boundary classes")
    # pullback_ve's own gate, quoted so the difference is visible
    loose = peripheral_images(theta, allow_reverse=True)

    comparison = compose(theta, phi)
    comparison.name = "comparison_map"

    # per-face description of the cover: capped or the labelled rose face
    face_rows = peripheral_images(comparison, allow_reverse=False)
    capped = [f for f, img in enumerate(face_rows) if img is None]
    described = {}
    for f, row in enumerate(diag["rows"]):
        if not row["tight"]:
            described[f] = {"kind": "capped", "image": (), "label": None}
        else:
            tgt = row["target"]
            described[f] = {"kind": "active", "image": rose.faces()[labels[tgt]],
                            "label": tgt, "rose_face": labels[tgt],
                            "raw": tuple(row["raw"]), "tight": tuple(row["tight"])}

    rank_cover = len(cover.edge_list()) - len(cover.vertices()) + 1
    rank_rose = collapse["rank_after"]
    report = {
        "B_next": rose,
        "comparison_map": comparison,
        "theta": theta,
        "diagonal": diag,
        "collapse": collapse,
        "labels": {rose_face: base_face for base_face, rose_face in
                   enumerate(labels)},
        "labels_reverse": tuple(labels),
        "peripheral_strict": tuple(labels),
        "peripheral_loose": tuple(loose),
        "faces": described,
        "n_petals": collapse["n_petals"],
        "rank_cover": rank_cover,
        "rank_rose": rank_rose,
        "collapsed_faces": tuple(diag["collapsed"]),
        "active_faces": tuple(diag["active"]),
    }
    report["capped_faces"] = tuple(capped)
    report["capping_exact"] = tuple(capped) == tuple(diag["collapsed"])
    report["rank_gap"] = rank_cover - rank_rose
    report["rank_gap_ok"] = report["rank_gap"] == len(diag["collapsed"])
    report["petals_ok"] = report["n_petals"] == diag["n_faces_base"] - 1
    report["faces_labelled"] = sorted(report["labels"]) == \
        list(range(len(rose.faces())))
    report["origin_labels"] = None
    if to_original is not None:
        origin = peripheral_images(to_original, allow_reverse=False)
        report["origin_labels"] = tuple(origin)
    report["ok"] = bool(report["capping_exact"] and report["rank_gap_ok"]
                        and report["petals_ok"] and report["faces_labelled"])
    return report


# ---------------------------------------------------------------- the state
def _dynamics(base, cover, phi):
    """``{target_puncture: source_puncture}`` from the two portraits.

    Same read as ``ve_state.VEInit`` but without positions: the cover face's
    phi-target face and its pi-target face are both actual faces here.
    """
    phi_rows, _ = face_map_phi(phi)
    pi_rows, _ = face_map_pi(cover, base)
    dyn = {}
    for f, row in enumerate(phi_rows):
        if not row["tight"]:
            continue
        p, q = row["target"], pi_rows[f]["target"]
        if p in dyn and dyn[p] != q:
            raise Refusal("label_over_two_faces", f"label {p}: {dyn[p]} and {q}")
        dyn[p] = q
    return dict(sorted(dyn.items()))


def certify_state(session, state, expect=None):
    """Exact certificates on a promoted state.

    Everything here is sphere/Riemann-Hurwitz/fibre bookkeeping that
    ``pullback_ve`` does NOT do: it only checks the two ``h`` gates.  The
    monodromy cover it builds already certifies that every cover dart projects
    to a base dart and that the sheet permutations are well defined; it does
    not certify that the result is a spine of a punctured sphere.
    """
    spine = state["new_spine"] if "new_spine" in state else state["spine"]
    P = state["P"]
    pi_p = state["pi_prime"]
    phi_p = state["phi_prime"]
    base = session.base
    d = len({sh for (_v, sh) in session.cover.vertices()})
    n_base_faces = len(base.faces())
    rep = {}
    rep["d"] = d
    rep["spine_rank"] = len(spine.edge_list()) - len(spine.vertices()) + 1
    rep["spine_faces"] = len(spine.faces())
    rep["rank_is_P_minus_1"] = rep["spine_rank"] == n_base_faces - 1
    rep["faces_is_P"] = rep["spine_faces"] == n_base_faces
    rep["spine_chi"] = (len(spine.vertices()) - len(spine.edge_list())
                        + len(spine.faces()))
    # the new cover, as a surface
    V, E, F_n = len(P.vertices()), len(P.edge_list()), len(P.faces())
    rep["cover_chi"] = V - E + F_n
    rep["cover_connected"] = is_connected(P)
    rep["cover_is_sphere"] = rep["cover_chi"] == 2
    rep["cover_punctures"] = F_n
    rep["punctured_chi"] = 2 - F_n
    rep["punctured_chi_expected"] = d * (2 - n_base_faces)
    rep["punctured_chi_ok"] = rep["punctured_chi"] == rep["punctured_chi_expected"]
    # Riemann-Hurwitz in the covering direction: the local degrees over each
    # spine face sum to the cover degree
    rows, base_walks = face_map_pi(P, spine)
    over = {}
    for row in rows:
        if row["target"] is None:
            raise Refusal("new_cover_not_a_covering",
                          f"a face of {P!r} does not project to a face power")
        over.setdefault(row["target"], []).append(row["degree"])
    rep["fibre_degrees"] = {k: tuple(v) for k, v in sorted(over.items())}
    rep["fibre_sums"] = {k: sum(v) for k, v in sorted(over.items())}
    rep["riemann_hurwitz_ok"] = (sorted(over) == list(range(len(base_walks)))
                                 and all(s == d for s in rep["fibre_sums"].values()))
    # face_map_pi above reads the cover's edge labels, not the exported map.
    # Certify that pi_prime is actually that monodromy projection as well.
    rep["pi_prime_is_projection"] = (
        all(pi_p.chains.get(e) == ((e[0], F),) for e in P.edge_list())
        and all(pi_p.image_vertex(v) == v[0] for v in P.vertices()))
    # the new VE
    rep["phi_pi1_surjective"] = is_pi1_surjective(phi_p, spine)
    try:
        diag = active_diagonal(phi_p)
        rep["next_diagonal_ok"] = diag["ok"]
        rep["next_diagonal_reason"] = diag["reason"]
        rep["next_collapsed"] = len(diag["collapsed"])
    except Exception as exc:            # pragma: no cover - diagnosis only
        rep["next_diagonal_ok"] = False
        rep["next_diagonal_reason"] = f"exception: {exc}"
    # labels carried by the accumulated comparison
    to_orig = peripheral_images(state["to_original"], allow_reverse=False)
    from_orig = peripheral_images(state["from_original"], allow_reverse=False)
    rep["to_original_labels"] = tuple(to_orig)
    rep["from_original_labels"] = tuple(from_orig)
    rep["to_original_is_bijection"] = (None not in to_orig and
                                       sorted(to_orig) ==
                                       list(range(n_base_faces)))
    rep["from_original_is_bijection"] = (None not in from_orig and
                                         sorted(from_orig) ==
                                         list(range(len(spine.faces()))))
    rep["inverse_permutations"] = (rep["to_original_is_bijection"]
                                   and rep["from_original_is_bijection"]
                                   and all(from_orig[to_orig[k]] == k
                                           for k in range(len(spine.faces()))))
    # The old and new labels live on different faces, so compare them through
    # the comparison's face bijection rather than equating the permutations.
    rep["labels_preserved"] = None
    if expect and "labels_in" in expect:
        rep["labels_preserved"] = False
        if "comparison" in state:
            zeta = peripheral_images(state["comparison"], allow_reverse=False)
            want = tuple(expect["labels_in"])
            rep["labels_preserved"] = bool(
                None not in zeta and None not in to_orig
                and len(zeta) == len(want)
                and all(0 <= zeta[p] < len(to_orig)
                        and want[p] == to_orig[zeta[p]]
                        for p in range(len(want))))
    # dynamics, read in the input labels
    dyn = _dynamics(spine, P, phi_p)
    rep["dynamics_raw"] = dyn
    if rep["to_original_is_bijection"]:
        # a spine face s carries input label to_orig[s]; the raw dynamics is on
        # spine faces, so transport it to input labels
        by_input = {}
        for p, q in dyn.items():
            by_input[to_orig[p]] = to_orig[q]
        rep["dynamics_in_input_labels"] = dict(sorted(by_input.items()))
        if expect and "dynamics_in" in expect:
            rep["dynamics_preserved"] = (rep["dynamics_in_input_labels"] ==
                                         expect["dynamics_in"])
    rep["ok"] = bool(rep["rank_is_P_minus_1"] and rep["faces_is_P"]
                     and rep["cover_is_sphere"] and rep["cover_connected"]
                     and rep["punctured_chi_ok"] and rep["riemann_hurwitz_ok"]
                     and rep["phi_pi1_surjective"] and rep["next_diagonal_ok"]
                     and rep["pi_prime_is_projection"]
                     and rep["to_original_is_bijection"]
                     and rep["from_original_is_bijection"]
                     and rep["inverse_permutations"]
                     and rep["labels_preserved"] is not False
                     and rep.get("dynamics_preserved", True))
    return rep


def one_step(session, tree=None, certify=True, state=None):
    """One comparison-preserving combinatorial step.

    With ``state=None`` the stage-0 state is the session itself:
    ``B_0 = session.base``, ``P_0 = session.cover``, ``phi_0 = session.phi``, and
    ``to_original`` the identity, so nothing about the input has been forgotten
    yet.  Otherwise ``state`` is a previous output of this function and the step
    runs on ``B_n = state['new_spine']``, ``P_n = state['P']``,
    ``phi_n = state['phi_prime']``, with ``to_original`` the accumulated
    comparison.  The ``session`` argument is always the ORIGINAL input: the new
    cover is the pullback of the original ``pi`` along the accumulated
    comparison, which is what preserves the VE.
    """
    if state is None:
        base, cover, phi = session.base, session.cover, session.phi
        spine = session.base
        to_original = identity_map(base)
    else:
        base, cover, phi = state["new_spine"], state["P"], state["phi_prime"]
        spine = state["new_spine"]
        to_original = state["to_original"]
    if not is_pi1_surjective(phi, base):
        raise Refusal("phi_not_pi1_surjective",
                      "no homotopy section exists, so no comparison can be promoted")
    build = build_comparison_map(base, cover, phi, tree=tree,
                                 to_original=to_original)
    labels_in = peripheral_images(to_original, allow_reverse=False)
    if None in labels_in or len(set(labels_in)) != len(labels_in):
        raise Refusal("labels_not_a_bijection",
                      f"accumulated comparison has face images {labels_in}")
    # the invariant the iteration must preserve: the puncture dynamics read in
    # the labels of the ORIGINAL input, not in the current spine's face order
    dyn_in = {labels_in[p]: labels_in[q]
              for p, q in _dynamics(base, cover, phi).items()}
    dyn_in = dict(sorted(dyn_in.items()))
    promoted = advance_comparison(session, spine, cover, phi, to_original,
                                  build["B_next"], build["comparison_map"])
    out = {
        "build": build,
        "state": promoted,
        "new_spine": promoted["new_spine"],
        "P": promoted["P"],
        "pi_prime": promoted["pi_prime"],
        "phi_prime": promoted["phi_prime"],
        "to_original": promoted["to_original"],
        "from_original": promoted["from_original"],
        "comparison": promoted["comparison"],
        "comparison_inverse": promoted["comparison_inverse"],
    }
    if certify:
        out["certificates"] = certify_state(
            session, promoted,
            expect={"labels_in": labels_in, "dynamics_in": dyn_in})
        out["certificates"]["dynamics_in"] = dyn_in
        out["certificates"]["ok"] = bool(
            out["certificates"]["ok"]
            and out["certificates"].get("dynamics_preserved", False))
    return out


def iterate(session, steps=3, tree=None):
    """``steps`` combinatorial steps from the session state, all certified.

    The invariant asserted at every step is the dynamics of the punctures read
    through the accumulated comparison: it must be the same map on the input
    labels at every stage.  The spine presentation is allowed (and expected) to
    change only at the first step: after it ``B_next`` is a rose, so every later
    collapse is the identity and ``comparison_map`` is ``phi_n`` itself.

    On every session in the current ribbon library the second call reproduces
    the first carried state, so the transition is already stationary.  Further
    steps re-certify that same state; they do not probe new combinatorics and
    are not convergence evidence.  The test suite pins this measured boundary.
    """
    out = []
    cur = None
    for k in range(steps):
        # The declared tree belongs to the original base; later bases are roses.
        record = one_step(session, tree=(tree if k == 0 else None), state=cur)
        record["index"] = k
        out.append(record)
        cur = record
    return {"steps": out, "terminal": cur}


def alternative_trees(base, limit=6):
    """Genuinely different spanning trees of ``base``.

    Permuting one tree's edge order gives the same collapse (collapsing a whole
    tree is order-independent), so the trees have to differ.  Each candidate
    swaps one tree edge of the fundamental cycle of a generator for that
    generator.
    """
    from pullback.validate import _tree_path
    tree, gens = spanning_tree(base)
    out = [sorted(tree, key=str)]
    for g in sorted(gens, key=str):
        s, t = base.edges[g]
        on_cycle = {d[0] for d in _tree_path(base, tree, s, t)} & set(tree)
        for e in sorted(on_cycle, key=str):
            cand = sorted((set(tree) - {e}) | {g}, key=str)
            if cand not in out:
                out.append(cand)
            if len(out) >= limit:
                return out
    return out


def tree_invariance(session, limit=6):
    """Run the step over different spanning trees and separate the two layers.

    ``invariant``  what the combinatorics pins: rank, chi, puncture count, the
    fibre degrees and the dynamics read in the input labels.  These agree
    across trees, and they are the numbers a numerical step must also
    reproduce.
    ``presentation`` what the declared tree moves: the chain table of
    ``comparison_map`` and the accumulated label permutation ``to_original``.
    Those are the isotopy-class content, i.e. exactly the part a geometric
    step supplies for itself.
    """
    trees = alternative_trees(session.base, limit=limit) or [[]]
    reports = []
    for tree in trees:
        try:
            out = one_step(session, tree=tree)
        except Refusal as exc:
            reports.append({"tree": tuple(tree), "refused": exc.reason})
            continue
        c = out["certificates"]
        reports.append({
            "tree": tuple(tree),
            "ok": c["ok"],
            "presentation": {
                "chains": tuple(sorted(
                    (str(k), tuple(tuple(map(str, d)) for d in v))
                    for k, v in out["build"]["comparison_map"].chains.items())),
                "to_original": tuple(c["to_original_labels"]),
                "peripheral_strict": tuple(out["build"]["peripheral_strict"]),
            },
            "invariant": {
                "rank": c["spine_rank"],
                "faces": c["spine_faces"],
                "chi": c["cover_chi"],
                "punctures": c["cover_punctures"],
                "fibre_sums": tuple(sorted(c["fibre_sums"].items())),
                "dynamics": tuple(sorted(c["dynamics_in_input_labels"].items())),
            },
        })
    invariants = {repr(r["invariant"]) for r in reports if "invariant" in r}
    chains = {repr(r["presentation"]["chains"]) for r in reports
              if "presentation" in r}
    ok = [r["ok"] for r in reports if "ok" in r]
    return {"reports": reports, "n_trees": len(reports),
            "same_invariant": len(invariants) == 1,
            "chains_differ": len(chains) > 1,
            "all_ok": bool(ok) and all(ok)}
