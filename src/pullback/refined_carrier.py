"""Graph-map carrier for the split ``RefinedVE`` cover and inclusion legs.

The geometry module refines a *triangulated* cover by splitting the ramified
base edge.  The additional auxiliary midpoint is not a dynamical puncture.
Consequently the required virtual endomorphism is not on the full dual of the
refined base (which has one extra peripheral class): it is obtained by the
explicit split collapses back to the ordinary four-puncture spine.

This module materialises those collapses, rather than searching for graph-map
chains.  A parent is recomputed from triangle vertices for every canonical dual
vertex.  Every unchanged edge keeps its label, each split half keeps its parent
label, and every chord contracts inside its parent triangle.  The resulting
cover and inclusion legs return to ``refined.init.spine`` and ``refined.init.P``
through canonical dual relabelling isomorphisms.  The certificates below check
that construction on every edge and on every peripheral class.
"""

from __future__ import annotations

from . import refined_cover as rr
from .homotopy import compose, peripheral_images
from .maps import make_graph_map
from .portrait import face_map_phi
from .ribbons import F, B
from .validate import is_pi1_surjective

__all__ = ["face_punctures", "refined_carrier"]


def _pairs(tris):
    return [(tuple(tr[0]), tuple(tr[1])) for tr in tris]


def _tri_of_face(tris, edges):
    """Input-triangle index of every canonical face of ``tri_graph``."""
    graph = rr.tri_graph(tris, edges)
    out = []
    for walk in graph.faces():
        vertices = set()
        for dart in walk:
            vertices.update(edges[dart[0]])
        hits = [k for k, tr in enumerate(tris) if set(tr[0]) == vertices]
        if len(hits) != 1:
            raise ValueError(
                f"canonical dual face {tuple(sorted(vertices))} matches "
                f"{len(hits)} triangles")
        out.append(hits[0])
    return out


def face_punctures(graph, edges):
    """Primal vertex enclosed by each face of a dual ribbon graph."""
    out = {}
    for face, walk in enumerate(graph.faces()):
        common = None
        for dart in walk:
            ends = set(edges[dart[0]])
            common = ends if common is None else common & ends
        if common is None or len(common) != 1:
            raise ValueError(f"dual face {face} does not enclose one vertex")
        puncture = next(iter(common))
        if puncture in out.values():
            raise ValueError(f"two dual faces enclose vertex {puncture}")
        out[face] = puncture
    return out


def _polygon_vertex(cells):
    """Source-graph vertex about each polygon of ``dual_cells(P)``."""
    graph = cells["graph"]
    out = []
    for walk in cells["corners"]:
        common = None
        for dart in walk:
            ends = {graph.src(dart), graph.tgt(dart)}
            common = ends if common is None else common & ends
        if common is None or len(common) != 1:
            raise ValueError("a dual polygon is not about one source vertex")
        out.append(next(iter(common)))
    return out


def _dual_isomorphism(tris, edges, target, polygon_vertices=None):
    """Canonical dual labelling -> the original same-dart graph."""
    source = rr.dual_of(tris, edges)
    order = _tri_of_face(tris, edges)
    if polygon_vertices is None:
        # The base triangles were created in mesh.delaunay_faces order, which
        # is precisely mesh.voronoi_spine's vertex order.
        images = {v: order[v] for v in source.vertices()}
    else:
        images = {v: polygon_vertices[order[v]] for v in source.vertices()}
    chains = {}
    for edge in source.edge_list():
        a, b = source.edges[edge]
        for ori in (F, B):
            dart = (edge, ori)
            if target.src(dart) == images[a] and target.tgt(dart) == images[b]:
                chains[edge] = (dart,)
                break
        else:
            raise ValueError(f"no target dart for dual isomorphism edge {edge!r}")
    return make_graph_map(source, target, chains, images)


def _parents(old_tris, old_edges, new_tris, auxiliary):
    """Old triangle index of every new triangle, including split children."""
    out = []
    for tr in new_tris:
        vertices = set(tr[0])
        hits = [k for k, old in enumerate(old_tris)
                if set(old[0]) == vertices]
        if not hits:
            midpoint = list(vertices & set(auxiliary))
            if len(midpoint) != 1:
                raise ValueError(f"new triangle {tr[0]} has no unique split parent")
            mid = midpoint[0]
            core = vertices - {mid}
            split_vertices = set(old_edges[auxiliary[mid]])
            hits = [k for k, old in enumerate(old_tris)
                    if core < set(old[0]) and split_vertices <= set(old[0])]
        if len(hits) != 1:
            raise ValueError(f"triangle {tr[0]} has {len(hits)} split parents")
        out.append(hits[0])
    return out


def _split_collapse(old_tris, old_edges, new_tris, new_edges, target, auxiliary,
                    source=None):
    """Definitional collapse of a split dual onto the old dual."""
    if source is None:
        source = rr.dual_of(new_tris, new_edges)
    old_source = rr.dual_of(old_tris, old_edges)
    old_order = _tri_of_face(old_tris, old_edges)
    new_order = _tri_of_face(new_tris, new_edges)
    old_vertex = {old_order.index(k): k for k in range(len(old_tris))}
    parents = _parents(old_tris, old_edges, new_tris, auxiliary)
    images = {v: old_vertex[parents[new_order[v]]] for v in source.vertices()}
    chains = {}
    for edge in source.edge_list():
        a, b = source.edges[edge]
        if images[a] == images[b]:
            chains[edge] = ()
            continue
        parent = edge[1] if isinstance(edge, tuple) and edge and edge[0] == "half" else edge
        if parent not in old_source.edges:
            midpoint = [u for u in parent if u in auxiliary]
            if len(midpoint) != 1:
                raise ValueError(f"cannot identify parent of split edge {edge!r}")
            parent = auxiliary[midpoint[0]]
        for ori in (F, B):
            dart = (parent, ori)
            if old_source.src(dart) == images[a] and old_source.tgt(dart) == images[b]:
                chains[edge] = (dart,)
                break
        else:
            raise ValueError(f"split edge {edge!r} has no parent dart")
    return make_graph_map(source, target, chains, images)


def _direct_cover(refined, source=None):
    """The corner-slot cover ``dual(K2) -> dual(T2)`` as a graph map."""
    if source is None:
        source = rr.dual_of(refined.K2["tris"], refined.K2["edges"])
    target = rr.dual_of(refined.T2["tris"], refined.T2["edges"])
    base_order = _tri_of_face(refined.T2["tris"], refined.T2["edges"])
    base_faces = [set(tr[0]) for tr in refined.T2["tris"]]
    images = {}
    for v, k in enumerate(_tri_of_face(refined.K2["tris"], refined.K2["edges"])):
        vertices = {refined.K2["proj"][u] for u in refined.K2["tris"][k][0]}
        hits = [j for j, face in enumerate(base_faces) if vertices == face]
        if len(hits) != 1:
            raise ValueError(f"lifted face {k} projects to {len(hits)} base faces")
        images[v] = base_order.index(hits[0])
    chains = {}
    for edge in source.edge_list():
        label = refined.K2["base_of"][edge]
        a, b = source.edges[edge]
        for ori in (F, B):
            dart = (label, ori)
            if target.src(dart) == images[a] and target.tgt(dart) == images[b]:
                chains[edge] = (dart,)
                break
        else:
            raise ValueError(f"no base dart below lifted edge {edge!r}")
    return make_graph_map(source, target, chains, images)


def refined_carrier(refined):
    """Return the certified split VE on the ordinary current geometric spine.

    ``pi_prime`` and ``phi_prime`` have the same source ``lifted`` and target
    ``spine``.  The source is the refined lifted dual ``dual(K2)``; the target
    has no artificial auxiliary puncture.  ``comparison`` is valid input for
    :func:`pullback.lift.advance_comparison`.  Construction is deterministic
    from ``RefinedVE``'s corner slots and needs no transport census.
    """
    init = refined.init
    base_tris = _pairs(refined.T)
    cover_tris = _pairs(refined.K_tris)
    base_iso = _dual_isomorphism(base_tris, refined.edgesT, init.spine)
    cover_iso = _dual_isomorphism(
        cover_tris, refined.K_edges, init.P, _polygon_vertex(init.cells))
    auxiliary_base = {refined.m: refined.split_edge}
    auxiliary_cover = {rec["midpoint"]: edge
                       for edge, rec in refined.K2["splits"].items()}
    lifted = rr.dual_of(refined.K2["tris"], refined.K2["edges"])
    base_collapse = _split_collapse(
        base_tris, refined.edgesT, refined.T2["tris"], refined.T2["edges"],
        base_iso.source, auxiliary_base)
    cover_collapse = _split_collapse(
        cover_tris, refined.K_edges, refined.K2["tris"], refined.K2["edges"],
        cover_iso.source, auxiliary_cover, source=lifted)
    direct = _direct_cover(refined, source=lifted)
    pi_prime = compose(base_iso, compose(base_collapse, direct))
    collapsed_pi = compose(init.pi_prime, compose(cover_iso, cover_collapse))
    phi_prime = compose(init.phi_prime, compose(cover_iso, cover_collapse))

    # These are edge-by-edge identities, not merely equality on a basis.
    commuting = all(pi_prime.chains[e] == collapsed_pi.chains[e]
                    for e in pi_prime.source.edge_list())
    if not commuting or pi_prime.vertex_images != collapsed_pi.vertex_images:
        raise ValueError("refined cover does not commute with the coarse cover")
    if not is_pi1_surjective(phi_prime, init.spine):
        raise ValueError("refined inclusion leg is not pi_1-surjective")

    punctures = face_punctures(pi_prime.source, refined.K2["edges"])
    point_to_face = {p: f for f, p in init.face_points.items()}
    expected_peripheral = [
        point_to_face[refined.target[u]] if u in refined.target else None
        for _face, u in sorted(punctures.items())]
    actual_peripheral = peripheral_images(phi_prime)
    if actual_peripheral != expected_peripheral:
        raise ValueError(
            f"refined inclusion portrait {actual_peripheral} != "
            f"{expected_peripheral}")
    phi_rows, _base_walks = face_map_phi(phi_prime)
    for face, expected in enumerate(expected_peripheral):
        if expected is None:
            if phi_rows[face]["tight"]:
                raise ValueError(f"inactive source face {face} has a nonempty image")
        elif (phi_rows[face]["target"], phi_rows[face]["degree"]) != (expected, 1):
            raise ValueError(
                f"source face {face} has oriented target/degree "
                f"{(phi_rows[face]['target'], phi_rows[face]['degree'])}, "
                f"expected {(expected, 1)}")
    aux_vertices = set(auxiliary_cover)
    auxiliary_faces = [face for face, u in punctures.items() if u in aux_vertices]
    if any(actual_peripheral[f] is not None for f in auxiliary_faces):
        raise ValueError("an auxiliary lifted puncture was not collapsed")

    base_punctures = face_punctures(direct.target, refined.T2["edges"])
    base_images = peripheral_images(base_collapse)
    aux_base_faces = [face for face, u in base_punctures.items() if u == refined.m]
    if len(aux_base_faces) != 1 or base_images[aux_base_faces[0]] is not None:
        raise ValueError("the auxiliary base point does not collapse")

    complex_certificate = refined.certificates()
    if not (complex_certificate["cover2"]["ok"] and complex_certificate["ribbon2_ok"]):
        raise ValueError("the refined complex failed its cover/ribbon certificate")
    source = pi_prime.source
    return {
        "lifted": source,
        "spine": init.spine,
        "pi_prime": pi_prime,
        "phi_prime": phi_prime,
        "refined_base": direct.target,
        "direct_cover": direct,
        "base_isomorphism": base_iso,
        "cover_isomorphism": cover_iso,
        "base_collapse": base_collapse,
        "cover_collapse": cover_collapse,
        "collapsed_cover": collapsed_pi,
        "source_face_punctures": punctures,
        "base_face_punctures": base_punctures,
        "certificates": {
            "commuting_edges": len(source.edge_list()),
            "commuting_vertex_images": True,
            "pi1_surjective": True,
            "cover2": complex_certificate["cover2"],
            "ribbon2_ok": complex_certificate["ribbon2_ok"],
            "peripheral_images": actual_peripheral,
            "expected_peripheral": expected_peripheral,
            "phi_degrees": [row["degree"] for row in phi_rows],
            "active_source_faces": {
                f: refined.target[u] for f, u in punctures.items()
                if u in refined.target},
            "auxiliary_source_faces": auxiliary_faces,
            "inactive_source_faces": [f for f, image in enumerate(actual_peripheral)
                                      if image is None],
            "auxiliary_base_faces": aux_base_faces,
            "counts": [len(source.vertices()), len(source.edge_list()),
                       len(source.faces())],
        },
    }
