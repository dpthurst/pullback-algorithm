"""Transport a graph virtual endomorphism to a geometrically chosen spine.

h_inverse : new_spine -> base specifies the geometric comparison.  Chain lifts
build the fibre-product cover P and its two legs.  The transported inclusion leg
needs BOTH comparison directions:

    h_inverse : new_spine -> base     (pull back the cover),
    h         : base -> new_spine     (retarget phi),

so phi_prime = h . phi . proj_to_cover lands on new_spine, just as pi_prime does.
When h is omitted, its equal-rank homotopy inverse is computed by the based
section algorithm.  Peripheral and inverse-on-pi_1 checks reject arbitrary
basis transports, but the actual geometric isotopy class remains an input: an
unmarked abstract session does not determine it.

The native ribbon dual gives the lifted cell decomposition.  Its labelled darts
keep parallel edges distinct, the metric inherits base Delaunay lengths with
local-degree cone angles, and the solver boundary refuses complexes outside
round_realise's representation; large input cone angles are allowed by discrete uniformisation.
"""

from __future__ import annotations

import numpy as np

from .covers import monodromy_cover
from .homotopy import (compose, homotopy_section,
                       is_peripheral_preserving)
from .maps import GraphMap
from .portrait import face_map_phi, face_map_pi
from .ribbons import F
from .validate import is_pi1_surjective, rose_words, spanning_tree

__all__ = ["lift_chain", "pullback_ve", "advance_comparison", "dual_cells",
           "dual_polygons", "parallel_edge_pairs", "lifted_delaunay_metric",
           "active_puncture_labels", "round_realise_cells"]


def lift_chain(cover, start, chain):
    """Lift a base chain of darts starting from the cover vertex `start`.

    A base dart (label, ori) is matched against cover darts: the label's first
    component is the base label and the orientation must agree.  Returns the
    final cover vertex and the lifted chain of cover darts.
    """
    cur = start
    out = []
    for (lab, ori) in chain:
        for d in cover.flowers[cur]:
            if d[0][0] == lab and d[1] == ori:
                out.append(d)
                cur = cover.tgt(d)
                break
        else:
            raise KeyError(f"no lift of {(lab, ori)} at vertex {cur}")
    return cur, tuple(out)


def _is_identity_on_pi1(graph_map, graph):
    """Check a based endomorphism on a free spanning-tree basis."""
    if graph_map.image_vertex(graph.vertices()[0]) != graph.vertices()[0]:
        return False
    tree, gens = spanning_tree(graph)
    got = [tuple((gens.index(d[0]), d[1]) for d in word)
           for word in rose_words(graph_map, tree, tree, gens)]
    return got == [((i, F),) for i in range(len(gens))]


def pullback_ve(base, cover, phi, new_spine, h_inverse, h=None,
                validate_h=True):
    """The pulled-back virtual endomorphism on new_spine (the reduced spine).

    `h_inverse : new_spine -> base` must be the geometric comparison supplied by
    the punctured-surface construction, not an arbitrary free-basis transport.
    By default verify two necessary conditions: equal-rank pi_1-surjectivity
    (hence a homotopy equivalence), and bijection of peripheral face classes.
    The latter is what rejects the original rabbit's arbitrary basis transport.
    Passing the checks is not by itself a proof that the map is the intended
    isotopy class.  `validate_h=False` exists for negative structural fixtures,
    not for a production pullback.

    BOTH directions are needed.  The inverse `h_inverse` pulls back the cover;
    after that, `h : base -> new_spine` retargets the other leg.  If `h` is not
    supplied, the equal-rank homotopy inverse is computed by `homotopy_section`.
    Omitting this second composition was the reason the WIP's `phi_prime` still
    lived on the old base and its face images were usually not face walks.

    Returns (P, pi_prime, phi_prime, proj_to_cover), where pi_prime is the
    induced cover of new_spine and phi_prime = h . phi . proj_to_cover is the
    other leg on that same new spine.
    """
    if validate_h:
        if h_inverse.source is not new_spine or h_inverse.target is not base:
            raise ValueError("h_inverse has the wrong source or target graph")
        source_rank = len(new_spine.edge_list()) - len(new_spine.vertices()) + 1
        target_rank = len(base.edge_list()) - len(base.vertices()) + 1
        if source_rank != target_rank:
            raise ValueError("h_inverse does not join graphs of equal rank")
        if not is_pi1_surjective(h_inverse, base):
            raise ValueError("h_inverse is not pi_1-surjective")
        if not is_peripheral_preserving(h_inverse):
            raise ValueError(
                "h_inverse does not permute peripheral face classes; arbitrary "
                "basis transport is not the geometric comparison of a punctured sphere")
    sheets = tuple(sorted({sh for (_v, sh) in cover.vertices()}, key=str))

    # ---------------------------------------------------------------- monodromy
    permutations = {}
    for e in new_spine.edge_list():
        chain = list(h_inverse.chains[e])
        if not chain:
            permutations[e] = {}
            continue
        head_of_chain = base.src(chain[0])
        per = {}
        for sh in sheets:
            # A path between arbitrary base vertices lifts from its actual
            # source, not from an artificial global basepoint.
            end_vertex, _darts = lift_chain(cover, (head_of_chain, sh), chain)
            per[sh] = end_vertex[1]
        permutations[e] = per

    P = monodromy_cover(new_spine, sheets, permutations)

    # ------------------------------------------------------------- projections
    pc_chains = {}
    for e in P.edge_list():
        label, sheet = e          # cover edge label is (base edge label, sheet)
        ch = list(h_inverse.chains[label])
        if not ch:
            pc_chains[e] = ()
            continue
        head_of_chain = base.src(ch[0])
        _end, darts = lift_chain(cover, (head_of_chain, sheet), ch)
        pc_chains[e] = darts

    # vertex images: (v, sheet) in P goes to (rho(v), sheet) in the cover
    pc_vertices = {}
    for (v, sh) in P.vertices():
        w = h_inverse.image_vertex(v) if v in h_inverse.vertex_images else None
        if w is None:
            raise ValueError(f"no base image for new_spine vertex {v}")
        pc_vertices[(v, sh)] = (w, sh)

    proj_to_cover = GraphMap(P, cover, pc_chains, pc_vertices)
    proj_to_cover.src, proj_to_cover.tgt = P, cover
    proj_to_cover.name = "proj_to_cover"

    pi_prime = GraphMap(P, new_spine,
                        {e: ((e[0], 1),) for e in P.edge_list()},
                        {(v, sh): v for (v, sh) in P.vertices()})
    pi_prime.src, pi_prime.tgt = P, new_spine
    pi_prime.name = "pi_prime"

    if h is None:
        # h_inverse : new_spine -> base is surjective on pi_1.  Section its
        # surjection to get h : base -> new_spine with h_inverse . h = id_base;
        # equal ranks make this a two-sided homotopy inverse.
        h = homotopy_section(base, new_spine, h_inverse)["psi"]
    if validate_h:
        if h.source is not base or h.target is not new_spine:
            raise ValueError("h has the wrong source or target graph")
        if not _is_identity_on_pi1(compose(h_inverse, h), base):
            raise ValueError("h_inverse . h is not the identity on pi_1(base)")
        if not _is_identity_on_pi1(compose(h, h_inverse), new_spine):
            raise ValueError("h . h_inverse is not the identity on pi_1(new_spine)")
    phi_prime = compose(h, compose(phi, proj_to_cover))
    phi_prime.src, phi_prime.tgt = P, new_spine
    phi_prime.name = "phi_prime"
    return P, pi_prime, phi_prime, proj_to_cover


def advance_comparison(session, spine, lifted, phi_prime, to_original,
                       new_spine, upsilon):
    """Transport a VE to `new_spine`, preserving its comparison to the input.

    This is the comparison algebra of `notes/spine-reformulation.md`.  `spine`
    is the current geometric spine, `lifted` and `phi_prime` are its cover and
    inclusion leg, and `to_original : spine -> session.base` is the *accumulated*
    comparison with the original input graph, not the virtual endomorphism on
    pi_1.  `upsilon : lifted -> new_spine` is the retraction-induced comparison
    supplied by the geometric output.

    If psi is a homotopy section of phi_prime, set k = upsilon . psi.  Then

        from_original_new = k . from_original,
        to_original_new   = to_original . k^{-1}.

    Both directions are retained explicitly so no later step has to reconstruct
    or silently reset the relation to the input Gamma_0.  The pair is supplied to
    `pullback_ve`, whose validation checks the two inverse-on-pi_1 identities.
    The geometric construction of upsilon is deliberately outside this function.
    """
    if upsilon.source is not lifted or upsilon.target is not new_spine:
        raise ValueError("upsilon must map the current lifted graph to new_spine")
    section = homotopy_section(spine, lifted, phi_prime)
    comparison = compose(upsilon, section["psi"])
    # comparison : spine -> new_spine, so its right inverse lives on the
    # target: homotopy_section(base=new_spine, cover=spine, phi=comparison)
    # returns new_spine -> spine.  Passing spine first reproduced comparison's
    # own direction and was invisible whenever new_spine was literally spine.
    comparison_inverse = homotopy_section(new_spine, spine, comparison)["psi"]
    new_to_original = compose(to_original, comparison_inverse)
    from_original = homotopy_section(session.base, spine, to_original)["psi"]
    new_from_original = compose(comparison, from_original)
    P, pi_prime, phi_new, projection = pullback_ve(
        session.base, session.cover, session.phi, new_spine,
        new_to_original, h=new_from_original)
    return {
        "P": P,
        "pi_prime": pi_prime,
        "phi_prime": phi_new,
        "projection": projection,
        "new_spine": new_spine,
        "to_original": new_to_original,
        "from_original": new_from_original,
        "comparison": comparison,
        "comparison_inverse": comparison_inverse,
        "section": section["psi"],
    }


def dual_cells(graph):
    """The cell decomposition dual to `graph`, with darts still attached.

    `graph.dual()` is the authoritative construction: unlike a rotation keyed
    by neighbour vertices it distinguishes parallel edges.  A dual vertex is a
    primal face index, and a dual face is a polygon around a primal vertex.

    The returned `polygons` contain dual vertex indices, suitable for geometric
    calculations.  `corners` contains the corresponding boundary darts.  Keep
    the latter whenever edge lengths or incidences matter: two corners can join
    the same pair of vertices through different edges (the rabbit has one such
    doubled edge), so a dict keyed by an unordered vertex pair loses data.

    The structural check is on darts, not vertex pairs.  Every dual dart occurs
    in exactly one dual face, and Euler's formula is checked against the
    multigraph edge count.  This is the multigraph-tolerant replacement for
    asking `Mesh.validate` to certify this coarse decomposition.
    """
    dual = graph.dual()
    corners = tuple(tuple(walk) for walk in dual.faces())
    polygons = tuple(tuple(dual.src(d) for d in walk) for walk in corners)
    vertices = tuple(dual.vertices())
    edges = tuple(dual.edge_list())

    seen = {}
    non_simple = []
    for k, walk in enumerate(corners):
        # Ribbon boundary walks may be bigons or revisit a vertex.  They remain
        # valid combinatorial darts; only the numerical mesh/solver boundary
        # needs simple polygon corners (see round_realise_cells).
        if len(walk) < 3 or len(set(polygons[k])) != len(polygons[k]):
            non_simple.append(k)
        for dart in walk:
            if dart in seen:
                raise ValueError(f"dual dart {dart} occurs in two faces")
            seen[dart] = k
    missing = set(dual.darts()) - set(seen)
    if missing:
        raise ValueError(f"dual darts absent from every face: {sorted(missing, key=str)}")
    if len(vertices) <= max(max(p) for p in polygons):
        raise ValueError("a polygon names an unknown dual vertex")
    chi = len(vertices) - len(edges) + len(polygons)
    if chi != 2:
        raise ValueError(f"dual cell decomposition has Euler characteristic {chi}, not 2")

    return {
        "graph": graph,
        "dual": dual,
        "vertices": vertices,
        "edges": edges,
        "polygons": polygons,
        "corners": corners,
        "non_simple_faces": non_simple,
        "counts": {"V": len(vertices), "E": len(edges), "F": len(polygons)},
    }


def dual_polygons(graph):
    """Compatibility wrapper returning `(primal_faces, dual_polygons)`.

    New code should use `dual_cells`, since polygons alone discard the edge
    identities needed for multigraphs and for lifted metric data.
    """
    return graph.faces(), dual_cells(graph)["polygons"]


def parallel_edge_pairs(cells):
    """`{(vertex, vertex): [edge labels]}` for genuinely parallel dual edges."""
    dual = cells["dual"]
    by_pair = {}
    for edge in cells["edges"]:
        a, b = dual.edges[edge]
        key = tuple(sorted((a, b), key=str))
        by_pair.setdefault(key, []).append(edge)
    return {pair: es for pair, es in by_pair.items() if len(es) > 1}


def _base_geometry(points, faces):
    """Chordal lengths and polygon-angle sums of a spherical cell decomposition."""
    from .mesh import to_vector

    vectors = np.array([to_vector(p) for p in points])
    lengths = {}
    edge_uses = {}
    cone = {v: 0.0 for v in range(len(points))}
    for face in faces:
        if len(face) < 3:
            raise ValueError(f"base face {face} has fewer than three corners")
        for k, i in enumerate(face):
            j = face[(k + 1) % len(face)]
            key = tuple(sorted((i, j)))
            lengths[key] = float(np.linalg.norm(vectors[i] - vectors[j]))
            edge_uses[key] = edge_uses.get(key, 0) + 1
            before = vectors[face[k - 1]] - vectors[i]
            after = vectors[j] - vectors[i]
            denom = float(np.linalg.norm(before) * np.linalg.norm(after))
            if denom == 0.0:
                raise ValueError(f"coincident vertices at corner {i} of {face}")
            cone[i] += float(np.arccos(np.clip(
                np.dot(before, after) / denom, -1.0, 1.0)))
    # A polygonal Delaunay block has no diagonals, so count its boundary edges.
    bad = {edge: uses for edge, uses in edge_uses.items() if uses != 2}
    if bad:
        raise ValueError(f"base faces do not share each edge exactly twice: {bad}")
    return vectors, lengths, cone


def lifted_delaunay_metric(cells, new_spine, points, base_faces=None):
    """Lift the base Delaunay chordal metric through a pulled-back spine.

    `cells` must come from `dual_cells(P)`, where `P` covers `new_spine`, and
    `new_spine` is the Voronoi spine of `points`.  Every dual edge retains the
    pulled-back spine edge label `(base_edge, sheet)`, so its length is simply
    the chordal length of `base_edge`.  A dual vertex is a face of `P`;
    `face_map_pi` supplies its base puncture and local degree, and its cone
    angle is `degree * base_cone_angle`.

    Returns explicit per-edge lengths (parallel edges have separate entries),
    per-vertex cone angles, the base angle sums, the cover portrait, and the
    list of vertices with cone angle >= 2*pi.  The last item is diagnostic, NOT
    a realisation obstruction: discrete conformal vertex scaling changes angles.
    (The separate BDS prescribed-angle construction calls those hyper-ideal.)
    """
    from .mesh import delaunay_faces, voronoi_face_points

    if base_faces is None:
        base_faces = delaunay_faces(points)
    _vectors, base_lengths, base_cone = _base_geometry(points, base_faces)
    face_point = voronoi_face_points(new_spine, len(points))
    portrait, _base_walks = face_map_pi(cells["graph"], new_spine)

    lengths = {}
    for edge in cells["edges"]:
        try:
            base_edge = tuple(sorted(edge[0]))
        except (TypeError, IndexError) as exc:
            raise ValueError(f"lifted edge {edge} carries no base-edge label") from exc
        if len(base_edge) != 2 or base_edge not in base_lengths:
            raise ValueError(f"lifted edge {edge} has unknown base edge {base_edge}")
        lengths[edge] = base_lengths[base_edge]

    cone = {}
    for vertex in cells["vertices"]:
        row = portrait[vertex]
        if row["target"] is None or row["degree"] is None:
            raise ValueError(f"cover face {vertex} has no Delaunay portrait target")
        point = face_point[row["target"]]
        cone[vertex] = row["degree"] * base_cone[point]

    return {
        "lengths": lengths,
        "cone_angles": cone,
        "base_cone_angles": base_cone,
        "base_lengths": base_lengths,
        "portrait": portrait,
        "face_points": face_point,
        "hyperideal": [v for v in cells["vertices"]
                       if cone[v] >= 2.0 * np.pi - 1e-12],
    }



def active_puncture_labels(phi_prime, face_points, pi_prime=None, dynamics=None):
    """Return `{postcritical_label: dual_vertex}` for the active faces.

    `face_points` is Voronoi face index -> analytic postcritical label on
    phi_prime's target, computed by `mesh.voronoi_face_points`.  A non-collapsed
    face of a transported ribbon VE must tighten to an actual target face walk;
    a merely non-empty non-face word signals the wrong comparison map, not a
    label that can safely be used.

    Supplying both `pi_prime` and `dynamics` (label -> image label) additionally
    checks the other leg: an active puncture p must project to dynamics[p].
    """
    if dynamics is not None and pi_prime is None:
        raise ValueError("dynamics verification also needs pi_prime")

    rows, _ = face_map_phi(phi_prime)
    cover_rows = (face_map_pi(pi_prime.source, pi_prime.target)[0]
                  if dynamics is not None else None)
    out = {}
    for face, row in enumerate(rows):
        if not row["tight"]:
            continue
        if row["target"] is None:
            raise ValueError(
                f"active face {face} tightens to a non-face word: the comparison "
                "map does not produce a ribbon virtual endomorphism")
        point = face_points[row["target"]]
        if point in out:
            raise ValueError(f"two active faces carry postcritical label {point}")
        if cover_rows is not None:
            actual = face_points[cover_rows[face]["target"]]
            if actual != dynamics[point]:
                raise ValueError(
                    f"active puncture {point}: pi' lands on {actual}, "
                    f"expected {dynamics[point]}")
        out[point] = face
    return out


def round_realise_cells(cells, metric, x0=None, require_canonical=False,
                        require_embedded=False, tol=1e-13, **kwargs):
    """Round-realise an edge-labelled lifted multigraph and return a certificate.

    Edge identities are retained, including parallel edges.  The returned dict
    records the residual and the rank/excess certificate; `canonical` is a
    local, transverse uniqueness statement up to Möbius.  The separate
    embedding record certifies injectivity, nondegenerate faces and consistent
    orientation.  Residual <= tol alone is NOT a completed Thurston step.  Use
    `require_canonical=True` against extra conformal moduli and
    `require_embedded=True` against folded/collapsed branches.
    """
    from .lift_round import round_realise_cells as realise

    return realise(cells, metric, x0=x0, require_canonical=require_canonical,
                   require_embedded=require_embedded, tol=tol, **kwargs)
