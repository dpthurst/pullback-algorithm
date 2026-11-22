"""Proposed replacement for `pullback.lift.round_realise_cells`.

Drop-in shape: same arguments, same tolerance keywords, but

  * the complex is taken as an explicit EDGE-LABELLED multigraph, so the two
    lifts of a base edge become two rows and neither is dropped or merged;
  * parallel edges and non-simple boundary walks no longer refuse -- the
    criterion is per edge and needs no face at all, so those are not obstacles;
  * what comes back is a *certificate* rather than a bare triple: V, E,
    rank(A), the number of redundant edge pairs, rank(P J_a) on the product of
    spheres, and `excess = 2V - 6 - rank(PJ)` -- the number of real moduli of
    the conformal class the cell complex cannot see;
  * `canonical` is True exactly when excess == 0, i.e. when the round
    realisation is unique up to Mobius (generically), which is the only case in
    which the output is a well-defined point of Teich and can be compared
    between runs.  excess > 0 is reported, never silently averaged away.

The wiring into `lift.py` is three lines: build `edges` from `cells["edges"]`
and `dual.edges`, take lengths from `metric["lengths"]`, and call
`round_realise_cells` below instead of `rounding.round_realise`.
"""

from __future__ import annotations

import numpy as np

try:                                    # inside the package
    from . import rounding_multi as mg
except ImportError:                     # imported from the swarm directory
    from pullback import rounding_multi as mg

__all__ = ["labelled_cells", "labelled_corners", "round_realise_cells"]


def labelled_cells(cells, metric=None):
    """`({edge_label: (vertex, vertex)}, {edge_label: length})` of a dual complex.

    `cells` is `pullback.lift.dual_cells`'s dict.  Its `"edges"` are dual-edge
    labels (for a lifted spine they are `(base_edge, sheet)` pairs), and
    `cells["dual"].edges[label]` gives the two dual vertices.  Parallel labels
    keep separate entries; the vertex-pair view is never taken.
    """
    dual = cells["dual"]
    edges = {}
    for label in cells["edges"]:
        a, b = dual.edges[label]
        edges[label] = (int(a), int(b))
    lengths = {}
    if metric is not None:
        for label in cells["edges"]:
            lengths[label] = float(metric["lengths"][label])
    return edges, lengths


def labelled_corners(cells):
    """Per-face cyclic sequences of edge labels, for the output cone angles.

    `cells["corners"][k]` is the boundary walk of face k as dual darts; a dart's
    first component is the dual edge label.  Handing the labels through is what
    lets a corner of a doubled edge know which copy it measures.
    """
    return [tuple(dart[0] for dart in walk) for walk in cells["corners"]]


def round_realise_cells(cells, metric, x0=None, require_canonical=False,
                        require_embedded=False, tol=1e-13, **kwargs):
    """Round-realise a lifted cell metric; return geometry + certificates.

    Raises ValueError when the local solver cannot reach `tol`, when canonical
    uniqueness is required but `excess > 0`, or when an embedded realisation is
    required but the determinant/orientation certificate fails.  `canonical` is
  a local rank statement; `embedding` independently detects folds, coincidences
  and degenerate faces.  Neither is inferred from residual tolerance.
    """
    from . import embedding

    edges, lengths = labelled_cells(cells, metric)
    x, residual, iters = mg.round_realise(edges, lengths, x0=x0, tol=tol,
                                          **kwargs)
    if not np.isfinite(residual) or residual > max(tol, 1e-12):
        raise ValueError(
            f"lifted metric is not round-realizable on this complex: "
            f"residual {residual:.3e} after {iters} iterations")
    info = mg.structure(edges, x)
    info["residual"] = residual
    info["iterations"] = iters
    info["cone_angles"] = mg.cone_angles(labelled_corners(cells), edges, x)
    try:
        labels = labelled_corners(cells)
        oriented = embedding.orient_faces(cells["polygons"], labels)
        info["embedding"] = embedding.classify(x, oriented)
    except ValueError as exc:
        info["embedding"] = {"embedded": False, "reason": str(exc)}
    if require_embedded and not info["embedding"]["embedded"]:
        raise ValueError(
            "round realisation is not an embedded sphere: "
            + info["embedding"]["reason"])
    if require_canonical and not info["canonical"]:
        raise ValueError(
            f"round realisation is not unique up to Mobius: excess "
            f"{info['excess']} (V={info['V']}, E={info['E']}, "
            f"rank(PJ)={info['rank_PJ']}, redundant pairs "
            f"{info['redundant']}); the cell complex under-determines the "
            f"conformal class, so no canonical step exists on it")
    return {"x": x, "edges": edges, "lengths": lengths, **info}
