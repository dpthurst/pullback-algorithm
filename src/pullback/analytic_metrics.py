"""Independent rational-map ground truth and intrinsic metric choices.

A dense-sheet assembly is not certified merely by closing a mesh or by small
conformal residual.  Given an *independent* rational map ``f`` and a mesh on
the output sphere, this module checks the defining datum directly:

    f(lifted_vertex) == base_vertex[source[0]].

The ``source`` convention is the one used by :func:`pullback.mesh.lift_by_edges`:
``(base_vertex, sheet)``.  Positions in other gauges can be compared after an
explicit Möbius transport; that transport must be included in the certificate.

Metric choices are kept separate and named:

* ``base_chordal`` -- the chordal metric of the target/base sphere.  This is
  the input used by the old frozen VE construction.  At ramification it needs
  proper cone/fan data, rather than implicitly assuming a smooth chart.
* ``graph_product`` -- the product-chordal metric of the graph of ``f``.
  It uses both source and output projections and remains finite at poles.
* ``graph_affine`` -- the induced affine metric of the graph of ``f`` in
  ``C^2``.  Use only where every requested coordinate is finite.

The graph metrics are intrinsic compatible inputs for a different possible
uniformization of the same holomorphic cover.  They are not interchangeable
with the base-only pullback problem; a comparison must also record which
metric determines the discrete conformal class.
"""
from __future__ import annotations

import mpmath as mp
import numpy as np

from . import mesh


def _finite(z):
    # numpy.isfinite does not accept high-precision mpmath complex values.
    return bool(np.ndim(z) == 0 and mp.isfinite(z))


def _distance(z, w):
    """Chordal distance on the Riemann sphere, including infinity."""
    if np.ndim(z) != 0 or np.ndim(w) != 0:
        raise ValueError("points must be scalar chart coordinates")
    if not _finite(z) and not _finite(w):
        return 0.0
    dz = mesh.to_vector(complex(z) if _finite(z) else mp.inf)
    dw = mesh.to_vector(complex(w) if _finite(w) else mp.inf)
    return float(np.linalg.norm(dz - dw))


def _image(f, z):
    if not _finite(z):
        return f.f_infinity()
    return f(z)


def rational_certificate(f, lifted, base, tol=1e-14):
    """Check rational images and report ramification for a lifted mesh.

    ``lifted`` is any object with ``points`` and ``source`` matching the
    ``mesh.lift_by_edges`` convention. ``base`` needs ``points``.  Every
    lifted vertex is checked; an error or omission is never converted into a
    degree-zero certificate.  The return contains one row per vertex, so
    masked or discarded defects cannot be mistaken for ``ok``.
    """
    if len(lifted.points) != len(lifted.source):
        raise ValueError("points and source must have the same length")
    rows = []
    for u, q in enumerate(lifted.points):
        pair = tuple(lifted.source[u])
        if len(pair) != 2:
            raise ValueError(f"invalid source pair {pair!r} at {u}")
        i, sheet = pair
        if not isinstance(i, (int, np.integer)) or not 0 <= i < len(base.points):
            raise ValueError(f"invalid base vertex {i!r} at lifted vertex {u}")
        expected = base.points[i]
        got = _image(f, q)
        err = _distance(got, expected)
        row = {
            "vertex": u, "base_vertex": int(i), "sheet": sheet,
            "point": q, "expected_image": expected, "actual_image": got,
            "image_error": float(err),
            "local_degree": (int(f.local_degree_infinity())
                             if not _finite(q) else int(f.local_degree(q))),
        }
        if not np.isfinite(err) or err > float(tol):
            raise ValueError(
                f"lifted vertex {u} (sheet {sheet}) is not over base vertex {i}: "
                f"image {got!r}, expected {expected!r}, chordal error {err:.3e}")
        rows.append(row)
    return {"ok": bool(rows), "tolerance": float(tol), "vertices": rows,
            "ramified_vertices": [r["vertex"] for r in rows
                                  if r["local_degree"] > 1]}


def edge_metrics(f, lifted, base, family="base_chordal"):
    """Return explicit edge lengths for a rational mesh.

    Only ``Mesh``-like data (``.points``, ``.source``, ``.edges()``) are
    needed.  Product/affine cases base the value on the source coordinate
    and the *same rational map* used by ``rational_certificate``; they never
    infer a sheet relation by nearest Euclidean preimage.
    """
    edges = lifted.edges()
    X = [mesh.to_vector(q) if _finite(q) else mesh.to_vector(np.inf)
         for q in lifted.points]
    if family not in ("base_chordal", "graph_product", "graph_affine"):
        raise ValueError(f"unknown metric family {family}")
    out = {}
    if family == "graph_affine" and not all(_finite(q) for q in lifted.points):
        raise ValueError("graph_affine needs finite chart coordinates")
    for a, b in edges:
        i, j = lifted.source[a][0], lifted.source[b][0]
        if family == "base_chordal":
            si = mesh.to_vector(base.points[i])
            sj = mesh.to_vector(base.points[j])
            ell = float(np.linalg.norm(si - sj))
        else:
            zi, zj = lifted.points[a], lifted.points[b]
            if family == "graph_affine":
                q = complex(zi) - complex(zj)
                w = _image(f, zi) - _image(f, zj)
                ell = float(np.sqrt(abs(q) ** 2 + abs(w) ** 2))
            else:
                ell = float(np.sqrt(_distance(zi, zj) ** 2
                                    + _distance(_image(f, zi), _image(f, zj)) ** 2))
        if not np.isfinite(ell) or ell <= 0.0:
            raise ValueError(f"degenerate {family} edge {(a, b)}: {ell}")
        out[a, b] = ell
    return out
