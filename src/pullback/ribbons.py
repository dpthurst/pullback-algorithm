"""Ribbon graphs: a graph with a cyclic order of half-edges at each vertex.

A *dart* is an oriented edge: (label, ori) with ori F (+1) or B (-1).
F e is e traversed from src to tgt; B e the reverse.  The rotation
system (cyclic orders) determines the thickening and its boundary
components (= faces) as cycles of sigma . alpha on darts.
"""

from __future__ import annotations

F = 1
B = -1
Dart = tuple  # (label, ori); ori in {F, B}


def rev(dart: Dart) -> Dart:
    return (dart[0], -dart[1])


def reduce_chain(walk):
    """Free reduction: drop immediate backtracks.  Valid for edge chains
    (paths); preserves the endpoints."""
    stack = []
    for d in walk:
        if stack and stack[-1] == rev(d):
            stack.pop()
        else:
            stack.append(d)
    return tuple(stack)


def tighten(walk):
    """Free reduction followed by cyclic reduction.  Valid only for
    CLOSED walks (face boundaries): the cyclic step changes the
    endpoints of an open chain, so edge images must use reduce_chain."""
    stack = list(reduce_chain(walk))
    changed = True
    while changed and stack:
        changed = False
        if stack[0] == rev(stack[-1]):
            stack.pop(0)
            stack.pop()
            changed = True
    return tuple(stack)


class RibbonGraph:
    """Vertices with flowers: vertex -> tuple of outgoing darts, in the
    counterclockwise cyclic order.  Loops appear twice in the same
    flower (once per orientation)."""

    def __init__(self, flowers: dict):
        # flowers: {vertex: [dart, ...]}, every dart of every edge
        # appearing exactly once across all flowers.
        self.flowers = {v: tuple(ds) for v, ds in flowers.items()}
        # edge endpoints: (label, F) starts at its flower vertex; the
        # reverse dart's vertex is found from the flowers.
        self._dart_vertex = {}
        for v, ds in self.flowers.items():
            for d in ds:
                if d in self._dart_vertex:
                    raise ValueError(f"dart {d} appears twice")
                self._dart_vertex[d] = v
        self.edges = {}
        for (label, ori), v in self._dart_vertex.items():
            if ori == F:
                self.edges[label] = [v, None]
        for (label, ori), v in self._dart_vertex.items():
            if ori == B:
                self.edges[label][1] = v
        for label, (s, t) in self.edges.items():
            if s is None or t is None:
                raise ValueError(f"edge {label} missing an end")
        self._rotation = {}
        for v, ds in self.flowers.items():
            n = len(ds)
            for i, d in enumerate(ds):
                self._rotation[d] = ds[(i + 1) % n]

    def vertices(self):
        return list(self.flowers)

    def edge_list(self):
        return list(self.edges)

    def src(self, dart: Dart):
        return self._dart_vertex[dart]

    def tgt(self, dart: Dart):
        return self._dart_vertex[rev(dart)]

    def alpha(self, dart: Dart) -> Dart:
        """Reverse the dart."""
        return rev(dart)

    def sigma(self, dart: Dart) -> Dart:
        """Next dart in the cyclic order at the dart's vertex."""
        return self._rotation[dart]

    def darts(self):
        return list(self._dart_vertex)

    def faces(self):
        """Boundary components of the thickening: cycles of sigma . alpha,
        each returned as a tuple of darts (the boundary walk).

        The list is canonical: every walk starts at its smallest dart (by
        `str`) and the walks are sorted by that key.  Determinism matters:
        the earlier version started from `set.pop()`, so the face numbering
        (and hence anything keyed by face index, e.g. `type_classification`)
        changed with PYTHONHASHSEED between processes.
        """
        remaining = set(self._dart_vertex)
        walks = []
        for d0 in self._dart_vertex:        # dict order is insertion order
            if d0 not in remaining:
                continue
            walk = [d0]
            remaining.discard(d0)
            d = self.sigma(self.alpha(d0))
            while d != d0:
                remaining.discard(d)
                walk.append(d)
                d = self.sigma(self.alpha(d))
            walks.append(tuple(walk))

        def key(w):
            return tuple(str(d) for d in w)

        def canonical_rotation(w):
            n = len(w)
            return min((w[i:] + w[:i] for i in range(n)), key=key)

        return sorted((canonical_rotation(w) for w in walks), key=key)

    def dual(self):
        """The dual ribbon graph.

        Vertices are the faces of this graph (the punctures, for a spine of
        S^2 - P), edges are dual to this graph's edges, and the cyclic order at
        a dual vertex is the boundary walk of the corresponding face, reversed --
        the standard reversal for the dual of an oriented map.  The dual's faces
        are then this graph's vertices, so `dual` is an involution up to
        relabelling, which the tests check.

        The construction is one line because a face walk is a cycle of darts and
        each dart of the walk is one half-edge of the dual edge crossing it:
        the dual flower at a face is the reversed walk.
        """
        face_of_dart = {}
        for i, walk in enumerate(self.faces()):
            for d in walk:
                face_of_dart[d] = i
        flowers = {}
        for i, walk in enumerate(self.faces()):
            flowers[i] = tuple(reversed([(d[0], d[1]) for d in walk]))
        return RibbonGraph(flowers)

    def mirror(self):
        """The globally reversed ribbon structure (every cyclic order
        reversed).  For a planar spine this is the mirror image, and its
        branched self-cover is the complex conjugate (see plan 50)."""
        return RibbonGraph({v: tuple(reversed(ds)) for v, ds in self.flowers.items()})

    def serialization(self):
        """A deterministic form: per vertex (sorted), the flower with its
        rotation normalized to start at the smallest dart."""
        out = []
        for v in sorted(self.flowers, key=str):
            ds = self.flowers[v]
            n = len(ds)
            rots = [ds[i:] + ds[:i] for i in range(n)]
            out.append((str(v), min(rots)))
        return tuple(sorted(out))

    def canonical_orientation(self):
        """(canonical graph, was_mirrored): the lexicographically smaller
        of this graph and its mirror, so sessions and fixtures can be
        compared across orientations."""
        s_self = self.serialization()
        s_mir = self.mirror().serialization()
        if s_mir < s_self:
            return self.mirror(), True
        return self, False

    def __repr__(self):
        fl = " ".join(
            f"{v}:{''.join(_fmt(d) for d in ds)}" for v, ds in self.flowers.items()
        )
        return f"RibbonGraph({fl})"


def _fmt(dart: Dart) -> str:
    label, ori = dart
    return str(label) if ori == F else str(label).upper()


def parse_dart(token: str) -> Dart:
    """Harmonic convention: lowercase = forward, uppercase = backward."""
    if len(token) != 1:
        raise ValueError(f"bad dart {token!r}")
    if token.islower():
        return (token, F)
    return (token.lower(), B)


def fmt_dart(dart: Dart) -> str:
    return _fmt(dart)
