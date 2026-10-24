"""Maps between graphs, given by an edge path in the target for each
edge of the source (Harmonic's makeGraphMap convention).

Edges not listed get the empty chain (collapsed to their image vertex).
Vertex images are deduced from the chains by propagation, exactly as
Harmonic's union-find does: the image of a vertex is constrained by
the endpoints of the chains of its incident edges; isolated vertices
must be given explicitly.
"""

from __future__ import annotations

from .ribbons import F, rev, reduce_chain


class GraphMap:
    def __init__(self, source, target, chains: dict, vertex_images: dict):
        self.source = source
        self.target = target
        self.chains = chains            # edge label -> tuple of target darts
        self.vertex_images = vertex_images  # vertex -> target vertex

    def image_dart(self, dart):
        """Image of a source dart as a chain of target darts."""
        label, ori = dart
        ch = self.chains[label]
        if ori == F:
            return tuple(ch)
        return tuple(rev(d) for d in reversed(ch))

    def image_vertex(self, v):
        return self.vertex_images[v]


def make_graph_map(source, target, chains: dict, vertex_images: dict | None = None):
    """chains: {source edge label: [target darts]}; unlisted edges are
    constant.  Deduce the images of all vertices reachable from the
    chains; verify consistency; require explicit vertex_images for the
    rest."""
    chains = {e: reduce_chain(tuple(chains.get(e, ()))) for e in source.edge_list()}
    vim = dict(vertex_images or {})

    def chain_endpoints(e):
        ch = chains[e]
        if not ch:
            return None
        return (target.src(ch[0]), target.tgt(ch[-1]))

    changed = True
    while changed:
        changed = False
        for e in source.edge_list():
            ends = chain_endpoints(e)
            if ends is None:
                continue
            s, t = ends
            for sv, tv in ((source.edges[e][0], s), (source.edges[e][1], t)):
                if sv in vim:
                    if vim[sv] != tv:
                        raise ValueError(
                            f"inconsistent images for {sv}: {vim[sv]} vs {tv}")
                else:
                    vim[sv] = tv
                    changed = True

    # constant edges constrain their endpoints to coincide
    changed = True
    while changed:
        changed = False
        for e in source.edge_list():
            if chains[e]:
                continue
            s, t = source.edges[e]
            if s in vim and t not in vim:
                vim[t] = vim[s]
                changed = True
            elif t in vim and s not in vim:
                vim[s] = vim[t]
                changed = True
            elif s in vim and t in vim and vim[s] != vim[t]:
                raise ValueError(
                    f"constant edge {e} but {vim[s]} != {vim[t]}")

    missing = [v for v in source.vertices() if v not in vim]
    if missing:
        raise ValueError(f"vertex images not determined: {missing}")
    return GraphMap(source, target, chains, vim)
