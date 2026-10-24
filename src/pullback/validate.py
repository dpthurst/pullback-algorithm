"""Validation of graph virtual endomorphisms (plan item 100).

- `is_connected`: the cover is connected (pi_0-bijectivity is then
  automatic for a connected base).
- `is_pi1_surjective`: the induced map on pi_1 is surjective, decided
  on generators: collapse spanning trees on both sides, read the
  phi-images of the source's free generators as words in the target's,
  and decide by Stallings folding whether those words generate the
  free group.
"""

from __future__ import annotations

from .ribbons import F, B, rev
from .portrait import tighten


def is_connected(graph) -> bool:
    verts = graph.vertices()
    if not verts:
        return True
    seen = {verts[0]}
    stack = [verts[0]]
    while stack:
        v = stack.pop()
        for d in graph.flowers[v]:
            w = graph.tgt(d)
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return len(seen) == len(verts)


class _DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def spanning_tree(graph):
    """Return (tree edge labels, generator edge labels) for a connected graph."""
    verts = graph.vertices()
    tree, seen, frontier = set(), {verts[0]}, [verts[0]]
    while frontier:
        v = frontier.pop()
        for d in graph.flowers[v]:
            w = graph.tgt(d)
            if w not in seen:
                tree.add(d[0])
                seen.add(w)
                frontier.append(w)
    gens = [e for e in graph.edge_list() if e not in tree]
    return tree, gens


def _tree_path(graph, tree, u, v):
    """Unique reduced path in the tree from u to v, as darts."""
    prev = {u: None}
    queue = [u]
    while queue:
        x = queue.pop(0)
        if x == v:
            break
        for d in graph.flowers[x]:
            if d[0] not in tree:
                continue
            w = graph.tgt(d)
            if w not in prev:
                prev[w] = (x, d)
                queue.append(w)
    path = []
    x = v
    while prev[x] is not None:
        px, d = prev[x]
        path.append(rev(d))
        x = px
    return tuple(path)


def rose_words(phi, tree0, tree1, gens1):
    """Images of the source's generator loops as words in the target
    generators.  A generator edge e of the source (after choosing a
    spanning tree T_1 and basepoint) represents the loop
    t_base,src(e) . e . t_tgt(e),base; its image is the phi-image of
    that composite, with target-tree letters deleted and the result
    freely reduced."""
    src_g = phi.source
    base_v = src_g.vertices()[0]
    words = []
    for e in gens1:
        s, t = src_g.edges[e]
        path = (_tree_path(src_g, tree1, base_v, s)
                + ((e, F),)
                + _tree_path(src_g, tree1, t, base_v))
        full = []
        for d in path:
            full.extend(phi.image_dart(d))
        w = tighten(tuple(x for x in full if x[0] not in tree0))
        words.append(w)
    return words


def _stallings_rose(wordlist, n):
    """Fold the wedge of labeled cycles; the subgroup they generate is
    all of F_n iff the folded graph is the rose: one vertex class with
    every generator label appearing."""
    dsu = _DSU()
    edges = []  # directed (u, (g, s), v); reverse registered below
    nxt = 1
    root = 0
    for w in wordlist:
        if not w:
            continue
        u = root
        L = len(w)
        for i, lab in enumerate(w):
            v = root if i == L - 1 else nxt
            if i != L - 1:
                nxt += 1
            edges.append((u, lab, v))
            u = v
    # add reverse directions so folding sees both orientations
    directed = []
    for (u, lab, v) in edges:
        directed.append((u, lab, v))
        directed.append((v, (lab[0], -lab[1]), u))
    directed.append((root, None, root))  # keep root alive for empty lists
    changed = True
    while changed:
        changed = False
        got = {}
        for (u, lab, v) in directed:
            ru, rv = dsu.find(u), dsu.find(v)
            key = (ru, lab)
            if key in got:
                if dsu.find(got[key]) != rv:
                    dsu.union(got[key], rv)
                    changed = True
            else:
                got[key] = rv
    classes = {dsu.find(u) for (u, _, _) in directed}
    present = {lab[0] for (_, lab, _) in directed if lab is not None}
    return len(classes) == 1 and len(present) == n, classes, present


def generates_free_group(words, n):
    ok, _, _ = _stallings_rose(words, n)
    return ok


def is_pi1_surjective(phi, base) -> bool:
    if not (is_connected(phi.source) and is_connected(base)):
        return False
    tree0, gens0 = spanning_tree(base)
    tree1, gens1 = spanning_tree(phi.source)
    words = rose_words(phi, tree0, tree1, gens1)
    wordlist = []
    for w in words:
        wordlist.append(tuple((gens0.index(d[0]), d[1]) for d in w))
    return generates_free_group(wordlist, len(gens0))


def validate(phi, base):
    """The plan-100 bundle as a report dict."""
    return {
        "cover_connected": is_connected(phi.source),
        "base_connected": is_connected(base),
        "pi1_surjective": is_pi1_surjective(phi, base),
    }
