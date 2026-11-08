"""Homotopy sections and the active puncture set.

For a compatible pair (pi : Gamma_1 -> Gamma_0, phi : Gamma_1 -> Gamma_0), the
*homotopy section* is a graph map psi : Gamma_0 -> Gamma_1 with phi o psi
homotopic to the identity on Gamma_0.  It is not a lift in the covering sense --
phi is not a covering -- and it always exists for compatible data:

* P_0 is contained in P_1 since f(P_0) is inside P_0, so the inclusion
  S^2 - P_1 -> S^2 - P_0 is pi_1-surjective (a loop can be pushed off the finite
  set P_1 - P_0), and
* a surjection of free groups splits, so on pi_1 there is a splitting, and since
  connected graphs are aspherical, based maps between them are determined by
  their induced homomorphism on pi_1.

The algorithm is Pilgrim-Thurston's own proof of [ConfDim], `subsecn:sections`
(`refs/confdim-src/ve.tex:1331-1398`, `lemma:sections`): choose a spanning tree
on each spine, compute phi_* on the free bases, then find a preimage for each
target basis element by BFS in the Cayley graph, and realise the splitting as a
graph map.  `is_pi1_surjective(phi, base)` (in `validate.py`) is the existence
certificate, and in fact obtaining a based psi is itself one.

Reimplemented from `.ultramath/smw7z960p/m4/section_check.py` after verifying
that it reproduces based sections on all 15 library sessions.

Also `active_faces(phi)`: the *non-collapsed* punctures -- the faces of the cover
whose phi-image walk does not tighten to nothing, which is the [PChar,
`thm:spine-surf`] criterion, `spines.tex:376-390`.  The alternative "local degree
1 preimages" criterion is a coincidence of degree 2 with |P_0| = 4, and is wrong
as a set even there.
"""

from __future__ import annotations

from .maps import make_graph_map
from .portrait import face_map_phi
from .ribbons import F, reduce_chain, tighten
from .validate import (is_pi1_surjective, rose_words, spanning_tree,
                       _tree_path)


# ----------------------------------------------------------------- utilities
def compose(f, g):
    """(f o g): the graph map from g.source to f.target."""
    chains = {}
    for e in g.source.edge_list():
        ch = []
        for d in g.chains[e]:
            ch.extend(f.image_dart(d))
        chains[e] = reduce_chain(tuple(ch))
    vim = {v: f.image_vertex(g.image_vertex(v)) for v in g.source.vertices()}
    return make_graph_map(g.source, f.target, chains, vim)


def _loop_of_generator(graph, tree, base_v, e):
    """The based loop (dart path) realising generator edge e of the spine."""
    s, t = graph.edges[e]
    return (_tree_path(graph, tree, base_v, s) + ((e, F),)
            + _tree_path(graph, tree, t, base_v))


def expand(graph, tree, base_v, gens, word):
    """Word in the free generators of pi_1(graph) -> dart-path loop at base_v."""
    loops = {i: _loop_of_generator(graph, tree, base_v, e)
             for i, e in enumerate(gens)}
    out = []
    for (i, o) in word:
        p = loops[i]
        out.extend(p if o == F
                   else tuple((d[0], -d[1]) for d in reversed(p)))
    return reduce_chain(tuple(out))


def eval_word(images, word):
    """Evaluate a word in the abstract generators of pi_1(Gamma_1) at `images`,
    free-reduced words in the basis of pi_1(Gamma_0)."""
    out = ()
    for (i, o) in word:
        w = images[i] if o == F else tuple((j, -t)
                                          for (j, t) in reversed(images[i]))
        out = free_reduce(out + w)
    return out


def free_reduce(w):
    stack = []
    for d in w:
        if stack and stack[-1] == (d[0], -d[1]):
            stack.pop()
        else:
            stack.append(d)
    return tuple(stack)


# ----------------------------------------------------------- the section
def section_words(images, n):
    """BFS in the Cayley graph of F_n with the given images: for each target
    basis element e_i (i in 0..n-1), find a word u in the *source* basis letters
    with eval_word(u) = e_i.  Terminates exactly when phi_* is surjective."""
    found = {}
    start = ()
    seen = {start: ()}
    frontier = [start]
    while frontier and len(found) < n:
        nxt = []
        for img in frontier:
            word = seen[img]
            for i in range(len(images)):
                for o in (F, -F):
                    nimg = free_reduce(img + eval_word(images, ((i, o),)))
                    if nimg in seen:
                        continue
                    seen[nimg] = free_reduce(word + ((i, o),))
                    nxt.append(nimg)
                    if len(nimg) == 1 and nimg[0][1] == F and nimg[0][0] not in found:
                        found[nimg[0][0]] = seen[nimg]
        frontier = nxt
    return found


def homotopy_section(base, cover, phi):
    """A based homotopy section psi : Gamma_0 -> Gamma_1.

    `base` is Gamma_0 (the RibbonGraph), `cover` is Gamma_1, `phi` the graph map.
    Returns a dict with

      psi        : the GraphMap psi : base -> cover,
      composed   : phi . psi, the identity on pi_1 (checked),
      images     : phi_* on the free basis of Gamma_1, in Gamma_0's basis,
      surjective : the is_pi1_surjective certificate.

    Raises ValueError if phi_* is not surjective, since a section cannot exist
    then.  The corresponding ValueError about a missing basepoint lift now means
    the construction failed to realise that splitting, not that no section
    exists.  Recall that free-group surjections between finite graphs always
    have a section.  (The distinction matters when reusing this routine to
    invert a geometric comparison, where earlier cyclic-word bookkeeping could
    reject a valid inverse.)
    """
    tree0, gens0 = spanning_tree(base)
    tree1, gens1 = spanning_tree(cover)
    surjective = is_pi1_surjective(phi, base)
    if not surjective:
        raise ValueError(
            "phi_* is not pi_1-surjective: a homotopy section cannot exist")
    base0 = base.vertices()[0]
    idwords = [((i, F),) for i in range(len(gens0))]
    for base1 in cover.vertices():
        if phi.image_vertex(base1) != base0:
            continue
        # The homomorphism depends on the based point even though its image's
        # surjectivity does not.  Reusing words computed at cover.vertices()[0]
        # for every fibre point silently changes it by different conjugacies.
        words = rose_words(phi, tree0, tree1, gens1, base_v=base1)
        images = [tuple((gens0.index(d[0]), d[1]) for d in w) for w in words]
        found = section_words(images, len(gens0))
        chains = {e: expand(cover, tree1, base1, gens1, found[i])
                  for i, e in enumerate(gens0)}
        psi = make_graph_map(base, cover, chains, {base0: base1})
        comp = compose(phi, psi)
        ctree, cgens = spanning_tree(base)
        cimg = [tuple((cgens.index(d[0]), d[1]) for d in w)
                for w in rose_words(comp, ctree, ctree, cgens)]
        if cimg == idwords and comp.image_vertex(base0) == base0:
            return {"psi": psi, "composed": comp, "images": images,
                    "surjective": surjective, "basepoint": base1,
                    "tree": (tree0, tree1)}
    raise ValueError(
        "a section exists on pi_1, but no basepoint lift realises it as a based"
        " graph map: phi does not respect a based section")


def active_faces(phi):
    """The ACTIVE faces of the cover: those whose phi-image walk is not empty.

    A cover face is *collapsed* (filled in) when phi pushes its boundary walk to
    a null-homotopic walk (tightens to nothing), which is [PChar]'s criterion in
    the proof of `thm:spine-surf`.  The active punctures are the ones kept when
    passing to the reduced map.  Measured on the library: |active| = |P_0| on 14
    of 15 sessions (grigorchuk, which is not a ribbon presentation, has 6 against
    |P_0| = 5).
    """
    rows, _ = face_map_phi(phi)
    return [i for i, e in enumerate(rows) if e["tight"]]


# ------------------------------------------------------- peripheral checks
def _closed_word_class(walk, allow_reverse=True):
    """Canonical cyclic reduction of a closed word, optionally up to reversal."""
    w = tuple(tighten(tuple(walk)))
    if not w:
        return ()
    candidates = [w[k:] + w[:k] for k in range(len(w))]
    if allow_reverse:
        inverse = tuple((d[0], -d[1]) for d in reversed(w))
        candidates.extend(inverse[k:] + inverse[:k]
                          for k in range(len(inverse)))
    return min(candidates, key=lambda word: tuple(str(d) for d in word))


def peripheral_images(graph_map, allow_reverse=True):
    """Target face conjugacy classes hit by the source's face boundaries.

    Returns one entry per source face: the target face index when its image is
    a degree-one peripheral word, otherwise `None`.  Reversal is allowed by
    default because this is only the orientation-unordered, necessary check.

    A map induced by the identity (or any homeomorphism) of a punctured sphere
    must permute these classes.  Arbitrary free-basis transport need not: on
    the rabbit's Voronoi spine one face maps to `a^-1 b c^-1`, which is not any
    of the four target peripheral words.  Such a map cannot be the common-
    inclusion comparison `h = upsilon . psi` of the pullback reformulation.
    """
    target_classes = {}
    for i, walk in enumerate(graph_map.target.faces()):
        target_classes.setdefault(
            _closed_word_class(walk, allow_reverse), []).append(i)
    out = []
    for walk in graph_map.source.faces():
        raw = []
        for dart in walk:
            raw.extend(graph_map.image_dart(dart))
        hits = target_classes.get(_closed_word_class(raw, allow_reverse), [])
        out.append(hits[0] if len(hits) == 1 else None)
    return out


def is_peripheral_preserving(graph_map, allow_reverse=True):
    """Does `graph_map` bijectively permute all puncture conjugacy classes?"""
    images = peripheral_images(graph_map, allow_reverse)
    return (None not in images and
            sorted(images) == list(range(len(graph_map.target.faces()))))


# ------------------------------------------------------- basis transport
def basis_transport(source, target):
    """An ARBITRARY homotopy equivalence, by free-basis index transport.

    This utility knows only the ranks and edge orderings of the two graphs.  It
    is not the retraction-induced comparison of the pullback reformulation and
    must not be passed as that map without further geometric certification:
    outer automorphisms of a free group generally do not preserve peripheral
    classes, while the common-inclusion comparison must do so.  Use
    `is_peripheral_preserving` as a necessary check (not a sufficient proof of
    being the intended geometric map).

    Choose spanning trees on each spine (so each has a free basis), and map the
    i-th generator of source to the i-th generator of target, as a based loop.
    `basis_transport(target, source)` gives an inverse on pi_1 for those same
    ordered bases.  Different bases genuinely give different elements of
    Out(F_n), not merely 'base cases'.
    """
    t0, gens0 = spanning_tree(source)
    t1, gens1 = spanning_tree(target)
    if len(gens0) != len(gens1):
        raise ValueError("the two spines have different ranks: no homotopy"
                         " equivalence")
    base_s = source.vertices()[0]
    base_t = target.vertices()[0]
    chains = {}
    for i, e in enumerate(source.edge_list()):
        if e in t0:
            chains[e] = ()
        else:
            idx = gens0.index(e)
            chains[e] = _loop_of_generator(target, t1, base_t, gens1[idx])
    m = make_graph_map(source, target, chains, {base_s: base_t})
    m.src, m.tgt = source, target
    m.name = 'basis_transport'
    return m
