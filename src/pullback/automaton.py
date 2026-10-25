"""Self-similar automata from graph virtual endomorphisms.

An Automaton carries the wreath recursion of a graph virtual
endomorphism in normal form (plan item 130/140):

  letters  = the free generators of pi_1(Gamma_0)
  sheets   = the covering sheets (letters of the tree)
  sigma[g] = the sheet permutation of g's lifts
  sections[g][i] = the phi-image of the lift of g at sheet i, as a word
                   in the generators (signs = orientations)

Which biset this is (plan item 140; author's decision, 2026-10-25): the
*COLLAPSED* single-basepoint group biset -- pi_1(Gamma_0) acting on the
sheets after 130's collapse of Gamma_1 to the rose -- and not the groupoid
biset over the vertices of Gamma_0.  Two consequences before extending it:
the collapse need not be effective, so there can be a nontrivial kernel
pi_1 -> IMG (session `twisted-z2i` has an infinite one; `AGENTS.md`
A0.1-2), and IMG's `ThurstonObstruction`/`SpiderAlgorithm` take exactly
this collapsed form, so the rational-map cross-check is unaffected.

Level-n action and the group operations (products, inverses, sections,
nucleus, contracting test) are implemented by the wreath recursion
g(x.w) = sigma_g(x) . (g|_x)(w), with words acting by composition
(left factor outermost).  CONSEQUENCE for `sections`: a section word is
a group element, so its FIRST entry acts LAST.  A geometric section is a
path d_1...d_k in traversal order, whose induced action applies d_1
first; it must therefore be stored as (d_k, ..., d_1).  `normal_form` and
`RoseAutomaton` do that reversal.  Getting it wrong is invisible when
every section is a single letter (or an involution at level 1) and shows
up first at level 4 otherwise -- theta's tree {p} was 2^26 instead of
2^25 because of it.
"""

from __future__ import annotations

from itertools import product

from .ribbons import F, B


def _reduce_word(word):
    """Reduce a word given as a tuple of (state_index, sign)."""
    stack = []
    for (g, s) in word:
        if stack and stack[-1] == (g, -s):
            stack.pop()
        else:
            stack.append((g, s))
    return tuple(stack)


class Automaton:
    def __init__(self, letters, sheets, sigma, sections):
        self.letters = tuple(letters)
        self.sheets = tuple(sheets)
        self.index = {g: i for i, g in enumerate(self.letters)}
        self.sheet_index = {s: i for i, s in enumerate(self.sheets)}
        # normalize to sheet/state indices
        self.sigma = {g: [self.sheet_index[s] if s in self.sheet_index else s
                          for s in sigma[g]] for g in self.letters}
        self.sections = {g: [tuple((self.index[d[0]] if d[0] in self.index else d[0], d[1])
                                  for d in sec) for sec in sections[g]]
                         for g in self.letters}

    def __eq__(self, other):
        return (self.letters == other.letters and self.sheets == other.sheets
                and self.sigma == other.sigma and self.sections == other.sections)

    # -- single-state actions ------------------------------------------------
    def state_action(self, g, word):
        """Apply state g to a level word (list of sheet indices)."""
        if not word:
            return []
        x = word[0]
        out = self.sigma[g][x]
        rest = self._apply_word(self.sections[g][x], word[1:])
        return [out] + rest

    def state_inverse_action(self, g, word):
        if not word:
            return []
        x = word[0]
        y = self.sigma[g].index(x)
        rest = self._inverse_word(self.sections[g][y], word[1:])
        return [y] + rest

    def _apply_word(self, sec_word, word):
        for (gi, sign) in reversed(sec_word):
            g = self.letters[gi]
            word = (self.state_action(g, word) if sign == F
                    else self.state_inverse_action(g, word))
        return word

    def _inverse_word(self, sec_word, word):
        for (gi, sign) in sec_word:
            g = self.letters[gi]
            word = (self.state_inverse_action(g, word) if sign == F
                    else self.state_action(g, word))
        return word

    # -- words ---------------------------------------------------------------
    def apply(self, word, level_word):
        """Apply a word (tuple of (g, sign)) to a level word."""
        return self._apply_word(tuple(word), level_word)

    def is_identity_on_level(self, word, n):
        d = len(self.sheets)
        for w in product(range(d), repeat=n):
            if self.apply(word, list(w)) != list(w):
                return False
        return True

    def inverse_word(self, word):
        return tuple((g, -s) for (g, s) in reversed(word))

    def _section_state(self, gi, s, x):
        """Section of the single state (gi, s) at letter x, as a word."""
        g = self.letters[gi]
        if s == F:
            return _reduce_word(self.sections[g][x])
        y = self.sigma[g].index(x)
        return _reduce_word(tuple((h, -t) for (h, t) in reversed(self.sections[g][y])))

    def section(self, word, x):
        """The section of a word at letter x: (g w)|_x = g|_{w(x)} . w|_x."""
        word = _reduce_word(tuple(word))
        if not word:
            return ()
        gi, s = word[0]
        rest = word[1:]
        if not rest:
            return self._section_state(gi, s, x)
        x_rest = self.apply(rest, [x])[0]
        return _reduce_word(self._section_state(gi, s, x_rest) + self.section(rest, x))

    def element_sections(self, word):
        return [self.section(word, x) for x in range(len(self.sheets))]

    def _level_signature(self, word, level):
        """The action of `word` on every level-n word, as a hashable key.

        Used to dedupe states by ELEMENT rather than by the word spelling it
        (see `nucleus`)."""
        d = len(self.sheets)
        return tuple(tuple(self.apply(word, list(w)))
                     for w in product(range(d), repeat=level))

    def nucleus(self, cap=4096, depth=64, level=None):
        """Closure of the generators (and inverses) under taking sections.

        Returns (states, closed), with closed False if cap or depth was hit.

        The letters are FREE generators, so a reduced word is an element of
        pi_1 = F_g.  With level=None the dedup is therefore exact in pi_1:
        this is the pi_1-nucleus, and it terminates iff pi_1 is contracting on
        the tree.  That is a SEMI-decision -- halting proves contraction;
        when the nucleus is infinite the enumeration simply runs (the
        non-termination is not an artefact, since the dedup is exact).  It
        says nothing about the IMG, which is a quotient: measured 2026-10-25
        on the shipped twisted-rabbit machine, the pi_1 closure reaches 106
        words and is still open, while the closure by the action on level 4
        is 23 classes and stays 23 at levels 5 through 8.  So the image
        contracts and pi_1 does not: the kernel of pi_1 -> IMG has infinitely
        many sections.  (Author, 2026-10-25, from an informal conversation
        with Nekrashevych: the questions differ, and when IMG contraction
        implies contraction on pi_1^orb is not known.)

        With level=n, states are deduped by their action on X^n, i.e. in the
        image of pi_1 in Sym(X^n).  That universe is FINITE, so this closure
        always terminates (unless cap/depth is hit); `closed` carries no
        information there.  The informative output is the number of classes,
        and whether it stabilises as n grows -- which is the IMG question, up
        to the level-n kernel.  No a-priori separating level is known, and
        whether contraction of the IMG is decidable is open.
        """
        gens = [(self.index[g], F) for g in self.letters] + \
               [(self.index[g], B) for g in self.letters]
        seen = {}
        frontier = []
        for w in gens:
            w = _reduce_word((w,))
            k = w if level is None else self._level_signature(w, level)
            if k not in seen:
                seen[k] = w
                frontier.append((w, 0))
        while frontier:
            w, d = frontier.pop()
            if d >= depth or len(seen) > cap:
                return set(seen.values()), False
            for x in range(len(self.sheets)):
                s = self.section(w, x)
                k = s if level is None else self._level_signature(s, level)
                if k not in seen:
                    seen[k] = s
                    frontier.append((s, d + 1))
        return set(seen.values()), True

    def is_contracting(self, level=None):
        """Is the section closure finite?

        level=None asks about pi_1 (free on the letters): the dedup is exact
        (free reduction), so True proves pi_1 contracts; an infinite nucleus
        just makes the enumeration run, so False is "not proved".  level=n
        asks about the image in Sym(X^n), which is the IMG up to the level-n
        kernel; that closure always terminates (the universe is finite) and
        the number of classes, not the flag, is what matters.  The two can
        disagree -- contraction of the IMG does not imply contraction of
        pi_1, since kernel elements can have infinitely many sections.  See
        `nucleus`.
        """
        return self.nucleus(level=level)[1]

    def level_generators(self, n):
        """Per letter, the permutation on level-n words as a dict
        word-index -> image-index."""
        d = len(self.sheets)
        out = {}
        for gi, g in enumerate(self.letters):
            perm = {}
            for w in range(d ** n):
                word = []
                x = w
                for _ in range(n):
                    word.append(x % d)
                    x //= d
                word.reverse()
                img = self.state_action(self.letters[gi], word)
                v = 0
                for a in img:
                    v = v * d + a
                perm[w] = v
            out[g] = perm
        return out


    def schreier_graph(self, n):
        """The level-n Schreier graph: vertices = level-n words, one
        edge per generator and vertex, w -- g --> g.w."""
        from itertools import product
        d = len(self.sheets)
        vertices = list(product(range(d), repeat=n))
        edges = []
        for w in vertices:
            for g in self.letters:
                img = self.state_action(g, list(w))
                edges.append((w, g, tuple(img)))
        return vertices, edges


class RoseAutomaton(Automaton):
    """Build the automaton directly for a single-vertex (rose) base:
    states are the edge labels, sections the phi-images of the lifts."""

    def __init__(self, base, cover, phi):
        verts = base.vertices()
        if len(verts) != 1:
            raise ValueError("RoseAutomaton: base is not a rose")
        letters = tuple(base.edge_list())
        sheets = tuple(sorted({s for v in cover.vertices() for (_, s) in [v]}))
        sigma = {}
        sections = {}
        for e in letters:
            sig = []
            secs = []
            for s in sheets:
                ch = phi.chains[(e, s)]
                tgt_sheet = cover.edges[(e, s)][1][1]
                # ch is a path in traversal order; a stored section word is
                # a group element with its left factor outermost, so store
                # the reverse (see normalform.normal_form).
                secs.append(tuple((d[0][0], d[1]) for d in reversed(ch)))
                sig.append(tgt_sheet)
            sigma[e] = sig
            sections[e] = secs
        super().__init__(letters, sheets, sigma, sections)

