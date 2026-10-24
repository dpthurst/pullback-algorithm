"""Self-similar automata from graph virtual endomorphisms.

An Automaton carries the wreath recursion of a graph virtual
endomorphism in normal form (plan item 130/140):

  letters  = the free generators of pi_1(Gamma_0)
  sheets   = the covering sheets (letters of the tree)
  sigma[g] = the sheet permutation of g's lifts
  sections[g][i] = the phi-image of the lift of g at sheet i, as a word
                   in the generators (signs = orientations)

Level-n action and the group operations (products, inverses, sections,
nucleus, contracting test) are implemented by the wreath recursion
g(x.w) = sigma_g(x) . (g|_x)(w), with words acting by composition
(left factor outermost).
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

    def nucleus(self, cap=4096, depth=64):
        """Closure of the generators (and inverses) under taking
        sections; returns (elements, closed) with closed False if the
        cap or depth was hit."""
        gens = [(self.index[g], F) for g in self.letters] + \
               [(self.index[g], B) for g in self.letters]
        seen = set()
        frontier = []
        for w in gens:
            w = _reduce_word((w,))
            if w not in seen:
                seen.add(w)
                frontier.append((w, 0))
        while frontier:
            w, d = frontier.pop()
            if d >= depth or len(seen) > cap:
                return seen, False
            for x in range(len(self.sheets)):
                s = self.section(w, x)
                if s not in seen:
                    seen.add(s)
                    frontier.append((s, d + 1))
        return seen, True

    def is_contracting(self):
        elements, closed = self.nucleus()
        return closed

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
                src_sheet = cover.edges[(e, s)][0][1]
                tgt_sheet = cover.edges[(e, s)][1][1]
                secs.append(tuple((d[0][0], d[1]) for d in ch))
                sig.append(tgt_sheet)
            sigma[e] = sig
            sections[e] = secs
        super().__init__(letters, sheets, sigma, sections)

