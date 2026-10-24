"""Self-similar automata from graph virtual endomorphisms (rose case).

For the present scope (M1/M2 boundary) the base must be a single-vertex
rose: its edge labels are the free generators.  The automaton has one
state per base letter; for generator g and sheet i the lift (g, i) of g
starts at sheet i, ends at sheet sigma_g(i), and its phi-image is the
section word (in base letters).  Level-n action is computed by the
wreath recursion g(x.w) = sigma_g(x) . (g|_x)(w).

State-word conventions are irrelevant to the generated subgroup at each
level (a generator's action is defined up to the choice of side), which
is what the equality checks compare.
"""

from __future__ import annotations

from .ribbons import F, B


class RoseAutomaton:
    def __init__(self, base, cover, phi):
        verts = base.vertices()
        if len(verts) != 1:
            raise ValueError("RoseAutomaton: base is not a rose")
        self.letters = tuple(base.edge_list())      # state names
        self.index = {e: i for i, e in enumerate(self.letters)}
        sheets = sorted({s for v in cover.vertices() for (_, s) in [v]})
        self.sheets = tuple(sheets)
        self.sheet_index = {s: i for i, s in enumerate(sheets)}
        # sections[g][i] = tuple of (state_index, sign); sigma[g][i] = j
        self.sections = {}
        self.sigma = {}
        for e in self.letters:
            secs = []
            sig = []
            for s in self.sheets:
                ch = phi.chains[(e, s)]
                src = cover.edges[(e, s)][0]
                tgt = cover.edges[(e, s)][1]
                # sections are loops at the (single) base vertex
                secs.append(tuple((self.index[d[0]], d[1]) for d in ch))
                sig.append(self.sheet_index[tgt[1]])
            self.sections[e] = secs
            self.sigma[e] = sig

    # -- wreath recursion ---------------------------------------------------
    def _apply_step(self, letter_idx, word, depth):
        """Act by the single state with index letter_idx on a level-word
        (list of sheet indices) of length `depth`; returns the image list."""
        if not word:
            return []
        g = self.letters[letter_idx]
        x = word[0]
        out = self.sigma[g][x]
        sec = self.sections[g][x]
        rest = word[1:]
        # apply the section word to the remaining letters
        rest = self._apply_word(sec, rest)
        return [out] + rest

    def _apply_word(self, sec_word, word):
        # composition: apply rightmost first
        for (gi, sign) in reversed(sec_word):
            if sign == F:
                word = self._apply_step(gi, word, len(word))
            else:
                word = self._inverse_step(gi, word)
        return word

    def _inverse_step(self, gi, word):
        if not word:
            return []
        g = self.letters[gi]
        x = word[0]
        # find sheet y with sigma_g(y) = x
        y = self.sigma[g].index(x)
        sec = self.sections[g][y]
        rest = self._inverse_word(sec, word[1:])
        return [y] + rest

    def _inverse_word(self, sec_word, word):
        for (gi, sign) in sec_word:
            if sign == F:
                word = self._inverse_step(gi, word)
            else:
                word = self._apply_step(gi, word, len(word))
        return word

    def generator_action(self, g, n):
        """Permutation of the state g on level-n words, as a dict
        word -> image."""
        d = len(self.sheets)
        out = {}
        for w in range(d ** n):
            word = []
            x = w
            for _ in range(n):
                word.append(x % d)
                x //= d
            word.reverse()
            img = self._apply_step(self.index[g], word, n)
            v = 0
            for a in img:
                v = v * d + a
            out[w] = v
        return out

    def level_generators(self, n):
        return {g: self.generator_action(g, n) for g in self.letters}
