"""The scripting facade: one object per dataset.

A session is a JSON file -- a ribbon graph Gamma_0, a cover pi: Gamma_1 ->
Gamma_0 and a graph map phi in Harmonic's string format -- plus every result
the engine can derive from it.  `Session` is the whole pipeline behind one
name: each step is a method that computes on first call and caches.

This is the surface scripts should use, and the only surface the desktop app
uses.  The app adds no arithmetic of its own: every number it shows is the
return value of one of these methods (see `pullback.app.panels`).

    from pullback import Session
    s = Session.load("sessions/theta.json")
    s.type_classification()["type"]        # 'hyperbolic type'
    s.automaton()["tree"]                  # ['q']
    s.nucleus()["classes"]                 # 5
    s.level_order(6)                       # 16384
    s.energy_bounds(p=2, n=1)              # [{'n': 1, 'E': ..., 'E_root': ...}]

Nothing here reimplements a computation: each method calls the module that
owns it and lets its exceptions through.  Method calls signal cost;
properties are free.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from .classify import type_classification
from .harmonic_io import check_session_fields, parse_cmap, parse_cover, parse_graph
from .normalform import admissible_trees, normal_form
from .portrait import face_map_phi, face_map_pi
from .validate import spanning_tree, validate


class Session:
    """One dataset, with lazy cached results.

    Attributes
    ----------
    base, cover, phi : the engine's objects (RibbonGraph, RibbonGraph, GraphMap)
    name, path       : a label, and where it was read from (None if in memory)
    raw              : the session dict as loaded (the JSON, unvalidated but
                       field-checked)
    """

    def __init__(self, base, cover, phi, name=None, path=None, raw=None):
        self.base = base
        self.cover = cover
        self.phi = phi
        self.name = name or "session"
        self.path = path
        self.raw = raw if raw is not None else {}
        self._cache = {}

    # ---------------------------------------------------------- construction
    @classmethod
    def load(cls, path):
        """Read a session JSON file.  Raises on a malformed file -- a bad
        dataset is an exception here, for the caller to show or handle."""
        path = Path(path)
        data = json.loads(path.read_text())
        check_session_fields(data)
        base = parse_graph(data["graph"])
        cover, _sheets = parse_cover(base, data["cover"])
        phi = parse_cmap(cover, base, data["phi"])
        return cls(base, cover, phi, name=path.stem, path=str(path), raw=data)

    @classmethod
    def from_dict(cls, data, name=None):
        """Build from a session dict without touching the filesystem."""
        check_session_fields(data)
        base = parse_graph(data["graph"])
        cover, _sheets = parse_cover(base, data["cover"])
        phi = parse_cmap(cover, base, data["phi"])
        return cls(base, cover, phi, name=name, raw=data)

    def as_dict(self):
        """The session as the JSON it round-trips through."""
        return dict(self.raw)

    # -------------------------------------------------------------- helpers
    def _memo(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    @property
    def degree(self):
        """deg(pi): cover vertices over base vertices (an exact division,
        since pi is a covering)."""
        return len(self.cover.vertices()) // len(self.base.vertices())

    def labels(self):
        """The base edge labels, in the graph's order."""
        return list(self.base.edge_list())

    def __repr__(self):
        return f"<Session {self.name!r} degree {self.degree}>"

    # ------------------------------------------------------------ the steps
    def validation(self):
        """The plan-100 bundle: cover connected, base connected, pi_1
        surjective."""
        return self._memo("validation", lambda: validate(self.phi, self.base))

    def portrait(self):
        """The critical portrait (plan item 120).

        Returns a dict with `base_faces`, `cover_faces` (the rows of
        `face_map_pi`: face, projection, target base face, degree),
        `phi_faces` (where phi sends each cover face; target None means
        collapsed), `degrees` (the portrait degree multiset) and
        `unprojected` (cover faces whose boundary is not a power of a base
        face -- a data error for a genuine cover).
        """
        def run():
            rows, base_walks = face_map_pi(self.cover, self.base)
            phi_rows, _ = face_map_phi(self.phi)
            return {
                "base_faces": [tuple(f) for f in base_walks],
                "cover_faces": rows,
                "phi_faces": phi_rows,
                "degrees": [r["degree"] for r in rows
                            if r["degree"] is not None],
                "unprojected": [i for i, r in enumerate(rows)
                                if r["target"] is None],
            }
        return self._memo("portrait", run)

    def type_classification(self):
        """Hyperbolic / cycle without branch point (plan item 320).

        Raises for data that is not a ribbon presentation (e.g. a session
        with a non-ribbon map, or the Grigorchuk group, which is not a
        ribbon-boundary map at all) -- the refusal is the answer, not a bug.
        """
        return self._memo(
            "type", lambda: type_classification(self.base, self.cover, self.phi))

    def automaton(self):
        """The rose normal form, and the tree it was built on.

        Returns {"automaton", "tree", "admissible"}.  An admissible tree is
        preferred.  Where none exists the engine's shortcut is not exact
        (the section words are not those read off the generators' lifts),
        so the automaton is built on the first spanning tree with
        `require_admissible=False`: a valid presentation of the same virtual
        endomorphism, different section words.  `admissible` says which case
        this is; nothing about the group depends on it.
        """
        def run():
            cands = admissible_trees(self.base, self.cover, self.phi)
            if cands:
                tree = sorted(cands[0])
                return {"automaton": normal_form(self.base, self.cover, self.phi,
                                                 tree=tree),
                        "tree": tree, "admissible": True}
            tree, _gens = spanning_tree(self.base)
            tree = sorted(tree)
            return {"automaton": normal_form(self.base, self.cover, self.phi,
                                             tree=tree,
                                             require_admissible=False),
                    "tree": tree, "admissible": False}
        return self._memo("automaton", run)

    def nucleus(self, level=None, cap=4096, depth=64):
        """Size and closure of the section closure of the generators.

        `level=None` is the pi_1 nucleus (dedup on reduced words): closed
        True proves pi_1 contracts, closed False means "not proved" -- a
        semi-decision, not a verdict.  `level=n` dedups by the action on
        X^n, so it always terminates (the universe is finite); the class
        count is the IMG statement up to the level-n kernel.
        """
        return self._memo(("nucleus", level, cap, depth), lambda: self._nucleus(level, cap, depth))

    def _nucleus(self, level, cap, depth):
        words, closed = self.automaton()["automaton"].nucleus(
            cap=cap, depth=depth, level=level)
        return {"classes": len(words), "closed": closed, "level": level,
                "cap": cap, "depth": depth}

    def level_order(self, n):
        """Order of the image of pi_1 in Sym(X^n) -- the engine's level-n
        group, by Schreier-Sims (sympy) on `level_generators`."""
        def run():
            a = self.automaton()["automaton"]
            perms = a.level_generators(n)
            d = len(a.sheets)
            N = d ** n
            from sympy.combinatorics import Permutation, PermutationGroup
            gens = [Permutation([perms[g][w] for w in range(N)], size=N)
                    for g in perms]
            return PermutationGroup(gens).order()
        return self._memo(("order", n), run)

    def energy_bounds(self, p=2, n=1):
        """The lifting bound on E^p_p(phi^n) (plan item 180), n = 1..`n`.

        Upper bounds on the asymptotic energy; exact for integral p.  The
        metric is the session's own (no [Elast] section 6 minimisation), so
        a bound of exactly 1 says the representative is unit-speed, not that
        the map is a certificate.
        """
        p = Fraction(p)

        def run():
            from .bounds import upper_bound
            rows = upper_bound(self.cover, self.phi, p, n)
            return [{"n": k, "E": float(e), "E_root": float(r)}
                    for (k, e, r, _beta) in rows]
        return self._memo(("energy", p, n), run)


def load(path):
    """`Session.load`, for scripts that read top to bottom."""
    return Session.load(path)


def library(root=None):
    """Every session in `sessions/`, name -> Session, sorted by name."""
    if root is None:
        root = Path(__file__).resolve().parents[2] / "sessions"
    out = {}
    for path in sorted(Path(root).glob("*.json")):
        out[path.stem] = Session.load(path)
    return out
