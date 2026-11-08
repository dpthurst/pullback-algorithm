"""Initial geometric data for a graph virtual endomorphism, with no analytic map.

What this module does
---------------------
A session (`pullback.session.Session`) carries a graph virtual endomorphism as
three combinatorial objects: the base ribbon graph Gamma_0, a covering
pi: Gamma_1 -> Gamma_0 and a graph map phi: Gamma_1 -> Gamma_0.  Nothing in it
is geometric, and nothing here reads an analytic rational map, its critical
values or its preimages.

This module turns *that* into geometric data:

    points         one explicit point of C-hat per puncture, with labels
    spine          the Voronoi spine of those points -- a genuine spine of
                   S^2 - P (Voronoi faces are the punctures, one per point)
    h              a peripheral-preserving homotopy equivalence
                   spine -> Gamma_0, found by an exhaustive combinatorial
                   search (no geometry used in the search)
    cover          the transported virtual endomorphism (P, pi', phi') =
                   lift.pullback_ve(..., h), with the frozen lifted cell
                   complex dual_cells(P)
    labels         which cell of the lifted cover is which puncture, and the
                   dynamics of the puncture labels
    step / iterate the frozen pullback step: move the base points, let every
                   lifted edge inherit its base edge's chordal length,
                   discretely-conformally realise the frozen lifted cells
                   (rounding_multi), read the labelled vertices.

The realisation is deliberately *arbitrary*: the points are a deterministic
but otherwise unconstrained spread, and several distinct h exist for the same
session (360 for the rabbit with edge words of length <= 1).  Choosing among
them is an initialisation choice, not a missing input: the certificates below
say which choices are admissible (pi_1-surjective and peripheral-preserving)
and the limit is compared, not asserted.

`certificates()` re-derives the structural facts from the constructed objects;
`purity_guard()` is a regression net for the analytic entry points registered
below, not a proof that no analytic data can be reached by other Python paths.
"""

from __future__ import annotations

import contextlib
import itertools

import mpmath as mp
import numpy as np

from pullback import mesh
from pullback.homotopy import is_peripheral_preserving, peripheral_images
from pullback.lift import (active_puncture_labels, dual_cells,
                           lifted_delaunay_metric, pullback_ve,
                           round_realise_cells)
from pullback.maps import make_graph_map
from pullback.portrait import face_map_pi, face_map_phi
from pullback.ribbons import B, F
from pullback.validate import is_pi1_surjective

mp.mp.dps = 40

__all__ = ["spread_points", "find_transports", "VEInit", "purity_guard",
           "chart"]


# --------------------------------------------------------------------- points
def spread_points(n, seed=0):
    """`n` points of C-hat, deterministic and unconstrained (a Fibonacci sphere).

    `seed` only moves their common phase about the z-axis.  It therefore gives
    an exact rotation (and preserves all chordal distances), not an arbitrary
    perturbation.  Pass `points=` directly for genuinely different geometric
    initial data; Nothing HERE reads an analytic configuration.
    """
    out = []
    for k in range(n):
        z = 1.0 - 2.0 * (k + 0.5) / n
        r = float(np.sqrt(max(0.0, 1.0 - z * z)))
        phi = np.pi * (3.0 - np.sqrt(5.0)) * k + seed
        out.append(mesh.to_point(np.array([r * np.cos(phi), r * np.sin(phi), z])))
    return out


# ----------------------------------------------------------------- transports
def find_transports(spine, base, max_word=1, limit=None):
    """Every `h: spine -> base` that is pi_1-surjective and peripheral-preserving.

    Edges of `spine` are sent to base edge words of length <= `max_word`
    (all vertices go to a base vertex; a bouquet target has only one).  The two
    tests are the ones `pullback_ve` demands with `validate_h=True`: equal rank
    plus pi_1-surjectivity makes `h` a homotopy equivalence, and
    `is_peripheral_preserving` makes it induce a permutation of the punctures
    rather than an arbitrary `Out(F_n)` element (the twist that a
    basis-only transport supplies -- measured on this repo before: a
    `basis_transport` candidate whose fourth face walk is `a^-1 b c^-1`).
    """
    darts = [(e, o) for e in base.edge_list() for o in (F, B)]
    words = [()] + [(d,) for d in darts]
    if max_word >= 2:
        words += [(d1, d2) for d1 in darts for d2 in darts]
    vimages = {v: base.vertices()[0] for v in spine.vertices()}
    sols = []
    for combo in itertools.product(words, repeat=len(spine.edge_list())):
        chains = dict(zip(spine.edge_list(), combo))
        try:
            h = make_graph_map(spine, base, chains, vimages)
        except Exception:
            continue
        if is_pi1_surjective(h, base) and is_peripheral_preserving(h):
            sols.append(chains)
            if limit and len(sols) >= limit:
                break
    return sols


def chart(config):
    """The Moebius-invariant labelled coordinate of a configuration."""
    from pullback.discrete import chart_coordinate
    return chart_coordinate([mesh.to_vector(p) for p in config])


# ------------------------------------------------------------------- the state
class VEInit:
    """Initial geometric data + frozen lifted cover of one session.

    Attributes
    ----------
    points, labels      : puncture positions (labels are point indices)
    spine, face_points  : the geometric spine and spine face -> point label
    h, marking          : the chosen transport and the puncture permutation
    P, pi_prime, phi_prime, cells : the frozen lifted cover
    dynamics            : label -> label, read off the two legs of the VE
    lifted_labels       : label -> vertex of the lifted cell complex
    metric              : the initial lifted metric (chordal lengths, cone angles)
    """

    def __init__(self, session, points=None, seed=0, chains=None, max_word=1):
        self.session = session
        n = len(session.base.faces())
        self.points = list(points) if points is not None else spread_points(n, seed)
        if len(self.points) != n:
            raise ValueError(f"{n} punctures need {n} points, got {len(self.points)}")
        self.spine = spine = mesh.voronoi_spine(self.points)
        self.face_points = mesh.voronoi_face_points(spine, n)
        sols = find_transports(spine, session.base, max_word=max_word)
        if not sols:
            raise ValueError("no peripheral-preserving homotopy equivalence found")
        self.transports = len(sols)
        self.chains = dict(chains) if chains is not None else sols[0]
        self.h = h = make_graph_map(
            spine, session.base, self.chains,
            {v: session.base.vertices()[0] for v in spine.vertices()})
        self.marking = peripheral_images(h)
        P, pi_p, phi_p, _pc = pullback_ve(session.base, session.cover, session.phi,
                                          spine, h)
        self.P, self.pi_prime, self.phi_prime = P, pi_p, phi_p
        self.cells = dual_cells(P)
        self.base_faces = mesh.delaunay_faces(self.points)
        self.pi_rows, self.phi_rows = (face_map_pi(P, spine)[0],
                                       face_map_phi(phi_p)[0])
        dyn, active = {}, {}
        for f, row in enumerate(self.phi_rows):
            if row["target"] is None:
                continue
            p = self.face_points[row["target"]]
            q = self.face_points[self.pi_rows[f]["target"]]
            if p in dyn and dyn[p] != q:
                raise ValueError(f"label {p} sits over two spine faces")
            dyn[p] = q
            active[p] = f
        self.dynamics = dict(sorted(dyn.items()))
        self.active = dict(sorted(active.items()))
        self.lifted_labels = active_puncture_labels(
            phi_p, self.face_points, pi_p, self.dynamics)
        if self.lifted_labels != self.active:
            raise ValueError("the two label derivations disagree")
        self.metric = lifted_delaunay_metric(self.cells, spine, self.points,
                                             base_faces=self.base_faces)
        self.x = None
        self.convergence = session.convergence()

    # -------------------------------------------------------------- geometry
    def step(self, config, x0=None, require_embedded=False):
        """One frozen pullback step at the configuration `config`.

        `config[i]` is the position of the puncture labelled `i` (a point index
        of the initial realisation; the labelling never changes).
        """
        pts = [mp.mpc(c) if mp.isfinite(c) else mp.inf for c in config]
        metric = lifted_delaunay_metric(self.cells, self.spine, pts,
                                        base_faces=self.base_faces)
        out = round_realise_cells(self.cells, metric,
                                  x0=self.x if x0 is None else x0, maxit=400,
                                  require_embedded=require_embedded)
        self.x = out["x"]
        new = {p: mesh.to_point(out["x"][v])
               for p, v in self.lifted_labels.items()}
        if set(new) != set(range(len(pts))):
            raise ValueError(f"the step read labels {sorted(new)}")
        out = dict(out)
        out["config"] = [new[i] for i in range(len(pts))]
        out["coordinate"] = chart(out["config"])
        return out

    def iterate(self, config, n, **kw):
        from .convergence import require_convergence_allowed
        require_convergence_allowed(self.session)
        cur = list(config)
        hist = []
        for _ in range(int(n)):
            out = self.step(cur, **kw)
            cur = out["config"]
            hist.append(out)
        return hist

    def fixed_displacement(self, config):
        """Moebius-corrected distance between `config` and one step of it.

        The step is only defined up to Moebius, so the comparison is on the
        labelled chart, not on the points.
        """
        out = self.step(config)
        return abs(complex(out["coordinate"]) - complex(chart(config)))

    # --------------------------------------------------------- certificates
    def certificates(self, realise=True):
        """Structural facts, re-derived from the constructed objects."""
        spine, base, P = self.spine, self.session.base, self.P
        cert = {}
        cert["punctures"] = len(self.points)
        cert["spine_counts"] = {"V": len(spine.vertices()),
                                "E": len(spine.edge_list()),
                                "F": len(spine.faces())}
        cert["spine_euler"] = (cert["spine_counts"]["V"]
                               - cert["spine_counts"]["E"]
                               + cert["spine_counts"]["F"])
        cert["base_counts"] = {"V": len(base.vertices()),
                               "E": len(base.edge_list()),
                               "F": len(base.faces())}
        cert["cover_counts"] = {"V": len(P.vertices()),
                                "E": len(P.edge_list()),
                                "F": len(P.faces())}
        cert["cover_degree"] = len(P.vertices()) // len(spine.vertices())
        cert["session_degree"] = self.session.degree
        cert["cells_counts"] = dict(self.cells["counts"])
        cert["cells_euler"] = (self.cells["counts"]["V"]
                               - self.cells["counts"]["E"]
                               + self.cells["counts"]["F"])
        # ribbon face correspondence: the multiset of local degrees over every
        # spine face must be the session's own, transported by the marking
        def profile(cover, target, rows):
            out = {}
            for row in rows:
                if row["target"] is None:
                    raise ValueError("a cover face has no target")
                out.setdefault(row["target"], []).append(row["degree"])
            return {k: sorted(v) for k, v in out.items()}
        want = profile(self.session.cover, base,
                       face_map_pi(self.session.cover, base)[0])
        got = profile(P, spine, self.pi_rows)
        cert["degree_profile"] = got
        cert["degree_profile_matches_session"] = (
            {self.marking[k]: v for k, v in got.items()} == want)
        cert["degrees_sum"] = sum(sum(v) for v in got.values())
        # links: at every dual vertex the incident dual faces form one cycle
        dual = self.cells["dual"]
        at = {}
        for k, walk in enumerate(self.cells["corners"]):
            for dart in walk:
                at.setdefault(dual.src(dart), []).append(k)
        links_ok = True
        for v, fs in at.items():
            if len(fs) != len(set(fs)):
                links_ok = False
        cert["links_single_cycle"] = links_ok
        cert["non_simple_dual_faces"] = list(self.cells["non_simple_faces"])
        cert["cone_angles"] = {k: float(v) for k, v in
                               self.metric["cone_angles"].items()}
        cert["hyperideal"] = list(self.metric["hyperideal"])
        cert["dynamics"] = dict(self.dynamics)
        cert["lifted_labels"] = dict(self.lifted_labels)
        cert["marking"] = list(self.marking)
        cert["transports_found"] = self.transports
        if realise:
            out = round_realise_cells(self.cells, self.metric, maxit=400)
            cert["residual"] = float(out["residual"])
            cert["excess"] = int(out["excess"])
            cert["canonical"] = bool(out["canonical"])
            cert["embedding"] = out.get("embedding")
            cert["embedded"] = bool((out.get("embedding") or {}).get("embedded"))
        return cert


# ------------------------------------------------------------------- purity
ANALYTIC_ENTRY_POINTS = [
    ("pullback.rational", "RationalMap"),
    ("pullback.discrete", "pcf_data"),
    ("pullback.discrete", "pullback_step"),
    ("pullback.iteration", "critical_values"),
    ("pullback.iteration", "single_preimage_vertices"),
    ("pullback.iteration", "base_mesh_with_critical_values"),
    ("pullback.mesh", "lift_by_edges"),
    ("pullback.mesh", "lift_path"),
    ("pullback.thurston", "rabbit_map"),
    ("pullback.thurston", "img_golden_maps"),
    ("pullback.thurston", "strictly_rational_rabbit"),
]


@contextlib.contextmanager
def purity_guard(names=None):
    """Patch the registered analytic entry points so that touching one raises.

    This is a name-level regression net, not a purity proof: a previously
    imported alias or an unregistered analytic path is outside its scope.  A
    constructor using one of the listed map, critical-value or preimage entry
    points fails with `AssertionError`.  Every patch is restored on exit.
    """
    import importlib
    saved = []
    try:
        for modname, attr in (names or ANALYTIC_ENTRY_POINTS):
            mod = importlib.import_module(modname)
            if not hasattr(mod, attr):
                continue
            saved.append((mod, attr, getattr(mod, attr)))

            def boom(*_a, __name=f"{modname}.{attr}", **_k):
                raise AssertionError(f"analytic entry point {__name} was used")

            setattr(mod, attr, boom)
        yield
    finally:
        for mod, attr, old in saved:
            setattr(mod, attr, old)
