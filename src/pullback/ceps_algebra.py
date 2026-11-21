"""Convention-pinning algebra kernel for the GSC §7 special-vertex spherical CEPS.

This module contains **no optimiser and no certification of outputs**.  It is a
transcription of the exact equations of the two primary sources into executable
form, together with the symbolic/numeric identity checks that pin the
conventions.  Everything here is meant to be imported by the actual solver
(``src/pullback/spherical_ceps.py``) so that the sign conventions
(delta -> bounds -> energy) are fixed in exactly one place.

Primary sources (line numbers refer to the checked-in texts)::

    GSC = refs/gillespie-springborn-crane-ceps.txt
          §§3.5, 4.2.1-4.2.3, 7.1, 7.2, 7.3, 7.4
    SPR = refs/springborn-1707.06848.txt
          Rem. 3.1, Props. 3.2, 3.3, 3.4, 3.6, Thm. 4.7 (10), Def. 5.9,
          eq. (13), Props. 5.10, 5.16, Def. 5.14, Def. 6.1, Def. 7.1,
          Prop. 7.2, Def. 7.16, eq. (56), Prop. 7.17 (i)-(v), Thm. 7.18,
          Thm. 10.5.

Conventions, all verified in ``tests/test_ceps_algebra.py``
----------------------------------------------------------

1.  **Penner coordinates.**  ``lam[e]`` is the *signed distance of horocycles*
    (SPR §3), not Penner's lambda-length.  The lambda-length is

        ell = exp(lam / 2)                                       (SPR 3.1, (8))

    and a Euclidean face ``(i, j, k)`` with Penner coordinates
    ``lam_ij, lam_jk, lam_ki`` is the Euclidean triangle with **sides**
    ``ell_ij, ell_jk, ell_ki`` (SPR Def. 7.1: sides ``e^{x_j}`` with
    ``x = lam/2``).  ``L = 2 log ell`` is therefore the Euclidean PL metric
    convention of the pullback project: ``lam = 2 log(length)``.

2.  **Fiber / scaling.**  ``Lam(u)_e = lam_e + u_v1 + u_v2`` (SPR Prop. 3.2 (2)),
    i.e. ``GSC``'s ``lam'_ij = lam_ij + u_i + u_j``.  Moving the horocycle at
    ``v`` by ``u_v`` *towards the cusp* shrinks it: the horocyclic arc at ``v``
    scales by ``exp(-u_v)`` (GSC just below eq. (17)).  ``u_v = +inf`` means the
    horocycle at ``v`` is *missing* (SPR Def. 5.9) and all its arcs are zero.

3.  **Horocyclic arcs.**  In the face ``(i, j, k)`` with ``lam_i := lam_jk`` the
    edge opposite ``i``,

        a_i^{jk} = exp((lam_i - lam_j - lam_k) / 2) = ell_jk / (ell_ij ell_ki)
                                                                 (SPR 3.3 (3), GSC (17))

    Note this is *not* scale invariant -- it depends on the decoration, exactly
    as it must, since it is a length in the Poincare model.

4.  **Ideal Delaunay condition.**  Edge ``ij`` of the quadrilateral with
    opposite vertices ``k, l`` is locally ideal Delaunay iff

        a_k^{ij} + a_l^{ij}  <  a_i^{jk} + a_i^{jl} + a_j^{ik} + a_j^{jl}
                                                                       (GSC (18))

    that is, **arcs at the two vertices opposite the edge < arcs at the two
    endpoints of the edge**.  In the partially decorated case every arc at a
    vertex ``v`` carries the weight ``exp(-u_v)`` (SPR (13)); with ``u = +inf``
    on the shrunk vertices, GSC's peacock argument follows
    (``tests/test_ceps_algebra.py::test_delaunay_arc_condition_is_the_euclidean_one``
    proves this pairing *is* the Euclidean "opposite angles sum <= pi" test, and
    that the other pairing is not).

    .. warning::
       SPR (10)/(13) call the two arcs at the vertices *opposite* the edge the
       arcs "incident with e".  Read with SPR's own phrase taken literally
       ("incident" = at the endpoints of ``e``) the inequality is the reverse of
       the geometric one.  The numeric identity test decides it; this module
       uses GSC (18) with SPR's weights, which is the geometrically correct
       condition.

5.  **Triangle function** (SPR Def. 7.1, eq. (22))::

        f(x1,x2,x3) = a1 x1 + a2 x2 + a3 x3 + L(a1) + L(a2) + L(a3)

    where ``a_j`` is the angle opposite the side ``exp(x_j)`` and
    ``L(a) = -int_0^a log|2 sin t| dt`` is Milnor's Lobachevsky function.
    SPR Prop. 7.2::

        f(x + h 1) = f(x) + pi h                                (24)
        d f / d x_i = a_i                                       (25)
        D^2 f = cot(a1)(dx2-dx3)^2 + cot(a2)(dx3-dx1)^2 + cot(a3)(dx1-dx2)^2   (26)

6.  **Special-vertex energy** (SPR eq. (56) = GSC §7.2)::

        Ebar(u) = sum_{t in F°} 2 f(lam~_e1/2, lam~_e2/2, lam~_e3/2)
                  - pi * sum_{e in E°} lam~_e
                  - 2 pi * sum_{v in V°} (log c_v(Delta, lam) - u_v)         (56)

    where ``V°, E°, F°`` are the cells of the disk ``T° = T \\ star(v_inf)``,
    ``lam~ = Lam(u)``, and ``c_v`` is the total horocycle length at ``v``
    (SPR Prop. 3.4, eq. (5)).  ``log c_v`` does not depend on ``u``, so GSC's
    written form ``2 pi sum u_i - pi sum lam~_ij + 2 sum f`` differs from (56)
    by the constant ``+2 pi sum_v log c_v``; ``energy(..., variant=...)``
    implements both.

    Gradient (SPR (58) = GSC §7.2)::

        d Ebar / d u_v = -Theta~_v + pi (deg2(v) - deg1(v) + 2)

    with ``Theta~_v`` the total angle at ``v`` in the flat disk metric, and
    ``deg1, deg2`` the edge- and triangle-degrees *in the disk*.  Hence
    interior vertices: ``2 pi - Theta~_v``; boundary vertices: ``pi - Theta~_v``.

    Hessian (SPR (59))::

        D^2 Ebar = (1/4) sum_{e in E°} w_e (d u_v1(e) - d u_v2(e))^2,
        w_e = cot a_e + cot a'_e  (one term on the boundary, 0 if no face)

    Scaling (SPR (60)): ``Ebar(u + h 1) = Ebar(u) + 2 pi h``.  In the written
    formulas the coefficient is ``|F°| - |E°| + |V°| = chi(disk) = 1`` times
    ``2 pi h``; ``tests`` also checks the *unbounded* direction: ``Ebar`` is
    linear of slope ``2 pi`` along ``1`` and is **not coercive**, so without the
    bounds the problem has no minimiser.

7.  **Bounds and the sign of delta** (SPR Def. 5.14, Prop. 5.16, Thm. 7.18 (61)::

        minimise Ebar(u)  subject to  u_v >= delta_{Delta,lam}(v, v_inf)

    where ``delta_{Delta,lam}(v1,v2)`` is the signed shortest horocycle
    distance (Def. 5.14), computed by Prop. 5.16: keep the horocycle at ``v2``,
    shrink every other horocycle to a point (``u = 0`` at ``v2``, ``+inf``
    elsewhere), flip to the adjusted Delaunay triangulation, and read the
    Penner coordinates of the new edges ``v1 v2`` **in the original
    decoration** (GSC §7.3: "read off the distances between the original
    horocycles").  Ptolemy flips commute with ``Lam(u)`` (both sides of SPR (9)
    are multiplied by the same factor), so the flip *sequence* may be computed
    in the degenerate decoration while the *values* are flipped from ``lam``.

    **Sign reconciliation.**  GSC §7.3 writes the constraint as
    ``u_j >= -lam_{i*j}``; SPR Thm. 7.18 (61) writes ``u_v >= delta``.  They
    differ in sign.  ``delta`` is right: by Prop. 3.2 moving the horocycle at
    ``v`` by ``u_v`` towards the cusp changes its signed distance to the
    horocycle at ``v_inf`` by ``-u_v``, so ``u_v = delta_{Delta,lam}(v,v_inf)``
    is exactly the tangency condition "the horocycle at ``v`` touches the
    (missing) horocycle at ``v_inf``" -- which is the geometry of the
    realisability condition (r2) of SPR Def. 6.1 and of the fixed
    triangulation statement quoted by GSC §7.3 (Bobenko et al. 2015,
    Prop. 3.2.1: the boundary edges get the right cross ratio).  The minimiser
    of ``Ebar`` sits on the bound because ``Ebar`` decreases along ``-1``.
    ``bound_residual`` implements ``delta_v - u_v`` with the documented sign.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Hashable, Iterable, Iterator, Mapping, Sequence

import numpy as np

__all__ = [
    "DiskTriangulation",
    "lambda_length",
    "penner_scale",
    "face_angles",
    "face_arcs",
    "lobachevsky",
    "triangle_f",
    "triangle_f_grad",
    "triangle_f_hess",
    "horocycle_length",
    "energy",
    "energy_grad",
    "energy_grad_closed_form",
    "energy_hess",
    "energy_hess_dense",
    "scaling_defect",
    "local_delaunay_margin",
    "is_locally_ideal_delaunay",
    "ptolemy_flip",
    "delta_from_peacock",
    "bound_residual",
    "bound_active",
    "orient_faces",
    "flip_edge",
    "peacock",
    "delta_certified",
    "PeacockAmbiguity",
    "delta_arc_search",
]


# ---------------------------------------------------------------------------
# Penner coordinates
# ---------------------------------------------------------------------------


def lambda_length(lam: float) -> float:
    """Lambda-length ``ell = exp(lam/2)`` of a signed horocycle distance (SPR (8))."""
    return math.exp(0.5 * lam)


def penner_scale(lam: float, u: Mapping[Hashable, float], i: Hashable, j: Hashable) -> float:
    """``lam~_ij = lam_ij + u_i + u_j`` (SPR Prop. 3.2 (2) = GSC's eq. (1)-(3))."""
    return lam + u[i] + u[j]


def _triangle_sides(lam_ij: float, lam_jk: float, lam_ki: float) -> tuple[float, float, float]:
    """Euclidean sides ``(ell_ij, ell_jk, ell_ki)`` of the face, ascending check included."""
    return (lambda_length(lam_ij), lambda_length(lam_jk), lambda_length(lam_ki))


def face_angles(lam_ij: float, lam_jk: float, lam_ki: float,
                tol: float = 1e-13) -> tuple[float, float, float]:
    """Euclidean angles ``(a_i, a_j, a_k)`` of the face from its Penner coordinates.

    ``a_i`` is the angle at ``i``, opposite the side ``ell_jk``.  The face must
    satisfy the strict triangle inequality; that is exactly the hypothesis
    ``A`` of SPR Def. 7.1 (``exp(x1) > exp(x2) + exp(x3)`` etc., i.e.
    ``ell_jk < ell_ki + ell_ij``).  A violated inequality is an error, never
    silently clamped: on the boundary of ``A`` the energy is extended by
    SPR Prop. 7.2 (b)/(iv) with limit values, which the solver must handle
    explicitly.
    """
    a, b, c = _triangle_sides(lam_ij, lam_jk, lam_ki)  # a=ell_ij, b=ell_jk, c=ell_ki
    # angle at i is opposite b; at j opposite c; at k opposite a
    # SPR Def. 7.1 defines f on the *open* set A (strict triangle inequalities),
    # so equality is also an error here: the boundary values are limits.
    slack = min(a + b - c, b + c - a, c + a - b)
    if slack <= tol * max(a, b, c):
        if True:
            raise ValueError(
                "face is not a Euclidean triangle: sides %r violate the "
                "triangle inequality (hypothesis A of SPR Def. 7.1)"
                % ((a, b, c),))
    # law of cosines with the numerically best-conditioned expression
    def _ang(opp: float, p: float, q: float) -> float:
        # angle at the vertex where the sides p, q meet, opposite side `opp`
        val = (p * p + q * q - opp * opp) / (2.0 * p * q)
        return math.acos(max(-1.0, min(1.0, val)))

    return (_ang(b, a, c), _ang(c, a, b), _ang(a, b, c))


def face_arcs(lam_ij: float, lam_jk: float, lam_ki: float) -> tuple[float, float, float]:
    """Horocyclic arcs ``(a_i^{jk}, a_j^{ki}, a_k^{ij})`` (SPR Prop. 3.3 (3), GSC (17)).

    ``a_i^{jk} = exp((lam_i - lam_j - lam_k)/2)`` with ``lam_i = lam_jk``.
    """
    return (
        math.exp(0.5 * (lam_jk - lam_ki - lam_ij)),
        math.exp(0.5 * (lam_ki - lam_ij - lam_jk)),
        math.exp(0.5 * (lam_ij - lam_jk - lam_ki)),
    )


# ---------------------------------------------------------------------------
# The triangle function f (SPR Def. 7.1, Prop. 7.2)
# ---------------------------------------------------------------------------


def lobachevsky(alpha: float) -> float:
    """Milnor's Lobachevsky function ``L(a) = -int_0^a log|2 sin t| dt`` (SPR (23)).

    ``L(a) = (1/2) Cl2(2a)`` with Clausen's integral; evaluated through
    ``mpmath`` (30 digits) so that the energy values are accurate enough to
    serve as a reference for finite-difference identity checks.  ``L`` is odd
    and ``pi``-periodic.
    """
    import mpmath

    with mpmath.workdps(30):
        return float(mpmath.clsin(2, 2 * mpmath.mpf(alpha)) / 2)


def clausen2(theta: float) -> float:
    """Clausen's integral ``Cl2(t) = sum_{k>=1} sin(k t)/k^2`` (reference value)."""
    import mpmath

    with mpmath.workdps(30):
        return float(mpmath.clsin(2, mpmath.mpf(theta)))


def _f_angles(x1: float, x2: float, x3: float) -> tuple[float, float, float]:
    """Angles ``(a1, a2, a3)``, where ``a_j`` is opposite the side ``exp(x_j)``.

    ``face_angles(lam_ij, lam_jk, lam_ki)`` returns the angles at ``i, j, k``,
    i.e. opposite the sides ``(ell_jk, ell_ki, ell_ij)``.  With
    ``(lam_ij, lam_jk, lam_ki) = (2x1, 2x2, 2x3)`` the angle opposite
    ``exp(x1) = ell_ij`` is the *third* entry, opposite ``exp(x2)`` the first and
    opposite ``exp(x3)`` the second.
    """
    a_i, a_j, a_k = face_angles(2.0 * x1, 2.0 * x2, 2.0 * x3)
    return (a_k, a_i, a_j)


def triangle_f(x: Sequence[float]) -> float:
    """``f(x1,x2,x3) = sum a_j x_j + sum L(a_j)`` (SPR Def. 7.1, eq. (22))."""
    a1, a2, a3 = _f_angles(*x)
    return (a1 * x[0] + a2 * x[1] + a3 * x[2]
            + lobachevsky(a1) + lobachevsky(a2) + lobachevsky(a3))


def triangle_f_grad(x: Sequence[float]) -> np.ndarray:
    """``df/dx_i = a_i`` (SPR Prop. 7.2 (25)); returned as the angle vector."""
    return np.array(_f_angles(*x))


def triangle_f_hess(x: Sequence[float]) -> np.ndarray:
    """``D^2 f`` from SPR Prop. 7.2 (26), as a symmetric 3x3 matrix.

        cot(a1)(dx2-dx3)^2 + cot(a2)(dx3-dx1)^2 + cot(a3)(dx1-dx2)^2
    """
    a1, a2, a3 = _f_angles(*x)
    c1, c2, c3 = 1.0 / math.tan(a1), 1.0 / math.tan(a2), 1.0 / math.tan(a3)
    # (dx2-dx3)^2 with coefficient c1 contributes c1*(e2-e3)(e2-e3)^T
    e = np.eye(3)
    m = np.zeros((3, 3))
    for c, p, q in ((c1, 1, 2), (c2, 2, 0), (c3, 0, 1)):
        d = e[p] - e[q]
        m += c * np.outer(d, d)
    return m


# ---------------------------------------------------------------------------
# The special-vertex disk
# ---------------------------------------------------------------------------


def _ekey(i: Hashable, j: Hashable) -> tuple[Hashable, Hashable]:
    return (i, j) if i <= j else (j, i)


@dataclass(frozen=True)
class DiskTriangulation:
    """A closed oriented triangulation of a sphere with one distinguished vertex.

    ``faces`` are oriented triples of the whole triangulation (so each edge
    occurs in exactly two faces, with opposite induced orientations).  The
    *disk* ``T°`` is the triangulation with ``v_inf`` and its incident faces and
    edges deleted (GSC §7.2: "the mesh obtained by removing the special vertex
    i* together with its incident edges and faces").
    """

    v_inf: Hashable
    faces: tuple[tuple[Hashable, Hashable, Hashable], ...]

    # -- construction / validation ------------------------------------------------
    @classmethod
    def from_faces(cls, v_inf: Hashable,
                   faces: Iterable[Sequence[Hashable]]) -> "DiskTriangulation":
        tf = tuple(tuple(int(x) if isinstance(x, (int, np.integer)) else x for x in f)
                   for f in faces)
        obj = cls(v_inf=v_inf, faces=tf)
        obj.validate()
        return obj

    def vertices(self) -> tuple[Hashable, ...]:
        return tuple(sorted({v for f in self.faces for v in f}))

    def edges(self) -> tuple[tuple[Hashable, Hashable], ...]:
        return tuple(sorted({_ekey(a, b)
                             for f in self.faces for a, b in
                             ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))}))

    def edge_faces(self) -> dict[tuple[Hashable, Hashable], list[int]]:
        out: dict[tuple[Hashable, Hashable], list[int]] = {}
        for fi, f in enumerate(self.faces):
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                out.setdefault(_ekey(a, b), []).append(fi)
        return out

    def validate(self) -> None:
        verts = set()
        for f in self.faces:
            if len(set(f)) != 3:
                raise ValueError(f"degenerate face {f}")
            verts.update(f)
        if self.v_inf not in verts:
            raise ValueError("v_inf is not a vertex")
        ef = self.edge_faces()
        for e, fl in ef.items():
            if len(fl) != 2:
                raise ValueError(f"edge {e} has {len(fl)} incident faces, expected 2")
        # oriented-edge bookkeeping: each directed edge occurs exactly once
        directed: dict[tuple[Hashable, Hashable], int] = {}
        for f in self.faces:
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                directed[(a, b)] = directed.get((a, b), 0) + 1
        if any(c != 1 for c in directed.values()):
            raise ValueError("faces are not consistently oriented")
        if len(verts) - len(ef) + len(self.faces) != 2:
            raise ValueError("not a sphere: V - E + F != 2")

    # -- disk cells ---------------------------------------------------------------
    @property
    def disk_face_indices(self) -> tuple[int, ...]:
        return tuple(i for i, f in enumerate(self.faces) if self.v_inf not in f)

    def disk_faces(self) -> tuple[tuple[Hashable, Hashable, Hashable], ...]:
        return tuple(self.faces[i] for i in self.disk_face_indices)

    def disk_edges(self) -> tuple[tuple[Hashable, Hashable], ...]:
        return tuple(e for e, fl in self.edge_faces().items()
                     if self.v_inf not in e)

    def disk_vertices(self) -> tuple[Hashable, ...]:
        return tuple(v for v in self.vertices() if v != self.v_inf)

    def disk_edge_faces(self, e: tuple[Hashable, Hashable]) -> list[int]:
        """Indices of the *disk* faces containing ``e`` (0, 1 or 2 of them)."""
        fi = self.edge_faces()[_ekey(*e)]
        return [i for i in fi if i in set(self.disk_face_indices)]

    def deg1(self, v: Hashable) -> int:
        """Edge degree of ``v`` in the disk ``T°``."""
        return sum(1 for e in self.disk_edges() if v in e)

    def deg2(self, v: Hashable) -> int:
        """Triangle degree of ``v`` in the disk ``T°``."""
        return sum(1 for f in self.disk_faces() if v in f)

    def is_boundary_edge(self, e: tuple[Hashable, Hashable]) -> bool:
        return len(self.disk_edge_faces(e)) == 1

    def boundary_vertices(self) -> tuple[Hashable, ...]:
        out = set()
        for e in self.disk_edges():
            if self.is_boundary_edge(e):
                out.update(e)
        return tuple(sorted(out))

    def disk_euler(self) -> int:
        return len(self.disk_vertices()) - len(self.disk_edges()) + len(self.disk_faces())

    # -- angles and arcs ----------------------------------------------------------
    def face_angles_of(self, lam: Mapping[tuple[Hashable, Hashable], float],
                       face: Sequence[Hashable]) -> tuple[float, float, float]:
        i, j, k = face
        return face_angles(lam[_ekey(i, j)], lam[_ekey(j, k)], lam[_ekey(k, i)])

    def face_arcs_of(self, lam: Mapping[tuple[Hashable, Hashable], float],
                     face: Sequence[Hashable]) -> tuple[float, float, float]:
        i, j, k = face
        return face_arcs(lam[_ekey(i, j)], lam[_ekey(j, k)], lam[_ekey(k, i)])

    def scaled_lambda(self, lam: Mapping[tuple[Hashable, Hashable], float],
                      u: Mapping[Hashable, float]) -> dict[tuple[Hashable, Hashable], float]:
        """``Lam(u)`` on the edges whose two endpoints are in ``u``.

        ``u`` is only defined on ``V°`` (the vertex ``v_inf`` has the infinite
        coordinate of the special-vertex construction), so edges incident with
        ``v_inf`` keep their original coordinates: they are exactly the edges
        excluded from ``E°`` in SPR (56), and GSC §7.3 reads their values in the
        original decoration anyway.
        """
        out = {}
        for e, v in lam.items():
            if e[0] in u and e[1] in u:
                out[e] = v + u[e[0]] + u[e[1]]
            else:
                out[e] = v
        return out

    # -- totals -------------------------------------------------------------------
    def total_angle(self, lam: Mapping[tuple[Hashable, Hashable], float],
                    u: Mapping[Hashable, float] | None = None) -> dict[Hashable, float]:
        """``Theta~_v``: total angle around ``v`` in the flat disk metric."""
        if u is not None:
            lam = self.scaled_lambda(lam, u)
        out: dict[Hashable, float] = {v: 0.0 for v in self.disk_vertices()}
        for f in self.disk_faces():
            for v, a in zip(f, self.face_angles_of(lam, f)):
                out[v] += a
        return out

    def horocycle_length(self, lam: Mapping[tuple[Hashable, Hashable], float],
                         v: Hashable) -> float:
        """Total horocycle length ``c_v`` at ``v`` (SPR Prop. 3.4, eq. (5)).

        The total length equals the sum of the horocyclic arcs at ``v`` over the
        faces around ``v`` (SPR's (5) is the same number written in terms of the
        Penner coordinates of the incident edges).
        """
        total = 0.0
        for f in self.disk_faces():
            if v in f:
                idx = f.index(v)
                total += self.face_arcs_of(lam, f)[idx]
        return total


# ---------------------------------------------------------------------------
# Energy, gradient, Hessian (SPR (56), (58), (59); GSC §7.2)
# ---------------------------------------------------------------------------


def energy(disk: DiskTriangulation,
           lam: Mapping[tuple[Hashable, Hashable], float],
           u: Mapping[Hashable, float],
           variant: str = "gsc") -> float:
    """Special-vertex energy ``Ebar(u)`` on the disk.

    ``variant="gsc"``        : ``2 pi sum_{V°} u - pi sum_{E°} lam~ + 2 sum_{F°} f``
    ``variant="springborn"`` : the same plus the constant ``-2 pi sum_{V°} log c_v``
                               of SPR eq. (56).

    The two differ by a ``u``-independent constant (GSC say so explicitly), so
    they have identical gradients, Hessians and minimisers.
    """
    lam_s = disk.scaled_lambda(lam, u)
    e_disk = disk.disk_edges()
    total = 0.0
    for f in disk.disk_faces():
        i, j, k = f
        x = (0.5 * lam_s[_ekey(i, j)], 0.5 * lam_s[_ekey(j, k)], 0.5 * lam_s[_ekey(k, i)])
        total += 2.0 * triangle_f(x)
    total -= math.pi * sum(lam_s[e] for e in e_disk)
    total += 2.0 * math.pi * sum(u[v] for v in disk.disk_vertices())
    if variant == "gsc":
        return total
    if variant == "springborn":
        return total - 2.0 * math.pi * sum(
            math.log(disk.horocycle_length(lam, v)) for v in disk.disk_vertices())
    raise ValueError(f"unknown variant {variant!r}")


def energy_grad(disk: DiskTriangulation,
                lam: Mapping[tuple[Hashable, Hashable], float],
                u: Mapping[Hashable, float]) -> dict[Hashable, float]:
    """``d Ebar / d u_v`` differentiated straight from eq. (56).

    Per face, the two edges at ``v`` contribute; with ``d f / d x_i = a_i`` and
    ``x = lam~/2`` this is ``pi - a_v`` per incident face, whence

        g_v = sum_{t ∋ v} (pi - a_v(t)) - pi deg1(v) + 2 pi.
    """
    lam_s = disk.scaled_lambda(lam, u)
    g = {v: 2.0 * math.pi - math.pi * disk.deg1(v) for v in disk.disk_vertices()}
    for f in disk.disk_faces():
        for v, a in zip(f, disk.face_angles_of(lam_s, f)):
            g[v] += math.pi - a
    return g


def energy_grad_closed_form(disk: DiskTriangulation,
                            lam: Mapping[tuple[Hashable, Hashable], float],
                            u: Mapping[Hashable, float]) -> dict[Hashable, float]:
    """SPR (58) / GSC §7.2 in closed form: ``-Theta~_v + pi (deg2 - deg1 + 2)``."""
    lam_s = disk.scaled_lambda(lam, u)
    theta = disk.total_angle(lam_s)
    return {v: -theta[v] + math.pi * (disk.deg2(v) - disk.deg1(v) + 2)
            for v in disk.disk_vertices()}


def energy_hess(disk: DiskTriangulation,
                lam: Mapping[tuple[Hashable, Hashable], float],
                u: Mapping[Hashable, float]) -> dict[tuple[Hashable, Hashable], float]:
    """Quadratic-form weights of ``D^2 Ebar`` (SPR (59)).

    Returns the edge weights ``(1/2) w_e`` in

        ``D^2 Ebar = sum_e (1/2) w_e (d u_v1 - d u_v2)^2``,

    with ``w_e = cot a + cot a'`` the two cotangents of the angles opposite
    ``e`` (one summand if ``e`` is a boundary edge of the disk; cotangents of
    removed faces are zero, GSC §7.2).

    Equivalently the matrix is the cotan Laplacian with off-diagonal entries
    ``-1/2 (cot a + cot a')`` summed over parallel edges -- GSC §4.2.3.

    .. warning::
       SPR eq. (34) and its special-vertex form (59) print the prefactor
       ``1/4`` in front of ``sum_e w_e (du_v1 - du_v2)^2``.  That is a factor 2
       away from the derivative of the gradient (SPR (33)/(58)), which
       :func:`energy_hess` matches to ``1e-11``.  The chain rule from SPR
       Def. 7.1 + Prop. 7.2 (26) gives, for one face, a Hessian
       ``(1/2) sum_i cot(a_i) u_i u_i^T`` with ``u_i`` the coboundary of the
       edge opposite vertex ``i`` (using ``P = 1 1^T - 1`` and ``D^2 f 1 = 0``),
       i.e. exactly the ``1/2`` used here.  Do not use the printed ``1/4``.
    """
    lam_s = disk.scaled_lambda(lam, u)
    out = {}
    for e in disk.disk_edges():
        w = 0.0
        for fi in disk.disk_edge_faces(e):
            f = disk.faces[fi]
            # the angle opposite e is at the vertex of f not in e
            opp = [v for v in f if v not in e]
            if len(opp) != 1:
                raise RuntimeError("edge/face bookkeeping error")
            idx = f.index(opp[0])
            a = disk.face_angles_of(lam_s, f)[idx]
            w += 1.0 / math.tan(a)
        out[e] = 0.5 * w
    return out


def energy_hess_dense(disk: DiskTriangulation,
                      lam: Mapping[tuple[Hashable, Hashable], float],
                      u: Mapping[Hashable, float]) -> tuple[list[Hashable], np.ndarray]:
    """Dense ``D^2 Ebar`` matrix on ``R^{V°}`` together with the vertex order."""
    verts = list(disk.disk_vertices())
    pos = {v: i for i, v in enumerate(verts)}
    m = np.zeros((len(verts), len(verts)))
    for e, w in energy_hess(disk, lam, u).items():
        i, j = pos[e[0]], pos[e[1]]
        m[i, i] += w
        m[j, j] += w
        m[i, j] -= w
        m[j, i] -= w
    return verts, m


def scaling_defect(disk: DiskTriangulation,
                   lam: Mapping[tuple[Hashable, Hashable], float],
                   u: Mapping[Hashable, float],
                   h: float,
                   variant: str = "gsc") -> float:
    """``Ebar(u + h 1) - Ebar(u) - 2 pi h`` (zero by SPR Prop. 7.17 (v), eq. (60))."""
    u2 = {v: u[v] + h for v in u}
    return energy(disk, lam, u2, variant) - energy(disk, lam, u, variant) - 2.0 * math.pi * h


# ---------------------------------------------------------------------------
# Ideal Delaunay condition and the Ptolemy flip
# ---------------------------------------------------------------------------


def local_delaunay_margin(disk: DiskTriangulation,
                          lam: Mapping[tuple[Hashable, Hashable], float],
                          edge: tuple[Hashable, Hashable],
                          u: Mapping[Hashable, float] | None = None) -> float:
    """Margin of the ideal Delaunay condition (GSC (18), weights from SPR (13)).

    Positive margin means locally Delaunay.  The edge ``ij`` must have two
    opposite vertices ``k, l`` (one in each incident face).  Weighted form::

        margin = sum_{arcs at i, j} a * exp(-u_vertex) - sum_{arcs at k, l} a * exp(-u_vertex)

    ``u=None`` means the undecorated (all-zero) decoration.  ``u[v] = +inf`` is
    allowed and gives weight ``0`` (SPR: "the respective arc length in condition
    (10) is zero" for a missing horocycle).  The arcs are evaluated in the
    *given* decoration and then weighted: this is exact because the scaling law
    ``a_v -> exp(-u_v) a_v`` holds identically (GSC below eq. (17)), so no
    infinite quantity is ever formed.
    """
    i, j = edge
    fi = disk.edge_faces()[_ekey(i, j)]
    if len(fi) != 2:
        raise ValueError("a Delaunay test needs two incident faces")
    def w(v: Hashable) -> float:
        if u is None:
            return 1.0
        uv = u[v]
        return 0.0 if math.isinf(uv) else math.exp(-uv)

    opposite = []
    endpoint_arcs = 0.0
    for f in (disk.faces[fi[0]], disk.faces[fi[1]]):
        arcs = disk.face_arcs_of(lam, f)
        for v, a in zip(f, arcs):
            if v in (i, j):
                endpoint_arcs += a * w(v)
            else:
                opposite.append((v, a))
    opposite_margin = sum(a * w(v) for v, a in opposite)
    return endpoint_arcs - opposite_margin


def is_locally_ideal_delaunay(disk: DiskTriangulation,
                              lam: Mapping[tuple[Hashable, Hashable], float],
                              edge: tuple[Hashable, Hashable],
                              u: Mapping[Hashable, float] | None = None,
                              tol: float = 0.0) -> bool:
    """``local_delaunay_margin(edge) > tol`` -- GSC (18) with strict inequality."""
    return local_delaunay_margin(disk, lam, edge, u) > tol


def ptolemy_flip(lam_a: float, lam_b: float, lam_c: float, lam_d: float,
                 lam_f: float) -> float:
    """Ptolemy flip (SPR Prop. 3.6, eq. (9)): ``ell_e = (ell_a ell_c + ell_b ell_d)/ell_f``.

    ``a, c`` and ``b, d`` are the two pairs of opposite sides of the decorated
    ideal quadrilateral, ``f`` the old diagonal and ``e`` the new one.  Returned
    in logarithmic (Penner) coordinates.  The relation commutes with
    ``Lam(u)`` -- both sides of (9) scale by ``exp((u_i+u_j+u_k+u_l)/2)`` -- which
    is what lets the flip *sequence* be computed in the degenerate decoration
    while the *values* are transported in the original one.
    """
    left = math.exp(0.5 * (lam_a + lam_c))
    right = math.exp(0.5 * (lam_b + lam_d))
    return 2.0 * math.log(left + right) - lam_f


# ---------------------------------------------------------------------------
# delta (SPR Def. 5.14, Prop. 5.16) and the bounds (SPR Thm. 7.18 (61))
# ---------------------------------------------------------------------------


def delta_from_peacock(lam_peacock_spokes: Mapping[tuple[Hashable, Hashable], float],
                       v_inf: Hashable) -> dict[Hashable, float]:
    """``delta_{Delta,lam}(v, v_inf)`` read off the peacock (SPR Prop. 5.16).

    The caller must supply the Penner coordinates of the peacock edges
    ``(v_inf, v)`` **transported from the original decoration** (Ptolemy flips
    applied to ``lam``, not to the degenerate ``Lam(u)``), as in GSC §7.3:
    "read off the distances between the original horocycles (rather than those
    that have been scaled down to points)".  The peacock (flip sequence) itself
    is found with :func:`is_locally_ideal_delaunay` at
    ``u = {v_inf: 0, everything else: +inf}``.
    """
    out = {}
    for (a, b), val in lam_peacock_spokes.items():
        if a == v_inf:
            out[b] = val
        elif b == v_inf:
            out[a] = val
        else:
            raise ValueError("not a spoke of the peacock")
    return out


def bound_residual(u_v: float, delta_v: float) -> float:
    """``delta_v - u_v``: ``<= 0`` is feasible, ``= 0`` is the tangency/bound case.

    By SPR Prop. 3.2, moving the horocycle at ``v`` by ``u_v`` towards the cusp
    reduces its signed distance to the horocycle at ``v_inf`` by exactly
    ``u_v``, so ``u_v = delta_v`` is the tangency condition and
    ``u_v >= delta_v`` (SPR Thm. 7.18 (61)) is the correct sign.  GSC §7.3
    write the opposite sign; see the module docstring.
    """
    return delta_v - u_v


def bound_active(u_v: float, delta_v: float, tol: float = 1e-12) -> bool:
    """True where the minimiser sits on the constraint (KKT (63))."""
    return abs(u_v - delta_v) <= tol * max(1.0, abs(delta_v))


# ---------------------------------------------------------------------------
# Flip transport: SPR Prop. 3.6 / Prop. 5.10 / Prop. 5.16
# ---------------------------------------------------------------------------


def orient_faces(faces: Iterable[Sequence[Hashable]]) -> tuple[tuple[Hashable, Hashable, Hashable], ...]:
    """Orient an unoriented sphere triangulation consistently, or raise.

    Breadth-first over the dual graph: two faces sharing an edge must traverse
    it in opposite directions.  Raises ``ValueError`` if the result is not a
    consistently oriented sphere triangulation.
    """
    fs = [tuple(f) for f in faces]
    edge_to_faces: dict[tuple[Hashable, Hashable], list[int]] = {}
    for fi, f in enumerate(fs):
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            edge_to_faces.setdefault(_ekey(a, b), []).append(fi)
    out: list[tuple[Hashable, Hashable, Hashable] | None] = [None] * len(fs)
    out[0] = fs[0]
    stack = [0]
    while stack:
        fi = stack.pop()
        f = out[fi]
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            for fj in edge_to_faces[_ekey(a, b)]:
                if fj == fi:
                    continue
                g = out[fj]
                if g is None:
                    raw = fs[fj]
                    oriented = None
                    for cand in (raw, (raw[0], raw[2], raw[1])):
                        if any((x, y) == (b, a) for x, y in
                               ((cand[0], cand[1]), (cand[1], cand[2]), (cand[2], cand[0]))):
                            oriented = cand
                            break
                    if oriented is None:
                        raise ValueError("faces do not share an edge")
                    out[fj] = oriented
                    stack.append(fj)
    if any(f is None for f in out):
        raise ValueError("triangulation is not connected")
    directed: dict[tuple[Hashable, Hashable], int] = {}
    for f in out:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            directed[(a, b)] = directed.get((a, b), 0) + 1
    if any(c != 1 for c in directed.values()):
        raise ValueError("cannot orient consistently (not a manifold)")
    verts = {v for f in out for v in f}
    if len(verts) - len(edge_to_faces) + len(out) != 2:
        raise ValueError("not a sphere: V - E + F != 2")
    return tuple(out)  # type: ignore[return-value]


def flip_edge(faces: Sequence[Sequence[Hashable]],
              edge: tuple[Hashable, Hashable]) -> tuple[tuple[Hashable, Hashable, Hashable], ...]:
    """The Ptolemy flip of ``edge``: combinatorial part (SPR Thm. 4.8)."""
    key = _ekey(*edge)
    fs = [tuple(f) for f in faces]
    incident = [fi for fi, f in enumerate(fs)
                if key[0] in f and key[1] in f]
    if len(incident) != 2:
        raise ValueError("edge does not have two incident faces")
    opposites = []
    for fi in incident:
        opposites.extend(v for v in fs[fi] if v not in key)
    if len(set(opposites)) != 2:
        raise ValueError("edge is not flippable")
    a, b = opposites
    new_faces = [f for fi, f in enumerate(fs) if fi not in incident]
    new_faces.append((key[0], a, b))
    new_faces.append((key[1], b, a))
    return orient_faces(new_faces)


class PeacockAmbiguity(RuntimeError):
    """The peacock (hence delta) is not determined by the local flip rule alone.

    Raised by :func:`delta_certified` when the peacock flip sequences produced by
    different deterministic orders do not agree, or do not reach a full peacock
    (``deg(v_inf) = n - 1``).  This is a real obstruction of the naive "flip the
    violators" rule, see the module docstring of :func:`peacock`; do not paper
    over it with one arbitrary order.
    """


def delta_certified(faces: Sequence[Sequence[Hashable]],
                    lam: Mapping[tuple[Hashable, Hashable], float],
                    v_inf: Hashable,
                    orders: Sequence[str] = ("sorted", "reverse", "greedy", "lazy"),
                    tol: float = 1e-12) -> dict:
    """``delta(v, v_inf)`` only when every deterministic flip order agrees.

    Returns the ``peacock`` result of the first successful order if all complete
    runs agree on the spoke coordinates to ``tol``; otherwise raises
    :class:`PeacockAmbiguity` listing what the orders produced.
    """
    results, failures = {}, {}
    for order in orders:
        try:
            results[order] = peacock(faces, lam, v_inf, order=order)
        except (RuntimeError, ValueError) as exc:
            failures[order] = str(exc)
    if not results:
        raise PeacockAmbiguity("no order produced a complete peacock: %r" % (failures,))
    ref_order = next(iter(results))
    ref = results[ref_order]["delta"]
    spread = 0.0
    for order, res in results.items():
        if set(res["delta"]) != set(ref):
            raise PeacockAmbiguity(
                "peacock spoke sets differ between orders: %r" % ({k: sorted(v["delta"])
                                                                   for k, v in results.items()},))
        spread = max(spread, max(abs(res["delta"][v] - ref[v]) for v in ref))
    if len(results) < len(orders) or spread > tol:
        raise PeacockAmbiguity(
            "delta is order dependent: spread %.6g over orders %r (failures: %r). "
            "The modified local Delaunay condition alone does not select the "
            "peacock on this input; the missing ingredient is Springborn Prop. "
            "5.16's minimal-geodesic property, not a tolerance."
            % (spread, {k: {v: round(x, 6) for v, x in sorted(r["delta"].items())}
                        for k, r in results.items()}, failures))
    return ref


def peacock(faces: Sequence[Sequence[Hashable]],
            lam: Mapping[tuple[Hashable, Hashable], float],
            v_inf: Hashable,
            tol: float = 0.0,
            order: str = "sorted",
            max_flips: int = 100000) -> dict:
    """Springborn Prop. 5.16 peacock and the signed horocycle distances ``delta``.

    Algorithm (all of it from the sources, no optimiser):

    * decoration: ``u = 0`` at ``v_inf`` and ``u = +inf`` at every other vertex,
      i.e. every other horocycle is shrunk to a point (SPR Def. 5.9; GSC §7.3);
    * ideal Delaunay condition in that decoration: GSC (18) with SPR (13)
      weights ``exp(-u)`` -- see :func:`local_delaunay_margin`;
    * flip every *strictly* violating edge (margin ``< -tol``).  Such an edge
      ``j1 j2`` has ``v_inf`` as one of its two opposite vertices, so the flip
      replaces it by ``v_inf x``: it strictly increases ``deg(v_inf)``, which
      also proves termination.  Edges with margin exactly ``0`` are
      *nonessential* (SPR Prop. 5.10) and are **not** flipped: they only occupy
      the rim of the wheel, and flipping them would not change any spoke value
      (a rim flip never touches an edge incident with ``v_inf``);
    * transport ``lam`` along the same flip sequence with the Ptolemy relation
      (SPR Prop. 3.6 (9)); this is legitimate because Penner scaling commutes
      with Ptolemy (:func:`ptolemy_flip` docstring), so the *sequence* may be
      found in the degenerate decoration while the *values* are transported in
      the original one -- GSC §7.3 "read off the distances between the original
      horocycles";
    * ``delta(v, v_inf) = lam~_{(v, v_inf)}`` for the spokes (SPR Prop. 5.16).

    Returns a dict with ``faces``, ``lam``, ``spokes`` (= ``delta``),
    ``flips`` (the transported flip history) and ``deg`` of ``v_inf``.
    """
    fs = tuple(tuple(f) for f in faces)
    lam_cur = dict(lam)
    flips: list[tuple[Hashable, Hashable]] = []
    u_degenerate = {v: (0.0 if v == v_inf else math.inf)
                    for v in {x for f in fs for x in f}}

    def edgeset(ff):
        return {_ekey(a, b) for f in ff for a, b in
                ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))}

    def flippable(ff, e) -> bool:
        """``e`` can be flipped iff the other diagonal of its quadrilateral is
        not already an edge (only possible when the four vertices carry all six
        edges, i.e. never for a genuine quad)."""
        fs_here = [f for f in ff if e[0] in f and e[1] in f]
        if len(fs_here) != 2:
            return False
        opp = [v for f in fs_here for v in f if v not in e]
        if len(set(opp)) != 2:
            return False
        return _ekey(opp[0], opp[1]) not in edgeset(ff)

    def violating(ff: Sequence[Sequence[Hashable]],
                  ll: Mapping[tuple[Hashable, Hashable], float]) -> list[tuple[Hashable, Hashable]]:
        disk = DiskTriangulation(v_inf=v_inf, faces=tuple(ff))
        out = []
        for e in disk.edges():
            if v_inf in e:
                continue
            if not flippable(ff, e):
                continue
            if local_delaunay_margin(disk, ll, e, u_degenerate) < -tol:
                out.append(e)
        return out

    while True:
        bad = violating(fs, lam_cur)
        if not bad:
            break
        if order in ("greedy", "lazy"):
            disk_now = DiskTriangulation(v_inf=v_inf, faces=fs)
            scored = [(local_delaunay_margin(disk_now, lam_cur, e, u_degenerate), e)
                      for e in bad]
            scored.sort(reverse=(order == "lazy"))
            e = scored[0][1]
        else:
            bad.sort(reverse=(order == "reverse"))
            e = bad[0]
        # Ptolemy value for the new diagonal, in the *original* decoration
        key = e
        incident = [f for f in fs if key[0] in f and key[1] in f]
        opposites = []
        for f in incident:
            opposites.extend(v for v in f if v not in key)
        a, b = opposites
        # quad in cyclic order: key[0], a, key[1], b ; sides are the four
        # edges of the quad, the old diagonal is `key`
        lam_new = ptolemy_flip(
            lam_cur[_ekey(key[0], a)], lam_cur[_ekey(a, key[1])],
            lam_cur[_ekey(key[1], b)], lam_cur[_ekey(b, key[0])],
            lam_cur[_ekey(key[0], key[1])])
        lam_cur.pop(_ekey(*key))
        new_key = _ekey(a, b)
        if new_key in lam_cur:
            raise ValueError("flip would duplicate an edge")
        lam_cur[new_key] = lam_new
        fs = flip_edge(fs, key)
        flips.append(key)
        if len(flips) > max_flips:
            raise RuntimeError("peacock flip algorithm did not terminate")
    disk = DiskTriangulation(v_inf=v_inf, faces=fs)
    verts = disk.vertices()
    deg_full = sum(1 for e in edgeset(fs) if v_inf in e)
    if deg_full != len(verts) - 1:
        raise RuntimeError(
            "peacock not reached: deg(v_inf)=%d but there are %d other vertices; "
            "every vertex must be adjacent to v_inf (GSC 7.3).  This is a real "
            "failure of the modified flip phase for this input, not a tolerance "
            "artefact." % (deg_full, len(verts) - 1))
    spokes = {e: lam_cur[e] for e in disk.edges() if v_inf in e}
    return {"faces": fs, "lam": lam_cur, "spokes": spokes,
            "delta": delta_from_peacock(spokes, v_inf),
            "flips": flips, "deg": deg_full}


def delta_arc_search(faces: Sequence[Sequence[Hashable]],
                     lam: Mapping[tuple[Hashable, Hashable], float],
                     v_inf: Hashable,
                     depth: int = 4,
                     max_states: int = 400000) -> dict:
    """``delta(v, v_inf)`` as the minimum over arcs, by bounded flip-graph search.

    Why this is the right definition (SPR Def. 5.14 + Prop. 5.16): ``delta`` is
    the *minimum over lifts* of the signed horocycle distance between the
    horocycle at ``v`` and the one at ``v_inf``.  Every ideal arc from ``v`` to
    ``v_inf`` in *any* ideal triangulation gives such a pair of lifts, so

        delta(v, v_inf)  <=  lam~(v, v_inf)      for every reachable arc,

    and Prop. 5.16 says the inequality is an equality for the peacock arc.  So
    the minimum over the searched arcs is an **upper bound** on ``delta`` which
    becomes exact as soon as the search contains the peacock arc.

    This function enumerates the flip graph from ``faces`` up to ``depth``
    flips (states deduplicated), transporting ``lam`` by Ptolemy (SPR Prop. 3.6)
    along every path, and returns per-vertex minima.  ``report``-style extras
    (search size, whether a complete peacock attaining all minima was seen) are
    returned under ``"certified"``: it is ``True`` only if a triangulation with
    ``deg(v_inf) = n - 1`` was found whose spoke coordinates equal all minima --
    that is the certificate Prop. 5.16 asks for.  Otherwise the values are
    firmly-typed *candidates*: never feed them to a solver without saying so.

    Cost: tetra/octahedron are instant; the icosahedron needs ``depth=5``
    (51 401 states, ~10 s) for the minima to stabilise, and no complete peacock
    at ``depth=6`` (348 155 states, ~78 s) attains all of them simultaneously,
    so on that mesh the certificate is *not* obtained and the values stay
    candidates (they are, however, exactly constant on the orbits of the
    stabiliser of ``v_inf``, which every true ``delta`` must be).
    """
    def ek(a, b):
        return (a, b) if a <= b else (b, a)

    def canon(fs):
        return tuple(sorted(tuple(sorted(f)) for f in fs))

    def flips_of(fs):
        ef: dict[tuple[Hashable, Hashable], list] = {}
        for f in fs:
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                ef.setdefault(ek(a, b), []).append(f)
        out = []
        for e, fl in ef.items():
            if len(fl) != 2:
                continue
            opp = [v for f in fl for v in f if v not in e]
            if len(set(opp)) != 2:
                continue
            a, b = opp
            if ek(a, b) in ef:
                continue
            out.append((e, a, b))
        return out

    fs0 = tuple(tuple(f) for f in faces)
    n_vertices = len({v for f in fs0 for v in f})
    best: dict[Hashable, float] = {}
    attained = False
    seen = {canon(fs0)}
    from collections import deque

    # seed with the arcs of the input triangulation itself (a peacock need not
    # require any flip, e.g. the tetrahedron)
    for (x, y), val in lam.items():
        if v_inf in (x, y):
            w = y if x == v_inf else x
            best[w] = min(best.get(w, val), val)
    f0 = fs0
    n0 = sum(1 for e in {ek(a, b) for f in f0 for a, b in
                         ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))} if v_inf in e)
    if n0 == n_vertices - 1 and set(best) >= {v for v in {x for f in f0 for x in f} - {v_inf}}:
        pass  # attainment is decided inside the loop too; see below
    q = deque([(fs0, dict(lam), 0)])
    while q and len(seen) < max_states:
        fs, ll, d = q.popleft()
        for e, a, b in flips_of(fs):
            new = ptolemy_flip(ll[ek(e[0], a)], ll[ek(a, e[1])], ll[ek(e[1], b)],
                               ll[ek(b, e[0])], ll[ek(*e)])
            ll2 = dict(ll)
            ll2.pop(ek(*e))
            ll2[ek(a, b)] = new
            spokes = {}
            for (x, y), val in ll2.items():
                if v_inf in (x, y):
                    w = y if x == v_inf else x
                    spokes[w] = val
                    if w not in best or val < best[w] - 1e-12:
                        best[w] = val
            if len(spokes) == n_vertices - 1 and set(spokes) == set(best):
                if all(abs(spokes[k] - best[k]) < 1e-9 for k in best):
                    attained = True
            if d + 1 < depth:
                nf = [f for f in fs if e[0] not in f or e[1] not in f]
                nf.append((e[0], a, b))
                nf.append((e[1], b, a))
                try:
                    nf = orient_faces(nf)
                except ValueError:
                    continue
                c = canon(nf)
                if c in seen:
                    continue
                seen.add(c)
                q.append((nf, ll2, d + 1))
    if (not attained) and n0 == n_vertices - 1:
        seed_spokes = {((y if x == v_inf else x)): val for (x, y), val in lam.items()
                       if v_inf in (x, y)}
        if set(seed_spokes) == set(best) and all(
                abs(seed_spokes[k] - best[k]) < 1e-9 for k in best):
            attained = True
    return {"delta": dict(best), "certified": attained,
            "states": len(seen), "depth": depth}
