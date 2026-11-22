"""Complete fibres of an explicit analytic rational map.

This module is the analytic counterpart of ``preimage_lift``: it computes full
preimages of a finite point set with local multiplicities, without meshes,
CEPS, unknown-cover uniformisation, or local-series domain choices.  It is the
operation needed by preimage-tree algorithms and by Julia-set or bifurcation
locus computations.

The numerical primitive is :meth:`RationalMap.preimage_with_degrees`; this file
only gives it a reusable full-point-set API and a consistency guard.  Each
:class:`Fibre` records the original target and every ``(position, local_degree)``
pair.  Its degree sum must equal ``f.deg`` or :class:`IncompleteFibre` is
raised.  When the interface also supplies combinatorial information, an optional
``degree_profiles`` mapping gives the declared sorted multiplicity profile for
each target; disagreement raises :class:`ProfileMismatch`.  That check
deliberately replaces a plausible partial answer by an error.  It is not an
interval proof and cannot detect every wrong-sheet swap or repeated/near-root
classification error: the existing ``CLUSTER_TOL`` close-root ambiguity remains
a limitation of the underlying primitive.

Positions are chart values (finite complex numbers or ``mpmath.inf``), not 3-D
sphere vectors.  Targets, duplicate targets included, retain input order.  A
numerical root ordering is not interpreted as monodromy sheet labelling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .rational import RationalMap


class IncompleteFibre(RuntimeError):
    """The numerical fibre is inconsistent with the rational map's degree."""


class ProfileMismatch(RuntimeError):
    """Numerical and declared combinatorial ramification profiles differ."""


def _profile(entries):
    """One positive-integer local-degree profile, sorted for comparison."""
    out = []
    for local_degree in entries:
        if not isinstance(local_degree, int) or local_degree < 1:
            raise ValueError("local degrees must be positive integers")
        out.append(local_degree)
    return tuple(sorted(out))


@dataclass(frozen=True)
class Fibre:
    """One complete target fibre, with the target's declared map degree.

    ``entries`` contains ``(chart_position, local_degree)`` pairs in the order
    supplied by the numerical primitive.  The sum is checked on construction,
    not asserted to be an exact multiplicities certificate.
    """

    target: object
    entries: tuple
    degree: int
    declared_profile: tuple | None = None

    def __post_init__(self):
        if self.degree < 1:
            raise ValueError("a rational fibre needs a positive map degree")
        for entry in self.entries:
            if len(entry) != 2:
                raise ValueError("a fibre entry must be (position, local_degree)")
        actual = _profile(local_degree for _position, local_degree in self.entries)
        total = sum(actual)
        if total != self.degree:
            raise IncompleteFibre(
                f"target {self.target!r}: local degrees sum to {total}, "
                f"expected {self.degree}; refusing a partial or misclassified fibre")
        if self.declared_profile is not None:
            expected = _profile(self.declared_profile)
            object.__setattr__(self, "declared_profile", expected)
            if expected != actual:
                raise ProfileMismatch(
                    f"target {self.target!r}: numerical profile {actual} != "
                    f"declared combinatorial profile {expected}")

    @property
    def positions(self):
        """Chart positions, without their local multiplicities."""
        return tuple(position for position, _local_degree in self.entries)

    @property
    def local_degrees(self):
        """Local multiplicities in the same order as ``positions``."""
        return tuple(local_degree for _position, local_degree in self.entries)


def full_preimages(rational_map: RationalMap, targets: Iterable, *,
                   degree_profiles=None):
    """Return one checked :class:`Fibre` per target, preserving input order.

    Finite and infinite targets, multiple preimages at infinity and ramified
    fibres are handled by :class:`RationalMap`.  No CEPS/local-series code is
    imported.  The return is a list, rather than a target-keyed dictionary, so
    distinct occurrences of the same target remain available to a traversal.

    If ``degree_profiles`` is supplied, it must map every target to the local
    multiplicities declared by the caller's combinatorial data.  The sorted
    profile must match, or :class:`ProfileMismatch` replaces the output.  This
    validates aggregate combinatorics only; no numerical root order is
    interpreted as a sheet labelling.
    """
    fibres = []
    for target in targets:
        profile = None
        if degree_profiles is not None:
            if target not in degree_profiles:
                raise ProfileMismatch(
                    f"no declared combinatorial profile for target {target!r}")
            profile = degree_profiles[target]
        fibres.append(Fibre(
            target, tuple(rational_map.preimage_with_degrees(target)),
            int(rational_map.deg), declared_profile=profile))
    return fibres
