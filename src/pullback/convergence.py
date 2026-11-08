"""Computed dynamical gates for Thurston iteration.

The gate reads the Thurston orbifold and the cycles of the postcritical
portrait from the ribbon VE itself.  A hyperbolic orbifold makes a session a
*candidate*.  Parabolic (including Lattès-portrait) and spherical orbifolds
are excluded; an undetermined portrait is not a permission.  An unbranched
cycle does not remove a candidate: pullback remains meaningful there, while
obstruction and convergence require separate analysis.

A candidate is not a contraction, fixed-point, unobstructedness or
convergence certificate.
In particular, a four-point Lattès portrait is not confused with the flexible
subclass: flexibility remains undetermined until the torus-homology matrix is
computed, except where the square-degree necessary condition rules it out.
The historical ``dynamical_class`` session label never overrides these
computed checks.
"""

from __future__ import annotations

from .orbifold import orbifold_guard

__all__ = ["ConvergenceError", "convergence_guard", "require_convergence_allowed"]


class ConvergenceError(ValueError):
    """The iteration is excluded by its computed dynamical class."""


def convergence_guard(session):
    """Return the computed orbifold, dynamical status and permission.

    The detailed status (hyperbolic/parabolic/spherical/undetermined) is kept
    as ``orbifold_status``.  The public ``status`` retains the facade's
    candidate/unbranched-cycle vocabulary.  ``candidate`` is exactly the
    computed hyperbolic-orbifold permission; ``type`` and the reason retain
    any unbranched-cycle warning.  No metadata shortcut is consulted.
    """
    analysis = orbifold_guard(session)
    out = dict(analysis)
    out["orbifold_status"] = analysis["status"]
    out["candidate"] = bool(analysis["allowed"])
    if not analysis.get("unbranched_cycles"):
        out["type"] = "hyperbolic type"
    else:
        out["type"] = "cycle without branch point"
    if analysis["allowed"]:
        out["status"] = "candidate"
        out["reason"] += ": convergence is conditional, not certified by this gate"
    elif analysis.get("unbranched_cycles"):
        out["status"] = "unbranched cycle"
    else:
        out["status"] = analysis["status"]
    return out


def require_convergence_allowed(session):
    """Raise `ConvergenceError` when the computed gate excludes the dynamics."""
    report = convergence_guard(session)
    if not report["allowed"]:
        raise ConvergenceError(report["reason"])
    return report
