#!/usr/bin/env python3
"""The refinement study: does the discrete pullback step converge to sigma?

The question the author asked (2026-10-26): is there a way to check convergence
of discrete uniformisation *followed by forgetting*, under refinement of the
mesh before pullback, before sigma's definition is pinned?

The step is `discrete.pullback_step`: a labelled configuration of P, a base mesh
on it with the chordal (inscribed polyhedron) metric, the congruent lift through
the map, `rounding.round_realise`, and the diagonal copy of P in f^{-1}(P) read
off.  Everything in it is checkable without sigma: at a genuine solution of
sigma(tau) = tau the step has to return tau, and the rabbit's configuration is
known exactly from the map, so the step error at tau_rabbit is the discretisation
error of the scheme, measured.

Refinement is `mesh.subdivide_face_centres` (one vertex inside each face, on the
sphere), applied to the base mesh *before* the pullback.  So the table below is:
for each refinement level, the step error at the fixed point, and the ratio
between successive errors, which is the measured order of convergence.

**The measured answer (2026-10-26): no.**  The step error is constant in the
refinement (ratio 1.00 at every level for both maps), while the round
realisation converges (residual ~1e-12).  The reason is structural, not numerical:

* a vertex scaling multiplies every length at a vertex by the same factor, so
  *cone angles are invariants of discrete conformal equivalence*;
* pulling a flat metric back through a cover multiplies the cone angle at a lift
  by its local degree, so the lifted metric has angle e * theta_a at a critical
  preimage, tending to 4 pi as the mesh refines;
* the chordal metric of the analytic preimages has angle < 2 pi at every vertex.

So the lifted metric and the analytic metric are never discretely conformally
equivalent, at any refinement, and the round realisation of the lifted metric is
a configuration that does not depend on the mesh -- which is exactly what the
table shows.  The step as built converges to the round realisation of the
pulled-back *cone* metric, not to sigma.

What the scheme needs is the prescribed-angle (hyper-ideal) treatment: a cone
angle above 2 pi is a vertex *outside* the sphere, and the round realisation is
then a hyper-ideal polyhedron.  That is Bobenko-Dimitrov-Sechelmann, *Discrete
uniformization of polyhedral surfaces ... and branched covers over the sphere
via hyper-ideal circle patterns* (DCG 57 (2017)) -- Springborn's bibliography
entry [5], already flagged in the plan as the closest paper to this setting.

Usage:  python3 scripts/refinement_study.py [--refine 2] [--maxit 200]
"""

import argparse
import sys
import time

sys.path.insert(0, "src")

import mpmath as mp

mp.mp.dps = 40

from pullback import mesh as msh                        # noqa: E402
from pullback import thurston as th                       # noqa: E402
from pullback.discrete import base_mesh, pcf_data, pullback_step  # noqa: E402
from pullback.session import Session                     # noqa: E402


def targets():
    out = [("rabbit", th.rabbit_map(), "polynomial, hyperbolic: a convergence target"),
           ("rabbit-by-mobius", th.strictly_rational_rabbit()["map"],
            "strictly rational, hyperbolic: a convergence target")]
    return out


def rabbit_spine():
    """The engine's ribbon structure for the rabbit, for the order check."""
    return Session.load("sessions/rabbit.json").base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refine", type=int, default=2,
                    help="deepest refinement level to run")
    ap.add_argument("--maxit", type=int, default=60,
                    help="round_realise iteration cap per step")
    ap.add_argument("--tol", type=float, default=1e-12)
    args = ap.parse_args()

    print("refinement study: the discrete pullback step at a known fixed point")
    print("refine = face-centre subdivisions of the BASE mesh, before pullback")
    print()
    for name, f, note in targets():
        d = pcf_data(f)
        print("%s (%s)" % (name, note))
        print("   |P| = %d, dynamics %s, degrees %s"
              % (len(d["points"]), d["dynamics"], d["degrees"]))
        print("   %-6s %-9s %-9s %-10s %-11s %-9s %s"
              % ("refine", "base V/F", "lift V/F", "residual", "disc. error",
                 "step err", "ratio"))
        if name == "rabbit":
            # the mesh is a ribbon graph: does its topological order agree with
            # the ribbon structure of the engine's spine, at the punctures?
            spine = rabbit_spine()
            base = base_mesh(d["points"], f, spine=spine)
            n = len(msh.consistent_markings(base, spine))
            print("   mesh vs ribbon structure: the topological order at the"
                  " punctures agrees under %d of the injective markings" % n)
        previous = None
        for refine in range(args.refine + 1):
            t0 = time.time()
            try:
                res = pullback_step(d["points"], d["dynamics"], f,
                                    refine=refine, maxit=args.maxit,
                                    tol=args.tol)
            except ValueError as exc:
                print("   %-6d refused: %s" % (refine, str(exc)[:60]))
                break
            except Exception as exc:
                print("   %-6d lift failed: %s" % (refine, str(exc)[:70]))
                break
            err = res["step_error"]
            ratio = (previous / err) if (previous and err > 0) else float("nan")
            print("   %-6d %-9s %-9s %-10.2e %-11.4f %-9.3e %-4s  (%.0fs)"
                  % (refine,
                     "%d/%d" % (res["base_vertices"], res["base_faces"]),
                     "%d/%d" % (res["lifted_vertices"], res["lifted_faces"]),
                     res["residual"], res["discretisation_error"], err,
                     ("%.2f" % ratio) if ratio == ratio else "-",
                     time.time() - t0))
            previous = err
        print("   (step err = |sigma_discrete(tau) - tau| at the known fixed point;")
        print("    ratio = previous error / this error, the measured order)")
        print()


if __name__ == "__main__":
    main()
