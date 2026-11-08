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

**The measured answer (2026-10-28, corrected): yes, it converges.**  Step error at
the rabbit's known fixed point, with the edge-midpoint refinement:

    refine 0: 1.2277e-01     refine 1: 3.7865e-02 (ratio 3.24)
    refine 2: 1.3812e-02 (ratio 2.74)

Order ~1.5-1.7, so the scheme behaves.  (Rabbit-by-Moebius fails at refine 1 on the
known arc-pairing fragility, not on the scheme.)

**What this replaces, and why the earlier answer was wrong.**  The first version of
this study subdivided by inserting a vertex in each face at `mesh.face_sample` --
whose weights are deliberately skewed, because a symmetric centroid degenerates on
a face with an antipodal pair.  Iterated, the skew compounds, thin triangles pile up
at the corners, and the cone angle at each puncture *diverges*: measured 4.53, 7.07,
9.61, 12.08 rad over three levels.  With that refinement the step error was constant
(ratio 1.00).  It was constant because the meshes were not converging to the round
metric at all, not because of anything about the scheme.

That mistake came bundled with a false claim, now retracted: that cone angles are
invariants of discrete conformal equivalence.  They are not -- a vertex scaling
l~_ij = exp((u_i + u_j)/2) l_ij changes every triangle angle (measured: an angle
moving 18.7 degrees, another 43.6 degrees, for u = (0, 0.9, -0.4)) -- and making the
cone angles all 2*pi is precisely what discrete uniformization does.  The correct
refinement converges them to 2*pi at O(h^2): deficits 5.10e-01, 1.39e-01, 3.56e-02,
8.97e-03.  `tests/test_mesh.py::TestRefinementConverges` is the regression test for
both the false claim and the divergent refinement.

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
    # Bartholdi's IMG table entries [4,15,1]/[4,15,2] (gap/examples.gi): two
    # strictly rational quadratic maps with a critical 4-cycle, exact algebraic
    # coefficients, |P| = 4 and f|_P a permutation
    for name, f, c in th.img_golden_maps():
        out.append((name, f, "strictly rational, critical 4-cycle, exact surd data"))
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
    print("refine = edge-midpoint subdivisions of the BASE mesh, before pullback")
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
