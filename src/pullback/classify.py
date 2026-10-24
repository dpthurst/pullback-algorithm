"""Type classification (plan item 7; [PChar] §1.2, intro.tex).

Definitions (paper): a branched self-cover is of *hyperbolic type* if
each cycle of P contains a branch point; of *non-compact type* if each
component contains a point of P eventually falling into a cycle with a
branch point.  A third state is the one that makes ᴱ = 1 forced: a
cycle of P with no branch point.

How the data gives this:

* The *marked* cover face of a base face C is the cover face whose
  phi-boundary image wraps C (the embedding pattern; face_map_phi).
  Every x in P lies in a unique face of Gamma_1; the marked face of C
  is that face, and f(x_C) is the puncture of pi(that face).
* The local degree of f at x_C is the pi-degree of the marked face, so
  x_C is a branch point iff that degree is at least 2.
"""

from __future__ import annotations

from .portrait import face_map_pi, face_map_phi


def type_classification(base, cover, phi):
    prows, bwalks = face_map_pi(cover, base)
    frows, _ = face_map_phi(phi)
    F0 = len(bwalks)

    marked = {}
    for r in frows:
        if r["target"] is not None:
            marked.setdefault(r["target"], []).append(r)
    for C in range(F0):
        ms = marked.get(C, [])
        if len(ms) != 1:
            raise ValueError(
                f"base face {C} has {len(ms)} marked cover faces; the data "
                "does not single one out (not a ribbon presentation?)")

    # f on P: base face C -> base face of the marked face's pi-image
    dyn = {}
    critical = {}
    for C in range(F0):
        g = ms = marked[C][0]["face"]
        idx = [i for i, f in enumerate(cover.faces()) if f == g][0]
        dyn[C] = prows[idx]["target"]
        critical[C] = prows[idx]["degree"] >= 2

    # cycles of dyn
    seen = set()
    cycles = []
    for C in range(F0):
        if C in seen:
            continue
        path = []
        x = C
        while x not in seen:
            seen.add(x)
            path.append(x)
            x = dyn[x]
        if x in path:
            cycles.append(path[path.index(x):])

    unbranched = [cyc for cyc in cycles if not any(critical[c] for c in cyc)]
    if unbranched:
        kind = "cycle without branch point"
    else:
        kind = "hyperbolic type"
    return {
        "dynamics": dyn,
        "critical": critical,
        "cycles": cycles,
        "unbranched_cycles": unbranched,
        "type": kind,
    }
