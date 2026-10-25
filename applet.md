# The applet

A standalone Python desktop app over the engine, and a scripting facade for
the same engine. Both use `pullback.Session`; neither reimplements anything.

## Run it

    make applet                       # the session library
    make applet SESSION=sessions/theta.json

or, without make:

    PYTHONPATH=src python3 -m pullback.app [sessions/theta.json]

Five panels, over the fifteen sessions in `sessions/` or any session file
you open:

* **Summary** — the data (spine, cover, φ), the plan-100 checks, and the
  type classification. A refusal is displayed here, not raised: e.g.
  `grigorchuk` has no ribbon presentation, so the type is refused and the
  algebra still computes.
* **Portrait** — the critical portrait: each cover face's boundary, the
  base face it projects to, and the degree. The degree multiset is the one
  `tests/library_fixtures.json` pins.
* **Automaton** — the rose normal form: letters, sheets, the spanning tree
  it was built on (and whether that tree is *admissible*, i.e. whether the
  section-word shortcut is exact — it need not be, and then the panel says
  so), the π₁-nucleus, the level-n class counts, and the level-n group
  orders.
* **Map** — the session's rational map (its `map` field, e.g.
  `(1+z^2)/(1-z^2)`), drawn on the Riemann sphere: the basin depth of its
  attracting cycles, on a sphere grid, rasterised on the CPU — drag to
  rotate, wheel to zoom. Two projections: **ball** (orthographic — what a
  sphere looks like; circles near the limb are ellipses, which is
  perspective, not a defect) and **conformal** (stereographic — angle
  preserving, so circles stay circles, and the sphere is drawn minus the
  point at the far pole, filling the panel). The screen uses one scale for
  both axes, so the ball is round in any window shape. Sessions whose `map`
  is not a number (`Douady rabbit z^2+c`) say so, and so do sessions with no
  map.
* **Energies** — the lifting bound of plan item 180 for a chosen $p$ and
  $n$. Upper bounds, exact at integral $p$, on the session's own metric;
  a value of 1 means the representative is unit-speed, not that the map is
  certified. Computed synchronously on **Compute** — at $n \ge 3$ this can
  take tens of seconds and the window will wait.

## Script it

    from pullback import Session, library

    theta = Session.load("sessions/theta.json")
    theta.validation()                 # the plan-100 checks
    theta.portrait()["degrees"]        # [2, 1, 1, 2]
    theta.type_classification()["type"]          # 'hyperbolic type'
    theta.automaton()["tree"]                    # ['q'], admissible=True
    theta.nucleus()["classes"]                   # 5, closed
    theta.level_order(6)                         # 16384 (FR cross-check)
    theta.energy_bounds(p=2, n=1)                # [{'n': 1, 'E': 1.0, ...}]

    for name, s in library().items():
        print(name, s.degree, s.type_classification()["type"])

`Session` caches each result after the first call. A malformed session
raises at `load`; data that parses but cannot be read by a particular step
raises there (the app catches it and shows the message).

## Verify it

    make check-applet     # the headless check: 15 tests

Qt runs with `QT_QPA_PLATFORM=offscreen`, so the real window is built, the
θ session opened, and every panel compared with published or FR-verified
values (`tests/test_app_headless.py`); the view models are tested without
any toolkit (`pullback/app/panels.py` has no Qt import). This machine has no
display: the headless check verifies the wiring and the numbers, **not** that
the layout looks right. The first visual check is you opening it.

## Not in this version

No session editor (the data are the JSON files; plan items 80–85 remain the
input front ends), no rational-map reconstruction (M4), and no 2D plane plot
— the Map panel is the sphere, and the same kernel draws the plane. The
picture is unverified visually: this machine has no display, so only its
geometry and its depth statistics are tested — the aspect ratio (a ball in
any window shape), the conformality of the stereographic chart (measured
Jacobian: |du|/|dv| = 1.000000, cos = 7e-10) and the *non*-conformality of
the orthographic one (1.209, cos = 0.19) are all asserted.
