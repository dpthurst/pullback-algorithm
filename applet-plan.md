# Applet for graph virtual endomorphisms — plan of attack

Updated 2026-11-08. Item labels (10–410) are stable identifiers, not list
ordinals; cross-references use the bold label. Decisions are in §8.

**At a glance.** The input is a ribbon graph virtual endomorphism plus elastic
lengths. The app derives the critical portrait, the topological automaton and
its self-similar group, elastic energies and iterates, and — where it exists —
a numerical rational map with explicit verification. The engine is Python,
exposes a scripting API, and has a desktop view; a browser page remains an
optional later port.

**Milestone status.**

- M0/M1 done: the library, validation, portraits and fixtures are in place.
- M2 core done: normal form, automaton, level actions, nucleus and Schreier
  graphs agree with Bartholdi's FR. The IMG as a group and word growth remain
  deferred; contraction is reported only for the group actually computed.
- M3 p = 2 core implemented and certified on the fixtures. Obstruction
  matrices are not built. The p ≠ 2 layer is paused for a rewrite.
- M4 has a working discrete-conformal step for known analytic maps and an
  exact VE-native initial portrait, but no accepted arbitrary-start
  convergence or rational reconstruction. The current blocker is a sound,
  embedded, orientation-controlled lifted cover; see 330.
- M5 has a scripting facade and desktop panels, but not final packaging.

A candidate in the convergence gate is not a convergence, fixed-point or
unobstructedness certificate.

## Sources

[PChar] D. Thurston, *A positive characterization of rational maps*,
Ann. of Math. **192** (2020) 1–46, arXiv:1612.04424.
[Elast] D. Thurston, *Elastic Graphs*, Forum Math. Sigma **7** (2019) e24,
arXiv:1607.00340.
[Rubber] D. Thurston, *From rubber bands to rational maps*,
Res. Math. Sci. **3**:15 (2016), arXiv:1502.02561.
[KPT] Kahn–Pilgrim–Thurston, *Conformal surface embeddings and extremal
length*, arXiv:1507.05294.
[ConfDim] Pilgrim–Thurston, *Ahlfors-regular conformal dimension and energies
of graph maps*, Groups Geom. Dyn. **20** (2026) 971–1050, arXiv:2112.09041.
[Springborn] B. Springborn, *Discrete conformal maps and ideal hyperbolic
polyhedra*, Geom. Topol. **19** (2015), arXiv:1707.06848 (M4 uniformisation).
[Harmonic] the author's literate-Haskell implementation and its example data.
Other primary sources are cited where used. Local source copies and reading
notes are outside the tracked tree.

## 1. What the applet takes in and puts out

**Input** = one graph virtual endomorphism plus an elastic structure:

- **10.** a ribbon graph Γ₀, given by a rotation system. Drawn in the plane it
  is a spine of S²∖P, one puncture per face.
- **20.** a covering π: Γ₁ → Γ₀ of degree d. Γ₁ inherits its ribbon structure
  by pullback and |V₁| = d|V₀|, |E₁| = d|E₀|.
- **30.** a graph map φ: Γ₁ → Γ₀ up to homotopy, given by vertex images and one
  edge path in Γ₀ for each edge of Γ₁.
- **40.** elastic lengths α(e) ∈ ℚ_{>0} on Γ₀, pulled back through the tower.
  Results are invariant under a common scaling of α; the app normalises it and
  checks that invariance.

**Output families**

- **1.** **Branched self-cover.** Validation, ribbon-map check, critical
  portrait, local degrees, dynamical type.
- **2.** **Algebraic objects.** Wreath recursion/biset, self-similar group,
  level actions, nucleus, contraction test, Schreier graphs. Word growth and
  the IMG as a group are deferred; see 150.
- **3.** **Energies.** Emb[φ], E^p_p[φ], iterates φⁿ, asymptotic energies, and
  obstruction matrices when built.
- **4.** **Rational map, when it exists.** Numerical reconstruction,
  independent verification, Mandelbrot/Julia and limit-space pictures.
  Reconstruction is not a proof; 240 specifies the verification boundary.

## 2. Data model and input

- **50.** **Γ₀ as a rotation system.** A planar straight-line drawing supplies
  cyclic orders; faces are boundary walks of the thickening. Every face is a
  puncture. A global mirror reverses all cyclic orders and complex-conjugates
  the branched cover; both orientations are accepted, but each session records
  which one it uses so comparisons are never silently made across a mirror.
- **60.** **The cover by monodromy.** Pick a spanning tree T₀. The remaining
  g = |E₀|−|V₀|+1 edges generate π₁(Γ₀)=F_g; the user supplies a permutation
  in S_d for each generator, and the app builds Γ₁, its pulled-back ribbon
  structure and edge labels.
- **70.** **φ by edge paths.** The user gives vertex images; shortest paths are
  an editable default. Only the homotopy class matters. Collapse T₀ and
  π⁻¹(T₀), fix the vertex, and check invariance under inserted detours.
- **80.** **Sessions as JSON.** One file holds 10–40, shared by engine, scripts
  and views. The text session is the source of truth (D3).
- **85.** **Reverse input route, eventual.** From a PCF rational map (or one
  with attracting cycles) construct the graph data. Prior art: IMG's
  `SphereMachineOfBranchedCovering`; Hubbard trees/spiders for polynomials.
  Not committed to M0–M5 unless it falls out of M4.
- **90.** **Example library first, because it is the test suite.** [PChar]
  Figures 3 and 4; the barycentric and census #3.2 carpets; z²+i; [Elast]
  Example 1.17; and machine-readable [Harmonic] data for theta, bouquet,
  barbell, slit/blown-up Lattès, rabbit/twisted rabbit, Grigorchuk,
  tetrahedral, barycentric and census #53. The format is [Harmonic] `GraphIO`:
  flowers give the ribbon structure, unlisted edges map to constant paths.
  [ConfDim] adds the barycentric carpet fixture E²₂[φ²]=10/13 and the
  fat/skinny Devaney families; the latter are drawings, not yet encoded data.

## 3. Algorithms

### A. Validation and critical portrait

- **100.** **π₀/π₁ checks.** Connectedness makes π₀ automatic; π₁-surjectivity
  is checked on generators via the half-edge structure.
- **110.** **Ribbon map.** The lift Nφ exists exactly when the local orders on
  φ⁻¹(y) are consistent ([PChar] Def. 2.7, Rem. 2.10). Use a clear local-order
  algorithm first; test on the 1/5-versus-2/5 rabbits.
- **120.** **Critical portrait.** For each face of Γ₁ record its target face in
  Γ₀, local degree = ratio of boundary-cycle lengths, and inactive/collapsed
  faces. Validate the Riemann–Hurwitz identity and preimage count.

### B. Automaton, biset and self-similar group

- **130.** **Normal form and transport.** Collapse a spanning tree to a rose
  R₀, collapse π⁻¹(T₀) to R₁, fix the vertex and label edges by the generator
  and its φ-image. Every φ can be homotoped to kill the lifted tree, so R₁→R₀
  always exists; there is no inadmissible map. Section words are group
  elements in left-factor-outermost order. Tree changes give presentations of
  the same virtual endomorphism, not different groups; a tree-independent
  geometric level-action oracle is the check. The current implementation uses
  based homotopy sections and `lift.pullback_ve`, including non-rose bases.
- **140.** **Biset.** Sheets X={1,…,d} with cover monodromy on the right and
  sections on the left; the wreath recursion
  g = σ_g·(g|_x)_{x∈X} is the same object. Author decision: carry the
  **collapsed single-basepoint group biset**, not the groupoid biset. The
  price is effectiveness: π₁→IMG need not be injective (twisted-z2i is an
  example). IMG's external Thurston and spider algorithms consume precisely
  this collapsed machine.
- **150.** **Group operations and levels.** Products, inverses, sections,
  level-n action, nucleus, Schreier graphs and the FR cross-check are
  implemented. `nucleus(level=n)` is the IMG up to the level-n kernel;
  `nucleus()` with word deduplication is the π₁-nucleus and a semi-decision.
  Always name the group whose contraction is reported.
  The IMG as a group, its word problem and word growth are deferred: only
  level images are required for the main applet. On the graph π₁, word growth
  is the free-group formula; the interesting growth is the IMG's and is not
  computed.

### C. Energies

- **160.** **Emb(ψ) for an explicit PL map.** The integrand
  y ↦ Σ_{x∈ψ⁻¹(y)}|ψ′(x)| is piecewise constant away from images of vertices
  of Γ₁, so its essential supremum is a finite maximum. Exact rational
  arithmetic whenever the data are rational.
- **170.** **Emb[φ], E^p_p[φ] and the homotopy minimiser.** By [Elast] Thm 1,
  Emb[φ] equals the supremum of EL(φ∘c)/EL(c) over weighted multicurves and
  is attained by one. The implementation performs [Elast] §6: harmonic
  representative → tension weights → pushforward → length iteration, with the
  boundary/escape procedure where needed. `iter` need not converge for every
  input; non-maximal fixed points require the escape procedure.
  An independent **upper bound** always exists: if δ(f)=l₀(φf)/α₁(f), set
  M[l][l′]=Σ δ(f)^{p−1} over traversals of a lift of base edge l above edge
  l′. Then Fill^p(φⁿ₀)=Mⁿ·1 and
  limsup (E^p_p[φⁿ])^{1/n} ≤ (max_l(Mⁿ1)_l)^{1/(pn)} → ρ(M)^{1/p}. At
  p=2 this bounds the asymptotic embedding energy.
  Status: the p=2 exact engine and the convex cover minimiser are implemented
  and reproduce the hand/computed fixtures (including [Elast] Ex. 1.17,
  bouquet 1, 3/4, 5/8, 23/48, 187/480, and theta at the published n). The
  exact path is exponential; the convex cover path is the scalable route.
  The escape procedure of Algorithm 6.49 is not implemented because no test
  has needed it. The p ≠ 2 layer is paused for a rewrite; its equations are
  linear only at p=2.
- **180.** **Two-sided bounds.** Upper: Emb(ψ) for an explicit ψ∈[φ], e.g. a
  shortened/harmonic representative. Lower: EL(φ∘c)/EL(c) for a weighted
  multicurve. Display the gap so the user sees what is certified.
- **190.** **Iterates and asymptotics.** Build X_n recursively ([PChar] Def.
  5.1), compute Emb[φⁿ] and E²₂[φⁿ]=√Emb, and report
  (E^p_p[φⁿ])^{1/n}, which decreases to ᴱE^p_p. Emb[φⁿ]<1 for one n is a
  positive rationality certificate. ᴱ is independent of elastic weights and
  spine; the finite terms and the smallest sufficient n are not, so expose α
  and spine variation. Carpet sequences need not decrease initially; an
  undecided answer is legitimate. E^p_∞ of an iterate and ᴱE^∞ are not
  implemented. The tropical max-cycle-mean formulation is the intended route.
- **200.** **Obstruction matrices and twisted-z2i.** For a multicurve C,
  M_{C,p}[i,j] sums deg(D→C_i)^{1−p} over components D of π*C_i whose φ-image
  is C_j. Perron–Frobenius λ≥1 gives a p-obstruction and hence ᴱE^p_p≥1. At
  p=2 the search reduces to simple multicurves. The implementation is not
  built. Twisted-z2i has the z²+i portrait and orbifold signature
  (2,2,2,∞), χ_orb=−1/2; obstruction remains deliberately open. On the
  4-punctured sphere only the three 2+2 multicurves need checking.
- **210.** **Plots, not an interactive page.** Produce deterministic scripts
  and static figures for Emb[φⁿ]^{1/n} with brackets and for ᴱE^p_p versus p.
  M5 may port them to the view.

### D. The rational map

- **220.** **Decision principle.** By [PChar] Thm 1, rationality is equivalent
  to ᴱE²₂<1. The certificates of 190 and 200 give finite positive/negative
  tests; otherwise report “undecided”.
- **230.** **Reconstruction.** The decided engine is Thurston pullback: keep a
  labelled configuration of P, build the rational map with the prescribed
  branch behaviour, pull back the complex structure through the cover, and
  read the new preimages. Use arbitrary precision throughout; per-step root
  finding is univariate, not a coupled portrait solve. The combinatorial
  shadow is the [Elast] iteration of 170. For clustering points, re-coordinate
  by Möbius blow-ups rather than losing precision; tree-of-spheres and
  rescaling-limit literature supplies the framework. IMG's
  `ThurstonAlgorithm`/`ThurstonObstruction` are the independent prior art.
- **240.** **Verification of a reconstructed map.** Recompute its portrait and
  compare with 120; check the marked orbit; reconstruct a VE and test homotopy
  equivalence with the input; compare energies and the Julia/limit-space
  pictures. Label the output “numerical, verified at this precision”, never
  “proved”.
- **245.** **Exact PCF closure, deferred beyond the numerical gate.** First
  Newton-polish the numerical map and certify critical-orbit closure exactly
  (algebraic or interval/Krawczyk). Then construct a spine and VE from the PCF
  map and verify ribbon homotopy equivalence and labelled dynamics with the
  input. The map remains numerical until both stages pass.

## 4. Architecture

- **250.** **Python engine and scripting surface.** Exact `Fraction`
  arithmetic, symbolic linear algebra, flows/LP and NumPy iterates. The engine
  is authoritative and independent of the view. D1: Python first; the
  scripting API is a deliverable in its own right.
- **260.** **Views.** The desktop app is the first view; model/view are
  separate and every displayed number comes from an engine method. A
  single-page browser app is the optional later port, not a second engine.
  Numerical-map views use CPU rendering (no reliable OpenGL context in the
  headless environment), preserve aspect ratio, and default to the conformal
  stereographic chart.
- **270.** **Headless verification.** Py-compile and unit tests everywhere;
  Qt under the offscreen platform for the real desktop widgets; jsdom/node for
  a later web view. No headless check can replace the author opening the
  actual window/page.
- **280.** **Build and documentation.** Make targets for the engine, app and
  app checks. The applet owns its build; it is not embedded in the
  core-entropy project.

**Code map.** Inputs and portraits: `ribbons.py`, `covers.py`, `maps.py`,
`portrait.py`, `validate.py`, `classify.py`; algebra: `normalform.py`,
`automaton.py`; energies: `refined.py`, `minimise.py`, `fastmin.py`,
`covermin.py`, `certificates.py`, `bounds.py`, `iterate_graphs.py`,
`iterates.py`, `energies.py`; M4: `discrete.py`, `mesh.py`, `rounding.py`,
`rounding_multi.py`, `lift.py`, `lift_round.py`, `embedding.py`,
`homotopy.py`, `iteration.py`, `initial.py`, `ve_initial.py`, `ve_state.py`,
`rational.py`, `thurston.py`; interface: `session.py`, `app/`; orbifold and
gate: `orbifold.py`, `convergence.py`. Library source of truth:
`scripts/build_library.py`. Cross-checks: `scripts/img_crosscheck.py`,
`scripts/db_iteration_check.py`, `scripts/refinement_study.py`; `make check`
is the gate.

## 5. Milestones

D6 order: algebra before energies.

- **290.** **M0 — paper example.** (1+z²)/(1−z²) through data, portrait and
  automaton, matching [PChar] Fig. 3/3d. Status: done.
- **300.** **M1 — data, validation, library.** Items 50–120 and 90, with
  predicted outputs recorded as fixtures. Status: done.
- **310.** **M2 — algebra.** Items 130–150: normal form, collapsed biset,
  group machinery, Schreier graphs and Fig. 3d tests. Status: core done and
  FR-cross-checked. Deferred: IMG as a group and word growth. Contraction is
  currently reported on graph π₁; π₁^orb and IMG contraction remain future
  questions.
- **320.** **M3 — energies.** Items 160–210. The type classification
  (“hyperbolic type”, “cycle without branch point”, non-compact; [PChar] §1.2)
  belongs here with the energy theorem, not with data validity. The p=2
  implementation and exact/cover minimisers exist; obstruction matrices and
  ᴱE^∞ do not. p ≠ 2 is paused. An unbranched cycle does not make an input
  invalid and does not by itself settle contraction or obstruction.
- **330.** **M4 — rational map, numerical.** Items 220–245. Use Springborn's
  discrete-conformal uniformisation with IMG as an independent cross-check.
  Current state and gates:

  * **VE-native initial geometry exists.** `initial.InitialState` derives the
    puncture projection, diagonal marking, dynamics, local degrees and frozen
    edge-labelled lifted dual from (π,φ) alone. `ve_state.VEInit` constructs an
    arbitrary geometric realisation, a homotopy comparison and the transported
    VE. No analytic map is read.
  * **Frozen iteration exists.** For a known analytic map, `FrozenStep` fixes
    the lifted cover once and moves labelled positions and lengths; it
    reproduces the rabbit fixtures and converges to the known marking at the
    tested refinements. This is not yet an arbitrary-VE constructor.
  * **Necessary geometric gates.** A step must have a sound branched cover and
    satisfy: rank/excess E−rank A−redundant=2V−6; measured residual; injective
    vertices and consistently oriented nondegenerate faces; a fixed mirror
    branch; and labelled Möbius covariance. Residual or zero excess alone is
    not acceptance.
  * **Mirror orientation.** Chordal lengths are invariant under O(3), so a
    reflected configuration solves the same length equations and conjugates
    the labelled moduli coordinate. Orientation must be carried from the
    initial embedding/ribbon structure and through continuation; it cannot be
    inferred from the lengths alone.
  * **Measured enrichment boundary.** Splitting the rabbit's ramified–ramified
    base edge removes its one parallel lifted pair and gives V=8, E=18,
    excess=0, but the realisation still folds (6 of 12 faces oppositely
    oriented). The fan/corner-slot complex has parallel classes for the same
    structural reason. A genuine embedded branched lift remains to be found;
    the deferred Ptolemy/Delaunay experiment (335) is one candidate route.
  * **Mesh decisions.** Keep uniform refinement and the simple mesh for now;
    no BDS implementation and no adaptive meshing. Correct edge-midpoint
    refinement makes puncture angles converge at O(h²) and the rabbit step
    error falls 1.2277e-1, 3.7865e-2, 1.3812e-2. The earlier claim that
    discrete conformal equivalence fixes cone angles was false and is
    retracted; the pushed commit message remains as an error of record.
  * **Test-data limit.** |P|=3 sessions have no Teichmüller modulus and cannot
    test iteration; the |P|=4 sessions are the current test bed. A |P|≥5
    session is still needed. The Devaney family supplies such maps but its
    spines are drawings, not yet encoded sessions. The author will hand-encode
    the first one when wanted.
  * **No arbitrary-start convergence claim yet.** The coarse lifted dual can
    underdetermine the conformal class; the enriched rabbit cover is canonical
    in rank but folded. Consequently the current step is not accepted as a
    Thurston iterate and no fixed point or reconstruction is claimed. The
    orientation-controlled embedded-cover gate is the immediate blocker.
- **335.** **Deferred geometric experiment: Ptolemy/Delaunay
  retriangulation.** A Ptolemy flip preserves the induced flat/Penner
  structure while changing the triangulation, so variable Delaunay
  retriangulation may reach embedded realisations that a fixed complex cannot.
  This is an experiment, not a default: enumerate candidates, check the
  invariant, Delaunay condition, rank/excess, embedding and labelled
  coordinate, and compare with flips disabled. Promote it only if it repairs a
  concrete failed realisation.
- **336.** **Computed orbifold and convergence gate.** `orbifold.py` derives
  Thurston's exact ν from the VE, with ν=∞ exactly on critical cycles, and
  reports signature and χ_orb as exact rationals. Milnor's four-point Lattès
  **portrait** test is separate from flexibility: flexible requires n=2 and
  integer affine derivative, hence square degree; otherwise flexibility is
  undetermined until the torus-homology matrix is computed. This is an
  implementation limit, not a claim that the full VE cannot determine it.
  `convergence_guard` requires a hyperbolic orbifold; parabolic/spherical
  portraits and undetermined markings are refused. **An unbranched cycle is
  not an exclusion** (author's correction): pullback still makes sense, but
  obstruction and contraction require separate analysis. All 13 readable
  library sessions have hyperbolic orbifolds; `julia-4422` and `twisted-z2i`
  remain candidates with unbranched-cycle warnings. `slit-lattes` and
  `blown-up-lattes` are not Lattès maps (signature (3,6,∞), χ_orb=−1/2), and
  historical class labels never override the computation. Candidates are not
  convergence or unobstructedness certificates.
- **340.** **M5 — packaging.** Items 260 and 280. Publish only on the author's
  word; nothing becomes web-visible without asking.

## 6. Verification, and what is not verified

- **350.** **Fixtures.** [Elast] Ex. 1.17 (Emb=3/5, minimiser not harmonic),
  [Harmonic] bouquet 1, 3/4, 5/8, 23/48, 187/480 and theta-family limit
  2^{−1/3}, [ConfDim] barycentric E²₂[φ²]=10/13, FR level orders, and the
  portrait/ribbon checks.
- **360.** **Two-sided energies.** Exact rational arithmetic where possible;
  certified brackets from 180; sampling/eigenvector lower bounds as an
  independent route. D5: no interval arithmetic initially; bracketed numerics
  are enough.
- **370.** **Cross-implementation.** Python versus any JS port over the whole
  library, reported as “k differences over m inputs”. Do not compare a second
  implementation that merely transcribes the first.
- **380.** **Not verified.** Real interactive GUI use without the author at the
  machine; a numerical rational map beyond the criteria of 240; unbounded
  searches beyond the complexity actually run; and any convergence claim
  based only on a small residual.

## 7. Out of scope, or explicitly not claimed

- **390.** Non-compact type and orbigraphs except where explicitly integrated;
  infinite graphs; disconnected branched surfaces; transcendental maps.
- **400.** General rational-map reconstruction as proof. Only numerical
  reconstruction with independent verification is promised.
- **410.** Closed symbolic forms for asymptotic energies. ᴱ is a limit with
  certificates on either side, not a symbolic evaluation.

**Deferred input front ends** (they emit sessions; they do not change the
engine):

- **Schwarz–Christoffel importer.** From (f,P,spine) to session. Analytic
  portraits exist; still needed are model-specific spines/retractions, curve
  tracking through Γ₁=f⁻¹(Γ₀), and emission. First target julia-4442.
- **Twisting.** From a curve word and exponent to the induced π₁ automorphism,
  then transform and renormalise the VE. Acceptance: portrait, type and
  energies invariant under twisting.

## 8. Decisions

All D1–D6 were settled 2026-10-23; amendments below preserve their labels.

- **D1.** Python for the initial implementation, with strict model/view
  separation. Reconfirmed 2026-10-26: the first deliverable is the standalone
  desktop app, and scripting is a first-class interface; the browser page is
  an optional later port.
- **D2.** A separate repository, `dpthurst/pullback-algorithm`. Deploy Pages
  only on the author's word.
- **D3.** The text session is the source of truth; graphical views render
  from it. Streaming/reverse input is eventual (80, 85).
- **D4.** Example data come from [Harmonic] in machine-readable form (90).
- **D5.** Exact certificates plus bracketed numerics; no interval arithmetic
  initially. Reconfirmed 2026-10-25. Revisit interval methods for M4 if
  needed.
- **D6.** Algebra (M2) before energies (M3).

The reconstruction engine in 230 is Thurston pullback, not a direct Newton
solve of critical-portrait equations.
