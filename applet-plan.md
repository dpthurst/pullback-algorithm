# Applet for graph virtual endomorphisms — plan of attack

Draft for review, 2026-10-23; all decisions D1–D6 settled the same day (§8).
Nothing is built yet; this file is the thing to argue with before any code is
written. Items carry stable semantic labels (multiples of 10, non-sequential),
rendered as bold bullets so that source and rendered numbers agree; the labels
— not list ordinals — are the cross-references throughout.

Sources used, all in `.scratch/`:
[PChar] D. Thurston, *A positive characterization of rational maps*,
Ann. of Math. **192** (2020) 1–46 (arXiv:1612.04424; LaTeX source + MetaPost
figure code extracted to `.scratch/poschar/`);
[Elast] D. Thurston, *Elastic Graphs*, Forum of Math. Sigma **7** (2019) e24
(arXiv:1607.00340; source in `.scratch/elastic/`);
[Rubber] *From rubber bands to rational maps: a research report*,
Res. Math. Sci. **3**:15 (2016) (arXiv:1502.02561; `.scratch/rubber/`);
[KPT] Kahn–Pilgrim–Thurston, *Conformal surface embeddings and extremal
length* (arXiv:1507.05294);
[ConfDim] K. Pilgrim & D. Thurston, *Ahlfors-regular conformal dimension and
energies of graph maps*, Groups Geom. Dyn. **20** (2026) 971–1050
(arXiv:2112.09041; source in `.scratch/confdim-src/`);
[Harmonic] the author's old literate-Haskell code,
github.com/dpthurst/Harmonic (cloned to `.scratch/Harmonic/`): machine-readable
example data (`Examples.lhs`), computed stretch-factor tables (`output.tex`),
and a prior implementation of harmonic maps and stretch-factor minimisation.
Nekrashevych, *Self-similar groups* (Math. Surveys Monogr. 117) and
*Combinatorial models of expanding dynamical systems* for the algebraic side;
Bartholdi's GAP packages FR and IMG are prior art for the group algorithms and
for Thurston's algorithm (150, 230).

## 1. What the applet takes in, and what it puts out

**Input** = one graph virtual endomorphism plus an elastic structure:

- **10.** a ribbon graph Γ₀ (the cyclic order of half-edges at each vertex); drawn in
    the plane, it is a spine of S²∖P with one puncture per face;
- **20.** a covering π: Γ₁ → Γ₀ of degree d — a covering of *graphs*, so
    |V₁| = d|V₀|, |E₁| = d|E₀|, and Γ₁ inherits its ribbon structure by
    pull-back;
- **30.** a map φ: Γ₁ → Γ₀ up to homotopy, given by vertex images and one edge path
    in Γ₀ for each edge of Γ₁;
- **40.** elastic lengths α(e) ∈ ℚ_{>0} on the edges of Γ₀ (pulled back to Γ₁ and,
    below, to the whole iterate tower; every quantity computed is invariant
    under scaling α, so the app normalises it and checks that invariance).

**Output**, in the four families the request names:

- **1.** **The branched self-cover.** Validation (π₀-bijective, π₁-surjective, φ a
   ribbon map in the sense of [PChar, Def. 2.7]), and the critical portrait the
   data determines: each face of Γ₁ maps to a face of Γ₀ and the local degree is
   the ratio of boundary-cycle lengths — exactly the read-off in [PChar, §2] that
   recovers (2.6) from Figure 3a. The app also reports the type: hyperbolic,
   non-compact, or a cycle in P with no branch point (for z²+i the boundary
   loops force ᴱ = 1, [PChar, §1.2]).
- **2.** **Algebraic objects.** The dual Moore diagram (the three steps of
   [PChar, §2]); the **biset** B of the topological automaton — the degree-d
   G–G-bimodule over G = π₁(Γ₀), displayed both as a bimodule table and as the
   wreath recursion of [PChar, Fig. 3d]; the associated self-similar group:
   products, inverses, sections, level-n action on Xⁿ, nucleus, the
   contracting/finite-state test, word growth, the iterated monodromy group,
   and the level-n Schreier graphs.
- **3.** **Energies.** Emb(ψ) for an explicit PL representative; the homotopy
   invariant Emb[φ] (= SF_EL[φ] by [Elast, Thm 1]); E^p_p[φ] for
   p ∈ {1, 2, ∞} ([PChar, §6]: Emb = (E²₂)²); the iterate tower X_n
   ([PChar, Def. 5.1]) with Emb[φⁿ] and E²₂[φⁿ]; the asymptotic energies
   ᴱE^p_p[π,φ] = lim (E^p_p[φⁿ])^{1/n}; and the p-obstruction matrices M_{C,p}
   of [PChar, §7.4] with their Perron–Frobenius eigenvalues.
- **4.** **The rational map, when it exists.** Decision (see 3.D), numerical
   reconstruction, verification, and pictures: the Julia set by inverse
   iteration beside the inverse limit of the tower X_n, which Nekrashevych's
   theorem identifies with the Julia set in the expanding case
   ([PChar, §1.2]).

## 2. Data model and input

- **50.** **Γ₀** as a half-edge (rotation-system) structure. Drawing it in the
    browser gives the cyclic orders directly (angular order at each vertex of a
    planar straight-line drawing), and the faces are computed by following the
    boundary of the thickening. Each face gets a puncture marker; the app
    draws them exactly as [PChar, Fig. 3a] does.
    **Orientation and conjugation:** reversing every cyclic order (a global
    mirror) gives another valid ribbon structure, and its branched self-cover
    is the complex conjugate of the original's. Both orientations are correct
    and equivalent for the applet's purposes; a global flip is a common
    user-input slip, so the engine accepts either, can tell a session from its
    mirror, and the editor offers a one-click global reversal. Every session
    records which orientation it uses (`spine_mirrored` in the fixtures, via
    `canonical_orientation`), so comparisons — against papers, across
    implementations, and between sessions — are made in a known orientation
    rather than silently across a mirror.
- **60.** **The cover π** by monodromy: choose a spanning tree T₀ of Γ₀; the
    g = |E₀| − |V₀| + 1 remaining edges generate π₁(Γ₀) = F_g; the user gives a
    permutation σ_e ∈ S_d per generator; the app builds the covering graph,
    its pulled-back ribbon structure and the edge labels. This is the standard
    encoding of a finite cover and it is what makes "input of these graphs"
    tractable — Γ₁ has d times as many edges and the user should never have to
    type them.
- **70.** **φ** by edge paths: the user gives the image of each vertex, and the app
    auto-fills an edge path (shortest in Γ₀) for each edge of Γ₁, editable
    click-by-click. Only the homotopy class matters; the app normalises it
    (collapse T₀ to a rose R₀, collapse π⁻¹(T₀) to get R₁, homotop φ so that
    vertices go to the vertex) so that all invariants are computed from the
    normal form, and checks invariance under adding a detour when the user
    edits.
- **80.** **Sessions** as JSON — one file holds all of 10–40 — so examples are
    reproducible and the Python engine and the page share a format. D3
    confirmed 2026-10-23: the text-based session is the source of truth;
    graphical views are rendered from it (text → graphical), never the other
    way round.
- **85.** **Reverse input route** (eventual, author's request 2026-10-23): from a
    PCF rational map — or one with attracting cycles — reconstruct the graph
    data (Γ₀, π, φ) as a session. Prior art: IMG's
    `SphereMachineOfBranchedCovering`; for polynomials, the Hubbard
    tree/spider gives the spine. Not in M0–M5 unless it falls out of M4.
- **90.** **Example library**, built first, because it is the test suite:
    [PChar, Fig. 3] ((1+z²)/(1−z²), with the automaton relations of Fig. 3d as
    the predicted output), [PChar, Fig. 4] (the 1/5 and 2/5 rabbits: same graph
    virtual endomorphism, different ribbon structures — the discriminating test
    for the ribbon-map check), the barycentric subdivision map and census #3.2
    of [PChar, §1.2] (Sierpiński carpet, n > 1 needed), z²+i as the
    non-hyperbolic warning case, the elastic example with weights (1, √2, 2)
    whose figure code is in `.scratch/poschar/draws/graphs.mp`, and
    [Elast, Ex. 1.17] (two tripods with elastic constants (1,3,3) and (1,1,1),
    where Emb = 3/5 and the minimiser is *not* the harmonic map — a
    hand-checkable value).
    Data (settles D4, 2026-10-23): [Harmonic] supplies machine-readable data
    for the theta example (= (1+z²)/(1−z²), per its `output.tex`), the
    bouquet/barbell forms, slit Lattés, census #3.2 (with tuned weights),
    twisted rabbit, Grigorchuk, tetrahedral and barycentric subdivision,
    blown-up Lattès and census #53; the format is specified by `GraphIO.lhs`
    (flowers give the ribbon structure; unlisted edges map to constant paths).
    More carpet examples from [ConfDim]: the barycentric carpet (exact fixture
    E²₂[φ²] = 10/13), the fat and skinny Devaney families (spines in
    `.scratch/confdim-src/draws/`, with the ŌN ≥ r_n matrix test), the
    Milnor–Tan quadratic carpet, and a quartic non-carpet (ŌN = √2) as a
    negative test.

## 3. Algorithms

### A. Validation and the critical portrait
- **100.** π₀-bijectivity and π₁-surjectivity: direct groupoid computations on the
     half-edge structure (Γ₁ connected covers Γ₀ connected, so π₀ is
     automatic; surjectivity is checked on generators).
- **110.** Ribbon map ([PChar, Def. 2.7], Rem. 2.10): the lift Nφ exists iff the
     local orders of φ⁻¹(y) on the cross-sections are consistent. I implement
     the local-order search (the paper's "inefficient algorithm") first, since
     it is clearly correct, and only then look for the efficient local version;
     the 1/5-vs-2/5-rabbit example is the acceptance test.
- **120.** Critical portrait: face of Γ₁ → small neighbourhood mapped into a face of
     Γ₀, degree = ratio of boundary-cycle lengths. Output as a labelled picture
     plus a text table; the predicted table for (1+z²)/(1−z²) is in [PChar, §2].

### B. Automaton, biset, self-similar group
- **130.** Spanning tree → rose R₀ (g loops) → R₁ = Γ₁/π⁻¹(T₀) (d vertices, d·g
     edges) → homotop φ to fix the vertex → label every edge of R₁ by (its
     label in R₁, the element of π₁(R₀) represented by φ(e)). The normalised
     form is decided by the choice of T₀ up to conjugacy; the app can show the
     effect of a different T₀ as a consistency check.
     *Correction (measured 2026-10-24; the sentence above was too strong).* Not
     every spanning tree can be used: the collapse π⁻¹(T₀) ↦ pt inside Γ₁ is
     compatible with φ only if φ sends each lift of a tree edge to a **point** of
     R₀ — this is the condition the paper states for its own example ("the two
     c edges of Γ₁ map with a constant map to the two vertices of Γ₀", c being
     the tree). Call such a tree *admissible*. Theta has three spanning trees
     and exactly one admissible one, {q}, which is the paper's tree c (paper
     a=p, b=r, c=q); it reproduces Fig. 3d and has level orders
     2, 2², 2³, 2⁵, 2⁸, 2¹⁴, 2²⁵. On the inadmissible tree {p} the section
     closure does not terminate (5105 distinct level-5 actions by depth 40) and
     the two natural readings of the section — φ of the whole lifted loop
     τgτ⁻¹, or φ of the generator's lift alone — disagree (level-7 orders 2²⁶ vs
     2⁴, against 2²⁵); neither is the map's. Over the 58 spanning trees of the
     14 library sessions, admissibility holds for 14, and exactly on those do
     the two readings give identical data; on the sessions with two admissible
     trees (barycentric, tetrahedral) the automata agree up to relabelling, so
     the consistency check of this item is a check over admissible trees.
     Implemented as `normalform.is_admissible_tree` / `admissible_trees`, with
     the default tree chosen admissible; an inadmissible tree is refused unless
     asked for explicitly (`require_admissible=False`). Two library sessions
     have **no** admissible tree — `blown-up-lattes` and `census-53` — so their
     rose automaton is not justified by their data as it stands; open.
- **140.** The **biset** B: sheets X = {1..d} with the free right G-action of the
     cover monodromy and the left G-action by the sections of 130; the wreath
     recursion g = σ_g · (g|_x)_{x∈X} is the same data in Nekrashevych's form.
     I will pin the exact convention (left/right, states = generators vs
     states = sheets) against [Nekrashevych, *Self-similar groups*, §2] and
     against [PChar, Fig. 3c–d] before writing the display code.
     Author's note (2026-10-23): the natural object is the **groupoid
     biset** — the graph presentation is already a groupoid presentation
     (objects = vertices of Γ₀), and the single-basepoint group biset of
     Bartholdi's program is its collapsed shadow, losing effectiveness by
     insisting on one basepoint. Internal model: the groupoid biset over the
     vertex set; the collapsed group biset (after 130's collapse to the
     rose) is a derived view, used for the IMG cross-check and for the
     display conventions pinned here.
- **150.** Group operations by recursion: products, inverses, sections, the action on
     Xⁿ (a permutation of dⁿ letters), the nucleus, the contracting test,
     word growth for finite-state automata, the image of π₁(Γ₀) in Aut(X*)
     (the iterated monodromy group), and the level-n Schreier graphs.
     Cross-check: [PChar, Fig. 3d] for (1+z²)/(1−z²), and — if GAP with
     Bartholdi's FR (the successor of `selfsim`) can be installed here — a
     second implementation; its companion package IMG additionally covers
     Thurston's algorithm, see 230.

### C. Energies — the core, and the part with real risk
- **160.** **Emb(ψ) for a given map** is elementary and exact: the integrand
     y ↦ Σ_{x∈ψ⁻¹(y)} |ψ'(x)| is piecewise constant on Γ₀ minus the finite set
     of images of vertices of Γ₁, so the essential supremum is a finite max.
     Everything below is about removing the dependence on the representative.
- **170.** **Emb[φ] and E^p_p[φ].** By [Elast, Thm 1], Emb[φ] = SF_EL[φ] = the
     supremum over weighted multicurves of EL[φ∘c]/EL[c], realised by a
     weighted multi-curve. Computing it: implement [Elast, §6]
     ("Minimizing embedding energy"): the length iteration of Algorithm *iter*
     (§6.2, Algorithm 6.22) — find the harmonic representative, read off tension weights,
     push them forward, convert back to lengths — and the escape procedure of
     Algorithm *Emb* (§6.5, Algorithm 6.49) for the boundary/local-maximum cases. The
     harmonic-map step is a finite combinatorial computation (taut maps and
     max-flow/min-cut, [Elast, §3 and §5]); prior art for it beyond the paper
     (review of 2026-10-23, `taut-prior-art.md`): multiflow
     theory — Karzanov–Lomonosov and Lomonosov's locking theorem for the
     multi-terminal flow/cut corollary, Seymour's sums of circuits and
     Fleischner's transition systems for the balanced-train-track
     decomposition. Honest caveats: convergence of
     *iter* is not guaranteed in general ([Elast, §6.5, Question 6.48] says so and leaves an
     open question); fixed points can be non-maximal, and Algorithm *Emb* is
     the fix; the paper says it "appears to run quickly, at least for small
     examples", which is exactly our regime.
- **180.** **Independent lower and upper bounds** (always available, no convergence
     question):
     - upper: Emb(ψ) for any explicit ψ ∈ [φ] (compute a good ψ by iterating
       170's shortening steps, or by harmonicity);
     - lower: EL(φ∘c)/EL(c) for any explicit weighted multicurve, found by
       search or by the maximising structure 170 returns.
     Their gap is what the app displays, so the user sees how much is proved.
- **190.** **Iterates.** Build X_n recursively ([PChar, Def. 5.1]) and compute
     Emb[φⁿ] and E²₂[φⁿ] (= √Emb) for n = 1, 2, 3, … until size forces
     pruning; report E^p_p[φⁿ]^{1/n}, which converges to ᴱE^p_p from above
     ([PChar, Prop. 5.6]: the limit equals the infimum). Two decisive facts
     for the app:
     - **ᴱ < 1 can be certified by one finite computation**: if Emb[φⁿ] < 1
       for some n, then ᴱ ≤ Emb[φⁿ]^{1/n} < 1, and by Theorem 1 (condition (2))
       the self-cover is rational. This is a *positive certificate*;
     - ᴱ does not depend on the elastic structure or the spine
       ([PChar, Props. 5.7, 5.14]), but Emb[φ] and whether n = 1 suffices do —
       which is exactly the content of Conjecture 1.18 and Question 1.19, so
       the app should let the user vary α (and the spine) and report the
       smallest n that works.
     - Caveat (author, 2026-10-23): in carpet examples the sequence starts
       high even though it eventually decays at the definite rate ᴱ. Census
       #3.2 with unit weights has E²₂[φ] ≈ 2.66, and ≈ 1.11 at n = 1 even
       after tuning α, while ᴱ ≈ 0.94 (Harmonic/output.tex); the old code
       never reached a certificate for it. Consequences: the terms need not
       be monotone in n (only the infimum is guaranteed); absence of a
       small-n certificate is not evidence of an obstruction — "undecided"
       is a common honest state for carpets; and optimizing α (and the
       spine) is essential for reaching any certificate, not a refinement.
       The old code's OptFilling.lhs is the seed of that optimizer.
- **200.** **Obstruction matrices** ([PChar, §7.4]): for a multi-curve C on Γ₀,
     M_{C,p}[i,j] = Σ over components D of π\*C_i with φ_*D ∼ C_j of
     deg(D→C_i)^{1−p}; Perron–Frobenius λ(M_{C,p}) ≥ 1 iff a p-obstruction is
     supported on C, which implies ᴱE^p_p ≥ 1. For p = 2 the search may be
     restricted to simple multi-curves (Prop. 7.18) — a finite search by
     complexity — giving a **negative certificate**: an obstruction exhibited
     by an eigenvector. Together with 190 the app decides rationality whenever
     it finds either certificate, and says "undecided" otherwise.
- **210.** **Plots**: Emb[φⁿ]^{1/n} against n with the certified bracket; ᴱE^p_p
     against p ∈ [1, ∞] (monotone by [PChar, Prop. 6.10], with the
     d^{−1/p+1/q} lower bound as a drawn band) — the app can generate data for
     the open questions in [PChar, §6] and for Q(π,φ) in §7.5.

### D. The rational map
- **220.** **Decision.** By Theorem 1, rational ⟺ ᴱE²₂ < 1; the certificates of 190
     and 200 make this a finite check in both directions, with an explicit
     "undecided" state and no pretence beyond it.
- **230.** **Reconstruction** (numerical; see 300). **Decided 2026-10-23: the
     reconstruction engine is the (Bill) Thurston pullback algorithm.**
     Work on normalized configurations of the postcritical points (pin three
     at 0, 1, ∞); each step builds the rational map with the required branch
     behaviour from the current configuration and the covering data, pulls
     the complex structure back along the cover, and reads the new positions
     off the appropriate preimages — per-step univariate root-finding of
     degree d, never a coupled system. The iteration contracts exactly when
     ᴱE²₂ < 1, so the positive certificate of 190 is also the convergence
     hypothesis, and the [Elast] iteration of 170 is its combinatorial
     shadow (same data, possible warm start). The route first proposed here —
     solving the critical-portrait equations directly by Newton or homotopy
     continuation — was rejected by the author: the equations grow rapidly
     in degree and the roots cluster too quickly already at moderate size.
     Root clustering still appears as precision loss across iterations, so
     the engine uses arbitrary-precision complex arithmetic (mpmath) from
     the start; verification stays as in 240. Prior art for conventions and
     a cross-check: Bartholdi's GAP package IMG (depends on FR) implements
     `ThurstonAlgorithm`, `SpiderAlgorithm`, `P1MapBySphereMachine` and
     `ThurstonObstruction`, taking the sphere machine (= the biset of 140)
     as input. The polynomial fast path stays, with the circle-map and
     itinerary machinery of the core-entropy repo (tent_map.py, src/) reused
     for it and for the Julia-set pictures:
     - polynomials first: the critical portrait plus the external-angle/kneading
       theory gives c (or the map) by Newton on the landing conditions;
     - general degree: the Thurston pullback, as above.

     Design note (author's idea, 2026-10-23; not yet developed): when marked
     points cluster in the current coordinate, blow the cluster up by a
     Möbius change of coordinates instead of losing precision — re-coordinate
     freely between steps, keep the configuration as a tree of scales, and
     solve locally in blown-up charts. Theory pointers: dynamics on trees of
     spheres (Arfeux), Selinger's extension of the Thurston pullback to
     augmented Teichmüller space (cluster collisions are the boundary
     strata), rescaling limits (Epstein, Kiwi). Caveat: blow-ups preserve
     relative precision inside a cluster; the intrinsic 1/|f′| loss for
     preimages near critical values stays, bounded per step, and is what the
     arbitrary-precision arithmetic is for. Stress test: the skinny Devaney
     family (conformal-dimension paper, skinny examples) — one limit
     configuration with necks at different scales.
- **240.** **Verification of a reconstructed map** (never trust the numerics):
     recompute its critical portrait and compare with 120; check the orbit of
     the marked points; recompute a graph virtual endomorphism of the map and
     test homotopy equivalence with the input data; compare Emb/SF values and
     the Julia set picture with the limit space of 190. The app must label the
     result "numerical, verified at this precision", not "proved".

## 4. Architecture

- **250.** **Python engine** (authoritative): exact rational arithmetic (fractions;
     sympy for symbolic linear algebra), networkx for flows/LP, numpy for the
     iterates. CLI, test suite, heavy computations (high n, searches, map
     reconstruction). D1 settled 2026-10-23: Python it is — the initial
     deliverable is a standalone Python app (slightly more convenient for the
     author), with strict model/view separation — the engine knows nothing
     about the view — so the front end can be ported later.
- **260.** **Single-page app** in the browser, in the idiom of the core-entropy project
     (tent_map.py builds a self-contained tent_map.html): a canvas graph
     editor, live recomputation of the cheap and interactive parts
     (validation, portrait, automaton, biset, obstruction matrices, Emb by
     iteration on small inputs), plots, session import/export. Per D1
     (2026-10-23) the page is the M5 port of the view layer, not a second
     engine: it mirrors the engine's arithmetic in JS only for the parts that
     must be live, with the same cross-check arrangement the core-entropy repo
     already uses for Python vs JS (370), from M5 on.
- **270.** **Headless checks**: node v20 is available at ~/node/bin/node (jsdom is
     not installed globally; the core-entropy project's harnesses in its `.scratch/js` show the
     pattern — install what is needed into a git-excluded folder). Plus
     py_compile, and the usual "no real GUI here" caveat: nothing visual can
     be verified beyond headless rendering; the first real test is you opening
     the page.
- **280.** **Makefile targets** (`make applet`, `make check-applet`) and docs, so the
     applet has its own build and its own page, not inside the core-entropy one.

## 5. Milestones

Order confirmed 2026-10-23 (D6): as listed — algebra (M2) before energies
(M3).

- **290.** **M0 — spike, the paper's own example.** (1+z²)/(1−z²): data in, portrait
     out, automaton out, matching [PChar, Fig. 3]. Small, and it exercises the
     whole data model; it is also the smallest possible demonstration that the
     plan's foundations are right.
- **300.** **M1 — data model, input, validation, library.** 50–90, 100–120 (the
     item numbers of §2 and §3.A): the data model and JSON sessions made
     final, the validation suite (π₀/π₁ checks of 100, the ribbon-map
     local-order search of 110), the portrait output of 120, and the example
     library of 90 with expected outputs recorded as fixtures (350).
- **310.** **M2 — algebra.** 130–150: the normal form, the biset, the group
     machinery, level-n Schreier graphs; the Fig. 3d relations as a test.
- **320.** **M3 — energies.** (The type classification of §1 — hyperbolic
  type / cycle without branch point / non-compact, [PChar] §1.2 — belongs here
  with the main theorem's hypothesis; it is not a data-validity check: author,
  2026-10-24.) 160–210; the longest and riskiest milestone: [Elast,
     §3, §5, §6] need a careful read before coding, and the convergence
     caveats of 170 are real. Deliverable: a number for Emb[φ], with its
     certified bracket, and the iterate plot with certificates.
- **330.** **M4 — the rational map.** 220–240; numerical; scoped to the paper's
     examples and the polynomial family first, with the Thurston pullback of
     230 (decided 2026-10-23) as the engine and Bartholdi's IMG as the
     cross-check.
Deferred beyond the milestones (end-user **input front ends**, not engine work —
they emit sessions and never modify the engine; deliberately postponed until a
first implementation exists):

* **Schwarz–Christoffel importer.** Input $(f, P, \text{spine coordinates})$;
  output a session. Pieces: (1) analytic portraits from $f$'s coefficients —
  **done** (`rational.py`, cross-checked against four sessions); (2) the spine
  and an explicit retraction $\rho$ per model family (model-specific; the
  nearest-point retraction onto a rectangle's boundary is not one — punctures
  pinch it); (3) tracking $\Gamma_1 = f^{-1}(\Gamma_0)$ through critical values
  and reading off $\varphi$'s words; (4) emission. Acceptance is free: the
  ribbon boundary pattern, $\pi_1$, and the portrait oracle. First target:
  `julia-4442` (doubled rectangle, aspect 1.279; portrait pinned at
  {2,2,1,1,1,1}). Estimated: an afternoon for (1) [done], a focused day per
  model family for (2), one to two days for (3).
* **Twisting by mapping-class elements.** Input a curve in $S^2\setminus P$
  (word in $\pi_1$ up to conjugacy) and an exponent, or a product of twists;
  transform the VE by the automorphism the twist induces on $\pi_1$
  (BMW / Bartholdi–Nekrashevych picture) and re-normalise. Technical issues:
  curve specification, the automorphism's effect on the spanning-tree
  bookkeeping. Acceptance: portrait, type and energies are **invariant** under
  twisting — already computable, so the test exists before the feature.
  The twisted-rabbit session's data stays as found until then (author may
  supply the correct string; the app will eventually derive twists itself).

- **340.** **M5 — packaging.** 260, 280; a page on the same site only when you say
     it is finished (nothing web-visible is pushed without asking).

## 6. Verification, and what is not verified

- **350.** Fixtures with predicted outputs: 90, plus the hand-computable
     [Elast, Ex. 1.17] (Emb = 3/5, with the minimiser not harmonic), the exact
     iterate values of [Harmonic]'s `output.tex` (bouquet degeneration of the
     theta example: Emb[f²] = 3/4, f³ = 5/8, f⁴ = 23/48, f⁵ = 187/480;
     theta-family limit 2^{−1/3}) and [ConfDim]'s E²₂[φ²] = 10/13 for the
     barycentric carpet.
- **360.** Two-sided energies: exact rational arithmetic wherever the quantity is
     rational; certified brackets from 180; sampling/eigenvector lower bounds
     as a third, independent route. D5 settled 2026-10-23: no interval
     arithmetic initially — exact certificates plus bracketed numerics (180)
     are enough; interval methods may return for the M4 numerics.
- **370.** Cross-implementation: Python vs the page's JS on the whole library;
     totals reported as "k differences over m inputs", as in the core-entropy repo's
     habit.
- **380.** Not verified by me, ever: any real GUI interaction (no display here), the
     numerical rational map (only its verification criteria 240), and the
     unbounded searches (ribbon-map orders, obstruction multi-curves) beyond
     the complexity actually run.

## 7. Out of scope, or explicitly not claimed

- **390.** Non-compact type and orbigraphs ([PChar, §1.2]'s future work); infinite
     graphs; disconnected surfaces of [PChar, Def. 1.4]; transcendental maps.
- **400.** General reconstruction of the rational map as a *proof* — only numerical
     reconstruction with verification.
- **410.** Exactness of the asymptotic energies as closed forms: ᴱ is computed as a
     limit with certificates on either side, not as a symbolic expression.

## 8. Decisions

All settled 2026-10-23 (original numbering preserved):

D1 — Python for the initial implementation; a standalone Python app is
slightly more convenient; strict model/view separation so the view can be
ported later (250, 260).
D2 — a separate repository, `dpthurst/pullback-algorithm`; its page deploys
to that repository's Pages only on the author's word.
D3 — the text-based session is the source of truth; text → graphical views
wanted; eventually the reverse input route, rational map → graph data
(80, 85).
D4 — example data from [Harmonic], machine-readable (90).
D5 — no interval arithmetic initially; exact certificates + bracketed
numerics (360).
D6 — algebra (M2) before energies (M3) (§5).
Also amended after review: the reconstruction engine of 230 is the Thurston
pullback algorithm, not a direct solve of the critical-portrait equations.

If you want extra hands on the two hard cores (M3's algorithm, M4's
reconstruction), I can run an Ultramath swarm on them in parallel; it would
work in `.ultramath/` and not touch this plan.
