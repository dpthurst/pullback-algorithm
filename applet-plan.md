# Applet for graph virtual endomorphisms — plan of attack

Draft for review, 2026-10-23; all decisions D1–D6 settled the same day (§8).
Nothing is built yet; this file is the thing to argue with before any code is
written. Items carry stable semantic labels (multiples of 10, non-sequential),
rendered as bold bullets so that source and rendered numbers agree; the labels
— not list ordinals — are the cross-references throughout.

Sources used, all in `.scratch/`:
[PChar] D. Thurston, *A positive characterization of rational maps*,
Ann. of Math. **192** (2020) 1–46 (arXiv:1612.04424; LaTeX source + MetaPost
figure code extracted to `refs/poschar/`);
[Elast] D. Thurston, *Elastic Graphs*, Forum of Math. Sigma **7** (2019) e24
(arXiv:1607.00340; source in `refs/elastic/`);
[Rubber] *From rubber bands to rational maps: a research report*,
Res. Math. Sci. **3**:15 (2016) (arXiv:1502.02561; `refs/rubber/`);
[KPT] Kahn–Pilgrim–Thurston, *Conformal surface embeddings and extremal
length* (arXiv:1507.05294);
[ConfDim] K. Pilgrim & D. Thurston, *Ahlfors-regular conformal dimension and
energies of graph maps*, Groups Geom. Dyn. **20** (2026) 971–1050
(arXiv:2112.09041; source in `refs/confdim-src/`);
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
    whose figure code is in `refs/poschar/draws/graphs.mp`, and
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
    `refs/confdim-src/draws/`, with the ŌN ≥ r_n matrix test), the
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
     *Correction (2026-10-24, author's argument; supersedes an earlier note of
     the same day that claimed too much).* Every component of π⁻¹(T₀) is a tree,
     hence contractible, so **every** φ is homotopic to a map constant on each
     component — homotope φ|C to a point and drag the incident edges along; the
     homotopy leaves φ_* alone, so the virtual endomorphism h = φ_*∘π_*⁻¹ is
     untouched and R₁ → R₀ always exists. There is no obstruction and no
     "inadmissible" session: the construction is available for every (π, φ) and
     every spanning tree, the tree only choosing a presentation.
     What is *not* available for every tree is the shortcut the engine used: read
     the section as φ of the lift of τgτ⁻¹. That is exact only when φ already
     kills the lifted tree ("admissible" in the earlier note), because otherwise
     the collapsed part of the loop carries a nontrivial loop into R₀ and the
     reading silently differs from the homotoped one. Measured: the two readings
     agree on exactly 14 of the 58 spanning trees of the library, and they are
     exactly the trees where φ kills the lifted tree. So the correct recursion
     comes from a transversal: with t(x) ∈ π₁(Γ₀) a loop lifting to the vertex of
     class x,
         g|_x = h( t(σ(g,x))⁻¹ · g · t(x) ),   g·t(x) = t(σ(g,x))·(g|_x),
     and the composite lies in H = π_*π₁(Γ₁) by construction, so its lift to Γ₁
     closes — a check that can fail and does when the bookkeeping is wrong.
     *Status (2026-10-24): **not fixed yet**.* The construction is implemented in
     `.scratch/nf_transversal.py`; the matrix of the four convention choices is in
     AGENTS.md §A11. With σ read as the class of the lift at x, "t(x) first"
     composition closes on all 7 trees but still returns 2²⁶ for theta {p};
     with the other composition all three theta trees return 2²⁵ but two
     blown-up-lattes trees fail the closure assertion. So the bug is narrowed but
     not found, and nothing is shipped: the engine keeps the admissible-tree
     shortcut.
     *Fixed (2026-10-25).* It was none of the four conventions: a section word is
     a group element and the engine applies it left-factor-outermost, but
     `normal_form` and `RoseAutomaton` stored the φ-image of the lifted loop in
     traversal order. For a two-letter section the two differ, and the mirror
     machine agrees at level 2 exactly when every letter acts as an involution on
     level 1 — so theta's {q} (all sections one letter) was right and {p}
     (r|₁ = q·r⁻¹) gave 2²⁶. Both constructors now store the reversed word. All
     three theta trees then give 2¹, 2², 2³, 2⁵, 2⁸, 2¹⁴, 2²⁵ and are
     contracting, FR agrees on the corrected machine, and the independent check
     is a geometric level-n action of (π, φ) computed by lifting through the
     pullback tower with no tree at all (it reproduces the rabbit's and theta's
     published sequences and gives 2²⁵). The transversal recursion above is
     therefore not needed for this; it was measuring the same mirror machine.
     AGENTS.md §A0, "Resolved", and §A0.6 have the detail and the two regression
     tests.
     *Check to run when the library is bigger (author's suspicion, 2026-10-24:
     "most of the time that check will fail").* Record, over the sessions as they
     are added, the fraction of spanning trees that pass `is_admissible_tree`.
     On today's 14 sessions it is 14 of 58 — the check fails more often than it
     passes — and theta alone has 1 of 3. If it keeps failing at that rate the
     gate is not a practical primary path: it exists only to make the current
     shortcut exact, so the transversal construction should replace it and the
     tree should be chosen arbitrarily. Until then the gate stays, and the number
     to watch is that fraction, not a proof.
     `normalform.is_admissible_tree` / `admissible_trees` and the default-tree
     choice remain in force for the shipped engine, as a *shortcut validity*
     check, not as a claim about the data. The groupoid biset of item 140 is
     the natural object, but the single-basepoint recursion does not need it:
     the transversal suffices, and 140 now records the decision that the
     collapsed shadow is what the engine carries.
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
     insisting on one basepoint.
     **Author's decision (2026-10-25): the collapsed shadow suffices for
     now.** The engine carries the single-basepoint group biset after 130's
     collapse to the rose — the wreath recursion above, which is what
     `Automaton`/`RoseAutomaton` hold; no groupoid-over-the-vertex-set
     machinery is built. It is enough for two reasons. (a) What lies downstream
     already consumes the collapsed form: in IMG 0.3.3 `ThurstonAlgorithm` and
     `ThurstonObstruction` take an `IsSphereMachine` and `SpiderAlgorithm` an
     `IsFRMachine` — wreath recursions, not groupoids (`img/gap/thurston.gd`,
     `machine.gd`, checked 2026-10-25). (b) Every question asked so far is
     single-basepoint — the level action, the nucleus, the Schreier graphs, the
     FR cross-check. The price is effectiveness, and it is real rather than
     notional: the collapsed representation can have a kernel π₁ → IMG, which
     `twisted-z2i` exhibits (`AGENTS.md` A0.1–2). Revisit only if something
     needs the vertex-anchored biset, or effectiveness as a statement.
     *Convention, as the engine implements it (settled 2026-10-25):* states =
     generators of π₁(Γ₀), alphabet = the d sheets, **left** action, left factor
     outermost, sections stored as group elements, so a geometric section given
     in traversal order is reversed (130). FR acts on the right and
     `scripts/img_crosscheck.py` reverses multi-letter sections on emission.
     [Nekrashevych, *Self-similar groups*, §2] is the formal citation for the
     convention; what settled it in practice was the geometric tower oracle
     agreeing with the engine, and FR agreeing with both (0 mismatches).
- **150.** Group operations by recursion: products, inverses, sections, the action on
     Xⁿ (a permutation of dⁿ letters), the nucleus, the contracting test,
     word growth for finite-state automata, the image of π₁(Γ₀) in Aut(X*)
     (the iterated monodromy group), and the level-n Schreier graphs.
     *Status (2026-10-25).* Implemented: products, inverses, sections, the level-n
     action, the nucleus, level-n Schreier graphs, the Fig. 3d check and the FR
     cross-check (0 mismatches; σ/sections and the level action validated).  Not
     implemented, and deferred by the author (see below): word growth, and the IMG as a group
     (only its level images exist — the kernel of π₁ → IMG can be nontrivial, e.g. `twisted-z2i`).
     Contraction: **the author's decision (2026-10-25) is to work with the graph π₁
     for now** — much simpler — and generalise to π₁^orb and the IMG later.  The
     engine's `nucleus()` is exactly the π₁-nucleus, computed with exact dedup (free
     reduction), so it is a semi-decision; `nucleus(level=n)` gives the image on Xⁿ,
     which is the IMG up to the level-n kernel.  The two differ on `twisted-z2i`
     (π₁ does not contract, the image does), which is why the group has to be named
     whenever contraction is reported.
     Word growth: deferred (author, 2026-10-25).  On the graph π₁ it is trivial — free of
     rank g, so 2g(2g−1)^{n−1} elements of length n — and the interesting growth is that of
     the IMG, which presupposes the IMG as a group (deferred, see below; only its level images exist).
     Nothing else in the pipeline consumes it.  The author's reasons for flagging it: it is
     related to the p = ∞ energy "in some way I have not personally fully understood"; the
     Grigorchuk group has intermediate growth and its exact growth rate is not known
     (Bartholdi, 2000; later work unchecked).  The 1/2 he recalls is the **asymptotic**
     energy ᴱE^∞ = lim (E^∞_∞(φⁿ))^{1/n}, which is independent of the starting weights —
     not Lip on an explicit representative (the library's Lip values: theta 1, julia-4422 1,
     julia-4442 2, census-53 ≈ 2.906; grigorchuk has no weights) — and for the Grigorchuk
     example it comes from a suitably orbifolded graph, i.e. from π₁^orb, deferred with the
     rest of the orbifold.  ᴱE^∞ is not computable in the engine: E^p_∞ of an iterate is not
     implemented.  Author's expectation (2026-10-25): for examples with an unbranched cycle,
     ᴱE^∞ will usually be 1.
     The cheap stand-ins the app already computes are the level-image orders |G_n| and the
     nucleus size, which measure the action's growth, not the word metric.
     **The IMG as a group is deferred** (author, 2026-10-25): computing it is a challenging
     task and it is **not necessary for the main applet**. The deliverable stays the level-n
     images plus `nucleus(level=n)` — the IMG up to the level-n kernel — and the kernel of
     π₁ → IMG is named where it matters (`twisted-z2i`) but never computed. Out of scope unless
     the author asks: the quotient π₁ → IMG as an object, its word problem, its exact nucleus,
     `FRGroupByVirtualEndomorphism` as a check on the derivation, and the IMG's word growth.
     Emitting a sphere machine for IMG's *external* `ThurstonObstruction`/`SpiderAlgorithm` is a
     different thing (200, 140) and is unaffected: that consumes σ/sections, which the engine has.
     Cross-check: [PChar, Fig. 3d] for (1+z²)/(1−z²), and Bartholdi's FR
     (the successor of `selfsim`) as a second implementation; its companion
     package IMG additionally covers Thurston's algorithm, see 230.
     *Status of that cross-check (2026-10-24).* FR needs IO, GAPDoc,
     Polycyclic and FGA; only IO was missing, and FR uses it in four places,
     all in the graphviz/Jupyter *drawing* path, never in the automaton
     mathematics. FR and IMG therefore stay byte-identical to their published
     tarballs (no patching). IO is absent from this machine, possibly by design
     (it is the package that gives GAP files and sockets), so the stub that
     satisfies the loader lives *inside the project* as a private GAP root,
     `.scratch/gap-root/pkg/io`, opted into only by `.scratch/gap.sh`; nothing
     is installed in the shared GAP root. Under that wrapper `LoadPackage("fr")`
     and `LoadPackage("img")` are both true and FR's own `tst/testall.g` reports
     0 failures in 6 files; plain `gap` still has no IO.
     *Done (2026-10-24): 0 mismatches.* `scripts/img_crosscheck.py --gap
     .scratch/gap.sh` emits each session's wreath recursion as FRGroup
     definitions (from the engine's own σ/sections, nothing transcribed) and
     compares the level actions. theta: 2, 4, 8, 32, 256, 16384, 2^25 — FR and
     the engine agree at every level 1–7. rabbit: 2, 8, 128, 16384, 2^28, 2^56 —
     agree at every level 1–6. Both are contracting for FR's `IsContracting`
     too. The nucleus sizes differ *by definition*, and both are right: ours is
     the closure of the generators and their inverses under sections, counted as
     reduced words (theta 5, rabbit 7); FR's `NucleusOfFRSemigroup` is the set of
     limit states of all elements (theta 4, rabbit 13). theta reconciles exactly,
     because r is an involution and our 5 words are 4 elements; for the rabbit
     ours is a subset (all 7 words are in FR's 13) and FR's extra states include
     products such as a⁻¹b, which the limit-state definition admits. So the
     sharp comparison is the level orders, not the nucleus size.
     *Convention note (2026-10-25).* FR acts on words on the **right** — its
     manual defines (xv)a = x^{π_a}·(v)a_x, with a acting first on each subtree
     xX* and then permuting them — so ρ_FR(g) = ρ_ours(g⁻¹), and every
     multi-letter section word is reversed when it is handed to FR;
     `scripts/img_crosscheck.py` does that and prints what it reversed. The
     agreement above is unaffected, because the machines with single-letter
     sections (theta's {q}, the rabbit) need no reversal. *Correction
     (2026-10-25):* the earlier reading of theta {p} here was wrong. The engine
     was storing the *traversal* word of a section rather than the group
     element, so reversing it for FR produced a machine that was right for FR
     and wrong for the engine, and both reported 67108864; the number was 2²⁶
     because of that bug, not because of the convention. With the storage fixed
     (item 130) the reversal does what it says: FR on `r=<q^-1,q*r^-1>` gives
     33554432, the engine's number, and the string that gives 67108864 is the
     old one. FR still only arbitrates the *machine*; what checks the
     derivation from (π, φ) is now the geometric tower action of item 130,
     which FR also confirms on a 3-sheet product-section witness (orders 6, 12,
     12). This validates σ/sections, the level action and the closure against an
     independent implementation; for the derivation, either `FRGroupByVirtualEndomorphism`
     or IMG's rational-map route (PoirierExamples, PolynomialSphereMachine), see 230.

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
     *Independent upper bound, no minimisation needed (2026-10-24; the
     author's correction of an earlier miscount).* πⁿ₀ is a covering of degree
     deg(π)ⁿ; π is an isometry on edges, so the dilation of φⁿ₀ at a preimage
     telescopes to the *product* of the dilations at x₁…x_n, and summing over
     the tree of preimages is a matrix power on the base edges,
     M[l][l′] = Σ δ(f)^(p−1) over traversals of base edge l by φ(f) with
     π(f) = l′. Then Fill^p(φⁿ₀) = Mⁿ·1 and
     ᴱE^p_p ≤ (max_l (Mⁿ1)_l)^{1/(pn)} → ρ(M)^{1/p}. For p = 2 this is a cheap
     *upper* bound on Emb[φⁿ] that the [Elast, §6] minimiser must come in at or
     below — an acceptance check for Algorithm *iter*, not a substitute for it.
     *Status (2026-10-25).* **Implemented and exact.** `src/pullback/refined.py`
     (marked elastic graphs, PL maps, the refined constant-derivative piece
     model of lipschitz.tex def:const-deriv, the class certificate),
     `src/pullback/minimise.py` (alg:iter with the harmonic step as an exact
     rational QP over the relaxed polytope of harmonic.tex def:relaxed — the
     max-flow/min-cut side of taut.tex thm:maxflow-mincut — and an exact
     fixed-point eigen-solve verified against the real iteration),
     `src/pullback/certificates.py` (the SF_EL lower bounds of eq:sf-el),
     `src/pullback/iterate_graphs.py` (the explicit G_n).  Verified exact:
     [Elast] Ex. 1.17 Emb = 3/5; examp:partial-filling Emb = 2 (boundary fixed
     point, no λ-filling map — the value comes from the two-sided bracket,
     not the escape); the barbell Emb[φ¹] = 5 < 6 (the explicit bound is not
     sharp); the bouquet Emb[fⁿ] = 1, 3/4, 5/8, 23/48 at n = 1..4, matching
     the published Harmonic values exactly from the fixed point alone.
     **Scaling (2026-10-25).** The exact path above builds the relaxed polytope from all
     simple marked curves of the source, which is exponential in |E(G_n)| (3, 23, 97, 5378
     constraints at n = 1..4). `src/pullback/fastmin.py` adds the paper's own harmonic step
     — a least-squares problem in the source vertices' *positions* (Harmonic.lhs `findPos`)
     plus the push/pull type repair — in float64, and reproduces θ at n = 1..8 to ~1e-7
     (0.864357 … 0.178296 and 1.5 … 0.227530). The exact engine remains the certificate
     path at n <= 3.
     **The convex formulation (2026-10-26).** The fast path's bouquet failure is structural
     — its type search has barriers (a vertex inside one target edge cannot move into a
     neighbouring one without passing through their common vertex, where Dir is higher) —
     so `src/pullback/covermin.py` lifts the energy to the universal cover of the target,
     where F(x) = Σ_e d_T(x_u, τ_e x_v)²/α₁(e) is convex (the author's "the convexity of
     the problem seems key"): no type is searched, the minimiser is unique up to the global
     deck translation, and no cover is built (a lifted point is a reduced word plus an
     offset; all arithmetic exact `Fraction`). The harmonic step is coordinate descent with
     an exact rational block minimiser (`minimise_one`: the Steiner tree of the landing
     points, covered by the atomic segments of the pairwise geodesics, a convex quadratic
     per segment), and the run carries its own certificate — the directional tension gaps,
     all ≤ 0 at the optimum, which is the balance condition of harmonic.tex
     cor:harmonic-triangle. Verified: the bouquet 1, 3/4, 5/8, 23/48, 187/480 at n = 1..5
     and θ (1,1,1) 3/2, 0.773459, 6/7 at n = 1..3 (|λ − value| ≤ 5.1e-17; exact at
     bouquet n = 1,3 and θ n = 1), θ (4,1,4) n = 1 = 0.864356777 (the paper's); the
     harmonic step agrees with the exact QP at unit lengths over six cases
     (0 ≤ Dir − Dir_QP ≤ 3.5e-9, |v − v_QP| ≤ 1.9e-5, Dir_id exact). Slower than fastmin
     (bouquet n = 5 in 31 s; θ n = 3..7 in 13, 215, 291, 2267, 5960 s — it matches
     the paper's 0.309249 at n = 7; n = 8 attempted and abandoned 2026-10-26, a
     cross-check not worth the hours) but correct everywhere; two speed knobs measured
     2026-10-26: a tolerance stop on alg:iter's factor (the exact-repeat stop never fired, so
     every cold run burned the 40-step cap; at a relative 1e-12, θ (4,1,4) n=2,3
     go 25.4 → 6.0 s and 67.1 → 17.9 s, the λ agreeing to ≤ 1e-12), and a warm
     start from the previous iterate's fixed-point metric (θ (4,1,4) n=3:
     66.9 → 14.2 s warm alone, 4.9 s warm + tol, 13.7×; but θ (1,1,1) n=5 gets
     slower, 291 → 331 s, because that row's fixed points oscillate — chain the
     warm start only where they vary smoothly). A third knob, a diminishing-
     precision schedule for the early harmonic descents (`htol_loose`), is in the
     code but off: measured, it does not help reliably (the jitter delays the
     tolerance stop). `tests/test_covermin.py` (16 tests) cross-checks it
     against the QP, the published values, the certificate and determinism.
     A database-wide iteration check (2026-10-26, all 15 sessions at n = 1..2,
     12 steps, comparison-free invariants: certificate, Dir(harmonic) ≤
     Dir(input), v ≥ 0, DR ≤ explicit bound, DR monotonicity up to 1e-6, and
     Emb(input) vs the matrix power) passes 23 of 30 runs (one of them, barbell
     n = 2, with a recorded note: the G_n enumeration undercounts there, the
     known recorded corner; the two agree at n = 1 on every session), with 6
     documented skips (census-3.2 fails in `session_iterate` exactly as the
     exact engine; census-53's irrational elastics are outside the
     exact-rational engine; the twisted sessions' G_2 is disconnected at
     n = 2) and 1 documented exception (twisted-z2i's DR decrease — the
     descent stalls at a start-dependent balanced point, the free class has
     several). `make check` keeps the n = 1 slice (`tests/test_database.py`);
     the fuller n = 1..2 sweep is the tracked `scripts/db_iteration_check.py`,
     run on demand with `make dbcheck` (and deliberately not part of `check`),
     which whitelists that one exception and exits 0 while still printing it.
     It caught one real bug: `emb`
     returned the multiplicity dict instead of the scalar essential supremum —
     fixed with a regression test.
     **Not implemented:** the escape procedure of Algorithm 6.49 (unnecessary
     on every test case so far, including the paper's boundary example). The
     fast path's bouquet wrongness (n ≥ 2) is fixed by covermin, not by fastmin —
     fastmin keeps its recorded limitation (a skipped test) and stays the fast path.
     The session schema now carries `marked` (base vertices whose lifts are
     pinned; empty = the free class, the sessions' convention) and reserves
     `orbifold` (cone points, orbigraphs — independent of `marked`, so a
     graph may carry both); nothing reads cone points yet.
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
     *Status (2026-10-25).* On `p-ne-2-energies` the explicit-representative energies
     E^p_p[φⁿ] exist for every p via the fill matrix, but **E^p_∞ of an iterate is not
     implemented** (it needs the edge decomposition, not just the filling), so the
     asymptotic ᴱE^∞ cannot be computed yet.  The author notes that this is the quantity
     he had in mind for the Grigorchuk example — he recalls ᴱE^∞ = 1/2 there — and that it
     would come from a suitably *orbifolded* graph, i.e. from π₁^orb, which is deferred.
     Since ᴱ is weight- and spine-independent (above), the explicit Lip values are the
     wrong object for the question.  Author's expectation: for examples with an unbranched
     cycle, ᴱE^∞ will usually be 1.
     *Note (agent, 2026-10-25, not the author's).* ᴱE^∞ should be the *easiest* of the
     asymptotic energies to implement: since l₀(φⁿ(e))/α_n(e) telescopes into a product of
     the edge dilations δ along the orbit (A0.2), Lip(φⁿ) is a max over length-n paths in a
     finite graph, so ᴱE^∞ = lim Lip(φⁿ)^{1/n} is the **max-plus (tropical) eigenvalue** of
     the dilation matrix — the maximum over cycles of the geometric mean of δ — computable by
     Karp's algorithm, no Perron–Frobenius needed.  That matches "the p = ∞ energy is easily
     computed", and it explains the unbranched-cycle expectation: a boundary loop carried
     along the cycle has dilation 1, so the max cycle mean is 1 and ᴱE^∞ = 1.
       The old code's OptFilling.lhs is the seed of that optimizer.
- **200.** **Obstruction matrices** ([PChar, §7.4]): for a multi-curve C on Γ₀,
     M_{C,p}[i,j] = Σ over components D of π\*C_i with φ_*D ∼ C_j of
     deg(D→C_i)^{1−p}; Perron–Frobenius λ(M_{C,p}) ≥ 1 iff a p-obstruction is
     supported on C, which implies ᴱE^p_p ≥ 1. For p = 2 the search may be
     restricted to simple multi-curves (Prop. 7.18) — a finite search by
     complexity — giving a **negative certificate**: an obstruction exhibited
     by an eigenvector. Together with 190 the app decides rationality whenever
     it finds either certificate, and says "undecided" otherwise.
     *Deferred example (author, 2026-10-25).* The session `twisted-z2i` — the
     string Harmonic wrote for the "Hamal birthday conference" example,
     `b0Bab c0BAbab b1c` — is, in the author's belief, a twist of z²+i, and
     whether it is obstructed is deliberately left as an open question, to be
     settled by this machinery when it exists. In principle it can be read off
     from Bartholdi–Nekrashevych, *Thurston equivalence of topological
     polynomials*, or Belk–Lanier–Margalit–Winarski, *Recognizing topological
     polynomials by lifting trees*; the author prefers not to chase it by hand.
     The orbifold is hyperbolic (χ_orb = −1), so the usual criterion applies,
     and on a 4-punctured sphere only the three 2+2 multi-curves need checking.

- **210.** **Plots**: Emb[φⁿ]^{1/n} against n with the certified bracket; ᴱE^p_p
     against p ∈ [1, ∞] (monotone by [PChar, Prop. 6.10], with the
     d^{−1/p+1/q} lower bound as a drawn band) — the app can generate data for
     the open questions in [PChar, §6] and for Q(π,φ) in §7.5.
     Author's decision (2026-10-25): **iterate plots are scripts and static
     figures**, not an interactive page — there is no applet shell yet, and M3
     is not to wait for one. Each plot script writes a deterministic figure
     (fixed seed, no display) from the library, and M5 may port it later.

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
- **Viewing numerical maps (author, 2026-10-25).** The app will hold non-combinatorial
     objects as well: actual numerical rational maps, with a viewer — the Julia set, and
     eventually a 3D view of the Julia set on the Riemann sphere. A ray tracer is overkill.
     Built first: the sphere view in the app's Map panel (`pullback/julia.py`,
     `app/sphere.py`), CPU-rasterised, because this environment has neither Qt3D nor a
     usable OpenGL context under the offscreen platform, so a GL view could not be verified.
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
     *Status (2026-10-25).* **Done:** the normal form on any spanning tree (130; the
     section-word order bug fixed, all θ trees agreeing), group operations and level-n
     Schreier graphs (150), the Fig. 3d check, and the FR cross-check (0 mismatches). The
     biset (140) is settled as its collapsed single-basepoint shadow (author, 2026-10-25),
     which is what the wreath recursion already is; no groupoid biset is built.
     **Deferred, by the author's decision of 2026-10-25:** the IMG as a group (a challenging
     task, not necessary for the main applet — see 150); word growth (150).
     **Open in M2, in the author's order:** contraction on the graph
     π₁ only (his decision; π₁^orb and the IMG later); the rabbit's three presentations
     verified only to level 6.

- **320.** **M3 — energies.** (The type classification of §1 — hyperbolic
  type / cycle without branch point / non-compact, [PChar] §1.2 — belongs here
  with the main theorem's hypothesis; it is not a data-validity check: author,
  2026-10-24.) 160–210; the longest and riskiest milestone: [Elast,
     §3, §5, §6] need a careful read before coding, and the convergence
     caveats of 170 are real. Deliverable: a number for Emb[φ], with its
     certified bracket, and the iterate plot with certificates.
     *Contraction is type-conditional (author, 2026-10-24).* For a map with a
     cycle in P containing no branch point (e.g. z²+i), the automaton is **not**
     contracting on π₁(S²∖P) and is not meant to be: it is contracting only on
     the orbifold quotient of π₁. So "the automaton is not contracting" is a
     *defect* only for hyperbolic type, and the type classification gates every
     reading of it. Two consequences for the engine: (a) never report
     non-contraction as a data error — for the unbranched-cycle type it is the
     expected answer and the invariant lives on the orbifold; (b) the IMG and its
     nucleus should be computed on the orbifold quotient when the type calls for
     it. And a caveat on our own test: `Automaton.nucleus` closes the generators
     under sections **deduping on reduced words**, so its non-termination is not
     a proof of non-contraction — it can fail to terminate for a contracting
     group whose generating set has relations among the section words. Before any
     contraction claim, dedupe by element (the action on a level, with a stated
     separation bound) or build the finite Mealy machine when one exists.
     *State (2026-10-24).* The p ≠ 2 part is paused by the author and expected
     to be rewritten — the equations are linear only at p = 2. Two facts fixed
     for whatever replaces it: πⁿ₀ is a covering of degree deg(π)ⁿ (author),
     so |V(G_n)| = deg(π)ⁿ|V(G₀)| and |E(G_n)| = deg(π)ⁿ|E(G₀)|; and edges of
     G_n are cut where the coordinates cross vertices, all n coordinates at the
     same parameter values, which is what makes each edge map homeomorphically.
     Enumerating G_n by edges whose φ-image is a single dart undercounts —
     census-3.2's max Fill² at n = 2 comes out 4 instead of 7. The p ≠ 2
     implementation lives on the local branch `p-ne-2-energies`, not on main;
     the matrix bound in 170 is the cheap check it has to respect.
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
ported later (250, 260). **Reconfirmed and widened by the author 2026-10-26:**
the first deliverable is the standalone **desktop** app ("applet" means that,
not the browser page), and the project is to be used for **scripting** as
well as display, so the engine's public surface is a deliverable in its own
right. Model/view separation unchanged; the browser page stays the optional
later port of the view.
D2 — a separate repository, `dpthurst/pullback-algorithm`; its page deploys
to that repository's Pages only on the author's word.
D3 — the text-based session is the source of truth; text → graphical views
wanted; eventually the reverse input route, rational map → graph data
(80, 85).
D4 — example data from [Harmonic], machine-readable (90).
D5 — no interval arithmetic initially; exact certificates + bracketed
numerics (360). **Reconfirmed by the author 2026-10-25** ("no need for interval
arithmetic at this time"), which also settles how the energies layer's
`Energy(base, exponent)` is meant to be read: exact base, numeric bracket.
D6 — algebra (M2) before energies (M3) (§5).
Also amended after review: the reconstruction engine of 230 is the Thurston
pullback algorithm, not a direct solve of the critical-portrait equations.

If you want extra hands on the two hard cores (M3's algorithm, M4's
reconstruction), I can run an Ultramath swarm on them in parallel; it would
work in `.ultramath/` and not touch this plan.
