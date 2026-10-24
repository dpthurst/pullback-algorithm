# Prior work on tautness — a review of [Elast] §3 ("Reduced and taut maps")

2026-10-23. Requested by the author before work starts on the applet.
Source reviewed: `taut.tex` of arXiv:1607.00340 (local copy in
`.scratch/elastic/`), i.e. §3 of *Elastic Graphs*, Forum of Math. Sigma **7**
(2019) e24.

**What the section proves.** Reduced and taut maps exist in every homotopy
class of maps from a weighted graph to a graph; the four-way equivalence
taut ⇔ locally taut ⇔ carries a compatible balanced train track ⇔ carries a
compatible weighted multi-curve (Thm. `maxflow-mincut`), proved through local
models to star graphs with uncrossed nested families of minimal cuts; a
multi-terminal flow/cut corollary (`maxflow-mincut-std`); finiteness of
combinatorial types of strongly reduced maps; piecewise-linearity of $N_{[f]}$
in the weights.

**Citation audit.** The section itself cites exactly two works — Penner–Harer
(`PH92`) and Bestvina–Handel (`BH92`), both in the remark about train tracks.
The sentence "The max-flow/min-cut theorem says that…" has no reference, and
nothing from flow theory appears anywhere in the paper's bibliography.

## Direct overlap — should be cited

1. **Lomonosov's locking theorem; Karzanov–Lomonosov free multiflows.**
   Corollary `maxflow-mincut-std` (simultaneous $v_i$-cuts $S_i$ and pairwise
   flows $c_{ij}$ with $\sum_{j\ne i} w(c_{ij}) = w(S_i)$, each side
   minimal/maximal for every terminal) is essentially the singleton case of
   the locking theorem: a multiflow *locks* a family of terminal subsets if it
   is simultaneously maximal for each, and families with no "3-cross" —
   disjoint singletons trivially qualify — are lockable.
   - M. V. Lomonosov, *Combinatorial approaches to multiflow problems*,
     Discrete Appl. Math. **11** (1985) 1–93.
   - Karzanov–Lomonosov, *Systems of flows in undirected networks* (1978).
   - Directed analogue: Ibaraki–Karzanov–Nagamochi (1998); the
     "Lomonosov–Frank theorem". Terminology confirmed via Hirai–Koichi,
     arXiv:1006.5520.
   - Expositions: Schrijver, *Combinatorial Optimization*, Vol. C; A. Frank,
     *Connections in Combinatorial Optimization*.
   - Caveat on this review: the exact 1985 statement was verified from the
     secondary canon, not the primary source (Semantic Scholar rate-limited
     the check) — worth two minutes in Schrijver Vol. C before citing.
   The paper's theorem adds what the flow literature does not have: the
   train-track and homotopy-class formulation, and globalization along maps
   to an arbitrary target graph.

2. **The classical min–max lineage.** Menger (1927); Ford–Fulkerson (1956).
   Adjacent: Hu's two-commodity theorem (1963); Gomory–Hu (1961) — cut-trees
   are the same laminar family of minimal cuts that Lemma `complete-nested`
   completes. The uncrossing arguments (Lemmas `nest-one-end`,
   `nest-two-ends`) are the Edmonds–Giles (1977) submodularity technique.

3. **Prop. `tt-curve`** (balanced ⇒ carried saturating multi-curve) is a
   weighted cycle-decomposition under gate constraints:
   - Seymour, *Sums of circuits* (1979) — the cycle cone (subset version, cut
     triangle inequalities; allowing curves to traverse an edge twice is what
     replaces his cut condition with vertex triangle inequalities);
   - Fleischner, *Eulerian Graphs and Related Topics* (1990–91) — transition
     systems and compatible circuit decompositions (gates generalize his
     pairings to partitions).

## Same toolbox — one-line pointers

4. **"Gates" terminology**: established for train-track maps of graphs in
   Bestvina–Feighn–Handel, *The Tits alternative for $\mathrm{Out}(F_n)$ I*,
   Ann. of Math. **151** (2000) — Def. `tt` should be cross-referenced
   against it (the paper's gates need no BFH train-track-map axioms).
5. Stallings, *Topology of finite graphs* (1983): the "standard fact" in
   Lemma `taut-curve` (unique locally injective representative); the
   reduction operation is folding/pulling tight (also Dunwoody's folding
   sequences).
6. Cooper's bounded cancellation lemma (1987): Prop. `strong-red-finite` is a
   bounded-cancellation-type finiteness.
7. Parametric max-flow (Gallo–Grigoriadis–Tarjan 1989): Prop. `nf-cont-pl`
   (min-cut value piecewise-linear in capacities) is standard there.
8. Francaviglia–Martino *is* cited (in `lipschitz.tex`) — correctly; their
   optimal maps / tension forest are the closest analogue of the
   balanced-train-track witness for Emb.

## Conceptual lineage

9. "Taut" + "minimal in class iff there is a calibrating witness" matches
   Rummler–Sullivan for taut foliations (Rummler 1979; Sullivan, *Cycles for
   the dynamical study of foliated manifolds*, 1976) and Harvey–Lawson
   calibrations (1982): taut foliation ⇔ leaves volume-minimizing ⇔ closed
   transversal form; taut map ⇔ multiplicity-minimizing ⇔ carried saturating
   multi-curve.
10. The triangle inequalities at a vertex say precisely that the gate weights
    admit a vector equilibrium (form a polygon): Maxwell–Cremona stresses;
    Kapovich–Millson (1996).

## What appears to be genuinely new

The globalization: assembling local star-graph models along a homotopy class
of maps to an **arbitrary target graph**, with the four-way equivalence and
the strongly-reduced finiteness powering the piecewise-linear theory. No
flow-literature result was found that moves the duality across a nontrivial
target's topology.

## Audit footnotes

- 12 entries in `Elastic.bbl` are never `\cite`d in the text (GS92,
  Kerckhoff80, Wolf95, Mercat01, Smirnov01, Duffin68, Ferrand44, Isaacs41,
  Lovasz04, Picard05, PP93, Stephenson05, CS11) — probably `.aux` leftovers;
  Gromov–Schoen being in the bibliography but uncited looks accidental given
  the harmonic section.
- For the applet (plan, M3): items 1–3 mean the taut-map computation can lean
  on mature multiflow algorithms and the multiway-cut literature
  (Dahlhaus–Johnson–Papadimitriou–Seymour–Yannakakis 1994), instead of
  implementing the existence proof as an algorithm.
