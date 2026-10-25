# Contraction, and the three groups

*Background note, 2026-10-25. Written down from the discussion with the author so that it
does not have to be reconstructed. Statements are marked where they are his (and where his
source is informal), where they are measured in this engine, and where they are from the
literature. The working notes are in `AGENTS.md` §A0; this file is the readable version.*

## 1. Which group?

A graph virtual endomorphism $(\pi,\phi)$ carries three nested groups, and "is the wreath
recursion contracting" is three different questions:

1. **The graph $\pi_1$**: $\pi_1(\Gamma_0)$, free of rank $|E_0|-|V_0|+1$. The automaton's
   letters are its free generators, so a reduced word *is* an element.
2. **The orbifold $\pi_1^{\mathrm{orb}}$**: $\pi_1(\Gamma_0)$ modulo the relations
   $[\text{small loop around }x]^{e_x}=1$ at each $x\in P$, where $e_x$ is the local degree
   (the cone order). **This only makes sense when there is a ribbon structure** (author,
   2026-10-25): the cone orders are read off the ribbon data — a puncture is a face, and its
   order is the degree of the map on faces, i.e. a ratio of boundary-cycle lengths. With no
   ribbon structure there are no faces and no orders, so the orbifold group is not defined;
   the planar/cyclic-order structure is part of the input, not a decoration.
3. **The IMG**: the image of $\pi_1$ in $\mathrm{Aut}(T)$, i.e. the faithful quotient that
   actually acts on the tree. It is a quotient of both of the above.

Contraction is a property of the *action*, hence of the IMG. The graph $\pi_1$ can act
non-faithfully, and its kernel is exactly where the three answers part company.

## 2. The recursion and the nucleus

The wreath recursion $g = \sigma_g\,(g|_x)_{x\in X}$ makes each group a self-similar group.
The **nucleus** is the set of sections $g|_v$ of all elements $g$ at all words $v$; the group
is **contracting** when the nucleus is finite.

Write $N_{\pi_1}\supseteq N_{\mathrm{orb}}\supseteq N_{\mathrm{IMG}}$ for the three nuclei
(the last is the image of the others). Since a section of a quotient is the quotient of a
section, $N_{\mathrm{IMG}}$ is the image of $N_{\pi_1^{\mathrm{orb}}}$, and likewise for the
free group. Hence

$$\pi_1\text{ contracting}\;\Longrightarrow\;\pi_1^{\mathrm{orb}}\text{ contracting}
\;\Longrightarrow\;\text{IMG contracting},$$

and **not** conversely. The obstruction is the kernel: if
$\ker(\pi_1^{\mathrm{orb}}\to\mathrm{IMG})$ is finite then each section lies in a finite union
of cosets of it, so contraction descends; an infinite kernel need not, and then the IMG can
contract while $\pi_1$ does not. The author believes (2026-10-25, from an informal
conversation with Nekrashevych; his memory may be unreliable) that the conditions under which
IMG contraction implies contraction of $\pi_1^{\mathrm{orb}}$ are **unknown**. Nekrashevych's
own use of contraction is mostly on the IMG, and contraction often fails on $\pi_1$ alone.

**Measured instance** (the engine, 2026-10-25). The session `twisted-z2i` — the string
Harmonic wrote for the "Hamal birthday conference" example — has

| closure deduped by | size | closed |
|---|---|---|
| reduced words ($=\pi_1$ elements) | 106 | **no** |
| action on level 2 | 7 | yes |
| action on level 3 | 17 | yes |
| action on level 4 … 8 | 23 | yes |

So the image contracts and $\pi_1$ does not: the kernel of $\pi_1\to\mathrm{IMG}$ has
infinitely many sections. It is the library's only machine where the two answers differ; the
other thirteen sessions close at 5–9 words.

## 3. What the engine computes

* `Automaton.nucleus()` — dedup on reduced words. Since the letters are free, this is the
  **$\pi_1$-nucleus** exactly, and it is a semi-decision: if it halts, $\pi_1$ contracts; if
  the nucleus is infinite the enumeration simply runs.
* `Automaton.nucleus(level=n)` — dedup on the action on $X^n$, i.e. in the image of $\pi_1$
  in $\mathrm{Sym}(X^n)$. That universe is finite, so this always terminates; the informative
  number is the class count, and whether it stabilises as $n$ grows. That is the IMG question
  up to the level-$n$ kernel; no a-priori separating level is known.
* `Automaton.is_contracting(level)` takes the same argument.

Nothing in the engine computes $\pi_1^{\mathrm{orb}}$ yet. The data for it are present — the
type classifier already computes the cone orders $e_x$ — but the quotient has to be carried
through the wreath recursion, and that is a later iteration.

## 4. Decidability

For a self-similar group given by a finite Mealy automaton:

* **Semi-decidable.** Elements of an automaton group can be compared: a finite-state tree
  automorphism is the identity iff every state reachable from it is inert (identity
  permutation, all sections inert), a finite reachability check. So the nucleus can be
  enumerated with exact dedup, and the enumeration halts iff the group is contracting. The
  same holds for $\pi_1$ (free reduction is the dedup) and will hold for
  $\pi_1^{\mathrm{orb}}$ once a normal form is implemented.
* **Decidability is open** in general. I could find no decidability theorem and no
  undecidability theorem. Bartholdi's FR documentation for `NucleusOfFRSemigroup` says:
  "This function returns `fail` if no such N exists. Usually, it loops forever without being
  able to decide whether N is finite or infinite." FR also has a partial negative test: a
  state that recurs at a fixed vertex and is provably of infinite order forces the nucleus to
  be infinite.
* **Decidable sufficient classes.** Bounded automata (finitely many "active" states along any
  path) are always contracting, and boundedness of a Mealy automaton is decidable;
  Bondarenko–Nekrashevych identify bounded $\Leftrightarrow$ post-critically finite limit
  space. Finitary automata likewise.
* The algorithmic literature on automaton groups is otherwise full of undecidability
  (finiteness of automaton semigroups, order and Engel problems, conjugacy), but I found no
  reduction to contraction.

## 5. Which group should the app ask about

The type classification decides:

* **hyperbolic type** (every cycle of $P$ contains a branch point): ask about the graph
  $\pi_1$; the usual story applies.
* **cycle without branch point** (e.g. $z^2+i$): the recursion is *not* contracting on
  $\pi_1$ and is not meant to be; the right object is $\pi_1^{\mathrm{orb}}$.
* The IMG is the safest invariant — it is the group that acts — but it is also the hardest.

**Decision for M2** (author, 2026-10-25): work with the graph $\pi_1$ for now — much simpler —
and generalise to $\pi_1^{\mathrm{orb}}$ and the IMG later. So the M2 contraction gate is
`nucleus()` on $\pi_1$; the orbifold quotient waits until the cone orders from the type
classification are carried through the recursion.

**And the IMG itself is out of scope** (author, 2026-10-25): computing it is a challenging task
and it is not necessary for the main applet. So the ``safest invariant'' above is a mathematical
remark, not a deliverable: the app reports the $\pi_1$ answer and, where it helps, the level-$n$
class counts, and does not build $\pi_1 \to \mathrm{IMG}$ as a quotient, its word problem or its
nucleus.
The honest report is three-valued and names its group: *contracting* (with the closure size
and the level at which it was certified), *not contracting* (with a certificate, e.g. a
self-recurrent infinite-order state), or *undecided*.

## 6. The twisted rabbit and the twisted $z^2+i$

The author confirmed (2026-10-25) that

```
graph "xaAbBcC"   cover "a01"   phi "b0Bab c0BAbab a0c"
```

is the actual twisted rabbit: the rabbit's $b_0,c_0$ images conjugated by $ab$, with the
rabbit's own third token. It is hyperbolic, level orders $2,8,128,16384,2^{28},2^{56}$,
nucleus 21, closed.

The string Harmonic actually wrote, `b0Bab c0BAbab b1c` (now the session `twisted-z2i`),
differs in that third token and drops $a_0\mapsto c$ altogether. The author believes it
describes a twist of $z^2+i$ — consistent with the combinatorics: on this spine the untwisted
map with the same first two tokens, `b0a c0b b1c`, has type "cycle without branch point" like
$z^2+i$, while the rabbit's `a0c` variant is hyperbolic. It is not verified as $z^2+i$.

Whether it is **obstructed** is deliberately left open (author, 2026-10-25). In principle it
can be read off from Bartholdi–Nekrashevych, *Thurston equivalence of topological polynomials*,
or Belk–Lanier–Margalit–Winarski, *Recognizing topological polynomials by lifting trees*; the
author prefers to let the app's obstruction-matrix machinery (plan item 200) settle it. The
orbifold is hyperbolic ($\chi_{\mathrm{orb}}=-1$: four postcritical points, cone orders
$2,2,1,1$), so the ordinary criterion applies, and on a 4-punctured sphere only the three
$2{+}2$ multi-curves need checking.

## 7. Word growth (deferred)

Not needed by anything else in the pipeline, and deferred (author, 2026-10-25). On the graph
$\pi_1$ it is trivial — free of rank $g$, so $2g(2g-1)^{n-1}$ elements of length $n$ — and the
interesting object is the IMG, which the engine does not build as a group (only its level
images). The author's reasons for flagging it anyway: it is related to the $p=\infty$ energy
"in some way I have not personally fully understood"; the Grigorchuk group has intermediate
growth and its exact growth rate is not known (Bartholdi, 2000; later work not checked).

The $1/2$ he recalled is the **asymptotic** energy
$\hat E^\infty=\lim (E^\infty_\infty(\phi^n))^{1/n}$, not $\mathrm{Lip}$ on an explicit
representative: the asymptotic energy is independent of the starting weights, while
$\mathrm{Lip}(\phi)$ is not. For the Grigorchuk example it comes from a suitably *orbifolded*
graph, i.e. it is a statement about $\pi_1^{\mathrm{orb}}$, and is deferred with the rest of
the orbifold. For comparison, the explicit-representative values in the library are theta 1,
julia-4422 1, julia-4442 2, census-53 $\approx 2.906$ (grigorchuk has no weights) — the wrong
object for this question. The engine cannot compute $\hat E^\infty$ yet: $E^p_\infty$ of an
iterate is not implemented (it needs the edge decomposition, not just the filling). Author's
expectation: for examples with an unbranched cycle, $\hat E^\infty$ will usually be 1.

*How to compute it when it is built* (agent's note, 2026-10-25): the dilation telescopes along
orbits, so $\mathrm{Lip}(\phi^n)$ is a max over length-$n$ paths in a finite graph, and
$\hat E^\infty$ is the **max-plus eigenvalue** of the dilation matrix — the maximum over cycles
of the geometric mean of $\delta$, by Karp's algorithm; no Perron–Frobenius. That is why the
$p=\infty$ energy is the easy one, and it explains the unbranched-cycle expectation: a
boundary loop carried along the cycle has dilation 1, so the maximum cycle mean is 1.

## 8. References

* V. Nekrashevych, *Self-similar groups*, Math. Surveys Monogr. 117, AMS (2005) — contraction,
  the nucleus, the IMG.
* L. Bartholdi, FR (GAP package), documentation of `IsContracting`,
  `NucleusOfFRSemigroup`, `NucleusMachine`.
* I. Bondarenko, V. Nekrashevych, *Post-critically finite self-similar groups*,
  arXiv:math/0701824 — bounded $\Leftrightarrow$ PCF, bounded $\Rightarrow$ contracting.
* L. Bartholdi, M. Noce, *Tree languages and branched groups*, arXiv:2203.12963 — contraction
  carried as a promise/oracle, which is where the decidability gap shows.
* L. Bartholdi, V. Nekrashevych, *Thurston equivalence of topological polynomials*.
* J. Belk, J. Lanier, D. Margalit, M. Winarski, *Recognizing topological polynomials by
  lifting trees*.
