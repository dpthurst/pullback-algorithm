#!/usr/bin/env python3
"""Cross-check the engine's automata against Bartholdi's FR (plan item 150).

Emits the wreath recursion of every session as FRGroup definition strings and,
when a GAP with FR is reachable, runs FR on them and compares:

  * the order of the image on each level X^n  (engine: level_generators),
  * the size of the nucleus                  (engine: Automaton.nucleus).

The engine's nucleus is the closure of the generators and their inverses under
sections, counted as reduced WORDS.  FR's (NucleusOfFRSemigroup) is the set of
*limit states of every element of the group*, which is closed under sections and
contains states that are products, e.g. a^-1*b for the rabbit; our set is always
a subset of it.  The level orders are the sharp comparison: they are the same
number for the same data or they are not.

Usage:
    python3 scripts/img_crosscheck.py                # emit defs only
    python3 scripts/img_crosscheck.py --gap .scratch/gap.sh
    python3 scripts/img_crosscheck.py --gap gap --sessions theta rabbit

GAP needs the FR package; on this machine that means the project's private
IO stub and the wrapper, see AGENTS.md.  Nothing here patches FR.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sympy.combinatorics import Permutation, PermutationGroup

from pullback.harmonic_io import load_session
from pullback.normalform import normal_form

DEFAULT_LEVELS = {"theta": 7, "rabbit": 6}


def engine_orders(aut, n_max):
    out = []
    for n in range(1, n_max + 1):
        gens = [Permutation([p[i] for i in range(len(p))])
                for p in aut.level_generators(n).values()]
        out.append(PermutationGroup(gens).order())
    return out


def fr_definitions(aut):
    """FRGroup(...) definition strings: name=<sections>[sigma].

    ORDER OF MULTIPLICATION INSIDE A SECTION WORD — read this before comparing
    with FR.  FR acts on the right: its manual (chapter 2, "the Wreath
    recursion") defines the action by  (xv)a = x^{pi_a} . (v)a_x,  and says the
    automorphism "is defined by acting first on each of the subtrees ... and
    then permuting these subtrees".  So for FR, w.(ab) = (w.a).b, i.e. its
    representation is an anti-homomorphism: rho_FR(g) = rho_ours(g^-1).  One
    sees the same thing in FR's own basilica example, written a=<1,b>(1,2),
    b=<1,a> with decomposition [[identity, b], [2,1]] -- the mirror of the
    usual left-action presentation a = sigma(b,1), b = (a,1).
    Our engine is a left action: rho(gh) = rho(g) o rho(h) (last letter first).
    Hence a section word must be REVERSED when handed to FR.  Single-letter
    sections are unaffected; a two-letter one is mirrored, and the mirrored
    machine is a different action: theta {p} has r|_1 = q*r^-1, and the
    unreversed string gives FR level-7 order 33554432 where ours is 67108864,
    with level-4 permutations equal on eight words and paired-transposed on the
    other eight (pinned down 2026-10-24).  Reversing that one section in our
    engine reproduces FR's permutation entry for entry, which is what
    identifies the cause as the convention rather than an error.
    """
    names = aut.letters
    defs, served = [], []
    for g in names:
        secs = []
        for sec in aut.sections[g]:
            if not sec:
                secs.append("1")
                continue
            letters = [names[j] + ("^-1" if s < 0 else "") for j, s in sec]
            if len(letters) > 1:
                served.append((g, tuple(letters)))
                letters = list(reversed(letters))     # FR's order
            secs.append("*".join(letters))
        sigma = "[" + ",".join(str(s + 1) for s in aut.sigma[g]) + "]"
        defs.append(f'{g}=<{",".join(secs)}>{sigma}')
    if served:
        print("note: multi-letter sections reversed for FR's convention:")
        for g, word in served:
            print(f"   {g}: {'*'.join(word)} -> {'*'.join(reversed(word))}")
    return defs


def build(sessions):
    """Return [(name, defs, orders, nucleus_words)] for each session."""
    out = []
    for name in sessions:
        base, cover, phi = load_session(ROOT / "sessions" / f"{name}.json")
        aut = normal_form(base, cover, phi)
        levels = DEFAULT_LEVELS.get(name, 4)
        out.append((name, fr_definitions(aut),
                    engine_orders(aut, levels), len(aut.nucleus()[0])))
    return out


def gap_script(built):
    lines = ['LoadPackage("fr");;']
    for name, defs, orders, nuc in built:
        args = ", ".join(f'"{d}"' for d in defs)
        lines.append(f'G_{name} := FRGroup({args});;')
        lines.append(f"exp_{name} := {orders};;")
        lines.append(f"nuc_{name} := {nuc};;")
    lines.append("""
Check := function(name, G, exp, nuc)
  local n, o, s, bad;
  bad := 0;
  Print("== ", name, " ==\\n");
  Print("nucleus: FR ", Length(Nucleus(G)), " engine words ", nuc, "\\n");
  for n in [1..Length(exp)] do
    o := Size(Image(EpimorphismPermGroup(G, n)));
    s := "MISMATCH";
    if o = exp[n] then s := "MATCH"; else bad := bad + 1; fi;
    Print("level ", n, ": FR ", o, " engine ", exp[n], " ", s, "\\n");
  od;
  return bad;
end;
""")
    lines.append("bad := 0;")
    for name, *_ in built:
        lines.append(f'bad := bad + Check("{name}", G_{name}, exp_{name}, nuc_{name});')
    lines.append('Print("total mismatches: ", bad, "\\n");')
    lines.append("QUIT;")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", help="GAP command (e.g. .scratch/gap.sh); "
                                  "without it the script is only emitted")
    ap.add_argument("--sessions", nargs="+", default=list(DEFAULT_LEVELS))
    ap.add_argument("--out", default=str(ROOT / ".scratch" / "img_crosscheck.g"))
    args = ap.parse_args()

    built = build(args.sessions)
    script = gap_script(built)
    Path(args.out).write_text(script)
    print(f"wrote {args.out}")
    for name, defs, orders, nuc in built:
        print(f"  {name}: {'  '.join(defs)}")
        print(f"    engine orders {orders}, nucleus {nuc} words")

    if not args.gap:
        print("\nno --gap given: definitions emitted only")
        return 0
    print(f"\nrunning: {args.gap} -q {args.out}")
    proc = subprocess.run([args.gap, "-q", args.out], capture_output=True,
                          text=True, timeout=1800)
    keep = [ln for ln in (proc.stdout + proc.stderr).splitlines()
            if ln.startswith(("==", "nucleus", "level", "total"))]
    print("\n".join(keep))
    return 1 if "MISMATCH" in "\n".join(keep) else 0


if __name__ == "__main__":
    raise SystemExit(main())
