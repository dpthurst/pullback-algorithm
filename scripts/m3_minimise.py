#!/usr/bin/env python3
"""The M3 table and iterate figure (plan items 180/190/210).

Per session and iterate n: the explicit-representative bound (what
scripts/m3_energies.py reports), the minimised Emb[phi^n_0] with its
certified bracket [LB, UB], and the dual lower bound.  Static figure to
results/ (never committed), table to stdout.  Deterministic: nothing depends
on set or dict iteration order.

    python3 scripts/m3_minimise.py [--n 3] [--sessions bouquet barbell] [--figure results/m3.png]

Scope: the minimiser (alg:iter) is exact but enumerates the simple marked
curves of the source up front, which is complete for the library's graphs
at n <= 3 (every edge crossed once); beyond that the curve enumeration
explodes and the dual search is the practical bound.  Both are reported.
"""

import argparse
import json
import sys
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pullback.refined import session_map
from pullback.minimise import Minimiser, minimise
from pullback.certificates import lower_bound
from pullback.iterate_graphs import session_iterate
from pullback.iterates import iterate_energy
from pullback.harmonic_io import load_session

ALL_SESSIONS = sorted(p.stem for p in (ROOT / "sessions").glob("*.json"))
# sessions whose minimiser is tractable at n <= 3 (curve enumeration)
MINIMISE_SESSIONS = {"bouquet", "barbell"}
DUAL_SEARCH = [{"max_darts": 6, "top_k": 6, "max_parts": 4, "max_weight": 6},
               {"max_darts": 8, "top_k": 6, "max_parts": 3, "max_weight": 5}]


def fmt(x):
    if x is None:
        return "--"
    if isinstance(x, F):
        return f"{x} = {float(x):.6f}"
    return str(x)


def best_lower(phin):
    best = None
    for kw in DUAL_SEARCH:
        try:
            low = lower_bound(phin, **kw)
        except Exception:
            continue
        if low is not None and (best is None or low[0] > best[0]):
            best = low
    return best


def session_rows(name, n_max):
    rows = []
    path = ROOT / "sessions" / f"{name}.json"
    base, cover, phi = load_session(path)
    for n in range(1, n_max + 1):
        e = iterate_energy(cover, phi, n, F(2))
        explicit = e.pow(2).exact_value()
        row = {"session": name, "n": n, "explicit": explicit}
        try:
            Gn, phin, G2 = session_iterate(path, n)
            Gn.marked = frozenset(Gn.deg)   # curve family needs marked points
            row["built_emb"] = phin.emb()
            low = best_lower(phin)
            row["dual"] = low[0] if low else None
            if name in MINIMISE_SESSIONS:
                C = Minimiser(Gn, G2, phin.chains)
                res = minimise(C)
                row["min_LB"] = res["LB"]
                row["min_fixed"] = [(lam, all(v > 0 for v in l.values()))
                                    for lam, l, h in res["fixed_points"]]
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--sessions", nargs="*", default=ALL_SESSIONS)
    ap.add_argument("--figure", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows = []
    for name in args.sessions:
        rows.extend(session_rows(name, args.n))

    print(f"{'session':16s} {'n':>2s} {'explicit':>22s} {'dual':>22s} "
          f"{'min LB':>22s}  bracket")
    print("-" * 100)
    for r in rows:
        lb = r.get("min_LB", r.get("dual"))
        ub = r["explicit"]
        if lb is not None and lb == ub:
            br = f"= {fmt(lb)}"
        elif lb is not None:
            br = f"[{fmt(lb)}, {fmt(ub)}]"
        else:
            br = f"(<= {fmt(ub)})"
        print(f"{r['session']:16s} {r['n']:2d} {fmt(r['explicit']):>22s} "
              f"{fmt(r.get('dual')):>22s} {fmt(lb):>22s}  {br}"
              + (f"   [{r['error']}]" if "error" in r else ""))

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2, default=str) + "\n")

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs, ys, lo, hi, labels = [], [], [], [], []
        for r in rows:
            if r.get("min_LB") is None and r.get("dual") is None:
                continue
            lb = r.get("min_LB", r.get("dual"))
            xs.append(r["n"])
            ys.append(float(r["explicit"]))
            lo.append(float(lb))
            hi.append(float(r["explicit"]))
            labels.append(r["session"])
        fig, ax = plt.subplots(figsize=(7, 5))
        for name in dict.fromkeys(labels):
            idx = [i for i, l in enumerate(labels) if l == name]
            ax.plot([xs[i] for i in idx], [ys[i] for i in idx], "o--",
                    label=f"{name} explicit")
            ax.plot([xs[i] for i in idx], [lo[i] for i in idx], "s-",
                    label=f"{name} certified LB")
        ax.set_xlabel("n")
        ax.set_ylabel("Emb[phi^n]^(1/n) bracket endpoints")
        ax.set_title("M3 iterate brackets (plan 190/210)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=110)
        print(f"\nwrote {args.figure}")


if __name__ == "__main__":
    main()
