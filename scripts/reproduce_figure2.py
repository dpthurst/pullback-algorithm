#!/usr/bin/env python3
"""Reproduce Figure 2 of [Rubber] ("From Rubber Bands to Rational Maps: a
research report", arXiv:1502.02561): the log-plot of SF[phi_n] for
f(z) = (1+z^2)/(1-z^2) as n varies, for four elastic parameter sets, with
the asymptotic line 2^{-n/3}.

The paper's figure is `refs/rubber/figs/logplot4.pdf`; its plotted data
were read out of the PDF's vector paths (the axes are x = 26.2 +
44.8329872267*n and y = 160.306173207 + 32.3585745517*log2(SF), from the
tick marks) and agree digit for digit with the author's table in
`.scratch/Harmonic/output.tex`, whose columns are SF[f^(n)]^{1/n}: the
figure plots the n-th power of those entries.

Which curves this engine can recompute:
  (1,1,0)  the middle edge degenerates to a rose = the bouquet session:
           Emb[f^n] = 1, 3/4, 5/8, 23/48 exactly at n = 1..4 (the minimiser
           of plan item 170); n = 5 is the published 187/480 (out of reach
           of the minimiser, the dual bound there is 17/55).
  (1,1,1)  the theta session with unit elastics: Emb = 3/2 exactly at n = 1
           and 6/7 exactly at n = 3 (interior fixed points); n = 2 is
           irrational, and the DR sequence converges to 0.773459.
  (4,4,1)  the theta session with (middle, left, right) = (1,4,4): the
           fixed point is irrational at n = 1..3, and the DR sequence
           converges to the paper's 0.864357, 0.704124, 0.563299.
  (1,1,-1) the barbell obtained by a NEGATIVE middle edge parameter: not
           reproducible here.  It is not the repo's `barbell` session (that
           is Harmonic's different Barbell example, whose Emb[phi^1] is 5),
           and the negative-length construct is not implemented.

Nothing is committed: the figure goes to results/ (git-excluded).
"""

import argparse
import sys
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# The paper's plotted data (logplot4.pdf, via the vector paths; equal to
# Harmonic output.tex's SF[f^(n)]^{1/n} raised to the n).
PAPER = {
    "(4,4,1)": [1.0, 0.864357, 0.704124, 0.563299, 0.448620, 0.356328,
                0.282990, 0.224600, 0.178296],
    "(1,1,1)": [1.0, 1.500000, 0.773459, 0.857143, 0.526978, 0.508475,
                0.350397, 0.309249, 0.227530],
    "(1,1,0)": [1.0, 1.000000, 0.750000, 0.625000, 0.479167, 0.389583,
                0.303487, 0.243990, 0.191761],
    "(1,1,-1)": [1.0, 3.000000, 0.864357, 1.615385, 0.678261, 0.904156,
                 0.495179, 0.525777, 0.339539],
}

# What this engine computes (n, SF) from plan-170 minimiser runs.
ENGINE_EXACT = {
    "(1,1,0)": {1: F(1), 2: F(3, 4), 3: F(5, 8), 4: F(23, 48)},
    "(1,1,1)": {1: F(3, 2), 3: F(6, 7)},
}
# irrational fixed points: DR after 14 iterations of alg:iter (a certified
# lower bound; it converges to the paper's value from below)
ENGINE_LIMIT = {
    "(4,4,1)": {1: 0.864356777, 2: 0.704124145, 3: 0.563299316},
    "(1,1,1)": {2: 0.773459080},
}


def compute_curves(n_max):
    """Run the fast (position-based) minimiser on the theta session for the
    two parameter sets of the figure, n = 1..n_max."""
    from pullback.iterate_graphs import session_iterate
    from pullback.fastmin import alg_iter

    def edge_letter(e):
        lab = e[-1] if isinstance(e, tuple) else e
        while isinstance(lab, tuple) and len(lab) >= 1 and not isinstance(lab[0], str):
            lab = lab[0]
        return lab[0] if isinstance(lab, tuple) else lab

    out = {}
    for label, a in (("(4,4,1)", {"p": 4.0, "q": 1.0, "r": 4.0}),
                     ("(1,1,1)", {"p": 1.0, "q": 1.0, "r": 1.0})):
        vals = {}
        for n in range(1, n_max + 1):
            Gn, phin, G2 = session_iterate(ROOT / "sessions" / "theta.json", n)
            alpha1 = {e: a[edge_letter(e)] for e in Gn.edges}
            lams, _ell2, _m = alg_iter(Gn, G2, phin.chains, alpha1, dict(a))
            vals[n] = lams[-1]
            print(f"  computed {label} n={n}: {lams[-1]:.9f}", flush=True)
        out[label] = vals
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "figure2_rubber.png"))
    ap.add_argument("--compute", type=int, default=0, metavar="N",
                    help="also compute the theta curves with the fast minimiser "
                         "up to n=N (0 = only the paper's data)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ns = list(range(9))
    colours = {"(4,4,1)": "#7f1109", "(1,1,1)": "#b8860b",
               "(1,1,0)": "#1a7f37", "(1,1,-1)": "#1f4ea8"}
    for label, ys in PAPER.items():
        ax.plot(ns, ys, "-", color=colours[label], lw=1.4,
                label=f"$\\alpha,\\beta,\\gamma$ = {label}")
    ax.plot(ns, [2 ** (-n / 3) for n in ns], "--", color="black", lw=1.1,
            label="limit $2^{-n/3}$")

    # engine values: exact (filled circles) and one-sided limits (triangles)
    for label, d in ENGINE_EXACT.items():
        xs = sorted(d)
        ax.plot(xs, [float(d[n]) for n in xs], "o", ms=6, mfc="none",
                mec=colours[label], mew=1.6, label=f"engine, exact {label}")
    for label, d in ENGINE_LIMIT.items():
        xs = sorted(d)
        ax.plot(xs, [d[n] for n in xs], "^", ms=6, mfc="none",
                mec=colours[label], mew=1.4, label=f"engine, DR limit {label}")

    if args.compute:
        computed = compute_curves(args.compute)
        for label, vals in computed.items():
            xs = sorted(vals)
            ax.plot(xs, [vals[n] for n in xs], "x", ms=7, mew=1.6,
                    color=colours[label], label=f"engine (fast) {label}")

    ax.set_yscale("log", base=2)
    ax.set_yticks([4, 2, 1, 1 / 2, 1 / 4, 1 / 8])
    ax.set_yticklabels(["4", "2", "1", "1/2", "1/4", "1/8"])
    ax.set_ylim(1 / 8, 4)
    ax.set_xlim(0, 8)
    ax.set_xticks(ns)
    ax.set_xlabel("$n$")
    ax.set_ylabel("SF")
    ax.set_title("Figure 2 of [Rubber], regenerated")
    ax.legend(fontsize=6.5, loc="lower left", ncol=1, framealpha=0.9)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
