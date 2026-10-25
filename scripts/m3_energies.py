"""M3 energies for p != 2: the lifting bound on ^E E^p_p over the library.

For each session and each p, computes E^p_p(phi^n_0)^(1/n) for n = 1..N
after tuning the q-conformal structure on Gamma_0 (bounds.upper_bound),
and reports the sharpest.  Every number is an *upper* bound on the
asymptotic energy (plan item 180) and is exact for integral p (plan D5);
the homotopy-class minimisation of [Elast] §6 is not applied, so the
bounds are as good as the session's explicit representative allows.
A bound of exactly 1 means the representative is unit-speed for the best
metric — no certificate, not a failure (plan item 190 caveat).

Usage:  python3 scripts/m3_energies.py [--n 3] [--p 1.5,2,3,4] [--session NAME]
"""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pullback.bounds import upper_bound  # noqa: E402
from pullback.harmonic_io import load_session  # noqa: E402


def parse_ps(text):
    out = []
    for tok in text.split(","):
        out.append(Fraction(tok) if "/" in tok else Fraction(str(tok)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--p", default="1.5,2,3,4")
    ap.add_argument("--session", default=None)
    args = ap.parse_args(argv)

    ps = parse_ps(args.p)
    paths = sorted((ROOT / "sessions").glob("*.json"))
    if args.session:
        paths = [p for p in paths if p.stem == args.session]

    print(f"| session | p | n | E^p_p(phi^n)^(1/n) | E^p_p (exact) |")
    print(f"|---|---|---|---|---|")
    for path in paths:
        try:
            base, cover, phi = load_session(path)
        except Exception as exc:  # a session the energies cannot read
            print(f"| {path.stem} | - | - | unreadable: {exc} | - |")
            continue
        for p in ps:
            try:
                rows = upper_bound(cover, phi, p, args.n)
            except Exception as exc:
                print(f"| {path.stem} | {p} | - | error: {exc} | - |")
                continue
            n, e, bound, beta = min(rows, key=lambda r: r[2].value)
            exact = e.exact_value()
            shown = f"{float(bound.value):.6f}"
            if exact is not None:
                shown += f" = {exact}"
            print(f"| {path.stem} | {p} | {n} | {shown} | "
                  f"{e.base}^{e.exponent} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
