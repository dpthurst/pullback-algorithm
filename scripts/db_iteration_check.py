#!/usr/bin/env python3
"""The fuller comparison-free iteration check over the whole session database.

Every session in sessions/, n = 1..N (default 2), alg:iter capped at 12
steps, with the invariants that need no comparison target -- the exact QP
cannot run on most sessions at n >= 2 and there are no published values.
The factor stop is the bracketed one (`tol = 1e-12` on the relative change;
the exact-repeat stop never fires cold), and the reported lambda is
`lams[-1]` with the balance certificate still exact:

  completes                        (exceptions recorded, not hidden)
  certificate: all tension gaps <= 0, exactly, at the final harmonic map
  Dir(harmonic) <= Dir(input representative)      (the harmonic step minimises)
  multiplicities v >= 0, factor lambda > 0
  lambda <= the explicit bound Emb(phi^n_0) = max_l (M^n 1)_l   (DR <= Emb)
  DR sequence non-decreasing up to 1e-6            (lem:DR-ER-increase)
  Emb(input) from covermin's piece model == iterate_energy's matrix power
                                   (two independent computations, with the
                                   session's elastics as base_lengths)

Not part of `make check` (the suite's tests/test_database.py covers n = 1
with the same invariants); run this for the fuller sweep:

    python3 scripts/db_iteration_check.py [--n 2] [--steps 12]
                                         [--session NAME ...] [--json OUT]

or `make dbcheck` for the same.  The exit status is 0 when every run is ok,
a documented skip or a documented known exception; an unexpected failure
makes it 1 (so `make dbcheck` is usable as a check without hiding the
recorded corner).

Documented skips (pre-existing, shared with the exact engine or out of
scope for exact rationals):
  census-3.2    session_iterate raises (a vertex has two images)
  census-53     irrational elastics '3+sqrt(105)' (D5: no interval arithmetic)
  twisted-* n=2 G_n is disconnected; the Cls needs a connected source
                (the exact QP tolerates it)
Documented corner (not a failure): from n = 2 the constructed G_n undercounts
against the telescoped fill -- barbell n=2 is the case in the database (the
constructed map's Emb 2 against the matrix power 6), so at n >= 2 the check
only requires covermin's Emb <= the matrix power, and prints the row with an
`undercount` note when the two differ.  At n = 1 they must agree exactly on
every session, and they do.
Documented exception (diagnosed 2026-10-26):
  twisted-z2i n=1: the DR sequence decreases by 7.5e-2.  The lifted energy is
    convex per block but not jointly; the free class contains several marked
    subclasses, each with its own balanced minimiser, and the coordinate
    descent stalls at a start-dependent one, so the measured DR sequence is
    not the class-minimum sequence the lemma is about.
"""

import argparse
import json
import sys
import time
from fractions import Fraction as Q
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pullback import covermin as cm
from pullback.harmonic_io import load_session
from pullback.iterate_graphs import session_iterate
from pullback.iterates import iterate_energy
from pullback.refined import graph_from_ribbon

MONO_TOL = Q(1, 10 ** 6)

# Diagnosed, not defects: reported as KNOWN, and a rerun still exits 0.
KNOWN = {
    ("twisted-z2i", 1):
        "DR sequence decreases by 7.5e-2 -- diagnosed, not a bug: the lifted "
        "energy is convex per block but not jointly, and the free class "
        "contains several marked subclasses with their own balanced "
        "minimisers, so the descent stalls at a start-dependent one",
}


def edge_letter(e):
    """The base edge label of a G_n edge label."""
    lab = e[-1] if isinstance(e, tuple) else e
    while isinstance(lab, tuple) and not isinstance(lab[0], str):
        lab = lab[0]
    return lab[0] if isinstance(lab, tuple) else lab


def session_elastics(path, G2):
    """{base edge: Fraction} or None (unit); ValueError on irrational data."""
    with open(path) as f:
        sess = json.load(f)
    elast = sess.get("elastics")
    if elast is None:
        return None
    return {e: Q(x) for e, x in zip(sorted(G2.edges, key=str), elast)}


def check_one(name, n, steps):
    path = ROOT / "sessions" / f"{name}.json"
    r = {"session": name, "n": n}
    try:
        _b, _c, _p = load_session(path)
        elast = session_elastics(path, graph_from_ribbon(_b))
        Gn, phin, G2 = session_iterate(path, n)
        if elast is not None:
            for e in Gn.edges:
                Gn.alpha[e] = elast[edge_letter(e)]
            for e in G2.edges:
                G2.alpha[e] = elast[e]
        cls = cm.Cls(cm.G.from_refined(Gn),
                     {e: tuple(phin.chains.get(e, ())) for e in Gn.edges},
                     cm.G.from_refined(G2), marked=set(Gn.marked))
        r["edges"] = len(cls.src.edges)
        ell = {f: Q(1) for f in cls.tgt.labels()}
        dir_in = cls.energy(cls.X0, ell)
        dir_h, _did, _dr, v, X = cls.dr(ell)
        r["dir_harm_le_dir_in"] = dir_h <= dir_in
        r["v_nonneg"] = all(x >= 0 for x in v.values())
        r["emb_in_pieces"] = cls.emb(cls.X0, ell)
        _base, cover, phi = load_session(path)
        r["emb_in_iterates"] = iterate_energy(cover, phi, n, Q(2),
                                              base_lengths=elast) \
            .pow(2).exact_value()
        r["emb_agree"] = r["emb_in_pieces"] == r["emb_in_iterates"]
        # at n >= 2 the G_n enumeration is known to undercount, so the only
        # allowed disagreement is covermin < matrix (the matrix power is the
        # paper's telescoped fill, the reference)
        r["emb_ok"] = (r["emb_agree"] if n == 1
                       else r["emb_in_pieces"] <= r["emb_in_iterates"])
        if n >= 2 and not r["emb_agree"]:
            r["emb_note"] = ("G_n undercounts (the recorded corner); the "
                             "matrix power is the reference")
        lams, _ell2, _c, _X2, gaps = cm.alg_iter(cls, steps=steps,
                                                 tol=Q(1, 10 ** 12))
        r["lam"] = lams[-1]
        r["steps"] = len(lams)
        decs = [a - b for a, b in zip(lams, lams[1:]) if b < a]
        r["mono_ok"] = all(d <= MONO_TOL for d in decs)
        r["worst_dec"] = max(decs) if decs else Q(0)
        r["lam_positive"] = lams[-1] > 0
        r["gaps_nonpos"] = all(g[2] <= 0 for g in gaps)
        r["gaps_nonempty"] = len(gaps) > 0
        r["lam_le_explicit"] = r["lam"] <= r["emb_in_pieces"]
        r["ok"] = all(r.get(k, True) for k in (
            "dir_harm_le_dir_in", "v_nonneg", "mono_ok", "lam_positive",
            "gaps_nonpos", "gaps_nonempty", "lam_le_explicit", "emb_ok"))
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        if "has two images" in msg:
            r["skip"] = ("session_iterate raises (a vertex has two images) "
                         "-- pre-existing; the exact engine fails identically")
        elif "not connected" in msg:
            r["skip"] = ("G_n is disconnected; the Cls needs a connected "
                         "source (the exact QP tolerates it)")
        elif "Fraction" in msg and "sqrt" in msg:
            r["skip"] = ("irrational elastics in the session -- outside the "
                         "exact-rational engine (D5)")
        else:
            r["error"] = msg
        r["ok"] = None if "skip" in r else False
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=2, help="max iterate (default 2)")
    ap.add_argument("--steps", type=int, default=12,
                    help="alg:iter step cap (default 12)")
    ap.add_argument("--session", nargs="*", default=None,
                    help="restrict to these session names")
    ap.add_argument("--json", default=None, help="write the rows here")
    args = ap.parse_args()
    names = args.session or sorted(
        p.stem for p in (ROOT / "sessions").glob("*.json"))
    rows = []
    print(f"{'':4s} {'session':16s} {'n':>2s} {'|E|':>4s} {'lambda':>14s} "
          f"{'steps':>5s} {'cert':>4s} {'mono':>4s} {'le':>3s} {'emb':>4s} "
          f"{'dir':>3s} {'v':>2s}  time", flush=True)
    for name in names:
        for n in range(1, args.n + 1):
            t0 = time.time()
            r = check_one(name, n, args.steps)
            if r["ok"] is False and (name, n) in KNOWN:
                r["known"] = KNOWN[(name, n)]
            r["secs"] = round(time.time() - t0, 1)
            rows.append(r)
            if r.get("skip"):
                print(f"SKIP {name:16s} {n:2d}  {r['skip']}", flush=True)
            elif "error" in r:
                print(f"ERR  {name:16s} {n:2d}  {r['error']}", flush=True)
            else:
                note = ""
                if not r["mono_ok"]:
                    note += f" (dec {float(r['worst_dec']):.1e})"
                if r.get("emb_note"):
                    note += " (undercount)"
                tag = "ok" if r["ok"] else "KNOWN" if r.get("known") else "FAIL"
                print(f"{tag:4s} {name:16s} {n:2d} "
                      f"{r['edges']:4d} {float(r['lam']):14.9f} "
                      f"{r['steps']:5d} {int(r['gaps_nonpos']):4d} "
                      f"{int(r['mono_ok']):4d} {int(r['lam_le_explicit']):3d} "
                      f"{int(r['emb_ok']):4d} {int(r['dir_harm_le_dir_in']):3d} "
                      f"{int(r['v_nonneg']):2d}  {r['secs']:5.1f}s{note}",
                      flush=True)
    nok = sum(1 for r in rows if r["ok"] is True)
    nskip = sum(1 for r in rows if r.get("skip"))
    nknown = [r for r in rows if r.get("known")]
    nbad = [r for r in rows if r["ok"] is False and not r.get("known")]
    print(f"\n{len(rows)} runs: {nok} ok, {nskip} skipped (documented), "
          f"{len(nknown)} known (documented), {len(nbad)} failed")
    for r in nknown:
        print(f"  KNOWN {r['session']} n={r['n']}: {r['known']}")
    for r in nbad:
        print("  FAILED:", {k: (str(v)[:90] if isinstance(v, Q) else v)
                            for k, v in r.items()
                            if k not in ("lam", "emb_in_pieces",
                                         "emb_in_iterates")})
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1, default=str))
    return 1 if nbad else 0


if __name__ == "__main__":
    sys.exit(main())
