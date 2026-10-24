#!/usr/bin/env python3
"""Print the critical portrait of a session as a text table
(plan item 120): each face of the cover, its boundary labels, the
target face, and the degree."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pullback.harmonic_io import load_session
from pullback.portrait import face_map_pi
from pullback.ribbons import fmt_dart


def labels(face):
    return " ".join(fmt_dart(d) for d in face)


def _rotations_str(face):
    toks = [fmt_dart(d) for d in face]
    n = len(toks)
    return { "".join(toks[i:] + toks[:i]) for i in range(n) }


def main(path):
    import json
    base, cover, phi = load_session(path)
    sess = json.loads(Path(path).read_text())
    pnames = {}
    for entry in sess.get("punctures", []):
        pnames[entry["face"]] = entry["name"]
    rows, base_walks = face_map_pi(cover, base)

    def name_of(face):
        for r in _rotations_str(face):
            if r in pnames:
                return pnames[r]
        return None

    blab = [labels(f) for f in base_walks]
    names = [name_of(f) for f in base_walks]
    print(f"{Path(path).name}")
    print(f"{'face':28s} {'-> base face':16s} degree")
    for row in rows:
        tgt = row["target"]
        tgt_s = f"[{blab[tgt]}]" if tgt is not None else "-"
        if tgt is not None and names[tgt]:
            tgt_s += f" ({names[tgt]})"
        deg = row["degree"] if row["degree"] is not None else "-"
        print(f"[{labels(row['face']):27s}] {tgt_s:16s} {deg}")


if __name__ == "__main__":
    for p in sys.argv[1:] or [str(ROOT / "sessions" / "theta.json")]:
        main(p)
        print()
