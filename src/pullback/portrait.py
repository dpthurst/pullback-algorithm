"""The critical-portrait read-off of [PChar] §2.

Two computations live here:

* `face_map_pi` — the portrait proper: each face of the cover maps to
  a face of the base via the *covering* pi (regions map to regions
  with degree = ratio of boundary lengths).  This is the read-off the
  paper performs in the text ("surrounded by an a edge and a b edge").
  It uses only the cover data.

* `face_map_phi` — the boundary walks pushed forward by phi and
  tightened.  For a *ribbon* phi this agrees with pi; in general it is
  the raw material for the ribbon-map check of [PChar, Def. 2.7 /
  Rem. 2.10] (plan item 110).  A face whose phi-image tightens to
  nothing maps to a disk without puncture (it is *collapsed*), which
  is allowed — e.g. the face over a non-marked preimage point such as
  0 in the theta example.
"""

from __future__ import annotations

from .ribbons import rev, tighten
from .covers import cover_map_down




def _rotations(walk):
    n = len(walk)
    for i in range(n):
        yield walk[i:] + walk[:i]


def face_map_pi(cover, target, base_faces=None, cover_faces=None):
    """Portrait via the covering: project each cover face's boundary to
    the base labels; it must be a power of a base face boundary (the
    degree).  Returns (rows, base_walks); rows carry target=None if the
    projection is not a face power (a data/convention error, since pi
    is a genuine cover)."""
    if base_faces is None:
        base_faces = target.faces()
    if cover_faces is None:
        cover_faces = cover.faces()
    base_walks = [tuple(f) for f in base_faces]

    out = []
    for face in cover_faces:
        proj = tuple(cover_map_down(d) for d in face)
        hit = None
        for i, bw in enumerate(base_walks):
            k, rem = divmod(len(proj), len(bw))
            if rem:
                continue
            if any(proj == r * k for r in _rotations(bw)):
                hit = (i, k)
                break
        out.append({
            "face": face,
            "projection": proj,
            "target": hit[0] if hit else None,
            "degree": hit[1] if hit else None,
        })
    return out, base_walks


def face_map_phi(phi, base_faces=None, cover_faces=None):
    """For each face of the cover: the raw image walk, its tightening,
    and (target face, degree) — or None if the image is null-homotopic
    (degenerate)."""
    source, target = phi.source, phi.target
    if base_faces is None:
        base_faces = target.faces()
    if cover_faces is None:
        cover_faces = source.faces()
    base_walks = [tuple(f) for f in base_faces]
    out = []
    for face in cover_faces:
        raw = []
        for d in face:
            raw.extend(phi.image_dart(d))
        tight = tighten(raw)
        hit = None
        if tight:
            for i, bw in enumerate(base_walks):
                k, rem = divmod(len(tight), len(bw))
                if rem:
                    continue
                if any(tuple(tight) == r * k for r in _rotations(bw)):
                    hit = (i, k)
                    break
        out.append({
            "face": face,
            "raw": tuple(raw),
            "tight": tight,
            "target": hit[0] if hit else None,
            "degree": hit[1] if hit else None,
        })
    return out, base_walks
