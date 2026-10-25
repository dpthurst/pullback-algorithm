"""View models: what each panel shows, as plain data.

Deliberately free of Qt (and of any GUI toolkit): these run with no display,
so the text and tables the app presents can be tested directly, and the
desktop window (`main.py`) only puts their return values into widgets.  Every
number here is a `Session` method's return value -- the view does no
arithmetic.
"""

from __future__ import annotations

from ..ribbons import fmt_dart


def _face_labels(face):
    return " ".join(fmt_dart(d) for d in face)


def summary(session):
    """The identity and validation block: what data this is, and does it
    pass the plan-100 checks."""
    base, cover = session.base, session.cover
    lines = [
        f"session      {session.name}",
        f"path         {session.path or '(in memory)'}",
        f"base         {session.raw.get('graph', '?')}   "
        f"({len(base.vertices())} vertices, {len(base.edge_list())} edges)",
        f"cover        {session.raw.get('cover', '?')}   "
        f"(degree {session.degree}: {len(cover.vertices())} vertices, "
        f"{len(cover.edge_list())} edges)",
        f"phi          {session.raw.get('phi', '?')}",
    ]
    try:
        v = session.validation()
        lines.append("validation   " + "  ".join(f"{k}={v[k]}" for k in
                                                 ("cover_connected", "base_connected",
                                                  "pi1_surjective")))
        lines.append(f"             {'all checks pass' if all(v.values()) else 'CHECKS FAIL'}")
    except Exception as exc:                                # noqa: BLE001
        lines.append(f"validation   refused: {exc}")
    try:
        t = session.type_classification()
        lines.append(f"type         {t['type']}")
        lines.append(f"             {len(t['cycles'])} cycle(s) in P, "
                     f"{len(t['unbranched_cycles'])} without a branch point")
    except Exception as exc:                                # noqa: BLE001
        lines.append(f"type         refused: {exc}")
    return "\n".join(lines)


def portrait(session):
    """(headers, rows, note) for the critical-portrait table."""
    p = session.portrait()
    headers = ("cover face", "boundary", "target base face", "degree")
    rows = []
    for i, r in enumerate(p["cover_faces"]):
        target = "—" if r["target"] is None else str(r["target"])
        degree = "—" if r["degree"] is None else str(r["degree"])
        rows.append((str(i), _face_labels(r["face"]), target, degree))
    degrees = sorted(p["degrees"])
    note = f"portrait degrees {degrees}"
    if p["unprojected"]:
        note += f"; {len(p['unprojected'])} cover face(s) do not project"
    return headers, rows, note


def automaton(session):
    """The algebra block: the rose normal form and what is known of it."""
    info = session.automaton()
    a = info["automaton"]
    tree = ", ".join(info["tree"]) or "(rose: no edges in the tree)"
    how = ("admissible tree (sections read off the generators' lifts)"
           if info["admissible"] else
           "no admissible tree -- built on any spanning tree; valid, but a "
           "different presentation (the shortcut is not exact)")
    lines = [
        f"letters      {', '.join(a.letters)}",
        f"sheets       {len(a.sheets)}",
        f"tree         {tree}",
        f"             {how}",
    ]
    nuc = session.nucleus()
    state = "closed" if nuc["closed"] else "open (not proved; semi-decision)"
    lines.append(f"nucleus pi1  {nuc['classes']} words, {state}")
    lines.append(f"             cap {nuc['cap']}, depth {nuc['depth']}")
    try:
        classes = ", ".join(f"L{n}: {session.nucleus(level=n)['classes']}"
                            for n in (1, 2, 3))
        lines.append(f"level classes {classes}"
                     "   (image up to the level-n kernel)")
    except Exception as exc:                                # noqa: BLE001
        lines.append(f"level classes refused: {exc}")
    d = len(a.sheets)
    orders, n = [], 1
    # Schreier-Sims cost is set by the number of points, and jumps sharply:
    # measured, d = 3 at level 4 (81 points) is 0.13 s and level 5 (243) is
    # ~8 s; barycentric's level 4 (1296 points) is 50 s.  The panel stops at
    # 81 points; `Session.level_order(n)` has no such limit for scripts.
    try:
        while n <= 6 and d ** n <= 81:
            orders.append(str(session.level_order(n)))
            n += 1
        if orders:
            more = ("; the panel stops at 81 points per level"
                    if n <= 6 and d ** n > 81 else "")
            lines.append(f"level orders {', '.join(orders)}"
                         f"   (levels 1-{len(orders)}{more})")
    except Exception as exc:                                # noqa: BLE001
        lines.append(f"level orders refused: {exc}")
    return "\n".join(lines)


def energies(session, p=2, n=1):
    """(headers, rows, note) for the lifting-bound table."""
    rows = session.energy_bounds(p=p, n=n)
    headers = ("n", "E^p_p(phi^n)", "(E^p_p)^(1/n)")
    body = [(str(r["n"]), f"{r['E']:.10g}", f"{r['E_root']:.10g}") for r in rows]
    note = ("upper bounds on the asymptotic energy (plan item 180); exact at "
            f"integral p, here p = {p}.  The metric is the session's own: no "
            "[Elast] §6 minimisation, so a value of 1 is unit-speed, not a "
            "certificate.")
    return headers, body, note
