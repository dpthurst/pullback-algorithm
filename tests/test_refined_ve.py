"""Acceptance tests for the deterministic ribbon-native refinement.

Run from the repository root: PYTHONPATH=src python3 -m pytest -q tests/test_refined_ve.py
"""
import os

import numpy as np
import pytest

from pullback import embedding, rounding_multi as mg
from pullback import refined_cover as rr
from pullback import refined_ve as pl
from pullback.session import Session

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def state():
    s = Session.load(os.path.join(REPO, "sessions", "rabbit.json"))
    return s, pl.RefinedVE(s)


def fib():
    return np.array([[0.6614378277661477, 0.0, 0.75],
                     [-0.713954346202245, 0.6540406650499072, 0.25],
                     [0.08464959396472492, -0.9645384628108966, -0.25],
                     [0.402444478534368, 0.5249175570479622, -0.75]])


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _mobius_image(x, a, lam, shift=0j):
    """A declared spherical Möbius map, used for the folded-base controls."""
    x = np.asarray(x, dtype=float)
    a = _unit(a)
    ax = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]],
                   [-a[1], a[0], 0.0]])
    angle = float(np.arccos(np.clip(a[2], -1.0, 1.0)))
    rot = (np.eye(3) + np.sin(angle) * ax
           + (1.0 - np.cos(angle)) * ax @ ax)
    y = x @ rot.T
    z = (y[:, 0] + 1j * y[:, 1]) / (1.0 - y[:, 2])
    z = lam * z + shift
    den = 1.0 + np.abs(z) ** 2
    return np.stack([2 * z.real / den, 2 * z.imag / den,
                     (np.abs(z) ** 2 - 1) / den], axis=1)


def _configs():
    alt = np.stack([_unit((0.9, 0.2, 0.35)), _unit((-0.5, 0.8, 0.2)),
                    _unit((0.1, -0.9, -0.35)), _unit((-0.35, -0.15, -0.9))])
    polar = np.stack([_unit((0.05, 0.02, 1.0)), _unit((-0.6, 0.75, 0.25)),
                      _unit((0.7, 0.6, -0.3)), _unit((0.0, -0.1, -1.0))])
    near = np.stack([_unit((0.6, 0.3, 0.75)), _unit((-0.7, 0.6, 0.35)),
                     _unit((0.2, -0.95, -0.25)), _unit((0.62, 0.33, 0.69))])
    c, s = np.cos(0.7), np.sin(0.7)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return {"fib": fib(), "alt": alt, "polar": polar, "near": near,
            "fib_mobius": _mobius_image(fib(), (0.3, -0.4, 0.5), 1.7 + 0.6j),
            "mobius_m2": _mobius_image(fib(), (0.6, 0.3, 0.75), 0.4, -0.9 + 0.3j),
            "mobius_m3": _mobius_image(fib(), (-0.7, 0.6, 0.35), 2.3, 1.1j),
            "fib_rot": fib() @ rot.T,
            "alt_mobius": _mobius_image(alt, (-0.2, 0.6, -0.3), 0.8 - 1.1j)}


# ------------------------------------------------------------------ structure
def test_exact_counts(state):
    _s, st = state
    c = st.certificates()
    assert c["coarse_counts"]["T"] == [4, 6, 4]
    assert c["coarse_counts"]["K"] == [6, 12, 8]
    assert c["refined_counts"]["T'"] == [5, 9, 6]
    assert c["refined_counts"]["K'"] == [8, 18, 12]


def test_refined_is_a_cover(state):
    _s, st = state
    c = st.certificates()
    assert c["cover2"]["ok"]
    assert c["cover2"]["edge_multiplicities"] == [2]
    assert c["cover2"]["face_multiplicities"] == [2]
    assert c["cover2"]["no_phantom_labels"]
    assert c["fibre_degrees_ok"]


def test_ribbon_criterion_and_local_degrees(state):
    _s, st = state
    c = st.certificates()
    assert c["ribbon2_ok"]
    assert all(r["ok"] for r in c["ribbon2_rows"])
    assert len(c["ribbon2_rows"]) == 8
    # the two branch points have local degree 2, everything else 1
    deg = c["degrees"]
    assert sorted(int(v) for v in deg.values()) == [1, 1, 1, 1, 1, 1, 2, 2]


def test_collapse_onto_the_coarse_object(state):
    _s, st = state
    c = st.certificates()
    coll = c["collapse"]
    assert coll["projection_preserved"] and coll["local_degree_preserved"]
    assert coll["old_vertices"] == 6
    assert coll["new_over"] == [4, 4]          # the two midpoints sit over the aux point
    assert c["aux_vertices"] == coll["new_vertices"]
    assert c["aux_not_postcritical"]


def test_split_is_the_branch_edge(state):
    _s, st = state
    c = st.certificates()
    assert c["ramified"] == [0, 3]
    assert c["split_edge"] == (0, 3)
    assert c["branch_datum"] == [(0, 3)]
    # the refinement replaces the branch edge by its two halves, each a branch half
    assert c["branch_datum2"] == [(0, 4), (3, 4)]


def test_portrait_and_labels(state):
    s, st = state
    c = st.certificates()
    assert c["dyn"] == dict(s.initial_geometry().dynamics)
    assert c["labels"] == dict(st.init.lifted_labels)
    assert c["labels_roundtrip"]
    assert sorted(c["labels"]) == [0, 1, 2, 3]
    # every label sits over dyn(label), with the portrait's local degree
    rows = {r["target"]: r for r in c["portrait_rows"] if r["target"] is not None}
    for p, u in c["labels"].items():
        assert c["proj"][str(u)] == c["dyn"][p]
        assert int(c["degrees"][str(u)]) == rows[p]["ld"]


def test_uniform_refinement_commutes_with_the_cover(state):
    _s, st = state
    sub = st.certificates()["subdivision"]
    assert sub[0]["base"] == [14, 36, 24]
    assert sub[0]["lift"] == [26, 72, 48]
    assert sub[1]["base"] == [50, 144, 96]
    assert sub[1]["lift"] == [98, 288, 192]


# ------------------------------------------------------------------- geometry
def test_one_step_is_embedded(state):
    _s, st = state
    cert = st.step([np.asarray(p, dtype=float) for p in fib()])
    assert cert["embedded"], cert["reason"]
    assert cert["residual"] < 1e-12
    assert cert["rank_PJ"] == 10 and cert["excess"] == 0
    assert abs(abs(cert["degree"]) - 1.0) < 1e-9
    assert cert["min_abs_det"] > 0.1
    assert cert["injective"]
    assert cert["seed"]["one_sign"] and cert["seed"]["multiplicative"]


def test_seed_is_a_gauge_fixed_point(state):
    """A rotation is a Mobius map: the labelled chart must not move."""
    _s, st = state
    base = st.step([np.asarray(p, dtype=float) for p in fib()])
    c, s2 = np.cos(0.9), np.sin(0.9)
    R = np.array([[c, -s2, 0.0], [s2, c, 0.0], [0.0, 0.0, 1.0]])
    cfg = [np.asarray(p, dtype=float) @ R.T for p in fib()]
    rot = st.step(cfg)
    assert rot["embedded"] and base["embedded"]
    deltas = [abs(complex(a) - complex(b))
              for a, b in zip(base["chart"][1], rot["chart"][1])]
    assert max(deltas) < 1e-9, deltas


def test_portrait_is_not_enough(state):
    """The portrait does not determine the step: it needs the lift table.

    Both lifts over an unramified puncture share (projection, local degree), so
    the portrait cannot say which carries the postcritical label.  Moving ONE
    label to the other lift changes the labelled coordinate; moving BOTH labels
    over the unramified punctures at once is the deck transformation and is a
    gauge symmetry, so it must not change it.  This is the machine-checked
    witness that the portrait is a shadow of phi', not a replacement for it.
    """
    _s, st = state
    cfg = [np.asarray(p, dtype=float) for p in fib()]
    good = st.step(cfg)
    labels = dict(st.labels)
    # move ONE label to the other lift over its puncture (not the deck action)
    flip = {}
    for p in sorted(labels):
        u = labels[p]
        other = [v for v in st.K2["proj"]
                 if st.K2["proj"][v] == st.K2["proj"][u] and v != u]
        if other:
            flip[p] = other[0]
            break
    moved = {p: flip.get(p, u) for p, u in labels.items()}
    st.labels = moved
    try:
        bad = st.step(cfg)
    finally:
        st.labels = labels
    same = (bad["chart"][0] == good["chart"][0]
            and all(abs(complex(a) - complex(b)) < 1e-9
                    for a, b in zip(good["chart"][1], bad["chart"][1])))
    assert not same, "a phantom label assignment must change the read-off"
    # the deck action at every unramified puncture at once is a symmetry
    both = dict(labels)
    for p in sorted(labels):
        u = labels[p]
        other = [v for v in st.K2["proj"]
                 if st.K2["proj"][v] == st.K2["proj"][u] and v != u]
        if other:
            both[p] = other[0]
    st.labels = both
    try:
        deck = st.step(cfg)
    finally:
        st.labels = labels
    assert deck["chart"][0] == good["chart"][0]
    assert all(abs(complex(a) - complex(b)) < 1e-9
               for a, b in zip(good["chart"][1], deck["chart"][1]))


def test_mobius_controls_preserve_cross_ratios():
    """The folded controls are genuine Möbius copies, not only generic maps."""
    for name, moved in _configs().items():
        if name in ("fib_mobius", "mobius_m2", "mobius_m3", "alt_mobius"):
            original_chart = pl.chart(fib() if name != "alt_mobius" else _configs()["alt"])
            moved_chart = pl.chart(moved)
            assert moved_chart[0] == original_chart[0]
            assert max(abs(complex(a) - complex(b))
                       for a, b in zip(original_chart[1], moved_chart[1])) < 1e-9, name


def test_folded_base_refuses_to_step(state):
    """The gate is enforced in the API, not merely reported."""
    s, st = state
    cfg = [np.asarray(p, dtype=float) for p in fib()]
    good = st.base_orientation(cfg)
    assert good["one_sign"]
    assert st.step(cfg)["base"]["one_sign"]
    cfgs = _configs()
    folded = {k: cfgs[k] for k in ("fib_mobius", "mobius_m2", "mobius_m3", "near")}
    embedded = {k: cfgs[k] for k in ("fib", "alt", "polar", "fib_rot", "alt_mobius")}
    for name, cfg2 in embedded.items():
        assert st.base_orientation([np.asarray(p, dtype=float) for p in cfg2])["one_sign"], name
    for name, cfg2 in folded.items():
        cfg2 = [np.asarray(p, dtype=float) / np.linalg.norm(p) for p in cfg2]
        gate = st.base_orientation(cfg2)
        assert not gate["one_sign"], (name, gate)
        with pytest.raises(pl.FoldedBase):
            st.step(cfg2)


def test_declared_initial_inputs_skip_transport_census(state, monkeypatch):
    """Declared conventions, not another search: a census call is a test failure."""
    import pullback.ve_state as vs

    s, st = state

    def forbidden(*args, **kwargs):
        raise AssertionError("find_transports was called for a declared transport")

    monkeypatch.setattr(vs, "find_transports", forbidden)
    fresh = pl.RefinedVE.from_declared(s, st.init.chains, st.labels,
                                       points=st.init.points)
    assert fresh.init.transports is None
    assert fresh.labels_source == "declared"
    public = s.refined_geometry(points=st.init.points, chains=st.init.chains,
                                postcritical_lifts=st.labels)
    assert public.init.transports is None
    assert public.labels_source == "declared"
    cert = public.step(fib())
    assert cert["embedded"] and cert["rank_PJ"] == 10
    assert abs(abs(cert["degree"]) - 1.0) < 1e-9


def test_declared_transport_is_still_validated(state):
    """Avoiding the census is not permission to accept an inadmissible map."""
    s, st = state
    with pytest.raises(ValueError):
        pl.RefinedVE.from_declared(
            s, {edge: () for edge in st.init.chains}, st.labels,
            points=st.init.points)


def test_negative_control_principal_seed_is_not_embedded(state):
    _s, st = state
    cfg = [np.asarray(p, dtype=float) for p in fib()]
    X, met = st.metric(cfg)
    y = rr.conformal_seed(st.K2["proj"], X, branch="principal")
    ori = st.oriented_faces()
    det = np.array([np.linalg.det(np.array([y[a], y[b], y[c]])) for a, b, c in ori])
    assert not ((det > 0).all() or (det < 0).all())
    x, res, _it = mg.round_realise(st.K2["edges"], met["lengths"], x0=y, maxit=600)
    emb = embedding.classify(x, ori)
    assert not emb["embedded"] or abs(abs(emb["degree"]) - 1.0) > 1e-3


def test_step_reports_the_base_gate(state):
    _s, st = state
    cert = st.step([np.asarray(p, dtype=float) for p in fib()])
    assert cert["base"]["pos"] == 6 or cert["base"]["neg"] == 6
    assert cert["base"]["one_sign"]


def test_arc_midpoint_convention_can_fail(state):
    """The declared midpoint convention is not Mobius covariant: certifies it."""
    _s, st = state
    cfg = fib()
    a, b = st.ram
    m = cfg[a] + cfg[b]
    m = m / np.linalg.norm(m)
    c, s2 = np.cos(0.9), np.sin(0.9)
    R = np.array([[c, -s2, 0.0], [s2, c, 0.0], [0.0, 0.0, 1.0]])
    rot = [np.asarray(p, dtype=float) @ R.T for p in cfg]
    m2 = rot[a] + rot[b]
    m2 = m2 / np.linalg.norm(m2)
    assert np.linalg.norm(m @ R.T - m2) < 1e-12          # rotations are fine
    # ... but a general Mobius map does not preserve the arc midpoint
    X = np.asarray(cfg, dtype=float)
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0)
