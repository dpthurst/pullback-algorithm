"""Tests for the exact-rational CEPS fixtures / oracle (ceps_exact.py).

Run with:  PYTHONPATH=src python -m pytest tests/test_ceps_exact.py -q
"""
import math
import os
import sys
from fractions import Fraction as F

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pullback import ceps_exact as ce  # noqa: E402

TETRA = ce.fixture_tetra()
TETRA_FACETS = ce.fixture_tetra_facets()
OCTA = ce.fixture_octahedron()
OCTA_FACETS = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
               (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
BIPY = ce.fixture_bipyramid()
BIPY_FACETS = ce.fixture_bipyramid_facets()

V_TETRA = [F(0), F(0), F(0), F(0)]
V_SCALED = [F(0), F(1, 2), F(-1, 3), F(2, 5)]


# ---------------------------------------------------------------- fixtures
@pytest.mark.parametrize("points", [TETRA, OCTA, BIPY])
def test_fixtures_are_rational_points_of_the_unit_sphere(points):
    assert ce.check_on_sphere(points) == []
    assert ce.check_distinct(points) == []
    for p in points:
        assert all(isinstance(c, F) for c in p)


@pytest.mark.parametrize("points,facets", [(TETRA, TETRA_FACETS),
                                           (OCTA, OCTA_FACETS),
                                           (BIPY, BIPY_FACETS)])
def test_fixtures_are_strictly_supporting_and_outward(points, facets):
    assert ce.check_supporting(points, facets) == []
    assert ce.check_orientation(points, facets) == []
    oriented = ce.orient_outward(points, facets)
    d = ce.radial_degree_exact(points, oriented)
    assert d["degree_one"] is True
    assert d["positive"] == len(facets) and d["negative"] == 0 and d["zeros"] == 0


def test_facet_lists_are_consistently_oriented():
    """Regression: every fixture is a consistently oriented PL sphere.

    An earlier version of this file used
    TETRA_FACETS = [(0,1,2),(0,1,3),(0,2,3),(1,2,3)], which carries dart (0,1)
    in two faces.  Production ``pullback.dense_assembly.base_from_face_lists``
    refuses it, and every orientation-dependent conclusion drawn from it was
    unsupported.  Assert the opposite-dart closure directly, and assert that the
    production checker accepts each fixture and refuses the defective list.
    """
    from pullback import dense_assembly as A
    for points, facets in ((TETRA, TETRA_FACETS), (OCTA, OCTA_FACETS),
                           (BIPY, BIPY_FACETS)):
        assert ce.check_opposite_darts(facets) == []
        A.base_from_face_lists(facets)          # raises if not oriented
        assert ce.check_orientation(points, facets) == []
    bad = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    assert ce.check_opposite_darts(bad)
    with pytest.raises(ValueError):
        A.base_from_face_lists(bad)


def test_orientation_is_judged_in_the_given_order():
    """Acceptance must not silently reorient: the inward list is refused."""
    inward = [(0, 1, 2), (1, 0, 3), (2, 1, 3), (0, 2, 3)]
    assert ce.check_opposite_darts(inward) == []      # consistent ...
    assert ce.check_orientation(TETRA, inward)        # ... but inward
    assert ce.check_orientation(TETRA, TETRA_FACETS) == []


def test_supporting_margin_values_are_exact():
    margins = ce.facet_margins(TETRA, TETRA_FACETS)
    assert [m for _k, m in margins] == [F(2, 5)] * 4


def test_origin_radial_test_is_invalid_for_an_inscribed_tetrahedron():
    """Regression: supporting planes, not origin-radial signs, decide convexity.

    The rational tetrahedron fixture lies in a closed hemisphere, so the origin
    is on one facet plane and the radial map has no degree.  A test that used
    origin-radial signs would wrongly reject a perfectly good inscribed
    realisation.
    """
    assert ce.origin_radial_degree(TETRA, TETRA_FACETS)["origin"] == "on_surface"
    assert ce.origin_radial_degree(TETRA, TETRA_FACETS)["degree"] is None
    for pts, fac in ((OCTA, OCTA_FACETS), (BIPY, BIPY_FACETS)):
        info = ce.origin_radial_degree(pts, fac)
        assert info["origin"] == "inside" and info["degree"] == 1


def test_mobius_invariant_values_and_rotation_invariance():
    assert abs(ce.mobius_invariant(TETRA) - complex(-0.5, 0.5)) < 1e-11
    assert abs(ce.mobius_invariant(OCTA) - complex(-1.0, 0.0)) < 1e-11
    # exact rational rotation (3/5,-4/5,0),(4/5,3/5,0),(0,0,1) is an isometry of S^2
    rot = ((F(3, 5), F(-4, 5), F(0)), (F(4, 5), F(3, 5), F(0)), (F(0), F(0), F(1)))
    rotated = [ce._rotate(rot, p) for p in TETRA]
    assert ce.check_on_sphere(rotated) == []
    assert abs(ce.mobius_invariant(rotated) - ce.mobius_invariant(TETRA)) < 1e-11


# ------------------------------------------------- vertex-scaled fixtures
@pytest.mark.parametrize("v", [V_TETRA, V_SCALED])
def test_vertex_scaled_fixture_identity_is_exact(v):
    """chord^2 = l'^2 exp(u_i+u_j) holds exactly at the original configuration
    with u = -v: the answer for the scaled input is the original up to Mobius."""
    fx = ce.VertexScaledFixture(TETRA, TETRA_FACETS, v, pole=3)
    resid, per = ce.scaling_tolerance(TETRA, fx.input_lengths_sq(),
                                      [-float(t) for t in v], allow_constant=False)
    assert resid < 1e-12, per
    # the additive constant is pinned for a four-vertex sphere: shifting u fails
    shifted = [3.0 - float(t) for t in v]
    resid2, _ = ce.scaling_tolerance(TETRA, fx.input_lengths_sq(), shifted,
                                     allow_constant=False)
    assert resid2 > 1.0


@pytest.mark.parametrize("v", [V_TETRA, V_SCALED])
def test_scaling_factor_recovered_from_positions(v):
    """The least-squares conformal factor of the known answer is mean-shifted -v."""
    fx = ce.VertexScaledFixture(TETRA, TETRA_FACETS, v, pole=3)
    got = ce.scaling_from_positions(TETRA, fx.input_lengths_sq())
    want = fx.expected_u()
    assert max(abs(a - b) for a, b in zip(got, want)) < 1e-12


def test_scaled_input_lengths_are_not_the_plain_chords():
    fx = ce.VertexScaledFixture(TETRA, TETRA_FACETS, V_SCALED, pole=3)
    plain = ce.chord_metric_sq(TETRA)
    scaled = fx.input_lengths_sq()
    assert any(abs(scaled[e] / float(plain[e]) - 1.0) > 1e-3 for e in plain)


# --------------------------------------------------------- independent oracle
@pytest.mark.parametrize("v", [V_TETRA, V_SCALED])
@pytest.mark.parametrize("pole", [0, 3])
def test_oracle_recovers_the_known_answer_up_to_mobius(v, pole):
    """The scaling-free characterisation (no sec.-7 energy, no delta, no peacock)
    recovers the known answer and its conformal factor u = -v."""
    fx = ce.VertexScaledFixture(TETRA, TETRA_FACETS, v, pole=pole)
    branches = ce.enumerate_scaling_branches(fx.input_lengths_sq(), pole=pole,
                                             seeds=30)
    assert len(branches) == 2, "expected an orientation-reversed pair"
    target = fx.expected_invariant()
    hits = [b for b in branches
            if ce.same_oriented_mobius_class(b["positions"], TETRA)]
    assert len(hits) == 1
    # The conformal factor is representative dependent: it can be compared with
    # u = -v only in the gauge whose representative is the original
    # configuration itself.  For TETRA that is the gauge that puts vertices
    # (0,1,2) at e1,e2,e3, i.e. pole = 3.
    if pole == 3:
        got_u = hits[0]["u"]
        want_u = fx.expected_u()
        assert max(abs(a - b) for a, b in zip(got_u, want_u)) < 1e-8


@pytest.mark.parametrize("v", [V_TETRA, V_SCALED])
def test_oriented_input_selects_one_branch(v):
    """Branch selection is decided by the orientation of the given facets.

    The two scaling branches are mirror images.  Exactly one of them has the
    facet order of the input pointing outward, and it is the one that matches
    the answer's *oriented* Mobius class; the other must be refused as a mirror
    image.  A solver that only checks a scaling residual and convexity cannot
    tell them apart.
    """
    fx = ce.VertexScaledFixture(TETRA, TETRA_FACETS, v, pole=3)
    branches = ce.enumerate_scaling_branches(fx.input_lengths_sq(), pole=3, seeds=30)
    assert len(branches) == 2
    oriented = [b for b in branches
                if ce.check_orientation(b["positions"], TETRA_FACETS) == []]
    assert len(oriented) == 1
    win = oriented[0]
    assert ce.same_oriented_mobius_class(win["positions"], TETRA)
    other = [b for b in branches if b is not win][0]
    assert ce.is_mirror_image(other["positions"], TETRA)
    assert abs(other["invariant"] - win["invariant"]) > 0.1


# ------------------------------------------------------------- falsification
def test_naive_two_log_bound_rules_do_not_reproduce_the_answer():
    """Negative control: delta = +-2 log l(v,pole) is *not* the peacock delta.

    Each rule is fed to the exact all-active four-vertex construction; the
    resulting Mobius class is different from the known answer, so either rule
    would be caught.  The peacock value (Springborn Prop. 5.16) is what the
    production solver must compute.
    """
    fx = ce.VertexScaledFixture(TETRA, TETRA_FACETS, V_SCALED, pole=3)
    rep = ce.candidate_rule_report(fx)
    target = rep["original_configuration"]
    for name in ("plus_2log", "minus_2log"):
        assert abs(rep[name] - target) > 0.1, name


def test_negative_controls_are_rejected():
    ctl = ce.negative_controls()
    assert ctl["repeated_vertex"], "repeated vertices must be rejected"
    assert ctl["off_sphere"], "points off the sphere must be rejected"
    assert ctl["collapse"], "a collapsed configuration must be rejected"


def test_supporting_test_rejects_a_non_facet_triangle():
    """The rim triangle of the bipyramid is not a facet: the poles lie on both
    sides of its plane, so the supporting test must reject it."""
    bad = BIPY_FACETS + [(2, 3, 4)]
    defects = ce.check_supporting(BIPY, bad)
    assert defects, "a non-supporting triangle must be rejected"
