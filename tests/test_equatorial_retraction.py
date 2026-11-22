"""The two-cap equatorial carrier: its mesh labels, its chart, its refusals.

The carrier is one object in three places -- the Delaunay decomposition, the
primal-pair labels of the dual Voronoi spine, and the cap transitions read off
a source arc.  These tests pin the coupling between them, and pin each refusal
that replaces a guess.
"""
import json
import os
import subprocess
import sys
import unittest
from collections import defaultdict
from unittest import mock

import mpmath as mp
import numpy as np

from pullback import equatorial_retraction as eq
from pullback import mesh
from pullback.geometric_retraction import GeometricRetractionError, geometric_upsilon
from pullback.ribbons import B, F, RibbonGraph

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _unit(x):
    x = np.asarray(x, dtype=float)
    return x / np.linalg.norm(x)


def _ring(z=0.0, n=4):
    """``n`` points of one circle.  ``z != 0`` makes it a small circle."""
    q = np.array([[0.6 * np.cos(2 * np.pi * k / n),
                   0.6 * np.sin(2 * np.pi * k / n), z] for k in range(n)])
    return np.array([_unit(v) for v in q])


def _two_vertices():
    return RibbonGraph({0: (("arc", F),), 1: (("arc", B),)})


#: Arcs whose crossings lie between different puncture pairs of `_ring()`.
_ARCS = {
    (0, 1): np.array([_unit((0.7, 0.4, 0.5)), _unit((0.8, 0.2, -0.2))]),
    (2, 3): np.array([_unit((0.2, -0.7, 0.4)), _unit((-0.3, 0.6, -0.5))]),
    (1, 2): np.array([_unit((-0.6, 0.3, 0.5)), _unit((0.1, 0.8, -0.4))]),
}


def _head_darts(points):
    """The rotation system HEAD's ``(face, neighbour)`` keying produced.

    Recomputed here from :func:`mesh.delaunay_faces` alone, so the comparison
    does not read the production code it is checking.
    """
    faces = mesh.delaunay_faces(points)
    edge_to_faces = defaultdict(list)
    for idx, face in enumerate(faces):
        n = len(face)
        for k in range(n):
            edge_to_faces[mesh.edge_key(face[k], face[(k + 1) % n])].append(idx)
    rot, labels = {}, {}
    for idx, face in enumerate(faces):
        nbrs = []
        for k in range(len(face)):
            e = mesh.edge_key(face[k], face[(k + 1) % len(face)])
            other = [j for j in edge_to_faces[e] if j != idx][0]
            nbrs.append(other)
            labels[(idx, other)] = e
        rot[idx] = list(reversed(nbrs))
    return {v: tuple((labels[(v, u)], F if v < u else B) for u in ns)
            for v, ns in rot.items()}


def assert_bigon_periphery(test, spine, n=4):
    """Each bigon is bounded by the two actual edges about one puncture."""
    face_points = mesh.voronoi_face_points(spine, n)
    test.assertEqual(set(face_points.values()), set(range(n)))
    for face, walk in enumerate(spine.faces()):
        test.assertEqual(len(walk), 2)
        common = set(walk[0][0]) & set(walk[1][0])
        test.assertEqual(common, {face_points[face]})


class TestTwoCapSpine(unittest.TestCase):
    def test_exact_equator_has_two_caps_and_four_parallel_edges(self):
        points = [mesh.to_point(v) for v in _ring()]
        faces = mesh.delaunay_faces(points)
        self.assertEqual(faces, [(2, 3, 0, 1), (2, 1, 0, 3)])
        spine = mesh.voronoi_spine(points)
        self.assertEqual((len(spine.vertices()), len(spine.edge_list()),
                          len(spine.faces())), (2, 4, 4))
        self.assertEqual(spine.vertices(), [0, 1])
        # One label per puncture pair: the parallel dual edges are told apart
        # by the primal edge they cross, not by an index of the dual endpoints.
        self.assertEqual(set(spine.edge_list()), {(0, 1), (0, 3), (1, 2), (2, 3)})
        for edge in spine.edge_list():
            self.assertEqual(set(spine.edges[edge]), {0, 1})
        assert_bigon_periphery(self, spine)
        for walk in spine.faces():
            self.assertNotEqual(walk[0][0], walk[1][0])

    def test_parallel_edges_keep_the_primal_labels_of_the_delaunay_faces(self):
        # The coupling, stated without any side table: the spine's edge labels
        # ARE the primal edges of the decomposition it came from.
        for points in ([mesh.to_point(v) for v in _ring()],
                       [mesh.to_point(v) for v in _ring(0.5)],
                       [mesh.to_point(v) for v in _ring(0.3, 5)]):
            faces = mesh.delaunay_faces(points)
            primal = {mesh.edge_key(f[k], f[(k + 1) % len(f)])
                      for f in faces for k in range(len(f))}
            spine = mesh.voronoi_spine(points)
            self.assertEqual(set(spine.edge_list()), primal)
            self.assertEqual(len(spine.edge_list()), len(points))
            self.assertEqual(set(mesh.voronoi_face_points(spine, len(points)).values()),
                             set(range(len(points))))

    def test_generic_decomposition_keeps_the_head_rotation_system(self):
        # A cocircular block is the ONLY case whose labels change.  Everywhere
        # else the emitted darts must equal the (face, neighbour) keying, in
        # the same order: that is what "no regression for generic data" means
        # and no test asserted it.
        tetrahedron = (np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
                                dtype=float) / np.sqrt(3.0))
        five = [_unit(v) for v in [(0.3, 0.2, 1.0), (-0.8, 0.1, 0.4),
                                   (0.2, -0.9, 0.3), (0.9, 0.8, -0.5),
                                   (-0.4, -0.5, -0.9)]]
        for vectors in (tetrahedron, five):
            points = [mesh.to_point(v) for v in vectors]
            expected = RibbonGraph(_head_darts(points))
            self.assertEqual(mesh.voronoi_spine(points).serialization(),
                             expected.serialization())

    def test_same_cap_boundary_works_on_a_small_circle(self):
        # A circle whose plane misses the sphere centre is still a two-cap
        # carrier; the chart must not assume a great circle.
        points = [mesh.to_point(v) for v in _ring(0.5)]
        faces = mesh.delaunay_faces(points)
        self.assertEqual(faces, [(2, 3, 0, 1), (2, 1, 0, 3)])
        spine = mesh.voronoi_spine(points)
        self.assertEqual((len(spine.vertices()), len(spine.edge_list()),
                          len(spine.faces())), (2, 4, 4))
        assert_bigon_periphery(self, spine)
        geo = eq.layout(_ring(0.5), faces)
        self.assertGreater(geo["offset"], 0.5)
        self.assertAlmostEqual(geo["radius"], np.sqrt(1 - geo["offset"] ** 2))

    def test_five_concyclic_punctures_give_five_parallel_edges(self):
        # Nothing is sized around four punctures: the two-cap carrier is the
        # decomposition for any concyclic set, and a fivefold block has no
        # fourfold cross-ratio to certify it -- the coplanarity measures do.
        q = _ring(0.4, 5)
        points = [mesh.to_point(v) for v in q]
        faces = mesh.delaunay_faces(points)
        self.assertEqual(len(faces), 2)
        self.assertEqual(sorted(map(len, faces)), [5, 5])
        spine = mesh.voronoi_spine(points)
        self.assertEqual((len(spine.vertices()), len(spine.edge_list()),
                          len(spine.faces())), (2, 5, 5))
        self.assertEqual(set(spine.edge_list()),
                         {mesh.edge_key(i, j) for i in range(5)
                          for j in range(i + 1, 5) if j - i in (1, 4)})
        assert_bigon_periphery(self, spine, 5)
        out = geometric_upsilon(_two_vertices(), _ARCS[(0, 1)], points,
                                steps=(32, 64))
        self.assertTrue(out["equatorial"])
        self.assertEqual(out["equatorial_route"]["route"], "equatorial")
        self.assertAlmostEqual(out["equatorial_route"]["plane_thickness"], 0.0)
        self.assertEqual(len(out["target"].edge_list()), 5)

    def test_the_real_airplane_has_an_exact_cocircular_configuration(self):
        # A structural acceptance case, not a tolerance heuristic: every point
        # of the real period-3 centre's postcritical set is real.
        from pullback import discrete, thurston
        from pullback.rational import RationalMap
        c = dict(thurston.period_three_centres())["airplane"]
        f = RationalMap([mp.mpf(1), mp.mpf(0), c], [mp.mpf(1)])
        points = discrete.pcf_data(f)["points"]
        vectors = np.array([mesh.to_vector(p) for p in points])
        self.assertTrue(np.all(vectors[:, 1] == 0.0))
        faces = mesh.delaunay_faces(points)
        spine = mesh.voronoi_spine(points)
        self.assertEqual((len(spine.vertices()), len(spine.edge_list()),
                          len(spine.faces())), (2, 4, 4))
        geo = eq.layout(vectors, faces)
        self.assertEqual(geo["coplanarity_residual"], 0.0)
        self.assertAlmostEqual(geo["min_angular_gap"], 1.0358959519612752,
                               places=12)
        self.assertGreater(geo["min_angular_gap"], 1.0)
        self.assertAlmostEqual(eq.require_real_fourfold(vectors).imag, 0.0)
        assert_bigon_periphery(self, spine)
        # The pinning question "is the ordinary centred use a great circle?"
        # is measured, not assumed: here the plane does pass through the
        # carrier's chart origin, in both gauges.
        self.assertAlmostEqual(geo["offset"], 0.0, places=12)
        _, centred, _gauge = __import__(
            "pullback.hyperbolic_centre", fromlist=["centred_coordinates"]
        ).centred_coordinates(np.empty((0, 3)), vectors)
        self.assertAlmostEqual(eq.layout(centred, mesh.delaunay_faces(
            [mesh.to_point(v) for v in centred]))["offset"], 0.0, places=12)

    def test_partial_cocircular_block_remains_one_polygon(self):
        # The special rule applies to ALL points on one circle.  A coplanar
        # block plus another puncture keeps the existing supporting facets.
        ring = [mesh.to_point(np.array([0.6 * np.cos(k * np.pi / 2),
                                        0.6 * np.sin(k * np.pi / 2), 0.5]))
                for k in range(4)]
        pts = ring + [mesh.to_point(np.array([0.0, 0.0, -1.0]))]
        self.assertEqual(sorted(map(len, mesh.delaunay_faces(pts))),
                         [3, 3, 3, 3, 4])


class TestEquatorialRetraction(unittest.TestCase):
    def test_crossing_uses_the_actual_consecutive_puncture_pair(self):
        points = [mesh.to_point(v) for v in _ring()]
        out = geometric_upsilon(_two_vertices(), _ARCS[(0, 1)], points,
                                steps=(64, 128))
        self.assertEqual(out["paths"]["arc"]["cells"], (0, 1))
        self.assertEqual(out["paths"]["arc"]["transition_labels"], [(0, 1)])
        self.assertEqual(list(out["paths"]["arc"]["darts"]), [((0, 1), F)])
        self.assertEqual(out["map"].chains["arc"], (((0, 1), F),))
        self.assertEqual(out["map"].vertex_images, {0: 0, 1: 1})
        self.assertEqual(out["equatorial_route"]["route"], "equatorial")
        assert_bigon_periphery(self, out["target"])

    def test_different_arcs_use_different_parallel_labels(self):
        # All four dual edges join the same two cap vertices, so a label that
        # is merely "some parallel edge" is useless.  Two arcs must carry two
        # different labels, and each must be the pair its crossing is between.
        points = [mesh.to_point(v) for v in _ring()]
        labels = {}
        for pair, arc in _ARCS.items():
            out = geometric_upsilon(_two_vertices(), arc, points, steps=(64, 128))
            self.assertEqual(out["paths"]["arc"]["cells"], (0, 1))
            labels[pair] = out["paths"]["arc"]["transition_labels"]
        self.assertEqual(labels, {(0, 1): [(0, 1)], (2, 3): [(2, 3)],
                                  (1, 2): [(1, 2)]})

    def test_a_first_pair_labelling_is_a_detectable_known_bad_control(self):
        # Known-bad control: replace the incidence with "the first boundary
        # pair" -- the mutation that a membership-only assertion cannot see --
        # and require the test above to notice.  Without this, "the label is in
        # the target" and "the label is geometrically right" are conflated.
        points = [mesh.to_point(v) for v in _ring()]
        good = geometric_upsilon(_two_vertices(), _ARCS[(1, 2)], points,
                                 steps=(64, 128))
        self.assertEqual(good["paths"]["arc"]["transition_labels"], [(1, 2)])

        def first_pair(x, geo, tol=eq.DEFAULT_TOLERANCE):
            return geo["pairs"][0]

        with mock.patch.object(eq, "_boundary_pair", first_pair):
            bad = geometric_upsilon(_two_vertices(), _ARCS[(1, 2)], points,
                                    steps=(64, 128))
        self.assertEqual(bad["paths"]["arc"]["transition_labels"], [(2, 3)])
        self.assertNotEqual(good["paths"]["arc"]["darts"],
                            bad["paths"]["arc"]["darts"])

    def test_an_unstable_transition_label_is_refused(self):
        # The two sampling levels must agree on the *label*, not only on the
        # cell sequence: two cap ids and four parallel edges leave the dart
        # still ambiguous.  A near-tangency that moves the label between the
        # two resolutions is measure-zero in a test fixture, so the control is
        # injected: make the second resolution answer with a different pair.
        points = [mesh.to_point(v) for v in _ring()]
        real = eq.cell_sequence
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            cells, margin, clearance, labels = real(*args, **kwargs)
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                return cells, margin, clearance, [(0, 1) if l else l for l in labels]
            return cells, margin, clearance, labels

        with mock.patch.object(eq, "cell_sequence", flaky):
            with self.assertRaisesRegex(GeometricRetractionError,
                                        "unstable equatorial transition labels"):
                geometric_upsilon(_two_vertices(), _ARCS[(1, 2)], points,
                                  steps=(64, 128))

    def test_an_exact_half_turn_gap_does_not_false_refuse_the_cyclic_order(self):
        # This is an uncentred synthetic layout control, not an ordinary
        # zero-barycentre configuration: a consecutive exact half-turn in a
        # centred view forces |P| = 2, where layout refuses earlier.  Here
        # np.unwrap leaves a -pi jump in place, even though the true ordered
        # gaps are pi, pi/3, pi/3, pi/3.
        angles = np.deg2rad([0.0, 60.0, 120.0, 180.0])
        q = np.array([0.6 * np.cos(angles), 0.6 * np.sin(angles),
                      np.zeros(4)]).T
        geo = eq.layout(q, mesh.delaunay_faces([mesh.to_point(v) for v in q]))
        self.assertGreater(geo["min_angular_gap"], 0.0)
        self.assertAlmostEqual(geo["min_angular_gap"], np.pi / 3, places=12)
        self.assertTrue(np.all(np.diff(geo["angles"]) > 0.0))

    def test_a_crossing_on_a_sampling_point_is_one_transition(self):
        q = _ring()
        points = [mesh.to_point(v) for v in q]
        normal = eq.layout(q, mesh.delaunay_faces(points))["normal"]
        x = _unit(q[0] + q[1])
        arc = np.array([_unit(x + 0.3 * normal), _unit(x - 0.3 * normal)])
        out = geometric_upsilon(_two_vertices(), arc, points,
                                steps=(32, 64), centre="origin")
        self.assertEqual(out["paths"]["arc"]["cells"], (0, 1))
        self.assertEqual(out["paths"]["arc"]["transition_labels"], [(0, 1)])
        self.assertEqual(out["map"].chains["arc"], (((0, 1), F),))

    def test_a_same_cap_touch_is_refused_as_a_tangency(self):
        q = _ring(0.5)
        points = [mesh.to_point(v) for v in q]
        geo = eq.layout(q, mesh.delaunay_faces(points))
        normal = geo["normal"]
        x = geo["offset"] * normal + geo["radius"] * (
            geo["u"] + geo["w"]) / np.sqrt(2.0)
        tangent = np.cross(normal, x)
        delta = 0.3
        arc = np.array([_unit(np.cos(delta) * x + np.sin(delta) * tangent),
                        _unit(np.cos(delta) * x - np.sin(delta) * tangent)])
        with self.assertRaisesRegex(GeometricRetractionError,
                                    "touches the equatorial boundary without crossing"):
            geometric_upsilon(_two_vertices(), arc, points,
                              steps=(32, 64), centre="origin")

    def test_a_small_circle_crossing_is_classified_by_arc_incidence(self):
        # centre="origin" leaves the carrier in its own plane; with offset
        # 0.6402 the boundary arc between two punctures is NOT a great-circle
        # arc, and the affine cone test misclassifies this crossing.  The
        # angular test must give the pair the crossing geometrically has.
        q = _ring(0.5)
        points = [mesh.to_point(v) for v in q]
        faces = mesh.delaunay_faces(points)
        geo = eq.layout(q, faces)
        self.assertGreater(geo["offset"], 0.5)
        arc = np.array([_unit((0.4, -0.6, 0.3)), _unit((-0.7, 0.1, 0.6))])
        out = geometric_upsilon(_two_vertices(), arc, points, steps=(64, 128),
                                centre="origin")
        self.assertTrue(out["equatorial"])
        self.assertEqual(out["equatorial_route"]["route"], "equatorial")
        self.assertEqual(out["paths"]["arc"]["cells"], (1, 0))
        self.assertEqual(out["paths"]["arc"]["transition_labels"], [(0, 3)])
        # The crossing really is nearer punctures 3 and 0 than any others.
        crossing = _unit(np.array([0.22408981, -0.73481133, 0.64018440]))
        order = np.argsort(np.linalg.norm(q - crossing, axis=1))
        self.assertEqual(set(order[:2]), {0, 3})

    def test_angular_and_affine_incidence_agree_on_a_great_circle(self):
        # The angular test is the production one because it is the only one
        # that survives a small circle.  Where the affine test is valid -- a
        # great circle -- they must agree; otherwise the repair would have
        # changed the answer instead of generalising it.
        q = _ring()
        geo = eq.layout(q, mesh.delaunay_faces([mesh.to_point(v) for v in q]))
        geo["q"] = q
        rng = np.random.default_rng(20261122)
        checked = 0
        for _ in range(400):
            a, b = _unit(rng.normal(size=3)), _unit(rng.normal(size=3))
            sa = float(a @ geo["normal"] - geo["offset"])
            sb = float(b @ geo["normal"] - geo["offset"])
            if abs(sa) < 2e-2 or abs(sb) < 2e-2 or sa * sb > 0:
                continue
            arc = __import__("pullback.geometric_retraction",
                             fromlist=["arc_points"]).arc_points(a, b, 4096)
            s = arc @ geo["normal"] - geo["offset"]
            k = int(np.argmax(np.sign(s[:-1]) * np.sign(s[1:]) < 0))
            lo, hi = arc[k], arc[k + 1]
            for _step in range(80):
                mid = _unit((lo + hi) / 2)
                if float(mid @ geo["normal"] - geo["offset"]) * float(
                        lo @ geo["normal"] - geo["offset"]) >= 0:
                    lo = mid
                else:
                    hi = mid
            x = _unit((lo + hi) / 2)
            x -= float(x @ geo["normal"] - geo["offset"]) * geo["normal"]
            x = _unit(x)
            self.assertEqual(eq._angular_boundary_pair(x, geo),
                             eq._affine_boundary_pair(x, geo))
            checked += 1
        self.assertGreater(checked, 100)

    def test_an_equatorial_source_vertex_is_slid_consistently(self):
        points = [mesh.to_point(v) for v in _ring()]
        source_positions = np.array([_unit((0.8, 0.6, 0.0)),
                                     _unit((-0.4, 0.3, -0.6))])
        out = geometric_upsilon(_two_vertices(), source_positions, points,
                                steps=(16, 32))
        self.assertEqual(out["vertex_cells"][0], (0, 1))
        self.assertEqual(out["map"].vertex_images[0], 0)
        for label, _orientation in out["paths"]["arc"]["darts"]:
            self.assertIn(label, out["target"].edge_list())

    def test_arc_along_equator_is_refused(self):
        points = [mesh.to_point(v) for v in _ring()]
        source_positions = np.array([_unit((np.cos(0.25), np.sin(0.25), 0.0)),
                                     _unit((np.cos(1.35), np.sin(1.35), 0.0))])
        with self.assertRaisesRegex(GeometricRetractionError, "entire source arc"):
            geometric_upsilon(_two_vertices(), source_positions, points,
                              steps=(16, 32))

    def test_a_degenerate_angular_gap_is_refused_rather_than_ordered(self):
        # Two punctures a nanoradian apart have no decided cyclic order, and
        # the two-cap carrier is exactly a cyclic order.  centre="origin" keeps
        # the failure inside this component instead of the centration.
        q = _ring().copy()
        q[3] = _unit((0.6 * np.cos(np.pi + 1e-9),
                      0.6 * np.sin(np.pi + 1e-9), 0.0))
        with self.assertRaisesRegex(GeometricRetractionError,
                                    "point order is undecided"):
            geometric_upsilon(None, None, [mesh.to_point(v) for v in q],
                              centre="origin")

    def test_nearest_margin_is_a_measurement(self):
        # The bisector margin is the gap between the two nearest punctures.  It
        # was hard-coded to 0.0 in the archived prototype; a consumer could not
        # tell "on a bisector" from "not measured".
        points = [mesh.to_point(v) for v in _ring()]
        out = geometric_upsilon(_two_vertices(), _ARCS[(1, 2)], points,
                                steps=(64, 128))
        row = out["paths"]["arc"]["nearest_margin"]
        self.assertGreater(row, 0.0)
        self.assertLess(row, out["paths"]["arc"]["closest_puncture"])
        # Independently: the same quantity, read off the arc samples.
        arc = __import__("pullback.geometric_retraction",
                         fromlist=["arc_points"]).arc_points(
            _ARCS[(1, 2)][0], _ARCS[(1, 2)][1], 128)
        q = out["centred_output"]
        chord = np.sort(np.linalg.norm(q[None, :, :] - arc[:, None, :], axis=2), axis=1)
        self.assertAlmostEqual(row, float(np.min(chord[:, 1] - chord[:, 0])), places=12)
        self.assertAlmostEqual(out["paths"]["arc"]["closest_puncture"],
                               float(np.min(chord[:, 0])), places=12)

    def test_hash_seed_does_not_change_the_retraction(self):
        # A dictionary-ordered label set must not leak into a homotopy class.
        script = (
            "import json, numpy as np\n"
            "from pullback import mesh\n"
            "from pullback.geometric_retraction import geometric_upsilon\n"
            "from pullback.ribbons import B, F, RibbonGraph\n"
            "def u(x):\n"
            "    x = np.asarray(x, float); return x/np.linalg.norm(x)\n"
            "q = np.array([u((0.6*np.cos(k*np.pi/2), 0.6*np.sin(k*np.pi/2), 0.0))"
            " for k in range(4)])\n"
            "arc = np.array([u((0.7, 0.4, 0.5)), u((0.8, 0.2, -0.2))])\n"
            "src = RibbonGraph({0: (('arc', F),), 1: (('arc', B),)})\n"
            "out = geometric_upsilon(src, arc, [mesh.to_point(v) for v in q],"
            " steps=(64, 128))\n"
            "print(json.dumps([out['target'].serialization(),"
            " out['paths']['arc']['cells'],"
            " out['paths']['arc']['transition_labels'],"
            " [[list(d) for d in e] for e in out['map'].chains.values()]],"
            " default=str))\n")
        runs = []
        for seed in ("1", "987654"):
            env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"),
                       PYTHONHASHSEED=seed)
            runs.append(subprocess.run([sys.executable, "-c", script], env=env,
                                       check=True, capture_output=True,
                                       text=True).stdout)
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(json.loads(runs[0])[2], [[0, 1]])


class TestRetentionUnit(unittest.TestCase):
    """The carrier is one commit unit; a botched split must stay importable.

    ``pullback.geometric_retraction`` is imported by ``marked_iteration`` and by
    two committed test modules, so a module-scope import of an uncommitted file
    would take the whole package surface down.  The unit is
    ``{equatorial_retraction.py, the equatorial hunks of geometric_retraction.py,
    the two-cap hunks of mesh.py, and these tests}``; if it is ever split by
    mistake, the module must degrade to the pre-carrier behaviour instead.
    """

    def test_import_survives_a_missing_two_cap_module(self):
        script = (
            "import sys\n"
            "sys.modules['pullback.equatorial_retraction'] = None\n"
            "import pullback.geometric_retraction as g\n"
            "assert g.eq is None, g.eq\n"
            "assert g._EQUATORIAL_ERRORS == (), g._EQUATORIAL_ERRORS\n"
            "try:\n"
            "    g.geometric_upsilon(None, None, [1.0, 1.0j, -1.0, -1.0j])\n"
            "except g.GeometricRetractionError as exc:\n"
            "    print(exc)\n")
        env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"),
                   PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-c", script], env=env,
                                check=True, capture_output=True, text=True)
        self.assertIn("sphere centre", result.stdout)


if __name__ == "__main__":
    unittest.main()
