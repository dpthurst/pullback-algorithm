"""Geometry and homotopy certificates for the output retraction ``upsilon``."""
import os
import unittest

import numpy as np
import pytest

from pullback import mesh
from pullback.geometric_retraction import (
    GeometricRetractionError,
    _face_hits,
    arc_points,
    cell_sequence,
    geometric_upsilon,
    refined_spine,
    refined_upsilon,
    refined_vertex_positions,
)
from pullback.ribbons import B, F, RibbonGraph, tighten
from pullback.session import Session
from pullback.validate import is_pi1_surjective, rose_words, spanning_tree

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RABBIT_TRI_OF_VERTEX = [5, 4, 7, 6, 9, 8, 11, 10, 0, 1, 2, 3]


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _tetrahedron():
    q = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
                 dtype=float) / np.sqrt(3.0)
    return q, [mesh.to_point(v) for v in q]


def _closed_class(walk):
    """Cyclic free reduction, minimised over rotations and reversal."""
    w = tuple(tighten(tuple(walk)))
    if not w:
        return ()
    inv = tuple((d[0], -d[1]) for d in reversed(w))
    candidates = [w[k:] + w[:k] for k in range(len(w))]
    candidates += [inv[k:] + inv[:k] for k in range(len(inv))]
    return min(candidates, key=lambda c: tuple(str(d) for d in c))


def _source_face_punctures(source, edges):
    out = {}
    for face, walk in enumerate(source.faces()):
        common = None
        for dart in walk:
            ends = set(edges[dart[0]])
            common = ends if common is None else common & ends
        assert common is not None and len(common) == 1
        out[face] = next(iter(common))
    return out


class TestGeometricRetraction(unittest.TestCase):
    """Run under the repository's unittest gate as well as under pytest."""

    @classmethod
    def setUpClass(cls):
        session = Session.load(os.path.join(REPO, "sessions", "rabbit.json"))
        refined = session.refined_geometry(seed=0)
        cert = refined.step(refined.points)
        upsilon = refined_upsilon(refined, cert, steps=(256, 512))
        cls.rabbit = (refined, cert, upsilon)

    def test_short_arc_has_the_declared_endpoints_and_midpoint(self):
        a, b = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        arc = arc_points(a, b, 4)
        assert arc.shape == (5, 3)
        np.testing.assert_allclose(arc[0], a, atol=1e-15)
        np.testing.assert_allclose(arc[-1], b, atol=1e-15)
        np.testing.assert_allclose(np.linalg.norm(arc, axis=1), 1.0, atol=1e-15)
        np.testing.assert_allclose(arc[2], _unit(a + b), atol=1e-15)

    def test_short_arc_refuses_coincident_and_antipodal_endpoints(self):
        a = np.array([1.0, 0.0, 0.0])
        with pytest.raises(GeometricRetractionError, match="coincident"):
            arc_points(a, a, 8)
        with pytest.raises(GeometricRetractionError, match="antipodal"):
            arc_points(a, -a, 8)

    def test_arc_through_a_puncture_is_refused_even_when_all_samples_miss_it(self):
        q, points = _tetrahedron()
        u = _unit(np.cross(q[0], np.array([0.0, 0.0, 1.0])))
        angle = 0.8
        a = np.cos(angle) * q[0] + np.sin(angle) * u
        b = np.cos(angle) * q[0] - np.sin(angle) * u
        # Neither odd grid contains t=1/2, where the minor arc crosses q[0].
        for steps in (33, 65):
            arc = arc_points(a, b, steps)
            assert np.min(np.linalg.norm(arc - q[0], axis=1)) > 1e-3
        source = RibbonGraph({0: (("e", F),), 1: (("e", B),)})
        with pytest.raises(GeometricRetractionError, match="puncture 0"):
            geometric_upsilon(source, np.array([a, b]), points, steps=(33, 65))

    def test_duplicate_output_punctures_are_refused_before_construction(self):
        points = [0.0, 0.0, 1.0, -1.0]
        with pytest.raises(GeometricRetractionError, match="punctures 0 and 1"):
            geometric_upsilon(None, None, points)

    def test_degenerate_target_data_have_a_typed_refusal(self):
        with pytest.raises(GeometricRetractionError, match="Delaunay/Voronoi"):
            geometric_upsilon(None, None, [1.0, 1.0j, -1.0, -1.0j])

    def test_delaunay_facet_membership_and_a_single_cell_arc(self):
        q, points = _tetrahedron()
        faces = mesh.delaunay_faces(points)
        a = _unit(q[0] + q[1] + q[2])
        assert set(_face_hits(a, q, faces)) == {
            i for i, face in enumerate(faces) if set(face) == {0, 1, 2}}
        b = _unit(2 * q[0] + q[1] + q[2])
        cells, margin, clearance = cell_sequence(a, b, q, faces, steps=32)
        assert cells == (next(i for i, f in enumerate(faces) if set(f) == {0, 1, 2}),)
        # A Delaunay face centre is equidistant from its three primal vertices;
        # zero nearest-puncture margin is therefore valid here, not a cell hit failure.
        assert margin >= -1e-15
        assert clearance > 0.0

    def test_chordal_vector_output_points_use_the_same_delaunay_route(self):
        q, _points = _tetrahedron()
        a = _unit(q[0] + q[1] + q[2])
        b = _unit(2 * q[0] + q[1] + q[2])
        source = RibbonGraph({0: (("e", F),), 1: (("e", B),)})
        out = geometric_upsilon(source, np.array([a, b]), q, steps=(32, 64))
        assert (len(out["target"].vertices()), len(out["target"].edge_list()),
                len(out["target"].faces())) == (4, 6, 4)
        assert out["map"].chains["e"] == ()

    def test_cocircular_polygon_membership_uses_the_whole_block(self):
        q = []
        for k in range(4):
            angle = 2 * np.pi * k / 4
            q.append([np.sqrt(1 - 0.2**2) * np.cos(angle),
                      np.sqrt(1 - 0.2**2) * np.sin(angle), -0.2])
        q.append([0.0, 0.0, 1.0])
        q = np.asarray(q)
        points = [mesh.to_point(v) for v in q]
        faces = mesh.delaunay_faces(points)
        polygon = next(i for i, face in enumerate(faces) if len(face) == 4)
        # Both points are outside the first triangle of the polygon's fan.  The
        # old A,B,C-only test raised by unpacking four vertices, and would have
        # rejected these points even if it had ignored the fourth.
        a = _unit(q[:4].sum(axis=0) + 0.2 * q[3])
        b = _unit(q[:4].sum(axis=0) + 0.3 * q[3])
        assert _face_hits(a, q, faces) == [polygon]
        assert _face_hits(b, q, faces) == [polygon]
        cells, _margin, clearance = cell_sequence(a, b, q, faces, steps=32)
        assert cells == (polygon,)
        assert clearance > 0.0

    def test_central_cell_route_refuses_a_hull_not_containing_the_origin(self):
        q = []
        for k in range(4):
            angle = 2 * np.pi * k / 4
            q.append([np.sqrt(1 - 0.2**2) * np.cos(angle),
                      np.sqrt(1 - 0.2**2) * np.sin(angle), 0.2])
        q.append([0.0, 0.0, 1.0])
        points = [mesh.to_point(v) for v in np.asarray(q)]
        # This is a valid Delaunay/Voronoi configuration, but rays from the origin
        # can meet two facets.  It needs a re-chart, not an arbitrary facet choice.
        with pytest.raises(GeometricRetractionError, match="sphere centre"):
            geometric_upsilon(None, None, points)

    def test_edge_ambiguity_uses_one_shared_puncture_pair_transition(self):
        q, points = _tetrahedron()
        faces = mesh.delaunay_faces(points)
        midpoint = _unit(q[0] + q[1])
        hits = tuple(_face_hits(midpoint, q, faces))
        assert len(hits) == 2
        assert all({0, 1} <= set(faces[i]) for i in hits)
        destination_face = hits[1]
        destination = _unit(sum((q[i] for i in faces[destination_face]),
                                np.zeros(3)))
        source = RibbonGraph({0: (("e", F),), 1: (("e", B),)})
        out = geometric_upsilon(source, np.array([midpoint, destination]), points,
                                steps=(32, 64))
        assert out["vertex_cells"][0] == hits
        assert out["vertex_images"] == {0: hits[0], 1: hits[1]}
        assert len(out["paths"]["e"]["darts"]) == 1
        dart = out["paths"]["e"]["darts"][0]
        assert dart[0] == (0, 1)
        assert out["target"].src(dart) == hits[0]
        assert out["target"].tgt(dart) == hits[1]

    def test_refined_positions_follow_canonical_dual_faces_not_triangle_order(self):
        refined, cert, upsilon = self.rabbit
        # This permutation is a property of K2's canonical face ordering.  Using
        # range(12) here was the original, homotopy-changing indexing defect.
        tri_of_vertex = RABBIT_TRI_OF_VERTEX
        assert tri_of_vertex != list(range(12))
        assert refined_spine(refined).vertices() == list(range(12))
        expected = []
        for k in tri_of_vertex:
            tri = refined.K2["tris"][k][0]
            expected.append(_unit(cert["x"][list(tri)].sum(axis=0)))
        np.testing.assert_allclose(
            refined_vertex_positions(refined, cert["x"]), np.array(expected),
            atol=2e-15, rtol=0.0)

    def test_declared_source_edges_stay_in_their_two_lifted_cells(self):
        refined, cert, _upsilon = self.rabbit
        source = refined_spine(refined)
        positions = refined_vertex_positions(refined, cert["x"])
        index = {v: i for i, v in enumerate(source.vertices())}
        canonical_of_triangle = {k: v for v, k in enumerate(RABBIT_TRI_OF_VERTEX)}
        checked = 0
        for edge in source.edge_list():
            a, b = source.edges[edge]
            raw = cell_sequence(positions[index[a]], positions[index[b]], cert["x"],
                                refined.oriented_faces(), steps=64)[0]
            cells = tuple(tuple(canonical_of_triangle[i] for i in row)
                          if isinstance(row, tuple) else canonical_of_triangle[row]
                          for row in raw)
            flat = [i for row in cells for i in (row if isinstance(row, tuple) else (row,))]
            assert flat[0] == a and flat[-1] == b, (edge, cells, a, b)
            assert set(flat) <= {a, b}, (edge, cells, a, b)
            checked += 1
        assert checked == 18

    def test_real_rabbit_graph_map_and_sampling_certificate(self):
        refined, cert, upsilon = self.rabbit
        source, target = upsilon["source"], upsilon["target"]
        assert (len(source.vertices()), len(source.edge_list()), len(source.faces())) == \
            (12, 18, 8)
        assert (len(target.vertices()), len(target.edge_list()), len(target.faces())) == \
            (4, 6, 4)
        assert upsilon["sampling_levels"] == (256, 512)
        assert cert["embedded"] and cert["injective"]
        assert cert["excess"] == 0 and cert["residual"] < 1e-12
        assert len(set(cert["config"][i].tobytes()
                       for i in range(len(cert["config"])))) == 4
        graph_map = upsilon["map"]
        assert graph_map.source is source and graph_map.target is target
        assert is_pi1_surjective(graph_map, target)
        for edge in source.edge_list():
            chain = graph_map.chains[edge]
            assert chain == upsilon["paths"][edge]["darts"]
            a, b = source.edges[edge]
            if chain:
                assert target.src(chain[0]) == graph_map.vertex_images[a]
                assert target.tgt(chain[-1]) == graph_map.vertex_images[b]
            else:
                assert graph_map.vertex_images[a] == graph_map.vertex_images[b]
        # These based words independently agree with the radial-exit retraction;
        # individual edge chains may differ by the permitted vertex slides.
        target_tree, target_gens = spanning_tree(target)
        source_tree, source_gens = spanning_tree(source)
        got = [tuple((target_gens.index(d[0]), d[1]) for d in word)
               for word in rose_words(graph_map, target_tree, source_tree, source_gens)]
        assert got == [
            ((1, B), (0, F), (2, F)),
            (),
            (),
            ((2, B), (0, B)),
            ((2, B),),
            (),
            ((2, B), (0, B), (1, F)),
        ]

    def test_real_rabbit_peripheral_classes(self):
        refined, _cert, upsilon = self.rabbit
        graph_map = upsilon["map"]
        source, target = graph_map.source, graph_map.target
        face_puncture = _source_face_punctures(source, refined.K2["edges"])
        lifted_to_label = {v: p for p, v in refined.labels.items()}
        active = {face: lifted_to_label[p]
                  for face, p in face_puncture.items() if p in lifted_to_label}
        assert active == {1: 2, 5: 0, 6: 1, 7: 3}
        target_point = upsilon["target_face_points"]
        target_face = {point: face for face, point in target_point.items()}
        for face, walk in enumerate(source.faces()):
            image = []
            for dart in walk:
                image.extend(graph_map.image_dart(dart))
            got = _closed_class(image)
            if face in active:
                want = _closed_class(target.faces()[target_face[active[face]]])
            else:
                want = ()
            assert got == want, (face, active.get(face), got, want)


if __name__ == "__main__":
    unittest.main()
