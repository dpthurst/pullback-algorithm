"""The Julia-set kernel and the sphere view: known-answer checks.

The answers here are mathematical, not recordings:
- the map strings the library carries must evaluate to the maps they name
  (theta's f(2) = -5/3, the barycentric F(2) = 1, (2z^2-4)/z^2 at 2 = 1);
- for f(z) = z^2 the attracting cycle is {0}, the unit circle is the Julia
  set, and z = 1 is a *repelling* fixed point -- it must never count as
  settled, or the picture would call the Julia set interior;
- for f(z) = z^2 - 1 the critical orbit is 0 -> -1 -> 0, the attracting
  cycle {0, -1};
- depth grows towards the Julia set: 0.5 settles sooner than 0.9 for z^2;
- the sphere view is tested on a synthetic half-and-half texture, so the
  geometry (equator across the middle, yaw spins, pitch tilts) is checked
  independently of any map.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys  # noqa: E402
import unittest  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, "src")

from pullback.julia import (  # noqa: E402
    attracting_cycles,
    basin_counts,
    colourise,
    escape_counts,
    parse_map,
    sphere_grid,
    sphere_texture,
)

from PySide6.QtWidgets import QApplication  # noqa: E402

from pullback.app.sphere import SphereView  # noqa: E402


def evaluate(text, z):
    num, den = parse_map(text)
    return np.polyval(num, z) / np.polyval(den, z)


class TestMapStrings(unittest.TestCase):
    def test_theta(self):
        self.assertAlmostEqual(abs(evaluate("(1+z^2)/(1-z^2)", 2.0) + 5 / 3), 0.0, places=12)

    def test_barycentric(self):
        self.assertAlmostEqual(abs(evaluate("4/27 (z^2-z+1)^3/(z(z-1))^2", 2.0) - 1.0), 0.0, places=12)

    def test_implicit_multiplication_and_power(self):
        self.assertAlmostEqual(abs(evaluate("(2z^2-4)/z^2", 2.0) - 1.0), 0.0, places=12)

    def test_imaginary_unit(self):
        want = -1 - 2j / 3
        self.assertAlmostEqual(abs(evaluate("(1 + z^2/(1-i))/(1-z^2)", 2.0) - want), 0.0, places=12)

    def test_rabbit_has_no_number(self):
        with self.assertRaises(ValueError):
            parse_map("Douady rabbit z^2+c")


class TestEscape(unittest.TestCase):
    """The polynomial escape kernel, kept as the simple case."""

    def test_unit_circle_never_escapes(self):
        z = np.array([1, -1, 1j, -1j, np.exp(0.7j)])
        self.assertTrue((escape_counts([1, 0], [1], z, maxiter=100) == 100).all())

    def test_outside_escapes(self):
        counts = escape_counts([1, 0, 0], [1], np.array([2.0, -2.0, 3.0j]), maxiter=100)
        self.assertTrue((counts < 100).all() and (counts >= 1).all())


class TestBasin(unittest.TestCase):
    def test_z2_attracting_cycle_is_zero(self):
        cycles = attracting_cycles([1, 0, 0], [1])
        self.assertEqual(len(cycles), 1)
        self.assertLess(abs(cycles[0][0]), 1e-6)

    def test_z2_depths(self):
        z = np.array([0.0, 0.5, 0.9, 1.0, 1j, 2.0])
        d = basin_counts([1, 0, 0], [1], z, maxiter=100)
        self.assertEqual(int(d[0]), 1)          # fixed point 0
        self.assertLess(int(d[1]), int(d[2]))   # deeper towards the Julia set
        self.assertEqual(int(d[3]), 100)        # 1 is a repelling fixed point
        self.assertEqual(int(d[4]), 100)        # i is on the Julia set
        self.assertLess(int(d[5]), 100)         # 2 runs off to infinity

    def test_z2_minus_1_has_the_two_cycle(self):
        cycles = attracting_cycles([1, 0, -1], [1])
        self.assertEqual(len(cycles), 1)
        self.assertEqual(len(cycles[0]), 2)
        d = basin_counts([1, 0, -1], [1], np.array([0.0, -1.0, 2.0]), maxiter=100)
        self.assertLess(int(d[0]), 5)
        self.assertLess(int(d[1]), 5)


class TestSphereTexture(unittest.TestCase):
    def test_grid_is_open_in_theta(self):
        theta, phi, Z = sphere_grid(8, 16)
        self.assertTrue((theta > 0).all() and (theta < np.pi).all())
        self.assertEqual(Z.shape, (8, 16))

    def test_z2_depth_is_deepest_at_the_equator(self):
        counts = sphere_texture([1, 0, 0], [1], 48, 96, maxiter=200)
        equator = np.median(counts[22:26, :])
        pole = np.median(counts[:3, :])
        self.assertGreater(float(equator), float(pole))

    def test_z2_even_map_has_phi_symmetry(self):
        counts = sphere_texture([1, 0, 0], [1], 16, 32, maxiter=100)
        self.assertTrue(np.array_equal(counts, np.roll(counts, 16, axis=1)))

    def test_theta_map_settles_everywhere_but_with_structure(self):
        # (1+z^2)/(1-z^2): one attracting cycle, containing infinity, so no
        # finite cycle is found and every grid point settles -- what makes
        # the picture is the depth, not an "exterior".
        num, den = parse_map("(1+z^2)/(1-z^2)")
        self.assertEqual(attracting_cycles(num, den), [])
        counts = sphere_texture(num, den, 48, 96, maxiter=200)
        self.assertGreater(len(np.unique(counts)), 10)
        self.assertEqual(int(counts.min()), 4)      # the 2-cycle is reached fast

    def test_colourise_marks_the_unsettled(self):
        rgb = colourise(np.array([[0, 50, 100]]), maxiter=100)
        self.assertEqual(tuple(rgb[0, 2]), (12, 12, 38))
        self.assertNotEqual(tuple(rgb[0, 0]), tuple(rgb[0, 2]))


def image_array(img):
    """RGB pixels of a QImage as an (h, w, 3) uint8 array (rows may be padded)."""
    h, w, bpl = img.height(), img.width(), img.bytesPerLine()
    a = np.frombuffer(img.constBits(), dtype=np.uint8, count=bpl * h)
    # copy: the view does not keep the QImage alive, so two images made in
    # sequence would otherwise be read through the same freed buffer
    return a.reshape(h, bpl)[:, : w * 3].reshape(h, w, 3).copy()


class TestSphereView(unittest.TestCase):
    """Geometry on a synthetic texture: the northern half never settles."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.tex = np.full((48, 96), 10, dtype=np.int32)
        cls.tex[:24, :] = 150
        cls.inside = (12, 12, 38)

    def test_renders_and_is_not_blank(self):
        view = SphereView(self.tex, maxiter=150)
        img = view.render_image(160, 160)
        self.assertFalse(img.isNull())
        colours = {tuple(c) for c in image_array(img)[::7, ::7].reshape(-1, 3)}
        self.assertGreater(len(colours), 1)

    def test_north_is_up_and_is_the_unsettled_colour(self):
        view = SphereView(self.tex, maxiter=150)
        a = image_array(view.render_image(160, 160))
        self.assertEqual(tuple(int(v) for v in a[40, 80]), self.inside)   # v ~ +0.5
        self.assertNotEqual(tuple(int(v) for v in a[120, 80]), self.inside)

    def test_yaw_does_not_change_a_phi_constant_texture(self):
        view = SphereView(self.tex, maxiter=150)
        a = image_array(view.render_image(120, 120))
        view.yaw = np.pi / 3
        b = image_array(view.render_image(120, 120))
        self.assertTrue(np.array_equal(a, b))

    def test_pitch_tilts_the_picture(self):
        view = SphereView(self.tex, maxiter=150)
        a = image_array(view.render_image(120, 120))
        view.pitch = np.pi / 2
        b = image_array(view.render_image(120, 120))
        self.assertGreater(int(np.count_nonzero(a != b)), 100)

    def test_aspect_ratio_is_preserved(self):
        # Regression (author, 2026-10-25): normalising x and y separately
        # stretched the sphere into an ellipse on a wide panel.
        view = SphereView(self.tex, maxiter=150)
        for w, h in ((300, 180), (180, 300), (240, 240)):
            a = image_array(view.render_image(w, h))
            mask = np.any(a != np.array([24, 24, 28], np.uint8), axis=2)
            ys, xs = np.nonzero(mask)
            self.assertEqual(int(xs.max() - xs.min()), int(ys.max() - ys.min()),
                             (w, h))

    def test_the_picture_does_not_depend_on_the_window_shape(self):
        view = SphereView(self.tex, maxiter=150)
        wide = image_array(view.render_image(300, 180))
        square = image_array(view.render_image(180, 180))
        # the 180x180 centre of the wide render is the square render
        self.assertTrue(np.array_equal(wide[:, 60:240], square))

    @staticmethod
    def _tangents(S, x, y):
        """Screen -> sphere tangent vectors at one pixel, five-point central.

        A one-pixel step with the two-point formula carries O(h^2) curvature
        error (5e-5 in the dot product at size 201), which is the stencil,
        not the projection.
        """
        du = (S[:, y, x - 2] - 8 * S[:, y, x - 1]
              + 8 * S[:, y, x + 1] - S[:, y, x + 2]) / 12.0
        dv = (S[:, y - 2, x] - 8 * S[:, y - 1, x]
              + 8 * S[:, y + 1, x] - S[:, y + 2, x]) / 12.0
        return du, dv

    def test_orthographic_is_not_conformal_away_from_the_centre(self):
        # u ~ 0.60, v ~ 0.25 (deliberately asymmetric: at u = v the ratio is
        # 1 by symmetry, which is why the first version of this test passed
        # vacuously).
        view = SphereView(self.tex, maxiter=150)
        _inside, S = view._unit_points(401, 401)
        du, dv = self._tangents(S, 320, 150)
        self.assertGreater(abs(float(np.linalg.norm(du) / np.linalg.norm(dv)) - 1.0), 0.01)

    def test_stereographic_is_conformal(self):
        view = SphereView(self.tex, maxiter=150)
        view.set_projection("stereographic")
        _inside, S = view._unit_points(401, 401)
        du, dv = self._tangents(S, 320, 150)
        nu, nv = float(np.linalg.norm(du)), float(np.linalg.norm(dv))
        self.assertAlmostEqual(nu / nv, 1.0, places=6)
        self.assertLess(abs(float(np.dot(du, dv)) / (nu * nv)), 1e-6)

    def test_without_texture_it_still_renders(self):
        view = SphereView()
        self.assertFalse(view.render_image(80, 80).isNull())
        self.assertFalse(view.has_texture())


if __name__ == "__main__":
    unittest.main()
