"""Headless check of the desktop app (plan items 270, 280).

Qt runs with QT_QPA_PLATFORM=offscreen, so the real window is constructed,
a session opened, and every panel compared with the engine's values.  There
is no display on this machine: this checks the wiring and the view models,
not that the layout looks right -- the first visual check is the author
opening it.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtWidgets import QApplication  # noqa: E402

from pullback import Session  # noqa: E402
from pullback.app import panels  # noqa: E402
from pullback.app.main import MainWindow  # noqa: E402

THETA = ROOT / "sessions" / "theta.json"


class TestPanels(unittest.TestCase):
    """The view models, with no toolkit involved at all."""

    @classmethod
    def setUpClass(cls):
        cls.s = Session.load(THETA)

    def test_summary(self):
        text = panels.summary(self.s)
        self.assertIn("theta", text)
        self.assertIn("degree 2", text)
        self.assertIn("all checks pass", text)
        self.assertIn("hyperbolic type", text)

    def test_portrait(self):
        headers, rows, note = panels.portrait(self.s)
        self.assertEqual(headers[0], "cover face")
        self.assertEqual(len(rows), 4)
        self.assertEqual(sorted(r[3] for r in rows), ["1", "1", "2", "2"])
        self.assertIn("[1, 1, 2, 2]", note)

    def test_automaton(self):
        text = panels.automaton(self.s)
        self.assertIn("letters      p, r", text)
        self.assertIn("tree         q", text)
        self.assertIn("admissible", text)
        self.assertIn("5 words, closed", text)
        self.assertIn("2, 4, 8, 32, 256, 16384", text)

    def test_panel_level_cap_keeps_big_images_out(self):
        # Barycentric's level 4 has 1296 points and took 50 s in Schreier-Sims;
        # the panel stops at 81 points, while the API has no cap.
        s = Session.load(ROOT / "sessions" / "barycentric.json")
        text = panels.automaton(s)
        self.assertIn("level orders 6, 648", text)
        self.assertIn("stops at 81 points", text)
        self.assertEqual(s.level_order(3), 1028294561267712)

    def test_map_picture(self):
        # theta's map has one attracting cycle (through infinity), so every
        # grid point settles; the picture is the depth, and it must have
        # structure rather than be a single block.
        label, counts = panels.map_picture(self.s)
        self.assertIn("(1+z^2)/(1-z^2)", label)
        self.assertEqual(counts.shape, (96, 192))
        self.assertGreater(len(set(counts.ravel().tolist())), 10)

    def test_map_picture_refuses_a_non_numeric_map(self):
        s = Session.load(ROOT / "sessions" / "rabbit.json")
        label, counts = panels.map_picture(s)
        self.assertIsNone(counts)
        self.assertIn("not a rational function", label)

    def test_energies(self):
        headers, rows, note = panels.energies(self.s, p=2, n=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0], ("1", "1", "1"))
        self.assertIn("upper bounds", note)


class TestWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_library_is_listed_without_opening(self):
        w = MainWindow()
        self.assertEqual(w.session_list.count(), 15)
        self.assertIsNone(w.session)

    def test_open_theta(self):
        w = MainWindow()
        self.assertTrue(w.open_session(THETA))
        self.assertIn("hyperbolic type", w.summary_text())
        rows = w.portrait_rows()
        self.assertEqual(len(rows), 4)
        self.assertEqual(sorted(r[3] for r in rows), ["1", "1", "2", "2"])
        self.assertIn("5 words, closed", w.automaton_text())
        self.assertEqual(w.tabs.count(), 5)

    def test_energy_button_runs_the_engine(self):
        w = MainWindow()
        w.open_session(THETA)
        w.run_button.click()                      # p = 2, n = 1
        self.assertEqual(w.energy_rows(), [["1", "1", "1"]])

    def test_showing_the_window_keeps_the_open_session(self):
        # Regression: a QListWidget selects its first row when shown, and
        # the window used to open that row's session over the opened one.
        w = MainWindow()
        w.open_session(THETA)
        w.show()
        self.app.processEvents()
        self.assertEqual(w.session.name, "theta")
        self.assertTrue(w.sphere.has_texture())
        self.assertEqual(w.session_list.currentItem().text(), "theta")
        w.close()

    def test_projection_toggle(self):
        w = MainWindow()
        w.open_session(THETA)
        self.assertEqual(w.sphere.projection, "orthographic")
        w.proj_combo.setCurrentIndex(1)
        self.assertEqual(w.sphere.projection, "stereographic")
        self.assertFalse(w.sphere.render_image(120, 120).isNull())
        w.proj_combo.setCurrentIndex(0)
        self.assertEqual(w.sphere.projection, "orthographic")

    def test_bad_file_is_reported_not_raised(self):
        scratch = ROOT / ".scratch" / "app-test"
        scratch.mkdir(parents=True, exist_ok=True)
        bad = scratch / "broken.json"
        bad.write_text('{"graph": "apqr bRQP", "cover": ')
        try:
            w = MainWindow()
            self.assertFalse(w.open_session(bad))
            self.assertIn("error opening", w.summary_text())
            self.assertEqual(w.portrait_rows(), [])
            self.assertIsNone(w.session)
        finally:
            bad.unlink()

    def test_map_tab_shows_the_sphere(self):
        w = MainWindow()
        w.open_session(THETA)
        self.assertTrue(w.sphere.has_texture())
        self.assertIn("(1+z^2)/(1-z^2)", w.map_label.text())
        self.assertFalse(w.sphere.render_image(120, 120).isNull())

    def test_map_tab_clears_for_a_session_without_a_map(self):
        w = MainWindow()
        w.open_session(THETA)
        w.open_session(ROOT / "sessions" / "barbell.json")
        self.assertFalse(w.sphere.has_texture())
        self.assertIn("no map", w.map_label.text())

    def test_refusal_is_shown_and_the_rest_still_renders(self):
        # grigorchuk has no ribbon presentation: the type is refused, but
        # the automaton (its wreath recursion) still exists and is shown.
        w = MainWindow()
        self.assertTrue(w.open_session(ROOT / "sessions" / "grigorchuk.json"))
        self.assertIn("type         refused", w.summary_text())
        self.assertIn("letters", w.automaton_text())


if __name__ == "__main__":
    unittest.main()
