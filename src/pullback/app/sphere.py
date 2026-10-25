"""A sphere view for the Julia set on the Riemann sphere.

CPU rasteriser, no OpenGL: Qt3D/QtQuick3D/PyOpenGL are absent from this
environment and `QOpenGLWidget` gets no context under the offscreen platform,
so a GL view could not be verified here at all.  Instead, each pixel of the
disc is one analytic ray-sphere intersection, carried to spherical
coordinates and looked up in a precomputed escape-count texture (`julia.py`).
One intersection per pixel, no reflection and no shading: this is not a ray
tracer.

The widget is a normal `QWidget`, so `render_image()` returns a `QImage` and
the headless tests can check the picture without a display.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QWidget

from ..julia import colourise


class SphereView(QWidget):
    """Shows an escape-count texture on a sphere; drag to rotate, wheel to zoom."""

    def __init__(self, counts=None, maxiter=200, parent=None):
        super().__init__(parent)
        self._counts = None
        self._rgb = None
        self.maxiter = maxiter
        self.yaw = 0.0
        self.pitch = 0.0
        self.zoom = 1.0
        # "orthographic": the ball, one ray per pixel, perspective
        # foreshortening near the limb (circles there are ellipses).
        # "stereographic": the conformal chart -- screen plane -> sphere by
        # inverse stereographic projection, circles stay circles, the sphere
        # is drawn minus the point at the far pole.
        self.projection = "orthographic"
        self._last_image = None
        self._drag = None
        if counts is not None:
            self.set_texture(counts, maxiter)

    # ---------------------------------------------------------------- data
    def set_texture(self, counts, maxiter=None):
        if maxiter is not None:
            self.maxiter = maxiter
        self._counts = np.asarray(counts)
        self._rgb = colourise(self._counts, self.maxiter)
        self._last_image = None
        self.update()

    def clear(self):
        self._counts = None
        self._rgb = None
        self._last_image = None
        self.update()

    def has_texture(self):
        return self._counts is not None

    # ------------------------------------------------------------ geometry
    def set_projection(self, name):
        if name not in ("orthographic", "stereographic"):
            raise ValueError(f"unknown projection {name!r}")
        self.projection = name
        self._last_image = None
        self.update()

    def _unit_points(self, width, height):
        """(inside, S) with S the (3, height, width) unit vectors.

        The two screen axes share ONE scale, 2/min(width, height): separate
        scales stretched the sphere into an ellipse on a wide panel (the
        picture was not conformal, circles off round -- author, 2026-10-25).
        """
        ys, xs = np.mgrid[0:height, 0:width]
        scale = 2.0 / min(width, height)
        u = ((xs + 0.5) - width / 2.0) * scale * self.zoom
        v = -((ys + 0.5) - height / 2.0) * scale * self.zoom
        r2 = u * u + v * v
        if self.projection == "stereographic":
            den = 1.0 + r2
            x, y, z = 2.0 * u / den, 2.0 * v / den, (1.0 - r2) / den
            inside = np.ones(u.shape, dtype=bool)
        else:
            inside = r2 <= 1.0
            x, y = u, v
            z = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        x1 = x
        y1 = y * cp - z * sp
        z1 = y * sp + z * cp
        x2 = x1 * cy + z1 * sy
        z2 = -x1 * sy + z1 * cy
        # camera frame (x right, y up, z toward viewer) -> world (x toward
        # viewer, y right, z up at yaw = pitch = 0)
        return inside, np.stack([z2, x2, y1])

    def _spherical(self, width, height):
        """(inside, theta, phi) for the screen pixels."""
        inside, S = self._unit_points(width, height)
        theta = np.arccos(np.clip(S[2], -1.0, 1.0))
        phi = np.mod(np.arctan2(S[1], S[0]), 2.0 * np.pi)
        return inside, theta, phi

    def render_image(self, width=480, height=480):
        """The current picture as a QImage (background outside the disc)."""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :] = (24, 24, 28)
        if self._rgb is None:
            return QImage(img.data, width, height, 3 * width,
                          QImage.Format.Format_RGB888).copy()
        inside, theta, phi = self._spherical(width, height)
        n_theta, n_phi = self._rgb.shape[:2]
        i = np.clip((theta / np.pi * n_theta).astype(np.int32), 0, n_theta - 1)
        j = np.clip((phi / (2 * np.pi) * n_phi).astype(np.int32), 0, n_phi - 1)
        img[inside] = self._rgb[i[inside], j[inside]]
        return QImage(np.ascontiguousarray(img).data, width, height, 3 * width,
                      QImage.Format.Format_RGB888).copy()

    # -------------------------------------------------------------- events
    def paintEvent(self, _event):
        if self._last_image is None or self._last_image.size() != self.size():
            self._last_image = self.render_image(max(self.width(), 1),
                                                 max(self.height(), 1))
        QPainter(self).drawImage(QPoint(0, 0), self._last_image)

    def resizeEvent(self, event):
        self._last_image = None
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = event.position()

    def mouseMoveEvent(self, event):
        if self._drag is None:
            return
        delta = event.position() - self._drag
        self._drag = event.position()
        self.yaw = (self.yaw + delta.x() * 0.01) % (2 * np.pi)
        self.pitch = float(np.clip(self.pitch + delta.y() * 0.01,
                                   -np.pi / 2, np.pi / 2))
        self._last_image = None
        self.update()

    def mouseReleaseEvent(self, _event):
        self._drag = None

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        self.zoom = float(np.clip(self.zoom * (0.9 ** steps), 0.5, 8.0))
        self._last_image = None
        self.update()
