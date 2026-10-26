"""The moduli chart for four points, and Möbius normalisation.

For a Thurston map with |P| = 4, Teichmüller space Teich(S^2 - P) is one
complex dimension (the author's decision of 2026-10-25: work with |P| = 4
first).  A configuration of four points on the Riemann sphere is recorded by
its cross-ratio, up to the six classical substitutions, and PSL(2, C) acts by
fractional linear maps; this module is the coordinate system the pullback
iteration will live in.

Conventions: for four distinct points,

    cr(a, b, c, d) = (a - c)(b - d) / ((a - d)(b - c)),

so the standard normalisation holds in the author's form: cr(inf, 0; 1, z) = z
(check the slot order -- with 0 and 1 first and infinity third the same
formula gives (z - 1)/z, which is one of the six substitutes and describes
the same configuration), and the four points are concyclic (or collinear)
iff the cross-ratio is real.  A Möbius map changes the value by one of the
six substitutions z, 1/z, 1-z, 1/(1-z), z/(z-1), (z-1)/z, so the chart is the
quotient of C - {0, 1} by S_3: two cross-ratios related by one of these
describe the same configuration.

Infinite entries are handled by the limits, not by floating point: a point of
P is frequently infinity (every polynomial has it), and `(inf - c)` is nan.
"""

from __future__ import annotations

import numpy as np

SIX = (
    lambda z: z,
    lambda z: 1.0 / z,
    lambda z: 1.0 - z,
    lambda z: 1.0 / (1.0 - z),
    lambda z: z / (z - 1.0),
    lambda z: (z - 1.0) / z,
)


def is_inf(z):
    z = complex(z)
    return bool(np.isinf(z.real) or np.isinf(z.imag))


def cross_ratio(a, b, c, d):
    """cr(a, b, c, d) = (a-c)(b-d)/((a-d)(b-c)); at most one entry may be infinite."""
    a, b, c, d = complex(a), complex(b), complex(c), complex(d)
    infs = [is_inf(v) for v in (a, b, c, d)]
    if sum(infs) > 1:
        raise ValueError("cross-ratio needs at most one infinite point")
    if infs[0]:                       # a = inf
        return (b - d) / (b - c)
    if infs[1]:                       # b = inf
        return (a - c) / (a - d)
    if infs[2]:                       # c = inf
        return (b - d) / (a - d)
    if infs[3]:                       # d = inf
        return (a - c) / (b - c)
    return ((a - c) * (b - d)) / ((a - d) * (b - c))


def normalise(points):
    """The fourth point after the Möbius map sending the first three to 0, 1, inf."""
    p0, p1, p2, p3 = points
    return cross_ratio(p0, p1, p2, p3)


def apply_mobius(m, z):
    """(a z + b)/(c z + d) for m = (a, b, c, d), with infinity respected."""
    a, b, c, d = m
    if is_inf(z):
        return complex(np.inf) if c == 0 else a / c
    z = complex(z)
    den = c * z + d
    return complex(np.inf) if den == 0 else (a * z + b) / den


def orbit(z):
    """The six cross-ratio values belonging to one configuration."""
    return [f(z) for f in SIX]


def same_configuration(z, w, tol=1e-9):
    """Do two cross-ratios describe the same four points up to Möbius?"""
    if abs(z) < tol or abs(z - 1.0) < tol:
        raise ValueError("degenerate cross-ratio")
    for v in orbit(z):
        if abs(v - w) < tol:
            return True
        if abs(v) > 1 / tol and abs(w) > 1 / tol and abs(1 / v - 1 / w) < tol:
            return True
    return False

class Configuration:
    """A LABELLED configuration: the Teichmüller object, not the moduli point.

    Teich(S^2, P) is configurations of the labelled points p_i modulo PSL(2, C);
    moduli space is the further quotient by relabelling.  The distinction is
    not cosmetic here.  The pullback map sigma is defined on Teichmüller
    space: after pulling back the structure, the preimage of p_i is again
    labelled p_i, so sigma(τ) carries the same labels, and a fixed point of
    sigma is an equation about labelled configurations.  On moduli the map is
    not well defined -- only up to the action of the relabelling group -- so
    iterating there can converge to a point of the right moduli orbit that is
    not a fixed point of sigma at all.

    The labels used by the engine are the faces of the spine, i.e. the
    punctures, which the session fixes; `pullback_labels` will transport them
    through f|_P once sigma exists.
    """

    def __init__(self, labels, points):
        if len(labels) != len(points):
            raise ValueError("labels and points differ in number")
        self.labels = tuple(labels)
        self.points = tuple(complex(p) for p in points)
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("duplicate labels")

    def _ordered(self):
        order = sorted(range(len(self.labels)), key=lambda i: str(self.labels[i]))
        return [self.points[i] for i in order]

    def coordinate(self):
        """The Teichmüller coordinate: normalise the first three labelled
        points to 0, 1, infinity and read the fourth.

        Unlike `modulus`, this uses the LABELS to order the points, so two
        configurations that differ only by a relabelling have different
        coordinates -- which is exactly the information moduli discards.
        """
        pts = self._ordered()
        if len(pts) != 4:
            raise ValueError("the chart is for four points")
        return cross_ratio(*pts)

    def permuted(self, perm):
        """The same four points with the labels permuted against the positions.

        `perm` is a permutation of range(4): the point at label i becomes the
        point that was at label perm[i].  This is the relabelling the moduli
        chart cannot see: the set of points is unchanged, so `modulus` and
        `same_configuration` agree, but the label-to-position map -- which is
        what sigma acts on -- has changed.
        """
        if sorted(perm) != list(range(len(self.labels))):
            raise ValueError("perm must be a permutation of range(len(labels))")
        return Configuration(self.labels, [self.points[perm[i]] for i in range(len(perm))])

    def __eq__(self, other):
        return (isinstance(other, Configuration)
                and self.labels == other.labels and self.points == other.points)

    def __repr__(self):
        return f"Configuration({self.labels}, {self.points})"


def pullback_labels(labels, dynamics):
    """Relabel a configuration by the preimages in P: new label of the point
    that f sends to p is p itself.

    `dynamics` maps each label to the label of its image under f (the engine's
    type_classification()['dynamics'] gives exactly this on the punctures).
    Returns the ordering of the preimages: preimage_of[p] for each p.
    """
    preimage = {}
    for q, img in dynamics.items():
        if img in preimage:
            raise ValueError(f"two preimages of {img} inside P: not a Thurston marking")
        preimage[img] = q
    missing = [lab for lab in labels if lab not in preimage]
    if missing:
        raise ValueError(f"no preimage in P for {missing}")
    return [preimage[lab] for lab in labels]
