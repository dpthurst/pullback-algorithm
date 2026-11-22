"""pullback — graph virtual endomorphisms of branched self-covers.

Ribbon graphs, monodromy covers, graph maps by edge paths, the critical
portrait of [PChar] §2, the rose normal form and its wreath recursion, the
energies, and the desktop applet (`pullback.app`).

Scripting entry point: `Session` (see `pullback.session`) -- one object per
dataset, every step a cached method.  Conventions follow the author's
Harmonic code (GraphIO.lhs) for input, with the ribbon structure retained
(the Haskell code discards it).
"""

from .ribbons import RibbonGraph, Dart, F, B
from .covers import monodromy_cover
from .maps import GraphMap, make_graph_map
from .portrait import face_map_pi, face_map_phi
from .session import Session, library, load

__all__ = [
    "RibbonGraph", "Dart", "F", "B",
    "monodromy_cover", "GraphMap", "make_graph_map",
    "face_map_pi", "face_map_phi",
    "Session", "library", "load",
]
