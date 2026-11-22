"""Marked, comparison-preserving iteration on the refined rabbit carrier.

This layer deliberately does not claim convergence or rational-map recovery.
It connects the independently certified pieces: the refined cover/inclusion
carrier, the realised output retraction, and ``advance_comparison``.

A continuation is rebuilt from the *accumulated* map to the original session,
never from a fresh transport census.  It must reproduce the preceding spine,
cover, inclusion chain and vertex images exactly; otherwise it is refused.  The
source basepoint is part of that state.
"""

from __future__ import annotations

from . import mesh
from .geometric_retraction import geometric_upsilon, refined_vertex_positions
from .homotopy import compose, homotopy_section
from .lift import _is_identity_on_pi1, advance_comparison
from .maps import make_graph_map
from .refined_carrier import refined_carrier
from .refined_ve import RefinedVE
from .ribbons import RibbonGraph
from .ve_state import VEInit

__all__ = ["marked_step", "continue_refined", "iterate_marked"]


def _rebase(graph, root):
    """The same ribbon graph, with ``root`` first for based constructions."""
    if root not in graph.flowers:
        raise ValueError(f"unknown output basepoint {root!r}")
    order = [root] + [v for v in graph.vertices() if v != root]
    return RibbonGraph({v: graph.flowers[v] for v in order})


def marked_step(refined, cert=None, x0=None, sampling_steps=(256, 512)):
    """One certified marked step from a ``RefinedVE`` at its current points.

    The numerical realisation must be an embedded, zero-excess, injective
    degree-one certificate.  The output spine is rebased at the retraction
    image of the section's basepoint lift; without this presentation choice,
    the inverse homotopy-section calculation can fail even though an unbased
    inverse exists.  No graph word is changed by this rebasing.
    """
    if cert is None:
        cert = refined.step(refined.points, x0=x0)
    if not cert.get("embedded") or not cert.get("injective"):
        raise ValueError(f"marked step needs an embedded realisation: {cert.get('reason')}")
    if cert.get("excess") != 0 or abs(abs(cert.get("degree", 0.0)) - 1.0) > 1e-9:
        raise ValueError("marked step needs zero excess and spherical degree one")
    carrier = refined_carrier(refined)
    source = carrier["lifted"]
    positions = refined_vertex_positions(refined, cert["x"])
    output = [mesh.to_point(v) for v in cert["config"]]
    upsilon = geometric_upsilon(source, positions, output, steps=sampling_steps)

    section = homotopy_section(refined.init.spine, source, carrier["phi_prime"])
    old_root = refined.init.spine.vertices()[0]
    cover_root = section["psi"].image_vertex(old_root)
    new_root = upsilon["map"].image_vertex(cover_root)
    target = _rebase(upsilon["target"], new_root)
    graph_map = make_graph_map(source, target, upsilon["map"].chains,
                               upsilon["map"].vertex_images)
    upsilon["geometric_map"] = upsilon["map"]
    upsilon["geometric_target"] = upsilon["target"]
    upsilon["map"] = upsilon["upsilon"] = graph_map
    upsilon["target"] = target
    upsilon["basepoints"] = {
        "current": old_root, "lift": cover_root, "output": new_root,
        "section": section["psi"],
    }

    comparison = advance_comparison(
        refined.session, refined.init.spine, source, carrier["phi_prime"],
        refined.init.h, target, graph_map)
    identities = {
        "to_then_from": _is_identity_on_pi1(
            compose(comparison["to_original"], comparison["from_original"]),
            refined.session.base),
        "from_then_to": _is_identity_on_pi1(
            compose(comparison["from_original"], comparison["to_original"]),
            target),
    }
    if not all(identities.values()):
        raise ValueError(f"marked comparison is not invertible: {identities}")
    return {
        "refined": refined,
        "carrier": carrier,
        "numerical": cert,
        "upsilon": upsilon,
        "comparison": comparison,
        "identities": identities,
        "points": [mesh.to_point(v) for v in cert["config"]],
        "x0_was_supplied": x0 is not None,
    }


def continue_refined(step):
    """Rebuild the next refined state without forgetting its comparison.

    The previous ``to_original`` and its basepoint completely specify the next
    marked state.  For fixed rabbit combinatorics the reconstruction must agree
    literally with ``marked_step``'s pullback output.  A different Delaunay or
    homotopy presentation is a stop condition, not something to hide behind a
    new arbitrary transport.
    """
    previous = step["refined"]
    comparison = step["comparison"]
    init = VEInit(previous.session, points=step["points"],
                  chains=comparison["to_original"].chains,
                  basepoint=comparison["new_spine"].vertices()[0])
    refined = RefinedVE(previous.session, init=init)
    if init.transports is not None:
        raise ValueError("continuation unexpectedly ran a transport census")
    checks = {
        "spine": init.spine.serialization() == comparison["new_spine"].serialization(),
        "cover": init.P.serialization() == comparison["P"].serialization(),
        "to_original_chains": init.h.chains == comparison["to_original"].chains,
        "to_original_vertices": (init.h.vertex_images ==
                                 comparison["to_original"].vertex_images),
        "phi_chains": init.phi_prime.chains == comparison["phi_prime"].chains,
        "phi_vertices": (init.phi_prime.vertex_images ==
                         comparison["phi_prime"].vertex_images),
    }
    if not all(checks.values()):
        raise ValueError(f"refined continuation changed its marked carrier: {checks}")
    return refined


def iterate_marked(refined, steps=2, sampling_steps=(256, 512)):
    """Run ``steps`` marked steps, continuing from the embedded realisation.

    The previous realisation is reused only when the refined combinatorics and
    vertex count are unchanged.  Otherwise the step returns to the deterministic
    F2 seed.  Every numerical step still has to prove embeddedness, injectivity,
    zero excess and degree one.  This returns an orbit of configurations and
    their certificates, not a convergence assertion.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    current = refined
    previous = None
    x0 = None
    records = []
    for _index in range(steps):
        record = marked_step(current, x0=x0, sampling_steps=sampling_steps)
        records.append(record)
        previous = current
        current = continue_refined(record)
        same = (previous.K2["edges"].keys() == current.K2["edges"].keys()
                and previous.K2["proj"] == current.K2["proj"]
                and record["numerical"]["x"].shape[0] ==
                1 + max(current.K2["proj"]))
        x0 = record["numerical"]["x"] if same else None
        record["next_seed"] = "previous_realisation" if same else "F2"
    return {"steps": records, "terminal": current}
