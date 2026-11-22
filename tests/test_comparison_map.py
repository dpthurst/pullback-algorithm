"""Tests for the combinatorial one-step prototype.

Run from this directory:

    PYTHONPATH=<repo>/src python3 -m unittest -v test_comparison_map

The suite is grouped as the four things the coordinator asked to see:

* ``ActiveDiagonal``  stage (a): the diagnostics, and that they refuse the two
  library sessions the project already records as non-ribbon.
* ``Rose``            stage (b): the collapse really is a surface-preserving
  move, so ``B_next`` is a spine of the SAME punctured sphere.
* ``ComparisonMap``   stage (c): ``comparison_map : P_n -> B_next`` and the
  promotion through ``advance_comparison``, with the checks that
  ``pullback_ve`` does not do.
* ``Refusals``        the negative direction: capping a TARGET petal is refused
  by the equal-rank gate, an orientation-reversed comparison is refused by the
  strict peripheral test, and the non-ribbon sessions are refused up front.
"""

from __future__ import annotations

import os
import unittest

from pullback.homotopy import is_peripheral_preserving, peripheral_images
from pullback.maps import make_graph_map
from pullback.ribbons import B, F, RibbonGraph
from pullback.session import Session
from pullback.validate import spanning_tree

from pullback.comparison_map import (Refusal, active_diagonal, build_comparison_map,
                            certify_state, collapse_to_rose, identity_map, iterate,
                            one_step, tree_invariance)

SESSIONS = os.environ.get("PULLBACK_SESSIONS", "sessions")

# every library session whose face map is a ribbon presentation
RIBBON = ("barbell", "barycentric", "blown-up-lattes", "bouquet", "census-3.2",
          "census-53", "julia-4422", "rabbit", "slit-lattes", "tetrahedral",
          "theta", "twisted-rabbit", "twisted-z2i")


def load(name):
    return Session.load(os.path.join(SESSIONS, f"{name}.json"))


def rose(n, prefix="x"):
    """An n-petal rose, with its generated edge labels."""
    labels = [f"{prefix}{i}" for i in range(n)]
    flower = tuple((label, ori) for label in labels for ori in (F, B))
    return RibbonGraph({0: flower}), labels


def carried_fingerprint(record):
    """The state data consumed by the next step, including vertex images."""
    state = record["state"]

    def graph(g):
        return [(v, tuple(ds)) for v, ds in
                sorted(g.flowers.items(), key=lambda kv: str(kv[0]))]

    def graph_map(m):
        chains = [(e, tuple(ch)) for e, ch in
                  sorted(m.chains.items(), key=lambda kv: str(kv[0]))]
        vertices = sorted(m.vertex_images.items(),
                          key=lambda kv: str(kv[0]))
        return chains, vertices

    return (graph(state["new_spine"]), graph(state["P"]),
            graph_map(state["phi_prime"]), graph_map(state["to_original"]))


class ActiveDiagonal(unittest.TestCase):
    """Stage (a): does phi_n collapse a face, or is it not a ribbon map?"""

    def test_rabbit_diagonal_is_a_degree_one_bijection(self):
        s = load("rabbit")
        diag = active_diagonal(s.phi)
        self.assertTrue(diag["ok"])
        self.assertTrue(diag["injective"])
        self.assertTrue(diag["exhaustive"])
        self.assertEqual(sorted(diag["targets"]), [0, 1, 2, 3])
        self.assertEqual(sorted(set(diag["degrees"].values())), [1])
        self.assertEqual(len(diag["collapsed"]), 2)

    def test_collapsed_faces_account_for_the_rank_gap(self):
        """#collapsed = |faces(P)| - |faces(B)| = rank(P) - rank(B)."""
        for name in RIBBON:
            with self.subTest(session=name):
                s = load(name)
                diag = active_diagonal(s.phi)
                rank_cover = (len(s.cover.edge_list())
                              - len(s.cover.vertices()) + 1)
                rank_base = (len(s.base.edge_list())
                             - len(s.base.vertices()) + 1)
                self.assertTrue(diag["ok"])
                self.assertEqual(len(diag["collapsed"]), rank_cover - rank_base)
                self.assertEqual(len(s.cover.faces()) - len(s.base.faces()),
                                 rank_cover - rank_base)

    def test_grigorchuk_is_refused_not_silently_accepted(self):
        """6 active faces against |P| = 5: not a ribbon presentation."""
        s = load("grigorchuk")
        diag = active_diagonal(s.phi)
        self.assertFalse(diag["ok"])
        self.assertEqual(diag["reason"], "non_base_face_word")
        self.assertEqual(len(diag["non_face"]), 1)
        self.assertGreater(len(diag["active"]), diag["n_faces_base"])

    def test_julia_4442_is_refused(self):
        """Every active face tightens to a non-face word: no base face is hit."""
        s = load("julia-4442")
        diag = active_diagonal(s.phi)
        self.assertFalse(diag["ok"])
        self.assertEqual(diag["reason"], "non_base_face_word")
        self.assertFalse(diag["exhaustive"])
        with self.assertRaises(Refusal) as caught:
            build_comparison_map(s.base, s.cover, s.phi)
        self.assertEqual(caught.exception.reason, "non_base_face_word")


class Rose(unittest.TestCase):
    """Stage (b): the collapse is surface-preserving, so B_next is a spine."""

    def test_collapse_keeps_chi_faces_and_rank(self):
        for name in RIBBON:
            with self.subTest(session=name):
                s = load(name)
                rep = collapse_to_rose(s.base)
                rose = rep["rose"]
                self.assertEqual(len(rose.vertices()), 1)
                self.assertTrue(rep["faces_preserved"])
                self.assertTrue(rep["rank_preserved"])
                self.assertEqual(len(rose.edge_list()), len(s.base.edge_list())
                                 - len(s.base.vertices()) + 1)
                self.assertEqual(len(rose.faces()), len(s.base.faces()))
                # a rose with n petals has n + 1 faces, and rank n
                self.assertEqual(len(rose.faces()), len(rose.edge_list()) + 1)
                chi = (len(rose.vertices()) - len(rose.edge_list())
                       + len(rose.faces()))
                self.assertEqual(chi, 2)

    def test_collapse_is_peripheral_and_orientation_sensitive(self):
        for name in RIBBON:
            with self.subTest(session=name):
                s = load(name)
                rep = collapse_to_rose(s.base)
                theta = rep["theta"]
                strict = peripheral_images(theta, allow_reverse=False)
                self.assertNotIn(None, strict)
                self.assertEqual(sorted(strict), list(range(len(s.base.faces()))))
                self.assertTrue(is_peripheral_preserving(theta,
                                                         allow_reverse=False))

    def test_theta_induces_the_identity_on_pi1(self):
        s = load("tetrahedral")
        rep = collapse_to_rose(s.base)
        from pullback.lift import _is_identity_on_pi1
        from pullback.homotopy import compose
        back = tuple(peripheral_images(rep["theta"], allow_reverse=False))
        self.assertEqual(len(back), len(s.base.faces()))
        # theta is a homotopy equivalence: its composite with a section of
        # itself is the identity on pi_1 of the rose
        from pullback.homotopy import homotopy_section
        section = homotopy_section(rep["rose"], s.base, rep["theta"])["psi"]
        self.assertTrue(_is_identity_on_pi1(
            compose(rep["theta"], section), rep["rose"]))


class ComparisonMap(unittest.TestCase):
    """Stage (c): the comparison and the promoted state."""

    def test_every_ribbon_session_certifies_one_step(self):
        for name in RIBBON:
            with self.subTest(session=name):
                s = load(name)
                out = one_step(s)
                c = out["certificates"]
                b = out["build"]
                self.assertTrue(c["ok"], (name, {k: v for k, v in c.items()
                                                 if isinstance(v, bool)}))
                self.assertTrue(b["capping_exact"])
                self.assertTrue(b["rank_gap_ok"])
                self.assertTrue(b["petals_ok"])
                self.assertTrue(b["faces_labelled"])
                self.assertEqual(b["n_petals"], len(s.base.faces()) - 1)
                # what pullback_ve does not certify:
                self.assertTrue(c["cover_is_sphere"])
                self.assertTrue(c["cover_connected"])
                self.assertTrue(c["punctured_chi_ok"])
                self.assertTrue(c["riemann_hurwitz_ok"])
                self.assertTrue(c["phi_pi1_surjective"])

    def test_collapsed_face_indices_and_rank_accounting(self):
        """Bookkeeping identities; the normal-closure proof is in the module."""
        for name in RIBBON:
            with self.subTest(session=name):
                s = load(name)
                out = one_step(s, certify=False)
                b = out["build"]
                self.assertEqual(b["capped_faces"], b["collapsed_faces"])
                self.assertEqual(b["rank_cover"] - b["rank_rose"],
                                 len(b["collapsed_faces"]))
                for f, row in b["faces"].items():
                    self.assertEqual(row["kind"] == "capped",
                                     f in b["collapsed_faces"])

    def test_phi_prime_points_from_the_new_cover_to_the_new_spine(self):
        s = load("theta")
        out = one_step(s)
        self.assertIs(out["phi_prime"].source, out["P"])
        self.assertIs(out["phi_prime"].target, out["new_spine"])
        self.assertIs(out["pi_prime"].source, out["P"])
        self.assertIs(out["pi_prime"].target, out["new_spine"])

    def test_rabbit_dynamics_is_carried(self):
        """The rabbit has a fixed puncture and a 3-cycle: 0 -> 2 -> 3 -> 0."""
        s = load("rabbit")
        out = one_step(s)
        c = out["certificates"]
        self.assertEqual(c["dynamics_in"], {0: 2, 1: 1, 2: 3, 3: 0})
        self.assertTrue(c["dynamics_preserved"])
        self.assertTrue(c["inverse_permutations"])

    def test_direct_certificate_honours_supplied_expectations(self):
        """A direct certificate must not ignore labels or failed dynamics."""
        s = load("rabbit")
        out = one_step(s)
        good = dict(out["certificates"]["dynamics_in"])
        rep = certify_state(s, out["state"],
                            expect={"labels_in": list(range(len(good))),
                                    "dynamics_in": good})
        self.assertTrue(rep["dynamics_preserved"])
        self.assertTrue(rep["labels_preserved"])
        self.assertTrue(rep["ok"])

        wrong = dict(good)
        first = sorted(wrong)[0]
        wrong[first] = (wrong[first] + 1) % len(wrong)
        rep = certify_state(s, out["state"],
                            expect={"labels_in": list(range(len(good))),
                                    "dynamics_in": wrong})
        self.assertFalse(rep["dynamics_preserved"])
        self.assertFalse(rep["ok"])

        swapped = list(range(len(good)))
        swapped[0], swapped[1] = swapped[1], swapped[0]
        rep = certify_state(s, out["state"],
                            expect={"labels_in": swapped,
                                    "dynamics_in": good})
        self.assertFalse(rep["labels_preserved"])
        self.assertFalse(rep["ok"])

        rep = certify_state(s, out["state"])
        self.assertNotIn("dynamics_preserved", rep)
        self.assertTrue(rep["ok"])

    def test_barycentric_degree_six(self):
        """The barycentric carpet: d = 6, |P| = 3, 5 of 8 faces collapse."""
        s = load("barycentric")
        self.assertEqual(s.degree, 6)
        out = one_step(s)
        b, c = out["build"], out["certificates"]
        self.assertTrue(c["ok"])
        self.assertEqual(b["n_petals"], 2)
        self.assertEqual(len(b["collapsed_faces"]), 5)
        self.assertEqual(len(b["active_faces"]), 3)
        self.assertEqual(b["rank_cover"], 7)
        self.assertEqual(b["rank_rose"], 2)
        # 6 vertices, 12 edges, and |faces(P)| = 2 + d(|P| - 2) = 8
        self.assertEqual(len(out["P"].vertices()), 6)
        self.assertEqual(len(out["P"].edge_list()), 12)
        self.assertEqual(len(out["P"].faces()), 8)
        self.assertEqual(c["punctured_chi"], -6)
        self.assertEqual(c["punctured_chi_expected"], 6 * (2 - 3))
        # this map sends every puncture to puncture 0
        self.assertEqual(c["dynamics_in"], {0: 0, 1: 0, 2: 0})
        self.assertTrue(c["dynamics_preserved"])

    def test_three_steps_keep_the_dynamics_and_certify(self):
        for name in ("rabbit", "barycentric", "theta", "tetrahedral",
                     "census-3.2"):
            with self.subTest(session=name):
                s = load(name)
                run = iterate(s, steps=3)
                dyn = [st["certificates"]["dynamics_in_input_labels"]
                       for st in run["steps"]]
                for k, st in enumerate(run["steps"]):
                    self.assertTrue(st["certificates"]["ok"], (name, k))
                    self.assertEqual(st["build"]["n_petals"],
                                     len(s.base.faces()) - 1)
                    self.assertEqual(dyn[k], dyn[0])

    def test_the_second_step_reproduces_the_first_carried_state(self):
        """Deeper runs repeat a fixed point; they do not probe new states."""
        for name in RIBBON:
            with self.subTest(session=name):
                s = load(name)
                first = one_step(s)
                second = one_step(s, state=first)
                self.assertEqual(carried_fingerprint(first),
                                 carried_fingerprint(second))
                self.assertTrue(second["certificates"]["ok"])

    def test_after_the_first_step_the_spine_is_a_rose_and_the_collapse_trivial(self):
        s = load("theta")
        run = iterate(s, steps=2)
        second = run["steps"][1]
        self.assertEqual(len(second["build"]["collapse"]["tree"]), 0)
        self.assertEqual(tuple(second["build"]["peripheral_strict"]),
                         tuple(sorted(second["build"]["peripheral_strict"])))

    def test_iterate_with_an_explicit_tree_uses_it_only_at_step_zero(self):
        for name in ("theta", "tetrahedral", "census-3.2", "barbell",
                     "julia-4422"):
            with self.subTest(session=name):
                s = load(name)
                tree = sorted(spanning_tree(s.base)[0], key=str)
                run = iterate(s, steps=3, tree=tree)
                self.assertEqual(len(run["steps"]), 3)
                self.assertTrue(all(st["certificates"]["ok"]
                                    for st in run["steps"]))
                self.assertEqual(tuple(run["steps"][0]["build"]["collapse"]["tree"]),
                                 tuple(tree))
                for st in run["steps"][1:]:
                    self.assertEqual(st["build"]["collapse"]["tree"], ())

    def test_pi_prime_is_certified_to_be_the_monodromy_projection(self):
        s = load("barycentric")
        state = one_step(s)["state"]
        P, spine = state["P"], state["new_spine"]
        constant = make_graph_map(
            P, spine, {e: () for e in P.edge_list()},
            {v: spine.vertices()[0] for v in P.vertices()})
        rep = certify_state(s, dict(state, pi_prime=constant))
        self.assertTrue(rep["riemann_hurwitz_ok"])
        self.assertFalse(rep["pi_prime_is_projection"])
        self.assertFalse(rep["ok"])

    def test_tree_choice_changes_the_chains_not_the_certified_data(self):
        """The tree is the declared choice; the labels and the type are not.

        Distinct spanning trees give distinct comparison tables and distinct
        accumulated label permutations, while rank, chi, puncture count, fibre
        degrees and the dynamics in input labels stay fixed.  That is the
        boundary between "combinatorially forced" and "supplied by the step".
        """
        for name in ("tetrahedral", "census-3.2", "theta"):
            with self.subTest(session=name):
                rep = tree_invariance(load(name))
                self.assertGreater(rep["n_trees"], 1)
                self.assertTrue(rep["all_ok"], rep["reports"])
                self.assertTrue(rep["same_invariant"], rep["reports"])
                self.assertTrue(rep["chains_differ"], rep["reports"])


class Refusals(unittest.TestCase):
    """Negative controls: what the step must NOT do."""

    def test_a_tree_that_is_not_a_spanning_tree_is_refused(self):
        s = load("theta")
        tree, gens = spanning_tree(s.base)
        tree = sorted(tree, key=str)
        bouquet = load("bouquet")
        cases = ((s.base, [], "not_a_spanning_tree"),
                 (s.base, ["nope"], "tree_edge_missing"),
                 (s.base, tree + [gens[0]], "not_a_spanning_tree"),
                 (bouquet.base, [bouquet.base.edge_list()[0]],
                  "not_a_spanning_tree"))
        for base, bad, reason in cases:
            with self.subTest(tree=bad):
                with self.assertRaises(Refusal) as caught:
                    collapse_to_rose(base, tree=bad)
                self.assertEqual(caught.exception.reason, reason)

    def _one_petal_rose(self):
        """What "send the collapsed petal to a point" would give for the theta."""
        vertex = ("capped",)
        rose = RibbonGraph({vertex: (("t", F), ("t", B))})
        return rose, vertex

    def test_capping_a_target_petal_fails_the_equal_rank_gate(self):
        """The carpet objection, made operational, at the exact gate.

        A rose with one petal FEWER than |P|-1 is what capping a target petal
        produces.  ``pullback_ve`` refuses it on the rank check before any
        homotopy question is asked: a spine of ``S^2 - P'`` with
        ``|P'| = |P| - 1`` has rank ``|P| - 2`` and cannot be compared to the
        input spine with a homotopy equivalence.
        """
        from pullback.lift import pullback_ve
        s = load("theta")
        rose, vertex = self._one_petal_rose()
        self.assertEqual(len(rose.edge_list()), 1)      # one petal
        self.assertEqual(len(rose.faces()), 2)          # |P'| = 2, not 3
        down = make_graph_map(rose, s.base, {e: () for e in rose.edge_list()},
                              {vertex: s.base.vertices()[0]})
        with self.assertRaises(ValueError) as caught:
            pullback_ve(s.base, s.cover, s.phi, rose, down)
        self.assertIn("equal rank", str(caught.exception))

    def test_capping_a_target_petal_also_breaks_the_promotion(self):
        """The same cap, driven through ``advance_comparison``: still refused."""
        from pullback.lift import advance_comparison
        s = load("theta")
        rose, vertex = self._one_petal_rose()
        comparison = make_graph_map(
            s.cover, rose,
            {e: (("t", F),) if e[0] == "q" else () for e in s.cover.edge_list()},
            {v: vertex for v in s.cover.vertices()})
        with self.assertRaises(ValueError):
            advance_comparison(s, s.base, s.cover, s.phi, identity_map(s.base),
                               rose, comparison)

    def test_unhit_target_face_is_a_refusal_not_a_collapse(self):
        """When |P'| < |P| the builder refuses before any capping happens."""
        base, _ = rose(2, "a")
        cover, (x, y) = rose(2, "x")
        phi = make_graph_map(cover, base, {x: (), y: ()}, {0: 0})
        self.assertEqual(active_diagonal(phi)["reason"], "target_petal_unhit")
        with self.assertRaises(Refusal) as caught:
            build_comparison_map(base, cover, phi)
        self.assertEqual(caught.exception.reason, "target_petal_unhit")

    def test_degree_two_active_face_is_refused(self):
        base, (a,) = rose(1, "a")
        cover, (x, y) = rose(2, "x")
        phi = make_graph_map(cover, base,
                             {x: ((a, F), (a, F)), y: ((a, B), (a, B))},
                             {0: 0})
        self.assertEqual(active_diagonal(phi)["reason"], "active_winding_gt_1")
        with self.assertRaises(Refusal) as caught:
            build_comparison_map(base, cover, phi)
        self.assertEqual(caught.exception.reason, "active_winding_gt_1")

    def test_two_active_faces_over_one_petal_are_refused(self):
        base, (a,) = rose(1, "a")
        cover, (x, y, z) = rose(3, "x")
        phi = make_graph_map(cover, base,
                             {x: ((a, F),), y: ((a, B),), z: ((a, B),)},
                             {0: 0})
        self.assertEqual(active_diagonal(phi)["reason"], "active_not_injective")
        with self.assertRaises(Refusal) as caught:
            build_comparison_map(base, cover, phi)
        self.assertEqual(caught.exception.reason, "active_not_injective")

    def test_orientation_reversed_comparison_is_refused_by_the_strict_test(self):
        """pullback_ve's peripheral gate allows reversal; the strict one does not.

        The collapse passes the strict test, so requiring it costs nothing and
        closes the chirality hole that ``allow_reverse=True`` leaves open.
        """
        s = load("theta")
        rose = collapse_to_rose(s.base)["rose"]
        mirror = rose.mirror()
        flipped = make_graph_map(rose, mirror,
                                 {e: ((e, F),) for e in rose.edge_list()},
                                 {v: v for v in rose.vertices()})
        self.assertTrue(is_peripheral_preserving(flipped, allow_reverse=True))
        self.assertFalse(is_peripheral_preserving(flipped, allow_reverse=False))
        self.assertEqual(peripheral_images(flipped, allow_reverse=False),
                         [None] * len(rose.faces()))

    def test_a_state_whose_phi_is_not_pi1_surjective_is_refused_up_front(self):
        """No section exists, so no comparison can be promoted."""
        s = load("theta")
        vertex = s.base.vertices()[0]
        flat = make_graph_map(s.cover, s.base,
                              {e: () for e in s.cover.edge_list()},
                              {v: vertex for v in s.cover.vertices()})
        state = {"new_spine": s.base, "P": s.cover, "phi_prime": flat,
                 "to_original": identity_map(s.base)}
        with self.assertRaises(Refusal) as caught:
            one_step(s, state=state)
        self.assertEqual(caught.exception.reason, "phi_not_pi1_surjective")


if __name__ == "__main__":
    unittest.main()
