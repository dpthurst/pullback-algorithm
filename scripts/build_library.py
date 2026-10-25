#!/usr/bin/env python3
"""Build the example library (plan item 90): emit JSON sessions from
the Harmonic-format example strings (transcribed from
.scratch/Harmonic/Examples.lhs), and the fixture file with expected
outputs (validation results and portraits as computed, to be
cross-checked and then frozen).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pullback.harmonic_io import load_session, parse_graph, parse_cover, parse_cmap
from pullback.validate import validate
from pullback.portrait import face_map_pi
from pullback.ribbons import fmt_dart

# Transcribed from Examples.lhs (section names noted).  Extra data
# (elastic weights) recorded for M3.
EXAMPLES = {
    "theta": {
        "source": "Harmonic Examples.lhs 'Double theta to theta'; [PChar] Fig. 3",
        "map": "(1+z^2)/(1-z^2)",
        "graph": "apqr bRQP", "cover": "q01", "phi": "p0P p1r r0Q r1q",
        "punctures": [{"face": "qP", "name": "-1"}, {"face": "rQ", "name": "1"},
                      {"face": "Rp", "name": "inf (exterior)"}],
        "elastics": [4, 1, 4], "lengths": [2, 1, 3],
        "expected": {"portrait_degrees": [1, 1, 2, 2]},
    },
    "bouquet": {
        "source": "Harmonic Examples.lhs 'Bouquet' (contraction of theta)",
        "graph": "XaABb", "cover": "a01 b01", "phi": "a0A a1b",
        "expected_from_output_tex": "Emb[f^2]=3/4, f^3=5/8, f^4=23/48, f^5=187/480; limit 2^(-1/3)",
    },
    "barbell": {
        "source": "Harmonic Examples.lhs 'Barbell'",
        "graph": "aqpP bQRr", "cover": "p01 r01", "phi": "p1P p0qrQ",
    },
    "slit-lattes": {
        "source": "Harmonic Examples.lhs 'Slit Lattes' (Kevin Pilgrim); output.tex",
        "map": "z^2(z-1)/(z-1/3)^3",
        "graph": "apqr bRQP", "cover": "r012 q01", "phi": "q1P p1Q p2q q2r",
    },
    "census-3.2": {
        "source": "Harmonic Examples.lhs 'Quadratic Sierpinski Carpet'; [PChar] 1.2 census #3.2",
        "graph": "atpq bTrs cuQP dUSR", "cover": "r01",
        "phi": "s0p p0q q0trU q1R p1S s1S r0t r1U",
        "elastics_tuned": [189.0, 211.0, 68.8, 170.0, 1.0, 1.0],
        "note": "ASF about 0.94; n=1 insufficient for a certificate (plan 190 caveat)",
    },
    "rabbit": {
        "source": "Harmonic Examples.lhs (untwisted rabbit)",
        "map": "Douady rabbit z^2+c",
        "graph": "xaAbBcC", "cover": "a01", "phi": "b0a c0b a0c",
        "note": ("Harmonic has 'b0a c0b b0c': the third token is a duplicate (b,0) "
                 "and was silently dropped by Map.fromList.  Corrected against the "
                 "IMG package's rabbit machine PoirierExamples(1)=PolynomialSphereMachine(2,[1/7]): "
                 "f1 = sigma(f1^-1 f2^-1, f3 f2 f1), f2 = (f1, 1), f3 = (f2, 1), "
                 "with a=f1, b=f2, c=f3.  The IMG machine's literal chain form "
                 "('b0a c0b a0AB a1cba') is automaton-equal to the author's 'b0a c0b a0c' "
                 "(levels <= 6, test_rabbit_versions) but FAILS the ribbon boundary "
                 "pattern (3 essential faces vs 4 required), while 'a0c' passes it — so "
                 "the session uses the author's form, the ribbon representative.  The old "
                 "code never ran the untwisted rabbit for energies."),
    },
    "twisted-rabbit": {
        "source": "Harmonic Examples.lhs 'Hamal birthday conference' (Dehn twist of rabbit); third token corrected 2026-10-25",
        "graph": "xaAbBcC", "cover": "a01", "phi": "b0Bab c0BAbab a0c",
        "note": ("TWISTED rabbit as a topological map hR (h fixing P): the dynamics ON P is that of the "
                 "rabbit, so every cycle of P must contain a branch point -- and it does: hyperbolic type. "
                 "The first two tokens are the rabbit's b0, c0 conjugated by ab, exactly as the Harmonic "
                 "text says; the third is the rabbit's own a0c.  THE AUTHOR CONFIRMED 2026-10-25 that this "
                 "is the actual twisted rabbit.  The Harmonic import had 'b1c' there, which drops a0 -> c "
                 "and adds b1 -> c; that string is now the session twisted-z2i."),
    },
    "twisted-z2i": {
        "source": "Harmonic Examples.lhs 'Hamal birthday conference', string as written there",
        "graph": "xaAbBcC", "cover": "a01", "phi": "b0Bab c0BAbab b1c",
        "note": ("THE AUTHOR BELIEVES (2026-10-25) this describes a twist of z^2+i, not of the rabbit, and "
                 "has NOT checked whether it is obstructed ('some twists of z^2+i are obstructed').  "
                 "Consistent with that: on this spine the untwisted map with the same first two tokens is "
                 "'b0a c0b b1c', whose type is 'cycle without branch point' -- as for z^2+i, whose critical "
                 "orbit is i -> -1+i <-> -i with no branch point on the 2-cycle -- while the rabbit's 'a0c' "
                 "variant is hyperbolic.  NOT VERIFIED as z^2+i.  Kept because it is the library's only "
                 "machine where pi_1 fails to contract on the tree while the image contracts: pi_1 closure "
                 "106 words and open, 23 classes by the level-4 action and still 23 through level 8 (the "
                 "IMG-versus-pi_1 case, AGENTS.md A0)."),
    },
    "grigorchuk": {
        "source": "Harmonic Examples.lhs 'The Grigorchuk group'",
        "graph": "apPqQrRsS", "cover": "p01", "phi": "q1r r1s s1q q0p r0p",
        "category": "group",
        "note": ("The GRIGORCHUK group example: a group-theoretic illustration "
                 "(amenable, intermediate growth) with NO natural ribbon structure.  "
                 "The ribbon boundary pattern failing is expected and is not a defect; "
                 "group-level checks only."),
    },
    "tetrahedral": {
        "source": "Harmonic Examples.lhs 'Tetrahedral subdivision' (Buff page)",
        "graph": "1pqr 2Pts 3QSu 4RUT", "cover": "sac tab ubc",
        "phi": "ubR qcq scp rbU tbT pas",
    },
    "barycentric": {
        "source": "Harmonic Examples.lhs 'Barycentric subdivision'; [ConfDim] 5.2",
        "map": "4/27 (z^2-z+1)^3/(z(z-1))^2",
        "graph": "0pqr 1RQP", "cover": "rabc|def qad|bf|ce",
        "phi": "qaq qfQ qbr qeR qcp qdP",
        "expected_from_confdim": "E^2_2[phi^2] = 10/13",
    },
    "blown-up-lattes": {
        "source": "Harmonic Examples.lhs 'Blown-up Lattes'",
        "graph": "0abc 1CBA", "cover": "axyz cxy",
        "phi": "cxb cyA axc bzc czc",
    },
    "census-53": {
        "source": "Harmonic Examples.lhs 'Census #53'",
        "graph": "xRrg yGBb", "cover": "b012 r12", "phi": "g2r g0G r0b",
        "elastics": ["24", "3+sqrt(105)", "18+2*sqrt(105)"],
        "elastics_order": "b g r",
    },
    "julia-4442": {
        "source": "Harmonic Julia.lhs 'Example 1: Quadratic, (4,4,4,2) orbifold'",
        "map": "(2z^2-4)/z^2",
        "graph": "apqrs bSRQP", "cover": "q01",
        "phi": "r0P s0Q p0Q p1q s1r r1s",
        "elastics": [2, 1, 2, 1],
        "elastics_order": "p q r s",
        "note": ("Julia set is the whole sphere (post-critically finite with no critical "
                 "periodic cycles); hyperbolic orbifold (4,4,4,2).  The MAIN THEOREM OF [PChar] "
                 "does not apply to this example.  THE AUTHOR HAS NOT CHECKED THIS EXAMPLE "
                 "CAREFULLY — treat the data as unverified until the engine's checks pass.  "
                 "CHECKED 2026-10-25 AND IT FAILS: the shipped phi collapses all six cover "
                 "faces (face_map_phi: 6 collapsed, 0 essential), so it is not a ribbon map "
                 "and type_classification refuses it ('base face 0 has 0 marked cover faces'). "
                 "The string is verbatim from Harmonic Julia.lhs line 80, so this is inherited, "
                 "not a transcription slip; no rotation of the two flowers and no choice of "
                 "the monodromy edge rescues it (searched: all dihedral variants at a and b "
                 "crossed with q01/p01/r01/s01, never more than 0 essential faces).  The "
                 "covering portrait is consistent with (2z^2-4)/z^2: degrees [2,1,1,2,1,1], "
                 "the two critical values.  A corrected phi is needed.  Author's prediction "
                 "(2026-10-25): the corrected data has no branched cycles — the postcritical "
                 "dynamics is inf ->(2) 2 -> 1 <-> -2, and no critical point lies on the "
                 "1 <-> -2 cycle.  Old-code targets (M3): EL_v/EL_h converges toward 1.637 = (1.279)^2 from "
                 "the Schwarz-Christoffel rectangle on S^2-{inf,-2,1,2}; old starting weights "
                 "(2,1,2,1) on p,q,r,s, ratio table in Julia.lhs."),
    },
    "julia-4422": {
        "source": "Harmonic Julia.lhs 'Example 2: Quadratic, (4,4,2,2) orbifold'",
        "map": "(1 + z^2/(1-i))/(1-z^2)",
        "graph": "apqr bPtsQ cRST", "cover": "q01 s01",
        "phi": "q0p s0q s1Q q1R r1S p1T",
        "elastics": [1, 1, 1, 1, 1],
        "elastics_order": "p q r s t",
        "note": ("Flowers corrected per the author (2026-10-24): the b and c flowers were "
                 "transcribed clockwise; 'apqr bPtsQ cRST' is the unique planar embedding "
                 "(a flower fixed) whose cover is a sphere with 6 faces and portrait degrees "
                 "{1,1,1,1,2,2}, matching |f^-1(P)| = 6 and chi(Sigma_1) = 2 chi(Sigma_0).  "
                 "Its global mirror 'aprq bPQst cRTS' is the complex-conjugate map and "
                 "equally correct (global orientation reversal = complex conjugation; a common "
                 "user-input slip, immaterial here).  Julia set is the whole sphere; hyperbolic orbifold "
                 "(4,4,2,2); unique quadratic rational map with it up to conjugation.  The MAIN "
                 "THEOREM OF [PChar] does not apply.  THE AUTHOR HAS NOT CHECKED THIS EXAMPLE "
                 "CAREFULLY — treat the data as unverified until the engine's checks pass.  "
                 "Old-code targets (M3): triangle shapes toward (1, 1.19586, 1.25727) "
                 "(square roots of EL ratios from Schwarz-Christoffel integrals on "
                 "S^2-{0,1,2i,inf}); table in Julia.lhs."),
    },
}


def main():
    outdir = ROOT / "sessions"
    outdir.mkdir(exist_ok=True)
    fixtures = {}
    for name, ex in EXAMPLES.items():
        sess = {k: v for k, v in ex.items()
                if k in ("source", "map", "graph", "cover", "phi", "elastics",
                         "lengths", "elastics_tuned", "elastics_order", "note",
                         "punctures", "category",
                         "expected_from_output_tex", "expected_from_confdim")}
        sess["name"] = name
        (outdir / f"{name}.json").write_text(json.dumps(sess, indent=2) + "\n")

        base, cover, phi = load_session(outdir / f"{name}.json")
        rep = validate(phi, base)
        portrait, base_walks = face_map_pi(cover, base)
        degs = []
        bad = []
        for row in portrait:
            degs.append(row["degree"])
            if row["degree"] is None:
                bad.append("".join(fmt_dart(d) for d in row["face"]))
        from pullback.portrait import face_map_phi
        rep2, _bw = face_map_phi(phi)
        ess = [r for r in rep2 if r["degree"] is not None]
        col = [r for r in rep2 if r["degree"] is None]
        F0, F1 = len(base.faces()), len(cover.faces())
        boundary_ok = (len(col) == F1 - F0 and all(r["degree"] == 1 for r in ess)
                       and len({r["target"] for r in ess}) == len(ess))
        _, spine_mirrored = base.canonical_orientation()
        chi_base = len(base.vertices()) - len(base.edge_list()) + len(base.faces())
        chi_cover = len(cover.vertices()) - len(cover.edge_list()) + len(cover.faces())
        fixtures[name] = {
            "validation": rep,
            "spine_euler": chi_base,
            "spine_mirrored": spine_mirrored,
            "ribbon_boundary": {"essential": len(ess), "collapsed": len(col),
                                "ok": boundary_ok},
            "cover_euler": chi_cover,
            "portrait_degrees": sorted(d for d in degs if d is not None),
            "portrait_nonface_projections": bad,
        }
        flag = "" if rep["cover_connected"] and rep["pi1_surjective"] and not bad else "  <-- CHECK"
        print(f"{name:16s} nF={len(portrait)} degrees={fixtures[name]['portrait_degrees']}"
              f" connected={rep['cover_connected']} pi1-surj={rep['pi1_surjective']}"
              f" nonface={bad}{flag}")

    (ROOT / "tests" / "library_fixtures.json").write_text(json.dumps(fixtures, indent=2) + "\n")
    print("\nwrote sessions/*.json and tests/library_fixtures.json")


if __name__ == "__main__":
    main()
