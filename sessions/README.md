# Sessions

A session is a JSON file holding one graph virtual endomorphism — the
source of truth (plan D3, item 80).  All fields are strings or small
JSON structures; the graph data uses the author's Harmonic text format
(documented in `src/pullback/harmonic_io.py`, from `GraphIO.lhs`).

| field | meaning |
|---|---|
| `name` | session id (file stem) |
| `source` | provenance (paper section, Harmonic file, figure) |
| `map` | optional: the rational map the example comes from |
| `graph` | Γ₀ as flowers: `"apqr bRQP"` — vertex, then its darts in counterclockwise cyclic order; lowercase = forward, uppercase = backward |
| `cover` | monodromy per edge: `"q01"`, `"rabc\|def"` — edge letter, then disjoint cycles; sheets are single chars; unlisted edges are the identity |
| `phi` | edge images: `"p0P p1r"` — cover edge (letter, sheet) ↦ path of darts; unlisted edges are constants.  Chains are freely reduced on load; a backtrack is a homotopy |
| `elastics` / `elastics_order` | elastic weights α (M3 material) |
| `punctures` | optional: `[{"face": "qP", "name": "-1"}, …]` — base face (boundary darts, any rotation) ↦ puncture name, for reading the portrait table |
| `note` | caveats, corrections, verification state — read before trusting a session |

Conventions worth knowing:

* **Orientation**: a global reversal of every flower gives the
  complex-conjugate map (plan item 50).  Both are valid; fixtures record
  which orientation each session uses (`spine_mirrored`).
* **Ribbon data is not checked by the energies** — a session can be
  right for M3 and wrong for the portrait.  `ribbon_boundary` in the
  fixtures records the necessary embedding pattern (`face_map_phi`), and
  `note` records known defects: `julia-4442` and `grigorchuk` currently
  fail it (their φ is not a ribbon representative), while the rabbit
  session uses the author's `a0c` form, which passes.
* Sheets are single characters; digits are read as integers (`0`, `1`),
  letters as letters (`a`…`f`).
