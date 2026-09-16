# Tier 3a results — capillary-distance effect vs O₂ delivery (trimmed domain, wall-confounded out)

Both 3a runs completed cleanly (d1 → t=2500 ms, d10 → t=3000 ms, 20 pkls each).
Re-ran `analysis_tier25_wall_confound.py` across all four delivery levels. The
capillary-distance→SD protection is **real and wall-independent at every trimmed
delivery level**, but its **magnitude is dynamic-range-limited**: it is large only
when delivery is high enough to pull a layer off its failure ceiling.

## L2 (where the effect lives) — INTERIOR (wall-free) stratum + partial logit
| Delivery | L2 SD | interior Δ (near→far) | cap_dist β (p) | wall_dist p |
|---|---|---|---|---|
| **d1  physiological** | 79.6% | **+1.1%** | +0.150 (2e-4) ✓ | 0.91 n.s. |
| **d10 intermediate**  | 82.0% | **+3.7%** | +0.229 (1.8e-7) ✓ | 0.81 n.s. |
| **d100 trimmed**      | 51.3% | **+8.3%** | +0.270 (3e-17) ✓ | 0.88 n.s. |
| d100 *untrimmed*      | 16.6% | +1.1% (raw +15.8% inflated) | +0.880 | **1e-11 (wall confound)** |

## L4 — same ceiling-saturation story
| Delivery | L4 SD | interior Δ | cap_dist p |
|---|---|---|---|
| d1  | 97.7% | +0.4% | 0.77 n.s. (at ceiling) |
| d10 | 97.4% | +0.5% | 0.28 n.s. (at ceiling) |
| d100 trimmed | 75.6% | +5.7% | 2e-7 ✓ |

L6 saturates at 97–100% (no effect anywhere); L5 too low-SD/noisy for a stable signal.

## Interpretation
1. **The effect is genuine and not a wall artifact** — `wall_dist` is n.s. in every
   trimmed run while `cap_dist` stays significant. The untrimmed d100 wall confound
   (raw +15.8%, wall p=1e-11) is the only place walls matter, confirming the trim was
   necessary.
2. **The effect is throttled by dynamic range, not by physics of distance.** At
   physiological delivery (d1) L2 sits at ~80% SD — near its failure ceiling — so
   distance can only move ~1%. As delivery rises and pulls the layer off the ceiling
   (d100: L2 → 51%), the same distance mechanism moves +8%. Distance always matters;
   it is only *visible* when cells are spread across the pump sigmoid instead of piled
   at the ceiling.

## Why this motivates Tier 3b
At physiological delivery the protective effect is real but small **because the
well-mixed O₂ bath keeps the whole layer uniformly hypoxic near the ceiling** — not
because distance is unimportant. Tier 3b (deepen the pericapillary gradient by lowering
`D_eff`) is precisely the lever to make distance decisive *at physiological delivery*:
a steep gradient spreads neighbouring cells across the pump sigmoid (vessel-adjacent
survive, mid-tissue fail) via spatial heterogeneity, instead of needing supra-
physiological global delivery to expose the effect. 3a is the strongest no-model-change
statement; 3b is the model change that would move the effect from "modest, needs O₂
flooding" to "primary, at normal O₂."
