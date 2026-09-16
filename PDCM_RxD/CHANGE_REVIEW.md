# PDCM_RxD — Complete Change Review (June–August 2026)

*Reconstructed 2026-08-31 by diffing the surviving pristine baseline against every
derivative file, cross-checked against the run logs, findings docs, and cross-session
research notes. There is **no git history** for this project, so "original vs. modified"
was rebuilt from backup files (`*.bak_*`), the version chain (v2→v3→v4), file mtimes,
and the in-tree reports.*

---

## 0. First: what the "original copy" actually is

You remembered the pristine triplet as **`cfgPopWei.py`, `netParamsPopsWei.py`, and
`initPopWei.py`**. Verified against disk — two of three are right, one never existed:

| File you named | Verdict |
|---|---|
| `netParamsPopsWei.py` | ✅ **Genuinely pristine.** mtime **2026-06-29**, byte-identical to its own `.bak_pre_ecsfix_20260716` backup — untouched since June. This is the true original network model. |
| `cfgPopWei.py` | ⚠️ **Original config, but lightly edited.** Last touched **2026-07-23**. Only two trivial changes vs the July-22 `nightly_test/` copy: (1) `fig_file` path fixed to `../test_mask.npy`; (2) added `'figSize': (10, 24)` to stop 11 recordTraces overlapping. Scientifically pristine. |
| `initPopWei.py` | ❌ **Does not exist** — not in the project, not anywhere on the filesystem. |

**What plays the "init" role:** `run_v2.py` carries the header comment
`# v0.0 - direct copy from ../uniformdensity/init.py`. So the original driver was
`../uniformDensity/init.py`. That directory (`newton/uniformDensity/`) is now **empty** —
its contents were lost in the same 2026-08-13 scratch→project move that dropped
`test_mask.npy`. The closest surviving snapshot of the original driver is
`run_v2.py.bak_pre_teleport_port` (2026-07-10, 27 KB).

**Bottom line:** the real, on-disk baseline is `netParamsPopsWei.py` (June 29) +
`cfgPopWei.py`. Crucially, your working model **imports from** the pristine
`netParamsPopsWei.py` rather than editing it in place — every substantive change lives
in a *new* `netParams_v2/_v3/_v4/…` file. That's why the whole change history is
cleanly diffable, which is what this review does.

---

## 1. The shape of the work

Changes fall into two big buckets:

1. **Infrastructure / correctness** (June–July): make the model actually run under MPI
   on the cluster without segfaulting or silently corrupting results, place cells
   correctly, and record SD onsets.
2. **Scientific levers** (July–August): a chain of model variants — `o2fix` → `v3` → `v4`
   → `penumbra` → `o2drive sweep` — each a hypothesis about *what makes capillary
   proximity matter for spreading-depolarization (SD) onset.*

The version lineage:

```
netParamsPopsWei.py (pristine, Jun 29)
        │  + crash-boundary fix, reaction-combining, teleport bridge
        ▼
   netParams_v2.py (Jul 20)  ── run_v2.py (Jul 27), the working baseline
        │  + tortuosity un-hardcode
        ▼
   netParams_v2_o2fix.py (Jul 27)
        ├──► _o2fix_trim.py        (domain crop)
        ├──► _penumbra.py → _penumbra4.py   (spatial stroke penumbra)
        ▼
   netParams_v3.py (Jul 23)   depth-dependent capillary O2
        ▼
   netParams_v4.py (Jul 27)   v3 + domain trim
```

---

## 2. Infrastructure changes (the foundation, June–July)

These are the edits in `netParamsPopsWei.py → netParams_v2.py` and the run driver.
They changed **no physics** — they made the existing physics runnable and measurable.

### 2.1 RxD crash-boundary fix (the big one)
**Change:** the O2-consumption reaction product was redirected from the extracellular
grid to the cytosol:
`"product": "o2_consumed[ecs_o2]"` → `"product": "o2_consumed[cyt_<pop>]"`, and
`o2_consumed` was given cytosolic regions in addition to `ecs_o2`.

**Why:** a `MultiCompartmentReaction` whose reactant **and** product both live in the
extracellular grid is unsupported in NEURON 9.0.1 — it segfaults inside
`ECS_Grid_node::initialize_multicompartment_reaction()` during `addRxD()`. (Upstream
support arrived in PR #3296, post-9.0.1.) Depleting extracellular O2 but depositing the
consumed O2 into cyt makes it a standard, supported ECS→cyt reaction.

**Result:** the model builds and runs. O2 dynamics are unchanged; neuronal consumption
is now recorded as `o2_consumedi` instead of `o2_consumedo`. **Known side-effect:** the
saved `o2con_*.npy` extracellular-consumption grid is now **all-zero** in every run, so a
"proper" OEF can't be read from it — use an O2-depletion proxy from `o2_*.npy` instead.

### 2.2 Reaction-combining to dodge a second NEURON bug
**Change:** the three separate `nkcc1_current1/2/3` reactions were merged into one
stoichiometric reaction (`2*cl + kk + na` → same on ecs side); same for `kcc2` (2 legs → 1).
The Na/K pump was **deliberately left as two legs** (combining it makes ecs both source
and destination alongside cyt species, which rxd rejects).

**Why:** NEURON 9.0.1's `register_rate()` has an offset-accumulation bug in `grids.cpp`
that corrupts internal buffers a little more with each `register_rate()` call under MPI,
eventually segfaulting after ~52 reactions on 10 ranks. Fewer reactions = fewer calls =
stays under the threshold.

**Result:** this, plus the local patched NEURON build (`neuron_patches/` — a
`defer-multicompartment-reaction-init` patch you wrote, with a repro and a drafted
upstream GitHub issue), is what let the full model run. The equivalent official fix
(PR #3736) merged to nrn master 2026-07-21 but is **not in any tagged release yet**, so
you're still on the local patched build.

### 2.3 RxD thread-pool reset fix
**Change:** `rxd.nthread(6)` is re-asserted at the top of `runIntervalFunc` in `run_v2.py`.

**Why:** `h.finitialize()` → `clear_rates()` silently resets the RxD reaction thread pool
back to 1, undoing the `addRxD(nthreads=6)` set at build time.

**Result:** the sim actually uses 6 RxD threads through the run instead of collapsing to 1.

### 2.4 Cell placement: teleport port + KDTree + self-validation
**Change (712 changed lines, `run_v2.py.bak_pre_teleport_port → run_v2.py`):**
- `GLOBAL_CAPILLARY_LIST` bridge added to `netParams_v2.py` (tagged "added 7/6") so the
  placement script can see the capillary list.
- A **KDTree** over capillary coordinates ("THE SPEED UPGRADE") for fast
  distance-to-nearest-capillary queries.
- **Tail-first, capacity-aware teleport placement** of cells.
- **Per-layer KS-test self-validation** (`ks_2samp` vs the empirical `all_LayerN_distance.csv`
  depth distributions), which **retries and escalates on failure**.

**Why / result:** correct, reproducible, empirically-matched cell positions with a guard
against the silent no-op placement failure mode; positions are the substrate for the entire
distance-to-capillary analysis. Validated 2026-07-21.

### 2.5 SD-onset instrumentation
**Change:** `run_v2.py` tracks `cellSDOpen`/`cellSDClosed` per gid, stamping an SD onset
when soma V crosses `cfg.SDThreshold = −40 mV`; open→close intervals ≤ 2.5 ms are
discarded as ordinary action potentials.

**Result:** this is the measurement instrument for the whole project. **Two important
caveats you discovered:**
- **Resume corrupts onsets:** `cellSDOpen/Closed` are in-memory globals *not* in the
  checkpoint, so any resumed run re-stamps every already-depolarized cell's onset at
  ~restore time. Onset **timing** from resumed runs is invalid (SD *fraction* is fine).
- **End-of-sim artifact:** cells firing an AP in the last ~2.5 ms get mis-recorded as SD;
  fixed post-hoc by `sd_onset_utils.load_filtered_onsets`.

---

## 3. Scientific levers (July–August)

The scientific question throughout: **does a neuron's distance to the nearest functioning
capillary set its SD-onset timing?** Each variant below is an attempt to make that signal
appear (or a control to test whether an apparent signal is real).

### 3.1 `o2fix` — un-hardcode O2 tortuosity  *(netParams_v2 → _o2fix, Jul 27)*
**Change:** the `ecs_o2` region's `tortuosity` was changed from a hardcoded `1.0` to
`getattr(cfg, "tort_ecs_o2", cfg.tort_ecs)` (default 1.6). Volume fraction was
**deliberately left at 1.0**.

**Why:** with tortuosity pinned at 1.0, `cfg.tort_ecs` never reached O2, so O2 diffused
isotropically-fast and washed out any microscale capillary→neuron gradient. Tortuosity is
the calibration-safe lever (it only sets `D_eff = d/τ²`); changing volume fraction would
rescale all O2 concentrations and break the `o2_bath`/pump-sigmoid calibration.

**Result:** `o2fix` became the standard physiological setting for all subsequent runs.
Sharpens the pericapillary gradient — but (see §4) not enough to make distance predict SD.

### 3.2 `v3` — depth-dependent capillary O2  *(o2fix → v3, Jul 23; 74 lines)*
**Change:** a per-voxel target PO2 (`o2_target`) built from Li et al. 2019 per-layer O2
setpoints replaces the flat scalar `o2_bath` in the O2 source rate. Two modes
(`absolute` / mean-preserving `modulate`), optional depth-dependent variance, all gated
behind `cfg.o2_depth_gradient`. When off, behaviour is exactly v2.

**Why:** real cortex has a depth gradient in capillary O2; a flat `o2_bath` can't express
depth-structured supply.

**Result:** Phase-0 flat-vs-depth test under hypoxia = **NULL** — depth-dependent O2 did
not change SD outcome. (Arteriole work was gated on PI sign-off and not pursued.)

### 3.3 `v4` — domain trim  *(v3 → v4, Jul 27; and `_o2fix_trim`)*
**Change:** crops the acellular top/bottom margins off the domain: capillary source image
cropped `img[1000:, …]` → `img[1649:-617, …]`; `popDepths` remapped to preserve absolute
cortical depth on the smaller `sizeY` (with a documented 1.09 overshoot from the somaR
clip term). `Vtissue`/`rs`/`somaR` decoupling constraint respected.

**Why:** (a) a physics-neutral ~16% voxel reduction for speed; (b) to test whether the L2
near-wall SD anomaly was a no-flux domain-wall artifact.

**Result (confirmed 2026-08-31):** in the **matched** o2drive=100 regime, the trim
**erases** the L2 near-wall failing-enrichment (failing-vs-nonfailing wall distance
45.1 vs 101.0 µm, p=2.6e-48 → 101.6 vs 106.1 µm, p=0.43, n.s.). ⇒ that spatial pattern
was a **no-flux x/z domain-wall artifact**. Side-effect: trimming lowers the whole-domain
O2 budget, so overall failure fraction rises (59.7%→66.6%).

### 3.4 `penumbra` variants — spatial stroke model  *(_penumbra, _penumbra4, _penumbra_central)*
**Change:** a `penumbra_scale` per-voxel mask multiplied into the O2 source rate: an
occluded ischemic **core**, a linear-ramp penumbra **shell**, and a perfused **periphery**.
Iterated from an off-center 2-zone (`_penumbra`), to a 4-zone (`_penumbra4`), to a
**centered radial** version (`_penumbra_central`, so the SD-initiating K⁺ bolus sits in the
core = physiological ordering).

**Why:** the uniform-hypoxia model has *no* penumbra; a real supply gradient might be what's
needed for distance-to-capillary to matter.

**Result:** the occlusion **does** impose a genuine standing macro-gradient (r(O₂,mask) up
to +0.69), but SD-onset timing stays **flat** against it. The centered version's 300 ms
smoke was a decisive well-powered null: SD fraction by zone **core 40.6% / shell 42.8% /
healthy 40.3%**, r(SD, distance) = **+0.003**. Root cause: SD propagates as a **synaptic
wave** (`cfg.connected=True`) that sweeps the whole 700 µm domain in ~200 ms and homogenizes
every zone. **The full 3000 ms run was correctly NOT launched.** A `cfg.connected=False`
disconnected control is staged (import-verified) but not run — it's a scientific design
change awaiting your/PI's OK.

### 3.5 `o2drive` sweep — the pivot to a global lever
**Change:** instead of spatial occlusion, sweep `cfg.o2drive` (global perfusion multiplier)
across a huge range and look for a bifurcation `o2drive*` separating a viable O2 plateau
from collapse→SD. Dozens of `cfgPopWei_o2sweep_d*` configs (d0.10 … d1000) + upper-bound
probes.

**Why:** `o2drive` is the global analogue of the per-voxel `penumbra_scale` — it
de-spatializes the penumbra and gives full statistics per point with no box/boundary
artifact.

**Result:** **NO bifurcation anywhere in [0.1, 1000]** — a 10,000× delivery range. Corrected
real-neuron SD fraction stays single-regime ~50–65% (0.597@100 → 0.505@500 → 0.509@1000).
Bulk O2 rises with delivery but the activity-hotspot O2 *minima* stay pinned near the floor.
The lever is **not** global O2 delivery.

---

## 4. The through-line result (why so many of these came back null)

Across every lever, the same wall: **capillary distance does not set SD-onset timing** in
this model, except in a narrow Layer-2 window. The reasons you established:

- **Synaptic propagation homogenizes SD.** With `cfg.connected=True`, an SD wave sweeps the
  whole domain and overrides local O2 differences (penumbra null, o2drive null). The synaptic
  pathway drives *whether/where* the wave goes; O2 geometry barely enters.
- **Deep-layer O2 is consumption-dominated, not supply-dominated**, so capillary geometry
  can't set timing there — the deep-layer null is *inherent/geometry-limited*, not a
  consumption mask (proven by the disconnected control: no deep layer gains a capillary
  effect when synapses are off).
- **The L2 effect is real but conditional** — it needs a structured O2 field (after ~400 ms)
  *and* O2 headroom (fail before the depletion floor). L2i is intrinsic/network-independent
  (r ≈ −0.72 → −0.74 connected→disconnected); L2e's early failure is **disinhibited drive**,
  not intrinsic vulnerability (collapses under `bothoff`).

---

## 5. Analysis & tooling built (not model physics, but part of the work)

A large analysis layer was written alongside the model: `analysis_sd_capillary_corr*.py`
(+ sweep/generic), `analysis_sd_wave_distance.py` (rejected the traveling-wave confound),
`analysis_causal_timepoints.py`, `analysis_connected_vs_disconnected_sd.py`,
`analysis_failing_subpop.py` (+`_trim`), the per-layer suite
(`analysis_layer_sd_prob/firing_rate/oef_depth.py`), `analysis_o2drive_bifurcation.py`,
`analysis_penumbra_central.py`, `sd_onset_utils.py`, `measure_depth_gradient.py`, and the
`neuron_patches/` bug-repro + patch package.

**One correction baked into the newer tools you should know about:** the historical
`sd_frac` divided by **142,940** grid positions, but only **~12,344** are real neurons (the
rest are NetStim placeholders). This **deflated every historical SD fraction ~11.6×** — the
"5.5% SD floor" was a denominator bug; true failure rate is ~50–65%. The newer analyses
(`^L\d+[ei]$` denominator) use the corrected count; be careful comparing old vs new numbers.

---

## 6. Data-integrity caveats to carry forward

- `o2con_*.npy` is **all-zero** (side-effect of the §2.1 fix) — use an O2 depletion proxy.
- **Resumed runs have invalid onset *timing*** (§2.5); SD *fraction* is fine.
- **`dx = 14 µm` fast-dev grid is numerically divergent** from production `dx = 10 µm` and
  was permanently excluded — never coarsen dx for speed (use the v4 trim or shorter duration).
- **No-flux domain walls trap O2** (right edge ~2.6× interior at L5 depth) — prefer analytic
  µm-coordinate analyses over near-wall field-voxel sampling; this is what the v4 trim addresses.
- **The `.mod` files** (`AMPA/GABA/IntFire_PD.mod`) show 2026-08-26 mtimes but no backups —
  these are standard mechanism files that were recompiled when the env/`test_mask` were
  restored, not scientifically modified.

---

*Sources: pristine-vs-derivative diffs of all `netParams*/run_*/cfg*` files;
`summer_report_data.md` (2026-08-04); `PENUMBRA_THREAD3_REPORT.md`;
`THREAD5_OVERNIGHT_STATUS.md`; `neuron_patches/README.md`; `FINDINGS_*.txt`; and
cross-session research memory.*
