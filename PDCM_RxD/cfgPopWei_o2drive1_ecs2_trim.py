"""
GO-FORWARD DEFAULT config: o2drive1_ecs2, o2fix physics + v4 DEPTH TRIM.

This is the canonical "ready to run" weak-drive config folding in BOTH standing
go-forward defaults (user decisions 2026-07-30):
  1. DEPTH TRIM of the acellular top/bottom ECS margins (memory
     pdcm_rxd_v4_domain_trim) -- removes the pia/WM no-flux O2-accumulation bands.
  2. 2500 ms duration for weak drive (memory pdcm_rxd_duration_speedup_bimodal)
     -- captures ~96-97% of onsets at ~17% wall saving.

It is a pure additive delta over cfgPopWei_o2drive1_ecs2_o2fix.py:
  * inherits identical o2fix physics (tortuous ecs_o2 diffusion),
  * applies the frozen-Vtissue/rs/somaR depth trim so the geometry change does
    NOT rescale any neuron's biophysics (the whole point of the v4 decoupling).
Runs under run_v2_o2fix_trim.py / netParams_v2_o2fix_trim.py (the trimmed
img-crop + popDepths remap). Output paths are isolated so it cannot collide
with any live o2fix / penumbra / comparison run.
"""
import os
from cfgPopWei_o2drive1_ecs2 import cfg

# ---- o2fix physics: tortuous O2 diffusion (same as cfgPopWei_o2drive1_ecs2_o2fix) ----
cfg.tort_ecs_o2 = cfg.tort_ecs

# ============================================================
# v4 DEPTH TRIM folded onto the o2fix lineage.
# ------------------------------------------------------------
# The base cfg already computed Vtissue -> rs -> somaR -> cyt_fraction from the
# UNTRIMMED cfg.sizeY, so those biophysical quantities are ALREADY frozen-correct
# on the ~1mm^3 calibration volume. We only (a) record that untrimmed extent as
# sizeY_orig and (b) overwrite the DOMAIN/PLACEMENT extent cfg.sizeY with the
# trimmed value. Vtissue/rs/somaR/cyt_fraction are deliberately NOT recomputed
# -> the geometry trim stays physics-neutral (validated v3-vs-v4: SD outcome
# 12281 vs 12340 cells, median onset 158 vs 159 ms).
cfg.sizeY_orig    = cfg.sizeY                        # 2131.2851 um (frozen calib volume)
cfg.top_margin    = 0.08 * cfg.sizeY_orig            # 170.5028 um (L1 exclusion == popDepths[0][0])
cfg.bottom_margin = 2 * cfg.somaR                    # 162.0078 um (L6 soma clip, FROZEN somaR)
cfg.sizeY         = cfg.sizeY_orig - cfg.top_margin - cfg.bottom_margin  # 1798.7745 um (15.6% smaller)

# ---- weak-drive standard duration ----
cfg.duration = 2500  # go-forward standard for weak drive (drive1/drive10)

cfg.simLabel   = f"SDL13_37_{cfg.ox}_{cfg.duration}ms_o2drive1_ecs2_trim"
cfg.saveFolder = "./data/hypoxic_o2drive1_ecs2_trim"
cfg.restoredir = cfg.saveFolder if cfg.restore else None

graph_folder = "./graphs/hypoxic_o2drive1_ecs2_trim"
os.makedirs(graph_folder, exist_ok=True)
cfg.graph_folder = graph_folder
cfg.analysis['plotRaster']['saveFig'] = f"{graph_folder}/raster_{cfg.ox}_{cfg.duration}ms.png"
cfg.analysis['plotTraces']['saveFig'] = f"{graph_folder}/traces_{cfg.ox}_{cfg.duration}ms.png"
