"""
MATCHED-REGIME test: the trimmed o2fix lineage at o2drive=100 (strong delivery),
full 3000 ms. Thread #5 follow-up.

Purpose: re-run analysis_failing_subpop.py on THIS trimmed run and compare the L2
near-x/z-wall SD-FAILING enrichment against the UNTRIMMED o2drive=100 comparator
data/o2sweep_d100p0 (graphs/subpop_map: L2 med_dist_fail 45.1 vs 101.0 um,
p=2.6e-48, wall-hugging FAILURE). This tests whether the v4 depth-trim removes that
near-wall FAILING artifact in the SAME operating point where it was observed --
unlike the hypoxic v3/v4 pair, which is the opposite-sign regime (near-wall cells
SURVIVE there). See memory pdcm_rxd_v4_trim_l2_wall_test.

Pure additive delta over cfgPopWei_o2drive1_ecs2_trim.py:
  * inherits o2fix physics (tortuous ecs_o2) + the frozen-Vtissue/rs/somaR depth
    trim (sizeY 2131.2851 -> 1798.7745 um),
  * raises delivery o2drive 1 -> 100 to match the subpop_map comparator,
  * duration 2500 -> 3000 ms to match data/o2sweep_d100p0 production length
    (t=2900 dumps).
Runs under run_v2_o2fix_trim.py / netParams_v2_o2fix_trim.py (which bakes
cfg.o2drive into the O2 source rate at import).
"""
import os
from cfgPopWei_o2drive1_ecs2_trim import cfg

cfg.o2drive    = 100          # << match the untrimmed d100 comparator (subpop_map)
cfg.duration   = 3000         # match data/o2sweep_d100p0 production length
cfg.simLabel   = f"SDL13_37_{cfg.ox}_{cfg.duration}ms_o2drive100_ecs2_trim"
cfg.saveFolder = "./data/o2drive100_o2fix_trim"
cfg.restoredir = cfg.saveFolder if cfg.restore else None

graph_folder = "./graphs/o2drive100_o2fix_trim"
os.makedirs(graph_folder, exist_ok=True)
cfg.graph_folder = graph_folder
cfg.analysis['plotRaster']['saveFig'] = f"{graph_folder}/raster_{cfg.ox}_{cfg.duration}ms.png"
cfg.analysis['plotTraces']['saveFig'] = f"{graph_folder}/traces_{cfg.ox}_{cfg.duration}ms.png"
