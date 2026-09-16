"""
Tier 3a: the trimmed o2fix lineage at o2drive=10 (INTERMEDIATE delivery, 3000 ms).

Closes the Tier 2.5 gap. Tier 2.5 proved the L2/L4 capillary-distance protection
survives the domain trim at o2drive=100 (strong delivery, wall_dist coeff -> n.s.,
cap_dist p=3e-17). But the intermediate regime (o2drive=10) is where the RAW L2
near->far gap peaks (~+10%, memory pdcm_rxd_capillary_distance_reframe). This run
lets us re-run analysis_tier25_wall_confound.py at d10 and confirm the effect is
still wall-independent where it is LARGEST -- the strongest physiological-side
statement we can make without any model-design change.

Pure additive delta over cfgPopWei_o2drive1_ecs2_trim.py (which folds o2fix physics
+ v4 depth trim + frozen Vtissue/rs/somaR). Only raises delivery and matches the
untrimmed comparator's production length:
  * o2drive 1 -> 10   (matches data/o2sweep_d10p0 comparator)
  * duration 2500 -> 3000 ms (matches data/o2sweep_d10p0 production length)
Runs under run_v2_o2fix_trim.py / netParams_v2_o2fix_trim.py. Output isolated.
"""
import os
from cfgPopWei_o2drive1_ecs2_trim import cfg

cfg.o2drive    = 10           # << intermediate delivery, matches untrimmed d10 comparator
cfg.duration   = 3000         # match data/o2sweep_d10p0 production length
cfg.simLabel   = f"SDL13_37_{cfg.ox}_{cfg.duration}ms_o2drive10_ecs2_trim"
cfg.saveFolder = "./data/o2drive10_o2fix_trim"
cfg.restoredir = cfg.saveFolder if cfg.restore else None

graph_folder = "./graphs/o2drive10_o2fix_trim"
os.makedirs(graph_folder, exist_ok=True)
cfg.graph_folder = graph_folder
cfg.analysis['plotRaster']['saveFig'] = f"{graph_folder}/raster_{cfg.ox}_{cfg.duration}ms.png"
cfg.analysis['plotTraces']['saveFig'] = f"{graph_folder}/traces_{cfg.ox}_{cfg.duration}ms.png"
