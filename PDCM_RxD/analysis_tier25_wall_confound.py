"""
Tier 2.5 -- disentangle L2 capillary-distance protection from the x/z box-wall artifact.

The L2 "far-from-capillary cells fail more" signal is confounded: in L2, far-from-capillary
also correlates with near-the-x/z-tissue-wall, where no-flux O2 boundary-trapping occurs
(memory pdcm_rxd_v4_trim_l2_wall_test / pdcm_rxd_o2_band_x_gradient). This script tests
whether the capillary-distance effect is REAL local physiology or the wall in disguise, via
two independent probes:

  (A) WALL-STRATIFIED near/far split. Within a layer, split cells into near-wall vs
      interior (median x/z-wall distance), then inside EACH wall stratum recompute the
      capillary near/far SD gap. If the gap persists in the INTERIOR stratum (cells far
      from any wall), the capillary effect is not a wall artifact.

  (B) LOGISTIC REGRESSION  P(SD) ~ capillary_dist + wall_dist  (per layer). Reports the
      partial capillary-distance coefficient (effect at fixed wall distance) with p-value.

  (C) TRIM TEST. Re-run (A)/(B) on the depth-trimmed d100 domain
      (data/o2drive100_o2fix_trim) which removes the acellular pia/WM O2-accumulation
      bands, and compare the L2 capillary gap trimmed-vs-untrimmed. If the trim collapses
      the gap it was (top/bottom) boundary; if it survives, it is real.

x/z-wall distance = min(x, sizeX-x, z, sizeZ-z) using the O2-grid extent (nx*DX, nz*DX).

Usage:  python analysis_tier25_wall_confound.py
"""
import os, json, glob, re
import numpy as np
from scipy import ndimage
import statsmodels.api as sm
from sd_onset_utils import load_filtered_onsets

REAL = re.compile(r"^L\d+[ei]$")
DX = 10.0
START_MS = 50.0
GUARD_MS = 2.5

# (label, dir, duration_ms)
DATASETS = [
    ("d1  TRIMMED (physiological)", "data/hypoxic_o2drive1_ecs2_trim", 2500),
    ("d10 TRIMMED (intermediate)",  "data/o2drive10_o2fix_trim", 3000),
    ("d100 untrimmed",              "data/o2sweep_d100p0", 3000),
    ("d100 TRIMMED",                "data/o2drive100_o2fix_trim", 3000),
]


def layer_of(pop):
    m = re.match(r"^(L\d+)[ei]$", pop)
    return m.group(1) if m else "?"


def build_rows(data_dir, duration):
    """Per real neuron: capillary-dist, x/z-wall-dist, layer, failed(bool)."""
    pos = json.load(open(glob.glob(os.path.join(data_dir, "network_positions_*.json"))[0]))
    onsets = load_filtered_onsets(data_dir, duration, GUARD_MS, start_ms=START_MS)
    fs = sorted(glob.glob(os.path.join(data_dir, "o2_*.npy")),
                key=lambda p: int(p.split("_")[-1].split(".")[0]))
    early = np.load(fs[0])
    src = early >= np.quantile(early, 0.90)
    dist = ndimage.distance_transform_edt(~src) * DX
    nx, ny, nz = early.shape
    xmax, zmax = (nx - 1) * DX, (nz - 1) * DX

    rows = []
    for k, info in pos.items():
        if not (isinstance(info, dict) and REAL.match(str(info.get("pop", "")))):
            continue
        gid = int(k)
        xv = min(int(info["x"] / DX), nx - 1)
        yv = min(int(abs(info["y"]) / DX), ny - 1)
        zv = min(int(info["z"] / DX), nz - 1)
        capd = float(dist[xv, yv, zv])
        xu, zu = float(info["x"]), float(info["z"])
        walld = min(xu, xmax - xu, zu, zmax - zu)
        rows.append((capd, walld, layer_of(info["pop"]), gid in onsets))
    return rows


def near_far_gap(cap, fail):
    """SD rate for near (<=median cap-dist) vs far (>median); returns (near, far, far-near, n)."""
    if len(cap) < 20:
        return None
    med = np.median(cap)
    near = fail[cap <= med]
    far = fail[cap > med]
    if len(near) == 0 or len(far) == 0:
        return None
    return near.mean(), far.mean(), far.mean() - near.mean(), (len(near), len(far))


def analyze(label, rows):
    cap = np.array([r[0] for r in rows])
    wall = np.array([r[1] for r in rows])
    lay = np.array([r[2] for r in rows])
    fail = np.array([r[3] for r in rows], dtype=float)
    print(f"\n{'='*72}\n{label}: {len(rows)} real neurons | SD {fail.mean():.1%} | "
          f"wall-dist median {np.median(wall):.0f}um")

    for L in ["L2", "L4", "L6", "L5"]:
        m = lay == L
        if m.sum() < 40:
            continue
        capL, wallL, failL = cap[m], wall[m], fail[m]
        g = near_far_gap(capL, failL)
        print(f"\n  --- {L} (n={m.sum()}, SD {failL.mean():.1%}) ---")
        if g:
            print(f"    RAW cap near->far: {g[0]:.1%} -> {g[1]:.1%}  Delta={g[2]:+.1%}  (n {g[3]})")

        # (A) wall-stratified: interior stratum = cells FAR from any wall
        wmed = np.median(wallL)
        for sname, smask in [("near-WALL", wallL <= wmed), ("INTERIOR", wallL > wmed)]:
            gs = near_far_gap(capL[smask], failL[smask])
            if gs:
                print(f"    [{sname:9s}] cap near->far: {gs[0]:.1%} -> {gs[1]:.1%}  "
                      f"Delta={gs[2]:+.1%}  (n {gs[3]})")

        # (B) logistic P(SD) ~ cap + wall  (standardized predictors)
        try:
            X = np.column_stack([
                (capL - capL.mean()) / (capL.std() + 1e-9),
                (wallL - wallL.mean()) / (wallL.std() + 1e-9),
            ])
            X = sm.add_constant(X)
            res = sm.Logit(failL, X).fit(disp=0)
            bcap, pcap = res.params[1], res.pvalues[1]
            bwall, pwall = res.params[2], res.pvalues[2]
            print(f"    LOGIT  cap_dist beta={bcap:+.3f} (p={pcap:.1e}) | "
                  f"wall_dist beta={bwall:+.3f} (p={pwall:.1e})  [+beta => farther=more SD]")
        except Exception as e:
            print(f"    LOGIT failed: {e}")


if __name__ == "__main__":
    os.chdir("/nfs/roberts/project/pi_rm693/tys7/newton/PDCM_RxD")
    for label, d, dur in DATASETS:
        if not os.path.isdir(d):
            print(f"\n[skip] {label}: {d} missing")
            continue
        analyze(label, build_rows(d, dur))
    print()
