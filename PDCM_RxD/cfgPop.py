from netpyne import specs
import numpy as np
from neuron.units import sec, mM, M, s
import cv2
import json

# ------------------------------------------------------------------------------
#
# SIMULATION CONFIGURATION
#
# ------------------------------------------------------------------------------

# Run parameters
cfg = specs.SimConfig()  # object of class cfg to store simulation configuration
cfg.duration = 5000  # ms; matches the fit window so a run here scores directly against the
cfg.oldDuration = 5000  # study values ([3500,5000] -> trial 80 = 67.99). Raise to 10000+ to
                        # test whether the point holds past the fitted window.
cfg.restore = False  
cfg.hParams["celsius"] = 37.0
cfg.hParams["v_init"] = -70
cfg.v_balance = -70  # mV
cfg.Cm = 1.0  # pF/cm**2
cfg.Ra = 100
cfg.dt = 0.025  # Internal integration timestep to use
cfg.verbose = False  # Show detailed messages
cfg.recordStep = 0.1  # Step size in ms to save data (eg. V traces, LFP, etc)
cfg.savePickle = True  # Save params, network and sim output to pickle file
cfg.saveDataInclude = ["simConfig", "simData"]
cfg.compactConnFormat = True
cfg.saveJson = False
cfg.recordStim = False
cfg.SDThreshold = -40
# Threshold for recording sustained deploarization.

### Options to save memory in large-scale simulations
cfg.gatherOnlySimData = True  # Original
cfg.random123 = True
# set the following 3 options to False when running large-scale versions of the model (>50% scale) to save memory
cfg.saveCellSecs = False
cfg.saveCellConns = False
cfg.createPyStruct = False
cfg.printPopAvgRates = True
cfg.singleCells = False  # create one cell in each population
cfg.printRunTime = False  # will break save/restore via CVode events if True
cfg.Kceil = 15.0
cfg.nRec = 25

# ---- Early-warning / early-abort for parameter search ----
# SD is the tail of a slow ECS-K+ buildup; a candidate that will go into SD
# shows max ECS [K+] rising with positive slope well before any cell crosses
# SDThreshold. Abort such runs early instead of paying the full duration.
cfg.earlyAbort = True       # enable early-abort logic in runIntervalFunc
cfg.abortWarmup = 300.0     # ms; don't judge before this (past 200ms Poisson ramp + settle)
cfg.abortKmax = 15.0        # mM; max ECS [K+] above which, if still rising, we call SD-bound
                            # (15 = the batch value the trial-15 point was fit under; was 12)
cfg.abortKslope = 0.002     # mM/ms; required positive slope over window to confirm divergence
cfg.abortWindow = 100.0     # ms; trailing window used to estimate the slope
# Optional global firing-rate gates (Hz, mean over the trailing window). None = disabled.
cfg.abortMinRate = None
cfg.abortMaxRate = None
cfg.cellPops = [
    "L2e",
    "L2i",
    "L4e",
    "L4i",
    "L5e",
    "L5i",
    "L6e",
    "L6i",
]  # record only spikes of cells (not ext stims)
cfg.cellPopsInit = {"mean": -70, "std": 5, "thresh": -55}
cfg.recordCellsSpikes = cfg.cellPops
if cfg.recordStim:
    if cfg.poisson_ramp_ms > 0:
        cfg.recordCellsSpikes += [
            f"poissL{i}{ei}_{gidx}"
            for i in [2, 4, 5, 6]
            for ei in ["e", "i"]
            for gidx in range(cfg.poisson_ramp_split)
        ]
    else:
        cfg.recordCellsSpikes += [
            f"poissL{i}{ei}" for i in [2, 4, 5, 6] for ei in ["e", "i"]
        ]
    cfg.recordCellsSpikes += [f"bkg_THL{i}{ei}" for i in [4, 6] for ei in ["e", "i"]]

cfg.recordCells = [
    (f"L{i}{ei}", idx) for i in [2, 4, 5, 6] for ei in ["e", "i"] for idx in range(10)
]
cfg.recordTraces = {
    f"{var}_soma": {"sec": "soma", "loc": 0.5, "var": var}
    for var in ["v", "nai", "ki", "cli", "dumpi", "o2o", "o2i", "ATPi", "ADPi", "AMPi", "Posi", "volumei"]
}
cfg.scStartV = None  # single-cell test: if set, pin every cell's start voltage (mV)
cfg.iGnaScale = 1.0  # coupled gnabar scale for block-prone I-pops (L2i/L4i/L5i/L6i)
# Approach-A stability oracle: per-pop single-cell param overrides applied in
# netParamsPops.py. Batch sets scalar attrs named sc_<pop>_<gkbar|pmax|gnabar>
# (e.g. cfg.sc_L2i_gkbar = 0.006); unset pops keep their phase3 values.
cfg.seed = 0
cfg.seeds = {
    "conn": 2 + cfg.seed,
    "stim": 3 + cfg.seed,
    "loc": 4 + cfg.seed,
    "cell": 5 + cfg.seed,
    "rec": 1 + cfg.seed,
}
# Network dimensions
cfg.fig_file = "../test_mask.npy"
# img = cv2.imread(cfg.fig_file, cv2.IMREAD_GRAYSCALE)  # image used for capillaries
# img = np.rot90(img, k=3)
img = np.load(cfg.fig_file)
cfg.px = 0.2627  # side of image pixel (microns)
cfg.dx = 50  # side of ECS voxel (microns)
cfg.sizeX = 700  # img.shape[1] * cfg.px#250.0 #1000
# ---- cortical depth window taken from the histology (26Sep2026) --------------------------
# test_mask.npy is rot90(255-A22-312CD34NEUN_region_0_cellLabels_ed001.tif, k=3), so its ROWS
# are the tif's columns and run along cortical depth. Registered against the NeuN channel of
# A22-312CD34NEUN_region_0_original.tif (somata isolated by dark+saturated magenta):
#     row    0 (   0 um)  pia            -- somata near zero for the first 227 um (L1)
#     row  866 ( 227 um)  L1/L2 border   -- somata jump 0.012 -> 0.041 area fraction
#     row 8679 (2280 um)  GM/WM          -- somata collapse 0.025 -> 0.005
# The OLD crop img[1000:] started at 263 um, i.e. BELOW the L1/L2 border, and ran to the image
# edge at 2394 um -- so ynorm 0 was really the top of L2 (true ynorm 0.115), every layer sat
# ~0.12 of cortical depth too deep, and the deepest 114 um of "L6" sampled WHITE MATTER, where
# vessel density is about half. See notebookAI.org and capillary_layer_alignment.png.
# Now: start at the pia (row 0). Row count is UNCHANGED at 8113, so sizeY is bit-identical to
# before -- deliberate, because sizeY -> Vtissue -> cfg.rs -> cfg.somaR, and moving somaR by
# even 2% would invalidate every fitted conductance/pmax. Consequence: ynorm 1 lands at 2131 um
# = true ynorm 0.935, so the deepest 6.5% of L6 is not represented (better than including WM).
# To span pia->GM/WM exactly instead, set imgRow1 = 8679; that makes sizeY 2280 um and somaR
# 2.3% larger, and REQUIRES refitting the single-cell parameters.
cfg.imgRow0 = 0     # pia
cfg.imgRow1 = 8113  # keeps sizeY == (9113-1000)*px exactly; 8679 would reach the GM/WM border
cfg.sizeY = (cfg.imgRow1 - cfg.imgRow0) * cfg.px
cfg.sizeZ = cfg.sizeX  # 200.0
cfg.Nz = int(cfg.sizeZ / cfg.dx) - 1
cfg.Vtissue = cfg.sizeX * cfg.sizeY * cfg.sizeZ

# scaling factors
cfg.poissonRateFactor = 1.0
cfg.connected = True 


# slice conditions
cfg.ox = "perfused"
if cfg.ox == "perfused":
    cfg.o2_bath = 0.06  # ~36 mmHg
    cfg.o2_init = 0.04  # ~24 mmHg
    cfg.alpha_ecs = 0.2
    cfg.tort_ecs = 1.6
    # 25Sep2026: o2drive = 3.85, FITTED for a stable resting <O2> = 0.040 mM (was 10.0, before
    # that 2.0). Two 5 s runs of the SAME point (trial 14) at drive 2 and 10 calibrate the ECS
    # O2 balance  d<O2>/dt = K*o2drive*(o2_bath - <O2>) - D.  Joint least squares on both
    # trajectories (shared K and D; the runs differ only in o2drive and their rates agree to
    # <4%, so D is common) gives K = 0.0897 /s per unit drive and D = 0.00690 mM/s. Validated
    # by forward-integrating each run from t=1 s: predicted <O2>(4.9 s) = 0.0522 vs 0.0526
    # observed at drive 10, and 0.0299 vs 0.0293 at drive 2 -- both within 0.6 uM.
    #   steady state  <O2>ss = 0.06 - D/(K*o2drive):
    #     drive  2   -> 0.015 (in practice it never gets there: it falls ~linearly toward 0)
    #     drive  3.85-> 0.040   <-- resting target, o2_init
    #     drive  5   -> 0.045
    #     drive 10   -> 0.052  (measured 0.0526, still creeping up -- matches)
    #   tau = 1/(K*drive) = 2.9 s at 3.85, so it settles inside a 5 s run.
    #   D is +-20% uncertain -> the bracket is o2drive 3.1-4.6.
    # WHY THIS MATTERS -- the Kir, not ATP. The O2 gate is half-off at <O2> = gliaO2Half/32 =
    # 0.0156 mM. At drive 2 mean Kir capacity decayed 0.98 -> 0.83 over 5 s with 756/8428 voxels
    # past half-off; at drive 10 it held 0.994 with 2 voxels. ATP is NOT O2-limited (Ko2 = 0.3 uM
    # vs 12-33 uM tissue): the drive-10 rerun left every pop's ATP/Nai/AMP identical to 3 s.f.
    # (L6i -10.6% vs -10.7%), and the fitness was 91.77 vs 88.23 -- i.e. within noise.
    # Earlier note on the 10.0 setting: At 2.0 the capillary source cannot
    # hold resting tissue O2 in this model AT ALL: measured across the 10 safe 5 s trials,
    # ECS O2 falls ~linearly at 0.0023 mM/s from 0.04 with no levelling by 5 s, and the rate
    # is INDEPENDENT of firing (corr 0.22; slopes -0.00229..-0.00245 across sumHz 18-27), so
    # it is basal demand, not spiking. The source itself works (942/8428 voxels GAIN O2, field
    # spatial corr 0.88 from 1->4.9 s) but supply ~1.0e-3 vs demand ~3.4e-3 mM/s = ~3.4x short.
    # Term is numcap*o2drive*eps_o2*(o2_bath - O2), i.e. linear in o2drive, and ~11% of voxels
    # carry capillaries -> o2drive ~8-12 to hold 0.03-0.04 mM. 10.0 is the midpoint.
    # WHY IT MATTERS (not ATP): Ko2 = 0.3 uM vs o2i 12-33 uM, so ATP synthase stays 97.5-99.1%
    # O2-saturated and ADP is the limiting substrate -- the L6i ATP droop is NOT an O2 problem.
    # The casualty is the glial Kir, whose O2 gate is half-off at ECS O2 = gliaO2Half/32 =
    # 0.0156 mM: mean Kir capacity falls 0.98 -> 0.83 over 5 s, with 756 voxels below half by
    # 4.9 s. That is the SD-suppression mechanism decaying underneath every trial.
    cfg.o2drive = 3.85  # fitted above; 2.0 cannot hold resting O2 at all, 10.0 overshoots to 0.052
elif cfg.ox == "hypoxic":
    cfg.o2_bath = 0.06  # ~4 mmHg
    cfg.o2_init = 0.005
    cfg.alpha_ecs = 0.07
    cfg.tort_ecs = 1.8
    cfg.o2drive = 0.1 / 6  # 0.013 * (1 / 6)
cfg.prep = "invivo"  # "invitro"
# Size of Network. Adjust this constants, please!
cfg.ScaleFactor = 0.16  # used for batch param search  # = 80.000

# neuron params
cfg.betaNrn = (
    0.29  # 0.59 intracellular volume fraction (Rice & Russi-Menna 1997) ~80% neuronal
)
cfg.N_Full = [20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948, 902]
cfg.Ncell = sum([max(1, int(i * cfg.ScaleFactor)) for i in cfg.N_Full])
# Single cell parameter based on Scale 0.16
cfg.NcellRxD = sum([max(1, int(i * 0.16)) for i in cfg.N_Full])
cfg.rs = ((cfg.betaNrn * cfg.Vtissue) / (2 * np.pi * cfg.NcellRxD)) ** (1 / 3)

cfg.epas = -70.00000000000013  # False
cfg.sa2v = 3.4  # False


cfg.kleakMin = 1e-5  # mS/cm^2 -- this may changed pmax
# Neuron parameters
params = json.load(open("phase3_pump_results.json", "r"))
cfg.pmax_scale = 1.05  # jointFitKir winner (was 1)
cfg.gkbar_scale = 1
cfg.gnabar_scale = 1   # global gnabar scale (fit by batchJointFitKir to lift E-cell firing at Nai=13)
for k, v in params.items():
    if hasattr(v, "keys"):
        for pop, value in v.items():
            if hasattr(cfg,k):
                getattr(cfg, k)[pop] = value
            else:
                setattr(cfg, k, {pop:value})
    else:
        setattr(cfg, k, v)
# Published cotransporter rates (Wei et al. / cfgMidOx) as the network anchor, replacing
# phase3's unreliable fitted values (drifted 30-100x). Phase-1 refits ukcc2/unkcc1 per-pop
# (KCC2 shapes inhibition via ECl); update these dicts with the phase-1 results. unkcc1=0.1
# also keeps the Cl- leak feasible at Nai=10 (phase3's ~7 caused negative gclbar_l).
_pops8 = ["L2e", "L2i", "L4e", "L4i", "L5e", "L5i", "L6e", "L6i"]
cfg.ukcc2 = {p: 0.3 for p in _pops8}
cfg.unkcc1 = {p: 0.1 for p in _pops8}
cfg.setVinit = False # 900 #ms
"""
cfg.v_initial = {
    "L2e": -68.96740764517631,
    "L2i": -70.86630055249161,
    "L4e": -68.33055040581962,
    "L4i": -67.29443078799166,
    "L5e": -66.30655004521006,
    "L5i": -69.65019972863567,
    "L6e": -68.70245023293091,
    "L6i": -67.18512742545232
}
"""
cfg.excWeight_L2e *= 0.63
cfg.excWeight_L2i *= 1.3 

cfg.excWeight_L4e *= 0.53
cfg.excWeight_L4i *= 1.4

cfg.excWeight_L5e *= 2.5
cfg.excWeight_L5i *= 1.26

cfg.excWeight_L6e *= 0.09
cfg.excWeight_L6i *= 0.44

"""
cfg.inhWeightScale_L2e = 7.353391928790916
cfg.inhWeightScale_L2i = 9.586468480064418
cfg.inhWeightScale_L4e = 9.585523901837426
cfg.inhWeightScale_L4i = 2.6998275888570147
cfg.inhWeightScale_L5e = 5.9341219428822845
cfg.inhWeightScale_L5i = 6.776965723613618
cfg.inhWeightScale_L6e = 6.037892310881581
cfg.inhWeightScale_L6i = 7.769695037938091
"""

# parameter updated to fix early spontaneous SD
cfg.gkbar["L6i"] = 0.008


# default values
cfg.weightMin = 0.1
cfg.dWeight = 0.1
cfg.scaleConnWeightNetStims = 1   # EXTERNAL (NetStim) exc weight only -- input variance floor
cfg.scaleConnWeightNetStimStd = 1
# RECURRENT (bio->bio) excitatory weight multiplier. Distinct from scaleConnWeightNetStims,
# which only touches the artificial external drive. The firing rundown is a RECURRENT
# variance collapse (E-cells stop supplying each other's input variance, var ~ w^2*rate),
# so this is the lever that strengthens the self-sustaining loop. Inert at 1.0.
cfg.excWeightScale = 1.0

# ============== jointFitExcW5s trial 94 -- CURRENT BEST, fitness 49.93 ==================
# 28Sep2026, 14 dims, scored on [3500,5000] ms. Exact optuna floats, verified against BOTH the
# study DB and gen_94/trial_94_cfg.json (0 mismatches).
#   rate=49.7 slope=0.18 -- the flattest run in any study to date; 6 of 8 pops flat or rising
#   over the fit window.
#   rates/target on [3.5,5]s: L2e .05/.48  L2i 2.17/2.36  L4e 2.57/3.85  L4i 7.30/4.94
#                             L5e 3.34/6.51 L5i 5.74/7.65 L6e .10/.57  L6i 6.31/6.86
#   Beat trial 80 (67.99) almost entirely on L5e: 1.79 -> 3.34 Hz, penalty 36.5 -> 16.5.
#   NB L5e's WHOLE-RUN netstats average is only ~2.0 Hz -- it is still rising through the run,
#   so judge this point on the [3.5,5] s window, not on netstats_5.00s.json.
#   Residual is now split about evenly: L5e 16.5 (51% of target) and L4i 15.9 (148%, an
#   OVERSHOOT -- the L4i problem is now to rein it in, not to drive it). L4e 7.7 third.
# Keeps trial 80's two key features -- high excWeight_L6i (~3.2x) and high inhWeightScale_L4e
# (~7.7) -- and adds excWeight_L4e/L5e at ~1.2x with excWeightScale driven back to ~1.0, i.e.
# the per-population gains do the work and the global recurrent E->E boost is inert.
# THIS BLOCK MUST STAY BELOW `cfg.excWeightScale = 1.0` above, or that default overwrites the
# fitted value. The excWeight_* lines are ABSOLUTE and override the `*=` products further up:
#   L4i 1.967x   L6i 3.154x   L6e 1.422x   L4e 1.179x   L5e 1.214x   of their base values.
cfg.inhWeightScale_L2e = 2.8572469166177648
cfg.inhWeightScale_L2i = 8.565025449785882
cfg.inhWeightScale_L4e = 7.74684517203956
cfg.inhWeightScale_L4i = 0.5390143539347368
cfg.inhWeightScale_L5e = 5.8835489559460665
cfg.inhWeightScale_L5i = 3.552837813546092
cfg.inhWeightScale_L6e = 4.774358342235912
cfg.inhWeightScale_L6i = 7.796227662893626
cfg.excWeight_L4i = 0.006664649293926673
cfg.excWeight_L6i = 0.008224789533168414
cfg.excWeight_L6e = 0.0008227744795595474
cfg.excWeight_L4e = 0.0026732998331460128
cfg.excWeight_L5e = 0.00953630955148853
cfg.excWeightScale = 1.009501716745112
# =========================================================================================

"""
# original model
cfg.gnabar = 30 / 100
cfg.gkbar = 25 / 1000
cfg.ukcc2 = 0.3
cfg.unkcc1 = 0.1
cfg.pmax = 3
cfg.gpas = 0.0001
"""
cfg.ATPss = 3.18  # mM PMC3524514 -- whole brain
cfg.ATPDc = 0.445  # um**2/ms
cfg.Ko2 = 0.3e-3  # mM  # Km for O2 at cytochrome c oxidase
cfg.KmADP_synthase = 0.022  # mM, from PMC3833997 (human skeletal muscle)
cfg.KmADP_synthase_hc = 1.9  
cfg.KmPi_synthase = 1.0  # mM, from PMC8434986 (cardiac tissue)
#cfg.KiATP_synthase = (
#    1.0  # mM, competitive inhibition constant for ATP (allows steady-state flux)
#)
cfg.ADPss = 0.094444444444444  # such that D2 (MgADP == 0.05 mM)
cfg.tauADP = 1
cfg.Pss = 4.2
cfg.tauP = 1
cfg.ATPase_basal_density = 0.05  # mM/ms
# Diagnostic knob: scales the ATP-synthase (ATPRestore) rate. 1.0 = normal;
# 0.0 disables ATP production to bisect the adenine-pool non-conservation
# (watch adenine_pool.txt -- Σ should go flat if ATPRestore is the source).
cfg.atpRestoreScale = 1.8
# Companion diagnostic knobs (each default 1.0). Set ALL FOUR (incl. atpRestoreScale)
# to 0.0 to disable every adenine reaction at once: if adenine_pool.txt still climbs,
# the injection is at the RxD-solver level, not the chemistry. pump-ADP is scaled
# independently of the Na/K pump so ion dynamics are unaffected.
cfg.basalATPScale = 1.0   # basal ATP hydrolysis (BasalATP)
cfg.pumpADPScale = 1.0    # pump ATP->ADP branch only (pump_current_ADP)
cfg.adkScale = 1.0        # adenylate kinase (AMP+ATP <-> 2ADP)

# Adenylate kinase equilibrium: 2*ADP <-> ATP + AMP
# At equilibrium: Keq = [ATP][AMP]/[ADP]^2 ≈ 1 (typical for adenylate kinase)
# Solving: AMP = Keq * ADP^2 / ATP = 1.0 * (0.05)^2 / 2.59 ≈ 0.001 mM
# Solving adenylateKinase rate_f == rate_b at steady-state gives exact value.
cfg.AMPss = (
    0.0692795435459248  # mM, from adenylate kinase equilibrium with ADPss and ATPss
)
cfg.Mg = 0.5  # mM (free Mg) https://doi.org/10.3390/ijms20143439

cfg.glia = {
    "nai": 55.0 * mM,
    "ki": 80.0 * mM,
    "ATP": 10 * mM,
    "ADP": 10 / 15.4 * mM,
    "Pos": cfg.Pss,
}
cfg.pH = 7.0
cfg.NaKPump = {
    "Tref": 310,
    "q10": 3.2,
    "Delta": -0.031,
    "k1p": 1050 / s,
    "k1m": 172.1 / s / mM,
    "k2p": 481 / s,
    "k2m": 40 / s,
    "k3p": 2000 / s,
    "k3m": 79.3e3 / s / mM**2,
    "k4p": 320 / s,
    "k4m": 40 / s,
    "KATP": 2.51 * mM,
    "KHPi": 6.77 * mM,
    "KKPi": 292 * mM,
    "KNaPi": 224 * mM,
    "PiT": 4.2 * mM,
    "KKe": 0.213 * mM,
    "KKi": 0.5 * mM,
    "KNae0": 15.5 * mM,
    "KNai0": 2.49 * mM,
}

# Glial Kir4.1 K+-uptake sigmoid: activation = 1/(1+exp((gliaKHalf - [K+]_ecs)/gliaKSlope))
# The legacy half-activation of 18 mM (Cressman et al. 2009) is phenomenological and
# sits ABOVE the entire non-SD physiological range (seizure ceiling ~10-12 mM; Heinemann
# & Lux 1977), so glial clearance never reaches half-max during normal activity. Measured
# glial Na/K-ATPase K0.5 ~3.6 mM (Larsen 2014). Lowering to ~12 mM engages clearance as
# [K+]_ecs approaches the SD threshold while staying near-silent at the 3 mM resting level.
# Primary search axis for suppressing spontaneous SD; sweep 8-18 mM.
cfg.gliaKHalf = 9.61539501826942   # mM; PINNED in jointFitExcW (trial 23 winner of jointFitKirNai13Fixed);
                                   # the trial-15 weights above were fit at exactly this value
cfg.gliaKSlope = 2.5   # mM; sigmoid slope (Cressman value; keep fixed)

# Glial Kir O2 gate: g_glia = g_gliamax/(1+exp(-((o2ecs*32) - gliaO2Half)/gliaO2Slope)).
# o2ecs*32 converts mM -> mg/mL. The legacy half-point of 2.5 mg/mL (o2ecs=0.078 mM
# ~48 mmHg) sits ABOVE perfused tissue O2 (init 0.04 -> 1.28 mg/mL ~24 mmHg;
# bath 0.06 -> 1.92 mg/mL), so the Kir term is ~0.2-5% ON and gliaKHalf is inert.
# Lowering gliaO2Half turns Kir on in normoxia (fail only under ischemia, as intended).
# NOTE: default kept at the legacy 2.5 so parameterization is behavior-preserving;
# batchGliaO2gate.py sweeps it downward. Lower it here once the sweep picks a value.
cfg.gliaO2Half = 0.5   # mg/mL; jointFitKir winner -- Kir ON in normoxia (was 2.5)
cfg.gliaO2Slope = 0.2  # mg/mL; O2-gate steepness (keep fixed)

cfg.Ggliamax = 5.0  # mM/sec originally 5mM/sec
# we scaled pump by ~4.84 so apply a corresponding
# reduction by channels (K, Kir and NKCC1) in glia.

cfg.gkleak_scale = 1
cfg.KKo = 5.3
cfg.KNai = 27.9
# Scaled to match original 1/3 scaling at Ko=3; i.e.
# a = 3*(1 + np.exp(3.5-3))
# GliaKKo = np.log(a-1) + 3
cfg.GliaKKo = 3.5  # 4.938189537703508  # originally 3.5 mM
cfg.GliaPumpScale = 1.0  # jointFitKir winner (was 1/3)
cfg.scaleConnWeight = 1

# converstionFactor: μmol·min−1·mg−1 -> mM/ms
converstionFactor = 49 * 9.7e-7 / 16000  # mg of enzyme/m^3
converstionFactor *= 60e3 * 1e6  # μmol/min -> mol/ms
#    1g tissue = 9.7e-7 m^3
#    49 units/g  (of tissue) brain
# 1,600 units/mg (of enzyme) muscle
# unit 1 μmol/min
cfg.AK = {
    "KmAMP": 0.12,  # mM
    "KiAMP": 3.3,  # mM
    "KmMgATP": 0.06,  # mM
    "KmADP": 0.028,  # mM
    "KiADP": 0.91,  # mM
    "KmMgADP": 0.033,  # mM
    "kp1": 14_000 * converstionFactor,  # mM/ms
    "km1": 8_000 * converstionFactor,  # mM/ms
    "kp2": 710 * converstionFactor,  # mM/ms
    "km2": 960 * converstionFactor,  # mM/ms
    "KMg": 2.5,  # /mM (stability constant)
}


if cfg.sa2v:
    cfg.somaR = (cfg.sa2v * cfg.rs**3 / 2.0) ** (1 / 2)
else:
    cfg.somaR = cfg.rs
cfg.cyt_fraction = cfg.rs**3 / cfg.somaR**3

# sd init params
cfg.k0 = 3.5 
cfg.r0 = 100
cfg.k0Layer = None # layer of elevated extracellular K+

###########################################################
# Network Options
###########################################################

# DC=True ;  TH=False; Balanced=True   => Reproduce Figure 7 A1 and A2
# DC=False;  TH=False; Balanced=False  => Reproduce Figure 7 B1 and B2
# DC=False ; TH=False; Balanced=True   => Reproduce Figure 8 A, B, C and D
# DC=False ; TH=False; Balanced=True   and run to 60 s to => Table 6
# DC=False ; TH=True;  Balanced=True   => Figure 10A. But I want a partial reproduce so I guess Figure 10C is not necessary


# External input DC or Poisson
cfg.DC = False  # True = DC // False = Poisson

# Thalamic input in 4th and 6th layer on or off
cfg.TH = True  # True = on // False = off

# Balanced and Unbalanced external input as PD article
cfg.Balanced = True  # False #True=Balanced // False=Unbalanced

cfg.poisson_ramp_ms = 200  # ramp up Poisson drive over the start of the sim to avoid a large initial population spike.
cfg.poisson_ramp_split = 10  # split the cells into groups

cfg.ouabain = False

simLabel = "best94_o2d3.85" #f"jointPool_v_balance{cfg.v_balance}_ramp{cfg.poisson_ramp_ms}_{cfg.seed}_layer{cfg.k0Layer}_K0{cfg.k0}_{cfg.prep}_o2d{cfg.o2drive}_o2b_{cfg.o2_init}"
cfg.simLabel = f"{simLabel}_{cfg.duration/1000:0.2f}s"
cfg.saveFolder = f"./data/{simLabel}_{cfg.oldDuration/1000:0.2f}s"
# cfg.simLabel = f"test_{cfg.ox}"
# cfg.saveFolder = f"/tmp/test"
# cfg.saveFolder = f"/tera/adam/{cfg.simLabel}/" # for neurosim
cfg.restoredir = cfg.saveFolder if cfg.restore else None
# v0.0 - combination of cfg from ../uniformdensity and netpyne PD thalamocortical model
# v1.0 - cfg for o2 sources based on capillaries identified from histology
