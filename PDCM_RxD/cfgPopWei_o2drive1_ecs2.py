from netpyne import specs
import numpy as np
from neuron.units import sec, mM, M, s
import cv2
import json
import os

# ------------------------------------------------------------------------------
#
# SIMULATION CONFIGURATION
#
# "Run C" ECS-loosened O2 supply sweep (PI-requested follow-up to Run B):
# after seeing that o2drive={1,4,7,10} under the default hypoxic ECS
# (alpha_ecs=0.07, tort_ecs=1.8) barely changed local O2 or SD onset, PI's
# read was that the tight ECS -- not the supply rate -- was the bottleneck:
# any extra O2 gets diluted into so little volume / diffuses so poorly that
# local consumption saturates it regardless of drive. This variant borrows
# the *perfused* condition's ECS geometry (alpha_ecs=0.2, tort_ecs=1.6)
# while keeping cfg.ox="hypoxic" (o2_bath/o2_init unchanged), and sweeps
# cfg.o2drive over {1, 10, 100} to look for a dose-response curve: PI's
# hypothesis is a graded outcome (insufficient O2 -> SD as before; ample O2
# -> recovery; intermediate -> delayed SD). See also
# cfgPopWei_o2drive{10,100}_ecs2.py for the other sweep points, and the
# original cfgPopWei_o2drive{1,4,7,10}.py for the tight-ECS baseline sweep.
# Uses normal run_v2.py (capillary-distance-matched placement).
#
# ------------------------------------------------------------------------------

# Run parameters
cfg = specs.SimConfig()  # object of class cfg to store simulation configuration
cfg.duration = 3000  # Duration of the simulation, in ms
cfg.oldDuration = 500
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
    for var in [
        "v",
        "nai",
        "ki",
        "cli",
        "o2_consumedi",
        "nao",
        "ko",
        "clo",
        "o2o",
        "o2_consumedo",
        "o2", # added 7/6
    ]
}
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
cfg.px = 0.2627  # side of image pixel (microns) ## pixel to micron scale factor
cfg.dx = 10  # side of ECS voxel (microns) ##ECS voxel side, microns
cfg.sizeX = 700  # img.shape[1] * cfg.px#250.0 #1000 ## Hardcoded to be 700 instead of image derived
cfg.sizeY = (img.shape[0] - 1000) * cfg.px  # 250.0 #1000 ## still image derived
cfg.sizeZ = cfg.sizeX  # 200.0 ## inherits the same hardcode as sizeX
cfg.Nz = int(cfg.sizeZ / cfg.dx) - 1
cfg.Vtissue = cfg.sizeX * cfg.sizeY * cfg.sizeZ ## derived from Vtissue

# scaling factors
cfg.poissonRateFactor = 1.0
cfg.connected = True


# slice conditions
cfg.ox = "hypoxic"
if cfg.ox == "perfused":
    cfg.o2_bath = 0.06  # ~36 mmHg
    cfg.o2_init = 0.04  # ~24 mmHg
    cfg.alpha_ecs = 0.2
    cfg.tort_ecs = 1.6
    cfg.o2drive = 2.5  # 0.013
elif cfg.ox == "hypoxic":
    cfg.o2_bath = 0.06  # ~4 mmHg, changed from 0.06 to 0.005 just for fun
    cfg.o2_init = 0.005
    cfg.alpha_ecs = 0.07
    cfg.tort_ecs = 1.8
    cfg.o2drive = 1.0 / 6  # 0.013 * (1 / 6)
# PI-requested ECS-loosening override: borrow the perfused condition's ECS
# geometry (looser volume fraction + lower tortuosity) while staying on the
# hypoxic o2_bath/o2_init, to test whether the tight hypoxic ECS was masking
# any dose-response effect of o2drive.
cfg.alpha_ecs = 0.2
cfg.tort_ecs = 1.6
cfg.o2drive = 1  # Run C sweep point
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


cfg.kleakMin = 5e-5  # mS/cm^2 -- this may changed pmax
# Neuron parameters
# Scale synapses weights -- optimized with min K-leak 1e-5
params = json.load(open("phase2_cfg_results.json", "r"))
for k, v in params.items():
    value = v if "inhWeightScale" in k else v
    setattr(cfg, k, value)

# replace L2e and L4e with phase 3 optimization values
# these avoid -ve na leaks
params = json.load(open("phase3_cfg_results.json", "r"))
for k, v in params.items():
    if hasattr(v, "keys"):
        for pop, value in v.items():
            getattr(cfg, k)[pop] = value
    else:
        setattr(cfg, k, v)

cfg.excWeight_L2e *= 1.15  # 1.5
cfg.excWeight_L2i *= 3.0
cfg.inhWeightScale_L2i *= 0.8

cfg.excWeight_L4e *= 3.5
cfg.excWeight_L4i *= 1.5
cfg.inhWeightScale_L4i *= 0.75

cfg.excWeight_L5e *= 3.0
cfg.inhWeightScale_L5e *= 0.75
cfg.excWeight_L5i *= 1.75
cfg.inhWeightScale_L5i *= 0.75


cfg.excWeight_L6e /= 2.5
cfg.inhWeightScale_L6e *= 3.0

# default values
cfg.weightMin = 0.1
cfg.dWeight = 0.1
cfg.scaleConnWeightNetStims = 1
cfg.scaleConnWeightNetStimStd = 1

"""
# original model
cfg.gnabar = 30 / 100
cfg.gkbar = 25 / 1000
cfg.ukcc2 = 0.3
cfg.unkcc1 = 0.1
cfg.pmax = 3
cfg.gpas = 0.0001
"""
cfg.pH = 7.0

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
cfg.GliaPumpScale = 1 / 3  # 1 / 3  # originally 1/3
cfg.scaleConnWeight = 1

if cfg.sa2v:
    cfg.somaR = (cfg.sa2v * cfg.rs**3 / 2.0) ** (1 / 2)
else:
    cfg.somaR = cfg.rs
cfg.cyt_fraction = cfg.rs**3 / cfg.somaR**3

# sd init params
cfg.k0 = 3.5
cfg.r0 = 100
cfg.k0Layer = None  # layer of elevated extracellular K+

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

cfg.poisson_ramp_ms = 100  # ramp up Poisson drive over the start of the sim to avoid a large initial population spike.
cfg.poisson_ramp_split = 10  # split the cells into groups

cfg.ouabain = False

# added 7/8
# The file name: e.g., "SDL13_37_hypoxic_3000ms"
cfg.simLabel = f"SDL13_37_{cfg.ox}_{cfg.duration}ms_o2drive1_ecs2"

# The master data folder -- isolated from the other hypoxic_* runs
# (see module docstring above).
cfg.saveFolder = "./data/hypoxic_o2drive1_ecs2"

# ==============================================================
# ROUTE THE GRAPHS INTO THEIR OWN FOLDERS
# ==============================================================
graph_folder = "./graphs/hypoxic_o2drive1_ecs2"
import sys
# Only create this folder when THIS config is the one actually being run (its
# filename is on the command line via simConfig=...), not when a derived config
# (_o2fix, _disconn) or an analysis script imports it just for `cfg` -- those
# imports used to leave an empty graphs/hypoxic_o2drive1_ecs2 shell behind.
if any(a.endswith("cfgPopWei_o2drive1_ecs2.py") for a in sys.argv):
    os.makedirs(graph_folder, exist_ok=True) # Automatically creates the folder safely
cfg.graph_folder = graph_folder  # exposed on cfg so run_v2.py's end-of-run plots (e.g. o2 heatmap) can find it too

cfg.analysis['plotRaster'] = {
    'saveFig': f"{graph_folder}/raster_{cfg.ox}_{cfg.duration}ms.png",
    'timeRange': [0, cfg.duration]
}
cfg.analysis['plotTraces'] = {
    'include': [0],
    'figSize': (10, 24),  # default (10,8) overlaps badly with 11 recordTraces vars
    'saveFig': f"{graph_folder}/traces_{cfg.ox}_{cfg.duration}ms.png",
    'timeRange': [0, cfg.duration]
}

cfg.restoredir = cfg.saveFolder if cfg.restore else None
# v0.0 - combination of cfg from ../uniformdensity and netpyne PD thalamocortical model
# v1.0 - cfg for o2 sources based on capillaries identified from histology
