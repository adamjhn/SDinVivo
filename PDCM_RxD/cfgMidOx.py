from netpyne import specs
import numpy as np
import cv2

# ------------------------------------------------------------------------------
#
# SIMULATION CONFIGURATION
#
# ------------------------------------------------------------------------------

# Run parameters
cfg = specs.SimConfig()  # object of class cfg to store simulation configuration
cfg.duration = 100  # Duration of the simulation, in ms
cfg.oldDuration = cfg.duration
cfg.restore = False
cfg.hParams["v_init"] = -70.0  # set v_init to -65 mV
cfg.hParams["celsius"] = 37.0
cfg.Cm = 1.0  # pF/cm**2
cfg.Ra = 100
cfg.dt = 0.025  # Internal integration timestep to use
cfg.verbose = False  # Show detailed messages
cfg.recordStep = 1  # Step size in ms to save data (eg. V traces, LFP, etc)
cfg.savePickle = True  # Save params, network and sim output to pickle file
cfg.saveJson = False
cfg.recordStim = True

### Options to save memory in large-scale ismulations
cfg.gatherOnlySimData = True  # Original

# set the following 3 options to False when running large-scale versions of the model (>50% scale) to save memory
cfg.saveCellSecs = False
cfg.saveCellConns = False
cfg.createPyStruct = False
cfg.printPopAvgRates = True
cfg.singleCells = False  # create one cell in each population
cfg.printRunTime = 1
cfg.Kceil = 15.0
cfg.nRec = 25
cfg.recordCellsSpikes = [
    "L2e",
    "L2i",
    "L4e",
    "L4i",
    "L5e",
    "L5i",
    "L6e",
    "L6i",
]  # record only spikes of cells (not ext stims)
cfg.recordCells = [
        (f"L{i}{ei}",{idx}) for i in [2, 4, 5, 6] for ei in ["e", "i"] for idx in range(10)
]
cfg.recordTraces = {
    f"{var}_soma": {"sec": "soma", "loc": 0.5, "var": var}
    for var in ["v", "nai", "ki", "cli", "dumpi"]
}
cfg.seeds = {"conn": 1, "stim": 1, "loc": 1, "cell": 1, "rec": 1}

# Network dimensions
cfg.fig_file = "../test_mask.tif"
img = cv2.imread(cfg.fig_file, cv2.IMREAD_GRAYSCALE)  # image used for capillaries
img = np.rot90(img, k=-1)
cfg.px = 0.2627  # side of image pixel (microns)
cfg.dx = 50  # side of ECS voxel (microns)
cfg.sizeX = 700  # img.shape[1] * cfg.px#250.0 #1000
cfg.sizeY = (img.shape[0] - 1000) * cfg.px  # 250.0 #1000
cfg.sizeZ = cfg.sizeX  # 200.0
cfg.Nz = int(cfg.sizeZ / cfg.dx) - 1
cfg.Vtissue = cfg.sizeX * cfg.sizeY * cfg.sizeZ

# scaling factors
cfg.poissonRateFactor = 1.0
cfg.connected = True

# cfg.scaleConnWeight = 1e-6
# cfg.scaleConnWeightNetStims = 1e-6

# slice conditions
cfg.ox = "hypoxic"
if cfg.ox == "perfused":
    cfg.o2_bath = 0.06  # ~36 mmHg
    cfg.o2_init = 0.04  # ~24 mmHg
    cfg.alpha_ecs = 0.2
    cfg.tort_ecs = 1.6
    cfg.o2drive = 0.013
elif cfg.ox == "hypoxic":
    cfg.o2_bath = 0.005  # ~4 mmHg
    cfg.o2_init = 0.005
    cfg.alpha_ecs = 0.07
    cfg.tort_ecs = 1.8
    cfg.o2drive = 0.013 * (1 / 6)
cfg.prep = "invivo"  # "invitro"
# Size of Network. Adjust this constants, please!
cfg.ScaleFactor = 0.16  # 0.02 used for batch param search  # = 80.000

# neuron params
cfg.betaNrn = (
    0.29  # 0.59 intracellular volume fraction (Rice & Russi-Menna 1997) ~80% neuronal
)
cfg.N_Full = [20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948, 902]
cfg.Ncell = sum([max(1, int(i * cfg.ScaleFactor)) for i in cfg.N_Full])
# Single cell parameter based on Scale 0.16
cfg.NcellRxD = sum([max(1, int(i * 0.16)) for i in cfg.N_Full])
cfg.rs = ((cfg.betaNrn * cfg.Vtissue) / (2 * np.pi * cfg.NcellRxD)) ** (1 / 3)

cfg.epas = -70.00000000000013 # False
cfg.sa2v = 3.4  # False

# Neuron parameters
#cfg.excWeight = 3.5472779489790976e-06
#cfg.inhWeightScale = 5.603334528668638
#cfg.scaleConnWeightNetStims = 0.513905416594529
#cfg.scaleConnWeightNetStimStd = 0.9355311387530357
cfg.excWeight = 4.95893e-6#7.270833e-7#2.4436e-6#1.11818e-6#2.35667e-7 #2.696e-6 #1.0974084295155178e-05
cfg.inhWeightScale = 5.06875#1.495457#8.80234#5.33374#1.2611 #9.75 #1.0050051831946134
cfg.scaleConnWeightNetStims = 0.34544#0.3164673#0.30577#0.34982#4.3904e-6 #0.01239 #5.260251723238228e-06
cfg.scaleConnWeightNetStimStd = 0.30401#0.8725078#0.181515#0.65593#5.39272e-6 #0.43407 #2.8118679638940484e-06

cfg.excWeight = 2.4436e-6#1.11818e-6#2.35667e-7 #2.696e-6 #1.0974084295155178e-05
cfg.inhWeightScale = 8.80234#5.33374#1.2611 #9.75 #1.0050051831946134
cfg.scaleConnWeightNetStims = 0.30577#0.34982#4.3904e-6 #0.01239 #5.260251723238228e-06
cfg.scaleConnWeightNetStimStd = 0.181515#0.65593#5.39272e-6 #0.43407 #2.8118679638940484e-06



"""
# original model
cfg.gnabar = 30 / 100
cfg.gkbar = 25 / 1000
cfg.ukcc2 = 0.3
cfg.unkcc1 = 0.1
cfg.pmax = 3
cfg.gpas = 0.0001
"""

cfg.Ggliamax = 5.0  # mM/sec originally 5mM/sec
# we scaled pump by ~4.84 so apply a corresponding
# reduction by channels (K, Kir and NKCC1) in glia.

# from optuna
cfg.gkbar = 0.027774094506433675
cfg.gnabar = 0.09996740846941617
cfg.gpas = 9.301731582270169e-07
cfg.pmax = 20.344379085226635
cfg.ukcc2 = 0.010894505022895047
cfg.unkcc1 = 0.08314528796758339
cfg.gkleak_scale = 1
cfg.KKo = 5.3
cfg.KNai = 27.9
# Scaled to match original 1/3 scaling at Ko=3; i.e.
# a = 3*(1 + np.exp(3.5-3))
# GliaKKo = np.log(a-1) + 3
cfg.GliaKKo = 4.938189537703508  # originally 3.5 mM
cfg.GliaPumpScale = 1 #1 / 3  # 1 / 3  # originally 1/3
cfg.scaleConnWeight = 1

if cfg.sa2v:
    cfg.somaR = (cfg.sa2v * cfg.rs**3 / 2.0) ** (1 / 2)
else:
    cfg.somaR = cfg.rs
cfg.cyt_fraction = cfg.rs**3 / cfg.somaR**3

# sd init params
cfg.k0 = 3.5 
cfg.r0 = 150.0
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
cfg.Balanced = False  # False #True=Balanced // False=Unbalanced

cfg.ouabain = False

cfg.simLabel = f"SDStim_layer{cfg.k0Layer}_{cfg.scaleConnWeightNetStims}_{cfg.scaleConnWeightNetStimStd}_GP{cfg.GliaKKo}_{cfg.excWeight}_{cfg.inhWeightScale}_K{cfg.k0}_scale{cfg.ScaleFactor}_{cfg.prep}_{cfg.ox}_pois{cfg.poissonRateFactor}_o2d{cfg.o2drive}_o2b_{cfg.o2_init}_Balanced{cfg.Balanced}_13kpmm_1mm3_dx{cfg.dx}_{cfg.oldDuration/1000:0.2f}s"
cfg.simLabel = "SDTest"
cfg.saveFolder = f"/ddn/adamjhn/data/{cfg.simLabel}/"
# cfg.simLabel = f"test_{cfg.ox}"
#cfg.saveFolder = f"/tmp/testSS2"
# cfg.saveFolder = f"/tera/adam/{cfg.simLabel}/" # for neurosim
cfg.restoredir = cfg.saveFolder if cfg.restore else None
# v0.0 - combination of cfg from ../uniformdensity and netpyne PD thalamocortical model
# v1.0 - cfg for o2 sources based on capillaries identified from histology
