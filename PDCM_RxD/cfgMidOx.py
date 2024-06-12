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
cfg.duration = 30e3  # Duration of the simulation, in ms
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
cfg.recordStim = False

### Options to save memory in large-scale ismulations
cfg.gatherOnlySimData = True  # Original

# set the following 3 options to False when running large-scale versions of the model (>50% scale) to save memory
cfg.saveCellSecs = False 
cfg.saveCellConns = False
cfg.createPyStruct = False
cfg.printPopAvgRates = True
cfg.printRunTime = 1
cfg.Kceil = 15.0
cfg.nRec = 240
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
cfg.o2drive = "0.13"

# cfg.scaleConnWeight = 1e-6
# cfg.scaleConnWeightNetStims = 1e-6

# slice conditions
cfg.ox = "perfused"
if cfg.ox == "perfused":
    cfg.o2_bath = 0.06
    cfg.o2_init = 0.04
    cfg.alpha_ecs = 0.2
    cfg.tort_ecs = 1.6
    cfg.o2drive = 0.13
elif cfg.ox == "hypoxic":
    cfg.o2_bath = 0.01
    cfg.o2_init = 0.01
    cfg.alpha_ecs = 0.07
    cfg.tort_ecs = 1.8
    cfg.o2drive = 0.13 * (1 / 6)

cfg.prep = "invivo"  # "invitro"

# neuron params
cfg.betaNrn = 0.24
cfg.Ncell = 12767
cfg.rs = ((cfg.betaNrn * cfg.Vtissue) / (2 * np.pi * cfg.Ncell)) ** (1 / 3)

cfg.epas = -70  # False
cfg.sa2v = 3.0  # False

# Neuron parameters
Wscale = 2.5
cfg.excWeight = 0.3e-5 * (Wscale)
cfg.inhWeightScale = 14
cfg.inhWeight = cfg.inhWeightScale * cfg.excWeight
cfg.gnabar = 30 / 1000
cfg.gkbar = 25 / 1000
cfg.ukcc2 = 0.3
cfg.unkcc1 = 0.1
cfg.pmax = 3
cfg.gpas = 0.0001

cfg.gkleak_scale = 1.0
cfg.gkbar = 0.012846651215426686
cfg.gnabar = 0.09995149718124381
cfg.gpas = 0.0009680849770246202
cfg.pmax = 3.8731513311322754
cfg.ukcc2 = 0.01417219231784303
cfg.unkcc1 = 0.9270812583375462
"""
cfg.excWeight = 0.008354547285548327
cfg.inhWeightScale = 5.799331499698029
cfg.gnabar = 0.09018714621136505
cfg.gkbar = 0.08616357060930685
cfg.ukcc2 = 0.14832347247652633
cfg.unkcc1 = 0.41410117511406297
cfg.pmax = 29.628601823899942
cfg.gpas = 0.0007944086444045182

"""
"""
cfg.excWeight = 0.00038502539808378517
cfg.inhWeightScale = 12.09670350162555
cfg.gnabar = 0.011017780584676423
cfg.gkbar = 0.027942726890048692 * 0.8
cfg.ukcc2 = 0.010009940428388016
cfg.unkcc1 = 0.7043763591503
cfg.pmax = 2.2896135259851844
cfg.gpas = 0.0002844471707125443
"""
"""
cfg.excWeight = 0.00010494211799199032
cfg.inhWeightScale = 11.849385401494027
cfg.gnabar = 0.013996841763437441
cfg.gkbar = 0.005713120530219498
cfg.ukcc2 = 0.9441935474841756
cfg.unkcc1 = 0.7007058562364242
cfg.pmax = 3.3864399808956533
cfg.gpas = 0.0005764994797376824
"""

"""
# from optuna
cfg.excWeight = 0.00023221135768927112 * 12
cfg.inhWeightScale = 12.288163937519913
cfg.gnabar = 0.09993943087539545
cfg.gkbar = 0.0030250462863877555
cfg.ukcc2 = 0.01488947351890433
cfg.unkcc1 = 0.6827438083487398
cfg.pmax = 1.3742040794796173
cfg.gpas = 0.0009710440265276351

cfg.excWeight = 0.0003607996757525843
cfg.inhWeightScale = 10.2416856976259
cfg.gnabar = 0.099746620475437
cfg.gkbar = 0.010924245043318587
cfg.ukcc2 = 0.011305434619351806
cfg.unkcc1 = 0.04786427363424489
cfg.pmax = 1.3234402225571111
cfg.gpas = 0.0009698978116799372

cfg.excWeight = 0.005468114518060687
cfg.inhWeightScale = 8.058499415702913
cfg.gnabar = 0.09794161502981583
cfg.gkbar = 0.004906727999254224
cfg.ukcc2 = 0.03220155161509672
cfg.unkcc1 = 0.35941732468829013
cfg.pmax = 0.7151845127361263
cfg.gpas = 0.00041536957698698305


cfg.gkleak_scale = 1
cfg.inhWeight = cfg.inhWeightScale * cfg.excWeight
"""
cfg.gkleak_scale = 1
cfg.KKo = 5.3
cfg.KNai = 27.9
# Scaled to match original 1/3 scaling at Ko=3; i.e.
# a = 3*(1 + np.exp(3.5-3))
# GliaKKo = np.log(a-1) + 3
cfg.GliaKKo = 4.938189537703508  # originally 3.5 mM
cfg.GliaPumpScale = 1  # originally 1/3
cfg.scaleConnWeight = 1

cfg.scaleConnWeightNetStims = 15e-6 / Wscale
cfg.scaleConnWeightNetStimsVar = 15e6**2 / Wscale**2

if cfg.sa2v:
    cfg.somaR = (cfg.sa2v * cfg.rs**3 / 2.0) ** (1 / 2)
else:
    cfg.somaR = cfg.rs
cfg.cyt_fraction = cfg.rs**3 / cfg.somaR**3

# sd init params
cfg.k0 = 250.0
cfg.r0 = 150.0
cfg.k0Layer = 4     # layer of elevated extracellular K+

###########################################################
# Network Options
###########################################################

# DC=True ;  TH=False; Balanced=True   => Reproduce Figure 7 A1 and A2
# DC=False;  TH=False; Balanced=False  => Reproduce Figure 7 B1 and B2
# DC=False ; TH=False; Balanced=True   => Reproduce Figure 8 A, B, C and D
# DC=False ; TH=False; Balanced=True   and run to 60 s to => Table 6
# DC=False ; TH=True;  Balanced=True   => Figure 10A. But I want a partial reproduce so I guess Figure 10C is not necessary

# Size of Network. Adjust this constants, please!
cfg.ScaleFactor = 0.16  # = 80.000

# External input DC or Poisson
cfg.DC = False  # True = DC // False = Poisson

# Thalamic input in 4th and 6th layer on or off
cfg.TH = True  # True = on // False = off

# Balanced and Unbalanced external input as PD article
cfg.Balanced = False  # False #True=Balanced // False=Unbalanced

cfg.ouabain = False

cfg.simLabel = f"SD_layer{cfg.k0Layer}_{cfg.scaleConnWeightNetStims}_{cfg.scaleConnWeightNetStimsVar}_GP{cfg.GliaKKo}_{cfg.excWeight}_{cfg.inhWeightScale}_K{cfg.k0}_scale{cfg.ScaleFactor}_{cfg.prep}_{cfg.ox}_pois{cfg.poissonRateFactor}_o2d{cfg.o2drive}_o2b_{cfg.o2_init}_Balanced{cfg.Balanced}_13kpmm_1mm3_dx{cfg.dx}_{cfg.oldDuration/1000:0.2f}s"
cfg.saveFolder = f"/ddn/adamjhn/data/{cfg.simLabel}/"
# cfg.saveFolder = f"/tera/adam/{cfg.simLabel}/" # for neurosim
cfg.restoredir = cfg.saveFolder if cfg.restore else None
# v0.0 - combination of cfg from ../uniformdensity and netpyne PD thalamocortical model
# v1.0 - cfg for o2 sources based on capillaries identified from histology
