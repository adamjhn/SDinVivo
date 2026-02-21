from netpyne import specs
import numpy as np
from neuron.units import sec, mM, M, s
import cv2

# ------------------------------------------------------------------------------
#
# SIMULATION CONFIGURATION
#
# ------------------------------------------------------------------------------

# Run parameters
cfg = specs.SimConfig()  # object of class cfg to store simulation configuration
cfg.duration = 1000  # Duration of the simulation, in ms
cfg.oldDuration = 1000
cfg.restore = False
cfg.hParams["celsius"] = 37.0
cfg.hParams["v_init"] = -70
cfg.Cm = 1.0  # pF/cm**2
cfg.Ra = 100
cfg.dt = 0.025  # Internal integration timestep to use
cfg.verbose = False  # Show detailed messages
cfg.recordStep = 1  # Step size in ms to save data (eg. V traces, LFP, etc)
cfg.savePickle = True  # Save params, network and sim output to pickle file
cfg.saveJson = False
cfg.recordStim = False
cfg.SDThreshold = -40  # Threshold for recording sustained deploarization.

### Options to save memory in large-scale ismulations
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
cfg.cellPopsInit = (-85, -60)
cfg.recordCellsSpikes = cfg.cellPops
if cfg.recordStim:
    cfg.recordCellsSpikes += [
        f"poissL{i}{ei}" for i in [2, 4, 5, 6] for ei in ["e", "i"]
    ]
    cfg.recordCellsSpikes += [f"bkg_THL{i}{ei}" for i in [4, 6] for ei in ["e", "i"]]

cfg.recordCells = [
    (f"L{i}{ei}", idx) for i in [2, 4, 5, 6] for ei in ["e", "i"] for idx in range(10)
]
cfg.recordTraces = {
    f"{var}_soma": {"sec": "soma", "loc": 0.5, "var": var}
    for var in ["v", "nai", "ki", "cli", "dumpi", "oxygeno", "ATPi", "ADPi", "AMPi"]
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


# slice conditions
cfg.ox = "perfused"
if cfg.ox == "perfused":
    cfg.o2_bath = 0.06  # ~36 mmHg
    cfg.o2_init = 0.04  # ~24 mmHg
    cfg.alpha_ecs = 0.2
    cfg.tort_ecs = 1.6
    cfg.o2drive = 50  # 0.013
elif cfg.ox == "hypoxic":
    cfg.o2_bath = 0.06  # ~4 mmHg
    cfg.o2_init = 0.005
    cfg.alpha_ecs = 0.07
    cfg.tort_ecs = 1.8
    cfg.o2drive = 1.0 / 6  # 0.013 * (1 / 6)
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


cfg.kleakMin = 1e-5 #mS/cm^2 -- this may changed pmax
# Neuron parameters
# Scale synapses weights -- optimized with min K-leak 1e-5
cfg.excWeight_L2e = 0.12861529088835885
cfg.excWeight_L4e = 0.18169356779579005
cfg.excWeight_L5e = 0.043806142795524264
cfg.excWeight_L6e = 0.27642308287182193
cfg.excWeight_L2i = 0.13146266535439496
cfg.excWeight_L4i = 0.06530070683093302
cfg.excWeight_L5i = 0.1820856969048531
cfg.excWeight_L6i = 0.22909354259474382
cfg.inhWeightScale_L2e = 10.888846469402536
cfg.inhWeightScale_L4e = 0.9710028368124768
cfg.inhWeightScale_L5e = 7.109779329177732
cfg.inhWeightScale_L6e = 1.2440275843551034
cfg.inhWeightScale_L2i = 2.2076546856300467
cfg.inhWeightScale_L4i = 5.941188668318516
cfg.inhWeightScale_L5i = 2.236815467906211
cfg.inhWeightScale_L6i = 4.20741098901948

cfg.pmax = 4708.757564140781
cfg.gnabar = 0.02187606398167006


cfg.gkbar = 0.004001629507118593
cfg.ukcc2 = 0.0019830617654271222
cfg.unkcc1 = 6.506198176269446
cfg.gpas = 4.2407540290597475e-05



# default values
cfg.weightMin = 0.1
cfg.dweight = 0.1
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
cfg.ATPss = 3.18 #mM PMC3524514 -- whole brain
cfg.ATPDc = 0.445 #um**2/ms
cfg.Ko2 = 0.3e-3 #mM  # Km for O2 at cytochrome c oxidase
cfg.KmADP_synthase = 0.025  # mM, from PMC3833997 (human skeletal muscle)
cfg.KmPi_synthase = 1.0     # mM, from PMC8434986 (cardiac tissue)
cfg.KiATP_synthase = 10.0   # mM, competitive inhibition constant for ATP (allows steady-state flux)
cfg.ADPss = 0.0944444444444444 # such that D2 (MgADP == 0.05 mM)
cfg.tauADP = 1
cfg.Pss = 4.2
cfg.tauP = 1
cfg.ATPase_basal_density = 0.05 # mM/ms

# Adenylate kinase equilibrium: 2*ADP <-> ATP + AMP
# At equilibrium: Keq = [ATP][AMP]/[ADP]^2 ≈ 1 (typical for adenylate kinase)
# Solving: AMP = Keq * ADP^2 / ATP = 1.0 * (0.05)^2 / 2.59 ≈ 0.001 mM
# Solving adenylateKinase rate_f == rate_b at steady-state gives exact value.
cfg.AMPss = 0.0692795435459248 # mM, from adenylate kinase equilibrium with ADPss and ATPss
cfg.Mg = 0.5 #mM (free Mg) https://doi.org/10.3390/ijms20143439

cfg.glia = {
    'nai'   : 18.0 * mM,
    'ki'    : 80.0 * mM,
    'ATP'   : 10 * mM,
    'ADP'   : 10/ 15.4 * mM,
    'Pos'   : cfg.Pss
} 
cfg.pH = 7.0
cfg.NaKPump = {
    "Tref"  : 310,
    "q10"   : 3.2,
    "Delta" : -0.031,
    "k1p"   : 1050/s,
    "k1m"   : 172.1/s/mM,
    "k2p"   : 481/s,
    "k2m"   : 40/s,
    "k3p"   : 2000/s,
    "k3m"   : 79.3e3/s/mM**2,
    "k4p"   : 320/s,
    "k4m"   : 40/s,
    "KATP"  : 2.51*mM,
    "KHPi"  : 6.77*mM,
    "KKPi"  : 292*mM,
    "KNaPi" : 224*mM,
    "PiT"   : 4.2*mM,
    "KKe"   : 0.213*mM,
    "KKi"   : 0.5*mM,
    "KNae0" : 15.5*mM,
    "KNai0"  : 2.49*mM,
}

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

#converstionFactor: μmol·min−1·mg−1 -> mM/ms
converstionFactor = 49*9.7e-7/16000 # mg of enzyme/m^3
converstionFactor *= 60e3 * 1e6 # μmol/min -> mol/ms
#    1g tissue = 9.7e-7 m^3 
#    49 units/g  (of tissue) brain
# 1,600 units/mg (of enzyme) muscle 
# unit 1 μmol/min
cfg.AK = {'KmAMP'   :   0.12, # mM
          'KiAMP'   :   3.3,  # mM
          'KmMgATP' :   0.06, # mM
          'KmADP'   :	0.028,# mM
          'KiADP'   :   0.91, # mM
          'KmMgADP' :	0.033,# mM
          'kp1'	    :   14_000 * converstionFactor, #mM/ms
          'km1'     :	8_000 * converstionFactor,  #mM/ms
          'kp2'     :	710  * converstionFactor,   #mM/ms
          'km2'     :	960 * converstionFactor,    #mM/ms
          'KMg'     :   2.5,    #/mM (stability constant)
}


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

cfg.ouabain = False

simLabel = f"SDPump{cfg.kleakMin}_{cfg.seed}_layer{cfg.k0Layer}_K0{cfg.k0}_{cfg.prep}_o2d{cfg.o2drive}_o2b_{cfg.o2_init}"
cfg.simLabel = f"{simLabel}_{cfg.duration/1000:0.2f}s"
cfg.saveFolder = f"./data/{simLabel}_{cfg.oldDuration/1000:0.2f}s"
# cfg.simLabel = f"test_{cfg.ox}"
# cfg.saveFolder = f"/tmp/test"
# cfg.saveFolder = f"/tera/adam/{cfg.simLabel}/" # for neurosim
cfg.restoredir = cfg.saveFolder if cfg.restore else None
# v0.0 - combination of cfg from ../uniformdensity and netpyne PD thalamocortical model
# v1.0 - cfg for o2 sources based on capillaries identified from histology
