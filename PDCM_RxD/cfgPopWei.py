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
cfg.hParams["celsius"] = 34.0
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
        "dumpi",
        "nao",
        "ko",
        "clo",
        "o2_extracellularo",
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


cfg.kleakMin = 5e-5  # mS/cm^2 -- this may changed pmax
# Neuron parameters
# Scale synapses weights -- optimized with min K-leak 1e-5
cfg.gnabar = {}
cfg.gkbar = {}
cfg.ukcc2 = {}
cfg.unkcc1 = {}
cfg.pmax = {}
cfg.gpas = {}
cfg.excWeight_L2e = 0.0008176488304411623
cfg.inhWeightScale_L2e = 2.4409601687877513
cfg.gnabar["L2e"] = 0.012030142308850161
cfg.gkbar["L2e"] = 0.002403123065918643
cfg.ukcc2["L2e"] = 0.5172281327260541
cfg.unkcc1["L2e"] = 3.302872894576172
cfg.pmax["L2e"] = 24.788203392884817
cfg.gpas["L2e"] = 7.542267762649311e-06

cfg.excWeight_L2i = 0.0007733104902872044
cfg.inhWeightScale_L2i = 3.2398539426917825
cfg.gnabar["L2i"] = 0.013721566764293322
cfg.gkbar["L2i"] = 0.010930264279244933
cfg.ukcc2["L2i"] = 0.46647388270751106
cfg.unkcc1["L2i"] = 3.8283166484663798
cfg.pmax["L2i"] = 40.96663962985401
cfg.gpas["L2i"] = 1.1657975162063223e-05

cfg.excWeight_L4e = 0.005070019552135673
cfg.inhWeightScale_L4e = 4.239532126424872
cfg.gnabar["L4e"] = 0.010071912435125238
cfg.gkbar["L4e"] = 0.009645007688103082
cfg.ukcc2["L4e"] = 0.47902739188899435
cfg.unkcc1["L4e"] = 2.501647216563663
cfg.pmax["L4e"] = 61.36228697597751
cfg.gpas["L4e"] = 1.4909608041310682e-05

cfg.excWeight_L4i = 0.005260502448351308
cfg.inhWeightScale_L4i = 6.089876471850459
cfg.gnabar["L4i"] = 0.011228444987260913
cfg.gkbar["L4i"] = 0.009669995411234412
cfg.ukcc2["L4i"] = 0.5859656269111796
cfg.unkcc1["L4i"] = 3.5516586068657636
cfg.pmax["L4i"] = 25.629472232361145
cfg.gpas["L4i"] = 1.3622485969940014e-06

cfg.excWeight_L5e = 0.0007665902281535269
cfg.inhWeightScale_L5e = 3.9268306017326418
cfg.gnabar["L5e"] = 0.014962164527472666
cfg.gkbar["L5e"] = 0.0026984780972302085
cfg.ukcc2["L5e"] = 0.6639463740623153
cfg.unkcc1["L5e"] = 1.9160564854553972
cfg.pmax["L5e"] = 44.60115222254619
cfg.gpas["L5e"] = 6.974086265690971e-06

cfg.excWeight_L5i = 0.004921691561447839
cfg.inhWeightScale_L5i = 5.005249233480747
cfg.gnabar["L5i"] = 0.02026336158166894
cfg.gkbar["L5i"] = 0.011800877019628052
cfg.ukcc2["L5i"] = 0.49949922643577954
cfg.unkcc1["L5i"] = 3.7755138397716124
cfg.pmax["L5i"] = 74.33445368244894
cfg.gpas["L5i"] = 1.131318020327028e-05

cfg.excWeight_L6e = 0.0012497882181424125
cfg.inhWeightScale_L6e = 2.1066517510883425
cfg.gnabar["L6e"] = 0.016845101148977394
cfg.gkbar["L6e"] = 0.005434804584030627
cfg.ukcc2["L6e"] = 0.5469398387983161
cfg.unkcc1["L6e"] = 2.1241877548337564
cfg.pmax["L6e"] = 57.69612110930161
cfg.gpas["L6e"] = 6.936474242330599e-06

cfg.excWeight_L6i = 0.00075172199408421
cfg.inhWeightScale_L6i = 3.5139420242597446
cfg.gnabar["L6i"] = 0.030007017386865495
cfg.gkbar["L6i"] = 0.010552699500435709
cfg.ukcc2["L6i"] = 0.45891872801994
cfg.unkcc1["L6i"] = 3.7239436010503018
cfg.pmax["L6i"] = 66.3765041652136
cfg.gpas["L6i"] = 1.108814684058013e-05

# =================B===========================#
cfg.excWeight_L2e = 0.0011149068127034485
cfg.inhWeightScale_L2e = 2.278771454175264
cfg.gnabar["L2e"] = 0.011157933164265818
cfg.gkbar["L2e"] = 0.008582535899834256
cfg.ukcc2["L2e"] = 0.4859840438562875
cfg.unkcc1["L2e"] = 3.2623261317823316
cfg.pmax["L2e"] = 32.290956380214595
cfg.gpas["L2e"] = 9.703914923178296e-06

cfg.excWeight_L2i = 0.0007584227680404089
cfg.inhWeightScale_L2i = 4.235221472990324
cfg.gnabar["L2i"] = 0.021048454003198446
cfg.gkbar["L2i"] = 0.012481695619713493
cfg.ukcc2["L2i"] = 0.46474754539093915
cfg.unkcc1["L2i"] = 4.832981170069923
cfg.pmax["L2i"] = 31.001323833979797
cfg.gpas["L2i"] = 1.4335370986585162e-05

cfg.excWeight_L4e = 0.005069972570242502
cfg.inhWeightScale_L4e = 4.248214887072051
cfg.gnabar["L4e"] = 0.010080450608819683
cfg.gkbar["L4e"] = 0.009422696717748445
cfg.ukcc2["L4e"] = 0.4835556071529514
cfg.unkcc1["L4e"] = 3.027565923374372
cfg.pmax["L4e"] = 55.16567598653684
cfg.gpas["L4e"] = 1.3827856968984594e-05

cfg.excWeight_L4i = 0.005265110963403909
cfg.inhWeightScale_L4i = 6.984191511398166
cfg.gnabar["L4i"] = 0.015379536503197403
cfg.gkbar["L4i"] = 0.009689316869172597
cfg.ukcc2["L4i"] = 0.5389210094294019
cfg.unkcc1["L4i"] = 3.5340997952701088
cfg.pmax["L4i"] = 81.27675172991593
cfg.gpas["L4i"] = 7.164024733577939e-06

cfg.excWeight_L5e = 0.00075116421774201
cfg.inhWeightScale_L5e = 2.543139911058244
cfg.gnabar["L5e"] = 0.011982948315494379
cfg.gkbar["L5e"] = 0.012750313737656931
cfg.ukcc2["L5e"] = 0.6620718737464736
cfg.unkcc1["L5e"] = 4.76116372229989
cfg.pmax["L5e"] = 71.33680336382845
cfg.gpas["L5e"] = 1.0358637933649888e-06

cfg.excWeight_L5i = 0.004914749535592288
cfg.inhWeightScale_L5i = 3.9471452738687702
cfg.gnabar["L5i"] = 0.011794548764776551
cfg.gkbar["L5i"] = 0.011751702214679992
cfg.ukcc2["L5i"] = 0.5153229211896472
cfg.unkcc1["L5i"] = 1.9904250580584375
cfg.pmax["L5i"] = 55.05169761305394
cfg.gpas["L5i"] = 1.0975035884015666e-05

cfg.excWeight_L6e = 0.0012933525510202331
cfg.inhWeightScale_L6e = 2.621912742442865
cfg.gnabar["L6e"] = 0.024971748986166862
cfg.gkbar["L6e"] = 0.006066356665419904
cfg.ukcc2["L6e"] = 0.5364134826927495
cfg.unkcc1["L6e"] = 1.6130903419200613
cfg.pmax["L6e"] = 54.959766298021016
cfg.gpas["L6e"] = 7.274513212710913e-06

cfg.excWeight_L6i = 0.0007537998592634884
cfg.inhWeightScale_L6i = 3.9148641389482406
cfg.gnabar["L6i"] = 0.03315439982040316
cfg.gkbar["L6i"] = 0.008922200794604498
cfg.ukcc2["L6i"] = 0.4737471879400182
cfg.unkcc1["L6i"] = 2.7222695687678797
cfg.pmax["L6i"] = 77.49362731020607
cfg.gpas["L6i"] = 6.614589092308221e-06


# ==============C======================#
"""cfg.excWeight_L2e = 0.0008176488304411623
cfg.inhWeightScale_L2e = 2.4409601687877513
cfg.gnabar['L2e'] = 0.012030142308850161
cfg.gkbar['L2e'] = 0.002403123065918643
cfg.ukcc2['L2e'] = 0.5172281327260541
cfg.unkcc1['L2e'] = 3.302872894576172
cfg.pmax['L2e'] = 24.788203392884817
cfg.gpas['L2e'] = 7.542267762649311e-06

cfg.excWeight_L2i = 0.0007733104902872044
cfg.inhWeightScale_L2i = 3.2398539426917825
cfg.gnabar['L2i'] = 0.013721566764293322
cfg.gkbar['L2i'] = 0.010930264279244933
cfg.ukcc2['L2i'] = 0.46647388270751106
cfg.unkcc1['L2i'] = 3.8283166484663798
cfg.pmax['L2i'] = 40.96663962985401
cfg.gpas['L2i'] = 1.1657975162063223e-05

cfg.excWeight_L4e = 0.005070019552135673
cfg.inhWeightScale_L4e = 4.239532126424872
cfg.gnabar['L4e'] = 0.010071912435125238
cfg.gkbar['L4e'] = 0.009645007688103082
cfg.ukcc2['L4e'] = 0.47902739188899435
cfg.unkcc1['L4e'] = 2.501647216563663
cfg.pmax['L4e'] = 61.36228697597751
cfg.gpas['L4e'] = 1.4909608041310682e-05

cfg.excWeight_L4i = 0.005260502448351308
cfg.inhWeightScale_L4i = 6.089876471850459
cfg.gnabar['L4i'] = 0.011228444987260913
cfg.gkbar['L4i'] = 0.009669995411234412
cfg.ukcc2['L4i'] = 0.5859656269111796
cfg.unkcc1['L4i'] = 3.5516586068657636
cfg.pmax['L4i'] = 25.629472232361145
cfg.gpas['L4i'] = 1.3622485969940014e-06

cfg.excWeight_L5e = 0.0007665902281535269
cfg.inhWeightScale_L5e = 3.9268306017326418
cfg.gnabar['L5e'] = 0.014962164527472666
cfg.gkbar['L5e'] = 0.0026984780972302085
cfg.ukcc2['L5e'] = 0.6639463740623153
cfg.unkcc1['L5e'] = 1.9160564854553972
cfg.pmax['L5e'] = 44.60115222254619
cfg.gpas['L5e'] = 6.974086265690971e-06

cfg.excWeight_L5i = 0.004921691561447839
cfg.inhWeightScale_L5i = 5.005249233480747
cfg.gnabar['L5i'] = 0.02026336158166894
cfg.gkbar['L5i'] = 0.011800877019628052
cfg.ukcc2['L5i'] = 0.49949922643577954
cfg.unkcc1['L5i'] = 3.7755138397716124
cfg.pmax['L5i'] = 74.33445368244894
cfg.gpas['L5i'] = 1.131318020327028e-05

cfg.excWeight_L6e = 0.0012497882181424125
cfg.inhWeightScale_L6e = 2.1066517510883425
cfg.gnabar['L6e'] = 0.016845101148977394
cfg.gkbar['L6e'] = 0.005434804584030627
cfg.ukcc2['L6e'] = 0.5469398387983161
cfg.unkcc1['L6e'] = 2.1241877548337564
cfg.pmax['L6e'] = 57.69612110930161
cfg.gpas['L6e'] = 6.936474242330599e-06

cfg.excWeight_L6i = 0.00075172199408421
cfg.inhWeightScale_L6i = 3.5139420242597446
cfg.gnabar['L6i'] = 0.030007017386865495
cfg.gkbar['L6i'] = 0.010552699500435709
cfg.ukcc2['L6i'] = 0.45891872801994
cfg.unkcc1['L6i'] = 3.7239436010503018
cfg.pmax['L6i'] = 66.3765041652136
cfg.gpas['L6i'] = 1.108814684058013e-05

#========================D=======================#
cfg.excWeight_L2e = 0.003503599233340524
cfg.inhWeightScale_L2e = 4.267511038318158
cfg.gnabar['L2e'] = 0.016905794472757644
cfg.gkbar['L2e'] = 0.012756385052572091
cfg.ukcc2['L2e'] = 0.6762223271790135
cfg.unkcc1['L2e'] = 2.132818964187024
cfg.pmax['L2e'] = 35.57438813656186
cfg.gpas['L2e'] = 8.421774695768611e-06

cfg.excWeight_L2i = 0.0009368983905150062
cfg.inhWeightScale_L2i = 4.8516189570971076
cfg.gnabar['L2i'] = 0.024167540395884732
cfg.gkbar['L2i'] = 0.008952957242304539
cfg.ukcc2['L2i'] = 0.5725116019583326
cfg.unkcc1['L2i'] = 4.006703155801908
cfg.pmax['L2i'] = 33.984158862533135
cfg.gpas['L2i'] = 1.3426449341073681e-05

cfg.excWeight_L4e = 0.005115707464571911
cfg.inhWeightScale_L4e = 5.773055617117094
cfg.gnabar['L4e'] = 0.01813581125887672
cfg.gkbar['L4e'] = 0.007153214370597945
cfg.ukcc2['L4e'] = 0.5058906078265724
cfg.unkcc1['L4e'] = 3.7654994289159847
cfg.pmax['L4e'] = 40.69156024222763
cfg.gpas['L4e'] = 1.2929122600092631e-05

cfg.excWeight_L4i = 0.005275491423620891
cfg.inhWeightScale_L4i = 6.930252246595792
cfg.gnabar['L4i'] = 0.018445361216682044
cfg.gkbar['L4i'] = 0.01245484067785961
cfg.ukcc2['L4i'] = 1.3027959800281925
cfg.unkcc1['L4i'] = 5.061107345066842
cfg.pmax['L4i'] = 45.93046647962599
cfg.gpas['L4i'] = 1.098331519980131e-05

cfg.excWeight_L5e = 0.0007509305007332031
cfg.inhWeightScale_L5e = 3.5796547226402744
cfg.gnabar['L5e'] = 0.01594424264177733
cfg.gkbar['L5e'] = 0.011012825823132398
cfg.ukcc2['L5e'] = 0.6811228605895183
cfg.unkcc1['L5e'] = 1.7318675305103413
cfg.pmax['L5e'] = 39.89777775871499
cfg.gpas['L5e'] = 1.0039742319063617e-06

cfg.excWeight_L5i = 0.004922266403307469
cfg.inhWeightScale_L5i = 4.2080375380359385
cfg.gnabar['L5i'] = 0.018437472453325054
cfg.gkbar['L5i'] = 0.005234138618027619
cfg.ukcc2['L5i'] = 1.1502155070386932
cfg.unkcc1['L5i'] = 5.531607508152945
cfg.pmax['L5i'] = 20.24523606420374
cfg.gpas['L5i'] = 9.75286730450836e-06

cfg.excWeight_L6e = 0.002986744304980348
cfg.inhWeightScale_L6e = 2.316726364607777
cfg.gnabar['L6e'] = 0.014979097109244707
cfg.gkbar['L6e'] = 0.007824015891585046
cfg.ukcc2['L6e'] = 0.6268880977124006
cfg.unkcc1['L6e'] = 2.1038715652929096
cfg.pmax['L6e'] = 37.62377978058194
cfg.gpas['L6e'] = 1.3735899729516566e-05

cfg.excWeight_L6i = 0.0008335076906942475
cfg.inhWeightScale_L6i = 3.0890617545303622
cfg.gnabar['L6i'] = 0.030405323696888104
cfg.gkbar['L6i'] = 0.008936565656269314
cfg.ukcc2['L6i'] = 0.46924689517479656
cfg.unkcc1['L6i'] = 2.8480884468475094
cfg.pmax['L6i'] = 81.10878028498482
cfg.gpas['L6i'] = 1.2871288337941506e-05

#====================E===================#
cfg.excWeight_L2e = 0.0030252772724701175
cfg.inhWeightScale_L2e = 3.9052135921101203
cfg.gnabar['L2e'] = 0.01641837406348288
cfg.gkbar['L2e'] = 0.01173349110192753
cfg.ukcc2['L2e'] = 0.7985487415642804
cfg.unkcc1['L2e'] = 3.183623111685377
cfg.pmax['L2e'] = 23.81938431097514
cfg.gpas['L2e'] = 9.765400313659646e-06

cfg.excWeight_L2i = 0.0008491559517365399
cfg.inhWeightScale_L2i = 4.614983689357048
cfg.gnabar['L2i'] = 0.021146377901562107
cfg.gkbar['L2i'] = 0.011793028435632142
cfg.ukcc2['L2i'] = 0.4583212862548262
cfg.unkcc1['L2i'] = 3.287320207996732
cfg.pmax['L2i'] = 78.40033778251731
cfg.gpas['L2i'] = 1.077174188536356e-05

cfg.excWeight_L4e = 0.005091860289099377
cfg.inhWeightScale_L4e = 5.130333347378982
cfg.gnabar['L4e'] = 0.01505837169576863
cfg.gkbar['L4e'] = 0.007071170052096471
cfg.ukcc2['L4e'] = 0.4677968264557894
cfg.unkcc1['L4e'] = 3.7080536222235354
cfg.pmax['L4e'] = 41.102706109628684
cfg.gpas['L4e'] = 1.446538392033642e-05

cfg.excWeight_L4i = 0.0052643820506500264
cfg.inhWeightScale_L4i = 7.119029926316359
cfg.gnabar['L4i'] = 0.018531537469717013
cfg.gkbar['L4i'] = 0.01230939086491735
cfg.ukcc2['L4i'] = 0.747360066548885
cfg.unkcc1['L4i'] = 4.65886616443627
cfg.pmax['L4i'] = 43.66466400407004
cfg.gpas['L4i'] = 1.5006431633134091e-05

cfg.excWeight_L5e = 0.0007512201733802658
cfg.inhWeightScale_L5e = 3.7366543404210173
cfg.gnabar['L5e'] = 0.01634036407241359
cfg.gkbar['L5e'] = 0.010671081355137852
cfg.ukcc2['L5e'] = 0.6742481606980759
cfg.unkcc1['L5e'] = 2.267614440804938
cfg.pmax['L5e'] = 54.68718888118849
cfg.gpas['L5e'] = 1.8251398993404197e-06

cfg.excWeight_L5i = 0.005143139843435173
cfg.inhWeightScale_L5i = 4.272303562898888
cfg.gnabar['L5i'] = 0.013620614585159573
cfg.gkbar['L5i'] = 0.010273263072860526
cfg.ukcc2['L5i'] = 0.5505556576118583
cfg.unkcc1['L5i'] = 2.4046690307768332
cfg.pmax['L5i'] = 68.85216681768311
cfg.gpas['L5i'] = 7.897103821953744e-06

cfg.excWeight_L6e = 0.0024029625312369414
cfg.inhWeightScale_L6e = 1.9377174938204877
cfg.gnabar['L6e'] = 0.01425338233945488
cfg.gkbar['L6e'] = 0.007497006922015551
cfg.ukcc2['L6e'] = 0.694484950566414
cfg.unkcc1['L6e'] = 2.0240506012393786
cfg.pmax['L6e'] = 44.71268377144419
cfg.gpas['L6e'] = 1.3711860688432056e-05

cfg.excWeight_L6i = 0.0008137744248161596
cfg.inhWeightScale_L6i = 3.3481385696285066
cfg.gnabar['L6i'] = 0.031352789022209546
cfg.gkbar['L6i'] = 0.0032384573296706715
cfg.ukcc2['L6i'] = 0.466529426293488
cfg.unkcc1['L6i'] = 4.636572450122609
cfg.pmax['L6i'] = 78.54634270172664
cfg.gpas['L6i'] = 1.2636678028271004e-05

#==================F======================#
cfg.excWeight_L2e = 0.0031354213601586917
cfg.inhWeightScale_L2e = 4.049382675204058
cfg.gnabar['L2e'] = 0.018957827748214362
cfg.gkbar['L2e'] = 0.012537830952855992
cfg.ukcc2['L2e'] = 0.9224850769628294
cfg.unkcc1['L2e'] = 2.554631969286257
cfg.pmax['L2e'] = 48.65191657291509
cfg.gpas['L2e'] = 8.808291345693465e-06

cfg.excWeight_L2i = 0.0010763234816039806
cfg.inhWeightScale_L2i = 4.996533819929763
cfg.gnabar['L2i'] = 0.026159881539099324
cfg.gkbar['L2i'] = 0.009485030544289726
cfg.ukcc2['L2i'] = 0.6382484260191936
cfg.unkcc1['L2i'] = 3.6957921827339404
cfg.pmax['L2i'] = 58.438988543374556
cfg.gpas['L2i'] = 1.4155927121043265e-05

cfg.excWeight_L4e = 0.005071202733296597
cfg.inhWeightScale_L4e = 4.822084206163648
cfg.gnabar['L4e'] = 0.0128942808546363
cfg.gkbar['L4e'] = 0.010243114942950068
cfg.ukcc2['L4e'] = 0.4768709419221407
cfg.unkcc1['L4e'] = 3.193322163275185
cfg.pmax['L4e'] = 73.19276951212831
cfg.gpas['L4e'] = 1.0473815436677694e-05

cfg.excWeight_L4i = 0.005389558340889789
cfg.inhWeightScale_L4i = 7.889035988048459
cfg.gnabar['L4i'] = 0.02309935131485438
cfg.gkbar['L4i'] = 0.007303969158808115
cfg.ukcc2['L4i'] = 0.8876350708198739
cfg.unkcc1['L4i'] = 3.467701095580546
cfg.pmax['L4i'] = 54.84478100994974
cfg.gpas['L4i'] = 2.905315364129982e-06

cfg.excWeight_L5e = 0.001267907227087012
cfg.inhWeightScale_L5e = 5.447583509540081
cfg.gnabar['L5e'] = 0.02230832051183163
cfg.gkbar['L5e'] = 0.0023257232118147576
cfg.ukcc2['L5e'] = 0.6629021084012938
cfg.unkcc1['L5e'] = 5.191811285568908
cfg.pmax['L5e'] = 73.12662940025828
cfg.gpas['L5e'] = 6.620411658070666e-06

cfg.excWeight_L5i = 0.0052105792045255075
cfg.inhWeightScale_L5i = 3.692629872092166
cfg.gnabar['L5i'] = 0.013283295003161997
cfg.gkbar['L5i'] = 0.011309730573153647
cfg.ukcc2['L5i'] = 0.9338759496488591
cfg.unkcc1['L5i'] = 5.464791626825611
cfg.pmax['L5i'] = 45.13538983191219
cfg.gpas['L5i'] = 9.008516278605913e-06

cfg.excWeight_L6e = 0.0027132828692119576
cfg.inhWeightScale_L6e = 2.5020945169627513
cfg.gnabar['L6e'] = 0.016816041270802697
cfg.gkbar['L6e'] = 0.0066924456689543286
cfg.ukcc2['L6e'] = 0.5526588819277575
cfg.unkcc1['L6e'] = 1.4634330632925958
cfg.pmax['L6e'] = 82.44531119306204
cfg.gpas['L6e'] = 1.2732865766471846e-05


cfg.excWeight_L6i = 0.0007930936148229042
cfg.inhWeightScale_L6i = 3.6555153344851194
cfg.gnabar['L6i'] = 0.028449062123818526
cfg.gkbar['L6i'] = 0.011494554843845452
cfg.ukcc2['L6i'] = 0.4664669459036797
cfg.unkcc1['L6i'] = 3.576462726771986
cfg.pmax['L6i'] = 32.62660897636149
cfg.gpas['L6i'] = 2.6738785703332244e-06
"""
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

simLabel = f"SDWeiB_34_v_balance{cfg.v_balance}_ramp{cfg.poisson_ramp_ms}_{cfg.seed}_layer{cfg.k0Layer}_K0{cfg.k0}_{cfg.prep}_o2d{cfg.o2drive}_o2b_{cfg.o2_init}"
cfg.simLabel = f"{simLabel}_{cfg.duration/1000:0.2f}s"
cfg.saveFolder = f"./data/{simLabel}_{cfg.oldDuration/1000:0.2f}s"
# cfg.simLabel = f"test_{cfg.ox}"
# cfg.saveFolder = f"/tmp/test"
# cfg.saveFolder = f"/tera/adam/{cfg.simLabel}/" # for neurosim
cfg.restoredir = cfg.saveFolder if cfg.restore else None
# v0.0 - combination of cfg from ../uniformdensity and netpyne PD thalamocortical model
# v1.0 - cfg for o2 sources based on capillaries identified from histology
