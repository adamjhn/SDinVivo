from netpyne import specs
import numpy as np
from cfgMidOx import cfg
from neuron.units import sec, mM
import math
import cv2


############################################################
#
#                    NETWORK PARAMETERS
#
############################################################
def findCapillaries(img):
    th, threshed = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    centers = []
    for i in contours:
        if cv2.contourArea(i) > 1000:
            M = cv2.moments(i)
            try:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append([cx, cy])
            except:
                passes = passes + 1
    return centers


def takeStep(pos, xmax, ymax, dz=5, px=0.2627):
    samp = np.random.rand()
    if samp < 0.44:
        newpos = [pos[0], pos[1]]
    elif samp < 0.51:
        newpos = [pos[0], pos[1] + int(dz * px)]
    elif samp < 0.58:
        newpos = [pos[0], pos[1] - int(dz * px)]
    elif samp < 0.65:
        newpos = [pos[0] + int(dz * px), pos[1]]
    elif samp < 0.72:
        newpos = [pos[0] - int(dz * px), pos[1]]
    elif samp < 0.79:
        newpos = [pos[0] + int(dz * px), pos[1] + int(dz * px)]
    elif samp < 0.86:
        newpos = [pos[0] - int(dz * px), pos[1] - int(dz * px)]
    elif samp < 0.93:
        newpos = [pos[0] + int(dz * px), pos[1] - int(dz * px)]
    else:
        newpos = [pos[0] - int(dz * px), pos[1] + int(dz * px)]
    if (0 < newpos[0] < xmax) and (0 < newpos[1] < ymax):
        return newpos
    else:
        return pos  # takeStep(pos, xmax, ymax, dz=dz, px=px)


def extrudeCapillaries(positions, Nz, xmax, ymax, dz=5, px=0.2627):
    caps = []
    for cap in positions:
        zpos = [cap]
        for i in range(Nz):
            zpos.append(takeStep(zpos[-1], xmax, ymax))
        caps.append(zpos)
    return caps


def mask3D(capillaries, xsz, ysz, px, dx):
    mask = np.zeros(
        (round(ysz * px / dx), round(xsz * px / dx), len(capillaries[0])),
        dtype=np.int16,
    )
    for cap in capillaries:
        for z in range(len(cap)):
            mask[round(cap[z][0] * px / dx) - 1, round(cap[z][1] * px / dx) - 1, z] = (
                mask[round(cap[z][0] * px / dx) - 1, round(cap[z][1] * px / dx) - 1, z]
                + 1
            )
    return mask


def generateO2sources(fig_file, Nz, px, dx, x=None, y=None, z=None):
    img = cv2.imread(fig_file, cv2.IMREAD_GRAYSCALE)
    img = np.rot90(img, k=-1)
    img = img[1000:, : round(cfg.sizeX / px)]
    centers = findCapillaries(img)
    capillaries = extrudeCapillaries(
        centers, int(img.shape[0] * px / dx) - 1, img.shape[0], img.shape[1]
    )
    o2sources = mask3D(capillaries, img.shape[0], img.shape[1], px, dx)
    return o2sources


# Reescaling function (move to separate module?)
def Reescale(ScaleFactor, C, N_Full, w_p, f_ext, tau_syn, Inp, InpDC):
    if ScaleFactor < 1.0:
        # This is a good approximation of the F_out param for the Balanced option "True".
        # Note for the Balanced=False option, it should be possible to calculate a better approximation.
        F_out = np.array([0.860, 2.600, 4.306, 5.396, 8.142, 8.188, 0.941, 7.3])

        Ncon = np.vstack(
            [np.column_stack([0 for i in range(0, 8)]) for i in range(0, 8)]
        )
        for r in range(0, 8):
            for c in range(0, 8):
                Ncon[r][c] = (
                    np.log(1.0 - C[r][c]) / np.log(1.0 - 1.0 / (N_Full[r] * N_Full[c]))
                ) / N_Full[r]

        w = w_p * np.column_stack(
            [np.vstack([[1.0, -4.0] for i in range(0, 8)]) for i in range(0, 4)]
        )
        w[0][2] = 2.0 * w[0][2]

        x1_all = w * Ncon * F_out
        x1_sum = np.sum(x1_all, axis=1)
        x1_ext = w_p * Inp * f_ext
        I_ext = np.column_stack([0 for i in range(0, 8)])
        I_ext = (
            0.001
            * tau_syn
            * (
                (1.0 - np.sqrt(ScaleFactor)) * x1_sum
                + (1.0 - np.sqrt(ScaleFactor)) * x1_ext
            )
        )

        InpDC = np.sqrt(ScaleFactor) * InpDC * w_p * f_ext * tau_syn * 0.001  # pA
        w_p = w_p / np.sqrt(ScaleFactor)  # pA
        InpDC = InpDC + I_ext
        N_ = [int(ScaleFactor * N) for N in N_Full]
    else:
        InpDC = InpDC * w_p * f_ext * tau_syn * 0.001
        N_ = N_Full

    return InpDC, N_, w_p


###########################################################
#  Network Constants
###########################################################

# Frequency of external input
f_ext = 8.0  # (Hz)
# Postsynaptic current time constant
tau_syn = 0.5  # (ms)
# Membrane time constant
tau_m = 10  # (ms)
# Other constants defined inside .mod
"""
#Absolute refractory period
tauref =2 (ms) 
#Reset potential
Vreset = -65 (mV) : -49 (mV) :
#Fixed firing threshold
Vteta  = -50 (mV)"""
# Membrane capacity
C_m = cfg.Cm / (2e-8 * np.pi * cfg.somaR**2)  # pF
# Mean amplitude of the postsynaptic potential (in mV).
w_v = 0.15
# Mean amplitude of the postsynaptic potential (in pA).
w_p = (
    (C_m) ** (-1)
    * tau_m
    * tau_syn
    / (tau_syn - tau_m)
    * (
        (tau_m / tau_syn) ** (-tau_m / (tau_m - tau_syn))
        - (tau_m / tau_syn) ** (-tau_syn / (tau_m - tau_syn))
    )
) ** (-1)
w_p = w_v * w_p  # (pA)

# C probability of connection
C = np.array(
    [
        [0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0.0, 0.0076, 0.0],
        [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.0, 0.0042, 0.0],
        [0.0077, 0.0059, 0.0497, 0.135, 0.0067, 0.0003, 0.0453, 0.0],
        [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.0, 0.1057, 0.0],
        [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.0],
        [0.0548, 0.0269, 0.0257, 0.0022, 0.06, 0.3158, 0.0086, 0.0],
        [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
        [0.0364, 0.001, 0.0034, 0.0005, 0.0277, 0.008, 0.0658, 0.1443],
    ]
)

# Population size N
L = ["L2e", "L2i", "L4e", "L4i", "L5e", "L5i", "L6e", "L6i"]
N_Full = np.array([20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948, 902])

# Number of Input per Layer
Inp = np.array([1600, 1500, 2100, 1900, 2000, 1900, 2900, 2100])
if cfg.Balanced == False:
    InpUnb = np.array([2000, 1850, 2000, 1850, 2000, 1850, 2000, 1850])



###########################################################
# Reescaling calculation
###########################################################
if cfg.DC == True:
    InpDC = Inp
    if cfg.Balanced == False:
        InpDC = InpUnb
else:
    InpDC = np.zeros(8)
    InpPoiss = Inp * cfg.ScaleFactor
    if cfg.Balanced == False:
        InpPoiss = InpUnb * cfg.ScaleFactor

InpDC, N_, w_p = Reescale(cfg.ScaleFactor, C, N_Full, w_p, f_ext, tau_syn, Inp, InpDC)

############################################################
# NetPyNE Network Parameters (netParams)
############################################################

netParams = (
    specs.NetParams()
)  # object of class NetParams to store the network parameters

netParams.delayMin_e = 1.5
netParams.ddelay = 0.5
netParams.delayMin_i = 0.75
netParams.weightMin = w_p
netParams.dweight = 0.1

netParams.sizeX = (
    cfg.sizeX
)  # - 2*cfg.somaR # x-dimension (horizontal length) size in um
netParams.sizeY = (
    cfg.sizeY
)  # - 2*cfg.somaR # y-dimension (vertical height or cortical depth) size in um
netParams.sizeZ = (
    cfg.sizeZ
)  # - 2*cfg.somaR # z-dimension (horizontal length) size in um

popDepths = [
    [0.08, 0.27],
    [0.08, 0.27],
    [0.27, 0.58],
    [0.27, 0.58],
    [0.58, 0.73],
    [0.58, 0.73],
    [0.73, 1.0],
    [0.73, 1.0],
]

#Scale weights
netParams.scaleConnWeight = cfg.scaleConnWeight
#Does not do anything directly -- insteat apply to weights 
#netParams.scaleConnWeightNetStims = cfg.scaleConnWeightNetStims

# ------------------------------------------------------------------------------
# create populations
for i in range(0, 8):
    if (i == 0) or (i == 1):
        netParams.popParams[L[i]] = {
            "cellType": str(L[i]),
            "numCells": int(N_[i]),
            "cellModel": L[i],
            "xRange": [0.0, cfg.sizeX],
            "yRange": [popDepths[i][0] * cfg.sizeY, cfg.sizeY * popDepths[i][1]],
            "zRange": [0.0, cfg.sizeZ],
        }
    elif (i == 6) or (i == 7):
        netParams.popParams[L[i]] = {
            "cellType": str(L[i]),
            "numCells": int(N_[i]),
            "cellModel": L[i],
            "xRange": [0.0, cfg.sizeX],
            "yRange": [
                popDepths[i][0] * cfg.sizeY,
                cfg.sizeY * popDepths[i][1] - 2 * cfg.somaR,
            ],
            "zRange": [0.0, cfg.sizeZ],
        }
    else:
        netParams.popParams[L[i]] = {
            "cellType": str(L[i]),
            "numCells": int(N_[i]),
            "cellModel": L[i],
            "xRange": [0.0, cfg.sizeX],
            "yRange": [popDepths[i][0] * cfg.sizeY, cfg.sizeY * popDepths[i][1]],
            "zRange": [0.0, cfg.sizeZ],
        }

# To atualization of Point Neurons
# netParams.popParams['bkg_IF'] = {'numCells': 1, 'cellModel': 'NetStim','rate': 40000,  'start':0.0, 'noise': 0.0, 'delay':0}

# cell property rules
for pop in L:
    cellRule = netParams.importCellParams(
        label="cellRule",
        fileName="Neuron.py",
        conds={"cellType": pop, "cellModel": pop},
        cellName=pop,
    )
    netParams.cellParams[pop + "Rule"] = cellRule

netParams.synMechParams["exc"] = {
    "mod": "Exp2Syn",
    "tau1": 0.8,
    "tau2": 5.3,
    "e": 0,
}  # NMDA synaptic mechanism
netParams.synMechParams["inh"] = {
    "mod": "Exp2Syn",
    "tau1": 0.6,
    "tau2": 8.5,
    "e": -75,
}  # GABA synaptic mechanism

# added bkg inputs
# netParams.stimSourceParams['bkg'] = {'type': 'NetStim', 'rate': 5, 'noise': 0.3, 'start' : 100}
# # netParams.stimTargetParams['bkg->L4e'] = {'source': 'bkg', 'conds': {'pop': ['L4e'], 'cellList' : [i for i in range(100)]}, 'weight': 0.05, 'delay': 1, 'synMech': 'exc'}
# # netParams.stimTargetParams['bkg->L4i'] = {'source': 'bkg', 'conds': {'pop': ['L4i'], 'cellList' : [i for i in range(50)]}, 'weight': 0.05, 'delay': 1, 'synMech': 'exc'}
# netParams.stimTargetParams['bkg->L4e'] = {'source': 'bkg', 'conds': {'pop': ['L4e']}, 'weight': 0.007, 'delay': 1, 'synMech': 'exc'}
# netParams.stimTargetParams['bkg->L4i'] = {'source': 'bkg', 'conds': {'pop': ['L4i']}, 'weight': 0.007, 'delay': 1, 'synMech': 'exc'}

if cfg.DC == False:  # External Input as Poisson
    for r in range(0, 8):
        netParams.popParams["poiss" + str(L[r])] = {
            "numCells": N_[r],
            "cellModel": "NetStim",
            "rate": InpPoiss[r] * f_ext * cfg.poissonRateFactor,
            "start": 0.0,
            "noise": 1.0,
            "delay": 0,
        }

        auxConn = np.array([range(0, N_[r], 1), range(0, N_[r], 1)])
        netParams.connParams["poiss->" + str(L[r])] = {
            "preConds": {"pop": "poiss" + str(L[r])},
            "postConds": {"pop": L[r]},
            "connList": auxConn.T,
            "weight": f"max(0, {cfg.excWeight*cfg.scaleConnWeightNetStims} * (weightMin+normal(0,dweight*weightMin)))",
            "delay": 0.5,
            "synMech": "exc",
        }  # 1 delay

# Thalamus Input: increased of 15Hz that lasts 10 ms
# 0.15 fires in 10 ms each 902 cells -> number of spikes = T*f*N_ = 0.15*902 -> 1 spike each N_*0.15
if cfg.TH == True:
    fth = 15  # Hz
    Tth = 10  # ms
    InTH = [0, 0, 93, 84, 0, 0, 47, 34]
    for r in [2, 3, 6, 7]:
        nTH = int(np.sqrt(cfg.ScaleFactor) * InTH[r] * fth * Tth / 1000)
        netParams.popParams["bkg_TH" + str(L[r])] = {
            "numCells": N_[r],
            "cellModel": "NetStim",
            "rate": 2 * (1000 * nTH) / Tth,
            "start": 200.0,
            "noise": 1.0,
            "number": nTH,
            "delay": 0,
        }
        auxConn = np.array([range(0, N_[r], 1), range(0, N_[r], 1)])
        netParams.connParams["bkg_TH->" + str(L[r])] = {
            "preConds": {"pop": "bkg_TH" + str(L[r])},
            "postConds": {"pop": L[r]},
            "connList": auxConn.T,
            "weight": f"max(0, {cfg.excWeight * cfg.scaleConnWeightNetStims} * (weightMin +normal(0,dweight*weightMin)))",
            "delay": 0.5,
            "synMech": "exc",
        }  # 1 delay

############################################################
# Connectivity parameters
############################################################

if cfg.connected:
    for r in range(0, 8):
        for c in range(0, 8):
            if L[c][-1] == "e":
                syn = "exc"
                weightScale = cfg.excWeight
            else:
                syn = "inh"
                weightScale = cfg.inhWeight
            if (c % 2) == 0:
                if c == 2 and r == 0:
                    netParams.connParams[str(L[c]) + "->" + str(L[r])] = {
                        "preConds": {"pop": L[c]},  # conditions of presyn cells
                        "postConds": {"pop": L[r]},  # conditions of postsyn cells
                        "divergence": cfg.ScaleFactor
                        * (
                            np.log(1.0 - C[r][c])
                            / np.log(1.0 - 1.0 / (N_Full[r] * N_Full[c]))
                        )
                        / N_Full[c],
                        "weight": f"2*max(0, {weightScale} * (weightMin +normal(0,dweight*weightMin)))",  # synaptic weight
                        "delay": "max(0.1, delayMin_e +normal(0,ddelay*delayMin_e))",  # transmission delay (ms)
                        "synMech": syn,
                    }
                else:
                    netParams.connParams[str(L[c]) + "->" + str(L[r])] = {
                        "preConds": {"pop": L[c]},  # conditions of presyn cells
                        "postConds": {"pop": L[r]},  # conditions of postsyn cells
                        "divergence": cfg.ScaleFactor
                        * (
                            np.log(1.0 - C[r][c])
                            / np.log(1.0 - 1.0 / (N_Full[r] * N_Full[c]))
                        )
                        / N_Full[c],
                        "weight": f"max(0, {weightScale} *(weightMin +normal(0,dweight*weightMin)))",  # synaptic weight
                        "delay": "max(0.1, delayMin_e +normal(0,ddelay*delayMin_e))",  # transmission delay (ms)
                        "synMech": syn,
                    }  # synaptic mechanism
            else:  # synaptic mechanism
                netParams.connParams[str(L[c]) + "->" + str(L[r])] = {
                    "preConds": {"pop": L[c]},  # conditions of presyn cells
                    "postConds": {"pop": L[r]},  # conditions of postsyn cells
                    "divergence": cfg.ScaleFactor
                    * (
                        np.log(1.0 - C[r][c])
                        / np.log(1.0 - 1.0 / (N_Full[r] * N_Full[c]))
                    )
                    / N_Full[c],
                    "weight": f"4*max(0, {weightScale} * (weightMin +normal(0,dweight*weightMin)))",  # synaptic weight
                    "delay": "max(0.1, delayMin_i +normal(0,ddelay*delayMin_i))",  # transmission delay (ms)
                    "synMech": syn,
                }  # synaptic mechanism

# netParams.connParams['S2->M'] = {
# 	'preConds': {'pop': 'bkg_IF'},
# 	'postConds': {'cellModel': 'IntFire_PD'},
# 	'probability': 1,
# 	'weight': 0,
# 	'delay': 0.5}

############################################################
# RxD params
############################################################

### constants
e_charge = 1.60217662e-19
scale = 1e-14 / e_charge
alpha = 5.3
px = cfg.px
dx = cfg.dx
Nz = cfg.Nz
o2sources = generateO2sources(
    cfg.fig_file, Nz, px, dx, x=netParams.sizeX, y=None, z=netParams.sizeY
)

constants = {
    "e_charge": e_charge,
    "scale": scale,
    "gnabar": cfg.gnabar * scale,  # molecules/um2 ms mV ,
    "gnabar_l": (0.0247 / 1000) * scale,
    "gkbar": cfg.gkbar * scale,
    "gkbar_l": (0.05 / 1000) * scale,
    "gclbar_l": (0.1 / 1000) * scale,
    "ukcc2": cfg.ukcc2 * mM / sec,
    "unkcc1": cfg.unkcc1 * mM / sec,
    "alpha": alpha,
    "epsilon_k_max": 0.25 / sec,
    "epsilon_o2": 0.17 / sec,
    "vtau": 1 / 250.0,
    "g_gliamax": 5 * mM / sec,
    "beta0": 7.0,
    "avo": 6.0221409 * (10**23),
    "p_max": cfg.pmax * mM / sec,
    "nao_initial": 144.0,
    "nai_initial": 18.0,
    "gnai_initial": 18.0,
    "gki_initial": 80.0,
    "ko_initial": 3.5,
    "ki_initial": 140.0,
    "clo_initial": 130.0,
    "cli_initial": 6.0,
    "o2_bath": cfg.o2_bath,
    "o2_init": cfg.o2_init,
    "v_initial": cfg.hParams["v_init"],
    "xbins": np.linspace(0, cfg.sizeX, o2sources.shape[1], endpoint=True),
    "ybins": np.linspace(-cfg.sizeY, 0, o2sources.shape[0], endpoint=True),
    "zbins": np.linspace(0, cfg.sizeZ, o2sources.shape[2], endpoint=True),
    "o2sources": o2sources,
}


# sodium activation 'm'
alpha_m = "(0.32 * (rxd.v + 54.0))/(1.0 - rxd.rxdmath.exp(-(rxd.v + 54.0)/4.0))"
beta_m = "(0.28 * (rxd.v + 27.0))/(rxd.rxdmath.exp((rxd.v + 27.0)/5.0) - 1.0)"
alpha_m0 = (0.32 * (constants["v_initial"] + 54.0)) / (
    1.0 - math.exp(-(constants["v_initial"] + 54) / 4.0)
)
beta_m0 = (0.28 * (constants["v_initial"] + 27.0)) / (
    math.exp((constants["v_initial"] + 27.0) / 5.0) - 1.0
)
m_initial = alpha_m0 / (beta_m0 + 1.0)

# sodium inactivation 'h'
alpha_h = "0.128 * rxd.rxdmath.exp(-(rxd.v + 50.0)/18.0)"
beta_h = "4.0/(1.0 + rxd.rxdmath.exp(-(rxd.v + 27.0)/5.0))"
alpha_h0 = 0.128 * math.exp(-(constants["v_initial"] + 50.0) / 18.0)
beta_h0 = 4.0 / (1.0 + math.exp(-(constants["v_initial"] + 27.0) / 5.0))
h_initial = alpha_h0 / (beta_h0 + 1.0)

# potassium activation 'n'
alpha_n = "(0.032 * (rxd.v + 52.0))/(1.0 - rxd.rxdmath.exp(-(rxd.v + 52.0)/5.0))"
beta_n = "0.5 * rxd.rxdmath.exp(-(rxd.v + 57.0)/40.0)"
alpha_n0 = (0.032 * (constants["v_initial"] + 52.0)) / (
    1.0 - math.exp(-(constants["v_initial"] + 52.0) / 5.0)
)
beta_n0 = 0.5 * math.exp(-(constants["v_initial"] + 57.0) / 40.0)
n_initial = alpha_n0 / (beta_n0 + 1.0)


### reactions
gna = "gnabar*mgate**3*hgate"
gk = "gkbar*ngate**4"
fko = "1.0 / (1.0 + rxd.rxdmath.exp(16.0 - kk[ecs] / vol_ratio[ecs]))"
nkcc1A = "rxd.rxdmath.log((kk[cyt] * cl[cyt] / vol_ratio[cyt]**2) / (kk[ecs] * cl[ecs] / vol_ratio[ecs]**2))"
nkcc1B = "rxd.rxdmath.log((na[cyt] * cl[cyt] / vol_ratio[cyt]**2) / (na[ecs] * cl[ecs] / vol_ratio[ecs]**2))"
nkcc1 = "unkcc1 * (%s) * (%s+%s)" % (fko, nkcc1A, nkcc1B)
kcc2 = "ukcc2 * rxd.rxdmath.log((kk[cyt] * cl[cyt] * vol_ratio[cyt]**2) / (kk[ecs] * cl[ecs] * vol_ratio[ecs]**2))"

# Nerst equation - reversal potentials
ena = "26.64 * rxd.rxdmath.log(na[ecs]*vol_ratio[cyt]/(na[cyt]*vol_ratio[ecs]))"
ek = "26.64 * rxd.rxdmath.log(kk[ecs]*vol_ratio[cyt]/(kk[cyt]*vol_ratio[ecs]))"
ecl = "26.64 * rxd.rxdmath.log(cl[cyt]*vol_ratio[ecs]/(cl[ecs]*vol_ratio[cyt]))"

o2ecs = "o2_extracellular[ecs_o2]"
o2switch = "(1.0 + rxd.rxdmath.tanh(1e4 * (%s - 5e-4))) / 2.0" % (o2ecs)
p = "%s / (1.0 + rxd.rxdmath.exp((20.0 - (%s/vol_ratio[ecs]) * 640)/3.0))" % (
    o2switch,
    o2ecs,
)
pumpA = f"(1.0 / (1.0 + rxd.rxdmath.exp(({cfg.KNai} - na[cyt] / vol_ratio[cyt])/3.0)))"
pumpB = f"(1.0 / (1.0 + rxd.rxdmath.exp({cfg.KKo} - kk[ecs] / vol_ratio[ecs])))"
pump_max = "p_max * (%s) * (%s)" % (pumpA, pumpB)  # pump with unlimited oxygen
pump = "(%s) * (%s)" % (p, pump_max)
gliapump = (
    "(1.0/3.0) * p_max * (%s / (1.0 + rxd.rxdmath.exp((25.0 - gnai_initial) / 3.0))) * (1.0 / (1.0 + rxd.rxdmath.exp(3.5 - kk[ecs]/vol_ratio[ecs])))"
    % (p)
)
g_glia = (
    "g_gliamax / (1.0 + rxd.rxdmath.exp(-((%s)*alpha/vol_ratio[ecs] - 2.5)/0.2))"
    % (o2ecs)
)
glia12 = "(%s) / (1.0 + rxd.rxdmath.exp((18.0 - kk[ecs] / vol_ratio[ecs])/2.5))" % (
    g_glia
)

# epsilon_k = "(epsilon_k_max/(1.0 + rxd.rxdmath.exp(-(((%s)/vol_ratio[ecs]) * alpha - 2.5)/0.2))) * (1.0/(1.0 + rxd.rxdmath.exp((-20 + ((1.0+1.0/beta0 -vol_ratio[ecs])/vol_ratio[ecs]) /2.0))))" % (o2ecs)
epsilon_kA = (
    "(epsilon_k_max/(1.0 + rxd.rxdmath.exp(-((%s/vol_ratio[ecs]) * alpha - 2.5)/0.2)))"
    % (o2ecs)
)
epsilon_kB = "(1.0/(1.0 + rxd.rxdmath.exp((-20 + ((1.0+1.0/beta0 - vol_ratio[ecs])/vol_ratio[ecs]) /2.0))))"
epsilon_k = "%s * %s" % (epsilon_kA, epsilon_kB)


volume_scale = "1e-18 * avo * %f" % (1.0 / cfg.sa2v)

avo = 6.0221409 * (10**23)
osm = "(1.1029 - 0.1029*rxd.rxdmath.exp( ( (na[ecs] + kk[ecs] + cl[ecs] + 18.0)/vol_ratio[ecs] - (na[cyt] + kk[cyt] + cl[cyt] + 132.0)/vol_ratio[cyt])/20.0))"
scalei = str(avo * 1e-18)
scaleo = str(avo * 1e-18)


# update constants to ensure net zero flux at RMP
evalInit = {
    "vol_ratio[ecs]": "1.0",
    "vol_ratio[cyt]": "1.0",
    "rxd.rxdmath": "math",
    "kk[cyt]": constants["ki_initial"],
    "kk[ecs]": constants["ko_initial"],
    "na[cyt]": constants["nai_initial"],
    "na[ecs]": constants["nao_initial"],
    "cl[cyt]": constants["cli_initial"],
    "cl[ecs]": constants["clo_initial"],
    "o2_extracellular[ecs_o2]": constants["o2_init"],
    "ngate": n_initial,
    "mgate": m_initial,
    "hgate": h_initial,
}


def initEval(ratestr):
    for k, v in evalInit.items():
        ratestr = ratestr.replace(k, str(v))
    for k, v in constants.items():
        ratestr = ratestr.replace(k, str(v))
    return eval(ratestr)


min_pmax = f"p_max * ({nkcc1} + {kcc2} + {gk} * (v_initial - {ek})/({volume_scale}))/(2*{pump_max})"
pmin = initEval(min_pmax)
if constants["p_max"] < pmin:
    print("Pump current is too low to balance K+ currents")
    print(f"p_max set to {pmin}")
    constants["p_max"] = pmin

# rescale pmax
"""
pA = "(1.0 / (1.0 + rxd.rxdmath.exp((25.0 - na[cyt] / vol_ratio[cyt])/3.0)))"
pB = "(1.0 / (1.0 + rxd.rxdmath.exp(3.5 - kk[ecs] / vol_ratio[ecs])))"
rA = f"(1.0 / (1.0 + rxd.rxdmath.exp(({cfg.KNai} - na[cyt] / vol_ratio[cyt])/3.0)))"
rB = f"(1.0 / (1.0 + rxd.rxdmath.exp({cfg.KKo} - kk[ecs] / vol_ratio[ecs])))"
rescale = initEval(f"{pA} * {pB}/({rA} * {rB})")
constants["p_max"] = constants["p_max"] * rescale
"""

clbalance = f"-((2.0 * {nkcc1} +  {kcc2}) * {volume_scale})/({ecl} - v_initial)"
kbalance = f"-(({nkcc1} + {kcc2} - 2 * {pump_max}) * {volume_scale} + ({gk} * (v_initial - {ek})))/(v_initial-{ek})"
nabalance = f"-(({nkcc1} + 3 * {pump_max}) * {volume_scale} + ({gna} * (v_initial - {ena})))/(v_initial-{ena})"

constants["gclbar_l"] = initEval(clbalance)
constants["gkbar_l"] = cfg.gkleak_scale * initEval(kbalance)
constants["gnabar_l"] = initEval(nabalance)


netParams.rxdParams["constants"] = constants

### regions
regions = {}

#### ecs dimensions
# margin = cfg.somaR
x = [0, cfg.sizeX]
y = [-cfg.sizeY, 0]
z = [0, cfg.sizeZ]

regions["ecs"] = {
    "extracellular": True,
    "xlo": x[0],
    "xhi": x[1],
    "ylo": y[0],
    "yhi": y[1],
    "zlo": z[0],
    "zhi": z[1],
    "dx": dx,
    "volume_fraction": cfg.alpha_ecs,
    "tortuosity": cfg.tort_ecs,
}

regions["ecs_o2"] = {
    "extracellular": True,
    "xlo": x[0],
    "xhi": x[1],
    "ylo": y[0],
    "yhi": y[1],
    "zlo": z[0],
    "zhi": z[1],
    "dx": dx,
    "volume_fraction": 1.0,
    "tortuosity": 1.0,
}

# xregions['cyt'] = {'cells': 'all', 'secs': 'all', 'nrn_region': 'i',
#                 'geometry': {'class': 'FractionalVolume',
#                 'args': {'volume_fraction': cfg.cyt_fraction, 'surface_fraction': 1}}}

# xregions['mem'] = {'cells' : 'all', 'secs' : 'all', 'nrn_region' : None, 'geometry' : 'membrane'}

regions["cyt"] = {
    "cells": L,
    "secs": "all",
    "nrn_region": "i",
    "geometry": {
        "class": "FractionalVolume",
        "args": {"volume_fraction": cfg.cyt_fraction, "surface_fraction": 1},
    },
}

regions["mem"] = {"cells": L, "secs": "all", "nrn_region": None, "geometry": "membrane"}

netParams.rxdParams["regions"] = regions

### species
species = {}

k_init_str = (
    "ki_initial if isinstance(node, rxd.node.Node1D) else (%f if ((node.x3d - %f/2)**2+(node.y3d + %f/2)**2+(node.z3d - %f/2)**2 <= %f**2) else ko_initial)"
    % (cfg.k0, cfg.sizeX, cfg.sizeY, cfg.sizeZ, cfg.r0)
)
# k_init_str = 'ki_initial if isinstance(node, rxd.node.Node1D) else (%f if (((node.x3d - %f/2)**2+(node.z3d - %f/2)**2 < %f**2) and (-1100 < node.y3d < -900)) and  else ko_initial)' % (cfg.k, cfg.sizeX, cfg.sizeZ, cfg.r0)
species["kk"] = {
    "regions": ["cyt", "mem", "ecs"],
    "d": 2.62,
    "charge": 1,
    "initial": k_init_str,
    "ecs_boundary_conditions": constants["ko_initial"]
    if cfg.prep == "invitro"
    else None,
    "name": "k",
}

species["na"] = {
    "regions": ["cyt", "mem", "ecs"],
    "d": 1.78,
    "charge": 1,
    "initial": "nai_initial if isinstance(node, rxd.node.Node1D) else nao_initial",
    "ecs_boundary_conditions": constants["nao_initial"]
    if cfg.prep == "invitro"
    else None,
    "name": "na",
}

species["cl"] = {
    "regions": ["cyt", "mem", "ecs"],
    "d": 2.1,
    "charge": -1,
    "initial": "cli_initial if isinstance(node, rxd.node.Node1D) else clo_initial",
    "ecs_boundary_conditions": constants["clo_initial"]
    if cfg.prep == "invitro"
    else None,
    "name": "cl",
}

# o2_init_str = 'o2_bath if isinstance(node, rxd.node.Node1D) else (0.4*o2sources[numpy.argmin((ybins-node.y3d)**2),numpy.argmin((xbins-node.x3d)**2),numpy.argmin((zbins-node.z3d)**2)] if o2sources[numpy.argmin((ybins-node.y3d)**2),numpy.argmin((xbins-node.x3d)**2),numpy.argmin((zbins-node.z3d)**2)] else 0.04)'
# species['o2_extracellular'] = {'regions' : ['ecs_o2'], 'd' : 3.3, 'initial' : 0.04,
#                 'ecs_boundary_conditions' : 0.04, 'name' : 'o2'}
zscale = dx / (cfg.sizeZ / o2sources.shape[2])
yscale = dx / (cfg.sizeY / o2sources.shape[1])
xscale = dx / (cfg.sizeX / o2sources.shape[0])
o2_init_str = f"o2_bath if isinstance(node, rxd.node.Node1D) else 0.4*max(o2sources[int(node._j*{xscale}), int(node._i*{yscale}), int(node._k*{zscale})],0.1)"
species["o2_extracellular"] = {
    "regions": ["ecs_o2"],
    "d": 3.3,
    "initial": constants["o2_init"],
    "ecs_boundary_conditions": constants["o2_bath"] if cfg.prep == "invitro" else None,
    "name": "o2",
}
# species['o2_extracellular'] = {'regions' : ['ecs_o2'], 'd' : 3.3, 'initial' : constants['o2_bath'],
#                 'ecs_boundary_conditions' : constants['o2_bath'], 'name' : 'o2'}

netParams.rxdParams["species"] = species

### parameters
params = {}
params["dump"] = {"regions": ["cyt", "ecs", "ecs_o2"], "name": "dump"}

params["ecsbc"] = {
    "regions": ["ecs", "ecs_o2"],
    "name": "ecsbc",
    "value": "1 if (abs(node.x3d - ecs._xlo) < ecs._dx[0] or abs(node.x3d - ecs._xhi) < ecs._dx[0] or abs(node.y3d - ecs._ylo) < ecs._dx[1] or abs(node.y3d - ecs._yhi) < ecs._dx[1] or abs(node.z3d - ecs._zlo) < ecs._dx[2] or abs(node.z3d - ecs._zhi) < ecs._dx[2]) else 0",
}

if cfg.ouabain:
    params["iscore"] = {
        "regions": ["ecs_o2", "ecs", "mem"],
        "name": "iscore",
        "value": "(0 if ((node.x3d - %f/2)**2+(node.y3d + %f/2)**2+(node.z3d - %f/2)**2 <= %f**2) else 1)"
        % (cfg.sizeX, cfg.sizeY, cfg.sizeZ, cfg.r0),
    }

if cfg.o2drive:
    params["numcap"] = {
        "regions": ["ecs_o2"],
        "name": "numcap",
        "value": f"o2sources[int(node._i*{xscale}), int(node._j*{yscale}), int(node._k*{zscale})]",
    }

netParams.rxdParams["parameters"] = params

### states
netParams.rxdParams["states"] = {
    "vol_ratio": {"regions": ["cyt", "ecs"], "initial": 1.0, "name": "volume"},
    "mgate": {"regions": ["cyt", "mem"], "initial": m_initial, "name": "mgate"},
    "hgate": {"regions": ["cyt", "mem"], "initial": h_initial, "name": "hgate"},
    "ngate": {"regions": ["cyt", "mem"], "initial": n_initial, "name": "ngate"},
}


### reactions
mcReactions = {}

## volume dynamics
mcReactions["vol_dyn"] = {
    "reactant": "vol_ratio[cyt]",
    "product": "dump[ecs]",
    "rate_f": "-1 * (%s) * vtau * ((%s) - vol_ratio[cyt])" % (scalei, osm),
    "membrane": "mem",
    "custom_dynamics": True,
    "scale_by_area": False,
}

mcReactions["vol_dyn_ecs"] = {
    "reactant": "dump[cyt]",
    "product": "vol_ratio[ecs]",
    "rate_f": "-1 * (%s) * vtau * ((%s) - vol_ratio[cyt])" % (scaleo, osm),
    "membrane": "mem",
    "custom_dynamics": True,
    "scale_by_area": False,
}

# # CURRENTS/LEAKS ----------------------------------------------------------------
# sodium (Na) current
mcReactions["na_current"] = {
    "reactant": "na[cyt]",
    "product": "na[ecs]",
    "rate_f": "%s * (rxd.v - %s )" % (gna, ena),
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}

# potassium (K) current
mcReactions["k_current"] = {
    "reactant": "kk[cyt]",
    "product": "kk[ecs]",
    "rate_f": "%s * (rxd.v - %s)" % (gk, ek),
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}

# nkcc1 (Na+/K+/2Cl- cotransporter)
mcReactions["nkcc1_current1"] = {
    "reactant": "cl[cyt]",
    "product": "cl[ecs]",
    "rate_f": "2.0 * (%s) * (%s)" % (nkcc1, volume_scale),
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}

mcReactions["nkcc1_current2"] = {
    "reactant": "kk[cyt]",
    "product": "kk[ecs]",
    "rate_f": "%s * %s" % (nkcc1, volume_scale),
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}

mcReactions["nkcc1_current3"] = {
    "reactant": "na[cyt]",
    "product": "na[ecs]",
    "rate_f": "%s * %s" % (nkcc1, volume_scale),
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}

# ## kcc2 (K+/Cl- cotransporter)
mcReactions["kcc2_current1"] = {
    "reactant": "cl[cyt]",
    "product": "cl[ecs]",
    "rate_f": "%s * %s" % (kcc2, volume_scale),
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}

mcReactions["kcc2_current2"] = {
    "reactant": "kk[cyt]",
    "product": "kk[ecs]",
    "rate_f": "%s * %s" % (kcc2, volume_scale),
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}

## sodium leak
mcReactions["na_leak"] = {
    "reactant": "na[cyt]",
    "product": "na[ecs]",
    "rate_f": "gnabar_l * (rxd.v - %s)" % (ena),
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}

# ## potassium leak
mcReactions["k_leak"] = {
    "reactant": "kk[cyt]",
    "product": "kk[ecs]",
    "rate_f": "gkbar_l * (rxd.v - %s)" % (ek),
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}

# ## chlorine (Cl) leak
mcReactions["cl_current"] = {
    "reactant": "cl[cyt]",
    "product": "cl[ecs]",
    "rate_f": "gclbar_l * (%s - rxd.v)" % (ecl),
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}

# ## Na+/K+ pump current in neuron (2K+ in, 3Na+ out)
mcReactions["pump_current"] = {
    "reactant": "kk[cyt]",
    "product": "kk[ecs]",
    "rate_f": "(-2.0 * %s * %s)" % (pump, volume_scale),
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}

mcReactions["pump_current_na"] = {
    "reactant": "na[cyt]",
    "product": "na[ecs]",
    "rate_f": "(3.0 * %s * %s)" % (pump, volume_scale),
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}

# O2 depletrion from Na/K pump in neuron
mcReactions["oxygen"] = {
    "reactant": o2ecs,
    "product": "dump[cyt]",
    "rate_f": "(%s) * (%s)" % (pump, volume_scale),
    "membrane": "mem",
    "custom_dynamics": True,
}

netParams.rxdParams["multicompartmentReactions"] = mcReactions

# RATES--------------------------------------------------------------------------
rates = {}
## dm/dt
rates["m_gate"] = {
    "species": "mgate",
    "regions": ["cyt", "mem"],
    "rate": "((%s) * (1.0 - mgate)) - ((%s) * mgate)" % (alpha_m, beta_m),
}

## dh/dt
rates["h_gate"] = {
    "species": "hgate",
    "regions": ["cyt", "mem"],
    "rate": "((%s) * (1.0 - hgate)) - ((%s) * hgate)" % (alpha_h, beta_h),
}

## dn/dt
rates["n_gate"] = {
    "species": "ngate",
    "regions": ["cyt", "mem"],
    "rate": "((%s) * (1.0 - ngate)) - ((%s) * ngate)" % (alpha_n, beta_n),
}

if cfg.o2drive:
    rates["o2source"] = {
        "species": o2ecs,
        "regions": ["ecs_o2"],
        "rate": "numcap * (epsilon_o2 * (%s - %s))" % (cfg.o2drive, o2ecs),
    }

# rates['o2diff'] = {'species' : o2ecs, 'regions' : ['ecs_o2'],
#     'rate' : '(epsilon_o2 * (o2_bath - %s/vol_ratio[ecs]))' % (o2ecs)} # o2everywhere

# rates['o2diff'] = {'species' : o2ecs, 'regions' : ['ecs_o2'],
#     'rate' : '(epsilon_o2 * (o2_bath - %s))' % (o2ecs)} # o2everywhereNoVolScale


if cfg.prep == "invitro":
    ## diffusion
    rates["o2diff"] = {
        "species": o2ecs,
        "regions": ["ecs_o2"],
        "rate": "ecsbc * (epsilon_o2 * (o2_bath - %s/vol_ratio[ecs]))" % (o2ecs),
    }  # original

    rates["kdiff"] = {
        "species": "kk[ecs]",
        "regions": ["ecs"],
        "rate": "ecsbc * ((%s) * (ko_initial - kk[ecs]/vol_ratio[ecs]))" % (epsilon_k),
    }

    rates["nadiff"] = {
        "species": "na[ecs]",
        "regions": ["ecs"],
        "rate": "ecsbc * ((%s) * (nao_initial - na[ecs]/vol_ratio[ecs]))" % (epsilon_k),
    }

    rates["cldiff"] = {
        "species": "cl[ecs]",
        "regions": ["ecs"],
        "rate": "ecsbc * ((%s) * (clo_initial - cl[ecs]/vol_ratio[ecs]))" % (epsilon_k),
    }

## Glia K+/Na+ pump current
rates["glia_k_current"] = {
    "species": "kk[ecs]",
    "regions": ["ecs"],
    "rate": "(-(%s) - (2.0 * (%s)))" % (glia12, gliapump),
}

rates["glia_na_current"] = {
    "species": "na[ecs]",
    "regions": ["ecs"],
    "rate": "(3.0 * (%s))" % (gliapump),
}

## Glial O2 depletion
rates["o2_pump"] = {
    "species": o2ecs,
    "regions": ["ecs_o2"],
    "rate": "-(%s)" % (gliapump),
}

netParams.rxdParams["rates"] = rates

# # plot statistics for 10% of cells
# scale = 10 #max(1,int(sum(N_[:8])/1000))
# include = [(pop, range(0, netParams.popParams[pop]['numCells'], scale)) for pop in L]
# cfg.analysis['plotSpikeStats']['include'] = include


# v0.0 - combination of netParams from ../uniformdensity and netpyne PD thalamocortical model
# v0.1 - got rid of all non cortical cells, background stim
# v0.2 - adding connections back in
# v0.3 - scaled original connections by 1e-7
# v0.4 - original poisson, rather than netstim inputs
# v0.5 - now includes thalamocortical inputs
# v0.6 - different weight scaling for excitation and inhibition
# v0.7 - using netParams.scaleConnWeight (which excludes netstims)
# v0.8 - normally distributed connection weights based on whats worked for fixed conns
# v1.0 - added in o2 sources based on capillaries identified from histology
# v1.1 - updated parameters to avoid SD in baseline simulations
