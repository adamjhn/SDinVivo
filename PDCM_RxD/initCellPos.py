from netpyne import sim
from netParamsMidOx import netParams
import numpy as np
import os
import sys
import pickle
from neuron import h, rxd
import random
from matplotlib import pyplot as plt
from stats import networkStatsFromSim

cfg, netParams = sim.readCmdLineArgs(
    simConfigDefault="cfgMidOx.py", netParamsDefault="netParamsMidOx.py"
)

# Additional sim setup
## parallel context
pc = h.ParallelContext()
pcid = pc.id()
nhost = pc.nhost()
pc.timeout(0)
pc.set_maxstep(100)  # required when using multiple processes
random.seed(pcid + cfg.seeds["rec"])


def restoreSS():
    """restore sim state from saved files"""
    restoredir = cfg.restoredir
    svst = h.SaveState()
    f = h.File(os.path.join(restoredir, "save_test_" + str(pcid) + ".dat"))
    svst.fread(f)
    svst.restore()


def fi(cells):
    """set steady state RMP each cell -- when not restoring from a previous simulation"""
    cfg.e_pas = {}
    for c in cells:
        # skip artificial cells
        if not hasattr(c.secs, "soma"):
            continue
        seg = c.secs.soma.hObj(0.5)
        isum = 0
        isum = (
            (seg.ina if h.ismembrane("na_ion") else 0)
            + (seg.ik if h.ismembrane("k_ion") else 0)
            + (seg.ica if h.ismembrane("ca_ion") else 0)
            + (seg.iother if h.ismembrane("other_ion") else 0)
        )
        seg.e_pas = cfg.hParams["v_init"] + isum / seg.g_pas

    ## restore from previous sim
    if cfg.restoredir:
        restoredir = cfg.restoredir
        restoreSS()


sim.initialize(
    simConfig=cfg, netParams=netParams
)  # create network object and set cfg and net params
sim.net.createPops()  # instantiate network populations
sim.net.createCells()  # instantiate network cells based on defined populations


pops = {}
cells = {}
for cell in sim.net.cells:
    pop = cell.tags["pop"]
    x, y, z = cell.tags["x"], cell.tags["y"], cell.tags["z"]
    cells[cell.gid] = [x, y, z]
    if pop in pops:
        pops[pop].append(cell.gid)
    else:
        pops[pop] = [cell.gid]

cells = [cells[x] for x in range(max(list(cells.keys())))]

pickle.dump(pops, open(f'populationsSeed_{cfg.seeds["loc"]}.pkl', "wb"))
np.save(open(f'positionsSeed_{cfg.seeds["loc"]}.npy', "wb"), cells)
