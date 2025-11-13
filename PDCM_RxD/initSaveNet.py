from netpyne import sim
import numpy as np
import os
import sys
import pickle
from neuron import h
import random
from stats import networkStatsFromSim
import json


def rand_uniform(gid, lb=0, ub=1):

    r = h.Random()
    r.Random123(gid, 1, 1)
    return r.uniform(lb, ub)


cfg, netParams = sim.readCmdLineArgs(
    simConfigDefault="cfgMidOx.py", netParamsDefault="netParamsMidOx.py"
)
subdir = f"{cfg.ox}_{cfg.k0Layer}_{cfg.k0}_{cfg.o2drive}"
outdir = cfg.saveFolder + os.path.sep + subdir

# Additional sim setup
# parallel context
pc = h.ParallelContext()
pcid = pc.id()
nhost = pc.nhost()
pc.timeout(0)
pc.set_maxstep(100)  # required when using multiple processes
random.seed(pcid + cfg.seeds["rec"])


def restoreSS():
    global lastss
    """restore sim state from saved files"""
    print(f"restore Save State from {cfg.restoredir}")
    svst = h.SaveState()
    f = h.File(
        os.path.join(
            os.path.join(cfg.restoredir, subdir), "save_test_" + str(pcid) + ".dat"
        )
    )
    print("loaded file", f)
    svst.fread(f)
    print("read file", svst)
    svst.restore()
    print("restored", h.t)
    rvSeq = pickle.load(
        open(os.path.join(outdir, "save_randvar_" + str(pcid) + ".pkl"), "rb")
    )
    setRandSeq(rvSeq)
    lastss = h.t


def fi(cells):
    """set steady state RMP each cell
    when not restoring from a previous simulation"""

    # Don't run this -- it does not account for elevated K+
    """
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
    """
    # restore from previous sim
    if cfg.restore:
        restoreSS()


def fi0(cells):
    for cell in cells:
        v = rand_uniform(cell.gid, cfg.cellPopsInit[0], cfg.cellPopsInit[1])
        for sec in cell.secs.values():
            if "hObj" in sec:
                sec["hObj"].v = v


sim.initialize(
    simConfig=cfg, netParams=netParams
)  # create network object and set cfg and net params
sim.cfg.saveCellConns = True
sim.cfg.createPyStruct = True
sim.net.createPops()  # instantiate network populations
sim.net.createCells()  # instantiate network cells based on defined populations

sim.net.connectCells()  # create connections between cells based on params
sim.net.addStims()  # add external stimulation to cells (IClamps etc)

myN = len(sim.net.cells)
allN = pc.py_allgather(myN)
N = sum(allN)

Wexc = np.zeros((N, N))
Winh = np.zeros((N, N))
for gid in range(N):
    cell = sim.cellByGid(gid)
    if hasattr(cell, "conns"):
        for conn in cell.conns:
            if "preGid" in conn:
                pid = int(conn["preGid"])
                if conn["synMech"] == "exc":
                    Wexc[pid][gid] += conn["weight"]
                else:
                    Winh[pid][gid] += conn["weight"]

if pcid != 0:
    np.save(f"tmp_Wexc{pcid}.npy", Wexc)
    np.save(f"tmp_Winh{pcid}.npy", Winh)
    del Wexc
    del Winh


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


if (nhost > 0 and pcid != 1) or (pcid != 0):
    pickle.dump(pops, open(f'tmp{pcid}_populationsSeed_{cfg.seeds["loc"]}.pkl', "wb"))
    pickle.dump(cells, open(f'tmp{pcid}_positionsSeed_{cfg.seeds["loc"]}.pkl', "wb"))

pc.barrier()
if pcid == 0:
    print("merge connections")
    # merge connections
    for i in range(1, nhost):
        Wexc += np.load(f"tmp_Wexc{i}.npy")
        Winh += np.load(f"tmp_Winh{i}.npy")
        os.remove(f"tmp_Wexc{i}.npy")
        os.remove(f"tmp_Winh{i}.npy")
    np.save("Wexc.npy", Wexc)
    np.save("Winh.npy", Winh)
    print("saved connections")

if (nhost > 0 and pcid == 1):
    print("merge positions")
    # merge positions
    for i in range(nhost):
        if nhost > 0 and i == 1:
            continue
        newPops = pickle.load(
            open(f"tmp{i}_populationsSeed_{cfg.seeds['loc']}.pkl", "rb")
        )
        newCells = pickle.load(
            open(f'tmp{i}_positionsSeed_{cfg.seeds["loc"]}.pkl', "rb")
        )
        for k, v in newPops.items():
            if k in pops:
                pops[k] = pops[k] + v
            else:
                pops[k] = v
        cells.update(newCells)
        os.remove(f"tmp{i}_populationsSeed_{cfg.seeds['loc']}.pkl")
        os.remove(f'tmp{i}_positionsSeed_{cfg.seeds["loc"]}.pkl')

    for pop in pops:
        pops[pop].sort()

    pickle.dump(pops, open(f"populationsSeed_{cfg.seeds['loc']}.pkl", "wb"))
    cells = [cells[x] for x in range(max(list(cells.keys())))]
    np.save(f'positionsSeed_{cfg.seeds["loc"]}.npy', cells)
    print("saved positions")

pc.barrier()
print("done")
