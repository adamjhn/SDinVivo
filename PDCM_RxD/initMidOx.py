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
    simConfigDefault="cfgMidOx.py", netParamsDefault="netParamsPump.py"
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
sim.net.createPops()  # instantiate network populations
sim.net.createCells()  # instantiate network cells based on defined populations

sim.net.connectCells()  # create connections between cells based on params
sim.net.addStims()  # add external stimulation to cells (IClamps etc)
sim.net.addRxD(nthreads=6)  # add reaction-diffusion (RxD)
sim.setupRecording()  # setup variables to record for each cell
fih = h.FInitializeHandler(1, lambda: fi(sim.net.cells))
if not cfg.restore:
    fih0 = h.FInitializeHandler(0, lambda: fi0(sim.getCellsList(include=cfg.cellPops)))

# only single core stuff
if pcid == 0:
    # create output dir
    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except:
            print("Unable to create the directory %r for the data and figures" % outdir)
            os._exit(1)

    # set variables for ecs concentrations
    k_ecs = sim.net.rxd["species"]["kk"]["hObj"][sim.net.rxd["regions"]["ecs"]["hObj"]]
    na_ecs = sim.net.rxd["species"]["na"]["hObj"][sim.net.rxd["regions"]["ecs"]["hObj"]]
    cl_ecs = sim.net.rxd["species"]["cl"]["hObj"][sim.net.rxd["regions"]["ecs"]["hObj"]]
    o2_ecs = sim.net.rxd["species"]["oxygen"]["hObj"][
        sim.net.rxd["regions"]["ecs"]["hObj"]
    ]
    o2con = sim.net.rxd["states"]["o2_consumed"]["hObj"][
        sim.net.rxd["regions"]["ecs"]["hObj"]
    ]

# manually record from cells from each layer
rng = np.random.default_rng(seed=pcid + cfg.seeds["rec"])
rec_cells = {}
for lab, pop in sim.net.pops.items():
    if "xRange" in pop.tags:
        rec_cells[lab] = {
            "gid": (
                rng.choice(
                    pop.cellGids, size=int(min(1, cfg.nRec / nhost)), replace=False
                )
                if len(pop.cellGids) > min(1, cfg.nRec / nhost)
                else pop.cellGids
            )
        }
        rec_cells[lab]["pos"] = []
        for k in ["v", "ki", "nai", "cli", "ko", "nao", "clo", "o2o"]:
            rec_cells[lab][k] = []
        for idx in rec_cells[lab]["gid"]:
            cell = sim.cellByGid(idx)
            soma = cell.secs["soma"]["hObj"]
            rec_cells[lab]["pos"].append(cell.getSomaPos())
            for k in ["v", "ki", "nai", "cli", "ko", "nao", "clo", "o2o"]:
                rec_cells[lab][k].append(
                    h.Vector().record(getattr(soma(0.5), f"_ref_{k}"))
                )
if pcid == 0:
    rec_cells["time"] = h.Vector().record(h._ref_t)


def getRandSeq():
    """Return a dict of gid->rv.get_seq for all cells with
    cellModel=='NetStim'
    In this model the NetStim are cells so are not listed in
    netParams.stimSourceParams, so cannot access them with
    include=['allNetStims'].
    """
    saveSeq = {"seq": {}, "ids": {}}
    for cell in sim.getCellsList(include=["all"]):
        if cell.tags["cellModel"] == "NetStim":
            saveSeq["seq"][cell.gid] = cell.hPointp.ranvar.get_seq()
            saveSeq["ids"][cell.gid] = list(cell.hPointp.ranvar.get_ids().as_numpy())
    return saveSeq


def setRandSeq(seqDict):
    """Take a dict of gid->seq and apply it to all cells"""
    done = []
    for cell in sim.getCellsList(include=["all"]):
        if cell.tags["cellModel"] == "NetStim":
            if cell.gid in seqDict["seq"]:
                cell.hPointp.ranvar.set_seq(seqDict["seq"][cell.gid])
                # cell.hPointp.ranvar.set_ids(*seqDict['ids'][cell.gid])
                done.append(cell.gid)
            else:
                raise Exception(
                    f"Failed to set randvar for cell {cell} gid {cell.gid} -- missing value"
                )
    for gid in seqDict["seq"]:
        if gid not in done:
            raise Exception(f"Failed to set randvar for cell gid {gid}")


def runSS():
    svst = h.SaveState()
    svst.save()
    f = h.File(os.path.join(outdir, "save_test_" + str(pcid) + ".dat"))
    svst.fwrite(f)
    rvSeq = getRandSeq()
    pickle.dump(
        rvSeq, open(os.path.join(outdir, "save_randvar_" + str(pcid) + ".pkl"), "wb")
    )


def saveconc():
    np.save(os.path.join(outdir, "k_%i.npy" % int(h.t)), k_ecs.states3d)
    np.save(os.path.join(outdir, "na_%i.npy" % int(h.t)), na_ecs.states3d)
    np.save(os.path.join(outdir, "cl_%i.npy" % int(h.t)), cl_ecs.states3d)
    np.save(os.path.join(outdir, "o2_%i.npy" % int(h.t)), o2_ecs.states3d)
    np.save(os.path.join(outdir, "o2con_%i.npy" % int(h.t)), o2con.states3d)


def progress_bar(tstop, size=40):
    """report progress of the simulation"""
    prog = h.t / float(tstop)
    fill = int(size * prog)
    empt = size - fill
    progress = "#" * fill + "-" * empt
    sys.stdout.write(
        "[%s] %2.1f%% %6.1fms of %6.1fms\r" % (progress, 100 * prog, h.t, tstop)
    )
    sys.stdout.flush()


fout = None
lastss = 0
if pcid == 0:
    # record the wave progress
    fout = open(os.path.join(outdir, "wave_progress.txt"), "a")
    if cfg.k0Layer is None:
        yoff = cfg.sizeY / 2.0
    elif cfg.k0Layer == 2 or cfg.k0Layer == 3:
        yoff = sum(netParams.popParams["L2e"]["yRange"]) / 2
    elif cfg.k0Layer == 4:
        yoff = sum(netParams.popParams["L4e"]["yRange"]) / 2
    elif cfg.k0Layer == 5:
        yoff = sum(netParams.popParams["L5e"]["yRange"]) / 2
    elif cfg.k0Layer == 6:
        yoff = sum(netParams.popParams["L6e"]["yRange"]) / 2

cellSDOpen, cellSDClosed = {}, {}


def runIntervalFunc(t):
    """Write the wave_progress every 1ms"""
    global lastss, cellSDOpen, cellSDClosed
    saveint = 100  # save concentrations interval
    ssint = 1000  # save state interval
    lastss = 0
    if pcid == 0:
        if int(t) % saveint == 0:
            # plot extracellular concentrations averaged over depth every 100ms
            saveconc()
    for cell in sim.getCellsList(include=cfg.cellPops):
        v = cell.secs["soma"]["hObj"].v

        # check previously depolarized cells
        if cell.gid in cellSDOpen:
            if v <= cfg.SDThreshold:
                a = cellSDOpen[cell.gid]

                # if the cell was only depolarized for a single interval
                # it was probably just an AP -- remove it
                del cellSDOpen[cell.gid]
                if abs(a - h.t) > 2.5:
                    if cell.gid in cellSDClosed:
                        cellSDClosed[cell.gid].append((a, h.t))
                    else:
                        cellSDClosed[cell.gid] = [(a, h.t)]
        else:
            if v > cfg.SDThreshold:
                cellSDOpen[cell.gid] = h.t 
    if ((int(t) % ssint == 0) and (h.t - lastss) > ssint) or (cfg.duration - t) < 1:
        runSS()
        lastss = t

        # sustained depolarization at current time
        cellSD = cellSDClosed.copy()
        for cell, a in cellSDOpen.items():
            if cell in cellSD:
                cellSD[cell].append([a, None])
            else:
                cellSD[cell] = [(a, None)]
        json.dump(cellSD, open(os.path.join(outdir, f"cellsSD_{pcid}.json"), "w"))

    if pcid == 0:
        progress_bar(cfg.duration)
        dist = 0
        dist1 = 1e9
        for nd in sim.net.rxd.species["kk"]["hObj"].nodes:
            if str(nd.region).split("(")[0] == "Extracellular":
                r = (
                    (nd.x3d - cfg.sizeX / 2.0) ** 2
                    + (nd.y3d + yoff) ** 2
                    + (nd.z3d - cfg.sizeZ / 2.0) ** 2
                ) ** 0.5
                if nd.concentration > cfg.Kceil and r > dist:
                    dist = r
                if nd.concentration <= cfg.Kceil and r < dist1:
                    dist1 = r
        fout.write("%g\t%g\t%g\n" % (h.t, dist, dist1))
        fout.flush()


sim.runSimWithIntervalFunc(1, runIntervalFunc)
sim.gatherData()
if pcid == 0:
    networkStatsFromSim(
        sim, filename=os.path.join(outdir, f"netstats_{cfg.duration/1000:0.2f}s.json")
    )

sim.saveData()
sim.analysis.plotData()
# merge rec_cells
rec_all = {}
for lab in rec_cells:
    # time only on pcid==0 - don't gather
    if lab == "time":
        rec_all["time"] = rec_cells["time"]
    else:
        rec_all[lab] = {}
        for k in rec_cells[lab]:
            # merge lists
            rec_all[lab][k] = pc.py_gather(rec_cells[lab][k], 0)

if pcid == 0:
    progress_bar(cfg.duration)
    fout.close()
    for lab in rec_all:
        if cfg.restore:
            rec_old = pickle.load(open(os.path.join(outdir, f"recs_{lab}.pkl"), "rb"))
            if lab == "time":
                rec_old.append(rec_all["time"])
            else:
                for k in rec_all[lab]:
                    if k != "pos" and k != "pop" and k != "gid":
                        for u, v in zip(rec_all[lab][k], rec_old[k]):
                            for x, y in zip(u, v):
                                y.append(x)
            pickle.dump(rec_old, open(os.path.join(outdir, f"recs_{lab}.pkl"), "wb"))
        else:
            pickle.dump(
                rec_all[lab], open(os.path.join(outdir, f"recs_{lab}.pkl"), "wb")
            )
    print("\nSimulation complete. Plotting membrane potentials")

# v0.0 - direct copy from ../uniformdensity/init.py
# v1.0 - added in o2 sources based on capillaries identified from histology
# v1.1 - set pas.e to maintain RMP and move restore state function
# v1.2 - replace centermembrane_potential with layer specific recordings
# v1.3 - fix save state (in NEURON 9) by restoring seq in NMODLRandom
