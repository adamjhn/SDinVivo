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
subdir = f"{cfg.ox}_{cfg.k0Layer}_{cfg.o2drive}"
outdir = cfg.saveFolder + os.path.sep + subdir

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
        print(seg, seg.e_pas)

    ## restore from previous sim
    if cfg.restoredir:
        restoredir = cfg.restoredir
        restoreSS()


sim.initialize(
    simConfig=cfg, netParams=netParams
)  # create network object and set cfg and net params
sim.net.createPops()  # instantiate network populations
sim.net.createCells()  # instantiate network cells based on defined populations
sim.net.connectCells()  # create connections between cells based on params
sim.net.addStims()  # add external stimulation to cells (IClamps etc)
sim.net.addRxD(nthreads=6)  # add reaction-diffusion (RxD)
# fih = h.FInitializeHandler(2, lambda: fi(sim.net.cells))
sim.setupRecording()  # setup variables to record for each cell (spikes, V traces, etc)

all_secs = [sec for sec in h.allsec()]
cells_per_node = len(all_secs)
if cfg.singleCells:
    rec_inds = list(range(cells_per_node))
else:
    rec_inds = random.sample(range(cells_per_node), int(cfg.nRec / nhost))
rec_cells = [h.Vector().record(all_secs[ind](0.5)._ref_v) for ind in rec_inds]
pos = [
    [all_secs[ind].x3d(0), all_secs[ind].y3d(0), all_secs[ind].z3d(0)]
    for ind in rec_inds
]
pops = [str(all_secs[ind]).split(".")[1].split("s")[0] for ind in rec_inds]

## only single core stuff
if pcid == 0:
    ## create output dir
    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except:
            print("Unable to create the directory %r for the data and figures" % outdir)
            os._exit(1)

    ## set variables for ecs concentrations
    k_ecs = sim.net.rxd["species"]["kk"]["hObj"][sim.net.rxd["regions"]["ecs"]["hObj"]]
    na_ecs = sim.net.rxd["species"]["na"]["hObj"][sim.net.rxd["regions"]["ecs"]["hObj"]]
    cl_ecs = sim.net.rxd["species"]["cl"]["hObj"][sim.net.rxd["regions"]["ecs"]["hObj"]]
    o2_ecs = sim.net.rxd["species"]["o2_extracellular"]["hObj"][
        sim.net.rxd["regions"]["ecs_o2"]["hObj"]
    ]
    o2con = sim.net.rxd["species"]["o2_extracellular"]["hObj"][
        sim.net.rxd["regions"]["o2con"]["hObj"]
    ]

    ## manually record from cells according to distance from the center of the slice
    rng = np.random.default_rng(seed=pcid + cfg.seeds["rec"])
    rec_cells = {}
    for lab, pop in sim.net.pops.items():
        if "xRange" in pop.tags:
            rec_cells[lab] = {
                "gid": rng.choice(pop.cellGids, size=cfg.nRec, replace=False)
                if len(pop.cellGids) > cfg.nRec
                else pop.cellGids
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

    rec_cells["time"] = h.Vector().record(h._ref_t)


def saveRxd():
    for sp in rxd.species._all_species:
        s = sp()
        np.save(
            os.path.join(outdir, s.name + "_concentrations_" + str(pcid) + ".npy"),
            s.nodes.concentration,
        )


def runSS():
    svst = h.SaveState()
    svst.save()
    f = h.File(os.path.join(outdir, "save_test_" + str(pcid) + ".dat"))
    svst.fwrite(f)


def saveconc():
    np.save(os.path.join(outdir, "k_%i.npy" % int(h.t)), k_ecs.states3d)
    np.save(os.path.join(outdir, "na_%i.npy" % int(h.t)), na_ecs.states3d)
    np.save(os.path.join(outdir, "cl_%i.npy" % int(h.t)), cl_ecs.states3d)
    np.save(os.path.join(outdir, "o2_%i.npy" % int(h.t)), o2_ecs.states3d)
    np.save(os.path.join(outdir, "o2con_%i.npy" % int(h.t)), o2com.states3d)


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


def runIntervalFunc(t):
    """Write the wave_progress every 1ms"""
    global lastss
    saveint = 100  # save concentrations interval
    ssint = 10000  # save state interval
    lastss = 0
    if pcid == 0:
        if int(t) % saveint == 0:
            # plot extracellular concentrations averaged over depth every 100ms
            saveconc()
    if (int(t) % ssint == 0) and (t - lastss) > ssint or (cfg.duration - t) < 1:
        runSS()
        lastss = t
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
    networkStatsFromSim(sim, filename=os.path.join(outdir, "netstats.json"))
sim.saveData()
sim.analysis.plotData()

if pcid == 0:
    progress_bar(cfg.duration)
    fout.close()
    for lab in rec_cells:
        with open(os.path.join(outdir, f"recs_{lab}.pkl"), "wb") as fout:
            pickle.dump(rec_cells[lab], fout)
    print("\nSimulation complete. Plotting membrane potentials")

with open(os.path.join(outdir, "centermembrane_potential_%i.pkl" % pcid), "wb") as pout:
    pickle.dump([rec_cells, pos, pops], pout)


# v0.0 - direct copy from ../uniformdensity/init.py
# v1.0 - added in o2 sources based on capillaries identified from histology
# v1.1 - set pas.e to maintain RMP and move restore state function
