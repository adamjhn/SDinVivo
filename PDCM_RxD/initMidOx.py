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
random.seed(pcid + cfg.seed)


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
sim.net.connectCells()  # create connections between cells based on params
sim.net.addStims()  # add external stimulation to cells (IClamps etc)
sim.net.addRxD(nthreads=6)  # add reaction-diffusion (RxD)
fih = h.FInitializeHandler(2, lambda: fi(sim.net.cells))
sim.setupRecording()  # setup variables to record for each cell (spikes, V traces, etc)

all_secs = [sec for sec in h.allsec()]
cells_per_node = len(all_secs)
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
    if not os.path.exists(cfg.saveFolder):
        try:
            os.makedirs(cfg.saveFolder)
        except:
            print(
                "Unable to create the directory %r for the data and figures"
                % cfg.saveFolder
            )
            os._exit(1)

    ## set variables for ecs concentrations
    k_ecs = sim.net.rxd["species"]["kk"]["hObj"][sim.net.rxd["regions"]["ecs"]["hObj"]]
    na_ecs = sim.net.rxd["species"]["na"]["hObj"][sim.net.rxd["regions"]["ecs"]["hObj"]]
    cl_ecs = sim.net.rxd["species"]["cl"]["hObj"][sim.net.rxd["regions"]["ecs"]["hObj"]]
    o2_ecs = sim.net.rxd["species"]["o2_extracellular"]["hObj"][
        sim.net.rxd["regions"]["ecs_o2"]["hObj"]
    ]

    ## manually record from cells according to distance from the center of the slice
    cell_positions = [
        (
            (sec.x3d(0) - cfg.sizeX / 2.0) ** 2
            + (sec.y3d(0) + cfg.sizeY / 2.0) ** 2
            + (sec.z3d(0) - cfg.sizeZ / 2.0) ** 2
        )
        ** (0.5)
        for sec in h.allsec()
    ]
    t = h.Vector().record(h._ref_t)
    soma_v = []
    soma_ki = []
    soma_nai = []
    soma_cli = []
    soma_nao = []
    soma_ko = []
    soma_clo = []
    soma_o2 = []
    rpos = []
    cell_type = []
    for i in range(int(cfg.sizeY // 10)):
        for r, soma in zip(cell_positions, h.allsec()):
            if (10.0 * i - 2.5) < r < (10.0 * i + 2.5):
                print(i, r)
                # rpos.append((soma.x3d(0)-cfg.sizeX/2, soma.y3d(0)+cfg.sizeY/2, soma.z3d(0)-cfg.sizeZ/2))
                rpos.append((soma.x3d(0), soma.y3d(0), soma.z3d(0)))
                cell_type.append(soma.name().split(".")[1].split("s")[0])
                soma_v.append(h.Vector().record(soma(0.5)._ref_v))
                soma_nai.append(h.Vector().record(soma(0.5)._ref_nai))
                soma_ki.append(h.Vector().record(soma(0.5)._ref_ki))
                soma_cli.append(h.Vector().record(soma(0.5)._ref_cli))
                soma_nao.append(h.Vector().record(soma(0.5)._ref_nao))
                soma_ko.append(h.Vector().record(soma(0.5)._ref_ko))
                soma_clo.append(h.Vector().record(soma(0.5)._ref_clo))
                soma_o2.append(
                    h.Vector().record(
                        o2_ecs.node_by_location(
                            soma.x3d(0), soma.y3d(0), soma.z3d(0)
                        )._ref_concentration
                    )
                )
                break

    recs = {
        "v": soma_v,
        "ki": soma_ki,
        "nai": soma_nai,
        "cli": soma_cli,
        "t": t,
        "ko": soma_ko,
        "nao": soma_nao,
        "clo": soma_clo,
        "pos": rpos,
        "o2": soma_o2,
        "rad": cell_positions,
        "cell_type": cell_type,
    }

outdir = cfg.saveFolder


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
    np.save(os.path.join(cfg.saveFolder, "k_%i.npy" % int(h.t)), k_ecs.states3d)
    np.save(os.path.join(cfg.saveFolder, "na_%i.npy" % int(h.t)), na_ecs.states3d)
    np.save(os.path.join(cfg.saveFolder, "cl_%i.npy" % int(h.t)), cl_ecs.states3d)
    np.save(os.path.join(cfg.saveFolder, "o2_%i.npy" % int(h.t)), o2_ecs.states3d)


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
if pcid == 0:
    # record the wave progress
    name = ""
    fout = open(os.path.join(cfg.saveFolder, "wave_progress%s.txt" % name), "a")
lastss = 0


def runIntervalFunc(t):
    """Write the wave_progress every 1ms"""
    global lastss
    saveint = 100
    ssint = 500
    lastss = 0
    if pcid == 0:
        if int(t) % saveint == 0:
            # plot extracellular concentrations averaged over depth every 100ms
            saveconc()
        if (int(t) % ssint == 0) and (t - lastss) > 10:
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
                    + (nd.y3d - cfg.sizeY / 2.0) ** 2
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
    networkStatsFromSim(sim, filename=os.path.join(cfg.saveFolder, "netstats.json"))
sim.saveData()
sim.analysis.plotData()


if pcid == 0:
    progress_bar(cfg.duration)
    fout.close()
    with open(os.path.join(cfg.saveFolder, "recs.pkl"), "wb") as fout:
        pickle.dump(recs, fout)
    print("\nSimulation complete. Plotting membrane potentials")

with open(
    os.path.join(cfg.saveFolder, "centermembrane_potential_%i.pkl" % pcid), "wb"
) as pout:
    pickle.dump([rec_cells, pos, pops], pout)

pc.barrier()  # wait for all processes to save
runSS()

# v0.0 - direct copy from ../uniformdensity/init.py
# v1.0 - added in o2 sources based on capillaries identified from histology
# v1.1 - set pas.e to maintain RMP and move restore state function
