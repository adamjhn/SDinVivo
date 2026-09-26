from netpyne import sim
import numpy as np
import os
import sys
import pickle
from neuron import h
import random
from stats import networkStatsFromSim
import json
from time import time


def rand_uniform(gid, lb=0, ub=1):
    r = h.Random()
    r.Random123(gid, 1, 1)
    return r.uniform(lb, ub)


def rand_truncnorm(gid, mean=0, var=1, lb=None, ub=None):
    r = h.Random()
    r.Random123(gid, 1, 1)
    val = r.normal(mean, var)
    if lb is not None:
        val = max(lb, val)
    if ub is not None:
        val = min(val, ub)
    return val


cfg, netParams = sim.readCmdLineArgs(
    simConfigDefault="cfgPop.py", netParamsDefault="netParamsPops.py"
)
subdir = f"{cfg.ox}_{cfg.k0Layer}_{cfg.k0}_{cfg.o2drive}_gps{cfg.GliaPumpScale:.3f}_gk{cfg.gkbar_scale:.2f}_pm{cfg.pmax_scale:.2f}_kh{cfg.gliaKHalf:.1f}_o2h{cfg.gliaO2Half:.2f}"
if getattr(cfg, "scStartV", None) is not None:
    subdir += f"_sv{cfg.scStartV:.0f}"
for _p in cfg.cellPops:
    for _knob, _tag in (("gkbar", "gk"), ("pmax", "pm"), ("gnabar", "gn")):
        _v = getattr(cfg, f"sc_{_p}_{_knob}", None)
        if _v is not None:
            subdir += f"_{_p}{_tag}{_v:g}"
if getattr(cfg, "poisson_ramp_ms", 200) != 200:
    subdir += f"_rmp{cfg.poisson_ramp_ms:g}"
if getattr(cfg, "iGnaScale", 1.0) != 1.0:
    subdir += f"_ign{cfg.iGnaScale:g}"
if getattr(cfg, "poissonRateFactor", 1.0) != 1.0:
    subdir += f"_prf{cfg.poissonRateFactor:g}"
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


class EarlyAbort(Exception):
    """Raised (symmetrically on all ranks) to break out of the interval-func
    run loop when a run is committed to SD or off-target firing.
    Used during optimization."""

    pass


def fi0(cells):
    for cell in cells:
        v = rand_truncnorm(
            cell.gid,
            cfg.cellPopsInit["mean"],
            cfg.cellPopsInit["std"] ** 2,
            ub=cfg.cellPopsInit["thresh"],
        )
        for sec in cell.secs.values():
            if "hObj" in sec:
                sec["hObj"].v = v


def getV():
    """average membrane potential for each populations
    Used to set v_init distributions by population.
    """
    vinit = {}
    for label, pop in sim.net.pops.items():
        cells = sim.getCellsList(pop.cellGids)
        if len(cells) > 0 and hasattr(cells[0].secs, "soma"):
            for cell in cells:
                v = cell.secs["soma"]["hObj"].v
                if v > 30:
                    continue
                if label in vinit:
                    vinit[label].append(v)
                else:
                    vinit[label] = [v]
    vinit_mean = {}
    for pop, v in vinit.items():
        gsum = pc.allreduce(sum(v), 1)
        glen = pc.allreduce(len(v), 1)
        vinit_mean[pop] = gsum / glen if glen > 0 else 0
    return vinit_mean


def spikeRateInWindow(t0, t1):
    """Mean firing rate (Hz) across all cells over [t0, t1] ms.

    MUST be called on every rank: the cross-rank sum is a collective. spkt is
    appended in time order, so scan back from the end and stop once past t0.
    """
    n = 0
    try:
        spkt = sim.simData["spkt"]
        for i in range(len(spkt) - 1, -1, -1):
            st = spkt[i]
            if st < t0:
                break
            if st <= t1:
                n += 1
    except Exception:
        n = 0
    n = pc.allreduce(n, 1)  # sum across ranks -- collective, all ranks
    ncell = max(1, cfg.Ncell)
    dur_s = max(1e-9, (t1 - t0) / 1000.0)
    return n / (ncell * dur_s)


def checkEarlyAbort(t, maxK, rate=None):
    """Check for SD or off-target rates and abort.
    Returns (stop:int, verdict:dict|None).
    """
    if not np.isfinite(maxK):
        return 1, {
            "verdict": "nan",
            "t": t,
            "maxK": maxK,
            "reason": f"non-finite maxK={maxK} (numerical blowup)",
        }
    if t < cfg.abortWarmup:
        return 0, None
    # K+ divergence: max ECS [K+] high AND still climbing over the window
    window = [k for (tt, k) in kmax_traj if tt >= t - cfg.abortWindow]
    slope = (window[-1] - window[0]) / cfg.abortWindow if len(window) > 1 else 0.0
    if maxK >= cfg.abortKmax and slope > cfg.abortKslope:
        return 1, {
            "verdict": "SD",
            "t": t,
            "maxK": maxK,
            "slope": slope,
            "reason": f"maxK={maxK:.2f}mM >= {cfg.abortKmax} and slope={slope:.4f} > {cfg.abortKslope} mM/ms",
        }
    # optional firing-rate gates (rate precomputed collectively by the caller)
    if cfg.abortMinRate is not None or cfg.abortMaxRate is not None:
        if rate is not None:
            if cfg.abortMinRate is not None and rate < cfg.abortMinRate:
                return 1, {
                    "verdict": "rate_low",
                    "t": t,
                    "rate": rate,
                    "reason": f"rate={rate:.2f}Hz < {cfg.abortMinRate}",
                }
            if cfg.abortMaxRate is not None and rate > cfg.abortMaxRate:
                return 1, {
                    "verdict": "rate_high",
                    "t": t,
                    "rate": rate,
                    "reason": f"rate={rate:.2f}Hz > {cfg.abortMaxRate}",
                }
    return 0, None


sim.initialize(
    simConfig=cfg, netParams=netParams
)  # create network object and set cfg and net params
sim.net.createPops()  # instantiate network populations
sim.net.createCells()  # instantiate network cells based on defined populations

sim.net.connectCells()  # create connections between cells based on params
sim.net.addStims()  # add external stimulation to cells (IClamps etc)
sim.net.addRxD(nthreads=1)  # add reaction-diffusion (RxD)
"""clamps = []
for cell in sim.net.cells:
    if cell.tags['cellModel'] != "VecStim" and cell.tags['cellModel'] != "NetStim":
        vclamp = h.VClamp(cell.secs['soma']['hObj'](0.5))
        vclamp.dur[0] = 25
        vclamp.dur[1] = 0
        vclamp.dur[2] = 0
        vclamp.amp[0] = cfg.hParams["v_init"] 
        clamps.append(vclamp)
"""
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
maxK = None


def runIntervalFunc(t):
    """Write the wave_progress every 1ms"""
    global lastss, cellSDOpen, cellSDClosed, maxK
    saveint = 100  # save concentrations interval
    ssint = 1000  # save state interval
    lastss = 0
    if getattr(cfg, "v_reinit", False):
        if t > 900:
            vinit = getV()
            for pop, v in vinit.items():
                if pop in vreinit:
                    vreinit[pop].append(v)
                else:
                    vreinit[pop] = [v]
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
        kk = sim.net.rxd.species["kk"]["hObj"]
        ecs = sim.net.rxd["regions"]["ecs"]["hObj"]
        for nd in kk[ecs].nodes:
            c = nd.concentration
            r = (
                (nd.x3d - cfg.sizeX / 2.0) ** 2
                + (nd.y3d + yoff) ** 2
                + (nd.z3d - cfg.sizeZ / 2.0) ** 2
            ) ** 0.5
            if c > cfg.Kceil and r > dist:
                dist = r
            if c <= cfg.Kceil and r < dist1:
                dist1 = r
        vals = kk[ecs].nodes.value
        meanK = np.mean(vals)
        maxK = max(vals)
        fout.write("%g\t%g\t%g\t%g\t%g\n" % (h.t, dist, dist1, maxK, meanK))
        fout.flush()

    # ---- early-warning / early-abort -- use during optimization
    if getattr(cfg, "earlyAbort", False):
        stop = 0
        verdict = None
        rate = None
        if t >= cfg.abortWarmup and (
            cfg.abortMinRate is not None or cfg.abortMaxRate is not None
        ):
            rate = spikeRateInWindow(t - cfg.abortWindow, t)
        if pcid == 0:
            stop, verdict = checkEarlyAbort(t, maxK, rate)
        stop = int(pc.allreduce(stop, 2))  # max across ranks; all ranks agree
        if stop:
            if pcid == 0 and verdict is not None:
                verdict["aborted"] = True
                abort_reason = verdict["verdict"]  # "SD" | "rate_low" | "nan"
                json.dump(
                    verdict, open(os.path.join(outdir, "verdict.json"), "w"), indent=2
                )
                print(
                    "\n[earlyAbort] %s -> stopping at t=%gms" % (verdict["reason"], t)
                )
            aborted = True
            # break out of runSimWithIntervalFunc's while loop symmetrically on
            # every rank (its condition keys off h.t, so h.stoprun would spin).
            raise EarlyAbort()


try:
    sim.runSimWithIntervalFunc(1, runIntervalFunc)
except EarlyAbort:
    sim.pc.barrier()
    try:
        sim.timing("stop", "runTime")
    except Exception:
        pass

if pcid == 0 and not aborted:
    json.dump(
        {"verdict": "completed", "aborted": False, "t": h.t, "maxK": maxK},
        open(os.path.join(outdir, "verdict.json"), "w"),
        indent=2,
    )
sim.gatherData()


if pcid == 0:
    networkStatsFromSim(
        sim, filename=os.path.join(outdir, f"netstats_{cfg.duration/1000:0.2f}s.json")
    )
    if getattr(cfg, "v_reinit", False):
        vinit_mean = {p: np.mean(v) for p, v in vreinit.items()}
        json.dump(vinit_mean, open("v_initial_pop.json", "w"), indent=4)


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
