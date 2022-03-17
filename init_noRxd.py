from netpyne import sim
from netParams_noRxd import netParams
from cfg_noRxd import cfg
import numpy as np
import os 
import sys
import pickle
from neuron import h 
import random 
from matplotlib import pyplot as plt


# Instantiate network 
sim.initialize(netParams, cfg)  # create network object and set cfg and net params
sim.net.createPops()                  # instantiate network populations
sim.net.createCells()                 # instantiate network cells based on defined populations
sim.net.connectCells()                # create connections between cells based on params
sim.net.addStims()                    # add external stimulation to cells (IClamps etc)
# sim.net.addRxD(nthreads=6)    # add reaction-diffusion (RxD)
# sim.setupRecording()             # setup variables to record for each cell (spikes, V traces, etc)
# sim.simulate()
# sim.analyze()

pc = h.ParallelContext()
pcid = pc.id()
nhost = pc.nhost()
pc.timeout(0)
pc.set_maxstep(100) # required when using multiple processes

random.seed(pcid+120194)
all_secs = [sec for sec in h.allsec()]
cells_per_node = len(all_secs)
rec_inds = random.sample(range(cells_per_node), int(cfg.nRec / nhost))
rec_cells = [h.Vector().record(all_secs[ind](0.5)._ref_v) for ind in rec_inds]
pos = [[all_secs[ind].x3d(0), all_secs[ind].y3d(0), all_secs[ind].z3d(0)] for ind in rec_inds]
pops = [str(all_secs[ind]).split('.')[1].split('s')[0] for ind in rec_inds]

## only single core stuff
if pcid == 0:
    ## create output dir 
    if not os.path.exists(cfg.filename):
        try:
            os.makedirs(cfg.filename)
        except:
            print("Unable to create the directory %r for the data and figures"
                % cfg.filename)
            os._exit(1)

    cell_positions = [((sec.x3d(0)-cfg.sizeX/2.0)**2 + (sec.y3d(0)+cfg.sizeY/2.0)**2 + (sec.z3d(0)-cfg.sizeZ/2.0)**2)**(0.5) for sec in h.allsec()]
    t = h.Vector().record(h._ref_t)
    soma_v = []
    rpos = []
    for i in range(int(cfg.sizeX//10)):
        for r, soma in zip(cell_positions, h.allsec()):
            if (10.0*i-2.5) < r < (10.0*i+2.5):
                print(i,r)
                rpos.append((soma.x3d(0)-cfg.sizeX/2, soma.y3d(0)+cfg.sizeY/2, soma.z3d(0)-cfg.sizeZ/2))
                soma_v.append(h.Vector().record(soma(0.5)._ref_v))
    recs = {'v':soma_v, 't':t, 'pos':rpos}

def progress_bar(tstop, size=40):
    """ report progress of the simulation """
    prog = h.t/float(tstop)
    fill = int(size*prog)
    empt = size - fill
    progress = '#' * fill + '-' * empt
    sys.stdout.write('[%s] %2.1f%% %6.1fms of %6.1fms\r' % (progress, 100*prog, h.t, tstop))
    sys.stdout.flush()

def run(tstop):
    last_print = 0
    time = []
    saveint = 100
    while h.t < tstop:
        time.append(h.t)
        if pcid == 0: progress_bar(tstop)
        pc.psolve(pc.t(0)+h.dt)  # run the simulation for 1 time step
    if pcid == 0:
        progress_bar(tstop)
        fout.close()
        with open(os.path.join(cfg.filename,"recs.pkl"),'wb') as fout:
            pickle.dump(recs,fout)
        print("\nSimulation complete. Plotting membrane potentials")

    with open(os.path.join(cfg.filename,"centermembrane_potential_%i.pkl" % pcid),'wb') as pout:
        pickle.dump([rec_cells, pos, pops, time], pout)

    pc.barrier()    # wait for all processes to save

h.load_file('stdrun.hoc')
h.finitialize(cfg.hParams['v_init'])
h.celsius = cfg.hParams['celsius']
h.dt = cfg.dt

run(cfg.duration)

if pcid == 0:
    from analysis import *
    traceExamples(cfg.filename, cfg.filename + 'traces.png', iss=[0,4,8])
    plt.close()
    rasterPlot(cfg.filename, figname=cfg.filename+'raster.png')
    plt.close()
    pc.barrier()
    h.quit()