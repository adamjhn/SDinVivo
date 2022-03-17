from netpyne import sim
from netParams import netParams
from cfg import cfg
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
sim.net.addRxD(nthreads=6)    # add reaction-diffusion (RxD)
sim.setupRecording()             # setup variables to record for each cell (spikes, V traces, etc)
sim.simulate()
sim.analyze()