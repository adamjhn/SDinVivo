import sys 
sys.path.insert(0,'/home/ckelley/netpyne/')
# sys.path.insert(0, '/u/craig/netpyne/')
from netpyne import specs
import numpy as np
import cv2
#------------------------------------------------------------------------------
#
# SIMULATION CONFIGURATION
#
#------------------------------------------------------------------------------

# Run parameters
cfg = specs.SimConfig()       # object of class cfg to store simulation configuration
cfg.duration = 2e3        # Duration of the simulation, in ms
cfg.hParams['v_init'] = -70.0   # set v_init to -65 mV
cfg.hParams['celsius'] = 37.0
cfg.dt = 0.025 #0.025              # Internal integration timestep to use
cfg.verbose = False            # Show detailed messages 
cfg.recordStep = 1             # Step size in ms to save data (eg. V traces, LFP, etc)
exp_dir = '/expanse/lustre/scratch/ckelley/temp_project/SDinVivoData/'
cfg.filename = exp_dir + 'connected1.7e-6_o2d2_pr1_thF_13kpmm_dx50_2s//'   # Set file output name
# cfg.filename = 'Data/unconnected_poisRate_0.3_o2drive_2_500ms_v1/'   # Set file output name
cfg.printPopAvgRates = True
cfg.printRunTime = 1
cfg.Kceil = 15.0
cfg.nRec = 240
cfg.recordCellsSpikes = ['L2e', 'L2i', 'L4e', 'L4i', 'L5e', 'L5i','L6e', 'L6i'] # record only spikes of cells (not ext stims)

 # Network dimensions
cfg.fig_file = '../test_mask.tif'
img = cv2.imread(cfg.fig_file, cv2.IMREAD_GRAYSCALE) # image used for capillaries 
img = np.rot90(img, k=-1)
cfg.px = 0.2627 # side of image pixel (microns)
cfg.dx = 50 # side of ECS voxel (microns)
cfg.sizeX = 1700 #img.shape[1] * cfg.px#250.0 #1000
cfg.sizeY = (img.shape[0]-1000) * cfg.px #250.0 #1000
cfg.sizeZ = cfg.sizeX #200.0
cfg.Nz = int(cfg.sizeZ/cfg.dx)-1
cfg.density = 90000.0
cfg.Vtissue = cfg.sizeX * cfg.sizeY * cfg.sizeZ

# scaling factors 
cfg.poissonFactor = '1e-5' #'7e-7'
cfg.connFactor = '1e-7'
cfg.poissonRateFactor = 0.1
cfg.connected = True 
cfg.o2drive = '2.0'

# slice conditions 
cfg.ox = 'perfused'
if cfg.ox == 'perfused':
    cfg.o2_bath = 0.04
    cfg.alpha_ecs = 0.2 
    cfg.tort_ecs = 1.6
elif cfg.ox == 'hypoxic':
    cfg.o2_bath = 0.01
    cfg.alpha_ecs = 0.07 
    cfg.tort_ecs = 1.8

cfg.prep = 'invitro'

# neuron params 
cfg.betaNrn = 0.24
cfg.Ncell = 80000 #int(cfg.density*(cfg.sizeX*cfg.sizeY*cfg.sizeZ*1e-9)) # default 90k / mm^3
cfg.rs = ((cfg.betaNrn * cfg.Vtissue) / (2 * np.pi * cfg.Ncell)) ** (1/3)
# if cfg.density == 90000.0:
#     cfg.rs = ((cfg.betaNrn * cfg.Vtissue) / (2 * np.pi * cfg.Ncell)) ** (1/3)
# else:
#     cfg.rs = 7.52

cfg.epas = -70 # False
cfg.gpas = 0.0001
cfg.sa2v = 3.0 # False
if cfg.sa2v:
    cfg.somaR = (cfg.sa2v * cfg.rs**3 / 2.0) ** (1/2)
else:
    cfg.somaR = cfg.rs
cfg.cyt_fraction = cfg.rs**3 / cfg.somaR**3

# sd init params 
cfg.k0 = 3.5
cfg.r0 = 100.0

###########################################################
# Network Options
###########################################################

# DC=True ;  TH=False; Balanced=True   => Reproduce Figure 7 A1 and A2
# DC=False;  TH=False; Balanced=False  => Reproduce Figure 7 B1 and B2
# DC=False ; TH=False; Balanced=True   => Reproduce Figure 8 A, B, C and D
# DC=False ; TH=False; Balanced=True   and run to 60 s to => Table 6 
# DC=False ; TH=True;  Balanced=True   => Figure 10A. But I want a partial reproduce so I guess Figure 10C is not necessary

# Size of Network. Adjust this constants, please!
cfg.ScaleFactor = 1.0 #= 80.000 

# External input DC or Poisson
cfg.DC = False #True = DC // False = Poisson

# Thalamic input in 4th and 6th layer on or off
cfg.TH = False #True = on // False = off

# Balanced and Unbalanced external input as PD article
cfg.Balanced = False #False #True=Balanced // False=Unbalanced

# v0.0 - combination of cfg from ../uniformdensity and netpyne PD thalamocortical model
# v1.0 - cfg for o2 sources based on capillaries identified from histology