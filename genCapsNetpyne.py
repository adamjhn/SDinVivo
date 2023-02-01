from PIL import Image 
import numpy as np 
import numpy as np
# from neuron import h, rxd 
from netpyne import specs, sim
from neuron.units import sec, mM
import sys
import os 
from matplotlib import pyplot as plt
import cv2 

def findCapillaries(img):
    th, threshed = cv2.threshold(img, 100, 255, 
        cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    contours = cv2.findContours(threshed, cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    centers = []
    for i in contours:
        if cv2.contourArea(i) > 1000:
            M = cv2.moments(i)
            try:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                centers.append([cx, cy])
            except:
                passes = passes + 1
    return centers

def takeStep(pos, xmax, ymax, dx=5, px=0.2627):
    samp = np.random.rand()
    if samp < 0.44:
        newpos = [pos[0], pos[1]]
    elif samp < 0.51:
        newpos = [pos[0], pos[1] + int(dx*px)]
    elif samp < 0.58:
        newpos = [pos[0], pos[1] - int(dx*px)]
    elif samp < 0.65:
        newpos = [pos[0] + int(dx*px), pos[1]]
    elif samp < 0.72:
        newpos = [pos[0] - int(dx*px), pos[1]]
    elif samp < 0.79:
        newpos = [pos[0] + int(dx*px), pos[1] + int(dx*px)]
    elif samp < 0.86:
        newpos = [pos[0] - int(dx*px), pos[1] - int(dx*px)]
    elif samp < 0.93:
        newpos = [pos[0] + int(dx*px), pos[1] - int(dx*px)]
    else:
        newpos = [pos[0] - int(dx*px), pos[1] + int(dx*px)]
    if (0 < newpos[0] < xmax) and (0 < newpos[1] < ymax):
        return newpos
    else:
        return takeStep(pos)

def extrudeCapillaries(positions, Nz, xmax, ymax, dx=5, px=0.2627):
    caps = []
    for cap in positions:
        zpos = [cap]
        for i in range(Nz):
            zpos.append(takeStep(zpos[-1], xmax, ymax))
        caps.append(zpos)
    return caps 

def mask3D(capillaries, xshape, yshape, dx=25, px=0.2627, zstack=0):
    xy_mask = np.zeros((xshape, yshape, len(capillaries[0])), dtype=np.int16)
    for cap in capillaries:
        for z in range(len(cap)):
            xy_mask[cap[z][0], cap[z][1], z] = 1
    return xy_mask

fig_file = 'test_mask.tif'
img = cv2.imread(fig_file, cv2.IMREAD_GRAYSCALE)
centers = findCapillaries(img)
capillaries = extrudeCapillaries(centers, 20, img.shape[1], img.shape[0])
mask = mask3D(capillaries, img.shape[1], img.shape[0])

def generateMask(filename, dx=25, px=0.2627, x=None, y=None):
    im = Image.open(filename)
    # imarray = np.array(im).transpose()
    imarray = np.array(im)
    ## make image array binary 
    ones = np.argwhere(imarray == 0)
    zs = np.argwhere(imarray == 255)
    for z in zs:
        imarray[z[0],z[1]] = 0
    for one in ones:
        imarray[one[0],one[1]] = 1
    
    binsz = int(dx/px) #int(dx/px)
    if not x:
        x = imarray.shape[1] * px
    if not y:
        y = imarray.shape[0] * px 
    # xdim = imarray.shape[1] * px
    # ydim = imarray.shape[0] * px 

    rstart = 0
    cstart = 0
    mask = []
    while (rstart+binsz)*px < y:
        row = []
        while (cstart+binsz)*px < x:
            row.append(np.sum(imarray[rstart:rstart+binsz, cstart:cstart+binsz]))
            cstart = cstart + binsz 
        mask.append(row)
        rstart = rstart + binsz + 1
        cstart = 0
    mask = np.array(mask)
    mask[:,0] = 0
    mask[0,:] = 0

    return mask 

def extrudeCapillaries(mask, zbins):
    pass
mask = generateMask('test_mask.tif', dx=5)
plt.ion()
plt.imshow(mask)

# xbins = np.linspace(-xdim/2, xdim/2, mask.shape[1], endpoint=True)
# ybins = np.linspace(-ydim/2, ydim/2, mask.shape[0], endpoint=True)
xbins = np.linspace(0, xdim, mask.shape[1], endpoint=True)
ybins = np.linspace(-ydim, 0, mask.shape[0], endpoint=True)

#------------------------------------------------------------------------------------------
netParams = specs.NetParams()

netParams.sizeX = xdim       # x-dimension (horizontal length) size in um
netParams.sizeY = ydim        # y-dimension (vertical height or cortical depth) size in um
netParams.sizeZ = xdim        # z-dimension (horizontal length) size in um

### constants
from neuron.units import sec, mM
import math 

e_charge =  1.60217662e-19
scale = 1e-14/e_charge
alpha = 5.3
constants = {'e_charge' : e_charge,
            'scale' : scale,
            'gnabar' : (30/1000) * scale,     # molecules/um2 ms mV ,
            'gnabar_l' : (0.0247/1000) * scale,
            'gkbar' : (25/1000) * scale,
            'gkbar_l' : (0.05/1000) * scale,
            'gclbar_l' : (0.1/1000) * scale,
            'ukcc2' : 0.3 * mM/sec ,
            'unkcc1' : 0.1 * mM/sec ,
            'alpha' : alpha,
            'epsilon_k_max' : 0.25/sec,
            'epsilon_o2' : 0.17/sec,
            'vtau' : 1/250.0,
            'g_gliamax' : 5 * mM/sec,
            'beta0' : 7.0,
            'avo' : 6.0221409*(10**23),
            'p_max' : 0.8, # * mM/sec,
            'nao_initial' : 144.0,
            'nai_initial' : 18.0,
            'gnai_initial' : 18.0,
            'gki_initial' : 80.0,
            'ko_initial' : 3.5,
            'ki_initial' : 140.0,
            'clo_initial' : 130.0,
            'cli_initial' : 6.0,
            'o2_bath' : 0.04,
            'v_initial' : -70.0,
            'r0' : 100.0, 
            'k0' : 70.0, 
            'mask' : mask,
            'xbins' : np.linspace(0, xdim, mask.shape[1], endpoint=True),
            'ybins' : np.linspace(-ydim, 0, mask.shape[0], endpoint=True)}
netParams.rxdParams['constants'] = constants

x = [0, xdim]
y = [-ydim, 0]
z = [0, 1000]
regions = {}
regions['ecs_o2'] = {'extracellular' : True, 'xlo' : x[0],
                                            'xhi' : x[1],
                                            'ylo' : y[0],
                                            'yhi' : y[1],
                                            'zlo' : z[0],
                                            'zhi' : z[1],
                                            'dx' : 100,
                                            'volume_fraction' : 1.0,
                                            'tortuosity' : 1.0}
netParams.rxdParams['regions'] = regions

### species 
species = {}
o2_init_str = 'o2_bath if isinstance(node, rxd.node.Node1D) else (0.1 if mask[numpy.argmin((ybins-node.y3d)**2)][numpy.argmin((xbins-node.x3d)**2)] else 0.04)'
species['o2_extracellular'] = {'regions' : ['ecs_o2'], 'd' : 3.3, 'initial' : o2_init_str,
                'ecs_boundary_conditions' : constants['o2_bath'], 'name' : 'o2'}
netParams.rxdParams['species'] = species

### params 
params = {}
params['ecsbc'] = {'regions' : ['ecs_o2'], 'name' : 'ecsbc', 'value' :
    '1 if (abs(node.x3d - ecs_o2._xlo) < ecs_o2._dx[0] or abs(node.x3d - ecs_o2._xhi) < ecs_o2._dx[0] or abs(node.y3d - ecs_o2._ylo) < ecs_o2._dx[1] or abs(node.y3d - ecs_o2._yhi) < ecs_o2._dx[1] or abs(node.z3d - ecs_o2._zlo) < ecs_o2._dx[2] or abs(node.z3d - ecs_o2._zhi) < ecs_o2._dx[2]) else 0'}

params['hascap'] = {'regions' : ['ecs_o2'], 'name' : 'hascap', 'value' :
    '1.0 if mask[numpy.argmin((ybins-node.y3d)**2)][numpy.argmin((xbins-node.x3d)**2)] else 0'}

netParams.rxdParams['parameters'] = params

### rates 
o2ecs = "o2_extracellular[ecs_o2]"
rates = {}
rates['o2diff'] = {'species' : o2ecs, 'regions' : ['ecs_o2'],
    'rate' : 'ecsbc * (epsilon_o2 * (o2_bath - %s))' % (o2ecs)}

rates['o2source'] = {'species' : o2ecs, 'regions' : ['ecs_o2'],
    'rate' : 'hascap * (epsilon_o2 * (1.0 - - %s))' % (o2ecs)}

netParams.rxdParams['rates'] = rates 

### config 
cfg = specs.SimConfig()       # object of class cfg to store simulation configuration
cfg.duration = 1e3        # Duration of the simulation, in ms
cfg.hParams['v_init'] = -70.0   # set v_init to -65 mV
cfg.hParams['celsius'] = 37.0
cfg.dt = 0.1 #0.025              # Internal integration timestep to use
cfg.verbose = False            # Show detailed messages 
cfg.recordStep = 1             # Step size in ms to save data (eg. V traces, LFP, etc)
cfg.filename = 'test_cap_netpyne/'   # Set file output name

cfg.sizeX = xdim #250.0 #1000
cfg.sizeY = ydim #250.0 #1000
cfg.sizeZ = 1000.0 #200.0

sim.initialize(netParams, cfg)  # create network object and set cfg and net params
sim.net.createPops()                  # instantiate network populations
sim.net.createCells()                 # instantiate network cells based on defined populations
sim.net.connectCells()                # create connections between cells based on params
sim.net.addStims()                    # add external stimulation to cells (IClamps etc)
sim.net.addRxD(nthreads=6)    # add reaction-diffusion (RxD)

from neuron import h
## parallel context 
pc = h.ParallelContext()
pcid = pc.id()
nhost = pc.nhost()
pc.timeout(0)
pc.set_maxstep(100) # required when using multiple processes


def progress_bar(tstop, size=40):
    """ report progress of the simulation """
    prog = h.t/float(tstop)
    fill = int(size*prog)
    empt = size - fill
    progress = '#' * fill + '-' * empt
    sys.stdout.write('[%s] %2.1f%% %6.1fms of %6.1fms\r' % (progress, 100*prog, h.t, tstop))
    sys.stdout.flush()

## set variables for ecs concentrations 
o2_ecs = sim.net.rxd['species']['o2_extracellular']['hObj'][sim.net.rxd['regions']['ecs_o2']['hObj']]

def plot_image_data(data, min_val, max_val, filename, title):
    """Plot a 2d image of the data"""
    # sb = scalebar.ScaleBar(1e-6)
    # sb.location='lower left'
    plt.imshow(data, extent=o2_ecs.extent('xy'), vmin=min_val,
                  vmax=max_val, interpolation='nearest', origin='lower')
    plt.colorbar()
    # sb = scalebar.ScaleBar(1e-6)
    # sb.location='lower left'
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    # ax.add_artist(sb)
    plt.title(title)
    plt.xlim(o2_ecs.extent('x'))
    plt.ylim(o2_ecs.extent('y'))
    plt.savefig(os.path.join(cfg.filename,filename))
    plt.close()

def run(tstop):
    last_print = 0
    time = []
    saveint = 100
    while h.t < tstop:
        time.append(h.t)
        if int(h.t) % saveint == 0 and pcid == 0:
            np.save(os.path.join(cfg.filename,'o2_%i.npy' % int(h.t)), o2_ecs.states3d)
            plot_image_data(o2_ecs.states3d.mean(2), 0.03, 0.1, 
                                'o2_mean_%05d' % int(h.t/100),
                                'Oxygen concentration; t = %6.0fms'
                                % h.t)
        if pcid == 0: progress_bar(tstop)
        pc.psolve(pc.t(0)+h.dt)

try:
    os.makedirs(cfg.filename)
except:
    pass

run(500)

# plt.imshow(mask)
# plt.ion()
# plt.show()

# v0.0 - generates ecs filled with o2 sources based on capillary labelled histology image