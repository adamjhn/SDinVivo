from PIL import Image 
import numpy as np 
import numpy as np
from neuron import h, rxd 
from neuron.units import sec, mM
import sys
import os 
from matplotlib import pyplot as plt

im = Image.open('test_mask.tif')
imarray = np.array(im).transpose()
ones = np.argwhere(imarray == 0)
zs = np.argwhere(imarray == 255)
for z in zs:
    imarray[z[0],z[1]] = 0
for one in ones:
    imarray[one[0],one[1]] = 1

dx = 25 
px = 2.6270
binsz = int(100/px) #int(dx/px)

xdim = imarray.shape[1] * px
ydim = imarray.shape[0] * px 

rstart = 0
cstart = 0

mask = []
while rstart+binsz < imarray.shape[0]:
    row = []
    while cstart+binsz < imarray.shape[1]:
        row.append(np.sum(imarray[rstart:rstart+binsz, cstart:cstart+binsz]))
        cstart = cstart + binsz 
    mask.append(row)
    rstart = rstart + binsz + 1
    cstart = 0
mask = np.array(mask)
mask[:,0] = 0
mask[0,:] = 0

xbins = np.linspace(-xdim/2, xdim/2, mask.shape[1], endpoint=True)
ybins = np.linspace(-ydim/2, ydim/2, mask.shape[0], endpoint=True)

#------------------------------------------------------------------------------------------
pc = h.ParallelContext()
pcid = pc.id()
nhost = pc.nhost()
pc.timeout(0)
pc.set_maxstep(100) # required when using multiple processes


rxd.nthread(6)  # set the number of rxd threads - original 4
rxd.options.enable.extracellular = True # enable extracellular rxd

h.load_file('stdrun.hoc')
h.celsius = 37
# h.dt = 0.025
h.dt = 0.1 # original above, going for longer runs

def progress_bar(tstop, size=40):
    """ report progress of the simulation """
    prog = h.t/float(tstop)
    fill = int(size*prog)
    empt = size - fill
    progress = '#' * fill + '-' * empt
    sys.stdout.write('[%s] %2.1f%% %6.1fms of %6.1fms\r' % (progress, 100*prog, h.t, tstop))
    sys.stdout.flush()

epsilon_o2 = 0.17/sec
outdir = 'cap_test/'
try:
    os.makedirs(outdir)
except:
    pass
x = [-xdim/2, xdim/2]
y = [-ydim/2, ydim/2]
z = [-1000, 1000]
ecs_o2 = rxd.Extracellular(x[0], y[0], z[0], x[1], y[1], z[1], dx=100,
                        volume_fraction=1.0, tortuosity=1.0)

### species 
# o2_extracellular = rxd.Species([ecs_o2], name='o2', d=3.3, initial= lambda nd: 100 if
#                 isinstance(nd, rxd.node.Node1D) else ( 0.1
#                 if (nd.x3d**2 + nd.y3d**2) <= 4000**2 else 0.04), ecs_boundary_conditions=None) # changed for separate ecs for o2 
o2_extracellular = rxd.Species([ecs_o2], name='o2', d=3.3, initial= lambda nd: 100 if
                isinstance(nd, rxd.node.Node1D) else (0.1 if mask[np.argmin((ybins-nd.y3d)**2)][np.argmin((xbins-nd.x3d)**2)] else 0.04), ecs_boundary_conditions=0.04) # changed for separate ecs for o2 
# o2_extracellular = rxd.Species([ecs_o2], name='o2', d=3.3, initial= lambda nd: 100 if
#                 isinstance(nd, rxd.node.Node1D) else ( 100
#                 if mask[np.argmin((ybins-nd.y3d)**2)][np.argmin(xbins-nd.x3d)**2] > 0 else 0.1), ecs_boundary_conditions=0.1) # changed for separate ecs for o2 
# o2_extracellular = rxd.Species([ecs_o2], name='o2', d=3.3, initial= lambda nd: 100 if
#                 isinstance(nd, rxd.node.Node1D) else (mask[np.argmin((xbins-nd.x3d)**2)][np.argmin(ybins-nd.y3d)**2]), ecs_boundary_conditions=0.1) # changed for separate ecs for o2 
o2ecs = o2_extracellular[ecs_o2]

# core conditions
def cap(nd):
    if mask[np.argmin((ybins-nd.y3d)**2)][np.argmin((xbins-nd.x3d)**2)] > 0:
        return 1.0
    return 0.0

def core(nd):
    if nd.x3d**2 + nd.y3d**2 + nd.z3d**2 <= 400**2:
        return 1.0
    return 0.0
    
def bc(nd):
    if (abs(nd.x3d - ecs_o2._xlo) < ecs_o2._dx[0] or
        abs(nd.x3d - ecs_o2._xhi) < ecs_o2._dx[0] or
        abs(nd.y3d - ecs_o2._ylo) < ecs_o2._dx[1] or
        abs(nd.y3d - ecs_o2._yhi) < ecs_o2._dx[1] or
        abs(nd.z3d - ecs_o2._zlo) < ecs_o2._dx[2] or
        abs(nd.z3d - ecs_o2._zhi) < ecs_o2._dx[2]):
        return 1.0
    return 0.0

ecsbc = rxd.Parameter([ecs_o2], name='ecsbc', value = lambda nd: bc(nd))

# iscore = rxd.Parameter([ecs_o2], name='iscore', value = lambda nd: core(nd))

hascap = rxd.Parameter([ecs_o2], name='hascap', value = lambda nd: cap(nd))

o2_source = rxd.Rate(o2ecs, hascap*(epsilon_o2 * (1.0 - o2ecs)))
# o2_source = rxd.Rate(o2ecs, hascap*10)
o2diff = rxd.Rate(o2ecs, ecsbc*(epsilon_o2 * (0.04 - o2ecs))) 

# for nd in o2ecs.nodes:
#     carg = np.argmin((xbins-nd.x3d)**2)
#     rarg = np.argmin((ybins-nd.y3d)**2)
#     if mask[rarg][carg]:
#         nd.value = 0.1

def plot_image_data(data, min_val, max_val, filename, title):
    """Plot a 2d image of the data"""
    # sb = scalebar.ScaleBar(1e-6)
    # sb.location='lower left'
    plt.imshow(data, extent=o2ecs.extent('xy'), vmin=min_val,
                  vmax=max_val, interpolation='nearest', origin='lower')
    plt.colorbar()
    # sb = scalebar.ScaleBar(1e-6)
    # sb.location='lower left'
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    # ax.add_artist(sb)
    plt.title(title)
    plt.xlim(o2ecs.extent('x'))
    plt.ylim(o2ecs.extent('y'))
    plt.savefig(os.path.join(outdir,filename))
    plt.close()

def run(tstop):
    last_print = 0
    time = []
    saveint = 100
    while h.t < tstop:
        time.append(h.t)
        if int(h.t) % saveint == 0 and pcid == 0:
            np.save(os.path.join(outdir,'o2_%i.npy' % int(h.t)), o2ecs.states3d)
            plot_image_data(o2ecs.states3d.mean(2), 0.0, 0.15, 
                                'o2_mean_%05d' % int(h.t/100),
                                'Oxygen concentration; t = %6.0fms'
                                % h.t)
        if pcid == 0: progress_bar(tstop)
        pc.psolve(pc.t(0)+h.dt)

h.finitialize(-70)

run(500)

# plt.imshow(mask)
# plt.ion()
# plt.show()

# v0.0 - generates ecs filled with o2 sources based on capillary labelled histology image