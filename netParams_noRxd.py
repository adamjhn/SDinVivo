from netpyne import specs

from cfg_noRxd import cfg

#------------------------------------------------------------------------------
#
# NETWORK PARAMETERS
#
#------------------------------------------------------------------------------

netParams = specs.NetParams()  # object of class NetParams to store the network parameters

netParams.sizeX = cfg.sizeX        # x-dimension (horizontal length) size in um
netParams.sizeY = cfg.sizeY        # y-dimension (vertical height or cortical depth) size in um
netParams.sizeZ = cfg.sizeZ        # z-dimension (horizontal length) size in um
netParams.propVelocity = 100.0     # propagation velocity (um/ms)
netParams.defaultDelay = 2.0       # default conn delay (ms)
netParams.probLengthConst = 150.0  # length constant for conn probability (um)

#------------------------------------------------------------------------------
## Cell types
secs = {} # sections dict
secs['soma'] = {'geom': {}, 'mechs': {}}                                                # soma params dict
secs['soma']['geom'] = {'diam': 15, 'L': 14, 'Ra': 120.0, 'pt3d' : []}                               # soma geometry
secs['soma']['geom']['pt3d'].append((0,0,0,15))
secs['soma']['geom']['pt3d'].append((0,0,15,15))
secs['soma']['mechs']['hh'] = {'gnabar': 0.13, 'gkbar': 0.036, 'gl': 0.003, 'el': -70}  # soma hh mechanism
netParams.cellParams['E'] = {'secs': secs}                                              # add dict to list of cell params

secs = {} # sections dict
secs['soma'] = {'geom': {}, 'mechs': {}}                                                # soma params dict
secs['soma']['geom'] = {'diam': 15.0, 'L': 15.0, 'Ra': 110.0, 'pt3d' : []}                               # soma geometry
secs['soma']['geom']['pt3d'].append((0,0,0,15.0))
secs['soma']['geom']['pt3d'].append((0,0,15.0,15.0))                            # soma geometry
secs['soma']['mechs']['hh'] = {'gnabar': 0.11, 'gkbar': 0.036, 'gl': 0.003, 'el': -70}  # soma hh mechanism
netParams.cellParams['I'] = {'secs': secs}                                              # add dict to list of cell params

## Population parameters
netParams.popParams['E2'] = {'cellType': 'E', 'numCells': cfg.N_L23_E, 'yRange': [2 * cfg.somaR, cfg.sizeY / 3]}
netParams.popParams['I2'] = {'cellType': 'I', 'numCells': cfg.N_L23_I, 'yRange': [2 * cfg.somaR, cfg.sizeY / 3]}
netParams.popParams['E4'] = {'cellType': 'E', 'numCells': cfg.N_L4_E, 'yRange': [cfg.sizeY / 3, cfg.sizeY * (2/3)]}
netParams.popParams['I4'] = {'cellType': 'I', 'numCells': cfg.N_L4_I, 'yRange': [cfg.sizeY / 3, cfg.sizeY * (2/3)]}
netParams.popParams['E5'] = {'cellType': 'E', 'numCells': cfg.N_L5_E, 'yRange': [cfg.sizeY * (2/3), cfg.sizeY - 2*cfg.somaR]}
netParams.popParams['I5'] = {'cellType': 'I', 'numCells': cfg.N_L5_I, 'yRange': [cfg.sizeY * (2/3), cfg.sizeY - 2*cfg.somaR]}

#------------------------------------------------------------------------------
## Connectivity rules

## Synaptic mechanism parameters
netParams.synMechParams['exc'] = {'mod': 'Exp2Syn', 'tau1': 0.8, 'tau2': 5.3, 'e': 0}  # NMDA synaptic mechanism
netParams.synMechParams['inh'] = {'mod': 'Exp2Syn', 'tau1': 0.6, 'tau2': 8.5, 'e': -75}  # GABA synaptic mechanism

## Stimulation parameters
netParams.stimSourceParams['bkg'] = {'type': 'NetStim', 'rate': 20, 'noise': 0.3}
netParams.stimTargetParams['bkg->all'] = {'source': 'bkg', 'conds': {'cellType': ['E','I']}, 'weight': 0.5, 'delay': 'max(1, normal(5,2))', 'synMech': 'exc'}

# netParams.stimSourceParams['Ebkg'] = {'type': 'NetStim', 'rate': 50, 'noise': 0.3}
# netParams.stimTargetParams['bkg->E'] = {'source': 'Ebkg', 'conds': {'cellType': ['E']}, 'weight': 0.5, 'delay': 'max(1, normal(5,2))', 'synMech': 'exc'}
### weights -> 0.05 

## Connection parameters
netParams.connParams['E->all'] = {
    'preConds': {'cellType': 'E'}, 'postConds': {'cellType' : ['E']},  #  E -> all (100-1000 um)
    'probability': 0.1 ,                  # probability of connection
    'weight': '0.005',         # synaptic weight - original 0.005*post_ynorm
    'delay': 'defaultDelay+dist_3D/propVelocity',      # transmission delay (ms)
    'synMech': 'exc'}                     # synaptic mechanism

netParams.connParams['I->E'] = {
    'preConds': {'cellType': 'I'}, 'postConds': {'cellType': 'E'},   #  I -> E
    'probability': '0.4*exp(-dist_3D/probLengthConst)',              # probability of connection
    'weight': 0.001,                                                 # synaptic weight
    'delay': 'defaultDelay+dist_3D/propVelocity',                    # transmission delay (ms)
    'synMech': 'inh'}                                                # synaptic mechanism