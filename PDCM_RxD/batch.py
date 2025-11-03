import sys, os
import numpy as np
from netpyne import specs
from netpyne.batch import Batch

def batch():
        params = specs.ODict()
        params['k0Layer'] = [2, 4, 5, 6]  # each layer
        params['k0'] = [100, 1000]      # bolus of K+
        b = Batch(params=params, cfgFile='cfgMidOx.py', netParamsFile='netParamsMidOx.py')
        b.batchLabel = 'SDthresh'
        b.saveFolder = f"/tmp/{b.batchLabel}"
        b.method = 'grid'
        b.runCfg = {'type': 'mpi_bulletin', 'script': 'initMidOx.py', 'skip': True}
        b.run()

batch()
