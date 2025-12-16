import sys, os
import numpy as np
from netpyne import specs
from netpyne.batch import Batch

def batch():
        params = specs.ODict()
        params['k0Layer'] = [2, 4, 5, 6]  # each layer
        params['k0'] = [100, 500, 1000]      # bolus of K+
        b = Batch(params=params, cfgFile='cfgMidOx.py', netParamsFile='netParamsMidOx.py')
        b.batchLabel = 'SDthreshK0'
        b.saveFolder = f"/ddn/adamjhn/data/{b.batchLabel}"
        b.method = 'grid'
        b.runCfg = {'type': 'hpc_sge', 
                    'script': 'initMidOx.py', 
                    'skip': True,
                    'cores': 8,
                    'vmem': '24G',
                    'walltime': '48:00:00'}

        b.run()

batch()
