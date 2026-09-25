import sys, os
import numpy as np
from netpyne import specs
from netpyne.batch import Batch

def batch():
        params = specs.ODict()
        params['k0layer'] = [2, 4, 5, 6]  # each layer
        params['k0'] = [10, 20, 30, 40, 50]
        b = Batch(params=params, cfgFile='cfgPop.py', netParamsFile='netParamsPops.py')
        b.batchLabel = 'k0layer'
        b.saveFolder = f"/tera/adam/data/{b.batchLabel}"
        b.method = 'grid'
        b.runCfg = {'type': 'mpi_direct', 
                    'script': 'initPop.py', 
                    'skip': False,
                    'cores': 4,
                    'vmem': '24G',
                    'walltime': '96:00:00'}

        b.run()

batch()
