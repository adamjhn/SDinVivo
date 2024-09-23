"""
batch.py

Batch simulation 
"""
import sys 
from netpyne.batch import Batch


def batchSD():
    k0Layer = [None, 2,4,5,6]
    ox = ["perfused", "hypoxic"]
    o2_bath = [0.06,0.005]  # ~36 mmHg
    o2_init = [0.04,0.005]
    alpha_ecs = [0.2, 0.07]
    tort_ecs = [1.6, 1.8]
    o2drive = [0.013, 0.013 * (1 / 6)]
    grouped = ['ox','o2_bath','o2_init', 'alpha_ecs','tort_ecs', 'o2drive']
    params = {'k0Layer':k0Layer,
              'ox':ox,
              'o2_bath':o2_bath,
              'o2_init':o2_init,
              'alpha_ecs':alpha_ecs,
              'tort_ecs':tort_ecs,
              'o2drive':o2drive
    }

    b = Batch(cfgFile='cfgMidOx.py', netParamsFile='netParamsMidOx.py', params=params, groupedParams=grouped)
    
    b.batchLabel = 'batchSD'
    b.saveFolder = '/tmp/'+b.batchLabel
    b.method = 'grid'

    b.runCfg = {'type': 'hpc_sge',
                'script': 'initMidOx.py',
                'cores': 32, 
                'walltime':'24:00:00',
            'vmem': '64G',
            'queueName': 'cpu.q',
            'pre': """
#$ -N SDsim
#$ -cwd

source ~/.bashrc
conda activate py311
"""
        }
    
    b.run()


# Main code
if __name__ == '__main__':
    batchSD()
