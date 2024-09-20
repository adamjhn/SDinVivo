import pickle
from netpyne import specs
from netpyne.batch import Batch
import numpy as np
from cfgMidOx import cfg
from stats import networkStatsFromSimData


def batch():
    # parameters space to explore
    params = specs.ODict()
    params["excWeight"] = [5e-8, 5e-5]
    params["inhWeightScale"] = [1, 12]
    params["scaleConnWeightNetStims"] = [1e-9, 1e-5]
    params["scaleConnWeightNetStimStd"] = [1e-9, 1e-5]
    pops = {}
    count = 0
    L = ["L2e", "L2i", "L4e", "L4i", "L5e", "L5i", "L6e", "L6i"]
    for pop, nfull in zip(L, cfg.N_Full):
        n = int(nfull * cfg.ScaleFactor)
        pops[pop] = {"cellGids": [i for i in range(count, count + n)]}
        count += n
    # fitness function
    fitnessFuncArgs = {}
    fitnessFuncArgs["maxFitness"] = 1_000_000_000_000
    fitnessFuncArgs["penalty"] = 100  # score for nan results
    fitnessFuncArgs["pops"] = pops
    fitnessFuncArgs["duration"] = cfg.duration
    fitnessFuncArgs["rates"] = {
        "L2e": (4.4517376851012385, 0.09587250400159386),
        "L2i": (17.99271168274384, 0.19016893788323996),
        "L4e": (14.685795778665147, 0.20435136576837587),
        "L4i": (25.027625570776255, 0.1582611776867428),
        "L5e": (16.113015463917527, 0.9320118348433256),
        "L5i": (34.67764705882354, 0.27917079392359573),
        "L6e": (0.6606600086843248, 0.057015171715013496),
        "L6i": (30.65074309978769, 0.2896035295633283),
    }

    fitnessFuncArgs["irregularity"] = {
        "L2e": (4.337725293047559, 0.07672409603372116),
        "L2i": (3.801410035517878, 0.05816191733773294),
        "L4e": (3.6488074270926574, 0.06710335605966755),
        "L4i": (3.9592778025187547, 0.07569079213550525),
        "L5e": (3.976601552656574, 0.07792531715770835),
        "L5i": (3.8653199758671235, 0.1014409256064447),
        "L6e": (4.796650246426289, 0.40734237486204977),
        "L6i": (3.566417870413877, 0.12387776240017094),
    }

    fitnessFuncArgs["synchrony"] = {
        "L2e": (5.852835618571649, 1.1661723132756037),
        "L2i": (2.4538815177085045, 0.35593145390121905),
        "L4e": (11.885934799167812, 2.1447465471128546),
        "L4i": (2.8161919369483255, 0.413587519241556),
        "L5e": (7.867577254200212, 2.025607609142892),
        "L5i": (0.6803147426176923, 0.13028024310393632),
        "L6e": (20.398885918133935, 5.0428049341489105),
        "L6i": (1.2313477570357696, 0.10353904407754076),
    }

    fitnessFuncArgs["scale"] = {"rates": 1, "irregularity": 0, "synchrony": 0}

    def fitnessFunc(sd, **kwargs):
        print("calc fitness")

        stats = networkStatsFromSimData(
            sd, kwargs["pops"], kwargs["duration"], filename="netstats.json"
        )
        print(stats)
        score = 0
        scale = kwargs["scale"]
        for k in stats:
            for pop in kwargs[k]:
                if np.isnan(stats[k][pop]):
                    score += kwargs["penalty"]
                else:
                    score += (
                        scale[k]
                        * ((stats[k][pop] - kwargs[k][pop][0]) / kwargs[k][pop][1]) ** 2
                    )

        # check rxd variables
        rate, vscore, rxdscore, o2score, kscore = 0, 0, 0, 0, 0
        for ion in ["k", "na", "cl"]:
            for trace in sd[f"{ion}i_soma"].values():
                rxdscore += abs(trace[0] - trace[-1]) / trace[0]
                if ion == "k":
                    rxdscore += (
                        5 * abs(trace[0] - trace[-1]) / trace[0]
                        if trace[-1] > trace[0]
                        else 0
                    )
                    kscore += (
                        abs(
                            trace[-2]
                            - trace[-1]
                        )
                        / 100
                    )
            for trace in sd["dumpi_soma"].values():
                o2score += trace[-1]  # amount of oxygen consumed
            for trace in sd["v_soma"].values():
                vscore += abs(np.mean(trace) + 70)
        return min(
            kwargs["maxFitness"],
            10 * score + rxdscore + kscore + o2score + vscore,
        )

    # create Batch object with paramaters to modify, and specifying files to use
    b = Batch(params=params, cfgFile="cfgMidOx.py", netParamsFile="netParamsMidOx.py")

    # Set output folder, grid method (all param combinations), and run configuration
    b.method = "optuna"
    b.runCfg = {
        "type": "hpc_slurm",
        "script": "initMidOx.py",
        # options required only for mpi_direct or hpc
        "mpiCommand": "",
        "nodes": 1,
        "coresPerNode": 1,
        "walltime": "0-01:00:00",
        "partition": "scavenge",
        "allocation": "mcdougal",
        # "email": "adam.newton@yale.edu",
        #'reservation': None,
        "folder": "/home/ajn48/project/SDinVivo/PDCM_RxD",
        "custom": """#SBATCH --partition=scavenge
#SBATCH --requeue
#module load miniconda
#module load OpenMPI/4.0.5-GCC-10.2.0 
#source /vast/palmer/apps/avx2/software/miniconda/23.1.0/etc/profile.d/conda.sh
#conda activate py310
"""
        #'custom': 'export LD_LIBRARY_PATH="$HOME/.openmpi/lib"' # only for conda users
    }
    b.batchLabel = "paramsRateWeak"
    print(f"/vast/palmer/scratch/mcdougal/ajn48/{b.batchLabel}")
    b.saveFolder = "/vast/palmer/scratch/mcdougal/ajn48/" + b.batchLabel

    b.optimCfg = {
        "fitnessFunc": fitnessFunc,  # fitness expression (should read simData)
        "fitnessFuncArgs": fitnessFuncArgs,
        "maxFitness": fitnessFuncArgs["maxFitness"],
        "maxiters": 50000000,  #    Maximum number of iterations (1 iteration = 1 function evaluation)
        "maxtime": 16 * 60 * 60,  #    Maximum time allowed, in seconds
        "maxiter_wait": 180,
        "time_sleep": 30,
    }

    # Run batch simulations
    b.run()


# Main code
if __name__ == "__main__":
    batch()  # 'simple' or 'complex'
