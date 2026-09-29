#!/bin/bash
#SBATCH --job-name=o2drive100_o2fix_trim
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=22:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=USR1@600

# Thread #5 matched-regime test: trimmed o2fix lineage at o2drive=100, 3000ms.
# Cloned from the KNOWN-GOOD run_o2sweep_d500p0.sbatch (correct project paths post
# scratch->project move) -- NOT from run_o2drive1_ecs2_trim_smoke.sbatch, which
# still hardcodes the DEAD /nfs/roberts/scratch path (relocation trap).
# Config: cfgPopWei_o2drive1_ecs2_trim_d100.py -> ...trim.py -> ...ecs2.py
# Analysis on completion: analysis_failing_subpop.py on data/o2drive100_o2fix_trim,
# compare L2 near-wall FAILING enrichment vs untrimmed data/o2sweep_d100p0.

echo "=== PDCM_RxD trim x o2fix at o2drive=100 (3000ms, matched-regime near-wall test) ==="
echo "Job ID: $SLURM_JOB_ID"; echo "Node: $(hostname)"; echo "Start: $(date)"; echo ""

cd /nfs/roberts/project/pi_rm693/tys7/newton/PDCM_RxD/

module load OpenMPI/5.0.3-GCC-13.3.0
module load miniconda/24.11.3
eval "$(conda shell.bash hook)"
conda activate spatial_model

PATCHED_PREFIX=/nfs/roberts/project/pi_rm693/tys7/newton/PDCM_RxD/neuron_patches/build/install
export PATH="$PATCHED_PREFIX/bin:$PATH"
export PYTHONPATH="$PATCHED_PREFIX/lib/python:$PYTHONPATH"
export PYTHONPATH="/nfs/roberts/project/pi_rm693/tys7/newton/PDCM_RxD/nightly_test:$PYTHONPATH"   # nightly_test provides the `stats` module
export NEURONHOME="$PATCHED_PREFIX/share/nrn"   # RxD librxdmath path after scratch->project move
export CXX=g++                                  # rxd runtime reaction compiler
export LD_LIBRARY_PATH="$PATCHED_PREFIX/lib:$LD_LIBRARY_PATH"

mkdir -p logs data/o2drive100_o2fix_trim graphs/o2drive100_o2fix_trim

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPI_MCA_btl="^openib"

echo "Using patched nrniv: $(which nrniv)"
srun nrniv -mpi -python run_v2_o2fix_trim.py simConfig=cfgPopWei_o2drive1_ecs2_trim_d100.py

echo ""; echo "Done: $(date)"
