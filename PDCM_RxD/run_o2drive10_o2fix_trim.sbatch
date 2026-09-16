#!/bin/bash
#SBATCH --job-name=o2drive10_o2fix_trim
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=22:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=USR1@600

# Tier 3a: trimmed o2fix lineage at o2drive=10 (INTERMEDIATE delivery, 3000ms).
# Closes the Tier 2.5 gap at normal O2. Cloned from run_o2drive100_o2fix_trim.sbatch
# (correct project paths post scratch->project move). Config chain:
# cfgPopWei_o2drive1_ecs2_trim_d10.py -> cfgPopWei_o2drive1_ecs2_trim.py
# Analysis on completion: analysis_tier25_wall_confound.py (add data/hypoxic_o2drive1_ecs2_trim).

echo "=== PDCM_RxD trim x o2fix at o2drive=10 (3000ms, intermediate-delivery wall-confound test) ==="
echo "Job ID: $SLURM_JOB_ID"; echo "Node: $(hostname)"; echo "Start: $(date)"; echo ""

cd /nfs/roberts/project/pi_rm693/tys7/newton/PDCM_RxD/

module load OpenMPI/5.0.3-GCC-13.3.0
module load miniconda/24.11.3
eval "$(conda shell.bash hook)"
conda activate spatial_model

PATCHED_PREFIX=/nfs/roberts/project/pi_rm693/tys7/newton/PDCM_RxD/neuron_patches/build/install
export PATH="$PATCHED_PREFIX/bin:$PATH"
export PYTHONPATH="$PATCHED_PREFIX/lib/python:$PYTHONPATH"
export PYTHONPATH="/nfs/roberts/project/pi_rm693/tys7/newton/PDCM_RxD/nightly_test:$PYTHONPATH"
export NEURONHOME="$PATCHED_PREFIX/share/nrn"
export CXX=g++
export LD_LIBRARY_PATH="$PATCHED_PREFIX/lib:$LD_LIBRARY_PATH"

mkdir -p logs data/o2drive10_o2fix_trim graphs/o2drive10_o2fix_trim

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPI_MCA_btl="^openib"

echo "Using patched nrniv: $(which nrniv)"
srun nrniv -mpi -python run_v2_o2fix_trim.py simConfig=cfgPopWei_o2drive1_ecs2_trim_d10.py

echo ""; echo "Done: $(date)"
