#!/usr/bin/bash
#SBATCH --job-name=SDinVivo
#SBATCH -A mcdougal
#SBATCH --partition scavenge
#SBATCH --ntasks=24 --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=1-00:00:00 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=adam.newton@yale.edu
module load OpenMPI/4.0.5-GCC-10.2.0
source $HOME/.bashrc
cd /home/ajn48/project/SDinVivo/PDCM_RxD
time mpiexec -np 24 nrniv -mpi -python initMidOx.py
