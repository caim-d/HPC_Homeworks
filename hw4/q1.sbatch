#!/bin/bash

#SBATCH --job-name=pingpong
#SBATCH --nodes=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB 
#SBATCH --time=01:00:00 
#SBATCH --mail-type=END
#SBATCH --mail-user=cmd781@nyu.edu
#SBATCH --output=slurm_%j.out

module purge 
module load  amber/openmpi/intel/20.06 

mpiexec -n 4 ./pingpong 0 2
