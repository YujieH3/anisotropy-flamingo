#!/bin/bash -l

#SBATCH --ntasks 16           # The number of cores you need...
#SBATCH -J h0mc_test    #Give it something meaningful.
#SBATCH -o standard_output_file.%J.out  # J is the job ID, %J is unique for each job.
#SBATCH -e standard_error_file.%J.err
#SBATCH -p cosma-analyse #or some other partition, e.g. cosma, cosma8, etc.
#SBATCH -A do012
#SBATCH -t 1:00:00  #D-HH:MM:SS
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=yujiehe@strw.leidenuniv.nl

module purge

module load gnu_comp/14.1.0
module load openmpi

conda deactivate
conda activate halo-cosma

mpiexec -n 16 python h0mc.py /cosma/home/do012/dc-he4/anisotropy-flamingo/data/processed/catalogues/lc0000/samples_in_lightcone0000_duplicate_excision_outlier_excision.csv --sampler emcee