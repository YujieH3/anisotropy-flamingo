#!/bin/bash -l

#SBATCH --ntasks 9 # The number of cores you need...
#SBATCH -J make_lightcones #Give it something meaningful.
#SBATCH -o /cosma8/data/do012/dc-he4/log/standard_output_file.%J.out  # J is the job ID, %J is unique for each job.
#SBATCH -e /cosma8/data/do012/dc-he4/log/standard_error_file.%J.err
#SBATCH -p cosma-analyse #or some other partition, e.g. cosma, cosma8, etc.
#SBATCH -A do012 #e.g. dp004
#SBATCH -t 24:00:00  #D-HH:MM:SS
##SBATCH --exclusive
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90,TIME_LIMIT
#SBATCH --mail-user=yujiehe@strw.leidenuniv.nl #PLEASE PUT YOUR EMAIL ADDRESS HERE (without the <>)

module purge

module load intel_comp
module load compiler-rt tbb compiler mpi
module load openmpi

conda activate halo-cosma

soap_dir="/cosma8/data/dp004/flamingo/Runs/L2800N5040/HYDRO_FIDUCIAL/SOAP"
output_dir="/cosma8/data/do012/dc-he4/mock_lightcones"

time mpiexec -n 9 python /cosma/home/do012/dc-he4/anisotropy-flamingo/src/__fast_make_lightcone_mpi.py -i $soap_dir -o $output_dir -z 0.35 -N 12 -L 2800