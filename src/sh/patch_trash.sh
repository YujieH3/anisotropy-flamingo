#!/bin/bash -l

#SBATCH --ntasks 1 # The number of cores you need...
#SBATCH -J patch_sort #Give it something meaningful.
#SBATCH -o /cosma8/data/do012/dc-he4/log/standard_output_file.%J.out  # J is the job ID, %J is unique for each job.
#SBATCH -e /cosma8/data/do012/dc-he4/log/standard_error_file.%J.err
#SBATCH -p cosma-analyse #or some other partition, e.g. cosma, cosma8, etc.
#SBATCH -A do012 #e.g. dp004
#SBATCH -t 24:00:00  #D-HH:MM:SS
##SBATCH --exclusive
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90,TIME_LIMIT
#SBATCH --mail-user=yujiehe@strw.leidenuniv.nl #PLEASE PUT YOUR EMAIL ADDRESS HERE (without the <>)

module purge

# module load intel_comp
# module load compiler-rt tbb compiler mpi
# module load openmpi

conda activate halo-cosma


# config
N3=1728      #total number of lightcones
analyse_dir="/cosma8/data/do012/dc-he4/analysis"           #directory of analysis results
trash_dir="/cosma8/data/do012/dc-he4/trash"           #directory of analysis results

# make output directory if doesn't exist
mkdir $analyse_dir -p

# run analysis, in src directory
# cd /cosma/home/dp004/dc-he4/anisotropy-flamingo/src
for i in $(seq 0 $((N3-1)))
do
    lc=$(printf "%04d" $i)
    #echo "Analysing lightcone${lc}"

    mkdir "${trash_dir}/lc${lc}" -p
    trash_dir_lc="${trash_dir}/lc${lc}/"
    
    cd "${analyse_dir}/lc${lc}"
    
    mv bootstrap* $trash_dir_lc
    mv fit* $trash_dir_lc
    mv h0* $trash_dir_lc
    mv scan* $trash_dir_lc

done

cd /cosma/home/do012/dc-he4/anisotropy-flamingo/src/sh
