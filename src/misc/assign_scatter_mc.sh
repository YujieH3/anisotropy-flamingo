#!/bin/bash -l

#SBATCH --ntasks 1 # The number of cores you need...
#SBATCH -J assign_scatter_mc #Give it something meaningful.
#SBATCH -o /cosma8/data/do012/dc-he4/log/standard_output_file.%J.out  # J is the job ID, %J is unique for each job.
#SBATCH -e /cosma8/data/do012/dc-he4/log/standard_error_file.%J.err
#SBATCH -p cosma-analyse #or some other partition, e.g. cosma, cosma8, etc.
#SBATCH -A do012 #e.g. dp004
#SBATCH -t 24:00:00  #D-HH:MM:SS
##SBATCH --exclusive
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yujiehe@strw.leidenuniv.nl #PLEASE PUT YOUR EMAIL ADDRESS HERE (without the <>)

module purge

# module load intel_comp
# module load compiler-rt tbb compiler mpi
# module load openmpi

conda activate halo-cosma


# config
data_dir="/cosma8/data/do012/dc-he4/mock_lightcones_copy"  #directory of halo_properties_in_ligthcone0000.hdf5 (or 0001, 0002, etc.)
analyse_dir="/cosma8/data/do012/dc-he4/analysis"           #directory of analysis results

for i in $(seq 0 500)
do
    lc=$(printf "%04d" $i)
    echo "Analysing lightcone${lc}"

    # The halo lightcone input file
    input="${analyse_dir}/lc${lc}/samples_in_lightcone${lc}_duplicate_excision_outlier_excision.csv"
    output="${analyse_dir}/lc${lc}/samples_in_lightcone${lc}_duplicate_excision_outlier_excision_scatter_assigned_mc.csv"
    fitall="${analyse_dir}/lc${lc}/fit_all_mc.csv"

    # Check if the sample file exists
    if [ -f $input ]; then
        echo "File $input found."
    else
        continue
    fi

    # Do not run if output exists
    if [ -f $output ]
    then
        continue
    else
        python /cosma/home/do012/dc-he4/anisotropy-flamingo/src/misc/assign_scatter_mc.py -i $input -o $output -f $fitall
    fi

done
