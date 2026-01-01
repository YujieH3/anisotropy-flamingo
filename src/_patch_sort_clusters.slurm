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
data_dir="/cosma8/data/do012/dc-he4/mock_lightcones_copy"  #directory of halo_properties_in_ligthcone0000.hdf5 (or 0001, 0002, etc.)
analyse_dir="/cosma8/data/do012/dc-he4/analysis"           #directory of analysis results
tree="/cosma8/data/dp004/jch/FLAMINGO/MergerTrees/ScienceRuns/L2800N5040/HYDRO_FIDUCIAL/trees_f0.1_min10_max100/vr_trees.hdf5"
soap_dir="/cosma8/data/dp004/flamingo/Runs/L2800N5040/HYDRO_FIDUCIAL/SOAP"

# make output directory if doesn't exist
mkdir $analyse_dir -p

# run analysis, in src directory
# cd /cosma/home/dp004/dc-he4/anisotropy-flamingo/src
for i in $(seq 0 $((N3-1)))
do
    lc=$(printf "%04d" $i)
    #echo "Analysing lightcone${lc}"

    # the halo lightcone input file
    input="${analyse_dir}/lc${lc}/samples_in_lightcone${lc}_duplicate_excision_outlier_excision.csv"

    # check if the sample file exists
    if [ -f $input ]; then
        echo "File $input found."
    else
        continue
    fi

    # patch
    python _patch_sort_clusters.py -i $input -o $input # inplace sorting

done
