#!/bin/bash -l

#SBATCH --ntasks 112          # The number of cores you need...
#SBATCH -J h0_fit_all_mc_scatter     #Give it something meaningful.
#SBATCH -o /cosma8/data/do012/dc-he4/log/standard_output_file.%J.out  # J is the job ID, %J is unique for each job.
#SBATCH -e /cosma8/data/do012/dc-he4/log/standard_error_file.%J.err
#SBATCH -p cosma-analyse #or some other partition, e.g. cosma, cosma8, etc.
#SBATCH -A do012 #e.g. dp004
#SBATCH -t 24:00:00  #D-HH:MM:SS
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yujiehe@strw.leidenuniv.nl #PLEASE PUT YOUR EMAIL ADDRESS HERE (without the <>)
##SBATCH --exclusive #no don't need exclusive

module purge

# module load intel_comp
# module load compiler-rt tbb compiler mpi
# module load openmpi

conda activate halo-cosma


# config
n=112         #number of cores
N=1728      #total number of lightcones
data_dir="/cosma8/data/do012/dc-he4/mock_lightcones_copy"  #directory of halo_properties_in_ligthcone0000.hdf5 (or 0001, 0002, etc.)
analyse_dir="/cosma8/data/do012/dc-he4/analysis"           #directory of analysis results

# make output directory if doesn't exist
mkdir $analyse_dir -p

cd /cosma/home/do012/dc-he4/anisotropy-flamingo/src/h0_anisotropy_direct_compare_scatter
# run analysis
for i in $(seq 0 $((N-1)))
do
    lc=$(printf "%04d" $i)

    # the halo lightcone input file
    input="${analyse_dir}/lc${lc}/samples_in_lightcone${lc}_duplicate_excision_outlier_excision_scatter_assigned.csv"

    # check if the sample file exists
    if [ -f $input ]; then
        echo "File $input found."
    else
        continue
    fi

    output="${analyse_dir}/lc${lc}" # BE CAREFUL! THE OUTPUT FILENAME IS EDITED IN THE PYTHON FILE
    if [ -f "${output}/fit-all-mc-scatter.done" ] #use a file flag
    then
        continue
    else
        python fit-all-mc.py -i $input -o $output -t $n -n 150 --overwrite && echo > "${output}/fit-all-mc-scatter.done"
    fi

done

    
    

