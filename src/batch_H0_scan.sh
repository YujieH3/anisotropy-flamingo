#!/bin/bash -l

#SBATCH --ntasks 1 # The number of cores you need...
#SBATCH -J H0-analysis #Give it something meaningful.
#SBATCH -o /cosma8/data/do012/dc-he4/log/standard_output_file.%J.out  # J is the job ID, %J is unique for each job.
#SBATCH -e /cosma8/data/do012/dc-he4/log/standard_error_file.%J.err
#SBATCH -p cosma-analyse #or some other partition, e.g. cosma, cosma8, etc.
#SBATCH -A do012 #e.g. dp004
#SBATCH -t 24:00:00  #D-HH:MM:SS
##SBATCH --exclusive
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80,TIME_LIMIT_90,TIME_LIMIT
#SBATCH --mail-user=yujiehe@strw.leidenuniv.nl #PLEASE PUT YOUR EMAIL ADDRESS HERE (without the <>)

module purge

module load intel_comp
module load compiler-rt tbb compiler mpi
module load openmpi

conda activate halo-cosma


# config
#N1=12         #number of lightcones in each dimension
N3=1728      #total number of lightcones
data_dir="/cosma8/data/do012/dc-he4/mock_lightcones_copy"  #directory of halo_properties_in_ligthcone0000.hdf5 (or 0001, 0002, etc.)
analyse_dir="/cosma8/data/do012/dc-he4/analysis"           #directory of analysis results
tree="/cosma8/data/dp004/jch/FLAMINGO/MergerTrees/ScienceRuns/L2800N5040/HYDRO_FIDUCIAL/trees_f0.1_min10_max100/vr_trees.hdf5"
soap_dir="/cosma8/data/dp004/flamingo/Runs/L2800N5040/HYDRO_FIDUCIAL/SOAP"

# make output directory if doesn't exist
mkdir $analyse_dir -p

# run analysis
for i in $(seq 0 $((N3-1)))
do
    lc=$(printf "%04d" $i)
    echo "Analysing lightcone${lc}"

    # the halo lightcone input file
    input="${data_dir}/halo_properties_in_lightcone${lc}.hdf5"
        
    # check if mock lightcone exists
    if [ -f $input ]; then
        echo "File $input found."
    else
        echo "File $input does not exist, skipping..."
        continue
    fi

    # skip the last file so we don't accidentally operate on the file being written, except when all lightcones are created
    last_file=$(ls -1 $data_dir | tail -n 1)
    last_file="${data_dir}/${last_file}"
    if [ $i != $((N3-1)) ] && [ $last_file == $input ] #in case the file doesn't exist or last file is incomplete
    then
        echo "File ${input} possibly incomplete, skipping..."
        continue
    fi

    # make directories
    mkdir "${analyse_dir}/lc${lc}" -p

    
done

    
    

