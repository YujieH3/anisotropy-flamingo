#!/bin/bash -l

#SBATCH --ntasks 16           # The number of cores you need...
#SBATCH -J h0_anisotropy_mc_zcos     #Give it something meaningful.
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

conda deactivate
conda activate halo-cosma


# config
n=16          #multithreading doesn't pay off much
N=1728      #total number of lightcones
data_dir="/cosma8/data/do012/dc-he4/mock_lightcones_copy"  #directory of halo_properties_in_ligthcone0000.hdf5 (or 0001, 0002, etc.)
analyse_dir="/cosma8/data/do012/dc-he4/analysis"           #directory of analysis results
#tree="/cosma8/data/dp004/jch/FLAMINGO/MergerTrees/ScienceRuns/L2800N5040/HYDRO_FIDUCIAL/trees_f0.1_min10_max100/vr_trees.hdf5"
#soap_dir="/cosma8/data/dp004/flamingo/Runs/L2800N5040/HYDRO_FIDUCIAL/SOAP"

# make output directory if doesn't exist
mkdir $analyse_dir -p

# run analysis
for i in $(seq 0 576)
do
    lc=$(printf "%04d" $i)
    # echo "Analysing lightcone${lc}"

    # the halo lightcone input file
    input="${analyse_dir}/lc${lc}/samples_in_lightcone${lc}_duplicate_excision_outlier_excision.csv"

    # check if the sample file exists
    if [ -f $input ]; then
        echo "File $input found."
    else
        continue
    fi

    # make mcmc plot directory
    chaindir="${analyse_dir}/lc${lc}/h0mc_zcos_chains"
    mkdir $chaindir -p

    output="${analyse_dir}/lc${lc}"

    if ! [ -f "${output}/h0mc_zcos.done" ] #use a file flag
    then
        python /cosma/home/do012/dc-he4/anisotropy-flamingo/src/h0_anisotropy_mcmc/h0mc_zcos.py -i $input -o "${output}/h0_mcmc_zcos.csv" -d $chaindir -n $n --overwrite && echo > "${output}/h0mc_zcos.done"
    else
        echo "h0mc_zcos already done, skipping..."
    fi
done

    
    

