#!/bin/bash -l

#SBATCH --ntasks 1 # The number of cores you need...
#SBATCH -J prep_lightcones #Give it something meaningful.
#SBATCH -o /cosma8/data/do012/dc-he4/log/standard_output_file.%J.out  # J is the job ID, %J is unique for each job.
#SBATCH -e /cosma8/data/do012/dc-he4/log/standard_error_file.%J.err
#SBATCH -p cosma-analyse #or some other partition, e.g. cosma, cosma8, etc.
#SBATCH -A do012 #e.g. dp004
#SBATCH -t 72:00:00  #hh:mm:ss
##SBATCH --exclusive
#SBATCH --mail-type=END # notifications for job done & fail
#SBATCH --mail-user=yujiehe@strw.leidenuniv.nl #PLEASE PUT YOUR EMAIL ADDRESS HERE (without the <>)

module purge

module load intel_comp
module load compiler-rt tbb compiler mpi
module load openmpi

conda activate halo-cosma


# config
N1=12         #number of lightcones in each dimension
N3=1728      #total number of lightcones
data_dir="/cosma8/data/do012/dc-he4/mock_lightcones_copy"  #directory of halo_properties_in_ligthcone0000.hdf5 (or 0001, 0002, etc.)
analyse_dir="/cosma8/data/do012/dc-he4/analysis"           #directory of analysis results
tree="/cosma8/data/dp004/jch/FLAMINGO/MergerTrees/ScienceRuns/L2800N5040/HYDRO_FIDUCIAL/trees_f0.1_min10_max100/vr_trees.hdf5"
soap_dir="/cosma8/data/dp004/flamingo/Runs/L2800N5040/HYDRO_FIDUCIAL/SOAP"

mkdir $analyse_dir -p

# prep all data
python _1_combine_lightcone.py -i $data_dir
python _2_band_patch.py -i $data_dir
python _3_rotate_lightcone.py -i $data_dir

# run analysis
for i in $(seq 0 $((N3-1)))
do
    lc=$(printf "%04d" $i)
    echo "Analysing lightcone${lc}"

    # make directories
    mkdir "${analyse_dir}/lc${lc}" -p

    # create samples
    input="${data_dir}/halo_properties_in_lightcone${lc}.hdf5"
    output="${analyse_dir}/lc${lc}/samples_in_lightcone${lc}.csv"
    if [ -f $output ]; then
        echo "File $output exists, skipping..."
    else
        python 1make-samples.py -i $input -o $output
    fi

    # remove duplicates
    sample=$output
    output="${sample%.csv}_duplicate_excision.csv"
    if [ -f $output ]; then
        echo "File $output exists, skipping..."
    else
        python 2link-tree-remove-duplicates.py -i $sample -o $output -t $tree
    fi

    # remove outliers; sample file is small, don't worry about file size
    sample=$output
    output="${sample%.csv}_outlier_excision.csv"
    if [ -f $output ]; then
        echo "File $output exists, skipping..."
    else
        python 3remove-outliers.py -i $sample -o $output
    fi


    
    

