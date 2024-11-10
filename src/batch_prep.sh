#!/bin/bash -l

#SBATCH --ntasks 1 # The number of cores you need...
#SBATCH -J prep_lightcones #Give it something meaningful.
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

source /cosma/local/anaconda3/5.2.0/etc/profile.d/conda.sh   # cron doesn't load conda by default
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
    input="${data_dir}/halo_properties_in_lightcone${lc}.hdf5"
        
    # check if mock lightcone exists
    if [ -f $input ]; then
        echo "File $input found."
    else
        continue  #don't echo anything here, else it will cramp up the log file
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

    # combine - band patch - rotate; scripts will skip automatically if already done
    python /cosma/home/do012/dc-he4/anisotropy-flamingo/src/_1_combine_lightcone.py -i $input
    python /cosma/home/do012/dc-he4/anisotropy-flamingo/src/_2_band_patch.py -i $input
    python /cosma/home/do012/dc-he4/anisotropy-flamingo/src/_3_rotate_lightcone.py -i $input

    # make samples
    output="${analyse_dir}/lc${lc}/samples_in_lightcone${lc}.csv"
    if [ -f $output ]; then
        echo "File $output exists, skipping..."
    else
        python /cosma/home/do012/dc-he4/anisotropy-flamingo/src/1make-samples.py -i $input -o $output
    fi

    # remove duplicates
    sample=$output
    output="${sample%.csv}_duplicate_excision.csv"
    if [ -f $output ]; then
        echo "File $output exists, skipping..."
    else
        python /cosma/home/do012/dc-he4/anisotropy-flamingo/src/2link-tree-remove-duplicates.py -i $sample -o $output -t $tree
    fi

    # remove outliers; sample file is small, don't worry about file size
    sample=$output
    output="${sample%.csv}_outlier_excision.csv"
    if [ -f $output ]; then
        echo "File $output exists, skipping..."
    else
        python /cosma/home/do012/dc-he4/anisotropy-flamingo/src/3remove-outliers.py -i $sample -o $output
    fi

    # sort clusters, sorting performed in place. 
    # This is actually already done in 3remove-outliers.py... but just in case,
    # if new changes are made in the pipeline.
    python /cosma/home/do012/dc-he4/anisotropy-flamingo/src/_patch_sort_clusters.py -i $output -o $output
    echo "Sorted, lightcone${lc} done."
done

    
    

