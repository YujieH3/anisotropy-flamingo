#!/bin/bash -l

#SBATCH --ntasks 118           # The number of cores you need...
#SBATCH -J h0_anisotropy_bsns     #Give it something meaningful.
#SBATCH -o /cosma8/data/do012/dc-he4/log/standard_output_file.%J.out  # J is the job ID, %J is unique for each job.
#SBATCH -e /cosma8/data/do012/dc-he4/log/standard_error_file.%J.err
#SBATCH -p cosma-analyse #or some other partition, e.g. cosma, cosma8, etc.
#SBATCH -A do012 #e.g. dp004
#SBATCH -t 24:00:00  #D-HH:MM:SS
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90,TIME_LIMIT
#SBATCH --mail-user=yujiehe@strw.leidenuniv.nl #PLEASE PUT YOUR EMAIL ADDRESS HERE (without the <>)

module purge

# module load intel_comp
# module load compiler-rt tbb compiler mpi
# module load openmpi

conda deactivate

conda activate halo-cosma


# config
n=118         #number of cores
N=1728      #total number of lightcones
data_dir="/cosma8/data/do012/dc-he4/mock_lightcones_copy"  #directory of halo_properties_in_ligthcone0000.hdf5 (or 0001, 0002, etc.)
analyse_dir="/cosma8/data/do012/dc-he4/analysis"           #directory of analysis results
soap_dir="/cosma8/data/dp004/flamingo/Runs/L2800N5040/HYDRO_FIDUCIAL/SOAP"

# make output directory if doesn't exist
mkdir $analyse_dir -p

cd /cosma/home/do012/dc-he4/anisotropy-flamingo/src/h0_anisotropy_direct_compare
# run analysis
for i in $(seq 0 $((N-1)))
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

    output="${analyse_dir}/lc${lc}"
    best_fit="${output_dir}/scan_best_fit_${scaling_relation}_theta${cone_size}.csv"
    if ! [ -f "${output}/scan-bootstrap-near-scan.done" ] && [ -f "${output}/fit-all.done" ] && [ -f "${output}/scan-best-fit.done" ] #use a file flag
    then
        python scan-bootstrap-near-scan.py -i $input -r "${output}/fit_all.csv" -o $output -b $output -t $n -n 500 --overwrite && echo > "${output}/scan-bootstrap-near-scan.done"
    else
        echo "scan-bootstrap-near-scan already done or fit_all output not found or best_fit output not found, skipping..."
    fi

done
    
    

