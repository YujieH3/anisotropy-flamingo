#!/bin/bash -l

#SBATCH --ntasks 1           # The number of cores you need...
#SBATCH -J h0_anisotropy_mc0     #Give it something meaningful.
#SBATCH -o /cosma8/data/do012/dc-he4/log/standard_output_file.%J.out  # J is the job ID, %J is unique for each job.
#SBATCH -e /cosma8/data/do012/dc-he4/log/standard_error_file.%J.err
#SBATCH -p cosma-analyse #or some other partition, e.g. cosma, cosma8, etc.
#SBATCH -A do012 #e.g. dp004
#SBATCH -t 24:00:00  #D-HH:MM:SS
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90,TIME_LIMIT
#SBATCH --mail-user=yujiehe@strw.leidenuniv.nl #PLEASE PUT YOUR EMAIL ADDRESS HERE (without the <>)
##SBATCH --exclusive #no don't need exclusive

module purge

# module load gnu_comp/14.1.0
# module load openmpi

conda deactivate
conda activate halo-cosma


# config
n=1          #multithreading doesn't pay off much
analyse_dir="/cosma/home/do012/dc-he4/anisotropy-flamingo/data/processed/analysis"           #directory of analysis results

# run analysis
for i in $(seq 0 1728)
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

    if ! [ -f "${output}/h0mc.done" ] #use a file flag
    then
        python /cosma/home/do012/dc-he4/anisotropy-flamingo/src/h0_anisotropy_mcmc/h0mc.py -i $input -o "${output}/h0_mcmc.csv" -n $n --overwrite && echo > "${output}/h0mc.done"
    else
        echo "h0mc already done, skipping..."
    fi
done

    
    

