#!/bin/bash
# this script runs all analysis on all lightcones

# activate conda environment
# conda activate /data1/yujiehe/conda-env/halo

parent_dir="/data1/yujiehe/data/mock_lightcone/lightcones"

# make directory -p doesn't raise error if directory exists
mkdir $parent_dir -p

N3=1000
N=10

for lc in $(seq 0 $((N3-1)))
do
    formatted_lc=$(printf "%04d" $lc)
    echo $formatted_lc
    
    # make directories
    mkdir "${parent_dir}/lc${formatted_lc}" -p

    # create samples
    input="/data1/yujiehe/data/mock_lightcone/halo_lightcone_catalogue/halo_properties_in_lightcones.hdf5"
    python 1make-samples.py -i input -o "${parent_dir}/lc${formatted_lc}/samples_in_lightcone${formatted_lc}" -n $lc
    echo 1make-samples.py done for lightcone $lc

    # remove outliers
    input="${parent_dir}/lc${formatted_lc}/samples_in_lightcone${formatted_lc}.csv"
    samples="${parent_dir}/lc${formatted_lc}/samples_in_lightcone${formatted_lc}_outlier_excision.csv"
    python 2outlier-removal.py -i $input -o $samples
    echo 2outlier-removal.py done for lightcone $lc
done

for lc in $(seq 0 $((N3-1)))
do
    formatted_lc=$(printf "%04d" $lc)
    echo $formatted_lc

    sample="${parent_dir}/lc${formatted_lc}/samples_in_lightcone${formatted_lc}_outlier_excision.csv"

    # fit all clusters
    output="${parent_dir}/lc${formatted_lc}/full_sky_bootstrap"
    python 3fit-all.py -i $sample -o $output -t 8 -n 500 > "${parent_dir}/lc${formatted_lc}/fit-all-${formatted_lc}.log"

    # H0 scan: full sky
    python 4scan-best-fit.py -i $sample 

done
