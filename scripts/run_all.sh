#!/bin/bash
# This script runs all analysis

# Activate conda environment
conda activate /data1/yujiehe/conda-env/halo

dir="/data1/yujiehe/data/"
file_prefix="/data1/yujiehe/data/halo_properties_in_"
file_ids="lightcone0 lightcone1"

for file_id in $file_ids
do
    input="${dir}halo_properties_in_${file_id}.hdf5"
    python 0link-tree-remove-duplicates.py -i $input -o /data1/yujiehe/data/

    input="${dir}halo_properties_in_${file_id}_with_trees_duplicate_excision.hdf5"
    python 1make-samples.py -i $input # use default output

    input="${dir}samples_in_${file_id}_with_trees_duplicate_excision.csv"
    samples="${dir}samples_in_${file_id}_with_trees_duplicate_excision_outlier_excision.csv"
    python 2find-outlier-id.py -i $input -o $samples

    input=$samples
    output="${dir}/fits/testrun/full_sky_bootstrap"
    python 3fit-all.py -i $input -o $output -t 8 -n 500 --overwrite

    # Some heavy computation
    input=$samples
    output="${dir}/fits/testrun/"
    python 4scan-best-fit.py -i $input -o $output -t 8 -s 60
    python 4scan-best-fit.py -i $input -o $output -t 8 -s 75

    # Most heavy computation
    input=$samples
    output="${dir}/fits/testrun/"
    python 5scan_bootstrapping.py -i $input -o $output -t 20 -n 500 -s 60
    python 5scan_bootstrapping.py -i $input -o $output -t 20 -n 500 -s 75

    input="${dir}/fits/testrun/"
    python 6calculate-significance-map.py -i $input #--overwrite
done
