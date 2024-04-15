#!/bin/bash
# This script runs all analysis

# Activate conda environment
conda activate /data1/yujiehe/conda-env/halo

dir="/data1/yujiehe/data"
file_prefix="/data1/yujiehe/data/halo_properties_in_"
names="lightcone0 lightcone1"

for name in $names
do
    input="${dir}/halo_properties_in_${name}.hdf5"
    python 0link-tree-remove-duplicates.py -i $input -o /data1/yujiehe/data/ #--overwrite
    echo 0link-tree-remove-duplicates.py done

    input="${dir}/halo_properties_in_${name}_with_trees_duplicate_excision.hdf5"
    python 1make-samples.py -i $input # use default output
    echo 1make-samples.py done

    input="${dir}/samples_in_${name}_with_trees_duplicate_excision.csv"
    samples="${dir}/samples_in_${name}_with_trees_duplicate_excision_outlier_excision.csv"
    python 2find-outlier-id.py -i $input -o $samples
    echo 2find-outlier-id.py done

    # make directory -p doesn't raise error if directory exists
    mkdir ${dir}/fits/testrun/${name} -p

    input=$samples
    output="${dir}/fits/testrun/${name}/full_sky_bootstrap"
    python 3fit-all.py -i $input -o $output -t 8 -n 500 > fit-all-${name}.log #--overwrite
    echo 3fit-all.py done

    # Some heavy computation
    input=$samples
    output="${dir}/fits/testrun/${name}"
    python 4scan-best-fit.py -i $input -o $output -t 8 -s 60 #--overwrite
    python 4scan-best-fit.py -i $input -o $output -t 8 -s 75 #--overwrite
    echo 4scan-best-fit.py done

    # Most heavy computation
    input=$samples
    output="${dir}/fits/testrun/${name}"
    python 5scan-bootstrapping.py -i $input -o $output -t 16 -n 500 -s 60 #--overwrite
    python 5scan-bootstrapping.py -i $input -o $output -t 16 -n 500 -s 75 #--overwrite
    echo 5scan-bootstrapping.py done

    input="${dir}/fits/testrun/${name}"
    python 6calculate-significance-map.py -d $input #--overwrite
done
