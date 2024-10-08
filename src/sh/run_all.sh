#!/bin/bash
# this script runs all analysis on all lightcones

# activate conda environment
# conda activate /data1/yujiehe/conda-env/halo

data_dir="/data1/yujiehe/data/mock_lightcone/lightcones"

# make directory -p doesn't raise error if directory exists
mkdir $data_dir -p

N3=1728 # 1728=12*12*12
N=12

# first job: (main job) 2.8Gpc


# prep all data
python _1_combine_lightcone.py -i $data_dir
python _2_band_patch.py -i $data_dir
python _3_rotate_lightcone.py -i $data_dir

# create lightcones
for lc in $(seq 0 $((N3-1)))
do
    formatted_lc=$(printf "%04d" $lc)
    echo $formatted_lc
    
    # make directories
    mkdir "${data_dir}/lc${formatted_lc}" -p # only double quote allows formatting like ${}

    # create samples
    input="/data1/yujiehe/data/mock_lightcone/halo_lightcone_catalogue/halo_properties_in_lightcones.hdf5"
    python 1make-samples.py -i input -o "${data_dir}/lc${formatted_lc}/samples_in_lightcone${formatted_lc}.csv" -n $lc
    echo 1make-samples.py done for lightcone $lc

    # remove outliers
    input="${data_dir}/lc${formatted_lc}/samples_in_lightcone${formatted_lc}.csv"
    samples="${data_dir}/lc${formatted_lc}/samples_in_lightcone${formatted_lc}_outlier_excision.csv"
    python 2outlier-removal.py -i $input -o $samples
    echo 2outlier-removal.py done for lightcone $lc
done


# run analysis
for lc in $(seq 0 $((N3-1)))
do
    formatted_lc=$(printf "%04d" $lc)
    echo $formatted_lc

    sample="${data_dir}/lc${formatted_lc}/samples_in_lightcone${formatted_lc}_outlier_excision.csv"

    # fit all clusters
    output="${data_dir}/lc${formatted_lc}/full_sky_bootstrap"
    python 3fit-all.py -i $sample -o $output -t 8 -n 500 > "${data_dir}/lc${formatted_lc}/fit-all-${formatted_lc}.log"
    # - add a column of chandra temperature 
    # - fit twice, one for original temperature, one for chandra temperature

    # SEC1 H0 scan: best fit
    output="${data_dir}/lc${formatted_lc}"
    python 4compare-scan-best-fit.py -i $sample -o $output -t 8 -s 60
    # - use chandra temperature
    # - 60 for Ysz-T (LX-Ysz); 75 for Lx-T Mgas-T (Lx-Mgas)

    # SEC1 H0 scan: bootstrapping
    output="${data_dir}/lc${formatted_lc}"
    python 5compare-scan-bootstrapping.py -i $sample -o $output -t 16 -n 500 -s 60
    # - use chandra temperature
    # - do bootstrapping only for most extreme direction **

    # SEC1 H0 scan: significance map
    input="${data_dir}/lc${formatted_lc}"
    python 6calculate-significance-map.py -d $input
    # - use chandra temperature

    # SEC1 bulk flow. skip best fit, do bootstrapping only
    output="${data_dir}/lc${formatted_lc}"
    mpiexec -n 17 python 8compare-bulk-flow-bootstrapping.py -i $sample -o $output 
    # - use chandra temperature
    # - don't do directions # full thing for only a few (10) lightcones
    # - set the redshift spheres
    ## - add Lx-Ysz, Lx-Mgas, Ysz-Mgas relations




    # SEC2 H0 variation
    output="${data_dir}/lc${formatted_lc}"
    plotdir="${data_dir}/lc${formatted_lc}/h0mcmcplot"
    mkdir $plotdir -p
    python 9proper-H0-model-mcmc.py -i $sample -o $output -d $plotdir
    # - use original temperature
    # - run it in a separate shell. Parallelizing is probably too much trouble 
    ## - add Lx-Ysz, Lx-Mgas, Ysz-Mgas relations

    # SEC2 bulk flow
    output="${data_dir}/lc${formatted_lc}"
    plotdir="${data_dir}/lc${formatted_lc}/bfmcmcplot"
    mkdir $plotdir -p
    python 10proper-bulk-flow-model-mcmc.py -i $sample -o $output -d $plotdir
    # - use original temperature
    # - run it in a separate shell
    # - set the redshift spheres
    ## - add Lx-Ysz, Lx-Mgas, Ysz-Mgas relations

done

