#!/bin/bash
conda activate /data1/yujiehe/conda-env/halo
time python physical-properties-correlation.py -i ../data/samples_in_lightcone0_with_trees_duplicate_excision_outlier_excision.csv -o ../data/fits/testrun/lightcone0 -s 75
time python physical-properties-correlation.py -i ../data/samples_in_lightcone1_with_trees_duplicate_excision_outlier_excision.csv -o ../data/fits/testrun/lightcone1 -s 75
