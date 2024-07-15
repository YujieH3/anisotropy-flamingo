#!/bin/bash
conda activate /data1/yujiehe/conda-env/parallel
time mpiexec -n 17 python 7bulk-flow-model.py -i '/data1/yujiehe/data/samples_in_lightcone0_with_trees_duplicate_excision_outlier_excision.csv' -o '/data1/yujiehe/data/fits/bulk_flow_lightcone0.csv'  
time mpiexec -n 17 python 7bulk-flow-model.py -i '/data1/yujiehe/data/samples_in_lightcone1_with_trees_duplicate_excision_outlier_excision.csv' -o '/data1/yujiehe/data/fits/bulk_flow_lightcone1.csv'
# takes roughly 1*2=2 days

time mpiexec -n 17 python 8bulk-flow-bootstrap.py -i '/data1/yujiehe/data/samples_in_lightcone1_with_trees_duplicate_excision_outlier_excision.csv' -o '/data1/yujiehe/data/fits/bulk_flow_bootstrap_lightcone1.csv'  
time mpiexec -n 17 python 8bulk-flow-bootstrap.py -i '/data1/yujiehe/data/samples_in_lightcone0_with_trees_duplicate_excision_outlier_excision.csv' -o '/data1/yujiehe/data/fits/bulk_flow_bootstrap_lightcone0.csv' 
# takes roughly 3*2=6 days 