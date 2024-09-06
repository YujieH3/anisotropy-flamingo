#!/bin/bash
tree="/cosma8/data/dp004/jch/FLAMINGO/MergerTrees/ScienceRuns/L2800N5040/HYDRO_FIDUCIAL/trees_f0.1_min10_max100/vr_trees.hdf5"
soap="/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/SOAP/"
halo_crossing="/cosma8/data/do012/dc-he4/halo_crossing.hdf5"
catalogue="/cosma8/data/do012/dc-he4/halo_properties_in_lightcones.hdf5"

time python _a_halo_crossing.py -i $tree -o $halo_crossing -z 0.3 -N 20 -L 2800
time python _b_fetch_soap.py -i $halo_crossing -s $soap
time python _c_interpolate_lightcone.py -i $halo_crossing -o $catalogue
time python _d_combine_lightcone.py -i $catalogue