# ---------------------------------------------
# This script scans the full sky and fits the 
# scaling relations for each direction. No 
# bootstrapping only best fit.
#     - Cone size set to 60 if YSZ is in the 
# scaling relation
# 
# Author                       : Yujie He
# Created on (MM/YYYY)         : 03/2024
# Last Modified on (MM/YYYY)   : 09/2024
# ---------------------------------------------

import sys
sys.path.append('/cosma/home/do012/dc-he4/anisotropy-flamingo/tools')
import clusterfit as cf
import numpy as np
import os
from numba import njit, prange, set_num_threads

# ------------------------------- configuration --------------------------------
input_file = '/data1/yujiehe/data/samples-lightcone0-clean.csv'
output_dir = '/data1/yujiehe/data/fits'

n_threads = 2

one_relation = False # give the name of the relation to fit if you want to fit only one. Set to False if you want to fit all relations.
cone_size    = 60 # all angles in this scipt are in degrees unless with a _rad suffix

lon_step     = 4  # longitude step. Longitude from -180 to 180 deg
lat_step     = 2  # latitude step. Latitude from -90 to 90 deg
# Note that in our catalogue, phi_on_lc is longitude, theta_on_lc is latitude...
# might want to change this in the future...
B_step       = 0.003
logA_step    = 0.003
scat_step    = 0.005

RELATIONS = ['LX-T', 'YSZ-T', 'M-T'] # pick from 'LX-T', 'M-T', 'LX-YSZ', 'LX-M', 'YSZ-M', 'YSZ-T'
# # Set the parameter space
# FIT_RANGE = cf.FIVE_MAX_RANGE
# -------------------------- command line arguments ------------------------------

import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Scan the full sky and get the best fit parameters.")

# Add arguments
parser.add_argument('-i', '--input', type=str, help='Input file path.', default=input_file)
parser.add_argument('-o', '--output', type=str, help='Output directory.', default=output_dir)
parser.add_argument('-r', '--range_file', type=str, help='File path of 3fit-all.py output, for setting range of fitting parameters.', default=None)
parser.add_argument('-t', '--threads', type=int, help='Number of threads.', default=n_threads)
# parser.add_argument('-s', '--cone_size', type=int, help='Cone size in degrees.', default=cone_size)
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')

# Parse the arguments
args = parser.parse_args()
input_file = args.input
output_dir = args.output
n_threads = args.threads
# cone_size = args.cone_size
overwrite = args.overwrite
range_file = args.range_file

FIT_RANGE = cf.get_range(range_file, n_sigma=4)      #4 sigma range
# ------------------------------------------------------------------------------


@njit(fastmath=True, parallel=True)
def scan_anisotropy(A_arr, B_arr, scat_arr, lon_c_arr, lat_c_arr,
                    lon, lat, logY_, logX_, cone_size, lon_step, lat_step, 
                    B_min, B_max, logA_min, logA_max, scat_min, scat_max,
                    scat_step, B_step, logA_step
                    ):
    # unit conversions
    theta = cone_size # set alias
    theta_rad = theta * np.pi / 180
    lat_rad = lat * np.pi / 180 # the memory load is not very high so we can do this
    lon_rad = lon * np.pi / 180

    n_tot = len(lon)
    if len(lat) != n_tot or len(logY_) != n_tot or len(logX_) != n_tot:
        raise ValueError("Longitude, latitude, logY_, and logX_ arrays must have the same length.")
    
    for lon_c in prange(-180, 180):

        if lon_c % lon_step != 0: # numba parallel only supports step size of 1
            continue
        lon_c_rad = lon_c * np.pi / 180

        for lat_c in range(-90, 90):

            if lat_c % lat_step != 0: # if you are wondering, 0 % lat_step = 0
                continue
            lat_c_rad = lat_c * np.pi / 180

            a = np.pi / 2 - lat_c_rad # center of cone to zenith
            b = np.pi / 2 - lat_rad   # cluster to zenith
            costheta = np.cos(a)*np.cos(b) + np.sin(a)*np.sin(b)*np.cos(lon_rad - lon_c_rad) # cosθ=cosa*cosb+sina*sinb*cosA
            mask = costheta > np.cos(theta_rad)
            n_clusters = np.sum(mask)

            cone_logY_ = logY_[mask]
            cone_logX_ = logX_[mask]
            costheta = costheta[mask]

            # fit the relation
            best_fit = cf.run_fit(cone_logY_, cone_logX_,
                                logA_min  = logA_min,
                                logA_max  = logA_max,
                                B_min     = B_min,
                                B_max     = B_max,
                                scat_min  = scat_min,
                                scat_max  = scat_max,
                                scat_step = scat_step,
                                B_step    = B_step,
                                logA_step = logA_step,
                                weight    = 1 / costheta
                                )

            # calculate index without crossing reference for clean parallel
            # idx = n_lon_steps * n_lat_steps + n_lat_step
            idx = (lon_c+180)//lon_step * 180//lat_step + (lat_c+90)//lat_step

            # save the fit parameters
            lon_c_arr[idx] = lon_c
            lat_c_arr[idx] = lat_c
            A_arr[idx]     = 10**best_fit['logA']
            B_arr[idx]     = best_fit['B']
            scat_arr[idx]  = best_fit['scat']

            print(idx, 'Direction: l', lon_c, 'b', lat_c, 'Clusters:',n_clusters,'Best fit parameters:', best_fit)

    return lon_c_arr, lat_c_arr, A_arr, B_arr, scat_arr




if __name__ == '__main__':
    import pandas as pd
    import datetime

    set_num_threads(n_threads)

    cluster_data = pd.read_csv(input_file)

    for scaling_relation in RELATIONS:

        if one_relation is not False:
            if scaling_relation != one_relation:
                continue

        # Cone size set to 60 if YSZ is in the scaling relation
        if 'YSZ' in scaling_relation:
            cone_size = 60
        else:
            cone_size = 75

        # Skip if the file already exists
        output_file = f'{output_dir}/scan_best_fit_{scaling_relation}_theta{cone_size}.csv'
        if os.path.exists(output_file) and not overwrite:
            print(f'File exists: {output_file}')
            continue

        t = datetime.datetime.now()
        print(f'[{t}] Scanning full sky: {scaling_relation}')
        n_clusters = cf.CONST[scaling_relation]['N']

        _ = scaling_relation.find('-')
        Y = cluster_data[cf.COLUMNS[scaling_relation[:_  ]]][:n_clusters].values
        X = cluster_data[cf.COLUMNS[scaling_relation[_+1:]]][:n_clusters].values
        z = cluster_data['ObservedRedshift'][:n_clusters].values

        logY_ = cf.logY_(Y, z=z, relation=scaling_relation)
        logX_ = cf.logX_(X, relation=scaling_relation)

        lon = cluster_data['phi_on_lc'][:n_clusters].values
        lat = cluster_data['theta_on_lc'][:n_clusters].values

        # preallocate arrays
        n_steps = 360//lon_step * 180//lat_step
        print(f'Steps: {n_steps}')
        A_arr     = np.zeros(n_steps)
        B_arr     = np.zeros(n_steps)
        scat_arr  = np.zeros(n_steps)
        lon_c_arr = np.zeros(n_steps)
        lat_c_arr = np.zeros(n_steps)


        lon_c_arr, lat_c_arr, A_arr, B_arr, scat_arr = scan_anisotropy(
            A_arr     = A_arr,
            B_arr     = B_arr,
            scat_arr  = scat_arr,
            lon_c_arr = lon_c_arr,
            lat_c_arr = lat_c_arr,
            lon       = lon,
            lat       = lat,
            logY_     = logY_,
            logX_     = logX_,
            cone_size = cone_size,
            lon_step  = lon_step,
            lat_step  = lat_step,
            scat_step = scat_step,
            B_step    = B_step,
            logA_step = logA_step,
            **FIT_RANGE[scaling_relation],
        )

        pd.DataFrame({
            'Glon'        : lon_c_arr, # Glon/Glat means galactic longitude/latitude
            'Glat'        : lat_c_arr,
            'A'           : A_arr,
            'B'           : B_arr,
            'TotalScatter': scat_arr
            }).to_csv(output_file, index=False)
        
        t = datetime.datetime.now()
        print(f'[{t}] Scanning finishied: {output_file}')







            

            
            

            


