# ---------------------------------------------
# This script scans the full sky and fits the 
# scaling relations for each direction. No 
# bootstrapping only best fit.
# 
# Author                       : Yujie He
# Created on (MM/YYYY)         : 03/2024
# Last Modified on (MM/YYYY)   : 11/2024
# ---------------------------------------------

import sys
import os
sys.path.append('/cosma/home/do012/dc-he4/anisotropy-flamingo/tools')
import clusterfit as cf
import numpy as np
from numba import njit, set_num_threads
import pandas as pd
import datetime

# --------------------------------CONFIGURATION---------------------------------
input_file = '/data1/yujiehe/data/samples-lightcone0-clean.csv'
output_dir = '/data1/yujiehe/data/fits'

n_threads = 24

relations = ['LX-T', 'YSZ-T', 'M-T'] # pick from 'LX-T', 'M-T', 'LX-YSZ', 'LX-M', 'YSZ-M', 'YSZ-T'
n_bootstrap = 500 # number of bootstrapping for each direction

# Set the step size or number of steps for A
n_B_steps = 200
n_logA_steps = 200

# Set the step size for the total scatter. The scatter is on the outer loop so 
# number of steps is undefinable, depends on when can chi2 reduced ~1. 
scat_step    = 0.007   # could be too large

# -------------------------COMMAND LINE ARGUMENTS-------------------------------
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Scan the sky for scaling relations with bootstrapping.")

# Add arguments
parser.add_argument('-i', '--input', type=str, help='Input file path.', default=input_file)
parser.add_argument('-o', '--output', type=str, help='Output directory.', default=output_dir)
parser.add_argument('-r', '--range_file', type=str, help='File path of 3fit-all.py output, for setting range of fitting parameters.', default=None)
parser.add_argument('-b', '--best_fit_dir', type=str, help='File directory of scan-best-fit.py', default=None)
parser.add_argument('-t', '--threads', type=int, help='Number of threads.', default=n_threads)
parser.add_argument('-n', '--bootstrap', type=int, help='Number of bootstrap steps.', default=n_bootstrap)
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')

# Parse the arguments
args = parser.parse_args()
input_file = args.input
output_dir = args.output
n_threads = args.threads
n_bootstrap = args.bootstrap
overwrite = args.overwrite
range_file = args.range_file
best_fit_dir = args.best_fit_dir

FIT_RANGE = cf.get_range(range_file, n_sigma=5, n_sigma_scat=3)      # 3 -> 5 sigma range # no wait maybe it's I shouldn't set scatter range...

# ------------------------------------------------------------------------------


@njit(fastmath=True, parallel=False) # bootstrap_fit is already parallelized, avoid nested parallel
def scan_bootstrapping(Nbootstrap : int, 
                       A_arr      : np.ndarray,
                       B_arr      : np.ndarray,
                       scat_arr   : np.ndarray,
                       lon_c_arr  : np.ndarray,
                       lat_c_arr  : np.ndarray,
                       lons_c     : np.ndarray,
                       lats_c     : np.ndarray,
                       lon        : np.ndarray,
                       lat        : np.ndarray,
                       logY_      : np.ndarray,
                       logX_      : np.ndarray,
                       cone_size  : float,
                       B_min      : float,
                       B_max      : float,
                       logA_min   : float,
                       logA_max   : float,
                       scat_min   : float,
                       scat_max   : float,
                       scat_step  : float,
                       B_step     : float,
                       logA_step  : float
                    ):
    """
    Warning
    ----
    All angle parameter input should be in degree, NOT RADIANS!
    """
    # Alias
    nb = Nbootstrap

    # Unit conversions
    cone_size_rad = cone_size * np.pi / 180
    lats_rad = lat * np.pi / 180 # the memory load is not very high so we can do this
    lons_rad = lon * np.pi / 180
    lons_c_rad = lons_c * np.pi / 180
    lats_c_rad = lats_c * np.pi / 180

    n_tot = len(lon)
    if len(lat) != n_tot or len(logY_) != n_tot or len(logX_) != n_tot:
        raise ValueError("Longitude, latitude, logY_, and logX_ arrays must have the same length.")

    # Prepare iteration
    iter_idx = 0
    for lon_c_rad, lat_c_rad in zip(lons_c_rad, lats_c_rad):
        a = np.pi / 2 - lat_c_rad # center of cone to zenith                # okay very bad idea to use plural form for array, don't ever do this again
        b = np.pi / 2 - lats_rad   # cluster to zenith
        costheta = np.cos(a)*np.cos(b) + np.sin(a)*np.sin(b)*np.cos(lons_rad - lon_c_rad) # costheta=cosa*cosb+sina*sinb*cosA

        # select the clusters inside the cone
        mask = costheta > np.cos(cone_size_rad)
        n_clusters = np.sum(mask) # number of clusters for bootstrapping
        cone_logY_ = logY_[mask]
        cone_logX_ = logX_[mask]
        costheta = costheta[mask]

        # here we use the parallel version of bootstrap_fit
        logA, B, scat = cf.bootstrap_fit(
            Nbootstrap = Nbootstrap,
            Nclusters = n_clusters,
            logY_     = cone_logY_,
            logX_     = cone_logX_,
            weight    = 1 / costheta, # inverse because weight is applied on the denominator
            logA_min  = logA_min,
            logA_max  = logA_max,
            B_min     = B_min,
            B_max     = B_max,
            scat_min  = scat_min,
            scat_max  = scat_max,
            scat_step = scat_step,
            B_step    = B_step,
            logA_step = logA_step,
            )

        # Calculate index without crossing reference for clean parallel
        # idx = n_lon_directions * n_lat_steps + n_lat_step

        # Save the fit parameters
        lon_c_arr[iter_idx*nb:(iter_idx+1)*nb] = np.repeat(lon_c_rad * 180 / np.pi, nb)
        lat_c_arr[iter_idx*nb:(iter_idx+1)*nb] = np.repeat(lat_c_rad * 180 / np.pi, nb)
        A_arr[iter_idx*nb:(iter_idx+1)*nb]     = 10**logA
        B_arr[iter_idx*nb:(iter_idx+1)*nb]     = B
        scat_arr[iter_idx*nb:(iter_idx+1)*nb]  = scat

        iter_idx += 1

        print('Direction: l', lon_c_rad * 180 / np.pi, 'b', lat_c_rad * 180 / np.pi, 'Clusters:',n_clusters)
    return lon_c_arr, lat_c_arr, A_arr, B_arr, scat_arr





# -----------------------------------MAIN---------------------------------------
if __name__ == '__main__':
    set_num_threads(n_threads)

    cluster_data = pd.read_csv(input_file)

    t00 = datetime.datetime.now()
    print(f'[{t00}] Begin scanning: {relations}.')
    print(f'Threads: {n_threads}')


    for scaling_relation in relations:
        # set the cone size
        if 'YSZ' in scaling_relation:
            cone_size = 60
        else:
            cone_size = 75

        # Skip if the output file already exists
        output_file = f'{output_dir}/scan_bootstrap_{scaling_relation}_theta{cone_size}.csv'
        if os.path.exists(output_file) and not overwrite:
            print(f'File exists and overwrite is set to False: {output_file}')
            continue
        
        # if best_fit_dir does not exists
        best_fit_file = f'{output_dir}/scan_best_fit_{scaling_relation}_theta{cone_size}.csv'
        if not os.path.exists(best_fit_file):
            raise FileNotFoundError(f'Best fit file does not exist: {best_fit_file}')
        else:
            best_fit = pd.read_csv(best_fit_file)

        # find the longitude and latitude of the largest dipole
        _A_ = best_fit['A'].values
        _lons_ = best_fit['Glon'].values
        _lats_ = best_fit['Glat'].values
        lon_max, lat_max, _ = cf.find_max_dipole(map=_A_, lons=_lons_, lats=_lats_)
        lon_min, lat_min = cf.opposite_direction(lon=lon_max, lat=lat_max)
        print(f'Maximum dipole: l={lon_max}, b={lat_max}')

        # set the sample points
        lonlatgrid_p = cf.grid_around_lonlat(center_lon=lon_max,
                                           center_lat=lat_max,
                                           tilde_lat_space=np.pi/2 - np.arange(1, 3) * np.radians(15),
                                           tilde_lon_space=np.linspace(-np.pi, np.pi, 8, endpoint=False))
        lonlatgrid_m = cf.grid_around_lonlat(center_lon=lon_min,
                                             center_lat=lat_min,
                                             tilde_lat_space=np.pi/2 - np.arange(1, 3) * np.radians(15),
                                             tilde_lon_space=np.linspace(-np.pi, np.pi, 8, endpoint=False))
        lonlatgrid = np.concatenate([lonlatgrid_p, lonlatgrid_m], axis=1)

        # set the step size for A and B if the number of steps is given
        if n_B_steps is not None: # set the step size for A and B if the number of steps is given
            B_step = (FIT_RANGE[scaling_relation]['B_max'] - FIT_RANGE[scaling_relation]['B_min']) / n_B_steps
        if n_logA_steps is not None:
            logA_step = (FIT_RANGE[scaling_relation]['logA_max'] - FIT_RANGE[scaling_relation]['logA_min']) / n_logA_steps

        # Prepare the data, convert to logX_, logY_. Requires redshift for logY_
        t0 = datetime.datetime.now()
        print(f'[{t0}] Scanning for: {scaling_relation}')
        n_clusters = cf.CONST[scaling_relation]['N']

        # load the data according to the names
        _ = scaling_relation.find('-')
        Y = cluster_data[cf.COLUMNS[scaling_relation[:_  ]]][:n_clusters].values
        X = cluster_data[cf.COLUMNS[scaling_relation[_+1:]]][:n_clusters].values
        z = cluster_data['ObservedRedshift'][:n_clusters].values

        # logX' and logY' to be fitted
        logY_ = cf.logY_(Y, z=z, relation=scaling_relation)
        logX_ = cf.logX_(X, relation=scaling_relation)

        # longitude and latitude of clusters
        lon = cluster_data['phi_on_lc'][:n_clusters].values
        lat = cluster_data['theta_on_lc'][:n_clusters].values

        # Preallocate arrays, the memory should be able to hold
        nsize = lonlatgrid.shape[1]
        A_arr     = np.zeros(nsize*n_bootstrap)
        B_arr     = np.zeros(nsize*n_bootstrap)
        scat_arr  = np.zeros(nsize*n_bootstrap)
        lon_c_arr = np.zeros(nsize*n_bootstrap)
        lat_c_arr = np.zeros(nsize*n_bootstrap)

        # sys.exit(1)                   # for testing

        # Run bootstrapping
        lon_c_arr, lat_c_arr, A_arr, B_arr, scat_arr = scan_bootstrapping(
            Nbootstrap = n_bootstrap,
            A_arr     = A_arr,
            B_arr     = B_arr,
            scat_arr  = scat_arr,
            lon_c_arr = lon_c_arr,
            lat_c_arr = lat_c_arr,
            lons_c     = lonlatgrid[0, :],
            lats_c     = lonlatgrid[1, :],
            lon       = lon,
            lat       = lat,
            logY_     = logY_,
            logX_     = logX_,
            cone_size = cone_size,
            scat_step = scat_step,
            B_step    = B_step,
            logA_step = logA_step,
            **FIT_RANGE[scaling_relation],
        )

        # Make dataframe
        df = pd.DataFrame({
            'Glon'        : lon_c_arr, # Glon/Glat means galactic longitude/latitude
            'Glat'        : lat_c_arr,
            'A'           : A_arr,
            'B'           : B_arr,
            'TotalScatter': scat_arr,
            })
        df.to_csv(output_file, index=False)

        # Output log
        t = datetime.datetime.now()
        print(f'[{t}] Scanning finishied: {output_file}')
        print(f'Time taken: {t - t0}')
 
    t = datetime.datetime.now()
    print(f'[{t}] All done!')
    print(f'Time taken: {t - t00}')