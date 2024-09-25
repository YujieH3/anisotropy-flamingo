""" This script is modified from 5scan-bootstrapping.py to calculate the sky
distribution of A with fixed B to best-fit. What I do in the code is to break loop
in the first iteration. So set min_B to the fixed value you want. """

import sys
sys.path.append('/home/yujiehe/anisotropy-flamingo')
import os
import tools.clusterfit as cf
import numpy as np
from numba import njit, prange, set_num_threads

# --------------------------------CONFIGURATION---------------------------------
input_file = '/data1/yujiehe/data/samples_in_lightcone0_with_trees_duplicate_excision_outlier_excision.csv'
output_dir = '/data1/yujiehe/data/tests/'

n_threads = 16

overwrite = False
relations = ['LX-T', 'YSZ-T', 'M-T', 'LX-YSZ', 'LX-M', 'YSZ-M'] # pick from 'LX-T', 'M-T', 'LX-YSZ', 'LX-M', 'YSZ-M', 'YSZ-T'
n_bootstrap = 500 # number of bootstrapping for each direction

cone_size    = 60 # all angles in this scipt are in degrees unless with a _rad suffix
lon_step     = 4  # longitude step. Longitude from -180 to 180 deg
lat_step     = 2  # latitude step. Latitude from -90 to 90 deg
# Note that in our catalogue, phi_on_lc is longitude, theta_on_lc is latitude...
# might want to change this in the future...

# Set the parameter space
FIT_RANGE = cf.FIVE_MAX_RANGE

# Set the step size or number of steps for B
B_step    = 0.003
n_B_steps = 150

# And A
logA_step    = 0.003
n_logA_steps = 300

# Set the step size for the total scatter. The scatter is on the outer loop so 
# number of steps is undefinable, depends on when can chi2 reduced ~1. 
scat_step    = 0.007

# -------------------------COMMAND LINE ARGUMENTS-------------------------------
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Scan the sky for scaling relations with bootstrapping.")

# Add arguments
parser.add_argument('-i', '--input', type=str, help='Input file path.', default=input_file)
parser.add_argument('-o', '--output', type=str, help='Output directory.', default=output_dir)
parser.add_argument('-t', '--threads', type=int, help='Number of threads.', default=n_threads)
parser.add_argument('-n', '--bootstrap', type=int, help='Number of bootstrap steps.', default=n_bootstrap)
parser.add_argument('-s', '--cone_size', type=int, help='Cone size in degrees.', default=cone_size)
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')

# Parse the arguments
args = parser.parse_args()
input_file = args.input
output_dir = args.output
n_threads = args.threads
n_bootstrap = args.bootstrap
cone_size = args.cone_size
overwrite = args.overwrite

# ------------------------------------------------------------------------------

@njit(fastmath=True)
def run_fit_fixed_b(logY_, logX_, B_min, B_max, logA_min, logA_max, scat_min, 
            scat_max, scat_step, B_step, logA_step, weight=np.array([1.])):

    """ Numba accelerated function to iterate through the parameter space to
    find the best fits."""
    
    Nclusters = len(logY_)
    minx2 = 1e8

    if (weight == np.array([1])).all():
        weight = np.ones(Nclusters)
    for scat in np.arange(scat_min, scat_max, scat_step):
        for B in np.arange(B_min, B_max, B_step):
            for logA in np.arange(logA_min, logA_max, logA_step):
                x2 = np.sum((logY_ - logA - B * logX_)**2 / (scat * weight)**2) 
                x2 /= (Nclusters - 3)     # chi_res^2 = chi^2 / (N - dof), degree of freedom is the number of parameters to fit
                if x2 < minx2:      # update best fit if new lowest chi2 found
                    minx2 = x2
                    params = {
                        'logA' : logA,
                        'B'    : B,
                        'scat' : scat,
                        'chi2' : minx2
                    }
            break # fixing B!

        if minx2 < 1.04: # end after iterating through A and B space
            break

    return params



@njit(fastmath=True, parallel=False)
def bootstrap_fit_non_parallel_fixed_b(Nbootstrap, 
                  logY_, logX_, Nclusters,
                  B_min, B_max, 
                  logA_min, logA_max, 
                  scat_min, scat_max, weight = None,
                  scat_step=0.007, B_step=0.001, logA_step=0.003):

    """
    The non-parallel counter-part of the `bootstrap_fit` function. Only thing
    different is the `parallel=False` argument in the @njit decorator.

    Examples
    ---
    >>> logA, B, scat = bootstrap_fit(Nbootstrap=1000, ... )

    then the bootstrapping uncertainty can be given by `np.quantile`
    e.g. 1-sigma uncertainty is given by `np.quantile(logA, [0.16, 0.84])`.
    """

    logA = np.zeros(Nbootstrap)  # logA distribution
    B    = np.zeros(Nbootstrap)  # B distribution
    scat = np.zeros(Nbootstrap)  # scatter distribution

    if len(logY_) != Nclusters or len(logX_) != Nclusters:
        raise ValueError('Length of logY_ and logX_ must be equal to Nclusters.')

    for i in prange(Nbootstrap):
        idx = np.random.choice(Nclusters, size=Nclusters, replace=True)
        bootstrap_logY_  = logY_[idx]
        bootstrap_logX_  = logX_[idx]

        if weight is None:
            bootstrap_weight = np.ones(len(idx)) # Setting to int 1 will invoke numba typing error, so we do this
        else:
            bootstrap_weight = weight[idx]

        params = run_fit_fixed_b(
            logY_          = bootstrap_logY_,
            logX_          = bootstrap_logX_,
            B_min          = B_min,     B_max    = B_max,
            logA_min       = logA_min,  logA_max = logA_max,
            scat_min       = scat_min,  scat_max = scat_max,
            scat_step      = scat_step,
            B_step         = B_step,
            logA_step      = logA_step,
            weight         = bootstrap_weight
            )
        
        logA[i] = params['logA']
        B[i]    = params['B']
        scat[i] = params['scat']

    return logA, B, scat


@njit(fastmath=True, parallel=True)
def scan_bootstrapping(Nbootstrap, A_arr, B_arr, scat_arr, lon_c_arr, 
                    lat_c_arr, lon, lat, logY_, logX_, cone_size, lon_step, 
                    lat_step, B_min, B_max, logA_min, logA_max, scat_min, 
                    scat_max, scat_step, B_step, logA_step
                    ):
    # Alias
    nb = Nbootstrap

    # Nnit conversions
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

            if lat_c % lat_step != 0:
                continue
            lat_c_rad = lat_c * np.pi / 180

            a = np.pi / 2 - lat_c_rad # center of cone to zenith
            b = np.pi / 2 - lat_rad   # cluster to zenith
            costheta = np.cos(a)*np.cos(b) + np.sin(a)*np.sin(b)*np.cos(lon_rad - lon_c_rad) # cosθ=cosa*cosb+sina*sinb*cosA
            mask = costheta > np.cos(theta_rad)

            n_clusters = np.sum(mask) # number of clusters for bootstrapping
            cone_logY_ = logY_[mask]
            cone_logX_ = logX_[mask]
            costheta = costheta[mask]

            # Fit the relation
            # Prevent nested by making a non-parallel version of the function.
            # We'd still want scan_bootstrapping to be parallel because this
            # is a cleaner parallelized loop.
            logA, B, scat = bootstrap_fit_non_parallel_fixed_b(
                Nbootstrap = Nbootstrap,
                Nclusters = n_clusters,
                logY_     = cone_logY_,
                logX_     = cone_logX_,
                weight    = 1 / costheta,
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
            idx = (lon_c+180)//lon_step * 180//lat_step + (lat_c+90)//lat_step

            # Save the fit parameters
            lon_c_arr[idx * nb:(idx+1) * nb] = np.repeat(lon_c, nb)
            lat_c_arr[idx * nb:(idx+1) * nb] = np.repeat(lat_c, nb)
            A_arr[idx * nb:(idx+1) * nb]     = 10**logA
            B_arr[idx * nb:(idx+1) * nb]     = B
            scat_arr[idx * nb:(idx+1) * nb]  = scat

            print(idx, 'Direction: l', lon_c, 'b', lat_c, 'Clusters:',n_clusters)
    return lon_c_arr, lat_c_arr, A_arr, B_arr, scat_arr





# -----------------------------------MAIN---------------------------------------
if __name__ == '__main__':
    import pandas as pd
    import datetime

    set_num_threads(n_threads)

    cluster_data = pd.read_csv(input_file)

    t00 = datetime.datetime.now()
    print(f'[{t00}] Begin scanning: {relations} in {cone_size}°.')
    print(f'Threads: {n_threads}')

    for scaling_relation in cf.CONST.keys():

        # Skip if the scaling relation is not in the list
        if scaling_relation not in relations:
            continue

        # Skip if the output file already exists
        output_file = f'{output_dir}/fixed_slope_scan_bootstrap_{scaling_relation}_θ{cone_size}.csv'
        if os.path.exists(output_file) and not overwrite:
            print(f'File exists: {output_file}')
            continue

        if n_B_steps is not None: # set the step size for A and B if the number of steps is given
            B_step = (FIT_RANGE[scaling_relation]['B_max'] - FIT_RANGE[scaling_relation]['B_min']) / n_B_steps
        if n_logA_steps is not None:
            logA_step = (FIT_RANGE[scaling_relation]['logA_max'] - FIT_RANGE[scaling_relation]['logA_min']) / n_logA_steps

        # Fix B to the best fit
        FIT_RANGE[scaling_relation]['B_min'] = cf.LC0_BEST_FIT[scaling_relation]['B']

        # Prepare the data, convert to logX_, logY_. Requires redshift for logY_
        t0 = datetime.datetime.now()
        print(f'[{t0}] Scanning full sky: {scaling_relation}')
        n_clusters = cf.CONST[scaling_relation]['N']

        _ = scaling_relation.find('-')
        Y = cluster_data[cf.COLUMNS[scaling_relation[:_  ]]][:n_clusters]
        X = cluster_data[cf.COLUMNS[scaling_relation[_+1:]]][:n_clusters]
        z = cluster_data['ObservedRedshift'][:n_clusters]

        logY_ = cf.logY_(Y, z=z, relation=scaling_relation)
        logX_ = cf.logX_(X, relation=scaling_relation)

        lon = cluster_data['phi_on_lc'][:n_clusters]
        lat = cluster_data['theta_on_lc'][:n_clusters]
        lon = np.array(lon)
        lat = np.array(lat)

        # Preallocate arrays, the memory should be able to hold
        n_directions = 360//lon_step * 180//lat_step
        n_tot = n_directions * n_bootstrap
        print(f'Direction steps: {n_directions}')
        print(f'Bootstrap steps: {n_bootstrap}')
        print(f'Total: {n_tot}')

        A_arr     = np.zeros(n_tot)
        B_arr     = np.zeros(n_tot)
        scat_arr  = np.zeros(n_tot)
        lon_c_arr = np.zeros(n_tot)
        lat_c_arr = np.zeros(n_tot)

        lon_c_arr, lat_c_arr, A_arr, B_arr, scat_arr = scan_bootstrapping(
            Nbootstrap = n_bootstrap,
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

        df = pd.DataFrame({
            'Glon'        : lon_c_arr, # Glon/Glat means galactic longitude/latitude
            'Glat'        : lat_c_arr,
            'A'           : A_arr,
            'B'           : B_arr,
            'TotalScatter': scat_arr,
            })
        df.to_csv(output_file, index=False)
        
        t = datetime.datetime.now()
        print(f'[{t}] Scanning finishied: {output_file}')
        print(f'Time taken: {t - t0}')
    
    t = datetime.datetime.now()
    print(f'[{t}] All done!')
    print(f'Time taken: {t - t00}')