# mpiexec -n 17 python 7bulk-flow-model.py
# ---------------------------------------------
# This script calculates the best fit bulk flow 
# direction and amplitude for each scaling relation
# and saves the results to a csv file.
#
# Author                       : Yujie He
# Created on (MM/YYYY)         : 03/2024
# Last Modified on (MM/YYYY)   : 09/2024
# ---------------------------------------------

# -----------------------IMPORTS------------------------------------------------
import numpy as np
import pandas as pd
import os
import datetime

import sys
sys.path.append('/data1/yujiehe/anisotropy-flamingo')
import tools.constants as const
import tools.clusterfit as cf
import tools.xray_correct as xc
import tools.cosmocalc as cc

from numba import njit

from mpi4py import MPI
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=68.1, Om0=0.306)


# import astropy.coordinates as coord
# -----------------------CONFIGURATION------------------------------------------

# Input file is a halo catalog with lightcone data.
INPUT_FILE = '/data1/yujiehe/data/samples_in_lightcone0_with_trees_duplicate_excision_outlier_excision.csv'
OUTPUT_FILE = '/data1/yujiehe/data/fits/bulk_flow_lightcone0.csv'
# INPUT_FILE = '/data1/yujiehe/data/samples_in_lightcone1_with_trees_duplicate_excision_outlier_excision.csv'
# OUTPUT_FILE = '/data1/yujiehe/data/fits/bulk_flow_lightcone1.csv'
OVERWRITE = True

# Relations to fit
RELATIONS = ['LX-T', 'YSZ-T', 'M-T'] # pick from 'LX-T', 'M-T', 'LX-YSZ', 'LX-M', 'YSZ-M', 'YSZ-T'

# Amplitude range and step size
UBFMIN = 35 # ubf for bulk flow velocity
UBFMAX = 800
UBF_STEP = 15 # 800-35 = 15 * 17*3

# Longitude and latitude steps
LON_STEP = 20 # maybe change to 8. Considering we still need to bootstrap later 
LAT_STEP = 10

# Specify number of steps for each parameter 
B_NSTEPS    = 100
LOGA_NSTEPS = 100
SCAT_STEP = 0.0003

# B_STEP    = 0.009
# LOGA_STEP = 0.004

C = 299792.458                  # the speed of light in km/s
FIT_RANGE = const.ONE_MAX_RANGE_TIGHT_SCAT


# -----------------------------COMMAND LINE ARGUMENTS---------------------------

import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Calculate significance map for best fit scans.")

# Add arguments
parser.add_argument('-i', '--input', type=str, help='Input file', default=INPUT_FILE)
parser.add_argument('-o', '--output', type=str, help='Output file', default=OUTPUT_FILE)
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing.', default=OVERWRITE)

# Parse the arguments
args = parser.parse_args()
INPUT_FILE  = args.input
OUTPUT_FILE = args.output
overwrite   = args.overwrite
# -----------------------END CONFIGURATION--------------------------------------


# @njit(fastmath=True)
def fit_bulk_flow(Y, X, z_obs, phi_lc, theta_lc, yname, xname,
                  B_min, B_max, scat_min, scat_max, logA_min, logA_max,
                  rank, n_rank, comm):
    scaling_relation = f'{yname}-{xname}'
    # min_scat = 1000 # initialize a large number

    # Loop over the bulk flow direction and amplitude
    n_steps = (UBFMAX - UBFMIN)//UBF_STEP//n_rank * (360//LON_STEP) * (180//LAT_STEP)
    ubf_arr  = np.empty(n_steps, dtype=np.float64)
    vlon_arr = np.empty(n_steps, dtype=np.float64)
    vlat_arr = np.empty(n_steps, dtype=np.float64)
    scat_arr = np.ones(n_steps, dtype=np.float64)
    chi2_arr = np.empty(n_steps, dtype=np.float64)

    # The fit parameters
    B_STEP    = (B_max - B_min) / B_NSTEPS
    LOGA_STEP = (logA_max - logA_min) / LOGA_NSTEPS

    # The task division
    ubfmin = UBFMIN + rank * (UBFMAX - UBFMIN) // n_rank
    ubfmax = UBFMIN + (rank+1) * (UBFMAX - UBFMIN) // n_rank

    idx = 0
    for ubf in range(ubfmin, ubfmax, UBF_STEP):
        for vlon in range(-180, 180, LON_STEP):
            print(f'Rank {rank}: ubf={ubf}, vlon={vlon}, min scat={scat_max}', flush=True)
            for vlat in range(-90, 90, LAT_STEP):

                # Calculate the redshift
                angle = cf.angular_separation(phi_lc, theta_lc, vlon, vlat) * np.pi/180

                # From: z_bf = z_obs + ubf * (1 + z_bf) * np.cos(angle) / C # Maybe plot a bit the difference between the two
                z_bf = (z_obs + ubf * np.cos(angle) / C) / (1 - ubf * np.cos(angle) / C) # the ubf convention than the paper, same direction with H0 variation but opposite to cluster movement

                ## The relativistic correction
                #u_c_correct = ((1+z_obs)**2-1)/((1+z_obs)**2+1) + (1+z_obs)*ubf*np.cos(angle)/C
                #z_bf = np.sqrt((1+u_c_correct)/(1-u_c_correct))-1

                # Calculate the Luminosity distance
                if yname == 'LX':
                    # DL_zobs = cc.DL(z_obs, H0=68.1, Om=0.306, Ol=0.694)
                    # DL_zbf = cc.DL(z_bf, H0=68.1, Om=0.306, Ol=0.694)
                    DL_zobs = cosmo.luminosity_distance(z_obs).value
                    DL_zbf = cosmo.luminosity_distance(z_bf).value
                    Y_bf = Y*(DL_zbf)**2/(DL_zobs)**2
                elif yname == 'YSZ':
                    DA_zobs = cosmo.angular_diameter_distance(z_obs).value
                    DA_zbf = cosmo.angular_diameter_distance(z_bf).value
                    Y_bf = Y*(DA_zbf)**2/(DA_zobs)**2
                elif yname == 'M':
                    DA_zobs = cosmo.angular_diameter_distance(z_obs).value 
                    DA_zbf = cosmo.angular_diameter_distance(z_bf).value 
                    Y_bf = Y*(DA_zbf)**(5/2)/(DA_zobs)**(5/2)

                # To our fit parameters
                logY_ = cf.logY_(Y_bf, z=z_bf, relation=scaling_relation)
                logX_ = cf.logX_(X, relation=scaling_relation)

                # params = run_fit_log_scat(logY_, logX_, log_scat_step=LOG_SCAT_STEP,
                #                     B_step=B_STEP, logA_step=LOGA_STEP,
                #                     B_min=B_min,
                #                     B_max=B_max,
                #                     scat_min=scat_min,
                #                     scat_max=scat_max,
                #                     logA_min=logA_min,
                #                     logA_max=logA_max,
                #                     )

                params = cf.run_fit(logY_, logX_, scat_step=SCAT_STEP,
                                B_step=B_STEP, logA_step=LOGA_STEP,
                                B_min=B_min,
                                B_max=B_max,
                                scat_min=scat_min,
                                scat_max=scat_max,
                                logA_min=logA_min,
                                logA_max=logA_max,
                                )

                # idx = (ubf-UBFMIN)//UBF_STEP * (360//LON_STEP) * (180//LAT_STEP) \
                #    + (vlon+180)//LON_STEP * (180//LAT_STEP) \
                #    + (vlat+90)//LAT_STEP

                ubf_arr[idx]  = ubf
                vlon_arr[idx] = vlon
                vlat_arr[idx] = vlat
                scat_arr[idx] = params['scat']
                chi2_arr[idx] = params['chi2']

                idx += 1

                # Use the minimum scatter to set the next max scatter range, 
                # so the computational cost is reduced in each step
                scat_max = np.nanmin(scat_arr)
        # scat_max = np.nanmin(comm.allgather(scat_max)) # Find global minimum scatter, but this will introduce additional communication cost

    return ubf_arr, vlon_arr, vlat_arr, scat_arr, chi2_arr





def main():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    assert ((UBFMAX-UBFMIN)//UBF_STEP) % size == 0, 'UBFMAX-UBFMIN must be divisible by UBF_STEP * N_THREADS'
    
    # Flag for first entry
    first_entry_best_fit = True
    first_entry_all = True

    # Skip if the file already exists
    if os.path.exists(OUTPUT_FILE) and not OVERWRITE:
        print(f'File exists: {OUTPUT_FILE}')
        raise Exception('Output file exists and OVERWRITE==False.')

    # Load the sample
    halo_data = pd.read_csv(INPUT_FILE)

    for scaling_relation in RELATIONS:
        if rank==0:
            print(f'Fitting: {scaling_relation}', flush=True)

        n_clusters = cf.CONST[scaling_relation]['N']

        # Load the data
        _ = scaling_relation.find('-')
        yname = scaling_relation[:_]
        xname = scaling_relation[_+1:]
        Y = np.array(halo_data[cf.COLUMNS[yname]][:n_clusters])
        X = np.array(halo_data[cf.COLUMNS[xname]][:n_clusters])
        # Also load the position data
        phi_lc   = np.array(halo_data['phi_on_lc'][:n_clusters])
        theta_lc = np.array(halo_data['theta_on_lc'][:n_clusters])
        # the cosmological redshift from lightcone
        z_obs = np.array(halo_data['ObservedRedshift'][:n_clusters])

        # Go as high as 0.12
        for zmax in np.arange(0.06, 0.13, 0.02):
            zmask = (z_obs < zmax)

            # initialize the arrays for later gather
            if rank == 0:
                n_steps = (UBFMAX - UBFMIN)//UBF_STEP//size * (360//LON_STEP) * (180//LAT_STEP)
                ubf_arr_all  = np.empty((size, n_steps), dtype=np.float64)
                vlon_arr_all = np.empty((size, n_steps), dtype=np.float64)
                vlat_arr_all = np.empty((size, n_steps), dtype=np.float64)
                scat_arr_all = np.empty((size, n_steps), dtype=np.float64)
                chi2_arr_all = np.empty((size, n_steps), dtype=np.float64)
            else:
                ubf_arr_all  = None
                vlon_arr_all = None
                vlat_arr_all = None
                scat_arr_all = None
                chi2_arr_all = None # Use chi2 to break degeneracy

            ubf_arr, vlon_arr, vlat_arr, scat_arr, chi2_arr = fit_bulk_flow(Y=Y[zmask], X=X[zmask], 
                                         z_obs=z_obs[zmask], 
                                         phi_lc=phi_lc[zmask], theta_lc=theta_lc[zmask],
                                         yname=yname, xname=xname,
                                         rank=rank, n_rank=size,
                                         comm=comm,
                                         **FIT_RANGE[scaling_relation])

            # Gather the results
            comm.Gather(ubf_arr, ubf_arr_all, root=0)
            comm.Gather(vlon_arr, vlon_arr_all, root=0)
            comm.Gather(vlat_arr, vlat_arr_all, root=0)
            comm.Gather(scat_arr, scat_arr_all, root=0)
            comm.Gather(chi2_arr, chi2_arr_all, root=0)
            
            del ubf_arr, vlon_arr, vlat_arr, scat_arr

            if rank == 0:
                # Flatten the arrays
                ubf_arr_all = np.ravel(ubf_arr_all)
                vlon_arr_all = np.ravel(vlon_arr_all)
                vlat_arr_all = np.ravel(vlat_arr_all)
                scat_arr_all = np.ravel(scat_arr_all)
                chi2_arr_all = np.ravel(chi2_arr_all)

                # The best fit index
                min_sigma = np.nanmin(scat_arr_all) # But when there is degeneracy argmin only selects the first occurence
                min_sigma_mask = (scat_arr_all == min_sigma)
                min_chi2 = np.nanmin(chi2_arr_all[min_sigma_mask])
                fit_idx = np.where((chi2_arr_all == min_chi2) & min_sigma_mask)[0][0]

                # Save the best fit parameters
                fit_ubf = ubf_arr_all[fit_idx]
                fit_vlon = vlon_arr_all[fit_idx]
                fit_vlat = vlat_arr_all[fit_idx]
                min_scat = scat_arr_all[fit_idx]
                min_chi2 = chi2_arr_all[fit_idx]

                # Save the best fit parameters
                if first_entry_best_fit:
                    mode = 'w'
                else:
                    mode = 'a'
                with open(OUTPUT_FILE, mode) as f:
                    # Write the header on first entry
                    if first_entry_best_fit:
                        f.write('scaling_relation,zmax,ubf,lon,lat,sigma,chi2\n')
                        first_entry_best_fit = False

                    # Write the data
                    f.write(f'{scaling_relation},{zmax},{fit_ubf},{fit_vlon},{fit_vlat},{min_scat},{min_chi2}\n')

                    # System output
                    print(f'[{datetime.datetime.now()}]', flush=True)
                    print(f'{scaling_relation},z={zmax},ubf={fit_ubf},{fit_vlon},{fit_vlat},sigma={min_scat},chi2={min_chi2}', flush=True)

                # Save all the parameters
                if first_entry_all:
                    mode = 'w'
                else:
                    mode = 'a'
                with open(OUTPUT_FILE.replace('.csv', '_all.csv'), mode) as f:
                    # Write the header on first entry
                    if first_entry_all:
                        f.write('scaling_relation,zmax,ubf,lon,lat,sigma,chi2\n')
                        first_entry_all = False

                    # Write the data
                    for i in range(len(ubf_arr_all)):
                        f.write(f'{scaling_relation},{zmax},{ubf_arr_all[i]},{vlon_arr_all[i]},{vlat_arr_all[i]},{scat_arr_all[i]},{chi2_arr_all[i]}\n')

        # break # Do only one relation for now
    
    return 0


if __name__ == '__main__':
    assert main()==0, 'Program stopped before completion.'

