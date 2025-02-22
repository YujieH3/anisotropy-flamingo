# This script calculates the H0 variation using MCMC. 
#    - use COLUMNS_MC instead of COLUMNS to use raw SOAP spectroscopic-like
#    core-excised temperature instead of Chandra temperature.   
#
# Author                       : Yujie He
# Created on (MM/YYYY)         : 02/2025
# Last Modified on (MM/YYYY)   : 02/2025


# ---------------------------------------------------------------------------- #
#                            Command line arguments                            #
# ---------------------------------------------------------------------------- #


import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Calculate significance map for best fit scans.")

# Add arguments
parser.add_argument('-i', '--input', type=str, help='Input file')
parser.add_argument('-o', '--output', type=str, help='Output file')
parser.add_argument('-d', '--chaindir', type=str, help='Directory to save corner plots.')

parser.add_argument('-n', '--nthreads', type=int, help='Number of cores to use.', default=1)
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing.', default=False)

# Parse the arguments
args = parser.parse_args()
INPUT_FILE  = args.input
OUTPUT_FILE = args.output
CHAIN_DIR = args.chaindir

N_THREADS = args.nthreads
OVERWRITE   = args.overwrite

import os
os.environ["OMP_NUM_THREADS"] = f"{N_THREADS}"


# ---------------------------------------------------------------------------- #
#                                     Setup                                    #
# ---------------------------------------------------------------------------- #


import emcee
import numpy as np
import sys
sys.path.append('/cosma/home/do012/dc-he4/anisotropy-flamingo/tools')
import clusterfit as cf
from multiprocessing import Pool

# Constants
C = 299792.458                  # the speed of light in km/s

# Relations to fit
RELATIONS = ['LX-T', 'YSZ-T'] # pick from 'LX-T', 'M-T', 'YSZ-T'


# ---------------------------------------------------------------------------- #
#                              Internal functions                              #
# ---------------------------------------------------------------------------- #


def log_likelihood(theta      : np.ndarray, 
                   X          : np.ndarray,
                   Y          : np.ndarray,
                   z_obs      : np.ndarray,
                   phi_lc     : np.ndarray,
                   theta_lc   : np.ndarray,
                   sigma_obs_Y: np.ndarray,
                   sigma_obs_X: np.ndarray,
                   yname      : str,
                   xname      : str
    ): 

    lp = log_prior(theta)
    if not np.isfinite(lp):
       return -np.inf 
    
    delta, vlon, vlat, logA, B, sigma, = theta

    # Set the scaling relation to know the pivot point
    scaling_relation = f'{yname}-{xname}'

    # Anglular separation
    angle = cf.angular_separation(phi_lc, theta_lc, vlon, vlat) * np.pi/180

    # vary H0; since H0 enters distance to a simply *H0^(some power), we can just multiply the distance by a factor and not do the integral twice
    H0_ratio = 1 + delta*np.cos(angle)

    # Modified Y
    if yname == 'LX':
        Y_mod = Y*H0_ratio**-2
    elif yname == 'YSZ':
        Y_mod = Y*H0_ratio**-2
    elif yname == 'M':
        Y_mod = Y*H0_ratio**(-5/2) # (DA_modified)**(5/2)/(DA_default)**(5/2)
        
    # To our fit parameters
    logY_ = cf._logY_(Y_mod, 
                      z = z_obs,  
                      CY = cf.CONST_MC[scaling_relation]['CY'], 
                      gamma = cf.CONST_MC[scaling_relation]['gamma'])
    logX_ = cf._logX_(X, CX = cf.CONST_MC[scaling_relation]['CX'])

    model = B * logX_ + logA
    sigma_tot2 = sigma_obs_Y**2 + B**2 * sigma_obs_X**2 + sigma**2
    lnL = -0.5 * np.sum((logY_ - model) ** 2 / (sigma_tot2) + np.log(sigma_tot2)) # Kostas' implementation

    return lnL + lp


# set prior
def log_prior(theta):
    # A large flat prior for now
    delta, vlon, vlat, logA, B, sigma = theta # 6 parameters

    # If in range, p(theta)=1, else p(theta)=0
    if -1<logA<2 and 0.5<B<4 and 0.05<sigma<1 \
        and 0<delta<1 and -180<vlon<180 and -90<vlat<90:
        return 0.0
    else:
        return -np.inf



# ---------------------------------------------------------------------------- #
#                                  Main matter                                 #
# ---------------------------------------------------------------------------- #


# Load data
import pandas as pd
data = pd.read_csv(INPUT_FILE)

# Skip if the file already exists
if os.path.exists(OUTPUT_FILE) and not OVERWRITE:
    print(f'File exists: {OUTPUT_FILE}')
    raise Exception('Output file exists and OVERWRITE==False.')

first_entry = True

# Start
for scaling_relation in RELATIONS: 
    n_clusters = cf.CONST_MC[scaling_relation]['N']

    # Load the data
    yname, xname = cf.parse_relation_name(scaling_relation)

    Y = data[cf.COLUMNS_MC[yname]][:n_clusters].values
    X = data[cf.COLUMNS_MC[xname]][:n_clusters].values

    phi_lc   = data['phi_on_lc'][:n_clusters].values
    theta_lc = data['theta_on_lc'][:n_clusters].values

    z_obs = data['ObservedRedshift'][:n_clusters].values

    eY = data['e'+cf.COLUMNS_MC[yname]][:n_clusters].values   # in ratio 0-1
    eX = data['e'+cf.COLUMNS_MC[xname]][:n_clusters].values
    scat_obs_Y = np.log10(1 + eY) 
    scat_obs_X = np.log10(1 + eX)

    # set the starting point for chain
    pos0 = np.array([0, 0, 0, 1, 1, 0.1]) + 1e-1 * np.random.rand(32, 6)
    nwalkers, ndim = pos0.shape

    # Sampler on multiple threads
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, 
                                        ndim, 
                                        log_likelihood, 
                                        args = (X, Y, z_obs, phi_lc, theta_lc, scat_obs_Y, scat_obs_X, yname, xname),
                                        pool = pool
                                        )

        # Run
        sampler.run_mcmc(pos0, 15000, progress=False)  # now the chain is saved. progress spam the standard output, toggled to False

    # Convergence test
    try:
        tau = sampler.get_autocorr_time()
        print(tau)
    except emcee.autocorr.AutocorrError:
        print('The chain is too short to get a reliable autocorrelation time.')
        tau = 0

    # Get the samples
    flat_samples = sampler.get_chain(discard=1000, thin=80, flat=True)
    print(flat_samples.shape)

    # Save the chain
    np.save(os.path.join(CHAIN_DIR, f'{scaling_relation}_chain.npy'), flat_samples)


# ------------------------------ Postprocessing ------------------------------ #

    # For delta we use the 16, 50, 84 quantiles
    delta_distr = flat_samples[:, 0]
    lower_delta  = np.percentile(delta_distr, 16)
    median_delta = np.percentile(delta_distr, 50)
    upper_delta  = np.percentile(delta_distr, 84)
        
    # For saving
    delta = median_delta
    delta_err_lower = median_delta - lower_delta
    delta_err_upper = upper_delta - median_delta
    print(f'delta: {lower_delta} ~ {upper_delta} \nor {delta} -{delta_err_lower} +{delta_err_upper}')

    # latitude is not periodic!
    vlat_distr = flat_samples[:, 2]
    lower_vlat = np.percentile(vlat_distr, 16)
    median_vlat = np.percentile(vlat_distr, 50)
    upper_vlat = np.percentile(vlat_distr, 84)

    # For saving
    vlat = median_vlat
    vlat_err_lower = median_vlat - lower_vlat
    vlat_err_upper = upper_vlat - median_vlat
    print(f'vlat: {lower_vlat} ~ {upper_vlat} \nor {vlat} -{vlat_err_lower} +{vlat_err_upper}')


    # Find the range w.r.t. the peak.
    vlon, vlon_err_lower, vlon_err_upper, lower_vlon, upper_vlon = cf.periodic_error_range(flat_samples[:,1], full_range=360) 
    print(f'vlon: {lower_vlon} ~ {upper_vlon} \nor {vlon} -{vlon_err_lower} +{vlon_err_upper}')


    # Save the best fit parameters
    if first_entry:
        mode = 'w'
    else:
        mode = 'a'
        
    # Write line by line
    with open(OUTPUT_FILE, mode) as f:

        # Write the header on first entry
        if first_entry:
            f.write('scaling_relation,delta,delta_err_lower,delta_err_upper,vlon,vlon_err_lower,vlon_err_upper,vlat,vlat_err_lower,vlat_err_upper,convergence_time\n')
            first_entry = False

        # Write the data
        f.write(f'{scaling_relation},{delta},{delta_err_lower},{delta_err_upper},{vlon},{vlon_err_lower},{vlon_err_upper},{vlat},{vlat_err_lower},{vlat_err_upper},{np.mean(tau)}\n')
