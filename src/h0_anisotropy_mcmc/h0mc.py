# ---------------------------------------------
# This script calculates the H0 variation using
# MCMC. The script is not yet parallelized.
#    - use COLUMNS_MC instead of COLUMNS to use
# raw SOAP spectroscopic-like core-excised 
# temperature instead of Chandra temperature.
#    - now the chain is saved to chain_dir, which
# can be accessed later by
#   `reader = emcee.backends.HDFBackend(filename)`
# Author                       : Yujie He
# Created on (MM/YYYY)         : 06/2024
# Last Modified on (MM/YYYY)   : 09/2024
# ---------------------------------------------

import emcee
import numpy as np
import os
import sys
sys.path.append('/cosma/home/do012/dc-he4/anisotropy-flamingo/tools')
import clusterfit as cf
from multiprocessing import Pool
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=68.1, Om0=0.306, Ob0=0.0486)

C = 299792.458                  # the speed of light in km/s

# -----------------------CONFIGURATION------------------------------------------
# Input file is a halo catalog with lightcone data.
#INPUT_FILE = '/data1/yujiehe/data/samples_in_lightcone0_with_trees_duplicate_excision_outlier_excision.csv'
#OUTPUT_FILE = '/data1/yujiehe/data/fits/bulk_flow_mcmc_lightcone0.csv'
#CHAIN_DIR = '/data1/yujiehe/data/fits/7bulk-flow-model-mcmc-lightcone0'
INPUT_FILE = '/data1/yujiehe/data/samples_in_lightcone1_with_trees_duplicate_excision_outlier_excision.csv'
OUTPUT_FILE = '/data1/yujiehe/data/fits/bulk_flow_mcmc_lightcone1.csv'
CHAIN_DIR = '/data1/yujiehe/data/fits/7bulk-flow-model-mcmc-lightcone1'
OVERWRITE = False

# Relations to fit
RELATIONS = ['LX-T', 'YSZ-T', 'M-T'] # pick from 'LX-T', 'M-T', 'LX-YSZ', 'LX-M', 'YSZ-M', 'YSZ-T'

# -----------------------------COMMAND LINE ARGUMENTS---------------------------

import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Calculate significance map for best fit scans.")

# Add arguments
parser.add_argument('-i', '--input', type=str, help='Input file', default=INPUT_FILE)
parser.add_argument('-o', '--output', type=str, help='Output file', default=OUTPUT_FILE)
parser.add_argument('-n', '--nthreads', type=int, help='Number of cores to use.', default=1)
parser.add_argument('-d', '--chaindir', type=str, help='Directory to save corner plots.', default=CHAIN_DIR)
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing.', default=OVERWRITE)

# Parse the arguments
args = parser.parse_args()
INPUT_FILE  = args.input
OUTPUT_FILE = args.output
CHAIN_DIR = args.chaindir
N_THREADS = args.nthreads
OVERWRITE   = args.overwrite

os.environ["OMP_NUM_THREADS"] = f"{N_THREADS}"
# -----------------------END CONFIGURATION--------------------------------------






def log_likelihood(theta, X, Y, z_obs, phi_lc, theta_lc, yname, xname):
    """
    X, Y, z_obs, phi_lc, theta_lc, are from the data
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
       return -np.inf 
    
    delta, vlon, vlat, logA, B, sigma, = theta

    # Set the scaling relation to know the pivot point
    scaling_relation = f'{yname}-{xname}'

    # Calculate the redshift
    angle = cf.angular_separation(phi_lc, theta_lc, vlon, vlat) * np.pi/180

    # vary H0; since H0 enters distance to a simply *H0^(some power), we can just multiply the distance by a factor and not do the integral twice
    H0_ratio = 1 + delta*np.cos(angle)

    # Calculate the Luminosity distance
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
    lnL = -0.5 * np.sum((logY_ - model) ** 2 / (sigma**2) + np.log(sigma**2)) # Kostas' implementation

    return lnL + lp



# set prior
def log_prior(theta):
    # A large flat prior for now
    delta, vlon, vlat, logA, B, sigma = theta # 6 parameters

    # If in range, p(theta)=1, else p(theta)=0
    if -1<logA<1 and 0.5<B<3.5 and 0.05<sigma<1 \
        and 0<delta<1 and -180<vlon<180 and -90<vlat<90:
        return 0.0
    else:
        return -np.inf






# Load data
import pandas as pd
data = pd.read_csv(INPUT_FILE)

# Skip if the file already exists
if os.path.exists(OUTPUT_FILE) and not OVERWRITE:
    print(f'File exists: {OUTPUT_FILE}')
    raise Exception('Output file exists and OVERWRITE==False.')

first_entry = True



for scaling_relation in RELATIONS: 
    n_clusters = cf.CONST_MC[scaling_relation]['N']

    # Load the data
    _ = scaling_relation.find('-')
    yname = scaling_relation[:_]
    xname = scaling_relation[_+1:]
    Y = np.array(data[cf.COLUMNS_MC[yname]][:n_clusters])
    X = np.array(data[cf.COLUMNS_MC[xname]][:n_clusters])
    # Also load the position data
    phi_lc   = np.array(data['phi_on_lc'][:n_clusters])
    theta_lc = np.array(data['theta_on_lc'][:n_clusters])
    # the observed redshift from lightcone
    z_obs = np.array(data['ObservedRedshift'][:n_clusters])


    # scipy estimation of staring point
    # from scipy.optimize import minimize
    from scipy.optimize import differential_evolution
    nll = lambda *args: -log_likelihood(*args)
    initial = np.array([0, 0, 0, 1, 1, 0.1]) # initial guess
    bounds = [(0, 0.09), (-180, 180), (-90, 90), (0.1, 1), (0.5, 3.5), (0.05, 1)]

    soln = differential_evolution(nll, args=(X, Y, z_obs, phi_lc, theta_lc, yname, xname), 
                    bounds=bounds, popsize=10, strategy='rand1bin')
    print(soln.x)

    # set the starting point for chain
    pos0 = soln.x + 1e-2 * np.random.randn(32, 6)
    nwalkers, ndim = pos0.shape

    # # create the backend for saving the chain; we choose to save it for later analysis
    # filename = os.path.join(CHAIN_DIR, f'{scaling_relation}_chain.h5')
    # if os.path.exists(filename) and not OVERWRITE:
    #     print(f'File exists: {filename}')
    #     raise Exception('Chain file exists and OVERWRITE==False.')
    # else:
    #     backend = emcee.backends.HDFBackend(filename)
    #     backend.reset(nwalkers, ndim)

    # Create a sampler
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, 
                                        ndim, 
                                        log_likelihood, 
                                        # backend = backend,
                                        args    = (X, Y, z_obs, phi_lc, theta_lc, yname, xname),
                                        pool    = pool
                                        )

        # Run
        sampler.run_mcmc(pos0, 15000, progress=False)  # now the chain is saved. progress spam the standard output, toggled to False

    # Small convergence test
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
