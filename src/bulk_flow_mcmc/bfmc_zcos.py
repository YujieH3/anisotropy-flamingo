# ---------------------------------------------
# This script calculates the bulk flow using MCMC.
#     - use COLUMNS_MC instead of COLUMNS to use
# raw SOAP spectroscopic-like core-excised 
# temperature instead of Chandra temperature.
#
# Author                       : Yujie He
# Created on (MM/YYYY)         : 06/2024
# Last Modified on (MM/YYYY)   : 12/2024
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
#PLOT_DIR = '/data1/yujiehe/data/fits/7bulk-flow-model-mcmc-lightcone0'
INPUT_FILE = '/data1/yujiehe/data/samples_in_lightcone1_with_trees_duplicate_excision_outlier_excision.csv'
OUTPUT_FILE = '/data1/yujiehe/data/fits/bulk_flow_mcmc_lightcone1.csv'
PLOT_DIR = '/data1/yujiehe/data/fits/7bulk-flow-model-mcmc-lightcone1'
OVERWRITE = True

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
parser.add_argument('-d', '--chaindir', type=str, help='Directory to save corner plots.', default=None)
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing.', default=OVERWRITE)

# Parse the arguments
args = parser.parse_args()
INPUT_FILE  = args.input
OUTPUT_FILE = args.output
CHAIN_DIR = args.chaindir
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
    
    ubf, vlon, vlat, logA, B, sigma, = theta

    # Set the scaling relation to know the pivot point
    scaling_relation = f'{yname}-{xname}'

    # Calculate the redshift
    angle = cf.angular_separation(phi_lc, theta_lc, vlon, vlat) * np.pi/180
    
    # Correct the bulk flow accordingly: z_bf = z_obs + ubf * (1 + z_bf) * np.cos(angle) / C
    z_bf = (z_obs + ubf * np.cos(angle) / C) / (1 - ubf * np.cos(angle) / C) # Non-relativistic correction

    # The relativistic correction
    # u_c_correct=((1+z_obs)**2-1)/((1+z_obs)**2+1) + (1+z_obs)*ubf*np.cos(angle)/C
    # z_bf=np.sqrt((1+u_c_correct)/(1-u_c_correct))-1

    # Calculate the Luminosity distance
    if yname == 'LX':
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

    model = B * logX_ + logA
    lnL = -0.5 * np.sum((logY_ - model) ** 2 / (sigma**2) + np.log(sigma**2)) # Kostas' implementation

    return lnL + lp

# set prior
def log_prior(theta):
    # A large flat prior for now
    ubf, vlon, vlat, logA, B, sigma = theta # 6 parameters

    # If in range, p(theta)=1, else p(theta)=0
    if -1<logA<1 and 0.5<B<3.5 and 0.05<sigma<1 \
        and 0<ubf<1000 and -180<vlon<180 and -90<vlat<90:
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
    z_obs = np.array(data['redshift'][:n_clusters])


    # Load data and set zmax
    for zmax in [0.07, 0.10, 0.13]:
        zmask    = z_obs < zmax

        # Select data below some redshift
        z_obs    = z_obs[zmask]
        X        = X[zmask]
        Y        = Y[zmask]
        phi_lc   = phi_lc[zmask]
        theta_lc = theta_lc[zmask]

        print('Redshift shell:', zmax)
        print('Number of clusters:', np.sum(zmask)) # Number of clusters left

        # Optimize
        from scipy.optimize import minimize
        from scipy.optimize import differential_evolution
        nll = lambda *args: -log_likelihood(*args)
        initial = np.array([100, 0, 0, 1, 1, 0.1]) # initial guess
        # ubf, vlon, vlat, logA, B, sigma

        pos0 = initial + 1e-2 * np.random.randn(32, 6)
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
                                            pool    = pool,
                                            )
        
            # Run
            sampler.run_mcmc(pos0, 15000, progress=False)

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

        # Save flat samples
        if CHAIN_DIR is not None:
            np.save(os.path.join(f'{CHAIN_DIR}', f'{scaling_relation}_zmax{zmax}_chain.npy'), flat_samples)

        # For ubf we use the 16, 50, 84 quantiles
        ubf_distr = flat_samples[:, 0]
        lower_ubf  = np.percentile(ubf_distr, 16)
        median_ubf = np.percentile(ubf_distr, 50)
        upper_ubf  = np.percentile(ubf_distr, 84)
        
        # For saving
        ubf = median_ubf
        ubf_err_lower = median_ubf - lower_ubf
        ubf_err_upper = upper_ubf - median_ubf
        print(f'ubf: {lower_ubf} ~ {upper_ubf} \nor {ubf} -{ubf_err_lower} +{ubf_err_upper}')


        # Same with latitude, note that latitude is not periodic!
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
                f.write('scaling_relation,zmax,ubf,ubf_err_lower,ubf_err_upper,vlon,vlon_err_lower,vlon_err_upper,vlat,vlat_err_lower,vlat_err_upper,convergence_time\n')
                first_entry = False
            # Write the data
            f.write(f'{scaling_relation},{zmax},{ubf},{ubf_err_lower},{ubf_err_upper},{vlon},{vlon_err_lower},{vlon_err_upper},{vlat},{vlat_err_lower},{vlat_err_upper},{np.mean(tau)}\n')
