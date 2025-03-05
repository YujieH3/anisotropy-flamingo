# ---------------------------------------------------------------------------- #
# This script calculates the joint H0 variation using MCMC. The script is
# parallelized.  For joint analysis, there are 9 parameters, logA1, logA2, B1,
# B2, scat1, scat2, delta, lon, lat
#    - use COLUMNS_MC instead of COLUMNS to use
# raw SOAP spectroscopic-like core-excised 
# temperature instead of Chandra temperature.
#    - Now with support for mock scatter
#
# Author                       : Yujie He
# Created on (MM/YYYY)         : 02/2025
# Last Modified on (MM/YYYY)   : 02/2025
# ---------------------------------------------------------------------------- #

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

# Joint analysis
RELATION1 = 'LX-T'
RELATION2 = 'YSZ-T'

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
N_THREADS = args.nthreads
OVERWRITE   = args.overwrite

# Set number of threads
os.environ["OMP_NUM_THREADS"] = f"{N_THREADS}"
# -----------------------END CONFIGURATION--------------------------------------


# set prior
def log_prior(theta):
    # A large flat prior for now
    delta, vlon, vlat, logA1, B1, sigma1, logA2, B2, sigma2 = theta # 9 parameters

    # If in range, p(theta)=1, else p(theta)=0
    if -1<logA1<1 and 0.5<B1<3.5 and 0.05<sigma1<1 \
        and -1<logA2<1 and 0.5<B2<3.5 and 0.05<sigma2<1\
        and 0<delta<1 and -180<vlon<180 and -90<vlat<90:
        return 0.0
    else:
        return -np.inf


def log_likelihood(theta, X1, Y1, X2, Y2, z_obs, phi_lc, theta_lc, relation1, 
                   relation2, scat_obs_Y1, scat_obs_X1, scat_obs_Y2, scat_obs_X2):
    
    # Set prior
    lp = log_prior(theta)

    # If theta outside of range
    if not np.isfinite(lp):
        return -np.inf
    
    # Read in the parameters
    delta, vlon, vlat, logA1, B1, sigma1, logA2, B2, sigma2 = theta # 9 parameters

    # Angular separation
    angle = cf.angular_separation(phi_lc, theta_lc, vlon, vlat) * np.pi/180

    # Vary H0
    H0_ratio = 1 + delta * np.cos(angle)

    # Detect relation names
    _ = relation1.find('-')
    yname1 = relation1[:_]
    _ = relation2.find('-')
    yname2 = relation2[:_]

    # Modified Y1
    if yname1 == 'LX':
        Y1_mod = Y1*H0_ratio**-2
    elif yname1 == 'YSZ':
        Y1_mod = Y1*H0_ratio**-2
    elif yname1 == 'M':
        Y1_mod = Y1*H0_ratio**(-5/2) # (DA_modified)**(5/2)/(DA_default)**(5/2)
    
    # Modified Y2
    if yname2 == 'LX':
        Y2_mod = Y2*H0_ratio**-2
    elif yname2 == 'YSZ':
        Y2_mod = Y2*H0_ratio**-2
    elif yname2 == 'M':
        Y2_mod = Y2*H0_ratio**(-5/2) # (DA_modified)**(5/2)/(DA_default)**(5/2)
        
    # To fit quantities
    logY1_ = cf._logY_(Y = Y1_mod,
                       z = z_obs,
                       CY = cf.CONST_MC[relation1]['CY'],
                       gamma = cf.CONST_MC[relation1]['CX']
                       )
    logX1_ = cf._logX_(X = X1,
                       CX = cf.CONST_MC[relation1]['CX']
                       )
    # Same for second relation
    logY2_ = cf._logY_(Y = Y2_mod,
                       z = z_obs,
                       CY = cf.CONST_MC[relation2]['CY'],
                       gamma = cf.CONST_MC[relation2]['CX']
                       )
    logX2_ = cf._logX_(X = X2,
                       CX = cf.CONST_MC[relation2]['CX']
                       )

    # Likelihood, relation 1
    model1 = B1 * logX1_ + logA1
    sigma_tot2_1 = scat_obs_Y1**2 + B1**2 * scat_obs_X1**2 + sigma1**2
    lnL1 = -0.5 * np.sum((logY1_ - model1) ** 2 / (sigma_tot2_1) + np.log(sigma_tot2_1)) # Kostas' implementation
    # Likelihood, relation 2
    model2 = B2 * logX2_ + logA2
    sigma_tot2_2 = scat_obs_Y2**2 + B2**2 * scat_obs_X2**2 + sigma2**2
    lnL2 = -0.5 * np.sum((logY2_ - model2) ** 2 / (sigma_tot2_2) + np.log(sigma_tot2_2)) # Kostas' implementation

    # Joint likelihood is their product (sum in logspace)
    lnL = lnL1 + lnL2 + lp

    return lnL



# Load data
import pandas as pd
data = pd.read_csv(INPUT_FILE)

# Skip if the file already exists
if os.path.exists(OUTPUT_FILE) and not OVERWRITE:
    print(f'File exists: {OUTPUT_FILE}')
    raise Exception('Output file exists and OVERWRITE==False.')

# Prepare to write the first line, in the end of the script
first_entry = True

# Use the intersection of two group of clusters
n_clusters1 = cf.CONST_MC[RELATION1]['N']
n_clusters2 = cf.CONST_MC[RELATION2]['N']
n_clusters = np.min((n_clusters1, n_clusters2))

# Load the data
_ = RELATION1.find('-')
yname1 = RELATION1[:_]
xname1 = RELATION1[_+1:]
_ = RELATION2.find('-')
yname2 = RELATION2[:_]
xname2 = RELATION2[_+1:]
Y1 = data[cf.COLUMNS_MC[yname1]][:n_clusters].values
X1 = data[cf.COLUMNS_MC[xname1]][:n_clusters].values
Y2 = data[cf.COLUMNS_MC[yname2]][:n_clusters].values
X2 = data[cf.COLUMNS_MC[xname2]][:n_clusters].values

# Also load the position data
phi_lc   = data['phi_on_lc'][:n_clusters].values
theta_lc = data['theta_on_lc'][:n_clusters].values
# the observed redshift from lightcone
z_obs = data['ObservedRedshift'][:n_clusters].values

# Mock scatter for relation 1
eY1 = data['e'+cf.COLUMNS_MC[yname1]][:n_clusters].values   # in ratio 0-1
eX1 = data['e'+cf.COLUMNS_MC[xname1]][:n_clusters].values
scat_obs_Y1 = np.log10(1 + eY1) 
scat_obs_X1 = np.log10(1 + eX1)

# Mock scatter for relation 2
eY2 = data['e'+cf.COLUMNS_MC[yname2]][:n_clusters].values   # in ratio 0-1
eX2 = data['e'+cf.COLUMNS_MC[xname2]][:n_clusters].values
scat_obs_Y2 = np.log10(1 + eY2) 
scat_obs_X2 = np.log10(1 + eX2)

# Set starting point
init = np.array([0.1, 0, 0, 1, 1, 0.1, 1, 1, 0.1]) # delta, vlon, vlat, logA1, B1, C1, logA2, B2, C2

# set the starting point for chain
pos0 = init + 1e-2 * np.random.randn(32, 9)
nwalkers, ndim = pos0.shape # 9 dimension, 32 walkers

# Create a sampler
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, 
                                    ndim, 
                                    log_likelihood, 
                                    args    = (X1, Y1, X2, Y2, z_obs, phi_lc,
                                    theta_lc, RELATION1, RELATION2, scat_obs_Y1,
                                    scat_obs_X1, scat_obs_Y2, scat_obs_X2), # Sequence matters!
                                    pool    = pool
                                    )

    # Run
    sampler.run_mcmc(pos0, 50_000, progress=False)  # now the chain is saved. progress spam the standard output, toggled to False

# Small convergence test
try:
    tau = sampler.get_autocorr_time()
    print(tau)
except emcee.autocorr.AutocorrError:
    print('The chain is too short to get a reliable autocorrelation time.')
    tau = 0

# Get the samples
flat_samples = sampler.get_chain(discard=4000, thin=80, flat=True)
print(flat_samples.shape)

# Save the chain
if CHAIN_DIR is not None:
    np.save(os.path.join(CHAIN_DIR, f'{RELATION1}_{RELATION2}_chain.npy'), flat_samples)

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
        f.write('relation1,relation2,delta,delta_err_lower,delta_err_upper,vlon,vlon_err_lower,vlon_err_upper,vlat,vlat_err_lower,vlat_err_upper,convergence_time\n')
        first_entry = False
    # Write the data
    f.write(f'{RELATION1},{RELATION2},{delta},{delta_err_lower},{delta_err_upper},{vlon},{vlon_err_lower},{vlon_err_upper},{vlat},{vlat_err_lower},{vlat_err_upper},{np.mean(tau)}\n')
