#!/usr/bin/env python
"""
Author                 : Yujie He
Created on (MM/YYYY)   : 06/2024
Description:
    This script calculates the H0 variation using MCMC, MPI parallelised.

    Note: progress bar won't show when --mpi is enabled.
Example:
    mpiexec -n <number of cores> python h0mc.py --mpi # parallel
    python h0mc.py # serial
"""

import os
import sys
sys.path.append("../tools")

import unyt as u
import pandas as pd
import numpy as np
from loguru import logger
from numpy.typing import ArrayLike
from astropy.cosmology import FlatLambdaCDM
import astropy

import clusterfit as cf

# ---------------------------------------------------------------------------- #
#                                  PARAMETERS                                  #
# ---------------------------------------------------------------------------- #

# INPUT_FILE = "../data/Sample-Lx-Tx-CS.txt"
OUTPUT_DIR = "../data/processed"
SAMPLE_DIR = OUTPUT_DIR
SAMPLER = "emcee" # zeus seems to be much slower
NUM_RELATION = 0
SHUFFLE = False
NSTEPS = 50_000

# Global pivots
TEXP_PIVOT = 177 * u.s
CONC_PIVOT = 0.16

# Initial position for parameters
DELTA_INIT = 1e-3
DGLON_INIT = 0
DGLAT_INIT = 0
LOGA_INIT = 0
B_INIT = 1e-2
SINTR_INIT = 1e-2
KEXP_INIT = 0.0
KCONC_INIT = 0.0

# Preset
sampler_args = None   # populated before pool creation so MPI workers inherit it

TMIN_LIMIT = 0.5 * u.keV
ZMIN = 0.02
ZMAX = 0.35

# ------------------- Cosmology (flat LCDM): user-provided ------------------- #
H0 = 70 * u.km / u.s / u.Mpc                  # km/s/Mpc
OMEGA_M = 0.3
OMEGA_L = 0.7
cosmo = FlatLambdaCDM(H0=70, Om0=OMEGA_M)
c_km_s = 299792.458 * u.km / u.s      # km/s

USE_CONC = NUM_RELATION in (2, 3)
if NUM_RELATION == 0:
    # ----------------------------------- Lx-T ----------------------------------- #
    INPUT_FILE = "../data/Sample-Lx-Tx-CS.txt"
    Y_PIVOT = 5.5 * 1e42 * u.erg / u.s
    X_PIVOT = 1.8 * u.keV
    Z_PIVOT = 0.137
    # FIXED redshift evolution
    GAMMA = 1.88
    ALPHA_Y = -2.0
    ALPHA_X = 0.0
elif NUM_RELATION == 1:
    # ---------------------------------- Mgas-T ---------------------------------- #
    INPUT_FILE = "../data/Sample-Mgas-Tx-CS.txt"
    Y_PIVOT = 1.53 * 1e13 * u.Msun
    X_PIVOT = 1.8 * u.keV
    Z_PIVOT = 0.133
    # FIXED redshift evolution for Mgas correction (your default)
    GAMMA = -1.0
    ALPHA_Y = -2.5
    ALPHA_X = 0.0
elif NUM_RELATION == 2:
    # ----------------------------------- Ysz-T ---------------------------------- #
    INPUT_FILE = "../data/Sample-Ysz-Tx-CS.txt"
    Y_PIVOT = 23 * u.kpc**2
    X_PIVOT = 2.3 * u.keV
    Z_PIVOT = 0.139
    # FIXED redshift evolution (as requested)
    GAMMA = -2.0
    ALPHA_Y = -2.0
    ALPHA_X = 0.0
elif NUM_RELATION == 3:
    # --------------------------------- Mgas-Ysz --------------------------------- #
    INPUT_FILE = "../data/Sample-Mgas-Ysz-no-Mgas-T-CS.txt"
    Y_PIVOT = 1.7 * 1e13 * u.Msun  # in 1e13 Msun units, because we divide Mgas(1e11) by 100
    X_PIVOT = 24 * u.kpc**2
    Z_PIVOT = 0.15
    # FIXED redshift evolution
    GAMMA = 0.5
    ALPHA_Y = -2.5
    ALPHA_X = -2.0

# ---------------------------------------------------------------------------- #
#                                   Functions                                  #
# ---------------------------------------------------------------------------- #

def read_sample(path: str) -> pd.DataFrame: 
    """Read eROSITA data to a pandas dataframe.

    :param path: path of the file (in csv or txt)
    :type path: str
    :return: data
    :rtype: pandas.DataFrame
    """
    df = pd.read_csv(path, sep=r'\s+', engine='python')
    return df.rename(columns={'#Name': 'Name'})

def _unpack(params):
    delta, dglon, dglat, loga, b, sintr, kexp = params[:7]
    kconc = params[7] if USE_CONC else 0.0
    return delta, dglon, dglat, loga, b, sintr, kexp, kconc

def log_likelihood_global(params: ArrayLike):
    # Make data a global variable gives a 3 times speed up, according to
    # https://emcee.readthedocs.io/en/latest/tutorials/parallel/
    return log_likelihood(params, **sampler_args)

def log_likelihood(params: ArrayLike,
                   X: ArrayLike,
                   Y: ArrayLike,
                   X_sigma: ArrayLike,
                   Y_sigma: ArrayLike,
                   z: ArrayLike,
                   cluster_glon: ArrayLike,
                   cluster_glat: ArrayLike,
                   texp: ArrayLike,
                   conc: ArrayLike,
                   ):

    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf

    delta, dglon, dglat, loga, b, sintr, kexp, kconc = _unpack(params)

    # Angular separation
    theta = astropy.coordinates.angular_separation(
        cluster_glon * np.pi/180, 
        cluster_glat * np.pi/180, 
        dglon * np.pi/180,
        dglat * np.pi/180)    # returns float in radian

    # vary H0; since H0 enters distance to a simply *H0^(some power), we can
    # just multiply the distance by a factor and not do the integral twice
    H0_ratio = 1 + delta * np.cos(theta)

    # Compute logX' = X / CX, logY' = Y / CY * E(z)^gamma
    lhs = np.log10(Y / Y_PIVOT) + ALPHA_Y * np.log10(H0_ratio)
    rhs = loga + b * (np.log10(X / X_PIVOT) + ALPHA_X * np.log10(H0_ratio)) \
                    + GAMMA * np.log10(cf.E(z)/cf.E(Z_PIVOT)) \
                    + kexp * np.log10(texp/TEXP_PIVOT)
    if USE_CONC:
        rhs += kconc * np.log10(conc/CONC_PIVOT)

    sigma_tot2 = Y_sigma**2 + b**2 * X_sigma**2 + sintr**2
    lnL = -0.5 * np.sum((lhs - rhs) ** 2 / (sigma_tot2) + np.log(sigma_tot2))

    return lnL + lp


# set prior
def log_prior(params):
    (delta, dglon, dglat, 
     loga, b, sintr, kexp, kconc) = _unpack(params)

    # If in range, p(theta)=1, else p(theta)=0
    in_range = (
        0 <= delta < 1
        and -180 <= dglon <= 180
        and -90 <= dglat <= 90
        and -1 < loga < 1
        and 1e-5 < b < 5
        and 1e-5 < sintr < 1
        and -1 < kexp < 1
    )
    if USE_CONC:
        in_range = in_range and -1 < kconc < 1
    return 0.0 if in_range else -np.inf


def _fmt(name, val, lo, hi):
    """Log a parameter as  name: lo ~ hi  (val -lo_err +hi_err)"""
    logger.info(
        f"{name}: {lo:.4f} ~ {hi:.4f}"
        f"  ({val:.4f} -{val-lo:.4f} +{hi-val:.4f})"
    )

# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #

if __name__ == '__main__':

    # https://emcee.readthedocs.io/en/latest/tutorials/parallel/
    os.environ["OMP_NUM_THREADS"] = "1"

    import argparse
    from schwimmbad import choose_pool

    parser = argparse.ArgumentParser()
    parser.add_argument("--mpi", action="store_true",
                        help="Use MPI (mpiexec -n N python h0mc.py --mpi)")
    parser.add_argument("--ncores", type=int, default=1,
                        help="Local cores (default: 1 = serial)")
    args = parser.parse_args()

    logger.info("Initialising...")
    first_entry = True

    if SAMPLER.lower() == "zeus":
        import zeus
    elif SAMPLER.lower() == "emcee":
        import emcee
    else:
        raise ValueError(
            f"Sampler '{SAMPLER}' not supported. "
            "Choose from: ['emcee', 'zeus']."
        )

# --------------------------------- Load data -------------------------------- #

    data = read_sample(path=INPUT_FILE)

    # T and z selection
    T = data['T'] * u.keV
    data = data[T > TMIN_LIMIT]
    z = data['z']
    data = data[(ZMIN < z) & (z < ZMAX)]

    if NUM_RELATION == 0: # Lx-T
        Y = data['Lx_02_23(1e42)'] * 1e42 * u.erg/u.s
        Ymax = data['L500_max'] * 1e42 * u.erg/u.s
        Ymin = data['L500_min'] * 1e42 * u.erg/u.s
        X = data['T'] * u.keV
        Xmax = data['T_max'] * u.keV
        Xmin = data['T_min'] * u.keV
    elif NUM_RELATION == 1: # Mgas-T
        Y = data['Mgas(1e11)'] * 1e11 * u.Msun
        Ymax = data['Mgas_max'] * 1e11 * u.Msun
        Ymin = data['Mgas_min'] * 1e11 * u.Msun
        X = data['T'] * u.keV
        Xmax = data['T_max'] * u.keV
        Xmin = data['T_min'] * u.keV
    elif NUM_RELATION == 2: # Ysz-T
        Y5R500 = data['Y5R500(arcmin)']
        DA = cosmo.angular_diameter_distance(z).to('kpc').value * u.kpc
        Y = Y5R500 * (np.pi/60/180)**2 * DA**2
        Y_sigma = data['e_Y5R500'] * (np.pi/60/180)**2 * DA**2
        X = data['T'] * u.keV
        Xmax = data['T_max'] * u.keV
        Xmin = data['T_min'] * u.keV
    elif NUM_RELATION == 3: # Ysz-Mgas
        Y5R500 = data['Y5R500(arcmin)']
        DA = cosmo.angular_diameter_distance(z).to('kpc').value * u.kpc
        Y = Y5R500 * (np.pi/60/180)**2 * DA**2
        Y_sigma = data['e_Y5R500'] * (np.pi/60/180)**2 * DA**2
        X = data['Mgas(1e11)'] * 1e11 * u.Msun
        Xmax = data['Mgas_max'] * 1e11 * u.Msun
        Xmin = data['Mgas_min'] * 1e11 * u.Msun
    else:
        logger.debug("Other relation not ready yet.")

    try:
        Y_sigma = 0.5 * (Ymax - Ymin) / (Y * np.log(10))
    except NameError:
        Y_sigma
    X_sigma = 0.5 * (Xmax - Xmin) / (X * np.log(10))
    texp = data['EXP_TIME(s)'] * u.s
    conc = data['conc_R500'] if USE_CONC else None

    if SHUFFLE == True:
        logger.info("Shuffling position.")
        data[['Glon', 'Glat']] = data[['Glon', 'Glat']]\
            .sample(frac=1, replace=False).values
        logger.success("Position shuffled.")

    # The sky position data
    cluster_glon = data['Glon']
    cluster_glat = data['Glat']
    # the observed redshift from lightcone
    z = data['z']

    logger.success("Data loaded and preprocessed.")

    # Make data a global variable gives a 3 times speed up, according to
    # https://emcee.readthedocs.io/en/latest/tutorials/parallel/
    # Must be set BEFORE pool creation so MPI worker ranks inherit it.
    sampler_args = dict(X=X, Y=Y, X_sigma=X_sigma, Y_sigma=Y_sigma, z=z,
                        cluster_glon=cluster_glon, cluster_glat=cluster_glat,
                        texp=texp, conc=conc)

    pool = choose_pool(mpi=args.mpi, processes=args.ncores)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initial guess
    initial = np.array([
        DELTA_INIT, DGLON_INIT, DGLAT_INIT,
        LOGA_INIT, B_INIT, SINTR_INIT,
        KEXP_INIT,
        *([KCONC_INIT] if USE_CONC else [])])

    # MCMC setup
    ndim = 8 if USE_CONC else 7
    nwalkers = 64
    nchains = 8 if USE_CONC else 7    # independent chains (ZEUS/EMCEE is ensemble sampler,
    # different walker in same chain still communicate with each other)
    nsteps = NSTEPS
    # maximum steps (set this to a large value, since we use
    # convergence callback)
    pos0 = initial + 1e-3 * np.random.rand(nwalkers, ndim)

    flag_converge = -1 # -1 for no entry, 0 for unconverged, 1 for converged
    if SAMPLER == 'zeus':
        sampler = zeus.EnsembleSampler(
            nwalkers, ndim, log_likelihood_global, pool=pool)
        sampler.run_mcmc(pos0, nsteps)
        flat_samples = sampler.get_chain(flat=True, discard=0.5)  # type: ignore

    elif SAMPLER == 'emcee':
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_likelihood_global, pool=pool)
        sampler.run_mcmc(pos0, nsteps, progress=True)

        # Convergence test
        try:
            tau = sampler.get_autocorr_time()
            logger.success(
                "Chain converged. Autocorrelation time: {tau}", tau=tau)
            flag_converge = 1
        except emcee.autocorr.AutocorrError:
            logger.warning("Chain too short for reliable autocorrelation time.")
            flag_converge = 0

        # Get the samples, discard half
        flat_samples = sampler.get_chain(discard=nsteps//2, flat=True)
    else:
        raise Exception(f"""Sampler {SAMPLER} not supported. 
                        Currently supports: ['emcee', 'zeus'].""")
    
    logger.info("Flat samples of shape {}", np.shape(flat_samples))

    # Save the samples
    if SAMPLE_DIR is not None:
        sample_file = os.path.join(SAMPLE_DIR, 'chain.npy')
        np.save(sample_file, flat_samples)
        logger.success("Flat samples saved to {}", sample_file)

    # For delta and latitude (non-periodic): 16/50/84 quantiles
    lower_delta, delta, upper_delta = np.percentile(
        flat_samples[:, 0], [16, 50, 84])
    _fmt("delta", delta, lower_delta, upper_delta)

    lower_dglat, dglat, upper_dglat = np.percentile(
        flat_samples[:, 2], [16, 50, 84])
    _fmt("dglat", dglat, lower_dglat, upper_dglat)

    # Longitude is periodic
    dglon, dglon_err_lower, dglon_err_upper, lower_dglon, upper_dglon = (
        cf.periodic_error_range(flat_samples[:, 1], full_range=360)
    )
    logger.info(
        f"dglon: {lower_dglon:.4f} ~ {upper_dglon:.4f}"
        f"  ({dglon:.4f} -{dglon_err_lower:.4f} +{dglon_err_upper:.4f})"
    )

    # Save results
    output_file = os.path.join(OUTPUT_DIR, f"re{NUM_RELATION}.txt")
    mode = "a" if os.path.isfile(output_file) else "w"
    with open(output_file, mode) as f:
        if first_entry:
            f.write(
                "relation,\t"
                "delta,\tdelta_min,\tdelta_max,\t"
                "dglat,\tdglat_min,\tdglat_max,\t"
                "dglon,\tdglon_min,\tdglon_max\t"
                "converged\n"
            )
            first_entry = False
        f.write(
            f"{NUM_RELATION},\t"
            f"{delta:.4f},\t{lower_delta:.4f},\t{upper_delta:.4f},\t"
            f"{dglat:.4f},\t{lower_dglat:.4f},\t{upper_dglat:.4f},\t"
            f"{dglon:.4f},\t{lower_dglon:.4f},\t{upper_dglon:.4f},\t"
            f"{flag_converge}\n"
        )
    logger.success("Data written to {output_file}", output_file=output_file)
