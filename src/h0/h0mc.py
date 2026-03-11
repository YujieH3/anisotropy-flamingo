#!/usr/bin/env python
# mpiexec -n <number of cores> python h0mc.py <input.csv> <output.csv>
# srun does not allocate the correct number of cores. mpiexec by itself does not get cores from the cluster.
# Currently only works in a batch job script.
"""
Author                 : Yujie He
Created on (MM/YYYY)   : 06/2024
Description:
    This script calculates the H0 variation using MCMC, MPI parallelised.
Recent changes:
    - use COLUMNS_MC instead of COLUMNS to use raw SOAP spectroscopic-like
    core-excised temperature instead of Chandra temperature.
    - use zeus mcmc instead of emcee for faster inference.
"""

import argparse
import os
import sys
sys.path.append("/cosma/home/do012/dc-he4/anisotropy-flamingo/tools")

# from scipy.optimize import differential_evolution
import pandas as pd
import numpy as np
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=68.1, Om0=0.306, Ob0=0.0486)       # FLAMINGO cosmology

import clusterfit as cf


def log_likelihood(theta, X, Y, z_obs, phi_lc, theta_lc, yname, xname):
    """
    X, Y, z_obs, phi_lc, theta_lc, are from the data
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    (
        delta,
        vlon,
        vlat,
        logA,
        B,
        sigma,
    ) = theta

    # Set the scaling relation to know the pivot point
    scaling_relation = f"{yname}-{xname}"

    # Anglular separation
    angle = cf.angular_separation(phi_lc, theta_lc, vlon, vlat) * np.pi / 180

    # vary H0; since H0 enters distance to a simply *H0^(some power), we can just multiply the distance by a factor and not do the integral twice
    H0_ratio = 1 + delta * np.cos(angle)

    # Modified Y
    if yname == "LX":
        Y_mod = Y * H0_ratio**-2
    elif yname == "YSZ":
        Y_mod = Y * H0_ratio**-2
    elif yname == "M":
        Y_mod = Y * H0_ratio ** (-5 / 2)  # (DA_modified)**(5/2)/(DA_default)**(5/2)
    else:
        raise ValueError(f"Name of Y {yname} not supported. Supported: LX, YSZ, M.")

    # To our fit parameters
    logY_ = cf._logY_(
        Y_mod,
        z=z_obs,
        CY=cf.CONST_MC[scaling_relation]["CY"],
        gamma=cf.CONST_MC[scaling_relation]["gamma"],
    )
    logX_ = cf._logX_(X, CX=cf.CONST_MC[scaling_relation]["CX"])

    model = B * logX_ + logA
    lnL = -0.5 * np.sum(
        (logY_ - model) ** 2 / (sigma**2) + np.log(sigma**2)
    )  # Kostas' implementation

    return lnL + lp


# set prior
def log_prior(theta):
    # A large flat prior for now
    delta, vlon, vlat, logA, B, sigma = theta  # 6 parameters

    # If in range, p(theta)=1, else p(theta)=0
    if (
        -1 < logA < 1
        and 0.5 < B < 3.5
        and 0.01 < sigma < 1
        and 0 < delta < 1
        and -180 < vlon < 180
        and -90 < vlat < 90
    ):
        return 0.0
    else:
        return -np.inf



if __name__ == '__main__':
    print('starting')
    
    # Relations to fit
    RELATIONS = ["LX-T", "YSZ-T", "M-T"]  # pick from 'LX-T', 'M-T', 'YSZ-T'
    
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output = os.path.join(script_dir, "output.csv")     # the script directory

    # Create the parser
    parser = argparse.ArgumentParser(
        description="Calculate the H0 variation using MCMC."
    )

    # Positional arguments
    parser.add_argument("input", type=str, help="Input file")
    parser.add_argument("output", type=str, nargs='?', help="Output file (default: script directory)", default=default_output)

    # Optional arguments
    parser.add_argument(
        "-c", "--savesamples", type=str, help="Directory to save chain", default=None
        )
    parser.add_argument('--sampler', type=str, default='zeus', 
                       choices=['zeus', 'emcee'],
                       help='MCMC sampler to use')
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing", default=True
        )

    # Parse the arguments
    args = parser.parse_args()
    INPUT_FILE = args.input
    OUTPUT_FILE = args.output
    SAMPLE_DIR = args.savesamples
    OVERWRITE = args.overwrite
    SAMPLER = args.sampler

    # Load data
    data = pd.read_csv(INPUT_FILE)

    # Skip the script entirely if output file exists and overwrite is set to none.
    if os.path.exists(OUTPUT_FILE) and not OVERWRITE:
        print(f"File exists: {OUTPUT_FILE}")
        raise FileExistsError(f"Output file {OUTPUT_FILE} already exists and OVERWRITE is False.")

    first_entry = True

    for scaling_relation in RELATIONS:

        n_clusters = cf.CONST_MC[scaling_relation]["N"]

        # Load the data
        _ = scaling_relation.find("-")
        yname = scaling_relation[:_]
        xname = scaling_relation[_ + 1 :]
        Y = np.array(data[cf.COLUMNS_MC[yname]][:n_clusters])
        X = np.array(data[cf.COLUMNS_MC[xname]][:n_clusters])
        # Also load the position data
        phi_lc = np.array(data["phi_on_lc"][:n_clusters])
        theta_lc = np.array(data["theta_on_lc"][:n_clusters])
        # the observed redshift from lightcone
        z_obs = np.array(data["ObservedRedshift"][:n_clusters])


        # scipy estimation of staring point
        initial = np.array([0.1, 0, 0, 0.5, 1, 0.1])  # initial guess

        # MCMC setup
        ndim = 6        # number of parameters, fixed
        nwalkers = 48   # 
        nchains = 4     # independent chains (ZEUS/EMCEE is ensemble sampler, 
                        # different walker in same chain still communicate with each other)
        nsteps = 10_000  # maximum steps (set this to a large value, since we use convergence callback)

        pos0 = initial + 1e-2 * np.random.randn(nwalkers, ndim) # soln.x + ... if using scipy evolution

        if SAMPLER == 'zeus':
            import zeus
            from zeus import ChainManager

            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            print(f"Rank {comm.Get_rank()} of {comm.Get_size()}", flush=True)

            with ChainManager(nchains) as cm:
                rank = cm.get_rank

                # cb = zeus.callbacks.ParallelSplitRCallback(epsilon=0.01, chainmanager=cm) # Callbacks bugs out, doesn't really work
                sampler = zeus.EnsembleSampler(nwalkers, 
                                            ndim, 
                                            log_likelihood, 
                                            args=(X, Y, z_obs, phi_lc, theta_lc, yname, xname), 
                                            pool=cm.get_pool)
                sampler.run_mcmc(pos0, nsteps)#, callbacks=cb)
                flat_samples = sampler.get_chain(flat=True, discard=0.5)  # type: ignore # burn first half

                # Save the samples
                if SAMPLE_DIR is not None:
                    sample_file = os.path.join(SAMPLE_DIR, f'chain_{scaling_relation}_rank{rank}.npy')
                    np.save(sample_file, flat_samples)
                
        elif SAMPLER == 'emcee':
            import emcee
            from schwimmbad import MPIPool

            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            print(f"Rank {comm.Get_rank()} of {comm.Get_size()}", flush=True)

            # Create a sampler
            with MPIPool() as pool:
                if not pool.is_master():
                    pool.wait()
                    sys.exit(0)

                sampler = emcee.EnsembleSampler(
                    nwalkers,
                    ndim,
                    log_likelihood,
                    args=(X, Y, z_obs, phi_lc, theta_lc, yname, xname),
                    pool=pool,
                )

                sampler.run_mcmc(
                    pos0, nsteps, progress=False
                )

                # TODO add save chain 

            # Small convergence test
            try:
                tau = sampler.get_autocorr_time()
                print(tau)
            except emcee.autocorr.AutocorrError:
                print("The chain is too short to get a reliable autocorrelation time.")
                tau = 0

            # Get the samples
            flat_samples = sampler.get_chain(discard=nsteps//2, flat=True)
            

        else:
            raise ValueError(f"Sampler {SAMPLER} not supported. Currently supports: ['emcee', 'zeus'].")



        # Result processing and saving
        print(flat_samples.shape)

        # For delta we use the 16, 50, 84 quantiles
        delta_distr = flat_samples[:, 0]
        lower_delta = np.percentile(delta_distr, 16)
        median_delta = np.percentile(delta_distr, 50)
        upper_delta = np.percentile(delta_distr, 84)

        # For saving
        delta = median_delta
        delta_err_lower = median_delta - lower_delta
        delta_err_upper = upper_delta - median_delta
        print(
            f"delta: {lower_delta} ~ {upper_delta} \nor {delta} -{delta_err_lower} +{delta_err_upper}"
        )

        # Latitude is not periodic
        vlat_distr = flat_samples[:, 2]
        lower_vlat = np.percentile(vlat_distr, 16)
        median_vlat = np.percentile(vlat_distr, 50)
        upper_vlat = np.percentile(vlat_distr, 84)

        # For saving
        vlat = median_vlat
        vlat_err_lower = median_vlat - lower_vlat
        vlat_err_upper = upper_vlat - median_vlat
        print(
            f"vlat: {lower_vlat} ~ {upper_vlat} \nor {vlat} -{vlat_err_lower} +{vlat_err_upper}"
        )

        # Find the range w.r.t. the peak.
        vlon, vlon_err_lower, vlon_err_upper, lower_vlon, upper_vlon = (
            cf.periodic_error_range(flat_samples[:, 1], full_range=360)
        )
        print(
            f"vlon: {lower_vlon} ~ {upper_vlon} \nor {vlon} -{vlon_err_lower} +{vlon_err_upper}"
        )

        # Save the best fit parameters
        if first_entry:
            mode = "w"
        else:
            mode = "a"

        # Write line by line
        with open(OUTPUT_FILE, mode) as f:

            # Write the header on first entry
            if first_entry:
                f.write(
                    "scaling_relation,delta,delta_err_lower,delta_err_upper,vlon,vlon_err_lower,vlon_err_upper,vlat,vlat_err_lower,vlat_err_upper\n"
                )
                first_entry = False

            # Write the data
            f.write(
                f"{scaling_relation},{delta},{delta_err_lower},{delta_err_upper},{vlon},{vlon_err_lower},{vlon_err_upper},{vlat},{vlat_err_lower},{vlat_err_upper}\n"
            )
