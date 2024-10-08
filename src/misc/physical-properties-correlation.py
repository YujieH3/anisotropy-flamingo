import sys
sys.path.append('../tools/')
import clusterfit as cf
import numpy as np
from numba import njit, prange, set_num_threads
import pandas as pd
import datetime

# --------------------------------CONFIGURATION---------------------------------

lc = 0
input_file = f'../data/samples_in_lightcone{lc}_with_trees_duplicate_excision_outlier_excision.csv'
output_dir = '../data/fits'

n_threads = 20
relations = ['LX-T', 'YSZ-T', 'M-T', 'LX-YSZ', 'LX-M', 'YSZ-M'] # pick from 'LX-T', 'M-T', 'LX-YSZ', 'LX-M', 'YSZ-M', 'YSZ-T'
n_bootstrap = 10000 # number of bootstrapping for each direction

# Set the parameter space
FIT_RANGE = cf.FIVE_MAX_RANGE

# Set the step size or number of steps for A
B_step    = 0.003
n_B_steps = 150

# And B
logA_step    = 0.003
n_logA_steps = 150

# Set the step size for the total scatter. The scatter is on the outer loop so 
# number of steps is undefinable, depends on when can chi2 reduced ~1. 
scat_step = 0.007

cone_size = 60

# ------------------------------------------------------------------------------

import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Physical properties correlation')

# Add arguments
parser.add_argument('-i', '--input', type=str, help='Input file path', default=input_file)
parser.add_argument('-o', '--output', type=str, help='Output directory', default=output_dir)
parser.add_argument('-n', '--threads', type=int, help='Number of threads', default=n_threads)
parser.add_argument('-s', '--conesize', type=int, help='Cone size in degrees', default=cone_size)

# Parse the arguments
args = parser.parse_args()
input_file = args.input
output_dir = args.output
n_threads = args.threads
cone_size = args.conesize

# Find the lightcone number from the input file name. This is useful when you need different things for two lightcones.
loc = input_file.find('lightcone')
lc = input_file[loc+9]
lc = int(lc)


# 
@njit(fastmath=True, parallel=True)
def nbloops(logY_, logX_, n_cone, scaling_relation,
            sample_T, sample_LX, sample_YSZ, sample_M,
            sample_LcoreLtot, sample_flux, sample_z_obs,
            logA_min, logA_max, B_min, B_max, scat_min, scat_max,
            n_bootstrap=n_bootstrap, scat_step=scat_step, B_step=B_step, logA_step=logA_step):
    logA      = np.empty(n_bootstrap)  # logA distribution
    B         = np.empty(n_bootstrap)  # B distribution
    scat      = np.empty(n_bootstrap)  # scatter distribution
    T         = np.empty(n_bootstrap)  # chi2 distribution
    LX        = np.empty(n_bootstrap)  # LX distribution
    YSZ       = np.empty(n_bootstrap)  # YSZ distribution
    M         = np.empty(n_bootstrap)  # M distribution
    LcoreLtot = np.empty(n_bootstrap)  # Lcore/Ltot distribution
    logflux   = np.empty(n_bootstrap)  # flux distribution
    z_obs     = np.empty(n_bootstrap)  # observed redshift distribution

    sample_logflux = np.log10(sample_flux) # Use log average for flux

    for i in prange(n_bootstrap):
        #idx = np.random.choice(n_cone, size=n_cone, replace=True) # !!!!!!!!!!!!THIS IS VERY WRONG!!!!!!
        idx = np.random.choice(len(logY_), size=n_cone, replace=True)
        bootstrap_logY_  = logY_[idx]
        bootstrap_logX_  = logX_[idx]

        params = cf.run_fit(
            logY_          = bootstrap_logY_,
            logX_          = bootstrap_logX_,
            scat_step      = scat_step,
            B_step         = B_step,
            logA_step      = logA_step,
            logA_min = logA_min, logA_max = logA_max,
            B_min = B_min, B_max = B_max,
            scat_min = scat_min, scat_max = scat_max
            )
        
        logA[i]      = params['logA']
        B[i]         = params['B']
        scat[i]      = params['scat']
        T[i]         = np.mean(sample_T[idx])
        LcoreLtot[i] = np.mean(sample_LcoreLtot[idx])
        logflux[i]   = np.mean(sample_logflux[idx])
        z_obs[i]     = np.mean(sample_z_obs[idx])
        LX[i]        = np.mean(sample_LX[idx])
        YSZ[i]       = np.mean(sample_YSZ[idx])
        M[i]         = np.mean(sample_M[idx])
    
    return logA, B, scat, T, LX, YSZ, M, LcoreLtot, logflux, z_obs

if __name__ == '__main__':
    set_num_threads(n_threads)

    cluster_data = pd.read_csv(input_file)

    t00 = datetime.datetime.now()
    print(f'[{t00}] Begin scanning: {relations} in {cone_size}.')
    print(f'Threads: {n_threads}')

    for scaling_relation in cf.CONST.keys():

        if scaling_relation not in relations:
            continue

        if n_B_steps is not None: # set the step size for A and B if the number of steps is given
            B_step = (FIT_RANGE[scaling_relation]['B_max'] - FIT_RANGE[scaling_relation]['B_min']) / n_B_steps
        if n_logA_steps is not None:
            logA_step = (FIT_RANGE[scaling_relation]['logA_max'] - FIT_RANGE[scaling_relation]['logA_min']) / n_logA_steps

        # Prepare the data, convert to logX_, logY_. Requires redshift for logY_
        t0 = datetime.datetime.now()
        print(f'[{t0}] Running: {scaling_relation}')
        n_clusters = cf.CONST[scaling_relation]['N']

        data_cut = cluster_data[:n_clusters] # only use the first N clusters

        _ = scaling_relation.find('-')
        Y = data_cut[cf.COLUMNS[scaling_relation[:_  ]]]
        X = data_cut[cf.COLUMNS[scaling_relation[_+1:]]]
        z = data_cut['ObservedRedshift']
        Y = np.array(Y)
        X = np.array(X)
        z = np.array(z)

        logY_ = cf.logY_(Y, z=z, relation=scaling_relation)
        logX_ = cf.logX_(X, relation=scaling_relation)

        # Average number of clusters in a cone, calculated by integration
        n_cone = n_clusters * 1/2 * (1 - np.cos(cone_size*np.pi/180))
        n_cone = round(n_cone)
#        if cone_size == 60:
#            n_cone = round(n_clusters / 4)
#        else:
#            raise ValueError('Only support 60 deg cone size for now.')

        sample_T         = np.array(data_cut[cf.COLUMNS['T']])
        sample_LX        = np.array(data_cut[cf.COLUMNS['LX']])
        sample_YSZ       = np.array(data_cut[cf.COLUMNS['YSZ']])
        sample_M         = np.array(data_cut[cf.COLUMNS['M']])
        if lc == 0:
            sample_LcoreLtot = np.array(data_cut['2DLcore/Ltot'])
        elif lc == 1:
            sample_LcoreLtot = np.array(data_cut['3DLcore/Ltot'])
        sample_flux      = np.array(data_cut['Flux'])
        sample_z_obs     = np.array(data_cut['ObservedRedshift'])

        logA, B, scat, T, LX, YSZ, M, LcoreLtot, logflux, z_obs = nbloops(
            logY_, logX_, n_cone, scaling_relation=scaling_relation,
            sample_T=sample_T, sample_LX=sample_LX, sample_YSZ=sample_YSZ, sample_M=sample_M,
            sample_LcoreLtot=sample_LcoreLtot, sample_flux=sample_flux, sample_z_obs=sample_z_obs,
            n_bootstrap=n_bootstrap, scat_step=scat_step, B_step=B_step, logA_step=logA_step,
            **FIT_RANGE[scaling_relation]
        )

        # Save the results
        output_file = f'{output_dir}/physical-properties-corr-{scaling_relation}-{cone_size}deg-lc{lc}.csv'
        results = pd.DataFrame({
            'logA'            : logA,
            'B'               : B,
            'scat'            : scat,
            'T'               : T,
            'LX'              : LX,
            'YSZ'             : YSZ,
            'Mgas'            : M,
            'Lcore/Ltot'      : LcoreLtot,
            'logFlux'         : logflux,
            'ObservedRedshift': z_obs,
        })
        results.to_csv(output_file, index=False)
