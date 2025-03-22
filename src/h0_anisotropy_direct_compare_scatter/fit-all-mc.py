# ---------------------------------------------------------------------------- #
# This script fits all scaling relations and calculates the uncertainties using
# bootstrapping. Output stuff to a csv file. This script particularly takes care
# of the mock instrumental scatter.
# 
# Author                       : Yujie He
# Created on (MM/YYYY)         : 02/2025
# Last Modified on (MM/YYYY)   : 02/2025
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                                     Setup                                    #
# ---------------------------------------------------------------------------- #


import sys
sys.path.append('/cosma/home/do012/dc-he4/anisotropy-flamingo/tools')
import clusterfit as cf
import numpy as np
import pandas as pd
import datetime
import os
import warnings

from numba import set_num_threads

Relations = ['LX-T', 'YSZ-T'] #'LX-YSZ', 'LX-M', 'YSZ-M'] # give the name of the relation to fit if you want to fit only one. Set to False if you want to fit all relations.

BootstrapSteps = 400 # number of bootstrap steps to calculate uncertainties

ScatterStepSize = 0.006
BStepSize       = 0.003
logAStepSize    = 0.002

FIT_RANGE = cf.LARGE_RANGE


# ---------------------------------------------------------------------------- #
#                            Command line arguments                            #
# ---------------------------------------------------------------------------- #


import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Fit scaling relations with bootstrapping.")

# Add arguments
parser.add_argument('-i', '--input', type=str, help='Input file path')
parser.add_argument('-o', '--output', type=str, help='Output file directory')
parser.add_argument('-t', '--threads', type=int, help='Number of threads', default=1)
parser.add_argument('-n', '--bootstrap', type=int, help='Number of bootstrap steps', default=BootstrapSteps)
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')

# Parse the arguments
args = parser.parse_args()
InputFile = args.input
OutputFileDir = args.output
Nthreads = args.threads
BootstrapSteps = args.bootstrap
OutputFilePrefix = os.path.join(OutputFileDir, 'bootstrap_all')
Overwrite = args.overwrite


# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #


if __name__ == '__main__':
    
    set_num_threads(Nthreads) # set core number here.

    ClusterData = pd.read_csv(InputFile)

    first_entry = True

    for ScalingRelation in cf.CONST_MC.keys():

        if ScalingRelation not in Relations:
            continue
        
        t = datetime.datetime.now()
        print(f'[{t}] Bootstrapping fitting: {ScalingRelation}')
        Nclusters = cf.CONST_MC[ScalingRelation]['N'] # number of clusters we'd want

        yname, xname = cf.parse_relation_name(ScalingRelation)
        Y = ClusterData[cf.COLUMNS_MC[yname]][:Nclusters].values
        X = ClusterData[cf.COLUMNS_MC[xname]][:Nclusters].values
        z = ClusterData['ObservedRedshift'][:Nclusters].values
        logY_ = cf.logY_(Y, z=z, relation=ScalingRelation)
        logX_ = cf.logX_(X, relation=ScalingRelation)

        # Load error
        eY = ClusterData['e'+cf.COLUMNS_MC[yname]][:Nclusters].values   # in ratio 0-1
        eX = ClusterData['e'+cf.COLUMNS_MC[xname]][:Nclusters].values
        scat_obs_Y = np.log10(1 + eY) 
        scat_obs_X = np.log10(1 + eX)

# --------------------------------- Best fit --------------------------------- #

        # Best fit
        BestFitParams = cf.run_fit(
            logY_, logX_, **FIT_RANGE[ScalingRelation],
            scat_step  = ScatterStepSize,
            B_step     = BStepSize,
            logA_step  = logAStepSize,
            scat_obs_Y = scat_obs_Y,
            scat_obs_X = scat_obs_X,
            )
        
        print('Best fit parameters:', BestFitParams)

# ------------------------ Bootstrap resampling error ------------------------ #

        print(f'Begin bootstrapping steps: {BootstrapSteps}')

        logA, B, scat = cf.bootstrap_fit(
            Nbootstrap = BootstrapSteps,
            logY_      = logY_,
            logX_      = logX_,
            Nclusters  = Nclusters,
            scat_step  = ScatterStepSize,
            B_step     = BStepSize,
            logA_step  = logAStepSize,
            scat_obs_X = scat_obs_X,
            scat_obs_Y = scat_obs_Y,
            **FIT_RANGE[ScalingRelation],
            )
        A = 10**logA # convert logA back to A

        t = datetime.datetime.now()
        print(f'[{t}] Bootstrapping fit finishied.')

# -------------------------- Get 1-sigma uncertainty ------------------------- #

        # 1 sigma uncertainty around the best fit
        BestFitA    = 10**BestFitParams['logA']
        BestFitB    = BestFitParams['B']
        BestFitScat = BestFitParams['scat']

        # Calculate the +- 34th percentile from the best fit value, as in M20, M21.
        BestFitAPer = np.sum(A < BestFitA) / len(A) * 100
        BestFitBPer = np.sum(B < BestFitB) / len(B) * 100
        BestFitScatPer = np.sum(scat < BestFitScat) / len(scat) * 100

        if BestFitAPer < 34:
            warnings.warn(f'BestFitAPer={BestFitAPer}. LowerBoundA out of bounds. Setting to min.')
            BestFitAPer = 34
        if BestFitAPer > 66:
            warnings.warn(f'BestFitAPer={BestFitAPer}. UpperBoundA out of bounds. Setting to max.')
            BestFitAPer = 66
        
        if BestFitBPer < 34:
            warnings.warn(f'BestFitBPer={BestFitBPer}. LowerBoundB out of bounds. Setting to min.')
            BestFitBPer = 34
        if BestFitBPer > 66:
            warnings.warn(f'BestFitBPer={BestFitBPer}. UpperBoundB out of bounds. Setting to max.')
            BestFitBPer = 66

        LowerBoundA = np.percentile(A, BestFitAPer - 34)
        UpperBoundA = np.percentile(A, BestFitAPer + 34)
        LowerBoundB = np.percentile(B, BestFitBPer - 34)
        UpperBoundB = np.percentile(B, BestFitBPer + 34)

        # The range of scatter we don't care that much. But, do give a warning here
        if BestFitScatPer > 34:
            LowerBoundScat = np.percentile(scat, BestFitScatPer - 34)
        else:
            LowerBoundScat = np.percentile(scat, 0)
            warnings.warn(f'BestFitScatPer={BestFitScatPer}. LowerBoundScat out of bounds. Setting to min.')

        if BestFitScatPer < 66:
            UpperBoundScat = np.percentile(scat, BestFitScatPer + 34)
        else:
            UpperBoundScat = np.percentile(scat, 100) 
            warnings.warn(f'BestFitScatPer={BestFitScatPer}. UpperBoundScat out of bounds. Setting to max.')

        print(f'1-sigma bootstrapping uncertainty of {ScalingRelation} fit:')
        print(f'A: {BestFitA:.3f} + {UpperBoundA-BestFitA:.3f} - {BestFitA-LowerBoundA:.3f}')
        print(f'B: {BestFitB:.3f} + {UpperBoundB-BestFitB:.3f} - {BestFitB-LowerBoundB:.3f}')
        print(f'IntrinsicScatter: {BestFitScat:.3f} + {UpperBoundScat-BestFitScat:.3f} - {BestFitScat-LowerBoundScat:.3f}')

        #save the best fit and uncertainties to a csv
        BestFitOutputFile = os.path.join(OutputFileDir, 'fit_all_mc_scatter.csv')

        df = pd.DataFrame(
            {
                'Relation': [ScalingRelation],
                'BestFitA': [BestFitA],
                'BestFitB': [BestFitB],
                'BestFitScat': [BestFitScat],
                '1SigmaUpperA': [UpperBoundA],
                '1SigmaLowerA': [LowerBoundA],
                '1SigmaUpperB': [UpperBoundB],
                '1SigmaLowerB': [LowerBoundB],
                '1SigmaUpperScat': [UpperBoundScat],
                '1SigmaLowerScat': [LowerBoundScat],
            })

        # Check if file exists
        if not first_entry:
            # Append to existing file
            df.to_csv(BestFitOutputFile, mode='a', header=False, index=False)
        else:
            # Write new file with header
            df.to_csv(BestFitOutputFile, index=False)
            first_entry = False