# ---------------------------------------------
# This script fits all scaling relations and
# calculates the uncertainties using bootstrapping.
# Output stuff to a csv file.
# 
# Author                       : Yujie He
# Created on (MM/YYYY)         : 03/2024
# Last Modified on (MM/YYYY)   : 09/2024
# ---------------------------------------------


import sys
sys.path.append('/cosma/home/do012/dc-he4/anisotropy-flamingo/tools')
import clusterfit as cf
import numpy as np
import pandas as pd
import datetime
import os
import warnings

from numba import set_num_threads

# --------------------------------CONFIGURATION---------------------------------
InputFile = '/data1/yujiehe/data/samples_in_lightcone0_with_trees_duplicate_excision_outlier_excision.csv'
OutputFileDir = '/data1/yujiehe/data/fits/'

Nthreads = 4

Relations = ['LX-T', 'YSZ-T', 'M-T',] #'LX-YSZ', 'LX-M', 'YSZ-M'] # give the name of the relation to fit if you want to fit only one. Set to False if you want to fit all relations.
BootstrapSteps = 500 # number of bootstrap steps to calculate uncertainties
ScatterStepSize = 0.003
BStepSize       = 0.002
logAStepSize    = 0.002

# define now the range of parameters within which to fit, change this anytime 
FIT_RANGE = cf.MID_RANGE
# you want. But keep in mind that larger range means longer time to fit. We fixed
# step size instead of number of steps to control accuracy over computation time.

# -------------------------COMMAND LINE ARGUMENTS-------------------------------
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Fit scaling relations with bootstrapping.")

# Add arguments
parser.add_argument('-i', '--input', type=str, help='Input file path')
parser.add_argument('-o', '--output', type=str, help='Output file directory', default=OutputFileDir)
parser.add_argument('-t', '--threads', type=int, help='Number of threads', default=Nthreads)
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

# ------------------------------------MAIN--------------------------------------
if __name__ == '__main__':
    
    set_num_threads(Nthreads) # set core number here.

    ClusterData = pd.read_csv(InputFile)


    for ScalingRelation in cf.CONST.keys():

        if ScalingRelation not in Relations:
            continue
        
        t = datetime.datetime.now()
        print(f'[{t}] Bootstrapping fitting: {ScalingRelation}')
        Nclusters = cf.CONST[ScalingRelation]['N'] # number of clusters we'd want

        _ = ScalingRelation.find('-')
        Y = ClusterData[cf.COLUMNS[ScalingRelation[:_  ]]][:Nclusters].values
        X = ClusterData[cf.COLUMNS[ScalingRelation[_+1:]]][:Nclusters].values
        z = ClusterData['ObservedRedshift'][:Nclusters].values
        logY_ = cf.logY_(Y, z=z, relation=ScalingRelation)
        logX_ = cf.logX_(X, relation=ScalingRelation)

        # Best fit
        BestFitParams = cf.run_fit(
            logY_, logX_, **FIT_RANGE[ScalingRelation],
            scat_step  = ScatterStepSize,
            B_step     = BStepSize,
            logA_step  = logAStepSize,
            )
        
        print('Best fit parameters:', BestFitParams)

        # Bootstrapping              #TODO save best fit and uncertainties to another file
        OutputFile = f'{OutputFilePrefix}_{ScalingRelation}.csv'

        if os.path.exists(OutputFile) and not Overwrite:
            print(f'File exists: {OutputFile}')

            df = pd.read_csv(OutputFile) # skip bootstrapping but calculate the error nonetheless
            A  = df['A']
            B  = df['B']
            scat = df['TotalScatter']
        else:
            print(f'Begin bootstrapping steps: {BootstrapSteps}')

            logA, B, scat = cf.bootstrap_fit(
                Nbootstrap = BootstrapSteps,
                logY_      = logY_,
                logX_      = logX_,
                Nclusters  = Nclusters,
                scat_step  = ScatterStepSize,
                B_step     = BStepSize,
                logA_step  = logAStepSize,
                **FIT_RANGE[ScalingRelation],
                )
            A = 10**logA # convert logA back to A

            pd.DataFrame(
                {'A': A, 'B': B, 'TotalScatter': scat,}).to_csv(OutputFile, index=False
                )
            t = datetime.datetime.now()
            print(f'[{t}] Bootstrapping fit finishied: {OutputFile}')

        # 1 sigma uncertainty around the best fit
        BestFitA    = 10**BestFitParams['logA']
        BestFitB    = BestFitParams['B']
        BestFitScat = BestFitParams['scat']

        # Calculate the +- 34th percentile from the best fit value, as in M20, M21.
        BestFitAPer = np.sum(A < BestFitA) / len(A) * 100
        BestFitBPer = np.sum(B < BestFitB) / len(B) * 100
        BestFitScatPer = np.sum(scat < BestFitScat) / len(scat) * 100

        LowerBoundA    = np.percentile(A, BestFitAPer - 34)
        UpperBoundA    = np.percentile(A, BestFitAPer + 34)
        LowerBoundB    = np.percentile(B, BestFitBPer - 34)
        UpperBoundB    = np.percentile(B, BestFitBPer + 34)

        # The range of scatter we don't care that much. But, do give a warning here
        if BestFitScatPer > 34:
            LowerBoundScat = np.percentile(scat, BestFitScatPer - 34)
        else:
            LowerBoundScat = np.percentile(scat, 0)
            warnings.warn(f'LowerBoundScat out of bounds at {LowerBoundScat}. Setting to 0.')

        if BestFitScatPer < 66:
            UpperBoundScat = np.percentile(scat, BestFitScatPer + 34)
        else:
            UpperBoundScat = np.percentile(scat, 100)
            warnings.warn(f'UpperBoundScat out of bounds at {UpperBoundScat}. Setting to 0.')


        print(f'1-sigma bootstrapping uncertainty of {ScalingRelation} fit:')
        print(f'A: {BestFitA:.3f} + {UpperBoundA-BestFitA:.3f} - {BestFitA-LowerBoundA:.3f}')
        print(f'B: {BestFitB:.3f} + {UpperBoundB-BestFitB:.3f} - {BestFitB-LowerBoundB:.3f}')
        print(f'TotalScatter: {BestFitScat:.3f} + {UpperBoundScat-BestFitScat:.3f} - {BestFitScat-LowerBoundScat:.3f}')



        #save the best fit and uncertainties to a csv
        BestFitOutputFile = os.path.join(OutputFileDir, 'fit_all.csv')

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
        if os.path.isfile(BestFitOutputFile):
            # Append to existing file
            df.to_csv(BestFitOutputFile, mode='a', header=False, index=False)
        else:
            # Write new file with header
            df.to_csv(BestFitOutputFile, index=False)