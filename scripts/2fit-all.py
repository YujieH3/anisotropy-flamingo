import sys
sys.path.append('/home/yujiehe/anisotropy-flamingo')
import tools.clusterfit as cf
import numpy as np
import pandas as pd
import datetime
import os

from numba import set_num_threads

# --------------------------------CONFIGURATION---------------------------------
# NOTE outliers should be removed from the dataset before running the script.
InputFile = '/data1/yujiehe/data/samples-lightcone0-clean.csv'
OutputFilePrefix = '/data1/yujiehe/data/fits/bootstrap'

Nthreads = 4

Overwrite = False # if True, overwrite existing files

Relations = ['LX-T', 'YSZ-T', 'M-T', 'LX-YSZ', 'LX-M', 'YSZ-M'] # give the name of the relation to fit if you want to fit only one. Set to False if you want to fit all relations.
BootstrapSteps = 500 # number of bootstrap steps to calculate uncertainties
ScatterStepSize = 0.002
BStepSize       = 0.002
logAStepSize    = 0.002

# define now the range of parameters within which to fit, change this anytime 
FIT_RANGE = cf.MID_RANGE
# you want. But keep in mind that larger range means longer time to fit. We fixed
# step size instead of number of steps to control accuracy over computation time.

# ------------------------------------MAIN--------------------------------------

set_num_threads(Nthreads) # set core number here.

ClusterData = pd.read_csv(InputFile)


for ScalingRelation in cf.CONST.keys():

    if ScalingRelation not in Relations:
        continue
    
    t = datetime.datetime.now()
    print(f'[{t}] Bootstrapping fitting: {ScalingRelation}')
    Nclusters = cf.CONST[ScalingRelation]['N'] # number of clusters we'd want

    _ = ScalingRelation.find('-')
    Y = ClusterData[cf.COLUMNS[ScalingRelation[:_  ]]][:Nclusters]
    X = ClusterData[cf.COLUMNS[ScalingRelation[_+1:]]][:Nclusters]
    z = ClusterData['ObservedRedshift'][:Nclusters]
    logY_ = cf.logY_(Y, z=z, relation=ScalingRelation)
    logX_ = cf.logX_(X, relation=ScalingRelation)

    # Best fit
    BestFitParams = cf.run_fit(
        logY_, logX_, **FIT_RANGE[ScalingRelation],
        scat_step  = ScatterStepSize,
        B_step     = BStepSize,
        logA_step  = logAStepSize,
        weight     = 1,
        )
    
    print('Best fit parameters:', BestFitParams)

    # Bootstrapping
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
            weight     = None,
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


    # 1 sigma uncertainty
    LowerBoundA    = np.percentile(A, 16)
    UpperBoundA    = np.percentile(A, 84)
    LowerBoundB    = np.percentile(B, 16)
    UpperBoundB    = np.percentile(B, 84)
    LowerBoundScat = np.percentile(scat, 16)
    UpperBoundScat = np.percentile(scat, 84)

    BestFitA    = 10**BestFitParams['logA']
    BestFitB    = BestFitParams['B']
    BestFitScat = BestFitParams['scat']

    print(f'1-sigma bootstrapping uncertainty of {ScalingRelation} fit:')
    print(f'A: {BestFitA:.3f} + {UpperBoundA-BestFitA:.3f} - {BestFitA-LowerBoundA:.3f}')
    print(f'B: {BestFitB:.3f} + {UpperBoundB-BestFitB:.3f} - {BestFitB-LowerBoundB:.3f}')
    print(f'TotalScatter: {BestFitScat:.3f} + {UpperBoundScat-BestFitScat:.3f} - {BestFitScat-LowerBoundScat:.3f}')

print('All done!')  