# ---------------------------------------------
# This script finds outliers in the sample and 
# removes them. The outliers are found by fitting
# the scaling relations and removing the points
# that are too far (4 sigma away) from the best fit.
#     - updated to fit and clean (remove outlier) 
# for all clusters instead of first N highest 
# Lcore/Ltot ones.
#
# Author                       : Yujie He
# Created on (MM/YYYY)         : 03/2024
# Last Modified on (MM/YYYY)   : 09/2024
# ---------------------------------------------

import sys
sys.path.append('../tools')
import clusterfit as cf
import numpy as np
import pandas as pd

# ----------------CONFIGURATION-------------------------------------------------
InputFile = '/data1/yujiehe/data/samples-lightcone0.csv'
OutputFile = '/data1/yujiehe/data/samples-lightcone0-clean.csv'

# ----------------------COMMAND LINE ARGUMENTS----------------------------------

# Command line arguments
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Find outliers in the sample. and remove them.")

# Add arguments
parser.add_argument('-i', '--input', type=str, help='Input file path')
parser.add_argument('-o', '--output', type=str, help='Output file path', default=OutputFile)

# Parse the arguments
args = parser.parse_args()
InputFile = args.input
OutputFile = args.output

# define now the range of parameters within which to fit, change this anytime 
# you want. But keep in mind that larger range means longer time to fit
FIT_RANGE = { # matching the variable names in fit function.
    'LX-T': {
        'B_min'   : 1.5,  'B_max'   : 3,
        'logA_min': -0.1, 'logA_max': 1,
        'scat_min': 0.1,  'scat_max': 1,
    },
    'YSZ-T': {
        'B_min'   : 2,    'B_max'   : 3.5,
        'logA_min': -1,   'logA_max': 1,
        'scat_min': 0.01, 'scat_max': 1,
    },
    'M-T': {
        'B_min'   : 1,    'B_max'   : 2.5,
        'logA_min': -1,   'logA_max': 1,
        'scat_min': 0.01, 'scat_max': 1,
    },
    'LX-YSZ': {
        'B_min'   : 0.2,  'B_max'   : 2,
        'logA_min': -1,   'logA_max': 1,
        'scat_min': 0.01, 'scat_max': 1,
    },
    'LX-M': {
        'B_min'   : 0.5,  'B_max'   : 2,
        'logA_min': -1,   'logA_max': 1,
        'scat_min': 0.01, 'scat_max': 1,
    },
    'YSZ-M': {
        'B_min'   : 0.5,  'B_max'   : 2.5,
        'logA_min': -1,   'logA_max': 1,
        'scat_min': 0.01, 'scat_max': 1,
    },
}

RELATIONS = ['LX-T', 'YSZ-T', 'M-T']

# --------------END CONFIGURATION-----------------------------------------------

print(f'Finding outlier in: {InputFile}')

ClusterData = pd.read_csv(InputFile)
AllOutlierIDs = np.array([])

for ScalingRelation in RELATIONS:

    #Nclusters = cf.CONST[ScalingRelation]['N'] # number of clusters we'd want

    _ = ScalingRelation.find('-')
    Y = ClusterData[cf.COLUMNS[ScalingRelation[:_  ]]].values
    X = ClusterData[cf.COLUMNS[ScalingRelation[_+1:]]].values
    z = ClusterData['ObservedRedshift'].values
    logY_ = cf.logY_(Y, z=z, relation=ScalingRelation)
    logX_ = cf.logX_(X, relation=ScalingRelation)

    Nclusters = len(logY_)  # fit all instead of first N highest Lcore/Ltot

    BestFitParams, OutlierIDs = cf.fit(
        logY_, logX_, N=Nclusters, **FIT_RANGE[ScalingRelation],
        remove_outlier=True, id=ClusterData['TopLeafID'],
        )
    
    AllOutlierIDs = np.union1d(AllOutlierIDs, OutlierIDs)

# Remove outliers and save to file
CleanClusterData = ClusterData[~ClusterData['TopLeafID'].isin(AllOutlierIDs)]
CleanClusterData.to_csv(OutputFile, index=False)
print(f'Outliers removed in total: {len(AllOutlierIDs)}')
print(f'Cleaned data: {len(CleanClusterData)}')