import sys
sys.path.append('/home/yujiehe/anisotropy-flamingo')
import tools.clusterfit as cf
import numpy as np
import pandas as pd

# ----------------CONFIGURATION-------------------------------------------------
InputFile = '/data1/yujiehe/data/samples-lightcone0.csv'
OutputFile = '/data1/yujiehe/data/samples-lightcone0-clean.csv'
OutlierIDFile = '/data1/yujiehe/data/outliers.csv'

# Actually, no need to save the outliers. You can find them comparing the input
# output easily.
SaveOutliers = False

# ----------------------COMMAND LINE ARGUMENTS----------------------------------

# Command line arguments
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Find outliers in the sample. and remove them.")

# Add arguments
parser.add_argument('-i', '--input', type=str, help='Input file path')
parser.add_argument('-o', '--output', type=str, help='Output file path', default=OutputFile)
parser.add_argument('--outlier', type=str, help='Outlier ID file path', default=OutlierIDFile)

# Parse the arguments
args = parser.parse_args()
InputFile = args.input
OutputFile = args.output
OutlierIDFile = args.outlier

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

# --------------END CONFIGURATION-----------------------------------------------

print(f'Finding outlier in: {InputFile}')

ClusterData = pd.read_csv(InputFile)
AllOutlierIDs = np.array([])

for ScalingRelation in cf.CONST.keys():

    Nclusters = cf.CONST[ScalingRelation]['N'] # number of clusters we'd want

    _ = ScalingRelation.find('-')
    Y = ClusterData[cf.COLUMNS[ScalingRelation[:_  ]]]
    X = ClusterData[cf.COLUMNS[ScalingRelation[_+1:]]]
    z = ClusterData['ObservedRedshift']
    logY_ = cf.logY_(Y, z=z, relation=ScalingRelation)
    logX_ = cf.logX_(X, relation=ScalingRelation)

    BestFitParams, OutlierIDs = cf.fit(
        logY_, logX_, N=Nclusters, **FIT_RANGE[ScalingRelation],
        remove_outlier=True, id=ClusterData['TopLeafID'],
        )
    
    AllOutlierIDs = np.union1d(AllOutlierIDs, OutlierIDs)

# Save outlier IDs to file
if SaveOutliers:
    OutlierClusterData = ClusterData[ClusterData['TopLeafID'].isin(AllOutlierIDs)]
    OutlierClusterData.to_csv(OutlierIDFile, index=False)
    print(f'Outlier IDs saved: {OutlierIDFile}')

# Remove outliers and save to file
CleanClusterData = ClusterData[~ClusterData['TopLeafID'].isin(AllOutlierIDs)]
CleanClusterData.to_csv(OutputFile, index=False)
print(f'Cleaned data saved: {OutputFile}')