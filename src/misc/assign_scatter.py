# ---------------------------------------------------------------------------- #
#                            Command line arguments                            #
# ---------------------------------------------------------------------------- #


import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Calculate significance map for best fit scans.")

# Add arguments
parser.add_argument('-i', '--input', type=str, help='Input file')
parser.add_argument('-o', '--output', type=str, help='Output file')
parser.add_argument('-f', '--fit_all_file', type=str, help='Best fit file fit-all.csv')

# Parse the arguments
args = parser.parse_args()
INPUT_FILE   = args.input
FIT_ALL_FILE = args.fit_all_file
OUTPUT_FILE  = args.output

# ---------------------------------------------------------------------------- #
#                                     Setup                                    #
# ---------------------------------------------------------------------------- #


import clusterfit as cf
import numpy as np
import pandas as pd

RELATIONS = ['LX-T', 'YSZ-T'] # pick from 'LX-T', 'M-T', 'YSZ-T'
COLUMNS = cf.COLUMNS


# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #


# Load data
data = pd.read_csv(INPUT_FILE)

# --------------------------- Apply intrinsic error -------------------------- #
for scaling_relation in RELATIONS: 

    # Selecting N clusters are not needed for no fitting is done in this script.
    # # How many clusters to do
    # n_clusters = cf.CONST[scaling_relation]['N']

    # Load the data
    yname, xname = cf.parse_relation_name(scaling_relation)
    Y = data[COLUMNS[yname]].values
    X = data[COLUMNS[xname]].values

    # Also load the position data
    phi_lc   = data['phi_on_lc'].values
    theta_lc = data['theta_on_lc'].values

    # the observed redshift from lightcone
    z_obs = data['ObservedRedshift'].values

    logY_ = cf.logY_(Y, z=z_obs, relation=scaling_relation)
    logX_ = cf.logX_(X, relation=scaling_relation)

    # Load the best_fit
    best_fit = pd.read_csv(FIT_ALL_FILE)
    
    # Amplify the scatter
    logA_fit = np.log10(best_fit[best_fit['Relation']==scaling_relation]['BestFitA'].values)
    B_fit = best_fit[best_fit['Relation']==scaling_relation]['BestFitB'].values
    logY_ = (logY_ - B_fit * logX_ - logA_fit) * cf.scat_boost(yname) + B_fit * logX_ + logA_fit
    Y = cf.Y(logY_, z=z_obs, relation=scaling_relation)

    # Update the data
    data[COLUMNS[yname]] = Y


# -------------------------- Apply measurement error ------------------------- #

# X-ray luminosity
L = data[COLUMNS['LX']].values
eL = cf.eL(size=L.shape)
L = L + np.random.choice(a=(1, -1), size=L.shape) * eL * L
data[COLUMNS['LX']] = L
data['e'+COLUMNS['LX']] = eL

# Ysz
Y = data[COLUMNS['YSZ']].values
eY = cf.eY(Y=Y)
Y = Y + np.random.choice(a=(1, -1), size=Y.shape) * eY * Y
data[COLUMNS['YSZ']] = Y
data['e'+COLUMNS['YSZ']] = eY

# Temperature, Chandra
T = data[cf.COLUMNS_MC['T']].values
eT = cf.eT(size=T.shape)
T = T + np.random.choice(a=(1, -1), size=T.shape) * eT * T
data[cf.COLUMNS_MC['T']] = T
data['e'+cf.COLUMNS_MC['T']] = eT

# Temperature, Chandra
T_ = data[COLUMNS['T']].values
eT_ = cf.eT(size=T_.shape)
T_ = T_ + np.random.choice(a=(1, -1), size=T_.shape) * eT_ * T_
data[COLUMNS['T']] = T_
data['e'+COLUMNS['T']] = eT_

# ----------------------------------- Save ----------------------------------- #

data.to_csv(OUTPUT_FILE, index=False)













# # --------------------------------- Test fit --------------------------------- #

#     logY_ = cf.logY_(Y, z=z_obs, relation=scaling_relation)
#     logX_ = cf.logX_(X, relation=scaling_relation)

#     scat_step = 0.006
#     B_step = 0.002
#     logA_step = 0.003

#     # Make a fit
#     best_fit = cf.run_fit(
#         logY_, logX_, **FIT_RANGE[scaling_relation],
#         scat_step  = scat_step,
#         B_step     = B_step,
#         logA_step  = logA_step,
#         scat_obs_Y = np.log10(1 + eY), 
#         scat_obs_X = np.log10(1 + eX),
#         )
    
#     scat_tot = np.mean(np.log10(1 + eY)**2 + best_fit['B']**2 * np.log10(1 + eX)**2 + best_fit['scat']**2)**0.5 
    
