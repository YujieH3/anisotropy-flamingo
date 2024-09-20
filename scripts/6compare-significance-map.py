# ---------------------------------------------
# This script calculates the significance map 
# given the best fit and bootstrapping scans.
# It modifies the best fit file on the spot 
# to include significance map.
#
# Author                       : Yujie He
# Created on (MM/YYYY)         : 03/2024
# Last Modified on (MM/YYYY)   : 09/2024
# ---------------------------------------------


import pandas as pd
import sys
sys.path.append('/home/yujiehe/anisotropy-flamingo')
import tools.clusterfit as cf
import os

# --------------------------------CONFIGURATION---------------------------------

data_dir = '/data1/yujiehe/data/fits/testrun/lightcone1/'
best_fit_scan_names = 'scan_best_fit'
bootstrap_scan_names = 'scan_bootstrap'

# -----------------------------COMMAND LINE ARGUMENTS---------------------------

import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Calculate significance map for best fit scans.")

# Add arguments
parser.add_argument('-d', '--data-dir', type=str, help='Data directory.', default=data_dir)
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing.')

# Parse the arguments
args = parser.parse_args()
data_dir = args.data_dir
overwrite = args.overwrite

# --------------------------------------MAIN------------------------------------

if __name__ == '__main__':
    for relation in cf.CONST.keys():

        flag1 = False
        flag2 = False
        for root, dirs, files in os.walk(data_dir):

            # For each bootstrap scan
            for name in files:
                flag1 = False
                if bootstrap_scan_names in name and relation in name:
                    bootstrap_file = os.path.join(root, name)
                    print(f'Found bootstrap scan: {bootstrap_file}')
                    flag1 = True
                else:
                    continue

                # Find the best fit scan
                for name in files:
                    if best_fit_scan_names in name and relation in name\
                    and bootstrap_scan_names not in name and bootstrap_file[-7:] in name:
                        best_fit_file = os.path.join(root, name)
                        print(f'Found best fit scan: {best_fit_file}')
                        best_fit = pd.read_csv(best_fit_file)
                        flag2 = True
                        break
            
                # Calculate significance map if not exist in best fit file
                if 'n_sigma' in best_fit.columns and overwrite == False:
                    print(f'Significance map already exist for {relation}.')
                    continue
                else:
                    print(f'Calculating significance map: {relation}.')
                    df = cf.significance_map(best_fit_file, bootstrap_file)
                    best_fit['n_sigma'] = df['n_sigma']
                    best_fit['sigma'] = df['sigma']
                    best_fit.to_csv(best_fit_file, index=False)
                    print(f'Significance map saved: {best_fit_file}')

            break # only search in the first level of the directory

        if flag1 == False and flag2 == False:
            raise FileNotFoundError(f'No best fit or bootstrap scan found for {relation}.')
