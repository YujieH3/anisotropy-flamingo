# ---------------------------------------------
# This script links the lightcone catalog with 
# the merger tree ids and removes duplicates 
# based on TopLeafID.
# 
# Author                       : Yujie He
# Created on (MM/DD/YYYY)      : 01/15/2024
# Last Modified on (MM/DD/YYYY): 09/19/2024
# ---------------------------------------------


import h5py
import pandas as pd
import numpy as np
import sys
import os
sys.path.append('../tools')
import clusterfit as cf

# --------------------------------configurations--------------------------------

CATALOG_FILE = None # the lightcone linked catalog
VR_TREE_FILE = None     # the merger tree file

OUTPUT = None
OVERWRITE = True       #if toggled true, will overwrite the 'catalog with tree' file. Otherwise will use existing file

# -------------------------------command line arguments-------------------------
import argparse

parser = argparse.ArgumentParser(description="Link the lightcone catalog with the merger tree ids and remove duplicates based on TopLeafID.")

# Add arguments
parser.add_argument('-i', '--input', type=str, help="Input file path. The lightcone catalog.", default=CATALOG_FILE)
parser.add_argument('-t', '--tree', type=str, help='Input file path. The merger tree file.', default=VR_TREE_FILE)
parser.add_argument('-o', '--output', type=str, help='Output file.', default=OUTPUT)
parser.add_argument('--overwrite', action='store_true', help='Overwrite the catalog with tree file.', default=False)

# Parse the arguments
args = parser.parse_args()
CATALOG_FILE = args.input
VR_TREE_FILE = args.tree
OUTPUT = args.output
OVERWRITE = args.overwrite

# ---------------------------preset output filenames----------------------------

if OVERWRITE==False and os.path.isfile(OUTPUT):
    raise ValueError('Output file already exists. Please set OVERWRITE=True or change the output filename.')
# -------------------------------------main-------------------------------------

print(f'Removing duplicates in {CATALOG_FILE}...')

catalog = pd.read_csv(CATALOG_FILE)
vr_tree = h5py.File(VR_TREE_FILE, 'r')

# Matching the full lightcone soap catalogue with merger tree ids!
# Might take a few minutes
catalog['GalaxyID'] = -1
catalog['TopLeafID'] = -1
for i in range(len(catalog)):
    snap_num = catalog.loc[i, 'snap_num']
    soapid = catalog.loc[i, 'SOAPID']
    treeid = vr_tree['SOAP/Snapshot00'+str(snap_num)][soapid]
    
    catalog.loc[i, 'GalaxyID'] = treeid + 1
    catalog.loc[i, 'TopLeafID'] = vr_tree['MergerTree/TopLeafID'][treeid]

# Sort by redshift
catalog = catalog.sort_values(by=['redshift'], ascending=True)
catalog.reset_index(drop=True, inplace=True)

# # essentially all clusters with distance larger than L/2 are duplicates
# dup_mask = (np.abs(catalog['x_lc']) > L/2) | (np.abs(catalog['y_lc']) > L/2) | (np.abs(catalog['z_lc']) > L/2)
# catalog_dup = catalog[dup_mask]
# catalog_near = catalog[~dup_mask]

# # We match the duplicates in the sample and the near universe counterparts in the 
# dup_df = pd.merge(catalog_dup[['SOAPID', 'snap_num', 'GalaxyID', 'TopLeafID', 'x_lc', 'y_lc', 'z_lc']], 
#                 catalog_near[['SOAPID', 'snap_num', 'GalaxyID', 'TopLeafID', 'x_lc', 'y_lc', 'z_lc']], 
#                 on=['TopLeafID'], 
#                 suffixes=('_dup', '_near'),
#                 )
# print(len(catalog_dup), len(dup_df))
# print('Matched:', len(dup_df))
# print('Multiple near universe match:', np.sum(dup_df.duplicated(subset=['SOAPID_dup', 'snap_num_dup'])))
# print('Multiple duplicate match:', np.sum(dup_df.duplicated(subset=['SOAPID_near', 'snap_num_near'])))
# dup_df.reset_index(drop=True, inplace=True)

# # Unmatched duplicates
# mask = catalog_dup['GalaxyID'].isin(dup_df['GalaxyID_dup'])
# unmatched_dup = catalog_dup[~mask]
# print('Unmatched duplicates:', len(unmatched_dup))
# unmatched_dup.reset_index(drop=True, inplace=True)

# if OUTPUT_MATCH_LIST:
#     dup_df.to_hdf(OUTPUT_MATCHED_DUPLICATES, key='s', mode='w')
#     unmatched_dup.to_hdf(OUTPUT_UNMATCHED_DUPLICATES, key='s', mode='w')

# Remove duplicates
catalog = catalog.drop_duplicates(subset=['TopLeafID'], keep='first') # keep the lowest counterparts
catalog.to_csv(OUTPUT, index=False)

# # Sanity check
# # matching and remove by topleafid should give the same number of clusters
# assert len(duplicate_excision) == len(catalog_near) + len(unmatched_dup)
print('Duplicates removed:', len(catalog))