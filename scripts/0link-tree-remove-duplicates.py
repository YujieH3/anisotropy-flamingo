"""
This script link the the samples with the merger tree ids and then remove
the duplicates.
"""

import h5py
import pandas as pd
import numpy as np
import sys
import os

# --------------------------------configurations--------------------------------

CATALOG_FILE = '../data/halo_properties_in_lightcone0.hdf5' # the lightcone linked catalog
VR_TREE_FILE = '/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/merger_trees/vr_trees.hdf5' # the merger tree file

OUTPUTDIR = '../data/'
# OUTPUT_MATCH_LIST = True
OVERWRITE = True # if toggled true, will overwrite the 'catalog with tree' file. Otherwise will use existing file

# -------------------------------command line arguments-------------------------
import argparse

parser = argparse.ArgumentParser(description="Link the lightcone catalog with the merger tree ids and remove duplicates based on TopLeafID.")

# Add arguments
parser.add_argument('-i', '--input', type=str, help="Input file path. The lightcone linked catalog output by Roi's script.")
parser.add_argument('-t', '--tree', type=str, help='Input file path. The merger tree file.', default=VR_TREE_FILE)
parser.add_argument('-o', '--output', type=str, help='Output directory.', default=OUTPUTDIR)
parser.add_argument('--overwrite', action='store_true', help='Overwrite the catalog with tree file.', default=False)

# Parse the arguments
args = parser.parse_args()
CATALOG_FILE = args.input
VR_TREE_FILE = args.tree
OUTPUTDIR = args.output
OVERWRITE = args.overwrite

# ---------------------------preset output filenames----------------------------

catalog_basename = os.path.basename(CATALOG_FILE)
OUTPUT_CATALOG_WITH_TREE = os.path.join(OUTPUTDIR, catalog_basename.replace('.hdf5', '_with_trees.hdf5')) # the output catalog with tree ids
OUTPUT_CATALOG_WITH_TREE_DUP_EXCISION = OUTPUT_CATALOG_WITH_TREE.replace('.hdf5', '_duplicate_excision.hdf5') # the output catalog with tree ids and duplicates excised

if OVERWRITE==False and os.path.isfile(OUTPUT_CATALOG_WITH_TREE_DUP_EXCISION):
    sys.exit()
# -------------------------------------main-------------------------------------

catalog = pd.read_hdf(CATALOG_FILE)
vr_tree = h5py.File(VR_TREE_FILE, 'r')

# Matching the full lightcone soap catalogue with merger tree ids!
# Might take a few minutes
if os.path.isfile(OUTPUT_CATALOG_WITH_TREE):
    catalog = pd.read_hdf(OUTPUT_CATALOG_WITH_TREE)
else:
    catalog['GalaxyID'] = -1
    catalog['TopLeafID'] = -1
    for i in range(len(catalog)):
        snap_num = catalog.loc[i, 'snap_num']
        soapid = catalog.loc[i, 'SOAPID']
        treeid = vr_tree['SOAP/Snapshot00'+str(snap_num)][soapid]
        
        catalog.loc[i, 'GalaxyID'] = treeid + 1
        catalog.loc[i, 'TopLeafID'] = vr_tree['MergerTree/TopLeafID'][galaxyid]
    catalog.to_hdf(OUTPUT_CATALOG_WITH_TREE, key='lightcone', mode='w')

# Sort by redshift
catalog = catalog.sort_values(by=['redshift'], ascending=True)
catalog.reset_index(drop=True, inplace=True)

dup_mask = (np.abs(catalog['x_lc']) > 500) | (np.abs(catalog['y_lc']) > 500) | (np.abs(catalog['z_lc']) > 500)
catalog_dup = catalog[dup_mask]
catalog_near = catalog[~dup_mask]


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
duplicate_excision = catalog.drop_duplicates(subset=['TopLeafID'], keep='first') # keep the lowest counterparts
duplicate_excision.to_hdf(OUTPUT_CATALOG_WITH_TREE_DUP_EXCISION, key='lightcone', mode='w')

# Sanity check
# matching and remove by topleafid should give the same number of clusters
# assert len(duplicate_excision) == len(catalog_near) + len(unmatched_dup)
print('Done!')