# This script matches the duplicate clusters in our sample to it's near universe
# counterpart.
import h5py
import pandas as pd
import numpy as np
import os
samples = pd.read_csv('../data/samples_in_lightcone0.csv')
catalog = pd.read_hdf('../data/halo_properties_in_lightcone0.hdf5')
vr_tree = h5py.File('/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/merger_trees/vr_trees.hdf5', 'r')


# Matching soap catalogue with merger tree ids!
samples_with_tree = '../data/samples_in_lightcone0_with_tree.csv'
if os.path.isfile(samples_with_tree):
    samples = pd.read_csv(samples_with_tree)
else:
    samples['GalaxyID'] = -1
    samples['TopLeafID'] = -1
    for i in range(len(samples)):
        snap_num = samples.loc[i, 'snap_num']
        soapid = samples.loc[i, 'SOAPID']
        galaxyid = vr_tree['SOAP/Snapshot00'+str(snap_num)][soapid]
        
        samples.loc[i, 'GalaxyID'] = galaxyid
        samples.loc[i, 'TopLeafID'] = vr_tree['MergerTree/TopLeafID'][galaxyid]
    samples.to_csv('../data/samples_in_lightcone0_with_tree.csv', index=False)

dup_mask = (np.abs(samples['x_lc']) > 500) | (np.abs(samples['y_lc']) > 500) | (np.abs(samples['z_lc']) > 500)
samples_dup = samples[dup_mask]
# samples_near = samples[~dup_mask]

# Matching soap catalogue with merger tree ids!
catalog_with_tree = '../data/halo_properties_in_lightcone0_with_tree.csv'
if os.path.isfile(catalog_with_tree):
    catalog = pd.read_csv(catalog_with_tree)
else:
    catalog['GalaxyID'] = -1
    catalog['TopLeafID'] = -1
    for i in range(len(catalog)):
        snap_num = catalog.loc[i, 'snap_num']
        soapid = catalog.loc[i, 'SOAPID']
        galaxyid = vr_tree['SOAP/Snapshot00'+str(snap_num)][soapid]
        
        catalog.loc[i, 'GalaxyID'] = galaxyid
        catalog.loc[i, 'TopLeafID'] = vr_tree['MergerTree/TopLeafID'][galaxyid]
    catalog.to_csv('../data/halo_properties_in_lightcone0_with_tree.csv', index=False)

dup_mask = (np.abs(catalog['x_lc']) > 500) | (np.abs(catalog['y_lc']) > 500) | (np.abs(catalog['z_lc']) > 500)
# catalog_dup = catalog[dup_mask]
catalog_near = catalog[~dup_mask]

del dup_mask # avoid accidental use of this temporary mask
# dup_df = pd.merge(samples_dup, catalog, on=['TopLeafID', 'snap_num'], how='left')

# We match the duplicates in the sample and the near universe counterparts in the 
dup_df = pd.merge(samples_dup[['SOAPID', 'snap_num', 'GalaxyID', 'TopLeafID', 'x_lc', 'y_lc', 'z_lc']], 
                  catalog_near[['SOAPID', 'snap_num', 'GalaxyID', 'TopLeafID', 'x_lc', 'y_lc', 'z_lc']], 
                  on=['TopLeafID'], 
                  suffixes=('_dup', '_near'),
                  )
print(len(samples_dup), len(samples_dup))
print('Matched:', len(dup_df))
print('Multiple near universe match:', np.sum(dup_df.duplicated(subset=['SOAPID_dup', 'snap_num_dup'])))
print('Multiple duplicate match:', np.sum(dup_df.duplicated(subset=['SOAPID_near', 'snap_num_near'])))
dup_df.reset_index(drop=True, inplace=True)
dup_df.to_csv('../data/matched_duplicates.csv', index=True)

# Unmatched duplicates
mask = samples_dup['GalaxyID'].isin(dup_df['GalaxyID_dup'])
unmatched_dup = samples_dup[~mask]
print('Unmatched duplicates:', len(unmatched_dup))
unmatched_dup.reset_index(drop=True, inplace=True)
unmatched_dup.to_csv('../data/unmatched_duplicates.csv', index=True)
