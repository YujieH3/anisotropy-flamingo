
import h5py
import pandas as pd
import numpy as np
samples = pd.read_csv('../data/samples_in_lightcone0.csv')
catalog = pd.read_hdf('../data/halo_properties_in_lightcone0.hdf5')
vr_tree = h5py.File('/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/merger_trees/vr_trees.hdf5', 'r')


# Matching soap catalogue with merger tree ids!
samples['GalaxyID'] = -1
samples['TopLeafID'] = -1
for i in range(len(samples)):
    snap_num = samples.loc[i, 'snap_num']
    soapid = samples.loc[i, 'SOAPID']
    galaxyid = vr_tree['SOAP/Snapshot00'+str(snap_num)][soapid]
    
    samples.loc[i, 'GalaxyID'] = galaxyid
    samples.loc[i, 'TopLeafID'] = vr_tree['MergerTree/TopLeafID'][galaxyid]

# I doubt it's needed to account for progenitors and descendants,
# because the dynamics is different after mergers.

dup_mask = (np.abs(samples['x_lc']) > 500) | (np.abs(samples['y_lc']) > 500) | (np.abs(samples['z_lc']) > 500)
samples_dup = samples[dup_mask]
samples_near = samples[~dup_mask]

# Earliest snapshot
top_snapshot = samples_dup['snap_num'].min()

print('Number of duplicates:', len(samples_dup))

samples_dup.reset_index(drop=True, inplace=True)  # dataframe preserves old index, reset to 01234...
for i in range(len(samples_dup)):
    test_top_leaf_id = samples_dup.loc[i, 'TopLeafID']
    print(test_top_leaf_id)
    for snap_num in range(77, top_snapshot, -1):
        count = np.sum(vr_tree['MergerTree/TopLeafID'][:][vr_tree['SOAP/Snapshot00'+str(snap_num)][:]] == test_top_leaf_id)  # number of clusters that have the same top leaf id in each snapshots
        print('snap_num:', snap_num, 'count:', count)