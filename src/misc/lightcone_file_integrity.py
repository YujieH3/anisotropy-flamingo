# ---------------------------------------------
# This script check the integrity of the cluster
# files, check number of entries and flag
# problematic files. File removal needs to be done
# manually to avoid accidentally wipe everything.
#
# Main output are produced to log, to run the script:
# >>> python lightcone_file_integrity.py > ../log/lightcone_file_integrity.log
# 
# Two things are checked:
#     1. if number of entries are correct.
#     2. if number of clusters in all entries match.
# 
# Author                       : Yujie He
# Created on (MM/YYYY)         : 11/2024
# Last Modified on (MM/YYYY)   : 11/2024
# ---------------------------------------------

import os
import h5py
import warnings
import numpy as np

original_lcfolder = '/cosma/home/do012/dc-he4/anisotropy-flamingo/data/mock_lightcones'
copied_lcfolder = '/cosma/home/do012/dc-he4/anisotropy-flamingo/data/mock_lightcones_copy'


old_bad_lc_list = []
copy_bad_lc_list = []

for lc in range(1728):

    ogflag = 0     # false for no error
    original_lightcone = os.path.join(original_lcfolder, f'halo_properties_in_lightcone{lc:04d}.hdf5')
    with h5py.File(original_lightcone, 'r') as f:
        for snapname in ['Snapshot0078', 'Snapshot0076', 'Snapshot0074', 'Snapshot0072']:
            # check number of entries
            if len(f[snapname]) != 19:
                ogflag += 1
                warnings.warn(f'{original_lightcone} missing data in {snapname}, {len(f[snapname])}/19 entries')
            # check number of clusters
            for i, (key, value) in enumerate(f[snapname].items()):
                if i == 0:
                    len0 = len(value)
                else:
                    if len(value) != len0:
                        ogflag != 1
                        warnings.warn(f'{original_lightcone}')
    if ogflag > 0:
        print(f'{lc:04d} flagged in original lightcones')
        old_bad_lc_list.append(lc)

    cpflag = 0
    copied_lightcone = os.path.join(copied_lcfolder, f'halo_properties_in_lightcone{lc:04d}.hdf5') 
    with h5py.File(copied_lightcone, 'r') as f:
        # check number of entries
        if len(f) != 32:
            cpflag += 1
            warnings.warn(f'{copied_lightcone}, missing data, {len(f)}/32 entries')
        # check number of clusters
        for i, (key, value) in enumerate(f.items()):
            if i == 0:
                len0 = len(value)
            else:
                if len(value) != len0:
                    ogflag != 1
                    warnings.warn(f'{copied_lightcone}')
    if cpflag > 0:
        print(f'{lc:04d} flagged in copied lightcones')
        copy_bad_lc_list.append(lc)

print('original bad lightcone list:', old_bad_lc_list)
print('copied bad lightcone list:', copy_bad_lc_list)

union = np.union(old_bad_lc_list, copy_bad_lc_list)
print('union:', union)