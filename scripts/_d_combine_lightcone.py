
"""
Join the output of _c_interpolate_lightcone.py into a halo_lightcone file linked with soap catalogue  
"""
import h5py
import pandas as pd
INPUT = '/data1/yujiehe/data/halo_properties_in_lightcones.hdf5'



with h5py.File(INPUT, 'a') as fout:
    for observer in list(fout.keys()):

        # combine all snapshots to one catalogue
        for i, snapshot_name in enumerate(fout[observer].keys()):
            if i == 0:
                df = pd.read_hdf(INPUT, f'{observer}/{snapshot_name}')
            else:
                df = pd.concat([df, pd.read_hdf(INPUT, f'{observer}/{snapshot_name}')])
        df.to_hdf(INPUT, key=f'{observer}', mode='a')

        # remove groups for individual snapshots; deletion on disc will be registered at fout.close()
        for snapshot_name in list(fout[observer].keys()):
            del fout[f'{observer}/{snapshot_name}']

fout.close()