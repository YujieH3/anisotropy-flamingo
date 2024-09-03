
"""
Join the output of _c_interpolate_lightcone.py into a halo_lightcone file linked with soap catalogue  
"""
import h5py
import pandas as pd
import numpy as np
INPUT = '/data1/yujiehe/data/mock_lightcone/halo_lightcone_catalogue/halo_properties_in_lightcones.hdf5'


dict = {}
with h5py.File(INPUT, 'a') as f:
    for obs_name, obs_group in f.items():
        print(obs_name, obs_group)
        for snap_name, snap_group in obs_group.items():
            print(snap_name, snap_group)
            for qty_key, qty_dataset in snap_group.items():
                if qty_key in dict.keys():
                    dict[qty_key] = np.concatenate((dict[qty_key], qty_dataset[:]))
                else:
                    dict[qty_key] = qty_dataset[:]
            
            del f[f'{obs_name}/{snap_name}'] # del snap_group won't work

        for save_key, save_data in dict.items():
            obs_group.create_dataset(name=save_key, data=save_data)
        
print('Lightcone combined.') 

