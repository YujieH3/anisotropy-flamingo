# ---------------------------------------------
# This script joins the output of 
# _c_interpolate_lightcone.py into a halo_lightcone
# (like) file linked with soap catalogue
#
# Author                       : Yujie He
# Created on (MM/DD/YYYY)      : 01/15/2024
# Last Modified on (MM/DD/YYYY): 09/19/2024
# ---------------------------------------------
"""
Join the output of _c_interpolate_lightcone.py into a halo_lightcone file linked with soap catalogue  
"""
import h5py
import numpy as np

# ------------------------------- command line arguments -----------------------
import argparse
parser = argparse.ArgumentParser(description='Join the interpolated properties with SOAP catalogue.')
parser.add_argument('-i', '--input', type=str, help='Input file directory')

# parse the arguments
args = parser.parse_args()
INPUT = args.input
# ------------------------------------------------------------------------------


with h5py.File(INPUT, 'a') as f:
    if 'Snapshot' not in list(f.keys())[0]:
        print('Already combined.')
    else:
        to_save = {}
        for snap_name, snap_group in f.items():
            print(snap_name, snap_group)
            for qty_key, qty_dataset in snap_group.items():
                if qty_key in to_save.keys():
                    to_save[qty_key] = np.concatenate((to_save[qty_key], qty_dataset[:]))
                else:
                    to_save[qty_key] = qty_dataset[:]
            
            del f[f'{snap_name}']        # del snap_group won't work, for unknown reason

    for save_key, save_data in to_save.items():
        f.create_dataset(name=save_key, data=save_data)
    
print('Lightcone combined.') 



