# ---------------------------------------------
# This script is to correct a small issue 
# (not actually a bug). When saving the SOAP
# catalogue all three bands were saved for Lx,
# but future codes support working with only one
# column of data.
#
# This script split LX0... to LX0..., LX1..., LX2...,
# keeping data of other bands while allowing reuse
# of old codes.
#
# Author                       : Yujie He
# Created on (MM/DD/YYYY)      : 01/15/2024
# Last Modified on (MM/DD/YYYY): 09/19/2024
# ---------------------------------------------

import h5py
import pandas as pd
import numpy as np
from glob import glob
INPUT_DIR = '/data/yujiehe/data/mock_lightcones/halo_lightcones'

# ------------------------------- command line arguments -----------------------
import argparse
parser = argparse.ArgumentParser(description='Join the interpolated properties with SOAP catalogue.')
parser.add_argument('-i', '--input_dir', type=str, help='Input file directory', default=INPUT_DIR)

# parse the arguments
args = parser.parse_args()
INPUT_DIR = args.input_dir
# ------------------------------------------------------------------------------

flist = glob(f'{INPUT_DIR}/*.hdf5')
flist.sort()        #loop in order
for file in flist:
    print(file)
    with h5py.File(file, 'a') as f:
        
        skip = False
        for key in f.keys():
            if 'LX1' in key:
                skip = True
                break
        if skip == True:
            print('Already patched, file skipped.')
            continue

        for key, dst in f.items():
            if 'LX0' in key:
                LX = dst[:]     # extract the data
            
                del f[key]      # remove the original data

                f.create_dataset(name=key, data=LX[:,0])
                f.create_dataset(name=key.replace('LX0', 'LX1'), data=LX[:,1])
                f.create_dataset(name=key.replace('LX0', 'LX2'), data=LX[:,2])
            else:
                continue        # do nothing
            
        print('Patched.')