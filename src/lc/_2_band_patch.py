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
# Author                    : Yujie He
# Created on (MM/YYYY)      : 01/2024
# Last Modified on (MM/YYYY): 09/2024
# ---------------------------------------------

import h5py

# ------------------------------- command line arguments -----------------------
import argparse
parser = argparse.ArgumentParser(description='Join the interpolated properties with SOAP catalogue.')
parser.add_argument('-i', '--input', type=str, help='Input file directory')

# parse the arguments
args = parser.parse_args()
INPUT = args.input
# ------------------------------------------------------------------------------

with h5py.File(INPUT, 'a') as f:
        
    skip = False
    for key in f.keys():
        if 'LX1' in key:
            skip = True
            break
        
    if skip == True:
        print('Already patched, file skipped.')
    else:
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