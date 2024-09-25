# ---------------------------------------------
# This script rotates the lightcone catalog in 
# 3D space to avoid aligned structures in between
# different lightcones. It also saves the rotation
# angles for reproducibility.
#
# Author                    : Yujie He
# Created on (MM/YYYY)      : 01/2024
# Last Modified on (MM/YYYY): 09/2024
# ---------------------------------------------


import h5py
import numpy as np
from glob import glob
import healpy
INPUT = '/data/yujiehe/data/mock_lightcones/halo_lightcones'

# ------------------------------- command line arguments -----------------------
import argparse
parser = argparse.ArgumentParser(description='Join the interpolated properties with SOAP catalogue.')
parser.add_argument('-i', '--input', type=str, help='Input INPUT directory', default=INPUT)

# parse the arguments
args = parser.parse_args()
INPUT = args.input
# ------------------------------------------------------------------------------

def Rx(theta : float) -> np.matrix:
    if theta > 2*np.pi or theta < 0:
        raise ValueError(f'theta should be between 0 and 2pi. Received {theta} instead.')
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    return np.matrix([
        [1,          0,         0],
        [0,  cos_theta, sin_theta],
        [0, -sin_theta, cos_theta],
    ])

def Ry(theta : float) -> np.matrix:
    if theta > 2*np.pi or theta < 0:
        raise ValueError(f'theta should be between 0 and 2pi. Received {theta} instead.')
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    return np.matrix([
        [cos_theta, 0, -sin_theta],
        [0,         1,          0],
        [sin_theta, 0,  cos_theta],
    ])
    
def Rz(theta : float) -> np.matrix:
    if theta > 2*np.pi or theta < 0:
        raise ValueError(f'theta should be between 0 and 2pi. Received {theta} instead.')
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    return np.matrix([
        [ cos_theta, sin_theta, 0],
        [-sin_theta, cos_theta, 0],
        [0,          0,         1],
    ])


index = INPUT.find('.hdf5')
seed = int(INPUT[index-4:index])    #use lightcone number as random seed for reproducibility
with h5py.File(INPUT, 'a') as f:
    if 'x_lc_norot' in list(f.keys()):
        print('Already rotated. Skipping.')
    else:
        X = f['x_lc'][:]
        Y = f['y_lc'][:]
        Z = f['z_lc'][:]

        coords = np.stack([X, Y, Z], axis=1)
        # print(coords.shape)

        np.random.seed(seed)
        theta_x = np.random.rand() * 2*np.pi
        theta_y = np.random.rand() * 2*np.pi
        theta_z = np.random.rand() * 2*np.pi
        print('Rotation angle:', theta_x, theta_y, theta_z)
        new_coords = coords * Rx(theta_x) * Ry(theta_y) * Rz(theta_z)
        print(new_coords.shape)
        new_coords = np.array(new_coords)       #remove matrix constraints

        # save new xyz coords
        f['x_lc'][:] = new_coords[:,0]
        f['y_lc'][:] = new_coords[:,1]
        f['z_lc'][:] = new_coords[:,2]
        # also save the old ones
        f.create_dataset(name='x_lc_norot', data=X)
        f.create_dataset(name='y_lc_norot', data=Y)
        f.create_dataset(name='z_lc_norot', data=Z)

        # convert to lon lat
        new_lon, new_lat = healpy.rotator.vec2dir(new_coords[:,0], new_coords[:,1], new_coords[:,2], lonlat=True)
        
        # new coords saved at phi_on_lc and theta_on_lc for easy access in other scripts
        old_lon = f['phi_on_lc'][:]
        old_lat = f['theta_on_lc'][:]
        f['phi_on_lc'][:] = new_lon
        f['theta_on_lc'][:] = new_lat
        f.create_dataset(name='phi_on_lc_norot', data=old_lon)
        f.create_dataset(name='theta_on_lc_norot', data=old_lat)

        # write the rotation angle in attributes
        for key in ['phi_on_lc', 'theta_on_lc', 'x_lc', 'y_lc', 'z_lc']:
            f[key].attrs['theta_x'] = theta_x
            f[key].attrs['theta_y'] = theta_y
            f[key].attrs['theta_z'] = theta_z 

        
        print('Rotation done.')