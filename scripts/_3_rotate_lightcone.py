import h5py
import pandas as pd
import numpy as np
from glob import glob
import healpy
INPUT_DIR = '/data/yujiehe/data/mock_lightcones/halo_lightcones'

# ------------------------------- command line arguments -----------------------
import argparse
parser = argparse.ArgumentParser(description='Join the interpolated properties with SOAP catalogue.')
parser.add_argument('-i', '--input_dir', type=str, help='Input file directory', default=INPUT_DIR)

# parse the arguments
args = parser.parse_args()
INPUT_DIR = args.input_dir
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

flist = glob(f'{INPUT_DIR}/*.hdf5')
flist.sort()        #loop in order
for file in flist:
    print(file)
    with h5py.File(file, 'a') as f:
        X = f['x_lc'][:]
        Y = f['y_lc'][:]
        Z = f['z_lc'][:]

        coords = np.stack([X, Y, Z], axis=1)
        print(coords.shape)

        theta_x = np.random.rand() * 2*np.pi
        theta_y = np.random.rand() * 2*np.pi
        theta_z = np.random.rand() * 2*np.pi
        # print(theta_x, theta_y, theta_z)
        new_coords = coords * Rx(theta_x) * Ry(theta_y) * Rz(theta_z)
        print(new_coords.shape)

        new_coords = np.array(new_coords)       #remove matrix constraints
        new_lon, new_lat = healpy.rotator.vec2dir(new_coords[:,0], new_coords[:,1], new_coords[:,2], lonlat=True)
        
        f.create_dataset(name='phi_on_lc', data=new_lon)
        f.create_dataset(name='')
        # print(new_lon, new_lat) 
    break