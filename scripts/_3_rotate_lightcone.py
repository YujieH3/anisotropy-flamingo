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
    index = file.find('.hdf5')
    seed = int(file[index-4:index])    #lightcone number, used as random seed so that the same lightcone is comparable
    with h5py.File(file, 'a') as f:
        if 'x_lc_norot' in list(f.keys()):
            print('Already rotated. Skipping.')
            continue

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