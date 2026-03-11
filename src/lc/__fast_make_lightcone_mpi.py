
"""
Create 12 lightcones on each side:
time mpiexec -n 9 python __fast_make_lightcone_mpi.py -N 12
"""

import healpy as hp
import scipy.optimize as opt
import h5py
import numpy as np
from mpi4py import MPI
import warnings
#import numba
#import sys
#import time
import os

# on cosma8
SOAP_DIR = '/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/SOAP'
OUTPUT_DIR = '/cosma8/data/do012/dc-he4/mock_lightcones_test'

# on hypernova
# OUTPUT = '/data1/yujiehe/data/mock_lightcone/halo_lightcone_catalogue/halo_crossing.hdf5'
MAX_REDSHIFT = 0.35 # max redshift considered

L = 1000   # simulation box size in comoving Mpc (cMpc)

# ---------------------------- command line arguments --------------------------
import argparse
parser = argparse.ArgumentParser('find the two snapshots at crossing the lightcone.')
parser.add_argument('-i', '--input', type=str, help='Input file path', default=SOAP_DIR)
parser.add_argument('-o', '--output', type=str, help='Output file path', default=OUTPUT_DIR)
parser.add_argument('-z', '--redshift', type=float, help='Max redshift considered', default=MAX_REDSHIFT)
parser.add_argument('-N', '--lightcone_number', type=int, help='Number of lightcones on one side', default=1)
parser.add_argument('-L', '--box_size', type=int, help='Box size in comoving Mpc', default=L)

# parse the arguments
args = parser.parse_args()
SOAP_DIR = args.input
OUTPUT_DIR = args.output
MAX_REDSHIFT = args.redshift
N = args.lightcone_number
L = args.box_size
# ------------------------------------------------------------------------------

# set the step size to match data structure of specific box size
if L == 2800:
    dz = 0.10
    snap_list=[f'Snapshot{x:04d}' for x in np.arange(78, 0, -2)]
elif L == 1000:
    dz = 0.05
    snap_list=[f'Snapshot{x:04d}' for x in np.arange(77, 0, -1)]
else:
    raise ValueError(f'Box size not supported. Only 2800 and 1000 are supported, {L} given.')

# single or multi observer run
if N == 1:
    # Single observer test
    Xobs = 3*L/4
    Yobs = 3*L/4
    Zobs = 3*L/4

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=68.1, Om0=0.306)

#-------------------------------------------------------------------------------

def _center_coord(x, x0, L):
    """
    For coordinate (1 axis) x ranging from -L/2 to L/2, shift the center point x=0
    to x0. The coordinate is shifted periodically such that after the shift, 
    the x0 point lies at the center of the box.
    """
    x_x0 = (x - x0 + L/2) % L - L/2
    return x_x0


def _mesh(N : int, L : int):
    """
    Make N**3 evenly spaced coordinates.
    """
    X = np.linspace(0, L, num=N, endpoint=False)
    Y = np.linspace(0, L, num=N, endpoint=False)
    Z = np.linspace(0, L, num=N, endpoint=False)
    Xcoord, Ycoord, Zcoord = np.meshgrid(X, Y, Z, indexing='ij')
    
    # output as a 1D list
    Xcoord = np.ravel(Xcoord)
    Ycoord = np.ravel(Ycoord)
    Zcoord = np.ravel(Zcoord)

    return Xcoord, Ycoord, Zcoord


def _redshift_crossing(r : np.ndarray):
    """
    Given a comoving distance, find the redshift at which the comoving 
    crosses the lightcone.
    """
    z = opt.newton(lambda z: cosmo.comoving_distance(z).value - r, np.full(np.shape(r), fill_value=0.1))
    return z

# ------------------------------------------------------------------------------


# Initialize list of observers/lightcones
if N == 1:
    Xobsarr = np.array([Xobs])
    Yobsarr = np.array([Yobs])
    Zobsarr = np.array([Zobs])
else:
    Xobsarr, Yobsarr, Zobsarr = _mesh(N=N, L=L)


# mpi
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 9:
    raise Exception(f'The scripts is configured with 9 cores only. {size} cores detected.')

if not os.path.exists(OUTPUT_DIR):
    raise Exception(f'Output path {OUTPUT_DIR} does not exist')

# initialize 
if rank != 0:
    soap_h5names = [
        'SO/500_crit/GasMass',
        'SO/500_crit/XRayLuminosityWithoutRecentAGNHeating',
        'SO/500_crit/XRayLuminosityWithoutRecentAGNHeatingCoreExcision',
        'SO/500_crit/XRayLuminosityInRestframeWithoutRecentAGNHeating',
        'SO/500_crit/XRayLuminosityInRestframeWithoutRecentAGNHeatingCoreExcision',
        'SO/500_crit/SpectroscopicLikeTemperatureWithoutRecentAGNHeatingCoreExcision',
        'SO/5xR_500_crit/ComptonYWithoutRecentAGNHeating',
    ]
    save_h5names = [
        'GasMass',
        'LX0WithoutRecentAGNHeating',
        'LX0WithoutRecentAGNHeatingCoreExcision',
        'LX0InRestframeWithoutRecentAGNHeating',
        'LX0InRestframeWithoutRecentAGNHeatingCoreExcision',
        'SpectroscopicLikeTemperatureWithoutRecentAGNHeatingCoreExcision',
        'Y5R500WithoutRecentAGNHeating' 
    ]

obsnum = 0
for Xobs, Yobs, Zobs in zip(Xobsarr, Yobsarr, Zobsarr):
    # observer coordinates
    obs_coords = np.array([Xobs, Yobs, Zobs])
    lightcone_name = f'lightcone{obsnum:04d}'
    output_file = os.path.join(OUTPUT_DIR, 'halo_properties_in_' + lightcone_name + '.hdf5')

    # skip if the observer exists
    if os.path.isfile(output_file):
        if rank == 0:
            print(f'{output_file} already exists, skipping')
        obsnum += 1
        continue

    # create the output file
    if rank == 0:
        print(f'Creating observer: {obsnum} at ({Xobs}, {Yobs}, {Zobs})')
        # prepare output file
        with h5py.File(output_file, 'a') as f:
            # Save the observer coordinates, in Mpc
            f.attrs['Xobs'] = Xobs
            f.attrs['Yobs'] = Yobs
            f.attrs['Zobs'] = Zobs

    z_snap = 0.0
    for i, snap in enumerate(snap_list): # 2.8Gpc soap has only 0078, 0076, 0074, 0072, 0070.. in cadence of 2
        # redshift
        if z_snap > MAX_REDSHIFT:
            break
        
        if rank == 0:
            print(snap)
        # the snapshot number
        snap_num = int(snap[-4:])

        # the SOAP properties
        soap_file = os.path.join(SOAP_DIR, f'halo_properties_{snap_num:04d}.hdf5')
        
        if rank == 0:
            print('soap file found:', soap_file)

            # get soapids
            print('getting soapids...')
            with h5py.File(soap_file, 'r') as f:
                soapids = f['VR/ID'][:]
            print(f'soapids loaded: {soapids.shape}')
        elif rank == 1:
            # mass filter: most useful
            print('loading M500...')
            with h5py.File(soap_file, 'r') as f:
                M500 = f['SO/500_crit/TotalMass'][:]
            print(f'M500 loaded: {M500.shape}')
        elif rank == 2:
            print('loading coordinates...')
            with h5py.File(soap_file, 'r') as f:
                coords = f['SO/500_crit/CentreOfMass'][:]
                # coords = f['VR/CentreOfPotential'][:]
            print(f'Coordinates loaded: {coords.shape}')

        # send M500 to rank0
        if rank == 1:
            comm.send(len(M500), 0, tag=8)
            M500 = M500.astype(np.float64)
            comm.Send([M500, MPI.DOUBLE], 0, tag=11)
        elif rank == 0:
            datasize = comm.recv(source=1, tag=8)
            M500 = np.empty(datasize, dtype=np.float64)
            comm.Recv([M500, MPI.DOUBLE], source=1, tag=11)

        # mask with M500
        if rank == 0:
            print('applying M500 mask...')
            mask = M500 > 1e13
            soapids = soapids[mask]
            M500 = M500[mask]
            print(f'M500 mask applied: {soapids.shape}')
        
        # send coords to rank0
        if rank == 2:
            comm.send(len(coords), 0, tag=16)
            coords = coords.astype(np.float64)
            comm.Send([coords, MPI.DOUBLE], 0, tag=15)
        elif rank == 0:
            datasize = comm.recv(source=2, tag=16)
            coords = np.empty((datasize, 3), dtype=np.float64)
            comm.Recv([coords, MPI.DOUBLE], source=2, tag=15)

            # mask coords with existing mass mask
            coords = coords[mask, :]

            # physical to comoving
            coords *= (1 + z_snap)
            
        if rank == 0:
            z_shell_low = z_snap - dz/2 if z_snap > 0 else 0
            z_shell_high = z_snap + dz/2

            coords = _center_coord(coords, obs_coords, L)

            print('applying lightcone mask...')
            r = np.sqrt(np.sum(coords**2, axis=1))
            r_z_low = cosmo.comoving_distance(z_shell_low).value
            r_z_high = cosmo.comoving_distance(z_shell_high).value
            mask = (r > r_z_low) & (r < r_z_high)
            # mask the quantities we need
            soapids = soapids[mask]
            r       = r[mask]
            coords  = coords[mask]
            M500    = M500[mask]
            print(f'Halo crossed: {soapids.shape}')

            if len(soapids) == 0:
                print('No halo crossed lightcone, skipping snapshot...')

        # broadcast datasize for saving
        if rank == 0:
            datasize = len(soapids)
        else:
            datasize = None
        datasize = comm.bcast(datasize, root=0)

        if datasize == 0:
            comm.Barrier()
            z_snap += dz
            continue
        # print(datasize)               # so this works fine

        # broadcast final list of soapids
        if rank == 0:
            soapids = soapids.astype('i')
        else:
            soapids = np.empty(datasize, dtype='i')
        comm.Bcast(soapids, root=0)     # broadcasting numpy array return None, data is saved in buffer
        # print(soapids)                # works now


        if rank == 0:
            # get the crossing redshift
            print('getting crossing redshift...')
            z_cross = _redshift_crossing(r)
            print(f'crossing redshift obtained: {z_cross.shape}')

            # longitude, latitude
            phi_lc, theta_lc = hp.rotator.vec2dir(coords[:,0], coords[:,1], coords[:,2], lonlat=True)

            # save the properties
            print('saving properties...')
            with h5py.File(soap_file, 'r') as cat:
                with h5py.File(output_file, 'a', ) as f:
                    f_snap = f.require_group(snap)
                    f_snap.create_dataset('SOAPID', data=soapids)
                    print('SOAPID saved.')

                    # lightcone quantities
                    print('saving lightcone quantities...')
                    f_snap.create_dataset('redshift', data=z_cross)
                    f_snap.create_dataset('snap_num', data=np.full_like(soapids, snap_num))
                    f_snap.create_dataset('x_lc', data=coords[:, 0])
                    f_snap.create_dataset('y_lc', data=coords[:, 1])
                    f_snap.create_dataset('z_lc', data=coords[:, 2])
                    f_snap.create_dataset('phi_on_lc', data=phi_lc)
                    f_snap.create_dataset('theta_on_lc', data=theta_lc)
                    print('lightcone quantities saved.')

                    # save M500 since its already in rank0's memory
                    f_snap.create_dataset('M500', data=M500)
            
        # save soap quantities by loading all quantities in each thread and pass to the first to save
        elif rank != size - 1:
            print(f'rank {rank} saving soap quantities...')
            with h5py.File(soap_file, 'r') as cat:
                if datasize < 10000:
                    data = cat[soap_h5names[rank - 1]][soapids]
                else:
                    data = cat[soap_h5names[rank - 1]][:][soapids]
            print(f'properties loaded by process {rank}')
        elif rank == size - 1:
            print(f'rank {rank} saving soap quantities...')
            with h5py.File(soap_file, 'r') as cat:
                if datasize < 10000:
                    V = cat['SO/500_crit/CentreOfMassVelocity'][soapids]
                else:
                    V = cat['SO/500_crit/CentreOfMassVelocity'][:][soapids]
            print(f'properties loaded by process {rank}')

        # save the soap properties. velocity is treated separately
        if rank == 0:
            comm.send(100, 1, tag=rank + 1)
        elif rank != 0:
            for j in range(size - 2): # size = 9, so 1-7 here
                if rank == j + 1:
                    comm.recv(source=rank - 1, tag=rank)    # send signal so that only one thread save at a time
                    with h5py.File(output_file, 'a') as f:
                        f_snap = f.require_group(snap)
                        f_snap.create_dataset(save_h5names[j], data=data)
                    if rank != size - 1:                    # if it's the last one, don't send
                        _ = comm.send(100, rank + 1, tag=rank + 1)
                    print(f'properties saved by process {rank}')
        
        comm.Barrier()      # start after other saves are done

        # save velocities with last thread
        if rank == size - 1:
            with h5py.File(output_file, 'a') as f:
                f_snap = f.require_group(snap)
                f_snap.create_dataset('Vx', data=V[:,0])
                f_snap.create_dataset('Vy', data=V[:,1])
                f_snap.create_dataset('Vz', data=V[:,2])
            print(f'velocity saved by process {rank}.')

        
        
        # synchronize before move on
        comm.Barrier() 
            
                    

        # up to next snapshot
        z_snap += dz

    obsnum += 1

comm.Barrier()
if rank == 0:
    print('All done.')


