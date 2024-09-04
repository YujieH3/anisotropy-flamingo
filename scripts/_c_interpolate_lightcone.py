"""
This script use the output of -2find-crossing-from-tree.py and interpolate to
find exactly when the halo crosses the lightcone. Such that \int cdt/a = r.
This script also get various properties of the halo at the crossing point
from the SOAP catalogue, with or without interpolation. The output takes a
similar format as Roi's linking for code reuse.
"""


import h5py
import pandas as pd
import numpy as np
import healpy as hp
import os
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=68.1, Om0=0.306)
import scipy.optimize as opt

INPUT = '/data1/yujiehe/data/mock_lightcone/halo_lightcone_catalogue/halo_crossing.hdf5'
OUTPUT = '/data1/yujiehe/data/mock_lightcone/halo_lightcone_catalogue/halo_properties_in_lightcones.hdf5'

# ------------------------------- command line arguments -----------------------
import argparse
parser = argparse.ArgumentParser(description='Interpolate the crossing data.')
parser.add_argument('-i', '--input', type=str, help='Input file path', default=INPUT)
parser.add_argument('-o', '--output', type=str, help='Output file path', default=OUTPUT)

# parse the arguments
args = parser.parse_args()
INPUT = args.input
OUTPUT = args.output
# ------------------------------------------------------------------------------



#-------------------------------------------------------------------------------
def redshift_crossing(r):
    """
    Given a comoving distance, find the redshift at which the comoving 
    crosses the lightcone.
    """
    z = opt.newton(lambda z: cosmo.comoving_distance(z).value - r, np.full(np.shape(r), fill_value=0.1))
    return z
#-------------------------------------------------------------------------------


# Read the crossing data
fin = h5py.File(INPUT, 'r')
fout = h5py.File(OUTPUT, 'a')

for observer in list(fin.keys()):
    print(f'Processing {observer}')
    
    # The observer group, in which there are lightlike and spacelike groups
    f_lightcone = fin[observer]

    # The output observer group
    fout.require_group(observer)
    fout_lightcone = fout[observer]

    # Copy the observer coordinates in attrs
    fout_lightcone.attrs['Xobs'] = f_lightcone.attrs['Xobs']
    fout_lightcone.attrs['Yobs'] = f_lightcone.attrs['Yobs']
    fout_lightcone.attrs['Zobs'] = f_lightcone.attrs['Zobs']

    # The lightlike and spacelike groups of input
    f_lightlike = f_lightcone['lightlike'] # Before crossing
    f_spacelike = f_lightcone['spacelike'] # After crossing
    
    # Loop over the snapshots
    redshift = 0.0
    for snap_name0 in reversed(list(f_spacelike.keys())):
        # The dictionary to save the data for each snapshot
        to_save = {}
        to_save_tmp = {}

        snap_num0 = int(snap_name0[-2:])
        snap_num1 = snap_num0 - 1
        snap_name1 = f'Snapshot{snap_num1:04d}'
        print(snap_name0)
        
        # For each snapshot, we get all quantities from the crossing data
        # and the SOAP catalogue, and interpolate the crossing time mainly.

        # Get from the crossing data
        Xcminpot0 = f_spacelike[snap_name0]['Xcminpot'][:]
        Ycminpot0 = f_spacelike[snap_name0]['Ycminpot'][:]
        Zcminpot0 = f_spacelike[snap_name0]['Zcminpot'][:]


        # get out of the loop if low number of clusters identified
        if len(Xcminpot0) == 0:
            continue


        # Get the crossing redshift with the later snapshot 
        r = (Xcminpot0**2 + Ycminpot0**2 + Zcminpot0**2)**0.5
        redshift_lc = redshift_crossing(r)

        # Mask: we use the snapshot which redshift is closer to the crossing redshift
        use_current_snapshot = redshift_lc < redshift + 0.025



        # Load the current snapshot
        galaxy_ids0 = f_spacelike[snap_name0]['GalaxyID'][:]
        top_leaf_ids0 = f_spacelike[snap_name0]['TopLeafID'][:]
        soap_ids0 = f_spacelike[snap_name0]['SOAPID'][:]
        M_fof0 = f_spacelike[snap_name0]['Mass_tot'][:]
        snap_num0_arr = np.full(len(galaxy_ids0), snap_num0)




        # Load the last snapshot (higher redshift)
        galaxy_ids1 = f_lightlike[snap_name1]['GalaxyID'][:]
        top_leaf_ids1 = f_lightlike[snap_name1]['TopLeafID'][:]
        soap_ids1 = f_lightlike[snap_name1]['SOAPID'][:]
        M_fof1 = f_lightlike[snap_name1]['Mass_tot'][:]
        snap_num1_arr = snap_num0_arr - 1

        Xcminpot1 = f_lightlike[snap_name1]['Xcminpot'][:]
        Ycminpot1 = f_lightlike[snap_name1]['Ycminpot'][:]
        Zcminpot1 = f_lightlike[snap_name1]['Zcminpot'][:]





        # The coordinates
        Xcminpot = np.where(use_current_snapshot, Xcminpot0, Xcminpot1)
        Ycminpot = np.where(use_current_snapshot, Ycminpot0, Ycminpot1)
        Zcminpot = np.where(use_current_snapshot, Zcminpot0, Zcminpot1)

        lc_coords = np.array([Xcminpot, Ycminpot, Zcminpot])
        phi_lc, theta_lc = hp.rotator.vec2dir(lc_coords, lonlat=True)

        # SOAP ID and snapshot number
        soap_ids = np.where(use_current_snapshot, soap_ids0, soap_ids1)
        snap_num_arr = np.where(use_current_snapshot, snap_num0_arr, snap_num1_arr)




        # Interpolate lightcone quantities: coordiantes, redshift, mass_fof
        to_save['redshift']    = redshift_lc
        to_save['GalaxyID']    = np.where(use_current_snapshot, galaxy_ids0, galaxy_ids1)
        to_save['TopLeafID']   = np.where(use_current_snapshot, top_leaf_ids0, top_leaf_ids1)
        to_save['SOAPID']      = soap_ids
        to_save['snap_num']    = snap_num_arr
        to_save['M_fof_lc']    = np.where(use_current_snapshot, M_fof0, M_fof1)
        to_save['x_lc']        = Xcminpot
        to_save['y_lc']        = Ycminpot
        to_save['z_lc']        = Zcminpot
        to_save['phi_on_lc']   = phi_lc
        to_save['theta_on_lc'] = theta_lc
 

        # Interpolate SOAP properties. For now we just use the properties from the nearest snapshot.
        for key in f_spacelike[snap_name0].keys():
            if key not in ['GalaxyID', 'SOAPID', 'TopLeafID', 'Mass_tot', 'Xcminpot', 'Ycminpot', 'Zcminpot', 'redshift']:
                to_save[key] = np.where(use_current_snapshot, f_spacelike[snap_name0][key][:], f_lightlike[snap_name1][key][:])


        # Save the data under temp group name like 'Snapshot0077Snapshot0076'
        fout_lightcone.create_group(snap_name0+snap_name1)
        for key, value in to_save.items():
            fout_lightcone[snap_name0+snap_name1].create_dataset(name=key, data=value)

        redshift += 0.05


fin.close()
fout.close()



            