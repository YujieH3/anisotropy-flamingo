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

INPUT = '/data1/yujiehe/data/halo_crossing.hdf5'
OUTPUT = '/data1/yujiehe/data/halo_properties_in_lightcones.hdf5'
SOAP_DIR = '/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/SOAP/'


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
    crossing_observer = fin[observer]

    # The output observer group
    fout.require_group(observer)
    halo_cat_group = fout[observer]

    # Copy the observer coordinates.
    halo_cat_group.attrs['Xobserver'] = crossing_observer.attrs['Xobserver']
    halo_cat_group.attrs['Yobserver'] = crossing_observer.attrs['Yobserver']
    halo_cat_group.attrs['Zobserver'] = crossing_observer.attrs['Zobserver']

    # The lightlike and spacelike groups of input
    crossing_lightlike = crossing_observer['lightlike'] # Before crossing
    crossing_spacelike = crossing_observer['spacelike'] # After crossing
    
    # Loop over the snapshots
    redshift = 0.0
    for snap_name0 in reversed(list(crossing_spacelike.keys())):
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
        Xcminpot0 = crossing_spacelike[snap_name0]['Xcminpot'][:]
        Ycminpot0 = crossing_spacelike[snap_name0]['Ycminpot'][:]
        Zcminpot0 = crossing_spacelike[snap_name0]['Zcminpot'][:]

        # Get the crossing redshift with the later snapshot 
        r = (Xcminpot0**2 + Ycminpot0**2 + Zcminpot0**2)**0.5
        redshift_lc = redshift_crossing(r)

        # Mask: we use the snapshot which redshift is closer to the crossing redshift
        use_current_snapshot = redshift_lc < redshift + 0.025


        # Load the current snapshot
        galaxy_ids0 = crossing_spacelike[snap_name0]['GalaxyID'][:]
        top_leaf_ids0 = crossing_spacelike[snap_name0]['TopLeafID'][:]
        soap_ids0 = crossing_spacelike[snap_name0]['SOAPID'][:]
        M_fof0 = crossing_spacelike[snap_name0]['Mass_tot'][:]
        snap_num0_arr = np.full(len(galaxy_ids0), snap_num0)

        # Load the last snapshot (higher redshift)
        galaxy_ids1 = crossing_lightlike[snap_name1]['GalaxyID'][:]
        top_leaf_ids1 = crossing_lightlike[snap_name1]['TopLeafID'][:]
        soap_ids1 = crossing_lightlike[snap_name1]['SOAPID'][:]
        M_fof1 = crossing_lightlike[snap_name1]['Mass_tot'][:]
        snap_num1_arr = snap_num0_arr - 1

        Xcminpot1 = crossing_lightlike[snap_name1]['Xcminpot'][:]
        Ycminpot1 = crossing_lightlike[snap_name1]['Ycminpot'][:]
        Zcminpot1 = crossing_lightlike[snap_name1]['Zcminpot'][:]



        # The coordinates
        Xcminpot = np.where(use_current_snapshot, Xcminpot0, Xcminpot1)
        Ycminpot = np.where(use_current_snapshot, Ycminpot0, Ycminpot1)
        Zcminpot = np.where(use_current_snapshot, Zcminpot0, Zcminpot1)

        lc_coords = np.array([Xcminpot, Ycminpot, Zcminpot])
        theta_lc, phi_lc = hp.rotator.vec2dir(lc_coords, lonlat=True)

        # SOAP ID and snapshot number
        soap_ids = np.where(use_current_snapshot, soap_ids0, soap_ids1)
        snap_num_arr = np.where(use_current_snapshot, snap_num0_arr, snap_num1_arr)

        # Interpolate lightcone quantities: coordiantes, redshift, mass_fof
        to_save['redshift'] = redshift_lc
        to_save['GalaxyID'] = np.where(use_current_snapshot, galaxy_ids0, galaxy_ids1)
        to_save['TopLeafID'] = np.where(use_current_snapshot, top_leaf_ids0, top_leaf_ids1)
        to_save['SOAPID'] = soap_ids
        to_save['snap_num'] = snap_num_arr
        to_save['M_fof_lc'] = np.where(use_current_snapshot, M_fof0, M_fof1)
        to_save['x_lc'] = Xcminpot
        to_save['y_lc'] = Ycminpot
        to_save['z_lc'] = Zcminpot
        to_save['phi_on_lc'] = phi_lc
        to_save['theta_on_lc'] = theta_lc
 

        # Interpolate SOAP properties. For now we just use the properties from the nearest snapshot.
        for key in to_save_tmp.keys():
            to_save[key] = np.where(use_current_snapshot, to_save[key], to_save_tmp[key])

        # Save the data
        df = pd.DataFrame(to_save)
        df.to_hdf(halo_cat_group, key=snap_name0, mode='a')

        redshift += 0.05


fin.close()
fout.close()



            