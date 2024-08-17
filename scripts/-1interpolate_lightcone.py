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
    z = opt.newton(lambda z: cosmo.comoving_distance(z) - r, np.full(np.shape(r), fill_value=0.1))
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
    fout.create_group(observer)
    halo_cat_group = fout[observer]

    # Copy the observer coordinates.
    halo_cat_group.attrs['Xobserver'] = crossing_observer.attrs['Xobserver']
    halo_cat_group.attrs['Yobserver'] = crossing_observer.attrs['Yobserver']
    halo_cat_group.attrs['Zobserver'] = crossing_observer.attrs['Zobserver']

    # The lightlike and spacelike groups of input
    crossing_lightlike = crossing_observer['lightlike']
    crossing_spacelike = crossing_observer['spacelike']
    
    # Loop over the snapshots
    redshift = 0.0
    for snapshot_name in list(crossing_lightlike.keys()):
        # The dictionary to save the data for each snapshot
        to_save = {}
        to_save_tmp = {}

        snap_num0 = int(snapshot_name[-2:])
        snap_num1 = snap_num0 - 1
        
        # For each snapshot, we get all quantities from the crossing data
        # and the SOAP catalogue, and interpolate the crossing time mainly.

        # Get from the crossing data
        redshift0 = crossing_lightlike[snapshot_name]['redshift'][:]        
        Xcmbp_bh0 = crossing_lightlike[snapshot_name]['Xcmbp_bh'][:]
        Ycmbp_bh0 = crossing_lightlike[snapshot_name]['Ycmbp_bh'][:]
        Zcmbp_bh0 = crossing_lightlike[snapshot_name]['Zcmbp_bh'][:]

        # Get the crossing redshift with the later snapshot 
        r = (Xcmbp_bh0**2 + Ycmbp_bh0**2 + Zcmbp_bh0**2)**0.5
        redshift_lc = redshift_crossing(r)

        # Mask: we use the snapshot which redshift is closer to the crossing redshift
        use_current_snapshot = redshift_lc < redshift + 0.025


        # Load the current snapshot
        galaxy_ids0 = crossing_lightlike[snapshot_name]['GalaxyID'][:]
        top_leaf_ids0 = crossing_lightlike[snapshot_name]['TopLeafID'][:]
        soap_ids0 = crossing_lightlike[snapshot_name]['SOAPID'][:]
        M_fof0 = crossing_lightlike[snapshot_name]['M_fof'][:]
        snap_num0_arr = np.full(len(galaxy_ids0), int(snapshot_name[-2:]))

        # Load the last snapshot (higher redshift)
        galaxy_ids1 = crossing_spacelike[snapshot_name]['GalaxyID'][:]
        top_leaf_ids1 = crossing_spacelike[snapshot_name]['TopLeafID'][:]
        soap_ids1 = crossing_spacelike[snapshot_name]['SOAPID'][:]
        M_fof1 = crossing_spacelike[snapshot_name]['M_fof'][:]
        snap_num1_arr = snap_num0_arr - 1

        Xcmbp_bh1 = crossing_spacelike[snapshot_name]['Xcmbp_bh'][:]
        Ycmbp_bh1 = crossing_spacelike[snapshot_name]['Ycmbp_bh'][:]
        Zcmbp_bh1 = crossing_spacelike[snapshot_name]['Zcmbp_bh'][:]



        # The coordinates
        Xcmbp_bh = np.where(use_current_snapshot, Xcmbp_bh0, Xcmbp_bh1)
        Ycmbp_bh = np.where(use_current_snapshot, Ycmbp_bh0, Ycmbp_bh1)
        Zcmbp_bh = np.where(use_current_snapshot, Zcmbp_bh0, Zcmbp_bh1)

        lc_coords = np.array([Xcmbp_bh, Ycmbp_bh, Zcmbp_bh])
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
        to_save['x_lc'] = Xcmbp_bh
        to_save['y_lc'] = Ycmbp_bh
        to_save['z_lc'] = Zcmbp_bh
        to_save['phi_on_lc'] = phi_lc
        to_save['theta_on_lc'] = theta_lc
 

        # Get from the SOAP catalogue
        catalogue = h5py.File(os.path.join(SOAP_DIR, f'halo_properties_{snap_num0:04d}.hdf5'), 'r')

        # Having Mfof from SOAP and the halo lightcone to check if the matching is successful

        to_save["MfofSOAP"] = catalogue["FOFSubhaloProperties/TotalMass"][soap_ids0]
        
        to_save['M500'] = catalogue['SO/500_crit/TotalMass'][soap_ids0]
        to_save["GasMass"] = catalogue["SO/500_crit/GasMass"][soap_ids0]
        
        to_save["LX0WithoutRecentAGNHeating"] = catalogue["SO/500_crit/XRayLuminosityWithoutRecentAGNHeating"][:,0][soap_ids0]  # eRosita 0.2-2.3 keV band and exclude gas recently heated by AGN.
        to_save["LX0InRestframeWithoutRecentAGNHeating"] = catalogue["SO/500_crit/XRayLuminosityInRestframeWithoutRecentAGNHeating"][:,0][soap_ids0]  # eRosita 0.2-2.3 keV band and exclude gas recently heated by AGN.
        to_save["LX0WithoutRecentAGNHeatingCoreExcision"] = catalogue["SO/500_crit/XRayLuminosityWithoutRecentAGNHeatingCoreExcision"][:,0][soap_ids0]    
        to_save["LX0InRestframeWithoutRecentAGNHeatingCoreExcision"] = catalogue["SO/500_crit/XRayLuminosityInRestframeWithoutRecentAGNHeatingCoreExcision"][:,0][soap_ids0]    

        to_save["GasTemperatureWithoutRecentAGNHeatingCoreExcision"] = catalogue["SO/500_crit/GasTemperatureWithoutRecentAGNHeatingCoreExcision"][soap_ids0]
        to_save["SpectroscopicLikeTemperatureWithoutRecentAGNHeatingCoreExcision"] = catalogue["SO/500_crit/SpectroscopicLikeTemperatureWithoutRecentAGNHeatingCoreExcision"][soap_ids0]
        to_save["Y5R500WithoutRecentAGNHeating"] = catalogue["SO/5xR_500_crit/ComptonYWithoutRecentAGNHeating"][soap_ids0]
        
        to_save["Vx"] = catalogue["SO/500_crit/CentreOfMassVelocity"][:,0][soap_ids0]
        to_save["Vy"] = catalogue["SO/500_crit/CentreOfMassVelocity"][:,1][soap_ids0]
        to_save["Vz"] = catalogue["SO/500_crit/CentreOfMassVelocity"][:,2][soap_ids0]


        catalogue.close() # Don't forget to close the file



        # Get from next snapshot SOAP
        catalogue = h5py.File(os.path.join(SOAP_DIR, f'halo_properties_{snap_num1:04d}.hdf5'), 'r')

        to_save_tmp["MfofSOAP"] = catalogue["FOFSubhaloProperties/TotalMass"][soap_ids1]
        
        to_save_tmp['M500'] = catalogue['SO/500_crit/TotalMass'][soap_ids1]
        to_save_tmp["GasMass"] = catalogue["SO/500_crit/GasMass"][soap_ids1]
        
        to_save_tmp["LX0WithoutRecentAGNHeating"] = catalogue["SO/500_crit/XRayLuminosityWithoutRecentAGNHeating"][:,0][soap_ids1]  # eRosita 0.2-2.3 keV band and exclude gas recently heated by AGN.
        to_save_tmp["LX0InRestframeWithoutRecentAGNHeating"] = catalogue["SO/500_crit/XRayLuminosityInRestframeWithoutRecentAGNHeating"][:,0][soap_ids1]  # eRosita 0.2-2.3 keV band and exclude gas recently heated by AGN.
        to_save_tmp["LX0WithoutRecentAGNHeatingCoreExcision"] = catalogue["SO/500_crit/XRayLuminosityWithoutRecentAGNHeatingCoreExcision"][:,0][soap_ids1]    
        to_save_tmp["LX0InRestframeWithoutRecentAGNHeatingCoreExcision"] = catalogue["SO/500_crit/XRayLuminosityInRestframeWithoutRecentAGNHeatingCoreExcision"][:,0][soap_ids1]    

        to_save_tmp["GasTemperatureWithoutRecentAGNHeatingCoreExcision"] = catalogue["SO/500_crit/GasTemperatureWithoutRecentAGNHeatingCoreExcision"][soap_ids1]
        to_save_tmp["SpectroscopicLikeTemperatureWithoutRecentAGNHeatingCoreExcision"] = catalogue["SO/500_crit/SpectroscopicLikeTemperatureWithoutRecentAGNHeatingCoreExcision"][soap_ids1]
        to_save_tmp["Y5R500WithoutRecentAGNHeating"] = catalogue["SO/5xR_500_crit/ComptonYWithoutRecentAGNHeating"][soap_ids1]
        
        to_save_tmp["Vx"] = catalogue["SO/500_crit/CentreOfMassVelocity"][:,0][soap_ids1]
        to_save_tmp["Vy"] = catalogue["SO/500_crit/CentreOfMassVelocity"][:,1][soap_ids1]
        to_save_tmp["Vz"] = catalogue["SO/500_crit/CentreOfMassVelocity"][:,2][soap_ids1]

        catalogue.close() # Don't forget to close the file
        



        # Interpolate the properties. For now we just use the properties from the nearest snapshot.
        for key in to_save_tmp.keys():
            to_save[key] = np.where(use_current_snapshot, to_save[key], to_save_tmp[key])

        # Save the data
        df = pd.DataFrame(to_save)
        df.to_hdf(halo_cat_group, key=snapshot_name, mode='a')

        redshift += 0.05


fin.close()
fout.close()


# Concatenate the different snapshots
with h5py.File(OUTPUT, 'a') as fout:
    for observer in list(fout.keys()):
        # Concatenate the different snapshots
        for i, snapshot_name in enumerate(fout[observer].keys()):
            if i == 0:
                df = pd.read_hdf(OUTPUT, f'{observer}/{snapshot_name}')
            else:
                df = pd.concat([df, pd.read_hdf(OUTPUT, f'{observer}/{snapshot_name}')])
        df.to_hdf(fout, key=f'{observer}', mode='a')

        # Remove the individual snapshots
        for snapshot_name in list(fout[observer].keys()):
            del fout[f'{observer}/{snapshot_name}']
            