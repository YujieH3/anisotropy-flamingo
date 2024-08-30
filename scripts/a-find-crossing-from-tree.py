"""
This script defines a (list of) observers and create corresponding lightcones.
The properties before and after crossing lightcone is saved for later 
interpolation. 

We separate this procedure such that the interpolation scheme can be tested 
quickly. A mass selection of 5e12 is applied to mask out subhalos and 
galaxies, and to save disk space.
"""

import h5py
import pandas as pd # Use pandas' simple I/O
import numpy as np
from tqdm import tqdm

REDSHIFT_RANGE = 0.25
MERGER_TREE_FILE = '/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/merger_trees/vr_trees.hdf5'
OUTPUT = '/data1/yujiehe/data/halo_crossing.hdf5'
OBSERVER_NUM = 0

L = 1000                        # Box size in Mpc
Xobserver = L/4                 # The first coord of the observer
Yobserver = L/4
Zobserver = L/4

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=68.1, Om0=0.306)

#-------------------------------------------------------------------------------
def is_light_like(z, r, cosmo):
    """
    Tell if an object at a comoving coordinate is observable at given redshift0.
    Output false if the spacetime interval is space-like and true if light-like.  

    light-like = (cdt/a(t))^2 > (dr)^2
    space-like = (cdt/a(t))^2 < (dr)^2
    """
    flags = (cosmo.comoving_distance(z).value > r)  # Bool or a list of bool

    return flags


def center_coord(x, x0, L):
    """
    For coordinate (1 axis) x ranging from -L/2 to L/2, shift the center point x=0
    to x0. The coordinate is shifted periodically such that after the shift, 
    the x0 point lies at the center of the box.
    """
    x_x0 = (x - x0) % L - L/2
    return x_x0

#-------------------------------------------------------------------------------

# Read the merger tree
tree = h5py.File(MERGER_TREE_FILE, 'r')

# Set snapshot names
snapshot_names = list(tree['SOAP'].keys())
snapshot_names = snapshot_names[::-1] # In reversed order, redshift0 0, snapshot 77 of 1Gpc or 78 of 2.8 Gpc at the front

# Create/Open hdf5 file, and create group for the observer
f0 = h5py.File(OUTPUT, 'a')
observer_name = f'observer{OBSERVER_NUM}'
f0.require_group(observer_name)

# Save the observer coordinates, in Mpc
f = f0[observer_name] # We only do one observer at a time
f.attrs['Xobserver'] = Xobserver
f.attrs['Yobserver'] = Yobserver
f.attrs['Zobserver'] = Zobserver

# Create groups for lightlike and spacelike
f.require_group('lightlike')
f.require_group('spacelike')
f_lightlike = f['lightlike'] # lightlike is before crossing
f_spacelike = f['spacelike'] # spacelike is after crossing
f_lightlike.attrs['Note'] = 'before crossing'
f_spacelike.attrs['Note'] = 'after crossing'

# Initialize
redshift0 = 0.0
redshift1 = 0.05
for i, snap0 in enumerate(tqdm(snapshot_names)):
    snap1 = snapshot_names[i + 1]   # snapshot with redshift0 z + 0.05
    snapnum0 = int(snap0[-2:])
    snapnum1 = int(snap1[-2:])
    print(snap0, snap1)                    # log

    # Only account up to a given range
    if redshift0 > REDSHIFT_RANGE:
        break

    
    """
    We find the halos that are lightlike in the current snapshot, and its
    progenitors are spacelike in the next snapshot. We save the coordinates
    and redshift0 of these halos and their progenitors. The redshift0 is simply
    the snapshot redshift0 and the next snapshot redshift0. The precise redshift0
    at crossing is to be interpolated later.
    """
    

    # Go to SOAP find the GalaxyID of the clusters in a catalogue
    galaxyid = tree['SOAP/' + snap0][:] + 1
    # Also the SOAP id, soap id are array indices in the SOAP catalogue, starts from 0.
    soapid0 = np.arange(len(galaxyid))
    # Retrieve the galaxy id of the last progenitor
    lastprog_galaxyid = tree['MergerTree/LastProgID'][:][galaxyid - 1] # beware that galaxy id start from 1
    topleafid0 = tree['MergerTree/TopLeafID'][:][galaxyid - 1]

    # Sanity check if the galaxy id & soap id are matching
    assert np.sum(tree['Subhalo/ID'][:][galaxyid - 1] - 1 - soapid0) == 0 # The Subhalo/ID is 1 + the SOAP id (array index in the SOAP catalogue)
    print('SOAP id and Galaxy id matched.')

    # select only those last progenitors that are in snapshot0076
    real_snapnum1 = tree['Subhalo/SnapNum'][:][lastprog_galaxyid - 1]
    mask = (real_snapnum1 == snapnum1)
    galaxyid          = galaxyid[mask]
    lastprog_galaxyid = lastprog_galaxyid[mask]
    topleafid0        = topleafid0[mask]
    soapid0           = soapid0[mask]

    # Retrieve the soapid of the last progenitor
    soapid1 = tree['Subhalo/ID'][:][lastprog_galaxyid - 1] - 1 # The Subhalo/ID is 1 + the SOAP id (array index in the SOAP catalogue)
    topleafid1 = tree['MergerTree/TopLeafID'][:][lastprog_galaxyid - 1]

    # # Sanity check
    # assert np.sum(topleafid0 - topleafid1) == 0


    # Go to merger tree and find the coordinates in
    # halo potential minimum & most bound black hole particles
    Xcminpot0 = tree['Subhalo/Xcminpot'][:][galaxyid - 1] / (redshift0 + 1)  # convert pMpc to cMpc
    Ycminpot0 = tree['Subhalo/Ycminpot'][:][galaxyid - 1] / (redshift0 + 1)
    Zcminpot0 = tree['Subhalo/Zcminpot'][:][galaxyid - 1] / (redshift0 + 1)
    # Shift the coordinate center to observer center
    Xcminpot0 = center_coord(x=Xcminpot0, x0=Xobserver, L=L)
    Ycminpot0 = center_coord(x=Ycminpot0, x0=Yobserver, L=L)
    Zcminpot0 = center_coord(x=Zcminpot0, x0=Zobserver, L=L)
    # The comoving distance to the halo min potential
    r0 = (Xcminpot0**2 + Ycminpot0**2 + Zcminpot0**2) ** 0.5
    # Boolean mask of lightlike of first snapshot
    lightlike_mask0 = is_light_like(z=redshift0, r=r0, cosmo=cosmo)

  

    # Find coordinates of the last progenitor
    Xcminpot1 = tree['Subhalo/Xcminpot'][:][lastprog_galaxyid - 1] / (redshift1 + 1) # convert pMpc to cMpc
    Ycminpot1 = tree['Subhalo/Ycminpot'][:][lastprog_galaxyid - 1] / (redshift1 + 1)
    Zcminpot1 = tree['Subhalo/Zcminpot'][:][lastprog_galaxyid - 1] / (redshift1 + 1)
    # Shift the coordinate center to observer center
    Xcminpot1 = center_coord(x=Xcminpot1, x0=Xobserver, L=L)
    Ycminpot1 = center_coord(x=Ycminpot1, x0=Yobserver, L=L)
    Zcminpot1 = center_coord(x=Zcminpot1, x0=Zobserver, L=L)
    # The comoving distance to the halo min potential
    r1 = (Xcminpot1**2 + Ycminpot1**2 + Zcminpot1**2) ** 0.5
    # Boolean mask of lightlike of second snapshot
    lightlike_mask1 = is_light_like(z=redshift0 + 0.05, r=r1, cosmo=cosmo)




    # If snap0 is spacelike and snap1 is lightlike, the halo 
    # has crossed the lightcone in between the snapshots and we save the properties.
    halo_crossed = (lightlike_mask0 == False) & (lightlike_mask1 == True)
    halo_crossed_count = np.sum(halo_crossed)
    print(f'Object crossed: {halo_crossed_count}')




    # Also save the FOF mass of the halos to later compare if the matching is successful,
    # and we can apply a mass cut to save disk space.
    fof_mass0 = tree['Subhalo/Mass_tot'][:][galaxyid - 1] * 1e10 # in Msun   
    fof_mass1 = tree['Subhalo/Mass_tot'][:][lastprog_galaxyid - 1] * 1e10 # in Msun
    structure_type0 = tree['Subhalo/Structuretype'][:][galaxyid - 1]
    structure_type1 = tree['Subhalo/Structuretype'][:][lastprog_galaxyid - 1]
    # We select those objects that are halos and have a mass above 5e12 Msun in both snapshots
    halo_selection = (fof_mass0 > 1e13) & (structure_type1 == 10) # & (fof_mass1 > 1e13) & (structure_type0 == 10)
    # update to do this selection early in the beginning, potentially speed up the computation.
    # IF YOU NEED TO.



    # Final mask
    halo_crossed = halo_crossed & halo_selection
    halo_crossed_count = np.sum(halo_crossed)
    print(f'Halo crossed: {halo_crossed_count}')

    # Save the properties
    f_spacelike.require_group(snap0)
    f_lightlike.require_group(snap1)
    f_spacelike_snap0 = f_spacelike[snap0]
    f_lightlike_snap1 = f_lightlike[snap1]

    # Save the properties of the halos that crossed the lightcone
    f_spacelike_snap0.create_dataset('GalaxyID', data=galaxyid[halo_crossed])
    f_spacelike_snap0.create_dataset('TopLeafID', data=topleafid0[halo_crossed])
    f_spacelike_snap0.create_dataset('SOAPID', data=soapid0[halo_crossed])
    f_spacelike_snap0.create_dataset('Xcminpot', data=Xcminpot0[halo_crossed])
    f_spacelike_snap0.create_dataset('Ycminpot', data=Ycminpot0[halo_crossed])
    f_spacelike_snap0.create_dataset('Zcminpot', data=Zcminpot0[halo_crossed])
    f_spacelike_snap0.create_dataset('M_fof', data=fof_mass0[halo_crossed])
    f_spacelike_snap0.create_dataset('redshift0', data=np.full(halo_crossed_count, redshift0))

    # And their progenitors
    f_lightlike_snap1.create_dataset('GalaxyID', data=lastprog_galaxyid[halo_crossed])
    f_lightlike_snap1.create_dataset('TopLeafID', data=topleafid1[halo_crossed])
    f_lightlike_snap1.create_dataset('SOAPID', data=soapid1[halo_crossed])
    f_lightlike_snap1.create_dataset('Xcminpot', data=Xcminpot1[halo_crossed])
    f_lightlike_snap1.create_dataset('Ycminpot', data=Ycminpot1[halo_crossed])
    f_lightlike_snap1.create_dataset('Zcminpot', data=Zcminpot1[halo_crossed])
    f_lightlike_snap1.create_dataset('M_fof', data=fof_mass1[halo_crossed])
    f_lightlike_snap1.create_dataset('redshift0', data=np.full(halo_crossed_count, redshift0 + 0.05))

    redshift0 += 0.05 # Throughout the simulation the redshift0 cadence is 0.05
    redshift1 += 0.05



# Close the file
f0.close()
