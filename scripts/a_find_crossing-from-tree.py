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
    Tell if an object at a comoving coordinate is observable at given redshift.
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
    x_x0 = (x - x0 - L/2) % L + L/2
    return x_x0

#-------------------------------------------------------------------------------

# Read the merger tree
merger_tree = h5py.File(MERGER_TREE_FILE, 'r')

# Set snapshot names
snapshot_names = list(merger_tree['SOAP'].keys())
snapshot_names = snapshot_names[::-1] # In reversed order, redshift 0, snapshot 77 of 1Gpc or 78 of 2.8 Gpc at the front

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
f_lightlike = f['lightlike']
f_spacelike = f['spacelike']


# Initialize
redshift = 0.0
for i, snap0 in tqdm(enumerate(snapshot_names)):
    snap1 = snapshot_names[i + 1]   # snapshot with redshift z + 0.05
    print(snap0, snap1)                    # log

    # Only account up to a given range
    if redshift > REDSHIFT_RANGE:
        break

    
    """
    We find the halos that are lightlike in the current snapshot, and its
    progenitors are spacelike in the next snapshot. We save the coordinates
    and redshift of these halos and their progenitors. The redshift is simply
    the snapshot redshift and the next snapshot redshift. The precise redshift
    at crossing is to be interpolated later.
    """
    

    # Go to SOAP find the GalaxyID of the clusters in a catalogue
    galaxy_ids0 = merger_tree['SOAP/' + snap0][:] + 1

    # Also the SOAP ids, soap ids are array indices in the SOAP catalogue, starts from 0.
    soap_ids0 = np.arange(len(galaxy_ids0))
    
    # Sanity check if the galaxy ids & soap ids are matching
    print(np.sum(merger_tree['Subhalo/ID'][:][galaxy_ids0 - 1] -1 - soap_ids0))
    assert np.sum(merger_tree['Subhalo/ID'][:][galaxy_ids0 - 1] -1 - soap_ids0) == 0 # The Subhalo/ID is 1 + the SOAP id (array index in the SOAP catalogue)
    print('SOAP ids and Galaxy ids matched.')
 
    # Retrieve the galaxy id of the last progenitor
    last_prog_galaxy_ids0 = merger_tree['MergerTree/LastProgID'][:][galaxy_ids0 - 1] # beware that galaxy ids start from 1
    top_leaf_ids0 = merger_tree['MergerTree/TopLeafID'][:][galaxy_ids0 - 1]

    # Go to merger tree and find the coordinates in
    # halo potential minimum & most bound black hole particles
    Xcmbp_bh0 = merger_tree['Subhalo/Xcmbp_bh'][:][galaxy_ids0 - 1]
    Ycmbp_bh0 = merger_tree['Subhalo/Ycmbp_bh'][:][galaxy_ids0 - 1]
    Zcmbp_bh0 = merger_tree['Subhalo/Zcmbp_bh'][:][galaxy_ids0 - 1]

    # Shift the coordinate center to observer center
    Xcmbp_bh0 = center_coord(x=Xcmbp_bh0, x0=Xobserver, L=L)
    Ycmbp_bh0 = center_coord(x=Ycmbp_bh0, x0=Yobserver, L=L)
    Zcmbp_bh0 = center_coord(x=Zcmbp_bh0, x0=Zobserver, L=L)

    # The comoving distance to the halo min potential
    r0 = (Xcmbp_bh0**2 + Ycmbp_bh0**2 + Zcmbp_bh0**2) ** 0.5

    # Boolean mask of lightlike of first snapshot
    lightlike_mask0 = is_light_like(z=redshift, r=r0, cosmo=cosmo)

    """ Now the progenitors. """    

    # Retrieve the soapids of the last progenitor
    soap_ids1 = merger_tree['Subhalo/ID'][:][last_prog_galaxy_ids0 - 1] - 1 # The Subhalo/ID is 1 + the SOAP id (array index in the SOAP catalogue)
    top_leaf_ids1 = merger_tree['MergerTree/TopLeafID'][:][last_prog_galaxy_ids0 - 1]

    # Find coordinates of the last progenitor
    Xcmbp_bh1 = merger_tree['Subhalo/Xcmbp_bh'][:][last_prog_galaxy_ids0 - 1]
    Ycmbp_bh1 = merger_tree['Subhalo/Ycmbp_bh'][:][last_prog_galaxy_ids0 - 1]
    Zcmbp_bh1 = merger_tree['Subhalo/Zcmbp_bh'][:][last_prog_galaxy_ids0 - 1]

    # Shift the coordinate center to observer center
    Xcmbp_bh1 = center_coord(x=Xcmbp_bh1, x0=Xobserver, L=L)
    Ycmbp_bh1 = center_coord(x=Ycmbp_bh1, x0=Yobserver, L=L)
    Zcmbp_bh1 = center_coord(x=Zcmbp_bh1, x0=Zobserver, L=L)

    # The comoving distance to the halo min potential
    r1 = (Xcmbp_bh1**2 + Ycmbp_bh1**2 + Zcmbp_bh1**2) ** 0.5

    # Boolean mask of lightlike of second snapshot
    lightlike_mask1 = is_light_like(z=redshift + 0.05, r=r1, cosmo=cosmo)


    # If the first snapshot is lightlike and the second is spacelike, the halo 
    # has crossed the lightcone in between the snapshots and we save the properties.
    halo_crossed = (lightlike_mask0 is True) & (lightlike_mask1 is False)
    halo_crossed_count = np.sum(halo_crossed)
    print(f'Object crossed: {halo_crossed_count}')


    # Also save the FOF mass of the halos to later compare if the matching is successful,
    # and we can apply a mass cut to save disk space.
    fof_mass0 = merger_tree['Subhalo/Mass_tot'][:][galaxy_ids0 - 1] * 1e10 # in Msun   
    fof_mass1 = merger_tree['Subhalo/Mass_tot'][:][last_prog_galaxy_ids0 - 1] * 1e10 # in Msun
    structure_type0 = merger_tree['Subhalo/Structuretype'][:][galaxy_ids0 - 1]
    structure_type1 = merger_tree['Subhalo/Structuretype'][:][last_prog_galaxy_ids0 - 1]

    # We select those objects that are halos and have a mass above 5e12 Msun in both snapshots
    halo_selection = (fof_mass0 > 5e12) & (fof_mass1 > 5e12) & (structure_type0 == 10) & (structure_type1 == 10)
    
    # Final mask
    halo_crossed = halo_crossed & halo_selection
    halo_crossed_count = np.sum(halo_crossed)
    print(f'Object crossed: {halo_crossed_count}')

    # Save the properties
    f_lightlike.require_group(snap0)
    f_spacelike.require_group(snap1)
    f_lightlike_snap0 = f_lightlike[snap0]
    f_spacelike_snap1 = f_spacelike[snap1]

    # Save the properties of the halos that crossed the lightcone
    f_lightlike_snap0.create_dataset('GalaxyID', data=galaxy_ids0[halo_crossed])
    f_lightlike_snap0.create_dataset('TopLeafID', data=top_leaf_ids0[halo_crossed])
    f_lightlike_snap0.create_dataset('SOAPID', data=soap_ids0[halo_crossed])
    f_lightlike_snap0.create_dataset('Xcmbp_bh_lc', data=Xcmbp_bh0[halo_crossed])
    f_lightlike_snap0.create_dataset('Ycmbp_bh_lc', data=Ycmbp_bh0[halo_crossed])
    f_lightlike_snap0.create_dataset('Zcmbp_bh_lc', data=Zcmbp_bh0[halo_crossed])
    f_lightlike_snap0.create_dataset('M_fof', data=fof_mass0[halo_crossed])
    f_lightlike_snap0.create_dataset('redshift', data=np.full(halo_crossed_count, redshift))

    # And their progenitors
    f_spacelike_snap1.create_dataset('GalaxyID', data=last_prog_galaxy_ids0[halo_crossed])
    f_spacelike_snap1.create_dataset('TopLeafID', data=top_leaf_ids1[halo_crossed])
    f_spacelike_snap1.create_dataset('SOAPID', data=soap_ids1[halo_crossed])
    f_spacelike_snap1.create_dataset('Xcmbp_bh_lc', data=Xcmbp_bh1[halo_crossed])
    f_spacelike_snap1.create_dataset('Ycmbp_bh_lc', data=Ycmbp_bh1[halo_crossed])
    f_spacelike_snap1.create_dataset('Zcmbp_bh_lc', data=Zcmbp_bh1[halo_crossed])
    f_spacelike_snap1.create_dataset('M_fof', data=fof_mass1[halo_crossed])
    f_spacelike_snap1.create_dataset('redshift', data=np.full(halo_crossed_count, redshift + 0.05))

    redshift += 0.05 # Throughout the simulation the redshift cadence is 0.05



# Close the file
f0.close()
