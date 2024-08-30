
"""
Regard any confusion one may have:
    - soapid is the index in soap catalogue
    - treeid is the index in merger tree. SOAP/Snapshot00xx takes soapid and gives treeid.
    - Subhalo/ID is the velociraptor ID, higher than soapid by 1
    - MergerTree/GalaxyID is treeid + 1, noted as galaxyid here
"""

import h5py
import numpy as np

# on cosma8
# MERGER_TREE_FILE = '/cosma8/data/dp004/jch/FLAMINGO/MergerTrees/ScienceRuns/L1000N1800/HYDRO_FIDUCIAL/trees_f0.1_min10_max100/vr_trees.hdf5'
# OUTPUT = './halo_crossing.hdf5'

# on hypernova
MERGER_TREE_FILE = '/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/merger_trees/vr_trees.hdf5'
OUTPUT = '/data1/yujiehe/data/halo_crossing.hdf5'
MAX_REDSHIFT = 0.25 # max redshift considered

L = 1000   # simulation box size in comoving Mpc (cMpc)

RUN_TYPE = 'single'
if RUN_TYPE == 'single':
    # Single observer test
    Xobs = 3*L/4
    Yobs = 3*L/4
    Zobs = 3*L/4
    N = 0
elif RUN_TYPE == 'multi':   
    # Multi-observer run
    N = 4



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
    x_x0 = (x - x0 + L/2) % L - L/2
    return x_x0

#-------------------------------------------------------------------------------

def get_crossed_mask(
                  X0, Y0, Z0, z0,
                  X1, Y1, Z1, z1,
                  ):
    """
    Given the coordinates of same objects between two snapshots, find if they 
    crossed the lightcone. Will shift the coordinate to observer center.

    Note
    --
    All coodinates are comoving.
    """
    r0 = (X0**2 + Y0**2 + Z0**2)**0.5
    r1 = (X1**2 + Y1**2 + Z1**2)**0.5

    mask = (is_light_like(z=z0, r=r0, cosmo=cosmo) == False) & (is_light_like(z=z1, r=r1, cosmo=cosmo))
    
    return mask


def mesh(N, L):
    """
    Make N**3 evenly spaced coordinates.
    """
    X = np.linspace(-L/2, L/2, num=N)
    Y = np.linspace(-L/2, L/2, num=N)
    Z = np.linspace(-L/2, L/2, num=N)
    Xcoord, Ycoord, Zcoord = np.meshgrid(X, Y, Z, indexing='ij')
    
    # output as a 1D list
    Xcoord = np.ravel(Xcoord)
    Ycoord = np.ravel(Ycoord)
    Zcoord = np.ravel(Zcoord)

    return Xcoord, Ycoord, Zcoord




# Initialize list of observers/lightcones
if RUN_TYPE == 'single':
    Xobsarr = np.array([Xobs])
    Yobsarr = np.array([Yobs])
    Zobsarr = np.array([Zobs])
elif RUN_TYPE == 'multi':
    Xobsarr, Yobsarr, Zobsarr = mesh(N=N, L=L)


# Create/Open hdf5 file
f0 = h5py.File(OUTPUT, 'a')


obsnum = 0
for Xobs, Yobs, Zobs in zip(Xobsarr, Yobsarr, Zobsarr):

    print(f'Creating observer: {obsnum}')
    # Create group for output
    lightcone_name = f'lightcone{obsnum:04d}'
    f0.require_group(lightcone_name)
    # Save the observer coordinates, in Mpc
    f = f0[lightcone_name]
    f.attrs['Xobs'] = Xobs
    f.attrs['Yobs'] = Yobs
    f.attrs['Zobs'] = Zobs
    # Create groups for lightlike and spacelike
    f.require_group('lightlike')
    f.require_group('spacelike')
    f_lightlike = f['lightlike']
    f_spacelike = f['spacelike']
     


    # Read the merger tree
    tree = h5py.File(MERGER_TREE_FILE, 'r')
    # Set snapshot names
    snapshot_names = list(tree['SOAP'].keys())
    snapshot_names = snapshot_names[::-1] # In reversed order, redshift 0, snapshot 77 of 1Gpc or 78 of 2.8 Gpc at the front


    redshift0 = 0.0       # redshift of snapshot0
    redshift1 = 0.05      # redshift of snapshot1
    for i, snap0 in enumerate(snapshot_names):
        # Only account up to a given range
        if redshift0 > MAX_REDSHIFT:
            break

        snap1 = snapshot_names[i + 1]   # snapshot with redshift z + 0.05
        print(snap0, snap1)                    # log
        real_snapnum0 = int(snap0[-2:])
        real_snapnum1 = int(snap1[-2:])


        # go to SOAP find the GalaxyID of the clusters in a catalogue
        treeid0 = tree['SOAP/' + snap0][:] # SOAP/Snapshot00XX + 1 = MergerTree/GalaxyID
        galaxyid0 = treeid0 + 1
        # retrieve the galaxy id of the main progenitor
        treeid1 = treeid0 + 1
        galaxyid1 = treeid1 + 1
        # galaxyid1 = tree['MergerTree/GalaxyID'][:][treeid0 + 1]

        # mass + structure type + snapnum filter
        # load data
        massfof = tree['Subhalo/Mass_tot'][:] * 1e10
        structure_type = tree['Subhalo/Structuretype'][:]
        snapnum = tree['Subhalo/SnapNum'][:]
        # select mass, structure type, snapnum based on treeid
        massfof0 = massfof[treeid0]
        massfof1 = massfof[treeid1]
        structure_type0 = structure_type[treeid0]
        structure_type1 = structure_type[treeid1]
        snapnum0 = snapnum[treeid0]
        snapnum1 = snapnum[treeid1]
        # one line mask
        halo_mask = (massfof0 > 1e13) & (massfof1 > 1e13) &\
            (structure_type0 == 10) & (structure_type1 == 10) &\
            (snapnum0 == real_snapnum0) & (snapnum1 == real_snapnum1) 
        # save some memory
        del massfof
        del structure_type
        del snapnum




        # filter treeids, galaxyids, massfofs, structure_types, snapnums
        treeid0 = treeid0[halo_mask]
        treeid1 = treeid1[halo_mask]
        galaxyid0 = galaxyid0[halo_mask]
        galaxyid1 = galaxyid1[halo_mask]
        massfof0 = massfof0[halo_mask]
        massfof1 = massfof1[halo_mask]
        structure_type0 = structure_type0[halo_mask]
        structure_type1 = structure_type1[halo_mask]
        snapnum0 = snapnum0[halo_mask]
        snapnum1 = snapnum1[halo_mask]
        # from now on only use filtered quantities




        # go to merger tree and find the minimal potential coordinates; most bound black hole is 0 for some objects so we'd rather use this
        Xcminpot0 = tree['Subhalo/Xcminpot'][:][treeid0] * (redshift0 + 1) # pMpc to cMpc
        Ycminpot0 = tree['Subhalo/Ycminpot'][:][treeid0] * (redshift0 + 1)
        Zcminpot0 = tree['Subhalo/Zcminpot'][:][treeid0] * (redshift0 + 1)
        # progenitors
        Xcminpot1 = tree['Subhalo/Xcminpot'][:][treeid1] * (redshift1 + 1)
        Ycminpot1 = tree['Subhalo/Ycminpot'][:][treeid1] * (redshift1 + 1)
        Zcminpot1 = tree['Subhalo/Zcminpot'][:][treeid1] * (redshift1 + 1)
        print(np.min(Xcminpot1), np.max(Xcminpot1))
        # shift to observer center
        Xcminpot0 = center_coord(Xcminpot0, x0=Xobs, L=L)
        Ycminpot0 = center_coord(Ycminpot0, x0=Yobs, L=L)
        Zcminpot0 = center_coord(Zcminpot0, x0=Zobs, L=L)
        Xcminpot1 = center_coord(Xcminpot1, x0=Xobs, L=L)
        Ycminpot1 = center_coord(Ycminpot1, x0=Yobs, L=L)
        Zcminpot1 = center_coord(Zcminpot1, x0=Zobs, L=L)
        print(np.min(Xcminpot1), np.max(Xcminpot1))

        # the halos that crossed the lightcone
        cross_mask = get_crossed_mask(X0=Xcminpot0, Y0=Ycminpot0, Z0=Zcminpot0, z0=redshift0,
                                      X1=Xcminpot1, Y1=Ycminpot1, Z1=Zcminpot1, z1=redshift1)
        # XYZ shifted to observer-centered
        



        # filter treeids, galaxyids, massfofs, snapnums, coordinates
        # structure type is not needed
        treeid0 = treeid0[cross_mask]
        treeid1 = treeid1[cross_mask]
        galaxyid0 = treeid0 + 1
        galaxyid1 = treeid1 + 1
        massfof0 = massfof0[cross_mask] # in Msun
        massfof1 = massfof1[cross_mask]
        snapnum0 = snapnum0[cross_mask] # snapshot number
        snapnum1 = snapnum1[cross_mask]
        Xcminpot0 = Xcminpot0[cross_mask] # in cMpc
        Ycminpot0 = Ycminpot0[cross_mask]
        Zcminpot0 = Zcminpot0[cross_mask]
        Xcminpot1 = Xcminpot1[cross_mask]
        Ycminpot1 = Ycminpot1[cross_mask]
        Zcminpot1 = Zcminpot1[cross_mask]

        # log output
        halo_cross_count = len(treeid0) 
        print('Halo crossed:', len(treeid0)) # output log

        # also get soapid; further filtering not needed cause treeid already filtered
        soapid0 = tree['Subhalo/ID'][:][treeid0] - 1
        soapid1 = tree['Subhalo/ID'][:][treeid1] - 1
        topleafid0 = tree['MergerTree/TopLeafID'][:][treeid0]
        topleafid1 = tree['MergerTree/TopLeafID'][:][treeid1]




        # create group
        f_spacelike.require_group(snap0)
        f_lightlike.require_group(snap1)
        f_spacelike_snap0 = f_spacelike[snap0]
        f_lightlike_snap1 = f_lightlike[snap1]
        # save properties
        f_spacelike_snap0.create_dataset('GalaxyID', data=galaxyid0)
        f_spacelike_snap0.create_dataset('TopLeafID', data=topleafid0)
        f_spacelike_snap0.create_dataset('SOAPID', data=soapid0)
        f_spacelike_snap0.create_dataset('Xcminpot', data=Xcminpot0)
        f_spacelike_snap0.create_dataset('Ycminpot', data=Ycminpot0)
        f_spacelike_snap0.create_dataset('Zcminpot', data=Zcminpot0)
        f_spacelike_snap0.create_dataset('Mass_tot', data=massfof0)
        f_spacelike_snap0.create_dataset('redshift', data=np.full(halo_cross_count, redshift0))
        # spacelike
        f_lightlike_snap1.create_dataset('GalaxyID', data=galaxyid1)
        f_lightlike_snap1.create_dataset('TopLeafID', data=topleafid1)
        f_lightlike_snap1.create_dataset('SOAPID', data=soapid1)
        f_lightlike_snap1.create_dataset('Xcminpot', data=Xcminpot1)
        f_lightlike_snap1.create_dataset('Ycminpot', data=Ycminpot1)
        f_lightlike_snap1.create_dataset('Zcminpot', data=Zcminpot1)
        f_lightlike_snap1.create_dataset('Mass_tot', data=massfof1)
        f_lightlike_snap1.create_dataset('redshift', data=np.full(halo_cross_count, redshift1))


        redshift0 += 0.05
        redshift1 += 0.05
    
    obsnum += 1


tree.close()
f0.close()
        



    

