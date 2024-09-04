"""
This script match the output of a-halo-crossing.py with the SOAP catalogue.
"""

import h5py
import os
from tqdm import tqdm

SOAP_DIR = '/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/SOAP/'
INPUT = '/data1/yujiehe/data/mock_lightcone/halo_lightcone_catalogue/halo_crossing.hdf5'

# -------------------------- command line arguments ----------------------------
import argparse
parser = argparse.ArgumentParser(description='Match the crossing data with SOAP catalogue.')
parser.add_argument('-i', '--input', type=str, help='Input file path', default=INPUT)
parser.add_argument('-s', '--soap', type=str, help='SOAP catalogue directory', default=SOAP_DIR)

# parse the arguments
args = parser.parse_args()
INPUT = args.input
SOAP_DIR = args.soap
# ------------------------------------------------------------------------------




# monkey patch a method to overwrite dataset if already exists and create new if doesn't
def write_dataset(self, name, data):
    if name in self:
        self[name][:] = data
    else:
        self.create_dataset(name, data=data) 
h5py._hl.group.Group.write_dataset = write_dataset

# Read the crossing data
f = h5py.File(INPUT, 'a')


for observer in list(f.keys()):
    
    # The observer group, in which there are lightlike and spacelike groups
    crossing_observer = f[observer]
    # The lightlike and spacelike groups of input
    f_lightlike = crossing_observer['lightlike'] # before crossing
    f_spacelike = crossing_observer['spacelike'] # after crossing
    
    # Loop over the snapshots
    for snap0 in reversed(list(f_spacelike.keys())):
        snap_num0 = int(snap0[-2:])
        snap_num1 = snap_num0 - 1
        snap1 = f'Snapshot{snap_num1:04d}'
        print(snap0, snap1)

        # Save the properties
        f_spacelike.require_group(snap0)
        f_lightlike.require_group(snap1)
        f_spacelike_snap0 = f_spacelike[snap0]
        f_lightlike_snap1 = f_lightlike[snap1]

        soapid0 = f_spacelike_snap0['SOAPID'][:]
        soapid1 = f_lightlike_snap1['SOAPID'][:]




        # Get from the SOAP catalogue
        cat = h5py.File(os.path.join(SOAP_DIR, f'halo_properties_{snap_num0:04d}.hdf5'), 'r')

        # Save the properties of the halos that crossed the lightcone
        f_spacelike_snap0.write_dataset('MfofSOAP', data=cat["FOFSubhaloProperties/TotalMass"][:][soapid0])
        f_spacelike_snap0.write_dataset('M500', data=cat['SO/500_crit/TotalMass'][:][soapid0])
        f_spacelike_snap0.write_dataset('GasMass', data=cat['SO/500_crit/GasMass'][:][soapid0])
        f_spacelike_snap0.write_dataset('LX0WithoutRecentAGNHeating', data=cat['SO/500_crit/XRayLuminosityWithoutRecentAGNHeating'][:,0][soapid0]) # eRosita 0.2-2.3 keV band
        f_spacelike_snap0.write_dataset('LX0WithoutRecentAGNHeatingCoreExcision', data=cat['SO/500_crit/XRayLuminosityWithoutRecentAGNHeatingCoreExcision'][:,0][soapid0])
        f_spacelike_snap0.write_dataset('LX0InRestframeWithoutRecentAGNHeating', data=cat['SO/500_crit/XRayLuminosityInRestframeWithoutRecentAGNHeating'][:,0][soapid0])
        f_spacelike_snap0.write_dataset('LX0InRestframeWithoutRecentAGNHeatingCoreExcision', data=cat['SO/500_crit/XRayLuminosityInRestframeWithoutRecentAGNHeatingCoreExcision'][:,0][soapid0])
        f_spacelike_snap0.write_dataset('SpectroscopicLikeTemperatureWithoutRecentAGNHeatingCoreExcision', data=cat['SO/500_crit/SpectroscopicLikeTemperatureWithoutRecentAGNHeatingCoreExcision'][:][soapid0])
        f_spacelike_snap0.write_dataset('Y5R500WithoutRecentAGNHeating', data=cat['SO/5xR_500_crit/ComptonYWithoutRecentAGNHeating'][:][soapid0])
        f_spacelike_snap0.write_dataset('Vx', data=cat['SO/500_crit/CentreOfMassVelocity'][:,0][soapid0])
        f_spacelike_snap0.write_dataset('Vy', data=cat['SO/500_crit/CentreOfMassVelocity'][:,1][soapid0])
        f_spacelike_snap0.write_dataset('Vz', data=cat['SO/500_crit/CentreOfMassVelocity'][:,2][soapid0])

        cat.close() # Don't forget to close the file



        # Get from next snapshot SOAP
        cat = h5py.File(os.path.join(SOAP_DIR, f'halo_properties_{snap_num1:04d}.hdf5'), 'r')

        # Save the properties of the halos that crossed the lightcone
        f_lightlike_snap1.write_dataset('MfofSOAP', data=cat["FOFSubhaloProperties/TotalMass"][:][soapid1])
        f_lightlike_snap1.write_dataset('M500', data=cat['SO/500_crit/TotalMass'][:][soapid1]) 
        f_lightlike_snap1.write_dataset('GasMass', data=cat['SO/500_crit/GasMass'][:][soapid1])
        f_lightlike_snap1.write_dataset('LX0WithoutRecentAGNHeating', data=cat['SO/500_crit/XRayLuminosityWithoutRecentAGNHeating'][:,0][soapid1]) # eRosita 0.2-2.3 keV band
        f_lightlike_snap1.write_dataset('LX0WithoutRecentAGNHeatingCoreExcision', data=cat['SO/500_crit/XRayLuminosityWithoutRecentAGNHeatingCoreExcision'][:,0][soapid1])
        f_lightlike_snap1.write_dataset('LX0InRestframeWithoutRecentAGNHeating', data=cat['SO/500_crit/XRayLuminosityInRestframeWithoutRecentAGNHeating'][:,0][soapid1])
        f_lightlike_snap1.write_dataset('LX0InRestframeWithoutRecentAGNHeatingCoreExcision', data=cat['SO/500_crit/XRayLuminosityInRestframeWithoutRecentAGNHeatingCoreExcision'][:,0][soapid1])
        f_lightlike_snap1.write_dataset('SpectroscopicLikeTemperatureWithoutRecentAGNHeatingCoreExcision', data=cat['SO/500_crit/SpectroscopicLikeTemperatureWithoutRecentAGNHeatingCoreExcision'][:][soapid1])
        f_lightlike_snap1.write_dataset('Y5R500WithoutRecentAGNHeating', data=cat['SO/5xR_500_crit/ComptonYWithoutRecentAGNHeating'][:][soapid1])
        f_lightlike_snap1.write_dataset('Vx', data=cat['SO/500_crit/CentreOfMassVelocity'][:,0][soapid1])
        f_lightlike_snap1.write_dataset('Vy', data=cat['SO/500_crit/CentreOfMassVelocity'][:,1][soapid1])
        f_lightlike_snap1.write_dataset('Vz', data=cat['SO/500_crit/CentreOfMassVelocity'][:,2][soapid1])

        cat.close() # Don't forget to close the file



f.close()