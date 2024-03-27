# ---------------------------------------------
# This script draw a sample from the FLAMINGO
# SOAP catalogue and lightcone data that mimicks
# the selection in Migkas 2020, Migkas 2021
# doi.org/10.1051/0004-6361/202140296
# 
# It accounts for:
#   - relativistic peculiar redshift due 
# to motion of clusters;
#   - band conversion from 0.2-2.3 keV in SOAP 
# catalog to 0.1-2.4 keV of eRosita;
#   - manual conversion to observer frame from rest
# frame of source;
#   - add the Lcore/Ltot for discussion without
# any cut based on it. You can decide how to make
# that cut later. The output list is sorted in 
# decending order of Lcore/Ltot fraction, so that 
# you can make the most concentrated N clusters 
# sample by getting the first N lines.
#
# Author                       : Yujie He
# Created on (MM/DD/YYYY)      : 01/15/2024
# Last Modified on (MM/DD/YYYY): 01/31/2024
# ---------------------------------------------

import pandas as pd
import sys
sys.path.append('/home/yujiehe/anisotropy-flamingo')
from tools.xray_correct import *
import numpy as np
import argparse

# ---------------COMMAND LINE ARGUMENTS-----------------------------------------
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="A description of what your program does")

# Add arguments
parser.add_argument('-i', '--input', type=str, help='Input file path')
parser.add_argument('-o', '--output', type=str, help='Output file path', default=None)

# Parse the arguments
args = parser.parse_args()

# Now you can use the arguments
input = args.input
output = args.output

if output is None:
    output = input.replace('.hdf5', '.csv')
    output = output.replace('halo_properties_in_', 'samples_in_')

# ----------------CONFIGURATION-------------------------------------------------

# # Input file is the conbination of lightcone and the SOAP catalog
# input = '/data1/yujiehe/data/halo_properties_in_PLANCK_lightcone0.hdf5'

# # Output
# output = '/data1/yujiehe/data/samples-PLANCK-lightcone0.csv'

# Save only larger than this flux
flux_cut = 5e-12

# Saving latitudes larger than this cut (degrees)
latitude_cut = 20 

# Make a namespace for our descriptive yet horribly long column names
class Columns:
    LX             = 'LX0InRestframeWithoutRecentAGNHeating'
    LXCoreExcision = 'LX0InRestframeWithoutRecentAGNHeatingCoreExcision'
    GasT           = 'GasTemperatureWithoutRecentAGNHeatingCoreExcision'
    SpecT          = 'SpectroscopicLikeTemperatureWithoutRecentAGNHeatingCoreExcision'
    YSZ            = 'Y5R500WithoutRecentAGNHeating'

# ------------END CONFIGURATION-------------------------------------------------


print(f'Input data: {input}')

# Load data
data = pd.read_hdf(input, 'lightcone')
Noriginal = len(data)

# Drop duplicates, keep the lowest redshift one
data.sort_values(by='redshift', inplace=True)
data = data.drop_duplicates(subset=['SOAPID', 'snap_num'], keep='first') 
# Update Mar 13 2024, SOAP IDs are not unique, now we only drop duplicates if it's from the same snapshot.
# This way we removed the absolutely same clusters, but did not remove the same cluster at different redshifts.

# Unit conversion
data[Columns.GasT]      /= 11604525 # temperature unit Kelvin to keV
data[Columns.SpecT]     /= 11604525 # temperature unit Kelvin to keV
data[Columns.YSZ]       /= (3.08567758e+21)**2 # Ysz unit cm^2 to kpc^2

# Remove small, cool objects (which are not clusters).
data = data[data[Columns.SpecT] > 0.5]    # Temperature cut at 0.5 keV
data = data[data['M500'] > 5e12]      # Mass cut 5e12 Msun

print(f'{Noriginal} clusters to begin with, after dropping duplicate, T > 0.5keV and M500 > 5e12 cut. {len(data)} clusters left.')

# Correct for X-ray bug in SOAP catalogue
data[Columns.LX]             /= (data['redshift'] + 1)**3
data[Columns.LXCoreExcision] /= (data['redshift'] + 1)**3


# ---------CALCULATE PECULIAR REDSHIFT------------------------------------------
c = 299792.458                  # the speed of light in km/s

vx = data['Vx']                 # velocities in km/s
vy = data['Vy']
vz = data['Vz']

x = data['x_lc']
y = data['y_lc']
z = data['z_lc']

los_v = (vx*x + vy*y + vz*z) / (x**2 + y**2 + z**2)**0.5
beta = los_v / c
z_pec = (1 + los_v/c) / (1 - beta**2)**0.5 - 1
z_obs = (data['redshift'] + 1) * (z_pec + 1) - 1


# ------------------CALCULATE FLUX----------------------------------------------
# distance used lightcone 'lightconeXcminpot'. I think it's comoving?
Dco = (data['x_lc']**2 + data['y_lc']**2 + data['z_lc']**2)**0.5 
DL = Dco * (z_obs + 1)      # luminosity distance from comoving distan
DL *= 3.08567758e24        # from Mpc to cm, we use the exact definition of parsec and AU: 648000/np.pi*149_597_870_700*1e8

flux = data[Columns.LX]\
        / (4 * np.pi * DL**2)\
        * band_conv(T=data[Columns.SpecT])\
        * k_corr(T=data[Columns.SpecT], z=z_obs)     # flux in erg/s/cm^2 # Since in observation all temperature are spectroscopic and we are in fact using the observational conversions, we use spectroscopic-like temperatures here also.




# ---------------MAKE FLUX AND LATITUDE CUT-------------------------------------
cut = (flux >= flux_cut) & (np.abs(data['theta_on_lc']) > latitude_cut)
cut_data = data[cut].copy(deep=True)
cut_z_obs = z_obs[cut]
cut_flux = flux[cut]
print(f'Final sample: {len(cut_data)}')


# --------------CALCULATE FRACTION----------------------------------------------
fraction = (cut_data[Columns.LX] - cut_data[Columns.LXCoreExcision]) / cut_data[Columns.LX]


# Add other useful quantities to our samples
cut_data['Lcore/Ltot']       = fraction
cut_data['ObservedRedshift'] = cut_z_obs
cut_data['Flux']             = cut_flux
# Sort descending w.r.t. Lcore/Ltot
cut_data.sort_values('Lcore/Ltot', ascending=False, inplace=True)

# Save
cut_data.to_csv(output, index=False)

print(f'Sample saved: {output}.')
