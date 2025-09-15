# Fisrt we find a halo and extract its coordinate and r500.

import h5py
import swiftsimio as sw

from unyt import unyt_array
from unyt import Mpc, Msun
import numpy as np

Nbin = 30
bintype = 'log'

snapshot = '/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/snapshots/flamingo_0077/flamingo_0077.hdf5'
cat = h5py.File('/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/SOAP/halo_properties_0077.hdf5', 'r')
redshift = 0

mass_mask = cat['SO/500_crit/TotalMass'][:] > 10**14.5
idx_list = np.where(mass_mask == True)[0]  # get M>10^14.5 clusters' index

# Track progress
count = 0
N = len(idx_list)
print('total of {} clusters'.format(N))

SB_list = []
for idx in idx_list:
    xc = unyt_array(cat['VR/CentreOfPotential'][:, 0], Mpc)[idx] / (1 + redshift) # Dp = a * Sk(r) = Sk(r) / (1+z), Dcomoving = Sk(r)
    yc = unyt_array(cat['VR/CentreOfPotential'][:, 1], Mpc)[idx] / (1 + redshift)
    zc = unyt_array(cat['VR/CentreOfPotential'][:, 2], Mpc)[idx] / (1 + redshift)
    r500c = unyt_array(cat['SO/500_crit/SORadius'][()], Mpc)[idx]

    # Load region of snapshot
    load_region = unyt_array([[xc - r500c, xc + r500c], [yc - r500c, yc + r500c], [zc - r500c, zc + r500c]], xc.units)

    # Convert to comoving
    comoving_load_region = unyt_array(load_region * (1 + redshift), load_region[0][0].units)

    mask = sw.mask(snapshot)
    mask.constrain_spatial(comoving_load_region)
    data = sw.load(snapshot, mask=mask)

    r = ((data.gas.coordinates[:, 0] - xc)**2 + (data.gas.coordinates[:, 1] - yc)**2)**0.5
    z = data.gas.coordinates[:, 2] - zc

    if bintype == 'log':
        logR = np.linspace(-2, 0, Nbin)
        R = 10**logR
    elif bintype == 'equal':  # even spaced bins
        R = np.linspace(0.01, 1, Nbin)

    SB = np.zeros(Nbin - 1)
    for i in range(Nbin - 1):
        binmask = (r > R[i] * r500c) & (r <= R[i+1] * r500c) & (np.abs(z) < 1.5 * r500c) # from 0.01 to 1 r500c
        area = np.pi * (R[i+1]**2 - R[i]**2) # S = πr+^2 - πr-^2, use precise area
        SB[i] += np.sum(data.gas.xray_luminosities.ROSAT[binmask]) / area

    SB_list.append(SB)

    count += 1              # track progress5
    print('progress: {}/{}'.format(count, N))

SB_list = np.array(SB_list)
np.savetxt('sb_30logbins.txt', SB_list)
