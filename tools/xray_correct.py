# ---------------------------------------------
# This script provide functions to apply 
# k-corrections and 0.2-2.3 keV to 0.1-2.4 keV 
# corrections for xray flux/luminosity
#
# Used the conversion table provided by Kostas
# Author:                      Yujie He
# Created on (MM/DD/YYYY):   11/08/2023
# ---------------------------------------------

import numpy as np
from scipy import interpolate

def band_conv(T):
    """
        Convert Luminosity/flux from 0.2-2.3keV band to 0.1-2.4keV band. 
        Temperature is in unit keV.

        Lx(0.1-2.4 keV) = band_conv * Lx(0.2-2.3 keV)
        fx(0.1-2.4 keV) = band_conv * fx(0.2-2.3 keV)
    """ 
    temperature = np.array([
        0.685, 0.768, 0.861, 0.967, 1.085, 1.217, 1.366, 1.534, 1.929, 2.429, 
        3.056, 3.430, 3.849, 4.319, 4.850, 5.432, 6.100, 6.846, 7.680, 8.617,
        9.670, 12.173, 15.320 
    ])
    conversion_ratio = np.array([
        1.162, 1.167, 1.173, 1.174, 1.176, 1.167, 1.156, 1.149, 1.138, 1.131, 
        1.127, 1.126, 1.126, 1.125, 1.123, 1.121, 1.120, 1.118, 1.118, 1.116, 
        1.116, 1.114, 1.114
    ])
    f = interpolate.interp1d(temperature, conversion_ratio, fill_value="extrapolate")
    return f(T)

def k_corr(T, z):
    """
        Convert flux from rest frame to observer frame to account for
        spectrum redshift. Temperature is in unit keV and redshift 
        is the observed redshift = peculiar redshift + cosmological redshift.

        fx(0.2-2.3 keV) = k_corr * Lx(0.2-2.3 keV) / 4 pi D_L^2
    """
    T = np.array(T)
    z = np.array(z)

    TRange = np.array([1, 2, 3, 5, 7, 8, 11])
    zRange = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.35, 0.4])
    TSpace, zSpace = np.meshgrid(TRange, zRange)
    TzSpace = np.stack((TSpace, zSpace), axis=2)
    TzSpace = np.reshape(TzSpace, (-1, 2))
    correction = np.array([
        [0.999, 1, 1.002, 1.004, 1.005, 1.005, 1.006],
        [0.99867, 1.0109131, 1.0174743, 1.024137, 1.02733288502, 1.02939756, 1.03132507],
        [1.0029477, 1.02601734938, 1.03524173, 1.0474193664, 1.05426951359, 1.05814524913, 1.0611594898],
        [1.00121082751, 1.0394055, 1.05513809, 1.0724321, 1.082270983, 1.0880963697355, 1.09232538648],
        [0.997952187561, 1.04640380173, 1.06894742799, 1.0937243687772, 1.10735911443, 1.11600206664, 1.12154738],
        [0.991628392, 1.058497091, 1.09395654946, 1.13424356355, 1.155561367, 1.16859396592, 1.17897757],
        [np.nan, np.nan, np.nan, 1.1532518034, 1.1790929, 1.19443598, 1.20629009765],
        [np.nan, np.nan, np.nan, np.nan, 1.20179262721, 1.219978874848, 1.23322758909],
    ])
    correction = np.ravel(correction)

    # Using the 'not nan' part to interpolate / extrapolate
    mask = correction > 0
    TzSpace = TzSpace[mask]
    correction = correction[mask]

    # Add method='nearest' for points outside of the region. 
    # https://stackoverflow.com/questions/41544829/use-fill-value-outside-boundaries-with-scipy-interpolate-griddata-and-method-nea/41550512#41550512

    # specifically for the table I used, there are no data here
    mask_out = ((z > 0.3) & (T < 5)) | ((z > 0.35) & (T < 7)) | (z < zRange[0]) | (z > zRange[-1]) | (T < TRange[0]) | (T > TRange[-1])
    corr = np.zeros(np.shape(T))
    if len(corr[mask_out]) != 0:
        # If outside of our data range, use the nearest value.
        Ti = T[mask_out]
        zi = z[mask_out]
        corr[mask_out] = interpolate.griddata(TzSpace, correction, xi=(Ti, zi), method='nearest')    
        
        # if inside, use linear interpolation.
        Ti = T[mask_out == False]
        zi = z[mask_out == False]
        corr[mask_out == False] = interpolate.griddata(TzSpace, correction, xi=(Ti, zi), method='linear') 
    else:
        corr = interpolate.griddata(TzSpace, correction, xi=(T, z), method='linear') 
    
    return corr
    
# test
if __name__ == '__main__':
    print("Testing with single digit values, in or out of the region.")
    print(band_conv(T=12), band_conv(T=0.5))
    print(k_corr(T=0.5, z=0.01), k_corr(T=9, z=0.4), k_corr(T=2, z=0.4))

    print("Testing with arrays.")
    print(band_conv(T=np.array([12, 0.5])))
    print(k_corr(T=np.array([0.5, 9, 2]), z=np.array([0.01, 0.4, 0.4])))


    

