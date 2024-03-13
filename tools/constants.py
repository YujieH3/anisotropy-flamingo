import numpy as np

# Make a namespace for our descriptive yet horribly long column names
COLUMNS ={
    'LX'             : 'LX0InRestframeWithoutRecentAGNHeating',
    'T'              : 'SpectroscopicLikeTemperatureWithoutRecentAGNHeatingCoreExcision',
    'YSZ'            : 'Y5R500WithoutRecentAGNHeating',
    'M'              : 'GasMass',
    'LXCoreExcision' : 'LX0InRestframeWithoutRecentAGNHeatingCoreExcision',
}

# Parameters as in M21 (doi.org/10.1051/0004-6361/202140296) table 2
# changed pivot value CX to minimum A-B correlation at 2024-02-20
CONST = { 
    'LX-T': {
        'CY'   : 1e44, #1e44, # erg/s         # median 1.260e44
        'CX'   : 3.8,    #4,    # keV           # median 3.321
        'gamma': -1,
        'N'    : 313,
    },
    'YSZ-T': {
        'CY'   : 20, #35, # kpc^2            # median 21.906
        'CX'   : 3.2,  #5,  # keV              # median 3.247
        'gamma': 1,
        'N'    : 260
    },
    'M-T': {
        'CY'   : 3e13, #2e13, # Msun           # median 2.874e13
        'CX'   : 3.6,  #3,    # keV            # median 3.335
        'gamma': 1,
        'N'    : 300, # temporary
    },
    'LX-YSZ': {
        'CY'   : 3e43, #1e44, # erg/s          # median 3.056e43
        'CX'   : 20,   #60,   # kpc^2          # median 21.244
        'gamma': -8/5,
        'N'    : 460,
    },
    'LX-M': {
        'CY'   : 1e44, #1e44, # erg/s          # median 1.244e44
        'CX'   : 3e13, #2e13, # Msun           # median 2.747e13
        'gamma': -2,
        'N'    : 300, # temporary
    },
    'YSZ-M': {
        'CY'   : 7,    #20,   # kpc^2         # median 6.753
        'CX'   : 3e13, #2e13, # Msun          # median 2.747e13
        'gamma': -2/3,
        'N'    : 300, # temporary
    },
}

LARGE_RANGE = {
    'LX-T': {
        'logA_min': np.log10(0.1), 'logA_max': np.log10(2.0),
        'B_min'   : 1,             'B_max'   : 3.5,
        'scat_min': 0.09,          'scat_max': 1,
    },
    'YSZ-T': {
        'logA_min': np.log10(0.1), 'logA_max': np.log10(3.0),
        'B_min'   : 1.5,           'B_max'   : 3.5,
        'scat_min': 0.06,          'scat_max': 1,
    },
    'M-T': {
        'logA_min': np.log10(0.1), 'logA_max': np.log10(2.0),
        'B_min'   : 1.0,           'B_max'   : 3,
        'scat_min': 0.05,          'scat_max': 1,
    },
    'LX-YSZ': {
        'logA_min': np.log10(1), 'logA_max': np.log10(10),
        'B_min'   : 0.5,         'B_max'   : 2.0,
        'scat_min': 0.10,        'scat_max': 1,
    },
    'LX-M': {
        'logA_min': np.log10(0.1), 'logA_max': np.log10(1.5),
        'B_min'   : 1.0,           'B_max'   : 2.0,
        'scat_min': 0.04,          'scat_max': 1,
    },
    'YSZ-M': {
        'logA_min': np.log10(1), 'logA_max': np.log10(10),
        'B_min'   : 0.5,         'B_max'   : 2.0,
        'scat_min': 0.06,        'scat_max': 1,
    },
}

MID_RANGE = { # round 5 sigma range to .5. Round up upper range and round down lower range
    'LX-T': {
        'logA_min': np.log10(1.0), 'logA_max': np.log10(2.0),
        'B_min'   : 1.5,           'B_max'   : 3.5,
        'scat_min': 0.05,          'scat_max': 1,
    },
    'YSZ-T': {
        'logA_min': np.log10(0.5), 'logA_max': np.log10(2.5),
        'B_min'   : 2,             'B_max'   : 3.5,
        'scat_min': 0.05,          'scat_max': 1,
    },
    'M-T': {
        'logA_min': np.log10(0.5), 'logA_max': np.log10(1.5),
        'B_min'   : 1.5,           'B_max'   : 2.5,
        'scat_min': 0.05,          'scat_max': 1,
    },
    'LX-YSZ': {
        'logA_min': np.log10(2), 'logA_max': np.log10(4),
        'B_min'   : 0.5,         'B_max'   : 1.5,
        'scat_min': 0.05,        'scat_max': 1,
    },
    'LX-M': {
        'logA_min': np.log10(1), 'logA_max': np.log10(1.5),
        'B_min'   : 1,           'B_max'   : 1.5,
        'scat_min': 0.05,        'scat_max': 1,
    },
    'YSZ-M': {
        'logA_min': np.log10(3), 'logA_max': np.log10(5),
        'B_min'   : 1,           'B_max'   : 2,
        'scat_min': 0.05,        'scat_max': 1,
    },
}

FIVE_MAX_RANGE = { # Use check_set_range.ipynb to set this range
    'LX-T': {
        'logA_min': np.log10(1.19), 'logA_max': np.log10(1.84),
        'B_min'   : 1.69,           'B_max'   : 3.31,
        'scat_min': 0.05,           'scat_max': 1,
    },
    'YSZ-T': {
        'logA_min': np.log10(0.80), 'logA_max': np.log10(1.60),
        'B_min'   : 2.40,           'B_max'   : 3.40,
        'scat_min': 0.05,          'scat_max': 1,
    },
    'M-T': {
        'logA_min': np.log10(0.93), 'logA_max': np.log10(1.21),
        'B_min'   : 1.55,           'B_max'   : 2.45,
        'scat_min': 0.05,           'scat_max': 1,
    },
    'LX-YSZ': {
        'logA_min': np.log10(2.27), 'logA_max': np.log10(3.68),
        'B_min'   : 0.70,           'B_max'   : 1.06,
        'scat_min': 0.05,           'scat_max': 1,
    },
    'LX-M': {
        'logA_min': np.log10(0.99), 'logA_max': np.log10(1.46),
        'B_min'   : 1.09,           'B_max'   : 1.42,
        'scat_min': 0.05,           'scat_max': 1,
    },
    'YSZ-M': {
        'logA_min': np.log10(3.11), 'logA_max': np.log10(4.56),
        'B_min'   : 1.09,           'B_max'   : 1.63,
        'scat_min': 0.05,           'scat_max': 1,
    },
}