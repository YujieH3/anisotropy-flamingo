import numpy as np
from numba import njit

# Make a namespace for our descriptive yet horribly long column names
COLUMNS_RAW ={
    'LX'             : 'LX0InRestframeWithoutRecentAGNHeating',
    'LXCoreExcision' : 'LX0InRestframeWithoutRecentAGNHeatingCoreExcision',
    'T'              : 'SpectroscopicLikeTemperatureWithoutRecentAGNHeatingCoreExcision',
    'YSZ'            : 'Y5R500WithoutRecentAGNHeating',
    'M'              : 'GasMass',
}

COLUMNS = {
    'LX'             : 'LX0InRestframeWithoutRecentAGNHeating',
    'LXCoreExcision' : 'LX0InRestframeWithoutRecentAGNHeatingCoreExcision',
    'T'              : 'ChandraT',
    'YSZ'            : 'Y5R500WithoutRecentAGNHeating',
    'M'              : 'GasMass', 
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


def gamma_to_str(gamma):
    if gamma == 1:
        return ''
    elif gamma == -8/5:
        return '-8/5'
    elif gamma == -2/3:
        return '-2/3'
    else:
        return str(gamma)

labels = {
    'LX-T': {
        'label': '$L_\\mathrm{{X}}-T$',
    },
    'YSZ-T': {
        'label': '$Y_\\mathrm{{SZ}}-T$'
    },
    'M-T': {
        'label': '$M_\\mathrm{{gas}}-T$'
    },
    'LX-YSZ': {
        'label':'$L_\\mathrm{{X}}-Y_\\mathrm{{SZ}}$'  
    },
    'LX-M': {
        'label':'$L_\\mathrm{{X}}-M_\\mathrm{{gas}}$'
        
    },
    'YSZ-M': {
        'label':'Y_\\mathrm{{SZ}}-M_\\mathrm{{gas}}'
        
    },
}


# Global variables for function get_const(). Using of dict have to be outside of the function.
NAMES = list(CONST.keys())
ARRAY_CONST = [list(CONST[relation].values()) for relation in CONST.keys()]
ARRAY_CONST = np.array(ARRAY_CONST, dtype=np.float64)

@njit()
def get_const(relation, key, array_const=ARRAY_CONST):
    """
    A numba compatible function that does not use dictionary directly.
    Works almost the same as CONST dictionary, except the key words are variables.
    Beware, this is about to get ugly.
    """
    if relation == 'LX-T':
        idx = 0
    elif relation == 'YSZ-T':
        idx = 1
    elif relation == 'M-T':
        idx = 2
    elif relation == 'LX-YSZ':
        idx = 3
    elif relation == 'LX-M':
        idx = 4
    elif relation == 'YSZ-M':
        idx = 5
    else:
        raise ValueError(f'Invalid relation: {relation}')
    
    if key == 'CY':
        return array_const[idx,0]
    elif key == 'CX':
        return array_const[idx,1]
    elif key == 'gamma':
        return array_const[idx,2]
    elif key == 'N':
        return array_const[idx,3]
    else:
        raise ValueError(f'Invalid key: {key}')
    

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

# Use check_set_range.ipynb to set this range. Five times the max-min range.
# The scatter follows the MID_RANGE scatter range.
FIVE_MAX_RANGE = { 
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

# Use check_set_range.ipynb to set this range. Five times the max-min range.
# Set the scatter range tighter for bulk flow model. In this case we need very small
# scatter step to see the small difference. Small steps leads to computational cost,
# so we compensate by settting a shorter range. I round down the lower range to 
# 0.005 and minus another 0.015.
ONE_MAX_RANGE_TIGHT_SCAT = { 
    'LX-T': {
        'logA_min': np.log10(1.45), 'logA_max': np.log10(1.60),
        'B_min'   : 2.29,           'B_max'   : 2.62,
        'scat_min': 0.100,          'scat_max': 1,
    },
    'YSZ-T': {
        'logA_min': np.log10(1.01), 'logA_max': np.log10(1.12),
        'B_min'   : 2.62,           'B_max'   : 2.84,
        'scat_min': 0.070,          'scat_max': 1,
    },
    'M-T': {
        'logA_min': np.log10(1.05), 'logA_max': np.log10(1.11),
        'B_min'   : 1.91,           'B_max'   : 2.08,
        'scat_min': 0.060,          'scat_max': 1,
    },
    'LX-YSZ': {
        'logA_min': np.log10(2.87), 'logA_max': np.log10(3.18),
        'B_min'   : 0.81,           'B_max'   : 0.92,
        'scat_min': 0.120,          'scat_max': 1,
    },
    'LX-M': {
        'logA_min': np.log10(1.19), 'logA_max': np.log10(1.27),
        'B_min'   : 1.21,           'B_max'   : 1.28,
        'scat_min': 0.060,          'scat_max': 1,
    },
    'YSZ-M': {
        'logA_min': np.log10(3.63), 'logA_max': np.log10(3.90),
        'B_min'   : 1.30,           'B_max'   : 1.42,
        'scat_min': 0.075,          'scat_max': 1,
    },
}


THREE_MAX_RANGE_TIGHT_SCAT = { 
    'LX-T': {
        'logA_min': np.log10(1.29), 'logA_max': np.log10(1.76),
        'B_min'   : 1.96,           'B_max'   : 2.95,
        'scat_min': 0.100,           'scat_max': 1,
    },
    'YSZ-T': {
        'logA_min': np.log10(0.90), 'logA_max': np.log10(1.23),
        'B_min'   : 2.40,           'B_max'   : 3.06,
        'scat_min': 0.075,          'scat_max': 1,
    },
    'M-T': {
        'logA_min': np.log10(0.99), 'logA_max': np.log10(1.17),
        'B_min'   : 1.73,           'B_max'   : 2.25,
        'scat_min': 0.060,           'scat_max': 1,
    },
    'LX-YSZ': {
        'logA_min': np.log10(2.56), 'logA_max': np.log10(3.50),
        'B_min'   : 0.70,           'B_max'   : 1.03,
        'scat_min': 0.120,           'scat_max': 1,
    },
    'LX-M': {
        'logA_min': np.log10(1.10), 'logA_max': np.log10(1.36),
        'B_min'   : 1.14,           'B_max'   : 1.36,
        'scat_min': 0.060,           'scat_max': 1,
    },
    'YSZ-M': {
        'logA_min': np.log10(3.36), 'logA_max': np.log10(4.17),
        'B_min'   : 1.18,           'B_max'   : 1.54,
        'scat_min': 0.075,           'scat_max': 1,
    },
}


LC0_BEST_FIT = {
    'LX-T': {
        'A': 1.521, 'B': 2.454, 'scat': 0.120, 
    },
    'YSZ-T': {
        'A': 1.064, 'B': 2.732, 'scat': 0.112, 
    },
    'M-T': {
        'A': 1.079, 'B': 1.988, 'scat': 0.070
    },
    'LX-YSZ': {
        'A': 3.027, 'B': 0.862, 'scat': 0.140
    },
    'LX-M': {
        'A': 1.225, 'B': 1.244, 'scat': 0.080
    },
    'YSZ-M': {
        'A': 3.759, 'B': 1.360, 'scat': 0.096
    },
}

LC1_BEST_FIT = {
    'LX-T': {
        'A': 1.542, 'B': 2.446, 'scat': 0.138
    },
    'YSZ-T': {
        'A': 1.099, 'B': 2.782, 'scat': 0.104
    },
    'M-T': {
        'A': 1.094, 'B': 2.012, 'scat': 0.072
    },
    'LX-YSZ': {
        'A': 2.958, 'B': 0.874, 'scat': 0.150
    },
    'LX-M': {
        'A': 1.219, 'B': 1.226, 'scat': 0.092
    },
    'YSZ-M': {
        'A': 3.812, 'B': 1.358, 'scat': 0.104
    },
}
