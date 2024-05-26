import numpy as np
from numba import njit, prange
import pandas as pd
import sys
sys.path.append('/home/yujiehe/anisotropy-flamingo')
from tools.constants import *

@njit(fastmath=True)
def E(z, Omega_m=0.306, Omega_L=0.694):
    Ez = (Omega_m * (1 + z)**3 + Omega_L)**0.5
    return Ez

@njit(fastmath=True)
def _logX_(X, CX):
    """ logX' = X / CX """
    result = np.log10(X / CX)
    return result

@njit(fastmath=True)
def _logY_(Y, z, CY, gamma, Omega_m=0.306, Omega_L=0.694):
    """ logY' = Y / CY * E(z)^gamma """
    Ez = E(z=z, Omega_m=Omega_m, Omega_L=Omega_L)
    result = np.log10(Y / CY * Ez**gamma)
    return result

@njit(fastmath=True)
def logX_(X, relation):
    """ Same as _logX_ but with predifined constants for specific scaling 
    relations. 
    
    Parameters
    ---
    `relation`
        Accepts one of three options: 'LX-T', 'LX-YSZ', 'YSZ-T'
    Specify relation to use default parameters.
    """
    return _logX_(X=X, CX=get_const(relation, 'CX'))


@njit(fastmath=True)
def logY_(Y, z, relation, Omega_m=0.306, Omega_L=0.694):
    """ Same as _logY_ but with predefined constants for specific scaling 
    relations. 
    
    Parameters
    ---
    `relation`
        Accepts one of six options: 'LX-T', 'LX-YSZ', 'YSZ-T', 'M-T', 'LX-M',
        'YSZ-M'. Specify relation to use default parameters defined in global 
        constant CONST.
    """
    return _logY_(Y        = Y, 
                  z        = z,
                  CY       = get_const(relation, 'CY'),
                  gamma    = get_const(relation, 'gamma'),
                  Omega_m  = Omega_m,
                  Omega_L  = Omega_L
            )


def predictY_(X_, **params):
    """ Predict Y' given X' and best fit parameters `logA` and `B`. """
    return 10**(params['logA'] + params['B'] * np.log10(X_))


def predictlogY_(logX_, **params):
    """ Predict logY' given logX' and best fit parameters `logA` and `B`. """
    return params['logA'] + params['B'] * logX_


def fit(logY_, logX_, N,
        B_min, B_max, 
        logA_min, logA_max, 
        scat_min, scat_max,
        scat_step=0.007, B_step=0.001, logA_step=0.003,
        remove_outlier=False, id=None):
    """
    Fit the scaling relation logY' = logA + B * logX' with intrinsic scatter.

    Parameters
    ---
    `logY_`
        Logarithm of the dependent variable. Length must be larger than N and
        large enough to have N non-outlier data points.
    `logX_`
        Logarithm of the independent variable. Length must be equal to `logY_`.
    `N`
        Number of clusters that goes into the fit. The first N non-outliers are 
        used for fitting.
    `remove_outlier`
        If True, iteratively remove outliers and refit with fixed number of
        clusters.
    `id`
        Halo id of the clusters. If provided, outliers will be tracked and
        returned separately.

    Returns
    ---
    `params`
        Dictionary containing the best fit parameters. It has the following
        keys: 'logA', 'B', 'scat', 'chi2'.
    `outlier_id`
        Halo id of the outliers. Only returned if `id` is provided and `remove_outlier` is True.


    Notes
    ---
    For reference, the best fit value in observation is

    | Relation | A     | logA  | B     | intr_scat |
    |----------|-------|-------|-------|-----------|
    | LX-T     | 1.132 | 0.054 | 2.086 | 0.233     |
    | LX-YSZ   | 2.737 | 0.437 | 0.928 | 0.108     |
    | YSZ-T    | 1.110 | 0.045 | 2.546 | 2.546     |

    """
    params = run_fit(logY_=logY_[:N], logX_=logX_[:N],
                    B_min=B_min, B_max=B_max, 
                    logA_min=logA_min, logA_max=logA_max, 
                    scat_min=scat_min, scat_max=scat_max,
                    scat_step=scat_step, B_step=B_step, logA_step=logA_step)
    print('Best fit found: ', params)
        
    # Iteratively remove outliers and refit with fixed number of clusters
    if remove_outlier:
        itr_count     = 0
        outlier_count = 0
        if id is not None:
            outlier_id = np.array([])

        while True:
            itr_count += 1

            # ------------FIND AND REMOVE OUTLIERS----------------
            outlier = find_outlier(logY_=logY_[:N], logX_=logX_[:N], 
                        best_fit_params=params) # find outliers as a boolean array. But search only sample clusters, the first N that goes into fitting.

            outlier_found = np.sum(outlier) # track number of outliers
            print(f'Outliers found: {outlier_found}')

            if outlier_found == 0: # if no outlier found in this iteration, then last fit is the final best fit.
                break

            outlier_count += outlier_found # track number of outliers

            logY_ = np.concatenate((logY_[:N][~outlier], logY_[N:]))  # select non-outliers for next fit
            logX_ = np.concatenate((logX_[:N][~outlier], logX_[N:]))  #__~ is the logical not operator__

            if id is not None: # if id is provided, track halo id of outliers
                outlier_id = np.append(outlier_id, id[:N][outlier]) # track halo id of outliers
                print(f'Outlier ids: {outlier_id}')
                id = np.concatenate((id[:N][~outlier], id[N:]))     # sync id with logY_ and logX_
            # ------------END FIND AND REMOVE OUTLIERS----------------

            if len(logY_) > 0:
                print(f'Fit: {itr_count + 1}')
                print(f'Iteration: {itr_count}')
                params = run_fit(logY_=logY_[:N], logX_=logX_[:N], 
                            B_min=B_min, B_max=B_max, 
                            logA_min=logA_min, logA_max=logA_max, 
                            scat_min=scat_min, scat_max=scat_max,
                            scat_step=scat_step, B_step=B_step, logA_step=logA_step)
                print('Best fit found: ', params)
            else:
                raise ValueError('All data points are outliers. No fit can be made.')

            break # one iteration. limitation too strong otherwise.
        
    # __Numba doesn't support dictionaries with non-scalar values__ 
    # __So we return outlier_id separately__
    if remove_outlier is True and id is not None:
        return params, outlier_id
    else:
        return params, np.array([])

@njit(fastmath=True)
def run_fit(logY_, logX_, B_min, B_max, logA_min, logA_max, scat_min, 
            scat_max, scat_step, B_step, logA_step, weight=np.array([1.])):

    """ Numba accelerated function to iterate through the parameter space to
    find the best fits."""
    
    Nclusters = len(logY_)
    minx2 = 1e8

    if (weight == np.array([1])).all():
        weight = np.ones(Nclusters)
    for scat in np.arange(scat_min, scat_max, scat_step):
        for B in np.arange(B_min, B_max, B_step):
            for logA in np.arange(logA_min, logA_max, logA_step):
                x2 = np.sum((logY_ - logA - B * logX_)**2 / (scat * weight)**2) 
                x2 /= (Nclusters - 3)     # chi_res^2 = chi^2 / (N - dof), degree of freedom is the number of parameters to fit
                if x2 < minx2:      # update best fit if new lowest chi2 found
                    minx2 = x2
                    params = {
                        'logA' : logA,
                        'B'    : B,
                        'scat' : scat,
                        'chi2' : minx2
                    }

        if minx2 < 1.04: # end after iterating through A and B space
            break
    if minx2 < 1.04:
        return params
    else:
        return {         # No fit is found if chi2 >= 1.04 for all parameters
            'logA' : np.nan,
            'B'    : np.nan,
            'scat' : np.nan,
            'chi2' : np.nan
        }      


def find_outlier(logY_, logX_, best_fit_params, outlier_sigma=4):
    """ Find outliers based on the best fit parameters. Return a boolean outlier.

    Parameters
    ---
    `logY_`
        Logarithm of the dependent variable.
    `logX_`
        Logarithm of the independent variable.
    `params`
        Dictionary containing the best fit parameters. 
    `outlier_sigma`
        Number of standard deviations from the best fit line to remove. 

    Returns
    ---
    `outlier`
        Boolean outlier of the same length as `logY_` and `logX_`. True means
        outlier.
    """
    # logY_pred = predictlogY_(logX_, **best_fit_params)
    residual = logY_ - best_fit_params['logA'] - best_fit_params['B'] * logX_
    # sigma = np.std(residual)
    sigma = best_fit_params['scat']
    outlier = np.abs(residual) > outlier_sigma * sigma
    return outlier


@njit(fastmath=True, parallel=True)
def bootstrap_fit(Nbootstrap, 
                  logY_, logX_, Nclusters,
                  B_min, B_max, 
                  logA_min, logA_max, 
                  scat_min, scat_max, weight = None,
                  scat_step=0.007, B_step=0.001, logA_step=0.003):

    """
    Examples
    ---
    >>> logA, B, scat = bootstrap_fit(Nbootstrap=1000, ... )

    then the bootstrapping uncertainty can be given by `np.quantile`
    e.g. 1-sigma uncertainty is given by `np.quantile(logA, [0.16, 0.84])`.
    """

    logA = np.zeros(Nbootstrap)  # logA distribution
    B    = np.zeros(Nbootstrap)  # B distribution
    scat = np.zeros(Nbootstrap)  # scatter distribution

    if len(logY_) != len(logX_):
        raise ValueError('Length of logY_ and logX_ must be equal.')

    for i in prange(Nbootstrap):
        idx = np.random.choice(Nclusters, size=Nclusters, replace=True)
        bootstrap_logY_ = logY_[idx]
        bootstrap_logX_ = logX_[idx]

        if weight is None:
            bootstrap_weight = np.array([1.]) # Setting to int 1 will invoke numba typing error, so we do this
        else:
            bootstrap_weight = weight[idx]

        params = run_fit(
            logY_          = bootstrap_logY_,
            logX_          = bootstrap_logX_,
            B_min          = B_min,     B_max    = B_max,
            logA_min       = logA_min,  logA_max = logA_max,
            scat_min       = scat_min,  scat_max = scat_max,
            scat_step      = scat_step,
            B_step         = B_step,
            logA_step      = logA_step,
            weight         = bootstrap_weight
            )
        
        logA[i] = params['logA']
        B[i]    = params['B']
        scat[i] = params['scat']

    return logA, B, scat


@njit(fastmath=True, parallel=False)
def bootstrap_fit_non_parallel(Nbootstrap, 
                  logY_, logX_, Nclusters,
                  B_min, B_max, 
                  logA_min, logA_max, 
                  scat_min, scat_max, weight = None,
                  scat_step=0.007, B_step=0.001, logA_step=0.003):

    """
    The non-parallel counter-part of the `bootstrap_fit` function. Only thing
    different is the `parallel=False` argument in the @njit decorator.

    Examples
    ---
    >>> logA, B, scat = bootstrap_fit(Nbootstrap=1000, ... )

    then the bootstrapping uncertainty can be given by `np.quantile`
    e.g. 1-sigma uncertainty is given by `np.quantile(logA, [0.16, 0.84])`.
    """

    logA = np.zeros(Nbootstrap)  # logA distribution
    B    = np.zeros(Nbootstrap)  # B distribution
    scat = np.zeros(Nbootstrap)  # scatter distribution

    if len(logY_) != Nclusters or len(logX_) != Nclusters:
        raise ValueError('Length of logY_ and logX_ must be equal to Nclusters.')

    for i in prange(Nbootstrap):
        idx = np.random.choice(Nclusters, size=Nclusters, replace=True)
        bootstrap_logY_  = logY_[idx]
        bootstrap_logX_  = logX_[idx]

        if weight is None:
            bootstrap_weight = np.ones(len(idx)) # Setting to int 1 will invoke numba typing error, so we do this
        else:
            bootstrap_weight = weight[idx]

        params = run_fit(
            logY_          = bootstrap_logY_,
            logX_          = bootstrap_logX_,
            B_min          = B_min,     B_max    = B_max,
            logA_min       = logA_min,  logA_max = logA_max,
            scat_min       = scat_min,  scat_max = scat_max,
            scat_step      = scat_step,
            B_step         = B_step,
            logA_step      = logA_step,
            weight         = bootstrap_weight
            )
        
        logA[i] = params['logA']
        B[i]    = params['B']
        scat[i] = params['scat']

    return logA, B, scat




def significance_map(best_fit_file, btstrp_file):
    """ Calculate the significance map of the dipole anisotropy. """
    pd.options.mode.copy_on_write = True

    best_fit = pd.read_csv(best_fit_file) # best fit value # ft stores the best fit values
    btstrp_fits = pd.read_csv(btstrp_file) # bts is the bootstrapping results
    df = best_fit[['Glon', 'Glat']].copy()  # df stores the dipole anisotropy significance map
    df['n_sigma'] = 0.0
    df['sigma'] = 0.0

    for i in range(len(df)):
        glon = df['Glon'][i]
        glat = df['Glat'][i]
        if glat < 0: # only need to do half of the directions # I also skipped glat=90, why?
            continue
        dp_glon = glon + 180 if glon < 0 else glon - 180 # zero to - 180
        dp_glat = - glat
        # print(dp_glon, dp_glat)

        A1 = best_fit.loc[i, 'A'] # query directly by index to save computation, this gives a np.float64 number directly so no need for conversion
        A2 = best_fit.loc[(df['Glon'] == dp_glon) & (df['Glat'] == dp_glat), 'A']
        A2 = float(A2.iloc[0]) # convert pandas series to float. Pandas suggested this over float(A2)

        btstrp_A1 = btstrp_fits.loc[(btstrp_fits['Glon'] == glon) & (btstrp_fits['Glat'] == glat)].A # pandas series
        upper_A1 = np.percentile(btstrp_A1, 84)
        lower_A1 = np.percentile(btstrp_A1, 16)
        sigma_A1 = A1 - lower_A1 if A1 > A2 else upper_A1 - A1 # use the uncertrainty 'along the direction of the dipole'. i.e. the larger one takes sigma towards the lower bounds and vice versa.

        btstrp_A2 = btstrp_fits.loc[(btstrp_fits['Glon'] == dp_glon) & (btstrp_fits['Glat'] == dp_glat)].A
        upper_A2 = np.percentile(btstrp_A2, 84)
        lower_A2 = np.percentile(btstrp_A2, 16)
        sigma_A2 = upper_A2 - A2 if A1 > A2 else A2 - lower_A2

        sigma = np.sqrt(sigma_A1**2 + sigma_A2**2)
        n_sigma = (A1 - A2) / sigma

        df.loc[i, 'n_sigma'] = n_sigma
        df.loc[(df['Glon'] == dp_glon) & (df['Glat'] == dp_glat), 'n_sigma'] = - n_sigma
        
        df.loc[i, 'sigma'] = sigma
        df.loc[(df['Glon'] == dp_glon) & (df['Glat'] == dp_glat), 'sigma'] = sigma

    return df


def A_variance_map(best_fit_file, btstrp_file):
    """
    Calculate the variance of A's. We calculate both the standard deviation and
    the upper and lower 1-sigma deviation obtained by 50-16 and 84-50 percentile.
    """

    best_fit = pd.read_csv(best_fit_file) # best fit value # ft stores the best fit values
    btstrp_fits = pd.read_csv(btstrp_file) # bts is the bootstrapping results
    df = best_fit[['Glon', 'Glat']].copy()  # df stores the dipole anisotropy significance map
    
    df['A_std']   = 0.0
    df['A_upper'] = 0.0
    df['A_lower'] = 0.0

    for i in range(len(df)):
        glon = df['Glon'][i]
        glat = df['Glat'][i]
        # print(dp_glon, dp_glat)

        A = best_fit.loc[i, 'A'] # query directly by index to save computation, this gives a np.float64 number directly so no need for conversion

        btstrp_A = btstrp_fits.loc[(btstrp_fits['Glon'] == glon) & (btstrp_fits['Glat'] == glat), 'A'] # pandas series
        
        std_A = np.std(btstrp_A)
        median_A = np.percentile(btstrp_A, 50)

        df.loc[i, 'A_std']   = std_A
        df.loc[i, 'A_lower'] = median_A - np.percentile(btstrp_A, 16)
        df.loc[i, 'A_upper'] = np.percentile(btstrp_A, 84) - median_A

    return df


@njit(fastmath=True)
def angular_separation(lon1, lat1, lon2, lat2):
    """
    Calculate the angular separation between two points on the sky.

    Parameters
    ----------
    lon1 : float
        Longitude of the first point in degrees.
    lat1 : float
        Latitude of the first point in degrees.
    lon2 : float
        Longitude of the second point in degrees.
    lat2 : float
        Latitude of the second point in degrees.

    Returns
    -------
    separation : float
        Angular separation in degrees, from 0 to 180.
    """
    # Convert to radians
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    # Apply formula
    separation = np.arccos(np.sin(lat1) * np.sin(lat2) +
                           np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2))

    # Convert back to degrees
    separation = np.degrees(separation)

    return separation


@njit(fastmath=True)
def opposite_direction(lon, lat):
    """ Calculate the opposite direction of a given longitude and latitude. """
    lon = lon + 180 if lon < 0 else lon - 180
    lat = - lat
    return lon, lat


def _map_to_dipole_map_(f, mid):
    """Convert a map to a dipole map by force symmetry between one point and its
    opposite."""

    f = np.array(f)
    assert len(f)==8100, "Only support 4 deg longitude and 2 deg latitude resolution. Cause I'm lazy."
    
    lons = np.arange(-180, 180, 4)
    lats = np.arange(-90, 90, 2)
    coord = np.meshgrid(lons, lats, indexing='ij')
    coord = np.array(coord)
    coord = np.reshape(coord, (2, 8100))

    for i in range(8100):
        lon, lat = coord[:, i]
        if lat < 0: # do only positive directions
            continue

        # Find the direction
        dp1 = f[i]
        
        # Find the opposite direction
        lon_dp, lat_dp = opposite_direction(lon, lat)
        i_dp = np.where((coord[0] == lon_dp) & (coord[1] == lat_dp))[0][0]
        dp2 = f[i_dp]

        # Symmetrize around mid while keep there difference f[i]-f[i_dp] = dp1 - dp2
        f[i] = (dp1 - dp2)/2 + mid
        f[i_dp] = (dp2 - dp1)/2 + mid

    return f
        
