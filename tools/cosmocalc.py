import numpy as np
from numba import njit

@njit(fastmath=True)
def cosmo_integrate(f, x1, x2, N, H0, Om, Ol):
    """
    Integration function, specifically for cosmology. Because numba does not support
    dinamical **kwargs, we need to pass the parameters as arguments directly.
    """
    dx = (x2 - x1) / N
    # vectorized function to calculate for an array of x2
    result = np.empty(len(x2))
    for i in range(len(x2)):
        result[i] = np.sum(f(np.linspace(x1, x2[i], N), H0=H0, Om=Om, Ol=Ol) * dx[i])

    return result

@njit(fastmath=True)
def DL(z, H0, Om, Ol):
    """
    Luminosity distance in Mpc.

    Parameters
    ----------
    z : np.ndarray
        A numpy array of redshifts.
    H0 : float
        The present day Hubble parameter in km/s/Mpc.
    Om : float
        The present day matter density parameter Omega_m,0.
    Ol : float
        The present day dark energy density parameter Omega_lambda,0.

    Returns
    -------
    np.ndarray
        A numpy array of luminosity distances in Mpc, length of output matches
        the length of the input redshifts.
    """
    def f(z, H0, Om, Ol):
        return 1 / np.sqrt(Om * (1 + z)**3 + Ol) * 299792.458 / H0
    return cosmo_integrate(f, x1=0, x2=z, N=10**5, H0=H0, Om=Om, Ol=Ol) * (1 + z)

@njit(fastmath=True)
def DA(z, H0, Om, Ol):
    """
    Angular diameter distance in Mpc.

    Parameters
    ----------
    z : np.ndarray
        A numpy array of redshifts.
    H0 : float
        The present day Hubble parameter in km/s/Mpc.
    Om : float
        The present day matter density parameter Omega_m,0.
    Ol : float
        The present day dark energy density parameter Omega_lambda,0.

    Returns
    -------
    np.ndarray
        A numpy array of angular diameter distances in Mpc, length of output matches
        the length of the input redshifts.
    """
    def f(z, H0, Om, Ol):
        return 1 / np.sqrt(Om * (1 + z)**3 + Ol) * 299792.458 / H0
    return cosmo_integrate(f, x1=0, x2=z, N=10**5, H0=H0, Om=Om, Ol=Ol) / (1 + z)
    