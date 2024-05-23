import emcee
import numpy as np
import sys
sys.path.append('/data1/yujiehe/anisotropy-flamingo')
import tools.constants as const
import tools.clusterfit as cf


C = 299792.458                  # the speed of light in km/s



def log_likelihood(theta, X, Y, z_obs, phi_lc, theta_lc, yname, xname):
    """
    X, Y, z_obs, phi_lc, theta_lc, are from the data
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf 
    
    A, B, sigma, ubf, vlon, vlat = theta

    # Set the scaling relation to know the pivot point
    scaling_relation = f'{yname}-{xname}'

    # Calculate the redshift
    angle = cf.angular_separation(phi_lc, theta_lc, vlon, vlat)
    
    # Correct the bulk flow accordingly: z_bf = z_obs + ubf * (1 + z_bf) * np.cos(angle) / C
    z_bf = (z_obs + ubf * np.cos(angle) / C) / (1 - ubf * np.cos(angle) / C) # 

    # To our fit parameters
    logY_ = cf.logY_(Y, z=z_bf, relation=scaling_relation)
    logX_ = cf.logX_(X, relation=scaling_relation)

    model = A * logX_ + B
    lnL = -0.5 * np.sum((logY_ - model) ** 2 / sigma**2 + np.log(2*np.pi*sigma**2))

    return lp + lnL

# set prior
def log_prior(theta):
    # A large flat prior for now
    logA, B, sigma, ubf, vlon, vlat = theta # 6 parameters

    # If in range, p(theta)=1, else p(theta)=0
    if 0.1<logA<1 and 0.5<B<3.5 and 0.05<sigma<1 \
        and 0<ubf<1000 and -180<vlon<180 and -90<vlat<90:
        return 0.0
    else:
        return -np.inf


# build io



# run tonight