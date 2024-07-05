import emcee
import numpy as np
import os
import sys
sys.path.append('/data1/yujiehe/anisotropy-flamingo')
import tools.constants as const
import tools.clusterfit as cf
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=68.1, Om0=0.306, Ob0=0.0486)

C = 299792.458                  # the speed of light in km/s

# -----------------------CONFIGURATION------------------------------------------
# Input file is a halo catalog with lightcone data.
#INPUT_FILE = '/data1/yujiehe/data/samples_in_lightcone0_with_trees_duplicate_excision_outlier_excision.csv'
#OUTPUT_FILE = '/data1/yujiehe/data/fits/bulk_flow_mcmc_lightcone0.csv'
INPUT_FILE = '/data1/yujiehe/data/samples_in_lightcone1_with_trees_duplicate_excision_outlier_excision.csv'
OUTPUT_FILE = '/data1/yujiehe/data/fits/bulk_flow_mcmc_lightcone1.csv'
OVERWRITE = True

# Relations to fit
RELATIONS = ['LX-T', 'YSZ-T', 'M-T'] # pick from 'LX-T', 'M-T', 'LX-YSZ', 'LX-M', 'YSZ-M', 'YSZ-T'


# -----------------------END CONFIGURATION--------------------------------------






def log_likelihood(theta, X, Y, z_obs, phi_lc, theta_lc, yname, xname):
    """
    X, Y, z_obs, phi_lc, theta_lc, are from the data
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
       return -np.inf 
    
    ubf, vlon, vlat, logA, B, sigma, = theta

    # Set the scaling relation to know the pivot point
    scaling_relation = f'{yname}-{xname}'

    # Calculate the redshift
    angle = cf.angular_separation(phi_lc, theta_lc, vlon, vlat) * np.pi/180
    
    # Correct the bulk flow accordingly: z_bf = z_obs + ubf * (1 + z_bf) * np.cos(angle) / C
    z_bf = (z_obs + ubf * np.cos(angle) / C) / (1 - ubf * np.cos(angle) / C) # Non-relativistic correction

    # The relativistic correction
    # u_c_correct=((1+z_obs)**2-1)/((1+z_obs)**2+1) + (1+z_obs)*ubf*np.cos(angle)/C
    # z_bf=np.sqrt((1+u_c_correct)/(1-u_c_correct))-1

    # Calculate the Luminosity distance
    if yname == 'LX':
        # DL_zobs = cc.DL(z_obs, H0=68.1, Om=0.306, Ol=0.694)
        # DL_zbf = cc.DL(z_bf, H0=68.1, Om=0.306, Ol=0.694)
        DL_zobs = cosmo.luminosity_distance(z_obs).value
        DL_zbf = cosmo.luminosity_distance(z_bf).value
        Y_bf = Y*(DL_zbf)**2/(DL_zobs)**2
    elif yname == 'YSZ':
        DA_zobs = cosmo.angular_diameter_distance(z_obs).value
        DA_zbf = cosmo.angular_diameter_distance(z_bf).value
        Y_bf = Y*(DA_zbf)**2/(DA_zobs)**2
    elif yname == 'M':
        DA_zobs = cosmo.angular_diameter_distance(z_obs).value 
        DA_zbf = cosmo.angular_diameter_distance(z_bf).value 
        Y_bf = Y*(DA_zbf)**(5/2)/(DA_zobs)**(5/2)
        
    # To our fit parameters
    logY_ = cf.logY_(Y_bf, z=z_bf, relation=scaling_relation)
    logX_ = cf.logX_(X, relation=scaling_relation)

    model = B * logX_ + logA
    lnL = -0.5 * np.sum((logY_ - model) ** 2 / (sigma**2) + np.log(sigma**2)) # Kostas' implementation

    return lnL + lp

# set prior
def log_prior(theta):
    # A large flat prior for now
    ubf, vlon, vlat, logA, B, sigma = theta # 6 parameters

    # If in range, p(theta)=1, else p(theta)=0
    if -1<logA<1 and 0.5<B<3.5 and 0.05<sigma<1 \
        and 0<ubf<1000 and -180<vlon<180 and -90<vlat<90:
        return 0.0
    else:
        return -np.inf






# Load data
import pandas as pd
data = pd.read_csv(INPUT_FILE)

# Skip if the file already exists
if os.path.exists(OUTPUT_FILE) and not OVERWRITE:
    print(f'File exists: {OUTPUT_FILE}')
    raise Exception('Output file exists and OVERWRITE==False.')

first_entry = True



for scaling_relation in RELATIONS: 
    n_clusters = cf.CONST[scaling_relation]['N']

    # Load the data
    _ = scaling_relation.find('-')
    yname = scaling_relation[:_]
    xname = scaling_relation[_+1:]
    Y = np.array(data[cf.COLUMNS[yname]][:n_clusters])
    X = np.array(data[cf.COLUMNS[xname]][:n_clusters])
    # Also load the position data
    phi_lc   = np.array(data['phi_on_lc'][:n_clusters])
    theta_lc = np.array(data['theta_on_lc'][:n_clusters])
    # the observed redshift from lightcone
    z_obs = np.array(data['ObservedRedshift'][:n_clusters])


    # Load data and set zmax
    for zmax in np.arange(0.06, 0.13, 0.02):
        zmask    = z_obs < zmax

        # Select data below some redshift
        z_obs    = z_obs[zmask]
        X        = X[zmask]
        Y        = Y[zmask]
        phi_lc   = phi_lc[zmask]
        theta_lc = theta_lc[zmask]

        print(np.sum(zmask)) # Number of clusters left



        # Optimize
        from scipy.optimize import minimize
        from scipy.optimize import differential_evolution
        nll = lambda *args: -log_likelihood(*args)
        initial = np.array([0, 0, 0, 1, 1, 0.1]) # initial guess
        bounds = [(0, 1000), (-180, 180), (-90, 90), (0.1, 1), (0.5, 3.5), (0.05, 1)]

        soln = differential_evolution(nll, args=(X, Y, z_obs, phi_lc, theta_lc, yname, xname), 
                        bounds=bounds, popsize=10, strategy='rand1bin')
        print(soln.x)


        pos0 = soln.x + 1e-2 * np.random.randn(32, 6)
        nwalkers, ndim = pos0.shape

        # Create a sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, 
                                        args=(X, Y, z_obs, phi_lc, theta_lc, yname, xname))

        # Run
        sampler.run_mcmc(pos0, 15000, progress=True)

        # Small convergence test
        try:
            tau = sampler.get_autocorr_time()
            print(tau)
        except emcee.autocorr.AutocorrError:
            print('The chain is too short to get a reliable autocorrelation time.')
            tau = 0




        # Get the samples
        flat_samples = sampler.get_chain(discard=1000, thin=80, flat=True)
        print(flat_samples.shape)


        # Plot and save
        import corner
        import matplotlib.pyplot as plt

        fig = corner.corner(
            flat_samples, 
            labels=['ubf', 'vlon', 'vlat', 'logA', 'B', 'sigma'],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True, 
            title_fmt='.3f',
            title_kwargs={"fontsize": 15},
            label_kwargs={"fontsize": 17},
            smooth=1,
            levels=[0.39],
        )

        # Title
        fig.suptitle(f'{yname}-{xname} with z<{zmax}', fontsize=20)

        # Save plots
        if not os.path.exists('../data/plots/7bulk-flow-model-mcmc'):
            os.makedirs('../data/plots/7bulk-flow-model-mcmc')
        fig.savefig(f'../data/plots/7bulk-flow-model-mcmc/{yname}-{xname}-zmax{zmax}.png')



        # For ubf we use the 16, 50, 84 quantiles
        ubf_distr = flat_samples[:, 0]
        lower_ubf  = np.percentile(ubf_distr, 16)
        median_ubf = np.percentile(ubf_distr, 50)
        upper_ubf  = np.percentile(ubf_distr, 84)
        
        # For saving
        ubf = median_ubf
        ubf_err_lower = median_ubf - lower_ubf
        ubf_err_upper = upper_ubf - median_ubf
        print(f'ubf: {lower_ubf} ~ {upper_ubf} \nor {ubf} -{ubf_err_lower} +{ubf_err_upper}')



        # Same with latitude, note that latitude is not periodic!
        vlat_distr = flat_samples[:, 2]
        lower_vlat = np.percentile(vlat_distr, 16)
        median_vlat = np.percentile(vlat_distr, 50)
        upper_vlat = np.percentile(vlat_distr, 84)

        # For saving
        vlat = median_vlat
        vlat_err_lower = median_vlat - lower_vlat
        vlat_err_upper = upper_vlat - median_vlat
        print(f'vlat: {lower_vlat} ~ {upper_vlat} \nor {vlat} -{vlat_err_lower} +{vlat_err_upper}')


        # For longitude we use the 34th percentile around the peak value
        # Longitude is periodic so we are free to shift the peak value for convenience
        vlon_distr = flat_samples[:, 1]
        hist, edges = np.histogram(vlon_distr, bins=20, density=True)
        peak_vlon = edges[np.argmax(hist)]

        # Shift to peak=0 to avoid breaking near the edge
        vlon_distr = (vlon_distr - peak_vlon - 180) % 360 - 180 # Despite the shift, keep the range in 180 to 180

        # 34th percentile around the peak value
        peak_percentile = np.sum(vlon_distr < 0) / len(vlon_distr) * 100
        l = np.percentile(vlon_distr, peak_percentile - 34)
        u = np.percentile(vlon_distr, peak_percentile + 34)

        # Convert back to the original coordinates
        lower_vlon = (l + peak_vlon + 180) % 360 - 180
        upper_vlon = (u + peak_vlon + 180) % 360 - 180

        # For saving
        vlon = peak_vlon
        vlon_err_lower = -l
        vlon_err_upper = u
        print(f'vlon: {lower_vlon} ~ {upper_vlon} \nor {peak_vlon} -{vlon_err_lower} +{vlon_err_upper}')

        

        # Save the best fit parameters
        if first_entry:
            mode = 'w'
        else:
            mode = 'a'
        
        # Write line by line
        with open(OUTPUT_FILE, mode) as f:

            # Write the header on first entry
            if first_entry:
                f.write('scaling_relation,zmax,ubf,ubf_err_lower,ubf_err_upper,vlon,vlon_err_lower,vlon_err_upper,vlat,vlat_err_lower,vlat_err_upper,convergence_time\n')
                first_entry = False

            # Write the data
            f.write(f'{scaling_relation},{zmax},{ubf},{ubf_err_lower},{ubf_err_upper},{vlon},{vlon_err_lower},{vlon_err_upper},{vlat},{vlat_err_lower},{vlat_err_upper},{np.mean(tau)}\n')
