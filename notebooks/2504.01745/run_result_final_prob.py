# %% [markdown]
# # Probability Calculation
# Calculate the probability by summing over the kde above the point.
# 
# H0 bootstrapping: Lx-T 8.7, 2.8sigma; Ysz-T 14.0, 4.1sigma
# 
# H0 mcmc: Lx-T 8.7, 2.4sigma; Ysz-T 14.0, 2.6sigma

# %%
# Read the data
import pandas as pd
data = pd.read_csv('/cosma/home/do012/dc-he4/anisotropy-flamingo/data/analysis_all/h0_direct_compare.csv')
data['Delta_H0'] *= 100    # Also brings delta_h0 to the order of magnitude of significance


# %%
data_joint = pd.merge(left=data[data['Relations']=='$L_\\mathrm{{X}}-T$'], 
                      right=data[data['Relations']=='$Y_\\mathrm{{SZ}}-T$'],
                      on='Lightcone', how='inner')
data_joint

# %%
# Calculate the angular correlation
import sys
sys.path.append('/cosma/home/do012/dc-he4/anisotropy-flamingo/tools')
import clusterfit as cf

# coordinates
xlon = data_joint['Glon_x'].values
xlat = data_joint['Glat_x'].values
ylon = data_joint['Glon_y'].values
ylat = data_joint['Glon_y'].values

# calculate angular separation
theta = cf.angular_separation(xlon, xlat, ylon, ylat)

# %% [markdown]
# We make a five variate kde at M21 point. theta=9.125

# %%
cf.angular_separation(274, -9, 268, -16)

# %%
# Add angular separation to the data set
data_joint['Theta'] = theta

# %%
# kde estimation
import scipy.stats as stats
import numpy as np

# Gaussian kde estimation
dataset = data_joint[['Delta_H0_x', 'Delta_H0_y', 'Significance_x', 'Significance_y', 'Theta']]
kde = stats.gaussian_kde(dataset.T.values)

# # Create grid points (5D so cannot be too dense)
# a_grid = np.linspace(0, 30, 10)
# b_grid = np.linspace(0, 30, 10)
# c_grid = np.linspace(0, 10, 10)
# d_grid = np.linspace(0, 10, 10)
# e_grid = np.linspace(0, 50, 10)
# A, B, C, D, E = np.meshgrid(a_grid, b_grid, c_grid, d_grid, e_grid, indexing='ij')
# positions = np.vstack([A.ravel(), B.ravel(), C.ravel(), D.ravel(), E.ravel()])
# Z = kde(positions).reshape(A.shape)
# print(Z.shape)

# # %%
# # Sum up the probability mass
# point = np.array([8.7, 14, 2.8, 4.1, 9.125])
# dx = (a_grid[1]-a_grid[0])\
#     * (b_grid[1] - b_grid[0])\
#     * (c_grid[1] - c_grid[0])\
#     * (d_grid[1] - d_grid[0])\
#     * (e_grid[1] - e_grid[0])
# prob = kde(point) * dx
# prob_mass = Z * dx
# prob_above = np.sum(prob_mass[prob_mass > prob])
# print(np.sum(prob_mass))
# print(prob_above)
# print('p-value', 1-prob_above)
# # print(np.sqrt(-2*np.log(1-prob_above)))




# Monte Carlo integration
num_samples = 1000000  # Increase this number for better accuracy
samples = kde.resample(num_samples)
values = kde(samples)

# Approximate the integral
integral_approximation = np.mean(values)
print("Integral approximation:", integral_approximation)

# Sum up the probability mass
point = np.array([8.7, 14, 2.8, 4.1, 9.125])
prob = kde(point)
# prob_mass = values * (30 * 30 * 10 * 10 * 50) / num_samples  # Volume of the grid space divided by number of samples
# prob_above = np.sum(prob_mass[prob_mass > prob])
# print("Sum of probability mass:", np.sum(prob_mass))
# print("Probability mass above the point:", prob_above)
# print('p-value:', 1 - prob_above)

# Directly get the probability mass below
prob_below = np.sum(values < prob)
print('p-value:', prob_below)
prob_above = np.sum(values > prob)
print(f'Probability above: {prob_above}')