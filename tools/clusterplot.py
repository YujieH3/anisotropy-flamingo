import numpy as np
import matplotlib.pyplot as plt

plt.style.use("./nice.mplstyle")
from mpl_toolkits.basemap import Basemap


def _small_evs_(distribution, target_value, x_vals, gpd_pdf, threshold, percentile, sigma_evt, color):
    """ Make a small EVS plot with no labels and legends. For appendix.

    Parameters
    --
    distribution :
      one-dimensional distribution of projected values.
    target_value :
      EVS target of evaluation. The point under the same projection that you
      wanted to find the probability about.
    x_vals : 
      The x-values for gpd_pdf plot. Usually an array from np.linspace.
    gpd_pdf :
      Fitted generalised pareto distribution
    threshold :
      The threshold above which the generalised pareto distribution is fitted.
    percentile :
      The percentile above which the generalised pareto distribution is fitted.
      In between 0~100. Usually 90 or 95.
    sigma_evt :
      The evaluated sigma number of EVS analysis. To be annotated on the plot.
    color : 
      The histrogram color.

    Returns
    --
    fig, ax
    """

    # Plotting results
    fig, ax = plt.subplots(figsize=(3, 2.2))
    plt.hist(distribution, bins=20, density=True, alpha=0.6, color=color, label='FLAMINGO', histtype='step')

    plt.axvline(target_value, color='red', linestyle='-', label=f'Migkas+21')
    plt.plot(x_vals[1:], gpd_pdf[1:], color='g', label='Fitted GPD (Tail)')
    plt.axvline(threshold, color='blue', linestyle='--', label=f'Threshold ({percentile}th Percentile)')
    
    # Annotate the plot
    plt.annotate(f'{sigma_evt:.2g}$\\sigma$', 
         xy=(0.8, 0.2), xycoords='figure fraction',
         xytext=(0.8, 0.2), textcoords='figure fraction',
         horizontalalignment='right', verticalalignment='center', color='red')

    # ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.tight_layout() # equivalent to bbox_inches='tight' in savefig

    return fig, ax