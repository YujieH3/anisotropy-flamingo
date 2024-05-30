import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./nice.mplstyle')
from mpl_toolkits.basemap import Basemap


def _sky_map_(f, colorbar_label, ax=None, **kwargs):
    """Plot the sky map of the given function f. This is an internal helper function
    because f only support 4 deg longitude resolution and 2 deg latitude resolution.
    """
    # Convert f to numpy array
    f = np.array(f)

    # Create ax if not specified
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    # Use hammer projection
    m = Basemap(projection='hammer', lon_0=0)

    # Coordinates
    lon = np.linspace(-180, 180, 90)
    lat = np.linspace(-90, 90, 90)
    lons, lats = np.meshgrid(lon, lat)
    # x, y = m(lons, lats)

    # Reshape the input array to 90x90 as supported by imshow. In our case
    # the longitude and latitude are matched this way.
    f = f.reshape(90, 90).T

    # Plot the color map
    im = m.imshow(f,  extent=[-180, 180, -90, 90],
                cmap='inferno', ax=ax, **kwargs)

    # Colorbar
    plt.colorbar(im, label=colorbar_label)
    
    # Axis label
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    return 0