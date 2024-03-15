# Anisotropy Flamingo
The project study the anisotropy of the universe, especially the Hubble expansion and bulk flow. 

## Data description

The sample data includes the following columns:

| Name | Description | Unit |
| ---- | ----------- | ---- |
| lc_id | Cluster ID of the lightcone, for verification purpose only | - |
| redshift | Cosmological redshift of the cluster (without redshift caused by peculiar velocity) | - |
| theta_on_lc | Latitude of the cluster. There is no galactic plane so the origin point is irrelevent. (I don't know where it is) | degree |
| phi_on_lc | Longitude of the cluster. | degree |
| M_fof_lc | FOF mass of the cluster in the lightcone, for verification purpose only | M_sun |
| x_lc | x comoving coordinate of the cluster in the lightcone | Mpc |
| y_lc | y comoving coordinate of the cluster in the lightcone | Mpc |
| z_lc | z comoving coordinate of the cluster in the lightcone | Mpc |
| snap_num | Snapshot number of the cluster in the lightcone | - |
| MfofSOAP | FOF mass of the cluster in the SOAP catalogue, for verification purpose only | M_sun |
| SOAPID | Cluster ID of the SOAP catalogue, for verification purpose only | - |
| M500 | M500 mass of the cluster | M_sun |
| GasMass | Gas mass of the cluster | M_sun |
| LX0InRestframeWithoutRecentAGNHeating | XRayLuminosityInRestframeWithoutRecentAGNHeating from SOAP. Wavelength 0.2-2.3 keV | erg/s |
| LX0InRestframeWithoutRecentAGNHeatingCoreExcision | XRayLuminosityInRestframeWithoutRecentAGNHeatingCoreExcision from SOAP. Wavelength 0.2-2.3 keV | erg/s |
| GasTemperatureWithoutRecentAGNHeatingCoreExcision | GasTemperatureWithoutRecentAGNHeatingCoreExcision from SOAP | K |
| SpectroscopicLikeTemperatureWithoutRecentAGNHeatingCoreExcision | SpectroscopicLikeTemperatureWithoutRecentAGNHeatingCoreExcision from SOAP | K |
| Y5R500WithoutRecentAGNHeating | ComptonYWithoutRecentAGNHeating from SOAP | cm^2 |
| Vx | x component of the peculiar velocity of the cluster | km/s |
| Vy | y component of the peculiar velocity of the cluster | km/s |
| Vz | z component of the peculiar velocity of the cluster | km/s |

