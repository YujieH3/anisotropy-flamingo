# Note that the direction of the variation is neglected. The opposite
# direction also has positive variation and significance.

import os
import pandas as pd
import numpy as np
import warnings

# a specific function for the file
import numpy as np
import pandas as pd
import sys
sys.path.append('/cosma/home/do012/dc-he4/anisotropy-flamingo/tools')
import clusterfit as cf
def get_h0_var(scan_bootstrap_file : str, 
               scan_best_fit_file : str, 
               fit_all_file : str, 
               relation : str) -> tuple:
    """
    Given the bootstrap and best fit file, calculate the significance and the 
    H0 variation amplitude. Both files are needed for the both the significance
    and the amplitude so it needs to be in one function.
    """
    if relation == 'LX-T':
        h0p = 0.5
    elif relation == 'YSZ-T':
        h0p = 0.5
    elif relation == 'M-T':
        h0p = 2/5
    else:
        raise ValueError('relation not recognised')

    df_fitall = pd.read_csv(fit_all_file)
    A_all = df_fitall.loc[df_fitall['Relation']==relation, 'BestFitA'].values[0]

    df_bestfit = pd.read_csv(scan_best_fit_file)
    Glon_scan = df_bestfit['Glon'].values
    Glat_scan = df_bestfit['Glat'].values
    A_scan = df_bestfit['A'].values

    df_btstrp = pd.read_csv(scan_bootstrap_file)
    Glon_btstrp = df_btstrp['Glon'].values
    Glat_btstrp = df_btstrp['Glat'].values
    A_btstrp = df_btstrp['A'].values

    # A unique set of directions for iteration
    unique_Glon, unique_Glat = df_btstrp[['Glon', 'Glat']].drop_duplicates().values.T
    n_sigmas = np.zeros_like(unique_Glon)
    variation = np.zeros_like(unique_Glon) # in percentage
    for glon, glat in zip(unique_Glon, unique_Glat):
        #print(glon, glat)
        # bootstrapping distribution of A of the direction
        dir_mask = (cf.periodic_distance(Glon_btstrp, glon) < 1) & (np.abs(Glat_btstrp - glat) < 1) # within 1 degree
        A = A_btstrp[dir_mask]

        # get A of the opposite direction
        dplon, dplat = cf.opposite_direction(glon, glat)
        #print(dplon, dplat)
        dpdir_mask = (cf.periodic_distance(Glon_btstrp, dplon) < 3) & (np.abs(Glat_btstrp - dplat) < 3) 
        
        # loosen the criteria if lat or dplat is near the pole, numerical error near the poles could cause a mismatch
        if glat>=88 or dplat>=88:
            dpdir_mask = (cf.periodic_distance(Glon_btstrp, dplon) < 50) & (np.abs(Glat_btstrp - dplat) < 5)
        
        A_dp = A_btstrp[dpdir_mask]

        # debugging
        if len(A_dp) == 0:
            print('lc', lc00)    # print the lc number. A global variablt but only for this case
            print(relation, 'No data for the opposite of', glon, glat)
            continue

        # In case our criteria is too loose
        if len(A) > 500:
            raise ValueError(relation, 'Too many data points for', glon, glat)
        elif len(A_dp) > 500:
            raise ValueError(relation, 'Too many data points for the opposite direction', dplon, dplat)
            

        # get the best fit value
        # dir_mask = (np.abs(Glon_scan - glon) < 2) & (np.abs(Glat_scan - glat) < 1)
        # scanning didn't cover lon=180 and lat=90, it stops at 176 and 88. We
        # extend the mask to include the closest values
        dir_mask = cf.periodic_distance(Glon_scan, glon) <= 2
        if glat > 88:
            dir_mask = dir_mask & (np.abs(Glat_scan - glat) <= 2)
        else:
            dir_mask = dir_mask & (np.abs(Glat_scan - glat) <= 1)

        A_bestfit = A_scan[dir_mask][0]

        # get the best fit value of the opposite direction
        # dpdir_mask = (np.abs(Glon_scan - dplon) < 2) & (np.abs(Glat_scan - dplat) < 1)
        # if len(A_scan[dpdir_mask]) == 0:
        #     dpdir_mask = (np.abs(Glon_scan - dplon) < 4) & (np.abs(Glat_scan - dplat) < 2) 
        dpdir_mask = cf.periodic_distance(Glon_scan, dplon) <= 2
        if dplat > 88:
            dpdir_mask = dpdir_mask & (np.abs(Glat_scan - dplat) <= 2)
        else:
            dpdir_mask = dpdir_mask & (np.abs(Glat_scan - dplat) <= 1)

        A_bestfit_dp = A_scan[dpdir_mask][0]

        if A_bestfit > A_bestfit_dp:
            # calculate the significance
            sigma = np.sqrt((A_bestfit - np.percentile(A, 16))**2 + 
                            (np.percentile(A_dp, 84) - A_bestfit_dp)**2)
            n_sigma = (A_bestfit - A_bestfit_dp) / sigma

            # calculate the variation
            vari = (A_bestfit/A_all)**h0p - (A_bestfit_dp/A_all)**h0p
        else:
            # calculate the significance
            sigma = np.sqrt((A_bestfit_dp - np.percentile(A_dp, 16))**2 + 
                            (np.percentile(A, 84) - A_bestfit)**2)
            n_sigma = (A_bestfit_dp - A_bestfit) / sigma

            # calculate the variation
            vari = (A_bestfit_dp/A_all)**h0p - (A_bestfit/A_all)**h0p


        # return the significance and the amplitude
        n_sigmas[(unique_Glon == glon) & (unique_Glat == glat)] = n_sigma
        variation[(unique_Glon == glon) & (unique_Glat == glat)] = vari

    return n_sigmas, variation, unique_Glon, unique_Glat

# ------------------------------------------------------------------------------

data_dir = '/cosma8/data/do012/dc-he4/analysis'
output_file = '/cosma8/data/do012/dc-he4/analysis_all/h0_direct_compare.csv'

h0_vari = np.array([])
n_sigma = np.array([])
lc_list = []                    # save also the lightcone number for direct comparison
relations = []
for relation, relation_name in zip(['LX-T', 'YSZ-T', 'M-T'], 
                                   ['$L_\\mathrm{{X}}-T$',
                                    '$Y_\\mathrm{{SZ}}-T$',
                                    '$M_\\mathrm{{gas}}-T$']):
    h0_variation = []
    significance = []
    for lc in range(1728):
        lc00 = f'{lc:04d}'
        print(lc00)

        # matching cone size
        if 'YSZ' in relation:
            cone_size = 60
        else:
            cone_size = 75

        # flag filenames
        flag1 = f'{data_dir}/lc{lc00}/fit-all.done'
        flag2 = f'{data_dir}/lc{lc00}/scan-best-fit.done'
        flag3 = f'{data_dir}/lc{lc00}/scan-bootstrap-near-scan.done'

        # set filename
        btstrp_file = f'{data_dir}/lc{lc00}/scan_bootstrap_{relation}_theta{cone_size}.csv'
        bestfit_file = f'{data_dir}/lc{lc00}/scan_best_fit_{relation}_theta{cone_size}.csv'
        fit_all_file = f'{data_dir}/lc{lc00}/fit_all.csv'

        # check if file exists
        if os.path.exists(flag1) and os.path.exists(flag2) and os.path.exists(flag3):
            # read in data
            n_sigmas, variation, glons, glats = get_h0_var(
                btstrp_file,
                bestfit_file,
                fit_all_file,
                relation)
            
            # if "All-NaN slice encountered", usually means the data is corrupted, remove the completion flag and we can rerun it later
            try:
                _argmax = np.nanargmax(n_sigmas) 
            except ValueError: 
                os.remove(flag3)
                continue

            if n_sigmas[_argmax] > 0 and variation[_argmax] > 0:   
                h0_variation.append(variation[_argmax])
                significance.append(n_sigmas[_argmax])
                lc_list.append(lc00)
            else:
                warnings.warn(f'Impossible significance or variation: {n_sigmas[_argmax]}, {variation[_argmax]}')
                os.remove(flag3)
        else:
            continue

    print(len(significance))

    h0_vari = np.concatenate((h0_vari, h0_variation))
    n_sigma = np.concatenate((n_sigma, significance))
    
    relations += [relation_name for i in range(len(significance))] 

lc_list = np.array(lc_list)

# save the data
df = pd.DataFrame({'Relations': relations, 
                   'Delta_H0': h0_vari, 
                   'Significance': n_sigma,
                   'Lightcone': lc_list})
df.to_csv(output_file, index=False)