import pandas as pd
import sys
sys.path.append('/home/yujiehe/anisotropy-flamingo')
import tools.clusterfit as cf
import os

# --------------------------------CONFIGURATION---------------------------------

data_dir = '/home/yujiehe/anisotropy-flamingo/data/fits/'
best_fit_scan_names = 'scan_best_fit'
bootstrap_scan_names = 'scan_btstrp'

overwrite = True

# --------------------------------------MAIN------------------------------------

if __name__ == '__main__':
    for relation in cf.CONST.keys():

        flag1 = False
        flag2 = False
        for root, dirs, files in os.walk(data_dir):
            for name in files:
                if bootstrap_scan_names in name and relation in name:
                    bootstrap_file = os.path.join(root, name)
                    print(f'Found bootstrap scan: {bootstrap_file}')
                    flag1 = True
                    break
            for name in files:
                if best_fit_scan_names in name and relation in name\
                and bootstrap_scan_names not in name and bootstrap_file[-7:] in name:
                    best_fit_file = os.path.join(root, name)
                    print(f'Found best fit scan: {best_fit_file}')
                    best_fit = pd.read_csv(best_fit_file)
                    flag2 = True
                    break
            break # only search in the first level of the directory

        if flag1 == False and flag2 == False:
            raise FileNotFoundError(f'No best fit or bootstrap scan found for {relation}.')
                
        if 'n_sigma' in best_fit.columns and overwrite == False:
            print(f'Significance map already exist for {relation}.')
            continue
        else:
            print(f'Calculating significance map: {relation}.')
            df = cf.significance_map(best_fit_file, bootstrap_file)
            best_fit['n_sigma'] = df['n_sigma']
            best_fit.to_csv(best_fit_file, index=False)
            print(f'Significance map saved: {best_fit_file}')

    print('All done!')
