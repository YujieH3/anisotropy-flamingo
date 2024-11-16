import pandas as pd
import pandas as pd
import os
import numpy as np
import sys
sys.path.append('/cosma/home/do012/dc-he4/anisotropy-flamingo/tools/')
import clusterfit as cf
data_dir = '/cosma8/data/do012/dc-he4/analysis'
output_file = '/cosma/home/do012/dc-he4/anisotropy-flamingo/data/analysis_all/h0_angle_constraints.csv'
output_mc = '/cosma/home/do012/dc-he4/anisotropy-flamingo/data/analysis_all/h0_angle_constraint_mc.csv'
output_scan = '/cosma/home/do012/dc-he4/anisotropy-flamingo/data/analysis_all/h0_angle_constraint_scan.csv'

glons_scan = []
glats_scan = []
lc_scan = []
relations_scan = []

glons_mc = []
glats_mc = []
lc_mc = []
relations_mc = []
for relation, relation_name in zip(['LX-T', 'YSZ-T', 'M-T'], 
                                   ['$L_\\mathrm{{X}}-T$',
                                    '$Y_\\mathrm{{SZ}}-T$',
                                    '$M_\\mathrm{{gas}}-T$']):
    for lc in range(1728):
        lc00 = f'{lc:04d}'
        # print(lc00)
        # Get the direction constraint for MCMC
        filename = f'{data_dir}/lc{lc00}/h0_mcmc.csv'

        if os.path.exists(filename):
            df = pd.read_csv(filename)
            
            glon_max = df[df['scaling_relation']==relation]['vlon'].values[0]
            glat_max = df[df['scaling_relation']==relation]['vlat'].values[0]
            glons_mc.append(glon_max) 
            glats_mc.append(glat_max) 
            
            lc_mc.append(lc)
            relations_mc.append(relation)
            print(f'Relation {relation}, MC results found: {lc00}, Direction: {glon_max}, {glat_max}')
        else:
            continue

        # Get the chi-square bootstrapping results
        if 'YSZ' in relation:
            cone_size = 60
        else:
            cone_size = 75

        filename = f'{data_dir}/lc{lc00}/scan_best_fit_{relation}_theta{cone_size}.csv'
        
        if os.path.exists(filename):
            # read the data
            df = pd.read_csv(filename)
            
            # the largest dipole is found using the best fit scan. No significance done.
            max_lon, max_lat, max_dipole_value = cf.find_max_dipole(map=df['A'], lons=df['Glon'], lats=df['Glat'])
            glons_scan.append(max_lon)
            glats_scan.append(max_lat)

            lc_scan.append(lc)
            relations_scan.append(relation)
            print(f'Relation {relation}, Scan results found: {lc00}, Direction: {max_lon}, {max_lat}')
        else:
            continue
        

    df_mc = pd.DataFrame({
        'Relations': relations_mc, 
        'Glons': glons_mc,
        'Glats': glats_mc,
        'Lightcone': lc_mc,
    })
    df_mc.to_csv(output_mc)

    df_scan = pd.DataFrame({
        'Relations': relations_scan,
        'Glons': glons_scan,
        'Glats': glats_scan,
        'Lightcone': lc_scan
    })
    df_scan.to_csv(output_scan)

# Merge on 
data = pd.merge(df_mc, df_scan, on=('Relations', 'Lightcone'), how='inner', suffixes=('_mc', '_scan'))
data.to_csv(output_file)