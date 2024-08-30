
import h5py
import pandas as pd
INPUT = '/data1/yujiehe/data/halo_properties_in_lightcones.hdf5'


# Concatenate the different snapshots
with h5py.File(INPUT, 'a') as fout:
    for observer in list(fout.keys()):
        # Concatenate the different snapshots
        for i, snapshot_name in enumerate(fout[observer].keys()):
            if i == 0:
                df = pd.read_hdf(INPUT, f'{observer}/{snapshot_name}')
            else:
                df = pd.concat([df, pd.read_hdf(INPUT, f'{observer}/{snapshot_name}')])
        df.to_hdf(fout, key=f'{observer}', mode='a')

        # Remove the individual snapshots
        for snapshot_name in list(fout[observer].keys()):
            del fout[f'{observer}/{snapshot_name}']