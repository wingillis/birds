import h5py
# import pandas as pd

def extract_data(f):
    with h5py.File(f, 'r') as df:
        t = df['data']['tdt']['times_aligned'][()].flatten()
        stim = df['data']['tdt']['response']
        if len(stim.shape) > 2:
            stim = stim[0,:,:]
        fs = df['data']['tdt']['fs'].value
        dataframe = {}
        dataframe['t'] = t
        dataframe['stim'] = stim.value
        dataframe['fs'] = fs

    return dataframe

def get_current(data, f):
    keys = data.keys()
    keys = list(filter(lambda a: 'current_approx' in a, keys))
    for key in keys:
        if f in data[key]:
            return key[1]
    print('No current found for {}'.format(f))
