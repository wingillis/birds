import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import h5py
import data_handling as handler
from glob import glob
from scipy.signal import butter,filtfilt
from generate_param_dump import generate_data_summary

color_palette = [(0.7200000000000001, 0.11839999999999996, 0.07999999999999996),
 (0.7200000000000001, 0.50240000000000007, 0.07999999999999996),
 (0.55359999999999987, 0.7200000000000001, 0.07999999999999996),
 (0.16959999999999992, 0.7200000000000001, 0.07999999999999996),
 (0.07999999999999996, 0.7200000000000001, 0.37440000000000023),
 (0.07999999999999996, 0.68159999999999987, 0.7200000000000001),
 (0.07999999999999996, 0.29759999999999948, 0.7200000000000001),
 (0.24640000000000042, 0.07999999999999996, 0.7200000000000001),
 (0.63039999999999996, 0.07999999999999996, 0.7200000000000001),
 (0.7200000000000001, 0.07999999999999996, 0.42559999999999992)]

def main(y_range=[-2, 2]):
    files = glob('stim*.mat')
    colors = sns.hls_palette(9, l=0.4, s=0.85)
    for f in files:
        handle_data(f, colors, y_range)

    return True

def handle_data(f, colors, y_range):
    df = h5py.File(f, 'r')
    data = df['data']
    plot(data, f[:-4], colors, y_range)
    df.close()

def summary_plot(data, name, colors, y_range):
    '''old, used for looping through the data and generating
    figures based on that'''
    t = np.array(data['tdt']['times_aligned'])
    t_ind = np.where(((t>-0.0005)*(t<0.006)))[0]
    stim = data['tdt']['response']
    plt.figure(figsize=(10,8))
    if len(stim.shape) > 2:
        lines = plt.plot(t[t_ind]*1000, stim[1,t_ind,:]*1000, color=colors[4])
    else:
        lines = plt.plot(t[t_ind]*1000, stim[t_ind,:]*1000, color=colors[4])
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.xlim([-0.5, 5])
    plt.ylim(y_range)
    plt.title('Current: {}uA; N reps: {};\nactive electrodes: {}; negative first: {}'
       .format(int(data['stim']['current_uA'].value),
       int(data['stim']['n_repetitions'].value),
       ''.join(map(lambda a: str(int(a)), data['stim']['active_electrodes'].value.flatten())),
       ''.join(map(lambda a: str(int(a)), data['stim']['negativefirst'].value.flatten()))))
    plt.savefig(name+'.png')
    plt.clf()
    plt.close()

def lowpass_filter(mix, freq=5):
    data, fs = mix
    b,a = butter(4, freq/(fs/2), 'highpass')
    return filtfilt(b,a,data.T).T, fs

def average(mix):
    data, fs = mix
    col, row = data.shape
    if col < row:
        return np.mean(data, axis=0).reshape((1,-1)), fs
    else:
        return np.mean(data, axis=1).reshape((-1,1)), fs

def descriptive_title(f):
    params = generate_data_summary(f)
    current = round(params['current'],1)
    bird = params['bird']
    stim_electrodes = ','.join([str(i+1) for i, x in enumerate(params['stim_electrodes']) if x==1])
    negative = ','.join([str(i+1) for i,x in enumerate(params['negative_first']) if x == 1])
    pulse_width = params['pulse_halftime']
    if not negative:
        negative = 'none'
    ipi = params['interpulse_interval']
    return 'bird: {}; current: {}uA; electrodes: {};<br> negative polarity: {}; ipi: {}us; pulse width: {}us'.format(
        bird, current, stim_electrodes, negative, ipi, pulse_width*1e6)

def plot_comparison(f1, f2, data, xs=[-0.5, 7], ys=[-400, 1000], s=0):
    '''Compare file 1 (red) with file 2 (blue), taking the mean and std
    of the signal'''
    d = handler.extract_data(f1)
    d2 = handler.extract_data(f2)
    # due to the 60Hz noise of the water heater, I have to filter it out with a notch filter
    temp =  np.array([56.0,64.0])/(d['fs']/2.0)
    # 3 is the highest order I can go without it going crazy
    b,a = butter(3, temp[0], btype='bandstop')

    y = filtfilt(b,a,lowpass_filter((d['stim']*1e6, d['fs']))[0].T).T
    y2 = filtfilt(b,a,lowpass_filter((d2['stim']*1e6, d2['fs']))[0].T).T

    ystd = np.std(y, axis=1)
    ymean = average((y, d['fs']))[0][:,0]
    t = d['t']*1e3
    y2std = np.std(y2, axis=1)
    y2mean = average((y2, d2['fs']))[0][:,0]
    t2 = d2['t']*1e3
    plt.figure(figsize=(8,6))
    plt.fill_between(t, (ymean-ystd), (ymean+ystd), alpha=0.5, facecolor=color_palette[0])
    plt.plot(t, ymean, color=color_palette[0])
    plt.fill_between(t2, y2mean - y2std, y2mean + y2std, alpha=0.5, facecolor=color_palette[5])
    plt.plot(t2, y2mean, color=color_palette[5])
    plt.xlim(xs)
    plt.ylim(ys)
    plt.ylabel('Voltage (µV)')
    plt.xlabel('Time (ms)')
    plt.title(descriptive_title(f1).replace('<br>', '\n'))
    plt.legend([str(handler.get_current(data,f1))+'µA', str(handler.get_current(data,f2))+'µA'])
    if s and type(s) == str:
        plt.savefig(s + '.pdf')
    elif s:
        plt.savefig(f1[:-4]+ f2[:-4]+'-comaparison.pdf')


def plot_stim(f, y_range=None, x_range=None, func=None, title=None, save=False):
    '''Used mainly for reference when looking at the notebook'''
    title = title if title else 'Stimulation Response'
    with h5py.File(f, 'r') as df:
        data = df['data']
        fs = data['tdt']['fs'].value
        t = data['tdt']['times_aligned'][()].flatten()
        stim = data['tdt']['response']
        if len(stim.shape) >2:
            stim = stim[1,:,:]
        else:
            stim = stim[:,:]
        if func:
            stim, fs = func((stim, fs))
        plt_data = []

        for i in range(stim.shape[1]):
            plt_data += [Scatter(x=t*1000, y=stim[:,i]*1e6, showlegend=False, hoverinfo='none',
                                 line={'color': 'rgb({},{},{})'.format(*color_palette[4])})]

        layout = {'title':title,
                 'xaxis': {'title': 'Time (ms)', 'range':x_range},
                 'yaxis': {'title': 'Voltage (uV)', 'range':y_range}}

        fig = {'data':plt_data, 'layout':layout}

        iplot(fig)
        if save:
            return fig
