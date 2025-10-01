import numpy as np
import matplotlib.pyplot as plt
import glob

from IPython.core.pylabtools import figsize
from scipy.signal import savgol_filter

plt.rcParams['figure.figsize'] = (24, 10)
plt.rcParams['font.size'] = 14


def load_and_plot_file(filename, good_channel_ids, axes=None, label=None, filter_data = False, start_id = 20000, stop_id = 40000, xoffsets =0):
    t_sample = 8e-6
    path = f'/data/{filename.split('_')[1]}/data/'
    f = np.load(path+filename, allow_pickle=True)
    dd = f['data']
    times = np.arange(len(dd[:, 0]))*t_sample
    if axes is None:
        fig, axes = plt.subplots(1, len(good_channel_ids))
    print(channel_id)
    if filter_data:
        times_ = savgol_filter(times, window_length=21, polyorder=3)[start_id:stop_id]
        data_ = savgol_filter(dd[:, channel_id], window_length=21, polyorder=3)[start_id:stop_id]
    else:
        times_ = times[start_id:stop_id+int(xoffsets/t_sample)]
        data_ = dd[:, channel_id][start_id:stop_id+int(xoffsets/t_sample)]
    ydata = data_ - data_[np.argmin(abs(times_-xoffsets))]
    xdata = times_- xoffsets
    line, = axes.plot(xdata[xdata>0], ydata[xdata>0], alpha=1, label=label)
    median_value = np.median(dd[:, channel_id][20000:40000])
    last_time_index = len(times_) - 1 #get the last index
    last_data_point = data_[last_time_index] #get the last data point
    # axes.text(
    #     times_[last_time_index],  # x-coordinate: the last time point
    #     last_data_point,  # y-coordinate: the last data point on the line
    #     f"{median_value:.3f}",
    #     ha='right',  # Horizontal alignment: right (so the text is to the left of the point)
    #     va='bottom',  # Vertical alignment: bottom (or top, depending on which looks best)
    #     color=line.get_color()
    # )
    # axes.set_title(label)
    axes.set(ylim=(-0.5, 0.5))
    axes.grid(ls='--', alpha=0.5, zorder=0)



if __name__ == "__main__":
    good_channel_ids = [2, 3, 9, 13,24, 28] #18, too noisy
    # good_channel_ids = [2,3,9,13,24]
    good_channel_ids = [24]
    choice = 7
    title = None
    if choice == 1:
        filenames = [ '/data/20250304/data/20250304_103626_data.npz', '/data/20250304/data/data_20250304_105533_60mK.npz', '/data/20250304/data/data_20250304_120106_60mK_bias0v005.npz',
                      '/data/20250304/data/data_20250304_113026_24mK_bias0v005_from60mK.npz', '/data/20250304/data/data_20250304_113133_24mK_bias0v005_from60mK_biasnow_0v0.npz',
                      '/data/20250304/data/data_20250304_123717_24mK_reset.npz'
                      ]
        labels = ['24 mK, no bias', '60 mK no bias', ' 60mK Vb= 5mV', '24mK Vb =5 mV', '24 mK no bias', '24 mK reset']
    elif choice == 2:
        filenames = [ '/data/20250304/data/data_20250304_125931_24mK_start.npz',  '/data/20250304/data/data_20250304_130148_24mK_laseron_nobias.npz',
                     '/data/20250304/data/data_20250304_130241_24mK_laseron_bias0v22.npz', '/data/20250304/data/data_20250304_130351_24mK_laseroff_bias0v22.npz',
                     '/data/20250304/data/data_20250304_130431_24mK_laseroff_bias0v0.npz', '/data/20250304/data/data_20250304_131120_24mK_laseroff_bias0v0_reset.npz'
                     ]
        labels = ['Laser OFF, bias = 0 mV', 'Laser ON, bias = 0 mV', 'Laser ON, bias = 22 mV', 'Laser OFF, bias = 22 mV', 'Laser OFF, bias = 0 mV', 'Laser ON, bias = 0 mV']
    elif choice == 3:
        filenames = [ '/data/20250304/data/data_20250304_150359_24mK_laseron_bias0v0.npz',
                     '/data/20250304/data/data_20250304_150437_24mK_laseron_bias0v2.npz', '/data/20250304/data/data_20250304_150556_24mK_laseroff_bias0v2.npz',
                     '/data/20250304/data/data_20250304_150631_24mK_laseroff_bias0v0.npz',
                     '/data/20250304/data/data_20250304_150703_24mK_laseron_bias0v0.npz']
        labels = [ 'Laser ON, bias = 0 mV', 'Laser ON, bias = 200 mV',
                  'Laser OFF, bias = 200 mV', 'Laser OFF, bias = 0 mV', 'Laser ON, bias = 0 mV']
        title = 'Persistent current experiments at 24 mK bath \n MEMS (x,y) = (-0.1, -0.44) \t Laser settings: pw = 20 ns, HL = 10V, LL = 10 mV, pp = 20 ms'

    elif choice == 4:
        filenames = ['/data/20250304/data/data_20250304_154147_24mK_laseroff_bias0v0.npz', '/data/20250304/data/data_20250304_154219_24mK_laseron_bias0v0.npz',
                     '/data/20250304/data/data_20250304_154247_24mK_laseron_bias0v2.npz', '/data/20250304/data/data_20250304_154321_24mK_laseroff_bias0v2.npz',
                     '/data/20250304/data/data_20250304_154352_24mK_laseroff_bias0v0.npz', '/data/20250304/data/data_20250304_154423_24mK_laseron_bias0v0.npz']
        labels = ['Laser OFF, bias = 0 mV', 'Laser ON, bias = 0 mV', 'Laser ON, bias = 200 mV',
                  'Laser OFF, bias = 200 mV', 'Laser OFF, bias = 0 mV', 'Laser ON, bias = 0 mV']
        title = 'Persistent current experiments at 24 mK bath \n MEMS (x,y) = (0.0, -0.44) \t Laser settings: pw = 20 ns, HL = 10V, LL = 10 mV, pp = 20 ms'

    elif choice == 5:
        filenames = ['/data/20250304/data/data_20250304_162628_24mK_laseroff_bias0v0.npz', '/data/20250304/data/data_20250304_162658_24mK_laseron_bias0v0.npz',
                     '/data/20250304/data/data_20250304_162722_24mK_laseron_bias0v2.npz', '/data/20250304/data/data_20250304_162747_24mK_laseroff_bias0v2.npz',
                     '/data/20250304/data/data_20250304_162832_24mK_laseroff_bias0v0.npz', '/data/20250304/data/data_20250304_162905_24mK_laseron_bias0v0.npz']
        labels = ['Laser OFF, bias = 0 mV', 'Laser ON, bias = 0 mV', 'Laser ON, bias = 200 mV',
                  'Laser OFF, bias = 200 mV', 'Laser OFF, bias = 0 mV', 'Laser ON, bias = 0 mV']
        title = 'Persistent current experiments at 24 mK bath \n MEMS (x,y) = (-0.1, -0.2) \t Laser settings: pw = 20 ns, HL = 10V, LL = 10 mV, pp = 20 ms'

    elif choice == 6:
        filenames = ['data_20250305_113559__simultaneous.npz', 'data_20250305_114103__simultaneous_laseroff.npz']
        labels = ['laseron', 'laseroff']

    elif choice == 7:
        filenames = ['data_20250305_154037_ch4120_HL10V.npz', 'data_20250305_154659_ch4120_HL9V5.npz', 'data_20250305_154936_ch4120_HL9V.npz']
        good_channel_ids = [18,24]
        labels = ['10V', '9.5V', '9V']
        xoffsets = [0, 0.018, 0.007]
        # xoffsets = [0, 0 , 0]
        startd_ids = [0, 2250, 875]
        # startd_ids = [0,0,0]

    elif choice == 8:
        filenames = ['data_20250305_154156_ch4098_HL10V.npz', 'data_20250305_154743_ch4098_HL9V5.npz', 'data_20250305_155024_ch4098_HL9V.npz']
        good_channel_ids = [18, 24]
        labels = ['10 V', '9.5 V', '9V']
        xoffsets = [0.0255,0,0.0663]
        startd_ids = [0, 0, 0]
    fig, ax = plt.subplots(  len(good_channel_ids), 1, sharex=True,figsize=(10,8),)
    fig.suptitle(f'Current traces for targeted laser beam on {filenames[0].split('_')[3]}')
    for i, channel_id in enumerate(good_channel_ids):
        ax[i].text(0.45, 0.85, 'Channel {}'.format(channel_id+4096), transform=ax[i].transAxes)
        ax[i].set_ylabel(ylabel=r'Current ($\phi_0$)')
        for j, filename in enumerate(filenames):
            load_and_plot_file(filename, channel_id, axes = ax[i], label = labels[j], filter_data=False, start_id=0, stop_id=int(0.4/8e-6), xoffsets = xoffsets[j])
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.05, hspace = 0.05)
        plt.tight_layout()
        ax[i].legend(title='Laser high level')
    ax[i].set(xlabel = 'Time(s)')
    plt.tight_layout()
    plt.show()


