import numpy as np
import os
import matplotlib.pyplot as plt
import scipy
plt.rcParams['font.size'] = 10

fmax = 17e3
f_nyq = 125e3 / 2
phi0 = scipy.constants.value(u"mag. flux quantum")

def load_noise_file(filename):
    f = np.load(filename, allow_pickle=True)
    psd = f['Pxx'].item()[162005088]
    freqs = f['f'].item()[162005088]
    return psd, freqs

def phi0_to_amp(x):
    min_SI = 180.5e-12 # 248e-12
    min_phi0_per_amp = min_SI / phi0
    return x / min_phi0_per_amp

def get_parent_path(filename):
    return os.path.abspath(f"/data/{filename.split('_')[1]}/noise/{filename}")

def plot_multiple_channels(filename, good_channels=None):
    if good_channels is None:
        good_channels = np.arange(32)
    file_path = get_parent_path(filename)
    psd, freqs = load_noise_file(file_path)
    white_noise_level = {}
    fig, ax = plt.subplots(figsize=(10, 6))
    for channel in good_channels:
        psd2 = psd[:, channel]
        ax.loglog(freqs, np.sqrt(psd2) * 1e6, label=channel)
        white_noise_level[channel] = round(np.sqrt(np.nanmedian(
            psd2[(freqs > 0.86 * fmax) & (freqs < 0.99 * fmax)])) * 1e6, 2)

    ax.set(ylim=(1, 2e3), xlim=(0.5, f_nyq), xlabel='Frequency (Hz)',
           ylabel=r'Power spectral density ($\mu\Phi_{0}/\sqrt{\mathrm{Hz}}$)',
           title='Noise PSD')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    for loc in np.array([0.86, 0.99]) * fmax:
        ax.axvline(x=loc, linestyle="--", color="gray")

    ax.legend(title='Channels', ncols=3, loc=3)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for channel in good_channels:
        ax2.plot(channel, white_noise_level[channel], 'k.')
    ax2.set(xlabel='Channel', ylabel=r'Noise PSD ($\mu \phi_0$/$\sqrt{Hz}$)')
    ax2.grid(True, which='both', linestyle=':', alpha=0.5, lw=0.3)

def compare_noise_spectra(datasets, channels=None, yval='flux'):
    if channels is None:
        channels = [11]  # default to ch11

    cmap = plt.get_cmap('plasma')
    vlines = np.array([0.86, 0.99]) * fmax

    for channel in channels:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        # for v in vlines:
        #     ax.axvline(x=v, linestyle="--", color="gray")

        for i, config in enumerate(datasets):
            fname = config['filename']
            label = config.get('label', f'File {i+1}')
            color = config.get('color', cmap(i / len(datasets)))
            zorder = config.get('zorder', i)
            alpha = config.get('alpha', 1.0)

            file_path = get_parent_path(fname)
            psd, freqs = load_noise_file(file_path)
            psd2 = psd[:, channel]

            if yval == 'flux':
                ydata = np.sqrt(psd2) * 1e6
                ylabel = r'Power spectral density ($\mu\Phi_{0}/\sqrt{\mathrm{Hz}}$)'
                wn = round(np.sqrt(np.nanmedian(
                    psd2[(freqs > 0.86 * fmax) & (freqs < 0.99 * fmax)])) * 1e6, 2)
            else:
                ydata = phi0_to_amp(np.sqrt(psd2)) * 1e12
                ylabel = r'Power spectral density (pA$/\sqrt{\mathrm{Hz}}$)'
                wn = round(phi0_to_amp(np.sqrt(np.nanmedian(
                    psd2[(freqs > 0.86 * fmax) & (freqs < 0.99 * fmax)]))) * 1e12, 2)

            ax.loglog(freqs, ydata, label=f"{label}", color=color,
                      zorder=zorder, alpha=alpha)
            # ax.text(0.8, 0.75 - 0.05 * i, f"{wn}", color=color, transform=ax.transAxes)

        ax.set(ylim=(1, 6e3), xlim=(0.5, f_nyq), xlabel='Frequency (Hz)', ylabel=ylabel, title=f'Noise PSD for Channel {channel}')
        ax.legend(ncols=2)
        ax.grid(True, which='both', linestyle=':', alpha=0.8, lw=0.75, zorder= -1)



if __name__=="__main__":
    rbias= 741 #1965.4
    f_nyq = 125e3 / 2
    Rsh = 250e-6
    Rn_fixed = 6.75e-3

    good_channels = np.arange(31)
    plot_multiple_channels(filename = 'noise_20250915_115242_20mK_vbias_0v0.npz',
                           good_channels=good_channels)
    # filenames = ['noise_20250219_145647_20mK_bias0v0_.npz', #0% Rn
    #              'noise_20250219_153221_20mK_bias0v236_.npz', #4.5%
    #              'noise_20250219_155858_20mK_bias0v237_.npz', #4.7%
    #              'noise_20250219_151955_20mK_bias0v239_.npz', #5% Rn
    #              'noise_20250219_145355_20mK_bias0v25_.npz', #7% Rn
    #              'noise_20250219_152045_20mK_bias0v269_.npz',  #9% Rn
    #              'noise_20250219_152343_20mK_bias0v285_.npz' #11% Rn
    #              ]
    # filenames = [
    #              'noise_20250219_145355_20mK_bias0v25_.npz',  # 7% Rn
    #             'noise_20250221_130915_20mK_bias0v25_newconnx2.npz',
    #             'noise_20250226_121145_20mK_bias0v28.npz',
    #              'noise_20250226_131737_20mK_bias0v25.npz'
    #              ]
    # filenames =  ['noise_20250303_095432_24mK_bias0v015_mems.npz',
    #              'noise_20250303_084707_24mK_bias0v015.npz'] # both to give Rtes = 469 mOhm on ch11
    # filenames = ['noise_20250303_082134_24mK_bias0v0.npz',
    #              'noise_20250303_083606_24mK_bias0v006.npz'
    #              ]
    # filenames = ['noise_20250226_131737_20mK_bias0v25.npz', # A12 taken at 7% Rn, 20 mK scepter
    #              'noise_20250310_101512_24mK_bias0v244.npz',
    #              'noise_20250310_102236_24mK_bias0v244_mems-cable-disconnected.npz'] # A14 taken at 7% Rn Ch 11, 24 mK scepter
    # filenames = ['noise_20250226_131737_20mK_bias0v25.npz',
    #              'noise_20250313_111019_24mK_vbias_0v244_memsdisabled.npz', 'noise_20250313_132225_24mK_vbias_0v24_memsdisabled.npz',
    #               'noise_20250313_133012_24mK_vbias_0v241_memsdisabled.npz', 'noise_20250313_134301_24mK_vbias_0v241_memsenabled.npz']
    # # filenames = ['noise_20250219_145647_20mK_bias0v0_.npz','noise_20250317_093907_20mK_vbias_0v0_memsdisabled.npz',
    # #              'noise_20250317_094737_20mK_vbias_0v0_memsdisabled_hemtgndremoved.npz',
    # #              'noise_20250317_095234_20mK_vbias_0v0_memsdisabled_hemtgndremoved_lnachassisgnd.npz', 'noise_20250317_130950_45dbm.npz', 'noise_20250317_131230_47dbm.npz']
    #
    # filenames = ['noise_20250219_145647_20mK_bias0v0_.npz','noise_20250219_145355_20mK_bias0v25_.npz', 'noise_20250317_133633_20mK_vbias_0v28.npz',
    #              'noise_20250317_133800_20mK_vbias_0v30.npz', 'noise_20250317_134800_21mK_vbias_0v28.npz',
    #              'noise_20250317_134926_21mK_vbias_0v32.npz', 'noise_20250317_131230_47dbm.npz', 'noise_20250317_155626_21mK_vbias_0v32_memsunplugged.npz', 'noise_20250317_161329_21mK_vbias_0v32_memsdisabled2.npz',
    #              'noise_20250317_161618_21mK_vbias_0v32_memsenabled_ch4017.npz']
    #
    # filenames = ['noise_20250318_135602_21mK_vbias0v32_memsdisabled.npz',
    #              'noise_20250318_140553_21mK_vbias0v32_memsenabled_x0_y0.npz',
    #              'noise_20250318_150516_21mK_vbias0v32_mems_sine_11Hz.npz','noise_20250318_151641_21mK_vbias0v32_mems_sine_37Hz.npz',
    #              'noise_20250318_142009_21mK_vbias0v32_memsenabled_x1_y1.npz',
    #              'noise_20250318_144753_21mK_vbias0v32_memsenabled_scanning.npz',
    #              'noise_20250317_155626_21mK_vbias_0v32_memsunplugged.npz']
    # filenames = ['noise_20250219_145647_20mK_bias0v0_.npz',
    #               'noise_20250402_135634_15mK_vbias_0v0_lnagnd_tonepowerm42_2.npz']#, "noise_20250407_162622_15mK_vbias_0v25.npz", 'noise_20250402_140400_15mK_vbias_0v325.npz', 'noise_20250403_122958_15mK_vbias_0v35.npz',
    #               #'noise_20250403_123354_15mK_vbias_0v45.npz']
    #
    # filenames = ['noise_20250507_122944_15mK_vbias_0v0_125kHz_m51dBm.npz','noise_20250507_122834_15mK_vbias_0v0_125kHz_m47dBm.npz',
    #               'noise_20250507_103106_15mK_vbias_0v0_m45dBm.npz', 'noise_20250507_123642_15mK_vbias_0v0_125kHz_m42dBm.npz',
    #               'noise_20250507_123044_15mK_vbias_0v0_125kHz_m40dBm.npz', 'noise_20250507_123152_15mK_vbias_0v0_125kHz_m38dBm.npz']
    #
    # zorders = [5, 0,1,2,3,4]
    # alphas = [0.6, 1,1,1,1,1]
    # vbias = [0, 0.236,0.237, 0.239, 0.25, 0.269, 0.285]
    # labels = ['old direct connx', 'new coax panel', '2/26:0.28Vbias', '2/26:0.25Vbias']
    # labels = ['mems: A13', '15 mV bias: A13']
    # labels= ['0', '5 mV']
    # labels = ['A12', 'A14', 'A14-mems disconnected']
    # labels = ['A12,no bias','A12: Vbias=0.25, no mems', 'mems disabled: vbias=0.244', 'mems disabled: Vbias=0.24V',
    #           'mems disabled: Vbias=0.241V', 'mems enabled: vbias=0.241V']
    # # labels = ['A12: 20mK, bias = 0V', 'A15: 20mK, bias = 0V', 'A15: 20mK, bias = 0V, HEMT GND removed',
    # #           'A15: 20mK, bias = 0V, HEMT GND removed, LNA chassis grounded', '45dbm','47dbm']
    # labels = ['A12,no bias','A12: 0.25V', 'A15:0.28V', 'A15:0.3V', 'A15, 21mK:0.28V', 'A15,21mK:0.32V', 'unbiased', 'mems unplugged', 'mems disabled, plugged in', 'mems enabled, pointed at 4107']
    # labels = ['MEMS Disabled', 'Enabled (0,0)', '11Hz', '37Hz', 'Enabled (1,1)', 'MEMS scanning']
    # labels = ['A12: flood illumination',  'A17: MEMS', 'A17:0.25V', 'A17:0.325V', 'A17:0.35V', 'A17:0.45V']
    # labels = ['-51 dBm', '-47 dBm', '-45 dBm', '-42 dBm', '-40 dBm', '-38 dBm']

    # datasets = [
    #     {"filename": "noise_20250507_122944_15mK_vbias_0v0_125kHz_m51dBm.npz", "label": "-51 dBm"},
    #     {"filename": "noise_20250507_122834_15mK_vbias_0v0_125kHz_m47dBm.npz", "label": "-47 dBm"},
    #     {"filename": "noise_20250507_103106_15mK_vbias_0v0_m45dBm.npz", "label": "-45 dBm"},
    #     {"filename": "noise_20250507_123642_15mK_vbias_0v0_125kHz_m42dBm.npz", "label": "-42 dBm"},
    #     {"filename": "noise_20250507_123044_15mK_vbias_0v0_125kHz_m40dBm.npz", "label": "-40 dBm"},
    #     {"filename": "noise_20250507_123152_15mK_vbias_0v0_125kHz_m38dBm.npz", "label": "-38 dBm"}
    # ]
    #
    # datasets = [
    #     {"filename": "noise_20250516_095808_15mK_vbias_0v0.npz", "label": "0V, LNA GND in"},
    #     {"filename": "noise_20250516_095948_15mK_vbias_0v0_LNAgndout.npz", "label": "LNA GND out"},
    #     {"filename": "noise_20250516_100318_15mK_vbias_0v0_QBoxgndout.npz", "label": "Qbox GND out, LNA GND in"},
    #     {"filename": "noise_20250516_100910_15mK_vbias_0v1_QBoxgndout.npz", "label": "0.1 V"},
    #     {"filename": "noise_20250516_101004_15mK_vbias_0v2_QBoxgndout.npz", "label": "0.2 V"},
    #     # {"filename": "noise_20250507_123152_15mK_vbias_0v0_125kHz_m38dBm.npz", "label": "-38 dBm"}
    # ]

    datasets = [{"filename": 'noise_20250318_135602_21mK_vbias0v32_memsdisabled.npz', "label": "Disabled"},
                 {"filename": 'noise_20250318_140553_21mK_vbias0v32_memsenabled_x0_y0.npz', "label": r"Centered (0$^{\circ}$, 0$^{\circ}$)"},
                {"filename": 'noise_20250318_150516_21mK_vbias0v32_mems_sine_11Hz.npz', "label": "Sine (11 Hz)"},
                {"filename": 'noise_20250318_151641_21mK_vbias0v32_mems_sine_37Hz.npz', "label": "Sine (37 Hz)"},
                {"filename": 'noise_20250318_142009_21mK_vbias0v32_memsenabled_x1_y1.npz', "label": r"Offset (5$^{\circ}$, 5$^{\circ}$)"},
                {"filename": 'noise_20250318_144753_21mK_vbias0v32_memsenabled_scanning.npz', "label": "Scanning"},
                 {"filename": 'noise_20250317_155626_21mK_vbias_0v32_memsunplugged.npz', "label": "Unplugged"}]

    datasets = [{"filename": 'noise_20250910_102316_25mK_vbias_0v25.npz', "label": "25 mK 0.25 V"},
                {"filename": 'noise_20250910_115604_25mK_vbias_0v30.npz', "label": "25 mK 0.30 V"},
                {"filename": 'noise_20250910_115147_25mK_vbias_0v65.npz', "label": "25 mK 0.65 V"},
                {"filename": 'noise_20250910_115056_25mK_vbias_0v75.npz', "label": "25 mK 0.75 V"},

                {"filename": 'noise_20250910_151754_25mK_vbias_1v0.npz', "label": "25 mK 1.0 V"},
                {"filename": 'noise_20250910_152026_25mK_vbias_1v5.npz', "label": "25 mK 1.5 V"},]

    datasets = [{"filename": 'noise_20250915_115242_20mK_vbias_0v0.npz', "label": "20 mK 0 V"},
                {"filename": 'noise_20250915_114720_20mK_vbias_0v65.npz', "label": "25 mK 0.65 V"},
                {"filename": 'noise_20250915_115011_20mK_vbias_0v75.npz', "label": "25 mK 0.75 V"},
                {"filename": 'noise_20250915_114817_20mK_vbias_1v00.npz', "label": "25 mK 1.0 V"},

                {"filename": 'noise_20250910_151754_25mK_vbias_1v0.npz', "label": "25 mK 1.0 V chip A"},
                {"filename": 'noise_20250915_115124_20mK_vbias_1v25.npz', "label": "25 mK 1.25 V"}, ]

    datasets = [{"filename": "noise_20250917_100304_20mK_vbias_0v0.npz", "label": " 20 mK, unbiased"},
                {"filename": "noise_20250917_142330_20mK_vbias_0v8.npz" , "label": " 20 mK, 0.8 V"},]


    compare_noise_spectra(datasets, channels=[21,23], yval='current')
    plt.show()
