import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, irfft, rfftfreq
from lmfit import Model
from matplotlib.table import Table


data_sc = np.load("/data/20250515/Complex_Z/SCvsN/tbase_14mK/bias_0v00/vbias_0v00_vac_0v010vpp_tbase_14mK.npz",
                    allow_pickle=True)
data_n = np.load("/data/20250515/Complex_Z/SCvsN/tbase_99mK/bias_0v00/vbias_0v00_vac_0v010vpp_tbase_99mK.npz",
                    allow_pickle=True)


stim_fs = data_sc['fs_sch']
print(f"Stimulation Frequency (SC): {stim_fs} Hz")


channel = 4106


tau_sc = -3.356405e-3


i_sc = data_sc["data"].item()[f"chan{channel}"]
i_n = data_n["data"].item()[f"chan{channel}"]


i_sc_fft = rfft(i_sc)
i_n_fft = rfft(i_n)
n_points = len(i_n)
fs = 125000
frequencies = rfftfreq(n_points, 1/fs)

f_idx = np.searchsorted(frequencies, stim_fs)



omega = 2 * np.pi * frequencies
phase = np.exp(-1j * omega * tau_sc)
i_sc_fft_shifted = i_sc_fft * phase
i_n_fft_shifted = i_n_fft * phase


ratio_complex = i_sc_fft_shifted[f_idx] / i_n_fft_shifted[f_idx]


def transfer_function_ratio(omega, Rshunt, L, Rnormal):
    """Model for the complex impedance ratio."""
    TF_ratio = (Rnormal + Rshunt + 1j * omega * L)/(Rshunt + 1j * omega * L)
    return TF_ratio


model = Model(transfer_function_ratio)


params = model.make_params(Rshunt=250e-6, L=75e-9, Rnormal=7e-3)

params['Rshunt'].min = 240e-6
params['Rshunt'].max = 260e-6
params['Rshunt'].vary = False
#
params['L'].min = 10e-9
params['L'].max = 200e-9
#
# params['Rnormal'].min = 1e-3
# params['Rnormal'].max = 1e-2
# params['Rnormal'].vary = False


omega_fit = 2 * np.pi * stim_fs
complex_ratio_data = np.array([ratio_complex])


result = model.fit(complex_ratio_data, omega=omega_fit, params=params)


print("\n--- lmfit Results for Complex Ratio ---")
print(result.fit_report())


best_Rshunt = result.params['Rshunt'].value
best_L = result.params['L'].value
best_Rnormal = result.params['Rnormal'].value
err_L = result.params['L'].stderr if result.params['L'].stderr is not None else np.nan
err_Rnormal = result.params['Rnormal'].stderr if result.params['Rnormal'].stderr is not None else np.nan

plt.figure(figsize=(8, 6))
ax = plt.gca()
plt.semilogx(stim_fs, np.real(ratio_complex), 'ro', label="Real(Ratio)")
plt.semilogx(stim_fs, np.real(transfer_function_ratio(omega_fit, best_Rshunt, best_L, best_Rnormal)), 'r-',
         label=f"Real(Fit)")
plt.semilogx(stim_fs, np.imag(ratio_complex), 'bo', label="Imag(Ratio)")
plt.semilogx(stim_fs, np.imag(transfer_function_ratio(omega_fit, best_Rshunt, best_L, best_Rnormal)), 'b-',
         label=f"Imag(Fit)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Real and Imaginary Parts of Ratio")
plt.title(f"SC/N transfer function ratio for channel {channel}")
plt.legend(loc="lower left")
plt.grid(True)

table_data = [
    ['Parameter', 'Value', 'Error (1σ)'],
    ['Rshunt', f'{best_Rshunt:.2e}', 'Fixed'],
    ['L', f'{best_L:.2e}', f'{err_L:.2e}'],
    ['Rnormal', f'{best_Rnormal:.2e}', f'{err_Rnormal:.2e}'],
]

table = Table(ax, loc='upper right', bbox=[0.65, 0.7, 0.325, 0.25])    # [left, bottom, width, height]

for i, row in enumerate(table_data):

    cell_0 = table.add_cell(i, 0, 1/len(row), 1, text=row[0], facecolor='white') # Default background
    cell_1 = table.add_cell(i, 1, 1/len(row), 1, text=row[1], facecolor='white')
    cell_2 = table.add_cell(i, 2, 1/len(row), 1, text=row[2], facecolor='white')

    if i == 0: # This is the header row
        cell_0.set_facecolor('lightgray')
        cell_1.set_facecolor('lightgray')
        cell_2.set_facecolor('lightgray')
        cell_0.set_text_props(fontweight='bold')
        cell_1.set_text_props(fontweight='bold')
        cell_2.set_text_props(fontweight='bold')


table.auto_set_font_size(False)
table.set_fontsize(8)
ax.add_table(table)
plt.show()