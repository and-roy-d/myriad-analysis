import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy
from scipy.fft import rfft, irfft, rfftfreq
from matplotlib.colors import Normalize
from matplotlib.cm import coolwarm

phi0 = scipy.constants.value(u"mag. flux quantum")


def arb_to_amp(in_val):
    min_SI = 248e-12
    min_phi0_per_amp = min_SI / phi0
    arbs_per_phi0 = 4096
    amp_per_arb = 1 / min_phi0_per_amp / arbs_per_phi0
    return in_val * amp_per_arb


channel_id = 4106
channel_id_str = f"chan{channel_id}"
base_dir = Path("/data/20250514/Complex_Z/20250514_122625")
fs = 125000


zero_bias_shift = -0.1488233

sample_shifts_nonzerobias = [0]
for sample_shift in sample_shifts_nonzerobias:
    non_zero_bias_shift = zero_bias_shift + sample_shift * 1 / fs

    # Dictionary to store tau_shift for each bias point
    tau_shifts_by_bias = {}

    # Temporary storage for raw data and other parameters for all bias points
    # This will hold (bias_val, raw_current_data, fs_sch_array, true_voltage_amplitudes)
    all_raw_data_info = []

    # --- First Pass: Load all raw data and determine tau_shifts ---
    for bias_dir in sorted(base_dir.glob('bias_*')):
        bias_val = float(bias_dir.name.split('_', 1)[-1].replace('v', '.'))

        # Determine tau_shift for the current bias value
        if bias_val == 0.0:
            current_tau_shift = zero_bias_shift
        else:
            current_tau_shift = non_zero_bias_shift
        tau_shifts_by_bias[bias_val] = current_tau_shift

        npz_file = list(bias_dir.glob("*.npz"))[0]
        data = np.load(npz_file, allow_pickle=True)

        raw_current_data = data["data"].item()[channel_id_str]
        fs_sch_array = data['fs_sch']
        true_voltage_amplitudes = data.get('true_voltage_at_frequencies')
        n_points = len(raw_current_data)
        full_fft_frequencies = rfftfreq(n_points, 1 / fs)

        # Store the raw data along with other necessary info
        all_raw_data_info.append((bias_val, raw_current_data, fs_sch_array,
                                  true_voltage_amplitudes, full_fft_frequencies))

    # --- Calculate the mean of the zero-bias raw current data ---
    mean_zero_bias_raw_data = 0.0 # Initialize as float zero
    found_zero_bias_raw = False
    for bias_val, raw_data, _, _, _ in all_raw_data_info:
        if bias_val == 0.0:
            mean_zero_bias_raw_data = np.mean(raw_data)
            found_zero_bias_raw = True
            print(f"Calculated mean of zero-bias raw current data: {mean_zero_bias_raw_data}")
            break
    if not found_zero_bias_raw:
        print("Warning: Zero bias raw data not found. Cannot perform zero-bias raw data mean subtraction.")

    # --- Second Pass: Process each bias point with mean subtraction and FFT shifting ---
    shifted_raw_data_by_bias = []
    admittance_data_shifted = {}

    for bias_val, raw_current_data, fs_sch_array, true_voltage_amplitudes, full_fft_frequencies in all_raw_data_info:

        # 1. Remove the mean from the real-valued raw current data (using zero-bias mean)
        if found_zero_bias_raw: # Only subtract if zero bias raw data was found
            raw_current_data_mean_subtracted = raw_current_data - mean_zero_bias_raw_data
            print(f"Bias {bias_val}: Mean of raw data before zero-bias subtraction: {np.mean(raw_current_data)}")
            print(f"Bias {bias_val}: Mean of raw data after zero-bias subtraction: {np.mean(raw_current_data_mean_subtracted)}")
        else:
            raw_current_data_mean_subtracted = raw_current_data

        # Get the tau_shift for the current bias value
        current_tau_shift = tau_shifts_by_bias[bias_val]

        # 2. FFT shift the mean-subtracted data
        raw_current_fft_mean_subtracted = rfft(raw_current_data_mean_subtracted)
        phase_factor_full_spectrum = np.exp(-2j * np.pi * full_fft_frequencies * current_tau_shift)
        raw_current_fft_shifted_full_spectrum = raw_current_fft_mean_subtracted * phase_factor_full_spectrum

        # 3. Get the inverse FFT (shifted time-domain data)
        shifted_current_data = irfft(raw_current_fft_shifted_full_spectrum)
        shifted_raw_data_by_bias.append((bias_val, shifted_current_data))

        admittance_real_for_bias = []
        admittance_imag_for_bias = []

        for j, stim_freq in enumerate(fs_sch_array):
            f_idx = np.searchsorted(full_fft_frequencies, stim_freq)
            # 4. Convert to amplitude: Use the complex FFT component from the shifted FFT data
            shifted_current_fft_at_stim_freq = raw_current_fft_shifted_full_spectrum[f_idx]
            voltage_amplitude_at_this_freq = true_voltage_amplitudes[j]

            channel_current_complex_amp = arb_to_amp(shifted_current_fft_at_stim_freq)
            impedance_circuit =  voltage_amplitude_at_this_freq /channel_current_complex_amp
            Z_RL = 1j*2*np.pi*stim_freq*74.8e-9 + 250e-6
            impedance_tes = impedance_circuit - Z_RL
            admittance_complex = 1/impedance_tes

            admittance_real_for_bias.append(np.real(admittance_complex))
            admittance_imag_for_bias.append(np.imag(admittance_complex))

        admittance_data_shifted[bias_val] = (
        fs_sch_array, np.array(admittance_real_for_bias), np.array(admittance_imag_for_bias))


    # Plot raw shifted traces with colorbar
    plt.figure(figsize=(12, 7))
    ax1 = plt.gca()
    sorted_raw_data = sorted(shifted_raw_data_by_bias, key=lambda x: x[0])
    bias_values_raw = [item[0] for item in sorted_raw_data]
    time_axis = np.arange(len(sorted_raw_data[0][1])) / fs * 1e3
    norm_raw = Normalize(vmin=min(bias_values_raw), vmax=max(bias_values_raw))
    sm_raw = plt.cm.ScalarMappable(cmap=coolwarm, norm=norm_raw)
    sm_raw.set_array([])

    for bias_val, shifted_data in sorted_raw_data:
        ax1.plot(time_axis, shifted_data, color=coolwarm(norm_raw(bias_val)), alpha=0.8)

    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel(f"Shifted Current (Channel {channel_id}) [Arb Units]")
    ax1.set_title(f"Shifted Raw Current Data (Zero Bias Shift = {zero_bias_shift:.2e} s, Non-Zero Bias Shift = {non_zero_bias_shift:.2e} s)") # Updated title
    ax1.grid(True)
    cbar_raw = plt.colorbar(sm_raw, ax=ax1, orientation='vertical', pad=0.02)
    cbar_raw.set_label("Bias Voltage (V)")
    plt.tight_layout()


    fig2, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig2.suptitle(f"Admittance Analysis (Zero Bias Shift = {zero_bias_shift:.2e} s, Nonzero sample shift = {sample_shift} samples) Across Bias Points", fontsize=16) # Updated title

    cmap_imp = plt.cm.coolwarm
    num_biases_imp = len(admittance_data_shifted)
    bias_vals_sorted_imp = sorted(admittance_data_shifted.keys())

    # Create a single ScalarMappable for the entire figure's colorbar
    norm_imp = Normalize(vmin=min(bias_vals_sorted_imp), vmax=max(bias_vals_sorted_imp))
    sm_imp = plt.cm.ScalarMappable(cmap=cmap_imp, norm=norm_imp)
    sm_imp.set_array([])

    for i, bias_val in enumerate(bias_vals_sorted_imp):

        fs_sch_array, admittance_real_array, admittance_imag_array = admittance_data_shifted[bias_val]
        color_imp = cmap_imp(norm_imp(bias_val))

        axs[0, 0].semilogx(fs_sch_array, admittance_real_array*1e3, 'o-', color=color_imp)
        axs[0, 1].semilogx(fs_sch_array, admittance_imag_array*1e3, 'o-', color=color_imp)
        magnitude_array = np.sqrt(admittance_real_array ** 2 + admittance_imag_array ** 2)
        axs[1, 0].semilogx(fs_sch_array, magnitude_array*1e3, 'o-', color=color_imp)
        axs[1, 1].plot(admittance_real_array*1e3, admittance_imag_array*1e3, 'o-', color=color_imp)

    # Set labels and titles for each subplot
    axs[0, 0].set_title("Real(Y) vs Frequency")
    axs[0, 0].set_xlabel("Frequency [Hz]")
    axs[0, 0].set_ylabel("Re(Y) [mS]")
    axs[0, 0].grid(True, which="both")

    axs[0, 1].set_title("Imag(Y) vs Frequency")
    axs[0, 1].set_xlabel("Frequency [Hz]")
    axs[0, 1].set_ylabel("Im(Y) [mS]")
    axs[0, 1].grid(True, which="both")

    axs[1, 0].set_title("Abs(Y) vs Frequency")
    axs[1, 0].set_xlabel("Frequency [Hz]")
    axs[1, 0].set_ylabel("|Y| [mS]")
    axs[1, 0].grid(True, which="both")

    axs[1, 1].set_title("Complex Admittance Plot (Im(Y) vs Re(Y))")
    axs[1, 1].set_xlabel("Re(Y) [mS]")
    axs[1, 1].set_ylabel("Im(Y) [mS]")
    axs[1, 1].axis('equal')
    axs[1, 1].grid(True)


    cbar_ax = fig2.add_axes([0.92, 0.15, 0.02, 0.7])

    cbar_imp = fig2.colorbar(sm_imp, cax=cbar_ax, orientation='vertical')
    cbar_imp.set_label("Bias Voltage (V)", rotation=270, labelpad=15)


    fig2.tight_layout(rect=[0, 0.03, 0.9, 0.95])

plt.show()