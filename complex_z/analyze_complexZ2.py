import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
import warnings
from scipy.fft import rfft, irfft, rfftfreq
from pathlib import Path
from matplotlib.colors import Normalize
from matplotlib.cm import coolwarm
import re
import sys

# Import necessary functions for IV curve reading
try:
    from iv_ic_analysis.iv_reader import convert_ang2_to_ites, convert_vbias_to_ibias, get_ites_from_iv_curve
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from iv_ic_analysis.iv_reader import convert_ang2_to_ites, convert_vbias_to_ibias, get_ites_from_iv_curve

# NaN ignored in processing step
warnings.filterwarnings('ignore', message='invalid value encountered in divide')
warnings.filterwarnings('ignore', message='invalid value encountered in scalar divide')

phi0 = scipy.constants.value(u"mag. flux quantum")

plt.rcParams['font.size'] = 14


# User's specific arb_to_amp definition for Complex Z data (kept as provided)
def arb_to_amp(in_val):
    min_SI = 248e-12  # This value is specific to the user's context
    min_phi0_per_amp = min_SI / phi0
    arbs_per_phi0 = 4096  # Specific for Complex Z data
    amp_per_arb = 1 / min_phi0_per_amp / arbs_per_phi0
    return in_val * amp_per_arb


# User's specific channel and directory settings


def find_npz_files(directory):
    """
    Locates all .npz files in the specified directory.
    Uses regular expressions for robust filename parsing.

    Args:
        directory (str): The path to the directory containing the .npz files.

    Returns:
        dict: A dictionary where keys are Tbase values (in K) and values
              are the corresponding file paths.  Returns an empty dictionary
              if the directory doesn't exist or if no valid files are found.
    """
    npz_files = {}
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' not found.")
        return npz_files

    pattern = os.path.join(directory, f"*.npz")
    files = glob.glob(pattern)

    tbase_regex = re.compile(r"_(\d+)mK\.npz$")

    for file_path in files:
        filename = os.path.basename(file_path)
        match = tbase_regex.search(filename)
        if match:
            try:
                tbase_mK = int(match.group(1))
                tbase = tbase_mK / 1000.0
                npz_files[tbase] = file_path
            except ValueError:
                print(f"Warning: Could not extract valid integer Tbase from '{filename}'. Skipping.")

    if not npz_files and files:
        print("Warning: No files matched the Tbase pattern (_XXXmK.npz).")
    elif not files:
        print(f"Warning: No .npz files found in '{directory}'.")

    return npz_files


def remove_offset(arr):
    """Removes offset from a 2D array by subtracting the last element of each row."""
    if not isinstance(arr, np.ndarray) or arr.ndim != 2:
        print("Warning: Input to remove_offset is not a 2D NumPy array.")
        return np.array([])
    if arr.size == 0 or arr.shape[1] <= 1:
        return arr
    last_elements = arr[:, -1]
    return arr - last_elements[:, np.newaxis]


# NEW: Rtes function (added back as it's needed for Rnormal calculation and plotting)
def Rtes(ibias, ites, Rshunt=250e-6):
    """Calculates TES resistance (Rtes)."""
    ites_safe = np.where(ites == 0, np.nan, ites)
    rtes_values = Rshunt * (ibias - ites_safe) / ites_safe
    return rtes_values


if __name__ == "__main__":
    # --- Configuration for IV curve reference ---
    # The IV curve file to use for getting the Ites reference at each bias point
    iv_filename_for_ref = '20250516_154203_iv.npz'  # Example IV file
    channel_id = 4096 + 22  # This is the channel ID from the Complex Z data
    channel_id_str = f"chan{channel_id}"
    base_dir = Path("/data/20250514/Complex_Z/20250514_122625")  # Base directory for raw data
    fs = 125000  # Sample rate for raw data

    zero_bias_shift = -0.1488233

    # Calculate the corresponding IV channel ID based on the 4096 offset
    iv_channel_id_for_offset = channel_id - 4096
    if iv_channel_id_for_offset < 0:
        print(f"Error: Calculated IV channel ID ({iv_channel_id_for_offset}) is negative. "
              f"Ensure Complex Z channel ID is >= 4096.")
        exit()

    # --- Dummy Data Creation for IV file (if it doesn't exist) ---
    # Initialize iv_npz_file_full_path outside the try block
    iv_npz_file_full_path = None
    try:
        date_str_iv = iv_filename_for_ref.split('_')[0]
        # Assign iv_npz_file_full_path here
        iv_npz_file_full_path = f'/data/{date_str_iv}/iv/{iv_filename_for_ref}'
        if not os.path.exists(iv_npz_file_full_path):
            print(f"Error: Constructed IV file path does not exist: {iv_npz_file_full_path}")
            # Fallback for testing: if the /data path doesn't exist, assume it's in current dir
            # Assign iv_npz_file_full_path here for fallback
            iv_npz_file_full_path = './' + iv_filename_for_ref
            if not os.path.exists(iv_npz_file_full_path):
                print(f"Error: IV file not found in current directory either: {iv_filename_for_ref}")
                # Create a dummy IV file for demonstration if it doesn't exist
                print("Creating dummy IV file for demonstration...")
                dummy_iv_vbias = np.linspace(-0.1, 0.1, 100)  # Example Vbias
                # Ensure dummy ang2 has enough channels for iv_channel_id_for_offset
                num_dummy_iv_channels = max(10, iv_channel_id_for_offset + 1)
                dummy_iv_ang2 = np.random.rand(100, num_dummy_iv_channels) * 1e-6
                # Make the specific channel's ang2 roughly proportional to vbias for a "TES-like" IV
                dummy_iv_ang2[:, iv_channel_id_for_offset] = dummy_iv_vbias * 1e-6 + np.random.rand(100) * 1e-7
                np.savez(iv_npz_file_full_path, vb=dummy_iv_vbias, ang2=dummy_iv_ang2)
                print(f"Dummy IV file created at: {iv_npz_file_full_path}")

    except IndexError:
        print(f"Error: Could not parse date from filename '{iv_filename_for_ref}'. Ensure format is BCEYYYYMMDD_*.npz")
        exit()  # Cannot proceed without a valid IV reference

    # Add a check here in case iv_npz_file_full_path is still None (e.g., if dummy creation failed)
    if iv_npz_file_full_path is None or not os.path.exists(iv_npz_file_full_path):
        print(f"Fatal Error: IV NPZ file path could not be determined or file does not exist: {iv_npz_file_full_path}")
        exit()

    # --- Load the full IV curve data once ---
    rbias_value = 1965.4  # Define rbias_value here for IV curve loading and Rtes calculation
    Rshunt_value = 250e-6  # Define Rshunt_value here for Rtes calculation

    vbias_iv_full, ites_iv_full = get_ites_from_iv_curve(iv_npz_file_full_path, rbias=rbias_value,
                                                         channel_id=iv_channel_id_for_offset, correct_shift=True)

    if vbias_iv_full is None or ites_iv_full is None:
        print("Error: Failed to load IV curve data. Cannot perform DC level replacement.")
        exit()  # Exit if IV data is critical and missing

    print(f"\n--- IV-Curve Loaded for DC Level Reference ---")
    print(f"Using Complex Z channel {channel_id} (IV channel {iv_channel_id_for_offset})")
    print(f"IV curve data loaded from: {iv_npz_file_full_path}")
    print(f"----------------------------------------------------------\n")

    # Find Ites minimum at indices where Rtes > 0 and calculate Rnormal
    # Calculate Ibias for the full IV curve (needed for Rtes calculation)
    ibias_iv_full = convert_vbias_to_ibias(vbias_iv_full, rbias_value)
    # Calculate Rtes for the full IV curve
    rtes_iv_full = Rtes(ibias_iv_full, ites_iv_full, Rshunt=Rshunt_value)

    # Filter for Rtes > 0
    positive_rtes_indices = np.where((rtes_iv_full > 50e-6) & np.isfinite(rtes_iv_full))[0]

    if len(positive_rtes_indices) == 0:
        print("Warning: No points found where Rtes > 0. Cannot determine stop_bias_val/Rnormal.")
        stop_bias_val = None
        Rnormal = None
    else:
        # Get the subset of IV data where Rtes > 0
        vbias_positive_rtes = vbias_iv_full[positive_rtes_indices]
        ites_positive_rtes = ites_iv_full[positive_rtes_indices]
        rtes_positive_rtes = rtes_iv_full[positive_rtes_indices]

        # Find the index of the global minimum current within this subset
        min_ites_idx_positive_rtes = np.argmin(np.abs(ites_positive_rtes))

        # The stop_bias_val is the vbias corresponding to this minimum current
        stop_bias_val = vbias_positive_rtes[min_ites_idx_positive_rtes]
        ites_at_stop_bias = ites_positive_rtes[min_ites_idx_positive_rtes]

        # Rnormal is the Rtes value at this point
        Rnormal = rtes_positive_rtes[min_ites_idx_positive_rtes]

        print(f"\n--- IV Curve Analysis ---")
        print(f"Stop Bias Voltage (Ites minimum where Rtes > 0): {stop_bias_val:.4f} V")
        print(f"TES Current at Stop Bias: {ites_at_stop_bias:.4e} A")
        print(f"Rnormal (Rtes at Ites minimum where Rtes > 0): {Rnormal:.4e} Ohms")
        print(f"-------------------------\n")

    # --- Bias Range Filtering Configuration ---
    # Automatically determine start_bias_val_filter
    # 'positive_rtes_indices' is already determined in the Rnormal calculation block.
    if len(positive_rtes_indices) > 0:
        start_bias_val_filter = vbias_iv_full[positive_rtes_indices[-4]]
        print(f"Automatically set start_bias_val_filter (first Rtes > 0 point): {start_bias_val_filter:.4f} V")
    else:
        print("Warning: No Rtes > 0 points found. Setting start_bias_val_filter to default (0.0V).")
        start_bias_val_filter = 0.0  # Fallback default

    # end_bias_val_filter is set based on stop_bias_val from the previous change.
    if stop_bias_val is not None:
        end_bias_val_filter = stop_bias_val
    else:
        print("Warning: stop_bias_val not determined, using default end_bias_val_filter of 0.3V.")
        end_bias_val_filter = 0.3  # Or set to a safe default

    # --- Main processing loop (from your original snippet, now with per-bias DC level replacement) ---
    sample_shifts_nonzerobias = [0]  # This array is used for the main loop, currently only 0
    # List to store bias values that were actually processed for plotting on IV curve
    processed_bias_vals = []

    # NEW: Store I_0 for each frequency (unnormalized FFT component)
    zero_bias_freq_currents = {}

    zero_bias_data_info = None

    for bias_dir_path in sorted(base_dir.glob('bias_*')):
        bias_val_temp = float(bias_dir_path.name.split('_', 1)[-1].replace('v', '.'))
        if bias_val_temp == 0.0:
            npz_file_raw_data_temp = list(bias_dir_path.glob("*.npz"))[0]
            data_raw_temp = np.load(npz_file_raw_data_temp, allow_pickle=True)
            raw_current_data_arb_temp = data_raw_temp["data"].item()[channel_id_str]
            raw_current_data_amps_temp = arb_to_amp(raw_current_data_arb_temp)
            fs_sch_array_temp = data_raw_temp['fs_sch']
            true_voltage_amplitudes_temp = data_raw_temp.get('true_voltage_at_frequencies')
            n_points_temp = len(raw_current_data_amps_temp)
            full_fft_frequencies_temp = rfftfreq(n_points_temp, 1 / fs)

            zero_bias_data_info = (bias_val_temp, raw_current_data_amps_temp, fs_sch_array_temp,
                                   true_voltage_amplitudes_temp, full_fft_frequencies_temp, n_points_temp)
            break

    if zero_bias_data_info is None:
        print("Error: Zero bias data (bias_0.0v) not found in raw data directories. Cannot apply new Ztes calculation.")
        sys.exit(-1)

    # Unpack zero bias data
    bias_val_0, raw_current_data_amps_0, fs_sch_array_0, true_voltage_amplitudes_0, full_fft_frequencies_0, n_points_0 = zero_bias_data_info

    # Apply DC level replacement to zero bias data (if needed, otherwise leave as is)
    current_mean_raw_data_amps_0 = np.mean(raw_current_data_amps_0)
    # Target DC for zero bias is likely 0 from IV curve at 0V, or some other expected value.
    target_dc_level_in_amps_for_this_bias_0 = np.interp(bias_val_0, vbias_iv_full, ites_iv_full)
    raw_current_data_dc_replaced_amps_0 = (
                                                      raw_current_data_amps_0 - current_mean_raw_data_amps_0) + target_dc_level_in_amps_for_this_bias_0

    # Determine tau_shift for bias_val=0.0 (needs to be determined earlier, or assumed)
    # For now, we assume zero_bias_shift is the correct tau_shift for 0.0V.
    current_tau_shift_0 = zero_bias_shift  # Direct use of global zero_bias_shift

    raw_current_fft_dc_replaced_0 = rfft(raw_current_data_dc_replaced_amps_0)
    phase_factor_full_spectrum_0 = np.exp(-2j * np.pi * full_fft_frequencies_0 * current_tau_shift_0)
    raw_current_fft_shifted_full_spectrum_0 = raw_current_fft_dc_replaced_0 * phase_factor_full_spectrum_0

    # Populate zero_bias_freq_currents with unnormalized FFT components
    for j, stim_freq in enumerate(fs_sch_array_0):
        f_idx = np.searchsorted(full_fft_frequencies_0, stim_freq)
        # if np.isclose(stim_freq / 60.0, np.round(stim_freq / 60.0)):
        #     print(f"Skipping stim_freq {stim_freq} Hz (multiple of 60 Hz).")
        #     continue
        zero_bias_freq_currents[stim_freq] = raw_current_fft_shifted_full_spectrum_0[f_idx]

    print(f"Successfully processed zero bias data for I_0 reference.")

    for sample_shift in sample_shifts_nonzerobias:
        non_zero_bias_shift = zero_bias_shift + sample_shift * 1 / fs

        # Dictionary to store tau_shift for each bias point
        tau_shifts_by_bias = {}

        # Temporary storage for raw data and other parameters for all bias points
        all_raw_data_info = []

        # --- First Pass: Load all raw data and determine tau_shifts ---
        # Filter bias directories based on the specified range
        for bias_dir_path in sorted(base_dir.glob('bias_*')):
            bias_val = float(bias_dir_path.name.split('_', 1)[-1].replace('v', '.'))

            # Apply bias value filter
            if not (start_bias_val_filter <= bias_val <= end_bias_val_filter):
                print(
                    f"Skipping bias {bias_val}v as it's outside the specified range [{start_bias_val_filter}V, {end_bias_val_filter}V].")
                continue  # Skip to the next directory if outside range

            # Add bias_val to the list of processed bias values
            processed_bias_vals.append(bias_val)

            # Determine tau_shift for the current bias value
            if bias_val == 0.0:
                current_tau_shift = zero_bias_shift
            else:
                current_tau_shift = non_zero_bias_shift
            tau_shifts_by_bias[bias_val] = current_tau_shift

            npz_file_raw_data = list(bias_dir_path.glob("*.npz"))[0]
            data_raw = np.load(npz_file_raw_data, allow_pickle=True)

            raw_current_data_arb = data_raw["data"].item()[channel_id_str]  # Original data in arb units
            # Convert raw_current_data to Amps right away
            raw_current_data_amps = arb_to_amp(raw_current_data_arb)

            fs_sch_array = data_raw['fs_sch']
            true_voltage_amplitudes = data_raw.get('true_voltage_at_frequencies')
            n_points = len(raw_current_data_amps)  # Use length of amps data
            full_fft_frequencies = rfftfreq(n_points, 1 / fs)

            # Store the data in Amps along with other necessary info
            all_raw_data_info.append((bias_val, raw_current_data_amps, fs_sch_array,
                                      true_voltage_amplitudes, full_fft_frequencies, n_points))

        # --- Second Pass: Process each bias point with IV-based mean replacement and FFT shifting ---
        shifted_raw_data_by_bias = []
        admittance_data_shifted = {}

        for bias_val, raw_current_data_amps, fs_sch_array, true_voltage_amplitudes, full_fft_frequencies, n_points in all_raw_data_info:


            target_dc_level_in_amps_for_this_bias = ites_iv_full[np.argmin(np.abs(bias_val - vbias_iv_full))]

            # Get the current mean of the raw_current_data (now already in Amps)
            current_mean_raw_data_amps = np.mean(raw_current_data_amps)

            # Replace the DC current level (all in Amps)
            raw_current_data_dc_replaced_amps = (
                                                            raw_current_data_amps - current_mean_raw_data_amps) + target_dc_level_in_amps_for_this_bias

            print(f"Processing Bias {bias_val}v:")
            print(f"  Target DC current from IV curve: {target_dc_level_in_amps_for_this_bias:.4e} A")
            print(f"  Original Mean of raw data: {current_mean_raw_data_amps:.4e} A")
            print(f"  Mean of raw data after DC level replacement: {np.mean(raw_current_data_dc_replaced_amps):.4e} A")

            # Get the tau_shift for the current bias value
            current_tau_shift = tau_shifts_by_bias[bias_val]

            # 2. FFT shift the DC-replaced data (which is in Amps)
            raw_current_fft_dc_replaced = rfft(raw_current_data_dc_replaced_amps)
            phase_factor_full_spectrum = np.exp(-2j * np.pi * full_fft_frequencies * current_tau_shift)
            raw_current_fft_shifted_full_spectrum = raw_current_fft_dc_replaced * phase_factor_full_spectrum

            # 3. Get the inverse FFT (shifted time-domain data)
            shifted_current_data_amps = irfft(raw_current_fft_shifted_full_spectrum)
            shifted_raw_data_by_bias.append((bias_val, shifted_current_data_amps))
            print(f"  Mean of IRFFT trace (shifted_current_data): {np.mean(shifted_current_data_amps):.4e} A")

            admittance_real_for_bias = []
            admittance_imag_for_bias = []

            for j, stim_freq in enumerate(fs_sch_array):
                f_idx = np.searchsorted(full_fft_frequencies, stim_freq)

                # NEW: I_bias is the unnormalized FFT component from current bias data
                I_bias = raw_current_fft_shifted_full_spectrum[f_idx]

                # NEW: Get I_0 (unnormalized FFT component) for this frequency from stored zero_bias_freq_currents
                I_0 = zero_bias_freq_currents.get(stim_freq)

                # Define Z_shunt_L (equivalent to Z_RL from previous context)
                Z_shunt_L = 1j * 2 * np.pi * stim_freq * 74.8e-9 + 250e-6

                if I_0 is None or np.abs(I_0) == 0:  # Check for None or very small/zero magnitude I_0
                    print(
                        f"Warning: I_0 not found or is zero for Bias {bias_val}V, Freq {stim_freq} Hz. Skipping Ztes calculation.")
                    impedance_tes = np.nan + 1j * np.nan  # Set to NaN if I_0 is problematic
                else:
                    Ratio = I_bias / I_0
                    # Check for Ratio being very close to zero to avoid division by zero (Z_shunt_L * (1/Ratio - 1))
                    if np.abs(Ratio) < 1e-12:  # Check if Ratio is practically zero
                        print(
                            f"Warning: Ratio (I_bias/I_0) is effectively zero for Bias {bias_val}V, Freq {stim_freq} Hz. Ztes will be infinite. Setting to NaN.")
                        impedance_tes = np.nan + 1j * np.nan
                    else:
                        impedance_tes = Z_shunt_L * (1 / Ratio - 1)

                # The channel_current_complex_amp is implicitly represented by I_bias here if needed for logging.
                # It is not used in the impedance_circuit calculation for Ztes in this new method.
                # If you need this for plotting or other uses, you can define it here as 2 * I_bias / n_points
                # (or 1/N for DC) to get its true Amp value.
                # For consistency with previous plotting and debug messages, let's keep it.
                if f_idx == 0:  # DC component
                    channel_current_complex_amp = I_bias / n_points
                else:  # Non-DC component
                    channel_current_complex_amp = 2 * I_bias / n_points

                admittance_complex = 1 / impedance_tes

                admittance_real_for_bias.append(np.real(admittance_complex))
                admittance_imag_for_bias.append(np.imag(admittance_complex))

            admittance_data_shifted[bias_val] = (
                fs_sch_array, np.array(admittance_real_for_bias), np.array(admittance_imag_for_bias))

    # --- Plot the IV curve with selected bias points highlighted ---
    plt.figure(figsize=(10, 7))
    plt.plot(vbias_iv_full * 1e3, ites_iv_full * 1e6, 'b-', label='Full IV Curve')  # Convert to mV and uA

    # Plot the processed bias points
    ites_at_processed_biases = np.array(
        [ites_iv_full[np.argmin(np.abs(p_val - vbias_iv_full))] for p_val in processed_bias_vals])
    plt.plot(np.array(processed_bias_vals) * 1e3, ites_at_processed_biases * 1e6,
             'ro', markersize=4, label='Processed Bias Points')  # Convert to mV and uA

    plt.xlabel("Bias Voltage (mV)")
    plt.ylabel("TES Current (μA)")
    plt.title(f"IV Curve for Channel {iv_channel_id_for_offset} with Processed Bias Points")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Plot Rtes vs Vbias with processed points and %Rn labels
    if Rnormal is not None:
        plt.figure(figsize=(10, 7))
        plt.plot(vbias_iv_full * 1e3, rtes_iv_full * 1e3, 'g-', label='Full Rtes Curve')  # Convert to mV and mOhm

        # Plot Rnormal line
        plt.axhline(y=Rnormal * 1e3, color='grey', linestyle='--', label=f'$R_n$ = {Rnormal * 1e3:.2f} mΩ')

        # Plot the processed bias points on the Rtes curve
        rtes_at_processed_biases = np.array(
            [rtes_iv_full[np.argmin(np.abs(p_val - vbias_iv_full))] for p_val in processed_bias_vals])
        plt.plot(np.array(processed_bias_vals) * 1e3, rtes_at_processed_biases * 1e3,
                 'ro', markersize=8, label='Processed Bias Points')  # Convert to mV and mOhm

        # Add %Rn labels to each processed point
        for i, (v_bias_point, r_tes_point) in enumerate(zip(processed_bias_vals, rtes_at_processed_biases)):
            percent_rn = (r_tes_point / Rnormal) * 100
            plt.text(v_bias_point * 1e3, r_tes_point * 1e3,
                     f'{percent_rn:.1f}%Rn',
                     fontsize=9, ha='right', va='bottom',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

        plt.xlabel("Bias Voltage (mV)")
        plt.ylabel("TES Resistance (mΩ)")
        plt.title(f"Rtes vs. Vbias for Channel {iv_channel_id_for_offset}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
    else:
        print("Skipping Rtes vs Vbias plot as Rnormal could not be determined.")

    # Plot raw shifted traces with colorbar
    plt.figure(figsize=(12, 7))
    ax1 = plt.gca()
    sorted_raw_data = sorted(shifted_raw_data_by_bias, key=lambda x: x[0])
    # Check if any data was processed before attempting to plot
    if not sorted_raw_data:
        print("\nWarning: No raw data processed within the specified bias range. Skipping raw data plots.")
    else:
        bias_values_raw = [item[0] for item in sorted_raw_data]
        time_axis = np.arange(len(sorted_raw_data[0][1])) / fs * 1e3
        norm_raw = Normalize(vmin=min(bias_values_raw), vmax=max(bias_values_raw))
        sm_raw = plt.cm.ScalarMappable(cmap=coolwarm, norm=norm_raw)
        sm_raw.set_array([])

        for bias_val, shifted_data_amps in sorted_raw_data:
            # Plotting in Amps
            ax1.plot(time_axis, shifted_data_amps, color=coolwarm(norm_raw(bias_val)), alpha=0.8)

        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel(f"Shifted Current (Channel {channel_id}) [A]")  # Y-axis label changed to Amps
        ax1.set_title(
            f"Shifted Raw Current Data (Zero Bias Shift = {zero_bias_shift:.2e} s, Non-Zero Bias Shift = {non_zero_bias_shift:.2e} s)")  # Updated title
        ax1.grid(True)
        cbar_raw = plt.colorbar(sm_raw, ax=ax1, orientation='vertical', pad=0.02)
        cbar_raw.set_label("Bias Voltage (V)")
        plt.tight_layout()

    fig2, axs = plt.subplots(2, 2, figsize=(16, 10))
    # Check if any data was processed before attempting to plot
    if not admittance_data_shifted:
        print("\nWarning: No admittance data processed within the specified bias range. Skipping admittance plots.")
        plt.close(fig2)  # Close the empty figure
    else:
        fig2.suptitle(
            f"Admittance Analysis (Zero Bias Shift = {zero_bias_shift:.2e} s, Nonzero sample shift = {sample_shift} samples) Across Bias Points",
            fontsize=16)  # Updated title

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

            axs[0, 0].semilogx(fs_sch_array, admittance_real_array, 'o-', color=color_imp)
            axs[0, 1].semilogx(fs_sch_array, admittance_imag_array, 'o-', color=color_imp)
            magnitude_array = np.sqrt(admittance_real_array ** 2 + admittance_imag_array ** 2)
            axs[1, 0].semilogx(fs_sch_array, magnitude_array, 'o-', color=color_imp)
            axs[1, 1].plot(admittance_real_array, admittance_imag_array, 'o-', color=color_imp)

        # Set labels and titles for each subplot
        axs[0, 0].set_title("Real(Y) vs Frequency")
        axs[0, 0].set_xlabel("Frequency [Hz]")
        axs[0, 0].set_ylabel("Re(Y) [S]")
        axs[0, 0].grid(True, which="both")

        axs[0, 1].set_title("Imag(Y) vs Frequency")
        axs[0, 1].set_xlabel("Frequency [Hz]")
        axs[0, 1].set_ylabel("Im(Y) [S]")
        axs[0, 1].grid(True, which="both")

        axs[1, 0].set_title("Abs(Y) vs Frequency")
        axs[1, 0].set_xlabel("Frequency [Hz]")
        axs[1, 0].set_ylabel("|Y| [S]")
        axs[1, 0].grid(True, which="both")

        axs[1, 1].set_title("Complex Admittance Plot (Im(Y) vs Re(Y))")
        axs[1, 1].set_xlabel("Re(Y) [S]")
        axs[1, 1].set_ylabel("Im(Y) [S]")
        axs[1, 1].axis('equal')
        axs[1, 1].grid(True)

        cbar_ax = fig2.add_axes([0.92, 0.15, 0.02, 0.7])

        cbar_imp = fig2.colorbar(sm_imp, cax=cbar_ax, orientation='vertical')
        cbar_imp.set_label("Bias Voltage (V)", rotation=270, labelpad=15)

        fig2.tight_layout(rect=[0, 0.03, 0.9, 0.95])

    plt.show()