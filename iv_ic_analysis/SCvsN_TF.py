import numpy as np
import os
import glob
import re # Import regular expressions for parsing

from conda.core.index import calculate_channel_urls
from scipy.fft import fft, fftfreq
from scipy.signal import windows
# Added import for physical constants
from scipy import constants as const
# Added import for curve fitting
from scipy.optimize import curve_fit
import pandas as pd # Using pandas for easier output handling
import traceback # For detailed error printing
# Added import for plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm # For colormaps
import matplotlib.gridspec as gridspec # For subplot layout



AC_EXCITATIONS_PP = {
    0.010: '0v010', # 10 mV p-p
    0.020: '0v020', # 20 mV p-p
    0.040: '0v040'  # 40 mV p-p
}
# Corresponding Peak Voltages (V_PEAK = V_PP / 2)
AC_PEAKS = {v_pp / 2.0: suffix for v_pp, suffix in AC_EXCITATIONS_PP.items()}

# Temperatures and corresponding filename parts
TEMP_NORMAL = '100mK'
TEMP_SUPER = '15mK'

# Bias voltage filename part (expecting zero bias)
BIAS_STR = 'vbias0v0'


MIN_SI_CURRENT = 248e-12 # Amperes (User provided constant)
# Magnetic flux quantum in Weber (V*s)
PHI0_FLUX_QUANTUM = const.value(u"mag. flux quantum")


MIN_PHI0_PER_AMP_FACTOR = MIN_SI_CURRENT / PHI0_FLUX_QUANTUM
if np.isclose(MIN_PHI0_PER_AMP_FACTOR, 0):
    print("Error: Conversion factor MIN_PHI0_PER_AMP_FACTOR is zero. Check constants.")
    exit()

# Sample time (inverse of sample rate) - User provided
T_SAMPLE = 8e-6 # seconds
SAMPLE_RATE = 1.0 / T_SAMPLE

WRAP_PERIOD = 1


NPZ_DIRECTORY = '/data/20250429/data/'


DATA_ARRAY_KEY = 'data'


CHANNELS_TO_PROCESS = [11]

CHANNEL_TO_PLOT = 11

PLOT_TIMESTREAM_COMPARISON = True
PLOT_FFT_COMPARISON = True

ZOOM_FFT_PLOT = True


OUTPUT_CSV_FILE = 'current_ratio_multi_channel_vac_results_unwrapped.csv'


STIM_FMULTS = [int(fmult) for fmult in np.logspace(0, 3, 25)]
STIM_F0 = 100.0 # Hz
STIMULUS_FREQUENCIES = np.unique(np.array([fmult * STIM_F0 for fmult in STIM_FMULTS]))


# --- Conversion Function ---
def convert_raw_to_amperes(raw_value):
    """
    Removes the DC offset (estimated by the median) from the raw data
    and then converts the result to Amperes based on the provided
    formula: (value - median(value)) / (min_SI / phi0).

    Args:
        raw_value (np.ndarray): The input raw data timestream array.

    Returns:
        np.ndarray: The DC-offset-removed data converted to Amperes.
                    Returns original scaled data if input is not suitable for median.
    """
    if len(raw_value) == 0:
        print("Warning: Empty array passed to convert_raw_to_amperes.")
        return np.array([]) # Return empty array

    # Calculate the median of the raw data to estimate DC offset
    try:
        dc_offset = np.median(raw_value)
    except Exception as e:
        print(f"Warning: Could not calculate median for conversion. Skipping DC offset removal. Error: {e}")
        # Fallback: just scale without removing offset
        return raw_value / MIN_PHI0_PER_AMP_FACTOR

    # Subtract the DC offset and then apply the scaling factor
    offset_subtracted = raw_value - dc_offset
    converted_value = offset_subtracted / MIN_PHI0_PER_AMP_FACTOR
    return converted_value


# --- Phase Unwrapping Function ---
def unwrap_data(raw_data, period=WRAP_PERIOD):
    """
    Unwraps data assuming it represents a phase wrapping with a given period.

    Args:
        raw_data (np.ndarray): The input raw data array.
        period (float): The value range over which the data wraps (e.g., 4096).

    Returns:
        np.ndarray: The unwrapped data in the original units.
    """
    if period <= 0:
        print("Warning: Wrap period must be positive. Skipping unwrap.")
        return raw_data

    # Scale data to +/- pi range for np.unwrap
    scaled_data = (raw_data % period) - (period / 2.0) # Center around zero
    scaled_data_rad = scaled_data * (np.pi / (period / 2.0)) # Scale to +/- pi

    # Unwrap the scaled data (in radians)
    unwrapped_rad = np.unwrap(scaled_data_rad)

    # Scale back to original units
    unwrapped_data = unwrapped_rad * ((period / 2.0) / np.pi) + (period / 2.0) # Shift back and scale

    # Need to potentially add multiples of the period back based on the original data's mean
    # This aligns the unwrapped result with the original data's approximate level
    mean_offset = np.round((np.mean(raw_data) - np.mean(unwrapped_data)) / period) * period
    unwrapped_data += mean_offset

    return unwrapped_data


# --- Function to find file pairs ---
def find_file_pairs(directory, vac_suffix, channel_id):
    """
    Finds the 15mK and 100mK file pair for a given VAC suffix and channel.
    Assumes a filename structure like: *_TEMP_vbias0v0_vac_VACSUFFIX.npz
    """
    pattern_15mk = f"*_{TEMP_SUPER}_{BIAS_STR}_vac_{vac_suffix}.npz"
    pattern_100mk = f"*_{TEMP_NORMAL}_{BIAS_STR}_vac_{vac_suffix}.npz"

    files_15mk = glob.glob(os.path.join(directory, pattern_15mk))
    files_100mk = glob.glob(os.path.join(directory, pattern_100mk))

    # Basic check: Ensure exactly one file is found for each temperature
    if len(files_15mk) == 1 and len(files_100mk) == 1:
        return files_15mk[0], files_100mk[0]
    elif len(files_15mk) == 0:
        print(f"Warning: No file found for 15mK, VAC={vac_suffix}, Channel={channel_id}. Pattern: {pattern_15mk}")
        return None, None
    elif len(files_100mk) == 0:
        print(f"Warning: No file found for 100mK, VAC={vac_suffix}, Channel={channel_id}. Pattern: {pattern_100mk}")
        return None, None
    else:
        print(f"Warning: Found multiple files for VAC={vac_suffix}, Channel={channel_id}. Skipping.")
        print(f"  15mK matches: {files_15mk}")
        print(f"  100mK matches: {files_100mk}")
        return None, None

# --- Demodulation Function ---
def demodulate_at_frequency(times, signal_amps, frequency_hz):
    """
    Extracts the complex amplitude of a signal at a specific frequency
    using sine/cosine demodulation and averaging.

    Args:
        times (np.ndarray): Time vector for the signal.
        signal_amps (np.ndarray): Timestream signal in Amperes.
        frequency_hz (float): The frequency to demodulate at.

    Returns:
        complex: The complex phasor representing the signal component
                 at frequency_hz (Peak Amplitude and Phase). Returns NaN+NaNj
                 if calculation fails.
    """
    if len(times) != len(signal_amps) or len(times) == 0:
        return np.nan + 1j*np.nan

    omega = 2 * np.pi * frequency_hz
    ref_sin = np.sin(omega * times)
    ref_cos = np.cos(omega * times)

    # Multiply and average (low-pass filter)
    real_part_raw = np.nanmean(signal_amps * ref_cos)
    imag_part_raw = np.nanmean(signal_amps * ref_sin)

    if np.isnan(real_part_raw) or np.isnan(imag_part_raw):
        return np.nan + 1j*np.nan

    # Scale to get peak amplitude components
    real_part = 2.0 * real_part_raw
    imag_part = 2.0 * imag_part_raw

    return real_part + 1j * imag_part


# --- Core Calculation Function (Modified for Current Ratio using Demodulation) ---
def calculate_current_ratio(filepath_15mk, filepath_100mk, stimulus_frequencies, channel_id, v_peak_ac):
    """
    Loads data for a specific channel from a pair of NPZ files (15mK, 100mK),
    unwraps the raw data, converts the unwrapped timestream to Amperes, performs
    demodulation at each stimulus frequency, and computes the complex
    current ratio I_15mK / I_100mK.

    Also calculates FFTs (of original and unwrapped) for optional plotting comparison.

    Args:
        filepath_15mk (str): Path to the 15mK NPZ file.
        filepath_100mk (str): Path to the 100mK NPZ file.
        stimulus_frequencies (np.ndarray): Array of frequencies (Hz) expected
                                           in the stimulus.
        channel_id (int): The channel index to extract data from.
        v_peak_ac (float): The peak AC voltage stimulus used for this pair (for context).

    Returns:
        tuple: (results_list, intermediate_data_dict or None)
               results_list: List of dictionaries containing results for each frequency.
               intermediate_data_dict: Dictionary with original/unwrapped timestreams
                                       and FFTs if channel_id matches CHANNEL_TO_PLOT,
                                       otherwise None.
    """
    results_for_pair_channel = []
    intermediate_data = None # Initialize
    filename_15mk_short = os.path.basename(filepath_15mk)
    filename_100mk_short = os.path.basename(filepath_100mk)
    is_target_plot_channel = (channel_id == CHANNEL_TO_PLOT)

    try:
        # --- Load and Process 15mK Data ---
        with np.load(filepath_15mk, allow_pickle=True) as f_15mk:
            if DATA_ARRAY_KEY not in f_15mk: return [], None
            dd_15mk = f_15mk[DATA_ARRAY_KEY]
            if dd_15mk.ndim < 2 or dd_15mk.shape[1] <= channel_id: return [], None
            raw_15mk = dd_15mk[:, channel_id]

        # --- Unwrap Raw Data ---
        raw_15mk_unwrapped = unwrap_data(raw_15mk, WRAP_PERIOD)
        if is_target_plot_channel: print(f"  Unwrapped 15mK data (Ch {channel_id}).")

        # --- Convert UNWRAPPED data to Amperes ---
        current_15mk_unwrapped = convert_raw_to_amperes(raw_15mk_unwrapped)

        n_points_15mk = len(current_15mk_unwrapped)
        if n_points_15mk == 0: return [], None
        times_15mk = np.arange(n_points_15mk) * T_SAMPLE

        # --- Load and Process 100mK Data ---
        with np.load(filepath_100mk, allow_pickle=True) as f_100mk:
            if DATA_ARRAY_KEY not in f_100mk: return [], None
            dd_100mk = f_100mk[DATA_ARRAY_KEY]
            if dd_100mk.ndim < 2 or dd_100mk.shape[1] <= channel_id: return [], None
            raw_100mk = dd_100mk[:, channel_id]

        # --- Unwrap Raw Data ---
        raw_100mk_unwrapped = unwrap_data(raw_100mk, WRAP_PERIOD)
        if is_target_plot_channel: print(f"  Unwrapped 100mK data (Ch {channel_id}).")

        # --- Convert UNWRAPPED data to Amperes ---
        current_100mk_unwrapped = convert_raw_to_amperes(raw_100mk_unwrapped)

        n_points_100mk = len(current_100mk_unwrapped)
        if n_points_100mk == 0: return [], None
        times_100mk = np.arange(n_points_100mk) * T_SAMPLE
        if n_points_100mk != n_points_15mk:
             print(f"Warning: Different n_points {filename_15mk_short} ({n_points_15mk}) vs {filename_100mk_short} ({n_points_100mk}).")


        # --- Prepare Intermediate Data for Plotting (Includes FFTs for visualization) ---
        if is_target_plot_channel:
            intermediate_data = {
                'times_15mk': times_15mk,
                'current_15mk_orig': convert_raw_to_amperes(raw_15mk), # Original (wrapped) in Amps
                'current_15mk_unwrapped': current_15mk_unwrapped, # Unwrapped in Amps
                'n_points_15mk': n_points_15mk,
                'times_100mk': times_100mk,
                'current_100mk_orig': convert_raw_to_amperes(raw_100mk), # Original (wrapped) in Amps
                'current_100mk_unwrapped': current_100mk_unwrapped, # Unwrapped in Amps
                'n_points_100mk': n_points_100mk
            }
            # Calculate FFTs of both original (wrapped) and unwrapped for comparison plot
            window_15mk = windows.hann(n_points_15mk)
            fft_unwrapped_15mk = fft(current_15mk_unwrapped * window_15mk)
            fft_orig_15mk = fft(intermediate_data['current_15mk_orig'] * window_15mk)
            fft_freqs_15mk = fftfreq(n_points_15mk, d=T_SAMPLE)
            pos_idx_15mk = np.where(fft_freqs_15mk >= 0)[0]
            intermediate_data['pos_ffts_15mk_unwrapped'] = fft_unwrapped_15mk[pos_idx_15mk]
            intermediate_data['pos_ffts_15mk_orig'] = fft_orig_15mk[pos_idx_15mk]
            intermediate_data['pos_freqs_15mk'] = fft_freqs_15mk[pos_idx_15mk]

            window_100mk = windows.hann(n_points_100mk)
            fft_unwrapped_100mk = fft(current_100mk_unwrapped * window_100mk)
            fft_orig_100mk = fft(intermediate_data['current_100mk_orig'] * window_100mk)
            fft_freqs_100mk = fftfreq(n_points_100mk, d=T_SAMPLE)
            pos_idx_100mk = np.where(fft_freqs_100mk >= 0)[0]
            intermediate_data['pos_ffts_100mk_unwrapped'] = fft_unwrapped_100mk[pos_idx_100mk]
            intermediate_data['pos_ffts_100mk_orig'] = fft_orig_100mk[pos_idx_100mk]
            intermediate_data['pos_freqs_100mk'] = fft_freqs_100mk[pos_idx_100mk]


        # --- Calculate Ratio for each stimulus frequency using DEMODULATION on UNWRAPPED data ---
        for f_stim in stimulus_frequencies:
            # Demodulate 15mK unwrapped signal
            Ites_ac_15mk = demodulate_at_frequency(times_15mk, current_15mk_unwrapped, f_stim)

            # Demodulate 100mK unwrapped signal
            Ites_ac_100mk = demodulate_at_frequency(times_100mk, current_100mk_unwrapped, f_stim)

            # Calculate Current Ratio
            Current_Ratio = np.nan + 1j*np.nan
            valid_15mk = not np.isnan(Ites_ac_15mk)
            valid_100mk = not np.isnan(Ites_ac_100mk) and not np.isclose(Ites_ac_100mk, 0+0j)

            if valid_15mk and valid_100mk:
                Current_Ratio = Ites_ac_15mk / Ites_ac_100mk
            elif valid_15mk and not valid_100mk:
                 print(f"  Warning: Demodulated Ites_ac_100mk is zero or NaN at {f_stim:.1f} Hz for channel {channel_id}. Cannot calculate ratio.")

            # Append results
            results_for_pair_channel.append({
                'channel_id': channel_id,
                'frequency_hz': f_stim,
                'vac_peak_V': v_peak_ac,
                'current_ratio_real': np.real(Current_Ratio),
                'current_ratio_imag': np.imag(Current_Ratio),
                'current_ratio_mag': np.abs(Current_Ratio),
                'current_ratio_phase_deg': np.angle(Current_Ratio, deg=True),
                'Ites_ac_15mk_real': np.real(Ites_ac_15mk),
                'Ites_ac_15mk_imag': np.imag(Ites_ac_15mk),
                'Ites_ac_100mk_real': np.real(Ites_ac_100mk),
                'Ites_ac_100mk_imag': np.imag(Ites_ac_100mk),
            })

        return results_for_pair_channel, intermediate_data

    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        return [], None
    except Exception as e:
        print(f"An unexpected error occurred processing channel {channel_id} for pair {filename_15mk_short}/{filename_100mk_short}: {e}")
        traceback.print_exc()
        return [], None


# --- FFT Comparison Plotting Function ---
def plot_fft_comparison(directory, vac_suffix, v_peak_ac, target_channel_id, intermediate_data=None):
    """
    Plots FFT amplitude spectra for 15mK and 100mK data, showing both
    original (wrapped) and unwrapped spectra if intermediate_data is provided.

    Args:
        directory (str): Base directory to search for files.
        vac_suffix (str): The VAC suffix string (e.g., '0v010').
        v_peak_ac (float): The peak AC voltage for this excitation level.
        target_channel_id (int): The specific channel ID to plot.
        intermediate_data (dict, optional): Dictionary containing original/unwrapped
                                            FFT data for plotting comparison. Defaults to None.
    """
    print(f"\n--- Plotting FFT Comparison for VAC={vac_suffix} ({v_peak_ac*2*1000:.0f} mV p-p), Channel={target_channel_id} ---")

    if intermediate_data is None:
        filepath_15mk, filepath_100mk = find_file_pairs(directory, vac_suffix, target_channel_id)
        if not filepath_15mk or not filepath_100mk:
            print("  Skipping FFT comparison plot due to missing files.")
            return
        print("  Warning: Intermediate data not provided to plot_fft_comparison. Plotting may be incomplete.")
        return


    fig, ax = plt.subplots(figsize=(12, 6))
    plot_title = f'FFT Amplitude Spectrum Comparison (Channel {target_channel_id}, {v_peak_ac*2*1000:.0f} mV p-p)'
    ax.set_title(plot_title)
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Amplitude (µA)')
    ax.set(xscale="log", yscale="log")
    ax.grid(True, which='major', linestyle='-', linewidth=0.5)

    plotted_something = False

    # Plot Unwrapped Data FFT
    if 'pos_freqs_15mk' in intermediate_data and 'pos_ffts_15mk_unwrapped' in intermediate_data: # Check new key name
        amp_spec_15mk_unwrapped = (2.0 / intermediate_data['n_points_15mk']) * np.abs(intermediate_data['pos_ffts_15mk_unwrapped']) * 1e6
        ax.plot(intermediate_data['pos_freqs_15mk'] / 1000, amp_spec_15mk_unwrapped, '-', label=f'{TEMP_SUPER} Unwrapped', color='blue', linewidth=1.5)
        plotted_something = True
    if 'pos_freqs_100mk' in intermediate_data and 'pos_ffts_100mk_unwrapped' in intermediate_data: # Check new key name
        amp_spec_100mk_unwrapped = (2.0 / intermediate_data['n_points_100mk']) * np.abs(intermediate_data['pos_ffts_100mk_unwrapped']) * 1e6
        ax.plot(intermediate_data['pos_freqs_100mk'] / 1000, amp_spec_100mk_unwrapped, '--', label=f'{TEMP_NORMAL} Unwrapped', color='red', linewidth=1.5)
        plotted_something = True

    # Plot Original (Wrapped) Data FFT if available
    if 'pos_ffts_15mk_orig' in intermediate_data:
        amp_spec_15mk_orig = (2.0 / intermediate_data['n_points_15mk']) * np.abs(intermediate_data['pos_ffts_15mk_orig']) * 1e6
        ax.plot(intermediate_data['pos_freqs_15mk'] / 1000, amp_spec_15mk_orig, ':', label=f'{TEMP_SUPER} Original', color='cyan', alpha=0.7)
        plotted_something = True
    if 'pos_ffts_100mk_orig' in intermediate_data:
        amp_spec_100mk_orig = (2.0 / intermediate_data['n_points_100mk']) * np.abs(intermediate_data['pos_ffts_100mk_orig']) * 1e6
        ax.plot(intermediate_data['pos_freqs_100mk'] / 1000, amp_spec_100mk_orig, ':', label=f'{TEMP_NORMAL} Original', color='magenta', alpha=0.7)
        plotted_something = True


    # Add vertical lines for stimulus frequencies
    if plotted_something:
        stim_freqs_for_plot = STIMULUS_FREQUENCIES
        nyquist_freq = SAMPLE_RATE / 2.0
        ymin, ymax = ax.get_ylim()
        for f_stim in stim_freqs_for_plot:
            if f_stim <= nyquist_freq:
                 ax.axvline(f_stim / 1000, color='grey', linestyle=':', linewidth=0.8, alpha=0.7, ymin=0.05, ymax=0.95)

    # Finalize plot appearance
    min_stim_freq = STIMULUS_FREQUENCIES.min()
    max_stim_freq = STIMULUS_FREQUENCIES.max()

    if ZOOM_FFT_PLOT:
        plot_min_freq = max(0, min_stim_freq * 0.8)
        plot_max_freq = min(nyquist_freq, max_stim_freq * 1.2)
        ax.set_xlim(plot_min_freq / 1000, plot_max_freq / 1000)
    else:
        ax.set_xlim(0, nyquist_freq / 1000)

    ax.legend(title="Temp / Correction")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=False)

# --- Timestream Comparison Plotting Function ---
def plot_timestream_comparison(directory, vac_suffix, v_peak_ac, target_channel_id, intermediate_data=None):
    """
    Plots original (wrapped) and unwrapped timestreams for 15mK and 100mK data
    if intermediate_data is provided.

    Args:
        directory (str): Base directory to search for files.
        vac_suffix (str): The VAC suffix string (e.g., '0v010').
        v_peak_ac (float): The peak AC voltage for this excitation level.
        target_channel_id (int): The specific channel ID to plot.
        intermediate_data (dict, optional): Dictionary containing original/unwrapped
                                            timestream data for plotting comparison. Defaults to None.
    """
    print(f"\n--- Plotting Timestream Comparison for VAC={vac_suffix} ({v_peak_ac*2*1000:.0f} mV p-p), Channel={target_channel_id} ---")

    if intermediate_data is None:
         print("  Skipping timestream comparison plot: intermediate data not available.")
         return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True) # Two subplots, shared x-axis
    fig.suptitle(f'Timestream Comparison (Channel {target_channel_id}, {v_peak_ac*2*1000:.0f} mV p-p)', fontsize=14)

    axes[0].set_title("Original (Wrapped) Data")
    axes[0].set_ylabel('Current (µA)')
    axes[0].grid(True, which='major', linestyle='-', linewidth=0.5)

    axes[1].set_title("Unwrapped Data")
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Current (µA)')
    axes[1].grid(True, which='major', linestyle='-', linewidth=0.5)

    plotted_original = False
    plotted_unwrapped = False

    # Plot 15mK data
    if 'times_15mk' in intermediate_data and 'current_15mk_orig' in intermediate_data:
        axes[0].plot(intermediate_data['times_15mk'] * 1000, intermediate_data['current_15mk_orig'] * 1e6,
                     '-', label=f'{TEMP_SUPER} Original', color='cyan', alpha=0.8)
        plotted_original = True
    if 'times_15mk' in intermediate_data and 'current_15mk_unwrapped' in intermediate_data:
        axes[1].plot(intermediate_data['times_15mk'] * 1000, intermediate_data['current_15mk_unwrapped'] * 1e6,
                 '-', label=f'{TEMP_SUPER} Unwrapped', color='blue', alpha=0.9)
        plotted_unwrapped = True


    # Plot 100mK data
    if 'times_100mk' in intermediate_data and 'current_100mk_orig' in intermediate_data:
        axes[0].plot(intermediate_data['times_100mk'] * 1000, intermediate_data['current_100mk_orig'] * 1e6,
                 '--', label=f'{TEMP_NORMAL} Original', color='magenta', alpha=0.8)
        plotted_original = True
    if 'times_100mk' in intermediate_data and 'current_100mk_unwrapped' in intermediate_data:
        axes[1].plot(intermediate_data['times_100mk'] * 1000, intermediate_data['current_100mk_unwrapped'] * 1e6,
                 '--', label=f'{TEMP_NORMAL} Unwrapped', color='red', alpha=0.9)
        plotted_unwrapped = True


    # Finalize plot appearance
    if plotted_original: axes[0].legend(title="Temp / Type", fontsize='small')
    if plotted_unwrapped: axes[1].legend(title="Temp / Type", fontsize='small')

    if plotted_original or plotted_unwrapped:
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show(block=False)
    else:
        plt.close()


# --- New Current Ratio Plotting Function (No Fitting) ---
def plot_current_ratio(csv_filepath, target_channel_id):
    """
    Loads current ratio data from CSV, filters for a specific channel,
    and generates plots of the ratio's real and imaginary parts vs frequency,
    with different curves for each AC excitation level.

    Args:
        csv_filepath (str): Path to the CSV file containing results.
        target_channel_id (int): The specific channel ID to plot.
    """
    print(f"\n--- Plotting Current Ratio (15mK/100mK) vs Frequency for Channel {target_channel_id} ---")
    try:
        # Load the data
        df = pd.read_csv(csv_filepath)
        print(f"Loaded data from {csv_filepath}")

        if df.empty:
            print("Error: CSV file is empty. No data to plot.")
            return

        # --- Filter data for the target channel ---
        channel_mask = df['channel_id'] == target_channel_id
        df_channel = df[channel_mask].copy()

        if df_channel.empty:
            print(f"Error: No data found for Channel ID {target_channel_id} in the CSV file.")
            available_channels = sorted(df['channel_id'].unique())
            print(f"Available Channel IDs in CSV: {available_channels}")
            return

        print(f"Found {len(df_channel)} data points for Channel {target_channel_id}.")

        # --- Data for plotting ---
        # Remove rows where ratio calculation might have failed (resulted in NaN)
        df_cleaned = df_channel.dropna(subset=['current_ratio_real', 'current_ratio_imag'])
        if df_cleaned.empty:
            print(f"Error: No valid Current Ratio data points remaining for Channel {target_channel_id} after removing NaNs.")
            return
        if len(df_cleaned) < len(df_channel):
            print(f"Note: Removed {len(df_channel) - len(df_cleaned)} rows with NaN Current Ratio values.")

        # Get unique AC peak voltages and sort them for this channel
        unique_vac_peaks = sorted(df_cleaned['vac_peak_V'].unique())
        print(f"Found {len(unique_vac_peaks)} unique AC excitations for plotting Channel {target_channel_id}.")

        # --- Generate Plots ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 6)) # Real and Imag plots
        fig.suptitle(f'Current Ratio (I$_{{15mK}}$ / I$_{{100mK}}$) vs Frequency (Channel {target_channel_id})', fontsize=16)

        # Create a colormap for AC excitations
        colors = cm.plasma(np.linspace(0, 1, len(unique_vac_peaks))) # Use plasma colormap

        # Group data by AC peak voltage for plotting
        grouped = df_cleaned.groupby('vac_peak_V')

        for i, (vac_peak, group) in enumerate(grouped):
            # Sort each group by frequency for clean lines
            group_sorted = group.sort_values(by='frequency_hz')

            freq = group_sorted['frequency_hz']
            # Use the current ratio columns
            ratio_real = group_sorted['current_ratio_real']
            ratio_imag = group_sorted['current_ratio_imag']
            # Convert peak voltage back to peak-to-peak for label
            vac_pp_label = f'{vac_peak * 2 * 1000:.0f} mV p-p'
            color = colors[i]

            # Plot 1: Real(Ratio) vs Frequency (log scale)
            axes[0].plot(freq, ratio_real, marker='o', linestyle='-', label=vac_pp_label, color=color, markersize=4)

            # Plot 2: Imag(Ratio) vs Frequency (log scale)
            axes[1].plot(freq, ratio_imag, marker='o', linestyle='-', label=vac_pp_label, color=color, markersize=4)


        # --- Finalize Plots ---
        # Plot 1: Real Part vs Frequency
        axes[0].set_xscale('log')
        axes[0].set_xlabel('Frequency (Hz)')
        axes[0].set_ylabel('Real(Current Ratio)')
        axes[0].set_title('Real Part vs. Frequency')
        axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)
        axes[0].legend(title="AC Excitation", fontsize='small', loc='best')

        # Plot 2: Imaginary Part vs Frequency
        axes[1].set_xscale('log')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Imag(Current Ratio)')
        axes[1].set_title('Imaginary Part vs. Frequency')
        axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)
        axes[1].legend(title="AC Excitation", fontsize='small', loc='best')


        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
        # Use plt.show(block=True) here to pause execution until plots are closed
        plt.show(block=True)


    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_filepath}")
    except KeyError as e:
        print(f"Error: Column {e} not found in CSV file. Check column names.")
    except Exception as e:
        print(f"An error occurred during plotting: {e}")
        traceback.print_exc()


# --- Main Script ---
if __name__ == "__main__":
    print(f"Starting Current Ratio (15mK/100mK) calculation for NPZ files in: {os.path.abspath(NPZ_DIRECTORY)}")
    print(f"Processing channels: {CHANNELS_TO_PROCESS}")
    print(f"Plotting channel: {CHANNEL_TO_PLOT}") # Updated print
    if CHANNEL_TO_PLOT not in CHANNELS_TO_PROCESS:
         print(f"Error: CHANNEL_TO_PLOT ({CHANNEL_TO_PLOT}) must be included in CHANNELS_TO_PROCESS.")
         exit()

    # Regenerate stimulus frequencies globally
    print(f"Regenerating stimulus frequencies...")
    # STIMULUS_FREQUENCIES = np.array([fmult * STIM_F0 for fmult in STIM_FMULTS]) # Ensure it's defined
    print(f"{STIMULUS_FREQUENCIES=}")
    if not STIMULUS_FREQUENCIES.size:
         print("Error: Failed to generate stimulus frequencies. Exiting.")
         exit()


    # --- Find files and process in pairs ---
    all_results = []
    processed_pairs_count = 0
    skipped_pairs_count = 0
    example_plot_data = {} # Dictionary to store intermediate data for plotting

    # Iterate through desired AC excitations and channels
    for v_peak_ac, vac_suffix in AC_PEAKS.items():
        print(f"\nLooking for files with VAC = {vac_suffix} ({v_peak_ac*2*1000:.0f} mV p-p)...")
        for channel_id in CHANNELS_TO_PROCESS:
            # Find the pair of files for this VAC and channel
            filepath_15mk, filepath_100mk = find_file_pairs(NPZ_DIRECTORY, vac_suffix, channel_id)

            if filepath_15mk and filepath_100mk:
                print(f"  Processing Pair for Channel {channel_id}:")
                print(f"    Super (15mK): {os.path.basename(filepath_15mk)}")
                print(f"    Normal (100mK): {os.path.basename(filepath_100mk)}")

                # Calculate the Current ratio for this pair and channel
                results_for_pair, intermediate_data_for_plot = calculate_current_ratio( # Renamed function
                    filepath_15mk, filepath_100mk, STIMULUS_FREQUENCIES, channel_id, v_peak_ac
                )

                # Store intermediate data if it's for the target channel and VAC level
                if intermediate_data_for_plot is not None:
                    example_plot_data[vac_suffix] = intermediate_data_for_plot


                if results_for_pair:
                    processed_pairs_count += 1
                    for res_dict in results_for_pair:
                        res_dict['filename_15mk'] = os.path.basename(filepath_15mk)
                        res_dict['filename_100mk'] = os.path.basename(filepath_100mk)
                    all_results.extend(results_for_pair)
                else:
                    skipped_pairs_count += 1
            else:
                skipped_pairs_count += 1


    print(f"\nSuccessfully processed data for {processed_pairs_count} channel/VAC pairs.")
    if skipped_pairs_count > 0:
         print(f"Skipped {skipped_pairs_count} channel/VAC pairs (due to missing files or errors during processing).")


    if all_results:
        # Create a Pandas DataFrame for nice formatting and CSV output
        results_df = pd.DataFrame(all_results)
        # Reorder columns for clarity - update for current ratio
        results_df = results_df[['filename_15mk', 'filename_100mk', 'channel_id', 'vac_peak_V', 'frequency_hz',
                                 'current_ratio_mag', 'current_ratio_phase_deg',
                                 'current_ratio_real', 'current_ratio_imag',
                                 'Ites_ac_15mk_real', 'Ites_ac_15mk_imag',
                                 'Ites_ac_100mk_real', 'Ites_ac_100mk_imag']]

        print("\n--- Summary (First 20 rows) ---")
        print(results_df.head(20).to_string(index=False)) # Print head without index

        # Save to CSV
        try:
            # Update output CSV filename
            results_df.to_csv(OUTPUT_CSV_FILE, index=False, float_format='%.6e')
            print(f"\nResults saved to: {OUTPUT_CSV_FILE}")

            # --- Plot Timestream Comparisons ---
            if PLOT_TIMESTREAM_COMPARISON:
                print("\n--- Plotting Timestream Comparisons ---")
                for v_peak_ac, vac_suffix in AC_PEAKS.items():
                    intermediate = example_plot_data.get(vac_suffix)
                    if intermediate:
                         plot_timestream_comparison(NPZ_DIRECTORY, vac_suffix, v_peak_ac, CHANNEL_TO_PLOT, intermediate_data=intermediate)
                    else:
                         print(f"  Skipping timestream plot for VAC={vac_suffix}, intermediate data not found.")


            # --- Plot FFT Comparisons Section ---
            if PLOT_FFT_COMPARISON:
                print("\n--- Plotting FFT Comparisons ---")
                for v_peak_ac, vac_suffix in AC_PEAKS.items():
                     intermediate = example_plot_data.get(vac_suffix)
                     if intermediate:
                         plot_fft_comparison(NPZ_DIRECTORY, vac_suffix, v_peak_ac, CHANNEL_TO_PLOT, intermediate_data=intermediate)
                     else:
                         print(f"  Skipping FFT plot for VAC={vac_suffix}, intermediate data not found.")
            # --- End FFT Comparisons Section ---

            # --- Plotting Current Ratio Results ---
            print("\n--- Plotting Current Ratio Results ---") # Updated message
            if not results_df.empty:
                 # Call the new plotting function
                 plot_current_ratio(OUTPUT_CSV_FILE, CHANNEL_TO_PLOT)
            else:
                print("No data available for plotting.")
            # --- End Plotting Section ---


            # --- Final plt.show() to display all non-blocking plots ---
            print("\nDisplaying all generated plots...")
            plt.show() # This will block until all plot windows are closed


        except Exception as e:
            print(f"\nError saving results to CSV or during plotting setup: {e}")
            traceback.print_exc()
    else:
        print("\nNo results were successfully calculated.")

    print("\nScript finished.")
