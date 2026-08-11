import numpy as np
import os
import glob
import re # Import regular expressions for parsing
from scipy.fft import fft, fftfreq
from scipy.signal import windows
# Added import for physical constants
from scipy import constants as const
import pandas as pd # Using pandas for easier output handling
import traceback # For detailed error printing
# Added import for plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm # For colormaps


# --- Constants ---

V_PEAK = 0.005*0.05


# Circuit parameters
R_SHUNT = 250e-6 # Ohms (Shunt resistance)
R_BIAS = 1952.4  # Ohms (Bias resistance)


MIN_SI_CURRENT = 249.5e-12 #Measured from IV curve and confirmed with Ben for the umux2Mv1.0 mux chip (32 channel)
# Magnetic flux quantum in Weber (V*s)
PHI0_FLUX_QUANTUM = const.value(u"mag. flux quantum")

# Pre-calculate the conversion factor denominator as specified
# Note: Units are A/Wb. The physical meaning depends on the units of the input value.
MIN_PHI0_PER_AMP_FACTOR = MIN_SI_CURRENT / PHI0_FLUX_QUANTUM

# Pre-calculate constant part of denominator for Z_tes calculation
Z_TES_DENOMINATOR = (1.0 + R_BIAS / R_SHUNT)

PLOT_FFT_EXAMPLE = True
PLOT_TIMESTREAM_EXAMPLE = True
example_plots_done_for_channel = False

# Sample time (inverse of sample rate) - User provided
T_SAMPLE = 8e-6 # seconds
SAMPLE_RATE = 1.0 / T_SAMPLE

WRAP_PERIOD=1

# Maximum allowed AC current amplitude
MAX_AC_CURRENT_AMPS = 1e-6 # 1 microampere



NPZ_DIRECTORY = '/data/20250507/fake_data'


DATA_ARRAY_KEY = 'data'


CHANNELS_TO_PROCESS = [10]
# Specify which single channel ID to plot at the end
CHANNEL_TO_PLOT = 10
ZOOM_FFT_PLOT = False

OUTPUT_CSV_FILE = 'tes_impedance_multi_channel_freq_bias_results.csv' # Updated filename



f_low = 10
STIM_F0 = 100
f_nyquist = 62.5e3
potential_multipliers_float = np.logspace(0, 3, 100)

int_multipliers_with_repeats = [int(m) for m in potential_multipliers_float]
unique_int_multipliers = sorted(list(set(int_multipliers_with_repeats)))

max_allowable_multiplier = int(f_nyquist / STIM_F0)

STIM_FMULTS = [
    m for m in unique_int_multipliers if m <= max_allowable_multiplier and m >= 1
]
STIM_FMULTS.append(f_low/STIM_F0)



STIM_OUTPUT_RATE_HZ = 5e6


def get_stimulus_frequencies():


    fs = [fmult * STIM_F0 for fmult in STIM_FMULTS]
    print(f"Regenerated {len(fs)} stimulus frequencies from {fs[0]:.1f} Hz to {fs[-1]:.1f} Hz.")
    return np.array(fs)


def get_voltage_bias_from_filename(filepath):
    """
    Extracts the POSITIVE bias voltage from the file path/name.
    Looks for patterns like '_XvY_' (e.g., '_0v5_', '_1v25_').

    Args:
        filepath (str): The full path to the NPZ file.

    Returns:
        float: The bias voltage in Volts, or np.nan if not found/parsed.
    """
    filename = os.path.basename(filepath)

    match = re.search(r'(\d+)v(\d+)', filename, re.IGNORECASE) # Removed optional '-'

    if match:
        integer_part = match.group(1) # Group index shifted
        fractional_part = match.group(2) # Group index shifted
        try:
            voltage_str = f"{integer_part}.{fractional_part}"
            voltage = float(voltage_str) # Sign is always positive
            return voltage
        except ValueError:
            # This case should be rare now due to pre-filtering, but kept for safety
            print(f"  Warning: Could not convert parsed bias voltage '{voltage_str}' to float in {filename}.")
            return np.nan # Return Not a Number if conversion fails
    else:

        return np.nan # Return Not a Number if pattern not found


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
    offset_subtracted = raw_value
    converted_value = offset_subtracted / MIN_PHI0_PER_AMP_FACTOR
    return converted_value

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


    # mean_offset = np.round((np.mean(raw_data) - np.mean(unwrapped_data)) / period) * period
    # unwrapped_data += mean_offset

    return unwrapped_data

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


def plot_fft_example(freqs, fft_complex_result, n_points, sample_rate, title):
    """
    Generates a plot of the FFT amplitude spectrum (linear scale).

    Args:
        freqs (np.ndarray): Array of positive frequencies.
        fft_complex_result (np.ndarray): Array of corresponding complex FFT results
                                        (positive frequencies only).
        n_points (int): Total number of points in the original timestream.
        sample_rate (float): The sample rate of the original data.
        title (str): Title for the plot.
    """
    # Calculate Amplitude Spectrum: (2/N) * |FFT|
    # Multiply by 1e6 to plot in microamperes for better scaling
    amplitude_spectrum_uA = (2.0 / n_points) * np.abs(fft_complex_result) * 1e6

    plt.figure(figsize=(12, 6)) # Increased width
    plt.loglog(freqs / 1000, amplitude_spectrum_uA) # Freq in kHz, Amplitude in uA
    plt.title(f'Example FFT Amplitude Spectrum - {title}')
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Amplitude (µA)') # Updated label
    stimulus_frequencies = get_stimulus_frequencies()
    for f_stim in stimulus_frequencies:
        if f_stim <= 62.5e3:
            plt.axvline(f_stim / 1000, color='red', linestyle=':', linewidth=1.2, alpha=0.7, ymin=0.05, ymax=0.95, zorder=10)

    # Determine frequency limits for zooming
    min_stim_freq = STIM_F0 * STIM_FMULTS[0]
    max_stim_freq = STIM_F0 * STIM_FMULTS[-1]
    nyquist_freq = sample_rate / 2.0

    if ZOOM_FFT_PLOT:
        # Zoom slightly around the stimulus frequency range, up to Nyquist
        plot_min_freq = max(0, min_stim_freq * 0.8)
        plot_max_freq = min(nyquist_freq, max_stim_freq * 1.2)
        plt.xlim(plot_min_freq / 1000, plot_max_freq / 1000) # Set limits in kHz
        print(f"Zooming FFT plot to {plot_min_freq:.1f} Hz - {plot_max_freq:.1f} Hz")
    else:
        plt.xlim(0, nyquist_freq / 1000) # Show up to Nyquist frequency in kHz



    plt.grid(True, which='major', linestyle='-', linewidth=0.5)
    print(f"Displaying example FFT plot: {title}")
    plt.show(block=False)

def plot_timestream_example(times, current_amps, label, num_points_to_show=5e6, axes=None):
    """
    Generates a plot of the timestream data (current in Amperes).

    Args:
        times (np.ndarray): Array of time values.
        current_amps (np.ndarray): Array of corresponding current values (Amperes).
        title (str): Title for the plot.
        num_points_to_show (int): Max number of points to plot to avoid slowness.
    """
    if axes is None:
        fig, axes = plt.figure(figsize=(12, 6))
    plot_slice = slice(0, min(len(times), num_points_to_show))  # Limit points plotted
    axes.plot(times[plot_slice] * 1000, current_amps[plot_slice] * 1e6, label=label)  # Time in ms, Current in uA
    # axes.set_title(f'Example Timestream - {title} (First {plot_slice.stop} points)')
    axes.set_xlabel('Time (ms)')
    axes.set_ylabel('Current (µA)')
    axes.grid(True)
    # print(f"Displaying example Timestream plot: {title}")
    axes.legend(loc='best')
    plt.show(block=False)


F_REF_HZ = 10.0


def calculate_admittance_and_impedance(filepath, stimulus_frequencies, bias_voltage, channel_id):
    """
    Loads data for a specific channel from an NPZ file, converts raw timestream
    to Amperes, performs demodulation, applies phase correction using a
    reference frequency (F_REF_HZ), and calculates Admittance (Y) and TES dynamic
    impedance (Z_tes) for each frequency.
    Optionally plots FFT and timestream examples.

    Args:
        filepath (str): Path to the NPZ file.
        stimulus_frequencies (np.ndarray): Array of frequencies (Hz) expected
                                           in the stimulus.
        bias_voltage (float): The pre-parsed bias voltage for this file.
        channel_id (int): The channel index to extract data from.

    Returns:
        list: A list of dictionaries, where each dictionary contains results
              for one frequency component found. Returns empty list on error.
    """
    global example_plots_done_for_channel
    results_for_file_channel = []
    filename_short = os.path.basename(filepath)
    try:
        with np.load(filepath, allow_pickle=True) as f:
            if DATA_ARRAY_KEY not in f:
                print(f"Warning: Key '{DATA_ARRAY_KEY}' not found in {filename_short}. Skipping.")
                return []
            dd = f[DATA_ARRAY_KEY]
            if dd.ndim < 2 or dd.shape[1] <= channel_id:
                print(f"Warning: Data array in {filename_short} does not have expected shape "
                      f"or channel index {channel_id} is out of bounds (shape={dd.shape}). Skipping channel.")
                return []
            raw_phi0_data = dd[:, channel_id]

        current_timestream_raw = unwrap_data(raw_phi0_data, period=WRAP_PERIOD)  # Ensure WRAP_PERIOD is appropriate
        current_timestream_amps = convert_raw_to_amperes(current_timestream_raw)
        # current_timestream_amps = convert_raw_to_amperes(raw_phi0_data)

        n_points = len(current_timestream_amps)
        if n_points == 0:
            print(f"Warning: Timestream data for channel {channel_id} is empty in {filename_short}. Skipping channel.")
            return []

        sample_rate = SAMPLE_RATE
        times = np.arange(n_points) * T_SAMPLE
        # duration = n_points / sample_rate # Not directly used in phase correction logic below

        # --- Demodulate all frequencies first to get raw phasors ---
        raw_phasors = {}
        for f_stim_iter in stimulus_frequencies:
            phasor = demodulate_at_frequency(times, current_timestream_amps, f_stim_iter)
            raw_phasors[f_stim_iter] = phasor

        corrected_phasors = {}
        ref_phasor_at_f_ref = raw_phasors.get(F_REF_HZ)

        if ref_phasor_at_f_ref is None or np.isnan(ref_phasor_at_f_ref):
            print(
                f"  Warning: Reference frequency {F_REF_HZ} Hz not found or NaN in demodulated data for {filename_short}, channel {channel_id}. Skipping phase correction.")
            # Fallback: use raw phasors or skip (here, we'll use raw if ref is bad)
            corrected_phasors = raw_phasors
        else:
            # raw_phase_f_ref = np.angle(ref_phasor_at_f_ref) # This is -phi_signal. We use -angle for phi_signal
            psi_f_ref = -np.angle(ref_phasor_at_f_ref)  # This is the raw measured phase psi_0

            for f_stim_iter, raw_phasor_val in raw_phasors.items():
                if np.isnan(raw_phasor_val):
                    corrected_phasors[f_stim_iter] = raw_phasor_val
                    continue

                raw_magnitude = np.abs(raw_phasor_val)
                # raw_phase_f_stim = np.angle(raw_phasor_val) # -phi_signal for f_stim
                psi_f_stim = -np.angle(raw_phasor_val)  # raw measured phase psi_k

                # Apply phase correction: psi_k_corrected = psi_k - (f_k / f_0) * psi_0
                # As per your derivation: Phase(Zk)_corrected = TruePhase(Zk) - (fk/f0)*TruePhase(Zf0)
                # This means the individual V and I phases are corrected as:
                # psi_V,k,corrected = psi_V,k - (fk/f0) * psi_V,f0
                # psi_I,k,corrected = psi_I,k - (fk/f0) * psi_I,f0
                # This correction is applied to the *signal's phase* (psi_k)

                if f_stim_iter == 0 and F_REF_HZ == 0:  # Avoid division by zero if both are DC
                    phase_correction_factor = 0  # Or handle DC appropriately if it's a target
                elif F_REF_HZ == 0:  # Avoid division by zero if reference is DC but stim is not
                    print(
                        f"  Warning: F_REF_HZ is zero, cannot calculate phase correction ratio for f_stim={f_stim_iter}. Skipping phase correction for this frequency.")
                    phase_correction_factor = 0
                else:
                    phase_correction_factor = (f_stim_iter / F_REF_HZ) * psi_f_ref

                psi_k_corrected = psi_f_stim - phase_correction_factor

                # Reconstruct the phasor with corrected phase.
                # Original demodulate_at_frequency gives Ae^(-j*phi_signal) = Ae^(j*psi_signal)
                # So, new phasor is raw_magnitude * exp(1j * (-psi_k_corrected))
                # But we defined psi_k = -angle(raw_phasor). So angle(raw_phasor) = -psi_k
                # New angle should be -psi_k_corrected
                corrected_phasors[f_stim_iter] = raw_magnitude * np.exp(-1j * psi_k_corrected)

        if (PLOT_FFT_EXAMPLE or PLOT_TIMESTREAM_EXAMPLE) and \
                channel_id == CHANNEL_TO_PLOT and not example_plots_done_for_channel:
            plot_title = f"{filename_short} - Channel {channel_id}"
            if PLOT_FFT_EXAMPLE:
                # Perform FFT on the current timestream (Amperes) for plotting
                window_fft = windows.hann(n_points)
                current_windowed_fft = current_timestream_amps * window_fft
                fft_result_plot = fft(current_windowed_fft)
                fft_freqs_plot = fftfreq(n_points, d=1 / sample_rate)
                positive_freq_indices_plot = np.where(fft_freqs_plot >= 0)[0]
                if len(positive_freq_indices_plot):
                    positive_ffts_plot = fft_result_plot[positive_freq_indices_plot]
                    positive_fft_freqs_plot = fft_freqs_plot[positive_freq_indices_plot]
                    plot_fft_example(positive_fft_freqs_plot, positive_ffts_plot, n_points, sample_rate, plot_title)
                else:
                    print(f"Warning: No positive frequencies for FFT plot in {filename_short}, Ch {channel_id}")

            if PLOT_TIMESTREAM_EXAMPLE:
                fig, ax = plt.subplots(figsize=(12, 6))  # Changed from 8,8 to 12,6
                plot_timestream_example(times, current_timestream_amps, label='Corrected Timestream', axes=ax)
            example_plots_done_for_channel = True

        Vbias_ac_phasor = V_PEAK + 0j  # Assuming V_PEAK is the peak of EACH frequency component
        if np.isclose(V_PEAK, 0):
            print(f"Warning: V_PEAK is zero, cannot calculate results for {filename_short}, channel {channel_id}.")
            return []

        # --- Analysis for each stimulus frequency using corrected phasors ---
        for f_stim in stimulus_frequencies:
            Ites_ac_phasor = corrected_phasors.get(f_stim)

            if Ites_ac_phasor is None or np.isnan(Ites_ac_phasor):
                print(f"  Warning: Corrected phasor for f_stim={f_stim} Hz is None or NaN. Skipping this frequency.")
                # Optionally append NaN results or simply continue
                results_for_file_channel.append({
                    'channel_id': channel_id,
                    'frequency_hz': f_stim,
                    'conductance_S': np.nan,
                    'susceptance_S': np.nan,
                    'impedance_real_Ohm': np.nan,
                    'impedance_imag_Ohm': np.nan,
                    'fft_freq_hz': f_stim,  # Or actual FFT bin if using FFT method
                    'bias_voltage_V': bias_voltage,
                    'raw_Ites_re': np.nan, 'raw_Ites_im': np.nan,  # For debugging
                    'corrected_Ites_re': np.nan, 'corrected_Ites_im': np.nan  # For debugging
                })
                continue

            # For debugging, store raw and corrected phasor components
            raw_phasor_val_debug = raw_phasors.get(f_stim, np.nan + 1j * np.nan)
            raw_Ites_re = np.real(raw_phasor_val_debug)
            raw_Ites_im = np.imag(raw_phasor_val_debug)
            corrected_Ites_re = np.real(Ites_ac_phasor)
            corrected_Ites_im = np.imag(Ites_ac_phasor)

            # --- Check AC Current Amplitude (using magnitude of corrected phasor) ---
            I_peak_magnitude = np.abs(Ites_ac_phasor)
            if I_peak_magnitude >= MAX_AC_CURRENT_AMPS:
                print(f"  WARNING: Calculated AC current ({I_peak_magnitude:.3e} A) "
                      f"at {f_stim:.1f} Hz for channel {channel_id} in {filename_short} "
                      f"exceeds threshold ({MAX_AC_CURRENT_AMPS:.1e} A).")

            # --- Calculate Admittance (Y) and TES Impedance (Z_tes) ---
            conductance_G = np.nan
            susceptance_B = np.nan
            impedance_real = np.nan
            impedance_imag = np.nan

            # if np.isclose(Vbias_ac_phasor, 0 + 0j):  # Should have been caught by V_PEAK check
            #     Y_complex = np.nan + 1j * np.nan
            # elif np.isclose(Ites_ac_phasor, 0 + 0j):
            #     print(
            #         f"  Warning: Ites_ac_phasor is zero at {f_stim:.1f} Hz for channel {channel_id} in {filename_short}. Y will be zero, Z will be Inf.")
            #     Y_complex = 0 + 0j  # Or handle as NaN if preferred to avoid Inf in Z
            # else:
            Y_complex = Ites_ac_phasor / Vbias_ac_phasor

            conductance_G = np.real(Y_complex)
            susceptance_B = np.imag(Y_complex)

            if np.isclose(Y_complex, 0 + 0j):
                # Avoid division by zero if Y_complex is zero
                Z_trans_measured = np.inf + 1j * np.inf  # Or some large number / NaN
            else:
                Z_trans_measured = 1.0 / Y_complex

            if np.isinf(np.real(Z_trans_measured)) or np.isinf(np.imag(Z_trans_measured)):
                Z_tes_complex = Z_trans_measured  # Propagate Inf
            else:
                Z_tes_complex = (Z_trans_measured - R_BIAS) / Z_TES_DENOMINATOR

            impedance_real = np.real(Z_tes_complex)
            impedance_imag = np.imag(Z_tes_complex)

            results_for_file_channel.append({
                'channel_id': channel_id,
                'frequency_hz': f_stim,
                'conductance_S': conductance_G,
                'susceptance_S': susceptance_B,
                'impedance_real_Ohm': impedance_real,
                'impedance_imag_Ohm': impedance_imag,
                'fft_freq_hz': f_stim,  # This was 'actual_fft_freq', now just f_stim with demod
                'bias_voltage_V': bias_voltage,
                'raw_Ites_re': raw_Ites_re, 'raw_Ites_im': raw_Ites_im,  # For debugging
                'corrected_Ites_re': corrected_Ites_re, 'corrected_Ites_im': corrected_Ites_im  # For debugging
            })

        return results_for_file_channel

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred processing channel {channel_id} in {filepath}: {e}")
        traceback.print_exc()
        return []

# --- Plotting Function ---
def plot_admittance_vs_frequency(csv_filepath, target_channel_id):
    """
    Loads admittance data from CSV, filters for a specific channel, and generates
    plots of admittance vs frequency, with different curves for each bias voltage.

    Args:
        csv_filepath (str): Path to the CSV file containing results.
        target_channel_id (int): The specific channel ID to plot.
    """
    print(f"\n--- Plotting Admittance vs Frequency for Channel {target_channel_id} ---")
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
        # Remove rows where admittance calculation might have failed (resulted in NaN)
        df_cleaned = df_channel.dropna(subset=['conductance_S', 'susceptance_S'])
        if df_cleaned.empty:
            print(f"Error: No valid admittance data points remaining for Channel {target_channel_id} after removing NaNs.")
            return
        if len(df_cleaned) < len(df_channel):
            print(f"Note: Removed {len(df_channel) - len(df_cleaned)} rows with NaN admittance values.")


        # Get unique bias voltages and sort them for this channel
        unique_biases = sorted(df_cleaned['bias_voltage_V'].unique())
        print(f"Found {len(unique_biases)} unique bias voltages for plotting Channel {target_channel_id}.")

        # --- Generate Plots ---
        fig, axes = plt.subplots(1, 3, figsize=(20, 6)) # Increased figure width
        fig.suptitle(f'Admittance vs Frequency Analysis (Channel {target_channel_id})', fontsize=16)

        # Create a colormap for bias voltages
        colors = cm.viridis(np.linspace(0, 1, len(unique_biases)))

        # Group data by bias voltage for plotting
        grouped = df_cleaned.groupby('bias_voltage_V')

        for i, (bias_v, group) in enumerate(grouped):
            # Sort each group by frequency for clean lines
            group_sorted = group.sort_values(by='frequency_hz')

            freq = group_sorted['frequency_hz']
            # Use the admittance columns
            conductance_G = group_sorted['conductance_S']
            susceptance_B = group_sorted['susceptance_S']
            impedance_real = group_sorted['impedance_real_Ohm']
            impedance_imag = group_sorted['impedance_imag_Ohm']
            label = f'{bias_v:.3f} V' # Format bias voltage for legend
            color = colors[i]

            # Plot 1: Conductance (G) vs Frequency (log scale)
            axes[0].plot(freq, np.hypot(impedance_real,impedance_imag)*1e3, marker='o', linestyle='-', label=label, color=color, markersize=4) # Plot in mS

            # Plot 2: Susceptance (B) vs Frequency (log scale)
            axes[1].plot(freq, impedance_imag*1e3, marker='o', linestyle='-', label=label, color=color, markersize=4) # Plot in mS

            # Plot 3: Admittance Plane Plot (B vs G) - Traces for each bias
            axes[2].plot(impedance_real * 1e3, impedance_imag * 1e3, marker='.', linestyle='-', label=label, color=color, markersize=5) # Plot in mS

        # --- Finalize Plots ---
        # Plot 1: G vs Frequency
        axes[0].set_xscale('log')
        axes[0].set_xlabel('Frequency (Hz)')
        axes[0].set_ylabel('Re(Z) (mOhm)')
        axes[0].set_title('Real Impedance vs. Frequency')
        axes[0].grid(True, which='both', linestyle='--', linewidth=0.5) # Grid for major and minor ticks on log scale
        axes[0].legend(title="Bias Voltage", fontsize='small', loc='best')

        # Plot 2: B vs Frequency
        axes[1].set_xscale('log')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Im(Z) (mOhm)')
        axes[1].set_title('Imaginary Impedance vs. Frequency')
        axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)
        axes[1].legend(title="Bias Voltage", fontsize='small', loc='best')

        # Plot 3: Admittance Plane
        axes[2].set_xlabel('Re(Z) (mOhm)')
        axes[2].set_ylabel('Im(Z) (mOhm)')
        axes[2].set_title('Nyquist plot')
        axes[2].grid(True)
        axes[2].axis('equal') # Ensure aspect ratio is equal
        axes[2].legend(title="Bias Voltage", fontsize='small', loc='best')


        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        # Use plt.show(block=True) here to pause execution until plots are closed
        plt.show(block=True)


    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_filepath}")
    except KeyError as e:
        print(f"Error: Column {e} not found in CSV file. Check column names.")
    except Exception as e:
        print(f"An error occurred during plotting: {e}")
        traceback.print_exc()



if __name__ == "__main__":
    print(f"Starting multi-frequency TES impedance calculation for NPZ files in: {os.path.abspath(NPZ_DIRECTORY)}")
    print(f"Processing channels: {CHANNELS_TO_PROCESS}")
    print(f"Plotting channel: {CHANNEL_TO_PLOT}")
    if CHANNEL_TO_PLOT not in CHANNELS_TO_PROCESS:
         print(f"Error: CHANNEL_TO_PLOT ({CHANNEL_TO_PLOT}) must be included in CHANNELS_TO_PROCESS.")
         exit()


    # Regenerate the list of stimulus frequencies ONCE

    stimulus_frequencies = get_stimulus_frequencies()
    if not stimulus_frequencies.size:
        print("Error: Failed to generate stimulus frequencies. Exiting.")
        exit()
    if F_REF_HZ not in stimulus_frequencies:
        if not any(np.isclose(sf, F_REF_HZ) for sf in stimulus_frequencies):
            print(
                f"Warning: Reference frequency {F_REF_HZ} Hz is not in the list of stimulus frequencies. Phase correction might not work as expected.")
            print(f"Stimulus frequencies: {stimulus_frequencies}")
    print(f"Generated {len(stimulus_frequencies)} stimulus frequencies: {stimulus_frequencies}")

    # Find ALL NPZ files initially
    all_npz_files = glob.glob(os.path.join(NPZ_DIRECTORY, '*.npz'))
    if not all_npz_files:
        print(f"Error: No NPZ files found in directory '{NPZ_DIRECTORY}'.")
        exit()
    print(f"Found {len(all_npz_files)} total NPZ files.")

    # --- Filter files based on filename pattern BEFORE processing ---
    bias_voltage_pattern = re.compile(r'_(\d+)v(\d+)', re.IGNORECASE)
    files_to_process = []
    file_bias_voltages = {} # Store parsed bias voltage to avoid parsing again
    skipped_zero_bias_count = 0
    skipped_parse_error_count = 0
    skipped_no_match_count = 0

    for filepath in all_npz_files:
        filename = os.path.basename(filepath)
        match = bias_voltage_pattern.search(filename)
        if match:
            # Try parsing the voltage now to ensure it's valid
            integer_part = match.group(1)
            fractional_part = match.group(2)
            try:
                voltage_str = f"{integer_part}.{fractional_part}"
                voltage = float(voltage_str)
                # --- Skip if bias voltage is zero ---
                # if np.isclose(voltage, 0.0):
                #     # print(f"Skipping file: Parsed bias voltage is zero: {filename}") # Optional print
                #     skipped_zero_bias_count += 1
                #     continue  # Skip to next file
                # # --- End skip zero bias ---

                files_to_process.append(filepath)
                file_bias_voltages[filepath] = voltage  # Store the parsed voltage
            except ValueError:
                print(f"Skipping file: Could not convert parsed bias voltage '{voltage_str}' in {filename}.")
                skipped_parse_error_count += 1
        else:  # Optionally print files that don't match
            # print(f"Skipping file: Does not match bias voltage pattern: {filename}")
            skipped_no_match_count += 1

    if not files_to_process:
        print(f"Error: No NPZ files matching the bias pattern '_(\d+)v(\d+)_' found in directory '{NPZ_DIRECTORY}'.")
        exit()

    print(f"Processing {len(files_to_process)} files matching the bias voltage pattern.")

    # --- Process the filtered files ---
    all_results_impedance = []  # MODIFIED: Renamed from all_results for clarity
    collected_fft_data_for_plot = []
    processed_files_count = 0
    # Reset example plot flag before processing files
    example_plots_done_for_channel = False
    # Use the pre-filtered list 'files_to_process'
    for filepath in sorted(files_to_process):
        filename_short = os.path.basename(filepath)
        bias_voltage = file_bias_voltages[filepath]
        print(f"Processing: {os.path.basename(filepath)}...")
        # Get the pre-parsed bias voltage
        try:
            with np.load(filepath, allow_pickle=True) as f:
                if DATA_ARRAY_KEY not in f:
                    print(
                        f"  Warning (FFT plot): Key '{DATA_ARRAY_KEY}' not found in {filename_short}. Skipping for this plot.")
                else:
                    dd = f[DATA_ARRAY_KEY]
                    if dd.ndim < 2 or dd.shape[1] <= CHANNEL_TO_PLOT:
                        print(f"  Warning (FFT plot): Data array in {filename_short} for Ch {CHANNEL_TO_PLOT} "
                              f"is invalid for FFT plot. Skipping.")
                    else:
                        raw_phi0_data = dd[:, CHANNEL_TO_PLOT]
                        # Assuming unwrap_data and convert_raw_to_amperes are defined globally or imported
                        current_timestream_raw = unwrap_data(raw_phi0_data, period=WRAP_PERIOD)
                        current_timestream_amps = convert_raw_to_amperes(current_timestream_raw)
                        n_points = len(current_timestream_amps)

                        if n_points > 0:
                            sample_rate = SAMPLE_RATE  # Global constant
                            # Ensure 'signal.windows' is correctly imported/aliased
                            window_fft = windows.hann(n_points)
                            current_windowed_fft = current_timestream_amps * window_fft

                            # Assuming 'fft' and 'fftfreq' from scipy.fft are imported
                            fft_complex_result = fft(current_windowed_fft)
                            fft_freqs = fftfreq(n_points, d=1 / sample_rate)

                            positive_freq_indices = np.where(fft_freqs >= 0)[0]
                            if len(positive_freq_indices) > 0:
                                positive_ffts_complex = fft_complex_result[positive_freq_indices]
                                positive_fft_freqs = fft_freqs[positive_freq_indices]

                                collected_fft_data_for_plot.append({
                                    'bias_voltage': bias_voltage,
                                    'frequencies': positive_fft_freqs,
                                    'fft_magnitudes_complex': positive_ffts_complex,
                                    'n_points': n_points
                                })
                        # else: print warning if n_points is 0 (can be handled in plot func too)
        except Exception as e:
            print(f"  Error extracting data for FFT plot from {filename_short}: {e}")
        results_for_file = []  # Collect results for all channels in this file
        # --- Loop through specified channels ---
        for channel_id in CHANNELS_TO_PROCESS:
            # Call the updated function name
            results_for_channel = calculate_admittance_and_impedance(  # Renamed function
                filepath, stimulus_frequencies, bias_voltage, channel_id
            )
            if results_for_channel:
                results_for_file.extend(results_for_channel)
        # --- End channel loop ---

        if results_for_file:
            processed_files_count += 1
            # Add filename identifier to each result dictionary
            for res_dict in results_for_file:
                res_dict['filename'] = os.path.basename(filepath)
            all_results_impedance.extend(results_for_file)
            # print(f"  -> Extracted impedance for {len(results_for_file)} data points (freq*chan).") # Less verbose
        # else: # Error messages handled inside the function

    print(f"\nSuccessfully processed {processed_files_count} files.")
    skipped_files_at_runtime = len(files_to_process) - processed_files_count
    if skipped_files_at_runtime > 0:
        print(
            f"Skipped {skipped_files_at_runtime} files during processing (check warnings above for reasons like missing keys, empty data, etc.).")
    # if collected_fft_data_for_plot:
    #     default_max_freq_plot = None
    #     if stimulus_frequencies.size > 0:
    #         max_stim_freq = np.max(stimulus_frequencies)
    #         # Cap plot frequency a bit beyond max stimulus or at Nyquist
    #         default_max_freq_plot = min(max_stim_freq * 1.5, SAMPLE_RATE / 2.0 * 0.95)
    #
    #     print(f"\n--- Plotting All FFTs with Coolwarm Colormap (Channel {CHANNEL_TO_PLOT}) ---")
    #     # Assuming plot_all_ffts_coolwarm is defined elsewhere in your script
    #     plot_all_ffts(collected_fft_data_for_plot,
    #                            max_plot_freq=default_max_freq_plot,
    #                            plot_channel_id=CHANNEL_TO_PLOT)
    # else:
    #     print("\nNo FFT data was collected to generate the coolwarm plot.")
    if all_results_impedance:
        results_df = pd.DataFrame(all_results_impedance)
        cols = ['filename', 'channel_id', 'bias_voltage_V', 'frequency_hz',
                'conductance_S', 'susceptance_S',
                'impedance_real_Ohm', 'impedance_imag_Ohm',
                'fft_freq_hz']
        # ADDED: Logic to include new debug columns in CSV output
        if 'raw_Ites_re' in results_df.columns:
            cols.extend(['raw_Ites_re', 'raw_Ites_im', 'corrected_Ites_re', 'corrected_Ites_im'])

        final_cols = [col for col in cols if col in results_df.columns]
        results_df = results_df[final_cols]

        print("\n--- Summary (First 20 rows) ---")
        print(results_df.head(20).to_string(index=False))

        try:
            results_df.to_csv(OUTPUT_CSV_FILE, index=False, float_format='%.6e')
            print(f"\nResults saved to: {OUTPUT_CSV_FILE}")

            print("\n--- Plotting Final Admittance Results ---")
            if not results_df.empty:
                # ADDED: Check for all-NaN data before attempting to plot
                if not results_df[['conductance_S', 'susceptance_S']].isnull().all().all():
                    plot_admittance_vs_frequency(OUTPUT_CSV_FILE, CHANNEL_TO_PLOT)
                else:
                    print("No valid admittance data (all NaN) available for plotting.")
            else:
                print("No data available for admittance plotting.")

        except Exception as e:
            print(f"\nError saving results to CSV or during plotting setup: {e}")
            traceback.print_exc()
    else:
        print("\nNo results were successfully calculated from the processed files.")

    print("\nScript finished.")

