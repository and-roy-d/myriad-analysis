import numpy as np
# Scipy imports for FFT and windowing
from scipy.fft import fft, fftfreq
from scipy.signal import windows

# Try to import matplotlib for plotting, but don't fail if it's not installed.
PLOT_ENABLED = False
try:
    import matplotlib.pyplot as plt

    PLOT_ENABLED = True
except ImportError:
    print("matplotlib not found, plotting will be disabled.")
    print("To enable plotting, install matplotlib: pip install matplotlib")


def sins_superimposed_v2(fmults, f0, f_low, n_periods_of_f_low, output_rate_hz, multiplier=0.06):
    """
    Generates a waveform by superimposing multiple sine waves, including a
    specified low-frequency component, and calculates its FFT using Scipy.

    The amplitude of each sine wave is scaled by 'multiplier'.
    The total duration of the waveform is determined by the number of periods
    of the low-frequency component. High-frequency components are derived
    from f0 and fmults. All generated sine waves are summed together.
    Nyquist checks for high frequencies and final clipping are currently commented out
    as per user's previous version.

    Args:
        fmults (list of float): Multipliers for the high-frequency components.
                                Each multiplier is applied to f0.
        f0 (float): Base frequency for the high-frequency components (Hz).
        f_low (float): Frequency of the low-frequency component (Hz). This
                       component is added for phase tracking or as a baseline.
        n_periods_of_f_low (int): Number of periods of the low-frequency
                                  component to generate. This determines the
                                  total duration of the waveform.
        output_rate_hz (float): The sampling rate of the output waveform (Hz).
        multiplier (float, optional): Amplitude multiplier for each sine wave.
                                      Defaults to 0.1.

    Returns:
        tuple: (all_input_frequencies, v_signal, fft_plot_frequencies, fft_plot_magnitudes)
            - all_input_frequencies (list of float): List of all input frequencies
              included in the waveform [f_low, f_h1, f_h2, ...].
            - v_signal (numpy.ndarray): The final superimposed waveform.
            - fft_plot_frequencies (numpy.ndarray): Positive frequencies for the FFT plot (Hz).
            - fft_plot_magnitudes (numpy.ndarray): Magnitudes of the FFT components
              (corresponding to fft_plot_frequencies).

    Raises:
        ValueError: If parameters lead to Nyquist violations (for low freq) or zero points.
        AssertionError: If input types/values are invalid.
    """
    # Input validation
    assert isinstance(n_periods_of_f_low, int) and n_periods_of_f_low > 0, \
        "n_periods_of_f_low must be a positive integer."
    assert f_low > 0, "f_low must be positive."
    assert f0 > 0, "f0 must be positive."
    assert output_rate_hz > 0, "output_rate_hz must be positive."
    assert isinstance(fmults, list), "fmults must be a list."

    # Calculate total duration and number of points based on the low-frequency component
    t_duration = n_periods_of_f_low / f_low
    npts_float = t_duration * output_rate_hz
    npts = int(round(npts_float))  # Round to the nearest integer

    # Warning if rounding occurred
    if not np.isclose(npts_float, npts):
        print(
            f"Warning: npts calculation for the total waveform resulted in a non-integer value ({npts_float}). Rounded to {npts}.")

    # Check for a reasonable number of points
    if npts == 0:
        raise ValueError(
            "Calculated number of points is zero. Adjust parameters (e.g., increase n_periods_of_f_low or output_rate_hz).")
    if npts < 8:  # Arbitrary small number
        print(
            f"Warning: Total calculated points ({npts}) is very low. Waveform fidelity might be poor. Consider adjusting parameters.")

    print(
        f"Generating a single waveform with {npts} points over a duration of {t_duration * 1e3:.2f} ms ({t_duration:.3f} s).")
    print(f"Total points: {npts} ({npts / 1e6:.2f} MSa)")

    # Memory depth check (1 MSa = 2^20 samples)
    MSA_LIMIT = 1048576
    if npts > MSA_LIMIT:
        print(
            f"Warning: Total points ({npts}) exceed standard 1 MSa memory limit ({MSA_LIMIT}). This may fail if the MEM option is not installed.")

    # Time vector for the entire waveform
    t = np.arange(npts) / output_rate_hz

    # Initialize the superimposed waveform and list of frequencies
    v_superimposed = np.zeros(npts)
    all_input_frequencies = []

    # --- Low-frequency component ---
    all_input_frequencies.append(f_low)
    points_per_period_low = output_rate_hz / f_low
    if points_per_period_low < 2:  # Nyquist criterion
        raise ValueError(
            f"Low frequency {f_low:.2f} Hz violates Nyquist criterion "
            f"({points_per_period_low:.2f} pts/period). "
            f"Increase output_rate_hz or adjust f_low."
        )
    if points_per_period_low < 10:  # Practical minimum for good sine shape
        print(
            f"Warning: Low frequency {f_low:.2f} Hz has only "
            f"{points_per_period_low:.2f} points per period. "
            f"Waveform shape may be poor."
        )
    phase_low = 2 * np.pi * t * f_low
    v_superimposed += multiplier * np.sin(phase_low)

    # --- High-frequency components ---
    for f_mult in fmults:
        f_h = f_mult * f0
        if f_h <= 0:
            print(f"Warning: Skipping non-positive frequency f_h = {f_h:.2f} Hz resulting from f_mult = {f_mult}.")
            continue
        all_input_frequencies.append(f_h)
        # Nyquist/practical limit checks for high frequencies were commented out by user in their version
        # points_per_period_high = output_rate_hz / f_h
        # if points_per_period_high < 2: ...
        # if points_per_period_high < 10: ...
        phase_high = 2 * np.pi * t * f_h
        v_superimposed += multiplier * np.sin(phase_high)

    # Final signal (clipping was commented out by user in their version)
    v_signal = v_superimposed
    # v_signal = np.clip(v_superimposed, -0.99999, 0.99999)

    # --- FFT Calculation using Scipy ---
    fft_plot_frequencies = np.array([])
    fft_plot_magnitudes = np.array([])
    if npts > 0:
        window_hann = windows.hann(npts)
        wave_to_fft = v_signal * window_hann
        fft_complex = fft(wave_to_fft)
        calculated_fft_frequencies = fftfreq(npts, d=1.0 / output_rate_hz)
        num_plot_points = npts // 2
        fft_plot_frequencies = calculated_fft_frequencies[:num_plot_points]
        fft_plot_magnitudes = np.abs(fft_complex[:num_plot_points])
    else:
        print("Warning: FFT not computed as npts is zero.")

    return all_input_frequencies, v_signal, fft_plot_frequencies, fft_plot_magnitudes


if __name__ == '__main__':
    # Configure parameters for the waveform generation
    fmults_log = (np.logspace(0, 3, 25))
    f0_example = 100.0
    fmults_example = list(np.int32(fmults_log))

    print(f"Using fmults (count: {len(fmults_example)}): {fmults_example[:5]}... (first 5 shown if many)")

    f_low_example = 10.0  # Low frequency component (Hz)
    n_periods_low_example = 5  # Number of periods of the low frequency component
    output_rate_example = 2e6  # Output sampling rate (Sa/s)
    amplitude_multiplier = 0.08  # Multiplier for individual sine wave amplitudes

    # Generate waveform and FFT data
    input_frequencies, waveform, fft_freqs, fft_mags = sins_superimposed_v2(
        fmults_example,
        f0_example,
        f_low_example,
        n_periods_low_example,
        output_rate_example,
        multiplier=amplitude_multiplier
    )

    print(f"Generated waveform with {len(waveform)} points.")
    print(f"Number of input frequencies superimposed: {len(input_frequencies)}")

    if PLOT_ENABLED:
        # Plot time-domain waveform
        time_vector = np.arange(len(waveform)) / output_rate_example

        plt.figure(figsize=(12, 6))
        plt.plot(time_vector, waveform)
        plt.title('Superimposed Sine Wave')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)

        # Plot FFT spectrum (log-log)
        plt.figure(figsize=(12, 6))
        if len(fft_freqs) > 0:
            # If DC component (fft_freqs[0] == 0) exists, skip for log-log or handle appropriately
            start_index = 0
            if fft_freqs[0] == 0 and len(fft_freqs) > 1:
                start_index = 1  # Skip DC for log-log plot if it's truly 0 Hz

            if len(fft_freqs[start_index:]) > 0:  # Ensure there's something to plot
                plt.loglog(fft_freqs[start_index:], fft_mags[start_index:])
            else:
                print("Not enough frequency points to plot FFT (after skipping DC).")

        else:
            print("No FFT data to plot.")

        plt.title('FFT Spectrum (Log-Log)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (abs)')
        plt.grid(True)
        plt.show()
    else:
        print("Plotting disabled as matplotlib is not available.")
        print("If you wish to see plots, please install matplotlib (e.g., 'pip install matplotlib')")

