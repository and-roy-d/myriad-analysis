import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constants

phi0 = constants.value(u"mag. flux quantum")



def phi0_to_amp(inval):
    # It's generally better to define constants globally if they don't change
    min_SI = 248e-12
    min_phi0_per_amp = min_SI / phi0
    return inval * (1 / min_phi0_per_amp)

def load_avg_pulse(filename, pulse_arrival_sample=0):
    try:
        with np.load(filename, allow_pickle=True) as f:
            # Assuming 'array1' is the correct key for the average pulse
            avg_pulse_raw = f[array_name] / 4096.0 # Use float division
            # Baseline subtraction: ensure slice indices are valid
            start_idx = 1
            end_idx = min(baseline_samples, len(avg_pulse_raw) -1) # Avoid index out of bounds
            if start_idx < end_idx:
                 baseline = np.mean(avg_pulse_raw[start_idx:end_idx])
            else:
                 baseline = avg_pulse_raw[0] # Fallback baseline
            avg_pulse_baselined = avg_pulse_raw - baseline
            times = (np.arange(len(avg_pulse_baselined)) - pulse_arrival_sample) * sample_time
            return phi0_to_amp(avg_pulse_baselined), times
    except FileNotFoundError:
        print(f"Error: Pulse file not found at {filename}")
        return None, None
    except Exception as e:
        print(f"Error loading pulse file {filename}: {e}")
        return None, None


if __name__ == "__main__":
    # --- Configuration ---
    channels = [4107, 4123]
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size if needed
    for channel in channels:
        npz_filename = f'/data/20250226/0004/20250226_0004_chan{channel}_avgpulse_test.npz'
        array_name = 'array1'  # Name of the array inside the .npz file
        arrival_time_sample = 396  # Sample index offset
        sample_time = 8e-6  # Time interval between samples (in seconds)
        baseline_samples = 200  # Number of initial samples for baseline calculation
        # --- Plotting ---

        pulse_data, time_data = load_avg_pulse(npz_filename, arrival_time_sample)
        ax.plot(time_data, np.abs(pulse_data), label=channel)

    # Add labels and title
    ax.set_xlabel(f"Time (s)")
    ax.set_yscale("log")
    ax.set_ylabel("Current (Amp)")
    ax.set_title("Pulse Height vs. Time")
    ax.grid(True) # Add a grid for easier reading
    ax.legend()


    plt.show()
