import numpy as np
import os
from scipy import constants as const

# from analysis.SCvsN_TF import STIMULUS_FREQUENCIES

MIN_SI_CURRENT = 249.5e-12 #Measured from IV curve and confirmed with Ben for the umux2Mv1.0 mux chip (32 channel)
# Magnetic flux quantum in Weber (V*s)
PHI0_FLUX_QUANTUM = const.value(u"mag. flux quantum")


R_SHUNT_ANALYSIS = 250e-6  # Ohms (Shunt resistance used in analysis de-embedding)
R_BIAS_ANALYSIS = 1952.4  # Ohms (Bias resistance used in analysis de-embedding)

K_CALIBRATION_FACTOR_RAWUNITS_PER_AMP = PHI0_FLUX_QUANTUM / MIN_SI_CURRENT

V_PEAK_ANALYSIS = 0.004 * 0.03  # Peak voltage of EACH sine component in Volts
SAMPLE_RATE_ANALYSIS = 125e3  # Hz (must match analysis)
T_SAMPLE_ANALYSIS = 1.0 / SAMPLE_RATE_ANALYSIS

# --- Parameters for the "True" Physical Model ---
R_TES_MODEL = 0.0  # Ohms
R_SHUNT_MODEL = 250e-6  # Ohms
L_MODEL = 50e-9  # Henries (50 nH)


f_low = 10
STIM_F0 = 100
f_nyquist = 62.5e3
f_mults_num = 100
potential_multipliers_float = np.logspace(0, 3, f_mults_num)

int_multipliers_with_repeats = [int(m) for m in potential_multipliers_float]
unique_int_multipliers = sorted(list(set(int_multipliers_with_repeats)))

max_allowable_multiplier = int(f_nyquist / STIM_F0)

STIM_FMULTS = [
    m for m in unique_int_multipliers if m <= max_allowable_multiplier and m >= 1
]
STIM_FMULTS.append(f_low/STIM_F0)
STIMULUS_FREQUENCIES = np.sort(np.array(STIM_FMULTS)*STIM_F0)
print(STIMULUS_FREQUENCIES)

# --- Fake Data Generation Parameters ---
DURATION_SEC = 2  # Duration of the fake signal
DC_OFFSET_RAW_UNITS = 0  # Arbitrary DC offset for the raw data
OUTPUT_NPZ_FILENAME = "test_0v0_FAKEDATA.npz"
OUTPUT_NPZ_DIRECTORY = "/data/20250507/fake_data/"  # Create this directory if it doesn't exist
CHANNEL_ID_FAKE = 0  # Which channel column to put the fake data into
DATA_ARRAY_KEY_FAKE = 'data'  # Key for the data in NPZ, must match analysis

MAX_CHANNEL_ID_IN_FAKE_NPZ = 31


def generate_stimulus_phasors(frequencies, v_peak_per_component):
    """Generates complex voltage phasors (V_peak + 0j) for each frequency."""
    phasors = {}
    for f in frequencies:
        phasors[f] = v_peak_per_component + 0j
    return phasors


# (generate_time_domain_signal_from_phasors - can be kept if used, or removed if direct freq->time reconstruction is done)

def main_generate_fake_data():
    """Generates and saves the fake TES data."""
    print(
        f"Generating fake data for model: Rtes={R_TES_MODEL} Ohm, Rshunt={R_SHUNT_MODEL * 1e6} uOhm, L={L_MODEL * 1e9} nH")
    print(f"Analysis constants assumed: Rbias={R_BIAS_ANALYSIS} Ohm, Vpeak_per_tone={V_PEAK_ANALYSIS} V")
    print(f"Calibration factor K (RawUnits/Amp) used for generation: {K_CALIBRATION_FACTOR_RAWUNITS_PER_AMP:.4e}")

    # 1. Time Vector
    n_points = int(DURATION_SEC * SAMPLE_RATE_ANALYSIS)
    times = np.arange(n_points) * T_SAMPLE_ANALYSIS
    print(
        f"Generated time vector: {n_points} points, duration {DURATION_SEC}s, sample rate {SAMPLE_RATE_ANALYSIS / 1e3} kHz")

    # 2. Generate Stimulus Voltage Phasors
    voltage_phasors_stim = generate_stimulus_phasors(STIMULUS_FREQUENCIES, V_PEAK_ANALYSIS)

    # 3. Simulate Current Response (Frequency and Time Domain)
    current_phasors_response = {}
    for freq, v_phasor in voltage_phasors_stim.items():
        omega = 2 * np.pi * freq
        z_model_at_freq = R_SHUNT_MODEL + 1j * omega * L_MODEL  # R_TES_MODEL is 0
        i_phasor = v_phasor / z_model_at_freq
        current_phasors_response[freq] = i_phasor

    # Reconstruct time-domain current
    i_response_t = np.zeros_like(times)
    for freq, i_phasor in current_phasors_response.items():
        omega = 2 * np.pi * freq
        amplitude = np.abs(i_phasor)
        phase = np.angle(i_phasor)  # Phase relative to the (real) voltage phasor
        i_response_t += amplitude * np.sin(omega * times + phase)  # Assuming stimulus sines had 0 initial phase

    print(
        f"Simulated current response. Example: I_peak for {STIMULUS_FREQUENCIES[0]} Hz = {np.abs(current_phasors_response[STIMULUS_FREQUENCIES[0]]):.2e} A")

    # 4. Convert Simulated Current to "Raw" Data Format
    # RawData = CurrentAmps * K_factor_RawUnits_per_Amp + Offset_RawUnits
    raw_phi0_single_channel_t = (i_response_t * K_CALIBRATION_FACTOR_RAWUNITS_PER_AMP) + DC_OFFSET_RAW_UNITS
    print(
        f"Converted current to raw units. Mean raw: {np.mean(raw_phi0_single_channel_t):.2e}, Std raw: {np.std(raw_phi0_single_channel_t):.2e}")

    # 5. Prepare data for NPZ (MODIFIED for multiple channels)
    num_columns_npz = MAX_CHANNEL_ID_IN_FAKE_NPZ + 1
    output_array = np.zeros((n_points, num_columns_npz))

    print(f"Populating NPZ data array with {num_columns_npz} columns (channels 0 to {MAX_CHANNEL_ID_IN_FAKE_NPZ}).")
    for channel_idx in range(num_columns_npz):
        # Writing the *same* generated timestream to all channels for this test
        output_array[:, channel_idx] = raw_phi0_single_channel_t

    # 6. Save Fake Data as NPZ
    if not os.path.exists(OUTPUT_NPZ_DIRECTORY):
        os.makedirs(OUTPUT_NPZ_DIRECTORY)
    output_filepath = os.path.join(OUTPUT_NPZ_DIRECTORY, OUTPUT_NPZ_FILENAME)
    np.savez_compressed(output_filepath, **{DATA_ARRAY_KEY_FAKE: output_array})
    print(f"Fake data saved to: {output_filepath} with shape {output_array.shape}")

    # 7. Print Expected Z_tes for verification (same as before)
    print("\n--- Expected Z_TES from Analysis Script (based on model) ---")
    print(f"Formula: ( (Rshunt_model + j*omega*L_model) - Rbias_analysis ) / (1 + Rbias_analysis/Rshunt_analysis)")
    denominator_analysis = (1 + R_BIAS_ANALYSIS / R_SHUNT_ANALYSIS)
    print(f"Denominator in analysis: {denominator_analysis:.4f}")

    for freq in STIMULUS_FREQUENCIES[:5]:  # Print for a few frequencies
        omega = 2 * np.pi * freq
        # z_trans_model = R_SHUNT_MODEL + 1j * omega * L_MODEL # This is what analysis should find for Z_trans_measured

        expected_z_tes_real = (R_SHUNT_MODEL - R_BIAS_ANALYSIS) / denominator_analysis
        expected_z_tes_imag = (omega * L_MODEL) / denominator_analysis

        print(
            f"Freq: {freq:7.1f} Hz -> Expected Z_tes_calc: Real={expected_z_tes_real * 1e3:8.4f} mOhm, Imag={expected_z_tes_imag * 1e3:8.4f} mOhm")
    print("Note: Real part should be constant. Imaginary part should be linear with frequency.")


if __name__ == "__main__":
    main_generate_fake_data()

