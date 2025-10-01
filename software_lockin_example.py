import numpy as np
import matplotlib.pyplot as plt

# --- Simulation Parameters ---
T_SAMPLE = 8e-6  # Sample time (from your code)
SAMPLE_RATE = 1.0 / T_SAMPLE
DURATION = 0.005 # seconds (simulate a short segment for clarity)
times = np.arange(0, DURATION, T_SAMPLE)
n_points = len(times)

# --- Signal Parameters ---
frequency_hz = 10000.0 # 10 kHz - Frequency to demodulate
signal_amplitude = 0.5 # Amplitude of the component at frequency_hz
signal_phase_rad = np.pi / 4 # Phase of the component at frequency_hz (45 degrees)
dc_offset = 0.1
noise_level = 0.05

# Create the synthetic input signal (signal_amps)
signal_component = signal_amplitude * np.cos(2 * np.pi * frequency_hz * times + signal_phase_rad)
noise = noise_level * np.random.randn(n_points)
signal_amps = dc_offset + signal_component + noise

# --- Demodulation Steps (Intermediate) ---
omega = 2 * np.pi * frequency_hz
ref_cos = np.cos(omega * times)
ref_sin = np.sin(omega * times)

# *** Intermediate Step: Multiplication ***
product_cos = signal_amps * ref_cos # Signal mixed with Cosine reference
product_sin = signal_amps * ref_sin # Signal mixed with Sine reference

# *** Final Step (for reference): Averaging ***
# This is what np.nanmean does in your function
real_part_raw = np.mean(product_cos) # Use np.mean for simplicity here
imag_part_raw = np.mean(product_sin)

# The actual amplitude components (scaled) would be 2*real_part_raw and 2*imag_part_raw
demod_real_scaled = 2.0 * real_part_raw
demod_imag_scaled = 2.0 * imag_part_raw
demod_mag = np.sqrt(demod_real_scaled**2 + demod_imag_scaled**2)
demod_phase = np.arctan2(demod_imag_scaled, demod_real_scaled)

print(f"--- Demodulation Results (for comparison) ---")
print(f"Target Amplitude : {signal_amplitude:.4f}")
print(f"Target Phase (rad): {signal_phase_rad:.4f}")
print(f"Demodulated Mag  : {demod_mag:.4f}")
print(f"Demodulated Phase: {demod_phase:.4f}")
print(f"Raw Avg (Real)   : {real_part_raw:.4f} (Expected ~ {0.5*signal_amplitude*np.cos(signal_phase_rad):.4f})")
print(f"Raw Avg (Imag)   : {imag_part_raw:.4f} (Expected ~ {0.5*signal_amplitude*np.sin(signal_phase_rad):.4f})")


# --- Plotting ---
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
fig.suptitle(f'Intermediate Demodulation Steps (f = {frequency_hz/1000:.1f} kHz)', fontsize=14)

# Plot 1: Original Signal and Reference Cosine
ax = axes[0]
ax.plot(times * 1000, signal_amps, label='Input Signal (`signal_amps`)', color='black', alpha=0.8)
ax.plot(times * 1000, ref_cos, label='Reference Cosine', color='green', alpha=0.5, linestyle='--')
ax.set_ylabel('Amplitude')
ax.grid(True, linestyle=':')
ax.legend(loc='upper right')
ax.set_title('Input Signal & Reference Cosine Wave')

# Plot 2: Product with Cosine (In-Phase Mixer Output)
ax = axes[1]
ax.plot(times * 1000, product_cos, label='Product (`signal_amps * ref_cos`)', color='blue')
# Plot the mean value (DC component) which is extracted by LPF (averaging)
ax.axhline(real_part_raw, color='red', linestyle='--', label=f'Mean = {real_part_raw:.3f}')
ax.set_ylabel('Amplitude')
ax.grid(True, linestyle=':')
ax.legend(loc='upper right')
ax.set_title('Intermediate Product (In-Phase Component)')

# Plot 3: Product with Sine (Quadrature Mixer Output)
ax = axes[2]
ax.plot(times * 1000, product_sin, label='Product (`signal_amps * ref_sin`)', color='purple')
# Plot the mean value (DC component) which is extracted by LPF (averaging)
ax.axhline(imag_part_raw, color='orange', linestyle='--', label=f'Mean = {imag_part_raw:.3f}')
ax.set_ylabel('Amplitude')
ax.set_xlabel('Time (ms)')
ax.grid(True, linestyle=':')
ax.legend(loc='upper right')
ax.set_title('Intermediate Product (Quadrature Component)')


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()