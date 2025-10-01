import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

# plt.ion()
# plt.close("all")

# Constants
Rs = 250e-6  # Ohms
L = 50e-9  # Henries
fs = 1e6  # Sample rate (Hz)
duration = 0.5 # seconds
n_samples = int(fs * duration)
t = np.linspace(0, duration, n_samples, endpoint=False)

# Choose frequencies as unique integer multiples of 100 Hz using approximate log spacing
fundamental = 100
max_freq = 100000
logspace_freqs = np.logspace(np.log10(fundamental), np.log10(max_freq), 20)
freqs = np.unique((np.round(logspace_freqs / fundamental) * fundamental).astype(int))
print(f"Frequencies used: {freqs}")

# Generate Iin_base as a sum of sinusoids
Iin_base = np.sum([np.sin(2 * np.pi * f * t) for f in freqs], axis=0)

# Repeat Iin_base 5 times to make I_in
t_rep = np.tile(Iin_base, 5)
t_full = np.linspace(0, duration * 5, len(t_rep), endpoint=False)

# Compute V_out using frequency domain approach
I_fft = fft(t_rep)
freq_fft = fftfreq(len(t_rep), 1/fs)

# Create impedance array in frequency domain
omega_fft = 2 * np.pi * freq_fft
Z_fft = Rs + 1j * omega_fft * L

# Avoid division by zero for DC component
Z_fft[Z_fft == 0] = np.inf

V_fft = I_fft * Z_fft
V_out = np.real(ifft(V_fft))

# Plotting single period and 2-periods comparison
t_two_periods = np.linspace(0, 2 * duration, 2 * n_samples, endpoint=False)
Iin_repeat = np.tile(Iin_base, 2)
Iin_direct = np.sum([np.sin(2 * np.pi * f * t_two_periods) for f in freqs], axis=0)

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(t, Iin_base)
plt.title("Single period of I_in")
plt.xlabel("Time (s)")
plt.ylabel("Current (A)")

plt.subplot(3, 1, 2)
plt.plot(t_two_periods, Iin_repeat, label='Repeated Periods')
plt.plot(t_two_periods, Iin_direct, '--', label='Direct Computation')
plt.title("Two Periods of I_in: Repeated vs Direct Calculation")
plt.xlabel("Time (s)")
plt.ylabel("Current (A)")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, V_out[:n_samples])
plt.title("Single period of V_out")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.tight_layout()
plt.show()

# Plot FFT of I_in and V_out
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.loglog(np.abs(freq_fft[:len(freq_fft)//2]), np.abs(I_fft[:len(I_fft)//2]))
plt.scatter(freqs, np.abs(fft(t_rep)[[np.argmin(np.abs(freq_fft - f)) for f in freqs]]), color='red', label='Chosen freqs')
plt.title("FFT Magnitude of I_in")
plt.xlabel("Frequency (Hz)")
plt.ylabel("|I(f)|")
plt.legend()

plt.subplot(2, 1, 2)
plt.loglog(np.abs(freq_fft[:len(freq_fft)//2]), np.abs(V_fft[:len(V_fft)//2]))
plt.scatter(freqs, np.abs(V_fft[[np.argmin(np.abs(freq_fft - f)) for f in freqs]]), color='red', label='Chosen freqs')
plt.title("FFT Magnitude of V_out")
plt.xlabel("Frequency (Hz)")
plt.ylabel("|V(f)|")
plt.legend()
plt.tight_layout()
plt.show()

# FFT method
I_fft = fft(t_rep)
V_fft = fft(V_out)
freq_fft = fftfreq(len(t_rep), 1/fs)

Z_fft_method = []
freqs_plot = []
for f in freqs:
    idx = np.argmin(np.abs(freq_fft - f))
    Z_est = V_fft[idx] / I_fft[idx]
    Z_fft_method.append(Z_est)
    freqs_plot.append(f)

# Sin/cos demodulation method
Z_sincos_method = []
for f in freqs:
    sin_wave = np.sin(2 * np.pi * f * t_full)
    cos_wave = np.cos(2 * np.pi * f * t_full)

    V_sin = np.sum(V_out * sin_wave)
    V_cos = np.sum(V_out * cos_wave)

    I_sin = np.sum(t_rep * sin_wave)
    I_cos = np.sum(t_rep * cos_wave)

    V_complex = V_cos + 1j * V_sin
    I_complex = I_cos + 1j * I_sin
    Z = V_complex / I_complex
    Z_sincos_method.append(Z)

# Calculate theoretical Z
Z_true = [Rs + 1j * 2 * np.pi * f * L for f in freqs]

# Plot the impedance comparisons
plt.figure(figsize=(10, 6))
freqs_plot = np.array(freqs_plot)
plt.loglog(freqs_plot, np.abs(Z_fft_method), 'o-', label='FFT Method')
plt.loglog(freqs_plot, np.abs(Z_sincos_method), 's-', label='Sin/Cos Method')
plt.loglog(freqs_plot, np.abs(Z_true), 'k--', label='Theoretical')
plt.xlabel("Frequency (Hz")
plt.ylabel("|Z| (Ohms)")
plt.title("Magnitude of Impedance vs Frequency")
plt.grid(True, which="both")
plt.legend()
plt.show()