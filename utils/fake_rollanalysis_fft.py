import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, irfft, rfftfreq
from scipy.optimize import curve_fit

# plt.ion()
plt.close("all")

# Constants
Rs = 10e-3  # Ohms
L = 50e-9  # Henries
fs = 1e6  # Sample rate (Hz)
duration = 0.2  # seconds
n_samples = int(fs * duration)
t = np.linspace(0, duration, n_samples, endpoint=False)

# Choose frequencies as unique integer multiples of 100 Hz using approximate log spacing
fundamental = 5
assert fundamental == 1/duration
max_freq = 62500
logspace_freqs = np.logspace(np.log10(fundamental), np.log10(max_freq), 20)
freqs = np.unique((np.round(logspace_freqs / fundamental) * fundamental).astype(int))
print(f"Frequencies used: {freqs}")

# Generate Iin_base as a sum of sinusoids
Iin_base = np.sum([np.cos(2 * np.pi * f * t) for f in freqs], axis=0)
inds_roll_orig = n_samples // 8
Iin_roll = np.roll(Iin_base, inds_roll_orig)

# Repeat Iin_base 5 times to make I_in
t_rep = np.tile(Iin_base, 5)
t_full = np.linspace(0, duration * 5, len(t_rep), endpoint=False)

# Compute V_out using frequency domain approach
I_fft = rfft(t_rep)
I_fft_roll = rfft(np.tile(Iin_roll, 5))
freq_fft = rfftfreq(len(t_rep), 1/fs)

# Create impedance array in frequency domain
omega_fft = 2 * np.pi * freq_fft
Z_fft = Rs + 1j * omega_fft * L

# Avoid division by zero for DC component
Z_fft[Z_fft == 0] = np.inf

V_fft = I_fft * Z_fft
V_fft_roll = I_fft_roll * Z_fft
V_out = np.real(irfft(V_fft))
V_out_roll = np.real(irfft(V_fft_roll))

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
# plt.show()

# Plot FFT of I_in and V_out
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.loglog(np.abs(freq_fft), np.abs(I_fft))
plt.scatter(freqs, np.abs(rfft(t_rep)[[np.argmin(np.abs(freq_fft - f)) for f in freqs]]), color='red', label='Chosen freqs')
plt.title("FFT Magnitude of I_in")
plt.xlabel("Frequency (Hz)")
plt.ylabel("|I(f)|")
plt.legend()

plt.subplot(2, 1, 2)
plt.loglog(np.abs(freq_fft), np.abs(V_fft))
plt.scatter(freqs, np.abs(V_fft[[np.argmin(np.abs(freq_fft - f)) for f in freqs]]), color='red', label='Chosen freqs')
plt.title("FFT Magnitude of V_out")
plt.xlabel("Frequency (Hz)")
plt.ylabel("|V(f)|")
plt.legend()
plt.tight_layout()
# plt.show()

# FFT method
I_fft = rfft(t_rep)
V_fft = rfft(V_out)
V_fft_roll = rfft(V_out_roll)
freq_fft = rfftfreq(len(t_rep), 1/fs)

inds = np.searchsorted(freq_fft, freqs)
Z_fft_method = V_fft[inds] / I_fft[inds]
Z_fft_roll_method = V_fft_roll[inds] / I_fft_roll[inds]

# Sin/cos demodulation method
Z_sincos_method = []
for f in freqs:
    sin_wave = np.sin(2 * np.pi * f * t_full)
    cos_wave = np.cos(2 * np.pi * f * t_full)

    V_sin = np.sum(V_out * sin_wave)
    V_cos = np.sum(V_out * cos_wave)

    V_sin_roll = np.sum(V_out_roll * sin_wave)
    V_cos_roll = np.sum(V_out_roll * cos_wave)

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
plt.loglog(freqs, np.abs(Z_fft_method), 'o-', label='FFT Method')
plt.loglog(freqs, np.abs(Z_sincos_method), 's-', label='Sin/Cos Method')
plt.loglog(freqs, np.abs(Z_true), 'k--', label='Theoretical')
plt.xlabel("Frequency (Hz")
plt.ylabel("|Z| (Ohms)")
plt.title("Magnitude of Impedance vs Frequency")
plt.grid(True, which="both")
plt.legend()
# plt.show()

phase_lowf_Iin = np.angle(I_fft[inds[0]])
phase_lowf_Vout = np.angle(V_fft[inds[0]])
phase_lowf_Vroll = np.angle(V_fft_roll[inds[0]])
phase_roll = (phase_lowf_Vroll-phase_lowf_Iin)
t_roll_from_phase = -phase_roll/(2 * np.pi * freqs[0])
t_roll = inds_roll_orig/fs
V_fft_roll_unrolled = np.exp(1j * 2 * np.pi * freq_fft * t_roll) * V_fft_roll
inds_roll_with_err = inds_roll_orig-1
t_roll_with_err = inds_roll_with_err/fs
V_fft_roll_unrolled_with_err = np.exp(1j * 2 * np.pi * freq_fft * t_roll_from_phase) * V_fft_roll


print(f"{phase_lowf_Iin=}")
print(f"{phase_lowf_Vout=}")
print(f"{phase_lowf_Vroll=}")
print(f"{phase_roll=}")
print(f"{t_roll=}")
print(f"inds roll {t_roll*fs} compared to {inds_roll_orig=}")
print(f"{t_roll_from_phase=}")

fig, axs = plt.subplots(2,2, figsize=(12, 12))
axs[0,0].semilogx(freqs, np.angle(I_fft[inds]), 'o-', label='I_in FFT')
axs[0,0].semilogx(freqs, np.angle(V_fft[inds]), 'o-', label='V_o FFT')
axs[0,0].semilogx(freqs, np.angle(V_fft_roll[inds]), 'o-', label='V_o FFT Roll')
axs[0,0].semilogx(freqs, np.angle(V_fft_roll_unrolled[inds]), 'o-', label='V unrolled FFT Roll')
axs[0,0].semilogx(freqs, np.angle(V_fft_roll_unrolled_with_err[inds]), 'o-', label='V unrolled FFT Roll with err')

# axs[0,0].semilogx(freqs, np.angle(I_fft[inds]), 'o-', label='I FFT')
# axs[0,0].semilogx(freqs, np.angle(I_fft_roll[inds]), 'o-', label='I FFT Roll')
axs[0,0].set_xlabel("Frequency (Hz)")
axs[0,0].set_ylabel("Phase (radians)")
axs[0,0].legend()
axs[0,1].semilogx(freqs, np.abs(Z_fft_method), 'o-', label='FFT')
axs[0,1].semilogx(freqs, np.abs(Z_fft_roll_method), 'o-', label='FFT Roll')
axs[0,1].set_xlabel("Frequency (Hz)")
axs[0,1].set_ylabel("Magnitude (Ohms)")
axs[0,1].legend()
axs[1,0].plot(t, V_out[:len(t)], label='normal')
axs[1,0].plot(t, V_out_roll[:len(t)], label='roll')
axs[1,0].set_xlabel("time (s)")
axs[1,0].set_ylabel("V out (V)")
axs[1,0].legend()
# plt.pause(30)
plt.show()











