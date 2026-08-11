import numpy as np
import matplotlib.pyplot as plt


# Frequency range (e.g., 1 Hz to 1 MHz on a log scale)
frequencies = np.logspace(0, 6, 500) # 500 points from 10^0 to 10^6 Hz
omega = 2 * np.pi * frequencies # Angular frequency

# Example circuit parameters (replace with your actual model)
R1 = 0.01 # Ohms
R2 = 250e-6  # Ohms
L = 120e-9   # 1 mH
C = 1e-6   # 1 uF

# Example Impedances (replace with your actual circuit calculation)
# Impedance of inductor: j*omega*L
Z_L = 1j * omega * L
# Impedance of capacitor: 1 / (j*omega*C)
Z_C = 1 / (1j * omega * C)

# Example: Current ratio for a simple circuit like a voltage divider
# Let's assume Current Ratio = Z2 / (Z1 + Z2)
# Example Z1 = R1 + Z_L
# Example Z2 = R2 + Z_C
Z1 = R2+ Z_L
Z2 = R1+R2 + Z_L
# Calculate the complex current ratio
current_ratio = Z1 / Z2

# --- Plotting ---
plt.figure(figsize=(10, 6))

# Plot the real part of the current ratio
plt.plot(frequencies, current_ratio.real, label='Re(Current Ratio)', color='blue')
plt.plot(frequencies, current_ratio.imag, label='Im(Current Ratio)', color='red')

# Plot the imaginary part for comparison (optional)
# plt.plot(frequencies, current_ratio.imag, label='Im(Current Ratio)', color='red', linestyle='--')

# --- Formatting the Plot ---
plt.xscale('log') # Use logarithmic scale for frequency
plt.xlabel('Frequency (Hz)')
plt.ylabel('Current Ratio')
# plt.title('Real Part of Current Ratio vs. Frequency')
plt.grid(True, which='both', linestyle='--', linewidth=0.5) # Add grid lines for both major and minor ticks
plt.legend()

# Ensure y-axis starts from 0 or slightly below if needed to show decay properly
min_real = np.min(current_ratio.real)
max_real = np.max(current_ratio.real)
# Add a small buffer to the y-axis limits
y_buffer = (max_real - min_real) * 0.05
# plt.ylim(min(0, min_real - y_buffer), max_real + y_buffer) # Ensure y=0 is visible

# Add a horizontal line at y=0 for reference
plt.axhline(0, color='black', linewidth=0.8, linestyle=':')

plt.tight_layout() # Adjust layout to prevent labels overlapping
plt.show()