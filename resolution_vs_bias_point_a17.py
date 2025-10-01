import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 12})


fwhm_vals = [0.793, 0.826, 0.796, 0.762, 0.824, 0.852, 0.925, 1.17]
fwhm_errs = [0.012,0.008, 0.026, 0.009, 0.004, 0.017, 0.032, 0.042]



bias_point_Rns = [14.6, 18.3, 19.1, 20.0, 22, 26.4, 30.7, 49.5]



plt.figure(figsize=(8, 6)) # Optional: Adjust figure size
plt.errorbar(bias_point_Rns, fwhm_vals, yerr=fwhm_errs, fmt='o', capsize=5, linestyle='--', lw=0.75  )

plt.xlabel("Bias point (% Rn)")
plt.ylabel("FWHM of single photon peak (eV)")
plt.title("Resolution vs bias point using pulsed green laser \n (Myriad dtest, pixel 6 (ch 4107),  Cooldown A17)")
# plt.xticks(bias_point_Rns)
# plt.legend()
plt.grid(True, axis='both', linestyle=':', alpha=0.7) # Optional: Add grid lines

plt.show()