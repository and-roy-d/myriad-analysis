import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import qsdumux.instruments.alyE5701B_usb as alyE5071B_usb

############################################################################################################

# User-defined parameters:
f_lo = 4.75e9                        # Start frequency of wide survey (Hz).
f_hi = 5.5e9                        # End frequency of wide survey (Hz).
filename = "/home/pcuser/Runs/Cooldown_A17/widesurvey_qtest2_robi_100mK_20250429_vfr0v0.npz"   # Name of file to save the data.

############################################################################################################

# Initialize VNA
vna = alyE5071B_usb.alyE5071B_usb()

# Take wide VNA surveys
tempdata = vna.measureSurvey(fi=f_lo/1e6,ff=f_hi/1e6,fres=0.01,power=-55,mtype='S21',averfact=4)
f_wide = tempdata[:,0]*1e6
s21_wide_55 = tempdata[:,1] + 1j*tempdata[:,2]
np.savez(filename,f_wide=f_wide,s21_wide_55=s21_wide_55)

tempdata = vna.measureSurvey(fi=f_lo/1e6,ff=f_hi/1e6,fres=0.01,power=-45,mtype='S21',averfact=2)
s21_wide_45 = tempdata[:,1] + 1j*tempdata[:,2]
np.savez(filename,f_wide=f_wide,s21_wide_55=s21_wide_55,s21_wide_45=s21_wide_45)

tempdata = vna.measureSurvey(fi=f_lo/1e6,ff=f_hi/1e6,fres=0.01,power=-35,mtype='S21',averfact=1)
s21_wide_35 = tempdata[:,1] + 1j*tempdata[:,2]
np.savez(filename,f_wide=f_wide,s21_wide_55=s21_wide_55,s21_wide_45=s21_wide_45,s21_wide_35=s21_wide_35)
