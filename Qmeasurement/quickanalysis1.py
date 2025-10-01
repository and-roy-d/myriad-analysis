import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import guessResonanceFrequenciesBen as grf
import fitresonance as fr

temp = np.load("/home/pcuser/Runs/Cooldown_A17/widesurvey_qtest2_robi_100mK_20250429_vfr0v0.npz")
# temp = np.load("widesurvey_aSi80s_20240913.npz")
f = temp["f_wide"]
s21 = temp["s21_wide_55"]


pf = np.polyfit(f,np.unwrap(np.angle(s21)),deg=1)
s21 = s21 * np.exp(1j*(-pf[0]*f + -pf[1]))
f = f[:100000]
s21 = s21[:100000]

plt.figure(111,figsize=(7,5))
plt.plot(f/1e9,20*np.log10(np.abs(s21))+1)
plt.xlim(4.9,5.4)
plt.xlabel("Frequency (GHz)")
plt.ylim(-30,+10.0)
plt.ylabel("|S21| (dB)")

bw_lo = 100e3
bw_hi = 1000e3
tau,good,f0good = grf.guess_resonance_frequencies(f,s21,bw_lo,bw_hi,threshold_diam=0.02,threshold_spacing=2.0,threshold_circlefiterr=0.1)

nres = len(good)
f0fits = np.zeros(nres)
Qcfits = np.zeros(nres)
Qifits = np.zeros(nres)
for n in range(nres):
    ftemp = f[good[n]-100:good[n]+100]
    s21temp = s21[good[n]-100:good[n]+100]  # Already removed earlier
    #s21temp = s21[good[n]-100:good[n]+100] * np.exp(+2j*np.pi*(tau*ftemp))
    try:
        f0fittemp,Qcfittemp,Qifittemp = fr.fit_resonance(ftemp,s21temp,showplot=False)
        f0fits[n] = f0fittemp
        Qcfits[n] = Qcfittemp
        Qifits[n] = Qifittemp
    except:
        print('Fitting failed on resonator {:d}'.format(n))
        f0fits[n] = 0.0
        Qcfits[n] = 0.0
        Qifits[n] = 0.0

plt.figure(222,figsize=(7,7))
plt.subplot(2,1,1)
plt.semilogy(f0fits/1e9,Qifits/1e3,"o")
plt.xlim(4.9,5.4)
plt.xlabel("Frequency (GHz)")
plt.ylim(10,100)
plt.ylabel("Qi/1000")
plt.subplot(2,1,2)
plt.hist(Qifits/1e3,bins=np.arange(0,100,2))
plt.xlim(0,100)
plt.xlabel("Qi/1000")
