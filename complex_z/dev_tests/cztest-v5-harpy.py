from math import pi
import sys, time
import qsghw.core.helpers as corehelpers
import qsghw.core.interfaces as interfaces
import qsghw.core.packetparser as packetparser
import qsghw.fpgaip.helpers as fpgaiphelpers
import datetime
import os
import numpy as np
import pylab as pl
from scipy.signal import argrelextrema

import scipy.optimize
pl.ion()

#
# Configuration
#
import argparse
cfg = argparse.Namespace()

cfg.data_dir = '/data'
date_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d")
time_str = datetime.datetime.fromtimestamp(time.time()).strftime("%H%M%S")
cfg.date_time_str = "%s_%s" % (date_str, time_str)
cfg.data_dir_today = "%s/%s" % (cfg.data_dir, date_str)
os.makedirs(cfg.data_dir_today, exist_ok=True)
cfg.cz_dir = "%s/cz" % (cfg.data_dir_today,)
os.makedirs(cfg.cz_dir, exist_ok=True)


(regaccess, debug, verbose) = (False, False, True)

(ctrlurl, dataurl, modampl) = ('udp://10.0.15.16#fset3', 'udp://10.0.15.16', float(sys.argv[1]))
#modampl is p-p V out of the abaco 
print(ctrlurl)
(ctrlif, dataif, eids) = corehelpers.open(ctrlurl, dataurl, opendata=True, quiet=True) #verbose=False)
print("cmd =", " ".join(sys.argv))

#useful definitions
def func_1pole(f,A,f1):
    return A/(1+1j*f/f1)

def cos_func(t, A, f, phi, offset):
    return A*np.cos(2*np.pi*f*t + phi) + offset

def fit_cos(ang, t, f0):
    A0 = (np.max(ang)-np.min(ang))/2
    #phi0 = 2*np.pi*first_max_t/T0
    phi0 = np.pi/4
    offset0 = (np.max(ang)+np.min(ang))/2
    print(phi0)
    print('fit_cos')
    def func_fit(t, A, phi, offset):
        return cos_func(t, A, f0, phi, offset)

    out = scipy.optimize.curve_fit(func_fit, t, ang, [A0, phi0, offset0])
    fit_t = np.arange(10*len(ang))/fr/10
    fit = func_fit(fit_t, *out[0])
    return out, t, fit_t, fit

def fit_cos_fl(ang, t, f0):
    A0 = (np.max(ang)-np.min(ang))/2
    #phi0 = 2*np.pi*first_max_t/T0
    phi0 = np.pi/4
    offset0 = (np.max(ang)+np.min(ang))/2
    print(phi0)
    print('fit_cos_fl')
    def func_fit(t, A, phi, offset, f):
        return cos_func(t, A, f, phi, offset)

    out = scipy.optimize.curve_fit(func_fit, t, ang, [A0, phi0, offset0, f0])
    fit_t = np.arange(10*len(ang))/fr/10
    fit = func_fit(fit_t, *out[0])
    return out, t, fit_t, fit


def fit_cos_dtb(ang, t, sync_idx, f0):
    print('fit_cos_dtb', f0, len(ang), sync_idx[0][0], 10*int(fr/f0))
    idx = np.arange(sync_idx[0][0], sync_idx[0][0] + 10*int(fr/f0))
    if len(ang) < idx[-1]:
        raise Exception("Not enough periods")
    mix_cos = np.cos(2*pi*f0*(t[idx] - t[idx[0]]))
    mix_sin = np.sin(2*pi*f0*(t[idx] - t[idx[0]]))
    sum_cos = np.sum(mix_cos*ang[idx])
    sum_sin = np.sum(mix_sin*ang[idx])
    phase_dtb = np.arctan2(-sum_sin, sum_cos)
    amp_dtb = np.sqrt(sum_cos**2 + sum_sin**2)/len(idx)*2
    offset = np.mean(ang)

    fit_t = np.arange(10*len(ang))/fr/10
    fit = amp_dtb * np.cos(2*pi*f0*(fit_t - t[idx[0]]) + phase_dtb) + offset
    out = ([amp_dtb, phase_dtb, offset, f0], [])
    return out, t, fit_t, fit, idx
        

fr = 1e6/256/8*1e3

#frequencies to measure
modflist = np.logspace(1, 5, num=45)
#modflist = np.array([10.0, 50.0])
modflist = fr/2**np.arange(2, 16)[::-1]
modflist = fr/np.array([2**15, 2**13, 2**11, 2**9, 2**7, 2**6, 2**5, 2**4, 2**3, 2**2])

freq_start = 10*fr/2**15
freq_end = fr/2**2
freq_end = fr/2**1*0.9
#modflist = np.logspace(np.log10(freq_start), np.log10(freq_end), 50)
modflist = np.logspace(np.log10(freq_start), np.log10(freq_end), 10)

#modampl_fact = [10, 10, 10, 10, 5, 1, 1, 1, 1, 1]
modampl_fact = np.ones_like(modflist)
modampl_fact[modflist > 10e3] = 100

pltlist = np.arange(len(modflist)) #indices of frequencies to plot
#pltlist = [0, 1, 48, 49] #indices of frequencies to plot
#pltlist = [0, 1, 8, 9] #indices of frequencies to plot
#pltlist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] #indices of frequencies to plot
plotchan = 0 #channel to plot for debugging
abaco_atten = 1 #attenuation on theoutput of abaco
sine_amp_out = modampl/abaco_atten*1000 #the actual sine wave voltage
t_min = 10*1/modflist
t_samp = 1/fr
npack_min = np.ceil(t_min/t_samp)*8192  #minimum number of packets to grab per frequency point
#npack_min_hf = np.ceil(1/32/1e3/t_samp)*8192*8
npack_min_hf = np.ceil(t_min/t_samp)*8192*10
date_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d")
time_str = datetime.datetime.fromtimestamp(time.time()).strftime("%H%M%S")
date_time_str = "%s_%s" % (date_str, time_str)


fset = fpgaiphelpers.makeinstance(ctrlif, eids, ":fullset:", verbose=debug)
tonegen = fpgaiphelpers.makeinstance(ctrlif, eids, ":tonegen:", verbose=debug)
frontend = fpgaiphelpers.makeinstance(ctrlif, eids, interfaces.channelizer, verbose=debug)
backend = fpgaiphelpers.makeinstance(ctrlif, eids, ":backend:", verbose=debug)
assert fset and tonegen and frontend and backend
fset.debug = tonegen.debug = frontend.debug = backend.debug = (regaccess, debug, verbose)

# route sync signal (the settings below are the default)
fset.write_sync5sel(5) # tonegen's sync-Q signal (syncsrc bit #5) to syncbus #5
fset.write_sync5mode(0) # pass through

# backend output selection: disable normal output, enable ATAN1 on monitor output
backend.write_pktrst(1)
backend.write_pktsrc("None", "pktdata")
backend.write_pktrst(0)

backend.write_monrst(1)
backend.write_pktsrc("ATAN1", "monitor")
#backend.write_monmerge(32)
backend.write_monmerge(4)
backend.write_mondec(1)
backend.write_monrst(0)

# set up module signal generation: sine wave on tone generator's 2nd ('Q') output
samplerate = tonegen.samplerate = frontend.samplerate
iqorder = tonegen.get_iqorder()
print("samplerate = %8.3f MHz   iqorder = %d   %s" % (samplerate/1e6, iqorder, tonegen))

badchan = [1, 2, 3, 4, 5, 6, 7, 8]
badchan = [3, 4, 5, 6, 7, 8]

for f in range(len(modflist)):

    #set frequency
    modfreq = float(modflist[f])
    tonegen.set_q(modfreq, modampl * modampl_fact[f], phase=0., mode="ddstaylor-q", duty=0.50)
    print("set_q params", modfreq, modampl * modampl_fact[f])
    #tonegen.dumpfields()
    if modfreq>1e3:
        toread = int(npack_min[f])
        #toread = np.int(npack_min_hf[f])
    else:
        toread = int(npack_min[f])

    # collect data
    time.sleep(3)
    dataif.reset()
    (nbytes, buffer) = dataif.readall(toread)
    if f == 0: print(nbytes, buffer[0:31])

    (headeroffsets, payloadoffsets, payloadlengths, seqnos) = packetparser.findlongestcontinuous(buffer, incr=1024)
    (consumed, alldata) = packetparser.parsemany(buffer, headeroffsets[0], payloadoffsets, verbose = (f == 0))
    print("longest continuous:", toread, "->", nbytes, "->", consumed, "=", len(payloadoffsets), "packets =", sum(payloadlengths), "byte total payload")
    print(alldata.data.shape, "x", alldata.data.dtype)

    #get sync signal and unwrap + convert atan
    sync = alldata.data['syncbus'].astype(int) & 32
    atan1 = np.unwrap(alldata.data['scal2']/2**32*2*np.pi, axis=0)

    #now fit each channel to a cosine, get the phase of the sync signal
    #fr = 1e6/512/32*1e3
    t1 = 1/fr
    times = np.arange(0, len(atan1[:,0]))*t1
    sync_ind = np.where(np.diff(sync[:,0].T)<-30)
    sync_t = times[sync_ind[0]]
    print('tsync = %f s'%sync_t[0])
    T0 = 1.0/modfreq

    if f==0:
        ampls = np.zeros((len(atan1[0,:]), len(modflist)))
        phi_fit = np.zeros((len(atan1[0,:]), len(modflist)))
        phi_sync = np.zeros((len(atan1[0,:]), len(modflist)))
        shift_tot = np.zeros((len(atan1[0,:]), len(modflist)))
        fmeas = np.zeros((len(atan1[0,:]), len(modflist)))
        ampls_dtb = np.zeros((len(atan1[0,:]), len(modflist)))
        phi_fit_dtb = np.zeros((len(atan1[0,:]), len(modflist)))

    #fit each channel
    for ch in range(len(atan1[0,:])):
        if ch in badchan:
            print('ch %d bad'%ch)
            continue
        #(out_tmp, t, fit_t, fit_ch) = fit_cos(atan1.T[0], times, freq)
        if modfreq<100:
            (out_tmp, t, fit_t, fit_ch) = fit_cos_fl(atan1.T[ch], times, modfreq)
            mfreq=out_tmp[0][3]
        else:
            (out_tmp, t, fit_t, fit_ch) = fit_cos(atan1.T[ch], times, modfreq)
            mfreq=modfreq
        modfreq_true = tonegen.n2freq(tonegen.freq2n(modfreq))
        (out_tmp_2, t_2, fit_t_2, fit_ch_2, idx) = fit_cos_dtb(atan1.T[ch], times, sync_ind, modfreq_true)

        if ch == plotchan and f in pltlist:
            pl.figure();
            pl.plot(times, atan1[:,ch], 'o')
            pl.plot(fit_t, fit_ch)
            pl.plot(fit_t_2, fit_ch_2)
            pl.plot(times[sync_ind[0]], atan1[sync_ind[0],ch], 'ks')
            pl.plot(times, sync[:, 0]/32.0)
            pl.title('ch %d, frq %f'%(ch, mfreq))
        phase_sync1 = 2*np.pi*mfreq*times[sync_ind[0]]
        sync2 = phase_sync1-np.arange(len(sync_ind[0]))*np.pi*2
        phase_sync_avg = np.mean(sync2)

        #print(phase_sync_avg)
        #if ampl is negative, flip it and shift by pi
        if out_tmp[0][0]<0:
            print('flip')
            ampl = out_tmp[0][0]*-1
            phi_cos = out_tmp[0][1] - np.pi
        else:
            ampl = out_tmp[0][0]
            phi_cos = out_tmp[0][1]
        ampl_dtb = out_tmp_2[0][0]
        phi_cos_dtb = out_tmp_2[0][1]

        shift_total = phase_sync_avg + phi_cos
        if shift_total > 2*np.pi:
            shift_total = shift_total - 2*np.pi

        #print('phishift total: %f'%shift_total)
        #print('phase_sync: %f'%phase_sync_avg)
        print('chan: %d, ampl: %f, shift cosfit: %f, shift phase sync: %f, freq %.2f'%(ch, ampl, phi_cos, phase_sync_avg, mfreq))
        ampls[ch,f] = ampl / modampl_fact[f]
        phi_fit[ch,f] = phi_cos
        phi_sync[ch,f] = phase_sync_avg
        shift_tot[ch,f] = shift_total
        fmeas[ch, f] = mfreq
        ampls_dtb[ch,f] = ampl_dtb / modampl_fact[f]
        phi_fit_dtb[ch,f] = phi_cos_dtb

    #option to save raw data
    #np.savez('/data/20220511/cz/raw/cz_1p5V_20mV_105mK_%d_raw.npz'%modfreq, sync=sync, atan1=atan1, fmeas = mfreq)
    print(modfreq, ampl, shift_total)

dataif.close()
ctrlif.close()

Y = np.zeros_like(ampls)*1j
Y_dtb = np.zeros_like(ampls)*1j
pl.figure(99)
pl.clf()
pl.figure(102)
pl.clf()
pl.figure(103)
pl.clf()

#plot real and imag
for ch in range(len(atan1[0,:])):
    pl.figure(99)
    Y[ch,:] = ampls[ch,:]*np.exp(1j*(shift_tot[ch,:]))
    Y_dtb[ch,:] = ampls_dtb[ch,:]*np.exp(1j*(phi_fit_dtb[ch,:]))
    #Y[ch,:] = Y[ch,:]*np.exp(1j*np.angle(func_1pole(modflist, 1, 2.95e3)))
    pl.plot(np.imag(Y[ch,:]), np.real(Y[ch,:]), 'o-')
    pl.figure(102)
    pl.semilogx(modflist,np.imag(Y[ch,:]), 'o-')
    pl.figure(103)
    pl.semilogx(modflist,np.real(Y[ch,:]), 'o-')

#plot amplitude vs frequency
pl.figure()
pl.semilogx(modflist,ampls[plotchan,:], 'o-')

pl.figure(110)
pl.clf()
pl.subplot(2, 1, 1)
pl.loglog(modflist, np.abs(Y.T)[:, 0], '.-')
#pl.loglog(modflist, np.abs(Y_dtb.T)[:, 0], '.-')
pl.subplot(2, 1, 2)
pl.semilogx(modflist, np.angle(Y.T)[:, 0]*180/np.pi, '.-')
#pl.semilogx(modflist, np.angle(Y_dtb.T)[:, 0]*180/np.pi, '.-')


#np.savez('/data/20220223/cz/%s_czb2_%sV_%dmVpp_70mK.npz'%(date_time_str, bias, sine_amp_out), shift_tot=shift_tot, phi_fit=phi_fit, ampls=ampls, modflist=modflist, fr=fr, npack_min=npack_min, fmeas=fmeas)

#
# Save cz data
#
fname = '%s/cz_%s_100mK_1.8Vb.npz' % (cfg.cz_dir, cfg.date_time_str)
print('filename: %s' % fname)
np.savez(fname, shift_tot=shift_tot, phi_fit=phi_fit, ampls=ampls, modflist=modflist, fr=fr, npack_min=npack_min, fmeas=fmeas, modampl=modampl, modampl_fact=modampl_fact)
