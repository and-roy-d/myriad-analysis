import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
plt.ion()

'''
f0guesses = guessResonanceFrequencies(f,s21,bw,threshold_diamratio,threshold_bwratio,threshold_spacing)

This function takes as inputs a numpy vector of frequency values, f, a complex numpy vector of S21
values, z, and a user guess at the typical bandwidth of the resonances. It first attempts to remove
the cable delay from the data. Then it applies a digital filter for a smoothed derivative of S21,
using the user guess at bandwidth to define a safe cut-off of the smoothing low-pass.

It then applies two thresholds looking at the 3-point circles for half-bandwidth-spaced data points
to require that the circles be at least a certain fraction of the "off-resonance" transmission and
that the arc angles be consistent with resonances of at most a maximum bandwidth relative to the
expected bandwidth.

It finally looks within the "valid" regions for the points of maximum derivative of S21 and picks
those points as the guess indices, requiring a minimum spacing in bandwidths between resonances.
These points are returned both as a list of indices and a list of frequencies.

Note that this function should only be used to guess at resonance frequencies. A more complete fit,
taking into account things like launch effects, should be used for precise resonance frequency
determination.

Example usage:

import guessResonanceFrequencies as grf
tau,good,f0good = grf.guessResonanceFrequencies(f,s21,bw,threshold_diamratio=0.2,threshold_bwratio=2,threshold_spacing=3)
'''

def remove_cable_delay(f,s21,showplot=True):

    rawphase = np.unwrap(np.angle(s21))
    
    pf = np.polyfit(f,rawphase,deg=1)
    tau = -pf[0] / (2*np.pi)
    print("Cable delay = {:f} ns".format(tau*1e9))
    
    s21_corrected = s21 * np.exp(-1j*(pf[1]+pf[0]*f))

    if showplot:
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(f,rawphase,'-b',f,np.polyval(pf,f),'--r')
        plt.ylabel('Phase (rad)')
        plt.subplot(2,1,2)
        plt.plot(f,np.angle(s21_corrected),'-b')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Corrected phase (rad)')
    
    return s21_corrected,tau


def circle_fit(x,y,showplot=False):
    """ Algebraic circle fit by Taubin

    Parameters:
    ===========
    x,y        : x- and y-coordinates of points to fit

    Returns:
    ===========
    xc,yc      : x- and y-coordinates of best-fit circle center
    r          : radius of best-fit circle

    Fit algorithm based off:
        G. Taubin, "Estimation Of Planar Curves, Surfaces And Nonplanar
                    Space Curves Defined By Implicit Equations, With
                    Applications To Edge And Range Image Segmentation",
        IEEE Trans. PAMI, Vol. 13, pages 1115-1138, (1991)
    """

    # Force inputs into column vectors
    npts = len(x)
    x1 = np.mean(x)
    y1 = np.mean(y)
    xtemp = x.reshape((npts,1)) - x1
    ytemp = y.reshape((npts,1)) - y1

    z = xtemp**2 + ytemp**2
    zmean = np.mean(z)
    z1 = (z - zmean) / (2*np.sqrt(zmean))

    (U,S,V) = np.linalg.svd(np.hstack((z1,xtemp,ytemp)),0)

    A = V[2,:]
    A[0] = A[0] / (2*np.sqrt(zmean))
    A = np.append(A,-zmean*A[0])
    xc = -(A[1])/A[0]/2 + x1
    yc = -(A[2])/A[0]/2 + y1
    r = np.sqrt(A[1]*A[1] + A[2]*A[2] - 4*A[0]*A[3])/np.abs(A[0])/2

    sqerr = np.sum((np.sqrt((x-xc)**2 + (y-yc)**2) - r)**2)

    if showplot:
        plt.figure()
        plt.plot(x,y,'ob')
        thetatemp = np.linspace(0,2*np.pi,num=100)
        plt.plot(xc+r*np.cos(thetatemp),yc+r*np.sin(thetatemp),'--r')
        plt.axis('equal')

    return xc,yc,r,sqerr


def guess_resonance_frequencies(f,s21,bw_lo,bw_hi,threshold_diam=0.05,threshold_spacing=2.0,threshold_circlefiterr=0.1):

    s21,tau = remove_cable_delay(f,s21,showplot=False)

    df = f[1]-f[0]
    expbws = bw_lo * np.logspace(0,np.log2(bw_hi/bw_lo),1+2*int(np.ceil(np.log2(bw_hi/bw_lo))),base=2)
    nhbws = (np.round(expbws/df)).astype(int)

    d = np.zeros((len(f),len(expbws)))
    resind = []

    for m in range(len(expbws)):
        nhbw = nhbws[m]
        print("Number of data points for half-bandwidth = {:d}, pass {:d}/{:d}".format(nhbw,m+1,len(expbws)))

        s21t = sig.savgol_filter(np.real(s21),window_length=(2*(nhbw//4)+1),polyorder=3,deriv=0) + 1j*sig.savgol_filter(np.imag(s21),window_length=(2*(nhbw//4)+1),polyorder=3,deriv=0)
        ds21t = sig.savgol_filter(np.real(s21),window_length=(2*(nhbw//4)+1),polyorder=3,deriv=1) + 1j*sig.savgol_filter(np.imag(s21),window_length=(2*(nhbw//4)+1),polyorder=3,deriv=1)

        for n in range(2*nhbw,len(f)-2*nhbw):
            try:
                # Fit from -BW/2 to +BW/2 to a circle in the complex plane
                xc,yc,r,sqerr = circle_fit(np.real(s21[n-nhbw:n+1+nhbw]),np.imag(s21[n-nhbw:n+1+nhbw]),showplot=False)
                if (np.sqrt(sqerr/(2*nhbw - 1))/r > threshold_circlefiterr):
                    # print("Circle fit poor at index {:d}".format(n))
                    pass
                elif ((2*r/(r+np.sqrt(xc**2 + yc**2))) < threshold_diam):
                    # print("Circle too small at index {:d}".format(n))
                    pass
                else:
                    tht = np.unwrap(np.angle(s21t[n-2*nhbw:n+1+2*nhbw]-(xc+1j*yc)))
                    tht_lo = tht[3*nhbw]
                    tht_hi = tht[nhbw]
                    if (tht_hi-tht_lo) > (0.8*4*np.arctan(1)):
                        dtht_mid = sig.savgol_filter(tht,window_length=(2*(nhbw//2)+1),polyorder=3,deriv=1)[2*nhbw]
                        d[n,m] = -dtht_mid*2*nhbw
            except RuntimeError:
                # print("Circle fit failed at index {:d}".format(n))
                pass
                
        resind.append(sig.find_peaks(d[:,m],distance=2*nhbw*threshold_spacing)[0])

    # Remove duplicates from different BW passes
    resind_nodup = np.array(resind[0])      # Include all resonators from first pass
    for m in range(1,len(expbws)):
        for n in range(len(resind[m])):
            if np.min(np.abs(resind_nodup-resind[m][n])) > threshold_spacing*2*nhbws[m]:
                resind_nodup = np.append(resind_nodup,resind[m][n])

    resind_nodup = np.sort(resind_nodup)
    f0guesses = f[resind_nodup]
    print("Found {:d} potential resonances.".format(len(resind_nodup)))

    plt.figure(444)
    plt.plot(f,np.abs(s21),'-b')
    plt.plot(f0guesses,np.abs(s21[resind_nodup]),'ob')

    plt.figure(555)
    good = np.zeros(0,dtype=int)
    bad = np.zeros(0,dtype=int)
    for k in range(len(resind_nodup)):
        plt.figure(555)
        plt.clf()
        tempind = np.arange(resind_nodup[k]-5*nhbw,1+resind_nodup[k]+5*nhbw,1,dtype='int')
        plt.subplot(1,2,1)
        plt.plot(np.real(s21[tempind]),np.imag(s21[tempind]),'ob')
        plt.axis("equal")
        plt.subplot(1,2,2)
        plt.plot(tempind,np.abs(s21[tempind]),'-b')
        plt.draw()
        plt.figure(444)
        plt.plot(f[resind_nodup[k]],np.abs(s21[resind_nodup[k]]),'o',color="tab:orange")
        plt.draw()
        s = input('[Enter] to accept point, text+[Enter] to reject point: ')
        if s:
            bad = np.append(bad,resind_nodup[k])
            plt.plot(f[resind_nodup[k]],np.abs(s21[resind_nodup[k]]),'or')
        else:
            good = np.append(good,resind_nodup[k])
            plt.plot(f[resind_nodup[k]],np.abs(s21[resind_nodup[k]]),'og')
        plt.draw()
    plt.figure(555)
    plt.close()
    plt.figure(444)

    f0good = f[good]
    return tau,good,f0good
