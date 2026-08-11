#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sherpa.data import Data1D, Data1DInt
from sherpa.stats import LeastSq, Cash, Chi2, CStat
from sherpa.fit import Fit
import sys
from pathlib import Path

try:
    from tes_models.SherpaFitModels import MultiGauss, MultiGaussGauss
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from tes_models.SherpaFitModels import MultiGauss, MultiGaussGauss
from sherpa.plot import ModelPlot, FitPlot, DataPlot
import time
from scipy.signal import find_peaks

plt.rcParams['font.size'] = 14


def loadDf(filename):
    df = pd.read_csv(filename)
    print('DataFrame is loaded')
    return df


def getSpectrum(df, binWidth=0.5, eRange=None, queryStr='', plot_spectrum=False):
    if len(queryStr):
        df = df.query(queryStr)
    data = df['5lagy']
    data = data[data >= 0]
    if eRange is None:
        loLim = min(data)
        hiLim = max(data)
    else:
        loLim = eRange[0]
        hiLim = eRange[1]
    nBins = int(np.abs(hiLim - loLim) / binWidth)

    counts, bins = np.histogram(data, bins=nBins, range=eRange)
    bins_shifted = bins[0:-1] + 0.5 * (bins[1] - bins[0])
    if plot_spectrum:
        fig, ax = plt.subplots()
        ax.step(bins_shifted, counts, lw=0.5, alpha=1)
        ax.set(xlabel='5lagy', ylabel=f'counts/{binWidth} bin')
        plt.show()

    return counts, bins_shifted


def fitSpectrum(state='E', binWidth=0.05):
    centers_df = pd.read_csv('/home/pcuser/Runs/Cooldown_A12/peak_analysis_results_004.csv')
    guess_ns = centers_df[f'ns_{state}'].to_numpy().astype(int)
    guess_centers = centers_df[f'centers_{state}'].to_numpy()
    guess_ns = guess_ns[guess_ns >= 0]
    guess_centers = guess_centers[guess_centers >= 0]
    nMax = guess_ns[-1]
    nMin = guess_ns[0]
    nPeaks = nMax - nMin + 1
    fwhmToSigma = 1. / 2.35482
    sigmaToFwhm = 1. / fwhmToSigma
    mg = MultiGauss('MG', nPeaks)
    mg = MultiGaussGauss('MG', nPeaks)
    mg.bumpDeltaE = 1.2
    mg.bumpSigma = fwhmToSigma * 35
    mg.bumpFraction = 1e-3#5e-3

    Eph = 1239 / 515

    counts, bins = getSpectrum(df, binWidth=binWidth, queryStr=f'state_label=="{state}"')
    FWHM0 = 0.65
    FWHMdegrade = 0.5E-4
    ns = np.arange(nMin, nMax + 1)
    ampls = np.zeros_like(ns);
    mus = np.zeros_like(ns)
    for i, n in enumerate(ns):
        mu = guess_centers[i]
        fwhm = FWHM0  #+ FWHMdegrade*n

        ampl = counts[np.argmin(np.abs(bins - mu))] * (fwhmToSigma * fwhm * np.sqrt(2 * np.pi))
        ampls[i] = ampl;
        mus[i] = mu
        mg.setGaussComponent(i, ampl, mu, fwhmToSigma * fwhm)

    model = mg  # + bkg
    fig, ax = plt.subplots()
    plt.step(bins, counts, lw=0.5)

    import copy
    startingModel = copy.deepcopy(model)

    data = Data1DInt('Laser', bins - binWidth / 2, bins + binWidth / 2, counts, staterror=np.sqrt(counts))
    statistics = Cash()  # this will give an error estimate, with or without staterror

    dplot = DataPlot()
    dplot.prepare(data)
    dplot.plot()
    mplot = ModelPlot()
    mplot.prepare(data, model)
    mplot.plot()
    fplot = FitPlot()
    fplot.prepare(dplot, mplot)
    fplot.plot()

    opt = LevMar()
    fitter = Fit(data, model, stat=statistics, method=opt)
    print('Starting fit...')
    t0 = time.time()
    fitResult = fitter.fit()
    tElapsed = time.time() - t0
    if fitResult.succeeded:
        print('Fit succeeded. Elapsed time:', tElapsed, 's')
    else:
        print('Fit did not succeed. Elapsed time:', tElapsed, 's')
    print(fitResult.format())
    tableDict, nonTableDict = model.extractFitResults(
        fitResult)  # somewhat non-elegant way to get the fit results out...
    tableDict['n'] = ns
    resultsDf = pd.DataFrame(tableDict)
    resultsDf['Es'] = resultsDf.n * Eph
    resultsDf['fwhm'] = sigmaToFwhm * resultsDf.sigma
    resultsDf['fwhmStd'] = sigmaToFwhm * resultsDf.sigmaStd
    resultsDf['counts'] = resultsDf.ampl / binWidth
    # errors = fitter.est_errors()   # not entirely needed (may take a long time?!)
    # parValErrors = dict(zip(errors.parnames, zip(errors.parvals,errors.parmaxes)))
    # fwhm, fwhmError = parValErrors['Gaussian.fwhm']
    fig, axes = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(12, 10))
    yFit = data.eval_model(model)
    residuals = counts - yFit
    normedResiduals = residuals / (0.5 + np.sqrt(0.25 + counts))
    axes[0].plot(bins, yFit, '-r',
                 label=f'fit ({statistics.name}-stat.)', lw=2.0)
    axes[0].step(bins, counts, 'k', lw=0.5, )
    axes[0].set(ylabel=f'Counts/{binWidth} unit bin')
    axes[0].legend(loc='best')
    axes[1].bar(bins, normedResiduals, width=binWidth, color='k')
    axes[1].set(xlabel='5lagy (arb. units)', ylabel=r'Residuals/$\sqrt{\mathrm{Counts}}$', ylim=(-2.5, 2.5))
    fig.suptitle(f'Multi-Gaussian fit for state {state}')
    fig.tight_layout()
    return resultsDf


if __name__ == '__main__':
    filename = '/home/pcuser/Runs/Cooldown_A12/20250212_0004_pulsetable.csv'
    # filename = '/home/pcuser/Runs/Cooldown_A12/20250310_0008_pulsetable.csv'
    df = loadDf(filename)
    state = 'K';
    binWidth = 0.05
    # counts,bins= getSpectrum(df,binWidth = binWidth, queryStr=f'state_label== "{state}"', plot_spectrum = False)
    df_results = fitSpectrum(state=state, binWidth=binWidth)
    print(df_results.head())
    # df_results.to_csv(f'/home/pcuser/Runs/Cooldown_A12/fit_results_state{state}.csv', index=False)
    plt.show()
