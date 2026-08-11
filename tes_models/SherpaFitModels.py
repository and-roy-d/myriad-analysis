#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sherpa import models

def gaussian(x, mu, A, sigma):
    return A * np.exp(-0.5 * np.square(x-mu) / sigma ** 2) / (np.sqrt(2 * np.pi * sigma ** 2))


class MultiGauss(models.RegriddableModel1D):
    def __init__(self, name, nComponents):
        params = []
        for i in range(nComponents):
            amplitude = models.Parameter(name, 'A%d' % i, 1, min=0, hard_min=0)
            mu = models.Parameter(name, 'mu%d' % i, 1)
            sigma = models.Parameter(name, 'sigma%d' % i, 1, min=0, hard_min=0)
            setattr(self, "A%d" % i, amplitude)
            setattr(self, "mu%d" % i, mu)
            setattr(self, "sigma%d" % i, sigma)
            params.append(amplitude)
            params.append(mu)
            params.append(sigma)
        self.nComponents = nComponents

        models.RegriddableModel1D.__init__(self, name, params)

    def extractFitResults(self, fitRes):
        # Get covariance errors from fitResult
        # see sherpa.fit.FitResults.format; not sure why they don't provide it on the API
        covarErr = np.sqrt(fitRes.covar.diagonal())
        parDict = dict(zip(fitRes.parnames, zip(fitRes.parvals, covarErr)))
        mus = np.zeros((self.nComponents,), dtype=float)
        muStds = np.zeros_like(mus)
        sigmas = np.zeros_like(mus)
        sigmaStds = np.zeros_like(mus)
        ampls = np.zeros_like(mus)
        amplStds = np.zeros_like(ampls)
        modelName = self.name
        for i in range(self.nComponents):
            mu = parDict['%s.mu%d' % (modelName, i)]
            mus[i], muStds[i] = mu[0], mu[1]
            sigma = parDict['%s.sigma%d' % (modelName, i)]
            sigmas[i], sigmaStds[i] = sigma
            ampl = parDict['%s.A%d' % (modelName, i)]
            ampls[i], amplStds[i] = ampl[0], ampl[1]
        tableDict = dict(mu=mus, muStd=muStds, ampl=ampls, amplStd=amplStds, sigma=sigmas, sigmaStd=sigmaStds)

        return tableDict

    def setGaussComponent(self, i, amplitude, mu, sigma):
        getattr(self, "A%d" % i).val = amplitude
        getattr(self, "mu%d" % i).val = mu
        getattr(self, "sigma%d" % i).val = sigma
        
    def getGaussComponent(self, i): # Unused, I think
        return getattr(self, "A%d" % i).val, getattr(self, "mu%d" % i).val, getattr(self, "sigma%d" % i).val
        
    def amplitude(self, i):  # this was not as useful as I thought
        return getattr(self, "A%d" % i)
    
    def mu(self, i):
        return getattr(self, "mu%d" % i)
    
    def sigma(self, i):
        return getattr(self, "sigma%d" % i)
    
    def calc(self, pars, x, *args, **kwargs):
        """Evaluate the model"""
        total = np.zeros_like(x)
        for i in range(self.nComponents):
            A = pars[3*i+0]
            mu = pars[3*i+1]
            sigma = pars[3*i+2]
            #print(A, mu, sigma)
            total += gaussian(x, mu, A, sigma)
        return total


class MultiGaussGauss(models.RegriddableModel1D):
    def __init__(self, name, nComponents):
        params = []
        for i in range(nComponents):
            amplitude = models.Parameter(name, 'A%d' % i, 1, min=0, hard_min=0)
            mu = models.Parameter(name, 'mu%d' % i, 1)
            sigma = models.Parameter(name, 'sigma%d' % i, 1, min=0, hard_min=0)
            setattr(self, "A%d" % i, amplitude)
            setattr(self, "mu%d" % i, mu)
            setattr(self, "sigma%d" % i, sigma)
            params.append(amplitude)
            params.append(mu)
            params.append(sigma)
        self.nComponents = nComponents

        self.bumpFraction = models.Parameter(name, "bumpFraction", 5E-3, min=0, hard_min=0)
        self.bumpDeltaE = models.Parameter(name, "bumpDeltaE", 1.2, min=0, hard_min=0)
        self.bumpSigma = models.Parameter(name, "bumpSigma", 0.17, min=0, hard_min=0)
        params.append(self.bumpFraction)
        params.append(self.bumpDeltaE)
        params.append(self.bumpSigma)

        models.RegriddableModel1D.__init__(self, name, params)

    def extractFitResults(self, fitRes):
        # Get covariance errors from fitResult
        # see sherpa.fit.FitResults.format; not sure why they don't provide it on the API
        covarErr = np.sqrt(fitRes.covar.diagonal())
        parDict = dict(zip(fitRes.parnames, zip(fitRes.parvals, covarErr)))
        mus = np.zeros((self.nComponents,), dtype=float)
        muStds = np.zeros_like(mus)
        sigmas = np.zeros_like(mus)
        sigmaStds = np.zeros_like(mus)
        ampls = np.zeros_like(mus)
        amplStds = np.zeros_like(ampls)
        modelName = self.name
        for i in range(self.nComponents):
            mu = parDict['%s.mu%d' % (modelName, i)]
            mus[i], muStds[i] = mu[0], mu[1]
            sigma = parDict['%s.sigma%d' % (modelName, i)]
            sigmas[i], sigmaStds[i] = sigma
            ampl = parDict['%s.A%d' % (modelName, i)]
            ampls[i], amplStds[i] = ampl[0], ampl[1]
        tableDict = dict(mu=mus, muStd=muStds, ampl=ampls, amplStd=amplStds, sigma=sigmas, sigmaStd=sigmaStds)
        nonTableDict = dict(bumpFraction=parDict['%s.bumpFraction' % modelName],
                            bumpDeltaE=parDict['%s.bumpDeltaE' % modelName],
                            bumpSigma=parDict['%s.bumpSigma' % modelName])
        return tableDict, nonTableDict

    def setGaussComponent(self, i, amplitude, mu, sigma):
        getattr(self, "A%d" % i).val = amplitude
        getattr(self, "mu%d" % i).val = mu
        getattr(self, "sigma%d" % i).val = sigma

    def calc(self, pars, x, *args, **kwargs):
        """Evaluate the model"""
        total = np.zeros_like(x)
        f0 = pars[-3]
        bumpDeltaE = pars[-2]
        bumpSigma = pars[-1]
        for i in range(self.nComponents):
            A = pars[3*i+0]
            mu = pars[3*i+1]
            sigma = pars[3*i+2]
            n = mu/3.06
            fBump = f0 * n
            #print(A, mu, sigma)
            total += gaussian(x, mu, A, sigma) + gaussian(x, mu-bumpDeltaE, fBump*A, np.sqrt(sigma**2+bumpSigma**2))
        return total


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time

    from sherpa.data import Data1D
    from sherpa.stats import LeastSq, Cash
    from sherpa.fit import Fit
    from sherpa.optmethods import LevMar
    from sherpa.plot import ModelPlot, FitPlot, DataPlot

    nPeaks = 40
    mg = MultiGauss('MG3', nPeaks)

    for i in range(nPeaks):
        A = 50
        mu = i*3.06
        sigma = 0.25+0.01*i
        mg.setGaussComponent(i, A, mu, sigma)

    x = np.arange(-3, 200, 0.02)
    y = mg(x)
    y += 2*np.sin(x/3.06*np.pi)**2
    #y += np.random.normal(4., 1, x.shape)*np.sin(x/3.06*np.pi)**2
    plt.plot(x, y, 'ko')
    d = Data1D('example', x, y)
    print(d)

    # Unclear what upside of plotting via Sherpa is
    dplot = DataPlot()
    dplot.prepare(d)
    dplot.plot()

    g = MultiGauss('MG3', nPeaks)
    for i in range(nPeaks):
        A, mu, sigma = mg.getGaussComponent(i)
        A = A*0.9
        mu = mu+0.25
        sigma = sigma*1.05
        g.setGaussComponent(i, A, mu, sigma)

    print('Guess:', g)

    mplot = ModelPlot()
    mplot.prepare(d, g)
    fplot = FitPlot()
    fplot.prepare(dplot, mplot)
    fplot.plot()


    stat = LeastSq()
    #stat = Cash()
    opt = LevMar()

    gfit = Fit(d, g, stat=stat, method=opt)
    print(gfit)
    t0 = time.time()
    gres = gfit.fit()
    tElapsed = time.time()-t0
    print('Elapsed time:', tElapsed)
    print(gres.succeeded)
    #print(gres.format())

    fplot = FitPlot()
    mplot.prepare(d, g)
    fplot.prepare(dplot, mplot)
    fplot.plot()
