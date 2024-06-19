from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d


@dataclass
class CLresults:
    bf: float = 0.0
    mean: float = 0.0
    sigma: float = 0.0
    low1: float = 0.0
    high1: float = 0.0
    low2: float = 0.0
    high2: float = 0.0
    low3: float = 0.0
    high3: float = 0.0


def ConfidenceLevels(par, post, oversamples=-1) -> CLresults:
    lim1 = 0.6827
    lim2 = 0.9545
    lim3 = 0.9973

    if oversamples > len(par):
        f = interp1d(par, post, kind="cubic")
        par = np.linspace(par.min(), par.max(), oversamples)
        post = f(par)

    nstepslevs = 10001
    levels = np.linspace(1, 0, nstepslevs)

    post /= post.max()
    pmean = np.sum(post * par) / post.sum()
    psigma = np.sqrt(np.sum(post * par**2) / post.sum() - pmean**2)
    pmax = par[np.where(post == post.max())[0][0]]
    prob = np.zeros(nstepslevs)
    indexlim = np.zeros((2, nstepslevs), dtype=int)
    for ilev in np.arange(nstepslevs):
        lev = levels[ilev]
        po = np.argwhere(np.diff(np.sign(post - lev)) != 0).reshape(-1)
        if np.size(po) == 2:
            prob[ilev] = np.sum(post[po[0] : po[1]]) / post.sum()
            indexlim[:, ilev] = po
        if np.size(po) == 1:
            prob[ilev] = np.sum(post[0 : po[0]]) / post.sum()
            indexlim[1, ilev] = po
        if np.size(po) == 0:
            prob[ilev] = 1
            indexlim[:, ilev] = [0, np.size(par)]
    ilev = np.where(prob > lim1)[0][0]
    mplim1, pplim1 = par[indexlim[0, ilev]], par[indexlim[1, ilev]]
    ilev = np.where(prob > lim2)[0][0]
    mplim2, pplim2 = par[indexlim[0, ilev]], par[indexlim[1, ilev]]
    ilev = np.where(prob > lim3)[0][0]
    mplim3, pplim3 = par[indexlim[0, ilev]], par[indexlim[1, ilev]]

    return CLresults(
        bf=pmax,
        mean=pmean,
        sigma=psigma,
        low1=mplim1,
        high1=pplim1,
        low2=mplim2,
        high2=pplim2,
        low3=mplim3,
        high3=pplim3,
    )
