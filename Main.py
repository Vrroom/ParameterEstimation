from Model import *
import matplotlib.ticker as ticker
from Search import *
from Losses import *
from itertools import product
from EKF import *
import numpy as np
import pandas

def getModel () : 
    params = {
        'tl'    : 21, 
        'te'    : 42,
        'k0'    : 1/7, 
        'kt'    : 0.075,
        'mu'    : 1/7,
        'sigma' : 1/38,
        'gamma1': 1/19,
        'gamma2': 1/14,
        'gamma3': 1/14,
        'N'     : 1.1e8,
        'beta'  : 0.31,
        'beta1' : 0.71 
    }
    return Sixer(params)

def getData (xs) : 
    o = xs[:, -2] + xs[:, 2] + xs[:, 5]
    return xs[:, -2], o, xs[:, 1] + xs[:, 4] + o

if __name__ == "__main__" : 
    def pltColumn (idx) : 
        dstd = np.sqrt(np.array([np.diag(P)[idx] for P in Ps_]))
        d   = np.array([x[idx] for x in xs])
        d_  = np.array([x[idx] for x in xs_])
        x = np.arange(days)
        plt.plot(x, d , c='red' , label=f'True {names[idx]}')
        plt.plot(x, d_, c='blue', label=f'Estimate {names[idx]}')
        plt.fill_between(x, np.maximum(d_ - dstd, 0), d_ + dstd, alpha=0.5, facecolor='grey')
        plt.legend()
        plt.show()

    T = 200; T2 = 25

    samplesPerDay = 1
    names = ['S', 'E', 'A', 'I', 'Xs', 'Xe', 'Xa', 'Xi', 'P', 'R']
    N   = 1e8
    A0, I0 = 1, 1
    init = np.array([N - A0 - I0, A0, I0, 0, 0, 0, 0, 0])

    A0_, I0_ = 5, 2
    init_ = np.array([N - A0_ - I0_, A0_, I0_, 0, 0, 0, 0, 0])

    model = getModel()

    xs = simulator(model, init, np.arange(T))[:-17]

    deaths = (xs[:,2] + xs[:,-2]) * 0.02
    days = deaths.size

    R = 5
    P0 = np.diag([1e3, 1e3, 1e3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
    H = np.array([[0, 0, 1, 0, 0, 0, 1, 0]]) * 0.02
    xs_, Ps_ = extendedKalmanFilter(model.timeUpdate, init_, P0, H, R, deaths, days)
    ps = [np.sqrt(np.diag(p)) for p in Ps_]

    a1 = np.array([p[-2] for p in ps])
    a2 = np.array([p[-2] + p[2] + p[5] for p in ps])
    a3 = np.array([p[-2] + p[1] + p[4] + p[2] + p[5] for p in ps])

    t_e = 24 + 21
    pLock, totalLock, total2Lock = getData(xs)
    
    fig, ax = plt.subplots(facecolor='w', figsize=(20, 7))
#    fig, (ax, ax2) = plt.subplots(1, 2, facecolor='w', figsize=(20, 7))
    t = np.linspace(0, T, samplesPerDay * T + 1)
    ax.fill_between(t[:len(pLock)], pLock - a1, pLock + a1, alpha=0.5, facecolor='grey')
    ax.plot(t[:len(pLock)], pLock, alpha=0.5, lw=2, label='predicted P')
    ax.plot(t[:len(totalLock)], totalLock, lw=2, label='predicted P + Xi + i')
    ax.fill_between(t[:len(totalLock)], totalLock - a2, totalLock + a2, alpha=0.5, facecolor='steelblue')
    ax.plot(t[:len(total2Lock)], total2Lock, alpha=0.5, lw=2, label='predicted P + Xi + i + A + Xa')
    ax.fill_between(t[:len(total2Lock)], total2Lock - a3, total2Lock + a3, alpha=0.5, facecolor='palegreen')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number of people')
    ax.set_yscale('log')
    ax.set_ylim(1, 1e8)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(15))
    ax.set_xticklabels(['', 'March 14', 'March 29', 'April 13', 'April 28', 'May 13', 'May 28', 'June 12', 'June 27',\
                       'July 12', 'July 27', 'August 11', 'August 26', 'Sept 10', 'Sept 25', 'Oct 10', 'Oct 25',\
                       'Nov 9', 'Nov 24', 'Dec 9', 'Dec 24', 'Jan 8'], rotation = 'vertical')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)

    t = np.linspace(0, T2, samplesPerDay * T2 + 1)
    # ax2.plot(t, pLock[:len(t)], alpha=0.5, lw=2, label='predicted P')
    # ax2.plot(t, totalLock[:len(t)], lw=2, label='predicted P + Xi + i')
    # # ax2.plot(t, total2Lock[:len(t)], alpha=0.5, lw=2, label='predicted P + Xi + i + A + Xa')
    # ax2.scatter(list(range(len(actualI))), actualI, c = 'r', label='actual P')
    # ax2.set_xlabel('Time /days')
    # ax2.set_ylabel('Number of people')
    # ax2.xaxis.set_major_locator(ticker.MultipleLocator(3))
    # ax2.set_xticklabels(['', 'March 14', 'March 17', 'March 20', 'March 23', 'March 27', 'March 30', 'April 3', \
    #                     'April 6', 'April 9', 'April 12', 'April 15', 'April 18', 'April 21', 'April 24'], rotation = 'vertical')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)

    plt.show()
