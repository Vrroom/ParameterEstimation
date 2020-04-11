from Model import *
from Search import *
from Losses import *
from itertools import product
from EKF import *
import numpy as np
import pandas

# Important events
startDate = Date('29 Feb')
firstCases = Date('14 Mar')
firstDeath = Date('17 Mar')
endDate = Date('7 Apr')

def getModel () : 
    lockdownBegin = Date('24 Mar') - startDate
    lockdownEnd = Date('14 Apr') - startDate
    params = {
        'tl'    : lockdownBegin, 
        'te'    : lockdownEnd,
        'k0'    : 1/7, 
        'kt'    : 0.075,
        'mu'    : 1/7,
        'sigma' : 1/5,
        'gamma1': 1/21,
        'gamma2': 1/21,
        'gamma3': 1/17,
        'N'     : 1.1e8,
        'beta'  : 0.16,
        'beta1' : 1.8,
        'beta2' : 0.1,
        'f'     : 0.1
    }
    return Spaxire(params)

if __name__ == "__main__" : 

    def pltColumn (idx) : 
        dstd = np.sqrt(np.array([np.diag(P)[idx] for P in Ps_]))
        d_  = np.array([x[idx] for x in xs_])
        x = np.arange(T)
        plt.plot(x, d_, c='blue', label=f'Estimate {names[idx]}')
        plt.fill_between(x, np.maximum(d_ - dstd, 0), d_ + dstd, alpha=0.5, facecolor='grey')
        plt.legend()

    def H (date) : 
        h1    = [0,0,.02,0,0,0,.02,0]
        h2    = [0,0,0.0,0,0,0,1.0,0]
        zeros = [0,0,0.0,0,0,0,0.0,0]
        if date < firstCases : 
            return np.array([h1, zeros])
        elif date >= firstCases and date < startDate + (endDate - firstDeath) :
            return np.array([h1, h2])
        else : 
            return np.array([zeros, h2])

    names = ['S', 'A', 'I', 'Xs', 'Xa', 'Xi', 'P', 'R']
    T = endDate - startDate
    N   = 1.1e8

    data = pandas.read_csv('./Data/maha_data7apr.csv')
    totalDeaths = data['Total Deaths'].to_numpy()
    
    deaths = data['New Deaths'][data['Total Deaths'] > 0].to_numpy()
    deaths = np.pad(deaths, ((0, T - deaths.size)))

    P = (data['Total Cases'] - data['Total Recoveries'] - data['Total Deaths']).to_numpy()
    P = np.pad(P, ((T - P.size, 0)))

    zs = np.stack([deaths, P[:]]).T

    A0_, I0_ = 25, 25
    init_ = np.array([N - A0_ - I0_, A0_, I0_, 0, 0, 0, 0, 0])

    model = getModel()

    R = np.diag([1, 1])
    P0 = np.diag([1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2])

    xs_, Ps_ = extendedKalmanFilter(model.timeUpdate, init_, P0, H, R, zs, startDate, endDate)

    plt.scatter(np.arange(T), P, c='red', label='P (Actual Data)')
    pltColumn(-2)
    pltColumn(2)
    pltColumn(1)
    plt.show()

