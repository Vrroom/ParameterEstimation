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
        x = np.arange(T + 10)

        dstd = np.sqrt(np.array([np.diag(P)[idx] for P in Ps_]))
        d_  = np.array([x[idx] for x in xs_])

        c = model.colors[idx]
        name = model.names[idx]

        plt.plot(x, d_, c=c, label=f'Estimate {name}')
        plt.fill_between(x, np.maximum(d_ - dstd, 0), d_ + dstd, alpha=0.5, facecolor='grey')
        plt.legend()

    def H (date) : 
        h1    = [0,0,0,.02,0,0,0,0.02,.02,0]
        h2    = [0,0,0,0.0,0,0,0,0,1.0,0]
        if date < firstCases : 
            return np.array([h1])
        elif firstCases <= date <= endDate - 17 :
            return np.array([h1, h2])
        elif endDate - 17 < date <= endDate : 
            return np.array([h2])
        else :
            return np.array([])

    def z (date) : 
        if date < firstCases : 
            m = deaths[date - startDate]
            return np.array([m])
        elif firstCases <= date <= endDate - 17 :
            m = deaths[date - startDate]
            p = P[date - firstCases]
            return np.array([m, p])
        elif endDate - 17 < date <= endDate : 
            p = P[date - firstCases]
            return np.array([p])
        else :
            return np.array([])

    def R (date) : 
        if date < firstCases : 
            return np.array([1])
        elif firstCases <= date <= endDate - 17 :
            return np.eye(2)
        elif endDate - 17 < date <= endDate : 
            return np.array([1])
        else :
            return np.array([])


    T = endDate - startDate
    N   = 1.1e8

    data = pandas.read_csv('./Data/maha_data7apr.csv')
    totalDeaths = data['Total Deaths'].to_numpy()
    
    deaths = data['New Deaths'][data['Total Deaths'] > 0].to_numpy()

    P = (data['Total Cases'] - data['Total Recoveries'] - data['Total Deaths']).to_numpy()

    E0, A0, I0 = 25, 25, 25
    init = np.array([N - E0 - A0 - I0, E0, A0, I0, 0, 0, 0, 0, 0, 0])

    model = getModel()

    P0 = np.diag([1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4])
    Q = np.diag([1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1, 1])
    Q = np.zeros(P0.shape)


    xs_, Ps_ = extendedKalmanFilter(model.timeUpdate, init, P0, Q, H, R, z, startDate, endDate + 10)

    # plt.scatter(, P, c='red', label='P (Actual Data)')
    pltColumn(-2)
    # pltColumn(2)
    # pltColumn(1)
    plt.show()

