from Model import *
from Search import *
from Losses import *
from itertools import product
from EKF import *
import numpy as np
import pandas

# Important events

def processTimeSeries (state) : 
    fname = state + '.csv'
    path = osp.join('./Data/time_series', fname)
    data = pandas.read_csv(path)
    
    firstCases = Date(data['Date'][0])
    firstDeath = Date(data[data['Total Dead'] > 0]['Date'][0])
    startDate = firstDeath - 17
    endDate = Date(data['Date'][-1])

    T = endDate - startDate

    totalDeaths = data['Total Dead'].to_numpy()
    
    deaths = data['New Deaths'][data['Total Deaths'] > 0].to_numpy()
    deaths = np.pad(deaths, ((0, T - deaths.size)))

    P = (data['Total Cases'] - data['Total Recovered'] - data['Total Dead']).to_numpy()
    P = np.pad(P, ((T - P.size, 0)))
    zs = np.stack([deaths, P[:]]).T

    return startDate, firstCases, firstDeath, endDate, zs


def getModel (state) : 
    startDate, firstCases, firstDeath, endDate, _ = processTimeSeries(state)

    lockdownBegin = Date('24 Mar') - startDate
    lockdownEnd = Date('14 Apr') - startDate

    contactHome = np.loadtxt('./Data/home.csv', delimiter=',')
    contactTotal = np.loadtxt('./Data/total.csv', delimiter=',')

    changeContactStart = math.inf
    changeContactEnd   = math.inf

    changeKt = math.inf
    deltaKt  = math.inf

    Nbar = readStatePop(state)

    params = {
        'tl'                : lockdownBegin, 
        'te'                : lockdownEnd,
        'k0'                : partial(bumpFn, ti=lockdownBegin, tf=lockdownEnd, x1=0, x2=1/7),
        'kt'                : partial(climbFn, ti=changeKt, tf=changeKt+deltaKt, xi=0.5, xf=1.0),
        'mu'                : partial(stepFn, t0=lockdownEnd, x1=0, x2=1/7),
        'sigma'             : 1/5,
        'gamma1'            : 1/21,
        'gamma2'            : 1/21,
        'gamma3'            : 1/19,
        'N'                 : 1.1e8,
        'beta'              : 0.16,
        'beta2'             : 0.1,
        'f'                 : 0.2,
        'lockdownLeakiness' : 0.9,
        'contactHome'       : partial(bumpFn, ti=changeContactStart, tf=changeContactEnd, x1=contactHome, x2=0.5*contactHome),
        'contactTotal'      : partial(bumpFn, ti=changeContactStart, tf=changeContactEnd, x1=contactTotal, x2=0.5*contactTotal),
        'bins'              : 3,
        'Nbar'              : Nbar,
        'adultBins'         : [1],
        'testingFraction1'  : partial(climbFn, ti=changeKt, tf=changeKt+deltaKt, xi=1/13, xf=0.8),
        'testingFraction2'  : partial(climbFn, ti=changeKt, tf=changeKt+deltaKt, xi=0, xf=0.5),
        'testingFraction3'  : partial(climbFn, ti=changeKt, tf=changeKt+deltaKt, xi=0, xf=0.5),
    }

    model = SpaxireAgeStratified(params)
    return model

def estimate (state) : 

    def pltColumn (idx) : 
        x = np.arange(T)

        dstd = np.sqrt(np.array([np.diag(P)[idx] for P in Ps_]))
        d_  = np.array([x[idx] for x in xs_])

        c = model.colors[idx]
        name = model.names[idx]

        plt.plot(x, d_, c=c, label=f'Estimate {name}')
        plt.fill_between(x, np.maximum(d_ - dstd, 0), d_ + dstd, alpha=0.5, facecolor='grey')
        plt.legend()

    def H (date) : 
        z = [0,0,0]
        h1    = [*z,*z,*z,*m,*z,*z,*z,*m,*m,*z]
        h2    = [*z,*z,*z,*z,*z,*z,*z,*z,1,1,1,*z]
        zeros = [*z,*z,*z,*z,*z,*z,*z,*z,*z,*z]
        if date < firstCases : 
            return np.array([h1, zeros])
        elif date >= firstCases and date < startDate + (endDate - firstDeath) :
            return np.array([h1, h2])
        elif date < dataEndDate : 
            return np.array([zeros, h2])
        else : 
            return np.array([zeros, zeros])

    m = (getAgeMortality(state) * 0.01).tolist()
    startDate, firstCases, firstDeath, endDate, zs = processTimeSeries(state)
    model = getModel(state)
    Nbar = readStatePop(state)

    x0 = np.array([*(Nbar.tolist()), 0, 56.0, 0, 0, 210.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0])

    R = np.diag([5, 5])
    P0 = np.eye(30) * 1e3

    xs_, Ps_ = extendedKalmanFilter(model.timeUpdate, x0, P0, H, R, zs, startDate, endDate)
    pass


if __name__ == "__main__" : 
    pass
