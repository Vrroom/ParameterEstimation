from Model import *
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
        'gamma1': 1/21,
        'gamma2': 1/21,
        'gamma3': 1/17,
        'N'     : 1.1e8,
        'beta'  : 0.2,
        'beta1' : 0.1,
        'beta2' : 0.1
        'f'     : 0.1
    }
    return Spaxire(params)

if __name__ == "__main__" : 
    startDate = '29 Feb'
    N   = 1e8
    A0, I0 = 1, 1
    init = np.array([N - A0 - I0, A0, I0, 0, 0, 0, 0, 0])

    data = pandas.read_csv('./Data/maha_data.csv')
    firstDeathIdx = data[data['Total Deaths'] > 0].index[0]
    deaths = data['New Deaths'][firstDeathIdx:].to_numpy()
    days = deaths.size
    model = getModel()
    R = 1
    P0 = np.diag([100, 100, 100, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
    H = np.array([[0, 1, 1, 1, 1, 1, 1, 0]]) * 0.02
    active = data['Total Cases'] - data['Total Recoveries'] - data['Total Deaths']
    active = active.to_numpy()
    xs = extendedKalmanFilter(model.timeUpdate, init, P0, H, R, deaths, days)
    predictedNewDeaths = xs @ H.T
    print (xs)
    plt.scatter(range(17, 17 + days), deaths, c='violet', label='new deaths (mar 17 onwards)')
    plt.plot(range(17, 17 + days), predictedNewDeaths, c='red', label='predicted new deaths (mar 17 onwards)')
    plt.plot(range(days), xs[:,2] + xs[:,5], c='blue', label='predicted infected')
    plt.plot(range(days), xs[:,1] + xs[:,4], c='yellow', label='predicted asymptomatic')
    plt.plot(range(14, 14 + active.size), active, c='green', label='active cases')

    plt.legend()
    plt.show()
