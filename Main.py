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
        'gamma1': 1/19,
        'gamma2': 1/14,
        'gamma3': 1/14,
        'N'     : 1.1e8,
        'beta'  : 0.31,
        'beta1' : 0.71 
    }
    return Sixer(params)

if __name__ == "__main__" : 
    startDate = '29 Feb'
    N   = 1e8
    A0, I0 = 1, 1
    init = [N - A0 - I0, A0, I0, 0, 0, 0, 0, 0]

    data = pandas.read_csv('./Data/maha_data.csv')
    firstDeathIdx = data[data['Total Deaths'] > 0].index[0]
    deaths = data['New Deaths'][firstDeathIdx:].to_numpy()
    days = deaths.size
    model = getModel()
    R = 1
    P0 = np.diag([100, 100, 100, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
    H = np.array([[0, 1, 1, 1, 1, 1, 1, 0]]) * 0.02
    xs = extendedKalmanFilter(model.timeUpdate, init, P0, H, R, deaths, days)
    predictedNewDeaths = xs @ H.T
    plt.scatter(range(days), deaths, c='violet', label='new deaths (mar 17 onwards)')
    plt.plot(range(days), predictedNewDeaths, c='red', label='predicted new deaths')
    plt.plot(range(days), x[:,2], c='blue', label='infected')

    plt.legend()
    plt.show()
