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
    T = 70
    N   = 1e8
    A0, I0 = 20, 20
    init = np.array([N - A0 - I0, A0, I0, 0, 0, 0, 0, 0])

    A0_, I0_ = 25, 25
    init_ = np.array([N - A0_ - I0_, A0_, I0_, 0, 0, 0, 0, 0])
    print(init, init_)

    model = getModel()

    data = pandas.read_csv('./Data/maha_data7apr.csv')
    totalDeaths = data['Total Deaths'].to_numpy()
    deaths = data['New Deaths'].to_numpy()
    P = (data['Total Cases'] - data['Total Recoveries'] - data['Total Deaths']).to_numpy()
    deaths = deaths[totalDeaths > 0] 
    days = deaths.size
    zs = np.stack([deaths, P[:]])
    import pdb 
    pdb.set_trace()

    xs = simulator(model, init, np.arange(T))[:days]

    R = 5
    P0 = np.diag([1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2])
    H = np.array([[0, 0, .02, 0, 0, 0, 0.02, 0], [0, 0, 0, 0, 0, 0, 1, 0]])
    xs_, Ps_ = extendedKalmanFilter(model.timeUpdate, init_, P0, H, R, deaths, days)

    istd = np.sqrt(np.array([np.diag(P)[-2] for P in Ps_]))
    i   = np.array([x[-2] for x in xs])
    i_  = np.array([x[-2] for x in xs_])
    
    plt.plot(np.arange(days), i, c='red', label='True P')
    plt.plot(np.arange(days), i_, c='blue', label='Estimate P')
    plt.fill_between(np.arange(days), i_ - istd, i_ + istd, facecolor="grey", alpha=0.5)
    plt.legend()
    plt.show()


