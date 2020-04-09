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
    A0, I0 = 1, 1
    init = np.array([N - A0 - I0, A0, I0, 0, 0, 0, 0, 0])

    A0_, I0_ = 50, 20
    init_ = np.array([N - A0_ - I0_, A0_, I0_, 0, 0, 0, 0, 0])
    print(init, init_)

    model = getModel()

    xs = simulator(model, init, np.arange(T))[:-17]

    deaths = (xs[:,2] + xs[:,-2]) * 0.02
    days = deaths.size

    R = 50
    P0 = np.diag([1e4, 1e4, 1e4, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
    H = np.array([[0, 0, 1, 0, 0, 0, 1, 0]]) * 0.02
    xs_, Ps_ = extendedKalmanFilter(model.timeUpdate, init_, P0, H, R, deaths, days)
    print(xs_[0])


    istd = np.sqrt(np.array([np.diag(P)[-2] for P in Ps_]))
    i   = np.array([x[-2] for x in xs])
    i_  = np.array([x[-2] for x in xs_])
    
    plt.plot(np.arange(days), i_ - istd, c='orange', label='Lower Bound on I')
    plt.plot(np.arange(days), i_ + istd, c='red', label='Upper Bound on I')
    plt.plot(np.arange(days), i, c='blue', label='True I')
    plt.plot(np.arange(days), i_, c='green', label='Estimate I')
    plt.legend()
    plt.show()


