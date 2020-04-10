from Model import *
from Search import *
from Losses import *
from itertools import product
from EKF import *
import numpy as np
import pandas

def getModel () : 
    params = {
        'tl'    : 10, 
        'te'    : 20,
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
        d   = np.array([x[idx] for x in xs])
        d_  = np.array([x[idx] for x in xs_])
        
        print(dstd)
        x = np.arange(days)
        plt.plot(x, d , c='red' , label=f'True {names[idx]}')
        plt.plot(x, d_, c='blue', label=f'Estimate {names[idx]}')
        plt.fill_between(x, d_ - dstd, d_ + dstd, alpha=0.5, facecolor='grey')
        plt.legend()
        plt.show()

    names = ['S', 'E', 'A', 'I', 'Xs', 'Xe', 'Xa', 'Xi', 'P', 'R']
    T = 40
    N   = 1.1e8
    E0, A0, I0 = 28, 238, 1
    init = np.array([N - E0 - A0 - I0, E0, A0, I0, 0, 0, 0, 0, 0, 0])

    E0_, A0_, I0_ = 50, 200, 1
    init_ = np.array([N - E0_ - A0_ - I0_, E0, A0_, I0_, 0, 0, 0, 0, 0, 0])

    model = getModel()

    xs = simulator(model, init, np.arange(0, T))[:-17]
    print(xs[:,0])
    print(xs[:,1])

    deaths = (xs[:,3] + xs[:,-2]) * 0.02
    days = deaths.size

    R = 5
    P0 = np.diag([150**2, 50**2, 50**2, 50**2, 1, 1, 1, 1, 1, 1])
    H = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 1, 0]]) * 0.02
    xs_, Ps_ = extendedKalmanFilter(model.timeUpdate, init_, P0, H, R, deaths, days)

    # pltColumn(2)
