from Model import *
from Search import *
from Losses import *
from itertools import product
from EKF import *
import numpy as np
import pandas

def getModel () : 
    params = {
        'tl'    : 40, 
        'te'    : 42,
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
        
        x = np.arange(days)
        plt.plot(x, d , c='red' , label=f'True {names[idx]}')
        plt.plot(x, d_, c='blue', label=f'Estimate {names[idx]}')
        plt.fill_between(x, d_ - dstd, d_ + dstd, alpha=0.5, facecolor='grey')
        plt.legend()
        plt.show()

    names = ['S', 'E', 'A', 'I', 'Xs', 'Xe', 'Xa', 'Xi', 'P', 'R']
    T = 36
    N   = 1.1e8
    E0, A0, I0, P0 = 28, 238, 0, 14
    init = [N - E0 - A0 - I0 - P0, E0, A0, I0, 0, 0, 0, 0, P0, 0]
    model = getModel()

    t = np.linspace(0, T, 10 * T)
    xs = simulator(model, init, t)
    plt.plot(t, xs[:, -2])
    plt.show()

