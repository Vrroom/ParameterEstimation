from Util import * 
from Model import *
from Simulate import *
from Losses import *
import matplotlib.pyplot as plt

if __name__ == "__main__" : 
    params = {
        'tl'    : 10, 
        'te'    : 31,
        'k0'    : 1/7, 
        'kt'    : 0.075,
        'mu'    : 1/7,
        'sigma' : 1/38,
        'gamma1': 1/19,
        'gamma2': 1/14,
        'gamma3': 1/14,
        'N'     : 1.1e8,
        'beta'  : 0.3,
        'beta1' : 0.825
    }
    totalInfected = 14 * 20; pIgivenA = 0.33
    A0 = totalInfected * (1.0 - pIgivenA) ;I0 = totalInfected * pIgivenA ;Xs0 = 0;Xa0 = 0;Xi0 = 0;P0 = 14; R0 = 0

    # Everyone else, S0, is susceptible to infection initially.
    S0 = params['N'] - A0 - I0 - Xs0 - Xa0 - Xi0 - P0 - R0
    x0 = [S0, A0, I0, Xs0, Xa0, Xi0, P0, R0]

    sixer = Sixer(x0, params)
    T = np.linspace(0, 30, 300)
    pred = simulator(sixer, T)

    i = pred[:, 2]
    plt.plot(T, i)
    plt.show()
